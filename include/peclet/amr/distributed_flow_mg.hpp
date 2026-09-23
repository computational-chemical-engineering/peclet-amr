// core — distributed device openness multigrid for the AMR flow pressure (rung 3 of
// docs/amr_distributed_flow.md): the distributed counterpart of Multigrid (multigrid.hpp),
// i.e. the aperture-path pressure hierarchy AmrFlow::presMG_ runs — graded octree, cut-cell
// openness area-averaged down the ladder, Jacobi smoothing, per-level nullspace projection —
// with every level a ghost-slot face CSR + its own LeafHalo.
//
// Structure (all proven pieces composed):
//   * Level ladder: per-rank coarsenIf of the LOCAL octree (same as AmrMultigrid::build),
//     each level a COPY of the flow's DistributedOctree with the coarsened local() — the copy
//     carries the decomposition, so a rebalanced (weighted-ORB) flow octree keeps correct
//     owner lookups. Ranks reaching their root brick early PAD to the global max level count
//     (Allreduce MAX) by repeating the root brick: the padded levels' transfers are identity
//     (the covering c2p maps a leaf to itself) and their smoothing exactly mirrors what the
//     whole-domain COMM_SELF ladder does in its already-coarse regions (the
//     GradedDistributedMultigrid argument) — and the level counts must match anyway for the
//     per-level halo point-to-points to pair up.
//   * Per level: an AmrPoisson with the LeafHalo resolver seam (setResolver/setGhosts/
//     setFrameShift — probes that exit the block resolve to ghost slots; world-coordinate
//     evaluations in the GLOBAL frame), built to the miss-collect fixpoint, then host
//     assembleFv → device FvOp whose faceNbr may reference the ghost tail.
//   * Openness: level 0 α from the world-coord openFn (ghost rows sampled locally — the same
//     world points as the owner, bit-identical); level L>0 LOCAL α by the exact
//     AmrMultigrid::coarsenOpenness child-face averaging (children of a local coarse cell are
//     always local, visited in the same relative Z-order as COMM_SELF ⇒ bit-identical), and
//     GHOST α rows exchanged from the owner once per build (exact by construction — never
//     re-derived).
//   * V-cycle: jacobiFv with a ghost refresh before every sweep, local restrict/prolong
//     (parents never cross blocks), Allreduce'd volume-weighted mean removal (removeMean).
//   * Below the in-place ladder: a STAGE (mg_stage.hpp, docs/amr_mg_depth.md §5.6/§6.5) moves the
//     coarsest level onto a new decomposition of its own grid and continues the ladder there. The
//     V-cycle contains no MPI of its own for this — every message lives behind MgStage::apply, so
//     WO4b's sibling-merge and repartition stages need no change here.
//
// Bit-exactness: at np=1 every probe resolves locally (zero ghosts) and the whole cycle is
// the single-rank Multigrid arithmetic verbatim. Across ranks the smoother/transfers are
// order-independent ⇒ WORLD==SELF bit-exact with removeMean OFF; the mean removal (a global
// reduction) and PCG dots are np-invariant only to reduction order ⇒ tolerance, the suite's
// Krylov contract.
//
// Kokkos + MPI header (include in device TUs; the AmrFlow oracle stays single-rank).
#ifndef PECLET_AMR_DISTRIBUTED_FLOW_MG_HPP
#define PECLET_AMR_DISTRIBUTED_FLOW_MG_HPP

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/amr/common.hpp"
#include "peclet/amr/distributed_octree.hpp"
#include "peclet/amr/fv_op.hpp"
#include "peclet/amr/leaf_halo.hpp"
#include "peclet/amr/mg_stage.hpp"   // the telescoping primitive (docs/amr_mg_depth.md §5.6/§6.5)
#include "peclet/amr/multigrid.hpp"  // restrictField / prolongAdd (shared transfer kernels)
#include "peclet/amr/poisson.hpp"
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::amr {

/// removeMeanFv with the two mean sums folded across ranks (identical kernels; the Allreduce
/// sits between the reduce and the subtract). np=1 == removeMeanFv bit-for-bit.
inline void removeMeanFvDist(const FvOp& op, View<double> u, MPI_Comm comm) {
  auto invVol = op.invVol;
  auto fs = op.faceStart;
  auto fw = op.faceW;
  auto bc = op.bcDiag;
  double sx = 0.0, sv = 0.0;
  Kokkos::parallel_reduce(
      "amr::fv_rmean", op.n,
      KOKKOS_LAMBDA(const Index i, double& an, double& ad) {
        double d = bc(i);
        for (Index k = fs(i); k < fs(i + 1); ++k)
          d += fw(k);
        if (d > 1e-30) {
          an += u(i) / invVol(i);
          ad += 1.0 / invVol(i);
        }
      },
      sx, sv);
  double loc[2] = {sx, sv}, glob[2] = {0.0, 0.0};
  MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
  if (glob[1] <= 0.0)
    return;
  const double m = glob[0] / glob[1];
  Kokkos::parallel_for(
      "amr::fv_rmean_sub", op.n, KOKKOS_LAMBDA(const Index i) {
        double d = bc(i);
        for (Index k = fs(i); k < fs(i + 1); ++k)
          d += fw(k);
        if (d > 1e-30)
          u(i) -= m;
      });
}

template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class DistributedFlowMultigrid {
 public:
  using DO = DistributedOctree<Dim, Bits>;
  using Octree = typename DO::Octree;
  using Poisson = AmrPoisson<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;

  /// Build with cut-cell openness `openFn(faceCentreWorld, axis) → [0,1]` on the flow's
  /// distributed octree (graded + cross-block 2:1 balanced). Collective.
  ///
  /// `shared0` (optional): a FINALIZED LeafHalo whose registry is a superset of level 0's
  /// face probes — the distributed AmrFlow passes its ±2 flow registry so the level-0
  /// operator's ghost columns index the SAME extended layout as the flow's cell vectors
  /// (φ, the overlay chains, the PCG/BiCGStab scratch): one layout, no cross-indexing.
  /// Level-0 probes (±1 face reach incl. finer sub-neighbours) are a subset of the flow's
  /// forEachFaceFull discovery, so every probe resolves from the frozen registry.
  ///
  /// `liftRoot` / `bottomExtent` continue the ladder BELOW the root brick (docs/amr_mg_depth.md
  /// §6.2/§6.4): the lift is LOCKSTEP across ranks, so every rank builds the same number of levels
  /// and every per-level collective still pairs up.
  template <class OpenFn>
  void build(const DO& finest, double h0, OpenFn&& openFn,
             const LeafHalo<Dim, Bits>* shared0 = nullptr, bool liftRoot = true,
             Index bottomExtent = 4) {
    build(finest, detail::filledVec<Dim>(h0), std::forward<OpenFn>(openFn), shared0, liftRoot,
          bottomExtent);
  }
  /// Phase 3: the finest spacing per axis; every level inherits the aspect ratio (AM1).
  template <class OpenFn>
  void build(const DO& finest, const Vec<Dim>& h0, OpenFn&& openFn,
             const LeafHalo<Dim, Bits>* shared0 = nullptr, bool liftRoot = true,
             Index bottomExtent = 4) {
    buildImpl(finest, h0, shared0, liftRoot, bottomExtent);
    // Openness ladder: finest level directly from the world-coord openFn (local + ghost rows,
    // both exact); coarser levels by the exact single-rank child-face averaging for local rows
    // + a one-time owner exchange for ghost rows.
    levels_[0]->ap.buildOpenness(openFn);
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L)
      coarsenOpennessTo(L);
    finishOps();
    buildStage();  // §6.5: below the in-place ladder the level MOVES to a new decomposition
  }

  /// Per-level nullspace projection for the singular (periodic pure-Neumann) pressure. The stage's
  /// continued ladder follows this flag rather than being pinned on: on the production path flow
  /// sets it true, which is docs/amr_mg_depth.md §6.5.1 step 4's `setRemoveMean(true)`, and with it
  /// off the stage stays what §6.5 calls it — the continuation of the same V-cycle — so the
  /// WORLD==SELF bitwise contract of §10 still holds where the stage fires.
  void setRemoveMean(bool on) {
    removeMean_ = on;
    if (stage_)
      stage_->setRemoveMean(on);
  }

  std::size_t numLevels() const { return levels_.size(); }
  /// How many of those levels are IN PLACE — octree coarsenings plus lifted levels, i.e. the ones
  /// that keep the ORB and whose transfers are local. A stage's levels (docs/amr_mg_depth.md
  /// §6.5) are appended after them. Compare against `predictPressureLadder(...).numInPlace()`.
  std::size_t numInPlaceLevels() const { return levels_.size(); }
  /// How many levels the ladder gained BELOW the root brick by lifting (§6.2).
  Index liftDepth() const { return liftDepth_; }
  /// Whether a STAGE fires below the in-place ladder (docs/amr_mg_depth.md §5.6/§6.5): the
  /// coarsest in-place level is still larger than `bottomExtent` on some axis, so that level moves
  /// onto a new decomposition of its own grid and the ladder continues there. Only the REPLICATED
  /// instantiation exists today (one block on every rank, moved by `Allgatherv`); the sibling-merge
  /// and repartition stages are WO4b and slot in behind the same `MgStage` interface.
  bool hasStage() const { return stage_ != nullptr; }
  const char* stageKind() const { return stage_ ? stage_->kind() : "none"; }
  /// Levels of the continued ladder (its level 0 IS the moved level, so the full ladder is
  /// `numInPlaceLevels() + numStageLevels()` — the convention `predictPressureLadder` reports).
  std::size_t numStageLevels() const { return stage_ ? stage_->numLevels() : 0u; }
  Index stageLeaves(std::size_t L = 0) const { return stage_->numLeaves(L); }
  /// The global grid the stage moves (the coarsest in-place level's extent).
  const IVec<Dim>& stageFrom() const { return stageFrom_; }
  /// The moved level's solution on THIS rank's target block. For the replicated stage every rank
  /// holds an identical copy, which is what §11.2's cross-rank bitwise check reads.
  View<double> stageSolution() { return stage_->targetX(); }
  MgStage<Dim, Bits>& stage() { return *stage_; }
  /// `"jacobi"` | (WO5) `"amg"`, suffixed by the stage's own spelling — `"+tail"` for the
  /// replicated one (docs/amr_mg_depth.md §6.7).
  std::string bottomName() const {
    return std::string("jacobi") + (stage_ ? stage_->diagnosticSuffix() : "");
  }
  Index numLeaves(std::size_t L = 0) const { return levels_[L]->n; }
  Index extendedSize(std::size_t L = 0) const { return levels_[L]->nExt; }
  View<double> x(std::size_t L = 0) { return levels_[L]->x; }
  View<double> b(std::size_t L = 0) { return levels_[L]->b; }
  const FvOp& op(std::size_t L = 0) const { return levels_[L]->op; }
  MPI_Comm comm() const { return comm_; }
  const Poisson& poisson(std::size_t L = 0) const { return levels_[L]->ap; }
  const LeafHalo<Dim, Bits>& halo(std::size_t L = 0) const { return *levels_[L]->hp; }

  /// Refresh the ghost tail of a level-L vector (the PCG matvec hook uses level 0).
  void sync(std::size_t L, View<double> v) const { levels_[L]->ex.exchange(v); }

  /// One V-cycle on level L (correction scheme), the distributed mirror of
  /// Multigrid::vcycle: ghost refresh before every Jacobi sweep / residual, local transfers,
  /// Allreduce'd mean removal.
  void vcycle(int pre = 2, int post = 2, int bottom = 40, double omega = 0.8, std::size_t L = 0) {
    Level& lv = *levels_[L];
    View<const double> bc(lv.b);
    if (L + 1 == levels_.size()) {
      if (stage_) {
        // §6.5: the level MOVES onto the stage's decomposition, the ladder continues there, the
        // correction comes back. No MPI in this function — the messages live behind the stage.
        stage_->apply(bc, lv.x, lv.n, pre, post, bottom, omega);
      } else {
        for (int s = 0; s < bottom; ++s) {
          lv.ex.exchange(lv.x);
          jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
        }
      }
      if (removeMean_)
        removeMeanFvDist(lv.op, lv.x, lv.comm);
      return;
    }
    for (int s = 0; s < pre; ++s) {
      lv.ex.exchange(lv.x);
      jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
    }
    lv.ex.exchange(lv.x);
    residualFv(lv.op, View<const double>(lv.x), bc, lv.res);
    Level& cl = *levels_[L + 1];
    restrictField(lv.childStart, lv.childIdx, View<const double>(lv.res), cl.b, cl.n);
    Kokkos::deep_copy(cl.x, 0.0);
    vcycle(pre, post, bottom, omega, L + 1);
    prolongAdd(lv.c2p, View<const double>(cl.x), lv.x, lv.n);
    for (int s = 0; s < post; ++s) {
      lv.ex.exchange(lv.x);
      jacobiFv(lv.op, lv.x, bc, lv.tmp, omega);
    }
    if (removeMean_)
      removeMeanFvDist(lv.op, lv.x, lv.comm);
  }

 private:
  struct Level {
    DO d;  // coarsened copy of the flow octree (carries the decomposition)
    Poisson ap;
    LeafHalo<Dim, Bits> halo;                 // own registry (unused when hp aliases shared0)
    const LeafHalo<Dim, Bits>* hp = nullptr;  // the registry in force (own or the shared one)
    LeafHaloExchange ex;
    FvOp op;
    Index n = 0, nExt = 0;
    /// THIS level's communicator. Every level shares the parent's today; a stage on a
    /// sub-communicator (WO4b) is where they start to differ, and every per-level collective
    /// (`removeMeanFvDist`, the halo build) already reads it from here rather than from the
    /// multigrid (docs/amr_mg_depth.md §6.5).
    MPI_Comm comm = MPI_COMM_NULL;
    View<double> x, b, res, tmp;  // x/b sized nExt (PCG deep_copies match); res/tmp local
    View<Index> c2p, childStart, childIdx;
    std::vector<Index> c2pHost;  // kept for the openness coarsening
  };

  /// §6.5, once per build. Below the in-place ladder the coarsest level does not just get
  /// smoothed: if its global grid is still larger than `bottomExtent` on some axis, it MOVES onto
  /// a new decomposition of its own grid and the ladder continues there. Only the REPLICATED
  /// instantiation exists today — one block on every rank, `Allgatherv` — which is the degenerate
  /// stage and the fallback; the sibling-merge and repartition stages of §5.6 are WO4b and land
  /// behind the same interface without touching `vcycle`.
  ///
  /// It runs AFTER the openness ladder, because what the stage carries over is that ladder's own α
  /// rows for the moved level — re-sampling the geometry at a coarsened cut face is not the same
  /// number.
  void buildStage() {
    stage_.reset();
    if (levels_.empty())
      return;
    Level& lt = *levels_.back();
    stageFrom_ = lt.d.globalRootSize();
    int size = 1;
    MPI_Comm_size(comm_, &size);
    Index mx = 0;
    for (int a = 0; a < Dim; ++a)
      mx = std::max(mx, stageFrom_[a]);
    if (size <= 1 || mx <= bottomExtent_)
      return;  // single-rank: the in-place ladder IS the whole hierarchy; small enough: no stage
    auto tail = std::make_unique<ReplicatedTailStage<Dim, Bits>>();
    tail->build(lt.d, h0_, lt.ap, lt.n, lt.nExt, bottomExtent_, lt.comm);
    tail->setRemoveMean(removeMean_);
    stage_ = std::move(tail);
  }

  void buildImpl(const DO& finest, const Vec<Dim>& h0, const LeafHalo<Dim, Bits>* shared0,
                 bool liftRoot, Index bottomExtent) {
    comm_ = finest.comm();
    h0_ = h0;
    bottomExtent_ = bottomExtent;
    levels_.clear();
    // Ladder of coarsened copies of the SAME distributed octree (decomposition preserved).
    {
      auto l0 = std::make_unique<Level>();
      l0->d = finest;
      levels_.push_back(std::move(l0));
      for (;;) {
        Octree c = levels_.back()->d.local();
        const Index before = c.numLeaves();
        const Index merged = c.coarsenIf([](Code, unsigned) { return true; });
        if (merged == 0 || c.numLeaves() == before)
          break;
        auto lv = std::make_unique<Level>();
        lv->d = levels_.back()->d;
        lv->d.local() = std::move(c);
        levels_.push_back(std::move(lv));
        if (levels_.back()->d.local().numLeaves() == 1)
          break;
      }
      // Pad to the global max level count (identity root-brick levels): the per-level halo
      // point-to-points and build collectives must pair up across ranks, and the extra
      // smoothing exactly mirrors COMM_SELF's already-coarse regions.
      int nl = static_cast<int>(levels_.size()), gnl = nl;
      MPI_Allreduce(&nl, &gnl, 1, MPI_INT, MPI_MAX, comm_);
      while (static_cast<int>(levels_.size()) < gnl) {
        auto lv = std::make_unique<Level>();
        lv->d = levels_.back()->d;
        levels_.push_back(std::move(lv));
      }
      // Below the root brick (docs/amr_mg_depth.md §6.2/§6.4). Every rank now sits at its root
      // brick (the coarsenIf loop ran to exhaustion; the padded ranks repeat theirs), so the lift
      // is a pure function of the global root grid and the ORB blocks — and it is LOCKSTEP: the
      // depth is the MIN over ranks of the depth each rank's own block allows, Allreduced BEFORE
      // any level is built. Every rank therefore appends the SAME number of levels and the
      // per-level halo collectives below keep pairing up (the padding argument, extended by k).
      if (liftRoot)
        liftDepth_ = allowedLiftDepth(levels_.back()->d, bottomExtent);
      for (Index j = 0; j < liftDepth_; ++j) {
        auto lv = std::make_unique<Level>();
        lv->d = levels_.back()->d;  // copy: carries the decomposition AND the leaf set
        lv->d.liftRoot();           // root brick halved, lmax + 1, codes/levels untouched
        const Index before = lv->d.local().numLeaves();
        const Index merged = lv->d.local().coarsenIf([](Code, unsigned) { return true; });
        if (merged == 0 || lv->d.local().numLeaves() == before)
          break;  // defensive: a level that did not shrink would stall the ladder
        levels_.push_back(std::move(lv));
      }
      // The level counts MUST be equal on every rank — every per-level halo build and exchange
      // below is a collective. Cheap (once per build) and a hard error, not an assert.
      {
        int nlv = static_cast<int>(levels_.size()), lo = 0, hi = 0;
        MPI_Allreduce(&nlv, &lo, 1, MPI_INT, MPI_MIN, comm_);
        MPI_Allreduce(&nlv, &hi, 1, MPI_INT, MPI_MAX, comm_);
        if (lo != hi)
          throw std::runtime_error(
              "amr::DistributedFlowMultigrid: the pressure ladder has different level counts on "
              "different ranks (the lift is not lockstep)");
      }
    }
    // Per level: seam install + discovery fixpoint + topology freeze. Collective per level —
    // every rank walks its levels in the same order (counts padded), so the resolveMisses /
    // coverLevels rounds stay matched.
    const Vec<Dim> gorigin = finest.globalGeometry().origin;
    std::array<long, Dim> shift{};
    for (int a = 0; a < Dim; ++a)
      shift[a] = finest.blockFineOrigin()[a];
    shift_ = shift;
    bool first = true;
    for (auto& lvp : levels_) {
      Level& lv = *lvp;
      lv.comm = comm_;  // every level shares the parent's communicator until WO4b's sub-comms
      lv.n = lv.d.local().numLeaves();
      lv.ap.init(lv.d.local(), h0_);
      lv.ap.setOrigin(gorigin);
      lv.ap.setFrameShift(shift);
      // Install the ghost metadata of a registry into ap (level/lo lookups during the walks).
      auto installGhosts = [&](const LeafHalo<Dim, Bits>& h) {
        std::vector<std::array<long, Dim>> glo(static_cast<std::size_t>(h.numGhosts()));
        std::vector<unsigned> glv(static_cast<std::size_t>(h.numGhosts()));
        for (Index g = 0; g < h.numGhosts(); ++g) {
          for (int a = 0; a < Dim; ++a)
            glo[static_cast<std::size_t>(g)][a] = static_cast<long>(h.ghostCoord(g)[a]) - shift[a];
          glv[static_cast<std::size_t>(g)] = static_cast<unsigned>(h.level(h.numLocal() + g));
        }
        lv.ap.setGhosts(std::move(glo), std::move(glv));
      };
      if (first && shared0) {
        // Level 0 on the flow's frozen ±2 registry: same extended layout as the flow's cell
        // vectors; every ±1 face probe is already cached ⇒ const lookups, no discovery.
        lv.hp = shared0;
        lv.ap.setResolver([shared0, shift](const std::array<long, Dim>& p) -> Index {
          std::array<long, Dim> g = p;
          for (int a = 0; a < Dim; ++a)
            g[a] += shift[a];
          return shared0->lookupGlobal(g);
        });
        installGhosts(*shared0);
      } else {
        lv.hp = &lv.halo;
        lv.halo.init(lv.d);
        LeafHalo<Dim, Bits>* hp = &lv.halo;
        lv.ap.setResolver([hp, shift](const std::array<long, Dim>& p) -> Index {
          std::array<long, Dim> g = p;
          for (int a = 0; a < Dim; ++a)
            g[a] += shift[a];
          return hp->resolveGlobal(g);
        });
        // Ghosts must be (re-)installed into ap at the TOP of every fixpoint round: a probe
        // that resolved in an earlier round returns its ghost slot immediately, and probeSlot
        // then reads ap.levelOf(slot) — which must already cover it (newly-PENDING coords are
        // fine: they return kPending and are skipped until the next round).
        for (;;) {
          installGhosts(lv.halo);
          for (Index i = 0; i < lv.n; ++i)
            lv.ap.forEachFaceNeighbor(i, [](Index, Real, int, double) {});
          if (lv.halo.resolveMisses() == 0)
            break;
        }
        lv.halo.finalize();
      }
      lv.nExt = lv.hp->extendedSize();
      lv.ex.init(*lv.hp);
      first = false;
    }
    // Local covering-leaf transfers (parents never cross blocks); identity on padded levels.
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L) {
      const Octree& f = levels_[L]->d.local();
      const Octree& c = levels_[L + 1]->d.local();
      const Index nf = f.numLeaves(), nc = c.numLeaves();
      std::vector<Index>& c2p = levels_[L]->c2pHost;
      c2p.assign(static_cast<std::size_t>(nf), -1);
      std::vector<Index> cnt(static_cast<std::size_t>(nc), 0);
      for (Index i = 0; i < nf; ++i) {
        const Index p = c.find(f.code(i));  // covering construction (== single-rank Multigrid)
        c2p[static_cast<std::size_t>(i)] = p;
        if (p >= 0)
          ++cnt[static_cast<std::size_t>(p)];
      }
      std::vector<Index> start(static_cast<std::size_t>(nc) + 1, 0);
      for (Index p = 0; p < nc; ++p)
        start[static_cast<std::size_t>(p) + 1] =
            start[static_cast<std::size_t>(p)] + cnt[static_cast<std::size_t>(p)];
      std::vector<Index> idx(static_cast<std::size_t>(start[static_cast<std::size_t>(nc)]));
      std::vector<Index> cur(start.begin(), start.end() - 1);
      for (Index i = 0; i < nf; ++i) {  // fine order ⇒ deterministic restrict accumulation
        const Index p = c2p[static_cast<std::size_t>(i)];
        if (p >= 0)
          idx[static_cast<std::size_t>(cur[static_cast<std::size_t>(p)]++)] = i;
      }
      levels_[L]->c2p = toDevice(c2p, "dfmg_c2p");
      levels_[L]->childStart = toDevice(start, "dfmg_cstart");
      levels_[L]->childIdx = toDevice(idx, "dfmg_cidx");
    }
  }

  // Area-average level-L face openness onto level L+1 — the EXACT AmrMultigrid::coarsenOpenness
  // arithmetic for the local rows (children of a local coarse cell are local, summed in local
  // Z-order == COMM_SELF's relative order ⇒ bit-identical), then ghost α rows exchanged from
  // the owners through the level-(L+1) halo (kFaces one-time host exchanges).
  void coarsenOpennessTo(std::size_t L) {
    const Octree& f = levels_[L]->d.local();
    const Octree& c = levels_[L + 1]->d.local();
    Poisson& capL = levels_[L]->ap;
    const int F = 2 * Dim;
    const Index nc = c.numLeaves();
    const Index ngc = levels_[L + 1]->hp->numGhosts();
    std::vector<double> ca(static_cast<std::size_t>(nc + ngc) * F, 0.0);
    std::vector<int> cnt(static_cast<std::size_t>(nc) * F, 0);
    const std::vector<Index>& c2p = levels_[L]->c2pHost;
    for (Index i = 0; i < f.numLeaves(); ++i) {
      const Index p = c2p[static_cast<std::size_t>(i)];
      if (p < 0)
        continue;
      const std::size_t base = static_cast<std::size_t>(p) * F;
      if (c.level(p) == f.level(i)) {
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            const int fi = Poisson::faceIndex(axis, dir);
            ca[base + static_cast<std::size_t>(fi)] += capL.faceOpenness(i, axis, dir);
            cnt[base + static_cast<std::size_t>(fi)] += 1;
          }
      } else {
        const unsigned oct = M::from_code(f.code(i)).child_index(f.level(i));
        for (int axis = 0; axis < Dim; ++axis) {
          const int dir = ((oct >> axis) & 1) ? +1 : -1;
          const int fi = Poisson::faceIndex(axis, dir);
          ca[base + static_cast<std::size_t>(fi)] += capL.faceOpenness(i, axis, dir);
          cnt[base + static_cast<std::size_t>(fi)] += 1;
        }
      }
    }
    for (std::size_t k = 0; k < static_cast<std::size_t>(nc) * F; ++k)
      ca[k] = cnt[k] ? ca[k] / cnt[k] : 1.0;
    // Ghost rows from the owners (their local rows were computed by the identical arithmetic).
    const LeafHalo<Dim, Bits>& h = *levels_[L + 1]->hp;
    std::vector<double> col(static_cast<std::size_t>(levels_[L + 1]->nExt), 0.0);
    for (int fi = 0; fi < F; ++fi) {
      for (Index i = 0; i < nc; ++i)
        col[static_cast<std::size_t>(i)] = ca[static_cast<std::size_t>(i) * F + fi];
      h.exchangeHost(col);
      for (Index g = 0; g < ngc; ++g)
        ca[static_cast<std::size_t>(nc + g) * F + fi] = col[static_cast<std::size_t>(nc + g)];
    }
    levels_[L + 1]->ap.setOpennessRaw(std::move(ca));
  }

  // Host-assemble each level's operator through the resolver seam and upload (the distributed
  // mirror of the D5 device assembly — same CSR, ghost columns included), then the scratch.
  void finishOps() {
    for (auto& lvp : levels_) {
      Level& lv = *lvp;
      auto A = lv.ap.assembleFv();
      lv.op.n = lv.n;
      lv.op.invVol = toDevice(A.invVol, "dfmg_invvol");
      lv.op.faceStart = toDevice(A.start, "dfmg_start");
      lv.op.faceNbr = toDevice(A.nbr, "dfmg_nbr");
      lv.op.faceW = toDevice(A.coef, "dfmg_w");
      lv.op.bcDiag = toDevice(A.bcDiag, "dfmg_bc");
      lv.x = View<double>("dfmg_x", static_cast<std::size_t>(lv.nExt));
      lv.b = View<double>("dfmg_b", static_cast<std::size_t>(lv.nExt));
      lv.res = View<double>("dfmg_res", static_cast<std::size_t>(lv.n));
      lv.tmp = View<double>("dfmg_tmp", static_cast<std::size_t>(lv.n));
      Kokkos::deep_copy(lv.x, 0.0);
      Kokkos::deep_copy(lv.b, 0.0);
    }
  }

  /// How many lifts the §6.2 rule allows, as the MIN over ranks (one Allreduce). Iterated
  /// arithmetically on `(G, blockOrigin, blockBrick) / 2^j` so no octree is copied to find out.
  static Index allowedLiftDepth(const DO& d, Index bottomExtent) {
    IVec<Dim> G = d.globalRootSize(), o = d.blockOriginRoot(), b = d.blockBrick();
    int kLocal = 0;
    for (;;) {
      Index mx = 0;
      for (int a = 0; a < Dim; ++a)
        mx = std::max(mx, G[a]);
      if (mx <= bottomExtent)
        break;  // the bottom smoother is already exact here (§6.6)
      bool ok = true;
      for (int a = 0; a < Dim; ++a)
        if ((G[a] % 2) != 0 || (G[a] / 2) < 2 || (o[a] % 2) != 0 || (b[a] % 2) != 0) {
          ok = false;  // grid-limited, or decomposition-limited (this rank's block turns odd)
          break;
        }
      if (!ok)
        break;
      for (int a = 0; a < Dim; ++a) {
        G[a] /= 2;
        o[a] /= 2;
        b[a] /= 2;
      }
      ++kLocal;
    }
    int kGlobal = kLocal;
    MPI_Allreduce(&kLocal, &kGlobal, 1, MPI_INT, MPI_MIN, d.comm());
    return static_cast<Index>(kGlobal);
  }

  MPI_Comm comm_ = MPI_COMM_NULL;
  Vec<Dim> h0_ = detail::filledVec<Dim>(1.0);
  Index bottomExtent_ = 4;
  Index liftDepth_ = 0;
  IVec<Dim> stageFrom_{};
  std::unique_ptr<MgStage<Dim, Bits>> stage_;
  std::array<long, Dim> shift_{};
  std::vector<std::unique_ptr<Level>> levels_;
  bool removeMean_ = false;
};

}  // namespace peclet::amr

#endif  // PECLET_AMR_DISTRIBUTED_FLOW_MG_HPP
