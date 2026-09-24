// amr — a multigrid STAGE: move a level, at its own resolution, onto a new decomposition of that
// level's grid, and continue the ladder there. docs/amr_mg_depth.md §5.6 / §6.5, and
// suite/docs/archive/MG_TELESCOPING_PLAN.md §4.0, whose primitive this is.
//
// The in-place ladder (§6.2) lifts the root while every rank's ORB block stays even in origin and
// size. It stops somewhere — grid-limited on a badly factored brick mesh, decomposition-limited as
// soon as a block turns odd, which under a WEIGHTED (brick-granular, DEM-shared) partition is
// typically the root brick level itself. Below that point the hierarchy does not stop: the level
// MOVES onto a coarser decomposition of its own grid and the ladder continues there. §5.6's three
// instantiations, in the order the trigger tries them:
//
//   * sibling merge — target `BlockDecomposer::agglomerated(d)`, group gather; proportional trees;
//   * repartition   — target a fresh proportional ORB on np_L <= np ranks, moved by core's
//                     `redistributeGridFields`; weighted trees, where no d > 0 is liftable;
//   * REPLICATED    — target one block on every rank, moved by `Allgatherv`; the bottom on the
//                     last sub-communicator, and the fallback. THIS FILE implements this one.
//
// The other two are WO4b. They are why this is an interface rather than a function: what differs
// between them is only the target decomposition and the two movement steps (`moveUp` / `moveDown`);
// the continued ladder, the per-level communicator and the way `vcycle` calls the stage are shared.
// `DistributedFlowMultigrid::vcycle` never sees a message — it calls `apply` at the stage point.
#ifndef PECLET_AMR_MG_STAGE_HPP
#define PECLET_AMR_MG_STAGE_HPP

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "peclet/amr/block_octree.hpp"
#include "peclet/amr/common.hpp"
#include "peclet/amr/distributed_octree.hpp"
#include "peclet/amr/multigrid.hpp"
#include "peclet/amr/poisson.hpp"
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

namespace peclet::amr {

/// One stage of the pressure hierarchy (docs/amr_mg_depth.md §6.5). The five things a stage IS —
/// a target decomposition, a target communicator with an active flag, the moved fields, the
/// level's cells on the target block as a `BlockOctree`, and the continued ladder below it — are
/// the five accessors below; the two movement steps are the only part the instantiations differ in.
///
/// Owned by `DistributedFlowMultigrid` (at most one, built in its `buildStage` after the openness
/// ladder) and driven from its `vcycle` through `apply` at the stage point; `pressure_mg_levels`
/// and `pressure_mg_bottom` report it. Collective semantics are per method, below: movement is
/// collective on the PARENT communicator, the continued ladder on `targetComm()`.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class MgStage {
 public:
  using Octree = BlockOctree<Dim, Bits>;                     ///< a level's cells on one block
  using Decomposition = core::decomp::BlockDecomposer<Dim>;  ///< the target partition's type

  virtual ~MgStage() = default;

  // ---- what the stage is ---------------------------------------------------------------------
  /// `"replicated"` | (WO4b) `"sibling"` | `"repartition"`.
  virtual const char* kind() const = 0;
  /// What `pressure_mg_bottom` appends for this stage — `"+tail"` for the replicated one, which is
  /// the spelling docs/amr_mg_depth.md §6.7 fixes.
  virtual const char* diagnosticSuffix() const = 0;
  /// The decomposition of the moved level's grid this stage targets.
  virtual const Decomposition& targetDecomposition() const = 0;
  /// The communicator every collective of the continued ladder runs on.
  virtual MPI_Comm targetComm() const = 0;
  /// Whether THIS rank owns a target block, i.e. takes part in the continued ladder. Ranks that do
  /// not still take part in the MOVEMENT, on the parent communicator, and skip the recursion.
  virtual bool active() const = 0;
  /// The moved level's cells on this rank's target block.
  virtual const Octree& targetOctree() const = 0;

  // ---- the continued ladder (its level 0 IS the moved level) ---------------------------------
  /// Levels of the continued ladder, its level 0 (the moved level) included. Local.
  virtual std::size_t numLevels() const = 0;
  /// Cells of continued-ladder level `L` on this rank's target block — the GLOBAL count for the
  /// replicated stage, whose one block is the whole level. Local.
  virtual Index numLeaves(std::size_t L) const = 0;
  /// `"jacobi"` | `"amg"`: what solves the continued ladder's coarsest level, WITHOUT the stage's
  /// `diagnosticSuffix()` (the owner appends it). Local.
  virtual std::string bottomName() const = 0;
  /// Per-level constant-nullspace projection for the singular periodic pressure; the owner forwards
  /// its own flag (`DistributedFlowMultigrid::setRemoveMean`). Local.
  virtual void setRemoveMean(bool on) = 0;
  /// What solves the continued ladder's coarsest level (docs/amr_mg_depth.md §6.6). The exact
  /// bottom lives in the single-rank `Multigrid`, so this is where the selector lands.
  virtual void setBottom(typename Multigrid<Dim, Bits>::Bottom b) = 0;
  /// The continued ladder's level-0 solution — what `moveDown` reads. A test uses it to assert the
  /// stage produced the same answer everywhere it is replicated (§11.2).
  virtual View<double> targetX() = 0;

  // ---- the movement: the ONLY part the three instantiations differ in ------------------------
  /// `src` (this rank's rows of the moved level, device; may be longer than `nSrc`) → the target's
  /// level-0 rhs. Collective on the PARENT communicator: every rank takes part, active or not.
  virtual void moveUp(View<const double> src, Index nSrc) = 0;
  /// One cycle of the continued ladder on `targetComm()`. Called on active ranks only.
  virtual void cycle(int pre, int post, int bottom, double omega) = 0;
  /// The target's level-0 solution → the first `nDst` rows of `dst`.
  virtual void moveDown(View<double> dst, Index nDst) = 0;

  /// The whole stage for one V-cycle, in the CORRECTION SCHEME §6.5 prescribes: the caller hands
  /// in the residual at resolution L and gets back a correction to ADD. That is what makes the
  /// stage legal at every level, including L = 0 — where the level being moved is the finest one
  /// (no in-place lift was possible at all) and its iterate must not be thrown away. Where the
  /// stage fires below level 0 the incoming iterate is zero, so residual == rhs and add ==
  /// overwrite, and this is bit-for-bit §6.5.1's simpler description.
  void apply(View<const double> residual, View<double> correction, Index n, int pre, int post,
             int bottom, double omega) {
    moveUp(residual, n);
    if (active())
      cycle(pre, post, bottom, omega);
    moveDown(correction, n);
  }
};

/// The REPLICATED stage (docs/amr_mg_depth.md §6.5.1): the target decomposition is one block —
/// the whole level — on every rank, the movement is an `Allgatherv` keyed by global cell id, and
/// the continued ladder is a single-rank `Multigrid` on the gathered grid, running its own §6.2
/// lifts down to its own §6.6 bottom. Every rank is active and every rank computes the identical
/// tail, so `moveDown` is a local pick with no communication and the result is
/// decomposition-independent by construction.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class ReplicatedTailStage final : public MgStage<Dim, Bits> {
 public:
  using Base = MgStage<Dim, Bits>;                     ///< the stage interface
  using Octree = typename Base::Octree;                ///< the gathered level, one block
  using Decomposition = typename Base::Decomposition;  ///< one block: the whole level
  using DO = DistributedOctree<Dim, Bits>;             ///< the level the stage fires at
  using Poisson = AmrPoisson<Dim, Bits>;               ///< that level's FV operator

  /// Global cell id (x-fastest, CONVENTIONS.md) of a coordinate on the moved level's grid.
  static long long gidOf(const IVec<Dim>& g, const IVec<Dim>& G) {
    long long id = 0;
    for (int d = Dim - 1; d >= 0; --d)
      id = id * static_cast<long long>(G[d]) + static_cast<long long>(g[d]);
    return id;
  }

  /// Build the stage from the level it fires at: `d` is that level's distributed octree (lifted,
  /// so `rootSpan()` is the W of §6.5's `g_i = (blockFineOrigin + lo_i)/W`), `ap` its `AmrPoisson`
  /// — whose `opennessRaw()` rows are the DISTRIBUTED openness ladder's own output and are carried
  /// over rather than re-sampled — and `n` its local cell count. Collective on `comm`.
  ///
  /// Every rank then holds the whole level (`G = d.globalRootSize()` cells) and a single-rank
  /// `Multigrid` on it, lifted down to `bottomExtent` with its own §6.6 bottom (default selection
  /// `auto`; the owner re-sets it). `nExt` is unused (the extended size of the caller's views,
  /// kept for the WO4b stages' signature). Memory: O(G) doubles and one host `Multigrid` per rank.
  ///
  /// @pre `d` is nested (every local leaf is a root cell of the lifted grid) and periodic on
  ///      every axis.
  /// @throws std::runtime_error if an axis is not periodic, if the ranks' leaves do not tile the
  ///         global grid (the lift was not nested), or on a global id out of range.
  void build(const DO& d, const Vec<Dim>& h0, const Poisson& ap, Index n, Index nExt,
             Index bottomExtent, MPI_Comm comm) {
    comm_ = comm;
    int size = 1;
    MPI_Comm_size(comm_, &size);
    G_ = d.globalRootSize();
    nt_ = 1;
    for (int a = 0; a < Dim; ++a)
      nt_ *= G_[a];
    for (int a = 0; a < Dim; ++a)
      if (!d.periodic()[a])
        throw std::runtime_error(
            "amr::ReplicatedTailStage: periodic-only (as the whole distributed pressure path is); "
            "a non-periodic bottom must be gated the way flow gates its anchored path");
    dec_.init(1, G_);  // the target decomposition: ONE block, the whole level, on every rank
    oct_.init(G_, d.lmax(), IVec<Dim>{});
    const Index W = d.rootSpan();

    // This rank's rows, keyed by global cell id. Every leaf of the moved level is a root cell of
    // the lifted root grid, so the division by W is exact (that is what nesting buys).
    std::vector<long long> gid(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
      auto b = d.local().bounds(i);
      IVec<Dim> g{};
      for (int a = 0; a < Dim; ++a)
        g[a] = (static_cast<Index>(b[0][a]) + d.blockFineOrigin()[a]) / W;
      gid[static_cast<std::size_t>(i)] = gidOf(g, G_);
    }

    counts_.assign(static_cast<std::size_t>(size), 0);
    int ni = static_cast<int>(n);
    MPI_Allgather(&ni, 1, MPI_INT, counts_.data(), 1, MPI_INT, comm_);
    displs_.assign(static_cast<std::size_t>(size), 0);
    int tot = 0;
    for (int r = 0; r < size; ++r) {
      displs_[static_cast<std::size_t>(r)] = tot;
      tot += counts_[static_cast<std::size_t>(r)];
    }
    if (tot != static_cast<int>(nt_))
      throw std::runtime_error(
          "amr::ReplicatedTailStage: the moved level does not tile its global grid (the lift is "
          "not nested)");
    std::vector<long long> gidAll(static_cast<std::size_t>(tot));
    MPI_Allgatherv(gid.data(), ni, MPI_LONG_LONG, gidAll.data(), counts_.data(), displs_.data(),
                   MPI_LONG_LONG, comm_);

    std::vector<Index> rowOfGid(static_cast<std::size_t>(nt_), -1);
    for (Index r = 0; r < nt_; ++r) {
      auto b = oct_.bounds(r);
      IVec<Dim> g{};
      for (int a = 0; a < Dim; ++a)
        g[a] = static_cast<Index>(b[0][a]) / W;
      rowOfGid[static_cast<std::size_t>(gidOf(g, G_))] = r;
    }
    rowOfGathered_.resize(static_cast<std::size_t>(tot));
    for (int k = 0; k < tot; ++k) {
      const long long id = gidAll[static_cast<std::size_t>(k)];
      if (id < 0 || id >= static_cast<long long>(nt_))
        throw std::runtime_error("amr::ReplicatedTailStage: gid out of range");
      rowOfGathered_[static_cast<std::size_t>(k)] = rowOfGid[static_cast<std::size_t>(id)];
    }
    rowOfLocal_.resize(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i)
      rowOfLocal_[static_cast<std::size_t>(i)] =
          rowOfGid[static_cast<std::size_t>(gid[static_cast<std::size_t>(i)])];

    // The moved level's α rows, gathered and permuted into target leaf order.
    const int F = 2 * Dim;
    const std::vector<double>& aloc = ap.opennessRaw();
    std::vector<double> aSend(static_cast<std::size_t>(n) * F, 1.0);
    if (!aloc.empty())
      std::copy(aloc.begin(), aloc.begin() + static_cast<std::ptrdiff_t>(n) * F, aSend.begin());
    std::vector<int> cF(static_cast<std::size_t>(size)), dF(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
      cF[static_cast<std::size_t>(r)] = counts_[static_cast<std::size_t>(r)] * F;
      dF[static_cast<std::size_t>(r)] = displs_[static_cast<std::size_t>(r)] * F;
    }
    std::vector<double> aAll(static_cast<std::size_t>(tot) * F);
    MPI_Allgatherv(aSend.data(), ni * F, MPI_DOUBLE, aAll.data(), cF.data(), dF.data(), MPI_DOUBLE,
                   comm_);
    std::vector<double> aTail(static_cast<std::size_t>(nt_) * F, 1.0);
    for (int k = 0; k < tot; ++k) {
      const std::size_t dst =
          static_cast<std::size_t>(rowOfGathered_[static_cast<std::size_t>(k)]) *
          static_cast<std::size_t>(F);
      const std::size_t src = static_cast<std::size_t>(k) * static_cast<std::size_t>(F);
      for (int f = 0; f < F; ++f)
        aTail[dst + static_cast<std::size_t>(f)] = aAll[src + static_cast<std::size_t>(f)];
    }

    mg_ = std::make_unique<Multigrid<Dim, Bits>>();
    mg_->buildRaw(oct_, h0, std::move(aTail), /*periodic=*/true, /*liftRoot=*/true, bottomExtent);

    (void)nExt;
    sendBuf_.assign(static_cast<std::size_t>(n), 0.0);
    recvBuf_.assign(static_cast<std::size_t>(tot), 0.0);
    host_.assign(static_cast<std::size_t>(nt_), 0.0);
    // Sized by the level's OWN cell count, and staged through device scratch, so the caller may
    // hand in a longer view (the distributed levels carry a ghost tail) without a subview.
    dSrc_ = View<double>("stage_dsrc", static_cast<std::size_t>(n));
    dDst_ = View<double>("stage_ddst", static_cast<std::size_t>(n));
    srcMirror_ = Kokkos::View<double*, Kokkos::HostSpace>("stage_src", static_cast<std::size_t>(n));
    dstMirror_ = Kokkos::View<double*, Kokkos::HostSpace>("stage_dst", static_cast<std::size_t>(n));
    tbMirror_ = Kokkos::View<double*, Kokkos::HostSpace>("stage_tb", static_cast<std::size_t>(nt_));
    txMirror_ = Kokkos::View<double*, Kokkos::HostSpace>("stage_tx", static_cast<std::size_t>(nt_));
  }

  const char* kind() const override { return "replicated"; }
  const char* diagnosticSuffix() const override { return "+tail"; }
  const Decomposition& targetDecomposition() const override { return dec_; }
  MPI_Comm targetComm() const override { return comm_; }
  bool active() const override { return true; }  // replicated: every rank owns the whole level
  const Octree& targetOctree() const override { return oct_; }

  std::size_t numLevels() const override { return mg_->numLevels(); }
  Index numLeaves(std::size_t L) const override { return mg_->numLeaves(L); }
  std::string bottomName() const override { return mg_->bottomName(); }
  void setRemoveMean(bool on) override { mg_->setRemoveMean(on); }
  void setBottom(typename Multigrid<Dim, Bits>::Bottom b) override { mg_->setBottom(b); }
  View<double> targetX() override { return mg_->x(0); }

  /// The continued ladder itself — a single-rank `Multigrid`, exposed for the WO5 bottom selector.
  Multigrid<Dim, Bits>& multigrid() { return *mg_; }

  void moveUp(View<const double> src, Index nSrc) override {
    auto ds = dSrc_;
    Kokkos::parallel_for("amr::stage_pack", nSrc, KOKKOS_LAMBDA(const Index i) { ds(i) = src(i); });
    Kokkos::deep_copy(srcMirror_, dSrc_);
    for (Index i = 0; i < nSrc; ++i)
      sendBuf_[static_cast<std::size_t>(i)] = srcMirror_(i);
    MPI_Allgatherv(sendBuf_.data(), static_cast<int>(nSrc), MPI_DOUBLE, recvBuf_.data(),
                   counts_.data(), displs_.data(), MPI_DOUBLE, comm_);
    for (std::size_t k = 0; k < recvBuf_.size(); ++k)
      host_[static_cast<std::size_t>(rowOfGathered_[k])] = recvBuf_[k];
    for (Index r = 0; r < nt_; ++r)
      tbMirror_(r) = host_[static_cast<std::size_t>(r)];
    Kokkos::deep_copy(mg_->b(0), tbMirror_);
    Kokkos::deep_copy(mg_->x(0), 0.0);
  }

  void cycle(int pre, int post, int bottom, double omega) override {
    mg_->vcycle(pre, post, bottom, omega);
  }

  void moveDown(View<double> dst, Index nDst) override {
    Kokkos::deep_copy(txMirror_, mg_->x(0));
    for (Index i = 0; i < nDst; ++i)
      dstMirror_(i) = txMirror_(rowOfLocal_[static_cast<std::size_t>(i)]);
    Kokkos::deep_copy(dDst_, dstMirror_);
    auto dd = dDst_;
    Kokkos::parallel_for(
        "amr::stage_scatter", nDst, KOKKOS_LAMBDA(const Index i) { dst(i) = dd(i); });
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  IVec<Dim> G_{};
  Index nt_ = 0;
  Decomposition dec_;
  Octree oct_;
  std::unique_ptr<Multigrid<Dim, Bits>> mg_;
  std::vector<int> counts_, displs_;
  std::vector<Index> rowOfLocal_, rowOfGathered_;
  std::vector<double> sendBuf_, recvBuf_, host_;
  View<double> dSrc_, dDst_;
  /// Persistent host staging for the device↔host round trip (one allocation per build, not one per
  /// V-cycle). Explicitly HostSpace so this compiles on a device backend as well as on a host one.
  Kokkos::View<double*, Kokkos::HostSpace> srcMirror_, dstMirror_, tbMirror_, txMirror_;
};

}  // namespace peclet::amr

#endif  // PECLET_AMR_MG_STAGE_HPP
