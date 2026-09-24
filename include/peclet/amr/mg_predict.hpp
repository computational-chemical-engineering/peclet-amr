// amr — what pressure-multigrid ladder a given grid, tree depth and rank count will produce.
//
// The prediction tool of docs/amr_mg_depth.md §6.7. It is a PURE HOST FUNCTION (no MPI, no Kokkos,
// no octree): it re-runs the §6.2 ladder rule on the ORB the distributed octree would build
// (peclet::core::decomp::BlockDecomposer on the global ROOT grid), so a test can assert the
// hierarchy the solver built against the hierarchy the design says it should build, instead of
// against a literal that goes stale.
//
// The rule, once:
//   * levels above the root brick are the octree's own coarsenings — `lmax + 1` of them for a mesh
//     refined to level 0 (the case §6.2's table describes; see the `lmax` note on the function);
//   * below the root the ladder LIFTS while the global grid halves into a cube level AND every
//     rank's block stays even in origin and size (§6.2); the lift is lockstep, so one rank's odd
//     block stops the ladder for all;
//   * when the coarsest in-place grid is still larger than `bottomExtent` on some axis and there
//     is more than one rank, the tail gathers it and continues redundantly with the single-rank
//     ladder (§6.5);
//   * the bottom is exact damped Jacobi while the final extent is <= `bottomExtent`, and the
//     agglomerated GraphAMG-PCG solve otherwise (§6.6).
//
// The prediction was written before the tail (WO4) and the GraphAMG bottom (WO5) were built, as
// the ladder of the finished design; both have since landed and match it, which is what makes it
// the specification the built ladder is checked against (`amr_mg_predict`, `amr_mg_tail`,
// `amr_mg_bottom{,_dist}`). It predicts the default bottom selection, `auto`, on the ORB a
// freshly initialised DistributedOctree builds — not a weighted or rebalanced partition.
#ifndef PECLET_AMR_MG_PREDICT_HPP
#define PECLET_AMR_MG_PREDICT_HPP

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "peclet/amr/common.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/decomp/stage_target.hpp"

namespace peclet::amr {

/// Which stage the distributed pressure ladder uses where its in-place lift blocks
/// (docs/amr_mg_depth.md §5.6, WO4b; docs/amr_mg_core_boundary.md §11.1, §9.6, §9.7). With
/// `enabled` — the DEFAULT since 2026-09-24, on timings that measured it no slower anywhere and up
/// to 1.9x faster (amr_mg_depth.md WO4b) — core's `chooseStageTarget` chooses between a sibling
/// merge, a repartition and the replicated tail. With `enabled == false` every stage is the
/// replicated tail (WO4), kept as the reference and an ablation.
/// `DistributedFlowMultigrid::StagePolicy` is this type, and `predictPressureLadder` takes it, so
/// the prediction and the built ladder read the same policy.
struct PressureStagePolicy {
  bool enabled = true;  ///< false: the replicated tail only (WO4)
  /// Core's economic trigger / extent cap. 4, flow's trigger verbatim (§9.7) — INERT here, because
  /// the policy is consulted only where the §6.2 lift has already stopped.
  int minExtent = 4;
  /// Core's bound on a target block's cells. NEGATIVE (the default) = derive it: the LEAF count of
  /// the largest finest-level block (§9.6 — the rank's real unknowns, not its fine-cell count). 0 =
  /// no repartition (flow's byte-identical mode). A stage's continued ladder inherits the resolved
  /// value, since its own level 0 is not the finest level.
  Index maxBlockCells = -1;
};

/// amr's lift rule (docs/amr_mg_depth.md §6.2) as core's stage predicate: the level grid halves
/// into a cube level with at least two cells per axis, and every block is even in origin and size
/// on EVERY axis. Pure; replicated whenever `dec` is.
template <int Dim>
bool amrStageLiftable(const core::decomp::BlockDecomposer<Dim>& dec) {
  const IVec<Dim>& G = dec.globalSize();
  for (int a = 0; a < Dim; ++a)
    if ((G[a] % 2) != 0 || (G[a] / 2) < 2)
      return false;
  for (std::size_t b = 0; b < dec.numBlocks(); ++b)
    for (int a = 0; a < Dim; ++a)
      if ((dec.origins()[b][a] % 2) != 0 || (dec.sizes()[b][a] % 2) != 0)
        return false;
  return true;
}

/// Where a predicted level comes from.
enum class MgLevelKind {
  Octree,       ///< a coarsening of the octree itself (at or above the root brick)
  Lifted,       ///< below the root brick, in place: the ORB blocks nest, transfers stay local
  Tail,         ///< below the in-place ladder: the gathered grid, solved redundantly on every rank
  Sibling,      ///< the continued ladder of a sibling-merge stage (fewer ranks, `agglomerated(d)`)
  Repartition,  ///< the continued ladder of a repartition stage (a fresh ORB on np_L ranks)
};

/// What solves the coarsest level.
enum class MgBottomKind {
  Jacobi,  ///< damped Jacobi is exact there (max extent <= bottomExtent)
  Amg      ///< the agglomerated GraphAMG-PCG bottom
};

inline const char* toString(MgLevelKind k) {
  switch (k) {
    case MgLevelKind::Octree:
      return "octree";
    case MgLevelKind::Lifted:
      return "lifted";
    case MgLevelKind::Tail:
      return "tail";
    case MgLevelKind::Sibling:
      return "sibling";
    case MgLevelKind::Repartition:
      return "repartition";
  }
  return "?";
}
inline const char* toString(MgBottomKind k) {
  return k == MgBottomKind::Jacobi ? "jacobi" : "amg";
}

/// One predicted level of the pressure ladder (docs/amr_mg_depth.md §6.7).
template <int Dim>
struct MgLevelPrediction {
  IVec<Dim> extent{};   ///< global grid extent in THIS level's cell units
  long long cells = 0;  ///< product of `extent` — the global cell count of a UNIFORM mesh
  MgLevelKind kind = MgLevelKind::Octree;  ///< where the level comes from
};

/// The whole predicted ladder: what `predictPressureLadder` returns and what
/// `peclet.amr.predict_hierarchy` hands to Python as a dict. The built ladder is compared
/// against it level by level (`AmrFlow::pressureMgLevels`, `DistributedFlowMultigrid`).
template <int Dim>
struct MgLadderPrediction {
  /// Finest first; each stage's levels appended after the ladder it continues. A stage's first
  /// level is the SAME grid as the last level above it (moved, not coarsened), as
  /// `AmrFlow::pressureMgLevels` also lists it; its kind is the stage's.
  std::vector<MgLevelPrediction<Dim>> levels;
  /// The stages met on the way down, outermost first (`Tail`, `Sibling`, `Repartition`).
  std::vector<MgLevelKind> stages;
  bool tail = false;                           ///< a replicated tail engages (at any depth)
  IVec<Dim> tailFrom{};                        ///< the grid the (first) stage moves
  MgBottomKind bottom = MgBottomKind::Jacobi;  ///< what solves the coarsest level under `auto`

  /// Whether any stage fires — compare against `DistributedFlowMultigrid::hasStage()`.
  bool hasStage() const { return !stages.empty(); }
  /// The levels ABOVE the first stage (kinds `Octree` and `Lifted`); compare against
  /// `DistributedFlowMultigrid::numInPlaceLevels()`.
  std::size_t numInPlace() const {
    std::size_t k = 0;
    while (k < levels.size() &&
           (levels[k].kind == MgLevelKind::Octree || levels[k].kind == MgLevelKind::Lifted))
      ++k;
    return k;
  }
  /// `"jacobi"` / `"amg"`, suffixed by every stage innermost first (§6.7's spelling:
  /// `"+tail"`, `"+sibling"`, `"+repartition"`) — what `DistributedFlowMultigrid::bottomName`
  /// reports.
  std::string bottomName() const {
    std::string s = toString(bottom);
    for (auto it = stages.rbegin(); it != stages.rend(); ++it)
      s += std::string("+") + toString(*it);
    return s;
  }
};

namespace detail {

template <int Dim>
Index maxExtent(const IVec<Dim>& G) {
  Index mx = 0;
  for (int d = 0; d < Dim; ++d)
    mx = std::max(mx, G[d]);
  return mx;
}

/// The grid half of the §6.2 rule: stop when the bottom smoother is already exact, and when the
/// grid cannot halve into a cube level with at least two cells per axis.
template <int Dim>
bool gridCanHalve(const IVec<Dim>& G, Index bottomExtent) {
  if (maxExtent<Dim>(G) <= bottomExtent)
    return false;
  for (int d = 0; d < Dim; ++d)
    if ((G[d] % 2) != 0 || (G[d] / 2) < 2)
      return false;
  return true;
}

template <int Dim>
long long cellsOf(const IVec<Dim>& G) {
  long long c = 1;
  for (int d = 0; d < Dim; ++d)
    c *= static_cast<long long>(G[d]);
  return c;
}

template <int Dim>
void pushLevel(MgLadderPrediction<Dim>& p, const IVec<Dim>& G, MgLevelKind kind) {
  MgLevelPrediction<Dim> lv;
  lv.extent = G;
  lv.cells = cellsOf<Dim>(G);
  lv.kind = kind;
  p.levels.push_back(lv);
}

}  // namespace detail

/// Predict the pressure-multigrid ladder for a global ROOT grid `G` (in root cells, i.e. the
/// `globalRootSize` `DistributedOctree::init` takes — the finest grid divided by `2^lmax`), a tree
/// depth `lmax`, and `numRanks` ranks.
///
/// `lmax` is taken to be the number of octree coarsenings the mesh supports, which is the tree's
/// `lmax` for a mesh that is refined to level 0 somewhere (every graded mesh, and the uniform mesh
/// of `Octree(cells, lmax=0)`). An UNREFINED `Octree(cells, lmax=k>0)` is the same mesh as
/// `Octree(cells / 2^k, lmax=0)` and must be predicted as such — its leaves are all root cells, so
/// its octree ladder is one level, not `k + 1`.
///
/// LOCAL and pure: no MPI, no Kokkos, no mesh; any `numRanks` may be asked for on one process.
/// `numRanks < 1` is treated as 1. The ORB is `BlockDecomposer<Dim>(numRanks, G)`, the partition
/// `DistributedOctree::init` builds without weights; a weighted (`rebalance`d) partition may nest
/// less deeply than predicted. `bottomExtent` is `AmrFlow::pressureBottomExtent()` (default 4).
///
/// @pre every `G[d] >= 1` and `bottomExtent >= 1` (the Python binding checks both). Does not
/// throw.
template <int Dim>
MgLadderPrediction<Dim> predictPressureLadder(IVec<Dim> G, unsigned lmax, int numRanks,
                                              Index bottomExtent = 4,
                                              const PressureStagePolicy& policy = {}) {
  MgLadderPrediction<Dim> p;
  if (numRanks < 1)
    numRanks = 1;

  // --- the octree levels: the finest grid coarsened down to the root brick.
  for (unsigned j = 0; j <= lmax; ++j) {
    IVec<Dim> e{};
    for (int d = 0; d < Dim; ++d)
      e[d] = G[d] << (lmax - j);
    detail::pushLevel<Dim>(p, e, MgLevelKind::Octree);
  }

  // --- the in-place lifted levels: lockstep while the grid halves and every block stays even —
  //     then, where they block above `bottomExtent`, a STAGE, whose continued ladder lifts again on
  //     its own decomposition and may stage again (the same loop, on the target decomposition).
  namespace cd = core::decomp;
  cd::BlockDecomposer<Dim> dec(static_cast<std::size_t>(numRanks), G);
  Index maxBlockCells = policy.maxBlockCells;
  if (maxBlockCells < 0) {  // the leaf count of the largest finest block, uniform mesh (§9.6)
    maxBlockCells = static_cast<Index>(cd::largestBlockCells(dec));
    for (unsigned j = 0; j < lmax; ++j)
      maxBlockCells <<= Dim;
  }
  IVec<Dim> cur = G;
  MgLevelKind liftKind = MgLevelKind::Lifted;
  for (;;) {
    while (detail::gridCanHalve<Dim>(cur, bottomExtent) && amrStageLiftable<Dim>(dec)) {
      IVec<Dim> two{};
      for (int d = 0; d < Dim; ++d)
        two[d] = 2;
      dec = dec.coarsened(two);
      cur = dec.globalSize();
      detail::pushLevel<Dim>(p, cur, liftKind);
    }
    if (dec.numBlocks() <= 1 || detail::maxExtent<Dim>(cur) <= bottomExtent)
      break;  // one rank: its bottom solves the level whole; or the Jacobi bottom is exact here
    if (!p.hasStage())
      p.tailFrom = cur;
    if (policy.enabled) {
      const cd::StageTarget<Dim> t = cd::chooseStageTarget(
          dec, cur, [](const cd::BlockDecomposer<Dim>& c) { return amrStageLiftable<Dim>(c); },
          policy.minExtent, maxBlockCells);
      if (t.kind == cd::StageKind::SiblingMerge || t.kind == cd::StageKind::Repartition) {
        liftKind =
            t.kind == cd::StageKind::SiblingMerge ? MgLevelKind::Sibling : MgLevelKind::Repartition;
        p.stages.push_back(liftKind);
        detail::pushLevel<Dim>(p, cur, liftKind);  // the continued ladder's level 0: moved
        dec = t.dec;
        continue;
      }
    }
    // --- the redundant tail: the gathered grid, continued on one rank.
    p.tail = true;
    p.stages.push_back(MgLevelKind::Tail);
    detail::pushLevel<Dim>(p, cur,
                           MgLevelKind::Tail);  // the tail's own level 0 == the gathered grid
    while (detail::gridCanHalve<Dim>(cur, bottomExtent)) {
      for (int d = 0; d < Dim; ++d)
        cur[d] /= 2;
      detail::pushLevel<Dim>(p, cur, MgLevelKind::Tail);
    }
    break;
  }

  p.bottom = detail::maxExtent<Dim>(cur) > bottomExtent ? MgBottomKind::Amg : MgBottomKind::Jacobi;
  return p;
}

}  // namespace peclet::amr

#endif  // PECLET_AMR_MG_PREDICT_HPP
