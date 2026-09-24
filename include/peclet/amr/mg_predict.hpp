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

namespace peclet::amr {

/// Where a predicted level comes from.
enum class MgLevelKind {
  Octree,  ///< a coarsening of the octree itself (at or above the root brick)
  Lifted,  ///< below the root brick, in place: the ORB blocks nest, transfers stay local
  Tail     ///< below the in-place ladder: the gathered grid, solved redundantly on every rank
};

/// What solves the coarsest level.
enum class MgBottomKind {
  Jacobi,  ///< damped Jacobi is exact there (max extent <= bottomExtent)
  Amg      ///< the agglomerated GraphAMG-PCG bottom
};

inline const char* toString(MgLevelKind k) {
  return k == MgLevelKind::Octree ? "octree" : (k == MgLevelKind::Lifted ? "lifted" : "tail");
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
/// `peclet.amr.predict_pressure_hierarchy` hands to Python as a dict. The built ladder is compared
/// against it level by level (`AmrFlow::pressureMgLevels`, `DistributedFlowMultigrid`).
template <int Dim>
struct MgLadderPrediction {
  /// Finest first; tail levels appended. The first tail level is the SAME grid as the last
  /// in-place one (moved, not coarsened), as `AmrFlow::pressureMgLevels` also lists it.
  std::vector<MgLevelPrediction<Dim>> levels;
  bool tail = false;                           ///< the redundant tail engages
  IVec<Dim> tailFrom{};                        ///< the coarsest in-place grid the tail gathers
  MgBottomKind bottom = MgBottomKind::Jacobi;  ///< what solves the coarsest level under `auto`

  /// Levels that keep the ORB (kinds `Octree` and `Lifted`); compare against
  /// `DistributedFlowMultigrid::numInPlaceLevels()`.
  std::size_t numInPlace() const {
    std::size_t k = 0;
    for (const auto& lv : levels)
      if (lv.kind != MgLevelKind::Tail)
        ++k;
    return k;
  }
  /// `"jacobi"` / `"amg"`, with the `"+tail"` suffix when the tail engages (§6.7's spelling).
  std::string bottomName() const { return std::string(toString(bottom)) + (tail ? "+tail" : ""); }
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
                                              Index bottomExtent = 4) {
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

  // --- the in-place lifted levels: lockstep while the grid halves and every block stays even.
  core::decomp::BlockDecomposer<Dim> dec(static_cast<std::size_t>(numRanks), G);
  std::vector<IVec<Dim>> origin(static_cast<std::size_t>(numRanks)),
      brick(static_cast<std::size_t>(numRanks));
  for (int r = 0; r < numRanks; ++r) {
    const auto b = dec.block(static_cast<std::size_t>(r));
    origin[static_cast<std::size_t>(r)] = b.origin;
    brick[static_cast<std::size_t>(r)] = b.size;
  }
  IVec<Dim> cur = G;
  for (;;) {
    if (!detail::gridCanHalve<Dim>(cur, bottomExtent))
      break;  // grid-limited
    bool blocksEven = true;
    for (int r = 0; r < numRanks && blocksEven; ++r)
      for (int d = 0; d < Dim; ++d)
        if ((brick[static_cast<std::size_t>(r)][d] % 2) != 0 ||
            (origin[static_cast<std::size_t>(r)][d] % 2) != 0) {
          blocksEven = false;  // decomposition-limited (the Allreduce(MIN) of §6.2)
          break;
        }
    if (!blocksEven)
      break;
    for (int d = 0; d < Dim; ++d)
      cur[d] /= 2;
    for (int r = 0; r < numRanks; ++r)
      for (int d = 0; d < Dim; ++d) {
        brick[static_cast<std::size_t>(r)][d] /= 2;
        origin[static_cast<std::size_t>(r)][d] /= 2;
      }
    detail::pushLevel<Dim>(p, cur, MgLevelKind::Lifted);
  }

  // --- the redundant tail: the gathered coarsest in-place grid, continued on one rank.
  if (numRanks > 1 && detail::maxExtent<Dim>(cur) > bottomExtent) {
    p.tail = true;
    p.tailFrom = cur;
    detail::pushLevel<Dim>(p, cur,
                           MgLevelKind::Tail);  // the tail's own level 0 == the gathered grid
    while (detail::gridCanHalve<Dim>(cur, bottomExtent)) {
      for (int d = 0; d < Dim; ++d)
        cur[d] /= 2;
      detail::pushLevel<Dim>(p, cur, MgLevelKind::Tail);
    }
  }

  p.bottom = detail::maxExtent<Dim>(cur) > bottomExtent ? MgBottomKind::Amg : MgBottomKind::Jacobi;
  return p;
}

}  // namespace peclet::amr

#endif  // PECLET_AMR_MG_PREDICT_HPP
