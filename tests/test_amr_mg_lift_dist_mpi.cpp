// The DISTRIBUTED pressure-multigrid ladder below the root brick — docs/amr_mg_depth.md §6.2/§6.4,
// work order WO3. The lift is LOCKSTEP: the depth is one Allreduce(MIN) of the depth each rank's
// own ORB block allows, so every rank builds the same levels and the per-level halo collectives
// keep pairing up. `predictPressureLadder` (§6.7) is the specification — it re-runs the rule on the
// ORB without building a mesh — and this test asserts the ladder DistributedFlowMultigrid actually
// builds against it, at np = 1, 2, 4, 8, on uniform meshes where the prediction is exact:
//
//   * 8^3 root, lmax 0   — one lift (8 -> 4) at every rank count here;
//   * 8^3 root, lmax 1   — one octree coarsening (16 -> 8) then the same lift;
//   * 12^3 root, lmax 0  — the DECOMPOSITION-limited stop: np = 1 lifts twice (12 -> 6 -> 3), but
//                          at np = 2, 4, 8 the second lift would leave a block at an odd root
//                          origin, so every rank stops after one (the tail of §6.5 / WO4 is what
//                          continues from there; `numInPlaceLevels()` is what this test compares).
//
// It also asserts the level counts are equal on every rank and that a lifted level's global leaf
// count is the predicted global cell count — i.e. the lift really merged every octet everywhere.
#include <array>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/amr/distributed_flow_mg.hpp"
#include "peclet/amr/distributed_octree.hpp"
#include "peclet/amr/mg_predict.hpp"
#include "peclet/core/common/mpi.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;

// One uniform case: `root` root cells per axis at depth `lmax`, refined everywhere so the mesh
// supports exactly `lmax` octree coarsenings (the predictor's convention).
void oneCase(long root, unsigned lmax, int size) {
  const double h0 = 1.0 / static_cast<double>(root * (1L << lmax));
  AmrGeometry<3> geo;
  geo.setIsotropic(h0);
  const std::array<bool, 3> per{true, true, true};

  DO d;
  d.init(IVec<3>{root, root, root}, lmax, geo, per, MPI_COMM_WORLD);
  // init() leaves every cell at the ROOT level, so refine everywhere down to level 0: the mesh
  // then supports exactly `lmax` octree coarsenings, which is the predictor's convention.
  for (unsigned k = 0; k < lmax; ++k)
    d.local().refineIf([](typename DO::Code, unsigned lvl) { return lvl > 0; });
  d.balance();
  auto openFn = [](const Vec<3>&, int) { return 1.0; };
  DistributedFlowMultigrid<3, kBits> mg;
  mg.build(d, h0, openFn);

  const auto p = predictPressureLadder<3>(IVec<3>{root, root, root}, lmax, size);

  // Level counts: equal to the prediction's in-place count, and equal on every rank.
  PECLET_AMR_CHECK_EQ((long)mg.numInPlaceLevels(), (long)p.numInPlace());
  long nl = (long)mg.numLevels(), lo = 0, hi = 0;
  MPI_Allreduce(&nl, &lo, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&nl, &hi, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  PECLET_AMR_CHECK_EQ(lo, hi);

  // Per level: the global leaf count is the predicted global cell count (the lift merged every
  // octet on every rank — a partial merge anywhere would show up here).
  for (std::size_t L = 0; L < mg.numInPlaceLevels(); ++L) {
    long n = (long)mg.numLeaves(L), g = 0;
    MPI_Allreduce(&n, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    PECLET_AMR_CHECK_EQ(g, (long)p.levels[L].cells);
  }
}

void run() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  oneCase(8, 0, size);   // one lift: 8 -> 4
  oneCase(8, 1, size);   // one octree coarsening (16 -> 8) then the same lift
  oneCase(12, 0, size);  // decomposition-limited: 12 -> 6 -> 3 at np = 1, 12 -> 6 otherwise

  // The decomposition-limited stop is the point of the 12^3 case — assert it is really what
  // happened rather than trusting the prediction to be non-trivial.
  const auto p12 = predictPressureLadder<3>(IVec<3>{12, 12, 12}, 0u, size);
  if (size == 1) {
    PECLET_AMR_CHECK_EQ((long)p12.numInPlace(), 3L);  // 12 -> 6 -> 3, no tail
    PECLET_AMR_CHECK(!p12.hasStage());
  } else {
    PECLET_AMR_CHECK_EQ((long)p12.numInPlace(), 2L);  // 12 -> 6, then an odd block origin
    PECLET_AMR_CHECK(p12.hasStage());
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int fails = peclet::amr::test::g_failures, total = 0;
  MPI_Reduce(&fails, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (rank == 0) {
    if (total == 0) {
      std::printf("OK\n");
      return 0;
    }
    std::fprintf(stderr, "%d failure(s)\n", total);
    return 1;
  }
  return 0;
}
