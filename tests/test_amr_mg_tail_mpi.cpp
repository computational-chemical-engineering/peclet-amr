// The REPLICATED STAGE of the pressure multigrid — docs/amr_mg_depth.md §5.6/§6.5, work order WO4.
//
// When the in-place ladder stops with a coarsest global grid still bigger than `bottomExtent`,
// that level MOVES onto a new decomposition of its own grid and the ladder continues there
// (`MgStage`, mg_stage.hpp). Of §5.6's three instantiations only the REPLICATED one exists today:
// the target decomposition is one block on every rank and the movement is an Allgatherv keyed by
// global cell id. It is not a separate solve — it is the continuation of the same V-cycle — so the
// whole distributed cycle must reproduce what a single-rank run of the same mesh does with its own
// deeper ladder.
//
// The case is §6.2's worked row: a 12^3 root grid. At np = 1 the ladder lifts twice in place
// (12 -> 6 -> 3) and there is no tail. At np = 2, 4, 8 the second lift would leave a block at an
// odd root origin, so the in-place ladder stops at 6^3 and the tail carries 6 -> 3. Either way the
// hierarchy below 12^3 is the same arithmetic on the same cells, which is what this asserts:
//
//   (1) the built ladder (in place + tail) equals predictPressureLadder;
//   (2) fixed V-cycles are WORLD == SELF BIT-EXACT with mean removal off — the tail's level 0 is
//       the gathered coarsest in-place level and its operator is that level's operator, so the
//       gather/scatter is a permutation of exact copies;
//   (3) the tail solution is bit-identical on every rank (§11.2's homogeneous-hardware contract):
//       identical kernels on identical data, so the scatter needs no communication;
//   (4) the distributed MG-PCG takes the SAME number of iterations as the single-rank one and
//       lands on the same solution to Krylov tolerance.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/amr/distributed_flow_mg.hpp"
#include "peclet/amr/distributed_octree.hpp"
#include "peclet/amr/mg_predict.hpp"
#include "peclet/amr/multigrid.hpp"
#include "peclet/amr/pcg.hpp"
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/view.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

constexpr long kNr = 12;  // 12^3 roots at lmax 0: even, but 6 is not evenly splittable by the ORB

double sphereSdf(const Vec<3>& p) {
  const double dx = p[0] - 0.5, dy = p[1] - 0.5, dz = p[2] - 0.5;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.25;
}

double fAt(Code gc, double h0) {
  auto o = M::from_code(gc).decode();
  const double cx = ((double)o[0] + 0.5) * h0, cy = ((double)o[1] + 0.5) * h0,
               cz = ((double)o[2] + 0.5) * h0;
  const double k = 2.0 * M_PI;
  return std::sin(k * cx) * std::cos(k * cy) + std::cos(k * cz) * std::sin(k * cx);
}

std::vector<double> down(const View<double>& d) {
  std::vector<double> h(d.extent(0));
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (std::size_t i = 0; i < h.size(); ++i)
    h[i] = m(i);
  return h;
}

/// Order-independent bit hash of a host array (XOR-folded bit patterns + a sum of the raw bits):
/// two arrays hash equal only if every double matches bit for bit.
std::uint64_t bitHash(const std::vector<double>& v) {
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::uint64_t b = 0;
    std::memcpy(&b, &v[i], sizeof(double));
    h ^= b + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  }
  return h;
}

void runPolicy(const PressureStagePolicy& pol) {
  const double h0 = 1.0 / static_cast<double>(kNr);
  AmrGeometry<3> geo;
  geo.setIsotropic(h0);
  const std::array<bool, 3> per{true, true, true};
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto openFn = [&](const Vec<3>& fc, int) -> double {
    const double a = 0.5 + sphereSdf(fc) / h0;
    return a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
  };

  DO world;
  world.init(IVec<3>{kNr, kNr, kNr}, 0u, geo, per, MPI_COMM_WORLD);
  const Index n = world.local().numLeaves();
  DO self;
  self.init(IVec<3>{kNr, kNr, kNr}, 0u, geo, per, MPI_COMM_SELF);
  const Index ns = self.local().numLeaves();

  DistributedFlowMultigrid<3, kBits> dmg;
  dmg.setStagePolicy(pol);
  dmg.build(world, h0, openFn);
  Multigrid<3, kBits> smg;
  smg.build(self.local(), h0, openFn, /*periodic=*/true);

  // (1) the built ladder == the prediction.
  const auto p = predictPressureLadder<3>(IVec<3>{kNr, kNr, kNr}, 0u, size, 4, pol);
  PECLET_AMR_CHECK_EQ((long)dmg.numInPlaceLevels(), (long)p.numInPlace());
  PECLET_AMR_CHECK_EQ((long)dmg.hasStage(), (long)p.hasStage());
  PECLET_AMR_CHECK(dmg.bottomName() == p.bottomName());
  PECLET_AMR_CHECK_EQ((long)(dmg.numInPlaceLevels() + dmg.numStageLevels()), (long)p.levels.size());
  PECLET_AMR_CHECK_EQ((long)smg.numLevels(), 3L);  // 12 -> 6 -> 3, no tail on one rank
  if (size > 1) {
    PECLET_AMR_CHECK(dmg.hasStage());
    PECLET_AMR_CHECK_EQ((long)dmg.numInPlaceLevels(), 2L);  // 12 -> 6, then odd block origins
    PECLET_AMR_CHECK_EQ((long)dmg.numStageLevels(), 2L);    // gathered 6^3, then 3^3
    PECLET_AMR_CHECK_EQ((long)dmg.stageFrom()[0], 6L);
  } else {
    PECLET_AMR_CHECK(!dmg.hasStage());
  }

  std::vector<double> bw((std::size_t)dmg.extendedSize(0), 0.0), bs((std::size_t)ns);
  for (Index i = 0; i < n; ++i)
    bw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i)
    bs[(std::size_t)i] = fAt(self.globalCode(i), h0);

  // (2) fixed V-cycles, WORLD == SELF bit-exact (mean removal off on both sides).
  {
    dmg.setRemoveMean(false);
    smg.setRemoveMean(false);
    View<double> bwv = toDevice(bw, "bw");
    Kokkos::deep_copy(dmg.b(0), bwv);
    Kokkos::deep_copy(dmg.x(0), 0.0);
    View<double> bsv = toDevice(bs, "bs");
    Kokkos::deep_copy(smg.b(0), bsv);
    Kokkos::deep_copy(smg.x(0), 0.0);
    for (int c = 0; c < 3; ++c) {
      dmg.vcycle(2, 2, 60, 0.8);
      smg.vcycle(2, 2, 60, 0.8);
    }
    std::vector<double> xw = down(dmg.x(0)), xs = down(smg.x(0));
    double dmax = 0.0;
    for (Index i = 0; i < n; ++i) {
      const Index si = self.local().find(world.globalCode(i));
      dmax = std::max(dmax, std::fabs(xw[(std::size_t)i] - xs[(std::size_t)si]));
    }
    double gdmax = 0.0;
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    PECLET_AMR_CHECK(gdmax == 0.0);

    // (3) every rank's tail solution is bit-identical.
    if (dmg.hasStage() && std::string(dmg.stageKind()) == "replicated") {
      const std::vector<double> tx = down(dmg.stageSolution());
      PECLET_AMR_CHECK_EQ((long)tx.size(), (long)(6 * 6 * 6));
      long long h = (long long)(bitHash(tx) >> 1), lo = 0, hi = 0;
      MPI_Allreduce(&h, &lo, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&h, &hi, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
      PECLET_AMR_CHECK_EQ(lo, hi);
    }
  }

  // (4) MG-PCG: the same iteration count and the same solution.
  {
    dmg.setRemoveMean(true);
    smg.setRemoveMean(true);
    PCG<3, kBits> dpcg, spcg;
    dpcg.setVcycle(2, 2, 60, 0.8);
    spcg.setVcycle(2, 2, 60, 0.8);
    dpcg.setSingular(true);
    spcg.setSingular(true);
    dpcg.setDistributed([&dmg](View<double> v) { dmg.sync(0, v); },
                        [](double s) {
                          double g = 0.0;
                          MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                          return g;
                        },
                        dmg.extendedSize(0));
    View<double> bwv = toDevice(bw, "bw2");
    View<double> bsv = toDevice(bs, "bs2");
    View<double> xw("xw", (std::size_t)dmg.extendedSize(0)), xs("xs", (std::size_t)ns);
    auto rw = dpcg.solve(dmg, xw, View<const double>(bwv), 200, 1e-10);
    auto rs = spcg.solve(smg, xs, View<const double>(bsv), 200, 1e-10);
    PECLET_AMR_CHECK(rw.res <= 1e-9 * rw.res0);
    PECLET_AMR_CHECK(rs.res <= 1e-9 * rs.res0);
    PECLET_AMR_CHECK_EQ((long)rw.iters, (long)rs.iters);
    std::vector<double> hw = down(xw), hs = down(xs);
    double umax = 0.0;
    for (Index i = 0; i < ns; ++i)
      umax = std::max(umax, std::fabs(hs[(std::size_t)i]));
    double dmax = 0.0;
    for (Index i = 0; i < n; ++i) {
      const Index si = self.local().find(world.globalCode(i));
      dmax = std::max(dmax, std::fabs(hw[(std::size_t)i] - hs[(std::size_t)si]));
    }
    double gdmax = 0.0;
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (size == 1)
      PECLET_AMR_CHECK(gdmax == 0.0);
    else
      PECLET_AMR_CHECK(gdmax <= 1e-7 * umax);
    if (rank == 0)
      std::printf("[tail] np=%d policy %s stage %s levels %zu+%zu pcg it %d (self %d) dmax %.3e\n",
                  size, pol.enabled ? "on" : "off", dmg.stageKind(), dmg.numInPlaceLevels(),
                  dmg.numStageLevels(), rw.iters, rs.iters, gdmax);
  }
}

void run() {
  // The policy OFF is WO4's replicated tail — the reference the core-machinery replicated
  // movement was proved bitwise against; ON is WO4b's sibling-merge / repartition policy
  // (docs/amr_mg_depth.md WO4b). The prediction reads the same policy, so ladder == predict under
  // both.
  PressureStagePolicy off;
  off.enabled = false;
  PressureStagePolicy on;
  on.enabled = true;
  runPolicy(off);
  runPolicy(on);
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
