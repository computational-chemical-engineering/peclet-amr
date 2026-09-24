// The SIBLING-MERGE and REPARTITION stages of the pressure multigrid — docs/amr_mg_depth.md §5.6
// and work order WO4b; docs/amr_mg_core_boundary.md §8 row S3 and §11.6 gate G-B5.
//
// Where the in-place ladder blocks, core's `chooseStageTarget` (with amr's all-axes lift rule as
// the predicate) moves the level onto a decomposition of its own grid on FEWER ranks and the ladder
// continues there as a `DistributedFlowMultigrid` on the stage's sub-communicator
// (`DistributedStage`, distributed_flow_mg.hpp). This test runs the WO4b gate:
//
//   A  a WEIGHTED partition of a 24^3 root grid (the balancer's plain weighted ORB, whose blocks
//      are odd from the start, so nothing lifts in place and the stage fires at LEVEL 0). At np =
//      2, 4, 8 the ladder must be the single-rank one (24 -> 12 -> 6 -> 3, the moved level listed
//      once), fixed V-cycles must reproduce the single-rank ones to <= 1e-13 relative (the stage
//      is a correction at level 0, so the same linear map in a different order of operations —
//      the bottom_dist test's argument), and the MG-PCG iteration count must equal the
//      single-rank one;
//   B  the 12^3 root of the replicated-tail test with the policy on: the level blocks at 6^3,
//      BELOW level 0, where the incoming iterate is zero, so the V-cycle stays BIT-EXACT against
//      the single-rank one (as the replicated tail was).
//
// The stage policy's two economic inputs (`minExtent`, `maxBlockCells`) are set explicitly here:
// both arms run with minExtent in {0, 4}, and maxBlockCells = the largest finest-level block's
// cell count (on these uniform lmax = 0 meshes the leaf count and the fine-cell count coincide).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/amr/distributed_flow_mg.hpp"
#include "peclet/amr/distributed_octree.hpp"
#include "peclet/amr/multigrid.hpp"
#include "peclet/amr/pcg.hpp"
#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;
using DFMG = DistributedFlowMultigrid<3, kBits>;
using Stage = DistributedStage<3, kBits>;

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

/// The GLOBAL cell count of every level of the built ladder, stage levels included and the moved
/// level listed once (it is the same grid on both sides of the stage). Collective on `comm`; the
/// result is replicated. `kinds` receives the stage kinds met on the way down, outermost first.
std::vector<long long> globalLadder(DFMG& mg, MPI_Comm comm, std::vector<std::string>& kinds) {
  std::vector<long long> out;
  for (std::size_t L = 0; L < mg.numInPlaceLevels(); ++L) {
    long long c = (long long)mg.numLeaves(L), g = 0;
    MPI_Allreduce(&c, &g, 1, MPI_LONG_LONG, MPI_SUM, comm);
    out.push_back(g);
  }
  if (!mg.hasStage())
    return out;
  const std::string kind = mg.stageKind();
  kinds.push_back(kind);
  std::vector<long long> below;
  std::vector<std::string> belowKinds;
  if (kind == "replicated") {
    for (std::size_t L = 0; L < mg.numStageLevels(); ++L)
      below.push_back((long long)mg.stageLeaves(L));  // global on every rank
  } else {
    auto& st = dynamic_cast<Stage&>(mg.stage());
    if (st.active())
      below = globalLadder(st.multigrid(), st.targetComm(), belowKinds);
    long long nb = (long long)below.size(), nk = (long long)belowKinds.size();
    MPI_Bcast(&nb, 1, MPI_LONG_LONG, 0, comm);  // parent rank 0 always owns target block 0
    MPI_Bcast(&nk, 1, MPI_LONG_LONG, 0, comm);
    below.resize((std::size_t)nb);
    MPI_Bcast(below.data(), (int)nb, MPI_LONG_LONG, 0, comm);
    belowKinds.resize((std::size_t)nk);
    for (auto& s : belowKinds) {
      int len = (int)s.size();
      MPI_Bcast(&len, 1, MPI_INT, 0, comm);
      s.resize((std::size_t)len);
      MPI_Bcast(s.data(), len, MPI_CHAR, 0, comm);
    }
  }
  for (std::size_t k = 0; k < below.size(); ++k)
    if (!(k == 0 && below[0] == out.back()))
      out.push_back(below[k]);
  kinds.insert(kinds.end(), belowKinds.begin(), belowKinds.end());
  return out;
}

/// Heap-like root weights: most of the load low in z (a settled bed), so the weighted ORB puts
/// its splits at odd planes and the level blocks at once.
std::vector<double> heapWeights(long nr) {
  std::vector<double> w((std::size_t)(nr * nr * nr));
  for (long k = 0; k < nr; ++k)
    for (long j = 0; j < nr; ++j)
      for (long i = 0; i < nr; ++i) {
        const double z = ((double)k + 0.5) / (double)nr, x = ((double)i + 0.5) / (double)nr;
        w[(std::size_t)(i + nr * (j + nr * k))] = 1.0 + 6.0 * std::exp(-z / 0.3) + 0.5 * x;
      }
  return w;
}

/// One arm: `world` on `dec`, the single-rank reference on the same grid, the policy on.
void runArm(const char* name, long nr, const decomp::BlockDecomposer<3>& dec, double radius,
            int minExtent, bool expectBitwise) {
  const double h0 = 1.0 / (double)nr;
  AmrGeometry<3> geo;
  geo.setIsotropic(h0);
  const std::array<bool, 3> per{true, true, true};
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto openFn = [&](const Vec<3>& fc, int) -> double {
    const double dx = fc[0] - 0.5, dy = fc[1] - 0.5, dz = fc[2] - 0.5;
    const double a = 0.5 + (std::sqrt(dx * dx + dy * dy + dz * dz) - radius) / h0;
    return a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
  };

  DO world;
  world.initDecomposed(dec, 0u, geo, per, MPI_COMM_WORLD);
  const Index n = world.local().numLeaves();
  DO self;
  self.init(IVec<3>{nr, nr, nr}, 0u, geo, per, MPI_COMM_SELF);
  const Index ns = self.local().numLeaves();

  DFMG::StagePolicy pol;
  pol.enabled = true;
  pol.minExtent = minExtent;
  pol.maxBlockCells = decomp::largestBlockCells(dec);  // the largest finest-level block
  DFMG dmg;
  dmg.setStagePolicy(pol);
  dmg.build(world, h0, openFn);
  Multigrid<3, kBits> smg;
  smg.build(self.local(), h0, openFn, /*periodic=*/true);

  // (1) the ladder: the single-rank one, level for level.
  std::vector<std::string> kinds;
  const std::vector<long long> lad = globalLadder(dmg, MPI_COMM_WORLD, kinds);
  std::vector<long long> ref;
  for (std::size_t L = 0; L < smg.numLevels(); ++L)
    ref.push_back((long long)smg.numLeaves(L));
  PECLET_AMR_CHECK(lad == ref);

  std::vector<double> bw((std::size_t)dmg.extendedSize(0), 0.0), bs((std::size_t)ns);
  for (Index i = 0; i < n; ++i)
    bw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i)
    bs[(std::size_t)i] = fAt(self.globalCode(i), h0);

  // (2) fixed V-cycles against the single-rank ones (mean removal off on both sides).
  double vrel = 0.0;
  {
    dmg.setRemoveMean(false);
    smg.setRemoveMean(false);
    View<double> bwv = toDevice(bw, "bw");
    Kokkos::deep_copy(dmg.b(0), bwv);
    Kokkos::deep_copy(dmg.x(0), 0.0);
    View<double> bsv = toDevice(bs, "bs");
    Kokkos::deep_copy(smg.b(0), bsv);
    Kokkos::deep_copy(smg.x(0), 0.0);
    for (int c = 0; c < 5; ++c) {
      dmg.vcycle(2, 2, 60, 0.8);
      smg.vcycle(2, 2, 60, 0.8);
    }
    std::vector<double> xw = down(dmg.x(0)), xs = down(smg.x(0));
    double dmax = 0.0, umax = 0.0;
    for (Index i = 0; i < ns; ++i)
      umax = std::max(umax, std::fabs(xs[(std::size_t)i]));
    for (Index i = 0; i < n; ++i) {
      const Index si = self.local().find(world.globalCode(i));
      dmax = std::max(dmax, std::fabs(xw[(std::size_t)i] - xs[(std::size_t)si]));
    }
    double gdmax = 0.0;
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    vrel = gdmax / umax;
    if (expectBitwise || size == 1)
      PECLET_AMR_CHECK(gdmax == 0.0);
    else
      PECLET_AMR_CHECK(vrel <= 1e-13);
  }

  // (3) MG-PCG: the single-rank iteration count, and the solution.
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
  PECLET_AMR_CHECK_EQ((long)rw.iters, (long)rs.iters);
  std::vector<double> hw = down(xw), hs = down(xs);
  double umax = 0.0, dmax = 0.0;
  for (Index i = 0; i < ns; ++i)
    umax = std::max(umax, std::fabs(hs[(std::size_t)i]));
  for (Index i = 0; i < n; ++i) {
    const Index si = self.local().find(world.globalCode(i));
    dmax = std::max(dmax, std::fabs(hw[(std::size_t)i] - hs[(std::size_t)si]));
  }
  double gdmax = 0.0;
  MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (size == 1)
    PECLET_AMR_CHECK(gdmax == 0.0);
  if (rank == 0) {
    std::string ks;
    for (const auto& k : kinds)
      ks += (ks.empty() ? "" : ",") + k;
    std::string ls;
    for (long long c : lad)
      ls += (ls.empty() ? "" : " ") + std::to_string(c);
    std::printf(
        "[%s] np=%d minExtent=%d maxBlockCells=%ld stages {%s} ladder %s | vcycle rel %.3e"
        " | pcg it %d (self %d) sol rel %.3e | bottom %s\n",
        name, size, minExtent, (long)pol.maxBlockCells, ks.c_str(), ls.c_str(), vrel, rw.iters,
        rs.iters, gdmax / umax, dmg.bottomName().c_str());
  }
}

void run() {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int me : {0, 4}) {
    // A: the weighted 24^3 partition (the gate).
    const long nr = 24;
    const decomp::BlockDecomposer<3> heap((std::size_t)size, IVec<3>{nr, nr, nr}, heapWeights(nr));
    runArm("weighted24", nr, heap, 0.3, me, /*expectBitwise=*/false);
    // B: the replicated-tail test's 12^3 root, blocked BELOW level 0 (bit-exact).
    const decomp::BlockDecomposer<3> prop((std::size_t)size, IVec<3>{12, 12, 12});
    runArm("root12", 12, prop, 0.25, me, /*expectBitwise=*/true);
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
