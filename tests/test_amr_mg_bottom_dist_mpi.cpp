// The exact coarsest-level solve INSIDE the replicated stage — docs/amr_mg_depth.md §6.6 with
// §6.5, work order WO5's last acceptance line.
//
// A 10^3 root grid is the case where both halves of the design are needed at once. At np = 1 the
// ladder lifts once (10 -> 5, and 5 is odd so it stops there) and 5 > bottomExtent, so the
// agglomerated GraphAMG-PCG bottom runs on 125 cells. At np = 2, 4, 8 the ORB blocks are odd from
// the start, so NOTHING lifts in place: the single level is handed to the replicated stage, whose
// continued ladder is the same 10 -> 5 with the same exact bottom, replicated on every rank.
//
// The two must therefore agree, which is what this asserts at np = 1, 2, 4, 8: the ladder matches
// `predictPressureLadder`, the bottom reports "amg", fixed V-cycles reproduce the single-rank ones
// with mean removal off, and the MG-PCG takes the same iteration count and lands on the same
// solution (§9's acceptance: <= 1e-12).
//
// WHY THE V-CYCLE IS BIT-EXACT AT np = 1 AND ONLY 1e-12 ABOVE IT — and why that is not a defect.
// A stage moves the RESIDUAL and brings back a CORRECTION (§6.5), so where it fires the cycle is
// `x += V(0, b − Lx)`. That is algebraically the same map as the single-rank `V(x, b)` — a V-cycle
// IS `x + M^-1(b − Lx)` for a fixed linear M^-1, which is the whole reason it may precondition a
// Krylov method — but it is a different ORDER of the same floating-point operations. At np >= 2 on
// this grid nothing lifts in place, so the stage fires at LEVEL 0 and the reassociation is visible
// from the second V-cycle on (measured 4.8e-16 relative). At np = 1 there is no stage at all and
// the agreement is exact; in tests/test_amr_mg_tail_mpi.cpp the stage fires BELOW level 0, where
// the incoming iterate is zero, residual == rhs and add == overwrite, so it stays bit-exact there.
#include <array>
#include <cmath>
#include <cstdio>
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
constexpr long kNr = 10;  // 10^3 roots: 10 -> 5 and there it stops, 5 > bottomExtent = 4

double sphereSdf(const Vec<3>& p) {
  const double dx = p[0] - 0.5, dy = p[1] - 0.5, dz = p[2] - 0.5;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.28;
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

void runPolicy(const PressureStagePolicy& pol) {
  const double h0 = 1.0 / (double)kNr;
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

  const auto p = predictPressureLadder<3>(IVec<3>{kNr, kNr, kNr}, 0u, size, 4, pol);
  PECLET_AMR_CHECK_EQ((long)dmg.numInPlaceLevels(), (long)p.numInPlace());
  PECLET_AMR_CHECK_EQ((long)dmg.hasStage(), (long)p.hasStage());
  PECLET_AMR_CHECK(dmg.bottomName() == p.bottomName());
  PECLET_AMR_CHECK(p.bottomName().substr(0, 3) == std::string("amg"));  // predict says amg
  PECLET_AMR_CHECK(dmg.bottomName().substr(0, 3) == std::string("amg"));
  PECLET_AMR_CHECK(smg.bottomName() == std::string("amg"));
  PECLET_AMR_CHECK_EQ((long)smg.bottomSize(), 125L);
  if (size > 1) {
    PECLET_AMR_CHECK_EQ((long)dmg.numInPlaceLevels(), 1L);  // odd ORB blocks: nothing lifts
    PECLET_AMR_CHECK_EQ((long)dmg.numStageLevels(), 2L);    // the gathered 10^3, then 5^3
    // policy off: the replicated tail; on: a repartition onto one rank (no liftable merge fits
    // one finest block, and no proportional ORB on more ranks lifts a 10^3 grid).
    PECLET_AMR_CHECK(dmg.bottomName() == std::string(pol.enabled ? "amg+repartition" : "amg+tail"));
  }

  std::vector<double> bw((std::size_t)dmg.extendedSize(0), 0.0), bs((std::size_t)ns);
  for (Index i = 0; i < n; ++i)
    bw[(std::size_t)i] = fAt(world.globalCode(i), h0);
  for (Index i = 0; i < ns; ++i)
    bs[(std::size_t)i] = fAt(self.globalCode(i), h0);

  // Fixed V-cycles, WORLD == SELF bit-exact (mean removal off on both sides). The exact bottom is
  // a deterministic host CG on identical data, so it does not loosen this.
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
    double gdmax = 0.0, umax = 0.0;
    for (Index i = 0; i < ns; ++i)
      umax = std::max(umax, std::fabs(xs[(std::size_t)i]));
    MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (size == 1)
      PECLET_AMR_CHECK(gdmax == 0.0);
    else
      PECLET_AMR_CHECK(gdmax <= 1e-12 * umax);  // §9's number; the header says why not bitwise
    if (rank == 0)
      std::printf("[bottom-dist] np=%d policy %s vcycle dmax/|x| %.3e\n", size,
                  pol.enabled ? "on" : "off", gdmax / umax);
  }

  // MG-PCG: the same iteration count, the same solution (<= 1e-12 of §9).
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
    PECLET_AMR_CHECK(gdmax <= 1e-12 * std::max(umax, 1e-300));
    if (rank == 0)
      std::printf(
          "[bottom-dist] np=%d levels %zu+%zu bottom %s pcg it %d (self %d) dmax/|u| %.3e\n", size,
          dmg.numInPlaceLevels(), dmg.numStageLevels(), dmg.bottomName().c_str(), rw.iters,
          rs.iters, gdmax / umax);
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
