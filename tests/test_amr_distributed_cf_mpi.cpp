// The distributed C/F QUADRATIC scheme (include/peclet/amr/cf_scheme.hpp, AmrFlow::setCfScheme):
// the Martin-Cartwright tangential closure across a block seam.
//
// What this gate exists to catch. The closure substitutes, for the coarse side of a 2:1 sub-face,
// an interpolation of the coarse cell at the fine cell's tangential position. That needs the
// COARSE cell's tangential +- neighbours and, when such a neighbour is FINER, the 2^Dim children
// covering the coarse-size region. Both reaches leave the rank's block at a seam. The tangential
// neighbour has always gone through AmrPoisson::periodicNeighbor -> probeSlot -> the LeafHalo
// resolver; the child enumeration used BlockOctree::find directly until 2026-09-21, on a
// coordinate it had wrapped modulo `t.brick() * 2^lmax` by hand. Single-rank that wrap is the
// domain period and everything is right. Multi-rank the block is NOT the domain, so the wrap
// folds a child that belongs to a neighbouring rank back into THIS rank's block and `find`
// returns a real but geometrically unrelated leaf: either the cover guard rejects it on level or
// fluid and the tangential side drops silently to the RAW coarse value, or it passes and the
// stencil reads a cell on the far side of the block. No crash, no missing ghost, no error
// message - just a decomposition-dependent wrong answer at the seams. A test that only asserts
// "it runs" cannot see it; measured below, it is a 131 % error.
//
// So the comparison has to be WORLD (initMpi) vs SELF (single-rank on the same octree), leaf by
// leaf through the global Morton code, with 2:1 interfaces that STRADDLE the block seams:
//   np = 1 : BITWISE. Every wrapped probe lands back in the block, zero ghosts exist, and the two
//            paths must execute identical arithmetic.
//   np > 1 : the established decomposition-independence class (tests/test_amr_distributed_seam_mpi
//            .cpp), plus the positive checks that ghosts exist AND that the C/F overlays actually
//            read them (numCfGhostColumns) - without the second, a fallback-everywhere build
//            would pass the first.
//
// Advection is OFF on purpose: that is the configuration in which nothing else refreshes u's
// ghost tail between the projection and the next predictor, so it also gates the syncVel that the
// momentum C/F delta (cfMom_, which reads u at the coarse cell's tangential neighbours) needs.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/amr/distributed_octree.hpp"
#include "peclet/amr/flow.hpp"
#include "peclet/core/common/types.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using DO = DistributedOctree<3, kBits>;
using M = DO::M;
using Code = DO::Code;

constexpr long kNr = 4;        // 4^3 roots
constexpr unsigned kLmax = 3;  // -> 32^3 fine, periodic [0,1)^3
constexpr double kH0 = 1.0 / static_cast<double>(kNr * (1 << kLmax));

// The centre is deliberately OFF-CENTRE, and that is load-bearing. With the sphere at
// (0.5, 0.5, 0.5) the mesh is mirror-symmetric about every plane the ORB cuts the box with, so a
// coarse cell's tangential neighbour ACROSS a seam is always its own mirror image and therefore
// always the same level - the finer-neighbour child enumeration then never crosses a seam and the
// bug this test exists for is invisible. Measured at np = 2: centred, 0 child probes leave the
// block; at (0.41, 0.57, 0.5), 13800 do and every one resolves to a ghost slot. The negative
// control (child enumeration back on BlockOctree::find) is bitwise-clean at np = 1 and 131 %
// wrong at np = 2 and 4 on this geometry.
double sphereSdf(const Vec<3>& p) {
  const double dx = p[0] - 0.41, dy = p[1] - 0.57, dz = p[2] - 0.5;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.22;
}

/// A self-similar graded ladder around the sphere: a finest band wide enough for the (2,2) ghost
/// closure, then two coarser shells. Every level jump is a closed surface centred on the sphere,
/// so it crosses whatever plane the ORB cuts the box with at np = 2, 4 and 8 - which is the whole
/// point (a mesh whose 2:1 interfaces avoided the seams could not detect the bug above).
unsigned targetLevel(const Vec<3>& ctr) {
  const double d = std::fabs(sphereSdf(ctr));
  if (d < 3.5 * kH0)
    return 0;
  if (d < 7.0 * kH0)
    return 1;
  if (d < 14.0 * kH0)
    return 2;
  return 3;
}

void makeGradedMesh(DO& d) {
  for (unsigned pass = 0; pass < kLmax; ++pass) {
    d.local().refineIf([&](Code cd, unsigned lvl) -> bool {
      if (lvl == 0)
        return false;
      auto o = M::from_code(cd).decode();
      const double s = static_cast<double>(1u << lvl);
      Vec<3> ctr{};
      for (int a = 0; a < 3; ++a)
        ctr[a] = (static_cast<double>((long)o[a] + d.blockFineOrigin()[a]) + 0.5 * s) * kH0;
      return lvl > targetLevel(ctr);
    });
  }
  d.balance();
}

struct Fields {
  std::array<std::vector<double>, 3> u;
  std::vector<double> p;
};

Fields runSteps(AmrFlow<kBits>& f, int steps) {
  for (int s = 0; s < steps; ++s)
    f.step(200, 60);
  Fields r;
  for (int c = 0; c < 3; ++c)
    r.u[(std::size_t)c] = f.velocity(c);
  r.p = f.pressure();
  return r;
}

/// CF_STANDARD=1 rebuilds the same mesh on the STANDARD scheme. It gates nothing (the standard
/// scheme has its own distributed tests); it is the control that isolates what the C/F discovery
/// arm costs - the same geometry, the same fixpoint, the arm switched off. Measured at np = 2,
/// 5902 leaves, OMP_NUM_THREADS=1: the fixpoint takes 4 rounds EITHER WAY (the arm's probes
/// resolve inside the rounds the existing probers already need), the arm adds 10 of 911 ghosts on
/// one rank and none on the other, and setSolid on the slowest rank goes 0.094 s -> 0.195 s, of
/// which ~58 ms is the C/F overlay CSR build itself (intrinsic to the scheme, single-rank too)
/// and ~45 ms is the discovery arm spread over the four rounds.
bool standardControl() {
  const char* e = std::getenv("CF_STANDARD");
  return e && e[0] == '1';
}

void configure(AmrFlow<kBits>& f) {
  f.setDensity(1.0);
  f.setViscosity(1.0);
  f.setBodyForce(1.0, 0.0, 0.0);
  f.setDt(1e6);
  f.setGhostProjection(true, 2, 2);
  if (!standardControl())
    f.setCfScheme(static_cast<int>(CfScheme::quadratic));
  f.setSolid(sphereSdf);
}

void run() {
  AmrGeometry<3> geo;
  geo.setIsotropic(kH0);
  const std::array<bool, 3> per{true, true, true};
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  DO world;
  world.init(IVec<3>{kNr, kNr, kNr}, kLmax, geo, per, MPI_COMM_WORLD);
  makeGradedMesh(world);
  const Index n = world.local().numLeaves();
  DO self;
  self.init(IVec<3>{kNr, kNr, kNr}, kLmax, geo, per, MPI_COMM_SELF);
  makeGradedMesh(self);

  AmrFlow<kBits> fw;
  fw.initMpi(world);
  const double t0 = MPI_Wtime();
  configure(fw);
  const double tSetup = MPI_Wtime() - t0;
  {
    double tmax = 0.0;
    long nl = (long)n, ntot = 0, cfg = (long)fw.numCfGhostColumns(), cfgTot = 0;
    MPI_Allreduce(&tSetup, &tmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&nl, &ntot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&cfg, &cfgTot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf(
          "[cf-mpi] setup np=%d: %ld leaves total, slowest rank %.3f s = %.1f us/leaf; "
          "%ld C/F overlay entries read a ghost slot\n",
          size, ntot, tmax, 1e6 * tmax / ((double)ntot / size), cfgTot);
  }
  if (size > 1 && !standardControl()) {
    PECLET_AMR_CHECK(fw.numGhostCells() > 0);
    long cfg = (long)fw.numCfGhostColumns(), cfgTot = 0;
    MPI_Allreduce(&cfg, &cfgTot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    // The quadratic closure must REACH ACROSS the seam. Zero here means every seam-adjacent
    // tangential sample fell back to the raw coarse value, i.e. the scheme silently degraded to
    // the standard one at the block boundaries - the defect this test exists for.
    PECLET_AMR_CHECK(cfgTot > 0);
  } else if (size == 1) {
    PECLET_AMR_CHECK_EQ(fw.numGhostCells(), 0);
    PECLET_AMR_CHECK_EQ(fw.numCfGhostColumns(), 0);
  }
  const int kSteps = std::getenv("CF_STEPS") ? std::atoi(std::getenv("CF_STEPS")) : 3;
  const Fields w = runSteps(fw, kSteps);

  AmrFlow<kBits> fs;
  fs.init(self.local(), kH0, Vec<3>{0.0, 0.0, 0.0});
  configure(fs);
  const Fields s = runSteps(fs, kSteps);

  double dmax = 0.0, scale = 0.0;
  for (Index i = 0; i < n; ++i) {
    const Index si = self.local().find(world.globalCode(i));
    PECLET_AMR_CHECK(si >= 0);
    for (int c = 0; c < 3; ++c) {
      scale = std::max(scale, std::fabs(s.u[(std::size_t)c][(std::size_t)si]));
      dmax = std::max(dmax, std::fabs(w.u[(std::size_t)c][(std::size_t)i] -
                                      s.u[(std::size_t)c][(std::size_t)si]));
    }
    dmax = std::max(dmax, std::fabs(w.p[(std::size_t)i] - s.p[(std::size_t)si]));
  }
  double gdmax = 0.0, gscale = 0.0;
  MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&scale, &gscale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("[cf-mpi] np=%d quadratic C/F WORLD vs SELF: |d|max %.3e (scale %.3e, rel %.3e)\n",
                size, gdmax, gscale, gdmax / (gscale + 1e-300));
  if (size == 1)
    PECLET_AMR_CHECK(gdmax == 0.0);  // BITWISE, the np=1 contract
  else
    PECLET_AMR_CHECK(gdmax <= 5e-6 * gscale);  // decomposition independence
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
