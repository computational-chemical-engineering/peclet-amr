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
// TWO ARMS on that one mesh.
//   (1) Advection OFF, and on purpose: that is the configuration in which nothing else refreshes
//       u's ghost tail between the projection and the next predictor, so it also gates the
//       syncVel that the momentum C/F delta (cfMom_, which reads u at the coarse cell's
//       tangential neighbours) needs.
//   (2) Advection ON (WO4 / gate G4 of docs/amr_cf_convective.md): the seam reconstruction of the
//       advected value, whose records the two advective kernels gather THROUGH GHOST SLOTS. See
//       the block above advectionArm() below for why only a distributed advective march can see a
//       wrong ghost reach, and for the constants that keep the march non-vacuous.
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

/// CF_SAMPLED=1 turns the mixed-level cut band on as well. That combination is what the discovery
/// arm's gate-free design is really for: on the sampled path mom_ is not built AT ALL inside the
/// fixpoint (D3's setup-cost fix moved its build after it), so the builders' fluid predicates are
/// unusable during discovery and a probe arm that leaned on them would miss coordinates. Not a
/// registered ctest - the gated configuration is the classic overlay, which keeps the run cheap -
/// but the knob keeps the combination one command away. Measured: np = 1 bitwise, np = 2
/// rel 1.611e-07, np = 4 rel 1.104e-07 - the same numbers as the classic overlay, because on a
/// uniform finest band the sampled overlay is identity slots. What does change is the fixpoint,
/// which goes from 4 rounds / ~1000 ghosts to 6-7 rounds / 2000-5200: that is the LS clouds'
/// reach, not the C/F arm's.
bool sampledBand() {
  const char* e = std::getenv("CF_SAMPLED");
  return e && e[0] == '1';
}

void configure(AmrFlow<kBits>& f) {
  f.setDensity(1.0);
  f.setViscosity(1.0);
  f.setBodyForce(1.0, 0.0, 0.0);
  f.setDt(1e6);
  f.setGhostProjection(true, 2, 2);
  if (sampledBand())
    f.setGhostSampled(true);
  if (!standardControl())
    f.setCfScheme(static_cast<int>(CfScheme::quadratic));
  f.setSolid(sphereSdf);
}

// ---- WO4 / gate G4 (docs/amr_cf_convective.md §8, §9): the ADVECTIVE arm ----------------------
//
// The seam reconstruction of the ADVECTED value (§5) is the one piece of this family that reaches
// through ghost slots from the kernel's own gather: a SAMPLE record is the coarse cell plus
// cfAppendStencil's tangential entries, a LAYER record the four fine cells behind a coarse cell's
// face, and at a block seam either can live on another rank. §5.4 is explicit that the J-side of
// a descriptor needs a reach the face sweep does not already register — the four face-layer
// corners BEYOND a ghost neighbour and the tangential reach of a ghost's own coarser upstream —
// and that is what probeSeamLayer() exists for. A reach that is one probe short is INVISIBLE
// single-rank (every coordinate is a local leaf, so the record is simply built) and wrong on 2+
// ranks: the record is withheld and the slot silently falls back to today's O(h) arithmetic, or
// resolves to an unrelated leaf. Neither of this repo's distributed flow tests enabled advection
// on a graded mesh, so nothing saw it.
//
// The arm therefore marches the SAME graded ladder with advection ON. Its constants are chosen so
// that the advective term is a material part of the answer rather than a rounding-level addition:
// an amplitude-1 Taylor-Green initial velocity (analytically divergence free, and a pure function
// of the cell centre so WORLD and SELF start from bitwise the same field) crossing every 2:1
// shell from the first step, nu = 0.02 (cell Reynolds |u| h / nu = 1.6 on the finest cell) and
// dt = 0.01 (CFL 0.32 there). The NON-VACUITY control measures that directly: the same SELF march
// with setSeamReconstruction(false) must differ from the seam-on march by MORE than the
// decomposition tolerance the WORLD/SELF comparison is judged at, or the comparison could not see
// a broken ghost reach at all. The seam census is asserted positive per rank for the same reason.
//
// Measured (host-openmp, OMP_NUM_THREADS=2, this mesh, 3 steps): WORLD vs SELF rel 0 (bitwise) at
// np = 1, 6.322e-09 at np = 2, 4.701e-09 at np = 4, 7.782e-09 at np = 8 — a decade under the
// ~3e-7 class the advection-off arm above sits in, because dt = 0.01 needs far fewer Krylov
// iterations than the dt = 1e6 steady arm for its reductions to reorder in. Seam census: 3064
// sample + 766 layer records single-rank, rising to Σ 3264 / 872 at np = 8 (a seam pair that
// straddles a block boundary is built on BOTH of its ranks — Σ_ranks is NOT the single-rank count,
// which is why only the multiset, gated in Python, can be compared across np). Record entries
// reading a ghost slot: 0 at np = 1, Σ 3264 / 5000 / 5816 at np = 2 / 4 / 8. Non-vacuity margin
// (seam ON vs OFF): 8.335e-02 = 10.1 % of the velocity scale, 2.0e4 times the tolerance.
constexpr double kAdvNu = 0.02;
constexpr double kAdvDt = 0.01;

void configureAdvect(AmrFlow<kBits>& f, bool seamRecon) {
  f.setDensity(1.0);
  f.setViscosity(kAdvNu);
  f.setBodyForce(0.0, 0.0, 0.0);  // the initial vortex is the drive
  f.setDt(kAdvDt);
  f.setGhostProjection(true, 2, 2);
  f.setCfScheme(static_cast<int>(CfScheme::quadratic));  // the scheme the seam tables ride
  f.setAdvection(true);
  f.setAdvectionScheme(0);  // SOU (the default); the seam branch is scheme-independent
  f.setSeamReconstruction(seamRecon);
  f.setSolid(sphereSdf);
}

/// u = (sin 2πx cos 2πy cos 2πz, −½ cos sin cos, −½ cos cos sin) — divergence free analytically,
/// and a pure function of the cell CENTRE, which is what keeps the np = 1 contract bitwise: the
/// two builds compute the same expression on the same world coordinate.
void seedTaylorGreen(AmrFlow<kBits>& f) {
  const std::vector<double> c = f.slotCenters();
  const Index n = f.numLeaves();
  const double k = 2.0 * M_PI;
  std::array<std::vector<double>, 3> u;
  for (int a = 0; a < 3; ++a)
    u[(std::size_t)a].resize((std::size_t)n);
  for (Index i = 0; i < n; ++i) {
    const double x = c[(std::size_t)i * 3 + 0], y = c[(std::size_t)i * 3 + 1],
                 z = c[(std::size_t)i * 3 + 2];
    u[0][(std::size_t)i] = std::sin(k * x) * std::cos(k * y) * std::cos(k * z);
    u[1][(std::size_t)i] = -0.5 * std::cos(k * x) * std::sin(k * y) * std::cos(k * z);
    u[2][(std::size_t)i] = -0.5 * std::cos(k * x) * std::cos(k * y) * std::sin(k * z);
  }
  for (int a = 0; a < 3; ++a)
    f.setVelocity(a, u[(std::size_t)a]);
}

/// Seam-record entries that read a GHOST slot — the seam tables' analogue of numCfGhostColumns,
/// read off faceTopology() so no production accessor has to exist for it. Zero at np = 1 (there
/// are no ghosts); zero at np > 1 would mean every seam record was built from local leaves only,
/// i.e. the seams of this rank's blocks are not actually crossed by the tables.
long seamGhostEntries(const AmrFlow<kBits>& f) {
  const auto top = f.faceTopology();
  const Index n = f.numLeaves();
  long g = 0;
  for (const Index c : top.recCell)
    if (c >= n)
      ++g;
  return g;
}

/// max |WORLD − SELF| over this rank's leaves, mapped through the global Morton code.
double worldSelfDiff(const Fields& w, const Fields& s, const DO& world, DO& self, Index n) {
  double d = 0.0;
  for (Index i = 0; i < n; ++i) {
    const Index si = self.local().find(world.globalCode(i));
    PECLET_AMR_CHECK(si >= 0);
    for (int c = 0; c < 3; ++c)
      d = std::max(
          d, std::fabs(w.u[(std::size_t)c][(std::size_t)i] - s.u[(std::size_t)c][(std::size_t)si]));
    d = std::max(d, std::fabs(w.p[(std::size_t)i] - s.p[(std::size_t)si]));
  }
  return d;
}

void advectionArm(DO& world, DO& self, Index n, int rank, int size) {
  const int kSteps = std::getenv("CF_ASTEPS") ? std::atoi(std::getenv("CF_ASTEPS")) : 3;

  AmrFlow<kBits> fw;
  fw.initMpi(world);
  configureAdvect(fw, true);
  seedTaylorGreen(fw);
  {
    // The census. Sample and layer records are LOCAL counts (a seam pair straddling a block
    // boundary is built on BOTH ranks, each from its own side, so Σ_ranks exceeds the single-rank
    // count by design — the record MULTISET equality across np is gated in Python by
    // python_amr_seam_records, WO1c, not here). What this asserts is only that the mesh is not
    // vacuous: every rank owns seam faces of both kinds, and at np > 1 its tables really do read
    // across the block boundary.
    const long sr = (long)fw.numSeamSampleRecords(), lr = (long)fw.numSeamLayerRecords();
    const long ge = seamGhostEntries(fw);
    PECLET_AMR_CHECK(sr > 0);
    PECLET_AMR_CHECK(lr > 0);
    if (size > 1)
      PECLET_AMR_CHECK(ge > 0);
    else
      PECLET_AMR_CHECK_EQ(ge, 0L);
    long srT = 0, lrT = 0, geT = 0, srMin = 0, lrMin = 0, geMin = 0;
    MPI_Allreduce(&sr, &srT, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lr, &lrT, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ge, &geT, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sr, &srMin, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lr, &lrMin, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ge, &geMin, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf(
          "[cf-mpi] advective arm np=%d seam census: sample records Σ %ld (min/rank %ld), layer "
          "records Σ %ld (min/rank %ld), record entries reading a ghost slot Σ %ld (min/rank "
          "%ld)\n",
          size, srT, srMin, lrT, lrMin, geT, geMin);
  }
  const Fields w = runSteps(fw, kSteps);

  AmrFlow<kBits> fs;
  fs.init(self.local(), kH0, Vec<3>{0.0, 0.0, 0.0});
  configureAdvect(fs, true);
  seedTaylorGreen(fs);
  const Fields s = runSteps(fs, kSteps);

  double scale = 0.0;
  const Index nSelf = fs.numLeaves();
  for (Index i = 0; i < nSelf; ++i)
    for (int c = 0; c < 3; ++c)
      scale = std::max(scale, std::fabs(s.u[(std::size_t)c][(std::size_t)i]));
  const double dmax = worldSelfDiff(w, s, world, self, n);
  double gdmax = 0.0;
  MPI_Allreduce(&dmax, &gdmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(
        "[cf-mpi] np=%d seam reconstruction, advection ON, WORLD vs SELF: |d|max %.3e (scale "
        "%.3e, rel %.3e)\n",
        size, gdmax, scale, gdmax / (scale + 1e-300));

  {
    // NON-VACUITY. The same SELF march with the seam reconstruction switched off: if turning the
    // feature off moved the answer by less than the tolerance the WORLD/SELF comparison is judged
    // at, that comparison could not detect a seam path broken at the block boundaries, and this
    // arm would gate nothing. It is a property of the CASE (mesh, dt, viscosity, step count), so
    // it belongs in the test, not in a study.
    AmrFlow<kBits> fo;
    fo.init(self.local(), kH0, Vec<3>{0.0, 0.0, 0.0});
    configureAdvect(fo, false);
    seedTaylorGreen(fo);
    const Fields o = runSteps(fo, kSteps);
    double doff = 0.0;
    for (Index i = 0; i < nSelf; ++i) {
      for (int c = 0; c < 3; ++c)
        doff = std::max(doff, std::fabs(s.u[(std::size_t)c][(std::size_t)i] -
                                        o.u[(std::size_t)c][(std::size_t)i]));
      doff = std::max(doff, std::fabs(s.p[(std::size_t)i] - o.p[(std::size_t)i]));
    }
    if (rank == 0)
      std::printf(
          "[cf-mpi] np=%d non-vacuity: seam ON vs OFF on the SAME march |d|max %.3e (rel %.3e, "
          "tolerance 5.000e-06)\n",
          size, doff, doff / (scale + 1e-300));
    PECLET_AMR_CHECK(doff > 5e-6 * scale);
  }

  if (size == 1)
    PECLET_AMR_CHECK(gdmax == 0.0);  // BITWISE, the np=1 contract
  else
    PECLET_AMR_CHECK(gdmax <= 5e-6 * scale);  // decomposition independence (G4)
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

  {
    // Gate C' (docs/amr_cf_flux_gate.md §10): this mesh's cut band is uniformly FINEST (targetLevel
    // returns 0 inside 3.5 h0 of the surface), so no cut cell has a 2:1 face and the per-face C/F
    // gate withholds nothing. The census must be exactly zero — on every rank and single-rank.
    long cw = static_cast<long>(fw.numCfCutFaces()), cwTot = 0;
    MPI_Allreduce(&cw, &cwTot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    PECLET_AMR_CHECK_EQ(cwTot, 0L);
    PECLET_AMR_CHECK_EQ(static_cast<long>(fs.numCfCutFaces()), 0L);
  }

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

  advectionArm(world, self, n, rank, size);  // arm (2): WO4 / G4
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
