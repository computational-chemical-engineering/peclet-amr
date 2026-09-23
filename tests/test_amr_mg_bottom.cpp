// The EXACT coarsest-level solve of the pressure multigrid — docs/amr_mg_depth.md §6.6, work
// order WO5.
//
// The §6.2 ladder lifts the root while the grid halves into a cube level, and stops once the
// coarsest extent is at or below `bottomExtent`, where 60 damped-Jacobi sweeps ARE a solve. On a
// badly factored root grid it stops earlier — 10^3 stops at 5^3, 100^3 at 25^3 — and there the
// sweeps are a smoother, not a solve (§6.6's amplification table: 5e-6 at E=5, 0.60 at E=25). The
// bottom there is the agglomerated GraphAMG-preconditioned CG on the level's own FV operator.
//
// What this asserts (single-rank; the distributed case runs inside the tail, amr_mg_bottom_dist):
//   (1) selection — a 10^3 root engages the exact bottom on 5^3 = 125 cells, a 16^3 root does not
//       (it reaches 4^3); `smoother` and `agglomerated` force each way;
//   (2) the §6.6 CONSISTENCY GATE: the CSR solution satisfies the V-cycle's own FvOp to
//       max|b − Lx| / max|b| <= 1e-9, both on the clean 10^3 case and on a cut-cell case built to
//       give the bottom exactly one identity row (a fully closed coarse cell) and TWO connected
//       fluid components (a sealed pocket) — the two structures §6.6 steps 2 and 3 exist for;
//   (3) the exact bottom is what a bottom is FOR: at a root grid whose ladder runs out (50^3 ->
//       25^3) the MG-PCG iteration count with `auto` is within 1 of the same problem on a root
//       grid that reaches 4^3 (64^3), while `smoother` on the same 50^3 hierarchy needs more.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/amr/block_octree.hpp"
#include "peclet/amr/multigrid.hpp"
#include "peclet/amr/pcg.hpp"
#include "peclet/core/common/view.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using Octree = BlockOctree<3, kBits>;
using MG = Multigrid<3, kBits>;

Octree uniformTree(long N) {
  Octree t;
  t.init(IVec<3>{N, N, N}, 0u);
  return t;
}

/// Which of the pocket case's three regions a cell lies in: 0 = the sealed single coarse cell
/// Q = [0,2)^3, 1 = the sealed 2^3 coarse block P = [4,8)^3, 2 = everything else.
int regionOf(const Octree& t, Index i) {
  auto lo = t.bounds(i)[0];
  bool inQ = true, inP = true;
  for (int d = 0; d < 3; ++d) {
    if (!((long)lo[d] >= 0 && (long)lo[d] < 2))
      inQ = false;
    if (!((long)lo[d] >= 4 && (long)lo[d] < 8))
      inP = false;
  }
  return inQ ? 0 : (inP ? 1 : 2);
}

std::vector<double> rhsOn(const Octree& t, double h0) {
  const Index n = t.numLeaves();
  std::vector<double> b(static_cast<std::size_t>(n));
  double s = 0.0;
  for (Index i = 0; i < n; ++i) {
    auto lo = t.bounds(i)[0];
    const double x = ((double)lo[0] + 0.5) * h0, y = ((double)lo[1] + 0.5) * h0,
                 z = ((double)lo[2] + 0.5) * h0;
    const double k = 2.0 * M_PI;
    b[(std::size_t)i] = std::sin(k * x) * std::cos(k * y) + std::cos(k * z) * std::sin(k * x);
    s += b[(std::size_t)i];
  }
  s /= (double)n;  // mean-zero: the singular operator's compatibility condition
  for (Index i = 0; i < n; ++i)
    b[(std::size_t)i] -= s;
  return b;
}

/// Cut-cell openness on a 10^3 grid, designed so that the 5^3 BOTTOM has exactly one identity row
/// and two connected fluid components. Both sealed regions are aligned to the coarse grid, so the
/// area-averaged coarse apertures are exactly 0 on their boundaries:
///   * Q = [0,2)^3 fine = the single coarse cell (0,0,0), sealed on all six sides => identity row;
///   * P = [4,8)^3 fine = the 2^3 coarse block [2,4)^3, sealed on all six sides => a second
///     component of 8 cells, its interior faces open.
struct PocketOpenness {
  double h0;
  static bool sealedBox(long plane, long t0, long t1, long lo, long hi, long N) {
    // A face of the box [lo,hi) on this axis: the plane is lo or hi (hi == N wraps to 0), and both
    // tangential cell indices lie inside the box.
    const bool onPlane =
        (plane == lo) || (plane == hi) || (hi == N && plane == 0) || (lo == 0 && plane == N);
    return onPlane && t0 >= lo && t0 < hi && t1 >= lo && t1 < hi;
  }
  double operator()(const Vec<3>& fc, int axis) const {
    const long N = 10;
    long idx[3];
    for (int d = 0; d < 3; ++d)
      idx[d] = (d == axis) ? (long)std::lround(fc[d] / h0) : (long)std::floor(fc[d] / h0 + 1e-9);
    const int a = (axis + 1) % 3, b = (axis + 2) % 3;
    if (sealedBox(idx[axis], idx[a], idx[b], 0, 2, N))
      return 0.0;
    if (sealedBox(idx[axis], idx[a], idx[b], 4, 8, N))
      return 0.0;
    return 1.0;
  }
};

/// A ROUGH mean-zero rhs: a smooth one is a couple of Fourier modes that PCG kills in three
/// iterations whatever the bottom does, so it cannot tell a solved coarsest level from a smoothed
/// one. A deterministic per-cell hash spreads energy over the whole spectrum, which is what a
/// coarse-grid correction is actually for.
std::vector<double> roughRhsOn(const Octree& t, double h0) {
  (void)h0;
  const Index n = t.numLeaves();
  std::vector<double> b(static_cast<std::size_t>(n));
  double s = 0.0;
  for (Index i = 0; i < n; ++i) {
    auto lo = t.bounds(i)[0];
    std::uint64_t h = 1469598103934665603ull;
    for (int d = 0; d < 3; ++d)
      h = (h ^ static_cast<std::uint64_t>(lo[d])) * 1099511628211ull;
    b[(std::size_t)i] = (double)((h >> 11) & 0xFFFFF) / (double)0x100000 - 0.5;
    s += b[(std::size_t)i];
  }
  s /= (double)n;
  for (Index i = 0; i < n; ++i)
    b[(std::size_t)i] -= s;
  return b;
}

/// PCG iteration count for a uniform periodic Poisson on an N^3 root grid.
struct PcgRun {
  int iters = 0;
  std::size_t levels = 0;
  long coarsest = 0;
  std::string bottom;
};
PcgRun pcgRun(long N, MG::Bottom kind) {
  const double h0 = 1.0 / (double)N;
  Octree t = uniformTree(N);
  MG mg;
  mg.build(t, h0);
  mg.setBottom(kind);
  mg.setRemoveMean(true);
  std::vector<double> b = roughRhsOn(t, h0);
  View<double> bv = toDevice(b, "b");
  View<double> x("x", (std::size_t)t.numLeaves());
  PCG<3, kBits> pcg;
  pcg.setVcycle(2, 2, 60, 0.8);
  pcg.setSingular(true);
  auto r = pcg.solve(mg, x, View<const double>(bv), 300, 1e-10);
  PECLET_AMR_CHECK(r.res <= 1e-9 * r.res0);
  PcgRun out;
  out.iters = r.iters;
  out.levels = mg.numLevels();
  out.coarsest = (long)mg.numLeaves(mg.numLevels() - 1);
  out.bottom = mg.bottomName();
  return out;
}

void run() {
  // ---- (1) selection -------------------------------------------------------------------------
  {
    Octree t10 = uniformTree(10);
    MG mg;
    mg.build(t10, 0.1);
    PECLET_AMR_CHECK_EQ((long)mg.numLevels(), 2L);  // 10 -> 5, then 5 is odd
    PECLET_AMR_CHECK(mg.bottomName() == std::string("amg"));
    PECLET_AMR_CHECK_EQ((long)mg.bottomSize(), 125L);
    PECLET_AMR_CHECK_EQ((long)mg.bottomComponents(), 1L);
    PECLET_AMR_CHECK_EQ((long)mg.bottomIdentityRows(), 0L);
    mg.setBottom(MG::Bottom::Smoother);
    PECLET_AMR_CHECK(mg.bottomName() == std::string("jacobi"));
    mg.setBottom(MG::Bottom::Agglomerated);
    PECLET_AMR_CHECK(mg.bottomName() == std::string("amg"));

    Octree t16 = uniformTree(16);
    MG mg16;
    mg16.build(t16, 1.0 / 16.0);
    PECLET_AMR_CHECK_EQ((long)mg16.numLevels(), 3L);  // 16 -> 8 -> 4
    PECLET_AMR_CHECK(mg16.bottomName() == std::string("jacobi"));
    mg16.setBottom(MG::Bottom::Agglomerated);
    PECLET_AMR_CHECK(mg16.bottomName() == std::string("amg"));
  }

  // ---- (2a) the consistency gate on the clean 10^3 case --------------------------------------
  {
    const double h0 = 0.1;
    Octree t = uniformTree(10);
    MG mg;
    mg.build(t, h0);
    mg.setBottomCheck(true);
    std::vector<double> b = rhsOn(t, h0);
    View<double> bv = toDevice(b, "b2");
    Kokkos::deep_copy(mg.b(0), bv);
    Kokkos::deep_copy(mg.x(0), 0.0);
    for (int c = 0; c < 3; ++c)
      mg.vcycle(2, 2, 60, 0.8);
    std::printf("[bottom] 10^3 clean: n_b %lld comps %d ident %lld cg-it %d |b-Lx|/|b| %.3e\n",
                (long long)mg.bottomSize(), mg.bottomComponents(),
                (long long)mg.bottomIdentityRows(), mg.bottomIters(), mg.bottomResidual());
    PECLET_AMR_CHECK(mg.bottomResidual() <= 1e-9);
  }

  // ---- (2b) the consistency gate with identity rows and two components -----------------------
  {
    const double h0 = 0.1;
    Octree t = uniformTree(10);
    MG mg;
    PocketOpenness open{h0};
    mg.build(t, h0, open, /*periodic=*/true);
    mg.setBottomCheck(true);
    PECLET_AMR_CHECK(mg.bottomName() == std::string("amg"));
    PECLET_AMR_CHECK_EQ((long)mg.bottomIdentityRows(), 1L);
    PECLET_AMR_CHECK_EQ((long)mg.bottomComponents(), 2L);
    // A singular operator only has a solution for a rhs orthogonal to its nullspace, and with a
    // sealed pocket that nullspace is a constant PER COMPONENT: the rhs must be mean-zero on each
    // component separately, and zero on the sealed cell that becomes the bottom's identity row
    // (which is what a solid region's divergence is in the flow). The restriction is conservative,
    // so a fine rhs built that way stays compatible all the way down.
    std::vector<double> b = rhsOn(t, h0);
    {
      double sum[3] = {0, 0, 0};
      long cnt[3] = {0, 0, 0};
      for (Index i = 0; i < t.numLeaves(); ++i) {
        const int r = regionOf(t, i);
        sum[r] += b[(std::size_t)i];
        ++cnt[r];
      }
      for (Index i = 0; i < t.numLeaves(); ++i) {
        const int r = regionOf(t, i);
        b[(std::size_t)i] = (r == 0) ? 0.0 : b[(std::size_t)i] - sum[r] / (double)cnt[r];
      }
    }
    View<double> bv = toDevice(b, "b3");
    Kokkos::deep_copy(mg.b(0), bv);
    Kokkos::deep_copy(mg.x(0), 0.0);
    for (int c = 0; c < 3; ++c)
      mg.vcycle(2, 2, 60, 0.8);
    std::printf("[bottom] 10^3 pocket: n_b %lld comps %d ident %lld cg-it %d |b-Lx|/|b| %.3e\n",
                (long long)mg.bottomSize(), mg.bottomComponents(),
                (long long)mg.bottomIdentityRows(), mg.bottomIters(), mg.bottomResidual());
    PECLET_AMR_CHECK(mg.bottomResidual() <= 1e-9);
  }

  // ---- (3) the exact bottom buys back the iteration count ------------------------------------
  for (long N : {32L, 50L, 64L, 100L, 128L}) {
    const PcgRun a = pcgRun(N, MG::Bottom::Auto);
    const PcgRun sm = pcgRun(N, MG::Bottom::Smoother);
    std::printf("[bottom] N=%3ld levels %zu coarsest %6ld | auto %-7s %3d it | smoother %3d it\n",
                N, a.levels, a.coarsest, a.bottom.c_str(), a.iters, sm.iters);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  PECLET_AMR_RETURN_TEST_RESULT();
}
