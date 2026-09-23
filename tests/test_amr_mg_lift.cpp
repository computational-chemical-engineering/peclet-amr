// The multigrid level below the root brick (docs/amr_mg_depth.md, ROADMAP C1, WO1).
//
// A level below the root brick is the SAME octree with its root LIFTED: the brick halved on every
// axis and `lmax` incremented, leaf codes untouched, so `coarsenIf` keeps merging. The claim that
// makes the design small is that a lifted level is bit-for-bit the level a DEEPER TREE on the same
// mesh would have produced — a uniform 64^3 mesh inside a depth-0 tree and the same mesh inside a
// depth-3 tree differ in two scalars and in nothing the solver reads.
//
// That emulation is the oracle here (§6.3). Validates:
//   (1) liftRoot() leaves codes, levels and `brick << lmax` invariant;
//   (2) Multigrid on Octree(64, lmax=0) and on the everywhere-refined Octree(64, lmax=3) build the
//       same number of levels, the same per-level leaf counts, and per-level face CSRs
//       (faceStart / faceNbr / faceW / bcDiag / invVol) that are BITWISE equal;
//   (3) five V-cycles from the same rhs give a bitwise equal x(0);
//   (4) level 0 is untouched by the lift (its CSR equals the unlifted build's), and
//       liftRoot=false reproduces the pre-C1 hierarchy exactly (one level on a uniform mesh).
//
#include <cmath>
#include <Kokkos_Core.hpp>
#include <vector>

#include "peclet/amr/block_octree.hpp"
#include "peclet/amr/multigrid.hpp"
#include "peclet/amr/poisson.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

constexpr unsigned kBits = 21;
using BO = BlockOctree<3, kBits>;
using Code = BO::Code;
using M = BO::M;

constexpr Index kN = 64;  // the uniform mesh both trees carry, in cells per axis

// A smooth openness in [~0.25, ~0.95] (never fully solid => no zero-diagonal cells), so the CSR
// comparison exercises the cut-cell weights and the whole coarsened-openness ladder.
double openFn(const Vec<3>& p, int /*axis*/) {
  const double k = 2.0 * M_PI;
  return 0.6 + 0.35 * std::sin(k * p[0]) * std::cos(k * p[1]) * std::cos(k * p[2]);
}

template <class T>
std::vector<T> getDev(View<const T> v) {
  auto m = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(m, v);
  return std::vector<T>(m.data(), m.data() + v.extent(0));
}

template <class T>
bool sameDev(View<T> a, View<T> b) {
  if (a.extent(0) != b.extent(0))
    return false;
  const auto ha = getDev(View<const T>(a)), hb = getDev(View<const T>(b));
  return ha == hb;  // bitwise: std::vector<double> operator== is element-wise ==
}

// The mesh both hierarchies run on, expressed in a tree of depth `lmax`: a root brick of
// kN / 2^lmax cells, refined everywhere down to level 0.
BO uniformMeshIn(unsigned lmax) {
  const Index root = kN >> lmax;
  BO t(IVec<3>{root, root, root}, lmax);
  for (unsigned r = 0; r < lmax; ++r)
    t.refineIf([](Code, unsigned L) { return L > 0; });
  return t;
}

void runLiftInvariants() {
  BO t = uniformMeshIn(0);
  const auto codes0 = t.codes();
  const auto levels0 = t.levels();
  const Index span0 = t.brick()[0] * (Index(1) << t.lmax());
  PECLET_AMR_CHECK(t.canLiftRoot());
  t.liftRoot();
  PECLET_AMR_CHECK(t.codes() == codes0);    // the mesh is untouched ...
  PECLET_AMR_CHECK(t.levels() == levels0);  // ... codes AND level bytes
  PECLET_AMR_CHECK_EQ(t.brick()[0] * (Index(1) << t.lmax()), span0);
  PECLET_AMR_CHECK_EQ((long long)t.brick()[0], (long long)(kN / 2));
  PECLET_AMR_CHECK_EQ((long long)t.lmax(), 1LL);
  // An odd brick cannot lift.
  BO odd(IVec<3>{5, 4, 4}, 0);
  PECLET_AMR_CHECK(!odd.canLiftRoot());
  // Nor can an even brick at an odd global origin (the distributed case, §6.4).
  BO offs(IVec<3>{4, 4, 4}, 0, IVec<3>{3, 0, 0});
  PECLET_AMR_CHECK(!offs.canLiftRoot());
}

void runEmulationOracle() {
  const double h0 = 1.0 / static_cast<double>(kN);
  const BO a = uniformMeshIn(0);  // Octree(64, lmax=0)
  const BO b = uniformMeshIn(3);  // Octree(64, lmax=3), refined everywhere
  // Same mesh, two trees: same leaf set, two scalars apart.
  PECLET_AMR_CHECK(a.codes() == b.codes());
  PECLET_AMR_CHECK(a.levels() == b.levels());
  PECLET_AMR_CHECK_EQ((long long)a.lmax(), 0LL);
  PECLET_AMR_CHECK_EQ((long long)b.lmax(), 3LL);

  Multigrid<3, kBits> mgA, mgB;
  mgA.build(a, h0, openFn, /*periodic=*/true);
  mgB.build(b, h0, openFn, /*periodic=*/true);

  // (a) the ladder: 64 -> 32 -> 16 -> 8 -> 4 either way (§6.2's first worked row).
  PECLET_AMR_CHECK_EQ((long long)mgA.numLevels(), (long long)mgB.numLevels());
  PECLET_AMR_CHECK_EQ((long long)mgA.numLevels(), 5LL);
  PECLET_AMR_CHECK_EQ((long long)mgA.numLeaves(mgA.numLevels() - 1), 4LL * 4 * 4);

  // (b) per-level leaf counts and face CSRs, bitwise.
  for (std::size_t L = 0; L < mgA.numLevels() && L < mgB.numLevels(); ++L) {
    PECLET_AMR_CHECK_EQ((long long)mgA.numLeaves(L), (long long)mgB.numLeaves(L));
    PECLET_AMR_CHECK_EQ((long long)mgA.numLeaves(L),
                        (long long)((kN >> L) * (kN >> L) * (kN >> L)));
    const FvOp& oa = mgA.op(L);
    const FvOp& ob = mgB.op(L);
    PECLET_AMR_CHECK(sameDev(oa.faceStart, ob.faceStart));
    PECLET_AMR_CHECK(sameDev(oa.faceNbr, ob.faceNbr));
    PECLET_AMR_CHECK(sameDev(oa.faceW, ob.faceW));
    PECLET_AMR_CHECK(sameDev(oa.bcDiag, ob.bcDiag));
    PECLET_AMR_CHECK(sameDev(oa.invVol, ob.invVol));
  }

  // (c) five V-cycles from the same rhs -> bitwise equal x(0).
  const Index n0 = mgA.numLeaves(0);
  std::vector<double> rhs((std::size_t)n0);
  for (Index i = 0; i < n0; ++i) {
    auto o = M::from_code(mgA.octreeCode(0, i)).decode();
    const double k = 2.0 * M_PI;
    rhs[(std::size_t)i] = std::sin(k * ((double)o[0] + 0.5) * h0) *
                          std::cos(k * ((double)o[1] + 0.5) * h0) *
                          std::cos(k * ((double)o[2] + 0.5) * h0);
  }
  auto load = [&](Multigrid<3, kBits>& mg) {
    auto m = Kokkos::create_mirror_view(mg.b(0));
    for (Index i = 0; i < n0; ++i)
      m(i) = rhs[(std::size_t)i];
    Kokkos::deep_copy(mg.b(0), m);
    Kokkos::deep_copy(mg.x(0), 0.0);
    for (int c = 0; c < 5; ++c)
      mg.vcycle(2, 2, 60, 0.8);
  };
  load(mgA);
  load(mgB);
  PECLET_AMR_CHECK(sameDev(mgA.x(0), mgB.x(0)));

  // (d) level 0 is untouched by the lift, and liftRoot=false is the pre-C1 hierarchy.
  Multigrid<3, kBits> mgOff;
  mgOff.build(a, h0, openFn, /*periodic=*/true, /*immersedWall=*/false, /*liftRoot=*/false);
  PECLET_AMR_CHECK_EQ((long long)mgOff.numLevels(), 1LL);
  PECLET_AMR_CHECK(sameDev(mgA.op(0).faceStart, mgOff.op(0).faceStart));
  PECLET_AMR_CHECK(sameDev(mgA.op(0).faceNbr, mgOff.op(0).faceNbr));
  PECLET_AMR_CHECK(sameDev(mgA.op(0).faceW, mgOff.op(0).faceW));
  PECLET_AMR_CHECK(sameDev(mgA.op(0).bcDiag, mgOff.op(0).bcDiag));
  PECLET_AMR_CHECK(sameDev(mgA.op(0).invVol, mgOff.op(0).invVol));
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  runLiftInvariants();
  runEmulationOracle();
  Kokkos::finalize();
  PECLET_AMR_RETURN_TEST_RESULT();
}
