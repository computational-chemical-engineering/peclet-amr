// ABC/Basilisk divergence-free FACE field on the host oracle AmrFlow. The collocated projection
// only makes the *cell* field approximately divergence-free (O(h²)); the face field uf_f =
// ½(u_i+u_j) − (φ₊−φ₋)/d is divergence-free to the pressure-solve residual, because L = D·G_face on
// the same (sub)faces ⇒ D(uf) = D u* − Lφ. This must hold across 2:1 interfaces too (the coarse
#include <cmath>

#include "peclet/amr/block_octree.hpp"
#include "peclet/amr/flow_oracle.hpp"
#include "peclet/amr/refine.hpp"
#include "peclet/core/common/types.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;
using BO = BlockOctree<3, 21>;
using Code = BO::Code;

namespace {

// Run a Stokes sphere to steady, return (divNorm cell, divNorm face).
std::pair<double, double> run(BO& t, double R, Vec<3> c, int presIters, bool ghost = false,
                              int cf = -1) {
  const double mu = 0.1, f = 1e-3, dt = 60.0;
  oracle::AmrFlow<21>
      fl;  // NB scheme pinned below: this test asserts APERTURE face-field properties
  fl.init(t, 1.0, Vec<3>{0, 0, 0});
  fl.setGhostProjection(ghost);  // default: explicit aperture (the scheme under test)
  if (cf >= 0)
    fl.setCfScheme(cf);
  fl.setDensity(1.0);
  fl.setViscosity(mu);
  fl.setDt(dt);
  fl.setBodyForce(f, 0, 0);
  fl.setAdvection(false);
  fl.setSolid([&](const Vec<3>& p) {
    double dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz) - R;
  });
  // The face field's divergence-free property is algebraic (div(uf) = div(u*) − Lφ), so it holds at
  // any state — a handful of steps is enough to exercise it without running to full steady state.
  for (int it = 0; it < 25; ++it)
    fl.step(60, presIters, 2);
  return {fl.divNormL2(fl.velocityRef()), fl.divNormFace()};
}

void run_test() {
  // (1) uniform finest N=16: the face field is far cleaner than the cell field, and tightening the
  //     pressure solve shrinks it (it is the solve residual, not a fixed O(h²) error).
  const unsigned L = 4;
  const long N = 1L << L;
  const double R = std::pow(0.125 * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * N, cc = N / 2.0;
  BO t6(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k)
    t6.refineIf([](Code, unsigned) { return true; });
  auto [dCell6, dFace6] = run(t6, R, Vec<3>{cc, cc, cc}, 6);
  BO t30(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k)
    t30.refineIf([](Code, unsigned) { return true; });
  auto [dCell30, dFace30] = run(t30, R, Vec<3>{cc, cc, cc}, 30);
  PECLET_AMR_CHECK(dFace6 <
                   0.05 * dCell6);  // face field ≥20× more divergence-free than the cell field
  PECLET_AMR_CHECK(dFace30 < 0.05 * dCell30);  // ditto at the tighter solve
  // The face divergence used to TRACK the pressure-solve residual, so tightening the solve shrank
  // it. Since the gauge-exact cell gradient became the default (2026-08-18) it is ~85x smaller and
  // sits on a floor instead: measured here 5.87e-06 -> 5.54e-06 with the legacy gradient against
  // 6.6992e-08 -> 6.6995e-08 with the gauge-exact one. So assert what still holds -- it does not
  // GROW when the solve is tightened -- and pin the absolute level the floor sits at.
  PECLET_AMR_CHECK(dFace30 <= 1.05 * dFace6);
  PECLET_AMR_CHECK(dFace30 < 1e-6);
  //         (the cell-field divergence is the fixed O(h²) approximate-projection error, ~unchanged)

  // (2) graded 2:1 grid: the face field stays divergence-free across the coarse–fine interfaces,
  // where
  //     the cell field's divergence is actually larger.
  const unsigned lmax = 4;
  const long Nf = 1L << lmax;
  BO tg(IVec<3>{2, 2, 2}, lmax);
  AmrGeometry<3> geo;
  geo.setIsotropic(1.0);
  const double Rg = std::pow(0.125 * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * (2 * Nf),
               cg = (2.0 * Nf) / 2.0;
  refineToSdf(
      tg, geo,
      [&](const Vec<3>& p) {
        double dx = p[0] - cg, dy = p[1] - cg, dz = p[2] - cg;
        return std::sqrt(dx * dx + dy * dy + dz * dz) - Rg;
      },
      /*target_level=*/0, /*band=*/3.0, /*balance=*/true);
  PECLET_AMR_CHECK(tg.isBalanced());
  auto [dCellG, dFaceG] = run(tg, Rg, Vec<3>{cg, cg, cg}, 30, false, /*cf=*/0);
  PECLET_AMR_CHECK(dFaceG <
                   0.01 * dCellG);  // across 2:1: face field ≥100× cleaner than the cell field

  // ...and the same case on the DEFAULT C/F scheme (quadratic since 2026-09-21), where that
  // assertion does NOT hold and must not be expected to. This is the second face of the trap arm
  // (3) gates: divFaceNorm measures the PLAIN area-weighted face divergence, and cf=1 solves a
  // DIFFERENT constraint -- buildCfUfDelta adds cfUfVel_/cfUfPhi_ to uf precisely so "the
  // advecting flux matches the (quad) divergence constraint". Measured on this mesh:
  //
  //     cf=0:  cell 1.3711e-01   face 1.3476e-09     (face IS the residual)
  //     cf=1:  cell 4.8407e-01   face 6.6780e-03     (face measures a constraint not solved)
  //
  // The scheme is not degrading anything -- it is markedly BETTER. On the same geometry as a
  // graded 14120-leaf mesh against a uniform-fine 32^3 reference, the volume-averaged velocity
  // (i.e. the permeability) errs by 5.15e-02 at cf=0 and 1.22e-02 at cf=1: the default is 4.2x
  // more accurate. So assert the RELATION that survives the scheme change -- cf=1 does not make
  // the face field worse than the cell field -- and pin the cf=0 numbers above as the statement
  // about the standard scheme they actually are.
  auto [dCellGq, dFaceGq] = run(tg, Rg, Vec<3>{cg, cg, cg}, 30, false, /*cf=*/1);
  PECLET_AMR_CHECK(dFaceGq < dCellGq);
  PECLET_AMR_CHECK(dFaceGq > 100.0 * dFaceG);  // the diagnostic, not the solver, is what moved

  // (3) THE SCOPE OF divNormFace, gated so it cannot quietly become a trap again.
  //
  // Everything above runs the APERTURE scheme, where divNormFace IS the projection residual: the
  // face field comes out ~5e4x cleaner than the cell field. Under the GHOST projection — the
  // production DEFAULT — the constraint actually solved is that divergence PLUS the overlay delta
  // (ghostDivergDelta), which is a functional of the CELL velocities and so cannot appear in any
  // norm of uf. divNormFace then measures a constraint the solver never solved, and carries no
  // information: measured here at N=16 on the oracle,
  //
  //     aperture:  cell 3.4321e-03   face 6.8846e-08     (face is the residual, ~5e4x cleaner)
  //     ghost:     cell 5.1770e-02   face 5.1709e-02     (face == cell; it says nothing)
  //
  // on a solve that is healthy — the device ghost path matches peclet.flow's collocated solver to
  // 1e-6 on this very geometry (docs/amr_flow_uniform_parity.md §2, §2a). NB the ORACLE's
  // divNormL2 is the plain aperture divergence on both schemes; it is the DEVICE AmrFlow's
  // divNormL2() that folds the overlay delta in and is the right ghost residual.
  BO tgh(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k)
    tgh.refineIf([](Code, unsigned) { return true; });
  auto [dCellGh, dFaceGh] = run(tgh, R, Vec<3>{cc, cc, cc}, 30, /*ghost=*/true);
  // Orders of magnitude above the aperture path's residual, where the same call IS the residual.
  PECLET_AMR_CHECK(dFaceGh > 1e3 * dFace30);
  // ...and indistinguishable from the cell divergence, i.e. it adds nothing over divergence_norm.
  PECLET_AMR_CHECK(dFaceGh > 0.5 * dCellGh);
  // If either check ever fails, either the overlay delta has been folded into divFaceNorm (good --
  // retire this block and the caveats on AmrFlow::divNormFace and its binding) or the ghost
  // constraint has changed (investigate before re-blessing).
}

}  // namespace

int main() {
  run_test();
  if (peclet::amr::test::g_failures == 0)
    std::printf("OK\n");
  return peclet::amr::test::g_failures == 0 ? 0 : 1;
}
