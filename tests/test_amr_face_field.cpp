// ABC/Basilisk divergence-free FACE field on the host oracle AmrFlow. The collocated projection
// only makes the *cell* field approximately divergence-free (O(h²)); the face field uf_f =
// ½(u_i+u_j) − (φ₊−φ₋)/d is divergence-free to the pressure-solve residual, because L = D·G_face on
// the same (sub)faces ⇒ D(uf) = D u* − Lφ. This must hold across 2:1 interfaces too (the coarse
#include <algorithm>
#include <cmath>
#include <cstdio>

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

// What one Stokes run reports. `dCell`/`dFace` are the whole-field cell and face divergence
// norms; the `*Reg` pair is the same two over REGULAR fluid rows only (cut rows excluded, where
// the ghost scheme's D_std(uf) != 0 by design); `ident` is rule (I) of
// docs/amr_cf_flux_gate.md -- ||D_std(d_uf) - d_cfDiv||, which must be at round-off for uf to be
// a conservative flux -- with `identRef` its scale; `cutFaces` is the C/F census.
struct Res {
  double dCell = 0.0, dFace = 0.0;
  double ident = 0.0, identRef = 0.0;
  double dCellReg = 0.0, dFaceReg = 0.0;
  Index cutFaces = 0;
};

// Run a Stokes sphere to steady and report the above.
Res run(BO& t, double R, Vec<3> c, int presIters, bool ghost = false, int cf = -1,
        bool sampled = false) {
  const double mu = 0.1, f = 1e-3, dt = 60.0;
  oracle::AmrFlow<21>
      fl;  // NB scheme pinned below: this test asserts APERTURE face-field properties
  fl.init(t, 1.0, Vec<3>{0, 0, 0});
  fl.setGhostProjection(ghost);  // default: explicit aperture (the scheme under test)
  if (sampled)
    fl.setGhostSampled(true);  // mixed-level cut band: the only ghost closure valid at band=0
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
  Res r;
  r.dCell = fl.divNormL2(fl.velocityRef());
  r.dFace = fl.divNormFace();
  const auto id = fl.cfFluxIdentity();
  r.ident = id.first;
  r.identRef = id.second;
  r.dCellReg = fl.divNormL2Regular(fl.velocityRef());
  r.dFaceReg = fl.divNormFaceRegular();
  r.cutFaces = fl.numCfCutFaces();
  return r;
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
  const Res r6 = run(t6, R, Vec<3>{cc, cc, cc}, 6);
  const double dCell6 = r6.dCell, dFace6 = r6.dFace;
  BO t30(IVec<3>{1, 1, 1}, L);
  for (unsigned k = 0; k < L; ++k)
    t30.refineIf([](Code, unsigned) { return true; });
  const Res r30 = run(t30, R, Vec<3>{cc, cc, cc}, 30);
  const double dCell30 = r30.dCell, dFace30 = r30.dFace;
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
  const Res rG = run(tg, Rg, Vec<3>{cg, cg, cg}, 30, false, /*cf=*/0);
  const double dCellG = rG.dCell, dFaceG = rG.dFace;
  PECLET_AMR_CHECK(dFaceG <
                   0.01 * dCellG);  // across 2:1: face field ≥100× cleaner than the cell field

  // ...and the same case on the DEFAULT C/F scheme (quadratic since 2026-09-21), which must hold
  // it just as well. uf's whole purpose is to be the CONSERVATIVE ADVECTING FLUX: advect a scalar
  // with it and mass is conserved, a uniform field stays uniform. That is the statement
  //     sum_faces alpha * A * dir * uf = 0   per cell,
  // i.e. exactly what divNormFace measures, and it must hold whatever C/F scheme is selected.
  //
  // It did not, until 2026-09-22: uf was built as avg(u*) + dvel - G_std*phi + dphi, and the
  // projection makes only the first three divergence-free. dphi substitutes a quadratic coarse*
  // into uf's FACE GRADIENT while the pressure matrix is L = D_std*G_std, so nothing in the solve
  // balanced it. Decomposed on this mesh (identity holds to 2.3e-15):
  //     solve residual ||rhs - L*phi||   1.34e-12
  //     ||D(dvel) - d_cfDiv||            8.71e-17   <- velocity part: balanced by construction
  //     ||D(dphi)||                      6.675e-03  <- the ENTIRE imbalance
  // The phi part is no longer applied (flow.hpp::finishProjection), and the face field is a
  // conservative flux again: 6.675e-03 -> 1.34e-12 here, 0.240 -> 3.8e-11 at the worst transient,
  // and 4.0e-02 -> 1.2e-05 with advection on. The steady answer is untouched -- phi -> 0 at the
  // fixed point, so this only ever corrupted transients, which is why every steady drag gate
  // passed over it.
  const Res rGq = run(tg, Rg, Vec<3>{cg, cg, cg}, 30, false, /*cf=*/1);
  const double dCellGq = rGq.dCell, dFaceGq = rGq.dFace;
  PECLET_AMR_CHECK(dFaceGq < 0.01 * dCellGq);  // the SAME bar cf=0 is held to, above
  // and within an order of magnitude of it -- not the 5e6x it was before the fix.
  PECLET_AMR_CHECK(dFaceGq < 100.0 * dFaceG);
  // Gate I' (docs/amr_cf_flux_gate.md §10): on THIS mesh the cut cells are uniformly finest, so no
  // cut cell has a 2:1 face, the per-FACE C/F gate withholds nothing, and rule (I) was already at
  // round-off (8.7e-17) before the gate changed shape. It must stay there.
  std::printf("[face-field] band=3 cf=1: cfCutFaces %lld, identity %.4e (ref %.4e)\n",
              static_cast<long long>(rGq.cutFaces), rGq.ident, rGq.identRef);
  PECLET_AMR_CHECK_EQ(rGq.cutFaces, Index(0));
  PECLET_AMR_CHECK(rGq.ident <= 1e-14 * std::max(1.0, rGq.identRef));

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
  const Res rGh = run(tgh, R, Vec<3>{cc, cc, cc}, 30, /*ghost=*/true);
  const double dCellGh = rGh.dCell, dFaceGh = rGh.dFace;
  // Orders of magnitude above the aperture path's residual, where the same call IS the residual.
  PECLET_AMR_CHECK(dFaceGh > 1e3 * dFace30);
  // ...and indistinguishable from the cell divergence, i.e. it adds nothing over divergence_norm.
  PECLET_AMR_CHECK(dFaceGh > 0.5 * dCellGh);
  // If either check ever fails, either the overlay delta has been folded into divFaceNorm (good --
  // retire this block and the caveats on AmrFlow::divNormFace and its binding) or the ghost
  // constraint has changed (investigate before re-blessing).

  // (4) THE CUT BAND *ON* THE LEVEL BOUNDARY (docs/amr_cf_flux_gate.md). Everything above keeps
  // the cut cells uniformly finest, so no cut cell has a 2:1 face and the C/F treatment never
  // meets the wall. refineToSdf(band=0) creates the finest level ONLY where the surface actually
  // cuts, so the 2:1 interface coincides with the cut band: 2:1 sub-faces with a cut cell on one
  // side and a regular one on the other. That is a single flux shared by two cells whose books
  // disagreed under the old per-ROW gate -- the regular cell booked the quadratic C/F correction
  // in its divergence constraint, the cut cell did not, and uf carried it for both. The imbalance
  // is proportional to the VELOCITY, not to phi, so it does not decay at steady state: measured
  // ||D(d_vel) - d_cfDiv|| = 2.018e-01 dominating ||div(uf)|| = 2.016e-01 on this very mesh.
  //
  // The fix is that the quadratic face value applies iff BOTH incident cells are regular fluid,
  // through ONE emitter that feeds the divergence RHS, the ABC gradient substitution and uf alike.
  // The three assertions below are the gates: the configuration is PRESENT (census > 0), uf is a
  // conservative flux again (the graded bar of case 2), and rule (I) holds identically.
  const unsigned lb = 2;
  const double Nb = static_cast<double>(1L << (lb + 3));  // 8 roots * 2^lmax = 32 fine
  BO tb(IVec<3>{8, 8, 8}, lb);
  AmrGeometry<3> geob;
  geob.setIsotropic(1.0);
  const double Rb = std::pow(0.125 * 3.0 / (4.0 * M_PI), 1.0 / 3.0) * Nb, cb = Nb / 2.0;
  auto sdfB = [&](const Vec<3>& p) {
    const double dx = p[0] - cb, dy = p[1] - cb, dz = p[2] - cb;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - Rb;
  };
  refineToSdf(tb, geob, sdfB, /*target_level=*/0, /*band=*/0.0, /*balance=*/true);
  PECLET_AMR_CHECK(tb.isBalanced());
  const Res rb = run(tb, Rb, Vec<3>{cb, cb, cb}, 30, /*ghost=*/false, /*cf=*/1);
  std::printf(
      "[face-field] band=0 (cut band ON the level boundary), aperture cf=1: cfCutFaces %lld | "
      "cell %.4e face %.4e | identity %.4e (ref %.4e)\n",
      static_cast<long long>(rb.cutFaces), rb.dCell, rb.dFace, rb.ident, rb.identRef);
  // The mesh must actually contain the configuration, or this case gates nothing.
  PECLET_AMR_CHECK(rb.cutFaces > 0);
  PECLET_AMR_CHECK(rb.dFace < 0.01 * rb.dCell);                      // the graded bar of case (2)
  PECLET_AMR_CHECK(rb.ident <= 1e-14 * std::max(1.0, rb.identRef));  // rule (I)

  // (4b) the same mesh on the production GHOST closure. band=0 is the mixed-level cut band, so the
  // valid ghost closure there is the SAMPLED one (the classic overlay's +-2 reach would cross a
  // 2:1 boundary and setGhostProjection would throw). divNormFace as a whole says nothing under
  // the ghost scheme (case 3), but restricted to REGULAR fluid rows it is the solve residual --
  // and those are exactly the rows that carried the 2e-1-class mismatch before the gate changed.
  const Res rbg = run(tb, Rb, Vec<3>{cb, cb, cb}, 30, /*ghost=*/true, /*cf=*/1, /*sampled=*/true);
  std::printf(
      "[face-field] band=0 sampled ghost cf=1: cfCutFaces %lld | regular rows cell %.4e face "
      "%.4e | identity %.4e (ref %.4e)\n",
      static_cast<long long>(rbg.cutFaces), rbg.dCellReg, rbg.dFaceReg, rbg.ident, rbg.identRef);
  PECLET_AMR_CHECK(rbg.cutFaces > 0);
  PECLET_AMR_CHECK(rbg.dFaceReg < 0.01 * rbg.dCellReg);
  PECLET_AMR_CHECK(rbg.ident <= 1e-14 * std::max(1.0, rbg.identRef));
}

}  // namespace

int main() {
  run_test();
  if (peclet::amr::test::g_failures == 0)
    std::printf("OK\n");
  return peclet::amr::test::g_failures == 0 ? 0 : 1;
}
