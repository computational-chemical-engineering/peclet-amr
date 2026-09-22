// C/F interface schemes for the collocated flow operators (cf_scheme.hpp): a-priori truncation
// on a graded (2:1) mesh with smooth manufactured fields, measured at the level-boundary rows.
//
//  (1) the L delta (buildCfLapDelta, factor 1) must reproduce the VALIDATED
//      applyLaplacianQuad − applyLaplacian to round-off (anti-drift lock vs P5b);
//  (2) D (½/½ face-average divergence): standard is low-order at C/F rows (the average is
//      normally offset from the face); the scheme's distance-weighted + tangential-corrected
//      value restores ~2nd order. The quad D stays exactly conservative (Σ V·D = 0, periodic).
//  (3) G (ABC cell gradient): standard ½/½ side recombination misses the cell center at C/F
//      rows; the scheme's reweighting + coarse* substitution restores ~2nd order.
//  (4) momentum ∇² (the velocity operator): the tangential-only flux correction restores
//      ~2nd order at C/F rows (the P5b result, now on the velocity path).
//
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "peclet/amr/block_octree.hpp"
#include "peclet/amr/cf_scheme.hpp"
#include "peclet/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Code = BO::Code;

constexpr double kPi = 3.14159265358979323846;

struct Geo {
  BO t;
  AmrPoisson<3, 21> ap;
  double h;   // the axis-0 spacing (== hv[0])
  Vec<3> hv;  // the per-axis root spacing (anisotropic meshes use all three)
  Index n;
  std::vector<Vec<3>> cen;
  std::vector<char> cfRow;  // 1 = adjacent to a 2:1 face
};

// `aspect` scales the root spacing per axis: {1,1,1} is the cubic mesh every order gate below
// runs on, anything else an ANISOTROPIC octree (section 7).
Geo buildGeo(long N, Vec<3> aspect = Vec<3>{1.0, 1.0, 1.0}) {
  Geo g;
  unsigned L = 0;
  while ((1L << L) < N)
    ++L;
  g.t = BO(IVec<3>{1, 1, 1}, L);
  // uniform to level 1, then the lower octant to level 0 (finest), 2:1 balanced.
  for (unsigned k = 0; k + 1 < L; ++k)
    g.t.refineIf([](Code, unsigned l) { return l > 1; });
  const long half = N / 2;
  g.t.refineIf([&](Code c, unsigned l) {
    if (l != 1)
      return false;
    auto o = BO::M::from_code(c).decode();
    return o[0] < half && o[1] < half && o[2] < half;
  });
  g.t.balance2to1();
  g.h = 1.0 / static_cast<double>(N);
  for (int d = 0; d < 3; ++d)
    g.hv[d] = g.h * aspect[d];
  g.ap.init(g.t, g.hv);
  g.n = g.t.numLeaves();
  g.cen.resize(static_cast<std::size_t>(g.n));
  g.cfRow.assign(static_cast<std::size_t>(g.n), 0);
  for (Index i = 0; i < g.n; ++i) {
    auto b = g.t.bounds(i);
    double s = static_cast<double>(Index(1) << g.t.level(i));
    for (int d = 0; d < 3; ++d)
      g.cen[static_cast<std::size_t>(i)][d] = (static_cast<double>(b[0][d]) + 0.5 * s) * g.hv[d];
    const unsigned Li = g.t.level(i);
    g.ap.forEachFaceFull(i, [&](Index j, int, int, double, double, double) {
      if (g.t.level(j) != Li)
        g.cfRow[static_cast<std::size_t>(i)] = 1;
    });
  }
  return g;
}

double phiMan(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return std::sin(tp * p[0]) * std::cos(tp * p[1]) + std::sin(tp * p[1]) * std::cos(tp * p[2]) +
         std::sin(tp * p[2]) * std::cos(tp * p[0]);
}
Vec<3> phiManGrad(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return {tp * std::cos(tp * p[0]) * std::cos(tp * p[1]) -
              tp * std::sin(tp * p[2]) * std::sin(tp * p[0]),
          -tp * std::sin(tp * p[0]) * std::sin(tp * p[1]) +
              tp * std::cos(tp * p[1]) * std::cos(tp * p[2]),
          -tp * std::sin(tp * p[1]) * std::sin(tp * p[2]) +
              tp * std::cos(tp * p[2]) * std::cos(tp * p[0])};
}
double phiManLap(const Vec<3>& p) {
  return -2.0 * (2.0 * kPi) * (2.0 * kPi) * phiMan(p);
}
Vec<3> velMan(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return {std::sin(tp * p[0]) * std::cos(tp * p[1]), std::sin(tp * p[1]) * std::cos(tp * p[2]),
          std::sin(tp * p[2]) * std::cos(tp * p[0])};
}
double velManDiv(const Vec<3>& p) {
  const double tp = 2.0 * kPi;
  return tp *
         (std::cos(tp * p[0]) * std::cos(tp * p[1]) + std::cos(tp * p[1]) * std::cos(tp * p[2]) +
          std::cos(tp * p[2]) * std::cos(tp * p[0]));
}

// Standard ½/½ face-average divergence (the oracle::AmrFlow::divergence form, α=1).
void divStd(const Geo& g, const std::array<std::vector<double>, 3>& u, std::vector<double>& d) {
  d.assign(static_cast<std::size_t>(g.n), 0.0);
  for (Index i = 0; i < g.n; ++i) {
    double acc = 0.0;
    g.ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double) {
      acc += area * dir * 0.5 *
             (u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(i)] +
              u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)]);
    });
    d[static_cast<std::size_t>(i)] = acc / g.ap.cellVolume(i);
  }
}

// RULE (I) of docs/amr_cf_flux_gate.md §4, measured on the manufactured field with NO solve.
// The divergence RHS and the advecting face field must carry the SAME per-face value delta, so
// that D_std(Δuf) ≡ Δ_cfDiv entry by entry — the property that makes uf a conservative flux
// (D_std uf = rhs − Lφ = the solver residual). Both CSRs come out of the one emitter
// cfAppendFaceValueDelta, so this is an algebraic identity on ANY mesh and ANY field, and it is
// the trip-wire for the two builders drifting apart again. Returns {‖D_std(Δuf) − Δ_cfDiv‖₂,
// ‖Δ_cfDiv‖₂}.
std::pair<double, double> cfFluxIdentity(const Geo& g,
                                         const std::array<std::vector<double>, 3>& u) {
  auto all = [](Index) { return true; };
  const CfCompCsr dd = buildCfDivDelta(g.ap, g.t, all, all, CfScheme::quadratic);
  const CfUfDelta ufd = buildCfUfDelta(g.ap, g.t, all, all, CfScheme::quadratic);
  // Δuf on every forEachFaceFull slot (cell-major, the CSR's own row numbering).
  const Index nSlots = static_cast<Index>(ufd.vel.start.size()) - 1;
  std::vector<double> duf(static_cast<std::size_t>(nSlots), 0.0);
  cfApplyCompHost(ufd.vel, u, duf);
  // D_std of it: invV · Σ α·A·dir·Δuf, the same sweep in the same order.
  std::vector<double> dFromUf(static_cast<std::size_t>(g.n), 0.0);
  Index slot = 0;
  for (Index i = 0; i < g.n; ++i) {
    double acc = 0.0;
    g.ap.forEachFaceFull(i, [&](Index, int, int dir, double area, double, double alpha) {
      acc += alpha * area * static_cast<double>(dir) * duf[static_cast<std::size_t>(slot)];
      ++slot;
    });
    dFromUf[static_cast<std::size_t>(i)] = acc / g.ap.cellVolume(i);
  }
  std::vector<double> dCf(static_cast<std::size_t>(g.n), 0.0);
  cfApplyCompHost(dd, u, dCf);
  double e2 = 0.0, r2 = 0.0;
  for (Index i = 0; i < g.n; ++i) {
    const double d = dFromUf[static_cast<std::size_t>(i)] - dCf[static_cast<std::size_t>(i)];
    e2 += d * d;
    r2 += dCf[static_cast<std::size_t>(i)] * dCf[static_cast<std::size_t>(i)];
  }
  return {std::sqrt(e2), std::sqrt(r2)};
}

// Standard ABC cell gradient (the oracle::AmrFlow::gradOf form, α=1).
double gradStd(const Geo& g, const std::vector<double>& f, Index i, int c) {
  const double fi = f[static_cast<std::size_t>(i)];
  double gp = 0, gm = 0;
  int np = 0, nm = 0;
  g.ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double) {
    if (axis != c)
      return;
    double gg = (dir > 0) ? (f[static_cast<std::size_t>(j)] - fi) / dist
                          : (fi - f[static_cast<std::size_t>(j)]) / dist;
    if (dir > 0) {
      gp += gg;
      ++np;
    } else {
      gm += gg;
      ++nm;
    }
  });
  return 0.5 * ((np ? gp / np : 0.0) + (nm ? gm / nm : 0.0));
}

double orderOf(double prev, double cur, long Nprev, long N) {
  return std::log2(prev / cur) / std::log2(static_cast<double>(N) / Nprev);
}

void run() {
  auto all = [](Index) { return true; };
  std::printf("%5s | %10s %6s %10s %6s | %10s %6s %10s %6s | %10s %6s %10s %6s\n", "N", "D_std",
              "ord", "D_quad", "ord", "G_std", "ord", "G_quad", "ord", "L_std", "ord", "L_quad",
              "ord");
  double pDs = 0, pDq = 0, pGs = 0, pGq = 0, pLs = 0, pLq = 0;
  long pN = 0;
  double oDq = 0, oGq = 0, oLq = 0, oDs = 0, oGs = 0, oLs = 0;
  for (long N : {16L, 32L, 64L}) {
    Geo g = buildGeo(N);
    // (1) L delta == applyLaplacianQuad − applyLaplacian (bit-parity vs the validated P5b impl).
    std::vector<double> phi(static_cast<std::size_t>(g.n));
    for (Index i = 0; i < g.n; ++i)
      phi[static_cast<std::size_t>(i)] = phiMan(g.cen[static_cast<std::size_t>(i)]);
    {
      std::vector<double> ls, lq, dl(static_cast<std::size_t>(g.n), 0.0);
      g.ap.applyLaplacian(phi, ls);
      g.ap.applyLaplacianQuad(phi, lq);
      CfCsr cs = buildCfLapDelta(g.ap, g.t, 1.0, all, all, CfScheme::quadratic);
      cfApplyHost(cs, phi, dl);
      double scale = 0.0, dmax = 0.0;
      for (Index i = 0; i < g.n; ++i) {
        scale = std::max(scale, std::fabs(lq[static_cast<std::size_t>(i)]));
        dmax = std::max(
            dmax, std::fabs(dl[static_cast<std::size_t>(i)] -
                            (lq[static_cast<std::size_t>(i)] - ls[static_cast<std::size_t>(i)])));
      }
      PECLET_AMR_CHECK(dmax < 1e-12 * scale);  // the delta IS the P5b quad correction
    }
    // fields
    std::array<std::vector<double>, 3> u;
    for (int c = 0; c < 3; ++c) {
      u[static_cast<std::size_t>(c)].resize(static_cast<std::size_t>(g.n));
      for (Index i = 0; i < g.n; ++i)
        u[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] =
            velMan(g.cen[static_cast<std::size_t>(i)])[c];
    }
    // (2) divergence
    std::vector<double> ds;
    divStd(g, u, ds);
    std::vector<double> dq = ds;
    CfCompCsr dd = buildCfDivDelta(g.ap, g.t, all, all, CfScheme::quadratic);
    cfApplyCompHost(dd, u, dq);
    double eDs = 0, eDq = 0, consQ = 0;
    for (Index i = 0; i < g.n; ++i) {
      const double ex = velManDiv(g.cen[static_cast<std::size_t>(i)]);
      consQ += g.ap.cellVolume(i) * dq[static_cast<std::size_t>(i)];
      if (!g.cfRow[static_cast<std::size_t>(i)])
        continue;
      eDs = std::max(eDs, std::fabs(ds[static_cast<std::size_t>(i)] - ex));
      eDq = std::max(eDq, std::fabs(dq[static_cast<std::size_t>(i)] - ex));
    }
    PECLET_AMR_CHECK(std::fabs(consQ) < 1e-10);  // quad D exactly conservative (periodic)
    // (3) gradient
    std::array<CfCsr, 3> gd = buildCfGradDelta(g.ap, g.t, all, all, CfScheme::quadratic);
    std::array<std::vector<double>, 3> gq;
    for (int c = 0; c < 3; ++c) {
      gq[static_cast<std::size_t>(c)].assign(static_cast<std::size_t>(g.n), 0.0);
      cfApplyHost(gd[static_cast<std::size_t>(c)], phi, gq[static_cast<std::size_t>(c)]);
    }
    double eGs = 0, eGq = 0;
    for (Index i = 0; i < g.n; ++i) {
      if (!g.cfRow[static_cast<std::size_t>(i)])
        continue;
      const Vec<3> ge = phiManGrad(g.cen[static_cast<std::size_t>(i)]);
      for (int c = 0; c < 3; ++c) {
        const double gs = gradStd(g, phi, i, c);
        eGs = std::max(eGs, std::fabs(gs - ge[c]));
        eGq = std::max(
            eGq,
            std::fabs(gs + gq[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] - ge[c]));
      }
    }
    // (4) momentum ∇² (the velocity operator): SOLUTION-level order for the Helmholtz solve
    // (I − ∇²)u = f, manufactured. The quad flux is conservative with an O(1) POINTWISE row
    // truncation at C/F faces (the sample point of the two-point C/F gradient is normally
    // offset from the face) — the flux error telescopes and the SOLUTION is 2nd order (the P5b
    // result); pointwise operator truncation is the wrong metric here.
    CfCsr lm = buildCfLapDelta(g.ap, g.t, 1.0, all, all, CfScheme::quadratic);
    auto solveHelmholtz = [&](bool quad) {
      // u solved by unpreconditioned-BiCGStab-with-Jacobi on (I − L[ − Δ]) u = b.
      const std::size_t ns = static_cast<std::size_t>(g.n);
      std::vector<double> b(ns);
      for (Index i = 0; i < g.n; ++i)
        b[static_cast<std::size_t>(i)] = phiMan(g.cen[static_cast<std::size_t>(i)]) -
                                         phiManLap(g.cen[static_cast<std::size_t>(i)]);
      auto applyA = [&](const std::vector<double>& x, std::vector<double>& y) {
        g.ap.applyLaplacian(x, y);
        if (quad)
          cfApplyHost(lm, x, y);
        for (std::size_t p = 0; p < ns; ++p)
          y[p] = x[p] - y[p];
      };
      auto dot = [&](const std::vector<double>& a, const std::vector<double>& c) {
        double s = 0;
        for (std::size_t p = 0; p < ns; ++p)
          s += a[p] * c[p];
        return s;
      };
      std::vector<double> x(ns, 0.0), r(ns), rh(ns), p(ns, 0.0), v(ns, 0.0), s(ns), tt(ns);
      applyA(x, r);
      for (std::size_t q = 0; q < ns; ++q)
        r[q] = b[q] - r[q];
      rh = r;
      const double r0 = std::sqrt(dot(r, r));
      double rho = 1, alpha = 1, omega = 1;
      for (int it = 0; it < 4000; ++it) {
        const double rhoN = dot(rh, r);
        if (rhoN == 0)
          break;
        const double beta = (rhoN / rho) * (alpha / omega);
        for (std::size_t q = 0; q < ns; ++q)
          p[q] = r[q] + beta * (p[q] - omega * v[q]);
        applyA(p, v);
        alpha = rhoN / dot(rh, v);
        for (std::size_t q = 0; q < ns; ++q)
          s[q] = r[q] - alpha * v[q];
        applyA(s, tt);
        const double t2 = dot(tt, tt);
        omega = (t2 != 0) ? dot(tt, s) / t2 : 0;
        for (std::size_t q = 0; q < ns; ++q) {
          x[q] += alpha * p[q] + omega * s[q];
          r[q] = s[q] - omega * tt[q];
        }
        if (std::sqrt(dot(r, r)) < 1e-11 * r0)
          break;
        rho = rhoN;
        if (omega == 0)
          break;
      }
      return x;
    };
    std::vector<double> uS = solveHelmholtz(false), uQ = solveHelmholtz(true);
    double eLs = 0, eLq = 0;
    for (Index i = 0; i < g.n; ++i) {
      if (!g.cfRow[static_cast<std::size_t>(i)])
        continue;
      const double ex = phiMan(g.cen[static_cast<std::size_t>(i)]);
      eLs = std::max(eLs, std::fabs(uS[static_cast<std::size_t>(i)] - ex));
      eLq = std::max(eLq, std::fabs(uQ[static_cast<std::size_t>(i)] - ex));
    }
    // (5) uf face field: the face-AVERAGE half at the 2:1 sub-faces. uf_k = ½(u_i+u_j) −
    // (φ₊−φ₋)/d, and the C/F delta the solver builds and applies is the face-average half ALONE
    // (the φ half stopped being applied at 1b0d5b5 and has since been deleted). That half is the
    // steady advecting velocity, and it is what is gated here: ~2nd order at the sub-face
    // centroid, where the standard ½/½ average — whose sample point is normally offset at a 2:1
    // face — is not.
    {
      // (regularOk, fluidOk): this mesh has no solid, so `all` is both — and the per-FACE
      // C/F gate (docs/amr_cf_flux_gate.md) is therefore inert here, as it is on every mesh
      // whose cut cells are not at a level boundary.
      CfUfDelta ufd = buildCfUfDelta(g.ap, g.t, all, all, CfScheme::quadratic);
      Index nSlots = 0;
      for (Index i = 0; i < g.n; ++i)
        g.ap.forEachFaceFull(i, [&](Index, int, int, double, double, double) { ++nSlots; });
      std::vector<Vec<3>> fc(static_cast<std::size_t>(nSlots));
      std::vector<int8_t> fAxis(static_cast<std::size_t>(nSlots));
      std::vector<char> fCf(static_cast<std::size_t>(nSlots), 0);
      Index slot = 0;
      for (Index i = 0; i < g.n; ++i) {
        const unsigned Li = g.t.level(i);
        g.ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double, double) {
          const unsigned Lj = g.t.level(j);
          fCf[static_cast<std::size_t>(slot)] = (Lj != Li) ? 1 : 0;
          fAxis[static_cast<std::size_t>(slot)] = static_cast<int8_t>(axis);
          // sub-face centroid = the FINER cell's face center toward the other cell.
          const Index fine = (Lj < Li) ? j : i;
          const double sgn = (fine == i) ? 1.0 : -1.0;  // dir points i→j
          Vec<3> c = g.cen[static_cast<std::size_t>(fine)];
          c[axis] += sgn * dir * 0.5 * g.ap.cellWidth(fine);
          fc[static_cast<std::size_t>(slot)] = c;
          ++slot;
        });
      }
      // WHY THERE IS NO GATE ON THE WHOLE uf (it had one, `oUq >= 0.9`, until 2026-09-22; it
      // reconstructed a φ overlay the solver no longer applies and asserted a convergence the
      // scheme never promised). With the φ half gone, the pointwise error of the WHOLE uf at a
      // C/F sub-face is O(1)·|∇_tφ| BY DESIGN of the standard matrix — not O(h). uf's face
      // gradient is the compact two-point (φ_C − φ_F)/d, and at a 2:1 face the two centres are
      // offset TANGENTIALLY by h/2 on each tangential axis over a normal distance 1.5h, so
      //     (φ_C − φ_F)/d = ∂_nφ ± (1/3)·∂_{t1}φ ± (1/3)·∂_{t2}φ + O(h).
      // The tangential leak does not shrink with h. Here |∂φ| ≤ 2π, so the error tends to
      // (2/3)·2π = 4.19 — exactly what the measured whole-uf error converges to: 4.547e+00 →
      // 4.319e+00 → 4.224e+00 at N = 16/32/64 (order 0.07, 0.03). That is a property of
      // L = D_std·G_std (the recorded decision "the pressure matrix stays standard"), and φ → 0
      // at the projection's fixed point, so it never touches the steady answer. The property the
      // solver DOES have is rule (I) — D_std(Δuf) ≡ Δ_cfDiv — gated in section (7).
      std::vector<double> avS(static_cast<std::size_t>(nSlots), 0.0), avQ;
      {
        Index k = 0;
        for (Index i = 0; i < g.n; ++i)
          g.ap.forEachFaceFull(i, [&](Index j, int axis, int, double, double, double) {
            avS[static_cast<std::size_t>(k++)] =
                0.5 * (u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(i)] +
                       u[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)]);
          });
        avQ = avS;
        cfApplyCompHost(ufd.vel, u, avQ);
      }
      double eAs = 0, eAq = 0;
      for (Index k = 0; k < nSlots; ++k) {
        if (!fCf[static_cast<std::size_t>(k)])
          continue;
        const Vec<3>& c = fc[static_cast<std::size_t>(k)];
        const int a = fAxis[static_cast<std::size_t>(k)];
        const double exA = velMan(c)[a];
        eAs = std::max(eAs, std::fabs(avS[static_cast<std::size_t>(k)] - exA));
        eAq = std::max(eAq, std::fabs(avQ[static_cast<std::size_t>(k)] - exA));
      }
      static double pAq = 0, pAs = 0;
      static double oAq = 0, oAs = 0;
      oAq = pN ? orderOf(pAq, eAq, pN, N) : 0;
      oAs = pN ? orderOf(pAs, eAs, pN, N) : 0;
      // Rule (I) on the SAME mesh and field, no solve: the identity must hold to round-off.
      const auto id = cfFluxIdentity(g, u);
      std::printf(
          "      | uf avg: std %.3e ord %5.2f quad %.3e ord %5.2f | rule (I) %.3e (ref %.3e)\n",
          eAs, oAs, eAq, oAq, id.first, id.second);
      pAq = eAq;
      pAs = eAs;
      PECLET_AMR_CHECK(id.first <= 1e-14 * std::max(1.0, id.second));
      if (N == 64) {
        PECLET_AMR_CHECK(oAq >= 1.7);        // steady advecting velocity ~2nd order
        PECLET_AMR_CHECK(oAs <= oAq - 0.5);  // standard average is lower order
      }
    }
    oDs = pN ? orderOf(pDs, eDs, pN, N) : 0;
    oDq = pN ? orderOf(pDq, eDq, pN, N) : 0;
    oGs = pN ? orderOf(pGs, eGs, pN, N) : 0;
    oGq = pN ? orderOf(pGq, eGq, pN, N) : 0;
    oLs = pN ? orderOf(pLs, eLs, pN, N) : 0;
    oLq = pN ? orderOf(pLq, eLq, pN, N) : 0;
    std::printf(
        "%5ld | %10.3e %6.2f %10.3e %6.2f | %10.3e %6.2f %10.3e %6.2f | %10.3e %6.2f "
        "%10.3e %6.2f\n",
        N, eDs, oDs, eDq, oDq, eGs, oGs, eGq, oGq, eLs, oLs, eLq, oLq);
    pDs = eDs;
    pDq = eDq;
    pGs = eGs;
    pGq = eGq;
    pLs = eLs;
    pLq = eLq;
    pN = N;
  }
  // (6) island corners: a fine BALL in a coarse sea curves, so coarse cells at the interface
  // have FINER tangential neighbours — the rows where P5b's coarseStar falls back to the raw
  // value (locally 1st-order). The upgraded stencil samples the fine cover (2^Dim-child volume
  // average) instead. Assert the branch actually fires (the delta differs from
  // applyLaplacianQuad, which still skips those axes) and that it does not degrade — and
  // improves the max — the C/F truncation of the gradient.
  {
    std::printf("  [corners] fine-ball mesh:\n");
    double pC = 0, pP = 0;
    long pNc = 0;
    for (long N : {16L, 32L, 64L}) {
      Geo g;
      unsigned L = 0;
      while ((1L << L) < N)
        ++L;
      g.t = BO(IVec<3>{1, 1, 1}, L);
      for (unsigned k = 0; k + 1 < L; ++k)
        g.t.refineIf([](Code, unsigned l) { return l > 1; });
      const double rad = 0.27, cx = 0.51, cy = 0.49, cz = 0.52;
      g.h = 1.0 / static_cast<double>(N);
      g.t.refineIf([&](Code c, unsigned l) {
        if (l != 1)
          return false;
        auto o = BO::M::from_code(c).decode();
        const double x = (static_cast<double>(o[0]) + 1.0) * g.h;  // level-1 cell center
        const double y = (static_cast<double>(o[1]) + 1.0) * g.h;
        const double z = (static_cast<double>(o[2]) + 1.0) * g.h;
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz) < rad * rad;
      });
      g.t.balance2to1();
      g.ap.init(g.t, g.h);
      g.n = g.t.numLeaves();
      g.cen.resize(static_cast<std::size_t>(g.n));
      g.cfRow.assign(static_cast<std::size_t>(g.n), 0);
      for (Index i = 0; i < g.n; ++i) {
        auto b = g.t.bounds(i);
        double s = static_cast<double>(Index(1) << g.t.level(i));
        for (int d = 0; d < 3; ++d)
          g.cen[static_cast<std::size_t>(i)][d] = (static_cast<double>(b[0][d]) + 0.5 * s) * g.h;
        const unsigned Li = g.t.level(i);
        g.ap.forEachFaceFull(i, [&](Index j, int, int, double, double, double) {
          if (g.t.level(j) != Li)
            g.cfRow[static_cast<std::size_t>(i)] = 1;
        });
      }
      std::vector<double> phi(static_cast<std::size_t>(g.n));
      for (Index i = 0; i < g.n; ++i)
        phi[static_cast<std::size_t>(i)] = phiMan(g.cen[static_cast<std::size_t>(i)]);
      std::vector<double> ls, lq, dl(static_cast<std::size_t>(g.n), 0.0);
      g.ap.applyLaplacian(phi, ls);
      g.ap.applyLaplacianQuad(phi, lq);  // the P5b form (corner axes fall back)
      CfCsr cs = buildCfLapDelta(g.ap, g.t, 1.0, all, all, CfScheme::quadratic);
      cfApplyHost(cs, phi, dl);
      double branch = 0.0, eCorner = 0.0, eP5b = 0.0;
      for (Index i = 0; i < g.n; ++i) {
        branch = std::max(
            branch, std::fabs(dl[static_cast<std::size_t>(i)] -
                              (lq[static_cast<std::size_t>(i)] - ls[static_cast<std::size_t>(i)])));
        if (!g.cfRow[static_cast<std::size_t>(i)])
          continue;
        const double ex = phiManLap(g.cen[static_cast<std::size_t>(i)]);
        eCorner = std::max(eCorner, std::fabs(ls[static_cast<std::size_t>(i)] +
                                              dl[static_cast<std::size_t>(i)] - ex));
        eP5b = std::max(eP5b, std::fabs(lq[static_cast<std::size_t>(i)] - ex));
      }
      const double oC = pNc ? orderOf(pC, eCorner, pNc, N) : 0;
      const double oP = pNc ? orderOf(pP, eP5b, pNc, N) : 0;
      std::printf(
          "  [corners] N=%3ld branch |delta-p5b| = %.2e; L trunc: upgraded %.3e "
          "(ord %5.2f) vs p5b-fallback %.3e (ord %5.2f)\n",
          N, branch, eCorner, oC, eP5b, oP);
      PECLET_AMR_CHECK(branch > 0.0);            // the corner branch fires on this mesh
      PECLET_AMR_CHECK(eCorner <= eP5b * 1.02);  // upgraded stencil never worse, at worst ties
      pC = eCorner;
      pP = eP5b;
      pNc = N;
    }
  }

  // (7) RULE (I) ON AN ANISOTROPIC GRADED OCTREE — the only gate this repo has on the
  // anisotropic C/F path, and it is a gate on a LIVE BUG that was fixed by the shared emitter
  // (docs/amr_cf_flux_gate.md §6.2, cf_scheme.hpp::cfAppendFaceValueDelta).
  //
  // Until 2026-09-22 buildCfUfDelta took BOTH cell widths on axis 0 (`cellWidth(i)`, the cubic
  // spelling) and divided them by `forEachFaceFull`'s `dist`, which is on the FACE axis. On
  // h0 = (1, ½, 2) a y-directed 2:1 sub-face therefore got wF = (½·1·2)/(1.5·½) = 4/3 and
  // wC = (½·1)/(1.5·½) = 2/3 — wF + wC = 2, i.e. roughly DOUBLE the velocity on every off-axis
  // C/F sub-face of an anisotropic graded mesh. buildCfDivDelta was dimensionally right (it took
  // H, h AND d = ½(H+h) all on axis 0, so only their ratios entered), so the two books disagreed
  // on EVERY C/F sub-face, cut or not — an O(1) violation of rule (I) with no cut cell in sight.
  // Measured here with the pre-emitter builders of `main` at 2426ef4: rule (I) = 7.447e+02
  // against a 7.766e+01 reference — O(1), and TEN TIMES the delta it is supposed to equal. The
  // emitter takes both widths on the face-normal axis, and the identity is exact (4.399e-15).
  // The cubic arm below is the control: it reads 5.286e-15 with BOTH the old and the new
  // builders, bit for bit, which is the inertness the byte gate asserts globally.
  //
  // Nothing else in the suite covers this: the byte gate's scenarios are all extent=[1,1,1] and
  // test_amr_drag's dragKAniso is uniformly refined (no 2:1 face at all).
  {
    std::printf("  [aniso] rule (I) on anisotropic graded octrees:\n");
    for (const Vec<3>& aspect : {Vec<3>{1.0, 0.5, 2.0}, Vec<3>{1.0, 1.0, 1.0}}) {
      Geo ga = buildGeo(32, aspect);
      std::array<std::vector<double>, 3> ua;
      for (int c = 0; c < 3; ++c) {
        ua[static_cast<std::size_t>(c)].resize(static_cast<std::size_t>(ga.n));
        for (Index i = 0; i < ga.n; ++i)
          ua[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] =
              velMan(ga.cen[static_cast<std::size_t>(i)])[c];
      }
      // The mesh must actually carry 2:1 faces on the squashed and stretched axes, or the arm
      // gates nothing: count the C/F sub-faces per axis.
      Index cfPerAxis[3] = {0, 0, 0};
      for (Index i = 0; i < ga.n; ++i) {
        const unsigned Li = ga.t.level(i);
        ga.ap.forEachFaceFull(i, [&](Index j, int axis, int, double, double, double) {
          if (ga.ap.levelOf(j) != Li)
            ++cfPerAxis[axis];
        });
      }
      const auto id = cfFluxIdentity(ga, ua);
      std::printf(
          "  [aniso] h0 = (%.3g, %.3g, %.3g)  C/F sub-face slots per axis %lld/%lld/%lld  "
          "rule (I) %.3e (ref %.3e)\n",
          aspect[0], aspect[1], aspect[2], static_cast<long long>(cfPerAxis[0]),
          static_cast<long long>(cfPerAxis[1]), static_cast<long long>(cfPerAxis[2]), id.first,
          id.second);
      for (int a = 0; a < 3; ++a)
        PECLET_AMR_CHECK(cfPerAxis[a] > 0);
      PECLET_AMR_CHECK(id.second > 1.0);  // the delta is O(1): the identity below is not trivial
      PECLET_AMR_CHECK(id.first <= 1e-14 * id.second);
    }
  }

  // Gates: the scheme restores ~2nd order at C/F rows for all three operators; the standard
  // treatment is measurably lower order there.
  PECLET_AMR_CHECK(oDq >= 1.7);
  PECLET_AMR_CHECK(oGq >= 1.7);
  PECLET_AMR_CHECK(oLq >= 1.7);
  PECLET_AMR_CHECK(oDs <= oDq - 0.5);
  PECLET_AMR_CHECK(oGs <= oGq - 0.5);
  PECLET_AMR_CHECK(oLs <= oLq - 0.5);
}

}  // namespace

int main() {
  run();
  PECLET_AMR_RETURN_TEST_RESULT();
}
