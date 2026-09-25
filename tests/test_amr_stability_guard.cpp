// The collocated STABILITY GUARD — no face-acceleration "kick" on amr's collocated path.
//
// WHAT IT GUARDS. On a collocated grid the lagged pressure gradient and every force must enter
// the momentum equation INSIDE the implicit (viscous) predictor, never as a face acceleration
// added after the implicit solve (the Basilisk centered.h "kick", uf += dt*a, whatever it is
// called). Added after A^{-1}, -dt G P^n / rho lies exactly in the range the projection removes,
// so P^n never reaches u: the step is non-incremental Chorin (a dt-dependent steady state), and
// the rotational update -kappa*div then acts on P as an explicit diffusion,
// P^{n+1} = -4 kappa dt S P^n / (rho h^2), S = sum_axes sin^2(theta/2): -12 kappa dt/(rho h^2) at
// (pi,pi,pi), unstable once mu dt/(rho h^2) > 1/12. flow's collocated variable-density rung V8
// shipped exactly that in 2026-09 (measured -12.0000) although a text prohibition already
// existed, because nothing TESTED the signature. amr is predictor-form today (flow.hpp step():
// -grad p^n and f in momRhs, rotational presUpdate), so this test PASSES on amr main; it is here
// to catch the regression when amr gains variable density / VoF (flow
// doc/collocated_varrho_forces.md §13) or anyone "saves a solve" by moving a force after it.
//
// Design: flow doc/collocated_varrho_forces.md §7 (§7.6 is the protocol, WO-G1 this amr
// counterpart of flow's tests/python/test_collocated_stability_guard.py). Register: suite-wide
// "Collocated pressure and forces stay in the implicit predictor — never a face acceleration after
// the viscous solve" (suite docs/decisions/suite-wide.md). It is a GATE, not a grep: the defect
// is a property of the discrete step and comes back under any name.
//
// PROTOCOL (Stokes, advection off: the step is affine, so the seeded-minus-twin difference evolves
// exactly by the homogeneous step operator). mu = rho = 1, finest h = 1, so mu dt/(rho h^2) = dt.
// Four meshes: lmax = 0 (uniform 8^3) and one refined region (root 8^3 at lmax = 1, a 16^3-finest
// patch), each all-fluid and with an immersed sphere (cut cells, the AUTO ghost projection).
//
//  G2 growth. P and u are seeded with eps*(-1)^(i+j+k) + eps*random (i,j,k the leaf's index at its
//     own level) and marched 100 steps at dt in {0.1, 1, 10, 100} beside the unseeded twin.
//     Measured: A_P = |(pi,pi,pi) amplitude of P - P_twin| (volume-weighted projection on the
//     per-leaf checkerboard, fluid-mean removed; on the uniform mesh the FFT coefficient) and
//     U = max|du - <du>|, du = u - u_twin with its fluid-volume mean removed per component (the
//     kick grows the checkerboard, not the box mean; the mean carries a separate, report-only open
//     defect — probeMeanMomentum). Pass iff at step 100 each is <= 2x its step-0 value and
//     <= (1 + 1e-6)x its step-50 value. A kick fails at dt >= 0.1 (x12 per step at dt = 1).
//     Run at amr's DEFAULT settings (quadratic C/F scheme, AUTO ghost projection).
//  G3 dt-independence. Sphere meshes, dt in {1, 10, 100}, each marched to
//     max|du per step| <= 1e-12 max|u|. Pass iff max|u(dt_i) - u(dt_j)| <= 1e-9 max|u|. With the
//     rotational term switched off a kick is stable but Chorin; G3 is what catches that.
//
// Two translations to amr's API, forced by it (amr is triply periodic, a wall is an immersed
// solid, and the only drive is a UNIFORM body force): flow's G3 "walls at +-y + TG forcing" is an
// immersed sphere driven by a uniform force (the configuration whose drag V8's Chorin error moved
// by -40 %), and there is no all-fluid G3 — a uniform force in a periodic all-fluid box has no
// steady state. The dt = 10 and 100 runs of G3 start from the dt = 1 fixed point (the documented
// dt switch: setDt -> setSolid -> setVelocity/setPressure); the stopping test is the same, so a
// dt-dependent fixed point still moves the state and fails the comparison.
//
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/amr/block_octree.hpp"
#include "peclet/amr/flow.hpp"
#include "peclet/amr/leaf_field.hpp"
#include "peclet/amr/refine.hpp"
#include "peclet/core/common/types.hpp"
#include "test_util.hpp"

using namespace peclet::core;
using namespace peclet::amr;

namespace {

using BO = BlockOctree<3, 21>;
using Flow = AmrFlow<21>;

constexpr double kEps = 1e-3;        // seed amplitude
constexpr double kSolveTol = 1e-12;  // momentum + pressure relative tolerance (both twins)
constexpr int kG2Steps = 100;
constexpr int kG3MaxSteps = 20000;

struct Config {
  std::string name;
  bool refined;
  bool solid;
};

// The sphere: centre of the box, radius 2.3 (uniform 8^3) / 3.3 (the 16^3-finest refined mesh).
struct Sphere {
  double c, r;
  double operator()(const Vec<3>& p) const {
    const double dx = p[0] - c, dy = p[1] - c, dz = p[2] - c;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - r;  // > 0 fluid
  }
};

Sphere sphereOf(const Config& cfg) {
  return cfg.refined ? Sphere{8.0, 3.3} : Sphere{4.0, 2.3};
}

BO buildMesh(const Config& cfg) {
  if (!cfg.refined)
    return BO(IVec<3>{8, 8, 8}, 0);  // lmax = 0: the uniform 8^3 brick, h = 1
  BO t(IVec<3>{8, 8, 8}, 1);         // root 8^3 at h = 2, refinable once to h = 1
  const auto geo = AmrGeometry<3>::isotropicAt(Vec<3>{}, 1.0);
  if (cfg.solid) {
    refineToSdf(t, geo, sphereOf(cfg), 0, 2.0);  // the cut band at the finest level
  } else {
    // One refined region: the block [4, 12)^3 at the finest level, 2:1 seams on its six faces.
    refineToSdf(
        t, geo,
        [](const Vec<3>& p) {
          const bool in =
              p[0] >= 4 && p[0] < 12 && p[1] >= 4 && p[1] < 12 && p[2] >= 4 && p[2] < 12;
          return in ? 0.0 : 1e3;
        },
        0, 0.0);
  }
  return t;
}

void initFlow(Flow& f, const BO& t, const Config& cfg, double dt) {
  f.init(t, 1.0);
  f.setDensity(1.0);
  f.setViscosity(1.0);
  f.setDt(dt);
  f.setPressureTol(kSolveTol);
  f.setMomentumTol(kSolveTol);
  // A drive only where a steady state exists (with the solid); the all-fluid periodic box would
  // just translate uniformly.
  if (cfg.solid)
    f.setBodyForce(1e-2, 0.0, 0.0);
  if (cfg.solid)
    f.setSolid(sphereOf(cfg));
  else
    f.setSolid([](const Vec<3>&) { return 1e3; });
}

// Deterministic, library-independent uniform [-1, 1) (splitmix64).
double uniformPm1(std::uint64_t k) {
  std::uint64_t z = k + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return 2.0 * static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0) - 1.0;
}

// max(acc, |v|) that cannot swallow a NaN: std::max(acc, NaN) returns acc, so a blown-up field
// would otherwise read as small. Any non-finite value makes the result +inf (a failure everywhere).
double maxAbs(double acc, double v) {
  return std::isfinite(v) ? std::max(acc, std::fabs(v)) : HUGE_VAL;
}

struct Leafs {
  std::vector<double> chk, vol;
  std::vector<char> fluid;
};

Leafs leafData(const BO& t, const Flow& f) {
  const Index n = t.numLeaves();
  Leafs L;
  L.chk.resize(static_cast<std::size_t>(n));
  L.vol.resize(static_cast<std::size_t>(n));
  L.fluid.resize(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    const auto b = t.bounds(i);
    const long s = 1L << t.level(i);
    long par = 0;
    for (int d = 0; d < 3; ++d)
      par += b[0][d] / s;  // the leaf's index at its own level
    L.chk[static_cast<std::size_t>(i)] = (par % 2 == 0) ? 1.0 : -1.0;
    L.vol[static_cast<std::size_t>(i)] = static_cast<double>(s) * s * s;
    L.fluid[static_cast<std::size_t>(i)] = f.isFluid(i) ? 1 : 0;
  }
  return L;
}

struct G2Measure {
  double ap, u;
};

// Seed P and u with eps*checkerboard + eps*random (solid cells stay 0); four disjoint streams.
void seed(Flow& f, const Leafs& L, Index n) {
  std::vector<double> p(static_cast<std::size_t>(n), 0.0);
  std::array<std::vector<double>, 3> u;
  for (auto& v : u)
    v.assign(static_cast<std::size_t>(n), 0.0);
  for (Index i = 0; i < n; ++i) {
    const auto k = static_cast<std::size_t>(i);
    if (!L.fluid[k])
      continue;
    const auto key = static_cast<std::uint64_t>(i) * 4u;
    p[k] = kEps * L.chk[k] + kEps * uniformPm1(key);
    for (int c = 0; c < 3; ++c)
      u[static_cast<std::size_t>(c)][k] = kEps * L.chk[k] + kEps * uniformPm1(key + 1u + c);
  }
  f.setPressure(p);
  for (int c = 0; c < 3; ++c)
    f.setVelocity(c, u[static_cast<std::size_t>(c)]);
}

// A_P: the (pi,pi,pi) amplitude of dP = P - P_twin (fluid-mean removed). U: max|du - <du>|, du =
// u - u_twin with its fluid-VOLUME MEAN removed per component. The kick grows the checkerboard,
// not the box mean; the mean is excluded because the quadratic C/F pressure gradient leaks net
// momentum at 2:1 seams (a separate open defect, probed below and NOT gated here — see
// probeMeanMomentum and amr CLAUDE.md "Gotchas").
G2Measure measure(const Leafs& L, const Flow& fs, const Flow& ft) {
  const auto ps = fs.pressure(), pt = ft.pressure();
  const auto us = fs.velocities(), ut = ft.velocities();
  const std::size_t n = ps.size();
  double sv = 0, sp = 0;
  std::array<double, 3> su{0, 0, 0};
  for (std::size_t i = 0; i < n; ++i)
    if (L.fluid[i]) {
      sv += L.vol[i];
      sp += L.vol[i] * (ps[i] - pt[i]);
      for (int c = 0; c < 3; ++c)
        su[static_cast<std::size_t>(c)] += L.vol[i] * (us[i * 3 + c] - ut[i * 3 + c]);
    }
  const double mean = sp / sv;
  double a = 0, um = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (L.fluid[i]) {
      a += L.vol[i] * L.chk[i] * (ps[i] - pt[i] - mean);
      for (int c = 0; c < 3; ++c)
        um = maxAbs(um, us[i * 3 + c] - ut[i * 3 + c] - su[static_cast<std::size_t>(c)] / sv);
    }
  return {std::fabs(a) / sv, um};
}

// G2 on one mesh and one dt. Returns true on pass; prints the record.
bool runG2(const Config& cfg, double dt) {
  BO t = buildMesh(cfg);
  Flow fs, ft;
  initFlow(fs, t, cfg, dt);
  initFlow(ft, t, cfg, dt);
  const Leafs L = leafData(t, fs);
  const Index n = t.numLeaves();
  seed(fs, L, n);
  const G2Measure m0 = measure(L, fs, ft);
  G2Measure m50{}, m100{};
  for (int s = 1; s <= kG2Steps; ++s) {
    fs.step(200, 200);
    ft.step(200, 200);
    if (s == kG2Steps / 2)
      m50 = measure(L, fs, ft);
  }
  m100 = measure(L, fs, ft);
  const bool okP =
      std::isfinite(m100.ap) && m100.ap <= 2.0 * m0.ap && m100.ap <= (1 + 1e-6) * m50.ap;
  const bool okU = std::isfinite(m100.u) && m100.u <= 2.0 * m0.u && m100.u <= (1 + 1e-6) * m50.u;
  // Mean per-step multiplier over steps 50..100 (the growth factor a kick would show).
  const double gP = std::pow(m100.ap / m50.ap, 1.0 / (kG2Steps / 2));
  const double gU = std::pow(m100.u / m50.u, 1.0 / (kG2Steps / 2));
  std::printf(
      "[G2] %-14s dt=%-5g  A_P: %.4e -> %.4e -> %.4e (x%.6f/step)  U: %.4e -> %.4e -> %.4e "
      "(x%.6f/step)  %s\n",
      cfg.name.c_str(), dt, m0.ap, m50.ap, m100.ap, gP, m0.u, m50.u, m100.u, gU,
      (okP && okU) ? "ok" : "FAIL");
  return okP && okU;
}

// March `f` to max|du per step| <= 1e-12 max|u|; returns the steps taken (-1: not converged).
int marchToSteady(Flow& f) {
  auto prev = f.velocities();
  for (int s = 1; s <= kG3MaxSteps; ++s) {
    f.step(500, 500);
    const auto u = f.velocities();
    double d = 0, m = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      d = maxAbs(d, u[i] - prev[i]);
      m = maxAbs(m, u[i]);
    }
    if (!std::isfinite(d) || !std::isfinite(m))
      return -1;
    if (d <= 1e-12 * m)
      return s;
    prev = u;
  }
  return -1;
}

bool runG3(const Config& cfg) {
  BO t = buildMesh(cfg);
  const std::array<double, 3> dts{1.0, 10.0, 100.0};
  std::array<std::vector<double>, 3> us;
  Flow f;
  initFlow(f, t, cfg, dts[0]);
  bool ok = true;
  for (std::size_t k = 0; k < dts.size(); ++k) {
    if (k > 0) {  // the dt switch: state out, rebuild the operator at the new dt, state back in
      const auto p = f.pressure();
      std::array<std::vector<double>, 3> u{f.velocity(0), f.velocity(1), f.velocity(2)};
      f.setDt(dts[k]);
      f.setSolid(sphereOf(cfg));
      f.setPressure(p);
      for (int c = 0; c < 3; ++c)
        f.setVelocity(c, u[static_cast<std::size_t>(c)]);
    }
    const int steps = marchToSteady(f);
    us[k] = f.velocities();
    std::printf("[G3] %-14s dt=%-5g steady after %d steps\n", cfg.name.c_str(), dts[k], steps);
    ok = ok && steps > 0;
  }
  double m = 0;
  for (const auto& u : us)
    for (double v : u)
      m = maxAbs(m, v);
  for (std::size_t a = 0; a < dts.size(); ++a)
    for (std::size_t b = a + 1; b < dts.size(); ++b) {
      double d = 0;
      for (std::size_t i = 0; i < us[a].size(); ++i)
        d = maxAbs(d, us[a][i] - us[b][i]);
      const bool pass = std::isfinite(d) && std::isfinite(m) && d <= 1e-9 * m;
      std::printf("[G3] %-14s max|u(dt=%g) - u(dt=%g)| / max|u| = %.3e  %s\n", cfg.name.c_str(),
                  dts[a], dts[b], d / m, pass ? "ok" : "FAIL");
      ok = ok && pass;
    }
  return ok;
}

// REPORT-ONLY PROBE, NOT GATED — an OPEN DEFECT, recorded so it is not hidden by the mean
// removal in G2's U. With the default quadratic C/F scheme (setCfScheme(1)) the pressure gradient
// at 2:1 seams is not momentum-conservative: on the refined all-fluid periodic mesh a decaying
// pressure perturbation puts net momentum into the box (the neutral uniform mode), growing with dt;
// with setCfScheme(0) the mean is conserved. This is flow doc/collocated_varrho_forces.md §13.2's
// open question (is amr's pair exact-transpose at C/F faces?) and needs an architect/amr-owner
// decision before amr multiphase (amr CLAUDE.md "Gotchas"). It is NOT the kick signature.
void probeMeanMomentum() {
  const Config cfg{"refined/fluid", true, false};
  for (int cf : {1, 0})
    for (double dt : {1.0, 10.0, 100.0}) {
      BO t = buildMesh(cfg);
      Flow f;
      f.init(t, 1.0);
      f.setDensity(1.0);
      f.setViscosity(1.0);
      f.setDt(dt);
      f.setPressureTol(kSolveTol);
      f.setMomentumTol(kSolveTol);
      f.setCfScheme(cf);
      f.setSolid([](const Vec<3>&) { return 1e3; });
      const Leafs L = leafData(t, f);
      seed(f, L, t.numLeaves());
      auto meanU = [&] {
        const auto u = f.velocities();
        std::array<double, 3> m{0, 0, 0};
        double sv = 0;
        for (std::size_t i = 0; i < L.vol.size(); ++i) {
          sv += L.vol[i];
          for (int c = 0; c < 3; ++c)
            m[static_cast<std::size_t>(c)] += L.vol[i] * u[i * 3 + c];
        }
        for (auto& v : m)
          v /= sv;
        return m;
      };
      const auto m0 = meanU();
      for (int s = 0; s < kG2Steps; ++s)
        f.step(200, 200);
      const auto m1 = meanU();
      std::printf(
          "[probe, not gated] mean-u drift, refined/fluid cf=%d dt=%-5g: (%+.3e, %+.3e, "
          "%+.3e) after %d steps (seed amplitude %.0e)\n",
          cf, dt, m1[0] - m0[0], m1[1] - m0[1], m1[2] - m0[2], kG2Steps, kEps);
    }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const std::array<Config, 4> cfgs{
        Config{"uniform/fluid", false, false}, Config{"uniform/sphere", false, true},
        Config{"refined/fluid", true, false}, Config{"refined/sphere", true, true}};
    for (const auto& cfg : cfgs)
      for (double dt : {0.1, 1.0, 10.0, 100.0})
        PECLET_AMR_CHECK(runG2(cfg, dt));
    for (const auto& cfg : cfgs)
      if (cfg.solid)
        PECLET_AMR_CHECK(runG3(cfg));
    probeMeanMomentum();
  }
  Kokkos::finalize();
  PECLET_AMR_RETURN_TEST_RESULT();
}
