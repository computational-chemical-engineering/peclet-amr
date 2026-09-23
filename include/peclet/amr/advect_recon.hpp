// core — shared, backend-agnostic high-order advection face reconstruction.
//
// The SOU / Koren-TVD reconstruction of the advected face value from the two upwind cells (upup,
// up) and the downwind cell (down) is the numerically delicate part of the advection scheme —
// exactly the kind of formula that must not drift between the serial host solver (AmrFlow::hoFace,
// flow.hpp) and the Kokkos device solver (deferredSou / advectExplicit, flow.hpp). This is that
// formula, written ONCE as a MORTON_HD function (host- and device-callable; empty MORTON_HD in the
// pure-C++ build). min/max/abs are expressed with branches rather than std::/Kokkos:: math so the
// body is identical on both backends (and bit-identical to the previous std::fmin/fmax form for the
// finite velocity/scalar data the solver produces).
#ifndef PECLET_AMR_ADVECT_RECON_HPP
#define PECLET_AMR_ADVECT_RECON_HPP

#include <morton/morton.hpp>

#include "peclet/amr/common.hpp"
#ifndef MORTON_HD
#define MORTON_HD
#endif

namespace peclet::amr {

/// psi = max(0, min(2r, (1+2r)/3, 2)) — the Koren limiter, written once for the two
/// reconstructions below (same operations, same order: factoring it out is bit-inert).
MORTON_HD inline double korenPsi(double r) {
  const double t = (1.0 + 2.0 * r) / 3.0;
  double m = (2.0 * r < t) ? (2.0 * r) : t;
  if (m > 2.0)
    m = 2.0;
  return (m > 0.0) ? m : 0.0;
}

/// High-order advected face value from the two upwind cells (`upup`, `up`) and the downwind cell
/// (`down`). `scheme` 0 = second-order upwind (SOU = 1.5·up − 0.5·upup); else Koren TVD limiter.
///
/// It assumes the three cells are EQUALLY SPACED and their centres tangentially aligned with the
/// face, which is true of every face but the ones a 2:1 octree seam touches — those take
/// `hoFaceValueSeam` instead, and only those: the two are not bit-identical in the regular-face
/// limit (docs/amr_cf_convective.md §5.1), so routing a plain slot through the seam form would
/// move digits for nothing.
MORTON_HD inline double hoFaceValue(double upup, double up, double down, int scheme) {
  if (scheme == 0)
    return 1.5 * up - 0.5 * upup;  // SOU
  const double den = down - up;
  const double aden = (den < 0.0) ? -den : den;
  const double r = (aden < 1e-10) ? 0.0 : (up - upup) / den;
  return up + 0.5 * korenPsi(r) * den;  // Koren TVD
}

/// The same reconstruction with every distance and every sample made explicit — the rule of
/// docs/amr_cf_convective.md §5.1, for a face whose upwind-side stencil crosses an octree level:
///
///     gUp = (up − upup) / dUU              the upwind slope on U's OWN column
///     SOU   : phi_face = upStar + d1·gUp
///     Koren : den = downStar − upStar;  r = gUp·dD/den;  phi_face = upStar + (d1/dD)·psi(r)·den
///
/// `up` is the raw upwind value and `upStar` the same cell SAMPLED at the face's tangential column
/// (they differ only when the upwind cell is the coarse side of a 2:1 sub-face); the slope stays on
/// the raw column, where the two stencil points carry the SAME tangential offset, so the sample
/// correction enters exactly once and with weight one. `upup` is the upstream probe at its true
/// distance `dUU`, `d1` the upwind cell's half width along the face axis and `dD` the face's
/// centre-to-centre distance. With `d1 = dD/2` and `dUU = dD` this reduces to `hoFaceValue`
/// analytically — but NOT bitwise, which is why plain slots never come here.
MORTON_HD inline double hoFaceValueSeam(double upup, double up, double upStar, double downStar,
                                        double d1, double dUU, double dD, int scheme) {
  const double gUp = (up - upup) / dUU;
  if (scheme == 0)
    return upStar + d1 * gUp;  // SOU
  const double den = downStar - upStar;
  const double aden = (den < 0.0) ? -den : den;
  const double r = (aden < 1e-10) ? 0.0 : gUp * dD / den;
  return upStar + (d1 / dD) * korenPsi(r) * den;  // Koren TVD
}

}  // namespace peclet::amr

#endif  // PECLET_AMR_ADVECT_RECON_HPP
