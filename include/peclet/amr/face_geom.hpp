// core — FaceGeom: the static (sub)face geometry CSR of the collocated AMR projection.
//
// Factored out of flow.hpp into its own header so the device face-geometry assembler
// (facegeom_assembly.hpp) and the flow driver (flow.hpp) can both name the type without a
// circular include (flow.hpp consumes the assembler, the assembler produces a FaceGeom).
//
// Per (sub)face (matching AmrPoisson::forEachFaceFull, 2:1 sub-faces): neighbour, axis, dir, α·area
// (divergence weight), raw area (advection flux), face-normal distance, openness α (gradient gate),
// and the two SOU upstream-probe leaves. Per cell: 1/V and a fluid flag. Plus, since ROADMAP
// B5, the seam-reconstruction descriptors and records of docs/amr_cf_convective.md §5.2 (built
// by seam_recon.hpp, read by the two advective kernels only; empty where there is no 2:1 seam).
//
// Requires a Kokkos build (View).
#ifndef PECLET_AMR_FACE_GEOM_HPP
#define PECLET_AMR_FACE_GEOM_HPP

#include "peclet/amr/common.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"

namespace peclet::amr {

struct FaceGeom {
  View<Index> start;       ///< CSR row offsets, size n+1
  View<Index> nbr;         ///< neighbour leaf per face, size nFaces
  View<int> axis;          ///< face axis 0/1/2, size nFaces
  View<int> dir;           ///< face direction +1/-1, size nFaces
  View<double> alphaArea;  ///< α·area (physical) per face, size nFaces
  View<double> rawArea;    ///< raw face area (physical, no openness) per face — advection flux
  View<double> dist;       ///< face-normal distance (physical) per face, size nFaces
  View<double> alpha;      ///< openness per face (gradient gate), size nFaces
  View<Index> upupI;  ///< upstream-of-i probe (periodicNeighbor(i,axis,−dir)) — SOU, size nFaces
  View<Index> upupJ;  ///< upstream-of-j probe (periodicNeighbor(j,axis,+dir)) — SOU, size nFaces
  View<double> invVol;  ///< 1/V_i per cell, size n
  View<char> fluid;     ///< per-cell fluid flag, size n
  Index n = 0;
  // ---- seam reconstruction of the ADVECTED value (docs/amr_cf_convective.md §5.2) -------------
  // Empty unless the seam builder ran (seam_recon.hpp; setSolid). `seam` is the per-slot
  // descriptor id, −1 = plain slot; the descriptor arrays are indexed by it and the records by a
  // descriptor's four record ids. Read ONLY by the two advective kernels.
  View<Index> seam;      ///< descriptor id per face slot, −1 = plain slot; size nFaces
  View<Index> sampI;     ///< per descriptor: sample record when i is the COARSE cell, else −1
  View<Index> sampJ;     ///< per descriptor: sample record when j is the COARSE cell, else −1
  View<Index> uuRecI;    ///< per descriptor: upstream-probe record for up == i, else −1
  View<Index> uuRecJ;    ///< per descriptor: upstream-probe record for up == j, else −1
  View<double> d1I;      ///< per descriptor: half width of i along the face axis (physical)
  View<double> d1J;      ///< per descriptor: half width of j along the face axis (physical)
  View<Index> recStart;  ///< record CSR offsets, size nRec+1
  View<double> recDist;  ///< per record: dUU for UPSTREAM use (unused as a φ_U* sample)
  View<Index> recCell;   ///< record entries: cell slot
  View<double> recW;     ///< record entries: weight
};

/// The seam tables as the advective kernels capture them: const Views plus the ablation flag
/// (`AmrFlow::setSeamReconstruction`). `on == false` — the switch off, or no tables built — makes
/// every slot take today's reconstruction, which is what gate G0 of docs/amr_cf_convective.md §9
/// asserts is bit-identical.
struct SeamView {
  View<const Index> seam, sampI, sampJ, uuRecI, uuRecJ, recStart, recCell;
  View<const double> d1I, d1J, recDist, recW;
  bool on = false;

  /// Σ over the record's entries IN RECORD ORDER, never reordered — that is what keeps np = 1
  /// bitwise against single-rank (docs/amr_cf_convective.md §5.3).
  template <class FldFn>
  KOKKOS_INLINE_FUNCTION double gather(Index r, const FldFn& fld) const {
    double v = 0.0;
    for (Index e = recStart(r); e < recStart(r + 1); ++e)
      v += recW(e) * fld(recCell(e));
    return v;
  }
};

/// Bundle a FaceGeom's seam tables for the kernels. `on` is the ablation switch; tables that were
/// never built (no 2:1 seam, or the standard C/F scheme) turn it off by themselves.
inline SeamView makeSeamView(const FaceGeom& g, bool on) {
  SeamView sv;
  sv.on = on && g.seam.extent(0) > 0;
  sv.seam = g.seam;
  sv.sampI = g.sampI;
  sv.sampJ = g.sampJ;
  sv.uuRecI = g.uuRecI;
  sv.uuRecJ = g.uuRecJ;
  sv.recStart = g.recStart;
  sv.recCell = g.recCell;
  sv.d1I = g.d1I;
  sv.d1J = g.d1J;
  sv.recDist = g.recDist;
  sv.recW = g.recW;
  return sv;
}

}  // namespace peclet::amr

#endif  // PECLET_AMR_FACE_GEOM_HPP
