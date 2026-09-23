// core — the prebuilt tables the convective flux needs at a 2:1 coarse/fine seam (ROADMAP B5).
//
// docs/amr_cf_convective.md: the SOU/Koren reconstruction of the ADVECTED value is O(h) wrong at
// every face whose upwind-side stencil crosses a level — the coarse upwind cell's column is
// TANGENTIALLY offset from the sub-face centroid (case 1), and the second upwind probe of the two
// same-level faces beside the seam sits at the wrong DISTANCE and at a tangential offset as well
// (cases 2 and 3). The fix is a level-aware, upwind-side reconstruction; everything it needs that
// depends only on the mesh is prebuilt here, once, in setSolid.
//
// THREE KINDS OF ENTRY, one CSR of records (§5.2):
//
//   * a SAMPLE record, one per 2:1 sub-face pair (C, F): `φ_C` sampled at F's tangential column,
//     i.e. the entry (C, 1) followed VERBATIM by `cf_scheme.hpp::cfAppendStencil`'s
//     (coarse* − coarse) delta — so the record's weights sum to 1 and the gathered value is the
//     Martin–Cartwright tangential quadratic the C/F operator overlays already use. It is read
//     twice: as the sub-face's own `φ_U*` (case 1) and as F's upstream probe on the face behind it
//     (case 2), where its `recDist` = 1.5·width(F) is the probe's TRUE distance.
//   * a LAYER record, one per (coarse cell C, face) whose far side is refined: the mean of the
//     four fine cells in that face layer, `recDist` = 0.75·width(C) — the upstream probe of C's
//     opposite face (case 3), again at its true distance.
//   * a DESCRIPTOR, one per face slot that references at least one record: which record plays
//     `φ_U*` / `φ_D*` and which plays the upstream probe, for each of the two flow directions,
//     plus the half widths `d1I/d1J`. A slot with no live field gets `seam = −1` and the kernels
//     take TODAY'S line, bit for bit.
//
// GATES (§5.4). No sample record unless both incident cells are REGULAR (fluid and not cut — the
// per-face C/F flux gate of docs/amr_cf_flux_gate.md, ghost-visible under MPI); no layer record
// unless the coarse cell is regular and all four fine cells are fluid; never a partial record. A
// withheld record leaves the slot on today's arithmetic, so cut seams are bit-identical.
//
// DETERMINISM. The walk is SEQUENTIAL over the local rows in `forEachFaceFull` order and record
// ids are assigned on first reference, so the tables are a pure function of the mesh — which is
// what makes the np = 1 build bitwise identical to the single-rank one (records are gathered in
// record order by the kernel, never reordered).
#ifndef PECLET_AMR_SEAM_RECON_HPP
#define PECLET_AMR_SEAM_RECON_HPP

#include <array>
#include <map>
#include <utility>
#include <vector>

#include "peclet/amr/cf_scheme.hpp"
#include "peclet/amr/common.hpp"
#include "peclet/amr/poisson.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::amr {

/// Host-side seam-reconstruction tables (docs/amr_cf_convective.md §5.2). Uploaded into FaceGeom.
struct SeamReconHost {
  std::vector<Index> seam;   ///< per face slot: descriptor id, −1 = plain slot
  std::vector<Index> sampI;  ///< per descriptor: sample record when i is the COARSE cell, else −1
  std::vector<Index> sampJ;  ///< per descriptor: sample record when j is the COARSE cell, else −1
  std::vector<Index> uuRecI;    ///< per descriptor: upstream-probe record for up == i, else −1
  std::vector<Index> uuRecJ;    ///< per descriptor: upstream-probe record for up == j, else −1
  std::vector<double> d1I;      ///< per descriptor: half width of i along the face axis (physical)
  std::vector<double> d1J;      ///< per descriptor: half width of j along the face axis (physical)
  std::vector<Index> recStart;  ///< record CSR offsets, size nRec+1
  std::vector<double> recDist;  ///< per record: dUU for UPSTREAM use (unused as a φ_U* sample)
  std::vector<Index> recCell;   ///< record entries: cell slot
  std::vector<double> recW;     ///< record entries: weight
  Index numSampleRecords = 0;   ///< diagnostics: how many of the records are sample records
  Index numLayerRecords = 0;    ///< diagnostics: how many are layer records
  Index numDescriptors() const { return static_cast<Index>(sampI.size()); }
};

namespace detail {

/// The four sub-face neighbour leaves across `C`'s face (`axis`, `dir`), in `forEachFaceFull`'s
/// own sub-face order. False unless all four resolve and sit EXACTLY one level finer than `C`
/// (the 2:1 balance guarantees that when the far side is refined at all).
template <unsigned Bits>
inline bool seamFaceLayer(const AmrPoisson<3, Bits>& ap, Index C, int axis, int dir,
                          std::array<Index, 4>& cells) {
  const unsigned Lc = ap.levelOf(C);
  if (Lc == 0)
    return false;  // C is at the finest level: nothing below it
  const std::array<long, 3> lo = ap.loOf(C);
  const long sc = 1L << Lc, sh = sc >> 1;
  std::array<long, 3> p = lo;
  p[axis] = (dir > 0) ? lo[axis] + sc : lo[axis] - 1;
  for (int k = 0; k < 4; ++k) {
    std::array<long, 3> q = p;
    int bit = 0;
    for (int t = 0; t < 3; ++t) {
      if (t == axis)
        continue;
      q[t] = lo[t] + (((k >> bit) & 1) ? sh : 0L);
      ++bit;
    }
    const auto [c, Lq] = ap.probeSlot(q);
    if (c < 0 || Lq + 1 != Lc)
      return false;
    cells[static_cast<std::size_t>(k)] = c;
  }
  return true;
}

}  // namespace detail

/// Build the seam-reconstruction tables over the LOCAL rows of `ap` (docs/amr_cf_convective.md
/// §5.2). `regularOk` is the ghost-visible `fluid && !cut` flag the C/F overlays are gated on and
/// `fluidOk` the plain fluid predicate `cfAppendStencil` drops a tangential side on; `scheme`
/// selects the tangential sample (CfScheme::quadratic — §5.1 Q-A: ONE tangential-sample operator
/// suite-wide). Every probe it issues is registered by the distributed discovery fixpoint
/// (`AmrFlow::probeCfScheme` + `AmrFlow::probeSeamLayer`), so a ghost cell resolves rather than
/// throwing.
template <unsigned Bits, class RegularFn, class FluidFn>
SeamReconHost buildSeamRecon(const AmrPoisson<3, Bits>& ap, RegularFn&& regularOk,
                             FluidFn&& fluidOk, CfScheme scheme) {
  const Index n = ap.octree().numLeaves();
  SeamReconHost out;
  std::vector<std::vector<detail::ScalarEnt>> rec;  // entries, one vector per record
  std::vector<double> dist;                         // recDist, one per record

  // (coarse, fine, axis) -> sample record id; (coarse, axis, dir) -> layer record id. A withheld
  // record is cached as −1 so a gate is evaluated once per key, not once per referencing slot.
  std::map<std::array<Index, 3>, Index> sampleOf, layerOf;

  auto sampleRecord = [&](Index C, Index F, int axis) -> Index {
    if (C < 0 || F < 0)
      return -1;
    const std::array<Index, 3> key{C, F, static_cast<Index>(axis)};
    const auto it = sampleOf.find(key);
    if (it != sampleOf.end())
      return it->second;
    Index id = -1;
    if (regularOk(C) && regularOk(F)) {
      std::vector<detail::ScalarEnt> e;
      e.push_back(detail::ScalarEnt{C, 1.0});  // the raw coarse value; the stencil is its DELTA
      detail::cfAppendStencil(ap, e, C, F, axis, 1.0, fluidOk, scheme);
      id = static_cast<Index>(rec.size());
      rec.push_back(std::move(e));
      dist.push_back(1.5 * static_cast<double>(ap.cellWidth(F, axis)));
      ++out.numSampleRecords;
    }
    sampleOf.emplace(key, id);
    return id;
  };

  auto layerRecord = [&](Index C, int axis, int dir) -> Index {
    if (C < 0)
      return -1;
    const std::array<Index, 3> key{C, static_cast<Index>(axis), static_cast<Index>(dir)};
    const auto it = layerOf.find(key);
    if (it != layerOf.end())
      return it->second;
    Index id = -1;
    std::array<Index, 4> cells{};
    if (regularOk(C) && detail::seamFaceLayer(ap, C, axis, dir, cells)) {
      bool ok = true;
      for (const Index c : cells)
        ok = ok && fluidOk(c);
      if (ok) {
        std::vector<detail::ScalarEnt> e;
        for (const Index c : cells)
          e.push_back(detail::ScalarEnt{c, 0.25});
        id = static_cast<Index>(rec.size());
        rec.push_back(std::move(e));
        dist.push_back(0.75 * static_cast<double>(ap.cellWidth(C, axis)));
        ++out.numLayerRecords;
      }
    }
    layerOf.emplace(key, id);
    return id;
  };

  for (Index i = 0; i < n; ++i) {
    ap.forEachFaceFull(i, [&](Index j, int a, int d, double, double, double) {
      const unsigned Li = ap.levelOf(i), Lj = ap.levelOf(j);
      Index rsI = -1, rsJ = -1, ruI = -1, ruJ = -1;
      // Case 1 — this slot IS a 2:1 sub-face: the coarse side's value is sampled at the fine
      // side's tangential column. (A level in this octree counts DOWN: 0 is the finest.)
      if (Lj != Li) {
        if (Li > Lj)
          rsI = sampleRecord(i, j, a);
        else
          rsJ = sampleRecord(j, i, a);
      }
      // Cases 2/3 — the upstream probe of each side crosses a level: coarser ⇒ the same
      // tangential sample at its true distance; finer ⇒ the mean of that face's fine layer.
      const Index uI = ap.periodicNeighbor(i, a, -d);
      if (uI >= 0) {
        const unsigned Lu = ap.levelOf(uI);
        if (Lu > Li)
          ruI = sampleRecord(uI, i, a);
        else if (Lu < Li)
          ruI = layerRecord(i, a, -d);
      }
      const Index uJ = ap.periodicNeighbor(j, a, d);
      if (uJ >= 0) {
        const unsigned Lu = ap.levelOf(uJ);
        if (Lu > Lj)
          ruJ = sampleRecord(uJ, j, a);
        else if (Lu < Lj)
          ruJ = layerRecord(j, a, d);
      }
      if (rsI < 0 && rsJ < 0 && ruI < 0 && ruJ < 0) {
        out.seam.push_back(-1);  // plain slot: today's reconstruction, bit for bit
        return;
      }
      out.seam.push_back(out.numDescriptors());
      out.sampI.push_back(rsI);
      out.sampJ.push_back(rsJ);
      out.uuRecI.push_back(ruI);
      out.uuRecJ.push_back(ruJ);
      out.d1I.push_back(0.5 * static_cast<double>(ap.cellWidth(i, a)));
      out.d1J.push_back(0.5 * static_cast<double>(ap.cellWidth(j, a)));
    });
  }

  const Index nRec = static_cast<Index>(rec.size());
  out.recStart.assign(static_cast<std::size_t>(nRec) + 1, 0);
  for (Index r = 0; r < nRec; ++r)
    out.recStart[static_cast<std::size_t>(r) + 1] =
        out.recStart[static_cast<std::size_t>(r)] +
        static_cast<Index>(rec[static_cast<std::size_t>(r)].size());
  out.recDist = std::move(dist);
  out.recCell.reserve(static_cast<std::size_t>(out.recStart[static_cast<std::size_t>(nRec)]));
  out.recW.reserve(static_cast<std::size_t>(out.recStart[static_cast<std::size_t>(nRec)]));
  for (Index r = 0; r < nRec; ++r)
    for (const auto& e : rec[static_cast<std::size_t>(r)]) {
      out.recCell.push_back(e.cell);
      out.recW.push_back(e.w);
    }
  return out;
}

}  // namespace peclet::amr

#endif  // PECLET_AMR_SEAM_RECON_HPP
