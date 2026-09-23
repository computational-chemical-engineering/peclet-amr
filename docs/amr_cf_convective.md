# The convective flux at a 2:1 seam — design note (ROADMAP B5)

> Fable design pass, 2026-09-23, branch `b5-cf-convective` of `peclet-amr` (base `3c6c343`).
> Brief: `docs/briefs/cf_convective_flux.md`. Prototype: `tests/study/amr_cf_seam_proto.py`
> (numpy replica of the operator; every number below is reproducible from it in ~2 min).
> Implemented by someone who has seen neither the brief nor the conversation — it stands alone.

## 0. The decision

**Build it.** At every face whose upwind-side stencil crosses a level, reconstruct the advected
value from the upwind side with **level-aware probes**: the coarse upwind cell's value is
**tangentially sampled** at the sub-face's column (the existing `cfAppendStencil` sample, applied
*once*, to the extrapolated value), and an upstream probe at another level is taken **at its true
distance** (a coarser one as the same tangential sample, a finer one as the mean of the four
face-layer children). Three faces per seam change; the bulk, cut seams and the implicit FOU matrix
are untouched, bit for bit. Measured on the operator at N = 64: the interface truncation falls
**7.50e-3 → 2.16e-3 (3.5×)**, the layer behind the seam **4.61e-3 → 4.02e-4 (11×)**, and the
seam's truncation converges at the intrinsic first order (×3.7 per doubling) instead of drifting
to zeroth (×2). The 22–36 % of the brief's prototypes was an artefact of a weight error, not the
ceiling. Predicted headline: the graded mesh's *shape* error against the unrefined mesh at N = 64
goes from **3.27× to ≈ 1.3×**; the *amplitude* half (numerical viscosity of the seam fluxes, out
of scope) stays, so `m3` improves only from 1.56× to ≈ 1.37× — the honest limit of this item.

## 1. Problem and scope

**In scope.** The reconstruction of the *advected* value `φ_face` in `flow.hpp::deferredSou` /
`advectExplicit` (and its implicit twin's deferred correction) at the faces whose SOU/Koren stencil
touches two octree levels: the 2:1 sub-face itself, and the two same-level faces next to it whose
second upwind probe (`upupI/J`) lies across the seam. The prebuilt data those kernels need, its
host builder, its distributed probe reach, its cut-cell gate, the gates and the expected numbers.

**Out of scope.** The advecting face velocity `uf`; the pressure path (options A/B of
`amr_pressure_iteration.md` §14, exonerated); the implicit FOU matrix (`buildFou`); the *amplitude*
half of the (G)−(C) gap (the seam fluxes' numerical viscosity — §14.4c shows it responds to the
limiter and the C/F scheme, this note does not touch it); cut-cell items A2/A3; `peclet.flow`.
`scalar_transport.hpp` is first-order upwind (`scalar_transport.hpp:85`) and does **not** share
this reconstruction — the brief's §10 is wrong on that point; nothing there inherits anything.

## 2. What is actually wrong — the mechanism, measured

Fine width `h`, coarse `H = 2h`. Linear extrapolation from two column points at distances `a`, `b`
from a face has error `(a·b/2)·φ_nn`. Regular fine face: `(3/8)h²`. Regular coarse: `(3/2)h²`.

At a 2:1 sub-face whose upwind cell is the coarse `C`, the shipped stencil is `1.5·φ_C − 0.5·φ_CC`
— the coarse column's own extrapolation, correct along the normal (`(3/2)h²`), but the column is
**tangentially offset** from the sub-face centroid by `δ = ±h/2` in *both* tangential directions.
The value is therefore wrong by `δ·∂_tφ = O(h)`: **first order**. The instrument's coarse-upwind
face error decomposes (`scratch diag`, N = 16/32/64):

| N | total | normal extrapolation | tangential offset |
|---|---|---|---|
| 16 | 1.610e-1 | 9.90e-2 | 1.223e-1 |
| 32 | 5.208e-2 | 1.50e-2 | 4.949e-2 |
| 64 | 2.407e-2 | 4.98e-3 | 2.345e-2 |

The tangential part halves per doubling (first order); the normal part is second order.

That O(h) face error is single-valued, so its consequences on the two sides differ. The four
sub-faces of one coarse face carry `±δ` in each direction: the errors **sum to zero over the coarse
face**, so the coarse cell's budget is barely touched — but each **fine** cell sees its own
sub-face's O(h) error against its other face's O(h²), divided by `h`: an **O(1)** local
truncation, alternating in sign across the 2×2 fine patch. That is the zeroth-order seam
truncation the instrument sees (fine interface cells 1.10e-2 at N = 64 against 2.78e-4 in the bulk,
and converging ×2 per doubling), and the alternating fine-layer forcing is the *shape* excess
`amr_tg_graded.md` §4(6) measures.

Two more faces are broken, both same-level faces the brief's instrument lumped into "bulk":

* **Case 2** — the fine cell `F`'s *downstream* face (flow `C → F → F2`): `upup = C` sits at
  `2h` from that face, not `1.5h`, *and* is tangentially offset by `δ`. The (1.5, −0.5) weights
  give `φ_F + (3/4)h·φ' + (1/2)δ·∂_tφ` against the exact `φ_F + (1/2)h·φ'`: **two O(h) errors**.
* **Case 3** — the coarse cell `C`'s downstream face (flow `F → C → C2`): `upup` is *one* fine
  leaf (whichever `periodicNeighbor` finds) at `2.5h` with a `δ` offset: the same two O(h) errors.

Cells owning a case-2/3 face but no sub-face ("layer 2") have truncation 4.61e-3 at N = 64 —
**16× the true bulk** (2.78e-4). The brief's "bulk" column was this layer; its own convergence
(N-order 2.0 instead of 3) was the symptom.

**Why both earlier prototypes under-delivered.** `star` sampled `φ_up` and `φ_upup` *independently*
and then combined them with (1.5, −0.5); on the staircase, where a tangential neighbour of `up`
is refined, the level check skipped `up` on 54 % of its lookups but `upup` on only 11 %, so the
correction entered with weight **−0.5 instead of +1** — worse than nothing. (My first prototype
made the mirror mistake, sampling only `up`: weight 1.5.) `dist` fixed the distance of case 2/3 but
not their offset. The correct rule applies the tangential correction **once, to the extrapolated
value** — §5.

## 3. What the literature does, and whether to follow

* **Berger & Colella (JCP 82, 1989).** The coarse/fine flux is always computed *by the fine grid*:
  fine ghost cells are filled by conservative piecewise-linear interpolation from the coarse grid
  (slopes limited, tangentially interpolated to the fine positions), the fine scheme runs on them,
  and the coarse flux is replaced by the sum of fine fluxes (refluxing). Composed, a linear normal
  fill from the coarse column plus a linear tangential sample is *exactly* "extrapolate on the
  coarse column, then correct tangentially" — our case 1. Their coarse level is stored everywhere
  under the fine patches, so the coarse value beside a refined region exists as a restricted
  (child-averaged) value: our block/layer probes are the leaf-octree transcription of that.
* **Martin & Colella (JCP 163, 2000); Martin, Colella & Graves (JCP 227, 2008).** Cell-centred AMR
  projection: coarse values are interpolated **quadratically along the interface** to the fine
  tangential position (the `D_t`, `D_tt` form `coarseStar`/`cfAppendStencil` already use), then
  quadratically **along the normal** through the two nearest *fine* cells and that value. The
  normal quadratic mixes the downstream fine cells into the seam value.
* **Chombo / EBChombo.** The same fill; near the embedded boundary the ghost fill falls back to a
  least-squares reconstruction "from the available cells" — the per-face gate of
  `docs/amr_cf_flux_gate.md` is this repo's version of that fallback, and this note inherits it.
* **Gerris / Basilisk (Popinet, JCP 190, 2003).** Leaf octree like ours: resolution-boundary
  "halo" cells are filled linearly from the parent using the parent's gradient; fluxes at the
  resolution boundary are single-valued from the fine side. Linear, upwind-side, unlimited.
* **Order theory.** A conservative scheme whose flux error is one order lower on a codimension-1
  set keeps its global order (Gustafsson, Math. Comp. 29, 1975; the supraconvergence results of
  Kreiss, Manteuffel, Swartz, Wendroff & White, Math. Comp. 47, 1986). This is why the item is a
  *constant* and why the seam's local truncation can be first order at best (§4).

**Follow Berger–Colella / Popinet, not Martin–Colella's normal quadratic.** The upwind-side
linear reconstruction keeps the SOU/Koren character at the seam (no downstream data enters the
advected value), which is what the throat meshes' stability and the out-of-scope amplitude half
both depend on. Deviations, both deliberate: (i) **no slope limiting** of the tangential sample —
the momentum field is smooth, SOU itself is unlimited, and minmod measured 14 % worse (2.457e-3 vs
2.156e-3 at N = 64); the bounded remedy, if ever needed, is §11 Q-D. (ii) Case 2/3 use the probe
at its **true distance** instead of a ghost at fine spacing filled with the coarse slope: same
order, one fewer cell, and it spreads the unavoidable jump of the flux-error constant over two
faces (§4, the telescoping argument).

## 4. The objective — pointwise face value or cancelling errors? (the brief's question 1)

Cell truncation is `(ε_down − ε_up)/width` for a cell with faces of flux-error `ε`. Three facts:

1. **For the coarse-upwind sub-face the two objectives coincide.** After the tangential fix its
   error is the coarse column's own `(3/2)h²·φ_nn`, i.e. *the same as the coarse cell's upstream
   regular face*: the coarse cell with outflow into the fine region becomes bulk-like (measured:
   case 1 leaves `if C` unchanged, 3.79e-3 → 3.80e-3, exactly as the zero-sum argument says), and
   the fine cells lose their O(1). "Make the four sub-face errors cancel against the coarse opposite
   face" is *already* what the correct pointwise rule does, because the correction is a pure
   redistribution among the four sub-faces. `star`'s "worse pointwise, better divergence" was the
   sign-flip artefact of §2, not a different objective at work.
2. **The h → 2h jump in the flux-error constant is intrinsic and cannot be cancelled by any
   sub-face stencil.** With every face second order, flow `C2 → C → F → F2` sees face errors
   `3/2 | 3/2 | 1/2 | 3/8` (×h²φ_nn): `C` gets 0, `F` gets `−h`, `F2` gets `−h/8`. Flow
   `F2 → F → C → C2` sees `3/8 | 3/8 | 5/4 | 3/2`: `F` gets 0, `C` gets `(7/16)h`, `C2` `(1/8)h`.
   The sum over the seam cells telescopes to the difference of the two regular constants whatever
   the seam stencil does. First order on the seam is the floor, and the gate "interface/bulk ratio
   must stop growing" (brief §8) is unattainable by construction — the right gate is the seam
   converging at ×4 per doubling (first order) and the layer behind it becoming bulk-like (§10).
3. **Where there is freedom — how the jump is spread — the true-distance probes are the L2-better
   choice, by the same arithmetic.** Case 3 with the 4-cell face layer at `2.5h` gives the seam
   pair `(7/16, 1/8)h`, with the 8-child block at `3h` it gives `(9/16, 0)h`: L2 ratio
   `sqrt(0.207/0.316) = 0.81`. Measured `if C`: 2.040e-3 vs 2.516e-3 = **0.81**. The design is
   therefore pointwise-consistent upwind reconstruction (fact 1) plus true-distance probes (fact 3);
   no separate "cancellation" scheme exists to build.

## 5. The design

### 5.1 The rule (one reconstruction for every face; only seam faces take the new path)

For face slot `k` with upwind cell `U`, downwind `D`, face axis `a`, after `velOut` fixes the
direction exactly as today:

```
gUp      = (φ_U − φ_UU*) / dUU                 upwind-side slope on U's OWN column
SOU  :   φ_face = φ_U* + d1 · gUp
Koren:   den = φ_D* − φ_U*;   r = gUp · dD / den (0 if |den| < 1e-10);   ψ = Koren(r)
         φ_face = φ_U* + (d1/dD) · ψ · den
```

with `d1` = half width of `U` along `a`, `dD = dist(k)` (centre to centre, already in `FaceGeom`),
`dUU` the distance from `U`'s centre to the upstream probe point, and

* `φ_U*` = `φ_U` + **sample correction** `[s(U, D) − φ_U]` when `U` is the *coarse* cell of a 2:1
  sub-face (case 1); otherwise `φ_U`.
* `φ_D*` = `φ_D` + `[s(D, U) − φ_D]` when `D` is the coarse cell of a 2:1 sub-face (Koren only —
  for SOU `φ_D*` cancels; do not gather it); otherwise `φ_D`.
* `φ_UU*` = the **upstream probe**: the single leaf `upupI/J` when it is at `U`'s level (`dUU` =
  centre to centre); the **sample** `s(UU, U)` when `UU` is *coarser* (case 2 — `UU` and `U` are
  then a 2:1 pair; `dUU = 1.5·width(U)`); the **face-layer mean** of the four fine cells adjacent
  to `U` across that face when the far side is *finer* (case 3; `dUU = 0.75·width(U)`).

`s(C, F)` — *the coarse cell `C` sampled at the column of its fine sub-face neighbour `F`* — is
`cf_scheme.hpp::cfAppendStencil(ap, out, C, F, axis, scale = 1, fluidOk, scheme = quadratic)`
**verbatim**: `φ_C` plus, per tangential axis, the two-sided quadratic `δ·D_t + ½δ²·D_tt` from
`C`'s ± neighbours (a finer neighbour = its 2^3-child volume mean, the register's island-corner
rule; a coarser one, a solid or a closed face = that side dropped, and the builder's own
one-sided-linear / raw fallbacks apply). Weights sum to 1; on cubes with `δ = ±H/4` the two-sided
weights are `(7/8; 5/32, −3/32; 5/32, −3/32)`.

Regular-face limit of the rule (`d1 = h/2, dUU = dD = h`): SOU gives `φ_U + ½(φ_U − φ_UU)`, Koren
gives today's formula exactly — but `φ_U + ½(φ_U − φ_UU)` is **not bit-identical** to
`1.5φ_U − 0.5φ_UU`, so plain slots never take this path (§5.3).

Applying the sample correction *once* (to `φ_U*`, not inside `gUp`) is the whole point: the slope
`gUp` stays on `C`'s column, where `C` and `CC` are both at the same offset, and the correction
enters with weight exactly 1. Sampling `CC` too would be equivalent at second order and costs a
second stencil; sampling only one of them without this split is the §2 bug.

### 5.2 What is prebuilt (device Views, built on the host in `setSolid` after the C/F overlays)

```
FaceGeom +=
  View<Index>  seam;        // size nFaces: descriptor id, −1 = plain slot (int32 is enough)
  // per seam descriptor (only slots where at least one field is live):
  View<Index>  sampI, sampJ;      // record of s(i, j) / s(j, i): ≥ 0 iff i / j is the COARSE
                                  // cell of this 2:1 slot (at most one of the two)
  View<Index>  uuRecI, uuRecJ;    // upstream-probe record when up == i / up == j crosses a
                                  // level: a sample record (coarser) or a layer record (finer);
                                  // −1 = use upupI / upupJ as today
  View<double> d1I, d1J;          // half width of i / of j along the face axis (physical)
  // records, one CSR:
  View<Index>  recStart;          // nRec + 1
  View<double> recDist;           // dUU for upstream use (1.5·width(F) for a sample record used
                                  // by F; 0.75·width(C) for a layer record); unused for sampI/J
  View<Index>  recCell;  View<double> recW;
```

Records: one **sample record** per 2:1 sub-face pair `(C, F)` with `regular(C) && regular(F)`
(the ghost-visible flag `cf_regular`, `flow.hpp:911–938`; the gate of `amr_cf_flux_gate.md`),
entries from `cfAppendStencil`; one **layer record** per (coarse cell `C`, face) whose far side is
refined, entries = the four sub-face neighbours of that face in `forEachFaceFull`'s sub-face order,
weight ¼ each, built only if `regular(C)` and all four are fluid. A sample record is referenced by
the sub-face slot in both rows (as `sampI` or `sampJ`) and by `F`'s opposite-face slot in both of
*its* rows (as `uuRecI/J`, case 2). A layer record is referenced by `C`'s opposite-face slot in
both rows (case 3). A slot with all four fields −1 gets no descriptor (`seam = −1`).

Three-level neighbourhoods (`lmax ≥ 2`) need no extra rule: a coarser upstream of a coarse `U` is
again a 2:1 pair `(UU, U)` with its own sample record; a finer upstream of a fine `U` is again a
layer record. The rule is level-relative.

### 5.3 The kernel (both `deferredSou` and `advectExplicit`; `buildFou` untouched)

```
for k in row(i):  j = nb(k); if (!fl(j)) continue;  velOut as today; up/down as today;
  if (!seamOn || seam(k) < 0) { phiFace = hoFaceValue(phiUpUp, phiUp, phiDown, scheme); }  // VERBATIM today's lines
  else {
    s = seam(k);  upIsI = velOut > 0;
    rs   = upIsI ? sampI(s) : sampJ(s);    ru = upIsI ? uuRecI(s) : uuRecJ(s);
    rd   = upIsI ? sampJ(s) : sampI(s);    d1 = upIsI ? d1I(s) : d1J(s);
    phiUp   = fld(up);   upStar = (rs >= 0) ? gather(rs) : phiUp;
    if (ru >= 0) { uu = gather(ru); dUU = recDist(ru); }
    else         { uu = (upup >= 0 && fl(upup)) ? fld(upup) : phiUp;  dUU = dist-of-(up,upup); }
    downStar = (scheme == Koren && rd >= 0) ? gather(rd) : fld(down);
    phiFace = hoFaceValueSeam(uu, phiUp, upStar, downStar, d1, dUU, dist(k), scheme);
  }
  sou += ra(k) * velOut * phiFace;   fou += ra(k) * velOut * fld(up);   // fou unchanged
```

`gather(r) = Σ_{e ∈ [recStart(r), recStart(r+1))} recW(e)·fld(recCell(e))` in record order, no
reordering — that is what makes np = 1 bitwise against single-rank. The plain-`upup` `dUU` in the
`ru < 0` branch is needed only when `uu` is at `up`'s level, i.e. `dUU = 2·d1`. `hoFaceValueSeam`
lives beside `hoFaceValue` in `advect_recon.hpp`. `seamOn` is the ablation flag (§5.5). Factor the
per-face reconstruction into one inline helper both kernels call, so they cannot drift.

### 5.4 Cut cells, block seams, missing neighbours, anisotropy

* **Cut cells.** Per-face gate, as for every C/F correction: no sample record unless
  `regular(C) && regular(F)`; no layer record unless `regular(C)` and the four fine cells are
  fluid; inside `cfAppendStencil` the existing `fluidOk` / openness gates drop a side. Withheld ⇒
  the slot has no descriptor ⇒ **today's arithmetic, bit for bit** at cut seams. Gates only
  shrink output; `num_cf_cut_faces` counts them.
* **Block seams (MPI).** All probed cells are face neighbours of `C` (or `CC`, the shipped probe):
  the ±2 leaf reach the kernels already use. Sample records need nothing new —
  `probeCfScheme()` (`flow.hpp:1942`) already collects every coordinate `cfAppendStencil` can
  touch from the coarse cell of every local C/F face, ghost coarse included. Layer records for a
  local row `C2` whose neighbour `C` is a *ghost* need `C`'s far-side face layer: add
  `probeSeamLayer()` beside `probeCfScheme()` in the same miss-collect phase (`flow.hpp:1891`):
  for each local face `(i, j)` at equal level whose far neighbour of `j` along `(axis, dir)` is
  finer, `probeSlot` the four face-layer corners `lo(j) + width(j)·e_axis` (or `− 1` for
  `dir < 0`) at tangential offsets `{0, h}²`. Values read through ghost slots as every CSR does.
* **Missing neighbours.** Non-periodic boundary or an unresolved probe (`< 0`): the record is not
  built (the plain path stays), never a partial record.
* **Anisotropic roots (`amr_anisotropic.md`).** Every distance is per axis: `d1I/J`, `recDist` are
  stored in physical units by the builder (`h0[axis]·2^level·…`); `cfAppendStencil` is already
  per-axis. The kernel does no `2^level` arithmetic.

### 5.5 Switch

`AmrFlow::setSeamReconstruction(bool)` / `Flow.diagnostics.set_seam_reconstruction(bool)`
(developer tier, like `set_uf_advection`), default **on**; the builder always builds, the kernel
captures one bool. Off must be bit-identical to today (gate G0). Not an environment variable.

## 6. Why it cannot destabilise the throat meshes

1. **The implicit part is untouched.** `buildFou` and the momentum matrix do not change; the
   deferred correction `ρ(SOU − FOU)` is what changes, and only on seam slots. The implicit FOU is
   what carries dt = 60 … 1e20 (brief §5); it carries exactly what it carried.
2. **Upwind-side data only.** Every value in `φ_face` comes from `U`'s side: `C`'s column and `C`'s
   tangential neighbours (case 1), `F` and the coarse cell behind it (case 2), `C` and the fine
   layer behind it (case 3). For SOU no downstream value enters at all; for Koren the downstream
   value enters only through the limiter, as today. The fixed-point character of the deferred
   correction at large dt is that of today's SOU. (The tempting alternative — tangential slopes
   from the four *downstream* fine cells, always available — was rejected on exactly this
   mechanism: it makes the fine cell's own value enter its inflow with a positive coefficient,
   `+u_n/h`, an anti-diffusive tangential coupling inside the 2×2 patch.)
3. **Bounded, zero-sum correction.** The sample correction has weights summing to zero and
   `|corr| ≤ ¼·|Δ_tφ| + 1/32·|Δ_ttφ|` per tangential axis; the four sub-faces' corrections sum to
   zero, so the coarse cell's outflow is unchanged and the fine cells receive a redistribution.
   Case 2/3 change the extrapolation weight from ½ to ⅓ and ⅔.
4. **Cut seams are bit-identical** (§5.4) — and the throat meshes' hazard region is the cut band.
5. **The gate is still run** (G6): 12 throat meshes, cf = 1, 400 steps, all finite, ±0.3 %.

## 7. Cost

* **Memory.** `seam`: one int32 per face slot (N = 64 TG: 48 448 leaves × ~7 slots ≈ 1.4 MB).
  Descriptors: ~2·(2:1 slots) + 2·(case-2 slots) + 2·(case-3 slots) ≈ 45 k at N = 64, 48 B each
  ≈ 2 MB. Records: 9 984 sample records × 5–33 entries (mean ≈ 13 on this staircase shell) plus
  ~2 500 layer records × 4, 16 B/entry ≈ 2.3 MB. Total ≈ 6 MB against the FaceGeom's ≈ 30 MB;
  in a bed the seam fraction is smaller.
* **Per step.** Bulk slots: one extra int32 load per slot in the two advective kernels (the
  ×3 components). Seam slots: ≤ ~40 extra loads. Expected < 2 % on the advective phase
  (`PECLET_AMR_PROFILE_STEP`), < 0.5 % on the step. Gate: +3 % / +1 %.
* **Setup.** One more pass over local rows in `setSolid` (records) and one probe pass; both
  linear in the seam size. Expected < 1 % of `setSolid`.
* **MPI.** No new exchanged field, no new message: the regular flag is already ghost-visible, the
  velocity ghost tails are already synced before the kernels (`flow.hpp:1241`), and the probes
  ride the existing miss-collect fixpoint.

## 8. Work orders (commit-sized, in order; Opus unless marked)

**WO0 — the instrument carries the design (Sonnet).** Fold `amr_cf_seam_proto.py` into
`tests/study/amr_cf_advection.py --flux` as mode `seam` (keep `none`; drop `star`/`dist`/`both`
or keep them labelled "history — weight bug, see design note §2") and add the class split (`if C`,
`if F`, `layer2`, true `bulk`). Acceptance: the mode reproduces Appendix A to all printed digits.

**WO1 — inert plumbing.** `FaceGeom` fields of §5.2, the host builder in `setSolid` after the four
`buildCf*Delta` calls (it reuses their `regularOk`/`fluidOk` and `cfAppendStencil`),
`probeSeamLayer()`, and `Flow.diagnostics.face_topology()` extended with the descriptor and
record arrays. No kernel reads them. Acceptance: (a) the byte gate is identical (np = 1 and 2);
(b) a new Python test builds the G mesh at N = 32 and checks the C++ records against the numpy
rule cell by cell — 2 688 sample records, entry sets and weights equal to 1e-15, the layer
records' cells equal to the row's sub-face neighbours; (c) `mpirun -np 2` and `-np 4` produce
the same record multiset as np = 1 (compare after mapping ghost slots to global ids).

**WO2 — the kernels.** `hoFaceValueSeam` in `advect_recon.hpp`; the seam branch of §5.3 in a
shared inline helper used by `deferredSou` and `advectExplicit`; `setSeamReconstruction` +
binding + `_amr.pyi`. Acceptance: G0, G1, G2 below; the byte gate re-recorded **in its own
commit** with the list of moved keys (expected: `amr.flow_ghost`, `amr.distributed_flow`,
`amr.distributed_flow_np2`, and `amr.flow_sampled` only if its scenario runs advection on a mesh
with 2:1 faces; every octree/adapt/poisson/rebalance key unchanged).

**WO3 — the headline.** Run `tests/study/amr_tg_graded.py` at N = 32, 64, CFL 0.5, both arms
(C is unchanged by construction — verify bit-identical); re-record the `python_amr_tg_graded`
reference levels in their own commit; update `amr_tg_graded.md` §3–§4(6) and ROADMAP B5 with the
measured numbers. Acceptance: G3.

**WO4 — distributed.** Extend `tests/test_amr_distributed_seam_mpi.cpp` (or the
`amr_distributed_cf` binary) with an advection-on march on a graded mesh: np = 1 bitwise vs
single-rank, np = 2/4/8 in the ~3e-7 class. Acceptance: G4.

**WO5 — stability and cost.** `amr_two_sphere_gap.py` battery; `bench_amr_flow` and
`PECLET_AMR_PROFILE_STEP=1` before/after. Acceptance: G5, G6.

**WO6 — the record.** Register entry (`../docs/decisions/amr.md`): *"Seam reconstruction of the
advected value: upwind-side, level-aware probes (coarse upwind tangentially sampled by
`cfAppendStencil`, applied once; coarser/finer upstream probes at true distance). Rejected:
Martin–Colella quadratic normal fill (mixes downstream fine cells; amplitude/stability), minmod
tangential slope (+14 %), 8-child block probe for the normal upstream (+23 % on coarse seam
cells; needs ±2 children), downstream-patch tangential slopes (anti-diffusive)."* Close B5 in
the ROADMAP with the G3 numbers; open the amplitude-half item if G3 leaves `m3` G/C > 1.2.

## 9. Verification gates (falsifiable; N = 64 unless stated; today → required, expected)

* **G0 — inertness.** `set_seam_reconstruction(False)`: byte gate identical to today's reference
  at np = 1 and 2; `--step` with advection **off** identical to today with the switch on or off
  (1.129e-3 / 5.140e-4 at N = 64, to all digits — the change is inert without advection).
  `div(uf) ≤ 4e-12` unchanged (the flux is single-valued; `uf` is not touched).
* **G1 — the operator** (`amr_cf_advection.py --flux`, mode `seam`, Appendix A): interface
  divergence rms 7.499e-3 → **≤ 2.5e-3** (2.156e-3); fine-interface class 1.098e-2 → ≤ 2.8e-3
  (2.324e-3); coarse-interface 3.792e-3 → ≤ 2.3e-3 (2.040e-3); layer-2 4.609e-3 → ≤ 5e-4
  (4.024e-4); true bulk **2.779e-4 unchanged to all printed digits**; coarse-upwind face error
  2.407e-2 → ≤ 6.5e-3 (5.588e-3). Convergence 32 → 64 of the interface rms **≥ 3.4×** (3.73×;
  today 2.37×) — first order on the seam. The ratio to the *true* bulk becomes ≈ 8 (from 27) and
  is *expected to grow ×2 per doubling* thereafter; that is the intrinsic seam order, not a defect.
* **G2 — the whole solver, one step** (`--step`, advection on): interface rms 7.881e-3 →
  **≤ 3.5e-3** (≈ 2.5e-3: the operator's 2.2e-3 plus the advection-off 1.1e-3 in quadrature);
  interface/bulk 9.23 → ≤ 5 (≈ 4). At N = 32: 1.774e-2 → ≤ 9e-3.
* **G3 — the headline** (`amr_tg_graded.py`, N = 64, CFL 0.5). Shape error of G: 7.417e-3 →
  **≤ 3.6e-3** (≈ 3.0e-3; C is 2.265e-3), i.e. G/C shape 3.27 → ≤ 1.6 (≈ 1.3). `m3` of G:
  1.4328e-2 → ≤ 1.30e-2 (≈ 1.26e-2), G/C 1.56 → ≤ 1.45 (≈ 1.37 — the amplitude half, 1.96e-2 vs
  1.43e-2 in `1−amp`, is untouched by design; if `1−amp` also drops, say so). At N = 32 the
  §14.4c "G-only shape excess" 2.51e-2 → ≤ 1.2e-2. **Falsification:** if G/C shape stays > 2.0
  with G1 met, the premise that the seam reconstruction is the shape excess is wrong — stop,
  report, do not tune.
* **G4 — distributed.** np = 1 bitwise vs single-rank; np = 2/4/8 within the ~3e-7 march class;
  record multisets identical across np (WO1c).
* **G5 — stability.** 12 throat meshes, cf = 1, 400 steps: all finite, policy error within
  ±0.3 % of today's.
* **G6 — cost.** `bench_amr_flow` step ≤ +1 %; advective phase ≤ +3 %; `setSolid` ≤ +1 %.
* **Battery.** 96 C++ + 5 `bench` + 5 `python` ctests green (106); `python_amr_tg_graded`
  re-recorded (WO3); byte gate re-recorded (WO2).

## 10. Risks and open questions (each with its default)

* **Q-A (preference) — quadratic vs linear tangential sample.** Reusing `cfAppendStencil` keeps
  the `D_tt` term (2.156e-3) ; linear-only measures 2.280e-3 and would need a new builder. Default:
  **reuse verbatim**, one tangential-sample operator suite-wide.
* **Q-B (fact) — does the fix move the amplitude half?** Unknown; G3 measures `1−amp` for free.
  Default: report it; if `m3` G/C stays > 1.2, open a ROADMAP item for the seam fluxes' numerical
  viscosity (limiter-/scheme-responsive per §14.4c) — a separate decision, not this one.
* **Q-C (preference) — ship with or without the ablation switch.** Default: **with**, developer
  tier, default on; it is what makes G0 and the instrument's ablations possible.
* **Q-D (fact) — Koren boundedness on a bed.** The unlimited tangential sample can leave
  `[min, max]` of its cells by ≤ ¼ of the tangential jump; cut seams are gated, so this needs a
  tangential velocity discontinuity in open fluid to matter. Default: ship unlimited; the remedy,
  if a bed run ever shows a seam overshoot, is a **clamp of the gathered sample to the min/max of
  its record's cells** (three lines in `gather`, monotone, no new data) — not minmod.
* **Q-E (fact) — the N = 16 / thin-shell regime.** At a four-cell shell radius the fallbacks fire
  more and the gain is 5.40e-2 → 2.92e-2 (1.8×) instead of 3.5×. Expected; no action.
* **Q-F (preference) — priority.** This note's recommendation: **medium**, up from low–medium.
  The fix is small (one builder reusing an existing stencil, one kernel branch), local, inert
  outside advection-on graded runs, and removes ~70 % of the item's stated defect; the seam of a
  bed's refined band lies in open fluid, so beds benefit too. Estimated two to three Opus days
  including gates.
* **Risk — the `--step` and G3 predictions are extrapolations** from the operator's truncation
  through a solver with projection and time stepping; they carry ±40 %. The gates are set at the
  loose end; the falsification clause in G3 is what protects against tuning to a wrong premise.
* **Risk — `hoFaceValueSeam` on plain slots.** Never route a plain slot through it: it is not
  bit-identical to `hoFaceValue` (§5.1). WO2's G0 catches this.

## Appendix A — the prototype table (`tests/study/amr_cf_seam_proto.py 16 32 64`)

Columns: divergence-error rms by cell class (interface = owns a 2:1 face; `layer2` = owns a
case-2/3 face but no 2:1 face; `bulk` = the rest; `bulk(old)` = the brief's "not interface"), and
face-value rms by face class (coarse-upwind sub-face, fine-upwind sub-face, case 2, case 3, other
regular faces). "design" = §5 as specified (quadratic tangential sample, block for a finer
tangential neighbour, face layer for case 3).

```
# N=16: 768 (C,F) sub-face pairs, 768 case-2 slots, 192 case-3 slots (flow-direction dependent)
                               variant    div if      if C      if F    layer2      bulk bulk(old) |     f upC     f upF      f c2      f c3     f reg
                                  none 5.395e-02 2.990e-02 8.437e-02 3.444e-02 1.877e-02 2.428e-02 | 1.610e-01 2.643e-02 7.489e-02 9.992e-02 4.766e-02
   design (quad, block tan., layer c3) 2.915e-02 3.012e-02 2.701e-02 2.163e-02 1.877e-02 1.962e-02 | 1.014e-01 2.643e-02 2.557e-02 8.322e-02 4.766e-02
                           case 1 only 4.016e-02 3.018e-02 5.548e-02 3.444e-02 1.877e-02 2.428e-02 | 1.014e-01 2.643e-02 7.489e-02 9.992e-02 4.766e-02
                             cases 1+2 2.919e-02 3.018e-02 2.701e-02 2.698e-02 1.877e-02 2.142e-02 | 1.014e-01 2.643e-02 2.557e-02 9.992e-02 4.766e-02
           cases 2+3 (no sub-face fix) 3.742e-02 2.928e-02 5.033e-02 2.163e-02 1.877e-02 1.962e-02 | 1.610e-01 2.643e-02 2.557e-02 8.322e-02 4.766e-02
           linear tangential (no D_tt) 2.999e-02 3.006e-02 2.984e-02 2.155e-02 1.877e-02 1.960e-02 | 1.093e-01 2.643e-02 2.376e-02 8.322e-02 4.766e-02
                     minmod tangential 3.257e-02 3.012e-02 3.716e-02 2.181e-02 1.877e-02 1.968e-02 | 1.167e-01 2.643e-02 3.050e-02 8.322e-02 4.766e-02
                block probe for case 3 3.066e-02 3.227e-02 2.701e-02 2.328e-02 1.877e-02 2.015e-02 | 1.014e-01 2.643e-02 2.557e-02 9.884e-02 4.766e-02
            layer probe for tangential 2.921e-02 3.017e-02 2.711e-02 2.163e-02 1.877e-02 1.962e-02 | 1.021e-01 2.643e-02 2.537e-02 8.322e-02 4.766e-02
# N=32: 2688 (C,F) sub-face pairs, 2688 case-2 slots, 672 case-3 slots (flow-direction dependent)
                               variant    div if      if C      if F    layer2      bulk bulk(old) |     f upC     f upF      f c2      f c3     f reg
                                  none 1.781e-02 1.061e-02 2.537e-02 9.969e-03 2.314e-03 4.082e-03 | 5.208e-02 7.284e-03 3.721e-02 4.412e-02 1.661e-02
   design (quad, block tan., layer c3) 8.047e-03 8.386e-03 7.475e-03 2.091e-03 2.314e-03 2.288e-03 | 1.769e-02 7.284e-03 6.002e-03 2.432e-02 1.661e-02
                           case 1 only 1.330e-02 1.065e-02 1.667e-02 9.969e-03 2.314e-03 4.082e-03 | 1.769e-02 7.284e-03 3.721e-02 4.412e-02 1.661e-02
                             cases 1+2 9.549e-03 1.065e-02 7.475e-03 5.981e-03 2.314e-03 3.002e-03 | 1.769e-02 7.284e-03 6.002e-03 4.412e-02 1.661e-02
           cases 2+3 (no sub-face fix) 1.101e-02 8.354e-03 1.426e-02 2.091e-03 2.314e-03 2.288e-03 | 5.208e-02 7.284e-03 6.002e-03 2.432e-02 1.661e-02
           linear tangential (no D_tt) 8.318e-03 8.422e-03 8.149e-03 2.058e-03 2.314e-03 2.284e-03 | 1.905e-02 7.284e-03 5.669e-03 2.432e-02 1.661e-02
                     minmod tangential 8.538e-03 8.411e-03 8.736e-03 2.175e-03 2.314e-03 2.297e-03 | 2.133e-02 7.284e-03 6.694e-03 2.432e-02 1.661e-02
                block probe for case 3 9.124e-03 1.002e-02 7.475e-03 2.431e-03 2.314e-03 2.328e-03 | 1.769e-02 7.284e-03 6.002e-03 2.980e-02 1.661e-02
            layer probe for tangential 8.140e-03 8.407e-03 7.694e-03 2.092e-03 2.314e-03 2.288e-03 | 1.820e-02 7.284e-03 5.930e-03 2.432e-02 1.661e-02
# N=64: 9984 (C,F) sub-face pairs, 9984 case-2 slots, 2496 case-3 slots (flow-direction dependent)
                               variant    div if      if C      if F    layer2      bulk bulk(old) |     f upC     f upF      f c2      f c3     f reg
                                  none 7.499e-03 3.792e-03 1.098e-02 4.609e-03 2.779e-04 1.016e-03 | 2.407e-02 1.789e-03 1.686e-02 1.626e-02 4.881e-03
   design (quad, block tan., layer c3) 2.156e-03 2.040e-03 2.324e-03 4.024e-04 2.779e-04 2.847e-04 | 5.588e-03 1.789e-03 1.895e-03 6.432e-03 4.881e-03
                           case 1 only 5.291e-03 3.795e-03 6.994e-03 4.609e-03 2.779e-04 1.016e-03 | 5.588e-03 1.789e-03 1.686e-02 1.626e-02 4.881e-03
                             cases 1+2 3.295e-03 3.795e-03 2.324e-03 2.995e-03 2.779e-04 6.921e-04 | 5.588e-03 1.789e-03 1.895e-03 1.626e-02 4.881e-03
           cases 2+3 (no sub-face fix) 4.380e-03 2.043e-03 6.503e-03 4.024e-04 2.779e-04 2.847e-04 | 2.407e-02 1.789e-03 1.895e-03 6.432e-03 4.881e-03
           linear tangential (no D_tt) 2.280e-03 2.057e-03 2.587e-03 3.807e-04 2.779e-04 2.833e-04 | 6.045e-03 1.789e-03 1.760e-03 6.432e-03 4.881e-03
                     minmod tangential 2.457e-03 2.057e-03 2.970e-03 4.458e-04 2.779e-04 2.876e-04 | 6.808e-03 1.789e-03 2.002e-03 6.432e-03 4.881e-03
                block probe for case 3 2.442e-03 2.516e-03 2.324e-03 3.558e-04 2.779e-04 2.818e-04 | 5.588e-03 1.789e-03 1.895e-03 7.799e-03 4.881e-03
            layer probe for tangential 2.219e-03 2.060e-03 2.445e-03 3.989e-04 2.779e-04 2.844e-04 | 5.883e-03 1.789e-03 1.836e-03 6.432e-03 4.881e-03
```

Reading it: case 1 alone fixes the fine cells' sub-face (if F 1.10e-2 → 6.99e-3) but leaves
`layer2` and `bulk(old)` untouched; cases 2+3 alone fix the layer but not the sub-face; all three
are needed. The `bulk` column never moves — the design is inert away from the seam.

## Appendix B — the flux-error constants used in §4

Linear extrapolation from column points at distances `a`, `b` from the face: error `(ab/2)φ_nn`.

| face | `a` | `b` | error / (h²φ_nn) |
|---|---|---|---|
| regular fine | h/2 | 3h/2 | 3/8 |
| regular coarse | h | 3h | 3/2 |
| 2:1 sub-face, coarse upwind (fixed) | h | 3h | 3/2 (+ O(h²) tangential residual) |
| 2:1 sub-face, fine upwind | h/2 | 3h/2 | 3/8 |
| case 2, `s(C,F)` at true distance | h/2 | 2h | 1/2 |
| case 3, face layer | h | 5h/2 | 5/4 |
| case 3, 8-child block | h | 3h | 3/2 |
| shipped case 2 / case 3 | — | — | O(h): `(1/4)h φ' + (1/4)h ∂_tφ` |
| shipped coarse-upwind sub-face | — | — | O(h): `δ·∂_tφ`, `δ = ±h/2` twice |
