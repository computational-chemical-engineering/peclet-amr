# The C/F flux correction at a cut cell: gate per FACE, one emitter, one flux

> Design note, 2026-09-22, against `peclet-amr` `2426ef4` + branch `uffix` (assumed landed).
> Brief: `docs/briefs/cf_flux_gate_design.md`. Implementer: `opus-implementer`, work orders §9.
> No production code was written for this note.

## 1. The decision

**The quadratic C/F face value is applied on a 2:1 sub-face if and only if BOTH incident cells are
regular fluid (`fluid && !cut`), and that one per-face predicate — evaluated with a ghost-visible
cut flag — drives the divergence RHS, the ABC cell-gradient substitution and the advecting face
field alike, through one shared face-value emitter, so that `D_std(Δuf) ≡ Δ_cfDiv` holds
identically on every mesh, cut band at the level boundary included.**

Cut rows still receive no C/F delta (the recorded stability fix is kept in full); what changes is
that the regular row on the other side of a cut-adjacent face no longer receives it either. The
face reverts to the standard two-point value — the scheme every face ran under `cf=0` — instead of
carrying a correction that only one of its two cells booked.

The pressure matrix stays standard (`L = D_std·G_std`); the delta-on-a-standard-operator
formulation survives (§4). The configuration is not forbidden (§5, option E).

## 2. Problem and scope

**Problem.** `buildCfDivDelta` (the constraint's C/F correction, one row per CELL) is gated by
`rowRegular(i) = fluid(i) && !cut(i)`; `buildCfUfDelta` (the same correction in the advecting FACE
field, one row per face slot) is gated by `fluid(i) && fluid(j)`. A face flux is one number shared
by two cells. At a 2:1 sub-face with a cut cell on one side, the regular cell books the correction
in its constraint, the cut cell does not, and the face field carries it for both. The pressure
solve therefore balances the correction on one side only, and `uf` — whose sole purpose is to be a
conservative flux, `Σ_faces α·A·dir·uf = 0` per cell — has a permanent mass source at the cut cell.
Measured (`tests/study_amr_uf_div.cpp`, branch `ufgate`): on a mesh whose cut band meets the level
boundary (`refineToSdf(band=0)`, N=32, lmax=2, 336 of 2768 C/F rows cut),
`‖D(Δvel) − Δ_cfDiv‖ = 2.018e-01` and dominates `‖div(uf)‖ = 2.016e-01`; on a `band=3` mesh both
are ~1e-12 (there are no cut-adjacent C/F faces). It is proportional to the velocity, not to φ, so
it does not decay at steady state.

**In scope.** The gating rule and the code structure that makes the two books agree by
construction; the MPI ghost flag it needs; the tests and gates; the register entries.

**Out of scope.** The `uffix` φ-half fix (assumed landed); `cf=1` as default (done); the
distributed C/F discovery machinery (done); the ghost-projection face field at CUT rows — under the
ghost scheme `D_std(uf) ≠ 0` at cut rows by design (the overlay's constraint is a functional of the
cell velocities, documented at `divNormFace`), which is a separate, pre-existing property this note
does not touch; `peclet.flow`; the uniform-grid parity work; sub-face closures (§5 option C, deferred
with a default).

## 3. What the literature does at an irregular cell on a coarse/fine boundary

Three mechanisms, all three older than this repo, decide the question.

**(a) One flux per face, and every divergence — regular or irregular — is computed from it.**
Berger & Colella (J. Comput. Phys. 82, 1989, 64–84) introduced refluxing: after the fine level
advances, the coarse cell adjacent to the interface has its flux on that face *replaced* by the
sum of the fine sub-face fluxes, so the composite scheme is conservative. The embedded-boundary
line generalises this unchanged: the flux is aperture-weighted (`α·A·F`), the flux register
(Chombo `EBFluxRegister`; AMReX's `EBFluxRegister` documents "two versions of CrseAdd, one for
regular ... and the other for ... cutcells") holds the coarse–fine mismatch, and the coarse
irregular cell's divergence is formed from the refluxed value. There is no per-cell exception:
irregular cells at the interface are refluxed like any other. Trebotich & Graves (Comm. App. Math.
Comp. Sci. 10, 2015, 43–82; EB + AMR incompressible Navier–Stokes, the closest published relative
of this solver) state the two operator-level rules verbatim: *"Projection operator gradients are
matched at coarse-fine interfaces by simple averaging as in [30]"* (Martin & Colella, J. Comput.
Phys. 163, 2000, 271–312) and, for advection, *"at coarse-fine boundaries, we use linear
interpolation and flux matching [31]"* (Martin, Colella & Graves, J. Comput. Phys. 227, 2008,
1863–1886).

**(b) The order of the C/F interpolation is a property of the FACE's stencil, and it degrades
where the stencil is unavailable — it is never a property of a cell's row.** Trebotich & Graves:
*"For adaptive calculations, we use higher-order (quadratic) interpolation to fill ghost cells for
second-order elliptic operators (Laplacian) at coarse-fine boundaries in order to avoid O(1)
truncation error [30]"* — quadratic where it can be built. chombo-discharge (the current
open-source Chombo-EB descendant) documents the fallback explicitly: ghost cells near the EB are
*"reconstruct[ed] ... (to specified order) ... using the available cells around the ghost cell"*,
and *"this reconstruction does not use coarse-level grid cells that are covered by the fine
level"*. This is exactly the structure `cfAppendStencil` already has: per tangential axis, the
quadratic where both tangential samples exist and are open, the one-sided linear where one does,
the raw coarse value where neither. The only per-ROW gate in the whole family is the outer
`rowOk` — and it is the one that broke flux form.

**(c) The embedded boundary crosses level boundaries as a matter of course.** In Chombo-EB the
geometry is generated on the finest level and coarsened by graph coarsening (*"The volume of a
coarse cell is exactly the volume of the fine cells that it comprises"*), so cut cells exist at
every level and the EB meets every refinement boundary that reaches the wall. None of these codes
forbid the configuration; they handle it with (a) + (b).

**Should this repo follow?** Yes, and it already does in structure: the face field `uf` is the
MAC-projected advection velocity of Martin & Colella (exactly divergence-free in the flux-form
divergence because the matrix is `D_std·G_face` on the same sub-faces), the cell field is the
approximate projection, and the coarse row of a 2:1 face already telescopes the fine sub-face
values ("conservative telescoping"). The one deviation is the row gate. The design below removes
it and replaces it with (b): a per-face availability rule, whose fallback is the standard face
value. The canonical move does dissolve the question — once the face value is single-valued and
every divergence is computed from it, there is nothing left to gate per cell.

## 4. Constraints and invariants

**The projection identity.** With `F` the face-average operator (standard ½/½ plus the C/F
face-value delta), `Gf` the compact face gradient `(φ₊−φ₋)/d`, and `D_std` the flux-form
divergence `invV·Σ α·A·dir·(·)`:

    uf  = F(u*) − Gf φ
    rhs = D_std F(u*)            (= divergence(geom_) + cfDiv_ + ghost overlay)
    L   = D_std Gf               (the standard matrix, MG/PCG/ghost-BiCGStab untouched)
    ⇒ D_std uf = rhs − Lφ = solver residual        … (I)

(I) is what makes `uf` a conservative flux. It holds iff **every term of `uf` has its counterpart
in `rhs` or in `L`**: a term in `F` must be in both `rhs` and `uf` (same face, same value, same
gate); a term in `Gf` must be in `L`. The `uffix` bug violated the second clause (a quadratic
`coarse*` in `uf`'s gradient that `L` never inverted); the bug of this note violates the first
(a term in `uf`'s `F` that `rhs`'s `F` withholds on one side). The design makes the first clause
hold *by construction*: one emitter produces the face-value delta, the row CSR and the slot CSR
are the same entries times the divergence coefficient.

**Conventions that must hold.**
- `cut(i)` = fluid cell with at least one solid face-neighbour centre (`cut_cell.hpp:124`);
  `cut_` is sized `n`, `fluid_` is sized `n + nghost` (ghost tail SDF-sampled in the global
  frame, bit-identical to the owner's sample). Any rule that asks `cut(j)` for a ghost `j` needs
  the flag communicated first (§6.4).
- Host oracle (`flow_oracle.hpp`) and device (`flow.hpp`) build the SAME host CSRs through the
  same builders; parity by construction stays.
- np = 1 bitwise vs single-rank; np = 2/4/8 in the seam test's ~3e-7 class; the byte gate's ten
  scenarios move only where the note says (§10).
- **Bit-identity where the change is inert by geometry.** On any mesh with no cut-adjacent C/F
  sub-face (every uniform or finest-band mesh: `band ≥ 1`), every CSR must be bit-identical to
  today's. The emitter is specified (§6.2) so that this holds without a rounding-level move.
- The pressure matrix stays standard (decision "C/F scheme pressure matrix/MG/PCG stays standard
  order — placement is (1,2)"), see §5 option D for why this note does not supersede it.
- The ABC approximate projection, never Rhie–Chow.
- No environment variable changes a result (QUALITY_PLAN D3).

**Performance.** Everything here is setup-time host work in `setSolid`: three CSR builders whose
gates only shrink their output, one extra ghost exchange of one scalar per `setSolid`
(`syncScalar` on an `nExt`-sized View — microseconds against a build that samples the SDF over
every leaf). Zero per-step cost change; no device kernel changes; CSR sizes shrink slightly on
mixed-level meshes.

## 5. Options considered

**A. Per-face predicate `regular(i) && regular(j)`, applied to D, the G substitution and uf
through one emitter — CHOSEN.** Restores (I) on every mesh; keeps cut rows exactly as today
(the recorded stability fix is retained by the same predicate: a cut row has no passing face);
the only rows that change are regular rows with a cut neighbour across a 2:1 face, and on those
faces the scheme reverts to the standard value. Cost: §7. Stability: §8. Structure: the
literature's (b) — availability is a face property with a lower-order fallback.

**B. `rowFluid` on both builders (apply the delta at cut rows too, so the books agree at the
quadratic value) — REJECTED, measured.** This is the 2026-08-27 configuration: 2 of 12
throat-graded meshes march to k ~ 1e12 by step ~100, the cfDiv delta alone carries it. Mechanism
(register entry "cfDiv/cfGrad row gate must be rowRegular"): the ghost closure owns the cut row's
constraint and gradient; adding a smooth-field C/F substitution to its constraint gives it velocity
reads its gradient never sees — an invisible-subspace component the projection cannot remove
(`flow/doc/collocated_invisible_subspace.md`). Not a parameter problem; a support-consistency
violation. Re-entering it is not an option.

**C. Fold the C/F substitution into the ghost closure at cut rows (constraint AND gradient), so
cut-adjacent faces keep the quadratic value consistently — DEFERRED, with a default.** This is the
principled completion of B: the closure's sample rows would carry the `coarse*` stencil for their
C/F faces and the closure's gradient the matching one. It is the unfinished "sub-face closures"
rung of `docs/amr_mixed_level_cut_band_plan.md`. Cost: a change inside `ghost_projection.hpp` /
`ghost_projection_sampled.hpp` (two builders, host + device + distributed probe set), a new
invisible-subspace analysis, and a re-run of the whole throat ladder. Benefit: the order of the
face value on a codimension-2 set (§7). Default: do not build it now; revisit only if the census
of §6.5 on production mixed-level meshes and the ladder show a measurable loss (open question 5,
§11).

**D. Move the correction into the operator, `L = D_std·G_quad` — REJECTED, and it does not
address this bug.** This bug lives in `F` (the u-part of `uf`), not in `Gf`; a quadratic `Gf`
would leave the row/face disagreement in `F` untouched. On its own merits: the quadratic C/F
stencil is non-symmetric (Martin–Cartwright's operator is), so the pressure matrix would stop being
SPD — PCG is no longer valid (BiCGStab/GMRES: ~2× the work per iteration, no monotone convergence
guarantee), and the MG hierarchy (host, device, distributed) would need a quad-aware coarse
operator at every level. The gain is the order of `uf`'s face gradient on C/F faces during
transients only: at the fixed point φ → 0, and during a transient the error of `G_std φ` on a C/F
face is `O(h)·|∇φ| = O(h·dt·∂_t p)` — a term of the same order as the incremental projection's own
splitting error, confined to C/F faces. Formally nothing is gained. **The register entry stands;
this note adds the transient half of its justification and the rule (I) that `uffix` established,
as a new clarifying entry (WO5), not a supersession.**

**E. Forbid the configuration (throw when a cut cell touches a level boundary) — REJECTED.** A
level jump that reaches the wall puts cut cells on both sides of a 2:1 face; that is the defining
configuration of the mixed-level cut band (`refine_to_sdf_graded`, `refine_to_gap_floor`,
`set_ghost_sampled`), a shipped, distributed feature whose measured payoff on throat geometries is
4–8 cells across the throat at 0.1–1.3 % policy error for 18–30× fewer cells (M2a table). The
throw would fire on `tests/test_amr_distributed_seam_mpi` (a two-level latitude map on the cut
band) and on the byte gate's `amr.flow_sampled` scenario, i.e. it would retire the feature. The
reference codes (§3c) do not forbid it either. What IS right is a *census*, so the degradation is
measured rather than silent: §6.5. A throw is kept only as a programming-error guard (§6.4: a
builder that would read a ghost flag that was never filled).

**F. Derive `cfDiv` from the `cfUf` slot CSR by summation (`Δ_cfDiv := D_std(Δ_cfUf)`) — chosen in
spirit, not in mechanics.** Summing slot rows into cell rows multiplies `(wC·c)` by `scale` where
today's row builder computes `(scale·wC)·c` — a rounding-level difference that would move every
graded byte-gate hash for no numerical reason. The shared emitter of §6.2 gives the same
single-source-of-truth with today's association order on both sides, so inert meshes stay
bit-identical. An equality test (`‖D_std(Δ_cfUf) − Δ_cfDiv‖ ≤ 1e-15·‖Δ_cfDiv‖`) pins the property.

**G. Symmetric gate on `buildCfUfDelta` alone (brief §6.1) — REJECTED, and the brief's reading of
it is wrong.** Gating only the face builder would leave `Δ_cfDiv` at the regular row (its row
gate passes) while `uf` loses the delta on that face: the mismatch moves from the cut row to the
regular row, it is not removed. Only gating both books with the same predicate removes it. The
measured "exactly zero change on band=3" is consistent with either reading (no such faces there).

## 6. The design

### 6.1 The predicate

One per-cell flag, valid on `[0, nExt)`:

    regular(s) := fluid(s) && !cut(s)          s < n     : from mom_ (as today's rowRegular)
                                              s ≥ n     : the OWNER's value, exchanged (§6.4)

One per-face predicate, used by every projection-family builder:

    cfFace(i, j) := levelOf(j) != levelOf(i) && regular(i) && regular(j)

`regular ⇒ fluid`, so `fluidOk(i) && fluidOk(j)` (today's uf gate) is implied. The tangential
SAMPLES inside `cfAppendStencil` keep their `fluidOk` gate unchanged: a cut cell's velocity is a
solved fluid value and reading it as a tangential sample is what happens today on every stable
mesh; the instability mechanism (§5 B) is about which ROW owns a constraint, not which cells it
reads. Changing the sample gate would move inert meshes for no reason.

### 6.2 One emitter for the face value

`detail::cfAppendFaceValueDelta(ap, out, coarse, fine, axis, dist, scale, fluidOk, scheme, proto)`
appends, for one directed 2:1 sub-face, exactly the entries `buildCfDivDelta` and `buildCfUfDelta`
each push today, in today's order and with today's association of the scale:

    H  = ap.cellWidth(coarse, axis),  h = ap.cellWidth(fine, axis)        (per-axis widths)
    d  = dist                                                            (from forEachFaceFull)
    wF = (0.5·H)/d,  wC = (0.5·h)/d
    push(fine,   scale·(wF − 0.5))
    push(coarse, scale·(wC − 0.5))
    cfAppendStencil(ap, out, coarse, fine, axis, scale·wC, fluidOk, scheme, proto)

- `buildCfDivDelta` calls it with `scale = invV·α·A·dir` (its row coefficient); `buildCfUfDelta`
  with `scale = 1`. Then `Δ_cfDiv[i] = Σ_{faces of i} (invV·α·A·dir) · Δuf[slot]` entry by entry —
  (I)'s first clause by construction.
- **Bit-identity on isotropic meshes.** Today's row builder uses `cellWidth(coarse)` (axis 0)
  and `d = 0.5·(H+h)`; the face builder uses `dist`. `cellWidth` is `h0·2^L` (exact power-of-two
  scaling, `poisson.hpp:255–261`), and `dist = 0.5·(2^Lc + 2^Lf)·h0[axis]`
  (`poisson.hpp:348`). With `2^Lf = 2^Lc/2`, both `H + h` and `(2^Lc+2^Lf)·h0` are the single
  correctly-rounded value of the real `3·2^Lf·h0`; `0.5·` is exact. So `d == dist` bitwise
  whenever `h0[axis] == h0[0]`, and `scale = 1` reproduces the face builder's `wF − 0.5`, `wC`
  exactly. Every entry the emitter produces is bit-identical to today's on isotropic meshes;
  WO1's acceptance is that the byte gate does not move at all.
- **Anisotropic finding (not in the brief).** Today the row builder takes `H, h` on axis 0 and
  `d = 0.5(H+h)`, the face builder takes `dist` on the face axis but `H, h` on axis 0, so on an
  anisotropic octree `wF + wC ≠ 1` in `uf` and the two CSRs disagree on EVERY C/F sub-face, cut
  or not. The emitter's per-axis widths fix that as a side effect. No anisotropic graded test
  exists; WO1 adds none (out of scope) but records the fact in the emitter's docstring, and the
  equality test of §6.6 would catch it if one is ever written.

### 6.3 The three builders

Signatures become `buildCf{Div,Grad,Uf}Delta(ap, t, regularOk, fluidOk, scheme)` — the row
predicate parameter is renamed and re-purposed; `buildCfLapDelta` keeps `(rowOk, fluidOk)` and
is not touched (the momentum deferred-correction term is not part of the projection pair and was
exonerated by the 2026-08-27 bisection; open question 1, §11).

- `buildCfDivDelta`: `if (!regularOk(i)) return;` stays as the early-out; the face callback tests
  `cfFace(i, j)` in place of `Lj != Li`; the body is the emitter call.
- `buildCfUfDelta`: the face callback tests `cfFace(i, j)` in place of
  `Lj != Li && fluidOk(i) && fluidOk(j)`; the vel body is the emitter call with `scale = 1`. The
  φ part: see §6.7.
- `buildCfGradDelta`: `if (!regularOk(i)) return;` stays; in the first sweep `cf[axis][s] = true`
  only when `cfFace(i, j)`; in the second sweep the `coarse*` substitution (b) fires only when
  `cfFace(i, j)`. The side reweighting (a) is a row property and is unchanged in form; its trigger
  is now "a PASSING C/F face on this axis". Rationale: D and G must be paired face by face
  (§8, point 3) — a face whose value D takes as standard must be one whose gradient G takes as
  standard.
- The slot numbering of `buildCfUfDelta` (cell-major, one row per `forEachFaceFull` slot) is
  unchanged; withheld faces are empty rows, as they are today for same-level faces.

### 6.4 The ghost-visible regular flag (MPI)

In `AmrFlow::setSolid`, after `mom_.build` and before the C/F builders (i.e. at `flow.hpp:911`,
where `dhex_` already exists — `prepareDistributed` at 719 initialises it at 1853):

    std::vector<char> regular(nExt_, 0);
    for s < n:  regular[s] = mom_.isFluid(s) && !mom_.isCut(s)
    if (dist_):
        View<double> x("cf_regular_x", nExt_); host-fill x[s] = regular[s] for s < n, 0 for the tail;
        syncScalar(x);                            // the owner's value lands in the ghost tail
        regular[s] = (x[s] > 0.5) for s ≥ n
    regularOk = [&](Index s) { return regular[s] != 0; }

Why exchange and not recompute from the SDF: `cut(g)` for a ghost needs the SDF sign at g's six
face-neighbour centres, and those neighbours are not guaranteed resolvable — the discovery
fixpoint registers the coarse cell's tangential reach (`probeCfTangential`) and the face sweep's
±2 upstream reach, not the six neighbours of a fine ghost. Recomputing would require new probes in
`probeCfScheme` and a second ghost pass in `cut_cell.hpp`; the exchange is one call, needs no
discovery change (the new predicate only ever short-circuits probes the gate-free discovery already
issued, so the registry is a superset of what it can ask for), and is exact by definition — it IS
the owner's flag, and the owner's flag is decomposition-independent because its neighbours' SDF
samples are.

Guard: the builders assert `s < regular.size()` for every `s` they test (debug-only or a cheap
`if` that throws `std::logic_error` — this is a programming-error guard, not a configuration guard).

The oracle is single-rank: `regular` is the local vector, no exchange.

### 6.5 The census

`Index AmrFlow::numCfCutFaces() const` — the number of owned `forEachFaceFull` slots `(i, j)` with
`levelOf(j) != levelOf(i)`, `fluid(i) && fluid(j)`, and `!(regular(i) && regular(j))`: the C/F
sub-faces on which the quadratic face value is withheld. Computed in the same sweep that builds
`cfUf` (it enumerates exactly those slots); `0` on every uniform or finest-band mesh; local count
under MPI (like `numCfGhostColumns`), and by ownership `Σ_ranks numCfCutFaces` is
decomposition-invariant — an exact integer gate (§10). Python: `Flow.diagnostics.num_cf_cut_faces`
(developer tier, next to `divergence_norm_face`; the spelling must be checked against
`../docs/NAMING.md` before it ships — open question 4). No stderr notice (D3: no noise; the number
is a diagnostic, not a warning).

### 6.6 The equality test

A test-side helper (oracle, host): from `fl.cfUf_.vel` and the oracle's face geometry, form
`D_std(Δuf)` per cell as `invV·Σ α·A·dir·Δuf[slot]` applied to the current `u`, and compare with
`cfApplyCompHost(fl.cfDiv_, u)`. Bound: `‖D_std(Δuf) − Δ_cfDiv‖₂ ≤ 1e-14·max(1, ‖Δ_cfDiv‖₂)`. Today
this reads 8.7e-17 on `band=3` and 2.018e-01 on `band=0`; after WO3 both are at round-off. This
pins (I)'s first clause independently of any solve.

### 6.7 The φ part of `cfUf`

`uffix` stopped applying `cfUfPhi_` and left it built. Under (I) it must never be applied while the
matrix is standard, so a built-but-forbidden CSR is a trap for the next session. Default (open
question 3): stop building it — `CfUfDelta` becomes the vel CSR alone, `cfUfPhi_` / `cfUf_.phi`
and their uploads are deleted, the `finishProjection` comment is shortened to the rule (I) and a
pointer to this note. The `ufgate` study that reads it computes `D(Δφ)` itself if it is ever
rebased.

## 7. What it costs at cut cells

On a cut-adjacent 2:1 sub-face the face value is the standard `½(u_F + u_C)`, whose sample point
is offset `(H−h)/4 = h/4` from the face along the normal and by the fine cell's tangential offset:
an **O(h) pointwise error in the face velocity** on those faces, against the quadratic scheme's
O(h²). This is exactly the local order the whole mesh ran at under `cf=0` (2.00 normal / 0.41
tangential on the ladder, register entry of 2026-09-21), now confined to the set of faces where a
level boundary meets the wall.

Where that set lives decides the global cost:

- **Production mixed-level meshes** (`refine_to_sdf_graded`, `refine_to_gap_floor`): a level
  boundary is a surface, the cut band a shell; their intersection is a **curve** (codimension 2).
  The affected cells are an O(h²) volume fraction, so an O(h) local error contributes O(h²) to the
  L2 error — it does not lower the global order. L∞ near those cells is O(h), and the cut cell's
  own closure there is the ghost/sampled closure, unchanged.
- **The `band=0` mesh** of the measurement (the entire cut band ON the level boundary,
  codimension 1) is the worst case: an O(h) volume fraction, L2 contribution O(h^{3/2}). It is a
  diagnostic mesh, not a production one, and it is exactly the mesh on which today's rule leaves a
  non-decaying O(1) mass error.
- Nothing changes on any finest-band mesh: the ladder's 2.00 / 1.60 orders and the Z&H graded
  permeability (1.22e-02 vs uniform-fine) are bit-identical after this change, because those meshes
  keep cut cells uniformly finest (the ladder's docstring says so; §10 asserts it).

The census (§6.5) reports the size of the affected set for any mesh, so the cost is measured, never
silent.

## 8. Why it cannot destabilise the throat meshes

Argued from the mechanism of the recorded instability, not from a test pass.

1. **Cut rows are unchanged.** The mechanism is "the ghost closure owns the cut row's constraint
   and gradient; a C/F substitution added to that constraint reads velocities the row's gradient
   never sees". Under `cfFace`, a cut row has no passing face, so its `cfDiv`, `cfGrad` and its
   side of every `uf` face are exactly what the row gate gives today. The protection the row gate
   was added for is retained in full, by the same predicate.
2. **Regular rows only lose terms, and what replaces them is the `cf=0` scheme.** The only rows
   that change are regular rows with a cut neighbour across a 2:1 face, and the change is to
   withhold the delta on that face in D, in the G substitution and in `uf` — the face reverts to
   the standard value. The standard scheme on EVERY face was the shipped default until
   2026-09-21 and is the `cf=0` arm of `amr_two_sphere_diverge_probe.py`, which the probe's own
   design records as stable on all 12 throat meshes ("Stable at cf=0 ⇒ that term"). A scheme that
   is a face-wise mixture of two schemes each stable on its own is not automatically stable, so
   point 3 is the load-bearing one.
3. **The D–G pairing is preserved face by face.** The instability needs a constraint component
   the gradient cannot see. Under the new rule every C/F substitution in D at row `i` on face
   `(i, j)` has its counterpart in G at row `i` on the same face (both fire on `cfFace(i, j)`),
   and the face value in `uf` on `(i, j)` is the value D used (same emitter). No term is added
   anywhere; terms are removed in matched pairs. The set of constraint reads at every row is a
   subset of today's, and the set of gradient reads is the matching subset. There is no new
   invisible component to accumulate.
4. **The lagged momentum term is untouched.** `cfMom_` (the deferred-correction diffusion delta)
   keeps its gate; it was exonerated by the bisection and nothing here changes what it reads.

The throat ladder is still run (§10, gate S) because the argument above is about the mechanism
that was identified, and a test is what shows no other one was hiding behind it.

## 9. Work orders

Commit-sized, in dependency order. Every one is a separate commit on a topic branch off `main`
after `uffix` lands; WO1–WO2 are inert and must be proved so before WO3 changes a number. Run the
battery as `CLAUDE.md` says (`OMP_NUM_THREADS=2 OMP_PROC_BIND=false`, np8 last). Format every C++
file touched with clang-format 18.1.8.

**WO1 — the shared emitter (inert refactor).** In `cf_scheme.hpp` add
`detail::cfAppendFaceValueDelta` per §6.2; make `buildCfDivDelta` and `buildCfUfDelta` call it
(the div builder passes `dist` from `forEachFaceFull` instead of computing `0.5(H+h)`, and per-axis
widths; the face builder passes `scale = 1`). No predicate change yet. Docstring records the
anisotropic finding. *Acceptance:* the full battery green; `python/state_hash.py --check` passes
with ALL ten hashes unchanged (this is the bit-identity claim of §6.2 — if any hash moves, stop:
the association order was not reproduced).

**WO2 — the ghost-visible regular flag (inert plumbing).** Per §6.4 in `flow.hpp::setSolid`:
build `regular` on `[0, nExt)`, exchange the tail, keep `regularOk`. Nothing reads the ghost tail
yet (the builders still take today's predicates). Add `numCfCutFaces()` (§6.5) computed from the
new flag, and its `diagnostics` binding (name per open question 4; regenerate `packaging/_amr.pyi`).
*Acceptance:* battery green, all ten hashes unchanged; in `tests/test_amr_distributed_cf_mpi` add
the assertion `Σ_ranks numCfCutFaces() == 0` (its mesh is a finest band — the count must be zero on
every rank) and in `tests/test_amr_distributed_seam_mpi` the assertion
`Σ_ranks numCfCutFaces()(np) == numCfCutFaces()(SELF)` exactly, and `> 0` (its two-level latitude
map on the cut band is the configuration; if the count is 0 there, the mesh is not what its
docstring says — stop and report).

**WO3 — the per-face predicate (the numerics change).** Per §6.1 and §6.3: change the three
builders' signatures and gates in `cf_scheme.hpp`; thread `regularOk` from `flow.hpp:911–925` and
`flow_oracle.hpp:219–224`. Add the guard of §6.4. Tests, all in the same commit:
- `tests/test_amr_face_field.cpp`: case (4), the cut band AT the level boundary —
  `BO t(IVec<3>{8,8,8}, 2)` (N = 32, lmax = 2, as the study's `run()`), `refineToSdf(t, geo,
  sphere, 0, /*band=*/0.0, true)`, sphere `R = (0.125·3/(4π))^{1/3}·32` at the centre; aperture
  path, `cf = 1`, 25 steps at `presIters = 30`; assert `numCfCutFaces() > 0`,
  `dFace < 0.01·dCell` (the existing graded bar), and the §6.6 equality bound. Also assert the
  equality bound on the existing `band=3` case (it is 8.7e-17 today and must stay).
- The device path on the same mesh: extend the existing device-vs-oracle parity case
  (`tests/test_amr_flow_solver.cpp` or `tests/test_amr_flow.cpp`, whichever compares `AmrFlow` to
  `oracle::AmrFlow` cell by cell) with the `band=0` mesh at its existing tolerance.
- Ghost path on the `band=0` mesh (oracle): compute host-side `‖D_std(uf)‖₂` restricted to
  REGULAR fluid rows (exclude cut rows, whose `D_std(uf) ≠ 0` is the documented ghost-scheme
  property) and assert it is `< 0.01 ×` the same norm of the cell field. Before WO3 the regular
  rows adjacent to cut cells carry the 2e-1-class mismatch; after, they are at residual level.
*Acceptance:* battery green EXCEPT `python_state_hash` (expected to fail on exactly one scenario,
`amr.flow_sampled` — see WO4); np=1 bitwise and np=2/4/8 in class on the seam test; the census
gates of WO2 still hold (the counts do not change — WO3 changes what the flag gates, not the
flag).

**WO4 — byte gate re-record.** `python/state_hash.py --save`, own commit, message naming the moved
scenario(s). Expected: `amr.flow_sampled` moves (its mesh is `refine_to_sdf_graded(..., band=2.0)`
with cut cells at two levels); `amr.flow_ghost` and `amr.distributed_flow.np{1,2}` are
`refine_to_sdf(band=3.0)` finest bands and must NOT move; the six non-flow scenarios must not move.
*Acceptance:* exactly that set. Any other movement is a finding to be understood before recording
(most likely a mesh that is not the finest band its scenario claims — report the census).

**WO5 — dead φ part, docs, register.** Delete the φ CSR per §6.7 (oracle + device + upload +
the `numCfGhostColumns` count of it); shorten the `finishProjection` comment to rule (I) and a
pointer here. Update `cf_scheme.hpp`'s header comment ("Robustness gating" paragraph: the row gate
is gone, the face gate is the rule) and `docs/amr_collocated_projection.md` (one paragraph: (I),
the face gate, the census). Register (`../docs/decisions/amr.md`, and the index): (i) a NEW entry
"C/F face-value delta is gated per FACE (`regular(i) && regular(j)`), never per row; one emitter
feeds D, the G substitution and uf" with the rejected alternatives B, D, E, G of §5 and this note
as provenance; (ii) mark "cfDiv/cfGrad row gate must be rowRegular, not rowFluid" as **superseded
by** (i) — its mechanism stands, its fix is replaced; (iii) a clarifying entry under "pressure
matrix stays standard": the transient half of the argument and rule (I). `ROADMAP.md`: the live
item closes; option C is listed as the deferred completion with its default.
*Acceptance:* battery green, hashes unchanged from WO4 (deleting an unapplied CSR is inert).

**WO6 — the study gates (measurement, no code).** Run and record in `docs/data/` (or the ladder's
own log): gate S (throat stability), gate O (orders), gate P (permeability), gate M (policy
error), §10. Report the before/after tables in the commit message.

## 10. Verification gates

| gate | what | configuration | expected after | today |
|---|---|---|---|---|
| **I** identity | `‖D_std(Δuf) − Δ_cfDiv‖₂` (§6.6) | `band=0` N=32 lmax=2, aperture, cf=1 | ≤ 1e-14 | 2.018e-01 |
| **I'** identity, inert mesh | same | `band=3` | ≤ 1e-14 (8.7e-17 today, unchanged) | 8.7e-17 |
| **F** face divergence | `divNormFace` vs `divNormL2` | `band=0`, aperture, cf=1, 25 steps, presIters=30 | `dFace < 0.01·dCell`; absolute at the solve-residual level (report it) | 2.016e-01 (O(1) of the cell field) |
| **F'** inert mesh | `divNormFace` | `band=3`, cf=1 | bitwise identical to today (~1.3e-12 at 300 iters) | 1.34e-12 |
| **B** bit-identity of inert refactors | byte gate | WO1, WO2, WO5 | all ten hashes identical | — |
| **B'** byte gate after WO3 | byte gate | WO4 | `amr.flow_sampled` moves; the other nine identical | — |
| **C** census invariance | `Σ_ranks numCfCutFaces()` | seam test np=1,2,4,8 vs SELF | exact integer equality, and > 0 | (new) |
| **C'** census zero | same | `distributed_cf` test, `band=3` face-field cases | == 0 | (new) |
| **D** distributed class | seam test WORLD vs SELF | np=1 / np=2,4,8 | bitwise / ≤ its ~3e-7 class bars | passes today with the mismatch present on every rank |
| **S** stability | `amr_two_sphere_gap.py` + `amr_two_sphere_diverge_probe.py` | the 12 throat meshes (gaps 8,16 × n 1,2,3,4,6,8), cf=1, 400 steps | all 12 finite to the horizon; g=8 n=3,4 do not diverge | all 12 stable under the row gate |
| **M** policy error | `amr_two_sphere_gap.py` k vs uniform finest band | same meshes | every mesh's policy error within ±0.3 % absolute of the value measured at the base commit in the same run, and ≤ 1.5 % | 0.1–1.3 % (M2a) |
| **O** orders | `graded_poiseuille_ladder.py` | self-similar family, cf=1 | 2.00 normal / 1.60 tangential, bitwise identical (finest-band mesh) | 2.00 / 1.60 |
| **P** permeability | Z&H graded vs uniform fine | cf=1 | 1.22e-02, bitwise identical (finest-band mesh) | 1.22e-02 |
| **T** battery | ctest | host-openmp, np8 last | all green; count unchanged + the new cases | — |

Gates O and P are expected NOT to move at all; if either moves, a finest-band mesh contains a
cut-adjacent C/F face (the census will say so) — stop and report before proceeding, because that
means the accuracy statement of §7 applies to a mesh it was not expected to apply to.

## 11. Risks and open questions

Each carries a default so work proceeds unattended.

1. **Should `buildCfLapDelta` (momentum) take the same face gate?** *Needs the user's preference*
   (uniformity vs. minimal change). Default: **no** — it is not part of the projection pair, it was
   exonerated by the bisection, and its quadratic value at a cut-adjacent face is a legitimate
   fluid read. Reversible by one predicate swap.
2. **Should the aperture path keep `rowFluid` (the C2 mechanism needs the ghost closure, which the
   aperture path does not have)?** *Needs a fact* (a stability run of the 12 throat meshes with
   `setGhostProjection(false)` and `rowFluid`) and then a preference. Default: **one rule for
   both paths** — the aperture path is the fallback scheme, and two rules is a second thing to
   keep in lockstep.
3. **Delete the φ part of `cfUf` or keep it built?** *User's preference.* Default: **delete**
   (§6.7) — a CSR that must never be applied is a trap; the study that reads it is on an unmerged
   branch.
4. **The Python name of the census.** *User's preference / NAMING.md.* Default:
   `Flow.diagnostics.num_cf_cut_faces` (developer tier, `num_*` count convention). C++
   `numCfCutFaces()` regardless.
5. **Build option C (sub-face closures) later?** *Needs a fact:* the census on the production
   mixed-level meshes of the M2a study and gate M's numbers — if the policy error moves by less
   than the ghost scheme's own 0.2–0.3 % bias, the O(h)-on-a-curve loss is below the noise and C
   is not worth its cost. Default: **defer**; record the census numbers with gate M so the decision
   can be made from data.
6. **Two-cut faces.** A 2:1 face with cut cells on BOTH sides today has no delta in D (both rows
   gated) and a delta in `uf` (both fluid) — the same mismatch, on both rows. `cfFace` withholds it
   everywhere; no separate handling. Stated so no one adds one.
7. **The anisotropic inconsistency (§6.2)** is fixed as a side effect but has no test. *Needs a
   fact* whether anyone runs graded anisotropic meshes. Default: no new test in this package; the
   §6.6 equality test is the trip-wire if one is added.

## 12. What the brief got wrong or understated

- §6.1: gating `buildCfUfDelta` alone would not "remove the correction entirely at those faces";
  it would move the mismatch from the cut row to the regular row (§5 G). The conclusion — do not
  do it — is right for a different reason.
- §3 "Reachability today": the configuration is not reachable only by forcing `band=0`. The
  distributed seam test (`tests/test_amr_distributed_seam_mpi.cpp`, a two-level latitude map ON
  the cut band) and the byte gate's `amr.flow_sampled` scenario (`refine_to_sdf_graded`,
  `band=2.0`) contain it today and pass with the mismatch present — the seam test because it
  compares WORLD to SELF (both wrong identically), the byte gate because it hashes whatever the
  code produces. Gate C makes the seam test say so in numbers.
- The two builders are also inconsistent on anisotropic octrees, independently of cut cells
  (§6.2). Inert on every current test (all isotropic).
- The register entry for the row gate is right about the mechanism and wrong about the fix's
  scope: a row gate cannot be conservative. WO5 records the supersession explicitly.
