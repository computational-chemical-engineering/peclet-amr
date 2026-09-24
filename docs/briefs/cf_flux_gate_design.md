# Design brief: the C/F flux correction cannot be gated per ROW and per FACE at once

> For an architect design pass. Written 2026-09-22 against `peclet-amr` `4b96cb3` + branch `uffix`.

## 1. The question

**At a 2:1 coarse/fine interface where one of the two cells is a CUT cell, should the C/F flux
correction be applied — and how must the cell-divergence bookkeeping and the face flux be made to
agree, without reintroducing the instability the current row gate was added to prevent?**

A decision-shaped answer names the gating rule (or the reformulation that dissolves the question),
says what it costs in accuracy at cut cells, and says why it cannot destabilise the throat meshes
that motivated today's rule.

## 2. Why this needs an architect pass

Three things make it more than an implementation choice.

1. **It is not satisfiable as posed.** A face flux is ONE number shared by two cells — that shared
   value is what makes the scheme conservative. The current rule for "does the correction apply"
   is a property of a CELL (`rowRegular` = fluid and not cut). When two cells sharing a face
   disagree, no single flux value satisfies both sets of books. Any fix changes the *shape* of the
   rule, not its parameters.
2. **It is stability-sensitive, with history.** The row gate exists because the looser gate marched
   2 of 12 throat-graded meshes to k ~ 1e12 by step ~100 (register entry quoted in §6). A fix that
   applies the correction in *more* places risks re-entering that failure; a fix that applies it in
   *fewer* trades accuracy at exactly the cells the immersed boundary is about.
3. **There is a literature answer and we should not re-derive it badly.** Conservative AMR flux
   matching at refinement boundaries is classical — Berger & Colella refluxing; Martin & Colella's
   cell-centred AMR projection method; Martin, Colella & Graves (2008); and for the cut-cell half,
   the Chombo/EB line (Colella, Graves et al.) which faces precisely this "irregular cell meets
   coarse-fine boundary" case. The scheme in this repo is named for Martin & Cartwright. **Please
   check what that literature does at an irregular cell on a coarse/fine boundary before
   designing** — the canonical move (make the coarse flux equal the sum of the fine fluxes *by
   construction*, and compute every divergence from that single flux) may dissolve the question
   rather than answer it.

## 3. Current state, with the code

Two builders produce the same C/F correction for two different consumers. Both live in
`include/peclet/amr/cf_scheme.hpp`.

**(a) The cell-divergence delta — gated per ROW, skips cut cells:**

```cpp
inline CfCompCsr buildCfDivDelta(const AmrPoisson<3, Bits>& ap, const BlockOctree<3, Bits>& t,
                                 RowFn&& rowOk, FluidFn&& fluidOk, CfScheme scheme) {
  ...
    hostParFor(n, [&](Index i) {
      if (!rowOk(i))            // <-- rowRegular: fluid AND NOT CUT. cut rows get NOTHING.
        return;
      ...
      ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double area, double, double alpha) {
        if (Lj == Li) return;
        const double scale = invV * alpha * area * static_cast<double>(dir);
        ... row.push_back(eF); row.push_back(eC);
        detail::cfAppendStencil(ap, row, coarse, fine, axis, scale * wC, fluidOk, scheme, proto);
```

Its own docstring states the conservation intent: *"Both incident cells use the identical value
(conservative telescoping)."* That holds for the VALUE; the row gate then denies one incident cell
its half.

**(b) The face-flux delta — gated per FACE, both centres fluid:**

```cpp
inline CfUfDelta buildCfUfDelta(const AmrPoisson<3, Bits>& ap, const BlockOctree<3, Bits>& t,
                                FluidFn&& fluidOk, CfScheme scheme) {
  ...
    ap.forEachFaceFull(i, [&](Index j, int axis, int dir, double, double dist, double) {
      if (Lj != Li && fluidOk(i) && fluidOk(j)) {     // <-- no cut test at all
```

Docstring: *"Emitted only for faces whose BOTH centers are fluid (the advection gate)"*, and
*"as in buildCfDivDelta, so the advecting flux matches the divergence constraint"* — the two are
DOCUMENTED to match and do not, at a cut cell on a 2:1 interface.

The predicates come from `AmrFlow::setSolid` (`include/peclet/amr/flow.hpp:911`):

```cpp
auto fluidOk    = [&](Index i) { return mom_.isFluid(i); };
auto rowRegular = [&](Index i) { return mom_.isFluid(i) && !mom_.isCut(i); };
```

**Measured consequence.** On a mesh where the cut band is made to meet the level boundary
(`refine_to_sdf(..., band=0)`, 336 of 2768 C/F rows cut), the mismatch dominates everything else:

| quantity | value |
|---|---|
| `‖Δvel − Δ_cfDiv‖` (the disagreement) | **2.018e-01** |
| `‖div(uf)‖` total | 2.016e-01 |
| the same on a normal mesh (`band=3`, 0 of 4024 C/F rows cut) | ~1e-12 |

It does **not** decay at steady state: it is proportional to the velocity, not to φ. (Contrast the
defect fixed on branch `uffix`, which was proportional to φ and therefore transient-only.)

**Physically:** the flux crossing that face carries a correction one neighbour accounted for and
the other did not; the pressure solve was never told, so it never cancelled it. That cell has a
permanent spurious source/sink of mass at the wall.

**Reachability today.** It needs a cut cell to touch a level boundary. Normal practice refines a
band AROUND the solid, so cut cells sit inside the fine region and never meet a level jump — the
measurement above had to force `band=0`. But `refine_to_sdf_graded` / `refine_to_gap_floor` exist
precisely to put cut cells at several levels, and the mixed-level cut band is a shipped feature.

## 4. Constraints and invariants

- **Conservation is the point.** `uf` is the advecting flux; `Σ_faces α·A·dir·uf = 0` per cell is
  what makes scalar advection conservative. Just restored in general on `uffix`; this is the
  remaining hole.
- **np = 1 bitwise, np = 2/4/8 in the established class** (`tests/test_amr_distributed_seam_mpi`,
  ~3e-7 march class). The byte gate `python/state_hash.py` has ten scenarios.
- **MPI: `isCut` is NOT ghost-visible.** `cut_` is sized `n`, not `n + nghost`; `fluid_` is the
  ghost-extended one. Any rule that asks "is my NEIGHBOUR cut" needs that flag communicated first.
  (An experiment that did so without noticing read past the end — inert single-rank, wrong
  multi-rank.)
- The pressure MATRIX stays standard — see §5.
- Header-only C++20 + Kokkos; the host oracle (`flow_oracle.hpp`) and the device path must stay in
  lockstep (parity ctests compare them).

## 5. Already decided — do not reopen

From `../../docs/decisions/amr.md`, verbatim:

- **"cf=1 (quadratic C/F flux) is not optional on graded meshes"** — *"the standard flux cannot
  converge on graded meshes"*. cf=0 is rejected there, and cf=1 is now the shipped DEFAULT (user
  decision, 2026-09-21).
- **"C/F scheme pressure matrix/MG/PCG stays standard order — placement is (1,2)"** — *"Pressure
  matrix/MG/PCG/ghost-BiCGStab stay standard (φ→0 at the fixed point ⇒ matrix C/F order can't move
  steady — (1,2) placement)"*. **Note:** that reasoning is about the STEADY solution and is sound
  for it; its consequence for the face field during TRANSIENTS was not traced, and is what branch
  `uffix` just fixed (§6). If your design wants `L = D·G_quad`, say so explicitly as a new decision
  superseding this one, with the argument.
- The ABC approximate projection, never Rhie–Chow.

**Genuinely open — this is what to decide:** the gating rule at a cut cell on a 2:1 interface, and
whether the delta-on-a-standard-operator formulation should survive at all there.

## 6. Already tried, with evidence — do not re-walk these

1. **Symmetric per-face gate `rowRegular(i) && rowRegular(j)` on `buildCfUfDelta` alone.** Measured:
   changed the result by **exactly zero** on a `band=3` mesh (6.6751e-03 → 6.6751e-03, `.so`
   verified newer than the header). Reason: 0 of 4024 C/F rows are cut there, so the gate is inert.
   It also would have read past the end of `cut_` under MPI (§4). It was **not** tested on a
   `band=0` mesh, where the analysis says it cancels on *both* sides rather than one — i.e. it
   removes the correction entirely at those faces instead of making the two agree.
2. **Adding `Δ_cfDiv` to the face-divergence diagnostic** (to "measure the constraint that was
   solved"). Measured: made it ~72× worse (6.675e-03 → 4.766e-01), because `uf` already carries
   that correction exactly — `‖D(Δ_cfUf,vel) − Δ_cfDiv‖ = 8.7e-17`. The cell field has no C/F
   correction in it and the face field does; that asymmetry is real and is why `divNormL2` adds one
   and `divNormFace` must not.
3. **The φ half of the same family — FIXED, on branch `uffix`**, and it is the model for what a
   good answer looks like. `uf` carried a quadratic `coarse*` in its FACE GRADIENT while the matrix
   inverted `G_std`, so nothing balanced it. Dropping it: `div(uf)` 6.675e-03 → 1.34e-12; worst
   transient 0.240 → 3.8e-11; with advection 4.0e-02 → 1.2e-05; **steady answer unchanged**
   (permeability 47.182462 before and after). 83/83 ctests, 17/17 np8, four graded byte-gate hashes
   re-recorded.

The decomposition that produced these (identity holds to 2.3e-15) is
`d = (rhs − Lφ) − Δ_cfDiv + D(Δvel) + D(Δφ)`, and the instrumentation to re-run it is
`tests/study_amr_uf_div.cpp` on branch `ufgate` (`./build_q/tests/study_amr_uf_div 25 1`).

## 7. How the answer will be verified

The implementation that follows must pass, and the design note should say which of these it expects
to move and why:

- `‖div(uf)‖` at solver tolerance on a `band=0` mesh (today 2.0e-01), and unchanged (~1e-12) on
  `band=3`.
- **Stability:** the throat-graded meshes that motivated the row gate. `tests/study/amr_two_sphere_gap.py`
  and `amr_two_sphere_diverge_probe.py` are the harnesses; the failure mode to exclude is k ~ 1e12
  by step ~100 on 2 of 12 meshes.
- **Accuracy:** the graded-mesh convergence ladder `tests/study/convergence/graded_poiseuille_ladder.py`
  (self-similar mesh family; today order 2.00 normal / 1.60 tangential at cf=1) must not regress,
  and the Z&H permeability against a uniform-fine reference (today 1.22e-02 at cf=1) must not
  regress.
- 83/83 + 17/17 np8, byte gate re-recorded with the moved scenarios named.
- np=1 bitwise vs single-rank; np=2/4 in the march class.

## 8. Deliverable

A design note at `docs/amr_cf_flux_gate.md` containing:

1. **The decision**, in one sentence.
2. **What the literature does** at an irregular cell on a coarse/fine boundary, with citations, and
   whether this repo should follow it or deviate (and why).
3. **The options considered**, each with the reason it was or was not chosen — including the
   reformulation option (single-valued flux by construction, divergence computed from it) and the
   "forbid the configuration in mesh generation" option.
4. **What it costs at cut cells** — the accuracy the chosen rule gives up, stated as an order.
5. **Why it cannot destabilise** the throat meshes, argued from the mechanism, not from a test pass.
6. **Work orders** for an Opus implementer: the functions to change, the predicates to thread, the
   MPI ghost-flag work, and the tests to add.
7. **The gates** from §7, with the expected numbers.

No production code from this pass.

## 9. Out of scope

- The `uffix` φ fix (done; assume it lands).
- Making `cf=1` the default (done).
- The distributed C/F discovery machinery (done).
- Anything in `peclet.flow`.
- The uniform-grid parity work.
