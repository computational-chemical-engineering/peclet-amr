# peclet-amr roadmap

One page, live items only. Everything here is open; everything closed lives in the campaign notes
and in `docs/archive/`. Rewritten in place, not appended. [README.md](README.md) in this directory
indexes every note.

**Where the code stands (2026-09-23).** The octree, the distributed octree with leaf/field
rebalance, solution-adaptive refinement, the collocated cut-cell Navier–Stokes solver (ghost
projection default, aperture fallback, mixed-level sampled cut band), the AMR multigrid / MG-PCG /
BiCGStab stack — now with a pressure multigrid that continues below the root brick (C1) — and the
parallel `setSolid` builders are all **built, distributed and gated**: 111 C++ ctests at np = 1, 2,
4, 8, 5 C++ studies and 8 Python, a SHA-256 byte gate over every public entry path, a
uniform-grid parity gate against `peclet.flow`, and a graded time-accurate gate against an exact
unsteady solution. The package is 0.x: the API may still move (the D9 exception to the suite's
clean-break 1.0.0).

**The graded solver is now validated against analytic solutions, steady and unsteady**: order 2.00
on the steady Poiseuille ladder (`amr_graded_convergence.md`) and **2.22 on a decaying Taylor–Green
vortex with advection on** (`amr_tg_graded.md`; 2.07 before the B5 seam reconstruction), with the
advecting face field conservative to 4e-12 throughout. What is NOT yet done is the layer above: no
published validation page (the gate on it is lifted — §E1) and no at-scale multi-GPU numbers. The
4–7× step cost against `flow` was the pressure hierarchy stopping at the root brick, and is gone
(§C1).

---

## A. Close the parity story (cheap, and it de-risks everything downstream)

`docs/amr_flow_uniform_parity.md` established that at `lmax = 0` this solver and `flow`'s
`SolverColocated` are the same discretization to solver tolerance, with two measured exceptions.
Finishing that is the cheapest accuracy work available, because it turns `flow` — a validated
solver with published anchors — into a reference oracle for every uniform-grid case.

- **A1 — the advecting velocity (parity note P1)** — **DONE 2026-09-21.** The user decided the
  projected divergence-free face velocities are correct; `amr` kept them and `flow` adopted them
  (`flow` `8e21724`). The parity gate's probe flipped cleanly: `flow` now matches `amr`'s default
  to **4.414e-11** and its own former convention to 3.949e-04 — the two arms exactly swapped. The
  last uniform-grid discretization difference between the engines is closed. Remaining nit: drop
  the legacy arm of the probe once this has held for a while.
- **A2 — the aperture pair (P2).** Identify which `flow` scheme, if any, matches this solver's
  aperture path, or add a matched mode. Today only `ghost` ↔ `ghost` is an exact pair, which is
  enough for production but leaves the scheme matrix incomplete.
- **A3 — cut cells under advection (P5).** A 1.6e-4 relative difference that is NOT the advecting
  velocity: the two engines reconstruct the advective flux differently at cut faces. Read the two
  reconstructions side by side; it is a half-day of reading, not a campaign.
- **A3′ — `divergence_norm_face()` lies under the ghost scheme** — **DONE 2026-09-21.** It sums
  the plain area-weighted face divergence and omits the ghost overlay delta that is part of the
  constraint actually solved, so it reads 34 at N = 32 and 184 at N = 64 on a healthy solve, and is
  unnormalized besides. Folding the overlay in is not possible — the delta is a functional of the
  CELL velocities, so the ghost residual cannot be written as a norm of `uf` at all; `divNormL2()`
  already adds it and is the right residual there. Closed by documenting it precisely in the C++ and
  the binding, and by GATING it: `test_amr_face_field` now runs a ghost arm and asserts the
  diagnostic is >1e3× the aperture residual and indistinguishable from the cell divergence.
- **A5 — a mesh whose cut cells sit BELOW the finest level was SILENTLY WRONG** — **FIXED
  2026-09-21** (`docs/amr_graded_convergence.md` §4). Found by the convergence study. Plane Poiseuille with grid-aligned walls, four **physically
  identical** 4096-leaf meshes of leaf size 2 over a 32³ box:

  | tree | max err | shape of the error |
  |---|---:|---|
  | `lmax=0, spacing 2.0` | 9.2e-14 | exact |
  | `lmax=1` unrefined | **1.1250** | the exact parabola, shifted bodily |
  | `lmax=2` unrefined | 1.4062 | ” |
  | `lmax=3` unrefined | 1.4766 | ” |

  The exact peak is 31.5, so ~3.6–4.7 %. The error is a **pure constant offset** (spread 5.8e-14),
  so the interior operator is fine and the no-slip wall value is wrong; it fits
  `1.5·(1 − 4^−lmax)` to every digit — a per-level `4^−k` term summed down the tree — and depends
  only on `lmax`, which is a tree *capacity* parameter, not a physical one. Unaffected by
  `set_ghost_projection(False)` or by `set_ghost_sampled(True)`. Refining the wall band to the
  finest level (the supported configuration) removes most of it.

  Root cause: `AmrCutCell::build` Pass 2 hoisted one `beta = μ/h_finest²` (and the diagonal built
  from it) outside the per-leaf loop, so a cut cell at level L got a wall row `4^L` too stiff. A
  bug, not an unenforced contract — the mixed-band note is explicit that a uniformly-coarse band is
  legal, and every other row was already level-aware. Fixed by scaling per leaf; bit-for-bit inert
  at the finest level. One byte-gate scenario moved (`amr.flow_sampled`), re-baselined separately.
  **Still to re-take:** every `set_ghost_sampled(True)` graded-band number in
  `amr_mixed_level_cut_band_plan.md`, headline +0.256 % included.
- **A4 — tolerance API (P4)** — **DONE 2026-09-21.** `Flow.set_pressure_tolerance(rtol)` (feeding
  both the MG-PCG and the ghost BiCGStab, as flow does) and `Flow.set_momentum_tolerance(rtol)`,
  public tier, defaults unchanged. Measured: rtol 1e-10/1e-6/1e-3 → 11/7/4 pressure iterations on
  the Z&H sphere for the same answer to 6 digits.

## A′. Catch up with `flow`'s 2026-09 solver-control work

`flow` reworked its solver *controls* in September (`3e37758`, `8acab7c`, `8a62f3a`, `70cf548`):
one **string selector** per solver instead of a boolean per option, the momentum solver chosen by
the operator's **condition number** rather than by geometry, and Chebyshev moved to the diagnostics
tier. None of that changed the method, and `amr` is not behind on *capability* — where `flow`'s
collocated path is explicitly barred from the velocity V-cycle (`!Grid::collocated`, V8
unvalidated), `amr` runs the Galerkin velocity multigrid as its momentum preconditioner by default.
What `amr` is behind on is **surface**, and `../docs/NAMING.md` asks for one spelling per concept
across the suite:

- **A′1 — string selectors.** `flow`: `diagnostics.set_velocity_solver('auto' | 'gauss_seidel' |
  'multigrid' | 'chebyshev')`. `amr`: four booleans that interact — `set_momentum_mg`,
  `set_momentum_gs`, `set_velocity_mg_staircase`, `set_momentum_mg_solver` — where some
  combinations are meaningless. Collapse them the way `flow` did.
- **A′2 — the pressure DRIVER is selectable from Python** — **DONE 2026-09-24.**
  `Flow.set_pressure_pcg(on)`, public as in `flow`: `True` (default) the MG-PCG, `False` the
  bounded stationary V-cycle (`setPressurePCG`). Inert under the ghost projection (BiCGStab).
  `python/test_amr.py` projects one field with each driver on a graded mesh and requires the same
  velocity. Two differences from `flow`'s `set_pressure_pcg(on, max_iter, rtol)` remain, both
  because `amr` has what `flow` lacks: `on=False` selects the V-cycle where `flow` raises (it has
  no stationary driver), and the cap and tolerance stay `pres_iters` / `set_pressure_tolerance`.
  The tolerances are bound since §A4 and the bottom since C1 — public, with `flow`'s spellings,
  since 2026-09-24 (`Flow.set_pressure_bottom(mode=)`, `set_pressure_bottom_extent(cells=)`,
  `predict_hierarchy`; `amr_mg_depth.md` §11.6). `flow`'s FCG and Chebyshev drivers have no `amr`
  counterpart.
- **A′3 — condition-number auto-selection.** `flow` picks RB-GS vs V-cycle from
  κ = 1 + 4·dt·µ·Σw/ρ. `amr` always runs the MG-preconditioned BiCGStab, which is the right default
  at large dt and probably over-solves at small dt — that is plausibly part of the §C1 cost. Worth
  measuring before copying: this is an empirical question, not an API one.

None of these change a number. They are the difference between two codes that agree numerically
and two codes that *feel* like one suite.

- **A6 — `set_cf_scheme('quadratic')` IS the DEFAULT** — **DONE 2026-09-21 (user decision).** The
  register already says cf=1 "is not optional on graded meshes" and explicitly rejects cf=0 there,
  but the shipped default is cf=0. Measured on a self-similar ladder
  (`docs/amr_graded_convergence.md` §3): on an interface NORMAL to the variation the default is a
  clean order 2.00 and the quadratic scheme is inert, so it costs nothing; on a TANGENTIAL
  interface the default manages **order 0.41** against quadratic's **1.60**, with the absolute gap
  widening from 3.8× at n=32 to **8.8× at n=64**. A real graded mesh around a curved body has both
  orientations. For the flip: it aligns the code with the register and is **inert by geometry** on
  any uniform or finest-band mesh (no C/F faces ⇒ no delta), so only graded runs move. Against: it
  is a shipped default, every graded result moves with it, and cf=1 has a stability history at cut
  rows (the `rowRegular` row-gate register entry).

  It could not be flipped until the scheme ran multi-rank: `setSolid` threw for
  `dist_ && cfScheme_ != standard`, so the default would have broken every MPI run. That guard is
  gone (`85e2664`, and `961ba10` for the two bugs it was hiding — a block-period wrap that could
  return a geometrically unrelated leaf across ranks, and `t.level()` on ghost indices). Four
  byte-gate hashes re-recorded: exactly the scenarios that build a graded mesh AND run a `Flow`.
  Direct check on a graded sphere against a uniform-fine reference: the permeability error falls
  from 5.15e-02 to 1.22e-02. **Attribution corrected 2026-09-22:** that pair is NOT "the byte
  gate's own graded sphere" (`state_hash.py`'s `sph`, r=0.22 in the unit box, lmax=1), as this
  line said until then — that mesh gives 2.969e-02 → 4.586e-03, a factor 1.7 / 2.7 away. It is
  `1b0d5b5`'s recipe, the Z&H sphere (solid fraction 0.125) in cell units at N=32, **lmax=2**,
  band=3.0, which reproduces at 5.538e-02 → 1.176e-02 and whose absolute k matches the 47.182462
  that commit records to 0.18 %. Both recipes are now in
  `tests/study/convergence/zh_graded_permeability.py`, and both have `num_cf_cut_faces == 0`, so
  both are bit-identical across the per-FACE C/F gate (`docs/amr_cf_flux_gate.md` gate P).
  One blocker is gone: until 2026-09-21 `setSolid` THREW for `dist_ && cfScheme_ != standard`, so
  the scheme could not have been a default at all. It is now distributed
  (`docs/amr_setup_parallel_plan.md` §7, ctests `amr_distributed_cf_np{1,2,4,8}`).

## B. The accuracy question the parity work did NOT answer

Parity at `lmax = 0` says the two agree **where the mesh is uniform**. It says nothing about the
refined mesh, which is the entire point of the package.

- **B1 — a graded-mesh convergence study against an analytic solution** — **rung 1 done
  2026-09-21, and the answer is SECOND ORDER** (`docs/amr_graded_convergence.md`). Plane Poiseuille
  is exactly quadratic, so the uniform arm is machine-zero (1e-13) and any graded departure is the
  C/F treatment alone: max error 0.375 → 0.09375 across a 32 → 64 refinement, exactly ÷4, with the
  fine region (walls included) exact and the whole error in the coarse cells. Open rungs, in order:
  the **tangential** interface orientation (where `set_cf_scheme('quadratic')` can actually act —
  it is inert by construction on the normal orientation, and correctly so); and a genuinely
  **mixed-level** cut band via `refine_to_sdf_graded` + `set_ghost_sampled`, whose accuracy is
  unmeasured since the A5 fix. The **smooth NS rung is DONE 2026-09-23** — see B2.
- **B2 — an unsteady NS test on a graded mesh** — **DONE 2026-09-23**
  (`docs/amr_tg_graded.md`, driver `tests/study/amr_tg_graded.py`, gate ctest
  `python_amr_tg_graded`). Open in `amr_collocated_projection.md` since 2026-07-22. A decaying
  Taylor–Green vortex with advection on, against its exact solution, on three meshes: uniform fine
  (U), uniform coarse (C), and the graded mesh (G) that IS C with a spherical shell refined — so
  G − C prices the 2:1 interface by itself. **The graded solver is second order in an unsteady
  flow: 2.07 in the volume velocity error on the 32 → 64 rung**, 1.82 in the interface-generated
  part — **2.22 and 2.08 since the B5 seam reconstruction** (`amr_tg_graded.md` §3). `div(uf)` stays ≤ 4e-12 over hundreds of advecting steps on a mesh with ~7 000 2:1
  sub-faces, which is the conservation benefit B2 was opened to exercise. It also settled the
  pressure-iteration question (§B4).
- **B3 — sub-face closures** (`amr_mixed_level_cut_band_plan.md` §8a, risk register). Bounded, not
  retired: 0.20 % of rows at depth 7, 1.57 % at depth 8. An accuracy item with an open design fork,
  not a stability blocker. **It has acquired a second reason to exist** (`amr_cf_flux_gate.md` §5
  option C): folding the C/F substitution into the ghost closure at cut rows — constraint AND
  gradient together — is what would restore the quadratic face value on the sub-faces where a level
  boundary meets the wall, which the per-FACE gate now withholds. Default: **defer**. The decision
  wants data, and `Flow.diagnostics.num_cf_cut_faces` plus the two-sphere policy-error gate supply
  it: if the policy error moves by less than the ghost scheme's own 0.2–0.3 % bias, the O(h)-on-a-
  curve loss is below the noise and C is not worth its cost (a change inside `ghost_projection*.hpp`,
  a new invisible-subspace analysis, and a re-run of the whole throat ladder).

- **B4 — the deferred-corrected C/F pressure gradient (option B)** — **DESIGNED, PARKED
  2026-09-23.** Design `amr_pressure_iteration.md` §6–§8, verdict §14. The worry was that the
  advecting face velocity at a 2:1 sub-face carries a tangential leak of the pressure increment
  that the standard pressure matrix cannot cancel. Measured directly on the benchmark, the leak is
  **1.0e-4 at N = 32, CFL 0.5 — four parts in a thousand of the face error the solver makes anyway
  — and it is O(dt²)**, so it shrinks twice as fast as the time step (gate S, answered at last).
  The gate as originally written (`m1/m3 ≥ 0.3`) fired for an unrelated reason and is re-spelled in
  §14.3 on the leak itself. (B) is a designed, unbuilt option; its trigger is a consumer running a
  graded octree at `dt ≫` CFL with an evolving pressure (a scalar-transport or coupling driver),
  not this benchmark. Nothing to do until then.

- **B5 — the 2:1 interface truncation constant** — **DONE 2026-09-23**
  (`docs/amr_cf_convective.md`, verdict and gates §12). At a 2:1 face the coarse cell's column is
  offset tangentially from the sub-face by half a fine cell, so the advected value was **first
  order** there — and because the four sub-faces of one coarse face carry that error with opposite
  signs, the coarse cell barely noticed while each **fine** cell saw an O(1) forcing alternating
  across the 2×2 patch. Two same-level faces beside the seam were broken too (their second upwind
  sample sits across the jump), at 16× the true bulk truncation. The fix reconstructs from the
  upwind side with level-aware probes — the coarse value tangentially sampled by the existing
  `cfAppendStencil`, applied **once** to the extrapolated value, and an upstream probe at another
  level taken at its true distance. Measured at N = 64: the one-step seam truncation
  **7.881e-03 → 2.008e-03** and its ratio to the bulk **9.23 → 3.40**; the graded mesh against the
  *same mesh without its refined shell* **1.556× → 1.218×**, against a perfect-seam ceiling of
  **0.96×** (the shell is 6.8 % of the volume, so refining it can buy 4 %). Inert without
  advection and on any uniform mesh, bit-identical with the switch off, ~5e-9 at np = 2/4/8, no
  measurable cost. `Flow.diagnostics.set_seam_reconstruction(bool)`, default on.

  Two things the work established that outlive it. **Refinement pays once it covers enough of the
  domain**: thicken the shell to 42 % of the volume and the graded mesh reads 0.93× the unrefined
  one, where before the fix it still lost at 1.46×. And **the penalty does not compound over a deep
  hierarchy** — a finest-level band plus `balance`, one seam per level, gives 1.27× / 1.18× / 1.05×
  of the ceiling at 2 / 3 / 4 levels with 33 % *more* seam faces, because each seam's cost is set
  by the cells it separates and the coarsest one dominates.

- **B6 — matching the flux-error constants across a seam** (NEW 2026-09-23, **low priority**).
  What is left after B5 is the jump in the truncation constant itself when the cell width doubles:
  adjacent faces no longer share an error that cancels in their difference, which is what makes a
  second-order scheme second order. It is one order lower on a codimension-1 set, which keeps the
  global order (Gustafsson 1975; Kreiss et al. 1986) and leaves the constant every AMR code
  carries. The lever is **matching** the two sides' constants, not maximising either — a higher-
  order flux on the coarse cells beside the seam, or a buffer of intermediate constant, or the mesh
  generator's band thickness. `amr_cf_convective.md` §4 fact 3 already spends the little freedom
  there is (the true-distance probes, L2 ratio 0.81 measured). Worth ~0.22× on a smooth flow;
  less where refinement sits on the feature. Do not reopen this as a sub-face stencil question —
  §12.1 closes that.

- **A7 — the C/F face-value delta is gated per FACE** — **DONE 2026-09-22.** `buildCfDivDelta` was
  gated per CELL (`rowRegular`) and `buildCfUfDelta` per face-pair fluidity, so at a 2:1 sub-face
  with a cut cell on one side the regular cell booked the C/F correction in its divergence
  constraint, the cut cell did not, and `uf` carried it for both: a permanent mass source,
  `‖D(Δvel) − Δ_cfDiv‖ = 2.018e-01` against `‖div(uf)‖ = 2.016e-01`. A face flux is one number
  shared by two cells, so no per-cell rule can be conservative. The quadratic face value now
  applies iff BOTH incident cells are regular fluid, and that one predicate drives D, the ABC
  gradient substitution and `uf` through one emitter. See `amr_cf_flux_gate.md`; the cost is the
  standard two-point face value on a codimension-2 set, counted by `num_cf_cut_faces`.

## C. Cost

- **C1 — CLOSED 2026-09-23. The step cost was the PRESSURE multigrid, whose hierarchy stopped at
  the root brick; it now continues below it.** Design `docs/amr_mg_depth.md`; work orders WO1–WO6.

  The cause was structural: `AmrMultigrid::build` coarsens by merging octree siblings, and a leaf
  that is already a ROOT cell has no siblings — so `levels = lmax + 1`, and on a **uniform mesh, at
  any lmax, the hierarchy was ONE level**, i.e. a smoother with no coarse-grid correction at all.
  The fix is that a level below the root brick is **the same octree with its root LIFTED** (brick
  halved, `lmax` incremented, leaf codes untouched), so `coarsenIf` simply keeps merging and every
  builder downstream — `AmrPoisson::init`, the openness ladder, the covering-leaf `c2p`, the device
  face-CSR assembly, the per-level `LeafHalo` — works verbatim. Measured (host-openmp, 4 threads,
  ghost projection, advection on; `tests/study/amr_pressure_depth.py`):

  | N | lmax | mesh | leaves | root brick | MG levels | ms/step | pres it |
  |---|---|---|---|---|---|---|---|
  | 32 | 0 | uniform | 32 768 | 32³ | 1 → **4** | 144.7 → **33.0** | 13 → 11 |
  | 32 | 1 | uniform | 4 096 | 16³ | 1 → 3 | 18.0 → 9.5 | 8 → 9 |
  | 32 | 1 | graded | 17 928 | 16³ | 2 → 4 | 34.8 → 20.1 | 9 → 9 |
  | 32 | 2 | graded | 17 200 | 8³ | 3 → 4 | 20.5 → 18.1 | 10 → 9 |
  | 64 | 0 | uniform | 262 144 | 64³ | 1 → **5** | 1941.8 → **~300** | 23 → 13 |
  | 64 | 1 | graded | 83 672 | 32³ | 2 → 5 | 204.9 → 85.6 | 14 → 12 |
  | 64 | 2 | graded | 65 360 | 16³ | 3 → 5 | 71.9 → 59.6 | 12 → 12 |
  | 64 | 3 | graded | 64 240 | 8³ | 4 → 5 | 58.0 → 56.3 | 12 → 12 |

  **Every configuration is faster and none is slower**; the uniform 64³ case is 6.4× and its
  pressure iterations fall 23 → 13. The per-cell comparison that motivated the work now reads:
  on a properly graded mesh (N = 64, lmax = 3) amr costs ~0.88 µs/leaf against `flow`'s 0.84
  µs/cell, and the uniform 64³ case — the one the parity harness lives on — is no longer an
  artefact of the hierarchy stopping at the root. **The 4–7× against `flow` was the uniform-mesh
  artefact, and it is gone.**

  What landed, in order: **WO1** `BlockOctree::liftRoot` + the single-rank ladder, with the
  everywhere-refined deeper tree as a bitwise oracle; **WO2** `predictPressureLadder` + the
  `pressure_mg_levels` / `pressure_mg_bottom` read-outs, so tests assert against the rule rather
  than a literal; **WO3** the distributed lockstep lift (depth Allreduced before any level is built,
  the ORB carried by `BlockDecomposer::coarsened`); **WO4** the `MgStage` abstraction with its
  replicated instantiation — below the nested ladder a level MOVES onto a new decomposition of its
  own grid and the ladder continues there, the movement behind the stage and never in `vcycle`
  (`docs/amr_mg_core_boundary.md`; sibling-merge and repartition stages are WO4b); **WO5** the
  exact agglomerated `GraphAMG`-PCG bottom for ladders that run out above the bottom extent, with
  `set_pressure_bottom`. Six new ctests at np = 1, 2, 4, 8 where distributed.

  Left open, each with its evidence recorded rather than its question re-openable:
  - **the pressure iteration count still grows mildly with N** (11/13/15 at N = 32/64/128 on the
    cut-cell sphere). `amr_mg_depth.md` §11.9's discriminator has been RUN: the same growth appears
    on the openness-free periodic Poisson (14/16/17, rough rhs), so it is **the ladder's transfer
    pair**, not the level-0 cut-cell operator, and the transfer pair is where a future session
    should look. Its own item.
    - *(sub-observation, not worth acting on — §11.12.)* The deepest level is **exactly free**
      without a solid (5 iterations, flat) and costs ~1 iteration plus visible variance (11–15
      against 11–13) with one, even though it is the same 64-cell grid with the same coarsened
      geometry in the uniform and graded ladders. Different phenomenon from the N-growth above;
      recorded with its evidence in case it ever looks like it matters.
  - **the bottom extent stays 4** — measured and resolved, §11.4: the dependence is ~1 iteration
    and 3–5 % of the step on one case in three, zero on graded meshes and zero without a solid.
  - **the exact bottom's host/device threshold** (§11.11): the host CG is ~16 % *faster* than 60
    Jacobi sweeps at a 64-cell bottom and ~28 % *slower* at 4096 rows, so §5.5's 10⁴-row threshold
    is about an order of magnitude too high. Numbers waiting for whoever builds the device path.
  - **WO4b is BUILT (S3, 2026-09-24) but OPT-IN**: the sibling-merge and repartition stages run on
    core's stage machinery and pass their gate (`amr_mg_depth.md` WO4b as built); making them the
    default waits on two policy inputs the notes leave open, `amr_mg_core_boundary.md` §9.6
    (`maxBlockCells`: leaf or fine-cell count) and §9.7 (`minExtent`), and then on
    `predictPressureLadder` learning those rows. **WO7** (coarse-first alignment) landed for
    `rebalance` (`chooseAlignedWeighted`); `init`'s unweighted ORB is unchanged. **WO8** (the
    velocity multigrid, `liftRoot = false` until then).

- **C2 — the setup cost at bed scale.** `set_solid` with a Python SDF callable is the measured
  bottleneck (>1h43m of numpy on an 11.35M-leaf bed); `set_solid_spheres` exists as the escape
  hatch. Whether the general callable path needs a device/batched form is an open question with a
  known workaround.

## D. Distributed and at-scale

- **D1 — cluster runs** (`archive/amr_march_perf_and_distributed_plan.md`, rung D3b). Needs billed
  Snellius GPU time and the user's go-ahead; the prefix has to be rebuilt first. **Blocked on
  the user, not on work.**
- **D2 — production `rho` / `max_samples`** for the sampled band (rung M2b). The M2a economy table
  is measured and in hand; picking the shipped pair is a decision, not an experiment.
- **D3 — deferred integration** of the mixed-level band: advection/`uf` seams and adapt-during-run
  rebuild (`amr_mixed_level_cut_band_plan.md` Phase 4). The MPI half is done.

## E. Surface

- **E1 — validation page** in `peclet-examples`, mirroring what `flow` and `voro` have. **The gate
  on it is lifted 2026-09-23**: the graded mesh now has two convergence results to publish — the
  steady Poiseuille ladder (B1, `amr_graded_convergence.md`) and the unsteady Taylor–Green one
  (B2, `amr_tg_graded.md`), whose (U)/(C)/(G) table and second-order rung are the page's spine.
  `docs/data/amr_tg_graded.json` is the raw data in the form the gallery's static pages want.
- **E2 — boundary conditions.** `Flow` is **triply periodic, full stop** — the box is periodic and
  the only driving is a uniform body force; `flow` carries no-slip walls, inflow/outflow, lid and
  profile BCs plus per-position profiles. The workaround in the tree is to express a wall as an
  immersed solid (the parity note's channel case does exactly that, and it is second-order there),
  which covers no-slip but NOT inflow, outflow or a prescribed pressure drop. Any driven-flow or
  open-domain application needs real BCs. Scope it before B1 if an application demands it, after
  otherwise.

---

## Decision log for this page

| when | decision | why |
|---|---|---|
| 2026-09-21 | The parity gate lives in **`amr`**, skipping with 77 when no `flow` build is pointed at (`-DPECLET_AMR_FLOW_PYTHONPATH=`). | `amr` is the code whose claim it is ("we reduce to flow"); the skip protocol already exists here for the morton-guarded tests, so CI needs no new concept. Alternative — putting it in `flow` or the umbrella — was rejected: `flow` has no reason to depend on `amr`, and the umbrella runs no tests. |
| 2026-09-21 | `set_uf_advection` is a **diagnostics-tier ablation**, default unchanged; no numerics moved. | The byte gate is IDENTICAL, so the switch costs nothing and buys the A/B that isolated the one difference. Changing the default would be reversing `../docs/decisions/amr.md:13`. |
