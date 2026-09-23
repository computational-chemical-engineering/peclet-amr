# peclet-amr roadmap

One page, live items only. Everything here is open; everything closed lives in the campaign notes
and in `docs/archive/`. Rewritten in place, not appended.

**Where the code stands (2026-09-21).** The octree, the distributed octree with leaf/field
rebalance, solution-adaptive refinement, the collocated cut-cell Navier–Stokes solver (ghost
projection default, aperture fallback, mixed-level sampled cut band), the AMR multigrid / MG-PCG /
BiCGStab stack and the parallel `setSolid` builders are all **built, distributed and gated**: 92
C++ ctests at np = 1, 2, 4, 8 plus 4 Python, a SHA-256 byte gate over every public entry path, and
— new — a uniform-grid parity gate against `peclet.flow`. The package is 0.x: the API may still
move (the D9 exception to the suite's clean-break 1.0.0).

What is NOT yet done is the layer above: no published validation page, no convergence study of the
*graded* solver against an analytic solution, no at-scale multi-GPU numbers, and the advective
path costs 4–7× what `flow` costs for the same step.

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
- **A′2 — the pressure driver is not selectable from Python at all.** `setPressurePCG` exists in
  C++ (`flow.hpp:504`) and is **unbound**; so is `setMomentumTol` (§A4). `flow` exposes
  `set_pressure_pcg / set_pressure_fcg / set_pressure_chebyshev / set_pressure_bottom`. A user of
  `peclet.amr` cannot currently choose the pressure driver or either tolerance.
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
  part. `div(uf)` stays ≤ 4e-12 over hundreds of advecting steps on a mesh with ~7 000 2:1
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

- **B5 — the 2:1 interface truncation constant.** **NEW 2026-09-23**, from B2's control arm. On a
  smooth flow the graded mesh's *shape* error is ~3× the unrefined coarse mesh's — enough that the
  refined shell makes the answer 1.56× worse than not refining it at all (`amr_tg_graded.md` §4(6)).
  It is dt-independent, converges at order ≈ 2 (so a constant, not an order loss), is present only
  with the convective term, and is insensitive to the limiter, implicit/explicit advection, the
  advecting face velocity and the C/F interpolation order — i.e. it is the convective flux's
  non-telescoping truncation at the 2:1 face, a first-order source on a codimension-1 set. It is
  **not** the pressure-increment leak (60× too small) and **not** the initial projection (≤ 15 %).
  `amr_pressure_iteration.md` §14.4 has the ablation table and names the one experiment that would
  identify the term: a one-step a-priori probe with `set_pressure(exact)` after four warm-up steps,
  advection on vs off, reading the interface layer against the bulk. Priority **low–medium**: a
  constant, and in a bed the refined band sits on cut cells where the resolution is genuinely
  bought, so it competes with the cut-cell error rather than with nothing.

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

- **C1 — the advective step costs 4–7× `flow`.** Measured on the same 32³ cut-cell sphere, same
  host, same threads: Stokes 303 ms vs `flow`'s 313 ms (parity), but with advection on, 277 ms vs
  41 ms. The Stokes number says the operator and the pressure solve are competitive; the advective
  number says the cost is in the momentum path (deferred-correction assembly + the momentum
  iterations it drives). Profile it — `PECLET_AMR_PROFILE_STEP=1` and `bench_amr_flow` exist for
  exactly this — before any at-scale work. This subsumes the older "profile the momentum solve at
  128³" item.
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

- **E1 — validation page** in `peclet-examples`, mirroring what `flow` and `voro` have. Gated on
  B1: there is nothing to publish until the graded mesh has a convergence result.
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
