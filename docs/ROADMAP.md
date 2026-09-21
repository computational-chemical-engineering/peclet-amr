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

- **A1 — the advecting velocity (parity note P1)** — **DECIDED 2026-09-21 by the user: the
  projected divergence-free face velocities are correct.** `amr` keeps what it does; `flow` is
  adopting it, in the flow repo, in parallel. Nothing to do here except watch: the parity gate no
  longer pins a convention, it PROBES which one `flow` is on and says so, passing either way and
  failing only if `flow` matches neither. When the swap has landed and stuck, drop the legacy arm
  of that probe.
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

- **A6 — should `set_cf_scheme('quadratic')` be the DEFAULT?** *A decision for the user.* The
  register already says cf=1 "is not optional on graded meshes" and explicitly rejects cf=0 there,
  but the shipped default is cf=0. Measured on a self-similar ladder
  (`docs/amr_graded_convergence.md` §3): on an interface NORMAL to the variation the default is a
  clean order 2.00 and the quadratic scheme is inert, so it costs nothing; on a TANGENTIAL
  interface the default manages **order 0.41** against quadratic's **1.60**, with the absolute gap
  widening from 3.8× at n=32 to **8.8× at n=64**. A real graded mesh around a curved body has both
  orientations. For the flip: it aligns the code with the register and is **inert by geometry** on
  any uniform or finest-band mesh (no C/F faces ⇒ no delta), so only graded runs move. Against: it
  is a shipped default, every graded result moves with it, and cf=1 has a stability history at cut
  rows (the `rowRegular` row-gate register entry). Not flipped here.

## B. The accuracy question the parity work did NOT answer

Parity at `lmax = 0` says the two agree **where the mesh is uniform**. It says nothing about the
refined mesh, which is the entire point of the package.

- **B1 — a graded-mesh convergence study against an analytic solution** — **rung 1 done
  2026-09-21, and the answer is SECOND ORDER** (`docs/amr_graded_convergence.md`). Plane Poiseuille
  is exactly quadratic, so the uniform arm is machine-zero (1e-13) and any graded departure is the
  C/F treatment alone: max error 0.375 → 0.09375 across a 32 → 64 refinement, exactly ÷4, with the
  fine region (walls included) exact and the whole error in the coarse cells. Open rungs, in order:
  the **tangential** interface orientation (where `set_cf_scheme('quadratic')` can actually act —
  it is inert by construction on the normal orientation, and correctly so); a smooth **NS** ladder
  on a graded mesh, since Poiseuille's projection is a no-op and so covers neither advection nor
  the projection; and a genuinely **mixed-level** cut band via `refine_to_sdf_graded` +
  `set_ghost_sampled`, whose accuracy is unmeasured since the A5 fix.
- **B2 — an unsteady NS test.** Carried open in `amr_collocated_projection.md` since 2026-07-22:
  every steady case in the tree is `uf`-invariant by construction, so nothing currently exercises
  the conservation benefit that motivated the `uf` advection. A decaying Taylor–Green or a shedding
  case, checking tracer/energy conservation. The parity harness now gives this a free oracle at
  `lmax = 0` (§A), which is how it should be built: match `flow` uniform first, then refine.
- **B3 — sub-face closures** (`amr_mixed_level_cut_band_plan.md` §8a, risk register). Bounded, not
  retired: 0.20 % of rows at depth 7, 1.57 % at depth 8. An accuracy item with an open design fork,
  not a stability blocker.

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
