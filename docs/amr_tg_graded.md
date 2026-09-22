# The graded time-accurate benchmark

> Built 2026-09-22/23 as WO0 of [`amr_pressure_iteration.md`](amr_pressure_iteration.md) §9.
> Driver `tests/study/amr_tg_graded.py`, data `docs/data/amr_tg_graded.json`, gate ctest
> `python_amr_tg_graded` (label `bench`, ~20 s).

Until now the graded solver had **no unsteady test against an exact solution** — `ROADMAP.md`
carried that gap as B1/B2 and `amr_collocated_projection.md` had carried it since 2026-07-22.
Every graded number the project held was either a steady permeability or a manufactured-solution
order test. This closes it, and settles the open question of whether the advecting face velocity
at a 2:1 interface needs the deferred-correction machinery designed in `amr_pressure_iteration.md`.

## 1. The case

A decaying Taylor–Green vortex in a triply periodic box, uniform in z, with advection on
(ρ = 1, μ = 0.05, U₀ = 1, cell spacing 1, box side N, k = 2π/N):

    u =  U₀ sin kx cos ky · e^{−2νk²t}       p = (U₀²/4)(cos 2kx + cos 2ky) · e^{−4νk²t}
    v = −U₀ cos kx sin ky · e^{−2νk²t}       w = 0

This is an exact Navier–Stokes solution, and it needs the convective term: the Stokes Taylor–Green
has p ≡ 0 and therefore cannot exercise a pressure-gradient defect at all.

Three meshes, all initialised from the exact field at cell centres and run for **2 convective
times** (T = 2N), so a fixed cell CFL = U₀dt/h_fine = dt divides the horizon exactly at every N:

| arm | mesh | at N = 64 |
|---|---|---|
| **U** | uniform, every cell h = 1 | 262 144 leaves |
| **C** | uniform coarse, every cell h = 2 (`lmax = 1`, unrefined) — the control | 32 768 |
| **G** | **graded**: C with a spherical shell of radius N/4 at the box centre refined to h = 1 | 48 448 |

G ⊇ C in resolution everywhere: it is the same mesh with a shell refined. That is what makes the
comparison a clean price tag for the 2:1 interface.

## 2. What it measures

* **m3** — volume-weighted L2 of the cell velocity error, split into an **amplitude** part
  (`1−amp`: the vortex decays at the wrong rate) and a **shape** part (`m3 shape`: what is left
  after the best-fit amplitude is removed). Without the split a mesh that merely damps the vortex
  is indistinguishable from one that deforms it — and it turns out the two behave very differently.
* **m1** — the worst face-normal velocity error over the **2:1 sub-faces**, each compared to the
  exact normal velocity at its own centroid (the finer cell's face centre).
* **m2r** — the same over the **regular** (same-level) faces, and **m6 = m1/m2r**.
* **m4** — the divergence of the advecting face field.

## 3. Results (cell CFL 0.5, time-converged: halving dt again moves m3 by 0.7 % on G, 2.7 % on C)

| arm | N | leaves | m3 | shape | 1−amp | m1 (C/F) | m2r (reg) | m6 | div(uf) |
|---|---|---|---|---|---|---|---|---|---|
| U | 16 | 4 096 | 3.3205e-02 | 3.284e-03 | 7.65e-02 | — | 5.381e-02 | — | 3.0e-13 |
| U | 32 | 32 768 | 8.3112e-03 | 1.402e-03 | 1.48e-02 | — | 1.441e-02 | — | 8.7e-14 |
| U | 64 | 262 144 | 1.6729e-03 | 3.897e-04 | 2.60e-03 | — | 3.245e-03 | — | 6.9e-14 |
| C | 16 | 512 | 1.4949e-01 | 9.414e-03 | 3.46e-01 | — | 2.233e-01 | — | 2.4e-16 |
| C | 32 | 4 096 | 5.1402e-02 | 6.664e-03 | 9.23e-02 | — | 7.848e-02 | — | 5.8e-14 |
| C | 64 | 32 768 | 9.9231e-03 | 2.345e-03 | 1.54e-02 | — | 1.738e-02 | — | 1.5e-14 |
| **G** | 16 | 1 632 | 1.3244e-01 | 5.345e-02 | 2.81e-01 | 1.909e-01 | 2.534e-01 | 0.75 | 1.2e-13 |
| **G** | 32 | 8 184 | 6.1982e-02 | 2.595e-02 | 1.02e-01 | 9.073e-02 | 1.075e-01 | 0.84 | 1.3e-13 |
| **G** | 64 | 48 448 | 1.4940e-02 | 7.423e-03 | 2.07e-02 | 3.382e-02 | 3.192e-02 | 1.06 | 2.1e-13 |

Observed orders on the 32 → 64 rung:

| | m3 | shape | amplitude |
|---|---|---|---|
| U | 2.31 | 1.85 | 2.51 |
| C | 2.37 | 1.51 | 2.58 |
| **G** | **2.05** | **1.81** | **2.30** |

## 4. What it says

**(1) The graded solver is second order in an unsteady flow.** 2.05 on the cell velocity error
against the exact solution, 32 → 64, at a time step where the time error is negligible. The
interface-generated part (the shape error) converges at 1.81 — the 2:1 interface costs a constant
factor, not an order. This is the validation the project did not have.

**(2) The advecting face field stays a conservative flux.** div(uf) ≤ 4e-12 at every N and every
CFL ≤ 2, over hundreds of steps with advection on, on a mesh with ~7 000 2:1 sub-faces. The
property restored at `1b0d5b5` / `9da368c` survives an unsteady graded flow; it is now gated.

**(3) A 2:1 sub-face is not a worse place for the advecting velocity than an ordinary face.**
m6 = m1/m2r is 0.75, 0.84, 1.06 at N = 16/32/64 — the worst face-velocity error on the mesh is at
a regular face, or on a par with the C/F one. And m1 falls with refinement (orders 1.07, 1.42 at
fixed cell CFL), which disposes of "an error that does not shrink under refinement".

**(4) The pressure-increment leak at C/F faces is below 2 % of the C/F face error.** m1 at N = 32
over a 16× range of time step:

| CFL | 0.0625 | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|
| m1 | 9.2652e-02 | 9.2364e-02 | 9.1729e-02 | 9.0727e-02 | 9.0760e-02 | 9.8431e-02 | 1.4466e-01 | 2.5730e-01 |
| m3 | 6.2692e-02 | 6.2555e-02 | 6.2327e-02 | 6.1982e-02 | 6.2042e-02 | 6.8885e-02 | 1.0321e-01 | 1.9275e-01 |

The leak is proportional to the pressure increment φ, so it must **grow** with dt. m1 instead
*falls* by 2 % from CFL 0.0625 to 1 and then grows only in step with m3 — i.e. with the whole
solution going wrong, not with anything special at the interface. Whatever the leak is, it is
under 2 % of m1 at every time step anyone would call time-accurate.

**(5) The quadratic C/F scheme earns its default.** Same meshes, `set_cf_scheme('standard')`:
m3 at N = 64 is 1.6838e-02 against the quadratic's 1.4940e-02 (11 % worse) and m1 is 5.697e-02
against 3.382e-02 (68 % worse). The 2026-09-21 default stands.

**(6) The open item: a refined shell in a smooth flow costs more accuracy than it buys.**
G against C at N = 64 — the same mesh with a shell refined, 48 % more cells:

    m3       1.494e-02  vs  9.923e-03   (1.51x worse)
    shape    7.423e-03  vs  2.345e-03   (3.2x worse)
    1-amp    2.07e-02   vs  1.54e-02    (1.34x worse)

Both parts are worse, and the shape error — the genuinely local, interface-generated deformation —
is worse by more than three. The extra dissipation is not localised at the shell either: the
coarse cells far from it carry it too. Part of this is by construction (the shell is refined where
the flow has no feature, so the refinement can only pay through the interface), but the size is
worth chasing. The standing suspect is the O(h) **normal**-offset term that the Martin–Cartwright
substitution leaves in the C/F face gradient — `amr_pressure_iteration.md` §5.4 identifies it and
says it needs a three-point normal stencil, a different scheme from the tangential fix. That is a
ROADMAP item, not a defect in anything shipped.

## 5. Gate W — the deferred correction (B) is parked

`amr_pressure_iteration.md` §10 gate W asks for m5 = m1/m3 at the largest time-accurate dt, and
builds (B) if m5 ≥ 0.3. Measured, m5 is **1.46** at N = 32, CFL 1 — the gate's own trigger.

**The statistic does not measure what the gate wanted.** m5 divides a max over a codimension-1 set
by a volume L2; it reads 1.4–2.3 on this mesh at *every* time step, including CFL 0.0625 where the
pressure increment is 16× smaller. A ratio that is flat in dt cannot be a measure of a
dt-proportional leak. The quantities that do measure it are (3) and (4) above, and both say the
C/F face is unremarkable.

**Recommendation: park (B), as its own §9 default anticipated** ("Otherwise the option is parked
with this note"). The note's design (§6–§8) stands and can be built from as written the day a
consumer appears — a scalar-transport or coupling run on a graded octree at dt ≫ CFL with an
evolving pressure (§12 risk 2). Gate S is answered too: m1 shows **no** order-1 dt dependence, so
§1's argument does not fail.

## 6. Re-running

```bash
PYTHONPATH=<build> python tests/study/amr_tg_graded.py --gate              # the ~20 s regression gate
PYTHONPATH=<build> python tests/study/amr_tg_graded.py \
    --n 16 32 64 --arms U C G --cfl 0.5 2 --json docs/data/amr_tg_graded.json   # §3, ~7 min
PYTHONPATH=<build> python tests/study/amr_tg_graded.py \
    --n 32 --arms G --cfl 0.0625 0.125 0.25 0.5 1 2 4 8                    # §4 (4), ~30 s
```

`Flow.diagnostics.face_topology()` (added with this benchmark) is what makes m1/m2r possible from
Python: it returns the CSR row offsets, neighbour, axis and direction that `Flow.face_field()` is
indexed by, so a 2:1 sub-face and its centroid can be located outside C++.
