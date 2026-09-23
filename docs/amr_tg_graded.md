# The graded time-accurate benchmark

> Built 2026-09-22/23 as WO0 of [`amr_pressure_iteration.md`](amr_pressure_iteration.md) §9,
> reviewed there in §14. Driver `tests/study/amr_tg_graded.py`, data
> `docs/data/amr_tg_graded.json`, gate ctest `python_amr_tg_graded` (label `bench`, ~25 s).

Until now the graded solver had **no unsteady test against an exact solution** — `ROADMAP.md`
carried that gap as B1/B2 and `amr_collocated_projection.md` had carried it since 2026-07-22.
Every graded number the project held was either a steady permeability or a manufactured-solution
order test. This closes it, and settles whether the advecting face velocity at a 2:1 interface
needs the deferred-correction machinery designed in `amr_pressure_iteration.md`.

## 1. The case

A decaying Taylor–Green vortex in a triply periodic box, uniform in z, with advection on
(ρ = 1, μ = 0.05, U₀ = 1, cell spacing 1, box side N, k = 2π/N):

    u =  U₀ sin kx cos ky · e^{−2νk²t}       p = (ρU₀²/4)(cos 2kx + cos 2ky) · e^{−4νk²t}
    v = −U₀ cos kx sin ky · e^{−2νk²t}       w = 0

An exact Navier–Stokes solution, and it needs the convective term: the Stokes Taylor–Green has
p ≡ 0 and cannot exercise a pressure-gradient defect at all. **Both** velocity and pressure are set
from the exact solution at t = 0 — without the pressure the first step is an impulsive start whose
projection must manufacture the whole pressure field at once, worth 7–8e-3 of error on every arm
(`amr_pressure_iteration.md` §14.4b).

Three meshes, run for **2 convective times** (T = 2N), so a fixed cell CFL = U₀dt/h_fine = dt
divides the horizon exactly at every N:

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
  is indistinguishable from one that deforms it — and the two behave very differently here.
* **m1** — the worst face-normal velocity error over the **2:1 sub-faces**, each against the exact
  normal velocity at its own centroid (the finer cell's face centre); **m2r** the same over the
  **regular** faces, and **m6 = m1/m2r**.
* **ε_cf** — the **pressure-increment leak** itself: `max |Δ_G φ|` over the 2:1 sub-faces, built
  from the solver's own last-step `φ = (dt/ρ)(pⁿ⁺¹ − pⁿ)`. This is the quantity the whole
  deferred-correction question turns on. m1 is a *total* and is dominated by the solver's spatial
  error; ε_cf is the defect alone.
* **m4** — the divergence of the advecting face field.

## 3. Results (cell CFL 0.5)

| arm | N | leaves | m3 | shape | 1−amp | m1 (C/F) | m2r (reg) | m6 | ε_cf | ε_cf/m2r | div(uf) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U | 16 | 4 096 | 2.6594e-02 | 3.299e-03 | 6.11e-02 | — | 4.475e-02 | — | — | — | 3.0e-13 |
| U | 32 | 32 768 | 5.7998e-03 | 1.223e-03 | 1.03e-02 | — | 1.123e-02 | — | — | — | 1.6e-14 |
| U | 64 | 262 144 | 9.2494e-04 | 2.783e-04 | 1.41e-03 | — | 2.285e-03 | — | — | — | 6.3e-14 |
| C | 16 | 512 | 1.4578e-01 | 9.549e-03 | 3.37e-01 | — | 2.188e-01 | — | — | — | 2.2e-16 |
| C | 32 | 4 096 | 4.9354e-02 | 6.670e-03 | 8.85e-02 | — | 7.567e-02 | — | — | — | 5.7e-14 |
| C | 64 | 32 768 | 9.2112e-03 | 2.265e-03 | 1.43e-02 | — | 1.647e-02 | — | — | — | 2.1e-14 |
| **G** | 16 | 1 632 | 1.2388e-01 | 4.822e-02 | 2.64e-01 | 1.712e-01 | 2.364e-01 | 0.72 | 2.32e-04 | 9.8e-04 | 7.3e-14 |
| **G** | 32 | 8 184 | 5.2383e-02 | 2.248e-02 | 8.57e-02 | 6.510e-02 | 9.058e-02 | 0.72 | 9.10e-05 | 1.0e-03 | 2.2e-13 |
| **G** | 64 | 48 448 | 1.1216e-02 | 5.318e-03 | 1.58e-02 | 1.614e-02 | 2.132e-02 | 0.76 | 1.39e-05 | 6.5e-04 | 1.2e-13 |

(The **G** rows carry the B5 seam reconstruction, on by default since 2026-09-23
— `amr_cf_convective.md`. Before it they read m3 1.2900e-01 / 6.0236e-02 / 1.4328e-02, shape
5.442e-02 / 2.616e-02 / 7.417e-03, 1−amp 2.71e-01 / 9.82e-02 / 1.96e-02, m1 1.855e-01 /
8.865e-02 / 3.334e-02. **U** and **C** have no 2:1 faces and are unchanged to the last digit.)

Observed orders on the 32 → 64 rung:

| | m3 | shape | amplitude |
|---|---|---|---|
| U | 2.65 | 2.14 | 2.87 |
| C | 2.42 | 1.56 | 2.63 |
| **G** | **2.22** | **2.08** | **2.44** |

## 4. What it says

**(1) The graded solver is second order in an unsteady flow.** 2.22 on the cell velocity error
against the exact solution, 32 → 64, at a time step where the time error is small. The
interface-generated part (the shape error) converges at 2.08 — the 2:1 interface costs a constant
factor, not an order. This is the validation the project did not have.

**(2) The advecting face field stays a conservative flux.** div(uf) ≤ 4e-12 at every N and every
CFL ≤ 2, over hundreds of steps with advection on, on a mesh with ~7 000 2:1 sub-faces. The
property restored at `1b0d5b5` / `9da368c` survives an unsteady graded flow; it is now gated.

**(3) The pressure-increment leak at the coarse/fine faces is four parts in a thousand of the face
error the solver makes anyway.** Measured directly on the graded mesh:

| N | CFL 0.5 | CFL 1 | CFL 2 | dt-order |
|---|---|---|---|---|
| 32 | ε_cf = 1.01e-04 (0.09 % of m2r) | 4.30e-04 (0.43 %) | 1.61e-03 (1.8 %) | 2.10, 1.90 |
| 64 | 1.72e-05 (0.05 %) | 6.93e-05 (0.22 %) | 2.39e-04 (0.83 %) | 2.01, 1.79 |

It is **O(dt²)** at fixed mesh — the prediction `amr_pressure_iteration.md` §1 makes and §10 gate S
asks for — and it falls by ~6× per mesh doubling at fixed CFL. A leak growing like dt is what would
have forced building the deferred correction; this one shrinks twice as fast as the time step.

**(4) A 2:1 sub-face is not a worse place for the advecting velocity than an ordinary face.**
m6 = m1/m2r is 0.72, 0.72, 0.76 at N = 16/32/64 — the worst face-velocity error on the mesh is at a
regular face by a clear margin (it was 0.74 / 0.83 / 1.04 before the B5 seam reconstruction).

**(5) The quadratic C/F scheme earns its default.** On the same meshes with
`set_cf_scheme('standard')` the velocity error is 11 % worse and the C/F face error 68 % worse than
with the quadratic scheme (measured before the pressure IC was added; the ablation table of
`amr_pressure_iteration.md` §14.4c has the current numbers). The 2026-09-21 default stands.

**(6) A refined shell in a smooth flow still costs more accuracy than it buys — but far less, and
the item is closed.** G against C at N = 64, CFL 0.5 — the same mesh with a shell refined, 48 %
more cells — before and after the B5 seam reconstruction (`amr_cf_convective.md`):

    m3       1.433e-02 -> 1.122e-02   against C's 9.211e-03   (1.56x -> 1.22x)
    shape    7.417e-03 -> 5.318e-03   against C's 2.265e-03   (3.27x -> 2.35x)
    1-amp    1.96e-02  -> 1.58e-02    against C's 1.43e-02    (1.37x -> 1.10x)

**What the ceiling actually is.** Splicing the uniform-fine solution into the refined shell and the
uniform-coarse one everywhere else — the answer a *perfect* seam would give on this very mesh —
reads **8.83e-03**, i.e. **0.96x** C. The shell is only **6.8 % of the volume**, so refining it can
buy 4 % and no more; this benchmark refines where the flow has no feature, by design, to stress the
seam. The right figure of merit is therefore the distance to that ceiling, which went from **1.62x
to 1.27x**.

**Refinement does pay once it covers enough of the domain.** Thickening the shell at N = 64
(`--band`), measured against the perfect-seam ceiling for each:

| shell | leaves | volume refined | ceiling | before | after |
|---|---|---|---|---|---|
| 1 cell | 48 448 | 6.8 % | 0.97 | 1.56 | 1.22 |
| 3 cells | 60 488 | 12.1 % | 0.94 | 1.69 | 1.30 |
| 6 cells | 80 256 | 20.7 % | 0.90 | 1.80 | 1.31 |
| 12 cells | 129 144 | 42.0 % | 0.78 | 1.46 | **0.93** |

At 42 % refined the graded mesh now beats the unrefined one outright, where before it still lost.
The seam's toll is roughly constant in relative terms because it lives on a surface while the gain
lives in a volume.

**And it does not compound over a deep hierarchy** — the case that matters for real use. A one-cell
band at the finest level on a sphere plus `balance`, i.e. refine sharply and coarsen as fast as 2:1
allows, one seam per level:

| levels | leaves | seam faces | ceiling error | before | after |
|---|---|---|---|---|---|
| 2 | 48 448 | 19 968 | 8.83e-03 | 1.62x | 1.27x |
| 3 | 24 088 | 24 960 | 5.63e-02 | 1.39x | 1.18x |
| 4 | 21 624 | 26 496 | 2.10e-01 | 1.14x | **1.05x** |

A third more seam faces, half the cells, and the gap to a perfect seam *falls*. Each seam's cost is
set by the cells it separates, so the coarsest seam dominates — and there is only one of those. It
is a converging series, not an accumulating one. (Caveat: this flow has its error spread evenly. In
a bed the error sits at the interface where the finest cells are, and the balance shifts toward the
inner seams; the standing evidence there is the cut-cell sphere permeability, 1.2 % against a
uniform-fine reference.)

**What is left is intrinsic, not a stencil defect.** The residual is the jump in the flux-error
constant when the cell width doubles: adjacent faces no longer share an error that cancels in their
difference, which is what makes a second-order scheme second order. That is one order lower on a
codimension-1 set, which keeps the global order (Gustafsson 1975; Kreiss et al. 1986) and leaves a
constant every AMR code carries. The remaining lever is to **match** the two sides' error constants,
not to maximise either — `amr_cf_convective.md` §4 fact 3 — which is a different scheme, parked.

## 5. Gate W — the deferred correction (B) is parked

`amr_pressure_iteration.md` §10 gate W originally asked for m5 = m1/m3 ≥ 0.3. That statistic reads
1.4–2.6 here at *every* time step including the smallest, because both m1 and m3 are dominated by
the solver's spatial error — it cannot measure a defect proportional to the pressure increment.
The corrected gate (§14.3 of that note) is stated on ε_cf against m2r, and reads **1.8e-2 at N = 32
and 2.2e-3 at N = 64** at the largest time-accurate dt — sixteen to a hundred and thirty times under
the 0.3 threshold. **(B) is parked**, its design intact, to be built if a consumer appears that runs
a graded octree at dt ≫ CFL with an evolving pressure.

A note on "the largest time-accurate dt": with the exact pressure initial condition the uniform
arm's error is **not monotone in dt** — the spatial upwind dissipation and backward Euler's
under-damping have opposite signs and cross near CFL 1 (1−amp reads +1.03e-2, +3.41e-3, −1.83e-2 at
CFL 0.5/1/2, N = 32). The rule therefore admits CFL 2 at N = 32 and CFL 1 at N = 64; gate W is
quoted at each.

## 6. Re-running

```bash
PYTHONPATH=<build> python tests/study/amr_tg_graded.py --gate              # the ~25 s regression gate
PYTHONPATH=<build> python tests/study/amr_tg_graded.py \
    --n 16 32 64 --arms U C G --cfl 0.5 1 2 --json docs/data/amr_tg_graded.json   # §3, ~7 min
PYTHONPATH=<build> python tests/study/amr_tg_graded.py \
    --n 32 --arms G --cfl 0.0625 0.125 0.25 0.5 1 2 4 8                    # the dt ladder, ~40 s
PYTHONPATH=<build> python tests/study/amr_tg_graded.py --n 64 --arms G --seam off   # the B5 A/B
PYTHONPATH=<build> python tests/study/amr_tg_graded.py --n 64 --arms G --band 12     # §4(6)
```

The gate checks five things: div(uf) ≤ 1e-9, ε_cf/m2r ≤ 0.05 at CFL 1 (gate W's regression bound,
12× margin), m1 ≤ 1.1·m2r, the graded h-order (floored at 0.9 on the pre-asymptotic 16 → 32 rung;
second order is the 32 → 64 statement, 2.22), and the error levels themselves within 5 % of §3.
`--seam off` restores the pre-2026-09-23 advected-value reconstruction for an A/B on one build.

`Flow.diagnostics.face_topology()` (added with this benchmark) is what makes m1/m2r/ε_cf possible
from Python: it returns the CSR row offsets, neighbour, axis and direction that `Flow.face_field()`
is indexed by, so a 2:1 sub-face and its centroid can be located outside C++.
