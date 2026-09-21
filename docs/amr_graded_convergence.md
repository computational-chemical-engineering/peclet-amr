# Graded-mesh convergence: is the solver still second order across a 2:1 interface?

> Status: **rung 1 measured, 2026-09-21.** The answer is **yes, second order**, and getting there
> turned up a silent O(1) wall defect that had to be fixed first (§4).
>
> This is ROADMAP B1, the hole the uniform-grid parity work explicitly did not fill: parity with
> `peclet.flow` at `lmax = 0` establishes that the solver is right *where the mesh is uniform*, and
> says nothing about the refined mesh, which is the entire point of the package.

## 1. Why Poiseuille, and why exactness rather than a ladder

Plane Poiseuille driven by a body force has the exact solution

```
u_y(x) = (F / 2mu) (x - a) (b - x),    a < x < b,    u_x = u_z = 0
```

which is **exactly quadratic**. A second-order scheme is exact on quadratics, so it must reproduce
this AT EVERY NODE to solver tolerance — not "converge at order 2", *be exact*. That is a much
sharper instrument than an error ladder on a generic solution: the uniform-mesh arm comes out at
machine zero, so on a graded mesh **any** departure is the coarse/fine treatment laid bare, with no
discretization error of its own to hide behind. `peclet.flow` reasons the same way in
`scripts/verify_poiseuille_flow.py`.

Geometry, chosen so that one thing at a time can break it:

- walls at `x = a, b` on cell faces of **both** the coarse and the fine grid, so they are
  grid-aligned everywhere and the cut-cell aperture machinery is trivial — no cut-cell error to
  confound the C/F measurement;
- the **wall band** refined to the finest level, the channel interior left coarse. That keeps the
  cut cells uniformly finest (§4 is what happens when they are not) and puts the 2:1 interface in
  the interior, **normal to the direction the solution varies in** — the orientation where the flux
  across the interface has to be right;
- `div u = 0` identically and the pressure is constant, so the projection is a no-op
  (`last_pres_iters` comes back 0) and what is measured is the momentum operator plus the C/F
  scheme, nothing else.

Driver: `tests/study/convergence/graded_poiseuille.py`.

## 2. Result: second order

Box 32³, channel `x ∈ (8, 24)`, `lmax = 1`, wall band 2 cells, 30 steps at `dt = 1e6`
(i.e. a Stokes solve), host-OpenMP Release. Exact peak `u_y = 32`.

| n_fine | mesh | cf scheme | leaves | max err | ratio | L2 (vol-wtd) |
|---:|---|---|---:|---:|---:|---:|
| 32 | uniform fine | standard | 32768 | 1.0658e-13 | — | 5.3752e-14 |
| 32 | graded | standard | 18432 | 3.7500e-01 | — | 2.6517e-01 |
| 32 | graded | quadratic | 18432 | 3.7500e-01 | — | 2.6517e-01 |
| 64 | uniform fine | standard | 262144 | 1.4175e-12 | — | 9.4836e-13 |
| 64 | graded | standard | 90112 | 9.3750e-02 | **4.00** | 8.1190e-02 |
| 64 | graded | quadratic | 90112 | 9.3750e-02 | **4.00** | 8.1190e-02 |
| 128 | uniform fine | standard | 2097152 | 1.6698e-12 | — | 1.2494e-12 |
| 128 | graded | standard | 491520 | 2.3437e-02 | **4.00** | 2.1924e-02 |
| 128 | graded | quadratic | 491520 | 2.3437e-02 | **4.00** | 2.1924e-02 |

**The uniform arm is exact at every resolution** — 1.07e-13 at 32³ through 1.67e-12 at 2.1M leaves,
a 64× range in cell count — which is what licenses reading any graded departure as scheme error.
**The graded max error falls by exactly 4.00 at each refinement: clean second order**, on three
points.

### Read the max, not the L2 — this ladder is not a self-similar mesh family

The volume-weighted L2 column is printed for completeness but is **not a valid order estimate
here**, and the reason is a trap worth stating plainly because it is easy to publish by accident.
`refine_to_sdf`'s `band` is a count of CELLS, so the refined region shrinks *physically* as the
ladder climbs and the mesh changes shape under it. Measured coarse (level-1) fraction of the
channel: **0.111 at n=32, 0.273 at n=64, 0.216 at n=128** — not constant, and not even monotone. An
L2 norm taken over a different mix of coarse and fine cells at each rung is not comparing like with
like, which is why that column wanders (1.71, 1.89) instead of converging.

The **max** error is immune to this, which is why the second-order claim rests on it: it is a local
quantity attained at a 2:1 interface whose local geometry — a coarse cell of width `2h` against
fine cells of width `h` — is identical at every rung, however much volume sits on either side.

A ladder that wants a meaningful L2 has to hold the refined region fixed in PHYSICAL units, which
means building the mesh from an explicit physical predicate rather than a cell-count band. Not done
here; the max-norm result did not need it.

Where the error lives, at `n = 32` (levels: **0 is the finest**):

| | cells | max err |
|---|---:|---:|
| level 0 (fine, includes the walls) | 16384 | 1.21e-13 |
| level 1 (coarse, channel interior) | 2048 | **3.7500e-01** |

so the fine region — walls included — is exact, and the entire error is carried by the coarse cells
adjacent to the interface. That is the expected signature of a C/F flux error propagating into the
coarse region as a smooth (here nearly constant) correction.

## 3. `set_cf_scheme('quadratic')` is inert here, and that is correct

Standard and quadratic agree to **6.0e-14** — byte-identical for practical purposes. That is not a
bug and not a wiring failure: the Martin–Cartwright correction is a **tangential** quadratic, and
this geometry deliberately has *no tangential variation* along the interface (the solution varies in
`x`; the interface is a plane `x = const`). There is nothing for it to correct, and what remains is
the normal-direction two-point flux error, which that scheme does not address.

The consequence for reading `set_cf_scheme`'s docstring — *"0 = standard two-point flux (default,
1st-order at level boundaries), 1 = … (2nd-order)"* — is that those orders are **local truncation
orders at C/F rows**, measured as such in `tests/test_amr_cf_vector.cpp`. They are not the order of
the solution: here the default scheme delivers a second-order *solution* while being first-order in
local truncation at the interface, because those rows sit on a set of codimension 1 whose
contribution to the global error is one order higher. `docs/amr_mixed_level_cut_band_plan.md` makes
the same argument for seam rows on a codim-2 set.

**Not yet measured:** the complementary orientation, interface *parallel* to the variation, where
the tangential correction does have something to do. Driver written and ready:
`tests/study/convergence/graded_poiseuille_tangential.py`.

## 4. What had to be fixed first: the cut-row viscous coefficient (ROADMAP A5)

The first attempt at this study put the 2:1 interface in the interior and left the **walls coarse**.
That configuration is silently wrong, and the discovery is worth more than the study that found it.

Four **physically identical** meshes — 4096 leaves of size 2 over a 32³ box — differing only in
`lmax`:

| tree | max err (before) | max err (after) |
|---|---:|---:|
| `lmax=0, spacing 2.0` | 9.2371e-14 | 9.2371e-14 |
| `lmax=1` unrefined | 1.1250 | **9.2371e-14** |
| `lmax=2` unrefined | 1.4062 | **9.2371e-14** |
| `lmax=3` unrefined | 1.4766 | **9.2371e-14** |

~3.6–4.7 % of the peak, as a **pure constant offset** (spread 5.8e-14): the exact parabola, shifted
bodily, which is why it looks like a perfectly good answer.

**Mechanism.** `AmrFlow::betaPerAxis()` returns `mu / h_finest²`, and `AmrCutCell::build` Pass 2
hoisted that one `beta` (and the diagonal `AC0` built from it) *outside* the per-leaf loop. The
ξ-overlay cut row is a pure multiple of the `beta` it is handed, so a cut cell at level `L` was
`4^L` too stiff — an effective viscosity `mu·4^L` in the wall cells only. Solving the perturbed 1-D
system gives the offset in closed form,

```
offset = -(3/8) (F/mu) h_leaf^2 (1 - 4^-L)
```

which is `-1.5·(1 - 4^-L)` for this case and reproduces all four measured values to every digit.
Every *other* row in the operator was already level-aware: the regular fluid rows use physical areas
and volumes (`assembleOperator`, `-mu*invV*(a*c)`), `velocity_mg.hpp` computes `mu/cellWidth²` per
row, and `ghost_projection_sampled.hpp` uses "the ROW-LOCAL beta per axis" — but only on *seam*
rows, which is exactly why a uniformly **coarse** band, having no seams, kept the wrong coefficient.

**Verdict: a bug, not an unenforced contract.** `docs/amr_mixed_level_cut_band_plan.md` §"the
uniform finest band is policy, not discretization contract" says this mesh is legal, and the
projection side honours it; only the momentum ξ row did not. The fix scales `beta` and `AC0` by the
leaf's level inside the loop, in the host builder and its device twin. It is **exactly inert at the
finest level** (`f = 1.0`, so the expressions are bit-for-bit the old ones), which is what keeps the
blast radius to a single scenario.

**Blast radius, measured not assumed:** of the nine byte-gate scenarios exactly one moved —
`amr.flow_sampled`, the `refine_to_sdf_graded` case, half of whose band sits at level 1. Every
other hash, `amr.flow_ghost` (uniform finest band) included, is bit-identical. Re-baselined in its
own commit.

**Contaminated measurements to re-take:** every `set_ghost_sampled(True)` graded-band number in
`docs/amr_mixed_level_cut_band_plan.md`. Its headline (§P3c: gap-graded reproduces `k` to +0.256 %)
was taken at depth 7 where the mesh saving was only 1.07×, i.e. nearly every cut cell was still at
the finest level, so that arm barely exercised the defect; the depth-8 arm (1.62× saving) is where
it would bite.

## 5. Next rungs

1. **The tangential orientation** (§3) — where `set_cf_scheme('quadratic')` should finally show a
   difference, and the one place the documented 2nd-order C/F claim can be tested end to end.
2. **A smooth NS ladder on a graded mesh** — Taylor–Green with a refined sub-region, so the
   measurement covers advection and the projection rather than a Stokes momentum solve alone.
   Poiseuille cannot test those: its projection is a no-op by construction.
3. **Re-take the P3c gap-graded drag numbers** post-fix (§4).
4. **A cut band that is genuinely mixed-level**, via `refine_to_sdf_graded` + `set_ghost_sampled`,
   which is the configuration the sampled machinery exists for and the one whose accuracy is now
   unmeasured after the fix.
