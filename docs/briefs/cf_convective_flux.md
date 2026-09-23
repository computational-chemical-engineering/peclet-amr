# Design brief: the convective flux loses an order at a 2:1 coarse/fine face

> For a Fable design pass, 2026-09-23, on branch `b5-cf-convective` of `peclet-amr`
> (`ddeee84`, base `3c6c343`). ROADMAP item **B5**, opened by the WO0 benchmark's control arm.

## 1. The question

**What should the momentum convective flux be at a 2:1 sub-face, so that the graded solver's
truncation there is the same order as in the bulk — and is it worth what it costs?**

A decision-shaped answer names the stencil (or says the order cannot be recovered and why), says
what it costs per step and in memory, says how it stays conservative and stable, and says whether
the measured benefit justifies building it.

"Not worth it, park it with this note" is a legitimate outcome — the item is priority low–medium
and the error is a constant, not an order loss in the final answer.

## 2. What is wrong, measured

`docs/amr_tg_graded.md` §4(6): on a decaying Taylor–Green vortex the graded mesh (G) is **1.56×
worse** in the volume velocity error than the identical mesh without its refined shell (C), and its
*shape* error — what is left after the best-fit amplitude is removed — is **3.27× worse**. Adding
cells makes the answer worse. `amr_pressure_iteration.md` §14.4 ruled out the pressure-increment
leak (60× too small, and dt-flat where the leak is dt²) and the initial projection (≤ 15 %), and
showed the excess survives Koren, explicit advection, the un-projected advecting velocity and the
standard C/F scheme, collapsing tenfold only with **advection off**.

Two new instruments (`tests/study/amr_cf_advection.py`, committed with this brief) localise it.

**(a) The one-step truncation of the whole solver** (`--step`; warm up four steps, reset `u` and `p`
to the exact fields, take one step, read `(u¹−u⁰)/dt − ∂_t u_exact`). Taylor–Green solves both
Navier–Stokes and Stokes, so `∂_t u = −2νk²u` is the reference for both arms.

| N | arm | advection | rms @ interface cells | rms bulk | ratio |
|---|---|---|---|---|---|
| 16 | C | off | — | 6.05e-04 | — |
| 16 | G | off | 1.547e-02 | 1.189e-02 | 1.30 |
| 16 | G | **on** | 4.874e-02 | 3.106e-02 | **1.57** |
| 32 | C | off | — | 3.97e-05 | — |
| 32 | G | off | 4.408e-03 | 2.624e-03 | 1.68 |
| 32 | G | **on** | 1.774e-02 | 4.344e-03 | **4.08** |
| 64 | C | off | — | 2.51e-06 | — |
| 64 | G | off | 1.129e-03 | 5.140e-04 | 2.20 |
| 64 | G | **on** | 7.881e-03 | 8.536e-04 | **9.23** |

The interface/bulk ratio **doubles per refinement** with advection on — the signature of one order
lost — and the convective term is 4× the rest of it at N = 32.

**(b) The convective operator alone** (`--flux`): a numpy replica of `flow.hpp::advectExplicit`
built from `Flow.diagnostics.face_topology()`, with the advecting velocity taken **exact at the
sub-face centroid** so only the reconstruction of the advected value is measured, against the
analytic `div(u u_x) = u·∇u_x = (k/2) sin 2kx`. RMS of the face-value error `|φ_face − φ_exact|`
at the sub-face centroid, and of the resulting cell divergence error:

| N | div rms @ if | div rms bulk | face err, regular | face err, C/F **upwind = fine** | face err, C/F **upwind = coarse** |
|---|---|---|---|---|---|
| 16 | 5.395e-02 | 2.428e-02 | 5.224e-02 | 2.643e-02 | **1.610e-01** |
| 32 | 1.781e-02 | 4.082e-03 | 1.912e-02 | 7.284e-03 | **5.208e-02** |
| 64 | 7.499e-03 | 1.016e-03 | 5.944e-03 | 1.789e-03 | **2.407e-02** |

**The loss is entirely on the half of the sub-faces whose upwind cell is the COARSE one.** Where
the upwind is the fine cell the face value tracks — in fact beats — the regular faces, and converges
at the same rate (N-orders 1.86, 2.03 against the regular faces' 1.45, 1.69). Where the upwind is
coarse the error is 3–4× the regular faces' and converges at N-order 1.63 then 1.11 against the
regular faces' ~2.

(Refinement here is at fixed cell size with the box and the wavelength doubling, so a p-th order
scheme reads N-order p+1. The regular faces read ~2–3, the coarse-upwind C/F faces ~1–1.6.)

**Census** (N = 32; the same at 16 and 64): 5376 C/F sub-face slots, the upwind is the coarse cell
on exactly half, the stencil **never** crosses a level — `upup` is always at `up`'s own level — and
there is **never** a missing upwind probe. So the (1.5, −0.5) weights are not being applied across a
level jump *at the sub-face*; that hypothesis is dead.

## 3. Current state, with the code

`include/peclet/amr/flow.hpp`. Three kernels build the convective flux; all three walk the face CSR
with **no 2:1 case at all**. `advectExplicit` is the shipped high-order flux (`deferredSou` is the
same expression minus its FOU part; `buildFou` is the implicit first-order part):

```cpp
for (Index k = st(i); k < st(i + 1); ++k) {
  const Index j = nb(k);
  if (!fl(j)) continue;
  const int a = ax(k);
  const double uai = (a == 0) ? u0(i) : (a == 1) ? u1(i) : u2(i);
  const double uaj = (a == 0) ? u0(j) : (a == 1) ? u1(j) : u2(j);
  const double velOut = useFace ? dr(k) * uf(k) : dr(k) * 0.5 * (uai + uaj);
  const Index up   = (velOut > 0.0) ? i : j;
  const Index down = (velOut > 0.0) ? j : i;
  const Index upup = (velOut > 0.0) ? uiP(k) : ujP(k);
  const double phiUp    = fld(up);
  const double phiUpUp  = (upup >= 0 && fl(upup)) ? fld(upup) : phiUp;
  const double phiDown  = fld(down);
  const double phiFace  = hoFaceValue(phiUpUp, phiUp, phiDown, advScheme);
  sou += ra(k) * velOut * phiFace;
}
defc(i) = rho * sou * iv(i);
```

`include/peclet/amr/advect_recon.hpp`:

```cpp
MORTON_HD inline double hoFaceValue(double upup, double up, double down, int scheme) {
  if (scheme == 0) return 1.5 * up - 0.5 * upup;   // SOU, the default
  const double den = down - up;                    // Koren TVD below
  const double aden = (den < 0.0) ? -den : den;
  const double r = (aden < 1e-10) ? 0.0 : (up - upup) / den;
  const double t = (1.0 + 2.0 * r) / 3.0;
  double m = (2.0 * r < t) ? (2.0 * r) : t;
  if (m > 2.0) m = 2.0;
  const double psi = (m > 0.0) ? m : 0.0;
  return up + 0.5 * psi * den;
}
```

**The geometry.** `FaceGeom` (`include/peclet/amr/face_geom.hpp`) is built by
`facegeom_assembly.hpp::forEachFaceFull`. At a 2:1 face the COARSE cell's row carries **four**
entries, one per fine sub-face; each stores the **fine** area in `rawArea` (so the four sum to the
coarse area), `dist = 1.5·h_fine`, and the fine cell's openness. `upupI(k) = periodicNeighbor(i,
axis, −dir)` and `upupJ(k) = periodicNeighbor(j, axis, +dir)` — **both depend only on the cell, not
on which sub-face `k` is**, so all four sub-faces of a coarse face share one `up` and one `upup`
when the upwind is the coarse cell, and therefore one `φ_face`. The only thing that distinguishes
them is `uf(k)`.

**What the C/F machinery already provides.** `include/peclet/amr/cf_scheme.hpp` builds four deferred
corrections as CSR overlays — `buildCfLapDelta` (momentum diffusion), `buildCfDivDelta` (the
divergence constraint), `buildCfGradDelta` (the ABC cell gradient) and `buildCfUfDelta` (the
advecting face field `uf`). All four are **linear** operator deltas. The convective flux is
`uf · φ_face`, bilinear in the state, so none of this machinery expresses it, and none of it is
applied to the advected value today.

`include/peclet/amr/poisson.hpp:389` has the tangential interpolation the other four use:

```cpp
double coarseStar(const std::vector<double>& u, Index coarse, Index fine, int axis) const {
  const double uc = u[coarse];
  double val = uc;
  for (int t = 0; t < Dim; ++t) {
    if (t == axis) continue;
    const double H  = cellWidth(coarse, t);
    const double dt = ((bf[t] + 0.5*sf) - (bc[t] + 0.5*sc)) * h0_[t];
    Index cp = periodicNeighbor(coarse, t, +1), cm = periodicNeighbor(coarse, t, -1);
    if (cp < 0 || cm < 0) continue;
    if (levelOf(cp) != levelOf(coarse) || levelOf(cm) != levelOf(coarse)) continue;
    if (faceOpenness(coarse, t, +1) < 0.5 || faceOpenness(coarse, t, -1) < 0.5) continue;
    const double Dt  = (u[cp] - u[cm]) / (2.0 * H);
    const double Dtt = (u[cp] - 2.0*uc + u[cm]) / (H * H);
    val += dt * Dt + 0.5 * dt * dt * Dtt;
  }
  return val;
}
```

It is a HOST routine on `std::vector<double>`; the device path reaches the same values through the
prebuilt CSR overlays, not by calling it.

## 4. Two prototypes, both measured, neither sufficient

`tests/study/amr_cf_advection.py --flux` runs both (`--mode star` / `dist` / `both`).

* **`star`** — apply `coarseStar` to every stencil value whose cell is coarser than the sub-face
  (both `φ_up` and `φ_upup`), so the coarse column is sampled at the sub-face's tangential position.
* **`dist`** — keep the two stencil points but extrapolate with their **actual** distances to the
  sub-face centroid, `φ_up + (φ_up − φ_upup)·d₁/(d₂ − d₁)`, which is (1.5, −0.5) only when the pair
  is equally spaced.

| N | mode | div rms @ if | div rms bulk | face err C/F | face err regular |
|---|---|---|---|---|---|
| 32 | none | 1.781e-02 | 4.082e-03 | 3.719e-02 | 1.912e-02 |
| 32 | star | 1.452e-02 | 4.082e-03 | **4.173e-02** | 1.912e-02 |
| 32 | dist | 1.463e-02 | 3.262e-03 | 3.719e-02 | 1.754e-02 |
| 32 | both | **1.135e-02** | **3.262e-03** | 4.173e-02 | 1.754e-02 |
| 64 | none | 7.499e-03 | 1.016e-03 | 1.707e-02 | 5.944e-03 |
| 64 | both | **5.827e-03** | **8.124e-04** | 1.768e-02 | 5.349e-03 |

Together they cut the interface divergence error **22–36 %** and the bulk **13–20 %**, and **do not
restore the order** (the interface/bulk ratio still grows).

Two results worth your attention:

1. **`star` makes each C/F face VALUE worse at N ≥ 32 (3.72e-2 → 4.17e-2) and yet makes the
   interface DIVERGENCE better (1.78e-2 → 1.45e-2).** It improves the cancellation among the four
   sub-faces of a coarse face rather than the individual values. A face-value error `ε` enters the
   coarse cell's divergence as `ε/h`, so what matters is the *difference* of the errors across
   opposite faces, not their size. Any design that optimises the pointwise face value alone may be
   optimising the wrong functional.
2. **`dist` moves only the REGULAR faces**, because at the sub-face itself `up` and `upup` are at
   the same level (the census). The faces it fixes are the same-level faces **next to** the
   interface, whose `upup` two cells away is at the other level. That is a second, independent
   defect, cheap to fix, and it is not on any list.

## 5. Constraints and invariants

- **Conservation is not at risk and must not become so.** The flux is one value per sub-face shared
  by both incident cells, so refluxing is automatic (`coarseStar`'s own docstring makes the same
  point for the Laplacian). Any new stencil must keep `φ_face` a function of `(sub-face, upwind
  direction)` alone, identical from both sides. `div(uf) ≤ 4e-12` on these meshes must not move.
- **Device and distributed.** Header-only C++20 + Kokkos; the kernels above are `KOKKOS_LAMBDA` over
  the face CSR. Anything the kernel needs must be prebuilt into device Views. Under MPI the ±2 leaf
  ghost registry supplies neighbours; **`isCut` is NOT ghost-visible** (`cut_` is sized `n`, not
  `n + nghost`) — a rule that asks whether a *neighbour* is cut needs that flag communicated first.
  Acceptance is np = 1 bitwise vs single-rank, np = 2/4/8 in the ~3e-7 march class.
- **The byte gate** (`python/state_hash.py`, 13 scenario keys, np = 1 and `mpirun -np 2`) changes if
  the numerics change; that is allowed for a deliberate accuracy change, re-recorded in its own
  commit. Any *off* switch must be bit-identical.
- **Stability.** The throat-graded meshes are the hazard: a looser C/F rule once marched 2 of 12 of
  them to k ~ 1e12 by step ~100 (`tests/study/amr_two_sphere_gap.py`,
  `amr_two_sphere_diverge_probe.py`). Large dt is normal here — the steady drag runs use
  dt = 60 … 1e20 — and the implicit-FOU part is what carries them.
- **Cut cells.** The per-FACE gate of `docs/amr_cf_flux_gate.md` withholds the quadratic face value
  where a 2:1 sub-face has a cut cell on either side (counted by
  `Flow.diagnostics.num_cf_cut_faces`). Say whether the convective fix takes the same gate.
- **Koren must stay TVD.** The default is SOU; Koren is the limiter option and is what a bed run
  would use.
- **Performance.** §C1 of the ROADMAP already has the advective step at 4–7× `flow`'s. A fix that
  makes it slower needs to say by how much.

## 6. Already decided — do not reopen

From `../../docs/decisions/amr.md`:

- **The collocated coupling is the Almgren–Bell–Colella approximate projection, never Rhie–Chow.**
- **`cf = 1` (quadratic C/F) is not optional on graded meshes** and is the shipped default.
- **The C/F pressure matrix / MG / PCG stays standard order**; a quadratic face gradient is obtained
  by deferred correction on the right-hand side, never by changing the matrix.
- **The C/F flux correction is a property of the FACE, not of a cell's row** — one predicate
  (`regular(i) && regular(j)`) drives the divergence, the gradient substitution and `uf` through one
  emitter.
- **Inner predictor→projection iterations are declined**; the deferred-corrected C/F *pressure*
  gradient (option B) is designed and parked (`amr_pressure_iteration.md` §14). Neither is in scope
  here — this is the convective term, which those measurements explicitly exonerated them of.

**Genuinely open — what to decide:** the sub-face stencil for the ADVECTED value (and whether the
near-interface regular faces' distance weighting ships with it); where it lives (a prebuilt device
View of per-sub-face stencil data, versus an inline computation, versus something that reuses the
existing overlay machinery — note the flux is bilinear, so a linear CSR delta does not express it);
and whether the measured gain justifies the cost.

## 7. Already tried and rejected, with evidence — do not re-walk

1. **`coarseStar` on the stencil values alone** — 18 % on the interface divergence, and it makes the
   pointwise face values *worse* at N ≥ 32 (§4).
2. **Distance-weighted extrapolation alone** — 18 % on the interface divergence, 20 % on the bulk;
   inert at the sub-face itself (§4).
3. **Both together** — 36 % at N = 32, 22 % at N = 64; order not restored (§4).
4. **The "SOU weights are applied across a level jump" hypothesis** — refuted by the census: `upup`
   is at `up`'s own level on every C/F sub-face at N = 16, 32 and 64, and no probe is missing.
5. **The limiter, the time treatment, the advecting velocity and the C/F interpolation order** — all
   ablated in `amr_pressure_iteration.md` §14.4c; the shape excess sits at 2.15–2.57e-2 through all
   of them and drops to 2.65e-3 only with advection off.

## 8. How the answer will be verified

- `tests/study/amr_cf_advection.py --flux` — the interface/bulk divergence ratio must stop growing
  with N (today 2.22 → 4.36 → 7.38), and the coarse-upwind face error must converge at the regular
  faces' rate.
- `tests/study/amr_cf_advection.py --step` — the one-step interface/bulk ratio with advection on
  (today 1.57 → 4.08 → 9.23).
- `tests/study/amr_tg_graded.py` — the headline: the shape error of (G) against (C) at N = 64,
  CFL 0.5, today 7.417e-03 vs 2.265e-03 (3.27×), and `m3` 1.4328e-02 vs 9.2112e-03 (1.56×). **State
  what these should become** — that number is the reason the item exists.
- `python_amr_tg_graded` (the ctest gate), the 83/83 + 17/17 battery, the byte gate at np = 1 and 2.
- Stability: `amr_two_sphere_gap.py`, the 12 throat meshes, cf = 1, 400 steps, all finite, policy
  error within ±0.3 %.
- Cost: `bench_amr_flow` and `PECLET_AMR_PROFILE_STEP=1`.

## 9. Deliverable

`docs/amr_cf_convective.md`:

1. **The decision**, in one sentence, including "not worth building" if that is the answer.
2. **What the literature does** for the advective flux at a refinement boundary in a
   cell-centred composite-grid method — Berger–Colella, Martin & Colella's cell-centred AMR
   projection, Martin, Colella & Graves (2008), the Chombo/EB line — and whether this repo should
   follow or deviate. Refluxing is automatic here (single-valued flux); what is at issue is the
   *reconstruction*, which is the ghost-interpolation half of those schemes.
3. **The options considered**, each with the reason it was or was not chosen — including
   "improve the pointwise face value" versus "make the four sub-face errors cancel", which §4's
   first surprise says are different objectives.
4. **The stencil**, to implementation detail: what is sampled, with what weights, what geometry has
   to be prebuilt, and what happens at a cut cell, at a block seam, and when a tangential neighbour
   is missing.
5. **Why it cannot destabilise** the throat meshes, argued from the mechanism.
6. **Work orders** for an Opus implementer, and the gates of §8 with expected numbers.
7. **The cost**: per-step arithmetic, memory, and any new MPI communication.

No production code from this pass.

## 10. Out of scope

- The pressure-increment leak and options (A)/(B) — measured, parked, exonerated.
- `peclet.flow`; the uniform-grid parity work; the cut-cell items A2/A3.
- The *amplitude* half of the (G)−(C) gap — the C/F fluxes' numerical viscosity, which already
  responds to the limiter and the C/F scheme (`amr_pressure_iteration.md` §14.4c). This brief is
  about the **shape** half.
- Scalar transport (`scalar_transport.hpp`) — it shares the reconstruction and would inherit any
  fix, but its own gates are not in scope.
