# Inner pressure iterations (A) and the deferred-corrected C/F pressure gradient (B): analysis, a design for (B), and why neither is built today

> Design note, 2026-09-22, against `peclet-amr` `756dfcf` (main; the brief names `2426ef4`, since
> superseded by the merged C/F flux-gate note). Brief: `docs/briefs/pressure_iteration_options.md`.
> No production code was written for this note. Prior art it builds on:
> `docs/amr_cf_flux_gate.md` (rule (I), the O(1) tangential leak, §6.8's sketch of (B)),
> `docs/amr_collocated_projection.md` (the projection, `uf`, the (1,2) placement),
> `../flow/doc/collocated_invisible_subspace.md` (the attractor family, Props. 1–2).

## 1. The recommendation

**Build neither option now. Build the graded time-accurate benchmark (WO0), which the project
lacks for every purpose, and let it decide (B). (A) is declined on first-principles grounds and the
register entry "project ONCE per step" stands; (B) is fully designed below (§6–§8) so that it is a
bounded implementation job the day a consumer appears, and it does NOT reverse "pressure matrix
stays standard".**

The load-bearing finding is a scaling that the brief did not have. In the incremental rotational
scheme the projection potential is the *increment* of the pressure,
`(ρ/dt)·φ ≈ p^{n+1} − p^n ≈ dt·∂_t p`, so `φ ≈ dt²·∂_t p/ρ`, and the C/F face-value error the
brief wants removed is

    ε_cf = (2/3)·|∇_t φ| ≈ (2/3)·(dt²/ρ)·|∇_t ∂_t p|                            … (S)

— **second order in dt at fixed h**, while the backward-Euler time error of the same solver is
`dt·|∂_t u|` and the spatial error `O(h²)`. Under CFL-limited refinement (`dt ∝ h`) the "O(1) in h"
leak is `O(h²)`: one second-order term among the others, not a floor. Worked for the Taylor–Green
case of §9 (N = 32, ν = 0.05, CFL 0.5): `ε_cf ≈ 1.3e-4` against a BE error of `≈ 1.9e-3` and a
spatial error of `≈ 4e-2` — fifteen and three hundred times below what is already there. `ε_cf`
reaches the BE error only at `dt ≈ 1.5·|∂_t u|/|∇_t ∂_t p| = 1.5/k` for the Taylor–Green (ν cancels;
CFL ≈ 8 at N = 32), i.e. in the *quasi-steady*
regime (`dt = 60 … 1e20` cell units) whose transient nobody reads and where the incremental
potential is not small. That is exactly the regime of the numbers in the brief's §2 (0.240 at step 1
of an impulsive start at dt = 60): real, large, and physically uninteresting, because at that dt the
whole transient is unresolved.

So the defect is real, bounded, proportional to the per-step pressure change, and invisible in any
run that resolves its own transient. It matters only to a consumer of the face field during a
large-dt transient with genuinely evolving pressure — moving geometry (resolved CFD-DEM on the
octree) or time-dependent forcing on a graded mesh at `dt ≫` CFL. **No such consumer exists in the
suite today** (`coupling` does not run on `amr`). Building an option for it now is speculative work;
what is not speculative is that the project has no graded unsteady validation at all
(`ROADMAP.md`: "no convergence study of the graded solver against an analytic solution"). WO0
closes that gap and, as a by-product, measures (S) — which is a falsifiable prediction: if the
benchmark finds `ε_cf` at or above the global velocity error at CFL ≤ 2, this note is wrong and (B)
should be built from §6–§8 without further design.

The decision line for the log:

    DECISION: inner pressure iterations (A) / deferred-corrected C/F gradient (B) as options?
      → neither now; benchmark WO0 first; (B) designed and gated on WO0's criterion (§10 gate W);
        (A) declined, register entry stands. Alternative: build (B) now as a research knob (WO1–WO6
        stand alone, ~3 Opus-days). Reversible by: running WO1–WO6 — nothing in WO0 is undone.

## 2. Problem and scope

**Problem.** On a graded mesh the advecting face field is `uf = F u* − G_f φ`, with `F` the face
average (standard ½/½ plus the velocity-only quadratic C/F delta) and `G_f` the compact two-point
face gradient `(φ₊ − φ₋)/d`. The pressure matrix is `L = D_std·G_f`, so `D_std uf = rhs − Lφ` = the
solver residual (rule (I) of the gate note): `uf` is a conservative flux. On a 2:1 sub-face the two
centres of `G_f` are offset tangentially by `h/2` per axis over a normal distance `1.5h`, so
`(φ_C − φ_F)/d = ∂_nφ ± ⅓∂_{t1}φ ± ⅓∂_{t2}φ + O(h)`: the face VALUE carries an `O(1)·|∇_tφ|`
tangential leak that does not shrink with h at fixed dt (measured 4.547 → 4.319 → 4.224 at
N = 16/32/64 on `test_amr_cf_vector` §5, converging to `(2/3)·2π`). The accurate gradient
`G_q = G_f + Δ_G` (Martin–Cartwright `coarse*` substitution) cannot be put into `uf` while the
matrix inverts `G_f` — that was the `uffix` bug (a 4.0e-2 mass source for the whole transient).

**The question.** Whether to offer (A) inner predictor→projection iterations within a step (φ → 0
at convergence, so the gradient term in `uf` vanishes) and/or (B) a deferred correction
`L φ_{k+1} = rhs − (L_q − L) φ_k` whose fixed point inverts `L_q = D_std·G_q`, both default-off,
bit-identical and allocation-free when off; and whether either is worth having.

**In scope.** The analysis of both (§3–§5); the design of (B) to implementation detail (§6–§8); the
benchmark (§9); gates (§10); register consequences (§11). **Out of scope.** Building (A) (declined,
§4); a second-order time integrator (the correct instrument for time accuracy, §4.4 — a separate
design); the momentum operator's own lagged C/F deferred correction `cfMom_` (its lag is `O(dt)`
like BE and is already iterated by the Picard loop); `peclet.flow`; the O(h) normal-offset term of
`G_q` at C/F faces (§5.4 — needs a three-point normal stencil, a different scheme).

## 3. Constraints and invariants

- **Off ⇒ bit-identical.** All thirteen `python/state_hash.py` keys unchanged at np = 1 and under
  `mpirun -np 2` (four scenarios carry `.np1`/`.np2` keys); the battery green; no measurable
  step-time change. Mechanism: no kernel changes, no arithmetic change on the default path, one
  host-side branch (§7.1).
- **Off ⇒ allocation-free.** No CSR built, no View allocated, no collective added unless the option
  is on at `setSolid` time (§7.1).
- **Rule (I) holds at every outer count.** `D_std uf = ` the inner solver residual, on every mesh,
  for every k — not only at outer convergence (§6.2; this is stronger than the gate note's §6.8
  sketch and is the reason (B) can be shipped with a finite k).
- **The pressure matrix, the MG hierarchy (host / device / distributed), MG-PCG and the ghost
  BiCGStab are untouched.** `L_q` appears only through a right-hand-side CSR (§6.1).
- **The steady state is untouched.** At the fixed point `φ → 0`, so `(L_q − L)φ → 0` and
  `Δ_G φ → 0`: steady permeability, the ladder orders and the C2 dt-battery agree with the option
  off to the march's stationarity tolerance (not bitwise — the trajectory differs).
- **Support consistency at cut rows.** `Δ_G` is emitted only on `cfFace(i,j) = regular(i) &&
  regular(j)` sub-faces (the gate note's rule); cut rows and their faces receive nothing, so the
  ghost closure's constraint/gradient pairing is not touched — the mechanism of the 2026-08-27
  throat instability cannot be re-entered by this option.
- Conventions: the suite operator is `+L` (negative semi-definite); `dir_k = +1` on the + side of
  the owning cell; `(G_f φ)_k = (φ₊ − φ₋)/d_k` with the + side `= j` iff `dir_k > 0`; `dist` on
  the FACE axis (anisotropic finding of the gate note §6.2); SDF negative in solid.
- np = 1 bitwise vs single-rank; np = 2/4/8 in the seam test's ~3e-7 class; the oracle
  (`flow_oracle.hpp`) mirrors the device through the same host builders.

## 4. Option (A): inner pressure-correction iterations — analysed and declined

### 4.1 The iteration and its fixed point

One step with `uⁿ` frozen, exact linear solves, `N := −L` (SPD on the mean-free fluid range),
`A := (ρ/dt)I − μL_u (+ adv)`, `B := (ρ/dt)I − μL` (SPD), `G_c` the ABC cell gradient
(`grad3` = ½(g⁻+g⁺), plus `cfGrad_` and the ghost gradient):

    u*_k   = A⁻¹(b − G_c p_k)
    φ_k    = −N⁻¹ D F u*_k
    p_{k+1} = p_k + B φ_k                      (rotational update, div = D F u*_k)

A fixed point has `Bφ = 0 ⇒ φ = 0` (B is SPD), hence `D F u* = 0`, `u = u*`, `uf = F u`: the
brief's "trade dissolved" — **when the iteration converges to it.** The pressure-error map is
`e_{k+1} = (I − S) e_k` with `S = B N⁻¹ (−D F) A⁻¹ G_c`.

### 4.2 Convergence condition (question 2 of the brief)

**Bulk (uniform, periodic, no solid): no under-relaxation, at any dt.** There `D F = −G_cᵀ` in the
volume inner product (the pair is adjoint), and per Fourier mode with `ℓ = Σ_a 4 sin²(θ_a/2)/h²`
(compact Laplacian) and `w = Σ_a sin²θ_a/h²` (wide Laplacian):

    B = ρ/dt + μℓ,   N = ℓ,   G_cᵀA⁻¹G_c = w/(ρ/dt + μℓ)   ⇒   S = w/ℓ,

so the error factor is `1 − w/ℓ = Σ 4sin⁴(θ_a/2) / Σ 4sin²(θ_a/2) ∈ [0, 1)` for every non-constant
mode, **dt-independent**: the rotational update is exactly the Cahouet–Chabard Schur-complement
preconditioner and it is exact where the operators commute. Smooth modes contract at
`≈ (kh/2)²` per pass (a factor ~100 for `kh = 0.2`); near-grid modes (`θ → π`) at `→ 1`.

**Where the pair is not adjoint the condition is empirical, and it is one the project has already
measured.** At `dt = ∞` the time march *is* iteration (A) (the mass term vanishes; `uⁿ` enters only
the advection lag and the warm start), so every steady-drag march on record is an (A)-convergence
result:

| scheme | non-adjoint rows | (A) at large dt |
|---|---|---|
| ghost projection (2,2), beds | closure rows | converges — unconditionally stable dt 60…1e20, C2 to 2e-6 (flow + `amr_zh_c2.py`) |
| aperture + gauge-exact gradient, beds R ≥ 12 | cut rows | **diverges** at large dt (flow Layer 1: `Re λ(S) < 0` modes); no scalar under-relaxation can fix `Re λ < 0` (`|1 − wλ| > 1 ∀ w > 0`); the wall-banded blend is a resolution-dependent stopgap |
| graded C/F rows, cf = 1 | C/F rows (`⅜` vs `½` fine-side weight — §4.3) | converges on every graded mesh marched at dt = 60…1e20 (throat ladder, Z&H graded) — measured, not proven |

At finite dt the extra `(ρ/dt)` diagonal makes every case more contractive (the aperture growth
rate scales `∝ νdt/h²`), so **the condition is: (A) converges without relaxation exactly where the
steady march at that dt is stable for that scheme; where it is not, relaxation does not help.**

### 4.3 The asymmetry the brief asked about (question 1) — real, mechanism corrected

The brief's mechanism is that the cell correction's wide gradient annihilates the odd–even mode,
so a checkerboard component of φ survives (A) while `G_f` in `uf` still sees it. **On a uniform
periodic mesh that component is never there.** `rhs = D F u*` is a *wide* divergence whose Fourier
symbol `Σ_a i sinθ_a û_a/h` vanishes on exactly the eight modes `θ_a ∈ {0, π}` that span
`ker G_c`; because the bulk pair is adjoint, `range(D F) ⊥ ker G_c`, and `φ = −N⁻¹ rhs` inherits
that: φ has no invisible component to survive. (A pressure error `e ∈ ker G_c` persists, but it is
invisible to `u*`, `φ`, `u` and `uf` alike — harmless.)

**Where the asymmetry is real is where the pair is support-inconsistent**, which is flow's
Proposition 2 verbatim: (A) converges in `(u, uf, φ)` with `φ_∞ ≠ 0` and `p_k` drifting linearly
along `ker G_c` whenever `B⁻¹ ker G_c` meets the reachable set — the aperture path with cut cells
(`ker G_c ⊇` solid-supported fields, measured `m₁ = 1.8e-2` at every dt, `|P|` grown 75× the
staggered scale). There `uf = F u − G_f φ_∞ ≠ F u` and (A) leaves the C/F leak in place. At C/F
rows themselves the pair is non-adjoint in *weights* only — the adjoint of `D F` at a coarse row
reads the fine side at `⅜·mean(g_f)` where `grad3` uses `½`, and at a fine row `¾·g_cf` vs `½` —
but has no exact invisible mode (generic weights break the ±1 symmetry the checkerboard needs), so
the fixed point is unique there and only the *rate* is at stake (§4.2, measured convergent).

**(B) is immune** because it changes the operator `uf` is built from, not the size of φ:
`D_std uf` stays the inner residual and the face gradient becomes `G_q φ` for whatever `φ` the
step produces, family member or not. Neither option subsumes the other: (A) shrinks φ, (B) fixes
the operator; (A) at convergence would make (B) moot only where the family is absent.

### 4.4 Why (A) is declined even where it converges

1. **Its cost is the momentum solve.** Every pass re-solves the predictor (three MG-BiCGStab
   solves — the dominant cost of the step; `ROADMAP.md` §C1) plus one pressure solve. Two passes
   ≈ 2× the step. (B) costs pressure solves only.
2. **It is not a C/F fix; it is a scheme change.** Iterated pressure correction changes every cell
   of every mesh (the splitting error of the step), which is why the register's measurement found
   a 1.3e-3 move on a Stokes answer that should have been a no-op. Its steady state on the aperture
   path moves too (a different family member is selected). An option that alters results globally
   to cure a codimension-1 defect needs global re-validation; (B) alters nothing at steady state
   and nothing off the C/F faces.
3. **It attacks an error that is already below the time error.** The incremental scheme's
   splitting error is `O(dt²)` in velocity; the integrator is BE, `O(dt)`. Driving the splitting
   error further down with inner passes cannot improve time accuracy; a second-order integrator
   (BDF2 + second-order pressure extrapolation) can, and that is a different design. Converged (A)
   is the exact wide-stencil projection — `D F u = 0` exactly — the very thing the ABC approximate
   projection was introduced to avoid on cell-centred grids (Almgren–Bell–Szymczak 1996; Almgren,
   Bell & Crutchfield 2000), and the project's register already holds "ABC, never Rhie–Chow" on
   the same grounds.
4. **It inherits every instability of the march at that dt** (§4.2), while the option's users
   would be precisely the large-dt runs.

**Register.** "MG-as-solver with a Picard outer loop projects ONCE per step, never inside the
loop" **stands**; this note adds a clarifying line (§11) so the deliberate variant is not
re-proposed: its rejection is not the accidental coupling the entry records but the cost and the
scheme change above. That entry is *a* reason, not *the* reason, for declining (A).

## 5. Option (B): deferred-corrected C/F pressure gradient — analysis

### 5.1 The operators

`G_q = G_f + Δ_G` with `Δ_G` nonzero only on `cfFace` 2:1 sub-faces:

    (Δ_G φ)_k = σ_k · (φ_C* − φ_C) / d_k,     σ_k = +1 iff the coarse cell is on the + side,

`φ_C*` the scheme's tangential interpolation of the coarse value at the fine cell's tangential
position (`detail::cfAppendStencil`, the same `coarse*` every operator delta uses). Then
`L_q = D_std G_q = L + D_std Δ_G`, non-symmetric, with the constant in its kernel
(`Δ_G(const) = 0`) and a compatible range (both incident slots of a sub-face emit the identical
value, so `Σ_i V_i (D_stdΔ_G φ)_i = 0` by telescoping — the fluid-mean deflation of the PCG is
not disturbed).

### 5.2 The outer iteration and the key identity

    L φ_{k+1} = rhs − (L_q − L) φ_k,        φ_0 = 0   (so φ_1 is today's φ)
    uf_k     := F u* − G_f φ_{k+1} − Δ_G φ_k

    ⇒ D_std uf_k = rhs − L φ_{k+1} − D_stdΔ_G φ_k = r_{k+1}   (the inner solver residual)   … (I′)

**Rule (I) holds at every k**, because `uf` carries exactly the two gradient terms the k-th solve
balanced: the matrix's `G_f` on the new iterate and the lagged `Δ_G` on the old one. Conservation
is therefore not a property of outer convergence at all; only the *accuracy of the face value* is.
The gate note's §6.8, which assumed `uf = F u* − G_q φ` and conservation "at convergence of the
outer loop", is superseded by (I′): the lagged form is strictly better and costs nothing extra.

### 5.3 Contraction (question 3 of the brief)

With `L_q φ_∞ = rhs` and `e_k = φ_k − φ_∞`: `e_{k+1} = T e_k`, `T = N⁻¹ D_std Δ_G`,
`e_1 = −Tφ_∞`, and the face-gradient error of `uf_k` at a C/F sub-face is

    G_f e_{k+1} + Δ_G e_k = −G_f T^{k+1} φ_∞ − Δ_G T^k φ_∞.

At `k = 0` the second term is today's `O(1)` leak; each outer solve multiplies both by `T`.

**T contracts in the energy norm of the standard operator, with a LOCAL constant** — a norm bound,
not merely a spectral-radius bound, so no under-relaxation is needed and the iteration is
monotone. With `‖ψ‖²_N := ⟨ψ, Nψ⟩_V = Σ_faces w_k (G_fψ)_k²`, `w_k = α_k A_k d_k`, and the adjoint
identity `⟨D v, ψ⟩_V = −Σ_k w_k v_k (G_fψ)_k`:

    ‖Tφ‖²_N = ⟨Tφ, D_stdΔ_G φ⟩_V = −Σ_{k∈CF} w_k (Δ_Gφ)_k (G_f Tφ)_k ≤ ‖Δ_Gφ‖_{w,CF} · ‖Tφ‖_N
    ⇒ ‖Tφ‖_N ≤ ‖Δ_Gφ‖_{w,CF} ≤ κ ‖φ‖_N .

`κ` is local because every quantity `Δ_G` reads is a difference across an open face that
`‖φ‖_N` already contains with comparable weight (the tangential samples are gated to fluid
same-level neighbours across faces of openness ≥ 0.5; the finer-cover branch reads the four
C/F sub-face differences plus intra-fine ones — all open faces). For the shipped quadratic
stencil at tangential offset `±H/4`: per tangential axis `s_t = ±⅛(a_t + b_t) + (1/32)(a_t − b_t)`
with `a_t, b_t` the coarse cell's two tangential face differences, so `|s_t| ≤ (5/32)(|a_t|+|b_t|)`;
`(Δ_Gφ)_k = (s_1 + s_2)/(0.75H)`, `w_k = (3/16)H³`, hence `w_k(Δ_Gφ)_k² = (1/3)H(s_1+s_2)²` and the
four sub-faces of one coarse side sum to `≤ 0.13·H·Σ_t(a_t² + b_t²)`, against a face energy of
`H·Σ_t(a_t² + b_t²)` for those four tangential faces. Each tangential face is read by at most two
coarse cells on a planar interface:

- **planar interface, any φ:** `κ ≤ ≈ 0.5` (crude, from `0.26`);
- **smooth φ (`a_t ≈ b_t`), production meshes:** `κ² ≈ (1/6)·(fraction of the gradient energy
  within one coarse cell of the C/F surface)·(|∂_tφ|²/|∇φ|²)`, i.e. `κ ≈ 0.1–0.25` for a shell of
  three to eight fine cells;
- **island edges/corners** (a coarse cell with C/F sides on two or three axes, reading shared
  tangential faces): the crude bound rises to `≈ 0.9`. Loose, and the reason gate K measures κ.

**Deeply graded beds (the brief's question).** `κ` is a stencil constant; it does not grow with the
number of C/F faces. A large C/F surface enlarges the *set of faces being corrected* and the size
of `‖e_1‖`, not the contraction per outer solve. What a bed does is push κ toward its worst local
value (many island corners), so the count may rise from 1 to 2–3 — never toward non-convergence
while `κ < 1`. For the ghost matrix `ρ(L_bin + Δ_ghost)` the SPD argument holds only outside the
cut band (where `(L_q − L)` lives, by `cfFace`); κ there is measured, not proven (gate K, both
drivers).

**Iteration count.** Face-gradient error `≈ κ^k ×` leak: with `κ ≈ 0.2`, one correction (two
pressure solves per step) leaves 20 % of the leak, two leave 4 %. The loop is self-terminating
(§6.3): it stops when the correction's right-hand side is below the inner solve's own accuracy.
Default cap 2.

### 5.4 What (B) buys and what it does not

After outer convergence `uf = F u* − G_q φ`, which at a C/F sub-face equals the exact face-normal
velocity up to (a) the velocity-average part's `O(h²)` (`cfUfVel_`, order 1.95 measured) and (b)
the **`O(h)·|∂_{nn}φ|` normal-offset** of `G_q` (its two samples sit at `h/2` and `h` from the
face, the quotient is centred `h/4` into the coarse side). So the §5 order test of
`test_amr_cf_vector` should read `≈ 1` for the whole `uf` under (B) — the brief is right to expect
`≈ 1`, not 2 — and the residual `O(h)` term is `∝ φ ∝ dt²` like the leak was. Removing it needs a
three-point normal stencil in `G_q`; not this design.

## 6. The design of (B)

### 6.1 Two CSRs from one emitter

In `cf_scheme.hpp`, `detail::cfAppendFaceGradDelta(ap, out, coarse, fine, axis, dist, sideSign,
scale, fluidOk, scheme, proto)` appends, for one directed `cfFace` sub-face,
`cfAppendStencil(ap, out, coarse, fine, axis, −scale·sideSign/dist, fluidOk, scheme, proto)` —
i.e. the entries of `−σ_k(φ_C* − φ_C)/d_k`, the sign that puts `−Δ_Gφ` into `uf`. Two builders call
it with the gate note's per-face predicate `cfFace(i,j) = regular(i) && regular(j)`:

- `buildCfUfPhiDelta(ap, t, regularOk, fluidOk, scheme) → CfCsr` — one row per `forEachFaceFull`
  slot (the deleted `cfUfPhi_`, resurrected under the face gate), `scale = 1`. Row value on slot k:
  `(Δuf_φ)_k = −(Δ_Gφ)_k`.
- `buildCfPresRhsDelta(ap, t, regularOk, fluidOk, scheme) → CfCsr` — one row per CELL (regular
  rows only, early-out as `buildCfDivDelta`), `scale = invV_i·α_k·A_k·dir_k`. Row value:
  `(R φ)_i = Σ_{k∈CF faces of i} invV·αA·dir·(Δuf_φ)_k = −((L_q − L)φ)_i`.

So `R = −(L_q − L)` **is** `D_std` of the slot CSR, entry by entry (same stencil, the divergence
coefficient folded in, today's association order), and the right-hand side of the k-th solve is
`rhs + R φ_k` with no sign juggling. `dist` is the face-axis distance from `forEachFaceFull`
(anisotropic-safe). Both builders return empty CSRs when `scheme == standard` or no face passes.

### 6.2 State and the projection loop (`flow.hpp`)

New members: `int cfPresCorr_ = 0` (outer solves cap, default off), `bool cfPresBuilt_ = false`,
`int lastCfPresCorr_ = 0`, `CfCsrDev cfPresRhs_`, `CfCsrDev cfUfPhi_`, `View<double> phiLag_,
dphi_, rhsCorr_` — the three Views sized `nExt_` and allocated in `setSolid` **only when**
`cfPresCorr_ > 0` (the CSRs are built there under the same condition, and only if
`cfScheme_ != standard`; `cfPresBuilt_ = (cfPresCorr_ > 0)`). Setter
`setCfPressureCorrections(int n)` (`n ≤ 0` ⇒ off), to be called before `setSolid` like
`setCfScheme`; `project()` throws `std::runtime_error` if `cfPresCorr_ > 0 && !cfPresBuilt_`
("call setCfPressureCorrections before setSolid").

`project()` after the existing first solve (unchanged: `deep_copy(phi_, 0)`, PCG / BiCGStab /
V-cycles into `phi_`, `lastPresIters_`):

    lastCfPresCorr_ = 0
    if (cfPresCorr_ > 0):
        ρ₁ = sqrt(vdot(div_, div_))              # the first solve's initial residual scale (one reduction)
        deep_copy(phiLag_, 0)                    # φ_0
        for k in 1..cfPresCorr_:
            syncScalar(phi_)                     # ghost tail of φ_k (no-op single-rank / np=1)
            dphi_ = phi_ − phiLag_               # on [0, nExt): both tails current
            rhsCorr_ = 0;  cfApply(cfPresRhs_, dphi_, rhsCorr_)          # R·(φ_k − φ_{k−1})
            ν_k = sqrt(vdot(rhsCorr_, rhsCorr_))                            # one reduction
            if (ν_k ≤ presTol_·ρ₁): break        # the correction is below the inner solve's accuracy
            deep_copy(phiLag_, phi_)             # φ_k becomes the lag
            deep_copy(dphi_, 0)
            solve L·dphi_ = rhsCorr_  with the SAME driver as the first solve (PCG / ghost BiCGStab /
                presIters V-cycles), relative tolerance tol_k = min(1, presTol_·ρ₁/ν_k), cap presIters
            phi_ += dphi_;  lastPresIters_ += iters;  ++lastCfPresCorr_
    finishProjection(n)

The increment form is algebraically the outer iteration of §5.2 (telescoping the solves gives
`L φ_{k+1} = rhs + R φ_k + Σ r_j`); it exists so that every correction solve targets the SAME
absolute residual `presTol_·ρ₁` as the first — the natural stop, and no over-solving of a small
right-hand side to `1e-10` of itself. On every exit path `phi_ = φ_{K+1}` and `phiLag_ = φ_K` for
the last solve actually performed, so (I′) holds whether the loop breaks early or runs to the cap.

`finishProjection` gains one line after `cfApplyComp(cfUfVel_, …)`:
`cfApply(cfUfPhi_, View<const double>(phiLag_), uf_)` — an empty-CSR host early-return when off.
The cell correction uses `G_c φ_{K+1}` (`phi_`, unchanged code), the pressure update uses
`phi_` and the ORIGINAL `div_` (the rotational term is `−μ D F u*`, a property of the constraint,
untouched by a gradient-side change).

**Oracle** (`flow_oracle.hpp::project`): the same loop on host vectors with `presIters` V-cycles
per increment, host CSRs `cfPresRhs_` / `cfUf_.phi`, the same stop rule (host norms), and the same
lag applied in `buildFaceField`. Parity by construction through the shared builders.

### 6.3 Diagnostics

`int lastCfPressureCorrections() const` (0 when off); `lastPresIters()` sums all solves of the step
(unchanged when off). Python: `Flow.set_cf_pressure_corrections(n)` (public tier, beside
`set_cf_scheme` — it is a numerics option) and `Flow.diagnostics.last_cf_pres_corrections`
(developer tier) — spellings per open question 3. Regenerate `packaging/_amr.pyi`.

### 6.4 Combinations (question 5 of the brief)

| with | status | why |
|---|---|---|
| aperture MG-PCG, cf = quadratic | supported (primary) | §5–§6 verbatim |
| bounded V-cycle driver (`setPressurePCG(false)`) | supported | fixed-count increments; no tolerance, so the stop rule on `ν_k` still applies, `tol_k` unused |
| ghost projection, finest band | supported | `(L_q − L)` has no cut rows; BiCGStab on the increment with the same coupled-subspace projection; κ measured (gate K) |
| ghost sampled (mixed-level band) | supported | cut-adjacent C/F faces are withheld by `cfFace`; the LS closures are not read; gate S re-run |
| MPI | supported | +1 `syncScalar` and +1 all-reduce per outer solve; the CSR reads are a subset of the reach `probeCfScheme` already registers (the `coarse*` tangential probes) — no discovery change |
| Picard `outerIters_ > 1` | supported, orthogonal | `project()` runs once after the Picard loop; two counters, two stop rules |
| `ufAdvect_ = false` (ablation) | supported | `uf` is built and unused |
| `cfScheme_ == standard` | **inert** (empty CSRs, loop breaks at k = 1 with `ν_1 = 0`) | there is no C/F delta to correct; `last_cf_pres_corrections = 0` says so; same rule as cf = quadratic on a uniform mesh |
| `beginAdapt/finishAdapt`, `rebalanceMpi` | supported | rebuild through `setSolid`; `phiLag_` reallocated with everything else |
| (A) | not built | §4 |

Nothing must throw except the ordering guard of §6.2.

## 7. Zero cost when off (question 4 of the brief)

1. `cfPresCorr_ == 0` ⇒ `setSolid` builds no CSR and allocates no View; the two `CfCsrDev` members
   are default-constructed (`n = 0`), exactly `cfMom_`'s state on a uniform mesh.
2. `project()`: one host `if (cfPresCorr_ > 0)` after the first solve — no kernel, no collective.
3. `finishProjection`: `cfApply(cfUfPhi_, …)` returns at `c.n == 0` before any launch — the same
   host branch `cfApplyComp(cfUfVel_, …)` already takes on every uniform mesh.
4. No arithmetic on the default path changes ⇒ every byte-gate key is unchanged (gate B).
5. When ON but on a mesh with no `cfFace` sub-face, the CSRs are empty, `ν_1 = 0`, the loop breaks
   at k = 1 after one reduction: bit-identical to off (gate B″).

## 8. Work orders (only if gate W of §10 fires, or the user asks for the knob)

Commit-sized, in order, on a topic branch off `main`. Battery as `CLAUDE.md`
(`OMP_NUM_THREADS=2 OMP_PROC_BIND=false`, np8 last); clang-format 18.1.8 on every C++ file touched.

**WO1 — builders (inert).** `cfAppendFaceGradDelta`, `buildCfUfPhiDelta`, `buildCfPresRhsDelta`
in `cf_scheme.hpp` (§6.1). Test in `test_amr_cf_vector`: a new section asserting
`‖D_std(Δuf_φ) − Rφ‖₂ ≤ 1e-14·max(1, ‖Rφ‖₂)` on the graded mesh for a random φ (single-rank host,
the gate note's §6.6 pattern), and `R·1 = 0` to 1e-15. *Acceptance:* battery green; all thirteen
byte-gate keys unchanged at np = 1 and `mpirun -np 2` (nothing calls the builders yet).

**WO2 — the option (device + oracle).** §6.2–§6.3: members, setter, the loop, the `uf` line, the
oracle mirror, the ordering guard, the binding + stub. *Acceptance:* battery green; byte gate
unchanged (option off everywhere); new ctest cases in `test_amr_flow_solver` (or the file that holds
`test_graded_cf_quadratic`): (i) graded sphere, aperture, cf = 1, option on with cap 2, 25 steps at
`presTol 1e-10`: `divNormFace ≤ 1e-9` at every step (gate F″) and `lastCfPressureCorrections ∈
{1, 2}`; (ii) oracle == device on the same case, rel ≤ 1e-10 (the existing parity class); (iii)
ghost projection arm of (i) with `divNormFace` restricted to regular rows (as the gate note's WO3
ghost check); (iv) the ordering guard throws.

**WO3 — the order gate.** Re-add to `test_amr_cf_vector` §5 the whole-`uf` arm with `Δuf_φ`
applied (it existed until the gate note's §6.7 deletion; the CSR is back under the face gate):
assert order ≥ 0.9 at N = 64 (gate O′). *Acceptance:* the printed orders (expected ≈ 1 whole,
≈ 1.95 average) recorded in the commit message.

**WO4 — κ study.** `tests/study/amr_cf_pres_kappa.py` (or C++ under `bench`): power iteration on
`T = N⁻¹ R` — 20 applications of (`cfApply(R)`, pressure solve to 1e-12) to a random mean-free
fluid vector, report `‖Tⁱ⁺¹v‖_N/‖Tⁱv‖_N` — on (a) the Z&H graded sphere N = 32 lmax = 2 band 3,
(b) a throat mesh of `amr_two_sphere_gap.py` (gap 8, n = 3), (c) the byte gate's `r = 0.22`
sphere, aperture and ghost drivers. *Acceptance:* every κ < 1 (gate K); numbers into
`docs/data/` and this note's §5.3.

**WO5 — distributed.** `test_amr_distributed_seam_mpi` and `test_amr_distributed_cf_mpi` gain an
arm with the option on: np = 1 bitwise vs SELF, np = 2/4/8 in the ~3e-7 class, and
`Σ_ranks` of the per-rank `lastCfPressureCorrections` equal across np (it is a collective decision
— the stop rule uses global norms — so the count must be identical on every rank; assert it).
*Acceptance:* those bars.

**WO6 — docs and register.** `docs/amr_collocated_projection.md` (one paragraph: (I′), the lagged
form, the option); the `finishProjection` comment (point at (I′)); `ROADMAP.md`; the register
entries of §11 (the clarifying line under "pressure matrix stays standard" becomes a full entry).
Gates S, M, O, P of the gate note re-run with the option ON (steady numbers must agree with OFF to
the stationarity tolerance; gate S all twelve throat meshes finite to the horizon).

## 9. WO0 — the graded time-accurate benchmark (build this first, in any case)

**Case.** Decaying 2-D Taylor–Green in a periodic box, uniform in z, advection ON (the Stokes
Taylor–Green has `p ≡ 0` and hence `φ ≡ 0` — it cannot exercise the defect; the brief's candidate is
right only with the convective term). Cell units, `spacing = 1`, `ρ = 1`, `μ = 0.05`, `U₀ = 1`,
`k = 2π/N`: `u = U₀ sin kx cos ky·e^{−2νk²t}`, `v = −U₀ cos kx sin ky·e^{−2νk²t}`, `w = 0`,
`p = (ρU₀²/4)(cos 2kx + cos 2ky)·e^{−4νk²t}`, set at cell centres at `t = 0` (`set_velocity`,
`set_pressure`; the parity driver `tests/study/flow_parity/run_amr.py` already does the velocity
half at `init: taylor_green`). No solid: `set_solid(lambda x,y,z: 1e3)` as the parity driver does.

**Meshes.** `Octree(cells=(N,N,N), lmax=1)` with a ball of radius `N/4` at the box centre refined
to `lmax` (`refine_to_sphere(center, radius)` + `balance`): a spherical C/F shell on which the
pressure gradient has tangential components everywhere. Arms: (U) uniform `lmax = 0` at N;
(G) graded, cf = quadratic (today); (G+B) graded with the option on, cap 1 and cap 2 (only if WO2
exists — until then (U) and (G) alone deliver the validation and the decisive metric m1).
N = 32 and 64. Horizon `T = 20` time units (≈ 8 % decay at N = 32).

**dt ladder.** CFL `= U₀dt/h_fine ∈ {0.5, 2, 8, 32}`; the implicit FOU carries large CFL, the
explicit SOU/Koren deferred correction may not — report where it stops.

**Metrics at `t = T`.** m1: `max` over C/F sub-faces of `|uf_k − u_exact·n(sub-face centroid)|`
(exact face-normal velocity at the sub-face centroid — the §5 test's definition); m2: the same over
all faces (bulk reference); m3: volume-weighted L2 of `u − u_exact` at cell centres; m4:
`divergence_norm_face()`; m5: the ratio `m1/m3`. Also the *time-refinement* order of m3 on (U)
(expect 1) and the *h-refinement* order of m3 on (G) at fixed CFL (expect ≈ 2 away from the shell
and the shell's contribution visible in m1's h-scaling: `O(1)` at fixed dt, `O(h²)` at fixed
CFL — the direct test of (S)).

**Prediction (S), falsifiable:** at CFL ≤ 2, `m5 < 0.1` on (G); m1 scales as `dt²` at fixed h and
as `h²` at fixed CFL; m1 reaches m3 only at CFL ≳ 8.

**Worth-it criterion (gate W).** (B) is built if, at the largest dt in the ladder at which arm (U)'s
m3 is still within 2× of its CFL-0.5 value (the largest dt anyone would call time-accurate),
arm (G) reads `m5 ≥ 0.3`. Otherwise the option is parked with this note, and the benchmark stays
as the graded transient validation (its (U) and (G) rows belong on the validation page regardless).

**Second regime (report only).** The graded Z&H sphere impulsive start at `dt = 60`, advection on,
25 steps (the brief's numbers): m1 relative to `|u|` per step. No exact solution and no
time-accurate reference exist at that dt, so no accuracy criterion is possible there — the number is
recorded so the size of the defect in the quasi-steady regime is on file next to the time-accurate
one.

## 10. Verification gates

| gate | what | configuration | criterion |
|---|---|---|---|
| **B** off = bit-identical | byte gate | WO1, WO2, WO3 (option off) | all 13 keys unchanged, np = 1 AND `mpirun -np 2` |
| **B″** on but inert | byte gate with `set_cf_pressure_corrections(2)` on every scenario whose mesh has no `cfFace` sub-face | WO2 | identical hashes to off (the loop breaks at k = 1) |
| **T** battery | ctest | host-openmp, np8 last | green; count = today + the new cases |
| **I″** builder identity | `‖D_std(Δuf_φ) − Rφ‖₂` | graded N = 32 lmax = 2, random φ | ≤ 1e-14·max(1, ‖Rφ‖₂); `‖R·1‖ ≤ 1e-15` |
| **F″** rule (I′) | `divNormFace` every step, option on, cap 1 and 2 | graded sphere, aperture cf = 1, `presTol 1e-10`, 25 steps | ≤ 1e-9 absolute at every step (today's off value 1.3e-12 class; the cap-2 value must not exceed the cap-1 value by more than the extra solve's residual) |
| **F‴** ghost arm | same restricted to regular rows | ghost (2,2), finest band | ≤ 1e-9 |
| **O′** face-value order | `test_amr_cf_vector` §5 whole-`uf` with `Δuf_φ` | N = 16/32/64 | order ≥ 0.9 at N = 64 (≈ 1 expected; today 0.07/0.03) |
| **K** contraction | power iteration on `T` | the three meshes of WO4, both drivers | every κ < 1; report; if any κ > 0.7, raise the default cap to 3 and say so |
| **P′** steady invariance | Z&H graded permeability, ladder orders, C2 dt-battery | option on vs off | agree to the march's stationarity tolerance (≤ 1e-8 relative); orders unchanged to 0.01 |
| **S′** stability | the 12 throat meshes, cf = 1, option on cap 2, 400 steps | `amr_two_sphere_gap.py` | all finite; policy error within ±0.3 % of off |
| **D′** distributed | seam + cf MPI tests, option on | np = 1 / 2, 4, 8 | bitwise / ≤ ~3e-7 class; correction count identical on every rank and across np |
| **W** worth it | benchmark §9 | (U), (G), N = 32/64, CFL ladder | `m5 ≥ 0.3` at the largest time-accurate dt ⇒ build (B); else park |
| **S** scaling check | m1 on (G) | fixed h, dt halving; fixed CFL, h halving | order 2 in both (the prediction; a measured order 1 in dt would mean the leak is `∝ dt`, and this note's §1 argument fails — stop and report) |

## 11. Register consequences

- **"MG-as-solver with a Picard outer loop projects ONCE per step, never inside the loop" —
  stands.** Add a clarifying line, not a supersession: *"The deliberate variant (inner
  predictor→projection iterations as an option) was analysed 2026-09-22
  (`docs/amr_pressure_iteration.md` §4) and declined: it costs a momentum re-solve per pass, it is a
  global scheme change rather than a C/F fix, it cannot improve time accuracy below the BE
  integrator's, and it inherits the aperture path's attractor family and large-dt instability."*
- **"C/F scheme pressure matrix/MG/PCG stays standard order — placement is (1,2)" — stands, and
  (B) does not reverse it.** Confirmed reading: `L_q` enters only as the right-hand-side CSR
  `R = −(L_q − L)`; the matrix, every MG level, PCG and the ghost BiCGStab are untouched, and the
  entry's justification (`φ → 0` at the fixed point ⇒ the matrix's C/F order cannot move the
  steady solution) is exactly why (B) changes nothing at steady state. Add the clarifying line:
  *"A quadratic face gradient in `uf` is obtained, when wanted, by deferred correction on the
  right-hand side with the lagged `Δ_G` carried in `uf` (rule (I′)), never by changing the
  matrix."* This becomes a full entry with WO6 if (B) is built.
- **The ABC approximate projection, never Rhie–Chow** — untouched by either option; §4.4 point 3
  records that converged (A) would be the exact wide projection, which the same entry's reasoning
  rejects.
- No new decision is recorded by this note beyond the DECISION line of §1 (a scope call, logged by
  the caller); (B)'s design is a *default* awaiting gate W, not a settled choice.

## 12. Risks and open questions

Each with a default so work proceeds unattended.

1. **Is the `dt²` scaling (S) right in this solver?** *Needs a fact* — gate S of §10 measures it
   directly (m1 under dt-halving at fixed h). The argument assumes the incremental predictor
   (`−G_c pⁿ` in the momentum RHS, which `step()` has) and a pressure that is a smooth function of
   time. Default: trust (S) until WO0 says otherwise; if m1 measures order 1 in dt, build (B)
   (WO1–WO6) — the case for it is then the brief's, not this note's.
2. **Does a consumer exist that this note does not know of** — a scalar-transport or coupling run
   on a graded octree at `dt ≫` CFL with evolving pressure? *Needs the user's preference / a fact
   about planned work.* Default: none today; the option is one WO-set away when one appears.
3. **Names.** *User's preference / `../docs/NAMING.md`.* Default: C++
   `setCfPressureCorrections(int)` / `lastCfPressureCorrections()`; Python
   `Flow.set_cf_pressure_corrections(n)` (public tier, a numerics option like `set_cf_scheme`) and
   `Flow.diagnostics.last_cf_pres_corrections` (developer tier, the `last_*_iters` family). "Outer
   iterations" is deliberately NOT used — that spelling is the Picard loop's.
4. **Default cap when on.** *Needs a fact* (gate K's κ). Default: 2; raise to 3 if any production
   mesh measures κ > 0.7.
5. **Should the correction solves reuse the first solve's PCG state (warm start on `dphi_` from
   the previous increment)?** *Needs a fact* (iteration counts). Default: cold increment
   (`dphi_ = 0`); the increments are small and the absolute tolerance rule already limits the
   work; measure before optimising.
6. **The ghost-driver κ** has no SPD proof (§5.3). *Needs a fact* (gate K, ghost arm). Default: if
   κ_ghost ≥ 1 on any mesh, restrict the option to the aperture driver and throw on the ghost
   driver with a message naming this note — a one-line guard, reversible.
7. **The O(h) normal-offset term left in `G_q`** (§5.4). *User's preference* on whether a
   three-point normal stencil is ever wanted. Default: no; it is `∝ φ` and vanishes with the same
   `dt²` as the leak.
8. **Benchmark placement.** Default: `tests/study/amr_tg_graded.py` beside the other study
   drivers, registered under the `bench` label if its runtime is under two minutes on host-openmp
   at N = 32; its (U)/(G) rows go to `docs/data/` and, once the validation page exists, to it.

## 13. What the brief got wrong or understated

- **The "checkerboard survives (A)" mechanism (§3.1).** On a uniform periodic mesh φ never has an
  odd–even component: `rhs = D F u*` is a wide divergence and `range(D F) ⊥ ker G_c` because the
  bulk pair is adjoint. The surviving-φ phenomenon is flow's support-inconsistent family at cut
  cells (aperture path), and at C/F rows there is a weight mismatch (`⅜` vs `½`) but no invisible
  mode. The asymmetry between (A) and (B) is real; its mechanism is Prop. 2, not the checkerboard.
- **"An error that does not shrink under refinement" (§2)** is true at fixed dt and false at fixed
  CFL: `φ ∝ dt²`, so the leak is `O(h²)` under time-accurate refinement (§1, (S)). The brief's own
  large numbers come from `dt = 60` impulsive starts, where the transient is unresolved anyway.
- **"(A) … the trade dissolved rather than resolved" (§2)** holds only where the pair is
  support-consistent; on the aperture path with cut cells (A) converges to `φ_∞ ≠ 0` and leaves
  the leak in place (§4.3).
- **The gate note's §6.8 sketch of (B)** put `G_q φ` into `uf` and needed outer convergence for
  conservation. The lagged form (I′) is conservative at every k (§5.2) — (B) is better than its own
  sketch, and that is what makes a finite cap shippable.
- **"Pressure-correction loops of this family usually need under-relaxation" (§3.2)** — not in
  the bulk of this scheme at any dt (§4.2, the rotational update is the exact Schur gain), and
  where it would be needed (the non-adjoint cut rows at large dt) it does not work (flow's
  measured `Re λ < 0`).
- **§7's ≈ 1 expectation for the §5 order test under (B)** is right, and for the stated reason
  (the O(h) normal offset of `G_q`); recorded in §5.4 so nobody chases order 2 there.
