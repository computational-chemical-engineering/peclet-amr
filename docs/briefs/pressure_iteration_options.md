# Design brief: two optional routes out of the conservation-vs-accuracy trade at coarse/fine faces

> For a Fable design pass. Written 2026-09-22 against `peclet-amr` `2426ef4` (+ the unmerged
> `cfgate` branch). The user's instruction: *"Could they both be included as option (default off,
> and without performance penalty when off?). Let Fable consider this and if positive do the
> design."*

## 1. The question

**Can the solver offer (A) inner pressure-correction iterations within a time step and (B) a
deferred correction that makes the pressure equation effectively invert the accurate coarse/fine
gradient — both as options, both default OFF, both bit-identical and free when off? And is either
or both worth having?**

A negative answer is an acceptable deliverable if argued. If positive, design them.

## 2. Why the question exists

On a graded mesh the advecting face velocity is built as

    uf = (average of the two cell velocities) + Δvel − G φ

and the pressure equation is `L φ = D_accurate(avg u*)` with `L = D · G` — **the accurate
divergence on the right-hand side, the simple gradient inside the matrix.**

`D(uf) = 0` therefore holds *iff* the gradient in `uf` is the same `G` the matrix inverted. It was
not: a more accurate coarse/fine gradient was being substituted into `uf`, and nothing balanced it.
That left a spurious volumetric source: 0.240 at step 1, 6.7e-3 at step 25, 3.1e-4 at step 60 on a
graded sphere relaxing to steady state, and **4.0e-2 sitting flat with advection on** — the same
size as the cell-centred divergence, i.e. the face field had stopped being a flux for the whole
transient.

Fixed on `main` at `1b0d5b5` by removing the accurate gradient from `uf`: `D(uf)` is now 1.3e-12,
steady permeability bit-identical, ladder order bit-identical. **The cost is that the face VALUE at
coarse/fine faces is now wrong by ⅔·|∇_t φ|, an error that does not shrink under refinement.** It
lives on a codimension-1 set and is proportional to φ, so it is zero at steady state — but φ does
not vanish in a genuinely unsteady flow, which is exactly when the face field is used to advect.

Two ways out, both suggested by the user:

**(A) Inner iterations.** Iterate predictor→projection within the step: each pass updates the
pressure, the next predictor starts closer to solenoidal, the increment shrinks. Converged, φ → 0
and `uf` reduces to the plain average with no gradient term at all — conservation and accuracy
together, the trade dissolved rather than resolved.

**(B) Deferred correction.** Keep one projection per step but iterate the solve:

    L φ_{k+1} = rhs − (L_q − L) φ_k,     L_q = D · G_accurate

whose fixed point is `L_q φ = rhs`. Then `uf` may carry the accurate gradient and still cancel, at
any φ. Every solve uses the symmetric `L`, so MG-PCG is untouched.

## 3. What to decide

1. **Are both worth having, or does one subsume the other?** They are not equivalent. (A) shrinks
   φ; (B) removes the mismatch at finite φ. Note the asymmetry I believe matters: **(A) can only
   shrink the part of φ the cell velocities feel.** The cell correction uses the wide gradient
   (average of the two adjacent face gradients), which annihilates the odd–even mode; a checkerboard
   component of φ produces no cell correction, so the inner loop converges with it still present —
   and the face field, using the compact gradient, does see it. (B) is immune because it fixes the
   operator. Verify or refute this; the project's own
   `flow/doc/collocated_invisible_subspace.md` (attractor family, support inconsistency) is the
   related prior art.
2. **Does (A) converge, and under what conditions?** Pressure-correction loops of this family
   (SIMPLE and relatives) usually need under-relaxation. Does this one, at the dt this solver is
   used at (steady drag runs at dt = 1e6 … 1e20)? State the condition, not just the algorithm.
3. **Does (B) converge?** `(L_q − L)` is nonzero only on coarse/fine faces, so it should be a small
   perturbation — but give the contraction argument and an estimate of the iteration count, and say
   what happens when the coarse/fine surface is large relative to the domain (a deeply graded bed).
4. **Zero cost when off.** Both must be bit-identical and allocation-free when disabled. Say how
   (the `cfMom_` deferred-correction path is the in-tree precedent for a term that is simply an
   empty CSR when inactive).
5. **Interaction with the existing schemes.** The ghost projection solves a non-symmetric system
   with BiCGStab; the sampled mixed-level band adds least-squares closures; the momentum path
   already runs a Picard outer loop over lagged advection. Say which combinations are supported and
   which must throw.

## 4. Constraints and invariants

- **Default off ⇒ bit-identical.** `python/state_hash.py` has 13 scenario keys; all must be
  unchanged at np=1 and under `mpirun -np 2`.
- **No performance penalty when off.** No extra allocation, no extra collective, no branch in a hot
  device kernel that is not perfectly predicted.
- Device (Kokkos, CUDA/HIP/OpenMP) and distributed (MPI) must both work, or the option must throw
  where it does not.
- `np=1` bitwise vs single-rank; `np=2/4/8` in the ~3e-7 march class.

## 5. Settled decisions this touches

From `../../docs/decisions/amr.md`:

- **"MG-as-solver with a Picard outer loop projects ONCE per step, never inside the loop."**
  **Option (A) reverses this.** If you recommend (A), record a superseding decision with the
  argument; if you recommend against, say whether this entry is the reason.
- **"C/F scheme pressure matrix/MG/PCG stays standard order — placement is (1,2)"**, justified by
  φ → 0 at the fixed point. Option (B) does *not* reverse it — the matrix stays standard and `L_q`
  appears only on the right-hand side — which is a point in its favour; confirm that reading.
- The collocated coupling is the Almgren–Bell–Colella approximate projection, never Rhie–Chow.

## 6. Already established — do not re-derive

- `D(uf) = rhs − Lφ` identically; this is why the gradient in `uf` must be the matrix's gradient.
  (Called "rule (I)" in `docs/amr_cf_flux_gate.md`.)
- The face-value error is **O(1)·|∇_tφ|, not O(h)**: fine and coarse centres are offset tangentially
  by h/2 per axis over a normal distance 1.5h, giving
  `(φ_C − φ_F)/d = ∂_nφ ± ⅓∂_{t1}φ ± ⅓∂_{t2}φ + O(h)`. With |∂φ| ≤ 2π the limit is ⅔·2π = 4.19;
  measured 4.547 → 4.319 → 4.224 at n = 16/32/64.
- The cell-centred divergence is O(h²) and never vanishes — intrinsic to the approximate
  projection, not a defect to fix here.
- A separate, independent defect (per-row vs per-face gating of the C/F correction at cut cells) is
  fixed on the unmerged `cfgate` branch; assume it lands. Design note `docs/amr_cf_flux_gate.md`
  on `cfgate-design`.

## 7. Verification the design must specify

- Off: 13 byte-gate keys unchanged, np=1 and np=2; battery 83/83 + 17/17; no measurable step-time
  change.
- On: `D(uf)` at solver tolerance AND the face value convergent at coarse/fine faces — i.e. the
  `test_amr_cf_vector` §5 order test, which currently reads 0.07/0.03 for the shipped scheme,
  should recover ≈1 under (B).
- A time-accurate unsteady benchmark on a graded mesh. **We have none** — that is the gap that
  makes this speculative. Say what it should be (decaying Taylor–Green with a refined sub-region is
  the obvious candidate) and what it must show for the option to be worth its cost.
- Stability at large dt: the throat geometries run at dt = 60 … 1e20 (`tests/study/amr_throat_stability.py`,
  `amr_two_sphere_gap.py`).

## 8. Deliverable

`docs/amr_pressure_iteration.md`: the recommendation (both / one / neither, with the argument), the
convergence conditions, the zero-cost-when-off mechanism, work orders for an Opus implementer, and
the gates. Commit on a branch; do not merge.

If the answer is "not worth building until a time-accurate graded benchmark exists", say that
plainly — it is a legitimate outcome and the brief's §7 admits the gap.
