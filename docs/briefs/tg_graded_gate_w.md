# Review brief: WO0 is built and run — does gate W fire?

> For a Fable review pass, 2026-09-23, on branch `wo0-tg-graded` of `peclet-amr` (base `43de409`).
> The reviewer is the author of the design note being tested; the question is whether its own
> acceptance criterion should be honoured literally or corrected.

## 1. The question

`docs/amr_pressure_iteration.md` §10 gate W says: build the deferred correction (B) if, at the
largest time-accurate dt, the graded arm reads `m5 = m1/m3 ≥ 0.3`. **Measured m5 = 1.46.** The gate
fires.

I am recommending **park (B) anyway**, on the grounds that m5 cannot measure what the gate wanted
it to measure. **Is that right, or should (B) be built?**

If the recommendation is wrong, say so and (B) is built from §6–§8 as written. If it is right, say
what gate W should have been — the note's §10 row should be corrected, not deleted.

## 2. What was built and run

`tests/study/amr_tg_graded.py` implements §9: decaying 2-D Taylor–Green, advection on, exact
solution, horizon 2 convective times, cell CFL ladder. Two deviations from §9, both deliberate:

* **Horizon `T = 2N` (2 convective times), not 20 time units.** With `T = 2N` every CFL of the
  ladder divides the horizon exactly at every N, and the same convective progress is covered at
  each resolution. (The viscous decay fraction then differs between N — 22 % at N = 32, 12 % at
  N = 64 — which the exact solution accounts for.)
* **A third arm (C): the same graded mesh with the shell NOT refined** (uniform `lmax = 1`,
  h = 2 everywhere). §9 has only (U) and (G), and (U) cannot price the interface because it is a
  different resolution. (G) ⊇ (C) in resolution everywhere, so (G) − (C) is a clean price tag.

`refine_to_sphere` refines the shell the sphere surface passes through, not the ball, so (G) has
two closed C/F surfaces rather than one. `m1` is exactly §9's definition and exactly
`test_amr_cf_vector` §5's sample point (the finer cell's face centre).

Everything is in **`docs/amr_tg_graded.md`** — read that, not this brief, for the numbers. The
headline rows, cell CFL 0.5, time-converged:

| arm | N=64 leaves | m3 | shape | 1−amp | m1 (C/F) | m2r (regular) | m6 = m1/m2r |
|---|---|---|---|---|---|---|---|
| U | 262 144 | 1.6729e-03 | 3.897e-04 | 2.60e-03 | — | 3.245e-03 | — |
| C |  32 768 | 9.9231e-03 | 2.345e-03 | 1.54e-02 | — | 1.738e-02 | — |
| G |  48 448 | 1.4940e-02 | 7.423e-03 | 2.07e-02 | 3.382e-02 | 3.192e-02 | 1.06 |

Orders 32 → 64 on (G): m3 **2.05**, shape 1.81, amplitude 2.30. div(uf) ≤ 4e-12 everywhere at
CFL ≤ 2.

## 3. The argument for parking

1. **m5 is flat in dt.** At N = 32, over CFL 0.0625 → 1 (16×), m1 reads 9.2652e-02, 9.2364e-02,
   9.1729e-02, 9.0727e-02, 9.0760e-02 — it *decreases* by 2 %. m5 stays 1.46–1.48 throughout. A
   leak proportional to φ must grow with dt. This one does not appear at all: it is under 2 % of m1
   across a 16× dt range.
2. **m5 is an L∞-over-codimension-1 divided by a volume L2.** On this mesh it reads 1.4 at N = 32
   and 2.3 at N = 64 for reasons that have nothing to do with φ: m1 converges at ~1.4 and m3 at
   ~2.0, so the ratio grows like h^−0.6 by construction.
3. **m6 = m1/m2r — the same max restricted to regular faces — is 0.75 / 0.84 / 1.06** at
   N = 16/32/64. The worst advecting-velocity error on the mesh is at an ordinary face, not a C/F
   one. This is the comparison §9 should have asked for and did not.
4. **Gate S is answered and §1's argument survives**: m1 shows no order-1 dt dependence, so §12
   risk 1's "if m1 measures order 1 in dt, build (B)" does not trigger either.

## 4. Where I am least sure

* **Is the 2 % flatness in (1) really an upper bound on the leak, or could the leak be cancelling
  against another dt-dependent term?** Both m1 and m3 fall slightly as dt falls, so a leak of a few
  e-3 hidden under a solution error moving the same way is conceivable. I did not build the sharper
  instrument (one probe step of varying dt from a frozen time-converged state), judging that it
  could not change the verdict once m6 ≤ 1.06. Say if that judgement is wrong.
* **§6 of `amr_tg_graded.md` is a new finding I did not expect and may be misreading.** (G) is
  1.51× *worse* in m3 than (C) — the identical mesh without the refined shell, 48 % fewer cells —
  and its shape error is 3.2× worse. Both parts converge at ~2, so it is a constant, not an order
  loss. My reading is that the remaining O(h) **normal**-offset term of `G_q` (`amr_pressure_iteration.md`
  §5.4, explicitly out of scope there) is the standing suspect, and that this belongs on the
  ROADMAP as its own item. Is that the right suspect, and is 3.2× the size one would predict from
  it? An alternative reading I could not exclude: the t = 0 projection removes an O(h) divergence
  blob at the interface (the exact field's discrete divergence is not O(h²) across a 2:1 face), and
  what is measured at T is that initial transient, not a per-step defect.

## 5. What not to spend time on

* Whether (B)'s design is correct — it was reviewed when it was written and nothing here touches it.
* The benchmark's software (argument parsing, the ctest registration, the `face_topology()` binding).
* `peclet.flow`, the uniform-grid parity work, the cut-cell items (A2/A3).
* Re-deriving the O(1)·|∇_tφ| face-value analysis — this benchmark does not contradict it; it
  measures that the term is small at time-accurate dt, which is §13's own correction.

## 6. Deliverable

A short verdict, in `docs/amr_pressure_iteration.md` as an appended §14 (do not rewrite the note):
build (B) or park it; the corrected spelling of gate W; and a one-paragraph read on §4's second
bullet (the (G)-vs-(C) gap) saying whether it is the normal-offset term, the initial projection, or
something else, and what the ROADMAP item should be.
