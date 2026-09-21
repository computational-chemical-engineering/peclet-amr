# Uniform-grid parity: `peclet.amr.Flow` vs `peclet.flow.SolverColocated`

> Status: **measured, 2026-09-21.** This note answers, with numbers, a question the tree had
> carried open since 2026-07-22: *does the AMR collocated solver reduce to flow's collocated
> solver when the octree is uniform, and if not, where exactly does it differ?*
>
> `amr_collocated_projection.md` §"Open / next" item 4 recorded the state of knowledge then:
> a ~0.3–1 % gap in the Zick–Homsy drag coefficient, *"still not isolated to one line. The clean
> way: expose each engine's projection … feed both the SAME synthetic field on the SAME cut
> geometry, diff cell-by-cell. Neither exposes that yet."* That is now done, and the answer is
> better than the old table suggested: **on the production (ghost) scheme the two engines agree
> to 3e-7 on the cut-cell sphere, and there is exactly ONE discretization difference between
> them — in `flow`, not in `amr` — and that one was closed on 2026-09-21 (§3), so the two
> engines now agree to 4.4e-11 over 20 steps of full Navier–Stokes.**

## 1. The question, and why it has a crisp answer

Both codes solve the same method: cell-centred (collocated) velocity, Almgren–Bell–Colella
**approximate projection** (never Rhie–Chow — see `../docs/DECISIONS.md`), backward-Euler implicit
viscous predictor, rotational (Timmermans) incremental pressure update, cut-cell IBM over an SDF.
`amr` runs it on a `BlockOctree` through a face-CSR operator; `flow` runs it on a structured MAC
brick. At `lmax = 0` the octree IS a uniform brick, so the two discretizations must coincide —
or the difference is a bug in one of them.

The grids coincide exactly, which is what makes a cell-by-cell diff meaningful:

```
peclet.flow.SolverColocated(N, N, nz)  ->  cells [N,N,nz]  extent [N,N,nz]  spacing 1  centres 0.5,1.5,…
peclet.amr.Octree(cells=[N,N,nz], lmax=0, spacing=1.0)      extent [N,N,nz]  spacing 1  centres 0.5,1.5,…
```

`amr`'s face CSR enumerates each leaf's faces as `[-x, +x, -y, +y, -z, +z]`
(`facegeom_assembly.hpp:58`), so slot `2a` of leaf *i* is the **low** face on axis *a* — precisely
`flow`'s `uf/vf/wf(i) = ½(U(i)+U(i-1))` convention (`mac_approx_projection.hpp:30`). The face
fields are therefore directly comparable too, not just the cell fields.

The two modules **cannot share one process** (each finalizes Kokkos under the other's Views), so
the harness runs one subprocess per engine and diffs the `.npz` dumps.

## 2. The measured ladder

Host-OpenMP Release, `OMP_NUM_THREADS=2`, `flow` at `6adbb4b`, `amr` at `cd1a58a`.
"rel" is the L2 relative difference over shared fluid cells; pressure is compared
fluid-mean-removed (it is defined up to a constant in a periodic box).

| # | case | solid | scheme | rel diff `u` | verdict |
|---|------|-------|--------|-------------:|---------|
| L1 | Taylor–Green, 20 steps, advection OFF | none | — | **1.1e-15** | identical |
| L2a | Taylor–Green, **1** step, advection ON | none | — | **1.2e-10** | identical (solver tol) |
| L2 | Taylor–Green, 20 steps, advection ON | none | — | 2.5e-4 | **differs — §3** |
| L2′ | as L2, `amr` uf-advection ablated OFF | none | — | **1.6e-11** | identical (solver tol) |
| L3 | body-force channel, walls ON cell faces | slab | aperture | **1.5e-10** | identical |
| L4 | body-force channel, walls at 10.3/22.7 (cut) | slab | aperture | **1.6e-10** | identical |
| L5g | Zick–Homsy sphere, Stokes, 40 steps | sphere | **ghost** | **2.9e-7** | identical (solver tol) |
| L5a | same | sphere | `flow` gauge-exact / `amr` aperture | 1.8e-3 | **not a matched pair — §4** |
| L5p | same | sphere | `flow` plain / `amr` aperture | 7.6e-3 | not a matched pair |
| L7 | sphere, advection ON, Re≈180, CFL≈0.45, 20 steps | sphere | ghost | 1.6e-4 | **differs — §5** |

L1–L4 say the **entire shared core is already bit-parity to solver tolerance**: the 7-point
operator, the divergence, the ABC cell gradient, the face averaging, the backward-Euler momentum
operator, the cut-cell Robust-Scaled no-slip overlay, the rotational pressure update, the periodic
wrap. L5g says the **cut-cell ghost projection** — the production default in both engines — is too.

Iteration counts and cost on L5g (the case that actually exercises the projection):

| | `flow` | `amr` |
|---|---|---|
| pressure iterations / step | 11, 11, 11, 10, 10, 10 | 10, 10, 10, 10, 10, 10 |
| ms / step (32³, 2 threads) | 313 | 303 |

**L5g is a shared fixed point, not a coincidence at one step count.** Running the same case longer
shrinks the difference monotonically — the two engines are converging to the *same* steady state,
and what is left is each one's own distance from it:

| steps | 25 | 40 | 80 |
|---|---:|---:|---:|
| rel diff `u` | 8.2e-7 | 2.9e-7 | 1.7e-7 |
| rel diff `p` | 1.3e-4 | 1.0e-4 | 7.9e-5 |

And it is not an accident of one resolution. The same case at **N = 64** (sphere R = 19.87,
25 steps, 229 176 shared fluid cells): `u` 1.1e-6, `v`/`w` 3.9e-6, `p` 3.3e-4 — the same class,
with pressure iterations 14–15 (`flow`) vs 16–17 (`amr`) and 3.13 s vs 4.46 s per step.

### 2a. A diagnostic trap: `divergence_norm_face()` under the ghost scheme

Worth knowing before it costs someone a day. On the runs above `amr`'s
`Flow.diagnostics.divergence_norm_face()` reads **34** (N = 32) and **184** (N = 64) while
`flow`'s `max_open_divergence()` reads 2e-14 and 7e-13 — and the velocity fields agree to 1e-6.
The solver is fine; the diagnostic is measuring the wrong constraint. `divFaceNorm`
(`flow.hpp`) sums the plain area-weighted face divergence, but the **ghost** scheme's constraint is
that divergence **plus the overlay delta** (`ghostDivergDelta`, added to `div_` in `project()`
before the solve). The diagnostic omits that term, so under the default scheme it reports the
residual of a constraint the solver never solved. On the **aperture** path, where the diagnostic
and the constraint do match, the same case reads 5.2e-12.

It is also an *unnormalized* L2 sum over fluid cells, so it grows with both resolution and velocity
magnitude and cannot be read as an absolute number. `divergence_norm()` (the cell divergence) is
the honest residual of the approximate projection and is expected to be non-zero.

## 3. The one discretization difference: the advecting velocity

L2 vs L2a localizes it exactly. After step 1 the two engines agree to 1e-10 in **`u`, `p` AND the
face field `uf`** — everything. The divergence appears at step **2**, the first step at which a
projected face field exists, and it scales as **dt²**:

| dt | 0.005 | 0.05 | 0.5 | 5.0 |
|---|---|---|---|---|
| max abs diff in `u`, 2 steps | 3.8e-8 | 3.7e-6 | 3.8e-4 | 4.5e-2 |

— i.e. an O(dt) difference in the *advecting* velocity, insensitive to µ (100× change) and
unaffected by ablating the rotational term. The cause is one function:

- **`amr`** advects with the projected, divergence-free face field
  `uf = ½(u_i+u_j) − ∇φ` (`flow.hpp::buildFaceField`), falling back to the plain average only on
  the first step, before any projection has run. This is a **recorded settled decision**
  (`../docs/decisions/amr.md:13`, 2026-07-24) and it is the Almgren–Bell–Colella / Basilisk
  prescription: the field the projection just made solenoidal IS the conservative advecting flux.
- **`flow`** advected (until 2026-09-21 — see the RESOLVED box below) with the *un-projected*
  cell→face average
  `½(U(x)+U(x+1))` — `colocated_advection.hpp:29-36`. Its own header says the swap is pending:
  > *"in phase 2 (no pressure) the advecting velocity is the plain cell→face average … Once the
  > approximate projection lands (phase 3), the natural advecting field is the projected,
  > divergence-free face velocity — this header's `adv_vel` is where that swap happens."*

  Phase 3 landed; the swap did not. `doc/flow_colocated_plan.md` §1 step 3 prescribes it
  (*"These `u_f` become the advecting velocities for the next step's advection"*), and
  `../docs/decisions/flow.md:977` independently records that the FOU operator's conservative
  row-sum identity *"holds only for uniform/div-free advecting field"* — which the plain average
  is not.

Ablating `amr` down to `flow`'s choice closes the gap completely (L2′: 1.6e-11 over 20 steps of
full Navier–Stokes). The ablation is `Flow.diagnostics.set_uf_advection(False)` —
**developer tier, default unchanged**; it exists to A/B exactly this and to make the parity gate
above runnable without touching either solver's numerics.

**This was a `flow` defect against `flow`'s own design note, not an `amr` deviation.**

> **RESOLVED 2026-09-21 — `flow` took the swap.** `peclet.flow`'s collocated momentum advection
> now reads the projected divergence-free face field, with the same `set_uf_advection` knob (same
> spelling) as the ablation. `flow/doc/uf_advection.md` carries the implementation, the evidence
> and the limits; `suite/docs/decisions/flow.md` carries the decision. Measured afterwards, both
> engines at their default: **vel 1.90e-11, pres 4.41e-11** over the same 20 NS steps — so the
> gate case below is no longer an ablation but an agreement gate, and `tg_advect_matched.json`
> dropped its `"uf_advection": false` (the tolerance tightened 1e-8 → 1e-9). §7's P1 is closed.
> `amr`'s own scheme and `docs/decisions/amr.md:13` are unchanged: `flow` moved to `amr`.
>
> Landed in `flow` as `8e21724` / `f461151`, and measured independently from each side: `flow`
> reports 2.5e-4 → 1.9e-11 on its case, this repo 3.950e-04 → 4.414e-11 on the gate case. Two
> codebases, two harnesses, the same conclusion.


### 3a. …and how much it is worth, measured

The ablation makes the stake of that decision measurable without touching `flow`: run `amr` both
ways and compare to a case with a known answer. 2-D Taylor–Green in a periodic box is an **exact**
Navier–Stokes solution (the nonlinear term is balanced by the pressure gradient), 40 steps, ν=0.05,
dt=0.5:

| N | advecting velocity | L2 error vs exact | observed order |
|---|---|---:|---:|
| 32 | projected `uf` | 7.874e-3 | — |
| 32 | plain ½(u_i+u_j) | 7.909e-3 | — |
| 64 | projected `uf` | 1.467e-3 | 2.42 |
| 64 | plain ½(u_i+u_j) | 1.472e-3 | 2.43 |

and momentum conservation, on a Galilean-shifted vortex (mean(u) = 0.5, periodic, no body force,
60 steps) where the continuum answer is exactly constant:

| N | projected `uf` | plain ½(u_i+u_j) |
|---|---:|---:|
| 32 | 5.8e-15 | 5.3e-15 |
| 64 | 5.6e-15 | 5.8e-15 |

**On smooth flow the two choices are indistinguishable** — 0.4 % apart in error, same order, both
conserving momentum to round-off. The case for the swap is therefore *consistency*, not measured
accuracy: `flow`'s own design note prescribes it, the FOU row-sum identity needs it
(`../docs/decisions/flow.md:977`), it is the ABC/Basilisk convention, and it removes the last
uniform-grid difference between the two engines. The case against is the cost of re-blessing
`flow`'s baselines. **That trade is the user's, and this section is what makes it cheap to judge:
the change is low-risk and low-reward on smooth cases.** Where it could still bite is the regime
the dt² scaling points at — large-dt steady driving on cut-cell beds, which is exactly `flow`'s
production regime and is NOT covered by the two tests above.

**Outcome (2026-09-21).** The user took it, and `flow` measured that last regime on its own side:
a 32³ cut-cell sphere bed at dt=20 with implicit advection moves ⟨u_x⟩ by **1e-5 %**
(2.115907e-01 → 2.115886e-01), and every advection-free collocated baseline is bit-identical
because there is no advecting velocity in a Stokes run. The blast radius was smaller than the
section feared; the table above predicted the direction correctly. See
`flow/doc/uf_advection.md` §3.

## 4. What is not yet a matched pair

The **aperture** family. `flow` exposes `set_collocated_scheme('ghost' | 'gauge-exact' | 'embed' |
'plain')`; `amr` exposes `set_ghost_projection(on)` plus the developer-tier
`set_ghost_gradient(on)` / `set_aperture_order(order)`. `ghost` ↔ `ghost` is an exact pair (L5g).
`gauge-exact` is **not** the pair of `amr`'s aperture path with either setting of
`set_ghost_gradient` (1.8e-3), and `plain` is not either (7.6e-3). What `amr`'s aperture path
corresponds to on `flow`'s side — or whether a matched mode has to be added on one side — is open.
Since both engines default to `ghost`, this blocks nothing in production; it blocks a complete
scheme-by-scheme parity matrix.

`embed` (the Basilisk `embed.h` port, `flow`'s live candidate for the collocated accuracy ceiling)
has **no `amr` counterpart at all**.

## 5. The second difference: cut cells under advection

L7 adds advection to the cut-cell sphere at a usable CFL (dt = 0.05, `u_max` ≈ 0.12, CFL ≈ 0.45,
Re ≈ 180) and leaves a residual 1.6e-4 in velocity / 4.2e-4 in pressure — 200× the L5g Stokes
floor. Ablating the §3 advecting velocity does **not** move it (1.6497e-4 → 1.6484e-4), so this is
a *second*, independent difference: the two engines reconstruct the advective flux differently at
**cut faces**. `amr` uses fluid-fluid gates with raw-area fluxes (`../docs/decisions/amr.md:13`);
what `flow` does there has not been read side by side yet. Not diagnosed further here.

(Note the interaction with §3: the uf-vs-plain-average difference scales as dt², so at a
CFL-limited dt it is small and at a large steady-driving dt it dominates. The two differences
therefore show up in *different* regimes, which is why neither was visible to the other's test.)

The gate `sphere_ghost_advect` records this number with a deliberately loose tolerance. It is a
tripwire against regression, not a parity claim. (Re-measured 2026-09-21 after `flow` took the
projected face field: **1.634e-04**, from 1.6497e-04 — confirming §5's reading that this is an
independent cut-cell difference and not §3's.)

## 6. Reproducing

`tests/study/flow_parity/` (this repo) holds the harness: `parity_gate.py` (the ctest driver),
`run_amr.py` / `run_flow.py` (one engine each), `compare.py` (an interactive field-by-field diff),
`cases/*.json`, and the two §3a studies `uf_study.py` (accuracy vs the exact Taylor–Green) and
`uf_conserve.py` (momentum conservation) — both `amr`-only, so they need no `flow` build. Both engines are driven from the SAME case file, each in its own subprocess.

The gate is registered as the ctest `python_flow_parity`. `flow` lives in a sibling repo, so the
gate is told where its build tree is by the CMake cache variable `PECLET_AMR_FLOW_PYTHONPATH` — a
path, never a numerics switch (QUALITY_PLAN D3). Unset, the ctest exits 77 and ctest reports it
*skipped*, exactly as the morton-guarded tests do, so CI (which checks out no `flow`) shows it
skipped rather than absent.

```bash
cd amr && source ../.venv/bin/activate
cmake -S . -B build_q -DCMAKE_PREFIX_PATH=$PWD/../extern/install/host-openmp \
      -DPECLET_AMR_BUILD_TESTS=ON && cmake --build build_q -j
cd ../flow && cmake -S . -B build_parity -DCMAKE_PREFIX_PATH=$PWD/../extern/install/host-openmp \
      && cmake --build build_parity -j
cd ../amr && cmake -S . -B build_q -DPECLET_AMR_FLOW_PYTHONPATH=$PWD/../flow/build_parity
OMP_NUM_THREADS=2 OMP_PROC_BIND=false ctest --test-dir build_q -R python_flow_parity -V
```

## 7. Open decisions

| # | question | default if nobody decides |
|---|---|---|
| ~~P1~~ | ~~Does `flow`'s `cadv::adv_vel` swap to the projected face field (§3)?~~ | **CLOSED 2026-09-21 — taken.** `flow` swapped; both engines now agree to 1.9e-11 on the full NS step. Recorded in `suite/docs/decisions/flow.md`; evidence in `flow/doc/uf_advection.md` |
| P2 | Which `flow` scheme (if any) is the pair of `amr`'s aperture path (§4)? | left unmatched; `ghost` is the gate |
| P3 | Do the two engines have to agree on *iteration counts*, or only on results to solver tolerance? Exact iteration parity means one shared linear-algebra stack (`flow`'s structured RB-GS/CutcellMG vs `amr`'s face-CSR MG-PCG/BiCGStab), not a tuning exercise. On L5g they already agree to ±1. | results-to-tolerance is the gate; iteration count is reported, not gated |
| P5 | What makes the cut-cell advective flux differ (§5)? | undiagnosed; gated as a tripwire |
| P4 | Does `amr` gain a public tolerance API (`flow` has `set_pressure_pcg(on, iters, rtol)` and `set_velocity_residual_tolerance`; `amr` hard-codes the pressure tolerance at `flow.hpp:1363` and leaves `setMomentumTol` unbound in Python)? Needed for a tolerance-matched gate and wanted on its own merits. | yes — see the next-steps list |
