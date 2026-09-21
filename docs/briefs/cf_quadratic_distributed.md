# Work order: make the C/F quadratic scheme work on the distributed path

> Status: brief, 2026-09-21. Prerequisite for making `cf=1` the DEFAULT (user decision, ROADMAP A6).

## 1. The question, decision-shaped

`AmrFlow::setSolid` (`include/peclet/amr/flow.hpp:675-678`) throws on every multi-rank run that
does not use the standard C/F scheme:

```cpp
if (dist_ && cfScheme_ != CfScheme::standard)
  throw std::runtime_error(
      "amr::AmrFlow: the C/F quadratic scheme is not distributed yet (rung-4 follow-up)");
```

**Delete that guard and make it true.** Done when a distributed `Flow` with
`setCfScheme(CfScheme::quadratic)` runs and is bit-identical to single-rank at np = 1, and in the
established decomposition-independence class at np = 2, 4, 8.

## 2. Why it matters now

The user has decided `cf=1` becomes the DEFAULT. It cannot, while it throws multi-rank. Measured
justification for the flip (`docs/amr_graded_convergence.md` §3), as convergence orders on a
self-similar graded ladder:

| interface orientation | standard (today's default) | quadratic |
|---|---:|---:|
| normal to the variation | 2.00 | 2.00 (inert — nothing to correct) |
| tangential | **0.41** | **1.60** |

The register already held that `cf=1` "is not optional on graded meshes" while the shipped default
was `cf=0`; this closes that gap. The flip itself is NOT your job (§7).

## 3. Current state, with anchors

**The four builders** live in `include/peclet/amr/cf_scheme.hpp` and are called from
`AmrFlow::setSolid` at `flow.hpp:910-932`, gated on `cfScheme_ != CfScheme::standard`:
`buildCfLapDelta` (momentum diffusion delta), `buildCfDivDelta` (RHS divergence),
`buildCfGradDelta` (pressure gradients, 3 components), `buildCfUfDelta` (face-field vel + phi).
They are handed two row predicates built there:

```cpp
auto fluidOk    = [&](Index i) { return mom_.isFluid(i); };
auto rowRegular = [&](Index i) { return mom_.isFluid(i) && !mom_.isCut(i); };
```

**The reach that breaks multi-rank — and note the two probe kinds are NOT equivalent.** The
quadratic closure is a tangential substitution: for a directed C/F sub-face it interpolates the
coarse cell at the fine cell's tangential position, which needs the COARSE cell's tangential ±
neighbours, and, when such a neighbour is FINER, an enumeration of its `2^Dim` children covering
the coarse-size region.

- `ap.periodicNeighbor(coarse, tt, dir)` (`cf_scheme.hpp:124`) **already routes through the
  resolver**: `AmrPoisson::periodicNeighbor` (`poisson.hpp:375-380`) calls `probeSlot`, which
  dispatches to `extResolve_` when the coord leaves the block (`poisson.hpp:120-143`). Fine
  multi-rank; it registers a miss and resolves like everything else.
- The child enumeration (`cf_scheme.hpp:138`) is the **actual defect**. It calls `t.find(q)`
  directly on the single-block octree, bypassing the resolver:

  ```cpp
  const Index ch = t.find(q);
  if (ch < 0 || t.level(ch) + 1 != t.level(coarse) || !fluidOk(ch))
    return Samp{};                     // <- silent fallback
  ```

  At a block boundary `t.find` returns −1 for a child owned by another rank, the guard takes the
  fallback, and that tangential side silently drops back to the raw coarse value.

**So the failure mode of simply deleting the guard is not a crash or a missing ghost — it is a
silent, decomposition-dependent accuracy loss at block seams**, the scheme quietly degrading to
first order exactly where two ranks meet. A test that only checks "it runs" passes. Route the child
enumeration through `pres.probeSlot(q)` (as `forEachCoveringSlot` does,
`ghost_projection_sampled.hpp:147-229`); it returns `{slot, level}`, so use the returned level
rather than `t.level()` on what may be a ghost slot.

**The pattern to follow — this is the load-bearing part of the brief.** `prepareDistributed`
(`flow.hpp:1726`) runs a miss-collect fixpoint: each round installs ghost metadata, runs every
PROBER in discovery mode, then `dhalo_.resolveMisses()` returns the pending count and the loop
exits at zero. The SAMPLED ghost overlay — a strictly harder stencil, least-squares clouds of
radius `2.2·max(h,H)` — was carried across the seam exactly this way (rungs D0–D2, 2026-08-30) and
is the precedent to copy:

```cpp
if (ghostSampled_) {
  // ...the overlay runs in PROBE-ONLY discovery mode ... none of the least-squares work a
  // discovery round would discard is done. The REAL build (setSolid, after this fixpoint) runs
  // with every ghost resolved and never sets the flag.
  (void)buildGhostOverlaySampled(*t_, pres_, sdfFn, ..., /*discovery=*/true, gpsRho_, gpsMaxN_);
```

So the shape is: **add a C/F discovery arm inside that fixpoint; leave the real build where it is**
(`setSolid`, after `prepareDistributed` returns), by which point every ghost is resolved.

**The apply side probably needs nothing.** `cfApply` / `cfApplyComp` (`cf_scheme.hpp:581`) read
`f(sl(k))` by slot index, so a CSR carrying ghost slots (`>= n_`) just works, provided the SOURCE
view is sized `nExt_` and its ghost tail is synced. `step()` already calls `syncVel()` /
`syncScalar(p_)` before the consumers. **Verify this rather than assume it** — check every
`cfApply*` call site in `flow.hpp` (there are several: momentum RHS ~`:1235`, divergence ~`:1284`,
the gradient overlays, and the `uf` builders ~`:1425`) and confirm the field it reads is
ghost-tailed at that point. If one is not, syncing it is the fix, not resizing the CSR.

## 4. Constraints and invariants

- **np = 1 must be BITWISE identical** to the current single-rank result, for both schemes. The
  byte gate (`python/state_hash.py`) is the check; `amr.flow_sampled` is the graded scenario.
- **np = 2/4/8 in the established class.** The comparable gate is
  `tests/test_amr_distributed_seam_mpi.cpp`: np=1 bitwise, np=2/4 in the ~3e-7 march class.
- **Ghost-slot numbering must stay canonical across ranks.** `docs/amr_setup_parallel_plan.md:240+`
  is explicit: the miss set's iteration order feeds `resolveMisses`'s collective, so any
  parallelisation of a prober must preserve a canonical order or slot numbering diverges between
  ranks. `LeafHalo::resolve` already guarantees an order-canonical miss set — do not break that.
  If you add a host-parallel discovery loop, mirror how the existing `hostParFor` face sweep does
  it (`flow.hpp:1757`).
- **Do not change the scheme.** The arithmetic in `cf_scheme.hpp`'s closure is settled and gated by
  `test_amr_cf_vector` / `test_amr_cf_quadratic`. This is a neighbour-resolution change only.
- **The row gates are load-bearing and have bitten before.** `../docs/decisions/amr.md` records
  that `cfDiv`/`cfGrad` firing at CUT rows (gate was `rowFluid`, not `rowRegular`) violated support
  consistency and made 2 of 12 throat-graded meshes march to k~1e12 by step ~100. Keep
  `rowRegular` for `cfDiv_`/`cfGrad_`. A row whose tangential neighbour is now a GHOST must gate
  identically to how it would on a single rank — that is a new way to get this wrong.
- **Tags.** AMR direct point-to-point tags are 11 / 41 / 45 and must stay below 24576; core's
  `NbxEngine` rotates its own. Consecutive NBX rounds walking over AMR gather tags caused
  intermittent wrong ghosts until 2026-09-05 (`CLAUDE.md` Gotchas). If you add a collective, do not
  invent a new tag without checking.
- Header-only C++20, Kokkos, clang-format 18.1.8 is a BLOCKING CI gate over `include/ tests/
  python/ benchmarks/`.

## 5. Already decided — do not reopen

- The scheme is Martin–Cartwright tangential quadratic. Not up for discussion.
- `cf=1` becomes the default — but in a LATER commit, not yours (§7).
- The pressure MATRIX / MG / PCG / ghost-BiCGStab stay on the standard operator (register: "at the
  fixed point φ→0 the matrix C/F order cannot move the steady solution").
- The mixed-level sampled band and the classic ±2 overlay stay as they are.

## 6. How the answer is verified

Run, in this order, and report the numbers:

```bash
cd /home/frankp/Codes/suite/amr-parity2 && source ../.venv/bin/activate
cmake --build build_q -j10
OMP_NUM_THREADS=1 PYTHONPATH=$PWD/build_q python python/state_hash.py --check python/state_hash_reference.json
OMP_NUM_THREADS=2 OMP_PROC_BIND=false ctest --test-dir build_q --output-on-failure -LE 'bench|np8'
OMP_NUM_THREADS=1 OMP_PROC_BIND=false ctest --test-dir build_q --output-on-failure -L np8
```

plus a NEW gate you write: a distributed graded case with `cfScheme_ = quadratic`, asserting
np=1 bitwise vs single-rank and np=2/4 within the march class — model it on
`tests/test_amr_distributed_seam_mpi.cpp`, and register it through
`cmake/PecletAmrTest.cmake` like every other test (exit 77 to skip, `mpi` + `np8` labels).

**The test must be able to FAIL on the defect above.** A WORLD-vs-SELF comparison at np=1 cannot —
there are no seams at np=1. It needs np=2/4 with the 2:1 interface deliberately straddling a block
boundary, compared against the single-rank answer. If the existing decomposition cannot be made to
put a seam on a C/F interface, say so rather than shipping a test that cannot detect the bug it
exists to prevent.

The byte gate must come back IDENTICAL: the default is still `standard` while you work, so nothing
you do should move a hash. If one moves, stop and say so — it means you changed the single-rank
path.

## 7. Out of scope

- **Flipping the default.** The patch is already written and parked
  (`cf1_default.patch`, `cf1_register.patch` in the session scratchpad); it lands after your work,
  in its own commit, with its own byte-gate re-baseline.
- Re-taking the contaminated `set_ghost_sampled` campaign numbers (`ROADMAP` A5 note).
- The convergence study's remaining rungs.
- Anything in `peclet.flow` — another session is editing that repo right now.

## 8. Deliverable

The guard gone, the fixpoint extended, a distributed C/F ctest registered and passing, and a short
note appended to `docs/amr_setup_parallel_plan.md` (which currently says rung 4 "is unaffected …
so is rung 4" — that statement becomes stale) recording what the discovery arm probes and how many
fixpoint rounds it costs. Commit on the branch `parity2`; do not merge, do not push.
