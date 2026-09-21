# Parallel AMR setup: the builder restructure (post-scene-layer)

*Plan, 2026-08-30 (Fable analysis; rungs 0/0.5 EXECUTED BY FABLE — the shared-dependency and
abstraction-design rungs; rungs 1–5 are Opus execution). Companion to
`amr_mixed_level_cut_band_plan.md` (the campaign this serves) and
`../../docs/archive/AMR_GEOMETRY_SETUP_REQUIREMENTS.md` §5–6 (the scene-layer handoff this builds on).*

**Status: DONE 2026-08-29 (rungs 0, 0.5, 1, 2, 3, 4, 5 all landed and pushed).
`setSolid` on the depth-7 RCP bed: 10.5 → 2.4 µs/leaf at 8 threads, 1.7 at 16 — the ≤2.5
target is met. One escalation, F1 (§Findings): the distributed path stays serial. See §6 for
the measured result table.**

## 1. The measured starting point (the current denominator — do not reuse old numbers)

The scene layer took `AmrFlow::setSolid` from 127 → **13.1 µs/leaf** (depth-7 RCP bed, 1.79M
leaves, RTX 5080, single-threaded; `.sdf-campaign-probes/time_setsolid.py`). The remaining cost
is FIVE serial host builders (`PECLET_AMR_PROFILE_SETUP=1`, 2026-08-30):

| phase | µs/leaf | share | parallel structure (verified in source) |
|---|---:|---:|---|
| `mom_.build` (cut stencils) | 4.2 | 32% | per-leaf loops over `i < n`, per-leaf array writes |
| `presMG_.build` (openness per level) | 2.6 | 20% | per-level, per-cell loops |
| cf overlays (4 builders) | 2.2 | 17% | **already per-leaf staged** (`per[i]` + `compactCsr`) |
| `buildOpenness` (mixed rule) | 2.0 | 15% | per-leaf face loops |
| `buildGhostOverlaySampled` | 1.4 | 11% | **sequential row append** (`slot = ov.base.n`) — needs two-pass |
| pockets, MG builds, device assembly, uploads | ~0.9 | 5% | leave serial (already cheap) |

Target: **13.1 → ≤2.5 µs/leaf at 8–16 threads** ⇒ depth-8 `setSolid` ≈ 15–20 s (from 95.6 s),
depth-9 ≈ 1–2 min — and ~77% of a Snellius bed job's bill stops being serial host time.
Evaluation is NO LONGER the lever (~0.03–0.05 µs/eval); the lever is loop parallelism and
allocation hygiene. Re-measure every rung against the CURRENT number (handoff trap 6.3.4).

## 2. Decisions (settled here — Opus implements, does not revisit)

**D1′ — Host parallelism via PURE KOKKOS over an OpenMP host execution space (user decision
2026-08-30, revising the first-draft OpenMP-pragma route).** The suite's parallel model is
Kokkos, single-source — CUDA was retired for exactly that reason, and raw pragmas would be a
second model and a dead end for the later device-assembly campaign (Kokkos lambdas over a host
space are already the right shape to move to the device space).

Mechanically, three pieces:
1. **Prefix rebuild**: the `nvidia-cuda` Kokkos install is `SERIAL;CUDA`
   (`tools/bootstrap_deps.sh:49`), so host `parallel_for` runs serial today. Add
   `-DKokkos_ENABLE_OPENMP=ON` to that variant (OpenMP host + CUDA device is a standard Kokkos
   configuration) and rebuild the prefix + every dependent build tree (locally and on
   Snellius). This changes the shared dependency for flow/dem too — see rung 0's coordination
   and no-regression gates.
2. **Builders**: plain `Kokkos::parallel_for` / `parallel_scan` over
   `Kokkos::DefaultHostExecutionSpace` — no pragmas anywhere.
3. **Shim for the Kokkos-free builds**: the shared builder headers (`cf_scheme.hpp`,
   `ghost_projection_sampled.hpp`) also compile in the no-Kokkos host/oracle builds. A ~20-line
   `peclet::core::hostParFor(n, body)` / `hostParScan(...)` in `common/` maps to the Kokkos host
   space when compiled with Kokkos and to a serial loop otherwise — one abstraction, one place,
   serial semantics bitwise-unchanged.

`-mfma` on the amr_bindings host flags rides along (the §5 note; independent of the above).
*Why not device assembly now:* these builders are host-by-design for oracle parity (weights
host-built, shared verbatim — the parity-by-construction contract); device-resident overlay
assembly is a later campaign (`archive/amr_device_assembly_plan.md`), which D1′ feeds but does not
start.

**D2 — Determinism by construction, verified by thread-count invariance.** Every parallelized
builder must produce BITWISE-identical output at OMP_NUM_THREADS=1 and =16. This is achievable
because none of the five builders carries a cross-leaf floating-point reduction — each leaf's
rows/values depend only on (octree, scene). The patterns:
- per-leaf array writes (`mom_.build`, `buildOpenness`, `presMG_.build` per level): disjoint
  writes — deterministic under ANY schedule; nothing else needed (determinism must never
  depend on scheduling policy).
- per-leaf staged vectors (`cf` builders): parallelize the leaf loop filling `per[i]`;
  `compactCsr` stays serial (cheap, order-preserving).
- sequential append (`buildGhostOverlaySampled`): TWO-PASS — pass 1 (parallel) classifies each
  leaf (clean/row) and computes its slot counts; an exclusive `parallel_scan` on the INTEGER
  counts (exact regardless of execution order) assigns row indices and CSR offsets in leaf
  order; pass 2 (parallel) fills each row at its precomputed offsets.
  Row order = leaf order = today's serial order ⇒ the CSR is bitwise-identical, and the
  `test_seam_sampled` parity and mask references hold without re-blessing.
If any builder turns out to carry hidden sequential state, that is an ESCALATION (see §5),
not a workaround.

**D3 — The collect-points → batch-evaluate → build-rows restructure is DEFERRED, not dropped.**
With `SphereBedQuery` inlining into the templated `SdfFn` at ~0.04 µs/eval, batching buys the
HOST path little; its real value is device-resident assembly later. Restructuring row builders
around point batches now would churn the exact code being parallelized. It moves to the
device-assembly campaign; this plan's rungs must not make it harder (keep sampling call sites
localized per leaf).

**D4 — Scope guard.** No numerics change anywhere: same samples, same gates, same row content,
same order. No `geom/` edits (the scene layer is frozen; trap 6.3.1: never touch the
fma-canonical expressions). `findPocketCells`, nullspace, MG hierarchy internals, device
assembly: untouched. Distributed path: untouched (the builders run identically per rank; host-parallel
loops are rank-local and safe, but DO NOT restructure any distributed logic).

## 3. Rungs (commit per rung; every rung ends green and MEASURED)

*Execution log — all landed on `main` (core / umbrella): rung 0 `f990828` / `747b383`,
0.5 `c74cbee` / `9923dcc`, 1 `f586a9f` / `44d9d3c`, 2 `71a1e7a` / `d3983ec`,
3 `7704bae` / `c8b09df` (carries finding F1), 4 `409ff48` / `d21419a`, 5 = this document +
`../../docs/archive/AMR_GEOMETRY_SETUP_REQUIREMENTS.md` §1a + `amr_mixed_level_cut_band_plan.md` P3c
observation 4. The ctest count is now 149 (148 + `host_parallel`), so read gate (a) as 149/149.*

Gates for every rung: (a) 148/148 core ctests on `build_komp3` (OpenMP battery:
`OMP_NUM_THREADS=8 OMP_PROC_BIND=false`) and `build_kcuda2`; (b) `time_setsolid.py 7` bitwise
mask vs `.sdf-campaign-probes/mask6_before.npy` at depth 6; (c) thread-count invariance:
`set_solid` + 200 steps, fields bitwise at OMP_NUM_THREADS=1 vs 16; (d) the rung's µs/leaf,
in the commit message, measured at 1 and 8 threads.

- **Rung 0 (FABLE — executed 2026-08-30) — the prefix rebuild + no-regression fence (D1′
  piece 1; the only rung that touches a SHARED dependency).** Pre-verified before enabling:
  no code in core/flow/dem names a host execution space explicitly, so existing kernels keep
  their spaces; static Kokkos linkage means already-built flow/dem binaries are UNAFFECTED
  until their owners rebuild — but do NOT run mixed-prefix processes (e.g. coupling imports
  flow+dem in one interpreter) until both sides are rebuilt against the new prefix.
  (a) `bootstrap_deps.sh`: add `-DKokkos_ENABLE_OPENMP=ON` to the nvidia-cuda variant; rebuild
  the prefix; rebuild the dependent trees used by the gates (`build_kcuda2`,
  `python/build_cuda2`; flow/dem trees as their owners need them).
  (b) `-mfma` on amr_bindings host flags (expect ~15% at 1 thread — fma is a libm call today).
  (c) NO-REGRESSION FENCE before any builder change: 148/148 core ctests both nets;
  `.sdf-campaign-probes/flow_probe.py` at its pinned 4 threads (u_sum must still print
  6.74193610583927948e+05); `time_setsolid.py` depth 6/7 unchanged vs the §1 numbers (the
  OpenMP runtime being merely PRESENT must not move anything — builders are still serial
  here); geom bit-parity gates (`geom_batch_device` etc.) green. Document
  `OMP_NUM_THREADS`/`OMP_PROC_BIND=false` guidance in the suite CLAUDE.md — with OpenMP in the
  default prefix, the 48-unbound-threads trap becomes suite-wide.
- **Rung 0.5 — the `hostParFor`/`hostParScan` shim (D1′ piece 3)** in `common/`, with a unit
  test asserting serial ≡ parallel bitwise on a synthetic disjoint-write fill and an integer
  scan. No builder touched yet.
- **Rung 1 — cf builders parallel** (via the shim). The `per[i]` leaf loops in
  `buildCfLapDelta` / `buildCfDivDelta` / `buildCfGradDelta` / `buildCfUfDelta`. Lowest risk
  (staging already per-leaf), a clean template for the rest. 2.2 → ~0.3 µs/leaf expected at 8
  threads.
- **Rung 2 — `buildOpenness` + `presMG_.build` parallel.** Per-cell/per-level loops; verify
  per-face slot ownership is cell-major (it is — `forEachFaceFull` is cell-major) before
  writing. 4.6 → ~0.6.
- **Rung 3 — `mom_.build` parallel.** The largest phase; several consecutive per-leaf loops.
  PRE-CLEARED by Fable 2026-08-30: the build loops contain NO neighbour-indexed writes (grep
  over the build body: zero `[...j...] =` stores) — all writes are own-leaf slots, so the
  disjoint-write pattern applies directly. If implementation contradicts this, escalate. 4.2 →
  ~0.6.
- **Rung 4 — sampled overlay two-pass.** D2's third pattern. The delicate one: row order and
  slot CSR order are CONTRACTUAL (device upload, parity ctest, oracle equivalence). Pass-1
  classification must call the same gates as today's single pass (D6 determinism of the plan:
  row state is a pure function of geometry — so classifying twice is safe by contract).
  1.4 → ~0.3.
- **Rung 5 — re-measure + record.** `time_setsolid.py` at depths 6/7/8, 1/8/16 threads; update
  `AMR_GEOMETRY_SETUP_REQUIREMENTS.md` §1 table and `amr_mixed_level_cut_band_plan.md` P3c
  observation 4; memory hook. If ≤2.5 µs/leaf at 8 threads: DONE. The oracle
  (`flow_oracle.hpp`) shares the cf/overlay builders, so it speeds up for free — verify its
  ctests unchanged, do NOT add oracle-only pragmas beyond what it inherits.

## 3a. Result (rung 5, measured 2026-08-29 — RTX 5080 host, 180-sphere RCP bed)

Per-phase, depth 7 (1.79M leaves), µs/leaf. "before" is the rung-0 denominator; the parallel
columns are the same build at different `OMP_NUM_THREADS`:

| phase | before | 1 thr | 8 thr | 16 thr | rung |
|---|---:|---:|---:|---:|---|
| `mom_.build` | 2.4 | 2.6 | 0.4 | 0.2 | 3 (single-rank only, F1) |
| `presMG_.build` | 2.4 | 2.4 | 0.3 | 0.2 | 2 |
| cf overlays | 2.2 | 2.3 | 0.5 | 0.3 | 1 |
| `buildOpenness` | 1.7 | 1.7 | 0.2 | 0.1 | 2 |
| `buildGhostOverlaySampled` | 1.1 | 1.3 | 0.3 | 0.2 | 4 |
| velocity MG build | 0.4 | 0.3 | 0.3 | 0.3 | — still serial |
| `findPocketCells` | 0.2 | 0.2 | 0.2 | 0.2 | — still serial |
| device assembly + uploads | 0.1 | 0.1 | 0.1 | 0.1 | — |
| **total** | **10.5** | **10.9** | **2.4** | **1.7** | |

Totals by depth (µs/leaf at 1 / 8 / 16 threads): depth 6 — 10.0 / 2.5 / 2.0; depth 7 —
10.9 / 2.3 / 1.7; depth 8 (7.01M leaves) — / 2.4 / 1.8, i.e. `setSolid` in **16.9 s** at 8
threads where the §1 projection had 95.6 s serial.

The ~4% drift at 1 thread (10.5 → 10.9) is the two-pass restructures paying a second walk
(the C/F uf slot count, the MG c2p split); it is repaid by ×2 threads.

Every rung passed all four gates: 149/149 ctests on `build_komp3` and `build_kcuda2` (148 + the
new `host_parallel`), depth-6 fluid mask bitwise vs `mask6_before.npy`, `set_solid` + 200 steps
bitwise at 1 vs 16 threads, and the sampled overlay's build census unchanged to the last count.

**Where the next µs are.** Loop parallelism is spent: the two largest remaining items are the
serial velocity-MG build (0.3) and `findPocketCells` (0.2), and everything else is ≤0.5. The
next lever is device-resident assembly (`archive/amr_device_assembly_plan.md`, which D3 defers to) and,
for cluster runs, F1's distributed resolver.

## 4. What Opus must know (traps, verbatim from the handoff + this campaign)

- OpenMP batteries: `OMP_NUM_THREADS=8 OMP_PROC_BIND=false` — 48 unbound threads = an hour.
- Never modify classic-path kernels/builders' semantics; the "call both, empties no-op"
  pattern is contractual.
- `.sdf-campaign-probes/flow_probe.py` runs at 4 threads with a pinned u_sum — do not "fix" its
  thread sensitivity, it is a pre-existing host-reduction property, not a regression.
- Speedups do not multiply across baselines; measure each rung against its predecessor.
- Push after gated rungs: submodule first, umbrella last (dangling-pointer check in
  [[push-directly-to-main]]).

## 5. ESCALATION (stop and hand to Fable — do not improvise)

1. Any builder loop that writes to another leaf's slots, carries state across iterations, or
   needs an atomic/reduction to parallelize — stop, document the loop and the dependency in
   THIS file under a "## Findings" heading, leave that loop serial, continue other rungs.
2. Any bitwise gate failure (thread-count variance, mask diff, parity ctest) that is not an
   obvious ordering bug you can fix by restoring serial order — stop, record the failing gate,
   the diff magnitude, and the rung.
3. Any temptation to change WHAT a builder computes (gates, sample positions, row content) —
   forbidden outright; if a rung seems to require it, that rung is mis-scoped: stop.
4. Anything touching `geom/` or the distributed path's logic — out of scope; stop. Kokkos
   configuration: the ONE sanctioned change is rung 0's `Kokkos_ENABLE_OPENMP=ON` on the
   nvidia-cuda bootstrap variant; anything else (arch flags, other backends, other prefixes) —
   stop.
5. Rung 0's fence failing — any flow/dem probe or ctest moved by the prefix rebuild alone —
   stop IMMEDIATELY, before any builder work: that is a suite-wide interaction Fable (and
   possibly the flow/dem owners) must look at, not something to patch around.
Escalations go in "## Findings" here + the memory file (`amr-mixed-level-cut-band-plan`), so a
Fable session picks them up with context.

## Findings

### F1 (rung 3, 2026-08-29) — the DISTRIBUTED seam resolver is not thread-safe: `AmrCutCell::build` stays serial on multi-rank

**The loop.** `AmrCutCell::build` Pass 1 (`cut_cell.hpp`), `nb_[6i+k] = neighbor(i, k)`.

**The dependency.** `neighbor()` → `AmrPoisson::periodicNeighbor` → `AmrPoisson::probeSlot`, and
`probeSlot` calls `extResolve_(p)` for any coord outside the block. In a distributed build that
callable is `LeafHalo::resolveGlobal` (`leaf_halo.hpp:136` → `resolve`), which is **non-const and
mutating**: an unknown coord is `misses_.emplace(gc, kPending)`'d into the halo's miss map. That
map is what drives the discovery fixpoint in `AmrFlow::prepareDistributed`
(`for (;;) { build; if (dhalo_.resolveMisses() == 0) break; }`). So the loop body has hidden
shared mutable state — the plan's §5.1 trigger, arriving through a callback rather than through
the loop's own indices.

**The measurement.** With Pass 1/Pass 2 parallel unconditionally, `build_kcuda2`'s
`amr_distributed_momentum_np4` hung: >21 minutes wall (the same test passes in ~0.5 s), four
`prterun` ranks alive and spinning, no output. Consistent with a corrupted `std::map` under
concurrent `emplace`. Single-rank gates (mask parity, thread invariance, both host batteries'
serial tests) were all green — the race is only reachable through the seam.

**What was done (not a workaround for the numerics — a scope boundary).** `build` now dispatches
its two leaf loops through a local `forLeaves`: `hostParFor` when `extResolve_` is null
(single-rank — `probeSlot` is then a pure wrap + `BlockOctree::find`), the plain serial `for`
when a resolver is installed. The distributed path therefore executes *exactly* today's code,
which is what §D4/§5.4 require; multi-rank setup gets no speedup from this rung.

**What a Fable session should decide.** Making the distributed path parallel needs the resolver
to be parallel-safe. Two shapes, neither attempted here:
1. *Split discovery from use.* Keep one serial probe-collecting round (the fixpoint's purpose)
   and make the FINAL round — the one that actually fills `nb_`/`sdfC_` with every ghost already
   resolved — use `LeafHalo::lookup` (already `const`, throws on an unknown coord) instead of
   `resolve`. The final round is the expensive one, so this recovers nearly all of the rung.
2. *Per-thread miss buffers* merged after the loop. Simple, but the miss set's iteration order
   feeds `resolveMisses`'s collective — the merge would have to restore a canonical order or the
   ghost-slot numbering could differ from the serial one across ranks.
Note the same resolver is reachable from `AmrPoisson::forEachFaceNeighbor` / `forEachFaceFull`,
so it constrains any future parallelization of a builder that walks faces on the distributed
path. Rung 2's `buildOpenness` is safe because the distributed `openFn`s are pure functions of
the face centroid and the loop itself reads only `t_->bounds`.

(Two clauses of this paragraph have since expired. Rung 4's `setGhostSampled` lost its
single-rank guard on 2026-08-30, rungs D0–D2; the rung-1 C/F builders lost theirs on 2026-09-21 —
see the next section, which is also where the "they never run multi-rank" reasoning for them is
replaced.)

## 7. The C/F quadratic scheme across the seam (2026-09-21)

`AmrFlow::setSolid` used to throw for `dist_ && cfScheme_ != CfScheme::standard`. It no longer
does: the four `buildCf*Delta` builders discover their ghost neighbours inside
`prepareDistributed`'s miss-collect fixpoint, and the real build stays where it was — in
`setSolid`, after the fixpoint, by which point every ghost is resolved. This is the shape rungs
D0–D2 established for the sampled overlay.

**What the closure reaches for.** For one directed 2:1 sub-face, `cfAppendStencil`
(`cf_scheme.hpp`) substitutes the coarse-side value with an interpolation of the coarse cell at
the fine cell's tangential position. Per tangential axis `tt` and each direction it needs

1. the coarse cell's tangential neighbour, `ap.periodicNeighbor(coarse, tt, dir)`; and
2. when that neighbour is one level FINER, the `2^Dim` children covering the coarse-size region
   across that face.

Probe (1) already went through `AmrPoisson::probeSlot` and therefore through the resolver. Probe
(2) called `BlockOctree::find` **directly**, on a coordinate it had wrapped modulo
`t.brick() * 2^lmax` by hand — and that is the defect the guard was hiding. Single-rank the block
IS the domain and the hand wrap is the correct period. Multi-rank it is not: the wrap folds a
child belonging to a neighbouring rank back into this rank's own block, and `find` returns a real
but geometrically unrelated leaf. Either the cover guard rejects it on level or fluid and the
tangential side reverts silently to the raw coarse value, or it passes and the stencil reads a
cell on the far side of the block. No crash, no missing ghost, no message — a
decomposition-dependent wrong answer at the seams, measured at **131 % relative error** at
np = 2/4 (the negative-control table under **Gate** below). Both probes now go through `ap`
(`loOf` / `levelOf` / `probeSlot`), which also makes the builders safe when `coarse` or `fine` is
itself a ghost slot; `t.level(j)` on a ghost `j` was reading past the octree's level array in all
four builders.

**The discovery arm** is `AmrFlow::probeCfScheme` / `probeCfTangential`, run in the fixpoint
after the FaceGeom face sweep when `cfScheme_ != standard`. It is a light, **gate-free**
traversal rather than the builders in a probe-only mode. The builders' row and fluid predicates
come from `mom_`, which is mid-build during the fixpoint (and on the sampled path is not built at
all until after it), so a probe sequence short-circuited by a gate that is not yet valid would
leave a coordinate undiscovered — and `LeafHalo::resolve` throws on an unknown coord after
`finalize()`. Issuing the full probe set unconditionally is a strict superset of anything the
gates can ask for, and it is much cheaper than four CSR builds per round. It costs no extra
ghosts: a child coordinate landing inside a same-level or coarser neighbour canonicalizes to that
neighbour's own leaf anchor in `resolveMisses`, so the registry gains map entries, never slots.
One traversal covers all four builders — `buildCfLapDelta` walks `mom_.lap()` and the other three
walk `pres_`, but both are `AmrPoisson` over the same octree with the same resolver and ghost
list, and they enumerate the identical neighbour set. It is host-parallel on the same argument as
the face sweep (F1): the miss registration is mutex-guarded and the miss SET is coord-keyed, so
ghost-slot numbering stays canonical under any schedule.

**Cost.** Measured on `tests/test_amr_distributed_cf_mpi` (4³ roots, `lmax = 3`, graded ladder
around a sphere) with `PECLET_AMR_PROFILE_SETUP=1`, at np = 2 on 5902 leaves,
`OMP_NUM_THREADS=1`, against the same run with the scheme set to `standard` (`CF_STANDARD=1`,
the test's control knob):

| | rounds | ghosts (per rank) | `setSolid`, slowest rank |
|---|---:|---:|---:|
| standard (arm off) | 4 | 901 / 1060 | 0.094 s |
| quadratic (arm on) | 4 | 911 / 1060 | 0.195 s |

**The arm costs no extra rounds** — its probes resolve inside the rounds the existing probers
already need — and 10 extra ghosts out of 911 on one rank, none on the other. Of the ~100 ms
difference in `setSolid`, ~58 ms is the C/F overlay CSR build itself (intrinsic to the scheme and
present single-rank too) and ~45 ms is the discovery arm spread over the four rounds. The box was
heavily loaded during the measurement, so read these as indicative ratios, not as absolutes.

**Apply side.** The `cfApply*` consumers read `f(sl(k))` by slot, so a CSR carrying ghost slots
needs only an `nExt_`-sized source with a synced ghost tail. Five of the six call sites already
had one. The sixth did not: the momentum delta `cfApply(cfMom_, u_[c], bmom_)` reads `u` at the
coarse cell's tangential neighbours, and with advection OFF nothing had refreshed `u`'s ghost
tail since the previous step's projection — the same position the sampled momentum seam delta was
in (D2). The `syncVel()` guarding that case now covers the C/F scheme too.

**Gate.** `tests/test_amr_distributed_cf_mpi.cpp` (ctests `amr_distributed_cf_np{1,2,4,8}`):
WORLD vs SELF leaf by leaf through the global Morton code, bitwise at np = 1 and in the
decomposition-independence class at np > 1, plus the positive check
`AmrFlow::numCfGhostColumns() > 0` — the count of C/F overlay entries that read a ghost slot.

Two things about that test are load-bearing and were found the hard way.

*The sphere is OFF-CENTRE, at (0.41, 0.57, 0.5).* The first version put it at the box centre, and
the negative control (child enumeration restored to the hand-wrapped `find`) **passed** — a gate
that could not fail on the bug it exists for. The reason is that a centred sphere makes the mesh
mirror-symmetric about every plane the ORB cuts the box with, so a coarse cell's tangential
neighbour across a seam is its own mirror image and therefore always the SAME level: the
finer-neighbour child branch is entered 24960 times and not once does a child coordinate leave
the block. Off-centre, 13800 child coordinates leave the block at np = 2 and every one resolves to
a ghost slot.

*`numCfGhostColumns() > 0` is a liveness check, not the gate.* It is nonzero in the broken build
too, because it also counts the columns the tangential-neighbour probe contributes and that probe
was never broken. The gate is the WORLD-vs-SELF difference. On the off-centre geometry the
negative control gives, against a ceiling of 5e-6 relative:

| | np = 1 | np = 2 | np = 4 |
|---|---:|---:|---:|
| fixed | 0 (bitwise) | 1.61e-7 | 1.10e-7 |
| negative control (hand-wrapped `find`) | 0 (bitwise) | **1.31e+0** | **1.31e+0** |

Note the np = 1 column: the defect is invisible there by construction, which is why a
WORLD-vs-SELF check at np = 1 alone would not have been a gate at all.

**The sampled band combined with the quadratic scheme** — the configuration the gate-free
argument is really about, since `mom_` is not built inside the fixpoint at all on that path — is
not a registered ctest (the gated configuration is the classic overlay, which keeps the run
cheap) but is one command away, `CF_SAMPLED=1`. Measured: np = 1 bitwise, np = 2 rel 1.611e-07,
np = 4 rel 1.104e-07, i.e. the same numbers as the classic overlay, because on a uniform finest
band the sampled overlay is identity slots. The fixpoint does change — 4 rounds / ~1000 ghosts
becomes 6–7 rounds / 2000–5200 — and that is the least-squares clouds' reach, not the C/F arm's.
