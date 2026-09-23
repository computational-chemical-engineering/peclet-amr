# Coarse-level redistribution (telescoping, repartition, replication): what lives in `core`, what stays in the methods

> Layering decision, 2026-09-23, Fable pass on `c1-advective-cost`. Written to become a register
> entry (`suite/docs/DECISIONS.md`) and to move to `suite/docs/` when it lands. Companion to
> `docs/amr_mg_depth.md` (C1) and `suite/docs/archive/MG_TELESCOPING_PLAN.md` (flow's design).
> No production code.

## 1. The decision (register form)

**The machinery that moves one level of a multigrid hierarchy onto a new decomposition of that
level's grid — choosing the target decomposition, the communicators and membership, and the
repeatable movement of fields down and back up — lives in `peclet::core::decomp`. What a level
is, how its operator and openness are rebuilt on the target, the transfers, the smoother, the
bottom solver and the V-cycle orchestration stay in each method.**

Rejected alternatives:

- *Per-method private implementations.* Three exist or are forming today: flow's `Telescope`
  (`mac_cutcell_mg.hpp:331–743`, default-on, measured flat at 1536 ranks), amr's `MgStage` /
  `ReplicatedTailStage` (`include/peclet/amr/mg_stage.hpp`, C1 WO4), and voro's coming need at
  scale. The plan that introduced `agglomerated()` already named "dem/voro/AMR" as consumers.
- *A core "multigrid stage" that owns the continued hierarchy below the stage point.* Core cannot
  name a `CutcellMG::Level`, a `BlockOctree` or an `AmrPoisson`, and the suite's settled decision
  is that V-cycle orchestration and transfers are deliberately **not** consolidated. The seam is
  below the hierarchy, not through it.
- *Leaving flow's telescoping as is and serving only the new consumers from core.* Rejected
  because flow is the **first production consumer of the missing piece** (§6): load-balanced
  CFD-DEM runs on a weighted decomposition, which its sibling-merge telescoping cannot handle.

The test for "belongs in core" is the one `ARCHITECTURE.md` states: core provides *where data
lives and how it moves*, not *how the physics is integrated*. Everything in §3.1 can be named in
core's vocabulary (`BlockDecomposer`, `Block`, `MPI_Comm`, `T*`, an index functor); nothing in
§3.2 can.

## 2. Why now

Three findings from the tree, 2026-09-23:

1. `gatherGroups` / `scatterGroups` — the plan's "core share" (`MG_TELESCOPING_PLAN.md:259`) —
   were never written. Neither core nor flow has them; flow's `Telescope` carries its own
   `MPI_Comm_split` pair (`:775–776`) and `Gatherv`/`Scatterv` (`:1486, 1554, 1602`), and amr's
   `ReplicatedTailStage` has just written its own `Allgatherv` keyed by global id.
2. Sibling-merge telescoping restores parity only on a **proportional** ORB tree (split values
   halve with depth). On a **weighted** tree the search over `agglomerated(d)` finds no liftable
   `d > 0` and falls to `d = 0`: the whole level on one rank.
3. flow's own comment (`mac_cutcell_mg.hpp:513–517`): with a weighted `dec0` "the coarse-level
   transfer is only clean when `nLevels == 1` (pure RB-GS) — use that (or the decomposition-
   agnostic GraphAMG) for a weighted co-decomposition." Load-balanced CFD-DEM is therefore
   running its pressure solve as a one-level smoother or a full-grid algebraic solve today, and
   with telescoping default-on a weighted level 0 is `blocked` → `d = 0` → the fine grid is
   gathered to one rank every V-cycle. **This needs a measurement before it is called a defect
   (§9.1), but the code says it.**

## 3. The layering

### 3.1 Generic — goes to `core/decomp/`

| piece | what it is | exists? | consumers |
|---|---|---|---|
| target decomposition of a level's grid | `BlockDecomposer` on `G_L`: `agglomerated(d)`, a fresh proportional `init(np_L, G_L)`, one block | yes | flow, amr |
| **target policy** | pure function of `(currentDec, G_L, liftable-predicate, minExtent)` → `{kind, dec, ownerOf, groupOf}` | flow inline (`:745–770`); amr would copy | flow, amr |
| **stage communicators** | `MPI_Comm_split` into group comm + roots sub-comm (or first-`np_L` sub-comm), `active`, member lists, counts/displs | flow inline (`:775–800`); amr partial | flow, amr, voro (comm part) |
| **planned field movement** | build once from `(srcDec, dstDec, ownerOf, srcIndex, dstIndex, comm)`; `forward(fields)` / `backward(fields)` many times; kinds: replicated (`Allgatherv`), nested group (`Gatherv`/`Scatterv`), general (planned point-to-point) | only the one-shot, same-rank-numbering, box-layout `redistributeGridFields` | flow, amr |
| weighted + aligned ORB | `init(numBlocks, globalSize, weights, align)` — coarse-first on the weight grid (`DECOMPOSITION_AND_MULTIGRID.md` open problem 9) | no | flow (CFD-DEM), amr `rebalance`, coupling's shared factory |
| id-keyed replicated gather | `gatherByGlobalId(ids, values, comm)` → the global vector | flow inline (`gatherv` in the bottom); amr inline | flow, amr, voro (GraphAMG bottom) |

### 3.2 Method-specific — stays where it is

The level record and its operator (`CutcellMG::Level` with `AC…AT, ox, oy, oz`; amr's
`BlockOctree` + `AmrPoisson` + `FvOp` + `LeafHalo`); rebuilding the operator on the target block
(flow: move `ox/oy/oz`, `coarsenOpenAvg`; amr: move α rows, `Multigrid::buildRaw`); the lift /
coarsening rule that defines "liftable" (flow: per-axis even origin and size on coarsenable axes;
amr: all axes even, §6.2 of `amr_mg_depth.md`); restriction, prolongation, smoother, bottom
selection; the V-cycle and where in it the stage fires; the ladder prediction (`predict`), which
composes core's policy with the method's lift rule.

### 3.3 What only looks shared

- **The continued hierarchy below a stage.** amr's `MgStage` exposes `numLevels`, `cycle`,
  `setBottom`, `targetX` — those are the *ladder's* surface and stay in amr. In core terms a stage
  is only "target + comms + movement"; `MgStage` composes that with a `Multigrid` (replicated) or a
  sub-communicator `DistributedFlowMultigrid` (sibling / repartition).
- **voro.** Its coarse levels are graph aggregates, not grid levels; a box-keyed plan does not
  apply. voro shares the communicator bookkeeping and the id-keyed replicated gather, nothing
  else. Do not generalise the box plan to ids on voro's account.
- **dem.** Shares the decomposition (and will benefit from the aligned weighted ORB), consumes no
  stage.

## 4. Core API sketch

All in `peclet::core::decomp`, header-only, MPI + optional Kokkos as the rest of `decomp/`. Names
are proposals for the caller to settle against `NAMING.md`.

```
// --- policy: a pure function replicated on every rank, no communication -------------------
enum class StageKind { InPlace, SiblingMerge, Repartition, Replicated };

template <int Dim> struct StageTarget {
  StageKind kind;
  BlockDecomposer<Dim> dec;      // decomposition of the SAME level grid G_L
  std::vector<int> ownerOf;      // target block -> rank in the parent communicator
  std::vector<int> groupOf;      // parent block -> target block (SiblingMerge only; else empty)
};

// `liftable(dec)` is the METHOD's predicate ("every block can coarsen in place one more time").
// ORDER (corrected in §11 — the first draft's order could never reach Repartition, because the
// one-block candidate agglomerated(0) is always liftable):
//   1. InPlace if liftable(cur) (and not tooSmall);
//   2. the sibling search S = largest d with liftable(agglomerated(d)), fewer blocks, and the
//      extent rule (flow's search, verbatim — S1's own function);
//   3. S is ACCEPTED iff maxBlockCells == 0 (flow's byte-identical mode) or the largest target
//      block of S has <= maxBlockCells cells;
//   4. else Repartition: a proportional ORB of G_L on np_L ranks [0, np_L) (§11.2), if liftable;
//   5. else S if it exists; else Replicated.
// `maxBlockCells` is the caller's "a rank never holds more cells of a coarse level than it holds
// of its finest level" — the finest level's largest block; 0 disables Repartition.
template <int Dim, class Liftable>
StageTarget<Dim> chooseStageTarget(const BlockDecomposer<Dim>& cur, const IVec<Dim>& G_L,
                                   Liftable&& liftable, int minExtent, Index maxBlockCells = 0);

// --- communicators + membership ----------------------------------------------------------
struct StageComm {
  MPI_Comm parent, group, sub;   // group: this rank's target block's members; sub: owners only
  bool active;                   // this rank owns a target block
  int myTargetBlock;             // or -1
  std::vector<int> members;      // parent ranks in group-comm order (owner first)
};
template <int Dim> StageComm makeStageComm(MPI_Comm parent, const StageTarget<Dim>& t);

// --- planned, repeatable movement of fields at resolution L -------------------------------
// srcIndex(gcell) / dstIndex(gcell): the consumer's local slot of a global cell of its own
// block (flow: the padded flat formula; amr: a table built once from the octree). Built once;
// forward/backward run every V-cycle with no handshake: Allgatherv (Replicated), group
// Gatherv/Scatterv (SiblingMerge — deterministic, bitwise the same permutation flow does today),
// or a fixed point-to-point pattern from the box intersections (Repartition; the one-shot
// redistributeGridFields keeps its role for rebalancing, this is its planned twin).
template <int Dim, class T> class RedistributePlan {
 public:
  template <class SrcIndex, class DstIndex>
  void build(const BlockDecomposer<Dim>& src, const StageTarget<Dim>& dst, const StageComm& c,
             SrcIndex&& srcIndex, DstIndex&& dstIndex);
  void forward(const std::vector<const T*>& src, const std::vector<T*>& dst);   // L -> target
  void backward(const std::vector<const T*>& dst, const std::vector<T*>& src);  // target -> L
};

// --- the weighted, coarse-first ORB (open problem 9) --------------------------------------
// Sum `weights` onto the align-grid, run the weighted ORB there, refined(align) back: blocks
// are multiples of `align` by construction, so coarsened()/lifting nest for log2(align) levels.
// Reduces bit-exactly to init(numBlocks, globalSize, weights) when align == 1.
template <int Dim>
void BlockDecomposer<Dim>::init(std::size_t numBlocks, IVec<Dim> globalSize,
                                const std::vector<Real>& weights, const IVec<Dim>& align);

// --- the trivial one everybody has inline ---------------------------------------------------
template <class T>
void gatherByGlobalId(const std::vector<long long>& ids, const std::vector<T>& vals,
                      long long nGlobal, std::vector<T>& out, MPI_Comm comm);
```

Layout contract: `RedistributePlan` never assumes a layout; it calls the index functors. That is
what lets amr's Morton-ordered levels and flow's padded boxes use one plan without a scratch copy.
Host-staged first (buffers are the level's cells, small); device-resident later exactly as
`GridHalo` did it.

## 5. What each consumer keeps and what it drops

**flow.** Keeps `Telescope` as its per-level record, `restrictAvg`, `prolongAdd`, `coarsenOpenAvg`,
the smoother, `GridHalo`, the bottom. Drops its inline depth search (→ `chooseStageTarget` with
flow's per-axis predicate), its `MPI_Comm_split` pair (→ `makeStageComm`), and its three
`Gatherv`/`Scatterv` bodies (→ one `RedistributePlan`). Gains the `Repartition` kind for a
weighted `dec0` (§6) and the aligned weighted ORB for `rebalanceByWeights`.

**amr.** Keeps `MgStage` (the interface is right: it is the composition of a core stage with a
continued ladder), `buildRaw`, the lift rule, `predict`. `ReplicatedTailStage` drops its gid maps
and `Allgatherv` (→ plan, `Replicated` kind, bitwise the same values). WO4b's sibling-merge and
repartition stages are `chooseStageTarget` + `makeStageComm` + plan + a `DistributedFlowMultigrid`
continuation on `sub`.

**coupling.** `rebalance()` builds the shared decomposition through the aligned weighted `init`
(the alignment is decided by the flow side's imbalance budget; dem sees only the split positions).

**voro.** Later: `makeStageComm` + `gatherByGlobalId` under its GraphAMG bottom.

## 6. Load-balanced CFD-DEM, worked through

Setup: flow + dem share one `BlockDecomposer` (settled). `coupling.rebalance()` →
`flow.rebalance_by_weights(w)` → `Solver::rebalanceByWeights` builds `BlockDecomposer(size, G, w)`
(`flow_ibm_mpi.hpp:149`), `redistribute(newDec)`, `initMpi(newDec)` → `CutcellMG::initMpi(...,
dec0 = weighted)`.

**Today — measured 2026-09-24** (`flow/tests/study/weighted_dec0_telescope_probe.py`, flow
`0ae29d6`; table in `docs/SCALING_ISSUES.md` #2 TRAP). Level 0 is the weighted partition; `evenOn`
fails on some axis → `blocked`, and the `agglomerated(d)` search walks down the ORB tree and lands
at **the shallowest depth whose splits are all even** — which is `d = 0` (the whole level on one
rank) whenever the root split falls on an odd plane, and 2^d ranks or a collapse one level lower
otherwise. 96³ at np = 8 with a particle-heap weight: the telescope fires at level 0 with `d = 0`,
L1–L5 run on one rank, the projection is 2.25–2.6× slower and **iterations are unchanged** (8 → 8):
it is the same hierarchy on one rank, a cost in time, not in convergence. **The two escapes the
source comment recommends do not escape**: the GraphAMG bottom leaves the level-0 telescope
untouched (projection 0.085 → 0.218 s), `nLevels = 1` with the `auto` bottom runs a redundant
GraphAMG on the whole grid (~55× slower, before and after), and `nLevels = 1` with the smoother
bottom is slower than the collapsed telescope it replaces. S5 must correct the comment at
`mac_cutcell_mg.hpp:513–517`.

**What the measurement changes in the plan below:** the search is not "no liftable `d > 0`" in
general — it lifts until the shallowest odd split. So **aligning the weighted ORB (S2) directly
deepens where the telescope lands**, level for level, before any repartition stage is needed. That
raises S2's leverage relative to the repartition kind, and the order below stands.

**With §4.** Two pieces, in this order of leverage:

1. *Aligned weighted ORB* in `rebalanceByWeights`, `align = 2^a` with `a` chosen by the existing
   imbalance budget (1.05). flow's balancer cells are *fine cells* (blocks of ~24–48 per axis at
   384³/1536, hundreds at ≤ 384 ranks), so a quantum of 2–8 cells costs little: expect `a = 1–2`
   at 1536 ranks, `3–4` at ≤ 384. That buys `a` nested in-place levels and shrinks everything
   below by `8^a`. (amr is the opposite case — its balancer cells are bricks, ~8 per axis per
   rank, so alignment is expensive there and `a` stays 0–1; see `amr_mg_depth.md` §5.6.)
2. *Repartition stage* at the level where the aligned tree blocks: a fresh proportional ORB of
   that level's grid on `np_L` ranks, one planned point-to-point per direction per V-cycle. Volume
   per rank: `cells/rank / 8^a` doubles each way — 37k at `a = 0` (comparable to a few halo
   exchanges), 4.6k at `a = 1`, 580 at `a = 2`. Below it the tree is proportional and flow's
   existing sibling-merge telescoping runs to 3³ on one rank exactly as it does on the unweighted
   ladder — iteration counts become rank- and rebalance-independent.

Neither piece alone suffices: alignment cannot buy full depth at bounded imbalance on a weighted
tree; repartition at `a = 0` moves the whole fine grid per V-cycle. Together they are the general
form of what `MG_TELESCOPING_PLAN.md` open problem 9 asked for, and they are consumed by flow's
production path first. **This raises the priority of the core work above amr's WO4b's own need
for it.**

## 7. flow's existing telescoping — recommendation

**Migrate behind a byte-identity gate, in two commits; change no policy and no default.**

- The values moved by a `Gatherv` and by a planned point-to-point over the same box intersections
  are the same numbers in the same slots; `chooseStageTarget` is flow's search lifted verbatim with
  the predicate parameterised. So byte-identity is the *natural* gate, not an aspiration: `flow`'s
  `test_telescope_mpi` (forced telescope vs control 1e-14, starved partition vs single-rank
  2.5e-14), `cutcellmg_mpi`, `sdflow_mpi` at np = 1/2/4 — all byte-identical before and after; the
  384³ ladder table reproduced by `predict_hierarchy` at 24…1536.
- "Leave it and serve only the new consumers" is rejected in §1: flow needs the `Repartition` kind
  for CFD-DEM, and adding it privately would be the fourth copy.
- "Core-ify only comm + gather, leave the policy" is the fallback if the byte gate resists: the
  policy is the smallest and least risky part to move, so if anything stays it should be the
  movement, not the search. Default: move both.

## 8. Sequencing against WO4b, and ownership

WO4b has not started. Aim it at core:

| step | where | content | gate |
|---|---|---|---|
| S1 | core | `chooseStageTarget`, `makeStageComm`, `RedistributePlan` (Replicated + SiblingMerge kinds), `gatherByGlobalId`; unit tests np = 1…8 with **flow's inline code and amr's `ReplicatedTailStage` as the two reference implementations the tests must reproduce bitwise** (a Kokkos-free host test: box grids, random fields, `backward(forward(x)) == x`, nested == `Gatherv` values, replicated == `Allgatherv` values) | core ctests green; core tagged before any consumer (directive) |
| S2 | core | `RedistributePlan` general kind (planned point-to-point) + aligned weighted `init` | plan test on a weighted partition; `init(w, align=1)` bit-identical to `init(w)`; `coarsened()` nests for `log2(align)` levels on a weighted tree |
| S3 | amr (WO4b) | `ReplicatedTailStage` movement → plan (bitwise vs WO4's tail test); sibling + repartition stages on `sub`-comm `DistributedFlowMultigrid` | `amr_mg_depth.md` WO4b gate: weighted 24³-brick partition, np = 2/4/8, single-rank ladder and solution to ≤ 1e-13, np-independent iterations |
| S4 | flow | `Telescope` delegates to S1 (policy + comms + movement) | byte-identical (§7) |
| S5 | flow + coupling | `Repartition` kind for weighted `dec0`; `rebalanceByWeights` and `coupling.rebalance()` through the aligned weighted `init` | the CFD-DEM MPI tests; iteration count after `rebalance()` equal to before it ± 1; the §9.1 measurement closed |
| S6 | suite | register entry (§10), `ARCHITECTURE.md` core-module list gains the stage line, `MG_TELESCOPING_PLAN.md` status note | — |

S1 is small (~300–400 lines plus tests) and unblocks S3 and S4 in parallel. Ownership: S1/S2 are
core changes and need this decision recorded first (a cross-cutting change); S3 is C1's; S4/S5 are
flow's and should be owned by whoever owns flow's telescoping, with the byte gate as the handover.
If S1 cannot land before WO4b's implementer is free, WO4b may start on amr-local copies of
`chooseStageTarget` and the plan **only if** they are written against §4's signatures so S3 is a
delete-and-include, not a rewrite.

## 9. Open questions, each with a default

1. **Is flow's CFD-DEM telescoping really collapsing to one rank after a rebalance? — ANSWERED
   2026-09-24: yes.** Measured at np = 4 and 8 (§6); flow has no other guard, and the collapse lands
   at the shallowest odd split of the weighted tree (`d = 0` when the root split is odd). S5
   remains the fix and its priority stands. Not yet measured: np ≥ 16, GPU, a coupled `CfdDem` run
   (the flow-only path is the same call).
2. **Alignment depth for flow's weighted ORB (fact).** *Default:* let the existing 1.05 budget
   choose; log `a` in `check_decomposition.py --predict`.
3. **Device-resident plan buffers (fact).** *Default:* host-staged; measure at 384 GPUs before
   moving them, as `GridHalo` was.
4. **Names (preference).** `StageTarget`, `chooseStageTarget`, `StageComm`, `RedistributePlan`,
   `gatherByGlobalId`. *Default:* these, unless `NAMING.md` already has a spelling for "plan"
   (`GridHaloTopology` is the precedent for "topology once, exchange many" — `RedistributeTopology`
   would follow it).
5. **Does `MgStage`'s interface survive S3? (preference.)** *Default:* yes — it is the right
   consumer-side shape (target + comms + movement composed with a continued ladder); only its
   movement bodies change.

## 10. Register entry (proposed text)

> **Coarse-level redistribution lives in core, hierarchies stay in the methods (2026-09-23).**
> The target decomposition of a multigrid level (sibling merge on the ORB tree, a fresh
> proportional ORB on fewer ranks, or replication), the stage communicators, and the planned
> up/down movement of level fields are `peclet::core::decomp` infrastructure with flow and amr as
> consumers (voro for the communicator and id-keyed gather). What a level is, its operator and
> openness on the target, transfers, smoother, bottom and V-cycle stay method-specific. Rejected:
> per-method private copies (three were forming); a core stage that owns the continued hierarchy
> (core cannot name a level, and V-cycle orchestration is deliberately not consolidated). Trigger:
> flow's sibling-merge telescoping cannot handle the weighted decomposition load-balanced CFD-DEM
> runs on (`mac_cutcell_mg.hpp:513–517`), and amr's C1 needed the same machinery. Evidence:
> `amr/docs/amr_mg_core_boundary.md`, `amr/docs/amr_mg_depth.md` §5.6, `MG_TELESCOPING_PLAN.md` §4.

## 11. S2 specification — the Repartition kind and the aligned weighted ORB

> Written 2026-09-24 after S1 landed (core branch `stage`: `stage_target.hpp`, `stage_comm.hpp`,
> `redistribute_topology.hpp`, `gather_by_global_id.hpp`; the policy reproduces flow's search on
> 4872 ladder levels bitwise, the movement matches flow's gather/scatter and amr's replicated
> gather bitwise) and after the measurement of §6. S1's Part A (`maxBlockCells == 0`, formerly
> `allowRepartition == false`) must keep passing unchanged.

### 11.1 The corrected policy order, and what excludes the one-block candidate

The first draft ordered *sibling search → Repartition → Replicated*, and the sibling search can
never fail: `agglomerated(0)` is one block at origin 0 of size `G_L`, liftable whenever the level
grid can halve. The measurement sharpens the picture: on a weighted tree the search lands at the
shallowest depth whose splits are all even, so what it returns is a *legal* sibling merge that
sheds far too many ranks for the level's size. Neither the depth `d`, nor the number of tree
levels merged, nor the shed *ratio* distinguishes that from flow's validated proportional ladder
(384³/1536 merges 1536 → 64, a 24× shed, and is fine): the discriminator is the **absolute size of
the block a rank would receive** — 12³ = 1 728 cells there, 96³ = 884 736 cells in the trap.

So the economic knob is one number with a physical meaning: **a rank never holds more cells of a
coarse level than it holds of the finest level.** The caller passes `maxBlockCells` = the largest
finest-level block (flow: level 0's `n_` Allreduced MAX; amr: the largest rank's root-brick cell
count `Π blockBrick · 2^(lmax·Dim)`, or its leaf count — the method's choice, replicated).

```
chooseStageTarget(cur, G_L, liftable, minExtent, maxBlockCells):
  1. tooSmall := minExtent > 0 && minBlockExtent(cur) < minExtent
     if nb <= 1 || (liftable(cur) && !tooSmall): return InPlace                 # S1, unchanged
  2. S := siblingMergeSearch(cur, liftable, minExtent)                          # S1's function, verbatim
  3. if S && (maxBlockCells == 0 || maxCells(S.dec) <= maxBlockCells): return SiblingMerge(S)
  4. R := repartitionTarget(cur, G_L, liftable, minExtent, maxBlockCells)       # §11.2
     if R: return Repartition(R)
  5. if S: return SiblingMerge(S)            # legal but heavy — the measured collapse, never wrong
  6. return Replicated
```

`maxCells(dec)` is the largest block's cell count. With `maxBlockCells == 0` steps 4–6 are never
reached from step 3 except through S1's existing fallthrough (no `S`), so Part A is unchanged
byte for byte. **The one-block candidate is excluded exactly when the level is larger than one
finest-level block** — which is precisely when replicating or collapsing it is a scaling defect,
and never at the true bottom (a 3³ or 6³ level is always accepted).

Worked against the §6 table (`maxBlockCells` = level-0 block, ≈ `96³/np`, up to 1.8× for the
weighted rows): heap np = 8 (`d = 0`, 884 736 cells > ~200k) → Repartition on 8; heap np = 4 →
Repartition on 4; flat bed np = 8 (`d = 2`, 4 ranks of 221k > 110k) → Repartition on 8; tilt-0.3
np = 4 at L1 (48³ = 110 592 ≤ 221k) → SiblingMerge accepted (measured cheap: 0.166 → 0.168);
flat bed np = 4 → InPlace (null). flow's 384³/1536 ladder: 1 728 ≤ 37k → SiblingMerge accepted —
**the validated proportional ladders are unchanged.**

### 11.2 `np_L` and the repartition target

```
repartitionTarget(cur, G_L, liftable, minExtent, maxBlockCells):
  np     := cur.numBlocks()
  cells  := Π_d G_L[d]
  npL    := clamp(ceil(cells / maxBlockCells), 1, np)          # as many ranks as the size justifies
  if minExtent > 0:                                            # the extent rule, when it is ON:
     cap := Π_d max(1, floor(G_L[d] / (2*minExtent)))          # blocks fat enough to survive the
     npL := min(npL, cap)                                      # halving that follows (flow's rule)
  # minExtent == 0 (flow's "economic trigger disabled"): no cap — npL is set by maxBlockCells only.
  for n in [npL, then the largest power of two <= npL, then halving]:
     R := BlockDecomposer(n, G_L)                              # proportional, unweighted, unaligned
     if liftable(R): return {kind = Repartition, dec = R, ownerOf = identity on [0, n), groupOf = {}}
  return none                                                  # step 5 / 6 take over
```

No division by zero: the extent cap is only formed when `minExtent > 0`, and `maxBlockCells > 0`
is the precondition of reaching this function. `np_L` never exceeds the current rank count (ranks
are only shed) and is at least 1. The liftability retry is what stops a stage from firing again at
the very next level: a proportional ORB of an even grid on a power-of-two rank count is liftable
under both flow's per-axis and amr's all-axes predicates; the loop is a pure, replicated function.
Owners are parent ranks `[0, np_L)` with `ownerOf[b] = b`, so S1's convention "parent rank `r`
owns target block `r`" holds for the active ranks unchanged.

`StageComm` for Repartition: `group = parent` (every rank takes part in the movement),
`sub = MPI_Comm_split(parent, rank < np_L ? 0 : MPI_UNDEFINED, rank)`, `active = rank < np_L`,
`myTargetBlock = active ? rank : -1`, `members = {}` (there are no groups). The continued
hierarchy runs on `sub` exactly as it does after a sibling merge — nothing new for the methods.

### 11.3 The planned point-to-point movement (`RedistributeTopology`, kind `Repartition`)

Shaped like S1's build: this rank's `srcSlots_` from `srcIndex` over its current block, per-segment
`dstSlots_` from `dstIndex`, byte-packed field-major per segment, and `forward` / `backward` as the
two directions of one fixed pattern. What changes is that the segments are **box intersections**
rather than whole member blocks, and the transport is point-to-point on `c.parent`:

- **Build (once).** For every target block `t` (owned by parent rank `t`): `I = my src block ∩
  dst.dec.block(t)`; if non-empty, a *send segment* `{dst = t, cells of I in x-fastest order via
  srcIndex}`. If active, for every source rank `s`: `J = dst.dec.block(rank) ∩ src.block(s)`; if
  non-empty, a *receive segment* `{src = s, cells of J via dstIndex}`. Every rank computes every
  intersection from the two decompositions alone — **no handshake, no NBX**: the pattern is known
  to both sides by construction, which is what makes it a topology rather than an exchange plan.
  The self-intersection (`t == rank`) is a direct copy, as in `redistributeGridFields`.
  Assertions: send segments tile my source block; receive segments tile my target block;
  `requireDistinct` on both slot lists (S1's).
- **`forward`.** Pack each send segment (field-major within the segment, S1's layout);
  `MPI_Irecv` every receive segment, `MPI_Isend` every send segment (one message each, `MPI_BYTE`,
  a fixed tag), self-copy, `MPI_Waitall`, unpack. **`backward`** is the mirror: active ranks send
  their receive segments back, every rank receives its send segments. Each cell has exactly one
  source and one destination, so the result is independent of message order — bitwise the same
  values as a one-shot `redistributeGridFields` over the same boxes (the test oracle, §11.6).
- **Tag.** One constant from the range core reserves below the AMR direct tags (11 / 41 / 45,
  `amr/CLAUDE.md`), offset by a per-topology `id` the caller passes at build (the level index) so
  two levels' stages in flight on the same communicator cannot pair messages across each other;
  `Waitall` before returning keeps the pattern serial in practice.
- **Later, not now:** persistent requests (`MPI_Send_init`) and device-resident buffers, exactly
  as `GridHalo` grew them; host-staged `Isend`/`Irecv` is the S2 form.

`gatherByGlobalId` and the SiblingMerge / Replicated paths are untouched.

### 11.4 The aligned weighted ORB — `init(numBlocks, globalSize, weights, align)`

**Contract.** `align[k] ≥ 1`, `globalSize[k] % align[k] == 0`, `weights` covers the global grid
x-fastest (`CONVENTIONS.md`). Result: a weighted ORB every one of whose split values, block origins
and block sizes is a multiple of `align[k]` on axis `k`, so `coarsened(align)` divides cleanly and
in-place lifting nests for `log2(align[k])` levels on every axis; `align_` is set to `align`.

**Construction — coarse-first, never snap-after** (`DECOMPOSITION_AND_MULTIGRID.md` §1.3, §2.4,
§2.5: snapping a chosen split cascaded 96|96 into 128|64; on the coarse grid one cell *is* the
quantum):

```
Gc[k]  := globalSize[k] / align[k]
wc[c]  := Σ weights over the align-box of coarse cell c        (x-fastest over Gc)
coarse := BlockDecomposer(); coarse.initImpl(numBlocks, Gc, &wc)   # the EXISTING weighted ORB, align_ = 1
*this  := coarse.refined(align)                                     # the EXISTING exact inverse of coarsened()
```

Both halves already exist: the weighted `initImpl` chooses each split on cumulative weight, and on
the coarse grid that cumulative weight is exactly the fine weight of the same boxes; `refined()`
scales splits, origins and sizes and sets `align_`. The unweighted aligned `init` (the snapping
one, `block_decomposer.hpp:394–400`) is left as it is — existing partitions stay byte-identical.

**Bit-exact reduction at `align = 1`:** `Gc = globalSize`; `wc[c]` is a one-term sum, so
`wc == weights` bitwise; `initImpl` is the same call; `refined({1,…})` multiplies integers by 1.
Hence `init(n, G, w, {1,…})` and `init(n, G, w)` produce identical `origins_`, `sizes_`, `tree_`
(gate G-A1).

**Choosing `a` (`align = 2^a` on every axis) from the imbalance budget — a pure, replicated
function** (`weights` is the global vector on every rank in both `rebalanceByWeights` and amr's
`rebalance`):

```
chooseAlignedWeighted(numBlocks, G, weights, budget = 1.05, aMax):
  aMax := min(aMax, min_k trailingZeros(G[k]), largest a with Π_k (G[k]/2^a) >= numBlocks and every G[k]/2^a >= 2)
  for a in aMax .. 1:
     D := init(numBlocks, G, weights, {2^a,…})
     imb := max_b W(b) / (W_total / numBlocks)              # weight imbalance, the balancer's own metric
     if imb <= budget: return (D, a)
  return (init(numBlocks, G, weights), 0)                   # today's partition — never worse than now
```

Cost: at most `aMax` ORB builds on grids shrinking by 8× each, once per rebalance — noise beside
the migration. Expected `a`: flow at 384³/1536 (blocks 24–48 cells per axis) `a = 1–2`; at ≤ 384
ranks `3–4`; amr's brick-granular balancer (`~8` bricks per axis per rank) `0–1`. Log `a` and
`imb` (`check_decomposition.py --predict`, amr's `predict`).

**Consequence for the measured trap.** The collapse lands at the shallowest odd split. With
`align = 2^a` every split is even for `a` lifts, so the telescope cannot fire above level `a`, and
whatever it then does happens on a level `8^a` smaller: on the §6 heap case with `a = 2`, level 2
is 24³ = 13 824 cells — below one rank's level-0 block, so even a collapse to one rank there is
accepted by §11.1 and costs nothing measurable. **S2a alone should return the probe to the
unweighted timing.**

### 11.5 Split S2 in two, in this order

| | content | why this order |
|---|---|---|
| **S2a — aligned weighted `init` + `chooseAlignedWeighted`** (core), then its use in `rebalanceByWeights` / `coupling.rebalance()` / amr `rebalance` (the consumers' halves belong to S5 / S3 but the core half lands first) | §11.4 | The measurement says the defect is *where the first odd split sits*, and alignment moves every split; it is a decomposition change with no new communication, and it fixes the measured flow case on its own (§11.4). |
| **S2b — the Repartition kind** (policy branch, `StageComm`, `RedistributeTopology` kind) | §11.1–11.3 | Needed where alignment cannot buy depth at bounded imbalance: amr's brick-granular balancer, `a = 0` forced by a very uneven weight, and large `np` where the budget yields `a = 1`. Also the correctness backstop that makes a rebalanced run never collapse a level larger than a rank's fine block. |

S2a first because it has the higher measured leverage and the smaller blast radius; S2b is not
optional — amr's S3 depends on it and it is what makes the policy honest under `maxBlockCells`.

### 11.6 Gates, with expected numbers

| gate | where | configuration | criterion |
|---|---|---|---|
| G-A1 reduction | core | 50 random weight fields, `np ∈ {1..8, 12, 24}`, grids 32³ / 48×32×16 / 96³ | `init(w, align=1)` == `init(w)`: `origins_`, `sizes_`, `tree_` bitwise |
| G-A2 nesting | core | same, `a ∈ {1, 2, 3}` | every block origin and size a multiple of `2^a`; `coarsened(2^a)` succeeds; after `j ≤ a` halvings every block passes BOTH flow's per-axis and amr's all-axes `liftable` |
| G-A3 chooser | core | same | returned `imb ≤ 1.05` whenever `a > 0`; `a` identical on every rank (Allreduce of a hash); `a = 0` returns today's partition bitwise |
| **G-A4 the probe, alignment only** | flow (S5a) | `weighted_dec0_telescope_probe.py`, heap np = 8 and 4, `rebalanceByWeights` through `chooseAlignedWeighted`, Repartition OFF | telescope fires no earlier than level `a` (log the ladder and `a`; expect `a = 2` at 96³/np = 8); projection ≤ **1.15×** the unweighted baseline (0.080 → ≤ 0.092 s at np = 8; was 0.180); iterations 8 → 8; momentum time within ±10 % of the weighted baseline (the cell imbalance is the balancer's, not ours) |
| G-B1 policy | core | the eight §6 trees exported as fixtures + S1's 4872-level fixture | with `maxBlockCells` = level-0 block: Repartition on 8 / 4 / 8 for the three `d = 0` rows and the flat-bed np = 8 row, SiblingMerge accepted for tilt-0.3 np = 4 at L1, InPlace for flat-bed np = 4; with `maxBlockCells = 0`: S1 Part A byte-identical |
| G-B2 movement | core | `np ∈ {1..8}`, random proportional and weighted `src`, proportional `dst` on `np_L ∈ {1, 2, np/2, np}`, 1–3 fields | `backward(forward(x)) == x` bitwise; `forward` == `redistributeGridFields` values on the box layout; inactive ranks receive nothing; send segments tile the source block, receive segments the target block |
| G-B3 the probe, repartition only | flow (S5b) | as G-A4 with alignment forced to `a = 0` and Repartition ON | Repartition at level 0 on all ranks; projection ≤ **1.3×** the unweighted baseline (one level-0 field exchange per direction per V-cycle); iterations 8 → 8 |
| G-B4 the probe, both | flow (S5) | defaults | ≤ 1.15× (alignment wins; repartition silent unless `a` is capped) — and the comment at `mac_cutcell_mg.hpp:513–517` corrected |
| G-B5 amr | amr (S3) | `amr_mg_depth.md` WO4b gate | weighted 24³-brick partition, np = 2/4/8: single-rank ladder and solution to ≤ 1e-13, np-independent iterations |

Byte-identity of everything that existed: `maxBlockCells = 0` and `align = 1` are the shipped
behaviours, so every existing MPI ctest in core, flow and amr must pass unchanged before either
default moves; moving a default (flow's `rebalanceByWeights` to the aligned init; amr's rebalance)
is its own commit with the G-A4 / G-B5 numbers in the message.
