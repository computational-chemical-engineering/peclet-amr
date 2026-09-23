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
// Order: InPlace if liftable(cur); else the largest d with liftable(agglomerated(d)) and
// minExtent(agglomerated(d)) >= 2*minExtent (flow's rule, verbatim); else a proportional ORB on
// np_L = min(cur.numBlocks(), cells(G_L) / (2*minExtent)^Dim) ranks [0, np_L) if liftable; else
// Replicated (one block; ownerOf = every rank).
template <int Dim, class Liftable>
StageTarget<Dim> chooseStageTarget(const BlockDecomposer<Dim>& cur, const IVec<Dim>& G_L,
                                   Liftable&& liftable, int minExtent);

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

**Today.** Level 0 is the weighted partition; `evenOn` fails on some axis → `blocked`; the
`agglomerated(d)` search finds no liftable `d > 0` on a weighted tree → `d = 0` → one rank
receives the whole level-0 residual each V-cycle and runs the rest of the hierarchy alone (or,
per the code comment, the user sets `nLevels = 1` / GraphAMG). At 384³ that is 56M cells to one
rank — the pressure solve is either serial or a smoother.

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

1. **Is flow's CFD-DEM telescoping really collapsing to one rank after a rebalance? (fact.)** The
   code says so; nothing has measured it. *Default:* before S5, run the coupling MPI test with
   telescoping on, read `predict_hierarchy` / the ladder print after `rebalance()`, and record it
   in `SCALING_ISSUES.md`. If the answer is "it selects `d = 0`", S5 is the fix; if flow already
   guards it some other way, S5 shrinks to the aligned `init`.
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
