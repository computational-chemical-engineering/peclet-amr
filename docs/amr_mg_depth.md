# The pressure multigrid below the root brick — design note (ROADMAP C1)

> Fable design pass, 2026-09-23, on branch `c1-advective-cost` (`f3a7ba0`, base `7d4dac9`).
> Answers `docs/briefs/mg_depth_below_root.md`. Implemented by an Opus engineer who has seen
> neither the brief nor the conversation; this note is meant to stand alone. No production code
> was written in this pass.

## 1. The decision, in one sentence

**A multigrid level below the root brick is the same `BlockOctree` with its root *lifted* — the
brick halved on every axis and `lmax` incremented, leaf codes untouched — so `coarsenIf` simply
keeps merging; the ORB blocks follow by nesting (lift in lockstep while every rank's block origin
and size are even), and when nesting stops before the coarsest global extent is ≤ 4 the coarsest
in-place level is gathered to every rank and the single-rank hierarchy continues *redundantly* from
there down to an exact bottom (damped Jacobi at extent ≤ 4, an agglomerated `GraphAMG`-PCG solve
otherwise).**

**The multigrid does not stop at the mesh of bricks — it continues *through* it.** A lifted level
IS the brick mesh coarsened: level `lmax + j` is the mesh of root bricks coarsened `j` times, with
the octree inside each coarse brick one level deeper. `64→32→16→8→4` in §6.2 is the brick grid
halving; `384→192→96→48` at 1536 ranks is the brick grid halving across ranks. Bricks remain the
unit of decomposition and load balance at every level (a coarse brick belongs to the rank that
owns its 2^Dim fine bricks), and the root brick mesh is meant to be *fine* — §5.4 rejects
coarsening it to help the solver. Everything downstream of the octree — `AmrPoisson::init`, the
area-averaged openness ladder, the covering-leaf `c2p`, the device face-CSR assembly, the
per-level `LeafHalo` — works on a lifted level verbatim. The ROADMAP's premise that "the coarse levels stop being octrees" is false, and
that is what makes the fix small.

## 2. Problem and scope

**Problem.** `AmrMultigrid::build` (`poisson.hpp:640`) and `DistributedFlowMultigrid::buildImpl`
(`distributed_flow_mg.hpp`) coarsen by merging complete sibling groups (`coarsenIf`,
`block_octree.hpp:217`). A root cell has no siblings, so the hierarchy stops at the root brick:
`levels = lmax + 1`, coarsest = `(cells / 2^lmax)³`. On a uniform mesh that is one level and no
coarse-grid correction; the step is 92 % pressure solve, and the emulated fix (the identical leaf
set inside a deeper tree) measures **2035.9 → 252.0 ms/step (8.1×)** on the uniform 64³ case,
saturating at four levels with an 8³ bottom.

**In scope.** The pressure hierarchy (`Multigrid` single-rank, `DistributedFlowMultigrid`
distributed; the host `AmrMultigrid` they build from; the Python `Poisson` class through the same
host builder): what a sub-root level is, its operator, transfers and openness, its ownership under
MPI, where the ladder stops, what solves the bottom, the prediction tool and diagnostics the tests
assert against, and the decomposition alignment that keeps a rebalanced partition nestable.

**Out of scope.** The velocity multigrid's `minCoarse` cap (§12 says how the answer carries over
and why it is its own work order); ROADMAP C2, B6, A/A′; `peclet.flow` itself; the host
`GradedDistributedMultigrid` / `DistributedMultigrid` oracles in `distributed_fv.hpp` /
`distributed_poisson.hpp` (untouched — §5.3 says why); sub-communicator telescoping (§11, deferred
with a trigger).

## 3. Constraints and invariants

- **Encoding (`block_octree.hpp:13–23`).** Leaf coordinates are fine units; a leaf at `level` L
  covers 2^L fine units per axis and its origin code has the low `L·Dim` bits zero; root cells sit
  at `level = lmax`; leaves are stored sorted by code. `brick[d]·2^lmax ≤ 2^Bits`. A lift keeps
  `brick·2^lmax` and therefore every code, every level byte and the leaf order.
- **What reads `lmax`/`brick` on the multigrid path** (grep, 2026-09-23): `coarsenIf`'s
  `L < lmax_` gate; `AmrPoisson::init`'s `fineExt_[d] = brick[d]·2^lmax` (`poisson.hpp:68`,
  the periodic wrap modulus — invariant); `LeafHalo` reads `blockBrick()·rootSpan()` from the
  `DistributedOctree`, not from the local octree (`leaf_halo.hpp:77` — invariant). `assembly.hpp`,
  `block_octree_view.hpp`, `fv_op.hpp`: nothing.
- **Octree cells are cubes in fine units.** A level cannot be `2h × 2h × h`; sub-root coarsening
  is isotropic by construction (consistent with decision AM1, `amr_anisotropic.md` §4). Semi-
  coarsening would need a second level type and is rejected (§5.5); anisotropic residue goes to the
  exact bottom.
- **Level 0 is untouched.** The finest level is the flow octree as today; only coarser *copies*
  are lifted. The level-0 face CSR, the quadratic C/F CSR, the ghost-projection overlay and the
  PCG all see bit-identical operators.
- **The pressure operator is singular** on every `AmrFlow` path: the three `presMG_.build` /
  `presMGD_.build` sites (`flow.hpp:824–851`) pass `periodic = true` and `setRemoveMean(true)`
  follows. flow's anchored-path caveat (`DECOMPOSITION_AND_MULTIGRID.md` open problem 4) does not
  arise here; if a non-periodic pressure path is ever added, the exact bottom must be gated the
  way flow gates it.
- **Parallel contract.** np = 1 bitwise vs single-rank; np = 2/4/8 in the ~3e-7 march class;
  every per-level operation that is a collective (halo build, `removeMeanFvDist`, the new
  Allreduce/Allgatherv) must be issued in the same order on every rank — the level counts are
  therefore equal on every rank (they are padded today, and the lift depth is Allreduced).
- **Byte gate.** `python/state_hash.py` keys that contain a pressure or Poisson solve WILL move
  (the preconditioner changes, so the PCG iterates change); the `Octree` topology keys MUST NOT.
  Re-record in its own commit (§10).
- **Determinism.** Jacobi, `restrictField`, `prolongAdd` are per-cell and order-independent; the
  redundant tail runs identical device kernels on identical data on every rank.
- **Performance envelope.** Sub-root levels are ≤ 1/8, 1/64, … of the root-grid level; their build
  and their V-cycle cost are bounded by 1/7 of the root level's. New per-V-cycle communication
  exists only when the tail engages (one `Allgatherv` of the coarsest in-place level).

## 4. What the literature and `flow` do, and where this design follows or deviates

**Textbook (Trottenberg–Oosterlee–Schüller ch. 6; hypre, MueLu, PETSc `PCTELESCOPE`).** A V-cycle
is mesh- and domain-independent only if its coarsest level is effectively solved. Two ways to get
there: coarsen geometrically until the coarsest grid is a handful of cells per axis, or stop early
and solve the bottom exactly (agglomerated direct/AMG). Under MPI the geometric ladder is bounded
by the per-rank block; the libraries redistribute coarse levels onto fewer ranks.

**`flow` (`mac_cutcell_mg.hpp:1821, 1857, 2056`; `DECOMPOSITION_AND_MULTIGRID.md` §1–2).**
Per-axis halving while every rank's block origin and size are even (`evenBlocks`); coarse levels
are the fine ORB `coarsened()` in place so transfers stay local; `auto` agglomerates when the
coarsest global extent exceeds 4 on any axis (singular path only): global CSR keyed by global cell
id, `Allgatherv` of the rhs each V-cycle, every rank runs the identical `GraphAMG`-PCG, scatter by
id — decomposition-independent by construction (np=6 vs np=1 to 4.5e-16). §2.7's table: an exact
bottom is depth-independent (4.0 iterations at 4 and at 8 levels) and *beats* full depth (69.5 vs
77.1 ms) because tiny levels cost more than the coarse solve they replace. Telescoping onto
sub-communicators (rung 2 of `archive/MG_TELESCOPING_PLAN.md`) is implemented, off by default,
and measured rank-independent iteration counts at 24–1536 ranks.

**This design follows flow on:** the extent-4 criterion; the exact bottom as an agglomerated global
CSR solved redundantly by `GraphAMG`-PCG (core's `peclet::core::solver::GraphAMG`, the same class
flow uses); the in-place nested ladder with local transfers; the "gather redundantly and continue
with the single-rank hierarchy" shape of telescoping rung 1.

**It deviates in two places, for reasons of this data structure:**

1. **No separate coarse-level type.** flow's levels are structured bricks; here the root brick is
   already a `BlockOctree`, and lifting the root gives a coarser `BlockOctree` for free. flow
   needs a level to *be* a brick; amr needs it to be an octree. Same hierarchy, no new type.
2. **Isotropic coarsening only.** flow semi-coarsens per axis; octree cells cannot. The anisotropic
   residue (a 64×64×4 root brick lifts to 32×32×2 and stops) is handled by the exact bottom, which
   flow also uses for exactly that case (`64×2×2`: 6.0 → 4.0 iterations).

**Are both halves needed?** Yes, in this order, and for different reasons.

- *Lifting* is what delivers the measured 8.1×; it costs essentially nothing to build or run and
  makes the bottom small. Without it the exact bottom would have to gather the root brick (262 144
  cells at 64³ — a 22 MB CSR per rank and a ~100 ms serial solve per V-cycle: no better than today).
- The *exact bottom* is what makes the iteration count a property of the problem rather than of the
  grid's factors of two and the rank count. On power-of-two grids at np ≤ 8 (every test in the
  battery) the lift reaches extent ≤ 4 and 60 damped-Jacobi sweeps are exact to 1e-8 (§6.6), so
  the exact bottom never engages there. It engages on badly factored roots (100³ → 25³) and inside
  the tail on badly factored global grids — flow's 400³ case is the cautionary measurement
  (iterations 96 → 191 between np=48 and 384). It is a port, and it can land after the lift
  without diminishing the measured prize.

## 5. Options considered

### 5.1 Lift the root of the existing octree (CHOSEN)

A sub-root level = copy of the previous level, `brick /= 2`, `globalOrigin /= 2`, `++lmax`, then
`coarsenIf` merges every root-cell octet. Bit-for-bit the octree the brief's emulation built
(`Octree(64, lmax=3)` refined everywhere has the same codes and levels as `Octree(64, lmax=0)`
with those two scalars changed), so the 8.1× is not an estimate of this design; it *is* this
design measured. Every builder downstream is unchanged. Cost: a ~10-line `BlockOctree` method and a
lift rule in two ladder loops.

### 5.2 Generalise the `inner_` chain (`distributed_fv.hpp:478`) — REJECTED

`inner_` is a `DistributedMultigrid` (`distributed_poisson.hpp:150`): a *second* level type — a
fresh `DistributedOctree` per level on a halved global root grid with its own ORB, nesting only
asserted (power-of-two grids and rank counts), no openness, host-only, and used by no production
path (the device `AmrFlow` uses `Multigrid` / `DistributedFlowMultigrid`, which have no `inner_`).
Carrying openness into it means re-deriving the area-averaged ladder across a second index space
(`innerMap_`) and a second halo. Lifting produces the same hierarchy inside the existing type with
the existing openness ladder. `inner_` becomes conceptually redundant; it stays as it is (host
oracle, tested) and may be retired in a later cleanup.

### 5.3 flow's agglomerated CSR bottom alone, on the root brick — REJECTED as the whole answer

Adopted as the *bottom* (§6.6), rejected as the *fix*: at 64³ the root brick is 262 144 cells; a
serial `GraphAMG`-PCG on it costs O(100 ms) per V-cycle on the host, times ~12 V-cycles per step —
the same order as today's 2 s/step, and it moves the solve off the device. §2.7's "shallow over
exact" finding assumes a bottom of a few cells per axis, which only lifting provides here.

### 5.4 "Require large `lmax`" — REJECTED (a policy, and an indefensible one)

The root grid is not a solver setting: it is the **rebalance weight grid** (settled decision:
"defined over root cells") and the unit of the ORB. Coarsening it to help the multigrid (a 16³
root for a 512³ domain) destroys the load balancer's resolution at 1536 ranks and changes
`refineToSdf` / `Octree(cells, lmax)` semantics for every user. It also does nothing for a graded
mesh whose refinement already fixes `lmax`. The hierarchy must go below the root without moving
the root.

### 5.5 Semi-coarsening below the root (per-axis halving as in flow) — REJECTED as a level type; the elongated brick mesh is handled by the exact bottom

The brick mesh is *intended* to be elongated — the octree is the cubic part, the aspect ratio
lives in the brick counts — so this is the design's normal regime, not an edge case. A cubic-cell
octree cannot represent a `2h × 2h × h` level; semi-coarsening would be a second level type with
its own operators, transfers and openness averaging, and that is what is rejected.

Adequacy argument. With brick counts carrying factors of two on every axis, isotropic lifting takes
the *short* axes to 2 and the bottom has

    n_b ≈ 2^Dim · Π_d (G_d / G_min)        (the aspect product, not the domain size)

cells: `4096×32×32` bricks → `256×2×2` = 1024; `64×64×4` → `32×32×2` = 2048. An exact solve on
that costs microseconds to a millisecond, and it is the reference's own answer to elongation:
`DECOMPOSITION_AND_MULTIGRID.md` §2.7's `64×2×2` row is 6.0 iterations smoothed vs **4.0 exact**,
and flow's full-depth *semi-coarsened* ladder (4.4) did not beat the exact bottom. Semi-coarsening
buys nothing on iterations; it only bounds `n_b`, which matters when the short axis has few factors
of two (`512×512×4` → `256×256×2` = 131 k cells, ~50 ms serial per V-cycle). Hence two rules,
absorbed by WO5 and `predict`:

- the exact bottom runs on the host for `n_b ≤ 10⁴` and on core's existing `GraphAMGDevice`
  (`graph_amg_device.hpp`) above that;
- design rule (joins §3 rule 1 of `DECOMPOSITION_AND_MULTIGRID.md`): choose brick counts for their
  factors of two on every axis, keep the short axis at ≥ 8 bricks, and read `n_b` off `predict`.

Revisit a structured semi-coarsening *tail* level (single-rank, uniform, no octree needed there)
only if a production slab measures a bottom above ~10⁵ cells on device.

### 5.6 Sub-communicator telescoping first (flow's rung 2) — DEFERRED, not rejected

The correct at-scale continuation (§11.1). Not first because (a) the redundant tail (§6.5) is
flow's rung 1, needs no sub-communicators or idle-rank branches, and its cost is bounded by the
coarsest in-place level — 48³ = 110 592 cells at 1536 ranks on 384³, i.e. 0.9 MB per rank per
V-cycle and a ~1 ms device V-cycle — and (b) it reuses `Multigrid` unchanged. Rung 2 is the
escalation when the gathered level exceeds ~10⁶ cells.

## 6. The design

### 6.1 The level model

Every level of the pressure hierarchy is a `BlockOctree<Dim, Bits>` plus an `AmrPoisson` built by
`AmrPoisson::init(level, h0)` with `h0` the **finest** spacing on every level (as today: a coarse
leaf's width is `2^level · h0[d]`). A level is one of:

| kind | how it is made from the level above | where it stops |
|---|---|---|
| octree level | `coarsenIf(always)` | when nothing merges: every leaf is a root cell |
| **lifted level** | copy, `liftRoot()`, `coarsenIf(always)` (merges every octet) | the lift rule (§6.2) |
| tail level (distributed only) | a fresh uniform `BlockOctree(G_t, lmax + k, 0)` on the gathered global grid, then lifted levels | as single-rank |

`BlockOctree::liftRoot()` (new, `block_octree.hpp`):

```
precondition: brick_[d] % 2 == 0 and globalOrigin_[d] % 2 == 0 for every d
brick_[d] /= 2;  globalOrigin_[d] /= 2;  ++lmax_;
postcondition: codes_, levels_ unchanged; brick_[d] << lmax_ unchanged
```

`canLiftRoot()` returns the precondition. Nothing else in `BlockOctree` changes; `balance2to1`,
`refine`, `find`, `code`, `level` are independent of the root.

### 6.2 The ladder rule (single-rank and distributed share it)

Let `G` be the global root grid in the *current* root units of the level (single-rank: the brick),
`bottomExtent = 4` (flow's `PECLET_FLOW_AGGLOM_EXTENT`; a parameter, §6.8).

```
after the octree levels are exhausted (every rank at its root brick, padded — as today):
  repeat:
    stop if max_d G[d] <= bottomExtent                      # the Jacobi bottom is exact here
    stop if any d: G[d] % 2 != 0 or G[d] / 2 < 2            # grid-limited: no cube level below
    stop if any rank: !local.canLiftRoot()                  # decomposition-limited (Allreduce MIN)
    lift on every rank (lockstep), coarsenIf, push the level
```

Single-rank the third test is the second (origin 0, brick = G). The lift depth `k` is computed
**before** any level is built, as one `MPI_Allreduce(MIN)` of the per-rank allowed depth
(iterate the predicate on `(brick/2^j, origin/2^j)`), so every rank builds the same number of
levels and every subsequent collective pairs up — the padding argument of
`distributed_flow_mg.hpp:15–24` extended by `k`.

Lifting is **lockstep only**: no rank lifts while another still merges octree children. That keeps
every sub-root level globally uniform (no level jump at any block seam beyond the 2:1 the octree
levels already have), so `forEachFaceNeighbor`'s probe reach never sees a >2:1 jump at a seam.
Ranks that reached their root early pad with identity levels, exactly as today; the padding costs
nothing extra because those ranks would idle at the root anyway.

Worked ladders (root grid → levels → bottom):

| case | ladder (global extents) | levels | bottom |
|---|---|---|---|
| uniform 64³, `lmax=0`, np=1 | 64→32→16→8→4 | 5 | Jacobi (E=4) |
| uniform 32³, `lmax=0`, np=1 | 32→16→8→4 | 4 | Jacobi |
| graded 64³, `lmax=3` (root 8³) | 3 octree levels then 8→4 | 5 | Jacobi |
| byte gate 32³, `lmax=1`, np=2 (blocks 8×16×16) | 32→16 (octree) →8→4 | 4 | Jacobi, no tail |
| byte gate, np=1 | identical ladder | 4 | identical arithmetic |
| 16³ root, np=8 (blocks 8³, origins 0\|8) | 16→8→4 | 3 | Jacobi |
| 12³ root, np=4 (blocks 6×6×12) | 12→6 in place (origin 3 is odd) → **tail** 6→3 | 2 + 2 | Jacobi (E=3) |
| 10³ root, np=1 | 10→5 | 2 | **GraphAMG** (E=5 > 4, 125 cells) |
| 384³, `lmax=0`, np=1536 (blocks 24×48×32) | 384→192→96→48 in place → tail 48→24→12→6→3 | 4 + 5 | Jacobi (E=3) |
| 64×64×4 root | 64×64×4 → 32×32×2 (z: 2/2 < 2 stops) | 2 | GraphAMG (E=32, 2048 cells) |

### 6.3 Operators, transfers, openness on a lifted level

Unchanged code, by construction:

- **Operator.** `AmrPoisson::init(level, h0)`; `fineExt_` is invariant; periodic wrap and the
  `coeff(si, sj, axis)` width-unit arithmetic are level-agnostic. Device assembly
  (`Multigrid::buildFaceCsr` → `assembleFv`) reproduces the host walk on the lifted leaf set.
- **Transfers.** `c2p[i] = coarse.find(fine.code(i))` (the covering-leaf rule, which the code
  already documents as correct for mixed-depth ladders); `childStart/childIdx` histogrammed from it
  in fine order. Restriction stays the volume average, prolongation piecewise constant. For a
  lifted level every coarse cell has exactly 2^Dim children — the same as an octree level.
- **Openness.** `AmrMultigrid::coarsenOpenness(L)` / `DistributedFlowMultigrid::coarsenOpennessTo(L)`
  area-average child faces to the parent through `child_index(level)` — Morton arithmetic that is
  valid for any level ≤ `Bits`. With lifted levels the ladder is one longer per lift; the ghost-α
  exchange at each new level uses that level's halo, as today. κ (`mg_kappa`) follows from the
  per-level `faceOpenness` unchanged.
- **Null space.** `removeMeanFv` / `removeMeanFvDist` per level, unchanged.

The invariant to test (WO1): building `Multigrid` on `Octree(64, lmax=0)` with lifting and on
`Octree(64, lmax=3)` refined everywhere (the emulation) must give **identical level sizes,
identical per-level face CSRs (`faceStart/faceNbr/faceW/bcDiag/invVol` bitwise) and identical
V-cycle output** — the emulation is the oracle for the lift.

### 6.4 Ownership under MPI: nesting, and the alignment that preserves it

In-place levels keep the ORB: rank `r`'s lifted block is its root block halved, so a coarse cell's
2^Dim children are all on `r` and transfers stay local; only the Jacobi smoother's halo talks.
The precondition is flow's `evenBlocks`: `blockOriginRoot[d]` and `blockBrick[d]` even on every
axis on every rank, re-evaluated per lift in the current root units.

Two partitions produce those blocks:

- **Unweighted `DistributedOctree::init`.** The plain ORB bisects at proportional positions; for
  power-of-two grids and rank counts every split is a multiple of the largest power of two the grid
  allows, so blocks nest to full depth (this is what `DistributedMultigrid` already relies on). For
  non-power-of-two `np` (3, 6, 12, 1536 = 2⁹·3) the splits land at thirds and nesting depth is
  whatever the arithmetic gives; the prediction tool (§6.7) reports it. Improvement: build the ORB
  **coarse-first** — decompose `G / 2^a` and `refined(2^a)` (core `BlockDecomposer::refined`,
  measured in `DECOMPOSITION_AND_MULTIGRID.md` §2.4 to balance as well or better) with the largest
  `a` whose imbalance ≤ 1.05. Pure function of `(np, G)`: every rank computes it identically. For
  power-of-two cases the result is the partition of today (bit-identical runs).
- **Weighted `rebalance`.** Today's weighted ORB has no alignment, so a rebalanced partition may
  stop lifting at the root brick (odd origins) — correct, because the tail then takes over, but
  the tail would gather the whole root grid every V-cycle. Fix, same mechanism: coarse-first on the
  weight grid — sum the root-cell weights into `2^a`-blocks, run the weighted ORB on the coarse
  weight grid, `refined(2^a)`; choose the largest `a ≤ a_max` (divisibility) whose measured
  imbalance ≤ 1.05, falling back to `a = 0` (today's partition). Compatible with the settled
  decision that the weight grid is over root cells (it still is; the split positions are snapped).

**The nesting constraint is an input to the balancer, not a post-hoc filter.** In the intended
regime (many bricks, shallow octrees, routine rebalancing, the decomposition shared with DEM) a
weighted partition will generally have odd origins, and without alignment every rebalanced run
gathers the whole brick grid each V-cycle. Coarse-first keeps whole bricks on ranks and whole
octrees migrating; the balancer's quantum becomes a `2^a`-brick group. `a` is chosen as the
*smallest* depth whose gathered level fits the tail budget (`N_bricks / 2^(a·Dim) ≤ ~10⁵`) within
the imbalance budget 1.05 — with hundreds of bricks per rank that is `a = 2–3` at a few per cent
imbalance. CFD-DEM sharing is unaffected: alignment constrains split *positions*, the weights stay
the combined ones. So WO7 belongs in core's `BlockDecomposer` (a weighted coarse-first `init`,
once), used by `DistributedOctree::init/rebalance` and the coupling's shared factory. **WO7 is
required for C1 to close** — before any rebalanced production run and before Snellius. Until it
lands the tail keeps non-nesting partitions correct, and `pressure_mg_bottom` must show the
gathered size so the cost is visible.

### 6.5 The redundant tail (distributed only)

Engaged iff, after the in-place ladder stops, `max_d G_t[d] > bottomExtent`, where `G_t` is the
coarsest in-place global grid. It replaces the bottom branch of `DistributedFlowMultigrid::vcycle`
for the coarsest level `L_t`:

**Once per build.**
1. Global coarse coordinate of local cell `i` at `L_t`: `g_i[d] = (blockFineOrigin[d] + lo_i[d]) / W`,
   `W = 2^(lmax + k)` fine units (exact division by nesting); `gid_i = linear(g_i)` x-fastest
   (`CONVENTIONS.md`).
2. `Allgatherv` of `(gid_i, α_i[2·Dim])` — the local rows of `L_t`'s `opennessRaw()` — into a
   gid-ordered array of the whole `G_t` (every rank ends with the same array).
3. Build `tail = BlockOctree(G_t, lmax + k, origin 0)`; map `gid → tail leaf` once via
   `tail.find(M::encode(g·W))`; permute α into tail leaf order.
4. Build a single-rank `Multigrid<Dim, Bits>` on `tail` with raw openness (new `buildRaw(octree, h0,
   alphaRaw, periodic=true)`: `hmg_->build(tail, h0)`, `op(0).setOpennessRaw(alpha)`, the existing
   `coarsenOpenness` ladder, `buildFromHostMg()`), `setRemoveMean(true)`. Its own ladder lifts as
   far as §6.2 allows on one rank and its own bottom is §6.6. It is exactly the hierarchy a
   single-rank run would build from level `L_t` down — flow's telescoping test measured that
   property as "matches the single-rank reference to 2.5e-14".

**Per V-cycle.** At `L_t`: `b` (the restricted residual, device) → host staging → `Allgatherv` of
`(gid, b)` → permute to tail order → `deep_copy(tail.b(0))`, `tail.x(0) = 0` → **one** tail
V-cycle (it is the continuation of the same V-cycle, not a separate solve) → each rank picks its
own rows from `tail.x(0)` by `gid` → `deep_copy` into `L_t`'s `x` → prolong as today. Every rank
computes the identical tail, so the scatter is a local pick with no communication.

**Why redundant, not rank-0.** No serialization point, no broadcast, the same pattern flow's bottom
uses; decomposition-independent because the tail is keyed by global id.

**`L_t`'s own halo/operator** are built as today (their cost at that size is negligible) and unused
when the tail engages; an implementer may skip them later.

### 6.6 The bottom: damped Jacobi at extent ≤ 4, `GraphAMG`-PCG otherwise

*Why 4 is exact for Jacobi.* On a periodic axis of `E` cells the slowest non-constant mode of the
7-point Laplacian has eigenvalue `2 − 2cos(2π/E)`; with diagonal 6 and ω = 0.8 the damped-Jacobi
amplification is `1 − 0.8(2 − 2cos(2π/E))/6`. Sixty sweeps (`pcg_.setVcycle(2, 2, 60, 0.8)`,
`flow.hpp:1162`) give: E=2: 0.47⁶⁰ ≈ 1e-20; E=3: 0.60⁶⁰ ≈ 5e-14; E=4: 0.73⁶⁰ ≈ 8e-9; E=5: 5e-6;
E=6: 2e-4; E=8: 8e-3; E=16: 0.30; E=25: 0.60. So ≤ 4 is "exact", 5–8 is adequate for a
preconditioner (the brief's §3 measured 12 iterations with a 16³ bottom, PCG covering the rest),
and beyond ~8 the bottom is not solved. The threshold is flow's measured 4; §11.4 discusses 8.

*The exact bottom* (new header, e.g. `amg_bottom.hpp`, host code; used by `Multigrid` single-rank
and therefore by the tail):

1. **Assemble** from the coarsest `AmrPoisson` via `assembleFv()` (host): per row `i`,
   `S_ii = Σ_k coef_k + bcDiag_i`, `S_ij = −coef_k` over the row's faces (periodic neighbours are
   already wrapped by the host walk). This is `S = −Vol·L`, symmetric, positive semi-definite, with
   `S·1 = 0` per connected fluid component when `bcDiag = 0`. `faceW` is double already, so flow's
   float row-sum defect (§2.7 lesson 3) does not arise — but resum `S_ii` from the off-diagonals in
   the singular case anyway; it costs nothing and makes `S·1 = 0` exact.
   The implementer confirms the sign and volume convention against `jacobiFv`/`residualFv`
   (`fv_op.hpp`) with the consistency check below, not by reading this note.
2. **Identity rows** for `S_ii == 0` (a fully closed coarse cell: all 2·Dim faces α = 0); their rhs
   is 0 and their solution 0.
3. **Components.** Union-find over edges with `coef > 0` among non-identity rows. The null space is
   the constant per component (flow lesson 1: projecting the all-cell mean instead leaves part of
   the null component alive and stalls the inner CG).
4. **Solve.** `b_i = res_i / invVol_i` (the volume-integrated rhs; the projected mean per component
   is then the volume-weighted mean of `res`, consistent with `removeMeanFv`); project `b`
   per component; `GraphAMG`-preconditioned CG (core `graph_amg.hpp`, default `AmgParams`) to
   relative 1e-8, cap 100 iterations, projecting the preconditioned residual per component each
   iteration; project the solution per component; `x_bottom = e`; then the existing `removeMeanFv`.
5. **Build once per `build()`** (the operator changes only with `setSolid`); the host↔device traffic
   per V-cycle is `n_b` doubles each way.
6. **Consistency gate (debug print, as flow's `AGMG_DEBUG`):** the CSR solution must satisfy the
   V-cycle's own bottom `FvOp` to `max|b − Lx| / max|b| ≤ 1e-9`.

Selection: `auto` (default) engages the exact bottom iff `max_d E[d] > bottomExtent`; `smoother`
forces Jacobi; `agglomerated` forces the exact bottom. Same three spellings as flow's
`set_pressure_bottom` (NAMING: one spelling per concept).

### 6.7 Prediction and diagnostics (what the tests assert against)

- `predictPressureLadder(G, lmax, np, bottomExtent)` — a pure host function (no MPI) returning per
  level the global cell count, whether it is in place or tail, and the bottom kind; it builds the
  ORB with the same factory `DistributedOctree::init` uses (`BlockDecomposer` on `G`). Python:
  `peclet.amr.predict_pressure_hierarchy(cells, lmax, num_ranks)` (name to be checked against
  `NAMING.md` by the caller; flow's is `predict_hierarchy`).
- `Flow.diagnostics.pressure_mg_levels -> list[int]` (per-level local leaf counts, level 0 first,
  tail levels appended) and `Flow.diagnostics.pressure_mg_bottom -> str` (`"jacobi"` | `"amg"`;
  suffix `"+tail"` when the tail is engaged). The `[step-prof]` header line prints the same.
- Tests replace level-count literals by `predict` (§10). `test_amr_distributed_mg_mpi.cpp:86` and
  `test_amr_distributed_view_mpi.cpp:111` pin classes this design does not touch and stay valid;
  `test_amr_fv_openness.cpp:71` (`mg.numLevels() == hmg.numLevels()`) stays valid because both
  sides lift.

### 6.8 Parameters (all with today's behaviour reachable)

| parameter | default | where | changes a result? |
|---|---|---|---|
| `liftRoot` | `true` for pressure, `false` for `VelocityMG` until WO5 | `AmrMultigrid::build`, `DistributedFlowMultigrid::build` | preconditioner only |
| `bottomExtent` | 4 | same | preconditioner only |
| `set_pressure_bottom` | `auto` | `Flow.diagnostics` (§11.6) | preconditioner only |
| Jacobi bottom sweeps | 60 (unchanged) | `pcg_.setVcycle` | preconditioner only |
| ORB alignment budget | 1.05 imbalance | `DistributedOctree::init/rebalance` (WO7) | partition (np non-power-of-two, rebalanced runs) |

No environment variable (QUALITY_PLAN D3).

## 7. Why it cannot degrade accuracy — and what would falsify that

The multigrid is the preconditioner of `PCG::solve` (`pcg.hpp:159`): the converged pressure is
defined by the **level-0 operator** and the stopping tolerance `presTol_ = 1e-10`, neither of which
this design touches (level 0 is a copy of the flow octree with the same `AmrPoisson`; the lifted
levels are coarser copies). The V-cycle remains a fixed linear operator per application (fixed
Jacobi sweep counts, linear transfers, symmetric pre/post) — adding levels preserves that — so PCG
converges to the same solution to solver tolerance. The exact bottom is an *inner* Krylov solve
to 1e-8, which makes the preconditioner very slightly non-stationary; flow runs the same
construction and measured parity, and the PCG's recurrence residual (`pcg.hpp:202–217` — it is the
recurrence, not a recomputed residual) is the thing to watch.

Falsifiers, each a gate in §10:

1. Two pressure solves of the same operator and rhs, old and new hierarchy, at `presTol = 1e-12`,
   differ by more than ~1e-10 relative in the volume-weighted norm.
2. The **true** residual `|b − L x|` recomputed by `residualFv` after a solve exceeds
   `presTol · res0` by more than 10× (exposes recurrence drift from the inner solve).
3. `flow_parity` (six cases) leaves its existing tolerance; Z&H permeability moves in the ninth
   digit or worse; `python_amr_tg_graded` error levels move beyond its 5 % gate.
4. Iterations do not fall (uniform 64³: 23 → ~12), or grow faster than mildly logarithmically with
   N at fixed geometry (a doubling per doubling of N would be a defect; 11/13/15 at N = 32/64/128,
   as WO1 measured, is the mild class — see §11.9).

What *will* change and is not a defect: every PCG iterate (hence every byte-gate key with a
pressure/Poisson solve), the iteration count, and the last bits of the converged pressure within
tolerance.

## 8. Cost

**Build.** Per lifted level: one `coarsenIf` (linear), one `AmrPoisson::init` + device assembly
(linear in that level's cells), the openness averaging, the `c2p` finds; distributed: one `LeafHalo`
build (an NBX round) and the ghost-α exchange. The lifted levels total < 1/7 of the root-grid
level; on a graded bed the root grid is ≪ the leaves, so this is a rounding error on
`presMG.build` (profiled under `PECLET_AMR_PROFILE_SETUP`). The tail adds one `Allgatherv` of
`2·Dim·n_t` doubles and a single-rank build on `n_t` cells. The `GraphAMG` setup, when engaged, is
serial host O(n_b) — tens of ms at 10⁴ cells, once per `setSolid`. Nothing here approaches C2's
`setSolid` cost.

**Memory.** Lifted levels: < 1/7 of the root-level footprint. Tail: `n_t · (7 + 4)` doubles plus
its own lifted levels per rank. `GraphAMG`: O(2 × nnz) host.

**Run.** Per V-cycle each lifted level costs `pre + post` Jacobi sweeps, a residual, a restrict and a
prolong on ≤ 1/8ⁿ of the root cells — arithmetic-free, launch-bound on CUDA (~8–10 launches per
level, ~50–100 µs per tiny level; 12 V-cycles/step × 4 levels ≈ 3–5 ms/step, against an 8× win).
The Jacobi bottom's 60 launches on a 64-cell level are the same class (§11.4).

**Collectives (new).** Build: one `MPI_Allreduce(MIN)` for the lift depth; when the tail engages,
one `Allgatherv` (α rows). Per V-cycle, only when the tail engages: one `Allgatherv` of `n_t`
doubles (1536 ranks on 384³: 0.9 MB into every rank, ~1.3 GB aggregate per V-cycle — acceptable
at 8 nodes; the reason rung-2 telescoping stays on the table, §11.1).

**Expected numbers.** Uniform 64³ `lmax=0`: ≤ 270 ms/step (the 4-level emulation measured 252.0;
the 5-level ladder should not be slower), 12 ± 1 pressure iterations. Uniform 32³ advection case:
≈ 30 ms (0.96 µs/leaf × 32 768) against flow's 27.5. Graded rows of the brief's table: within
±10 % of today (they gain one or two tiny levels).

## 9. Work orders (commit-sized, dependency order)

**WO1 — `liftRoot` + the single-rank ladder.** `BlockOctree::liftRoot/canLiftRoot`;
`AmrMultigrid::build(finest, h0, liftRoot=true, bottomExtent=4)` with the §6.2 rule (single-rank
form); `VelocityMG` passes `liftRoot=false`. `Multigrid` inherits it through `hmg_`.
*Accept:* `Multigrid` on `Octree(64, lmax=0)` and on the everywhere-refined `Octree(64, lmax=3)`
(the emulation) give identical `numLevels()`, identical per-level CSR arrays (bitwise) and
identical `x(0)` after 5 V-cycles from the same rhs; level-0 CSR bitwise identical to the
pre-change build; `test_amr_assembly`'s host/device parity holds on a lifted level; momentum keys
of the byte gate unchanged, `Octree` keys unchanged; `amr_pressure_depth.py` uniform rows ≤ 270 ms
(64³) and ≤ 35 ms (32³) at 12 ± 1 iterations.

**WO2 — prediction + diagnostics.** `predictPressureLadder`, its binding, `pressure_mg_levels`,
`pressure_mg_bottom`, the `[step-prof]` line. *Accept:* the prediction equals the built ladder on
every configuration of §6.2's table (single-rank half) and on the study script's eight rows.

**WO3 — the distributed lockstep lift.** `DistributedFlowMultigrid::buildImpl`: Allreduced lift
depth after the padding, lockstep lifted levels, per-level halo + ghost-α as today.
*Accept:* np = 1 bitwise vs `Multigrid` (the byte gate's np=1 distributed keys equal the single-rank
keys where they did before); np = 2/4/8 vs np = 1 ≤ 3e-7 on `amr_distributed_graded_mg_mpi` and
the seam test; ladder == `predict` at np = 1, 2, 4, 8; level counts equal on every rank (assert
with an Allreduce in debug builds).

**WO4 — the redundant tail.** §6.5, including `Multigrid::buildRaw`. *Accept:* on a 12³-root grid
at np = 4 (tail engages, 6→3) and np = 1 (no tail) the solutions agree to ≤ 1e-12 after the same
number of V-cycles and the iteration counts are equal; every rank's tail `x(0)` is bitwise equal
across ranks (Allreduce of a hash in the test); `predict` reports the tail.

**WO5 — the exact bottom.** §6.6 header, `set_pressure_bottom`, `auto`; host `GraphAMG` for `n_b ≤ 10⁴`, `GraphAMGDevice` above (§5.5). *Accept:* the consistency
gate ≤ 1e-9 on a 10³-root case (bottom 5³) and on a cut-cell case with a closed pocket (identity
rows + two components); a 100³-root uniform Poisson (bottom 25³) converges in the same iteration
count as 128³ (bottom 4³) ± 1; `smoother` reproduces WO1–4 bitwise; np = 2 vs np = 1 on the
10³ case ≤ 1e-12 (the bottom runs inside the tail).

**WO6 — byte-gate re-record + ROADMAP/CLAUDE.md.** `state_hash.py --save` in its own commit
naming the moved keys (expected: `Poisson`, both `Flow` keys, the distributed `Flow` keys at np=1
and np=2; NOT the `Octree`/`DistributedOctree` topology keys); ROADMAP C1 closed with the measured
table; `CLAUDE.md` architecture paragraph gains one sentence on lifted levels.

**WO7 — ORB alignment (coarse-first) for `init` and `rebalance` — REQUIRED for C1 to close (§6.4), in core's `BlockDecomposer`.** *Accept:* power-of-two
grids × power-of-two `np` give the partition of today (bitwise runs); on 384³ `predict` at
np = 1536 reaches ≥ 4 in-place levels; after `rebalance` on the weighted test case the ladder loses
at most one in-place level versus the unweighted ladder at imbalance ≤ 1.05.

**WO8 — velocity multigrid (separate package, §12).** Flip `liftRoot=true` for `VelocityMG`;
gates: `flow_parity` Stokes cases, `amr_two_sphere_gap.py` at cf = 1, momentum iterations on the
32³ Stokes case (25.6 today), byte gate re-record for the momentum keys.

## 10. Verification gates

| gate | configuration | criterion |
|---|---|---|
| depth study | `tests/study/amr_pressure_depth.py`, host-openmp 4 threads | uniform 64³ `lmax=0`: ≤ 1.05 × the everywhere-refined `lmax=3` emulation measured on the SAME box in the same session (the brief's 252 ms was another box; WO1 measured 291–312 at 13 it), 12–13 pres it (was 23); uniform 32³: ≤ 35 ms; graded rows within ±10 % |
| scalability | uniform periodic sphere, N = 32, 64, 128, `lmax=0`, `presTol=1e-10` | PCG iterations ≤ 13 at every N and varying by ≤ 1 across N |
| emulation oracle | WO1 acceptance | bitwise |
| parity | `tests/study/flow_parity/parity_gate.py` (`PECLET_AMR_FLOW_PYTHONPATH` set) | six cases within their existing tolerances; advection case amr ≤ 1.3 × flow ms/step |
| accuracy | `amr_zh_ladder.py` vs `docs/data/amr_zh_ladder_n64.log`; `amr_tg_graded.py --gate` | permeability agrees to ≥ 8 significant digits; tg gate passes unchanged |
| true residual | WO1 debug path (`PECLET_AMR_PRES_DEBUG=1`) | `|b − Lx|` after solve ≤ 10 · presTol · res0 on every profiled step |
| byte gate | `state_hash.py --check`, np = 1 and `mpirun -np 2` | before WO6: only pressure/Poisson/Flow keys differ; after WO6: all identical |
| battery | 96 C++ + 5 bench + 5 python, np = 1, 2, 4, 8 (`OMP_NUM_THREADS=2`, np8 last) | green; level-count literals replaced by `predict` where the class changed |
| distributed | `amr_distributed_graded_mg_mpi`, seam test, new lift/tail tests | np = 1 bitwise vs single-rank; np = 2/4/8 ≤ 3e-7 (V-cycle only: bitwise with `removeMean` off, as the file header promises) |
| tail | 12³ root at np = 4 vs np = 1 | ≤ 1e-12 after equal V-cycles; equal iteration counts |
| exact bottom | 10³ root np = 1, 2; 100³ vs 128³ root | consistency ≤ 1e-9; iteration parity ± 1; np-independence ≤ 1e-12 |
| stability | `amr_two_sphere_gap.py` cf = 1, `--advection 1` | no divergence; iteration class unchanged |
| CUDA | WO1 study on the `nvidia-cuda` prefix | same iteration counts as host; ms/step reported (§11.4) |

## 11. Risks and open questions — each with a default

1. **At-scale gather volume (needs a fact).** The tail's `Allgatherv` grows linearly with `np`
   under weak scaling (110 k cells at 1536 ranks on 384³; ~10⁶ at ~10⁴ ranks). *Default:* ship
   the redundant tail; measure on Snellius at 384 and 1536 ranks (packed bed, the D3b prefix
   rebuild first); escalate to sub-communicator telescoping (flow's rung 2, core's
   `BlockDecomposer::agglomerated`, merging sibling ORB blocks so lifting continues on 1/8 of the
   ranks) when the gathered level exceeds ~10⁶ cells or the gather exceeds 10 % of the step.
2. **Determinism of the redundant tail across ranks (needs a fact).** Identical device kernels on
   identical data give identical results on identical hardware; heterogeneous GPUs could differ by
   an ulp, which perturbs the preconditioner by ~1e-16 (harmless to convergence, breaks bitwise
   tail equality). *Default:* assert bitwise equality in the test on homogeneous nodes; treat an
   ulp difference on heterogeneous hardware as acceptable and documented, not as a failure.
3. **Non-nesting partitions before WO7 (fact + preference).** A rebalanced run may gather the whole
   root grid every V-cycle — correct but slow at bed scale. *Default:* WO7 lands in the same
   package; until then `pressure_mg_bottom` reports `"+tail"` with the gathered size so it is
   visible, and the study logs it.
4. **`bottomExtent` 4 vs 8, and 60 bottom sweeps on CUDA (needs a fact).** The amplification table
   says 8 is adequate; the brief's data say a 16³ bottom already saturated PCG on a cube; flow's 4
   came from a 2048-long channel. On CUDA the 60 bottom launches and the 4 extra tiny levels cost
   launch latency (~3–8 ms/step estimated). *Default:* keep 4 and 60 (follow the reference; isolate
   the structural change); a one-afternoon CUDA sweep over `bottomExtent ∈ {4, 8}` × `bottom ∈
   {30, 60}` after WO1 decides, as its own commit.
5. **Slab-like root bricks (preference).** Isotropic lifting leaves a long axis for the exact bottom
   (64×64×4 → 32×32×2, 2048 cells: fine; 512×512×4 → 256×256×2 = 131 k cells: a serial bottom of
   ~50 ms per V-cycle). *Default:* accept; revisit semi-coarsening only if a production slab
   domain measures badly.
6. **API tier for `set_pressure_bottom` (preference).** flow exposes it publicly; here it is an
   ablation switch whose answer is `auto`. *Default:* `Flow.diagnostics` tier, flow's spelling and
   values; the caller decides whether it should be public for symmetry.
7. **`GradedDistributedMultigrid::inner_` / `DistributedMultigrid` (preference).** Redundant in
   concept after this design, still tested host oracles. *Default:* leave them; retire in a later
   cleanup with their tests, not in this package.
8. **Recurrence-residual drift with the inexact bottom (needs a fact).** *Default:* the true-residual
   gate in §10; if it trips, tighten the inner tolerance to 1e-10 (flow measured no iteration
   change between 1e-5 and 1e-8) before considering flexible CG.

9. **Iterations grow mildly with N — 11/13/15 at N = 32/64/128 (WO1 measurement; needs a fact,
   separate item).** Not the lift's doing: the bottom is exact at every N. The first suspect is
   NOT the level-0 cut-cell operator (the cut band's share shrinks with N, which would *reduce*
   iterations) but the ladder's transfer pair — piecewise-constant prolongation + volume-average
   restriction, `m_P + m_R = 2`, which does not satisfy the strict `> 2m` accuracy condition for a
   rediscretized cell-centred hierarchy; MG-as-PCG-preconditioner is known to mask that to mild
   growth. *Discriminator before anyone opens a hypothesis:* the same N-ladder on the
   openness-free periodic `Poisson` with a smooth rhs. Growth there → the ladder (then try
   `cyclesPerPrec = 2`, ω = 6/7, or a Galerkin-scaled coarse operator); flat there → the operator.
   *Default:* open as its own ROADMAP item; C1 closes on the depth prize.
10. **Elongated brick meshes with a short axis of few factors of two (fact).** `n_b` from §5.5 can
   reach 10⁵; *default:* the device bottom (WO5) and the brick-count design rule; a structured
   semi-coarsening tail level only if a production slab measures badly.

## 12. The three questions the brief asked

**Deeper hierarchy or exact bottom on the root brick?** Both, and the order is forced: lifting is
nearly free and makes the bottom small; the exact bottom on an unlifted 64³ root (262 144 cells)
is a serial O(100 ms) solve per V-cycle that also leaves the device — no better than today. The
crossover is not a cell count but a *divisibility* fact: lift while the extent halves, solve
exactly when it cannot. Single-rank on power-of-two grids the lift reaches extent ≤ 4 and no
exact bottom is needed; at 1536 ranks the in-place lift stops where the blocks turn odd
(3×6×4 per rank on 384³), the tail lifts the gathered 48³ to 3³, and again Jacobi is exact — the
`GraphAMG` bottom is for badly factored grids at any rank count.

**Under MPI past one cell per rank?** The in-place levels never go below one cell per rank — the
lift rule stops at an odd origin or size — and below that the tail continues on every rank with
the single-rank hierarchy of the gathered level, decomposition-independent by construction. The
levels are never *split* across ranks: they are either nested in the ORB or replicated. Sub-
communicator telescoping (idle ranks at the bottom) is the escalation, deferred with a measured
trigger (§11.1).

**The velocity multigrid?** The mechanism generalises mechanically — `VelocityMG` builds through
the same `AmrMultigrid::build` and its per-level κ classification uses `ancestor(level+1)`, valid on
lifted levels — but the *guard* is the question, not the mechanism: `minCoarse = 256` caps by cell
count because a staircase-classified feature vanishes when coarsened below its scale, and that is
a property of the momentum operator, not of the root. With the cap kept, the uniform 64³ momentum
hierarchy goes from 1 level to 4 (262 144 → 512 cells). Expectation: the Stokes-case momentum
share (33.6 %, 25.6 iterations) drops by a factor of 2–3. It is its own work order (WO8) with the
two-sphere stability study as the falsifier, and `liftRoot=false` keeps the momentum path
bit-identical until then.
