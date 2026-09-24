# Design brief: the pressure multigrid cannot coarsen below the octree's root brick

> For an architect design pass, 2026-09-23, on branch `c1-advective-cost` of `peclet-amr` (`ed33b51`,
> base `7d4dac9`). ROADMAP item **C1**, re-diagnosed by profiling — the old entry blamed the
> momentum path and was wrong.

## 1. The question

**What should a multigrid level BE in this package below the octree's root brick, and how does the
ORB block decomposition follow it — so that a V-cycle's coarsest level is effectively solved on a
cut-cell mesh at any `lmax`, including `lmax = 0`?**

A decision-shaped answer says what the sub-root levels are made of, who owns them under MPI, how
cut-cell openness reaches them, where the hierarchy stops and what solves the bottom, and what it
costs to build and to run.

## 2. Why it needs the architect

Three things put it above routine work.

1. **It changes what a level is.** Every level today is an `Octree` and every operator is built by
   `Poisson::init(octree, h0)`. Below the root brick there are no octree cells left to merge, so a
   sub-root level is a different object. That ripples through `ops_`, `c2p_`, the device mirror,
   the transfer CSRs and the distributed variants.
2. **It touches the decomposition.** The ORB decomposes the **root grid**; each rank owns a
   sub-brick of root cells (`distributed_octree.hpp:63–80`). Coarsening below the root means
   coarsening *through* the partition — at some depth a rank's sub-brick is one cell, and below
   that the levels must span ranks. `../docs/DECOMPOSITION_AND_MULTIGRID.md` is the suite's
   reference for exactly this interaction and should be read before deciding.
3. **The cut-cell case is the one that matters and the one the existing partial answer excludes**
   (§4). Production meshes have openness everywhere near the solid.

## 3. What is wrong, measured

`AmrMultigrid::build` coarsens by merging octree siblings; a leaf that is already a **root** cell
has no siblings, so the hierarchy stops at the root brick:

    levels = lmax + 1,    coarsest grid = the root brick = (cells / 2**lmax)³

**On a uniform mesh, at any `lmax`, that is ONE level** — the "MG preconditioner" is a smoother with
no coarse-grid correction at all.

Profile of the 32³ cut-cell sphere, advection on, ghost projection, host-openmp 4 threads
(`PECLET_AMR_PROFILE_STEP=1`):

```
[step-prof] leaves 32768 | pressure MG levels 1: 32768
[step-prof] 20 steps | 147.729 ms/step | mom 5.7 it, pres 12.8 it
[step-prof]   advection build           0.723 ms/step    0.5%
[step-prof]   momentum solve           10.161 ms/step    6.9%
[step-prof]   pressure solve          136.289 ms/step   92.3%
[step-prof]     . MG preconditioner   133.710 ms/step   90.5%  (nested)
```

**The cost tracks the root brick, not the cell count** (`tests/study/amr_pressure_depth.py`):

| N | lmax | mesh | leaves | root brick | levels | ms/step | pres it |
|---|---|---|---|---|---|---|---|
| 64 | 0 | uniform | 262 144 | 64³ | **1** | 1947.0 | 23 |
| 64 | 1 | graded | 83 672 | 32³ | 2 | 210.7 | 14 |
| 64 | 2 | graded | 65 360 | 16³ | 3 | 71.9 | 12 |
| 64 | 3 | graded | 64 240 | 8³ | 4 | 58.8 | 12 |

**The prize, measured by emulation.** Refining *every* leaf to the finest level inside a tree of
depth `lmax` gives the **identical uniform 64³ mesh** (262 144 leaves, same geometry, same answer)
with a hierarchy of `lmax + 1` levels — i.e. exactly what this design would provide, built with the
machinery that already exists:

| lmax | MG levels (leaves per level) | ms/step | pres it |
|---|---|---|---|
| 0 | 1: 262144 | **2035.9** | 23 |
| 1 | 2: 262144 32768 | 416.8 | 15 |
| 2 | 3: 262144 32768 4096 | 274.8 | 12 |
| 3 | 4: 262144 32768 4096 512 | **252.0** | 12 |

**8.1× from depth alone**, saturating at four levels with a 512-cell (extent 8) bottom — above
`flow`'s agglomeration threshold of 4, so an exact bottom should still add. For scale: `flow` on
the 32³ advection case is 27.5 ms / 32 768 cells = **0.84 µs/cell**; `amr` at `lmax = 3` here is
252 ms / 262 144 = **0.96 µs/leaf**. The depth fix closes essentially the whole gap.

Head-to-head on the same case, host, threads (`tests/study/flow_parity/`): Stokes **amr 163 ms vs
flow 180 ms** — amr *faster*; with advection **amr 148 ms vs flow 27.5 ms**. Both codes' momentum
cost collapses under the implicit FOU; flow's pressure solve is cheap either way and amr's simply
becomes the whole step.

## 4. Current state, with the code

**Host hierarchy** (`include/peclet/amr/poisson.hpp:640`):

```cpp
void build(const Octree& finest, const Vec<Dim>& h0) {
  levels_.clear();
  levels_.push_back(finest);
  for (;;) {
    Octree c = levels_.back();
    Index merged = c.coarsenIf([](Code, unsigned) { return true; });
    if (merged == 0 || c.numLeaves() == levels_.back().numLeaves())
      break;                       // <-- the floor: a root cell has no siblings
    levels_.push_back(c);
    if (c.numLeaves() == 1) break;
  }
  ops_.resize(levels_.size());
  for (std::size_t L = 0; L < levels_.size(); ++L)
    ops_[L].init(levels_[L], h0);          // every level IS an Octree
  c2p_.assign(levels_.size() ? levels_.size() - 1 : 0, {});
  for (std::size_t L = 0; L + 1 < levels_.size(); ++L)
    hostParFor(levels_[L].numLeaves(), [&](Index i) {
      c2p_[L][i] = levels_[L + 1].find(levels_[L].code(i));   // covering-leaf
    });
}
```

`coarsenIf` (`block_octree.hpp:217`) merges complete sibling groups only.

**Device mirror** (`multigrid.hpp:268`) resizes from `hmg_->numLevels()`; per level it holds
`n, op (face CSR), x, b, res, tmp, kappa, c2p, childStart, childIdx`. **The bottom is pure
smoothing** — there is no coarse solve anywhere:

```cpp
void vcycle(int pre=2, int post=2, int bottom=40, double omega=0.8, std::size_t L=0) {
  Level& lv = levels_[L];
  if (L + 1 == levels_.size()) {            // coarsest level
    for (int s = 0; s < bottom; ++s)
      jacobiFv(lv.op, lv.x, View<const double>(lv.b), lv.tmp, omega);
    return;                                  // <-- that is the whole "bottom solve"
  }
  ...
}
```

**A partial answer already exists, and excludes the case that matters** (`distributed_fv.hpp:395`).
The distributed builder coarsens the same way, pads ranks to a common depth, and then:

```cpp
  if (!hasOpen_) {                    // <-- ONLY when there are no cut cells
    AmrGeometry<Dim> ig = geo;
    for (int d = 0; d < Dim; ++d)
      ig.h0[d] = geo.h0[d] * double(Index(1) << lmax);
    inner_ = std::make_unique<DistributedMultigrid<Dim, Bits>>();
    inner_->build(g, ig, per, comm);   // a UNIFORM hierarchy on the root grid, continuing down
  }
```

so an openness-free **distributed** run already telescopes below the root brick through `inner_`,
and `bottomSolve` chains into it (`distributed_fv.hpp:546`). With openness present it falls back to
`bottomSweeps_ = 400` Jacobi sweeps on the coarsest graded level. **Single-rank (`presMG_`) has no
`inner_` at all** — which is what every number in §3 measures.

Note the asymmetry the design should resolve: `AmrMultigrid` *already* coarsens cut-cell openness to
every level by area-averaging (`setOpenness` / `coarsenOpenAvg`, `multigrid.hpp:133`), so the
machinery to carry openness down exists; what is missing is the **grid** below the root.

**The decomposition** (`distributed_octree.hpp:63`): `dec_.init(size_, globalRootSize_)` — the ORB
partitions the **root grid**; each rank's `blockOriginRoot_` / `blockBrick_` is its sub-brick of
root cells. When a rank's local octree coarsens to its root brick, it *is* that sub-brick.

**The velocity multigrid inherits the same floor** (`velocity_mg.hpp:56`): `hmg_->build(finest, h0)`
then a `minCoarse = 256` cap that drops levels below the feature scale. In the Stokes case the
momentum solve is 33.6 % of the step (54.9 ms, 25.6 iterations), so this is not negligible — but see
§8, it is out of scope here.

**`flow`'s reference** (`flow/src/mac_cutcell_mg.hpp:1821, 1857, 2056`): geometric coarsening to a
requested depth, then `agglomerateBottom()` when the coarsest global grid exceeds
`PECLET_FLOW_AGGLOM_EXTENT` (4) **cells on any axis** (and only for the singular/periodic operator);
`buildAmg` assembles the coarsest level's rows keyed by **global cell id** with periodic wrapping;
`graphAmgSolveBottom` all-gathers `(gid, rhs)`, has **every rank solve the identical global CSR**
with GraphAMG-preconditioned CG, and scatters back by id — no rank-0 bottleneck, and
decomposition-independent by construction (measured np=6 vs np=1 to 4.5e-16).
`../docs/DECOMPOSITION_AND_MULTIGRID.md` §2.7 has the measured table: an exact agglomerated bottom
is **depth-independent** (4.0 iterations at 4 levels and at 8) and beats full geometric depth,
because the extra levels cost more than the coarse solve they replace.

## 5. Constraints and invariants

- **Cut cells are the production case.** Any answer that only works openness-free repeats the
  current gap. `AmrMultigrid` already area-averages openness to every level; a sub-root level needs
  a consistent openness too, or an argument for why it may drop it.
- **Device and distributed.** Header-only C++20 + Kokkos; all per-level state is device Views and
  the V-cycle is device kernels. Under MPI, per-level `LeafHalo`s exchange ghosts and the singular
  nullspace is removed by an Allreduce'd volume-weighted mean (`removeMeanFvDist`). Acceptance is
  np = 1 bitwise vs single-rank, np = 2/4/8 in the ~3e-7 march class.
- **The byte gate** (`python/state_hash.py`, 13 keys, np = 1 and `mpirun -np 2`) moves if the
  pressure solve's arithmetic changes — permitted for a deliberate change, re-recorded in its own
  commit, but say which keys and why.
- **`amr_distributed_mg_mpi` pins the level count** (`test_amr_distributed_mg_mpi.cpp:86`:
  `numLevels() == 4` for a 16³ grid). A deeper hierarchy breaks that gate by construction — say
  what it should assert instead.
- **The velocity MG's `minCoarse = 256` cap exists on purpose**: coarsening an immersed feature
  below its own scale makes it vanish and leaves a singular or divergent coarse operator. Whatever
  the pressure hierarchy does below the root must not reintroduce that failure.
- **Performance envelope.** `setSolid` is already the setup bottleneck at bed scale (ROADMAP C2,
  >1h43m on an 11.35M-leaf bed); a hierarchy that is expensive to *build* trades one problem for
  another. State the build cost.

## 6. Already decided — do not reopen

From `../../docs/decisions/amr.md` and `../../docs/DECISIONS.md`:

- **The pressure smoother is MG-PCG, not multicolor-GS**; Chebyshev was not pursued for pressure.
- **AMR keeps the ORB block decomposition**, not a global SFC partition.
- **The rebalance weight grid is defined over root cells**, not fine cells.
- **Projection, MG transfers and V-cycle orchestration are deliberately NOT consolidated** onto a
  shared abstraction — that separation is intentional.
- **USER DIRECTIVE — `peclet.flow` is the reference** for shared-method design elsewhere in the
  suite; study it before designing the same method in another code. §4 pastes the relevant part.
- **Every numerical method runs fully on-device and must be MPI-distributable**; host serial paths
  are oracles only.

**Genuinely open — what to decide:** what a sub-root level is (a structured grid? an octree of a
coarser root? something else); who owns it under MPI as the levels shrink past one cell per rank;
how openness reaches it; where the hierarchy stops and what solves the bottom (flow's agglomerated
CSR, the existing `inner_` chain generalised, or something else); and whether the single-rank and
distributed paths converge on one mechanism or stay separate.

## 7. Already established — do not re-derive

- The floor is `coarsenIf` having no siblings to merge at the root level. Not a bug, a consequence.
- The prize is **8.1×** on a uniform 64³ mesh (§3), measured by emulation with the existing
  machinery, and it saturates at four levels — so *depth to a small bottom* is what matters, not
  depth for its own sake. This agrees with flow's §2.7 table.
- The criterion for "small enough" is the coarsest grid's largest **extent**, not its cell count
  (flow: 4 cells on any axis; a 64 × 2 × 2 bottom is 256 cells and still costs 6.0 iterations
  against 4.0).
- An openness-free distributed continuation already exists (`inner_`) and is disabled with cut
  cells; single-rank has nothing.
- Per-cell, `amr` is **not** slower than `flow` once the hierarchy is deep (0.96 vs 0.84 µs/cell).
  The 4–7× in the old ROADMAP entry is this one defect, seen through the uniform parity case.

## 8. How the answer will be verified

- `tests/study/amr_pressure_depth.py` — the table of §3; the uniform rows must stop being outliers.
- The emulation ladder of §3 as the target: a `lmax = 0` uniform 64³ run should approach the
  252 ms/step that `lmax = 3` reaches today, i.e. **≥ 6×** on that case, with pressure iterations
  falling 23 → ~12.
- `tests/study/flow_parity/` — the six parity cases still agree with `flow` to solver tolerance
  (this changes the preconditioner, not the operator, so the converged answer must not move).
- Byte gate at np = 1 and `mpirun -np 2`; the battery (96 C++ + 5 bench + 5 python);
  `amr_distributed_mg_mpi` / `amr_distributed_graded_mg_mpi` at np = 1, 2, 4, 8.
- Cut-cell accuracy unmoved: `tests/study/amr_zh_ladder.py` (Z&H permeability),
  `python_amr_tg_graded`.
- Stability: `tests/study/amr_two_sphere_gap.py` at cf = 1 (note it runs Stokes — use
  `--advection 1` if the advective path matters to the answer).
- Distributed: np = 1 bitwise vs single-rank, np = 2/4/8 in the ~3e-7 class.

## 9. Deliverable

`docs/amr_mg_depth.md`:

1. **The decision**, in one sentence.
2. **What the literature and `flow` do**, and where this design follows or deviates — including
   whether the two halves (deeper levels, exact bottom) are both needed given §3's saturation.
3. **The options considered**, each with why it was or was not chosen — including generalising the
   existing `inner_` chain, flow's agglomerated CSR, and "require large `lmax`" (a policy, not a
   fix — say whether it is defensible on its own).
4. **The design** to implementation detail: what a sub-root level is, its operator, its transfers,
   its openness, its ownership under MPI, and where the hierarchy stops.
5. **Why it cannot degrade accuracy** — it is a preconditioner, so argue that the converged answer
   is untouched and say what would falsify that.
6. **Work orders** for an Opus implementer, and the gates of §8 with expected numbers.
7. **Cost**: build time (§5's warning), memory, and any new collective.

No production code from this pass.

## 10. Out of scope

- The **velocity** multigrid's `minCoarse` cap (§4) — same floor, different operator and a live
  reason for the cap; a separate item if the pressure answer generalises.
- ROADMAP C2 (`setSolid` at bed scale), B6 (matching the flux-error constants), and everything in
  A/A′.
- `peclet.flow` itself.
- The `divergence_norm_face` diagnostic's ghost-scheme reading (A3′, closed).
