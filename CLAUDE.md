# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`peclet-amr` is the adaptive-mesh-refinement package of the **peclet** suite (sibling repos under
`../`: `core`, `morton`, `flow`, `dem`, `voro`, `pnm`, `coupling`): the block-local-Morton **AMR
octree** and the **collocated-projection Navier–Stokes solver** with a cut-cell immersed boundary
that runs on it — `peclet::amr` in C++, `peclet.amr` in Python. It was the `amr/` subtree of
`peclet-core` until **2026-09-10**, when suite `docs/QUALITY_PLAN.md` D6 / G.2 relocated it here
*with its git history* (`git log --follow` reaches back through `core/include/peclet/core/amr/` and
the `tpx/amr/` era). Nothing was deleted; the ctest names, the docs and the campaign logs came along.

Header-only C++20. Depends on **`peclet-core`** (`../core`: decomposition, halos, SDF geometry, and
the face-CSR solver layer `peclet/core/solver/` that was lifted out of this tree in the same move) and
**`peclet-morton`** (`../morton`), on **Kokkos** (required: CUDA / HIP / OpenMP, chosen by the prefix
`CMAKE_PREFIX_PATH` points at) and on **MPI** (required: the distributed octree / flow and the Python
module link it, and core's `mpi_stub.hpp` covers the host halo only, not the Kokkos grid halo the
device flow reaches — a Kokkos + no-MPI build was never a core configuration either). The suite-wide contract is in `../docs/`
(`ARCHITECTURE.md`, `CONVENTIONS.md`, `NAMING.md`, `STYLE.md`); read them before cross-cutting changes.

## Settled decisions — do not reverse silently

Chosen *against* the obvious or textbook alternative, on measured evidence. Full entries with
verbatim quotes and provenance in [`../docs/decisions/amr.md`](../docs/decisions/amr.md); the index is
[`../docs/DECISIONS.md`](../docs/DECISIONS.md). Reversing one takes a new recorded decision, not a
judgement call in the moment.

- **The collocated pressure coupling is the Almgren–Bell–Colella approximate projection — NEVER
  Rhie–Chow.** The residual cell divergence is *intrinsic* to cell-centred velocity placement. This
  has been re-proposed by mistake repeatedly; it is not an "upgrade".
- **AMR keeps the ORB block decomposition**, not a global SFC partition.
- **The pressure smoother is MG-PCG, NOT multicolor-GS**; Chebyshev was not pursued for pressure.
- **The rebalance weight grid is defined over root cells**, not fine cells.
- **Cut-cell openness α is evaluated at the finer neighbour's actual lower corner.**
- **MG-as-solver with a Picard outer loop projects ONCE per step**, never inside the loop.
- **Projection, MG transfers and V-cycle orchestration are deliberately NOT consolidated** onto a
  shared abstraction — that separation is intentional.

## Build / test

```bash
source ../.venv/bin/activate    # THE suite venv (nanobind, numpy, mpi4py)
cmake -S . -B build_q -DCMAKE_PREFIX_PATH=/home/frankp/Codes/suite/extern/install/host-openmp \
      -DPECLET_AMR_BUILD_TESTS=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
cmake --build build_q -j8                                  # -> build_q/peclet/amr/_amr.*.so
OMP_NUM_THREADS=2 OMP_PROC_BIND=false ctest --test-dir build_q --output-on-failure -LE 'bench|np8'
OMP_NUM_THREADS=1 OMP_PROC_BIND=false ctest --test-dir build_q --output-on-failure -L np8   # last
ctest --test-dir build_q -L bench                          # studies + bench_amr_flow (~4 min)
PYTHONPATH=$PWD/build_q OMP_NUM_THREADS=1 python python/state_hash.py --check python/state_hash_reference.json
export PATH=/usr/local/cuda-13.2/bin:$PATH                 # for the nvidia-cuda prefix
```

**Counts** (host-openmp): **115** C++ ctests (31 single-rank + 21 distributed binaries × np = 1, 2,
4, 8) + 5 `bench` (four `study_amr_*` + `bench_amr_flow`) + 8 `python` (`python_amr`,
`python_amr_np2`, `python_amr_seam_records` ×3 at np = 1, 2, 4, `python_state_hash`,
`python_flow_parity`, `python_amr_tg_graded` — the last also carries the `bench` label) = **128**;
`-LE 'bench|np8'` runs 101 of them and `-L np8` the remaining 21. (111 / 124 until 2026-09-24,
when WO4b's `amr_mg_stages` — the sibling-merge and repartition stages — added a 21st distributed
binary; 92 / 101 until 2026-09-21,
when `amr_distributed_cf` — the distributed C/F quadratic scheme — added a 17th distributed
binary; 96 / 106 until 2026-09-23, when ROADMAP C1 added `amr_mg_lift`, `amr_mg_predict` and
`amr_mg_bottom` single-rank and `amr_mg_lift_dist`, `amr_mg_tail` and `amr_mg_bottom_dist`
distributed.) The battery was 92 + 2 Python ctests in core's `build_rel_k` / `build_rel_py` before the
move and reproduces here test for test.

**ctest protocol** (`cmake/PecletAmrTest.cmake`, the ONE place every test is registered through;
QUALITY_PLAN §3.D): a binary that cannot run exits 77 (`tests/test_util.hpp kSkipExitCode`; MPI
tests via `tests/test_skip_mpi.hpp`, which Init/Finalizes first so the launcher forwards the 77) and
ctest reports it "Not Run (skipped)", never Passed. Labels: `mpi` (every mpirun test), `np8` (a
LOCAL gate — CI's 4-core runners run `-LE np8`), `bench`, `python`. The launcher is pinned to the
MPI we link (`cmake/PecletAmrPinMpiexec.cmake`): a foreign `mpiexec` on PATH (ParaView's) makes every
rank a singleton and N non-communicating copies "pass"; the Python tests also exit non-zero when
`comm.size` differs from the `PECLET_AMR_TEST_NP` ctest launched them with. Thread bounds on this
host: `OMP_NUM_THREADS=2 OMP_PROC_BIND=false`, np=8 subset last (the nvidia-cuda prefix carries an
OpenMP host backend — an unbounded pool on 48 cores is an hour-long trap).

**The byte gate** (`python/state_hash.py`, QUALITY_PLAN §3.G): fixed-seed reference runs of every
public entry path — `Octree` refine/balance/adapt, `Poisson`, `Flow` with the ghost projection and
with the mixed-level sampled band, `DistributedOctree` + the distributed `Flow` (np = 1 and, under
`mpirun -np 2`, np = 2) — hashed (SHA-256 of the final arrays). `--save` records, `--check` compares;
`python/state_hash_reference.json` is the committed reference (a host-openmp Release build with the
suite's gcc; it carries a `toolchain` key, and a build of another compiler / version / build type
SKIPS the gate with exit 77 — CI shows it skipped — because the last bits of a Krylov solve are
toolchain-specific; `-O0` vs `-O3` alone changes them through FMA contraction). Any structural
change must leave every hash identical on the recording toolchain; a numerics change re-records the
reference (`--save`) in its own commit and says so.

**The graded time-accurate gate** (`tests/study/amr_tg_graded.py --gate`, ctest
`python_amr_tg_graded`, `docs/amr_tg_graded.md`): a decaying Taylor-Green vortex with advection on,
against its exact solution, on a graded octree. Four facts in ~45 s: the advecting face field is
still a conservative flux there (`div(uf) <= 1e-9`), the pressure-increment leak at the 2:1
sub-faces stays under 5 % of the regular faces' own error (gate W of
`docs/amr_pressure_iteration.md` §14.3 — today 0.4 %), a 2:1 sub-face is not a worse place for the
advecting velocity than an ordinary one, and the error levels themselves are within 5 % of what
`amr_tg_graded.md` §3 records. Labelled `bench` — it runs with the studies, not in the default
battery.

**The uniform-grid parity gate** (`tests/study/flow_parity/parity_gate.py`, ctest
`python_flow_parity`, `docs/amr_flow_uniform_parity.md`): at `lmax = 0` the octree IS the
structured brick `peclet.flow` runs on — same extent, spacing and cell centres — so `Flow` and
`flow.SolverColocated` must agree to solver tolerance, and six cases gate that from the shared
7-point core up to the cut-cell ghost projection. `flow` is a SIBLING REPO and the two modules
cannot share an interpreter, so the gate shells out one subprocess per engine and is pointed at
flow's build tree by the CMake cache variable `PECLET_AMR_FLOW_PYTHONPATH` (a path, not a numerics
switch). Unset — CI, and any checkout without a built `flow` — it exits 77 and ctest reports it
skipped. Configure it with
`-DPECLET_AMR_FLOW_PYTHONPATH=/path/to/flow/build_parity` and it runs in ~40 s.

CI (`.github/workflows/ci.yml`): gcc + clang Release (tests, Python on gcc) and gcc Debug (minus
`amr_flow_solver`, >40 min unoptimised), each with `core` (at `PECLET_CORE_REF`, `main` until the core release that
carries the solver layer) and `morton` (pinned tag) checked out as siblings. `quality.yml`:
clang-format 18.1.8 is **blocking** over `include/ tests/ python/ benchmarks/` (the tree was
reformatted once on relocation; run `clang-format -i` on every C++ file you touch).

CMake identifiers: `project(peclet_amr VERSION …)` reads the version from `pyproject.toml` (the one
version source; `CITATION.cff` and `docs/Doxyfile` are checked against it by the suite pre-flight);
target `peclet_amr` / `peclet::amr`; options `PECLET_AMR_BUILD_PYTHON`, `PECLET_AMR_BUILD_TESTS`,
`PECLET_CUDA_RUNTIME_WHEEL`; `cmake --install` exports
`find_package(peclet-amr CONFIG)`. Dependencies through `cmake/PecletDeps.cmake` (sibling checkouts
first, else fetched at `PECLET_CORE_TAG` / `PECLET_MORTON_TAG`; Kokkos from the prefix or vendored).
Python: `packaging/amr_init.py` is the package `__init__`, `packaging/_amr.pyi` the stub — regenerate
with `PYTHONPATH=build_q python -m nanobind.stubgen -m peclet.amr._amr -o packaging/_amr.pyi` after
changing a binding.

## Architecture

Header-only under `include/peclet/amr/` (namespace `peclet::amr`; `common.hpp` carries one
`using namespace peclet::core` so the sources keep core's vocabulary — `Index`, `Vec`, `View`,
`geom::`, `halo::` — unqualified, exactly as they did inside core):

- `block_octree.hpp` — the per-block octree (`BlockOctree<Dim, Bits>`: Morton-coded leaves, 2:1
  balance, `find`, face neighbours); `block_octree_view.hpp` its device view; `refine.hpp`
  (`refineToSdf`, `refineToSdfGraded` / `gapFloorTarget`), `adapt.hpp` + `indicators.hpp` (Löhner),
  `leaf_field.hpp`, `vtu_io.hpp`, `csr.hpp`, `barnes_hut.hpp` (the particle-tree mode).
- `distributed_octree.hpp` — `DistributedOctree<Dim>`: one block per rank on core's ORB
  `BlockDecomposer`, cross-block balance, `faceNeighborGather`, and `rebalance` (the Eulerian
  weighted-ORB load balancer, migrating leaves + fields). `leaf_halo.hpp` is the ±2 leaf ghost
  registry every distributed solve reads through; `distributed_adapt.hpp`, `distributed_view.hpp`,
  `distributed_fv.hpp`, `distributed_poisson.hpp`, `distributed_flow_mg.hpp` the distributed
  variants of the solver pieces.
- `flow.hpp` — the canonical device `AmrFlow` (collocated-projection Navier–Stokes with `maskSolid`
  and a div-free face field); `flow_oracle.hpp` an unexposed serial host reference. Device +
  distributed multigrid live in `pcg.hpp`, `multigrid.hpp`, `velocity_mg.hpp`, `momentum.hpp`
  (the Galerkin `MomentumMG`; the operator, colouring and BiCGStab it drives are
  `peclet::core::solver`, aliased back into this namespace) and the `distributed_*.hpp` set.
  **The pressure multigrid continues BELOW the root brick** (ROADMAP C1, `docs/amr_mg_depth.md`):
  a coarser level is the same `BlockOctree` with its root *lifted* — brick halved, `lmax`
  incremented, leaf codes untouched — so `coarsenIf` keeps merging and every builder downstream
  works verbatim, and a uniform mesh has a real hierarchy instead of the single level it used to
  have (64³: 1 → 5 levels, 1941.8 → ~300 ms/step). Distributed, the lift is lockstep (the depth is
  Allreduced before any level is built) and the ORB follows by `BlockDecomposer::coarsened`; where
  the blocks turn odd a `MgStage` (`mg_stage.hpp`) moves the level onto a new decomposition of its
  own grid, with target, communicators and movement from core (`peclet::core::decomp`:
  `chooseStageTarget`, `makeStageComm`, `RedistributeTopology`; `docs/amr_mg_core_boundary.md`) —
  by default (`PressureStagePolicy`, ON since 2026-09-24) a sibling merge or repartition onto fewer
  ranks whose continued ladder is a `DistributedFlowMultigrid` on the stage's sub-communicator
  (`DistributedStage`, WO4b; `maxBlockCells` = the largest finest block's LEAF count, `minExtent` 4
  and inert — `amr_mg_core_boundary.md` §9.6–9.7), the replicated tail where nothing lifts or
  with the policy off — and `rebalance` aligns its weighted ORB coarse-first
  (`chooseAlignedWeighted`, 1.05 budget) — and where the ladder runs out above
  `bottomExtent` the bottom is an agglomerated `GraphAMG`-PCG solve (`amg_bottom.hpp`,
  `Flow.set_pressure_bottom`). The momentum path is NOT lifted (`liftRoot = false`,
  guarded by `minCoarse`) until WO8 of `amr_mg_depth.md` §9 lands. `predictPressureLadder`
  (`mg_predict.hpp`, Python `predict_hierarchy`) is the ladder rule as a pure function;
  tests assert the built ladder against it, never against a literal level count.
  Cut-cell openness is `cut_cell.hpp` (host oracle assembly) with `assembly.hpp`,
  `momentum_assembly.hpp`, `facegeom_assembly.hpp` / `face_geom.hpp` the device builders;
  `cf_scheme.hpp` the quadratic coarse/fine schemes; `scalar_transport.hpp` + `advect_recon.hpp`
  transport.
  The immersed boundary has **two projection schemes**. The default (AUTO since 2026-08-25) is the
  **ghost projection** — `ghost_projection.hpp`, the fluid-only constraint scheme, selected
  explicitly with `setGhostProjection(true)`; it falls back to the older aperture projection with a
  stderr notice when the finest band is too thin for its ±2 closure reach. `(2, 2)` closure orders
  are the only pair cleared for production.
  On top of that sits the **mixed-level cut band** (`ghost_projection_sampled.hpp`,
  `AmrFlow::setGhostSampled` / Python `Flow.set_ghost_sampled`): it drops the uniform-finest-band
  contract so cut cells may live at SEVERAL octree levels — closure chain entries that cross a 2:1
  boundary become degree-2 least-squares virtual samples, and identity weights at same level keep a
  uniform band bit-identical to the classic path. Its mesh-generator side is
  `refine.hpp::refineToSdfGraded` / `gapFloorTarget` (Python `Octree.refine_to_sdf_graded` /
  `refine_to_gap_floor`). **DISTRIBUTED since 2026-08-30** (rungs D0–D2): the least-squares clouds
  are a deterministic probe set resolved through `probeSlot` (never a search over the local leaf
  array), the sampled builders probe inside `prepareDistributed`'s miss-collect fixpoint, and the
  clouds read ghost slots like any other CSR the step consumes. Acceptance: np=1 bitwise vs
  single-rank, np=2/4 in the ~3e-7 march class (`tests/test_amr_distributed_seam_mpi.cpp`).
  The two cloud-economy knobs of the M2a study are **explicit arguments** since 2026-09-10 (QUALITY_PLAN
  D3 — no environment variable changes a result): `setGhostSampled(on, rho = 2.2, maxSamples = 0)` /
  `set_ghost_sampled(on, rho=2.2, max_samples=0)` — `rho` the LS radius factor, `max_samples` the
  nearest-N candidate cap (0 = uncapped). The defaults are exactly the shipped behaviour; do not
  change them in production without reading the M2a table.
  Read `docs/amr_mixed_level_cut_band_plan.md` before touching any of it: it holds the design (§4),
  six decisions each with the alternative to revisit (§5), the measured phase results, and the risk
  register — of whose three unfinished rungs the distributed sample halo is DONE, leaving sub-face
  closures and pocket exclusion in LS clouds.
- `python/amr_bindings.cpp` (→ `peclet.amr`, nanobind over core's zero-copy `View`↔ndarray bridge):
  `Octree` / `DistributedOctree` (`Octree(cells, *, lmax, origin, spacing | extent)` — `cells` is the
  FINEST grid, as in flow; the root brick is `cells / 2**lmax`; the 16 members the two share are
  bound ONCE by `bindOctreeCommon<T>`), `Poisson`, and `Flow` over the device `AmrFlow`. **Two API
  tiers** (QUALITY_PLAN D2): the public surface is what a user needs to set up, run and read out a
  simulation; `Flow.diagnostics` (a view holding a reference to the Flow — the instruments
  `last_mom_iters`, `last_pres_iters`, `last_outer_iters`, `divergence_norm_face`,
  `face_topology`, `num_cf_cut_faces`, `num_seam_sample_records`, `num_seam_layer_records`,
  `pressure_mg_levels`, `pressure_mg_bottom`; and the solver-internals / ablation switches
  `set_momentum_mg`, `set_momentum_gs`, `set_velocity_mg_staircase`, `set_momentum_mg_solver`,
  `set_ghost_gradient`, `set_aperture_order`, `set_uf_advection`, `set_seam_reconstruction`) is
  what a developer uses to inspect or ablate. Every switch there has a production default; its docstring
  says what it is and why the switch exists. Two of the scheme selectors take an INT —
  `set_cf_scheme(0 = standard | 1 = quadratic)`, **before `set_solid`**, which is where the C/F
  overlays are built, and `set_advection_scheme(0 = SOU | 1 = Koren)`. That is a **divergence from
  `flow`, not a convention to copy**: `../docs/NAMING.md` wants one spelling per concept and `flow`
  spells these as strings. New selectors take a STRING. Where `flow` already has the concept, `amr`
  takes `flow`'s spelling, keyword, strings, default AND tier (flow is the reference): the
  pressure bottom is the public `Flow.set_pressure_bottom(mode="auto" | "smoother" |
  "agglomerated")` / `Flow.set_pressure_bottom_extent(cells=)` / `Flow.pressure_bottom_extent`,
  and the forecast is `predict_hierarchy` (until 2026-09-24, never released: `diagnostics.` with
  `kind=` / `extent=`, and `predict_pressure_hierarchy`). `cells=` there is a count of cells per
  axis on the coarsest grid, the one exemption from "never add cell-unit API" (`../docs/NAMING.md`
  §1.8). `predict_hierarchy` keeps amr's canonical arguments (`cells`, `lmax`, `num_ranks`,
  `bottom_extent`); `flow`'s `gnx, gny, gnz, np, …` are flow's own open NAMING row. The pressure
  DRIVER is `flow`'s public `set_pressure_pcg(on)`, but `on=False` selects amr's stationary
  V-cycle (flow has none and raises) and the cap / tolerance stay `pres_iters` /
  `set_pressure_tolerance` rather than flow's `max_iter` / `rtol` arguments. The two int
  selectors are NAMING items; do not copy either side's spelling into a third code without reading
  `../docs/NAMING.md`.
- Design notes (`docs/`, indexed by `docs/README.md`): `ROADMAP.md` (the one page of live items —
  **start here**),
  `amr_flow_uniform_parity.md` (what this solver shares with `peclet.flow`'s collocated solver at
  `lmax = 0`, measured cell by cell, and the two places it does not),
  `amr_mg_depth.md` (ROADMAP C1: what a multigrid level below the root brick IS, the lockstep lift,
  the telescoping stages and the exact bottom — with §11 carrying the measured open questions, and
  §11.10 the warning that the depth study's iteration column is a sample, not a constant),
  `amr_mg_core_boundary.md` (which half of the stage machinery belongs in `core`),
  `amr_collocated_projection.md` (the collocated projection + `uf`
  advection), `amr_tg_graded.md` (the graded time-accurate benchmark: second order in an unsteady
  flow, and why the C/F pressure-increment leak does not need fixing) with its design/verdict note
  `amr_pressure_iteration.md`, `amr_cf_convective.md` (the advected value at a 2:1 seam —
  level-aware upwind probes, and why what remains is intrinsic), `amr_cf_flux_gate.md` (the C/F
  face-value delta gated per FACE, so `uf` stays a flux), `amr_graded_convergence.md` (steady
  second order across a 2:1 interface, and the A5 cut-row fix), `amr_mixed_level_cut_band_plan.md`,
  `amr_setup_parallel_plan.md` (the parallel builders, D1′; §7 the distributed C/F scheme),
  `amr_anisotropic.md` (per-axis root spacing). `docs/briefs/` keeps the brief each design pass
  answered. The dated campaign records are in
  `docs/archive/` behind its README index — `amr_march_perf_and_distributed_plan.md` (march economics
  + the distributed band; its status table names the two items still open), `amr_distributed_flow.md`,
  `amr_device_assembly_plan.md`, `amr_aperture_advection_plan.md`, `comm_avoiding_pressure_driver.md`,
  `snellius_amr_bed.md`. Their measurement logs are `docs/data/`; the drivers `tests/study/*.py`
  (campaign tools, not ctests — `bed` scripts need the RCP packs they name).

## Environment variables

None changes a result (QUALITY_PLAN D3). The three that exist switch profiling / debug prints:

| variable | effect |
|---|---|
| `PECLET_AMR_PROFILE_SETUP=1` | per-phase timings of `setSolid` (host builders, overlays, uploads) to stderr |
| `PECLET_AMR_PROFILE_STEP=1` | the M0 step profiler: fenced per-phase timings over a window of steps (`PECLET_AMR_PROFILE_STEP_WINDOW`, default 50) |
| `PECLET_AMR_PRES_DEBUG=1` | per-cycle / per-solve pressure residual + RHS-compatibility trace to stderr (single-rank) |

(Until 2026-09-10 these were `PECLET_CORE_PROFILE_*` / `PECLET_CORE_AMR_PRES_DEBUG`, and
`PECLET_CORE_GPS_RHO` / `PECLET_CORE_GPS_MAXN` changed the sampled-band numerics — those two are the
`rho` / `max_samples` arguments of `set_ghost_sampled` now.) The transport / logging variables of
core's grid halo (`PECLET_CORE_GPU_AWARE_MPI`, `PECLET_CORE_HALO_VERBOSE`, `PECLET_CORE_HALO_TIMEOUT`)
still apply to the leaf halo.

## Gotchas

- **Consecutive NBX rounds and direct tags** (core's `CLAUDE.md`): the AMR direct point-to-point tags
  are 11 / 41 / 45 and stay below 24576; core's `NbxEngine` rotates its own wire tags. Until
  2026-09-05 family-0 NBX rounds walked over the AMR gather tags and
  `amr_distributed_{fv,mg,graded_mg,openness,poisson}_np{4,8}` returned wrong ghosts intermittently.
- The Python wrappers are `Releasable` (core's `kokkos_teardown.hpp`): the atexit hook drops their
  Views BEFORE `Kokkos::finalize()`, so an `Octree`/`Flow` at script scope does not abort the process
  on exit (CUDA: `cudaErrorCudartUnloading`). Every new wrapper class must register the same way.
- `Flow.set_solid` releases the GIL around the host builders' parallel regions (a Python SDF callable
  sampled from Kokkos host threads deadlocks otherwise at `OMP_NUM_THREADS > 1`); `finish_adapt` and
  `rebalance_mpi` rebuild through the same path. Prefer `set_solid_spheres` at bed scale — the
  callback, not the solve, dominates (measured >1h43m of numpy on an 11.35M-leaf bed).
- `set_dt` is baked into the momentum operator: a dt switch is `velocity()/pressure()` → `set_dt` →
  `set_solid` → `set_velocity()/set_pressure()`.
- `core/python/build*/`-style stale trees naming `tpx_amr` were NOT carried over; a fresh
  `build_q` is the recipe.
