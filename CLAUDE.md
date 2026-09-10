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

**Counts** (host-openmp): **92** C++ ctests (28 single-rank + 16 distributed binaries × np = 1, 2,
4, 8) + 5 `bench` (four `study_amr_*` + `bench_amr_flow`) + 3 `python` (`python_amr`,
`python_amr_np2`, `python_state_hash`) = 100. The battery was 92 + 2 Python ctests in core's
`build_rel_k` / `build_rel_py` before the move and reproduces here test for test.

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
  simulation; `Flow.diagnostics` (a view holding a reference to the Flow — `last_mom_iters`,
  `last_pres_iters`, `last_outer_iters`, `divergence_norm_face`, and the solver-internals /
  ablation switches `set_momentum_mg`, `set_momentum_gs`, `set_velocity_mg_staircase`,
  `set_momentum_mg_solver`, `set_ghost_gradient`, `set_aperture_order`) is what a developer uses to
  inspect or ablate. String modes: `set_cf_scheme('standard' | 'quadratic')`,
  `set_advection_scheme('sou' | 'koren')`.
- Design notes (`docs/`): `amr_collocated_projection.md` (the collocated projection + `uf`
  advection), `amr_mixed_level_cut_band_plan.md`, `amr_setup_parallel_plan.md` (the parallel
  builders, D1′), `amr_anisotropic.md` (per-axis root spacing). The dated campaign records are in
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
