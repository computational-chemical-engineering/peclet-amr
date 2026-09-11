# peclet-amr

> **Status: 0.x, under active development.** Functional and tested (92 C++ + 3 Python ctests, np = 1–8,
> byte-gated), but a research code: no external validation page yet, unfinished rungs recorded in
> `CLAUDE.md`, and an API that may change between minor versions until it graduates to 1.0.0 on its own
> merits (suite `docs/QUALITY_PLAN.md` D9, 2026-09-11). The rest of the peclet family is 1.0.0 and semver.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/computational-chemical-engineering/peclet-amr/blob/main/LICENSE)
[![CI](https://github.com/computational-chemical-engineering/peclet-amr/actions/workflows/ci.yml/badge.svg)](https://github.com/computational-chemical-engineering/peclet-amr/actions/workflows/ci.yml)

The adaptive-mesh-refinement package of the **peclet** suite: a **block-local-Morton AMR octree**
and the **collocated-projection incompressible Navier–Stokes solver** with a cut-cell immersed
boundary that runs on it (`peclet::amr`; Python `peclet.amr`). Relocated out of `peclet-core` with
its git history on 2026-09-10 (suite `docs/QUALITY_PLAN.md` D6 / G.2): core is the infrastructure
layer every method code uses, AMR is a method under active development, so it is its own package.

Header-only C++20 over [peclet-core](https://github.com/computational-chemical-engineering/peclet-core)
(decomposition, halos, SDF geometry, the face-CSR solver layer) and
[peclet-morton](https://github.com/computational-chemical-engineering/peclet-morton) (Z-order codes),
compiled through **Kokkos** (CUDA / HIP / OpenMP) over **MPI** (both required). The suite-wide design
contract lives in `../docs/` ([architecture](../docs/ARCHITECTURE.md),
[conventions](../docs/CONVENTIONS.md), [naming](../docs/NAMING.md), [style](../docs/STYLE.md)).

## What it is

- **Octree** (`include/peclet/amr/block_octree.hpp`, `refine.hpp`, `adapt.hpp`, `indicators.hpp`,
  `leaf_field.hpp`, `vtu_io.hpp`) — the per-block linear octree on Morton codes: 2:1 balance, refinement
  to a sphere / any SDF / a **graded** per-point target level (`refineToSdfGraded`, `gapFloorTarget`),
  Löhner-driven solution adaptivity with conservative remap, per-leaf fields, VTU output.
- **Distributed octree** (`distributed_octree.hpp`, `leaf_halo.hpp`, `distributed_adapt.hpp`,
  `distributed_view.hpp`, `distributed_fv.hpp`, `distributed_poisson.hpp`, `distributed_flow_mg.hpp`) —
  one ORB block per rank on core's `BlockDecomposer`, the ±2 leaf ghost registry (`LeafHalo`),
  cross-block balance, distributed adapt, and **weighted-ORB rebalancing** that migrates leaves with
  their fields (`DistributedOctree::rebalance`).
- **Flow** (`flow.hpp`, the device `AmrFlow`; `flow_oracle.hpp` the serial host reference) — the
  collocated Stokes / Navier–Stokes step: implicit cut-cell momentum (`cut_cell.hpp`,
  `momentum_assembly.hpp`, the Galerkin `MomentumMG` of `momentum.hpp`, `velocity_mg.hpp`) and the
  Almgren–Bell–Colella projection with the **ghost projection** (`ghost_projection.hpp`, the default)
  or the **mixed-level sampled cut band** (`ghost_projection_sampled.hpp`, `AmrFlow::setGhostSampled`),
  MG-preconditioned CG for the pressure (`poisson.hpp`, `multigrid.hpp`, `pcg.hpp`, `fv_op.hpp`),
  quadratic coarse/fine schemes (`cf_scheme.hpp`), scalar transport (`scalar_transport.hpp`),
  and the whole step multi-rank through the leaf halo. Single-rank and np=1 are bit-identical.
- **Python** (`python/amr_bindings.cpp` → `peclet.amr`): `Octree`, `DistributedOctree`, `Poisson`,
  `Flow` (+ `Flow.diagnostics` for the developer instruments), `spacing_from_extent` /
  `spacings_from_extent`. Type stub `packaging/_amr.pyi`.

## Build / test

```bash
source ../.venv/bin/activate          # the suite venv: nanobind, numpy, mpi4py
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp" \
      -DPECLET_AMR_BUILD_TESTS=ON
cmake --build build -j                 # -> build/peclet/amr/_amr.*.so  (import peclet.amr)
OMP_NUM_THREADS=2 OMP_PROC_BIND=false ctest --test-dir build --output-on-failure -LE 'bench|np8'
OMP_NUM_THREADS=1 ctest --test-dir build -L np8          # the 8-rank instances: a local gate
PYTHONPATH=$PWD/build python python/example_amr.py       # three worked examples
```

Requires the sibling checkouts `../core` and `../morton` (or, standalone, the pinned tags fetched
by `cmake/PecletDeps.cmake`), a Kokkos prefix on `CMAKE_PREFIX_PATH` (`../tools/bootstrap_deps.sh`;
`nvidia-cuda` for the GPU) and MPI. Tests are registered only with `PECLET_AMR_BUILD_TESTS=ON`:
92 C++ ctests (28 single-rank + 16 distributed tests at np = 1, 2, 4, 8), plus 5 `bench`-labelled
studies and three Python ctests
(`python_amr`, `python_amr_np2`, and `python_state_hash` — the byte gate of
`python/state_hash.py`, whose reference hashes are `python/state_hash_reference.json` — recorded on
one toolchain and SKIPPED on any other). A test that
cannot run in a configuration exits 77 and ctest reports it **skipped**, never passed. `CLAUDE.md`
carries the developer notes, the environment-variable table and the gotchas.

## Documentation

`docs/` holds the four reference notes describing the design as it ships —
[amr_collocated_projection.md](docs/amr_collocated_projection.md) (collocated projection,
`maskSolid`, the div-free face field), [amr_mixed_level_cut_band_plan.md](docs/amr_mixed_level_cut_band_plan.md)
(mixed-level cut band + graded refinement), [amr_setup_parallel_plan.md](docs/amr_setup_parallel_plan.md)
(the parallel `setSolid` builders) and [amr_anisotropic.md](docs/amr_anisotropic.md) (per-axis root
spacing). Dated campaign records live in [docs/archive/](docs/archive/README.md); the measurement
logs they cite in `docs/data/`; the campaign drivers in `tests/study/`. Doxygen API pages
(`docs/Doxyfile`) are published to GitHub Pages by `.github/workflows/docs.yml`.

## Status

Under active development. The octree, the distributed octree with rebalancing, the collocated
projection with the ghost / sampled cut bands and the distributed step are complete and tested
(np = 1 bit-exact to single-rank, np = 2/4/8 in the march class the seam tests gate); the open
items are those `docs/archive/amr_march_perf_and_distributed_plan.md` names (the production
`rho`/`N` choice for the least-squares clouds, sub-face closures, pocket exclusion).
