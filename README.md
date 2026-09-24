# peclet-amr

> **Status: 0.x, under active development.** Functional, tested (124 ctests: 111 C++ at np = 1–8, 5
> C++ studies, 8 Python; byte-gated) and validated against analytic solutions on graded meshes, but a
> research code: no published validation page yet, open items in [docs/ROADMAP.md](https://github.com/computational-chemical-engineering/peclet-amr/blob/main/docs/ROADMAP.md),
> and an API that may change between minor versions until it graduates to 1.0.0 on its own merits
> (suite `docs/QUALITY_PLAN.md` D9, 2026-09-11). The rest of the peclet family is 1.0.0 and semver.

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
contract lives in the umbrella repository's `docs/` ([architecture](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/ARCHITECTURE.md),
[conventions](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/CONVENTIONS.md), [naming](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/NAMING.md), [style](https://github.com/computational-chemical-engineering/peclet/blob/main/docs/STYLE.md)).

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
  quadratic coarse/fine schemes (`cf_scheme.hpp`, the default), scalar transport
  (`scalar_transport.hpp`), and the whole step multi-rank through the leaf halo. Single-rank and
  np=1 are bit-identical.
- **Pressure multigrid below the root brick** (`poisson.hpp`, `distributed_flow_mg.hpp`,
  `mg_stage.hpp`, `amg_bottom.hpp`, `mg_predict.hpp`) — a coarser level is the same octree with its
  root *lifted* (`BlockOctree::liftRoot`), so the hierarchy keeps coarsening past the root brick
  instead of stopping there: a uniform 64³ mesh goes from 1 level to 5, its pressure iterations
  from 23 to 13 and its step from 1942 to ~300 ms. Distributed, the lift is lockstep; where the ORB
  blocks turn odd the coarsest level is gathered onto every rank and continued (the replicated
  stage); where the ladder runs out above 4 cells per axis the bottom is an exact agglomerated
  `GraphAMG`-PCG solve (`Flow.set_pressure_bottom` / `set_pressure_bottom_extent`, `peclet.flow`'s
  spellings). `peclet.amr.predict_hierarchy` predicts the ladder before a run.
- **The 2:1 seam** (`cf_scheme.hpp`, `seam_recon.hpp`, `advect_recon.hpp`) — the momentum
  diffusion, the divergence constraint and the pressure gradient take the Martin–Cartwright
  tangential quadratic at a coarse/fine face (the pressure matrix stays two-point), gated per face
  so the advecting face field stays a conservative flux; the advected value is reconstructed
  from the upwind side with level-aware probes, which cuts the one-step seam truncation from
  7.9e-3 to 2.0e-3 at no measurable cost.
- **Python** (`python/amr_bindings.cpp` → `peclet.amr`): `Octree`, `DistributedOctree`, `Poisson`,
  `Flow` (+ `Flow.diagnostics` for the developer instruments and ablation switches — the
  pressure-multigrid ladder and bottom solver, the seam reconstruction, the C/F census),
  `predict_hierarchy`, `spacing_from_extent` / `spacings_from_extent`. Type stub
  `packaging/_amr.pyi`.

## Validation

| what | against | result | where |
|---|---|---|---|
| uniform grid, `lmax = 0` | `peclet.flow`'s collocated solver, cell by cell | same discretization to 4.4e-11 | [amr_flow_uniform_parity.md](https://github.com/computational-chemical-engineering/peclet-amr/blob/main/docs/amr_flow_uniform_parity.md), ctest `python_flow_parity` |
| graded mesh, steady | plane Poiseuille (exact) | order 2.00 across a 2:1 interface | [amr_graded_convergence.md](https://github.com/computational-chemical-engineering/peclet-amr/blob/main/docs/amr_graded_convergence.md) |
| graded mesh, unsteady, advection on | decaying Taylor–Green vortex (exact) | order 2.22 (32 → 64); face field divergence ≤ 4e-12 | [amr_tg_graded.md](https://github.com/computational-chemical-engineering/peclet-amr/blob/main/docs/amr_tg_graded.md), ctest `python_amr_tg_graded` |
| distributed | the single-rank solver | np = 1 bitwise; np = 2/4/8 in the ~3e-7 march class | `tests/test_amr_distributed_*_mpi.cpp` |
| every public entry path | its own recorded state | SHA-256 of the final arrays, identical | `python/state_hash.py`, ctest `python_state_hash` |

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
124 in all — 111 C++ ctests (31 single-rank + 20 distributed tests at np = 1, 2, 4, 8), 5
`bench`-labelled C++ studies, and 8 Python ctests (`python_amr`, `python_amr_np2`, the
seam-record tests at np = 1, 2, 4, `python_state_hash` — the byte gate of `python/state_hash.py`,
whose reference hashes are `python/state_hash_reference.json`, recorded on one toolchain and
SKIPPED on any other — the `flow` parity gate, and the graded Taylor–Green gate). A test that
cannot run in a configuration exits 77 and ctest reports it **skipped**, never passed. `CLAUDE.md`
carries the developer notes, the environment-variable table and the gotchas.

## Documentation

Start at [docs/ROADMAP.md](https://github.com/computational-chemical-engineering/peclet-amr/blob/main/docs/ROADMAP.md), the one page of live items. [docs/README.md](https://github.com/computational-chemical-engineering/peclet-amr/blob/main/docs/README.md)
indexes the rest: the reference notes that describe the design as it ships (the collocated
projection, the mixed-level cut band, the parallel builders, anisotropic cells, the pressure
multigrid below the root brick, the 2:1 seam), the validation notes, the design analyses that were
decided against building, the briefs each design pass answered, and the dated campaign records in
`docs/archive/`. The measurement logs are in `docs/data/`, the drivers in `tests/study/`. Doxygen
API pages (`docs/Doxyfile`) are published to GitHub Pages by `.github/workflows/docs.yml`.

## Status

Under active development. The octree, the distributed octree with rebalancing, the collocated
projection with the ghost / sampled cut bands, the quadratic C/F schemes, the seam reconstruction,
the deep pressure multigrid and the distributed step are complete, tested and validated (above).
Open, per the roadmap: a published validation page; physical boundary conditions (the box is
triply periodic and driven by a body force — a wall is an immersed solid); at-scale multi-GPU runs;
the sibling-merge and repartition multigrid stages and a velocity multigrid below the root brick;
the mild growth of the pressure iteration count with resolution (traced to the transfer pair); and
the production `rho`/`N` for the sampled band's least-squares clouds, sub-face closures and pocket
exclusion.
