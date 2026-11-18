# amr playbook (overrides the cce-code-health seed by id)

Read the seed first (`plugins/cce-code-health/skills/refactor-loop/playbook-seed.md`), then dem's playbook
(`suite/dem-health-api/.codehealth/playbook.md`, the recipes) and this file. Read `CLAUDE.md` (settled decisions)
before touching any solver header; a refactor never reverses one. The collocated pressure coupling (approximate
projection, never Rhie-Chow) and the in-predictor force placement are guarded by ctest `amr_stability_guard`.
Gate: `PECLET_SUITE=<suite> codehealth gate --mode refactor --base <prev> --ref-base <campaign base>`; bitwise =
`tests/regression/state_hash.py` + `python/state_hash.py` on the host/OpenMP build at ONE thread.

## Base gate (WO-31, amr origin/main 9205762, core 7cec32a, morton d264165; TRS2)
- Build: host-openmp prefix, `-DPECLET_AMR_BUILD_TESTS=ON -DPECLET_AMR_BUILD_PYTHON=ON`, Release, no extra flags (the
  committed `state_hash_reference.json` is that toolchain: GNU 14.2.0 Release x86_64 Kokkos 5.1.1). 0 warnings.
- ctest: 129 registered = 101 main (`-LE 'bench|np8'`, minus python_flow_parity) + 21 np8 + 6 bench + 1 parity.
  Base: 101/101 and 21/21 pass, 0 skips. python_flow_parity exits 77 without `-DPECLET_AMR_FLOW_PYTHONPATH` (a flow
  build tree): excluded by name, not configured, so the gate never reads another session's build tree.
- state_hash: `--check python/state_hash_reference.json` IDENTICAL at np 1; np 2 recorded (keys `.np2`).

## Traps
- Threads: battery at `OMP_NUM_THREADS=2 OMP_PROC_BIND=false` (CLAUDE.md), np8 subset LAST at one thread; state_hash
  at one thread only. `MPIEXEC_PREFLAGS` is the LIST form `--bind-to;none`; `MPIEXEC_EXECUTABLE=/usr/bin/mpirun`.
- Siblings: a worktree `suite/amr-<topic>` finds `../core`, `../morton` silently; the gate's base copy (no siblings)
  would FETCH them at the pinned tags. The config pins both with `FETCHCONTENT_SOURCE_DIR_PECLET_SIB_PECLET-{CORE,MORTON}`
  and compares the resolved include paths (`[deps]` lines from compile_commands.json) between base and head.
- `state_hash.py --save` takes a FILE (dem's takes a directory); a toolchain other than the reference's exits 77.
- quality.yml formats `include/ tests/ benchmarks/ python/*.cpp` only: a `python/*.hpp` is not in CI's glob; the
  gate's clang_format tier adds it.
- Distributed probes need `from mpi4py import MPI` before `peclet.amr` (the module never calls MPI_Init).

## Wall clock (TRS2, load ~10-30)
Base build cold 144 s; ctest main 261 s (-j8, OMP 2), np8 13 s (-j1, OMP 1); state_hash np1 + np2 4 s;
codehealth scan < 2 s.

## User-facing round (WO-31, 2026-11-17; deltas from dem's and flow's recipe)
- **Where checks live**: `python/amr_bindings_checks.hpp` (binding-only). Most setters are bound as
  `&chk::set_x<Flow>` (free templates on the TU's wrapper classes); the Flow constructors as
  `.def("__init__", &chk::init_flow<Flow, const Octree>, ...)` with keep_alive<1, 2>, the same signature as nb::init.
  The four methods with 6-7 parameters (refine_to_sphere, adapt, refine_to_gap_floor, Poisson.solve) call a
  `chk::*_args` line in their member body instead: a forwarding wrapper would be a second long-parameter-list
  function (diff 4.0 -> 11.0); this way amr_bindings.cpp stays at 4.0 and the header at 0.
- **`codehealth diff` needs the config at the base**: against origin/main (no .codehealth/) it runs the default
  python-only profile. Diff against the step-0 commit that adds the config.
- **Defaults that look silent but are documented**: `Octree(lmax=0)` (a uniform grid) and
  `set_ghost_sampled(max_samples=0)` (uncapped). The audit counts both; they stay.
- **Found beyond the audit** (probe every int and array, `from mpi4py import MPI` first for the distributed
  class): Octree took 2**22 cells per axis (Morton overflow; the block limit is 2**21) and hung on 2**40;
  `refine_leaf(-1)`, `set_velocity(3, ..)`, a short `set_pressure`/`set_velocity` array and both before
  `set_solid` wrote out of bounds; `rebalance` of a 1-D array read a shape it lacks and of (n, 0) died of SIGFPE;
  ghost orders other than 1/2 and cf/advection schemes other than 0/1 were silently remapped.
- **State errors stay RuntimeError** (`requireOperator`, as step() raises); value errors are ValueError. Checks that
  predate the round keep their RuntimeError (cells divisibility, spacing+extent, sizes axis, bottom extent); a NaN or
  negative `extent` now raises ValueError (it raised RuntimeError without the value).
- **pytest in CI**: amr's CI did not install it; `python_input_validation` / `python_stub_sync` need it (ci.yml).
- **Doc references**: the unresolved list must not quote its targets in backticks (the checker scans .codehealth/).
- **`codehealth gate --mode refactor` cannot judge this round**: docs, CMake, CI and .codehealth/ are outside its
  scope (FAIL scope/forbid/frozen before any tier runs). The tiers were run blind with the config's own commands.
- **Wall clock (TRS2)**: incremental module build 47 s; ctest main 103 tests 313 s, np8 14 s; state_hash 7 s;
  targeted pytest 0.5 s (208 cases); api audit 4 s; codehealth docs / diff < 2 s.

## Host-orchestration round (WO-37, 2026-11-18; RULES.md "Host-orchestration round")
- Recipes: dem's playbook (`suite/dem-health/.codehealth/playbook.md`): extract-kernel-phase, launch-params-struct,
  host-state-owner; anti-patterns "debt by lambda shuffling", "a phase helper with a parameter list as long as the
  phase". Kernel bodies (`KOKKOS_LAMBDA`, functor `operator()`, `KOKKOS_*FUNCTION`) move only whole, token-identical.
- Config: profiles `cpp@1` + `python@0`, zones of the campaign's `tools/zones-cpp1.toml`; forbid `tests/*`,
  `packaging/*`, the bindings and their check header; frozen `tests/regression/*`, cmake, CLAUDE.md, the public hash.
- Bitwise reference `tests/regression/state_hash.py --build <build>`: one process per case at OMP 1. Single-rank:
  `sampled_graded` (two-level cut band, sampled overlay), `ghost_adapt` (classic ghost, begin/finish_adapt),
  `aperture_cf0` (aperture projection, standard C/F, staircase velocity MG + GS). `mpirun -np 1, 2, 4`:
  `dist_sampled` (band coarsened on one side by `adapt`, so LS2 slots exist at np 2), `dist_adapt` (adapt +
  finish_adapt + rebalance_mpi), `fv` (`regression_fv_dump`: `DistributedFvOperator`, which peclet.amr does not
  expose, plain and openness-weighted, apply + 20 Jacobi sweeps, hex-float dump sorted by global code). Iteration
  counts are their own keys. Runs in ~32 s; byte-identical twice.
- Sibling pin: the `[deps]` lines carry the core/morton commit, so a shared checkout moving mid-round fails
  `same_lines` rather than comparing two programs.
- Timing case `tests/regression/setup_bench.py` (`set_solid_spheres` + 20 steps, 96^3 graded, sampled, advection;
  OMP 8): set_solid ~2 s, steps ~38 s. A Python SDF callback makes set_solid ~85 s (the GIL-bound callback, not
  the builders): time with the C++ sphere SDF.
