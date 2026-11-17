# amr playbook (overrides the cce-code-health seed by id)

Read the seed first (`plugins/cce-code-health/skills/refactor-loop/playbook-seed.md`), then dem's playbook
(`suite/dem-health-api/.codehealth/playbook.md`, the recipes) and this file. Read `CLAUDE.md` (settled decisions)
before touching any solver header; a refactor never reverses one. The collocated pressure coupling (approximate
projection, never Rhie-Chow) and the in-predictor force placement are guarded by ctest `amr_stability_guard`.
Gate: `PECLET_SUITE=<suite> codehealth gate`; bitwise = `python/state_hash.py` on the host/OpenMP build at ONE thread.

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
