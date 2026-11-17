# WO-31: broken documentation references left unedited (rule 5)

`codehealth docs --config <campaign>/tools/api_audit_docs.toml` at amr 9205762: 92 broken in living docs; after the
fix commit: 12, exactly the list below. Targets are written without code quotes so that this file adds no
reference of its own. Rules: brief WO-30, "Rule per broken reference".

| file:line | target | why not fixed |
|---|---|---|
| CLAUDE.md:44 | centered.h | Basilisk's source file, outside the suite; no path to point at. |
| docs/amr_anisotropic.md:194 | packaging/core_amr.pyi | core-era stub (amr 206616c), deleted; packaging/_amr.pyi was added new, not a git rename, and the heading names core-era paths throughout. |
| docs/amr_cf_flux_gate.md:32 | tests/study_amr_uf_div.cpp | lived on branch ufgate only; never on any ref in this repo (git log --all empty). |
| docs/briefs/cf_flux_gate_design.md:161 | tests/study_amr_uf_div.cpp | same as above. |
| docs/amr_flow_uniform_parity.md:202 | embed.h | Basilisk's source file, outside the suite. |
| docs/amr_pressure_iteration.md:417 | tests/study/amr_cf_pres_kappa.py | a planned WO4 driver ("or C++ under bench"); never committed (git log --all empty). |
| docs/amr_pressure_iteration.md:578 | tg_leak.py | scratch probe "in the session scratchpad" (the sentence says so); gone. |
| docs/amr_pressure_iteration.md:579 | tg_advvar.py | same. |
| docs/amr_pressure_iteration.md:579 | tg_noadv.py | same. |
| docs/amr_pressure_iteration.md:652 | tg_leak.py | same. |
| docs/amr_setup_parallel_plan.md:88 | geom/ | two candidates: core's include/peclet/core/geom/ (the scene layer when the plan was written, in core) and the geom repo it later moved to. |
| docs/amr_setup_parallel_plan.md:204 | geom/ | same. |

## Notes on fixed references (not counted)
- ../../docs/decisions/amr.md:13 (docs/ROADMAP.md:344, docs/amr_flow_uniform_parity.md:116, 145, 211): the path is fixed
  (rule 2), but line 13 of the register is now the maxBlockCells entry; the ABC entry these sentences cite is the heading
  "Collocated pressure coupling is ABC (Almgren-Bell-Colella), never Rhie-Chow" (line 370 today).
- ../../docs/decisions/flow.md:977 (docs/amr_flow_uniform_parity.md:127, 177): path fixed; line 977 is now an
  sdf-scene entry, not the FOU-operator record the sentence cites.
- Line anchors into flow sources (mac_approx_projection.hpp:30, colocated_advection.hpp:29-36,
  flow_ibm_mpi.hpp:149) keep their line numbers: rule 2 fitted first, and the checker skips paths outside the repo.
