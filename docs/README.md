# peclet-amr — documentation index

Start at [ROADMAP.md](ROADMAP.md): the one page of live items, with a pointer to the note behind
each. This page is the map of everything else in `docs/`. The authority on how the package behaves
is the code; these notes say *why* it behaves that way, and what was measured.

## Reference notes — the design as it ships

| note | what it answers |
|---|---|
| [amr_collocated_projection.md](amr_collocated_projection.md) | The collocated Almgren–Bell–Colella projection on the octree: solid-pressure handling (`maskSolid`), the divergence-free face field `uf`, and `uf` as the advecting velocity. |
| [amr_mixed_level_cut_band_plan.md](amr_mixed_level_cut_band_plan.md) | The mixed-level cut band (`Flow.set_ghost_sampled`): cut cells at several octree levels, least-squares virtual samples across 2:1 boundaries, graded surface refinement (`refine_to_sdf_graded`, `refine_to_gap_floor`), and the distributed probe set. |
| [amr_setup_parallel_plan.md](amr_setup_parallel_plan.md) | The multithreaded `setSolid` builders (D1′) and the distributed C/F quadratic scheme (§7). |
| [amr_anisotropic.md](amr_anisotropic.md) | Per-axis root spacing: box-shaped cells through the octree, the cut band and the sampled builders. |
| [amr_mg_depth.md](amr_mg_depth.md) | **The pressure multigrid below the root brick** (ROADMAP C1): a lifted root, the lockstep distributed ladder, the replicated-tail stage, the agglomerated `GraphAMG`-PCG bottom, `predict_hierarchy` and the `Flow.diagnostics` read-outs. §11 carries the measured answers to its open questions (§11.4: the bottom extent stays 4; §11.10: how to read an iteration count). |
| [amr_cf_convective.md](amr_cf_convective.md) | **The convective flux at a 2:1 seam** (ROADMAP B5): the upwind-side seam reconstruction with level-aware probes, `Flow.diagnostics.set_seam_reconstruction`, and why what remains is intrinsic (§12, as built and measured). |
| [amr_cf_flux_gate.md](amr_cf_flux_gate.md) | The C/F face-value delta gated per FACE through one emitter (ROADMAP A7), so `uf` is a conservative flux at every 2:1 sub-face; `Flow.diagnostics.num_cf_cut_faces`. |

## Validation — what was measured against an exact answer or another code

| note | what it establishes |
|---|---|
| [amr_flow_uniform_parity.md](amr_flow_uniform_parity.md) | At `lmax = 0` `peclet.amr.Flow` and `peclet.flow.SolverColocated` are the same discretization, cell by cell (gate ctest `python_flow_parity`). |
| [amr_graded_convergence.md](amr_graded_convergence.md) | The graded steady solver is second order across a 2:1 interface (plane Poiseuille), and the quadratic C/F scheme is what keeps it so on a tangential interface (ROADMAP B1, A5, A6). |
| [amr_tg_graded.md](amr_tg_graded.md) | The graded time-accurate benchmark: a decaying Taylor–Green vortex with advection on, second order (2.22 on the 32 → 64 rung since the B5 seam reconstruction, 2.07 before it) on a graded octree (gate ctest `python_amr_tg_graded`, ROADMAP B2). |

## Design analyses that were decided against building (yet)

| note | verdict |
|---|---|
| [amr_pressure_iteration.md](amr_pressure_iteration.md) | Inner pressure iterations (A) declined; the deferred-corrected C/F pressure gradient (B) designed and parked (§14), because the leak it would remove measures 4e-3 of the face error and is O(dt²) (ROADMAP B4). |
| [amr_mg_core_boundary.md](amr_mg_core_boundary.md) | Layering decision: coarse-level redistribution (telescoping, repartition, replication) lives in `peclet::core::decomp`; hierarchies stay in the methods. §11 specifies and measures the repartition kind and the aligned weighted ORB (S2). |

## Briefs — the questions the design notes answer

The `briefs/` directory keeps the brief each design pass was given, so the note can be read against
the question it was asked. They are inputs, not descriptions of the code.

| brief | answered by |
|---|---|
| [briefs/cf_quadratic_distributed.md](briefs/cf_quadratic_distributed.md) | [amr_setup_parallel_plan.md](amr_setup_parallel_plan.md) §7 (the distributed C/F quadratic scheme) |
| [briefs/cf_flux_gate_design.md](briefs/cf_flux_gate_design.md) | [amr_cf_flux_gate.md](amr_cf_flux_gate.md) |
| [briefs/pressure_iteration_options.md](briefs/pressure_iteration_options.md) | [amr_pressure_iteration.md](amr_pressure_iteration.md) |
| [briefs/tg_graded_gate_w.md](briefs/tg_graded_gate_w.md) | [amr_pressure_iteration.md](amr_pressure_iteration.md) §14 |
| [briefs/cf_convective_flux.md](briefs/cf_convective_flux.md) | [amr_cf_convective.md](amr_cf_convective.md) |
| [briefs/mg_depth_below_root.md](briefs/mg_depth_below_root.md) | [amr_mg_depth.md](amr_mg_depth.md) |

## History and data

- [archive/](archive/README.md) — dated campaign records (the distributed flow, the march economics,
  the aperture-advection stall, the Snellius bed recipe), each with its own status line. Not
  maintained; read the source before acting on one.
- `data/` — the measurement logs and JSON the notes cite; `../tests/study/` — the drivers that
  produced them.
- The API reference is the Doxygen build of `docs/Doxyfile` (the headers under
  `include/peclet/amr/` and the binding TU `python/amr_bindings.cpp`); the Python docstrings are
  the same text as `packaging/_amr.pyi`.
