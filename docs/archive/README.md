# Archive — design notes and campaign records

These are **dated design notes, campaign plans and session handoffs** of the AMR work, kept for the
record (suite `docs/QUALITY_PLAN.md` D6: AMR is under active development and nothing of it is
deleted; D7: docs describe the code that exists, so history lives here). They came with the tree
when it was relocated out of `peclet-core` on 2026-09-10, git history included. Each one steered a
piece of work that has since landed, been superseded, or been folded into a reference document;
the authority on how `peclet-amr` behaves today is the code plus [README.md](../../README.md),
[CLAUDE.md](../../CLAUDE.md) and the four reference notes in the parent directory
([amr_collocated_projection](../amr_collocated_projection.md),
[amr_mixed_level_cut_band_plan](../amr_mixed_level_cut_band_plan.md),
[amr_setup_parallel_plan](../amr_setup_parallel_plan.md), [amr_anisotropic](../amr_anisotropic.md)).

Nothing here is maintained: `file:line` citations, status lines and "next step" sections are
snapshots of their date — re-read the source before acting on any of them. Paths inside them are
the ones of their day: `core/include/peclet/core/amr/<file>` means `include/peclet/amr/<file>` here,
`peclet::core::amr` means `peclet::amr`, `peclet.core.amr` means `peclet.amr`, `core/docs/<note>`
means `docs/<note>` or `docs/archive/<note>`, and the `PECLET_CORE_PROFILE_*` / `PECLET_CORE_GPS_*`
variables are the `PECLET_AMR_PROFILE_*` variables and the `set_ghost_sampled(rho=, max_samples=)`
arguments (CLAUDE.md, "Environment variables"). The measurement logs they cite are in
[../data/](../data/).

| note | date | what it is |
|---|---|---|
| [amr_aperture_advection_plan.md](amr_aperture_advection_plan.md) | 2026-08-18/19 | The AMR aperture pressure solve stalling under advection: the three candidate mechanisms, what was measured, and the **RESOLVED 2026-08-19** verdict (un-deflated RHS mean + a stale PCG gate; deflation wins over a compatible-RHS repair). |
| [amr_device_assembly_plan.md](amr_device_assembly_plan.md) | 2026-06/07 | Plan to move AMR operator/geometry *assembly* (cut stencils, openness, face CSR, FOU) from serial host code onto the device, with the assembly inventory table. Parked: `amr_setup_parallel_plan.md` took the multithreaded-host route instead. |
| [amr_distributed_flow.md](amr_distributed_flow.md) | 2026-07-26/28 | Design + rung-by-rung record of distributing `AmrFlow` (LeafHalo, the resolver seam, the distributed multigrid, `step`/adapt/rebalance). All rungs shipped; the shipped surface is described in `CLAUDE.md` §Architecture. |
| [amr_march_perf_and_distributed_plan.md](amr_march_perf_and_distributed_plan.md) | 2026-08-30 … 09-04 | The march-time economics + distributed mixed-level band campaign (M0–M2c, F2, D0–D3): the step profiler, the attribution matrix, the cloud-economy table and the distributed band. Both phases executed; M2b (pick the production `rho`/`N`) is the one open decision, and the two LS-cloud knobs stay inert at their defaults. |
| [comm_avoiding_pressure_driver.md](comm_avoiding_pressure_driver.md) | 2026-08-19 | Proposal for an all-reduce-free pressure driver (once-per-solve RHS projection + Chebyshev V-cycles), from the `dev/aperture-compat-rhs` experiment. **Never implemented** — kept so the option is on the table when multi-GPU pressure-solve scaling is the work item. |
| [snellius_amr_bed.md](snellius_amr_bed.md) | 2026-08 | The Snellius recipe for the graded packed-bed march (`tests/study/amr_bed_graded.py`): build, sbatch conventions, the d7/d8 logs in `../data/`. |
