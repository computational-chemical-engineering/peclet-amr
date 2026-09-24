"""
peclet.amr — adaptive mesh refinement: per-block Octree (serial) and DistributedOctree (MPI ORB) for the mesh, the geometric-multigrid Poisson solver, and the device (Kokkos) Flow cut-cell Stokes/Navier-Stokes solver. Build a graded octree, refine to an SDF surface, read leaf geometry + per-leaf fields as numpy, load-rebalance, gather face neighbours, export VTU, and run the flow step on device.
"""

from collections.abc import Callable, Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def finalize() -> None:
    """
    Release every live object and zero-copy array of this module, then Kokkos::finalize() (deterministic teardown; also run automatically at interpreter exit). Idempotent. After it, the module's solver objects and *_view arrays must not be used.
    """

execution_space: str = 'OpenMP'

build_toolchain: str = 'GNU 14.2.0 Release x86_64 Kokkos 5.1.1'

def predict_hierarchy(cells: Sequence[int], lmax: int = 0, num_ranks: int = 1, bottom_extent: int = 4) -> dict:
    """
    The pressure-multigrid ladder a Flow builds on `cells` FINEST cells per axis at tree depth `lmax` over `num_ranks` ranks (docs/amr_mg_depth.md §6.2, §6.7) — a pre-flight tool. A pure function: it builds no mesh and needs no MPI; it re-runs the ladder rule on the ORB a freshly constructed DistributedOctree would produce (not a partition after `DistributedOctree.rebalance` / `Flow.rebalance_mpi`). Below the root brick the ladder halves the grid while every rank's block stays even; where it stops above `bottom_extent` cells per axis on more than one rank the coarsest level is gathered onto every rank (the replicated tail) and continued there; the coarsest level is then solved by damped-Jacobi sweeps if it has at most `bottom_extent` cells per axis, else by the agglomerated GraphAMG-PCG bottom. The prediction is that of the default `Flow.set_pressure_bottom('auto')`.

    Returns a dict, levels finest first: `extent` the level's global grid in CELLS per axis (a count, not a length), `cells` its total cell count as if the mesh were uniform (so it equals `Flow.diagnostics.pressure_mg_levels` leaf for leaf only for a uniform mesh at num_ranks=1), `kind` ('octree' at or above the root brick, 'lifted' below it in place, 'tail' on the gathered grid -- the first tail level is the SAME grid as the last in-place one, moved rather than coarsened, exactly as `pressure_mg_levels` lists it), `num_levels`, `num_in_place` (levels that keep the ORB), `tail` (bool) and `bottom` ('jacobi' or 'amg', suffixed '+tail' when the tail engages) — the spelling `Flow.diagnostics.pressure_mg_bottom` reports.

    `lmax` is the number of octree coarsenings THE MESH supports, i.e. the tree's lmax for a mesh refined to level 0 somewhere. An UNREFINED Octree(cells, lmax=k>0) is the same mesh as Octree(cells/2**k, lmax=0) — all its leaves are root cells — and must be predicted that way. In general pass depth = tree.lmax - levels().min() and cells = root * 2**depth. `bottom_extent` (default 4, the solver's default) is the `Flow.pressure_bottom_extent` the run will use; pass the same value.
    """

def spacing_from_extent(extent: Sequence[float], root_cells: Sequence[int], lmax: int) -> float:
    """
    The finest cell width h0 = extent / (root_cells * 2**lmax) of a PHYSICAL domain — the one place an AMR spacing is computed, so no caller writes one (see suite/docs/PHYSICAL_UNITS_PLAN.md). Returns ONE number, so it raises when the extent does not give cubic cells — use `spacings_from_extent` for a box mesh.
    """

def spacings_from_extent(extent: Sequence[float], root_cells: Sequence[int], lmax: int) -> list[float]:
    """
    The finest cell size (dx, dy, dz) = extent / (root_cells * 2**lmax), PER AXIS. The anisotropic form of `spacing_from_extent`: it accepts any positive extent, because the octree's cells are boxes (core/docs/amr_anisotropic.md).
    """

class Octree:
    """
    Serial single-block adaptive octree with a world placement (origin + finest spacing per axis). Leaves are addressed in Z-order slot order; every per-leaf array is indexed by that slot.
    """

    def __init__(self, cells: Sequence[int], *, lmax: int = 0, origin: Sequence[float] = [0.0, 0.0, 0.0], spacing: float | Sequence[float] | None = None, extent: Sequence[float] | None = None) -> None:
        """
        Build a uniform octree resolving `cells` FINEST cells per axis, refinable `lmax` levels (the root brick is cells/2**lmax root cells, which must divide exactly; lmax=0 is a uniform grid). `origin` is the block's lower corner. Every argument after `cells` is keyword-only.

        State the domain PHYSICALLY with `extent` = the box side lengths: the cell size is then extent/cells PER AXIS and the caller never writes a spacing. Alternatively give `spacing` (one number for cubic cells, or (dx, dy, dz)); default 1. Passing both raises.

        The cells are BOXES (core/docs/amr_anisotropic.md): any positive extent is accepted, the octree refines by 2 on every axis, and every level inherits the root aspect ratio. Read the three numbers back from `.spacing`.
        """

    def is_balanced(self) -> bool:
        """
        True iff every face-adjacent leaf pair differs by at most one level (2:1).
        """

    def find(self, x: Sequence[float]) -> int:
        """
        Index of the leaf containing world point x=(x,y,z), or -1 if outside the block.
        """

    def refine_to_sdf_graded(self, sdf: Callable[[float, float, float], float], target_level: Callable[[float, float, float], int], band: float = 2.0, balance: bool = True) -> int:
        """
        GRADED surface refinement (the mixed-level cut band, docs/amr_mixed_level_cut_band_plan.md §7): `target_level` is a callable f(x,y,z)->level giving the COARSEST acceptable level at a world point (0 = finest), so cut cells end up at SEVERAL levels — fine in throats/contacts, coarse on smooth caps. The band margin is measured in cells of the level being created (not in the finest spacing as in refine_to_sdf). Requires Flow.set_ghost_sampled(True) — the classic overlay contracts a uniform finest band and raises on these level jumps. Returns refinements performed.
        """

    def refine_to_gap_floor(self, sdf: Callable[[float, float, float], float], gap: Callable[[float, float, float], float], coarsest_level: int, n: float = 4.0, band: float = 2.0, balance: bool = True) -> int:
        """
        refine_to_sdf_graded driven by the plan's GAP-WIDTH FLOOR (§7 criterion 1): the target level at a point is the coarsest L with n*h_L <= gap(x), clamped to [0, coarsest_level]. `gap` is the local fluid-gap proxy f(x,y,z)->width — for a sphere packing the two-closest-surfaces sum d1+d2; a medial-axis or peclet.pnm throat-radius field substitutes verbatim. n=4 per the M1/M2 measurements. This is the AMReX multi-valued-cell rule inverted: coarsening never merges or disconnects fluid.
        """

    def refine_leaf(self, i: int) -> bool:
        """
        Split leaf `i` into its 8 children; returns True if it was split (level>0).
        """

    @property
    def cells(self) -> list[int]:
        """
        Finest-level cell counts per axis (root*2**lmax; the GLOBAL grid on a DistributedOctree).
        """

    @property
    def extent(self) -> list[float]:
        """Box side lengths in world units (cells*spacing)."""

    @property
    def spacing(self) -> list[float]:
        """Finest cell size (dx, dy, dz), per axis. Equal on a cubic octree."""

    @property
    def num_leaves(self) -> int:
        """Number of leaves (Z-order slots; this rank's on a DistributedOctree)."""

    @property
    def lmax(self) -> int:
        """Root-cell level (max refinement depth)."""

    @property
    def origin(self) -> list[float]:
        """Lower corner in world coordinates."""

    def centers(self) -> NDArray[numpy.float64]:
        """Leaf world centres, (num_leaves, 3) float64 (global coordinates)."""

    def sizes(self, axis: int = 0) -> NDArray[numpy.float64]:
        """
        Leaf world widths along `axis`: spacing[axis]*2**level, (num_leaves,) float64. A leaf is a BOX, so `axis` selects which of the three widths (0 by default, which is THE width on a cubic octree).
        """

    def levels(self) -> NDArray[numpy.int32]:
        """Leaf refinement levels, (num_leaves,) int32 (0 = finest)."""

    def codes(self) -> NDArray[numpy.uint64]:
        """Leaf block-local Morton origin codes, (num_leaves,) uint64."""

    def refine_to_sphere(self, center: Sequence[float], radius: float, target_level: int = 0, band: float = 1.0, balance: bool = True) -> int:
        """
        Refine leaves the sphere surface passes through (plus `band` cells) down to target_level; optionally restore 2:1 balance (cross-block, collective, on a DistributedOctree). Returns the (local) number of refinements performed.
        """

    def refine_to_sdf(self, sdf: Callable[[float, float, float], float], target_level: int = 0, band: float = 1.0, balance: bool = True) -> int:
        """
        Refine toward an arbitrary signed-distance field given as a callable f(x,y,z)->distance (suite sign: <0 inside solid), down to target_level — rings / packed beds / any non-sphere geometry. Collective when balance=True on a DistributedOctree. Returns refinements performed.
        """

    def balance(self) -> int:
        """
        Enforce 2:1 graded balance to a fixpoint (cross-block and collective on a DistributedOctree); returns (this rank's) refinements performed.
        """

    def lohner_indicator(self, field: Annotated[NDArray[numpy.float64], dict(order='C')], eps: float = 0.01) -> NDArray[numpy.float64]:
        """
        Löhner normalized-second-difference feature indicator E in [0,1] per leaf from a scalar field (num_leaves,); large E = steep feature (refine), small = smooth (coarsen). On a DistributedOctree it is evaluated across the owner-based halo (collective).
        """

    def adapt(self, field: Annotated[NDArray[numpy.float64], dict(order='C')], refine_thresh: float, coarsen_thresh: float, finest_level: int = 0, eps: float = 0.01, linear: bool = True) -> NDArray[numpy.float64]:
        """
        Solution-adaptive step (Löhner-driven): refine where the indicator > refine_thresh (to finest_level), coarsen sibling groups all < coarsen_thresh, 2:1-balance, and conservatively remap `field`. MUTATES the octree in place; returns the remapped field (M,). `linear` uses minmod-limited prolongation (else piecewise-constant). On a DistributedOctree: per block, cross-block balance, ORB ownership kept, bit-identical across rank counts (collective).
        """

    def write_vtu(self, path: str, name: str, field: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Write the octree (this rank's block on a DistributedOctree — one file per rank, combine in ParaView) + a per-leaf scalar field (num_leaves,) as a VTK UnstructuredGrid (.vtu, ASCII, one cell per leaf).
        """

class Poisson:
    """
    Cell-centered finite-volume Poisson solver (L u = rhs) on an Octree, by a geometric-multigrid V-cycle. L is the conservative two-point FV Laplacian (suite sign). The hierarchy snapshots the octree at construction; per-leaf arrays are (num_leaves,) float64 in Z-order slots.
    """

    def __init__(self, octree: Octree, periodic: bool = True) -> None:
        """
        Build the multigrid hierarchy from `octree`. periodic=True solves the singular periodic problem (constant null space removed each cycle).
        """

    @property
    def num_leaves(self) -> int:
        """Leaves on the finest level."""

    @property
    def num_levels(self) -> int:
        """
        Number of multigrid levels: the octree's own coarsenings down to the root brick, then levels BELOW it (the root lifted, docs/amr_mg_depth.md §6.1-§6.2) for as long as some axis has more than 4 cells and every axis stays even with at least 2 cells after halving. So a uniform lmax=0 mesh has a real hierarchy (64^3: 5 levels).
        """

    def apply(self, u: Annotated[NDArray[numpy.float64], dict(order='C')]) -> NDArray[numpy.float64]:
        """
        L applied to u (the FV Laplacian); use b = apply(u_exact) to manufacture a RHS. (num_leaves,) -> (num_leaves,).
        """

    def residual(self, u: Annotated[NDArray[numpy.float64], dict(order='C')], rhs: Annotated[NDArray[numpy.float64], dict(order='C')]) -> float:
        """Volume-weighted L2 residual norm sqrt(sum V*(rhs - L u)^2)."""

    def solve(self, rhs: Annotated[NDArray[numpy.float64], dict(order='C')], x0: Annotated[NDArray[numpy.float64], dict(order='C')] | None = None, cycles: int = 20, pre: int = 2, post: int = 2, tol: float = 0.0) -> tuple:
        """
        Solve L u = rhs with up to `cycles` V-cycles (pre/post Gauss-Seidel sweeps), from x0 or 0, stopping once residual <= tol (tol<=0 disables). Returns (u (num_leaves,), final_residual, cycles_done).
        """

class Flow:
    """
    Collocated incompressible Stokes/Navier-Stokes step on an Octree with a cut-cell immersed boundary (no-slip on an SDF solid). step() = implicit viscous momentum predictor + Almgren-Bell-Colella rotational projection, whose pressure Poisson equation is solved by multigrid-preconditioned CG (the hierarchy continues below the octree's root brick; `diagnostics.pressure_mg_levels`). The box is TRIPLY PERIODIC and the only driving is a uniform body force (set_body_force); a wall is an immersed solid. Any consistent unit system: density [mass/length^3], dynamic viscosity [mass/(length*time)], dt [time], lengths those of the Octree's extent. Iterate step() to a steady state or march in time; velocities and pressure are per-leaf (num_leaves,) arrays in Z-order slots.
    """

    @overload
    def __init__(self, octree: Octree, density: float = 1.0, viscosity: float = 1.0, dt: float = 1000000.0) -> None:
        """
        Create a flow on `octree` with the given density, dynamic viscosity and time step (defaults 1, 1 and 1e6). A large dt drives straight to the steady (Stokes) solution. The octree is borrowed by reference and must outlive the Flow's use of it.
        """

    @overload
    def __init__(self, octree: DistributedOctree, density: float = 1.0, viscosity: float = 1.0, dt: float = 1000000.0) -> None:
        """
        DISTRIBUTED (mpi4py, collective): run this solver on one ORB block of a DistributedOctree — the whole step (momentum, pressure, overlays, adaptivity) then executes multi-rank through the ±2 leaf ghost registry. Per-leaf arrays are this rank's LOCAL leaves. setSolid/step/project/begin_adapt/finish_adapt/rebalance_mpi are collective. np=1 is bit-identical to the single-rank constructor.
        """

    def rebalance_mpi(self, sdf: Callable[[float, float, float], float]) -> None:
        """
        DISTRIBUTED: weighted-ORB load rebalance — migrates the leaves WITH the state (u, p) to the new owners and rebuilds every solver structure (collective). num_leaves is refreshed.
        """

    @property
    def num_leaves(self) -> int:
        """Number of leaves."""

    def set_solid(self, sdf: Callable[[float, float, float], float]) -> None:
        """
        Build the cut-cell operators from a signed-distance callable f(x,y,z) (>0 fluid, <0 solid) and zero the fields. Call before stepping; re-call to change the geometry.
        """

    def set_solid_spheres(self, centers: Annotated[NDArray[numpy.float64], dict(order='C', writable=False)], radii: Annotated[NDArray[numpy.float64], dict(order='C', writable=False)], periodic: bool = True) -> None:
        """
        set_solid for a UNION OF SPHERES, evaluated natively instead of through a Python callback — the porous-media geometry. `centers` is (M,3), `radii` is (M,) or (1,); `periodic` uses the minimum-image convention over the octree's own box extent. Prefer this to set_solid at bed scale: set_solid samples the SDF tens of times per leaf (operator build, overlay classification at virtual positions, the openness probe), so a Python callback dominates everything else — measured >1h43m of pure numpy on an 11.35M-leaf 180-sphere bed before the GPU ran a single kernel.
        """

    def set_body_force(self, fx: float, fy: float, fz: float) -> None:
        """
        Set the body force per unit volume [force/length^3] driving the flow, e.g. a mean pressure gradient -dp/dx along x. It is the only driving the periodic box has. Default (0, 0, 0).
        """

    def set_advection(self, on: bool) -> None:
        """Enable explicit momentum advection (Navier-Stokes); off = Stokes."""

    def set_ghost_projection(self, on: bool, matrix_order: int = 2, rhs_order: int = 2) -> None:
        """
        DEFAULT since 2026-08-25 (AUTO: ghost, with an aperture fallback + stderr notice when the finest band is too thin): the fluid-only constraint scheme — family-free, unconditionally stable, protocol-independent (flow's attractor-campaign verdicts; == flow's set_collocated_scheme('ghost')). FULL directional ghost-cell projection (the AMR port): binary-openness pressure operator + wall-anchored closure overlay on the finest-band rows, MG-preconditioned BiCGStab, ghost-closed divergence constraint; implies set_ghost_gradient. (matrix_order, rhs_order) closure orders: (2, 2) default and the only pair cleared for production — the (1, 2) mixed form is march-UNSTABLE above ~2000 spheres (flow hardening Phase A), kept callable for parity records only. Raises if the finest band is too thin (a closure would cross a 2:1 boundary). Call before set_solid.
        """

    def set_ghost_sampled(self, on: bool, rho: float = 2.2, max_samples: int = 0) -> None:
        """
        MIXED-LEVEL CUT BAND (docs/amr_mixed_level_cut_band_plan.md): allow cut cells at MULTIPLE octree levels — the finest-band contract is dropped. Chain entries that cross a 2:1 boundary become degree-2 LS virtual samples at the uniform closure positions (identity weights at same level, so a uniform finest band is BIT-IDENTICAL to set_ghost_sampled(False)); face classification uses the level-aware canonical openness; the momentum xi-row seam correction and the wall-aware C/F tangential fallback ride along. Implies the ghost projection (engages when the resolved scheme is ghost — the AUTO default or an explicit set_ghost_projection(True)). Distributed since 2026-08-30 (the clouds are a deterministic probe set through the leaf halo). `rho` is the least-squares cloud radius factor (rho = factor * max(h, H); 2.2 = the shipped behaviour) and `max_samples` the nearest-N candidate cap (0 = uncapped) — the M2a cloud-economy knobs, inert at their defaults; do not change them in production without the M2a table. Call before set_solid.
        """

    def set_pressure(self, values: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Write the accumulated rotational pressure from a (num_leaves,) array — restart, or re-accumulation policies after finish_adapt (at steady-state dt the transferred p is the load-bearing state; zeroing it after a coarsening adapt lets it re-accumulate cleanly).
        """

    def set_dt(self, dt: float) -> None:
        """
        Change the time step. dt is BAKED INTO the momentum operator at build time (idiag = rho/dt), so a set_dt must be followed by set_solid or the operator stays stale. set_solid reallocates and zeroes u and p, so a dt SWITCH mid-march is: read velocity()/pressure() -> set_dt -> set_solid -> set_velocity()/set_pressure(). That sequence is the dt-cycling protocol of the attractor-family batteries.
        """

    def begin_adapt(self) -> None:
        """
        Snapshot the octree topology + (u, p) ahead of an external mesh mutation (adapt / refine_to_sphere / refine_to_sdf / balance on the SAME Octree object). Pair with finish_adapt.
        """

    def finish_adapt(self, sdf: Callable[[float, float, float], float]) -> None:
        """
        Rebuild the solver on the mutated octree and conservatively transfer the snapshotted u and p onto it (minmod-limited linear remap). `sdf` is re-sampled on the new leaves (pass the same geometry callable as set_solid); keep the cut band at the finest level on the new mesh (re-run refine_to_sdf on the geometry band after a solution-driven adapt). The advecting face field restarts from the cell average for one step.
        """

    def set_cf_scheme(self, scheme: int) -> None:
        """
        Coarse/fine (2:1) interface scheme. 1 = Martin-Cartwright tangential quadratic, THE DEFAULT since 2026-09-21: second order, applied to the momentum diffusion, the divergence constraint and the pressure gradients (the pressure matrix/MG stays standard, which does not move the steady solution). 0 = the standard two-point flux, first-order at level boundaries, kept only to reproduce pre-2026-09-21 graded results. The default is INERT BY GEOMETRY on a uniform or finest-band mesh -- no C/F faces, no delta -- and matters where a 2:1 interface sees tangential variation, which on a self-similar graded ladder is the difference between order 1.60 and 0.41. Works with both the aperture and the ghost projection. The seam reconstruction of the advected value (diagnostics.set_seam_reconstruction) rides on the quadratic scheme: with 0 no seam tables are built. Call before set_solid.
        """

    def set_advection_scheme(self, scheme: int) -> None:
        """
        High-order advection flux: 0 = second-order upwind (default), 1 = Koren TVD.
        """

    def set_implicit_advection(self, on: bool) -> None:
        """
        Implicit first-order-upwind deferred-correction advection (default on): unconditionally stable. Off = fully explicit high-order advection.
        """

    def set_outer_iterations(self, n: int, tol: float = 1e-06) -> None:
        """
        Picard outer iterations over the lagged advection per step (default 1).
        """

    def set_pressure_tolerance(self, rtol: float) -> None:
        """
        Relative tolerance of the PRESSURE solve, for both drivers -- the MG-PCG default and the ghost projection's BiCGStab (flow shares its set_pressure_pcg tolerance with its ghost BiCGStab the same way). Default 1e-10, the value that was hard-coded until 2026-09-21, so leaving it alone reproduces every earlier result. The iteration CAP is step()'s `pres_iters`; this is the accuracy it works to, and loosening it is the cheapest cost knob in the step.
        """

    def set_momentum_tolerance(self, rtol: float) -> None:
        """
        Relative tolerance of the per-component MOMENTUM solve (BiCGStab, MG-preconditioned by default). Default 1e-8. At the large dt used for steady drag the momentum operator degrades toward a bare elliptic Laplacian and the solve gets expensive; this bounds the over-solve. The cap is step()'s `mom_iters`. (flow spells the same concept set_velocity_residual_tolerance -- the momentum/velocity divergence between the two codes is a ../docs/NAMING.md item, not settled here.)
        """

    def set_pressure_bottom(self, mode: str) -> None:
        """
        What solves the coarsest level of the pressure multigrid (docs/amr_mg_depth.md §6.6). A V-cycle converges at a mesh-independent rate only if its coarsest level is effectively solved, and 60 damped-Jacobi sweeps solve a level only up to ~4 cells per axis (8e-9 at 4, 8e-3 at 8, 0.6 at 25). 'auto' (THE DEFAULT) engages the agglomerated GraphAMG-PCG bottom iff the coarsest level still has more than `pressure_bottom_extent` cells on some axis -- i.e. where the ladder ran out on an odd or badly factored grid -- and keeps the sweeps otherwise; 'smoother' always sweeps; 'agglomerated' always solves exactly. The choice changes the preconditioner, not the converged pressure (to the solve tolerance), and at the default extent it does not change the iteration count either (§11.4(a): identical counts in all nine measured 'smoother'/'agglomerated' pairs); leave it at 'auto' unless you are comparing bottom solves. The same three strings and default as flow's `Solver.set_pressure_bottom`. Takes effect at once on the hierarchy already built and is kept for later set_solid calls. On a DISTRIBUTED Flow call it on every rank: the exact bottom lives in the replicated stage's continued ladder, which the call may build. Raises on any other string.
        """

    def set_pressure_bottom_extent(self, cells: int) -> None:
        """
        Where the pressure ladder stops coarsening below the root brick and hands over to the bottom solve, in CELLS PER AXIS of the coarsest level -- a count, not a length: the limit is the bottom smoother's, which solves a level only up to ~4 cells per axis whatever the physical domain (../docs/NAMING.md §1.8; docs/amr_mg_depth.md §6.2/§11.4). The ladder stops once no axis has more than `cells` cells, and set_pressure_bottom('auto') engages the exact bottom where it stopped above it. Default 4, the value at which the 60 bottom sweeps are exact; it is measured rather than preferred (§11.4: 8 moves the iteration count by ~1 on one case in three; tests/study/amr_pressure_depth.py --sweep), so there is rarely a reason to change it. Changes the preconditioner, not the converged pressure (to the solve tolerance). Takes effect at the NEXT set_solid, where the ladder is built. Raises if cells < 1.
        """

    @property
    def pressure_bottom_extent(self) -> int:
        """
        The bottom extent, in cells per axis (a count, not a length), that set_pressure_bottom_extent last stored (default 4) -- the one the next set_solid builds the ladder with (docs/amr_mg_depth.md §6.8).
        """

    def step(self, mom_iters: int = 100, pres_iters: int = 60) -> None:
        """
        Advance one collocated projection step of length dt on device. `mom_iters` caps the momentum solve (BiCGStab, MG-preconditioned) and `pres_iters` the pressure solve (MG-PCG, or BiCGStab under the ghost projection); each stops earlier at its set_momentum_tolerance / set_pressure_tolerance. Collective on a distributed Flow.
        """

    def velocity(self, component: int) -> NDArray[numpy.float64]:
        """Per-leaf velocity component (0=x,1=y,2=z), (num_leaves,) float64."""

    def velocities(self) -> NDArray[numpy.float64]:
        """All three velocity components, (num_leaves, 3) float64."""

    def is_fluid(self) -> NDArray[numpy.bool_]:
        """Per-leaf fluid mask (False in the solid), (num_leaves,) bool."""

    def divergence_norm(self) -> float:
        """
        Volume-weighted L2 norm of the residual cell divergence (projection-quality diagnostic).
        """

    def pressure(self) -> NDArray[numpy.float64]:
        """Per-leaf pressure (incremental-rotational p), (num_leaves,) float64."""

    def face_field(self) -> NDArray[numpy.float64]:
        """
        ABC divergence-free FACE velocity, one value per CSR (sub)face (conservative flux / streamline post-processing).
        """

    def set_velocity(self, component: int, values: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Write velocity component c (0=x,1=y,2=z) from a (num_leaves,) array — initial conditions, restart, or warm-start. Call before step()/project().
        """

    def project(self, pres_iters: int = 60) -> None:
        """
        Pressure projection only (no momentum solve) — project an externally-set velocity field to divergence-free. Returns nothing; read the result via velocity()/velocities().
        """

    @property
    def diagnostics(self) -> FlowDiagnostics:
        """
        The developer tier (FlowDiagnostics, a view onto this Flow): iteration counts of the last step, the face-field divergence and topology, the C/F and seam-reconstruction censuses, the pressure-multigrid ladder and its bottom solver, and the solver-internals / ablation switches. Nothing here is needed to set up, run or read out a simulation; everything here has a production default.
        """

class FlowDiagnostics:
    """
    Developer instruments and ablation switches of a Flow, reached as `flow.diagnostics` (suite/docs/QUALITY_PLAN.md D2: the public Flow surface is what a user needs to set up, run and read out a simulation; this is what a developer uses to inspect or ablate it).
    """

    def last_mom_iters(self) -> int:
        """
        Total momentum BiCGStab iterations (summed over the 3 components) of the last step.
        """

    def last_pres_iters(self) -> int:
        """Pressure PCG iterations of the last step."""

    def last_outer_iters(self) -> int:
        """
        Picard outer iterations actually run in the last step (1 unless set_outer_iterations(>1)).
        """

    def divergence_norm_face(self) -> float:
        """
        L2 norm of the APERTURE-weighted divergence of the ABC face field. MEANINGFUL ON THE APERTURE PATH ONLY (set_ghost_projection(False)), where it is the pressure-solve residual, far below divergence_norm — including across 2:1 interfaces. Under the GHOST projection (the DEFAULT) the solved constraint is this divergence PLUS an overlay delta that is a functional of the CELL velocities, not of the face field, so this norm omits it and reads O(1) on a perfectly healthy solve (34 at N=32, 184 at N=64 on the Z&H sphere, with the velocities matching peclet.flow to 1e-6). There, divergence_norm() is the residual you want. Unnormalized either way: it grows with resolution and velocity magnitude, so read it as a trend, never as an absolute.
        """

    @property
    def num_cf_cut_faces(self) -> int:
        """
        How many 2:1 C/F sub-face slots of THIS RANK carry the standard two-point face value because the quadratic one is withheld: both incident cells must be REGULAR fluid (fluid and not cut) for the quadratic C/F scheme (set_cf_scheme(1), the default) to apply there, so this counts the sub-faces where a level boundary meets the wall (docs/amr_cf_flux_gate.md §6.5). 0 on every uniform or finest-band mesh; on a graded mesh it measures the size of the set that runs at the standard scheme's local order. Each sub-face contributes two slots (one per incident cell) and the sum over ranks equals the single-rank count.
        """

    def face_topology(self) -> dict:
        """
        The face CSR topology face_field() is indexed by, as a dict of arrays: 'start' (num_leaves+1 row offsets, int64), 'nbr' (neighbour leaf per (sub)face, int64), 'axis' (0/1/2, int32), 'dir' (+1/-1 from the owning cell toward the neighbour, int32), 'raw_area' (world area the ADVECTIVE flux uses -- the FINE area at a 2:1 sub-face, so a coarse face's four sub-faces sum to the coarse area), 'dist' (world centre distance, 1.5*h_fine at a 2:1 sub-face), 'alpha' (face openness in [0, 1]) and 'upup_i' / 'upup_j' (the second upwind probes the SOU/Koren reconstruction samples, -1 where none). A 2:1 sub-face is a slot whose two incident leaves have different Octree.levels(); its centroid is the FINER leaf's face centre. Host-copied on every call -- a diagnostic, not a step-loop read-out. Under MPI a neighbour index >= num_leaves is a ghost slot of this rank's registry, whose world centre (like every slot's) is row `slot` of 'cell_center', an (num_leaves + num_ghost_cells, 3) array. The SEAM RECONSTRUCTION tables of docs/amr_cf_convective.md come with it: 'seam' (one descriptor id per face slot, -1 = a plain slot that takes the ordinary SOU/Koren line), the per-descriptor 'samp_i' / 'samp_j' (the tangential-sample record when i / j is the COARSE cell of a 2:1 sub-face, else -1), 'uu_rec_i' / 'uu_rec_j' (the upstream probe's record when the second upwind cell of i / j crosses a level, else -1) and 'd1_i' / 'd1_j' (world half width along the face axis), and the record CSR 'rec_start', 'rec_cell', 'rec_w', 'rec_dist' (the world probe distance, used only where a record is an UPSTREAM probe). All empty with set_cf_scheme(0 = standard), which builds no tables. Rank-local under MPI.
        """

    @property
    def pressure_mg_levels(self) -> list[int]:
        """
        Leaf count of every pressure-multigrid level, finest first (docs/amr_mg_depth.md §6.7), as a list of int; empty before set_solid, which is where the hierarchy is built. Levels below the root brick are LIFTED levels — the same octree with its root halved — so a uniform mesh has a real hierarchy rather than a single level. Under MPI the in-place levels are THIS RANK's counts and the levels of a replicated tail (appended last) are GLOBAL counts, identical on every rank. Check it against `peclet.amr.predict_hierarchy` (level count, and leaf for leaf on a uniform mesh at np=1), never against a literal.
        """

    @property
    def pressure_mg_bottom(self) -> str:
        """
        What solves the coarsest pressure level of the hierarchy AS BUILT: 'jacobi' (60 damped-Jacobi sweeps, effectively exact at <= 4 cells per axis: the slowest mode falls by 8e-9) or 'amg' (the agglomerated GraphAMG-PCG solve), with the suffix '+tail' when the coarsest level was gathered onto every rank by the replicated stage (docs/amr_mg_depth.md §6.5-§6.7). Which one runs follows `Flow.set_pressure_bottom` (default 'auto': the exact bottom engages only where the ladder ran out above `Flow.pressure_bottom_extent` cells per axis). 'jacobi' before set_solid.
        """

    @property
    def num_seam_sample_records(self) -> int:
        """
        How many tangential-sample records the seam reconstruction built on THIS RANK -- one per 2:1 sub-face pair that passes the C/F face gate (both cells regular fluid). 0 on a uniform mesh and with the standard C/F scheme.
        """

    @property
    def num_seam_layer_records(self) -> int:
        """
        How many face-layer records the seam reconstruction built on THIS RANK -- one per (coarse cell, face) whose far side is refined and whose four fine cells are fluid (the case-3 upstream probe).
        """

    def set_momentum_mg(self, on: bool) -> None:
        """
        Use the Galerkin velocity multigrid as the momentum solve preconditioner (default on; makes the momentum solve scale with resolution). Call before set_solid.
        """

    def set_momentum_gs(self, on: bool) -> None:
        """
        Use the symmetric multicolour Gauss-Seidel smoother in the momentum multigrid (default off = weighted Jacobi). Call before set_solid.
        """

    def set_velocity_mg_staircase(self, on: bool) -> None:
        """
        Use the rediscretised staircase velocity-MG instead of Galerkin (default off).
        """

    def set_momentum_mg_solver(self, on: bool) -> None:
        """
        Solve the momentum predictor with the velocity-MG as the solver (no Krylov), mirroring flow's velocity solve (default off = BiCgStab with the MG as preconditioner).
        """

    def set_ghost_gradient(self, on: bool) -> None:
        """
        Directional ghost cell-gradient on cut cells for the pressure predictor and the projection's cell correction (2nd-order one-sided, never reads decoupled solid pressure — removes the gauge-dependent O(1/h) cut-cell gradient error of the plain ABC gradient). The ghost projection (the default) implies it; this switch matters for the aperture fallback only. Call before set_solid.
        """

    def set_aperture_order(self, order: int) -> None:
        """
        Aperture estimator for the (fallback) aperture projection: 2 = analytic marching-squares (DEFAULT since 2026-08-26), 1 = legacy one-sample model. Call before set_solid.
        """

    def set_uf_advection(self, on: bool) -> None:
        """
        ABLATION. The advecting velocity of the momentum advection: the projected, divergence-free face field uf (ON, the shipped Almgren-Bell-Colella scheme) or, with on=False, the un-projected 1/2(u_i+u_j) cell->face average that the first step uses before any projection has run. It is the ONE discretization difference between this solver and peclet.flow's SolverColocated on a uniform grid; turning it off makes the two agree to solver tolerance (docs/amr_flow_uniform_parity.md).
        """

    def set_seam_reconstruction(self, on: bool) -> None:
        """
        ABLATION. Reconstruct the ADVECTED value with level-aware probes at every face whose upwind-side stencil crosses a 2:1 coarse/fine seam (default ON, docs/amr_cf_convective.md): the coarse upwind cell is tangentially sampled at the sub-face's own column, and an upstream probe at another octree level is taken at its TRUE distance (a coarser one as the same tangential sample, a finer one as the mean of the four face-layer children). Without it the sub-face value carries the coarse column's tangential offset -- an O(h) face error, so an O(1) local truncation on the fine side of the seam. on=False restores that, bit for bit, and may be flipped between steps; the tables are built in set_solid either way. Inert with advection off, with set_cf_scheme(0 = standard), and on any mesh with no 2:1 face. Leave it on in production: the switch exists for the A/B of docs/amr_cf_convective.md §12, where it cuts the one-step seam truncation 7.9e-3 -> 2.0e-3 at no measurable cost.
        """

class DistributedOctree:
    """
    MPI octree: an ORB block decomposition of a global root grid (one BlockOctree per rank, over MPI_COMM_WORLD). Construct it collectively; refine/balance/rebalance/face_neighbor_gather are collective. Per-leaf arrays describe THIS rank's local block in global world coordinates.
    """

    def __init__(self, cells: Sequence[int], *, lmax: int = 0, origin: Sequence[float] = [0.0, 0.0, 0.0], spacing: float | Sequence[float] | None = None, periodic: Sequence[bool] = [True, True, True], extent: Sequence[float] | None = None) -> None:
        """
        Decompose a global grid of `cells` FINEST cells per axis (root cells = cells/2**lmax, each `lmax` levels deep) across the ranks of MPI_COMM_WORLD via ORB. `origin` places the global grid; `periodic` per axis. Every argument after `cells` is keyword-only.

        State the domain PHYSICALLY with `extent` = the global box side lengths; the cell size is then extent/cells PER AXIS. Alternatively give `spacing` (one number or (dx, dy, dz)); default 1. Passing both raises. Read the three numbers back from `.spacing`.
        """

    @property
    def rank(self) -> int:
        """This process's MPI rank."""

    @property
    def size(self) -> int:
        """Number of ranks (blocks)."""

    @property
    def block_origin_root(self) -> list[int]:
        """This rank's block lower corner, in global root-cell coordinates."""

    @property
    def block_brick(self) -> list[int]:
        """This rank's block size in root cells per axis."""

    @property
    def global_root_size(self) -> list[int]:
        """Global grid size in root cells per axis."""

    def rebalance(self, fields: Annotated[NDArray[numpy.float64], dict(order='C')]) -> NDArray[numpy.float64]:
        """
        Re-decompose by leaf count (weighted ORB) and migrate leaves + their fields. `fields` is (num_leaves, K) float64; returns this rank's (M, K) columns after migration. Pure redistribution; the partition is updated in place (collective).
        """

    def face_neighbor_gather(self, field: Annotated[NDArray[numpy.float64], dict(order='C')], sentinel: float = 0.0) -> NDArray[numpy.float64]:
        """
        For each local leaf, the field value across each of its 6 faces, gathered over the owner-based halo. `field` is (num_leaves,); returns (num_leaves, 6) laid out [+x,-x,+y,-y,+z,-z]; domain boundaries carry `sentinel` (collective).
        """

    @property
    def cells(self) -> list[int]:
        """
        Finest-level cell counts per axis (root*2**lmax; the GLOBAL grid on a DistributedOctree).
        """

    @property
    def extent(self) -> list[float]:
        """Box side lengths in world units (cells*spacing)."""

    @property
    def spacing(self) -> list[float]:
        """Finest cell size (dx, dy, dz), per axis. Equal on a cubic octree."""

    @property
    def num_leaves(self) -> int:
        """Number of leaves (Z-order slots; this rank's on a DistributedOctree)."""

    @property
    def lmax(self) -> int:
        """Root-cell level (max refinement depth)."""

    @property
    def origin(self) -> list[float]:
        """Lower corner in world coordinates."""

    def centers(self) -> NDArray[numpy.float64]:
        """Leaf world centres, (num_leaves, 3) float64 (global coordinates)."""

    def sizes(self, axis: int = 0) -> NDArray[numpy.float64]:
        """
        Leaf world widths along `axis`: spacing[axis]*2**level, (num_leaves,) float64. A leaf is a BOX, so `axis` selects which of the three widths (0 by default, which is THE width on a cubic octree).
        """

    def levels(self) -> NDArray[numpy.int32]:
        """Leaf refinement levels, (num_leaves,) int32 (0 = finest)."""

    def codes(self) -> NDArray[numpy.uint64]:
        """Leaf block-local Morton origin codes, (num_leaves,) uint64."""

    def refine_to_sphere(self, center: Sequence[float], radius: float, target_level: int = 0, band: float = 1.0, balance: bool = True) -> int:
        """
        Refine leaves the sphere surface passes through (plus `band` cells) down to target_level; optionally restore 2:1 balance (cross-block, collective, on a DistributedOctree). Returns the (local) number of refinements performed.
        """

    def refine_to_sdf(self, sdf: Callable[[float, float, float], float], target_level: int = 0, band: float = 1.0, balance: bool = True) -> int:
        """
        Refine toward an arbitrary signed-distance field given as a callable f(x,y,z)->distance (suite sign: <0 inside solid), down to target_level — rings / packed beds / any non-sphere geometry. Collective when balance=True on a DistributedOctree. Returns refinements performed.
        """

    def balance(self) -> int:
        """
        Enforce 2:1 graded balance to a fixpoint (cross-block and collective on a DistributedOctree); returns (this rank's) refinements performed.
        """

    def lohner_indicator(self, field: Annotated[NDArray[numpy.float64], dict(order='C')], eps: float = 0.01) -> NDArray[numpy.float64]:
        """
        Löhner normalized-second-difference feature indicator E in [0,1] per leaf from a scalar field (num_leaves,); large E = steep feature (refine), small = smooth (coarsen). On a DistributedOctree it is evaluated across the owner-based halo (collective).
        """

    def adapt(self, field: Annotated[NDArray[numpy.float64], dict(order='C')], refine_thresh: float, coarsen_thresh: float, finest_level: int = 0, eps: float = 0.01, linear: bool = True) -> NDArray[numpy.float64]:
        """
        Solution-adaptive step (Löhner-driven): refine where the indicator > refine_thresh (to finest_level), coarsen sibling groups all < coarsen_thresh, 2:1-balance, and conservatively remap `field`. MUTATES the octree in place; returns the remapped field (M,). `linear` uses minmod-limited prolongation (else piecewise-constant). On a DistributedOctree: per block, cross-block balance, ORB ownership kept, bit-identical across rank counts (collective).
        """

    def write_vtu(self, path: str, name: str, field: Annotated[NDArray[numpy.float64], dict(order='C')]) -> None:
        """
        Write the octree (this rank's block on a DistributedOctree — one file per rank, combine in ParaView) + a per-leaf scalar field (num_leaves,) as a VTK UnstructuredGrid (.vtu, ASCII, one cell per leaf).
        """
