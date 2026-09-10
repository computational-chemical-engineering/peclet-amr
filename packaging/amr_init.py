"""peclet.amr — the block-local-Morton AMR octree and the collocated-projection Navier–Stokes solver.

Relocated out of ``peclet.core.amr`` on 2026-09-10 (suite/docs/QUALITY_PLAN.md D6 / G.2): the
same classes, the same names, one package up. Everything is Kokkos (CUDA / HIP / OpenMP;
``execution_space`` says which) over MPI (``import mpi4py.MPI`` first):

- ``Octree(cells, *, lmax, origin, spacing | extent)`` — the serial single-block octree: refine to a
  sphere / SDF / graded target, 2:1 balance, Löhner-driven ``adapt``, leaf geometry as numpy, VTU.
- ``DistributedOctree(cells, *, lmax, origin, spacing | extent, periodic)`` — one ORB block per
  rank over ``MPI_COMM_WORLD``: the same refinement surface, cross-block balance, weighted-ORB
  ``rebalance``, ``face_neighbor_gather``, distributed ``adapt``.
- ``Poisson(octree, periodic)`` — the geometric-multigrid FV Poisson solver on an octree.
- ``Flow(octree | distributed_octree, density, viscosity, dt)`` — the device cut-cell IBM
  collocated Stokes / Navier–Stokes step (ghost projection by default; the mixed-level sampled cut
  band via ``set_ghost_sampled``). Developer instruments live under ``Flow.diagnostics``.
- ``spacing_from_extent`` / ``spacings_from_extent`` — the one place an AMR spacing is computed.
- ``finalize()`` releases the Kokkos state (also registered at exit).

Conventions: per-leaf arrays are in the octree's Z-order slot order, length ``num_leaves``; the
domain quartet ``origin`` / ``extent`` / ``cells`` / ``spacing`` follows suite/docs/NAMING.md §1.1
(``cells`` is the FINEST grid; the root brick is ``cells / 2**lmax``); SDF sign is negative inside
the solid.
"""

from ._amr import *  # noqa: F401,F403

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown".
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("peclet-amr")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"
