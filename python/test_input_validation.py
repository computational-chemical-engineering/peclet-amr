"""Contract: the public surface refuses a NaN/Inf or inadmissible argument with ValueError.

Validation lives in the binding (python/amr_bindings_checks.hpp, used from python/amr_bindings.cpp);
the message names the callable, the parameter, the admissible range with its dimension and the
offending value. Each validated argument is covered twice: every bad value raises ValueError naming
the parameter, and one valid value is accepted.

Where a sign or a 0 is a switch the check is "finite" only, and the valid value below is that
switch: Poisson.solve tol <= 0 disables the early stop; adapt's thresholds outside the indicator's
[0, 1] mean never refine / never coarsen; a refinement `band` may be negative (a thinner band).

Run: PYTHONPATH=<build> python -m pytest -q python/test_input_validation.py (ctest
python_input_validation). The DistributedOctree cases run on a one-rank MPI_COMM_WORLD.
"""
import math
import re

import numpy as np
import pytest
from mpi4py import MPI  # noqa: F401  (MPI_Init before the distributed classes are built)

from peclet import amr

NAN, INF = math.nan, math.inf
N = 8  # finest cells per axis; lmax = 1, so the root brick is 4^3 = 64 leaves
NL = 64


def _oct():
    return amr.Octree([N, N, N], lmax=1, extent=[1.0, 1.0, 1.0])


def _dist():
    return amr.DistributedOctree([N, N, N], lmax=1, extent=[1.0, 1.0, 1.0])


def _flow():
    return amr.Flow(_oct(), density=1.0, viscosity=0.1, dt=1.0)


def _ready():
    """A Flow with its operators built (all fluid, Stokes): what step/project/set_velocity need."""
    f = _flow()
    f.set_advection(False)
    f.set_solid(lambda x, y, z: 1.0)
    return f


def _plane(x, y, z):
    return x - 0.37


_Z = np.zeros(NL)

# (case id, call(v), bad values, a valid value, the parameter the message must name)
CASES = [
    # module functions
    ("predict_hierarchy.lmax", lambda v: amr.predict_hierarchy([N, N, N], lmax=v), [22, 64], 1,
     "lmax"),
    ("spacing_from_extent.extent", lambda v: amr.spacing_from_extent([v, v, v], [4, 4, 4], 0),
     [NAN, INF, 0.0, -1.0], 2.0, "extent[0]"),
    ("spacing_from_extent.lmax", lambda v: amr.spacing_from_extent([1.0] * 3, [4, 4, 4], v),
     [22, 64], 2, "lmax"),
    ("spacings_from_extent.extent",
     lambda v: amr.spacings_from_extent([1.0, 1.0, v], [4, 4, 4], 0), [INF, -1.0], 0.5,
     "extent[2]"),
    ("spacings_from_extent.lmax", lambda v: amr.spacings_from_extent([1.0] * 3, [4, 4, 4], v),
     [22], 21, "lmax"),
    # Octree constructor
    ("Octree.lmax", lambda v: amr.Octree([N, N, N], lmax=v), [22, 40], 3, "lmax"),
    ("Octree.cells", lambda v: amr.Octree([v, N, N]), [2**21 + 2, 2**40], 16, "cells[0]"),
    ("Octree.origin", lambda v: amr.Octree([N, N, N], origin=[0.0, v, 0.0]), [NAN, INF], -1.0,
     "origin[1]"),
    ("Octree.spacing", lambda v: amr.Octree([N, N, N], spacing=v), [NAN, INF, 0.0, -1.0], 0.5,
     "spacing"),
    ("Octree.spacing[1]", lambda v: amr.Octree([N, N, N], spacing=[1.0, v, 1.0]), [NAN, 0.0],
     2.0, "spacing[1]"),
    ("Octree.extent", lambda v: amr.Octree([N, N, N], extent=[1.0, 1.0, v]),
     [NAN, INF, 0.0, -1.0], 3.0, "extent[2]"),
    # Octree methods
    ("find.x", lambda v: _oct().find([v, 0.5, 0.5]), [NAN, INF], 0.25, "x[0]"),
    ("refine_leaf.i", lambda v: _oct().refine_leaf(v), [-1, NL], NL - 1, "i"),
    ("refine_to_sphere.center", lambda v: _oct().refine_to_sphere([0.5, v, 0.5], 0.2),
     [NAN, INF], 0.5, "center[1]"),
    ("refine_to_sphere.radius", lambda v: _oct().refine_to_sphere([0.5] * 3, v),
     [NAN, INF, 0.0, -1.0], 0.2, "radius"),
    ("refine_to_sphere.target_level",
     lambda v: _oct().refine_to_sphere([0.5] * 3, 0.2, target_level=v), [2, 30], 1,
     "target_level"),
    ("refine_to_sphere.band", lambda v: _oct().refine_to_sphere([0.5] * 3, 0.2, band=v),
     [NAN, INF], -0.5, "band"),
    ("refine_to_sdf.target_level", lambda v: _oct().refine_to_sdf(_plane, target_level=v), [2],
     0, "target_level"),
    ("refine_to_sdf.band", lambda v: _oct().refine_to_sdf(_plane, band=v), [NAN, -INF], 2.0,
     "band"),
    ("refine_to_sdf_graded.band",
     lambda v: _oct().refine_to_sdf_graded(_plane, lambda x, y, z: 0, band=v), [NAN, INF], 1.0,
     "band"),
    ("refine_to_gap_floor.coarsest_level",
     lambda v: _oct().refine_to_gap_floor(_plane, lambda x, y, z: 0.1, v), [2, 9], 1,
     "coarsest_level"),
    ("refine_to_gap_floor.n",
     lambda v: _oct().refine_to_gap_floor(_plane, lambda x, y, z: 0.1, 1, n=v),
     [NAN, INF, 0.0, -1.0], 2.0, "n"),
    ("refine_to_gap_floor.band",
     lambda v: _oct().refine_to_gap_floor(_plane, lambda x, y, z: 0.1, 1, band=v), [NAN], 1.0,
     "band"),
    ("lohner_indicator.eps", lambda v: _oct().lohner_indicator(_Z, eps=v), [NAN, INF, -1.0], 0.0,
     "eps"),
    ("adapt.refine_thresh", lambda v: _oct().adapt(_Z, v, 0.1), [NAN, INF], 2.0,
     "refine_thresh"),
    ("adapt.coarsen_thresh", lambda v: _oct().adapt(_Z, 0.5, v), [NAN, -INF], -1.0,
     "coarsen_thresh"),
    ("adapt.finest_level", lambda v: _oct().adapt(_Z, 0.5, 0.1, finest_level=v), [2], 1,
     "finest_level"),
    ("adapt.eps", lambda v: _oct().adapt(_Z, 0.5, 0.1, eps=v), [NAN, -0.1], 0.0, "eps"),
    # Poisson
    ("Poisson.solve.cycles", lambda v: amr.Poisson(_oct()).solve(_Z, cycles=v), [-1], 0,
     "cycles"),
    ("Poisson.solve.pre", lambda v: amr.Poisson(_oct()).solve(_Z, cycles=1, pre=v), [-1], 0,
     "pre"),
    ("Poisson.solve.post", lambda v: amr.Poisson(_oct()).solve(_Z, cycles=1, post=v), [-2], 1,
     "post"),
    ("Poisson.solve.tol", lambda v: amr.Poisson(_oct()).solve(_Z, cycles=1, tol=v), [NAN, INF],
     -1.0, "tol"),
    # Flow constructor and setters
    ("Flow.density", lambda v: amr.Flow(_oct(), density=v), [NAN, INF, 0.0, -1.0], 1000.0,
     "density"),
    ("Flow.viscosity", lambda v: amr.Flow(_oct(), viscosity=v), [NAN, INF, -1.0], 0.0,
     "viscosity"),
    ("Flow.dt", lambda v: amr.Flow(_oct(), dt=v), [NAN, INF, 0.0, -1.0], 1e-3, "dt"),
    ("set_body_force", lambda v: _flow().set_body_force(0.0, v, 0.0), [NAN, INF, -INF], -9.81,
     "f[1]"),
    ("set_ghost_projection.matrix_order", lambda v: _flow().set_ghost_projection(True, v),
     [0, 3], 1, "matrix_order"),
    ("set_ghost_projection.rhs_order", lambda v: _flow().set_ghost_projection(True, 2, v),
     [-1, 3], 2, "rhs_order"),
    ("set_ghost_sampled.rho", lambda v: _flow().set_ghost_sampled(True, v),
     [NAN, INF, 0.0, -1.0], 3.0, "rho"),
    ("set_ghost_sampled.max_samples", lambda v: _flow().set_ghost_sampled(True, 2.2, v), [-1],
     0, "max_samples"),
    ("set_dt", lambda v: _flow().set_dt(v), [NAN, INF, 0.0, -1.0], 0.5, "dt"),
    ("set_cf_scheme", lambda v: _flow().set_cf_scheme(v), [-1, 2], 0, "scheme"),
    ("set_advection_scheme", lambda v: _flow().set_advection_scheme(v), [-1, 2], 1, "scheme"),
    ("set_outer_iterations.n", lambda v: _flow().set_outer_iterations(v), [-1, 0], 2, "n"),
    ("set_outer_iterations.tol", lambda v: _flow().set_outer_iterations(2, v), [NAN, -1.0], 0.0,
     "tol"),
    ("set_pressure_tolerance", lambda v: _flow().set_pressure_tolerance(v), [NAN, INF, -1.0], 0.0,
     "rtol"),
    ("set_momentum_tolerance", lambda v: _flow().set_momentum_tolerance(v), [NAN, -1e-8], 1e-6,
     "rtol"),
    ("step.mom_iters", lambda v: _ready().step(v, 1), [-1], 1, "mom_iters"),
    ("step.pres_iters", lambda v: _ready().step(1, v), [-1], 1, "pres_iters"),
    ("project.pres_iters", lambda v: _ready().project(v), [-1], 1, "pres_iters"),
    ("set_velocity.component", lambda v: _ready().set_velocity(v, _Z), [-1, 3], 2, "component"),
    ("set_velocity.values", lambda v: _ready().set_velocity(0, np.zeros(v)), [3, NL + 1], NL,
     "values length"),
    ("set_pressure.values", lambda v: _ready().set_pressure(np.zeros(v)), [0, NL - 1], NL,
     "values length"),
    ("set_solid_spheres.centers",
     lambda v: _flow().set_solid_spheres(np.array([[0.5, v, 0.5]]), np.array([0.2])), [NAN, INF],
     0.5, r"centers.flat[1]"),
    ("set_solid_spheres.radii",
     lambda v: _flow().set_solid_spheres(np.array([[0.5, 0.5, 0.5]] * 2), np.array([0.2, v])),
     [NAN, INF, 0.0, -0.2], 0.1, "radii[1]"),
    # DistributedOctree and the distributed Flow (one rank)
    ("DistributedOctree.lmax", lambda v: amr.DistributedOctree([N, N, N], lmax=v), [22], 2,
     "lmax"),
    ("DistributedOctree.origin", lambda v: amr.DistributedOctree([N, N, N], origin=[v, 0.0, 0.0]),
     [NAN, INF], 1.0, "origin[0]"),
    ("DistributedOctree.spacing", lambda v: amr.DistributedOctree([N, N, N], spacing=v),
     [NAN, 0.0, -1.0], 0.5, "spacing"),
    ("DistributedOctree.extent", lambda v: amr.DistributedOctree([N, N, N], extent=[v, 1.0, 1.0]),
     [INF, -1.0], 2.0, "extent[0]"),
    ("DistributedOctree.refine_to_sphere.radius",
     lambda v: _dist().refine_to_sphere([0.5] * 3, v), [NAN, -1.0], 0.2, "radius"),
    ("DistributedOctree.adapt.finest_level",
     lambda v: _dist().adapt(np.zeros(NL), 0.5, 0.1, finest_level=v), [2], 0, "finest_level"),
    ("DistributedOctree.rebalance.ndim", lambda v: _dist().rebalance(np.zeros(v)),
     [(NL,), (NL, 1, 1)], (NL, 2), "fields.ndim"),
    ("DistributedOctree.rebalance.columns", lambda v: _dist().rebalance(np.zeros((NL, v))), [0],
     1, r"fields.shape[1]"),
    ("Flow(DistributedOctree).dt", lambda v: amr.Flow(_dist(), dt=v), [NAN, -1.0], 2.0, "dt"),
]

_BAD = [(c[0], c[1], b, c[4]) for c in CASES for b in c[2]]
_GOOD = [(c[0], c[1], c[3]) for c in CASES]


@pytest.mark.parametrize("case,call,value,param", _BAD,
                         ids=[f"{c[0]}={c[2]!r}" for c in _BAD])
def test_bad_value_raises_value_error_naming_the_parameter(case, call, value, param):
    with pytest.raises(ValueError) as err:
        call(value)
    msg = str(err.value)
    assert re.search(rf"{re.escape(param)} must be ", msg), msg
    assert "got" in msg, msg


@pytest.mark.parametrize("case,call,value", _GOOD, ids=[c[0] for c in _GOOD])
def test_valid_value_is_accepted(case, call, value):
    call(value)


def test_message_states_range_unit_and_value():
    f = _flow()
    with pytest.raises(ValueError, match=r"^set_dt: dt must be finite and > 0 \(time\), got -1$"):
        f.set_dt(-1.0)
    with pytest.raises(ValueError, match=r"^set_body_force: f\[2\] must be finite "
                                         r"\(force/length\^3\), got nan$"):
        f.set_body_force(0.0, 0.0, NAN)
    with pytest.raises(ValueError, match=r"^Flow: density must be finite and > 0 "
                                         r"\(mass/length\^3\), got -1$"):
        amr.Flow(_oct(), density=-1.0)
    with pytest.raises(ValueError, match=r"^set_cf_scheme: scheme must be an integer in \[0, 1\], "
                                         r"got 2$"):
        f.set_cf_scheme(2)


def test_state_before_set_solid_is_a_runtime_error():
    """set_pressure / set_velocity before set_solid wrote into unallocated state; now they raise
    the same RuntimeError as step() and project() there."""
    f = _flow()
    for call in (lambda: f.set_pressure(_Z), lambda: f.set_velocity(0, _Z)):
        with pytest.raises(RuntimeError, match="call setSolid\\(\\) first"):
            call()


def test_existing_checks_are_unchanged():
    """The checks that predate this file keep their RuntimeError and message."""
    with pytest.raises(RuntimeError, match=r"2\*\*lmax"):
        amr.Octree([N, N, N + 1], lmax=1)
    with pytest.raises(RuntimeError, match="EITHER"):
        amr.Octree([N, N, N], spacing=1.0, extent=[1.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="axis must be 0, 1 or 2"):
        _oct().sizes(3)
    with pytest.raises(RuntimeError, match="must be >= 1"):
        _flow().set_pressure_bottom_extent(0)
    with pytest.raises(RuntimeError, match="field length != num_leaves"):
        _oct().lohner_indicator(np.zeros(3))
