"""Gate P: Z&H graded permeability vs a uniform-fine reference, on the byte gate's own graded
sphere (state_hash.py's `sph`: r=0.22 at (0.5,0.5,0.5) in the unit box, cells=32^3, lmax=1,
refine_to_sdf(band=3.0) -- a FINEST-BAND mesh, so docs/amr_cf_flux_gate.md sec.7 says the number
must be bitwise identical across the per-face gate).

  PYTHONPATH=<build> OMP_NUM_THREADS=2 OMP_PROC_BIND=false python -u zh_graded_permeability.py
"""
import sys
import numpy as np
from peclet import amr

MU, FX, DT = 0.1, 1e-3, 60.0
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def sph(x, y, z):
    return ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) ** 0.5 - 0.22


def perm(graded, cf):
    t = amr.Octree(cells=[32, 32, 32], lmax=1 if graded else 0,
                   origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
    if graded:
        t.refine_to_sdf(sph, target_level=0, band=3.0)
    fl = amr.Flow(t, density=1.0, viscosity=MU, dt=DT)
    fl.set_cf_scheme(cf)
    fl.set_advection(False)
    fl.set_body_force(FX, 0.0, 0.0)
    fl.set_solid(sph)
    w = np.asarray(t.sizes(0)) * np.asarray(t.sizes(1)) * np.asarray(t.sizes(2))
    for _ in range(STEPS):
        fl.step(100, 60)
    u = np.asarray(fl.velocity(0))
    return MU * float((u * w).sum()) / FX, t.num_leaves


ref, nref = perm(False, 0)
print(f"uniform-fine reference (32^3, lmax=0): k = {ref:.10e}  ({nref} leaves)")
for cf in (0, 1):
    k, nl = perm(True, cf)
    print(f"graded band=3.0 cf={cf}: k = {k:.10e}  ({nl} leaves)  "
          f"relative error vs uniform-fine = {abs(k - ref) / abs(ref):.6e}")
