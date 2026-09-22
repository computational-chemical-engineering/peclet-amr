"""Gate P: graded permeability vs a uniform-fine reference on a FINEST-BAND mesh, which
docs/amr_cf_flux_gate.md sec.7 says must be bitwise identical across the per-FACE C/F gate.
Both recipes below have Flow.diagnostics.num_cf_cut_faces == 0, measured -- that is WHY they are
bit-identical, and it is now a measurement rather than an inference from a docstring.

Two recipes, because two records disagreed about which one produced the number the history
quotes (resolved 2026-09-22):

  A  "the byte gate's own graded sphere" -- state_hash.py's `sph`: r=0.22 at (0.5,0.5,0.5) in the
     UNIT BOX, cells=32^3, lmax=1, refine_to_sdf(band=3.0). Gives 2.969e-02 -> 4.586e-03.

  B  1b0d5b5's recipe, as its commit message states it: "graded sphere, N=32, lmax=2, band=3" --
     the Z&H sphere (solid fraction 0.125, R = (0.125*3/4pi)^(1/3)*N) in CELL UNITS (spacing=1).
     Gives 5.538e-02 -> 1.176e-02.

THE "5.15e-02 -> 1.22e-02" PAIR QUOTED IN ROADMAP sec A6 IS RECIPE B, NOT RECIPE A. ROADMAP
attributed it to "the byte gate's own graded sphere", which is recipe A and a factor 1.7 / 2.7
away; recipe B reproduces the pair to 8 % / 4 %, and its absolute k matches too (41.359
superficial = 47.267 interstitial against 1b0d5b5's recorded 47.182462, 0.18 %). The residual few
percent is the normalisation 1b0d5b5 never recorded -- superficial vs interstitial, and which
discrete fluid volume -- so this file pins a reproducible recipe for both and the ROADMAP
sentence has been corrected.

  PYTHONPATH=<build> OMP_NUM_THREADS=2 OMP_PROC_BIND=false python -u zh_graded_permeability.py
"""
import math
import sys

import numpy as np
from peclet import amr

MU, FX, DT = 0.1, 1e-3, 60.0
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def sph(x, y, z):
    return ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) ** 0.5 - 0.22


def perm(graded, cf):
    """Recipe A: the byte gate's own sphere, unit box, lmax=1."""
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


N_B = 32
R_B = (0.125 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N_B
C_B = N_B / 2.0


def zh(x, y, z):
    return ((x - C_B) ** 2 + (y - C_B) ** 2 + (z - C_B) ** 2) ** 0.5 - R_B


def perm_b(graded, cf):
    """Recipe B: 1b0d5b5's Z&H sphere, CELL UNITS (spacing=1), lmax=2."""
    t = amr.Octree(cells=[N_B, N_B, N_B], lmax=2 if graded else 0,
                   origin=[0, 0, 0], spacing=1.0)
    if graded:
        t.refine_to_sdf(zh, target_level=0, band=3.0)
    fl = amr.Flow(t, density=1.0, viscosity=MU, dt=DT)
    fl.set_cf_scheme(cf)
    fl.set_advection(False)
    fl.set_body_force(FX, 0.0, 0.0)
    fl.set_solid(zh)
    w = np.asarray(t.sizes(0)) * np.asarray(t.sizes(1)) * np.asarray(t.sizes(2))
    for _ in range(STEPS):
        fl.step(100, 60)
    u = np.asarray(fl.velocity(0))
    # superficial mean: divide by the TOTAL box volume, as recipe A does.
    return MU * float((u * w).sum()) / FX / N_B ** 3, t.num_leaves, fl.diagnostics.num_cf_cut_faces


print("=== recipe A: the byte gate's own sphere (r=0.22, unit box, lmax=1, band=3.0) ===")
ref, nref = perm(False, 0)
print(f"uniform-fine reference (32^3, lmax=0): k = {ref:.10e}  ({nref} leaves)")
for cf in (0, 1):
    k, nl = perm(True, cf)
    print(f"graded band=3.0 cf={cf}: k = {k:.10e}  ({nl} leaves)  "
          f"relative error vs uniform-fine = {abs(k - ref) / abs(ref):.6e}")

print("\n=== recipe B: 1b0d5b5's Z&H sphere (phi=0.125, cell units, N=32, lmax=2, band=3.0) ===")
refb, nrefb, _ = perm_b(False, 0)
print(f"uniform-fine reference (32^3, lmax=0): k = {refb:.10e}  ({nrefb} leaves)")
for cf in (0, 1):
    k, nl, cut = perm_b(True, cf)
    print(f"graded band=3.0 cf={cf}: k = {k:.10e}  ({nl} leaves, cfCutFaces {cut})  "
          f"relative error vs uniform-fine = {abs(k - refb) / abs(refb):.6e}")
