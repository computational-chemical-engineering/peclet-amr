#!/usr/bin/env python3
"""Graded-mesh verification, rung 1b: the coarse/fine interface PARALLEL to the variation.

Companion to graded_poiseuille.py, which puts the 2:1 interface NORMAL to the direction the
solution varies in. That orientation turned out to be the one where `set_cf_scheme(1)` cannot help
by construction: the Martin-Cartwright correction is a TANGENTIAL quadratic, and a solution with no
tangential variation along the interface gives it nothing to correct (measured: standard and
quadratic agree to 6e-14 there, and the whole error sits in the coarse cells).

This script supplies the other orientation. Same exactly-quadratic Poiseuille solution
u_y(x) = (F/2mu)(x-a)(b-x), same grid-aligned walls, but the mesh is refined so that:

  * the WALL BAND is fine everywhere, so the cut cells stay uniformly at the finest level and the
    silent below-finest wall defect (ROADMAP A5) cannot contaminate the measurement;
  * the half-space y < y_c is ALSO refined to finest. The resulting 2:1 interface is the plane
    y = y_c, and the solution varies in x -- i.e. TANGENTIALLY along that interface, which is
    exactly the configuration the quadratic scheme exists for and the one
    tests/test_amr_cf_vector.cpp measures at operator level (order ~1.95 there).

If the quadratic scheme is wired into the momentum path correctly, this is where it shows.

Run:  PYTHONPATH=<amr build> python -u graded_poiseuille_tangential.py
"""
import sys

import numpy as np

from peclet import amr

MU = 1.0
F = 1.0
DT = 1e6
STEPS = 30


def exact(x, a, b):
    return (F / (2.0 * MU)) * (x - a) * (b - x)


def build(n_fine, lmax, a, b, band, y_c, tangential):
    h_fine = 32.0 / n_fine
    o = amr.Octree(cells=[n_fine] * 3, lmax=lmax, origin=[0.0, 0.0, 0.0], spacing=h_fine)
    o.refine_to_sdf(lambda x, y, z: min(x - a, b - x), 0, band, True)   # wall band -> finest
    if tangential:
        # ...and the y < y_c half, leaving a 2:1 plane at y = y_c with tangential variation on it.
        # levels(): 0 is the FINEST, larger is coarser. refine_leaf shifts later slots only, so a
        # reverse-order pass is safe; repeat to a fixpoint because balance() can expose more.
        for _ in range(8):
            lv, c = o.levels(), o.centers()
            idx = [i for i in range(o.num_leaves) if lv[i] > 0 and c[i, 1] < y_c]
            if not idx:
                break
            for i in idx[::-1]:
                o.refine_leaf(int(i))
            o.balance()
    return o


def run(o, a, b, cf_scheme):
    f = amr.Flow(o, density=1.0, viscosity=MU, dt=DT)
    f.set_cf_scheme(cf_scheme)
    f.set_advection(False)
    f.set_solid(lambda x, y, z: min(x - a, b - x))
    f.set_body_force(0.0, F, 0.0)
    for _ in range(STEPS):
        f.step(mom_iters=400, pres_iters=400)
    c = o.centers()
    x = c[:, 0]
    uy = f.velocity(1)
    m = f.is_fluid() & (x > a) & (x < b)
    d = uy[m] - exact(x[m], a, b)
    vol = o.sizes(0)[m] * o.sizes(1)[m] * o.sizes(2)[m]
    l2 = np.sqrt(np.sum(vol * d ** 2) / np.sum(vol))
    return float(np.abs(d).max()), float(l2), o.num_leaves, d, o.levels()[m]


def main():
    a, b, band, y_c, lmax = 8.0, 24.0, 2.0, 16.0, 1
    print(__doc__.split("Run:")[0].rstrip())
    print(f"\nbox 32^3, channel x in ({a}, {b}), lmax={lmax}, wall band {band} refined, "
          f"y < {y_c} refined\nexact peak u_y = {exact((a + b) / 2, a, b):.4f}\n")
    print(f"{'n_fine':>7} {'mesh':>12} {'cf':>10} {'leaves':>8} {'max err':>12} {'L2':>12} "
          f"{'ord(L2)':>8}")
    prev = {}
    for n in (32, 64):
        for tang, mname in ((False, "normal-only"), (True, "tangential")):
            for cf, cfname in ((0, "standard"), (1, "quadratic")):
                o = build(n, lmax, a, b, band, y_c, tang)
                emax, l2, nl, d, lev = run(o, a, b, cf)
                key = (tang, cf)
                ordr = np.log2(prev[key] / l2) if key in prev and l2 > 0 else float("nan")
                prev[key] = l2
                print(f"{n:7d} {mname:>12} {cfname:>10} {nl:8d} {emax:12.4e} {l2:12.4e} "
                      f"{ordr:8.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
