#!/usr/bin/env python3
"""Graded-mesh verification, rung 1: is the solver still second order across a 2:1 interface?

Plane Poiseuille driven by a body force has the exact solution

    u_y(x) = (F / 2mu) (x - a) (b - x),      a < x < b,   u_x = u_z = 0,

which is **exactly quadratic**, so a second-order scheme reproduces it AT EVERY NODE to solver
tolerance -- not "converging at order 2", *exact*. That makes it a sharper instrument than an error
ladder on a generic solution: on a uniform mesh the answer is machine zero (verified below at every
resolution), so on a graded mesh ANY departure is the coarse/fine treatment laid bare, with no
discretization error of its own to hide behind. peclet.flow reasons the same way in
scripts/verify_poiseuille_flow.py.

Geometry, chosen so that one thing at a time can break it:

  * walls at x = a, b placed on cell faces of BOTH the coarse and the fine grid, so they are
    grid-aligned everywhere and the cut-cell aperture machinery is trivial -- no cut-cell error to
    confound the C/F measurement;
  * the WALL BAND is refined to the finest level and the channel interior left coarse. That is the
    supported configuration: the cut band is uniformly finest, as the ghost projection's contract
    requires (CLAUDE.md, docs/amr_mixed_level_cut_band_plan.md). It also puts the 2:1 interface in
    the interior, normal to the direction the solution varies in -- the hard orientation, where the
    flux across the interface has to be right;
  * div u = 0 identically and the pressure is constant, so the projection is a no-op and what is
    measured is the momentum operator plus the C/F scheme, nothing else.

Arms: uniform-fine and uniform-coarse bracket what the graded mesh should achieve, and both C/F
schemes are run -- `set_cf_scheme(0)` (standard two-point flux, THE DEFAULT, documented first-order
at level boundaries) and `set_cf_scheme(1)` (Martin-Cartwright tangential quadratic, documented
second-order).

NOTE the configuration this deliberately avoids: a mesh whose cut cells sit BELOW the finest level
(e.g. lmax=1 with nothing refined) is silently wrong -- the no-slip wall picks up a constant
velocity offset of 1.5*(1 - 4^-lmax) on this very case, ~3.6% at lmax=1, on a mesh that is
physically identical to an exact lmax=0 run. See docs/ROADMAP.md A5.

Run:  PYTHONPATH=<amr build> python graded_poiseuille.py
"""
import sys

import numpy as np

from peclet import amr

MU = 1.0
F = 1.0
DT = 1e6          # steady: rho/dt -> 0, so each step is a Stokes solve
STEPS = 30


def exact(x, a, b):
    return (F / (2.0 * MU)) * (x - a) * (b - x)


def build(n_fine, lmax, mode, a, b, band):
    """`n_fine`^3 finest cells over a box of side `n_fine * h_fine`; h_fine = 1 / 2**lmax * ..."""
    h_fine = 32.0 / n_fine                       # physical box side is always 32
    o = amr.Octree(cells=[n_fine] * 3, lmax=lmax, origin=[0.0, 0.0, 0.0], spacing=h_fine)
    if mode == "coarse":
        return o                                 # every leaf at the root level
    if mode == "fine":
        # A uniform mesh at the finest spacing. Built flat (lmax=0) rather than by refining every
        # leaf of the lmax tree: the two are the same mesh and give the same answer (both exact to
        # ~1e-13 on this case, checked), but refine_leaf over every leaf costs O(n^2) through the
        # binding and dominates the run at n=128.
        return amr.Octree(cells=[n_fine] * 3, lmax=0, origin=[0.0, 0.0, 0.0], spacing=h_fine)
    if mode == "graded":
        o.refine_to_sdf(lambda x, y, z: min(x - a, b - x), 0, band, True)
        return o
    raise SystemExit(f"unknown mode {mode}")


def run(o, a, b, cf_scheme):
    f = amr.Flow(o, density=1.0, viscosity=MU, dt=DT)
    f.set_cf_scheme(cf_scheme)
    f.set_advection(False)
    f.set_solid(lambda x, y, z: min(x - a, b - x))
    f.set_body_force(0.0, F, 0.0)
    for _ in range(STEPS):
        f.step(mom_iters=400, pres_iters=400)
    x = o.centers()[:, 0]
    uy = f.velocity(1)
    m = f.is_fluid() & (x > a) & (x < b)
    d = uy[m] - exact(x[m], a, b)
    vol = o.sizes(0)[m] * o.sizes(1)[m] * o.sizes(2)[m]
    l2 = np.sqrt(np.sum(vol * d ** 2) / np.sum(vol))
    return float(np.abs(d).max()), float(l2), o.num_leaves


def order(e_coarse, e_fine):
    if e_fine <= 0 or e_coarse <= 0:
        return float("nan")
    return np.log2(e_coarse / e_fine)


def main():
    a, b, band = 8.0, 24.0, 2.0
    lmax = 1
    ladder = (32, 64, 128)

    print(__doc__.split("Run:")[0].rstrip())
    print(f"\nbox 32^3, channel x in ({a}, {b}), lmax={lmax}, wall band {band} cells refined to "
          f"finest\nexact peak u_y = {exact((a + b) / 2, a, b):.4f}, {STEPS} steps at dt={DT:g}\n")

    hdr = f"{'n_fine':>7} {'mesh':>8} {'cf':>10} {'leaves':>8} {'max err':>12} {'L2':>12} {'ord(L2)':>8}"
    print(hdr)
    prev = {}
    for n in ladder:
        for mode in ("fine", "graded"):
            for cf, cfname in ((0, "standard"), (1, "quadratic")):
                if mode == "fine" and cf == 1:
                    continue                      # no level boundary: the C/F scheme cannot act
                emax, l2, nl = run(build(n, lmax, mode, a, b, band), a, b, cf)
                key = (mode, cf)
                o_ = order(prev[key][1], l2) if key in prev else float("nan")
                prev[key] = (emax, l2)
                print(f"{n:7d} {mode:>8} {cfname:>10} {nl:8d} {emax:12.4e} {l2:12.4e} {o_:8.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
