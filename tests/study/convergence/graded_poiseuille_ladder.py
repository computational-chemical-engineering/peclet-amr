#!/usr/bin/env python3
"""Graded-mesh convergence on a SELF-SIMILAR mesh family, both interface orientations.

Supersedes graded_poiseuille.py and graded_poiseuille_tangential.py, which built their meshes with
`refine_to_sdf(..., band=N)`. `band` counts CELLS, so the refined region shrinks physically as the
ladder climbs and the mesh family changes shape under it -- measured coarse fraction of the channel
0.111 / 0.273 / 0.216 at n = 32 / 64 / 128, not even monotone. A max-norm order survives that (it
is attained at a 2:1 interface whose local geometry is identical at every rung) but an L2 order does
not, and neither does any comparison between orientations whose meshes scale differently.

Here the refined region is an explicit PHYSICAL predicate, fixed in world units:

    refine to finest  <=>  |x - wall| < W          (the wall band; keeps cut cells uniformly
                                                    finest, which ROADMAP A5 was about)
                      or   y < y_c                 (tangential arm only)

which gives a coarse-fraction that is constant to four digits across the ladder (0.2727 for the
normal arm, 0.0698 for the tangential), i.e. a real refinement study.

The two orientations answer different questions:

  * NORMAL      -- the 2:1 interface is the plane x = const at the wall-band edge, normal to the
                   direction u_y(x) varies in. No tangential variation along it, so the
                   Martin-Cartwright quadratic correction has nothing to correct and is inert BY
                   CONSTRUCTION; what is measured is the normal two-point flux.
  * TANGENTIAL  -- the interface is the plane y = y_c, and the solution varies in x ALONG it. This
                   is what set_cf_scheme('quadratic') exists for.

Exact solution u_y(x) = (F/2mu)(x-a)(b-x) is exactly quadratic, so a second-order scheme is exact
on it and the uniform arm comes out at machine zero; any graded departure is the C/F treatment.

Run:  PYTHONPATH=<amr build> python -u graded_poiseuille_ladder.py
"""
import sys

import numpy as np

from peclet import amr

MU = 1.0
F = 1.0
DT = 1e6
STEPS = 30
BOX = 32.0


def exact(x, a, b):
    return (F / (2.0 * MU)) * (x - a) * (b - x)


def build(n, a, b, W, y_c, tangential):
    """lmax=1 tree refined to finest on a FIXED PHYSICAL region -> self-similar across the ladder."""
    o = amr.Octree(cells=[n] * 3, lmax=1, origin=[0.0, 0.0, 0.0], spacing=BOX / n)
    for _ in range(12):                       # fixpoint: balance() can expose more to refine
        lv, c = o.levels(), o.centers()
        want = np.minimum(np.abs(c[:, 0] - a), np.abs(c[:, 0] - b)) < W
        if tangential:
            want |= c[:, 1] < y_c
        idx = np.nonzero((lv > 0) & want)[0]  # levels(): 0 is the FINEST
        if idx.size == 0:
            break
        for i in idx[::-1]:                   # reverse: refining leaf i does not move i-1
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
    m = f.is_fluid() & (x > a) & (x < b)
    d = f.velocity(1)[m] - exact(x[m], a, b)
    vol = o.sizes(0)[m] * o.sizes(1)[m] * o.sizes(2)[m]
    coarse = float((o.levels()[m] == 1).sum()) / max(int(m.sum()), 1)
    return (float(np.abs(d).max()),
            float(np.sqrt(np.sum(vol * d ** 2) / np.sum(vol))),
            o.num_leaves, coarse)


def main():
    a, b, W, y_c = 8.0, 24.0, 2.0, 16.0
    ladder = (32, 64)
    print(__doc__.split("Run:")[0].rstrip())
    print(f"\nbox {BOX:g}^3, channel x in ({a}, {b}), lmax=1, refine |x-wall| < {W}"
          f" (+ y < {y_c} on the tangential arm)\nexact peak u_y = {exact((a+b)/2, a, b):.4f}, "
          f"{STEPS} steps at dt={DT:g}\n")
    print(f"{'n':>5} {'orientation':>12} {'cf':>10} {'leaves':>8} {'coarse%':>8} "
          f"{'max err':>12} {'ratio':>7} {'L2':>12} {'ratio':>7}")
    prev = {}
    for n in ladder:
        for tang, oname in ((False, "normal"), (True, "tangential")):
            o = build(n, a, b, W, y_c, tang)
            for cf, cfname in ((0, "standard"), (1, "quadratic")):
                emax, l2, nl, coarse = run(o, a, b, cf)
                k = (tang, cf)
                rm = prev[k][0] / emax if k in prev and emax > 0 else float("nan")
                rl = prev[k][1] / l2 if k in prev and l2 > 0 else float("nan")
                prev[k] = (emax, l2)
                print(f"{n:5d} {oname:>12} {cfname:>10} {nl:8d} {100*coarse:7.2f}% "
                      f"{emax:12.4e} {rm:7.2f} {l2:12.4e} {rl:7.2f}")
        print()
    print("ratio 4.00 = second order, 2.00 = first order, <1 = not converging")
    return 0


if __name__ == "__main__":
    sys.exit(main())
