#!/usr/bin/env python3
"""ROADMAP C1 — where the step time goes, and why the pressure solve used to dominate it.

The pressure multigrid's hierarchy used to stop at the ROOT BRICK: `AmrMultigrid::build` merges
complete sibling groups, and an octree leaf that is already a root cell has no siblings, so

    levels = lmax + 1,   coarsest grid = the root brick = (cells / 2**lmax)**3

On a UNIFORM mesh — any lmax, since unrefined leaves are all root cells — that was **one level**,
i.e. a smoother with no coarse-grid correction at all.  `suite/docs/DECOMPOSITION_AND_MULTIGRID.md`
§2.7 is the suite's recorded finding on exactly this: a V-cycle is domain-independent only if its
coarsest level is effectively solved, and the criterion is the coarsest grid's largest EXTENT
rather than its cell count.

`docs/amr_mg_depth.md` answers it: a level BELOW the root brick is the same octree with its root
LIFTED (brick halved, `lmax` incremented, leaf codes untouched), so `coarsenIf` keeps merging down
to an extent of 4.  This driver prints, per configuration, the hierarchy the solver actually built,
the hierarchy `predict_pressure_hierarchy` says it should have built, and what a step costs — so
every claim here is a measurement rather than a reading of the code.

Usage:  PYTHONPATH=<build> PECLET_AMR_PROFILE_STEP=1 python tests/study/amr_pressure_depth.py
        (without the env var it prints only the cost table)
"""
import argparse
import sys
import time

import numpy as np

from peclet import amr

MU = 1.0
FX = 0.1
DT = 0.05


def case(N, lmax, graded, ghost=True, steps=10, warm=3):
    R = 9.93 * N / 32.0
    sdf = lambda x, y, z: ((x - N / 2) ** 2 + (y - N / 2) ** 2 + (z - N / 2) ** 2) ** 0.5 - R
    o = amr.Octree(cells=[N] * 3, lmax=lmax, origin=[0.0] * 3, spacing=1.0)
    if graded and lmax:
        o.refine_to_sdf(sdf, 0, 4.0)
        o.balance()
    f = amr.Flow(o, density=1.0, viscosity=MU, dt=DT)
    f.set_advection(True)
    f.set_implicit_advection(True)
    f.set_ghost_projection(ghost)
    f.set_body_force(FX, 0.0, 0.0)
    f.set_solid(sdf)
    for _ in range(warm):
        f.step(mom_iters=400, pres_iters=400)
    t = time.perf_counter()
    for _ in range(steps):
        f.step(mom_iters=400, pres_iters=400)
    wall = (time.perf_counter() - t) / steps
    root = N // (1 << lmax)
    built = list(f.diagnostics.pressure_mg_levels)
    # The ladder `predict_pressure_hierarchy` describes.  Its `lmax` is the number of octree
    # coarsenings THE MESH supports, not the tree's declared lmax: an UNREFINED Octree(N, lmax=k)
    # is the same mesh as Octree(N // 2**k, lmax=0), all of whose leaves are root cells.
    depth = int(o.lmax) - int(np.min(o.levels()))
    pred = amr.predict_pressure_hierarchy(cells=[root * (1 << depth)] * 3, lmax=depth, num_ranks=1)
    return dict(N=N, lmax=lmax, mesh="graded" if graded else "uniform", leaves=o.num_leaves,
                root=root, ms=wall * 1e3, pres=int(f.diagnostics.last_pres_iters()),
                mom=int(f.diagnostics.last_mom_iters()), built=built,
                pred=int(pred["num_levels"]), bottom=f.diagnostics.pressure_mg_bottom,
                pred_bottom=pred["bottom"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghost", default="on", choices=["on", "off"])
    a = ap.parse_args()
    cfgs = [(32, 0, False), (32, 1, False), (32, 1, True), (32, 2, True),
            (64, 0, False), (64, 1, True), (64, 2, True), (64, 3, True)]
    print(f"{'N':>4} {'lmax':>5} {'mesh':>8} {'leaves':>8} {'root brick':>11} "
          f"{'ms/step':>9} {'pres it':>8} {'mom it':>7} {'levels':>7} {'pred':>5} {'bottom':>8}")
    bad = 0
    for N, lmax, graded in cfgs:
        r = case(N, lmax, graded, ghost=a.ghost == "on")
        ok = len(r["built"]) == r["pred"]
        bad += 0 if ok else 1
        print(f"{r['N']:>4} {r['lmax']:>5} {r['mesh']:>8} {r['leaves']:>8} "
              f"{str(r['root']) + '^3':>11} {r['ms']:>9.1f} {r['pres']:>8} {r['mom']:>7} "
              f"{len(r['built']):>7} {r['pred']:>5}{'' if ok else ' !'} {r['bottom']:>8}",
              flush=True)
        print(f"     levels: {' '.join(str(x) for x in r['built'])}", flush=True)
    if bad:
        print(f"WARNING: {bad} row(s) where the built ladder differs from the prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
