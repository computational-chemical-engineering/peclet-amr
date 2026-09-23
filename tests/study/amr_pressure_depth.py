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


def case(N, lmax, graded, ghost=True, steps=10, warm=3, bottom_extent=None, emulate=False,
         bottom=None, reps=1):
    """One configuration.  `emulate` refines EVERYWHERE to level 0, so an `lmax = k` tree carries
    the same N^3 uniform mesh as an `lmax = 0` one — the emulation the C1 design rests on and the
    reference the §10 depth gate now compares the lifted ladder against, on the same box."""
    R = 9.93 * N / 32.0
    sdf = lambda x, y, z: ((x - N / 2) ** 2 + (y - N / 2) ** 2 + (z - N / 2) ** 2) ** 0.5 - R
    o = amr.Octree(cells=[N] * 3, lmax=lmax, origin=[0.0] * 3, spacing=1.0)
    if emulate and lmax:
        o.refine_to_sdf(lambda x, y, z: 0.0, 0, 1.0e18)   # everywhere -> the uniform N^3 mesh
        o.balance()
    elif graded and lmax:
        o.refine_to_sdf(sdf, 0, 4.0)
        o.balance()
    f = amr.Flow(o, density=1.0, viscosity=MU, dt=DT)
    f.set_advection(True)
    f.set_implicit_advection(True)
    f.set_ghost_projection(ghost)
    f.set_body_force(FX, 0.0, 0.0)
    # Both knobs take effect at set_solid, which is where the pressure hierarchy is built.
    if bottom_extent is not None:
        f.diagnostics.set_pressure_bottom_extent(bottom_extent)   # §6.2/§11.4
    if bottom is not None:
        f.diagnostics.set_pressure_bottom(bottom)                 # §6.6
    f.set_solid(sdf)
    for _ in range(warm):
        f.step(mom_iters=400, pres_iters=400)
    # `reps` independent timed windows on the SAME solver; report the MINIMUM. The mean is the
    # wrong estimator on a shared box -- contention only ever adds time, so the minimum is the
    # closest thing to the machine's own number. (Measured noise floor on this host: ~3 %.)
    wall = None
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(steps):
            f.step(mom_iters=400, pres_iters=400)
        w = (time.perf_counter() - t) / steps
        wall = w if wall is None else min(wall, w)
    root = N // (1 << lmax)
    built = list(f.diagnostics.pressure_mg_levels)
    # The ladder `predict_pressure_hierarchy` describes.  Its `lmax` is the number of octree
    # coarsenings THE MESH supports, not the tree's declared lmax: an UNREFINED Octree(N, lmax=k)
    # is the same mesh as Octree(N // 2**k, lmax=0), all of whose leaves are root cells.
    depth = int(o.lmax) - int(np.min(o.levels()))
    be = 4 if bottom_extent is None else bottom_extent
    pred = amr.predict_pressure_hierarchy(cells=[root * (1 << depth)] * 3, lmax=depth,
                                          num_ranks=1, bottom_extent=be)
    return dict(N=N, lmax=lmax,
                mesh="emul" if emulate else ("graded" if graded else "uniform"),
                leaves=o.num_leaves,
                root=root, ms=wall * 1e3, pres=int(f.diagnostics.last_pres_iters()),
                mom=int(f.diagnostics.last_mom_iters()), built=built,
                pred=int(pred["num_levels"]), bottom=f.diagnostics.pressure_mg_bottom,
                pred_bottom=pred["bottom"])


def sweep(ghost):
    """docs/amr_mg_depth.md §11.4, folded into WO5: where should the ladder stop?

    The first pass of this sweep confounded two variables. `bottomExtent` decides BOTH how many
    (tiny, launch-bound) levels the ladder builds AND -- with the default `auto` bottom -- whether
    the coarsest level is SOLVED or merely smoothed: §6.6's justification for the 60-sweep damped
    Jacobi bottom is that it is exact at extent <= 4 (8e-9), and it is not at 8 (8e-3) or 16 (0.30).
    So a one-dimensional sweep over the extent compares "deeper ladder, exact bottom" against
    "shallower ladder, inexact bottom", which is not the question.

    This is the 2-D form: extent x bottom kind, with `agglomerated` forcing the exact solve at
    EVERY extent so that depth is the only variable on that arm. `smoother` is the other arm --
    always the sweeps -- so the pair also measures what the exact bottom is worth per extent.

    READ THE COLUMNS WITH CARE -- this is the trap that cost the C1 investigation four rounds.
    `ms/step` carries a ~3 % noise floor on this host, hence the minimum of `reps` timed windows
    (the floor is evidenced without modelling by the lmax 3 bE 8 / bE 16 rows, which build an
    IDENTICAL 4-level hierarchy and read within 1-3 % of each other). `pres it` LOOKS deterministic
    and is not: it is `last_pres_iters()` from ONE step, and it is a Krylov count against a moving
    right-hand side that wanders by +/-2 between consecutive steps of the same run. A single sample
    of it once read 15 against 11 where a 33-step trace of those same two configurations gives
    median 13 (range 11-15) against median 12 (range 11-13) -- the sample had caught the top of one
    distribution and the bottom of the other, and a one-iteration effect was investigated as a
    four-iteration one. Treat the column as an order of magnitude, and TRACE PER STEP before
    resting any conclusion on a difference smaller than about three iterations
    (docs/amr_mg_depth.md §11.10).
    """
    cfgs = [(64, 0, False), (64, 2, True), (64, 3, True)]
    print(f"{'N':>4} {'lmax':>5} {'mesh':>8} {'bottom':>13} {'bE':>4} {'ms/step':>9} "
          f"{'pres it':>8} {'levels':>7} {'coarsest':>9} {'reported':>10}")
    for N, lmax, graded in cfgs:
        for kind in ("smoother", "agglomerated"):
            for be in (4, 8, 16):
                r = case(N, lmax, graded, ghost=ghost == "on", bottom_extent=be, bottom=kind,
                         reps=3)
                print(f"{r['N']:>4} {r['lmax']:>5} {r['mesh']:>8} {kind:>13} {be:>4} "
                      f"{r['ms']:>9.1f} {r['pres']:>8} {len(r['built']):>7} "
                      f"{r['built'][-1]:>9} {r['bottom']:>10}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghost", default="on", choices=["on", "off"])
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the bottom extent (docs/amr_mg_depth.md §11.4) instead")
    a = ap.parse_args()
    if a.sweep:
        return sweep(a.ghost)
    cfgs = [(32, 0, False), (32, 1, False), (32, 1, True), (32, 2, True),
            (64, 0, False), (64, 1, True), (64, 2, True), (64, 3, True)]
    # The §10 depth gate's reference: the SAME 64^3 uniform mesh inside a depth-3 tree (the
    # emulation), measured on the same box in the same session.
    emul = [(64, 3)]
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
    for N, lmax in emul:
        r = case(N, lmax, False, ghost=a.ghost == "on", emulate=True)
        ok = len(r["built"]) == r["pred"]
        bad += 0 if ok else 1
        print(f"{r['N']:>4} {r['lmax']:>5} {r['mesh']:>8} {r['leaves']:>8} "
              f"{str(r['root']) + '^3':>11} {r['ms']:>9.1f} {r['pres']:>8} {r['mom']:>7} "
              f"{len(r['built']):>7} {r['pred']:>5}{'' if ok else ' !'} {r['bottom']:>8}",
              flush=True)
    if bad:
        print(f"WARNING: {bad} row(s) where the built ladder differs from the prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
