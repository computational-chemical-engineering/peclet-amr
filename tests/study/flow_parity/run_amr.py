#!/usr/bin/env python3
"""Run one parity case on peclet.amr's collocated solver (uniform grid, lmax=0) and dump to .npz.

Usage: run_amr.py <case.json> <out.npz>
The leaf field is de-Morton'd onto the (N,N,nz) x-fastest array flow uses, so the two dumps
are directly comparable cell for cell.
"""
import json
import sys
import time

import numpy as np

from peclet import amr


def leaf_to_grid(centers, cells, spacing, origin):
    """Permutation: amr leaf slot -> flat index of the (nx,ny,nz) C-order array."""
    idx = np.rint((centers - np.asarray(origin)) / np.asarray(spacing) - 0.5).astype(np.int64)
    nx, ny, nz = cells
    assert idx.min() >= 0 and (idx < np.array(cells)).all(), "leaf centres off grid"
    return (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]


def main():
    cfg = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    N, nz = cfg["N"], cfg["nz"]
    cells = [N, N, nz]
    oct_ = amr.Octree(cells=cells, lmax=0, origin=[0.0, 0.0, 0.0], spacing=1.0)
    f = amr.Flow(oct_, density=cfg["rho"], viscosity=cfg["mu"], dt=cfg["dt"])
    f.set_advection(cfg["advection"])
    if cfg["advection"]:
        f.set_advection_scheme({"sou": 0, "koren": 1}[cfg["adv_scheme"]])
        f.set_implicit_advection(cfg["implicit_adv"])
    f.set_ghost_projection(cfg["scheme"] == "ghost")
    if "uf_advection" in cfg:
        f.diagnostics.set_uf_advection(cfg["uf_advection"])
    if "ghost_gradient" in cfg:
        f.diagnostics.set_ghost_gradient(cfg["ghost_gradient"])
    if cfg.get("body_force"):
        f.set_body_force(*cfg["body_force"])

    if cfg["solid"] == "none":
        f.set_solid(lambda x, y, z: 1e3)
    elif cfg["solid"] == "channel":
        lo, hi = cfg["wall_lo"], cfg["wall_hi"]
        f.set_solid(lambda x, y, z: min(y - lo, hi - y))
    elif cfg["solid"] == "sphere":
        cx0, cy0, cz0 = cfg["center"]
        R = cfg["radius"]
        f.set_solid(lambda x, y, z: ((x-cx0)**2 + (y-cy0)**2 + (z-cz0)**2) ** 0.5 - R)
    else:
        raise SystemExit(f"unknown solid {cfg['solid']}")

    centers = oct_.centers()
    perm = leaf_to_grid(centers, cells, oct_.spacing, oct_.origin)

    if cfg["init"] == "taylor_green":
        k = 2.0 * np.pi / N
        cxl, cyl = centers[:, 0], centers[:, 1]
        U0 = cfg["U0"]
        f.set_velocity(0, U0 * np.sin(k * cxl) * np.cos(k * cyl))
        f.set_velocity(1, -U0 * np.cos(k * cxl) * np.sin(k * cyl))
        f.set_velocity(2, np.zeros_like(cxl))

    d = f.diagnostics
    pres_iters, mom_iters, wall = [], [], []
    for _ in range(cfg["steps"]):
        t0 = time.perf_counter()
        f.step(mom_iters=cfg["mom_iters"], pres_iters=cfg["pres_iters"])
        wall.append(time.perf_counter() - t0)
        pres_iters.append(int(d.last_pres_iters()))
        mom_iters.append(int(d.last_mom_iters()))

    ff = f.face_field()

    def grid(a):
        g = np.empty(N * N * nz)
        g[perm] = a
        return g.reshape(N, N, nz)

    np.savez(
        out,
        u=grid(f.velocity(0)), v=grid(f.velocity(1)), w=grid(f.velocity(2)),
        p=grid(f.pressure()),
        cx=np.unique(centers[:, 0]), cy=np.unique(centers[:, 1]), cz=np.unique(centers[:, 2]),
        pres_iters=np.array(pres_iters),
        mom_iters=np.array(mom_iters),
        wall=np.array(wall),
        div_cell=float(f.divergence_norm()),
        div_face=float(d.divergence_norm_face()),
        uf=grid(ff[0::6]), vf=grid(ff[2::6]), wf=grid(ff[4::6]),
        fluid=grid(f.is_fluid().astype(np.float64)),
    )
    print(f"[amr ] steps={cfg['steps']} pres_iters={pres_iters[:3]}..{pres_iters[-3:]} "
          f"mom_iters={mom_iters[:3]}..{mom_iters[-3:]} mean_wall={np.mean(wall)*1e3:.1f}ms "
          f"divcell={f.divergence_norm():.3e} divface={d.divergence_norm_face():.3e}")


if __name__ == "__main__":
    main()
