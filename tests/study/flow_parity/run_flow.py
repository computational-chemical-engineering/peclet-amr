#!/usr/bin/env python3
"""Run one parity case on peclet.flow's collocated solver and dump the state to .npz.

Usage: run_flow.py <case.json> <out.npz>
Grids are (N, N, nz), spacing 1, origin 0 -- identical to the peclet.amr lmax=0 octree.
"""
import json
import sys
import time

import numpy as np

from peclet import flow


def tg_fields(N, nz, U0, amp, centers):
    """2-D Taylor-Green, sampled at the solver's own cell centres."""
    k = 2.0 * np.pi / N
    cx, cy, _ = centers
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    u2 = U0 * amp * np.sin(k * X) * np.cos(k * Y)
    v2 = -U0 * amp * np.cos(k * X) * np.sin(k * Y)
    u = np.repeat(u2[:, :, None], nz, axis=2)
    v = np.repeat(v2[:, :, None], nz, axis=2)
    w = np.zeros((N, N, nz))
    return (np.asfortranarray(u), np.asfortranarray(v), np.asfortranarray(w))


def main():
    cfg = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    N, nz = cfg["N"], cfg["nz"]
    s = flow.SolverColocated(N, N, nz)
    s.set_rho(cfg["rho"])
    s.set_mu(cfg["mu"])
    s.set_dt(cfg["dt"])
    s.set_advection(cfg["advection"])
    if cfg["advection"]:
        s.set_advection_scheme(cfg["adv_scheme"])
        s.set_implicit_advection(cfg["implicit_adv"])
        # Same knob, same spelling, on both engines since flow took the projected face field as
        # its collocated advecting velocity (flow/doc/uf_advection.md, 2026-09-21). A case that
        # sets it therefore ablates BOTH engines together, which is the only way the ablation
        # keeps meaning a comparison.
        if "uf_advection" in cfg:
            s.diagnostics.set_uf_advection(cfg["uf_advection"])
    s.set_collocated_scheme(cfg["scheme"])
    if "rot_pressure" in cfg:
        s.diagnostics.set_rotational_pressure(cfg["rot_pressure"])
    if cfg.get("body_force"):
        s.set_body_force(tuple(cfg["body_force"]))
    s.diagnostics.set_velocity_solver_params(cfg["mom_iters"])
    s.set_pressure_pcg(True, cfg["pres_iters"], cfg["pres_rtol"])

    centers = s.cell_centers()
    cx, cy, cz = centers
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    if cfg["solid"] == "none":
        sdf = np.full((N, N, nz), 1e3)
    elif cfg["solid"] == "channel":
        lo, hi = cfg["wall_lo"], cfg["wall_hi"]
        sdf = np.minimum(Y - lo, hi - Y)
    elif cfg["solid"] == "sphere":
        c0 = np.array(cfg["center"])
        sdf = np.sqrt((X - c0[0])**2 + (Y - c0[1])**2 + (Z - c0[2])**2) - cfg["radius"]
    else:
        raise SystemExit(f"unknown solid {cfg['solid']}")
    s.set_solid(np.asfortranarray(sdf), cutcell_pressure=cfg["cutcell_pressure"])
    fluid = (sdf > 0.0).astype(np.float64)

    if cfg["init"] == "taylor_green":
        s.set_state(*tg_fields(N, nz, cfg["U0"], 1.0, centers))

    pres_iters, mom_res, wall = [], [], []
    for _ in range(cfg["steps"]):
        t0 = time.perf_counter()
        s.step()
        wall.append(time.perf_counter() - t0)
        pres_iters.append(int(s.diagnostics.last_pressure_iterations()))
        mom_res.append(float(s.diagnostics.last_momentum_residual()))

    np.savez(
        out,
        u=np.ascontiguousarray(s.get_u()),
        v=np.ascontiguousarray(s.get_v()),
        w=np.ascontiguousarray(s.get_w()),
        p=np.ascontiguousarray(s.get_p()),
        cx=cx, cy=cy, cz=cz,
        pres_iters=np.array(pres_iters),
        mom_res=np.array(mom_res),
        wall=np.array(wall),
        max_open_div=float(s.max_open_divergence()),
        fluid=fluid,
        uf=np.ascontiguousarray(s.get_uf()),
        vf=np.ascontiguousarray(s.get_vf()),
        wf=np.ascontiguousarray(s.get_wf()),
    )
    print(f"[flow] steps={cfg['steps']} pres_iters={pres_iters[:3]}..{pres_iters[-3:]} "
          f"mean_wall={np.mean(wall)*1e3:.1f}ms maxdiv={s.max_open_divergence():.3e}")


if __name__ == "__main__":
    main()
