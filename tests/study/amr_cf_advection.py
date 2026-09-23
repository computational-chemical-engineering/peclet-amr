#!/usr/bin/env python3
"""ROADMAP B5 — where the convective flux loses an order at a 2:1 coarse/fine face.

Two instruments, no time stepping and no projection in the second one.

`--step`  The one-step truncation of the whole solver (docs/amr_pressure_iteration.md §14.4's
          a-priori test): warm up four steps so the face field and deferred-correction caches are
          live, reset u and p to the exact Taylor-Green fields, take ONE step, and read
          (u1 - u0)/dt - du/dt|exact, split into the cells that own a 2:1 face and the rest, with
          advection on and off.  Taylor-Green is an exact solution of BOTH Navier-Stokes (with the
          TG pressure) and Stokes (with p = 0), so the same du/dt = -2 nu k^2 u is the reference
          for both arms and their difference isolates the convective term.

`--flux`  The convective operator ALONE, reproduced in numpy from Flow.diagnostics.face_topology():

              conv_i = (1/V_i) sum_k rawArea_k (dir_k uf_k) phiFace_k,
              phiFace = 1.5 phi[up] - 0.5 phi[upup]        (SOU, the shipped default)

          with uf taken EXACT at the sub-face centroid, so what is measured is the reconstruction
          of the ADVECTED value alone.  phi = u_x, and div(u u_x) = u.grad u_x = (k/2) sin 2kx
          exactly, because the Taylor-Green field is divergence free.  Four variants:

              none  as shipped
              star  Martin-Cartwright coarse* (poisson.hpp::coarseStar) on every cell in the
                    stencil that is COARSER than the sub-face -- the tangential fix
              dist  the same two stencil points extrapolated with their ACTUAL distances to the
                    sub-face centroid, phi_up + (phi_up - phi_upup) d1/(d2 - d1), which reduces to
                    (1.5, -0.5) only when the two are equally spaced
              both  star + dist

          and a census of what the stencil lands on.

Usage:  PYTHONPATH=<build> python tests/study/amr_cf_advection.py --flux --n 16 32 64
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from amr_tg_graded import MU, RHO, build, exact, pexact  # noqa: E402


def tg(c, N):
    """The Taylor-Green velocity at t = 0 (the manufactured field of the flux test)."""
    k = 2.0 * np.pi / N
    return np.stack([np.sin(k * c[..., 0]) * np.cos(k * c[..., 1]),
                     -np.cos(k * c[..., 0]) * np.sin(k * c[..., 1]),
                     np.zeros(c.shape[:-1])], axis=-1)


def interface_cells(o, top):
    """Mask of leaves that own at least one 2:1 sub-face, and the per-slot owner / C/F flags."""
    own = np.repeat(np.arange(o.num_leaves), np.diff(top["start"]))
    cf = o.levels()[own] != o.levels()[top["nbr"]]
    m = np.zeros(o.num_leaves, bool)
    m[own[cf]] = True
    return m, own, cf


def vrms(r, vol, mask):
    return float(np.sqrt((vol[mask] * r[mask] ** 2).sum() / vol[mask].sum())) if mask.any() else 0.0


# ---- instrument 1: the one-step truncation of the whole solver -----------------------------------

def step_probe(arm, N, cfl, advection, warm=4):
    dt = cfl
    o, f = build(arm, N, dt, "quadratic")
    f.set_advection(advection)
    for _ in range(warm):
        f.step()
    t = warm * dt
    c = o.centers()
    ue = np.stack(exact(c, t, N), axis=1)
    for k in range(3):
        f.set_velocity(k, ue[:, k].copy())
    f.set_pressure(pexact(c, t, N) if advection else np.zeros(len(c)))
    f.step()
    u1 = np.stack([f.velocity(0), f.velocity(1), f.velocity(2)], axis=1)
    k = 2.0 * np.pi / N
    r = (u1 - ue) / dt + 2.0 * (MU / RHO) * k * k * ue     # residual of du/dt = -2 nu k^2 u
    touch, _, _ = interface_cells(o, f.diagnostics.face_topology())
    vol = o.sizes(0) ** 3
    e2 = (r ** 2).sum(axis=1)
    return dict(arm=arm, N=N, adv=advection, n_if=int(touch.sum()),
                rms_if=vrms(np.sqrt(e2), vol, touch), rms_bulk=vrms(np.sqrt(e2), vol, ~touch),
                max_if=float(np.abs(r[touch]).max()) if touch.any() else 0.0,
                max_bulk=float(np.abs(r[~touch]).max()))


# ---- instrument 2: the convective operator alone -------------------------------------------------

_CACHE = {}


def _mesh(arm, N):
    if (arm, N) not in _CACHE:
        o, f = build(arm, N, 0.5, "quadratic")
        c, wid = o.centers(), o.sizes(0)
        lo = np.rint(c - wid[:, None] / 2.0).astype(np.int64)
        sz = np.rint(wid).astype(np.int64)
        owner = np.full((N, N, N), -1, np.int64)
        for i in range(o.num_leaves):
            x, y, z = lo[i]
            owner[x:x + sz[i], y:y + sz[i], z:z + sz[i]] = i
        _CACHE[(arm, N)] = (o, c, o.levels(), wid, lo, sz, owner, f.diagnostics.face_topology())
    return _CACHE[(arm, N)]


def flux_probe(arm, N, mode):
    o, c, lev, wid, lo, sz, owner, top = _mesh(arm, N)
    n = o.num_leaves
    start, nbr, axis, dr = top["start"], top["nbr"], top["axis"], top["dir"]
    ra, upI, upJ = top["raw_area"], top["upup_i"], top["upup_j"]
    ocell = np.repeat(np.arange(n), np.diff(start))
    phi = tg(c, N)[:, 0]
    k = 2.0 * np.pi / N
    sl = np.arange(len(nbr))

    fine = np.where(lev[nbr] < lev[ocell], nbr, ocell)
    sgn = np.where(fine == ocell, 1.0, -1.0)
    fc = c[fine].copy()
    fc[sl, axis] += sgn * dr * 0.5 * wid[fine]            # the sub-face centroid
    velOut = dr * tg(fc, N)[sl, axis]                     # exact advecting velocity there
    up = np.where(velOut > 0, ocell, nbr)
    upup = np.where(velOut > 0, upI, upJ)
    cf = lev[ocell] != lev[nbr]
    bad = upup < 0
    upupS = np.maximum(upup, 0)
    phiUp, phiUpUp = phi[up].copy(), np.where(bad, phi[up], phi[upupS])

    if mode in ("star", "both"):
        def star(cells, ref, ax_):
            v = phi[cells].copy()
            for t in range(3):
                for q in np.where((lev[cells] > lev[ref]) & (ax_ != t))[0]:
                    cc, rr = cells[q], ref[q]
                    H = wid[cc]
                    off = (lo[rr][t] + 0.5 * sz[rr]) - (lo[cc][t] + 0.5 * sz[cc])
                    pp, pm = lo[cc].copy(), lo[cc].copy()
                    pp[t] = (pp[t] + sz[cc]) % N
                    pm[t] = (pm[t] - 1) % N
                    cp, cm = owner[tuple(pp)], owner[tuple(pm)]
                    if lev[cp] != lev[cc] or lev[cm] != lev[cc]:
                        continue
                    v[q] += off * (phi[cp] - phi[cm]) / (2 * H) \
                        + 0.5 * off * off * (phi[cp] - 2 * phi[cc] + phi[cm]) / (H * H)
            return v
        phiUp[cf] = star(up[cf], fine[cf], axis[cf])
        ok = cf & ~bad
        phiUpUp[ok] = star(upupS[ok], fine[ok], axis[ok])

    if mode in ("dist", "both"):
        d1 = np.abs(fc[sl, axis] - c[up, axis])
        d2 = np.abs(fc[sl, axis] - c[upupS, axis])
        d1, d2 = np.minimum(d1, N - d1), np.minimum(d2, N - d2)   # periodic minimum image
        w = np.where(np.abs(d2 - d1) > 1e-12, d1 / (d2 - d1), 0.5)
        phiFace = phiUp + (phiUp - phiUpUp) * w
    else:
        phiFace = 1.5 * phiUp - 0.5 * phiUpUp
    phiFace = np.where(bad, phiUp, phiFace)

    fe = np.abs(phiFace - tg(fc, N)[:, 0])
    conv = np.bincount(ocell, weights=ra * velOut * phiFace, minlength=n) / wid ** 3
    r = conv - 0.5 * k * np.sin(2 * k * c[:, 0])          # u.grad u_x, exact
    touch = np.zeros(n, bool)
    touch[ocell[cf]] = True
    vol = wid ** 3
    upC = cf & (lev[up] > lev[fine])

    def fr(m):
        return float(np.sqrt((fe[m] ** 2).mean())) if m.any() else 0.0

    return dict(arm=arm, N=N, mode=mode, rms_if=vrms(r, vol, touch), rms_bulk=vrms(r, vol, ~touch),
                f_cf=fr(cf), f_reg=fr(~cf), f_upc=fr(upC), f_upf=fr(cf & ~upC),
                n_cf=int(cf.sum()), n_upc=int(upC.sum()),
                n_jump=int((cf & ~bad & (lev[upupS] != lev[up])).sum()), n_noupup=int((cf & bad).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--step", action="store_true", help="the whole-solver one-step truncation")
    ap.add_argument("--flux", action="store_true", help="the convective operator alone")
    a = ap.parse_args()
    if not (a.step or a.flux):
        a.step = a.flux = True

    if a.step:
        print("# one-step truncation of the whole solver, cell CFL 0.5")
        print(f"{'arm':>4} {'N':>4} {'adv':>6} {'#if':>7} {'rms @interface':>15} {'rms bulk':>11} "
              f"{'ratio':>7} {'max @if':>11} {'max bulk':>11}")
        for N in a.n:
            for arm in ("C", "G"):
                for adv in (False, True):
                    d = step_probe(arm, N, 0.5, adv)
                    q = d["rms_if"] / d["rms_bulk"] if d["rms_bulk"] else float("nan")
                    print(f"{d['arm']:>4} {d['N']:>4} {str(d['adv']):>6} {d['n_if']:>7} "
                          f"{d['rms_if']:>15.4e} {d['rms_bulk']:>11.4e} {q:>7.2f} "
                          f"{d['max_if']:>11.3e} {d['max_bulk']:>11.3e}", flush=True)

    if a.flux:
        for N in a.n:
            d0 = flux_probe("G", N, "none")
            print(f"\n# N={N}: {d0['n_cf']} C/F sub-face slots, upwind is the COARSE cell on "
                  f"{d0['n_upc']}; stencil crosses a level on {d0['n_jump']}; no upup on "
                  f"{d0['n_noupup']}")
            print(f"{'mode':>6} {'div rms @if':>12} {'div rms bulk':>13} {'face C/F':>11} "
                  f"{'face reg':>11} {'face up=C':>11} {'face up=F':>11}")
            for m in ("none", "star", "dist", "both"):
                d = flux_probe("G", N, m)
                print(f"{d['mode']:>6} {d['rms_if']:>12.4e} {d['rms_bulk']:>13.4e} "
                      f"{d['f_cf']:>11.4e} {d['f_reg']:>11.4e} {d['f_upc']:>11.4e} "
                      f"{d['f_upf']:>11.4e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
