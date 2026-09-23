#!/usr/bin/env python3
"""WO0 — the graded time-accurate benchmark (docs/amr_pressure_iteration.md §9).

Decaying 2-D Taylor-Green in a triply periodic box, uniform in z, advection ON, run on

  (U) a uniform octree (lmax = 0), the time-accuracy reference, and
  (G) a graded octree (lmax = 1) whose finest cells form a spherical shell at the box centre,
      so the pressure gradient has a tangential component on the whole coarse/fine surface,

over a ladder of time steps.  It answers two questions the project could not answer before:

  * how accurate the graded solver is in an UNSTEADY flow with an exact solution (the gap
    ROADMAP.md B1/B2 carry: "no convergence study of the graded solver against an analytic
    solution"), and
  * how large the face-velocity error at coarse/fine sub-faces is NEXT TO the velocity error
    the solver makes anyway — gate W of the pressure-iteration note, which decides whether the
    deferred-correction option (B) is worth building.

Exact solution (rho = 1, mu = nu, U0, k = 2*pi/N, cell spacing 1 so the box side is N):

    u =  U0 sin(kx) cos(ky) e^{-2 nu k^2 t}
    v = -U0 cos(kx) sin(ky) e^{-2 nu k^2 t}
    w =  0
    p = (U0^2/4) (cos 2kx + cos 2ky) e^{-4 nu k^2 t}

Metrics at t = T (§9): m1 the worst face-normal velocity error over the 2:1 sub-faces, m2 the
same over every face, m3 the volume-weighted L2 cell velocity error, m4 the divergence of the
face field, m5 = m1/m3.  m5inf = m1/(max-norm cell error) is reported beside m5 because m1 is a
max and m3 an L2; gate W is stated on m5.  m2r is m2 restricted to the REGULAR (same-level)
faces and m6 = m1/m2r answers the question m5 cannot: whether a coarse/fine sub-face is a worse
place for the advecting velocity than an ordinary face on the same mesh.

Time is measured in CONVECTIVE units L/U0 = N: the horizon is `--turnovers` of them, so the
step count at a fixed cell CFL = U0 dt / h_fine = dt doubles with N and every CFL of the ladder
divides it exactly.

Usage:
    PYTHONPATH=<build> python tests/study/amr_tg_graded.py                    # N=32, full ladder
    PYTHONPATH=<build> python tests/study/amr_tg_graded.py --n 64 --cfl 0.5 2
    PYTHONPATH=<build> python tests/study/amr_tg_graded.py --json docs/data/amr_tg_graded.json
"""
import argparse
import json
import sys
import time

import numpy as np

from peclet import amr

U0 = 1.0
RHO = 1.0
MU = 0.05


def exact(c, t, N):
    """Exact (u, v, w) at world points `c` (..., 3) and time `t`."""
    k = 2.0 * np.pi / N
    f = np.exp(-2.0 * (MU / RHO) * k * k * t)
    x, y = c[..., 0], c[..., 1]
    return (U0 * np.sin(k * x) * np.cos(k * y) * f,
            -U0 * np.cos(k * x) * np.sin(k * y) * f,
            np.zeros_like(x))


def pexact(c, t, N):
    """Exact pressure at world points `c` and time `t` (rho = 1, so p is in velocity-squared)."""
    k = 2.0 * np.pi / N
    return (RHO * U0 ** 2 / 4.0) * (np.cos(2 * k * c[..., 0]) + np.cos(2 * k * c[..., 1])) \
        * np.exp(-4.0 * (MU / RHO) * k * k * t)


SEAM = True          # --seam off flips the B5 seam reconstruction (docs/amr_cf_convective.md)
BAND = 1.0           # --band: shell thickness in cells, i.e. how much of the box gets refined


def build(arm, N, dt, cf_scheme):
    """Octree + Flow for one arm, initialised to the t = 0 Taylor-Green field."""
    lmax = 0 if arm == "U" else 1
    o = amr.Octree(cells=[N, N, N], lmax=lmax, origin=[0.0, 0.0, 0.0], spacing=1.0)
    if arm == "G":
        # The finest cells are the SHELL the sphere surface passes through (the octree's own
        # semantics), which gives two closed coarse/fine surfaces instead of one.
        o.refine_to_sphere([N / 2.0] * 3, N / 4.0, 0, BAND)
        o.balance()
    f = amr.Flow(o, density=RHO, viscosity=MU, dt=dt)
    f.set_advection(True)
    f.set_implicit_advection(True)
    f.set_cf_scheme({"standard": 0, "quadratic": 1}[cf_scheme])   # BEFORE set_solid: the C/F
    f.set_solid(lambda x, y, z: 1e3)          # overlays are built there, and there is no solid
    if not SEAM:
        f.diagnostics.set_seam_reconstruction(False)
    c = o.centers()
    u, v, w = exact(c, 0.0, N)
    f.set_velocity(0, u)
    f.set_velocity(1, v)
    f.set_velocity(2, w)
    # Without this the first step is an impulsive pressure start on every arm: the predictor runs
    # without -G p^n and the projection has to manufacture the whole pressure in one go
    # (docs/amr_pressure_iteration.md §14.4(b) measures 7-8e-3 of shape error from it).
    f.set_pressure(pexact(c, 0.0, N))
    return o, f


def leak(o, f, phi, N):
    """eps_cf: the pressure-increment leak at the 2:1 sub-faces, max |Delta_G phi| over them.

    The face gradient in uf is the compact two-point (phi_C - phi_F)/d; at a 2:1 face the two
    centres are offset TANGENTIALLY as well, so the face value carries
    Delta_G phi = (tangential offset) . grad phi / d.  grad phi is a least-squares fit over the
    coarse cell's own face neighbours; d = 1.5 h_fine.  This is the quantity gate W of
    docs/amr_pressure_iteration.md §14.3 is stated on -- not m1, which is the TOTAL face error.
    """
    c, lev, wid = o.centers(), o.levels(), o.sizes(0)
    top = f.diagnostics.face_topology()
    start, nbr, axis = top["start"], top["nbr"], top["axis"]
    own = np.repeat(np.arange(o.num_leaves), np.diff(start))
    cf = lev[own] != lev[nbr]
    if not cf.any():
        return 0.0
    coarse = np.where(lev[nbr] > lev[own], nbr, own)[cf]
    fine = np.where(lev[nbr] > lev[own], own, nbr)[cf]
    ax = axis[cf]
    off = (c[fine] - c[coarse] + N / 2.0) % N - N / 2.0     # periodic minimum image
    off[np.arange(len(ax)), ax] = 0.0                       # tangential part only
    uc, ci = np.unique(coarse, return_inverse=True)
    g = np.zeros((len(uc), 3))
    for n, i in enumerate(uc):
        js = nbr[start[i]:start[i + 1]]
        dx = (c[js] - c[i] + N / 2.0) % N - N / 2.0
        g[n] = np.linalg.lstsq(dx, phi[js] - phi[i], rcond=None)[0]
    return float(np.abs((off * g[ci]).sum(axis=1) / (1.5 * wid[fine])).max())


def metrics(o, f, t, N, phi=None):
    """m1..m7 of §9 / §14.3 at time `t`."""
    c = o.centers()
    lev = o.levels()
    wid = o.sizes(0)
    ue = exact(c, t, N)
    uc = np.stack([f.velocity(0), f.velocity(1), f.velocity(2)], axis=1)
    err = uc - np.stack(ue, axis=1)
    vol = wid ** 3
    e2 = (err ** 2).sum(axis=1)

    def l2(mask):
        v = vol[mask]
        return float(np.sqrt((v * e2[mask]).sum() / v.sum())) if v.size else 0.0

    m3 = l2(np.ones(len(vol), bool))
    m3inf = float(np.abs(err).max())
    # Split m3 into an AMPLITUDE error (the numerical field decays at the wrong rate) and a SHAPE
    # error (what is left after the best-fit amplitude is removed).  A graded mesh that merely
    # damps the vortex differently looks identical to one that deforms it, in m3 alone.
    ue3 = np.stack(ue, axis=1)
    amp = float((vol * (uc * ue3).sum(axis=1)).sum() / (vol * (ue3 ** 2).sum(axis=1)).sum())
    sh = uc - amp * ue3
    m3shape = float(np.sqrt((vol * (sh ** 2).sum(axis=1)).sum() / vol.sum()))

    top = f.diagnostics.face_topology()
    start, nbr, axis, dr = top["start"], top["nbr"], top["axis"], top["dir"]
    uf = np.asarray(f.face_field())
    own = np.repeat(np.arange(o.num_leaves), np.diff(start))
    cf = lev[own] != lev[nbr]
    # A sub-face's sample point is the FINER incident cell's face centre (levels: 0 = finest).
    fine = np.where(lev[nbr] < lev[own], nbr, own)
    sgn = np.where(fine == own, 1.0, -1.0)
    fc = c[fine].copy()
    fc[np.arange(len(fine)), axis] += sgn * dr * 0.5 * wid[fine]
    fe = np.stack(exact(fc, t, N), axis=1)[np.arange(len(fine)), axis]
    d = np.abs(uf - fe)
    m1 = float(d[cf].max()) if cf.any() else 0.0
    m2 = float(d.max())
    m2r = float(d[~cf].max())
    m4 = float(f.diagnostics.divergence_norm_face())
    m7 = leak(o, f, phi, N) if phi is not None else 0.0
    # Where the cell error lives: cells with a 2:1 face of their own vs the rest.
    touch = np.zeros(o.num_leaves, bool)
    touch[own[cf]] = True
    m3cf, m3bulk = l2(touch), l2(~touch)
    return dict(m1=m1, m2=m2, m2r=m2r, m3=m3, m3cf=m3cf, m3bulk=m3bulk,
                amp=amp, m3shape=m3shape, m7=m7, m7r=m7 / m2r if m2r else float("nan"),
                cf_cells=int(touch.sum()), m3inf=m3inf, m4=m4,
                m6=m1 / m2r if m2r else float("nan"),
                m5=m1 / m3 if m3 else float("nan"),
                m5inf=m1 / m3inf if m3inf else float("nan"),
                cf_faces=int(cf.sum()), leaves=int(o.num_leaves))


def run(arm, N, cfl, turnovers, cf_scheme, pres_tol):
    dt = cfl                                  # h_fine = 1, U0 = 1 => CFL == dt
    T = turnovers * N
    nsteps = int(round(T / dt))
    assert abs(nsteps * dt - T) < 1e-9, f"CFL {cfl} does not divide the horizon {T}"
    o, f = build(arm, N, dt, cf_scheme)
    if pres_tol:
        f.set_pressure_tolerance(pres_tol)
    t0 = time.perf_counter()
    for _ in range(nsteps - 1):
        f.step()
    p0 = np.array(f.pressure())
    f.step()                                  # phi of the LAST step drives the leak instrument
    phi = (dt / RHO) * (np.array(f.pressure()) - p0)
    wall = time.perf_counter() - t0
    r = metrics(o, f, T, N, phi)
    r.update(arm=arm, N=N, cfl=cfl, dt=dt, T=T, steps=nsteps, cf_scheme=cf_scheme,
             wall=wall, pres_iters=int(f.diagnostics.last_pres_iters()))
    return r


HDR = (f"{'arm':>4} {'N':>4} {'CFL':>6} {'steps':>6} {'leaves':>8} "
       f"{'m1 (C/F uf)':>12} {'m2r (reg uf)':>12} {'m3 (L2 cell)':>12} "
       f"{'m3 @C/F':>10} {'m3 shape':>10} {'1-amp':>9} {'m4 div(uf)':>11} {'m5':>7} {'m6':>6} {'eps_cf':>10} {'eps/m2r':>9} {'wall s':>7}")


def line(r):
    return (f"{r['arm']:>4} {r['N']:>4} {r['cfl']:>6g} {r['steps']:>6} {r['leaves']:>8} "
            f"{r['m1']:>12.4e} {r['m2r']:>12.4e} {r['m3']:>12.4e} {r['m3cf']:>10.3e} {r['m3shape']:>10.3e} {1 - r['amp']:>9.2e} "
            f"{r['m4']:>11.3e} {r['m5']:>7.3f} {r['m6']:>6.3f} {r['m7']:>10.3e} {r['m7r']:>9.2e} {r['wall']:>7.1f}")


def gate():
    """The regression gate (ctest `python_amr_tg_graded`, label `bench`): the three facts of
    docs/amr_tg_graded.md that a change to the graded solver must not break.  ~20 s."""
    rows = [run("G", N, cfl, 2.0, "quadratic", 0.0)
            for N in (16, 32) for cfl in (0.5, 1.0, 2.0)]
    rows += [run("U", 32, 0.5, 2.0, "quadratic", 0.0)]
    print(HDR)
    for r in rows:
        print(line(r))
    bad = []
    # (1) The advecting face field is a conservative flux on a graded mesh in an unsteady flow
    # with advection on -- the property restored at peclet-amr 1b0d5b5 / 9da368c.
    for r in rows:
        if r["m4"] > 1e-9:
            bad.append(f"div(uf) = {r['m4']:.3e} > 1e-9 at N={r['N']} CFL={r['cfl']:g}")
    # (2) A 2:1 sub-face is not a worse place for the advecting velocity than an ordinary face.
    for r in rows:
        if r["arm"] == "G" and r["m6"] > 1.1:
            bad.append(f"m1/m2r = {r['m6']:.3f} > 1.1 at N={r['N']} CFL={r['cfl']:g}")
    # (3) Gate W of docs/amr_pressure_iteration.md §14.3: the pressure-increment leak at the 2:1
    # sub-faces, against the worst face error the solver makes at an ORDINARY face of the same mesh,
    # at the largest time-accurate dt (CFL 1). 4.0e-3 when this was written; the bound is 12x that,
    # so it trips only on a real change to the C/F pressure gradient, never on solver noise.
    for r in rows:
        if r["arm"] == "G" and r["cfl"] == 1.0 and r["m7r"] > 0.05:
            bad.append(f"eps_cf/m2r = {r['m7r']:.3e} > 0.05 at N={r['N']} CFL=1")
    # (4) The graded solver still converges, and to the same answer. Second order is a 32 -> 64
    # statement (measured 2.07, docs/amr_tg_graded.md §3); the 16 -> 32 rung the gate can afford is
    # PRE-ASYMPTOTIC -- the shell is four cells across there -- and reads ~1.2, so the gate floors
    # it at 0.9 and pins the levels themselves instead.
    for cfl in (0.5, 2.0):
        a = next(r for r in rows if r["arm"] == "G" and r["N"] == 16 and r["cfl"] == cfl)
        b = next(r for r in rows if r["arm"] == "G" and r["N"] == 32 and r["cfl"] == cfl)
        o = np.log2(a["m3"] / b["m3"])
        print(f"# graded m3 order 16->32 at CFL {cfl:g}: {o:.2f} (pre-asymptotic; 2.22 at 32->64)")
        if o < 0.9:
            bad.append(f"graded m3 order {o:.2f} < 0.9 at CFL {cfl:g}")
    # (5) The error levels themselves, +-5 % of what docs/amr_tg_graded.md §3 records. Wide enough
    # for a compiler or thread-count change, narrow enough that a numerics change has to say so.
    for arm, N, cfl, ref in (("G", 16, 0.5, 1.2388e-01), ("G", 32, 0.5, 5.2383e-02),
                             ("G", 32, 1.0, 4.7745e-02), ("U", 32, 0.5, 5.7998e-03)):
        r = next(x for x in rows if x["arm"] == arm and x["N"] == N and x["cfl"] == cfl)
        if abs(r["m3"] / ref - 1.0) > 0.05:
            bad.append(f"m3 = {r['m3']:.4e} is {100 * (r['m3'] / ref - 1):+.1f} % off the recorded "
                       f"{ref:.4e} at arm {arm} N={N} CFL={cfl:g}")
    for b in bad:
        print(f"FAIL: {b}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="run the short regression gate instead of the ladder")
    ap.add_argument("--n", type=int, nargs="+", default=[32])
    ap.add_argument("--cfl", type=float, nargs="+", default=[0.5, 2.0, 8.0, 32.0])
    ap.add_argument("--arms", nargs="+", default=["U", "G"], choices=["U", "G", "C"],
                    help="U uniform fine (lmax 0); G graded (fine shell); "
                         "C uniform COARSE (lmax 1, unrefined) -- the control that says what the "
                         "graded arm's coarse bulk alone would cost")
    ap.add_argument("--cf-scheme", default="quadratic", choices=["quadratic", "standard"])
    ap.add_argument("--turnovers", type=float, default=2.0,
                    help="horizon in convective times L/U0 = N (default 2)")
    ap.add_argument("--pres-tol", type=float, default=0.0,
                    help="pressure solve rtol (0 = the solver default)")
    ap.add_argument("--seam", default="on", choices=["on", "off"],
                    help="the B5 seam reconstruction of the advected value (default on)")
    ap.add_argument("--band", type=float, default=1.0,
                    help="refined shell thickness in cells (how much of the box is refined)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    global SEAM, BAND
    SEAM = a.seam == "on"
    BAND = a.band
    if a.gate:
        return gate()

    print(f"# Taylor-Green, rho={RHO} mu={MU} U0={U0}, horizon {a.turnovers} convective times,"
          f" cf={a.cf_scheme}")
    print(HDR)
    rows = []
    for N in a.n:
        for arm in a.arms:
            for cfl in a.cfl:
                r = run(arm, N, cfl, a.turnovers, a.cf_scheme, a.pres_tol)
                rows.append(r)
                print(line(r), flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"# wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
