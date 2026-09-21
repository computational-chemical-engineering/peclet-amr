#!/usr/bin/env python3
"""Does the advecting-velocity choice actually buy accuracy?

2-D Taylor-Green in a triply-periodic box is an EXACT Navier-Stokes solution (the nonlinear term is
balanced by the pressure gradient), so there is a right answer to measure against:

    u =  U0 sin(kx) cos(ky) e^{-2 nu k^2 t},  v = -U0 cos(kx) sin(ky) e^{-2 nu k^2 t}

Run peclet.amr with the projected divergence-free face field as the advecting velocity (the shipped
scheme) and with the un-projected 1/2(u_i+u_j) cell->face average (peclet.flow's collocated choice),
and compare both to the analytic decay. This is the physical stake in parity note decision P1.
"""
import sys

import numpy as np

from peclet import amr


def run(N, use_uf, nu=0.05, dt=0.5, steps=40, U0=1.0, nz=4):
    k = 2.0 * np.pi / N
    o = amr.Octree(cells=[N, N, nz], lmax=0, origin=[0, 0, 0], spacing=1.0)
    f = amr.Flow(o, density=1.0, viscosity=nu, dt=dt)
    f.set_advection(True)
    f.diagnostics.set_uf_advection(use_uf)
    f.set_solid(lambda x, y, z: 1e3)
    c = o.centers()
    u0 = U0 * np.sin(k * c[:, 0]) * np.cos(k * c[:, 1])
    v0 = -U0 * np.cos(k * c[:, 0]) * np.sin(k * c[:, 1])
    f.set_velocity(0, u0.copy())
    f.set_velocity(1, v0.copy())
    f.set_velocity(2, np.zeros(len(c)))
    for _ in range(steps):
        f.step(mom_iters=200, pres_iters=200)
    amp = np.exp(-2.0 * nu * k * k * dt * steps)
    ue, ve = u0 * amp, v0 * amp
    uu, vv = f.velocity(0), f.velocity(1)
    l2 = np.sqrt(np.mean((uu - ue) ** 2 + (vv - ve) ** 2)) / np.sqrt(np.mean(u0 ** 2 + v0 ** 2))
    e_ratio = np.mean(uu ** 2 + vv ** 2) / np.mean(u0 ** 2 + v0 ** 2)
    return l2, e_ratio, amp * amp


def main():
    print("2-D Taylor-Green vs the exact decay (peclet.amr, lmax=0, advection ON, 40 steps)")
    print(f"{'N':>4} {'advecting velocity':>22} {'L2 err vs exact':>16} {'E/E0':>10} {'exact':>10}")
    prev = {}
    for N in (32, 64):
        for use_uf, name in ((True, "projected uf"), (False, "1/2(u_i+u_j)")):
            l2, em, ea = run(N, use_uf)
            print(f"{N:4d} {name:>22} {l2:16.4e} {em:10.5f} {ea:10.5f}")
            prev[(N, use_uf)] = l2
    print()
    for N in (32, 64):
        a, b = prev[(N, True)], prev[(N, False)]
        print(f"N={N}: projected uf is {b / a:.2f}x {'MORE' if b > a else 'LESS'} accurate "
              f"than the plain average")
    for use_uf, name in ((True, "projected uf"), (False, "1/2(u_i+u_j)")):
        r = prev[(32, use_uf)] / prev[(64, use_uf)]
        print(f"{name}: observed order {np.log2(r):.2f} (32 -> 64)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
