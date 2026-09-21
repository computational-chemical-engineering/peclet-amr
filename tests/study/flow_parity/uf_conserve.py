"""Momentum conservation under the two advecting velocities.

In a triply-periodic box with no body force, Navier-Stokes conserves total momentum exactly. A
Galilean-shifted Taylor-Green (u += U_mean) starts with a non-zero mean, so any drift is numerical.
The ABC/Basilisk argument for advecting with the PROJECTED face field is precisely this: a
non-solenoidal advecting flux injects a spurious u*(div u_adv) term that the flux form cannot cancel.
"""
import numpy as np
from peclet import amr

def run(N, use_uf, U_mean=0.5, nu=0.05, dt=0.5, steps=60, U0=1.0, nz=4):
    k = 2.0 * np.pi / N
    o = amr.Octree(cells=[N, N, nz], lmax=0, origin=[0, 0, 0], spacing=1.0)
    f = amr.Flow(o, density=1.0, viscosity=nu, dt=dt)
    f.set_advection(True)
    f.diagnostics.set_uf_advection(use_uf)
    f.set_solid(lambda x, y, z: 1e3)
    c = o.centers()
    u0 = U0 * np.sin(k*c[:,0]) * np.cos(k*c[:,1]) + U_mean
    v0 = -U0 * np.cos(k*c[:,0]) * np.sin(k*c[:,1])
    f.set_velocity(0, u0.copy()); f.set_velocity(1, v0.copy())
    f.set_velocity(2, np.zeros(len(c)))
    m0 = u0.mean()
    hist = []
    for s in range(steps):
        f.step(mom_iters=200, pres_iters=200)
        hist.append(f.velocity(0).mean())
    return m0, np.array(hist)

print("Galilean-shifted Taylor-Green, periodic, no body force: mean(u) must stay constant")
print(f"{'N':>4} {'advecting velocity':>22} {'mean(u) t=0':>13} {'mean(u) end':>13} {'rel drift':>12}")
for N in (32, 64):
    for use_uf, name in ((True, 'projected uf'), (False, '1/2(u_i+u_j)')):
        m0, h = run(N, use_uf)
        print(f"{N:4d} {name:>22} {m0:13.9f} {h[-1]:13.9f} {abs(h[-1]-m0)/abs(m0):12.3e}")
