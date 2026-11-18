"""Timing case for host-orchestration refactors: a full ``set_solid`` + 20 steps (setup included).

A graded octree around a sphere (cut cells at two levels), the sampled ghost projection, advection
on: ``set_solid_spheres`` (the same ``setSolid`` with a C++ SDF, so the time is the builders' and
not a Python callback's) builds every setup structure the round touches (sampled overlay, momentum seam
delta, C/F overlays, seam reconstruction, pressure and velocity multigrid). Prints one line,
``setup_bench set_solid <s> steps <s> total <s>``; the round's timing gate compares the minima of
interleaved base/head runs at fixed threads (OMP_NUM_THREADS=8 OMP_PROC_BIND=false, host OpenMP).

    PYTHONPATH=<build> OMP_NUM_THREADS=8 OMP_PROC_BIND=false python tests/regression/setup_bench.py
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=96, help="finest cells per axis")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    from peclet import amr

    def sph(x, y, z):
        return ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) ** 0.5 - 0.27

    n = args.cells
    t = amr.Octree(cells=[n, n, n], lmax=2, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
    t.refine_to_sdf_graded(sph, lambda x, y, z: 0 if x < 0.5 else 1, band=2.0)
    fl = amr.Flow(t, density=1.0, viscosity=0.05, dt=0.02)
    fl.set_advection(True)
    fl.set_ghost_sampled(True)
    fl.set_body_force(1.0, 0.0, 0.0)
    import numpy as np
    centers, radii = np.array([[0.5, 0.5, 0.5]]), np.array([0.27])
    t0 = time.perf_counter()
    fl.set_solid_spheres(centers, radii, False)  # a C++ SDF: times the builders, not a Python callback
    t1 = time.perf_counter()
    for _ in range(args.steps):
        fl.step(mom_iters=60, pres_iters=40)
    t2 = time.perf_counter()
    print(f"setup_bench set_solid {t1 - t0:.3f} steps {t2 - t1:.3f} total {t2 - t0:.3f} "
          f"leaves {t.num_leaves}")


if __name__ == "__main__":
    main()
