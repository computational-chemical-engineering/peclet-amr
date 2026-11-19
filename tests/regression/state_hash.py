"""Byte-exactness gate for host-side refactors of peclet.amr: SHA-256 of fixed runs, per case.

Complements ``python/state_hash.py`` (the public-entry-path gate with its committed reference) with
cases aimed at the SETUP paths a host-orchestration refactor moves: ``AmrFlow::setSolid`` in every
projection mode, the sampled ghost overlay and its momentum seam delta, the C/F overlays and the
seam reconstruction, ``finishAdapt`` (conservative remap + rebuild), the distributed flow multigrid
(and its stages) at np = 1, 2, 4, and the distributed finite-volume operator, which ``peclet.amr``
does not expose and which ``regression_fv_dump`` (tests/regression/fv_dump.cpp) drives instead.

Single-rank cases
  sampled_graded   graded octree (cut cells at two levels), ``set_ghost_sampled``, advection on
  ghost_adapt      classic ghost projection, 3 steps, one solution-adaptive cycle, 2 steps
  aperture_cf0     aperture projection, standard C/F scheme, staircase velocity MG + GS smoother
MPI cases (``mpirun -np N``, N = 1, 2, 4; every per-rank array gathered and sorted by leaf centre)
  dist_sampled     DistributedOctree, a band coarsened on one side by ``adapt`` (two-level cut
                   band), ``set_ghost_sampled``, 3 steps
  dist_adapt       classic ghost projection, 2 steps, adapt + band protection + ``finish_adapt``
                   + ``rebalance_mpi``, 2 steps
  fv               ``regression_fv_dump`` (plain and openness-weighted operator, apply + Jacobi)

Every flow hash covers the velocities, pressure, face field and divergence norm; the momentum and
pressure iteration counts of the last step are recorded as their own keys (exact integers).

    PYTHONPATH=<build> python tests/regression/state_hash.py --build <build> --save hashes.json

Each case runs in its own process (the same interpreter) at
OMP_NUM_THREADS=1: the device reductions are order-dependent at more threads. A case that fails or
an MPI case that cannot launch makes the script exit 1 (the latter prints "mpi cases skipped").
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.abspath(__file__)
SINGLE_CASES = ["sampled_graded", "ghost_adapt", "aperture_cf0"]
MPI_CASES = ["dist_sampled", "dist_adapt"]
MPI_NP = [1, 2, 4]
MPIRUN = "/usr/bin/mpirun"


def sha(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sph(x, y, z, c=(0.5, 0.5, 0.5), r=0.22):
    return ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) ** 0.5 - r


def flow_state(fl):
    import numpy as np
    return (np.asarray(fl.velocities()), np.asarray(fl.pressure()), np.asarray(fl.face_field()),
            np.array([fl.divergence_norm()]))


def iters(fl, out, key):
    dg = fl.diagnostics
    out[key + ".mom_iters"] = int(dg.last_mom_iters())
    out[key + ".pres_iters"] = int(dg.last_pres_iters())


def make_flow(amr, tree, advect=True):
    fl = amr.Flow(tree, density=1.0, viscosity=0.05, dt=0.02)
    fl.set_advection(advect)
    fl.set_body_force(1.0, 0.0, 0.0)
    return fl


# ---- single-rank ------------------------------------------------------------------------------
def case_sampled_graded(amr, out):
    t = amr.Octree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
    t.refine_to_sdf_graded(sph, lambda x, y, z: 0 if x < 0.5 else 1, band=2.0)
    fl = make_flow(amr, t)
    fl.set_ghost_sampled(True)
    fl.set_solid(sph)
    for _ in range(3):
        fl.step(mom_iters=60, pres_iters=40)
    out["sampled_graded"] = sha(*flow_state(fl), fl.diagnostics.num_seam_sample_records)
    iters(fl, out, "sampled_graded")


def case_ghost_adapt(amr, out):
    import numpy as np
    t = amr.Octree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
    t.refine_to_sdf(sph, target_level=0, band=3.0)
    fl = make_flow(amr, t)
    fl.set_solid(sph)
    for _ in range(3):
        fl.step(mom_iters=60, pres_iters=40)
    out["ghost_adapt.pre"] = sha(*flow_state(fl), t.centers())
    umag = np.linalg.norm(np.asarray(fl.velocities()), axis=1)
    fl.begin_adapt()
    t.adapt(umag, 0.2, 0.05, finest_level=0)
    t.refine_to_sdf(sph, 0, 3.0, True)
    fl.finish_adapt(sph)
    for _ in range(2):
        fl.step(mom_iters=60, pres_iters=40)
    out["ghost_adapt"] = sha(*flow_state(fl), t.centers(), t.levels())
    iters(fl, out, "ghost_adapt")


def case_aperture_cf0(amr, out):
    t = amr.Octree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
    t.refine_to_sdf(sph, target_level=0, band=3.0)
    fl = make_flow(amr, t, advect=False)
    fl.set_ghost_projection(False)
    fl.set_cf_scheme(0)
    fl.diagnostics.set_velocity_mg_staircase(True)
    fl.diagnostics.set_momentum_gs(True)
    fl.set_solid(sph)
    for _ in range(2):
        fl.step(mom_iters=60, pres_iters=40)
    out["aperture_cf0"] = sha(*flow_state(fl))
    iters(fl, out, "aperture_cf0")


# ---- distributed ------------------------------------------------------------------------------
def gather_sorted(comm, cen, *arrays):
    """Concatenate per-rank arrays over ranks and sort the rows by leaf centre (rank 0 only)."""
    import numpy as np
    cols = [np.ascontiguousarray(cen)] + [np.ascontiguousarray(a).reshape(len(cen), -1)
                                          for a in arrays]
    parts = comm.gather(cols, root=0)
    if comm.rank != 0:
        return None
    full = [np.concatenate([p[k] for p in parts], axis=0) for k in range(len(cols))]
    c = full[0]
    order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
    return [a[order] for a in full]


def dist_hash(comm, d, fl, out, key):
    import numpy as np
    rows = gather_sorted(comm, d.centers(), d.levels(), fl.velocities(), fl.pressure())
    uf = comm.gather(np.asarray(fl.face_field()), root=0)  # face order is per rank: rank order
    div = fl.divergence_norm()
    dg = fl.diagnostics
    mi, pi = int(dg.last_mom_iters()), int(dg.last_pres_iters())
    if comm.rank == 0:
        out[key] = sha(*rows, np.concatenate(uf), np.array([div]))
        out[key + ".mom_iters"] = mi
        out[key + ".pres_iters"] = pi


def case_dist_sampled(amr, comm, out):
    import numpy as np
    d = amr.DistributedOctree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0],
                              extent=[1.0, 1.0, 1.0], periodic=[True, True, True])
    d.refine_to_sdf(sph, target_level=0, band=2.0, balance=True)
    # Coarsen the band on the x > 0.5 side (a flat indicator there): cut cells at two levels.
    c = d.centers()
    f = np.where(c[:, 0] < 0.5, np.sin(25.0 * c[:, 0]) * np.cos(25.0 * c[:, 1]), 0.0)
    d.adapt(f, 0.2, 0.05, finest_level=0)
    fl = make_flow(amr, d)
    fl.set_ghost_sampled(True)
    fl.set_solid(sph)
    for _ in range(3):
        fl.step(mom_iters=60, pres_iters=40)
    rec = comm.allreduce(int(fl.diagnostics.num_seam_sample_records))
    dist_hash(comm, d, fl, out, f"dist_sampled.np{comm.size}")
    if comm.rank == 0:
        out[f"dist_sampled.np{comm.size}.seam_records"] = rec


def case_dist_adapt(amr, comm, out):
    import numpy as np
    d = amr.DistributedOctree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0],
                              extent=[1.0, 1.0, 1.0], periodic=[True, True, True])
    d.refine_to_sdf(sph, target_level=0, band=3.0, balance=True)
    fl = make_flow(amr, d)
    fl.set_solid(sph)
    for _ in range(2):
        fl.step(mom_iters=60, pres_iters=40)
    umag = np.linalg.norm(np.asarray(fl.velocities()), axis=1)
    fl.begin_adapt()
    d.adapt(umag, 0.2, 0.05, finest_level=0)
    d.refine_to_sdf(sph, 0, 3.0, True)
    fl.finish_adapt(sph)
    fl.rebalance_mpi(sph)
    for _ in range(2):
        fl.step(mom_iters=60, pres_iters=40)
    dist_hash(comm, d, fl, out, f"dist_adapt.np{comm.size}")


# ---- driver -----------------------------------------------------------------------------------
def worker(case, path):
    """One case in this process (under mpirun for the MPI cases); rank 0 writes `path`."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_PROC_BIND", "false")
    comm = None
    if case in MPI_CASES:
        from mpi4py import MPI  # before peclet.amr: the module never calls MPI_Init
        comm = MPI.COMM_WORLD
    from peclet import amr
    out = {}
    if comm is None:
        globals()["case_" + case](amr, out)
    else:
        globals()["case_" + case](amr, comm, out)
    if comm is None or comm.rank == 0:
        out["toolchain"] = getattr(amr, "build_toolchain", "unknown")
        with open(path, "w") as fh:
            json.dump(out, fh, sort_keys=True)


def run(cmd, log):
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    log.write(f"$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}\n")
    return p.returncode, p.stdout  # stdout only: stderr may carry launcher notices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", help="the build tree (holds tests/regression_fv_dump)")
    ap.add_argument("--save", metavar="FILE", help="write the hashes as JSON")
    ap.add_argument("--cases", help="comma-separated subset (default: all)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.out)
        return 0
    if not args.build:
        ap.error("--build is required")
    want = set(args.cases.split(",")) if args.cases else None

    def sel(c):
        return want is None or c in want

    fv_exe = os.path.join(args.build, "tests", "regression_fv_dump")
    out, failed, skipped = {}, [], []
    mpirun_ok = os.access(MPIRUN, os.X_OK)
    tmp = tempfile.mkdtemp(prefix="amr_state_hash_")
    log_path = os.path.join(tmp, "cases.log")
    with open(log_path, "w") as log:
        jobs = [(c, None) for c in SINGLE_CASES if sel(c)]
        jobs += [(c, n) for c in MPI_CASES if sel(c) for n in MPI_NP]
        for case, np_ in jobs:
            path = os.path.join(tmp, f"{case}.np{np_}.json")
            cmd = [sys.executable, HERE, "--worker", case, "--out", path]
            if np_ is not None:
                if not mpirun_ok:
                    skipped.append(f"{case}.np{np_}")
                    continue
                cmd = [MPIRUN, "-np", str(np_), "--bind-to", "none"] + cmd
            rc, _ = run(cmd, log)
            if rc != 0 or not os.path.exists(path):
                failed.append(f"{case}" + ("" if np_ is None else f".np{np_}"))
                continue
            with open(path) as fh:
                out.update(json.load(fh))
        if sel("fv"):
            for n in MPI_NP:
                if not (mpirun_ok and os.access(fv_exe, os.X_OK)):
                    skipped.append(f"fv.np{n}")
                    continue
                rc, text = run([MPIRUN, "-np", str(n), "--bind-to", "none", fv_exe], log)
                if rc != 0:
                    failed.append(f"fv.np{n}")
                    continue
                out[f"fv.np{n}"] = hashlib.sha256(text.encode()).hexdigest()
    for k in sorted(out):
        print(f"{k} {out[k]}")
    if skipped:
        print(f"state_hash: mpi cases skipped: {', '.join(skipped)}")
    if failed:
        print(f"state_hash: FAILED cases: {', '.join(failed)} (log {log_path})")
    if args.save:
        with open(args.save, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    if not failed and not skipped:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed or skipped else 0


if __name__ == "__main__":
    sys.exit(main())
