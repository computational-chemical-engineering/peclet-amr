"""Fixed-seed reference runs of every public entry path of peclet.amr, hashed.

The structural gate of suite/docs/QUALITY_PLAN.md §3.G: a refactor that moves code verbatim must
leave every final state BYTE-IDENTICAL. This script runs one deterministic scenario per public
entry path, hashes the final arrays (SHA-256 of the raw float64/int bytes) and prints them; with
``--save FILE`` it records them as JSON and with ``--check FILE`` it compares against a recording
and exits non-zero on any difference.

    PYTHONPATH=<python build tree> OMP_NUM_THREADS=1 python python/state_hash.py --save pre.json
    PYTHONPATH=<python build tree> OMP_NUM_THREADS=1 mpirun -np 2 python python/state_hash.py --check pre.json

Under ``mpirun -np N`` (N > 1) the distributed paths (DistributedOctree, the distributed Flow) run
too; every per-rank array is gathered to rank 0 in
rank order before hashing, so the hash names carry the rank count (``.np2``). Run at
OMP_NUM_THREADS=1: the device reductions are order-dependent at more than one thread.

The reference recorded on the pre-relocation peclet-core build (peclet.core.amr, 2026-09-10) is
``python/state_hash_reference.json``; the relocation reproduced it byte for byte.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def sha(*arrays):
    h = hashlib.sha256()
    flat = []
    for a in arrays:  # a binding may return a tuple of arrays: hash each member in order
        flat += list(a) if isinstance(a, (tuple, list)) and not np.isscalar(a[0]) else [a]
    for a in flat:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def gather_rows(comm, a):
    """Concatenate a per-rank (n, k) array over ranks in rank order (rank 0 gets the result)."""
    if comm is None or comm.size == 1:
        return np.ascontiguousarray(a)
    parts = comm.gather(np.ascontiguousarray(a), root=0)
    return np.concatenate(parts, axis=0) if comm.rank == 0 else None


# ---------------------------------------------------------------------------------------------
# AMR — Octree refine/balance/adapt, Poisson, Flow (ghost projection; mixed-level sampled band),
# DistributedOctree + distributed Flow at np > 1.
# ---------------------------------------------------------------------------------------------
def import_amr():
    from peclet import amr
    return amr


def toolchain():
    """The module's compiler / version / build type (`build_toolchain`). Hashes are comparable only
    between builds of one toolchain: the last bits of a Krylov iterate depend on FMA contraction and
    the optimisation level, so a reference recorded elsewhere is SKIPPED, not failed."""
    return getattr(import_amr(), "build_toolchain", "unknown")


def run_amr(out, comm):
    amr = import_amr()
    size, rank = (comm.size, comm.rank) if comm is not None else (1, 0)

    def sph(x, y, z, c=(0.5, 0.5, 0.5), r=0.22):
        return ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) ** 0.5 - r

    if rank == 0:
        # ---- Octree: refine to a sphere, balance, geometry, Löhner adapt ----
        t = amr.Octree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        t.refine_to_sphere(center=[0.5, 0.5, 0.5], radius=0.22, target_level=0, band=1.0,
                           balance=False)
        t.balance()
        out["amr.octree"] = sha(t.centers(), t.sizes(), t.levels(), t.codes())
        f = 2.0 + np.tanh((t.centers()[:, 0] - 0.5) / 0.05)
        ind = t.lohner_indicator(f, eps=0.01)
        f2 = t.adapt(f, refine_thresh=0.2, coarsen_thresh=0.05, finest_level=0)
        out["amr.adapt"] = sha(ind, f2, t.centers(), t.levels())

        # ---- Poisson multigrid on a graded octree ----
        tg = amr.Octree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        tg.refine_to_sphere(center=[0.5, 0.5, 0.5], radius=0.25, target_level=0, band=1.0)
        pg = amr.Poisson(tg, periodic=True)
        cc = tg.centers()
        k = 2 * np.pi
        ue = np.cos(k * cc[:, 0]) + np.sin(2 * k * cc[:, 1]) * np.cos(k * cc[:, 2])
        ue -= ue.mean()
        b = pg.apply(ue)
        u, r, ncyc = pg.solve(b, cycles=12, tol=0.0)
        out["amr.poisson"] = sha(b, u, np.array([r, float(ncyc)]))

        # ---- Flow, ghost projection (the default), uniform finest band around a sphere ----
        tf = amr.Octree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        tf.refine_to_sdf(sph, target_level=0, band=3.0)
        fl = amr.Flow(tf, density=1.0, viscosity=0.05, dt=0.02)
        fl.set_advection(True)
        fl.set_solid(sph)
        fl.set_body_force(1.0, 0.0, 0.0)
        for _ in range(3):
            fl.step(mom_iters=60, pres_iters=40)
        out["amr.flow_ghost"] = sha(fl.velocities(), fl.pressure(), fl.face_field(),
                                    np.array([fl.divergence_norm()]))

        # ---- Flow, mixed-level sampled band: cut cells at TWO levels (graded refinement) ----
        ts = amr.Octree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        ts.refine_to_sdf_graded(sph, lambda x, y, z: 0 if x < 0.5 else 1, band=2.0)
        fs = amr.Flow(ts, density=1.0, viscosity=0.05, dt=0.02)
        fs.set_ghost_sampled(True)
        fs.set_solid(sph)
        fs.set_body_force(1.0, 0.0, 0.0)
        for _ in range(3):
            fs.step(mom_iters=60, pres_iters=40)
        out["amr.flow_sampled"] = sha(fs.velocities(), fs.pressure(), fs.face_field(),
                                      np.array([fs.divergence_norm()]))

    # ---- DistributedOctree (collective; also meaningful at np=1) ----
    tag = f".np{size}"
    d = amr.DistributedOctree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0],
                              extent=[1.0, 1.0, 1.0], periodic=[True, True, True])
    d.refine_to_sphere(center=[0.5, 0.5, 0.5], radius=0.22, target_level=0, band=1.0,
                       balance=True)
    cen, lev = gather_rows(comm, d.centers()), gather_rows(comm, d.levels().reshape(-1, 1))
    if cen is not None:
        out["amr.distributed_octree" + tag] = sha(cen, lev)
    fields = np.column_stack([d.levels().astype(np.float64), d.centers()[:, 0]])
    moved = d.rebalance(fields)
    cen2, mv = gather_rows(comm, d.centers()), gather_rows(comm, moved)
    if cen2 is not None:
        out["amr.distributed_rebalance" + tag] = sha(cen2, mv)
    fa = 2.0 + np.tanh((d.centers()[:, 0] - 0.5) / 0.05)
    fa2 = d.adapt(fa, refine_thresh=0.2, coarsen_thresh=0.05, finest_level=0)
    cen3, f3 = gather_rows(comm, d.centers()), gather_rows(comm, fa2.reshape(-1, 1))
    if cen3 is not None:
        out["amr.distributed_adapt" + tag] = sha(cen3, f3)

    # ---- distributed Flow: the whole step multi-rank through the leaf halo ----
    df = amr.DistributedOctree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0],
                               extent=[1.0, 1.0, 1.0], periodic=[True, True, True])
    df.refine_to_sdf(sph, target_level=0, band=3.0, balance=True)
    fd = amr.Flow(df, density=1.0, viscosity=0.05, dt=0.02)
    fd.set_advection(True)
    fd.set_solid(sph)
    fd.set_body_force(1.0, 0.0, 0.0)
    for _ in range(3):
        fd.step(mom_iters=60, pres_iters=40)
    vc, vp = gather_rows(comm, df.centers()), gather_rows(comm, fd.velocities())
    pp = gather_rows(comm, fd.pressure().reshape(-1, 1))
    if vc is not None:
        # Slot order differs between partitions; sort by leaf centre so np=1 and np=2 hashes are
        # comparable across the same decomposition (the gate compares like with like anyway).
        order = np.lexsort((vc[:, 2], vc[:, 1], vc[:, 0]))
        out["amr.distributed_flow" + tag] = sha(vc[order], vp[order], pp[order])


RUNNERS = {"amr": run_amr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", default="amr", help="the module subset to run (only `amr` here)")
    ap.add_argument("--save", metavar="FILE", help="write the hashes as JSON")
    ap.add_argument("--check", metavar="FILE", help="compare against a JSON recording")
    args = ap.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        sys.stderr.write("state_hash: run with OMP_NUM_THREADS=1 (device reductions are order-dependent)\n")
    comm = None
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        pass
    rank = comm.rank if comm is not None else 0
    out = {}
    for name in args.modules.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            RUNNERS[name](out, comm)
        except ImportError as e:
            if rank == 0:
                print(f"# {name}: not importable ({e}); skipped")
    if rank != 0:
        return 0
    for k in sorted(out):
        print(f"{k} {out[k]}")
    rc = 0
    if args.check:
        ref = json.load(open(args.check))
        want = ref.pop("toolchain", None)
        have = toolchain()
        if want is not None and want != have:
            print(f"state_hash: reference recorded with toolchain '{want}', this build is '{have}' — "
                  "not comparable; SKIPPED (exit 77). Re-record with --save on this toolchain to gate it.")
            return 77
        # A recording may merge several rank counts; compare only the keys this run can produce
        # (no `.npN` suffix, or the suffix of the current communicator size).
        size = comm.size if comm is not None else 1
        ref = {k: v for k, v in ref.items() if ".np" not in k or k.endswith(f".np{size}")}
        for k in sorted(set(ref) | set(out)):
            if k not in out:
                print(f"MISSING {k}")
                rc = 1
            elif k not in ref:
                print(f"NEW {k}")
            elif ref[k] != out[k]:
                print(f"DIFFER {k}: {ref[k][:16]}... -> {out[k][:16]}...")
                rc = 1
        print("state_hash: " + ("IDENTICAL" if rc == 0 else "DIFFERENCES FOUND"))
    if args.save:
        out["toolchain"] = toolchain()
        with open(args.save, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
