#!/usr/bin/env python3
"""The seam-reconstruction tables, cell by cell against an independent numpy rule (ROADMAP B5).

`docs/amr_cf_convective.md` §5.2 prebuilds, in `setSolid`, everything the convective flux needs at
a 2:1 coarse/fine seam: a per-face-slot DESCRIPTOR and a CSR of RECORDS (a tangential sample of the
coarse cell at the fine sub-face's column, and the mean of a face's four fine cells).  This is WO1's
acceptance:

  (b) on the graded G mesh at N = 32 -- Octree(cells=N^3, lmax=1) refined to the sphere of radius
      N/4 and balanced, no solid -- every record the C++ builder produced equals the rule
      reimplemented here from the octree geometry alone: 2688 sample records (one per 2:1 sub-face
      pair), 672 face-layer records, entry weights to 1e-15, and the layer records' cells equal to
      the row's own sub-face neighbours in `forEachFaceFull` order.

  (c) the SAME mesh through `DistributedOctree` at np = 1, 2, 4 produces the same descriptors: for
      every OWNED row, slot for slot, the record entries mapped from local/ghost slots to world
      cell centres are identical to the single-rank build's.  That is the statement that makes
      np = 1 bitwise against single-rank meaningful -- a withheld record at a block seam would show
      up here as a missing entry long before it showed up as a moved digit.

Run: PYTHONPATH=<build> python python/test_amr_seam_records.py      (also under mpirun -np 2 / 4)
"""
import os
import sys

import numpy as np
from mpi4py import MPI

from peclet import amr

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
fail = 0

_np_env = os.environ.get("PECLET_AMR_TEST_NP")
if _np_env is not None and int(_np_env) != size:
    sys.exit(f"[rank {rank}] launched with PECLET_AMR_TEST_NP={_np_env} but "
             f"MPI.COMM_WORLD.size={size}: the launcher and mpi4py are different MPIs")

N = 32                      # the G mesh of docs/amr_cf_convective.md Appendix A
SPHERE_R = N / 4.0
N_SAMPLE_RECORDS = 2688     # Appendix A: (C, F) sub-face pairs at N = 32
N_LAYER_RECORDS = 672       # Appendix A: case-3 slots at N = 32


def check(cond, msg):
    global fail
    if not cond:
        fail += 1
        sys.stderr.write(f"[rank {rank}] FAIL: {msg}\n")


# ================================================================================================
# The rule, in numpy, from the octree geometry alone (no C++ table is consulted).
# ================================================================================================
class Mesh:
    """Finest-grid ownership map + the three geometric queries the rule needs."""

    def __init__(self, octree):
        self.n = octree.num_leaves
        self.h = np.asarray(octree.spacing, dtype=float)
        self.ncell = np.asarray(octree.cells, dtype=np.int64)
        c, w = octree.centers(), octree.sizes(0)
        self.level = octree.levels().astype(np.int64)
        self.size = np.rint(w / self.h[0]).astype(np.int64)          # width in finest cells
        self.lo = np.rint((c - np.asarray(octree.origin)) / self.h
                          - 0.5 * self.size[:, None]).astype(np.int64)
        self.owner = np.full(tuple(self.ncell), -1, dtype=np.int64)
        for i in range(self.n):
            a, s = self.lo[i], self.size[i]
            self.owner[a[0]:a[0] + s, a[1]:a[1] + s, a[2]:a[2] + s] = i

    def probe(self, q):
        """probeSlot: the leaf covering finest-grid coordinate q (periodic)."""
        x, y, z = (int(q[0]) % self.ncell[0], int(q[1]) % self.ncell[1], int(q[2]) % self.ncell[2])
        return int(self.owner[x, y, z])

    def neighbor(self, i, axis, d):
        """periodicNeighbor(i, axis, dir)."""
        p = list(self.lo[i])
        p[axis] += int(self.size[i]) if d > 0 else -1
        return self.probe(p)

    def face_layer(self, C, axis, d):
        """The four sub-face neighbours across C's face (axis, dir), in forEachFaceFull's order."""
        sc = int(self.size[C])
        sh = sc >> 1
        base = list(self.lo[C])
        base[axis] = self.lo[C][axis] + sc if d > 0 else self.lo[C][axis] - 1
        out = []
        for k in range(4):
            q, bit = list(base), 0
            for t in range(3):
                if t == axis:
                    continue
                q[t] = self.lo[C][t] + (sh if (k >> bit) & 1 else 0)
                bit += 1
            out.append(self.probe(q))
        return out

    def sub_faces(self, i, axis, d):
        """forEachFaceFull's neighbour list for ONE face of i: 1 entry, or 4 when the far side is
        refined."""
        j = self.neighbor(i, axis, d)
        if self.level[j] >= self.level[i]:
            return [j]
        return self.face_layer(i, axis, d)


def sample_stencil(m, C, F, axis):
    """`cfAppendStencil` (quadratic) plus the raw coarse value: the sample record's entries.

    No solid anywhere on this mesh, so `fluidOk` is always true and every face openness is 1 --
    the only fallback that can fire is a COARSER tangential neighbour, which drops that side.
    """
    ent = {C: 1.0}

    def add(cell, w):
        ent[cell] = ent.get(cell, 0.0) + w

    Lc = int(m.level[C])
    sc, sf = float(1 << Lc), float(1 << int(m.level[F]))
    for tt in range(3):
        if tt == axis:
            continue
        H = m.h[tt] * (1 << Lc)
        dt = ((m.lo[F][tt] + 0.5 * sf) - (m.lo[C][tt] + 0.5 * sc)) * m.h[tt]

        def tangential(d):
            nb = m.neighbor(C, tt, d)
            if nb < 0:
                return None
            if int(m.level[nb]) == Lc:
                return [(nb, 1.0)]
            if int(m.level[nb]) + 1 != Lc:
                return None                      # coarser neighbour: this side is dropped
            base = list(m.lo[C])
            base[tt] += (1 << Lc) if d > 0 else -(1 << Lc)
            sh = (1 << Lc) >> 1
            out = []
            for oct2 in range(8):
                q = [base[a] + (sh if (oct2 >> a) & 1 else 0) for a in range(3)]
                ch = m.probe(q)
                if ch < 0 or int(m.level[ch]) + 1 != Lc:
                    return None
                out.append((ch, 0.125))
            return out

        sp, sm = tangential(+1), tangential(-1)
        if sp is not None and sm is not None:
            add(C, -dt * dt / (H * H))
            for c, w in sp:
                add(c, (dt / (2.0 * H) + 0.5 * dt * dt / (H * H)) * w)
            for c, w in sm:
                add(c, (-dt / (2.0 * H) + 0.5 * dt * dt / (H * H)) * w)
        elif (sp is None) != (sm is None):
            s1 = sp if sp is not None else sm
            sgn = 1.0 if sp is not None else -1.0
            add(C, -sgn * dt / H)
            for c, w in s1:
                add(c, sgn * (dt / H) * w)
    return ent


def expected_descriptors(m):
    """The rule's descriptor for every face slot, in forEachFaceFull / FaceGeom slot order.

    Each entry is (samp_i, samp_j, uu_i, uu_j, d1_i, d1_j) where a record is either None or the
    pair (entries-as-{cell: weight}, recDist).
    """
    def sample(C, F, axis):
        if C < 0 or F < 0:
            return None
        return (sample_stencil(m, C, F, axis), 1.5 * m.h[axis] * (1 << int(m.level[F])))

    def layer(C, axis, d):
        if C < 0 or int(m.level[C]) == 0:
            return None
        cells = m.face_layer(C, axis, d)
        if any(c < 0 or int(m.level[c]) + 1 != int(m.level[C]) for c in cells):
            return None
        return ({c: 0.25 for c in cells}, 0.75 * m.h[axis] * (1 << int(m.level[C])))

    out = []
    for i in range(m.n):
        for axis in range(3):
            for d in (-1, +1):
                for j in m.sub_faces(i, axis, d):
                    Li, Lj = int(m.level[i]), int(m.level[j])
                    rsI = sample(i, j, axis) if Lj < Li else None
                    rsJ = sample(j, i, axis) if Lj > Li else None
                    uI = m.neighbor(i, axis, -d)
                    ruI = (sample(uI, i, axis) if int(m.level[uI]) > Li else
                           layer(i, axis, -d) if int(m.level[uI]) < Li else None)
                    uJ = m.neighbor(j, axis, d)
                    ruJ = (sample(uJ, j, axis) if int(m.level[uJ]) > Lj else
                           layer(j, axis, d) if int(m.level[uJ]) < Lj else None)
                    out.append((rsI, rsJ, ruI, ruJ,
                                0.5 * m.h[axis] * (1 << Li), 0.5 * m.h[axis] * (1 << Lj)))
    return out


# ================================================================================================
# The C++ tables, read back through Flow.diagnostics.face_topology().
# ================================================================================================
def cpp_record(top, r):
    """Record `r` as ({cell slot: weight}, recDist), or None for r < 0."""
    if r < 0:
        return None
    a, b = int(top["rec_start"][r]), int(top["rec_start"][r + 1])
    ent = {}
    for e in range(a, b):
        c = int(top["rec_cell"][e])
        ent[c] = ent.get(c, 0.0) + float(top["rec_w"][e])
    return (ent, float(top["rec_dist"][r]))


def cpp_descriptors(top):
    """The C++ descriptor for every face slot, in the same order as expected_descriptors."""
    out = []
    for k, s in enumerate(top["seam"]):
        s = int(s)
        if s < 0:
            out.append(None)
            continue
        out.append((cpp_record(top, int(top["samp_i"][s])),
                    cpp_record(top, int(top["samp_j"][s])),
                    cpp_record(top, int(top["uu_rec_i"][s])),
                    cpp_record(top, int(top["uu_rec_j"][s])),
                    float(top["d1_i"][s]), float(top["d1_j"][s])))
    return out


def same_record(got, want, key, relabel=None):
    """Entry sets, weights (1e-15) and the probe distance of one record."""
    if (got is None) != (want is None):
        check(False, f"{key}: record present={got is not None}, expected={want is not None}")
        return
    if got is None:
        return
    ge, gd = got
    we, wd = want
    if relabel is not None:
        ge = {relabel(c): w for c, w in ge.items()}
    if set(ge) != set(we):
        check(False, f"{key}: entry cells {sorted(ge)} != {sorted(we)}")
        return
    bad = max(abs(ge[c] - we[c]) for c in we) if we else 0.0
    check(bad <= 1e-15, f"{key}: weight mismatch {bad:.3e} > 1e-15")
    check(abs(gd - wd) <= 1e-15 * max(1.0, abs(wd)), f"{key}: recDist {gd!r} != {wd!r}")


def build_serial():
    o = amr.Octree(cells=[N] * 3, lmax=1, origin=[0.0] * 3, spacing=1.0)
    o.refine_to_sphere([N / 2.0] * 3, SPHERE_R)
    o.balance()
    f = amr.Flow(o, density=1.0, viscosity=1.0, dt=0.5)
    f.set_advection(True)
    f.set_cf_scheme(1)          # quadratic -- the scheme the seam tables ride
    f.set_solid(lambda x, y, z: 1e3)     # no solid: every cell regular fluid
    return o, f


# ---- (b) the rule, cell by cell, on the serial build ------------------------------------------
if rank == 0:
    o, f = build_serial()
    top = f.diagnostics.face_topology()
    check(f.diagnostics.num_seam_sample_records == N_SAMPLE_RECORDS,
          f"sample records {f.diagnostics.num_seam_sample_records} != {N_SAMPLE_RECORDS}")
    check(f.diagnostics.num_seam_layer_records == N_LAYER_RECORDS,
          f"layer records {f.diagnostics.num_seam_layer_records} != {N_LAYER_RECORDS}")
    check(len(top["rec_start"]) - 1 == N_SAMPLE_RECORDS + N_LAYER_RECORDS,
          "record CSR length disagrees with the census")

    m = Mesh(o)
    want = expected_descriptors(m)
    got = cpp_descriptors(top)
    check(len(got) == len(want), f"slot count {len(got)} != {len(want)}")
    n_live = 0
    for k, (g, w) in enumerate(zip(got, want)):
        live = any(r is not None for r in w[:4])
        if not live:
            check(g is None, f"slot {k}: a descriptor where the rule wants a plain slot")
            continue
        n_live += 1
        if g is None:
            check(False, f"slot {k}: no descriptor where the rule wants one")
            continue
        for name, gi, wi in zip(("samp_i", "samp_j", "uu_i", "uu_j"), g[:4], w[:4]):
            same_record(gi, wi, f"slot {k} {name}")
        check(abs(g[4] - w[4]) <= 1e-15 and abs(g[5] - w[5]) <= 1e-15,
              f"slot {k}: d1 ({g[4]}, {g[5]}) != ({w[4]}, {w[5]})")
    check(n_live == int((top["seam"] >= 0).sum()),
          f"live-slot count {int((top['seam'] >= 0).sum())} != rule's {n_live}")
    # Every sample record's weights sum to 1 (it substitutes a VALUE, not a delta).
    for r in range(len(top["rec_start"]) - 1):
        a, b = int(top["rec_start"][r]), int(top["rec_start"][r + 1])
        check(abs(float(top["rec_w"][a:b].sum()) - 1.0) <= 1e-14,
              f"record {r}: weights sum to {float(top['rec_w'][a:b].sum())!r}, not 1")
    sys.stderr.write(f"[rank 0] seam tables: {n_live} seam slots, "
                     f"{N_SAMPLE_RECORDS} sample + {N_LAYER_RECORDS} layer records checked\n")


# ---- (c) decomposition independence ------------------------------------------------------------
def owned_descriptors(flow, num_leaves):
    """{(centre of the owning leaf, axis, dir, ordinal): descriptor with cells as world centres}.

    Keyed and valued in GLOBAL coordinates, so the union over ranks is comparable between any two
    decompositions. Ghost slots are named by `cell_center`, which covers the whole extended array.
    """
    top = flow.diagnostics.face_topology()
    cen = np.round(top["cell_center"], 9)
    start = top["start"]
    desc = cpp_descriptors(top)
    out = {}
    for i in range(num_leaves):
        seen = {}
        for k in range(int(start[i]), int(start[i + 1])):
            a, d = int(top["axis"][k]), int(top["dir"][k])
            ordinal = seen.get((a, d), 0)
            seen[(a, d)] = ordinal + 1
            key = (tuple(cen[i]), a, d, ordinal)
            g = desc[k]
            if g is None:
                out[key] = None
                continue
            out[key] = tuple(
                None if r is None else
                (tuple(sorted((tuple(cen[c]), round(w, 15)) for c, w in r[0].items())),
                 round(r[1], 12))
                for r in g[:4]) + (round(g[4], 12), round(g[5], 12))
    return out


dist = amr.DistributedOctree(cells=[N] * 3, lmax=1, origin=[0.0] * 3,
                             extent=[float(N)] * 3, periodic=[True, True, True])
dist.refine_to_sphere(center=[N / 2.0] * 3, radius=SPHERE_R, target_level=0, band=1.0, balance=True)
fd = amr.Flow(dist, density=1.0, viscosity=1.0, dt=0.5)
fd.set_advection(True)
fd.set_cf_scheme(1)
fd.set_solid(lambda x, y, z: 1e3)
local = owned_descriptors(fd, dist.num_leaves)
parts = comm.gather(local, root=0)
if rank == 0:
    merged = {}
    for p in parts:
        for k, v in p.items():
            check(k not in merged, f"row/slot {k} owned by two ranks")
            merged[k] = v
    ref = owned_descriptors(f, o.num_leaves)
    check(set(merged) == set(ref),
          f"owned slot sets differ: {len(set(merged) ^ set(ref))} slots only on one side")
    ndiff = sum(1 for k in ref if k in merged and merged[k] != ref[k])
    check(ndiff == 0, f"{ndiff} owned slots have a different descriptor at np={size}")
    sys.stderr.write(f"[rank 0] np={size}: {len(merged)} owned slots match the single-rank build\n")

fail = comm.allreduce(fail, op=MPI.SUM)
if rank == 0 and fail == 0:
    print(f"test_amr_seam_records: OK (np={size})")
sys.exit(1 if fail else 0)
