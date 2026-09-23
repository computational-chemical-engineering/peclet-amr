#!/usr/bin/env python3
"""ROADMAP B5 — numpy prototype of the seam reconstruction of docs/amr_cf_convective.md.

The convective operator alone (as tests/study/amr_cf_advection.py --flux: exact advecting velocity
at the sub-face centroid, phi = u_x of the t = 0 Taylor-Green field, against div(u u_x) exact),
with the face value reconstructed by the level-aware upwind rule of the design note:

    case 1  2:1 sub-face, upwind = the COARSE cell C:
            phiFace = SOU(phi_C, phi_CC) + [s(C,F) - phi_C]
            s(C,F)  = C's value sampled at the sub-face's tangential position — the two-sided
                      quadratic tangential Taylor of cf_scheme.hpp::cfAppendStencil (a finer
                      tangential neighbour = its 2^3-child volume mean, a coarser one = the
                      leaf at 1.5 H); the correction is applied ONCE, to the extrapolated value.
    case 2  regular fine face whose upstream probe is the coarse cell C (flow C -> F -> F2):
            phiFace = phi_F + (d1/d_uu) (phi_F - s(C,F)),   d1 = h/2, d_uu = 1.5 h  (weight 1/3)
    case 3  regular coarse face whose upstream probe is a refined block (flow F -> C -> C2):
            phiFace = phi_C + (d1/d_uu) (phi_C - mean of the 4 face-layer fine cells),
            d1 = H/2, d_uu = 0.75 H  (weight 2/3)

`slope`  central | quad (central + the second-derivative term, = cfAppendStencil) | minmod
`probe`  block (2^3-child volume mean at H) | layer (4 face-adjacent children at 0.75 H) for a
         finer TANGENTIAL neighbour;  `c3probe` the same choice for case 3's upstream block.

The divergence error is split by cell class: coarse / fine cells owning a 2:1 face, the layer
behind them (cells owning a case-2/3 face but no 2:1 face), and the true bulk; `bulk(old)` is
the instrument's original "everything not at the interface", which the layer pollutes.

Usage:  PYTHONPATH=<build> python tests/study/amr_cf_seam_proto.py 16 32 64
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from amr_cf_advection import _mesh, tg  # noqa: E402


def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def run(N, cases=(1, 2, 3), probe="block", slope="quad", c3probe="layer", arm="G"):
    o, c, lev, wid, lo, sz, owner, top = _mesh(arm, N)
    n = o.num_leaves
    start, nbr, axis, dr = top["start"], top["nbr"], top["axis"], top["dir"]
    ra, upI, upJ = top["raw_area"], top["upup_i"], top["upup_j"]
    ocell = np.repeat(np.arange(n), np.diff(start))
    phi = tg(c, N)[:, 0]
    S = len(nbr)
    sl = np.arange(S)
    fine = np.where(lev[nbr] < lev[ocell], nbr, ocell)
    sgn = np.where(fine == ocell, 1.0, -1.0)
    fc = c[fine].copy()
    fc[sl, axis] += sgn * dr * 0.5 * wid[fine]
    velOut = dr * tg(fc, N)[sl, axis]
    up = np.where(velOut > 0, ocell, nbr)
    upup = np.where(velOut > 0, upI, upJ)
    upstream = np.where(velOut > 0, -dr, dr)  # the side of `up` the flow comes from
    cf = lev[ocell] != lev[nbr]
    upC = cf & (lev[up] > lev[fine])
    reg = ~cf
    case2 = reg & (lev[upup] > lev[up])
    case3 = reg & (lev[upup] < lev[up])

    def look(p):
        p = p % N
        return owner[p[:, 0], p[:, 1], p[:, 2]]

    def block_mean(corner, size, layer_t=None, layer_pos=None):
        acc = np.zeros(len(corner))
        cnt = 0
        for off in itertools.product(range(int(size)), repeat=3):
            off = np.array(off)
            if layer_t is not None and off[layer_t] != 0:
                continue
            p = corner + off[None, :]
            if layer_t is not None:
                p = p.copy()
                p[:, layer_t] = layer_pos
            acc += phi[look(p)]
            cnt += 1
        return acc / cnt

    def side_probe(C, t, s, which):
        Hc = sz[C].astype(float)
        p = lo[C].copy()
        p[:, t] = np.where(s > 0, p[:, t] + sz[C], p[:, t] - 1)
        o1 = look(p)
        v = phi[o1].astype(float)
        d = Hc.copy()
        coarser = lev[o1] > lev[C]
        finer = lev[o1] < lev[C]
        d[coarser] = 1.5 * Hc[coarser]
        if finer.any():
            idx = np.where(finer)[0]
            size = sz[C][idx]
            assert (size == size[0]).all()
            corner = lo[C][idx].copy()
            corner[:, t] = np.where(s > 0, corner[:, t] + size, corner[:, t] - size)
            if which == "block":
                v[idx] = block_mean(corner, size[0])
                d[idx] = Hc[idx]
            else:
                layer_pos = np.where(s > 0, lo[C][idx][:, t] + size, lo[C][idx][:, t] - 1)
                v[idx] = block_mean(corner, size[0], t, layer_pos)
                d[idx] = 0.75 * Hc[idx]
        return v, d

    def sampled(C, delta):
        """s(C, .): phi_C sampled at tangential offset delta (rows x 3, zero along the axis)."""
        v = phi[C].astype(float).copy()
        for t in range(3):
            m = delta[:, t] != 0
            if not m.any():
                continue
            vp, dp = side_probe(C[m], t, +1, probe)
            vm, dm = side_probe(C[m], t, -1, probe)
            dl = delta[m, t]
            if slope == "minmod":
                v[m] += dl * minmod((vp - phi[C[m]]) / dp, (phi[C[m]] - vm) / dm)
            else:
                v[m] += dl * (vp - vm) / (dp + dm)
                if slope == "quad":
                    v[m] += 0.5 * dl * dl * (vp - 2 * phi[C[m]] + vm) / (0.5 * (dp + dm)) ** 2
        return v

    phiUp = phi[up].astype(float)
    phiUpUp = phi[upup].astype(float)
    corr = np.zeros(S)
    w = np.full(S, 0.5)  # d1 / d_uu
    if 1 in cases:
        m = upC
        Cc = up[m]
        delta = np.zeros((m.sum(), 3))
        for t in range(3):
            tt = axis[m] != t
            delta[tt, t] = fc[m][tt, t] - c[Cc][tt, t]
        corr[m] = sampled(Cc, delta) - phi[Cc]
    if 2 in cases:
        m = case2
        Cc = upup[m]
        delta = np.zeros((m.sum(), 3))
        for t in range(3):
            tt = axis[m] != t
            dd = c[up[m]][tt, t] - c[Cc][tt, t]
            delta[tt, t] = (dd + N / 2) % N - N / 2
        phiUpUp[m] = sampled(Cc, delta)
        duu = np.abs(c[up[m], axis[m]] - c[Cc, axis[m]])
        duu = np.minimum(duu, N - duu)
        w[m] = 0.5 * wid[up[m]] / duu
    if 3 in cases:
        idx = np.where(case3)[0]
        Cc = up[idx]
        v = np.zeros(len(idx))
        d = np.zeros(len(idx))
        for a in range(3):
            sel = axis[idx] == a
            if not sel.any():
                continue
            Ca = Cc[sel]
            s = upstream[idx][sel]
            sza = sz[Ca]
            corner = lo[Ca].copy()
            corner[:, a] = np.where(s > 0, corner[:, a] + sza, corner[:, a] - sza)
            if c3probe == "block":
                v[sel] = block_mean(corner, sza[0])
                d[sel] = sza
            else:
                layer_pos = np.where(s > 0, lo[Ca][:, a] + sza, lo[Ca][:, a] - 1)
                v[sel] = block_mean(corner, sza[0], a, layer_pos)
                d[sel] = 0.75 * sza
        phiUpUp[idx] = v
        w[idx] = 0.5 * wid[Cc] / d
    phiFace = phiUp + w * (phiUp - phiUpUp) + corr

    fe = np.abs(phiFace - tg(fc, N)[:, 0])
    k = 2.0 * np.pi / N
    conv = np.bincount(ocell, weights=ra * velOut * phiFace, minlength=n) / wid ** 3
    r = conv - 0.5 * k * np.sin(2 * k * c[:, 0])
    vol = wid ** 3
    touch = np.zeros(n, bool)
    touch[ocell[cf]] = True
    coarse = lev > lev.min()
    layer2 = np.zeros(n, bool)
    layer2[ocell[case2 | case3]] = True
    layer2 &= ~touch
    bulk = ~touch & ~layer2

    def vr(m):
        return float(np.sqrt((vol[m] * r[m] ** 2).sum() / vol[m].sum())) if m.any() else 0.0

    def fr(m):
        return float(np.sqrt((fe[m] ** 2).mean())) if m.any() else 0.0

    return dict(N=N, if_all=vr(touch), if_C=vr(touch & coarse), if_F=vr(touch & ~coarse),
                layer2=vr(layer2), bulk=vr(bulk), bulk_old=vr(~touch),
                f_upC=fr(upC), f_upF=fr(cf & ~upC), f_c2=fr(case2), f_c3=fr(case3),
                f_reg=fr(reg & ~case2 & ~case3),
                n_pairs=int(cf.sum()) // 2, n_c2=int(case2.sum()), n_c3=int(case3.sum()))


VARIANTS = [
    ("none", dict(cases=())),
    ("design (quad, block tan., layer c3)", dict()),
    ("  case 1 only", dict(cases=(1,))),
    ("  cases 1+2", dict(cases=(1, 2))),
    ("  cases 2+3 (no sub-face fix)", dict(cases=(2, 3))),
    ("  linear tangential (no D_tt)", dict(slope="central")),
    ("  minmod tangential", dict(slope="minmod")),
    ("  block probe for case 3", dict(c3probe="block")),
    ("  layer probe for tangential", dict(probe="layer")),
]


def main():
    Ns = [int(x) for x in sys.argv[1:]] or [16, 32, 64]
    hdr = (f"{'variant':>38} {'div if':>9} {'if C':>9} {'if F':>9} {'layer2':>9} {'bulk':>9} "
           f"{'bulk(old)':>9} | {'f upC':>9} {'f upF':>9} {'f c2':>9} {'f c3':>9} {'f reg':>9}")
    for N in Ns:
        d0 = run(N, cases=())
        print(f"\n# N={N}: {d0['n_pairs']} (C,F) sub-face pairs, {d0['n_c2']} case-2 slots, "
              f"{d0['n_c3']} case-3 slots (flow-direction dependent)")
        print(hdr)
        for name, kw in VARIANTS:
            d = run(N, **kw)
            print(f"{name:>38} {d['if_all']:9.3e} {d['if_C']:9.3e} {d['if_F']:9.3e} "
                  f"{d['layer2']:9.3e} {d['bulk']:9.3e} {d['bulk_old']:9.3e} | {d['f_upC']:9.3e} "
                  f"{d['f_upF']:9.3e} {d['f_c2']:9.3e} {d['f_c3']:9.3e} {d['f_reg']:9.3e}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
