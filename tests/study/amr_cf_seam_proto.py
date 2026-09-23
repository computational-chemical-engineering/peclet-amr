#!/usr/bin/env python3
"""ROADMAP B5 -- the design note's ablation table (Appendix A of docs/amr_cf_convective.md).

The level-aware upwind reconstruction itself (docs/amr_cf_convective.md §5, cases 1/2/3) lives in
`amr_cf_advection.py::seam_probe` -- it is `--flux`'s mode `seam`.  This script is the historical
record: it drives that same function across the design's ablations (which cases are on, which
tangential slope, which probe for a finer neighbour) to reproduce the table that justified the
design (WO0, docs/amr_cf_convective.md §8).  It never defines the reconstruction itself -- only
`amr_cf_advection` does, and this script imports from it (never the reverse).

Usage:  PYTHONPATH=<build> python tests/study/amr_cf_seam_proto.py 16 32 64
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from amr_cf_advection import seam_probe as run  # noqa: E402

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
