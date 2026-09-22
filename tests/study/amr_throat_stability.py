"""Gate S (docs/amr_cf_flux_gate.md §10): the 12 throat-graded meshes of
tests/study/amr_two_sphere_gap.py (gaps 8,16 x n 1,2,3,4,6,8) marched at cf=1 to a 400-step
horizon -- the meshes whose k~1e12 march motivated the rowRegular row gate.

Reuses the study's own mesh builder and the diverge probe's `probe()` verbatim.

  PYTHONPATH=<build> python -u amr_throat_stability.py [--g 8,16] [--ns 1,2,3,4,6,8] [--steps 400]
"""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from amr_two_sphere_diverge_probe import probe   # noqa: E402

args = sys.argv[1:]
def opt(name, d):
    return args[args.index(name) + 1] if name in args else d

N = 128
gaps = [int(v) for v in opt("--g", "8,16").split(",")]
ns = [float(v) for v in opt("--ns", "1,2,3,4,6,8").split(",")]
steps = int(opt("--steps", "400"))
cfs = [int(v) for v in opt("--cf", "1").split(",")]
print(f"gate S: N={N} gaps={gaps} ns={ns} cf={cfs} horizon={steps}", flush=True)
print(f"{'g':>4} {'n':>4} {'cf':>3} {'outcome':>14} {'k_final':>14} {'secs':>7}", flush=True)
for g in gaps:
    for n in ns:
        for cf in cfs:
            t0 = time.time()
            try:
                at, k = probe(N, g, n, cf, 60.0, True, steps)
            except Exception as e:
                print(f"{g:>4} {n:>4g} {cf:>3} {'EXCEPTION':>14} {type(e).__name__}: {e}",
                      flush=True)
                continue
            out = f"DIVERGED@{at}" if at is not None else "finite"
            print(f"{g:>4} {n:>4g} {cf:>3} {out:>14} {k:>14.6e} {time.time()-t0:>7.1f}",
                  flush=True)
