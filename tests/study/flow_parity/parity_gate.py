#!/usr/bin/env python3
"""Uniform-grid parity gate: `peclet.amr.Flow` (lmax = 0) vs `peclet.flow.SolverColocated`.

At lmax = 0 the octree IS the structured brick flow runs on -- same extent, same spacing, same cell
centres -- so the two engines must produce the same fields to solver tolerance. This script runs
each case twice, once per engine in its OWN subprocess (the two modules cannot share an interpreter:
each finalizes Kokkos under the other's Views), and compares cell by cell over the shared fluid
cells. See docs/amr_flow_uniform_parity.md for the measured ladder and what each case pins down.

The two builds are located by PATH, never by a numerics-changing switch:

    PECLET_AMR_PYTHONPATH       the amr build tree   (required)
    PECLET_AMR_FLOW_PYTHONPATH  the flow build tree  (absent -> SKIP with 77)

Exit codes follow the repo's ctest protocol (tests/test_util.hpp): 0 pass, 1 fail, 77 skipped.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = 77

# case file -> (velocity tolerance, pressure tolerance, what the case pins down).
# Each tolerance is the MEASURED difference with roughly a decade of margin, not an aspiration --
# docs/amr_flow_uniform_parity.md §2 carries the measured column. A gate that tightens is a result
# worth recording; a gate that has to loosen is a regression worth explaining.
GATES = [
    ("tg_stokes.json", 1e-12, None,
     "the shared core: 7-point operator, divergence, ABC gradient, BE momentum, periodic wrap"),
    ("tg_advect_one_step.json", 1e-8, 1e-8,
     "the advection stencil itself (one step, before either engine has a projected face field)"),
    ("tg_advect_matched.json", 1e-8, 1e-8,
     "the full NS step with the advecting velocity matched (amr ablated to flow's choice)"),
    ("channel_cut.json", 1e-8, None,
     "the cut-cell Robust-Scaled no-slip momentum overlay at non-grid-aligned walls"),
    ("sphere_ghost.json", 1e-5, 1e-3,
     "the cut-cell GHOST projection (the production default in both engines)"),
    ("sphere_ghost_advect.json", 1e-3, 3e-3,
     "cut cells UNDER ADVECTION -- records a known open difference (uniform_parity.md P5), so the "
     "tolerance is loose on purpose: it is a tripwire, not a parity claim"),
]


def run(engine, case_path, out, pythonpath):
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OMP_PROC_BIND", "false")
    r = subprocess.run([sys.executable, os.path.join(HERE, f"run_{engine}.py"), case_path, out],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {engine} runner failed (exit {r.returncode}):\n{r.stdout}\n{r.stderr}")
        return False
    return True


def _masked(a, b, field, mask):
    A, B = a[field].copy(), b[field].copy()
    if field == "p":  # defined up to a constant in a periodic box
        A = A - A[mask].mean()
        B = B - B[mask].mean()
    return np.where(mask, A, 0.0), np.where(mask, B, 0.0)


def rel_diffs(a, b, mask):
    """(velocity, pressure) relative L2 differences over the shared fluid cells.

    The three velocity components share ONE denominator -- the magnitude of the velocity field --
    so a component that is legitimately zero in the case (w in a 2-D Taylor-Green, v in a plane
    channel) contributes its absolute error and cannot divide round-off by round-off. Pressure has
    its own, and is reported as None when it is numerically absent (Stokes Taylor-Green, plane
    Poiseuille: |p| below 1e-10 of the velocity scale), where a ratio would again be noise/noise.
    """
    num = den = 0.0
    for f in ("u", "v", "w"):
        A, B = _masked(a, b, f, mask)
        num += np.sum((A - B) ** 2)
        den += np.sum(A ** 2)
    vel_scale = np.sqrt(den)
    vel = np.sqrt(num) / (vel_scale if vel_scale > 0 else 1.0)

    A, B = _masked(a, b, "p", mask)
    p_scale = np.sqrt(np.sum(A ** 2))
    pres = None if p_scale <= 1e-10 * vel_scale else float(np.sqrt(np.sum((A - B) ** 2)) / p_scale)
    return float(vel), pres


def main():
    amr_pp = os.environ.get("PECLET_AMR_PYTHONPATH", "")
    flow_pp = os.environ.get("PECLET_AMR_FLOW_PYTHONPATH", "")
    if not amr_pp:
        print("SKIP: PECLET_AMR_PYTHONPATH not set")
        return SKIP
    if not flow_pp or not os.path.isdir(flow_pp):
        print("SKIP: PECLET_AMR_FLOW_PYTHONPATH unset or missing -- this gate needs a peclet.flow "
              "build tree beside the amr one (configure with -DPECLET_AMR_FLOW_PYTHONPATH=...)")
        return SKIP

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for case, vtol, ptol, what in GATES:
            path = os.path.join(HERE, "cases", case)
            name = case[:-5]
            print(f"[{name}] {what}")
            fo = os.path.join(tmp, f"flow_{name}.npz")
            ao = os.path.join(tmp, f"amr_{name}.npz")
            if not (run("flow", path, fo, flow_pp) and run("amr", path, ao, amr_pp)):
                failures.append((name, "runner failed"))
                continue
            a, b = np.load(fo), np.load(ao)
            for ax in ("cx", "cy", "cz"):
                if not np.allclose(a[ax], b[ax]):
                    failures.append((name, f"{ax} grids differ"))
                    continue
            mask = (a["fluid"] > 0.5) & (b["fluid"] > 0.5)
            vel, pres = rel_diffs(a, b, mask)
            bad = []
            if vel > vtol:
                bad.append(f"velocity {vel:.3e} > {vtol:.0e}")
            if pres is not None and ptol is not None and pres > ptol:
                bad.append(f"pressure {pres:.3e} > {ptol:.0e}")
            ptxt = "n/a" if pres is None else f"{pres:.3e} (tol {ptol:.0e})" if ptol else \
                f"{pres:.3e} (not gated)"
            print(f"  {'FAIL' if bad else 'OK  '} velocity {vel:.3e} (tol {vtol:.0e}), "
                  f"pressure {ptxt}; iters flow {list(a['pres_iters'][:4])} "
                  f"amr {list(b['pres_iters'][:4])}")
            if bad:
                failures.append((name, "; ".join(bad)))

    if failures:
        print("\nFAILED:")
        for n, why in failures:
            print(f"  {n}: {why}")
        return 1
    print(f"\nall {len(GATES)} uniform-grid parity gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
