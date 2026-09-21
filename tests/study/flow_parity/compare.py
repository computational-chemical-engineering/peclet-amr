#!/usr/bin/env python3
"""Diff two parity dumps (flow vs amr) cell by cell."""
import sys

import numpy as np

a = np.load(sys.argv[1])  # flow
b = np.load(sys.argv[2])  # amr
label = sys.argv[3] if len(sys.argv) > 3 else ""

for ax in ("cx", "cy", "cz"):
    assert np.allclose(a[ax], b[ax]), f"{ax} grids differ"

print(f"=== {label} ===")
print(f"{'field':>6} {'max|flow|':>11} {'max|amr|':>11} {'max abs diff':>13} {'rel (L2)':>11}")
m = (a["fluid"] > 0.5) & (b["fluid"] > 0.5) if "fluid" in a and "fluid" in b else np.ones_like(a["u"], bool)
print(f"(comparing {m.sum()} shared fluid cells of {m.size})")
for f in ("u", "v", "w", "p"):
    A, B = a[f].copy(), b[f].copy()
    if f == "p":  # pressure is defined up to a constant; compare the fluid-mean-removed fields
        A = A - A[m].mean()
        B = B - B[m].mean()
    A = np.where(m, A, 0.0)
    B = np.where(m, B, 0.0)
    d = np.abs(A - B)
    den = np.sqrt(np.mean(A ** 2)) or 1.0
    print(f"{f:>6} {np.abs(A).max():11.4e} {np.abs(B).max():11.4e} "
          f"{d.max():13.4e} {np.sqrt(np.mean((A-B)**2))/den:11.4e}")

print(f"\n{'':>6} {'flow':>26} {'amr':>26}")
print(f"{'pres':>6} {str(list(a['pres_iters'][:6])):>26} {str(list(b['pres_iters'][:6])):>26}")
if "mom_iters" in b:
    print(f"{'mom':>6} {'(residual-based)':>26} {str(list(b['mom_iters'][:6])):>26}")
print(f"{'ms/st':>6} {np.mean(a['wall'])*1e3:26.1f} {np.mean(b['wall'])*1e3:26.1f}")
print(f"{'div':>6} {float(a['max_open_div']):26.3e} {float(b['div_face']):26.3e}")
