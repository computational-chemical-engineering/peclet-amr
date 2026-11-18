"""Print one '[deps] <repo>=<include path> @ <commit>' line per sibling header tree the build compiles against.

Read from the build's compile_commands.json (the -I paths of the binding TU), so the line names the tree the
compiler actually saw. The commit makes the codehealth `same_lines` check fail when a shared sibling checkout
(suite/core, suite/morton) moves between the base and the head runs of a gate.

    python tests/regression/deps_lines.py <build>/compile_commands.json
"""
import json
import os
import re
import subprocess
import sys

with open(sys.argv[1]) as fh:
    cc = json.load(fh)
cmd = " ".join(e.get("command", " ".join(e.get("arguments", []))) for e in cc
               if e["file"].endswith("amr_bindings.cpp"))
for repo in ("core", "morton"):
    for inc in sorted(set(re.findall(rf"-I(\S*/{repo}/include)", cmd))):
        path = os.path.realpath(inc)
        sha = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip() or "no-git"
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no", "."],
                               capture_output=True, text=True, check=False).stdout.strip()
        print(f"[deps] {repo}={path} @ {sha}{' +dirty' if dirty else ''}")
