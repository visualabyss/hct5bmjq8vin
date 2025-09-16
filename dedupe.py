#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedupe.py
Wrapper that invokes curator_dedupe.py (which is configured to never move files).
Writes /workspace/scripts/dedupe.py.txt with flags and notes.
"""
import argparse, subprocess, shlex
from pathlib import Path

def _write_help(args):
    out = Path("/workspace/scripts") / (Path(__file__).name + ".txt")
    out.write_text(
        f"Runs curator_dedupe.py without moving files.\n"
        f"aligned={args.aligned}\nlogs={args.logs}\n", encoding="utf-8"
    )

def main():
    ap = argparse.ArgumentParser("dedupe wrapper (no-move)")
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    args = ap.parse_args()
    _write_help(args)
    cmd = (
        f"python3 -u /workspace/scripts/curator_dedupe.py "
        f"--aligned {shlex.quote(str(args.aligned))} "
        f"--logs {shlex.quote(str(args.logs))}"
    )
    import os
    env=dict(os.environ); env['PROGRESS_MUTE']='1'
    subprocess.run(cmd + ' >/dev/null 2>&1', shell=True, env=env)

if __name__ == "__main__":
    main()
