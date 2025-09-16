#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bin_table.py
Writes a live ASCII bin table to <logs>/bin_table.txt, matching the format you provided.
Called for every processed frame by curator_tag.py.
"""
from pathlib import Path
from typing import Dict, Any
import io, os, warnings

BAR_WIDTH = 16

def _bar(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(BAR_WIDTH * pct / 100.0))
    return "█"*filled + " "*(BAR_WIDTH - filled)

def _line_hdr(name: str, pct: float, count: int, fails: int=0, dupe: int=0, fps: int=None, eta: str=None) -> str:
    # TAG row shows FPS/ETA when available; other headers omit FPS/ETA
    bar = _bar(pct)
    left = f"{name:<11}|{bar}| {pct:6.1f}% {count:11d}    FAIL {fails:7d}    DUPE {dupe:5d}"
    if name == "TAG" and fps is not None and eta is not None:
        left += f"    {fps:3d} FPS    ETA {eta}"
    return left

def _line_cat(label: str, pct: float, count: int) -> str:
    bar = _bar(pct)
    return f"{label:<12}|{bar}| {pct:6.1f}% {count:7d}{'':>35}0"

def _sep(ch: str="=") -> str:
    return ch*112

def _write_help():
    out = Path("/workspace/scripts") / (Path(__file__).name + ".txt")
    out.write_text(
        "This file renders bin_table.txt live with sections: TAG, GAZE, EYES, MOUTH, SMILE, YAW, PITCH, TEMP, EXPOSURE.\n"
        "Update by calling update_bin_table(log_dir, state).\n", encoding="utf-8"
    )
_write_help()

def update_bin_table(log_dir: Path|str, s: Dict[str, Any]) -> None:
    warnings.filterwarnings("ignore")
    log_dir = Path(log_dir)
    out = log_dir / "bin_table.txt"

    total = int(s.get("total", 0)) or 1
    processed = int(s.get("processed", 0))
    fails = int(s.get("fails", 0))
    dupe = int(s.get("dupe", 0))
    fps = s.get("fps", None)
    eta = s.get("eta", None)

    # Section dicts: each is {label: count}
    gaze = dict(s.get("gaze", {}))
    eyes = dict(s.get("eyes", {}))
    mouth = dict(s.get("mouth", {}))
    smile = dict(s.get("smile", {}))
    yaw = dict(s.get("yaw", {}))
    pitch = dict(s.get("pitch", {}))
    temp = dict(s.get("temp", {}))
    expo = dict(s.get("expo", {}))

    def pct_of_total(n): return 100.0 * (n / total)
    def section(name, items: Dict[str,int], include_fails=False):
        buf = io.StringIO()
        count = sum(items.values())
        pct = pct_of_total(count)
        buf.write(_line_hdr(name, pct, count, fails if include_fails else 0, dupe) + "\n")
        buf.write(_sep("-") + "\n")
        for k,v in items.items():
            buf.write(_line_cat(k.upper(), pct_of_total(v), v) + "\n")
        return buf.getvalue()

    # Build full table
    buf = io.StringIO()
    buf.write(_sep("=") + "\n")
    # TAG (header/progress of whole job)
    buf.write(_line_hdr("TAG", 100.0 * processed/total, processed, fails, dupe, fps=fps, eta=eta) + "\n")
    buf.write(_sep("=") + "\n")

    buf.write(section("GAZE", gaze, include_fails=False))
    buf.write(_sep("=") + "\n")
    buf.write(section("EYES", eyes, include_fails=False))
    buf.write(_sep("=") + "\n")
    buf.write(section("MOUTH", mouth, include_fails=False))
    buf.write(_sep("=") + "\n")
    buf.write(section("SMILE", smile, include_fails=False))
    buf.write(_sep("=") + "\n")
    buf.write(section("YAW", yaw, include_fails=False))
    buf.write(_sep("=") + "\n")
    buf.write(section("PITCH", pitch, include_fails=False))
    buf.write(_sep("=") + "\n")
    buf.write(section("TEMP", temp, include_fails=False))
    buf.write(_sep("=") + "\n")
    buf.write(section("EXPOSURE", expo, include_fails=False))
    buf.write(_sep("=") + "\n")

    tmp = out.with_suffix(".txt.tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    os.replace(tmp, out)
