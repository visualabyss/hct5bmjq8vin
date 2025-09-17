#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ASCII bin-table renderer shared by tag.py and match.py.
- Exact display names/labels match your samples (bin_table.txt / bin_table_match.txt).
- Progress bars; fixed separators; single-space field gaps.
- MATCH view supports DUPE totals and per-row dupes; hidden when dedupe is off.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import io, os

BAR_WIDTH = 16
SEP_LEN   = 112

SECTION_ORDER = [
    "GAZE","EYES","MOUTH","SMILE","EMOTION","YAW","PITCH",
    "IDENTITY","QUALITY","TEMP","EXPOSURE",
]

# ---- helpers -----------------------------------------------------------

def _bar(pct: float) -> str:
    try:
        pct = max(0.0, min(100.0, float(pct)))
    except Exception:
        pct = 0.0
    filled = int(round(BAR_WIDTH * pct / 100.0))
    return ("█" * filled) + (" " * (BAR_WIDTH - filled))


def _sep(ch: str = "=") -> str:
    return ch * SEP_LEN


def _pct(count: int, total: int) -> float:
    total = max(1, int(total))
    return 100.0 * (float(count) / float(total))


def _hdr_line(mode: str, processed: int, total: int, fails: int,
              dupe_totals: Optional[int], fps: Optional[int], eta: Optional[str],
              title: str) -> str:
    pct = _pct(processed, total)
    bar = _bar(pct)
    base = f"{title:<11}|{bar}| {pct:6.1f}% {processed}/{total} FAIL {fails}"
    if mode == "MATCH" and dupe_totals is not None:
        base += f" DUPE {dupe_totals}"
    if fps is not None and eta:
        base += f" {int(fps)} FPS   ETA {eta}"
    return base


def _section_hdr_line(name: str, total: int, counts: Dict[str, int],
                       fails: int, mode: str, dupe_totals: Optional[int],
                       dedupe_on: bool) -> str:
    csum = int(sum(int(v) for v in counts.values()))
    pct  = _pct(csum, total)
    bar  = _bar(pct)
    line = f"{name:<11}|{bar}| {pct:6.1f}% {csum} FAIL {fails}"
    if mode == "MATCH" and dedupe_on and dupe_totals is not None:
        line += f" DUPE {dupe_totals}"
    return line


def _row_line(label: str, total: int, count: int,
              mode: str, dedupe_on: bool, dupe_row: Optional[int]) -> str:
    pct = _pct(count, total)
    bar = _bar(pct)
    if mode == "MATCH" and dedupe_on and dupe_row is not None:
        return f"{label:<12}|{bar}| {pct:6.1f}% {count} {dupe_row}"
    else:
        return f"{label:<12}|{bar}| {pct:6.1f}% {count}"


# ---- renderer ---------------------------------------------------------

def render_bin_table(
    mode: str,
    totals: Dict[str, Any],
    sections: Dict[str, Dict[str, Dict[str, int]]],
    dupe_totals: Optional[int] = None,
    dedupe_on: bool = True,
    fps: Optional[int] = None,
    eta: Optional[str] = None,
) -> str:
    mode = (mode or "TAG").upper()
    processed = int(totals.get("processed", 0))
    total     = int(totals.get("total", 0))
    fails     = int(totals.get("fails", 0))

    buf = io.StringIO()
    # header block
    buf.write(_sep('=' ) + "
")
    buf.write(_sep('=' ) + "
")
    buf.write(_hdr_line(mode, processed, max(1, total), fails,
                        dupe_totals if mode == "MATCH" else None,
                        fps, eta, title=mode) + "
")
    buf.write(_sep('=' ) + "
")
    buf.write(_sep('=' ) + "
")

    for sec in SECTION_ORDER:
        data = sections.get(sec, {}) or {}
        counts: Dict[str, int] = {k.upper(): int(v) for k, v in (data.get("counts", {}) or {}).items()}
        dupes_map: Dict[str, int] = {k.upper(): int(v) for k, v in (data.get("dupes", {}) or {}).items()}

        # section header
        buf.write(_section_hdr_line(sec, max(1, total), counts, fails=0,
                                    mode=mode, dupe_totals=dupe_totals,
                                    dedupe_on=dedupe_on) + "
")
        buf.write(_sep('-') + "
")

        # rows
        for label, cnt in counts.items():
            dupe_row = dupes_map.get(label) if (mode == "MATCH" and dedupe_on) else None
            buf.write(_row_line(label, max(1, total), int(cnt), mode=mode,
                                dedupe_on=dedupe_on, dupe_row=dupe_row) + "
")

        # section trailer
        buf.write(_sep('=') + "
")
        buf.write(_sep('=') + "
")

    return buf.getvalue()


# ---- writer -----------------------------------------------------------

def write_bin_table(log_dir: Path | str, text: str) -> None:
    """Write bin_table.txt atomically when possible; fall back if destination is locked."""
    out = Path(log_dir) / "bin_table.txt"
    tmp = out.with_suffix('.tmp')
    try:
        tmp.write_text(text, encoding="utf-8")
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            with open(out, 'w', encoding='utf-8') as f:
                f.write(text)
            try:
                tmp.unlink()
            except Exception:
                pass
    except Exception:
        try:
            with open(out, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception:
            pass

# Tiny help file once
try:
    help_path = Path(__file__).with_suffix('.txt')
    if not help_path.exists():
        help_path.write_text(
            "Shared bin-table renderer. Sections: GAZE, EYES, MOUTH, SMILE, EMOTION, YAW, PITCH, IDENTITY, QUALITY, TEMP, EXPOSURE.
",
            encoding='utf-8')
except Exception:
    pass

