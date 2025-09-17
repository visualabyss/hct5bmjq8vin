#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAG (Step 1): scaffolding + readers + auto-help + live bin_table output.
- Writes manifest.jsonl minimally (file + tool presence/raw rows pass-through where cheap).
- No fusion/bins yet (next steps). Sections show up with zeroed counts for now.
- Live bin_table updates using shared renderer in bin_table.py.

Usage (current):
  python3 -u /workspace/scripts/tag.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --openface3 on --mediapipe on --openseeface on \
    --arcface on --magface off --override

This step focuses on stable IO, file discovery, and progress + live table plumbing.
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse
from pathlib import Path
from typing import Dict, Any

from progress import ProgressBar
from bin_table import render_bin_table, write_bin_table, SECTION_ORDER

# ------------------------------ helpers ------------------------------
IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".webp"}

def _onoff(v, default=False) -> bool:
    if v is None: return bool(default)
    if isinstance(v,(int,float)): return bool(int(v))
    s=str(v).strip().lower()
    if s in {"1","on","true","yes","y"}: return True
    if s in {"0","off","false","no","n"}: return False
    return bool(default)

def _list_images(d: Path):
    xs=[p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return xs

# tolerant CSV reader (key by basename)

def _read_csv_map(path: Path, key_col_candidates=("file","path","name")) -> Dict[str, Dict[str,str]]:
    m: Dict[str,Dict[str,str]] = {}
    if not path.exists():
        return m
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        key_col = None
        for k in key_col_candidates:
            if k in cols:
                key_col = k; break
        if key_col is None:
            # try to infer: if 'path' exists but contains basename, split
            key_col = cols[0] if cols else None
        for row in rd:
            key = row.get(key_col, "") if key_col else ""
            if not key:
                continue
            key = os.path.basename(key)
            m[key] = row
    return m

# ------------------------------ auto-help ------------------------------
AUTO_HELP = r"""
PURPOSE
  • Consolidate tool outputs into a single manifest.jsonl for downstream match.py.
  • Live-write a human-friendly bin_table.txt (shared renderer in bin_table.py).
  • This step initializes IO + readers; fusion/bins/metrics come next.

USAGE (now)
  python3 -u /workspace/scripts/tag.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --openface3 on --mediapipe on --openseeface on \
    --arcface on --magface off --override

FLAGS
  --aligned DIR (req)              aligned images dir
  --logs DIR (req)                 logs root (writes manifest.jsonl, bin_table.txt)
  --openface3 on|off (on)          include OF3 readers if present
  --mediapipe on|off (on)          include MediaPipe readers if present
  --openseeface on|off (on)        include OpenSeeFace readers if present
  --arcface on|off (on)            include ArcFace readers if present
  --magface on|off (off)           include MagFace readers if present
  --override on|off (off)          overwrite manifest.jsonl instead of appending
  --help on|off (on)               write /workspace/scripts/tag.txt then proceed
  -h / --h                         print this help text and continue
"""

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    ap.add_argument('--aligned', required=True)
    ap.add_argument('--logs', required=True)
    ap.add_argument('--openface3', default='on')
    ap.add_argument('--mediapipe', default='on')
    ap.add_argument('--openseeface', default='on')
    ap.add_argument('--arcface', default='on')
    ap.add_argument('--magface', default='off')
    ap.add_argument('--override', default='off')
    ap.add_argument('--help', dest='auto_help', default='on')
    args = ap.parse_args()

    if getattr(args,'show_cli_help', False):
        print(AUTO_HELP)
    try:
        if _onoff(args.auto_help, True):
            sp = Path(__file__).resolve(); (sp.with_name(sp.stem + '.txt')).write_text(AUTO_HELP, encoding='utf-8')
    except Exception:
        pass

    use_of3  = _onoff(args.openface3, True)
    use_mp   = _onoff(args.mediapipe, True)
    use_osf  = _onoff(args.openseeface, True)
    use_af   = _onoff(args.arcface, True)
    use_mf   = _onoff(args.magface, False)

    aligned = Path(args.aligned)
    logs    = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)

    imgs = _list_images(aligned)
    total = len(imgs)
    if total == 0:
        print("TAG: no images under --aligned.")
        return 2

    # ---------------- readers (best-effort) ----------------
    src_maps: Dict[str, Dict[str, Dict[str,str]]] = {}

    if use_of3:
        p1 = logs/"openface3"/"openface3.csv"
        p2 = logs/"openface3"/"openface3_full.csv"
        m1 = _read_csv_map(p1)
        m2 = _read_csv_map(p2)
        # merge (prefer full on collisions)
        m = dict(m1); m.update(m2)
        src_maps['of3'] = m

    if use_mp:
        # try common filenames
        p = None
        for cand in (logs/"mediapipe"/"mediapipe.csv", logs/"mediapipe"/"mp_tags.csv"):
            if cand.exists(): p=cand; break
        if p: src_maps['mp'] = _read_csv_map(p)
        else: src_maps['mp'] = {}

    if use_osf:
        p = logs/"openseeface"/"osf_tags.csv"
        src_maps['osf'] = _read_csv_map(p)

    if use_af:
        p = logs/"arcface"/"arcface.csv"
        src_maps['af'] = _read_csv_map(p)

    if use_mf:
        p = logs/"magface"/"magface.csv"
        src_maps['mf'] = _read_csv_map(p)

    # face-extract video stats (optional)
    vs = logs/"video"/"video_stats.csv"
    src_maps['fx'] = _read_csv_map(vs)

    # ---------------- outputs ----------------
    manifest_path = logs/"manifest.jsonl"
    if _onoff(args.override, False) and manifest_path.exists():
        try: manifest_path.unlink()
        except: pass

    # init live bin table (all zero counts for now)
    def empty_sections():
        secs = {}
        for sec in SECTION_ORDER:
            secs[sec] = {"counts":{}}  # child rows to be filled in later steps
        return secs

    sections = empty_sections()

    # progress + live table
    bar = ProgressBar("TAG", total=total, show_fail_label=True)

    processed = 0
    fails = 0
    t0=time.time()

    # process sequentially; write manifest minimally
    with manifest_path.open("a", encoding="utf-8") as mf:
        for p in imgs:
            processed += 1
            fn = p.name
            rec = {
                "file": fn,
                "src": {
                    k: (src_maps.get(k,{}).get(fn) or {}) for k in ("of3","mp","osf","af","mf","fx")
                }
            }
            try:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                fails += 1
            # live bin table update (zeroed sections for now)
            elapsed = time.time()-t0
            fps = int(processed/elapsed) if elapsed>0 else 0
            eta = int((total-processed)/fps) if fps>0 else 0
            eta_s = f"{eta//60:02d}:{eta%60:02d}"
            tbl = render_bin_table(
                mode="TAG",
                totals={"processed":processed, "total":total, "fails":fails},
                sections=sections,
                dupe_totals=None,
                dedupe_on=True,
                fps=fps,
                eta=eta_s,
            )
            write_bin_table(logs, tbl)
            bar.update(processed, fails=fails, fps=fps)

    bar.close()
    print(f"TAG: done. images={total} fails={fails} manifest={manifest_path}")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
