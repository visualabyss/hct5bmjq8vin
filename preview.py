#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview.py — quick visual QA for bins.
Reads manifest.jsonl and writes:
  • logs/preview/preview_list.csv (category,label,file,score,path)
  • logs/preview/<CAT>__<LABEL>.jpg (grids of samples with overlays)

USAGE (common)
  python3 -u /workspace/scripts/preview.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --per_bin 12 --grid 6x2 --seed 0

Notes:
- Picks top-N by bin_score when available; falls back to random.
- Skips empty bins automatically.
- Overlays: file name, pose (yaw/pitch), and bin label.
"""
from __future__ import annotations
import argparse, json, random, csv, math
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".webp"}
CATS = [
  "GAZE","EYES","MOUTH","SMILE","EMOTION","YAW","PITCH","IDENTITY","QUALITY","TEMP","EXPOSURE"
]


def _read_manifest(p: Path) -> List[Dict]:
    xs=[]
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                xs.append(json.loads(ln))
            except Exception:
                continue
    return xs


def _pick(samples: List[Dict], n: int) -> List[Dict]:
    if len(samples) <= n:
        return samples
    # prefer top by bin_score if present
    if any(isinstance((s.get('scores') or {}).get('bin_score'), (int,float)) for s in samples):
        ys = sorted(samples, key=lambda r: float((r.get('scores') or {}).get('bin_score') or 0.0), reverse=True)
        return ys[:n]
    random.shuffle(samples)
    return samples[:n]


def _try_font(size:int=14):
    # Try common fonts; fall back to default PIL font
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        p = Path(name)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def _overlay(im: Image.Image, label: str, file: str, yaw: float|None, pitch: float|None):
    draw = ImageDraw.Draw(im)
    W,H = im.size
    font = _try_font(14)
    # Bottom bar
    bar_h = 22
    draw.rectangle([0, H-bar_h, W, H], fill=(0,0,0,160))
    txt = f"{label}  ·  {file}"
    draw.text((6, H-bar_h+3), txt, fill=(235,235,235), font=font)
    # Pose (top-right)
    if yaw is not None or pitch is not None:
        ptxt = f"yaw={yaw:.1f}  pitch={pitch:.1f}"
        w,h = draw.textsize(ptxt, font=font)
        draw.rectangle([W-w-8, 0, W, h+6], fill=(0,0,0,160))
        draw.text((W-w-4, 3), ptxt, fill=(235,235,235), font=font)


def _grid(out: Path, pairs: List[Tuple[Path,str,Dict]], cols: int, rows: int, tile: int = 256):
    if not pairs:
        return
    W, H = cols*tile, rows*tile + 36
    im = Image.new("RGB", (W,H), (20,20,20))
    draw = ImageDraw.Draw(im)
    title = out.stem.replace("__"," ")
    draw.text((8,4), title, fill=(240,240,240))
    y0 = 36
    i=0
    for r in range(rows):
        for c in range(cols):
            if i>=len(pairs): break
            p,label,rec = pairs[i]; i+=1
            try:
                src = Image.open(p).convert("RGB")
                src = src.resize((tile,tile), Image.LANCZOS)
                # overlay
                y = rec.get('pose', {}).get('yaw')
                pi = rec.get('pose', {}).get('pitch')
                _overlay(src, label, p.name, (float(y) if isinstance(y,(int,float)) else None), (float(pi) if isinstance(pi,(int,float)) else None))
                im.paste(src, (c*tile, y0 + r*tile))
            except Exception:
                pass
    im.save(out, quality=92)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--per_bin", type=int, default=12)
    ap.add_argument("--grid", type=str, default="6x2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cats", type=str, default=",")  # comma-separated subset of CATS
    args=ap.parse_args()

    random.seed(args.seed)
    aligned=Path(args.aligned)
    logs=Path(args.logs)
    man = logs/"manifest.jsonl"
    if not man.exists():
        print("preview: manifest.jsonl missing. Run tag.py first.")
        return 2

    rows=_read_manifest(man)

    # index by category/label
    by: Dict[str, Dict[str, List[Dict]]] = {c:{} for c in CATS}
    for r in rows:
        f = r.get('file'); p = aligned/f if f else None
        if not p or not p.exists():
            continue
        bins = r.get('bins') or {}
        for c in CATS:
            lbl = bins.get(c)
            if not lbl:
                continue
            by.setdefault(c, {}).setdefault(lbl, []).append({**r, '_path': p})

    # parse grid
    try:
        gc,gr = (int(x) for x in args.grid.lower().split('x'))
    except Exception:
        gc,gr = 6,2

    out_dir = logs/"preview"; out_dir.mkdir(parents=True, exist_ok=True)
    lst_path = out_dir/"preview_list.csv"

    allowed = [s for s in [s.strip().upper() for s in args.cats.split(',')] if s] or CATS

    with lst_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["category","label","file","score","path"])
        for cat in CATS:
            if cat not in allowed:
                continue
            for lbl, xs in (by.get(cat) or {}).items():
                sel = _pick(xs, args.per_bin)
                pairs=[(d['_path'], lbl, d) for d in sel]
                grid_path = out_dir/f"{cat}__{lbl}.jpg"
                _grid(grid_path, pairs, gc, gr)
                for d in sel:
                    w.writerow([cat, lbl, d.get('file'), (d.get('scores') or {}).get('bin_score'), str(d['_path'])])

    print(f"preview: wrote {lst_path} and images under {out_dir}")
    return 0

if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
