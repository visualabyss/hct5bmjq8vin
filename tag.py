#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAG (Step 2): add calibrations + face-only temperature/exposure + derived signals.
- Keeps Step 1 plumbing (live bin_table, manifest streaming, readers, auto-help).
- Adds per-clip calibration artifacts written to logs/tag/calib.json.
- Per-image derived fields saved into manifest.jsonl under "derived".
- No final bin fusion yet (that will be Step 3). Counts in bin_table remain zero for now.
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2

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
  • Step 2 adds: per-clip calibrations, OF3-derived signals, and face-only TEMP/EXPOSURE.

USAGE
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

# ------------------------------ calibration utils ------------------------------

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        s=str(x).strip()
        if s=="": return None
        return float(s)
    except Exception:
        return None


def _nanless(xs: List[Optional[float]]) -> np.ndarray:
    arr = np.array([x for x in xs if x is not None], dtype=np.float64)
    return arr


def _quantiles(arr: np.ndarray, qs=(0.33,0.66)) -> Tuple[float,float]:
    if arr.size==0:
        return (0.33,0.66)
    q1,q2 = np.quantile(arr, qs)
    return float(q1), float(q2)


# ArcFace threshold (fallback: 0.55)

def _calib_id_threshold(af_map: Dict[str,Dict[str,str]]) -> float:
    vals = _nanless([_to_float(v.get('id_score')) for v in af_map.values()])
    if vals.size < 16:
        return 0.55
    # Robust: pick the 10th percentile of scores as threshold floor, but cap within [0.50, 0.80]
    th = float(np.quantile(vals, 0.10))
    return float(np.clip(th, 0.50, 0.80))


# MagFace tertiles

def _calib_mag_tertiles(mf_map: Dict[str,Dict[str,str]]) -> Tuple[float,float]:
    mags = _nanless([_to_float(v.get('quality')) or _to_float(v.get('mag')) for v in mf_map.values()])
    if mags.size==0:
        return (0.0, 0.0)
    return _quantiles(mags, (0.33,0.66))


# OF3 AUs: ensure probabilities in [0,1]
AU_KEYS = ["1","2","4","6","9","12","25","26"]

def _calib_au_thresholds(of3_map: Dict[str,Dict[str,str]], mf_map: Dict[str,Dict[str,str]]) -> Dict[str,float]:
    # collect per-AU values; prefer high-quality frames if MagFace present
    mags = {k: _to_float(v.get('quality') or v.get('mag')) for k,v in mf_map.items()} if mf_map else {}
    vals: Dict[str, List[float]] = {k: [] for k in AU_KEYS}
    for fn, row in of3_map.items():
        q = mags.get(fn, None)
        if mags and (q is not None):
            pass  # use all; later we can filter by >= median
        for ak in AU_KEYS:
            for col in (f"au{ak}", f"AU{ak}"):
                if col in row:
                    v=_to_float(row.get(col));
                    if v is None: continue
                    # if outside [0,1], squash with sigmoid
                    if v<0 or v>1: v = 1.0/(1.0+math.exp(-float(v)))
                    vals[ak].append(float(np.clip(v,0.0,1.0)))
                    break
    thres: Dict[str,float] = {}
    for ak in AU_KEYS:
        arr = np.array(vals[ak], dtype=np.float64)
        if arr.size==0:
            thres[ak]=0.50
        else:
            # 80th percentile as "active" threshold per-clip
            thres[ak]=float(np.quantile(arr,0.80))
    return thres


# ------------------------------ face ROI + TEMP/EXPOSURE ------------------------------
SRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)


def _estimate_cct_xy(x: float, y: float) -> float:
    # McCamy approximation
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (y - ye + 1e-9)
    CCT = -449.0*(n**3) + 3525.0*(n**2) - 6823.3*n + 5520.33
    return float(np.clip(CCT, 1000.0, 25000.0))


def _face_mask_from_mp(mp_row: Dict[str,str], H:int, W:int) -> Optional[np.ndarray]:
    # Expect per-image JSON elsewhere; CSV may not have landmarks. Fallback returns None.
    return None


def _ellipse_face_mask(H:int, W:int, shrink:float=0.85) -> np.ndarray:
    mask = np.zeros((H,W), np.uint8)
    cy, cx = H//2, W//2
    ry, rx = int(H*shrink*0.5), int(W*shrink*0.5)
    cv2.ellipse(mask, (cx,cy), (rx,ry), 0, 0, 360, 255, -1)
    return mask


def _estimate_temp_exposure(im_bgr: np.ndarray, mp_row: Optional[Dict[str,str]]=None) -> Tuple[Optional[float], str, str]:
    H,W = im_bgr.shape[:2]
    mask = _face_mask_from_mp(mp_row, H, W)
    if mask is None:
        mask = _ellipse_face_mask(H,W,0.86)
    # apply mask
    m = (mask>0)
    if not np.any(m):
        return None, "NEUTRAL", "NORMAL"
    rgb = im_bgr[..., ::-1].astype(np.float64)/255.0
    rgb_lin = _srgb_to_linear(rgb)
    # compute mean chromaticity over ROI
    R = rgb_lin[...,0][m].mean(); G = rgb_lin[...,1][m].mean(); B = rgb_lin[...,2][m].mean()
    XYZ = SRGB2XYZ @ np.array([R,G,B])
    X,Y,Z = XYZ.tolist()
    den = (X+Y+Z+1e-12)
    x = X/den; y=Y/den
    cct = _estimate_cct_xy(x,y)
    # bucket temp
    if cct < 3700: temp_bin = "WARM"
    elif cct <= 5500: temp_bin = "NEUTRAL"
    else: temp_bin = "COOL"
    # exposure via log-average luminance and clipping
    L = 0.2126*rgb_lin[...,0] + 0.7152*rgb_lin[...,1] + 0.0722*rgb_lin[...,2]
    Lm = L[m]
    log_avg = math.exp(np.log(Lm+1e-6).mean())
    sdr = im_bgr.astype(np.float64)/255.0
    low_clip = float((sdr[...,0][m]<0.02).mean()*100 + (sdr[...,1][m]<0.02).mean()*100 + (sdr[...,2][m]<0.02).mean()*100)/3.0
    hi_clip  = float((sdr[...,0][m]>0.98).mean()*100 + (sdr[...,1][m]>0.98).mean()*100 + (sdr[...,2][m]>0.98).mean()*100)/3.0
    if hi_clip>2.0 and log_avg>0.65: exp_bin = "OVER"
    elif low_clip>2.0 and log_avg<0.15: exp_bin = "UNDER"
    else: exp_bin = "NORMAL"
    return float(cct), temp_bin, exp_bin


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
        m = dict(m1); m.update(m2)
        src_maps['of3'] = m

    if use_mp:
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

    vs = logs/"video"/"video_stats.csv"
    src_maps['fx'] = _read_csv_map(vs)

    # ---------------- per-clip calibrations ----------------
    id_thresh = _calib_id_threshold(src_maps.get('af', {})) if use_af else 0.55
    q_lo,q_hi = _calib_mag_tertiles(src_maps.get('mf', {})) if use_mf else (0.0,0.0)
    au_thr    = _calib_au_thresholds(src_maps.get('of3', {}), src_maps.get('mf', {})) if use_of3 else {k:0.5 for k in AU_KEYS}

    tag_dir = logs/"tag"; tag_dir.mkdir(parents=True, exist_ok=True)
    calib_path = tag_dir/"calib.json"
    calib = {
        "id_threshold": id_thresh,
        "mag_tertiles": {"low": q_lo, "high": q_hi},
        "au_thresholds": au_thr,
        "temp_thresholds": {"warm": 3700, "cool": 5500},
        "exposure": {"low_clip_pct": 2.0, "hi_clip_pct": 2.0, "logavg_under": 0.15, "logavg_over": 0.65},
    }
    try:
        calib_path.write_text(json.dumps(calib, indent=2), encoding='utf-8')
    except Exception:
        pass

    # ---------------- outputs ----------------
    manifest_path = logs/"manifest.jsonl"
    if _onoff(args.override, False) and manifest_path.exists():
        try: manifest_path.unlink()
        except: pass

    # init live bin table (zero counts for now)
    def empty_sections():
        secs = {}
        for sec in SECTION_ORDER:
            secs[sec] = {"counts":{}}
        return secs

    sections = empty_sections()

    # progress + live table
    bar = ProgressBar("TAG", total=total, show_fail_label=True)

    processed = 0
    fails = 0
    t0=time.time()

    # process sequentially; write manifest with derived block
    with manifest_path.open("a", encoding="utf-8") as mf:
        for p in imgs:
            processed += 1
            fn = p.name
            of3 = src_maps.get('of3',{}).get(fn) or {}
            mp  = src_maps.get('mp',{}).get(fn)  or {}
            osf = src_maps.get('osf',{}).get(fn) or {}
            af  = src_maps.get('af',{}).get(fn)  or {}
            mfq = src_maps.get('mf',{}).get(fn)  or {}
            fx  = src_maps.get('fx',{}).get(fn)  or {}

            # --- derived: OF3 gaze degrees ---
            gaze_deg = None
            try:
                gy = _to_float(of3.get('pose_yaw')); gp = _to_float(of3.get('pose_pitch'))
                if gy is not None and gp is not None:
                    gaze_deg = {"yaw": float(gy*180.0/math.pi), "pitch": float(gp*180.0/math.pi)}
            except Exception:
                gaze_deg = None

            # --- derived: TEMP/EXPOSURE over face-only ROI ---
            tempK=None; temp_bin="NEUTRAL"; exp_bin="NORMAL"
            try:
                im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is not None:
                    tempK, temp_bin, exp_bin = _estimate_temp_exposure(im, mp_row=mp)
            except Exception:
                pass

            rec = {
                "file": fn,
                "src": {"of3": of3, "mp": mp, "osf": osf, "af": af, "mf": mfq, "fx": fx},
                "derived": {
                    "gaze_deg": gaze_deg,
                    "tempK": tempK,
                    "temp_bin": temp_bin,
                    "exp_bin": exp_bin,
                }
            }
            try:
                mf.write(json.dumps(rec, ensure_ascii=False) + "
")
            except Exception:
                fails += 1

            # live bin table update (still zero counts in Step 2)
            elapsed = time.time()-t0
            fps = int(processed/elapsed) if elapsed>0 else 0
            eta = int((total-processed)/max(1,fps))
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
