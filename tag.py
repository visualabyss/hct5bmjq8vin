#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAG (Step 3): fusion → real bins + live counts + bin_score.
- Keeps live bin_table + manifest streaming and auto-help.
- Adds: pose/eyes/mouth/smile/teeth/gaze fusion, quality/identity bins, temp/exposure (face-only),
  human-friendly display names (exactly as in your raw tables), and per-image bin_score.
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2

from progress import ProgressBar
from bin_table import render_bin_table, write_bin_table, SECTION_ORDER

# ------------------------------ constants ------------------------------
IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".webp"}

GAZE_LABELS     = ["FRONT","LEFT","RIGHT","UP","DOWN"]
EYES_LABELS     = ["OPEN","HALF","CLOSED","W-LEFT","W-RIGHT"]
MOUTH_LABELS    = ["OPEN","SLIGHT","CLOSE"]
SMILE_LABELS    = ["TEETH","CLOSED","NONE"]
EMOTION_LABELS  = ["NEUTRAL","HAPPY","SAD","SUPRISE","FEAR","DISGUST","ANGER","CONTEMPT"]
YAW_LABELS      = ["FRONT","LEFT","RIGHT","3/4 LEFT","3/4 RIGHT","PROF LEFT","PROF RIGHT"]
PITCH_LABELS    = ["LEVEL","CHIN UP","CHIN DOWN"]
IDENTITY_LABELS = ["MATCH","MISMATCH"]
QUALITY_LABELS  = ["HIGH","MID","LOW"]
TEMP_LABELS     = ["WARM","NEUTRAL","COOL"]
EXPOSURE_LABELS = ["UNDER","NORMAL","OVER"]

AU_KEYS = ["1","2","4","6","9","12","25","26"]

# ------------------------------ helpers ------------------------------

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


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        s=str(x).strip()
        if s=="": return None
        return float(s)
    except Exception:
        return None


def _sigmoid(v: Optional[float]) -> Optional[float]:
    if v is None: return None
    try:
        if 0.0 <= v <= 1.0: return v
        return 1.0/(1.0+math.exp(-float(v)))
    except Exception:
        return None

# ------------------------------ temp/exposure (face-only) ------------------------------
SRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)


def _estimate_cct_xy(x: float, y: float) -> float:
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (y - ye + 1e-9)
    CCT = -449.0*(n**3) + 3525.0*(n**2) - 6823.3*n + 5520.33
    return float(np.clip(CCT, 1000.0, 25000.0))


def _ellipse_face_mask(H:int, W:int, shrink:float=0.86) -> np.ndarray:
    mask = np.zeros((H,W), np.uint8)
    cy, cx = H//2, W//2
    ry, rx = int(H*shrink*0.5), int(W*shrink*0.5)
    cv2.ellipse(mask, (cx,cy), (rx,ry), 0, 0, 360, 255, -1)
    return mask


def _estimate_temp_exposure(im_bgr: np.ndarray) -> Tuple[Optional[float], str, str]:
    H,W = im_bgr.shape[:2]
    mask = _ellipse_face_mask(H,W,0.86)  # MP/OSF mask fallback; ellipse for speed
    m = (mask>0)
    if not np.any(m):
        return None, "NEUTRAL", "NORMAL"
    rgb = im_bgr[..., ::-1].astype(np.float64)/255.0
    rgb_lin = _srgb_to_linear(rgb)
    R = rgb_lin[...,0][m].mean(); G = rgb_lin[...,1][m].mean(); B = rgb_lin[...,2][m].mean()
    XYZ = SRGB2XYZ @ np.array([R,G,B])
    X,Y,Z = XYZ.tolist(); den = (X+Y+Z+1e-12)
    x = X/den; y=Y/den
    cct = _estimate_cct_xy(x,y)
    if cct < 3700: temp_bin = "WARM"
    elif cct <= 5500: temp_bin = "NEUTRAL"
    else: temp_bin = "COOL"
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

# ------------------------------ fusion utils ------------------------------

def _fuse_pose(osf: Dict[str,str]) -> Tuple[Optional[float],Optional[float],Optional[float]]:
    yaw = _to_float(osf.get('yaw'))
    pitch = _to_float(osf.get('pitch'))
    roll = _to_float(osf.get('roll'))
    return yaw, pitch, roll


def _fuse_gaze(of3: Dict[str,str]) -> Tuple[Optional[float],Optional[float]]:
    gy = _to_float(of3.get('pose_yaw'))
    gp = _to_float(of3.get('pose_pitch'))
    if gy is None or gp is None:
        return None, None
    return float(gy*180.0/math.pi), float(gp*180.0/math.pi)


def _eyes_state(mp: Dict[str,str], osf: Dict[str,str]) -> str:
    # MP continuous openness
    ebl = _to_float(mp.get('eyeBlinkLeft'))
    ebr = _to_float(mp.get('eyeBlinkRight'))
    open_l = 1.0 - ebl if ebl is not None else None
    open_r = 1.0 - ebr if ebr is not None else None
    # OSF blink override
    bl = _to_float(osf.get('blink_l')); br = _to_float(osf.get('blink_r'))
    if bl is not None and bl>0.60: open_l = min(open_l or 0.0, 0.1)
    if br is not None and br>0.60: open_r = min(open_r or 0.0, 0.1)
    # decide
    if open_l is None or open_r is None:
        return "OPEN"  # fallback optimistic
    diff = open_l - open_r
    avg  = 0.5*(open_l+open_r)
    if abs(diff) >= 0.5 and avg >= 0.3:
        return "W-LEFT" if diff < 0 else "W-RIGHT"
    if avg >= 0.66: return "OPEN"
    if avg >= 0.33: return "HALF"
    return "CLOSED"


def _mouth_bins(of3: Dict[str,str], mp: Dict[str,str], au_thr: Dict[str,float]) -> Tuple[str,str,bool,float,float]:
    au25 = _sigmoid(_to_float(of3.get('au25') or of3.get('AU25')))
    au26 = _sigmoid(_to_float(of3.get('au26') or of3.get('AU26')))
    au12 = _sigmoid(_to_float(of3.get('au12') or of3.get('AU12')))
    jaw  = _to_float(mp.get('jawOpen'))
    mopen= _to_float(mp.get('mouthOpen'))
    smile_l = _to_float(mp.get('mouthSmileLeft'))
    smile_r = _to_float(mp.get('mouthSmileRight'))
    mouth_open = max([v for v in [au25,au26,jaw,mopen] if v is not None] or [0.0])
    smile_score = max([v for v in [au12,smile_l,smile_r] if v is not None] or [0.0])
    # bins
    if mouth_open >= max(0.60, au_thr.get('25',0.5)*0.9): m_bin = "OPEN"
    elif mouth_open >= max(0.30, au_thr.get('25',0.5)*0.6): m_bin = "SLIGHT"
    else: m_bin = "CLOSE"

    if mouth_open >= max(0.55, au_thr.get('25',0.5)*0.9) and smile_score >= max(0.50, au_thr.get('12',0.5)*0.9):
        s_bin = "TEETH"
        teeth = True
    elif smile_score >= max(0.50, au_thr.get('12',0.5)*0.9):
        s_bin = "CLOSED"
        teeth = False
    else:
        s_bin = "NONE"; teeth=False
    return m_bin, s_bin, teeth, float(mouth_open), float(smile_score)


def _gaze_bin(gyaw: Optional[float], gpitch: Optional[float]) -> str:
    if gyaw is None or gpitch is None:
        return "FRONT"
    if abs(gyaw) <= 12.0 and abs(gpitch) <= 10.0:
        return "FRONT"
    if abs(gyaw) >= abs(gpitch):
        return "LEFT" if gyaw < 0 else "RIGHT"
    else:
        return "UP" if gpitch > 0 else "DOWN"


def _yaw_bin(yaw: Optional[float]) -> str:
    if yaw is None:
        return "FRONT"
    a = abs(yaw)
    if a <= 15: return "FRONT"
    if a <= 35: return "LEFT" if yaw < 0 else "RIGHT"
    if a <= 55: return "3/4 LEFT" if yaw < 0 else "3/4 RIGHT"
    return "PROF LEFT" if yaw < 0 else "PROF RIGHT"


def _pitch_bin(pitch: Optional[float]) -> str:
    if pitch is None: return "LEVEL"
    if pitch >= 10: return "CHIN UP"
    if pitch <= -10: return "CHIN DOWN"
    return "LEVEL"


def _quality_bin(mag: Optional[float], q_lo: float, q_hi: float) -> str:
    if mag is None: return "MID"
    if mag >= q_hi: return "HIGH"
    if mag <  q_lo: return "LOW"
    return "MID"

# ------------------------------ auto-help ------------------------------
AUTO_HELP = r"""
PURPOSE
  • Consolidate tool outputs into manifest.jsonl and live bin_table.txt.
  • Fusion of pose/gaze/eyes/mouth/smile/teeth + quality/identity + temp/exposure (face-only).

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
    # ID threshold (robust fallback)
    def _calib_id_threshold(af_map: Dict[str,Dict[str,str]]) -> float:
        vals = [ _to_float(v.get('id_score')) for v in af_map.values() ]
        vals = [x for x in vals if x is not None]
        if len(vals) < 16:
            return 0.55
        th = float(np.quantile(np.array(vals), 0.10))
        return float(np.clip(th, 0.50, 0.80))

    id_thresh = _calib_id_threshold(src_maps.get('af', {})) if use_af else 0.55

    # MagFace tertiles
    def _calib_mag_tertiles(mf_map: Dict[str,Dict[str,str]]) -> Tuple[float,float]:
        mags=[]
        for v in mf_map.values():
            q = _to_float(v.get('quality') or v.get('mag'))
            if q is not None: mags.append(q)
        if not mags:
            return (0.0, 0.0)
        arr = np.array(mags, dtype=np.float64)
        return float(np.quantile(arr,0.33)), float(np.quantile(arr,0.66))

    q_lo,q_hi = _calib_mag_tertiles(src_maps.get('mf', {})) if use_mf else (0.0,0.0)

    # AU thresholds (80th percentile per-AU over clip)
    def _calib_au_thresholds(of3_map: Dict[str,Dict[str,str]]) -> Dict[str,float]:
        vals={k:[] for k in AU_KEYS}
        for row in of3_map.values():
            for ak in AU_KEYS:
                for col in (f"au{ak}", f"AU{ak}"):
                    if col in row:
                        v=_to_float(row.get(col))
                        if v is None: continue
                        v=_sigmoid(v)
                        if v is not None: vals[ak].append(float(np.clip(v,0.0,1.0)))
                        break
        thr={}
        for ak in AU_KEYS:
            arr = np.array(vals[ak], dtype=np.float64)
            thr[ak] = float(np.quantile(arr,0.80)) if arr.size>0 else 0.50
        return thr

    au_thr = _calib_au_thresholds(src_maps.get('of3', {})) if use_of3 else {k:0.5 for k in AU_KEYS}

    # store calibrations
    tag_dir = logs/"tag"; tag_dir.mkdir(parents=True, exist_ok=True)
    try:
        (tag_dir/"calib.json").write_text(json.dumps({
            "id_threshold": id_thresh,
            "mag_tertiles": {"low": q_lo, "high": q_hi},
            "au_thresholds": au_thr,
            "temp_thresholds": {"warm": 3700, "cool": 5500},
        }, indent=2), encoding='utf-8')
    except Exception:
        pass

    # ---------------- outputs ----------------
    manifest_path = logs/"manifest.jsonl"
    if _onoff(args.override, False) and manifest_path.exists():
        try: manifest_path.unlink()
        except: pass

    # init live bin table counts
    def init_sections_counts():
        return {
            "GAZE":     {"counts":{k:0 for k in GAZE_LABELS}},
            "EYES":     {"counts":{k:0 for k in EYES_LABELS}},
            "MOUTH":    {"counts":{k:0 for k in MOUTH_LABELS}},
            "SMILE":    {"counts":{k:0 for k in SMILE_LABELS}},
            "EMOTION":  {"counts":{k:0 for k in EMOTION_LABELS}},
            "YAW":      {"counts":{k:0 for k in YAW_LABELS}},
            "PITCH":    {"counts":{k:0 for k in PITCH_LABELS}},
            "IDENTITY": {"counts":{k:0 for k in IDENTITY_LABELS}},
            "QUALITY":  {"counts":{k:0 for k in QUALITY_LABELS}},
            "TEMP":     {"counts":{k:0 for k in TEMP_LABELS}},
            "EXPOSURE": {"counts":{k:0 for k in EXPOSURE_LABELS}},
        }

    sections = init_sections_counts()

    # progress + live table
    bar = ProgressBar("TAG", total=total, show_fail_label=True)

    processed = 0
    fails = 0
    t0=time.time()

    # process sequentially; write manifest with bins & scores
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

            # derived: gaze (deg)
            gyaw_deg, gpitch_deg = _fuse_gaze(of3)

            # pose: OSF primary
            yaw = _to_float(osf.get('yaw'))
            pitch = _to_float(osf.get('pitch'))
            roll = _to_float(osf.get('roll'))

            # temp/exposure (face-only ellipse)
            tempK=None; temp_bin="NEUTRAL"; exp_bin="NORMAL"
            try:
                im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is None: im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is not None:
                    tempK, temp_bin, exp_bin = _estimate_temp_exposure(im)
            except Exception:
                pass

            # eyes / mouth / smile
            eyes_bin = _eyes_state(mp, osf)
            m_bin, s_bin, teeth, mopen_val, smile_val = _mouth_bins(of3, mp, au_thr)

            # gaze bin
            gaze_bin = _gaze_bin(gyaw_deg, gpitch_deg)

            # yaw/pitch bins
            yaw_bin = _yaw_bin(yaw)
            pitch_bin = _pitch_bin(pitch)

            # identity & quality
            id_score = _to_float(af.get('id_score'))
            id_bin = None
            if id_score is not None:
                id_bin = "MATCH" if id_score >= id_thresh else "MISMATCH"
                sections["IDENTITY"]["counts"][id_bin] += 1
            mag = _to_float(mfq.get('quality') or mfq.get('mag'))
            q_bin = _quality_bin(mag, q_lo, q_hi) if use_mf else "MID"

            # sections increments
            sections["GAZE"]["counts"][gaze_bin] += 1
            sections["EYES"]["counts"][eyes_bin] += 1
            sections["MOUTH"]["counts"][m_bin] += 1
            sections["SMILE"]["counts"][s_bin] += 1
            # emotion (soft): map if present & confident, else skip increment
            emo_lbl=None
            emo = (of3.get('emotion') or of3.get('emotion_top1') or of3.get('emotion_label'))
            emo_conf = _to_float(of3.get('emotion_conf') or of3.get('emotion_prob') or of3.get('emotion_confidence'))
            if emo:
                lbl = str(emo).strip().upper()
                if lbl=="SURPRISE": lbl="SUPRISE"  # match table spelling
                if emo_conf is None or emo_conf>=0.60:
                    if lbl in sections["EMOTION"]["counts"]:
                        sections["EMOTION"]["counts"][lbl]+=1
                        emo_lbl = lbl
            sections["YAW"]["counts"][yaw_bin]+=1
            sections["PITCH"]["counts"][pitch_bin]+=1
            sections["QUALITY"]["counts"][q_bin]+=1
            sections["TEMP"]["counts"][temp_bin]+=1
            sections["EXPOSURE"]["counts"][exp_bin]+=1

            # pose centering score
            def _yaw_center(yb: str) -> float:
                return {"FRONT":0.0, "LEFT":-25.0, "RIGHT":25.0,
                        "3/4 LEFT":-45.0, "3/4 RIGHT":45.0,
                        "PROF LEFT":-70.0, "PROF RIGHT":70.0}.get(yb,0.0)
            def _yaw_halfwidth(yb: str) -> float:
                return {"FRONT":15.0, "LEFT":10.0, "RIGHT":10.0,
                        "3/4 LEFT":10.0, "3/4 RIGHT":10.0,
                        "PROF LEFT":15.0, "PROF RIGHT":15.0}.get(yb,10.0)
            pose_center = 0.0
            if yaw is not None:
                hw = _yaw_halfwidth(yaw_bin)
                pose_center = max(0.0, 1.0 - abs(yaw - _yaw_center(yaw_bin))/max(1e-6, hw))

            # expr bonus from smile
            expr_bonus = float(smile_val)
            # pnp penalty
            pnp = _to_float(osf.get('pnp_error')) or 0.0
            pnp_norm = min(1.0, pnp/0.02)

            # quality normalized approx
            if use_mf and mag is not None and q_hi>q_lo:
                q_norm = float(np.clip((mag - q_lo) / max(1e-9, (q_hi - q_lo)), 0.0, 1.0))
            else:
                q_norm = 0.5

            # id score default
            id_s = float(id_score) if id_score is not None else 0.55

            bin_score = 0.55*id_s + 0.30*q_norm + 0.10*pose_center + 0.05*expr_bonus - 0.05*pnp_norm

            # write manifest record
            rec = {
                "file": fn,
                "bins": {
                    "GAZE": gaze_bin,
                    "EYES": eyes_bin,
                    "MOUTH": m_bin,
                    "SMILE": s_bin,
                    "EMOTION": emo_lbl if emo_lbl else None,
                    "YAW": yaw_bin,
                    "PITCH": pitch_bin,
                    "IDENTITY": id_bin,
                    "QUALITY": q_bin,
                    "TEMP": temp_bin,
                    "EXPOSURE": exp_bin,
                },
                "scores": {
                    "id_score": id_score,
                    "q_magface": mag,
                    "pose_centering": pose_center,
                    "expr_strength": smile_val,
                    "pnp_error": pnp,
                    "bin_score": bin_score,
                },
                "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
                "gaze_deg": {"yaw": gyaw_deg, "pitch": gpitch_deg},
                "src": {"of3": of3, "mp": mp, "osf": osf, "af": af, "mf": mfq, "fx": fx}
            }
            try:
                with open(manifest_path, 'a', encoding='utf-8') as out:
                    out.write(json.dumps(rec, ensure_ascii=False) + '
')
            except Exception:
                fails += 1

            # live bin table update
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
