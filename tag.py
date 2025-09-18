#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAG: fuse tool outputs → human-friendly bins, live bin_table.txt, and manifest.jsonl rows per image.
- Sources: OpenFace3 (of3), MediaPipe (mp), OpenSeeFace (osf), ArcFace (af), MagFace (mf).
- Face-only temp/exposure; robust CSV merging; dataset-centered pose calibration.
- Auto-help writes tag.txt; --override is a flag.
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, math, statistics
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2

from progress import ProgressBar
from bin_table import render_bin_table, write_bin_table

IMAGE_EXTS = {".png",".jpg",".jpeg",".bmp",".webp"}

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
AU_KEYS         = ["1","2","4","6","9","12","25","26"]

SRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)

# ---------------- helpers ----------------

def _onoff(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"1","on","true","yes","y"}: return True
    if s in {"0","off","false","no","n"}: return False
    return bool(default)


def _list_images(d: Path):
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _read_csv_map(path: Path, key_cols=("file","path","name")) -> Dict[str, Dict[str,str]]:
    m: Dict[str, Dict[str,str]] = {}
    if not path.exists():
        return m
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        key = None
        for k in key_cols:
            if k in cols:
                key = k; break
        if key is None and cols:
            key = cols[0]
        for row in rd:
            k = os.path.basename(row.get(key, "")) if key else ""
            if k:
                m[k] = row
    return m


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _sigmoid(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        if 0.0 <= v <= 1.0:
            return v
        return 1.0/(1.0+math.exp(-float(v)))
    except Exception:
        return None

# ---- temp/exposure (face-only ROI) ----

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)


def _estimate_cct_xy(x: float, y: float) -> float:
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (y - ye + 1e-9)
    return float(np.clip(-449.0*(n**3) + 3525.0*(n**2) - 6823.3*n + 5520.33, 1000.0, 25000.0))


def _ellipse_mask(H: int, W: int, shrink: float=0.86) -> np.ndarray:
    m = np.zeros((H,W), np.uint8)
    cy, cx = H//2, W//2
    ry, rx = int(H*shrink*0.5), int(W*shrink*0.5)
    cv2.ellipse(m, (cx,cy), (rx,ry), 0, 0, 360, 255, -1)
    return m


def _estimate_temp_exposure(bgr: np.ndarray):
    H, W = bgr.shape[:2]
    m = _ellipse_mask(H, W, 0.86).astype(bool)
    if not np.any(m):
        return None, "NEUTRAL", "NORMAL"
    rgb = bgr[..., ::-1].astype(np.float64)/255.0
    lin = _srgb_to_linear(rgb)
    R, G, B = lin[...,0][m].mean(), lin[...,1][m].mean(), lin[...,2][m].mean()
    X, Y, Z = (SRGB2XYZ @ np.array([R,G,B])).tolist()
    den = X+Y+Z+1e-12
    x, y = X/den, Y/den
    cct = _estimate_cct_xy(x, y)
    temp_bin = "WARM" if cct < 3700 else ("NEUTRAL" if cct <= 5500 else "COOL")

    L = 0.2126*lin[...,0] + 0.7152*lin[...,1] + 0.0722*lin[...,2]
    Lm = L[m]
    log_avg = float(math.exp(np.log(Lm+1e-6).mean()))

    sdr = bgr.astype(np.float64)/255.0
    low = float(((sdr[...,0][m] < 0.02).mean() + (sdr[...,1][m] < 0.02).mean() + (sdr[...,2][m] < 0.02).mean())*100/3)
    hi = float(((sdr[...,0][m] > 0.98).mean() + (sdr[...,1][m] > 0.98).mean() + (sdr[...,2][m] > 0.98).mean())*100/3)
    exp_bin = "OVER" if (hi > 0.2 and log_avg > 0.52) else ("UNDER" if (low > 8.0 and log_avg < 0.09) else "NORMAL")
    return cct, temp_bin, exp_bin

# ---------------- fusion ----------------

def _eyes_state(mp: Dict[str,str], osf: Dict[str,str]) -> str:
    ebl = _to_float(mp.get('blend::eyeBlinkLeft') or mp.get('eyeBlinkLeft'))
    ebr = _to_float(mp.get('blend::eyeBlinkRight') or mp.get('eyeBlinkRight'))
    open_l = 1.0 - ebl if ebl is not None else None
    open_r = 1.0 - ebr if ebr is not None else None

    bl = _to_float(osf.get('blink_l'))
    br = _to_float(osf.get('blink_r'))
    if bl is not None and bl > 0.60: open_l = min(open_l or 0.0, 0.1)
    if br is not None and br > 0.60: open_r = min(open_r or 0.0, 0.1)

    if open_l is None or open_r is None:
        return "OPEN"

    diff = open_l - open_r
    avg = 0.5*(open_l + open_r)
    if abs(diff) >= 0.50 and avg >= 0.30:
        return "W-LEFT" if diff < 0 else "W-RIGHT"
    if avg >= 0.45: return "OPEN"
    if avg >= 0.25: return "HALF"
    return "CLOSED"


def _mouth_bins(of3: Dict[str,str], mp: Dict[str,str], au_thr: Dict[str,float]):
    au25 = _sigmoid(_to_float(of3.get('au25') or of3.get('AU25')))
    au26 = _sigmoid(_to_float(of3.get('au26') or of3.get('AU26')))
    au12 = _sigmoid(_to_float(of3.get('au12') or of3.get('AU12')))
    jaw = _to_float(mp.get('blend::jawOpen') or mp.get('jawOpen'))
    mopen= _to_float(mp.get('blend::mouthOpen') or mp.get('mouthOpen'))
    sL = _to_float(mp.get('blend::mouthSmileLeft') or mp.get('mouthSmileLeft'))
    sR = _to_float(mp.get('blend::mouthSmileRight') or mp.get('mouthSmileRight'))

    mouth_open = max([v for v in [au25,au26,jaw,mopen] if v is not None] or [0.0])
    smile_score = max([v for v in [au12,sL,sR] if v is not None] or [0.0])

    if mouth_open >= max(0.60, au_thr.get('25',0.5)*0.9):
        m_bin = "OPEN"
    elif mouth_open >= max(0.30, au_thr.get('25',0.5)*0.6):
        m_bin = "SLIGHT"
    else:
        m_bin = "CLOSE"

    if mouth_open >= 0.50 and (smile_score >= 0.45):
        s_bin, teeth = "TEETH", True
    elif smile_score >= 0.40:
        s_bin, teeth = "CLOSED", False
    else:
        s_bin, teeth = "NONE", False

    return m_bin, s_bin, teeth, float(mouth_open), float(smile_score)

# pose bins --------------------------------------------------------------

def _gaze_bin(gy: Optional[float], gp: Optional[float]) -> str:
    if gy is None and gp is None:
        return "FRONT"
    if gp is None:
        if abs(gy) <= 12.0: return "FRONT"
        return "LEFT" if gy < 0 else "RIGHT"
    if gy is None:
        if abs(gp) <= 10.0: return "FRONT"
        return "UP" if gp > 0 else "DOWN"
    if abs(gy) <= 12.0 and abs(gp) <= 10.0: return "FRONT"
    return ("LEFT" if gy < 0 else "RIGHT") if abs(gy) >= abs(gp) else ("UP" if gp > 0 else "DOWN")


def _yaw_bin(yaw: Optional[float]) -> str:
    if yaw is None: return "FRONT"
    a = abs(yaw)
    if a <= 15: return "FRONT"
    if a <= 35: return "LEFT" if yaw < 0 else "RIGHT"
    if a <= 55: return "3/4 LEFT" if yaw < 0 else "3/4 RIGHT"
    return "PROF LEFT" if yaw < 0 else "PROF RIGHT"


def _pitch_bin(pitch: Optional[float]) -> str:
    if pitch is None: return "LEVEL"
    if pitch >= 15: return "CHIN UP"
    if pitch <= -15: return "CHIN DOWN"
    return "LEVEL"


def _quality_bin(mag: Optional[float], q_lo: float, q_hi: float) -> str:
    if mag is None: return "MID"
    if mag >= q_hi: return "HIGH"
    if mag <  q_lo: return "LOW"
    return "MID"

# ---------------- auto help ----------------
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
  --override (flag)                overwrite manifest.jsonl instead of appending
  --help on|off (on)               write /workspace/scripts/tag.txt then proceed
  -h / --h                         print this help text and continue
"""

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    ap.add_argument('--aligned', required=True)
    ap.add_argument('--logs',    required=True)
    ap.add_argument('--openface3',   default='on')
    ap.add_argument('--mediapipe',   default='on')
    ap.add_argument('--openseeface', default='on')
    ap.add_argument('--arcface',     default='on')
    ap.add_argument('--magface',     default='off')
    ap.add_argument('--override', action='store_true')
    ap.add_argument('--help', dest='auto_help', default='on')
    args = ap.parse_args()

    if getattr(args,'show_cli_help', False):
        print(AUTO_HELP)
    try:
        if _onoff(args.auto_help, True):
            sp = Path(__file__).resolve(); (sp.with_name(sp.stem + '.txt')).write_text(AUTO_HELP, encoding='utf-8')
    except Exception:
        pass

    use_of3 = _onoff(args.openface3, True)
    use_mp  = _onoff(args.mediapipe, True)
    use_osf = _onoff(args.openseeface, True)
    use_af  = _onoff(args.arcface, True)
    use_mf  = _onoff(args.magface, False)

    aligned = Path(args.aligned)
    logs    = Path(args.logs); logs.mkdir(parents=True, exist_ok=True)

    imgs = _list_images(aligned); total = len(imgs)
    if total == 0:
        print("TAG: no images under --aligned.")
        return 2

    # readers (DEEP MERGE per file)
    src_maps: Dict[str, Dict[str, Dict[str,str]]] = {}
    if use_of3:
        of3a = _read_csv_map(logs/"openface3"/"openface3.csv")
        of3b = _read_csv_map(logs/"openface3"/"of3_tags.csv")
        src_maps['of3'] = {**of3a, **{k:{**of3a.get(k,{}), **v} for k,v in of3b.items()}}
    if use_mp:
        mpa = _read_csv_map(logs/"mediapipe"/"mediapipe.csv")
        mpb = _read_csv_map(logs/"mediapipe"/"mp_tags.csv")
        src_maps['mp'] = {**mpa, **{k:{**mpa.get(k,{}), **v} for k,v in mpb.items()}}
    if use_osf:
        src_maps['osf'] = _read_csv_map(logs/"openseeface"/"openseeface.csv")
    if use_af:
        src_maps['af'] = _read_csv_map(logs/"arcface"/"arcface.csv")
    if use_mf:
        src_maps['mf'] = _read_csv_map(logs/"magface"/"magface.csv")
    src_maps['fx'] = _read_csv_map(logs/"video"/"video_stats.csv")

    # video seed (pose centering)
    def _read_seed():
        for rel in ["compile/video_stats.csv", "video/video_stats.csv", "compile/video_logs.csv", "video/video_logs.csv"]:
            p = logs/rel
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    rd = csv.DictReader(f)
                    for row in rd:
                        yield row
    yaw_vals = []
    pitch_vals = []
    of3_yaw_vals = []
    of3_pitch_vals = []
    for row in _read_seed():
        y = _to_float(row.get('pose_yaw') or row.get('yaw') or row.get('of3_yaw'))
        p = _to_float(row.get('pose_pitch') or row.get('pitch') or row.get('of3_pitch'))
        if y is not None: yaw_vals.append(y)
        if p is not None: pitch_vals.append(p)
    for row in (src_maps.get('of3') or {}).values():
        y = _to_float(row.get('pose_yaw') or row.get('yaw'))
        p = _to_float(row.get('pose_pitch') or row.get('pitch'))
        if y is not None: of3_yaw_vals.append(y)
        if p is not None: of3_pitch_vals.append(p)
    med_yaw = float(np.median(yaw_vals)) if yaw_vals else 0.0
    med_pitch = float(np.median(pitch_vals)) if pitch_vals else 0.0
    med_of3_yaw = float(np.median(of3_yaw_vals)) if of3_yaw_vals else 0.0
    med_of3_pitch = float(np.median(of3_pitch_vals)) if of3_pitch_vals else 0.0

    # identity and quality calibrations
    def _calib_id(af_map):
        arr = np.array([_to_float(v.get('cos_ref') or v.get('id_conf')) for v in (af_map or {}).values() if _to_float(v.get('cos_ref') or v.get('id_conf')) is not None], dtype=np.float64)
        if arr.size == 0: return 0.50
        return float(np.quantile(arr,0.20))
    id_thresh = _calib_id(src_maps.get('af', {})) if use_af else 0.50

    def _calib_mag(mf_map):
        arr = np.array([_to_float(v.get('quality') or v.get('mag') or v.get('mag_norm') or v.get('mag_quality')) for v in (mf_map or {}).values() if _to_float(v.get('quality') or v.get('mag') or v.get('mag_norm') or v.get('mag_quality')) is not None], dtype=np.float64)
        if arr.size == 0: return 0.0, 0.0
        return float(np.quantile(arr,0.33)), float(np.quantile(arr,0.66))
    q_lo, q_hi = _calib_mag(src_maps.get('mf', {})) if use_mf else (0.0, 0.0)

    def _calib_au(of3_map):
        vals = {k:[] for k in AU_KEYS}
        for row in (of3_map or {}).values():
            for ak in AU_KEYS:
                for col in (f'au{ak}', f'AU{ak}'):
                    if col in row:
                        v = _sigmoid(_to_float(row.get(col)))
                        if v is not None:
                            vals[ak].append(float(np.clip(v,0.0,1.0)))
                        break
        thr = {}
        for ak in AU_KEYS:
            arr = np.array(vals[ak], dtype=np.float64)
            thr[ak] = float(np.quantile(arr,0.80)) if arr.size>0 else 0.50
        return thr
    au_thr = _calib_au(src_maps.get('of3', {})) if use_of3 else {k:0.5 for k in AU_KEYS}

    (logs/"tag").mkdir(parents=True, exist_ok=True)
    try:
        (logs/"tag"/"calib.json").write_text(json.dumps({
            'yaw_median_deg': med_yaw,
            'pitch_median_deg': med_pitch,
            'of3_yaw_median_deg': med_of3_yaw,
            'of3_pitch_median_deg': med_of3_pitch,
            'id_threshold': id_thresh,
            'mag_tertiles': {'low': q_lo, 'high': q_hi},
            'au_thresholds': au_thr,
            'temp_thresholds': {'warm': 3700, 'cool': 5500},
        }, indent=2), encoding='utf-8')
    except Exception:
        pass

    # outputs
    manifest = logs/"manifest.jsonl"
    if args.override and manifest.exists():
        try: manifest.unlink()
        except Exception: pass

    def init_counts():
        return {
            'GAZE':{'counts':{k:0 for k in GAZE_LABELS}},
            'EYES':{'counts':{k:0 for k in EYES_LABELS}},
            'MOUTH':{'counts':{k:0 for k in MOUTH_LABELS}},
            'SMILE':{'counts':{k:0 for k in SMILE_LABELS}},
            'EMOTION':{'counts':{k:0 for k in EMOTION_LABELS}},
            'YAW':{'counts':{k:0 for k in YAW_LABELS}},
            'PITCH':{'counts':{k:0 for k in PITCH_LABELS}},
            'IDENTITY':{'counts':{k:0 for k in IDENTITY_LABELS}},
            'QUALITY':{'counts':{k:0 for k in QUALITY_LABELS}},
            'TEMP':{'counts':{k:0 for k in TEMP_LABELS}},
            'EXPOSURE':{'counts':{k:0 for k in EXPOSURE_LABELS}},
        }
    sections = init_counts()

    bar = ProgressBar('TAG', total=total, show_fail_label=True)
    processed = 0; fails = 0; t0 = time.time()

    tags_csv = logs/"tags.csv"
    with tags_csv.open("w", encoding="utf-8", newline="") as fcsv, manifest.open("a", encoding="utf-8") as outm:
        wr = csv.writer(fcsv)
        wr.writerow(["file","pose_bin","eyes_bin","mouth_bin","smile_bin","emotion_bin","yaw_bin","pitch_bin",
                     "id_bin","quality_bin","temp_bin","exposure_bin","cleanup_ok","reasons"])  # stable header

        for img in imgs:
            fn = img.name
            try:
                of3 = (src_maps.get('of3',{}) or {}).get(fn, {})
                mp  = (src_maps.get('mp',{})  or {}).get(fn, {})
                osf = (src_maps.get('osf',{}) or {}).get(fn, {})
                af  = (src_maps.get('af',{})  or {}).get(fn, {})
                mfq = (src_maps.get('mf',{})  or {}).get(fn, {})
                fx  = (src_maps.get('fx',{})  or {}).get(fn, {})

                # identity/quality
                id_score = _to_float(af.get('cos_ref') or af.get('id_conf'))
                mag = _to_float(mfq.get('quality') or mfq.get('mag') or mfq.get('mag_norm') or mfq.get('mag_quality'))
                id_bin = "MATCH" if (id_score is not None and id_score >= id_thresh) else "MISMATCH"
                q_bin = _quality_bin(mag, q_lo, q_hi)

                # pose
                yaw   = _to_float(osf.get('yaw')   or of3.get('pose_yaw')   or mp.get('yaw'))
                pitch = _to_float(osf.get('pitch') or of3.get('pose_pitch') or mp.get('pitch'))
                roll  = _to_float(osf.get('roll')  or of3.get('pose_roll')  or mp.get('roll'))
                gyaw_deg   = _to_float(of3.get('gaze_yaw')   or of3.get('gaze_yaw_deg')   or osf.get('gaze_yaw'))
                gpitch_deg = _to_float(of3.get('gaze_pitch') or of3.get('gaze_pitch_deg') or osf.get('gaze_pitch'))

                # center by dataset medians (favor OF3 medians when present)
                yaw_center   = yaw   - (med_of3_yaw   if of3_yaw_vals else med_yaw)
                pitch_center = pitch - (med_of3_pitch if of3_pitch_vals else med_pitch)
                yaw_bin = _yaw_bin(yaw_center)
                pitch_bin = _pitch_bin(pitch_center)

                # gaze/eyes/mouth/smile
                gaze_bin = _gaze_bin(gyaw_deg, gpitch_deg)
                eyes_bin = _eyes_state(mp, osf)
                m_bin, s_bin, teeth, mouth_val, smile_val = _mouth_bins(of3, mp, au_thr)

                # emotion
                emo_lbl = (of3.get('emotion') or '').strip().upper()
                if emo_lbl not in EMOTION_LABELS: emo_lbl = 'NEUTRAL'

                # temp/exposure (face-only)
                cct, temp_bin, exp_bin = None, 'NEUTRAL', 'NORMAL'
                try:
                    bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
                    if isinstance(bgr, np.ndarray) and bgr.size > 0:
                        cct, temp_bin, exp_bin = _estimate_temp_exposure(bgr)
                except Exception:
                    pass

                # cleanup flags (placeholder — decisions-only policy)
                cleanup_ok, reasons = True, ""

                # update counts for live table
                for k, v in [("GAZE",gaze_bin),("EYES",eyes_bin),("MOUTH",m_bin),("SMILE",s_bin),("EMOTION",emo_lbl),
                             ("YAW",yaw_bin),("PITCH",pitch_bin),("IDENTITY",id_bin),("QUALITY",q_bin),
                             ("TEMP",temp_bin),("EXPOSURE",exp_bin)]:
                    sections[k]['counts'][v] += 1

                wr.writerow([fn, gaze_bin, eyes_bin, m_bin, s_bin, emo_lbl, yaw_bin, pitch_bin,
                             id_bin, q_bin, temp_bin, exp_bin, int(cleanup_ok), reasons])

                # compute a simple bin score for preview/match hints (not a gate)
                pose_center = float(np.clip(1.0 - (abs(yaw_center)/55.0 + abs(pitch_center)/25.0), 0.0, 1.0))
                expr_bonus = float(smile_val)
                pnp = _to_float(osf.get('pnp_error')) or 0.0
                pnp_norm = min(1.0, pnp/0.02)
                q_norm = float(np.clip((mag - q_lo)/max(1e-9,(q_hi-q_lo)), 0.0, 1.0)) if (use_mf and mag is not None and q_hi>q_lo) else 0.5
                id_s = float(id_score) if id_score is not None else 0.55
                bin_score = 0.55*id_s + 0.30*q_norm + 0.10*pose_center + 0.05*expr_bonus - 0.05*pnp_norm

                rec = {
                    'file': fn,
                    'bins': {
                        'GAZE': gaze_bin,
                        'EYES': eyes_bin,
                        'MOUTH': m_bin,
                        'SMILE': s_bin,
                        'EMOTION': emo_lbl,
                        'YAW': yaw_bin,
                        'PITCH': pitch_bin,
                        'IDENTITY': id_bin,
                        'QUALITY': q_bin,
                        'TEMP': temp_bin,
                        'EXPOSURE': exp_bin,
                    },
                    'scores': {
                        'id_score': id_score,
                        'q_magface': mag,
                        'pose_centering': pose_center,
                        'expr_strength': smile_val,
                        'pnp_error': pnp,
                        'bin_score': bin_score,
                    },
                    'pose': {'yaw': yaw, 'pitch': pitch, 'roll': roll},
                    'gaze_deg': {'yaw': gyaw_deg, 'pitch': gpitch_deg},
                    'src': {'of3': of3, 'mp': mp, 'osf': osf, 'af': af, 'mf': mfq, 'fx': fx},
                }

                outm.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # live table
                elapsed = time.time() - t0
                fps = int(processed/elapsed) if elapsed > 0 else 0
                eta = int((total-processed)/max(1, fps))
                eta_s = f"{eta//60:02d}:{eta%60:02d}"
                tbl = render_bin_table('TAG', {'processed':processed,'total':total,'fails':fails}, sections,
                                       dupe_totals=None, dedupe_on=True, fps=fps, eta=eta_s)
                write_bin_table(logs, tbl)
                bar.update(processed, fails=fails, fps=fps)

            except Exception as e:
                fails += 1
                bar.update(processed, fails=fails)
                # print the first error for visibility; keep quiet after that
                if fails == 1:
                    print(f"TAG: error on {fn}: {e}")
            finally:
                processed += 1

    bar.close()
    print(f"TAG: done. images={total} fails={fails} manifest={manifest}")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
