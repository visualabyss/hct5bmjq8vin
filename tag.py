#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, csv, json, time, argparse, math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
from progress import ProgressBar
from bin_table import render_bin_table, write_bin_table
IMAGE_EXTS = {".png",".jpg",".jpeg",".bmp",".webp"}
GAZE_LABELS     = ["FRONT","LEFT","RIGHT","UP","DOWN"]
EYES_LABELS     = ["OPEN","HALF","CLOSED","W-LEFT","W-RIGHT"]
MOUTH_LABELS    = ["OPEN","SLIGHT","CLOSE"]
SMILE_LABELS    = ["TEETH","CLOSED","NONE"]
EMOTION_LABELS  = ["NEUTRAL","HAPPY","SAD","SURPRISE","FEAR","DISGUST","ANGER","CONTEMPT"]
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

def _maybe_rad_to_deg(v: Optional[float]) -> Optional[float]:
    if v is None: return None
    a = abs(v)
    if a <= math.pi + 0.05:
        return float(v * 180.0 / math.pi)
    return float(v)

def _tenengrad(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    if max(h, w) > 512:
        scale = 512.0 / max(h, w)
        g = cv2.resize(g, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx*gx + gy*gy
    return float(np.mean(fm))

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)

def _estimate_cct_xy(x: float, y: float) -> float:
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (y - ye + 1e-9)
    return float(np.clip(449.0*(n**3) + 3525.0*(n**2) + 6823.3*n + 5520.33, 1000.0, 25000.0))

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
    temp_bin = "WARM" if cct < 4200 else ("NEUTRAL" if cct <= 6000 else "COOL")
    L = 0.2126*lin[...,0] + 0.7152*lin[...,1] + 0.0722*lin[...,2]
    Lm = L[m]
    log_avg = float(math.exp(np.log(Lm+1e-6).mean()))
    sdr = bgr.astype(np.float64)/255.0
    low = float(((sdr[...,0][m] < 0.02).mean() + (sdr[...,1][m] < 0.02).mean() + (sdr[...,2][m] < 0.02).mean())*100/3)
    hi  = float(((sdr[...,0][m] > 0.98).mean() + (sdr[...,1][m] > 0.98).mean() + (sdr[...,2][m] > 0.98).mean())*100/3)
    exp_bin = "OVER" if (hi > 0.05 and log_avg > 0.60) else ("UNDER" if (low > 8.0 and log_avg < 0.10) else "NORMAL")
    return cct, temp_bin, exp_bin

def _eyes_state(mp: Dict[str,str], osf: Dict[str,str]) -> str:
    def gs(k: str) -> Optional[float]:
        return _to_float(mp.get(f'blend::{k}') or mp.get(k))
    ebl = gs('eyeBlinkLeft');   ebr = gs('eyeBlinkRight')
    ewl = gs('eyeWideLeft');    ewr = gs('eyeWideRight')
    esl = gs('eyeSquintLeft');  esr = gs('eyeSquintRight')
    bl = _to_float(osf.get('blink_l')); br = _to_float(osf.get('blink_r'))
    open_l = 1.0 - ebl if ebl is not None else None
    open_r = 1.0 - ebr if ebr is not None else None
    if bl is not None and bl > 0.60: open_l = min(open_l or 0.0, 0.1)
    if br is not None and br > 0.60: open_r = min(open_r or 0.0, 0.1)
    if open_l is None or open_r is None:
        return "OPEN"
    diff = (open_l - open_r)
    avg  = 0.5*(open_l + open_r)
    if abs(diff) >= 0.35 and ((open_l > 0.35) or (open_r > 0.35)):
        return "W-LEFT" if diff < 0 else "W-RIGHT"
    wide = ((ewl or 0.0) + (ewr or 0.0)) * 0.5
    squi = ((esl or 0.0) + (esr or 0.0)) * 0.5
    if avg >= 0.55 or wide >= 0.30:
        return "OPEN"
    if avg >= 0.30 and squi < 0.65:
        return "HALF"
    return "CLOSED"

def _gaze_from_mp(mp: Dict[str,str]) -> Optional[str]:
    def g(k: str) -> float:
        v = _to_float(mp.get(f'blend::{k}') or mp.get(k))
        return float(v) if v is not None else 0.0
    L_up   = g('eyeLookUpLeft');    R_up   = g('eyeLookUpRight')
    L_dn   = g('eyeLookDownLeft');  R_dn   = g('eyeLookDownRight')
    L_in   = g('eyeLookInLeft');    R_in   = g('eyeLookInRight')
    L_out  = g('eyeLookOutLeft');   R_out  = g('eyeLookOutRight')
    left_s  = L_out + R_in
    right_s = L_in  + R_out
    up_s    = L_up  + R_up
    down_s  = L_dn  + R_dn
    scores = {'LEFT':left_s, 'RIGHT':right_s, 'UP':up_s, 'DOWN':down_s}
    mdir = max(scores, key=scores.get)
    mval = scores[mdir]
    if mval < 0.20:
        return "FRONT"
    return mdir

def _mouth_bins(mp: Dict[str,str], jaw_q: Tuple[float,float]):
    def gs(k: str) -> Optional[float]:
        return _to_float(mp.get(f'blend::{k}') or mp.get(k))
    jaw = gs('jawOpen')
    sL  = gs('mouthSmileLeft')
    sR  = gs('mouthSmileRight')
    j_lo, j_hi = jaw_q
    j = jaw if jaw is not None else 0.0
    if j >= max(0.35, j_hi*0.95):
        m_bin = "OPEN"
    elif j >= max(0.20, j_lo):
        m_bin = "SLIGHT"
    else:
        m_bin = "CLOSE"
    smile_score = max([v for v in [sL,sR] if v is not None] or [0.0])
    if m_bin == "OPEN" and smile_score >= 0.45:
        s_bin, teeth = "TEETH", True
    elif smile_score >= 0.40:
        s_bin, teeth = "CLOSED", False
    else:
        s_bin, teeth = "NONE", False
    return m_bin, s_bin, teeth, float(j), float(smile_score)

def _gaze_bin(gy_deg: Optional[float], gp_deg: Optional[float]) -> str:
    if gy_deg is None and gp_deg is None:
        return "FRONT"
    if gp_deg is None:
        if abs(gy_deg) <= 8.0: return "FRONT"
        return "LEFT" if gy_deg < 0 else "RIGHT"
    if gy_deg is None:
        if abs(gp_deg) <= 8.0: return "FRONT"
        return "UP" if gp_deg > 0 else "DOWN"
    if abs(gy_deg) <= 8.0 and abs(gp_deg) <= 8.0: return "FRONT"
    return ("LEFT" if gy_deg < 0 else "RIGHT") if abs(gy_deg) >= abs(gp_deg) else ("UP" if gp_deg > 0 else "DOWN")

def _yaw_bin(yaw_deg: Optional[float]) -> str:
    if yaw_deg is None: return "FRONT"
    a = abs(yaw_deg)
    if a < 8: return "FRONT"
    if a < 25: return "LEFT" if yaw_deg < 0 else "RIGHT"
    if a < 45: return "3/4 LEFT" if yaw_deg < 0 else "3/4 RIGHT"
    return "PROF LEFT" if yaw_deg < 0 else "PROF RIGHT"

def _pitch_bin(pitch_deg: Optional[float]) -> str:
    if pitch_deg is None: return "LEVEL"
    if pitch_deg >= 15: return "CHIN UP"
    if pitch_deg <= -15: return "CHIN DOWN"
    return "LEVEL"

def _quality_bin_from_score(qs: float) -> str:
    if qs >= 0.75: return "HIGH"
    if qs <= -0.75: return "LOW"
    return "MID"

AUTO_HELP = r"""
--aligned DIR  --logs DIR  [--openface3 on|off] [--mediapipe on|off] [--openseeface on|off]
[--arcface on|off] [--magface on|off] [--override] [--help on|off] [-h|--h]
"""

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
    mp_jaw_list: List[float] = []
    of3_y_deg: List[float] = []
    mp_y_deg: List[float] = []
    osf_y: List[float] = []
    for row in (src_maps.get('mp') or {}).values():
        j = _to_float(row.get('jawOpen') or row.get('blend::jawOpen'))
        if j is not None: mp_jaw_list.append(float(j))
        y = _maybe_rad_to_deg(_to_float(row.get('yaw')))
        if y is not None: mp_y_deg.append(float(y))
    for row in (src_maps.get('of3') or {}).values():
        y = _maybe_rad_to_deg(_to_float(row.get('pose_yaw') or row.get('yaw')))
        if y is not None: of3_y_deg.append(float(y))
    for row in (src_maps.get('osf') or {}).values():
        y = _to_float(row.get('yaw'))
        if y is not None: osf_y.append(float(y))
    if len(mp_jaw_list) > 0:
        jaw_q_lo = float(np.quantile(mp_jaw_list, 0.30))
        jaw_q_hi = float(np.quantile(mp_jaw_list, 0.60))
    else:
        jaw_q_lo, jaw_q_hi = 0.20, 0.35
    def yaw_has_both_signs(vals: List[float]) -> bool:
        if not vals: return False
        return (min(vals) < -1.0) and (max(vals) > 1.0)
    prefer_of3_yaw = yaw_has_both_signs(of3_y_deg)
    if not prefer_of3_yaw and yaw_has_both_signs(mp_y_deg):
        prefer_of3_yaw = False
    id_vals = []
    for v in (src_maps.get('af', {}) or {}).values():
        s = _to_float(v.get('id_score') or v.get('cos_ref') or v.get('id_conf') or v.get('cos_sim') or v.get('similarity') or v.get('cosine'))
        if s is not None: id_vals.append(float(s))
    if id_vals:
        id_mode = 'sim'
        id_thresh = float(np.quantile(id_vals, 0.20))
    else:
        id_mode = 'sim'; id_thresh = 0.50
    mag_vals = []
    for v in (src_maps.get('mf', {}) or {}).values():
        q = _to_float(v.get('quality') or v.get('mag') or v.get('mag_norm') or v.get('mag_quality'))
        if q is not None: mag_vals.append(float(q))
    mag_mu = float(np.mean(mag_vals)) if mag_vals else 0.0
    mag_sd = float(np.std(mag_vals)+1e-6) if mag_vals else 1.0
    (logs/"tag").mkdir(parents=True, exist_ok=True)
    try:
        (logs/"tag"/"calib.json").write_text(json.dumps({
            'jaw_quantiles': {'q30': jaw_q_lo, 'q60': jaw_q_hi},
            'id_mode': id_mode, 'id_threshold': id_thresh,
            'mag_stats': {'mean': mag_mu, 'std': mag_sd},
            'prefer_of3_yaw': prefer_of3_yaw,
        }, indent=2), encoding='utf-8')
    except Exception:
        pass
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
    tag_dir = logs/"tag"; tag_dir.mkdir(parents=True, exist_ok=True)
    tags_csv = tag_dir/"tags.csv"
    img_list = _list_images(aligned)
    with tags_csv.open("w", encoding="utf-8", newline="") as fcsv, manifest.open("a", encoding="utf-8") as outm:
        wr = csv.writer(fcsv)
        wr.writerow(["file","gaze_bin","eyes_bin","mouth_bin","smile_bin","emotion_bin","yaw_bin","pitch_bin","id_bin","quality_bin","temp_bin","exposure_bin","cleanup_ok","reasons"])
        for img in img_list:
            fn = img.name
            try:
                of3 = (src_maps.get('of3',{}) or {}).get(fn, {})
                mp  = (src_maps.get('mp',{})  or {}).get(fn, {})
                osf = (src_maps.get('osf',{}) or {}).get(fn, {})
                af  = (src_maps.get('af',{})  or {}).get(fn, {})
                mfq = (src_maps.get('mf',{})  or {}).get(fn, {})
                fx  = (src_maps.get('fx',{})  or {}).get(fn, {})
                raw_id = _to_float(af.get('id_score') or af.get('cos_ref') or af.get('id_conf') or af.get('cos_sim') or af.get('similarity') or af.get('cosine'))
                if id_mode == 'sim':
                    id_bin = "MATCH" if (raw_id is not None and raw_id >= id_thresh) else "MISMATCH"
                    id_score = raw_id
                else:
                    id_bin = "MATCH" if (raw_id is not None and raw_id <= id_thresh) else "MISMATCH"
                    id_score = (1.0 - float(raw_id or 1.0))
                mag = _to_float(mfq.get('quality') or mfq.get('mag') or mfq.get('mag_norm') or mfq.get('mag_quality'))
                z_mag = ((mag - mag_mu)/mag_sd) if (use_mf and mag is not None) else 0.0
                bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
                z_ten = (_tenengrad(bgr) - 0.0) / 1.0
                q_score = 0.60*z_mag + 0.40*z_ten
                q_bin = _quality_bin_from_score(q_score)
                yaw_of3   = _maybe_rad_to_deg(_to_float(of3.get('pose_yaw')   or of3.get('yaw')))
                pitch_of3 = _maybe_rad_to_deg(_to_float(of3.get('pose_pitch') or of3.get('pitch')))
                roll_of3  = _maybe_rad_to_deg(_to_float(of3.get('pose_roll')  or of3.get('roll')))
                yaw_mp    = _maybe_rad_to_deg(_to_float(mp.get('yaw')))
                pitch_mp  = _maybe_rad_to_deg(_to_float(mp.get('pitch')))
                roll_mp   = _maybe_rad_to_deg(_to_float(mp.get('roll')))
                yaw_osf   = _to_float(osf.get('yaw'))
                pitch_osf = _to_float(osf.get('pitch'))
                roll_osf  = _to_float(osf.get('roll'))
                if prefer_of3_yaw and yaw_of3 is not None:
                    yaw = yaw_of3
                else:
                    yaw = yaw_mp if yaw_mp is not None else (yaw_osf if yaw_osf is not None else yaw_of3)
                pitch = pitch_of3 if pitch_of3 is not None else (pitch_mp if pitch_mp is not None else pitch_osf)
                roll  = roll_of3 if roll_of3 is not None else (roll_mp if roll_mp is not None else roll_osf)
                gyaw_deg   = None
                gpitch_deg = None
                g_mp = _gaze_from_mp(mp)
                if g_mp is not None:
                    gaze_bin = g_mp
                else:
                    gaze_bin = _gaze_bin(gyaw_deg, gpitch_deg)
                eyes_bin = _eyes_state(mp, osf)
                m_bin, s_bin, teeth, mouth_val, smile_val = _mouth_bins(mp, (jaw_q_lo, jaw_q_hi))
                emo_lbl = (of3.get('emotion') or '').strip().upper()
                if emo_lbl not in EMOTION_LABELS: emo_lbl = 'NEUTRAL'
                cct, temp_bin, exp_bin = None, 'NEUTRAL', 'NORMAL'
                try:
                    if bgr is not None and bgr.size > 0:
                        cct, temp_bin, exp_bin = _estimate_temp_exposure(bgr)
                except Exception:
                    pass
                cleanup_ok, reasons = True, ""
                for k, v in [("GAZE",gaze_bin),("EYES",eyes_bin),("MOUTH",m_bin),("SMILE",s_bin),("EMOTION",emo_lbl),
                             ("YAW",_yaw_bin(yaw)),("PITCH",_pitch_bin(pitch)),("IDENTITY",id_bin),("QUALITY",q_bin),
                             ("TEMP",temp_bin),("EXPOSURE",exp_bin)]:
                    sections[k]['counts'][v] += 1
                wr.writerow([fn, gaze_bin, eyes_bin, m_bin, s_bin, emo_lbl, _yaw_bin(yaw), _pitch_bin(pitch), id_bin, q_bin, temp_bin, exp_bin, int(cleanup_ok), reasons])
                pose_center = float(np.clip(1.0 - (abs((yaw or 0.0))/55.0 + abs((pitch or 0.0))/25.0), 0.0, 1.0))
                expr_bonus = float(smile_val)
                pnp = _to_float(osf.get('pnp_error')) or 0.0
                pnp_norm = min(1.0, pnp/0.02)
                id_s = float(id_score) if id_score is not None else 0.55
                bin_score = 0.55*id_s + 0.30*max(-1.0,min(1.0,q_score)) + 0.10*pose_center + 0.05*expr_bonus - 0.05*pnp_norm
                rec = {
                    'file': fn,
                    'bins': {
                        'GAZE': gaze_bin,
                        'EYES': eyes_bin,
                        'MOUTH': m_bin,
                        'SMILE': s_bin,
                        'EMOTION': emo_lbl,
                        'YAW': _yaw_bin(yaw),
                        'PITCH': _pitch_bin(pitch),
                        'IDENTITY': id_bin,
                        'QUALITY': q_bin,
                        'TEMP': temp_bin,
                        'EXPOSURE': exp_bin,
                    },
                    'scores': {
                        'id_score': id_score,
                        'q_score': q_score,
                        'pose_centering': pose_center,
                        'expr_strength': smile_val,
                        'pnp_error': pnp,
                        'bin_score': bin_score,
                    },
                    'pose_deg': {'yaw': yaw, 'pitch': pitch, 'roll': roll},
                }
                outm.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
