#!/usr/bin/env python3
# Curator v4.9 (single-line progress; aligned final summary; fixed-width separators)
# Subcommands: tag, preview, match, build, remove, stats

import argparse, csv, json, math, os, sys, random, shutil, time
from pathlib import Path
from collections import defaultdict, Counter

# Optional deps
try:
    import numpy as np
except Exception:
    np = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image
except Exception:
    Image = None

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ================= utils =================
def list_images(d):
    p = Path(d)
    files = [x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def sha1_of_file(p):
    import hashlib
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_jsonl(rows, fp):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(fp):
    rows = []
    p = Path(fp)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

# ============== terminal formatting ==============
# EXACT divider width requested:
SEP = "====================================================================================================="
SUB = "-" * len(SEP)

# Column layout (kept constant across TAG + summary rows)
# [LABEL:8] '|' [BAR:28] '|  ' [PCT:6] '  ' [COUNT:13] '  ' ['UNKNOWN':7] '  ' [UNK_CNT:7] rest...
LAB_W = 8
BAR_W = 28
PCT_W = 6    # "100.0%"
CNT_W = 13   # "12345/67890"
UNK_L = 7    # "UNKNOWN"
UNK_N = 7

def _term_cols():
    try:
        return shutil.get_terminal_size(fallback=(120, 20)).columns
    except Exception:
        return 120

def _clamp_cols(s):
    cols = _term_cols()
    if len(s) < cols:
        return s + " " * (cols - len(s) - 1)
    return s[:max(1, cols-1)]

def _render_bar(ratio, width):
    ratio = max(0.0, min(1.0, float(ratio)))
    full = int(width * ratio)
    frac = width * ratio - full
    bar = "█" * max(0, min(full, width))
    partials = "▏▎▍▌▋▊▉"
    if full < width:
        idx = int(frac * len(partials))
        if idx > 0:
            bar += partials[idx-1]; full += 1
    if full < width:
        bar += " " * (width - full)
    return bar

def _fmt_eta(secs):
    if not math.isfinite(secs): return "--:--"
    secs = max(0, int(secs))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return (f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")

def _line_header(label, ratio, done, total, unk_count, fps=None, eta=None):
    # Build a single line with fixed columns; fps/eta are only shown for TAG
    bar = _render_bar(ratio, BAR_W)
    pct = f"{100.0*ratio:5.1f}%"
    cnt = f"{int(done)}/{int(total)}"
    # segments with padding
    s = ""
    s += f"{label:<{LAB_W}}|{bar}|  "
    s += f"{pct:>{PCT_W}}  "
    s += f"{cnt:>{CNT_W}}  "
    s += f"{'UNKNOWN':>{UNK_L}}  "
    s += f"{int(unk_count):>{UNK_N}}"
    if fps is not None or eta is not None:
        s += "  "
        if fps is not None:
            s += f"{float(fps):6.1f} fps"
        if eta is not None:
            s += "   ETA " + _fmt_eta(float(eta))
    return _clamp_cols(s)

def print_tag_progress(i, total, t0, unknown_running):
    elapsed = max(1e-6, time.time()-t0)
    fps = i / elapsed
    eta = (total - i)/fps if fps > 0 else float("inf")
    ratio = i / max(1, total)
    sys.stdout.write("\r" + _line_header("TAG", ratio, i, total, unknown_running, fps=fps, eta=eta))
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

def print_summary_table(C, total):
    # C: dict of counters (yaw, pitch, eyes, mouth, smile, gaze, light, expo)
    # Compute unknowns and known counts per category
    def known_and_unknown(counter: Counter, known_keys):
        known = sum(int(counter.get(k, 0)) for k in known_keys)
        unk = int(max(0, total - known))
        ratio = known / float(max(1, total))
        return known, unk, ratio

    # order + pretty names + sub-keys
    groups = [
        ("GAZE",  "gaze",  ["front","left","right","up","down"]),
        ("EYES",  "eyes",  ["open","half","closed"]),
        ("MOUTH", "mouth", ["wide","open","slight","closed"]),
        ("SMILE", "smile", ["smile_teeth","smile_closed","none"]),
        ("YAW",   "yaw",   ["front","slight_l","slight_r","strong_l","strong_r"]),
        ("PITCH", "pitch", ["front","up","down"]),
        ("TEMP",  "light", ["warm","neutral","cool"]),
        ("EXPOSURE","expo",["under","normal","over"]),
    ]

    print(SEP)
    for (title, key, subkeys) in groups:
        counter = C.get(key, Counter())
        known, unk, ratio = known_and_unknown(counter, subkeys)
        sys.stdout.write(_line_header(title, ratio, known, total, unk) + "\n")
        sys.stdout.write(SUB + "\n")
        # sub-bins: just counts, aligned under the UNKNOWN-count column
        for sk in subkeys:
            cnt = int(counter.get(sk, 0))
            label = sk.upper()
            left = f"{label:<{LAB_W}}|"  # keep vertical alignment
            right_pad = " " * (BAR_W + 2 + PCT_W + 2 + CNT_W + 2 + UNK_L + 2)
            sys.stdout.write(_clamp_cols(left + right_pad + f"{cnt:>{UNK_N}}") + "\n")
        print(SEP)

# ============== color / CCT ==============
def srgb_to_linear_img(img):
    a = 0.055
    low = img <= 0.04045
    out = img.copy()
    out[low]  = img[low] / 12.92
    out[~low] = ((np.maximum(img[~low], 1e-6) + a) / (1.0 + a)) ** 2.4
    return out

def xyz_to_cct_mccamy(xyz):
    X, Y, Z = xyz
    S = X + Y + Z
    if S <= 0:
        return None
    x = X / S; y = Y / S
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (y - ye + 1e-9)
    cct = 449*(n**3) + 3525*(n**2) + 6823.3*n + 5520.33
    try:
        if not math.isfinite(cct):
            return None
    except Exception:
        pass
    return float(cct)

def _imread_any_bgr(fp):
    # Robust read using cv2 if available; otherwise PIL
    if cv2 is not None and np is not None:
        try:
            data = np.fromfile(str(fp), dtype="uint8")
        except Exception:
            data = None
        im = cv2.imdecode(data, cv2.IMREAD_COLOR) if (data is not None) else cv2.imread(str(fp), cv2.IMREAD_COLOR)
        return im
    # PIL fallback
    if Image is not None and np is not None:
        try:
            im = Image.open(fp).convert("RGB")
            arr = np.asarray(im, dtype="uint8")  # RGB
            return arr[..., ::-1]  # to BGR
        except Exception:
            return None
    return None

def estimate_cct_from_image_bgr(img_bgr):
    if np is None:
        return None
    p = 6.0
    eps = 1e-6
    rgb = img_bgr[..., ::-1].astype("float32") / 255.0  # BGR->RGB
    rgb_lin = srgb_to_linear_img(rgb)
    R = rgb_lin[...,0]; G = rgb_lin[...,1]; B = rgb_lin[...,2]
    def minkowski(a): return (np.mean(np.power(np.maximum(a, 0.0), p)) + eps) ** (1.0/p)
    Rm, Gm, Bm = minkowski(R), minkowski(G), minkowski(B)
    v = np.array([Rm, Gm, Bm], dtype="float64")
    v = v / (float(v.max()) + eps)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype="float64")
    xyz = v @ M.T
    return xyz_to_cct_mccamy(xyz)

def bin_cct(cct, cool_k, warm_k, flip=0, neutral_pad=0.0):
    if cct is None:
        return "unknown"
    c_lo = float(cool_k) + float(neutral_pad)
    w_hi = float(warm_k) - float(neutral_pad)
    if int(flip):
        if cct < c_lo: return "cool"
        if cct > w_hi: return "warm"
        return "neutral"
    else:
        if cct < c_lo: return "warm"
        if cct > w_hi: return "cool"
        return "neutral"

# ============== exposure ==============
def center_mask(h, w, scale=0.70):
    if np is None:
        return None
    Y, X = np.ogrid[:h, :w]
    cy, cx = float(h)/2.0, float(w)/2.0
    ry, rx = (h*scale)/2.0, (w*scale)/2.0
    return (((Y - cy)/ry)**2 + ((X - cx)/rx)**2) <= 1.0

def exposure_metrics(img_bgr, center_weighted=True, center_scale=0.70):
    if np is None:
        return 0.5, 0.0, 0.0
    # Luma from sRGB
    rgb = img_bgr[..., ::-1].astype("float32") / 255.0  # to RGB 0..1
    L = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]  # BT.709
    if center_weighted:
        m = center_mask(L.shape[0], L.shape[1], scale=center_scale)
        if m is not None and m.sum() > 0:
            L = L[m]
    mean_L = float(np.mean(L))
    shadow_pct = float(np.mean(L < 0.05))
    highlight_pct = float(np.mean(L > 0.95))
    return mean_L, shadow_pct, highlight_pct

def bin_exposure(mean_L, sh_pct, hi_pct, mean_low, mean_high, sh_thr, hi_thr):
    if (mean_L < mean_low) or (sh_pct > sh_thr):
        return "under"
    if (mean_L > mean_high) or (hi_pct > hi_thr):
        return "over"
    return "normal"

# ============== OpenFace helpers ==============
def of_glob_csv(root):
    rows = []
    for f in Path(root).rglob("*.csv"):
        try:
            with f.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
                rd = csv.DictReader(fh)
                for r in rd:
                    r["_of_src"] = str(f)
                    rows.append(r)
        except Exception:
            pass
    return rows

def of_key_from_row(r):
    for k in ("filename","file","name","img","image","input","source","frame_name"):
        if k in r and r[k]:
            return Path(str(r[k])).name
    if "frame" in r and r["frame"] not in (None,""):
        try:
            fr = int(float(r["frame"]))
            return "{:06d}.png".format(fr)
        except Exception:
            return None
    return None

def of_build_map(rows):
    m = {}
    for r in rows:
        key = of_key_from_row(r)
        if key:
            m[key] = r
            continue
        src = r.get("_of_src")
        if src:
            base = Path(src).stem
            for ext in IMG_EXTS:
                m[base+ext] = r
    return m

def of_float(r, k, default=None):
    try:
        return float(r.get(k, default))
    except Exception:
        return default

def of_int(r, k, default=None):
    try:
        return int(float(r.get(k, default)))
    except Exception:
        return default

def of_calibrate(of_rows, conf_floor, lo_q, hi_q):
    vals = {"AU12_r":[], "AU25_r":[], "AU26_r":[], "AU45_r":[]}
    for r in of_rows:
        conf = of_float(r, "confidence", 1.0)
        succ = of_int(r, "success", 1)
        if conf is not None and conf < conf_floor:
            continue
        if succ is not None and succ == 0:
            continue
        for k in vals.keys():
            v = of_float(r, k, None)
            if v is None:
                continue
            try:
                if math.isfinite(v):
                    vals[k].append(v)
            except Exception:
                vals[k].append(v)
    def quant(bag, q):
        if not bag:
            return None
        vs = sorted(bag)
        idx = int(round(q * (len(vs)-1)))
        if idx < 0: idx = 0
        if idx >= len(vs): idx = len(vs)-1
        return vs[idx]
    return {
        "AU12_lo": quant(vals["AU12_r"], lo_q), "AU12_hi": quant(vals["AU12_r"], hi_q),
        "AU25_lo": quant(vals["AU25_r"], lo_q), "AU25_hi": quant(vals["AU25_r"], hi_q),
        "AU26_lo": quant(vals["AU26_r"], lo_q), "AU26_hi": quant(vals["AU26_r"], hi_q),
        "AU45_lo": quant(vals["AU45_r"], lo_q), "AU45_hi": quant(vals["AU45_r"], hi_q),
    }

def eyes_from_of(of_row, calib, pose_gate_yaw, pose_gate_pitch, yaw_deg=None, pitch_deg=None, eyes_margin=0.05):
    try:
        if yaw_deg is not None and abs(float(yaw_deg)) > float(pose_gate_yaw): return "unknown"
        if pitch_deg is not None and abs(float(pitch_deg)) > float(pose_gate_pitch): return "unknown"
    except Exception:
        pass
    a45 = of_float(of_row, "AU45_r", None)
    lo = calib.get("AU45_lo"); hi = calib.get("AU45_hi")
    if a45 is None or lo is None or hi is None:
        return "unknown"
    if a45 >= (float(hi) + float(eyes_margin)):
        return "closed"
    if a45 <= (float(lo) - float(eyes_margin)):
        return "open"
    return "half"

def mouth_state_from_of(of_row, calib, pose_gate_yaw, pose_gate_pitch,
                        yaw_deg=None, pitch_deg=None,
                        open_q=0.65, wide_q=0.90,
                        margin_close=0.03, margin_open=0.03, margin_wide=0.05):
    try:
        if yaw_deg is not None and abs(float(yaw_deg)) > float(pose_gate_yaw): return "unknown"
        if pitch_deg is not None and abs(float(pitch_deg)) > float(pose_gate_pitch): return "unknown"
    except Exception:
        pass
    a25 = of_float(of_row, "AU25_r", None)
    a26 = of_float(of_row, "AU26_r", None)
    vals = []
    if a25 is not None: vals.append(a25)
    if a26 is not None: vals.append(a26)
    if not vals:
        return "unknown"
    s_open = max(vals)
    closed_thr = calib.get("AU25_lo", 0.33)
    open_thr   = calib.get("AU25_hi", float(open_q))
    wide_thr   = calib.get("AU26_hi", float(wide_q))

    if s_open <= (float(closed_thr) + float(margin_close)):
        return "closed"
    if (a26 is not None) and (a26 >= (float(wide_thr) + float(margin_wide))) and (s_open >= (float(open_thr) + float(margin_open))):
        return "wide"
    if s_open >= (float(open_thr) + float(margin_open)):
        return "open"
    return "slight"

def smile_from_of(of_row, mouth_state, calib, open_q=0.65, smile_margin=0.05):
    a12 = of_float(of_row, "AU12_r", None)
    a25 = of_float(of_row, "AU25_r", None)
    a26 = of_float(of_row, "AU26_r", None)
    if a12 is None or calib.get("AU12_hi") is None or calib.get("AU12_lo") is None:
        return "none"
    a12_hi = float(calib["AU12_hi"]) + float(smile_margin)
    s_open = max([v for v in (a25, a26) if v is not None] or [0.0])
    open_thr = float(calib.get("AU25_hi", float(open_q))) + float(smile_margin)
    if a12 >= a12_hi and (mouth_state in ("open","wide") and s_open >= open_thr):
        return "smile_teeth"
    if a12 >= a12_hi and (mouth_state in ("closed","slight") and s_open < open_thr):
        return "smile_closed"
    return "none"

def gaze_bin_from_of(of_row, thr=6.0):
    gx = of_float(of_row, "gaze_angle_x", None)
    gy = of_float(of_row, "gaze_angle_y", None)
    if gx is None or gy is None:
        return None
    if abs(gx) <= thr and abs(gy) <= thr: return "front"
    if gx < -thr: return "left"
    if gx >  thr: return "right"
    if gy < -thr: return "up"
    if gy >  thr: return "down"
    return "front"

def of_pose_deg(of_row):
    rx = of_float(of_row, "pose_Rx", None)  # pitch
    ry = of_float(of_row, "pose_Ry", None)  # yaw
    if rx is None or ry is None:
        return None, None
    if abs(rx) <= 3.2 and abs(ry) <= 3.2:
        rx *= 180.0 / math.pi
        ry *= 180.0 / math.pi
    pitch_deg = rx
    yaw_deg   = ry
    return yaw_deg, pitch_deg

def yaw_bin_from_deg_of(yaw, front=10.0, slight=25.0):
    if yaw is None: return None
    if abs(yaw) <= front: return "front"
    if yaw < -slight:     return "strong_l"
    if yaw >  slight:     return "strong_r"
    return "slight_l" if yaw < 0 else "slight_r"

def pitch_bin_from_deg_of(pitch, front=8.0, up_thr=18.0):
    if pitch is None: return None
    if pitch >= up_thr:  return "up"
    if pitch <= -up_thr: return "down"
    return "front"

# ===== Landmark-based geometry (EAR/MAR) + Hybrid fusion =====
def _of_landmarks_xy(r):
    try:
        xs = []; ys = []
        for i in range(68):
            xi = r.get(f"x_{i}", None)
            yi = r.get(f"y_{i}", None)
            if xi is None or yi is None or xi == "" or yi == "":
                return (None, None)
            xs.append(float(xi)); ys.append(float(yi))
        return (xs, ys)
    except Exception:
        return (None, None)

def _dist(x1,y1,x2,y2):
    dx = (x1-x2); dy = (y1-y2)
    return math.sqrt(dx*dx + dy*dy)

def _eye_aspect_ratio(xs, ys):
    if xs is None or ys is None or len(xs) < 48:
        return None
    def ear_pair(idx):
        i0,i1,i2,i3,i4,i5 = idx
        vert = _dist(xs[i1],ys[i1],xs[i5],ys[i5]) + _dist(xs[i2],ys[i2],xs[i4],ys[i4])
        hori = _dist(xs[i0],ys[i0],xs[i3],ys[i3])
        if hori <= 1e-6: return None
        return vert / (2.0*hori)
    L = ear_pair((36,37,38,39,40,41))
    R = ear_pair((42,43,44,45,46,47))
    if L is None and R is None: return None
    if L is None: return R
    if R is None: return L
    return 0.5*(L+R)

def _mouth_aspect_ratio(xs, ys):
    if xs is None or ys is None or len(xs) < 68:
        return None
    v = (_dist(xs[61],ys[61],xs[67],ys[67]) +
         _dist(xs[62],ys[62],xs[66],ys[66]) +
         _dist(xs[63],ys[63],xs[65],ys[65]))
    w = _dist(xs[60],ys[60],xs[64],ys[64])
    if w <= 1e-6: return None
    return v / (2.0*w)

def _smile_spread(xs, ys):
    if xs is None or ys is None or len(xs) < 68:
        return None
    mouth = _dist(xs[48],ys[48],xs[54],ys[54])
    eye   = _dist(xs[36],ys[36],xs[45],ys[45])
    if eye <= 1e-6: return None
    return mouth / eye

def calibrate_landmarks(of_rows, qlo=0.20, qhi=0.80, smile_q=0.80, conf_floor=0.60):
    EARs, MARs, SPREAD = [], [], []
    for r in of_rows:
        try:
            conf = of_float(r, "confidence", 1.0)
            succ = of_int(r, "success", 1)
            if conf is not None and conf < conf_floor: 
                continue
            if succ is not None and succ == 0:
                continue
            xs, ys = _of_landmarks_xy(r)
            if xs is None: 
                continue
            e = _eye_aspect_ratio(xs, ys)
            m = _mouth_aspect_ratio(xs, ys)
            s = _smile_spread(xs, ys)
            if e is not None and math.isfinite(e): EARs.append(e)
            if m is not None and math.isfinite(m): MARs.append(m)
            if s is not None and math.isfinite(s): SPREAD.append(s)
        except Exception:
            continue
    def _q(vs, q):
        if not vs: return None
        vs = sorted(vs)
        idx = int(round(q*(len(vs)-1)))
        if idx < 0: idx = 0
        if idx >= len(vs): idx = len(vs)-1
        return vs[idx]
    return dict(
        EAR_lo=_q(EARs, qlo), EAR_hi=_q(EARs, qhi),
        MAR_lo=_q(MARs, qlo), MAR_open=_q(MARs, qhi), MAR_wide=_q(MARs, min(0.98, max(qhi, 0.96))),
        SMILE_thr=_q(SPREAD, smile_q)
    )

def eyes_from_landmarks(of_row, lm_cal, pose_gate_yaw, pose_gate_pitch, yaw_deg=None, pitch_deg=None, eyes_margin=0.05):
    try:
        if yaw_deg is not None and abs(float(yaw_deg)) > float(pose_gate_yaw): return "unknown"
        if pitch_deg is not None and abs(float(pitch_deg)) > float(pose_gate_pitch): return "unknown"
    except Exception:
        pass
    xs, ys = _of_landmarks_xy(of_row)
    if xs is None: return "unknown"
    ear = _eye_aspect_ratio(xs, ys)
    if ear is None: return "unknown"
    lo = lm_cal.get("EAR_lo"); hi = lm_cal.get("EAR_hi")
    if lo is None or hi is None: return "unknown"
    if ear <= (float(lo) - float(eyes_margin)): return "closed"
    if ear >= (float(hi) + float(eyes_margin)): return "open"
    return "half"

def mouth_state_from_landmarks(of_row, lm_cal, pose_gate_yaw, pose_gate_pitch, yaw_deg=None, pitch_deg=None,
                               open_q=0.80, wide_q=0.96, margin_close=0.03, margin_open=0.03, margin_wide=0.05):
    try:
        if yaw_deg is not None and abs(float(yaw_deg)) > float(pose_gate_yaw): return "unknown"
        if pitch_deg is not None and abs(float(pitch_deg)) > float(pose_gate_pitch): return "unknown"
    except Exception:
        pass
    xs, ys = _of_landmarks_xy(of_row)
    if xs is None: return "unknown"
    mar = _mouth_aspect_ratio(xs, ys)
    if mar is None: return "unknown"
    mar_lo = lm_cal.get("MAR_lo"); mar_open = lm_cal.get("MAR_open"); mar_wide = lm_cal.get("MAR_wide")
    if mar_lo is None or mar_open is None: return "unknown"
    if mar <= (float(mar_lo) + float(margin_close)): return "closed"
    if mar >= (float(mar_wide) + float(margin_wide)): return "wide"
    if mar >= (float(mar_open) + float(margin_open)): return "open"
    return "slight"

def select_eyes_label(of_lbl, lm_lbl, method, of_conf, of_conf_floor):
    m = (method or "hybrid").lower()
    if m == "openface":  return of_lbl
    if m == "landmark":  return lm_lbl
    if (of_conf is None) or (of_conf < of_conf_floor) or (of_lbl == "unknown"):
        return lm_lbl if lm_lbl != "unknown" else of_lbl
    if lm_lbl != "unknown" and of_lbl != lm_lbl: return lm_lbl
    return of_lbl

def select_mouth_state(of_state, lm_state, method, of_conf, of_conf_floor):
    m = (method or "hybrid").lower()
    if m == "openface":  return of_state
    if m == "landmark":  return lm_state
    if (of_conf is None) or (of_conf < of_conf_floor) or (of_state == "unknown"):
        return lm_state if lm_state != "unknown" else of_state
    if lm_state != "unknown" and of_state != lm_state: return lm_state
    return of_state

def smile_from_hybrid(of_row, mouth_state, of_cal, lm_cal, mouth_q_open, smile_margin, method):
    m = (method or "hybrid").lower()
    if m == "openface":
        return smile_from_of(of_row, mouth_state, of_cal, mouth_q_open, smile_margin)
    xs, ys = _of_landmarks_xy(of_row)
    if xs is None: return "none"
    sp = _smile_spread(xs, ys)
    thr = lm_cal.get("SMILE_thr")
    if thr is None or sp is None: return "none"
    if sp >= (float(thr) + float(smile_margin)):
        return "smile_teeth" if mouth_state in ("open","wide") else "smile_closed"
    return "none"

# ================= TAG =================
def cmd_tag(args):
    images = list_images(args.dataset)
    if not images:
        print("No images found in {}".format(args.dataset))
        return 2

    if (not args.openface_dir) or (not Path(args.openface_dir).exists()):
        print("ERROR: --openface_dir is required and must exist.")
        return 3

    of_rows = of_glob_csv(args.openface_dir)
    OF_MAP = of_build_map(of_rows)
    OF_CAL = of_calibrate(of_rows, args.of_conf_floor, args.au_low_q, args.au_high_q)
    LM_CAL = calibrate_landmarks(
        of_rows,
        qlo=float(getattr(args,"lm_low_q",0.20)),
        qhi=float(getattr(args,"lm_high_q",0.80)),
        smile_q=float(getattr(args,"smile_q",0.80)),
        conf_floor=float(getattr(args,"of_conf_floor",0.60)),
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir)
    (logs_dir/"wb").mkdir(parents=True, exist_ok=True)
    (logs_dir/"exposure").mkdir(parents=True, exist_ok=True)
    (logs_dir/"tags").mkdir(parents=True, exist_ok=True)
    tags_csv = logs_dir/"tags"/"tags.csv"
    manifest_fp = logs_dir/"manifest.jsonl"

    rows = []; manifest_rows = []

    C = dict(
        yaw=Counter(), pitch=Counter(), eyes=Counter(), mouth=Counter(),
        smile=Counter(), gaze=Counter(), light=Counter(), expo=Counter()
    )

    # initial divider then start TAG bar
    print(SEP)

    t0 = time.time()
    unknown_running = 0

    for i, img_path in enumerate(images, 1):
        base = img_path.name
        of = OF_MAP.get(base)
        if of is None:
            stem = img_path.stem
            for ext in IMG_EXTS:
                of = OF_MAP.get(stem+ext)
                if of is not None:
                    break

        R = {"file": str(img_path), "name": base}
        M = {"file": str(img_path), "name": base}

        try:
            M["hash"] = sha1_of_file(img_path)
        except Exception:
            M["hash"] = ""

        if of is not None:
            conf = of_float(of, "confidence", 1.0)
            succ = of_int(of, "success", 1)
            if (conf is None or conf >= args.of_conf_floor) and (succ in (None,1)):
                yaw_deg, pitch_deg = of_pose_deg(of)
                if int(args.of_pitch_invert) and (pitch_deg is not None):
                    pitch_deg = -pitch_deg
                if yaw_deg is not None:
                    R["yaw_deg"] = yaw_deg; M["yaw_deg"] = yaw_deg
                if pitch_deg is not None:
                    R["pitch_deg"] = pitch_deg; M["pitch_deg"] = pitch_deg
                yb = yaw_bin_from_deg_of(M.get("yaw_deg"), args.of_yaw_front, args.of_yaw_slight)
                pb = pitch_bin_from_deg_of(M.get("pitch_deg"), args.of_pitch_front, args.of_pitch_up)
                if yb is not None:
                    R["yaw_bin"] = yb; M["yaw_bin"] = yb; C["yaw"][yb]+=1
                if pb is not None:
                    R["pitch_bin"] = pb; M["pitch_bin"] = pb; C["pitch"][pb]+=1
                eyes_of = eyes_from_of(of, OF_CAL, args.pose_gate_yaw, args.pose_gate_pitch, M.get("yaw_deg"), M.get("pitch_deg"), args.eyes_margin)
                eyes_lm = eyes_from_landmarks(of, LM_CAL, args.pose_gate_yaw, args.pose_gate_pitch, M.get("yaw_deg"), M.get("pitch_deg"), args.eyes_margin)
                eyes_lbl = select_eyes_label(eyes_of, eyes_lm, args.eyes_method, conf, args.of_conf_floor)
                if eyes_lbl != "unknown":
                    R["eyes"] = eyes_lbl; M["eyes"] = eyes_lbl; C["eyes"][eyes_lbl]+=1
                else:
                    unknown_running += 1

                mouth_of = mouth_state_from_of(of, OF_CAL, args.pose_gate_yaw, args.pose_gate_pitch, M.get("yaw_deg"), M.get("pitch_deg"), args.mouth_q_open, args.mouth_q_wide, args.mouth_margin_close, args.mouth_margin_open, args.mouth_margin_wide)
                mouth_lm = mouth_state_from_landmarks(of, LM_CAL, args.pose_gate_yaw, args.pose_gate_pitch, M.get("yaw_deg"), M.get("pitch_deg"), args.mouth_q_open, args.mouth_q_wide, args.mouth_margin_close, args.mouth_margin_open, args.mouth_margin_wide)
                mouth_state = select_mouth_state(mouth_of, mouth_lm, args.mouth_method, conf, args.of_conf_floor)
                if mouth_state != "unknown":
                    R["mouth"] = mouth_state; M["mouth"] = mouth_state; C["mouth"][mouth_state]+=1

                sm = smile_from_hybrid(of, mouth_state, OF_CAL, LM_CAL, args.mouth_q_open, args.smile_margin, args.mouth_method)
                R["smile"] = sm; M["smile"] = sm; C["smile"][sm]+=1

                gz = gaze_bin_from_of(of, thr=args.gaze_thr)
                if R.get("eyes") == "closed":
                    gz = None
                if gz is not None:
                    R["gaze"] = gz; M["gaze"] = gz; C["gaze"][gz]+=1

                M["confidence"] = conf if conf is not None else ""
                M["success"] = succ if succ is not None else ""

        # Lighting & Exposure (from the actual image)
        try:
            img_bgr = _imread_any_bgr(img_path)
            if (img_bgr is not None) and (np is not None):
                h, w = img_bgr.shape[:2]
                M["w"] = int(w); M["h"] = int(h)
                cct = estimate_cct_from_image_bgr(img_bgr)
                R["cct_k"] = cct if cct is not None else ""
                M["cct_k"] = R["cct_k"]
                lt = bin_cct(cct, args.cct_cool_k, args.cct_warm_k, args.cct_flip, args.cct_neutral_pad)
                R["lighting_temp"] = lt; M["lighting_temp"] = lt; C["light"][lt]+=1

                mean_L, sh_pct, hi_pct = exposure_metrics(img_bgr, center_weighted=bool(args.ex_center_weighted), center_scale=args.ex_center_scale)
                R["mean_L"] = round(float(mean_L), 4)
                R["shadow_pct"] = round(float(sh_pct), 4)
                R["highlight_pct"] = round(float(hi_pct), 4)
                exb = bin_exposure(mean_L, sh_pct, hi_pct, args.ex_mean_low, args.ex_mean_high, args.ex_shadow_pct, args.ex_highlight_pct)
                R["exposure"] = exb; M["exposure"] = exb; C["expo"][exb]+=1
        except Exception:
            pass

        # link possible external logs if present
        of_dir = Path(args.openface_dir)
        of_guess = None
        guess_a = of_dir / (Path(base).stem + ".csv")
        guess_b = of_dir / (Path(base).name.replace(".png",".csv"))
        if guess_a.exists(): of_guess = guess_a
        elif guess_b.exists(): of_guess = guess_b
        if of_guess:
            M["of_csv"] = str(of_guess)

        if args.magface_dir:
            mf_path = Path(args.magface_dir) / (Path(base).stem + ".npy")
            if mf_path.exists():
                M["magface"] = str(mf_path)

        bbox_dir = Path(args.logs_dir) / "openface_bbox"
        bbox_path = bbox_dir / (Path(base).stem + ".txt")
        if bbox_path.exists():
            M["openface_bbox"] = str(bbox_path)

        rows.append(R)
        manifest_rows.append(M)

        # single-line TAG bar
        print_tag_progress(i, len(images), t0, unknown_running)

    # write outputs
    if rows:
        cols = sorted({k for r in rows for k in r.keys()})
        with open(str(tags_csv), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        print("[TAG] wrote {} ({} rows)".format(tags_csv, len(rows)))

    write_jsonl(manifest_rows, str(manifest_fp))
    print("[TAG] wrote {} ({} rows)".format(manifest_fp, len(manifest_rows)))
    print("[TAG] WB/Exposure logs under {}".format(logs_dir))

    # final aligned summary table
    print_summary_table(C, len(images))
    return 0

# ================= PREVIEW =================
def tile_images(img_paths, tile_size=128, per=100):
    if Image is None:
        return None
    paths = list(img_paths)
    random.shuffle(paths)
    imgs = []
    for p in paths[:per]:
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((tile_size, tile_size), Image.BICUBIC)
            imgs.append(im)
        except Exception:
            pass
    if not imgs:
        return None
    n = len(imgs)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(float(n)/float(cols)))
    canvas = Image.new("RGB", (cols*tile_size, rows*tile_size), (20,20,20))
    for idx, im in enumerate(imgs):
        r = int(idx // cols)
        c = int(idx % cols)
        canvas.paste(im, (c*tile_size, r*tile_size))
    return canvas

def cmd_preview(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags_csv = Path(args.logs_dir) / "tags" / "tags.csv"
    if not tags_csv.exists():
        print("tags.csv not found in {}".format(tags_csv))
        return 2
    with open(str(tags_csv), "r", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
    previews_dir = out_dir / "previews"; previews_dir.mkdir(parents=True, exist_ok=True)

    bins = [b.strip() for b in args.bins.split(",") if b.strip()]
    total_tiles = 0
    groups_per_bin = []
    for b in bins:
        groups = defaultdict(list)
        for r in rows:
            v = r.get(b, "")
            if not v:
                continue
            groups[v].append(r.get("file") or r.get("name"))
        groups_per_bin.append((b, groups))
        total_tiles += len(groups)

    done = 0
    for b, groups in groups_per_bin:
        for val, files in groups.items():
            paths = [Path(f) for f in files if f]
            img = tile_images(paths, tile_size=args.size, per=args.per)
            if img is None:
                done += 1
                continue
            fn = previews_dir / "{}__{}.jpg".format(b, val)
            try:
                img.save(fn, quality=90)
            except Exception:
                pass
            done += 1
            ratio = done / float(max(1, total_tiles))
            line = _line_header("PREVIEW", ratio, done, total_tiles, 0)
            sys.stdout.write("\r" + line); sys.stdout.flush()
    if total_tiles:
        sys.stdout.write("\n")
    print("[PREVIEW] written to {}".format(previews_dir))
    return 0

# ================= MATCH =================
KEYS_PRIMARY = ["yaw_bin","pitch_bin","mouth","eyes"]

def load_dst_rows(dst_manifest_or_tags):
    p = Path(dst_manifest_or_tags)
    if p.suffix.lower() == ".jsonl":
        return read_jsonl(p)
    with p.open("r", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return list(rd)

def build_hist(rows, keys):
    hist = Counter()
    for r in rows:
        key = tuple(r.get(k,"") for k in keys)
        if all(key):
            hist[key] += 1
    return hist

def cosine(a, b, eps=1e-8):
    if np is None:
        return 0.0
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a,b) / (na*nb))

def read_magface(vec_path):
    if np is None:
        return None
    try:
        return np.load(vec_path)
    except Exception:
        return None

def cmd_match(args):
    lib_manifest = Path(args.lib_manifest)
    if not lib_manifest.exists():
        print("Library manifest not found: {}".format(lib_manifest))
        return 2
    dst_path = Path(args.dst_manifest)
    if not dst_path.exists():
        print("Destination manifest/tags not found: {}".format(dst_path))
        return 2

    lib_rows = read_jsonl(lib_manifest)
    dst_rows = load_dst_rows(dst_path)

    keys_primary = list(KEYS_PRIMARY)
    if args.keys:
        keys_primary = [k.strip() for k in args.keys.split(",") if k.strip()]

    dst_hist = build_hist(dst_rows, keys_primary)
    dst_total = sum(dst_hist.values())
    if dst_total == 0:
        print("No destination histogram; nothing to match.")
        return 3

    if int(args.budget) <= 0:
        budget = max(1500, min(6000, int(round(1.5 * dst_total))))
    else:
        budget = int(args.budget)

    quotas = {}
    for key, cnt in dst_hist.items():
        quotas[key] = max(int(args.min_per_bin), int(round(budget * float(cnt) / float(dst_total))))

    def key_of(r):
        return tuple(r.get(k,"") for k in keys_primary)

    lib_by_key = defaultdict(list)
    for r in lib_rows:
        k = key_of(r)
        if all(k):
            lib_by_key[k].append(r)

    selected = []
    selected_set = set()
    per_key_selected = Counter()

    diversity = float(args.diversity)
    cache_mag = {}

    total_quota = sum(quotas.values())
    done_count = 0

    def print_match_bar(done, total):
        sys.stdout.write("\r" + _line_header("MATCH", done/float(max(1,total)), done, total, 0))
        sys.stdout.flush()

    def pick_from_bucket(bucket, need, k):
        nonlocal done_count
        if need <= 0 or (not bucket):
            return
        random.shuffle(bucket)
        for r in bucket:
            if len(selected) >= budget:
                break
            h = r.get("hash") or r.get("name") or r.get("file")
            if h in selected_set:
                continue
            ok = True
            if diversity > 0.0 and np is not None:
                vec = None
                vf = r.get("magface")
                if vf:
                    vec = cache_mag.get(vf)
                    if vec is None:
                        vec = read_magface(vf); cache_mag[vf] = vec
                if vec is not None:
                    for s in selected[-200:]:
                        if key_of(s) != key_of(r):
                            continue
                        vf2 = s.get("magface")
                        if not vf2:
                            continue
                        vec2 = cache_mag.get(vf2)
                        if vec2 is None:
                            vec2 = read_magface(vf2); cache_mag[vf2] = vec2
                        if vec2 is None:
                            continue
                        if cosine(vec, vec2) > (1.0 - diversity):
                            ok = False; break
            if not ok:
                continue
            selected.append(r); selected_set.add(h)
            per_key_selected[key_of(r)] += 1
            done_count += 1
            print_match_bar(done_count, total_quota)
            if per_key_selected[key_of(r)] >= need:
                break

    # primary pass
    for k, need in quotas.items():
        pick_from_bucket(lib_by_key.get(k, []), need, k)

    # neighbor/backfill
    def nearest_keys(k):
        from_key = dict(zip(keys_primary, k))
        res = []
        y = from_key.get("yaw_bin","")
        yaw_n = []
        if y == "front": yaw_n = ["slight_l","slight_r"]
        elif y in ("slight_l","slight_r"): yaw_n = ["front"] + (["strong_l"] if y=="slight_l" else ["strong_r"])
        elif y in ("strong_l","strong_r"): yaw_n = ["slight_l" if y=="strong_l" else "slight_r"]
        p = from_key.get("pitch_bin","")
        pit_n = []
        if p == "front": pit_n = ["up","down"]
        elif p in ("up","down"): pit_n = ["front"]
        for yn in yaw_n or [y]:
            for pn in pit_n or [p]:
                nk = tuple((yn if kk=="yaw_bin" else pn if kk=="pitch_bin" else from_key[kk]) for kk in keys_primary)
                if nk != k:
                    res.append(nk)
        return res

    if len(selected) < budget:
        for k, need in quotas.items():
            have = per_key_selected.get(k, 0)
            if have >= need:
                continue
            for nk in nearest_keys(k):
                pick_from_bucket(lib_by_key.get(nk, []), need - have, k)
                have = per_key_selected.get(k, 0)
                if have >= need:
                    break
        if len(selected) < budget:
            leftovers = [r for r in lib_rows if (r.get("hash") or r.get("name") or r.get("file")) not in selected_set]
            random.shuffle(leftovers)
            for r in leftovers:
                if len(selected) >= budget:
                    break
                selected.append(r); selected_set.add(r.get("hash") or r.get("name") or r.get("file"))
                done_count += 1
                print_match_bar(done_count, total_quota)

    out = args.out if args.out else str(Path(args.project_logs)/"curator_match.jsonl")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(selected, out)

    sel_hist = build_hist(selected, keys_primary)
    meta = {
        "keys_primary": keys_primary,
        "dst_total": dst_total,
        "budget": budget,
        "selected": len(selected),
        "per_key_selected": {str(k): int(v) for k,v in sel_hist.items()},
    }
    with open(str(Path(out).parent/"curator_match.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    sys.stdout.write("\n")
    print("[MATCH] wrote {} ({} rows)".format(out, len(selected)))
    print("[MATCH] meta: {}".format(Path(out).parent/'curator_match.meta.json'))
    return 0

# ================= BUILD =================
def safe_link(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try: os.link(str(src), str(dst))
        except Exception: shutil.copy2(str(src), str(dst))
    elif mode == "symlink":
        try: dst.symlink_to(src)
        except Exception: shutil.copy2(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def cmd_build(args):
    match_fp = Path(args.match)
    if not match_fp.exists():
        print("Match file not found: {}".format(match_fp))
        return 2
    sel = read_jsonl(match_fp)
    if not sel:
        print("Empty match list.")
        return 3

    aligned_out = Path(args.dst_aligned_out); aligned_out.mkdir(parents=True, exist_ok=True)
    logs_out = Path(args.logs_dir_out); logs_out.mkdir(parents=True, exist_ok=True)
    (logs_out/"openface_img").mkdir(parents=True, exist_ok=True)
    (logs_out/"magface").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in enumerate(sel, 1):
        src_img = Path(r.get("file",""))
        h = r.get("hash") or (sha1_of_file(src_img) if src_img.exists() else None)
        if (not src_img.exists()) or (not h):
            continue
        dst_img = aligned_out / "{}.png".format(h)
        safe_link(src_img, dst_img, args.link_mode)

        out_row = dict(r)
        out_row["file"] = str(dst_img)
        rows.append(out_row)

        ofp = r.get("of_csv")
        if ofp and Path(ofp).exists():
            safe_link(Path(ofp), logs_out/"openface_img"/"{}.csv".format(h), args.link_mode)
        mfp = r.get("magface")
        if mfp and Path(mfp).exists():
            safe_link(Path(mfp), logs_out/"magface"/"{}.npy".format(h), args.link_mode)

        sys.stdout.write("\r" + _line_header("BUILD", i/float(len(sel)), i, len(sel), 0))
        sys.stdout.flush()

    write_jsonl(rows, str(logs_out/"manifest_src.jsonl"))
    sys.stdout.write("\n")
    print("[BUILD] materialized {} images into {}".format(len(rows), aligned_out))
    print("[BUILD] logs at {}".format(logs_out))
    return 0

# ================= REMOVE =================
def cmd_remove(args):
    logs = Path(args.logs_dir)
    tags_csv = logs/"tags"/"tags.csv"
    manifest = logs/"manifest.jsonl"
    if (not tags_csv.exists()) and (not manifest.exists()):
        print("No project tags/manifest found under {}".format(logs))
        return 2
    rows = []
    if manifest.exists():
        rows = read_jsonl(manifest)
        key_file = "file"
    else:
        with tags_csv.open("r", encoding="utf-8") as fh:
            rd = csv.DictReader(fh); rows = list(rd)
        key_file = "file"

    expr = args.expr.strip()
    if "==" in expr:
        k, v = expr.split("==", 1); k=k.strip(); v=v.strip()
        def match(r): return str(r.get(k,"")) == v
    elif "!=" in expr:
        k, v = expr.split("!=", 1); k=k.strip(); v=v.strip()
        def match(r): return str(r.get(k,"")) != v
    else:
        print("Expr must contain == or !=")
        return 3

    victims = [Path(r.get(key_file,"")) for r in rows if match(r) and r.get(key_file)]
    trash = Path(args.trash_dir); trash.mkdir(parents=True, exist_ok=True)
    moved = 0
    for i, fp in enumerate(victims, 1):
        if fp.exists():
            dst = trash / fp.name
            try:
                fp.replace(dst)
                moved += 1
            except Exception:
                pass
        sys.stdout.write("\r" + _line_header("REMOVE", i/float(max(1,len(victims))), i, len(victims), 0))
        sys.stdout.flush()
    sys.stdout.write("\n")
    print("[REMOVE] moved {} files to {} for expr: {}".format(moved, trash, expr))
    return 0

# ================= STATS =================
def cmd_stats(args):
    logs = Path(args.logs_dir)
    rows = []
    manifest = logs/"manifest.jsonl"
    tags_csv = logs/"tags"/"tags.csv"
    if manifest.exists():
        rows = read_jsonl(manifest)
    elif tags_csv.exists():
        with tags_csv.open("r", encoding="utf-8") as fh:
            rd = csv.DictReader(fh); rows = list(rd)
    else:
        print("No manifest.jsonl or tags.csv under {}".format(logs))
        return 2

    if args.keys:
        keys = [k.strip() for k in args.keys.split(",")]
    else:
        keys = ["yaw_bin","pitch_bin","eyes","mouth","smile","gaze","lighting_temp","exposure"]

    C = {k: Counter() for k in keys}
    total = len(rows)
    step = max(1, total//200 or 1)
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        for k in keys:
            v = r.get(k, "")
            if v: C[k][v] += 1
        if (i % step == 0) or i == total:
            elapsed = max(1e-6, time.time()-t0)
            fps = i/elapsed
            eta = (total - i)/fps if fps>0 else float("inf")
            sys.stdout.write("\r" + _line_header("STATS", i/float(max(1,total)), i, total, 0, fps=fps, eta=eta))
            sys.stdout.flush()
    sys.stdout.write("\n")

    print("[STATS] rows={} from {}".format(total, manifest if manifest.exists() else tags_csv))
    for f in keys:
        c = C[f]
        items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        print("\n[{}] count={}".format(f, sum(c.values())))
        for k,v in items:
            kk = k if k else "∅"
            print("  {:16s} {:7d}".format(kk, v))
    return 0

# ================= CLI =================
def build_parser():
    p = argparse.ArgumentParser("Curator v4.9 (tag, preview, match, build, remove, stats)")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp_tag = sp.add_parser("tag")
    sp_tag.add_argument("--dataset", type=str, required=True)
    sp_tag.add_argument("--out_dir", type=str, required=True)
    sp_tag.add_argument("--insight_pack", type=str, default="")
    sp_tag.add_argument("--require_106", type=int, default=0)
    sp_tag.add_argument("--openface_dir", type=str, required=True)
    sp_tag.add_argument("--magface_dir", type=str, default="")
    sp_tag.add_argument("--of_conf_floor", type=float, default=0.60)
    sp_tag.add_argument("--au_low_q", type=float, default=0.30)
    sp_tag.add_argument("--au_high_q", type=float, default=0.70)
    sp_tag.add_argument("--mouth_q_open", type=float, default=0.72)
    sp_tag.add_argument("--mouth_q_wide", type=float, default=0.94)
    sp_tag.add_argument("--pose_gate_yaw", type=float, default=12.0)
    sp_tag.add_argument("--pose_gate_pitch", type=float, default=8.0)
    sp_tag.add_argument("--of_yaw_front", type=float, default=10.0)
    sp_tag.add_argument("--of_yaw_slight", type=float, default=25.0)
    sp_tag.add_argument("--of_pitch_front", type=float, default=8.0)
    sp_tag.add_argument("--of_pitch_up", type=float, default=18.0)
    sp_tag.add_argument("--of_pitch_invert", type=int, default=1)
    sp_tag.add_argument("--gaze_thr", type=float, default=6.0)
    sp_tag.add_argument("--cct_cool_k", type=float, default=5200.0)
    sp_tag.add_argument("--cct_warm_k", type=float, default=6000.0)
    sp_tag.add_argument("--cct_flip", type=int, default=1)
    sp_tag.add_argument("--cct_neutral_pad", type=float, default=150.0)
    sp_tag.add_argument("--logs_dir", type=str, default="/workspace/data_src/logs")
    sp_tag.add_argument("--ex_mean_low", type=float, default=0.35)
    sp_tag.add_argument("--ex_mean_high", type=float, default=0.60)
    sp_tag.add_argument("--ex_shadow_pct", type=float, default=0.15)
    sp_tag.add_argument("--ex_highlight_pct", type=float, default=0.05)
    sp_tag.add_argument("--ex_center_weighted", type=int, default=1)
    sp_tag.add_argument("--ex_center_scale", type=float, default=0.70)
    sp_tag.add_argument("--eyes_method",  type=str, choices=["openface","landmark","hybrid"], default="hybrid")
    sp_tag.add_argument("--mouth_method", type=str, choices=["openface","landmark","hybrid"], default="hybrid")
    sp_tag.add_argument("--lm_low_q",  type=float, default=0.20)
    sp_tag.add_argument("--lm_high_q", type=float, default=0.80)
    sp_tag.add_argument("--smile_q",   type=float, default=0.80)
    sp_tag.add_argument("--eyes_margin", type=float, default=0.05)
    sp_tag.add_argument("--mouth_margin_close", type=float, default=0.03)
    sp_tag.add_argument("--mouth_margin_open",  type=float, default=0.03)
    sp_tag.add_argument("--mouth_margin_wide",  type=float, default=0.05)
    sp_tag.add_argument("--smile_margin",       type=float, default=0.05)
    sp_tag.add_argument("--print_every", type=int, default=0)
    sp_tag.add_argument("--pb_every", type=int, default=1)

    sp_prev = sp.add_parser("preview")
    sp_prev.add_argument("--dataset", type=str, required=True)
    sp_prev.add_argument("--out_dir", type=str, required=True)
    sp_prev.add_argument("--bins", type=str, required=True)
    sp_prev.add_argument("--per", type=int, default=100)
    sp_prev.add_argument("--size", type=int, default=128)
    sp_prev.add_argument("--logs_dir", type=str, default="/workspace/data_src/logs")

    sp_match = sp.add_parser("match")
    sp_match.add_argument("--lib_manifest", type=str, required=True)
    sp_match.add_argument("--dst_manifest", type=str, required=True)
    sp_match.add_argument("--project_logs", type=str, default="/workspace/data_src/logs")
    sp_match.add_argument("--budget", type=int, default=0)
    sp_match.add_argument("--min_per_bin", type=int, default=10)
    sp_match.add_argument("--diversity", type=float, default=0.15)
    sp_match.add_argument("--keys", type=str, default="")
    sp_match.add_argument("--out", type=str, default="")

    sp_build = sp.add_parser("build")
    sp_build.add_argument("--match", type=str, required=True)
    sp_build.add_argument("--dst_aligned_out", type=str, default="/workspace/data_src/aligned")
    sp_build.add_argument("--logs_dir_out", type=str, default="/workspace/data_src/logs")
    sp_build.add_argument("--link_mode", type=str, default="hardlink", choices=["hardlink","symlink","copy"])

    sp_rm = sp.add_parser("remove")
    sp_rm.add_argument("--logs_dir", type=str, default="/workspace/data_src/logs")
    sp_rm.add_argument("--expr", type=str, required=True)
    sp_rm.add_argument("--trash_dir", type=str, default="/workspace/data_src/trash")

    sp_stats = sp.add_parser("stats")
    sp_stats.add_argument("--logs_dir", type=str, default="/workspace/data_src/logs")
    sp_stats.add_argument("--keys", type=str, default="")

    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "tag":
        rc = cmd_tag(args)
    elif args.cmd == "preview":
        rc = cmd_preview(args)
    elif args.cmd == "match":
        rc = cmd_match(args)
    elif args.cmd == "build":
        rc = cmd_build(args)
    elif args.cmd == "remove":
        rc = cmd_remove(args)
    elif args.cmd == "stats":
        rc = cmd_stats(args)
    else:
        rc = 1
    sys.exit(rc)

if __name__ == "__main__":
    main()
