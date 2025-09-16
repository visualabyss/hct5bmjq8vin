from insightface.app import FaceAnalysis
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, time, math, os, csv, shutil
from pathlib import Path
import cv2
import numpy as np

# --- COMPAT: accept `--log` as alias for legacy `--log_dir`
import sys as _sys

# --- [MSCALE SHIM START] ---
# Accept --mscale FLOAT even if argparse doesn't know it, and remove it from argv.
# Then monkey-patch InsightFace FaceAnalysis.get() to retry at det_size*x when no faces are found.
__MSCALE_SHIM__ = True
import sys as _sys
__MSCALE = 1.0
_newargv = []
_skip = False
for i,a in enumerate(_sys.argv):
    if _skip:
        _skip = False
        continue
    if a == "--mscale":
        if i+1 < len(_sys.argv):
            try:
                __MSCALE = float(_sys.argv[i+1])
            except Exception:
                pass
            _skip = True  # skip the value
        # drop "--mscale"
        continue
    if a.startswith("--mscale="):
        try:
            __MSCALE = float(a.split("=",1)[1])
        except Exception:
            pass
        # drop the arg entirely
        continue
    _newargv.append(a)
_sys.argv = _newargv

# Try to patch FaceAnalysis.get so it retries at higher scale only when no faces are found.
try:
    import cv2 as _cv2
    from insightface.app import FaceAnalysis as _FA
    _orig_get = _FA.get
    if not hasattr(_FA, "_mscale_patched"):
        def _patched_get(self, img, *args, **kwargs):
            faces = _orig_get(self, img, *args, **kwargs)
            try:
                s = float(globals().get("__MSCALE", 1.0) or 1.0)
            except Exception:
                s = 1.0
            if (not faces) and s > 1.0:
                try:
                    img2 = _cv2.resize(img, None, fx=s, fy=s, interpolation=_cv2.INTER_CUBIC)
                except Exception:
                    return faces
                faces2 = _orig_get(self, img2, *args, **kwargs)
                if faces2:
                    # map boxes/landmarks back to original scale
                    for f in faces2:
                        try:
                            f.bbox = f.bbox / s
                        except Exception:
                            pass
                        try:
                            f.kps = f.kps / s
                        except Exception:
                            pass
                    return faces2
            return faces
        _FA.get = _patched_get
        _FA._mscale_patched = True
        if (globals().get("__MSCALE", 1.0) or 1.0) > 1.0:
            print(f"[MSCALE] Fallback enabled: x{globals()['__MSCALE']:.2f} when base detection finds no faces.")
except Exception as _e:
    # Non-fatal: if InsightFace isn't imported yet, later imports still see the patched class
    pass
# --- [MSCALE SHIM END] ---
for _i, _a in enumerate(list(_sys.argv)):
    if _a == '--log':
        _sys.argv[_i] = '--log_dir'
    elif _a.startswith('--log='):
        _sys.argv[_i] = '--log_dir=' + _a.split('=',1)[1]
# --- END COMPAT ---

# ------------ single-line progress bar with percent + SAVED (CR-only) ------------
class _DFLBarX:
    def __init__(self, total):
        self.total = int(total) if total else 0
        self.n = 0
        self.saved = 0
        self.t0 = time.time()
        self._last = 0.0
    def set_saved(self, n):
        try: self.saved = int(n)
        except Exception: pass
    def _fmt(self, secs):
        if not math.isfinite(secs): return "--:--"
        secs = max(0, int(secs)); m, s = divmod(secs, 60); h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    def _render(self):
        el = max(1e-9, time.time() - self.t0)
        rate = self.n / el
        bw = 40
        if self.total:
            r = self.n / max(1, self.total)
            full = int(bw * r); frac = (bw * r) - full
            bar = "█" * max(0, min(full, bw))
            partials = "▏▎▍▌▋▊▉"
            if full < bw:
                idx = int(frac * len(partials))
                if idx > 0:
                    bar += partials[idx-1]; full += 1
            if full < bw:
                bar += " " * (bw - full)
            pct = 100.0 * r
            eta = (self.total - self.n)/rate if rate > 0 else float("inf")
            line = f"extract |{bar}|  {pct:5.1f}%  {self.n}/{self.total}   {self.saved:5d}   {rate:5.1f} fps   ETA {self._fmt(eta)}"
        else:
            line = f"extract |{' ' * bw}|   --.-%  {self.n}/?   {self.saved:5d}   {rate:5.1f} fps"
        try:
            cols = shutil.get_terminal_size(fallback=(120, 20)).columns
        except Exception:
            cols = 0
        if cols:
            if len(line) < cols:
                line = line + " " * (cols - len(line) - 1)
            else:
                line = line[:cols-1]
        sys.stderr.write("\r" + line); sys.stderr.flush()
    def update(self, inc=1):
        self.n += int(inc)
        now = time.time()
        if (now - self._last) >= 0.10 or (self.total and self.n >= self.total):
            self._last = now
            self._render()
    def close(self):
        try: self._render()
        finally:
            sys.stderr.write("\n"); sys.stderr.flush()

# Count saves from BOTH cv2.imwrite and numpy.ndarray.tofile (covers imencode(...).tofile(...))
def _dfl_install_save_hooks():
    try:
        _orig = cv2.imwrite
        def _wrap(path, img):
            ok = _orig(path, img)
            if ok:
                try:
                    global __SAVED; __SAVED += 1
                except Exception: pass
                try:
                    __BAR__.set_saved(__SAVED)
                except Exception: pass
            return ok
        cv2.imwrite = _wrap
    except Exception:
        pass
    try:
        _orig_tofile = np.ndarray.tofile
        def _wrap_tofile(self, *a, **k):
            out = _orig_tofile(self, *a, **k)
            try:
                global __SAVED; __SAVED += 1
            except Exception: pass
            try:
                __BAR__.set_saved(__SAVED)
            except Exception: pass
            return out
        np.ndarray.tofile = _wrap_tofile
    except Exception:
        pass

# ------------------------- Utils -------------------------
def imread_bgr(p: Path):
    p = str(p)
    data = np.fromfile(p, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im

def safe_makedirs(p: Path):
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def mean_gray(img):
    try:
        if img is None: return float("nan")
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return float(np.mean(g))
    except Exception:
        return float("nan")

# -------------------- Reference helpers ------------------
def _clahe_bgr(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def _gamma_bgr(im, g=1.0):
    if g is None or abs(float(g) - 1.0) < 1e-6: return im
    lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / float(g))) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(im, lut)

def _preprocess_ref(im, pad_frac=0.0, gamma=None, clahe=0, **_):
    if im is None: return None
    out = im
    if int(clahe): out = _clahe_bgr(out)
    if gamma is not None: out = _gamma_bgr(out, float(gamma))
    H, W = out.shape[:2]
    pad = int(round(float(pad_frac) * min(H, W)))
    if pad > 0:
        out = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    return out

def _collect_ref_paths(args):
    paths = []
    if getattr(args, "refs", None):
        for rp in args.refs:
            if isinstance(rp, str) and rp:
                paths.append(rp)
    if getattr(args, "refs_dir", None):
        rd = os.path.abspath(args.refs_dir)
        for root, _, files in os.walk(rd):
            for fn in files:
                lo = fn.lower()
                if lo.endswith((".png",".jpg",".jpeg",".webp",".bmp")):
                    paths.append(os.path.join(root, fn))
    paths = sorted(list(dict.fromkeys(paths)))
    return paths

def _embed_refs(app, ref_paths, args):
    """Return ref_mat (Nx512) or None. Prints clear logs."""
    ref_vecs = []
    if not ref_paths:
        if float(getattr(args, "sim", 0.0)) > 0.0:
            print("[WARN] No reference images resolved from --refs/--refs_dir; similarity filter disabled.", flush=True)
        return None
    print(f"[INFO] Embedding reference images… ({len(ref_paths)} found)", flush=True)
    ok = 0
    for rp in ref_paths:
        try:
            im = imread_bgr(Path(rp))
            im = _preprocess_ref(im, pad_frac=args.ref_pad, gamma=args.ref_gamma, clahe=args.ref_clahe)
            if im is None:
                print(f"[WARN] skipping ref {rp}: unreadable", flush=True); continue
            faces = app.get(im)
            faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= float(args.ref_det_thresh)]
            if not faces:
                print(f"[WARN] skipping ref {rp}: no face detected (det_thresh={args.ref_det_thresh})", flush=True); continue
            f = _pick_best_face(faces, im.shape[1], im.shape[0])
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                print(f"[WARN] skipping ref {rp}: no embedding", flush=True); continue
            ref_vecs.append(emb.astype(np.float32))
            ok += 1
            print(f"[OK] ref embedded: {rp}", flush=True)
        except Exception as e:
            print(f"[WARN] skipping ref {rp}: {e}", flush=True)
    if not ref_vecs:
        print("[WARN] No valid reference faces embedded — similarity filter disabled.", flush=True)
        return None
    print(f"[INFO] Embedded {ok}/{len(ref_paths)} refs.", flush=True)
    return np.stack(ref_vecs, axis=0)

# ---------------------- Detection ------------------------


def _pick_best_face(faces, W, H):
    def score(f):
        x1, y1, x2, y2 = map(float, f.bbox)
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        area = (w*h) / (W*H)
        cx = (x1+x2)*0.5; cy = (y1+y2)*0.5
        dx = abs(cx - W/2)/(W/2); dy = abs(cy - H/2)/(H/2)
        center = 1.0 - min(1.0, (dx+dy)*0.5)
        ds = float(getattr(f, "det_score", 0.0))
        return 0.60*area + 0.25*center + 0.15*ds
    return max(faces, key=score)

def _best_by_similarity(faces, ref_mat):
    if ref_mat is None or (hasattr(ref_mat, "size") and ref_mat.size == 0): return None, -1.0
    best, best_s = None, -1.0
    for f in faces:
        emb = getattr(f, "normed_embedding", None)
        if emb is None: continue
        v = emb.astype(np.float32); v /= (np.linalg.norm(v) + 1e-8)
        s = float(np.max(ref_mat @ v))
        if s > best_s: best_s, best = s, f
    return best, best_s

def _get_anchor(face, anchor_mode, offset_y, W, H):
    x1, y1, x2, y2 = map(float, face.bbox)
    bw = max(1.0, x2-x1); bh = max(1.0, y2-y1)
    side = max(bw, bh)
    cx = x1 + 0.5*bw; cy = y1 + 0.5*bh
    if anchor_mode in ("eyes", "nose", "auto"):
        kps = getattr(face, "kps", None)
        if kps is not None:
            k = np.asarray(kps).reshape(-1, 2)
            if anchor_mode == "nose" and len(k) >= 3:
                cx, cy = float(k[2,0]), float(k[2,1])
            else:
                if len(k) >= 2:
                    cx = float(0.5*(k[0,0]+k[1,0])); cy = float(0.5*(k[0,1]+k[1,1]))
    cy = cy + float(offset_y) * side
    cx = max(0, min(cx, W-1)); cy = max(0, min(cy, H-1))
    return cx, cy, side

def _estimate_yaw_deg_from_kps(face):
    """Approx yaw from 5-point landmarks. +deg ~ facing RIGHT, -deg ~ facing LEFT."""
    kps = getattr(face, "kps", None)
    if kps is None: return 0.0
    k = np.asarray(kps).reshape(-1, 2)
    if k.shape[0] < 3: return 0.0
    L, R = k[0], k[1]; N = k[2]
    eye_mid = ((float(L[0]) + float(R[0])) * 0.5, (float(L[1]) + float(R[1])) * 0.5)
    inter = float(np.hypot(R[0]-L[0], R[1]-L[1]) + 1e-6)
    dx = (float(N[0]) - eye_mid[0]) / inter
    yaw = math.degrees(math.atan(1.8 * dx))
    return float(max(-90.0, min(90.0, yaw)))

# ------------------- Cropping helpers --------------------
def resize_square(im, size):
    h, w = im.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LANCZOS4
    return cv2.resize(im, (size, size), interpolation=interp)

def extract_square_centered(frame, cx, cy, side, args):
    """
    Center a square of width=side at (cx,cy).
    Returns (crop, reason) where reason is:
      None (ok), 'skip_edge', 'pad_exceed', 'oob'.
    """
    H, W = frame.shape[:2]
    nx1 = int(round(cx - side/2)); ny1 = int(round(cy - side/2))
    nx2 = int(round(cx + side/2)); ny2 = int(round(cy + side/2))
    padL = max(0, -nx1); padT = max(0, -ny1)
    padR = max(0, nx2 - W); padB = max(0, ny2 - H)

    if padL==padR==padT==padB==0:
        crop = frame[ny1:ny2, nx1:nx2]
        return (crop if crop.size else None), ("oob" if crop.size==0 else None)

    policy = getattr(args, "edge_policy", "pad")

    if policy == "shift":
        nx1 += padL - padR; nx2 += padL - padR
        ny1 += padT - padB; ny2 += padT - padB
        nx1 = max(0, nx1); ny1 = max(0, ny1)
        nx2 = min(W, nx2); ny2 = min(H, ny2)
        if nx2 <= nx1 or ny2 <= ny1: return None, "oob"
        crop = frame[ny1:ny2, nx1:nx2]
        return (crop if crop.size else None), ("oob" if crop.size==0 else None)

    if policy == "skip":
        return None, "skip_edge"

    # pad mode
    miss_x = float(padL + padR) / max(1.0, side)
    miss_y = float(padT + padB) / max(1.0, side)
    if max(miss_x, miss_y) > float(getattr(args, "max_pad_frac", 0.20)):
        return None, "pad_exceed"

    ox1 = max(0, nx1); oy1 = max(0, ny1)
    ox2 = min(W, nx2); oy2 = min(H, ny2)
    if ox2 <= ox1 or oy2 <= oy1: return None, "oob"
    crop = frame[oy1:oy2, ox1:ox2]

    top = oy1 - ny1; left = ox1 - nx1
    bottom = ny2 - oy2; right = nx2 - ox2

    mode = getattr(args, "edge_fill", "reflect")
    if mode == "replicate":
        border = cv2.BORDER_REPLICATE
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, border)
    elif mode == "black":
        border = cv2.BORDER_CONSTANT
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, border, value=(0,0,0))
    else:
        border = cv2.BORDER_REFLECT_101
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, border)
    return crop, None

def _other_face_in_crop(faces, main_face, cx, cy, side, min_frac=0.25):
    """
    True if any secondary face center lies inside the crop square and
    its side >= min_frac * main_side (approximate size check).
    """
    mx1 = cx - side/2.0; my1 = cy - side/2.0
    mx2 = cx + side/2.0; my2 = cy + side/2.0
    x1m, y1m, x2m, y2m = map(float, main_face.bbox)
    main_side = max(x2m - x1m, y2m - y1m)
    for f in faces:
        if f is main_face: continue
        x1, y1, x2, y2 = map(float, f.bbox)
        fx = (x1 + x2) * 0.5; fy = (y1 + y2) * 0.5
        fside = max(x2 - x1, y2 - y1)
        if (mx1 <= fx <= mx2) and (my1 <= fy <= my2):
            if fside >= float(min_frac) * max(1.0, main_side):
                return True
    return False

# ------------------------- CLI ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Per-frame face extractor (stateless) with refs_dir + logging + end-of-run stats")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out", required=True, help="Output aligned dir")
    ap.add_argument("--refs", nargs="+", default=None, help="Reference face images (optional)")
    ap.add_argument("--refs_dir", default=None, help="Directory containing reference images (recursively loaded)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--sim", type=float, default=0.55, help="min cosine sim vs refs; set 0 to ignore")
    ap.add_argument("--max_per_frame", type=int, default=1)
    ap.add_argument("--print_every", type=int, default=1000)
    ap.add_argument("--debug_every", type=int, default=0)

    ap.add_argument("--det_size", type=int, default=960)
    ap.add_argument("--det_thresh", type=float, default=0.30)

    ap.add_argument("--square", type=int, default=1)
    ap.add_argument("--keep_aspect", type=int, default=1)
    ap.add_argument("--anchor", choices=["auto", "bbox", "eyes", "nose"], default="bbox")
    ap.add_argument("--anchor_offset_y", type=float, default=0.10)
    ap.add_argument("--bbox_pad", type=float, default=0.34)

    ap.add_argument("--edge_policy", choices=["shift", "pad", "skip"], default="skip")
    ap.add_argument("--edge_fill", choices=["reflect", "replicate", "black"], default="reflect")
    ap.add_argument("--max_pad_frac", type=float, default=0.18)

    ap.add_argument("--min_crop_px", type=int, default=512)
    ap.add_argument("--max_upscale", type=float, default=1.5)
    ap.add_argument("--min_face_frac_crop", type=float, default=0.30)

    # Back-compat quality args
    ap.add_argument("--smooth", type=int, default=0, help="compat only; extractor is stateless")
    ap.add_argument("--min_bbox_frac", type=float, default=0.0, help="min detected face bbox area / frame area")
    ap.add_argument("--min_tenengrad", type=float, default=0.0, help="min Tenengrad focus measure on pre-resize crop")
    ap.add_argument("--dark_relax", type=float, default=1.0, help="multiply min_tenengrad by this when mean gray < ~60")
    ap.add_argument("--min_brightness", type=float, default=0.0, help="min mean-gray to keep; <=1 for 0..1 scale, >1 for 0..255")

    # Ref preprocessing/detection
    ap.add_argument("--ref_pad", type=float, default=0.60)
    ap.add_argument("--ref_gamma", type=float, default=1.35)
    ap.add_argument("--ref_clahe", type=int, default=1)
    ap.add_argument("--ref_det_thresh", type=float, default=0.30)

    # Yaw-adaptive shift
    ap.add_argument("--yaw_shift", type=int, default=0, help="Enable yaw-adaptive horizontal shift (0/1)")
    ap.add_argument("--yaw_shift_frac", type=float, default=0.12, help="Max |shift| as fraction of face side @ |yaw|>=yawmax")
    ap.add_argument("--yaw_shift_yawmax", type=float, default=45.0, help="Yaw degrees where max shift applies")

    # Solo-face guard
    ap.add_argument("--solo_mode", choices=["off","reject","shrink"], default="reject",
                    help="Handle secondary faces inside crop: off=no check, reject=skip, shrink=reduce pad")
    ap.add_argument("--solo_min_other_frac", type=float, default=0.25,
                    help="Secondary face must be at least this fraction of main face side to count")
    ap.add_argument("--solo_shrink_pad", type=float, default=0.18,
                    help="Pad to try when solo_mode=shrink (fallback pad)")
    ap.add_argument("--log_dir", default=None, help="Directory to write video_logs.csv (default: <out>/../logs)")

    ap.add_argument("--low_cpu", action="store_true", help="Reduce CPU thread usage / disable OpenCL")
    ap.add_argument("--ep", default=None, help="Execution provider order, e.g. \"trt,cuda,cpu\" (default: trt,cuda,cpu)")
    return ap.parse_args()

# --------------------- Logging helper --------------------
def _open_log(out_dir: Path, video_path: str, log_dir: str|None=None):
    from pathlib import Path as _P  # safe local import
    from pathlib import Path as _P  # safe local import
    logs = (_P(log_dir).resolve() if log_dir else (out_dir.parent / 'logs'))
    logs.mkdir(parents=True, exist_ok=True)
    logp = logs / 'video_logs.csv'
    is_new = not logp.exists()
    f = open(logp, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=['file','frame_idx','sim','orig_side','up','face_frac','video_id'])
    if is_new: w.writeheader()
    from os.path import basename, splitext
    vid = splitext(basename(video_path))[0]
    return f, w, vid

def _pct(a, p):
    if not a: return float("nan")
    return float(np.percentile(np.array(a, dtype=np.float32), p))

# ------------------------- Main --------------------------

# ---- runtime helpers (added) ----
def _apply_low_cpu():
    import os
    try:
        import cv2
        cv2.setNumThreads(1)
        try: cv2.ocl.setUseOpenCL(False)
        except Exception: pass
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
    os.environ.setdefault("CV_NUM_THREADS","1")
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME","disabled")
    os.environ.setdefault("OPENCV_OPENCL_DEVICE","disabled")

def _select_providers(args):
    import onnxruntime as ort
    avail = list(ort.get_available_providers())
    # map tokens -> ORT names
    tokmap = {
        "trt":"TensorrtExecutionProvider",
        "tensorrt":"TensorrtExecutionProvider",
        "cuda":"CUDAExecutionProvider",
        "gpu":"CUDAExecutionProvider",
        "cpu":"CPUExecutionProvider",
    }
    if getattr(args, "ep", None):
        order = [tokmap[t.strip().lower()] for t in args.ep.split(",") if t.strip().lower() in tokmap]
    else:
        order = ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]
    chosen = [p for p in order if p in avail] or avail
    print("[INFO] ORT providers available:", avail, flush=True)
    print("[INFO] Using EP order:", chosen, flush=True)
    return chosen
# ---- end helpers ----

def main():
    args = parse_args()
    if args.low_cpu: _apply_low_cpu()
    providers_list = _select_providers(args)
    out_dir = safe_makedirs(args.out)

    # Log file
    _log_f = _log_w = None
    _video_id = ""
    try:
        _log_f, _log_w, _video_id = _open_log(out_dir, args.video, args.log_dir)
    except Exception as _e:
        print(f"[WARN] unable to open video_logs.csv: {_e}", flush=True)

    # Init InsightFace
    app = FaceAnalysis(name="buffalo_l", providers=providers_list)
    try:
        app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh,
                    allowed_modules=["detection", "landmark_2d_106", "recognition"])
    except TypeError:
        app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)
    try:
        print('[INFO] InsightFace DET providers:', app.det_model.session.get_providers(), flush=True)
    except Exception:
        pass
    print(f"[INFO] InsightFace initialized on GPU | det_size={args.det_size} det_thresh={args.det_thresh}", flush=True)

    # Embed references (optional)
    ref_paths = _collect_ref_paths(args)
    ref_mat = _embed_refs(app, ref_paths, args)

    # Stats
    stats = {
        "saved": 0, "no_faces": 0, "below_sim": 0,
        "bbox_frac_reject": 0, "edge_skip": 0, "pad_exceed_skip": 0,
        "face_small_frac": 0, "too_dark": 0, "too_blur": 0,
        "native_too_small": 0, "upscale_too_high": 0, "extract_fail": 0, "crowd": 0
    }
    saved_orig_side, saved_up, saved_face_frac = [], [], []

    # Video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[FATAL] cannot open video: {args.video}", flush=True); sys.exit(2)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3: src_fps = 30.0
    step = max(1, int(round(src_fps / max(1e-3, args.fps))))
    print(f"[INFO] Source FPS ~ {src_fps:.3f} | sampling every {step} frames (~{args.fps:.2f} FPS)", flush=True)

    # progress bar init + hooks
    _dfl_install_save_hooks()
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception:
        total_frames = 0
    __SAMP_TOTAL = ((total_frames + step - 1) // step) if total_frames > 0 else 0
    __BAR__ = _DFLBarX(__SAMP_TOTAL)
    __SAVED = 0

    count = 0
    frame_idx = -1
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        try:
            __BAR__.update(1)
        except Exception:
            pass

        H, W = frame.shape[:2]
        faces = app.get(frame)
        if not faces:
            stats["no_faces"] += 1
            if args.debug_every and (frame_idx % args.debug_every == 0):
                print(f"[DEBUG] frame {frame_idx}: no faces | gray mean={mean_gray(frame):.1f}", flush=True)
            continue

        # choose face (by refs if provided)
        sim_val = -1.0
        if ref_mat is not None and args.sim > 0:
            face, sim_val = _best_by_similarity(faces, ref_mat)
            if face is None or sim_val < float(args.sim):
                stats["below_sim"] += 1
                if args.debug_every and (frame_idx % args.debug_every == 0):
                    print(f"[DEBUG] frame {frame_idx}: below sim ({sim_val:.3f} < {args.sim})", flush=True)
                continue
        else:
            face = _pick_best_face(faces, W, H)

        # bbox area fraction guard
        if float(args.min_bbox_frac) > 0.0:
            x1,y1,x2,y2 = map(float, face.bbox)
            bbox_frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if bbox_frac < float(args.min_bbox_frac):
                stats["bbox_frac_reject"] += 1
                if args.debug_every and (frame_idx % args.debug_every == 0):
                    print(f"[DEBUG] frame {frame_idx}: bbox_frac={bbox_frac:.3f} < {args.min_bbox_frac}", flush=True)
                continue

        # anchor + yaw-adaptive shift
        cx_det, cy_det, side_det = _get_anchor(face, args.anchor, args.anchor_offset_y, W, H)
        if args.yaw_shift:
            yaw = _estimate_yaw_deg_from_kps(face)
            frac = min(1.0, abs(yaw) / max(1e-3, float(args.yaw_shift_yawmax))) * float(args.yaw_shift_frac)
            direction = -1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0)
            cx_det = cx_det + direction * (frac * side_det)
            cx_det = max(0.0, min(cx_det, W - 1.0))

        # base square side (with pad)
        side_use = side_det * (1.0 + 2.0 * float(args.bbox_pad))

        # Solo-face guard: reject or shrink if another face intrudes
        if args.solo_mode != 'off':
            if _other_face_in_crop(faces, face, cx_det, cy_det, side_use, args.solo_min_other_frac):
                if args.solo_mode == 'reject':
                    stats['crowd'] += 1
                    continue
                else:
                    side_try = side_det * (1.0 + 2.0 * float(args.solo_shrink_pad))
                    if _other_face_in_crop(faces, face, cx_det, cy_det, side_try, args.solo_min_other_frac):
                        stats['crowd'] += 1
                        continue
                    else:
                        side_use = side_try

        # crop according to edge policy
        crop, edge_reason = extract_square_centered(frame, cx_det, cy_det, side_use, args)
        if crop is None:
            if edge_reason == "skip_edge":
                stats["edge_skip"] += 1
            elif edge_reason == "pad_exceed":
                stats["pad_exceed_skip"] += 1
            else:
                stats["extract_fail"] += 1
            continue

        # face must not be too tiny in crop
        face_frac = side_det / max(1e-6, side_use)
        if float(args.min_face_frac_crop) > 0.0 and face_frac < float(args.min_face_frac_crop):
            stats["face_small_frac"] += 1
            continue

        # Quality: brightness + Tenengrad on pre-resize crop
        if float(args.min_brightness) > 0.0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mg = float(np.mean(gray))
            thrB = (args.min_brightness if args.min_brightness > 1.0 else args.min_brightness * 255.0)
            if mg < thrB:
                stats["too_dark"] += 1
                if args.debug_every and (frame_idx % args.debug_every == 0):
                    print(f"[DEBUG] frame {frame_idx}: too dark mean={mg:.1f} < {thrB:.1f}", flush=True)
                continue
        if float(args.min_tenengrad) > 0.0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ten = float(np.mean(gx*gx + gy*gy))
            mg = float(np.mean(gray))
            eff = float(args.min_tenengrad) * (float(args.dark_relax) if mg < 60.0 else 1.0)
            if ten < eff:
                stats["too_blur"] += 1
                if args.debug_every and (frame_idx % args.debug_every == 0):
                    print(f"[DEBUG] frame {frame_idx}: blur ten={ten:.1f} < {eff:.1f} (mg={mg:.1f})", flush=True)
                continue

        # PRE-RESIZE checks
        orig_side = max(crop.shape[:2])
        if orig_side < int(args.min_crop_px):
            stats["native_too_small"] += 1
            continue
        up = float(args.size) / float(orig_side)
        if up > float(args.max_upscale):
            stats["upscale_too_high"] += 1
            continue

        # resize + save PNG
        crop = resize_square(crop, args.size)
        count += 1
        stats["saved"] = count

        out_path = out_dir / f"{count:06d}.png"
        buf = cv2.imencode('.png', crop)[1]
        buf.tofile(str(out_path))
        try:
            __SAVED += 1
            __BAR__.set_saved(__SAVED)
        except Exception:
            pass# append to video_logs.csv
        try:
            if _log_w is not None:
                _log_w.writerow({
                    "file": f"{count:06d}.png",
                    "frame_idx": int(frame_idx),
                    "sim": (float(sim_val) if sim_val >= 0 else ""),
                    "orig_side": int(orig_side),
                    "up": float(up),
                    "face_frac": float(face_frac),
                    "video_id": _video_id,
                })
        except Exception as _e:
            print(f"[WARN] log write failed: {_e}", flush=True)

        if args.print_every and (count % args.print_every == 0):
            dt = time.time() - t0
            print(f"[INFO] {count} saved | {count / max(dt,1e-6):.1f} img/s", flush=True)

        if args.debug_every:
            dbg = f"sim={sim_val:.3f}" if sim_val >= 0 else "sim=n/a"
            print(f"[DEBUG] saved={count} frame={frame_idx} {dbg} orig_side={orig_side} up={up:.2f}x face_frac={face_frac:.2f}", flush=True)

    cap.release()
    try:
        __BAR__.close()
    except Exception:
        pass
    try:
        if _log_f is not None:
            _log_f.close()
    except Exception:
        pass

    # --------- End-of-run stats ---------
    up_arr = list(saved_up)
    os_arr = list(saved_orig_side)
    ff_arr = list(saved_face_frac)

    up_gt125 = sum(1 for u in up_arr if u > 1.25)
    up_gt150 = sum(1 for u in up_arr if u > 1.50)
    os_lt480 = sum(1 for s in os_arr if s < 480)
    os_ge512 = sum(1 for s in os_arr if s >= 512)

    print(f"[DONE] saved={count} → {out_dir}", flush=True)
    print("[STATS] saves:", count, flush=True)
    if count > 0:
        print(f"[STATS] native side (pre-resize) px: p10={_pct(os_arr,10):.0f}  p50={_pct(os_arr,50):.0f}  p90={_pct(os_arr,90):.0f}", flush=True)
        print(f"[STATS] upscale factor (target/orig): p10={_pct(up_arr,10):.2f}  p50={_pct(up_arr,50):.2f}  p90={_pct(up_arr,90):.2f}", flush=True)
        print(f"[STATS] shares: up>1.25={up_gt125}/{count} ({100*up_gt125/max(count,1):.1f}%),  up>1.50={up_gt150}/{count} ({100*up_gt150/max(count,1):.1f}%)", flush=True)
        print(f"[STATS] shares: native<480px={os_lt480}/{count} ({100*os_lt480/max(count,1):.1f}%),  native≥512px={os_ge512}/{count} ({100*os_ge512/max(count,1):.1f}%)", flush=True)
        if ff_arr:
            print(f"[STATS] face_frac (face side / crop side): p10={_pct(ff_arr,10):.2f}  p50={_pct(ff_arr,50):.2f}  p90={_pct(ff_arr,90):.2f}", flush=True)

    rej_total = (stats["no_faces"] + stats["below_sim"] + stats["bbox_frac_reject"] + stats["edge_skip"] +
                 stats["pad_exceed_skip"] + stats["face_small_frac"] + stats["too_dark"] + stats["too_blur"] +
                 stats["native_too_small"] + stats["upscale_too_high"] + stats["extract_fail"] + stats["crowd"])
    if rej_total:
        print("[REJECTS]", flush=True)
        for k in ["no_faces","below_sim","bbox_frac_reject","edge_skip","pad_exceed_skip",
                  "face_small_frac","too_dark","too_blur","native_too_small","upscale_too_high","extract_fail","crowd"]:
            v = stats[k]
            if v:
                print(f" - {k} = {v}", flush=True)

if __name__ == "__main__":
    main()
