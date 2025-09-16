#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_extract.py — renamed from extract_face_directly.py

Surgical cleanup (no core logic changes):
- Auto-help filename: write /workspace/scripts/face_extract.txt (not *.py.txt).
- Remove manual divider prints; dividers now handled by progress.py by default.
- Keep ETA alignment via progress.py; keep warning suppression.
"""

import argparse, sys, time, math, os, csv, shutil
from pathlib import Path
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Use shared progress style
from progress import ProgressBar

# ---------------------------
# COMPAT: legacy args shims
# ---------------------------
# Accept legacy "--log" as alias for "--log_dir"
for _i, _a in enumerate(list(sys.argv)):
    if _a == '--log':
        sys.argv[_i] = '--log_dir'
    elif _a.startswith('--log='):
        sys.argv[_i] = '--log_dir=' + _a.split('=',1)[1]

# --- [MSCALE SHIM START] ---
# Parse optional --mscale from argv, but delay any InsightFace patching until after import.
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
        _skip = True
        continue
    if a.startswith("--mscale="):
        try:
            __MSCALE = float(a.split("=",1)[1])
        except Exception:
            pass
        continue
    _newargv.append(a)
_sys.argv = _newargv

def _patch_mscale_on_faceanalysis(FaceAnalysis):
    """Patch FaceAnalysis.get to retry at scaled resolution when no faces found. Silent (no prints)."""
    try:
        import cv2 as _cv2
        _orig_get = FaceAnalysis.get
        if getattr(FaceAnalysis, "_mscale_patched", False):
            return
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
        FaceAnalysis.get = _patched_get
        FaceAnalysis._mscale_patched = True
    except Exception:
        pass
# --- [MSCALE SHIM END] ---

# -------------------------
# Progress save hooks
# -------------------------
# Count saves from BOTH cv2.imwrite and numpy.ndarray.tofile (covers imencode(...).tofile(...))
__SAVED_GLOBAL = 0

def _install_save_hooks():
    global __SAVED_GLOBAL
    try:
        _orig = cv2.imwrite
        def _wrap(path, img):
            ok = _orig(path, img)
            if ok:
                try:
                    __SAVED_GLOBAL += 1
                except Exception:
                    pass
            return ok
        cv2.imwrite = _wrap
    except Exception:
        pass
    try:
        _orig_tofile = np.ndarray.tofile
        def _wrap_tofile(self, *a, **k):
            out = _orig_tofile(self, *a, **k)
            try:
                __SAVED_GLOBAL += 1
            except Exception:
                pass
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
        if img is None:
            return float("nan")
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
    if g is None or abs(float(g) - 1.0) < 1e-6:
        return im
    lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / float(g))) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(im, lut)

def _preprocess_ref(im, pad_frac=0.0, gamma=None, clahe=0, **_):
    if im is None:
        return None
    out = im
    if int(clahe):
        out = _clahe_bgr(out)
    if gamma is not None:
        out = _gamma_bgr(out, float(gamma))
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
    if ref_mat is None or (hasattr(ref_mat, "size") and ref_mat.size == 0):
        return None, -1.0
    best, best_s = None, -1.0
    for f in faces:
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            continue
        v = emb.astype(np.float32); v /= (np.linalg.norm(v) + 1e-8)
        s = float(np.max(ref_mat @ v))
        if s > best_s:
            best_s, best = s, f
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
    """Approx yaw from 5-point landmarks. +deg ~ RIGHT, -deg ~ LEFT."""
    kps = getattr(face, "kps", None)
    if kps is None:
        return 0.0
    k = np.asarray(kps).reshape(-1, 2)
    if k.shape[0] < 3:
        return 0.0
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
    """ Center a square of width=side at (cx,cy).
    Returns (crop, reason) where reason is: None (ok), 'skip_edge', 'pad_exceed', 'oob'.
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
        if nx2 <= nx1 or ny2 <= ny1:
            return None, "oob"
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
    if ox2 <= ox1 or oy2 <= oy1:
        return None, "oob"

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
    """ True if any secondary face center lies inside the crop square and its side >= min_frac * main_side (approximate size check). """
    mx1 = cx - side/2.0; my1 = cy - side/2.0
    mx2 = cx + side/2.0; my2 = cy + side/2.0
    x1m, y1m, x2m, y2m = map(float, main_face.bbox)
    main_side = max(x2m - x1m, y2m - y1m)
    for f in faces:
        if f is main_face:
            continue
        x1, y1, x2, y2 = map(float, f.bbox)
        fx = (x1 + x2) * 0.5; fy = (y1 + y2) * 0.5
        fside = max(x2 - x1, y2 - y1)
        if (mx1 <= fx <= mx2) and (my1 <= fy <= my2):
            if fside >= float(min_frac) * max(1.0, main_side):
                return True
    return False

# ------------------------- CLI ---------------------------
AUTO_HELP_TEXT = r"""
PURPOSE
  • Sample a video at --fps (default 8), detect faces (InsightFace FaceAnalysis), and save aligned square crops
    to --out as PNGs named 000001.png, 000002.png, …
  • If --refs / --refs_dir are given and --sim>0, compute cosine similarity vs reference embeddings and keep a
    frame only if the BEST face meets sim ≥ --sim (default 0.55).
  • Crop geometry controls:
      - anchor: bbox (default), eyes, nose (+ --anchor_offset_y vertical bias)
      - sizing: square with pad via --bbox_pad (default 0.34)
      - edges: --edge_policy {shift|pad|skip} (default skip) with --edge_fill and --max_pad_frac
      - yaw-adaptive horizontal shift (optional): --yaw_shift*
      - single-face guard: --solo_mode {off|reject|shrink} + thresholds
  • Quality gates before saving: --min_crop_px, --max_upscale, --min_face_frac_crop,
    plus legacy blur/brightness guards (Tenengrad & mean gray).
  • Shows a one-line progress bar with FPS and SAVED count (self-contained via progress.py).

USAGE (common)
  python3 -u /workspace/scripts/face_extract.py \
    --video /path/to/video.mp4 \
    --out   /workspace/data_src/aligned \
    --fps 8 --size 512 \
    --refs_dir /workspace/refs --sim 0.55 \
    --det_size 960 --det_thresh 0.30 \
    --edge_policy skip --bbox_pad 0.34 \
    --min_crop_px 512 --max_upscale 1.5 --min_face_frac_crop 0.30

CLI FLAGS
  --video (req)                  input video path
  --out   (req)                  output aligned dir

  --refs [files...]              reference face images
  --refs_dir DIR                 directory of reference images (recursive)
  --size  INT (512)              output crop size (square)
  --fps   FLOAT (8.0)            sampling rate (frames per second)
  --sim   FLOAT (0.55)           min cosine sim vs refs; set 0 to ignore refs
  --max_per_frame INT (1)        max crops per sampled frame

  # detection / runtime
  --det_size INT (960)           detection canvas
  --det_thresh FLOAT (0.30)      detector threshold
  --low_cpu on|off (off)         reduce threads / prefer CPU provider
  --ep "trt,cuda,cpu"            execution provider order (ORT providers)

  # crop geometry
  --anchor {auto,bbox,eyes,nose} (bbox)
  --anchor_offset_y FLOAT (0.10)
  --bbox_pad FLOAT (0.34)
  --edge_policy {shift,pad,skip} (skip)
  --edge_fill {reflect,replicate,black} (reflect)
  --max_pad_frac FLOAT (0.18)

  # quality guards
  --min_crop_px INT (512)
  --max_upscale FLOAT (1.5)
  --min_face_frac_crop FLOAT (0.30)
  --min_bbox_frac FLOAT (0.0)          min face bbox area / frame area
  --min_tenengrad FLOAT (0.0)          blur gate on pre-resize crop
  --dark_relax FLOAT (1.0)             relax blur gate if dark
  --min_brightness FLOAT (0.0)         mean-gray gate

  # references preprocessing
  --ref_pad FLOAT (0.60)
  --ref_gamma FLOAT (1.35)
  --ref_clahe on|off (on)
  --ref_det_thresh FLOAT (0.30)

  # yaw / multi-face handling
  --yaw_shift on|off (off)             enable yaw-adaptive shift
  --yaw_shift_frac FLOAT (0.12)
  --yaw_shift_yawmax FLOAT (45.0)
  --solo_mode {off,reject,shrink} (reject)
  --solo_min_other_frac FLOAT (0.25)
  --solo_shrink_pad FLOAT (0.18)

  # logging & help
  --print_every INT (1000)
  --debug_every INT (0)
  --log_dir DIR                         where to write video_stats.csv (default parent(--out)/logs/video)
  --help on|off (on)                    write /workspace/scripts/face_extract.txt then proceed

LOG OUTPUT
  <logs>/video/video_stats.csv   (append mode; header is written if the file is new)

  Location:
    • If --log_dir is set → <log_dir>/video/video_stats.csv
    • Otherwise           → parent(--out)/logs/video/video_stats.csv
      Example: --out /workspace/data_src/aligned → /workspace/data_src/logs/video/video_stats.csv

  One row per SAVED crop; columns (exact order):
    file, frame_idx, sim, orig_side, up, face_frac, video_id
"""


def _parse_onoff(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"1","on","true","yes","y"}:
        return True
    if s in {"0","off","false","no","n"}:
        return False
    return bool(default)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-frame face extractor (stateless) with refs_dir + logging + end-of-run stats",
        add_help=False,
    )
    # manual -h/--help to avoid argparse exiting
    ap.add_argument('-h', '--h', dest='show_cli_help', action='store_true')

    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out", required=True, help="Output aligned dir")

    ap.add_argument("--refs", nargs="+", default=None, help="Reference face images (optional)")
    ap.add_argument("--refs_dir", default=None, help="Directory containing reference images (recursively loaded)")

    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--sim", type=float, default=0.55, help="min cosine sim vs refs; set 0 to ignore")
    ap.add_argument("--max_per_frame", type=int, default=1)

    # detection / runtime
    ap.add_argument("--det_size", type=int, default=960)
    ap.add_argument("--det_thresh", type=float, default=0.30)
    ap.add_argument("--low_cpu", default="off", help="on|off")
    ap.add_argument("--ep", default=None, help="Execution provider order, e.g. \"trt,cuda,cpu\"")

    # crop geometry
    ap.add_argument("--anchor", choices=["auto","bbox","eyes","nose"], default="bbox")
    ap.add_argument("--anchor_offset_y", type=float, default=0.10)
    ap.add_argument("--bbox_pad", type=float, default=0.34)
    ap.add_argument("--edge_policy", choices=["shift","pad","skip"], default="skip")
    ap.add_argument("--edge_fill", choices=["reflect","replicate","black"], default="reflect")
    ap.add_argument("--max_pad_frac", type=float, default=0.18)

    # quality guards
    ap.add_argument("--min_crop_px", type=int, default=512)
    ap.add_argument("--max_upscale", type=float, default=1.5)
    ap.add_argument("--min_face_frac_crop", type=float, default=0.30)
    ap.add_argument("--min_bbox_frac", type=float, default=0.0, help="min detected face bbox area / frame area")
    ap.add_argument("--min_tenengrad", type=float, default=0.0, help="min Tenengrad focus measure on pre-resize crop")
    ap.add_argument("--dark_relax", type=float, default=1.0, help="multiply min_tenengrad by this when mean gray < ~60")
    ap.add_argument("--min_brightness", type=float, default=0.0, help="min mean-gray to keep; <=1 for 0..1 scale, >1 for 0..255")

    # Ref preprocessing/detection
    ap.add_argument("--ref_pad", type=float, default=0.60)
    ap.add_argument("--ref_gamma", type=float, default=1.35)
    ap.add_argument("--ref_clahe", default="on", help="on|off")
    ap.add_argument("--ref_det_thresh", type=float, default=0.30)

    # Yaw-adaptive shift
    ap.add_argument("--yaw_shift", default="off", help="on|off")
    ap.add_argument("--yaw_shift_frac", type=float, default=0.12,
                    help="Max |shift| as fraction of face side @ |yaw|>=yawmax")
    ap.add_argument("--yaw_shift_yawmax", type=float, default=45.0,
                    help="Yaw degrees where max shift applies")

    # Solo-face guard
    ap.add_argument("--solo_mode", choices=["off","reject","shrink"], default="reject",
                    help="Handle secondary faces inside crop: off=no check, reject=skip, shrink=reduce pad")
    ap.add_argument("--solo_min_other_frac", type=float, default=0.25,
                    help="Secondary face must be at least this fraction of main face side to count")
    ap.add_argument("--solo_shrink_pad", type=float, default=0.18,
                    help="Pad to try when solo_mode=shrink (fallback pad)")

    # logging & debug
    ap.add_argument("--print_every", type=int, default=1000)
    ap.add_argument("--debug_every", type=int, default=0)
    ap.add_argument("--log_dir", default=None, help="Directory to write video_stats.csv (default parent(--out)/logs/video)")
    ap.add_argument("--help", dest="auto_help", default="on", help="on|off — write face_extract.txt then proceed")

    return ap.parse_args()

# --------------------- Logging helper --------------------
def _open_log(out_dir: Path, video_path: str, log_dir: str|None=None):
    logs_root = (Path(log_dir).resolve() if log_dir else (out_dir.parent / 'logs'))
    video_dir = logs_root / 'video'
    video_dir.mkdir(parents=True, exist_ok=True)
    logp = video_dir / 'video_stats.csv'
    is_new = not logp.exists()
    f = open(logp, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=['file','frame_idx','sim','orig_side','up','face_frac','video_id'])
    if is_new:
        w.writeheader()
    from os.path import basename, splitext
    vid = splitext(basename(video_path))[0]
    return f, w, vid

# ------------------ Quiet logging context ----------------
class _Quiet:
    def __enter__(self):
        self._null = open(os.devnull, 'w')
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = self._null
        sys.stderr = self._null
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = self._old_out, self._old_err
        try:
            self._null.close()
        except Exception:
            pass

# ------------------------- Main --------------------------
# ---- runtime helpers (kept) ----
def _apply_low_cpu():
    try:
        cv2.setNumThreads(1)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
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
    try:
        import onnxruntime as ort
        avail = list(ort.get_available_providers())
    except Exception:
        avail = ["CPUExecutionProvider"]
    tokmap = {
        "trt":"TensorrtExecutionProvider",
        "tensorrt":"TensorrtExecutionProvider",
        "cuda":"CUDAExecutionProvider",
        "gpu":"CUDAExecutionProvider",
        "cpu":"CPUExecutionProvider",
    }
    if getattr(args, "ep", None):
        order = [tokmap[t.strip().lower()] for t in str(args.ep).split(",") if t.strip().lower() in tokmap]
    else:
        order = ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]
    chosen = [p for p in order if p in avail] or avail
    return chosen

# ---- end helpers ----

def _normalize_args(args):
    args.low_cpu = _parse_onoff(getattr(args, 'low_cpu', 'off'), default=False)
    args.ref_clahe = 1 if _parse_onoff(getattr(args, 'ref_clahe', 'on'), default=True) else 0
    args.yaw_shift = 1 if _parse_onoff(getattr(args, 'yaw_shift', 'off'), default=False) else 0
    args.auto_help = _parse_onoff(getattr(args, 'auto_help', 'on'), default=True)
    return args


def _write_auto_help(script_path: Path, enabled=True):
    if not enabled:
        return
    try:
        out_txt = script_path.with_name(script_path.stem + '.txt')
        out_txt.write_text(AUTO_HELP_TEXT, encoding='utf-8')
    except Exception:
        pass


def _embed_refs_with_bar(app, ref_paths, args):
    """Return ref_mat (Nx512) or None; draws a single REFS progress row.
    FAIL shows number of rejected refs. No per-ref spam.
    """
    if not ref_paths:
        return None

    bar = ProgressBar(label="REFS", total=len(ref_paths), show_fail_label=True, show_fps=False, fail_label="FAIL")
    ref_vecs = []
    fails = 0
    for i, rp in enumerate(ref_paths, 1):
        ok = False
        try:
            im = imread_bgr(Path(rp))
            im = _preprocess_ref(im, pad_frac=args.ref_pad, gamma=args.ref_gamma, clahe=args.ref_clahe)
            if im is not None:
                faces = app.get(im)
                faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= float(args.ref_det_thresh)]
                if faces:
                    f = _pick_best_face(faces, im.shape[1], im.shape[0])
                    emb = getattr(f, "normed_embedding", None)
                    if emb is not None:
                        ref_vecs.append(emb.astype(np.float32))
                        ok = True
        except Exception:
            ok = False
        if not ok:
            fails += 1
        bar.update(i, fails=fails)  # no FPS
    bar.close()

    if not ref_vecs:
        return None
    return np.stack(ref_vecs, axis=0)


def main():
    # lower third-party noise
    os.environ.setdefault('ORT_LOG_SEVERITY_LEVEL', '3')  # onnxruntime: 0-4 (4=fatal)
    os.environ.setdefault('INSIGHTFACE_DISABLE_ANALYTICS', '1')

    args = parse_args()
    if getattr(args, 'show_cli_help', False):
        print(AUTO_HELP_TEXT)
    args = _normalize_args(args)

    if args.low_cpu:
        _apply_low_cpu()

    script_path = Path(__file__).resolve()
    _write_auto_help(script_path, enabled=args.auto_help)

    out_dir = safe_makedirs(Path(args.out))

    # Open log file
    _log_f = _log_w = None
    _video_id = ""
    try:
        _log_f, _log_w, _video_id = _open_log(out_dir, args.video, args.log_dir)
    except Exception as _e:
        pass

    # Init InsightFace quietly
    providers_list = _select_providers(args)
    from insightface.app import FaceAnalysis
    _patch_mscale_on_faceanalysis(FaceAnalysis)
    with _Quiet():
        app = FaceAnalysis(name="buffalo_l", providers=providers_list)
        try:
            app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh,
                        allowed_modules=["detection", "landmark_2d_106", "recognition"])
        except TypeError:
            app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)

    # Embed references with a single progress row
    ref_paths = _collect_ref_paths(args)
    ref_mat = None
    if ref_paths and float(getattr(args, 'sim', 0.0)) > 0.0:
        ref_mat = _embed_refs_with_bar(app, ref_paths, args)

    # Stats (unchanged semantics)
    stats = {
        "saved": 0,
        "no_faces": 0,
        "below_sim": 0,
        "bbox_frac_reject": 0,
        "edge_skip": 0,
        "pad_exceed_skip": 0,
        "face_small_frac": 0,
        "too_dark": 0,
        "too_blur": 0,
        "native_too_small": 0,
        "upscale_too_high": 0,
        "extract_fail": 0,
        "crowd": 0,
    }

    saved_orig_side, saved_up, saved_face_frac = [], [], []

    # Video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[FATAL] cannot open video: {args.video}")
        sys.exit(2)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3:
        src_fps = 30.0
    step = max(1, int(round(src_fps / max(1e-3, args.fps))))

    # progress bar init + hooks
    _install_save_hooks()
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception:
        total_frames = 0
    samp_total = ((total_frames + step - 1) // step) if total_frames > 0 else 0
    bar = ProgressBar(label="EXTRACT", total=max(1, samp_total), show_fail_label=True, fail_label="FACE", show_fps=True)

    count = 0
    processed_samples = 0
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        processed_samples += 1

        H, W = frame.shape[:2]
        faces = app.get(frame)
        if not faces:
            stats["no_faces"] += 1
            bar.update(processed_samples, fails=count)  # FACE shows number of saved frames
            continue

        sim_val = -1.0
        if ref_mat is not None and args.sim > 0:
            face, sim_val = _best_by_similarity(faces, ref_mat)
            if face is None or sim_val < float(args.sim):
                stats["below_sim"] += 1
                bar.update(processed_samples, fails=count)
                continue
        else:
            face = _pick_best_face(faces, W, H)

        if float(args.min_bbox_frac) > 0.0:
            x1,y1,x2,y2 = map(float, face.bbox)
            bbox_frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if bbox_frac < float(args.min_bbox_frac):
                stats["bbox_frac_reject"] += 1
                bar.update(processed_samples, fails=count)
                continue

        cx_det, cy_det, side_det = _get_anchor(face, args.anchor, args.anchor_offset_y, W, H)
        if args.yaw_shift:
            yaw = _estimate_yaw_deg_from_kps(face)
            frac = min(1.0, abs(yaw) / max(1e-3, float(args.yaw_shift_yawmax))) * float(args.yaw_shift_frac)
            direction = -1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0)
            cx_det = cx_det + direction * (frac * side_det)
            cx_det = max(0.0, min(cx_det, W - 1.0))

        side_use = side_det * (1.0 + 2.0 * float(args.bbox_pad))

        if args.solo_mode != 'off':
            if _other_face_in_crop(faces, face, cx_det, cy_det, side_use, args.solo_min_other_frac):
                if args.solo_mode == 'reject':
                    stats['crowd'] += 1
                    bar.update(processed_samples, fails=count)
                    continue
                else:
                    side_try = side_det * (1.0 + 2.0 * float(args.solo_shrink_pad))
                    if _other_face_in_crop(faces, face, cx_det, cy_det, side_try, args.solo_min_other_frac):
                        stats['crowd'] += 1
                        bar.update(processed_samples, fails=count)
                        continue
                    else:
                        side_use = side_try

        crop, edge_reason = extract_square_centered(frame, cx_det, cy_det, side_use, args)
        if crop is None:
            if edge_reason == "skip_edge":
                stats["edge_skip"] += 1
            elif edge_reason == "pad_exceed":
                stats["pad_exceed_skip"] += 1
            else:
                stats["extract_fail"] += 1
            bar.update(processed_samples, fails=count)
            continue

        face_frac = side_det / max(1e-6, side_use)
        if float(args.min_face_frac_crop) > 0.0 and face_frac < float(args.min_face_frac_crop):
            stats["face_small_frac"] += 1
            bar.update(processed_samples, fails=count)
            continue

        if float(args.min_brightness) > 0.0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mg = float(np.mean(gray))
            thrB = (args.min_brightness if args.min_brightness > 1.0 else args.min_brightness * 255.0)
            if mg < thrB:
                stats["too_dark"] += 1
                bar.update(processed_samples, fails=count)
                continue
        if float(args.min_tenengrad) > 0.0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ten = float(np.mean(gx*gx + gy*gy))
            mg = float(np.mean(gray))
            eff = float(args.min_tenengrad) * (float(args.dark_relax) if mg < 60.0 else 1.0)
            if ten < eff:
                stats["too_blur"] += 1
                bar.update(processed_samples, fails=count)
                continue

        orig_side = max(crop.shape[:2])
        if orig_side < int(args.min_crop_px):
            stats["native_too_small"] += 1
            bar.update(processed_samples, fails=count)
            continue

        up = float(args.size) / float(orig_side)
        if up > float(args.max_upscale):
            stats["upscale_too_high"] += 1
            bar.update(processed_samples, fails=count)
            continue

        crop = resize_square(crop, args.size)
        count += 1
        stats["saved"] = count
        out_path = out_dir / f"{count:06d}.png"
        buf = cv2.imencode('.png', crop)[1]
        buf.tofile(str(out_path))

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
        except Exception:
            pass

        bar.update(processed_samples, fails=count)

    cap.release()
    bar.close()

    up_arr = list(saved_up)
    os_arr = list(saved_orig_side)
    ff_arr = list(saved_face_frac)

    up_gt125 = sum(1 for u in up_arr if u > 1.25)
    up_gt150 = sum(1 for u in up_arr if u > 1.50)
    os_lt480 = sum(1 for s in os_arr if s < 480)
    os_ge512 = sum(1 for s in os_arr if s >= 512)

    print(f"[DONE] saved={count} → {out_dir}")
    print("[STATS] saves:", count)
    if count > 0:
        def _pct(a, p):
            if not a:
                return float("nan")
            return float(np.percentile(np.array(a, dtype=np.float32), p))
        print(f"[STATS] native side (pre-resize) px: p10={_pct(os_arr,10):.0f} p50={_pct(os_arr,50):.0f} p90={_pct(os_arr,90):.0f}")
        print(f"[STATS] upscale factor (target/orig): p10={_pct(up_arr,10):.2f} p50={_pct(up_arr,50):.2f} p90={_pct(up_arr,90):.2f}")
        print(f"[STATS] shares: up>1.25={up_gt125}/{count} ({100*up_gt125/max(count,1):.1f}%), up>1.50={up_gt150}/{count} ({100*up_gt150/max(count,1):.1f}%)")
        print(f"[STATS] shares: native<480px={os_lt480}/{count} ({100*os_lt480/max(count,1):.1f}%), native≥512px={os_ge512}/{count} ({100*os_ge512/max(count,1):.1f}%)")
        if ff_arr:
            print(f"[STATS] face_frac (face side / crop side): p10={_pct(ff_arr,10):.2f} p50={_pct(ff_arr,50):.2f} p90={_pct(ff_arr,90):.2f}")

    rej_total = (
        stats["no_faces"] + stats["below_sim"] + stats["bbox_frac_reject"] +
        stats["edge_skip"] + stats["pad_exceed_skip"] + stats["face_small_frac"] +
        stats["too_dark"] + stats["too_blur"] + stats["native_too_small"] +
        stats["upscale_too_high"] + stats["extract_fail"] + stats["crowd"]
    )
    if rej_total:
        print("[REJECTS]")
        for k in [
            "no_faces","below_sim","bbox_frac_reject","edge_skip","pad_exceed_skip",
            "face_small_frac","too_dark","too_blur","native_too_small","upscale_too_high","extract_fail","crowd"
        ]:
            v = stats[k]
            if v:
                print(f" - {k} = {v}")


if __name__ == "__main__":
    main()
