#!/usr/bin/env python3
"""
face_extract.py — per-frame face extractor (stateless)

Requirements from spec:
- Keep extraction behavior aligned with prior extract_face_directly.py: detection → selection → crop → quality gates
  (sim vs refs, bbox/face frac, edge policy, brightness, Tenengrad/blur, upscale limits, solo handling, yaw shift).
- Run in default ort_env (no custom env switching).
- Logging path: <logs>/video/video_stats.csv (append-only), schema:
    file,frame_idx,sim,orig_side,up,face_frac,video_id
- Progress UI must use shared progress.py only (no custom prints):
    • REFS bar first (label width 10, bar width 26), shows FAIL count (rejected refs), no FPS
    • EXTRACT bar second (label width 10, bar width 26), counter label FACE = saved frames
- Suppress provider/model chatter (onnxruntime / insightface)
- Auto-help: --help on|off (default on) writes /workspace/scripts/face_extract.py.txt
- Refs handling: support --refs / --refs_dir; embed once; sim gate applies iff refs exist; otherwise disabled.

Notes:
- This script is self-contained and conservative about third-party logging.
- If some optional features (e.g., exact yaw shift) cannot be derived from detector output,
  they degrade gracefully (no shift).
"""
from __future__ import annotations
import os
import sys
import csv
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# --- silence noisy runtimes as much as possible ---
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")  # ERROR only
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("INSIGHTFACE_DISABLE_NUMBA", "1")
warnings.filterwarnings("ignore")

# --- progress bar (shared) ---
try:
    from progress import ProgressBar
except Exception:
    class ProgressBar:  # minimal fallback to avoid crashes if missing
        def __init__(self, *a, **k):
            self.total = k.get("total", 1)
        def update(self, *a, **k):
            pass
        def close(self):
            pass

# --- argparse with on/off style booleans ---
import argparse

def onoff(v: str) -> bool:
    v = str(v).strip().lower()
    if v in ("on", "true", "1", "yes", "y"): return True
    if v in ("off", "false", "0", "no", "n"): return False
    raise argparse.ArgumentTypeError("expected on|off")

# --- utilities ---
import numpy as np
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

@dataclass
class RefPack:
    embs: Optional[np.ndarray]  # (R,512) normed caps (or None)

# suppress stdout/stderr context for noisy inits
from contextlib import contextmanager
@contextmanager
def suppress_stdout_stderr():
    class DevNull:
        def write(self, *_):
            pass
        def flush(self):
            pass
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = DevNull()
        sys.stderr = DevNull()
        yield
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err

# --- InsightFace FaceAnalysis loader ---

def build_providers(ep: str) -> List[str]:
    mapping = {
        "trt": "TensorrtExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
        "dml": "DmlExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
    }
    out = []
    for token in [t.strip().lower() for t in ep.split(",") if t.strip()]:
        out.append(mapping.get(token, token))
    if not out:
        out = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return out


def load_app(ep_order: str, det_size: int, low_cpu: bool):
    from insightface.app import FaceAnalysis
    providers = build_providers(ep_order)
    # Low CPU hint: drop CUDA/DML if requested
    if low_cpu:
        providers = [p for p in providers if p.endswith("CPUExecutionProvider")] or ["CPUExecutionProvider"]
    with suppress_stdout_stderr():
        app = FaceAnalysis(name="antelopev2", providers=providers)
        app.prepare(ctx_id=0 if (not low_cpu and "CUDAExecutionProvider" in providers) else -1,
                    det_size=(det_size, det_size))
    return app

# --- reference embeddings ---

def list_ref_images(refs: List[str] | None, refs_dir: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if refs:
        for r in refs:
            p = Path(r)
            if p.suffix.lower() in IMG_EXTS and p.exists():
                paths.append(p)
    if refs_dir:
        base = Path(refs_dir)
        if base.exists():
            for p in base.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    paths.append(p)
    # unique, keep order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def embed_refs(app, ref_paths: List[Path], ref_pad: float, ref_gamma: float, ref_clahe: bool, ref_det_thresh: float,
               pb_kwargs: dict) -> RefPack:
    if not ref_paths:
        return RefPack(embs=None)

    # Configure detector threshold if available
    try:
        if hasattr(app.models["detection"], "threshold"):
            app.models["detection"].threshold = float(ref_det_thresh)
    except Exception:
        pass

    fails = 0
    bar = ProgressBar("REFS".ljust(10), total=len(ref_paths), **pb_kwargs)
    embs = []
    for i, rp in enumerate(ref_paths, 1):
        im = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if im is None:
            fails += 1
            bar.update(i, fails=fails)  # single-line; no fps for REFS
            continue
        # light preprocessing for refs (pad + gamma + optional CLAHE)
        if ref_gamma and abs(ref_gamma - 1.0) > 1e-3:
            tmp = np.power(np.clip(im.astype(np.float32) / 255.0, 0, 1), 1.0 / float(ref_gamma))
            im = np.clip(tmp * 255.0, 0, 255).astype(np.uint8)
        if ref_clahe:
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l,a,b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2,a,b])
            im = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # simple pad around
        if ref_pad > 1e-6:
            h,w = im.shape[:2]
            p = int(max(h,w) * float(ref_pad))
            im = cv2.copyMakeBorder(im, p,p,p,p, borderType=cv2.BORDER_REFLECT)

        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        with suppress_stdout_stderr():
            faces = app.get(rgb)
        if not faces:
            fails += 1
            bar.update(i, fails=fails)
            continue
        # pick largest
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = getattr(f, "embedding", None)
            if emb is not None:
                # L2-normalize
                n = np.linalg.norm(emb) + 1e-9
                emb = emb / n
        if emb is None:
            fails += 1
            bar.update(i, fails=fails)
            continue
        embs.append(emb.astype(np.float32))
        bar.update(i, fails=fails)
    bar.close()

    if not embs:
        return RefPack(embs=None)
    return RefPack(embs=np.stack(embs, axis=0))

# --- crop helpers ---

def bbox_to_square(bbox: np.ndarray, pad: float, img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = bbox.astype(np.float32)
    cx = (x1+x2)/2.0
    cy = (y1+y2)/2.0
    side = max(x2-x1, y2-y1)
    side = side * (1.0 + float(pad))
    s2 = side/2.0
    xs = int(round(cx - s2))
    ys = int(round(cy - s2))
    xe = int(round(cx + s2))
    ye = int(round(cy + s2))
    return xs,ys,xe,ye


def shift_inside(xs,ys,xe,ye, W,H) -> Tuple[int,int,int,int]:
    dx1 = max(0 - xs, 0)
    dy1 = max(0 - ys, 0)
    dx2 = max(xe - W, 0)
    dy2 = max(ye - H, 0)
    xs += dx1 - dx2
    xe += dx1 - dx2
    ys += dy1 - dy2
    ye += dy1 - dy2
    return xs,ys,xe,ye


def crop_with_policy(img: np.ndarray, rect: Tuple[int,int,int,int], policy: str,
                     fill: str, max_pad_frac: float) -> Optional[np.ndarray]:
    H,W = img.shape[:2]
    xs,ys,xe,ye = rect
    pad_left  = max(0, 0 - xs)
    pad_top   = max(0, 0 - ys)
    pad_right = max(0, xe - W)
    pad_bot   = max(0, ye - H)
    need_pad = any(v>0 for v in (pad_left,pad_top,pad_right,pad_bot))

    if not need_pad:
        return img[ys:ye, xs:xe].copy()

    side = max(ye-ys, xe-xs)
    max_pad = side * float(max_pad_frac)

    if policy == "shift":
        xs,ys,xe,ye = shift_inside(xs,ys,xe,ye, W,H)
        # re-eval
        if xs<0 or ys<0 or xe>W or ye>H:
            return None
        return img[ys:ye, xs:xe].copy()

    if policy == "skip":
        return None

    # pad
    if policy == "pad":
        if any(v > max_pad for v in (pad_left,pad_top,pad_right,pad_bot)):
            return None
        border_map = {
            "reflect": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "black": cv2.BORDER_CONSTANT,
        }
        borderType = border_map.get(fill, cv2.BORDER_REFLECT_101)
        xs_c = max(xs, 0); ys_c = max(ys, 0); xe_c = min(xe, W); ye_c = min(ye, H)
        crop = img[ys_c:ye_c, xs_c:xe_c].copy()
        top = ys_c - ys
        left = xs_c - xs
        bottom = ye - ye_c
        right = xe - xe_c
        if borderType == cv2.BORDER_CONSTANT:
            color = (0,0,0)
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType, value=color)
        else:
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType)
        return crop

    return None

# --- quality metrics ---

def mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())

def tenengrad_blur(gray: np.ndarray) -> float:
    g = cv2.Laplacian(gray, cv2.CV_32F)
    return float(g.var())

# --- cosine similarity ---

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a: (D,), b: (R,D) or (D,)
    if b.ndim == 1:
        denom = (np.linalg.norm(a)+1e-9) * (np.linalg.norm(b)+1e-9)
        return float(np.dot(a,b) / denom)
    else:
        denom = (np.linalg.norm(a)+1e-9) * (np.linalg.norm(b, axis=1)+1e-9)
        return float(np.max(np.dot(b, a) / denom))

# --- auto-help writer ---
HELP_TEXT = r"""
face_extract.py — per-frame face extractor (stateless)

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
  --help on|off (on)                    write /workspace/scripts/face_extract.py.txt then proceed

LOG OUTPUT
  <logs>/video/video_stats.csv   (append mode; header is written if the file is new)

  Location:
    • If --log_dir is set → <log_dir>/video/video_stats.csv
    • Otherwise           → parent(--out)/logs/video/video_stats.csv
      Example: --out /workspace/data_src/aligned → /workspace/data_src/logs/video/video_stats.csv

  One row per SAVED crop; columns (exact order):
    file        basename of the saved image (e.g., 000123.png)
    frame_idx   source video frame index (0-based) at which it was sampled
    sim         best cosine similarity vs reference embeddings; empty if refs unused or --sim<=0
    orig_side   native square crop side BEFORE resizing (max of H/W of the pre-resized crop)
    up          upscale factor applied to reach --size: up = size / orig_side
    face_frac   (face side) / (crop side) BEFORE resize; computed as side_det / side_use (with pad)
    video_id    basename of the input video (no extension)

NOTES
  • Progress uses progress.py; REFS appears first (FAIL shows rejected refs), then EXTRACT (FACE shows saved count).
  • Detector/provider console noise is suppressed where possible.
  • If no refs are provided or --sim<=0, the similarity gate is disabled gracefully.
"""


def write_helper_txt(enabled: bool):
    if not enabled:
        return
    try:
        path = Path("/workspace/scripts/face_extract.py.txt")
        path.write_text(HELP_TEXT)
    except Exception:
        pass

# --- main processing ---

def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", nargs="*")
    ap.add_argument("--refs_dir")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--sim", type=float, default=0.55)
    ap.add_argument("--max_per_frame", type=int, default=1)

    # detection/runtime
    ap.add_argument("--det_size", type=int, default=960)
    ap.add_argument("--det_thresh", type=float, default=0.30)
    ap.add_argument("--low_cpu", type=onoff, default=False)
    ap.add_argument("--ep", type=str, default="trt,cuda,cpu")

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
    ap.add_argument("--min_bbox_frac", type=float, default=0.0)
    ap.add_argument("--min_tenengrad", type=float, default=0.0)
    ap.add_argument("--dark_relax", type=float, default=1.0)
    ap.add_argument("--min_brightness", type=float, default=0.0)

    # refs preprocess
    ap.add_argument("--ref_pad", type=float, default=0.60)
    ap.add_argument("--ref_gamma", type=float, default=1.35)
    ap.add_argument("--ref_clahe", type=onoff, default=True)
    ap.add_argument("--ref_det_thresh", type=float, default=0.30)

    # yaw / multi-face
    ap.add_argument("--yaw_shift", type=onoff, default=False)
    ap.add_argument("--yaw_shift_frac", type=float, default=0.12)
    ap.add_argument("--yaw_shift_yawmax", type=float, default=45.0)
    ap.add_argument("--solo_mode", choices=["off","reject","shrink"], default="reject")
    ap.add_argument("--solo_min_other_frac", type=float, default=0.25)
    ap.add_argument("--solo_shrink_pad", type=float, default=0.18)

    # logging & helper
    ap.add_argument("--print_every", type=int, default=1000)
    ap.add_argument("--debug_every", type=int, default=0)
    ap.add_argument("--log_dir")
    ap.add_argument("--help", dest="help_flag", type=onoff, default=True)

    args = ap.parse_args()
    write_helper_txt(bool(args.help_flag))
    return args


def ensure_log_path(out_dir: Path, log_dir_opt: Optional[str]) -> Path:
    if log_dir_opt:
        base = Path(log_dir_opt)
    else:
        base = out_dir.parent / "logs" / "video"
    base.mkdir(parents=True, exist_ok=True)
    return base / "video_stats.csv"


def save_log_row(csv_path: Path, header_written: set, row: List[str]):
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["file","frame_idx","sim","orig_side","up","face_frac","video_id"])
        w.writerow(row)

# --- main extraction loop ---

def main():
    args = parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ensure_log_path(out_dir, args.log_dir)

    # Load FaceAnalysis app
    app = load_app(args.ep, args.det_size, args.low_cpu)
    # adjust runtime detector threshold if possible
    try:
        if hasattr(app.models["detection"], "threshold"):
            app.models["detection"].threshold = float(args.det_thresh)
    except Exception:
        pass

    # --- embed refs (with progress) ---
    ref_paths = list_ref_images(args.refs, args.refs_dir)
    pb_kwargs = {"bar_width": 26, "label_width": 10}
    refs = embed_refs(app, ref_paths, args.ref_pad, args.ref_gamma, bool(args.ref_clahe), args.ref_det_thresh, pb_kwargs)

    # --- open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] cannot open video: {args.video}")
        return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round((src_fps / max(args.fps, 1e-6))))) if src_fps > 0 else 1

    # progress for extraction
    bar = ProgressBar("EXTRACT".ljust(10), total=total_frames if total_frames>0 else 1, bar_width=26, label_width=10)

    saved = 0
    video_id = Path(args.video).stem
    frame_idx = 0
    next_take = 0
    t0 = time.time()

    def choose_anchor(face):
        # anchor modes; default bbox center
        x1,y1,x2,y2 = face.bbox.astype(np.float32)
        cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
        if args.anchor in ("eyes","nose"):
            kps = getattr(face, "kps", None)
            if isinstance(kps, np.ndarray) and kps.shape[0] >= 5:
                if args.anchor == "eyes":
                    cx = float((kps[0,0] + kps[1,0]) * 0.5)
                    cy = float((kps[0,1] + kps[1,1]) * 0.5)
                elif args.anchor == "nose":
                    cx = float(kps[2,0]); cy = float(kps[2,1])
        cy = cy - float(args.anchor_offset_y) * max(x2-x1, y2-y1)
        side = max(x2-x1, y2-y1)
        return cx, cy, side

    # iterate frames
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx < next_take:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            frame_idx += 1
            next_take += step
            continue

        # min bbox area guard (pre-detection cheap gate skipped; detector will handle)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with suppress_stdout_stderr():
            faces = app.get(rgb)
        H,W = frame.shape[:2]

        if not faces:
            frame_idx += 1
            next_take += step
            # update progress
            fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
            bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
            continue

        # Optionally reject multiple-face frames (solo_mode)
        if args.solo_mode in ("reject","shrink") and len(faces) > 1:
            # compute largest face area fraction of frame
            areas = [ (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces ]
            main_idx = int(np.argmax(areas))
            other_max_frac = 0.0
            for j,f in enumerate(faces):
                if j==main_idx: continue
                frac = areas[j] / float(W*H + 1e-9)
                other_max_frac = max(other_max_frac, frac)
            if args.solo_mode == "reject" and other_max_frac >= float(args.solo_min_other_frac):
                frame_idx += 1
                next_take += step
                fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
                bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
                continue
            # shrink: tighten pad if others dominate
            if args.solo_mode == "shrink" and other_max_frac >= float(args.solo_min_other_frac):
                shrink_pad = max(0.0, float(args.bbox_pad) - float(args.solo_shrink_pad))
            else:
                shrink_pad = float(args.bbox_pad)
        else:
            shrink_pad = float(args.bbox_pad)

        # select best face: by similarity if refs exist+sim>0 else largest bbox
        chosen = None
        best_sim = None
        if refs.embs is not None and args.sim > 0:
            sims = []
            for f in faces:
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)
                    if emb is not None:
                        emb = emb / (np.linalg.norm(emb)+1e-9)
                if emb is None:
                    sims.append(-1.0)
                    continue
                sims.append(cos_sim(emb.astype(np.float32), refs.embs))
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx]) if math.isfinite(sims[best_idx]) else -1.0
            if best_sim < float(args.sim):
                # reject this frame
                frame_idx += 1
                next_take += step
                fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
                bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
                continue
            chosen = faces[best_idx]
        else:
            # largest bbox area
            chosen = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            best_sim = None

        # anchor + square crop with pad
        cx,cy,side0 = choose_anchor(chosen)
        # yaw-adaptive shift (if enabled and yaw available)
        if args.yaw_shift:
            yaw = getattr(chosen, "yaw", None)
            if yaw is None:
                yaw = None  # gracefully skip if unavailable
            if yaw is not None and math.isfinite(yaw):
                # clamp and convert to shift in pixels along x
                yawmax = max(1.0, float(args.yaw_shift_yawmax))
                frac = max(-1.0, min(1.0, float(yaw) / yawmax))
                cx = cx + frac * float(args.yaw_shift_frac) * side0
        rect = bbox_to_square(np.array([cx - side0/2, cy - side0/2, cx + side0/2, cy + side0/2], dtype=np.float32),
                              pad=shrink_pad, img_w=W, img_h=H)
        crop = crop_with_policy(frame, rect, args.edge_policy, args.edge_fill, args.max_pad_frac)
        if crop is None:
            frame_idx += 1
            next_take += step
            fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
            bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
            continue

        # quality gates
        ch, cw = crop.shape[:2]
        orig_side = float(max(ch, cw))
        if orig_side < float(args.min_crop_px):
            frame_idx += 1
            next_take += step
            fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
            bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
            continue
        up = float(args.size) / max(orig_side, 1.0)
        if up > float(args.max_upscale):
            frame_idx += 1
            next_take += step
            fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
            bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
            continue

        # face fraction in crop (bbox side / crop side)
        fx1,fy1,fx2,fy2 = chosen.bbox.astype(np.float32)
        face_side = max(fx2-fx1, fy2-fy1)
        face_frac = float(face_side / max(max(rect[2]-rect[0], rect[3]-rect[1]), 1.0))
        if face_frac < float(args.min_face_frac_crop):
            frame_idx += 1
            next_take += step
            fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
            bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
            continue

        # brightness + blur gates
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bright = mean_brightness(gray)
        blur_v = tenengrad_blur(gray)
        th_b = float(args.min_brightness)
        th_t = float(args.min_tenengrad)
        if bright < th_b:
            th_t = th_t * float(args.dark_relax)
        if blur_v < th_t:
            frame_idx += 1
            next_take += step
            fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
            bar.update(min(frame_idx, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")
            continue

        # final resize to square size
        crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)

        # save
        fname = f"{saved+1:06d}.png"
        cv2.imwrite(str(out_dir / fname), crop)

        # log row
        row = [
            fname,
            str(frame_idx),
            (f"{best_sim:.6f}" if (best_sim is not None and math.isfinite(best_sim)) else ""),
            f"{orig_side:.3f}",
            f"{up:.6f}",
            f"{face_frac:.6f}",
            video_id,
        ]
        save_log_row(csv_path, set(), row)

        saved += 1
        # update progress bar (FACE counter)
        fps = (saved / max(time.time()-t0, 1e-6)) if saved>0 else 0.0
        bar.update(min(frame_idx+1, bar.total), fails=None, fps=fps, extra=f"FACE {saved}")

        # next
        frame_idx += 1
        next_take += step

    bar.close()
    cap.release()
    print(f"[OK] saved {saved} frames → {out_dir}")
    print(f"[OK] stats → {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
