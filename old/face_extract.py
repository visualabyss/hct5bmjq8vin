#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
face_extract.py (DFL)
- Uses progress.py only (no embedded bars here)
- Two rows with dividers:
    REFS    (single snapshot, shows actual failed refs; no FPS)
    EXTRACT (live, counter label FACE = #saved)
- Silences noisy provider/model prints
- Tenengrad = variance of Laplacian
- Writes logs/video/video_stats.csv (append)
- --help on|off writes /workspace/scripts/face_extract.py.txt

Extraction logic mirrors the original extract_face_directly.py semantics:
  detect -> choose face (ref similarity if provided else area/center/det_score) ->
  optional yaw shift -> square crop with pad policy (skip/pad/shift) -> quality gates -> save
"""

import os, sys, csv, time, math, warnings, logging, contextlib, io
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
os.environ.setdefault("INSIGHTFACE_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
from progress import ProgressBar

# ----------------------------------------------------------------------------
# Helper / IO
# ----------------------------------------------------------------------------

HELP_TXT = """\
face_extract.py — video → aligned face crops

LOG
  <logs>/video/video_stats.csv  (append; header if new)
  Columns: file,frame_idx,sim,orig_side,up,face_frac,video_id

FLAGS (selected)
  --video, --out, --size 512, --fps 8, --sim 0.55, --max_per_frame 1
  --det_size 960, --det_thresh 0.30, --low_cpu on|off, --ep trt,cuda,cpu, --mscale 1.0
  --anchor bbox, --anchor_offset_y 0.10, --bbox_pad 0.34
  --edge_policy shift|pad|skip, --edge_fill reflect|replicate|black, --max_pad_frac 0.18
  --yaw_shift on|off, --yaw_shift_frac 0.12, --yaw_shift_yawmax 45
  --solo_mode off|reject|shrink, --solo_min_other_frac 0.25
  --min_crop_px 512, --max_upscale 1.5, --min_face_frac_crop 0.30
  --min_bbox_frac 0.0, --min_tenengrad 0.0, --dark_relax 1.0, --min_brightness 0.0
  --ref_pad 0.60, --ref_gamma 1.35, --ref_clahe on|off, --ref_det_thresh 0.30
  --print_every 1000, --debug_every 0, --log_dir DIR, --help on|off
"""

def write_helper_if_needed(flag_onoff="on"):
    if str(flag_onoff).strip().lower() not in ("on","true","yes"): return
    try: Path("/workspace/scripts/face_extract.py.txt").write_text(HELP_TXT)
    except Exception: pass


def build_logs_path(out_dir, log_dir):
    out_dir = Path(out_dir)
    root = Path(log_dir) if log_dir else (out_dir.parent / "logs")
    path = root / "video" / "video_stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_header_if_new(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["file","frame_idx","sim","orig_side","up","face_frac","video_id"])


def _append_row(csv_path, row):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


@contextlib.contextmanager
def _suppress_io():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield


# ----------------------------------------------------------------------------
# Reference handling
# ----------------------------------------------------------------------------

def _collect_ref_paths(args):
    paths = []
    if args.refs:
        paths += [Path(x) for x in args.refs if x]
    if args.refs_dir:
        for p in Path(args.refs_dir).rglob("*"):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
                paths.append(p)
    return [p for p in paths if p.exists()]


def _preprocess_ref(im, pad_frac=0.0, gamma=None, clahe=True):
    if im is None: return None
    out = im
    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    if gamma is not None:
        g = float(gamma)
        lut = np.clip(((np.arange(256)/255.0)**(1.0/g))*255.0 + 0.5, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    if pad_frac and pad_frac > 0:
        H, W = out.shape[:2]
        pad = int(round(float(pad_frac) * min(H, W)))
        if pad > 0:
            out = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    return out


def _collect_refs_embeddings(paths, app, det_thresh=0.30, ref_pad=0.60, ref_gamma=1.35, ref_clahe=True):
    embs, failures = [], 0
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        im = _preprocess_ref(im, pad_frac=ref_pad, gamma=ref_gamma, clahe=ref_clahe)
        if im is None:
            failures += 1; continue
        with _suppress_io():
            faces = app.get(im)
        if not faces:
            failures += 1; continue
        faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= float(det_thresh)]
        if not faces:
            failures += 1; continue
        f = max(faces, key=lambda x: float(getattr(x, "det_score", 0.0)))
        v = getattr(f, "normed_embedding", None)
        if v is None:
            failures += 1; continue
        embs.append(np.asarray(v, dtype=np.float32))
    if not embs:
        return None, failures
    return np.stack(embs, axis=0).astype(np.float32), failures


def _cos_sim(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


# ----------------------------------------------------------------------------
# Crop / quality gates (matching original semantics)
# ----------------------------------------------------------------------------

def _crop_from_bbox(im, bbox, pad_frac=0.34, out_size=512,
                    edge_policy="skip", edge_fill="reflect", max_pad_frac=0.18,
                    cx_shift_frac=0.0):
    h, w = im.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    side = float(max(bw, bh)) * (1.0 + pad_frac)
    cx = (x1 + x2) * 0.5 + cx_shift_frac * side
    cy = (y1 + y2) * 0.5

    x1s = int(round(cx - side/2)); y1s = int(round(cy - side/2))
    x2s = int(round(cx + side/2)); y2s = int(round(cy + side/2))

    if x1s < 0 or y1s < 0 or x2s > w or y2s > h:
        if edge_policy == "skip":
            return None, {"orig_side": int(round(side)), "face_frac": bw/max(1.0, side)}
        # pad or shift
        if edge_policy == "shift":
            # shift into frame bounds
            dx = 0; dy = 0
            if x1s < 0: dx = -x1s
            if x2s > w: dx = min(dx, w - x2s)
            if y1s < 0: dy = -y1s
            if y2s > h: dy = min(dy, h - y2s)
            x1s += dx; x2s += dx; y1s += dy; y2s += dy
        # after shift, if still OOB -> pad but clamp to max_pad_frac
        pad_left  = max(0, -x1s); pad_top   = max(0, -y1s)
        pad_right = max(0, x2s - w); pad_bot = max(0, y2s - h)
        if pad_left or pad_top or pad_right or pad_bot:
            max_pad = int(round(max_pad_frac * side))
            if max(pad_left, pad_top, pad_right, pad_bot) > max_pad:
                return None, {"orig_side": int(round(side)), "face_frac": bw/max(1.0, side)}
            borderType = {"reflect": cv2.BORDER_REFLECT_101,
                          "replicate": cv2.BORDER_REPLICATE,
                          "black": cv2.BORDER_CONSTANT}.get(edge_fill, cv2.BORDER_REFLECT_101)
            im = cv2.copyMakeBorder(im, pad_top, pad_bot, pad_left, pad_right, borderType=borderType)
            x1s += pad_left; x2s += pad_left; y1s += pad_top; y2s += pad_top
            h, w = im.shape[:2]

    crop = im[max(0,y1s):min(h,y2s), max(0,x1s):min(w,x2s)].copy()
    if crop.size == 0:
        return None, {"orig_side": int(round(side)), "face_frac": bw/max(1.0, side)}
    crop_side = max(crop.shape[0], crop.shape[1])
    up = float(out_size)/float(crop_side)
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop, {"orig_side": int(crop_side), "face_frac": float(bw/max(1.0, side)), "up": float(up)}


def _tenengrad_varlap(crop):
    g = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    return float(np.var(g))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", nargs="*", default=None)
    ap.add_argument("--refs_dir", type=str, default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--sim", type=float, default=0.55)
    ap.add_argument("--max_per_frame", type=int, default=1)

    # detection/runtime
    ap.add_argument("--det_size", type=int, default=960)
    ap.add_argument("--det_thresh", type=float, default=0.30)
    ap.add_argument("--low_cpu", type=str, choices=["on","off"], default="off")
    ap.add_argument("--ep", type=str, default="trt,cuda,cpu")
    ap.add_argument("--mscale", type=float, default=1.0)

    # geometry & crop
    ap.add_argument("--anchor", type=str, choices=["auto","bbox","eyes","nose"], default="bbox")
    ap.add_argument("--anchor_offset_y", type=float, default=0.10)
    ap.add_argument("--bbox_pad", type=float, default=0.34)
    ap.add_argument("--edge_policy", type=str, choices=["shift","pad","skip"], default="skip")
    ap.add_argument("--edge_fill", type=str, choices=["reflect","replicate","black"], default="reflect")
    ap.add_argument("--max_pad_frac", type=float, default=0.18)

    # yaw shift
    ap.add_argument("--yaw_shift", type=str, choices=["on","off"], default="off")
    ap.add_argument("--yaw_shift_frac", type=float, default=0.12)
    ap.add_argument("--yaw_shift_yawmax", type=float, default=45.0)

    # solo handling
    ap.add_argument("--solo_mode", type=str, choices=["off","reject","shrink"], default="reject")
    ap.add_argument("--solo_min_other_frac", type=float, default=0.25)

    # quality
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
    ap.add_argument("--ref_clahe", type=str, choices=["on","off"], default="on")
    ap.add_argument("--ref_det_thresh", type=float, default=0.30)

    # logging
    ap.add_argument("--print_every", type=int, default=1000)
    ap.add_argument("--debug_every", type=int, default=0)
    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--help", type=str, choices=["on","off"], default="on")

    args = ap.parse_args()
    write_helper_if_needed(args.help)

    # logs
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = build_logs_path(out_dir, args.log_dir); _append_header_if_new(csv_path)

    # low-CPU threading
    if str(args.low_cpu).lower()=="on":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    # InsightFace init (silenced)
    from insightface.app import FaceAnalysis
    providers = []
    for w in [x.strip().upper() for x in args.ep.split(",") if x.strip()]:
        if w == "TRT": providers.append("TensorrtExecutionProvider")
        elif w == "CUDA": providers.append("CUDAExecutionProvider")
        elif w == "CPU": providers.append("CPUExecutionProvider")
    if not providers: providers = ["CPUExecutionProvider"]

    with _suppress_io():
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if any(p.startswith(("CUDA","Tensorrt")) for p in providers) else -1,
                    det_size=(args.det_size, args.det_size))

    # ---------------- REFS row (single) ----------------
    DIV = "="*103
    print(DIV, flush=True)
    ref_paths = _collect_ref_paths(args)
    with _suppress_io():
        ref_mat, ref_fail = _collect_refs_embeddings(
            ref_paths, app,
            det_thresh=args.ref_det_thresh,
            ref_pad=args.ref_pad,
            ref_gamma=args.ref_gamma,
            ref_clahe=(args.ref_clahe=="on")
        )
    total_refs = len(ref_paths)
    if ref_mat is not None:
        ref_fail = max(0, min(total_refs, ref_fail))
    else:
        ref_fail = total_refs if total_refs>0 else 0

    pb_refs = ProgressBar("REFS", total=max(1,total_refs), show_fail_label=True, fail_label="FAIL", show_fps=False)
    pb_refs.update(total_refs, fails=ref_fail, fps=0.0)
    pb_refs.close()
    print(DIV, flush=True)

    # ---------------- Video open & sampling ----------------
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERR] cannot open video: {args.video}", flush=True); return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if src_fps <= 0:
        stride = max(1, int(round(30.0/args.fps)))
    else:
        stride = max(1, int(round(src_fps / max(0.1, args.fps))))
    total_samples = (n_frames // stride) if n_frames>0 else 0

    pb = ProgressBar("EXTRACT", total=max(1, total_samples or 1), show_fail_label=True, fail_label="FACE", show_fps=True)
    saved = 0
    sample_idx = 0
    next_index = len(list(out_dir.glob("*.png"))) + 1
    vid_id = Path(args.video).stem

    def _should_keep_by_quality(crop, meta, bbox, W, H):
        orig_side = int(round(meta.get("orig_side", 0)))
        if orig_side < args.min_crop_px: return False
        up = meta.get("up", float(args.size)/max(1, orig_side))
        if up > args.max_upscale: return False
        face_frac = float(meta.get("face_frac", 0.0))
        if face_frac < args.min_face_frac_crop: return False
        if args.min_bbox_frac > 0.0:
            x1,y1,x2,y2 = bbox
            if ((x2-x1)*(y2-y1))/(W*H+1e-9) < args.min_bbox_frac: return False
        if args.min_tenengrad > 0.0:
            tgv = _tenengrad_varlap(crop)
            mg  = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
            thr = float(args.min_tenengrad) * (float(args.dark_relax) if mg < 60.0 else 1.0)
            if tgv < thr: return False
        if args.min_brightness > 0.0:
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if float(g.mean()) < args.min_brightness: return False
        return True

    def _detect(img):
        with _suppress_io():
            return app.get(img)

    # ---------- frame loop ----------
    frame_idx = -1
    while True:
        ok = cap.grab()
        if not ok: break
        frame_idx += 1
        if (frame_idx % stride) != 0: continue
        ok, frame = cap.retrieve()
        sample_idx += 1
        if not ok or frame is None:
            pb.update(sample_idx, fails=saved); continue

        H, W = frame.shape[:2]
        faces = _detect(frame)

        # optional upscaled retry when no faces
        if (not faces) and args.mscale and args.mscale > 1.0:
            scale = float(args.mscale)
            big = cv2.resize(frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_LINEAR)
            faces_big = _detect(big)
            if faces_big:
                faces = []
                for fb in faces_big:
                    f = type("MiniFace", (), {})()
                    b = getattr(fb, "bbox", None)
                    if b is None: continue
                    f.bbox = b.astype("float32") / scale
                    kps = getattr(fb, "kps", None)
                    f.kps = (kps.astype("float32") / scale) if kps is not None else None
                    f.embedding = getattr(fb, "embedding", None)
                    f.normed_embedding = getattr(fb, "normed_embedding", None)
                    f.det_score = getattr(fb, "det_score", 0.0)
                    faces.append(f)

        if not faces:
            pb.update(sample_idx, fails=saved); continue

        # solo crowd reject (matches original guard)
        if args.solo_mode == "reject" and len(faces) > 1:
            def _area(f):
                b = getattr(f, "bbox", None)
                if b is None: return 0.0
                b = b.astype(float); return (b[2]-b[0])*(b[3]-b[1])
            areas = sorted((_area(f) for f in faces), reverse=True)
            a1 = areas[0]; a2 = areas[1] if len(areas) > 1 else 0.0
            if a1 > 0 and (a2 / a1) >= float(args.solo_min_other_frac):
                pb.update(sample_idx, fails=saved); continue

        # choose face (ref similarity if provided)
        if (args.sim > 0) and ('ref_mat' in locals()) and (ref_mat is not None):
            embs, idxs = [], []
            for i,f in enumerate(faces):
                e = getattr(f, "normed_embedding", None)
                if e is None: continue
                embs.append(np.asarray(e, dtype=np.float32)); idxs.append(i)
            if not embs:
                pb.update(sample_idx, fails=saved); continue
            E = np.stack(embs, axis=0).astype(np.float32)
            S = _cos_sim(E, ref_mat)
            j = int(np.argmax(S.max(axis=1))); best_sim = float(S[j].max())
            if best_sim < args.sim:
                pb.update(sample_idx, fails=saved); continue
            face = faces[idxs[j]]; sim_val = best_sim
        else:
            def _score(f):
                x1,y1,x2,y2 = map(float, f.bbox)
                w = max(1.0, x2-x1); h = max(1.0, y2-y1)
                area = (w*h) / max(1.0, W*H)
                cx = (x1+x2)*0.5; cy = (y1+y2)*0.5
                dx = abs(cx - W/2)/(W/2); dy = abs(cy - H/2)/(H/2)
                center = 1.0 - min(1.0, 0.5*(dx+dy))
                ds = float(getattr(f, "det_score", 0.0))
                return 0.60*area + 0.25*center + 0.15*ds
            face = max(faces, key=_score); sim_val = ""

        # yaw shift (heuristic by eye-nose asymmetry)
        yaw_shift_frac = 0.0
        if str(args.yaw_shift).lower()=="on":
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 3:
                le, re, nose = kps[0], kps[1], kps[2]
                d_r = float(np.linalg.norm(nose - re)) + 1e-6
                d_l = float(np.linalg.norm(nose - le)) + 1e-6
                yaw_deg = float(25.0 * math.log(d_r / d_l))
                yawmax = max(1e-3, float(args.yaw_shift_yawmax))
                yaw_shift_frac = float(args.yaw_shift_frac) * max(-1.0, min(1.0, yaw_deg / yawmax))

        # crop (square around bbox with pad)
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            pb.update(sample_idx, fails=saved); continue
        crop, meta = _crop_from_bbox(
            frame, bbox, pad_frac=args.bbox_pad, out_size=args.size,
            edge_policy=args.edge_policy, edge_fill=args.edge_fill,
            max_pad_frac=args.max_pad_frac, cx_shift_frac=yaw_shift_frac
        )
        if crop is None:
            pb.update(sample_idx, fails=saved); continue

        # gates
        if not _should_keep_by_quality(crop, meta, bbox, W, H):
            pb.update(sample_idx, fails=saved); continue

        # save
        fname = f"{next_index:06d}.png"; next_index += 1
        cv2.imwrite(str(out_dir / fname), crop)
        saved += 1

        # log row
        orig_side = int(round(meta.get("orig_side", args.size)))
        up = float(meta.get("up", float(args.size)/max(1, orig_side)))
        face_frac = float(meta.get("face_frac", 0.0))
        _append_row(csv_path, [fname, frame_idx, f"{sim_val:.6f}" if sim_val!="" else "", orig_side, f"{up:.4f}", f"{face_frac:.4f}", vid_id])

        pb.update(sample_idx, fails=saved)

    pb.close(); cap.release()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
