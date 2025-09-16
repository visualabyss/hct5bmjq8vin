#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, csv, json, contextlib, io, time, warnings
from pathlib import Path
import numpy as np
import cv2
from progress import ProgressBar

# Silence third‑party FutureWarnings (e.g., insightface numpy lstsq rcond)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".webp"}

AUTO_HELP_TEXT = r"""
PURPOSE
  • Compute ArcFace embeddings on a reference set to form a normalized centroid, then score each aligned image
    by cosine similarity to that centroid (higher = closer to refs).
  • Two progress rows: REFS (no FPS; shows FAIL=rejects) then ARCFACE (FPS + FAIL=embeds failed).
  • Writes a lean CSV for compatibility and richer artifacts for downstream analysis.

USAGE (common)
  python3 -u /workspace/scripts/arcface.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --ref_dir /workspace/refs \
    --override

CLI FLAGS
  --aligned DIR (req)               aligned images to score
  --logs DIR (req)                  log root (outputs under logs/arcface)
  --ref_dir DIR (req)               directory of reference images (recursive)
  --model any                       kept for interface parity (ignored)
  --batch INT (64)                  kept for parity; embedding is per-image in this runner
  --override (flag)                 rewrite arcface.csv instead of appending

  # reference preprocessing & detection
  --ref_pad FLOAT (0.60)            reflect-pad fraction around refs before detection
  --ref_gamma FLOAT (1.35)          gamma correction for refs (1=no change)
  --ref_clahe on|off (on)           apply CLAHE to L channel for refs
  --ref_det_thresh FLOAT (0.30)     min detector score for refs

  # logging & help
  --help on|off (on)                write /workspace/scripts/arcface.txt then proceed
  -h / --h                          print this help text and continue

OUTPUTS
  logs/arcface/arcface.csv          (append or rewrite if --override) schema: path,id_score
  logs/arcface/embeddings.npy       float32 N×D embeddings (ordered to CSV; D=512)
  logs/arcface/index.csv            row,file,ok (1 if embedded)
  logs/arcface/centroid.npy         float32 D vector (L2-normalized)
  logs/arcface/meta.json            backend, dim, ref_count, fallback flags
"""

def _parse_onoff(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"1","on","true","yes","y"}: return True
    if s in {"0","off","false","no","n"}: return False
    return bool(default)

@contextlib.contextmanager
def _quiet():
    old = (sys.stdout, sys.stderr)
    try:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = buf_out, buf_err
        yield
    finally:
        sys.stdout, sys.stderr = old

# ---------------- utils ----------------

def list_images(d: Path):
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def imread_bgr(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return im


def l2norm(v: np.ndarray, eps=1e-9):
    n = float(np.linalg.norm(v) + eps)
    return (v / n).astype(np.float32)


def get_available_providers():
    try:
        import onnxruntime as ort
        ps = list(ort.get_available_providers())
        order = ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]
        return [p for p in order if p in ps] or ps
    except Exception:
        return ["CPUExecutionProvider"]


def load_face_app():
    os.environ.setdefault('ORT_LOG_SEVERITY_LEVEL','3')
    os.environ.setdefault('INSIGHTFACE_DISABLE_ANALYTICS','1')
    from insightface.app import FaceAnalysis
    providers = get_available_providers()
    with _quiet():
        app = FaceAnalysis(name='buffalo_l', providers=providers)
        try:
            app.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.30,
                        allowed_modules=["detection","recognition","landmark_2d_106"])
        except TypeError:
            app.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.30)
    return app, providers

# ---------- reference preprocess (to avoid fallback to aligned) ----------

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

def _preprocess_ref(im, pad_frac=0.60, gamma=1.35, clahe=1):
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


def best_face_embedding(app, img_bgr, det_thresh=None):
    faces = app.get(img_bgr) or []
    if det_thresh is not None:
        faces = [f for f in faces if float(getattr(f,'det_score',0.0)) >= float(det_thresh)]
    if not faces:
        return None
    faces.sort(key=lambda f: (float(getattr(f,'det_score',0.0)), (float(f.bbox[2]-f.bbox[0])*float(f.bbox[3]-f.bbox[1]))), reverse=True)
    emb = getattr(faces[0], 'normed_embedding', None)
    if emb is None:
        return None
    v = np.asarray(emb, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return None
    return l2norm(v)


def compute_centroid_from_dir(app, ref_dir: Path, args):
    ref_paths = [p for p in sorted(ref_dir.rglob('*')) if p.suffix.lower() in IMAGE_EXTS]
    if not ref_paths:
        return None, 0
    bar_refs = ProgressBar("REFS", total=len(ref_paths), show_fail_label=True, show_fps=False, fail_label="FAIL")
    embs = []; fails = 0
    for i,p in enumerate(ref_paths,1):
        try:
            im = imread_bgr(p)
            im = _preprocess_ref(im, pad_frac=args.ref_pad, gamma=args.ref_gamma, clahe=1 if _parse_onoff(args.ref_clahe, True) else 0)
            v = best_face_embedding(app, im, det_thresh=args.ref_det_thresh)
            if v is None:
                fails += 1
            else:
                embs.append(v)
        except Exception:
            fails += 1
        bar_refs.update(i, fails=fails)
    bar_refs.close()
    if not embs:
        return None, 0
    C = l2norm(np.mean(np.stack(embs,0), axis=0))
    return C, len(embs)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser("ArcFace scorer (InsightFace) -> logs/arcface/arcface.csv", add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    ap.add_argument('--aligned', required=True, type=Path)
    ap.add_argument('--logs', required=True, type=Path)
    ap.add_argument('--ref_dir', required=True, type=Path)
    ap.add_argument('--model', default='any')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--override', action='store_true')
    ap.add_argument('--ref_pad', type=float, default=0.60)
    ap.add_argument('--ref_gamma', type=float, default=1.35)
    ap.add_argument('--ref_clahe', default='on')
    ap.add_argument('--ref_det_thresh', type=float, default=0.30)
    ap.add_argument('--help', dest='auto_help', default='on')
    args = ap.parse_args()

    if getattr(args, 'show_cli_help', False):
        print(AUTO_HELP_TEXT)
    try:
        if _parse_onoff(args.auto_help, True):
            sp = Path(__file__).resolve(); (sp.with_name(sp.stem + '.txt')).write_text(AUTO_HELP_TEXT, encoding='utf-8')
    except Exception:
        pass

    imgs = list_images(args.aligned)
    out_dir = args.logs / 'arcface'; out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'arcface.csv'
    emb_path = out_dir / 'embeddings.npy'
    idx_path = out_dir / 'index.csv'
    cen_path = out_dir / 'centroid.npy'
    meta_path = out_dir / 'meta.json'

    if not imgs:
        print('ARCFACE: no images under --aligned.')
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=['path','id_score']).writeheader()
        return 2

    app, providers = load_face_app()

    # centroid from refs; fallback to aligned if refs yield none
    centroid, ref_used = compute_centroid_from_dir(app, args.ref_dir, args)
    fallback_to_aligned = False
    if centroid is None:
        fallback_to_aligned = True
        bar_fb = ProgressBar("REFS", total=len(imgs), show_fail_label=True, show_fps=False, fail_label="FAIL")
        embs = []; fails=0
        for i,p in enumerate(imgs,1):
            try:
                im = imread_bgr(p)
                v = best_face_embedding(app, im)
                if v is None:
                    fails+=1
                else:
                    embs.append(v)
            except Exception:
                fails+=1
            bar_fb.update(i, fails=fails)
        bar_fb.close()
        if not embs:
            print('ARCFACE: no embeddings available (refs and aligned failed).')
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=['path','id_score']).writeheader()
            return 2
        centroid = l2norm(np.mean(np.stack(embs,0), axis=0))
        ref_used = len(embs)

    # main pass: embed aligned, score, and dump artifacts
    headers=['path','id_score']
    if args.override or not csv_path.exists():
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()
    N=len(imgs); D=512
    E = np.zeros((N,D), dtype=np.float32)
    OK = np.zeros((N,), dtype=np.uint8)

    bar_arc = ProgressBar('ARCFACE', total=N, show_fail_label=True, show_fps=True, fail_label='FAIL')
    fails=0
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        for i,p in enumerate(imgs,1):
            score_str=''
            try:
                im = imread_bgr(p)
                v = best_face_embedding(app, im)
                if v is not None:
                    OK[i-1]=1; E[i-1]=v
                    score = float(np.dot(v, centroid))
                    score_str = f'{score:.6f}'
                else:
                    fails += 1
            except Exception:
                fails += 1
            w.writerow({'path': p.name, 'id_score': score_str})
            bar_arc.update(i, fails=fails)
    bar_arc.close()

    # save artifacts
    try:
        np.save(emb_path, E)
        np.save(cen_path, centroid.astype(np.float32))
        with idx_path.open('w', newline='', encoding='utf-8') as f:
            iw = csv.writer(f); iw.writerow(['row','file','ok'])
            for i,p in enumerate(imgs):
                iw.writerow([i, p.name, int(OK[i])])
        meta = {
            'backend': 'insightface.app.FaceAnalysis',
            'model': 'buffalo_l',
            'providers': providers,
            'dim': int(D),
            'ref_count': int(ref_used),
            'fallback_centroid_from_aligned': bool(fallback_to_aligned),
            'aligned_count': int(N),
            'embeddings_path': str(emb_path),
            'centroid_path': str(cen_path),
            'index_path': str(idx_path),
            'csv_path': str(csv_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    except Exception:
        pass

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
