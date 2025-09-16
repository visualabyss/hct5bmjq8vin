cat > /workspace/scripts/arcface.py <<'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArcFace scoring over aligned images with InsightFace FaceAnalysis.

Additions (backward-compatible):
- Auto-help: writes /workspace/scripts/arcface.txt when --help on|off (default on). -h/--h prints and continues.
- Two progress rows via progress.py: REFS (no FPS; FAIL shows rejected refs) then ARCFACE (FPS + FAIL counter).
- Always extract everything useful (no opt-ins):
  • arcface.csv  — lean, stable: path,id_score (unchanged schema)
  • embeddings.npy — N×D embeddings in image order (D=512)
  • index.csv      — row mapping: row,file,ok (1 if embedded)
  • centroid.npy   — D vector used for scoring (L2-normalized)
  • meta.json      — backend info, dims, ref_count, fallback flags

Core scoring logic remains the same: cosine similarity to the centroid built from refs (or aligned fallback).
"""

import os, sys, argparse, csv, json, math, time, contextlib, io
from pathlib import Path
import numpy as np
import cv2
from progress import ProgressBar

IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".webp"}

# ---------------- auto-help ----------------
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
    --ref_dir /workspace/refs

CLI FLAGS
  --aligned DIR (req)               aligned images to score
  --logs DIR (req)                  log root (outputs under logs/arcface)
  --ref_dir DIR (req)               directory of reference images (recursive)
  --model any                       kept for interface parity (ignored)
  --batch INT (64)                  kept for parity; embedding is per-image in this runner

  # logging & help
  --help on|off (on)                write /workspace/scripts/arcface.txt then proceed
  -h / --h                          print this help text and continue

OUTPUTS
  logs/arcface/arcface.csv          (append mode; header if file is new) schema: path,id_score
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
    xs=[p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return xs


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


def best_face_embedding(app, img_bgr):
    faces = app.get(img_bgr) or []
    if not faces:
        return None
    # choose by det score then bbox area
    faces.sort(key=lambda f: (float(getattr(f,'det_score',0.0)), (float(f.bbox[2]-f.bbox[0])*float(f.bbox[3]-f.bbox[1]))), reverse=True)
    emb = getattr(faces[0], 'normed_embedding', None)
    if emb is None:
        return None
    v = np.asarray(emb, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return None
    return l2norm(v)


def compute_centroid_from_dir(app, ref_dir: Path):
    ref_paths = [p for p in sorted(ref_dir.rglob('*')) if p.suffix.lower() in IMAGE_EXTS]
    if not ref_paths:
        return None, 0
    bar = ProgressBar("REFS", total=len(ref_paths), show_fail_label=True, show_fps=False, fail_label="FAIL")
    embs = []
    fails = 0
    for i,p in enumerate(ref_paths,1):
        try:
            im = imread_bgr(p)
            v = best_face_embedding(app, im)
            if v is None:
                fails += 1
            else:
                embs.append(v)
        except Exception:
            fails += 1
        bar.update(i, fails=fails)
    bar.close()
    if not embs:
        return None, 0
    C = l2norm(np.mean(np.stack(embs,0), axis=0))
    return C, len(embs)


def main():
    ap = argparse.ArgumentParser("ArcFace scorer (InsightFace) -> logs/arcface/arcface.csv", add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    ap.add_argument('--aligned', required=True, type=Path)
    ap.add_argument('--logs', required=True, type=Path)
    ap.add_argument('--ref_dir', required=True, type=Path)
    ap.add_argument('--model', default='any')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--help', dest='auto_help', default='on')
    args = ap.parse_args()

    if getattr(args, 'show_cli_help', False):
        print(AUTO_HELP_TEXT)
    # write auto-help file
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
    centroid, ref_used = compute_centroid_from_dir(app, args.ref_dir)
    fallback_to_aligned = False
    if centroid is None:
        # Build centroid from aligned set as fallback, still respecting progress UX
        fallback_to_aligned = True
        bar = ProgressBar("REFS", total=len(imgs), show_fail_label=True, show_fps=False, fail_label="FAIL")
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
            bar.update(i, fails=fails)
        bar.close()
        if not embs:
            print('ARCFACE: no embeddings available (refs and aligned failed).')
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=['path','id_score']).writeheader()
            return 2
        centroid = l2norm(np.mean(np.stack(embs,0), axis=0))
        ref_used = len(embs)

    # main pass: embed aligned, score, and dump artifacts
    headers=['path','id_score']
    is_new = not csv_path.exists()
    if is_new:
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    N=len(imgs)
    D=512
    E = np.zeros((N,D), dtype=np.float32)
    OK = np.zeros((N,), dtype=np.uint8)

    bar2 = ProgressBar('ARCFACE', total=N, show_fail_label=True, show_fps=True, fail_label='FAIL')
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
            bar2.update(i, fails=fails)
    bar2.close()

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
EOF
chmod +x /workspace/scripts/arcface.py
