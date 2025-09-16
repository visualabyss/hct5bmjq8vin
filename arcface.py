#!/usr/bin/env python3
# --- auto-switch to dedicated ArcFace env (/workspace/envs/af_env) ---
def _ensure_af_env():
    import os, sys
    af_py = "/workspace/envs/af_env/bin/python"
    if sys.executable != af_py and os.path.exists(af_py):
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.execv(af_py, [af_py, os.path.abspath(__file__), *sys.argv[1:]])
_ensure_af_env()

import sys, csv, argparse, time
from pathlib import Path

# --- progress bar import (shared style) ---
try:
    from progress import ProgressBar
except Exception:
    class ProgressBar:
        def __init__(self, *a, **k): self.total = k.get("total", 1)
        def update(self, *a, **k): pass
        def close(self): pass

# --------------- utils ---------------
def _prep_dirs(args):
    aligned = Path(args.aligned)
    ref_dir = Path(args.ref_dir)
    logs = Path(args.logs)
    out_csv = logs / "arcface" / "arcface.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return aligned, ref_dir, out_csv

def _write_csv_header(path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["path","id_score"])

def _providers_ort(ort):
    avail = set(ort.get_available_providers())
    pref = [p for p in ("CUDAExecutionProvider","CPUExecutionProvider") if p in avail]
    return pref or ["CPUExecutionProvider"]

def _list_imgs(dir_path):
    ex = {".jpg",".jpeg",".png",".webp",".bmp"}
    ps = [p for p in Path(dir_path).glob("*") if p.suffix.lower() in ex]
    ps.sort()
    return ps

# -------- ONNX streaming embedder --------
def _embed_stream_onnx(sess, np, cv2, dir_path, batch, iname):
    # infer H,W from model
    H, W = 112, 112
    try:
        shp = sess.get_inputs()[0].shape
        if isinstance(shp, (list,tuple)) and len(shp)==4 and all(isinstance(v,int) for v in shp[2:4]):
            H, W = int(shp[2]), int(shp[3])
    except Exception:
        pass

    paths = _list_imgs(dir_path)
    b = max(1, int(batch))
    for i in range(0, len(paths), b):
        ims, kept = [], []
        for pth in paths[i:i+b]:
            im = cv2.imread(str(pth), cv2.IMREAD_COLOR)
            if im is None: 
                continue
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
            im = (im - 127.5) / 128.0
            im = np.transpose(im, (2,0,1))  # NCHW
            ims.append(im); kept.append(pth.name)
        if not ims:
            yield [], None
            continue
        inp = _np.stack(ims, axis=0).astype("float32")
        out = sess.run(None, {iname: inp})[0]
        out /= (__import__("numpy").linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        yield kept, out.astype("float32")

# -------- FaceAnalysis recognition fallback --------
def _load_face_recognition(prefer_gpu=True):
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        return None
    for name in ("buffalo_l","antelopev2"):
        try:
            app = FaceAnalysis(name=name, root="/workspace/tools/ifzoo")
            try:
                app.prepare(ctx_id=0 if prefer_gpu else -1)
            except Exception:
                app.prepare(ctx_id=-1)
            rec = getattr(app, "models", {}).get("recognition", None)
            if rec is not None:
                return rec  # has get_feat(img_bgr_112)
        except Exception:
            continue
    return None

def _embed_stream_face(recognizer, dir_path, batch):
    import cv2
    import numpy as _np
    paths = _list_imgs(dir_path)
    b = max(1, int(batch))
    for i in range(0, len(paths), b):
        feats, kept = [], []
        for pth in paths[i:i+b]:
            im = cv2.imread(str(pth), cv2.IMREAD_COLOR)
            if im is None: 
                continue
            im = cv2.resize(im, (112,112), interpolation=cv2.INTER_AREA)  # recognizer expects 112×112 BGR
            f = recognizer.get_feat(im)
            if f is None:
                continue
            v = _np.asarray(f, dtype="float32").reshape(-1)
            v /= (float((v**2).sum())**0.5 + 1e-9)
            feats.append(v); kept.append(pth.name)
        if not feats:
            yield [], None
        else:
            yield kept, _np.stack(feats, axis=0).astype("float32")





def main():
    import os, time, csv, argparse
    import numpy as np, cv2, warnings
    # Silence warnings/logs
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    try:
        import onnxruntime as _ort
        if hasattr(_ort, "set_default_logger_severity"):
            _ort.set_default_logger_severity(4)  # FATAL
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="ArcFace (FaceAnalysis-only) per-frame scoring with live CSV")
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--logs",    required=True)
    ap.add_argument("--model",   required=True)   # kept for interface parity
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--batch",   type=int, default=64)  # unused (per-frame), kept for CLI parity
    args = ap.parse_args()

    aligned, ref_dir, out_csv = _prep_dirs(args)
    ref_paths = _list_imgs(str(ref_dir))
    src_paths = _list_imgs(str(aligned))

    # Header immediately
    _write_csv_header(out_csv)
    f = out_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)

    # Load recognition backend
    rec = _load_face_recognition(prefer_gpu=True)
    if rec is None:
        print("[ERR] FaceAnalysis recognition unavailable. Leaving CSV header only.")
        f.close(); return 0

    def embed_one(path):
        try:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is None: return None
            im = cv2.resize(im, (112,112), interpolation=cv2.INTER_AREA)  # expects BGR 112
            feat = rec.get_feat(im)
            if feat is None: return None
            v = np.asarray(feat, dtype=np.float32).reshape(-1)
            v /= (np.linalg.norm(v) + 1e-9)
            return v
        except Exception:
            return None

    # Build centroid from refs — label REFS, show FAIL C counter, update per-frame
    total_ref = max(1, len(ref_paths))
    p_ref = ProgressBar("REFS", total=total_ref)
    sum_vec = np.zeros((512,), dtype=np.float32); n_ref = 0; ref_fail = 0
    t0 = time.time()
    for i, pth in enumerate(ref_paths, 1):
        v = embed_one(str(pth))
        if v is not None:
            sum_vec += v; n_ref += 1
        else:
            ref_fail += 1
        fps = 0 if (time.time()-t0)<=0 else i/(time.time()-t0)
        p_ref.update(i, fails=ref_fail, fps=fps)   # FAIL C= printed by progress.py
    p_ref.close()

    # If no refs, use aligned as refs (still label as REFS)
    if n_ref == 0:
        total_ref = max(1, len(src_paths))
        p_ref = ProgressBar("REFS", total=total_ref)
        sum_vec[...] = 0; n_ref = 0; ref_fail = 0; t0 = time.time()
        for i, pth in enumerate(src_paths, 1):
            v = embed_one(str(pth))
            if v is not None:
                sum_vec += v; n_ref += 1
            else:
                ref_fail += 1
            fps = 0 if (time.time()-t0)<=0 else i/(time.time()-t0)
            p_ref.update(i, fails=ref_fail, fps=fps)
        p_ref.close()

    if n_ref == 0:
        print("[ERR] No reference embeddings. Leaving CSV header only.")
        f.close(); return 0

    centroid = sum_vec / float(n_ref)
    centroid /= (np.linalg.norm(centroid) + 1e-9)

    # Score aligned — label ARCFACE, show FAIL C, flush every row, per-frame update
    total_src = max(1, len(src_paths))
    p_src = ProgressBar("ARCFACE", total=total_src)
    t0 = time.time()
    wrote = 0; src_fail = 0
    for i, pth in enumerate(src_paths, 1):
        v = embed_one(str(pth))
        if v is not None:
            sc = float(np.dot(v, centroid))
            w.writerow([pth.name, f"{sc:.6f}"])
            f.flush()
            wrote += 1
        else:
            src_fail += 1
        fps = 0 if (time.time()-t0)<=0 else i/(time.time()-t0)
        p_src.update(i, fails=src_fail, fps=fps)
    p_src.close()
    f.close()
    print(f"[OK] wrote {wrote} rows → {out_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
PY