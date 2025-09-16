#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, argparse, traceback, warnings, contextlib, io, inspect
from pathlib import Path
import json
import itertools
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# ---- constants ----
IMAGE_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}
EMOTIONS = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]
# OF3 (public) AU order (see repo images/au/*.png): 1,2,4,6,9,12,25,26
AU_ORDER = [1,2,4,6,9,12,25,26]
AU_INDEX = {1:0, 2:1, 4:2, 6:3, 9:4, 12:5, 25:6, 26:7}

# ---- helpers ----
def list_images(root: Path):
    return [p for p in sorted(Path(root).rglob("*")) if p.suffix.lower() in IMAGE_EXTS]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

@contextlib.contextmanager
def _quiet():
    old = warnings.filters[:]
    warnings.simplefilter("ignore")
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield
    warnings.filters = old

def _softmax(z):
    z = np.array(z, dtype=np.float32).reshape(-1)
    if z.size == 0: return z
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    return ez/s if s > 0 else np.zeros_like(z)

def _to1d(x):
    if hasattr(x, "detach"):
        try: return x.detach().float().cpu().view(-1).numpy()
        except Exception: pass
    return np.array(x).reshape(-1)

def _get_bar(total: int):
    try:
        from progress import ProgressBar
        return ProgressBar("OPENFACE3", total=total, show_fail_label=True)
    except Exception:
        class _FB:
            def __init__(self, total): self.total=int(total or 0)
            def update(self, i, fails=0):
                if self.total and ((i%500==0) or (i==self.total)):
                    pct = 100.0*i/self.total
                    print(f"[OPENFACE3] {i}/{self.total}  {pct:5.1f}%  fails={fails}", flush=True)
            def close(self): pass
        return _FB(total)

def _calibrate_au_index(multitask, of3_root: Path):
    """
    Build 8x8 matrix M where row r is model AU vector for reference image of AU_ORDER[r].
    Solve argmax over permutations perm of sum_r M[r, perm[r]].
    Returns (map_au_to_idx, M_list, row_margins, best_sum) or None.
    """
    try:
        au_set = [1,2,4,6,9,12,25,26]
        au_dir = of3_root / "images" / "au"
        refs = [(n, au_dir / f"au_{n}.png") for n in au_set]
        if not all(p.exists() for _,p in refs):
            return None
        def to1d(x):
            import torch, numpy as np
            if hasattr(x, "detach"):
                return x.detach().float().cpu().view(-1).numpy()
            return np.array(x).reshape(-1)
        import numpy as np, cv2
        rows = []
        for n, imgp in refs:
            im = cv2.imread(str(imgp))
            out = multitask.predict(im)
            if isinstance(out, dict):
                au = out.get("au")
            elif isinstance(out, (list, tuple)) and len(out)>=3:
                au = out[2]
            else:
                return None
            v = to1d(au)
            if v.size != 8: 
                return None
            rows.append(v)
        M = np.stack(rows, axis=0)  # 8x8
        best_perm = None; best_val = -1e9
        for perm in itertools.permutations(range(8)):
            val = float(M[range(8), list(perm)].sum())
            if val > best_val:
                best_val, best_perm = val, perm
        # per-row margins for diagnostics
        row_marg = []
        for r in range(8):
            row = M[r]
            top = float(row[best_perm[r]])
            srt = np.sort(row)[::-1]
            margin = float(top - (srt[1] if len(srt)>1 else 0.0))
            row_marg.append(margin)
        map_au_to_idx = {au_set[r]: int(best_perm[r]) for r in range(8)}
        return map_au_to_idx, M.tolist(), row_marg, float(best_val)
    except Exception:
        return None

# ---- package-only imports ----
def _import_of3():
    from openface.face_detection import FaceDetector  # type: ignore
    from openface.multitask_model import MultitaskPredictor  # type: ignore
    return FaceDetector, MultitaskPredictor

# ---- main ----
def main():
    ap = argparse.ArgumentParser("OpenFace-3.0 (package) -> logs/openface3/openface3.csv")
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--logs",    required=True, type=Path)
    ap.add_argument("--of3_root", required=True, type=Path)  # just for weights path
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--device",  default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--skip_detect", type=int, default=1)
    ap.add_argument("--detector_resize", type=float, default=1.0)
    ap.add_argument("--override", action="store_true")
    ap.add_argument("--per_image", type=int, default=1)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["fp32","fp16","bf16"])
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--no_calib", type=int, default=0)
    ap.add_argument("--calib_cache", type=int, default=1)
    ap.add_argument("--th_smile", type=float, default=0.5)
    ap.add_argument("--th_mouth", type=float, default=0.5)
    args = ap.parse_args()

    # perf knobs
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    try:
        torch.set_num_threads(1)
        if args.tf32 and args.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    AMP = bool(args.amp and args.device=="cuda")
    DTYPE = {"fp32":torch.float32, "fp16":torch.float16, "bf16":torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    imgs = list_images(args.aligned)
    if not imgs:
        print("No images under --aligned.")
        return 2

    FaceDetector, MultitaskPredictor = _import_of3()
    wdir = args.weights if args.weights else (args.of3_root / "weights")
    face_w = wdir / "Alignment_RetinaFace.pth"
    mtl_w  = wdir / "MTL_backbone.pth"
    mobilenet_pre = wdir / "mobilenetV1X0.25_pretrain.tar"
    if not mtl_w.exists():
        print(f"[ERROR] Missing multitask weights: {mtl_w}")
        return 2
    if not args.skip_detect and (not face_w.exists() or not mobilenet_pre.exists()):
        print(f"[ERROR] Missing detector weights: {face_w} and/or {mobilenet_pre}")
        return 2

    # init models
    with _quiet():
        detector = None
        if not args.skip_detect:
            detector = FaceDetector(model_path=str(face_w), device=args.device)
        sig = inspect.signature(MultitaskPredictor.__init__)
        if "tasks" in sig.parameters:
            try:
                multitask = MultitaskPredictor(model_path=str(mtl_w), device=args.device, tasks=["emotion","gaze","au"])
            except TypeError:
                multitask = MultitaskPredictor(model_path=str(mtl_w), device=args.device)
        else:
            multitask = MultitaskPredictor(model_path=str(mtl_w), device=args.device)
        for attr in ("enable_au","return_au","use_au","au_on"):
            if hasattr(multitask, attr):
                try: setattr(multitask, attr, True)
                except Exception: pass
        if hasattr(multitask, "set_tasks"):
            try: multitask.set_tasks(au=True, emotion=True, gaze=True)
            except Exception: pass
    # --- AU mapping: runtime calibration against reference images ---
    AU_INDEX_RUNTIME = None
    calib_meta = None
    # try cached mapping first
    if not args.no_calib and args.calib_cache:
        try:
            cache_p = (args.logs / "openface3" / "au_info.json")
            if cache_p.exists():
                import json
                data = json.loads(cache_p.read_text(encoding="utf-8"))
                c = data.get("calibration") or {}
                m = c.get("map_au_to_idx")
                if m:
                    AU_INDEX_RUNTIME = {int(k): int(v) for k,v in m.items()}
                    calib_meta = {"method":"cache","map_au_to_idx": AU_INDEX_RUNTIME}
        except Exception:
            pass
    if not args.no_calib and AU_INDEX_RUNTIME is None:
        cal = _calibrate_au_index(multitask, args.of3_root)
        if cal is not None:
            AU_INDEX_RUNTIME, M_mat, row_margins, best_sum = cal
            calib_meta = {
                "method": "runtime_calibration",
                "map_au_to_idx": AU_INDEX_RUNTIME,
                "matrix": M_mat,
                "row_margins": row_margins,
                "score": best_sum,
                "au_order": [1,2,4,6,9,12,25,26]
            }


    # outputs
    out_dir_meta = ensure_dir(args.logs / "openface3")
    try:
        (out_dir_meta/"au_info.json").write_text(json.dumps({
            "source": "openface-test",
            "order": [1,2,4,6,9,12,25,26],
            "names": ["AU1","AU2","AU4","AU6","AU9","AU12","AU25","AU26"],
            "note": "OF3 public multitask head (8 AUs); AU45 not provided",
            "calibration": calib_meta if 'calib_meta' in locals() else None
        }, indent=2), encoding='utf-8')
    except Exception:
        pass

    out_dir = ensure_dir(args.logs / "openface3")
    perimg_dir = ensure_dir(out_dir / "per_image")
    out_csv = out_dir / "openface3.csv"
    headers = ["path","file","pose_yaw","pose_pitch","eyes","mouth","smile","teeth",
               "emotion","emotion_conf","au12","au25","au26","au45","confidence","success"]
    mode = "w" if (args.override or not out_csv.exists()) else "a"
    f = out_csv.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=headers)
    if mode == "w": writer.writeheader()

    bar = _get_bar(len(imgs))
    fails = 0; wrote_debug = False

    for i, img in enumerate(imgs, 1):
        row = {"path":str(img), "file":img.name,
               "pose_yaw":"", "pose_pitch":"",
               "eyes":"", "mouth":"", "smile":"", "teeth":"",
               "emotion":"", "emotion_conf":"",
               "au12":"", "au25":"", "au26":"", "au45":"",
               "confidence":"", "success":0}
        try:
            # crop
            if args.skip_detect:
                crop = cv2.imread(str(img))
                if crop is None: raise RuntimeError("cannot_read_image")
            else:
                with torch.inference_mode():
                    if AMP:
                        with torch.amp.autocast("cuda", dtype=DTYPE):
                            crop, dets = detector.get_face(str(img), resize=args.detector_resize)  # type: ignore
                    else:
                        crop, dets = detector.get_face(str(img), resize=args.detector_resize)      # type: ignore
                if crop is None: raise RuntimeError("no_face")
                try:
                    if dets is not None and len(dets)>0 and len(dets[0])>=5:
                        row["confidence"] = f"{float(dets[0][4]):.4f}"
                except Exception: pass

            # predict -> (emotion, gaze, au8?) ; some builds return dict, others tuple
            with torch.inference_mode():
                if AMP:
                    with torch.amp.autocast("cuda", dtype=DTYPE):
                        out = multitask.predict(crop)
                else:
                    out = multitask.predict(crop)

            emo_logits = gaze = au = None
            if isinstance(out, dict):
                emo_logits = out.get("emotion"); gaze = out.get("gaze"); au = out.get("au")
            elif isinstance(out, (list, tuple)):
                if len(out) >= 3:
                    emo_logits, gaze, au = out[0], out[1], out[2]
                elif len(out) == 2:
                    emo_logits, gaze = out[0], out[1]
                else:
                    emo_logits = out
            else:
                emo_logits = out

            emo = _to1d(emo_logits) if emo_logits is not None else np.zeros(0)
            gaze_v = _to1d(gaze) if gaze is not None else np.zeros(0)
            au_v = _to1d(au) if au is not None else np.zeros(0)

            # pose
            if gaze_v.size >= 2:
                row["pose_yaw"]   = f"{float(gaze_v[0]):.3f}"
                row["pose_pitch"] = f"{float(gaze_v[1]):.3f}"

            # emotion
            if emo.size >= 1:
                probs = _softmax(emo); k = int(np.argmax(probs))
                row["emotion"] = EMOTIONS[k] if 0<=k<len(EMOTIONS) else str(k)
                row["emotion_conf"] = f"{float(probs[k]):.4f}"

            # AU mapping (8-dim public head)
            if au_v.size == 8:
                # Heuristics + columns we need
                idx_map = AU_INDEX_RUNTIME if AU_INDEX_RUNTIME else AU_INDEX
                def av(n): return float(au_v[idx_map[n]])
                au_map = {n: av(n) for n in AU_ORDER}
                # fill requested columns
                row["au12"] = f"{au_map[12]:.3f}"
                row["au25"] = f"{au_map[25]:.3f}"
                row["au26"] = f"{au_map[26]:.3f}"
                # AU45 not available in OF3 public head
                # derived tags
                row["eyes"] = "open"  # AU45 absent -> leave neutral
                mv = max(au_map[25], au_map[26])
                row["mouth"] = "open" if mv >= args.th_mouth else "closed"
                row["smile"] = "smile" if au_map[12] >= args.th_smile else "neutral"
                if row["smile"] == "smile" and row["mouth"] == "open":
                    row["teeth"] = "likely"
            elif au_v.size > 8:
                # Future-proof: if a bigger AU tensor appears with names in the model
                pass  # (script already works; we just don't know mapping yet)

            row["success"] = 1
            if not row.get("confidence"):
                row["confidence"] = row.get("emotion_conf") or "0.9900"

            # per-image TSV (always include all 8 AU values we have)
            if args.per_image:
                tsv = (perimg_dir / (img.stem + ".tsv"))
                with tsv.open("w", encoding="utf-8", newline="") as tf:
                    tw = csv.writer(tf, delimiter="\t")
                    tw.writerow(["gaze_yaw","gaze_pitch","AU_vec","emotion_logits"])
                    gy = f"{float(gaze_v[0]):.6f}" if gaze_v.size>0 else ""
                    gp = f"{float(gaze_v[1]):.6f}" if gaze_v.size>1 else ""
                    if au_v.size == 8:
                        au_list = [f"{float(au_v[i]):.6f}" for i in range(8)]
                    else:
                        au_list = []
                    emolist = [f"{float(v):.6f}" for v in emo.tolist()] if emo.size>0 else []
                    tw.writerow([gy, gp, ",".join(au_list), ",".join(emolist)])

        except Exception:
            fails += 1
            if not wrote_debug:
                try: (out_dir/"error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                except Exception: pass
                wrote_debug = True

        writer.writerow(row)
        bar.update(i, fails)

    bar.close(); f.close()
    print(f"[DONE] wrote {out_csv}")
    if wrote_debug:
        print(f"[NOTE] First error traceback: {out_dir/'error.txt'}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        print("FATAL:\n" + traceback.format_exc()); sys.exit(1)
