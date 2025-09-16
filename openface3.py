#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenFace 3.0 runner — logs/openface3/

Adjustments:
- Help file name: save as /workspace/scripts/openface3.txt (NOT *.py.txt).
- De-dupe: don't backfill `confidence` from `emotion_conf` when detector is skipped; leave empty unless detector provides it.
- Auto-help, on|off normalization, extras remain as previously added.
"""

import os, sys, csv, argparse, traceback, warnings, contextlib, io, inspect, json, itertools, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401
import cv2

# ---------------- constants ----------------
IMAGE_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}
EMOTIONS = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]
AU_ORDER = [1,2,4,6,9,12,25,26]
AU_INDEX = {1:0, 2:1, 4:2, 6:3, 9:4, 12:5, 25:6, 26:7}

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------- progress helper ---------------
def _get_bar(total: int):
    try:
        from progress import ProgressBar
        return ProgressBar("OPENFACE3", total=total, show_fail_label=True)
    except Exception:
        class _FB:
            def __init__(self, total):
                self.total=int(total or 0)
            def update(self, i, fails=0):
                if self.total and ((i%500==0) or (i==self.total)):
                    pct = 100.0*i/self.total
                    print(f"[OPENFACE3] {i}/{self.total} {pct:5.1f}% fails={fails}", flush=True)
            def close(self):
                pass
        return _FB(total)

# --------------- small utils ---------------
def list_images(root: Path):
    return [p for p in sorted(Path(root).rglob("*")) if p.suffix.lower() in IMAGE_EXTS]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

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
    if z.size == 0:
        return z
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    return ez/s if s > 0 else np.zeros_like(z)

def _to1d(x):
    if hasattr(x, "detach"):
        try:
            return x.detach().float().cpu().view(-1).numpy()
        except Exception:
            pass
    return np.array(x).reshape(-1)

# ---------------- AU calibration ----------------
def _calibrate_au_index(multitask, of3_root: Path):
    try:
        au_set = AU_ORDER
        au_dir = of3_root / "images" / "au"
        refs = [(n, au_dir / f"au_{n}.png") for n in au_set]
        if not all(p.exists() for _,p in refs):
            return None
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
            v = _to1d(au)
            if v.size != 8:
                return None
            rows.append(v)
        M = np.stack(rows, axis=0)
        best_perm = None; best_val = -1e9
        for perm in itertools.permutations(range(8)):
            val = float(M[range(8), list(perm)].sum())
            if val > best_val:
                best_val, best_perm = val, perm
        row_marg = []
        for r in range(8):
            row = M[r]; top = float(row[best_perm[r]])
            srt = np.sort(row)[::-1]; margin = float(top - (srt[1] if len(srt)>1 else 0.0))
            row_marg.append(margin)
        map_au_to_idx = {au_set[r]: int(best_perm[r]) for r in range(8)}
        return map_au_to_idx, M.tolist(), row_marg, float(best_val)
    except Exception:
        return None

# ---------------- package imports ----------------
def _import_of3():
    from openface.face_detection import FaceDetector  # type: ignore
    from openface.multitask_model import MultitaskPredictor  # type: ignore
    return FaceDetector, MultitaskPredictor

# ---------------- auto-help ----------------
AUTO_HELP_TEXT = r"""
PURPOSE
  • Run OpenFace 3.0 on aligned crops to extract head pose (yaw/pitch via gaze), emotion (softmax + top label),
    and 8 Action Units (AU1,2,4,6,9,12,25,26). Optional detection if --skip_detect off.
  • Calibrate AU index mapping at runtime from reference images and cache to logs/openface3/au_info.json.
  • Outputs:
      - logs/openface3/openface3.csv (append unless --override): one row per image.
      - logs/openface3/per_image/<stem>.tsv: raw gaze (yaw,pitch), AU vector(8), emotion logits.
      - logs/openface3/au_info.json: AU mapping metadata.

USAGE (common)
  of3_py -u /workspace/scripts/openface3.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --of3_root /workspace/tools/OpenFace-3.0 \
    --device cuda --amp on --dtype bf16 --tf32 on \
    --skip_detect on --per_image on --override

CLI FLAGS
  # input / IO
  --aligned DIR (req)               images to process
  --logs DIR (req)                  log root
  --override (flag)                 overwrite CSV header if new schema; default off unless flag present

  # runtime / compute
  --device {cuda,cpu} (cuda)
  --amp on|off (on if cuda)
  --dtype {fp32,fp16,bf16} (bf16)
  --tf32 on|off (on)

  # detection
  --skip_detect on|off (on)
  --detector_resize FLOAT (1.0)
  --weights DIR (defaults: <of3_root>/weights)
  --of3_root DIR (req)

  # thresholds & behavior
  --th_smile FLOAT (0.5)
  --th_mouth FLOAT (0.5)
  --no_calib on|off (off)
  --calib_cache on|off (on)

  # extras (opt-in/off by default)
  --extra_csv on|off (off)          write openface3_full.csv with all 8 AU + optional emotion top-k
  --csv_emotion_topk INT (0)        0 disables; >0 writes top-k as label:prob pairs in extra CSV
  --landmarks on|off (off)          save landmarks per-image if available
  --landmarks_fmt {json,pts,both} (json)

  # logging & help
  --help on|off (on)                write /workspace/scripts/openface3.txt then proceed
  -h / --h                          print this help text and continue

LOG OUTPUT
  logs/openface3/openface3.csv      (append mode; header written if file is new)
  logs/openface3/per_image/*.tsv    (always written when --per_image on)
  logs/openface3/au_info.json       (AU mapping cache & metadata)
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

# Pre-parse argv to allow on/off with legacy int flags
_ONOFF_KEYS = {"--amp","--tf32","--skip_detect","--per_image","--no_calib","--calib_cache"}
_newargv=[]; it=iter(range(len(sys.argv)))
for i in it:
    a=sys.argv[i]
    if a in _ONOFF_KEYS and i+1 < len(sys.argv):
        v=sys.argv[i+1].lower()
        if v in {"on","off"}:  # convert for argparse int parsers if any remain
            sys.argv[i+1] = "1" if v=="on" else "0"
    _newargv.append(a)


def parse_args():
    ap = argparse.ArgumentParser("OpenFace-3.0 (package) -> logs/openface3/openface3.csv", add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    # IO
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--of3_root", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=None)
    # runtime
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--amp", default="1")
    ap.add_argument("--dtype", default="bf16", choices=["fp32","fp16","bf16"])
    ap.add_argument("--tf32", default="1")
    # detection
    ap.add_argument("--skip_detect", default="1")
    ap.add_argument("--detector_resize", type=float, default=1.0)
    # legacy override flag remains store_true for compatibility
    ap.add_argument("--override", action="store_true")
    ap.add_argument("--per_image", default="1")
    # calib
    ap.add_argument("--no_calib", default="0")
    ap.add_argument("--calib_cache", default="1")
    # thresholds
    ap.add_argument("--th_smile", type=float, default=0.5)
    ap.add_argument("--th_mouth", type=float, default=0.5)
    # extras
    ap.add_argument("--extra_csv", default="0")
    ap.add_argument("--csv_emotion_topk", type=int, default=0)
    ap.add_argument("--landmarks", default="0")
    ap.add_argument("--landmarks_fmt", choices=["json","pts","both"], default="json")
    # help
    ap.add_argument("--help", dest="auto_help", default="on")
    args = ap.parse_args()
    # normalize
    args.amp = _parse_onoff(args.amp, default=(args.device=="cuda"))
    args.tf32 = _parse_onoff(args.tf32, default=True)
    args.skip_detect = _parse_onoff(args.skip_detect, default=True)
    args.per_image = _parse_onoff(args.per_image, default=True)
    args.no_calib = _parse_onoff(args.no_calib, default=False)
    args.calib_cache = _parse_onoff(args.calib_cache, default=True)
    args.extra_csv = _parse_onoff(args.extra_csv, default=False)
    args.landmarks = _parse_onoff(args.landmarks, default=False)
    args.auto_help = _parse_onoff(args.auto_help, default=True)
    return args


def _write_auto_help(enabled: bool):
    if not enabled:
        return
    try:
        script_path = Path(__file__).resolve()
        (script_path.with_name(script_path.stem + '.txt')).write_text(AUTO_HELP_TEXT, encoding='utf-8')
    except Exception:
        pass


def main():
    # perf knobs
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    args = parse_args()
    if getattr(args, 'show_cli_help', False):
        print(AUTO_HELP_TEXT)
    _write_auto_help(args.auto_help)

    if args.tf32 and args.device == "cuda":
        try:
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
    mtl_w = wdir / "MTL_backbone.pth"
    mobilenet_pre = wdir / "mobilenetV1X0.25_pretrain.tar"

    if not mtl_w.exists():
        print(f"[ERROR] Missing multitask weights: {mtl_w}")
        return 2
    if not args.skip_detect and (not face_w.exists() or not mobilenet_pre.exists()):
        print(f"[ERROR] Missing detector weights: {face_w} and/or {mobilenet_pre}")
        return 2

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

    # --- AU mapping: cached or runtime calibration ---
    AU_INDEX_RUNTIME = None
    calib_meta = None
    out_dir_meta = ensure_dir(args.logs / "openface3")
    cache_p = out_dir_meta / "au_info.json"

    if (not args.no_calib) and args.calib_cache and cache_p.exists():
        try:
            data = json.loads(cache_p.read_text(encoding="utf-8"))
            c = (data.get("calibration") or {})
            m = c.get("map_au_to_idx")
            if m: AU_INDEX_RUNTIME = {int(k): int(v) for k,v in m.items()}
            calib_meta = {"method":"cache","map_au_to_idx": AU_INDEX_RUNTIME}
        except Exception:
            pass

    if (not args.no_calib) and AU_INDEX_RUNTIME is None:
        cal = _calibrate_au_index(multitask, args.of3_root)
        if cal is not None:
            AU_INDEX_RUNTIME, M_mat, row_margins, best_sum = cal
            calib_meta = {"method":"runtime_calibration","map_au_to_idx": AU_INDEX_RUNTIME,
                          "matrix": M_mat, "row_margins": row_margins, "score": best_sum, "au_order": AU_ORDER}

    try:
        cache_p.write_text(json.dumps({
            "source": "openface3",
            "order": AU_ORDER,
            "names": [f"AU{n}" for n in AU_ORDER],
            "note": "OF3 public multitask head (8 AUs); AU45 not provided",
            "calibration": calib_meta
        }, indent=2), encoding='utf-8')
    except Exception:
        pass

    # --- outputs ---
    out_dir = out_dir_meta
    perimg_dir = ensure_dir(out_dir / "per_image")

    # main CSV (unchanged schema)
    out_csv = out_dir / "openface3.csv"
    headers = [
        "path","file","pose_yaw","pose_pitch","eyes","mouth","smile","teeth",
        "emotion","emotion_conf","au12","au25","au26","au45","confidence","success"
    ]
    mode = "w" if (args.override or not out_csv.exists()) else "a"
    f = out_csv.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=headers)
    if mode == "w":
        writer.writeheader()

    # optional extra CSV
    extra_writer = None
    if args.extra_csv:
        out_csv2 = out_dir / "openface3_full.csv"
        headers2 = ["path","file","pose_yaw","pose_pitch",
                    "emotion_topk"] + [f"au{n}" for n in AU_ORDER]
        mode2 = "w" if (args.override or not out_csv2.exists()) else "a"
        f2 = out_csv2.open(mode2, newline="", encoding="utf-8")
        extra_writer = (f2, csv.DictWriter(f2, fieldnames=headers2))
        if mode2 == "w":
            extra_writer[1].writeheader()
    else:
        f2 = None

    bar = _get_bar(len(imgs))
    fails = 0; wrote_debug = False

    for i, img in enumerate(imgs, 1):
        row = {k: "" for k in headers}
        row.update({"path":str(img), "file":img.name, "success":0})
        try:
            # crop
            det_conf = None
            if args.skip_detect:
                crop = cv2.imread(str(img))
                if crop is None:
                    raise RuntimeError("cannot_read_image")
            else:
                with torch.inference_mode():
                    if AMP:
                        with torch.amp.autocast("cuda", dtype=DTYPE):
                            crop, dets = detector.get_face(str(img), resize=args.detector_resize)  # type: ignore
                    else:
                        crop, dets = detector.get_face(str(img), resize=args.detector_resize)  # type: ignore
                if crop is None:
                    raise RuntimeError("no_face")
                try:
                    if dets is not None and len(dets)>0 and len(dets[0])>=5:
                        det_conf = float(dets[0][4])
                        row["confidence"] = f"{det_conf:.4f}"
                except Exception:
                    pass

            # predict -> (emotion, gaze, au)
            with torch.inference_mode():
                if AMP:
                    with torch.amp.autocast("cuda", dtype=DTYPE):
                        out = multitask.predict(crop)
                else:
                    out = multitask.predict(crop)

            emo_logits = gaze = au = None
            landmarks = None
            if isinstance(out, dict):
                emo_logits = out.get("emotion"); gaze = out.get("gaze"); au = out.get("au")
                landmarks = out.get("landmarks") or out.get("lm")
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
                row["pose_yaw"] = f"{float(gaze_v[0]):.3f}"
                row["pose_pitch"] = f"{float(gaze_v[1]):.3f}"

            # emotion
            if emo.size >= 1:
                probs = _softmax(emo)
                k = int(np.argmax(probs))
                row["emotion"] = EMOTIONS[k]
                row["emotion_conf"] = f"{float(probs[k]):.4f}"

            # AU logic
            au_map = {}
            if au_v.size == 8:
                idx = AU_INDEX_RUNTIME or AU_INDEX
                for n in AU_ORDER:
                    au_map[n] = float(au_v[idx[n]])
                row["au12"] = f"{au_map[12]:.4f}"
                row["au25"] = f"{au_map[25]:.4f}"
                row["au26"] = f"{au_map[26]:.4f}"
                # heuristics
                mv = max(au_map[25], au_map[26])
                row["mouth"] = "open" if mv >= args.th_mouth else "closed"
                row["smile"] = "smile" if au_map[12] >= args.th_smile else "neutral"
                if row["smile"] == "smile" and row["mouth"] == "open":
                    row["teeth"] = "likely"

            row["success"] = 1

            # per-image TSV
            if args.per_image:
                tsv = (perimg_dir / (img.stem + ".tsv"))
                with tsv.open("w", encoding="utf-8", newline="") as tf:
                    tw = csv.writer(tf, delimiter="\t")
                    tw.writerow(["gaze_yaw","gaze_pitch","AU_vec","emotion_logits"]) 
                    gy = f"{float(gaze_v[0]):.6f}" if gaze_v.size>0 else ""
                    gp = f"{float(gaze_v[1]):.6f}" if gaze_v.size>1 else ""
                    au_list = [f"{float(au_v[i]):.6f}" for i in range(8)] if au_v.size==8 else []
                    emolist = [f"{float(v):.6f}" for v in emo.tolist()] if emo.size>0 else []
                    tw.writerow([gy, gp, ",".join(au_list), ",".join(emolist)])

            # landmarks (optional)
            if args.landmarks and landmarks is not None:
                pts = np.array(landmarks).reshape(-1, 2)
                if args.landmarks_fmt in ("json","both"):
                    (perimg_dir / (img.stem + ".landmarks.json")).write_text(
                        json.dumps({"points": pts.tolist()}), encoding='utf-8')
                if args.landmarks_fmt in ("pts","both"):
                    with (perimg_dir / (img.stem + ".pts")).open("w", encoding="utf-8") as pf:
                        pf.write("version: 1\n"+"n_points: {}\n".format(len(pts))+"{\n")
                        for (x,y) in pts:
                            pf.write(f"{float(x):.3f} {float(y):.3f}\n")
                        pf.write("}\n")

            # optional extra CSV
            if extra_writer is not None:
                f2, w2 = extra_writer
                topk = ""
                if args.csv_emotion_topk and emo.size>0:
                    probs = _softmax(emo)
                    order = list(np.argsort(-probs))[:int(args.csv_emotion_topk)]
                    topk = ",".join([f"{EMOTIONS[j]}:{float(probs[j]):.4f}" for j in order])
                row2 = {
                    "path": str(img),
                    "file": img.name,
                    "pose_yaw": row.get("pose_yaw",""),
                    "pose_pitch": row.get("pose_pitch",""),
                    "emotion_topk": topk,
                }
                if au_v.size==8:
                    idx = AU_INDEX_RUNTIME or AU_INDEX
                    for n in AU_ORDER:
                        row2[f"au{n}"] = f"{float(au_v[idx[n]]):.4f}"
                else:
                    for n in AU_ORDER:
                        row2[f"au{n}"] = ""
                w2.writerow(row2)

        except Exception:
            fails += 1
            if not wrote_debug:
                try:
                    (out_dir/"error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass
                wrote_debug = True
        finally:
            writer.writerow(row)
            bar.update(i, fails)

    bar.close(); f.close()
    if extra_writer is not None:
        extra_writer[0].close()
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
