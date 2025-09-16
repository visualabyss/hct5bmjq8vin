#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe Face Landmarker v2 (CPU-only, silent).

Surgical updates (no behavior regressions):
- Auto-help: writes /workspace/scripts/mediapipe.txt (not *.py.txt) and supports -h/--h printing.
- Always extract everything useful:
  • CSV: full 52 blendshapes + yaw/pitch/roll + presence/landmarks.
  • Per-image JSON: landmarks_2d, all blendshapes, 4x4 facial transformation matrix, yaw/pitch/roll.
- Robust 4x4: read FaceLandmarkerResult.facial_transformation_matrixes as ndarray and flatten row-major.
- One-line progress via progress.py (dividers handled there by default).
"""

import os, sys, argparse, csv, math, json
from pathlib import Path
import imageio.v2 as iio
import numpy as np

# Hard-silence and CPU-only
os.environ.setdefault("GLOG_minloglevel","3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ["MEDIAPIPE_DISABLE_GPU"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ.setdefault("OPENCV_LOG_LEVEL","SILENT")

# Avoid self-shadowing our file name when importing mediapipe
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def import_mp():
    import importlib, sys as _sys
    removed=False
    if _SCRIPT_DIR in _sys.path:
        _sys.path.remove(_SCRIPT_DIR); removed=True
    try:
        mp = importlib.import_module("mediapipe")
        mp_tasks = importlib.import_module("mediapipe.tasks.python")
        mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
        return mp, mp_tasks, mp_vision
    finally:
        if removed:
            _sys.path.insert(0,_SCRIPT_DIR)

# Stderr squelch (C++ libs write before absl init)
class SquelchStderr:
    def __enter__(self):
        self._saved = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._saved, 2)
        os.close(self._devnull)
        os.close(self._saved)

from progress import ProgressBar

IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".webp"}

# Canonical 52 blendshape names (MediaPipe/ARKit-compatible)
BLEND_ALL = [
    "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
    "cheekPuff","cheekSquintLeft","cheekSquintRight",
    "eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft","eyeLookDownRight",
    "eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight",
    "eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft","eyeSquintRight",
    "eyeWideLeft","eyeWideRight",
    "jawForward","jawLeft","jawOpen","jawRight",
    "mouthClose","mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight",
    "mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight",
    "mouthPressLeft","mouthPressRight","mouthPucker","mouthRight",
    "mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
    "mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight",
    "noseSneerLeft","noseSneerRight","tongueOut"
]

AUTO_HELP_TEXT = r"""
PURPOSE
  • Run MediaPipe Face Landmarker v2 over aligned images and extract: face presence, landmark count,
    head pose (yaw/pitch/roll from the 4×4 facial transformation matrix), and all 52 facial blendshapes.
  • Outputs a single CSV row per image and per-image JSON payloads containing full data.
  • CPU-only, silence-first, single-line progress (via progress.py).

USAGE (common)
  python3 -u /workspace/scripts/mediapipe.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --mp_task /workspace/tools/mediapipe/face_landmarker.task \
    --fps 8 --override

CLI FLAGS
  # input / IO
  --aligned DIR (req)               images to process
  --logs DIR (req)                  log root
  --mp_task FILE (req)              Face Landmarker v2 .task model
  --override (flag)                 overwrite CSV header if schema changed

  # runtime
  --fps INT (8)                     timestamp step (ms = 1000/fps) for detect_for_video

  # logging & help
  --help on|off (on)                write /workspace/scripts/mediapipe.txt then proceed
  -h / --h                          print this help text and continue

CSV OUTPUT
  logs/mediapipe/mp_tags.csv  (append mode; header written if new)
  Columns (exact order):
    path, file, presence, landmarks, yaw, pitch, roll,
    blend::<52 canonical names>, success

PER-IMAGE JSON
  logs/mediapipe/per_image/<stem>.json (always written)
  {
    "file": "000001.png",
    "presence": 1,
    "landmarks_2d": [[x,y], ...],             # normalized in [0,1]
    "blendshapes": {"browInnerUp": 0.12, ...},
    "matrix_4x4": [[m00,m01,m02,m03], ...],  # row-major
    "yaw_pitch_roll": [yaw, pitch, roll]
  }
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


def list_images(d:Path):
    xs=[p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    xs.sort(); return xs


def mat4_to_euler(matlike):
    """Return (yaw, pitch, roll) in degrees from a 4x4 rotation matrix.
    Accepts list/tuple/ndarray; flattens row-major. Returns None on failure.
    Convention: yaw = atan2(r10, r00); pitch = asin(-r20); roll = atan2(r21, r22).
    """
    try:
        a = np.array(matlike, dtype=np.float64)
        if a.size < 16:
            # Some APIs wrap the 16 numbers as a list of lists; try to reshape
            a = a.reshape(-1)
            if a.size < 16:
                return None
        flat = a.reshape(-1)[:16]
        m = flat.reshape(4,4)
        m00,m01,m02,_m03 = m[0]
        m10,m11,m12,_m13 = m[1]
        m20,m21,m22,_m23 = m[2]
        pitch = math.degrees(math.asin(max(-1.0,min(1.0,-m20))))
        yaw   = math.degrees(math.atan2(m10, m00))
        roll  = math.degrees(math.atan2(m21, m22))
        return yaw, pitch, roll
    except Exception:
        return None


def cv2_imread_color(p:Path):
    try:
        import cv2
        cv2.setNumThreads(1)
        a = np.frombuffer(open(p,"rb").read(), dtype=np.uint8)
        im = cv2.imdecode(a, 1)  # BGR uint8
        if im is None:
            raise RuntimeError("cv2.imdecode failed")
        im = im[:,:,::-1]  # to RGB
    except Exception:
        im = iio.imread(p)
        if im.ndim==2:
            im=np.stack([im,im,im],-1)
        im = im[...,:3]
    # Ensure contiguous uint8 for mp.Image
    return np.ascontiguousarray(im, dtype=np.uint8)


def write_auto_help(enabled: bool):
    if not enabled:
        return
    try:
        script_path = Path(__file__).resolve()
        (script_path.with_name(script_path.stem + '.txt')).write_text(AUTO_HELP_TEXT, encoding='utf-8')
    except Exception:
        pass


def main():
    ap=argparse.ArgumentParser(description="MediaPipe Face Landmarker v2 (CPU-only, silent).", add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--mp_task", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--override", action="store_true")
    ap.add_argument("--help", dest="auto_help", default="on")
    args=ap.parse_args()
    args.auto_help = _parse_onoff(args.auto_help, default=True)

    if getattr(args,'show_cli_help', False):
        print(AUTO_HELP_TEXT)
    write_auto_help(args.auto_help)

    imgs=list_images(args.aligned); total=len(imgs)
    out_dir=args.logs/"mediapipe"; out_dir.mkdir(parents=True,exist_ok=True)
    perimg_dir=(out_dir/"per_image"); perimg_dir.mkdir(parents=True,exist_ok=True)
    csv_path=out_dir/"mp_tags.csv"
    if total==0:
        return 2

    done=set()
    if csv_path.exists() and not args.override:
        try:
            with csv_path.open("r",encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    done.add(r.get("file",""))
        except Exception:
            pass

    headers=[
        "path","file","presence","landmarks","yaw","pitch","roll",
        *[f"blend::{k}" for k in BLEND_ALL],
        "success"
    ]
    if args.override or not csv_path.exists():
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=headers).writeheader()

    mp, mp_tasks, mp_vision = import_mp()
    BaseOptions=mp_tasks.BaseOptions
    FaceLandmarker=mp_vision.FaceLandmarker
    FaceLandmarkerOptions=mp_vision.FaceLandmarkerOptions
    RunningMode=mp_vision.RunningMode

    with SquelchStderr():
        opts=FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.mp_task), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        lm=FaceLandmarker.create_from_options(opts)

    bar=ProgressBar("MEDIAPIPE", total=total, show_fail_label=True)
    fails=0; processed=0; ts=0; dt=max(1,int(1000/max(1,args.fps)))
    CHUNK=1000; buf=[]

    def flush():
        if not buf:
            return
        with csv_path.open("a",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=headers).writerows(buf)
        buf.clear()

    for p in imgs:
        if p.name in done:
            processed+=1; bar.update(processed,fails=fails); ts+=dt; continue
        im=cv2_imread_color(p)
        # mp.Image: use explicit kwargs with contiguous uint8
        with SquelchStderr():
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im)
        try:
            res = lm.detect_for_video(mp_image, ts)
        except Exception:
            res=None

        presence=0; landmarks_n=0; yaw=pitch=roll=""
        blends={k: "" for k in BLEND_ALL}
        mat4_json = []
        lm2d_json = []

        if res and res.face_landmarks:
            presence=1
            lms = res.face_landmarks[0]
            landmarks_n=len(lms)
            lm2d_json = [[float(pt.x), float(pt.y)] for pt in lms]

        if res and getattr(res,"facial_transformation_matrixes",None):
            M = res.facial_transformation_matrixes[0]
            ypr = mat4_to_euler(M)
            if ypr:
                yaw,pitch,roll=[f"{a:.3f}" for a in ypr]
            try:
                M = np.array(M).reshape(4,4)
                mat4_json = [[float(M[r,c]) for c in range(4)] for r in range(4)]
            except Exception:
                mat4_json = []

        if res and res.face_blendshapes:
            for cat in res.face_blendshapes[0]:
                n = str(cat.category_name)
                v = f"{float(cat.score):.4f}"
                if n in blends:
                    blends[n]=v
                else:
                    # Unknown name (future-proof): ignore in CSV, still store in JSON
                    pass

        # per-image JSON (always)
        try:
            j = {
                "file": p.name,
                "presence": int(presence),
                "landmarks_2d": lm2d_json,
                "blendshapes": {k: (float(blends[k]) if blends[k] != "" else 0.0) for k in BLEND_ALL},
                "matrix_4x4": mat4_json,
                "yaw_pitch_roll": [float(yaw) if yaw else 0.0, float(pitch) if pitch else 0.0, float(roll) if roll else 0.0],
            }
            (perimg_dir / (p.stem + ".json")).write_text(json.dumps(j), encoding="utf-8")
        except Exception:
            pass

        row = {
            "path": str(p),
            "file": p.name,
            "presence": presence,
            "landmarks": landmarks_n,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            **{f"blend::{k}": blends[k] for k in BLEND_ALL},
            "success": 1 if presence else 0,
        }
        buf.append(row)

        processed+=1
        if not presence:
            fails+=1
        bar.update(processed,fails=fails)
        ts+=dt
        if len(buf)>=CHUNK:
            flush()

    flush(); bar.update(total,fails=fails); bar.close()
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
