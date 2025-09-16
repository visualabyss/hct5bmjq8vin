#!/usr/bin/env python3
import os, sys, argparse, csv, math
from pathlib import Path
import imageio.v2 as iio
import numpy as np

# Hard-silence and CPU-only
os.environ.setdefault("GLOG_minloglevel","3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ["MEDIAPIPE_DISABLE_GPU"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ.setdefault("OPENCV_LOG_LEVEL","SILENT")

# Avoid self-shadowing our file name
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def import_mp():
    import importlib, sys as _sys
    removed=False
    if _SCRIPT_DIR in _sys.path: _sys.path.remove(_SCRIPT_DIR); removed=True
    try:
        mp = importlib.import_module("mediapipe")
        mp_tasks = importlib.import_module("mediapipe.tasks.python")
        mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
        return mp, mp_tasks, mp_vision
    finally:
        if removed: _sys.path.insert(0,_SCRIPT_DIR)

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
BLEND_KEEP=[
    "eyeBlinkLeft","eyeBlinkRight","browInnerUp",
    "browDownLeft","browDownRight","browOuterUpLeft","browOuterUpRight",
    "mouthOpen","mouthSmileLeft","mouthSmileRight","jawOpen","mouthPucker"
]

def list_images(d:Path):
    xs=[p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    xs.sort(); return xs

def mat4_to_euler_flat(flat16):
    try:
        if not flat16 or len(flat16)<16: return None
        m00,m01,m02,_m03, m10,m11,m12,_m13, m20,m21,m22,_m23, *_ = flat16
        pitch = math.degrees(math.asin(max(-1.0,min(1.0,-m20))))
        yaw   = math.degrees(math.atan2(m10,m00))
        roll  = math.degrees(math.atan2(m21,m22))
        return yaw,pitch,roll
    except Exception:
        return None

def cv2_imread_color(p:Path):
    try:
        import cv2
        cv2.setNumThreads(1)
        a = np.frombuffer(open(p,"rb").read(), dtype=np.uint8)
        im = cv2.imdecode(a, 1)   # BGR uint8
        if im is None: raise RuntimeError("cv2.imdecode failed")
        im = im[:,:,::-1]         # to RGB
    except Exception:
        im = iio.imread(p)
        if im.ndim==2: im=np.stack([im,im,im],-1)
        im = im[...,:3]
    # Ensure contiguous uint8 for mp.Image
    return np.ascontiguousarray(im, dtype=np.uint8)

def main():
    ap=argparse.ArgumentParser(description="MediaPipe Face Landmarker v2 (CPU-only, silent).")
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--logs",    required=True, type=Path)
    ap.add_argument("--mp_task", required=True, type=Path)
    ap.add_argument("--fps",     type=int, default=8)
    ap.add_argument("--override", action="store_true")
    args=ap.parse_args()

    imgs=list_images(args.aligned); total=len(imgs)
    out_dir=args.logs/"mediapipe"; out_dir.mkdir(parents=True,exist_ok=True)
    csv_path=out_dir/"mp_tags.csv"
    if total==0: return 2

    done=set()
    if csv_path.exists() and not args.override:
        with csv_path.open("r",encoding="utf-8") as f:
            for r in csv.DictReader(f): done.add(r["file"])

    headers=["path","file","presence","landmarks","yaw","pitch","roll",
             *[f"blend::{k}" for k in BLEND_KEEP], "success"]
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
            base_options=BaseOptions(model_asset_path=str(args.mp_task),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        lm=FaceLandmarker.create_from_options(opts)

    bar=ProgressBar("MEDIAPIPE", total=total, show_fail_label=True)
    fails=0; processed=0
    ts=0; dt=max(1,int(1000/max(1,args.fps)))
    CHUNK=1000; buf=[]

    def flush():
        if not buf: return
        with csv_path.open("a",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=headers).writerows(buf)
        buf.clear()

    for p in imgs:
        if p.name in done:
            processed+=1; bar.update(processed,fails=fails); ts+=dt; continue

        im=cv2_imread_color(p)
        # mp.Image: use positional args with contiguous uint8
        mp_image = None
        with SquelchStderr():
            mp_image = mp.Image(mp.ImageFormat.SRGB, im)
            try:
                res = lm.detect_for_video(mp_image, ts)
            except Exception:
                res=None

        presence=0; landmarks=0; yaw=pitch=roll=""
        blends={k:"" for k in BLEND_KEEP}

        if res and res.face_landmarks:
            presence=1; landmarks=len(res.face_landmarks[0])
            if getattr(res,"facial_transformation_matrixes",None):
                ypr=mat4_to_euler_flat(res.facial_transformation_matrixes[0])
                if ypr: yaw,pitch,roll=[f"{a:.3f}" for a in ypr]
            if res.face_blendshapes:
                for cat in res.face_blendshapes[0]:
                    n=cat.category_name
                    if n in blends: blends[n]=f"{cat.score:.4f}"

        buf.append({
            "path":p.name,"file":p.name,
            "presence":presence,"landmarks":landmarks,
            "yaw":yaw,"pitch":pitch,"roll":roll,
            **{f"blend::{k}":blends[k] for k in BLEND_KEEP},
            "success":1 if presence else 0
        })
        processed+=1
        if not presence: fails+=1
        bar.update(processed,fails=fails)
        ts+=dt
        if len(buf)>=CHUNK: flush()

    flush()
    bar.update(total,fails=fails); bar.close()
    return 0

if __name__=="__main__":
    try: sys.exit(main())
    except KeyboardInterrupt: sys.exit(130)
