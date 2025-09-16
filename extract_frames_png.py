#!/usr/bin/env python3
import os, sys, argparse, subprocess, shlex, glob, json

VIDEO_EXTS = ('.mp4','.mkv','.mov','.m4v','.avi','.webm','.mpg','.mpeg')

def first_video_in(path):
    if os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS):
        return path
    if os.path.islink(path):
        rp = os.path.realpath(path)
        if os.path.isfile(rp) and rp.lower().endswith(VIDEO_EXTS):
            return rp
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            p = os.path.join(path, name)
            if os.path.isfile(p) and p.lower().endswith(VIDEO_EXTS):
                return p
    return None

def run_ffmpeg(cmd):
    print(">>", " ".join(shlex.quote(x) for x in cmd), flush=True)
    return subprocess.run(cmd).returncode

def probe_fps(video):
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error",
            "-select_streams","v:0",
            "-show_entries","stream=avg_frame_rate,r_frame_rate",
            "-of","json", video
        ]).decode("utf-8","ignore")
        data = json.loads(out)
        st = data.get("streams",[{}])[0]
        for key in ("avg_frame_rate","r_frame_rate"):
            val = st.get(key, "0/0")
            if isinstance(val,str) and "/" in val:
                num, den = val.split("/",1)
                try:
                    num = float(num); den = float(den)
                    if den != 0:
                        fps = num/den
                        if fps > 0:
                            return fps
                except Exception:
                    pass
    except Exception as e:
        print(f"[warn] ffprobe failed: {e}", file=sys.stderr)
    return None

def main():
    ap = argparse.ArgumentParser(description="Extract PNG frames from a video (with CFR/VFR control).")
    ap.add_argument("--video", default="", help="Path to input video; if empty, auto-search common folders")
    ap.add_argument("--out", required=True, help="Output folder for frames")
    ap.add_argument("--fps", type=float, default=0.0, help="Target FPS when CFR=1; 0 means auto-detect native fps")
    ap.add_argument("--cfr", type=int, default=1, help="Force constant frame rate output (duplicates/drops as needed)")
    ap.add_argument("--vfr", type=int, default=0, help="Preserve variable frame rate; ignores --fps")
    ap.add_argument("--start_number", type=int, default=0, help="Starting index for output numbering")
    ap.add_argument("--compression_level", type=int, default=3, help="PNG compression level (0-9)")
    args = ap.parse_args()

    if args.cfr and args.vfr:
        print("[error] --cfr and --vfr are mutually exclusive. Choose one.", file=sys.stderr)
        sys.exit(2)
    if not args.cfr and not args.vfr:
        args.cfr = 1  # default to CFR

    video = args.video
    if not video:
        candidates = [
            ".", "/workspace", "/workspace/data_trg", "/workspace/data_src",
            "/workspace/target", "/workspace/source", "/workspace/source_video"
        ]
        for c in candidates:
            v = first_video_in(c)
            if v:
                video = v; break
    if not video or not os.path.exists(video):
        print("[fatal] No input video found. Use --video or place a video in a common folder.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)
    pattern = os.path.join(args.out, "%06d.png")

    # Decide filters & vsync
    vf_parts = []
    vsync = None
    if args.vfr:
        vsync = "vfr"
        vf_parts = []
    else:
        vsync = "cfr"
        fps_val = args.fps if args.fps and args.fps > 0 else (probe_fps(video) or 30.0)
        vf_parts.append(f"fps={fps_val:.6g}")

    vf_cuda = ",".join([*vf_parts, "hwdownload", "format=nv12"] if vf_parts else ["hwdownload","format=nv12"])
    cmd_cuda = [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-hwaccel","cuda","-hwaccel_output_format","cuda",
        "-i", video,
    ]
    if vsync:
        cmd_cuda += ["-vsync", vsync]
    if vf_cuda:
        cmd_cuda += ["-vf", vf_cuda]
    cmd_cuda += [
        "-pix_fmt","rgb24",
        "-start_number", str(args.start_number),
        "-compression_level", str(args.compression_level),
        pattern
    ]

    rc = run_ffmpeg(cmd_cuda)
    if rc != 0:
        vf_cpu = ",".join(vf_parts) if vf_parts else None
        cmd_cpu = ["ffmpeg","-hide_banner","-loglevel","error","-y",
                   "-i", video]
        if vsync:
            cmd_cpu += ["-vsync", vsync]
        if vf_cpu:
            cmd_cpu += ["-vf", vf_cpu]
        cmd_cpu += [
            "-pix_fmt","rgb24",
            "-start_number", str(args.start_number),
            "-compression_level", str(args.compression_level),
            pattern
        ]
        rc = run_ffmpeg(cmd_cpu)
        if rc != 0:
            print("[FATAL] ffmpeg failed (CUDA and CPU).", file=sys.stderr)
            sys.exit(rc)

    frames = sorted(glob.glob(os.path.join(args.out, "*.png")))
    print(f"[DONE] Extracted {len(frames)} PNG frames to {args.out}")
    if frames:
        print(f"[STATS] First: {os.path.basename(frames[0])}  Last: {os.path.basename(frames[-1])}")

if __name__ == "__main__":
    main()
