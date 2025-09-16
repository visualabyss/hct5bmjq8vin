#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, os, subprocess, sys, time, re, socket, struct, threading, queue, json
from pathlib import Path
from progress import ProgressBar

IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".webp"}

# ---------------- Auto-help ----------------
AUTO_HELP_TEXT = r"""
PURPOSE
  • Run OpenSeeFace facetracker on an aligned image sequence (compiled to MJPG) and record per-frame signals.
  • Extracted signals:
      - Head pose (yaw, pitch, roll), translation (tx, ty, tz), blinkL/blinkR, tracker confidence & fit error,
        PnP error, and per-stage timings (proc/detect/crop/track/points3d) from stdout.
      - UDP stream values (timestamp, face_id, width, height, yaw/pitch/roll, tx/ty/tz, blink L/R, pnp_error,
        quaternion raw [q0,q1,q2,q3]).
  • Outputs:
      - logs/openseeface/osf_tags.csv (append/refresh; one row per image in order).
      - logs/openseeface/per_image/<stem>.json (always) with full payload merged from stdout + UDP.
  • One-line progress via progress.py (dividers handled globally).

USAGE (common)
  python3 -u /workspace/scripts/openseeface.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --osf_root /workspace/tools/OpenSeeFace \
    --py /workspace/envs/osf_env/bin/python \
    --compile on --fps 8 --override

CLI FLAGS
  # input / IO
  --aligned DIR (req)               aligned image directory
  --logs DIR (req)                  log root (CSV + per-image JSON under logs/openseeface)
  --osf_root DIR (req)              OpenSeeFace repo root containing facetracker.py
  --py FILE                         Python interpreter for OSF (default: current)
  --override (flag)                 rewrite CSV header if schema changed

  # runtime
  --fps INT (8)                     FPS for the temporary MJPG video
  --compile on|off (on)             build MJPG from images (accepts on/off/1/0/true/false)
  --udp_port INT (11573)            UDP port to listen for OSF stream

  # logging & help
  --help on|off (on)                write /workspace/scripts/openseeface.txt then proceed
  -h / --h                          print this help text and continue
"""

def _parse_onoff(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, (int, float)):
        return bool(int(v))
    s=str(v).strip().lower()
    if s in {"1","on","true","yes","y"}: return True
    if s in {"0","off","false","no","n"}: return False
    return bool(default)

# ---------------- Utility ----------------
def list_images(d:Path):
    xs=[p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    xs.sort()
    return xs

def compile_video(imgs,fps,dst:Path):
    import cv2, imageio.v2 as iio, numpy as np
    vw=cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, tuple(iio.imread(imgs[0]).shape[1::-1]))
    if not vw.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed (pip install opencv-python-headless).")
    bar=ProgressBar("COMPILING", total=len(imgs), show_fail_label=False); bar._min_interval=0.12
    for i,p in enumerate(imgs,1):
        im=iio.imread(p)
        if im.ndim==2: im=np.stack([im,im,im],-1)
        vw.write(im[..., :3][:,:,::-1]); bar.update(i,None)
    vw.release(); bar.close()

def dump_help(py_exe:str, ft_path:Path, cwd:Path, log:Path):
    try:
        out=subprocess.check_output([py_exe, str(ft_path), "--help"], cwd=str(cwd),
                                    stderr=subprocess.STDOUT, text=True, timeout=8)
    except Exception as e:
        out=f"(help failed: {e})"
    with open(log,"a",encoding="utf-8") as lf:
        lf.write("\n[facetracker --help]\n"); lf.write(out); lf.write("\n")

# ---- OSF UDP parser (single face) ----
# Layout reference: timestamp(double), face_id(int32), width(float), height(float),
# blinkL(float), blinkR(float), success(uint8), pnp_error(float),
# quat(4*float), euler(3*float), tvec(3*float), ... (rest ignored)
# Source on structure: OSF docs + community parsers.

def parse_osf_udp_packet(pkt: bytes):
    o=0
    if len(pkt) < 73:  # minimal prefix we need
        return None
    ts   = struct.unpack_from("<d", pkt, o)[0]; o+=8
    face = struct.unpack_from("<i", pkt, o)[0]; o+=4
    W    = struct.unpack_from("<f", pkt, o)[0]; o+=4
    H    = struct.unpack_from("<f", pkt, o)[0]; o+=4
    blinkL = struct.unpack_from("<f", pkt, o)[0]; o+=4
    blinkR = struct.unpack_from("<f", pkt, o)[0]; o+=4
    success = pkt[o]; o+=1  # uint8
    pnp_err = struct.unpack_from("<f", pkt, o)[0]; o+=4
    # quat (raw order from OSF stream)
    q0,q1,q2,q3 = struct.unpack_from("<ffff", pkt, o); o+=16
    # euler (yaw, pitch, roll) — degrees
    e0,e1,e2 = struct.unpack_from("<fff", pkt, o); o+=12
    # translation (x,y,z)
    tx,ty,tz = struct.unpack_from("<fff", pkt, o); o+=12
    return {
        "timestamp": ts, "face_id": face, "width": W, "height": H,
        "blink_l": blinkL, "blink_r": blinkR,
        "success": int(success), "pnp_error": pnp_err,
        "yaw": e0, "pitch": e1, "roll": e2,
        "tx": tx, "ty": ty, "tz": tz,
        "quat_raw": [q0,q1,q2,q3],
    }

# ---------------- UDP listener ----------------

def udp_listener(port:int, out_q:queue.Queue, stop_evt:threading.Event, raw_path:Path=None):
    sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.2)
    raw_f = open(raw_path,"ab") if raw_path else None
    try:
        while not stop_evt.is_set():
            try:
                pkt,_=sock.recvfrom(65507)
            except socket.timeout:
                continue
            if raw_f: raw_f.write(pkt)
            data=parse_osf_udp_packet(pkt)
            if data: out_q.put(data)
    finally:
        if raw_f: raw_f.close()
        sock.close()

# ---------------- Main ----------------

def _write_auto_help(enabled: bool):
    if not enabled:
        return
    try:
        sp = Path(__file__).resolve()
        (sp.with_name(sp.stem + '.txt')).write_text(AUTO_HELP_TEXT, encoding='utf-8')
    except Exception:
        pass


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    ap.add_argument("--aligned",required=True,type=Path)
    ap.add_argument("--logs",required=True,type=Path)
    ap.add_argument("--osf_root",required=True,type=Path)
    ap.add_argument("--fps",type=int,default=8)
    ap.add_argument("--py",type=str,default=sys.executable,help="Python for facetracker")
    ap.add_argument("--override",action="store_true")
    ap.add_argument("--compile",default="on",help="on|off — build MJPG from images (default on)")
    ap.add_argument("--udp_port",type=int,default=11573)
    ap.add_argument("--help", dest="auto_help", default="on")
    args=ap.parse_args()

    if getattr(args,'show_cli_help', False):
        print(AUTO_HELP_TEXT)
    args.auto_help = _parse_onoff(args.auto_help, True)
    _write_auto_help(args.auto_help)
    args.compile = _parse_onoff(args.compile, True)

    imgs=list_images(args.aligned); total=len(imgs)
    out_dir=args.logs/"openseeface"; out_dir.mkdir(parents=True,exist_ok=True)
    perimg_dir=(out_dir/"per_image"); perimg_dir.mkdir(parents=True,exist_ok=True)
    vid_path=out_dir/"osf_input.avi"
    ft_path=args.osf_root/"facetracker.py"
    ft_log =out_dir/"facetracker.log"
    csv_path=out_dir/"osf_tags.csv"
    udp_bin =out_dir/"osf_udp_raw.bin"

    if total==0:
        print("OPENSEEFACE: no images under --aligned.")
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=["path","file","success"]).writeheader()
        return 2

    if args.compile:
        if vid_path.exists():
            try: vid_path.unlink()
            except: pass
        compile_video(imgs,args.fps,vid_path)
    else:
        if not vid_path.exists():
            print("OPENSEEFACE: --compile off but video missing. Re-run with --compile on first.")
            return 3

    # CSV with extended fields (stdout + UDP)
    headers = ["path","file","confidence","fit_error","eye_l","eye_r",
               "proc_ms","detect_ms","crop_ms","track_ms","points3d_ms",
               "yaw","pitch","roll","tx","ty","tz","blink_l","blink_r","pnp_error","success"]
    if args.override or not csv_path.exists():
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=headers); w.writeheader()
            for p in imgs:
                w.writerow({"path":p.name,"file":p.name,
                            "confidence":"","fit_error":"","eye_l":"","eye_r":"",
                            "proc_ms":"","detect_ms":"","crop_ms":"","track_ms":"","points3d_ms":"",
                            "yaw":"","pitch":"","roll":"","tx":"","ty":"","tz":"",
                            "blink_l":"","blink_r":"","pnp_error":"",
                            "success":0})

    # also build per-frame JSON placeholders (we'll write at end)
    json_frames=[{
        "file": p.name,
        "path": str(p),
        "timestamp": None,
        "face_id": None,
        "dims": None,
        "blink": {"L": None, "R": None},
        "confidence": None,
        "fit_error": None,
        "pnp_error": None,
        "pose_euler": {"yaw": None, "pitch": None, "roll": None},
        "pose_quat_raw": None,
        "tvec": {"x": None, "y": None, "z": None},
        "eyes": {"L": None, "R": None},
        "timings_ms": {"proc": None, "detect": None, "crop": None, "track": None, "points3d": None},
        "success": 0,
        "landmarks_2d": [],
        "landmarks_3d": [],
    } for p in imgs]

    # log --help and exact command
    dump_help(args.py, ft_path, args.osf_root, ft_log)
    cmd=[args.py, str(ft_path), "-c", str(vid_path), "-i","127.0.0.1","-p",str(args.udp_port)]
    with open(ft_log,"a",encoding="utf-8") as lf:
        lf.write("\n[facetracker cmd]\n"+" ".join(cmd)+"\n")

    # CPU-only for stability
    env=os.environ.copy()
    env["PYTHONUNBUFFERED"]="1"; env["CUDA_VISIBLE_DEVICES"]="-1"
    env["ORT_CUDA_UNAVAILABLE"]="1"; env["ORT_DISABLE_TENSORRT"]="1"

    # Start UDP listener
    q_udp = queue.Queue()
    stop_evt = threading.Event()
    t_udp = threading.Thread(target=udp_listener, args=(args.udp_port,q_udp,stop_evt,udp_bin), daemon=True)
    t_udp.start()

    proc=subprocess.Popen(cmd, cwd=str(args.osf_root), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    bar=ProgressBar("OPENSEEFAC", total=total, show_fail_label=True); bar._min_interval=0.10
    took_re = re.compile(
        r"Took\s+([0-9.]+)ms\s*\(\s*detect:\s*([0-9.]+)ms,\s*crop:\s*([0-9.]+)ms,\s*track:\s*([0-9.]+)ms,\s*3D points:\s*([0-9.]+)ms\)",
        re.I
    )
    conf_re = re.compile(r"Confidence\[\d+\]\s*:\s*([0-9.]+)\s*/\s*3D\s+fitting\s+error:\s*([0-9.]+)\s*/\s*Eyes:\s*([OC]),\s*([OC])", re.I)

    with csv_path.open("r",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    parsed_stdout = 0
    success_frames = 0
    last_timing=None
    last_flush=time.time()
    FLUSH_EVERY=0.5

    def flush_csv():
        nonlocal last_flush
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=headers); w.writeheader(); w.writerows(rows)
        last_flush=time.time()

    # helper: apply one UDP packet to next unfilled row and JSON cache
    def apply_udp(pkt):
        nonlocal success_frames
        # choose row = first with missing yaw (keeps sync with frame order)
        for idx,r in enumerate(rows):
            if r["yaw"]=="":
                r["yaw"]=f'{pkt["yaw"]:.3f}'; r["pitch"]=f'{pkt["pitch"]:.3f}'; r["roll"]=f'{pkt["roll"]:.3f}'
                r["tx"]=f'{pkt["tx"]:.3f}'; r["ty"]=f'{pkt["ty"]:.3f}'; r["tz"]=f'{pkt["tz"]:.3f}'
                r["blink_l"]=f'{pkt["blink_l"]:.3f}'; r["blink_r"]=f'{pkt["blink_r"]:.3f}'
                r["pnp_error"]=f'{pkt["pnp_error"]:.4f}'
                if pkt["success"]==1:  # mark as success if stdout didn’t already
                    if r["success"] in ("",0,"0"): r["success"]=1; success_frames+=1
                # JSON cache
                jf = json_frames[idx]
                jf["timestamp"] = float(pkt["timestamp"]) if pkt.get("timestamp") is not None else None
                jf["face_id"] = int(pkt.get("face_id")) if pkt.get("face_id") is not None else None
                jf["dims"] = [float(pkt.get("width",0.0)), float(pkt.get("height",0.0))]
                jf["blink"]["L"] = float(pkt["blink_l"]); jf["blink"]["R"] = float(pkt["blink_r"])
                jf["pnp_error"] = float(pkt["pnp_error"])
                jf["pose_euler"] = {"yaw": float(pkt["yaw"]), "pitch": float(pkt["pitch"]), "roll": float(pkt["roll"])}
                jf["pose_quat_raw"] = list(pkt.get("quat_raw", []))
                jf["tvec"] = {"x": float(pkt["tx"]), "y": float(pkt["ty"]), "z": float(pkt["tz"])}
                jf["success"] = int(r["success"]) if str(r["success"])!="" else int(pkt["success"])  # prefer CSV flag
                return True
        return False

    with open(ft_log,"a",encoding="utf-8") as lf:
        while True:
            # drain UDP first to keep up
            drained=0
            while True:
                try:
                    pkt=q_udp.get_nowait()
                except queue.Empty:
                    break
                apply_udp(pkt); drained+=1
            if drained and (time.time()-last_flush)>=FLUSH_EVERY:
                flush_csv()

            # read one stdout line (non-blocking-ish)
            line = proc.stdout.readline()
            if line:
                lf.write(line)
                mt = took_re.search(line)
                if mt:
                    last_timing = tuple(float(mt.group(i)) for i in range(1,6))
                    est_done = min(total, parsed_stdout + 1)
                    fails_so_far = max(0, est_done - success_frames)
                    bar.update(est_done, fails=fails_so_far)
                    continue

                mc = conf_re.search(line)
                if mc:
                    parsed_stdout += 1
                    idx = parsed_stdout-1
                    if idx < total:
                        conf = float(mc.group(1)); fit=float(mc.group(2)); el=mc.group(3); er=mc.group(4)
                        r = rows[idx]
                        r["confidence"]=f"{conf:.4f}"; r["fit_error"]=f"{fit:.4f}"; r["eye_l"]=el; r["eye_r"]=er
                        # JSON cache
                        jf = json_frames[idx]
                        jf["confidence"] = conf; jf["fit_error"] = fit; jf["eyes"] = {"L": el, "R": er}
                        if last_timing:
                            proc_ms, det_ms, cr_ms, tr_ms, p3d_ms = last_timing
                            r["proc_ms"]=f"{proc_ms:.2f}"; r["detect_ms"]=f"{det_ms:.2f}"
                            r["crop_ms"]=f"{cr_ms:.2f}";  r["track_ms"]=f"{tr_ms:.2f}"; r["points3d_ms"]=f"{p3d_ms:.2f}"
                            jf["timings_ms"] = {"proc":proc_ms, "detect":det_ms, "crop":cr_ms, "track":tr_ms, "points3d":p3d_ms}
                            last_timing=None
                        r["success"]=1; jf["success"]=1
                        success_frames += 1
                        if (time.time()-last_flush)>=FLUSH_EVERY or parsed_stdout==total:
                            flush_csv()
                    fails_so_far = max(0, parsed_stdout - success_frames)
                    bar.update(min(parsed_stdout,total), fails=fails_so_far)
            else:
                if proc.poll() is not None: break
                time.sleep(0.02)

    rc=proc.wait()
    stop_evt.set(); t_udp.join(timeout=1.0)

    final_fails = max(0, len([r for r in rows if r["success"] in ("",0,"0")]))
    bar.update(total, fails=final_fails); bar.close()
    flush_csv()

    # Write per-image JSON payloads (always)
    for p,jf in zip(imgs, json_frames):
        try:
            (perimg_dir / (p.stem + ".json")).write_text(json.dumps(jf, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    print(f"OPENSEEFACE: done. parsed_frames={parsed_stdout} success_frames={success_frames} fails={final_fails} video={vid_path.exists()} rc={rc}", flush=True)
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
