#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools.py
Pipeline orchestrator: run detectors (OpenFace3, MediaPipe, OpenSeeFace, ArcFace, MagFace) and then
aggregate with curator_tag.py. Flags are on/off. Writes a living help file tools.py.txt.

This script NEVER moves images; it only runs tools and writes logs + manifest.
"""
import os, sys, argparse, json, subprocess, shlex, traceback
from pathlib import Path

def _onoff(v:str)->bool:
    v=(v or "").strip().lower()
    if v in ("on","true","yes","y","1"): return True
    if v in ("off","false","no","n","0"): return False
    raise argparse.ArgumentTypeError("expected on/off")

def _ensure_dir(p:Path)->Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _write_help(parser: argparse.ArgumentParser, defaults: dict, notes: str):
    out = Path("/workspace/scripts")/ (Path(__file__).name + ".txt")
    try: usage = parser.format_help()
    except Exception: usage = "(no argparse help)\n"
    body=[]
    body.append("=== Purpose ===\nRun selected tools and then aggregate into manifest.jsonl (no moves).\n\n")
    body.append("=== Flags & Defaults ===\n")
    for k,v in defaults.items(): body.append(f"- {k}: {v}\n")
    body.append("\n=== Usage ===\n"); body.append(usage)
    body.append("\n=== Notes ===\n"); body.append(notes.strip()+"\n")
    out.write_text("".join(body), encoding="utf-8")

def _run(cmd: str, env: dict|None=None, cwd: str|None=None) -> int:
    print(f"\n[RUN] {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, env=env).returncode

def main():
    ap = argparse.ArgumentParser("tools: run detectors (on/off) and aggregate manifest")
    ap.add_argument("--aligned", type=Path, default=Path("/workspace/data_src/aligned"))
    ap.add_argument("--logs",    type=Path, default=Path("/workspace/data_src/logs"))

    # toggles
    ap.add_argument("--openface3",  choices=["on","off"], default="on")
    ap.add_argument("--mediapipe",  choices=["on","off"], default="on")
    ap.add_argument("--openseeface",choices=["on","off"], default="off")
    ap.add_argument("--arcface",    choices=["on","off"], default="off")
    ap.add_argument("--magface",    choices=["on","off"], default="off")
    ap.add_argument("--tag",        choices=["on","off"], default="on")

    # OpenFace3 params
    ap.add_argument("--of3_root", type=Path, default=Path("/workspace/tools/OpenFace-3.0"))
    ap.add_argument("--of3_device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--of3_amp", choices=["on","off"], default="on")
    ap.add_argument("--of3_dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--of3_tf32", choices=["on","off"], default="on")
    ap.add_argument("--of3_skip_detect", choices=["on","off"], default="on")
    ap.add_argument("--of3_per_image", choices=["on","off"], default="on")
    ap.add_argument("--of3_override", choices=["on","off"], default="off")

    # MediaPipe params
    ap.add_argument("--mp_task", type=Path, default=Path("/workspace/tools/mediapipe/face_landmarker_v2.task"))
    ap.add_argument("--mp_gpu_only", choices=["on","off"], default="off")
    ap.add_argument("--mp_override", choices=["on","off"], default="off")

    # OpenSeeFace params
    ap.add_argument("--osf_root", type=Path, default=Path("/workspace/tools/OpenSeeFace"))
    ap.add_argument("--osf_py",   type=Path, default=Path("/workspace/envs/osf_env/bin/python"))
    ap.add_argument("--osf_compile", choices=["on","off"], default="off")
    ap.add_argument("--osf_fps", type=int, default=8)
    ap.add_argument("--osf_override", choices=["on","off"], default="off")

    # ArcFace/MagFace params (placeholders; models wired later)
    ap.add_argument("--arc_model", type=Path, default=Path("/workspace/tools/arcface_r100.onnx"))
    ap.add_argument("--arc_refdir",type=Path, default=Path("/workspace/data_src_ref/aligned"))
    ap.add_argument("--mag_model", type=Path, default=Path("/workspace/models/magface.onnx"))

    # manifest/curator_tag
    ap.add_argument("--allow_dupes", choices=["on","off"], default="off")
    ap.add_argument("--use_list", type=Path, default=None)
    ap.add_argument("--override", choices=["on","off"], default="on")

    args = ap.parse_args()

    defaults = {k: getattr(args,k) if not isinstance(getattr(args,k), Path) else str(getattr(args,k))
                for k in vars(args)}
    _write_help(ap, defaults, notes="""
- This orchestrator launches individual scripts in their native envs:
  * OpenFace3 via `of3_py -u /workspace/scripts/openface3.py …`
  * MediaPipe via `mp_py -u /workspace/scripts/mediapipe.py …`
  * OpenSeeFace via system python: `/workspace/scripts/openseeface.py --py /workspace/envs/osf_env/bin/python …`
- Flags are on/off here; underlying tools may still accept numeric flags. This script converts accordingly.
- No images are ever moved here; dedupe/tag manifests inform downstream selection.
""")

    aligned = args.aligned; logs = args.logs
    _ensure_dir(aligned); _ensure_dir(logs)

    # helpers to convert on/off -> 1/0 for older scripts
    b = lambda v: "1" if _onoff(v) else "0"

    # 1) OpenFace3
    if _onoff(args.openface3):
        cmd = (
            f"of3_py -u /workspace/scripts/openface3.py "
            f"--aligned {shlex.quote(str(aligned))} "
            f"--logs {shlex.quote(str(logs))} "
            f"--of3_root {shlex.quote(str(args.of3_root))} "
            f"--device {args.of3_device} "
            f"--amp {b(args.of3_amp)} --dtype {args.of3_dtype} --tf32 {b(args.of3_tf32)} "
            f"--skip_detect {b(args.of3_skip_detect)} --per_image {b(args.of3_per_image)} "
            f"--override {b(args.of3_override)}"
        )
        rc = _run(cmd); 
        if rc != 0: print("[WARN] OpenFace3 step returned", rc)

    # 2) MediaPipe
    if _onoff(args.mediapipe):
        cmd = (
            f"mp_py -u /workspace/scripts/mediapipe.py "
            f"--aligned {shlex.quote(str(aligned))} "
            f"--logs {shlex.quote(str(logs))} "
            f"--mp_task {shlex.quote(str(args.mp_task))} "
            f"--gpu_only {b(args.mp_gpu_only)} --override {b(args.mp_override)}"
        )
        rc = _run(cmd); 
        if rc != 0: print("[WARN] MediaPipe step returned", rc)

    # 3) OpenSeeFace
    if _onoff(args.openseeface):
        cmd = (
            f"python3 -u /workspace/scripts/openseeface.py "
            f"--aligned {shlex.quote(str(aligned))} "
            f"--logs {shlex.quote(str(logs))} "
            f"--osf_root {shlex.quote(str(args.osf_root))} "
            f"--py {shlex.quote(str(args.osf_py))} "
            f"--compile {b(args.osf_compile)} --fps {args.osf_fps} --override {b(args.osf_override)}"
        )
        rc = _run(cmd); 
        if rc != 0: print("[WARN] OpenSeeFace step returned", rc)

    # 4) ArcFace (optional — will use arcface.py if/when configured)
    if _onoff(args.arcface):
        cmd = (
            f"python3 -u /workspace/scripts/arcface.py "
            f"--aligned {shlex.quote(str(aligned))} "
            f"--logs {shlex.quote(str(logs))} "
            f"--model {shlex.quote(str(args.arc_model))} "
            f"--ref_dir {shlex.quote(str(args.arc_refdir))}"
        )
        rc = _run(cmd); 
        if rc != 0: print("[WARN] ArcFace step returned", rc)

    # 5) MagFace (optional)
    if _onoff(args.magface):
        cmd = (
            f"python3 -u /workspace/scripts/magface.py "
            f"--aligned {shlex.quote(str(aligned))} "
            f"--logs {shlex.quote(str(logs))} "
            f"--model {shlex.quote(str(args.mag_model))}"
        )
        rc = _run(cmd); 
        if rc != 0: print("[WARN] MagFace step returned", rc)

    # 6) Aggregate with curator_tag
    if _onoff(args.tag):
        cmd = (
            f"python3 -u /workspace/scripts/curator_tag.py "
            f"--aligned {shlex.quote(str(aligned))} "
            f"--logs {shlex.quote(str(logs))} "
            f"--openface3 {'on' if _onoff(args.openface3) else 'off'} "
            f"--mediapipe {'on' if _onoff(args.mediapipe) else 'off'} "
            f"--openseeface {'on' if _onoff(args.openseeface) else 'off'} "
            f"--arcface {'on' if _onoff(args.arcface) else 'off'} "
            f"--magface {'on' if _onoff(args.magface) else 'off'} "
            f"--allow_dupes {args.allow_dupes} "
            f"--override {args.override} "
            + (f"--use_list {shlex.quote(str(args.use_list))}" if args.use_list else "")
        )
        rc = _run(cmd); 
        if rc != 0: print("[WARN] curator_tag step returned", rc)

    print("\n[OK] tools pipeline finished.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        print("FATAL:\n"+traceback.format_exc()); sys.exit(1)
