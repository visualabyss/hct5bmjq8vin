#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, json, argparse, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import math
from progress import ProgressBar
import shutil

VERBOSE = os.environ.get("OF2_VERBOSE", "0").strip().lower() in {"1","true","on","yes","y"}

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

AUTO_HELP = r"""
--aligned DIR --logs DIR [--download on|off] [--binary PATH] [--override]
[--img_tool auto|img|seq] [--extra_args "..."] [--help on|off]
Outputs: logs/openface2/per_image/*.csv and a merged logs/openface2/openface2.csv
"""


def _onoff(v, default=False) -> bool:
    if isinstance(v,bool): return v
    if v is None: return bool(default)
    s=str(v).strip().lower()
    if s in {"1","on","true","yes","y"}: return True
    if s in {"0","off","false","no","n"}: return False
    return bool(default)


def _list_images(d: Path) -> List[Path]:
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _count_rows(work_dir: Path) -> int:
    total = 0
    for e in os.scandir(work_dir):
        if not e.is_file():
            continue
        if not e.name.lower().endswith('.csv'):
            continue
        total += 1
    return total


def _run_with_progress(cmd: List[str], work_dir: Path, total: int, env: Optional[Dict[str,str]] = None, aligned_name: Optional[str] = None, force_of: Optional[str] = None) -> int:
    bar = ProgressBar("OPENFACE2", total=total, show_fail_label=True)
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, bufsize=0, env=env)
    start = time.time()
    processed = 0
    fps = 0
    bar.update(0, fails=0, fps=0)

    def pick_csv() -> Optional[Path]:
        cands = []
        if force_of:
            cands.append(work_dir/force_of)
            cands.append(work_dir/"processed"/force_of)
        if aligned_name:
            an = f"{aligned_name}.csv"
            cands += [work_dir/an, work_dir/"processed"/an, work_dir/Path(aligned_name)/an]
        newest = None; newest_t = -1
        for root,_,files in os.walk(work_dir):
            for fn in files:
                if not fn.lower().endswith('.csv'):
                    continue
                pth = Path(root)/fn
                try:
                    t = pth.stat().st_mtime
                except Exception:
                    continue
                for c in cands:
                    if pth == c and pth.exists():
                        return pth
                if t > newest_t:
                    newest_t = t; newest = pth
        return newest

    csv_path: Optional[Path] = None
    try:
        while True:
            if csv_path is None:
                csv_path = pick_csv()
            new_processed = processed
            if csv_path and csv_path.exists():
                try:
                    with csv_path.open('r', encoding='utf-8', newline='') as f:
                        c = sum(1 for _ in f)
                    new_processed = max(0, c - 1)
                except Exception:
                    new_processed = processed
            if new_processed != processed:
                processed = new_processed
                elapsed = max(1e-6, time.time() - start)
                fps = int(processed / elapsed)
                bar.update(min(processed, total), fails=0, fps=fps)
            if p.poll() is not None:
                break
            time.sleep(0.05)
        rc = p.wait()
        bar.update(min(processed, total), fails=0, fps=fps)
        bar.close()
        return int(rc)
    except KeyboardInterrupt:
        try:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        finally:
            bar.close()
        return 130

def _ensure_repo(repo_dir: Path, download: bool) -> None:(repo_dir: Path, download: bool) -> None:
    if repo_dir.exists() and (repo_dir/".git").exists():
        return
    if not download:
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git","clone","--depth","1","https://github.com/TadasBaltrusaitis/OpenFace", str(repo_dir)])
    models = repo_dir/"download_models.sh"
    if models.exists():
        subprocess.call(["bash", str(models)])


def _find_binary(repo_dir: Path, user_path: Optional[Path], tool: str) -> Optional[Path]:
    cand = []
    if user_path:
        if user_path.is_dir():
            cand += [user_path/"FaceLandmarkImg", user_path/"FeatureExtraction",
                     user_path/"FaceLandmarkImg.exe", user_path/"FeatureExtraction.exe"]
        else:
            cand += [user_path]
    cand += [repo_dir/"build"/"bin"/tool, repo_dir/"bin"/tool, repo_dir/"exe"/(tool+".exe")]
    for c in cand:
        if c and c.exists():
            return c
    return None


def _ensure_built(repo_dir: Path) -> None:
    fe = _find_binary(repo_dir, None, "FeatureExtraction")
    fl = _find_binary(repo_dir, None, "FaceLandmarkImg")
    if fe or fl:
        return
    inst = repo_dir/"install.sh"
    if inst.exists():
        subprocess.call(["bash", str(inst)], cwd=repo_dir)
        fe = _find_binary(repo_dir, None, "FeatureExtraction")
        fl = _find_binary(repo_dir, None, "FaceLandmarkImg")
        if fe or fl:
            return
    build = repo_dir/"build"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cmake","-D","CMAKE_BUILD_TYPE=Release",".."], cwd=build)
    subprocess.check_call(["make", f"-j{max(1, os.cpu_count() or 1)}"], cwd=build)


def _inject_env(repo_dir: Path) -> Dict[str,str]:
    env = os.environ.copy()
    env["OF2_HOME"] = str(repo_dir)
    env["PATH"] = str(repo_dir/"build"/"bin") + os.pathsep + str(repo_dir/"bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(repo_dir/"build"/"lib") + os.pathsep + "/usr/local/lib" + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["OPENCV_LOG_LEVEL"] = "ERROR"
    env["GLOG_minloglevel"] = "2"
    env["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
    return env


def _rad2deg(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    try: return float(x)*180.0/math.pi
    except Exception: return None


def _discover_fields(csv_path: Path) -> Tuple[List[str], List[str], List[str], List[str], bool, bool]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        flds = rd.fieldnames or []
    au_r_cols = [c for c in flds if c.startswith("AU") and c.endswith("_r")]
    au_c_cols = [c for c in flds if c.startswith("AU") and c.endswith("_c")]
    gaze_cols = [c for c in flds if c.startswith("gaze_")]
    pose_cols = [c for c in flds if c.startswith("pose_")]
    has_frame = ("frame" in flds)
    has_ts = ("timestamp" in flds)
    return au_r_cols, au_c_cols, gaze_cols, pose_cols, has_frame, has_ts


def _read_of_csv(csv_path: Path, files: List[Path], au_r_cols: List[str], au_c_cols: List[str], gaze_cols: List[str], pose_cols: List[str]) -> List[Dict[str,object]]:
    rows: List[Dict[str,object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fns = [p.name for p in files]
        for i, r in enumerate(rd):
            name = r.get("input") or r.get("filename")
            if not name:
                name = fns[i] if i < len(fns) else str(i)
            # angles in radians -> deg
            yaw_deg = _rad2deg(float(r.get("pose_Ry"))) if r.get("pose_Ry") not in (None, "") else None
            pitch_deg = _rad2deg(float(r.get("pose_Rx"))) if r.get("pose_Rx") not in (None, "") else None
            roll_deg = _rad2deg(float(r.get("pose_Rz"))) if r.get("pose_Rz") not in (None, "") else None
            out: Dict[str, object] = {"file": Path(name).name,
                "yaw_deg": yaw_deg, "pitch_deg": pitch_deg, "roll_deg": roll_deg,
                "success": int(float(r.get("success", 0))) if r.get("success") not in (None, "") else 0,
                "confidence": float(r.get("confidence", 0.0)) if r.get("confidence") not in (None, "") else 0.0}
            # frame/timestamp if present
            if "frame" in r: out["frame"] = int(float(r.get("frame")))
            if "timestamp" in r: out["timestamp"] = float(r.get("timestamp"))
            # copy pose raw values
            for c in pose_cols:
                v = r.get(c)
                out[c] = float(v) if v not in (None, "") else None
            # gaze vectors/angles
            for c in gaze_cols:
                v = r.get(c)
                out[c] = float(v) if v not in (None, "") else None
            # AU intensities and classes
            for c in au_r_cols:
                v = r.get(c)
                out[c] = float(v) if v not in (None, "") else None
            for c in au_c_cols:
                v = r.get(c)
                out[c] = int(float(v)) if v not in (None, "") else None
            rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--aligned', required=True)
    ap.add_argument('--logs', required=True)
    ap.add_argument('--download', default='on')
    ap.add_argument('--binary', default='')
    ap.add_argument('--override', action='store_true')
    ap.add_argument('--img_tool', default='auto')
    ap.add_argument('--extra_args', default='')
    ap.add_argument('--help', dest='auto_help', default='on')
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    args = ap.parse_args()

    if getattr(args,'show_cli_help', False):
        print(AUTO_HELP)
    try:
        if _onoff(args.auto_help, True):
            sp = Path(__file__).resolve(); (sp.with_name(sp.stem + '.txt')).write_text(AUTO_HELP, encoding='utf-8')
    except Exception:
        pass

    aligned = Path(args.aligned)
    logs = Path(args.logs)
    out_root = logs/"openface2"; out_root.mkdir(parents=True, exist_ok=True)
    work_dir = out_root/"per_image"; work_dir.mkdir(parents=True, exist_ok=True)
    if args.override:
        for e in list(os.scandir(work_dir)):
            try:
                if e.is_file():
                    os.unlink(e.path)
                elif e.is_dir():
                    shutil.rmtree(e.path)
            except Exception:
                pass
    repo_dir = Path(os.environ.get("OF2_HOME", "/workspace/tools/OpenFace"))

    _ensure_repo(repo_dir, _onoff(args.download, True))
    env = _inject_env(repo_dir)

    images = _list_images(aligned)
    if VERBOSE:
            print(f"[OF2] images={len(images)} aligned={aligned}")
    if not images:
        if VERBOSE:
            print("[OF2] no images under --aligned")
        return 2

    prefer = "FeatureExtraction"  # use sequence tool for -fdir so we can tail a single CSV
    bin_user = Path(args.binary) if args.binary else None
    bin_path = _find_binary(repo_dir, bin_user, prefer)
    if not bin_path:
        if VERBOSE:
        print("[OF2] building OpenFace...")
        _ensure_built(repo_dir)
        bin_path = _find_binary(repo_dir, None, prefer)
    if not bin_path:
        print("[OF2] ERROR: OpenFace binary not found. Build failed or wrong platform.")
        return 3

    out_name = f"{aligned.name}.csv"
    cmd = [str(bin_path), "-fdir", str(aligned), "-out_dir", str(work_dir), "-pose", "-gaze", "-aus", "-au_static", "-of", out_name]
    if args.extra_args.strip():
        cmd += args.extra_args.strip().split()
    if VERBOSE:
        print("[OF2] running:", " ".join(cmd))

    try:
        rc = _run_with_progress(cmd, work_dir, total=len(images), env=env, aligned_name=aligned.name, force_of=out_name)
    except KeyboardInterrupt:
        print("[OF2] interrupted")
        return 130
    if rc != 0:
        print("[OF2] ERROR: tool failed")
        return rc

    csv_files = sorted(work_dir.rglob("*.csv"))
    if not csv_files:
        print("[OF2] ERROR: no CSV produced under", work_dir)
        return 4

    au_r_cols, au_c_cols, gaze_cols, pose_cols, has_frame, has_ts = _discover_fields(csv_files[0])

    rows: List[Dict[str,object]] = []
    for c in csv_files:
        try:
            rows.extend(_read_of_csv(c, images, au_r_cols, au_c_cols, gaze_cols, pose_cols))
        except Exception:
            pass

    out_csv = out_root/"openface2.csv"
    if args.override and out_csv.exists():
        try: out_csv.unlink()
        except Exception: pass

    seen = set()
    total_rows = len(rows)
    bar2 = ProgressBar("OF2-MERGE", total=total_rows, show_fail_label=True)
    t0 = time.time(); processed=0; fails=0
    write_header = not out_csv.exists()
    base_hdr = ["file"] + (["frame"] if has_frame else []) + (["timestamp"] if has_ts else []) + ["success","confidence","yaw_deg","pitch_deg","roll_deg"]
    header = base_hdr + pose_cols + gaze_cols + au_r_cols + au_c_cols
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(header)
        for r in rows:
            try:
                key = r["file"]
                if key in seen:
                    processed += 1
                    bar2.update(processed, fails=fails)
                    continue
                seen.add(key)
                row = [r.get(k) for k in header]
                wr.writerow(row)
                processed += 1
                fps = int(processed/max(1,int(time.time()-t0)))
                bar2.update(processed, fails=fails, fps=fps)
            except Exception:
                fails += 1
                processed += 1
                bar2.update(processed, fails=fails)
    bar2.close()

    meta = {
        "repo": str(repo_dir),
        "input": str(aligned),
        "out_csv": str(out_csv),
        "per_image": str(work_dir),
        "count": len(seen),
        "binary": str(bin_path),
        "pose_cols": pose_cols,
        "gaze_cols": gaze_cols,
        "au_r_cols": au_r_cols,
        "au_c_cols": au_c_cols,
        "has_frame": has_frame,
        "has_timestamp": has_ts,
    }
    (out_root/"of2_meta.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"[OF2] done rows={len(seen)} -> {out_csv}")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
