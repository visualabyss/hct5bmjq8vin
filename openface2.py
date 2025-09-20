#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, json, argparse, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import math
from progress import ProgressBar

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
    for p in work_dir.rglob('*.csv'):
        try:
            with p.open('r', encoding='utf-8', newline='') as f:
                n = sum(1 for _ in f)
                if n > 1:
                    total += (n - 1)
        except Exception:
            pass
    return total


def _run_with_progress(cmd: List[str], work_dir: Path, total: int, env: Optional[Dict[str,str]] = None) -> int:
    bar = ProgressBar("OPENFACE2", total=total, show_fail_label=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    start = time.time()
    last_tick = 0.0
    try:
        while True:
            line = p.stdout.readline() if p.stdout else ""
            now = time.time()
            if (now - last_tick) >= 1.0:
                processed = _count_rows(work_dir)
                fps = int(processed/max(1, int(now - start)))
                bar.update(min(processed, total), fails=0, fps=fps)
                last_tick = now
            if not line and p.poll() is not None:
                break
        rc = p.wait()
        processed = _count_rows(work_dir)
        fps = int(processed/max(1, int(time.time() - start)))
        bar.update(min(processed, total), fails=0, fps=fps)
        bar.close()
        return int(rc)
    finally:
        try: bar.close()
        except Exception: pass


def _ensure_repo(repo_dir: Path, download: bool) -> None:
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
    env["LD_LIBRARY_PATH"] = str(repo_dir/"build"/"lib") + os.pathsep + "/usr/local/lib" + os.pathsep + env.get("LD_LIBRARY_PATH","")
    env["OPENCV_LOG_LEVEL"] = "ERROR"
    return env


def _rad2deg(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    try: return float(x)*180.0/math.pi
    except Exception: return None


def _discover_fields(csv_path: Path) -> Tuple[List[str], List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        flds = rd.fieldnames or []
    au_cols = [c for c in flds if c.startswith("AU") and c.endswith("_r")]
    gaze_extra = [c for c in flds if c.startswith("gaze_") and not c.startswith("gaze_angle")]
    return au_cols, gaze_extra


def _read_of_csv(csv_path: Path, files: List[Path], au_cols: List[str], gaze_extra: List[str]) -> List[Dict[str,object]]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fns = [p.name for p in files]
        for i, r in enumerate(rd):
            name = r.get("input") or r.get("filename")
            if not name:
                name = fns[i] if i < len(fns) else str(i)
            yaw = _rad2deg(float(r.get("pose_Ry"))) if r.get("pose_Ry") not in (None, "") else None
            pitch = _rad2deg(float(r.get("pose_Rx"))) if r.get("pose_Rx") not in (None, "") else None
            roll = _rad2deg(float(r.get("pose_Rz"))) if r.get("pose_Rz") not in (None, "") else None
            gx = _rad2deg(float(r.get("gaze_angle_x"))) if r.get("gaze_angle_x") not in (None, "") else None
            gy = _rad2deg(float(r.get("gaze_angle_y"))) if r.get("gaze_angle_y") not in (None, "") else None
            out: Dict[str, object] = {
                "file": Path(name).name,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "gaze_x": gx,
                "gaze_y": gy,
                "success": int(float(r.get("success", 0))) if r.get("success") not in (None, "") else 0,
                "confidence": float(r.get("confidence", 0.0)) if r.get("confidence") not in (None, "") else 0.0,
            }
            for c in au_cols:
                v = r.get(c)
                out[c] = float(v) if v not in (None, "") else None
            for c in gaze_extra:
                v = r.get(c)
                out[c] = float(v) if v not in (None, "") else None
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
    repo_dir = Path(os.environ.get("OF2_HOME", "/workspace/tools/OpenFace"))

    _ensure_repo(repo_dir, _onoff(args.download, True))
    env = _inject_env(repo_dir)

    images = _list_images(aligned)
    print(f"[OF2] images={len(images)} aligned={aligned}")
    if not images:
        print("[OF2] no images under --aligned")
        return 2

    prefer = "FeatureExtraction"  # force CSV-streaming tool
    bin_user = Path(args.binary) if args.binary else None
    bin_path = _find_binary(repo_dir, bin_user, prefer)
    if not bin_path:
        print("[OF2] building OpenFace...")
        _ensure_built(repo_dir)
        bin_path = _find_binary(repo_dir, None, prefer)
    if not bin_path:
        print("[OF2] ERROR: OpenFace binary not found. Build failed or wrong platform.")
        return 3

    cmd = [str(bin_path), "-fdir", str(aligned), "-out_dir", str(work_dir), "-pose", "-gaze", "-aus"]
    if args.extra_args.strip():
        cmd += args.extra_args.strip().split()
    print("[OF2] running:", " ".join(cmd))

    rc = _run_with_progress(cmd, work_dir, total=len(images), env=env)
    if rc != 0:
        print("[OF2] ERROR: tool failed")
        return rc

    csv_files = sorted(work_dir.rglob("*.csv"))
    if not csv_files:
        print("[OF2] ERROR: no CSV produced under", work_dir)
        return 4

    au_cols, gaze_extra = _discover_fields(csv_files[0])

    rows: List[Dict[str,object]] = []
    for c in csv_files:
        try:
            rows.extend(_read_of_csv(c, images, au_cols, gaze_extra))
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
    base_hdr = ["file","yaw","pitch","roll","gaze_x","gaze_y","success","confidence"]
    header = base_hdr + au_cols + gaze_extra
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
                row = [r.get(k) for k in base_hdr] + [r.get(c) for c in au_cols] + [r.get(c) for c in gaze_extra]
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
        "au_cols": au_cols,
        "gaze_extra": gaze_extra,
    }
    (out_root/"of2_meta.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"[OF2] done rows={len(seen)} -> {out_csv}")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
