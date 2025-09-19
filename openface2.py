#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, json, argparse, subprocess, time
from pathlib import Path
from typing import List, Optional, Dict
import math

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
AUTO_HELP = r"""
--aligned DIR --logs DIR [--download on|off] [--binary PATH] [--override]
[--img_tool auto|img|seq] [--extra_args "..."] [--help on|off]
Outputs: logs/openface2/openface2.csv and metadata to logs/openface2/.
"""

try:
    from progress import ProgressBar  # type: ignore
except Exception:
    class ProgressBar:  # minimal fallback
        def __init__(self, name, total=0, show_fail_label=False): self.name=name; self.total=int(total or 0)
        def update(self, i, fails=0, fps=0):
            if self.total and ((i % 200==0) or i==self.total):
                pct=100.0*i/max(1,self.total); print(f"[{self.name}] {i}/{self.total} {pct:5.1f}% fails={fails} fps={fps}")
        def close(self): pass


def _onoff(v, default=False) -> bool:
    if isinstance(v,bool): return v
    if v is None: return bool(default)
    s=str(v).strip().lower()
    if s in {"1","on","true","yes","y"}: return True
    if s in {"0","off","false","no","n"}: return False
    return bool(default)


def _list_images(d: Path) -> List[Path]:
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _run(cmd: List[str], cwd: Optional[Path]=None, env: Optional[Dict[str,str]]=None) -> int:
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    start = time.time()
    while True:
        line = p.stdout.readline() if p.stdout else ""
        if not line and p.poll() is not None:
            break
        if line:
            print(line.rstrip())
    rc = p.wait()
    print(f"[OF2] cmd done in {time.time()-start:.1f}s rc={rc}")
    return int(rc)


def _ensure_repo(repo_dir: Path, download: bool) -> None:
    if repo_dir.exists() and (repo_dir/".git").exists():
        return
    if not download:
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git","clone","--depth","1","https://github.com/TadasBaltrusaitis/OpenFace", str(repo_dir)])
    models = repo_dir/"download_models.sh"
    if models.exists():
        _run(["bash", str(models)])


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
        _run(["bash", str(inst)], cwd=repo_dir)
        fe = _find_binary(repo_dir, None, "FeatureExtraction")
        fl = _find_binary(repo_dir, None, "FaceLandmarkImg")
        if fe or fl:
            return
    build = repo_dir/"build"
    build.mkdir(parents=True, exist_ok=True)
    _run(["cmake","-D","CMAKE_BUILD_TYPE=Release",".."], cwd=build)
    _run(["make", f"-j{max(1, os.cpu_count() or 1)}"], cwd=build)


def _inject_env(repo_dir: Path) -> Dict[str,str]:
    env = os.environ.copy()
    env["OF2_HOME"] = str(repo_dir)
    path = env.get("PATH", "")
    env["PATH"] = str(repo_dir/"build"/"bin") + os.pathsep + str(repo_dir/"bin") + os.pathsep + path
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(repo_dir/"build"/"lib") + os.pathsep + \
                               "/usr/local/lib" + os.pathsep + ld
    return env


def _rad2deg(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    try: return float(x)*180.0/math.pi
    except Exception: return None


def _read_of_csv(csv_path: Path, files: List[Path]) -> List[Dict[str,object]]:
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
            out = {
                "file": Path(name).name,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "gaze_x": gx,
                "gaze_y": gy,
                "success": int(float(r.get("success", 0))) if r.get("success") not in (None, "") else 0,
                "confidence": float(r.get("confidence", 0.0)) if r.get("confidence") not in (None, "") else 0.0,
                "AU45_r": (float(r.get("AU45_r")) if r.get("AU45_r") not in (None, "") else None),
                "AU25_r": (float(r.get("AU25_r")) if r.get("AU25_r") not in (None, "") else None),
                "AU26_r": (float(r.get("AU26_r")) if r.get("AU26_r") not in (None, "") else None),
            }
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
    repo_dir = Path(os.environ.get("OF2_HOME", "/workspace/tools/OpenFace"))

    _ensure_repo(repo_dir, _onoff(args.download, True))
    env = _inject_env(repo_dir)

    images = _list_images(aligned)
    print(f"[OF2] images={len(images)} aligned={aligned}")
    if not images:
        print("[OF2] no images under --aligned")
        return 2

    prefer = "FaceLandmarkImg" if args.img_tool in ("auto","img") else "FeatureExtraction"
    bin_user = Path(args.binary) if args.binary else None
    bin_path = _find_binary(repo_dir, bin_user, prefer)
    if (not bin_path) and prefer=="FaceLandmarkImg":
        bin_path = _find_binary(repo_dir, bin_user, "FeatureExtraction")
    if not bin_path:
        print("[OF2] building OpenFace...")
        _ensure_built(repo_dir)
        bin_path = _find_binary(repo_dir, None, prefer) or _find_binary(repo_dir, None, "FeatureExtraction")
    if not bin_path:
        print("[OF2] ERROR: OpenFace binary not found. Build failed or wrong platform.")
        return 3

    work_dir = out_root/"processed"; work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(bin_path)]
    cmd += ["-fdir", str(aligned)]
    cmd += ["-out_dir", str(work_dir), "-pose", "-gaze", "-aus"]
    if args.extra_args.strip():
        cmd += args.extra_args.strip().split()

    print("[OF2] running:", " ".join(cmd))
    rc = _run(cmd, env=env)
    if rc != 0:
        print("[OF2] ERROR: tool failed")
        return rc

    of_csv = None
    cands = list(work_dir.rglob("*.csv"))
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        of_csv = cands[0]
    if not of_csv:
        print("[OF2] ERROR: no CSV produced under", work_dir)
        return 4

    print("[OF2] parsing:", of_csv)
    rows = _read_of_csv(of_csv, images)

    out_csv = out_root/"openface2.csv"
    if args.override and out_csv.exists():
        try: out_csv.unlink()
        except Exception: pass

    seen = set()
    total = len(rows)
    bar = ProgressBar("OPENFACE2", total=total, show_fail_label=True)
    t0 = time.time(); processed=0; fails=0
    write_header = not out_csv.exists()
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(["file","yaw","pitch","roll","gaze_x","gaze_y","success","confidence","AU45_r","AU25_r","AU26_r"])
        for r in rows:
            try:
                key = r["file"]
                if key in seen: 
                    continue
                seen.add(key)
                wr.writerow([r["file"], r["yaw"], r["pitch"], r["roll"], r["gaze_x"], r["gaze_y"], r["success"], r["confidence"], r["AU45_r"], r["AU25_r"], r["AU26_r"]])
                processed += 1
                elapsed = time.time()-t0
                fps = int(processed/max(1,elapsed))
                bar.update(processed, fails=fails, fps=fps)
            except Exception as e:
                print("[OF2] row error:", e)
                fails += 1
                processed += 1
                bar.update(processed, fails=fails)
    bar.close()

    meta = {
        "repo": str(repo_dir),
        "input": str(aligned),
        "out_csv": str(out_csv),
        "source_csv": str(of_csv),
        "count": processed,
        "binary": str(bin_path),
    }
    (out_root/"of2_meta.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"[OF2] done rows={processed} fails={fails} -> {out_csv}")
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
