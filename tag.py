#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curator_tag.py
- Primary index: logs/compile/video_logs.csv (7 canonical fields)
- Merge tool outputs (OF3/MP/OSF) on top
- Write logs/manifest.jsonl and live logs/bin_table.txt
- Optional: --dedupe on (calls dedupe.py; DUPE only in bin table)
- Console: single progress row (progress.py). Warnings silenced.
- Auto-writes /workspace/scripts/curator_tag.py.txt (flags/defaults/guide)
"""
import os, sys, csv, json, time, argparse, warnings, subprocess, shlex, traceback
from pathlib import Path
from typing import Dict, Any, List

warnings.filterwarnings("ignore")
IMAGE_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}

# -------- helpers --------
def _onoff(v: str) -> bool:
    v = (v or "").strip().lower()
    if v in ("on","true","yes","y","1"): return True
    if v in ("off","false","no","n","0"): return False
    raise argparse.ArgumentTypeError("expected on/off")

def _list_images(root: Path) -> List[Path]:
    root = Path(root)
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]

def _to_float(v):
    try: return float(v)
    except: return None

def _maybe_number(v):
    if v in ("", None): return None
    try:
        if isinstance(v,(int,float)): return v
        s = str(v)
        return float(s) if ('.' in s or 'e' in s.lower()) else int(s)
    except: return v

def _write_help_txt(args: argparse.Namespace):
    out = Path("/workspace/scripts") / (Path(__file__).name + ".txt")
    defaults = {k:(str(getattr(args,k)) if isinstance(getattr(args,k), Path) else getattr(args,k))
                for k in vars(args)}
    guide = []
    guide.append("=== Purpose ===\nUse logs/compile/video_logs.csv as primary index; merge OF3/MP/OSF; write logs/manifest.jsonl.\n")
    guide.append("Live bin table at logs/bin_table.txt. No moves; warnings silenced.\n\n")
    guide.append("=== Flags & Defaults ===\n")
    for k in sorted(defaults.keys()): guide.append(f"- {k}: {defaults[k]}\n")
    guide.append("\n=== Notes ===\n- If logs/compile/video_logs.csv exists, only rows in that file are indexed.\n")
    guide.append("- Else fall back to listing images under --aligned.\n- DUPE shown only in bin_table; selection uses --allow_dupes.\n")
    out.write_text("".join(guide), encoding="utf-8")

# -------- loaders --------
def _load_csv_first(csv_dir: Path, preferred: List[str]) -> Path | None:
    for name in preferred:
        p = csv_dir / name
        if p.exists(): return p
    for p in sorted(csv_dir.glob("*.csv")): return p
    return None

def _load_compile(logs: Path) -> List[Dict[str, Any]]:
    path = logs / "compile" / "video_logs.csv"
    if not path.exists(): return []
    keep = ["file","video_id","frame_idx","sim","orig_side","up","face_frac"]
    out = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r: out.append({k: row.get(k, "") for k in keep})
    return out

def _load_openface3(logs: Path) -> Dict[str, Dict[str, Any]]:
    d = {}
    csv_path = _load_csv_first(logs / "openface3", ["openface3.csv","openface3.csvv"])
    if not csv_path: return d
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pth = (row.get("path") or "").strip()
            fil = (row.get("file") or "").strip()
            if pth: d[pth] = row
            if fil: d[fil] = row
    return d

def _load_mediapipe(logs: Path) -> Dict[str, Dict[str, Any]]:
    d = {}
    csv_path = _load_csv_first(logs / "mediapipe", ["mediapipe.csv","mp.csv"])
    if not csv_path: return d
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        key = "path" if "path" in r.fieldnames else ("image" if "image" in r.fieldnames else None)
        if not key: return d
        for row in r:
            pth = (row.get(key) or "").strip()
            fil = (row.get("file") or "").strip()
            if pth: d[pth] = row
            if fil: d[fil] = row
    return d

def _load_osf(logs: Path) -> Dict[str, Dict[str, Any]]:
    d = {}
    csv_path = _load_csv_first(logs / "openseeface", ["openseeface.csv","osf.csv"])
    if not csv_path: return d
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        key = "path" if "path" in r.fieldnames else ("image" if "image" in r.fieldnames else None)
        if not key: return d
        for row in r:
            pth = (row.get(key) or "").strip()
            fil = (row.get("file") or "").strip()
            if pth: d[pth] = row
            if fil: d[fil] = row
    return d

def _load_dedupe(logs: Path) -> Dict[str, Any]:
    d = {}
    j = logs / "curator_dedupe" / "decisions.jsonl"
    if j.exists():
        for line in j.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                p = str(obj.get("path") or "")
                if p: d[p] = obj
            except: pass
        return d
    c = logs / "curator_dedupe" / "decisions.csv"
    if c.exists():
        with c.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                p = row.get("path") or ""
                if p: d[p] = row
    return d

def _get_row_any(tbl: dict, aligned: Path, img: Path):
    p_abs = str(img)
    p_join = str(aligned / img.name)
    bname = img.name
    return tbl.get(p_abs) or tbl.get(p_join) or tbl.get(bname)

# -------- binning helpers --------
def _cat_gaze(yaw, pitch):
    if yaw is None or pitch is None: return "FRONT"
    if yaw <= -0.35: return "LEFT"
    if yaw >=  0.35: return "RIGHT"
    if pitch <= -0.25: return "UP"
    if pitch >=  0.25: return "DOWN"
    return "FRONT"

def _cat_eyes(of3_row, mp_row):
    p = _maybe_number((mp_row or {}).get("eyes_open_prob"))
    if isinstance(p, (int,float)):
        if p >= 0.7: return "OPEN"
        if p >= 0.35: return "HALF"
        return "CLOSED"
    v = (of3_row or {}).get("eyes","") or ""
    v = v.strip().upper()
    return v if v in ("OPEN","HALF","CLOSED") else "OPEN"

def _cat_mouth(of3_row):
    a25 = _to_float(of3_row.get("au25")) if of3_row else None
    a26 = _to_float(of3_row.get("au26")) if of3_row else None
    mv = max([v for v in [a25,a26] if v is not None], default=0.0)
    if mv >= 0.80: return "WIDE"
    if mv >= 0.40: return "OPEN"
    if mv >= 0.10: return "SLIGHT"
    return "CLOSE"

def _cat_smile(of3_row):
    if not of3_row: return "NONE"
    if (of3_row.get("teeth") or "").lower() == "likely": return "TEETH"
    if (of3_row.get("mouth") or "").lower() == "closed": return "CLOSED"
    a12 = _to_float(of3_row.get("au12"))
    return "NONE" if (a12 is None or a12 < 0.5) else "TEETH" if (of3_row.get("teeth") or "").lower()=="likely" else "NONE"

def _cat_yaw(yaw):
    if yaw is None: return "FRONT"
    if yaw <= -0.35: return "STRONG_L"
    if yaw >=  0.35: return "STRONG_R"
    if yaw <= -0.10: return "SLIGHT_L"
    if yaw >=  0.10: return "SLIGHT_R"
    return "FRONT"

def _cat_pitch(pitch):
    if pitch is None: return "FRONT"
    if pitch <= -0.15: return "UP"
    if pitch >=  0.20: return "DOWN"
    return "FRONT"

# -------- main --------
def main():
    ap = argparse.ArgumentParser("curator_tag: compile-indexed tagging, live bin table, no moves")
    ap.add_argument("--aligned", required=True, type=Path)
    ap.add_argument("--logs",    required=True, type=Path)

    ap.add_argument("--openface3", choices=["on","off"], default="on")
    ap.add_argument("--mediapipe", choices=["on","off"], default="on")
    ap.add_argument("--openseeface", choices=["on","off"], default="off")
    ap.add_argument("--arcface", choices=["on","off"], default="off")
    ap.add_argument("--magface", choices=["on","off"], default="off")
    ap.add_argument("--dedupe", choices=["on","off"], default="off")

    ap.add_argument("--allow_dupes", choices=["on","off"], default="off")
    ap.add_argument("--use_list", type=Path, default=None)
    ap.add_argument("--override", choices=["on","off"], default="on")
    args = ap.parse_args()
    _write_help_txt(args)

    aligned = args.aligned; logs = args.logs

    # optional dedupe run (mute output)
    if _onoff(args.dedupe):
        env = dict(os.environ); env["PROGRESS_MUTE"] = "1"
        cmd = (
            f"python3 -u /workspace/scripts/dedupe.py "
            f"--aligned {shlex.quote(str(aligned))} --logs {shlex.quote(str(logs))}"
        )
        subprocess.run(cmd + " >/dev/null 2>&1", shell=True, env=env)

    # load compile index (primary)
    compile_rows = _load_compile(logs)
    use_compile = len(compile_rows) > 0

    # load tools
    of3 = _load_openface3(logs) if _onoff(args.openface3) else {}
    mp  = _load_mediapipe(logs) if _onoff(args.mediapipe) else {}
    osf = _load_osf(logs)       if _onoff(args.openseeface) else {}
    dec = _load_dedupe(logs)
    any_tools_have_rows = ((args.openface3=='on' and len(of3)>0) or (args.mediapipe=='on' and len(mp)>0) or (args.openseeface=='on' and len(osf)>0))
    # --- debug snapshot of loader sizes (silent) ---
    try:
        _ensure_dir(logs / "curator_tag")
        _dbg = {
            "use_compile": use_compile,
            "compile_rows": len(compile_rows),
            "total_base_paths": 0,  # filled below
            "of3_rows": len(of3),
            "mp_rows": len(mp),
            "osf_rows": len(osf),
            "dedupe_rows": len(dec),
            "of3_sample_keys": list(of3.keys())[:5],
            "mp_sample_keys": list(mp.keys())[:5],
            "osf_sample_keys": list(osf.keys())[:5],
        }
        (logs / "curator_tag" / "debug_loaders.json").write_text("{}", encoding="utf-8")
    except Exception:
        pass


    # index images
    if use_compile:
        base_paths = [str(aligned / row["file"]) for row in compile_rows]
    else:
        base_paths = [str(p) for p in _list_images(aligned)]

    total = len(base_paths)
    # fill base paths count into debug file
    try:
        _dbg["total_base_paths"] = len(base_paths)
        (logs / "curator_tag" / "debug_loaders.json").write_text(__import__("json").dumps(_dbg, indent=2), encoding="utf-8")
    except Exception:
        pass

    manifest_path = logs / "manifest.jsonl"
    if manifest_path.exists() and args.override == "off":
        return 0
    manifest_path.unlink(missing_ok=True)
    mf = manifest_path.open("w", encoding="utf-8")

    # progress row (single)
    try:
        from progress import ProgressBar
        bar = ProgressBar("TAG", total=total, show_fail_label=True)
        # force header once even if small/fast
        bar.update(0, 0)
    except Exception:
        bar = None

    from bin_table import update_bin_table

    # bins
    gaze_bins  = {k:0 for k in ("FRONT","LEFT","RIGHT","UP","DOWN")}
    eyes_bins  = {k:0 for k in ("OPEN","HALF","CLOSED")}
    mouth_bins = {k:0 for k in ("WIDE","OPEN","SLIGHT","CLOSE")}
    smile_bins = {k:0 for k in ("TEETH","CLOSED","NONE")}
    yaw_bins   = {k:0 for k in ("FRONT","SLIGHT_L","SLIGHT_R","STRONG_L","STRONG_R")}
    pitch_bins = {k:0 for k in ("FRONT","UP","DOWN")}
    temp_bins  = {k:0 for k in ("WARM","NEUTRAL","COOL")}
    expo_bins  = {k:0 for k in ("UNDER","NORMAL","OVER")}

    dupe_n = 0
    fails = 0
    t_start = time.time()
    whitelist = set()
    if args.use_list and args.use_list.exists():
        whitelist = set(x.strip() for x in args.use_list.read_text(encoding="utf-8").splitlines() if x.strip())
    compile_by_file = {row["file"]: row for row in compile_rows}

    for i, p in enumerate(base_paths, 1):
        img = Path(p)
        rec: Dict[str, Any] = {"path": p, "file": img.name}

        if use_compile:
            cr = compile_by_file.get(img.name, {})
            if cr:
                rec["video_id"]  = cr.get("video_id","")
                rec["frame_idx"] = _maybe_number(cr.get("frame_idx"))
                rec["sim"]       = _maybe_number(cr.get("sim"))
                rec["orig_side"] = _maybe_number(cr.get("orig_side"))
                rec["up"]        = _maybe_number(cr.get("up"))
                rec["face_frac"] = _maybe_number(cr.get("face_frac"))

        row_of3 = _get_row_any(of3, aligned, img)
        row_mp  = _get_row_any(mp, aligned, img)
        _row_osf= _get_row_any(osf, aligned, img)

        # FAIL only when no selected tool has a row
        if any_tools_have_rows:
            if (row_of3 is None) and (row_mp is None) and (_row_osf is None):
                fails += 1

        if row_of3:
            yaw = _to_float(row_of3.get("pose_yaw"))
            pit = _to_float(row_of3.get("pose_pitch"))
            rec["emotion"] = row_of3.get("emotion","")
            rec["emotion_conf"] = _to_float(row_of3.get("emotion_conf"))
            rec["pose_yaw"] = yaw
            rec["pose_pitch"] = pit

            # bins
            gaze_bins[_cat_gaze(yaw, pit)] += 1
            peyes = _maybe_number((row_mp or {}).get("eyes_open_prob"))
            if isinstance(peyes,(int,float)):
                eyes_bins["OPEN" if peyes>=0.7 else "HALF" if peyes>=0.35 else "CLOSED"] += 1
            else:
                e = (row_of3.get("eyes") or "").strip().upper()
                eyes_bins[e if e in ("OPEN","HALF","CLOSED") else "OPEN"] += 1
            a25 = _to_float(row_of3.get("au25")); a26 = _to_float(row_of3.get("au26"))
            mv = max([v for v in [a25,a26] if v is not None], default=0.0)
            mouth_bins["WIDE" if mv>=0.80 else "OPEN" if mv>=0.40 else "SLIGHT" if mv>=0.10 else "CLOSE"] += 1
            a12 = _to_float(row_of3.get("au12"))
            if (row_of3.get("teeth") or "").lower()=="likely": smile_bins["TEETH"] += 1
            elif (row_of3.get("mouth") or "").lower()=="closed": smile_bins["CLOSED"] += 1
            elif (a12 is not None and a12 >= 0.5): smile_bins["TEETH"] += 1
            else: smile_bins["NONE"]  += 1
            yaw_bins[_cat_yaw(yaw)] += 1
            pitch_bins[_cat_pitch(pit)] += 1

        # dedupe decision (advisory)
        d = dec.get(str(img)) or dec.get(str(aligned / img.name)) or dec.get(img.name) or {}
        is_dupe = False
        if "is_dupe" in d:
            is_dupe = bool(d["is_dupe"]) if isinstance(d["is_dupe"], bool) else (str(d["is_dupe"]).lower() in ("1","true","yes","on"))
        elif "keep" in d:
            is_dupe = not (str(d["keep"]).lower() in ("1","true","yes","on"))
        dupe_n += int(is_dupe)

        # final 'use'
        use = True
        if whitelist: use = (str(img) in whitelist or img.name in whitelist)
        if (args.allow_dupes == "off") and is_dupe: use = False
        rec["use"] = use

        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # update live table + progress
        now = time.time()
        fps = int(i / max(1e-6, (now - t_start)))
        remain = max(0, total - i)
        eta_s = int(remain / max(1, fps)) if fps>0 else 0
        eta = f"{eta_s//60:02d}:{eta_s%60:02d}"

        from bin_table import update_bin_table
        update_bin_table(logs, {
            "total": total, "processed": i, "fails": fails, "dupe": dupe_n, "fps": fps, "eta": f"00:{eta}",
            "gaze": gaze_bins, "eyes": eyes_bins, "mouth": mouth_bins, "smile": smile_bins,
            "yaw": yaw_bins, "pitch": pitch_bins, "temp": temp_bins, "expo": expo_bins
        })
        if bar: bar.update(i, fails)

    if bar:
        try: bar.close()
        except Exception: pass
    mf.close()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        print("FATAL:\n" + traceback.format_exc()); sys.exit(1)


# === FUSION: tool ingestion helpers (OF3, MP, OSF, ArcFace, MagFace) ===
import math
import csv

def _read_csv_safe(p, low_memory=True):
    """Return (DataFrame or None). Supports csv without pandas fallback."""
    try:
        import pandas as pd
    except Exception:
        pd = None
    p = Path(p)
    if not p.exists():
        return None
    if pd is None:
        # minimal csv reader to avoid pandas dependency breakage
        with p.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return None
        cols = rows[0]
        data = rows[1:]
        # build dict of columns
        cols_norm = [c.strip() for c in cols]
        col_ix = {i:c for i,c in enumerate(cols_norm)}
        # convert to simple list of dicts
        dd = [{col_ix[i]: v for i, v in enumerate(r)} for r in data]
        # very small faux-DF
        class _Mini:
            def __init__(self, rows, cols): self._rows, self.columns = rows, cols
            def __len__(self): return len(self._rows)
            def __iter__(self): return iter(self._rows)
            def get(self, k, d=None): return [r.get(k, d) for r in self._rows]
        return _Mini(dd, cols_norm)
    else:
        try:
            return pd.read_csv(p, low_memory=low_memory)
        except Exception:
            return None

def _pick(df, names):
    for n in names:
        if n in getattr(df, "columns", []): return n
    return None

def _to_float_series(s):
    import numpy as _np
    try:
        return s.astype("float32")
    except Exception:
        try:
            return _np.asarray([float(x) if x not in ("", None) else _np.nan for x in s], dtype="float32")
        except Exception:
            return _np.full((len(s),), _np.nan, dtype="float32")

def _maybe_rad_to_deg(v):
    """If typical magnitude < 3.2, assume radians → deg."""
    import numpy as _np
    arr = _np.asarray(v, dtype="float32")
    m = _np.nanmedian(_np.abs(arr))
    if _np.isnan(m): return arr
    return arr * (180.0/_np.pi) if m<3.2 else arr

def _norm_openface3(df):
    """Return dict with keys: path, of_yaw, of_pitch, of_roll, of_conf, of_gazex, of_gazey, of_eopen_l, of_eopen_r, of_au25, of_au26"""
    if df is None: return None
    cols = getattr(df, "columns", [])
    name_path = _pick(df, ["path","file","image","fname","name"])
    if name_path is None: return None
    # pose (attempt multiple variants)
    y = _pick(df, ["pose_yaw_deg","head_pose_yaw","yaw","pose_Ry","pose_yaw"])
    p = _pick(df, ["pose_pitch_deg","head_pose_pitch","pitch","pose_Rx","pose_pitch"])
    r = _pick(df, ["pose_roll_deg","head_pose_roll","roll","pose_Rz","pose_roll"])
    # gaze (OpenFace often exposes gaze_angle_x/y OR gaze_h,y)
    gx = _pick(df, ["gaze_angle_x","gaze_h","gaze_x"])
    gy = _pick(df, ["gaze_angle_y","gaze_v","gaze_y"])
    # eye openness / AUs
    eol = _pick(df, ["eye_l_opening","eye_open_l","eye_open_left"])
    eor = _pick(df, ["eye_r_opening","eye_open_r","eye_open_right"])
    au25 = _pick(df, ["AU25_r","au25","mouth_open"])
    au26 = _pick(df, ["AU26_r","au26","jaw_drop"])
    conf = _pick(df, ["confidence","of_conf","landmark_conf"])

    out = {}
    out["path"] = df[name_path]
    import numpy as _np
    out["of_yaw"]   = _maybe_rad_to_deg(_to_float_series(df[y])) if y else _np.full((len(df),), _np.nan, "float32")
    out["of_pitch"] = _maybe_rad_to_deg(_to_float_series(df[p])) if p else _np.full((len(df),), _np.nan, "float32")
    out["of_roll"]  = _maybe_rad_to_deg(_to_float_series(df[r])) if r else _np.full((len(df),), _np.nan, "float32")
    out["of_gazex"] = _to_float_series(df[gx]) if gx else _np.full((len(df),), _np.nan, "float32")
    out["of_gazey"] = _to_float_series(df[gy]) if gy else _np.full((len(df),), _np.nan, "float32")
    out["of_eopen_l"] = _to_float_series(df[eol]) if eol else _np.full((len(df),), _np.nan, "float32")
    out["of_eopen_r"] = _to_float_series(df[eor]) if eor else _np.full((len(df),), _np.nan, "float32")
    out["of_au25"]    = _to_float_series(df[au25]) if au25 else _np.full((len(df),), _np.nan, "float32")
    out["of_au26"]    = _to_float_series(df[au26]) if au26 else _np.full((len(df),), _np.nan, "float32")
    out["of_conf"]    = _to_float_series(df[conf]) if conf else _np.ones((len(df),), dtype="float32")
    return out

def _norm_mediapipe(df):
    """Return dict with keys: path, mp_yaw, mp_pitch, mp_roll, mp_conf, plus blendshapes used later."""
    if df is None: return None
    name_path = _pick(df, ["path","file","image","fname","name"])
    if name_path is None: return None
    # Some exports include 4x4 facial transform columns (rvec/tvec or matrix); here we rely on precomputed yaw/pitch/roll if present.
    yaw = _pick(df, ["mp_yaw_deg","pose_yaw_deg","yaw"])
    pit = _pick(df, ["mp_pitch_deg","pose_pitch_deg","pitch"])
    rol = _pick(df, ["mp_roll_deg","pose_roll_deg","roll"])
    conf = _pick(df, ["mp_conf","confidence","landmark_conf","presence"])

    # blendshapes
    bs = {}
    for k in [
        "eyeBlinkLeft","eyeBlinkRight","eyeSquintLeft","eyeSquintRight",
        "eyeLookUpLeft","eyeLookUpRight","eyeLookDownLeft","eyeLookDownRight",
        "eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight",
        "jawOpen","mouthOpen","mouthClose",
        "mouthSmileLeft","mouthSmileRight",
        "mouthFrownLeft","mouthFrownRight",
        "mouthPucker","mouthUpperUpLeft","mouthUpperUpRight","mouthLowerDownLeft","mouthLowerDownRight"
    ]:
        c = _pick(df, [k, f"bs_{k}", k.lower()])
        if c: bs[k] = c

    import numpy as _np
    out = {
        "path": df[name_path],
        "mp_yaw":   _maybe_rad_to_deg(_to_float_series(df[yaw])) if yaw else _np.full((len(df),), _np.nan, "float32"),
        "mp_pitch": _maybe_rad_to_deg(_to_float_series(df[pit])) if pit else _np.full((len(df),), _np.nan, "float32"),
        "mp_roll":  _maybe_rad_to_deg(_to_float_series(df[rol])) if rol else _np.full((len(df),), _np.nan, "float32"),
        "mp_conf":  _to_float_series(df[conf]) if conf else _np.ones((len(df),), dtype="float32"),
    }
    for k,c in bs.items():
        out[f"mp_{k}"] = _to_float_series(df[c])
    return out

def _norm_osf(df):
    """Return dict with keys: path, osf_yaw, osf_pitch, osf_roll, osf_conf, osf_eopen_l, osf_eopen_r."""
    if df is None: return None
    name_path = _pick(df, ["path","file","image","fname","name"])
    if name_path is None: return None
    yaw = _pick(df, ["yaw","pose_yaw_deg","pose_yaw"])
    pit = _pick(df, ["pitch","pose_pitch_deg","pose_pitch"])
    rol = _pick(df, ["roll","pose_roll_deg","pose_roll"])
    conf = _pick(df, ["confidence","osf_conf"])
    eol = _pick(df, ["eye_open_left","eye_l_open","left_eye_opening"])
    eor = _pick(df, ["eye_open_right","eye_r_open","right_eye_opening"])
    import numpy as _np
    out = {
        "path": df[name_path],
        "osf_yaw":   _maybe_rad_to_deg(_to_float_series(df[yaw])) if yaw else _np.full((len(df),), _np.nan, "float32"),
        "osf_pitch": _maybe_rad_to_deg(_to_float_series(df[pit])) if pit else _np.full((len(df),), _np.nan, "float32"),
        "osf_roll":  _maybe_rad_to_deg(_to_float_series(df[rol])) if rol else _np.full((len(df),), _np.nan, "float32"),
        "osf_conf":  _to_float_series(df[conf]) if conf else _np.ones((len(df),), dtype="float32"),
        "osf_eopen_l": _to_float_series(df[eol]) if eol else _np.full((len(df),), _np.nan, "float32"),
        "osf_eopen_r": _to_float_series(df[eor]) if eor else _np.full((len(df),), _np.nan, "float32"),
    }
    return out

def _norm_arcface(df):
    if df is None: return None
    name_path = _pick(df, ["path","file","image"])
    score = _pick(df, ["id_score","score","cosine","sim"])
    if name_path is None or score is None: return None
    return {"path": df[name_path], "arcface_id_score": _to_float_series(df[score])}

def _norm_magface(df):
    if df is None: return None
    name_path = _pick(df, ["path","file","image"])
    q = _pick(df, ["quality","magface_quality","magface_q"])
    if name_path is None or q is None: return None
    return {"path": df[name_path], "magface_quality": _to_float_series(df[q])}

def _to_df(d):
    """dict-of-series -> pandas DataFrame safely"""
    try:
        import pandas as pd
    except Exception:
        return None
    if d is None: return None
    return pd.DataFrame(d)

def load_tools_to_frames(logs_root):
    """
    Read all available tool csvs under logs_root and return a merged DataFrame on 'path'.
    Missing sources are tolerated.
    """
    logs_root = Path(logs_root)
    of = _read_csv_safe(logs_root / "openface3" / "openface3.csv")
    mp = _read_csv_safe(logs_root / "mediapipe" / "mp_tags.csv")
    osf= _read_csv_safe(logs_root / "openseeface" / "osf_tags.csv")
    af = _read_csv_safe(logs_root / "arcface" / "arcface.csv")
    mf = _read_csv_safe(logs_root / "magface" / "magface.csv")

    ofn = _to_df(_norm_openface3(of))
    mpn = _to_df(_norm_mediapipe(mp))
    osfn= _to_df(_norm_osf(osf))
    afn = _to_df(_norm_arcface(af))
    mfn = _to_df(_norm_magface(mf))

    try:
        import pandas as pd
    except Exception:
        return None

    dfs = [x for x in [ofn, mpn, osfn, afn, mfn] if x is not None]
    if not dfs:
        return None

    # progressive outer-merge on 'path'
    from functools import reduce
    def _merge(a,b): return a.merge(b, on="path", how="outer")
    out = reduce(_merge, dfs)
    # ensure 'path' is filename only (strip dirs if needed)
    out["path"] = out["path"].astype(str).apply(lambda s: os.path.basename(s))
    return out


# === FUSION: confidence-voted merge + binning (OF3 + MP + OSF + ArcFace + MagFace) ===
import math, csv, json, time
try:
    from progress import ProgressBar
except Exception:
    class ProgressBar:
        def __init__(self, *a, **k): self.total = k.get("total", 1)
        def update(self, *a, **k): pass
        def close(self): pass

def _wmedian(vals, weights):
    """Weighted median; vals/weights are finite lists of equal len."""
    import numpy as _np
    v = _np.asarray(vals, dtype="float32")
    w = _np.asarray(weights, dtype="float32")
    m = _np.isfinite(v) & _np.isfinite(w) & (w>0)
    if not _np.any(m): return _np.nan
    v, w = v[m], w[m]
    order = _np.argsort(v)
    v, w = v[order], w[order]
    cw = _np.cumsum(w)
    mid = 0.5 * w.sum()
    idx = _np.searchsorted(cw, mid)
    return float(v[min(idx, len(v)-1)])

def _bin_yaw(y):
    try:
        y = float(y)
    except Exception:
        return "UNK"
    if y < -20: return "L"
    if y <  -8: return "LC"
    if y <=   8: return "C"
    if y <=  20: return "RC"
    return "R"

def _bin_pitch(p):
    try:
        p = float(p)
    except Exception:
        return "UNK"
    if p < -12: return "DOWN"
    if p >  12: return "UP"
    return "MID"

def _eyes_from_sources(mp_blinks, of_open, osf_open, qhat):
    """
    mp_blinks: (blinkL, blinkR, squintL, squintR) in [0,1] if present else None
    of_open:   (openL, openR) arbitrary units (we auto-calibrate via percentiles outside)
    osf_open:  (openL, openR)
    """
    # Prefer MediaPipe: blink close ~1; open ~0
    if mp_blinks is not None:
        bl, br, sl, sr = mp_blinks
        b = float((bl + br)/2.0)
        # use squint as modifier to promote HALF
        sq = float((sl + sr)/2.0) if (sl is not None and sr is not None) else 0.0
        if b >= 0.60*qhat: return "CLOSED"
        if b >= 0.35*qhat or sq >= 0.45*qhat: return "HALF"
        return "OPEN"
    # Fallback to OF/OSF using calibrated openness later; placeholder here
    ol = of_open[0] if of_open is not None else None
    or_ = of_open[1] if of_open is not None else None
    base = None
    if ol is not None and or_ is not None:
        base = (float(ol) + float(or_))/2.0
    else:
        if osf_open is not None:
            ol = osf_open[0]; or_ = osf_open[1]
            if ol is not None and or_ is not None:
                base = (float(ol) + float(or_))/2.0
    if base is None or not math.isfinite(base):
        return "OPEN"  # conservative
    # thresholds are percentile-calibrated outside; here we assume normalized 0..1
    if base <= 0.20: return "CLOSED"
    if base <= 0.45: return "HALF"
    return "OPEN"

def _mouth_from_sources(mp_jaw, mp_smiles, of_au25, of_au26):
    # Prefer MP jawOpen if present
    if mp_jaw is not None and math.isfinite(mp_jaw):
        j = float(mp_jaw)
        if j < 0.15: return "CLOSE"
        if j < 0.35: return "SLIGHT"
        if j < 0.60: return "OPEN"
        return "WIDE"
    # Fallback to OF AUs
    au25 = of_au25 if (of_au25 is not None and math.isfinite(of_au25)) else 0.0
    au26 = of_au26 if (of_au26 is not None and math.isfinite(of_au26)) else 0.0
    if au26 >= 0.6: return "WIDE"
    if au25 >= 0.35: return "OPEN"
    if au25 >= 0.15: return "SLIGHT"
    return "CLOSE"

def _gaze_from_sources(of_gx, of_gy, mp_look):
    # Use OF gaze if available (degrees)
    if of_gx is not None and math.isfinite(of_gx) and of_gy is not None and math.isfinite(of_gy):
        gh = "L" if of_gx < -10 else ("R" if of_gx > 10 else "C")
        gv = "D" if of_gy < -8  else ("U" if of_gy > 8  else "C")
        return gh, gv
    # Fallback to MP eyeLook* (argmax)
    if mp_look is not None:
        lin, rin, lout, rout = mp_look   # sums of in/out components
        if all(math.isfinite(x) for x in (lin,rin,lout,rout)):
            h = "L" if (lout+rout)>(lin+rin) else ("R" if (lin+rin)>(lout+rout) else "C")
            # vertical using up/down (approx)
            return h, "C"
    return "C","C"

def _norm01(x, lo, hi):
    if not math.isfinite(x) or not math.isfinite(lo) or not math.isfinite(hi) or hi<=lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return 0.0 if v<0 else (1.0 if v>1 else float(v))

def fuse_and_write_tags(logs_root, out_dir=None, id_thresh=None, mag_floor=None,
                        of_w=1.0, mp_w=0.8, osf_w=0.6):
    """
    - reads all five tools via load_tools_to_frames()
    - computes final pose (weighted median), eyes/mouth/gaze states, id/quality flags
    - writes logs/curator/tags.csv (and bin_table.txt)
    """
    logs_root = Path(logs_root)
    out_dir = Path(out_dir or (logs_root / "curator"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "tags.csv"
    bin_table = out_dir / "bin_table.txt"

    from datetime import datetime
    import numpy as np
    df = load_tools_to_frames(str(logs_root))
    if df is None or len(df)==0:
        # Write header-only to remain consistent
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow([
                "path","pose_yaw_deg","pose_pitch_deg","pose_roll_deg",
                "eyes_state","mouth_state","gaze_h","gaze_v",
                "arcface_id_score","magface_quality",
                "wrong_id","too_blurry","bin_yaw","bin_pitch","bin_score"
            ])
        return str(out_csv)

    # Auto-calibrate thresholds from available columns
    af = df.get("arcface_id_score")
    mf = df.get("magface_quality")
    nan = np.nan
    if af is None: 
        af = np.full((len(df),), nan, dtype="float32")
    if mf is None:
        mf = np.full((len(df),), nan, dtype="float32")

    # Compute dataset percentiles for normalization
    def _npnanpercentile(a, p, dflt):
        try:
            v = float(np.nanpercentile(a, p))
            return v if math.isfinite(v) else dflt
        except Exception:
            return dflt

    id_p05 = _npnanpercentile(af, 5, 0.0)
    id_p95 = _npnanpercentile(af,95, 1.0)
    q_p05  = _npnanpercentile(mf, 5,  5.0)
    q_p95  = _npnanpercentile(mf,95, 40.0)

    T_id  = float(id_thresh) if id_thresh is not None else id_p05  # conservative
    Q_min = float(mag_floor) if mag_floor is not None else q_p05

    # Calibrations for OF/OSF openness → normalize to 0..1 using dataset percentiles
    e_of = []
    if "of_eopen_l" in df.columns and "of_eopen_r" in df.columns:
        earr = np.nanmean(np.stack([df["of_eopen_l"].values, df["of_eopen_r"].values], axis=1), axis=1)
        e_of = earr
        e_lo = _npnanpercentile(earr, 10, np.nanmin(earr) if len(earr) else 0.0)
        e_hi = _npnanpercentile(earr, 90, np.nanmax(earr) if len(earr) else 1.0)
    else:
        e_lo, e_hi = 0.0, 1.0

    e_osf = []
    if "osf_eopen_l" in df.columns and "osf_eopen_r" in df.columns:
        earr = np.nanmean(np.stack([df["osf_eopen_l"].values, df["osf_eopen_r"].values], axis=1), axis=1)
        e_osf = earr
        s_lo = _npnanpercentile(earr, 10, np.nanmin(earr) if len(earr) else 0.0)
        s_hi = _npnanpercentile(earr, 90, np.nanmax(earr) if len(earr) else 1.0)
    else:
        s_lo, s_hi = 0.0, 1.0

    total = len(df)
    bar = ProgressBar("CUR-TAG", total=max(1,total))
    t0 = time.time(); done = 0

    # Open outputs
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path","pose_yaw_deg","pose_pitch_deg","pose_roll_deg",
            "eyes_state","mouth_state","gaze_h","gaze_v",
            "arcface_id_score","magface_quality",
            "wrong_id","too_blurry","bin_yaw","bin_pitch","bin_score"
        ])

        for i, row in df.iterrows():
            pth = str(row["path"])
            # Weights with confidences and quality factor qhat (rowwise)
            q = float(row["magface_quality"]) if "magface_quality" in row and math.isfinite(row["magface_quality"]) else float("nan")
            qhat = _norm01(q, q_p05, q_p95) if math.isfinite(q) else 0.7  # default mid

            of_conf  = float(row["of_conf"])  if "of_conf"  in row and math.isfinite(row["of_conf"])  else 1.0
            mp_conf  = float(row["mp_conf"])  if "mp_conf"  in row and math.isfinite(row["mp_conf"])  else 1.0
            osf_conf = float(row["osf_conf"]) if "osf_conf" in row and math.isfinite(row["osf_conf"]) else 1.0

            # Pose voting
            of_yaw   = float(row["of_yaw"])   if "of_yaw"   in row and math.isfinite(row["of_yaw"])   else float("nan")
            mp_yaw   = float(row["mp_yaw"])   if "mp_yaw"   in row and math.isfinite(row["mp_yaw"])   else float("nan")
            osf_yaw  = float(row["osf_yaw"])  if "osf_yaw"  in row and math.isfinite(row["osf_yaw"])  else float("nan")

            of_pitch = float(row["of_pitch"]) if "of_pitch" in row and math.isfinite(row["of_pitch"]) else float("nan")
            mp_pitch = float(row["mp_pitch"]) if "mp_pitch" in row and math.isfinite(row["mp_pitch"]) else float("nan")
            osf_pitch= float(row["osf_pitch"])if "osf_pitch"in row and math.isfinite(row["osf_pitch"])else float("nan")

            of_roll  = float(row["of_roll"])  if "of_roll"  in row and math.isfinite(row["of_roll"])  else float("nan")
            mp_roll  = float(row["mp_roll"])  if "mp_roll"  in row and math.isfinite(row["mp_roll"])  else float("nan")
            osf_roll = float(row["osf_roll"]) if "osf_roll" in row and math.isfinite(row["osf_roll"]) else float("nan")

            wy = _wmedian([of_yaw, mp_yaw, osf_yaw],  [of_w*of_conf*qhat, mp_w*mp_conf*qhat, osf_w*osf_conf*qhat])
            wp = _wmedian([of_pitch, mp_pitch, osf_pitch],[of_w*of_conf*qhat, mp_w*mp_conf*qhat, osf_w*osf_conf*qhat])
            wr = _wmedian([of_roll, mp_roll, osf_roll],  [of_w*of_conf*qhat, mp_w*mp_conf*qhat, osf_w*osf_conf*qhat])

            # Eyes
            mp_blinks = None
            if "mp_eyeBlinkLeft" in df.columns and "mp_eyeBlinkRight" in df.columns:
                bl = row.get("mp_eyeBlinkLeft"); br = row.get("mp_eyeBlinkRight")
                sl = row.get("mp_eyeSquintLeft", float("nan"))
                sr = row.get("mp_eyeSquintRight", float("nan"))
                if math.isfinite(bl) and math.isfinite(br):
                    mp_blinks = (float(bl), float(br), float(sl) if math.isfinite(sl) else 0.0, float(sr) if math.isfinite(sr) else 0.0)

            def _norm(x, lo, hi):
                try:
                    return (float(x)-lo)/(hi-lo+1e-6)
                except Exception:
                    return float("nan")

            of_open = None
            if "of_eopen_l" in df.columns and "of_eopen_r" in df.columns:
                ol = _norm(row.get("of_eopen_l"), e_lo, e_hi)
                or_ = _norm(row.get("of_eopen_r"), e_lo, e_hi)
                if math.isfinite(ol) and math.isfinite(or_):
                    of_open = (ol, or_)

            osf_open = None
            if "osf_eopen_l" in df.columns and "osf_eopen_r" in df.columns:
                ol = _norm(row.get("osf_eopen_l"), s_lo, s_hi)
                or_ = _norm(row.get("osf_eopen_r"), s_lo, s_hi)
                if math.isfinite(ol) and math.isfinite(or_):
                    osf_open = (ol, or_)

            eyes = _eyes_from_sources(mp_blinks, of_open, osf_open, qhat)

            # Mouth
            mp_jaw = row.get("mp_jawOpen") if "mp_jawOpen" in df.columns else row.get("mp_jawopen") if "mp_jawopen" in df.columns else row.get("mp_jaw_open") if "mp_jaw_open" in df.columns else None
            mp_jaw = float(mp_jaw) if (mp_jaw is not None and math.isfinite(mp_jaw)) else None
            mouth = _mouth_from_sources(mp_jaw,
                                        (row.get("mp_mouthSmileLeft"), row.get("mp_mouthSmileRight")) if "mp_mouthSmileLeft" in df.columns else None,
                                        row.get("of_au25") if "of_au25" in df.columns else None,
                                        row.get("of_au26") if "of_au26" in df.columns else None)

            # Gaze
            of_gx = float(row.get("of_gazex")) if ("of_gazex" in df.columns and math.isfinite(row.get("of_gazex"))) else None
            of_gy = float(row.get("of_gazey")) if ("of_gazey" in df.columns and math.isfinite(row.get("of_gazey"))) else None
            mp_look = None
            keys = ["mp_eyeLookInLeft","mp_eyeLookInRight","mp_eyeLookOutLeft","mp_eyeLookOutRight"]
            if all(k in df.columns for k in keys):
                lin, rin, lout, rout = [row.get(k) for k in keys]
                mp_look = (float(lin) if math.isfinite(lin) else 0.0,
                           float(rin) if math.isfinite(rin) else 0.0,
                           float(lout) if math.isfinite(lout) else 0.0,
                           float(rout) if math.isfinite(rout) else 0.0)
            gh, gv = _gaze_from_sources(of_gx, of_gy, mp_look)

            # Identity & quality
            afs = float(row["arcface_id_score"]) if "arcface_id_score" in row and math.isfinite(row["arcface_id_score"]) else float("nan")
            mqs = float(row["magface_quality"])  if "magface_quality"  in row and math.isfinite(row["magface_quality"])  else float("nan")
            wrong_id   = int(math.isfinite(afs) and afs < T_id)
            too_blurry = int(math.isfinite(mqs) and mqs < Q_min)

            # Bin labels + bin score
            by = _bin_yaw(wy)
            bp = _bin_pitch(wp)
            afn = _norm01(afs, id_p05, id_p95)
            mfn = _norm01(mqs, q_p05,  q_p95)
            conf_combo = max(of_conf, mp_conf, osf_conf)
            bin_score = float(0.5*afn + 0.5*mfn) * (0.5 + 0.5*conf_combo)

            w.writerow([
                os.path.basename(pth),
                f"{wy:.3f}" if math.isfinite(wy) else "",
                f"{wp:.3f}" if math.isfinite(wp) else "",
                f"{wr:.3f}" if math.isfinite(wr) else "",
                eyes, mouth, gh, gv,
                f"{afs:.6f}" if math.isfinite(afs) else "",
                f"{mqs:.6f}" if math.isfinite(mqs) else "",
                wrong_id, too_blurry, by, bp,
                f"{bin_score:.4f}"
            ])

            done += 1
            fps = 0 if (time.time()-t0)<=0 else done/(time.time()-t0)
            bar.update(min(done, total), fails=None, fps=fps)
        bar.close()

    # Write a tiny bin table summary (yaw × pitch)
    import collections
    counts = collections.Counter()
    try:
        with out_csv.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                counts[(r["bin_yaw"], r["bin_pitch"])] += 1
    except Exception:
        pass
    with bin_table.open("w", encoding="utf-8") as f:
        f.write("# yaw\\pitch\tDOWN\tMID\tUP\n")
        for y in ["L","LC","C","RC","R","UNK"]:
            row = [str(counts.get((y,p),0)) for p in ["DOWN","MID","UP"]]
            f.write(f"{y}\t" + "\t".join(row) + "\n")

    return str(out_csv)

# --- Optional CLI hook (non-breaking): python curator_tag.py --fuse --logs <root> [--id_thresh x] [--mag_floor y]
def _cli_fuse_entry():
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fuse", action="store_true", help="Run the 5-tool fusion and write tags.csv")
    ap.add_argument("--logs", type=str, help="logs root (e.g., /workspace/data_src/logs)")
    ap.add_argument("--id_thresh", type=float, default=None)
    ap.add_argument("--mag_floor", type=float, default=None)
    args, _ = ap.parse_known_args()
    if args.fuse and args.logs:
        outp = fuse_and_write_tags(args.logs, id_thresh=args.id_thresh, mag_floor=args.mag_floor)
        print("[OK] wrote", outp)
        sys.exit(0)

# call hook but don't interfere with any existing main()
try:
    _cli_fuse_entry()
except SystemExit:
    raise
except Exception:
    pass


# === LEGACY CLI toggles + filtered ingestion (on/off per tool) ===
def _write_auto_help(script_path, flags_dict):
    """
    Write concise CLI help and current flag state to <script>.py.txt
    (policy: Every script writes concise help to <script>.py.txt on run)
    """
    try:
        sp = Path(script_path)
        hp = sp.with_suffix(sp.suffix + ".txt")  # e.g., curator_tag.py.txt
        lines = []
        lines.append("curator_tag.py — tool fusion and tagging")
        lines.append("")
        lines.append("USAGE (legacy toggles):")
        lines.append("  python3 -u /workspace/scripts/curator_tag.py \\")
        lines.append("    --aligned <ALIGNED> --logs <LOGS> \\")
        lines.append("    --openface3 on|off --mediapipe on|off --openseeface on|off \\")
        lines.append("    --arcface on|off --magface on|off \\")
        lines.append("    [--dedupe on|off] [--allow_dupe on|off] [--override]")
        lines.append("")
        lines.append("Also supported: --fuse --logs <LOGS> [--id_thresh x] [--mag_floor y]")
        lines.append("")
        lines.append("Current run flags:")
        for k,v in flags_dict.items():
            lines.append(f"  {k} = {v}")
        hp.write_text("\\n".join(lines))
    except Exception:
        pass

def _read_any_csv(path_candidates):
    for p in path_candidates:
        df = _read_csv_safe(p)
        if df is not None:
            return df
    return None

def load_tools_to_frames_filtered(logs_root, enable):
    """Conditional ingest based on 'enable' dict keys: of3, mp, osf, af, mf (True/False/None=auto)."""
    logs_root = Path(logs_root)
    of = mp = osf = af = mf = None
    if enable.get("of3", None) is not False:
        of = _read_any_csv([logs_root/"openface3"/"openface3.csv"])
    if enable.get("mp", None) is not False:
        mp = _read_any_csv([logs_root/"mediapipe"/"mp_tags.csv", logs_root/"mediapipe"/"mediapipe.csv"])
    if enable.get("osf", None) is not False:
        osf = _read_any_csv([logs_root/"openseeface"/"osf_tags.csv", logs_root/"openseeface"/"openseeface.csv"])
    if enable.get("af", None) is not False:
        af = _read_any_csv([logs_root/"arcface"/"arcface.csv"])
    if enable.get("mf", None) is not False:
        mf = _read_any_csv([logs_root/"magface"/"magface.csv"])

    ofn = _to_df(_norm_openface3(of)) if of is not None else None
    mpn = _to_df(_norm_mediapipe(mp)) if mp is not None else None
    osfn= _to_df(_norm_osf(osf))      if osf is not None else None
    afn = _to_df(_norm_arcface(af))   if af is not None else None
    mfn = _to_df(_norm_magface(mf))   if mf is not None else None

    try:
        import pandas as pd
    except Exception:
        return None

    dfs = [x for x in [ofn, mpn, osfn, afn, mfn] if x is not None]
    if not dfs:
        return None
    from functools import reduce
    def _merge(a,b): return a.merge(b, on="path", how="outer")
    out = reduce(_merge, dfs)
    out["path"] = out["path"].astype(str).apply(lambda s: os.path.basename(s))
    return out

def _apply_dedupe(df, logs_root, allow_dupe=True):
    """If a dedupe decisions file exists and allow_dupe=False, drop rows marked duplicate."""
    if allow_dupe:
        return df
    import pandas as pd, json
    logs_root = Path(logs_root)
    # Try a few candidates
    cands = [
        logs_root/"dedupe"/"decisions.csv",
        logs_root/"dedupe"/"dedupe.csv",
        logs_root/"curator"/"dedupe.csv",
        logs_root/"dedupe"/"decisions.jsonl",
    ]
    dec = None
    for p in cands:
        if p.suffix == ".csv" and p.exists():
            try:
                dec = pd.read_csv(p)
                break
            except Exception:
                pass
        if p.suffix == ".jsonl" and p.exists():
            try:
                rows = []
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
                if rows:
                    dec = pd.DataFrame(rows)
                    break
            except Exception:
                pass
    if dec is None or "path" not in dec.columns:
        return df
    # Keep rows where decision says keep (keep==1 or decision in {'keep','primary','best'})
    keep_mask = None
    if "keep" in dec.columns:
        keep_mask = dec["keep"].astype("int") == 1
    elif "decision" in dec.columns:
        keep_mask = dec["decision"].astype("str").str.lower().isin(["keep","primary","best"])
    if keep_mask is None:
        return df
    keep_df = dec.loc[keep_mask, ["path"]].copy()
    keep_df["path"] = keep_df["path"].astype(str).apply(lambda s: os.path.basename(s))
    df = df.merge(keep_df, on="path", how="inner")
    return df

def _cli_legacy_entry_curator_tag():
    import argparse, sys
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--aligned", type=str, required=False, help="Aligned faces dir (kept for compatibility)")
    ap.add_argument("--logs",    type=str, required=True, help="Logs root")
    # per-tool toggles
    ap.add_argument("--openface3",   choices=["on","off"], default=None)
    ap.add_argument("--mediapipe",   choices=["on","off"], default=None)
    ap.add_argument("--openseeface", choices=["on","off"], default=None)
    ap.add_argument("--arcface",     choices=["on","off"], default=None)
    ap.add_argument("--magface",     choices=["on","off"], default=None)
    # dedupe
    ap.add_argument("--dedupe",      choices=["on","off"], default=None, help="(merge decisions only; filesystem unchanged)")
    ap.add_argument("--allow_dupe",  choices=["on","off"], default="on", help="If off and decisions exist, drop dupes")
    # misc
    ap.add_argument("--override", action="store_true")
    # advanced thresholds (optional)
    ap.add_argument("--id_thresh", type=float, default=None)
    ap.add_argument("--mag_floor", type=float, default=None)

    args, _ = ap.parse_known_args()

    # Trigger legacy path only when these flags are present
    legacy_used = any(getattr(args, k) is not None for k in
                      ["openface3","mediapipe","openseeface","arcface","magface","dedupe","allow_dupe"])
    if not legacy_used:
        return  # let other mains (e.g., --fuse) handle

    # Auto-help file
    flags = {
        "openface3": args.openface3 or "auto",
        "mediapipe": args.mediapipe or "auto",
        "openseeface": args.openseeface or "auto",
        "arcface": args.arcface or "auto",
        "magface": args.magface or "auto",
        "dedupe": args.dedupe or "off",
        "allow_dupe": args.allow_dupe,
        "override": args.override,
        "id_thresh": args.id_thresh,
        "mag_floor": args.mag_floor,
    }
    _write_auto_help(__file__, flags)

    out_dir = Path(args.logs) / "curator"
    out_csv = out_dir / "tags.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not args.override:
        print(f"[SKIP] {out_csv} exists. Use --override to overwrite.")
        sys.exit(0)

    enable = {
        "of3": None if args.openface3   is None else (args.openface3=="on"),
        "mp":  None if args.mediapipe   is None else (args.mediapipe=="on"),
        "osf": None if args.openseeface is None else (args.openseeface=="on"),
        "af":  None if args.arcface     is None else (args.arcface=="on"),
        "mf":  None if args.magface     is None else (args.magface=="on"),
    }

    # Ingest with filters
    df = load_tools_to_frames_filtered(args.logs, enable)
    if df is None or len(df)==0:
        # fall back to generic fuse reader (tolerates missing sources)
        fuse_and_write_tags(args.logs, out_dir=str(out_dir), id_thresh=args.id_thresh, mag_floor=args.mag_floor)
        print(f"[OK] wrote {out_csv}")
        sys.exit(0)

    # Apply dedupe decisions if requested
    allow_dupe = (args.allow_dupe != "off")
    if args.dedupe == "on":
        df = _apply_dedupe(df, args.logs, allow_dupe=allow_dupe)

    # Reuse fusion logic by temporarily overriding reader
    global load_tools_to_frames
    def _lt_override(_): return df
    old = load_tools_to_frames
    load_tools_to_frames = _lt_override
    try:
        fuse_and_write_tags(args.logs, out_dir=str(out_dir), id_thresh=args.id_thresh, mag_floor=args.mag_floor)
    finally:
        load_tools_to_frames = old

    print(f"[OK] wrote {out_csv}")
    sys.exit(0)

# Register legacy CLI hook; don't block other mains
try:
    _cli_legacy_entry_curator_tag()
except SystemExit:
    raise
except Exception:
    pass
