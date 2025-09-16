#!/usr/bin/env python3
import argparse, subprocess, sys, os, shlex, re
from pathlib import Path
from typing import Tuple, List

# ---------------- ffmpeg / ffprobe ----------------
def which(binname: str) -> str:
    from shutil import which as _which
    p = _which(binname)
    if not p:
        sys.exit(f"[FATAL] '{binname}' not found. Install it (apt-get update && apt-get install -y ffmpeg).")
    return p

FFMPEG  = which("ffmpeg")
FFPROBE = which("ffprobe")

# ---------------- utils ----------------
VIDEO_EXTS = (".mp4",".mkv",".mov",".m4v",".webm",".avi",".wmv",".ts",".m2ts",".mpg",".mpeg")

def parse_time(ts: str) -> float:
    """Accept 'SS', 'MM:SS', 'HH:MM:SS(.ms)'."""
    ts = ts.strip()
    if re.fullmatch(r"\d+(\.\d+)?", ts):
        return float(ts)
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = parts
        return float(m)*60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return float(h)*3600 + float(m)*60 + float(s)
    raise ValueError(f"Invalid time format: {ts} (use SS | MM:SS | HH:MM:SS)")

def ffprobe_duration(src: str) -> float:
    out = subprocess.check_output(
        [FFPROBE, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', src],
        stderr=subprocess.STDOUT
    ).decode().strip()
    return float(out)

def ffprobe_has_audio(src: str) -> bool:
    out = subprocess.run(
        [FFPROBE, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', src],
        capture_output=True, text=True
    ).stdout.strip()
    return bool(out)

def find_single_media(inp: str) -> str:
    """
    If inp is a file → return it.
    If inp is a dir → pick the first top-level video file (sorted by name).
    """
    p = Path(inp)
    if p.is_file():
        return str(p)
    if not p.exists():
        sys.exit(f"[FATAL] Input not found: {inp}")
    vids = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    if not vids:
        sys.exit(f"[FATAL] No video files found directly in folder: {inp}")
    if len(vids) > 1:
        print("[WARN] Multiple videos found; using the first by name:\n" + "\n".join(f" - {v.name}" for v in vids), flush=True)
    return str(vids[0])

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def run_ffmpeg(cmd: List[str]) -> None:
    # Force non-interactive ffmpeg
    if cmd and cmd[0] == FFMPEG and "-nostdin" not in cmd:
        cmd.insert(1, "-nostdin")
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        sys.exit(f"[FATAL] ffmpeg failed with code {r.returncode}")

def safe_write_path(inp: str, out: str, overwrite: bool) -> Tuple[str, str, bool]:
    """
    Handles in-place output: if out == inp, write to a temp file then atomically replace.
    Returns (final_out, temp_path, used_temp)
    """
    if not overwrite and Path(out).exists():
        sys.exit(f"[FATAL] Output exists: {out} (use --overwrite)")
    if Path(out).resolve() == Path(inp).resolve():
        base = Path(inp)
        # keep real extension last so muxer is detected
        tmp = str(base.with_name(base.stem + ".tmpcut" + base.suffix))
        if Path(tmp).exists() and not overwrite:
            sys.exit(f"[FATAL] Temp exists: {tmp} (remove it or use --overwrite)")
        return (inp, tmp, True)
    else:
        ensure_parent(out)
        return (out, out, False)

def finalize_inplace(final_out: str, tmp: str, used_temp: bool):
    if used_temp:
        os.replace(tmp, final_out)

# ---------------- encoder helpers ----------------
def _enc_args(args, for_concat=False):
    """
    Build encoder args.
    Returns:
      in_args: input options (e.g., hwaccel) placed BEFORE -i
      v_args : video encoder/muxer opts (after maps)
      a_args : audio encoder opts
    Note: for_concat=True (filter path) -> disable hwaccel to avoid hwupload/download in filters.
    """
    in_args: List[str] = []
    v: List[str] = []
    a: List[str] = []
    if args.use_nvenc:
        enc = {"h264":"h264_nvenc", "hevc":"hevc_nvenc", "av1":"av1_nvenc"}[args.nv_codec]
        if args.hwaccel and not for_concat:
            in_args = ["-hwaccel","cuda","-hwaccel_output_format","cuda"]
        v += ["-c:v", enc, "-preset", args.nv_preset, "-rc", "vbr", "-cq", str(args.nv_cq), "-b:v", "0",
              "-movflags","+faststart"]
        a = ["-c:a","aac","-b:a","192k"]
    else:
        v += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-movflags","+faststart"]
        a = ["-c:a","aac","-b:a","192k"]
    return in_args, v, a

# ---------------- operations ----------------
def cut_from_start(inp_file: str, out_path: str|None, start_ts: str, reencode: bool, overwrite: bool, args):
    start = parse_time(start_ts)
    out = out_path or inp_file
    final_out, target, used_temp = safe_write_path(inp_file, out, overwrite)
    if not reencode:
        cmd = [FFMPEG, "-y" if overwrite else "-n", "-ss", f"{start}", "-i", inp_file, "-c", "copy",
               "-avoid_negative_ts", "make_zero", target]
    else:
        inargs, vargs, aargs = _enc_args(args, for_concat=False)
        cmd = [FFMPEG, "-y" if overwrite else "-n", *inargs, "-ss", f"{start}", "-i", inp_file, "-map","0", *vargs, *aargs, target]
    run_ffmpeg(cmd); finalize_inplace(final_out, target, used_temp); print(f"[DONE] Wrote {final_out}")

def keep_head(inp_file: str, out_path: str|None, end_ts: str, reencode: bool, overwrite: bool, args):
    end = parse_time(end_ts)
    out = out_path or inp_file
    final_out, target, used_temp = safe_write_path(inp_file, out, overwrite)
    if not reencode:
        cmd = [FFMPEG, "-y" if overwrite else "-n", "-i", inp_file, "-to", f"{end}", "-c", "copy", target]
    else:
        inargs, vargs, aargs = _enc_args(args, for_concat=False)
        cmd = [FFMPEG, "-y" if overwrite else "-n", *inargs, "-i", inp_file, "-to", f"{end}", "-map","0", *vargs, *aargs, target]
    run_ffmpeg(cmd); finalize_inplace(final_out, target, used_temp); print(f"[DONE] Wrote {final_out}")

def cut_tail(inp_file: str, out_path: str|None, tail_ts: str, reencode: bool, overwrite: bool, args):
    # If tail_ts looks like absolute time (has ':'), treat as end timestamp (alias to keep).
    if ":" in tail_ts:
        return keep_head(inp_file, out_path, tail_ts, reencode, overwrite, args)
    tail = parse_time(tail_ts)
    dur  = ffprobe_duration(inp_file)
    keep = dur - tail
    if keep <= 0:
        sys.exit(f"[FATAL] tail duration ({tail}s) >= video duration ({dur:.3f}s)")
    return keep_head(inp_file, out_path, str(keep), reencode, overwrite, args)

def remove_range_concat_copy(inp_file: str, out_path: str|None, start_ts: str, end_ts: str, overwrite: bool):
    """Stream-copy remove [start,end) by creating two parts and concatenating."""
    s = parse_time(start_ts); e = parse_time(end_ts)
    if e <= s:
        sys.exit("[FATAL] --remove END must be greater than START")
    out = out_path or inp_file
    final_out, target, used_temp = safe_write_path(inp_file, out, overwrite)
    base = Path(target)
    seg1 = str(base.with_name(base.stem + ".part1" + base.suffix))
    seg2 = str(base.with_name(base.stem + ".part2" + base.suffix))
    lst  = str(base.with_name(base.stem + ".concat.txt"))

    cmdA = [FFMPEG, "-y" if overwrite else "-n", "-i", inp_file, "-t", f"{s}", "-c", "copy",
            "-avoid_negative_ts", "make_zero", seg1]
    cmdB = [FFMPEG, "-y" if overwrite else "-n", "-ss", f"{e}", "-i", inp_file, "-c", "copy",
            "-avoid_negative_ts", "make_zero", seg2]
    run_ffmpeg(cmdA); run_ffmpeg(cmdB)
    with open(lst, "w", encoding="utf-8") as f:
        f.write(f"file '{seg1}'\n"); f.write(f"file '{seg2}'\n")
    cmdC = [FFMPEG, "-y" if overwrite else "-n", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", target]
    run_ffmpeg(cmdC)
    for tmp in (seg1, seg2, lst):
        try: os.remove(tmp)
        except Exception: pass
    finalize_inplace(final_out, target, used_temp); print(f"[DONE] Wrote {final_out}")

def remove_range_filter_reencode(inp_file: str, out_path: str|None, start_ts: str, end_ts: str, overwrite: bool, args):
    """Accurate remove with one re-encode pass using concat filter."""
    s = parse_time(start_ts); e = parse_time(end_ts)
    if e <= s:
        sys.exit("[FATAL] --remove END must be greater than START")
    out = out_path or inp_file
    final_out, target, used_temp = safe_write_path(inp_file, out, overwrite)
    has_aud = ffprobe_has_audio(inp_file)
    if has_aud:
        fc = (
            f"[0:v]trim=start=0:end={s},setpts=PTS-STARTPTS[v0];"
            f"[0:a]atrim=start=0:end={s},asetpts=PTS-STARTPTS[a0];"
            f"[0:v]trim=start={e},setpts=PTS-STARTPTS[v1];"
            f"[0:a]atrim=start={e},asetpts=PTS-STARTPTS[a1];"
            f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )
        maps = ["-map","[v]","-map","[a]"]
    else:
        fc = (
            f"[0:v]trim=start=0:end={s},setpts=PTS-STARTPTS[v0];"
            f"[0:v]trim=start={e},setpts=PTS-STARTPTS[v1];"
            f"[v0][v1]concat=n=2:v=1:a=0[v]"
        )
        maps = ["-map","[v]"]
    # for filter path, disable input hwaccel in _enc_args by passing for_concat=True
    _in, vargs, aargs = _enc_args(args, for_concat=True)
    cmd = [FFMPEG, "-y" if overwrite else "-n", "-i", inp_file, "-filter_complex", fc, *maps, *vargs, *aargs, target]
    run_ffmpeg(cmd); finalize_inplace(final_out, target, used_temp); print(f"[DONE] Wrote {final_out}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Video cutter: start at time, keep head, cut tail, or remove a middle range. "
                    "Accepts a FILE or a FOLDER for --input. If a folder is given, the first video inside is used. "
                    "By default outputs with the SAME FILENAME (in-place). Use --overwrite to confirm."
    )
    ap.add_argument("--input", required=True, help="Input video file OR folder (top-level, non-recursive)")
    ap.add_argument("--out", default=None, help="Output file OR directory (if directory given, uses same filename)")
    ap.add_argument("--reencode", action="store_true", help="Accurate cuts via re-encode (slower).")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting the original (in-place).")

    # NVENC defaults: GPU encode ON, CPU decode default (hwaccel 0) for stability
    ap.add_argument("--use_nvenc", type=int, default=1, help="Use NVIDIA NVENC when re-encoding (1=on,0=off)")
    ap.add_argument("--nv_codec", choices=["h264","hevc","av1"], default="h264", help="NVENC codec")
    ap.add_argument("--nv_preset", default="p5", help="NVENC preset p1..p7")
    ap.add_argument("--nv_cq", type=int, default=19, help="NVENC quality (lower=better)")
    ap.add_argument("--hwaccel", type=int, default=0, help="Use hardware decode (cuda) when re-encoding (0=off,1=on)")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--start", help="Start video at this time (SS | MM:SS | HH:MM:SS). Drops beginning.")
    g.add_argument("--keep", help="Keep only the first DURATION / end timestamp (e.g. 1:40 or 14:23). Alias: --end_at")
    g.add_argument("--cut_tail", help="Drop the tail. If value has ':', it's treated as END timestamp; else duration from end.")
    g.add_argument("--remove", help="Remove a middle range 'START-END' (e.g. 11:00-12:00).")
    ap.add_argument("--end_at", help=argparse.SUPPRESS)  # hidden alias to --keep

    args = ap.parse_args()

    # resolve input (folder -> first video file)
    src = find_single_media(args.input)
    print(f"[INFO] Using input: {src}", flush=True)

    # If --out is a directory, build output path with same filename
    if args.out is not None:
        outp = Path(args.out)
        if outp.exists() and outp.is_dir():
            args.out = str(outp / Path(src).name)

    # prefer --keep if alias provided
    if args.end_at and not args.keep:
        args.keep = args.end_at

    if args.start:
        cut_from_start(src, args.out, args.start, args.reencode, args.overwrite, args); return

    if args.keep:
        keep_head(src, args.out, args.keep, args.reencode, args.overwrite, args); return

    if args.cut_tail:
        cut_tail(src, args.out, args.cut_tail, args.reencode, args.overwrite, args); return

    if args.remove:
        m = re.fullmatch(r"\s*([^-\s]+)\s*-\s*([^-\s]+)\s*", args.remove)
        if not m:
            sys.exit("[FATAL] --remove expects 'START-END' (e.g. 11:00-12:00)")
        s, e = m.group(1), m.group(2)
        if args.reencode:
            remove_range_filter_reencode(src, args.out, s, e, args.overwrite, args)
        else:
            remove_range_concat_copy(src, args.out, s, e, args.overwrite)
        return

if __name__ == "__main__":
    main()
