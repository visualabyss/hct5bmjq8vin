#!/usr/bin/env python3
import argparse, csv, shutil, sys
from pathlib import Path

IMG_EXTS = {".png",".jpg",".jpeg",".webp",".bmp"}

def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def load_csv(p: Path):
    with p.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    return rows, rdr.fieldnames

def write_csv(p: Path, rows, fieldnames):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def resolve_sources(inp: Path):
    """
    Accept either:
      A) per-video root containing aligned/ and logs/
      B) an aligned/ folder directly (logs expected at parent/logs)
    Returns (src_aligned, src_logs, video_id) or (None, None, None) if invalid.
    """
    if (inp / "aligned").exists():
        src_aligned = inp / "aligned"
        src_logs    = inp / "logs"
        vid_id      = inp.name
        return src_aligned, src_logs, vid_id

    # If the path itself is an aligned folder or just a folder of images
    if inp.exists() and inp.is_dir():
        imgs = [p for p in inp.iterdir() if p.suffix.lower() in IMG_EXTS]
        if inp.name.lower() == "aligned" or imgs:
            src_aligned = inp
            clip_root   = inp.parent
            src_logs    = clip_root / "logs"
            vid_id      = clip_root.name
            return src_aligned, src_logs, vid_id

    return None, None, None

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Merge multiple video dumps into /workspace/{data_src|data_trg}/aligned and /logs.\n"
            "Inputs can be either per-video roots (containing 'aligned/' & 'logs/') OR the 'aligned/' folders themselves."
        )
    )
    ap.add_argument("--role", choices=["data_src","data_trg"], required=True,
                    help="Whether we are merging source or target videos.")
    ap.add_argument("--dst_root", required=True,
                    help="Destination root, e.g. /workspace/data_src or /workspace/data_trg")
    ap.add_argument("inputs", nargs="+",
                    help="Per-video roots or aligned folders (order = merge order)")
    args = ap.parse_args()

    dst_root    = Path(args.dst_root).resolve()
    dst_aligned = dst_root / "aligned"
    dst_logs    = dst_root / "logs"
    dst_aligned.mkdir(parents=True, exist_ok=True)
    dst_logs.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    new_idx = 0

    merged_video_logs = []
    vlog_fields = None

    for raw in args.inputs:
        in_path = Path(raw).resolve()
        src_aligned, src_logs, vid_id = resolve_sources(in_path)
        if src_aligned is None:
            print(f"[WARN] skip {in_path} (not a per-video root or aligned folder)", flush=True)
            continue

        if not src_aligned.exists():
            print(f"[WARN] skip {in_path} (aligned folder missing)", flush=True)
            continue

        imgs = list_images(src_aligned)
        if not imgs:
            print(f"[WARN] no images in {src_aligned}", flush=True)
            continue

        # copy frames (renumber sequentially)
        map_old_to_new = {}
        for p in imgs:
            new_idx += 1
            new_name = f"{new_idx:06d}{p.suffix.lower()}"
            shutil.copy2(p, dst_aligned / new_name)
            manifest_rows.append({
                "new_file": new_name,
                "src_dir": str(src_aligned),
                "old_file": p.name,
                "video_id": vid_id,
            })
            map_old_to_new[p.name] = new_name

        # merge video_logs.csv if present (best-effort)
        src_vlog = src_logs / "video_logs.csv"
        if src_vlog.exists():
            try:
                rows, flds = load_csv(src_vlog)
                if "file" not in flds:
                    print(f"[WARN] video_logs.csv at {src_vlog} lacks 'file' column; skipping logs for this input.")
                    rows = []
                if rows:
                    if vlog_fields is None:
                        vlog_fields = list(dict.fromkeys(["file","video_id"] + flds))
                    for r in rows:
                        old = r.get("file","")
                        if old in map_old_to_new:
                            rr = dict(r)
                            rr["video_id"] = vid_id
                            rr["file"] = map_old_to_new[old]
                            merged_video_logs.append(rr)
            except Exception as e:
                print(f"[WARN] failed reading {src_vlog}: {e}")

        print(f"[OK] merged {len(map_old_to_new)} frames from {vid_id}", flush=True)

    # Write manifest
    write_csv(dst_logs / "_manifest.csv", manifest_rows, ["new_file","src_dir","old_file","video_id"])

    # Write merged video_logs.csv only (scores.csv removed by design)
    key = lambda r: r.get("file","")
    if merged_video_logs:
        merged_video_logs.sort(key=key)
        write_csv(dst_logs / "video_logs.csv", merged_video_logs, vlog_fields)
        print(f"[OK] wrote merged video_logs → {dst_logs/'video_logs.csv'} ({len(merged_video_logs)} rows)", flush=True)
    else:
        print("[INFO] no video_logs.csv found in inputs; none merged.", flush=True)

    print(f"[DONE] total merged frames: {new_idx} → {dst_aligned}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
