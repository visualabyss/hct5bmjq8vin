#!/usr/bin/env python3
import argparse, sys, shutil, re, csv
from pathlib import Path

IMG_EXTS = {".png",".jpg",".jpeg",".webp",".bmp"}

def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def read_csv_any(p: Path):
    if not p.exists():
        return [], []
    with p.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        return rows, list(rdr.fieldnames or [])

def next_index(dst_aligned: Path):
    dst_aligned.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in dst_aligned.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            m = re.search(r"(\d+)", p.stem)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx

def main():
    ap = argparse.ArgumentParser(description="compile.py — compile aligned folders (no tags merge) + merge video_logs.csv into dst_logs/compile")
    ap.add_argument("--role", choices=["data_src","data_trg"], required=True, help="Destination role (for reporting only)")
    ap.add_argument("--dst_root", required=True, help="Destination root (contains/receives 'aligned/' and 'logs/')")
    ap.add_argument("inputs", nargs="+", help="Input aligned directories (or per-video roots containing aligned/ and logs/compile/video_logs.csv)")
    args = ap.parse_args()

    dst_root = Path(args.dst_root)
    dst_aligned = dst_root / "aligned"
    dst_logs = dst_root / "logs"
    dst_aligned.mkdir(parents=True, exist_ok=True)
    dst_logs.mkdir(parents=True, exist_ok=True)
    compile_dir = dst_logs / "compile"
    compile_dir.mkdir(parents=True, exist_ok=True)

    new_idx = next_index(dst_aligned)
    copied_total = 0
    compile_map_rows = []

    # For merging video_logs.csv
    video_logs_rows = []
    video_logs_header_union = []

    # De-duplicate inputs preserving order
    seen = set(); inputs = []
    for s in args.inputs:
        if s not in seen:
            seen.add(s); inputs.append(s)

    for src in inputs:
        src = Path(src)
        # Accept either aligned folder directly or per-video root with aligned/
        if src.name == "aligned" and src.is_dir():
            src_aligned = src
            src_root = src.parent
        else:
            src_aligned = src / "aligned"
            src_root = src
        if not src_aligned.exists():
            print(f"[WARN] skip input (no aligned/ found): {src}", flush=True)
            continue

        frames = list_images(src_aligned)
        if not frames:
            print(f"[INFO] input has no frames: {src_aligned}", flush=True)
            # Still try to merge video_logs if present
        else:
            for f in frames:
                new_idx += 1
                ext = f.suffix.lower()
                new_name = f"{new_idx:06d}{ext}"
                dst_file = dst_aligned / new_name
                shutil.copy2(str(f), str(dst_file))
                copied_total += 1
                compile_map_rows.append({
                    "src_root": str(src_root),
                    "src_path": f.as_posix(),
                    "dst_name": new_name,
                })

        # Merge video_logs.csv from this input if present at <src_root>/logs/compile/video_logs.csv
        vlog_path = src_root / "logs" / "compile" / "video_logs.csv"
        if vlog_path.exists():
            rows, hdr = read_csv_any(vlog_path)
            if rows:
                # annotate rows with source root (helps downstream provenance)
                for r in rows:
                    r.setdefault("src_root", str(src_root))
                # union header
                video_logs_header_union = list(dict.fromkeys([*video_logs_header_union, *hdr, "src_root"]))
                video_logs_rows.extend(rows)

    # Write mapping + brief report; DO NOT touch tags.csv
    write_csv(compile_dir / "compile_map.csv", ["src_root","src_path","dst_name"], compile_map_rows)
    (compile_dir / "compile_report.txt").write_text(
        f"role: {args.role}\n"
        f"inputs: {len(inputs)}\n"
        f"frames copied: {copied_total}\n"
        f"dst_aligned: {dst_aligned}\n"
        f"dst_logs: {dst_logs}\n"
        f"NOTE: tags.csv was not read or written by compile.py\n",
        encoding="utf-8"
    )
    print(f"[DONE] compiled {copied_total} frames -> {dst_aligned}", flush=True)

    # Write merged video_logs.csv if any
    if video_logs_rows:
        write_csv(compile_dir / "video_logs.csv", video_logs_header_union, video_logs_rows)
        print(f"[OK] merged video_logs -> {compile_dir/'video_logs.csv'} ({len(video_logs_rows)} rows)", flush=True)
    else:
        print("[INFO] no video_logs.csv found in inputs; nothing merged.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
