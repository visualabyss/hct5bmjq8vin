#!/usr/bin/env python3
# library.py v1.0
# - Subcommands: add, refresh
# - add: ingest aligned frames into a hash-addressed library and copy/link per-frame logs (OpenFace, MagFace, bbox).
# - refresh: remove manifest rows and logs for images that are missing from library/aligned.
#
# Library layout:
#   <lib_root>/aligned/<hash>.png
#   <lib_root>/logs/openface_img/<hash>.csv
#   <lib_root>/logs/openface_bbox/<hash>.txt
#   <lib_root>/logs/magface/<hash>.npy
#   <lib_root>/logs/manifest.jsonl
#
# Note: We do not re-run OpenFace/MagFace here. We only copy/link existing logs
#       from --src_logs if present. You can later run curator.py tag over the library
#       aligned folder to compute CCT/exposure and fill more fields if needed.

import argparse, json, os, sys, shutil, hashlib
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def list_images(d):
    p = Path(d)
    files = [x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def sha1_of_file(p: Path):
    h = hashlib.sha1()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def write_jsonl_append(rows, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(fp: Path):
    rows = []
    if not fp.exists(): return rows
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def safe_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try: os.link(src, dst)
        except Exception: shutil.copy2(src, dst)
    elif mode == "symlink":
        try: dst.symlink_to(src)
        except Exception: shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

# ----------------- add -----------------
def cmd_add(args):
    aligned = Path(args.aligned)
    lib_root = Path(args.lib_root)
    src_logs = Path(args.src_logs) if args.src_logs else None

    if not aligned.exists():
        print("ERR: --aligned not found:", aligned); return 2
    lib_aligned = lib_root/"aligned"
    lib_logs = lib_root/"logs"
    lib_of = lib_logs/"openface_img"
    lib_bbox = lib_logs/"openface_bbox"
    lib_mag = lib_logs/"magface"
    lib_manifest = lib_logs/"manifest.jsonl"
    for d in (lib_aligned, lib_of, lib_bbox, lib_mag):
        d.mkdir(parents=True, exist_ok=True)

    existing = read_jsonl(lib_manifest)
    have = set(r.get("hash") for r in existing if r.get("hash"))

    new_rows = []
    imgs = list_images(aligned)
    total = len(imgs)
    for i, img in enumerate(imgs, 1):
        h = sha1_of_file(img)
        if h in have:
            continue
        dst_img = lib_aligned/f"{h}.png"
        safe_link(img, dst_img, args.link_mode)

        row = {"hash": h, "file": str(dst_img)}

        if src_logs and src_logs.exists():
            stem = img.stem
            # prefer by hash if source filenames are already hashes; else by stem
            of_src = src_logs/"openface_img"/f"{stem}.csv"
            if of_src.exists():
                safe_link(of_src, lib_of/f"{h}.csv", args.link_mode)
                row["of_csv"] = str(lib_of/f"{h}.csv")
            mf_src = src_logs/"magface"/f"{stem}.npy"
            if mf_src.exists():
                safe_link(mf_src, lib_mag/f"{h}.npy", args.link_mode)
                row["magface"] = str(lib_mag/f"{h}.npy")
            bb_src = src_logs/"openface_bbox"/f"{stem}.txt"
            if bb_src.exists():
                safe_link(bb_src, lib_bbox/f"{h}.txt", args.link_mode)
                row["openface_bbox"] = str(lib_bbox/f"{h}.txt")

        new_rows.append(row)
        if i % max(1,args.pb_every) == 0 or i == total:
            frac = i/total
            bar = "█"*int(frac*28) + "░"*int((1-frac)*28)
            sys.stdout.write(f"\r[ADD] [{bar}] {i}/{total} {frac*100:5.1f}%")
            sys.stdout.flush()

    if new_rows:
        write_jsonl_append(new_rows, lib_manifest)
    if total>0: sys.stdout.write("\n")
    print(f"[ADD] added {len(new_rows)} new rows to {lib_manifest}")
    return 0

# ----------------- refresh -----------------
def cmd_refresh(args):
    lib_root = Path(args.lib_root)
    lib_aligned = lib_root/"aligned"
    lib_logs = lib_root/"logs"
    lib_of = lib_logs/"openface_img"
    lib_bbox = lib_logs/"openface_bbox"
    lib_mag = lib_logs/"magface"
    lib_manifest = lib_logs/"manifest.jsonl"

    rows = read_jsonl(lib_manifest)
    kept = []
    removed = 0

    for i, r in enumerate(rows, 1):
        h = r.get("hash")
        fp = Path(r.get("file",""))
        ok = h and fp.exists()
        if not ok:
            removed += 1
            # remove logs
            for candidate in [lib_of/f"{h}.csv", lib_bbox/f"{h}.txt", lib_mag/f"{h}.npy"]:
                try:
                    if candidate.exists(): candidate.unlink()
                except Exception: pass
        else:
            kept.append(r)

        if i % max(1,args.pb_every) == 0 or i == len(rows):
            frac = i/max(1,len(rows))
            bar = "█"*int(frac*28) + "░"*int((1-frac)*28)
            sys.stdout.write(f"\r[REFRESH] [{bar}] {i}/{len(rows)} {frac*100:5.1f}%")
            sys.stdout.flush()

    # rewrite manifest with kept only
    lib_manifest.write_text("", encoding="utf-8")
    write_jsonl_append(kept, lib_manifest)
    if rows: sys.stdout.write("\n")
    print(f"[REFRESH] kept {len(kept)} rows, removed {removed}. Manifest: {lib_manifest}")
    return 0

def build_parser():
    p = argparse.ArgumentParser("library v1.0 (add, refresh)")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp_add = sp.add_parser("add")
    sp_add.add_argument("--aligned", type=str, required=True)
    sp_add.add_argument("--lib_root", type=str, required=True)
    sp_add.add_argument("--src_logs", type=str, default="")
    sp_add.add_argument("--link_mode", type=str, default="hardlink", choices=["hardlink","symlink","copy"])
    sp_add.add_argument("--pb_every", type=int, default=500)

    sp_ref = sp.add_parser("refresh")
    sp_ref.add_argument("--lib_root", type=str, required=True)
    sp_ref.add_argument("--pb_every", type=int, default=1000)

    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "add":
        rc = cmd_add(args)
    elif args.cmd == "refresh":
        rc = cmd_refresh(args)
    else:
        rc = 1
    sys.exit(rc)

if __name__ == "__main__":
    main()
