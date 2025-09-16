#!/usr/bin/env python3
import argparse, collections
from pathlib import Path
from curator_lib.io import read_csv, ensure_dir, write_csv

def main():
    ap = argparse.ArgumentParser(description="curator_stats — aggregate bin counts")
    ap.add_argument("--logs", required=True)
    args = ap.parse_args()
    logs = Path(args.logs)
    tags = read_csv(logs / "tags.csv")
    outdir = ensure_dir(logs / "stats")
    if not tags:
        print("[warn] no tags.csv; nothing to summarize")
        return
    # simple counts by bins
    key_fields = ["bin_yaw","bin_pitch","bin_eyes","bin_smile","bin_teeth","bin_light"]
    counter = collections.Counter(tuple((t.get(k,"") for k in key_fields)) for t in tags)
    rows = []
    for k, n in counter.items():
        row = {f: v for f, v in zip(key_fields, k)}
        row["count"] = n
        rows.append(row)
    write_csv(outdir / "bin_counts.csv", key_fields + ["count"], rows)
    print(f"[ok] wrote {outdir/'bin_counts.csv'}")

if __name__ == "__main__":
    main()
