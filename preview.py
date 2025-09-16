#!/usr/bin/env python3
import argparse, random
from pathlib import Path
from curator_lib.io import safe_glob, ensure_dir, write_csv

def main():
    ap = argparse.ArgumentParser(description="curator_preview — sample preview list per bin (skeleton)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--samples_per_bin", type=int, default=8)
    args = ap.parse_args()

    ds = Path(args.dataset); logs = Path(args.logs)
    files = safe_glob(ds)
    # skeleton: random sample only
    sample = files[:args.samples_per_bin*10]
    out = ensure_dir(logs / "preview")
    write_csv(out / "preview_list.csv", ["path"], [{"path": p} for p in sample])
    print(f"[ok] wrote {out/'preview_list.csv'}")

if __name__ == "__main__":
    main()
