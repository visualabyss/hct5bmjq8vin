#!/usr/bin/env python3
import argparse
from pathlib import Path
from curator_lib.io import read_csv, ensure_dir, write_csv, safe_glob

def default_logs_for_source_dir(source_dir: Path) -> Path:
    cand = source_dir.parent / "logs"
    if cand.exists() or True:
        return cand
    return source_dir / "logs"

def maybe_target_logs_from_aligned(target_aligned: Path) -> Path:
    # try sibling logs dir of data_trg
    cand = target_aligned.parent.parent / "logs"
    return cand

def main():
    ap = argparse.ArgumentParser(description="curator_build — write training indices (outputs fixed to data_src/logs/build)")
    ap.add_argument("--source_dir", required=True, help="Aligned directory for PRIMARY SOURCE (e.g., data_src/aligned)")
    ap.add_argument("--target_aligned", required=True, help="Aligned directory for TARGET VIDEO / REPEATED TARGET for this run")
    ap.add_argument("--source_logs", default="", help="Logs for PRIMARY SOURCE (default: data_src/logs)")
    ap.add_argument("--target_logs", default="", help="Logs for TARGET (default: try data_trg/logs)")
    ap.add_argument("--matched_csv", default="", help="Override matched_source.csv (default: data_src/logs/match/matched_source.csv)")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    source_logs = Path(args.source_logs) if args.source_logs else default_logs_for_source_dir(source_dir)
    target_logs = Path(args.target_logs) if args.target_logs else maybe_target_logs_from_aligned(Path(args.target_aligned))

    # Outputs fixed by policy: data_src/logs/build
    build_dir = ensure_dir(source_logs / "build")
    match_dir = source_logs / "match"
    matched_csv = Path(args.matched_csv) if args.matched_csv else (match_dir / "matched_source.csv")

    # SRC index (optionally filter by cleanup_flag==0 if tags.csv exists)
    src_rows = read_csv(matched_csv)
    src_tags_map = {r.get("path",""): r for r in read_csv(source_logs / "tags.csv")}
    src_rows = [r for r in src_rows if src_tags_map.get(r.get("path",""), {}).get("cleanup_flag","0") in ("0","")]

    # TRG index (optionally filter by cleanup_flag==0 if tags.csv exists)
    trg_files = [{"path": p} for p in safe_glob(args.target_aligned)]
    tv_tags_map = {r.get("path",""): r for r in read_csv(target_logs / "tags.csv")}
    # tv tags may have basename keys
    def ok_trg(p: str):
        r = tv_tags_map.get(p, None) or tv_tags_map.get(Path(p).name, None)
        return (r is None) or (str(r.get("cleanup_flag","0")) in ("0",""))
    trg_files = [r for r in trg_files if ok_trg(r["path"])]

    write_csv(build_dir / "train_index_src.csv", ["path"], src_rows)
    write_csv(build_dir / "train_index_trg.csv", ["path"], trg_files)

    print(f"[ok] wrote {build_dir/'train_index_src.csv'} and {build_dir/'train_index_trg.csv'}")
    print(f"[ok] outputs fixed to {build_dir} (per policy)")

if __name__ == "__main__":
    main()
