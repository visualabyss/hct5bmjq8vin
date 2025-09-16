#!/usr/bin/env python3
import argparse, csv, os, sys, shutil
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = ('.jpg','.jpeg','.png','.webp','.bmp')

def read_csv_any(p: Path):
    if not p.exists():
        return [], []
    with p.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        return rows, list(rdr.fieldnames or [])

def write_csv(path: Path, headers: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

def main():
    ap = argparse.ArgumentParser(description="cleanup.py — move frames flagged in logs/tags.csv to aligned/bad/<reason>/")
    ap.add_argument("--aligned", required=True, help="Aligned directory (e.g., data_src/aligned or data_trg/aligned)")
    ap.add_argument("--logs", required=True, help="Logs directory (e.g., data_src/logs or data_trg/logs)")
    ap.add_argument("--apply", type=int, default=0, help="When 1, perform moves; default dry-run")
    ap.add_argument("--only_reasons", default="", help="Comma-separated reasons to move (subset of cleanup_reason). Empty = all")
    ap.add_argument("--ignore_reasons", default="", help="Comma-separated reasons to skip")
    ap.add_argument("--from_filelist", default="", help="Optional path to a text file with one relative image path per line to move (bypasses tags.csv)")
    args = ap.parse_args()

    aligned = Path(args.aligned)
    logs = Path(args.logs)
    tags_csv = logs / "tags.csv"
    stats_dir = logs / "stats"; stats_dir.mkdir(parents=True, exist_ok=True)

    only_reasons = {s.strip() for s in args.only_reasons.split(",") if s.strip()}
    ignore_reasons = {s.strip() for s in args.ignore_reasons.split(",") if s.strip()}

    candidates = []  # {path, reason}
    if args.from_filelist:
        fl = Path(args.from_filelist)
        if not fl.exists():
            print(f"[fatal] from_filelist not found: {fl}", file=sys.stderr); sys.exit(2)
        for line in fl.read_text(encoding="utf-8").splitlines():
            rel = line.strip()
            if not rel: continue
            candidates.append({"path": rel, "cleanup_reason": "manual"})
    else:
        rows, hdr = read_csv_any(tags_csv)
        if not rows:
            print(f"[warn] no tags.csv found at {tags_csv} or empty; nothing to do")
            return
        for r in rows:
            flag = str(r.get("cleanup_flag","0")).strip()
            if flag not in ("1","true","True","TRUE"):
                continue
            reason = str(r.get("cleanup_reason","cleanup")).strip() or "cleanup"
            # optional filters
            reason_set = {x.strip() for x in reason.split(",") if x.strip()} or {"cleanup"}
            if only_reasons and not (reason_set & only_reasons):
                continue
            if ignore_reasons and (reason_set & ignore_reasons):
                continue
            candidates.append({"path": r.get("path",""), "cleanup_reason": ",".join(sorted(reason_set))})

    if not candidates:
        print("[info] no frames matched cleanup criteria")
        return

    report_rows = []
    moved = 0
    for c in candidates:
        rel = c["path"]
        src = aligned / rel
        if not src.exists():
            # try basename match if relative path mismatched
            alt = aligned / Path(rel).name
            if alt.exists():
                src = alt
            else:
                report_rows.append({"path": rel, "action": "missing", "reason": c["cleanup_reason"], "dst": ""})
                continue
        # choose first reason as folder
        folder_reason = (c["cleanup_reason"].split(",")[0] or "cleanup").strip()
        dst_dir = aligned / "bad" / folder_reason
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if args.apply:
            try:
                shutil.move(str(src), str(dst))
                moved += 1
                report_rows.append({"path": rel, "action": "move", "reason": c["cleanup_reason"], "dst": str(dst.relative_to(aligned))})
            except Exception as e:
                report_rows.append({"path": rel, "action": f"error:{e}", "reason": c["cleanup_reason"], "dst": ""})
        else:
            report_rows.append({"path": rel, "action": "would_move", "reason": c["cleanup_reason"], "dst": str(dst.relative_to(aligned))})

    write_csv(stats_dir / "cleanup_report.csv", ["path","action","reason","dst"], report_rows)
    if args.apply:
        print(f"[apply] moved {moved} files. Report -> {stats_dir/'cleanup_report.csv'}")
    else:
        print(f"[dry-run] {len([r for r in report_rows if r['action']=='would_move'])} files would move. See {stats_dir/'cleanup_report.csv'}")

if __name__ == "__main__":
    main()
