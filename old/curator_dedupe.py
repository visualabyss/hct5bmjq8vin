from pathlib import Path

import os
# Silence progress output when requested by caller
if os.getenv("PROGRESS_MUTE","0").lower() in ("1","true","yes","on"):
    try:
        import progress as _pg
        class _SilentBar:
            def __init__(self, *a, **k): pass
            def update(self, *a, **k): pass
            def close(self): pass
        _pg.ProgressBar = _SilentBar
    except Exception:
        pass
#!/usr/bin/env python3
import shutil, argparse, os, sys, shutil, re, math, time
import shutil
from pathlib import Path
from typing import shutil, List, Dict, Tuple
import shutil, cv2
import shutil, numpy as np


# === NO-MOVE POLICY (project override) ===
import shutil, os
def _no_move(src, dst, *a, **kw):
    # Decision-only mode: do not move anything; return intended dst for logging if needed
    return dst
shutil.move = _no_move
# If code still calls os.rename, make it a no-op via wrapper name (kept unreachable)
_RENAMES_DISABLED__ = lambda *a, **k: None
from curator_lib.io import shutil, ensure_dir, write_csv, read_csv, load_yaml, safe_glob

IMAGE_EXTS = ('.jpg','.jpeg','.png','.webp','.bmp')

def list_images(root: Path) -> List[Path]:
    out = []
    for ext in IMAGE_EXTS:
        out.extend(root.rglob(f"*{ext}"))
    # exclude bad/dupe automatically
    out = [p for p in out if "aligned/bad/dupe" not in str(p).replace("\\","/")]
    return sorted(out)

def to_gray_small(img: np.ndarray, size: int = 32) -> np.ndarray:
    if img is None or img.size == 0:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def phash(img: np.ndarray, dct_size: int = 8) -> int:
    g = to_gray_small(img, 32)
    if g is None:
        return 0
    g = np.float32(g)
    dct = cv2.dct(g)
    d = dct[:dct_size, :dct_size]
    m = d.mean()
    bits = (d > m).flatten().astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)

def ahash(img: np.ndarray) -> int:
    g = to_gray_small(img, 8)
    if g is None:
        return 0
    m = g.mean()
    bits = (g > m).flatten().astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)

def dhash(img: np.ndarray) -> int:
    g = to_gray_small(img, 9)
    if g is None:
        return 0
    diff = g[:,1:] > g[:,:-1]
    bits = diff.flatten().astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)

def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())

def load_image(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def numeric_key(path: Path) -> int:
    m = re.search(r"(\d+)", path.name)
    return int(m.group(1)) if m else 0

def read_shots_csv(p: Path) -> Dict[str, str]:
    rows = read_csv(p)
    out = {}
    for r in rows:
        key = r.get("path") or r.get("file") or r.get("frame") or ""
        if not key:
            continue
        shot = r.get("shot") or r.get("shot_id") or r.get("scene") or ""
        if not shot:
            continue
        out[key] = str(shot)
    return out

def choose_keep(a: Path, b: Path, id_map: Dict[str,float], q_map: Dict[str,float]) -> Tuple[Path, Path]:
    ka = id_map.get(a.name, id_map.get(a.as_posix(), 0.0))
    kb = id_map.get(b.name, id_map.get(b.as_posix(), 0.0))
    qa = q_map.get(a.name, q_map.get(a.as_posix(), 0.0))
    qb = q_map.get(b.name, q_map.get(b.as_posix(), 0.0))
    sa = ka + qa
    sb = kb + qb
    if sa > sb: return (a, b)
    if sb > sa: return (b, a)
    return (a, b) if a.name <= b.name else (b, a)

# -------- In-memory pseudo-shot grouping (no files written) --------
def luminance_hist(img: np.ndarray, bins: int = 32) -> np.ndarray:
    if img is None or img.size == 0:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([img],[0],None,[bins],[0,256])
    h = h.astype(np.float32)
    s = h.sum()
    if s > 0: h /= s
    return h

def hist_distance(h1: np.ndarray, h2: np.ndarray, metric: str = "bhatt") -> float:
    if h1 is None or h2 is None:
        return 1.0
    if metric == "corr":
        # correlation in [-1,1]; convert to distance in [0,1]
        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        return 1.0 - max(-1.0, min(1.0, corr)) * 0.5 - 0.5
    if metric == "chisq":
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR))
    # default: Bhattacharyya (0=identical, 1=far)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

def derive_pseudo_shots(images: List[Path], thresh: float = 0.35, min_len: int = 8, metric: str = "bhatt") -> Dict[Path, str]:
    # sort by numeric_key to reflect temporal order
    images = sorted(images, key=numeric_key)
    feats = []
    for p in images:
        img = load_image(p)
        feats.append(luminance_hist(img, bins=32))
    shots = {}
    shot_idx = 1
    start = 0
    def label(i): return f"S{int(i):04d}"
    shots[images[0]] = label(shot_idx) if images else ""
    for i in range(1, len(images)):
        d = hist_distance(feats[i-1], feats[i], metric=metric)
        # enforce minimum length; if segment is long enough and distance large, start new
        if (i - start) >= min_len and d > thresh:
            shot_idx += 1
            start = i
        shots[images[i]] = label(shot_idx)
    return shots

def _write_help_txt(parser, details:str):
    try:
        txt = parser.format_help()
    except Exception:
        txt = '(no argparse help available)\n'
    out = Path('/workspace/scripts') / (Path(__file__).name + '.txt')
    body = []
    body.append('=== Purpose ===\ncurator_dedupe: compute dupes/near-dupes WITHOUT moving files.\n\n')
    body.append('=== Policy ===\n* All file moves are permanently DISABLED.\n')
    body.append('  Use the CSV/JSONL decisions with curator_tag / curator_build downstream.\n\n')
    body.append('=== Usage ===\n'); body.append(txt)
    body.append('\n=== Details ===\n'); body.append(details.strip()+"\n")
    out.write_text(''.join(body), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser(description="curator_dedupe — hashing (a/p/d) + temporal guard; safe moves to aligned/bad/dupe")
    ap.add_argument("--mode", choices=["primary","target"], required=True, help="primary=global dedupe; target=scene/shot-aware + temporal guard")
    ap.add_argument("--root", required=True, help="Aligned directory to dedupe (e.g., data_src/aligned or data_trg/aligned)")
    ap.add_argument("--logs", required=True, help="Logs root for this dataset (no new folders)")
    ap.add_argument("--shots_csv", default="", help="Optional path to logs/video/shots.csv for target mode")
    ap.add_argument("--auto_shot", type=int, default=1, help="If 1 and no shots_csv, derive pseudo-shots in memory")
    ap.add_argument("--shot_thresh", type=float, default=0.35, help="Histogram distance threshold to split pseudo-shots")
    ap.add_argument("--shot_min_len", type=int, default=8, help="Minimum frames per pseudo-shot")
    ap.add_argument("--shot_metric", choices=["bhatt","corr","chisq"], default="bhatt", help="Histogram distance metric for pseudo-shots")
    ap.add_argument("--apply", choices=["on","off"], default="off", help="(disabled) decisions only; no file moves")
    ap.add_argument("--ham_thresh", type=int, default=6, help="Global duplicate threshold on pHash Hamming distance (<= means duplicate)")
    ap.add_argument("--temporal_ham_thresh", type=int, default=4, help="Temporal guard Hamming threshold for consecutive frames")
    ap.add_argument("--temporal_window", type=int, default=3, help="Compare each frame to the next N frames in same shot")
    args = ap.parse_args()

    root = Path(args.root)
    logs = Path(args.logs)
    stats_dir = ensure_dir(logs / "stats")

    # Optional ID/quality maps for tie-breaks
    id_rows = read_csv(logs / "arcface" / "arcface.csv")
    q_rows  = read_csv(logs / "magface" / "magface.csv")
    id_map = { (r.get("path") or r.get("file") or ""): float(r.get("id_score", "0") or 0) for r in id_rows }
    id_map.update({ Path(k).name: v for k, v in id_map.items() })
    q_map  = { (r.get("path") or r.get("file") or ""): float(r.get("quality", "0") or 0) for r in q_rows }
    q_map.update({ Path(k).name: v for k, v in q_map.items() })

    images = list_images(root)
    if not images:
        print("[warn] no images found to dedupe")
        return

    # Precompute hashes
    print(f"[hash] computing hashes for {len(images)} files...")
    H = {}
    for p in images:
        img = load_image(p)
        if img is None:
            continue
        H[p] = (phash(img), ahash(img), dhash(img))

    to_drop = set()
    report_rows = []
    prefix_map = {}
    for p, (phv, ahv, dhv) in H.items():
        prefix = phv >> 48
        prefix_map.setdefault(prefix, []).append(p)

    def mark_duplicate(a: Path, b: Path, reason: str):
        keep, drop = choose_keep(a, b, id_map, q_map)
        if drop in to_drop:
            return
        to_drop.add(drop)
        rel = drop.relative_to(root).as_posix()
        report_rows.append({"path": rel, "action": "move", "reason": reason, "kept": keep.relative_to(root).as_posix()})

    # GLOBAL duplicates (within pHash buckets)
    for prefix, plist in prefix_map.items():
        n = len(plist)
        for i in range(n):
            pi = plist[i]
            if pi in to_drop: 
                continue
            phi, ahi, dhi = H[pi]
            for j in range(i+1, n):
                pj = plist[j]
                if pj in to_drop:
                    continue
                phj, ahj, dhj = H[pj]
                hph = hamming(phi, phj)
                if hph <= args.ham_thresh:
                    hah = hamming(ahi, ahj)
                    hdh = hamming(dhi, dhj)
                    if min(hah, hdh) <= args.ham_thresh + 2:
                        mark_duplicate(pi, pj, reason=f"global_hamming:{hph}/{min(hah,hdh)}")

    # TEMPORAL GUARD (target mode): in-memory pseudo-shots if needed
    if args.mode == "target":
        # try explicit shots.csv
        shots_map = {}
        scsv = Path(args.shots_csv) if args.shots_csv else (logs / "video" / "shots.csv")
        if scsv.exists():
            rows = read_shots_csv(scsv)
            # map back to Path
            # group images by shot
            groups = {}
            for p in images:
                key = p.relative_to(root).as_posix()
                shot = rows.get(key, rows.get(p.name, "default"))
                groups.setdefault(shot, []).append(p)
        else:
            # derive pseudo-shots in memory
            groups = {}
            pseudo = derive_pseudo_shots(images, thresh=args.shot_thresh, min_len=args.shot_min_len, metric=args.shot_metric) if args.auto_shot else {p:"default" for p in images}
            for p, sid in pseudo.items():
                groups.setdefault(sid, []).append(p)

        for shot, plist in groups.items():
            plist.sort(key=numeric_key)
            m = len(plist)
            for i in range(m):
                pi = plist[i]
                if pi in to_drop: 
                    continue
                phi, ahi, dhi = H.get(pi, (None,None,None))
                if phi is None:
                    continue
                for k in range(1, args.temporal_window+1):
                    j = i + k
                    if j >= m: break
                    pj = plist[j]
                    if pj in to_drop: 
                        continue
                    phj, ahj, dhj = H.get(pj, (None,None,None))
                    if phj is None: 
                        continue
                    hph = hamming(phi, phj)
                    if hph <= args.temporal_ham_thresh:
                        mark_duplicate(pi, pj, reason=f"temporal_hamming:{hph}@{shot}")

    # WRITE REPORT
    report = stats_dir / "dedupe_report.csv"
    write_csv(report, ["path","action","reason","kept"], report_rows)
    print(f"[report] {report} — candidates: {len(report_rows)}")

    # MOVE FILES (APPLY)
    if False and args.apply and to_drop:
        bad_dupe = root / "bad" / "dupe"
        bad_dupe.mkdir(parents=True, exist_ok=True)
        moved = 0
        for p in sorted(to_drop, key=lambda x: x.as_posix()):
            rel = p.relative_to(root)
            dst = bad_dupe / rel.name
            try:
                shutil.move(str(p), str(dst))
                moved += 1
            except Exception as e:
                print(f"[warn] move failed for {p}: {e}", file=sys.stderr)
        print(f"[apply] moved {moved} files to {bad_dupe}")
    else:
        print("[dry-run] no files moved (use --apply 1 to move)")

if __name__ == "__main__":
    main()
