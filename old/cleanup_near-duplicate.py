#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path
import numpy as np, cv2

def move_safe(dst_dir: Path, src_path: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src_path.name
    if target.exists():
        stem, ext = src_path.stem, src_path.suffix
        i = 1
        while True:
            cand = dst_dir / f"{stem}_{i}{ext}"
            if not cand.exists():
                target = cand; break
            i += 1
    shutil.move(str(src_path), str(target))

# ------------- pHash (perceptual hash) -------------
def phash64(img):
    """Return 64‑bit perceptual hash (OpenCV only)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = img if img.ndim==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA)
    small = np.float32(small)
    dct = cv2.dct(small)              # 32x32 DCT
    dct_low = dct[:8,:8]              # top‑left 8x8
    med = np.median(dct_low[1:].ravel())  # exclude DC term
    bits = (dct_low > med).astype(np.uint8).ravel()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return np.uint64(h)

def hamming(a: np.uint64, b: np.uint64) -> int:
    return int(bin(int(a ^ b)).count("1"))

# ------------- optional ArcFace path (kept for later) -------------
def get_recognizer():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640,640))
        rec = app.models.get("recognition", None)
        return rec
    except Exception:
        return None

def emb112(rec, img):
    try:
        crop = cv2.resize(img, (112,112), interpolation=cv2.INTER_AREA)
        e = rec.get(crop)
        e = np.asarray(e).astype(np.float32)
        return e if e.size>0 else None
    except Exception:
        return None

def is_dupe_cos(emb, kept_embs, thresh):
    if not kept_embs: return False
    K = np.stack(kept_embs, axis=0)
    sims = (K @ emb) / (np.linalg.norm(K,axis=1)* (np.linalg.norm(emb)+1e-8))
    return (1.0 - float(sims.max())) < thresh

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="Near‑duplicate removal for aligned crops (pHash or ArcFace).")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dupes", default="", help="Destination for dupes (default: <src>/dupes)")
    ap.add_argument("--dst", default="", help="Alias for --dupes")
    ap.add_argument("--mode", choices=["phash","arcface"], default="phash",
                    help="Duplicate test: perceptual hash (fast, robust) or ArcFace cosine.")
    ap.add_argument("--thresh", type=float, default=0.06, help="ArcFace cosine distance threshold.")
    ap.add_argument("--phash_hd", type=int, default=6, help="Max Hamming distance to count as duplicate (0‑64).")
    ap.add_argument("--print_every", type=int, default=500)
    ap.add_argument("--dry_run", action="store_true", help="Preview only; no moves.")
    args = ap.parse_args()

    SRC = Path(args.src)
    DUP = Path(args.dupes or args.dst) if (args.dupes or args.dst) else SRC / "dupes"

    exts = {".png",".jpg",".jpeg",".webp",".bmp"}
    files = [p for p in sorted(SRC.iterdir())
             if p.is_file() and p.suffix.lower() in exts
             and "bad" not in p.parts and "dupes" not in p.parts and "pose" not in p.parts]
    if not files:
        print(f"[ERR] No images in {SRC}"); return

    kept = moved = 0

    if args.mode == "phash":
        kept_hashes = []
        for i, fp in enumerate(files, 1):
            img = cv2.imread(str(fp))
            if img is None:
                moved += 1
                if not args.dry_run: move_safe(DUP, fp)
                continue
            h = phash64(img)
            dupe = any(hamming(h, kh) <= args.phash_hd for kh in kept_hashes)
            if dupe:
                moved += 1
                if not args.dry_run: move_safe(DUP, fp)
            else:
                kept_hashes.append(h)
                kept += 1
            if args.print_every and (i % args.print_every == 0):
                print(f"[INFO] {i}/{len(files)} | kept={kept} dupes={moved} (pHash)")
    else:
        rec = get_recognizer()
        if rec is None:
            print("[FATAL] ArcFace recognizer unavailable; install/configure InsightFace OR use --mode phash.")
            return
        kept_embs = []
        for i, fp in enumerate(files, 1):
            img = cv2.imread(str(fp))
            if img is None:
                moved += 1
                if not args.dry_run: move_safe(DUP, fp)
                continue
            e = emb112(rec, img)
            if e is None:
                kept += 1  # keep if no embedding
                continue
            if is_dupe_cos(e, kept_embs, args.thresh):
                moved += 1
                if not args.dry_run: move_safe(DUP, fp)
            else:
                kept_embs.append(e)
                kept += 1
            if args.print_every and (i % args.print_every == 0):
                print(f"[INFO] {i}/{len(files)} | kept={kept} dupes={moved} (ArcFace)")
    print(f"[DONE] total={len(files)} | kept={kept} dupes moved={moved} -> {DUP}")
    if args.dry_run:
        print("[DRY-RUN] Preview only — no files were moved.")

if __name__ == "__main__":
    main()
