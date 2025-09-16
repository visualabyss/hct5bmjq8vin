#!/usr/bin/env python3
import os, sys
from pathlib import Path
from PIL import Image

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 bbox_gen_centered.py <aligned_dir> <bbox_dir> [--frac_w 0.56] [--frac_h 0.70] [--y_bias 0.02]")
        sys.exit(2)
    aligned_dir = Path(sys.argv[1]); bbox_dir = Path(sys.argv[2])
    frac_w = 0.56; frac_h = 0.70; y_bias = 0.02
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--frac_w": frac_w = float(sys.argv[i+1])
        if sys.argv[i] == "--frac_h": frac_h = float(sys.argv[i+1])
        if sys.argv[i] == "--y_bias": y_bias = float(sys.argv[i+1])

    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    bbox_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in aligned_dir.iterdir() if p.suffix.lower() in exts]
    if not imgs:
        print("No images found in", aligned_dir); sys.exit(1)

    for p in imgs:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            print("Skip:", p.name, e); continue
        cx, cy = w * 0.5, h * (0.5 + y_bias)  # shift a hair downward
        bw, bh = w * frac_w, h * frac_h
        x0, y0 = int(max(0, cx - bw/2)), int(max(0, cy - bh/2))
        x1, y1 = int(min(w-1, cx + bw/2)), int(min(h-1, cy + bh/2))
        (bbox_dir / (p.stem + ".txt")).write_text(f"{x0} {y0} {x1} {y1}\n", encoding="utf-8")
    print("Wrote bbox txts to", bbox_dir)

if __name__ == "__main__":
    main()
