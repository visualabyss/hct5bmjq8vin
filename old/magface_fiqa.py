#!/usr/bin/env python3
"""
MagFace FIQA scorer (GPU if available).

Features:
- Tries InsightFace model zoo 'magface_r100' (ONNX). If not present, falls back to a local MagFace checkpoint (.pth).
- Computes mag_norm (quality), mag_cos_ref, mag_valid and writes them into video_logs.csv (atomic replace).
- Also writes an audit CSV to OUT_DIR/_tags/magface.csv.

Usage (typical):
  python3 -u /workspace/scripts/magface_fiqa.py \
    --dataset /workspace/data_src/aligned \
    --out_dir /workspace/data_src/curated

Optional:
  --ckpt /workspace/models/magface/magface_iresnet100_quality.pth  (if model zoo isn't available)
  --cos_thr 0.35  --batch 256
"""

from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
from typing import Callable, List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn

# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser("MagFace FIQA")
    p.add_argument("--dataset", required=True,
                   help="aligned/ folder or a folder that contains aligned/")
    p.add_argument("--out_dir", required=True,
                   help="root for outputs (audit CSV goes to out_dir/_tags)")
    p.add_argument("--logs", default="",
                   help="video_logs.csv (default: <dataset_parent>/logs/video_logs.csv)")
    p.add_argument("--ckpt", default="",
                   help="Optional MagFace checkpoint (.pth); used if model zoo is unavailable")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--cos_thr", type=float, default=0.35,
                   help="Cosine-to-reference validity threshold")
    p.add_argument("--no_zoo", action="store_true",
                   help="Skip InsightFace model zoo and force local --ckpt/.pth")
    return p.parse_args()

# --------------------- InsightFace / Torch backends ---------------------

def preprocess_bgr_for_onnx(img: np.ndarray) -> np.ndarray:
    # InsightFace ONNX expects BGR 112x112; normalization handled inside.
    return cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

def preprocess_bgr_for_torch(img: np.ndarray) -> np.ndarray:
    # iResNet100 style: RGB, 112x112, (x-127.5)/128 -> CHW float32
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    im = (im - 127.5) / 128.0
    im = np.transpose(im, (2, 0, 1))
    return im

def load_magface_via_zoo() -> Optional[Tuple[Callable, str]]:
    """
    Try InsightFace model zoo 'magface_r100' (ONNX).
    Returns (forward_fn, mode) or None if unavailable.
    """
    try:
        from insightface.model_zoo import get_model
        m = get_model('magface_r100')  # auto-downloads if available in your build
        if m is None:
            return None
        m.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        print("[MagFace] Using insightface model_zoo 'magface_r100' (ONNX).")
        def forward_onnx(batch_imgs_bgr_112: List[np.ndarray]) -> np.ndarray:
            feats = []
            for im in batch_imgs_bgr_112:
                feats.append(m.get_feat(im))  # (512,)
            return np.stack(feats, axis=0).astype(np.float32)
        return forward_onnx, "onnx"
    except Exception as e:
        print(f"[MagFace] Model zoo 'magface_r100' not available: {e}")
        return None

# Minimal iResNet-100 backbone to host MagFace state_dict (.pth)
class BasicBlockIR(nn.Module):
    def __init__(self, in_c, out_c, s):
        super().__init__()
        self.short = nn.Sequential()
        if in_c != out_c or s != 1:
            self.short = nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, out_c, 1, s, 0, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, 3, s, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
            nn.Conv2d(in_c if s!=1 else out_c, out_c, 3, 1, 1, bias=False) if False else nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.act = nn.PReLU(out_c)
    def forward(self, x):
        return self.act(self.body(x) + self.short(x))

def make_layer(in_c, out_c, n, s):
    blocks = [BasicBlockIR(in_c, out_c, s)]
    for _ in range(n - 1):
        blocks.append(BasicBlockIR(out_c, out_c, 1))
    return nn.Sequential(*blocks)

class IResNet100(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.l1 = make_layer(64, 64, 3, 2)
        self.l2 = make_layer(64, 128, 13, 2)
        self.l3 = make_layer(128, 256, 30, 2)
        self.l4 = make_layer(256, 512, 3, 2)
        self.bn = nn.BatchNorm2d(512)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(512 * 7 * 7, feat_dim, bias=False)
        self.out = nn.BatchNorm1d(feat_dim, affine=True)
    def forward(self, x):
        x = self.input(x); x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x); x = self.bn(x)
        x = torch.flatten(x, 1); x = self.drop(x); x = self.fc(x); x = self.out(x)
        return x


def load_state_dict_best_effort(model: nn.Module, ckpt: Path):
    """
    Load a MagFace state_dict but skip any keys that don't match our backbone
    (e.g., classifier heads like fc.weight [num_classes, 512]).
    """
    sd = torch.load(str(ckpt), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Strip common prefixes
    stripped = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "backbone.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        stripped[nk] = v

    md = model.state_dict()
    keep = {}
    dropped = []
    for k, v in stripped.items():
        if k in md and md[k].shape == v.shape:
            keep[k] = v
        else:
            # Skip non-matching (e.g., classifier heads, different shapes)
            try:
                vshape = tuple(v.shape)
            except Exception:
                vshape = None
            mshape = tuple(md[k].shape) if k in md else None
            dropped.append((k, vshape, mshape))

    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"[MagFace] Loaded {len(keep)} keys; skipped {len(dropped)}; "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if dropped:
        # Show a couple of the most likely culprits (classifier heads)
        show = dropped[:4]
        for k, vs, ms in show:
            print(f"[MagFace] SKIP {k}: ckpt {vs} != model {ms}")


def load_magface_via_ckpt(ckpt: Path, device: torch.device) -> Tuple[Callable, str]:
    net = IResNet100().to(device).eval()
    load_state_dict_best_effort(net, ckpt)
    print(f"[MagFace] Using state_dict: {ckpt}")
    def forward_torch(batch_chw_norm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(batch_chw_norm).to(device).float()
            return net(x).detach().cpu().numpy()
    return forward_torch, "torch"

def autodetect_ckpt() -> Optional[Path]:
    base = Path("/workspace/models/magface")
    if base.is_dir():
        pths = sorted(base.glob("*.pth"))
        if pths:
            return pths[0]
    return None

# --------------------------- CSV IO ---------------------------

def read_rows(csvp: Path) -> Tuple[list[dict], list[str]]:
    with csvp.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [dict(x) for x in r]
        fields = list(r.fieldnames or [])
    return rows, fields

def write_rows_atomic(csvp: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = csvp.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    tmp.replace(csvp)

# --------------------------- Main logic ---------------------------

def main():
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    if dataset.name.lower() != "aligned" and (dataset / "aligned").exists():
        dataset = dataset / "aligned"
    if dataset.name.lower() != "aligned":
        raise SystemExit(f"[FATAL] --dataset must be aligned/ or its parent: {dataset}")

    root = dataset.parent
    logs_path = Path(args.logs) if args.logs else (root / "logs" / "video_logs.csv")
    if not logs_path.exists():
        raise SystemExit(f"[FATAL] Missing {logs_path}")

    out_root = Path(args.out_dir).resolve()
    out_tags = out_root / "_tags"
    out_tags.mkdir(parents=True, exist_ok=True)
    out_csv = out_tags / "magface.csv"

    # Choose backend
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    forward, mode = (None, "")
    if not args.no_zoo:
        z = load_magface_via_zoo()
        if z is not None:
            forward, mode = z
    if forward is None:
        ckpt = Path(args.ckpt) if args.ckpt else autodetect_ckpt()
        if not ckpt or not ckpt.exists():
            raise SystemExit("[FATAL] No MagFace backend: model zoo unavailable AND no local --ckpt found.")
        forward, mode = load_magface_via_ckpt(ckpt, device)

    # Gather rows/files
    rows, fields = read_rows(logs_path)
    if not rows or "file" not in rows[0]:
        raise SystemExit("[FATAL] video_logs.csv missing 'file' column")
    files = [(r.get("file") or "").strip() for r in rows]
    keep_idx = [i for i, p in enumerate(files) if p and (dataset / p).is_file()]
    files = [files[i] for i in keep_idx]
    if not files:
        raise SystemExit("[FATAL] No aligned files found to score")

    # Embed in batches
    B = max(1, int(args.batch))
    feats: List[np.ndarray] = []
    batch_onnx: List[np.ndarray] = []
    batch_torch: List[np.ndarray] = []

    for i, rel in enumerate(files, 1):
        img = cv2.imread(str(dataset / rel), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if mode == "onnx":
            batch_onnx.append(preprocess_bgr_for_onnx(img))
            if len(batch_onnx) == B:
                feats.append(forward(batch_onnx)); batch_onnx = []
        else:
            batch_torch.append(preprocess_bgr_for_torch(img))
            if len(batch_torch) == B:
                feats.append(forward(np.stack(batch_torch, 0))); batch_torch = []
    if mode == "onnx" and batch_onnx:
        feats.append(forward(batch_onnx))
    if mode == "torch" and batch_torch:
        feats.append(forward(np.stack(batch_torch, 0)))

    if not feats:
        raise SystemExit("[FATAL] No embeddings produced — check backend/inputs")

    feats = np.concatenate(feats, axis=0).astype(np.float32)  # (N,512)
    if feats.shape[0] != len(files):
        print(f"[WARN] Embedding count {feats.shape[0]} != files {len(files)}; continuing with min")
        N = min(feats.shape[0], len(files)); feats = feats[:N]; keep_idx = keep_idx[:N]; files = files[:N]

    # MagFace quality = ||embedding||
    mag_norm = np.linalg.norm(feats, axis=1)

    # Build subject reference from top-K norms (robust when a single ID)
    K = max(5, int(0.02 * len(mag_norm)))
    topk = np.argpartition(-mag_norm, K - 1)[:K]
    ref = feats[topk].mean(axis=0)
    ref /= (np.linalg.norm(ref) + 1e-9)

    # Cosine to ref (normalize each feat)
    nfeats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    mag_cos_ref = (nfeats @ ref)
    mag_valid = (mag_cos_ref >= float(args.cos_thr)).astype(np.uint8)

    # Write back to logs (atomic), preserving order
    for j, irow in enumerate(keep_idx):
        rows[irow]["mag_norm"] = f"{mag_norm[j]:.6f}"
        rows[irow]["mag_cos_ref"] = f"{mag_cos_ref[j]:.6f}"
        rows[irow]["mag_valid"] = "1" if mag_valid[j] else "0"
    for col in ["mag_norm", "mag_cos_ref", "mag_valid"]:
        if col not in fields:
            fields.append(col)
    write_rows_atomic(logs_path, rows, fields)

    # Audit CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "mag_norm", "mag_cos_ref", "mag_valid"])
        w.writeheader()
        for j, irow in enumerate(keep_idx):
            w.writerow({
                "file": rows[irow].get("file", ""),
                "mag_norm": rows[irow].get("mag_norm", ""),
                "mag_cos_ref": rows[irow].get("mag_cos_ref", ""),
                "mag_valid": rows[irow].get("mag_valid", ""),
            })

    print(f"[OK] MagFace ({mode}) wrote mag_* -> {logs_path}; audit -> {out_csv}; N={len(keep_idx)}")

if __name__ == "__main__":
    main()
