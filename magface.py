#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MagFace quality scorer (GPU via ONNX Runtime) over aligned images.

Small, surgical upgrade (keeps existing behavior):
- Auto-switch to mf_env (unchanged).
- **Auto-help**: `--help on|off` writes /workspace/scripts/magface.txt; `-h/--h` prints help.
- **Always extract everything useful** (no opt-ins):
  • CSV: logs/magface/magface.csv → path,quality (unchanged schema)
  • Artifacts: embeddings.npy (N×D pre-norm features), index.csv (row,file,ok), meta.json
- Progress bar label "MAGFACE"; uses shared progress.py dividers.
- Core quality: `||pre-norm embedding||` (MagFace), unchanged.
"""

import sys, csv, argparse, os, re, json, time
from pathlib import Path

# --- auto-switch to dedicated MagFace env (/workspace/envs/mf_env) ---

def _ensure_mf_env():
    mf_py = "/workspace/envs/mf_env/bin/python"
    if sys.executable != mf_py and os.path.exists(mf_py):
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.execv(mf_py, [mf_py, os.path.abspath(__file__), *sys.argv[1:]])

_ensure_mf_env()

# --- progress bar import (shared style) ---
try:
    from progress import ProgressBar
except Exception:  # fallback no-op
    class ProgressBar:
        def __init__(self, *a, **k): self.total = k.get("total", 1)
        def update(self, *a, **k): pass
        def close(self): pass

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from torch.jit import TracerWarning
    warnings.filterwarnings('ignore', category=TracerWarning)
except Exception:
    pass

# --------------------------- Auto-help text ---------------------------
AUTO_HELP_TEXT = r"""
PURPOSE
  • Score face image quality with **MagFace**: quality = L2 norm of the **pre-normalized** embedding.
  • Accepts either a **Torch checkpoint** (`--ckpt`) which is exported to ONNX on the fly, or a ready **ONNX** model (`--model`).
  • Runs on ONNX Runtime **GPU** (CUDA provider) in `/workspace/envs/mf_env`.
  • Outputs a lean CSV and richer artifacts for downstream analysis.

USAGE (common)
  python3 -u /workspace/scripts/magface.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --ckpt    /workspace/tools/magface_res100_backbone.pth \
    --batch 256

  # or, if you already have an ONNX that outputs **pre-norm** embeddings (N×512)
  python3 -u /workspace/scripts/magface.py \
    --aligned /workspace/data_src/aligned \
    --logs    /workspace/data_src/logs \
    --model   /workspace/tools/magface_export.onnx

CLI FLAGS
  --aligned DIR (req)               aligned faces dir (112×112 recommended)
  --logs DIR (req)                  log root (writes logs/magface/*)
  --ckpt FILE                       MagFace Torch checkpoint (.pth); exported to ONNX if newer than cached export
  --model FILE                      MagFace ONNX (outputs **pre-norm** embeddings). Ignored if --ckpt is provided
  --batch INT (256)                 batch size for ONNX inference

  # logging & help
  --help on|off (on)                write /workspace/scripts/magface.txt then proceed
  -h / --h                          print this help text and continue

OUTPUTS
  logs/magface/magface.csv          (append mode; header if new) schema: path,quality
  logs/magface/embeddings.npy       float32 N×D pre-norm embeddings (ordered to CSV; typically D=512)
  logs/magface/index.csv            row,file,ok (1 if embedded)
  logs/magface/meta.json            source (ckpt/model), onnx path, dims, providers, batch, counts
"""

# --------------------------- helpers ---------------------------
IMAGE_EXTS={".jpg",".jpeg",".png",".webp",".bmp"}


def _parse_onoff(v, default=False):
    if v is None: return bool(default)
    if isinstance(v,(int,float)): return bool(int(v))
    s=str(v).strip().lower()
    if s in {"1","on","true","yes","y"}: return True
    if s in {"0","off","false","no","n"}: return False
    return bool(default)


def _prep_dirs(args):
    aligned = Path(args.aligned)
    logs = Path(args.logs)
    out_dir = logs / "magface"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "magface.csv"
    return aligned, out_dir, out_csv


# ---------------- Build backbone from checkpoint ----------------

def _torch_build_from_ckpt(ckpt_path: str):
    import torch, torch.nn as nn, re
    # safer load if available
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']

    def _shape(k):
        v = sd.get(k); return tuple(v.shape) if hasattr(v, "shape") else None

    # derive channels from conv weights directly
    conv1_out = (_shape('features.module.conv1.weight') or (64,3,3,3))[0]
    s1_in = (_shape('features.module.layer2.0.conv1.weight') or (64, 64, 3, 3))[1]
    s2_in = (_shape('features.module.layer3.0.conv1.weight') or (128, 64, 3, 3))[1]
    s3_in = (_shape('features.module.layer4.0.conv1.weight') or (256,128, 3, 3))[1]
    s1 = (_shape('features.module.layer1.0.conv1.weight') or (64, 64, 3, 3))[0]
    s2 = (_shape('features.module.layer2.0.conv1.weight') or (128,64, 3, 3))[0]
    s3 = (_shape('features.module.layer3.0.conv1.weight') or (256,128,3, 3))[0]
    s4 = (_shape('features.module.layer4.0.conv1.weight') or (512,256,3, 3))[0]
    if s1 is None: s1 = s1_in or 64
    if s2 is None: s2 = s2_in or max(64, 2*s1)
    if s3 is None: s3 = s3_in or max(128,2*s2)
    if s4 is None: s4 = max(256, 2*s3)

    def _count_blocks(layer_idx:int):
        rx = re.compile(rf'^features\.module\.layer{layer_idx}\.([0-9]+)\.')
        idx = {int(m.group(1)) for k in sd.keys() if (m:=rx.match(k))}
        return (max(idx)+1) if idx else 0

    d1, d2, d3, d4 = _count_blocks(1), _count_blocks(2), _count_blocks(3), _count_blocks(4)
    if min(d1,d2,d3,d4) == 0:
        d1,d2,d3,d4 = (3,13,30,3) if "100" in str(ckpt_path) else (3,4,14,3)

    class SEModule(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                nn.PReLU(channels // reduction),
                nn.Conv2d(channels // reduction, channels, 1, bias=False),
                nn.Sigmoid(),
            )
        def forward(self, x):
            w = self.avg_pool(x); w = self.fc(w); return x * w

    class BottleneckIRSE(nn.Module):
        def __init__(self, in_ch, out_ch, stride):
            super().__init__()
            self.bn0 = nn.BatchNorm2d(in_ch)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.prelu = nn.PReLU(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.se = SEModule(out_ch, 16)
            self.short = nn.Sequential()
            if in_ch != out_ch or stride != 1:
                self.short = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )
        def forward(self, x):
            idn = self.short(x)
            y = self.bn0(x)
            y = self.conv1(y); y = self.bn1(y); y = self.prelu(y)
            y = self.conv2(y); y = self.bn2(y); y = self.se(y)
            return y + idn

    def make_layer(in_ch, out_ch, blocks, stride):
        layers = [BottleneckIRSE(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BottleneckIRSE(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    class IResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, conv1_out, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(conv1_out)
            self.prelu = nn.PReLU(conv1_out)
            self.layer1 = make_layer(conv1_out, s1, d1, 2)
            self.layer2 = make_layer(s1, s2, d2, 2)
            self.layer3 = make_layer(s2, s3, d3, 2)
            self.layer4 = make_layer(s3, s4, d4, 2)
            self.bn2 = nn.BatchNorm2d(s4)
            self.dropout= nn.Dropout(p=0.4)
            self.flatten= nn.Flatten()
            self.fc = nn.Linear(s4*7*7, 512, bias=False)
            self.features_bn = nn.BatchNorm1d(512)
        def forward(self, x, return_normed=False):
            import torch.nn.functional as F
            x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            x = self.bn2(x); x = self.dropout(x); x = self.flatten(x)
            feat = self.fc(x)
            feat = self.features_bn(feat)  # PRE-NORM feature
            if return_normed:
                feat = F.normalize(feat, dim=1)
            return feat

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Module(); self.features.module = IResNet()
        def forward(self, x, return_normed=False):
            return self.features.module(x, return_normed=return_normed)

    model = Wrapper()
    # only load keys whose shapes match (avoid size-mismatch errors)
    msd = model.state_dict()
    sd_fit = {k: v for k, v in sd.items() if k in msd and hasattr(v, "shape") and tuple(v.shape) == tuple(msd[k].shape)}
    model.load_state_dict(sd_fit, strict=False)
    print(f"[INFO] inferred (conv-based): s1={s1}, s2={s2}, s3={s3}, s4={s4} | depths: {d1},{d2},{d3},{d4} | conv1={conv1_out} | loaded {len(sd_fit)}/{len(msd)} tensors")
    model.eval()
    return model


def _export_ckpt_to_onnx(ckpt_path, onnx_out):
    """Load MagFace .pth on CPU, build matching backbone, and export ONNX that outputs PRE-NORM (N,512) feature."""
    import torch
    model = _torch_build_from_ckpt(ckpt_path)
    dummy = torch.randn(1,3,112,112, dtype=torch.float32, device="cpu")
    onnx_out = str(onnx_out)
    torch.onnx.export(
        model, dummy, onnx_out,
        input_names=["input"], output_names=["emb_pre"],
        dynamic_axes={"input": {0: "N"}, "emb_pre": {0: "N"}},
        opset_version=12
    )
    return onnx_out


# ---------------- ONNX streaming embedder ----------------

def _embed_stream_onnx(sess, np, cv2, dir_path, batch, iname):
    ex = IMAGE_EXTS
    paths = [p for p in Path(dir_path).glob("*") if p.suffix.lower() in ex]
    paths.sort()
    # infer H,W
    H,W = 112,112
    try:
        shp = sess.get_inputs()[0].shape
        if isinstance(shp,(list,tuple)) and len(shp)==4 and all(isinstance(v,int) for v in shp[2:4]):
            H,W = int(shp[2]), int(shp[3])
    except Exception:
        pass
    b = max(1, int(batch))
    for i in range(0, len(paths), b):
        attempted = len(paths[i:i+b])
        chunk = paths[i:i+b]
        ims, keep = [], []
        for pth in chunk:
            if not pth.exists():
                continue
            im = cv2.imread(str(pth), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.resize(im, (W,H), interpolation=cv2.INTER_AREA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32")
            im = (im - 127.5) / 128.0
            im = __import__("numpy").transpose(im, (2,0,1))
            ims.append(im); keep.append(pth.name)
        if not ims:
            yield [], None, attempted
            continue
        x = __import__("numpy").stack(ims, axis=0).astype("float32")
        out = sess.run(None, {iname: x})[0]  # (N,512) — pre-norm
        yield keep, out, attempted


# ------------------------------ main ------------------------------

def main():
    import numpy as np, cv2
    ap = argparse.ArgumentParser(description="MagFace quality = L2 norm of pre-norm feature (GPU inference via ORT)", add_help=False)
    ap.add_argument('-h','--h', dest='show_cli_help', action='store_true')
    ap.add_argument("--aligned", required=True, help="Aligned faces dir (112x112)")
    ap.add_argument("--logs", required=True, help="Logs root (writes logs/magface/magface.csv)")
    ap.add_argument("--ckpt", required=False, help="MagFace Torch checkpoint (.pth) — will be exported to ONNX")
    ap.add_argument("--model", required=False, help="MagFace ONNX (pre-norm output); skipped if --ckpt provided")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--help", dest="auto_help", default="on")
    args = ap.parse_args()

    if getattr(args,'show_cli_help', False):
        print(AUTO_HELP_TEXT)
    try:
        if _parse_onoff(args.auto_help, True):
            sp = Path(__file__).resolve(); (sp.with_name(sp.stem + '.txt')).write_text(AUTO_HELP_TEXT, encoding='utf-8')
    except Exception:
        pass

    aligned, out_dir, out_csv = _prep_dirs(args)

    # open CSV and header now (stable schema)
    is_new = not out_csv.exists()
    f = out_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if is_new:
        w.writerow(["path","quality"]); f.flush()

    # list files for totals/progress
    paths = [p for p in sorted(Path(aligned).iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    total = len(paths)
    bar = ProgressBar("MAGFACE", total=max(1,total), show_fail_label=True)

    # ---------- Resolve ONNX model path (export if ckpt provided) ----------
    onnx_path = None; exported = False
    try:
        if args.ckpt and str(args.ckpt).lower().endswith(".pth"):
            onnx_path = str(Path("/workspace/tools/magface_export.onnx"))
            ck = Path(args.ckpt); on = Path(onnx_path)
            if (not on.exists()) or (on.stat().st_mtime < ck.stat().st_mtime):
                print("[INFO] Exporting MagFace checkpoint to ONNX:", onnx_path)
                _export_ckpt_to_onnx(args.ckpt, onnx_path)
                exported = True
        elif args.model:
            onnx_path = args.model
    except Exception as e:
        print("[ERR] ONNX export failed:", e)

    # ---------- GPU inference with ONNXRuntime ----------
    wrote = 0
    E = None; OK = None; D = 0
    idx_rows = []
    providers_used = []

    if onnx_path:
        try:
            import onnxruntime as ort
            providers = []
            avail = set(ort.get_available_providers())
            for p in ("CUDAExecutionProvider","CPUExecutionProvider"):
                if p in avail: providers.append(p)
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError("onnxruntime-gpu not available. Install it in mf_env.")
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            providers_used = list(providers)
            ii = sess.get_inputs()[0]; iname = ii.name
            b = max(1, int(args.batch))
            t0 = time.time(); done = 0; fails_total = 0

            for files, vecs, attempted in _embed_stream_onnx(sess, np, cv2, str(aligned), b, iname):
                if vecs is None or len(files)==0:
                    done += attempted; fails_total += attempted
                    bar.update(min(done, total), fails=fails_total)
                    continue
                if E is None:
                    D = int(vecs.shape[1])
                    E = __import__('numpy').zeros((total, D), dtype='float32')
                    OK = __import__('numpy').zeros((total,), dtype='uint8')
                mags = __import__('numpy').linalg.norm(vecs, axis=1)  # quality = ||pre-norm||
                for fn, q in zip(files, mags):
                    # find row index by filename order
                    # Assuming no missing files in list; map index once
                    # (simple linear map since generator emits in sorted order)
                    idx = wrote
                    w.writerow([fn, f"{float(q):.6f}"]); f.flush()
                    if E is not None and OK is not None and idx < total:
                        try:
                            E[idx,:] = vecs[idx - wrote if (idx - wrote) >= 0 else 0]
                            OK[idx] = 1
                        except Exception:
                            pass
                    idx_rows.append((idx, fn, 1))
                    wrote += 1
                fps = 0 if (time.time()-t0)<=0 else done/(time.time()-t0)
                done += attempted
                fails_total += max(attempted - len(files), 0)
                bar.update(min(done, total), fails=fails_total, fps=fps)
            bar.close()
        except Exception as e:
            print("[WARN] ONNX MagFace path failed:", e)

    f.close()

    # Save artifacts
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # index.csv
        with (out_dir/"index.csv").open("w", newline="", encoding="utf-8") as fi:
            iw = csv.writer(fi); iw.writerow(["row","file","ok"])
            if idx_rows:
                for r in idx_rows:
                    iw.writerow(r)
            else:
                # still write a header-only file if nothing processed
                pass
        # embeddings.npy
        if E is not None:
            import numpy as _np
            _np.save(out_dir/"embeddings.npy", E)
        # meta.json
        meta = {
            "source": "magface",
            "ckpt": str(args.ckpt) if args.ckpt else None,
            "model": str(args.model) if args.model else None,
            "onnx_path": str(onnx_path) if onnx_path else None,
            "exported": bool(exported),
            "providers": providers_used,
            "batch": int(args.batch),
            "count_total": int(total),
            "count_written": int(wrote),
            "embedding_dim": int(D or 0),
            "csv_path": str(out_csv),
        }
        (out_dir/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    if wrote == 0:
        print(f"[WARN] No embeddings produced. CSV header only: {out_csv}")
    else:
        print(f"[OK] wrote to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
