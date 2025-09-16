#!/usr/bin/env bash
set -euo pipefail

# Canonical dataset roots
SRC_DIR="/workspace/data_src/source"
TGT_DIR="/workspace/data_trg/target"

mkdir -p "$SRC_DIR" "$TGT_DIR"

# Keep /workspace anchors for back-compat, but they point to canonicals
ensure_anchor () {
  local anchor="$1"   # e.g., /workspace/source
  local target="$2"   # e.g., /workspace/data_src/source
  if [[ -e "$anchor" && ! -L "$anchor" ]]; then
    ts=$(date +%Y%m%d-%H%M%S)
    mv "$anchor" "${anchor}.bak-${ts}"
    echo "[INFO] Backed up existing $(basename "$anchor") to ${anchor}.bak-${ts}"
  fi
  ln -snf "$target" "$anchor"
}
ensure_anchor /workspace/source "$SRC_DIR"
ensure_anchor /workspace/target "$TGT_DIR"

# DIRECT convenience links at root → canonical paths (no intermediate hop)
ln -snf "$SRC_DIR" /source
ln -snf "$TGT_DIR" /target

# Helpers to create *_video links (first video found, recursive)
make_video_link () {
  local dir="$1"; local link="$2"
  local first
  first=$(find "$dir" -type f \
    \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.webm' -o -iname '*.mpg' -o -iname '*.mpeg' \) \
    -print -quit 2>/dev/null || true)
  if [[ -n "$first" ]]; then
    ln -snf "$first" "$link"
    echo "[OK] $(basename "$link") -> $(readlink -f "$link")"
  else
    [[ -L "$link" ]] && rm -f "$link"
    echo "[WARN] No videos under: $dir"
  fi
}

# Update video convenience links in both /workspace and /
make_video_link "$SRC_DIR" /workspace/source_video
make_video_link "$TGT_DIR" /workspace/target_video
ln -snf /workspace/source_video /source_video
ln -snf /workspace/target_video /target_video

echo "[OK] source -> $(readlink -f /source)"
echo "[OK] target -> $(readlink -f /target)"
