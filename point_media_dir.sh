#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   point_media_dir.sh source /mnt/mydisk/somefolder
#   point_media_dir.sh target /media/ssd/projectA
# Optional:
#   point_media_dir.sh source /path --allow-empty

ROLE="${1:-}"
NEW_DIR="${2:-}"
ALLOW_EMPTY="${3:-}"

if [[ "$ROLE" != "source" && "$ROLE" != "target" ]]; then
  echo "Usage: $0 <source|target> <absolute_path> [--allow-empty]" >&2
  exit 1
fi

if [[ -z "${NEW_DIR}" || "${NEW_DIR:0:1}" != "/" ]]; then
  echo "[FATAL] Provide an ABSOLUTE path to the external folder." >&2
  exit 1
fi

if [[ ! -d "$NEW_DIR" ]]; then
  echo "[FATAL] Directory does not exist: $NEW_DIR" >&2
  exit 1
fi

# Check for at least one video unless --allow-empty
has_video=$(find "$NEW_DIR" -type f \
  \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.webm' -o -iname '*.mpg' -o -iname '*.mpeg' \) \
  -print -quit | wc -l)
if [[ "$has_video" -eq 0 && "$ALLOW_EMPTY" != "--allow-empty" ]]; then
  echo "[WARN] No video files detected in: $NEW_DIR"
  echo "       Re-run with --allow-empty to proceed anyway."
  exit 2
fi

if [[ "$ROLE" == "source" ]]; then
  CANON="/workspace/data_src/source"
  VIDEO_LINK="/workspace/source_video"
else
  CANON="/workspace/data_trg/target"
  VIDEO_LINK="/workspace/target_video"
fi

# If canonical exists and is a real dir with files, back it up
if [[ -e "$CANON" && ! -L "$CANON" && -n "$(ls -A "$CANON" 2>/dev/null || true)" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  mv "$CANON" "${CANON}.bak-${ts}"
  echo "[INFO] Backed up existing $(basename "$CANON") dir to ${CANON}.bak-${ts}"
fi

# Point canonical location to the external dir
ln -snf "$NEW_DIR" "$CANON"
echo "[OK] ${ROLE^} canonical now: $CANON -> $NEW_DIR"

# Refresh anchors + convenience links
if [[ -x /workspace/scripts/refresh_source_target_links.sh ]]; then
  /workspace/scripts/refresh_source_target_links.sh
else
  echo "[INFO] Skipping refresh: /workspace/scripts/refresh_source_target_links.sh not found."
fi

# Show where the video pointer now resolves
if [[ -e "$VIDEO_LINK" ]]; then
  ls -l "$VIDEO_LINK"
else
  echo "[INFO] No $(basename "$VIDEO_LINK") link yet (no videos found under $CANON)."
fi
