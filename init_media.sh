#!/usr/bin/env bash
set -euo pipefail

# Ensure canonical roots exist
SRC_DIR="/workspace/data_src/source"
TGT_DIR="/workspace/data_trg/target"
mkdir -p "$SRC_DIR" "$TGT_DIR"

# Refresh anchors and convenience links
if [[ -x /workspace/scripts/refresh_source_target_links.sh ]]; then
  /workspace/scripts/refresh_source_target_links.sh
fi

# Harden: force root links to point DIRECTLY to canonicals (no /workspace hop)
ln -snf "$SRC_DIR" /source
ln -snf "$TGT_DIR" /target

# Show
echo "[SHOW] resolved links:"
ls -l /source /target
echo "$(readlink -f /source)"
echo "$(readlink -f /target)"
