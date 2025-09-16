#!/usr/bin/env bash
set -euo pipefail

# OpenFace runner with:
#   - bbox: generate centered bboxes for aligned crops (writes logs/openface_bbox)
#   - img : extract AU/gaze/pose CSVs for aligned crops (uses bbox if present; writes logs/openface_img)

usage() {
  cat <<EOF
Usage:
  openface.sh bbox <aligned_dir> [logs_dir]
  openface.sh img  <aligned_dir> [logs_dir]

Defaults:
  logs_dir: /workspace/data_src/logs

Outputs:
  logs_dir/openface_bbox/*.txt   (from bbox)
  logs_dir/openface_img/*.csv    (from img)
EOF
}

# --- find OpenFace binaries ---
find_bin() {
  local name="$1"
  local -a CAND=(
    "/usr/local/bin/${name}"
    "/opt/OpenFace/build/bin/${name}"
    "/opt/openface/build/bin/${name}"
  )
  for p in "${CAND[@]}"; do
    if [[ -x "$p" ]]; then echo "$p"; return 0; fi
  done
  return 1
}

FACEIMG="$(find_bin FaceLandmarkImg || true)"

cmd="${1:-}"; shift || true
case "${cmd}" in
  bbox)
    ALIGNED="${1:-}"; LOGS_DIR="${2:-/workspace/data_src/logs}"
    [[ -d "$ALIGNED" ]] || { echo "ERR: aligned_dir not found: $ALIGNED"; exit 2; }
    OUT="${LOGS_DIR%/}/openface_bbox"
    mkdir -p "$OUT"
    echo "[bbox] writing bbox txts to: $OUT"
    # Uses your existing bbox_gen_centered.py (centered boxes for aligned crops)
    python3 /workspace/scripts/bbox_gen_centered.py "$ALIGNED" "$OUT"
    echo "[bbox] done."
    ;;

  img)
    [[ -n "$FACEIMG" ]] || { echo "ERR: FaceLandmarkImg binary not found."; exit 3; }
    ALIGNED="${1:-}"; LOGS_DIR="${2:-/workspace/data_src/logs}"
    [[ -d "$ALIGNED" ]] || { echo "ERR: aligned_dir not found: $ALIGNED"; exit 2; }
    OF_OUT="${LOGS_DIR%/}/openface_img"
    BBOX_DIR="${LOGS_DIR%/}/openface_bbox"
    mkdir -p "$OF_OUT"

    echo "[img] FaceLandmarkImg: $FACEIMG"
    echo "[img] aligned_dir     : $ALIGNED"
    echo "[img] out_dir (csv)   : $OF_OUT"

    # Build command
    CMD=( "$FACEIMG" -fdir "$ALIGNED" -out_dir "$OF_OUT" -pose -gaze -aus -verbose -wild -nthreads "${NTHREADS:-8}" )
    # If we have bboxes, use them to bypass detector and avoid "Face too small" stalls
    if [[ -d "$BBOX_DIR" ]] && compgen -G "$BBOX_DIR/*.txt" >/dev/null; then
      echo "[img] using bbox_dir  : $BBOX_DIR"
      CMD+=( -bboxdir "$BBOX_DIR" )
    else
      echo "[img] bbox_dir not found or empty; proceeding without -bboxdir"
    fi

    "${CMD[@]}"
    echo "[img] done. CSVs in: $OF_OUT"
    ;;

  -h|--help|"")
    usage
    ;;

  *)
    echo "ERR: unknown subcommand: $cmd"; usage; exit 1;;
esac
