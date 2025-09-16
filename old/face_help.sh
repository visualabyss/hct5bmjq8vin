#!/usr/bin/env bash
# Colors
TITLE_RED='\033[31m'      # titles
PATH_GREEN='\033[32m'     # paths
RESET='\033[0m'

awk -v TITLE_RED="$TITLE_RED" -v PATH_GREEN="$PATH_GREEN" -v RESET="$RESET" '
function color_paths(s,   out, mstart, mlen, pre, matchstr) {
  out = ""
  while (match(s, /\/[[:alnum:]_\/\.\-]+/)) {
    mstart = RSTART; mlen = RLENGTH
    pre = substr(s, 1, mstart-1)
    matchstr = substr(s, mstart, mlen)
    out = out pre PATH_GREEN matchstr RESET
    s = substr(s, mstart + mlen)
  }
  return out s
}

# Return a copy with leading comment marker removed (for detection only)
function strip_hash(s) {
  sub(/^[[:space:]]*#?[[:space:]]*/, "", s)
  return s
}

# Detect a title:
#  - numbered headers like: "5) STRIDE — KEEP..." or "7) SCORE FILTER — ..."
#  - OR all-caps-ish headers (no lowercase letters) like "MOUNT"
function is_title(orig,  s) {
  s = strip_hash(orig)
  if (s ~ /^[0-9]+\)[[:space:]]+/) return 1
  if (s !~ /[a-z]/ && s ~ /[A-Z0-9]/) return 1
  return 0
}

{
  if (is_title($0)) {
    print TITLE_RED $0 RESET
  } else {
    print color_paths($0)
  }
}
' /workspace/scripts/COMMANDS.txt
