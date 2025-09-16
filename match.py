#!/usr/bin/env python3
import argparse, math, collections, json, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import csv

def read_csv_any(p: Path):
    if not p.exists():
        return [], []
    with p.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        return rows, list(rdr.fieldnames or [])

def truthy(x):
    s = str(x).strip().lower()
    return s in ("1","true","yes","y","t")

def detect_upscaled_set(logs_root: Path) -> set:
    \"\"\"
    Returns a set of dst_name (filenames) considered 'upscaled' by reading:
      logs/compile/video_logs.csv (per-input provenance rows)
      logs/compile/compile_map.csv (src_root+src_path -> dst_name mapping)
    Heuristics:
      - 'is_upscaled' or 'upscaled' truthy
      - 'scale' or 'scale_factor' > 1
      - 'type'/'kind' contains 'upscale'
    \"\"\"
    vlog = logs_root / "compile" / "video_logs.csv"
    cmap = logs_root / "compile" / "compile_map.csv"
    if not vlog.exists() or not cmap.exists():
        return set()
    vrows, _ = read_csv_any(vlog)
    mrows, _ = read_csv_any(cmap)
    # Build mapping keys
    map_a = {}  # (src_root, src_path) -> dst_name
    map_b = {}  # basename(src_path) -> dst_name
    for r in mrows:
        sr = str(r.get("src_root","")).rstrip("/")
        sp = str(r.get("src_path",""))
        dn = str(r.get("dst_name",""))
        if sr and sp and dn:
            map_a[(sr, sp)] = dn
            map_b[Path(sp).name] = dn
    up = set()
    for r in vrows:
        is_up = False
        if truthy(r.get("is_upscaled","")) or truthy(r.get("upscaled","")):
            is_up = True
        try:
            sc = float(r.get("scale","") or r.get("scale_factor","") or 1.0)
            if sc and sc > 1.0:
                is_up = True
        except Exception:
            pass
        kind = str(r.get("type","") or r.get("kind","")).lower()
        if "upscale" in kind:
            is_up = True
        if not is_up:
            continue
        # resolve to dst_name
        dn = r.get("dst_name","")
        if dn:
            up.add(str(dn))
            continue
        sr = str(r.get("src_root","")).rstrip("/")
        sp = str(r.get("src_path","") or r.get("path",""))
        dn = map_a.get((sr, sp), None) or map_b.get(Path(sp).name, None)
        if dn:
            up.add(str(dn))
    return up


from curator_lib.io import ensure_dir, read_csv, write_csv, append_jsonl, load_yaml

BIN_KEYS = ["bin_yaw","bin_pitch","bin_eyes","bin_smile","bin_teeth","bin_light"]
ID_COL = "id_score"
Q_COL  = "quality"

def load_tv_profile(logs_root: Path) -> Dict[str, int]:
    stats = logs_root / "stats" / "tv_profile.json"
    if stats.exists():
        try:
            data = json.loads(stats.read_text(encoding="utf-8"))
            bins = data.get("bins", {})
            out = collections.Counter()
            for k, n in bins.items():
                out[k] = int(n)
            return dict(out)
        except Exception:
            pass
    # fallback: compute from tags.csv
    tags = read_csv(logs_root / "tags.csv")
    counter = collections.Counter()
    for t in tags:
        tup = tuple(t.get(k,"") for k in BIN_KEYS)
        counter[str(tup)] += 1
    return dict(counter)

def key_from_row(row: dict) -> str:
    return str(tuple(row.get(k,"") for k in BIN_KEYS))

def pose_score(a: dict, b: dict) -> float:
    yaw_eq = int(a.get("bin_yaw","") == b.get("bin_yaw",""))
    pit_eq = int(a.get("bin_pitch","") == b.get("bin_pitch",""))
    if yaw_eq and pit_eq: return 1.0
    if yaw_eq or pit_eq:  return 0.5
    return 0.0

def light_score(a: dict, b: dict) -> float:
    la, lb = a.get("bin_light",""), b.get("bin_light","")
    if la and lb and la == lb: return 1.0
    if la and lb and la[:1] == lb[:1]: return 0.5
    return 0.0

def expr_score(a: dict, b: dict) -> float:
    eyes = int(a.get("bin_eyes","") == b.get("bin_eyes",""))
    smile= int(a.get("bin_smile","") == b.get("bin_smile",""))
    teeth= int(a.get("bin_teeth","") == b.get("bin_teeth",""))
    return (eyes + smile + teeth) / 3.0

def scalar(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def row_score(sr: dict, target_proto: dict, w_pose: float, w_light: float, w_expr: float) -> float:
    s = w_pose*pose_score(sr, target_proto) + w_light*light_score(sr, target_proto) + w_expr*expr_score(sr, target_proto)
    s += 0.05 * scalar(sr.get(ID_COL,"0")) + 0.05 * scalar(sr.get(Q_COL,"0"))
    return s

def select_for_bin(candidates: List[dict], target_proto: dict, need: int, weights: Tuple[float,float,float]) -> List[dict]:
    if need <= 0 or not candidates:
        return []
    w_pose, w_light, w_expr = weights
    ranked = sorted(candidates, key=lambda r: row_score(r, target_proto, w_pose, w_light, w_expr), reverse=True)
    return ranked[:need]

def default_logs_for_source_dir(source_dir: Path) -> Path:
    # Prefer data_src/logs when source_dir is data_src/aligned
    cand = source_dir.parent / "logs"
    if cand.exists() or True:  # we will create if missing
        return cand
    return source_dir / "logs"

def main():
    ap = argparse.ArgumentParser(description="curator_match — build MATCHED SOURCE with coverage insurance + weighting")
    ap.add_argument("--source_dir", required=True, help="Aligned directory for PRIMARY SOURCE (e.g., data_src/aligned)")
    ap.add_argument("--source_logs", default="", help="Logs root for PRIMARY SOURCE (default: data_src/logs)")
    ap.add_argument("--target_logs", required=True, help="Logs root for TARGET VIDEO / ONE OFF TARGET / REPEATED TARGET")
    ap.add_argument("--config", default="", help="Optional path to curator.yaml")
    ap.add_argument("--coverage_min", type=int, default=-1, help="Coverage insurance per non-empty TARGET VIDEO bin (overrides YAML)")
    ap.add_argument("--budget", type=int, default=0, help="Total MATCHED SOURCE size. 0 = sum of TARGET VIDEO bin counts")
    ap.add_argument("--target_upscaled", type=int, default=-1, help="Max total upscaled frames allowed (overrides YAML target_upscaled)")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    source_logs = Path(args.source_logs) if args.source_logs else default_logs_for_source_dir(source_dir)
    target_logs = Path(args.target_logs)

    # Outputs fixed by policy: data_src/logs/match
    match_dir = ensure_dir(source_logs / "match")

    cfg = load_yaml([args.config, str(source_logs/"curator.yaml"), str(target_logs/"curator.yaml")])
    weights = cfg.get("match_weights", {"pose":0.45,"lighting":0.35,"expression":0.20})
    w_pose, w_light, w_expr = float(weights.get("pose",0.45)), float(weights.get("lighting",0.35)), float(weights.get("expression",0.20))

    coverage_min = args.coverage_min if args.coverage_min >= 0 else int(cfg.get("coverage_min_per_bin", 5))
    # Upscaled policy
    upscaled_cap = args.target_upscaled if args.target_upscaled >= 0 else int(cfg.get("target_upscaled", 0))
    upscaled_set = detect_upscaled_set(source_logs)


    # Load SOURCE tags (filter cleanup_flag == 0); ensure files exist
    src_tags = read_csv(source_logs / "tags.csv")
    src_tags = [r for r in src_tags if str(r.get("cleanup_flag","0")) in ("0","")]
    src_tags = [r for r in src_tags if (source_dir / r.get("path","")).exists()]

    if not src_tags:
        print("[fatal] no SOURCE tags found (after cleanup filter or files missing).", file=sys.stderr)
        sys.exit(2)

    # Load TARGET demand
    tv_bins = load_tv_profile(target_logs)
    if not tv_bins:
        print("[fatal] no TARGET profile/bins found.", file=sys.stderr)
        sys.exit(2)

    def proto_from_key(k: str) -> dict:
        try:
            tup = eval(k)
            return {k2: tup[i] for i, k2 in enumerate(BIN_KEYS)}
        except Exception:
            parts = k.strip("()").split(",")
            parts = [x.strip().strip("'").strip('"') for x in parts]
            while len(parts) < len(BIN_KEYS):
                parts.append("")
            return {k2: parts[i] for i, k2 in enumerate(BIN_KEYS)}

    src_by_key = collections.defaultdict(list)
    def key_from_row_local(r): return str(tuple(r.get(k,"") for k in BIN_KEYS))
    for r in src_tags:
        src_by_key[key_from_row_local(r)].append(r)

    selected: List[dict] = []
    selected_paths = set()

    total_demand = sum(int(n) for n in tv_bins.values())
    budget = args.budget if args.budget > 0 else total_demand

    # coverage insurance
    for k, n in tv_bins.items():
        if int(n) <= 0: 
            continue
        proto = proto_from_key(k)
        cand = list(src_by_key.get(k, []))
        if len(cand) < coverage_min:
            # expand to near matches
            for r in src_tags:
                if r in cand: 
                    continue
                if (r.get("bin_yaw","")==proto.get("bin_yaw","")) or (r.get("bin_pitch","")==proto.get("bin_pitch","")) or (r.get("bin_smile","")==proto.get("bin_smile","")):
                    cand.append(r)
                    if len(cand) >= coverage_min*3:
                        break
        take = select_for_bin(cand, proto, coverage_min, (w_pose,w_light,w_expr))
        for r in take:
            p = r.get("path","")
            if p not in selected_paths:
                selected.append(r); selected_paths.add(p)

    # proportional fill
    remaining = max(0, budget - len(selected))
    if remaining > 0:
        sel_counter = collections.Counter(key_from_row_local(r) for r in selected)
        total_bins = sum(int(n) for n in tv_bins.values())
        for k, n in sorted(tv_bins.items(), key=lambda kv: -int(kv[1])):
            if remaining <= 0: break
            quota = int(round(remaining * (int(n)/total_bins)))
            already = sel_counter.get(k, 0)
            proto = proto_from_key(k)
            cand = [r for r in src_by_key.get(k, []) if r.get("path","") not in selected_paths]
            take = select_for_bin(cand, proto, max(0, quota - already), (w_pose,w_light,w_expr))
            for r in take:
                p = r.get("path","")
                if p not in selected_paths:
                    selected.append(r); selected_paths.add(p)
                    remaining -= 1
                    if remaining <= 0: break

    
    # include is_upscaled flag in output
    rows_out = []
    for r in selected:
        p = str(r.get("path",""))
        rows_out.append({"path": p, "is_upscaled": int(is_upscaled_row(r))})
    matched_csv = match_dir / "matched_source.csv"
    write_csv(matched_csv, ["path","is_upscaled"], rows_out)

    # report
    rep_rows = []
    sel_counter = collections.Counter(key_from_row_local(r) for r in selected)
    for k, n in tv_bins.items():
        rep_rows.append({
            "bin_key": k,
            "target_count": int(n),
            "selected": int(sel_counter.get(k, 0)),
            "coverage_min": coverage_min,
            "coverage_met": int(sel_counter.get(k, 0) >= (coverage_min if int(n)>0 else 0))
        })
    write_csv(match_dir / "match_report.csv", ["bin_key","target_count","selected","coverage_min","coverage_met"], rep_rows)

    append_jsonl(source_logs / "manifest.jsonl", [{
        "tool":"curator_match",
        "source_dir": str(source_dir),
        "source_logs": str(source_logs),
        "target_logs": str(target_logs),
        "out_dir": str(match_dir),
        "budget": budget,
        "coverage_min": coverage_min,
        "note": "Outputs fixed to data_src/logs/match per policy"
    }])

    print(f"[ok] matched_source -> {matched_csv}")
    print(f"[ok] match_report  -> {match_dir/'match_report.csv'}")

if __name__ == "__main__":
    main()
    # SELECTION
    selected: List[dict] = []
    selected_paths = set()
    selected_upscaled = 0

    total_demand = sum(int(n) for n in tv_bins.values())
    budget = args.budget if args.budget > 0 else total_demand

    def is_upscaled_row(r: dict) -> bool:
        p = str(r.get("path",""))
        if not p:
            return False
        name = Path(p).name
        return (name in upscaled_set) or (p in upscaled_set)

    # coverage insurance (native first, then upscaled if needed and under cap)
    for k, n in tv_bins.items():
        if int(n) <= 0:
            continue
        proto = proto_from_key(k)
        cand_all = list(src_by_key.get(k, []))
        cand_native = [r for r in cand_all if not is_upscaled_row(r)]
        cand_up = [r for r in cand_all if is_upscaled_row(r)]
        need = coverage_min

        take_native = select_for_bin(cand_native, proto, min(need, len(cand_native)), (w_pose,w_light,w_expr))
        for r in take_native:
            p = r.get("path","")
            if p not in selected_paths:
                selected.append(r); selected_paths.add(p)
        still = need - len(take_native)

        if still > 0 and upscaled_cap > 0 and selected_upscaled < upscaled_cap and cand_up:
            take_up = select_for_bin(cand_up, proto, min(still, upscaled_cap - selected_upscaled), (w_pose,w_light,w_expr))
            for r in take_up:
                p = r.get("path","")
                if p not in selected_paths:
                    selected.append(r); selected_paths.add(p)
                    selected_upscaled += 1

    # proportional fill to budget (native first)
    remaining = max(0, budget - len(selected))
    if remaining > 0:
        sel_counter = collections.Counter(key_from_row_local(r) for r in selected)
        total_bins = sum(int(n) for n in tv_bins.values())
        for k, n in sorted(tv_bins.items(), key=lambda kv: -int(kv[1])):
            if remaining <= 0:
                break
            proto = proto_from_key(k)
            cand_native = [r for r in src_by_key.get(k, []) if (r.get("path","") not in selected_paths and not is_upscaled_row(r))]
            want = int(round(remaining * (int(n)/total_bins)))
            if want <= 0:
                continue
            take_native = select_for_bin(cand_native, proto, want, (w_pose,w_light,w_expr))
            for r in take_native:
                p = r.get("path","")
                if p not in selected_paths:
                    selected.append(r); selected_paths.add(p)
                    remaining -= 1
                    if remaining <= 0:
                        break
            if remaining <= 0:
                break
            if upscaled_cap > 0 and selected_upscaled < upscaled_cap:
                cand_up = [r for r in src_by_key.get(k, []) if (r.get("path","") not in selected_paths and is_upscaled_row(r))]
                want_up = min(remaining, upscaled_cap - selected_upscaled)
                if want_up > 0 and cand_up:
                    take_up = select_for_bin(cand_up, proto, want_up, (w_pose,w_light,w_expr))
                    for r in take_up:
                        p = r.get("path","")
                        if p not in selected_paths:
                            selected.append(r); selected_paths.add(p)
                            selected_upscaled += 1
                            remaining -= 1
                            if remaining <= 0 or selected_upscaled >= upscaled_cap:
                                break

