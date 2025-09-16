#!/usr/bin/env python3
from __future__ import annotations
import csv, json, os
from pathlib import Path
from typing import Iterable, Dict, Any, List

def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _rows_from_any(rows):
    if rows is None:
        return []
    if isinstance(rows, list):
        return rows
    return list(rows)

def write_csv(path: os.PathLike | str, headers: Iterable[str], rows: Iterable[Dict[str, Any]] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = _rows_from_any(rows)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})

def append_csv(path: os.PathLike | str, headers: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})

def read_csv(path: os.PathLike | str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]

def write_jsonl(path: os.PathLike | str, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_jsonl(path: os.PathLike | str, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('a', encoding='utf-8') as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_text(path: os.PathLike | str, text: str) -> None:
    path = Path(path); ensure_dir(path.parent)
    path.write_text(text, encoding='utf-8')

def safe_glob(root: os.PathLike | str, exts=('.jpg','.jpeg','.png','.webp','.bmp')) -> list[str]:
    root = Path(root)
    out = []
    for ext in exts:
        out.extend([str(p) for p in root.rglob(f'*{ext}')])
    return sorted(out)

# --- YAML helpers (graceful fallback) ---
def load_yaml(path_candidates: list[str]) -> dict:
    for p in path_candidates:
        if p and Path(p).exists():
            try:
                import yaml  # type: ignore
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                # Try extremely simple loader for our limited skeleton
                try:
                    txt = Path(p).read_text(encoding='utf-8')
                    return _mini_yaml_parse(txt)
                except Exception:
                    continue
    return {}

def _mini_yaml_parse(text: str) -> dict:
    # NOTE: extremely naive YAML subset: key: value, one level maps/lists only
    data = {}
    current_map = None
    stack = [data]
    indent_stack = [0]
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        key_val = line.strip().split(":", 1)
        if len(key_val) == 2:
            key, val = key_val[0].strip(), key_val[1].strip()
            while indent < indent_stack[-1]:
                stack.pop(); indent_stack.pop()
            if val == "":
                m = {}
                stack[-1][key] = m
                stack.append(m); indent_stack.append(indent + 2)
            else:
                # try to parse number or list
                if val.startswith("[") and val.endswith("]"):
                    try:
                        arr = [x.strip() for x in val[1:-1].split(",") if x.strip()]
                        stack[-1][key] = [try_float(x) for x in arr]
                    except Exception:
                        stack[-1][key] = val
                else:
                    stack[-1][key] = try_float(val)
        elif line.strip().startswith("-"):
            # list item
            item = try_float(line.strip()[1:].strip())
            # attach to last map with an implicit list?
            if isinstance(stack[-1], list):
                stack[-1].append(item)
            else:
                # create list under a special key?
                pass
    return data

def try_float(x: str):
    try:
        if x.lower() in ("true","false"):
            return x.lower() == "true"
        if "." in x or "e" in x.lower():
            return float(x)
        return int(x)
    except Exception:
        return x
