
from __future__ import annotations
from typing import List, Dict
from .schemas import CMap, Shapes
from .linalg_gf2 import mul, eye
from .hashes import content_hash_of, timestamp_iso_lisbon, run_id, APP_VERSION
import csv, os

def compose_blocks(A: Dict[str, list], B: Dict[str, list]) -> Dict[str, list]:
    out = {}
    for k, Ak in A.items():
        Bk = B.get(k)
        if Bk is None:
            out[k] = Ak
        else:
            out[k] = mul(Ak, Bk)
    return out

def state_hash(blocks: Dict[str, list]) -> str:
    return content_hash_of({"blocks": blocks})

def run_tower(schedule: List[str], cmap: CMap, shapes: Shapes, seed: str, out_csv_path: str, schedule_name: str="sched") -> None:
    I_blocks = {}
    for k, mat in cmap.blocks.__root__.items():
        n = len(mat)
        I_blocks[k] = eye(n)
    C_blocks = cmap.blocks.__root__
    base_blocks = I_blocks.copy()
    baseline_hashes = []
    for i, step in enumerate(schedule):
        step_blocks = I_blocks if (i==0 or step=='I') else C_blocks
        base_blocks = compose_blocks(step_blocks, base_blocks)
        baseline_hashes.append(state_hash(base_blocks))
    cur_blocks = I_blocks.copy()
    rows = []
    ts = timestamp_iso_lisbon()
    content_bundle_hash = content_hash_of({"cmap": cmap.dict(), "shapes": shapes.dict(), "schedule": schedule, "seed": seed})
    rid = run_id(content_bundle_hash, ts, APP_VERSION)
    diverge_idx = None
    for i, step in enumerate(schedule):
        step_blocks = I_blocks if step=='I' else C_blocks
        cur_blocks = compose_blocks(step_blocks, cur_blocks)
        h = state_hash(cur_blocks)
        if diverge_idx is None and h != baseline_hashes[i]:
            diverge_idx = i
        rows.append([seed, schedule_name, i, h, (diverge_idx if diverge_idx is not None else ""), content_bundle_hash, rid, ts, APP_VERSION])
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed","schedule_name","idx","hash","diverges_from_baseline_at","content_hash","run_id","run_timestamp","app_version"])
        w.writerows(rows)
