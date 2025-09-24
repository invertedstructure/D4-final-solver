
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json, csv, os
from . import io
from .schemas import Boundaries, Shapes, CMap, TriangleSchema
from .unit_gate import unit_check
from .overlap_gate import overlap_check
from .triangle_gate import triangle_check
from .towers import run_tower
from .hashes import bundle_content_hash, timestamp_iso_lisbon, run_id, APP_VERSION

class ManifestError(Exception): ...

def run_manifest(manifest_path: str, report_dir: str) -> Dict[str, Any]:
    M = io.load_json(manifest_path)
    required = ["boundaries","shapes","cmap","towers","seed"]
    for k in required:
        if k not in M:
            raise ManifestError(f"Manifest missing '{k}'")
    B = io.parse_boundaries(io.load_json(M["boundaries"]))
    S = io.parse_shapes(io.load_json(M["shapes"]))
    C = io.parse_cmap(io.load_json(M["cmap"]))
    support = io.parse_support(io.load_json(M["support"])) if M.get("support") else None
    H = io.parse_cmap(io.load_json(M["homotopy"])) if M.get("homotopy") else None
    tri = io.parse_triangle_schema(io.load_json(M["triangle_schema"])) if M.get("triangle_schema") else None
    io.validate_bundle(B, S, C, support)

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    certs_dir = Path(report_dir) / "certs"
    towers_dir = Path(report_dir) / "towers"
    inputs_dir = Path(report_dir) / "inputs"
    certs_dir.mkdir(exist_ok=True, parents=True)
    towers_dir.mkdir(exist_ok=True, parents=True)
    inputs_dir.mkdir(exist_ok=True, parents=True)

    named = [("boundaries", B.dict()), ("shapes", S.dict()), ("cmap", C.dict())]
    if support: named.append(("support", support.dict()))
    if H: named.append(("homotopy", H.dict()))
    if tri: named.append(("triangle_schema", tri.dict()))
    content_hash = bundle_content_hash(named)
    ts = timestamp_iso_lisbon()
    rid = run_id(content_hash, ts, APP_VERSION)

    unit_res = unit_check(B, C, S)
    (certs_dir / "unit_pass.json").write_text(json.dumps({"result":unit_res,"content_hash":content_hash,"run_id":rid,"timestamp":ts,"version":APP_VERSION}, indent=2))

    if H:
        overlap_res = overlap_check(B, C, H)
        (certs_dir / "overlap_pass.json").write_text(json.dumps({"result":overlap_res,"content_hash":content_hash,"run_id":rid,"timestamp":ts,"version":APP_VERSION}, indent=2))

    if tri:
        tri_res = triangle_check(B, tri)
        (certs_dir / "triangle_pass.json").write_text(json.dumps({"result":tri_res,"content_hash":content_hash,"run_id":rid,"timestamp":ts,"version":APP_VERSION}, indent=2))

    tower_summaries = []
    for sched in M["towers"]:
        name = sched["name"]
        steps = sched["steps"]
        csv_path = str(towers_dir / f"tower-hashes_{name}.csv")
        run_tower(steps, C, S, M["seed"], csv_path, schedule_name=name)
        first_div = None
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                val = row["diverges_from_baseline_at"]
                if val:
                    first_div = int(val)
                    break
        tower_summaries.append({"name": name, "first_divergence": first_div, "csv": os.path.basename(csv_path)})

    (certs_dir / "tower_first_divergence.json").write_text(json.dumps({"towers":tower_summaries,"content_hash":content_hash,"run_id":rid,"timestamp":ts,"version":APP_VERSION}, indent=2))

    resolved = {"manifest": M, "content_hash": content_hash, "run_id": rid, "timestamp": ts, "version": APP_VERSION}
    (Path(report_dir) / "manifest_resolved.json").write_text(json.dumps(resolved, indent=2))

    for key in ["boundaries","shapes","cmap","support","homotopy","triangle_schema"]:
        if M.get(key):
            src = Path(M[key])
            try:
                data = io.load_json(src)
                (inputs_dir / src.name).write_text(json.dumps(data, indent=2))
            except Exception:
                pass

    return {"report_dir": str(report_dir), "content_hash": content_hash, "run_id": rid, "timestamp": ts, "version": APP_VERSION}
