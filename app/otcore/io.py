
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from .schemas import Boundaries, Shapes, CMap, Support, Pairings, Reps, TriangleSchema, \
    check_cmap_square_against_shapes, check_boundaries_against_shapes, check_support_against_cmap

def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def parse_boundaries(d: dict) -> Boundaries:
    return Boundaries(**d)

def parse_shapes(d: dict) -> Shapes:
    return Shapes(**d)

def parse_cmap(d: dict) -> CMap:
    return CMap(**d)

def parse_support(d: dict) -> Support:
    if "masks" not in d:
        d = {"masks": d}
    return Support(**d)

def parse_pairings(d: dict) -> Pairings:
    return Pairings(data=d)

def parse_reps(d: dict) -> Reps:
    return Reps(data=d)

def parse_triangle_schema(d: dict) -> TriangleSchema:
    return TriangleSchema(**d)

def validate_bundle(boundaries: Boundaries, shapes: Shapes, cmap: CMap, support: Support | None = None) -> None:
    check_cmap_square_against_shapes(cmap, shapes)
    check_boundaries_against_shapes(boundaries, shapes)
    if support is not None:
        check_support_against_cmap(support, cmap)
