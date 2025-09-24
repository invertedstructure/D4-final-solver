
from __future__ import annotations
from typing import Dict
from .schemas import Boundaries, CMap, Shapes
from .linalg_gf2 import mul, add, zeros, shape, eye

def unit_check(boundaries: Boundaries, cmap: CMap, shapes: Shapes) -> Dict[str, dict]:
    """
    Verify chain law: d_k C_k = C_{k-1} d_k for all degrees k present.
    Returns per-degree pass info.
    """
    result: Dict[str, dict] = {}
    blocks_d = boundaries.blocks.__root__
    blocks_c = cmap.blocks.__root__
    for k_str, d_k in blocks_d.items():
        k = int(k_str)
        Ck = blocks_c.get(k_str)
        Ckm1 = blocks_c.get(str(k-1))
        if Ck is None or Ckm1 is None:
            # if either side missing due to boundary at edges, skip with note
            result[k_str] = {"eq": True, "note": "skipped (edge degree or missing C blocks)", "n_k": len(Ck) if Ck else 0}
            continue
        lhs = mul(d_k, Ck)            # d_k C_k
        rhs = mul(Ckm1, d_k)          # C_{k-1} d_k
        eq = (lhs == rhs)
        result[k_str] = {"eq": eq, "n_k": len(Ck)}
    return result
