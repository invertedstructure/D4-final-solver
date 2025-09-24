
from __future__ import annotations
from typing import Dict
from .schemas import Boundaries, CMap, Shapes
from .linalg_gf2 import mul

def unit_check(boundaries: Boundaries, cmap: CMap, shapes: Shapes) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    blocks_d = boundaries.blocks.__root__
    blocks_c = cmap.blocks.__root__
    for k_str, d_k in blocks_d.items():
        k = int(k_str)
        Ck = blocks_c.get(k_str)
        Ckm1 = blocks_c.get(str(k-1))
        if Ck is None or Ckm1 is None:
            result[k_str] = {"eq": True, "note": "skipped (edge degree or missing C blocks)", "n_k": len(Ck) if Ck else 0}
            continue
        lhs = mul(d_k, Ck)
        rhs = mul(Ckm1, d_k)
        eq = (lhs == rhs)
        result[k_str] = {"eq": eq, "n_k": len(Ck)}
    return result
