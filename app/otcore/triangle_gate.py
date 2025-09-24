
from __future__ import annotations
from typing import Dict
from .schemas import Boundaries, TriangleSchema
from .linalg_gf2 import mul, add, zeros, shape

def triangle_check(boundaries: Boundaries, tri: TriangleSchema) -> Dict[str, dict]:
    res: Dict[str, dict] = {}
    d_blocks = boundaries.blocks.__root__
    for k_str, degobj in tri.by_degree.items():
        k = int(k_str)
        A = degobj.A
        B = degobj.B
        J = degobj.J
        term1 = None
        if str(k+1) in d_blocks:
            dk1 = d_blocks[str(k+1)]
            term1 = mul(dk1, J)
        term2 = None
        if str(k-1) in tri.by_degree and str(k) in d_blocks:
            Jkm1 = tri.by_degree[str(k-1)].J
            dk = d_blocks[str(k)]
            term2 = mul(Jkm1, dk)
        if term1 is not None and term2 is not None:
            LHS = add(term1, term2)
        elif term1 is not None:
            LHS = term1
        elif term2 is not None:
            LHS = term2
        else:
            r, c = shape(A)
            LHS = zeros(r, c)
        RHS = add(A, B)
        eq = (LHS == RHS)
        res[k_str] = {"eq": eq, "n_k": len(A)}
    return res
