
from __future__ import annotations
from typing import Dict
from .schemas import Boundaries, TriangleSchema
from .linalg_gf2 import mul, add

def triangle_check(boundaries: Boundaries, tri: TriangleSchema) -> Dict[str, dict]:
    """
    Verify for each degree k in tri:
      d_{k+1} J_k + J_{k-1} d_k = A_k + B_k   over GF(2).
    Missing boundary blocks are treated as zero (edges).
    """
    res: Dict[str, dict] = {}
    d_blocks = boundaries.blocks.__root__
    for k_str, degobj in tri.by_degree.items():
        k = int(k_str)
        A = degobj.A
        B = degobj.B
        J = degobj.J
        # Compute d_{k+1} J_k
        term1 = None
        if str(k+1) in d_blocks:
            dk1 = d_blocks[str(k+1)]
            term1 = mul(dk1, J)
        # Compute J_{k-1} d_k -- but tri provides only J_k; standard equation uses J_{k-1} at degree k-1.
        # In our schema, we only have J_k per degree; for k >= 2 we can fetch J_{k-1} if present.
        term2 = None
        if str(k-1) in tri.by_degree and str(k) in d_blocks:
            Jkm1 = tri.by_degree[str(k-1)].J
            dk = d_blocks[str(k)]
            term2 = mul(Jkm1, dk)
        # LHS
        if term1 is not None and term2 is not None:
            LHS = add(term1, term2)
        elif term1 is not None:
            LHS = term1
        elif term2 is not None:
            LHS = term2
        else:
            # Edge cases reduce to zeros
            from .linalg_gf2 import shape, zeros
            r, c = shape(A)
            LHS = zeros(r, c)
        RHS = add(A, B)
        eq = (LHS == RHS)
        res[k_str] = {"eq": eq, "n_k": len(A)}
    return res
