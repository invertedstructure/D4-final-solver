
from __future__ import annotations
from typing import Dict, Optional
from .schemas import Boundaries, TriangleSchema, Shapes
from .linalg_gf2 import mul, add, zeros

def _mat_shape(M):
    return (len(M), len(M[0]) if (isinstance(M, list) and M and isinstance(M[0], list)) else 0)

def triangle_check(boundaries: Boundaries, tri: TriangleSchema, shapes: Optional[Shapes]=None) -> Dict[str, dict]:
    res: Dict[str, dict] = {}
    d_blocks = boundaries.blocks.__root__
    for k_str, degobj in tri.by_degree.items():
        k = int(k_str)
        A = degobj.A
        B = degobj.B
        J = degobj.J
        # Explicit shape assertions (if shapes provided)
        if shapes is not None:
            n_k   = shapes.n.get(k_str, None)
            n_kp1 = shapes.n.get(str(k+1), None)
            n_km1 = shapes.n.get(str(k-1), None)
            if n_k is None:
                raise ValueError(f"Triangle: shapes missing n['{k_str}']")
            rJ, cJ = _mat_shape(J)
            if n_kp1 is not None:
                if rJ not in (0, n_kp1):
                    raise ValueError(f"Triangle: J_{k} rows={rJ} expected 0 or {n_kp1}")
            if cJ not in (0, n_k):
                raise ValueError(f"Triangle: J_{k} cols={cJ} expected 0 or {n_k}")
            rA, cA = _mat_shape(A); rB, cB = _mat_shape(B)
            if (rA, cA) not in [(0,0), (n_k, n_k)]:
                raise ValueError(f"Triangle: A_{k} shape {rA}x{cA} expected {n_k}x{n_k} (or empty)")
            if (rB, cB) not in [(0,0), (n_k, n_k)]:
                raise ValueError(f"Triangle: B_{k} shape {rB}x{cB} expected {n_k}x{n_k} (or empty)")
        # Compute LHS = d_{k+1} J_k + J_{k-1} d_k
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
            # XOR (GF(2)) add
            LHS = [[(term1[i][j] ^ term2[i][j]) for j in range(len(term1[0]))] for i in range(len(term1))]
        elif term1 is not None:
            LHS = term1
        elif term2 is not None:
            LHS = term2
        else:
            rA, cA = _mat_shape(A)
            LHS = zeros(rA, cA)
        # RHS = A_k + B_k
        if _mat_shape(A) != _mat_shape(B):
            raise ValueError(f"Triangle: A_{k} and B_{k} shapes differ: {_mat_shape(A)} vs {_mat_shape(B)}")
        RHS = [[(A[i][j] ^ B[i][j]) for j in range(len(A[0]) if A else 0)] for i in range(len(A))] if A else []
        eq = (LHS == RHS)
        res[k_str] = {"eq": eq, "n_k": len(A) if isinstance(A, list) else 0}
    return res
