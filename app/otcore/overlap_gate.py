
from __future__ import annotations
from typing import Dict
from .schemas import Boundaries, CMap
from .linalg_gf2 import mul, add, zeros, shape, eye

def overlap_check(boundaries: Boundaries, cmap: CMap, homotopy: CMap) -> Dict[str, dict]:
    """
    Chain homotopy against identity: C - I = d H + H d
    H_k : degree k -> k+1  (shape n_{k+1} x n_k)
    For top/bottom degrees where shapes don't align, equation reduces appropriately.
    """
    res: Dict[str, dict] = {}
    d_blocks = boundaries.blocks.__root__
    C_blocks = cmap.blocks.__root__
    H_blocks = homotopy.blocks.__root__  # expect 'k' keys for H_k
    degs = sorted(int(k) for k in set(d_blocks.keys()) | set(C_blocks.keys()))
    for k in degs:
        k_str = str(k)
        Ck = C_blocks.get(k_str)
        if Ck is None:
            continue
        n = len(Ck)
        Ik = eye(n)
        # Build RHS = d_{k+1} H_k + H_{k-1} d_k, carefully handling edge degrees
        term1 = None
        term2 = None
        if k_str in H_blocks and str(k+1) in d_blocks:
            Hk = H_blocks[k_str]              # n_{k+1} x n_k
            dk1 = d_blocks[str(k+1)]          # n_k x n_{k+1}
            term1 = mul(dk1, Hk)              # (n_k x n_{k+1})*(n_{k+1} x n_k) -> n_k x n_k
        if str(k-1) in H_blocks and k_str in d_blocks:
            Hkm1 = H_blocks[str(k-1)]         # n_k x n_{k-1}
            dk = d_blocks[k_str]              # n_{k-1} x n_k
            term2 = mul(Hkm1, dk)             # (n_k x n_{k-1})*(n_{k-1} x n_k) -> n_k x n_k
        RHS = None
        if term1 is not None and term2 is not None:
            RHS = add(term1, term2)
        elif term1 is not None:
            RHS = term1
        elif term2 is not None:
            RHS = term2
        else:
            RHS = zeros(n, n)
        # LHS = Ck - I = Ck + I over GF(2)
        LHS = add(Ck, Ik)
        eq = (LHS == RHS)
        res[k_str] = {"eq": eq, "n_k": n}
    return res
