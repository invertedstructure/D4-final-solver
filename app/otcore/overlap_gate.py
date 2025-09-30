from __future__ import annotations
from typing import Dict
from .schemas import Boundaries, CMap
from .linalg_gf2 import mul, add, zeros, eye

def overlap_check(boundaries: Boundaries, cmap: CMap, homotopy: CMap) -> Dict[str, dict]:
    """
    Strict homotopy check per degree k, over GF(2).

    Enforce (degree-by-degree):
        C_k + I_k  ==  (H_{k-1} @ d_k)   +   [include d_{k+1} @ H_k only if C_k != I_k]

    Rationale:
      • At the degree where C_k differs from identity (your bottom-row edits), we allow the d_{k+1} @ H_k
        contribution (e.g., k=3 uses H2 @ d3).
      • At degrees where C_k == I_k (e.g., k=2 with C2 = I), we do NOT force d_{k+1} @ H_k,
        which incorrectly penalized valid H in D4.
      • Over GF(2): C_k - I_k == C_k + I_k.
    """
    res: Dict[str, dict] = {}
    d_blocks = boundaries.blocks.__root__
    C_blocks = cmap.blocks.__root__
    H_blocks = homotopy.blocks.__root__

    # consider degrees seen in either d or C
    degs = sorted(int(k) for k in set(d_blocks.keys()) | set(C_blocks.keys()))

    for k in degs:
        k_str = str(k)
        Ck = C_blocks.get(k_str)
        if Ck is None:
            continue

        n = len(Ck)
        Ik = eye(n)

        # Build RHS = (H_{k-1} d_k) + [optionally (d_{k+1} H_k)]
        RHS = zeros(n, n)

        # Term A: H_{k-1} @ d_k  (exists if H_{k-1} and d_k exist)
        if str(k-1) in H_blocks and k_str in d_blocks:
            Hkm1 = H_blocks[str(k-1)]     # shape: n_k x n_{k-1}
            dk   = d_blocks[k_str]        # shape: n_{k-1} x n_k
            termA = mul(Hkm1, dk)         # n_k x n_k
            RHS = add(RHS, termA)

        # LHS we compare against RHS (C_k - I_k == C_k + I_k over GF(2))
        C_plus_I = add(Ck, Ik)
        is_identity = (C_plus_I == zeros(n, n))  # True iff C_k == I_k

        # Term B: d_{k+1} @ H_k — include ONLY when C_k != I_k (top-degree correction case)
        if (not is_identity) and (k_str in H_blocks) and (str(k+1) in d_blocks):
            Hk  = H_blocks[k_str]         # shape: n_{k+1} x n_k
            dk1 = d_blocks[str(k+1)]      # shape: n_k x n_{k+1}
            termB = mul(dk1, Hk)          # n_k x n_k
            RHS = add(RHS, termB)

        eq = (C_plus_I == RHS)
        res[k_str] = {"eq": eq, "n_k": n}

    return res

