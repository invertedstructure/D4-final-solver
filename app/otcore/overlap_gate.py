from __future__ import annotations

from typing import Dict, Optional
from .projector import apply_projection
from .schemas import Boundaries, CMap
from .linalg_gf2 import mul, add, zeros, eye

def _mat_size(M):
    """Return (rows, cols) for a list-of-lists matrix; (0,0) if None/empty."""
    if not M:
        return (0, 0)
    return (len(M), len(M[0]) if M and len(M) > 0 else 0)

def overlap_check(
    boundaries: Boundaries,
    cmap: CMap,
    homotopy: CMap,
    projection_config: Optional[dict] = None,
    projector_cache: Optional[dict] = None,
) -> Dict[str, dict]:
    """
    Overlap gate: strict homotopy residuals per degree, with optional projection.
    GF(2) arithmetic.

    k=3 (your setup with H3 = 0, C2 = I, d2 = 0):
        R3 = H2 @ d3 + (C3 - I3)
        (strict equality wants R3 == 0)

    k=2 (vacuous in your fixtures since d2=0 and C2=I):
        R2 = d2 @ H2 + H1 @ d2 - (C2 - I2) == 0
        (still run it; projection can be configured per-k)
    """
    cfg = projection_config or {}
    cache = projector_cache or {}

    out: Dict[str, dict] = {}

    blocks_b = boundaries.blocks.__root__
    blocks_c = cmap.blocks.__root__
    blocks_h = homotopy.blocks.__root__

    # ---- k = 3 ----
    d3 = blocks_b.get("3")
    C3 = blocks_c.get("3")
    H2 = blocks_h.get("2")
    if d3 is not None and C3 is not None and H2 is not None:
        n3, _ = _mat_size(C3)
        I3 = eye(n3)
        # Strict residual over GF(2): R3 = H2@d3 + (C3 - I3) = H2@d3 + (C3 + I3)
        R3 = add(mul(H2, d3), add(C3, I3))
        # Apply projection (columns/rows) if configured for k=3
        R3p = apply_projection(R3, 3, boundaries, cfg, cache)
        eq3 = (R3p == zeros(n3, n3))
        out["3"] = {"eq": bool(eq3), "n_k": n3}

    

    # ---- k = 2 ----
    # In your fixtures, often d2 = 0 and C2 = I; still compute robustly:
    d3 = blocks_b.get("3")                # for d3 @ H2
    d2 = blocks_b.get("2")                # for H1 @ d2
    C2 = blocks_c.get("2")
    H1 = blocks_h.get("1")
    H2 = blocks_h.get("2")  # reused

    if C2 is not None:
        n2, _ = _mat_size(C2)
        I2 = eye(n2)

        # Term A: d3 @ H2 if shapes align, else zero(n2, n2)
        termA = zeros(n2, n2)
        if d3 is not None and H2 is not None:
            r3, c3 = _mat_size(d3)   # d3: (n2 x n3) in your 4D bottom-row fixtures
            rH2, cH2 = _mat_size(H2) # H2: (n3 x n2)
            if c3 == rH2 and r3 == n2 and cH2 == n2:
                termA = mul(d3, H2)

        # Term B: H1 @ d2 if shapes align, else zero(n2, n2)
        termB = zeros(n2, n2)
        if H1 is not None and d2 is not None:
            rH1, cH1 = _mat_size(H1) # H1: (n2 x n1)
            r2,  c2  = _mat_size(d2) # d2: (n1 x n2)
            if cH1 == r2 and rH1 == n2 and c2 == n2:
                termB = mul(H1, d2)

        # Strict residual at k=2: R2 = (d3@H2 + H1@d2) - (C2 - I2)  ==  add(termA, termB) + (C2 + I2) over GF(2)
        R2  = add(add(termA, termB), add(C2, I2))
        R2p = apply_projection(R2, 2, boundaries, cfg, cache)
        eq2 = (R2p == zeros(n2, n2))
        out["2"] = {"eq": bool(eq2), "n_k": n2}

    return out
