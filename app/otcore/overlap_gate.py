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
    # In your fixtures, d2 = 0 and C2 = I, so this is identically zero,
    # but we still compute it so projected mode can be toggled consistently.
    d2 = blocks_b.get("2")
    C2 = blocks_c.get("2")
    H1 = blocks_h.get("1")
    H2 = blocks_h.get("2")  # reused

    if C2 is not None:
        n2, _ = _mat_size(C2)
        I2 = eye(n2)
        # d2 may be missing => treat as zeros of the right shape
        if d2 is None:
            d2 = zeros(n2, n2)
        if H1 is None:
            # shape-compatible zeros for H1
            d2_rows, d2_cols = _mat_size(d2)
            H1 = zeros(d2_cols, d2_rows) if d2_rows and d2_cols else zeros(n2, n2)

        # Strict residual at k=2: R2 = d2@H2 + H1@d2 - (C2 - I2)
        # Over GF(2), subtraction == addition
        R2 = add(add(mul(d2, H2 if H2 is not None else zeros(_mat_size(d2)[1], 0)),
                     mul(H1, d2)),
                 add(C2, I2))
        R2p = apply_projection(R2, 2, boundaries, cfg, cache)
        eq2 = (R2p == zeros(n2, n2))
        out["2"] = {"eq": bool(eq2), "n_k": n2}

    return out
