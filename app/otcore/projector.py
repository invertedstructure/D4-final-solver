# app/otcore/projector.py
from __future__ import annotations
import json
from typing import Dict, Optional, List, Any

from .linalg_gf2 import mul, zeros  # use your existing GF(2) ops

# ---------- config helpers ----------

_DEFAULT_CFG = {
    "enabled_layers": [],
    "modes": {},             # e.g., {"3": "columns", "2": "none"}
    "source": {},            # e.g., {"3": "auto", "2": "auto"}
    "projector_files": {}    # e.g., {"3": "projector_D3.json", "2": "projector_D2.json"}
}

def load_projection_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize types (keys as strings)
        data = {**_DEFAULT_CFG, **(data or {})}
        data["modes"] = {str(k): v for k, v in data.get("modes", {}).items()}
        data["source"] = {str(k): v for k, v in data.get("source", {}).items()}
        data["projector_files"] = {str(k): v for k, v in data.get("projector_files", {}).items()}
        return data
    except Exception:
        # no config = strict mode
        return dict(_DEFAULT_CFG)

def preload_projectors_from_files(cfg: Dict[str, Any]) -> Dict[str, List[List[int]]]:
    cache: Dict[str, List[List[int]]] = {}
    proj_files = cfg.get("projector_files", {})
    for k, fpath in proj_files.items():
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                cache[str(k)] = json.load(f)
        except Exception:
            # ignore missing/non-json
            pass
    return cache

# ---------- projector builders ----------

def projector_columns_from_dkp1(dkp1):
    """
    Build a diagonal projector Π_k of shape (n_k x n_k) that keeps only 'lane' columns.
    Here we pass d_k (shape n_{k-1} x n_k). Residual R_k is (n_k x n_k),
    so Π_k must be n_k x n_k.
    """
    if not dkp1 or not dkp1[0]:
        return []
    n_rows = len(dkp1)       # = n_{k-1}
    n_cols = len(dkp1[0])    # = n_k  <-- projector size
    lane_mask = [
        1 if any(dkp1[i][j] % 2 != 0 for i in range(n_rows)) else 0
        for j in range(n_cols)
    ]
    return [[1 if (i == j and lane_mask[j]) else 0 for j in range(n_cols)]
            for i in range(n_cols)]


# (Optional) rows-mode for real-valued math; unused in your GF(2) pass.
def projector_rows_from_dkp1_real(dkp1: List[List[float]]) -> List[List[float]]:
    try:
        import numpy as np
    except Exception:
        # Fallback: identity (no projection) if numpy not available
        if not dkp1:
            return []
        n_k = len(dkp1)
        return [[1.0 if i == j else 0.0 for j in range(n_k)] for i in range(n_k)]
    A = np.array(dkp1, dtype=float)       # shape (n_k, n_{k+1})
    if A.size == 0:
        return []
    Q, _ = np.linalg.qr(A)                # Q: (n_k, r)
    P = Q @ Q.T                           # (n_k, n_k)
    return P.tolist()


# ---------- projection application ----------
def apply_projection(
    Rk: List[List[int]],
    k: int,
    boundaries,                      # Boundaries (we'll use d_k for the mask)
    config: Optional[Dict[str, Any]] = None,
    projector_cache: Optional[Dict[str, List[List[int]]]] = None,
):
    """
    Apply configured projection to the residual Rk at degree k.

    - columns: Rk <- Rk @ Π_k, where Π_k is built from the *columns of d_k*.
      R_k is (n_k x n_k), and d_k has shape (n_{k-1} x n_k). So Π_k must be (n_k x n_k).
    - rows:    Rk <- P_row @ Rk   (intended for real-valued work; unused in GF(2) runs)

    If layer k not enabled or mode is 'none', returns Rk unchanged.
    """
    cfg = config or _DEFAULT_CFG
    enabled = set(cfg.get("enabled_layers", []))
    if k not in enabled or not Rk:
        return Rk

    mode = cfg.get("modes", {}).get(str(k), "none")
    if mode == "none":
        return Rk

    source = cfg.get("source", {}).get(str(k), "auto")
    cache = projector_cache or {}

    # Build/load projector Π_k using d_k (its columns index the residual's columns)
    if source == "file" and str(k) in cache:
        P = cache[str(k)]
    else:
        blocks_b = boundaries.blocks.__root__
        dk_for_mask = blocks_b.get(str(k))     # <-- use d_k (NOT d_{k+1})
        if dk_for_mask is None:
            return Rk  # nothing to project against
        if mode == "columns":
            P = projector_columns_from_dkp1(dk_for_mask)   # returns n_k x n_k diag
        elif mode == "rows":
            P = projector_rows_from_dkp1_real(dk_for_mask) # left-mult (real-only)
        else:
            return Rk

    # Apply Π_k with guards
    if mode == "columns":
        # Rk must be square and P must match its column count
        if not Rk[0] or not P or not P[0]:
            return Rk
        n_cols = len(Rk[0])
        if len(P) != n_cols or len(P[0]) != n_cols:
            return Rk  # size mismatch → no-op
        return mul(Rk, P)

    elif mode == "rows":
        # left multiply in float; convert back to ints if near-exact 0/1
        try:
            import numpy as np
            A = np.array(P) @ np.array(Rk)
            B = (np.abs(A) > 1e-9).astype(int).tolist()
            return B
        except Exception:
            return Rk

    else:
        return Rk


        # Apply
    if not Rk:
        return Rk

    if mode == "columns":
        # Guard: Rk must be square and P must match its column count
        if not Rk[0] or not P or not P[0]:
            return Rk
        n_cols = len(Rk[0])
        if len(P) != n_cols or len(P[0]) != n_cols:
            # sizes don't align → no-op (or swap in eye(n_cols) if you prefer)
            return Rk
        return mul(Rk, P)

    elif mode == "rows":
        # left multiply in float; convert back to ints if near-exact 0/1
        try:
            import numpy as np
            A = np.array(P) @ np.array(Rk)
            B = (np.abs(A) > 1e-9).astype(int).tolist()
            return B
        except Exception:
            return Rk

    else:
        return Rk

