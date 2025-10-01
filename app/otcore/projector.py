# app/otcore/projector.py
from __future__ import annotations
import json
from typing import Dict, Optional, List, Any

from .linalg_gf2 import mul, zeros  # use your existing GF(2) ops

import hashlib
from typing import Any, Dict, List

def projector_hash(P: List[List[int]]) -> str:
    s = json.dumps(P, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def save_projector(path: str, P: List[List[int]]) -> str:
    with open(path, "w") as f:
        json.dump({"P": P}, f, separators=(",", ":"), sort_keys=True)
    return projector_hash(P)

def load_projector_file(path: str) -> List[List[int]]:
    with open(path, "r") as f:
        d = json.load(f)
    return d.get("P") or d  # support bare matrix or {"P": ...}


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

from typing import Any, Dict, List, Optional
from .linalg_gf2 import mul

def apply_projection(
    Rk: List[List[int]],
    k: int,
    boundaries,                                  # Boundaries (we’ll use d_k columns)
    config: Optional[Dict[str, Any]] = None,
    projector_cache: Optional[Dict[str, List[List[int]]]] = None,
):
    """
    Project the *residual* Rk at degree k on the column side (GF(2)).

    Policy:
      - Only runs when k is enabled and mode[k] == "columns".
      - The projector Π_k is built from the *columns of d_k* (shape n_k x n_k),
        because R_k is n_k x n_k.
      - If source[k] == "file", use cached file projector but also compute auto Π_k
        and stash a drift warning in projector_cache if they differ.
    """
    cfg = config or {}
    cache = projector_cache or {}

    enabled = set(cfg.get("enabled_layers", []))
    if not Rk or k not in enabled:
        return Rk

    mode = cfg.get("modes", {}).get(str(k), "none")
    if mode != "columns":
        # (rows-mode is for real-valued layers; no-op for GF(2) runs)
        return Rk

    # Columns of d_k determine which residual columns are "lanes"
    blocks_b = boundaries.blocks.__root__
    dk = blocks_b.get(str(k))  # d_k : (n_{k-1} x n_k)
    if dk is None or not dk or not dk[0]:
        return Rk

    # Build Π_k (auto) from d_k columns (works for any binary matrix)
    n_k = len(dk[0])  # number of columns of d_k == columns of Rk
    lane_mask = [1 if any(row[j] & 1 for row in dk if j < len(row)) else 0 for j in range(n_k)]
    P_auto = [[1 if (i == j and lane_mask[j]) else 0 for j in range(n_k)] for i in range(n_k)]

    # Decide source
    source = cfg.get("source", {}).get(str(k), "auto")
    if source == "file" and str(k) in cache:
        P_file = cache[str(k)]
        # drift guard: compare file vs auto
        try:
            # projector_hash provided elsewhere in this module
            hf = projector_hash(P_file)
            ha = projector_hash(P_auto)
            if P_file != P_auto:
                cache[f"guard_warning_k{k}"] = {
                    "msg": "Projector drift: file vs auto differ",
                    "hash_file": hf,
                    "hash_auto": ha,
                }
        except Exception:
            # ignore hashing problems, still use P_file
            pass
        P = P_file
    else:
        P = P_auto

    # Size/shape guard: Rk and Π_k must be n_k x n_k
    if len(Rk) != n_k or (Rk and len(Rk[0]) != n_k) or len(P) != n_k or len(P[0]) != n_k:
        return Rk  # explicit no-op keeps behavior predictable

    # Project *residual* on the right
    return mul(Rk, P)


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



