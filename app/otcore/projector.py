# projector.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json, os

def projector_columns_from_dkp1(dkp1):
    """
    Build a columns projector Π_k = diag(lane_mask) over GF(2),
    where lane_mask[j] = 1 iff column j of d_{k+1} has any 1s.
    dkp1 is a list-of-lists matrix for d_{k+1} with shape (n_k, n_{k+1}).
    Returns a square (n_{k+1} x n_{k+1}) list-of-lists matrix.
    """
    if dkp1 is None:
        return None
    n_rows = len(dkp1)
    n_cols = len(dkp1[0]) if n_rows else 0
    lane_mask = []
    for j in range(n_cols):
        has_lane = any((dkp1[i][j] & 1) for i in range(n_rows))
        lane_mask.append(1 if has_lane else 0)
    Pi = [[1 if (i == j and lane_mask[j] == 1) else 0 for j in range(n_cols)] for i in range(n_cols)]
    return Pi

def projector_rows_from_dkp1_real(dkp1):
    """
    Optional (future): orthogonal projector onto im(d_{k+1}) over R.
    Not used in GF(2) runs; here for completeness.
    Returns P_row with shape (n_k x n_k) as list-of-lists.
    """
    try:
        import numpy as np
    except Exception:
        return None
    if dkp1 is None:
        return None
    A = np.array(dkp1, dtype=float)
    if A.size == 0:
        return (np.zeros((0,0))).tolist()
    Q, _ = np.linalg.qr(A)
    P = Q @ Q.T
    return P.tolist()

def load_projection_config(path: str) -> Dict[str, Any]:
    """
    Load a JSON config describing which layers to project and how.
    If the file doesn't exist, returns a config that disables projection.
    """
    if not path or not os.path.exists(path):
        return {"enabled_layers": [], "modes": {}, "source": {}, "projector_files": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preload_projectors_from_files(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    If cfg['source'][k] == 'file', preload the projector matrix for that k from cfg['projector_files'][k].
    Expects each file to contain a square matrix (list of lists).
    """
    out: Dict[str, Any] = {}
    files = cfg.get("projector_files", {}) or {}
    for k, fpath in files.items():
        try:
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    out[str(k)] = json.load(f)
        except Exception:
            pass
    return out

def apply_projection_GF2(Rk, k: int, boundaries, cfg: Dict[str, Any], cache: Optional[Dict[str, Any]] = None):
    """
    Decide and (optionally) return a columns projector for residual Rk at degree k.
    For GF(2) we project columns: Rk_proj = Rk @ Π_k.
    Returns either:
      - (Rk, Π_k)  when projection is enabled and a projector is available
      - Rk         unchanged when projection is disabled or no projector
    Caller should multiply with GF(2) matmul.
    """
    cache = cache or {}
    enabled = set(cfg.get("enabled_layers", []))
    if k not in enabled:
        return Rk

    modes = (cfg.get("modes", {}) or {})
    sources = (cfg.get("source", {}) or {})
    mode = modes.get(str(k), "none")
    source = sources.get(str(k), "auto")

    if mode == "none":
        return Rk

    P = None
    if source == "file":
        P = cache.get(str(k))
    else:
        dkp1 = boundaries.blocks.__root__.get(str(k+1))
        if mode == "columns":
            P = projector_columns_from_dkp1(dkp1)
        elif mode == "rows":
            # rows-mode isn't meaningful over GF(2) here; skip
            return Rk

    if P is None:
        return Rk

    # Signal to caller to do GF(2) right-multiply: Rk @ P
    return (Rk, P)
