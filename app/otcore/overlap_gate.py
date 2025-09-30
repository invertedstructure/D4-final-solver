from __future__ import annotations
from typing import Dict, Optional

from .schemas import Boundaries, CMap
from .linalg_gf2 import mul, add, zeros, eye

# Optional projector hook (auto if projector.py is present)
try:
    from .projector import (
        load_projection_config,
        preload_projectors_from_files,
        apply_projection_GF2,
    )
except Exception:
    load_projection_config = None
    preload_projectors_from_files = None
    apply_projection_GF2 = None


def _load_projection_pack(default_cfg_path: str = "projection_config.json"):
    """
    Best-effort load of projection config + cache.
    If file or module isn't available, returns (None, None).
    """
    if load_projection_config is None:
        return None, None
    cfg = load_projection_config(default_cfg_path)
    # disable if no layers selected
    if not cfg.get("enabled_layers"):
        return None, None
    cache = {}
    if preload_projectors_from_files is not None:
        cache = preload_projectors_from_files(cfg)
    return cfg, cache


def overlap_check(
    boundaries: Boundaries,
    cmap: CMap,
    homotopy: CMap,
    projection_config: Optional[dict] = None,
    projector_cache: Optional[dict] = None,
) -> Dict[str, dict]:
    """
    Strict homotopy check per degree k over GF(2), with an optional projection hook.

    Enforces, per k:
        C_k + I_k  ==  (H_{k-1} @ d_k)  +  [include d_{k+1} @ H_k only if C_k != I_k]

    Notes:
      • At the degree where C_k != I_k (your bottom-row edits), d_{k+1} @ H_k is allowed (e.g., k=3 uses H2 @ d3).
      • At degrees where C_k == I_k (e.g., k=2 with C2 = I), we DO NOT force d_{k+1} @ H_k.
      • Over GF(2), C_k - I_k == C_k + I_k.
      • If projection_config is present (or auto-loaded), we project columns of the residual Rk per config.
    """
    # Auto-load projection config if not provided
    if projection_config is None and projector_cache is None:
        projection_config, projector_cache = _load_projection_pack()

    res: Dict[str, dict] = {}
    d_blocks = boundaries.blocks.__root__
    C_blocks = cmap.blocks.__root__
    H_blocks = homotopy.blocks.__root__

    # degrees present in either d or C
    degs = sorted(int(k) for k in set(d_blocks.keys()) | set(C_blocks.keys()))

    for k in degs:
        k_str = str(k)
        Ck = C_blocks.get(k_str)
        if Ck is None:
            continue

        n = len(Ck)
        Ik = eye(n)

        # RHS = (H_{k-1} d_k) + [optionally (d_{k+1} H_k)]
        RHS = zeros(n, n)

        # Term A: H_{k-1} @ d_k   (only if both exist)
        if str(k - 1) in H_blocks and k_str in d_blocks:
            Hkm1 = H_blocks[str(k - 1)]  # shape: n_k x n_{k-1}
            dk = d_blocks[k_str]         # shape: n_{k-1} x n_k
            termA = mul(Hkm1, dk)        # n_k x n_k
            RHS = add(RHS, termA)

        # LHS (C_k - I_k) over GF(2)
        C_plus_I = add(Ck, Ik)
        is_identity = (C_plus_I == zeros(n, n))

        # Term B: d_{k+1} @ H_k  — include ONLY when C_k != I_k
        if (not is_identity) and (k_str in H_blocks) and (str(k + 1) in d_blocks):
            Hk = H_blocks[k_str]         # shape: n_{k+1} x n_k
            dk1 = d_blocks[str(k + 1)]   # shape: n_k x n_{k+1}
            termB = mul(dk1, Hk)         # n_k x n_k
            RHS = add(RHS, termB)

        # Strict residual over GF(2)
        Rk = add(C_plus_I, RHS)          # (Ck - Ik) - (H@d + d@H) == C_plus_I + RHS

        # Optional projection (columns) per config
        if projection_config and apply_projection_GF2 is not None:
            ap = apply_projection_GF2(Rk, k, boundaries, projection_config, projector_cache or {})
            if isinstance(ap, tuple):
                Rk_proj, P = ap
                Rk_proj = mul(Rk_proj, P)  # right-multiply columns projector Π_k
                eq = (Rk_proj == zeros(n, n))
            else:
                eq = (Rk == zeros(n, n))
        else:
            eq = (Rk == zeros(n, n))

        res[k_str] = {"eq": eq, "n_k": n}

    return res
