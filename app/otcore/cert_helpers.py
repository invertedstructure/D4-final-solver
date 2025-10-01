# app/otcore/cert_helpers.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from .linalg_gf2 import mul, add, zeros, eye
from . import hashes

def short_id_from_hash(h: str, n: int = 8) -> str:
    return h[:n]

def lane_mask_from_d(dk: List[List[int]]) -> List[int]:
    if not dk or not dk[0]:
        return []
    ncols = len(dk[0])
    return [1 if any(row[j] & 1 for row in dk) else 0 for j in range(ncols)]

def bottom_row(vecmat: List[List[int]]) -> List[int]:
    return [] if not vecmat else vecmat[-1][:]

def support_indices(colvec: List[int]) -> List[int]:
    return [j for j, v in enumerate(colvec) if (v & 1) != 0]

def split_lanes_ker(indices: List[int], lane_mask: List[int]) -> Tuple[List[int], List[int]]:
    lanes = [j for j in indices if lane_mask[j] == 1]
    ker   = [j for j in indices if lane_mask[j] == 0]
    return lanes, ker

def residual_tag_for(lanes: List[int], ker: List[int]) -> str:
    if lanes and ker: return "mixed"
    if lanes:         return "lanes"
    if ker:           return "ker"
    return "none"

def k3_strict_residual(boundaries, cmap, H) -> List[List[int]]:
    # R3 = H2 @ d3 + (C3 + I3)
    blocks_b = boundaries.blocks.__root__
    blocks_c = cmap.blocks.__root__
    blocks_h = H.blocks.__root__
    d3 = blocks_b.get("3"); C3 = blocks_c.get("3"); H2 = blocks_h.get("2")
    if d3 is None or C3 is None or H2 is None:
        return []
    n3 = len(C3)
    I3 = eye(n3)
    return add(mul(H2, d3), add(C3, I3))

def k3_projected_residual(R3_strict: List[List[int]], d3) -> List[List[int]]:
    # Π3 from d3 columns; R3p = R3 @ Π3
    from .projector import projector_columns_from_dkp1
    if not R3_strict or d3 is None: return R3_strict
    P = projector_columns_from_dkp1(d3)
    return mul(R3_strict, P)

def k2_strict_residual(boundaries, cmap, H) -> List[List[int]]:
    # R2 = d2@H2 + H1@d2 + (C2 + I2)   (GF(2); your d2 often 0 and C2=I)
    blocks_b = boundaries.blocks.__root__
    blocks_c = cmap.blocks.__root__
    blocks_h = H.blocks.__root__
    d2 = blocks_b.get("2"); C2 = blocks_c.get("2")
    H1 = blocks_h.get("1"); H2 = blocks_h.get("2")
    if C2 is None:
        return []
    n2 = len(C2); I2 = eye(n2)
    # shape-guarded terms
    def shape(M): return (len(M), len(M[0]) if M and M[0] else 0)
    R = add(C2, I2)
    if d2 is not None and H2 is not None and shape(d2)[1] == shape(H2)[0] == n2:
        R = add(R, mul(d2, H2))
    if H1 is not None and d2 is not None and shape(H1)[1] == shape(d2)[0] and shape(H1)[0] == n2:
        R = add(R, mul(H1, d2))
    return R

def sizes_from_blocks(boundaries) -> Dict[str, int]:
    # tiny dimension snapshot
    blocks_b = boundaries.blocks.__root__
    dims = {}
    for k in ("3","2","1"):
        M = blocks_b.get(k)
        if M: dims[f"n_{k}"] = len(M[0])  # columns (target size)
    return dims

def d_signature_simple(d3: List[List[int]]) -> Dict[str, Any]:
    # rough signature: rank over GF(2) is optional (skip if you want)
    if not d3 or not d3[0]:
        return {"k+1": 3, "rank": None, "ker_dim": None, "lane_pattern": ""}
    lane = "".join("1" if any(row[j] & 1 for row in d3) else "0" for j in range(len(d3[0])))
    # skip rank to avoid writing a GF(2) rref here; ker_dim implied by pattern
    ker_dim = lane.count("0")
    rank = None
    return {"k+1": 3, "rank": rank, "ker_dim": ker_dim, "lane_pattern": lane}
