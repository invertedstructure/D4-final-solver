
from __future__ import annotations
from typing import Dict
from .schemas import Boundaries, CMap
from .linalg_gf2 import mul, add, zeros, eye

import os, json
st.write("cfg file in CWD:", os.path.exists("projection_config.json"))

from .projector import load_projection_config, projector_columns_from_dkp1
cfg = load_projection_config("projection_config.json")
st.json({"cfg": cfg})


def overlap_check(boundaries: Boundaries, cmap: CMap, homotopy: CMap) -> Dict[str, dict]:
    res: Dict[str, dict] = {}
    d_blocks = boundaries.blocks.__root__
    C_blocks = cmap.blocks.__root__
    H_blocks = homotopy.blocks.__root__
    degs = sorted(int(k) for k in set(d_blocks.keys()) | set(C_blocks.keys()))
    for k in degs:
        k_str = str(k)
        Ck = C_blocks.get(k_str)
        if Ck is None:
            continue
        n = len(Ck)
        Ik = eye(n)
        term1 = None
        term2 = None
        if k_str in H_blocks and str(k+1) in d_blocks:
            Hk = H_blocks[k_str]              
            dk1 = d_blocks[str(k+1)]          
            term1 = mul(dk1, Hk)              
        if str(k-1) in H_blocks and k_str in d_blocks:
            Hkm1 = H_blocks[str(k-1)]         
            dk = d_blocks[k_str]              
            term2 = mul(Hkm1, dk)             
        if term1 is not None and term2 is not None:
            RHS = add(term1, term2)
        elif term1 is not None:
            RHS = term1
        elif term2 is not None:
            RHS = term2
        else:
            RHS = zeros(n, n)
        from .linalg_gf2 import add as add_gf2
        LHS = add_gf2(Ck, Ik)
        eq = (LHS == RHS)
        res[k_str] = {"eq": eq, "n_k": n}
    return res
