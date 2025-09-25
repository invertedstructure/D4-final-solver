
from __future__ import annotations
from typing import Dict, Any
from .schemas import Boundaries, CMap, Shapes
from .linalg_gf2 import mul

def _as_degree_map(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
        return obj["data"]
    return obj if isinstance(obj, dict) else {}

def _get_dom_cod_for_degree(rep_entry: Any, k_str: str):
    # reps[k] = {"dom": [[...]], "cod": [[...]]}  (nested)
    if isinstance(rep_entry, dict) and "dom" in rep_entry and "cod" in rep_entry:
        return rep_entry["dom"], rep_entry["cod"]
    # reps = {"dom": {k: [[...]]}, "cod": {k: [[...]]}}  (split)
    if isinstance(rep_entry, dict) and "dom" in rep_entry and isinstance(rep_entry["dom"], dict):
        dom = rep_entry["dom"].get(k_str)
        cod = rep_entry.get("cod", {}).get(k_str) if isinstance(rep_entry.get("cod"), dict) else None
        return dom, cod
    return None, None

def _mat_shape(M):
    return (len(M), len(M[0]) if (isinstance(M, list) and M and isinstance(M[0], list)) else 0)

def unit_check(boundaries: Boundaries, cmap: CMap, shapes: Shapes,
               reps: Any = None, enforce_rep_transport: bool = False) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    blocks_d = boundaries.blocks.__root__
    blocks_c = cmap.blocks.__root__
    # Core chain law per degree
    for k_str, d_k in blocks_d.items():
        k = int(k_str)
        Ck = blocks_c.get(k_str)
        Ckm1 = blocks_c.get(str(k-1))
        if Ck is None or Ckm1 is None:
            result[k_str] = {
                "eq": True,
                "note": "skipped (edge degree or missing C blocks)",
                "n_k": len(Ck) if Ck else 0
            }
            continue
        lhs = mul(d_k, Ck)
        rhs = mul(Ckm1, d_k)
        eq = (lhs == rhs)
        result[k_str] = {"eq": eq, "n_k": len(Ck)}
    # Optional: rep transport check
    if enforce_rep_transport:
        if reps is None:
            result["_reps"] = {"ok": False, "error": "rep transport enforced but no reps provided"}
        else:
            ok_all = True
            errors = []
            rep_map = _as_degree_map(reps if not hasattr(reps, "dict") else reps.dict())
            if "data" in rep_map and isinstance(rep_map["data"], dict):
                rep_map = rep_map["data"]
            for k_str, Ck in blocks_c.items():
                dom, cod = None, None
                entry = rep_map.get(k_str) if isinstance(rep_map, dict) else None
                if entry is not None:
                    dom, cod = _get_dom_cod_for_degree(entry, k_str)
                if dom is None or cod is None:
                    dom = rep_map.get("dom", {}).get(k_str) if isinstance(rep_map.get("dom"), dict) else dom
                    cod = rep_map.get("cod", {}).get(k_str) if isinstance(rep_map.get("cod"), dict) else cod
                if dom is None or cod is None:
                    continue  # nothing to check at this degree
                nC = len(Ck)
                r_dom = _mat_shape(dom)
                r_cod = _mat_shape(cod)
                if r_dom[0] != nC or r_cod[0] != nC or r_dom[1] != r_cod[1]:
                    ok_all = False
                    errors.append(f"degree {k_str}: reps shape mismatch; C is {nC}x{nC}, dom {r_dom}, cod {r_cod}")
                    continue
                prod = mul(Ck, dom)
                if prod != cod:
                    ok_all = False
                    errors.append(f"degree {k_str}: C * dom != cod")
            result["_reps"] = {"ok": ok_all, "errors": errors} if not ok_all else {"ok": True, "checked": True}
    return result
