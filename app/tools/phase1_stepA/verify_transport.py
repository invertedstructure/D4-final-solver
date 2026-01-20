"""verify_transport.py — Step‑A parity transport verifier (Phase‑1).

Reproduction-style entrypoints (see main.pdf):
  python verify_transport.py --matrix A.json --map append_zero --component-only
  python verify_transport.py --suite seeds.json --maps maps_stepA.json --component-only

This tool is read-only: it does not mutate certificates or solver state.

Exit code:
  0 if all checks PASS
  2 if any check FAIL (or input invalid)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import stepa_v1 as S
from phase1_paths import resolve_path


def _load_json(path: str) -> Any:
    rp = resolve_path(path, role="json")
    return json.loads(rp.resolved.read_text(encoding="utf-8"))


def _load_suite(path: str) -> List[Dict[str, Any]]:
    obj = _load_json(path)
    return S.load_seed_suite_obj(obj)


def _load_maps(path: Optional[str]) -> List[Dict[str, Any]]:
    # Path plumbing: prefer an on-disk frozen map suite (maps_stepA.json) when
    # present, but keep the legacy in-code default as a fallback.
    if not path:
        try:
            obj = _load_json("maps_stepA.json")
            return S.load_map_suite_obj(obj)
        except Exception:
            return S.default_map_suite_entries()
    obj = _load_json(path)
    return S.load_map_suite_obj(obj)


def _pick_map(maps: List[Dict[str, Any]], map_id: str) -> Dict[str, Any]:
    for m in maps:
        if str(m.get("map_id")) == str(map_id):
            return m
    raise SystemExit(f"Map id not found: {map_id}")


def run_one(A: List[List[int]], map_entry: Dict[str, Any]) -> Dict[str, Any]:
    cert0 = S.stepA_cert(A)
    if not cert0.get("defined"):
        return {
            "status": "UNDEFINED",
            "reason": cert0.get("reason"),
            "cert0": cert0,
        }

    mp = S.map_entry_to_concrete_ops(map_entry, n_cols=S.shape(A)[1])
    A1 = S.apply_stepA_map(A, mp)

    # Transport check (Corollary 1): transport A_comp(A, y*) and compare to recompute on A'.
    y = cert0["y"]
    comp0 = S.A_comp_with_y(A, y)
    comp_hat = S.transport_A_comp(A, y, comp0, mp)
    comp1 = S.A_comp_with_y(A1, y)

    tg = S.stepA_type_gate(A, A1)
    ok = bool(comp_hat == comp1)

    return {
        "status": "PASS" if ok else "FAIL",
        "type_gate": tg,
        "map_id": map_entry.get("map_id"),
        "map": {"map_id": mp.map_id, "ops": [op.__dict__ for op in mp.ops]},
        "y_star": y,
        "A_comp_src": comp0,
        "A_comp_transport": comp_hat,
        "A_comp_dst": comp1,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix", help="Path to a matrix JSON (list[list[0/1]] or {matrix: ...}).")
    src.add_argument("--suite", help="Path to a seed suite JSON.")

    ap.add_argument("--map", dest="map_id", help="Single map_id to apply (e.g., append_zero).")
    ap.add_argument("--maps", dest="maps_path", help="Path to maps_stepA.json (map suite).")

    ap.add_argument("--component-only", action="store_true", help="(Compatibility flag) Run component transport checks.")

    args = ap.parse_args(argv)

    maps = _load_maps(args.maps_path)

    # Determine maps to run
    if args.map_id:
        maps_to_run = [_pick_map(maps, args.map_id)]
    else:
        maps_to_run = list(maps)

    results: List[Dict[str, Any]] = []
    any_fail = False

    if args.matrix:
        A = S.load_matrix_obj(_load_json(args.matrix))
        for m in maps_to_run:
            r = run_one(A, m)
            results.append(r)
            if r.get("status") not in ("PASS", "UNDEFINED"):
                any_fail = True
    else:
        suite = _load_suite(args.suite)
        for seed in suite:
            A = S.load_matrix_obj(seed.get("matrix"))
            for m in maps_to_run:
                r = run_one(A, m)
                r["seed_id"] = seed.get("seed_id")
                results.append(r)
                if r.get("status") not in ("PASS", "UNDEFINED"):
                    any_fail = True

    out = {
        "tool": "verify_transport",
        "schema_version": "verify_transport.v1",
        "component_only": bool(args.component_only or True),
        "verdict": "FAIL" if any_fail else "PASS",
        "cases": results,
    }
    print(json.dumps(out, indent=2, sort_keys=False))
    return 2 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
