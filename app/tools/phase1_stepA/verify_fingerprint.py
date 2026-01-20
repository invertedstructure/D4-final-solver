"""verify_fingerprint.py — Step‑A full-triple (fp.v1) invariance verifier.

Reproduction-style entrypoints (see main.pdf):
  python verify_fingerprint.py --matrix A.json --map duplicate_last --full-triple
  python verify_fingerprint.py --suite seeds.json --maps maps_stepA.json --full-triple

Checks:
  - Step‑A TypeGate(A, A') (structural admissibility proxy)
  - CertGate: fp.v1 SHA-256 equality when defined
  - Canonical triple equality (y*, t*, A_comp) when defined

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


def _one_case(A: List[List[int]], map_entry: Dict[str, Any]) -> Dict[str, Any]:
    # Instantiate and apply map
    A0 = S.norm_bitmatrix(A, name="A")
    mp = S.map_entry_to_concrete_ops(map_entry, n_cols=(len(A0[0]) if A0 else 0))
    A1 = S.apply_stepA_map(A0, mp)

    tg = S.stepA_type_gate(A0, A1)

    c0 = S.stepA_cert(A0)
    c1 = S.stepA_cert(A1)
    fp0 = S.fpv1_sha256(c0)
    fp1 = S.fpv1_sha256(c1)

    if not (fp0.get("defined") and fp1.get("defined")):
        cg = {"status": "UNDEFINED", "reason": "CERT_UNDEFINED"}
        triple_ok = False
    else:
        cg = {
            "status": "PASS" if fp0.get("sha256") == fp1.get("sha256") else "FAIL",
            "fp0": fp0.get("sha256"),
            "fp1": fp1.get("sha256"),
        }
        triple_ok = bool(
            c0.get("y") == c1.get("y")
            and c0.get("t") == c1.get("t")
            and c0.get("A_comp") == c1.get("A_comp")
        )

    ok = bool(tg.get("status") == "PASS" and cg.get("status") == "PASS" and triple_ok)
    return {
        "map": {"map_id": mp.map_id, "ops": [op.__dict__ for op in mp.ops]},
        "type_gate": tg,
        "cert_gate": cg,
        "triple_ok": bool(triple_ok),
        "cert0": c0,
        "cert1": c1,
        "fp0": {"defined": fp0.get("defined"), "sha256": fp0.get("sha256")},
        "fp1": {"defined": fp1.get("defined"), "sha256": fp1.get("sha256")},
        "verdict": "PASS" if ok else "FAIL",
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix", help="Path to a matrix JSON (list[list[int]] or {matrix: ...})")
    src.add_argument("--suite", help="Path to a seed suite JSON")

    ap.add_argument("--map", dest="map_id", help="Apply one named map_id from the map suite")
    ap.add_argument("--maps", help="Path to maps_stepA.json (defaults to built-in suite)")
    ap.add_argument("--full-triple", action="store_true", help="Compatibility flag (no-op; this tool is full-triple)")

    args = ap.parse_args(argv)

    maps = _load_maps(args.maps)
    if args.map_id:
        maps = [m for m in maps if str(m.get("map_id")) == str(args.map_id)]
        if not maps:
            print(json.dumps({"verdict": "FAIL", "reason": f"MAP_NOT_FOUND: {args.map_id}"}, indent=2))
            return 2

    cases: List[Dict[str, Any]] = []

    if args.matrix:
        mat_obj = _load_json(args.matrix)
        A = S.load_matrix_obj(mat_obj)
        for m in maps:
            cases.append({"map_id": m.get("map_id"), "result": _one_case(A, m)})
    else:
        suite = _load_suite(args.suite)
        for seed in suite:
            A = seed.get("matrix")
            if A is None:
                continue
            for m in maps:
                cases.append(
                    {
                        "seed_id": seed.get("seed_id"),
                        "label": seed.get("label"),
                        "map_id": m.get("map_id"),
                        "result": _one_case(A, m),
                    }
                )

    ok_all = all(c.get("result", {}).get("verdict") == "PASS" for c in cases)
    out = {
        "tool": "verify_fingerprint",
        "verdict": "PASS" if ok_all else "FAIL",
        "n_cases": len(cases),
        "cases": cases,
    }
    print(json.dumps(out, indent=2, sort_keys=False))
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
