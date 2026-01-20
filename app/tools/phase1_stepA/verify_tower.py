"""verify_tower.py — Step‑A tower verifier (barcode + link audit).

Reproduction-style entrypoints (see main.pdf):
  python verify_tower.py --base A0.json --rule append_zero --levels 10 --component-only
  python verify_tower.py --base A0.json --rule duplicate_last --levels 10 --full-triple

This tool builds a length-(L+1) tower A0→A1→...→AL by repeatedly applying a Step‑A rule.
It emits a JSON tower_receipt to stdout (or to --out).

Exit code:
  0 if the tower is self-consistent under the requested mode
  2 otherwise
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import stepa_v1 as S
from phase1_paths import resolve_path


def _load_json(path: str) -> Any:
    rp = resolve_path(path, role="json")
    return json.loads(rp.resolved.read_text(encoding="utf-8"))


def _load_matrix(path: str) -> List[List[int]]:
    return S.load_matrix_obj(_load_json(path))


def _pick_rule_map(rule_id: str, A: List[List[int]], map_entries: List[Dict[str, Any]]) -> S.StepAMap:
    entry = S.resolve_map_entry(rule_id, map_entries)
    return S.map_entry_to_concrete_ops(entry, n_cols=S.shape(A)[1])


def _build_tower(A0: List[List[int]], rule_id: str, levels: int, map_entries: List[Dict[str, Any]]) -> List[Tuple[str, List[List[int]]]]:
    mats: List[Tuple[str, List[List[int]]]] = [("0", A0)]
    A = A0
    for k in range(1, int(levels) + 1):
        mp = _pick_rule_map(rule_id, A, map_entries)
        A = S.apply_stepA_map(A, mp)
        mats.append((str(k), A))
    return mats


def _barcode_constant(bc: Dict[str, Any]) -> bool:
    seq = bc.get("sequence") or []
    hashes = [e.get("hash") for e in seq if isinstance(e, dict) and e.get("status") == "OK"]
    if not hashes:
        return False
    h0 = hashes[0]
    return all(h == h0 for h in hashes)


def _tower_receipt(
    mats: List[Tuple[str, List[List[int]]]],
    *,
    rule_id: str,
    mode: str,
    map_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Barcode
    bc = S.barcode_from_mats(mats, barcode_kind=("component" if mode == "component" else "full"))

    # Link audit
    links: List[Dict[str, Any]] = []
    match = 0
    for i in range(len(mats) - 1):
        k0, A0 = mats[i]
        k1, A1 = mats[i + 1]

        gates = S.stepA_link_gate(A0, A1)
        type_ok = (gates.get("type_gate") or {}).get("status") == "PASS"
        cert_ok = (gates.get("cert_gate") or {}).get("status") == "PASS"

        mp = _pick_rule_map(rule_id, A0, map_entries)
        cert0 = S.stepA_cert(A0)
        y0 = cert0.get("y")
        comp0 = cert0.get("A_comp")

        transport_ok = None
        comp_hat = None
        comp_dst = None
        if cert0.get("defined") and isinstance(y0, str) and isinstance(comp0, list):
            try:
                comp_hat = S.transport_A_comp(A0, y0, comp0, mp)
                comp_dst = S.A_comp_with_y(A1, y0)
                transport_ok = bool(comp_hat == comp_dst)
            except Exception as e:
                transport_ok = False
                comp_hat = None
                comp_dst = None
                gates = dict(gates)
                gates["transport_error"] = str(e)

        if mode == "full":
            ok = bool(type_ok and cert_ok)
        else:
            # component-only: type gate must pass, and transport equality must hold when defined.
            ok = bool(type_ok and (transport_ok is True or transport_ok is None))

        if ok:
            match += 1

        links.append(
            {
                "from": k0,
                "to": k1,
                "map": {"map_id": mp.map_id, "ops": [op.__dict__ for op in mp.ops]},
                "type_gate": gates.get("type_gate"),
                "cert_gate": gates.get("cert_gate"),
                "transport_ok": transport_ok,
                "A_comp_transport": comp_hat,
                "A_comp_dst": comp_dst,
                "status": "MATCH" if ok else "MISMATCH",
            }
        )

    L = int(len(mats) - 1)
    summary = {
        "levels": L,
        "links": L,
        "links_matched": match,
        "links_mismatched": L - match,
        "barcode_constant": _barcode_constant(bc),
    }

    verdict_ok = bool(summary["links_mismatched"] == 0 and (summary["barcode_constant"] or mode == "full"))

    return {
        "tool": "verify_tower",
        "schema_version": "tower_receipt.v1",
        "base": mats[0][0] if mats else "0",
        "rule_id": rule_id,
        "mode": mode,
        "horizon": {"L": L},
        "barcode": bc,
        "links": links,
        "summary": summary,
        "verdict": "PASS" if verdict_ok else "FAIL",
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to base matrix JSON (A0)")
    ap.add_argument("--rule", required=True, help="Rule/map_id to apply at each step")
    ap.add_argument("--levels", type=int, default=10, help="Tower depth (L), build levels 0..L")
    ap.add_argument("--maps", help="Optional path to maps_stepA.json; default is built-in")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--component-only", action="store_true")
    mode.add_argument("--full-triple", action="store_true")

    ap.add_argument("--out", help="Write receipt JSON to this path instead of stdout")

    args = ap.parse_args(argv)

    A0 = _load_matrix(args.base)
    # Path plumbing: prefer an on-disk frozen map suite (maps_stepA.json) when
    # present, but keep the legacy in-code default as a fallback.
    if not args.maps:
        try:
            map_entries = S.load_map_suite_obj(_load_json("maps_stepA.json"))
        except Exception:
            map_entries = S.default_map_suite_entries()
    else:
        map_entries = S.load_map_suite_obj(_load_json(args.maps))

    mats = _build_tower(A0, rule_id=args.rule, levels=args.levels, map_entries=map_entries)
    receipt = _tower_receipt(
        mats,
        rule_id=args.rule,
        mode=("component" if args.component_only else "full"),
        map_entries=map_entries,
    )

    out_txt = json.dumps(receipt, indent=2, ensure_ascii=True)
    if args.out:
        Path(args.out).write_text(out_txt + "\n", encoding="utf-8")
    else:
        print(out_txt)

    return 0 if receipt.get("verdict") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
