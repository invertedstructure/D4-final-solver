"""stepa_v1.py — Phase-1 Step‑A reference utilities (Mode B wiring).

This module is intentionally small, dependency-free (standard library only), and
implements the frozen Step‑A surfaces used by the verifier scripts:

- Canonical certificate (y*, t*, A_comp) over F2
- fp.v1 hash core + SHA-256 over CanonicalJSON(core)
- Step‑A TypeGate and LinkGate receipts
- Step‑A admissible map application + component transport
- Barcode builder for towers

The definitions here mirror the Step‑A receipts wired into streamlit_app_rigored.py,
with the same left-to-right bitstring conventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Frozen profile identifiers (Phase‑1)
# -----------------------------------------------------------------------------

INV_STEP_A_PROFILE_ID = "axiomA.stepA.degree_pure.v1"
FPV1_PROFILE_ID = "fp.v1.fixed_key_order.v1"
TYPE_GATE_ID = "StepA.TypeGate.no_new_nonzero_type.v1"
BARCODE_PROFILE_ID = "barcode.v1.sequence_equality.v1"

# Enumerate kernel vectors only up to this dimension (2^k combinations).
MAX_KERNEL_ENUM_DIM = 20


# -----------------------------------------------------------------------------
# Basic helpers (bitstrings, matrices)
# -----------------------------------------------------------------------------

def popcount(x: int) -> int:
    return int(int(x).bit_count())


def shape(M: List[List[int]]) -> Tuple[int, int]:
    if not M:
        return (0, 0)
    return (len(M), len(M[0]) if isinstance(M[0], list) else 0)


def norm_bitmatrix(mat: Any, *, name: str = "A") -> List[List[int]]:
    """Normalize arbitrary JSON-ish input to list[list[int]] over F2.

    - Accepts [] and returns []
    - Enforces non-ragged rows
    - Coerces int/bool-like values; rejects floats/None
    """
    if mat is None:
        return []
    if not isinstance(mat, list):
        raise TypeError(f"{name}: expected list (rows), got {type(mat).__name__}")
    if len(mat) == 0:
        return []
    # row check
    if not all(isinstance(r, list) for r in mat):
        raise TypeError(f"{name}: expected list[list[int]]")
    n = len(mat[0])
    for i, r in enumerate(mat):
        if len(r) != n:
            raise ValueError(f"{name}: ragged rows at row {i} (len {len(r)} != {n})")

    out: List[List[int]] = []
    for r_i, row in enumerate(mat):
        out_row: List[int] = []
        for c_i, v in enumerate(row):
            if isinstance(v, bool):
                iv = 1 if v else 0
            elif isinstance(v, int):
                iv = int(v)
            else:
                # Reject floats/None/str
                raise TypeError(f"{name}[{r_i}][{c_i}] not int-like: {type(v).__name__}")
            out_row.append(iv & 1)
        out.append(out_row)
    return out


def col_as_bits(A: List[List[int]], j: int) -> List[int]:
    return [int(A[i][j]) & 1 for i in range(len(A))]


def vec_to_bitstring(v_bits: List[int]) -> str:
    # Frozen: v[0] is the leftmost bit.
    return "".join("1" if (int(b) & 1) else "0" for b in (v_bits or []))


def vec_to_intmask(v_bits: List[int]) -> int:
    """Internal bitmask convention: bit i corresponds to v[i]."""
    m = 0
    for i, b in enumerate(v_bits or []):
        if int(b) & 1:
            m |= (1 << i)
    return int(m)


def intmask_to_bitstring(x: int, m: int) -> str:
    return "".join("1" if ((int(x) >> i) & 1) else "0" for i in range(int(m)))


def bitstring_to_intmask(bs: str) -> int:
    x = 0
    for i, ch in enumerate(bs or ""):
        if ch == "1":
            x |= (1 << i)
        elif ch == "0":
            continue
        else:
            raise ValueError(f"bad bitstring char at {i}: {ch!r}")
    return int(x)


def dot_mask(a: int, b: int) -> int:
    return popcount(int(a) & int(b)) & 1


def col_types(A: List[List[int]], *, include_zero: bool = False) -> set[str]:
    """Return set of distinct column-type bitstrings."""
    m, n = shape(A)
    if m == 0 or n == 0:
        return {""} if include_zero else set()
    types: set[str] = set()
    for j in range(n):
        col = col_as_bits(A, j)
        if (not include_zero) and all(b == 0 for b in col):
            continue
        types.add(vec_to_bitstring(col))
    return types


# -----------------------------------------------------------------------------
# Step‑A TypeGate
# -----------------------------------------------------------------------------

def stepA_type_gate(A0: Any, A1: Any) -> Dict[str, Any]:
    """Directed structural gate A0→A1.

    PASS iff:
      - no NEW nonzero column types appear
      - no MISSING nonzero column types (no deletions)
      - no DECREASED multiplicities for any nonzero type
      - n does not decrease

    Null (all-zero) type is excluded from comparisons.
    """
    try:
        A0n = norm_bitmatrix(A0, name="A0")
        A1n = norm_bitmatrix(A1, name="A1")
    except Exception as e:
        return {"status": "NA", "reason": f"BAD_MATRIX: {e}"}

    m0, n0 = shape(A0n)
    m1, n1 = shape(A1n)
    if m0 != m1:
        return {"status": "NA", "reason": f"ROW_MISMATCH: m0={m0}, m1={m1}", "m0": m0, "m1": m1}

    def _counts(A: List[List[int]]) -> Dict[str, int]:
        m, n = shape(A)
        out: Dict[str, int] = {}
        if m == 0 or n == 0:
            return out
        for j in range(n):
            col = col_as_bits(A, j)
            if all(b == 0 for b in col):
                continue
            bs = vec_to_bitstring(col)
            out[bs] = out.get(bs, 0) + 1
        return out

    c0 = _counts(A0n)
    c1 = _counts(A1n)
    t0 = set(c0.keys())
    t1 = set(c1.keys())

    new_types = sorted(list(t1.difference(t0)))
    missing_types = sorted(list(t0.difference(t1)))

    decreased_types: List[str] = []
    for bs, k0 in c0.items():
        k1 = c1.get(bs, 0)
        if k1 < k0:
            decreased_types.append(bs)
    decreased_types.sort()

    n_monotone = bool(n1 >= n0)

    ok = (not new_types) and (not missing_types) and (not decreased_types) and n_monotone
    return {
        "profile_id": TYPE_GATE_ID,
        "status": "PASS" if ok else "FAIL",
        "new_nonzero_types": new_types,
        "missing_nonzero_types": missing_types,
        "decreased_nonzero_types": decreased_types,
        "types0": int(len(t0)),
        "types1": int(len(t1)),
        "n0": int(n0),
        "n1": int(n1),
        "n_monotone": bool(n_monotone),
    }


# -----------------------------------------------------------------------------
# GF(2) linear algebra for ker(A^T)
# -----------------------------------------------------------------------------

def gf2_rref(rows: List[int], n_vars: int) -> Tuple[List[int], List[int]]:
    """RREF over GF(2) for bitmask rows."""
    rr = [int(r) for r in (rows or [])]
    pivots: List[int] = []
    r = 0
    n_rows = len(rr)
    for c in range(int(n_vars)):
        if r >= n_rows:
            break
        piv = None
        for i in range(r, n_rows):
            if (rr[i] >> c) & 1:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            rr[r], rr[piv] = rr[piv], rr[r]
        for i in range(n_rows):
            if i != r and ((rr[i] >> c) & 1):
                rr[i] ^= rr[r]
        pivots.append(int(c))
        r += 1
    return rr, pivots


def gf2_nullspace_basis(eq_rows: List[int], n_vars: int) -> List[int]:
    """Return a GF(2) nullspace basis for eq_rows * x = 0."""
    rr, pivots = gf2_rref(eq_rows, n_vars)
    pivot_set = set(pivots)
    free_cols = [c for c in range(int(n_vars)) if c not in pivot_set]

    pivot_row_by_col: Dict[int, int] = {}
    for i, c in enumerate(pivots):
        if i < len(rr):
            pivot_row_by_col[int(c)] = int(rr[i])

    basis: List[int] = []
    for f in free_cols:
        x = (1 << int(f))
        for p in pivots:
            row = pivot_row_by_col.get(int(p), 0)
            rhs = popcount(row & x) & 1
            if rhs:
                x |= (1 << int(p))
            else:
                x &= ~(1 << int(p))
        if x != 0:
            basis.append(int(x))
    return basis


# -----------------------------------------------------------------------------
# Step‑A certificate + fp.v1
# -----------------------------------------------------------------------------

def stepA_cert(A: Any) -> Dict[str, Any]:
    """Compute Step‑A certificate triple (y*, t*, A_comp) if defined.

    Frozen:
      - y* ∈ ker(A^T), y* != 0; chosen by min Hamming weight, then lex on bitstring.
      - t* ∉ im(A); chosen by min weight then lex (implemented via ker(A^T) support).
      - A_comp = { columns c with y*·c = 1 }, multiplicities mod 2, lex-sorted.

    Returns dict with keys:
      defined, reason, m, n, y, t, A_comp, kernel_dim
    """
    A = norm_bitmatrix(A, name="A")
    m, n = shape(A)

    if m == 0:
        return {"defined": False, "reason": "M_ZERO_ROWS", "m": int(m), "n": int(n), "y": None, "t": None, "A_comp": [], "kernel_dim": 0}

    # Special: n==0 means A^T has no equations: ker(A^T) = F2^m.
    if n == 0:
        y_mask = 1 << (m - 1)
        y_bs = intmask_to_bitstring(y_mask, m)
        t_mask = 1 << (m - 1)
        t_bs = intmask_to_bitstring(t_mask, m)
        return {"defined": True, "reason": None, "m": int(m), "n": int(n), "y": y_bs, "t": t_bs, "A_comp": [], "kernel_dim": int(m)}

    # Build equations for A^T y = 0: each column is one equation dot(col_j, y)=0.
    eq_rows: List[int] = []
    for j in range(n):
        col_bits = col_as_bits(A, j)
        eq_rows.append(vec_to_intmask(col_bits))

    basis = gf2_nullspace_basis(eq_rows, m)
    kdim = int(len(basis))
    if kdim == 0:
        return {"defined": False, "reason": "KERNEL_TRIVIAL", "m": int(m), "n": int(n), "y": None, "t": None, "A_comp": [], "kernel_dim": 0}

    if kdim > MAX_KERNEL_ENUM_DIM:
        return {"defined": False, "reason": f"KERNEL_DIM_TOO_LARGE(k={kdim},max={MAX_KERNEL_ENUM_DIM})", "m": int(m), "n": int(n), "y": None, "t": None, "A_comp": [], "kernel_dim": kdim}

    best_mask: Optional[int] = None
    best_key: Optional[Tuple[int, str]] = None

    for comb in range(1, 1 << kdim):
        v = 0
        for i in range(kdim):
            if (comb >> i) & 1:
                v ^= int(basis[i])
        if v == 0:
            continue
        w = popcount(v)
        bs = intmask_to_bitstring(v, m)
        key = (int(w), bs)
        if best_key is None or key < best_key:
            best_key = key
            best_mask = int(v)

    if best_mask is None:
        return {"defined": False, "reason": "KERNEL_ENUM_EMPTY", "m": int(m), "n": int(n), "y": None, "t": None, "A_comp": [], "kernel_dim": kdim}

    y_mask = int(best_mask)
    y_bs = intmask_to_bitstring(y_mask, m)

    # Canonical top-support: choose a weight-1 unit vector in support of ker(A^T).
    support_mask = 0
    for v in basis:
        support_mask |= int(v)
    if support_mask == 0:
        return {"defined": False, "reason": "KERNEL_SUPPORT_EMPTY", "m": int(m), "n": int(n), "y": None, "t": None, "A_comp": [], "kernel_dim": kdim}

    max_i: Optional[int] = None
    for i in range(m - 1, -1, -1):
        if (support_mask >> i) & 1:
            max_i = int(i)
            break
    if max_i is None:
        return {"defined": False, "reason": "NO_SUPPORT_INDEX", "m": int(m), "n": int(n), "y": None, "t": None, "A_comp": [], "kernel_dim": kdim}

    t_mask = 1 << int(max_i)
    t_bs = intmask_to_bitstring(t_mask, m)

    # Component set with multiplicities mod 2.
    comp_counts: Dict[str, int] = {}
    for j in range(n):
        c_bits = col_as_bits(A, j)
        c_mask = vec_to_intmask(c_bits)
        if dot_mask(y_mask, c_mask) == 1:
            bs = intmask_to_bitstring(c_mask, m)
            comp_counts[bs] = (comp_counts.get(bs, 0) + 1) & 1

    A_comp = sorted([bs for bs, par in comp_counts.items() if par == 1])

    return {"defined": True, "reason": None, "m": int(m), "n": int(n), "y": y_bs, "t": t_bs, "A_comp": A_comp, "kernel_dim": kdim}


def fpv1_payload_core(cert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """fp.v1 hash core payload, with frozen key order."""
    y = cert.get("y")
    t = cert.get("t")
    A_comp = cert.get("A_comp")
    if not (cert.get("defined") is True and isinstance(y, str) and isinstance(t, str) and isinstance(A_comp, list)):
        return None

    return {
        "field": "F2",
        "lex_order": "left-to-right",
        "y": y,
        "t": t,
        "A_comp": list(A_comp),
    }


def fpv1_sha256(cert: Dict[str, Any]) -> Dict[str, Any]:
    """SHA-256 over CanonicalJSON(core_payload) (compact, no key sorting)."""
    core = fpv1_payload_core(cert)
    meta = {"schema": "fp_v1", "schema_version": "fp.v1", "profile_id": FPV1_PROFILE_ID}
    if core is None:
        return {"defined": False, "sha256": None, "canonical_json": None, "payload_core": None, "meta": meta, "reason": (cert.get("reason") or "UNDEFINED")}

    txt = json.dumps(core, separators=(",", ":"), sort_keys=False, ensure_ascii=True, allow_nan=False)
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    return {"defined": True, "sha256": h, "canonical_json": txt, "payload_core": core, "meta": meta}


def stepA_link_gate(A0: Any, A1: Any) -> Dict[str, Any]:
    """Compute (TypeGate, CertGate) receipts for a chosen pair."""
    tg = stepA_type_gate(A0, A1)

    try:
        c0 = stepA_cert(A0)
        c1 = stepA_cert(A1)
        f0 = fpv1_sha256(c0)
        f1 = fpv1_sha256(c1)
        if not (f0.get("defined") and f1.get("defined")):
            cg = {"status": "UNDEFINED", "reason": "CERT_UNDEFINED"}
        else:
            cg = {
                "status": "PASS" if f0.get("sha256") == f1.get("sha256") else "FAIL",
                "fp0": f0.get("sha256"),
                "fp1": f1.get("sha256"),
            }
    except Exception as e:
        cg = {"status": "UNDEFINED", "reason": f"CERT_ERROR: {e}"}

    return {"type_gate": tg, "cert_gate": cg}


# -----------------------------------------------------------------------------
# Step‑A admissible maps + component transport
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class StepAOp:
    op: str  # "permute" | "duplicate" | "append_zero"
    perm: Optional[List[int]] = None
    src: Optional[str] = None  # "first" | "last" | "index"
    index: Optional[int] = None
    count: Optional[int] = None


@dataclass(frozen=True)
class StepAMap:
    map_id: str
    ops: List[StepAOp]


def _rotate_left_perm(n: int) -> List[int]:
    if n <= 0:
        return []
    return list(range(1, n)) + [0]


def instantiate_map(map_id: str, n_cols: int) -> StepAMap:
    """Return a concrete StepAMap for a known map_id.

    Frozen map suite (Phase‑1):
      append_zero
      duplicate_last
      duplicate_first
      permute_then_append_zero   (rotate-left-1, then append_zero)
      append_copy_random         (deterministic: duplicate middle_floor index)

    NOTE: "append_copy_random" is deterministic by design here (middle_floor).
    """
    mid = int(n_cols // 2) if n_cols > 0 else 0

    if map_id == "append_zero":
        return StepAMap(map_id=map_id, ops=[StepAOp(op="append_zero", count=1)])
    if map_id == "duplicate_last":
        return StepAMap(map_id=map_id, ops=[StepAOp(op="duplicate", src="last")])
    if map_id == "duplicate_first":
        return StepAMap(map_id=map_id, ops=[StepAOp(op="duplicate", src="first")])
    if map_id == "permute_then_append_zero":
        return StepAMap(
            map_id=map_id,
            ops=[StepAOp(op="permute", perm=_rotate_left_perm(n_cols)), StepAOp(op="append_zero", count=1)],
        )
    if map_id == "append_copy_random":
        return StepAMap(map_id=map_id, ops=[StepAOp(op="duplicate", src="index", index=mid)])

    raise KeyError(f"Unknown map_id: {map_id}")


def apply_stepA_map(A: Any, mp: StepAMap) -> List[List[int]]:
    """Apply a StepAMap to matrix A (column-only)."""
    A = norm_bitmatrix(A, name="A")
    m, n = shape(A)

    # Represent as column masks for easier ops, then re-expand.
    cols: List[int] = []
    for j in range(n):
        cols.append(vec_to_intmask(col_as_bits(A, j)))

    for op in mp.ops:
        if op.op == "permute":
            perm = op.perm or []
            if len(perm) != len(cols):
                raise ValueError(f"permute: perm length {len(perm)} != n_cols {len(cols)}")
            cols = [cols[i] for i in perm]
        elif op.op == "append_zero":
            c = int(op.count or 0)
            if c < 0:
                raise ValueError("append_zero: count must be >= 0")
            cols.extend([0] * c)
        elif op.op == "duplicate":
            if not cols:
                # Duplicating in a 0-col matrix is a no-op under Step-A tool semantics.
                continue
            src = op.src or "last"
            if src == "first":
                idx = 0
            elif src == "last":
                idx = len(cols) - 1
            elif src == "index":
                if op.index is None:
                    raise ValueError("duplicate: src='index' requires index")
                idx = int(op.index)
                if idx < 0 or idx >= len(cols):
                    raise ValueError(f"duplicate: index {idx} out of range [0,{len(cols)-1}]")
            else:
                raise ValueError(f"duplicate: unknown src {src!r}")
            cols.append(int(cols[idx]))
        else:
            raise ValueError(f"Unknown op: {op.op}")

    # Expand back into row-major list[list[int]].
    n2 = len(cols)
    out: List[List[int]] = [[0 for _ in range(n2)] for _ in range(m)]
    for j, mask in enumerate(cols):
        for i in range(m):
            out[i][j] = 1 if ((int(mask) >> i) & 1) else 0
    return out


def A_comp_with_y(A: Any, y_bs: str) -> List[str]:
    """Compute A_comp(A,y) with multiplicities mod 2 and lex-sorted."""
    A = norm_bitmatrix(A, name="A")
    m, n = shape(A)
    if len(y_bs or "") != m:
        raise ValueError(f"y length {len(y_bs or '')} != m {m}")

    y_mask = bitstring_to_intmask(y_bs)

    comp_counts: Dict[str, int] = {}
    for j in range(n):
        c_mask = vec_to_intmask(col_as_bits(A, j))
        if dot_mask(y_mask, c_mask) == 1:
            bs = intmask_to_bitstring(c_mask, m)
            comp_counts[bs] = (comp_counts.get(bs, 0) + 1) & 1

    return sorted([bs for bs, par in comp_counts.items() if par == 1])


def transport_A_comp(A: Any, y_bs: str, A_comp_src: List[str], mp: StepAMap) -> List[str]:
    """Compute Transport_ι(A_comp(A,y)) for Step‑A column-only ops.

    Transport pushes columns through ι, reduces multiplicities mod 2, then lex-sorts.

    For the frozen op vocabulary this reduces to toggling membership when a duplicate
    of a y·c=1 column is appended.

    Implementation note: transport is computed *without* recomputing from scratch on A' —
    it updates the parity-set as ops are applied.
    """
    A = norm_bitmatrix(A, name="A")
    m, n = shape(A)
    if len(y_bs or "") != m:
        raise ValueError(f"y length {len(y_bs or '')} != m {m}")

    y_mask = bitstring_to_intmask(y_bs)

    # Work in column masks to track the evolving matrix under ops.
    cols: List[int] = [vec_to_intmask(col_as_bits(A, j)) for j in range(n)]

    parity: set[str] = set(A_comp_src or [])

    for op in mp.ops:
        if op.op == "permute":
            perm = op.perm or []
            if len(perm) != len(cols):
                raise ValueError(f"permute: perm length {len(perm)} != n_cols {len(cols)}")
            cols = [cols[i] for i in perm]
            # Parity-set is type-based; permutation does not change it.
        elif op.op == "append_zero":
            c = int(op.count or 0)
            if c < 0:
                raise ValueError("append_zero: count must be >= 0")
            cols.extend([0] * c)
            # zero has y·0 = 0; no toggle.
        elif op.op == "duplicate":
            if not cols:
                continue
            src = op.src or "last"
            if src == "first":
                idx = 0
            elif src == "last":
                idx = len(cols) - 1
            elif src == "index":
                if op.index is None:
                    raise ValueError("duplicate: src='index' requires index")
                idx = int(op.index)
                if idx < 0 or idx >= len(cols):
                    raise ValueError(f"duplicate: index {idx} out of range [0,{len(cols)-1}]")
            else:
                raise ValueError(f"duplicate: unknown src {src!r}")
            mask_src = int(cols[idx])
            cols.append(mask_src)

            if dot_mask(y_mask, mask_src) == 1:
                bs = intmask_to_bitstring(mask_src, m)
                if bs in parity:
                    parity.remove(bs)
                else:
                    parity.add(bs)
        else:
            raise ValueError(f"Unknown op: {op.op}")

    return sorted(parity)


# -----------------------------------------------------------------------------
# Barcode builder
# -----------------------------------------------------------------------------

def barcode_from_mats(mats: List[Tuple[str, Any]], *, barcode_kind: str = "full") -> Dict[str, Any]:
    """Compute a barcode sequence over (level_name, matrix) items."""
    kind = str(barcode_kind or "full")
    seq: List[Dict[str, Any]] = []

    for name, A in mats:
        try:
            cert = stepA_cert(A)
            fp = fpv1_sha256(cert)
            if not fp.get("defined"):
                seq.append({"level": name, "status": "UNDEFINED", "reason": cert.get("reason")})
                continue

            if kind == "component":
                comp = cert.get("A_comp") if isinstance(cert, dict) else None
                if not isinstance(comp, list):
                    comp = []
                comp_txt = json.dumps({"A_comp": list(comp)}, separators=(",", ":"), sort_keys=False, ensure_ascii=True, allow_nan=False)
                comp_h = hashlib.sha256(comp_txt.encode("utf-8")).hexdigest()
                seq.append({"level": name, "status": "OK", "hash": comp_h, "kind": "component"})
            else:
                seq.append({"level": name, "status": "OK", "hash": fp.get("sha256"), "kind": "full"})
        except Exception as e:
            seq.append({"level": name, "status": "UNDEFINED", "reason": f"ERROR: {e}"})

    return {
        "profile_id": BARCODE_PROFILE_ID,
        "barcode_kind": kind,
        "L": int(len(seq) - 1) if seq else 0,
        "sequence": seq,
        "equality": "strict sequence equality (same length + per-level hash equality)",
    }


# -----------------------------------------------------------------------------
# IO helpers for scripts
# -----------------------------------------------------------------------------


# --- Object-level loaders (scripts often load JSON themselves) ---

def load_matrix_obj(obj: Any) -> List[List[int]]:
    """Parse a matrix from an already-loaded JSON object."""
    if isinstance(obj, list):
        return norm_bitmatrix(obj, name="A")
    if isinstance(obj, dict):
        for k in ("matrix", "A", "mat"):
            if k in obj:
                return norm_bitmatrix(obj[k], name="A")
    raise ValueError("matrix json must be a list (rows) or object with key matrix/A/mat")


def load_seed_suite_obj(obj: Any) -> List[Dict[str, Any]]:
    """Parse a seed suite from an already-loaded JSON object.

    Accepts:
      - list of seed objects
      - object with key 'seeds' or 'suite'
      - a single seed object

    Output records are normalized to:
      {seed_id, label, matrix, expected_fp}
    """
    if isinstance(obj, dict):
        if "seeds" in obj and isinstance(obj["seeds"], list):
            seeds = obj["seeds"]
        elif "suite" in obj and isinstance(obj["suite"], list):
            seeds = obj["suite"]
        else:
            seeds = [obj]
    elif isinstance(obj, list):
        seeds = obj
    else:
        raise ValueError("seed suite must be list or object with key seeds/suite")

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(seeds):
        if not isinstance(s, dict):
            raise ValueError(f"seed[{i}] must be object")
        mat = s.get("matrix") if "matrix" in s else s.get("A")
        if mat is None:
            raise ValueError(f"seed[{i}] missing matrix")
        out.append(
            {
                "seed_id": s.get("seed_id") or s.get("id") or f"seed{i}",
                "label": s.get("label") or s.get("name") or f"seed{i}",
                "matrix": norm_bitmatrix(mat, name=f"seed[{i}].matrix"),
                "expected_fp": s.get("expected_fp") or s.get("expected_fpv1_sha256"),
            }
        )
    return out


def load_map_suite_obj(obj: Any) -> List[Dict[str, Any]]:
    """Parse a map suite from an already-loaded JSON object."""
    if isinstance(obj, dict) and "maps" in obj and isinstance(obj["maps"], list):
        return obj["maps"]
    if isinstance(obj, list):
        return obj
    raise ValueError("map suite must be object with key 'maps' or a list")


def default_map_suite_entries() -> List[Dict[str, Any]]:
    """Return the default (frozen) Step‑A map suite entries.

    These entries are suitable for serialization into maps_stepA.json.
    """
    return [
        {"map_id": "append_zero", "kind": "named"},
        {"map_id": "duplicate_last", "kind": "named"},
        {"map_id": "duplicate_first", "kind": "named"},
        {
            "map_id": "permute_then_append_zero",
            "kind": "parametric",
            "ops": [
                {"op": "permute", "rule": {"rule_id": "rotate_left_1"}},
                {"op": "append_zero", "count": 1},
            ],
        },
        {
            "map_id": "append_copy_random",
            "kind": "parametric",
            "ops": [
                {"op": "duplicate", "rule": {"rule_id": "middle_floor"}},
            ],
        },
    ]


def resolve_map_entry(map_id: str, map_suite: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Resolve map_id from an optional suite; fallback to builtin named map."""
    mid = str(map_id or "")
    if not mid:
        raise ValueError("map_id required")
    by_id = resolve_map_ids(map_suite)
    if mid in by_id:
        return by_id[mid]
    # Fallback: treat as builtin map_id.
    return {"map_id": mid, "kind": "named"}

def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_matrix_json(path: str | Path) -> List[List[int]]:
    j = load_json(path)
    if isinstance(j, list):
        return norm_bitmatrix(j, name="A")
    if isinstance(j, dict):
        for k in ("matrix", "A", "mat"):
            if k in j:
                return norm_bitmatrix(j[k], name="A")
    raise ValueError("matrix json must be a list (rows) or object with key matrix/A/mat")


def load_seed_suite(path: str | Path) -> List[Dict[str, Any]]:
    j = load_json(path)
    if isinstance(j, dict):
        if "seeds" in j and isinstance(j["seeds"], list):
            seeds = j["seeds"]
        elif "suite" in j and isinstance(j["suite"], list):
            seeds = j["suite"]
        else:
            # Accept a single seed object
            seeds = [j]
    elif isinstance(j, list):
        seeds = j
    else:
        raise ValueError("seed suite must be list or object with key seeds")

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(seeds):
        if not isinstance(s, dict):
            raise ValueError(f"seed[{i}] must be object")
        if "matrix" in s:
            mat = s["matrix"]
        elif "A" in s:
            mat = s["A"]
        else:
            raise ValueError(f"seed[{i}] missing matrix")
        out.append({
            "seed_id": s.get("seed_id") or s.get("id") or f"seed{i}",
            "label": s.get("label") or s.get("name") or f"seed{i}",
            "matrix": norm_bitmatrix(mat, name=f"seed[{i}].matrix"),
            "expected_fp": s.get("expected_fp") or s.get("expected_fpv1_sha256"),
        })
    return out


def load_map_suite(path: str | Path) -> List[Dict[str, Any]]:
    j = load_json(path)
    if isinstance(j, dict) and "maps" in j and isinstance(j["maps"], list):
        return j["maps"]
    if isinstance(j, list):
        return j
    raise ValueError("map suite must be object with key 'maps' or a list")


def resolve_map_ids(map_suite: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in map_suite or []:
        if isinstance(m, dict) and m.get("map_id"):
            out[str(m["map_id"])] = m
    return out


def map_entry_to_concrete_ops(entry: Dict[str, Any], *, n_cols: int) -> StepAMap:
    """Turn a JSON map entry into a concrete StepAMap.

    Supports:
      - kind="concrete" with explicit ops
      - kind="parametric" with rules from the frozen vocabulary
      - kind="named" (or missing kind) treated as built-in map_id
    """
    map_id = str(entry.get("map_id") or entry.get("id") or "")
    if not map_id:
        raise ValueError("map entry missing map_id")

    kind = str(entry.get("kind") or "named")

    if kind in ("named", "builtin"):
        return instantiate_map(map_id, n_cols)

    if kind == "parametric":
        # Frozen parametric vocab.
        ops: List[StepAOp] = []
        for op in entry.get("ops") or []:
            if not isinstance(op, dict):
                raise ValueError(f"parametric op must be object: {op!r}")
            op_name = str(op.get("op") or "")
            if op_name == "permute":
                rule = str((op.get("rule") or {}).get("rule_id") or "")
                if rule == "rotate_left_1":
                    ops.append(StepAOp(op="permute", perm=_rotate_left_perm(n_cols)))
                else:
                    raise ValueError(f"unknown permute rule: {rule}")
            elif op_name == "duplicate":
                rule = str((op.get("rule") or {}).get("rule_id") or "")
                if rule == "middle_floor":
                    idx = int(n_cols // 2) if n_cols > 0 else 0
                    ops.append(StepAOp(op="duplicate", src="index", index=idx))
                else:
                    raise ValueError(f"unknown duplicate rule: {rule}")
            elif op_name == "append_zero":
                count = int(op.get("count") or 1)
                ops.append(StepAOp(op="append_zero", count=count))
            else:
                raise ValueError(f"unknown parametric op: {op_name}")
        return StepAMap(map_id=map_id, ops=ops)

    if kind == "concrete":
        ops: List[StepAOp] = []
        for op in entry.get("ops") or []:
            if not isinstance(op, dict):
                raise ValueError(f"concrete op must be object: {op!r}")
            op_name = str(op.get("op") or "")
            if op_name == "permute":
                perm = op.get("perm")
                if not isinstance(perm, list):
                    raise ValueError("permute op requires perm list")
                ops.append(StepAOp(op="permute", perm=[int(x) for x in perm]))
            elif op_name == "duplicate":
                src = str(op.get("src") or "last")
                index = op.get("index")
                ops.append(StepAOp(op="duplicate", src=src, index=(int(index) if index is not None else None)))
            elif op_name == "append_zero":
                ops.append(StepAOp(op="append_zero", count=int(op.get("count") or 1)))
            else:
                raise ValueError(f"unknown concrete op: {op_name}")
        return StepAMap(map_id=map_id, ops=ops)

    raise ValueError(f"unknown map entry kind: {kind}")


def canonical_json(obj: Any) -> str:
    """Canonical JSON helper used by scripts (compact, no sort_keys)."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=False, ensure_ascii=True, allow_nan=False)


