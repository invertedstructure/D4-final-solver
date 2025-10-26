import streamlit as st
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# === canonical constants / helpers (single source of truth) ===

import os
import json as _json
import hashlib as _hashlib
import uuid as _uuid
import datetime as _datetime
from pathlib import Path
import os, tempfile
import copy as _copy
import json as _json
import hashlib as _hash
import json as _json
import csv as _csv
from pathlib import Path as _Path
import datetime as _dt
import random as _random
# Page config must be the first Streamlit command
SCHEMA_VERSION = "2.0.0"
ENGINE_REV     = "rev-20251022-1"





DIRS = {"root": "logs", "certs": "logs/certs"}

NA_CODES = {
    # strict
    "C3_NOT_SQUARE": True, "BAD_SHAPE": True,
    # projected(AUTO)
    "AUTO_REQUIRES_SQUARE_C3": True, "ZERO_LANE_PROJECTOR": True,
    # freezer
    "FREEZER_C3_NOT_SQUARE": True, "FREEZER_ZERO_LANE_PROJECTOR": True,
    "FREEZER_BAD_SHAPE": True, "FREEZER_ASSERT_MISMATCH": True,
    # file preview (Advanced only)
    "BAD_PROJECTOR_SHAPE": True, "NOT_IDEMPOTENT": True
}

os.makedirs(DIRS["certs"], exist_ok=True)

def short(h: str) -> str:
    return (h or "")[:8]

def district_from_hash(boundaries_hash: str) -> str:
    return "D" + short(boundaries_hash or "")

def _canonical_json(d: dict) -> str:
    return _json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _sha256_hex(s: bytes) -> str:
    return _hashlib.sha256(s).hexdigest()

def _bits_to_str(bits) -> str:
    return "".join("1" if (int(b) & 1) else "0" for b in (bits or []))

def _witness_pack(bottom_H2d3_bits=None, bottom_C3pI3_bits=None, lanes=None) -> dict:
    return {
        "bottom_H2d3": _bits_to_str(bottom_H2d3_bits),
        "bottom_C3pI3": _bits_to_str(bottom_C3pI3_bits),
        "lanes": (list(lanes) if lanes is not None else None),
    }

def _predicate_out(eq_bool_or_none, *, na_reason_code=None, bottom_H2d3_bits=None, bottom_C3pI3_bits=None,
                   selected_cols=None, mismatch_idxs=None, residual_tag_selected=None):
    return {
        "k": {"3": {"eq": (True if eq_bool_or_none is True else (False if eq_bool_or_none is False else None)),
                    "na_reason_code": na_reason_code}},
        "witness": _witness_pack(bottom_H2d3_bits=bottom_H2d3_bits, bottom_C3pI3_bits=bottom_C3pI3_bits),
        "selected_cols": list(selected_cols or []),
        "mismatch_cols_selected": list(mismatch_idxs or []),
        "residual_tag_selected": (residual_tag_selected or ""),
    }

def build_embed(*, inputs_sig_5, dims, district_id, fixture_label, policy_tag, projection_context):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "engine_rev": ENGINE_REV,
        "inputs_sig_5": inputs_sig_5,
        "dims": dims,
        "district_id": district_id,
        "fixture_label": fixture_label or "",
        "policy": policy_tag,  # "strict__VS__projected(columns@k=3,auto|file)"
        "projection_context": projection_context,
    }
    sig = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return payload, sig

# Guarded atomic writer (never writes from read-only panels)
def _guarded_guarded_atomic_write_json(path: Path, payload: dict) -> None:
    try:
        if hasattr(st, "session_state"):
            if not st.session_state.get("_solver_one_button_active"):
                # Panel is read-only; skip writes
                return
    except Exception:
        # Headless / tests: proceed
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# === /canonical block ===
# ───────────────────────────── C1: Coverage rollup + Health ping ─────────────────────────────
# Helpers are namespaced with _c1_ to avoid collisions.


def _c1_paths():
    """Return (coverage.jsonl, rollup.csv). Creates logs/reports/ if needed."""
    base = _Path("logs") / "reports"
    base.mkdir(parents=True, exist_ok=True)
    return (base / "coverage.jsonl", base / "coverage_rollup.csv")

def _c1_iter_jsonl(p: _Path):
    """Yield JSON objects from a JSONL file, skipping bad lines."""
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield _json.loads(line)
                except Exception:
                    # ignore malformed line
                    continue
    except FileNotFoundError:
        return

def _c1_rollup_rows(jsonl_path: _Path):
    """
    Aggregate coverage.jsonl by prox_label, computing counts and mean numeric metrics.
    Numeric keys are optional; missing values are ignored.
    """
    buckets = {}
    num_keys = ("sel_mismatch_rate", "offrow_mismatch_rate", "ker_mismatch_rate", "contradiction_rate")
    for rec in _c1_iter_jsonl(jsonl_path):
        label = (rec.get("prox_label") or "UNKNOWN")
        b = buckets.setdefault(label, {"_n": 0})
        b["_n"] += 1
        for k in num_keys:
            v = rec.get(k, None)
            if isinstance(v, (int, float)):
                b[k] = b.get(k, 0.0) + float(v)

    rows = []
    for label, b in buckets.items():
        n = b.pop("_n", 0)
        def mean_for(k):
            return (b.get(k, None) / n) if (n and (k in b)) else None
        rows.append({
            "prox_label": label,
            "count": n,
            "mean_sel_mismatch_rate": mean_for("sel_mismatch_rate"),
            "mean_offrow_mismatch_rate": mean_for("offrow_mismatch_rate"),
            "mean_ker_mismatch_rate": mean_for("ker_mismatch_rate"),
            "mean_contradiction_rate": mean_for("contradiction_rate"),
        })
    rows.sort(key=lambda r: r["prox_label"])
    return rows

def _c1_write_rollup_csv(rows, out_path: _Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = [
        "prox_label", "count",
        "mean_sel_mismatch_rate", "mean_offrow_mismatch_rate",
        "mean_ker_mismatch_rate", "mean_contradiction_rate"
    ]
    # atomic-ish write
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) if r.get(k) is not None else "" for k in hdr})
    tmp.replace(out_path)

def _c1_health_ping(jsonl_path: _Path, tail: int = 50):
    """
    Compute mean mismatch rates over the last `tail` events in coverage.jsonl.
    Returns dict or None if no data.
    """
    buf = []
    for rec in _c1_iter_jsonl(jsonl_path):
        buf.append(rec)
        if len(buf) > tail:
            buf.pop(0)
    if not buf:
        return None

    def avg(key):
        vals = [float(x.get(key)) for x in buf if isinstance(x.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "tail": len(buf),
        "mean_sel_mismatch_rate": avg("sel_mismatch_rate"),
        "mean_offrow_mismatch_rate": avg("offrow_mismatch_rate"),
        "mean_ker_mismatch_rate": avg("ker_mismatch_rate"),
        "mean_contradiction_rate": avg("contradiction_rate"),
    }

def _c1_badge(hp: dict):
    """
    Tiny UI chip classifier. Thresholds are conservative and easy to tweak.
    Returns (emoji, label, color_name).
    """
    s = hp.get("mean_sel_mismatch_rate") or 0.0
    o = hp.get("mean_offrow_mismatch_rate") or 0.0
    k = hp.get("mean_ker_mismatch_rate") or 0.0
    worst = max(s, o, k)
    if worst <= 0.05:
        return "✅", "Healthy", "green"
    if worst <= 0.12:
        return "🟨", "Watch", "orange"
    return "🟥", "Alert", "red"

# ── UI: Coverage rollup + Health ping ──
with st.expander("C1 — Coverage rollup & health ping", expanded=False):
    cov_path, csv_out = _c1_paths()
    st.caption(f"Source: {cov_path} · Output: {csv_out}")

    # Health chip (tail window)
    hp = _c1_health_ping(cov_path, tail=50)
    if hp is None:
        st.info("coverage.jsonl not found yet — run the solver to produce coverage events.")
    else:
        emoji, label, _ = _c1_badge(hp)
        def fmt(x): 
            return "—" if x is None else f"{x:.3f}"
        st.markdown(
            f"**C1 Health** {emoji} {label} · tail={hp['tail']} · "
            f"sel={fmt(hp.get('mean_sel_mismatch_rate'))} · "
            f"off={fmt(hp.get('mean_offrow_mismatch_rate'))} · "
            f"ker={fmt(hp.get('mean_ker_mismatch_rate'))} · "
            f"ctr={fmt(hp.get('mean_contradiction_rate'))}"
        )

    # Rollup button
    if cov_path.exists():
        if st.button("Build rollup CSV (group by prox_label)", key="btn_c1_rollup"):
            rows = _c1_rollup_rows(cov_path)
            if not rows:
                st.warning("No rows parsed from coverage.jsonl.")
            else:
                _c1_write_rollup_csv(rows, csv_out)
                st.success(f"Wrote {len(rows)} rows → {csv_out}")
                # Show a small table without requiring pandas
                st.table([{k: (None if v is None else (round(v, 6) if isinstance(v, float) else v))
                           for k, v in r.items()} for r in rows])
                try:
                    # Optional: download
                    st.download_button(
                        "Download rollup.csv",
                        data=open(csv_out, "rb").read(),
                        file_name="coverage_rollup.csv",
                        mime="text/csv",
                        key="btn_c1_download_rollup",
                    )
                except Exception:
                    pass
# ─────────────────────────────────────────────────────────────────────────────────────────────


# ---------- Policy receipt (B) ----------
def _policy_receipt(*, mode: str, posed: bool, lane_policy: str = "", lanes: list[int] | None = None,
                    projector_source: str = "", projector_hash: str | None = None,
                    n3: int | None = None) -> dict:
    canon = (
        "strict" if mode.startswith("strict")
        else "projected(columns@k=3,auto)" if "auto" in mode
        else "projected(columns@k=3,file)"
    )
    L = [int(x) & 1 for x in (lanes or [])]
    k = sum(L)
    n3v = int(n3 if n3 is not None else (len(L) if L else 0))
    density = (k / n3v) if (n3v > 0) else 0.0
    rec = {
        "canon": canon,
        "posed": bool(posed),
        "lane_policy": lane_policy or ("file projector" if "file" in canon else ("C bottom row" if "auto" in canon else "")),
        "lanes": L,
        "lane_sum": k,
        "lane_density": density,
    }
    if "file" in canon:
        pj = {}
        if projector_hash:
            pj["source"] = "file"
            pj["hash"]   = str(projector_hash)
            pj["shape"]  = [n3v, n3v] if n3v else [0, 0]
            pj["idempotent"] = True  # diagonal boolean Π → idempotent over 𝔽₂
        rec["projector"] = pj
    return rec

# ---------- FREEZER mismatch JSONL (C) ----------
def _log_freezer_mismatch(*, fixture_id: str, auto_lanes: list[int], file_lanes: list[int],
                          verdict_auto: bool | None, verdict_file: bool | None):
    """Append to reports/freezer_mismatch_log.jsonl when AUTO↔FILE disagree."""
    row = {
        "written_at_utc": _svr_now_iso(),
        "fixture_id": str(fixture_id),
        "auto_lanes": [int(x)&1 for x in (auto_lanes or [])],
        "file_lanes": [int(x)&1 for x in (file_lanes or [])],
        "verdict_auto": (None if verdict_auto is None else bool(verdict_auto)),
        "verdict_file": (None if verdict_file is None else bool(verdict_file)),
        "code": "FREEZER_ASSERT_MISMATCH",
    }
    try:
        _atomic_append_jsonl(Path("reports") / "freezer_mismatch_log.jsonl", row)
    except Exception:
        pass


# --- Canonical tiny helpers (early, guarded) ---
from typing import Iterable, List, Optional

if "_normalize_bit" not in globals():
    def _normalize_bit(v) -> int:
        try: return 1 if (int(v) & 1) else 0
        except Exception: return 0

if "_svr_mismatch_cols" not in globals():
    def _svr_mismatch_cols(residual_bottom_row_bits: Iterable[int],
                           selected_mask_bits: Iterable[int]) -> List[int]:
        idxs: List[int] = []
        try:
            for j, (r, s) in enumerate(zip(residual_bottom_row_bits, selected_mask_bits)):
                if _normalize_bit(s) and _normalize_bit(r):
                    idxs.append(j)
        except Exception:
            idxs = []
        return idxs

if "_svr_residual_bits" not in globals():
    def _svr_residual_bits(residual_bottom_row_bits: Iterable[int],
                           selected_mask_bits: Iterable[int] | None) -> str:
        try:
            if selected_mask_bits is None:
                row = list(residual_bottom_row_bits or [])
                return "".join("1" if _normalize_bit(x) else "0" for x in row)
            selected = [ _normalize_bit(r)
                         for r, s in zip(residual_bottom_row_bits or [], selected_mask_bits or [])
                         if _normalize_bit(s) ]
            return "".join("1" if _normalize_bit(x) else "0" for x in selected)
        except Exception:
            return ""

if "_selected_mask_strict" not in globals():
    def _selected_mask_strict(n3: int) -> List[int]:
        try: return [1] * int(n3)
        except Exception: return []

if "_selected_mask_auto" not in globals():
    def _selected_mask_auto(lanes: Iterable[int], n3: int) -> List[int]:
        try:
            clean = [ _normalize_bit(x) for x in (lanes or []) ]
            n3 = int(n3)
            if len(clean) != n3:
                clean = (clean + [0]*n3)[:n3]
            return clean
        except Exception:
            return []
# --- /tiny helpers ---
# NEW: full-matrix residual helpers (any 1 in a column across all rows)
if "_svr_mismatch_cols_from_R3" not in globals():
    def _svr_mismatch_cols_from_R3(R3, selected_mask_bits):
        try:
            n3 = len(R3[0]) if (R3 and R3[0]) else 0
            sel = [(int(x) & 1) for x in (selected_mask_bits or [])]
            if len(sel) != n3:
                sel = (sel + [0] * n3)[:n3]
            out = []
            for j in range(n3):
                if sel[j] and any((int(R3[i][j]) & 1) for i in range(len(R3))):
                    out.append(j)
            return out
        except Exception:
            return []

if "_svr_residual_tag_from_R3" not in globals():
    def _svr_residual_tag_from_R3(R3, selected_mask_bits):
        try:
            n3 = len(R3[0]) if (R3 and R3[0]) else 0
            sel = [(int(x) & 1) for x in (selected_mask_bits or [])]
            if len(sel) != n3:
                sel = (sel + [0] * n3)[:n3]
            bits = []
            for j in range(n3):
                one = (sel[j] and any((int(R3[i][j]) & 1) for i in range(len(R3))))
                bits.append("1" if one else "0")
            return "".join(bits)
        except Exception:
            return ""




# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
import json
import csv
import hashlib
import platform
import zipfile
import tempfile
import shutil
import importlib.util
import types
import secrets
import math
import uuid
from io import BytesIO
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import random
# ------------------------- End of Organized Helpers -------------------------
import uuid
import os
import shutil
import tempfile
import json
from pathlib import Path
from contextlib import contextmanager
# ======================= Canon Helpers SSOT - Deduped & Organized =======================

import json, hashlib, streamlit as st
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# Underscored aliases for helpers
import os as _os
import json as _json
import hashlib as _hashlib
import csv as _csv
import zipfile as _zipfile
import tempfile as _tempfile
import shutil as _shutil
from pathlib import Path as _Path
from uuid import uuid4

# Page config early so Streamlit is happy
# ────────────────────────────── PACKAGE LOADER ──────────────────────────────
HERE = Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
CORE = HERE / "core"
PKG_DIR = OTCORE if OTCORE.exists() else CORE
PKG_NAME = "otcore" if OTCORE.exists() else "core"

# Ensure pkg namespace exists
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PKG_DIR)]
    pkg.__file__ = str(PKG_DIR / "__init__.py")
    sys.modules[PKG_NAME] = pkg

def _load_pkg_module(fullname: str, rel_path: str):
    path = PKG_DIR / rel_path
    if not path.exists():
        raise ImportError(f"Required module file not found: {path}")
    spec = importlib.util.spec_from_file_location(fullname, str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = fullname.rsplit(".", 1)[0]
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Hot-reload core modules if Streamlit reruns
for _mod in (
    f"{PKG_NAME}.overlap_gate",
    f"{PKG_NAME}.projector",
    f"{PKG_NAME}.io",
    f"{PKG_NAME}.hashes",
    f"{PKG_NAME}.unit_gate",
    f"{PKG_NAME}.triangle_gate",
    f"{PKG_NAME}.towers",
    f"{PKG_NAME}.export",
):
    if _mod in sys.modules:
        del sys.modules[_mod]

# Load modules
overlap_gate = _load_pkg_module(f"{PKG_NAME}.overlap_gate", "overlap_gate.py")
projector = _load_pkg_module(f"{PKG_NAME}.projector", "projector.py")
otio = _load_pkg_module(f"{PKG_NAME}.io", "io.py")
hashes = _load_pkg_module(f"{PKG_NAME}.hashes", "hashes.py")
unit_gate = _load_pkg_module(f"{PKG_NAME}.unit_gate", "unit_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers = _load_pkg_module(f"{PKG_NAME}.towers", "towers.py")
export_mod = _load_pkg_module(f"{PKG_NAME}.export", "export.py")

# Legacy alias
io = otio

# App version string
APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")

# ───────────────────────────── RUN HEADER (read-only) ─────────────────────────
try:
    _hdr = st.session_state.get("last_run_header")
    if _hdr:
        st.info(_hdr)
except Exception:
    pass


# ───────────────────── DISTRICT MAP (optional) ─────────────────────
DISTRICT_MAP: dict[str, str] = {
    "9da8b7f605c113ee059160cdaf9f93fe77e181476c72e37eadb502e7e7ef9701": "D1",
    "4356e6b608443b315d7abc50872ed97a9e2c837ac8b85879394495e64ec71521": "D2",
    "28f8db2a822cb765e841a35c2850a745c667f4228e782d0cfdbcb710fd4fecb9": "D3",
    "aea6404ae680465c539dc4ba16e97fbd5cf95bae5ad1c067dc0f5d38ca1437b5": "D4",
}

# ---- Optional gallery display helper (Streamlit fallback)
try:
    import caas_jupyter_tools as _cj  # not available on Streamlit runner
    def _display_df(name, df):
        _cj.display_dataframe_to_user(name, df)
except Exception:
    def _display_df(name, df):
        import pandas as _pd
        import streamlit as _st
        # Accept list-of-dicts or DataFrame
        if not isinstance(df, _pd.DataFrame):
            df = _pd.DataFrame(df)
        _st.dataframe(df, use_container_width=True)

def _lane_mask_from_d3(*args):
    """
    Flexible helper (compat shim).

    Accepts any of:
      1) _lane_mask_from_d3(C3)
      2) _lane_mask_from_d3(d3, C3)
      3) _lane_mask_from_d3(blocks_dict)         # where blocks_dict.get("3") is C3
      4) _lane_mask_from_d3(d3_blocks, c3_blocks) # dicts with .get("3")

    Returns:
      list[int] of length n3 with bits from bottom row of C3 when square.
      If C3 is missing/non-square, returns [0]*n3 inferred from d3, or [] if n3 unknown.
    """
    C3 = None
    d3 = None

    # Unpack positional forms
    if len(args) == 1:
        x = args[0]
        if isinstance(x, dict):          # blocks dict
            C3 = x.get("3") or None
        else:                             # assume it's already C3 matrix
            C3 = x
    elif len(args) >= 2:
        a, b = args[0], args[1]
        # Accept dicts or raw matrices in either position
        d3 = (a.get("3") if isinstance(a, dict) else a)
        C3 = (b.get("3") if isinstance(b, dict) else b)

    # If we have a square C3, use its bottom row
    try:
        if C3 and isinstance(C3, list) and C3 and isinstance(C3[0], list):
            n_rows = len(C3)
            n_cols = len(C3[0]) if C3[0] is not None else 0
            if n_rows == n_cols and n_cols > 0:
                bottom = C3[-1] or []
                return [int(x) & 1 for x in bottom[:n_cols]]
    except Exception:
        pass

    # Fallback: infer n3 from d3 (columns) → zero mask (AUTO → N/A later)
    try:
        if d3 and isinstance(d3, list) and d3 and isinstance(d3[0], list):
            n3 = len(d3[0]) if d3[0] is not None else 0
            return [0] * n3
    except Exception:
        pass

    # Unknown shape
    return []
if "_lane_mask_from_d3" not in globals():
    def _lane_mask_from_d3(*args):
        # Prefer C3 if present
        C3 = None
        if len(args) == 1 and isinstance(args[0], dict):
            C3 = args[0].get("3")
        elif len(args) == 1:
            C3 = args[0]
        elif len(args) >= 2:
            C3 = (args[1].get("3") if isinstance(args[1], dict) else args[1])
        if C3 and isinstance(C3, list) and C3 and len(C3) == len(C3[0]):
            return [int(x) & 1 for x in (C3[-1] or [])]
        # fallback: zero mask from d3 width
        d3 = (args[0].get("3") if len(args) >= 1 and isinstance(args[0], dict) else (args[0] if len(args)>=1 else None))
        n3 = (len(d3[0]) if d3 and d3 and isinstance(d3[0], list) else 0)
        return [0]*n3 if n3 else []

def _svr_header_line(ib, sig8, projector_hash=None, policy_tag="strict", run_id=""):
    """
    Render a compact, one-line header for the last solver press.

    Fields shown:
      {policy_tag} | n3={n3} | B={b8} C={c8} H={h8} U={u8} S={sig8} | P={p8 or —} | D={district} | run={run8}

    Inputs:
      ib: SSOT inputs block (expects ib["hashes"], ib["dims"], optional ib["district_id"])
      sig8: first 8 hex chars of the A/B embed_sig (string, may be empty)
      projector_hash: either None, "" or "sha256:<hex>" (we show first 8 hex of the hash if present)
      policy_tag: display tag ("strict", "strict__VS__projected(columns@k=3,auto)", etc.)
      run_id: UUID string (we show first 8 non-dash chars)

    Returns:
      A single formatted string.
    """
    def _short(x, n=8):
        if not x:
            return ""
        s = str(x)
        if s.startswith("sha256:"):
            s = s.split("sha256:", 1)[1]
        s = s.replace("-", "")
        return s[:n]

    h = (ib or {}).get("hashes") or {}
    dims = (ib or {}).get("dims") or {}
    n3 = int(dims.get("n3") or 0)

    # Try a few common key variants to be forgiving
    b8 = _short(h.get("hash_d") or h.get("d_hash") or h.get("boundaries_hash") or h.get("B_hash"))
    c8 = _short(h.get("hash_suppC") or h.get("suppC_hash") or h.get("C_hash"))
    h8 = _short(h.get("hash_suppH") or h.get("suppH_hash") or h.get("H_hash"))
    u8 = _short(h.get("hash_U") or h.get("U_hash"))

    s8 = (sig8 or "")[:8]
    p8 = _short(projector_hash) if projector_hash else "—"
    run8 = _short(run_id)

    district = (ib or {}).get("district_id")
    if not district:
        # Fallback to D + boundaries hash if present
        bh = h.get("boundaries_hash") or ""
        district = ("D" + _short(bh)) if bh else "DUNKNOWN"

    return (
        f"{policy_tag} | n3={n3} | "
        f"B={b8 or '—'} C={c8 or '—'} H={h8 or '—'} U={u8 or '—'} S={s8 or '—'} | "
        f"P={p8} | D={district} | run={run8 or '—'}"
    )

# ---- District signature helper (robust, deterministic) ----
def _district_signature(*args, prefix: str = "D", size: int = 8, return_hash: bool = False):
    """
    Usage patterns (any of these works):
      _district_signature(ib)                          # where ib["hashes"]["boundaries_hash"] exists
      _district_signature(hashes_dict)                 # direct hashes dict with 'boundaries_hash'
      _district_signature(B_blocks_dict)               # e.g. pb["B"][1] where keys like "3" map to matrices
      _district_signature(d3_matrix)                   # raw d3 (n2 x n3) → hash over {"B":{"blocks":{"3":d3}}}

    Returns:
      - default:  'D' + first 8 hex of the boundaries hash (configurable via prefix/size)
      - if return_hash=True: (district_id, full_boundaries_hash)
    """
    # Local safe imports/fallbacks
    try:
        _json  # type: ignore
    except NameError:
        import json as _json  # noqa: F401
    try:
        _hashlib  # type: ignore
    except NameError:
        import hashlib as _hashlib  # noqa: F401

    # 1) Try to read a precomputed boundaries_hash
    boundaries_hash = None
    for x in args:
        if isinstance(x, dict):
            # ib-style: {"hashes": {...}}
            hashes = x.get("hashes") if "hashes" in x and isinstance(x.get("hashes"), dict) else x
            if isinstance(hashes, dict):
                boundaries_hash = (
                    hashes.get("boundaries_hash")
                    or hashes.get("B_hash")
                    or hashes.get("boundaries_sha256")
                )
                if boundaries_hash:
                    break

    # 2) If missing, compute from B blocks (or raw d3)
    if not boundaries_hash:
        B_blocks = None
        # Prefer a blocks dict like {"3": d3, ...}
        for x in args:
            if isinstance(x, dict) and any(k in x for k in ("3", 3, "2", 2)):
                B_blocks = x
                break
        # Or accept a raw d3 matrix
        if B_blocks is None and len(args) >= 1 and isinstance(args[0], list):
            B_blocks = {"3": args[0]}

        if B_blocks is not None:
            # Canonical JSON for hashing (sorted keys, tight separators)
            blocks_norm = {}
            for k in sorted(B_blocks.keys(), key=lambda z: str(z)):
                blocks_norm[str(k)] = B_blocks[k] or []
            blob = {"B": {"blocks": blocks_norm}}
            try:
                # Prefer your app's canonical hasher if present
                boundaries_hash = _svr_hash(blob)  # type: ignore
            except Exception:
                s = _json.dumps(blob, separators=(",", ":"), sort_keys=True).encode("ascii")
                boundaries_hash = _hashlib.sha256(s).hexdigest()

    # 3) Finalize
    if not boundaries_hash:
        sig = f"{prefix}UNKNOWN"
        return (sig, "") if return_hash else sig

    sig = f"{prefix}{str(boundaries_hash)[:int(size)]}"
    return (sig, str(boundaries_hash)) if return_hash else sig


# Optional convenience alias if you reference "district_from_hash" elsewhere
if "district_from_hash" not in globals():
    def district_from_hash(boundaries_hash: str, prefix: str = "D", size: int = 8) -> str:
        return f"{prefix}{str(boundaries_hash)[:int(size)]}"

# ---------- Frozen inputs signature (SSOT 5-hash) ----------
def _frozen_inputs_sig_from_ib(ib, as_tuple: bool = True):
    """
    Returns the 5-hash SSOT signature from the frozen inputs block `ib`,
    in this canonical order:
      (hash_d, hash_U, hash_suppC, hash_suppH, hash_shapes)

    - Accepts common alias keys for backwards compatibility.
    - Missing fields are returned as "" (empty string), preserving length.
    - Set as_tuple=False to get a list instead of a tuple.
    """
    h = (ib or {}).get("hashes") or {}

    def _pick(*keys):
        for k in keys:
            if k in h and h[k]:
                return str(h[k])
        return ""

    hash_d       = _pick("hash_d", "d_hash", "boundaries_hash", "B_hash")
    hash_U       = _pick("hash_U", "U_hash")
    hash_suppC   = _pick("hash_suppC", "suppC_hash", "C_hash")
    hash_suppH   = _pick("hash_suppH", "suppH_hash", "H_hash")
    hash_shapes  = _pick("hash_shapes", "shapes_hash")

    sig5 = (hash_d, hash_U, hash_suppC, hash_suppH, hash_shapes)
    return sig5 if as_tuple else list(sig5)

# Optional: single hex digest over the 5-hash tuple (stable, canonical JSON)
def _frozen_inputs_sig5_hex(ib):
    try:
        _json  # type: ignore
    except NameError:
        import json as _json  # noqa: F401
    try:
        _hashlib  # type: ignore
    except NameError:
        import hashlib as _hashlib  # noqa: F401

    sig5 = _frozen_inputs_sig_from_ib(ib, as_tuple=False)
    blob = _json.dumps(sig5, separators=(",", ":"), sort_keys=False).encode("ascii")
    return _hashlib.sha256(blob).hexdigest()

# --- pin freshness helper ------------------------------------------------------
def _pin_status_text(pin_obj, expected_sig: str) -> str:
    """Return a short freshness badge given a pin object and the expected embed_sig."""
    payload = (pin_obj or {}).get("payload") or {}
    have = str(payload.get("embed_sig",""))
    if not expected_sig:
        return "—"
    if not have:
        return "—"
    return "🟢 Fresh" if have == expected_sig else "⚠️ Stale"
# --- strict / projected(columns@k=3,auto) helpers (guarded) --------------------------------
if "_svr_shape_ok_for_mul" not in globals():
    def _svr_shape_ok_for_mul(A, B):
        return bool(A and B and A[0] and B[0] and (len(A[0]) == len(B)))

if "_svr_eye" not in globals():
    def _svr_eye(n: int):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

if "_svr_is_zero" not in globals():
    def _svr_is_zero(M):
        return (not M) or all((int(x) & 1) == 0 for row in M for x in row)

if "_svr_mul" not in globals():
    def _svr_mul(A, B):
        # GF(2) multiply with shape guard
        if not _svr_shape_ok_for_mul(A, B):
            return []
        m, k, n = len(A), len(A[0]), len(B[0])
        C = [[0]*n for _ in range(m)]
        for i in range(m):
            Ai = A[i]
            for t in range(k):
                if int(Ai[t]) & 1:
                    Bt = B[t]
                    for j in range(n):
                        C[i][j] ^= (int(Bt[j]) & 1)
        return C

if "_svr_xor" not in globals():
    def _svr_xor(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(int(A[i][j]) ^ int(B[i][j])) & 1 for j in range(c)] for i in range(r)]

if "_svr_strict_from_blocks" not in globals():
    pass

    def _svr_strict_from_blocks(bH: dict, bB: dict, bC: dict) -> dict:
        """
        Strict k=3: R3 = H2 @ d3 ⊕ (C3 ⊕ I3); pass iff R3 == 0.
        Returns {"2":{"eq": True|None}, "3":{"eq": True|False|None}, "na_reason_code": <opt>}
        N/A (None) when C3 not square or shapes don’t pose H2@d3.
        """
        H2 = (bH.get("2") or [])
        d3 = (bB.get("3") or [])
        C3 = (bC.get("3") or [])
        # guard A: C3 must be square
        if not (C3 and C3[0] and len(C3) == len(C3[0])):
            return {"2": {"eq": None}, "3": {"eq": None}, "na_reason_code": "C3_NOT_SQUARE"}
        # guard B: H2@d3 shapes must conform
        if not _svr_shape_ok_for_mul(H2, d3):
            return {"2": {"eq": None}, "3": {"eq": None}, "na_reason_code": "BAD_SHAPE"}
        I3  = _svr_eye(len(C3))
        R3s = _svr_xor(_svr_mul(H2, d3), _svr_xor(C3, I3))
        eq3 = _svr_is_zero(R3s)
        return {"2": {"eq": True}, "3": {"eq": bool(eq3)}}
if "_svr_projected_auto_from_blocks" not in globals():
    def _svr_projected_auto_from_blocks(bH: dict, bB: dict, bC: dict):
        """
        Projected(auto) k=3:
          lanes = bottom row of C3 (requires C3 square and non-zero mask)
          P = diag(lanes), test R3 @ P == 0.
        Returns (meta, lanes, out) where:
          meta = {"na": True, "reason": ...} on N/A; else {"na": False, "policy": "auto_c_bottom"}
          lanes = list[int] length n3 (or [] on N/A)
          out   = {"2":{"eq": ...}, "3":{"eq": ...}} (None on N/A)
        """
        H2 = (bH.get("2") or [])
        d3 = (bB.get("3") or [])
        C3 = (bC.get("3") or [])
        # guard A: C3 square
        if not (C3 and C3[0] and len(C3) == len(C3[0])):
            return {"na": True, "reason": "AUTO_REQUIRES_SQUARE_C3"}, [], {"2": {"eq": None}, "3": {"eq": None}}
        n3 = len(C3)
        lanes = [1 if int(x) == 1 else 0 for x in (C3[-1] if C3 else [])]
        # guard B: non-zero mask
        if sum(lanes) == 0:
            return {"na": True, "reason": "ZERO_LANE_PROJECTOR"}, lanes, {"2": {"eq": None}, "3": {"eq": None}}
        # guard C: shapes OK for H2@d3
        if not _svr_shape_ok_for_mul(H2, d3):
            return {"na": True, "reason": "BAD_SHAPE"}, lanes, {"2": {"eq": None}, "3": {"eq": None}}
        I3  = _svr_eye(n3)
        R3s = _svr_xor(_svr_mul(H2, d3), _svr_xor(C3, I3))
        # P = diag(lanes)
        P   = [[1 if (i == j and lanes[j] == 1) else 0 for j in range(n3)] for i in range(n3)]
        R3p = _svr_mul(R3s, P)
        eq3 = _svr_is_zero(R3p)
        return {"na": False, "policy": "auto_c_bottom"}, lanes, {"2": {"eq": True}, "3": {"eq": bool(eq3)}}


# --- inputs signature helpers (guarded) ----------------------------------------
if "_svr_inputs_sig" not in globals():
    def _svr_inputs_sig(ib: dict) -> list[str]:
        """
        Legacy 5-tuple used by existing cert writers.
        Order: boundaries, C, H, U, shapes. Missing entries become "".
        """
        h = (ib.get("hashes") or {})
        return [
            str(h.get("boundaries_hash", "")),
            str(h.get("C_hash", "")),
            str(h.get("H_hash", "")),
            str(h.get("U_hash", "")),
            str(h.get("shapes_hash", "")),
        ]

if "_svr_inputs_sig_map" not in globals():
    def _svr_inputs_sig_map(ib: dict) -> dict:
        """
        Map form for the unified embed signature. If you don’t yet persist
        separate support-C/H hashes, keep them as "" (stable schema).
        """
        h = (ib.get("hashes") or {})
        return {
            "hash_d":       str(h.get("boundaries_hash", "")),
            "hash_U":       str(h.get("U_hash", "")),
            "hash_suppC":   str(h.get("suppC_hash", "")),   # OK if missing
            "hash_suppH":   str(h.get("suppH_hash", "")),   # OK if missing
            "hash_shapes":  str(h.get("shapes_hash", "")),
        }

# --- Unified A/B embed signature (lane-aware, cert-aligned) -------------------
import json as _json
from pathlib import Path

def _inputs_sig_from_frozen_ib() -> list[str]:
    ib = st.session_state.get("_inputs_block") or {}
    h  = (ib.get("hashes") or {})
    return [
        str(h.get("boundaries_hash","")),
        str(h.get("C_hash","")),
        str(h.get("H_hash","")),
        str(h.get("U_hash","")),
        str(h.get("shapes_hash","")),
    ]

def _lanes_from_frozen_C3_bottom() -> list[int]:
    """Read C3 from the *frozen* SSOT file and return its bottom row as a lane mask (AUTO policy).
       Returns [] if not square/unavailable."""
    try:
        ib = st.session_state.get("_inputs_block") or {}
        pC = (ib.get("filenames") or {}).get("C", "")
        if not pC:
            return []
        j = _json.loads(Path(pC).read_text(encoding="utf-8"))
        C3 = (j.get("blocks") or {}).get("3") or []
        if C3 and len(C3) == len(C3[0]):
            bottom = C3[-1]
            return [1 if int(x) == 1 else 0 for x in bottom]
    except Exception:
        pass
    return []

def _embed_sig_unified() -> str:
    """Single source of truth for A/B freshness: same hash the solver uses in certs."""
    ib_sig = _inputs_sig_from_frozen_ib()
    rc     = st.session_state.get("run_ctx") or {}
    pol    = str(rc.get("policy_tag") or rc.get("mode") or "projected(columns@k=3,auto)")

    # Prefer FILE if explicitly selected
    if pol.endswith("(file)"):
        pj_hash = rc.get("projector_hash", "")
        if "_svr_embed_sig" in globals():
            return _svr_embed_sig(ib_sig, "projected(columns@k=3,file)", pj_hash)
        else:
            # minimal fallback if _svr_embed_sig is not present
            blob = {"inputs": ib_sig, "policy": "projected(columns@k=3,file)", "projector_hash": pj_hash}
            return _hashlib.sha256(_json.dumps(blob, separators=(",", ":"), sort_keys=True).encode("ascii")).hexdigest()

    # AUTO policy
    lanes = list(rc.get("lane_mask_k3") or []) or _lanes_from_frozen_C3_bottom()
    if "_svr_embed_sig" in globals():
        return _svr_embed_sig(ib_sig, "projected(columns@k=3,auto)", (lanes if lanes else "ZERO_LANE_PROJECTOR"))
    else:
        blob = {"inputs": ib_sig, "policy": "projected(columns@k=3,auto)", "lanes": lanes}
        return _hashlib.sha256(_json.dumps(blob, separators=(",", ":"), sort_keys=True).encode("ascii")).hexdigest()

# Back-compat: make the old freshness checker use the unified signature
_abx_embed_sig = _embed_sig_unified
# -------------------------------------------------------------------------------



def warn_stale_once(msg="STALE_RUN_CTX: Inputs changed; please click Run Overlap to refresh."):
    ss = st.session_state
    if not ss.get("_stale_warned_once"):
        st.warning(msg)
        ss["_stale_warned_once"] = True

    
def projector_hash_of(P_blocks: list[list[int]], *, mode: str = "blocks") -> str:
    """
    mode="blocks" → sha256(json.dumps({"blocks":{"3":P}}, sort_keys=True, separators=(",",":")))
    mode="file"   → sha256(file bytes)  # only when you have a filename
    """
    import json, hashlib, pathlib
    if mode == "blocks":
        blob = json.dumps({"blocks":{"3": P_blocks}}, sort_keys=True, separators=(",",":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    elif mode.startswith("file:"):
        path = mode.split(":",1)[1]
        try:    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        except: return ""
    return ""


# ---- helper for recomputing diag lanes if the snapshot lacks them
def _ab_lane_vectors_bottom(H2, d3, C3, lm):
    """Lane vectors as bottom-row probes (matches your earlier UI semantics)."""
    try:
        H2d3  = mul(H2, d3) if _ab_shape_ok(H2, d3) else []
        C3pI3 = _ab_xor(C3, _ab_eye(len(C3))) if (C3 and C3[0]) else []
    except Exception:
        H2d3, C3pI3 = [], []
    def _bottom(M): 
        try: return M[-1] if (M and len(M)) else []
        except Exception: return []
    bH, bC = _bottom(H2d3), _bottom(C3pI3)
    idx = [j for j,m in enumerate(lm or []) if m]
    vH = [int(bH[j]) & 1 for j in idx] if (bH and idx and max(idx) < len(bH)) else []
    vC = [int(bC[j]) & 1 for j in idx] if (bC and idx and max(idx) < len(bC)) else []
    return vH, vC

    return hv, cv

# ---------------- Fixture helpers (single source of truth) ----------------

def match_fixture_from_snapshot(snap: dict) -> dict:
    reg = _get_fixtures_cached() or {}
    ordering = list(reg.get("ordering") or [])
    fixtures = list(reg.get("fixtures") or [])

    # Build ordered list: declared ordering first, then the rest
    code_to_fixture = {fx.get("code"): fx for fx in fixtures}
    ordered = [code_to_fixture[c] for c in ordering if c in code_to_fixture] + \
              [fx for fx in fixtures if fx.get("code") not in ordering]

    # Extract normalized fields we match on
    district = str(((snap.get("identity") or {}).get("district_id") or "UNKNOWN"))
    policy   = str(((snap.get("policy")   or {}).get("canon")       or "strict")).lower()
    lanes    = [int(x) for x in ((snap.get("inputs") or {}).get("lane_mask_k3") or [])]
    Hb       = [int(x) for x in ((snap.get("diagnostics") or {}).get("lane_vec_H2@d3") or [])]
    Cb       = [int(x) for x in ((snap.get("diagnostics") or {}).get("lane_vec_C3+I3") or [])]
    strict_eq3 = bool((((snap.get("checks") or {}).get("k") or {}).get("3") or {}).get("eq", False))

    def _veq(a, b):
        a = [int(x) for x in (a or [])]; b = [int(x) for x in (b or [])]
        return len(a) == len(b) and a == b

    for fx in ordered:
        m = fx.get("match") or {}
        if m.get("district") and str(m["district"]) != district:
            continue
        pol_any = [str(x).lower() for x in (m.get("policy_canon_any") or [])]
        if pol_any and (policy not in pol_any):
            continue
        if "lanes" in m and not _veq(m["lanes"], lanes):
            continue
        if "H_bottom" in m and not _veq(m["H_bottom"], Hb):
            continue
        if "C3_plus_I3_bottom" in m and not _veq(m["C3_plus_I3_bottom"], Cb):
            continue
        if "strict_eq3" in m and bool(m["strict_eq3"]) != strict_eq3:
            continue

        return {
            "fixture_code": fx.get("code",""),
            "fixture_label": fx.get("label",""),
            "tag": fx.get("tag",""),
            "strictify": fx.get("strictify","tbd"),
            "growth_bumps": int(fx.get("growth_bumps", 0)),
        }

    # Fallback (never block)
    return {
        "fixture_code":  "",
        "fixture_label": f"{district} • lanes={lanes} • H={Hb} • C+I={Cb}",
        "tag":           "novelty",
        "strictify":     "tbd",
        "growth_bumps":  int(((snap.get("growth") or {}).get("growth_bumps") or 0)),
    }


def apply_fixture_to_session(fx: dict) -> None:
    ss = st.session_state
    ss["fixture_label"]     = fx.get("fixture_label","")
    ss["gallery_tag"]       = fx.get("tag","")
    ss["gallery_strictify"] = fx.get("strictify","tbd")
    ss["growth_bumps"]      = int(fx.get("growth_bumps", 0))

    rc = dict(ss.get("run_ctx") or {})
    rc["fixture_label"] = ss["fixture_label"]
    rc["fixture_code"]  = fx.get("fixture_code","")
    ss["run_ctx"] = rc

# --- baseline imports (defensive) ---
import os, json, time, uuid, shutil, tempfile, hashlib
from datetime import datetime, timezone
from pathlib import Path

# --- legacy aliases to avoid NameError from older code paths ---
_json = json                               # some helpers still used _json.*
_sha256_hex_bytes = lambda b: hashlib.sha256(b).hexdigest()
_sha256_hex = _sha256_hex_bytes            # older helpers referenced this name
_mul_gf2 = globals().get("mul")            # older helpers referenced _mul_gf2
_add_gf2 = globals().get("add")            # older helpers referenced _add_gf2
if _mul_gf2 is None:
    def _mul_gf2(A,B):
        if not A or not B or not A[0] or not B[0]: return []
        m, kA = len(A), len(A[0])
        kB, n = len(B), len(B[0])
        if kA != kB: return []
        C = [[0]*n for _ in range(m)]
        for i in range(m):
            Ai = A[i]
            for k in range(kA):
                if Ai[k] & 1:
                    Bk = B[k]
                    for j in range(n):
                        C[i][j] ^= (Bk[j] & 1)
        return C
if _add_gf2 is None:
    def _add_gf2(A,B):
        if not A: return B or []
        if not B: return A or []
        r,c = len(A), len(A[0])
        if len(B)!=r or len(B[0])!=c: return A
        return [[(A[i][j]^B[i][j]) for j in range(c)] for i in range(r)]

# --- reports/paths (canonical + compat) ---
if "REPORTS_DIR" not in globals():
    REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if "PARITY_REPORT_PATH" not in globals():
    PARITY_REPORT_PATH = REPORTS_DIR / "parity_report.json"
if "PARITY_SUMMARY_CSV" not in globals():
    PARITY_SUMMARY_CSV = REPORTS_DIR / "parity_summary.csv"

# back-compat aliases some code might still reference
PARITY_JSON_PATH = globals().get("PARITY_JSON_PATH", PARITY_REPORT_PATH)
PARITY_CSV_PATH  = globals().get("PARITY_CSV_PATH",  PARITY_SUMMARY_CSV)

LOGS_DIR = Path(globals().get("LOGS_DIR", "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)





# ------------------------- Hashing Helpers -------------------------
def _deep_intify(o):
    if isinstance(o, bool): return 1 if o else 0
    if isinstance(o, list): return [_deep_intify(x) for x in o]
    if isinstance(o, dict): return {k: _deep_intify(v) for k, v in o.items()}
    return o

def _sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _sha256_hex_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hash_json(obj) -> str:
    """Stable SHA-256 over a JSON-serializable object (bools → 0/1, sorted keys, tight separators)."""
    s = json.dumps(_deep_intify(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_hex_text(s)

# ------------------------- Fixtures Registry (cache) -------------------------
def _fixtures_bytes_and_hash(path: str):
    """Helper to read file bytes and compute hash."""
    try:
        b = Path(path).read_bytes()
        h = _sha256_hex_bytes(b)
        return b, h, path
    except Exception:
        return b"", "", path

def load_fixtures_registry() -> dict | None:
    """Load and cache the fixtures registry with cache invalidation."""
    fx_path = Path("configs") / "fixtures.json"
    try:
        fx_bytes = fx_path.read_bytes()
        fx_hash = _sha256_hex_bytes(fx_bytes)
    except Exception:
        st.session_state.pop("_fixtures_cache", None)
        st.session_state.pop("_fixtures_bytes_hash", None)
        return None

    if st.session_state.get("_fixtures_bytes_hash") != fx_hash:
        try:
            data = json.loads(fx_bytes.decode("utf-8"))
            cache = {
                "version": str(data.get("version", "")),
                "ordering": list(data.get("ordering") or []),
                "fixtures": list(data.get("fixtures") or []),
                "__hash": fx_hash,
                "__path": fx_path.as_posix(),
            }
            st.session_state["_fixtures_cache"] = cache
            st.session_state["_fixtures_bytes_hash"] = fx_hash
        except Exception:
            st.session_state.pop("_fixtures_cache", None)
            st.session_state["_fixtures_bytes_hash"] = fx_hash
            return None
    return st.session_state.get("_fixtures_cache")

def fixtures_load_cached(path: str = "configs/fixtures.json") -> dict:
    """Load fixtures cache with tolerant signature. Rehydrates cache if bytes hash changed."""
    b, h, p = _fixtures_bytes_and_hash(path)
    cache = st.session_state.get("_fixtures_cache")
    if not cache or st.session_state.get("_fixtures_bytes_hash") != h:
        try:
            data = json.loads(b.decode("utf-8")) if b else {}
            cache = {
                "version": str(data.get("version", "")),
                "ordering": list(data.get("ordering") or []),
                "fixtures": list(data.get("fixtures") or []),
                "__path": p,
            }
        except Exception:
            cache = {"version": "", "ordering": [], "fixtures": [], "__path": p}
        st.session_state["_fixtures_cache"] = cache
        st.session_state["_fixtures_bytes_hash"] = h
    return cache

# ------------------------- SSOT: Stable hashes for block-like objects -------------------------
def ssot_stable_blocks_sha(obj) -> str:
    """
    Compute stable sha256 of blocks-like objects.
    Accepts an object with .blocks.__root__ OR a plain dict {"blocks": ...}.
    """
    try:
        data = {"blocks": obj.blocks.__root__} if hasattr(obj, "blocks") else (
            obj if isinstance(obj, dict) and "blocks" in obj else {"blocks": {}}
        )
        s = json.dumps(_deep_intify(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return _sha256_hex_text(s)
    except Exception:
        return ""

# ------------------------- Current Inputs Signature (frozen) -------------------------
def ssot_frozen_sig_from_ib() -> tuple[str, str, str, str, str]:
    """Read the canonical 5-tuple from st.session_state['_inputs_block'] if present."""
    ib = st.session_state.get("_inputs_block") or {}
    h = ib.get("hashes") or {}
    b = str(h.get("boundaries_hash", ib.get("boundaries_hash", "")))
    C = str(h.get("C_hash",         ib.get("C_hash", "")))
    H = str(h.get("H_hash",         ib.get("H_hash", "")))
    U = str(h.get("U_hash",         ib.get("U_hash", "")))
    S = str(h.get("shapes_hash",    ib.get("shapes_hash", "")))
    if not any((b, C, H, U, S)):
        return ("", "", "", "", "")
    return (b, C, H, U, S)

def current_inputs_sig(*, _ib: dict | None = None) -> tuple[str, str, str, str, str]:
    """
    Canonical 5-tuple (D, C, H, U, SHAPES). If _ib provided, read from that,
    otherwise read from the frozen st.session_state['_inputs_block'].
    """
    if _ib is not None:
        h = dict((_ib or {}).get("hashes") or {})
        return (
            str(h.get("boundaries_hash") or ""),
            str(h.get("C_hash")         or ""),
            str(h.get("H_hash")         or ""),
            str(h.get("U_hash")         or ""),
            str(h.get("shapes_hash")    or ""),
        )
    return ssot_frozen_sig_from_ib()

# ------------------------- SSOT live fingerprint (what’s currently loaded in memory) -------------------------
def ssot_live_sig(boundaries_obj=None, cmap_obj=None, H_obj=None, shapes_obj=None) -> tuple[str, str, str, str, str]:
    """
    Compute the live 5-tuple signature from in-memory objects:
    (D, C, H, U, SHAPES). In this app U ≡ SHAPES, so we mirror the same hash for both.
    """
    boundaries_obj = boundaries_obj or globals().get("boundaries")
    cmap_obj       = cmap_obj       or globals().get("cmap")
    H_obj          = H_obj          or (st.session_state.get("overlap_H") or globals().get("H_obj"))
    shapes_obj     = shapes_obj     or globals().get("shapes")

    hB = ssot_stable_blocks_sha(boundaries_obj) if boundaries_obj else ""
    hC = ssot_stable_blocks_sha(cmap_obj)       if cmap_obj       else ""
    hH = ssot_stable_blocks_sha(H_obj)          if H_obj          else ""
    hU = ssot_stable_blocks_sha(shapes_obj)     if shapes_obj     else ""
    hS = hU  # mirror by design
    return (hB, hC, hH, hU, hS)

# --------------------- Publish _inputs_block after Overlap ---------------------
def ssot_publish_block(*, boundaries_obj, cmap_obj, H_obj, shapes_obj, n3: int, projector_filename: str = "") -> dict:
    """
    Publish canonical _inputs_block into session state and return change info.
    Also stamps dims, filenames, and sets freshness anchors.
    """
    ss = st.session_state
    before = ssot_frozen_sig_from_ib()

    hB, hC, hH, hU, hS = ssot_live_sig(boundaries_obj, cmap_obj, H_obj, shapes_obj)
    H2 = (H_obj.blocks.__root__.get("2") or []) if (H_obj and hasattr(H_obj, "blocks")) else []
    dims = {"n2": (len(H2) if H2 else 0), "n3": int(n3 or 0)}
    files = {
        "boundaries": ss.get("fname_boundaries", "boundaries.json"),
        "cmap":       ss.get("fname_cmap",       "cmap.json"),
        "H":          ss.get("fname_h",         "H.json"),
        "U":          ss.get("fname_shapes",    "shapes.json"),
        "shapes":     ss.get("fname_shapes",    "shapes.json"),
    }
    if projector_filename:
        files["projector"] = projector_filename

    hashes = {
        "boundaries_hash": hB, "C_hash": hC, "H_hash": hH, "U_hash": hU, "shapes_hash": hS,
    }

    # Save block (SSOT)
    ss["_inputs_block"] = {
        "hashes":    hashes,
        "dims":      dims,
        "filenames": files,
        # legacy flatten (readers that still look at top-level fields)
        "boundaries_hash": hB, "C_hash": hC, "H_hash": hH, "U_hash": hU, "shapes_hash": hS,
    }

    after = ssot_frozen_sig_from_ib()
    changed = (before != after)

    # Freshness anchors for stale detection
    ss["_has_overlap"]        = True
    ss["_live_fp_at_overlap"] = ssot_live_sig(boundaries_obj, cmap_obj, H_obj, shapes_obj)

    return {"before": before, "after": after, "changed": changed}

# --------------------- Staleness Check (non-blocking) ---------------------
def ssot_is_stale() -> bool:
    """True if current live fingerprint differs from frozen _inputs_block."""
    ss = st.session_state
    if not ss.get("_has_overlap"):
        return False
    frozen = ssot_frozen_sig_from_ib()
    if not any(frozen):
        return False
    live_now = ssot_live_sig()
    return tuple(frozen) != tuple(live_now)

# ------------------------- Projector helpers -------------------------
def _auto_pj_hash_from_rc(rc: dict) -> str:
    """Stable hash for AUTO projector spec derived from lane_mask_k3."""
    try:
        lm = rc.get("lane_mask_k3") or []
        blob = json.dumps(lm, sort_keys=True, separators=(",", ":"))
        return _sha256_hex_text(blob)
    except Exception:
        return ""

# ------------------------- Key Generators & Widget-Key Deduper -------------------------
def _mkkey(ns: str, name: str) -> str:
    """Deterministic widget key: '<ns>__<name>'."""
    return f"{ns}__{name}"

def ensure_unique_widget_key(key: str) -> str:
    """
    If a widget key was already used in this run, suffix it with __2/__3/…
    Use this when you cannot easily rename at call site.
    """
    ss = st.session_state
    used = ss.setdefault("_used_widget_keys", set())
    if key not in used:
        used.add(key); return key
    i = 2
    while True:
        k2 = f"{key}__{i}"
        if k2 not in used:
            used.add(k2)
            if not ss.get("_warned_dup_keys", False):
                st.caption("⚠️ auto-deduped a duplicate widget key; please rename keys in code.")
                ss["_warned_dup_keys"] = True
            return k2
        i += 1

class _WKey:
    shapes_up     = _mkkey("inputs",  "shapes_uploader")
    shapes_up_alt = _mkkey("inputsB", "shapes_uploader")
WKEY = _WKey()

# ------------------------- Time/UUID Utilities -------------------------
def new_run_id() -> str:
    return str(uuid.uuid4())

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

# ------------------------- Fixture Nonce Utilities -------------------------
def _ensure_fixture_nonce():
    ss = st.session_state
    if "fixture_nonce" not in ss:
        ss["fixture_nonce"] = int(ss.get("_fixture_nonce", 0)) or 1
        ss["_fixture_nonce"] = ss["fixture_nonce"]

def _bump_fixture_nonce():
    ss = st.session_state
    cur = int(ss.get("fixture_nonce", 0))
    ss["fixture_nonce"] = cur + 1
    ss["_fixture_nonce"] = ss["fixture_nonce"]

_mark_fixtures_changed      = _bump_fixture_nonce  # legacy alias
_soft_reset_before_overlap   = lambda: soft_reset_before_overlap()

# ------------------------- Freshness Guard (non-blocking variant) -------------------------
def require_fresh_run_ctx(*, stop_on_error: bool = False):
    """
    Non-blocking by default: returns rc or None and emits warnings.
    If stop_on_error=True, mimics old behavior and st.stop() on violations.
    """
    _ensure_fixture_nonce()
    ss = st.session_state
    rc = ss.get("run_ctx")
    def _halt(msg):
        st.warning(msg)
        if stop_on_error:
            st.stop()

    if not rc:
        _halt("STALE_RUN_CTX: Run Overlap first."); return None
    if int(rc.get("fixture_nonce", -1)) != int(ss.get("fixture_nonce", -2)):
        _halt("STALE_RUN_CTX: Inputs changed; please click Run Overlap to refresh."); return None
    n3 = int(rc.get("n3") or 0)
    lm = list(rc.get("lane_mask_k3") or [])
    if lm and n3 and len(lm) != n3:
        _halt("Context mask length mismatch; please click Run Overlap to refresh."); return None
    return rc

# ------------------------- Mask from d3 (truth mask) -------------------------
def _truth_mask_from_d3(d3: list[list[int]]) -> list[int]:
    if not d3 or not d3[0]:
        return []
    rows, cols = len(d3), len(d3[0])
    return [1 if any(int(d3[i][j]) & 1 for i in range(rows)) else 0 for j in range(cols)]

def rectify_run_ctx_mask_from_d3(stop_on_error: bool = False):
    ss = st.session_state
    rc = require_fresh_run_ctx(stop_on_error=stop_on_error)
    if not rc: return None
    d3 = rc.get("d3") or []
    n3 = int(rc.get("n3") or 0)
    if not d3 or n3 <= 0:
        msg = "STALE_RUN_CTX: d3/n3 unavailable. Run Overlap."
        st.warning(msg)
        if stop_on_error: st.stop()
        return None
    lm_truth = _truth_mask_from_d3(d3)
    if len(lm_truth) != n3:
        msg = f"STALE_RUN_CTX: lane mask length {len(lm_truth)} != n3 {n3}. Run Overlap."
        st.warning(msg)
        if stop_on_error: st.stop()
        return None
    lm_rc = list(rc.get("lane_mask_k3") or [])
    if lm_rc != lm_truth:
        rc["lane_mask_k3"] = lm_truth
        ss["run_ctx"] = rc
        st.info(f"Rectified run_ctx.lane_mask_k3 from {lm_rc} → {lm_truth} based on stored d3.")
    return ss["run_ctx"]

# ------------------------- Cache Reset -------------------------
def soft_reset_before_overlap():
    ss = st.session_state
    for k in (
        "run_ctx", "overlap_out", "overlap_cfg", "overlap_policy_label",
        "overlap_H", "residual_tags", "proj_meta", "ab_compare",
        "cert_payload", "last_cert_path", "_last_cert_write_key",
        "_projector_cache", "_projector_cache_ab",
    ):
        ss.pop(k, None)

# ------------------------- JSONL Helpers -------------------------
def _atomic_append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(row, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(blob); tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    with open(path, "a", encoding="utf-8") as final, open(tmp_name, "r", encoding="utf-8") as src:
        shutil.copyfileobj(src, final)
    os.remove(tmp_name)

def _read_jsonl_tail(path: Path, N: int = 200) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell(); chunk = 64 * 1024; data = b""
            while len(data.splitlines()) <= N + 1 and f.tell() > 0:
                step = min(chunk, f.tell())
                f.seek(-step, os.SEEK_CUR)
                data = f.read(step) + data
                f.seek(-step, os.SEEK_CUR)
            lines = data.splitlines()[-N:]
        return [json.loads(l.decode("utf-8")) for l in lines if l.strip()]
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-N:]
        out = []
        for ln in lines:
            try: out.append(json.loads(ln))
            except Exception: continue
        return out

# ------------------------- UI Predicates -------------------------
def is_projected_green(run_ctx: dict | None, overlap_out: dict | None) -> bool:
    if not run_ctx or not overlap_out: return False
    mode = str(run_ctx.get("mode") or "")
    return mode.startswith("projected") and bool(((overlap_out.get("3") or {}).get("eq", False)))

def is_strict_red_lanes(run_ctx: dict | None, overlap_out: dict | None, residual_tags: dict | None) -> bool:
    if not run_ctx or not overlap_out: return False
    if str(run_ctx.get("mode") or "") != "strict": return False
    if bool(((overlap_out.get("3") or {}).get("eq", True))): return False
    tag = ((residual_tags or {}).get("strict") or "")
    return tag == "lanes"

# ------------------------- Hash Key Builders -------------------------
def gallery_key(row: dict) -> tuple:
    pol = row.get("policy") or {}; h = row.get("hashes") or {}
    return (row.get("district",""), pol.get("policy_tag",""),
            h.get("boundaries_hash",""), h.get("C_hash",""), h.get("H_hash",""), h.get("U_hash",""))

def witness_key(row: dict) -> tuple:
    pol = row.get("policy") or {}; h = row.get("hashes") or {}
    return (row.get("district",""), row.get("reason",""), row.get("residual_tag",""),
            pol.get("policy_tag",""), h.get("boundaries_hash",""), h.get("C_hash",""),
            h.get("H_hash",""), h.get("U_hash",""))

if "_gallery_keys" not in st.session_state: st.session_state["_gallery_keys"] = set()
if "_witness_keys" not in st.session_state: st.session_state["_witness_keys"] = set()

# ------------------------- Run Stamp -------------------------
def run_stamp_line() -> str:
    ss = st.session_state
    rc = ss.get("run_ctx") or {}
    ib = (ss.get("_inputs_block") or {}).get("hashes", {})
    pol = rc.get("policy_tag", "?"); n3 = int(rc.get("n3") or 0)
    hB = (ib.get("boundaries_hash","") or "")[:8]
    hC = (ib.get("C_hash","")         or "")[:8]
    hH = (ib.get("H_hash","")         or "")[:8]
    hU = (ib.get("U_hash","")         or "")[:8]
    pH = (rc.get("projector_hash","") or "")[:8]
    rid= (rc.get("run_id","")         or "")[:8]
    return f"{pol} | n3={n3} | B {hB} · C {hC} · H {hH} · U {hU} | P {pH} | run {rid}"


# ============================= END TOP HELPERS — CANONICAL =============================
# ---------- Uploaded-file cache ----------
def _upload_cache() -> dict:
    return st.session_state.setdefault("_upload_cache", {})  # {sha256: {"bytes": b, "json": obj, "name": str}}

def _bytes_from_upload(upload) -> tuple[bytes, str]:
    """
    Return (bytes, display_name) from a variety of upload types:
    - streamlit UploadedFile (getvalue/read)
    - path-like (str/Path)
    - raw bytes/bytearray
    - already a parsed dict -> (None, "<dict>")  (caller should short-circuit)
    """
    if upload is None:
        return b"", ""
    # Already-parsed dict (some callers pass dicts)
    if isinstance(upload, dict):
        return b"", "<dict>"
    # Streamlit UploadedFile
    name = getattr(upload, "name", None) or "uploaded.json"
    if hasattr(upload, "getvalue"):
        try:
            return upload.getvalue(), name
        except Exception:
            pass
    if hasattr(upload, "read"):
        try:
            # Try not to consume permanently: read then rewind if possible
            pos = upload.tell() if hasattr(upload, "tell") else None
            data = upload.read()
            if pos is not None and hasattr(upload, "seek"):
                upload.seek(pos)
            return data, name
        except Exception:
            pass
    # File path
    if isinstance(upload, (str, Path)):
        p = Path(upload)
        return (p.read_bytes(), p.name) if p.exists() else (b"", "")
    # Bytes-like
    if isinstance(upload, (bytes, bytearray)):
        return (bytes(upload), name)
    return b"", ""

def read_json_file(upload):
    """
    Robust JSON reader with caching. Returns dict or None.
    Safe to call multiple times on the same UploadedFile.
    """
    # Short-circuit for dict
    if isinstance(upload, dict):
        return upload

    data, name = _bytes_from_upload(upload)
    if not data:
        return None

    h = hashlib.sha256(data).hexdigest()
    cache = _upload_cache()
    entry = cache.get(h)
    if entry and "json" in entry:
        return entry["json"]

    # Decode then parse
    try:
        # Try utf-8 first, fallback to latin-1 (very rare)
        try:
            txt = data.decode("utf-8")
        except UnicodeDecodeError:
            txt = data.decode("latin-1")
        obj = json.loads(txt)
        cache[h] = {"bytes": data, "json": obj, "name": name}
        return obj
    except Exception as e:
        st.warning(f"Failed to parse JSON from {name}: {e}")
        return None

def _stamp_filename(state_key: str, upload):
    """
    Record the uploaded filename (for provenance) without reading the file.
    """
    if upload is None:
        return
    name = getattr(upload, "name", None)
    if not name:
        # Attempt to infer from cache if we just parsed it
        data, name2 = _bytes_from_upload(upload)
        name = name2 or "uploaded.json"
    st.session_state[state_key] = name

def safe_expander(label: str, expanded: bool = False):
    try:
        return st.expander(label, expanded=expanded)
    except Exception:
        @contextlib.contextmanager
        def _noop():
            yield
        return _noop()

# ---- Policy/config helpers (minimal, canonical) ----
def cfg_strict() -> dict:
    return {
        "enabled_layers": [],
        "source": {},               # no layer sources in strict
        "projector_files": {},      # none
    }

def cfg_projected_base() -> dict:
    return {
        "enabled_layers": [3],      # we project layer 3
        "source": {"3": "auto"},    # default projector source
        "projector_files": {},      # filled only for 'file'
    }

def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src = (cfg.get("source") or {}).get("3", "auto")
    mode = "file" if src == "file" else "auto"
    # keep your established label shape
    return f"projected(columns@k=3,{mode})"

# (optional) projector-file validator; keep as no-op if you don't need it yet
def validate_projector_file_strict(P, *, n3: int, lane_mask: list[int]):
    return  # implement later if you want strict checks for Π

# ---- compat adapter for current_inputs_sig (handles 0-arg & 1-arg versions) ----
def _current_inputs_sig_compat(*args, **kwargs):
    """
    Try to call current_inputs_sig as-is. If the active def only accepts 0 args,
    fall back to reading from provided _ib (or session).
    Returns a 5-tuple (B,C,H,U,S).
    """
    try:
        # if your canonical version exists, this just works
        return current_inputs_sig(*args, **kwargs)
    except TypeError:
        # zero-arg legacy: extract _ib if caller provided it
        _ib = None
        if args:
            _ib = args[0]
        _ib = kwargs.get("_ib", _ib)
        if _ib is not None:
            h = dict((_ib or {}).get("hashes") or {})
            return (
                str(h.get("boundaries_hash") or ""),
                str(h.get("C_hash")         or ""),
                str(h.get("H_hash")         or ""),
                str(h.get("U_hash")         or ""),
                str(h.get("shapes_hash")    or ""),
            )
        # final fallback: whatever is frozen in session
        return ssot_frozen_sig_from_ib()







# ───────────────────────────────── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown("### Upload core inputs")
    st.caption(
        "**Shapes (required):**\n\n```json\n{\\\"n\\\": {\\\"3\\\":3, \\\"2\\\":2, \\\"1\\\":0}}\n```\n\n"
        "**Boundaries (required):**\n\n```json\n{\\\"blocks\\\": {\\\"3\\\": [[...]], \\\"2\\\": [[...]]}}\n```\n\n"
        "**CMap / Move (required):**\n\n```json\n{\\\"blocks\\\": {\\\"3\\\": [[...]], \\\"2\\\": [[...]]}}\n```\n\n"
        "**Support (optional):** either `{degree: mask}` or `{\\\"masks\\\": {degree: mask}}`.\n\n"
        "**Triangle schema (optional):** degree-keyed `{ \\\"2\\\": {\\\"A\\\":..., \\\"B\\\":..., \\\"J\\\":...}, ... }`."
    )

    # Uploaders
    f_shapes   = st.file_uploader("Shapes (shapes.json)",            type=["json"], key="shapes")
    f_bound    = st.file_uploader("Boundaries (boundaries.json)",    type=["json"], key="bound")
    f_cmap     = st.file_uploader("CMap / Move (Cmap_*.json)",       type=["json"], key="cmap")
        # immediately after those three lines:
    st.session_state["uploaded_shapes"]     = f_shapes
    st.session_state["uploaded_boundaries"] = f_bound
    st.session_state["uploaded_cmap"]       = f_cmap
    f_support  = st.file_uploader("Support policy (optional)",       type=["json"], key="support")
    f_triangle = st.file_uploader("Triangle schema (optional)",      type=["json"], key="tri")
    seed       = st.text_input("Seed", "super-seed-A")

    # filename stamps for provenance
    _stamp_filename("fname_shapes",     f_shapes)
    _stamp_filename("fname_boundaries", f_bound)
    _stamp_filename("fname_cmap",       f_cmap)

    # Show raw-bytes hash to help populate DISTRICT_MAP
    if f_bound is not None and hasattr(f_bound, "getvalue"):
        _raw   = f_bound.getvalue()
        _bhash = _sha256_hex_bytes(_raw)
        st.caption(f"boundaries raw-bytes hash: {_bhash}")
        st.code(f'DISTRICT_MAP["{_bhash}"] = "D?"  # ← set D1/D2/D3/D4', language="python")

# ───────────────────────────── LOAD CORE JSONS ───────────────────────────────
d_shapes = read_json_file(f_shapes)
d_bound  = read_json_file(f_bound)
d_cmap   = read_json_file(f_cmap)

# Shared inputs_block (SSOT) in session
st.session_state.setdefault("_inputs_block", {})
ib = st.session_state["_inputs_block"]

if d_shapes and d_bound and d_cmap:
    try:
        # Parse core objects
        shapes     = io.parse_shapes(d_shapes)
        boundaries = io.parse_boundaries(d_bound)
        cmap       = io.parse_cmap(d_cmap)
        support    = io.parse_support(read_json_file(f_support))            if f_support  else None
        triangle   = io.parse_triangle_schema(read_json_file(f_triangle))   if f_triangle else None

        # Prefer raw-bytes boundary hash when available
        try:
            if hasattr(f_bound, "getvalue"):
                boundaries_hash_fresh = _sha256_hex_bytes(f_bound.getvalue())
            else:
                boundaries_hash_fresh = _sha256_hex_obj(d_bound)
        except Exception:
            boundaries_hash_fresh = _sha256_hex_obj(d_bound)

        # Light district info (lane mask + signature)
        d3_block          = (boundaries.blocks.__root__.get("3") or [])
        lane_mask_k3_now  = _lane_mask_from_d3(boundaries)
        d3_rows           = len(d3_block)
        d3_cols           = (len(d3_block[0]) if d3_block else 0)
        district_sig      = _district_signature(lane_mask_k3_now, d3_rows, d3_cols)
        district_id_fresh = DISTRICT_MAP.get(boundaries_hash_fresh, "UNKNOWN")

        # Clear stale session bits if boundaries changed
        _prev_bhash = st.session_state.get("_last_boundaries_hash")
        if _prev_bhash and _prev_bhash != boundaries_hash_fresh:
            for k in ("ab_compare", "district_id", "_projector_cache"):
                st.session_state.pop(k, None)
        st.session_state["_last_boundaries_hash"] = boundaries_hash_fresh

        # ── SSOT: authoritative filenames, dims, and hashes (no recompute elsewhere) ──
        C2 = (cmap.blocks.__root__.get("2") or [])
        C3 = (cmap.blocks.__root__.get("3") or [])
        # H source = parsed overlap_H if present, else empty cmap shell (consistent schema)
        H_used = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})

        ib["filenames"] = {
            "boundaries": st.session_state.get("fname_boundaries", "boundaries.json"),
            "C":          st.session_state.get("fname_cmap",       "cmap.json"),
            "H":          "H.json",   # adjust if you have an uploader for H
            "U":          st.session_state.get("fname_shapes",     "shapes.json"),
        }
        ib["dims"] = {
            "n2": len(C2),
            "n3": (len(C3[0]) if C3 else (len(d3_block[0]) if (d3_block and d3_block[0]) else 0)),
        }
        ib["boundaries_hash"] = boundaries_hash_fresh
        ib["C_hash"]          = hash_json(cmap.dict())
        ib["H_hash"]          = hash_json(H_used.dict())
        ib["U_hash"]          = hash_json(shapes.dict())
        ib["shapes_hash"]     = ib["U_hash"]  # 3D alias

        # Mirror fresh district info for later blocks
        st.session_state["_district_info"] = {
            "district_id":        district_id_fresh,
            "boundaries_hash":    boundaries_hash_fresh,
            "lane_mask_k3_now":   lane_mask_k3_now,
            "district_signature": district_sig,
            "d3_rows": d3_rows,
            "d3_cols": d3_cols,
        }
        st.session_state["district_id"] = district_id_fresh

        # Validate schemas
        io.validate_bundle(boundaries, shapes, cmap, support)
        st.success("Core schemas validated ✅")
        st.caption(
            f"district={district_id_fresh} · bhash={boundaries_hash_fresh[:12]} · "
            f"k3={lane_mask_k3_now} · sig={district_sig} · dims(n2,n3)={ib['dims'].get('n2')},{ib['dims'].get('n3')}"
        )

        with safe_expander("Hashes / provenance"):
            named = [("boundaries", boundaries.dict()),
                     ("shapes", shapes.dict()),
                     ("cmap", cmap.dict()),
                     ("H_used", H_used.dict())]
            if support:  named.append(("support",  support.dict()))
            if triangle: named.append(("triangle", triangle.dict()))
            ch = hashes.bundle_content_hash(named)
            ts = hashes.timestamp_iso_lisbon()
            rid = hashes.run_id(ch, ts)
            st.code(
                f"content_hash = {ch}\nrun_timestamp = {ts}\nrun_id = {rid}\napp_version = {APP_VERSION}",
                language="bash"
            )
            if st.button("Export ./reports → report.zip (quick)"):
                reports_dir = Path(DIRS["reports"])
                if not reports_dir.exists():
                    st.warning("No ./reports yet. Run a Tower or Manifest first.")
                else:
                    zpath = reports_dir / "report.zip"
                    export_mod.zip_report(str(reports_dir), str(zpath))
                    st.success(f"Exported: {zpath}")
                    with open(zpath, "rb") as fz:
                        st.download_button("Download report.zip", fz, file_name="report.zip")

    except Exception as e:
        st.error(f"Validation error: {e}")
        st.stop()
else:
    missing = [name for name, f in [("Shapes", d_shapes), ("Boundaries", d_bound), ("CMap", d_cmap)] if not f]
    st.info("Upload required files: " + ", ".join(missing))
    st.stop()

# ===================== Projected(FILE) validation banner & guard =====================
def file_validation_failed() -> bool:
    """Convenience predicate: returns True if last attempt to use FILE Π failed validation."""
    return bool(st.session_state.get("_file_mode_error"))





# ───────────────────────── GF(2) ops shim for Tab 2 (global) ──────────────────────────
# Provides mul, add, eye exactly as Tab 2 expects. If the library is present,
# we import; otherwise we use local pure-python fallbacks (bitwise XOR math).

try:
    from otcore.linalg_gf2 import mul as _mul_lib, add as _add_lib, eye as _eye_lib
    mul = _mul_lib
    add = _add_lib
    eye = _eye_lib
except Exception:
    def mul(A, B):
        if not A or not B or not A[0] or not B[0]:
            return []
        m, kA = len(A), len(A[0])
        kB, n = len(B), len(B[0])
        if kA != kB:
            return []
        C = [[0]*n for _ in range(m)]
        for i in range(m):
            Ai = A[i]
            for k in range(kA):
                if Ai[k] & 1:
                    Bk = B[k]
                    for j in range(n):
                        C[i][j] ^= (Bk[j] & 1)
        return C

    def add(A, B):
        if not A: return B or []
        if not B: return A or []
        r, c = len(A), len(A[0])
        if len(B) != r or len(B[0]) != c:
            return A  # shape mismatch: leave A unchanged (safe fallback)
        return [[(A[i][j] ^ B[i][j]) for j in range(c)] for i in range(r)]

    def eye(n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

# ───────────────────────── Reuse canonical helpers (NO redefinitions here) ─────────────
# We purposefully do NOT redefine _deep_intify/hash_json/_sha256_* etc. Use the canonical
# ones you placed at the top of the file:
#   - hash_json(...)
#   - ssot_stable_blocks_sha(...)
#   - ssot_publish_block(...)
#   - ssot_live_sig(...), ssot_frozen_sig_from_ib(...), ssot_is_stale(...)
#   - _truth_mask_from_d3(...)
# If any of these are missing, define them in the canonical block (not here).

# ---------- SSOT publisher (alias to canonical) ----------
def publish_inputs_block(*, boundaries_obj, cmap_obj, H_obj, shapes_obj, n3: int, projector_filename: str = ""):
    """
    Thin alias so callers in this tab can keep using publish_inputs_block(...).
    Delegates to your canonical ssot_publish_block.
    """
    return ssot_publish_block(
        boundaries_obj=boundaries_obj,
        cmap_obj=cmap_obj,
        H_obj=H_obj,
        shapes_obj=shapes_obj,
        n3=n3,
        projector_filename=projector_filename,
    )
with tab2:
    st.subheader("Overlap")
    # Your provided code starts here
    def _xor_mat(A, B):
        # prefer library add() if available (keeps one implementation)
        if "add" in globals() and callable(globals()["add"]):
            return globals()["add"](A, B)
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]

    def _bottom_row(M):
        return M[-1] if (M and len(M)) else []

    def _load_h_local():
        """Best-effort H loader used in Tab 2; resilient to missing file/var."""
        try:
            # prefer a session-uploaded H if your UI stashes it in st.session_state
            up = st.session_state.get("f_H")
            if up is not None:
                return io.parse_cmap(read_json_file(up))
        except Exception:
            pass
        try:
            # fall back to a module-level f_H if present
            if 'f_H' in globals() and globals()['f_H'] is not None:
                return io.parse_cmap(read_json_file(globals()['f_H']))
        except Exception:
            pass
        try:
            # finally, fall back to whatever was produced by Overlap
            H_obj = st.session_state.get("overlap_H")
            if H_obj is not None:
                return H_obj
        except Exception:
            pass
        # last resort: empty cmap
        return io.parse_cmap({"blocks": {}})

    def _lane_mask_from_d3_strict(boundaries_obj):
        """Derive lane mask directly from d3 by column support (strict truth)."""
        try:
            d3 = boundaries_obj.blocks.__root__.get("3") or []
        except Exception:
            d3 = []
        return _truth_mask_from_d3(d3)

    def _lane_mask_from_d3_local(boundaries_obj):
        # alias maintained for existing call-sites
        return _lane_mask_from_d3_strict(boundaries_obj)

    def _derive_mode_from_cfg(cfg: dict) -> str:
        if not cfg or not cfg.get("enabled_layers"):
            return "strict"
        src = (cfg.get("source", {}) or {}).get("3", "auto")
        return "projected(columns@k=3,file)" if src == "file" else "projected(columns@k=3,auto)"

    # Projected(FILE) validation banner (single source)
    def file_validation_failed() -> bool:
        """Return True if last attempt to use FILE Π failed validation."""
        return bool(st.session_state.get("_file_mode_error"))

    def _shape(M):
        return (len(M), len(M[0]) if (M and M[0]) else 0)

    def _guard_r3_shapes(H2, d3, C3):
        """Ensure H2·d3 and (C3⊕I3) shapes are consistent; tolerate empty during exploration."""
        rH, cH = _shape(H2); rD, cD = _shape(d3); rC, cC = _shape(C3)
        if not (rH and cH and rD and cD and rC and cC):
            return  # allow empty while exploring
        n3, n2 = rH, cH
        if not (rD == n2 and cD == n3 and rC == n3 and cC == n3):
            raise RuntimeError(
                f"R3_SHAPE: expected H2({n3}×{n2})·d3({n2}×{n3}) and (C3⊕I3)({n3}×{n3}); "
                f"got H2({rH}×{cH}), d3({rD}×{cD}), C3({rC}×{cC})"
            )

    


# ------------------------------ UI: policy + H + projector ------------------------------
colA, colB = st.columns([2, 2])
with colA:
    policy_choice = st.radio(
        "Policy",
        ["strict", "projected(columns@k=3,auto)", "projected(columns@k=3,file)"],
        index=0,
        horizontal=True,
        key="ov_policy_choice",
    )
with colB:
    f_H = st.file_uploader("Homotopy H (optional)", type=["json"], key="H_up")
    st.session_state["uploaded_H"] = f_H

proj_upload = st.file_uploader(
    "Projector Π (k=3) file (only for projected(columns@k=3,file))",
    type=["json"],
    key="pj_up",
)

# Active configuration builder
def _cfg_from_policy(policy_choice_str: str, pj_path: str | None) -> dict:
    if policy_choice_str == "strict":
        return cfg_strict()
    cfg = cfg_projected_base()
    if policy_choice_str.endswith("(auto)"):
        cfg.setdefault("source", {})["3"] = "auto"
        cfg.setdefault("projector_files", {})
    else:
        cfg.setdefault("source", {})["3"] = "file"
        if pj_path:
            cfg.setdefault("projector_files", {})["3"] = pj_path
    return cfg

# Handle projector upload
pj_saved_path = ""
if proj_upload is not None:
    os.makedirs("projectors", exist_ok=True)
    pj_saved_path = os.path.join("projectors", proj_upload.name)
    with open(pj_saved_path, "wb") as _pf:
        _pf.write(proj_upload.getvalue())
    st.caption(f"Saved projector: `{pj_saved_path}`")
    st.session_state["ov_last_pj_path"] = pj_saved_path

# Compute active config
cfg_active = _cfg_from_policy(
    policy_choice,
    st.session_state.get("ov_last_pj_path") or pj_saved_path or "",
)

# Display active policy label
st.caption(f"Active policy: `{policy_label_from_cfg(cfg_active)}`")


def _ab_is_fresh_now(pin=None, expected_embed_sig: str | None = None, **kwargs):
    """
    Back-compat freshness check.
    Accepts extra kwargs (rc, ib, cfg, expected_sig5, ...).

    Priority:
      1) Compare pin.payload.embed_sig to expected_embed_sig  → AB_FRESH / AB_STALE_EMBED_SIG
      2) Fallback: compare 5-hash arrays (legacy)            → AB_FRESH / AB_STALE_INPUTS_SIG
      3) If neither available                                → AB_CANNOT_EVAL
    """
    # Validate pin
    if not isinstance(pin, dict) or pin.get("state") != "pinned":
        return (False, "AB_PIN_MISSING")

    payload = pin.get("payload") or {}
    have_embed = str(payload.get("embed_sig") or "")
    exp_embed  = str(expected_embed_sig or "")

    # 1) Preferred: embed_sig equality
    if exp_embed:
        return (have_embed == exp_embed,
                "AB_FRESH" if have_embed == exp_embed else "AB_STALE_EMBED_SIG")

    # 2) Fallback: legacy 5-hash equality (keeps old reason tag)
    exp_sig5 = kwargs.get("expected_sig5")
    pin_sig5 = payload.get("inputs_sig_5")
    if exp_sig5 is None and "ib" in kwargs and kwargs["ib"] is not None:
        # If they passed ib, compute canonical 5-hash
        try:
            exp_sig5 = _frozen_inputs_sig_from_ib(kwargs["ib"], as_tuple=False)
        except Exception:
            exp_sig5 = None

    if exp_sig5 is not None and pin_sig5 is not None:
        fresh = list(exp_sig5) == list(pin_sig5)
        return (fresh, "AB_FRESH" if fresh else "AB_STALE_INPUTS_SIG")

    # 3) Couldn’t evaluate freshness
    return (False, "AB_CANNOT_EVAL")



        
# === BEGIN PATCH: READ-ONLY OVERLAP HYDRATOR (uses frozen SSOT only) ===
def overlap_ui_from_frozen():
    """
    Read-only UI refresh that re-computes visuals strictly from the *frozen* SSOT.
    It does NOT resolve sources, does NOT write fname_*, and does NOT freeze state.
    Safe to call after the single-button solver completed.
    """

    from pathlib import Path
    import json as _json

    ib = st.session_state.get("_inputs_block") or {}
    fns = (ib.get("filenames") or {})
    pB, pC, pH = fns.get("boundaries",""), fns.get("C",""), fns.get("H","")
    if not (pB and pC and pH):
        st.info("Overlap UI: frozen SSOT missing; run solver first.")
        return

    def _read_blocks(p):
        try: return (_json.loads(Path(p).read_text(encoding="utf-8")).get("blocks") or {})
        except Exception: return {}

    bB = _read_blocks(pB); bC = _read_blocks(pC); bH = _read_blocks(pH)
    d3 = bB.get("3") or []; C3 = bC.get("3") or []; H2 = bH.get("2") or []
    n2, n3 = len(d3), (len(d3[0]) if (d3 and d3[0]) else 0)

    # Diagnostics (no writes)
    st.caption(f"[Overlap UI] n₂×n₃ = {n2}×{n3} · src B:{Path(pB).name} · C:{Path(pC).name} · H:{Path(pH).name}")
    if C3 and len(C3)==len(C3[0]):
        I3 = [[1 if i==j else 0 for j in range(len(C3))] for i in range(len(C3))]
        def _mul(A,B):
            if not A or not B or not A[0] or not B[0] or len(A[0])!=len(B): return []
            m,k = len(A), len(A[0]); n = len(B[0])
            C = [[0]*n for _ in range(m)]
            for i in range(m):
                for t in range(k):
                    if A[i][t] & 1:
                        for j in range(n): C[i][j] ^= (B[t][j] & 1)
            return C
        def _xor(A,B):
            if not A: return [r[:] for r in (B or [])]
            if not B: return [r[:] for r in (A or [])]
            r,c = len(A), len(A[0]); return [[(A[i][j]^B[i][j]) & 1 for j in range(c)] for i in range(r)]
        R3s = _xor(_mul(H2, d3), _xor(C3, I3)) if (H2 and d3) else []
        bottom_H = (_mul(H2, d3)[-1] if (H2 and d3) else [])
        bottom_CI = (_xor(C3, I3)[-1] if C3 else [])
        lanes_auto = (C3[-1] if C3 else [])
        st.caption(f"[Overlap UI] (H2·d3)_bottom={bottom_H} · (C3⊕I3)_bottom={bottom_CI} · lanes(auto from C₃ bottom)={lanes_auto}")
    else:
        st.caption("[Overlap UI] C₃ not square; projected(columns@k=3,auto) is N/A here.")
# === END PATCH: READ-ONLY OVERLAP HYDRATOR ===


    




# --- Policy pill + run stamp (single rendering) --------------------------------
_rc = st.session_state.get("run_ctx") or {}
_ib = st.session_state.get("_inputs_block") or {}
policy_tag = _rc.get("policy_tag") or policy_label_from_cfg(cfg_active)  # keep your existing policy labeler
n3 = _rc.get("n3") or ((_ib.get("dims") or {}).get("n3", 0))
def _short8(h): return (h or "")[:8]
_h = (_ib.get("hashes") or {})
bH = _short8(_h.get("boundaries_hash", _ib.get("boundaries_hash","")))
cH = _short8(_h.get("C_hash",          _ib.get("C_hash","")))
hH = _short8(_h.get("H_hash",          _ib.get("H_hash","")))
uH = _short8(_h.get("U_hash",          _ib.get("U_hash","")))
pH = _short8(_rc.get("projector_hash","")) if str(_rc.get("mode","")).startswith("projected(columns@k=3,file)") else "—"

st.markdown(f"**Policy:** `{policy_tag}`")
st.caption(f"{policy_tag} | n3={n3} | b={bH} C={cH} H={hH} U={uH} P={pH}")

# Gentle hint only if any core hash is blank
if any(x in ("", None) for x in (_h.get("boundaries_hash"), _h.get("C_hash"), _h.get("H_hash"), _h.get("U_hash"))):
    st.info("SSOT isn’t fully populated yet. Run Overlap once to publish provenance hashes.")

# --- A/B status chip (no HTML repr; no duplicate logic) ------------------------
ab_pin = st.session_state.get("ab_pin") or {}
if ab_pin.get("state") == "pinned":
    fresh, reason = _ab_is_fresh_now(
        rc=_rc,
        ib=_ib,
        ab_payload=(ab_pin.get("payload") or {})
    )
    if fresh:
        st.success("A/B: Pinned · Fresh (will embed)")
    else:
        st.warning(f"A/B: Pinned · Stale ({reason})")
else:
    st.caption("A/B: —")




# ---------- A/B canonical helpers (drop-in) ----------

from pathlib import Path
import json as _json

def _ab_frozen_inputs_sig_list() -> list[str]:
    """Canonical 5-hash signature, frozen if your freezer provides it."""
    try:
        if "ssot_frozen_sig_from_ib" in globals() and callable(globals()["ssot_frozen_sig_from_ib"]):
            sig = ssot_frozen_sig_from_ib()
            if sig: return list(sig)
    except Exception:
        pass
    ib = st.session_state.get("_inputs_block") or {}
    h  = (ib.get("hashes") or {})
    return [
        str(h.get("boundaries_hash", ib.get("boundaries_hash",""))),
        str(h.get("C_hash",          ib.get("C_hash",""))),
        str(h.get("H_hash",          ib.get("H_hash",""))),
        str(h.get("U_hash",          ib.get("U_hash",""))),
        str(h.get("shapes_hash",     ib.get("shapes_hash",""))),
    ]



def _ab_load_h_latest():
    """
    Always try to read H from the *current* file path used by SSOT,
    then fall back to any freezer/local helpers, then session.
    """
    ss = st.session_state
    # 1) _inputs_block.filenames.H (preferred)
    fn = ((ss.get("_inputs_block") or {}).get("filenames") or {}).get("H") or ""
    # 2) explicit filename in session (older UIs used fname_h)
    if not fn:
        fn = ss.get("fname_h", "") or ""
    try:
        if fn and Path(fn).exists():
            data = _json.loads(Path(fn).read_text(encoding="utf-8"))
            return io.parse_cmap(data)
    except Exception:
        pass
    # 3) your app’s loader, if present
    try:
        if "_load_h_local" in globals() and callable(globals()["_load_h_local"]):
            return _load_h_local()
    except Exception:
        pass
    # 4) last resort – session snapshot
    return ss.get("overlap_H") or io.parse_cmap({"blocks": {}})

# ====================== A/B compat shims (define only if missing) ======================
import copy as _copy

if "_shape_ok" not in globals():
    def _shape_ok(A, B):
        try:
            return bool(A and B and A[0] and B[0] and (len(A[0]) == len(B)))
        except Exception:
            return False

if "_xor_gf2" not in globals():
    def _xor_gf2(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]

if "_eye" not in globals():
    def _eye(n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

if "_is_zero" not in globals():
    def _is_zero(M):
        return (not M) or all((x & 1) == 0 for row in M for x in row)

if "_recompute_strict_out" not in globals():
    def _recompute_strict_out(*, boundaries_obj, cmap_obj, H_obj, d3) -> dict:
        H2 = (H_obj.blocks.__root__.get("2") or []) if H_obj else []
        C3 = (cmap_obj.blocks.__root__.get("3") or [])
        I3 = _eye(len(C3)) if C3 else []
        eq3 = False
        try:
            if _shape_ok(H2, d3) and C3 and C3[0] and (len(C3) == len(C3[0])):
                if "mul" not in globals() or not callable(globals()["mul"]):
                    raise RuntimeError("GF(2) mul(H2,d3) not available.")
                R3 = _xor_gf2(mul(H2, d3), _xor_gf2(C3, I3))  # type: ignore[name-defined]
                eq3 = _is_zero(R3)
        except Exception:
            eq3 = False
        return {"2": {"eq": True}, "3": {"eq": bool(eq3), "n_k": (len(d3[0]) if (d3 and d3[0]) else 0)}}

if "_recompute_projected_out" not in globals():
    def _recompute_projected_out(*, rc, boundaries_obj, cmap_obj, H_obj) -> tuple[dict, dict]:
        d3 = rc.get("d3") or (boundaries_obj.blocks.__root__.get("3") or [])
        n3 = len(d3[0]) if (d3 and d3[0]) else 0
        lm = list(rc.get("lane_mask_k3") or ([1] * n3))
        # Use active Π if present; else diagonal(lm)
        P  = rc.get("P_active") or [[1 if (i == j and int(lm[j]) == 1) else 0 for j in range(n3)] for i in range(n3)]

        H2 = (H_obj.blocks.__root__.get("2") or []) if H_obj else []
        C3 = (cmap_obj.blocks.__root__.get("3") or [])
        I3 = _eye(len(C3)) if C3 else []

        shapes = {
            "H2": (len(H2), len(H2[0]) if H2 else 0),
            "d3": (len(d3), len(d3[0]) if d3 else 0),
            "C3": (len(C3), len(C3[0]) if C3 else 0),
            "P_active": (len(P), len(P[0]) if P else 0),
        }

        R3s, R3p = [], []
        try:
            if "mul" not in globals() or not callable(globals()["mul"]):
                raise RuntimeError("GF(2) mul missing.")
            R3s = _xor_gf2(mul(H2, d3), _xor_gf2(C3, I3)) if (_shape_ok(H2, d3) and C3 and C3[0] and (len(C3) == len(C3[0]))) else []
            R3p = mul(R3s, P) if (R3s and P and len(R3s[0]) == len(P)) else []
        except Exception:
            R3s, R3p = [], []

        def _nz_cols(M):
            if not M: return []
            r, c = len(M), len(M[0])
            return [j for j in range(c) if any(M[i][j] & 1 for i in range(r))]

        debug = {
            "shapes": shapes,
            "R3_strict_nz_cols": _nz_cols(R3s),
            "R3_projected_nz_cols": _nz_cols(R3p),
        }
        eq3_proj = _is_zero(R3p)
        return ({"2": {"eq": True}, "3": {"eq": bool(eq3_proj), "n_k": n3}}, debug)

if "_lane_bottoms_for_diag" not in globals():
    def _lane_bottoms_for_diag(*, H_obj, cmap_obj, d3, lane_mask):
        def _bottom_row(M): return M[-1] if (M and len(M)) else []
        H2 = (H_obj.blocks.__root__.get("2") or []) if H_obj else []
        C3 = (cmap_obj.blocks.__root__.get("3") or [])
        I3 = _eye(len(C3)) if C3 else []
        try:
            if "mul" not in globals() or not callable(globals()["mul"]):
                raise RuntimeError("GF(2) mul missing.")
            H2d3  = mul(H2, d3) if _shape_ok(H2, d3) else []
            C3pI3 = _xor_gf2(C3, I3) if (C3 and C3[0]) else []
        except Exception:
            H2d3, C3pI3 = [], []
        idx = [j for j, m in enumerate(lane_mask or []) if m]
        bH = _bottom_row(H2d3); bC = _bottom_row(C3pI3)
        return ([bH[j] for j in idx] if (bH and idx) else [],
                [bC[j] for j in idx] if (bC and idx) else [])
# =================== /A/B compat shims ===================
# ============== A/B policy + embed signature helpers (compat) ==============
import hashlib as _hash
import json as _json

if "_canonical_policy_tag" not in globals():
    def _canonical_policy_tag(rc: dict | None) -> str:
        rc = rc or {}
        # 1) If your app already stamped a label, trust it.
        lbl = rc.get("policy_tag")
        if lbl:
            return str(lbl)
        # 2) If your app exposes a policy label helper, use it.
        if "policy_label_from_cfg" in globals() and "cfg_active" in globals():
            try:
                return str(policy_label_from_cfg(cfg_active))  # type: ignore[name-defined]
            except Exception:
                pass
        # 3) Fallback from mode → canonical strings used across your UI.
        m = str(rc.get("mode", "strict")).lower()
        if m == "strict":
            return "strict"
        if "projected" in m and "file" in m:
            return "projected(columns@k=3,file)"
        if "projected" in m:
            return "projected(columns@k=3,auto)"
        return "strict"

if "_inputs_sig_now_from_ib" not in globals():
    def _inputs_sig_now_from_ib(ib: dict | None) -> list[str]:
        ib = ib or {}
        h = (ib.get("hashes") or {})
        return [
            str(h.get("boundaries_hash", ib.get("boundaries_hash",""))),
            str(h.get("C_hash",          ib.get("C_hash",""))),
            str(h.get("H_hash",          ib.get("H_hash",""))),
            str(h.get("U_hash",          ib.get("U_hash",""))),
            str(h.get("shapes_hash",     ib.get("shapes_hash",""))),
        ]

if "_ab_embed_sig" not in globals():
    def _ab_embed_sig() -> str:
        """
        Canonical signature that gates whether a pinned A/B snapshot is still
        'fresh' enough to embed into the cert/gallery. Includes:
          - frozen/current inputs_sig
          - canonical policy tag
          - projector hash (only for projected(columns@k=3,file))
        """
        ss = st.session_state
        rc = ss.get("run_ctx") or {}
        ib = ss.get("_inputs_block") or {}

        # Prefer frozen sig if your SSOT helper exists; else current inputs.
        try:
            if "ssot_frozen_sig_from_ib" in globals() and callable(globals()["ssot_frozen_sig_from_ib"]):
                inputs_sig = list(ssot_frozen_sig_from_ib() or [])  # type: ignore[name-defined]
            else:
                inputs_sig = _inputs_sig_now_from_ib(ib)
        except Exception:
            inputs_sig = _inputs_sig_now_from_ib(ib)

        pol = _canonical_policy_tag(rc)
        pj  = rc.get("projector_hash","") if str(rc.get("mode","")) == "projected(columns@k=3,file)" else ""

        blob = {"inputs": inputs_sig, "policy": pol, "projector_hash": pj}
        return _hash.sha256(_json.dumps(blob, separators=(",", ":"), sort_keys=True).encode("ascii")).hexdigest()
# ============ /A/B policy + embed signature helpers (compat) ============
def _pin_ab_for_cert(strict_out: dict, projected_out: dict):
    """Pin A/B snapshot into session with a canonical embed_sig + one-shot ticket."""

    ss = st.session_state
    ib = ss.get("_inputs_block") or {}
    rc = ss.get("run_ctx") or {}

    # inputs_sig: prefer frozen SSOT if available, else current block
    try:
        if "ssot_frozen_sig_from_ib" in globals() and callable(globals()["ssot_frozen_sig_from_ib"]):
            inputs_sig = list(ssot_frozen_sig_from_ib() or [])  # type: ignore[name-defined]
        else:
            inputs_sig = _inputs_sig_now_from_ib(ib)
    except Exception:
        inputs_sig = _inputs_sig_now_from_ib(ib)

    pol_tag = _canonical_policy_tag(rc)
    pj_hash = rc.get("projector_hash","") if str(rc.get("mode","")) == "projected(columns@k=3,file)" else ""

    payload = {
        "pair_tag": f"strict__VS__{pol_tag}",
        "inputs_sig": inputs_sig,
        "policy_tag": pol_tag,
        "strict":    {"out": dict(strict_out or {})},
        "projected": {"out": dict(projected_out or {}), "policy_tag": pol_tag, "projector_hash": pj_hash},
    }

    # >>> stamp canonical embed signature <<<
    payload["embed_sig"] = _ab_embed_sig()

    ss["ab_pin"] = {"state": "pinned", "payload": payload, "consumed": False}
    ss["_ab_ticket_pending"] = int(ss.get("_ab_ticket_pending", 0)) + 1
    ss["write_armed"] = True
    ss["armed_by"]    = "ab_compare"

# ===== ABX fixture helpers (module-scope, guarded) =====
from pathlib import Path
import json as _json, hashlib as _hashlib

if "abx_is_uploaded_file" not in globals():
    def abx_is_uploaded_file(x):
        # duck-type Streamlit's UploadedFile
        return hasattr(x, "getvalue") and hasattr(x, "name")

if "abx_read_json_any" not in globals():
    def abx_read_json_any(x, *, kind: str) -> tuple[dict, str, str]:
        """
        Accepts a path string/Path, a Streamlit UploadedFile, or a plain dict.
        Returns (json_obj, canonical_path, origin_tag) where origin_tag∈{"file","upload","dict",""}.
        For uploads, saves a stable copy under logs/_uploads and returns that path.
        """
        if not x:
            return {}, "", ""
        # 1) path-like
        try:
            if isinstance(x, (str, Path)):
                p = Path(x)
                if p.exists():
                    try:
                        return _json.loads(p.read_text(encoding="utf-8")), str(p), "file"
                    except Exception:
                        return {}, "", ""
        except Exception:
            pass
        # 2) UploadedFile
        if abx_is_uploaded_file(x):
            try:
                raw  = x.getvalue()
                text = raw.decode("utf-8")
                j    = _json.loads(text)
            except Exception:
                return {}, "", ""
            uploads_dir = Path(st.session_state.get("LOGS_DIR", "logs")) / "_uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            h12  = _hashlib.sha256(raw).hexdigest()[:12]
            base = Path(getattr(x, "name", f"{kind}.json")).name
            outp = uploads_dir / f"{kind}__{h12}__{base}"
            try:
                outp.write_text(text, encoding="utf-8")
            except Exception:
                pass
            return j, str(outp), "upload"
        # 3) already-parsed dict
        if isinstance(x, dict):
            return x, "", "dict"
        return {}, "", ""

if "abx_blocks_view" not in globals():
    def abx_blocks_view(obj):
        """Return {'blocks': ...} for your model/dict, or empty structure."""
        try:
            if hasattr(obj, "blocks") and hasattr(obj.blocks, "__root__"):
                return {"blocks": obj.blocks.__root__}
        except Exception:
            pass
        return obj if (isinstance(obj, dict) and "blocks" in obj) else {"blocks": {}}

if "abx_lane_mask_from_d3" not in globals():
    def abx_lane_mask_from_d3(d):
        return [1 if any(int(d[i][j]) & 1 for i in range(len(d))) else 0
                for j in range(len(d[0]) if (d and d[0]) else 0)]

if "abx_hash_json" not in globals():
    def abx_hash_json(obj):
        try:
            blob = _json.dumps(obj, separators=(",", ":"), sort_keys=True)
            return _hashlib.sha256(blob.encode("utf-8")).hexdigest()
        except Exception:
            return ""

if "abx_diag_from_mask" not in globals():
    def abx_diag_from_mask(lm_):
        n = len(lm_ or [])
        return [[1 if (i == j and int(lm_[j]) == 1) else 0 for j in range(n)] for i in range(n)]

if "abx_policy_tag" not in globals():
    def abx_policy_tag():
        rc0 = st.session_state.get("run_ctx") or {}
        m = str(rc0.get("mode", "strict")).lower()
        if m == "strict": return "strict"
        if "projected" in m and "file" in m: return "projected(columns@k=3,file)"
        if "projected" in m: return "projected(columns@k=3,auto)"
        return "strict"
# ===== /helpers =====
# === helpers required by the single-button solver (uploads-first source resolver) ===
import os, json as _json, hashlib as _hashlib
from pathlib import Path

# dirs
if "LOGS_DIR" not in globals():
    LOGS_DIR = Path("logs")
UPLOADS_DIR = LOGS_DIR / "_uploads"
for _p in (LOGS_DIR, UPLOADS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# tiny I/O shims
def _svr_hash_json(obj) -> str:
    try:
        blob = _json.dumps(obj, separators=(",", ":"), sort_keys=True)
        return _hashlib.sha256(blob.encode("utf-8")).hexdigest()
    except Exception:
        return ""

def _ab_pick_pin(policy: str | None = None):
    """Pick the right pin from session. policy in {None,'auto','file'}."""
    ss = st.session_state
    if policy == "file":
        return ss.get("ab_pin_file") or ss.get("ab_pin")
    if policy == "auto":
        return ss.get("ab_pin_auto") or ss.get("ab_pin")
    # default: prefer auto
    return ss.get("ab_pin_auto") or ss.get("ab_pin_file") or ss.get("ab_pin")

def _bundle_last_paths():
    """Return latest bundle dir and the two A/B file paths if present."""
    ss = st.session_state
    bdir = str(ss.get("last_bundle_dir") or "")
    if not bdir:
        # try to reconstruct from last run header data in rc/ib
        rc = ss.get("run_ctx") or {}
        ib = ss.get("_inputs_block") or {}
        sig8 = (ss.get("last_run_header") or "").split(" S=")[-1][:8] if ss.get("last_run_header") else ""
        district = ib.get("district_id") or "DUNKNOWN"
        # fall back to logs/certs/D*/sig8
        bdir = str(Path("logs")/ "certs" / district / sig8)
    p_auto = Path(bdir) / next((p for p in os.listdir(bdir) if p.startswith("ab_compare__strict_vs_projected_auto__")), "") if bdir and os.path.isdir(bdir) else ""
    p_file = Path(bdir) / next((p for p in os.listdir(bdir) if p.startswith("ab_compare__strict_vs_projected_file__")), "") if bdir and os.path.isdir(bdir) else ""
    return (bdir, (str(p_auto) if p_auto else ""), (str(p_file) if p_file else ""))

def _ab_expected_embed_sig_from_file(path: str) -> str | None:
    try:
        j = _json.loads(Path(path).read_text(encoding="utf-8"))
        return str(((j or {}).get("ab_pair") or {}).get("embed_sig") or "")
    except Exception:
        return None

# =============================== Bundle helpers ===============================
def _svr_bundle_dir(district_id: str, sig8: str) -> Path:
    base = Path(DIRS.get("certs","logs/certs"))
    p = base / str(district_id) / str(sig8)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _svr_bundle_fname(kind: str, district_id: str, sig8: str) -> str:
    # Map kind to canonical filenames
    if kind == "strict":
        return f"overlap__{district_id}__strict__{sig8}.json"
    if kind == "projected_auto":
        return f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json"
    if kind == "ab_auto":
        return f"ab_compare__strict_vs_projected_auto__{sig8}.json"
    if kind == "freezer":
        return f"projector_freezer__{district_id}__{sig8}.json"
    if kind == "ab_file":
        return f"ab_compare__strict_vs_projected_file__{sig8}.json"
    if kind == "projected_file":
        return f"overlap__{district_id}__projected_columns_k_3_file__{sig8}.json"
    return f"{kind}__{district_id}__{sig8}.json"

def _svr_write_cert_in_bundle(bundle_dir: Path, filename: str, payload: dict) -> Path:
    # Compute content hash, write atomically; skip rewrite when unchanged
    payload.setdefault("integrity", {})
    payload["integrity"]["content_hash"] = _svr_hash(payload)
    p = (bundle_dir / filename)
    tmp = p.with_suffix(".json.tmp") if p.suffix==".json" else p.with_suffix(p.suffix + ".tmp")
    # Dedup: if exists and same content hash, skip rewrite
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
            if (old or {}).get("integrity", {}).get("content_hash") == payload["integrity"]["content_hash"]:
                # unchanged — skipping rewrite
                return p
        except Exception:
            pass
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    return p
# ==============================================================================
def _svr_guarded_atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# accept path / UploadedFile / dict
def _is_uploaded_file(x): return hasattr(x, "getvalue") and hasattr(x, "name")



# normalize per kind → "blocks" payload we hash/persist
def _svr_as_blocks(j: dict, kind: str) -> dict:
    if not isinstance(j, dict): return {}
    if kind in ("B","C","H"):
        # prefer canonical {"blocks": {...}}
        if isinstance(j.get("blocks"), dict):
            return dict(j["blocks"])
        # tolerate legacy top-level degrees
        blk = {}
        for deg in ("1","2","3"):
            if deg in j and isinstance(j[deg], list):
                blk[deg] = j[deg]
        return blk
    # shapes: keep json as-is (we only hash it; not validated for slices)
    if isinstance(j.get("blocks"), dict):
        return dict(j["blocks"])
    return dict(j)






#######################################################################################

def _svr_pick_source(kind: str):
    ss = st.session_state
    # previously frozen canonical filenames (from _svr_freeze_ssot)
    ib_files = ((ss.get("_inputs_block") or {}).get("filenames") or {})
    frozen_key = {"B":"boundaries","C":"C","H":"H","U":"U"}[kind]
    # uploads-first precedence across common aliases
    precedence = {
        "B": ["uploaded_boundaries","fname_boundaries","B_up","f_B","boundaries_file","boundaries_obj","boundaries"],
        "C": ["uploaded_cmap","fname_cmap","C_up","f_C","cmap_file","cmap_obj","cmap"],
        "H": ["uploaded_H","fname_h","H_up","f_H","h_file","H_obj","overlap_H","H"],
        "U": ["uploaded_shapes","fname_shapes","U_up","f_U","shapes_file","shapes_obj","shapes","U"],
    }[kind]
    for k in precedence:
        if k in ss and ss.get(k) is not None:
            return ss.get(k)
    if ib_files.get(frozen_key):
        return ib_files.get(frozen_key)
    for k in precedence:
        if k in globals() and globals().get(k) is not None:
            return globals().get(k)
    return None



def _svr_freeze_ssot(pb):
    """Stamp _inputs_block in session with filenames, hashes, dims."""
    (pB,bB) = pb["B"]; (pC,bC) = pb["C"]; (pH,bH) = pb["H"]; (pU,bU) = pb["U"]
    d3 = bB.get("3") or []
    n2, n3 = len(d3), (len(d3[0]) if (d3 and d3[0]) else 0)
    hashes = {
        "boundaries_hash": _svr_hash_json({"blocks": bB}),
        "C_hash":          _svr_hash_json({"blocks": bC}),
        "H_hash":          _svr_hash_json({"blocks": bH}),
        "U_hash":          _svr_hash_json({"blocks": bU}),
        "shapes_hash":     _svr_hash_json({"blocks": bU}),  # shapes carried as-is
    }
    ib = {
        "filenames": {"boundaries": pB, "C": pC, "H": pH, "U": pU},
        "hashes": dict(hashes),
        "dims": {"n2": n2, "n3": n3},
        "district_id": district_from_hash(hashes["boundaries_hash"]),
    }
    st.session_state["_inputs_block"] = ib
    # keep run_ctx consistent
    rc = dict(st.session_state.get("run_ctx") or {})
    rc.update({"n2": n2, "n3": n3, "d3": d3})
    st.session_state["run_ctx"] = rc
    return ib, rc
# === /helpers ===

# --- legacy A/B embed signature (guarded) --------------------------------------
# Used by older pin/cert code paths: embed_sig = sha256({"inputs": [...], "policy": "...", lanes|reason})
if "_svr_embed_sig" not in globals():
    def _svr_embed_sig(inputs_sig, policy_tag, lanes_or_reason):
        """
        inputs_sig: list[str]  # 5-hash in any order you're passing today
        policy_tag: str        # e.g. "projected(columns@k=3,auto)"
        lanes_or_reason: list[int] OR str (N/A reason)
        """
        try:
            blob = {"inputs": list(inputs_sig or []), "policy": str(policy_tag)}
            if isinstance(lanes_or_reason, (list, tuple)):
                blob["lanes"] = [int(x) & 1 for x in lanes_or_reason]
            else:
                blob["projected_na_reason"] = str(lanes_or_reason)
            payload = _json.dumps(blob, separators=(",", ":"), sort_keys=True).encode("utf-8")
            return _hashlib.sha256(payload).hexdigest()
        except Exception:
            # fail-safe: still return a stable-ish value to avoid crashes
            return _hashlib.sha256(b"svr-embed-sig-fallback").hexdigest()

# === SINGLE-BUTTON SOLVER — strict → projected(columns@k=3,auto) → A/B(auto) → freezer → A/B(file) ===
import os, json as _json, hashlib as _hashlib
from pathlib import Path
from datetime import datetime, timezone

# ---------- tiny helpers (only define if missing) ----------
if "_svr_now_iso" not in globals():
    def _svr_now_iso():
        try: return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
        except Exception: return ""

if "_svr_hash" not in globals():
    def _svr_hash(obj):  # sha256(canonical json)
        return _hashlib.sha256(_json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()

if "_svr_eye" not in globals():
    def _svr_eye(n): return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

if "_svr_is_zero" not in globals():
    def _svr_is_zero(M): return (not M) or all((int(x) & 1) == 0 for row in M for x in row)

if "_svr_mul" not in globals():
    def _svr_mul(A,B):
        if not A or not B or not A[0] or not B[0] or len(A[0])!=len(B): return []
        m,k = len(A), len(A[0]); n = len(B[0])
        C = [[0]*n for _ in range(m)]
        for i in range(m):
            Ai = A[i]
            for t in range(k):
                if int(Ai[t]) & 1:
                    Bt = B[t]
                    for j in range(n):
                        C[i][j] ^= (int(Bt[j]) & 1)
        return C

if "_svr_xor" not in globals():
    def _svr_xor(A,B):
        if not A: return [row[:] for row in (B or [])]
        if not B: return [row[:] for row in (A or [])]
        r,c = len(A), len(A[0])
        return [[(int(A[i][j]) ^ int(B[i][j])) & 1 for j in range(c)] for i in range(r)]

if "_svr_atomic_write_json" not in globals():
    def _svr_guarded_atomic_write_json(path: Path, payload: dict):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)



# cert scaffold (reuse your existing ones if present)
if "_svr_cert_common" not in globals():
    def _svr_cert_common(ib, rc, policy_tag: str) -> dict:
        return {
            "schema_version": globals().get("SCHEMA_VERSION", "1"),
            "engine_rev":     globals().get("ENGINE_REV", ""),
            "written_at_utc": _svr_now_iso(),
            "app_version":    globals().get("APP_VERSION", "dev"),
            "policy_tag":     str(policy_tag),
            "run_id":         (rc.get("run_id") or ""),
            "district_id":    (ib.get("district_id") or ""),
            "sig8":           "",
            "inputs":         dict(ib),
            "integrity":      {"content_hash": ""},
        }



# unified embed builder (AUTO/File)
def _svr_build_embed(ib: dict,
                     policy: str,
                     lanes: list[int] | None = None,
                     projector_hash: str | None = None,
                     na_reason: str | None = None):
    """
    Build canonical A/B embed + embed_sig.

    Canonical inputs_sig_5 order:
      (hash_d, hash_U, hash_suppC, hash_suppH, hash_shapes)
    """
    try:
        _json
    except NameError:
        import json as _json  # noqa
    try:
        _hashlib
    except NameError:
        import hashlib as _hashlib  # noqa

    # dims + ids
    dims = (ib or {}).get("dims") or {}
    n2 = int(dims.get("n2") or 0)
    n3 = int(dims.get("n3") or 0)
    district_id = (ib or {}).get("district_id") or "DUNKNOWN"

    # canonical 5-hash signature
    inputs_sig_5 = _frozen_inputs_sig_from_ib(ib, as_tuple=False)

    # schema/engine (be lenient if constants not defined)
    schema_version = globals().get("SCHEMA_VERSION", "2.0.0")
    engine_rev = globals().get("ENGINE_REV", "rev-UNSET")
    fixture_label = (ib or {}).get("fixture_label", "")

    embed = {
        "schema_version": str(schema_version),
        "engine_rev": str(engine_rev),
        "district_id": str(district_id),
        "fixture_label": str(fixture_label),
        "dims": {"n2": n2, "n3": n3},
        "policy": str(policy),
        "projection_context": {},
        "inputs_sig_5": list(inputs_sig_5),
    }
    if lanes is not None:
        embed["projection_context"]["lanes"] = [int(x) & 1 for x in lanes]
    if projector_hash is not None:
        embed["projection_context"]["projector_hash"] = str(projector_hash)
    if na_reason is not None:
        embed["projection_context"]["na_reason_code"] = str(na_reason)

    s = _json.dumps(embed, separators=(",", ":"), sort_keys=True).encode("ascii")
    embed_sig = _hashlib.sha256(s).hexdigest()
    return embed, embed_sig


def _svr_apply_sig8(cert: dict, embed_sig: str) -> None:
    cert["sig8"] = (embed_sig or "")[:8]



# small witness helper
def _bottom_row(M): return M[-1] if (M and len(M)) else []

# ---------- UI ----------
with st.expander("A/B compare (strict vs projected(columns@k=3,auto))", expanded=False):
    # small local helper for FILE embed sig (kept local to avoid name collisions)
    def _svr_embed_sig_file(inputs_sig, projector_hash: str):
        blob = {
            "inputs": list(inputs_sig),
            "policy": "projected(columns@k=3,file)",
            "projector_hash": str(projector_hash or ""),
        }
        return _hashlib.sha256(_json.dumps(blob, separators=(",", ":"), sort_keys=True).encode("ascii")).hexdigest()

    # Preflight (read-only)
    try:
        pf = _svr_resolve_all_to_paths()
        pB,pC,pH,pU = Path(pf["B"][0]).name, Path(pf["C"][0]).name, Path(pf["H"][0]).name, Path(pf["U"][0]).name
        st.caption(f"Sources → B:{pB} · C:{pC} · H:{pH} · U:{pU}")

        d3pf = pf["B"][1].get("3") or []; C3pf = pf["C"][1].get("3") or []; H2pf = pf["H"][1].get("2") or []
        n2p, n3p = len(d3pf), (len(d3pf[0]) if (d3pf and d3pf[0]) else 0)
        if n2p and n3p:
            I3pf = _svr_eye(len(C3pf)) if (C3pf and len(C3pf)==len(C3pf[0])) else []
            C3pIpf = _svr_xor(C3pf, I3pf) if I3pf else []
            bottom_H  = (_svr_mul(H2pf, d3pf)[-1] if (H2pf and d3pf and _svr_mul(H2pf, d3pf)) else [])
            bottom_C  = (C3pf[-1] if C3pf else [])
            bottom_CI = (C3pIpf[-1] if C3pIpf else [])
            st.caption(f"Preflight — n₂×n₃ = {n2p}×{n3p} · (H2·d3)_bottom={bottom_H} · C3_bottom={bottom_C} · (C3⊕I3)_bottom={bottom_CI}")
        else:
            st.info("Preflight: upload B/C/H/U to run.")
    except Exception:
        st.info("Preflight: unable to resolve sources yet.")

        # --- Run button (full replacement) ---
        run_btn = st.button("Run solver (one press → 5 certs; +1 if FILE)", key="btn_svr_run")
        if run_btn:
            ss = st.session_state
            if ss.get('_solver_busy', False):
                st.warning('Solver is already running — debounced.')
            else:
                ss['_solver_busy'] = True
                ss['_solver_one_button_active'] = True
                try:
                    # 1) Freeze SSOT (single source of truth)
                    pb = _svr_resolve_all_to_paths()
                    ib, rc = _svr_freeze_ssot(pb)
                    rc["run_id"] = str(uuid.uuid4())

                    # guard: must have complete inputs
                    H2 = pb["H"][1].get("2") or []
                    d3 = pb["B"][1].get("3") or []
                    C3 = pb["C"][1].get("3") or []
                    if not (H2 and d3 and C3 and C3 and C3[0]):
                        st.warning("Inputs incomplete — upload B/C/H/U and run Overlap first.")
                        raise RuntimeError("INCOMPLETE_INPUTS")

                    n3 = len(C3[0])
                    I3 = _svr_eye(len(C3)) if (len(C3) == len(C3[0])) else []
                    # residual R3 = H2 d3 ⊕ (C3 ⊕ I3) for witnesses + mismatch cols
                    H2d3 = _svr_mul(H2, d3)
                    C3pI3 = _svr_xor(C3, I3) if I3 else []
                    R3s = _svr_xor(H2d3, C3pI3) if I3 else []
                    bH = (H2d3[-1] if (H2d3 and H2d3[-1] is not None) else [])
                    bCI = (C3pI3[-1] if C3pI3 else [])
                    sel_all = [1] * n3

                    # district + target dir (stable per press)
                    h = ib.get("hashes", {}) or {}
                    district_id = ib.get("district_id") or ("D" + str(h.get("boundaries_hash", ""))[:8])

                    # 2) Strict lap
                    strict_out = _svr_strict_from_blocks(pb["H"][1], pb["B"][1], pb["C"][1])
                    strict_k3 = (strict_out or {}).get("3", {}).get("eq", None)
                    strict_k2 = (strict_out or {}).get("2", {}).get("eq", None)

                    # 3) Projected (AUTO) lap
                    proj_meta, lanes, projected_out = _svr_projected_auto_from_blocks(pb["H"][1], pb["B"][1], pb["C"][1])
                    posed_auto = not bool(proj_meta.get("na"))
                    proj_auto_k3 = (projected_out or {}).get("3", {}).get("eq", None) if posed_auto else None
                    proj_auto_k2 = (projected_out or {}).get("2", {}).get("eq", None) if posed_auto else None

                    # 4) Unified A/B(AUTO) embed → embed_sig + sig8 + bundle dir
                    embed_auto, embed_sig_auto = _svr_build_embed(
                        ib,
                        policy="strict__VS__projected(columns@k=3,auto)",
                        lanes=(list(lanes) if (posed_auto and isinstance(lanes, (list, tuple))) else None),
                        na_reason=(proj_meta.get("reason") if not posed_auto else None),
                    )
                    sig8 = (embed_sig_auto or "")[:8]
                    _bundle_dir = _svr_bundle_dir(district_id, sig8)

                    # === C1: pattern-only coverage append (guarded) ===
                    try:
                        # Build σ (pattern-only) from d3 and label C1 membership
                        d3_mat  = pb["B"][1].get("3") or []
                        n2_now  = len(d3_mat)
                        n3_now  = len(d3_mat[0]) if (d3_mat and isinstance(d3_mat[0], (list, tuple))) else 0
                        sig, sig_err = compute_signature_from_d3(d3_mat, n2_now, n3_now)
                        if sig_err:
                            mem = {"status":"NA","proximity":"","sig_str":""}
                        else:
                            mem = c1_membership(sig)

                        # Lane mask & diagnostics (strict residual-based)
                        lane_src = lanes if isinstance(lanes, (list, tuple)) else []
                        L = [int(x) & 1 for x in lane_src]
                        sel_sum = sum(L)

                        # Selected / off-row mismatch columns under strict residual R3s
                        sel_mis  = _svr_mismatch_cols_from_R3(R3s, L) if (R3s and L) else []
                        off_mask = [(1 - b) for b in L] if L else ([0] * (n3_now if n3_now else 0))
                        off_mis  = _svr_mismatch_cols_from_R3(R3s, off_mask) if (R3s and off_mask) else []

                        selected_mismatch_rate = (len(sel_mis) / max(1, sel_sum)) if L else 0.0
                        offrow_mismatch_rate   = (len(off_mis) / max(1, n3_now - sel_sum)) if n3_now else 0.0

                        # d3-kernel columns (d3 · e_j == 0 ⇔ column j all zeros)
                        ker_cols = []
                        if d3_mat and n3_now:
                            for j in range(n3_now):
                                ker_cols.append(1 if all((int(d3_mat[i][j]) & 1) == 0 for i in range(n2_now)) else 0)

                        # ker-RED: strict fails and some failing column is ker
                        failing_cols = []
                        if R3s and n3_now:
                            for j in range(n3_now):
                                if any((int(R3s[i][j]) & 1) for i in range(len(R3s))):
                                    failing_cols.append(j)
                        ker_red = (strict_k3 is False) and any(ker_cols[j] == 1 for j in failing_cols)

                        # Policy receipt for AUTO (posed iff proj_meta.na is empty/false)
                        policy_receipt = _policy_receipt(
                            mode="projected:auto",
                            posed=posed_auto,
                            lane_policy="C bottom row",
                            lanes=L,
                            n3=n3_now
                        )

                        lane_density = (sel_sum / n3_now) if n3_now else 0.0

                        coverage_row = {
                            "written_at_utc": _svr_now_iso(),
                            "district_id": ib.get("district_id") or "DUNKNOWN",
                            "signature": sig,
                            "membership": {
                                "status":    mem["status"],
                                "proximity": mem["proximity"],
                                "sig_str":   mem["sig_str"],
                            },
                            "policy": policy_receipt,
                            "checks": {
                                "strict_k3":         strict_k3,
                                "projected_k3":      (proj_auto_k3 if posed_auto else None),
                                # FILE verdict not known yet at this point; log as None for this row.
                                "projected_k3_file": None,
                                "k2_strict":         strict_k2,
                                "k2_projected":      (proj_auto_k2 if posed_auto else None),
                            },
                            "overlay": {
                                "lane_density":            lane_density,
                                "selected_mismatch_rate":  selected_mismatch_rate,
                                "offrow_mismatch_rate":    offrow_mismatch_rate,
                                "ker_columns":             ker_cols,
                                "ker_red":                 bool(ker_red),
                                "contradictory_lane_rate": selected_mismatch_rate,
                            },
                            "na_reason_code": (proj_meta.get("reason") if not posed_auto else ""),
                        "e1": {
                            "prox_label": (rc.get("prox_label") or ss.get("prox_label") or "unknown"),
                            "strict_eq": bool(strict_k3),
                            "proj_eq": (bool(proj_auto_k3) if posed_auto else None),
                            "ker_RED": bool(ker_red),
                            "verdict_class": None,
                            "strict_nnz": int(len(failing_cols) if isinstance(failing_cols, list) else 0),
                            "proj_nnz": None,
                            "strict_failing_cols_sig256": "",
                            "proj_failing_cols_sig256": "",
                            "lanes_bitvec": list(L or []),
                            "lanes_sig256": "",
                            "lane_frac": float(lane_density),
                            "ker_lane_rate": float((sum(1 for j,b in enumerate(L or []) if b and ker_cols[j]==1)/max(1,sum(L or [0]))) if (L and ker_cols) else 0.0),
                            "stability_jaccard": None,
                            "projector_health": None,
                            "perturbation_target": "projected",
                            "perturbation_outcome": "PROBE_ONLY"
                        }
}

                        # COVERAGE_JSONL fallback if not declared globally
                        _COV_JSONL = COVERAGE_JSONL
                        # --- E1 post-processing (hashes, verdict class, projector health) ---
                        try:
                            import hashlib as _hashlib
                            _lb = bytes([1 if int(x) else 0 for x in (L or [])])
                            lanes_sig256 = 'sha256:' + _hashlib.sha256(_lb).hexdigest()
                            coverage_row['e1']['lanes_sig256'] = lanes_sig256
                            strict_fcols = (failing_cols if isinstance(failing_cols, list) else [])
                            sfc = ('sha256:' + _hashlib.sha256(','.join(map(str, sorted(strict_fcols))).encode('ascii')).hexdigest()) if strict_fcols else ''
                            coverage_row['e1']['strict_failing_cols_sig256'] = sfc
                            proj_nnz = None
                            proj_fcols = []
                            if posed_auto and R3s and isinstance(L, (list, tuple)) and n3_now:
                                P = [[1 if (i==j and int(L[j])==1) else 0 for j in range(n3_now)] for i in range(n3_now)]
                                R3p = _svr_mul(R3s, P)
                                for j in range(n3_now):
                                    if any((int(R3p[i][j]) & 1) for i in range(len(R3p))):
                                        proj_fcols.append(j)
                                proj_nnz = len(proj_fcols)
                                pfc = ('sha256:' + _hashlib.sha256(','.join(map(str, sorted(proj_fcols))).encode('ascii')).hexdigest()) if proj_fcols else ''
                                coverage_row['e1']['proj_failing_cols_sig256'] = pfc
                            coverage_row['e1']['proj_nnz'] = (None if proj_nnz is None else int(proj_nnz))
                            if (strict_k3 is True) and posed_auto and (proj_auto_k3 is False):
                                coverage_row['e1']['verdict_class'] = 'PROJECTOR_INTEGRITY_FAIL'
                            else:
                                if strict_k3 is True:
                                    vc = 'GREEN'
                                else:
                                    if not posed_auto:
                                        vc = 'RED_UNPOSED'
                                    else:
                                        is_kernel_only = all(ker_cols[j]==1 for j in strict_fcols) if strict_fcols else False
                                        if is_kernel_only:
                                            intersects = any((int(L[j])==1) for j in strict_fcols)
                                            vc = ('KER_EXPOSED' if intersects else 'KER_FILTERED')
                                        else:
                                            vc = 'RED_BOTH'
                                coverage_row['e1']['verdict_class'] = vc
                            prev = ss.get('_prev_lanes_bitvec')
                            if isinstance(prev, list) and prev and L:
                                inter = sum(1 for a,b in zip(prev, L) if int(a)==1 and int(b)==1)
                                union = sum(1 for a,b in zip(prev, L) if int(a)==1 or int(b)==1)
                                coverage_row['e1']['stability_jaccard'] = (inter/union if union else 1.0)
                            ss['_prev_lanes_bitvec'] = list(L or [])
                            tau_size = 0.30; tau_ker = 0.10; tau_stab = 0.60
                            lane_frac = float(coverage_row['e1']['lane_frac'])
                            ker_rate = float(coverage_row['e1']['ker_lane_rate'])
                            stab = coverage_row['e1']['stability_jaccard']
                            if not posed_auto:
                                coverage_row['e1']['projector_health'] = 'INVALID'
                            else:
                                ok_size = (lane_frac >= tau_size)
                                ok_ker  = (ker_rate <= tau_ker)
                                ok_stab = (True if stab is None else (stab >= tau_stab))
                                bads = [not ok_size, not ok_ker, not ok_stab]
                                coverage_row['e1']['projector_health'] = ('HEALTHY' if not any(bads) else 'WARN')
                        except Exception:
                            pass
                        # --- E1 post-processing (hashes, verdict class, projector health) ---

                        try:

                            import hashlib as _hashlib, json as _json

                            # lanes hash

                            _lb = bytes([1 if int(x) else 0 for x in (L or [])])

                            lanes_sig256 = 'sha256:' + _hashlib.sha256(_lb).hexdigest()

                            coverage_row['e1']['lanes_sig256'] = lanes_sig256

                            # failing cols

                            strict_fcols = (failing_cols if isinstance(failing_cols, list) else [])

                            _sblob = _json.dumps(sorted([int(x) for x in strict_fcols])).encode('utf-8') if strict_fcols else b''

                            coverage_row['e1']['strict_failing_cols_sig256'] = ('sha256:' + _hashlib.sha256(_sblob).hexdigest()) if _sblob else ''

                            # projected failing cols by masking strict residual columns with lanes

                            proj_fcols = []

                            proj_nnz = None

                            if posed_auto and isinstance(R3s, list) and n3_now:

                                for j in range(n3_now):

                                    if (L and j < len(L) and int(L[j])==1):

                                        if any((int(R3s[i][j]) & 1) for i in range(len(R3s))):

                                            proj_fcols.append(j)

                                proj_nnz = len(proj_fcols)

                                _pblob = _json.dumps(sorted([int(x) for x in proj_fcols])).encode('utf-8') if proj_fcols else b''

                                coverage_row['e1']['proj_failing_cols_sig256'] = ('sha256:' + _hashlib.sha256(_pblob).hexdigest()) if _pblob else ''

                            coverage_row['e1']['proj_nnz'] = (None if proj_nnz is None else int(proj_nnz))

                            # stability vs previous lanes

                            prev = ss.get('_prev_lanes_bitvec')

                            if isinstance(prev, list) and prev and L:

                                inter = sum(1 for a,b in zip(prev, L) if int(a)==1 and int(b)==1)

                                union = sum(1 for a,b in zip(prev, L) if int(a)==1 or int(b)==1)

                                coverage_row['e1']['stability_jaccard'] = (inter/union if union else 1.0)

                            ss['_prev_lanes_bitvec'] = list(L or [])

                            # projector health

                            tau_size, tau_ker, tau_stab = 0.30, 0.10, 0.60

                            lane_frac = float(coverage_row['e1']['lane_frac'])

                            ker_rate = float(coverage_row['e1']['ker_lane_rate'])

                            stab = coverage_row['e1']['stability_jaccard']

                            if not posed_auto:

                                coverage_row['e1']['projector_health'] = 'INVALID'

                            else:

                                ok_size = (lane_frac >= tau_size)

                                ok_ker  = (ker_rate <= tau_ker)

                                ok_stab = (True if stab is None else (stab >= tau_stab))

                                coverage_row['e1']['projector_health'] = ('HEALTHY' if (ok_size and ok_ker and ok_stab) else ('WARN' if (ok_size and ok_ker) else 'INVALID'))

                            # verdict class per A4

                            strict_eq = bool(strict_k3)

                            proj_eq = (bool(proj_auto_k3) if posed_auto else None)

                            ker_RED = bool(ker_red)

                            lanes_set = { j for j,b in enumerate(L or []) if b }

                            supp_set  = set(int(x) for x in strict_fcols)

                            if strict_eq:

                                coverage_row['e1']['verdict_class'] = ('PROJECTOR_INTEGRITY_FAIL' if (posed_auto and (proj_eq is False)) else 'GREEN')

                            else:

                                if (not posed_auto) or (coverage_row['e1']['projector_health']=='INVALID'):

                                    coverage_row['e1']['verdict_class'] = 'RED_UNPOSED'

                                elif ker_RED:

                                    coverage_row['e1']['verdict_class'] = ('KER-EXPOSED' if any((j in lanes_set) for j in strict_fcols) else 'KER-FILTERED')

                                else:

                                    coverage_row['e1']['verdict_class'] = 'RED_BOTH'

                        except Exception:

                            pass

                        _atomic_append_jsonl(_COV_JSONL, coverage_row)
                        st.caption(f"Coverage row appended · σ={mem.get('sig_str','')} · {mem.get('status')} / {mem.get('proximity')}")
                    except Exception as _c1e:
                        st.warning(f"C1 coverage row not appended: {_c1e}")

                    # ----------------- WRITE CERTS -----------------

                    # Strict cert (+ full-R3 selected mismatch)
                    strict_cert = _svr_cert_common(ib, rc, "strict")
                    strict_cert["policy"] = _policy_receipt(mode="strict", posed=True, n3=n3)
                    strict_cert["witness"] = _witness_pack(bH, bCI, lanes=None)
                    strict_cert["results"] = {
                        "out": strict_out,
                        "selected_cols": sel_all,
                        "mismatch_cols_selected": _svr_mismatch_cols_from_R3(R3s, sel_all),
                        "residual_tag_selected": _svr_residual_tag_from_R3(R3s, sel_all),
                        "k2": strict_k2,
                        "na_reason_code": (strict_out.get("na_reason_code") if isinstance(strict_out, dict) else None),
                    }
                    _svr_apply_sig8(strict_cert, embed_sig_auto)
                    p_strict = _svr_write_cert_in_bundle(
                        _bundle_dir, _svr_bundle_fname("strict", district_id, sig8), strict_cert
                    )

                    # Projected(AUTO) cert
                    p_cert = _svr_cert_common(ib, rc, "projected(columns@k=3,auto)")
                    p_cert["policy"] = _policy_receipt(mode="projected:auto", posed=bool(posed_auto), lane_policy="C bottom row", lanes=(list(lanes) if isinstance(lanes,(list,tuple)) else []), n3=n3)
                    if not posed_auto:
                        p_cert["witness"] = _witness_pack(bH, bCI, lanes=None)
                        p_cert["results"] = {
                            "out": {"2": {"eq": None}, "3": {"eq": None}},
                            "na_reason_code": proj_meta.get("reason", ""),
                            "lane_policy": "C bottom row",
                            "lanes": (list(lanes) if isinstance(lanes,(list,tuple)) else []),
                            "selected_cols": (list(lanes) if isinstance(lanes,(list,tuple)) else []),
                            "mismatch_cols_selected": [],
                            "residual_tag_selected": "",
                            "k2": None,
                        }
                    else:
                        p_cert["witness"] = _witness_pack(bH, bCI, lanes=lanes)
                        p_cert["results"] = {
                            "out": projected_out,
                            "lane_policy": "C bottom row",
                            "lanes": (list(lanes) if isinstance(lanes,(list,tuple)) else []),
                            "selected_cols": (list(lanes) if isinstance(lanes,(list,tuple)) else []),
                            "mismatch_cols_selected": _svr_mismatch_cols_from_R3(R3s, (list(lanes) if isinstance(lanes,(list,tuple)) else [])),
                            "residual_tag_selected": _svr_residual_tag_from_R3(R3s, (list(lanes) if isinstance(lanes,(list,tuple)) else [])),
                            "k2": proj_auto_k2,
                        }
                    _svr_apply_sig8(p_cert, embed_sig_auto)
                    p_proj_auto = _svr_write_cert_in_bundle(
                        _bundle_dir, _svr_bundle_fname("projected_auto", district_id, sig8), p_cert
                    )

                    # Freezer (AUTO → FILE), always attempt
                    file_na_reason = None
                    projector_hash = ""
                    p_proj_file = None
                    p_cert_file = None
                    proj_file_k3 = None
                    proj_file_k2 = None

                    if not posed_auto:
                        # N/A freezer
                        r = str(proj_meta.get("reason") or "")
                        file_na_reason = (
                            "FREEZER_C3_NOT_SQUARE" if r == "AUTO_REQUIRES_SQUARE_C3"
                            else "FREEZER_ZERO_LANE_PROJECTOR" if r == "ZERO_LANE_PROJECTOR"
                            else "FREEZER_BAD_SHAPE"
                        )
                        freezer_cert = _svr_cert_common(ib, rc, "projector_freezer")
                        freezer_cert["freezer"] = {
                            "status": "N/A",
                            "lanes": (list(lanes) if isinstance(lanes,(list,tuple)) else []),
                            "projector_hash": None,
                            "na_reason_code": file_na_reason,
                        }
                        _svr_apply_sig8(freezer_cert, embed_sig_auto)
                        p_freezer = _svr_write_cert_in_bundle(
                            _bundle_dir, _svr_bundle_fname("freezer", district_id, sig8), freezer_cert
                        )
                    else:
                        # Build diagonal Π from lanes
                        L_used = (list(lanes) if isinstance(lanes, (list, tuple)) else [])
                        P = [[1 if (i == j and int(L_used[j]) == 1) else 0 for j in range(n3)] for i in range(n3)]
                        # FILE decision (k3): R3s @ P == 0?
                        R3pF = _svr_mul(R3s, P) if (R3s and P) else []
                        eq_file_k3 = _svr_is_zero(R3pF) if R3pF else None
                        proj_file_k3 = bool(eq_file_k3) if eq_file_k3 is not None else None
                        proj_file_k2 = proj_auto_k2  # same value for k2 here

                        # If AUTO↔FILE disagree, log it
                        try:
                            if (proj_auto_k3 is not None) and (proj_file_k3 is not None) and (proj_auto_k3 != proj_file_k3):
                                _log_freezer_mismatch(
                                    fixture_id=rc.get("run_id",""),
                                    auto_lanes=L_used,
                                    file_lanes=L_used,
                                    verdict_auto=proj_auto_k3,
                                    verdict_file=proj_file_k3,
                                )
                                st.warning("⚠ AUTO↔FILE mismatch on k3 (determinism breach). Check lane policy & Π file.")
                        except Exception:
                            pass

                        # FILE projector hash: sha256(JSON({blocks:{3:P}}))
                        _proj_blob = _json.dumps({"blocks": {"3": P}}, separators=(",", ":"), sort_keys=True).encode("ascii")
                        projector_hash = "sha256:" + _hashlib.sha256(_proj_blob).hexdigest()

                        # Write projected(FILE) cert
                        p_cert_file = _svr_cert_common(ib, rc, "projected(columns@k=3,file)")
                        p_cert_file["results"] = {
                            "out": {"2": {"eq": proj_file_k2}, "3": {"eq": proj_file_k3}},
                            "projector_hash": projector_hash,
                            "lanes": list(L_used),
                        }
                        p_cert_file["policy"] = _policy_receipt(
                            mode="projected:file",
                            posed=True,
                            lane_policy="file projector",
                            lanes=list(L_used),
                            projector_source="file",
                            projector_hash=projector_hash,
                            n3=n3
                        )
                        _svr_apply_sig8(p_cert_file, embed_sig_auto)
                        p_proj_file = _svr_write_cert_in_bundle(
                            _bundle_dir,
                            _svr_bundle_fname("projected_file", district_id, sig8),
                            p_cert_file,
                        )

                        # Freezer status
                        freezer_status = "OK"
                        if (proj_auto_k3 is True and proj_file_k3 is False) or (proj_auto_k3 is False and proj_file_k3 is True):
                            freezer_status = "ERROR"
                            file_na_reason = "FREEZER_ASSERT_MISMATCH"
                        freezer_cert = _svr_cert_common(ib, rc, "projector_freezer")
                        freezer_cert["freezer"] = {
                            "status": freezer_status,
                            "lanes": list(L_used),
                            "projector_hash": projector_hash,
                            "na_reason_code": (file_na_reason or ""),
                        }
                        _svr_apply_sig8(freezer_cert, embed_sig_auto)
                        p_freezer = _svr_write_cert_in_bundle(
                            _bundle_dir, _svr_bundle_fname("freezer", district_id, sig8), freezer_cert
                        )

                    # ----------------- A/B WRITES (outside freezer branch) -----------------

                    # A/B (AUTO)
                    ab_auto = _svr_cert_common(ib, rc, "A/B")
                    ab_auto["ab_pair"] = {
                        "pair_tag": "strict__VS__projected(columns@k=3,auto)",
                        "embed": embed_auto,
                        "embed_sig": embed_sig_auto,
                        "pair_vec": {
                            "k3": [strict_k3, proj_auto_k3],
                            "k2": [strict_k2, proj_auto_k2],
                        },
                        "strict_cert": {"path": str(p_strict), "hash": strict_cert["integrity"]["content_hash"]},
                        "projected_cert": {"path": str(p_proj_auto), "hash": p_cert["integrity"]["content_hash"]},
                    }
                    _svr_apply_sig8(ab_auto, embed_sig_auto)
                    p_ab_auto = _svr_write_cert_in_bundle(
                        _bundle_dir, _svr_bundle_fname("ab_auto", district_id, sig8), ab_auto
                    )
                    # Pin AUTO (include inputs_sig_5 for legacy readers)
                    st.session_state["ab_pin_auto"] = {
                        "state": "pinned",
                        "payload": {
                            "embed_sig": embed_sig_auto,
                            "policy_tag": "strict__VS__projected(columns@k=3,auto)",
                            "inputs_sig_5": list((ab_auto["ab_pair"]["embed"] or {}).get("inputs_sig_5") or []),
                        },
                        "refreshed_at": _svr_now_iso(),
                    }
                    st.session_state["ab_pin"] = st.session_state["ab_pin_auto"]

                    # A/B (FILE)
                    embed_file, embed_sig_file = _svr_build_embed(
                        ib,
                        policy="strict__VS__projected(columns@k=3,file)",
                        projector_hash=(projector_hash if not file_na_reason else None),
                        na_reason=(file_na_reason if file_na_reason else None),
                    )
                    ab_file = _svr_cert_common(ib, rc, "A/B")
                    ab_file["ab_pair"] = {
                        "pair_tag": "strict__VS__projected(columns@k=3,file)",
                        "embed": embed_file,
                        "embed_sig": embed_sig_file,
                        "pair_vec": {
                            "k3": [strict_k3, proj_file_k3],
                            "k2": [strict_k2, proj_file_k2],
                        },
                        "strict_cert": {"path": str(p_strict), "hash": strict_cert["integrity"]["content_hash"]},
                        "projected_cert": ({"path": str(p_proj_file), "hash": p_cert_file["integrity"]["content_hash"]} if p_proj_file else None),
                        "na_reason_code": (file_na_reason or ""),
                    }
                    _svr_apply_sig8(ab_file, embed_sig_file)
                    p_ab_file = _svr_write_cert_in_bundle(
                        _bundle_dir, _svr_bundle_fname("ab_file", district_id, sig8), ab_file
                    )
                    # Pin FILE
                    st.session_state["ab_pin_file"] = {
                        "state": "pinned",
                        "payload": {
                            "embed_sig": embed_sig_file,
                            "policy_tag": "strict__VS__projected(columns@k=3,file)",
                            "inputs_sig_5": list((ab_file["ab_pair"]["embed"] or {}).get("inputs_sig_5") or []),
                        },
                        "refreshed_at": _svr_now_iso(),
                    }

                    # 5(+1) bundle index
                    try:
                        fnames = [Path(p_strict).name, Path(p_proj_auto).name, Path(p_ab_auto).name, Path(p_freezer).name, Path(p_ab_file).name]
                        if p_proj_file:
                            fnames.append(Path(p_proj_file).name)
                        tstamp = lambda x: (x or {}).get("written_at_utc","")
                        timestamps = [
                            tstamp(strict_cert), tstamp(p_cert), tstamp(ab_auto), tstamp(freezer_cert), tstamp(ab_file)
                        ] + ([tstamp(p_cert_file)] if p_proj_file else [])
                        _svr_write_cert_in_bundle(
                            _bundle_dir, "bundle.json",
                            {
                                "run_id": rc.get("run_id",""),
                                "sig8": sig8,
                                "district_id": district_id,
                                "filenames": fnames,
                                "timestamps": timestamps,
                                "counts": {"written": (5 if not p_proj_file else 6)},
                            }
                        )
                    except Exception as _be:
                        st.warning(f"bundle.json not written: {_be}")

                    # remember paths for sanity panel
                    st.session_state["last_bundle_dir"] = str(_bundle_dir)
                    st.session_state["last_ab_auto_path"] = str(p_ab_auto)
                    st.session_state["last_ab_file_path"] = str(p_ab_file)

                    # Header line + quick lap/A/B readout (k2/k3)
                    rc["policy_tag"] = "strict"  # stable anchor
                    st.session_state["run_ctx"] = rc
                    st.session_state["last_run_header"] = _svr_header_line(
                        ib, sig8, (projector_hash if not file_na_reason else None), "strict", rc.get("run_id", "")
                    )

                    # Eyeball summary (k2/k3)
                    def _fmt(x): return "T" if x is True else ("F" if x is False else "N/A")
                    hdr = (
                        f"**k3**: strict={_fmt(strict_k3)} · proj_auto={_fmt(proj_auto_k3)} · proj_file={_fmt(proj_file_k3)}  \n"
                        f"**A/B(auto)**=( {_fmt(strict_k3)} , {_fmt(proj_auto_k3)} )    ·    "
                        f"**A/B(file)**=( {_fmt(strict_k3)} , {_fmt(proj_file_k3)} ){('  ·  reason: '+file_na_reason) if file_na_reason else ''}  \n"
                        f"**k2**: strict={_fmt(strict_k2)} · proj_auto={_fmt(proj_auto_k2)} · proj_file={_fmt(proj_file_k2)}"
                    )
                    st.markdown(hdr)

                    # Mini A/B SanBity (from latest bundle)
                    def _exp_embed_sig(path: str):
                        try:
                            j = _json.loads(Path(path).read_text(encoding="utf-8"))
                            return str(((j or {}).get("ab_pair") or {}).get("embed_sig") or ""), (j or {})
                        except Exception:
                            return "", {}

                    with st.container(border=True):
                        st.caption("Mini A/B Sanity (from latest bundle)")
                        bdir = st.session_state.get("last_bundle_dir") or ""
                        pab_auto = st.session_state.get("last_ab_auto_path") or ""
                        pab_file = st.session_state.get("last_ab_file_path") or ""

                        cols = st.columns(3)
                        cols[0].markdown(f"**Bundle:** `{bdir or '—'}`")
                        cols[1].markdown(f"**A/B(auto):** `{(Path(pab_auto).name if pab_auto else '—')}`")
                        cols[2].markdown(f"**A/B(file):** `{(Path(pab_file).name if pab_file else '—')}`")

                        exp_auto, j_auto = _exp_embed_sig(pab_auto) if pab_auto else ("", {})
                        exp_file, j_file = _exp_embed_sig(pab_file) if pab_file else ("", {})

                        # Pick pins safely (prefer specific pins, fallback to legacy)
                        pin_auto = st.session_state.get("ab_pin_auto") or st.session_state.get("ab_pin") or {}
                        pin_file = st.session_state.get("ab_pin_file") or st.session_state.get("ab_pin") or {}

                        fresh_auto, reason_auto = _ab_is_fresh_now(pin_auto, expected_embed_sig=(exp_auto or ""))
                        fresh_file, reason_file = _ab_is_fresh_now(pin_file, expected_embed_sig=(exp_file or ""))

                        ta = "✅ Fresh" if fresh_auto else f"⚠️ Stale ({reason_auto})"
                        tf = "✅ Fresh" if fresh_file else f"⚠️ Stale ({reason_file})"
                        st.write(f"**A/B(auto)** → {ta}  ·  **A/B(file)** → {tf}")

                        # Quick vectors at a glance
                        vec_auto = ((j_auto.get("ab_pair") or {}).get("pair_vec") or {})
                        vec_file = ((j_file.get("ab_pair") or {}).get("pair_vec") or {})
                        rows = []
                        if vec_auto:
                            rows.append({"pair": "A/B(auto)", "k3": str(vec_auto.get("k3")), "k2": str(vec_auto.get("k2")), "embed_sig8": (exp_auto or "")[:8]})
                        if vec_file:
                            rows.append({"pair": "A/B(file)", "k3": str(vec_file.get("k3")), "k2": str(vec_file.get("k2")), "embed_sig8": (exp_file or "")[:8]})
                        if rows:
                            try:
                                import pandas as _pd
                                st.dataframe(_pd.DataFrame(rows), use_container_width=True)
                            except Exception:
                                st.table(rows)

                except Exception as e:
                    st.error(f"Solver run failed: {e}")
                finally:
                    ss['_solver_one_button_active'] = False
                    ss['_solver_busy'] = False
        # --- End run button ---






    
                                  
                    
# ─────────────────────────── Solver: one-press pipeline ───────────────────────────
def _svr_run():
    """
    strict → projected(auto) → freezer → projected(file) + A/Bs → bundle
    Writes to logs/certs/<district_id>/<sig8>/ and updates session_state.
    """
    import json as _json, hashlib as _hashlib, uuid as _uuid
    from pathlib import Path
    ss = st.session_state

    # Resolve inputs & freeze SSOT
    pf = _svr_resolve_all_to_paths()   # {"B": (path, blocks), "C": ..., "H": ..., "U": ...}
    (pB, bB), (pC, bC), (pH, bH), (pU, bU) = pf["B"], pf["C"], pf["H"], pf["U"]
    ib_rc = _svr_freeze_ssot(pf)
    if isinstance(ib_rc, tuple):
        ib = ib_rc[0] or {}
        rc = ib_rc[1] if (len(ib_rc) > 1 and isinstance(ib_rc[1], dict)) else {}
    else:
        ib = ib_rc or {}
        rc = {}
    rc.setdefault("run_id", str(_uuid.uuid4()))
    rc.setdefault("fixture_nonce", int(ss.get("fixture_nonce", 0) or 1))

    # Dimensions & district
    H2 = bH.get("2") or []
    d3 = bB.get("3") or []
    C3 = bC.get("3") or []
    n3 = len(C3[0]) if (C3 and C3[0]) else 0
    district_id = str(ib.get("district_id") or "DUNKNOWN")

    # Residuals R3 = H2·d3 ⊕ (C3 ⊕ I3)
    I3 = _svr_eye(len(C3)) if (C3 and len(C3) == len(C3[0])) else []
    R3s = _svr_xor(_svr_mul(H2, d3), _svr_xor(C3, I3)) if (H2 and d3 and C3 and I3) else []

    # Strict & Projected(auto)
    strict_out = _svr_strict_from_blocks(bH, bB, bC)
    proj_meta, lanes, proj_out = _svr_projected_auto_from_blocks(bH, bB, bC)

    # Build embed for AUTO A/B; derive sig8 + bundle dir
    na_reason = (proj_meta.get("reason") if (proj_meta and proj_meta.get("na")) else None)
    embed_auto, embed_sig_auto = _svr_build_embed(
        ib, "strict__VS__projected(columns@k=3,auto)",
        lanes=(lanes if lanes else None),
        na_reason=na_reason,
    )
    sig8 = (embed_sig_auto or "")[:8]
    _bundle_dir = _svr_bundle_dir(district_id, sig8)

    # ---------- Write STRICT ----------
    strict_cert = _svr_cert_common(ib, rc, "strict")
    strict_cert["policy"] = _policy_receipt(mode="strict", posed=True, n3=n3)
    strict_cert["results"] = {"out": dict(strict_out or {})}
    _svr_apply_sig8(strict_cert, embed_sig_auto)
    p_strict = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("strict", district_id, sig8), strict_cert)

    # ---------- Write PROJECTED(AUTO) ----------
    p_cert = _svr_cert_common(ib, rc, "projected(columns@k=3,auto)")
    p_cert["policy"] = _policy_receipt(
        mode="projected:auto", posed=(not bool(proj_meta.get("na"))) if proj_meta else True,
        lane_policy="C bottom row", lanes=list(lanes or []), projector_source="auto", n3=n3
    )
    p_cert["results"] = {"out": dict(proj_out or {}), "lanes": list(lanes or [])}
    _svr_apply_sig8(p_cert, embed_sig_auto)
    p_proj_auto = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("projected_auto", district_id, sig8), p_cert)

    # ---------- Write A/B (AUTO) ----------
    ab_auto = _svr_cert_common(ib, rc, "strict__VS__projected(columns@k=3,auto)")
    ab_auto["ab_pair"] = {
        "embed": (embed_auto or {}),
        "embed_sig": str(embed_sig_auto or ""),
        "left":  {"policy": "strict", "out": strict_out},
        "right": {"policy": "projected(columns@k=3,auto)", "lanes": list(lanes or []), "out": proj_out},
    }
    _svr_apply_sig8(ab_auto, embed_sig_auto)
    p_ab_auto = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("ab_auto", district_id, sig8), ab_auto)

    # ---------- FILE projector from lanes (if possible) ----------
    projector_hash = None
    proj_file_k3 = None
    proj_file_k2 = None
    p_proj_file = None

    if lanes and n3 and R3s:
        L = [int(x) & 1 for x in lanes]
        # Π = diag(L)
        P = [[1 if (i == j and L[j] == 1) else 0 for j in range(n3)] for i in range(n3)]
        R3pF = _svr_mul(R3s, P) if (R3s and P) else []
        eq_file_k3 = _svr_is_zero(R3pF) if R3pF else None
        proj_file_k3 = bool(eq_file_k3) if eq_file_k3 is not None else None
        proj_file_k2 = (proj_out or {}).get("2", {}).get("eq", None)

        _proj_blob = _json.dumps({"blocks": {"3": P}}, separators=(",", ":"), sort_keys=True).encode("ascii")
        projector_hash = "sha256:" + _hashlib.sha256(_proj_blob).hexdigest()

        # projected(FILE)
        p_cert_file = _svr_cert_common(ib, rc, "projected(columns@k=3,file)")
        p_cert_file["policy"] = _policy_receipt(
            mode="projected:file", posed=True, lane_policy="file projector",
            lanes=list(L), projector_source="file", projector_hash=projector_hash, n3=n3
        )
        p_cert_file["results"] = {"out": {"2": {"eq": proj_file_k2}, "3": {"eq": proj_file_k3}},
                                  "projector_hash": projector_hash, "lanes": list(L)}
        _svr_apply_sig8(p_cert_file, embed_sig_auto)
        p_proj_file = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("projected_file", district_id, sig8), p_cert_file)

        # A/B (FILE)
        embed_file, embed_sig_file = _svr_build_embed(
            ib, "strict__VS__projected(columns@k=3,file)", lanes=list(L), projector_hash=projector_hash
        )
        ab_file = _svr_cert_common(ib, rc, "strict__VS__projected(columns@k=3,file)")
        ab_file["ab_pair"] = {
            "embed": (embed_file or {}), "embed_sig": str(embed_sig_file or ""),
            "left": {"policy": "strict", "out": strict_out},
            "right": {"policy": "projected(columns@k=3,file)", "lanes": list(L),
                      "out": {"2": {"eq": proj_file_k2}, "3": {"eq": proj_file_k3}}},
        }
        _svr_apply_sig8(ab_file, embed_sig_file)
        p_ab_file = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("ab_file", district_id, sig8), ab_file)

        # Freezer
        freezer_status = "OK"
        if ((proj_out or {}).get("3", {}).get("eq") is True and proj_file_k3 is False) or \
           ((proj_out or {}).get("3", {}).get("eq") is False and proj_file_k3 is True):
            freezer_status = "ERROR"
        freezer_cert = _svr_cert_common(ib, rc, "projector_freezer")
        freezer_cert["freezer"] = {
            "status": freezer_status, "lanes": list(L),
            "projector_hash": projector_hash, "na_reason_code": None if freezer_status=="OK" else "FREEZER_ASSERT_MISMATCH",
        }
        _svr_apply_sig8(freezer_cert, embed_sig_auto)
        p_freezer = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("freezer", district_id, sig8), freezer_cert)
    else:
        # Freezer N/A + minimal AB(file)
        L = list(lanes or [])
        na_reason = "FREEZER_BAD_SHAPE" if not R3s else ("FREEZER_ZERO_LANE_PROJECTOR" if not any(L) else "FREEZER_C3_NOT_SQUARE")
        freezer_cert = _svr_cert_common(ib, rc, "projector_freezer")
        freezer_cert["freezer"] = {"status": "N/A", "lanes": list(L), "projector_hash": None, "na_reason_code": na_reason}
        _svr_apply_sig8(freezer_cert, embed_sig_auto)
        p_freezer = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("freezer", district_id, sig8), freezer_cert)

        ab_file = _svr_cert_common(ib, rc, "strict__VS__projected(columns@k=3,file)")
        ab_file["ab_pair"] = {
            "embed": {"policy": "strict__VS__projected(columns@k=3,file)",
                      "projection_context": {"na_reason_code": na_reason, "lanes": list(L)}},
            "embed_sig": "",
            "left": {"policy": "strict", "out": strict_out},
            "right": {"policy": "projected(columns@k=3,file)", "lanes": list(L), "out": {}},
        }
        _svr_apply_sig8(ab_file, embed_sig_auto)
        p_ab_file = _svr_write_cert_in_bundle(_bundle_dir, _svr_bundle_fname("ab_file", district_id, sig8), ab_file)

    # ---------- Bundle index ----------
    try:
        names = ["strict","projected_auto","ab_auto","freezer","ab_file","projected_file"]
        fnames = []
        for k in names:
            p = _bundle_dir / _svr_bundle_fname(k, district_id, sig8)
            if p.exists(): fnames.append(p.name)
        _svr_write_cert_in_bundle(_bundle_dir, "bundle.json", {
            "run_id": rc.get("run_id",""), "sig8": sig8, "district_id": district_id,
            "filenames": fnames, "counts": {"written": len(fnames)}
        })
    except Exception:
        pass

    # Publish session anchors
    ss["last_bundle_dir"]   = str(_svr_bundle_dir(district_id, sig8))
    ss["last_ab_auto_path"] = str(_svr_bundle_dir(district_id, sig8) / _svr_bundle_fname("ab_auto", district_id, sig8))
    ss["last_ab_file_path"] = str(_svr_bundle_dir(district_id, sig8) / _svr_bundle_fname("ab_file", district_id, sig8))
    ss["last_solver_result"] = {"count": len(fnames) if 'fnames' in locals() else 0}
# ───────────────────────────────────────────────────────────────────────────






















# ============================== Cert & Provenance (lean, SSOT-aligned) ==============================
with safe_expander("Cert & provenance (read-only; solver writes bundles)", expanded=True):
    import os, json, hashlib, platform, time
    from pathlib import Path
    from datetime import datetime

    # ---------- constants ----------
    SCHEMA_VERSION = globals().get("SCHEMA_VERSION", "2.0.0")
    APP_VERSION    = globals().get("APP_VERSION",    "v0.1-core")

    class REASON:
        WROTE_CERT                      = "WROTE_CERT"
        SKIP_NO_MATERIAL_CHANGE         = "SKIP_NO_MATERIAL_CHANGE"
        SKIP_INPUTS_INCOMPLETE          = "SKIP_INPUTS_INCOMPLETE"
        SKIP_FILE_PI_INVALID            = "SKIP_FILE_PI_INVALID"
        SKIP_SSOT_STALE                 = "SKIP_SSOT_STALE"
        AB_EMBEDDED                     = "AB_EMBEDDED"
        AB_NONE                         = "AB_NONE"
        AB_STALE_INPUTS_SIG             = "AB_STALE_INPUTS_SIG"
        AB_STALE_POLICY                 = "AB_STALE_POLICY"
        AB_STALE_PROJECTOR_HASH         = "AB_STALE_PROJECTOR_HASH"
        SKIP_AB_TICKET_ALREADY_WRITTEN  = "SKIP_AB_TICKET_ALREADY_WRITTEN"

    # ---------- pure helpers ----------
    def _utc_now_z() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _sha256_hex(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _short(h: str, n: int = 8) -> str:
        return (h or "")[:n]

    def _sanitize(s: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "-_@=,") else "_" for ch in (s or ""))[:80]

    def _canon_policy(label_raw: str) -> str:
        t = (label_raw or "").lower()
        if "strict" in t: return "strict"
        if "projected" in t and "file" in t: return "projected:file"
        return "projected:auto"

    def _python_version_str() -> str:
        return f"python-{platform.python_version()}"

    def _write_json_atomic(path: Path, obj: dict) -> None:
        """
        single-writer discipline: only write when the solver one-button sets the flag.
        outside that path, this becomes a no-op (read-only panel).
        """
        try:
            if not st.session_state.get('_solver_one_button_active', False):
                return  # read-only outside solver
        except Exception:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tmp  = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(blob); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)

    def _append_witness(line: dict) -> None:
        try:
            p = Path("logs") / "witnesses.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, separators=(",", ":")) + "\n")
        except Exception:
            pass

    # ---------- SSOT snapshot (single read; no recompute) ----------
    ss  = st.session_state
    rc  = dict(ss.get("run_ctx") or {})
    out = dict(ss.get("overlap_out") or {})
    try:
        H_obj = ss.get("overlap_H") or io.parse_cmap({"blocks": {}})
    except Exception:
        H_obj = io.parse_cmap({"blocks": {}})
    try:
        C_obj = ss.get("overlap_C") or io.parse_cmap({"blocks": {}})
    except Exception:
        C_obj = io.parse_cmap({"blocks": {}})
    ib    = dict(ss.get("_inputs_block") or {})

    # Freshness (never blocks UI; only influences write decision)
    stale = False
    if "ssot_is_stale" in globals() and callable(globals()["ssot_is_stale"]):
        try: stale = bool(ssot_is_stale())
        except Exception: stale = False

    # derive inputs_sig safely from SSOT (U optional)
    h  = (ib.get("hashes") or {})
    inputs_sig = [
        str(h.get("boundaries_hash") or h.get("B_hash") or ""),
        str(h.get("C_hash") or ""),
        str(h.get("H_hash") or ""),
        str(h.get("U_hash") or ""),
        str(h.get("shapes_hash") or h.get("S_hash") or ""),
    ]
    inputs_complete = all(isinstance(x, str) and x for x in (inputs_sig[0], inputs_sig[1], inputs_sig[2], inputs_sig[4]))

    # policy + pass_vec (self-contained)
    policy_raw   = rc.get("policy_tag") or ss.get("overlap_policy_label") or "strict"
    policy_canon = _canon_policy(policy_raw)
    proj_hash    = rc.get("projector_hash", "") if policy_canon == "projected:file" else ""
    eq2 = bool(((out.get("2") or {}).get("eq", False)))
    eq3 = bool(((out.get("3") or {}).get("eq", False)))
    pass_vec = (eq2, eq3)

    # file Π validity (soft gate)
    file_pi_valid   = bool(ss.get("file_pi_valid", True))
    file_pi_reasons = list(ss.get("file_pi_reasons", []) or [])

    # A/B ticket / pin status (freshness checked against embed_sig when available)
    ab_pin = dict(ss.get("ab_pin") or {"state":"idle","payload":None,"consumed":False})
    is_ab_pinned = (ab_pin.get("state") == "pinned")

    def _current_embed_sig_fallback() -> str:
        # prefer helper if present
        if "_ab_embed_sig" in globals() and callable(globals()["_ab_embed_sig"]):
            try: return str(_ab_embed_sig())
            except Exception: pass
        # fallback: 5-hash + policy label + projector hash
        return "|".join([*inputs_sig, policy_canon, proj_hash])

    ab_status = REASON.AB_NONE
    ab_fresh  = False
    if is_ab_pinned:
        payload = ab_pin.get("payload") or {}
        pin_sig = str(payload.get("embed_sig") or (payload.get("payload") or {}).get("embed_sig", ""))
        cur_sig = _current_embed_sig_fallback()
        if pin_sig and (pin_sig != cur_sig):
            ab_status = REASON.AB_STALE_INPUTS_SIG
        elif _canon_policy(payload.get("policy_tag","")) != policy_canon:
            ab_status = REASON.AB_STALE_POLICY
        elif policy_canon == "projected:file" and (payload.get("projected",{}) or {}).get("projector_hash","") != proj_hash:
            ab_status = REASON.AB_STALE_PROJECTOR_HASH
        else:
            ab_fresh  = True
            ab_status = REASON.AB_EMBEDDED

    # A/B one-shot ticket override for stale SSOT
    ab_ticket_pending      = ss.get("_ab_ticket_pending")
    last_ab_ticket_written = ss.get("_last_ab_ticket_written")
    ticket_required = bool(is_ab_pinned and ab_fresh and (ab_ticket_pending is not None) and (ab_ticket_pending != last_ab_ticket_written))

    # write arming comes from the solver button
    write_armed  = bool(ss.get("write_armed", False))

    # dedupe key for single-write gate
    write_key = (tuple(inputs_sig), policy_canon, pass_vec, proj_hash)
    last_key  = ss.get("_last_cert_write_key")

    # ---------- header strip (compact) ----------
    c1, c2, c3, c4 = st.columns([2,2,3,3])
    with c1:
        if inputs_complete:
            st.success("Inputs OK")
            st.caption(f"b/C/H/U/S = {', '.join(_short(x) or '∅' for x in inputs_sig)}")
        else:
            missing = [k for k,v in zip(("b","C","H","S"), [inputs_sig[0],inputs_sig[1],inputs_sig[2],inputs_sig[4]]) if not v]
            st.error(f"Inputs MISSING · {','.join(missing)}")
    with c2:
        if policy_canon == "projected:file":
            if file_pi_valid:
                st.success("Mode: FILE · Π VALID")
            else:
                st.error("Mode: FILE · Π INVALID")
                if file_pi_reasons: st.caption(" · ".join(file_pi_reasons[:3]))
        elif policy_canon == "projected:auto":
            st.info("Mode: AUTO")
        else:
            st.info("Mode: STRICT")
    with c3:
        if is_ab_pinned:
            st.caption(f"A/B: {'Fresh' if ab_fresh else 'Stale'}{(' · '+ab_status) if not ab_fresh else ''}")
        else:
            st.caption("A/B: None")
    with c4:
        st.caption(
            "Write: {}"
            .format("Armed" if write_armed else "Idle")
        )
        if write_armed and last_key:
            li, lp, lv, lpj = last_key
            ci, cp, cv, cpj = write_key
            # tiny delta hint
            delta_bits = []
            if lp != cp: delta_bits.append("policy")
            if lv != cv: delta_bits.append("k")
            if lpj != cpj and ("file" in (lp+cp)): delta_bits.append("Π")
            if li != ci: delta_bits.append("hashes")
            if delta_bits:
                st.caption("Δ " + ",".join(delta_bits))

    # ---------- decisions (N/A + reason; never st.stop the app) ----------
    write_enabled = True

    if not inputs_complete:
        write_enabled = False
        _append_witness({
            "ts": _utc_now_z(), "outcome": REASON.SKIP_INPUTS_INCOMPLETE,
            "armed": write_armed, "armed_by": ss.get("armed_by",""),
            "key": {"inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                    "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)},
            "ab": ("PINNED" if is_ab_pinned else "NONE"),
            "file_pi": {"mode": ("file" if policy_canon=="projected:file" else policy_canon),
                        "valid": file_pi_valid, "reasons": file_pi_reasons[:3]}
        })
        st.caption("Inputs incomplete — write disabled (A/B & gallery remain viewable).")

    if write_enabled and stale and (not ticket_required):
        write_enabled = False
        _append_witness({
            "ts": _utc_now_z(), "outcome": REASON.SKIP_SSOT_STALE,
            "armed": write_armed, "armed_by": ss.get("armed_by",""),
            "key": {"inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                    "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)},
            "ab": ("PINNED" if is_ab_pinned else "NONE"),
            "ab_ticket": {"pending": ab_ticket_pending, "last_written": last_ab_ticket_written},
        })
        st.warning("SSOT is stale — run Overlap to refresh, or use A/B one-shot ticket (fresh) to override.")

    if write_enabled and (policy_canon == "projected:file") and (not file_pi_valid):
        write_enabled = False
        _append_witness({
            "ts": _utc_now_z(), "outcome": REASON.SKIP_FILE_PI_INVALID,
            "armed": write_armed, "armed_by": ss.get("armed_by",""),
            "key": {"inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                    "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)},
            "ab": ("PINNED" if is_ab_pinned else "NONE"),
            "file_pi": {"mode": "file", "valid": False, "reasons": file_pi_reasons[:3]},
        })
        st.caption("FILE Π invalid — write disabled (fix Π or re-freeze from AUTO).")

    should_write = write_enabled and write_armed and ((write_key != last_key) or ticket_required)

    if not should_write:
        skip_reason = REASON.SKIP_NO_MATERIAL_CHANGE
        if write_enabled and is_ab_pinned and (ab_ticket_pending == last_ab_ticket_written):
            skip_reason = REASON.SKIP_AB_TICKET_ALREADY_WRITTEN
        _append_witness({
            "ts": _utc_now_z(), "outcome": skip_reason,
            "armed": write_armed, "armed_by": ss.get("armed_by",""),
            "key": {"inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                    "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)},
            "ab": ("PINNED" if is_ab_pinned else "NONE"),
            "file_pi": {"mode": ("file" if policy_canon=="projected:file" else policy_canon),
                        "valid": file_pi_valid, "reasons": ([] if write_enabled else file_pi_reasons[:3])},
        })
        st.caption("Cert unchanged — skipping rewrite." if write_enabled else "Write disabled.")
        wrote_cert = False
    else:
        wrote_cert = True
        # stable run_id per inputs_sig (bumps on change)
        if ss.get("_last_inputs_sig") != inputs_sig:
            seed = _sha256_hex((":".join(inputs_sig) + f"|{int(time.time())}").encode())[:8]
            ss["_last_inputs_sig"] = inputs_sig
            ss["last_run_id"] = seed
            ss["run_idx"] = 0
        ss["run_idx"] = int(ss.get("run_idx", 0)) + 1

    # ---------- assemble cert (strict schema) ----------
    district_id = (ss.get("_district_info") or {}).get("district_id", ss.get("district_id","UNKNOWN"))
    n2 = int((ib.get("dims") or {}).get("n2") or rc.get("n2") or 0)
    n3 = int((ib.get("dims") or {}).get("n3") or rc.get("n3") or 0)
    lane_mask = list(rc.get("lane_mask_k3") or [])

    # GF(2) diagnostics (lane-masked bottom rows) — shape-safe, no solver calls
    def _xor(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]

    def _gf2_mul_local(A, B):
        if "_gf2_mul" in globals() and callable(globals()["_gf2_mul"]):
            return _gf2_mul(A, B)  # type: ignore[name-defined]
        if not A or not B: return []
        m, k, n = len(A), len(A[0]), len(B[0])
        C = [[0]*n for _ in range(m)]
        for i in range(m):
            Ai = A[i]
            for t in range(k):
                if Ai[t] & 1:
                    Bt = B[t]
                    for j in range(n):
                        C[i][j] ^= (Bt[j] & 1)
        return C

    H2 = (getattr(H_obj, "blocks", None).__root__.get("2") if getattr(H_obj, "blocks", None) else []) or []
    d3 = rc.get("d3") or []
    C3 = (getattr(C_obj, "blocks", None).__root__.get("3") if getattr(C_obj, "blocks", None) else []) or []
    I3 = [[1 if i == j else 0 for j in range(len(C3))] for i in range(len(C3))] if C3 else []

    H2d3 = _gf2_mul_local(H2, d3) if (H2 and d3 and H2[0] and d3[0] and len(H2[0]) == len(d3)) else []
    C3pI3 = _xor(C3, I3) if C3 else []

    def _bottom_masked_row(M, lm):
        if not M or not lm: return []
        row = M[-1]
        L = min(len(row), len(lm))
        return [int(row[j]) if int(lm[j]) else 0 for j in range(L)]

    lane_vec_H2d3  = _bottom_masked_row(H2d3, lane_mask)
    lane_vec_C3pI3 = _bottom_masked_row(C3pI3, lane_mask)

    # A/B embed snapshot (prefer rich, fallback to pin payload)
    def _ab_is_fresh(ab_snap: dict, rc_now: dict, ib_now: dict) -> bool:
        try:
            cur_inputs = tuple(inputs_sig)  # already derived
            snap_inputs = tuple(ab_snap.get("inputs_sig") or ())
            if snap_inputs and (snap_inputs != cur_inputs): return False
            if _canon_policy(ab_snap.get("policy_tag","")) != _canon_policy(rc_now.get("policy_tag","")):
                return False
            if _canon_policy(rc_now.get("policy_tag","")) == "projected:file":
                pj_now = rc_now.get("projector_hash","")
                pj_ab  = ((ab_snap.get("projected") or {}).get("projector_hash",""))
                return str(pj_now) == str(pj_ab)
            return True
        except Exception:
            return False

    ab_embed = {"fresh": bool(False)}
    ab_rich = ss.get("ab_compare") or {}
    use_rich = bool(is_ab_pinned and ab_fresh and _ab_is_fresh(ab_rich, rc_now=rc, ib_now=ib))
    ab_src = ab_rich if use_rich else (ab_pin.get("payload") or {})

    strict_block = ab_src.get("strict") or {}
    if not strict_block:
        strict_block = {"out": (ab_src.get("strict") or {}).get("out", {})}

    # projector hash “now” (for AUTO we use lanes hash; for STRICT empty)
    if policy_canon == "projected:file":
        proj_hash_now = rc.get("projector_hash","") or proj_hash
    elif policy_canon == "projected:auto":
        proj_hash_now = hashlib.sha256(json.dumps(rc.get("lane_mask_k3") or [], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    else:
        proj_hash_now = ""

    proj_src = ab_src.get("projected") or {}
    proj_out = proj_src.get("out") or {}
    projected_block = {
        "label":      proj_src.get("label", policy_canon),
        "policy_tag": proj_src.get("policy_tag", policy_canon),
        "out":        proj_out,
        "lane_vec_H2d3":      list(proj_src.get("lane_vec_H2d3") or []),
        "lane_vec_C3plusI3":  list(proj_src.get("lane_vec_C3plusI3") or []),
        "pass_vec":           list(proj_src.get("pass_vec") or []),
        "projector_filename": str(proj_src.get("projector_filename") or rc.get("projector_filename") or ""),
        "projector_hash":     str(proj_src.get("projector_hash") or proj_hash_now),
        "projector_consistent_with_d": bool(proj_src.get("projector_consistent_with_d", rc.get("projector_consistent_with_d", True))),
    }

    ab_embed.update({
        "pair_tag":     ab_src.get("pair_tag", f"strict__VS__{policy_canon}"),
        "inputs_sig":   list(ab_src.get("inputs_sig") or inputs_sig),
        "lane_mask_k3": list(ab_src.get("lane_mask_k3") or lane_mask[:]),
        "strict":       strict_block,
        "projected":    projected_block,
        "fresh":        bool(is_ab_pinned and ab_fresh),
    })
    if is_ab_pinned and not ab_fresh:
        ab_embed["stale_reason"] = ab_status

    # Growth / Gallery (kept minimal; no side effects)
    growth  = {"growth_bumps": int(ss.get("growth_bumps", 0)), "H_diff": ss.get("H_diff","")}
    gallery = {
        "projected_green": bool(ss.get("projected_green", pass_vec[1])),
        "tag": ss.get("gallery_tag",""),
        "strictify": ss.get("gallery_strictify","tbd"),
    }

    # Identity / Policy / Inputs
    district_id = (ss.get("_district_info") or {}).get("district_id", ss.get("district_id","UNKNOWN"))
    identity = {
        "district_id":   district_id,
        "run_id":        ss.get("last_run_id","00000000"),
        "run_idx":       int(ss.get("run_idx", 0)),
        "field":         ss.get("field_label", "B2"),
        "fixture_label": ss.get("fixture_label",""),
        "fixture_code":  (ss.get("run_ctx") or {}).get("fixture_code",""),
        "fixture_nonce": ss.get("_fixture_nonce", 0),
    }
    policy = {
        "label_raw":      policy_raw,
        "canon":          policy_canon,
        "projector_mode": ("strict" if policy_canon=="strict" else ("file" if policy_canon=="projected:file" else "auto")),
    }
    if policy_canon == "projected:file":
        policy["projector_hash"] = proj_hash

    filenames = dict((ib.get("filenames") or {}))
    filenames.setdefault("boundaries", ss.get("fname_boundaries","boundaries.json"))
    filenames.setdefault("shapes",     ss.get("fname_shapes","shapes.json"))
    filenames.setdefault("cmap",       ss.get("fname_cmap","cmap.json"))
    filenames.setdefault("H",          ss.get("fname_h","H.json"))
    filenames.setdefault("U",          ss.get("fname_U","U.json"))
    if policy_canon == "projected:file" and rc.get("projector_filename"):
        filenames["projector"] = rc.get("projector_filename")

    inputs = {
        "dims": {"n2": n2, "n3": n3},
        "lane_mask_k3": lane_mask[:],
        "filenames": filenames,
        "hashes": {
            "boundaries_hash": inputs_sig[0],
            "shapes_hash":     inputs_sig[4],
            "C_hash":          inputs_sig[1],
            "H_hash":          inputs_sig[2],
            "U_hash":          inputs_sig[3],
        },
        "inputs_sig": list(inputs_sig),
    }
    fx_cache = ss.get("_fixtures_cache") or {}
    inputs["registry"] = {
        "fixtures_version": fx_cache.get("version",""),
        "fixtures_path":    fx_cache.get("__path","configs/fixtures.json"),
        "fixtures_hash":    ss.get("_fixtures_bytes_hash",""),
        "ordering":         list(fx_cache.get("ordering") or []),
    }

    checks = {
        "k": {"2": {"eq": pass_vec[0]}, "3": {"eq": pass_vec[1]}},
        "grid":  bool(out.get("grid", True)),
        "fence": bool(out.get("fence", True)),
        "guards": {"ker_guard": (policy_canon=="strict"), "policy_guard": True},
    }
    diagnostics = {
        "lane_mask_k3": lane_mask[:],
        "lane_vec_H2@d3": lane_vec_H2d3,
        "lane_vec_C3+I3": lane_vec_C3pI3,
        "residual_tag": (ss.get("residual_tags", {}) or {}).get(policy_canon.split(":")[0], "none"),
    }

    cert = {
        "schema_version": SCHEMA_VERSION,
        "app_version":    APP_VERSION,
        "python_version": _python_version_str(),
        "identity":   identity,
        "policy":     policy,
        "inputs":     inputs,
        "checks":     checks,
        "diagnostics":diagnostics,
        "ab_embed":   ab_embed,
        "growth":     growth,
        "gallery":    gallery,
        "hashes":     {},
    }
    warns = []
    if len(lane_mask) != max(n3,0): warns.append("CERT_INVAR_WARN: lane_mask_k3 length != n3")
    if policy_canon == "projected:file" and not proj_hash: warns.append("CERT_INVAR_WARN: projector_hash missing for projected:file")
    if warns: cert["_warnings"] = warns

    cert["written_at_utc"] = _utc_now_z()
    cert_blob = json.dumps(cert, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cert["hashes"]["content_hash"] = _sha256_hex(cert_blob)

    # deterministic filename / single-writer path
    policy_sanitized = _sanitize(policy_raw)
    content12        = cert["hashes"]["content_hash"][:12]
    ab_tail          = "__AB__embedded" if ab_fresh else ""
    base_dir         = Path("logs") / "certs" / district_id / ss.get("last_run_id","00000000")
    fname            = f"overlap__{district_id}__{policy_sanitized}__{ss.get('last_run_id','00000000')}__{content12}{ab_tail}.json"
    fpath            = base_dir / fname

    _write_json_atomic(fpath, cert)

    # session updates
    ss["_last_cert_write_key"] = write_key
    ss["last_cert_path"] = str(fpath)
    ss["cert_payload"]   = cert
    ss["write_armed"]    = False

    # one-shot ticket bookkeeping
    if is_ab_pinned and (ab_ticket_pending is not None):
        ss["_last_ab_ticket_written"] = ss.get("_ab_ticket_pending")
    if ab_fresh and is_ab_pinned:
        ss["ab_pin"] = {"state":"idle","payload":None,"consumed":True}

    # witness
    _append_witness({
        "ts": cert["written_at_utc"],
        "outcome": REASON.WROTE_CERT if wrote_cert else REASON.SKIP_NO_MATERIAL_CHANGE,
        "armed": True,
        "armed_by": ss.get("armed_by",""),
        "key": {"inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)},
        "ab": (REASON.AB_EMBEDDED if (is_ab_pinned and ab_fresh) else (REASON.AB_NONE if not is_ab_pinned else ab_status)),
        "file_pi": {"mode": ("file" if policy_canon=="projected:file" else policy_canon),
                    "valid": True, "reasons": []},
        "path": fpath.as_posix()
    })

    # clear transient fixture mirrors (prevent carry-over)
    for _k in ("fixture_label","gallery_tag","gallery_strictify"):
        ss[_k] = ""
    _rc_tmp = ss.get("run_ctx") or {}
    _rc_tmp.pop("fixture_label", None)
    _rc_tmp.pop("fixture_code", None)
    ss["run_ctx"] = _rc_tmp

    # receipt
    st.success(f"Cert → {fpath.name} · dir={base_dir.as_posix()} · {content12}")

    # ---------- tail (read-only, compact) ----------
    with st.container():
        CERTS_ROOT = Path("logs") / "certs"
        try:
            found = list(CERTS_ROOT.rglob("*.json"))
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ab_only = st.checkbox("Show only certs with A/B embed", value=False, key="tail_ab_only_final")
            shown = 0
            st.caption(f"Latest certs — Found {len(found)}")
            for p in found:
                if shown >= 5: break
                try:
                    info = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                has_ab = bool(info.get("ab_embed",{}).get("fresh"))
                if ab_only and not has_ab:
                    continue
                ident  = info.get("identity", {})
                policy = info.get("policy", {})
                checks = info.get("checks", {})
                k2 = bool(((checks.get("k") or {}).get("2") or {}).get("eq", False))
                k3 = bool(((checks.get("k") or {}).get("3") or {}).get("eq", False))
                ts = info.get("written_at_utc","")
                flag = " · [A/B]" if has_ab else ""
                st.write(f"• {ts} · {ident.get('district_id','?')} · {policy.get('label_raw','?')} · k2/k3={int(k2)}/{int(k3)} · {p.name}{flag}")
                shown += 1
            if shown == 0:
                st.caption("No certs to show with current filter.")
        except Exception as e:
            st.warning(f"Tail listing failed: {e}")
# ============================== /Cert & Provenance ==============================



        







# ---------------- Fixture registry (lean, SSOT-aligned) ----------------
from pathlib import Path
import json as _json
import hashlib as _hashlib
import streamlit as st

# Cache keyed by file sig (mtime:size:sha12) so reruns are cheap
_FIXTURE_CACHE = {"sig": "", "data": {"version": "", "ordering": [], "fixtures": []}}

def _file_sig(p: Path) -> str:
    try:
        st_ = p.stat()
        return f"{int(st_.st_mtime)}:{st_.st_size}:{_hashlib.sha256(p.read_bytes()).hexdigest()[:12]}"
    except Exception:
        return ""

def _load_fixtures_json() -> tuple[dict, str]:
    """Load configs/fixtures.json (or empty), returning (data, file_sig)."""
    p = Path("configs/fixtures.json")
    if not p.exists():
        return {"version": "", "ordering": [], "fixtures": []}, ""
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
        sig  = _file_sig(p)
    except Exception:
        data, sig = {"version": "", "ordering": [], "fixtures": []}, ""
    return data, sig

def _get_fixtures_cached() -> dict:
    data, sig = _load_fixtures_json()
    if sig and _FIXTURE_CACHE.get("sig") == sig and _FIXTURE_CACHE.get("data") is not None:
        return _FIXTURE_CACHE["data"]
    _FIXTURE_CACHE["sig"], _FIXTURE_CACHE["data"] = sig, data
    # provenance (optional)
    st.session_state["_fixtures_bytes_hash"] = (sig.split(":")[-1] if sig else "")
    st.session_state["_fixtures_cache"] = data
    return data

def _vec_eq(a, b) -> bool:
    try:
        aa = [int(x) for x in (a or [])]
        bb = [int(x) for x in (b or [])]
        return len(aa) == len(bb) and aa == bb
    except Exception:
        return False

def _policy_canon_from_rc(rc: dict) -> str:
    pol = str(rc.get("policy_tag") or rc.get("mode") or "strict").lower()
    return pol.replace("  ", " ")

def _apply_fixture_after_overlap(*, rc: dict, diag: dict, commit: bool = True) -> dict:
    """
    Decide fixture from registry using SSOT-only predicates and return the chosen dict.
    If commit=True, minimally stamp run_ctx (fixture_label/code) + legacy mirrors.
    """
    ss = st.session_state
    reg = _get_fixtures_cached() or {}
    ordering = list(reg.get("ordering") or [])
    fixtures = list(reg.get("fixtures") or [])

    district     = (ss.get("_district_info") or {}).get("district_id") or rc.get("district_id") or "UNKNOWN"
    policy_canon = _policy_canon_from_rc(rc)

    # Pull vectors strictly from existing state (no recompute)
    lane_mask      = list(rc.get("lane_mask_k3") or diag.get("lane_mask") or [])
    lane_vec_H2    = list(diag.get("lane_vec_H2@d3") or diag.get("H_bottom") or [])
    lane_vec_C3pI3 = list(diag.get("lane_vec_C3+I3") or diag.get("C3_plus_I3_bottom") or [])

    # Order fixtures: declared 'ordering' first, then any extras
    by_code = {fx.get("code"): fx for fx in fixtures}
    ordered = [by_code[c] for c in ordering if c in by_code] + [fx for fx in fixtures if fx.get("code") not in by_code]

    chosen: dict = {}
    for fx in ordered:
        m = fx.get("match") or {}
        if m.get("district") and str(m["district"]) != str(district):
            continue
        pol_any = [str(x).lower() for x in (m.get("policy_canon_any") or [])]
        if pol_any and (policy_canon not in pol_any):
            continue
        if "lanes" in m and not _vec_eq(m["lanes"], lane_mask):
            continue
        if "H_bottom" in m and not _vec_eq(m["H_bottom"], lane_vec_H2):
            continue
        if "C3_plus_I3_bottom" in m and not _vec_eq(m["C3_plus_I3_bottom"], lane_vec_C3pI3):
            continue
        if "strict_eq3" in m and isinstance(rc.get("eq3_strict"), bool):
            if bool(m["strict_eq3"]) != bool(rc.get("eq3_strict")):
                continue

        chosen = {
            "fixture_code":  fx.get("code", ""),
            "fixture_label": fx.get("label", ""),
            "tag":           fx.get("tag", ""),
            "strictify":     fx.get("strictify", "tbd"),
            "growth_bumps":  int(fx.get("growth_bumps", 0) or 0),
        }
        break

    # Fallback (never blocks)
    if not chosen:
        chosen = {
            "fixture_code":  "",
            "fixture_label": f"{district} • lanes={lane_mask} • H={lane_vec_H2} • C+I={lane_vec_C3pI3}",
            "tag":           "novelty",
            "strictify":     "tbd",
            "growth_bumps":  int(ss.get("growth_bumps", 0) or 0),
        }

    if commit:
        rc2 = dict(rc)
        rc2["fixture_label"] = chosen["fixture_label"]
        rc2["fixture_code"]  = chosen["fixture_code"]
        st.session_state["run_ctx"] = rc2
        # legacy mirrors (harmless if unused)
        ss["fixture_label"]     = chosen["fixture_label"]
        ss["gallery_tag"]       = chosen["tag"]
        ss["gallery_strictify"] = chosen["strictify"]
        ss["growth_bumps"]      = chosen["growth_bumps"]

    return chosen





# ───────────────────────── Reports: Perturbation sanity (lanes-only) ──────────────────────
def run_reports__perturb_and_fence(*, max_flips: int, seed: str, include_fence: bool, enable_witness: bool):
    """
    Lanes-only flips over d3; writes:
      - reports/perturbation_sanity.csv
      - reports/perturbation_sanity__<hash12>.json
    Reads SSOT (B/C/H, run_ctx, lane_mask_k3). No “repair” mechanics. N/A + reason only.
    """
    ss = st.session_state
    REPORTS_DIR = Path(ss.get("REPORTS_DIR", "reports")); REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Preflight: hydrate B/C/H
    B0, C0, H0 = _reports_hydrate_BCH()
    d3_base = (B0.blocks.__root__.get("3") or []) if B0 else []
    C3      = (C0.blocks.__root__.get("3") or []) if C0 else []
    H2      = (H0.blocks.__root__.get("2") or []) if H0 else []

    n2 = len(d3_base)
    n3 = len(d3_base[0]) if (n2 and d3_base[0]) else 0
    if not (n2 and n3 and H2 and (H2[0] if H2 else []) and C3 and (C3[0] if C3 else [])):
        st.error("N/A (reason: missing H2/d3/C3). Run Overlap/Cert once to freeze SSOT.")
        st.stop()

    # Lane mask & projector
    lane_mask, P_diag, selected_cols = _pin_lane_mask_and_projector(d3_base, n2, n3)
    st.caption(f"Lane mask → {lane_mask} (lanes: {len(selected_cols)}/{n3})")

    # Projector FILE validation hint (does not block)
    rc = ss.get("run_ctx") or {}
    proj_status = {"status": "OK", "na_reason_code": ""}
    try:
        if rc.get("mode") == "projected(columns@k=3,file)" and file_validation_failed():
            proj_status = {"status":"N/A", "na_reason_code":"P3_FILE_INVALID"}
    except Exception:
        pass

    # Baseline checks
    R3_base = _strict_R3(H2, d3_base, C3)
    k3_strict_base = _eq_zero(R3_base)
    k3_proj_base   = (_eq_zero(_projected_R3(R3_base, P_diag)) if P_diag else None)

    # Flip generator (deterministic from seed)
    h = int(_hash.sha256(str(seed).encode("utf-8")).hexdigest(), 16)
    def _flip_targets(n2_, n3_, budget):
        i = (h % max(1, n2_)) if n2_ else 0
        j = ((h >> 8) % max(1, n3_)) if n3_ else 0
        for k in range(int(budget)):
            yield (i, j, k)
            i = (i + 1 + (h % 3)) % (n2_ or 1)
            j = (j + 2 + ((h >> 5) % 5)) % (n3_ or 1)

    # Walk flips: lanes-only domain
    rows_csv, results_json = [], []
    total_flips = in_domain_flips = 0
    for (r, c, k) in _flip_targets(n2, n3, max_flips):
        if c not in selected_cols:
            total_flips += 1
            rows_csv.append([k, "none", "none", "off-domain (ker column)"])
            results_json.append({
                "flip_id": int(k),
                "flip": {"row": int(r), "col": int(c), "lane_col": False, "skip_reason": "off-domain"},
                "baseline": {"k3_strict": bool(k3_strict_base), "k3_projected": (None if k3_proj_base is None else bool(k3_proj_base))}
            })
            continue

        total_flips += 1; in_domain_flips += 1
        # mutate a copy of d3
        d3_mut = [row[:] for row in d3_base]; d3_mut[r][c] ^= 1
        dB = B0.dict() if hasattr(B0, "dict") else {"blocks": {}}
        dB = _json.loads(_json.dumps(dB))  # deep-copy
        dB.setdefault("blocks", {})["3"] = d3_mut
        Bk = io.parse_boundaries(dB)

        _, tag_sK, eq_sK, _, eq_pK = _sig_tag_eq(Bk, C0, H0, P_diag)
        guard = _first_tripped_guard({"3": {"eq": bool(eq_sK)}})

        rows_csv.append([k, guard, guard, ""])
        results_json.append({
            "flip_id": int(k),
            "flip": {"row": int(r), "col": int(c), "lane_col": True},
            "baseline": {"k3_strict": bool(k3_strict_base), "k3_projected": (None if k3_proj_base is None else bool(k3_proj_base))},
            "after":    {"k3_strict": bool(eq_sK),            "k3_projected": (None if eq_pK is None else bool(eq_pK))},
            "residual_tag_after": str(tag_sK or "none"),
            "guard_tripped": guard,
        })

    # CSV
    csv_path = REPORTS_DIR / "perturbation_sanity.csv"
    _atomic_write_csv(
        csv_path,
        header=["flip_id","guard_tripped","expected_guard","note"],
        rows=rows_csv,
        meta_lines=[
            f"schema_version={SCHEMA_VERSION}", f"saved_at={_utc_iso_z()}",
            f"run_id={(rc or {}).get('run_id','')}", f"app_version={APP_VERSION}",
            f"seed={seed}", f"n2={n2}", f"n3={n3}"
        ],
    )

    # JSON payload
    policy_block = _policy_block_from_run_ctx(rc)
    inputs_block = _inputs_block_from_session(strict_dims=(n2, n3))
    need = ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")
    if not all((inputs_block.get("hashes") or {}).get(k, "") for k in need):
        st.error("N/A (reason: INPUT_HASHES_MISSING). Wire SSOT from Cert/Overlap; backfill disabled.")
        st.stop()

    payload = {
        "schema_version": SCHEMA_VERSION, "written_at_utc": _utc_iso_z(),
        "app_version": APP_VERSION, "field": FIELD,
        "identity": {"run_id": rc.get("run_id",""), "district_id": rc.get("district_id",""), "fixture_nonce": rc.get("fixture_nonce","")},
        "projector_validation": proj_status,
        "projected_check": {"enabled": bool(P_diag), "na_reason_code": ("" if P_diag else "LANE_MASK_MISSING")},
        "policy": policy_block, "inputs": inputs_block,
        "baseline": {"k3_strict": bool(k3_strict_base), "k3_projected": (None if k3_proj_base is None else bool(k3_proj_base))},
        "run": {"max_flips": int(max_flips), "domain": "d3_supports", "lanes_only": True, "seed": str(seed)},
        "results": results_json,
        "summary": {"flips": int(total_flips), "in_domain_flips": int(in_domain_flips)},
        "integrity": {"content_hash": ""},
    }
    payload["integrity"]["content_hash"] = _hash_json(payload)
    jname = f"perturbation_sanity__{payload['integrity']['content_hash'][:12]}.json"
    jpath = REPORTS_DIR / jname
    _guarded_atomic_write_json(jpath, payload)

    st.success(f"Perturbation sanity → {csv_path.name} + {jname}")
    with open(jpath, "rb") as jf:
        st.download_button("Download perturbation_sanity.json", jf, file_name=jname, key=f"dl_ps_json_{jname[-12:]}")
    with open(csv_path, "rb") as cf:
        st.download_button("Download perturbation_sanity.csv", cf, file_name=csv_path.name, key=f"dl_ps_csv_{csv_path.stem[-8:]}")

    # Fence (optional) — placeholder (your working fence runner can be called here)
    if include_fence:
        # e.g., run_reports__fence_stress(...)  # keep your existing implementation
        pass
# ───────────────────────── /Reports: Perturbation sanity ─────────────────────





# ============================== Coverage (lean, SSOT-aligned) ==============================
# Goal: keep a tiny baseline registry for patterns; delegate health/rollup to the existing C1 block.

import json as _json
import re as _re
from pathlib import Path as _Path

import streamlit as st

# ---------- Tiny helpers: tokens & normalization ----------
_PAT_RE = _re.compile(r"pattern=\[\s*(.*?)\s*\]")

def _extract_pattern_tokens(sig: str) -> list[str]:
    """Return list of bit-strings found inside pattern=[...]."""
    m = _PAT_RE.search(sig or "")
    if not m:
        return []
    inner = m.group(1)
    toks = []
    for raw in inner.split(","):
        s = "".join(ch for ch in raw.strip() if ch in "01")
        if s:
            toks.append(s)
    return toks

def _rebuild_signature_with_tokens(sig: str, tokens: list[str]) -> str:
    """Replace/insert pattern=[...] in sig with given tokens."""
    if "pattern=[" not in sig:
        return f"pattern=[{','.join(tokens)}]"
    return _re.sub(r"pattern=\[[^\]]*\]", f"pattern=[{','.join(tokens)}]", sig)

def _ensure_dims_or_seed_demo() -> tuple[int, int]:
    """Always return (n2,n3). Seed 2×3 once if missing so UI can flow."""
    rc = st.session_state.get("run_ctx") or {}
    n2 = int(rc.get("n2") or 0)
    n3 = int(rc.get("n3") or 0)
    if n2 <= 0 or n3 <= 0:
        n2, n3 = 2, 3
        rc["n2"], rc["n3"] = n2, n3
        st.session_state["run_ctx"] = rc
    return n2, n3

def _normalize_signature_line_to_n2(sig: str, *, n2: int, n3: int) -> tuple[str, str]:
    """
    Convert pattern=[t1,t2,...] to exactly n3 tokens, each exactly n2 bits.
    - Truncate from the top if longer; left-pad with '0' if shorter.
    Returns (full_signature_norm, pattern_only_norm).
    """
    toks = _extract_pattern_tokens(sig) or []
    norm = []
    for t in toks:
        bits = "".join(ch for ch in t if ch in "01")
        if len(bits) >= n2:
            norm.append(bits[:n2])
        else:
            norm.append(("0"*(n2 - len(bits))) + bits)
    while len(norm) < n3:
        norm.append("0"*n2)
    if len(norm) > n3:
        norm = norm[:n3]
    patt_norm = f"pattern=[{','.join(norm)}]"
    full_norm = _rebuild_signature_with_tokens(sig, norm)
    return full_norm, patt_norm

def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for s in items:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

# ---------- Baseline registry (no UI buttons; callable from elsewhere) ----------
def add_patterns_to_baseline(lines: list[str], *, normalize: bool = True):
    """
    Accepts a list of signature strings or raw tokens (e.g. '00,11,10').
    Normalizes to current (n2,n3) and stores:
      - run_ctx.known_signatures            → list[str] (full, deduped, in order)
      - run_ctx.known_signatures_patterns  → sorted list[str] of 'pattern=[...]'
    """
    if not lines:
        return {"ok": False, "reason": "EMPTY_INPUT"}

    n2, n3 = _ensure_dims_or_seed_demo()

    full_norm, patt_norm = [], []
    for s in lines:
        s = s.strip()
        if not s:
            continue
        # allow raw tokens like "00,01,11"
        if not s.startswith("pattern=["):
            toks = [t.strip() for t in s.split(",") if t.strip()]
            s = f"pattern=[{','.join(toks)}]"
        f, p = _normalize_signature_line_to_n2(s, n2=n2, n3=n3) if normalize else (s, s)
        full_norm.append(f); patt_norm.append(p)

    rc = st.session_state.get("run_ctx") or {}
    rc["known_signatures"] = _dedupe_keep_order(full_norm)
    rc["known_signatures_patterns"] = sorted(set(patt_norm))
    st.session_state["run_ctx"] = rc

    return {
        "ok": True,
        "count_full": len(rc["known_signatures"]),
        "count_patterns": len(rc["known_signatures_patterns"]),
        "n2": n2, "n3": n3,
    }

# ---------- Health & rollup: thin adapter to the existing C1 helpers ----------
def render_coverage_health_and_rollup():
    """
    Uses the already-defined C1 helpers in this app:
      - _c1_paths() → (coverage.jsonl, coverage_rollup.csv)
      - _c1_health_ping(jsonl_path, tail=50)
      - _c1_rollup_rows(jsonl_path)
      - _c1_write_rollup_csv(rows, out_path)
    If any are missing, we degrade gracefully.
    """
    # Prefer the SSOT C1 paths if available; else default to logs/reports/coverage.jsonl
    if "_c1_paths" in globals():
        jsonl_path, csv_path = _c1_paths()  # type: ignore[name-defined]
    else:
        base = _Path("logs") / "reports"; base.mkdir(parents=True, exist_ok=True)
        jsonl_path, csv_path = (base / "coverage.jsonl"), (base / "coverage_rollup.csv")

    st.caption(f"Source: {jsonl_path} · Output: {csv_path}")

    # Health
    if "_c1_health_ping" in globals():
        hp = _c1_health_ping(jsonl_path, tail=50)  # type: ignore[name-defined]
        if hp is None:
            st.info("No coverage.jsonl yet — run the solver to produce coverage events.")
        else:
            def fmt(x): return "—" if x is None else f"{x:.3f}"
            st.markdown(
                f"**C1 Health** tail={hp['tail']} · "
                f"sel={fmt(hp.get('mean_sel_mismatch_rate'))} · "
                f"off={fmt(hp.get('mean_offrow_mismatch_rate'))} · "
                f"ker={fmt(hp.get('mean_ker_mismatch_rate'))} · "
                f"ctr={fmt(hp.get('mean_contradiction_rate'))}"
            )
    else:
        st.info("C1 health helpers not loaded; skip health summary.")

    # Rollup
    col = st.columns(2)[0]
    if col.button("Build rollup CSV", key="btn_cov_rollup"):
        if all(n in globals() for n in ("_c1_rollup_rows", "_c1_write_rollup_csv")):
            rows = _c1_rollup_rows(jsonl_path)         # type: ignore[name-defined]
            _c1_write_rollup_csv(rows, csv_path)       # type: ignore[name-defined]
            st.success(f"Wrote {len(rows)} rows → {csv_path}")
            try:
                import pandas as _pd
                st.dataframe(_pd.DataFrame(rows), use_container_width=True)
            except Exception:
                st.table(rows)
        else:
            st.warning("C1 rollup helpers not loaded; cannot build rollup.")

# ---------- Minimal UI wrapper (optional) ----------
with st.expander("Coverage (baseline cache · health & rollup)", expanded=False):
    st.caption("Lean coverage utilities; baseline is optional, sampling uses C1.")
    # Paste-or-upload baseline patterns here if you still need them:
    pasted = st.text_area("Paste signatures or raw tokens (one per line, optional)", value="", key="cov_min_paste")
    if st.button("Add to baseline (normalize to current n₂×n₃)", key="cov_min_add"):
        lines = [ln for ln in (pasted or "").splitlines() if ln.strip()]
        res = add_patterns_to_baseline(lines, normalize=True)
        if res.get("ok"):
            rc = st.session_state.get("run_ctx") or {}
            st.success(f"Baseline now has {len(rc.get('known_signatures_patterns') or [])} patterns @ n₂={res['n2']} n₃={res['n3']}.")
        else:
            st.warning(res.get("reason", "Nothing added."))

    # Preview (lightweight)
    rc_view = st.session_state.get("run_ctx") or {}
    ks = list(rc_view.get("known_signatures") or [])
    if ks:
        st.caption(f"Baseline: {len(ks)} signatures (showing up to 5):")
        st.code("\n".join(ks[:5] + (["…"] if len(ks) > 5 else [])), language="text")
    else:
        st.caption("Baseline empty (optional).")

    st.divider()
    render_coverage_health_and_rollup()
# ============================ /Coverage (lean) ============================
