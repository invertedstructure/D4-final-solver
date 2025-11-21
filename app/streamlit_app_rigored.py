import streamlit as st
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
# == EARLY HELPERS (v2 wiring) ==
# Safe UI nonce (prevents "no _ui_nonce" warning)
try:
    import streamlit as _st
except Exception:
    class _DummyST:
        session_state = {}
    _st = _DummyST()
try:
    import secrets as _secrets
except Exception:
    class _S:
        @staticmethod
        def token_hex(n): 
            return "anon"
    _secrets = _S()
try:
    if "_ui_nonce" not in _st.session_state:
        _st.session_state["_ui_nonce"] = _secrets.token_hex(4)
except Exception:
    pass
    
import json as _json
import hashlib as _hash

# --- C1 canonical paths (tuple; JSON-first) ---
from pathlib import Path as _Path
def _c1_paths():
    base = _Path("logs") / "reports"
    base.mkdir(parents=True, exist_ok=True)
    return (base / "coverage.jsonl", base / "coverage_rollup.csv")


def _canon_dump_and_sig8(obj):
    """Return (canonical_json_text, first_8_of_sha256) for small cert payloads."""
    can = _v2_canonical_obj(obj)
    raw = _json.dumps(can, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = _hash.sha256(raw).hexdigest()
    return raw.decode("utf-8"), h[:8]
def _v2_coverage_path():
    try:
        root = _REPO_DIR
    except Exception:
        from pathlib import Path as _Path
        root = _Path(__file__).resolve().parents[1]
    p = root / "logs" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p / "coverage.jsonl"

def _v2_coverage_append(row: dict):
    """Append one JSON line to coverage.jsonl (best-effort)."""
    import json as _json, time as _time
    row = dict(row or {})
    row.setdefault("ts_utc", int(_time.time()))
    with _v2_coverage_path().open("a", encoding="utf-8") as f:
        f.write(_json.dumps(row, separators=(",", ":"), sort_keys=False) + "\n")

def _v2_coverage_count_for_snapshot(snapshot_id: str) -> int:
    """Count parseable rows matching a snapshot_id (best-effort)."""
    import json as _json
    p = _v2_coverage_path()
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = _json.loads(line)
            except Exception:
                continue
            if j.get("snapshot_id") == snapshot_id:
                n += 1
    return n


# Pass1/2 helpers to write bundle.json and loop_receipt
from pathlib import Path as _VPath
import json as _Vjson, hashlib as _Vhash

_V2_EXPECTED = [
    ("strict",               "overlap__", "__strict__"),
    ("projected_auto",       "overlap__", "__projected_columns_k_3_auto__"),
    ("ab_auto",              "ab_compare__strict_vs_projected_auto__", ""),
    ("freezer",              "projector_freezer__", "",),
    # ab_file: strict vs projected(FILE) pair
    ("ab_file",              "ab_compare__strict_vs_projected_file__", ""),
    ("projected_file",       "overlap__", "__projected_columns_k_3_file__"),
]


def _v2_sha256_path(p: _VPath) -> str:
    h = _Vhash.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _v2_find_expected_files(bdir: _VPath):
    out = {}
    for key, prefix, mid in _V2_EXPECTED:
        for fp in bdir.glob("*.json"):
            name = fp.name
            if key.startswith("ab_") and prefix in name:
                if (mid == "" or mid in name):
                    out[key] = fp
            elif prefix in name and (mid in name if mid else True):
                out[key] = fp
    return out

def _coverage_rollup_write_csv(snapshot_id: str | None = None):
    """
    Build C1 rollup over logs/reports/coverage.jsonl.
    If snapshot_id is provided, filter to that snapshot only (v2 uses the __real one).
    Writes logs/reports/coverage_rollup.csv and returns its Path (or None if no coverage file).
    """
    from pathlib import Path as _Path
    import json as _json, csv as _csv, re as _re
    from collections import defaultdict

    # locate files
    try:
        root = _REPO_DIR
    except Exception:
        root = _Path(__file__).resolve().parents[1]
    rep_dir = root / "logs" / "reports"
    cov_path = rep_dir / "coverage.jsonl"
    out_csv  = rep_dir / "coverage_rollup.csv"
    rep_dir.mkdir(parents=True, exist_ok=True)

    if not cov_path.exists():
        return None

    def _coerce_f(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _prox_label_from_fixture(fid: str | None):
        # Group by D-tag (D2/D3) as a stable prox label
        if not fid:
            return "UNKNOWN"
        m = _re.search(r"(?:^|_)D(\d+)", fid)
        return f"D{m.group(1)}" if m else "UNKNOWN"

    # read & normalize
    rows = []
    tau_event = None
    with cov_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = _json.loads(line)
            except Exception:
                continue

            unit = j.get("unit")
            kind = j.get("kind")

            # Time(τ) health events bypass snapshot_id filter for now.
            # We always take the latest such ping in the file, independent
            # of snapshot_id, so that τ health is visible even if the ping
            # was not snapshot-tagged correctly.
            if unit == "time_tau" and kind == "tau_c4_health":
                # Latest Time(τ) health ping wins; do not treat as a mismatch row.
                try:
                    tau_event = {
                        "n_fixtures_with_c3": int(j.get("n_fixtures_with_c3", 0) or 0),
                        "n_tau_pred_true": int(j.get("n_tau_pred_true", 0) or 0),
                        "n_tau_pred_false": int(j.get("n_tau_pred_false", 0) or 0),
                        "n_tau_emp_true": int(j.get("n_tau_emp_true", 0) or 0),
                        "n_tau_emp_false": int(j.get("n_tau_emp_false", 0) or 0),
                        "tau_mismatch_count": int(j.get("tau_mismatch_count", 0) or 0),
                    }
                except Exception:
                    # Best-effort; ignore malformed Time(τ) events.
                    pass
                continue

            if snapshot_id and j.get("snapshot_id") != snapshot_id:
                continue

            fid = j.get("fixture_label")

            # map fields (v2 preferred → legacy fallbacks)
            sel_raw = j.get("mismatch_sel")
            if sel_raw is None: sel_raw = j.get("sel_mismatch_rate")
            off_raw = j.get("mismatch_offrow")
            if off_raw is None: off_raw = j.get("offrow_mismatch_rate")
            ker_raw = j.get("mismatch_ker")
            if ker_raw is None: ker_raw = j.get("ker_mismatch_rate")

            sel = _coerce_f(sel_raw)
            off = _coerce_f(off_raw)
            ker = _coerce_f(ker_raw)

            # contradiction rate: trust known verdicts; else fallback to mismatches
            vcls = (j.get("verdict_class") or "").upper().strip()
            ctr = None
            if vcls in ("GREEN", "RED_BOTH", "KER-FILTERED", "KER-EXPOSED"):
                ctr = 0.0 if vcls == "GREEN" else 1.0
            if ctr is None:
                ctr = 1.0 if ((sel or 0.0) > 0.0 or (off or 0.0) > 0.0 or (ker or 0.0) > 0.0) else 0.0

            rows.append({
                "prox_label": _prox_label_from_fixture(fid),
                "sel": sel, "off": off, "ker": ker, "ctr": ctr,
            })

    # aggregate by prox_label + ALL
    agg = defaultdict(lambda: {"count":0, "sel_sum":0.0, "sel_n":0,
                               "off_sum":0.0, "off_n":0, "ker_sum":0.0, "ker_n":0,
                               "ctr_sum":0.0, "ctr_n":0})

    for r in rows:
        for key in (r["prox_label"], "ALL"):
            a = agg[key]
            a["count"] += 1
            if r["sel"] is not None: a["sel_sum"] += r["sel"]; a["sel_n"] += 1
            if r["off"] is not None: a["off_sum"] += r["off"]; a["off_n"] += 1
            if r["ker"] is not None: a["ker_sum"] += r["ker"]; a["ker_n"] += 1
            if r["ctr"] is not None: a["ctr_sum"] += r["ctr"]; a["ctr_n"] += 1

    # ensure an ALL row exists even if no rows matched (so the chip can still render)
    if "ALL" not in agg:
        _ = agg["ALL"]

    # Attach latest Time(τ) ping (if any) to the ALL row in-memory.
    if tau_event:
        all_row = agg["ALL"]
        all_row["time_tau_n_fixtures_with_c3"] = tau_event.get("n_fixtures_with_c3", 0)
        all_row["time_tau_tau_pred_true"] = tau_event.get("n_tau_pred_true", 0)
        all_row["time_tau_tau_pred_false"] = tau_event.get("n_tau_pred_false", 0)
        all_row["time_tau_tau_emp_true"] = tau_event.get("n_tau_emp_true", 0)
        all_row["time_tau_tau_emp_false"] = tau_event.get("n_tau_emp_false", 0)
        all_row["time_tau_tau_mismatch_count"] = tau_event.get("tau_mismatch_count", 0)

    # write csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow([
            "prox_label","count",
            "mean_sel_mismatch_rate","mean_offrow_mismatch_rate","mean_ker_mismatch_rate","mean_ctr_rate",
            "time_tau_n_fixtures_with_c3",
            "time_tau_tau_pred_true",
            "time_tau_tau_pred_false",
            "time_tau_tau_emp_true",
            "time_tau_tau_emp_false",
            "time_tau_tau_mismatch_count",
        ])
        def _avg(sum_v, n): return (sum_v / n) if n > 0 else ""
        for label in sorted(agg.keys(), key=lambda x: (x!="ALL", x)):
            a = agg[label]
            # Time(τ) columns live only on the ALL row; others get empty strings.
            if label == "ALL":
                t_n_c3 = a.get("time_tau_n_fixtures_with_c3", 0)
                t_pred_true = a.get("time_tau_tau_pred_true", 0)
                t_pred_false = a.get("time_tau_tau_pred_false", 0)
                t_emp_true = a.get("time_tau_tau_emp_true", 0)
                t_emp_false = a.get("time_tau_tau_emp_false", 0)
                t_mismatch = a.get("time_tau_tau_mismatch_count", 0)
            else:
                t_n_c3 = ""
                t_pred_true = ""
                t_pred_false = ""
                t_emp_true = ""
                t_emp_false = ""
                t_mismatch = ""

            w.writerow([
                label, a["count"],
                _avg(a["sel_sum"], a["sel_n"]),
                _avg(a["off_sum"], a["off_n"]),
                _avg(a["ker_sum"], a["ker_n"]),
                _avg(a["ctr_sum"], a["ctr_n"]),
                t_n_c3,
                t_pred_true,
                t_pred_false,
                t_emp_true,
                t_emp_false,
                t_mismatch,
            ])

    return out_csv


   
# ---- v2 canonicalization (stable JSON for hashing) ----
_V2_EPHEMERAL_KEYS = {
    # runtime/UI noise we never want to affect canonical hashes
    "created_at", "created_at_utc", "updated_at", "updated_at_utc",
    "_ui_nonce", "__ui_nonce", "__nonce", "__ts",
    # convenience blobs that shouldn’t enter canonical digests
    "bundle_dir", "filenames", "counts", "paths",
}

def _v2_canonical_obj(obj, exclude_keys=_V2_EPHEMERAL_KEYS):
    """
    Recursively sanitize an object so that json.dumps(obj, sort_keys=True, separators=(',', ':'))
    is stable across runs and platforms. Drops ephemeral keys and None values.
    """
    # dict → dict (sans ephemeral/None), recurse values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in exclude_keys or v is None:
                continue
            out[str(k)] = _v2_canonical_obj(v, exclude_keys)
        return out

    # list/tuple → list (recurse)
    if isinstance(obj, (list, tuple)):
        return [_v2_canonical_obj(x, exclude_keys) for x in obj]

    # set → sorted list (stable)
    if isinstance(obj, set):
        return sorted(_v2_canonical_obj(x, exclude_keys) for x in obj)

    # pathlib.Path or path-like → posix string
    try:
        # duck-typing: pathlib objects have .as_posix()
        if hasattr(obj, "as_posix"):
            return obj.as_posix()
    except Exception:
        pass

    # bytes → hex (stable text)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()

    # primitives pass through
    return obj



def compute_ab_file(strict_or_file_payload: dict, maybe_file_or_freezer: dict, ib: dict = None) -> dict:
    """
    Flexible shim so it works with either signature you might have in the app:
      A) compute_ab_file(strict_payload, projected_file_payload, ib)
      B) compute_ab_file(projected_file_payload, freezer_payload, ib)    # (legacy)
    We do NOT read any global `strict`.
    """
    import hashlib, json as _json

    def _k3eq(p):
        return ((p or {}).get("results", {}) or {}).get("k3", {}).get("eq")

    # Try to detect which argument is which
    a_eq = _k3eq(strict_or_file_payload)
    b_eq = _k3eq(maybe_file_or_freezer)

    if a_eq is not None and b_eq is not None:
        # Looks like A) strict vs file
        k_strict, k_file = a_eq, b_eq
    elif a_eq is None and b_eq is not None:
        # Looks like B) first arg is freezer (no k3), second is file
        k_strict, k_file = None, b_eq
    elif a_eq is not None and b_eq is None:
        # First is file, second is freezer
        k_strict, k_file = None, a_eq
    else:
        # Neither carries k3 → NA on both
        k_strict = k_file = None

    embed = {"policy": "strict__VS__projected(columns@k=3,file)"}
    embed_sig = hashlib.sha256(
        _json.dumps(embed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "ab_pair": {
            "policy": embed["policy"],
            "embed_sig": embed_sig,
            "pair_vec": {"k2": [None, None], "k3": [bool(k_strict) if k_strict is not None else None,
                                                    bool(k_file)   if k_file   is not None else None]},
        }
    }

def _v2_extract_ids_from_path(bdir: _VPath):
    """
    Given a bundle dir logs/certs/{district_id}/{fixture_label}/{sig8},
    return (district_id, sig8). Falls back to (None, None) on failure.
    """
    try:
        p = _VPath(bdir)
        sig8 = p.name
        fixture_label = p.parent.name
        district = p.parent.parent.name
        if district and district.startswith("D") and len(sig8) == 8:
            return district, sig8
    except Exception:
        pass
    return None, None


def _v2_extract_lanes_from_auto(auto_fp: _VPath):
    try:
        payload = _Vjson.loads(auto_fp.read_text("utf-8"))
        pc = payload.get("projection_context") or {}
        return pc.get("lanes_popcount"), pc.get("lanes_sig8")
    except Exception:
        return None, None

def _v2_presence_mask(keys_present):
    bit_order = ["strict","projected_auto","ab_auto","freezer","ab_file","projected_file"]
    mask = 0
    for i,k in enumerate(bit_order):
        if k in keys_present:
            mask |= (1<<i)
    return f"{mask:02X}"

def _v2_bundle_index_rebuild(bdir: _VPath):
    """
    Rebuild bundle.json in logs/certs/{district_id}/{fixture_label}/{sig8}.

    files{} are the core certs we actually see on disk, so we define:
      • counts.present      = len(files)
      • core_counts.written = len(files)

    district_id is the hashed D-tag (Dxxxx...), fixture_label is D*_H*_C*.
    """
    bdir = _VPath(bdir)

    files = _v2_find_expected_files(bdir)
    hashes = {k: _v2_sha256_path(p) for k, p in files.items()}
    sizes  = {k: p.stat().st_size for k, p in files.items()}

    district_id, sig8 = _v2_extract_ids_from_path(bdir)
    try:
        fixture_label = bdir.parent.name  # logs/certs/{district}/{fixture_label}/{sig8}
    except Exception:
        fixture_label = None

    # lanes popcount / sig8 from AUTO cert if present
    lanes_pop, lanes_sig8 = (None, None)
    if "projected_auto" in files:
        try:
            lanes_pop, lanes_sig8 = _v2_extract_lanes_from_auto(files["projected_auto"])
        except Exception:
            lanes_pop, lanes_sig8 = (None, None)

    presence_mask_hex = _v2_presence_mask(files.keys())
    core_written = len(files)

    bundle = {
        "district_id": district_id,
        "fixture_label": fixture_label,
        "sig8": sig8,
        "files": {k: str(p) for k, p in files.items()},
        "hashes": hashes,
        "sizes": sizes,
        "presence_mask_hex": presence_mask_hex,
        "counts": {"present": core_written},
        "core_counts": {"written": core_written},
        "lanes": {"popcount": lanes_pop, "sig8": lanes_sig8},
    }

    (bdir / "bundle.json").write_text(
        _Vjson.dumps(bundle, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle



def _v2_write_loop_receipt(bdir: _VPath, fixture_id: str, snapshot_id: str, bundle: dict):
    """
    Back-compat shim: always write loop_receipt.v2 with SSOT-anchored absolute paths
    derived from the fixture_id (D*_H*_C*), not the hashed district_id.
    """
    from pathlib import Path as _Path
    import re as _re

    # Repo roots
    repo_root   = _Path(__file__).resolve().parents[1]   # /mount/src/d4-final-solver
    inputs_root = repo_root / "app" / "inputs"

    # fixture_id like "D3_H10_C111" → D_tag="D3", H_tag="H10", C_tag="C111"
    _mD = _re.search(r"(?:^|_)D(\d+)", fixture_id or "")
    _mH = _re.search(r"(?:^|_)H(\d+)", fixture_id or "")
    _mC = _re.search(r"(?:^|_)C(\d+)", fixture_id or "")

    D_tag = f"D{_mD.group(1)}" if _mD else (fixture_id or "").split("_")[0]
    H_tag = f"H{_mH.group(1)}" if _mH else None
    C_tag = f"C{_mC.group(1)}" if _mC else None

    # Canonical absolute SSOT paths
    P = {
        "B": str((inputs_root / "B" / f"{D_tag}.json").resolve()),            # <-- B from D_tag
        "C": str((inputs_root / "C" / f"{C_tag}.json").resolve()) if C_tag else None,
        "H": str((inputs_root / "H" / f"{H_tag}.json").resolve()) if H_tag else None,
        "U": str((inputs_root / "U.json").resolve()),
    }

    # dims (optional)
    dims = bundle.get("dims")
    if not dims:
        n2, n3 = bundle.get("n2"), bundle.get("n3")
        if n2 is not None and n3 is not None:
            dims = {"n2": n2, "n3": n3}

    extra = {
        "schema":        "loop_receipt.v2",
        "district_id":   bundle.get("district_id"),
        "fixture_label": fixture_id,
        "sig8":          bundle.get("sig8"),
        "paths":         P,
    }
    if dims:
        extra["dims"] = dims

    ok, msg = _v2_write_loop_receipt_for_bundle(_Path(bdir), extra=extra)
    return {"ok": ok, "msg": msg}







def _solver_ret_as_tuple(ret):
    try:
        if isinstance(ret, (tuple, list)):
            ok  = bool(ret[0]) if len(ret)>=1 else False
            msg = str(ret[1]) if len(ret)>=2 else ""
            bdir = ret[2] if len(ret)>=3 else None
            return ok, msg, bdir
        if isinstance(ret, dict):
            ok  = bool(ret.get("ok", ret.get("success", True)))
            msg = str(ret.get("msg", ret.get("message", "")))
            bdir = ret.get("bundle_dir") or ret.get("bundle") or ret.get("dir")
            return ok, msg, bdir
        if isinstance(ret, bool):
            return ret, "", None
    except Exception:
        pass
    return False, "solver returned unrecognized payload", None


    
def _as3(ret):
    """Normalize any (ok,msg,count?) shape to exactly (ok:bool, msg:str, count:int)."""
    if isinstance(ret, (tuple, list)):
        if len(ret) >= 3: return bool(ret[0]), str(ret[1]), int(ret[2])
        if len(ret) == 2: return bool(ret[0]), str(ret[1]), 0
        if len(ret) == 1: return bool(ret[0]), "", 0
    if isinstance(ret, dict):
        ok  = bool(ret.get("ok", ret.get("success", False)))
        msg = str(ret.get("msg", ret.get("message", "")))
        n   = int(ret.get("count", ret.get("n", 0)) or 0)
        return ok, msg, n
    if isinstance(ret, bool):
        return ret, "", 0
    return False, "runner returned unexpected shape", 0
    


# ===== Helper: lanes sig8 + suite message (always defined) =====
def _lanes_sig8_from_list(L):
    import hashlib as _hashlib
    try:
        b = bytearray(); acc = 0; bit = 0
        for v in (int(x) for x in (L or [])):
            if v: acc |= (1 << bit)
            bit += 1
            if bit == 8:
                b.append(acc); acc = 0; bit = 0
        if bit: b.append(acc)
        return _hashlib.sha256(bytes(b)).hexdigest()[:8]
    except Exception:
        return None


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
import uuid, streamlit as st
import json as _json
from pathlib import Path as _Path
import hashlib as _hash
# Page config must be the first Streamlit command
SCHEMA_VERSION = "2.0.0"
ENGINE_REV     = "rev-20251022-1"

DIRS = {"root": "logs", "certs": "logs/certs", "snapshots": "logs/snapshots", "reports": "logs/reports", "suite_runs": "logs/suite_runs", "exports": "logs/exports"}



def _svr_current_snapshot_id() -> str | None:
    try:
        snaps_dir = _Path(DIRS.get("snapshots", "logs/snapshots"))
        ptr = snaps_dir / "world_snapshot.latest.json"
        if not ptr.exists():
            return None
        import json as _json
        data = _json.loads(ptr.read_text(encoding="utf-8"))
        return data.get("snapshot_id")
    except Exception:
        return None



def _suite_index_paths():
    base = _Path(DIRS.get("suite_runs", "logs/suite_runs"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "suite_index.jsonl", base / "suite_index.csv"

def _suite_index_add_row(row: dict) -> None:
    # append to jsonl and csv (create csv header if missing)
    import csv as _csv, json as _json
    jl, csvp = _suite_index_paths()
    with jl.open("a", encoding="utf-8") as jf:
        jf.write(_canonical_json(row) + "\n")
    header = list(row.keys())
    write_header = not csvp.exists()
    with csvp.open("a", newline="", encoding="utf-8") as cf:
        w = _csv.DictWriter(cf, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

# ---------- Suite helpers (v2) ----------

from pathlib import Path as _Path
import json as _json, hashlib as _hashlib, time as _time
import streamlit as _st

_APP_DIR  = _Path(__file__).resolve().parent         # .../app
_REPO_DIR = _APP_DIR.parent                          # repo root

def _abs_from_manifest(p_str: str) -> _Path:
    """
    Resolve manifest paths robustly:
      - absolute paths: return as-is
      - 'app/…'        : resolve from REPO root
      - everything else: resolve from APP dir
    """
    p = _Path(p_str)
    if p.is_absolute():
        return p
    s = str(p).replace("\\", "/").lstrip("./")
    if s.startswith("app/"):
        return (_REPO_DIR / s).resolve()
    return (_APP_DIR / s).resolve()

def _set_inputs_for_run(B, C, H, U):
    """
    Prime the single-press pipeline with file paths the same way
    the upload widgets would. We set both legacy and current keys
    to be safe.
    """
    ss = _st.session_state
    Bp, Cp, Hp, Up = map(lambda q: str(_abs_from_manifest(q)), (B, C, H, U))

    # Primary keys the resolver uses
    ss["uploaded_boundaries"] = Bp
    ss["uploaded_cmap"]       = Cp
    ss["uploaded_H"]          = Hp
    ss["uploaded_shapes"]     = Up

    # Common synonyms seen across earlier wiring
    ss["uploaded_B_path"] = Bp
    ss["uploaded_C_path"] = Cp
    ss["uploaded_H_path"] = Hp
    ss["uploaded_U_path"] = Up





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
# UI nonce for unique widget keys (only define once near the top)

ss = st.session_state
if "_ui_nonce" not in ss:
    ss["_ui_nonce"] = uuid.uuid4().hex[:8]


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



# =============================== TOP HELPERS — CANONICAL ===============================
# This block replaces the previous duplicate helpers. Single Source of Truth (SSOT).

# --- Imports expected to be available ---
# import os, json, time, uuid, shutil, tempfile, hashlib
# from pathlib import Path
# import streamlit as st
# from datetime import datetime, timezone

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

def canonical_json(obj) -> str:
    """Canonical JSON string for hashing (bools → 0/1, sorted keys, tight separators).

    v2 rule:
      - Drop ephemeral keys / normalize structure via _v2_canonical_obj.
      - Convert bools to 0/1 and recurse through lists/dicts via _deep_intify.
      - Dump with sorted keys, tight separators, ensure_ascii=True.
    """
    try:
        can = _v2_canonical_obj(obj)
    except NameError:
        # Fallback: if _v2_canonical_obj is not defined in some contexts, use obj directly
        can = obj
    return json.dumps(
        _deep_intify(can),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

def hash_json(obj) -> str:
    """Stable SHA-256 over a JSON-serializable object using canonical_json()."""
    s = canonical_json(obj)
    return _sha256_hex_text(s)

def hash_json_sig8(obj) -> str:
    """Short 'sig8' (first 8 hex chars) of hash_json(obj)."""
    return hash_json(obj)[:8]

# ------------------------- Fixture Label Helpers (v2 naming) -------------------------
def make_fixture_label(district_id: str, h_mask: str, c_mask: str) -> str:
    """Build canonical fixture label like 'D3_H10_C111' from pieces.

    Rules:
      - district_id is normalized to start with 'D' (e.g. '3' -> 'D3').
      - h_mask and c_mask are coerced to strings (no padding assumptions here).
    """
    # Normalize district_id to include leading 'D'
    d = str(district_id)
    if not d.startswith("D"):
        d = f"D{d}"
    # Coerce masks to strings
    h = str(h_mask)
    c = str(c_mask)
    return f"{d}_H{h}_C{c}"

def parse_fixture_label(label: str) -> tuple[str, str, str]:
    """Parse a canonical fixture label 'D3_H10_C111' into (district_id, h_mask, c_mask).

    This is intentionally strict: if the label does not match the expected pattern,
    a ValueError is raised so callers don't silently proceed with a malformed id.
    """
    import re as _re
    s = str(label or "").strip()
    m = _re.fullmatch(r"D(\d+)_H([01]+)_C([01]+)", s)
    if not m:
        raise ValueError(f"invalid fixture_label format: {label!r}")
    d_num, h_mask, c_mask = m.groups()
    district_id = f"D{d_num}"
    return district_id, h_mask, c_mask

# ------------------------- V2 Path Helpers (certs / bundles / receipts) -------------------------
def _v2_certs_root() -> _Path:
    """Canonical root for v2 cert bundles.

    Prefer _CERTS_DIR if defined (repo_root/logs/certs); otherwise fall back to ./logs/certs.
    """
    try:
        return _CERTS_DIR  # type: ignore[name-defined]
    except NameError:
        return _Path("logs") / "certs"

def make_bundle_dir(
    district_id: str,
    fixture_label: str,
    sig8: str,
    *,
    certs_root: _Path | None = None,
) -> _Path:
    """Build canonical bundle_dir = certs_root/D*/fixture_label/sig8.

    - district_id and fixture_label are kept as strings (no normalization here;
      callers should use make_fixture_label/parse_fixture_label as needed).
    - certs_root defaults to the global v2 certs root (repo_root/logs/certs).
    """
    base = certs_root if certs_root is not None else _v2_certs_root()
    return base / str(district_id) / str(fixture_label) / str(sig8)

def make_strict_cert_path(bundle_dir, district_id: str, sig8: str):
    """Return Path to the strict cert JSON inside a bundle dir.

    Pattern: overlap__{district_id}__strict__{sig8}.json
    """
    b = _Path(bundle_dir)
    return b / f"overlap__{district_id}__strict__{sig8}.json"

def make_projected_auto_cert_path(bundle_dir, district_id: str, sig8: str):
    """Return Path to the AUTO projected cert JSON inside a bundle dir.

    Pattern: overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json
    """
    b = _Path(bundle_dir)
    return b / f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json"

def make_projected_file_cert_path(bundle_dir, district_id: str, sig8: str):
    """Return Path to the FILE projected cert JSON inside a bundle dir.

    Pattern: overlap__{district_id}__projected_columns_k_3_file__{sig8}.json
    """
    b = _Path(bundle_dir)
    return b / f"overlap__{district_id}__projected_columns_k_3_file__{sig8}.json"

def make_loop_receipt_path(bundle_dir, fixture_label: str):
    """Return Path to the loop_receipt file for a fixture inside a bundle dir.

    Current pattern: loop_receipt__{fixture_label}.json
    """
    b = _Path(bundle_dir)
    return b / f"loop_receipt__{fixture_label}.json"

# ------------------------- V2 Artifact Builders (schemas & invariants) -------------------------

def _v2_default_schema_version() -> str:
    """Best-effort lookup for the global v2 schema version."""
    return str(globals().get("SCHEMA_VERSION", "2.0.0"))


def _v2_default_engine_rev() -> str:
    """Best-effort lookup for the global engine revision."""
    return str(globals().get("ENGINE_REV", "rev-UNSET"))


def build_v2_cert_payload(
    *,
    mode: str,
    district_id: str,
    fixture_label: str,
    inputs: dict | None = None,
    solver: dict | None = None,
    payload: dict | None = None,
    schema_version: str | None = None,
    engine_rev: str | None = None,
    kind: str = "cert_v2",
) -> dict:
    """Build a canonical v2 cert payload for strict / projected certs.

    This is intentionally header-agnostic: callers may still use _svr_cert_common /
    _svr_build_embed to construct richer headers and then merge them into `payload`
    upstream. Here we enforce only the outer envelope + sig8 convention.

    Invariants:
      - schema_version / engine_rev default from globals.
      - district_id / fixture_label are stored as strings.
      - sig8 is computed via hash_json_sig8 over the body *excluding* any existing sig8.
    """

    district_id = str(district_id)
    fixture_label = str(fixture_label)

    # Light sanity check: if fixture_label parses cleanly, prefer its D-tag
    try:
        d_from_label, _, _ = parse_fixture_label(fixture_label)
        if d_from_label.startswith("D") and district_id.startswith("D") and d_from_label != district_id:
            # Do not raise here; just keep the explicit district_id and allow callers to tighten later.
            pass
    except Exception:
        # If label is not canonical yet, skip the check.
        pass

    sv = str(schema_version or _v2_default_schema_version())
    ev = str(engine_rev or _v2_default_engine_rev())

    body: dict = {
        "schema_version": sv,
        "engine_rev": ev,
        "kind": str(kind),
        "mode": str(mode),
        "district_id": district_id,
        "fixture_label": fixture_label,
        "inputs": dict(inputs or {}),
        "solver": dict(solver or {}),
        "payload": dict(payload or {}),
    }

    sig_body = dict(body)
    sig_body.pop("sig8", None)
    body["sig8"] = hash_json_sig8(sig_body)
    return body


def build_v2_bundle_manifest(
    *,
    district_id: str,
    fixture_label: str,
    sig8: str,
    snapshot_id: str | None = None,
    bundle_dir: str | None = None,
    presence_mask_hex: str | None = None,
    core_counts: dict | None = None,
    sizes: dict | None = None,
    hashes: dict | None = None,
    dims: dict | None = None,
    written_at_utc: int | None = None,
    schema_version: str | None = None,
    engine_rev: str | None = None,
    kind: str = "bundle_v2",
    extra: dict | None = None,
) -> dict:
    """Build a canonical v2 bundle manifest payload (for bundle.json).

    This captures only the structural invariants we care about at v2; callers are free
    to extend the `extra` dict for UI-only fields.

    """

    district_id = str(district_id)
    fixture_label = str(fixture_label)

    sv = str(schema_version or _v2_default_schema_version())
    ev = str(engine_rev or _v2_default_engine_rev())

    bundle: dict = {
        "schema_version": sv,
        "engine_rev": ev,
        "kind": str(kind),
        "district_id": district_id,
        "fixture_label": fixture_label,
        "sig8": str(sig8),
        "snapshot_id": str(snapshot_id) if snapshot_id is not None else None,
        "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
        "presence_mask_hex": presence_mask_hex,
        "core_counts": dict(core_counts or {}),
        "sizes": dict(sizes or {}),
        "hashes": dict(hashes or {}),
        "written_at_utc": int(written_at_utc) if written_at_utc is not None else None,
    }

    if dims is not None:
        bundle["dims"] = dict(dims)
    if extra:
        bundle.update(extra)

    return bundle


def build_v2_loop_receipt(
    *,
    district_id: str,
    fixture_label: str,
    sig8: str,
    bundle_dir,
    paths: dict,
    snapshot_id: str | None = None,
    run_id: str | None = None,
    core_counts: dict | None = None,
    timestamps: dict | None = None,
    dims: dict | None = None,
    schema: str = "loop_receipt.v2",
    extra: dict | None = None,
) -> dict:
    """Build a canonical loop_receipt.v2 payload.

    Invariants:

      - schema is fixed to loop_receipt.v2 (unless explicitly overridden).

      - district_id / fixture_label / bundle_dir are strings.

      - sig8 is the underlying cert/embed sig8, *not* a self-hash.

      - receipt_sig8 is a self-hash over the receipt body (excluding receipt_sig8).

    """

    district_id = str(district_id)
    fixture_label = str(fixture_label)

    receipt: dict = {
        "schema": str(schema),
        "snapshot_id": str(snapshot_id) if snapshot_id is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "district_id": district_id,
        "fixture_label": fixture_label,
        "sig8": str(sig8),
        "bundle_dir": str(bundle_dir),
        "paths": dict(paths or {}),
        "core_counts": dict(core_counts or {}),
        "timestamps": dict(timestamps or {}),
    }

    if dims is not None:
        receipt["dims"] = dict(dims)
    if extra:
        receipt.update(extra)

    # Stable self-hash over the structural receipt body
    sig_body = dict(receipt)
    sig_body.pop("receipt_sig8", None)
    receipt["receipt_sig8"] = hash_json_sig8(sig_body)
    return receipt


def build_v2_world_snapshot_from_body(body_without_id: dict) -> dict:
    """Attach snapshot_id to a world_snapshot body using v2 hashing rules.

    The caller is responsible for assembling body_without_id with:

      - schema_version, engine_rev

      - manifests (paths to manifests)

      - inventory (SSOT hashes)

      - plan_full_scope (64-row plan)

    We simply ensure schema/engine defaults and inject snapshot_id.

    """

    body = dict(body_without_id or {})

    body.setdefault("schema_version", _v2_default_schema_version())
    body.setdefault("engine_rev", _v2_default_engine_rev())

    sig_body = dict(body)
    sig_body.pop("snapshot_id", None)
    snapshot_id = "ws__" + hash_json_sig8(sig_body)
    body["snapshot_id"] = snapshot_id
    return body


def build_v2_suite_row(
    *,
    snapshot_id: str,
    fixture_label: str,
    bundle_dir,
    lanes_popcount: int | None = None,
    lanes_sig8: str | None = None,
    district_id: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a canonical suite_index row for a v2 fixture run.

    This is a small, parseable summary we can safely append to suite_index.jsonl/csv.

    """

    fixture_label = str(fixture_label)
    snapshot_id = str(snapshot_id)
    row: dict = {
        "snapshot_id": snapshot_id,
        "fixture_label": fixture_label,
        "bundle_dir": str(bundle_dir),
    }

    if district_id is not None:
        row["district_id"] = str(district_id)
    if lanes_popcount is not None:
        row["lanes_popcount"] = int(lanes_popcount)
    if lanes_sig8 is not None:
        row["lanes_sig8"] = str(lanes_sig8)
    if extra:
        row.update(extra)
    return row






# =============================== V2 Artifact Builders (pure constructors) ===============================
def build_v2_cert_base_header(
    *,
    district_id: str,
    fixture_label: str,
    snapshot_id: str | None,
    inputs_sig_5: list[str] | tuple[str, ...],
    schema_version: str | None = None,
    engine_rev: str | None = None,
) -> dict:
    """Construct the common header for all v2 cert-like payloads.

    This is a pure helper: it does not touch the filesystem or session state.
    """
    schema = str(schema_version or globals().get("SCHEMA_VERSION", "2.0.0"))
    engine = str(engine_rev or globals().get("ENGINE_REV", "rev-UNSET"))
    return {
        "schema_version": schema,
        "engine_rev": engine,
        "district_id": str(district_id),
        "fixture_label": str(fixture_label),
        # fixture_id is the same as fixture_label in v2:
        "fixture_id": str(fixture_label),
        "snapshot_id": snapshot_id,
        "inputs_sig_5": list(inputs_sig_5 or []),
    }


def build_v2_strict_cert_payload(base_hdr: dict, verdict: bool | None) -> dict:
    """Strict cert payload from a base header and solver verdict.

    Shape:
      {
        ...base_hdr,
        "policy": "strict",
        "verdict": true/false/null,
      }
    """
    payload = dict(base_hdr or {})
    payload["policy"] = "strict"
    payload["verdict"] = bool(verdict) if isinstance(verdict, bool) else (verdict if verdict is None else bool(verdict))
    return payload


def build_v2_projected_auto_cert_payload(
    base_hdr: dict,
    *,
    lanes_vec: list[int] | None,
    lanes_popcount: int | None,
    lanes_sig8: str | None,
    proj_meta: dict | None,
    verdict: bool | None,
) -> dict:
    """AUTO projected cert payload from header + lanes + projection metadata + verdict."""
    payload = dict(base_hdr or {})
    payload["policy"] = "projected(columns@k=3,auto)"
    ctx = {
        "lanes": [int(x) & 1 for x in (lanes_vec or [])],
        "lanes_popcount": int(lanes_popcount or 0),
        "lanes_sig8": lanes_sig8,
    }
    payload["projection_context"] = ctx
    na = bool((proj_meta or {}).get("na")) if isinstance(proj_meta, dict) else False
    reason = (proj_meta or {}).get("reason") if isinstance(proj_meta, dict) else None
    payload["na"] = na
    payload["reason"] = reason
    payload["verdict"] = bool(verdict) if isinstance(verdict, bool) else (verdict if verdict is None else bool(verdict))
    return payload


def build_v2_projected_file_cert_payload(base_hdr: dict) -> dict:
    """FILE projected cert payload from a base header.

    v2 semantics: we only assert presence + cross-refs; verdict is left null.
    """
    payload = dict(base_hdr or {})
    payload["policy"] = "projected(columns@k=3,file)"
    payload["verdict"] = None
    return payload


def build_v2_ab_compare_payload(
    base_hdr: dict,
    *,
    policy: str,
    left_policy: str,
    left_sig8: str,
    right_policy: str,
    right_sig8: str,
) -> dict:
    """A/B compare payload for strict vs projected certs.

    Used for both AUTO and FILE A/B comparisons.
    """
    payload = dict(base_hdr or {})
    payload["policy"] = str(policy)
    payload["embed"] = {
        "left": {"policy": str(left_policy), "sig8": str(left_sig8)},
        "right": {"policy": str(right_policy), "sig8": str(right_sig8)},
    }
    return payload


def build_v2_projector_freezer_payload(
    base_hdr: dict,
    *,
    file_pi_valid: bool,
    file_pi_reasons: list[str] | None,
) -> dict:
    """Payload for projector_freezer sidecar cert."""
    payload = dict(base_hdr or {})
    payload["policy"] = "projector_freezer"
    payload["status"] = "OK" if file_pi_valid else "FAIL"
    payload["file_pi_valid"] = bool(file_pi_valid)
    payload["file_pi_reasons"] = list(file_pi_reasons or [])
    return payload


def build_v2_bundle_index_payload(
    *,
    run_id: str,
    sig8: str,
    district_id: str,
    bundle_dir,
    filenames: list[str],
    lanes_popcount: int | None,
    lanes_sig8: str | None,
    file_pi_valid: bool,
) -> dict:
    """Bundle index (bundle.json) payload.

    This only assembles the structure; callers are responsible for writing it.
    """
    bdir = _Path(bundle_dir)
    files = {
        "strict": str(make_strict_cert_path(bdir, district_id, sig8)),
        "projected_auto": str(make_projected_auto_cert_path(bdir, district_id, sig8)),
        "ab_auto": str(bdir / f"ab_compare__strict_vs_projected_auto__{sig8}.json"),
        "freezer": str(bdir / f"projector_freezer__{district_id}__{sig8}.json"),
        "projected_file": None,
        "ab_file": None,
    }
    if file_pi_valid:
        files["projected_file"] = str(make_projected_file_cert_path(bdir, district_id, sig8))
        files["ab_file"] = str(bdir / f"ab_compare__strict_vs_projected_file__{sig8}.json")
    return {
        "run_id": str(run_id or ""),
        "sig8": str(sig8 or ""),
        "district_id": str(district_id),
        "filenames": list(filenames or []),
        "files": files,
        "lanes": {"popcount": int(lanes_popcount or 0), "sig8": lanes_sig8},
        "counts": {"written": len(filenames or [])},
    }


def build_v2_loop_receipt_payload(
    *,
    run_id: str | None,
    district_id: str,
    fixture_label: str,
    sig8: str,
    bundle_dir,
    paths: dict,
    core_written: int,
    dims: dict | None = None,
    receipt_written_at: int | None = None,
) -> dict:
    """Pure constructor for loop_receipt.v2 payloads."""
    rec: dict = {
        "schema": "loop_receipt.v2",
        "run_id": run_id,
        "district_id": str(district_id),
        "fixture_label": str(fixture_label),
        "sig8": str(sig8),
        "bundle_dir": str(_Path(bundle_dir).resolve()),
        "paths": dict(paths or {}),
        "core_counts": {"written": int(core_written)},
    }
    if isinstance(dims, dict):
        rec["dims"] = {"n2": dims.get("n2"), "n3": dims.get("n3")}
    if receipt_written_at is not None:
        rec["timestamps"] = {"receipt_written_at": int(receipt_written_at)}
    return rec


def build_v2_manifest_row(
    fixture_label: str,
    *,
    paths: dict,
    dims: dict | None = None,
) -> dict:
    """Minimal manifest_full_scope row as used by v2 regeneration logic."""
    row: dict = {
        "fixture_label": str(fixture_label),
        "paths": {
            "B": str((paths or {}).get("B", "")),
            "C": str((paths or {}).get("C", "")),
            "H": str((paths or {}).get("H", "")),
            "U": str((paths or {}).get("U", "")),
        },
    }
    if isinstance(dims, dict):
        row["dims"] = {"n2": dims.get("n2"), "n3": dims.get("n3")}
    return row


def build_v2_world_snapshot_payload(
    *,
    manifest_path,
    inventory: list[dict],
    plan_full_scope: list[dict],
    created_at_utc: str | None = None,
    schema_version: str | None = None,
    engine_rev: str | None = None,
) -> dict:
    """Pure payload for world_snapshot.

    snapshot_id is *not* included here; callers attach it after hashing.
    """
    if created_at_utc is None:
        try:
            import datetime as _dt
            created_at_utc = _dt.datetime.utcnow().isoformat() + "Z"
        except Exception:
            created_at_utc = ""
    schema = str(schema_version or globals().get("SCHEMA_VERSION", "2.0.0"))
    engine = str(engine_rev or globals().get("ENGINE_REV", "rev-UNSET"))
    return {
        "schema_version": schema,
        "engine_rev": engine,
        "created_at_utc": created_at_utc,
        "manifests": {"full_scope": str(manifest_path)},
        "inventory": list(inventory or []),
        "plan_full_scope": list(plan_full_scope or []),
    }


def build_v2_world_snapshot_id(payload: dict) -> str:
    """Compute content-addressed snapshot_id for a world_snapshot payload."""
    # We deliberately use the canonical v2 JSON hasher here, ignoring snapshot_id itself.
    sig8 = hash_json_sig8(payload)
    return f"ws__{sig8}"

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


# ───────────────────────── Minimal tab scaffold (temporary) ─────────────────────────
# We still have legacy blocks like:
#   with tab2:  # Overlap
#   with tab3:  # Triangle
#   with tab4:  # Towers
#   with tab5:  # Export
# To avoid NameError while we refactor, create tabs (or fall back to plain containers).

if not all(name in globals() for name in ("tab1", "tab2", "tab3", "tab4", "tab5")):
    try:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Unit (legacy)", "Overlap", "Triangle", "Towers", "Export"]
        )
    except Exception:
        # Extremely defensive fallback: use plain containers if tabs fail for any reason.
        tab1 = st.container()
        tab2 = st.container()
        tab3 = st.container()
        tab4 = st.container()
        tab5 = st.container()






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
def _guarded_atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# accept path / UploadedFile / dict
def _is_uploaded_file(x): return hasattr(x, "getvalue") and hasattr(x, "name")

def abx_read_json_any(x, *, kind: str):
    """
    Return (json_obj, canonical_path_str, origin) where origin ∈ {"file","upload","dict",""}
    """
    if not x: return {}, "", ""
    # path-like
    try:
        if isinstance(x, (str, Path)):
            p = Path(x)
            if p.exists():
                return _json.loads(p.read_text(encoding="utf-8")), str(p), "file"
    except Exception:
        pass
    # UploadedFile
    if _is_uploaded_file(x):
        try:
            raw = x.getvalue()
            text = raw.decode("utf-8")
            j = _json.loads(text)
        except Exception:
            return {}, "", ""
        h12  = _hashlib.sha256(raw).hexdigest()[:12]
        base = Path(getattr(x, "name", f"{kind}.json")).name
        outp = UPLOADS_DIR / f"{kind}__{h12}__{base}"
        try: outp.write_text(text, encoding="utf-8")
        except Exception: pass
        return j, str(outp), "upload"
    # raw dict
    if isinstance(x, dict):
        return x, "", "dict"
    return {}, "", ""

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

# session precedence → source object
# --- robust blocks normalizer (handles dicts and parsed cmap objects) ---
def _svr_as_blocks_v2(j: object, kind: str) -> dict:
    """
    Return canonical {"1","2","3"} blocks dict for B/C/H/U.
    Accepts:
      - plain dict with {"blocks": {...}} or top-level {"1","2","3"}
      - cmap-like objects having .blocks.__root__
    Never writes; tolerant to shapes.
    """
    # 0) None / falsey
    if j is None:
        return {}
    # 1) cmap-like object
    try:
        if hasattr(j, "blocks") and hasattr(j.blocks, "__root__"):
            root = j.blocks.__root__
            if isinstance(root, dict):
                return dict(root)
    except Exception:
        pass
    # 2) plain dicts
    if isinstance(j, dict):
        try:
            if isinstance(j.get("blocks"), dict):
                return dict(j["blocks"])
        except Exception:
            pass
        # tolerate legacy degrees at top-level
        blk = {}
        for deg in ("1","2","3"):
            if deg in j and isinstance(j[deg], list):
                blk[deg] = j[deg]
        if blk:
            return blk
    # 3) last resort: try to parse via io.parse_cmap if available
    try:
        if "io" in globals() and hasattr(io, "parse_cmap"):
            o = io.parse_cmap(j)  # type: ignore[arg-type]
            if hasattr(o, "blocks") and hasattr(o.blocks, "__root__"):
                root = o.blocks.__root__
                if isinstance(root, dict):
                    return dict(root)
    except Exception:
        pass
    return {}


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

def _svr_resolve_all_to_paths():
    """
    Read B/C/H/U per uploads-first order, normalize, validate B[3], C[3], H[2] non-empty.
    Persist to logs/_uploads if needed. Return {"B":(path,blocks), ...}.
    """
    out, raw = {}, {}
    for kind, base in (("B","boundaries.json"),("C","cmap.json"),("H","H.json"),("U","shapes.json")):
        src = _svr_pick_source(kind)
        j, p, _ = abx_read_json_any(src, kind={"B":"boundaries","C":"cmap","H":"H","U":"shapes"}[kind])
        blocks = _svr_as_blocks_v2(j, kind)
        raw[kind] = {"p": p, "blocks": blocks, "base": base}

    # validate required slices before persisting
    bB = raw["B"]["blocks"]; bC = raw["C"]["blocks"]; bH = raw["H"]["blocks"]
    d3 = bB.get("3") or []
    C3 = bC.get("3") or []
    H2 = bH.get("2") or []
    reasons = []
    if not (d3 and d3[0]): reasons.append("B[3] empty")
    if not (C3 and C3[0]): reasons.append("C[3] empty")
    if not (H2 and H2[0]): reasons.append("H[2] empty")
    if reasons:
        raise RuntimeError("INPUTS_INCOMPLETE: " + ", ".join(reasons))

    # persist canonical jsons (only if no path yet)
    for kind in ("B","C","H","U"):
        blocks = raw[kind]["blocks"]; base = raw[kind]["base"]
        p = raw[kind]["p"]
        if not p:
            canon = _json.dumps({"blocks": blocks}, separators=(",", ":"), sort_keys=True)
            h12 = _hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]
            pth = UPLOADS_DIR / f"{kind.lower()}__{h12}__{base}"
            _guarded_atomic_write_json(pth, {"blocks": blocks})
            p = str(pth)
        out[kind] = (p, blocks)
        # also expose canonical filename back to session for future runs
        st.session_state[{"B":"fname_boundaries","C":"fname_cmap","H":"fname_h","U":"fname_shapes"}[kind]] = p
    # Fallback: if H[2] empty but a parsed H exists in session, try that (read-only)
    if (not H2 or not H2[0]) and "overlap_H" in st.session_state:
        try:
            h_obj = st.session_state.get("overlap_H")
            h_blocks = _svr_as_blocks_v2(h_obj, "H")
            if h_blocks.get("2"):
                raw["H"]["blocks"] = h_blocks
                H2 = h_blocks.get("2") or []
        except Exception:
            pass

    return out

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






# cert scaffold (v2 header; no integrity payload)
if "_svr_cert_common" not in globals():
    def _svr_cert_common(ib, rc, policy_tag: str, extra: dict | None = None) -> dict:
        """
        Canonical v2 cert header helper.

        - Pulls schema/engine from globals (SCHEMA_VERSION, ENGINE_REV)
        - Uses district_id from ib/rc
        - Treats fixture_label as canonical and keeps fixture_id as alias
        - Attaches snapshot_id and SSOT 5-hash (inputs_sig_5)
        - Leaves sig8 to be filled by _svr_apply_sig8 later.
        """
        ib = ib or {}
        rc = rc or {}

        schema_version = str(globals().get("SCHEMA_VERSION", "2.0.0"))
        engine_rev = str(globals().get("ENGINE_REV", ""))

        district_id = str(
            ib.get("district_id")
            or rc.get("district_id")
            or "DUNKNOWN"
        )

        fixture_label = str(
            ib.get("fixture_label")
            or ib.get("fixture_id")
            or rc.get("fixture_label")
            or rc.get("fixture_id")
            or ""
        )

        snapshot_id = str(
            ib.get("snapshot_id")
            or rc.get("snapshot_id")
            or ""
        )

        try:
            inputs_sig_5 = _frozen_inputs_sig_from_ib(ib, as_tuple=False)
        except Exception:
            inputs_sig_5 = list(ib.get("inputs_sig_5") or [])

        base_hdr = {
            "schema_version": schema_version,
            "engine_rev": engine_rev,
            "district_id": district_id,
            # keep fixture_id for backwards compat, but prefer fixture_label
            "fixture_id": fixture_label,
            "fixture_label": fixture_label,
            "snapshot_id": snapshot_id,
            "inputs_sig_5": list(inputs_sig_5),
            "policy": str(policy_tag),
        }

        run_id = rc.get("run_id")
        if run_id:
            base_hdr["run_id"] = str(run_id)

        if extra:
            base_hdr.update(extra)
        return base_hdr

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








    
                                  
                    




# ========================= Solver entrypoint (v2: emit baseline certs) =========================
def run_overlap_once(ss=st.session_state):
    """
    v2 mechanical writer (stable anchor):
      • Resolves inputs & freezes SSOT
      • Computes strict + projected(AUTO) summaries
      • Builds embed for AUTO using {fixture_id, snapshot_id, inputs_sig_5, lanes_vec}
      • Writes 4 core certs (+2 FILE certs when Π valid)
      • Updates bundle.json and returns a small receipt (dict)
    """
    import json as _json
    from pathlib import Path as _Path

    # --- Resolve inputs and freeze SSOT ---
    pf = _svr_resolve_all_to_paths()  # {"B": (path, blocks), "C": ..., "H": ..., "U": ...}
    (pB, bB), (pC, bC), (pH, bH), (pU, bU) = pf["B"], pf["C"], pf["H"], pf["U"]
    ib_rc = _svr_freeze_ssot(pf)  # ib (inputs bundle), rc (run context)
    if isinstance(ib_rc, tuple):
        ib = ib_rc[0] or {}
        rc = ib_rc[1] if (len(ib_rc) > 1 and isinstance(ib_rc[1], dict)) else {}
    else:
        ib, rc = (ib_rc or {}), {}

        # --- District / fixture / snapshot anchors ---
    district_id = str(ib.get("district_id") or "DUNKNOWN")
    fixture_id = str(
        ss.get("fixture_label") or ib.get("fixture_label") or ib.get("fixture_id") or "UNKNOWN_FIXTURE"
    )
    snapshot_id = str(
        ib.get("snapshot_id") or ss.get("world_snapshot_id") or "UNKNOWN_SNAPSHOT"
    )

    # --- Canonical SSOT 5-hash signature (inputs_sig_5) ---
    try:
        inputs_sig_5 = _frozen_inputs_sig_from_ib(ib, as_tuple=False)
    except Exception:
        inputs_sig_5 = list(ib.get("inputs_sig_5") or [])
    ib["inputs_sig_5"] = list(inputs_sig_5)

    # --- Touch coverage file (C1 preflight) ---
    try:
        COVERAGE_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(COVERAGE_JSONL, "a", encoding="utf-8"):
            pass
    except Exception:
        pass

    # --- Strict & Projected(AUTO) (shape-safe) ---
    strict_out = _svr_strict_from_blocks(bH, bB, bC)
    proj_meta, lanes, proj_out = _svr_projected_auto_from_blocks(bH, bB, bC)

    # --- Build a lanes vector (0/1 list) for embed + record popcount/sig8 for certs ---
    lanes_vec = None
    try:
        if isinstance(lanes, (list, tuple)):
            lanes_vec = [int(x) & 1 for x in lanes]
        elif isinstance(lanes, dict):
            for k in ("mask", "vec", "lanes", "bits"):
                if isinstance(lanes.get(k), (list, tuple)):
                    lanes_vec = [int(x) & 1 for x in lanes[k]]
                    break
        if lanes_vec is None:
            C3 = bC.get("3") or []
            if C3 and isinstance(C3[-1], list):
                lanes_vec = [int(x) & 1 for x in C3[-1]]
    except Exception:
        lanes_vec = None

    lanes_pop = int(sum(lanes_vec)) if lanes_vec else 0
    try:
        _lanes_raw = _json.dumps(lanes_vec or [], separators=(",", ":"), sort_keys=True).encode("utf-8")
        lanes_sig8 = _hash.sha256(_lanes_raw).hexdigest()[:8]
    except Exception:
        lanes_sig8 = None

    # --- Canonical embed for AUTO pair → sig8 (bundle anchor) ---
    na_reason = (proj_meta.get("reason") if (isinstance(proj_meta, dict) and proj_meta.get("na")) else None)
    # ensure fixture metadata + SSOT sig live on ib for embedding
    fixture_label = fixture_id
    ib["district_id"] = district_id
    ib["fixture_label"] = fixture_label
    ib["snapshot_id"] = snapshot_id
    ib["inputs_sig_5"] = list(inputs_sig_5) if isinstance(inputs_sig_5, (list, tuple)) else []
    embed_auto, embed_sig_auto = _svr_build_embed(
        ib,
        "strict__VS__projected(columns@k=3,auto)",
        lanes=(lanes_vec or []),
        na_reason=na_reason,
    )
    sig8 = (embed_sig_auto or "")[:8] if embed_sig_auto else "00000000"

    # --- Bundle dir ---
    bundle_dir = _Path("logs") / "certs" / district_id / fixture_id / sig8
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Helpers
    def _canon_dump_and_sig8(obj: dict) -> tuple[str, str]:
        can = _v2_canonical_obj(obj)
        raw = _json.dumps(can, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h = _hash.sha256(raw).hexdigest()
        return raw.decode("utf-8"), h[:8]

    def _write_json(path: _Path, payload: dict) -> str:
        try:
            _guarded_atomic_write_json(path, payload)
        except Exception:
            path.write_text(_json.dumps(_v2_canonical_obj(payload), sort_keys=True, separators=(",", ":")),
                            encoding="utf-8")
        return path.name

       # --- Shared header across certs (now non-empty + correct) ---
    base_hdr = {
        "schema_version": SCHEMA_VERSION,
        "engine_rev": ENGINE_REV,
        "district_id": district_id,
        # fixture_label is canonical; fixture_id kept as legacy alias
        "fixture_label": fixture_label,
        "fixture_id": fixture_label,
        "snapshot_id": snapshot_id,
        "inputs_sig_5": inputs_sig_5,
    }
    written = []

    # 1) strict
    strict_verdict = strict_out.get("pass") if isinstance(strict_out, dict) else None
    strict_payload = build_v2_strict_cert_payload(base_hdr, strict_verdict)
    strict_path = make_strict_cert_path(bundle_dir, district_id, sig8)
    _write_json(strict_path, strict_payload)
    written.append(strict_path.name)

    # 2) projected(columns@k=3,auto)
    proj_verdict = proj_out.get("pass") if isinstance(proj_out, dict) else None
    proj_auto_payload = build_v2_projected_auto_cert_payload(
        base_hdr,
        lanes_vec=lanes_vec,
        lanes_popcount=lanes_pop,
        lanes_sig8=lanes_sig8,
        proj_meta=proj_meta if isinstance(proj_meta, dict) else {},
        verdict=proj_verdict,
    )
    proj_auto_path = make_projected_auto_cert_path(bundle_dir, district_id, sig8)
    _write_json(proj_auto_path, proj_auto_payload)
    written.append(proj_auto_path.name)

    # 3) ab_compare (strict vs projected_auto)
    _, strict_sig8 = _canon_dump_and_sig8(strict_payload)
    _, auto_sig8   = _canon_dump_and_sig8(proj_auto_payload)
    ab_auto_payload = build_v2_ab_compare_payload(
        base_hdr,
        policy="strict__VS__projected(columns@k=3,auto)",
        left_policy="strict",
        left_sig8=str(strict_sig8),
        right_policy="projected(columns@k=3,auto)",
        right_sig8=str(auto_sig8),
    )
    ab_auto_path = bundle_dir / f"ab_compare__strict_vs_projected_auto__{sig8}.json"
    _write_json(ab_auto_path, ab_auto_payload)
    written.append(ab_auto_path.name)

    # 4) projector_freezer
    file_pi_valid   = bool(ss.get("file_pi_valid", True))
    file_pi_reasons = list(ss.get("file_pi_reasons", []) or [])
    freezer_payload = build_v2_projector_freezer_payload(
        base_hdr,
        file_pi_valid=file_pi_valid,
        file_pi_reasons=file_pi_reasons,
    )
    freezer_path = bundle_dir / f"projector_freezer__{district_id}__{sig8}.json"
    _write_json(freezer_path, freezer_payload)
    written.append(freezer_path.name)

    # 5) projected(FILE) + A/B(file) if projector valid
    if file_pi_valid:
        proj_file_payload = build_v2_projected_file_cert_payload(base_hdr)
        proj_file_path = make_projected_file_cert_path(bundle_dir, district_id, sig8)
        _write_json(proj_file_path, proj_file_payload)
        written.append(proj_file_path.name)

        _, file_sig8 = _canon_dump_and_sig8(proj_file_payload)
        ab_file_payload = build_v2_ab_compare_payload(
            base_hdr,
            policy="strict__VS__projected(columns@k=3,file)",
            left_policy="strict",
            left_sig8=str(strict_sig8),
            right_policy="projected(columns@k=3,file)",
            right_sig8=str(file_sig8),
        )
        ab_file_path = bundle_dir / f"ab_compare__strict_vs_projected_file__{sig8}.json"
        _write_json(ab_file_path, ab_file_payload)
        written.append(ab_file_path.name)
    # --- bundle.json ---

    # --- bundle.json ---
    bundle_idx = build_v2_bundle_index_payload(
        run_id=rc.get("run_id", ""),
        sig8=sig8,
        district_id=district_id,
        bundle_dir=bundle_dir,
        filenames=written,
        lanes_popcount=lanes_pop,
        lanes_sig8=lanes_sig8,
        file_pi_valid=file_pi_valid,
    )
    _guarded_atomic_write_json(bundle_dir / "bundle.json", bundle_idx)

    # --- Publish anchors expected by UI ---
    ss["last_bundle_dir"]   = str(bundle_dir)
    ss["last_ab_auto_path"] = str(bundle_dir / f"ab_compare__strict_vs_projected_auto__{sig8}.json")
    ss["last_ab_file_path"] = str(bundle_dir / f"ab_compare__strict_vs_projected_file__{sig8}.json")
    ss["last_solver_result"] = {"count": len(written)}

    return {
        "bundle_dir": str(bundle_dir),
        "sig8": sig8,
        "counts": {"written": len(written)},
        "paths": {"bundle": str(bundle_dir / "bundle.json")},
    }

# ======================= /Solver entrypoint (v2 emit) =========================





# Back-compat alias
one_press_solve = run_overlap_once


def _RUN_SUITE_CANON(manifest_path: str, snapshot_id: str):
    """
    Deterministic v2 runner — ALWAYS returns (ok: bool, msg: str, count: int).
    Reads the real manifest (JSONL), seeds B/C/H/U for each row, runs per-row worker,
    rebuilds bundle.json, and writes loop_receipt__{fixture}.json.
    """
    import json as _json
    import streamlit as _st
    from pathlib import Path as _Path

    # Resolve manifest path (string → Path)
    manifest_abs = _abs_from_manifest(manifest_path)
    if not manifest_abs or not manifest_abs.exists():
        _st.error(f"Manifest not found: {manifest_abs}")
        return False, f"Manifest not found: {manifest_abs}", 0

    # Read JSONL rows
    rows = []
    with manifest_abs.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(_json.loads(raw))
            except Exception as e:
                return False, f"Bad JSONL line: {raw[:120]}… ({e})", 0

    # Helper: coerce to absolute path (repo-root fallback)
    def _ensure_abs(p: str | None) -> _Path | None:
        if not p:
            return None
        q = _Path(p)
        if not q.is_absolute():
            try:
                root = _REPO_DIR  # prefer configured repo root
            except Exception:
                root = _Path(__file__).resolve().parents[1]
            q = (root / q).resolve()
        return q

    ok_count = 0
    total = len(rows)

    for i, rec in enumerate(rows, 1):
        # Fixture id (prefer v2 key)
        fid = rec.get("fixture_label") or rec.get("fixture") or rec.get("id") or f"fx{i:02d}"

        # Extract canonical paths (v2 nested first, then legacy flat)
        paths = rec.get("paths") or {}
        B = paths.get("B") or rec.get("B")
        C = paths.get("C") or rec.get("C")
        H = paths.get("H") or rec.get("H")
        U = paths.get("U") or rec.get("U")

        # Presence preflight (keys present)
        missing_keys = [k for k, v in {"B": B, "C": C, "H": H, "U": U}.items() if not v]
        if missing_keys:
            _st.warning(f"[{fid}] Missing keys in manifest: {', '.join(missing_keys)}")
            continue

        # Existence preflight (files exist)
        Bp, Cp, Hp, Up = map(_ensure_abs, (B, C, H, U))
        missing_files = [k for k, pth in {"B": Bp, "C": Cp, "H": Hp, "U": Up}.items() if not (pth and pth.exists())]
        if missing_files:
            _st.warning(f"[{fid}] Missing files: {', '.join(missing_files)}")
            continue

        # Seed inputs & session context
        try:
            _set_inputs_for_run(str(Bp), str(Cp), str(Hp), str(Up))
        except Exception as e:
            _st.warning(f"[{fid}] failed to seed inputs: {e}")
            continue

        _st.session_state["fixture_label"] = fid
        if "world_snapshot_id" not in _st.session_state:
            _st.session_state["world_snapshot_id"] = snapshot_id

        # Run the per-row worker
        try:
            ret = run_overlap_once()
            ok, msg, bundle_dir = _solver_ret_as_tuple(ret)
        except Exception as e:
            ok, msg, bundle_dir = False, f"solver error: {e}", None

        _st.write(f"{fid} → {'ok' if ok else 'fail'} · {msg}")
        if ok:
            ok_count += 1

        # Resolve bundle dir (fallback to most-recent cert dir)
        bdir = _Path(bundle_dir) if bundle_dir else None
        if not bdir or not bdir.exists():
            try:
                certs_root = _REPO_DIR if '_REPO_DIR' in globals() else _Path(__file__).resolve().parents[1]
                certs = list((certs_root / "logs" / "certs").glob("*/*"))
                bdir = max(certs, key=lambda p: p.stat().st_mtime) if certs else None
            except Exception:
                bdir = None

               # Rebuild bundle and (re)write v2 receipt; surface lanes + append coverage
        try:
            if bdir and bdir.exists():
                bundle = _v2_bundle_index_rebuild(bdir)

                # lanes sidecar (AUTO)
                lanes = bundle.get("lanes") or {}
                lanes_pop, lanes_sig8 = lanes.get("popcount"), lanes.get("sig8")

                # idempotent v2 receipt (SSOT-anchored absolute paths)
                _v2_write_loop_receipt(bdir, fid, snapshot_id, bundle)

                # --- per-row coverage append (minimal but parseable) ---
                try:
                    import uuid as _uuid
                    cov = {
                        "fixture_label":  fid,
                        "snapshot_id":    snapshot_id,
                        "run_id":         str(_uuid.uuid4()),
                        "district_id":    bundle.get("district_id"),
                        "sig8":           bundle.get("sig8"),
                        "policy":         "strict_vs_projected_auto",
                        # metrics (default to 0.0 if not present so reducers have mass)
                        "mismatch_sel":    float((bundle.get("metrics") or {}).get("mismatch_sel", 0.0)),
                        "mismatch_offrow": float((bundle.get("metrics") or {}).get("mismatch_offrow", 0.0)),
                        "mismatch_ker":    float((bundle.get("metrics") or {}).get("mismatch_ker", 0.0)),
                        "verdict_class":   (bundle.get("verdict_class") or "UNKNOWN"),
                        "lanes_popcount":  lanes_pop,
                        "na_reason":       None,
                    }
                    _v2_coverage_append(cov)
                except Exception as e:
                    try:
                        _st.warning(f"[{fid}] coverage append failed: {e}")
                    except Exception:
                        pass

                # suite index append (best-effort)
                try:
                    _suite_index_add_row({
                        "fixture_id": fid,
                        "snapshot_id": snapshot_id,
                        "bundle_dir": str(bdir),
                        "lanes_popcount": lanes_pop,
                        "lanes_sig8": lanes_sig8,
                    })
                except Exception:
                    pass
        except Exception as e:
            _st.warning(f"[{fid}] bundling warning: {e}")


    return True, f"Completed {ok_count}/{total} fixtures.", ok_count





# neutralized (final alias installed at EOF): run_suite_from_manifest = _RUN_SUITE_CANON

# =============================================================================
#-----------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

try:
    del run_suite_from_manifest  # avoid shadowing old defs
except Exception:
    pass

def run_suite_from_manifest(manifest_path: str, snapshot_id: str):
    """
    Dispatch to the available runner and ALWAYS return (ok, msg, count).
    """
    g = globals()
    if "_RUN_SUITE_V2_REPO_ONLY" in g:
        ret = g["_RUN_SUITE_V2_REPO_ONLY"](manifest_path, snapshot_id)
    elif "_RUN_SUITE_CANON" in g:
        ret = g["_RUN_SUITE_CANON"](manifest_path, snapshot_id)
    elif "run_suite_from_manifest__legacy" in g:
        ret = g["run_suite_from_manifest__legacy"](manifest_path, snapshot_id)
    else:
        return False, "Suite runner not found", 0

    if isinstance(ret, tuple):
        if len(ret) == 3:
            return ret
        if len(ret) == 2:
            ok, msg = ret
            return bool(ok), str(msg), 0
    return bool(ret), str(ret), 0






# ====================== V2 CANONICAL — COMPUTE-ONLY (no legacy, no harvest) ======================
import os, json, time, uuid, hashlib
from pathlib import Path as _Pco

def _co_hash8(obj) -> str:
    import hashlib, json as _j
    h = hashlib.sha256(_j.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return h[:8]

def _co_guess_H_variant(paths_dict: dict) -> str:
    # scan all path-like strings for H00/H01/H10/H11
    import re
    hay = " ".join([str(v) for v in (paths_dict or {}).values()])
    m = re.search(r"\b(H00|H01|H10|H11)\b", hay.upper())
    return m.group(1) if m else "H00"

def _co_guess_C_code(paths_dict: dict, fixtures_C_raw: str = "") -> str:
    import re
    hay = (" ".join([str(v) for v in (paths_dict or {}).values()]) + " " + str(fixtures_C_raw)).upper()
    m = re.search(r"\bC(\d{3})\b", hay)
    return f"C{m.group(1)}" if m else "C???.MISSING"

def _co_sig8_from_mats(H2, d3, C3) -> str:
    return _co_hash8({"H2": H2, "d3": d3, "C3": C3})

def _co_guess_district_id(D: str, d3) -> str:
    # mimic your D + short-hash style in a deterministic way
    return f"{D}{_co_hash8({'d3': d3})}"

def _co_guess_snapshot_id(ss, rc, paths_dict) -> str:
    sid = (ss or {}).get("world_snapshot_id") or rc.get("snapshot_id") or ""
    return sid if sid else f"ws__{_co_hash8(paths_dict or {})}"

def _co_write_json(p: _Pco, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)

def _co_bundle_dir(district_id: str, fixture_label: str, sig8: str) -> _Pco:
    return _Pco("logs/certs") / str(district_id) / str(fixture_label) / str(sig8)

def _co_stem(p):
    try:
        return (_Pco(p).stem if isinstance(p,str) else p.stem).upper()
    except Exception:
        return ""

def _co_fixture_label_from_paths(paths: dict, default="UNKNOWN_FIXTURE"):
    try:
        B = (paths.get("B") or "").upper()
        H = (paths.get("H") or "").upper()
        C = (paths.get("C") or "").upper()
        D = "D2" if "D2" in B else ("D3" if "D3" in B else "D?")
        Ht = "H00"
        for hh in ("H00","H01","H10","H11"):
            if hh in H: Ht = hh; break
        import re as _re
        m = _re.search(r"\bC\d{3}\b", C)
        Ct = m.group(0) if m else "C???.MISSING"
        return f"{D}_{Ht}_{Ct}"
    except Exception:
        return default

# --- tiny GF(2) helpers (list-of-lists, no numpy) ---
def _co_shape(A):
    if not A: return (0,0)
    return (len(A), len(A[0]) if A and isinstance(A[0], (list,tuple)) else 0)

def _co_mm2(A,B):
    # A[m x k] · B[k x n] mod 2
    m,k = _co_shape(A)
    k2,n = _co_shape(B)
    if k != k2:
        raise ValueError("shape mismatch for mm2")
    R = [[0]*n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for j in range(n):
            s = 0
            for t in range(k):
                s ^= (Ai[t] & B[t][j])
            R[i][j] = s & 1
    return R

def _co_xor(A,B):
    m,n = _co_shape(A)
    m2,n2 = _co_shape(B)
    if m!=m2 or n!=n2:
        raise ValueError("shape mismatch for xor")
    R = [[(A[i][j]^B[i][j]) for j in range(n)] for i in range(m)]
    return R

def _co_eye(n):
    return [[1 if i==j else 0 for j in range(n)] for i in range(n)]



def _co_masked_allzero_cols(M, lanes):
    # check columns with lanes[j]==1 are all-zero
    m,n = _co_shape(M)
    assert len(lanes) == n
    for i in range(m):
        row = M[i]
        for j in range(n):
            if lanes[j] and row[j]==1:
                return False
    return True

def _co_extract_mats(pb):
    def blocks(x):
        if isinstance(x, (list,tuple)) and len(x)>=2:
            return x[1] or {}
        return {}
    bB = blocks(pb.get("B"))
    bC = blocks(pb.get("C"))
    bH = blocks(pb.get("H"))
    H2 = bH.get("2") or []
    d3 = bB.get("3") or []
    C3 = bC.get("3") or []
    # ensure deep ints (0/1)
    def _as01(M):
        return [[1 if (int(x) & 1) else 0 for x in r] for r in (M or [])]
    return _as01(H2), _as01(d3), _as01(C3)




def _co_compute_freezer_FILE(ss, n3):
    # attempt to get file projector diag lanes from session
    lanes = None
    try:
        for key in ("file_lanes","file_pi_vec","file_projector_diag"):
            v = ss.get(key)
            if isinstance(v, (list,tuple)) and all((x in (0,1,True,False) for x in v)):
                lanes = [1 if x else 0 for x in list(v)]
                break
        if lanes is None and isinstance(ss.get("file_projector_json"), dict):
            diag = ss["file_projector_json"].get("diag") or ss["file_projector_json"].get("lanes")
            if isinstance(diag, list):
                lanes = [1 if x else 0 for x in diag]
    except Exception:
        lanes = None

    if lanes is None:
        return {"status": "NA", "na_reason_code": "FILE_PROJECTOR_MISSING"}, None
    if len(lanes) != n3:
        return {"status": "NA", "na_reason_code": "FILE_PROJECTOR_WRONG_SIZE"}, None
    if sum(lanes)==0:
        return {"status": "NA", "na_reason_code": "FILE_LANES_ZERO"}, lanes
    return {"status": "OK", "na_reason_code": None}, lanes

def _co_compute_projected_file(H2, d3, C3, lanes_file):
    mC,nC = _co_shape(C3)
    if not (mC==nC):
        return {
            "policy_tag": "projected(columns@k=3,file)",
            "results": {"k3":{"eq": None}}, "na_reason_code": "C3_NON_SQUARE"
        }
    try:
        Hd = _co_mm2(H2, d3)
        R3 = _co_xor(Hd, _co_xor(C3, _co_eye(nC)))
    except Exception as e:
        return {"policy_tag": "projected(columns@k=3,file)",
                "results": {"k3":{"eq": None}}, "na_reason_code": f"SHAPE_MISMATCH:{str(e)}"}
    eq = _co_masked_allzero_cols(R3, lanes_file)
    mism_cols_sel = []
    mR,nR = _co_shape(R3)
    for j in range(nR):
        if lanes_file[j]==1:
            for i in range(mR):
                if R3[i][j]==1:
                    mism_cols_sel.append(j); break
    return {
        "policy_tag": "projected(columns@k=3,file)",
        "results": {
            "k3": {"eq": bool(eq)},
            "selected_cols": list(lanes_file),
            "mismatch_cols_selected": mism_cols_sel,
        },
    }

def _co_ab_compare(strict, proj, policy_label):
    k3s = (strict.get("results",{}).get("k3",{}).get("eq"),
           proj.get("results",{}).get("k3",{}).get("eq"))
    embed = {"policy": policy_label}
    emb_bytes = json.dumps(embed, sort_keys=True).encode("utf-8")
    embed_sig = hashlib.sha256(emb_bytes).hexdigest()
    return {
        "ab_pair": {
            "policy": policy_label,
            "embed_sig": embed_sig,
            "pair_vec": {"k2":[None,None], "k3":[k3s[0], k3s[1]]},
        }
    }




    



# ─────────────────────────────────────────────────────────────────────────────
# V2 strict mechanics: receipts → manifest → suite → histograms
# self-contained helpers, no changes to your 1× pipeline
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path as _Path
import os as _os, json as _json, time as _time, glob as _glob, math as _math
import collections as _collections
import streamlit as _st

# --- Constants & dirs
_REPO_ROOT = _Path(__file__).resolve().parent.parent
_CERTS_DIR = _REPO_ROOT / "logs" / "certs"
_MANIFESTS_DIR = _REPO_ROOT / "logs" / "manifests"
_REPORTS_DIR = _REPO_ROOT / "logs" / "reports"
_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Atomic JSON write
def _v2_atomic_write_json(path: _Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        _json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    _os.replace(str(tmp), str(path))

# --- Latest bundle discovery (best-effort, by mtime)
def _v2_latest_bundle_dir() -> _Path | None:
    cand = []
    if _CERTS_DIR.exists():
        for p in _CERTS_DIR.rglob("bundle.json"):
            try:
                cand.append((p.stat().st_mtime, p.parent))
            except Exception:
                pass
    if not cand:
        return None
    cand.sort(reverse=True)
    return cand[0][1]  # dir hosting bundle.json

# --- Pull SSOT paths for receipt (prefers session_state)
def _v2_collect_paths_from_ssot() -> dict:
    ss = _st.session_state
    keys = [
        ("B", ["uploaded_B_path","uploaded_boundaries"]),
        ("C", ["uploaded_C_path","uploaded_cmap"]),
        ("H", ["uploaded_H_path","uploaded_H"]),
        ("U", ["uploaded_U_path","uploaded_shapes"]),
    ]
    out = {}
    for k, opts in keys:
        val = None
        for o in opts:
            v = ss.get(o)
            if isinstance(v, str) and v:
                val = v; break
        out[k] = str(_Path(val).resolve()) if val else None
    return out


# --- Write a loop_receipt (v2) into a given bundle dir
def _v2_write_loop_receipt_for_bundle(bdir, extra: dict | None = None):
    """
    Write loop_receipt.v2 into `bdir`. Robustly resolves:
      - fixture_label (D*_H*_C*)
      - district_id (hashed D-tag form OK, e.g., D3a5ca34ee)
      - sig8 (from path or bundle.json)
      - absolute SSOT paths for B/C/H/U from fixture_label
    Returns (ok: bool, msg: str)
    """
    import json as _json, time as _time, re as _re
    from pathlib import Path as _Path

    bdir = _Path(bdir)
    bundle = {}
    bj = bdir / "bundle.json"
    if bj.exists():
        try:
            bundle = _json.loads(bj.read_text(encoding="utf-8"))
        except Exception:
            bundle = {}

    # Prefer explicit extras, fallback to bundle fields, then path parse
    district_id   = (extra or {}).get("district_id")   or bundle.get("district_id")
    fixture_label = (extra or {}).get("fixture_label")  or bundle.get("fixture_label")
    sig8          = (extra or {}).get("sig8")           or bundle.get("sig8") or bdir.name

    # If fixture/district missing, parse from path .../certs/{district_id}/{fixture_label}/{sig8}
    try:
        parent2 = bdir.parent  # .../{fixture_label}
        parent1 = parent2.parent  # .../{district_id}
        if not district_id:
            district_id = parent1.name
        if not fixture_label:
            fixture_label = parent2.name
    except Exception:
        pass

    # Final guard on fixture_label; refuse to write UNKNOWN
    if not fixture_label or not _re.fullmatch(r"D\d+_H\d{2}_C\d{3}", str(fixture_label)):
        return False, f"Bad or missing fixture_label for {bdir}"

    # Derive canonical SSOT absolute paths from fixture_label
    mD = _re.search(r"(?:^|_)D(\d+)", fixture_label); D_tag = f"D{mD.group(1)}" if mD else None
    mH = _re.search(r"(?:^|_)H(\d+)", fixture_label); H_tag = f"H{mH.group(1)}" if mH else None
    mC = _re.search(r"(?:^|_)C(\d+)", fixture_label); C_tag = f"C{mC.group(1)}" if mC else None

    try:
        repo_root = _REPO_DIR
    except Exception:
        repo_root = _Path(__file__).resolve().parents[1]
    inputs_root = repo_root / "app" / "inputs"

    P = {
        "B": str((inputs_root / "B" / f"{D_tag}.json").resolve()) if D_tag else None,
        "C": str((inputs_root / "C" / f"{C_tag}.json").resolve()) if C_tag else None,
        "H": str((inputs_root / "H" / f"{H_tag}.json").resolve()) if H_tag else None,
        "U": str((inputs_root / "U.json").resolve()),
    }

    # Validate absolute existence
    from pathlib import Path as _P
    if not all(p and _P(p).is_absolute() and _P(p).exists() for p in P.values()):
        return False, f"[{fixture_label}] SSOT path(s) missing"

        # dims (nice to have)
    dims = None
    if isinstance(bundle, dict):
        if "dims" in bundle and isinstance(bundle["dims"], dict):
            dims = {
                "n2": bundle["dims"].get("n2"),
                "n3": bundle["dims"].get("n3"),
            }

    # core_counts: prefer bundle.core_counts.written, else len(files/filenames), else fallback
    core_written = None
    try:
        if isinstance(bundle, dict):
            cc = bundle.get("core_counts")
            if isinstance(cc, dict) and isinstance(cc.get("written"), int):
                core_written = int(cc["written"])
            elif isinstance(bundle.get("files"), dict):
                core_written = len(bundle["files"])
            elif isinstance(bundle.get("filenames"), (list, tuple)):
                core_written = len(bundle["filenames"])
    except Exception:
        core_written = None
    if core_written is None:
        core_written = 6  # conservative default for old runs

    receipt = {
        "schema": "loop_receipt.v2",
        "run_id": (extra or {}).get("run_id"),
        "district_id": district_id,
        "fixture_label": fixture_label,
        "sig8": sig8,
        "bundle_dir": str(bdir.resolve()),
        "paths": P,
        "core_counts": {"written": int(core_written)},
        "timestamps": {"receipt_written_at": int(_time.time())},
    }
    if dims:
        receipt["dims"] = dims


    # Always write with proper filename (no UNKNOWN)
    outp = bdir / f"loop_receipt__{fixture_label}.json"
    _hard_co_write_json(outp, receipt)
    return True, f"[{fixture_label}] wrote loop_receipt.v2"


# --- Regenerate manifest_full_scope.jsonl by scanning loop_receipts
def _v2_regen_manifest_from_receipts():
    """
    Scan logs/certs/**/loop_receipt__*.json (schema=loop_receipt.v2),
    validate absolute SSOT paths (B,C,H,U), deduplicate by fixture_label,
    and write logs/manifests/manifest_full_scope.jsonl atomically.

    Returns: (ok: bool, manifest_path: Path, kept_count: int)
    """
    import json as _json
    from pathlib import Path as _Path
    import time as _time
    import re as _re

    try:
        repo_root = _REPO_DIR
    except Exception:
        repo_root = _Path(__file__).resolve().parents[1]

    manifests_dir = repo_root / "logs" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "manifest_full_scope.jsonl"

    try:
        certs_root = _CERTS_DIR  # preferred if defined
    except Exception:
        certs_root = repo_root / "logs" / "certs"

    receipts = list(certs_root.rglob("loop_receipt__*.json"))

    records = []
    bad = 0
    grouped = {}  # fixture_label -> best (rec, score, src_path)

    def _score(rj: dict, pth: _Path) -> float:
        # prefer explicit receipt_written_at, fall back to file mtime
        ts = (((rj or {}).get("timestamps") or {}).get("receipt_written_at")) or 0
        try:
            ts = float(ts)
        except Exception:
            ts = 0.0
        return max(ts, pth.stat().st_mtime)

    for rp in receipts:
        try:
            rj = _json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            bad += 1
            continue

        if not isinstance(rj, dict) or rj.get("schema") != "loop_receipt.v2":
            bad += 1
            continue

        fid = rj.get("fixture_label") or ""
        paths = rj.get("paths") or {}
        B, C, H, U = paths.get("B"), paths.get("C"), paths.get("H"), paths.get("U")
        if not (B and C and H and U):
            bad += 1
            continue

        # All must be absolute and exist on disk
        Bp, Cp, Hp, Up = map(_Path, (B, C, H, U))
        if not (Bp.is_absolute() and Cp.is_absolute() and Hp.is_absolute() and Up.is_absolute()):
            bad += 1
            continue
        if not (Bp.exists() and Cp.exists() and Hp.exists() and Up.exists()):
            bad += 1
            continue

        # Enforce B comes from D-tag in fixture_label (never hashed district_id)
        mD = _re.search(r"(?:^|_)D(\d+)", fid or "")
        D_tag = f"D{mD.group(1)}" if mD else None
        if not D_tag or _Path(B).stem != D_tag:
            bad += 1
            continue

        sc = _score(rj, rp)
        keep = grouped.get(fid)
        if (keep is None) or (sc > keep[1]):
            # minimal row with dims if present
            row = {"fixture_label": fid, "paths": paths}
            if "dims" in rj and isinstance(rj["dims"], dict):
                row["dims"] = {"n2": rj["dims"].get("n2"), "n3": rj["dims"].get("n3")}
            grouped[fid] = (row, sc, rp)

    # finalize rows in stable order
    for fid in sorted(grouped.keys()):
        records.append(grouped[fid][0])

    # atomic write
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(_json.dumps(r, separators=(",", ":")) for r in records) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)

    # optional: warn if we dropped anything
    dropped = len(receipts) - bad - len(records)
    if dropped > 0:
        try:
            _st.warning(f"Manifest dedup: kept {len(records)} rows · dropped {dropped} older duplicates · skipped {bad} invalid receipts")
        except Exception:
            pass

    return True, manifest_path, len(records)


# --- Histogram reductions over coverage.jsonl
def _v2_build_histograms_from_coverage(snapshot_id: str | None = None):
    """
    Build simple histograms over coverage.jsonl with a tolerant field mapper.
    Writes logs/reports/histograms_v2.json and returns (ok, msg, out_path).
    """
    import json as _json
    from pathlib import Path as _Path
    from collections import defaultdict

    try:
        root = _REPO_DIR
    except Exception:
        root = _Path(__file__).resolve().parents[1]
    rep_dir = root / "logs" / "reports"
    cov_path = rep_dir / "coverage.jsonl"
    outp = rep_dir / "histograms_v2.json"
    rep_dir.mkdir(parents=True, exist_ok=True)

    if not cov_path.exists():
        return False, "coverage.jsonl not found.", None

    def _coerce_f(x):
        if x is None: return None
        try: return float(x)
        except Exception: return None

    def _bucket_rate_0_1(x):
        if x is None: return "NA"
        x = max(0.0, min(1.0, x))
        # 10 buckets: [0.0,0.1),[0.1,0.2),..., [0.9,1.0]
        b = int(x * 10)
        lo = b / 10.0
        hi = 1.0 if b == 9 else (b + 1) / 10.0
        return f"{lo:.1f}–{hi:.1f}"

    def _bucket_int_small(n):
        if n is None: return "NA"
        try: n = int(n)
        except Exception: return "NA"
        if n <= 0: return "0"
        if n <= 4: return str(n)
        if n <= 8: return "5–8"
        if n <= 16: return "9–16"
        return "17+"

    # bins
    bins = {
        "sel_mismatch_rate_buckets": defaultdict(int),
        "offrow_mismatch_rate_buckets": defaultdict(int),
        "ker_mismatch_rate_buckets": defaultdict(int),
        "lanes_popcount_buckets": defaultdict(int),
        "verdict_class": defaultdict(int),
        "by_district": defaultdict(int),
    }

    # read and map fields
    import re as _re
    with cov_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = _json.loads(line)
            except Exception:
                continue
            if snapshot_id and r.get("snapshot_id") != snapshot_id:
                continue

            fid = r.get("fixture_label", "")
            mD = _re.search(r"(?:^|_)D(\d+)", fid)
            dtag = f"D{mD.group(1)}" if mD else "UNKNOWN"

            # v2 preferred → legacy fallbacks
            lanes_pop = r.get("lanes_popcount")
            if lanes_pop is None:
                lanes_pop = r.get("lane_popcount_auto")  # legacy
            sel = r.get("mismatch_sel");    sel = sel if sel is not None else r.get("sel_mismatch_rate")
            off = r.get("mismatch_offrow"); off = off if off is not None else r.get("offrow_mismatch_rate")
            ker = r.get("mismatch_ker");    ker = ker if ker is not None else r.get("ker_mismatch_rate")
            vcls = (r.get("verdict_class") or "UNKNOWN").upper()

            # increment bins
            bins["lanes_popcount_buckets"][_bucket_int_small(lanes_pop)] += 1
            bins["sel_mismatch_rate_buckets"][_bucket_rate_0_1(_coerce_f(sel))] += 1
            bins["offrow_mismatch_rate_buckets"][_bucket_rate_0_1(_coerce_f(off))] += 1
            bins["ker_mismatch_rate_buckets"][_bucket_rate_0_1(_coerce_f(ker))] += 1
            bins["verdict_class"][vcls] += 1
            bins["by_district"][dtag] += 1

    # write out
    out = {k: dict(v) for k, v in bins.items()}
    try:
        outp.write_text(_json.dumps(out, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        return True, f"Wrote {outp}", outp
    except Exception as e:
        return False, f"histogram write failed: {e}", None


    


def _v2_pack_suite_fat_zip(snapshot_id: str):
    """
    Create a ZIP with:
      - all per-bundle JSONs (whatever exists: strict/auto + bundle + loop_receipt + sidecars)
      - manifest_full_scope.jsonl
      - coverage.jsonl, histograms_v2.json, coverage_rollup.csv (if present)
    Returns Path or None.
    """
    from zipfile import ZipFile, ZIP_STORED
    from pathlib import Path as _Path

    try:
        root = _REPO_DIR
    except Exception:
        root = _Path(__file__).resolve().parents[1]

    reports_dir = root / "logs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    outp = reports_dir / f"suite_fat__{snapshot_id}.zip"

    certs_dir = root / "logs" / "certs"
    man = root / "logs" / "manifests" / "manifest_full_scope.jsonl"
    cov = root / "logs" / "reports" / "coverage.jsonl"
    hist = root / "logs" / "reports" / "histograms_v2.json"
    roll = root / "logs" / "reports" / "coverage_rollup.csv"

    if not certs_dir.exists():
        return None

    try:
        with ZipFile(outp, "w", compression=ZIP_STORED) as z:
            # per-bundle JSONs
            for bj in certs_dir.rglob("bundle.json"):
                bdir = bj.parent
                for p in sorted(bdir.glob("*.json")):
                    try:
                        z.write(p, arcname=str(p.relative_to(root)))
                    except Exception:
                        pass
            # globals
            for extra in (man, cov, hist, roll):
                if extra.exists():
                    z.write(extra, arcname=str(extra.relative_to(root)))
        return outp
    except Exception:
        return None



# ====================== V2 COMPUTE-ONLY (HARD) — single source of truth ======================
import os as _os, json as _json, time as _time, uuid as _uuid, hashlib as _hashlib
from pathlib import Path as _Ph

def _hard_co_write_json(p: _Ph, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(_json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    _os.replace(tmp, p)

def _hard_co_shape(A):
    if not A: return (0,0)
    return (len(A), len(A[0]) if A and isinstance(A[0], (list, tuple)) else 0)

def _hard_co_eye(n):
    return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

def _hard_co_mm2(A,B):
    m,k = _hard_co_shape(A)
    k2,n = _hard_co_shape(B)
    if k != k2:
        raise ValueError("shape mismatch for mm2")
    R = [[0]*n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for j in range(n):
            s = 0
            for t in range(k):
                s ^= (Ai[t] & B[t][j])
            R[i][j] = s & 1
    return R

def _hard_co_xor(A,B):
    m,n = _hard_co_shape(A)
    m2,n2 = _hard_co_shape(B)
    if m!=m2 or n!=n2:
        raise ValueError("shape mismatch for xor")
    return [[(A[i][j]^B[i][j]) for j in range(n)] for i in range(m)]

def _hard_co_allzero_cols(M):
    m,n = _hard_co_shape(M)
    zero = [True]*n
    for i in range(m):
        row = M[i]
        for j in range(n):
            if row[j] == 1:
                zero[j] = False
    return zero

def _hard_co_masked_allzero_cols(M, lanes):
    m,n = _hard_co_shape(M)
    if len(lanes) != n:
        return False
    for i in range(m):
        row = M[i]
        for j in range(n):
            if lanes[j] and row[j]==1:
                return False
    return True

def _hard_co_hash8(obj) -> str:
    import hashlib
    h = hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return h[:8]

def _hard_co_bitsig256(bits):
    import hashlib
    return hashlib.sha256(_json.dumps([1 if b else 0 for b in bits], separators=(",", ":"), sort_keys=False).encode("utf-8")).hexdigest()

def _hard_co_zero_mask_from_cols(M):
    m, n = _hard_co_shape(M)
    z = [1]*n
    for i in range(m):
        row = M[i]
        for j in range(n):
            if row[j] == 1:
                z[j] = 0
    return z

def _hard_co_support_mask(M):
    z = _hard_co_zero_mask_from_cols(M)
    return [0 if z[j] else 1 for j in range(len(z))]

def _hard_co_subset(maskA, maskB):
    return all((a == 0) or (b == 1) for a, b in zip(maskA, maskB))

def _hard_verdict_class(strict_eq, supp_R3, lanes, ker_mask, posed: bool):
    if not posed:
        return None
    if strict_eq is True:
        return "GREEN"
    if _hard_co_subset(supp_R3, ker_mask):
        exposed = any((lanes[j] == 1 and supp_R3[j] == 1) for j in range(len(lanes)))
        return "KER-EXPOSED" if exposed else "KER-FILTERED"
    return "RED_BOTH"

def _hard_co_extract_mats(pb):
    def _blocks(x):
        if isinstance(x, (list, tuple)) and len(x)>=2:
            return x[1] or {}
        return {}
    bB = _blocks(pb.get("B")); bC = _blocks(pb.get("C")); bH = _blocks(pb.get("H"))
    H2 = bH.get("2") or []; d3 = bB.get("3") or []; C3 = bC.get("3") or []
    def _as01(M):
        return [[1 if (int(x) & 1) else 0 for x in r] for r in (M or [])]
    return _as01(H2), _as01(d3), _as01(C3)

def _hard_fixture_tuple_from_paths(pB, pH, pC):
    import os, re
    bname = os.path.basename(os.fspath(pB or ""))
    if "D2" in bname: D = "D2"
    elif "D3" in bname: D = "D3"
    else: D = "UNKNOWN_DISTRICT"
    hname = os.path.basename(os.fspath(pH or ""))
    mH = re.search(r"[Hh]([01]{2})", hname)
    H = f"H{mH.group(1)}" if mH else "H??"
    cname = os.path.basename(os.fspath(pC or ""))
    mC = re.search(r"[Cc]([01]{3})", cname)
    C = f"C{mC.group(1)}" if mC else "C???"
    return D, H, C, f"{D}_{H}_{C}"

def _hard_bundle_dir(district_id: str, fixture_label: str, sig8: str):
    """Canonical: logs/certs/{district_id}/{fixture_label}/{sig8}."""
    from pathlib import Path as _Path
    try:
        root = _CERTS_DIR
    except Exception:
        root = _Path(__file__).resolve().parents[1] / "logs" / "certs"
    bdir = root / str(district_id) / str(fixture_label) / str(sig8)
    bdir.mkdir(parents=True, exist_ok=True)
    return bdir


def _svr_run_once_computeonly_hard(ss=None):
    g = globals()
    resolver = g.get("_svr_resolve_all_to_paths") or g.get("resolve_all_to_paths")
    freezer  = g.get("_svr_freeze_ssot") or g.get("freeze_ssot")
    if not resolver or not freezer:
        return False, "Missing SSOT helpers (_svr_resolve_all_to_paths/_svr_freeze_ssot).", ""

    pb = resolver() or {}

    def _first(x):
        if isinstance(x, (list, tuple)) and x: return x[0]
        return x
    pB = _first(pb.get("B")); pC = _first(pb.get("C")); pH = _first(pb.get("H")); pU = _first(pb.get("U"))
    paths_dict = {"B": str(pB or ""), "C": str(pC or ""), "H": str(pH or ""), "U": str(pU or "")}

    if ss is None:
        try:
            import streamlit as _st
            ss = _st.session_state
        except Exception:
            ss = {}
    try:
        ss["_last_inputs_paths"] = dict(paths_dict)
    except Exception:
        pass

    ib, rc = freezer(pb); ib = dict(ib or {}); rc = dict(rc or {})

    H2, d3, C3 = _hard_co_extract_mats(pb)
    _, n3 = _hard_co_shape(d3)

    # One-time algebra
    mC, nC = _hard_co_shape(C3)
    CxorI_global = _hard_co_xor(C3, _hard_co_eye(nC)) if (mC == nC) else C3
    Hd_global = _hard_co_mm2(H2, d3) if (_hard_co_shape(H2)[1] == _hard_co_shape(d3)[0]) else [[0]*(_hard_co_shape(d3)[1])]
    R3_global = _hard_co_xor(Hd_global, CxorI_global) if CxorI_global else Hd_global
    supp_R3 = _hard_co_support_mask(R3_global)
    ker_mask = _hard_co_zero_mask_from_cols(d3)

    D, Htag, Ctag, fixture_label = _hard_fixture_tuple_from_paths(pB, pH, pC)
    district_id = rc.get("district_id") or f"{D}{_hard_co_hash8({'d3': d3})}"
    sig8        = rc.get("sig8") or (rc.get("embed_sig","")[:8] if rc.get("embed_sig") else _hard_co_hash8({"H2":H2,"d3":d3,"C3":C3}))
    snapshot_id = rc.get("snapshot_id") or (ss.get("world_snapshot_id") if isinstance(ss, dict) else None) or ""

    fixtures = {"district": D, "H": Htag, "C": Ctag, "U": "U"}

    # STRICT
    strict_eq_bool = (sum(supp_R3) == 0) if R3_global else False
    strict_payload = {
        "policy_tag": "strict(k=3)",
        "results": {"k3": {"eq": strict_eq_bool}},
        "metrics": {
            "R3_failing_cols_popcount": int(sum(supp_R3)),
            "ker_cols_popcount": int(sum(ker_mask)),
        },
        "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
        "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
    }

    # PROJECTED AUTO
    if not (mC == nC):
        auto_payload = {
            "policy_tag": "projected(columns@k=3,auto)",
            "results": {"k3": {"eq": None}, "selected_cols": []},
            "na_reason_code": "C3_NON_SQUARE",
            "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
            "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
        }
    else:
        lanes = list(C3[-1] if mC > 0 else [0]*nC)
        if sum(lanes) == 0:
            auto_payload = {
                "policy_tag": "projected(columns@k=3,auto)",
                "results": {"k3": {"eq": None}, "selected_cols": lanes},
                "na_reason_code": "LANES_ZERO",
                "lanes_sig256": _hard_co_bitsig256(lanes),
                "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
                "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
            }
        else:
            proj_fail_mask = [1 if (lanes[j] == 1 and supp_R3[j] == 1) else 0 for j in range(len(supp_R3))]
            proj_eq_bool = (sum(proj_fail_mask) == 0)
            vclass = _hard_verdict_class(strict_eq_bool, supp_R3, lanes, ker_mask, posed=True)
            auto_payload = {
                "policy_tag": "projected(columns@k=3,auto)",
                "results": {"k3": {"eq": proj_eq_bool}, "selected_cols": lanes},
                "metrics": {"proj_failing_cols_popcount": int(sum(proj_fail_mask))},
                "lanes_sig256": _hard_co_bitsig256(lanes),
                "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
                "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
                "proj_failing_cols_sig256": _hard_co_bitsig256(proj_fail_mask),
                "verdict_class": vclass,
            }

    # FILE lanes from session
    def _extract_file_lanes(_ss):
        try:
            for key in ("file_lanes","file_pi_vec","file_projector_diag"):
                v = _ss.get(key)
                if isinstance(v, (list,tuple)) and all(x in (0,1,True,False) for x in v):
                    return [1 if x else 0 for x in v]
            if isinstance(_ss.get("file_projector_json"), dict):
                diag = _ss["file_projector_json"].get("diag") or _ss["file_projector_json"].get("lanes")
                if isinstance(diag, list):
                    return [1 if x else 0 for x in diag]
        except Exception:
            pass
        return None

    lanes_file = _extract_file_lanes(ss)
    if lanes_file is None:
        freezer_payload = {"status":"NA", "na_reason_code":"FILE_PROJECTOR_MISSING"}
        file_payload = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq": None}, "selected_cols": []},
            "na_reason_code":"FILE_PROJECTOR_MISSING",
            "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
            "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
        }
    elif len(lanes_file) != (_hard_co_shape(d3)[1]):
        freezer_payload = {"status":"NA", "na_reason_code":"FILE_PROJECTOR_WRONG_SIZE"}
        file_payload = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq": None}, "selected_cols": []},
            "na_reason_code":"FILE_PROJECTOR_WRONG_SIZE",
            "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
            "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
        }
    elif sum(lanes_file)==0:
        freezer_payload = {"status":"OK", "na_reason_code": None}
        file_payload = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq": None}, "selected_cols": list(lanes_file)},
            "na_reason_code":"FILE_LANES_ZERO",
            "lanes_sig256": _hard_co_bitsig256(lanes_file),
            "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
            "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
        }
    else:
        freezer_payload = {"status":"OK", "na_reason_code": None}
        proj_fail_mask_f = [1 if (lanes_file[j] == 1 and supp_R3[j] == 1) else 0 for j in range(len(supp_R3))]
        proj_eq_bool_f = (sum(proj_fail_mask_f) == 0)
        vclass_f = _hard_verdict_class(strict_eq_bool, supp_R3, lanes_file, ker_mask, posed=True)
        file_payload = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq": proj_eq_bool_f}, "selected_cols": list(lanes_file)},
            "metrics": {"proj_failing_cols_popcount": int(sum(proj_fail_mask_f))},
            "lanes_sig256": _hard_co_bitsig256(lanes_file),
            "ker_mask_sig256": _hard_co_bitsig256(ker_mask),
            "strict_failing_cols_sig256": _hard_co_bitsig256(supp_R3),
            "proj_failing_cols_sig256": _hard_co_bitsig256(proj_fail_mask_f),
            "verdict_class": vclass_f,
        }

    # AB compares inline (no globals)
    def _mk_ab(policy_label, a_eq, b_eq):
        emb = {"policy": policy_label}
        embed_sig = _hashlib.sha256(_json.dumps(emb, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        def _b(v): return (bool(v) if v is not None else None)
        return {"ab_pair": {"policy": policy_label, "embed_sig": embed_sig,
                            "pair_vec": {"k2":[None,None], "k3":[_b(a_eq), _b(b_eq)]}}}

    strict_eq = ((strict_payload.get("results",{}) or {}).get("k3",{}) or {}).get("eq")
    auto_eq   = ((auto_payload.get("results",{}) or {}).get("k3",{}) or {}).get("eq")
    file_eq   = ((file_payload.get("results",{}) or {}).get("k3",{}) or {}).get("eq")
    ab_auto_payload = _mk_ab("strict__VS__projected(columns@k=3,auto)", strict_eq, auto_eq)
    ab_file_payload = _mk_ab("strict__VS__projected(columns@k=3,file)", strict_eq, file_eq)

              # --- WRITE MIN CORE (V2-pure) — exactly 2 certs per fixture ------------------
    def _as_dims(x):
        if isinstance(x, dict):
            if "dims" in x: return {"n2": x["dims"].get("n2"), "n3": x["dims"].get("n3")}
            if "n2" in x and "n3" in x: return {"n2": x["n2"], "n3": x["n3"]}
        return None
    
    dims = _as_dims(strict_payload) or _as_dims(auto_payload)
    
    def _stamp(obj, policy=None):
        o = dict(obj or {})
        if policy and "policy" not in o:
            o["policy"] = policy
        o.setdefault("fixture_label", fixture_label)
        if snapshot_id: o.setdefault("snapshot_id", snapshot_id)
        o.setdefault("sig8", sig8)
        o.setdefault("written_at_utc", int(_time.time()))
        return o
    
    bdir = _hard_bundle_dir(district_id, fixture_label, sig8)
    names = {
        "strict":         f"overlap__{district_id}__strict__{sig8}.json",
        "projected_auto": f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json",
    }
    
    # Write exactly two certs (no FILE, no A/B)
    _hard_co_write_json(bdir / names["strict"],         _stamp(strict_payload, "strict"))
    _hard_co_write_json(bdir / names["projected_auto"], _stamp(auto_payload, "projected(columns@k=3,auto)"))
    
        # Bundle manifest (2-core, v2)
    filenames = [names[k] for k in ("strict", "projected_auto")]
    lanes_extra = (auto_payload.get("lanes") or {}) if isinstance(auto_payload, dict) else {}
    metrics_extra = (auto_payload.get("metrics") or {}) if isinstance(auto_payload, dict) else {}

    bundle = build_v2_bundle_manifest(
        district_id=district_id,
        fixture_label=fixture_label,
        sig8=str(sig8),
        snapshot_id=snapshot_id or None,
        bundle_dir=str(bdir.resolve()),
        core_counts={"written": 2},
        dims=dims,
        written_at_utc=int(_time.time()),
        # keep legacy fields via `extra` so old readers still work
        extra={
            "filenames": filenames,
            "lanes": lanes_extra,
            "metrics": metrics_extra,
        },
    )
    _hard_co_write_json(bdir / "bundle.json", bundle)

    
    # SSOT-anchored receipt (v2)
    _v2_write_loop_receipt_for_bundle(bdir, extra={
        "district_id": district_id,
        "fixture_label": fixture_label,
        "sig8": sig8,
        "dims": dims,
    })


    



    # ---- C1 coverage append (v2 ker_RED — canonical, JSON-grounded) ----
    try:
        cov_path, _cov_csv = _c1_paths()
        _Lsrc  = locals().get("lanes", [])
        _Ssrc  = locals().get("supp_R3", [])
        _Ksrc  = locals().get("ker_mask", [])
        _Lfsrc = locals().get("lanes_file", None)
        strict_eq_bool = bool(locals().get("strict_eq_bool", False))
        snapshot_id = locals().get("snapshot_id", "")

        def _nz(x):
            try: return int(x) != 0
            except Exception: return False
        def _norm_bits(v, n):
            try: arr = [1 if _nz(x) else 0 for x in (v or [])]
            except Exception: arr = [0]*n
            if n and len(arr) != n: arr = (arr + [0]*n)[:n]
            return arr

        try: _mC, _nC = _hard_co_shape(C3)
        except Exception: _mC = _nC = 0
        if (not _nC) and isinstance(_Ssrc, list): _nC = len(_Ssrc)
        if (not _nC) and isinstance(_Ksrc, list): _nC = len(_Ksrc)
        n_cols = int(_nC or 0)

        L  = _norm_bits(_Lsrc,  n_cols)
        S  = _norm_bits(_Ssrc,  n_cols)
        K  = _norm_bits(_Ksrc,  n_cols)
        Lf = _norm_bits(_Lfsrc, n_cols) if _Lfsrc is not None else None

        # Backfill lanes from AUTO cert JSON if empty
        try:
            _auto_path = _Path(bdir) / f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json"
            if (not any(L)) and _auto_path.exists():
                _auto_js = _hard_co_read_json(_auto_path)
                _w = (_auto_js or {}).get("witness") or {}
                _r = (_auto_js or {}).get("results") or {}
                _Ljson = _w.get("lanes") or _r.get("selected_cols") or []
                L = _norm_bits(_Ljson, n_cols)
        except Exception:
            pass

        posed_auto = bool(sum(L) > 0 and _mC == _nC and n_cols > 0)
        try: posed_file = ((file_payload or {}).get("na_reason_code") is None)
        except Exception: posed_file = bool(isinstance(Lf, list) and len(Lf)==n_cols and sum(Lf)>0)

        sel_idx = [j for j in range(n_cols) if L[j]==1]
        off_idx = [j for j in range(n_cols) if L[j]==0]
        fail_idx = [j for j in range(n_cols) if S[j]==1]
        sel_fail_idx = [j for j in sel_idx if S[j]==1]
        off_fail_idx = [j for j in off_idx if S[j]==1]

        sel_sum = int(sum(L))
        sel_mismatch_rate = (len(sel_fail_idx)/sel_sum) if sel_sum>0 else (0.0 if posed_auto else None)
        off_denom = max(0, n_cols - sel_sum)
        offrow_mismatch_rate = (len(off_fail_idx)/off_denom) if off_denom>0 else None
        ker_mismatch_rate = None
        if sel_fail_idx:
            ker_hits = [j for j in sel_fail_idx if K[j]==1]
            ker_mismatch_rate = (len(ker_hits)/len(sel_fail_idx)) if sel_fail_idx else None

        R3_kernel_cols_popcount = int(sum(1 for j in fail_idx if K[j]==1))
        failing_pop = int(len(fail_idx))
        ker_red = bool(failing_pop>0 and R3_kernel_cols_popcount==failing_pop)
        ker_lane_count_auto = (int(sum(1 for j in sel_idx if K[j]==1)) if posed_auto else None)
        ker_lane_count_file = (int(sum(1 for j in range(n_cols) if Lf and Lf[j]==1 and K[j]==1)) if posed_file and isinstance(Lf,list) else None)
        lane_popcount_auto = (int(sel_sum) if posed_auto else None)
        lane_popcount_file = (int(sum(Lf)) if posed_file and isinstance(Lf,list) else None)

        try: vclass_auto = _hard_verdict_class(strict_eq_bool, S, L, K, posed=posed_auto)
        except Exception: vclass_auto = None
        try: vclass_file = _hard_verdict_class(strict_eq_bool, S, (Lf or [0]*n_cols), K, posed=bool(posed_file))
        except Exception: vclass_file = None

        proj_fail_auto = [1 if (S[j]==1 and L[j]==1) else 0 for j in range(n_cols)] if n_cols else []
        proj_failing_cols_sig256 = _hard_co_bitsig256(proj_fail_auto) if posed_auto else None

        try:
            _seed = f"{district_id}{sig8}"
            _h = int(_hash.sha256(_seed.encode('utf-8')).hexdigest()[:2], 16)
            prox_label = f"B{_h % 16:X}"
        except Exception:
            prox_label = "B0"

        cov_row = {
            "written_at_utc": _svr_now_iso(),
            "district_id": str(district_id),
            "fixture_label": str(fixture_label),
            "prox_label": prox_label,
            "dims": {"n2": int(_hard_co_shape(H2)[0]), "n3": int(_hard_co_shape(d3)[1])},
            "sel_mismatch_rate": sel_mismatch_rate,
            "offrow_mismatch_rate": offrow_mismatch_rate,
            "ker_mismatch_rate": ker_mismatch_rate,
            "contradiction_rate": 0.0 if posed_auto else None,
            "lane_popcount_auto": lane_popcount_auto,
            "lane_popcount_file": lane_popcount_file,
            "posed_auto": posed_auto,
            "posed_file": posed_file,
            "ker_red": ker_red,
            "R3_kernel_cols_popcount": R3_kernel_cols_popcount,
            "ker_lane_count_auto": ker_lane_count_auto,
            "ker_lane_count_file": ker_lane_count_file,
            "verdict_class_auto": vclass_auto,
            "verdict_class_file": vclass_file,
            "proj_failing_cols_sig256": proj_failing_cols_sig256,
            "strict_failing_cols_sig256": _hard_co_bitsig256(S),
            "ker_mask_sig256": _hard_co_bitsig256(K),
        }

        _atomic_append_jsonl(cov_path, cov_row)

        # snapshot_tally.jsonl append
        try:
            tally_path = (_Path("logs") / "reports" / "snapshot_tally.jsonl")
            bins = ["GREEN","KER-FILTERED","KER-EXPOSED","FILTERED_OFFLANE","RED_BOTH","RED_UNPOSED"]
            va = vclass_auto or ("RED_UNPOSED" if not posed_auto else None)
            vf = vclass_file or ("RED_UNPOSED" if not posed_file else None)
            tallies = {
                "written_at_utc": _svr_now_iso(),
                "district_id": str(district_id),
                "fixture_label": str(fixture_label),
                "coverage_row_count": 1,
                "counts": {"AUTO": {k:0 for k in bins}, "FILE": {k:0 for k in bins}},
                "lane_popcount_stats": {
                    "auto": {"min": lane_popcount_auto, "max": lane_popcount_auto, "mean": (float(lane_popcount_auto) if lane_popcount_auto is not None else None)},
                    "file": {"min": lane_popcount_file, "max": lane_popcount_file, "mean": (float(lane_popcount_file) if lane_popcount_file is not None else None)},
                },
                "projector_integrity_flag_count": 0,
            }
            if va: tallies["counts"]["AUTO"][va] += 1
            if vf: tallies["counts"]["FILE"][vf] += 1
            _atomic_append_jsonl(tally_path, tallies)
        except Exception:
            pass

    except Exception:
        try:
            cp, _ = _c1_paths()
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(str(cp), "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({"written_at_utc": _svr_now_iso(), "prox_label": "B0"}) + "\\n")
        except Exception:
            pass
    # ---- end C1 coverage append ----

    

    try:
        import streamlit as _st
        _st.session_state["last_bundle_dir"] = str(bdir)
    except Exception:
        pass

    return True, f"v2 compute-only (HARD) 1× bundle → {bdir}", str(bdir)



# ====================== END V2 COMPUTE-ONLY (HARD) ======================






# ====================== Time(τ) local flip toy helpers (v0.1) ======================

def time_tau_strict_core_from_blocks(blocks_B: dict, blocks_C: dict, blocks_H: dict) -> dict:
    """
    Minimal strict core for the Time(τ) toy:
      - pulls d3, C3, H2 from B/C/H blocks
      - checks basic shapes
      - computes H2·d3, C3⊕I3, and R3 = H2·d3 ⊕ (C3⊕I3)

    Returns a dict with matrices and shapes. Raises ValueError on incomplete data.
    """
    d3 = blocks_B.get("3") or []
    C3 = blocks_C.get("3") or []
    H2 = blocks_H.get("2") or []

    if not (d3 and d3[0] and C3 and C3[0] and H2 and H2[0]):
        raise ValueError("Required slices missing (need B[3], C[3], H[2]).")

    mB3, nB3 = _hard_co_shape(d3)
    mC3, nC3 = _hard_co_shape(C3)
    mH2, nH2 = _hard_co_shape(H2)

    # Basic consistency: H2·d3 and C3 must live in the same ambient space.
    if nH2 != mB3:
        raise ValueError(f"Shape mismatch: H2 cols ({nH2}) != d3 rows ({mB3}).")
    if nB3 != nC3:
        raise ValueError(f"Shape mismatch: d3 cols ({nB3}) != C3 cols ({nC3}).")
    if mH2 != mC3:
        raise ValueError(f"Shape mismatch: H2 rows ({mH2}) != C3 rows ({mC3}).")

    sqC = (mC3 == nC3)
    if not sqC:
        # For the toy we require C3 square so that C3⊕I3 and R3 are well-defined.
        raise ValueError("Local flip toy requires square C3 (n3 × n3).")

    H2d3 = _hard_co_mm2(H2, d3)
    I3   = _hard_co_eye(mC3)
    C3pI = _hard_co_xor(C3, I3)
    R3   = _hard_co_xor(H2d3, C3pI)

    return {
        "d3": d3,
        "C3": C3,
        "H2": H2,
        "H2d3": H2d3,
        "C3pI": C3pI,
        "R3": R3,
        "n3": nB3,
        "sqC": sqC,
        "shapes": {
            "H2": (mH2, nH2),
            "d3": (mB3, nB3),
            "C3": (mC3, nC3),
        },
    }


def time_tau_defect_set_from_R3(R3) -> list[int]:
    """
    Defect set D(σ) for the toy:
      D = { j : column j of R3 is non-zero }.
    Matches strict 'failing cols' with sel_all = 1.
    """
    if not R3 or not R3[0]:
        return []
    m, n = len(R3), len(R3[0])
    D = []
    for j in range(n):
        col_nonzero = False
        for i in range(m):
            if int(R3[i][j]) & 1:
                col_nonzero = True
                break
        if col_nonzero:
            D.append(int(j))
    return D


def time_tau_run_local_flip_toy_from_blocks(
    blocks_B: dict,
    blocks_C: dict,
    blocks_H: dict,
    max_flips_per_kind: int = 32,
) -> dict:
    """
    Pure-matrix Time(τ) toy:
      - build strict core from B/C/H blocks
      - compute base defect set D0 and parity p0
      - flip up to max_flips_per_kind bits in H2 and d3
      - for each flip, recompute R3 and log:
          * parity_after, delta_parity
          * changed_cols = symmetric difference of defect sets
          * parity_law_ok: delta_parity == (len(changed_cols) mod 2)

    No certs written, no side-effects.
    """
    core0 = time_tau_strict_core_from_blocks(blocks_B, blocks_C, blocks_H)
    R3_0 = core0["R3"]
    D0   = time_tau_defect_set_from_R3(R3_0)
    p0   = len(D0) % 2

    H2 = core0["H2"]
    d3 = core0["d3"]
    shapes = core0["shapes"]
    mH2, nH2 = shapes["H2"]
    mB3, nB3 = shapes["d3"]

    base_info = {
        "parity": int(p0),
        "defects": D0,
        "H2_shape": [int(mH2), int(nH2)],
        "d3_shape": [int(mB3), int(nB3)],
        "C3_shape": [int(shapes["C3"][0]), int(shapes["C3"][1])],
    }

    # H2 flips
    H2_flips = []
    count = 0
    for i in range(mH2):
        for j in range(nH2):
            if count >= max_flips_per_kind:
                break
            H2p = [row[:] for row in H2]
            H2p[i][j] = (int(H2p[i][j]) ^ 1) & 1

            H2d3_p = _hard_co_mm2(H2p, d3)
            R3_p   = _hard_co_xor(H2d3_p, core0["C3pI"])

            D1 = time_tau_defect_set_from_R3(R3_p)
            p1 = len(D1) % 2
            S  = sorted(set(D0) ^ set(D1))
            delta_p = p0 ^ p1

            H2_flips.append({
                "kind": "H2",
                "i": int(i),
                "j": int(j),
                "parity_after": int(p1),
                "delta_parity": int(delta_p),
                "changed_cols": [int(c) for c in S],
                "changed_cols_size": int(len(S)),
                "parity_law_ok": bool(delta_p == (len(S) & 1)),
            })
            count += 1
        if count >= max_flips_per_kind:
            break

    # d3 flips
    d3_flips = []
    count = 0
    for j in range(mB3):
        for k in range(nB3):
            if count >= max_flips_per_kind:
                break
            d3p = [row[:] for row in d3]
            d3p[j][k] = (int(d3p[j][k]) ^ 1) & 1

            H2d3_p = _hard_co_mm2(H2, d3p)
            R3_p   = _hard_co_xor(H2d3_p, core0["C3pI"])

            D1 = time_tau_defect_set_from_R3(R3_p)
            p1 = len(D1) % 2
            S  = sorted(set(D0) ^ set(D1))
            delta_p = p0 ^ p1

            d3_flips.append({
                "kind": "d3",
                "j": int(j),
                "k": int(k),
                "parity_after": int(p1),
                "delta_parity": int(delta_p),
                "changed_cols": [int(c) for c in S],
                "changed_cols_size": int(len(S)),
                "parity_law_ok": bool(delta_p == (len(S) & 1)),
            })
            count += 1
        if count >= max_flips_per_kind:
            break

    return {
        "schema_version": "time_tau_local_flip_v0.1",
        "base": base_info,
        "H2_flips": H2_flips,
        "d3_flips": d3_flips,
    }


def time_tau_run_local_flip_toy_from_ssot(max_flips_per_kind: int = 32) -> dict:
    """
    Convenience wrapper: resolve B/C/H/U from SSOT and run the toy on those blocks.
    Uses _svr_resolve_all_to_paths; does not write any new certs or receipts.
    """
    pf = _svr_resolve_all_to_paths() or {}
    (pB, bB) = pf.get("B") or (None, {})
    (pC, bC) = pf.get("C") or (None, {})
    (pH, bH) = pf.get("H") or (None, {})
    # U not used for this toy; we only need B/C/H.

    return time_tau_run_local_flip_toy_from_blocks(
        bB, bC, bH,
        max_flips_per_kind=max_flips_per_kind,
    )

# =================== /Time(τ) local flip toy helpers (v0.1) ===================












def _write_time_tau_artifacts(fixture_label, sig8, toy_out, summary, snapshot_id=None, run_id=None):
    """Write Time(τ) local flip toy artifacts (JSON + CSV) under logs/experiments/.

    This is a v2-compatible C1 promotion: it lives outside logs/certs and
    does not affect the main one-press pipeline.
    """
    import csv as _csv
    import json as _json
    import time as _time

    root = _repo_root()
    exp_dir = root / "logs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"time_tau_local_flip__{fixture_label}__{sig8}"
    json_path = exp_dir / f"{base_name}.json"
    csv_path  = exp_dir / f"{base_name}.csv"

    embed = {
        "schema_version": "2.0.0",
        "engine_rev": ENGINE_REV if 'ENGINE_REV' in globals() else None,
        "district_id": (fixture_label.split('_')[0] if fixture_label else None),
        "fixture_label": fixture_label,
        "strict_sig8": sig8,
        "policy": "time_tau_local_flip_toy",
    }
    if snapshot_id:
        embed["snapshot_id"] = snapshot_id
    if run_id:
        embed["run_id"] = run_id

    payload = {
        "schema_version": "time_tau_local_flip_v0.1",
        "written_at_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "embed": embed,
        "base": toy_out.get("base"),
        "H2_flips": toy_out.get("H2_flips") or [],
        "d3_flips": toy_out.get("d3_flips") or [],
        "summary": summary or {},
    }

    with json_path.open("w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2)

    # Flatten flips into a small CSV (one row per flip)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow([
            "kind", "i", "j", "k",
            "parity_after", "delta_parity",
            "changed_cols_size", "parity_law_ok",
        ])
        for row in toy_out.get("H2_flips") or []:
            w.writerow([
                "H2",
                row.get("i"),
                row.get("j"),
                None,
                row.get("parity_after"),
                row.get("delta_parity"),
                row.get("changed_cols_size"),
                row.get("parity_law_ok"),
            ])
        for row in toy_out.get("d3_flips") or []:
            w.writerow([
                "d3",
                row.get("j"),
                row.get("k"),
                None,
                row.get("parity_after"),
                row.get("delta_parity"),
                row.get("changed_cols_size"),
                row.get("parity_law_ok"),
            ])


# =================== Time(τ) local flip sweep helpers (C2, v0.1) ===================

def _time_tau_c2_build_row_from_manifest(rec: dict, max_flips_per_kind: int = 16) -> dict:
    """
    Build a single sweep row for C2 from a manifest record.

    This:
      - loads B/C/H blocks from the manifest paths,
      - runs the Time(τ) local flip toy on those blocks,
      - computes a strict_sig8 compatible with the C1 writer,
      - writes the per-fixture C1 artifacts (JSON + CSV),
      - and returns a flat dict with summary stats for the sweep CSV/JSONL.

    This helper is pure with respect to v2 core: it only reads manifest + inputs
    and writes under logs/experiments/.

    NA semantics:
      - tau_na_reason is a short ASCII tag or None;
      - hard errors are mapped to "ERROR:<ExceptionName>" and the row is still
        returned with neutral stats.
    """
    import json as _json
    import hashlib as _hashlib
    from pathlib import Path as _Path

    if not isinstance(rec, dict):
        rec = {}

    fixture_label = rec.get("fixture_label") or ""
    # For now, district_id is the D-tag prefix of the fixture_label (D2, D3, …).
    district_id = rec.get("district_id") or (fixture_label.split("_")[0] if fixture_label else "DUNKNOWN")

    # Try to pick a snapshot_id from the manifest row, otherwise fall back to the
    # current world snapshot pointer if available.
    try:
        snapshot_id = rec.get("snapshot_id") or _svr_current_snapshot_id()
    except Exception:
        snapshot_id = None

    # Resolve canonical paths from the manifest's "paths" field.
    paths = rec.get("paths") or {}
    B_src = paths.get("B") or ""
    C_src = paths.get("C") or ""
    H_src = paths.get("H") or ""

    # Defaults in case we hit an NA condition.
    tau_na_reason = None
    toy_out: dict = {}
    summary: dict = {}
    strict_sig8 = ""

    try:
        # Load JSON for B/C/H using the same tolerant reader as the SSOT helpers.
        jB, pB, _ = abx_read_json_any(B_src, kind="boundaries")
        jC, pC, _ = abx_read_json_any(C_src, kind="cmap")
        jH, pH, _ = abx_read_json_any(H_src, kind="H")

        blocks_B = _svr_as_blocks_v2(jB, "B")
        blocks_C = _svr_as_blocks_v2(jC, "C")
        blocks_H = _svr_as_blocks_v2(jH, "H")

        # Run the toy from blocks; this may raise on bad shapes.
        toy_out = time_tau_run_local_flip_toy_from_blocks(
            blocks_B,
            blocks_C,
            blocks_H,
            max_flips_per_kind=max_flips_per_kind,
        )

        summary = time_tau_summarize_local_flip_toy(toy_out) or {}

        # Compute a strict_sig8 exactly as in the C1 SSOT panel: hash the minimal
        # strict core (d3, C3, H2) over F2.
        core0 = time_tau_strict_core_from_blocks(blocks_B, blocks_C, blocks_H)
        core_repr = {
            "d3": core0.get("d3"),
            "C3": core0.get("C3"),
            "H2": core0.get("H2"),
        }
        strict_sig8 = _hashlib.sha256(
            _json.dumps(core_repr, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:8]

        # Try to see if the toy marked this fixture as NA in a structured way.
        tau_na_reason = (
            toy_out.get("tau_na_reason")
            or (toy_out.get("base") or {}).get("tau_na_reason")
            or summary.get("tau_na_reason")
            or None
        )

        # Only write per-fixture artifacts for in-domain cases; off-domain / NA
        # will still appear in the sweep CSV but without C1 JSON/CSV.
        if not tau_na_reason:
            try:
                _write_time_tau_artifacts(
                    fixture_label=fixture_label,
                    sig8=strict_sig8,
                    toy_out=toy_out,
                    summary=summary,
                    snapshot_id=snapshot_id,
                    run_id=None,
                )
            except Exception:
                # Best-effort: a failure to write artifacts should not poison
                # the sweep row itself.
                pass

    except Exception as _e:
        # Any hard error is treated as a NA in the sweep; we record a compact
        # reason but keep the row structurally valid.
        tau_na_reason = f"ERROR:{type(_e).__name__}"
        toy_out = {"base": {"parity": 0, "defects": []}, "H2_flips": [], "d3_flips": []}
        summary = {}

    # If summary is empty but we have a toy_out, recompute a minimal summary.
    if not summary and toy_out:
        try:
            summary = time_tau_summarize_local_flip_toy(toy_out) or {}
        except Exception:
            summary = {}

    base_parity = summary.get("base_parity", 0)
    base_defects = list(summary.get("base_defects") or [])

    H2_stats = summary.get("H2") or {}
    d3_stats = summary.get("d3") or {}

    row = {
        "fixture_label": fixture_label,
        "district_id": district_id,
        "snapshot_id": snapshot_id,
        "strict_sig8": strict_sig8 or "",
        "base_parity": base_parity,
        "base_defects_card": len(base_defects),
        "base_defects": base_defects,
        "H2_total": H2_stats.get("total", 0),
        "H2_toggle": H2_stats.get("toggle_parity", 0),
        "H2_preserve": H2_stats.get("preserve_parity", 0),
        "H2_law_ok": H2_stats.get("law_ok", 0),
        "d3_total": d3_stats.get("total", 0),
        "d3_toggle": d3_stats.get("toggle_parity", 0),
        "d3_preserve": d3_stats.get("preserve_parity", 0),
        "d3_law_ok": d3_stats.get("law_ok", 0),
        "global_tau_law_ok": bool(summary.get("global_tau_law_ok", False)),
        "tau_na_reason": tau_na_reason,
    }
    return row


def _time_tau_c2_run_sweep(manifest_path: str | None = None, max_flips_per_kind: int = 16) -> tuple[bool, str, dict]:
    """
    Run the Time(τ) local flip toy across all fixtures in the v2 manifest
    (C2 sweep).

    This:
      - reads logs/manifests/manifest_full_scope.jsonl,
      - for each row, builds a sweep row via _time_tau_c2_build_row_from_manifest,
      - writes a sweep CSV + JSONL under logs/experiments/,
      - and returns (ok, msg, summary_dict).

    It is entirely read-only with respect to v2 certs/receipts: the only writes
    are experiment artifacts under logs/experiments/.

    NA semantics (C2):
      - each row in the sweep has a simple string `tau_na_reason` or None;
      - a fixture is counted as NA iff `tau_na_reason` is truthy or the manifest
        row could not be parsed or the builder raised, in which case
        `tau_na_reason` is set to "ERROR:<ExceptionName>".

    Filenames:
      - time_tau_local_flip_sweep.csv/jsonl when snapshot_ids are mixed;
      - time_tau_local_flip_sweep__{snapshot_id}.csv/jsonl when all rows share
        the same snapshot_id.
    """
    import json as _json
    import csv as _csv
    from pathlib import Path as _Path

    # Locate manifest_full_scope.jsonl if no path is explicitly given.
    if manifest_path is None:
        try:
            manifests_dir = _MANIFESTS_DIR  # preferred if available
        except Exception:
            repo_root = _repo_root()
            manifests_dir = _Path(repo_root) / "logs" / "manifests"
        manifest_path = manifests_dir / "manifest_full_scope.jsonl"
    else:
        manifest_path = _Path(manifest_path)

    if not manifest_path.exists():
        msg = f"Manifest not found at {manifest_path}. Run the v2 core (64×) to populate manifest_full_scope.jsonl first."
        return False, msg, {}

    # Stream manifest rows and build sweep rows.
    rows = []
    n_total = n_in_domain = n_na = n_law_ok = 0
    snapshot_ids = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except Exception:
                n_na += 1
                continue

            sweep_row = _time_tau_c2_build_row_from_manifest(rec, max_flips_per_kind=max_flips_per_kind)
            rows.append(sweep_row)

            n_total += 1
            if sweep_row.get("tau_na_reason"):
                n_na += 1
            else:
                n_in_domain += 1
                if sweep_row.get("global_tau_law_ok"):
                    n_law_ok += 1

            sid = sweep_row.get("snapshot_id")
            if sid:
                snapshot_ids.add(str(sid))

    # If nothing was parsed, bail early.
    if not rows:
        msg = f"No fixtures found in manifest {manifest_path}."
        return False, msg, {}

    # Decide filename suffix based on snapshot_id uniqueness.
    snapshot_suffix = ""
    if len(snapshot_ids) == 1:
        snapshot_suffix = f"__{next(iter(snapshot_ids))}"

    # Prepare experiments dir.
    root = _repo_root()
    exp_dir = root / "logs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = exp_dir / f"time_tau_local_flip_sweep{snapshot_suffix}.csv"
    jsonl_path = exp_dir / f"time_tau_local_flip_sweep{snapshot_suffix}.jsonl"

    # Write CSV with a stable header.
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = _csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write JSONL (canonical JSON per row).
    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for r in rows:
            try:
                f_jsonl.write(_canonical_json(r) + "\n")
            except Exception:
                # Fallback to plain json if canonicalization fails for any reason.
                f_jsonl.write(_json.dumps(r, separators=(",", ":")) + "\n")

    summary = {
        "manifest_path": str(manifest_path),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "n_total": n_total,
        "n_in_domain": n_in_domain,
        "n_na": n_na,
        "n_tau_law_ok": n_law_ok,
    }
    msg = (
        f"τ-sweep completed: {n_total} fixtures, "
        f"{n_in_domain} in-domain, {n_law_ok}/{n_in_domain or 1} τ-law OK, "
        f"{n_na} NA."
    )
    return True, msg, summary

# ================= /Time(τ) local flip sweep helpers (C2, v0.1) ==================

# --- V2 CORE (64×) — one press → receipts → manifest → suite → hist/zip
_st.subheader("V2 — 64× → Receipts → Manifest → Suite/Histograms")

from pathlib import Path as _Path
import json as _json
import uuid as _uuid

def _repo_root():
    try:
        return _REPO_DIR
    except Exception:
        return _Path(__file__).resolve().parents[1]

snapshot_id = _st.text_input(
    "Snapshot id",
    value=_time.strftime("%Y%m%d-%H%M%S", _time.localtime()),
    key="v2_core_snapshot",
)

if _st.button("Run V2 core (64× → receipts → manifest → suite/hist/zip)", key="btn_v2_core_flow"):
    repo_root   = _repo_root()
    inputs_root = repo_root / "app" / "inputs"
    manifests_dir = _MANIFESTS_DIR if '_MANIFESTS_DIR' in globals() else (repo_root / "logs" / "manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    B_dir, H_dir, C_dir = inputs_root / "B", inputs_root / "H", inputs_root / "C"
    U_path = (inputs_root / "U.json").resolve()

    # preflight
    if not (B_dir.exists() and H_dir.exists() and C_dir.exists() and U_path.exists()):
        _st.error(f"Missing inputs dir/file. B:{B_dir.exists()} H:{H_dir.exists()} C:{C_dir.exists()} U:{U_path.exists()}")
    else:
        # discover D; hard-code H (4) and C (8) for 64×
        D_tags = sorted(p.stem for p in B_dir.glob("D*.json"))
        H_tags = ["H00", "H01", "H10", "H11"]
        C_tags = [f"C{n:03b}" for n in range(8)]  # C000..C111

        rows = []
        for D in D_tags:
            B_path = (B_dir / f"{D}.json").resolve()
            if not B_path.exists():
                continue
            for Ht in H_tags:
                Hp = (H_dir / f"{Ht}.json").resolve()
                if not Hp.exists():
                    continue
                for Ct in C_tags:
                    Cp = (C_dir / f"{Ct}.json").resolve()
                    if not Cp.exists():
                        continue
                    fid = f"{D}_{Ht}_{Ct}"
                    rows.append({"fixture_label": fid, "paths": {"B": str(B_path), "C": str(Cp), "H": str(Hp), "U": str(U_path)}})

        if not rows:
            _st.error("No rows produced — check that the 64× fixtures exist on disk.")
            _st.stop()

        # 1) bootstrap manifest with absolute paths
        man_boot = manifests_dir / "manifest_bootstrap__ALL.jsonl"
        man_boot.write_text("\n".join(_json.dumps(r, separators=(",", ":")) for r in rows) + "\n", encoding="utf-8")
        _st.success(f"Bootstrap manifest written with {len(rows)} rows → {man_boot}")

        # 2) run 64× to emit receipts — SNAPSHOT __boot
        snap_boot = (snapshot_id or _time.strftime("%Y%m%d-%H%M%S", _time.localtime())) + "__boot"
        ok1, msg1, cnt1 = run_suite_from_manifest(str(man_boot), snap_boot)
        (_st.success if ok1 else _st.warning)(f"Bootstrap run: {msg1} · rows={cnt1}")

        # 3) regenerate real manifest from receipts
        try:
            ok2, path2, n2 = _v2_regen_manifest_from_receipts()
            (_st.success if ok2 else _st.warning)(f"Manifest regenerated with {n2} rows → {path2}")
        except Exception as e:
            _st.warning(f"Manifest regen failed: {e}")
            ok2, path2, n2 = False, manifests_dir / "manifest_full_scope.jsonl", 0

        # always run suite (no toggles/clicks)
        if not ok2:
            _st.stop()

        real_man = manifests_dir / "manifest_full_scope.jsonl"
        if not real_man.exists():
            _st.error(f"Real manifest not found: {real_man}")
            _st.stop()

        # 4) run REAL manifest — SNAPSHOT __real
        snap_real = (snapshot_id or _time.strftime("%Y%m%d-%H%M%S", _time.localtime())) + "__real"
        ok3, msg3, cnt3 = run_suite_from_manifest(str(real_man), snap_real)
        (_st.success if ok3 else _st.warning)(f"Suite run: {msg3} · rows={cnt3}")

        # 5) histograms over coverage — prefer filtering to __real if supported
        try:
            try:
                okh, msgh, outp = _v2_build_histograms_from_coverage(snap_real)
            except TypeError:
                okh, msgh, outp = _v2_build_histograms_from_coverage()
            (_st.success if okh else _st.warning)(msgh)
            if okh and outp and outp.exists():
                _st.download_button(
                    "Download histograms_v2.json",
                    data=outp.read_bytes(),
                    file_name="histograms_v2.json",
                    mime="application/json",
                    key="btn_v2_download_hist_v2core",
                )
        except Exception as e:
            _st.warning(f"Histogram build failed: {e}")

        # convenience downloads
        try:
            if real_man.exists():
                _st.download_button(
                    "Download manifest_full_scope.jsonl",
                    data=real_man.read_bytes(),
                    file_name="manifest_full_scope.jsonl",
                    mime="text/plain",
                    key="btn_v2_download_manifest_core",
                )
            cov_path = _v2_coverage_path()
            if cov_path.exists():
                _st.download_button(
                    "Download coverage.jsonl",
                    data=cov_path.read_bytes(),
                    file_name="coverage.jsonl",
                    mime="text/plain",
                    key="btn_v2_download_coverage_core",
                )
        except Exception:
            pass

        # 6) FAT zip (certs/receipts + globals) — tied to __real snapshot
        try:
            zip_path = _v2_pack_suite_fat_zip(snap_real)
            if zip_path and zip_path.exists():
                _st.download_button(
                    "Download FAT suite bundle (all certs/receipts)",
                    data=zip_path.read_bytes(),
                    file_name=zip_path.name,
                    mime="application/zip",
                    key="btn_v2_download_suite_fat",
                )
        except Exception as e:
            _st.warning(f"FAT bundle zip failed: {e}")

        # 7) C1 health ping for THIS snapshot (__real)
        try:
            path_csv = _coverage_rollup_write_csv(snapshot_id=snap_real)
            if path_csv and path_csv.exists():
                # read the ALL row for quick chip
                import csv as _csv
                all_row = None
                with path_csv.open("r", encoding="utf-8") as f:
                    r = list(_csv.DictReader(f))
                    for row in r:
                        if row.get("prox_label") == "ALL":
                            all_row = row; break
                if all_row:
                    tail = int(all_row.get("count") or 0)
                    sel = all_row.get("mean_sel_mismatch_rate") or "—"
                    off = all_row.get("mean_offrow_mismatch_rate") or "—"
                    ker = all_row.get("mean_ker_mismatch_rate") or "—"
                    ctr = all_row.get("mean_ctr_rate") or "—"
                    _st.success(f"C1 Health ✅ Healthy · tail={tail} · sel={sel} · off={off} · ker={ker} · ctr={ctr}")
                # single CSV download
                _st.download_button(
                    "Download coverage_rollup.csv",
                    data=path_csv.read_bytes(),
                    file_name="coverage_rollup.csv",
                    mime="text/csv",
                    key="btn_v2_download_cov_rollup_final",
                )
        except Exception as e:
            _st.warning(f"C1 rollup failed: {e}")

        # 8) coverage sanity for __real (expect == executed)
        try:
            parsed = _v2_coverage_count_for_snapshot(snap_real)
            if parsed < cnt3:
                _st.warning(f"Coverage parsed {parsed}/{cnt3} rows for snapshot {snap_real} (expected ≥ executed).")
            else:
                _st.info(f"Coverage parsed rows: {parsed} (snapshot {snap_real})")
        except Exception:
            pass


# ───────────────────────────── C1: Coverage rollup + Health ping ─────────────────────────────
# Helpers are namespaced with _c1_ to avoid collisions.

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


def _c1_rollup_rows(cov_path):
    import json
    from collections import defaultdict

    def _coerce_f(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    acc = defaultdict(lambda: {"count": 0, "sel": [], "off": [], "ker": [], "ctr": []})
    try:
        with open(str(cov_path), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                prox = rec.get("prox_label") or "B0"
                a = acc[prox]
                a["count"] += 1

                # prefer new keys, fallback to legacy
                sel_v = rec.get("mismatch_sel")
                if sel_v is None:
                    sel_v = rec.get("sel_mismatch_rate")
                off_v = rec.get("mismatch_offrow")
                if off_v is None:
                    off_v = rec.get("offrow_mismatch_rate")
                ker_v = rec.get("mismatch_ker")
                if ker_v is None:
                    ker_v = rec.get("ker_mismatch_rate")
                ctr_v = rec.get("mismatch_ctr")
                if ctr_v is None:
                    ctr_v = rec.get("contradiction_rate")

                for key, raw in (("sel", sel_v), ("off", off_v), ("ker", ker_v), ("ctr", ctr_v)):
                    v = _coerce_f(raw)
                    if v is not None:
                        a[key].append(v)
    except FileNotFoundError:
        return []

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    rows = []
    for prox in sorted(acc.keys()):
        a = acc[prox]
        rows.append({
            "prox_label": prox,
            "count": a["count"],
            "mean_sel_mismatch_rate": _mean(a["sel"]),
            "mean_offrow_mismatch_rate": _mean(a["off"]),
            "mean_ker_mismatch_rate": _mean(a["ker"]),
            "mean_ctr_rate": _mean(a["ctr"]),
            "mean_contradiction_rate": _mean(a["ctr"]),
        })
    return rows


def _c1_write_rollup_csv(cov_path, csv_out):
    import json, csv
    from collections import defaultdict

    def _coerce_f(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    acc = defaultdict(lambda: {"count": 0, "sel": [], "off": [], "ker": [], "ctr": []})
    try:
        with open(str(cov_path), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                prox = rec.get("prox_label") or "B0"
                a = acc[prox]
                a["count"] += 1

                # prefer new keys, fallback to legacy
                sel_v = rec.get("mismatch_sel")
                if sel_v is None:
                    sel_v = rec.get("sel_mismatch_rate")
                off_v = rec.get("mismatch_offrow")
                if off_v is None:
                    off_v = rec.get("offrow_mismatch_rate")
                ker_v = rec.get("mismatch_ker")
                if ker_v is None:
                    ker_v = rec.get("ker_mismatch_rate")
                ctr_v = rec.get("mismatch_ctr")
                if ctr_v is None:
                    ctr_v = rec.get("contradiction_rate")

                for key, raw in (("sel", sel_v), ("off", off_v), ("ker", ker_v), ("ctr", ctr_v)):
                    v = _coerce_f(raw)
                    if v is not None:
                        a[key].append(v)
    except FileNotFoundError:
        pass

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    rows = []
    for prox in sorted(acc.keys()):
        a = acc[prox]
        rows.append({
            "prox_label": prox,
            "count": a["count"],
            "mean_sel_mismatch_rate": _mean(a["sel"]),
            "mean_offrow_mismatch_rate": _mean(a["off"]),
            "mean_ker_mismatch_rate": _mean(a["ker"]),
            "mean_ctr_rate": _mean(a["ctr"]),
            "mean_contradiction_rate": _mean(a["ctr"]),
        })

    hdr = [
        "prox_label",
        "count",
        "mean_sel_mismatch_rate",
        "mean_offrow_mismatch_rate",
        "mean_ker_mismatch_rate",
        "mean_ctr_rate",
        "mean_contradiction_rate",
    ]
    with open(str(csv_out), "w", encoding="utf-8", newline="") as wfh:
        w = csv.writer(wfh)
        w.writerow(hdr)
        for r in rows:
            w.writerow([
                r["prox_label"], r["count"],
                "" if r["mean_sel_mismatch_rate"] is None else f"{r['mean_sel_mismatch_rate']:.4f}",
                "" if r["mean_offrow_mismatch_rate"] is None else f"{r['mean_offrow_mismatch_rate']:.4f}",
                "" if r["mean_ker_mismatch_rate"] is None else f"{r['mean_ker_mismatch_rate']:.4f}",
                "" if r["mean_ctr_rate"] is None else f"{r['mean_ctr_rate']:.4f}",
                "" if r["mean_contradiction_rate"] is None else f"{r['mean_contradiction_rate']:.4f}",
            ])


def _c1_health_ping(jsonl_path: _Path, tail: int = 50):
    """
    Compute mean mismatch rates over the last `tail` events in coverage.jsonl.
    Returns dict or None if no data.

    Supports both legacy keys (sel_mismatch_rate, offrow_mismatch_rate, ...)
    and the newer mismatch_* names.
    """
    buf = []
    for rec in _c1_iter_jsonl(jsonl_path):
        buf.append(rec)
    if not buf:
        return None

    if tail > 0 and len(buf) > tail:
        buf = buf[-tail:]

    def _coerce_f(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    def avg(candidates):
        vals = []
        for rec in buf:
            val = None
            for key in candidates:
                if key in rec and rec[key] not in (None, "", "NA"):
                    val = _coerce_f(rec[key])
                    if val is not None:
                        vals.append(val)
                        break
        return (sum(vals) / len(vals)) if vals else None

    mean_sel = avg(["mismatch_sel", "sel_mismatch_rate"])
    mean_off = avg(["mismatch_offrow", "offrow_mismatch_rate"])
    mean_ker = avg(["mismatch_ker", "ker_mismatch_rate"])
    mean_ctr = avg(["mismatch_ctr", "contradiction_rate"])

    return {
        "tail": len(buf),
        "mean_sel_mismatch_rate": mean_sel,
        "mean_offrow_mismatch_rate": mean_off,
        "mean_ker_mismatch_rate": mean_ker,
        "mean_contradiction_rate": mean_ctr,
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


# ── UI: Coverage rollup (read-only C1 chip) ──
with st.expander("C1 — Coverage rollup & health ping", expanded=False):
    cov_path, csv_out = _c1_paths()
    st.caption(f"Source: {cov_path} · Rollup: {csv_out}")

    # Health chip (tail window over latest coverage.jsonl)
    hp = _c1_health_ping(cov_path, tail=64)
    if hp is None:
        st.info("coverage.jsonl not found yet — run the V2 core once to produce coverage events.")
    else:
        emoji, label, _ = _c1_badge(hp)

        def _fmt(x):
            return "—" if x is None else f"{x:.3f}"

        # Try to load ALL row from rollup for a stable ctr fallback
        all_row = None
        try:
            import csv as _csv
            if csv_out.exists():
                with csv_out.open("r", encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
                all_row = next((row for row in rows if row.get("prox_label") == "ALL"), None)
        except Exception:
            all_row = None  # purely informational

        def _ctr_display():
            # 1) Prefer live mean_contradiction_rate from coverage.jsonl
            v = hp.get("mean_contradiction_rate")
            if v is not None:
                return _fmt(v)
            # 2) Fallback to rollup's mean_ctr_rate (or mean_contradiction_rate) if present
            if all_row:
                raw = all_row.get("mean_ctr_rate") or all_row.get("mean_contradiction_rate")
                if raw not in (None, "", "NA"):
                    try:
                        return _fmt(float(raw))
                    except Exception:
                        pass
            # 3) Last resort
            return "—"

        st.markdown(
            f"**C1 Health** {emoji} {label} · tail={hp['tail']} · "
            f"sel={_fmt(hp.get('mean_sel_mismatch_rate'))} · "
            f"off={_fmt(hp.get('mean_offrow_mismatch_rate'))} · "
            f"ker={_fmt(hp.get('mean_ker_mismatch_rate'))} · "
            f"ctr={_ctr_display()}"
        )

        # Time(τ) health from coverage_rollup.csv (ALL row, if present)
        if all_row:
            def _int_or_zero(x):
                try:
                    return int(x)
                except Exception:
                    return 0

            n_c3 = _int_or_zero(all_row.get("time_tau_n_fixtures_with_c3"))
            n_pred_true = _int_or_zero(all_row.get("time_tau_tau_pred_true"))
            n_pred_false = _int_or_zero(all_row.get("time_tau_tau_pred_false"))
            n_emp_true = _int_or_zero(all_row.get("time_tau_tau_emp_true"))
            n_emp_false = _int_or_zero(all_row.get("time_tau_tau_emp_false"))
            tau_mismatch = _int_or_zero(all_row.get("time_tau_tau_mismatch_count"))

            if n_c3:
                st.markdown("**Time(τ) health**")
                if tau_mismatch:
                    st.warning(
                        f"τ-law mismatches on {tau_mismatch} / {n_c3} fixtures "
                        f"(pred true/false={n_pred_true}/{n_pred_false}, "
                        f"emp true/false={n_emp_true}/{n_emp_false})."
                    )
                else:
                    st.success(
                        f"τ-law prediction agrees with C3 on all {n_c3} fixtures "
                        f"(pred true/false={n_pred_true}/{n_pred_false}, "
                        f"emp true/false={n_emp_true}/{n_emp_false})."
                    )

        # Optional: surface last ALL row from coverage_rollup.csv (if present), read-only
        if all_row:
            st.caption(
                "Last rollup (ALL) · "
                f"count={all_row.get('count')} · "
                f"sel={all_row.get('mean_sel_mismatch_rate') or '—'} · "
                f"off={all_row.get('mean_offrow_mismatch_rate') or '—'} · "
                f"ker={all_row.get('mean_ker_mismatch_rate') or '—'} · "
                f"ctr={all_row.get('mean_ctr_rate') or '—'}"
            )

def time_tau_summarize_local_flip_toy(toy_out: dict) -> dict:
    """
    Compute small, stable summary stats for a time_tau_local_flip_v0.1 run.

    Returns a dict with keys:
      - schema_version
      - base_parity, base_defects
      - H2: {total, toggle_parity, preserve_parity, law_ok, law_total}
      - d3: {total, toggle_parity, preserve_parity, law_ok, law_total}
      - global_tau_law_ok: bool
    """
    if not isinstance(toy_out, dict):
        return {}

    base = toy_out.get("base") or {}
    try:
        base_parity = int(base.get("parity", 0)) & 1
    except Exception:
        base_parity = 0
    base_defects = list(base.get("defects") or [])

    H2_flips = list(toy_out.get("H2_flips") or [])
    d3_flips = list(toy_out.get("d3_flips") or [])

    def _stats(flips: list[dict]) -> dict:
        total = len(flips)
        toggle = 0
        preserve = 0
        law_ok = 0
        for f in flips:
            try:
                dp = int(f.get("delta_parity", 0)) & 1
            except Exception:
                dp = 0
            if dp == 1:
                toggle += 1
            else:
                preserve += 1
            if f.get("parity_law_ok"):
                law_ok += 1
        return {
            "total": total,
            "toggle_parity": toggle,
            "preserve_parity": preserve,
            "law_ok": law_ok,
            "law_total": total,
        }

    H2_stats = _stats(H2_flips)
    d3_stats = _stats(d3_flips)
    global_ok = (
        H2_stats["law_ok"] == H2_stats["law_total"]
        and d3_stats["law_ok"] == d3_stats["law_total"]
    )

    return {
        "schema_version": "time_tau_local_flip_summary_v0.1",
        "base_parity": base_parity,
        "base_defects": base_defects,
        "H2": H2_stats,
        "d3": d3_stats,
        "global_tau_law_ok": bool(global_ok),
    }
# ───────────────────────── Lab — Time(τ) local flip toy (read-only) ─────────────────────────
with st.expander("Lab — Time(τ) local flip toy (v3-prelude, read-only)", expanded=False):
    st.caption(
        "Toy experiment around strict residual R₃ for the current SSOT fixture. "
        "Flips a few H₂ / d₃ bits and logs how the obstruction parity changes. "
        "No certs or receipts are written."
    )
    nonce = st.session_state.get("_ui_nonce", "tau")
    max_flips = st.number_input(
        "Max flips per kind (H₂ / d₃)",
        min_value=1,
        max_value=256,
        value=16,
        step=1,
        key=f"time_tau_flip_max_{nonce}",
    )

    if st.button("Run local flip toy on current SSOT", key=f"time_tau_flip_run_{nonce}"):
        try:
            toy_out = time_tau_run_local_flip_toy_from_ssot(
                max_flips_per_kind=int(max_flips)
            )
            summary = time_tau_summarize_local_flip_toy(toy_out)

            base_parity = summary.get("base_parity")
            base_defects = summary.get("base_defects") or []
            H2_stats = summary.get("H2") or {}
            d3_stats = summary.get("d3") or {}
            global_ok = summary.get("global_tau_law_ok", False)

            st.markdown("**τ-toy summary**")
            st.markdown(
                f"Base parity: `{base_parity}` · "
                f"base defects: `{base_defects}`"
            )
            st.markdown(
                f"H₂ flips: {H2_stats.get('total', 0)} total · "
                f"{H2_stats.get('toggle_parity', 0)} toggle parity · "
                f"{H2_stats.get('preserve_parity', 0)} preserve · "
                f"{H2_stats.get('law_ok', 0)}/{H2_stats.get('law_total', 0)} obey τ-law"
            )
            st.markdown(
                f"d₃ flips: {d3_stats.get('total', 0)} total · "
                f"{d3_stats.get('toggle_parity', 0)} toggle parity · "
                f"{d3_stats.get('preserve_parity', 0)} preserve · "
                f"{d3_stats.get('law_ok', 0)}/{d3_stats.get('law_total', 0)} obey τ-law"
            )
            st.markdown(
                "Global τ-law: "
                + ("✅ all flips consistent" if global_ok else "⚠️ violation(s) found")
            )

            # Raw toy output for inspection / debugging
            st.json(toy_out)

            # C1: write τ-toy artifacts under logs/experiments/
            try:
                pf = _svr_resolve_all_to_paths() or {}
                (pB, bB) = pf.get("B") or (None, {})
                (pC, bC) = pf.get("C") or (None, {})
                (pH, bH) = pf.get("H") or (None, {})
                if pB and pC and pH:
                    from pathlib import Path as _Path
                    d_tag = _Path(pB).stem
                    h_tag = _Path(pH).stem
                    c_tag = _Path(pC).stem
                    fixture_label = f"{d_tag}_{h_tag}_{c_tag}"

                    import hashlib, json as _json
                    core0 = time_tau_strict_core_from_blocks(bB, bC, bH)
                    core_repr = {
                        "d3": core0["d3"],
                        "C3": core0["C3"],
                        "H2": core0["H2"],
                    }
                    strict_sig8 = hashlib.sha256(
                        _json.dumps(core_repr, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()[:8]

                    run_id = pf.get("run_id")
                    _write_time_tau_artifacts(fixture_label, strict_sig8, toy_out, summary, snapshot_id, run_id)
                    base_name = f"time_tau_local_flip__{fixture_label}__{strict_sig8}"
                    st.success(
                        f"Artifacts written under logs/experiments/ as '{base_name}.json' and '{base_name}.csv'"
                    )

                    # Offer direct downloads for quick sanity checks
                    try:
                        _root = _repo_root()
                        _json_path = _root / "logs" / "experiments" / f"{base_name}.json"
                        _csv_path  = _root / "logs" / "experiments" / f"{base_name}.csv"
                        if _json_path.exists():
                            with _json_path.open("r", encoding="utf-8") as _f:
                                st.download_button(
                                    "Download τ-toy JSON",
                                    _f.read(),
                                    file_name=f"{base_name}.json",
                                    mime="application/json",
                                    key=f"dl_tau_json_{base_name}",
                                )
                        if _csv_path.exists():
                            with _csv_path.open("r", encoding="utf-8") as _f:
                                st.download_button(
                                    "Download τ-toy CSV",
                                    _f.read(),
                                    file_name=f"{base_name}.csv",
                                    mime="text/csv",
                                    key=f"dl_tau_csv_{base_name}",
                                )
                    except Exception as _dl_err:
                        st.warning(f"Download helpers failed: {_dl_err}")
                else:
                    st.warning("Cannot write τ-toy artifacts: missing B/C/H paths in SSOT.")
            except Exception as ee:
                st.warning(f"Artifact write failed: {ee}")
        except Exception as e:
            st.warning(f"Local flip toy failed: {e}")

    # --- C2: τ-sweep over manifest (v2) ---
    st.markdown("---")
    st.markdown("**C2 — τ-sweep over manifest**")
    st.caption(
        "Run the same Time(τ) local flip toy for every fixture in the current v2 manifest. "
        "Writes per-fixture τ-toy artifacts (C1) and a sweep CSV/JSONL under logs/experiments/."
    )

    max_flips_c2 = st.number_input(
        "Max flips per kind (H₂ / d₃) for sweep",
        min_value=1,
        max_value=256,
        value=int(max_flips),
        step=1,
        key=f"time_tau_flip_max_c2_{nonce}",
    )

    if st.button("Run τ-sweep over current manifest (C2)", key=f"time_tau_sweep_run_{nonce}"):
        ok, msg, sweep = _time_tau_c2_run_sweep(
            max_flips_per_kind=int(max_flips_c2),
        )
        if ok:
            st.success(msg)
            st.caption(
                f"{sweep.get('n_total')} fixtures · "
                f"{sweep.get('n_in_domain')} in-domain · "
                f"{sweep.get('n_tau_law_ok')}/{sweep.get('n_in_domain') or 1} τ-law OK · "
                f"{sweep.get('n_na')} NA"
            )
            csv_path = sweep.get("csv_path")
            jsonl_path = sweep.get("jsonl_path")
            st.write("Sweep CSV:", csv_path)
            st.write("Sweep JSONL:", jsonl_path)

            # Download buttons for sweep artifacts (QoL only; failures here must
            # not affect the core C2 sweep semantics).
            try:
                from pathlib import Path as _Path
                if csv_path:
                    _csv_path = _Path(csv_path)
                    if _csv_path.exists():
                        with _csv_path.open("r", encoding="utf-8") as _f:
                            st.download_button(
                                "Download τ-sweep CSV",
                                _f.read(),
                                file_name=_csv_path.name,
                                mime="text/csv",
                                key=f"time_tau_sweep_csv_dl_{nonce}",
                            )
                if jsonl_path:
                    _jsonl_path = _Path(jsonl_path)
                    if _jsonl_path.exists():
                        with _jsonl_path.open("r", encoding="utf-8") as _f:
                            st.download_button(
                                "Download τ-sweep JSONL",
                                _f.read(),
                                file_name=_jsonl_path.name,
                                mime="application/json",
                                key=f"time_tau_sweep_jsonl_dl_{nonce}",
                            )
            except Exception as _dl_err:
                st.warning(f"Download helpers for τ-sweep failed: {_dl_err}")
        else:
            st.warning(msg)



# ----------------------------------------------------------------------
# Time(τ) — C3 v0.1 (recompute check) — Pass A + Pass B1
# ----------------------------------------------------------------------

# Ensure derived-worlds directory entry exists without mutating the DIRS
# literal near the top. Only C3 helpers will write into this directory.
if "c3_worlds" not in DIRS:
    DIRS["c3_worlds"] = "app/inputs/c3_derived_worlds"


# C3 v0.2 — τ-recompute constants (Pass A).
TIME_TAU_C3_RECEIPT_SCHEMA_VERSION = "time_tau_c3_receipt_v0.2"
TIME_TAU_C3_MUTATED_SCHEMA_VERSION = "time_tau_c3_mutated_v0.1"
TIME_TAU_C3_OBS_SCHEMA_VERSION = "time_tau_c3_obs_v0.1"
TIME_TAU_C3_PRED_SCHEMA_VERSION = "time_tau_c3_pred_v0.1"

# Flip-regime caps for C3 v0.2 (S1).
TIME_TAU_C3_MAX_FLIPS_H2 = 16
TIME_TAU_C3_MAX_FLIPS_D3 = 16


# Fixed NA codes for C3 v0.1. These literal strings are the only allowed
# values for any future `c3_na_reason` fields.
C3_NA = {
    "C2_ARTIFACT_MISSING": "C2_ARTIFACT_MISSING",
    "NO_LAW_OK_FLIPS": "NO_LAW_OK_FLIPS",
    "BASELINE_CERT_MISSING": "BASELINE_CERT_MISSING",
    "MUTATION_WRITE_ERROR": "MUTATION_WRITE_ERROR",
    "SOLVER_ERROR": "SOLVER_ERROR",
    "OBSERVATION_PARSE_ERROR": "OBSERVATION_PARSE_ERROR",
}




def _time_tau_c3_na(code_key: str) -> str:
    """
    Return the canonical NA reason string for a given code key.

    Raises:
        KeyError: if ``code_key`` is not a valid key in ``C3_NA``.

    Usage example:
        na_reason = _time_tau_c3_na("C2_ARTIFACT_MISSING")
    """
    return C3_NA[code_key]


# Receipt shape (Python-level, documentation only in Pass A/B1).
# Future helpers will populate a dict matching this schema:
#
#   c3_receipt = {
#       "schema_version": "time_tau_c3_recompute_v0.1",
#
#       "base": {
#           "fixture_label": str,
#           "district_id": str,
#           "baseline_strict_sig8": str,
#           "baseline_snapshot_id": str | None,
#       },
#
#       "flip_ref": {
#           "kind": str,                  # "H2" or "d3"
#           "i": int | None,
#           "j": int,
#           "k": int | None,
#           "flip_index": int,
#           "tau_toy_sig8": str,         # same as baseline_strict_sig8 in v0.1
#       },
#
#       "tau_toy_prediction": {
#           "base_parity_before": int,   # 0/1 from τ-toy base
#           "parity_after": int,         # 0/1
#           "delta_parity": int,         # 0/1
#           "parity_law_ok": bool,
#       },
#
#       "solver_observation": {
#           "base_parity_before": int,   # 0/1 recomputed by solver
#           "base_parity_after": int,    # 0/1
#           "delta_parity": int,         # 0/1
#           "strict_sig8_before": str,
#           "strict_sig8_after": str,
#           "verdict_before": str | bool,
#           "verdict_after": str | bool,
#           "snapshot_before": str | None,
#           "snapshot_after": str | None,
#       },
#
#       "c3_pass": bool,
#       "c3_na_reason": None | str,      # None or one of C3_NA[...] values
#   }
#
# No writers or solver calls are introduced in Pass A/B1; this block is
# purely structural/constant and safe to import in any context.




# C4 v0.1 — C3 stability rollup (constants only for Pass A).

TIME_TAU_C3_ROLLUP_SCHEMA_VERSION = "time_tau_c3_rollup_v0.2"

TIME_TAU_C3_ROLLUP_CSV = "time_tau_c3_rollup.csv"
TIME_TAU_C3_ROLLUP_JSONL = "time_tau_c3_rollup.jsonl"

TIME_TAU_TAU_MISMATCH_SCHEMA_VERSION = "time_tau_c3_tau_mismatch_v0.1"
TIME_TAU_TAU_MISMATCH_CSV = "time_tau_c3_tau_mismatches.csv"
TIME_TAU_TAU_MISMATCH_JSONL = "time_tau_c3_tau_mismatches.jsonl"

TIME_TAU_COVERAGE_SCHEMA_VERSION = "time_tau_coverage_v0.1"
TIME_TAU_COVERAGE_UNIT = "time_tau"
TIME_TAU_COVERAGE_KIND_C4_HEALTH = "tau_c4_health"



# Each C4 rollup row (per (district_id, fixture_label, strict_sig8)):
#
# c3_rollup_row = {
#     "schema_version": TIME_TAU_C3_ROLLUP_SCHEMA_VERSION,
#
#     "district_id": district_id,        # e.g. "D2", "D3"
#     "fixture_label": fixture_label,    # e.g. "D2_H00_C000"
#     "strict_sig8": strict_sig8,        # baseline strict_sig8 / tau_toy_sig8
#     "snapshot_id": snapshot_id,        # may be None
#
#     "n_flips_total": int,
#     "n_flips_H2": int,
#     "n_flips_d3": int,
#
#     "n_pass": int,                     # c3_pass == True, c3_na_reason is None
#     "n_fail": int,                     # c3_pass == False, c3_na_reason is None
#     "n_na_total": int,                 # c3_na_reason is not None
#
#     # per-NA reason counts (names match C3_NA keys)
#     "na_C2_ARTIFACT_MISSING": int,
#     "na_NO_LAW_OK_FLIPS": int,
#     "na_BASELINE_CERT_MISSING": int,
#     "na_MUTATION_WRITE_ERROR": int,
#     "na_SOLVER_ERROR": int,
#     "na_OBSERVATION_PARSE_ERROR": int,
#
#     # simple presence flags
#     "has_H2": bool,
#     "has_d3": bool,
# }


# ----------------------------------------------------------------------
# C4 v0.1 — Pass B: C3 receipt ingestion (path listing + normalization)
# ----------------------------------------------------------------------


def _time_tau_c4_iter_c3_receipt_paths() -> list[str]:
    """
    Return a list of absolute paths to all C3 receipt JSON files.

    Pattern:
      logs/experiments/time_tau_c3_recompute__*.json
    """
    root = _repo_root()
    exp_dir = root / "logs" / "experiments"
    if not exp_dir.exists():
        return []

    out: list[str] = []
    try:
        for p in exp_dir.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if not name.startswith("time_tau_c3_recompute__"):
                continue
            if not name.endswith(".json"):
                continue
            out.append(str(p))
    except Exception:
        # Be conservative: if listing fails, just return what we have so far.
        return sorted(out)

    return sorted(out)



def _time_tau_c4_parse_c3_receipt(path: str) -> dict | None:
    """
    Read a single C3 receipt JSON and normalize it for rollup.

    Returns a dict with the fields needed for aggregation, or None if the
    file is unreadable / malformed.
    """
    try:
        j, _p, _tag = abx_read_json_any(path, kind="c3_receipt")  # tolerant loader
    except Exception:
        return None

    if not isinstance(j, dict):
        return None

    base = j.get("base") or {}
    flip_ref = j.get("flip_ref") or {}
    tau_pred = j.get("tau_toy_prediction") or {}
    obs = j.get("solver_observation") or {}

    # Be robust against malformed payloads: require dicts, else treat as missing.
    if not isinstance(tau_pred, dict):
        tau_pred = {}
    if not isinstance(obs, dict):
        obs = {}

    expected_tau_law_holds = tau_pred.get("expected_tau_law_holds")

    obs_tau_law_holds = obs.get("tau_law_holds")
    metrics = obs.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    obs_parity_after = metrics.get("parity_after")
    obs_n_defect_cols = metrics.get("n_defect_cols")

    district_id = base.get("district_id") or "DUNKNOWN"
    fixture_label = base.get("fixture_label") or "UNKNOWN"
    strict_sig8 = base.get("baseline_strict_sig8") or ""
    snapshot_id = base.get("baseline_snapshot_id")  # may be None

    kind = flip_ref.get("kind") or "UNKNOWN"

    c3_pass = bool(j.get("c3_pass"))
    na_reason = j.get("c3_na_reason")
    if na_reason is not None:
        na_reason = str(na_reason)

    return {
        "path": path,
        "district_id": district_id,
        "fixture_label": fixture_label,
        "strict_sig8": strict_sig8,
        "snapshot_id": snapshot_id,
        "kind": kind,              # "H2" | "d3" | ...
        "c3_pass": c3_pass,
        "c3_na_reason": na_reason,  # None or one of C3_NA.values()
        "expected_tau_law_holds": expected_tau_law_holds,
        "obs_tau_law_holds": obs_tau_law_holds,
        "obs_parity_after": obs_parity_after,
        "obs_n_defect_cols": obs_n_defect_cols,
    }


def _time_tau_c4_build_rollup_rows(norm_rows: list[dict]) -> list[dict]:
    """
    Aggregate normalized C3 rows into per-fixture rollup dicts.

    Key is (district_id, fixture_label, strict_sig8).
    """
    agg: dict[tuple[str, str, str], dict] = {}

    for r in norm_rows:
        district_id = r.get("district_id") or "DUNKNOWN"
        fixture_label = r.get("fixture_label") or "UNKNOWN"
        strict_sig8 = r.get("strict_sig8") or ""
        snapshot_id = r.get("snapshot_id")

        key = (district_id, fixture_label, strict_sig8)
        row = agg.get(key)
        if row is None:
            row = {
                "schema_version": TIME_TAU_C3_ROLLUP_SCHEMA_VERSION,
                "district_id": district_id,
                "fixture_label": fixture_label,
                "strict_sig8": strict_sig8,
                "snapshot_id": snapshot_id,

                "n_flips_total": 0,
                "n_flips_H2": 0,
                "n_flips_d3": 0,

                "n_pass_H2": 0,
                "n_fail_H2": 0,
                "n_na_H2": 0,
                "n_pass_d3": 0,
                "n_fail_d3": 0,
                "n_na_d3": 0,

                "n_pass": 0,
                "n_fail": 0,
                "n_na_total": 0,

                "expected_tau_law_holds": None,
                "empirical_tau_law_holds": None,

                "na_C2_ARTIFACT_MISSING": 0,
                "na_NO_LAW_OK_FLIPS": 0,
                "na_BASELINE_CERT_MISSING": 0,
                "na_MUTATION_WRITE_ERROR": 0,
                "na_SOLVER_ERROR": 0,
                "na_OBSERVATION_PARSE_ERROR": 0,

                "has_H2": False,
                "has_d3": False,
            }
            agg[key] = row

        # Ensure snapshot_id is filled once if present.
        if row.get("snapshot_id") is None and snapshot_id is not None:
            row["snapshot_id"] = snapshot_id

        # Carry expected tau-law prediction into the fixture row (keep-first).
        exp = r.get("expected_tau_law_holds")
        if exp is not None and row.get("expected_tau_law_holds") is None:
            row["expected_tau_law_holds"] = bool(exp)

        # Increment counters.
        row["n_flips_total"] += 1

        kind = r.get("kind")
        is_H2 = (kind == "H2")
        is_d3 = (kind == "d3")

        if is_H2:
            row["n_flips_H2"] += 1
            row["has_H2"] = True
        elif is_d3:
            row["n_flips_d3"] += 1
            row["has_d3"] = True

        c3_pass = bool(r.get("c3_pass"))
        na_reason = r.get("c3_na_reason")

        if na_reason is not None:
            row["n_na_total"] += 1
            if is_H2:
                row["n_na_H2"] += 1
            elif is_d3:
                row["n_na_d3"] += 1
            # Map NA code to field name if possible.
            for code_key in C3_NA.keys():
                if na_reason == C3_NA[code_key]:
                    field = f"na_{code_key}"
                    if field in row:
                        row[field] += 1
                    break
        else:
            if c3_pass:
                row["n_pass"] += 1
                if is_H2:
                    row["n_pass_H2"] += 1
                elif is_d3:
                    row["n_pass_d3"] += 1
            else:
                row["n_fail"] += 1
                if is_H2:
                    row["n_fail_H2"] += 1
                elif is_d3:
                    row["n_fail_d3"] += 1

    # Derive empirical tau-law per fixture based on flip outcomes.
    for row in agg.values():
        tested = row.get("n_flips_total", 0) - row.get("n_na_total", 0)
        if tested <= 0:
            row["empirical_tau_law_holds"] = None
        elif row.get("n_fail", 0) > 0:
            row["empirical_tau_law_holds"] = False
        else:
            row["empirical_tau_law_holds"] = True

    # Return as a list, sorted for determinism.
    out = list(agg.values())
    out.sort(key=lambda r: (r["district_id"], r["fixture_label"], r["strict_sig8"]))
    return out


def _time_tau_c4_build_district_summary(rows: list[dict]) -> dict[str, dict]:
    """
    Build a simple district-level summary from per-fixture rows.
    """
    summary: dict[str, dict] = {}

    for r in rows:
        d = r.get("district_id") or "DUNKNOWN"
        s = summary.get(d)
        if s is None:
            s = {
                "district_id": d,
                "n_fixtures_with_c3": 0,
                "n_flips_total": 0,
                "n_pass": 0,
                "n_fail": 0,
                "n_na_total": 0,
                "n_fixtures_tau_pred_true": 0,
                "n_fixtures_tau_pred_false": 0,
                "n_fixtures_tau_emp_true": 0,
                "n_fixtures_tau_emp_false": 0,
            }
            summary[d] = s

        # Aggregate core flip counts.
        s["n_fixtures_with_c3"] += 1
        s["n_flips_total"] += r.get("n_flips_total", 0)
        s["n_pass"] += r.get("n_pass", 0)
        s["n_fail"] += r.get("n_fail", 0)
        s["n_na_total"] += r.get("n_na_total", 0)

        # Aggregate tau-law fixture-level booleans.
        exp = r.get("expected_tau_law_holds")
        if exp is True:
            s["n_fixtures_tau_pred_true"] += 1
        elif exp is False:
            s["n_fixtures_tau_pred_false"] += 1

        emp = r.get("empirical_tau_law_holds")
        if emp is True:
            s["n_fixtures_tau_emp_true"] += 1
        elif emp is False:
            s["n_fixtures_tau_emp_false"] += 1

    return summary





def _time_tau_c4_build_tau_mismatch_rows(rows: list[dict]) -> list[dict]:
    """
    From C4 v0.2 per-fixture rows, build a list of tau-mismatch rows.

    A row is included iff expected_tau_law_holds and empirical_tau_law_holds
    are both bool and disagree.
    """
    mismatches: list[dict] = []

    for r in rows:
        exp = r.get("expected_tau_law_holds")
        emp = r.get("empirical_tau_law_holds")

        # Require both sides to be concrete booleans.
        if exp is None or emp is None:
            continue

        exp_b = bool(exp)
        emp_b = bool(emp)
        if exp_b == emp_b:
            # They agree → not a mismatch.
            continue

        # Polarity-aware kind.
        if exp_b is True and emp_b is False:
            mismatch_kind = "EXP_TRUE_EMP_FALSE"   # local optimistic, global obstructed
        elif exp_b is False and emp_b is True:
            mismatch_kind = "EXP_FALSE_EMP_TRUE"   # local pessimistic, global OK
        else:
            mismatch_kind = "UNKNOWN"

        m = {
            "schema_version": TIME_TAU_TAU_MISMATCH_SCHEMA_VERSION,
            "district_id": r.get("district_id"),
            "fixture_label": r.get("fixture_label"),
            "strict_sig8": r.get("strict_sig8"),
            "snapshot_id": r.get("snapshot_id"),
            "expected_tau_law_holds": exp_b,
            "empirical_tau_law_holds": emp_b,
            "tau_mismatch_kind": mismatch_kind,
            # Flip statistics (copied from C4 row).
            "n_flips_total": r.get("n_flips_total", 0),
            "n_flips_H2": r.get("n_flips_H2", 0),
            "n_flips_d3": r.get("n_flips_d3", 0),
            "n_pass_H2": r.get("n_pass_H2", 0),
            "n_fail_H2": r.get("n_fail_H2", 0),
            "n_na_H2": r.get("n_na_H2", 0),
            "n_pass_d3": r.get("n_pass_d3", 0),
            "n_fail_d3": r.get("n_fail_d3", 0),
            "n_na_d3": r.get("n_na_d3", 0),
            "n_pass": r.get("n_pass", 0),
            "n_fail": r.get("n_fail", 0),
            "n_na_total": r.get("n_na_total", 0),
            "has_H2": r.get("has_H2", False),
            "has_d3": r.get("has_d3", False),
        }
        mismatches.append(m)

    # Optional deterministic ordering (district, fixture, sig).
    mismatches.sort(
        key=lambda m: (
            m.get("district_id") or "",
            m.get("fixture_label") or "",
            m.get("strict_sig8") or "",
        )
    )

    return mismatches




def _time_tau_c4_build_coverage_ping(
    rows: list[dict],
    summary: dict[str, dict] | None,
) -> dict:
    """
    Build a single Time(τ) coverage event from C4 rollup rows + district summary.

    This is what we log into coverage.jsonl for C1.
    """
    # Global aggregates from district summary.
    n_fixtures_with_c3 = 0
    n_pred_true = 0
    n_pred_false = 0
    n_emp_true = 0
    n_emp_false = 0

    for _d_key, s in (summary or {}).items():
        s = s or {}
        n_fixtures_with_c3 += int(s.get("n_fixtures_with_c3", 0) or 0)
        n_pred_true += int(s.get("n_fixtures_tau_pred_true", 0) or 0)
        n_pred_false += int(s.get("n_fixtures_tau_pred_false", 0) or 0)
        n_emp_true += int(s.get("n_fixtures_tau_emp_true", 0) or 0)
        n_emp_false += int(s.get("n_fixtures_tau_emp_false", 0) or 0)

    # Mismatches from the same logic as the gallery.
    mismatch_rows = _time_tau_c4_build_tau_mismatch_rows(rows or [])
    tau_mismatch_count = len(mismatch_rows)

    ping = {
        "schema_version": TIME_TAU_COVERAGE_SCHEMA_VERSION,
        "unit": TIME_TAU_COVERAGE_UNIT,
        "kind": TIME_TAU_COVERAGE_KIND_C4_HEALTH,
        "n_fixtures_with_c3": n_fixtures_with_c3,
        "n_tau_pred_true": n_pred_true,
        "n_tau_pred_false": n_pred_false,
        "n_tau_emp_true": n_emp_true,
        "n_tau_emp_false": n_emp_false,
        "tau_mismatch_count": tau_mismatch_count,
    }

    # Attach snapshot_id so C1 can see this Time(τ) ping.
    # Prefer the current world snapshot pointer; if unavailable, fall back
    # to any snapshot_id present on the C4 rollup rows.
    snapshot_id = None
    try:
        snapshot_id = _svr_current_snapshot_id()
    except Exception:
        snapshot_id = None

    if not snapshot_id:
        for r in rows or []:
            sid = r.get("snapshot_id")
            if sid:
                snapshot_id = sid
                break

    if snapshot_id:
        ping["snapshot_id"] = snapshot_id

    return ping

def _time_tau_c4_write_rollup_csv(rows: list[dict]) -> str:
    """
    Write the C3 rollup rows to logs/reports/time_tau_c3_rollup.csv.

    Returns the output path as a string.
    """
    root = _repo_root()
    rep_dir = root / "logs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    outp = rep_dir / TIME_TAU_C3_ROLLUP_CSV

    import csv

    fieldnames = [
        "district_id",
        "fixture_label",
        "strict_sig8",
        "snapshot_id",
        "n_flips_total",
        "n_flips_H2",
        "n_flips_d3",
        "n_pass_H2",
        "n_fail_H2",
        "n_na_H2",
        "n_pass_d3",
        "n_fail_d3",
        "n_na_d3",
        "n_pass",
        "n_fail",
        "n_na_total",
        "na_C2_ARTIFACT_MISSING",
        "na_NO_LAW_OK_FLIPS",
        "na_BASELINE_CERT_MISSING",
        "na_MUTATION_WRITE_ERROR",
        "na_SOLVER_ERROR",
        "na_OBSERVATION_PARSE_ERROR",
        "has_H2",
        "has_d3",
        "expected_tau_law_holds",
        "empirical_tau_law_holds",
        "schema_version",
    ]

    with outp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return str(outp)


def _time_tau_c4_write_rollup_jsonl(rows: list[dict]) -> str:
    """
    Write the C3 rollup rows to logs/reports/time_tau_c3_rollup.jsonl.

    Returns the output path as a string.
    """
    root = _repo_root()
    rep_dir = root / "logs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    outp = rep_dir / TIME_TAU_C3_ROLLUP_JSONL

    with outp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(_json.dumps(r, sort_keys=True))
            fh.write("\n")

    return str(outp)


def _time_tau_c4_write_tau_mismatches_csv(rows: list[dict]) -> str:
    """
    Write tau mismatch rows to logs/reports/time_tau_c3_tau_mismatches.csv.
    """
    root = _repo_root()
    rep_dir = root / "logs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    outp = rep_dir / TIME_TAU_TAU_MISMATCH_CSV

    import csv
    fieldnames = [
        "district_id",
        "fixture_label",
        "strict_sig8",
        "snapshot_id",
        "expected_tau_law_holds",
        "empirical_tau_law_holds",
        "tau_mismatch_kind",
        "n_flips_total",
        "n_flips_H2",
        "n_flips_d3",
        "n_pass_H2",
        "n_fail_H2",
        "n_na_H2",
        "n_pass_d3",
        "n_fail_d3",
        "n_na_d3",
        "n_pass",
        "n_fail",
        "n_na_total",
        "has_H2",
        "has_d3",
        "schema_version",
    ]

    with outp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return str(outp)


def _time_tau_c4_write_tau_mismatches_jsonl(rows: list[dict]) -> str:
    """
    Write tau mismatch rows to logs/reports/time_tau_c3_tau_mismatches.jsonl.
    """
    root = _repo_root()
    rep_dir = root / "logs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    outp = rep_dir / TIME_TAU_TAU_MISMATCH_JSONL

    with outp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(_json.dumps(r, sort_keys=True))
            fh.write("\n")

    return str(outp)


def time_tau_c4_build_rollup() -> tuple[list[dict], dict[str, dict]]:
    """
    C4 v0.1 rollup builder.

    - scans C3 receipts
    - parses them to normalized rows
    - aggregates to per-fixture rollup rows
    - writes CSV + JSONL under logs/reports

    Returns (rows, district_summary).
    """
    paths = _time_tau_c4_iter_c3_receipt_paths()
    norm_rows: list[dict] = []
    for p in paths:
        r = _time_tau_c4_parse_c3_receipt(p)
        if r is not None:
            norm_rows.append(r)

    rows = _time_tau_c4_build_rollup_rows(norm_rows)
    summary = _time_tau_c4_build_district_summary(rows)

    _time_tau_c4_write_rollup_csv(rows)
    _time_tau_c4_write_rollup_jsonl(rows)

    # Tau local/global mismatch gallery
    mismatch_rows = _time_tau_c4_build_tau_mismatch_rows(rows)
    _time_tau_c4_write_tau_mismatches_csv(mismatch_rows)
    _time_tau_c4_write_tau_mismatches_jsonl(mismatch_rows)

    # Time(τ) coverage ping for C1 (best-effort).
    try:
        ping = _time_tau_c4_build_coverage_ping(rows, summary)
        _v2_coverage_append(ping)
    except Exception:
        # Best-effort; do not break C4 rollup if coverage logging fails.
        pass

    return rows, summary

def _time_tau_c3_load_baseline_context(
    fixture_label: str,
    strict_sig8: str,
) -> tuple[bool, str, dict]:
    """
    Load minimal context for a C3 check:

      - C2 τ-toy artifact for this (fixture_label, strict_sig8)
      - baseline strict cert (best-effort)
      - district_id and snapshot_id (best-effort)

    Returns (ok, msg, ctx), where ctx contains:

      {
        "fixture_label": ...,
        "district_id": ...,
        "snapshot_id": ...,
        "strict_sig8": ...,
        "tau_artifact": {...},
        "strict_cert": {...} or None,
      }

    If the τ-toy artifact is missing, returns ok=False with
    msg=C3_NA["C2_ARTIFACT_MISSING"].
    """
    try:
        # Locate τ-toy artifact written by the C2 helpers.
        root = _repo_root()
    except Exception:
        # Fallback: treat current working directory as repo root.
        root = _Path(".").resolve()

    exp_dir = root / "logs" / "experiments"
    base_name = f"time_tau_local_flip__{fixture_label}__{strict_sig8}"
    json_path = exp_dir / f"{base_name}.json"

    if not json_path.exists():
        return False, C3_NA["C2_ARTIFACT_MISSING"], {}

    try:
        raw = json_path.read_text(encoding="utf-8")
        tau_artifact = _json.loads(raw)
    except Exception as e:
        # Treat parse failures as artifact-missing for v0.1.
        return False, C3_NA["C2_ARTIFACT_MISSING"], {}

    embed = tau_artifact.get("embed") or {}
    # Prefer district_id from embed; fall back to fixture prefix or DUNKNOWN.
    district_id = (
        embed.get("district_id")
        or (fixture_label.split("_")[0] if "_" in fixture_label else None)
        or "DUNKNOWN"
    )
    snapshot_id = embed.get("snapshot_id")

    # Best-effort baseline strict cert lookup: we never fail the whole
    # context just because the cert is missing.
    strict_cert = None
    try:
        certs_root = _CERTS_DIR if "_CERTS_DIR" in globals() else (root / "logs" / "certs")
        bundle_dir = certs_root / str(district_id) / str(fixture_label) / str(strict_sig8)
        strict_path = bundle_dir / f"overlap__{district_id}__strict__{strict_sig8}.json"
        if strict_path.exists():
            strict_cert = _json.loads(strict_path.read_text(encoding="utf-8"))
    except Exception:
        # Swallow errors here; the rest of the context is still usable.
        strict_cert = None

    ctx = {
        "fixture_label": fixture_label,
        "district_id": district_id,
        "snapshot_id": snapshot_id,
        "strict_sig8": strict_sig8,
        "tau_artifact": tau_artifact,
        "strict_cert": strict_cert,
    }
    return True, "OK", ctx


def _time_tau_c3_load_baseline_context_from_fixture(
    fixture_label: str,
    strict_sig8: str,
) -> tuple[bool, str | None, dict | None]:
    """
    S0 — Baseline context lookup for C3 v0.2.

    This is a thin wrapper over the legacy
    ``_time_tau_c3_load_baseline_context`` plus the C2 manifest wiring.

    It returns (ok, na_reason, base_ctx) where:

        ok=True:
            na_reason is None.
            base_ctx is a dict with at least::

                {
                    "district_id": str,
                    "fixture_label": str,
                    "strict_sig8": str,
                    "snapshot_id": str | None,
                    "paths": {
                        "B": str,
                        "C": str,
                        "H": str,
                        "d3": str,
                        "U": str,
                    },
                    "tau_toy_prediction": {
                        "schema_version": TIME_TAU_C3_PRED_SCHEMA_VERSION,
                        "expected_tau_law_holds": bool,
                    },
                }

        ok=False:
            base_ctx is None and na_reason is one of::

                C3_NA["C2_ARTIFACT_MISSING"],
                C3_NA["BASELINE_CERT_MISSING"].
    """
    # 1) Reuse the v0.1 loader to get τ-artifact + strict cert.
    ok_ctx, msg_ctx, ctx0 = _time_tau_c3_load_baseline_context(
        fixture_label=fixture_label,
        strict_sig8=strict_sig8,
    )
    if not ok_ctx:
        na = msg_ctx if msg_ctx in C3_NA.values() else _time_tau_c3_na("C2_ARTIFACT_MISSING")
        return False, na, None

    tau_artifact = ctx0.get("tau_artifact")
    strict_cert = ctx0.get("strict_cert")

    if not isinstance(tau_artifact, dict):
        # Treat malformed τ-artifacts as missing for C3 purposes.
        return False, _time_tau_c3_na("C2_ARTIFACT_MISSING"), None

    # strict_cert is optional for C3 v0.2: we only use it opportunistically
    # to lift district_id / snapshot_id metadata when present.
    if not isinstance(strict_cert, dict):
        strict_cert = None

    # 2) Resolve baseline B/C/H/U from the C2 manifest (full-scope).
    rec = _time_tau_c3_find_manifest_row_for_fixture(fixture_label)
    if not isinstance(rec, dict):
        # We do not introduce a separate NA code for manifest issues;
        # from C3's perspective the baseline is not fully anchored.
        return False, _time_tau_c3_na("BASELINE_CERT_MISSING"), None

    paths_raw = rec.get("paths") or {}
    B_path = paths_raw.get("B") or rec.get("B")
    C_path = paths_raw.get("C") or rec.get("C")
    H_path = paths_raw.get("H") or rec.get("H")
    U_path = paths_raw.get("U") or rec.get("U")

    if not (B_path and C_path and H_path and U_path):
        return False, _time_tau_c3_na("BASELINE_CERT_MISSING"), None

    B_path = str(B_path)
    C_path = str(C_path)
    H_path = str(H_path)
    U_path = str(U_path)

    # d3 lives inside the boundaries file; we surface it explicitly so that
    # S2 helpers have a stable slot to look at.
    paths = {
        "B": B_path,
        "C": C_path,
        "H": H_path,
        "d3": B_path,
        "U": U_path,
    }

    # 3) Build τ-prediction stub with a single boolean flag.
    #
    # For C3 v0.2 we take a coarse global view:
    #   expected_tau_law_holds == True  iff  all recorded flips in the τ-toy
    #   artifact satisfy parity_law_ok.
    #
    # If any flip reports parity_law_ok == False, we say the toy predicts a
    # violation of the τ-law on this baseline.
    H2_flips = tau_artifact.get("H2_flips") or []
    d3_flips = tau_artifact.get("d3_flips") or []
    all_flips: list[dict] = []
    for seq in (H2_flips, d3_flips):
        for f in seq:
            if isinstance(f, dict):
                all_flips.append(f)

    expected_tau_law_holds = True
    for f in all_flips:
        if not bool(f.get("parity_law_ok")):
            expected_tau_law_holds = False
            break

    tau_toy_prediction = {
        "schema_version": TIME_TAU_C3_PRED_SCHEMA_VERSION,
        "expected_tau_law_holds": bool(expected_tau_law_holds),
    }

    # 4) District/snapshot metadata.
    district_id = ctx0.get("district_id")
    if not district_id and isinstance(strict_cert, dict):
        embed = strict_cert.get("embed") or {}
        district_id = embed.get("district_id")

    if not district_id:
        district_id = (fixture_label.split("_")[0] if "_" in fixture_label else "DUNKNOWN")

    snapshot_id = ctx0.get("snapshot_id")
    if not snapshot_id and isinstance(strict_cert, dict):
        embed = strict_cert.get("embed") or {}
        snapshot_id = embed.get("snapshot_id")

    base_ctx = {
        "district_id": str(district_id or "DUNKNOWN"),
        "fixture_label": str(fixture_label),
        "strict_sig8": str(strict_sig8),
        "snapshot_id": snapshot_id,
        "paths": paths,
        "tau_toy_prediction": tau_toy_prediction,
        # Full τ-toy artifact (H₂/d₃ flip logs) carried through for S1.
        "tau_toy_artifact": tau_artifact,
    }
    return True, None, base_ctx


def _time_tau_c3_iter_flips_for_fixture(base_ctx: dict) -> list[dict]:
    """
    S1 — Flip regime for C3 v0.2.

    Given a baseline context, return a deterministic list of ``flip_ref`` dicts.
    Each flip_ref has the shape::

        {
            "kind": "H2" | "d3",
            "i": int | None,
            "j": int | None,
            "k": int | None,
        }

    The sequence is:

        * Deterministic (ordered as in the τ-toy artifact).
        * Bounded by TIME_TAU_C3_MAX_FLIPS_H2 / _D3.
        * Only includes flips that are 'law-OK' under the toy's constraints
          (``parity_law_ok == True``).

    The caller is responsible for treating an empty list as an S1 NA
    (``NO_LAW_OK_FLIPS``).
    """
    if not isinstance(base_ctx, dict):
        return []

    tau_artifact = base_ctx.get("tau_toy_artifact") or {}
    if not isinstance(tau_artifact, dict):
        return []

    H2_flips_raw = tau_artifact.get("H2_flips") or []
    d3_flips_raw = tau_artifact.get("d3_flips") or []

    def _select_flips(raw_seq, kind: str, max_flips: int) -> list[dict]:
        out: list[dict] = []
        if not isinstance(raw_seq, list):
            return out
        for f in raw_seq:
            if not isinstance(f, dict):
                continue
            # Only keep τ-law OK flips.
            if not bool(f.get("parity_law_ok")):
                continue

            # Normalize indices; older artifacts may omit some coordinates.
            i = f.get("i")
            j = f.get("j")
            k = f.get("k")

            flip_ref = {
                "kind": kind,
                "i": int(i) if i is not None else None,
                "j": int(j) if j is not None else None,
                "k": int(k) if k is not None else None,
            }
            out.append(flip_ref)
            if len(out) >= max_flips:
                break
        return out

    flips_H2 = _select_flips(H2_flips_raw, "H2", TIME_TAU_C3_MAX_FLIPS_H2)
    flips_d3 = _select_flips(d3_flips_raw, "d3", TIME_TAU_C3_MAX_FLIPS_D3)

    # Deterministic concatenation: H₂ first, then d₃.
    return flips_H2 + flips_d3
def _time_tau_c3_derived_artifact_path(
    base_ctx: dict,
    flip_ref: dict,
) -> str:
    """
    Pure path planner (no IO) for the C3 v0.2 mutated matrix artifact.

    Layout (relative to repo root)::

        DIRS["c3_worlds"]/ {fixture_label}__{strict_sig8}/{kind}/
            flip_i{i}_j{j}.json   (for H2)
            flip_j{j}_k{k}.json   (for d3)

    This helper is intentionally boring and does no filesystem work; it only
    normalizes the relative path as a string.
    """
    fixture_label = str(base_ctx.get("fixture_label") or "UNKNOWN")
    strict_sig8 = str(base_ctx.get("strict_sig8") or "UNKNOWN")
    kind = str(flip_ref.get("kind") or "unknown")

    base_dir = DIRS.get("c3_worlds", "app/inputs/c3_derived_worlds")

    i = flip_ref.get("i")
    j = flip_ref.get("j")
    k = flip_ref.get("k")

    if kind == "H2":
        i_tag = "x" if i is None else int(i)
        j_tag = "x" if j is None else int(j)
        fname = f"flip_i{i_tag}_j{j_tag}.json"
        subdir = "H2"
    elif kind == "d3":
        j_tag = "x" if j is None else int(j)
        k_tag = "x" if k is None else int(k)
        fname = f"flip_j{j_tag}_k{k_tag}.json"
        subdir = "d3"
    else:
        fname = "flip_unknown.json"
        subdir = kind or "unknown"

    try:
        rel = _Ph(base_dir) / f"{fixture_label}__{strict_sig8}" / subdir / fname  # type: ignore[name-defined]
        return str(rel)
    except Exception:
        # Fallback to a simple string join if Path is unavailable for any reason.
        return f"{base_dir}/{fixture_label}__{strict_sig8}/{subdir}/{fname}"


def _time_tau_c3_build_derived_world(
    base_ctx: dict,
    flip_ref: dict,
) -> tuple[bool, str | None, dict | None]:
    """
    S2 — Derived world materialization for C3 v0.2.

    Returns (ok, na_reason, derived_world).

    ok=True:
        na_reason is None.
        derived_world is a descriptor with::

            {
                "district_id": ...,
                "fixture_label": ...,
                "strict_sig8": ...,
                "snapshot_id": ...,
                "paths": { "B": ..., "C": ..., "H": ..., "U": ... },
                "flip_ref": {...},
                "derived_artifact_path": "...",
                # optional: "mutated_matrix_sig8": "...",
            }

        Semantics:

          * Baseline matrix is read from base_ctx["paths"]["H"] (for H2) or
            base_ctx["paths"]["d3"] / ["B"] (for d3). In practice, d3 is
            stored inside the boundaries ("B") JSON.
          * Flip is applied in memory over GF(2) to produce an expected
            matrix.
          * A separate mutated-matrix artifact is written/verified at the
            canonical path returned by _time_tau_c3_derived_artifact_path.
            If the file already exists but does not match the new expected
            matrix (or key metadata), it is overwritten.
          * The actual solver inputs (B/C/H/U) for the derived world are
            obtained via the existing v0.1 helper
            _time_tau_c3_build_derived_paths_for_flip, so v0.1 semantics
            remain intact.

    ok=False:
        na_reason = C3_NA["MUTATION_WRITE_ERROR"].
        derived_world is None.
    """
    if not isinstance(base_ctx, dict) or not isinstance(flip_ref, dict):
        return False, C3_NA["MUTATION_WRITE_ERROR"], None

    kind = flip_ref.get("kind")
    if kind not in ("H2", "d3"):
        return False, C3_NA["MUTATION_WRITE_ERROR"], None

    fixture_label = str(base_ctx.get("fixture_label") or "UNKNOWN")
    strict_sig8 = str(base_ctx.get("strict_sig8") or "")
    district_id = str(base_ctx.get("district_id") or "DUNKNOWN")
    snapshot_id = base_ctx.get("snapshot_id")

    paths_ctx = base_ctx.get("paths") or {}
    B_src = paths_ctx.get("B")
    C_src = paths_ctx.get("C")
    H_src = paths_ctx.get("H")
    U_src = paths_ctx.get("U")
    d3_src = paths_ctx.get("d3") or B_src

    if not (B_src and C_src and H_src and U_src and d3_src):
        return False, C3_NA["MUTATION_WRITE_ERROR"], None

    # Step 1: resolve manifest row and build derived solver paths using the
    # existing v0.1 helper. This keeps the strict-core wiring stable.
    try:
        rec = _time_tau_c3_find_manifest_row_for_fixture(fixture_label)
    except Exception:
        rec = None
    if not isinstance(rec, dict):
        return False, C3_NA["MUTATION_WRITE_ERROR"], None

    ok_paths, _msg_paths, derived_paths = _time_tau_c3_build_derived_paths_for_flip(
        rec=rec,
        flip=flip_ref,
        strict_sig8=strict_sig8,
    )
    if not ok_paths or not isinstance(derived_paths, dict):
        return False, C3_NA["MUTATION_WRITE_ERROR"], None

    # Step 2: recompute the expected mutated matrix from the baseline.
    try:
        i_val = flip_ref.get("i")
        j_val = flip_ref.get("j")
        k_val = flip_ref.get("k")
        i_idx = int(i_val) if i_val is not None else None
        j_idx = int(j_val) if j_val is not None else None
        k_idx = int(k_val) if k_val is not None else None

        if kind == "H2":
            jH, _pH, _src_tag = abx_read_json_any(H_src, kind="H")  # type: ignore[arg-type]
            blocks_H = _svr_as_blocks_v2(jH, "H")
            H2 = blocks_H.get("2")
            if H2 is None or i_idx is None or j_idx is None:
                return False, C3_NA["MUTATION_WRITE_ERROR"], None

            # Deep copies to avoid mutating the original object.
            baseline_matrix = [list(row) for row in H2]
            expected_matrix = [list(row) for row in H2]

            if i_idx < 0 or j_idx < 0 or i_idx >= len(expected_matrix) or j_idx >= len(expected_matrix[i_idx]):
                return False, C3_NA["MUTATION_WRITE_ERROR"], None

            val = expected_matrix[i_idx][j_idx]
            try:
                bit = int(val) & 1
            except Exception:
                bit = 0
            expected_matrix[i_idx][j_idx] = 1 - bit
            baseline_artifact_path = str(H_src)

        else:  # kind == "d3"
            jB, _pB, _src_tag = abx_read_json_any(d3_src, kind="boundaries")  # type: ignore[arg-type]
            blocks_B = _svr_as_blocks_v2(jB, "B")
            d3_block = blocks_B.get("3")
            if d3_block is None or j_idx is None or k_idx is None:
                return False, C3_NA["MUTATION_WRITE_ERROR"], None

            baseline_matrix = [list(row) for row in d3_block]
            expected_matrix = [list(row) for row in d3_block]

            if j_idx < 0 or k_idx < 0 or j_idx >= len(expected_matrix) or k_idx >= len(expected_matrix[j_idx]):
                return False, C3_NA["MUTATION_WRITE_ERROR"], None

            val = expected_matrix[j_idx][k_idx]
            try:
                bit = int(val) & 1
            except Exception:
                bit = 0
            expected_matrix[j_idx][k_idx] = 1 - bit
            baseline_artifact_path = str(d3_src)

        # Hash convention: use the same sig8 helper as strict_sig8.
        _baseline_json, baseline_sig8 = _canon_dump_and_sig8(baseline_matrix)
        _mut_json, mutated_sig8 = _canon_dump_and_sig8(expected_matrix)

        derived_artifact_path = _time_tau_c3_derived_artifact_path(base_ctx, flip_ref)

        root = _repo_root()
        mut_p = root / derived_artifact_path

        mutated_payload = {
            "schema_version": TIME_TAU_C3_MUTATED_SCHEMA_VERSION,
            "fixture_label": fixture_label,
            "strict_sig8": strict_sig8,
            "snapshot_id": snapshot_id,
            "kind": kind,
            "flip_ref": {
                "kind": kind,
                "i": i_idx,
                "j": j_idx,
                "k": k_idx,
            },
            "baseline_artifact_path": baseline_artifact_path,
            "baseline_matrix_hash": baseline_sig8,
            "mutated_matrix_hash": mutated_sig8,
            "matrix": expected_matrix,
        }

        # Step 3: verify/create the mutated artifact (Option B semantics).
        try:
            if mut_p.exists():
                try:
                    existing = _json.loads(mut_p.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}

                if (
                    existing.get("schema_version") != TIME_TAU_C3_MUTATED_SCHEMA_VERSION
                    or existing.get("fixture_label") != fixture_label
                    or existing.get("strict_sig8") != strict_sig8
                    or existing.get("kind") != kind
                    or existing.get("flip_ref") != mutated_payload["flip_ref"]
                    or existing.get("matrix") != expected_matrix
                ):
                    _hard_co_write_json(mut_p, mutated_payload)
            else:
                _hard_co_write_json(mut_p, mutated_payload)
        except Exception:
            return False, C3_NA["MUTATION_WRITE_ERROR"], None

    except Exception:
        return False, C3_NA["MUTATION_WRITE_ERROR"], None

    derived_world = {
        "district_id": district_id,
        "fixture_label": fixture_label,
        "strict_sig8": strict_sig8,
        "snapshot_id": snapshot_id,
        "paths": derived_paths,
        "flip_ref": {
            "kind": kind,
            "i": i_idx,
            "j": j_idx,
            "k": k_idx,
        },
        "derived_artifact_path": derived_artifact_path,
        "mutated_matrix_sig8": mutated_sig8,
    }
    return True, None, derived_world
def time_tau_c3_eval_world(derived_world: dict) -> dict:
    """
    S3 — Evaluate a derived world in τ-mode (core entrypoint for C3 v0.2).

    This helper is intentionally narrow:

      * It expects a ``derived_world`` descriptor as produced by
        ``_time_tau_c3_build_derived_world`` (in particular a ``paths``
        dict with keys {"B","C","H","U"}).
      * It loads B/C/H from disk using the generic JSON readers.
      * It builds the strict Time(τ) core and computes the defect set and
        parity on the derived world only.

    It does **not**:

      * re-run the baseline world,
      * write any certs, coverage events, or manifest entries.

    On success, it returns a raw ``core_obs`` dict that at minimum contains::

        {
            "parity_after": int,
            "defect_cols": list[int],
            "n_defect_cols": int,
        }

    On catastrophic failure (IO/shape issues, missing slices, etc.) this
    function is allowed to raise; the outer C3 driver is responsible for
    catching such errors and mapping them to ``C3_NA["SOLVER_ERROR"]``.
    """
    if not isinstance(derived_world, dict):
        raise ValueError("C3 S3: derived_world must be a dict")

    paths = derived_world.get("paths") or {}
    B_path = paths.get("B")
    C_path = paths.get("C")
    H_path = paths.get("H")

    if not (B_path and C_path and H_path):
        raise ValueError("C3 S3: incomplete derived paths (need B, C, H)")

    # Load JSON blocks for the derived world.
    jB, _pB, _tagB = abx_read_json_any(B_path, kind="boundaries")  # type: ignore[arg-type]
    jC, _pC, _tagC = abx_read_json_any(C_path, kind="C")           # type: ignore[arg-type]
    jH, _pH, _tagH = abx_read_json_any(H_path, kind="H")           # type: ignore[arg-type]

    blocks_B = _svr_as_blocks_v2(jB, "B")
    blocks_C = _svr_as_blocks_v2(jC, "C")
    blocks_H = _svr_as_blocks_v2(jH, "H")

    core = time_tau_strict_core_from_blocks(blocks_B, blocks_C, blocks_H)
    R3 = core.get("R3")
    D = time_tau_defect_set_from_R3(R3)
    parity_after = len(D) % 2

    core_obs: dict = {
        "parity_after": int(parity_after),
        "defect_cols": [int(c) for c in D],
        "n_defect_cols": int(len(D)),
    }
    return core_obs


def _time_tau_c3_normalize_observation(core_obs: dict) -> dict:
    """
    Normalize a raw core_obs dict into the minimal C3 solver_observation.

    Returns a dict of the form::

        {
            "schema_version": TIME_TAU_C3_OBS_SCHEMA_VERSION,
            "tau_law_holds": bool,
            "metrics": {
                "parity_after": int,
                "n_defect_cols": int,
            },
        }

    The current v0.2 policy is intentionally simple:

        tau_law_holds == True  iff  parity_after == 0

    If ``core_obs`` is missing the required field or it cannot be coerced
    to an integer, this helper raises ``ValueError``; the caller should map
    that to ``C3_NA["OBSERVATION_PARSE_ERROR"]`` at S4.
    """
    if not isinstance(core_obs, dict):
        raise ValueError("C3 S4: core_obs must be a dict")

    try:
        parity_after = int(core_obs.get("parity_after"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("C3 S4: missing or invalid 'parity_after'") from exc

    tau_law_holds = bool(parity_after == 0)

    solver_observation = {
        "schema_version": TIME_TAU_C3_OBS_SCHEMA_VERSION,
        "tau_law_holds": tau_law_holds,
        "metrics": {
            "parity_after": int(parity_after),
            "n_defect_cols": int(core_obs.get("n_defect_cols", 0)),
        },
    }
    return solver_observation
def _time_tau_c3_judge(
    tau_pred: dict,
    solver_observation: dict,
) -> tuple[bool, str | None, bool | None]:
    """
    S4 — Determine c3_pass or NA for a single flip.

    Inputs:
        tau_pred: {
            "schema_version": TIME_TAU_C3_PRED_SCHEMA_VERSION,
            "expected_tau_law_holds": bool,
        }

        solver_observation: {
            "schema_version": TIME_TAU_C3_OBS_SCHEMA_VERSION,
            "tau_law_holds": bool,
            ...
        }

    Returns (ok, na_reason, c3_pass):

        ok=True:
            na_reason is None.
            c3_pass is True or False, with::

                c3_pass = (expected_tau_law_holds == tau_law_holds)

        ok=False:
            na_reason = C3_NA["OBSERVATION_PARSE_ERROR"]
            c3_pass = None
    """
    # Basic shape and schema checks.
    if not isinstance(tau_pred, dict) or not isinstance(solver_observation, dict):
        return False, _time_tau_c3_na("OBSERVATION_PARSE_ERROR"), None

    try:
        pred_schema = tau_pred.get("schema_version")
        obs_schema = solver_observation.get("schema_version")
        if pred_schema != TIME_TAU_C3_PRED_SCHEMA_VERSION:
            raise ValueError("unexpected tau_pred schema_version")
        if obs_schema != TIME_TAU_C3_OBS_SCHEMA_VERSION:
            raise ValueError("unexpected solver_observation schema_version")

        expected = bool(tau_pred["expected_tau_law_holds"])
        actual = bool(solver_observation["tau_law_holds"])
    except Exception:
        return False, _time_tau_c3_na("OBSERVATION_PARSE_ERROR"), None

    return True, None, bool(expected == actual)












def _time_tau_c3_select_flips(tau_artifact: dict) -> dict:
    """
    From a τ-toy artifact, pick at most one H2 flip and at most one d3 flip.

    Policy v0.1:
      - first H2 flip with parity_law_ok == True
      - first d3 flip with parity_law_ok == True

    Returns:
      {
        "H2": { ... } or None,
        "d3": { ... } or None,
      }

    Each non-None entry has fields:
      {
        "kind": "H2" | "d3",
        "i": i or None,
        "j": j,
        "k": k or None,
        "flip_index": idx,
        "base_parity_before": base_parity,
        "parity_after": parity_after,
        "delta_parity": delta_parity,
        "parity_law_ok": bool(parity_law_ok),
      }
    """
    base = tau_artifact.get("base") or {}
    try:
        base_parity = int(base.get("parity", 0))
    except Exception:
        base_parity = 0

    flips_out: dict[str, dict | None] = {"H2": None, "d3": None}

    # H2 flips
    H2_flips = tau_artifact.get("H2_flips") or []
    for idx, rec in enumerate(H2_flips):
        try:
            if not bool(rec.get("parity_law_ok")):
                continue
        except Exception:
            continue
        flip = {
            "kind": "H2",
            "i": rec.get("i"),
            "j": rec.get("j"),
            "k": None,
            "flip_index": int(idx),
            "base_parity_before": base_parity,
            "parity_after": int(rec.get("parity_after", 0)),
            "delta_parity": int(rec.get("delta_parity", 0)),
            "parity_law_ok": bool(rec.get("parity_law_ok")),
        }
        flips_out["H2"] = flip
        break

    # d3 flips
    d3_flips = tau_artifact.get("d3_flips") or []
    for idx, rec in enumerate(d3_flips):
        try:
            if not bool(rec.get("parity_law_ok")):
                continue
        except Exception:
            continue
        flip = {
            "kind": "d3",
            "i": None,
            "j": rec.get("j"),
            "k": rec.get("k"),
            "flip_index": int(idx),
            "base_parity_before": base_parity,
            "parity_after": int(rec.get("parity_after", 0)),
            "delta_parity": int(rec.get("delta_parity", 0)),
            "parity_law_ok": bool(rec.get("parity_law_ok")),
        }
        flips_out["d3"] = flip
        break

    return flips_out



# ----------------------------------------------------------------------
# C3 v0.1 — Pass B2: manifest row lookup (no IO writes, no solver)
# ----------------------------------------------------------------------
def _time_tau_c3_find_manifest_row_for_fixture(fixture_label: str) -> dict | None:
    """Locate the manifest row for a given fixture_label.

    We reuse the same manifest file and location logic as the C2 runner:
      logs/manifests/manifest_full_scope.jsonl

    Returns the matching manifest row (as a dict) or None if not found or
    if the manifest file is missing/unreadable.
    """
    # Resolve manifests directory.
    root = _repo_root()
    manifests_dir = None
    try:
        manifests_dir = _MANIFESTS_DIR  # type: ignore[name-defined]
    except Exception:
        manifests_dir = None

    if not manifests_dir:
        manifests_dir = root / "logs" / "manifests"

    manifest_path = manifests_dir / "manifest_full_scope.jsonl"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    # Skip malformed lines; manifest is append-only.
                    continue

                # Prefer explicit fixture_label field; fall back to legacy "id".
                if rec.get("fixture_label") == fixture_label or rec.get("id") == fixture_label:
                    return rec
    except Exception:
        # Any IO failure is treated as "not found" in v0.1.
        return None

    return None



# ----------------------------------------------------------------------
# C3 v0.1 — Pass B3: build derived-world paths for a single flip
# ----------------------------------------------------------------------
def _time_tau_c3_build_derived_paths_for_flip(
    rec: dict,
    flip: dict,
    strict_sig8: str,
) -> tuple[bool, str, dict]:
    """
    Given a manifest row and a single flip description, write a mutated H or B (d3)
    file under DIRS["c3_worlds"] and return the input paths for the derived world.

    Returns (ok, msg, paths) where paths has keys {"B","C","H","U"} and all
    values are string paths suitable for feeding back into the v2 core helpers.

    On any IO/shape error, returns (False, C3_NA["MUTATION_WRITE_ERROR"], {}).
    """
    # Resolve original fixture inputs from the manifest row.
    paths = rec.get("paths") or {}
    B_src = paths.get("B") or rec.get("B") or ""
    C_src = paths.get("C") or rec.get("C") or ""
    H_src = paths.get("H") or rec.get("H") or ""
    U_src = paths.get("U") or rec.get("U") or ""

    if not (B_src and C_src and H_src and U_src):
        return False, C3_NA["MUTATION_WRITE_ERROR"], {}

    # Where to write derived worlds.
    root = _repo_root()
    worlds_dir = root / DIRS.get("c3_worlds", "app/inputs/c3_derived_worlds")
    try:
        worlds_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False, C3_NA["MUTATION_WRITE_ERROR"], {}

    kind = flip.get("kind")
    fixture_label = rec.get("fixture_label") or rec.get("id") or "UNKNOWN"

    try:
        if kind == "H2":
            # Load and mutate H2 block.
            jH, pH, _src_tag = abx_read_json_any(H_src, kind="H")  # type: ignore[arg-type]
            blocks_H = _svr_as_blocks_v2(jH, "H")  # {"1","2","3"}
            H2 = blocks_H.get("2")
            i = int(flip.get("i", 0))
            j = int(flip.get("j", 0))
            if H2 is None or i < 0 or j < 0 or i >= len(H2) or j >= len(H2[i]):
                return False, C3_NA["MUTATION_WRITE_ERROR"], {}

            # Toggle the bit in GF(2).
            val = H2[i][j]
            try:
                bit = int(val) & 1
            except Exception:
                bit = 0
            H2[i][j] = 1 - bit
            blocks_H["2"] = H2

            # Preserve any extra metadata if present.
            if isinstance(jH, dict):
                jH_mut = dict(jH)
                jH_mut["blocks"] = blocks_H
            else:
                jH_mut = {"blocks": blocks_H}

            outp = worlds_dir / f"{fixture_label}__H2_{i}_{j}__{strict_sig8}.json"
            _hard_co_write_json(outp, jH_mut)
            paths_out = {
                "B": str(B_src),
                "C": str(C_src),
                "H": str(outp),
                "U": str(U_src),
            }
            return True, "OK", paths_out

        elif kind == "d3":
            # Load and mutate d3 block inside B.
            jB, pB, _src_tag = abx_read_json_any(B_src, kind="boundaries")  # type: ignore[arg-type]
            blocks_B = _svr_as_blocks_v2(jB, "B")
            d3 = blocks_B.get("3")
            j_idx = int(flip.get("j", 0))
            k_idx = int(flip.get("k", 0))
            if d3 is None or j_idx < 0 or k_idx < 0 or j_idx >= len(d3) or k_idx >= len(d3[j_idx]):
                return False, C3_NA["MUTATION_WRITE_ERROR"], {}

            val = d3[j_idx][k_idx]
            try:
                bit = int(val) & 1
            except Exception:
                bit = 0
            d3[j_idx][k_idx] = 1 - bit
            blocks_B["3"] = d3

            if isinstance(jB, dict):
                jB_mut = dict(jB)
                jB_mut["blocks"] = blocks_B
            else:
                jB_mut = {"blocks": blocks_B}

            outp = worlds_dir / f"{fixture_label}__d3_{j_idx}_{k_idx}__{strict_sig8}.json"
            _hard_co_write_json(outp, jB_mut)
            paths_out = {
                "B": str(outp),
                "C": str(C_src),
                "H": str(H_src),
                "U": str(U_src),
            }
            return True, "OK", paths_out

        else:
            # Unknown flip kind.
            return False, C3_NA["MUTATION_WRITE_ERROR"], {}
    except Exception:
        # Any failure in IO or mutation is treated as a write error.
        return False, C3_NA["MUTATION_WRITE_ERROR"], {}



# ----------------------------------------------------------------------
# C3 v0.1 — Pass B4: solver observation via strict core recompute
# ----------------------------------------------------------------------
def _time_tau_c3_run_solver_observation(
    base_ctx: dict,
    flip: dict,
    derived_paths: dict,
) -> tuple[bool, str, dict]:
    """Recompute strict-core parity on the derived world.

    This is a pure Time(τ) strict-core recomputation:
      - load B/C/H for the derived world
      - build the strict core using time_tau_strict_core_from_blocks
      - compute defect set D1 and parity_after
      - compute a strict_sig8 for the derived world, using the same recipe as
        the τ-toy C1/C2 panel (hash of (d3, C3, H2)).

    Baseline quantities (parity_before, strict_sig8_before, snapshot_before)
    are taken from the τ-artifact / base_ctx; we do not re-run the baseline
    world in v0.1.

    Returns (ok, msg, solver_obs). On any error, returns
    (False, C3_NA["SOLVER_ERROR"], {}).
    """
    try:
        tau_artifact = base_ctx.get("tau_artifact") or {}
        base_block = tau_artifact.get("base") or {}

        try:
            base_parity_before = int(base_block.get("parity", 0))
        except Exception:
            base_parity_before = 0

        strict_sig8_before = str(base_ctx.get("strict_sig8") or "")
        snapshot_before = base_ctx.get("snapshot_id")

        B_path = derived_paths.get("B")
        C_path = derived_paths.get("C")
        H_path = derived_paths.get("H")

        if not (B_path and C_path and H_path):
            return False, C3_NA["SOLVER_ERROR"], {}

        # Load derived world blocks.
        jB, pB, _tagB = abx_read_json_any(B_path, kind="boundaries")  # type: ignore[arg-type]
        jC, pC, _tagC = abx_read_json_any(C_path, kind="C")           # type: ignore[arg-type]
        jH, pH, _tagH = abx_read_json_any(H_path, kind="H")           # type: ignore[arg-type]

        blocks_B = _svr_as_blocks_v2(jB, "B")
        blocks_C = _svr_as_blocks_v2(jC, "C")
        blocks_H = _svr_as_blocks_v2(jH, "H")

        core1 = time_tau_strict_core_from_blocks(blocks_B, blocks_C, blocks_H)
        R3_1 = core1["R3"]
        D1 = time_tau_defect_set_from_R3(R3_1)
        base_parity_after = len(D1) % 2

        core_repr1 = {
            "d3": core1.get("d3"),
            "C3": core1.get("C3"),
            "H2": core1.get("H2"),
        }
        strict_sig8_after = _hashlib.sha256(
            _json.dumps(core_repr1, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:8]

        solver_obs = {
            "base_parity_before": int(base_parity_before),
            "base_parity_after": int(base_parity_after),
            "delta_parity": int(base_parity_after) ^ int(base_parity_before),
            "strict_sig8_before": strict_sig8_before,
            "strict_sig8_after": strict_sig8_after,
            "verdict_before": bool(base_parity_before == 0),
            "verdict_after": bool(base_parity_after == 0),
            "snapshot_before": snapshot_before,
            "snapshot_after": snapshot_before,
        }
        return True, "OK", solver_obs
    except Exception:
        return False, C3_NA["SOLVER_ERROR"], {}



# ----------------------------------------------------------------------
# C3 v0.1 — Pass B5: build and write C3 receipt
# ----------------------------------------------------------------------
def _time_tau_c3_build_and_write_receipt(
    base_ctx: dict,
    flip_ref: dict,
    derived_world: dict | None,
    tau_pred: dict | None,
    solver_observation: dict | None,
    c3_pass: bool | None,
    na_reason: str | None,
) -> dict:
    """Assemble the C3 v0.2 receipt dict and write it to logs/experiments.

    This helper is intentionally boring:
      - it does no solver work,
      - it only normalizes the payload and writes JSON.

    It expects that S0–S4 helpers have already decided c3_pass/NA.
    """
    fixture_label = (
        base_ctx.get("fixture_label")
        or flip_ref.get("fixture_label")
        or "UNKNOWN"
    )
    district_id = str(base_ctx.get("district_id") or "DUNKNOWN")
    strict_sig8 = str(base_ctx.get("strict_sig8") or "")
    snapshot_id = base_ctx.get("snapshot_id")

    # Flip indices, cast to ints where present.
    kind = flip_ref.get("kind") or "UNKNOWN"
    i_val = flip_ref.get("i", None)
    j_val = flip_ref.get("j", None)
    k_val = flip_ref.get("k", None)
    try:
        i_idx = int(i_val) if i_val is not None else None
    except Exception:
        i_idx = None
    try:
        j_idx = int(j_val) if j_val is not None else None
    except Exception:
        j_idx = None
    try:
        k_idx = int(k_val) if k_val is not None else None
    except Exception:
        k_idx = None

    # τ-prediction block: prefer explicit tau_pred, fall back to base_ctx.
    if not isinstance(tau_pred, dict):
        tau_pred = base_ctx.get("tau_toy_prediction") or {}
    try:
        expected_tau_law_holds = bool(tau_pred.get("expected_tau_law_holds", True))
    except Exception:
        expected_tau_law_holds = True

    tau_toy_prediction = {
        "schema_version": TIME_TAU_C3_PRED_SCHEMA_VERSION,
        "expected_tau_law_holds": expected_tau_law_holds,
    }

    # Derived world block: only the mutated artifact identity is required here.
    if isinstance(derived_world, dict):
        derived_artifact_path = derived_world.get("derived_artifact_path")
        mutated_matrix_sig8 = derived_world.get("mutated_matrix_sig8")
    else:
        derived_artifact_path = None
        mutated_matrix_sig8 = None

    derived_world_block: dict = {
        "derived_artifact_path": derived_artifact_path,
    }
    if mutated_matrix_sig8 is not None:
        derived_world_block["mutated_matrix_sig8"] = mutated_matrix_sig8

    # Solver observation block: normalized obs dict or None.
    solver_obs_block = solver_observation if isinstance(solver_observation, dict) else None

    # Normalize NA / c3_pass invariants.
    final_na_reason: str | None = na_reason
    if final_na_reason is not None and final_na_reason not in C3_NA.values():
        # Any non-canonical NA is treated as a generic solver error.
        final_na_reason = _time_tau_c3_na("SOLVER_ERROR")

    if final_na_reason is not None:
        # NA dominates; c3_pass is considered undefined.
        c3_pass_out: bool | None = None
    else:
        final_na_reason = None
        c3_pass_out = bool(c3_pass) if c3_pass is not None else None

    receipt = {
        "schema_version": TIME_TAU_C3_RECEIPT_SCHEMA_VERSION,
        "base": {
            "district_id": district_id,
            "fixture_label": fixture_label,
            "baseline_strict_sig8": strict_sig8,
            "baseline_snapshot_id": snapshot_id,
        },
        "flip_ref": {
            "kind": kind,
            "i": i_idx,
            "j": j_idx,
            "k": k_idx,
        },
        "derived_world": derived_world_block,
        "tau_toy_prediction": tau_toy_prediction,
        "solver_observation": solver_obs_block,
        "c3_pass": c3_pass_out,
        "c3_na_reason": final_na_reason,
    }

    # Compute deterministic filename.
    root = _repo_root()
    exp_dir = root / "logs" / "experiments"
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot ensure the directory, still return the in-memory receipt.
        return receipt

    # Use a,b indices according to flip kind for human-readable naming.
    if kind == "H2":
        a_idx = i_idx if i_idx is not None else -1
        b_idx = j_idx if j_idx is not None else -1
    elif kind == "d3":
        a_idx = j_idx if j_idx is not None else -1
        b_idx = k_idx if k_idx is not None else -1
    else:
        a_idx = i_idx if i_idx is not None else -1
        b_idx = j_idx if j_idx is not None else -1

    base_name = f"time_tau_c3_recompute__{fixture_label}__{strict_sig8}__{kind}_{a_idx}_{b_idx}"
    outp = exp_dir / f"{base_name}.json"
    try:
        _hard_co_write_json(outp, receipt)
    except Exception:
        # Write failures should not crash the caller; the dict is still returned.
        pass

    return receipt



    # Use a,b indices according to flip kind for human-readable naming.
    if kind == "H2":
        a_idx = i_idx if i_idx is not None else -1
        b_idx = j_idx if j_idx is not None else -1
    elif kind == "d3":
        a_idx = j_idx if j_idx is not None else -1
        b_idx = k_idx if k_idx is not None else -1
    else:
        a_idx = -1
        b_idx = -1

    base_name = f"time_tau_c3_recompute__{fixture_label}__{strict_sig8}__{kind}_{a_idx}_{b_idx}"
    outp = exp_dir / f"{base_name}.json"
    try:
        _hard_co_write_json(outp, receipt)
    except Exception:
        # Write failures should not crash the caller; the dict is still returned.
        pass

    return receipt



# ----------------------------------------------------------------------
# C3 v0.1 — Pass C: single-fixture driver (max one H2 + one d3 flip)
# ----------------------------------------------------------------------
def time_tau_c3_run_for_fixture(
    fixture_label: str,
    strict_sig8: str,
) -> list[dict]:
    """C3 v0.2 driver for a single fixture+sig8.

    For a given (fixture_label, strict_sig8), this will:

      - S0: load a baseline context from SSOT / τ artifacts
             via _time_tau_c3_load_baseline_context_from_fixture.
      - S1: build a deterministic list of law-OK flips using
             _time_tau_c3_iter_flips_for_fixture.
      - S2: for each flip, materialize a derived world descriptor and
             mutated-matrix artifact via _time_tau_c3_build_derived_world.
      - S3: evaluate the derived world in τ-mode using time_tau_c3_eval_world
             and normalize the result via _time_tau_c3_normalize_observation.
      - S4: compare the normalized observation against the τ-toy prediction
             using _time_tau_c3_judge and write a C3 receipt.

    No certs, coverage events, or manifest entries are written.

    Returns a list of receipt dicts (one per processed flip).
    """
    receipts: list[dict] = []

    # S0 — Baseline context gate.
    ok_ctx, na_ctx, base_ctx = _time_tau_c3_load_baseline_context_from_fixture(
        fixture_label=fixture_label,
        strict_sig8=strict_sig8,
    )
    if not ok_ctx or not isinstance(base_ctx, dict):
        # No τ-artifact or baseline cert → no C3 run.
        return receipts

    tau_pred = base_ctx.get("tau_toy_prediction") or {
        "schema_version": TIME_TAU_C3_PRED_SCHEMA_VERSION,
        "expected_tau_law_holds": True,
    }

    # S1 — Flip regime.
    flip_refs = _time_tau_c3_iter_flips_for_fixture(base_ctx)
    if not flip_refs:
        # No law-OK flips under the toy's constraints.
        return receipts

    for flip_ref in flip_refs:
        # S2 — Derived world materialization.
        ok_world, na_world, derived_world = _time_tau_c3_build_derived_world(
            base_ctx=base_ctx,
            flip_ref=flip_ref,
        )
        if not ok_world or not isinstance(derived_world, dict):
            na_code = na_world if na_world in C3_NA.values() else _time_tau_c3_na("MUTATION_WRITE_ERROR")
            receipt = _time_tau_c3_build_and_write_receipt(
                base_ctx=base_ctx,
                flip_ref=flip_ref,
                derived_world=None,
                tau_pred=tau_pred,
                solver_observation=None,
                c3_pass=None,
                na_reason=na_code,
            )
            receipts.append(receipt)
            continue

        # S3 — Evaluate world in τ-mode and normalize observation.
        try:
            core_obs = time_tau_c3_eval_world(derived_world)
        except Exception:
            na_code = _time_tau_c3_na("SOLVER_ERROR")
            receipt = _time_tau_c3_build_and_write_receipt(
                base_ctx=base_ctx,
                flip_ref=flip_ref,
                derived_world=derived_world,
                tau_pred=tau_pred,
                solver_observation=None,
                c3_pass=None,
                na_reason=na_code,
            )
            receipts.append(receipt)
            continue

        try:
            solver_observation = _time_tau_c3_normalize_observation(core_obs)
        except Exception:
            na_code = _time_tau_c3_na("OBSERVATION_PARSE_ERROR")
            receipt = _time_tau_c3_build_and_write_receipt(
                base_ctx=base_ctx,
                flip_ref=flip_ref,
                derived_world=derived_world,
                tau_pred=tau_pred,
                solver_observation=None,
                c3_pass=None,
                na_reason=na_code,
            )
            receipts.append(receipt)
            continue

        # S4 — Verdict.
        ok_judge, na_judge, c3_pass = _time_tau_c3_judge(tau_pred, solver_observation)
        if not ok_judge:
            na_code = na_judge or _time_tau_c3_na("OBSERVATION_PARSE_ERROR")
            receipt = _time_tau_c3_build_and_write_receipt(
                base_ctx=base_ctx,
                flip_ref=flip_ref,
                derived_world=derived_world,
                tau_pred=tau_pred,
                solver_observation=solver_observation,
                c3_pass=None,
                na_reason=na_code,
            )
        else:
            receipt = _time_tau_c3_build_and_write_receipt(
                base_ctx=base_ctx,
                flip_ref=flip_ref,
                derived_world=derived_world,
                tau_pred=tau_pred,
                solver_observation=solver_observation,
                c3_pass=c3_pass,
                na_reason=None,
            )
        receipts.append(receipt)

    return receipts





# ── UI: C3 recompute check (single fixture, v0.1) ──
try:
    nonce_c3 = st.session_state.get("_ui_nonce", "tau_c3")  # type: ignore[name-defined]
except Exception:
    nonce_c3 = "tau_c3"


# Time(τ) C3 sweep manifest + sweep runner (v0.2 — Pass A/B)
# ----------------------------------------------------------------------


def time_tau_c3_build_manifest_from_c2_sweep(
    manifest_v2_path: str | None = None,
    c2_sweep_jsonl_path: str | None = None,
) -> tuple[bool, str, dict]:
    """Build the Time(τ) C3 sweep manifest from the v2 manifest and C2 sweep.

    This is a wiring helper only. It reads:

      - logs/manifests/manifest_full_scope.jsonl  (v2 manifest)
      - logs/experiments/time_tau_local_flip_sweep*.jsonl  (C2 sweep)

    and writes:

      - logs/manifests/time_tau_c3_manifest_full_scope.jsonl

    The output manifest has one row per fixture_label that appears in both the
    v2 manifest and the C2 sweep. Each row carries:

      - fixture_label
      - district_id
      - snapshot_id
      - strict_sig8
      - paths.{B,C,H,U} (from the v2 manifest)
      - tau_na_reason
      - global_tau_law_ok

    Returns:
        (ok, msg, summary_dict)
    """
    # Resolve canonical directories.
    try:
        manifests_dir = _Path("logs") / "manifests"
        exps_dir = _Path("logs") / "experiments"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        exps_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        manifests_dir = _Path("logs") / "manifests"
        exps_dir = _Path("logs") / "experiments"

    # v2 manifest (strict/proj suite fixtures).
    manifest_v2 = _Path(manifest_v2_path) if manifest_v2_path else (manifests_dir / "manifest_full_scope.jsonl")

    # C2 sweep JSONL (Time(τ) local flip toy).
    if c2_sweep_jsonl_path:
        c2_sweep = _Path(c2_sweep_jsonl_path)
    else:
        # Pick the lexicographically last sweep JSONL as a simple proxy for "latest".
        candidates = sorted(exps_dir.glob("time_tau_local_flip_sweep*.jsonl"))
        if not candidates:
            return False, "No C2 sweep JSONL found under logs/experiments/.", {}
        c2_sweep = candidates[-1]

    if not manifest_v2.exists():
        return False, f"v2 manifest not found at {manifest_v2}", {}
    if not c2_sweep.exists():
        return False, f"C2 sweep JSONL not found at {c2_sweep}", {}

    # Load v2 manifest rows, keyed by fixture_label.
    baseline: dict[str, dict] = {}
    n_manifest_rows = 0

    with manifest_v2.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            fid = rec.get("fixture_label") or ""
            if not fid:
                continue
            baseline[fid] = rec
            n_manifest_rows += 1

    # Load C2 sweep rows, keyed by fixture_label.
    c2_rows: dict[str, dict] = {}
    n_c2_rows = 0
    with c2_sweep.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            fid = rec.get("fixture_label") or ""
            if not fid:
                continue
            c2_rows[fid] = rec
            n_c2_rows += 1

    # Join on fixture_label and assemble C3 manifest rows.
    out_rows: list[dict] = []
    for fid in sorted(c2_rows.keys()):
        base = baseline.get(fid)
        if not base:
            # Skip fixtures that are not present in the v2 manifest.
            continue

        sweep_row = c2_rows[fid]
        paths = (base.get("paths") or {}).copy()

        fixture_label = fid
        district_id = sweep_row.get("district_id") or (fixture_label.split("_")[0] if fixture_label else "DUNKNOWN")
        snapshot_id = sweep_row.get("snapshot_id")
        strict_sig8 = sweep_row.get("strict_sig8") or ""
        tau_na_reason = sweep_row.get("tau_na_reason")
        global_tau_law_ok = bool(sweep_row.get("global_tau_law_ok", False))

        out_rows.append(
            {
                "fixture_label": fixture_label,
                "district_id": district_id,
                "snapshot_id": snapshot_id,
                "strict_sig8": strict_sig8,
                "paths": paths,
                "tau_na_reason": tau_na_reason,
                "global_tau_law_ok": global_tau_law_ok,
            }
        )

    manifest_c3 = manifests_dir / "time_tau_c3_manifest_full_scope.jsonl"
    tmp = manifest_c3.with_suffix(".jsonl.tmp")
    text = "".join(_json.dumps(r, separators=(",", ":")) + "\n" for r in out_rows)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(manifest_c3)

    summary = {
        "manifest_v2_path": str(manifest_v2),
        "c2_sweep_jsonl_path": str(c2_sweep),
        "manifest_c3_path": str(manifest_c3),
        "n_manifest_v2_rows": n_manifest_rows,
        "n_c2_rows": n_c2_rows,
        "n_c3_rows": len(out_rows),
    }
    msg = (
        f"C3 manifest built: {len(out_rows)} rows "
        f"(v2={n_manifest_rows}, c2_sweep={n_c2_rows})."
    )
    return True, msg, summary


def time_tau_c3_run_sweep(manifest_path: str | None = None) -> tuple[bool, str, dict]:
    """Run the Time(τ) C3 recompute check across all fixtures in the C3 manifest.

    This is the C3 sweep driver (v0.2). It:

      - reads logs/manifests/time_tau_c3_manifest_full_scope.jsonl,
      - for each row, calls time_tau_c3_run_for_fixture(fixture_label, strict_sig8),
      - accumulates basic counts, and
      - returns (ok, msg, summary_dict).

    It does not change C3 semantics; it only scales the existing single-fixture
    driver across the manifest.
    """
    manifests_dir = _Path("logs") / "manifests"
    manifest = _Path(manifest_path) if manifest_path else (manifests_dir / "time_tau_c3_manifest_full_scope.jsonl")

    if not manifest.exists():
        return False, f"C3 sweep manifest not found at {manifest}", {}

    n_rows = 0
    n_bad_rows = 0
    n_fixtures = 0
    n_fixtures_ok = 0
    n_fixtures_na = 0
    n_receipts_total = 0

    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                rec = _json.loads(line)
            except Exception:
                n_bad_rows += 1
                n_fixtures_na += 1
                continue
            if not isinstance(rec, dict):
                n_bad_rows += 1
                n_fixtures_na += 1
                continue

            fixture_label = rec.get("fixture_label") or ""
            strict_sig8 = rec.get("strict_sig8") or ""

            if not fixture_label or not strict_sig8:
                n_fixtures_na += 1
                continue

            n_fixtures += 1
            try:
                receipts = time_tau_c3_run_for_fixture(
                    fixture_label=fixture_label,
                    strict_sig8=strict_sig8,
                )
            except Exception:
                # Treat hard errors as NA for sweep accounting; do not crash the sweep.
                n_fixtures_na += 1
                continue

            if not receipts:
                # No receipts emitted (e.g. no law-OK flips or missing baseline context).
                n_fixtures_na += 1
                continue

            n_fixtures_ok += 1
            n_receipts_total += len(receipts)

    summary = {
        "manifest_path": str(manifest),
        "n_manifest_rows": n_rows,
        "n_bad_manifest_rows": n_bad_rows,
        "n_fixtures": n_fixtures,
        "n_fixtures_ok": n_fixtures_ok,
        "n_fixtures_na": n_fixtures_na,
        "n_receipts_total": n_receipts_total,
    }
    msg = (
        f"C3 sweep completed: {n_fixtures} fixtures, "
        f"ok={n_fixtures_ok}, na={n_fixtures_na}, "
        f"receipts={n_receipts_total} "
        f"(manifest_rows={n_rows}, bad_rows={n_bad_rows})."
    )
    return True, msg, summary
with st.expander("C3 — Time(τ) recompute check (v0.2)", expanded=False):  # type: ignore[name-defined]
    st.caption(  # type: ignore[name-defined]
        "Run the C3 recompute check for a single fixture. "
        "Requires a τ-toy artifact from the Lab panel or C2 sweep."
    )

    col1, col2 = st.columns(2)  # type: ignore[name-defined]
    with col1:
        fixture_label_c3 = st.text_input(
            "Fixture label",
            key=f"time_tau_c3_fixture_label_{nonce_c3}",
            placeholder="e.g. D2_H00_C000",
        )
    with col2:
        strict_sig8_c3 = st.text_input(
            "Baseline strict_sig8 (8-hex)",
            key=f"time_tau_c3_strict_sig8_{nonce_c3}",
            placeholder="from τ-toy artifact name",
        )

    run_btn_c3 = st.button(  # type: ignore[name-defined]
        "Run C3 for this fixture",
        key=f"time_tau_c3_run_{nonce_c3}",
    )

    if run_btn_c3:
        if not fixture_label_c3 or not strict_sig8_c3:
            st.warning("Please provide both fixture_label and strict_sig8.")  # type: ignore[name-defined]
        else:
            with st.spinner("Running C3 recompute (v0.1)..."):  # type: ignore[name-defined]
                try:
                    receipts = time_tau_c3_run_for_fixture(  # type: ignore[name-defined]
                        fixture_label_c3.strip(),
                        strict_sig8_c3.strip(),
                    )
                except Exception as e:  # pragma: no cover - UI-only
                    st.error(f"C3 runner error: {e}")  # type: ignore[name-defined]
                    receipts = []

            if not receipts:
                st.info(  # type: ignore[name-defined]
                    "No C3 receipts produced. "
                    "Check that a τ-toy artifact exists for this fixture+sig8, "
                    "and that the manifest contains this fixture."
                )
            else:
                for r in receipts:
                    flip_ref = (r.get("flip_ref") or {})  # type: ignore[assignment]
                    kind = flip_ref.get("kind")
                    i = flip_ref.get("i")
                    j = flip_ref.get("j")
                    k = flip_ref.get("k")
                    c3_pass = r.get("c3_pass")
                    na = r.get("c3_na_reason")

                    label = f"{kind} flip (i={i}, j={j}, k={k})"
                    st.write(f"**{label}** → `c3_pass={c3_pass}`, `na_reason={na}`")  # type: ignore[name-defined]
                    st.json(r)  # type: ignore[name-defined]
    st.markdown("---")  # type: ignore[name-defined]
    st.markdown("**C3 — τ-recompute sweep over manifest (v0.2)**")  # type: ignore[name-defined]
    st.caption(  # type: ignore[name-defined]
        "Run the C3 recompute check for all fixtures that have τ-toy artifacts from the C2 sweep. "
        "Uses the v2 manifest + C2 sweep to build a C3 manifest, then runs C3 per fixture."
    )

    if st.button(
        "Run C3 τ-recompute sweep over manifest (C3)",
        key=f"time_tau_c3_sweep_run_{nonce_c3}",
    ):  # type: ignore[name-defined]
        with st.spinner("Building C3 manifest from C2 sweep..."):  # type: ignore[name-defined]
            ok_m, msg_m, summary_m = time_tau_c3_build_manifest_from_c2_sweep()

        if not ok_m:
            st.warning(f"C3 manifest build failed: {msg_m}")  # type: ignore[name-defined]
        else:
            st.success(msg_m)  # type: ignore[name-defined]
            st.caption(  # type: ignore[name-defined]
                f"C3 manifest: {summary_m.get('manifest_c3_path')} · "
                f"v2 rows={summary_m.get('n_manifest_v2_rows')} · "
                f"C2 rows={summary_m.get('n_c2_rows')} · "
                f"C3 rows={summary_m.get('n_c3_rows')}"
            )

            with st.spinner("Running C3 sweep over manifest..."):  # type: ignore[name-defined]
                ok_s, msg_s, summary_s = time_tau_c3_run_sweep(
                    manifest_path=summary_m.get("manifest_c3_path"),
                )

            if ok_s:
                st.success(msg_s)  # type: ignore[name-defined]
                st.caption(  # type: ignore[name-defined]
                    f"fixtures={summary_s.get('n_fixtures')} · "
                    f"ok={summary_s.get('n_fixtures_ok')} · "
                    f"na={summary_s.get('n_fixtures_na')} · "
                    f"receipts={summary_s.get('n_receipts_total')}"
                )
            else:
                st.warning(f"C3 sweep failed: {msg_s}")  # type: ignore[name-defined]



# ── UI: C4 — C3 stability rollup (v0.2) ──  (v0.1=counts only, v0.2=+H2/d3+τ-law)
with st.expander("C4 — C3 stability rollup (v0.2)", expanded=False):  # type: ignore[name-defined]
    st.caption(  # type: ignore[name-defined]
        "Aggregate all C3 recompute receipts into a per-fixture rollup. "
        "This does not run the solver; it only summarizes existing artifacts."
    )

    if st.button("Build / refresh C3 rollup", key="time_tau_c4_build"):  # type: ignore[name-defined]
        with st.spinner("Building C3 rollup from receipts..."):  # type: ignore[name-defined]
            try:
                rows, summary = time_tau_c4_build_rollup()  # type: ignore[name-defined]
            except Exception as e:  # pragma: no cover - UI-only
                st.error(f"C4 rollup error: {e}")  # type: ignore[name-defined]
                rows, summary = [], {}

        if not rows:
            st.info(  # type: ignore[name-defined]
                "No C3 receipts found. Run C3 on some fixtures first."
            )
        else:
            st.success(  # type: ignore[name-defined]
                f"Built rollup for {len(rows)} fixture entries."
            )

            # τ local/global mismatch quick summary.
            try:
                mismatch_rows = _time_tau_c4_build_tau_mismatch_rows(rows)
                if mismatch_rows:
                    st.info(  # type: ignore[name-defined]
                        f"τ local/global mismatches: {len(mismatch_rows)} fixtures "
                        "(see logs/reports/time_tau_c3_tau_mismatches.*)"
                    )
            except Exception:
                # Best-effort UI sugar only.
                pass

            # Small district-level summary table.
            if summary:
                st.markdown("**District summary**")  # type: ignore[name-defined]
                st.json(summary)  # type: ignore[name-defined]

                # Friendly τ-law recap per district.
                try:
                    for d_key in sorted(summary.keys()):
                        s = summary[d_key] or {}
                        n_pred_true = s.get("n_fixtures_tau_pred_true", 0)
                        n_pred_false = s.get("n_fixtures_tau_pred_false", 0)
                        n_emp_true = s.get("n_fixtures_tau_emp_true", 0)
                        n_emp_false = s.get("n_fixtures_tau_emp_false", 0)
                        st.caption(  # type: ignore[name-defined]
                            f"{d_key} · fixtures_with_c3={s.get('n_fixtures_with_c3', 0)} · "
                            f"τ_pred true/false={n_pred_true}/{n_pred_false} · "
                            f"τ_emp true/false={n_emp_true}/{n_emp_false}"
                        )
                except Exception:
                    # Best-effort UI sugar only.
                    pass

            # Show a few rollup rows.
            st.markdown("**Sample of per-fixture rollup rows**")  # type: ignore[name-defined]
            for r in rows[:20]:
                st.write(  # type: ignore[name-defined]
                    f"{r['district_id']} · {r['fixture_label']} · {r['strict_sig8']} → "
                    f"flips={r['n_flips_total']}, pass={r['n_pass']}, "
                    f"fail={r['n_fail']}, na={r['n_na_total']}"
                )
                # Tiny sugar: per-kind counts and τ-law booleans.
                try:
                    n_pass_H2 = r.get("n_pass_H2", 0)
                    n_fail_H2 = r.get("n_fail_H2", 0)
                    n_na_H2 = r.get("n_na_H2", 0)
                    n_pass_d3 = r.get("n_pass_d3", 0)
                    n_fail_d3 = r.get("n_fail_d3", 0)
                    n_na_d3 = r.get("n_na_d3", 0)
                    tau_pred = r.get("expected_tau_law_holds")
                    tau_emp = r.get("empirical_tau_law_holds")

                    def _fmt_tau(b):
                        if b is True:
                            return "✓"
                        if b is False:
                            return "✗"
                        return "·"

                    st.caption(  # type: ignore[name-defined]
                        f"   H2 pass/fail/na={n_pass_H2}/{n_fail_H2}/{n_na_H2} · "
                        f"d3 pass/fail/na={n_pass_d3}/{n_fail_d3}/{n_na_d3} · "
                        f"τ_pred={_fmt_tau(tau_pred)} · τ_emp={_fmt_tau(tau_emp)}"
                    )
                except Exception:
                    # Best-effort UI sugar only.
                    pass


# --------------
