import streamlit as st
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
# --- Unified A/B embed signature (lane-aware, cert-aligned) -------------------
import json as _json
from pathlib import Path
import json, hashlib, streamlit as st
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
# --- baseline imports (defensive) ---
import os, json, time, uuid, shutil, tempfile, hashlib
from datetime import datetime, timezone
from pathlib import Path
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
from pathlib import Path as _Path
import json as _json, hashlib as _hashlib, time as _time
import streamlit as _st
# --- Canonical tiny helpers (early, guarded) ---
from typing import Iterable, List, Optional
# Page config must be the first Streamlit command
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

def _canon_dump_and_sig8(obj):
    """Return (canonical_json_text, first_8_of_sha256) for small cert payloads."""
    can = _v2_canonical_obj(obj)
    raw = _json.dumps(can, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = _hash.sha256(raw).hexdigest()
    return raw.decode("utf-8"), h[:8]

# Normalize solver return to (ok, msg, bundle_dir)
def _solver_ret_as_tuple(ret):
    try:
        if isinstance(ret, (tuple, list)):
            ok  = bool(ret[0]) if len(ret) >= 1 else False
            msg = str(ret[1]) if len(ret) >= 2 else ""
            bdir = ret[2] if len(ret) >= 3 else None
            return ok, msg, bdir
        if isinstance(ret, dict):
            ok  = bool(ret.get("ok", ret.get("success", True)))
            msg = str(ret.get("msg", ret.get("message", "")))
            bdir = ret.get("bundle_dir") or ret.get("bundle") or ret.get("dir")
            return ok, msg, bdir
        if isinstance(ret, bool):
            return ret, ("ok" if ret else "fail"), None
        if ret is None:
            return False, "solver returned None", None
    except Exception as e:
        return False, f"ret normalization error: {e}", None
    return False, "solver returned unexpected shape", None

# Pass1/2 helpers to write bundle.json and loop_receipt
from pathlib import Path as _VPath
import json as _Vjson, hashlib as _Vhash

_V2_EXPECTED = [
    ("strict",               "overlap__", "__strict__"),
    ("projected_auto",       "overlap__", "__projected_columns_k_3_auto__"),
    ("ab_auto",              "ab_compare__strict_vs_projected_auto__", ""),
    ("freezer",              "projector_freezer__", "",),
    ("ab_file",              "ab_compare__projected_columns_k_3_file__", ""),
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



def _v2_extract_ids_from_path(bdir: _VPath):
    try:
        sig8 = bdir.name
        district = bdir.parent.name
        if district.startswith("D") and len(sig8) == 8:
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
    bdir = _VPath(bdir)
    files = _v2_find_expected_files(bdir)
    hashes = {k: _v2_sha256_path(p) for k,p in files.items()}
    sizes  = {k: p.stat().st_size for k,p in files.items()}
    district, sig8 = _v2_extract_ids_from_path(bdir)
    lanes_pop, lanes_sig8 = (None, None)
    if "projected_auto" in files:
        lanes_pop, lanes_sig8 = _v2_extract_lanes_from_auto(files["projected_auto"])
    pres = _v2_presence_mask(files.keys())
    bundle = {
        "district_id": district,
        "sig8": sig8,
        "files": {k: str(p) for k,p in files.items()},
        "hashes": hashes,
        "sizes": sizes,
        "presence_mask_hex": pres,
        "counts": {"present": len(files)},
        "lanes": {"popcount": lanes_pop, "sig8": lanes_sig8},
    }
    (bdir / "bundle.json").write_text(_Vjson.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return bundle

def _v2_write_loop_receipt(bdir: _VPath, fixture_id: str, snapshot_id: str, bundle: dict):
    rec = {
        "schema_version": "2.0.0",
        "fixture_id": fixture_id,
        "snapshot_id": snapshot_id,
        "district_id": bundle.get("district_id"),
        "sig8": bundle.get("sig8"),
        "presence_mask_hex": bundle.get("presence_mask_hex"),
        "lanes": bundle.get("lanes"),
        "hashes": bundle.get("hashes"),
        "sizes": bundle.get("sizes"),
    }
    (_VPath(bdir) / f"loop_receipt__{fixture_id}.json").write_text(_Vjson.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    return rec




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

def _one_press_triple():
    g = globals()
    if "run_overlap_once" in g and callable(g["run_overlap_once"]):
        return _solver_ret_as_tuple(g["run_overlap_once"]())
    if "_svr_run_once" in g and callable(g["_svr_run_once"]):
        return _solver_ret_as_tuple(g["_svr_run_once"]())
    return False, "No solver entry found (run_overlap_once/_svr_run_once).", None
    
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
    





st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")


SCHEMA_VERSION = "2.0.0"
ENGINE_REV     = "rev-20251022-1"

DIRS = {"root": "logs", "certs": "logs/certs", "snapshots": "logs/snapshots", "reports": "logs/reports", "suite_runs": "logs/suite_runs", "exports": "logs/exports"}
# ---------- Suite helpers (v2) ----------

try:
    _Path
except NameError:
    from pathlib import Path as _Path

def _sha256_hex(b: bytes) -> str:
    import hashlib as _hashlib
    return _hashlib.sha256(b).hexdigest()

def _canonical_json(obj) -> str:
    import json as _json
    return _json.dumps(obj, sort_keys=True, separators=(",",":"))

def _repo_root() -> _Path:
    # repo root = parent of app dir
    return _Path(__file__).resolve().parent.parent



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

def ensure_suite_snapshot(manifest_path: str) -> tuple[bool, str, str | None]:
    """Create a content-addressed world snapshot from a manifest JSONL.
    Writes: logs/snapshots/world_snapshot__{sid}.json and world_snapshot.latest.json
    Returns (ok, message, snapshot_id).
    """
    try:
        mp = _abs_from_manifest(manifest_path)
        if not mp.exists():
            return False, f"Manifest not found: {mp}", None
        import json as _json, hashlib as _hashlib, datetime as _dt
        rows = []
        inventory = []
        seen = {}
        # walk manifest JSONL
        for line in mp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            # resolve absolute paths
            B = str(_abs_from_manifest(rec["B"]))
            C = str(_abs_from_manifest(rec["C"]))
            H = str(_abs_from_manifest(rec["H"]))
            U = str(_abs_from_manifest(rec["U"]))
            fid = rec.get("id") or ""
            rows.append({"B": B, "C": C, "H": H, "U": U, "id": fid})
            for tag, path in (("B",B),("C",C),("H",H),("U",U)):
                if path in seen:
                    continue
                try:
                    h = _sha256_hex(_Path(path).read_bytes())
                except Exception:
                    h = None
                seen[path] = {"tag": tag, "path": path, "sha256": h}
        inventory = list(seen.values())
        # canonical payload (no matrices)
        payload = {
            "schema_version": "2.0.0",
            "engine_rev": ENGINE_REV,
            "created_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "manifests": {"full_scope": str(mp)},
            "inventory": inventory,
            "plan_full_scope": rows,
        }
        sid = "ws__" + _sha256_hex(_canonical_json(payload).encode("utf-8"))[:8]
        payload["snapshot_id"] = sid
        # ensure dirs
        snaps_dir = _Path(DIRS.get("snapshots", "logs/snapshots"))
        snaps_dir.mkdir(parents=True, exist_ok=True)
        # write snapshot & pointer
        _guarded_atomic_write_json(snaps_dir / f"world_snapshot__{sid}.json", payload)
        _guarded_atomic_write_json(snaps_dir / "world_snapshot.latest.json", {"snapshot_id": sid, "path": f"world_snapshot__{sid}.json"})
        return True, f"Snapshot ready: {sid}", sid
    except Exception as e:
        return False, f"Snapshot build failed: {e}", None

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

def run_suite_from_manifest(manifest_path: str, snapshot_id: str):
    """
    Iterate JSONL manifest: seed inputs -> call run_overlap_once()
    -> record suite index rows.
    """
    manifest_abs = _abs_from_manifest(manifest_path)
    if not manifest_abs.exists():
        _st.error(f"Manifest not found: {manifest_abs}")
        return False, f"Manifest not found: {manifest_abs}"

    lines = []
    with manifest_abs.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(_json.loads(raw))
            except Exception as e:
                return False, f"Bad JSONL line: {raw[:120]}… ({e})"

    ok_count = 0
    for i, rec in enumerate(lines, 1):
        fid = rec.get("id") or f"fixture_{i:02d}"
        B = rec["B"]; C = rec["C"]; H = rec["H"]; U = rec["U"]
        # Seed inputs for the single-press path
        _set_inputs_for_run(B, C, H, U)

        # Optional: quick existence preflight for clearer errors
        miss = [p for p in (B, C, H, U) if not _abs_from_manifest(p).exists()]
        if miss:
            _st.warning(f"[{fid}] Missing files: {', '.join(miss)}")
            continue

        # One-press, arity-proof (always normalized)
        ok, msg, bundle_dir = _one_press_triple()


        _st.write(f"{fid} → {'ok' if ok else 'fail'} · {msg}")
        if ok:
            ok_count += 1

        # Append to suite index if we can read lanes info from the projected AUTO cert
        try:
            # Locate latest AUTO cert written by this press via the bundle.json index
            bdir = _Path(bundle_dir) if bundle_dir else None
            if not bdir or not bdir.exists():
                # fallback: last bundle under logs/certs (best-effort)
                bdir = max((_REPO_DIR / "logs" / "certs").glob("*"), key=lambda p: p.stat().st_mtime)
            bidx = _json.loads((bdir / "bundle.json").read_text("utf-8"))
            auto_path = bidx.get("files", {}).get("projected_auto")
            lanes_pop = None; lanes_sig8 = None
            if auto_path:
                payload = _json.loads(_Path(auto_path).read_text("utf-8"))
                pc = (payload.get("projection_context") or {})
                lanes_pop = pc.get("lanes_popcount")
                lanes_sig8 = pc.get("lanes_sig8")

            _suite_index_add_row({
                "fixture_id": fid,
                "snapshot_id": snapshot_id,
                "bundle_dir": str(bdir),
                "lanes_popcount": lanes_pop,
                "lanes_sig8": lanes_sig8,
            })
        except Exception:
            # Non-fatal; index is a convenience
            pass

    return True, f"Completed {ok_count}/{len(lines)} fixtures."



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

def _witness_pack(bottom_H2d3_bits=None, bottom_C3pI3_bits=None, lanes=None) -> dict:
    return {
        "bottom_H2d3": _bits_to_str(bottom_H2d3_bits),
        "bottom_C3pI3": _bits_to_str(bottom_C3pI3_bits),
        "lanes": (list(lanes) if lanes is not None else None),
    }





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







#-----------------------------------------------------

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





# ------------------------- Key Generators & Widget-Key Deduper -------------------------
def _mkkey(ns: str, name: str) -> str:
    """Deterministic widget key: '<ns>__<name>'."""
    return f"{ns}__{name}"







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



# ------------------------- Mask from d3 (truth mask) -------------------------
def _truth_mask_from_d3(d3: list[list[int]]) -> list[int]:
    if not d3 or not d3[0]:
        return []
    rows, cols = len(d3), len(d3[0])
    return [1 if any(int(d3[i][j]) & 1 for i in range(rows)) else 0 for j in range(cols)]




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










# ───────────────────────── Tabs — create once, early ─────────────────────────
if not all(n in globals() for n in ("tab1","tab2","tab3","tab4","tab5")):
    try:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Unit", "Overlap", "Triangle", "Towers", "Export"])
    except Exception as _e:
        st.error("Tab construction failed.")
        st.exception(_e)
        st.stop()

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





# --- ensure tabs exist even if earlier branches ran before creating them
if "tab1" not in globals():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Unit", "Overlap", "Triangle", "Towers", "Export"])

# ------------------------------ UNIT TAB -------------------------------------
with tab1:
    st.subheader("Unit gate")

    # Optional boundaries override in Unit tab
    f_B = st.file_uploader("Boundaries (boundaries*.json)", type=["json"], key="B_up")
    _stamp_filename("fname_boundaries", f_B)
    d_B = read_json_file(f_B) if f_B else None
    if d_B:
        boundaries = io.parse_boundaries(d_B)

        # Re-bind district from Unit override (prefer raw bytes hash)
        try:
            if hasattr(f_B, "getvalue"):
                boundaries_hash_fresh = _sha256_hex_bytes(f_B.getvalue())
            else:
                boundaries_hash_fresh = _sha256_hex_obj(d_B)
        except Exception:
            boundaries_hash_fresh = _sha256_hex_obj(d_B)

        d3_block         = (boundaries.blocks.__root__.get("3") or [])
        lane_mask_k3_now = _lane_mask_from_d3(boundaries)
        d3_rows          = len(d3_block)
        d3_cols          = (len(d3_block[0]) if d3_block else 0)
        district_sig     = _district_signature(lane_mask_k3_now, d3_rows, d3_cols)
        district_id_fresh = DISTRICT_MAP.get(boundaries_hash_fresh, "UNKNOWN")

        # Clear stale session bits if boundaries changed
        _prev_bhash = st.session_state.get("_last_boundaries_hash")
        if _prev_bhash and _prev_bhash != boundaries_hash_fresh:
            for k in ("ab_compare", "district_id", "_projector_cache"):
                st.session_state.pop(k, None)
        st.session_state["_last_boundaries_hash"] = boundaries_hash_fresh

        # Update SSOT
        st.session_state["_inputs_block"]["boundaries_filename"] = st.session_state.get("fname_boundaries", "boundaries.json")
        st.session_state["_inputs_block"]["boundaries_hash"]     = boundaries_hash_fresh
        st.session_state["_district_info"] = {
            "district_id":        district_id_fresh,
            "boundaries_hash":    boundaries_hash_fresh,
            "lane_mask_k3_now":   lane_mask_k3_now,
            "district_signature": district_sig,
            "d3_rows": d3_rows,
            "d3_cols": d3_cols,
        }
        st.caption(f"[Unit override] district={district_id_fresh} · bhash={boundaries_hash_fresh[:12]} · k3={lane_mask_k3_now} · sig={district_sig}")

    # Optional C-map override
    f_C = st.file_uploader("C map (optional)", type=["json"], key="C_up")
    _stamp_filename("fname_cmap", f_C)
    d_C = read_json_file(f_C) if f_C else None
    if d_C:
        cmap = io.parse_cmap(d_C)

    # Optional Shapes/U override
    f_U = st.file_uploader("Shapes / carrier U (optional)", type=["json"], key="U_up")
    _stamp_filename("fname_shapes", f_U)
    d_U = read_json_file(f_U) if f_U else None
    if d_U:
        shapes = io.parse_shapes(d_U)
        st.session_state["_inputs_block"]["U_filename"] = st.session_state.get("fname_shapes", "shapes.json")

    # Reps (if used)
    f_reps = st.file_uploader("Reps (optional)", type=["json"], key="reps_up")
    _stamp_filename("fname_reps", f_reps)
    d_reps = read_json_file(f_reps) if f_reps else None

    enforce = st.checkbox("Enforce rep transport (c_cod = C c_dom)", value=False)
    if st.button("Run Unit"):
        try:
            out = unit_gate.unit_check(boundaries, cmap, shapes, reps=d_reps, enforce_rep_transport=enforce)
            st.json(out)
        except Exception as e:
            st.error(f"Unit gate failed: {e}")

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




with tab2:
    st.subheader("Overlap")


    def _bottom_row(M):
        return M[-1] if (M and len(M)) else []

    

 




  




    


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

if "_svr_atomic_write_json" not in globals():
    def _guarded_atomic_write_json(path: Path, payload: dict):
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


                    


    #---------------------------------------------------------------------------------------------------------------------------------

        
def tail_and_download_ui():
    import os, json, zipfile
    from pathlib import Path
    ss = st.session_state
    last_dir = ss.get("last_bundle_dir", "")
    st.markdown("#### Latest cert files")
    if not last_dir or not os.path.isdir(last_dir):
        st.info("No bundle.json yet — run the solver to write certs.")
        return

    p_bundle = Path(last_dir) / "bundle.json"
    files = []
    try:
        files = (json.loads(p_bundle.read_text(encoding="utf-8")).get("filenames", []) if p_bundle.exists() else [])
    except Exception:
        files = []

    if files:
        tail = files[-6:][::-1]
        for fn in tail:
            st.write(f"• {fn}")
            try:
                fp = Path(last_dir) / fn
                if fp.suffix.lower() == ".json":
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    # compact peek like your earlier UI
                    if isinstance(data, dict) and "ab_pair" in data:
                        st.caption(f"pair_vec: {data['ab_pair'].get('pair_vec', {})}")
                    elif isinstance(data, dict) and "results" in data:
                        out = data["results"].get("out", {})
                        na = data["results"].get("na_reason_code") or data.get("na_reason_code")
                        if isinstance(out, dict) and "3" in out and "eq" in out["3"]:
                            st.caption(f"k3.eq: {out['3']['eq']} • NA: {na}")
                    elif isinstance(data, dict) and "status" in data:
                        st.caption(f"freezer: {data.get('status')} • {data.get('na_reason_code','')}")
            except Exception:
                pass
    else:
        st.info("No files listed in bundle.json.")

    # Build & serve bundle.zip (keys deduped via nonce)
    try:
        zpath = Path(last_dir) / "bundle.zip"
        with zipfile.ZipFile(str(zpath), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, fns in os.walk(last_dir):
                for fn in fns:
                    if fn.endswith(".zip"):  # avoid nesting
                        continue
                    full = Path(root)/fn
                    arc  = os.path.relpath(str(full), start=str(last_dir))
                    zf.write(str(full), arc)
        with open(zpath, "rb") as fh:
            st.download_button(
                "Download bundle.zip",
                data=fh,
                file_name="bundle.zip",
                mime="application/zip",
                key=f"btn_dl_bundle_zip__{ss['_ui_nonce']}"
            )
    except Exception as e:
        st.warning(f"Zip build/serve issue: {e}")

# call this immediately under your solver section:
tail_and_download_ui()












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

    # --- Compute inputs_sig_5 if missing (fallback) ---
    inputs_sig_5 = ib.get("inputs_sig_5")
    try:
        n3 = len((bC.get("3") or [[]])[0]) if (bC.get("3") and (bC.get("3")[0])) else 0
    except Exception:
        n3 = 0
    if not inputs_sig_5:
        # 5 small, deterministic pieces: sha(B), sha(C), sha(H), sha(U), sha(shapes)
        try:
            sB = _v2_sha256_path(pB)
            sC = _v2_sha256_path(pC)
            sH = _v2_sha256_path(pH)
            sU = _v2_sha256_path(pU)
            shapes_blob = _json.dumps({"n3": n3}, sort_keys=True).encode("utf-8")
            sSh = _hash.sha256(shapes_blob).hexdigest()
            inputs_sig_5 = [sB, sC, sH, sU, sSh]
        except Exception:
            inputs_sig_5 = []

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
    ib_embed = {
        "district_id": district_id,
        "fixture_id": fixture_id,
        "snapshot_id": snapshot_id,
        "inputs_sig_5": list(inputs_sig_5) if isinstance(inputs_sig_5, (list, tuple)) else [],
    }
    embed_auto, embed_sig_auto = _svr_build_embed(
        ib_embed,
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
        "fixture_id": fixture_id,
        "snapshot_id": snapshot_id,
        "inputs_sig_5": inputs_sig_5,
    }

    written = []

    # 1) strict
    strict_payload = dict(base_hdr)
    strict_payload.update({
        "policy": "strict",
        "verdict": strict_out.get("pass") if isinstance(strict_out, dict) else None,
    })
    _write_json(bundle_dir / f"overlap__{district_id}__strict__{sig8}.json", strict_payload)
    written.append(f"overlap__{district_id}__strict__{sig8}.json")

    # 2) projected(columns@k=3,auto)
    proj_auto_payload = dict(base_hdr)
    proj_auto_payload.update({
        "policy": "projected(columns@k=3,auto)",
        "projection_context": {
            "lanes": (lanes_vec or []),
            "lanes_popcount": lanes_pop,
            "lanes_sig8": lanes_sig8,
        },
        "na": bool(proj_meta.get("na")) if isinstance(proj_meta, dict) else False,
        "reason": na_reason,
        "verdict": proj_out.get("pass") if isinstance(proj_out, dict) else None,
    })
    _write_json(bundle_dir / f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json", proj_auto_payload)
    written.append(f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json")

    # 3) ab_compare (strict vs projected_auto)
    _, strict_sig8 = _canon_dump_and_sig8(strict_payload)
    _, auto_sig8   = _canon_dump_and_sig8(proj_auto_payload)
    ab_auto_payload = dict(base_hdr)
    ab_auto_payload.update({
        "policy": "strict__VS__projected(columns@k=3,auto)",
        "embed": {
            "left":  {"policy": "strict", "sig8": strict_sig8},
            "right": {"policy": "projected(columns@k=3,auto)", "sig8": auto_sig8},
        }
    })
    _write_json(bundle_dir / f"ab_compare__strict_vs_projected_auto__{sig8}.json", ab_auto_payload)
    written.append(f"ab_compare__strict_vs_projected_auto__{sig8}.json")

    # 4) projector_freezer
    file_pi_valid   = bool(ss.get("file_pi_valid", True))
    file_pi_reasons = list(ss.get("file_pi_reasons", []) or [])
    freezer_payload = dict(base_hdr)
    freezer_payload.update({
        "policy": "projector_freezer",
        "status": "OK" if file_pi_valid else "FAIL",
        "file_pi_valid": file_pi_valid,
        "file_pi_reasons": file_pi_reasons,
    })
    _write_json(bundle_dir / f"projector_freezer__{district_id}__{sig8}.json", freezer_payload)
    written.append(f"projector_freezer__{district_id}__{sig8}.json")

    # 5–6) FILE pair (only if Π valid)
    if file_pi_valid:
        proj_file_payload = dict(base_hdr)
        proj_file_payload.update({
            "policy": "projected(columns@k=3,file)",
            "verdict": None,  # v2: presence + cross-refs only
        })
        _write_json(bundle_dir / f"overlap__{district_id}__projected_columns_k_3_file__{sig8}.json", proj_file_payload)
        written.append(f"overlap__{district_id}__projected_columns_k_3_file__{sig8}.json")

        _, file_sig8 = _canon_dump_and_sig8(proj_file_payload)
        ab_file_payload = dict(base_hdr)
        ab_file_payload.update({
            "policy": "strict__VS__projected(columns@k=3,file)",
            "embed": {
                "left":  {"policy": "strict", "sig8": strict_sig8},
                "right": {"policy": "projected(columns@k=3,file)", "sig8": file_sig8},
            }
        })
        _write_json(bundle_dir / f"ab_compare__strict_vs_projected_file__{sig8}.json", ab_file_payload)
        written.append(f"ab_compare__strict_vs_projected_file__{sig8}.json")

    # --- bundle.json ---
    bundle_idx = {
        "run_id": rc.get("run_id", ""),
        "sig8": sig8,
        "district_id": district_id,
        "filenames": written,
        "files": {
            "strict": str(bundle_dir / f"overlap__{district_id}__strict__{sig8}.json"),
            "projected_auto": str(bundle_dir / f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json"),
            "ab_auto": str(bundle_dir / f"ab_compare__strict_vs_projected_auto__{sig8}.json"),
            "freezer": str(bundle_dir / f"projector_freezer__{district_id}__{sig8}.json"),
            "projected_file": (str(bundle_dir / f"overlap__{district_id}__projected_columns_k_3_file__{sig8}.json") if file_pi_valid else None),
            "ab_file": (str(bundle_dir / f"ab_compare__strict_vs_projected_file__{sig8}.json") if file_pi_valid else None),
        },
        "lanes": {"popcount": lanes_pop, "sig8": lanes_sig8},
        "counts": {"written": len(written)},
    }
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
    After each solver press, rebuilds bundle.json and writes loop_receipt__{fixture}.json.
    Also appends lanes fields into the suite index when available.
    """
    import json as _json
    import streamlit as _st
    from pathlib import Path as _Path

    manifest_abs = _abs_from_manifest(manifest_path)
    if not manifest_abs.exists():
        _st.error(f"Manifest not found: {manifest_abs}")
        return False, f"Manifest not found: {manifest_abs}", 0

    # Load JSONL
    lines = []
    with manifest_abs.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(_json.loads(raw))
            except Exception as e:
                return False, f"Bad JSONL line: {raw[:120]}… ({e})", 0

    ok_count = 0
    total = len(lines)

    for i, rec in enumerate(lines, 1):
        # inside _RUN_SUITE_CANON, in the fixture loop, before _one_press_triple()
        fid = rec.get("fixture") or rec.get("id") or f"fx{i:02d}"
        _st.session_state["fixture_label"] = fid
        # (optional, but helpful) pass snapshot id too, if not already set
        if "world_snapshot_id" not in _st.session_state:
            _st.session_state["world_snapshot_id"] = snapshot_id


        # Seed inputs
        try:
            _set_inputs_for_run(B, C, H, U)
        except Exception:
            pass

        # Existence preflight
        miss = [p for p in (B, C, H, U) if not _abs_from_manifest(p).exists()]
        if miss:
            _st.warning(f"[{fid}] Missing files: {', '.join(miss)}")
            continue

        # Call solver with arity guard
        try:
            ret = run_overlap_once()
        except Exception as e:
            ok, msg, bundle_dir = False, f"solver error: {e}", None
        else:
            ok, msg, bundle_dir = _solver_ret_as_tuple(ret)

        _st.write(f"{fid} → {'ok' if ok else 'fail'} · {msg}")
        if ok:
            ok_count += 1

        # Resolve bundle dir
        bdir = _Path(bundle_dir) if bundle_dir else None
        if not bdir or not bdir.exists():
            try:
                certs = list((_REPO_DIR / "logs" / "certs").glob("*/*"))
                bdir = max(certs, key=lambda p: p.stat().st_mtime) if certs else None
            except Exception:
                bdir = None

        # Pass1/2: bundle + receipt
        lanes_pop = None; lanes_sig8 = None
        try:
            if bdir and bdir.exists():
                bundle = _v2_bundle_index_rebuild(bdir)
                lanes = bundle.get("lanes") or {}
                lanes_pop, lanes_sig8 = lanes.get("popcount"), lanes.get("sig8")
                _v2_write_loop_receipt(bdir, fid, snapshot_id, bundle)
        except Exception as e:
            _st.warning(f"[{fid}] bundling warning: {e}")

        # suite index row
        try:
            _suite_index_add_row({
                "fixture_id": fid,
                "snapshot_id": snapshot_id,
                "bundle_dir": str(bdir) if bdir else None,
                "lanes_popcount": lanes_pop,
                "lanes_sig8": lanes_sig8,
            })
        except Exception:
            pass

    return True, f"Completed {ok_count}/{total} fixtures.", ok_count

# neutralized (final alias installed at EOF): run_suite_from_manifest = _RUN_SUITE_CANON

# =============================================================================
# FINAL, SINGLE-SOURCE RUNNER ALIAS (v2 repo-only preferred)
# Put this block at the very bottom of the file.
try:
    del run_suite_from_manifest  # if defined earlier, remove to avoid shadowing
except Exception:
    pass

def run_suite_from_manifest(manifest_path: str, snapshot_id: str):
    """
    Final dispatcher. Always returns a 3-tuple (ok: bool, msg: str, count: int).
    Prefers repo-only v2 runner, falls back to CANON, then legacy.
    """
    g = globals()
    if "_RUN_SUITE_V2_REPO_ONLY" in g:
        ret = g["_RUN_SUITE_V2_REPO_ONLY"](manifest_path, snapshot_id)
    elif "_RUN_SUITE_CANON" in g:
        ret = g["_RUN_SUITE_CANON"](manifest_path, snapshot_id)
    elif "run_suite_from_manifest__legacy" in g:
        ret = g["run_suite_from_manifest__legacy"](manifest_path, snapshot_id)
    else:
        return False, "No suite runner available (need _RUN_SUITE_V2_REPO_ONLY or _RUN_SUITE_CANON).", 0

    # Normalize to a 3-tuple
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
# =============================================================================




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

def _co_allzero_cols(M):
    m,n = _co_shape(M)
    zero = [True]*n
    for i in range(m):
        row = M[i]
        for j in range(n):
            if row[j] == 1:
                zero[j] = False
    return zero  # True means column is all zeros

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





def _svr_run_once_computeonly(ss=None):
    g = globals()
    resolver = g.get("_svr_resolve_all_to_paths") or g.get("resolve_all_to_paths")
    freezer  = g.get("_svr_freeze_ssot") or g.get("freeze_ssot")
    if not resolver or not freezer:
        return False, "Missing SSOT helpers (_svr_resolve_all_to_paths/_svr_freeze_ssot).", ""

    pb = resolver() or {}
        # --- Paths & matrices
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
    ss["_last_inputs_paths"] = dict(paths_dict)

    # Matrices
    H2, d3, C3 = _co_extract_mats(pb)
    _, n3 = _co_shape(d3)

    # Robust H/C inference
    Ht_guess = _co_guess_H_variant(paths_dict)         # H00/H01/H10/H11
    C_raw_stem = _co_stem(pC)                          # e.g., "CMAP__...__C110"
    Ct_guess = _co_guess_C_code(paths_dict, C_raw_stem)  # e.g., "C110"

    # District & IDs (deterministic fallbacks)
    D = "D2" if "D2" in paths_dict["B"].upper() else ("D3" if "D3" in paths_dict["B"].upper() else "D?")
    ib, rc = freezer(pb); ib = dict(ib or {}); rc = dict(rc or {})

    district_id = rc.get("district_id")
    if not district_id or district_id.startswith("UNKNOWN"):
        district_id = _co_guess_district_id(D, d3)

    sig8 = rc.get("sig8") or (rc.get("embed_sig","")[:8] if rc.get("embed_sig") else None)
    if not sig8 or sig8 == "00000000":
        sig8 = _co_sig8_from_mats(H2, d3, C3)

    snap_id = _co_guess_snapshot_id(ss, rc, paths_dict)

    # Final fixtures and human label (short codes)
    fixtures = {"district": D, "H": Ht_guess, "C": Ct_guess, "C_raw": C_raw_stem, "U": "U"}
    fixture_label = f"{D}_{Ht_guess}_{Ct_guess}"


    def _stamp(obj):
        o = dict(obj or {})
        o.setdefault("fixtures", fixtures)
        o.setdefault("fixture_label", fixture_label)
        o.setdefault("snapshot_id", snap_id)
        o.setdefault("sig8", sig8)
        o.setdefault("written_at_utc", int(time.time()))
        return o

    bdir  = _co_bundle_dir(district_id, fixture_label, sig8)
    names = {
        "strict":         f"overlap__{district_id}__strict__{sig8}.json",
        "projected_auto": f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json",
        "ab_auto":        f"ab_compare__strict_vs_projected_auto__{sig8}.json",
        "freezer":        f"projector_freezer__{district_id}__{sig8}.json",
        "projected_file": f"overlap__{district_id}__projected_columns_k_3_file__{sig8}.json",
        "ab_file":        f"ab_compare__projected_columns_k_3_file__{sig8}.json",
    }

    _co_write_json(bdir / names["strict"],         _stamp(strict))
    _co_write_json(bdir / names["projected_auto"], _stamp(auto))
    _co_write_json(bdir / names["ab_auto"],        _stamp(ab_auto))
    _co_write_json(bdir / names["freezer"],        _stamp(frz_info))
    _co_write_json(bdir / names["projected_file"], _stamp(pfile))
    _co_write_json(bdir / names["ab_file"],        _stamp(ab_file))

    bundle = {
        "district_id": district_id,
        "fixture_label": fixture_label,
        "fixtures": fixtures,
        "sig8": sig8,
        "filenames": [names[k] for k in ("strict","projected_auto","ab_auto","freezer","projected_file","ab_file")],
        "core_counts": {"written": 6},
        "written_at_utc": int(time.time())
    }
    _co_write_json(bdir / "bundle.json", bundle)
    _co_write_json(bdir / f"loop_receipt__{fixture_label}.json", {
        "snapshot_id": snap_id,
        "run_id": str(uuid.uuid4()),
        "district_id": district_id,
        "fixture_label": fixture_label,
        "sig8": sig8,
        "bundle_dir": str(bdir),
        "core_counts": {"written": 6, "ok": None, "na": None},
        "timestamps": {"receipt_written_at": time.time()}
    })

    try:
        import streamlit as _st
        _st.session_state["last_bundle_dir"] = str(bdir)
    except Exception:
        pass

    return True, f"v2 compute-only 1× bundle → {bdir}", str(bdir)

def one_press_v2_compute_only_ui():
    import streamlit as st
    ss = st.session_state
    st.markdown("### One-press (v2 compute-only, no legacy)")
    if st.button("Run solver (one press) — v2 compute-only", key="btn_svr_run_v2_compute_only"):
        ss["_solver_busy"] = True
        ss["_solver_one_button_active"] = True
        try:
            ok, msg, bundle_dir = _svr_run_once_computeonly(ss)
            if bundle_dir:
                ss["last_bundle_dir"] = bundle_dir
            (st.success if ok else st.error)(msg)
        except Exception as e:
            st.error(f"Solver run failed: {e}")
        finally:
            ss["_solver_one_button_active"] = False
            ss["_solver_busy"] = False

def tail_and_download_ui_v2_compute_only():
    import streamlit as st, os, json, zipfile
    from pathlib import Path
    ss = st.session_state
    last_dir = ss.get("last_bundle_dir", "")
    st.markdown("#### Latest cert files (v2 compute-only)")
    if not last_dir or not os.path.isdir(last_dir):
        st.info("No bundle.json yet — run the solver to write certs.")
        return
    try:
        p_bundle = Path(last_dir) / "bundle.json"
        if p_bundle.exists():
            bj = json.loads(p_bundle.read_text(encoding="utf-8"))
            lab = bj.get("fixture_label")
            if lab and ss.get("fixture_label") != lab:
                ss["fixture_label"] = lab
            files = bj.get("filenames", [])
        else:
            files = []
    except Exception:
        files = []
    if files:
        tail = files[-6:][::-1]
        for fn in tail:
            st.write(f"• {fn}")
    else:
        st.info("No files listed in bundle.json.")
    # zip
    try:
        zpath = Path(last_dir) / "bundle.zip"
        with zipfile.ZipFile(str(zpath), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, fns in os.walk(last_dir):
                for fn in fns:
                    if fn.endswith(".zip"):
                        continue
                    full = Path(root)/fn
                    arc  = os.path.relpath(str(full), start=str(last_dir))
                    zf.write(str(full), arc)
        with open(zpath, "rb") as fh:
            st.download_button("Download bundle.zip", data=fh, file_name="bundle.zip", mime="application/zip", key="btn_dl_bundle_zip_v2_compute_only")
    except Exception as e:
        st.warning(f"Zip build/serve issue: {e}")

try:
    import streamlit as st
    with st.expander("V2 compute-only runner", expanded=True):
        one_press_v2_compute_only_ui()
        tail_and_download_ui_v2_compute_only()
except Exception:
    pass
# ====================== END V2 CANONICAL — COMPUTE-ONLY ======================




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





def _hard_co_hash8(obj) -> str:
    h = _hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return h[:8]

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

def _hard_bundle_dir(district_id: str, fixture_label: str, sig8: str) -> _Ph:
    return _Ph("logs/certs") / str(district_id) / str(fixture_label) / str(sig8)


