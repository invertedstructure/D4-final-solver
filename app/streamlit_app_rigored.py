import csv as _csv
import hashlib as _hash
import hashlib as _hashlib
import hashlib as _Vhash
import importlib.util
import json as _json
import json as _Vjson
import json as _j
import os as _os
import re
import shutil
import sys
import tempfile
import time
import time as _time
import types
import uuid
import uuid as _uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path as _Path
from pathlib import Path as _Ph
from pathlib import Path as _VPath
# Third-party
import streamlit as st
import streamlit as _st
# --- C1 canonical paths (tuple; JSON-first) ---

def _c1_paths():
    base = _Path("logs") / "reports"
    base.mkdir(parents=True, exist_ok=True)
    return (base / "coverage.jsonl", base / "coverage_rollup.csv")

def _canon_dump_and_sig8(obj):
    """Return (canonical_json_text, first_8_of_sha256) for small cert payloads (v2).

    Thin wrapper around canonical_json/hash_json_sig8 so all v2 artifacts share the
    same hashing discipline.
    """
    can_text = canonical_json(obj)
    sig8 = hash_json_sig8(obj)
    return can_text, sig8


def _v2_coverage_path():
    try:
        root = _REPO_DIR
    except Exception:
        root = _Path(__file__).resolve().parents[1]
    p = root / "logs" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p / "coverage.jsonl"

# SSOT: canonical v2 coverage append into logs/reports/coverage.jsonl
def _v2_coverage_append(row: dict):
    """Append one JSON line to coverage.jsonl (best-effort)."""
    import json as _json, time as _time
    row = dict(row or {})
    row.setdefault("ts_utc", int(_time.time()))
    with _v2_coverage_path().open("a", encoding="utf-8") as f:
        f.write(_json.dumps(row, separators=(",", ":"), sort_keys=False) + "\n")


def _coverage_row_is_tau_ping(rec: dict) -> bool:
    """Return True iff rec is a Time(τ) health ping row.

    This centralizes the (unit, kind) check so C1/coverage logic can
    stay in sync with the Time(τ) writer helpers.
    """
    if not isinstance(rec, dict):
        return False
    return (
        rec.get("unit") == "time_tau"
        and rec.get("kind") == "tau_c4_health"
    )


def _coverage_row_is_fixture_event(rec: dict) -> bool:
    """Return True iff rec looks like a normal per-fixture coverage event.

    We explicitly exclude Time(τ) health pings and any rows that do not
    carry a fixture_label. This will be used to keep C1 counts and
    rollup rows aligned without baking the logic into multiple loops.
    """
    if not isinstance(rec, dict):
        return False
    if _coverage_row_is_tau_ping(rec):
        return False
    fid = rec.get("fixture_label")
    if not fid:
        return False
    return True


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
            if j.get("snapshot_id") != snapshot_id:
                continue
            if not _coverage_row_is_fixture_event(j):
                continue
            n += 1
    return n




def _d3_count_fixtures_for_snapshot(snapshot_id: str | None, manifest_path: _Path | None = None) -> int | None:
    """
    D3 helper: count manifest fixtures for a given snapshot.

    Only counts rows that:
      - have fixture_label set, and
      - have snapshot_id empty/None or equal to the provided snapshot_id.

    Returns None when snapshot_id is falsy, the manifest is missing, or no rows match.
    """
    if not snapshot_id:
        return None

    try:
        if manifest_path is None:
            manifest_path = _svr_current_run_manifest_path()
    except Exception:
        manifest_path = None

    if manifest_path is None:
        return None

    mp = _Path(manifest_path)
    if not mp.exists():
        return None

    n = 0
    try:
        for rec in iter_v2_suite_rows(mp):
            if not isinstance(rec, dict):
                continue
            sid = rec.get("snapshot_id")
            if sid not in (None, "", snapshot_id):
                continue
            fid = rec.get("fixture_label")
            if not fid:
                continue
            n += 1
    except Exception:
        # Analyzer-only helper: on any unexpected error, report no fixtures.
        return None

    return n if n > 0 else None


def _d3_integer_pass_stats(snapshot_id: str | None) -> dict:
    """
    D3 helper: compute integer-pass stats for a snapshot.

    All arithmetic here is analyzer-only and never gates the pipeline.
    """
    # Coverage rows for this snapshot (0 on any error or missing coverage file).
    rows = 0
    if snapshot_id:
        try:
            rows = _v2_coverage_count_for_snapshot(snapshot_id)
        except Exception:
            rows = 0

    # Manifest fixture count for this snapshot (None when unknown).
    n_fixtures = _d3_count_fixtures_for_snapshot(snapshot_id)

    passes = None
    has_integer = None
    if n_fixtures is not None and n_fixtures > 0:
        # D3 spine: rows(S) vs passes(S) — integer-passes invariant.
        # 0 rows ⇒ 0 passes, has_integer=True.
        passes = rows // n_fixtures
        has_integer = (rows % n_fixtures == 0)

    return {
        "snapshot_id": snapshot_id,
        "coverage_rows_for_snapshot": rows,
        "n_fixtures": n_fixtures,
        "passes_for_snapshot": passes,
        "has_integer_passes": has_integer,
    }


def _c1_debug_snapshot_summary(snapshot_id: str) -> dict:
    """
    Dev-only helper to sanity-check C1/τ for a given snapshot_id.

    Returns a dict with:
      - snapshot_id
      - coverage_rows_for_snapshot  (from coverage.jsonl)
      - rollup_all_row              (dict or None)
      - n_fixtures                  (from manifest; None when unknown)
      - passes_for_snapshot         (integer passes; None when unknown)
      - has_integer_passes          (True/False when n_fixtures known; else None)
    """
    cov_path, csv_out = _c1_paths()
    stats = _d3_integer_pass_stats(snapshot_id)

    all_row = None
    try:
        if csv_out.exists():
            import csv as _csv
            with csv_out.open("r", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    if row.get("prox_label") == "ALL":
                        all_row = row
                        break
    except Exception:
        all_row = None

    return {
        "snapshot_id": snapshot_id,
        "coverage_rows_for_snapshot": stats.get("coverage_rows_for_snapshot"),
        "rollup_all_row": all_row,
        "n_fixtures": stats.get("n_fixtures"),
        "passes_for_snapshot": stats.get("passes_for_snapshot"),
        "has_integer_passes": stats.get("has_integer_passes"),
    }

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

# SSOT: canonical C1 rollup over coverage.jsonl → coverage_rollup.csv
def _coverage_rollup_write_csv(snapshot_id: str | None = None):
    """
    Build C1 rollup over logs/reports/coverage.jsonl.
    If snapshot_id is provided, filter to that snapshot only (v2 uses the __real one).
    Writes logs/reports/coverage_rollup.csv and returns its Path (or None if no coverage file).
    """
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

            # Time(τ) health events: snapshot-aware when snapshot_id is provided.
            # If snapshot_id is None, keep the global "latest ping wins" behavior.
            if _coverage_row_is_tau_ping(j):
                # Time(τ) health pings are not treated as mismatch rows.
                try:
                    j_sid = j.get("snapshot_id")

                    if snapshot_id:
                        # For a snapshot-specific rollup, ignore τ pings from other
                        # snapshots (or untagged ones).
                        if not j_sid or j_sid != snapshot_id:
                            continue

                    # snapshot_id is None → keep prior behavior: latest ping wins globally.
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

            if not _coverage_row_is_fixture_event(j):
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

    # Compute integer-pass stats (ALL row only) for this snapshot, if available.
    n_fixtures = None
    passes_for_snapshot = None
    has_integer_passes = None
    coverage_rows_for_snapshot = None

    if snapshot_id:
        try:
            n_fixtures = _d3_count_fixtures_for_snapshot(snapshot_id)
        except Exception:
            n_fixtures = None

    if "ALL" in agg:
        try:
            coverage_rows_for_snapshot = agg["ALL"]["count"]
        except Exception:
            coverage_rows_for_snapshot = None

    if (
        n_fixtures is not None
        and n_fixtures > 0
        and coverage_rows_for_snapshot is not None
    ):
        passes_for_snapshot = coverage_rows_for_snapshot // n_fixtures
        has_integer_passes = (coverage_rows_for_snapshot % n_fixtures == 0)

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
            "n_fixtures",
            "coverage_rows_for_snapshot",
            "passes_for_snapshot",
            "has_integer_passes",
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

            if label == "ALL":
                n_fix_col = n_fixtures if n_fixtures is not None else ""
                cov_rows_col = (
                    coverage_rows_for_snapshot
                    if coverage_rows_for_snapshot is not None
                    else ""
                )
                passes_col = passes_for_snapshot if passes_for_snapshot is not None else ""
                has_int_col = ""
                if has_integer_passes is True:
                    has_int_col = "1"
                elif has_integer_passes is False:
                    has_int_col = "0"
            else:
                n_fix_col = ""
                cov_rows_col = ""
                passes_col = ""
                has_int_col = ""

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
                n_fix_col,
                cov_rows_col,
                passes_col,
                has_int_col,
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

# --- Canonical JSON + hashing helpers for v2 artifacts ---

# --- v2 canonical JSON + hashing (DO NOT PRUNE) ---

def canonical_json(obj) -> str:
    """Canonical JSON dump for v2 artifacts.

    - Normalizes via _v2_canonical_obj (drops ephemeral keys / None)
    - Uses deterministic json.dumps with sorted keys and compact separators
    """
    can = _v2_canonical_obj(obj)
    return _json.dumps(can, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def hash_json(obj) -> str:
    """Full SHA-256 hex digest of canonical_json(obj)."""
    raw = canonical_json(obj).encode("utf-8")
    return _hash.sha256(raw).hexdigest()

def hash_json_sig8(obj) -> str:
    """First 8 hex characters of hash_json(obj)."""
    return hash_json(obj)[:8]

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


def make_fixture_label(district_id: str, h_mask: str, c_mask: str) -> str:
    """Build canonical fixture label "D*_H*_C*" from components.

    Example:
      make_fixture_label("D3", "10", "111") -> "D3_H10_C111"
    """
    d = str(district_id or "").strip()
    h = str(h_mask or "").strip()
    c = str(c_mask or "").strip()
    # Normalize if caller passed "H10"/"C111" style masks.
    if h.startswith("H"):
        h = h[1:]
    if c.startswith("C"):
        c = c[1:]
    return f"{d}_H{h}_C{c}"


def parse_fixture_label(label: str) -> tuple[str, str, str]:
    """Parse a fixture label like "D3_H10_C111" into (district_id, h_mask, c_mask).

    Returns empty strings on failure rather than raising.
    """
    import re as _re

    s = str(label or "").strip()
    if not s:
        return "", "", ""

    # District: prefer an explicit D\d+ token, fallback to the first chunk.
    mD = _re.search(r"(?:^|_)(D\d+)", s)
    if mD:
        district = mD.group(1)
    else:
        district = s.split("_")[0]

    # Masks: capture the numeric part after H*/C*.
    mH = _re.search(r"_H([A-Za-z0-9]+)", s)
    mC = _re.search(r"_C([A-Za-z0-9]+)", s)
    h_mask = mH.group(1) if mH else ""
    c_mask = mC.group(1) if mC else ""

    return district, h_mask, c_mask

# Dev sanity (kept cheap and local to this module).
# assert parse_fixture_label(make_fixture_label("D3", "10", "111")) == ("D3", "10", "111")

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
  
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")


# Page config must be the first Streamlit command
SCHEMA_VERSION = "2.0.0"
ENGINE_REV     = "rev-20251022-1"

DIRS = {"root": "logs", "certs": "logs/certs", "snapshots": "logs/snapshots", "reports": "logs/reports", "suite_runs": "logs/suite_runs", "exports": "logs/exports"}
# ---------- Suite helpers (v2) ----------





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


def _svr_current_run_snapshot_id() -> str | None:
    """Preferred snapshot_id for the current v2 run.

    Order:
      1) st.session_state['run_ctx']['snapshot_id']
      2) _svr_current_snapshot_id() (world_snapshot.latest.json)
      3) None
    """
    import streamlit as st  # local import to avoid hard dependency at import time
    try:
        rc = dict(st.session_state.get("run_ctx") or {})
    except Exception:
        rc = {}
    sid = rc.get("snapshot_id")
    if sid:
        return str(sid)
    return _svr_current_snapshot_id()



def _svr_current_run_manifest_path() -> _Path:
    """Preferred manifest_full_scope.jsonl path for the current v2 run.

    Order:
      1) st.session_state['run_ctx']['manifest_full_scope_path']
      2) _MANIFESTS_DIR / "manifest_full_scope.jsonl"
      3) <repo_root>/logs/manifests/manifest_full_scope.jsonl

    Returns:
        A _Path; the caller is responsible for checking existence.
    """
    # Prefer the manifest recorded in the current run_ctx when available.
    try:
        import streamlit as st  # local import to avoid hard dependency at import time
        try:
            rc = dict(st.session_state.get("run_ctx") or {})
        except Exception:
            rc = {}
        mp = rc.get("manifest_full_scope_path")
        if mp:
            return _Path(mp)
    except Exception:
        # Fall back to repo-level defaults below.
        pass

    # Fall back to the canonical manifests directory under the repo root.
    try:
        manifests_dir = _MANIFESTS_DIR  # type: ignore[name-defined]
    except Exception:
        try:
            repo_root = _repo_root()
        except Exception:
            repo_root = _Path(__file__).resolve().parents[1]
        manifests_dir = _Path(repo_root) / "logs" / "manifests"

    return _Path(manifests_dir) / "manifest_full_scope.jsonl"

def _svr_world_snapshots_dir() -> _Path:
    """Return the directory for world snapshots, creating it if needed."""
    snaps_dir = _Path(DIRS.get("snapshots", "logs/snapshots"))
    snaps_dir.mkdir(parents=True, exist_ok=True)
    return snaps_dir


def _svr_world_snapshot_path(snapshot_id: str) -> _Path:
    """Canonical path for a world snapshot file."""
    snaps_dir = _svr_world_snapshots_dir()
    return snaps_dir / f"world_snapshot__{snapshot_id}.json"


# SSOT: canonical writer for world_snapshot__{snapshot_id}.json + world_snapshot.latest.json
def _svr_write_world_snapshot(snapshot: dict) -> _Path:
    """Write a world snapshot file and update the latest pointer.

    Returns the path to the written snapshot JSON.
    """
    import json as _json
    sid = str(snapshot.get("snapshot_id") or "UNSET")
    path = _svr_world_snapshot_path(sid)
    # Deterministic JSON for stability across runs with the same body.
    path.write_text(
        _json.dumps(snapshot, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )

    # Mirror pointer for quick lookup by other panels.
    snaps_dir = _svr_world_snapshots_dir()
    ptr = snaps_dir / "world_snapshot.latest.json"
    ptr_body = {"snapshot_id": sid, "path": str(path)}
    ptr.write_text(
        _json.dumps(ptr_body, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return path



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

_os.makedirs(DIRS["certs"], exist_ok=True)
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


# SSOT: canonical v2 bundle dir layout for certs/loop_receipts
def make_bundle_dir(district_id: str, fixture_label: str, sig8: str):
    """Canonical bundle dir for v2 certs.

    Pattern (under repo root): logs/certs/{district_id}/{fixture_label}/{sig8}
    """
    # Prefer the frozen _CERTS_DIR if available, else fall back to repo_root/logs/certs.
    try:
        root = _CERTS_DIR
    except Exception:
        try:
            root = _REPO_ROOT / "logs" / "certs"
        except Exception:
            root = _Path("logs") / "certs"
    return _Path(root) / str(district_id) / str(fixture_label) / str(sig8)


def make_loop_receipt_path(bundle_dir, fixture_label: str):
    """Return Path to the loop_receipt.v2 JSON inside a bundle dir.

    Pattern: {bundle_dir}/loop_receipt__{fixture_label}.json
    """
    b = _Path(bundle_dir)
    return b / f"loop_receipt__{fixture_label}.json"

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

# --- v2 cert writers (DO NOT PRUNE) ---

# SSOT: canonical payload builders for v2 strict/projected certs
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



# --- v2 loop receipts + bundle dirs (DO NOT PRUNE) ---

# SSOT: canonical payload builder for loop_receipt.v2
def build_v2_loop_receipt(
    *,
    run_id: str | None,
    district_id: str,
    fixture_label: str,
    sig8: str,
    bundle_dir,
    paths: dict,
    core_written: int,
    dims: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a canonical loop_receipt.v2 payload for a single bundle.

    This helper does **not** compute sig8; it trusts the caller to supply the
    short-hash (typically the strict cert sig8 for the fixture).
    """
    import time as _time

    schema_version = globals().get("SCHEMA_VERSION", "2.0.0")
    engine_rev = globals().get("ENGINE_REV", "rev-UNSET")

    bdir = _Path(bundle_dir) if bundle_dir is not None else None

    receipt: dict = {
        "schema": "loop_receipt.v2",
        "schema_version": str(schema_version),
        "engine_rev": str(engine_rev),
        "run_id": run_id,
        "district_id": str(district_id),
        "fixture_label": str(fixture_label),
        "sig8": str(sig8 or ""),
        "bundle_dir": str(bdir.resolve()) if bdir is not None else None,
        "paths": dict(paths or {}),
        "core_counts": {"written": int(core_written)},
        "timestamps": {"receipt_written_at": int(_time.time())},
    }

    if isinstance(dims, dict):
        receipt["dims"] = {"n2": dims.get("n2"), "n3": dims.get("n3")}

    if isinstance(extra, dict):
        # Only add keys that are not already present to avoid accidental override.
        for k, v in extra.items():
            if k not in receipt:
                receipt[k] = v

    return receipt


# --- v2 world snapshot + manifest builders (DO NOT PRUNE) ---

def build_v2_world_snapshot_from_body(body_without_id: dict) -> dict:
    """Attach a deterministic snapshot_id to a world snapshot body.

    The caller provides the core body (without snapshot_id); we fill
    schema_version / engine_rev if missing, and set::

        snapshot_id = "ws__" + hash_json_sig8(body_without_id_with_schema)

    The returned dict is safe to write as world_snapshot__{snapshot_id}.json
    and to mirror into world_snapshot.latest.json.
    """
    # Start from a shallow copy so we never mutate caller state.
    base = dict(body_without_id or {})

    schema_version = str(globals().get("SCHEMA_VERSION", "2.0.0"))
    engine_rev = str(globals().get("ENGINE_REV", "rev-UNSET"))

    base.setdefault("schema_version", schema_version)
    base.setdefault("engine_rev", engine_rev)
    base.setdefault("schema", "world_snapshot.v2")

    # Compute sig8 on the body *without* snapshot_id to keep it stable.
    tmp = dict(base)
    tmp.pop("snapshot_id", None)
    sig8 = hash_json_sig8(tmp)
    snapshot_id = f"ws__{sig8}"

    snapshot = dict(base)
    snapshot["snapshot_id"] = snapshot_id
    return snapshot



def build_v2_suite_row(
    *,
    snapshot_id: str | None,
    district_id: str,
    fixture_label: str,
    strict_sig8: str,
    bundle_dir,
    strict_flags: dict | None = None,
    projected_flags: dict | None = None,
) -> dict:
    """Build the canonical per-fixture suite row for v2 manifests.

    This row is designed to be usable both for JSONL and for CSV export.
    It focuses on identity fields; policy-specific details live in small
    nested dicts that callers may ignore or flatten later.

    The caller is responsible for attaching any additional fields such as
    ``paths`` or ``dims`` after this helper returns.
    """

    schema_version = str(globals().get("SCHEMA_VERSION", "2.0.0"))
    engine_rev = str(globals().get("ENGINE_REV", "rev-UNSET"))

    bdir = _Path(bundle_dir) if bundle_dir is not None else None

    row: dict = {
        "schema": "suite_row.v2",
        "schema_version": schema_version,
        "engine_rev": engine_rev,
        "snapshot_id": str(snapshot_id or ""),
        "district_id": str(district_id),
        "fixture_label": str(fixture_label),
        "strict_sig8": str(strict_sig8 or ""),
        "bundle_dir": str(bdir.resolve()) if bdir is not None else None,
    }

    if isinstance(strict_flags, dict):
        row["strict"] = dict(strict_flags)
    if isinstance(projected_flags, dict):
        row["projected"] = dict(projected_flags)

    return row


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



def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src = (cfg.get("source") or {}).get("3", "auto")
    mode = "file" if src == "file" else "auto"
    # keep your established label shape
    return f"projected(columns@k=3,{mode})"

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
     
    def _shape(M):
        return (len(M), len(M[0]) if (M and M[0]) else 0)

           

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

# ==============================================================================
def _guarded_atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush(); _os.fsync(f.fileno())
    _os.replace(tmp, path)

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
    rc = _svr_run_ctx_update(n2=n2, n3=n3, d3=d3)
    return ib, rc


def _svr_run_ctx_update(**updates):
    """
    Merge updates into st.session_state['run_ctx'] and return the new dict.
    Never drops existing keys; ignores None-valued updates.
    """
    import streamlit as st  # local import to avoid top-level hard dependency

    rc = dict(st.session_state.get("run_ctx") or {})
    for k, v in updates.items():
        if v is not None:
            rc[k] = v
    st.session_state["run_ctx"] = rc
    return rc


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
    bundle_dir = make_bundle_dir(district_id, fixture_label, sig8)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Helpers
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
    # Write canonical bundle index sidecar (v2)
    _write_json(bundle_dir / "bundle_index.v2.json", bundle_idx)
    # Mirror into legacy bundle.json for back-compat
    _write_json(bundle_dir / "bundle.json", bundle_idx)

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

# ─────────────────────────────────────────────────────────────────────────────
# V2 strict mechanics: receipts → manifest → suite → histograms
# self-contained helpers, no changes to your 1× pipeline
# ─────────────────────────────────────────────────────────────────────────────


# --- Constants & dirs
_REPO_ROOT = _Path(__file__).resolve().parent.parent
_CERTS_DIR = _REPO_ROOT / "logs" / "certs"
_MANIFESTS_DIR = _REPO_ROOT / "logs" / "manifests"
_REPORTS_DIR = _REPO_ROOT / "logs" / "reports"
_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Write a loop_receipt (v2) into a given bundle dir
# SSOT: canonical writer for loop_receipt.v2 into a bundle dir
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

    receipt = build_v2_loop_receipt(
        run_id=(extra or {}).get("run_id"),
        district_id=district_id,
        fixture_label=fixture_label,
        sig8=sig8,
        bundle_dir=bdir,
        paths=P,
        core_written=core_written,
        dims=dims,
        extra=extra,
    )


    # Always write with proper filename (no UNKNOWN)
    outp = make_loop_receipt_path(bdir, fixture_label)
    _hard_co_write_json(outp, receipt)
    return True, f"[{fixture_label}] wrote loop_receipt.v2"


# --- Regenerate manifest_full_scope.jsonl by scanning loop_receipts
# SSOT: canonical writer for logs/manifests/manifest_full_scope.jsonl
def _v2_regen_manifest_from_receipts():
    """
    Scan logs/certs/**/loop_receipt__*.json (schema=loop_receipt.v2),
    validate absolute SSOT paths (B,C,H,U), deduplicate by fixture_label,
    and write logs/manifests/manifest_full_scope.jsonl atomically.

    Returns: (ok: bool, manifest_path: Path, kept_count: int)
    """
    import json as _json
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
    # Best-effort snapshot_id for this manifest: prefer the current world
    # snapshot pointer if available, otherwise leave blank.
    try:
        manifest_snapshot_id = _svr_current_snapshot_id()
    except Exception:
        manifest_snapshot_id = None


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
            # Identity fields sourced from the receipt, with gentle fallbacks.
            district_id = rj.get("district_id") or D_tag or "DUNKNOWN"
            strict_sig8 = rj.get("sig8") or ""
            bdir = rj.get("bundle_dir") or str(rp.parent)

            # dims (if present on the receipt)
            dims = None
            if isinstance(rj.get("dims"), dict):
                dims = {
                    "n2": rj["dims"].get("n2"),
                    "n3": rj["dims"].get("n3"),
                }

            # Base canonical header row.
            header = build_v2_suite_row(
                snapshot_id=manifest_snapshot_id,
                district_id=district_id,
                fixture_label=fid,
                strict_sig8=strict_sig8,
                bundle_dir=bdir,
            )

            # Preserve existing manifest fields (paths + dims) as top-level keys.
            row = dict(header)
            row["paths"] = paths
            if dims is not None:
                row["dims"] = dims

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


def iter_v2_suite_rows(manifest_path=None):
    """Yield dict rows from the v2 suite manifest (manifest_full_scope.jsonl).

    If manifest_path is None, default to logs/manifests/manifest_full_scope.jsonl
    under the current repo root.
    """
    import json as _json

    if manifest_path is None:
        try:
            manifests_dir = _MANIFESTS_DIR  # type: ignore[name-defined]
        except Exception:
            repo_root = _repo_root()
            manifests_dir = _Path(repo_root) / "logs" / "manifests"
        manifest_path = manifests_dir / "manifest_full_scope.jsonl"

    mp = _Path(manifest_path)
    if not mp.exists():
        return

    with mp.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def iter_v2_suite_rows_for_run(run_ctx=None):
    """Yield v2 suite rows for the current run.

    Uses run_ctx['manifest_full_scope_path'] when available; otherwise falls
    back to the default v2 manifest.
    """
    import streamlit as st  # local import

    rc = dict(run_ctx or st.session_state.get("run_ctx") or {})
    mp = rc.get("manifest_full_scope_path")
    if mp:
        yield from iter_v2_suite_rows(mp)
    else:
        yield from iter_v2_suite_rows()



# --- Histogram reductions over coverage.jsonl
# SSOT: canonical v2 histograms over coverage.jsonl → histograms_v2.json
def _v2_build_histograms_from_coverage(snapshot_id: str | None = None):
    """
    Build simple histograms over coverage.jsonl with a tolerant field mapper.
    Writes logs/reports/histograms_v2.json and returns (ok, msg, out_path).
    """
    import json as _json
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

def _hard_co_write_json(p: _Ph, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(_json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    __os.replace(tmp, p)

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

    if not isinstance(rec, dict):
        rec = {}

    fixture_label = rec.get("fixture_label") or ""
    # For now, district_id is the D-tag prefix of the fixture_label (D2, D3, …).
    district_id = rec.get("district_id") or (fixture_label.split("_")[0] if fixture_label else "DUNKNOWN")

    # Try to pick a snapshot_id from the manifest row, otherwise fall back to the
    # current v2 run snapshot pointer if available.
    try:
        snapshot_id = rec.get("snapshot_id") or _svr_current_run_snapshot_id()
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
        # v2: strict_sig8 via canonical hash_json_sig8(core_repr)
        strict_sig8 = hash_json_sig8(core_repr)

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


# SSOT: canonical C2 sweep over manifest_full_scope.jsonl → time_tau_local_flip_sweep*.jsonl
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

    # Resolve manifest path: prefer the current v2 run manifest when available.
    if manifest_path is None:
        manifest_ptr = _svr_current_run_manifest_path()
    else:
        manifest_ptr = _Path(manifest_path)

    if not manifest_ptr or not _Path(manifest_ptr).exists():
        msg = (
            f"Manifest not found at {manifest_ptr}. "
            "Run the v2 core (64×) to populate manifest_full_scope.jsonl first."
        )
        return False, msg, {}

    manifest_path = _Path(manifest_ptr)

    # Stream manifest rows and build sweep rows.
    rows = []
    n_total = n_in_domain = n_na = n_law_ok = 0
    snapshot_ids = set()

    for rec in iter_v2_suite_rows(manifest_path):
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
    
# --- V2 CORE (64×) — one press → receipts → manifest → suite → hist/zip
_st.subheader("V2 — 64× → Receipts → Manifest → Suite/Histograms")

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

        # 1.5) canonical world snapshot (v2)
        try:
            legacy_body = {
                "suite_kind": "v2_overlap_64",
                "bootstrap_manifest": str(man_boot),
                "n_fixtures": len(rows),
                "fixtures": rows,
            }
            world_snapshot = build_v2_world_snapshot_from_body(legacy_body)
            world_snapshot_path = _svr_write_world_snapshot(world_snapshot)
            _st.success(
                f"World snapshot written → {world_snapshot_path} · "
                f"snapshot_id={world_snapshot.get('snapshot_id')}"
            )
            try:
                sid = str(world_snapshot.get("snapshot_id") or "")
                if sid:
                    _svr_run_ctx_update(
                        snapshot_id=sid,
                        world_snapshot_path=str(world_snapshot_path),
                        manifest_bootstrap_path=str(man_boot),
                    )
            except Exception as e:
                _st.warning(f"run_ctx snapshot wiring failed (non-fatal): {e}")

        except Exception as e:
            _st.warning(f"World snapshot write failed (non-fatal): {e}")

                # Decide canonical snapshot id for this run (prefer world snapshot if available).
        try:
            rc = dict(_st.session_state.get("run_ctx") or {})
        except Exception:
            rc = {}
        run_snapshot_id = rc.get("snapshot_id") or (snapshot_id or _time.strftime("%Y%m%d-%H%M%S", _time.localtime()))

# 2) run 64× to emit receipts — SNAPSHOT __boot
        snap_boot = f"{run_snapshot_id}__boot"
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

        try:
            _svr_run_ctx_update(
                manifest_full_scope_path=str(real_man),
            )
        except Exception as e:
            _st.warning(f"run_ctx manifest wiring failed (non-fatal): {e}")

        # 4) run REAL manifest — canonical snapshot id (v2)
        ok3, msg3, cnt3 = run_suite_from_manifest(str(real_man), run_snapshot_id)
        (_st.success if ok3 else _st.warning)(f"Suite run: {msg3} · rows={cnt3}")

        # 5) histograms over coverage — prefer filtering to __real if supported
        try:
            try:
                okh, msgh, outp = _v2_build_histograms_from_coverage(run_snapshot_id)
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
            zip_path = _v2_pack_suite_fat_zip(run_snapshot_id)
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
            path_csv = _coverage_rollup_write_csv(snapshot_id=run_snapshot_id)
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
                    rows = int(all_row.get("count") or 0)
                    sel = all_row.get("mean_sel_mismatch_rate") or "—"
                    off = all_row.get("mean_offrow_mismatch_rate") or "—"
                    ker = all_row.get("mean_ker_mismatch_rate") or "—"
                    ctr = all_row.get("mean_ctr_rate") or "—"
                    _st.success(f"C1 Health ✅ Healthy · rows={rows} · sel={sel} · off={off} · ker={ker} · ctr={ctr}")
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
            parsed = _v2_coverage_count_for_snapshot(run_snapshot_id)
            if parsed < cnt3:
                _st.warning(f"Coverage parsed {parsed}/{cnt3} rows for snapshot {run_snapshot_id} (expected ≥ executed).")
            else:
                _st.info(f"Coverage parsed rows: {parsed} (snapshot {run_snapshot_id})")
        except Exception:
            pass

        # D3.1.B3 — dev-only C1 snapshot vs rollup invariant check
        with _st.expander("D3 debug — C1 snapshot sanity"):
            try:
                dbg = _c1_debug_snapshot_summary(run_snapshot_id)
                cov_rows = dbg.get("coverage_rows_for_snapshot")
                rollup_row = dbg.get("rollup_all_row") or {}
                n_fixtures = dbg.get("n_fixtures")
                passes = dbg.get("passes_for_snapshot")
                has_integer = dbg.get("has_integer_passes")

                rollup_count = None
                if isinstance(rollup_row, dict):
                    raw = rollup_row.get("count")
                    try:
                        if raw not in (None, "", "—"):
                            rollup_count = int(raw)
                    except Exception:
                        rollup_count = None

                integer_violation = (
                    (n_fixtures not in (None, 0))
                    and (cov_rows not in (None, 0))
                    and (passes is not None)
                    and (has_integer is False)
                )

                _st.write(
                    {
                        "coverage_rows_for_snapshot": cov_rows,
                        "rollup_all_row_count": rollup_count,
                        "rollup_all_row": rollup_row,
                        "n_fixtures": n_fixtures,
                        "passes_for_snapshot": passes,
                        "has_integer_passes": has_integer,
                        "integer_pass_violation": bool(integer_violation),
                    }
                )

                if n_fixtures not in (None, 0) and cov_rows is not None:
                    _st.caption(
                        f"D3 passes: rows={cov_rows}, n_fixtures={n_fixtures}, "
                        f"passes={passes if passes is not None else '—'}"
                    )

                if integer_violation:
                    _st.warning(
                        "D3 integer-pass note: coverage_rows_for_snapshot is not a multiple of "
                        "n_fixtures (partial/malformed batch?)."
                    )

                if (
                    cov_rows is not None
                    and rollup_count is not None
                    and cov_rows != rollup_count
                ):
                    _st.warning(
                        "D3 spine debug (dev-only): coverage_rows_for_snapshot "
                        f"({cov_rows}) does not match ALL rollup count ({rollup_count})."
                    )
            except Exception as e:
                _st.info(f"D3 debug snapshot sanity unavailable: {e}")
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
# SSOT-UI: primary C1 coverage/health view for v2 pipeline
with st.expander("C1 — Coverage rollup & health ping", expanded=False):
    cov_path, csv_out = _c1_paths()
    st.caption(f"Source: {cov_path} · Rollup: {csv_out}")

    # Health chip over latest coverage.jsonl (canonical for the chip).
    # tail=64 ⇒ request window; hp["tail"] is the actual window length used.
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
            # 1) Prefer live mean_contradiction_rate from coverage.jsonl (hp).
            v = hp.get("mean_contradiction_rate")
            if v is not None:
                return _fmt(v)
            # 2) If we have a rollup ALL row, fall back to its mean_ctr_rate (or legacy mean_contradiction_rate).
            if all_row:
                raw = all_row.get("mean_ctr_rate") or all_row.get("mean_contradiction_rate")
                if raw not in (None, "", "NA"):
                    try:
                        return _fmt(float(raw))
                    except Exception:
                        pass
            # 3) Last resort: unknown/NA
            return "—"

        st.markdown(
            f"**C1 Health** {emoji} {label} · tail={hp['tail']} · "
            f"sel={_fmt(hp.get('mean_sel_mismatch_rate'))} · "
            f"off={_fmt(hp.get('mean_offrow_mismatch_rate'))} · "
            f"ker={_fmt(hp.get('mean_ker_mismatch_rate'))} · "
            f"ctr={_ctr_display()}"
        )

        # Time(τ) health from coverage_rollup.csv (ALL row, if present).
        # Populated by _coverage_rollup_write_csv(...) from τ C4 health pings.
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

            # Detect whether any τ columns are present at all (non-empty / non-zero).
            has_tau_cols = any(
                str(all_row.get(k, "")).strip() not in ("", "0")
                for k in [
                    "time_tau_n_fixtures_with_c3",
                    "time_tau_tau_pred_true",
                    "time_tau_tau_pred_false",
                    "time_tau_tau_emp_true",
                    "time_tau_tau_emp_false",
                    "time_tau_tau_mismatch_count",
                ]
            )

            if n_c3 > 0:
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
            elif has_tau_cols:
                # Edge case: τ columns present but n_c3 == 0
                st.info("**Time(τ) health** N/A — τ ping present but rollup reports n_c3=0; check C4 artifacts.")
            else:
                # Normal case when no τ ping for this snapshot (or rollup had no τ data).
                st.info("**Time(τ) health** N/A — no τ health ping for this snapshot yet.")
# Optional: surface last ALL row from coverage_rollup.csv (if present), read-only
        if all_row:
            passes_csv = all_row.get("passes_for_snapshot")
            n_fix_csv = all_row.get("n_fixtures")
            extra = ""
            if passes_csv and n_fix_csv:
                extra = f" · passes={passes_csv} · n_fixtures={n_fix_csv}"

            st.caption(
                "Last rollup (ALL) · "
                f"count={all_row.get('count')} · "
                f"sel={all_row.get('mean_sel_mismatch_rate') or '—'} · "
                f"off={all_row.get('mean_offrow_mismatch_rate') or '—'} · "
                f"ker={all_row.get('mean_ker_mismatch_rate') or '—'} · "
                f"ctr={all_row.get('mean_ctr_rate') or '—'}"
                f"{extra}"
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
# LAB-UI: experimental τ local flip toy (read-only, no pipeline writes)
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
                    d_tag = _Path(pB).stem
                    h_tag = _Path(pH).stem
                    c_tag = _Path(pC).stem
                    fixture_label = f"{d_tag}_{h_tag}_{c_tag}"

                    core0 = time_tau_strict_core_from_blocks(bB, bC, bH)
                    core_repr = {
                        "d3": core0["d3"],
                        "C3": core0["C3"],
                        "H2": core0["H2"],
                    }
                    # v2: strict_sig8 via canonical hash_json_sig8(core_repr)
                    strict_sig8 = hash_json_sig8(core_repr)


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
    # Prefer the current *run* snapshot pointer; if unavailable, fall back
    # to any snapshot_id present on the C4 rollup rows.
    snapshot_id = None
    try:
        snapshot_id = _svr_current_run_snapshot_id()
    except Exception:
        snapshot_id = None

    if not snapshot_id:
        for r in rows or []:
            sid = r.get("snapshot_id")
            if sid:
                snapshot_id = sid
                break

    # Always include a snapshot_id field on the ping (may be None).
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


# SSOT: canonical C4 rollup over C3 receipts → rollup CSV/JSONL + τ coverage ping
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

    # Snapshot selection (deterministic priority):
    #   1) manifest row snapshot_id (rec["snapshot_id"])
    #   2) strict_cert["snapshot_id"] or strict_cert["embed"]["snapshot_id"]
    #   3) _svr_current_run_snapshot_id()
    #   4) None
    snapshot_id = None
    try:
        if isinstance(rec, dict):
            snapshot_id = rec.get("snapshot_id") or None
    except Exception:
        snapshot_id = None

    if not snapshot_id and isinstance(strict_cert, dict):
        snapshot_id = (
            strict_cert.get("snapshot_id")
            or (strict_cert.get("embed") or {}).get("snapshot_id")
        )

    if not snapshot_id:
        try:
            snapshot_id = _svr_current_run_snapshot_id()
        except Exception:
            snapshot_id = None

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
# --- v2 time-τ C2/C3/C4 (DO NOT PRUNE) ---

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

# ----------------------------------------------------------------------
# C3 v0.1 — Pass B2: manifest row lookup (no IO writes, no solver)
# ----------------------------------------------------------------------
def _time_tau_c3_find_manifest_row_for_fixture(fixture_label: str) -> dict | None:
    """Locate the manifest row for a given fixture_label.

    We reuse the same manifest file selection as the v2 runner:
    _svr_current_run_manifest_path() → current run_ctx manifest or
    logs/manifests/manifest_full_scope.jsonl.

    Returns the matching manifest row (as a dict) or None if not found or
    if the manifest file is missing/unreadable.
    """
    # Resolve the current v2 manifest path.
    manifest_path = _svr_current_run_manifest_path()
    if not manifest_path or not manifest_path.exists():
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
        # Any IO failure is treated as "not found" in v0.2.
        return None

    return None


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


# SSOT: canonical builder for time_tau_c3_manifest_full_scope.jsonl from v2 manifest + C2 sweep
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
    if manifest_v2_path:
        manifest_v2 = _Path(manifest_v2_path)
    else:
        manifest_v2 = _svr_current_run_manifest_path()

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

    for rec in iter_v2_suite_rows(manifest_v2):
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


# SSOT: canonical C3 sweep driver over time_tau_c3_manifest_full_scope.jsonl
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
# SSOT-UI: τ C3 recompute check (v0.2) wired to v2 manifest + τ artifacts
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
# SSOT-UI: τ C4 stability rollup (v0.2) + coverage ping
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