import csv as _csv
import hashlib
import hashlib as _hash
import hashlib as _hashlib
import hashlib as _Vhash
import importlib.util
import json
import json as _json
import json as _Vjson
import json as _j
import os
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
from pathlib import Path
from pathlib import Path as _Path
from pathlib import Path as _Ph
from pathlib import Path as _VPath
# Third-party
import streamlit as st
import streamlit as _st

# -----------------------------------------------------------------------------
# Phase 0: Authority Map (semantic load-bearing; compression-safe)
#
# This block is intentionally small and *central*: it is the "do not drift"
# reference for what is allowed to decide meaning vs what is only observational.
#
# Authoritative inputs (may decide resolution/meaning):
#   - SSOT snapshot_id: the value returned by v2 SSOT helpers and enforced by
#     build_bundle_manifest_for_snapshot(...). Any caller-provided snapshot_id is
#     treated as a GUARD only.
#   - On-disk canonical neighborhoods under the repo root:
#       * logs/certs/{district_id}/{fixture_label}/{sig8}/  (v2 cert neighborhoods)
#       * logs/certs/d4/                                   (D4 certificates)
#       * logs/certs/parity/                                (S3 parity artifacts)
#   - Bundle-resident meta/bundle_manifest.json *after* B1 materialization,
#     specifically meta.t3_policy_profile_id (or legacy alias policy_profile_id)
#     which binds Track III policy and prevents UI/runtime toggle drift.
#
# Observational inputs (may be printed/carried but MUST NOT drive resolution):
#   - UI widgets / Streamlit state (including profile dropdowns), timestamps,
#     run_label decorations, "latest" heuristics, etc.
#   - snapshot_id *inside* Track III certificate builders: it is carried for
#     readability only and MUST NOT be used to scan/select artifacts.
#
# Resolver role partition:
#   - Read-only resolvers (MUST NOT mkdir / heal state):
#       _d4_cert_root_dir_readonly, _parity_cert_root_dir_readonly,
#       _b1_collect_bundle_state, _d4_resolve_certificate_for_snapshot,
#       _parity_resolve_pair_for_snapshot, _parity_try_resolve_pair_for_snapshot
#   - Writers (allowed to mkdir / generate artifacts):
#       write_d4_certificate_for_snapshot, _b1_write_bundle_tree_for_snapshot,
#       _b1_materialize_bundle_tree, b6_write_seal, etc.
# -----------------------------------------------------------------------------
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
    
def _repo_root():
    try:
        return _REPO_DIR
    except Exception:
        return _Path(__file__).resolve().parents[1]

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
    """Append one JSON line to coverage.jsonl (best-effort).

    Time(τ) coverage pings (unit="time_tau", kind="tau_c4_health") are
    serialized via canonical_json(...) so their hashing surface is stable
    and independent of ephemeral keys like ts_utc. Legacy/other coverage
    rows keep their existing JSON serialization.
    """
    import json as _json, time as _time
    rec = dict(row or {})
    rec.setdefault("ts_utc", int(_time.time()))
    try:
        if _coverage_row_is_tau_ping(rec):
            line = canonical_json(rec)
        else:
            line = _json.dumps(rec, separators=(",", ":"), sort_keys=False)
    except Exception:
        # Best-effort: fall back to plain JSON on any canonicalization error.
        line = _json.dumps(rec, separators=(",", ":"), sort_keys=False)
    with _v2_coverage_path().open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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
    # D3/F1 — Bootstrap gating: ignore coverage rows tagged as coming
    # from the v2 bootstrap 64× run so that C1/D3 counts only the real
    # suite passes. Historical rows without run_label are unaffected.
    run_label = rec.get("run_label")
    if run_label == "v2_bootstrap_64":
        return False
    fid = rec.get("fixture_label")
    if not fid:
        return False
    return True


def _v2_coverage_count_for_snapshot(snapshot_id: str) -> int:
    """Count parseable fixture rows matching a snapshot_id (best-effort).

    Requires a non-empty snapshot_id; returns 0 immediately if snapshot_id is falsy.
    """
    import json as _json

    if not snapshot_id:
        return 0

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
      - when snapshot_id is truthy: have snapshot_id equal to that value;
      - when snapshot_id is falsy: no snapshot filter is applied.

    Returns None when the manifest is missing or no rows match.
    """
    import json as _json

    try:
        if manifest_path is None:
            manifest_path = _svr_current_run_manifest_path()
    except Exception:
        manifest_path = None

    if not manifest_path:
        return None

    try:
        n = 0
        with _Path(manifest_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                sid = rec.get("snapshot_id")

                if snapshot_id:
                    # When filtering by snapshot, require an explicit match.
                    if sid and sid != snapshot_id:
                        continue
                # else: no snapshot filter (accept all snapshot_id values).

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

    If snapshot_id is falsy, default to the canonical v2 world snapshot (SSOT).
    """
    # Resolve snapshot to SSOT by default.
    if not snapshot_id:
        try:
            snapshot_id = _v2_current_world_snapshot_id(strict=False)
        except Exception:
            snapshot_id = None

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


def _c1_debug_snapshot_summary(snapshot_id: str | None) -> dict:
    """
    Dev-only helper to sanity-check C1/τ for a given snapshot_id.

    If snapshot_id is falsy, defaults to the canonical v2 world snapshot (SSOT).

    Returns a dict with:
      - snapshot_id
      - coverage_rows_for_snapshot  (from coverage.jsonl)
      - rollup_all_row              (dict or None)
      - n_fixtures                  (from manifest; None when unknown)
      - passes_for_snapshot         (integer passes; None when unknown)
      - has_integer_passes          (True/False when n_fixtures known; else None)
    """
    cov_path, csv_out = _c1_paths()

    if not snapshot_id:
        try:
            snapshot_id = _v2_current_world_snapshot_id(strict=False)
        except Exception:
            snapshot_id = None

    stats = _d3_integer_pass_stats(snapshot_id)

    all_row = None
    try:
        if csv_out.exists():
            import csv as _csv
            with csv_out.open("r", encoding="utf-8") as f:
                r = list(_csv.DictReader(f))
            for row in r:
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


# --- D4 helpers: digests for world snapshot (D4.B1) ---


def _d4_world_snapshot_digest(snapshot_id: str | None = None) -> dict:
    """Return a compact digest describing the world snapshot for D4.

    Shape:
      {
        "world_snapshot_path": <repo-rel path>,
        "world_snapshot_sig8": <sig8 over snapshot JSON>,
      }
    """
    sid = snapshot_id or _v2_current_world_snapshot_id(strict=True)
    path = _svr_world_snapshot_path(str(sid))

    # Repo-relative path ("logs/..." when possible).
    try:
        rel = _bundle_repo_relative_path(path)
    except Exception:
        # Fallback: best-effort relative string.
        try:
            rel = str(path.relative_to(_REPO_ROOT))
        except Exception:
            rel = str(path)

    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
        sig8 = hash_json_sig8(data)
    except Exception:
        sig8 = ""

    return {
        "world_snapshot_path": rel,
        "world_snapshot_sig8": sig8,
    }


# --- D4 helpers: manifest digest (D4.B2) ---


def _d4_manifest_digest(
    snapshot_id: str | None = None,
    manifest_path: _Path | None = None,
) -> dict:
    """Return a compact digest for the v2 manifest_full_scope.jsonl.

    Shape:
      {
        "manifest_path": <repo-rel path>,
        "manifest_sig8": <sig8 over file bytes>,
        "fixtures_set_sig8": <sig8 over stable fixture triplets>,
        "n_fixtures": int,
        "fixtures": [
          {"fixture_label": ..., "district_id": ..., "strict_sig8": ...},
          ...
        ],
      }

    NOTE (portability/identity): This digest intentionally ignores any per-row path-like fields
    (e.g. 'bundle_dir' or nested 'paths') that may be host-local. D4 binds suite identity via the
    stable triplet set {fixture_label, district_id, strict_sig8} for the snapshot_id, not via
    dereferencing row paths.
    """
    sid = snapshot_id or _v2_current_world_snapshot_id(strict=True)
    mpath = manifest_path or _svr_current_run_manifest_path()
    mpath = _Path(mpath)

    try:
        rel = _bundle_repo_relative_path(mpath)
    except Exception:
        try:
            rel = str(mpath.relative_to(_REPO_ROOT))
        except Exception:
            rel = str(mpath)

    sig8 = _hash_file_sig8(mpath)

    fixtures: list[dict] = []
    try:
        with mpath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except Exception:
                    continue
                if str(row.get("snapshot_id") or "") != str(sid):
                    continue
                fixtures.append(
                    {
                        "fixture_label": row.get("fixture_label"),
                        "district_id": row.get("district_id"),
                        "strict_sig8": row.get("strict_sig8"),
                    }
                )
    except Exception:
        fixtures = []

    try:
        fixtures_set_sig8 = hash_json_sig8(
            sorted(
                fixtures,
                key=lambda r: (
                    str((r or {}).get("fixture_label") or ""),
                    str((r or {}).get("district_id") or ""),
                    str((r or {}).get("strict_sig8") or ""),
                ),
            )
        )
    except Exception:
        fixtures_set_sig8 = ""

    return {
        "manifest_path": rel,
        "manifest_sig8": sig8,
        "fixtures_set_sig8": fixtures_set_sig8,
        "n_fixtures": len(fixtures),
        "fixtures": fixtures,
    }


def _d4_b5_digest(
    snapshot_id: str | None = None,
    manifest_path: _Path | None = None,
) -> dict:
    """Return a compact B5 semantic-identity digest for this snapshot.

    B5 digest is computed from manifest_full_scope.jsonl by hashing the
    strict-core semantic object for each fixture and then hashing the sorted
    set of per-fixture FP digests.
    """
    sid = snapshot_id or _v2_current_world_snapshot_id(strict=True)
    mpath = manifest_path or _svr_current_run_manifest_path()
    try:
        idx = _b5_compute_index_from_manifest(mpath, snapshot_id=str(sid), strict=True)
    except Exception as exc:
        # D4 prefers to be explicit: surface failure rather than silently omitting.
        return {
            "status": "FAIL",
            "reason": str(exc),
        }

    members = idx.get("members") if isinstance(idx, dict) else None
    n_members = len(members) if isinstance(members, list) else 0

    return {
        "status": "OK",
        "b5_set_hex": idx.get("b5_set_hex"),
        "b5_set_sig8": idx.get("b5_set_sig8"),
        "n_members": n_members,
    }




# --- D4 helpers: coverage & Time(τ) digests (D4.B3) ---


def _d4_coverage_rollup_for_snapshot(snapshot_id: str | None = None) -> dict:
    """Build a D4 coverage digest for a given snapshot_id.

    This is analyzer/export-only: it ensures the C1 rollup is up-to-date for
    the snapshot, then returns a compact summary based on the ALL row plus
    integer-pass stats.
    """
    sid = snapshot_id or _v2_current_world_snapshot_id(strict=True)

    # Ensure the C1 rollup CSV is up-to-date for this snapshot.
    try:
        _coverage_rollup_write_csv(snapshot_id=sid)
    except Exception:
        # Analyzer-only helper: never raise here; downstream can still see
        # that coverage and/or integer-pass stats are missing.
        pass

    # Locate coverage + rollup using the existing C1 helper.
    cov_path, rollup_path = _c1_paths()

    rollup_all: dict | None = None
    if rollup_path.exists():
        try:
            import csv as _csv  # local import to avoid global coupling

            with rollup_path.open("r", encoding="utf-8", newline="") as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    if (row.get("prox_label") or "") == "ALL":
                        rollup_all = dict(row)
                        break
        except Exception:
            rollup_all = None

    # Integer-pass stats for this snapshot (D3 spine helper).
    stats = None
    try:
        stats = _d3_integer_pass_stats(sid)
    except Exception:
        stats = None

    cov_rel = _bundle_repo_relative_path(cov_path)
    rollup_rel = _bundle_repo_relative_path(rollup_path)

    return {
        "coverage_jsonl_path": cov_rel,
        "coverage_rollup_csv_path": rollup_rel,
        "snapshot_rows": stats.get("coverage_rows_for_snapshot") if stats else None,
        "passes_for_snapshot": stats.get("passes_for_snapshot") if stats else None,
        "has_integer_passes": stats.get("has_integer_passes") if stats else None,
        "rollup_all": rollup_all,
    }


def _d4_time_tau_digest(snapshot_id: str | None = None) -> dict | None:
    """Return a compact Time(τ) digest for D4, if available.

    For now this reuses the τ summary baked into the C1 ALL row and
    attaches the canonical C4 rollup JSONL path.
    """
    sid = snapshot_id or _v2_current_world_snapshot_id(strict=True)

    cov_summary = _d4_coverage_rollup_for_snapshot(sid)
    rollup_all = cov_summary.get("rollup_all") or {}

    # Locate the canonical C4 rollup JSONL path under logs/reports.
    try:
        root = _repo_root()
    except Exception:
        root = _Path(__file__).resolve().parents[1]
    rep_dir = root / "logs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    try:
        rollup_jsonl_name = TIME_TAU_C3_ROLLUP_JSONL  # type: ignore[name-defined]
    except Exception:
        rollup_jsonl_name = "time_tau_c3_rollup.jsonl"

    c4_path = rep_dir / rollup_jsonl_name
    if c4_path.exists():
        c4_rel = _bundle_repo_relative_path(c4_path)
    else:
        c4_rel = None

    return {
        "c4_rollup_path": c4_rel,
        "time_tau_n_fixtures_with_c3": rollup_all.get("time_tau_n_fixtures_with_c3"),
        "time_tau_tau_pred_true": rollup_all.get("time_tau_tau_pred_true"),
        "time_tau_tau_pred_false": rollup_all.get("time_tau_tau_pred_false"),
        "time_tau_tau_emp_true": rollup_all.get("time_tau_tau_emp_true"),
        "time_tau_tau_emp_false": rollup_all.get("time_tau_tau_emp_false"),
        "time_tau_tau_mismatch_count": rollup_all.get("time_tau_tau_mismatch_count"),
    }




# --- D4 certificate builder (D4.D) ---


def build_d4_certificate_for_snapshot(
    snapshot_id: str | None = None,
    run_ctx: dict | None = None,
) -> dict:
    """Pure builder for the D4 certificate object (pre-hash, no I/O)."""
    # Resolve snapshot_id with a clear SSOT order.
    sid: str | None = None
    if snapshot_id is not None:
        sid = str(snapshot_id)
    elif run_ctx is not None and isinstance(run_ctx, dict):
        sid = str(run_ctx.get("snapshot_id") or "")
    if not sid:
        sid = _v2_current_world_snapshot_id(strict=True)
    # Resolve engine_rev (prefer run_ctx if supplied).
    engine_rev: str = "rev-UNSET"
    if run_ctx is not None and isinstance(run_ctx, dict):
        er = run_ctx.get("engine_rev")
        if isinstance(er, str) and er:
            engine_rev = er
    if engine_rev == "rev-UNSET":
        try:
            engine_rev = ENGINE_REV  # type: ignore[name-defined]
        except Exception:
            engine_rev = "rev-UNSET"

    # Resolve the v2 suite manifest path (prefer run_ctx if supplied).
    mpath = None
    try:
        if isinstance(run_ctx, dict):
            mp = run_ctx.get("manifest_full_scope_path") or run_ctx.get("manifest_path")
            if mp:
                mpath = _Path(str(mp))
    except Exception:
        mpath = None

    # Compose the D4 evidence blocks.
    world = _d4_world_snapshot_digest(sid)
    suite = _d4_manifest_digest(sid, manifest_path=mpath)
    b5 = _d4_b5_digest(sid, manifest_path=mpath)
    coverage = _d4_coverage_rollup_for_snapshot(sid)
    time_tau = _d4_time_tau_digest(sid)

    # Bundle manifest digest: compute the *expected* bundle_manifest sig8
    # deterministically (no scanning). The actual bundle is built after D4.
    bundle = None
    try:
        bm = build_bundle_manifest_for_snapshot(snapshot_id=str(sid), run_ctx=run_ctx)
        bm = stamp_bundle_manifest_sig8(bm)
        bm_sig8 = str(bm.get("sig8") or "")
        if bm_sig8:
            bundle = {
                "bundle_manifest_sig8": bm_sig8,
                "bundle_dir": f"logs/bundle/{sid}__{bm_sig8}",
                "bundle_manifest_path": f"logs/bundle/{sid}__{bm_sig8}/meta/bundle_manifest.json",
            }
    except Exception:
        bundle = None

    meta = {
        "created_at_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "notes": None,
    }

    return {
        "schema": D4_CERTIFICATE_SCHEMA,
        "schema_version": D4_CERTIFICATE_SCHEMA_VERSION,
        "engine_rev": engine_rev,
        "snapshot_id": sid,
        "sig8": "",  # filled in by the writer/stamper
        "world": world,
        "suite": suite,
        "b5": b5,
        "coverage": coverage,
        "time_tau": time_tau,
        "bundle": bundle,
        "meta": meta,
    }




# --- D4 certificate writer and sig8 stamper (D4.E1) ---


def _d4_stamp_sig8(cert: dict) -> dict:
    """Pure helper: compute and stamp sig8 for a D4 certificate.

    We hash the certificate's canonical JSON representation with `sig8` cleared,
    mirroring the bundle-manifest stamp helper.
    """
    base = dict(cert or {})

    # Ensure schema fields are present for stability.
    base.setdefault("schema", D4_CERTIFICATE_SCHEMA)
    base.setdefault("schema_version", D4_CERTIFICATE_SCHEMA_VERSION)

    # Exclude sig8 itself from the hash.
    tmp = dict(base)
    tmp["sig8"] = ""

    sig8 = hash_json_sig8(tmp)
    base["sig8"] = sig8
    return base


def write_d4_certificate_for_snapshot(
    snapshot_id: str | None = None,
    run_ctx: dict | None = None,
) -> _Path:
    """Build + sig8 + write the D4 certificate for a given snapshot.

    This is the small I/O wrapper around `build_d4_certificate_for_snapshot`.
    It returns the path to the written JSON file.
    """
    cert = build_d4_certificate_for_snapshot(snapshot_id=snapshot_id, run_ctx=run_ctx)
    cert = _d4_stamp_sig8(cert)

    sid = str(cert.get("snapshot_id") or snapshot_id or _v2_current_world_snapshot_id(strict=True))
    sig8 = str(cert.get("sig8") or "")

    # Resolve output path under logs/certs/d4/.
    out_path = _d4_cert_path(sid, sig8)

    txt = canonical_json(cert)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8")

    # --- A1: canonicalize D4 certificate selection for downstream bundle export ---
    # If a run_ctx is provided, record the exact certificate path + sig8.
    # This eliminates any need for mtime/lexicographic scanning when building
    # a B1 bundle: the exporter can explicitly reference the intended D4 cert.
    try:
        if isinstance(run_ctx, dict):
            run_ctx["d4_cert_path"] = str(out_path)
            run_ctx["d4_cert_sig8"] = str(sig8)
            run_ctx["d4_cert_snapshot_id"] = str(sid)
    except Exception:
        # Best-effort only; callers may pass non-mutable contexts.
        pass
    return out_path




# --- D4 certificate verifier (D4.E2) ---


def verify_d4_certificate(path: _Path) -> tuple[bool, str]:
    """Minimal local verifier for a D4 certificate JSON file.

    Checks:
      - file is readable JSON
      - sig8 matches the canonical hash
      - snapshot_id is at least self-consistent with the current SSOT
        (reported as an informational message, not a hard failure).
    """
    p = _Path(path)
    if not p.exists():
        return False, f"D4 cert path does not exist: {p}"

    try:
        raw = p.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read D4 cert: {e!r}"

    try:
        obj = _json.loads(raw)
    except Exception as e:
        return False, f"Failed to parse D4 cert JSON: {e!r}"

    got_sig8 = str(obj.get("sig8") or "")
    tmp = dict(obj)
    tmp["sig8"] = ""
    try:
        expected_sig8 = hash_json_sig8(tmp)
    except Exception as e:
        return False, f"Failed to recompute sig8: {e!r}"

    if got_sig8 != expected_sig8:
        return False, f"sig8 mismatch (got={got_sig8}, expected={expected_sig8})"

    sid = obj.get("snapshot_id")
    if not sid:
        return True, "sig8 ok; snapshot_id missing"

    try:
        ssot = _v2_current_world_snapshot_id(strict=False)
    except Exception:
        ssot = None

    if ssot is not None and str(ssot) != str(sid):
        return True, f"sig8 ok; snapshot_id={sid}, current={ssot}"

    return True, "ok"



# --- D4: verify a sealed bundle against a D4 certificate (CorePolicy.v0.1) ---

D4_VERIFY_RECEIPT_SCHEMA = "d4_verify_receipt.v0"
D4_VERIFY_RECEIPT_SCHEMA_VERSION = "0.1.0"
D4_CORE_POLICY_ID = "D4.CorePolicy.v0.1"

# Frozen check vocabulary for CorePolicy.v0.1 (D4-1 lock).
D4_CHECK_IDS_ORDERED_V0_1 = [
    "B6_SEAL_VERDICT",
    "BUNDLE_MANIFEST_READABLE",
    "D4_CERT_PRESENT",
    "D4_CERT_READABLE",
    "D4_CERT_SIG8",
    "SNAPSHOT_ID_MATCH",
    "BUNDLE_MANIFEST_SIG8_MATCH",
    "WORLD_SNAPSHOT_SIG8_MATCH",
    "B5_SET_HEX_MATCH",
    "SUITE_FIXTURES_SET_MATCH",
    "PARITY_CERT_VERIFIED",
]

# Required checks for BOOLEAN verdict (CorePolicy.v0.1).
# NOTE: BUNDLE_MANIFEST_READABLE is a gating precondition; it is not part of the
# boolean reduce-set under the current pipeline contract.
D4_REQUIRED_IDS_V0_1 = {
    "B6_SEAL_VERDICT",
    "D4_CERT_PRESENT",
    "D4_CERT_READABLE",
    "D4_CERT_SIG8",
    "SNAPSHOT_ID_MATCH",
    "BUNDLE_MANIFEST_SIG8_MATCH",
    "WORLD_SNAPSHOT_SIG8_MATCH",
    "B5_SET_HEX_MATCH",
    "SUITE_FIXTURES_SET_MATCH",
}

# D4 receipt hashing: non-core keys (D4-4 lock). These are ignored for payload
# identity when stamping payload_sig8/payload_sha256.
_D4_NON_CORE_KEYS_ALWAYS_V0_1 = {"created_at_utc"}
_D4_NON_CORE_KEYS_EXTRA_V0_1 = {"detail", "evidence", "reason_detail_code"}


def _d4_now_utc_iso() -> str:
    """Best-effort UTC timestamp for receipts."""
    try:
        return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.utcnow().isoformat() + "Z"


def _d4_check(check_id: str, ok: bool | None, *, detail: str = "", evidence: dict | None = None) -> dict:
    """Return a compact binary check record."""
    return {
        "check_id": str(check_id),
        "ok": (bool(ok) if ok is not None else None),
        "detail": str(detail or ""),
        "evidence": (evidence or {}),
    }



def _d4_relpath(bundle_root: _Path, p: _Path | str) -> str:
    """Return bundle-relative POSIX path for evidence fields.

    Absolute paths are suppressed (returning an empty string) because receipts are
    intended to be portable and must not leak host-local filesystem structure.
    """
    try:
        root = _Path(bundle_root).resolve()
        pp = _Path(p).resolve()
        rel = pp.relative_to(root)
        return rel.as_posix()
    except Exception:
        s = str(p)
        # If it looks absolute (POSIX or Windows-ish), suppress it.
        if s.startswith("/") or (len(s) >= 2 and s[1] == ":"):
            return ""
        return s.replace("\\", "/")




def _d4_io_detail(exc: Exception, rel_path: str = "") -> str:
    """Return a portable detail string for IO-ish exceptions (no absolute paths)."""
    try:
        cls = exc.__class__.__name__
    except Exception:
        cls = "Exception"
    try:
        eno = getattr(exc, "errno", None)
    except Exception:
        eno = None
    suffix = str(rel_path or "").strip()
    if eno is not None:
        return f"{cls}[errno={eno}] {suffix}".strip()
    return f"{cls} {suffix}".strip()

def _d4_non_core_keys_v0_1() -> set[str]:
    """Return the frozen non-core key set for D4 receipt stamping (v0.1)."""
    keys: set[str] = set()
    try:
        # Prefer B3's common set (keeps 'paths' by default, but D4 excludes evidence anyway).
        keys |= set(_B3_NON_CORE_COMMON_KEYS)
    except Exception:
        try:
            keys |= set(_V2_EPHEMERAL_KEYS)
        except Exception:
            pass
    keys |= set(_D4_NON_CORE_KEYS_ALWAYS_V0_1)
    keys |= set(_D4_NON_CORE_KEYS_EXTRA_V0_1)
    return keys


def _d4_core_policy_record_v0_1() -> dict:
    """Return a minimal closed-world policy record for CorePolicy.v0.1.

    This exists solely so receipts can bind to a deterministic `policy_sig8`.
    """
    return {
        "schema": "d4_policy_record.v0",
        "schema_version": "0.1.0",
        "policy_id": D4_CORE_POLICY_ID,
        "check_ids_ordered": list(D4_CHECK_IDS_ORDERED_V0_1),
        "required_check_ids": sorted(D4_REQUIRED_IDS_V0_1),
        "completion_rule": "TOTAL",
        "halt_rule": "PREFIX_VALUES_RETAIN_SUFFIX_NULL",
        "receipt_schema": D4_VERIFY_RECEIPT_SCHEMA,
        "receipt_schema_version": D4_VERIFY_RECEIPT_SCHEMA_VERSION,
        "receipt_hashing_non_core_keys": sorted(list(_d4_non_core_keys_v0_1())),
        "preconditions": {
            # Option A: missing expected-side claims are PRECOND_INVALID and D4 does not produce a boolean verdict.
            "missing_expected_claim": "PRECOND_INVALID",
            # D4 is gated by B6; a false B6 seal verdict halts D4.
            "b6_gate": True,
        },
    }


_D4_POLICY_SIG8_CACHE: dict[str, str] = {}


def _d4_policy_sig8(policy_id: str) -> str:
    """Return policy_sig8 for a supported D4 policy_id, or empty string."""
    pid = str(policy_id or "")
    if pid in _D4_POLICY_SIG8_CACHE:
        return _D4_POLICY_SIG8_CACHE[pid]
    if pid != D4_CORE_POLICY_ID:
        _D4_POLICY_SIG8_CACHE[pid] = ""
        return ""
    try:
        rec = _d4_core_policy_record_v0_1()
        sig8 = b3_payload_sig8(rec, non_core_keys=set())
    except Exception:
        sig8 = ""
    _D4_POLICY_SIG8_CACHE[pid] = sig8
    return sig8


def _d4_totalize_checks(checks: list[dict] | None, order: list[str]) -> list[dict]:
    """Return a TOTAL check vector in `order`, filling missing entries with ok:null."""
    checks = list(checks or [])
    by_id: dict[str, dict] = {}
    for c in checks:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("check_id") or "")
        if cid and cid not in by_id:
            by_id[cid] = c
    out: list[dict] = []
    for cid in order:
        if cid in by_id:
            out.append(by_id[cid])
        else:
            out.append(_d4_check(cid, None))
    return out


def _d4_null_suffix_from(
    checks: list[dict] | None,
    order: list[str],
    halt_at_check_id: str,
    *,
    inclusive: bool = True,
) -> list[dict]:
    """TOTALize then force ok:null starting at the halt point.

    If `inclusive` is True, the halt check itself becomes ok:null. If False, the
    halt check retains its ok value (if any) and ok:null begins *after* it.
    """
    out = _d4_totalize_checks(checks, order)
    halt_id = str(halt_at_check_id or "")
    k = None
    for i, c in enumerate(out):
        if str((c or {}).get("check_id") or "") == halt_id:
            k = i
            break
    if k is None:
        k = 0
    if not inclusive:
        k = k + 1
    if k < 0:
        k = 0
    for j in range(k, len(out)):
        try:
            out[j]["ok"] = None
        except Exception:
            pass
    return out


def _d4_halt(
    d4: dict,
    *,
    halt_at_check_id: str,
    status: str,
    reason_code: str,
    reason_detail_code: str = "",
    detail: str = "",
    # Control whether the halt check itself is nulled, and whether suffix-null is inclusive.
    null_halt_check: bool = True,
    suffix_inclusive: bool = True,
) -> dict:
    """Halt D4: no boolean verdict, deterministic reason, and suffix ok:null.

    - For *precondition* halts (missing/invalid inputs or expected-side claims),
      use the defaults (null_halt_check=True, suffix_inclusive=True).
    - For a *gating* check that was evaluated and failed (e.g., B6 gate),
      call with null_halt_check=False and suffix_inclusive=False so the halt
      check can remain ok:false while everything after is ok:null.
    """
    d4["status"] = str(status or "NOT_ATTEMPTED")
    d4["verdict"] = None
    d4["reason_code"] = str(reason_code or "INTERNAL_ERROR")
    if reason_detail_code:
        d4["reason_detail_code"] = str(reason_detail_code)
    else:
        d4.pop("reason_detail_code", None)

    hid = str(halt_at_check_id or "")
    # Ensure the halt check exists; optionally force ok:null on it.
    found = False
    for c in d4.get("checks") or []:
        if isinstance(c, dict) and str(c.get("check_id") or "") == hid:
            if null_halt_check:
                c["ok"] = None
            if detail:
                c["detail"] = str(detail)
            found = True
            break
    if not found:
        # If we didn't already record the halt check, use ok:null (conservative).
        d4.setdefault("checks", []).append(_d4_check(hid, None, detail=str(detail or "")))

    # TOTALize + suffix-null (inclusive or exclusive).
    d4["checks"] = _d4_null_suffix_from(
        d4.get("checks") or [],
        order=D4_CHECK_IDS_ORDERED_V0_1,
        halt_at_check_id=hid,
        inclusive=bool(suffix_inclusive),
    )
    return d4


def _d4_finalize_and_stamp(d4: dict) -> dict:
    """Finalize invariants + stamp payload_sig8/payload_sha256 for a D4 receipt."""
    # TOTALize checks in the frozen order.
    d4["checks"] = _d4_totalize_checks(d4.get("checks") or [], D4_CHECK_IDS_ORDERED_V0_1)

    st = str(d4.get("status") or "")
    if st not in B3_STATUS_SET:
        d4["status"] = "ERROR"
        d4["verdict"] = None
        d4["reason_code"] = "INTERNAL_ERROR"
        d4["reason_detail_code"] = "STATUS_INVALID"
        st = "ERROR"

    if st == "OK":
        # In OK state, reason fields must be absent; verdict must be boolean.
        d4.pop("reason_code", None)
        d4.pop("reason_detail_code", None)
        if d4.get("verdict") is None:
            d4["status"] = "ERROR"
            d4["verdict"] = None
            d4["reason_code"] = "INTERNAL_ERROR"
            d4["reason_detail_code"] = "VERDICT_MISSING"
        else:
            d4["verdict"] = bool(d4.get("verdict"))
    else:
        # Non-OK: no boolean verdict; reason_code must be a B3 token.
        d4["verdict"] = None
        rc = str(d4.get("reason_code") or "")
        if rc not in B3_REASON_CODES_SET:
            d4["reason_code"] = "INTERNAL_ERROR"
            if not str(d4.get("reason_detail_code") or ""):
                d4["reason_detail_code"] = "REASON_INVALID"

    # Stamp payload identity (B3).
    try:
        b3_stamp_payload_sig8(d4, non_core_keys=_d4_non_core_keys_v0_1(), include_full_sha256=True)
    except Exception:
        pass

    return d4


# --- Track III: Bundle certificate + Policy Profile (initial foothold, v0.1) ---
#
# Track III invariant: certificates must NOT depend on hidden UI/runtime toggles.
# Instead, the certificate binds to an explicit Policy Profile object that is
# persisted into the artifact.
#
# Track III thinness invariant: the certificate is a pure value-map over
# verifier receipts (B6/D4/Time(τ)). It must not scan the bundle tree, and it
# must not recompute or re-verify any verifier logic.

T3_POLICY_PROFILE_SCHEMA = "t3_policy_profile"
T3_POLICY_PROFILE_SCHEMA_VERSION = "t3_policy_profile.v0.1"

T3_BUNDLE_CERT_SCHEMA = "t3_bundle_certificate"
T3_BUNDLE_CERT_SCHEMA_VERSION = "t3_bundle_certificate.v0.1"

# Meaning-level profile id for the v0.1 policy regime.
T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_1 = "T3.PolicyProfile.pointer_gated_dw.v0.1"

# Meaning-level profile id for the v0.2 policy regime (τ required).
T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2 = "T3.PolicyProfile.pointer_gated_dw.v0.2"

# Meaning-level profile id for the v0.3 policy regime (derived worlds mandatory).
T3_POLICY_PROFILE_ID_MANDATORY_DW_V0_3 = "T3.PolicyProfile.mandatory_dw.v0.3"

# Meaning-level profile id for the v0.4 policy regime (DW pointer-gated + T0 mandatory).
T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_V0_4 = "T3.PolicyProfile.pointer_gated_dw_t0_mandatory.v0.4"

# Meaning-level profile id for the v0.5 policy regime (pointer-gated DW + parity required).
T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_5 = "T3.PolicyProfile.pointer_gated_dw.v0.5"

# Meaning-level profile id for the v0.6 policy regime (pointer-gated DW + T0 mandatory + parity required).
T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_PARITY_V0_6 = "T3.PolicyProfile.pointer_gated_dw_t0_mandatory_parity.v0.6"

# Track III claim ids (frozen surface for the initial foothold).
T3_CLAIM_B6_SEAL_VERDICT = "B6_SEAL_VERDICT"
T3_CLAIM_D4_CORE_POLICY_VERDICT = "D4_CORE_POLICY_VERDICT"
T3_CLAIM_TIME_TAU_VERDICT_OK = "TIME_TAU_VERDICT_OK"
T3_CLAIM_PARITY_CERT_VERIFIED = "PARITY_CERT_VERIFIED"


# Planning-only collapse criteria (no implementation in this session).
T3_DW_COLLAPSE_PLAN_V0_1: dict = {
    "planning_only": True,
    "collapse_trigger_conditions": [
        "Introduce a new Policy Profile with derived_worlds_regime='MANDATORY'.",
        "Time(τ) verifier supports both regimes without scanning (inventory-only).",
        "Certificates persist policy_profile_id so policy drift is explicit.",
    ],
    "migration_shape": [
        "Publish new profile_id and start emitting it in certificates.",
        "Verifier accepts both profiles for a deprecation window.",
        "Only after deprecation consider removing pointer-gated paths (explicitly deferred).",
    ],
}


def _t3_claim_ok_from_b3(receipt_obj) -> bool | None:
    """Tri-state mapping for B3-style receipts.

    - True/False only when status == OK and verdict is boolean.
    - None otherwise (missing, NOT_ATTEMPTED, ERROR, etc.).
    """
    if not isinstance(receipt_obj, dict):
        return None
    if str(receipt_obj.get("status") or "") != "OK":
        return None
    v = receipt_obj.get("verdict")
    if v is True:
        return True
    if v is False:
        return False
    return None


def _t3_claim_ok_from_time_tau(time_tau_obj) -> bool | None:
    """Tri-state mapping for the Time(τ) verifier receipt."""
    if not isinstance(time_tau_obj, dict):
        return None
    v = time_tau_obj.get("verdict_ok")
    if v is True:
        return True
    if v is False:
        return False
    return None


def _t3_find_check_ok(checks_obj, check_id: str) -> bool | None:
    """Find a check row's ok value by check_id in a receipt's checks[] list."""
    if not isinstance(check_id, str) or not check_id:
        return None
    if not isinstance(checks_obj, list):
        return None
    for row in checks_obj:
        if not isinstance(row, dict):
            continue
        if str(row.get("check_id") or "") == check_id:
            ok = row.get("ok")
            if ok is True:
                return True
            if ok is False:
                return False
            return None
    return None



def t3_build_policy_profile(profile_id: str | None = None) -> dict:
    """Build the explicit Track III Policy Profile.

    The policy profile is the explicit claim parameter: it records the
    derived-worlds participation regime and which claims are required.

    Args:
        profile_id: Meaning-level profile id. If None, defaults to the current
            pointer-gated regime (v0.2).

    Supported profile ids:
        - T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_1
        - T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2
        - T3_POLICY_PROFILE_ID_MANDATORY_DW_V0_3
        - T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_V0_4
        - T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_5
        - T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_PARITY_V0_6
    """
    pid = str(profile_id or T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2)

    t0_regime: str | None = None

    if pid == T3_POLICY_PROFILE_ID_MANDATORY_DW_V0_3:
        dw_regime = "MANDATORY"
    elif pid in {
        T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_V0_4,
        T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_PARITY_V0_6,
    }:
        dw_regime = "POINTER_GATED"
        t0_regime = "MANDATORY"
    elif pid in {
        T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_1,
        T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2,
        T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_5,
    }:
        dw_regime = "POINTER_GATED"
    else:
        raise ValueError(f"Unknown T3 policy profile_id: {pid!r}")

    prof = {
        "schema": T3_POLICY_PROFILE_SCHEMA,
        "schema_version": T3_POLICY_PROFILE_SCHEMA_VERSION,
        "profile_id": pid,
        "derived_worlds_regime": dw_regime,
        "dw_activation_pointer_spec": {
            # Canonical activation semantics (role + relpath).
            "role": TIME_TAU_PTR_ROLE_C3_DERIVED_WORLDS_MANIFEST,
            "relpath": TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_REL,
        },
        # Initial foothold: matches current "Current Spec" certification behavior.
        "required_claims": [
            T3_CLAIM_B6_SEAL_VERDICT,
            T3_CLAIM_D4_CORE_POLICY_VERDICT,
            T3_CLAIM_TIME_TAU_VERDICT_OK,
        ],
        "optional_claims": [],
        "dw_collapse_plan": dict(T3_DW_COLLAPSE_PLAN_V0_1),
    }
    if pid in {
        T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_5,
        T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_PARITY_V0_6,
    }:
        # S3: parity claim required (policy v0.5).
        try:
            req = list(prof.get("required_claims") or [])
            if T3_CLAIM_PARITY_CERT_VERIFIED not in req:
                req.append(T3_CLAIM_PARITY_CERT_VERIFIED)
            prof["required_claims"] = req
        except Exception:
            pass

    if t0_regime is not None:
        prof["t0_regime"] = t0_regime
    try:
        prof["sig8"] = hash_json_sig8(prof)
    except Exception:
        pass
    return prof


def t3_build_bundle_certificate_from_receipts(b6_receipt_obj: dict, *, policy_profile: dict | None = None) -> dict:
    """Build a Track III bundle certificate as a pure value-map over receipts."""
    b6r = b6_receipt_obj if isinstance(b6_receipt_obj, dict) else {}
    prof = policy_profile if isinstance(policy_profile, dict) else t3_build_policy_profile()
    profile_id = str(prof.get("profile_id") or "")

    required_claims = list(prof.get("required_claims") or [])
    optional_claims = list(prof.get("optional_claims") or [])

    # Snapshot_id is observational only; do not scan for it.
    snapshot_id = str(b6r.get("snapshot_id") or "").strip()
    if not snapshot_id:
        try:
            tt = b6r.get("time_tau") if isinstance(b6r.get("time_tau"), dict) else {}
            snapshot_id = str(tt.get("snapshot_id") or "").strip()
        except Exception:
            snapshot_id = ""

    d4r = b6r.get("d4") if isinstance(b6r.get("d4"), dict) else {}
    ttr = b6r.get("time_tau") if isinstance(b6r.get("time_tau"), dict) else {}

    b6_ok = _t3_claim_ok_from_b3(b6r)
    d4_ok = _t3_claim_ok_from_b3(d4r)
    tt_ok = _t3_claim_ok_from_time_tau(ttr)

    # Extract derived-worlds observational bits from Time(τ) checks.
    tt_checks = ttr.get("checks") if isinstance(ttr.get("checks"), list) else []
    dw_activated = _t3_find_check_ok(tt_checks, "TAU_DERIVED_WORLDS_ACTIVATED")
    dw_ptr_canonical_ok = _t3_find_check_ok(tt_checks, "TAU_DERIVED_WORLDS_POINTER_CANONICAL")

    # S3: Parity claim is derived from the D4 check row (status-gated).
    d4_checks = d4r.get("checks") if isinstance(d4r.get("checks"), list) else []
    parity_ok = None
    if str(d4r.get("status") or "") == "OK":
        parity_ok = _t3_find_check_ok(d4_checks, T3_CLAIM_PARITY_CERT_VERIFIED)

    claims: dict = {
        T3_CLAIM_B6_SEAL_VERDICT: {
            "claim_id": T3_CLAIM_B6_SEAL_VERDICT,
            "ok": b6_ok,
            "source": "b6_verify_receipt",
        },
        T3_CLAIM_D4_CORE_POLICY_VERDICT: {
            "claim_id": T3_CLAIM_D4_CORE_POLICY_VERDICT,
            "ok": d4_ok,
            "source": "d4_verify_receipt",
        },
        T3_CLAIM_PARITY_CERT_VERIFIED: {
            "claim_id": T3_CLAIM_PARITY_CERT_VERIFIED,
            "ok": parity_ok,
            "source": "d4_verify_receipt",
        },
        T3_CLAIM_TIME_TAU_VERDICT_OK: {
            "claim_id": T3_CLAIM_TIME_TAU_VERDICT_OK,
            "ok": tt_ok,
            "source": "time_tau_verify_receipt",
            "evidence": {
                "closure_set_sig8": str(ttr.get("closure_set_sig8") or "") or None,
                "tau_surface_sig8": str(ttr.get("tau_surface_sig8") or "") or None,
                "derived_worlds_activated": dw_activated,
                "derived_worlds_pointer_canonical_ok": dw_ptr_canonical_ok,
                "required_check_ids": ttr.get("required_check_ids") if isinstance(ttr.get("required_check_ids"), list) else None,
                "required_ok": ttr.get("required_ok") if isinstance(ttr.get("required_ok"), list) else None,
            },
        },
    }

    # Aggregation semantics (tri-state) driven only by the policy profile.
    any_false = False
    any_unknown = False
    for cid in required_claims:
        row = claims.get(cid) if isinstance(claims.get(cid), dict) else None
        ok = (row or {}).get("ok") if isinstance(row, dict) else None
        if ok is False:
            any_false = True
        elif ok is True:
            pass
        else:
            any_unknown = True

    verdict_ok = False if any_false else (None if any_unknown else True)

    cert = {
        "schema": T3_BUNDLE_CERT_SCHEMA,
        "schema_version": T3_BUNDLE_CERT_SCHEMA_VERSION,
        "policy_profile_id": profile_id,
        "policy_profile": prof,
        "snapshot_id": snapshot_id,
        "required_claims": required_claims,
        "optional_claims": optional_claims,
        "claims": claims,
        "verdict_ok": verdict_ok,
    }
    try:
        cert["sig8"] = hash_json_sig8(cert)
    except Exception:
        pass
    return cert


def d4_verify_bundle_dir(
    bundle_dir: _Path,
    *,
    cert_path: _Path | None = None,
    write_receipt: bool = True,
    policy_id: str = D4_CORE_POLICY_ID,
    t3_policy_profile_id: str = T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2,
) -> dict:
    """Verify a bundle using B6 seal + D4 certificate (CorePolicy.v0.1).

    Returns the B6 verify receipt with an embedded `d4` block. If `write_receipt`
    is True, writes meta/b6_verify_receipt.json (excluded from seal).

    Option A semantics (locked for this session):
      - Any missing/invalid required input or expected-side claim => NOT_ATTEMPTED/ERROR,
        verdict=null, deterministic reason_code, TOTAL check vector with suffix ok:null.
      - A boolean verdict is produced only when status == OK.
    """
    bdir = _Path(bundle_dir).resolve()

    # Always compute B6 receipt first (D4 is a thin wrapper).
    b6_receipt = b6_verify_bundle_seal(bdir)

    pid = str(policy_id or D4_CORE_POLICY_ID)

    d4: dict = {
        "schema": D4_VERIFY_RECEIPT_SCHEMA,
        "schema_version": D4_VERIFY_RECEIPT_SCHEMA_VERSION,
        "policy_id": pid,
        "policy_sig8": _d4_policy_sig8(pid),
        "verdict_mode": "BOOLEAN",
        "status": "NOT_ATTEMPTED",
        "verdict": None,
        "created_at_utc": _d4_now_utc_iso(),
        "checks": [],
    }

    def _finish_and_return() -> dict:
        # Finalize + stamp the embedded D4 receipt before returning.
        d4_final = _d4_finalize_and_stamp(d4)
        b6_receipt["d4"] = d4_final


        # Track III: build the explicit Policy Profile first (regime selector).
        # This is part of the certificate surface; do not infer regime from UI toggles.
        t3_policy = t3_build_policy_profile(profile_id=t3_policy_profile_id)

        # Track II (Time(τ)) closure verifier (diagnostic only).
        # This is written into meta/b6_verify_receipt.json which is excluded from the B6 seal domain.
        try:
            b6_receipt["time_tau"] = time_tau_verify_bundle_dir(
                bdir,
                derived_worlds_regime=str(t3_policy.get("derived_worlds_regime") or "POINTER_GATED"),
                t0_regime=(t3_policy.get("t0_regime") if isinstance(t3_policy, dict) else None),
            )
        except Exception as exc:
            b6_receipt["time_tau"] = {
                "schema": TIME_TAU_VERIFY_RECEIPT_SCHEMA,
                "schema_version": TIME_TAU_VERIFY_RECEIPT_SCHEMA_VERSION,
                "status": "ERROR",
                "verdict_ok": None,
                "error": str(exc),
            }

        # Track III: build and embed the explicit Policy Profile + certificate.
        # Pure value-map only; no scanning or re-verification logic here.
        try:
            b6_receipt["t3_certificate"] = t3_build_bundle_certificate_from_receipts(b6_receipt, policy_profile=t3_policy)
        except Exception as exc:
            b6_receipt["t3_certificate"] = {
                "schema": T3_BUNDLE_CERT_SCHEMA,
                "schema_version": T3_BUNDLE_CERT_SCHEMA_VERSION,
                "policy_profile_id": str(t3_policy.get("profile_id") or t3_policy_profile_id),
                "verdict_ok": None,
                "error": str(exc),
            }

        if write_receipt:
            try:
                b6_write_verify_receipt(bdir, b6_receipt)
            except Exception:
                pass
        return b6_receipt

    # Policy binding: only CorePolicy.v0.1 is supported in this verifier.
    if pid != D4_CORE_POLICY_ID:
        _d4_halt(
            d4,
            halt_at_check_id="B6_SEAL_VERDICT",
            status="NOT_ATTEMPTED",
            reason_code="POLICY_DISABLED",
            reason_detail_code="UNSUPPORTED_POLICY",
        )
        return _finish_and_return()

    # --- Check 1: B6 gate ---
    b6_status = str(b6_receipt.get("status") or "")
    b6_verdict = b6_receipt.get("verdict")

    b6_detail = str(b6_receipt.get("fail_codes") or b6_receipt.get("reason_detail_code") or "")
    if b6_status != "OK":
        d4["checks"].append(_d4_check("B6_SEAL_VERDICT", None, detail=b6_detail))
        _d4_halt(
            d4,
            halt_at_check_id="B6_SEAL_VERDICT",
            status="NOT_ATTEMPTED",
            reason_code=("DEP_NOT_ATTEMPTED" if b6_status in ("NOT_ATTEMPTED", "NA") else "DEP_ERROR"),
            reason_detail_code=f"B6_STATUS_{b6_status or 'UNKNOWN'}",
        )
        return _finish_and_return()

    if b6_verdict is not True:
        d4["checks"].append(_d4_check("B6_SEAL_VERDICT", False, detail=b6_detail))
        _d4_halt(
            d4,
            halt_at_check_id="B6_SEAL_VERDICT",
            status="NOT_ATTEMPTED",
            reason_code="DEP_ERROR",
            reason_detail_code="B6_VERDICT_FALSE",
            null_halt_check=False,
            suffix_inclusive=False,
        )
        return _finish_and_return()

    d4["checks"].append(_d4_check("B6_SEAL_VERDICT", True, detail=b6_detail))

    # From this point on, we *attempt* D4 verification.
    d4["status"] = "OK"
    d4.pop("reason_code", None)
    d4.pop("reason_detail_code", None)

    # --- Check 2: bundle_manifest.json readable + object ---
    manifest_rel = "meta/bundle_manifest.json"
    manifest_path = _b1_rel_dirs_for_root(bdir)["meta"] / "bundle_manifest.json"
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        d4["checks"].append(_d4_check("BUNDLE_MANIFEST_READABLE", None, detail=_d4_io_detail(exc, manifest_rel), evidence={"path": manifest_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_MISSING",
            reason_detail_code="MANIFEST_NOT_FOUND",
        )
        return _finish_and_return()
    except Exception as exc:
        d4["checks"].append(_d4_check("BUNDLE_MANIFEST_READABLE", None, detail=_d4_io_detail(exc, manifest_rel), evidence={"path": manifest_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="ERROR",
            reason_code="IO_ERROR",
            reason_detail_code="MANIFEST_READ_FAILED",
        )
        return _finish_and_return()

    try:
        manifest_obj = _json.loads(raw)
    except Exception as exc:
        d4["checks"].append(_d4_check("BUNDLE_MANIFEST_READABLE", None, detail=_d4_io_detail(exc, manifest_rel), evidence={"path": manifest_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="MANIFEST_PARSE_ERROR",
        )
        return _finish_and_return()

    if not isinstance(manifest_obj, dict):
        d4["checks"].append(_d4_check("BUNDLE_MANIFEST_READABLE", None, detail="not an object", evidence={"path": manifest_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="MANIFEST_NOT_OBJECT",
        )
        return _finish_and_return()

    # Manifest is usable. Also resolve any bundle-bound Track III policy profile id to avoid UI drift.
    meta_obj = manifest_obj.get("meta") if isinstance(manifest_obj.get("meta"), dict) else {}

    # S3 policy binding: canonical key and legacy alias.
    a_present = isinstance(meta_obj, dict) and ("t3_policy_profile_id" in meta_obj)
    b_present = isinstance(meta_obj, dict) and ("policy_profile_id" in meta_obj)
    a_val = meta_obj.get("t3_policy_profile_id") if a_present else None
    b_val = meta_obj.get("policy_profile_id") if b_present else None

    # Type validation: if key is present, must be a non-empty string.
    if a_present and not isinstance(a_val, str):
        d4["checks"].append(
            _d4_check("BUNDLE_MANIFEST_READABLE", None, detail="t3_policy_profile_id not a string", evidence={"path": manifest_rel})
        )
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="T3_POLICY_PROFILE_BINDING_INVALID",
        )
        return _finish_and_return()
    if b_present and not isinstance(b_val, str):
        d4["checks"].append(
            _d4_check("BUNDLE_MANIFEST_READABLE", None, detail="policy_profile_id not a string", evidence={"path": manifest_rel})
        )
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="T3_POLICY_PROFILE_BINDING_INVALID",
        )
        return _finish_and_return()

    a = str(a_val or "").strip() if isinstance(a_val, str) else ""
    b = str(b_val or "").strip() if isinstance(b_val, str) else ""
    if a_present and not a:
        d4["checks"].append(
            _d4_check("BUNDLE_MANIFEST_READABLE", None, detail="t3_policy_profile_id empty", evidence={"path": manifest_rel})
        )
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="T3_POLICY_PROFILE_BINDING_EMPTY",
        )
        return _finish_and_return()
    if b_present and not b:
        d4["checks"].append(
            _d4_check("BUNDLE_MANIFEST_READABLE", None, detail="policy_profile_id empty", evidence={"path": manifest_rel})
        )
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="T3_POLICY_PROFILE_BINDING_EMPTY",
        )
        return _finish_and_return()
    if a and b and a != b:
        d4["checks"].append(
            _d4_check("BUNDLE_MANIFEST_READABLE", None, detail="policy binding conflict", evidence={"path": manifest_rel})
        )
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="T3_POLICY_PROFILE_BINDING_CONFLICT",
        )
        return _finish_and_return()
    if a:
        t3_policy_profile_id = a
    elif b:
        t3_policy_profile_id = b

    d4["checks"].append(_d4_check("BUNDLE_MANIFEST_READABLE", True, evidence={"path": manifest_rel}))

    sid = str(manifest_obj.get("snapshot_id") or "")
    m_sig8 = str(manifest_obj.get("sig8") or "")
    d4["snapshot_id"] = sid
    d4["bundle_manifest_sig8"] = m_sig8

    # --- Check 3: D4_CERT_PRESENT ---
    cert_file: _Path | None = None

    if cert_path is not None:
        # Explicit path override: must be bundle-local.
        try:
            cp = _Path(cert_path).resolve()
            # Enforce bundle-local reads to avoid host contamination.
            _ = cp.relative_to(bdir)
            cert_file = cp
        except Exception:
            _d4_halt(
                d4,
                halt_at_check_id="D4_CERT_PRESENT",
                status="NOT_ATTEMPTED",
                reason_code="PRECOND_INVALID",
                reason_detail_code="CERT_PATH_OUTSIDE_BUNDLE",
            )
            return _finish_and_return()
        if cert_file is None or not cert_file.exists():
            _d4_halt(
                d4,
                halt_at_check_id="D4_CERT_PRESENT",
                status="NOT_ATTEMPTED",
                reason_code="PRECOND_MISSING",
                reason_detail_code="D4_CERT_NOT_FOUND",
            )
            return _finish_and_return()
    else:
        # Bundle-local locator via manifest meta: uses sig8 + canonical bundle path.
        if not sid:
            _d4_halt(
                d4,
                halt_at_check_id="D4_CERT_PRESENT",
                status="NOT_ATTEMPTED",
                reason_code="PRECOND_INVALID",
                reason_detail_code="MANIFEST_SNAPSHOT_ID_MISSING",
            )
            return _finish_and_return()

        ref = (manifest_obj.get("meta") or {}).get("d4_certificate") or {}
        if not isinstance(ref, dict):
            _d4_halt(
                d4,
                halt_at_check_id="D4_CERT_PRESENT",
                status="NOT_ATTEMPTED",
                reason_code="PRECOND_MISSING",
                reason_detail_code="D4_CERT_META_MISSING",
            )
            return _finish_and_return()

        ref_sig8 = str(ref.get("sig8") or "")
        if not ref_sig8:
            _d4_halt(
                d4,
                halt_at_check_id="D4_CERT_PRESENT",
                status="NOT_ATTEMPTED",
                reason_code="PRECOND_MISSING",
                reason_detail_code="D4_CERT_META_SIG8_MISSING",
            )
            return _finish_and_return()

        cert_file = _b1_d4_cert_path(bdir, sid, ref_sig8)
        if not cert_file.exists():
            _d4_halt(
                d4,
                halt_at_check_id="D4_CERT_PRESENT",
                status="NOT_ATTEMPTED",
                reason_code="PRECOND_MISSING",
                reason_detail_code="D4_CERT_NOT_FOUND",
            )
            return _finish_and_return()

    # Cert file exists and is bundle-local.
    d4["checks"].append(
        _d4_check("D4_CERT_PRESENT", True, evidence={"path": _d4_relpath(bdir, cert_file) or "certs/d4"})
    )

    # --- Check 4: D4_CERT_READABLE ---
    try:
        cert_raw = _Path(cert_file).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        d4["checks"].append(_d4_check("D4_CERT_READABLE", None, detail=_d4_io_detail(exc, _d4_relpath(bdir, cert_file) or "certs/d4")))
        _d4_halt(
            d4,
            halt_at_check_id="D4_CERT_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_MISSING",
            reason_detail_code="D4_CERT_NOT_FOUND",
        )
        return _finish_and_return()
    except Exception as exc:
        d4["checks"].append(_d4_check("D4_CERT_READABLE", None, detail=_d4_io_detail(exc, _d4_relpath(bdir, cert_file) or "certs/d4")))
        _d4_halt(
            d4,
            halt_at_check_id="D4_CERT_READABLE",
            status="ERROR",
            reason_code="IO_ERROR",
            reason_detail_code="D4_CERT_READ_FAILED",
        )
        return _finish_and_return()

    try:
        cert_obj = _json.loads(cert_raw)
    except Exception as exc:
        d4["checks"].append(_d4_check("D4_CERT_READABLE", None, detail=_d4_io_detail(exc, _d4_relpath(bdir, cert_file) or "certs/d4")))
        _d4_halt(
            d4,
            halt_at_check_id="D4_CERT_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="D4_CERT_PARSE_ERROR",
        )
        return _finish_and_return()

    if not isinstance(cert_obj, dict):
        d4["checks"].append(_d4_check("D4_CERT_READABLE", None, detail="not an object"))
        _d4_halt(
            d4,
            halt_at_check_id="D4_CERT_READABLE",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="D4_CERT_NOT_OBJECT",
        )
        return _finish_and_return()

    d4["checks"].append(_d4_check("D4_CERT_READABLE", True))
    d4["d4_cert_sig8"] = str(cert_obj.get("sig8") or "")

    # --- Check 5: D4_CERT_SIG8 (cert internal hash) ---
    try:
        ok_sig8, msg = verify_d4_certificate(_Path(cert_file))
    except Exception as exc:
        ok_sig8, msg = False, f"verify_d4_certificate raised: {exc!r}"
    d4["checks"].append(_d4_check("D4_CERT_SIG8", ok_sig8, detail=str(msg)))

    # --- Check 6: SNAPSHOT_ID_MATCH (expected-side required under Option A) ---
    if not sid:
        _d4_halt(
            d4,
            halt_at_check_id="SNAPSHOT_ID_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="MANIFEST_SNAPSHOT_ID_MISSING",
        )
        return _finish_and_return()

    cert_sid = str(cert_obj.get("snapshot_id") or "")
    if not cert_sid:
        _d4_halt(
            d4,
            halt_at_check_id="SNAPSHOT_ID_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="EXPECTED_SNAPSHOT_ID_MISSING",
        )
        return _finish_and_return()

    sid_match = bool(sid == cert_sid)
    d4["checks"].append(_d4_check("SNAPSHOT_ID_MATCH", sid_match, detail=f"bundle={sid}, cert={cert_sid}"))

    # --- Check 7: BUNDLE_MANIFEST_SIG8_MATCH (expected-side required) ---
    if not m_sig8:
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_SIG8_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="OBSERVED_BUNDLE_MANIFEST_SIG8_MISSING",
        )
        return _finish_and_return()

    exp_m_sig8 = str(((cert_obj.get("bundle") or {}) or {}).get("bundle_manifest_sig8") or "")
    if not exp_m_sig8:
        _d4_halt(
            d4,
            halt_at_check_id="BUNDLE_MANIFEST_SIG8_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="EXPECTED_BUNDLE_MANIFEST_SIG8_MISSING",
        )
        return _finish_and_return()

    m_match = bool(exp_m_sig8 == m_sig8)
    d4["checks"].append(_d4_check("BUNDLE_MANIFEST_SIG8_MATCH", m_match, detail=f"expected={exp_m_sig8}, observed={m_sig8}"))

    # --- Check 8: WORLD_SNAPSHOT_SIG8_MATCH (required artifact + expected-side required) ---
    exp_ws_sig8 = str(((cert_obj.get("world") or {}) or {}).get("world_snapshot_sig8") or "")
    if not exp_ws_sig8:
        _d4_halt(
            d4,
            halt_at_check_id="WORLD_SNAPSHOT_SIG8_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="EXPECTED_WORLD_SNAPSHOT_SIG8_MISSING",
        )
        return _finish_and_return()

    ws_rel = "world/world_snapshot.v2.json"
    ws_path = _b1_world_snapshot_path(bdir)
    try:
        ws_raw = ws_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        d4["checks"].append(_d4_check("WORLD_SNAPSHOT_SIG8_MATCH", None, detail=_d4_io_detail(exc, ws_rel), evidence={"path": ws_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="WORLD_SNAPSHOT_SIG8_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_MISSING",
            reason_detail_code="WORLD_SNAPSHOT_NOT_FOUND",
        )
        return _finish_and_return()
    except Exception as exc:
        d4["checks"].append(_d4_check("WORLD_SNAPSHOT_SIG8_MATCH", None, detail=_d4_io_detail(exc, ws_rel), evidence={"path": ws_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="WORLD_SNAPSHOT_SIG8_MATCH",
            status="ERROR",
            reason_code="IO_ERROR",
            reason_detail_code="WORLD_SNAPSHOT_READ_FAILED",
        )
        return _finish_and_return()

    try:
        ws_data = _json.loads(ws_raw)
    except Exception as exc:
        d4["checks"].append(_d4_check("WORLD_SNAPSHOT_SIG8_MATCH", None, detail=_d4_io_detail(exc, ws_rel), evidence={"path": ws_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="WORLD_SNAPSHOT_SIG8_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="WORLD_SNAPSHOT_PARSE_ERROR",
        )
        return _finish_and_return()

    try:
        ws_sig8 = hash_json_sig8(ws_data)
    except Exception as exc:
        d4["checks"].append(_d4_check("WORLD_SNAPSHOT_SIG8_MATCH", None, detail=_d4_io_detail(exc, ws_rel), evidence={"path": ws_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="WORLD_SNAPSHOT_SIG8_MATCH",
            status="ERROR",
            reason_code="INTERNAL_ERROR",
            reason_detail_code="WORLD_SNAPSHOT_SIG8_RECOMPUTE_FAILED",
        )
        return _finish_and_return()

    ws_ok = bool(ws_sig8 == exp_ws_sig8)
    d4["checks"].append(
        _d4_check(
            "WORLD_SNAPSHOT_SIG8_MATCH",
            ws_ok,
            detail=f"expected={exp_ws_sig8}, observed={ws_sig8}",
            evidence={"path": ws_rel},
        )
    )

    # --- Check 9: B5_SET_HEX_MATCH (required artifact + expected-side required) ---
    exp_b5 = str(((cert_obj.get("b5") or {}) or {}).get("b5_set_hex") or "")
    if not exp_b5:
        _d4_halt(
            d4,
            halt_at_check_id="B5_SET_HEX_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="EXPECTED_B5_SET_HEX_MISSING",
        )
        return _finish_and_return()

    b5_rel = "meta/b5_index.v1.json"
    b5_path = _b1_rel_dirs_for_root(bdir)["meta"] / "b5_index.v1.json"
    try:
        b5_raw = b5_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        d4["checks"].append(_d4_check("B5_SET_HEX_MATCH", None, detail=_d4_io_detail(exc, b5_rel), evidence={"path": b5_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="B5_SET_HEX_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_MISSING",
            reason_detail_code="B5_INDEX_NOT_FOUND",
        )
        return _finish_and_return()
    except Exception as exc:
        d4["checks"].append(_d4_check("B5_SET_HEX_MATCH", None, detail=_d4_io_detail(exc, b5_rel), evidence={"path": b5_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="B5_SET_HEX_MATCH",
            status="ERROR",
            reason_code="IO_ERROR",
            reason_detail_code="B5_INDEX_READ_FAILED",
        )
        return _finish_and_return()

    try:
        b5_idx = _json.loads(b5_raw)
    except Exception as exc:
        d4["checks"].append(_d4_check("B5_SET_HEX_MATCH", None, detail=_d4_io_detail(exc, b5_rel), evidence={"path": b5_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="B5_SET_HEX_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="B5_INDEX_PARSE_ERROR",
        )
        return _finish_and_return()

    if not isinstance(b5_idx, dict):
        d4["checks"].append(_d4_check("B5_SET_HEX_MATCH", None, detail="not an object", evidence={"path": b5_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="B5_SET_HEX_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="B5_INDEX_NOT_OBJECT",
        )
        return _finish_and_return()

    obs_b5 = str((b5_idx or {}).get("b5_set_hex") or "")
    if not obs_b5:
        _d4_halt(
            d4,
            halt_at_check_id="B5_SET_HEX_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="OBSERVED_B5_SET_HEX_MISSING",
        )
        return _finish_and_return()

    b5_ok = bool(exp_b5 == obs_b5)
    d4["checks"].append(
        _d4_check(
            "B5_SET_HEX_MATCH",
            b5_ok,
            detail=f"expected={exp_b5}, observed={obs_b5}",
            evidence={"path": b5_rel},
        )
    )

    # --- Check 10: SUITE_FIXTURES_SET_MATCH (expected-side required; no fallback) ---
    suite_cert = cert_obj.get("suite") or {}
    exp_set_sig8 = str((suite_cert or {}).get("fixtures_set_sig8") or "")
    if not exp_set_sig8:
        _d4_halt(
            d4,
            halt_at_check_id="SUITE_FIXTURES_SET_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="EXPECTED_FIXTURES_SET_SIG8_MISSING",
        )
        return _finish_and_return()

    v2_rel = "manifests/v2_suite_full_scope.jsonl"
    v2_path = _b1_manifest_v2_suite_path(bdir)
    try:
        suite_obs = _d4_manifest_digest(snapshot_id=sid, manifest_path=v2_path)
    except FileNotFoundError as exc:
        d4["checks"].append(_d4_check("SUITE_FIXTURES_SET_MATCH", None, detail=_d4_io_detail(exc, _d4_relpath(bdir, v2_path) or v2_rel), evidence={"path": _d4_relpath(bdir, v2_path) or v2_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="SUITE_FIXTURES_SET_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_MISSING",
            reason_detail_code="SUITE_MANIFEST_NOT_FOUND",
        )
        return _finish_and_return()
    except Exception as exc:
        d4["checks"].append(_d4_check("SUITE_FIXTURES_SET_MATCH", None, detail=_d4_io_detail(exc, _d4_relpath(bdir, v2_path) or v2_rel), evidence={"path": _d4_relpath(bdir, v2_path) or v2_rel}))
        _d4_halt(
            d4,
            halt_at_check_id="SUITE_FIXTURES_SET_MATCH",
            status="ERROR",
            reason_code="IO_ERROR",
            reason_detail_code="SUITE_MANIFEST_READ_FAILED",
        )
        return _finish_and_return()

    obs_set_sig8 = str((suite_obs or {}).get("fixtures_set_sig8") or "")
    if not obs_set_sig8:
        _d4_halt(
            d4,
            halt_at_check_id="SUITE_FIXTURES_SET_MATCH",
            status="NOT_ATTEMPTED",
            reason_code="PRECOND_INVALID",
            reason_detail_code="OBSERVED_FIXTURES_SET_SIG8_MISSING",
        )
        return _finish_and_return()

    suite_ok = bool(exp_set_sig8 == obs_set_sig8)
    d4["checks"].append(
        _d4_check(
            "SUITE_FIXTURES_SET_MATCH",
            suite_ok,
            detail=f"expected={exp_set_sig8}, observed={obs_set_sig8}",
            evidence={"path": _d4_relpath(bdir, v2_path) or v2_rel},
        )
    )

    # --- Check 11: PARITY_CERT_VERIFIED ---
    parity_ok: bool | None = None
    parity_detail: str = ""
    parity_evidence: dict = {}
    try:
        meta = manifest_obj.get("meta") if isinstance(manifest_obj.get("meta"), dict) else {}
        ref_i = (meta or {}).get("parity_instance") or {}
        ref_c = (meta or {}).get("parity_certificate") or {}
        if not isinstance(ref_i, dict) or not isinstance(ref_c, dict):
            parity_ok = None
            parity_detail = "parity meta missing"
        else:
            i_sig8 = str(ref_i.get("sig8") or "").strip()
            c_sig8 = str(ref_c.get("sig8") or "").strip()
            if not i_sig8 or not c_sig8:
                parity_ok = None
                parity_detail = "parity meta sig8 missing"
            else:
                inst_file = _b1_parity_instance_path(bdir, sid, i_sig8)
                cert_file = _b1_parity_certificate_path(bdir, sid, c_sig8)
                parity_evidence = {
                    "instance_path": _d4_relpath(bdir, inst_file) or "",
                    "certificate_path": _d4_relpath(bdir, cert_file) or "",
                    "instance_sig8": i_sig8,
                    "certificate_sig8": c_sig8,
                }
                if not inst_file.exists() or not cert_file.exists():
                    parity_ok = None
                    parity_detail = "parity files missing"
                else:
                    try:
                        inst_raw = inst_file.read_text(encoding="utf-8")
                        cert_raw = cert_file.read_text(encoding="utf-8")
                    except Exception as exc:
                        parity_ok = None
                        parity_detail = _d4_io_detail(exc, "certs/parity")
                    else:
                        try:
                            inst_obj = _json.loads(inst_raw)
                            cert_obj = _json.loads(cert_raw)
                        except Exception as exc:
                            parity_ok = None
                            parity_detail = _d4_io_detail(exc, "certs/parity")
                        else:
                            parity_ok = True if VerifyV0(inst_obj, cert_obj) else False
                            parity_detail = "verified" if parity_ok is True else "verify_failed"
    except Exception as exc:
        parity_ok = None
        parity_detail = f"error: {exc}"
    d4["checks"].append(
        _d4_check(
            "PARITY_CERT_VERIFIED",
            parity_ok,
            detail=parity_detail,
            evidence=(parity_evidence if isinstance(parity_evidence, dict) else {}),
        )
    )

    # Final verdict: all required checks must be true. No FAIL status (B3 vocab only).
    ok_map = {c.get("check_id"): c.get("ok") for c in (d4.get("checks") or []) if isinstance(c, dict)}
    d4["verdict"] = all(ok_map.get(cid) is True for cid in D4_REQUIRED_IDS_V0_1)
    d4["status"] = "OK"
    d4.pop("reason_code", None)
    d4.pop("reason_detail_code", None)

    return _finish_and_return()


# --- D4: locate latest cert and attach to bundle manifest meta (D4.E3) ---


def _d4_cert_root_dir_readonly() -> _Path:
    """Return the D4 cert root dir (logs/certs/d4) without creating it.

    IMPORTANT (A1 discipline): this helper MUST be read-only. Do not call
    `_d4_cert_root_dir()` from gate checks or other read-only code paths,
    because `_d4_cert_root_dir()` mkdirs.
    """
    try:
        root = _CERTS_DIR / "d4"  # type: ignore[name-defined]
    except Exception:
        try:
            root = _repo_root() / "logs" / "certs" / "d4"  # type: ignore[name-defined]
        except Exception:
            try:
                root = _REPO_ROOT / "logs" / "certs" / "d4"  # type: ignore[name-defined]
            except Exception:
                root = _Path("logs") / "certs" / "d4"
    return _Path(root).resolve()



def _d4_resolve_certificate_for_snapshot(snapshot_id: str, run_ctx: dict | None = None) -> tuple[Path, dict]:
    """STRICT resolver (pointer-gated): resolve the D4 certificate for a snapshot.

    - Requires run_ctx['d4_certificate_sig8'].
    - Resolves exactly the filename d4_certificate__<sid>__<sig8>.json under the D4 cert root.
    - Verifies the internal sig8 matches the pointer.
    """
    root = _d4_cert_root_dir_readonly()
    sid_str = _safe_snapshot_id_str(snapshot_id)
    sig8 = (run_ctx or {}).get("d4_certificate_sig8")

    if sig8 is None:
        raise RuntimeError(
            "D4 resolve: missing run_ctx.d4_certificate_sig8 pointer; "
            "pointer-gated resolution is required (use _d4_try_resolve_certificate_for_snapshot for UI discovery)."
        )

    p = (root / f"d4_certificate__{sid_str}__{sig8}.json").resolve()
    if not p.exists():
        raise RuntimeError(
            f"D4 resolve: certificate not found for snapshot_id {sid_str!r} and sig8={sig8!r} under {root}"
        )

    cert = _load_d4_certificate_payload(p)
    internal = (cert or {}).get("sig8")
    if internal != sig8:
        raise RuntimeError(
            f"D4 resolve: sig8 mismatch for snapshot_id {sid_str!r}: pointer={sig8!r} internal={internal!r}"
        )
    return p, cert

def _parity_cert_root_dir_readonly() -> _Path:
    """Return the parity cert root dir (logs/certs/parity) without creating it."""
    try:
        root = _CERTS_DIR / "parity"  # type: ignore[name-defined]
    except Exception:
        try:
            root = _repo_root() / "logs" / "certs" / "parity"  # type: ignore[name-defined]
        except Exception:
            try:
                root = _REPO_ROOT / "logs" / "certs" / "parity"  # type: ignore[name-defined]
            except Exception:
                root = _Path("logs") / "certs" / "parity"
    return _Path(root).resolve()



def _parity_resolve_pair_for_snapshot(snapshot_id: str, run_ctx: dict | None = None) -> tuple[tuple[Path, dict], tuple[Path, dict]]:
    """STRICT resolver (pointer-gated): resolve (parity_certificate, parity_instance) for a snapshot.

    Requires:
      - run_ctx['parity_certificate_sig8']
      - run_ctx['parity_instance_sig8']

    Resolves exact sig8-stamped filenames under the parity roots and verifies internal sig8 fields.
    """
    root = _parity_cert_root_dir_readonly()
    sid_str = _safe_snapshot_id_str(snapshot_id)

    cert_sig8 = (run_ctx or {}).get("parity_certificate_sig8")
    inst_sig8 = (run_ctx or {}).get("parity_instance_sig8")

    if cert_sig8 is None:
        raise RuntimeError(
            "Parity resolve: missing run_ctx.parity_certificate_sig8 pointer; "
            "pointer-gated resolution is required (use _parity_try_resolve_pair_for_snapshot for UI discovery)."
        )
    if inst_sig8 is None:
        raise RuntimeError("Parity resolve: missing parity_instance_sig8 in run_ctx")

    cert_path = (root / f"parity_certificate__{sid_str}__{cert_sig8}.json").resolve()
    if not cert_path.exists():
        raise RuntimeError(
            f"Parity resolve: parity certificate not found for snapshot_id {sid_str!r} and cert_sig8={cert_sig8!r} under {root}"
        )

    cert = _load_parity_certificate_payload(cert_path)
    internal_cert_sig8 = (cert or {}).get("sig8")
    if internal_cert_sig8 != cert_sig8:
        raise RuntimeError(
            f"Parity resolve: certificate sig8 mismatch for snapshot_id {sid_str!r}: pointer={cert_sig8!r} internal={internal_cert_sig8!r}"
        )

    inst_path = (root / f"parity_instance__{sid_str}__{inst_sig8}.json").resolve()
    if not inst_path.exists():
        raise RuntimeError(
            f"Parity resolve: parity instance not found for snapshot_id {sid_str!r} and instance_sig8={inst_sig8!r} under {root}"
        )

    inst = _load_parity_instance_payload(inst_path)
    internal_inst_sig8 = (inst or {}).get("sig8")
    if internal_inst_sig8 != inst_sig8:
        raise RuntimeError(
            f"Parity resolve: instance sig8 mismatch for snapshot_id {sid_str!r}: pointer={inst_sig8!r} internal={internal_inst_sig8!r}"
        )

    return (cert_path, cert), (inst_path, inst)

def _parity_try_resolve_pair_for_snapshot(
    snapshot_id: str | None = None,
    run_ctx: dict | None = None,
) -> tuple[_Path, dict, _Path, dict] | None:
    """Best-effort parity resolver.

    Phase-0 wiring intent (authority/selection discipline):
      - Parity artifacts are *optional* unless explicitly requested by run_ctx.
      - Absence (missing root or no certificate candidates for the snapshot) is
        not an error in optional mode.
      - Presence (any certificate candidate exists) triggers strict resolution:
        ambiguity or invalid pairs are surfaced as errors (no silent tie-breaks).

    This function is read-only: it MUST NOT create directories.

    Returns:
        The strict-resolved (instance_path, instance_obj, cert_path, cert_obj)
        iff parity artifacts appear present (or are explicitly requested).
        Otherwise returns None.
    """
    sid_str = str(snapshot_id or "").strip()
    if not sid_str:
        return None

    rc = run_ctx if isinstance(run_ctx, dict) else {}
    explicit = bool(
        (rc.get("parity_certificate_path") or rc.get("parity_cert_path"))
        or (rc.get("parity_certificate_sig8") or rc.get("parity_cert_sig8"))
        or (rc.get("parity_instance_path") or rc.get("parity_inst_path"))
        or (rc.get("parity_instance_sig8") or rc.get("parity_inst_sig8"))
    )

    # If explicitly requested, delegate to the strict resolver (which will fail
    # loudly if root/artifacts are missing).
    if explicit:
        return _parity_resolve_pair_for_snapshot(snapshot_id=sid_str, run_ctx=run_ctx)

    root = _parity_cert_root_dir_readonly()
    if not root.exists():
        return None

    # Optional mode: only attempt strict resolution if any certificate candidate
    # exists for this snapshot_id.
    try:
        candidates = [p for p in root.glob(f"parity_certificate__{sid_str}__*.json") if p.exists()]
    except Exception:
        candidates = []

    if not candidates:
        return None

    # Candidates exist: enforce strict resolution discipline.
    return _parity_resolve_pair_for_snapshot(snapshot_id=sid_str, run_ctx=run_ctx)

def _d4_latest_certificate_for_snapshot(
    snapshot_id: str | None = None,
) -> tuple[_Path | None, dict | None]:
    """Best-effort helper: locate and load the latest D4 cert for a snapshot.

    Returns (path, cert_dict) or (None, None) if no suitable certificate exists.
    """
    sid = snapshot_id or _v2_current_world_snapshot_id(strict=True)
    if not sid:
        return None, None

    try:
        root = _d4_cert_root_dir()
    except Exception:
        try:
            root = _CERTS_DIR / "d4"  # type: ignore[name-defined]
        except Exception:
            root = _Path("logs") / "certs" / "d4"

    try:
        pattern = f"d4_certificate__{sid}__*.json"
        candidates = list(_Path(root).glob(pattern))
    except Exception:
        candidates = []

    best: _Path | None = None
    for cand in candidates:
        if not cand.exists():
            continue
        if best is None:
            best = cand
        else:
            try:
                if cand.stat().st_mtime > best.stat().st_mtime:
                    best = cand
            except Exception:
                # If stat fails, keep the existing best path.
                pass

    if best is None:
        return None, None

    try:
        data = _json.loads(best.read_text(encoding="utf-8"))
    except Exception:
        data = None

    return best, data


def _d4_attach_certificate_meta_to_manifest(
    manifest: dict,
    snapshot_id: str | None = None,
    run_ctx: dict | None = None,
) -> dict:
    """Attach a compact D4 certificate reference into manifest['meta'] (A1).

    This is called from the B1 exporter *before* bundle-manifest sig8 is
    computed.

    A1 discipline:
      - D4 is mandatory for B1 export.
      - Selection MUST be explicit (run_ctx) or provably unique.
      - Never select by mtime or lexicographic tie-break.
    """
    base = dict(manifest or {})
    meta = dict(base.get("meta") or {})

    sid: str | None = None
    if snapshot_id is not None:
        sid = str(snapshot_id)
    elif isinstance(base, dict):
        sid = str(base.get("snapshot_id") or "")
    if (not sid) and run_ctx is not None and isinstance(run_ctx, dict):
        sid = str(run_ctx.get("snapshot_id") or "")
    if not sid:
        try:
            sid = _v2_current_world_snapshot_id(strict=True)
        except Exception:
            sid = None
    if not sid:
        base["meta"] = meta
        return base

    cert_path, cert_obj = _d4_resolve_certificate_for_snapshot(snapshot_id=sid, run_ctx=run_ctx)
    sig8 = str((cert_obj or {}).get("sig8") or "").strip()
    if not sig8:
        raise RuntimeError(f"D4 meta attach: resolved certificate missing sig8: {cert_path}")

    # Bundle-local locator for verifiers. (Do not leak host paths.)
    meta["d4_certificate"] = {
        "path": f"certs/d4/d4_certificate__{sid}__{sig8}.json",
        "sig8": sig8,
    }
    base["meta"] = meta
    return base





_V2_EXPECTED = [
    ("strict",               "overlap__", "__strict__"),
    ("projected_auto",       "overlap__", "__projected_columns_k_3_auto__"),
    ("ab_auto",              "ab_compare__strict_vs_projected_auto__", ""),
    ("freezer",              "projector_freezer__", "",),
    # ab_file: strict vs projected(FILE) pair
    ("ab_file",              "ab_compare__strict_vs_projected_file__", ""),
    ("projected_file",       "overlap__", "__projected_columns_k_3_file__"),
]




# --- B3 vocabulary (Termination & loop-closure; v0.2 lock) ---
# NOTE: Keep these "boring". Do not free-form status/mode/reason strings elsewhere.
B3_STATUS = ("OK", "NOT_ATTEMPTED", "NA", "ERROR")
B3_STATUS_SET = set(B3_STATUS)

B3_VERDICT_MODE = ("BOOLEAN", "PRESENCE_ONLY", "EMBED_ONLY")
B3_VERDICT_MODE_SET = set(B3_VERDICT_MODE)

B3_REASON_CODES = (
    "PRECOND_MISSING",
    "PRECOND_INVALID",
    "POLICY_DISABLED",
    "LOGIC_NA",
    "LIMIT_HIT",
    "DEP_NOT_ATTEMPTED",
    "DEP_ERROR",
    "INTERNAL_ERROR",
    "IO_ERROR",
    "WRITER_FAILURE",
)
B3_REASON_CODES_SET = set(B3_REASON_CODES)


# Loop termination (B3) vocabulary. Stored on loop_receipt.v2.
B3_TERMINAL_STATUS = (
    "TERMINATED_SUCCESS",
    "TERMINATED_FAIL",
    "TERMINATED_NA",
    "TERMINATED_CAP",
    "ABORTED_ERROR",
)
B3_TERMINAL_STATUS_SET = set(B3_TERMINAL_STATUS)

# Termination reason codes (minimal; may be elevated/extended in later phases).
B3_TERMINATION_REASON_CODES = (
    "SINGLE_PASS",
    "SUITE_ROW",
    "REBUILD",
    "LIMIT_HIT",
    "LOGIC_NA",
    "ERROR",
)
B3_TERMINATION_REASON_CODES_SET = set(B3_TERMINATION_REASON_CODES)


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
    # B3 self-id + timestamp fields must never affect canonical digests
    "payload_sig8", "payload_sha256", "timestamps",
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


# --- B3 payload identity (CJF-1) helpers ---
# Functional MUST: local content-addressed identity for every artifact payload.
# Canonicalization rules:
#   - drop non-core fields (schema-defined)
#   - drop self-hash fields (payload_sig8/payload_sha256)
#   - sort keys, compact separators, UTF-8
#   - disallow NaN/Inf
_B3_SELF_HASH_FIELDS = ("payload_sig8", "payload_sha256")

# Common non-core keys for B3 payload hashing. Schemas may override/extend.
# We intentionally KEEP "paths" by default (loop_receipt regen relies on it).
_B3_NON_CORE_COMMON_KEYS = set(_V2_EPHEMERAL_KEYS) | {"timestamps"}
_B3_NON_CORE_COMMON_KEYS.discard("paths")


def _b3_payload_canonical_obj(obj, *, non_core_keys=None, self_fields=_B3_SELF_HASH_FIELDS):
    """Return a canonical object for B3 payload hashing (CJF-1)."""
    exclude = set(non_core_keys) if non_core_keys is not None else set(_B3_NON_CORE_COMMON_KEYS)
    exclude.update(self_fields)
    return _v2_canonical_obj(obj, exclude_keys=exclude)


def b3_payload_sha256(obj, *, non_core_keys=None) -> str:
    """Full SHA-256 hex digest of the B3 payload identity surface (CJF-1)."""
    can = _b3_payload_canonical_obj(obj, non_core_keys=non_core_keys)
    txt = _json.dumps(can, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return _hash.sha256(txt.encode("utf-8")).hexdigest()


def b3_payload_sig8(obj, *, non_core_keys=None) -> str:
    """Short (8-hex) alias of b3_payload_sha256(...)."""
    return b3_payload_sha256(obj, non_core_keys=non_core_keys)[:8]


def b3_stamp_payload_sig8(payload: dict, *, non_core_keys=None, include_full_sha256: bool = False) -> dict:
    """Stamp payload_sig8 (and optionally payload_sha256) onto a payload dict.

    Hashing excludes these self-fields, so stamping is stable across runs.
    """
    sha = b3_payload_sha256(payload, non_core_keys=non_core_keys)
    payload["payload_sig8"] = sha[:8]
    if include_full_sha256:
        payload["payload_sha256"] = sha
    return payload

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
        "terminal_status": "TERMINATED_SUCCESS",
        "termination_reason_code": "SUITE_ROW",
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

# --- D4.UI helpers: SSOT + latest cert for UI (D4.UI.A) ---


def _ui_current_ssot_snapshot() -> str | None:
    """
    Return the current SSOT snapshot_id for the UI.

    This mirrors the semantics we already use elsewhere: prefer the
    existing v2 world-snapshot SSOT, but do not raise if it's missing.
    """
    try:
        sid = _v2_current_world_snapshot_id(strict=False)
    except Exception:
        sid = None
    if not sid:
        return None
    return str(sid)


def _ui_latest_d4_cert_for_ssot() -> tuple[_Path | None, dict | None]:
    """
    Convenience helper for the UI:

      - resolve the current SSOT snapshot_id
      - locate the latest D4 certificate for that snapshot
      - return (path, cert_obj) or (None, None) if nothing is found.
    """
    sid = _ui_current_ssot_snapshot()
    if not sid:
        return None, None

    try:
        path, obj = _d4_latest_certificate_for_snapshot(snapshot_id=sid)
    except Exception:
        return None, None

    if path is None or not obj:
        return None, None

    return path, obj



def _ui_latest_bundle_dir_for_ssot(snapshot_id: str | None) -> _Path | None:
    """Best-effort: locate the latest bundle dir for a given SSOT snapshot.

    Bundle dirs are expected under logs/bundle/ and named:
      {snapshot_id}__{sig8}

    Selection is by newest mtime of meta/bundle_manifest.json when present,
    otherwise by directory mtime.
    """
    sid = str(snapshot_id or "").strip()
    if not sid:
        return None

    # Prefer the canonical bundle root if present; otherwise fall back.
    try:
        base = _Path(_BUNDLE_ROOT)  # type: ignore[name-defined]
    except Exception:
        try:
            base = _Path(_REPO_ROOT) / "logs" / "bundle"  # type: ignore[name-defined]
        except Exception:
            base = _Path("logs") / "bundle"

    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    pattern = f"{sid}__*"
    try:
        cands = [p for p in base.glob(pattern) if p.is_dir()]
    except Exception:
        cands = []

    if not cands:
        return None

    def _mtime(p: _Path) -> float:
        try:
            mp = _Path(p) / "meta" / "bundle_manifest.json"
            if mp.exists():
                return float(mp.stat().st_mtime)
        except Exception:
            pass
        try:
            return float(_Path(p).stat().st_mtime)
        except Exception:
            return 0.0

    best = max(cands, key=_mtime)
    try:
        return _Path(best).resolve()
    except Exception:
        return _Path(best)


def _ui_read_b6_verify_receipt(bundle_dir: _Path) -> tuple[_Path | None, dict | None, str | None]:
    """Read meta/b6_verify_receipt.json from a bundle dir.

    Returns (receipt_path, receipt_obj, err_str).
    """
    import json as _json

    bdir = _Path(bundle_dir)
    rpath = bdir / B6_VERIFY_RECEIPT_REL_PATH
    if not rpath.exists():
        return rpath, None, f"missing receipt: {B6_VERIFY_RECEIPT_REL_PATH}"
    try:
        raw = rpath.read_text(encoding="utf-8")
    except Exception as e:
        return rpath, None, f"receipt read failed: {e!r}"
    try:
        obj = _json.loads(raw)
    except Exception as e:
        return rpath, None, f"receipt parse failed: {e!r}"
    if not isinstance(obj, dict):
        return rpath, None, "receipt is not a JSON object"
    return rpath, obj, None


def _ui_derive_bundle_verify_ui(receipt: dict | None) -> dict:
    """Derive a tiny UI status object *only* from the persisted receipt.

    Output keys:
      - ui_status: "CERTIFIED" | "NOT_CERTIFIED"
      - primary: short human string (minimal admissible claim)
      - issues: list[str] (minimal, receipt-derived)
    """
    r = receipt if isinstance(receipt, dict) else {}
    b6_status = str(r.get("status") or "")
    b6_verdict = r.get("verdict")
    b6_reason = str(r.get("reason_detail_code") or r.get("reason_code") or "")
    b6_fail_codes = r.get("fail_codes") if isinstance(r.get("fail_codes"), list) else []

    d4 = r.get("d4") if isinstance(r.get("d4"), dict) else {}
    d4_status = str(d4.get("status") or "")
    d4_verdict = d4.get("verdict")
    d4_reason = str(d4.get("reason_detail_code") or d4.get("reason_code") or "")
    checks = d4.get("checks") if isinstance(d4.get("checks"), list) else []

    # Success: certified == D4 OK + verdict true.
    if d4_status == "OK" and d4_verdict is True:
        return {
            "ui_status": "CERTIFIED",
            "primary": "Structurally Consistent (Current Spec)",
            "issues": [],
        }

    issues: list[str] = []

    # B6 precondition or error.
    if b6_status and b6_status != "OK":
        # Prefer mapping to the canonical seal artifact when possible.
        if b6_reason == "SEAL_MISSING":
            issues.append("Artifact missing: meta/bundle_hash.json")
        elif b6_reason in ("SEAL_PARSE_ERROR", "SEAL_NOT_OBJECT") or b6_reason.startswith("SEAL_MISSING_FIELDS"):
            issues.append("Artifact invalid: meta/bundle_hash.json")
        elif b6_reason:
            issues.append(f"B6 precondition: {b6_reason}")
        else:
            issues.append(f"B6 status: {b6_status}")
        primary = issues[0] if issues else "NOT CERTIFIED"
        return {"ui_status": "NOT_CERTIFIED", "primary": primary, "issues": issues}

    # B6 ran but failed.
    if b6_status == "OK" and b6_verdict is False:
        if b6_fail_codes:
            # Minimal headline uses the first fail code.
            issues.extend([f"Identity check inconsistent: B6::{c}" for c in b6_fail_codes])
        else:
            issues.append("Identity check inconsistent: B6")
        primary = issues[0] if issues else "NOT CERTIFIED"
        return {"ui_status": "NOT_CERTIFIED", "primary": primary, "issues": issues}

    # D4 missing or not verifiable.
    if d4_status and d4_status != "OK":
        # Try to surface an artifact path from the halted check evidence.
        ev_path: str | None = None
        for chk in checks:
            if not isinstance(chk, dict):
                continue
            if chk.get("ok") is None:
                ev = chk.get("evidence")
                if isinstance(ev, dict) and ev.get("path"):
                    ev_path = str(ev.get("path"))
                    break
        if ev_path:
            if "NOT_FOUND" in d4_reason or "MISSING" in d4_reason:
                issues.append(f"Artifact missing: {ev_path}")
            elif "READ_FAILED" in d4_reason or "PARSE_ERROR" in d4_reason or "NOT_OBJECT" in d4_reason or "RECOMPUTE_FAILED" in d4_reason:
                issues.append(f"Artifact invalid: {ev_path}")
            else:
                issues.append(f"Not verifiable: {d4_reason or 'D4 halted'}")
        else:
            issues.append(f"Not verifiable: {d4_reason or 'D4 halted'}")
        primary = issues[0] if issues else "NOT CERTIFIED"
        return {"ui_status": "NOT_CERTIFIED", "primary": primary, "issues": issues}

    # D4 completed but verdict false -> failing checks.
    if d4_status == "OK" and d4_verdict is False:
        bad_checks: list[str] = []
        for chk in checks:
            if not isinstance(chk, dict):
                continue
            if chk.get("ok") is False and chk.get("check_id"):
                bad_checks.append(str(chk.get("check_id")))
        if bad_checks:
            issues.append(f"Identity check inconsistent: {bad_checks[0]}")
            for cid in bad_checks[1:]:
                issues.append(f"Identity check inconsistent: {cid}")
        else:
            issues.append("Identity check inconsistent: D4")
        primary = issues[0] if issues else "NOT CERTIFIED"
        return {"ui_status": "NOT_CERTIFIED", "primary": primary, "issues": issues}

    # Catch-all.
    if d4_reason:
        issues.append(f"Not certified: {d4_reason}")
    elif b6_reason:
        issues.append(f"Not certified: {b6_reason}")
    else:
        issues.append("NOT CERTIFIED")
    return {"ui_status": "NOT_CERTIFIED", "primary": issues[0], "issues": issues}



def render_d4_certifier_panel() -> None:
    """
    D4 — Certifier & Bundle export panel.

    Shows current SSOT snapshot and wiring for:
      - building a D4 certificate for the current snapshot,
      - verifying the latest D4 certificate, and
      - exporting a full v2+τ + D4 bundle zip.
    """
    # Resolve current SSOT snapshot from the backend helpers.
    snapshot_id = _ui_current_ssot_snapshot()
    snapshot_str = snapshot_id or "None"

    # Engine revision: prefer run_ctx["engine_rev"], fall back to ENGINE_REV if present.
    engine_rev: str = "unknown"
    try:
        import streamlit as st  # local import to access session_state safely
        try:
            rc = dict(st.session_state.get("run_ctx") or {})
        except Exception:
            rc = {}
        er = rc.get("engine_rev")
        if er:
            engine_rev = str(er)
        else:
            try:
                # type: ignore[name-defined] - ENGINE_REV is provided by the solver build.
                engine_rev = str(ENGINE_REV)  # type: ignore[name-defined]
            except Exception:
                # Leave as "unknown" if ENGINE_REV is not wired.
                pass
    except Exception:
        # If even importing Streamlit/session_state fails, keep the sentinel.
        engine_rev = "unknown"

    # Latest D4 certificate (if any) for the current SSOT snapshot.
    cert_path, cert_obj = _ui_latest_d4_cert_for_ssot()
    has_cert = bool(cert_path and cert_obj)

    import streamlit as st  # UI import

    st.subheader("D4 — Certifier & Bundle export")

    # Tiny metadata block at the top.
    st.markdown(
        f"**SSOT snapshot_id:** `{snapshot_str}`  \n"
        f"**Engine revision:** `{engine_rev}`  \n"
        f"**D4 certificate present:** {'✅ yes' if has_cert else '⬜ no'}"
    )

    # Build D4 certificate button
    if st.button("Build D4 certificate for current snapshot"):
        ssot = _ui_current_ssot_snapshot()
        if not ssot:
            st.warning(
                "No current SSOT snapshot; run the v2+τ pipeline at least once first."
            )
        else:
            try:
                path = write_d4_certificate_for_snapshot(snapshot_id=ssot, run_ctx=None)
            except Exception as e:
                st.error(f"Failed to write D4 certificate: {e!r}")
            else:
                st.success(f"Wrote D4 certificate for snapshot `{ssot}`: {path}")
                # Optionally: show sig8 from the latest D4 certificate for this snapshot.
                try:
                    cert_path2, cert_obj2 = _d4_latest_certificate_for_snapshot(
                        snapshot_id=ssot
                    )
                except Exception:
                    cert_path2, cert_obj2 = None, None
                if cert_obj2 and isinstance(cert_obj2, dict):
                    sig8 = (
                        cert_obj2.get("sig8")
                        or cert_obj2.get("strict_sig8")
                        or cert_obj2.get("signature")
                    )
                    if sig8:
                        st.info(f"D4 certificate sig8: `{sig8}`")

    # Verify latest D4 certificate button
    if st.button("Verify latest D4 certificate"):
        cert_path2, cert_obj2 = _ui_latest_d4_cert_for_ssot()
        if not cert_path2:
            st.warning("No D4 certificate found for the current snapshot.")
        else:
            try:
                ok, msg = verify_d4_certificate(cert_path2)
            except Exception as e:
                st.error(f"Verification failed with an exception: {e!r}")
            else:
                # Try to show which snapshot we verified.
                snapshot_id2 = snapshot_id or _ui_current_ssot_snapshot()
                prefix = f"[snapshot `{snapshot_id2}`] " if snapshot_id2 else ""
                if ok:
                    st.success(f"{prefix}Verification OK: {msg}")
                else:
                    st.error(f"{prefix}Verification FAILED: {msg}")

                # Optional: surface sig8 again for quick cross-checks.
                if cert_obj2 and isinstance(cert_obj2, dict):
                    sig8 = (
                        cert_obj2.get("sig8")
                        or cert_obj2.get("strict_sig8")
                        or cert_obj2.get("signature")
                    )
                    if sig8:
                        st.info(f"D4 certificate sig8: `{sig8}`")


    # Track III policy profile selection (affects certification + Time(τ) required-set regime).
    _t3_profiles = {
        f"Pointer-gated DW (v0.2) — {T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2}": T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2,
        f"Pointer-gated DW + T0 mandatory (v0.4) — {T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_V0_4}": T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_V0_4,
        f"Pointer-gated DW + parity required (v0.5) — {T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_5}": T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_5,
        f"Pointer-gated DW + T0 mandatory + parity required (v0.6) — {T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_PARITY_V0_6}": T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_PARITY_V0_6,
        f"Mandatory DW (v0.3) — {T3_POLICY_PROFILE_ID_MANDATORY_DW_V0_3}": T3_POLICY_PROFILE_ID_MANDATORY_DW_V0_3,
    }
    _t3_choice = st.selectbox(
        "Track III policy profile",
        options=list(_t3_profiles.keys()),
        index=0,
        key="ui_t3_policy_profile_choice",
        help="Select the policy regime used when certifying this bundle (affects Time(τ) required checks).",
    )
    ui_t3_policy_profile_id = _t3_profiles.get(_t3_choice) or T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2
    ui_dw_mandatory = (ui_t3_policy_profile_id == T3_POLICY_PROFILE_ID_MANDATORY_DW_V0_3)

    # Export bundle button
    activate_dw = st.checkbox(
        "Activate C3 derived worlds in τ closure (policy-driven)",
        value=(True if ui_dw_mandatory else False),
        key="ui_activate_c3_derived_worlds",
        disabled=ui_dw_mandatory,
        help=(
            "When enabled, export writes a derived-worlds manifest + pointer so derived worlds join Cl(τ) "
            "and affect tau_surface_sig8. When disabled, derived worlds are excluded from the exported bundle "
            "and τ closure ignores them."
        ),
    )
    if ui_dw_mandatory:
        activate_dw = True
        st.caption("Derived worlds are mandatory under the selected policy; export forces DW into τ closure.")
    if st.button("Export bundle (v2+τ + D4 + zip)"):
        try:
            bundle_path = export_bundle_for_snapshot(snapshot_id=None, activate_derived_worlds=activate_dw)
        except Exception as e:
            st.error(f"Bundle export failed: {e!r}")
        else:
            st.success(f"Bundle exported to: {bundle_path}")
            # Record the last exported bundle zip + dir for the 1-click verifier.
            try:
                st.session_state["ui_last_bundle_zip"] = str(bundle_path)
                bn = _Path(bundle_path).name
                parts = bn.split("__")
                if len(parts) >= 3 and parts[0] == "bundle":
                    sid = parts[1]
                    sig8 = parts[2]
                    if sig8.endswith(".zip"):
                        sig8 = sig8[:-4]
                    bdir = _Path(bundle_path).parent / f"{sid}__{sig8}"
                    st.session_state["ui_last_bundle_dir"] = str(bdir)
            except Exception:
                pass

    # --- Verify/Certify bundle (Current Spec) ---
    st.markdown("---")
    st.markdown("### Verify/Certify bundle (Current Spec)")

    # Resolve a target bundle directory: prefer the most recent export, else fallback to latest for SSOT.
    target_bdir: _Path | None = None
    try:
        last_bdir = st.session_state.get("ui_last_bundle_dir")
    except Exception:
        last_bdir = None
    if last_bdir:
        try:
            p = _Path(str(last_bdir))
            if p.exists() and p.is_dir():
                target_bdir = p
        except Exception:
            target_bdir = None
    if target_bdir is None and snapshot_id:
        target_bdir = _ui_latest_bundle_dir_for_ssot(snapshot_id)

    if target_bdir:
        st.caption(f"Target bundle dir: `{target_bdir}`")
    else:
        st.caption("Target bundle dir: (none found)")

    # Run verification on click. UI truth is always derived from the persisted receipt file.
    if st.button("Verify/Certify bundle (B6 seal + D4)"):
        if not target_bdir:
            st.warning("No bundle directory found. Export a bundle first.")
        else:
            try:
                d4_verify_bundle_dir(target_bdir, write_receipt=True, t3_policy_profile_id=ui_t3_policy_profile_id)
            except Exception as e:
                st.error(f"Verifier raised an exception: {e!r}")
            try:
                st.session_state["ui_last_verify_bundle_dir"] = str(target_bdir)
            except Exception:
                pass

    # Render status (receipt-authoritative) if we have a target bundle.
    if target_bdir:
        receipt_path, receipt_obj, receipt_err = _ui_read_b6_verify_receipt(target_bdir)
        ui = _ui_derive_bundle_verify_ui(receipt_obj)

        if receipt_obj is None and receipt_err:
            # Tight minimal wording: missing receipt is an artifact missing condition.
            ui = {
                "ui_status": "NOT_CERTIFIED",
                "primary": f"Artifact missing: {B6_VERIFY_RECEIPT_REL_PATH}",
                "issues": [f"Artifact missing: {B6_VERIFY_RECEIPT_REL_PATH}"],
            }

        if str(ui.get("ui_status")) == "CERTIFIED":
            st.success(f"✅ CERTIFIED — {ui.get('primary')}")
        else:
            primary = str(ui.get("primary") or "NOT CERTIFIED")
            st.error(f"❌ NOT CERTIFIED — {primary}")

        # IMPORTANT: do not use st.expander here (the whole panel lives inside an expander).
        show_details_key = f"ui_d4_verify_details__{st.session_state.get('_ui_nonce','')}_bundle"
        show_details = st.checkbox("Show details (receipt-derived)", value=False, key=show_details_key)
        if show_details:
            if receipt_path:
                st.caption(f"Receipt path: `{receipt_path}`")
            if receipt_err and receipt_obj is None:
                st.warning(receipt_err)

            issues = ui.get("issues") or []
            if issues:
                st.markdown("**Issues:**")
                for it in issues:
                    st.markdown(f"- {it}")

            if receipt_obj is not None:
                st.markdown("**Raw receipt (JSON):**")
                st.json(receipt_obj)

    # Tiny status strip at the bottom (D4.UI.F).
    cert_path_str = str(cert_path) if cert_path else "none"
    sig8_status = "n/a"
    try:
        if cert_obj and isinstance(cert_obj, dict):
            sig8_status = (
                cert_obj.get("sig8")
                or cert_obj.get("strict_sig8")
                or cert_obj.get("signature")
                or "n/a"
            )
    except Exception:
        sig8_status = "n/a"

    st.markdown("---")
    st.caption("D4 status (current SSOT snapshot)")
    st.markdown(
        f"- **SSOT snapshot_id:** `{snapshot_str}`\n"
        f"- **Latest D4 cert path:** `{cert_path_str}`\n"
        f"- **Latest D4 cert sig8:** `{sig8_status}`"
    )

def _v2_current_world_snapshot_id(strict: bool = False) -> str | None:
    """Canonical v2 world snapshot_id (SSOT) for the solver.

    This is the only helper that should answer "what snapshot_id is this world on?"
    It simply delegates to _svr_current_snapshot_id(), but adds an optional
    strict mode for sanity checks.
    """
    sid = _svr_current_snapshot_id()
    if strict and not sid:
        # In strict mode we treat missing snapshot as a hard error.
        raise ValueError(
            "No canonical v2 world snapshot_id found; run bootstrap + suite first."
        )
    return sid


def _svr_current_run_label() -> str | None:
    """Optional inert metadata label for the current run.

    Source of truth: st.session_state['run_ctx'].get('run_label'), if present.
    This is never used for solver identity or filtering; it is purely metadata.
    """
    import streamlit as st  # local import to avoid hard dependency at import time
    try:
        rc = dict(st.session_state.get("run_ctx") or {})
    except Exception:
        rc = {}
    label = rc.get("run_label")
    if label is None:
        return None
    return str(label)




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


def make_b5_identity_path(bundle_dir, district_id: str, sig8: str):
    """Return Path to the per-fixture B5 identity record JSON.

    Pattern: b5_identity__{district_id}__{sig8}.json

    This is a per-fixture semantic-identity record (B5), separate from
    any bundle-level seal/hash.
    """
    b = _Path(bundle_dir)
    return b / f"b5_identity__{district_id}__{sig8}.json"

# --- v2 cert writers (DO NOT PRUNE) ---

# SSOT: canonical payload builders for v2 strict/projected certs
def build_v2_strict_cert_payload(base_hdr: dict, verdict: bool | None, *, na_reason_code: str | None = None) -> dict:
    """Strict cert payload from a base header and solver verdict.

    B3 semantics:
      - verdict_mode is BOOLEAN
      - status is OK when verdict is boolean, else NA
      - if verdict is None, we must propagate an explicit NA reason code (when available)

    Shape (superset; legacy fields preserved):
      {
        ...base_hdr,
        "policy": "strict",
        "status": "OK" | "NA",
        "verdict_mode": "BOOLEAN",
        "verdict": true/false/null,
        "reason_code": "LOGIC_NA" | null,
        "reason_detail_code": <string|null>,
        "na_reason_code": <string|null>,   # legacy alias (present when verdict is null)
        "payload_sig8": <string>,
      }
    """
    payload = dict(base_hdr or {})
    payload["policy"] = "strict"
    payload["verdict_mode"] = "BOOLEAN"

    v = verdict if (isinstance(verdict, bool) or verdict is None) else bool(verdict)
    payload["verdict"] = v

    if v is None:
        payload["status"] = "NA"
        payload["reason_code"] = "LOGIC_NA"
        # Preserve strict solver NA detail (when present).
        detail = str(na_reason_code) if na_reason_code is not None else None
        payload["reason_detail_code"] = detail
        # Legacy alias for back-compat with earlier anchors.
        payload["na_reason_code"] = detail
    else:
        payload["status"] = "OK"

    # Functional MUST: local content-addressed identity for every payload.
    b3_stamp_payload_sig8(payload)
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
    """AUTO projected cert payload from header + lanes + projection metadata + verdict.

    B3 semantics:
      - verdict_mode is BOOLEAN
      - status is OK when verdict is boolean and proj_meta.na is false
      - status is NA when proj_meta.na is true OR verdict is None
      - reason_code LOGIC_NA carries proj_meta.reason (when present)

    Legacy fields preserved:
      - na: bool
      - reason: string|null
    """
    payload = dict(base_hdr or {})
    payload["policy"] = "projected(columns@k=3,auto)"
    payload["verdict_mode"] = "BOOLEAN"

    ctx = {
        "lanes": [int(x) & 1 for x in (lanes_vec or [])],
        "lanes_popcount": int(lanes_popcount or 0),
        "lanes_sig8": lanes_sig8,
    }
    payload["projection_context"] = ctx

    meta = proj_meta if isinstance(proj_meta, dict) else {}
    na_flag = bool(meta.get("na")) if isinstance(meta, dict) else False
    reason = meta.get("reason") if isinstance(meta, dict) else None

    v = verdict if (isinstance(verdict, bool) or verdict is None) else bool(verdict)

    if na_flag or v is None:
        payload["status"] = "NA"
        payload["reason_code"] = "LOGIC_NA"
        payload["reason_detail_code"] = str(reason) if reason is not None else None
        payload["na"] = True
        payload["reason"] = reason
        payload["verdict"] = None
    else:
        payload["status"] = "OK"
        payload["na"] = False
        payload["reason"] = None
        payload["verdict"] = v

    b3_stamp_payload_sig8(payload)
    return payload

def build_v2_projected_file_cert_payload(
    base_hdr: dict,
    *,
    status: str = "OK",
    reason_code: str | None = None,
    reason_detail_code: str | None = None,
) -> dict:
    """FILE projected cert payload from a base header.

    B3 semantics:
      - This cert is structurally mandatory, even when file projection is not attempted.
      - verdict_mode is PRESENCE_ONLY and verdict is always null.
      - status encodes semantic optionality (OK vs NOT_ATTEMPTED vs ERROR/NA).
    """
    payload = dict(base_hdr or {})
    payload["policy"] = "projected(columns@k=3,file)"
    payload["verdict_mode"] = "PRESENCE_ONLY"

    st_norm = str(status or "OK")
    payload["status"] = st_norm
    payload["verdict"] = None

    if st_norm != "OK":
        payload["reason_code"] = str(reason_code or "PRECOND_INVALID")
        if reason_detail_code is not None:
            payload["reason_detail_code"] = str(reason_detail_code)

    b3_stamp_payload_sig8(payload)
    return payload

def build_v2_ab_compare_payload(
    base_hdr: dict,
    *,
    policy: str,
    left_policy: str,
    left_payload_sig8: str,
    right_policy: str,
    right_payload_sig8: str,
    status: str = "OK",
    reason_code: str | None = None,
    reason_detail_code: str | None = None,
) -> dict:
    """A/B compare payload for strict vs projected certs.

    B3 semantics (Step 6):
      - embed.* references are explicit payload_sig8 (no ambiguous 'sig8' keys)
      - verdict_mode is EMBED_ONLY (this artifact is a reference receipt, not a boolean verdict)
      - status encodes cascades (e.g. NOT_ATTEMPTED when the right side was not attempted)
      - reason_code is required whenever status != OK
    """
    payload = dict(base_hdr or {})
    payload["policy"] = str(policy)
    payload["verdict_mode"] = "EMBED_ONLY"
    st_norm = str(status or "OK")
    payload["status"] = st_norm
    if st_norm != "OK":
        payload["reason_code"] = str(reason_code or "INTERNAL_ERROR")
        if reason_detail_code is not None:
            payload["reason_detail_code"] = str(reason_detail_code)

    payload["embed"] = {
        "left": {"policy": str(left_policy), "payload_sig8": str(left_payload_sig8 or "")},
        "right": {"policy": str(right_policy), "payload_sig8": str(right_payload_sig8 or "")},
    }

    b3_stamp_payload_sig8(payload)
    return payload

def build_v2_projector_freezer_payload(
    base_hdr: dict,
    *,
    file_pi_valid: bool,
    file_pi_reasons: list[str] | None,
) -> dict:
    """Payload for projector_freezer sidecar cert.

    B3 semantics:
      - This artifact is always written.
      - status reflects whether the freezer payload itself is valid (typically OK).
      - file_pi_valid + file_pi_reasons describe whether projected(FILE) was admissible/usable.
    """
    payload = dict(base_hdr or {})
    payload["policy"] = "projector_freezer"
    payload["status"] = "OK"
    payload["file_pi_valid"] = bool(file_pi_valid)
    payload["file_pi_reasons"] = list(file_pi_reasons or [])

    b3_stamp_payload_sig8(payload)
    return payload

def build_v2_bundle_index_payload(
    *,
    run_id: str,
    # Step 11 (sig8 disambiguation): prefer bundle_sig8; keep legacy sig8 as alias.
    bundle_sig8: str | None = None,
    sig8: str | None = None,
    district_id: str,
    fixture_label: str,
    bundle_dir,
    roles: dict,
    lanes_popcount: int | None,
    lanes_sig8: str | None,
    file_pi_valid: bool,
) -> dict:
    """Bundle index (bundle_index.v2.json / legacy bundle.json) payload.

    Step 4 (B3 structural presence):
      - Projected(FILE) + A/B(FILE) are no longer optional-by-absence.
      - The index always describes all 6 overlap cert artifacts.

    B5 adjunct:
      - If present, the index may also describe the per-fixture B5 identity
        record (b5_identity__*). This is treated as an additional role in the
        closure map, but it is not part of the original "6 certs" contract.

    Step 5/7 (B3 self-ID + closure manifest):
      - bundle_index carries its own payload_sig8 (excluding local absolute paths + run_id).
      - roles{} is the *core* closure map: role -> {filename, expected_payload_sig8, status, reason_*}
      - files{} remains a local absolute-path convenience map (non-core).
    """
    bdir = _Path(bundle_dir)
    # Step 11: disambiguate bundle-level short id.
    bundle_sig8 = str(bundle_sig8 or sig8 or "")

    # Local absolute paths (non-core convenience; export rewrites later).
    files = {
        "strict": str(make_strict_cert_path(bdir, district_id, bundle_sig8)),
        "projected_auto": str(make_projected_auto_cert_path(bdir, district_id, bundle_sig8)),
        "ab_auto": str(bdir / f"ab_compare__strict_vs_projected_auto__{bundle_sig8}.json"),
        "freezer": str(bdir / f"projector_freezer__{district_id}__{bundle_sig8}.json"),
        "projected_file": str(make_projected_file_cert_path(bdir, district_id, bundle_sig8)),
        "ab_file": str(bdir / f"ab_compare__strict_vs_projected_file__{bundle_sig8}.json"),
        # B5 identity record (optional adjunct role; file presence is enforced elsewhere).
        "b5_identity": str(make_b5_identity_path(bdir, district_id, bundle_sig8)),
    }

    # Normalize role meta into a deterministic, schema-stable shape.
    role_order = (
        "strict",
        "projected_auto",
        "ab_auto",
        "freezer",
        "projected_file",
        "ab_file",
        "b5_identity",
    )
    roles_in = roles if isinstance(roles, dict) else {}
    roles_norm: dict = {}
    filenames: list[str] = []

    for r in role_order:
        meta = roles_in.get(r) if isinstance(roles_in, dict) else None
        if not isinstance(meta, dict):
            meta = {}
        fn = str(meta.get("filename") or "")
        if fn:
            filenames.append(fn)
        entry = {
            "filename": fn,
            "expected_payload_sig8": str(meta.get("expected_payload_sig8") or meta.get("payload_sig8") or ""),
            "status": str(meta.get("status") or "OK"),
        }
        rc = meta.get("reason_code")
        if rc is not None:
            entry["reason_code"] = str(rc)
        rd = meta.get("reason_detail_code")
        if rd is not None:
            entry["reason_detail_code"] = str(rd)
        roles_norm[r] = entry

    payload = {
        "run_id": str(run_id or ""),
        # Step 11: explicit naming (preferred).
        "bundle_sig8": bundle_sig8,
        "lanes_popcount": int(lanes_popcount or 0),
        "lanes_sig8": lanes_sig8,
        # Legacy aliases (kept for back-compat; excluded from payload_sig8 hashing).
        "sig8": bundle_sig8,
        "lanes": {"popcount": int(lanes_popcount or 0), "sig8": lanes_sig8},

        "district_id": str(district_id),
        "fixture_label": str(fixture_label or ""),
        # Convenience list (non-core for hashing; roles{} is the closure map).
        "filenames": filenames,
        # Local absolute paths (non-core).
        "files": files,
        # Core closure map (role -> {filename, payload_sig8, status, ...}).
        "roles": roles_norm,
        "file_pi_valid": bool(file_pi_valid),
        "counts": {"written": len(filenames)},
    }

    # Self-ID for the index must be stable across export contexts.
    # Exclude local abs path map + run_id from the hash surface.
    try:
        # Also exclude legacy alias keys ('sig8', 'lanes') so the identity surface
        # is expressed only via explicit bundle_sig8 / lanes_sig8 fields.
        nc = set(_B3_NON_CORE_COMMON_KEYS) | {"files", "run_id", "sig8", "lanes"}
        b3_stamp_payload_sig8(payload, non_core_keys=nc)
    except Exception:
        pass

    return payload

def build_v2_loop_receipt(
    *,
    run_id: str | None,
    district_id: str,
    fixture_label: str,
    # Step 11 (sig8 disambiguation): prefer bundle_sig8; keep legacy sig8 as alias.
    bundle_sig8: str | None = None,
    sig8: str | None = None,
    bundle_dir,
    paths: dict,
    core_written: int,
    # B3 termination witness (Step 8)
    terminal_status: str | None = None,
    termination_reason_code: str | None = None,
    termination_metrics: dict | None = None,
    closure: dict | None = None,
    dims: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a canonical loop_receipt.v2 payload for a single bundle.

    B3 (Termination & loop-closure) requirements:
      • Explicit terminal_status + termination_reason_code
      • Self-ID (payload_sig8) for local content-addressed verification
      • Non-core noise (timestamps / bundle_dir / run_id) excluded from hashing

    NOTE (Step 11): ``bundle_sig8`` is the bundle-level short hash used in the
    on-disk cert neighborhood directory name. ``sig8`` is kept only as a
    legacy alias for back-compat.
    """
    import time as _time

    schema_version = globals().get("SCHEMA_VERSION", "2.0.0")
    engine_rev = globals().get("ENGINE_REV", "rev-UNSET")

    bdir = _Path(bundle_dir) if bundle_dir is not None else None
    # Step 11: disambiguate bundle-level short id.
    bundle_sig8 = str(bundle_sig8 or sig8 or "")

    # Normalize termination fields (keep boring, deterministic defaults).
    ts = str(terminal_status or "TERMINATED_SUCCESS")
    tr = str(termination_reason_code or "SINGLE_PASS")

    receipt: dict = {
        "schema": "loop_receipt.v2",
        "schema_version": str(schema_version),
        "engine_rev": str(engine_rev),
        "run_id": run_id,
        "district_id": str(district_id),
        "fixture_label": str(fixture_label),
        # Step 11: explicit naming (preferred).
        "bundle_sig8": bundle_sig8,
        # Legacy alias (kept for back-compat; excluded from payload_sig8 hashing).
        "sig8": bundle_sig8,
        "bundle_dir": str(bdir.resolve()) if bdir is not None else None,
        "paths": dict(paths or {}),
        "core_counts": {"written": int(core_written)},
        # Step 8: explicit termination witness
        "terminal_status": ts,
        "termination_reason_code": tr,
        "timestamps": {"receipt_written_at": int(_time.time())},
    }

    if isinstance(termination_metrics, dict):
        receipt["termination_metrics"] = dict(termination_metrics)
    if isinstance(closure, dict):
        receipt["closure"] = dict(closure)

    if isinstance(dims, dict):
        receipt["dims"] = {"n2": dims.get("n2"), "n3": dims.get("n3")}

    if isinstance(extra, dict):
        # Only add keys that are not already present to avoid accidental override.
        for k, v in extra.items():
            if k not in receipt:
                receipt[k] = v

    # B3 self-ID: stable across export contexts (exclude run_id and timestamps/bundle_dir).
    try:
        # Also exclude legacy alias key ('sig8') so the identity surface is
        # expressed only via explicit bundle_sig8.
        nc = set(_B3_NON_CORE_COMMON_KEYS) | {"run_id", "sig8"}
        b3_stamp_payload_sig8(receipt, non_core_keys=nc)
    except Exception:
        pass

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
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

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
    bundle_sig8 = (embed_sig_auto or "")[:8] if embed_sig_auto else "00000000"

    # --- Bundle dir ---
    bundle_dir = make_bundle_dir(district_id, fixture_label, bundle_sig8)
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
    # _svr_strict_from_blocks returns {"3": {"eq": True|False|None}, "na_reason_code": <opt>}
    strict_verdict = None
    strict_na_reason_code = None
    if isinstance(strict_out, dict):
        s3 = strict_out.get("3")
        if isinstance(s3, dict):
            strict_verdict = s3.get("eq")
        strict_na_reason_code = strict_out.get("na_reason_code")
    strict_payload = build_v2_strict_cert_payload(base_hdr, strict_verdict, na_reason_code=strict_na_reason_code)
    strict_path = make_strict_cert_path(bundle_dir, district_id, bundle_sig8)
    _write_json(strict_path, strict_payload)
    written.append(strict_path.name)

    # 2) projected(columns@k=3,auto)
    # _svr_projected_auto_from_blocks returns out={"3": {"eq": True|False|None}}
    proj_verdict = None
    if isinstance(proj_out, dict):
        p3 = proj_out.get("3")
        if isinstance(p3, dict):
            proj_verdict = p3.get("eq")
    proj_auto_payload = build_v2_projected_auto_cert_payload(
        base_hdr,
        lanes_vec=lanes_vec,
        lanes_popcount=lanes_pop,
        lanes_sig8=lanes_sig8,
        proj_meta=proj_meta if isinstance(proj_meta, dict) else {},
        verdict=proj_verdict,
    )
    proj_auto_path = make_projected_auto_cert_path(bundle_dir, district_id, bundle_sig8)
    _write_json(proj_auto_path, proj_auto_payload)
    written.append(proj_auto_path.name)

    # 3) ab_compare (strict vs projected_auto)
    strict_payload_sig8 = (strict_payload or {}).get("payload_sig8") or _canon_dump_and_sig8(strict_payload)[1]
    auto_payload_sig8   = (proj_auto_payload or {}).get("payload_sig8") or _canon_dump_and_sig8(proj_auto_payload)[1]
    ab_auto_payload = build_v2_ab_compare_payload(
        base_hdr,
        policy="strict__VS__projected(columns@k=3,auto)",
        left_policy="strict",
        left_payload_sig8=str(strict_payload_sig8),
        right_policy="projected(columns@k=3,auto)",
        right_payload_sig8=str(auto_payload_sig8),
    )
    ab_auto_path = bundle_dir / f"ab_compare__strict_vs_projected_auto__{bundle_sig8}.json"
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
    freezer_path = bundle_dir / f"projector_freezer__{district_id}__{bundle_sig8}.json"
    _write_json(freezer_path, freezer_payload)
    written.append(freezer_path.name)

    # 5) projected(FILE) + A/B(file)
    # B3 structural presence: these files are always written.
    # Semantic optionality is expressed via status + reason_code inside the payloads.
    proj_file_status = "OK" if file_pi_valid else "NOT_ATTEMPTED"
    proj_file_reason_code = None
    proj_file_reason_detail = None
    if not file_pi_valid:
        proj_file_reason_code = "PRECOND_INVALID"
        if file_pi_reasons:
            proj_file_reason_detail = "|".join([str(x) for x in file_pi_reasons])
        else:
            proj_file_reason_detail = "FILE_PI_INVALID"

    proj_file_payload = build_v2_projected_file_cert_payload(
        base_hdr,
        status=proj_file_status,
        reason_code=proj_file_reason_code,
        reason_detail_code=proj_file_reason_detail,
    )
    proj_file_path = make_projected_file_cert_path(bundle_dir, district_id, bundle_sig8)
    _write_json(proj_file_path, proj_file_payload)
    written.append(proj_file_path.name)

    file_payload_sig8 = (proj_file_payload or {}).get("payload_sig8") or _canon_dump_and_sig8(proj_file_payload)[1]

    # Cascade rule: if the right side is NOT_ATTEMPTED, the compare receipt is also NOT_ATTEMPTED.
    ab_file_status = "OK" if proj_file_status == "OK" else "NOT_ATTEMPTED"
    ab_file_reason_code = None
    ab_file_reason_detail = None
    if ab_file_status != "OK":
        ab_file_reason_code = "DEP_NOT_ATTEMPTED" if proj_file_status == "NOT_ATTEMPTED" else "DEP_ERROR"
        ab_file_reason_detail = "projected_file_overlap"

    ab_file_payload = build_v2_ab_compare_payload(
        base_hdr,
        policy="strict__VS__projected(columns@k=3,file)",
        left_policy="strict",
        left_payload_sig8=str(strict_payload_sig8),
        right_policy="projected(columns@k=3,file)",
        right_payload_sig8=str(file_payload_sig8),
        status=ab_file_status,
        reason_code=ab_file_reason_code,
        reason_detail_code=ab_file_reason_detail,
    )
    ab_file_path = bundle_dir / f"ab_compare__strict_vs_projected_file__{bundle_sig8}.json"
    _write_json(ab_file_path, ab_file_payload)
    written.append(ab_file_path.name)

    # 6) B5 per-fixture identity record (semantic FP over strict-core (d3,C3,H2)).
    b5_identity_payload = build_b5_identity_payload(
        snapshot_id=snapshot_id,
        district_id=district_id,
        fixture_label=fixture_label,
        blocks_B=bB,
        blocks_C=bC,
        blocks_H=bH,
        inputs_sig_5=inputs_sig_5,
    )
    b5_identity_path = make_b5_identity_path(bundle_dir, district_id, bundle_sig8)
    _write_json(b5_identity_path, b5_identity_payload)
    written.append(b5_identity_path.name)

# --- bundle_index.v2 / bundle.json ---

    # --- bundle_index.v2 / bundle.json ---
    def _role_meta(payload: dict, filename: str) -> dict:
        m = {
            "filename": str(filename or ""),
            "expected_payload_sig8": (payload or {}).get("payload_sig8"),
            "status": (payload or {}).get("status", "OK"),
        }
        if isinstance(payload, dict):
            rc_ = payload.get("reason_code")
            if rc_ is not None:
                m["reason_code"] = rc_
            rd_ = payload.get("reason_detail_code")
            if rd_ is not None:
                m["reason_detail_code"] = rd_
        return m

    roles = {
        "strict": _role_meta(strict_payload, strict_path.name),
        "projected_auto": _role_meta(proj_auto_payload, proj_auto_path.name),
        "ab_auto": _role_meta(ab_auto_payload, ab_auto_path.name),
        "freezer": _role_meta(freezer_payload, freezer_path.name),
        "projected_file": _role_meta(proj_file_payload, proj_file_path.name),
        "ab_file": _role_meta(ab_file_payload, ab_file_path.name),
        "b5_identity": _role_meta(b5_identity_payload, b5_identity_path.name),
    }

    bundle_idx = build_v2_bundle_index_payload(
        run_id=rc.get("run_id", ""),
        bundle_sig8=bundle_sig8,
        district_id=district_id,
        fixture_label=fixture_label,
        bundle_dir=bundle_dir,
        roles=roles,
        lanes_popcount=lanes_pop,
        lanes_sig8=lanes_sig8,
        file_pi_valid=file_pi_valid,
    )
    # Write canonical bundle index sidecar (v2)
    _write_json(bundle_dir / "bundle_index.v2.json", bundle_idx)
    # Mirror into legacy bundle.json for back-compat
    _write_json(bundle_dir / "bundle.json", bundle_idx)
    # --- loop_receipt.v2 (B3 termination witness; always written) ---
    # This is the per-fixture termination/closure receipt. It is intentionally
    # emitted even for single-pass runs so downstream bundle closure is mechanical.
    lr_paths = {
        "B": str(pB),
        "C": str(pC),
        "H": str(pH),
        "U": str(pU),
    }
    lr_closure = {
        "bundle_index": {
            "filename": "bundle_index.v2.json",
            "payload_sig8": (bundle_idx or {}).get("payload_sig8"),
        },
        "roles_expected": list(roles.keys()),
    }
    loop_receipt = build_v2_loop_receipt(
        run_id=rc.get("run_id", ""),
        district_id=district_id,
        fixture_label=fixture_label,
        bundle_sig8=bundle_sig8,
        bundle_dir=bundle_dir,
        paths=lr_paths,
        core_written=len(roles),
        terminal_status="TERMINATED_SUCCESS",
        termination_reason_code="SINGLE_PASS",
        closure=lr_closure,
    )
    loop_path = make_loop_receipt_path(bundle_dir, fixture_label)
    _write_json(loop_path, loop_receipt)

    # --- Publish anchors expected by UI ---
    ss["last_bundle_dir"]   = str(bundle_dir)
    ss["last_ab_auto_path"] = str(bundle_dir / f"ab_compare__strict_vs_projected_auto__{bundle_sig8}.json")
    ss["last_ab_file_path"] = str(bundle_dir / f"ab_compare__strict_vs_projected_file__{bundle_sig8}.json")
    ss["last_solver_result"] = {"count": len(written)}

    return {
        "bundle_dir": str(bundle_dir),
        "bundle_sig8": bundle_sig8,
        "sig8": bundle_sig8,
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
                    # SSOT: stamp coverage rows with the canonical world snapshot_id.
                    ssot_snapshot_id = _v2_current_world_snapshot_id(strict=False)
                    run_label = None
                    try:
                        run_label = _svr_current_run_label()
                    except Exception:
                        run_label = None
                    cov = {
                        "fixture_label":  fid,
                        "snapshot_id":    ssot_snapshot_id,
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
                                        # Attach run_label as inert metadata if present.
                    if run_label is not None:
                        cov["run_label"] = run_label

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







# --- B1 bundle helpers: paths & canonical source ids ---
def _b1_rel_dirs_for_root(root: _Path) -> dict[str, _Path]:
    """Return canonical child directories for a B1 bundle rooted at ``root``.

    This helper is *pure*: it does not perform any I/O. Callers are responsible
    for creating the returned directories when materializing a bundle tree.
    """
    root = _Path(root)
    certs_root = root / "certs"
    tau_root = root / "tau"
    return {
        "root": root,
        "meta": root / "meta",
        "world": root / "world",
        "manifests": root / "manifests",
        "certs_d4": certs_root / "d4",
        "certs_parity": certs_root / "parity",
        "certs_fixtures": certs_root / "fixtures",
        "coverage": root / "coverage",
        "tau_c2": tau_root / "c2",
        "tau_c3_receipts": tau_root / "c3" / "receipts",
        "tau_c3_derived_worlds": tau_root / "c3" / "derived_worlds",
        "tau_c4": tau_root / "c4",
    }


def _b1_world_snapshot_path(root: _Path) -> _Path:
    """Bundle-relative path for the v2 world snapshot JSON."""
    return _b1_rel_dirs_for_root(root)["world"] / "world_snapshot.v2.json"


def _b1_manifest_v2_suite_path(root: _Path) -> _Path:
    """Bundle-relative path for the v2 full-scope suite manifest JSONL."""
    return _b1_rel_dirs_for_root(root)["manifests"] / "v2_suite_full_scope.jsonl"


def _b1_manifest_tau_c3_path(root: _Path) -> _Path:
    """Bundle-relative path for the Time(τ) C3 manifest JSONL."""
    return _b1_rel_dirs_for_root(root)["manifests"] / "time_tau_c3_manifest_full_scope.jsonl"


def _b1_manifest_tau_c3_receipts_manifest_path(root: _Path) -> _Path:
    """Bundle-relative path for the Time(τ) C3 receipts manifest JSON."""
    return _b1_rel_dirs_for_root(root)["manifests"] / "time_tau_c3_receipts_manifest.json"

def _b1_manifest_tau_c3_derived_worlds_manifest_path(root: _Path) -> _Path:
    """Bundle-relative path for the Time(τ) C3 derived worlds manifest JSON."""
    return _b1_rel_dirs_for_root(root)["manifests"] / "time_tau_c3_derived_worlds_manifest.json"


def _b1_manifest_time_tau_pointer_set_path(root: _Path) -> _Path:
    """Bundle-relative path for the Time(τ) pointer-set manifest JSON."""
    return _b1_rel_dirs_for_root(root)["manifests"] / "time_tau_pointer_set.json"



def _b1_d4_cert_path(root: _Path, snapshot_id: str, sig8: str | None = None) -> _Path:
    """Bundle-relative path for the D4 certificate JSON file.

    If ``sig8`` is provided we use the fully qualified name
    ``d4_certificate__{snapshot_id}__{sig8}.json``; otherwise we fall back to
    ``d4_certificate__{snapshot_id}.json``. This mirrors the on-disk naming
    used by the live pipeline while remaining bundle-root relative.
    """
    d4_dir = _b1_rel_dirs_for_root(root)["certs_d4"]
    if sig8:
        fname = f"d4_certificate__{snapshot_id}__{sig8}.json"
    else:
        fname = f"d4_certificate__{snapshot_id}.json"
    return d4_dir / fname


def _b1_parity_instance_path(root: _Path, snapshot_id: str, sig8: str) -> _Path:
    """Bundle-relative path for a parity instance JSON file (canonical, sig8-qualified)."""
    pdir = _b1_rel_dirs_for_root(root)["certs_parity"]
    fname = f"parity_instance__{snapshot_id}__{sig8}.json"
    return pdir / fname


def _b1_parity_certificate_path(root: _Path, snapshot_id: str, sig8: str) -> _Path:
    """Bundle-relative path for a parity certificate JSON file (canonical, sig8-qualified)."""
    pdir = _b1_rel_dirs_for_root(root)["certs_parity"]
    fname = f"parity_certificate__{snapshot_id}__{sig8}.json"
    return pdir / fname


def _b1_fixture_cert_dir(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    """Bundle-relative directory for a single fixture cert neighborhood."""
    fixtures_dir = _b1_rel_dirs_for_root(root)["certs_fixtures"]
    return fixtures_dir / str(district_id) / str(fixture_label) / str(strict_sig8)


def _b1_loop_receipt_path(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / f"loop_receipt__{fixture_label}.json"


def _b1_strict_cert_path(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / f"overlap__{district_id}__strict__{strict_sig8}.json"


def _b1_projected_auto_cert_path(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / f"overlap__{district_id}__projected_columns_k_3_auto__{strict_sig8}.json"


def _b1_projected_file_cert_path(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / f"overlap__{district_id}__projected_columns_k_3_file__{strict_sig8}.json"


def _b1_ab_compare_auto_path(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / f"ab_compare__strict_vs_projected_auto__{strict_sig8}.json"


def _b1_ab_compare_file_path(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / f"ab_compare__strict_vs_projected_file__{strict_sig8}.json"


def _b1_projector_freezer_path(root: _Path, district_id: str, strict_sig8: str, fixture_label: str | None = None) -> _Path:
    """Path for projector_freezer JSON.

    ``fixture_label`` is optional because the freezer payload itself does not
    depend on it, but placing it under the fixture directory keeps the tree
    uniform. When omitted we still route through the fixture directory, using
    the district_id + strict_sig8 as identity.
    """
    # fixture_label is already part of the directory structure; we keep it in
    # the signature so callers do not accidentally mismatch labels.
    if fixture_label is None:
        raise ValueError("_b1_projector_freezer_path requires fixture_label in B1")
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / f"projector_freezer__{district_id}__{strict_sig8}.json"


def _b1_bundle_index_path(root: _Path, district_id: str, fixture_label: str, strict_sig8: str) -> _Path:
    return _b1_fixture_cert_dir(root, district_id, fixture_label, strict_sig8) / "bundle_index.v2.json"


def _b1_coverage_log_path(root: _Path) -> _Path:
    return _b1_rel_dirs_for_root(root)["coverage"] / "coverage.jsonl"


def _b1_coverage_rollup_path(root: _Path) -> _Path:
    return _b1_rel_dirs_for_root(root)["coverage"] / "coverage_rollup.csv"


def _b1_histograms_v2_path(root: _Path) -> _Path:
    return _b1_rel_dirs_for_root(root)["coverage"] / "histograms_v2.json"


def _b1_tau_c2_paths(root: _Path, snapshot_id: str) -> tuple[_Path, _Path]:
    """Return (jsonl, csv) paths for the C2 local flip sweep for this snapshot."""
    base = _b1_rel_dirs_for_root(root)["tau_c2"]
    return (
        base / f"time_tau_local_flip_sweep__{snapshot_id}.jsonl",
        base / f"time_tau_local_flip_sweep__{snapshot_id}.csv",
    )




def _b1_tau_c4_rollup_paths(root: _Path) -> tuple[_Path, _Path]:
    """Return (jsonl, csv) paths for the C4 τ rollup."""
    base = _b1_rel_dirs_for_root(root)["tau_c4"]
    return (
        base / "time_tau_c3_rollup.jsonl",
        base / "time_tau_c3_rollup.csv",
    )


def _b1_tau_c4_mismatch_paths(root: _Path) -> tuple[_Path, _Path]:
    """Return (jsonl, csv) paths for the C4 τ mismatch log."""
    base = _b1_rel_dirs_for_root(root)["tau_c4"]
    return (
        base / "time_tau_c3_tau_mismatches.jsonl",
        base / "time_tau_c3_tau_mismatches.csv",
    )


def _b1_source_id_from_path(path: str | _Path) -> str:
    """Canonical, environment-independent source_id for a matrix input path.

    Policy (B1 portability discipline):

      - ``source_id`` MUST be repo-root relative (POSIX separators).
      - Host-dependent absolute paths MUST NOT enter bundle identity surfaces.
      - Paths outside the repo root (or containing ``..``) are rejected, because
        leaving this ambiguous would allow non-isomorphic continuations.

    This is used when normalizing the B/C/H/U ``paths.*`` fields for manifests
    and loop_receipts inside a B1 bundle.
    """
    p = _Path(path)

    # Prefer the configured repo root when present.
    try:
        repo_root = _REPO_DIR
    except Exception:
        try:
            repo_root = _REPO_ROOT  # type: ignore[name-defined]
        except Exception:  # pragma: no cover - extremely defensive
            repo_root = _Path(__file__).resolve().parents[1]
    repo_root = _Path(repo_root).resolve()

    # Relative inputs are treated as already repo-root relative.
    if not p.is_absolute():
        if ".." in p.parts:
            raise ValueError(f"_b1_source_id_from_path: illegal '..' in relative path: {p}")
        return p.as_posix()

    # Absolute inputs must be under repo_root.
    try:
        rel = p.resolve().relative_to(repo_root)
    except Exception as exc:
        raise ValueError(
            f"_b1_source_id_from_path: path is not under repo root ({repo_root}): {p}"
        ) from exc

    return rel.as_posix()


def _b1_source_ids_from_paths_map(paths: dict | None) -> dict[str, str | None]:
    """Normalize a mapping of raw paths (B/C/H/U) into canonical source_ids.

    ``paths`` is expected to be a dict like ``{"B": <path>, "C": <path>, ...}``.
    The returned dict has the same keys but with values rewritten using
    :func:`_b1_source_id_from_path`. Missing or falsy inputs are preserved as
    ``None`` so callers can distinguish between "unset" and "set".
    """
    result: dict[str, str | None] = {}
    if not isinstance(paths, dict):
        return result
    for key, raw in paths.items():
        if not raw:
            result[str(key)] = None
        else:
            result[str(key)] = _b1_source_id_from_path(str(raw))
    return result



# --- B1 bundle collectors: read existing logs into an in-memory plan ---


def _bundle_repo_abspath(p: _Path | str) -> _Path:
    """Inverse of _bundle_repo_relative_path for on-disk artifacts.

    If ``p`` is already absolute we return it as a Path. Otherwise we treat it
    as relative to the configured repository root (_REPO_ROOT / _REPO_DIR).
    """
    try:
        root = _REPO_DIR  # prefer configured repo root when available
    except Exception:  # pragma: no cover - extremely defensive
        try:
            root = _REPO_ROOT
        except Exception:
            root = _Path(__file__).resolve().parents[1]
    pp = _Path(p)
    if pp.is_absolute():
        return pp
    return _Path(root) / pp


def _b1_collect_bundle_state(snapshot_id: str | None = None, run_ctx: dict | None = None, *, activate_derived_worlds: bool = False) -> dict:
    """Collect all on-disk artifacts needed for a B1 world bundle.

    This helper is **read-only**: it does not create directories or write any
    files. It inspects the existing v2 / Time(\u03c4) / coverage / D4 artifacts
    under the repo root and returns a nested dict describing:

      - the SSOT snapshot_id,
      - world snapshot path,
      - v2 & \u03c4 C3 manifest paths,
      - coverage + histograms paths (C1),
      - Time(\u03c4) C2/C3/C4 paths,
      - per-fixture cert neighborhoods (strict/projected/loop receipts),
      - D4 certificate path.

    It **assumes** the v2 64\xd7 suite and Time(\u03c4) lab have already been
    run for the current world. If a required artifact is missing, it raises a
    RuntimeError with a short explanatory message. The writer (B1.C) can then
    use this state to materialize the canonical bundle tree without having to
    rediscover paths.
    """
    # First, reuse the existing SSOT logic from build_bundle_manifest_for_snapshot
    # to resolve the snapshot_id and the key repo-relative paths. Any mismatch
    # between the explicit snapshot_id and the SSOT world snapshot is treated
    # as a ValueError there.
    manifest = build_bundle_manifest_for_snapshot(snapshot_id=snapshot_id, run_ctx=run_ctx)
    sid = str(manifest.get("snapshot_id") or "").strip()
    if not sid:
        raise RuntimeError("B1: bundle manifest is missing snapshot_id; run the v2 suite first.")

    # Resolve repo root for convenience.
    try:
        repo_root = _REPO_DIR
    except Exception:
        try:
            repo_root = _REPO_ROOT
        except Exception:
            repo_root = _Path(__file__).resolve().parents[1]

    # --- World snapshot ---

    # Use the existing D4 helper to locate the world snapshot and sanity check
    # its hash, then resolve back to an absolute path.
    ws_digest = _d4_world_snapshot_digest(snapshot_id=sid)
    ws_rel = ws_digest.get("world_snapshot_path")
    if not ws_rel:
        raise RuntimeError("B1: could not resolve world snapshot path for snapshot_id %r." % sid)
    world_snapshot_path = _bundle_repo_abspath(ws_rel)
    if not world_snapshot_path.exists():
        raise RuntimeError(f"B1: world snapshot not found at {world_snapshot_path}.")
    try:
        world_snapshot = _json.loads(world_snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"B1: failed to parse world snapshot JSON at {world_snapshot_path}: {exc}.") from exc

    ws_sid = (
        world_snapshot.get("snapshot_id")
        or world_snapshot.get("world_snapshot_id")
        or world_snapshot.get("id")
    )
    if ws_sid and str(ws_sid) != sid:
        raise RuntimeError(
            f"B1: snapshot_id mismatch between bundle manifest ({sid!r}) and world snapshot ({ws_sid!r})."
        )

    # --- Manifests (v2 + Time(\u03c4) C3) ---

    manifests_section = dict(manifest.get("manifests") or {})
    v2_rel = manifests_section.get("v2_full_scope")
    if not v2_rel:
        raise RuntimeError("B1: bundle manifest is missing manifests.v2_full_scope.")
    # Pointer-gated resolution (Phase 7): require explicit pointer + hash match.
    pointers_section = dict(manifest.get("pointers") or {})
    ptr = pointers_section.get("v2_full_scope_manifest")
    if not isinstance(ptr, dict):
        raise RuntimeError("B1: bundle manifest missing pointers.v2_full_scope_manifest.")
    ptr_kind = str(ptr.get("artifact_kind") or "")
    ptr_rel = str(ptr.get("relpath") or "")
    ptr_sig8 = str(ptr.get("sig8") or "").strip().lower()
    if ptr_kind != "manifest_v2_full_scope":
        raise RuntimeError(
            "B1: pointers.v2_full_scope_manifest.artifact_kind mismatch: "
            f"{ptr_kind!r}"
        )
    if not ptr_rel:
        raise RuntimeError("B1: pointers.v2_full_scope_manifest.relpath missing.")
    if ptr_rel != str(v2_rel):
        raise RuntimeError("B1: pointers.v2_full_scope_manifest.relpath != manifests.v2_full_scope.")
    if not _is_sig8_hex(ptr_sig8):
        raise RuntimeError(f"B1: pointers.v2_full_scope_manifest.sig8 invalid: {ptr_sig8!r}")

    v2_manifest_path = _bundle_repo_abspath(ptr_rel)
    if not v2_manifest_path.exists():
        raise RuntimeError(f"B1: v2 manifest_full_scope.jsonl not found at {v2_manifest_path}.")
    actual_sig8 = _strict_hash_file_sig8(v2_manifest_path)
    if actual_sig8 != ptr_sig8:
        raise RuntimeError(
            "B1: v2 manifest sig8 mismatch: "
            f"pointer={ptr_sig8} actual={actual_sig8}"
        )
    # We keep manifest rows as paths for now; the writer can stream them as needed.
    # NOTE (portability/identity): fields inside manifest rows that look like filesystem paths
    # (e.g. 'bundle_dir', nested 'paths', or any host-absolute paths) are treated as non-authoritative
    # in the bundle regime. Bundle consumers must not dereference host-local paths; instead they should
    # use bundle-relative B1 canonical neighborhoods (certs/fixtures/... and world/manifests/...)
    # and the D4/B6 witnesses.


    c3_rel = manifests_section.get("time_tau_c3")
    tau_c3_manifest_path = None
    if c3_rel:
        tau_c3_manifest_path = _bundle_repo_abspath(c3_rel)
        if not tau_c3_manifest_path.exists():
            raise RuntimeError(
                f"B1: Time(\u03c4) C3 manifest expected at {tau_c3_manifest_path} but file is missing."
            )


    # Time(τ) C3 receipts manifest (explicit inventory; may be validly empty).
    c3_receipts_manifest_rel = manifests_section.get("time_tau_c3_receipts_manifest")
    tau_c3_receipts_manifest_path = None
    if c3_receipts_manifest_rel:
        tau_c3_receipts_manifest_path = _bundle_repo_abspath(c3_receipts_manifest_rel)
        if not tau_c3_receipts_manifest_path.exists():
            raise RuntimeError(
                f"B1: Time(τ) C3 receipts manifest expected at {tau_c3_receipts_manifest_path} but file is missing."
            )
    else:
        raise RuntimeError(
            "B1: bundle manifest is missing manifests.time_tau_c3_receipts_manifest; "
            "run C3 receipts manifest writer before exporting a B1 bundle."
        )

    # --- Coverage (C1) ---

    coverage_section = dict(manifest.get("coverage") or {})
    cov_jsonl_rel = coverage_section.get("coverage_jsonl_path")
    cov_csv_rel = coverage_section.get("coverage_rollup_csv_path")
    if not cov_jsonl_rel or not cov_csv_rel:
        raise RuntimeError("B1: bundle manifest is missing coverage paths; run C1 rollup first.")

    coverage_jsonl_path = _bundle_repo_abspath(cov_jsonl_rel)
    coverage_rollup_csv_path = _bundle_repo_abspath(cov_csv_rel)
    if not coverage_jsonl_path.exists():
        raise RuntimeError(f"B1: coverage.jsonl not found at {coverage_jsonl_path}.")
    if not coverage_rollup_csv_path.exists():
        raise RuntimeError(f"B1: coverage_rollup.csv not found at {coverage_rollup_csv_path}.")
    # histograms_v2.json is not currently referenced from the manifest; we
    # adopt the conventional location under logs/reports/ and treat absence as
    # non-fatal (the B1 checker can tighten this later).
    try:
        reports_dir = _REPORTS_DIR
    except Exception:
        reports_dir = _Path(repo_root) / "logs" / "reports"
    histograms_v2_path = (reports_dir / "histograms_v2.json").resolve()

    # --- Time(\u03c4) C2 / C3 / C4 ---

    time_tau_section = dict(manifest.get("time_tau") or {})

    # C2: local flip sweep (toy). We prefer snapshot-specific filenames when
    # present, otherwise fall back to the generic sweep.
    c2_jsonl_path = None
    c2_csv_path = None
    c2_toy_rel = time_tau_section.get("c2_toy_dir")
    if c2_toy_rel:
        c2_dir = _bundle_repo_abspath(c2_toy_rel)
        sid_suffix = f"__{sid}"
        c2_jsonl_candidate = c2_dir / f"time_tau_local_flip_sweep{sid_suffix}.jsonl"
        c2_csv_candidate = c2_dir / f"time_tau_local_flip_sweep{sid_suffix}.csv"
        if not c2_jsonl_candidate.exists():
            c2_jsonl_candidate = c2_dir / "time_tau_local_flip_sweep.jsonl"
        if not c2_csv_candidate.exists():
            c2_csv_candidate = c2_dir / "time_tau_local_flip_sweep.csv"
        if c2_jsonl_candidate.exists():
            c2_jsonl_path = c2_jsonl_candidate
        if c2_csv_candidate.exists():
            c2_csv_path = c2_csv_candidate

    if not c2_jsonl_path or not c2_csv_path:
        raise RuntimeError(
            "B1: Time(\u03c4) C2 sweep artifacts not found; run the \u03c4 sweep (C2) before exporting a B1 bundle."
        )

    # C3: recompute receipts live under logs/experiments (c3_receipts_dir).
    c3_receipts_rel = time_tau_section.get("c3_receipts_dir")
    tau_c3_receipts_dir = None
    if c3_receipts_rel:
        tau_c3_receipts_dir = _bundle_repo_abspath(c3_receipts_rel)
        if not tau_c3_receipts_dir.exists():
            raise RuntimeError(
                f"B1: Time(\u03c4) C3 receipts directory {tau_c3_receipts_dir} does not exist."
            )

    # Derived worlds directory for C3 mutated worlds, rooted under app/inputs/c3_derived_worlds
    try:
        worlds_rel = DIRS.get("c3_worlds", "app/inputs/c3_derived_worlds")
        tau_c3_worlds_dir = _bundle_repo_abspath(worlds_rel)
    except Exception:
        tau_c3_worlds_dir = (_Path(repo_root) / "app" / "inputs" / "c3_derived_worlds")
    tau_c3_worlds_dir = tau_c3_worlds_dir.resolve()
    if not tau_c3_worlds_dir.exists():
        if activate_derived_worlds:
            raise RuntimeError(
                f"B1: Time(\u03c4) C3 derived worlds directory {tau_c3_worlds_dir} does not exist; "
                "run the C3 recompute/derived world writer before exporting a B1 bundle."
            )
        tau_c3_worlds_dir = None

    if not tau_c3_manifest_path or not tau_c3_receipts_dir or not tau_c3_receipts_manifest_path:
        raise RuntimeError(
            "B1: Time(\u03c4) C3 artifacts incomplete; run C3 recompute/manifest build + receipts manifest writer before exporting a B1 bundle."
        )

    # C4: rollup + mismatch logs live under logs/reports.
    c4_rollup_rel = time_tau_section.get("c4_rollup_path")
    if not c4_rollup_rel:
        raise RuntimeError(
            "B1: bundle manifest is missing time_tau.c4_rollup_path; run the C4 rollup before exporting."
        )
    tau_c4_rollup_jsonl_path = _bundle_repo_abspath(c4_rollup_rel)
    if not tau_c4_rollup_jsonl_path.exists():
        raise RuntimeError(
            f"B1: Time(\u03c4) C4 rollup JSONL not found at {tau_c4_rollup_jsonl_path}."
        )
    # Derive the CSV sibling by convention.
    if tau_c4_rollup_jsonl_path.suffix == ".jsonl":
        tau_c4_rollup_csv_path = tau_c4_rollup_jsonl_path.with_suffix(".csv")
    else:
        tau_c4_rollup_csv_path = tau_c4_rollup_jsonl_path

    tau_c4_mismatches_jsonl_path = (reports_dir / "time_tau_c3_tau_mismatches.jsonl").resolve()
    tau_c4_mismatches_csv_path = (reports_dir / "time_tau_c3_tau_mismatches.csv").resolve()
    if not tau_c4_mismatches_jsonl_path.exists() or not tau_c4_mismatches_csv_path.exists():
        raise RuntimeError(
            "B1: Time(\u03c4) C4 mismatch logs not found; run the C4 mismatch reporter before exporting."
        )

    # --- D4 certificate ---

    certs_section = dict(manifest.get("certs") or {})
    d4_cert_dir_rel = certs_section.get("d4_cert_dir")
    if d4_cert_dir_rel:
        d4_dir = _bundle_repo_abspath(d4_cert_dir_rel)
    else:
        # Fall back to the conventional logs/certs/d4/ layout.
        try:
            certs_root = _CERTS_DIR
        except Exception:
            certs_root = _Path(repo_root) / "logs" / "certs"
        d4_dir = _Path(certs_root) / "d4"
    d4_dir = d4_dir.resolve()
    if not d4_dir.exists():
        raise RuntimeError(f"B1: D4 cert directory not found at {d4_dir}.")
    try:
        d4_cert_path, d4_cert_obj = _d4_resolve_certificate_for_snapshot(
            snapshot_id=sid,
            run_ctx=run_ctx,
        )
    except Exception as exc:
        raise RuntimeError(
            f"B1: failed to resolve D4 certificate for snapshot_id {sid!r}: {exc}"
        ) from exc

    d4_sig8 = str((d4_cert_obj or {}).get("sig8") or "").strip()
    if not d4_sig8:
        raise RuntimeError(f"B1: resolved D4 certificate missing sig8: {d4_cert_path}")


    # --- Parity artifacts (S3) ---
    # Phase-0 wiring: parity is optional unless explicitly requested. When
    # parity artifacts appear present, we resolve them strictly; otherwise we
    # treat parity as absent and bind Track III policy to the non-parity regime.
    parity_state: dict = {}
    try:
        resolved = _parity_try_resolve_pair_for_snapshot(
            snapshot_id=sid,
            run_ctx=run_ctx,
        )
    except Exception as exc:
        raise RuntimeError(f"B1: failed to resolve parity artifacts for snapshot_id {sid!r}: {exc}") from exc

    if resolved is not None:
        p_inst_path, p_inst_obj, p_cert_path, p_cert_obj = resolved
        parity_instance_sig8 = str((p_inst_obj or {}).get("sig8") or "").strip()
        parity_certificate_sig8 = str((p_cert_obj or {}).get("sig8") or "").strip()
        if not parity_instance_sig8 or not parity_certificate_sig8:
            raise RuntimeError(
                "B1: parity artifacts missing sig8 fields "
                f"(instance_sig8={parity_instance_sig8!r}, certificate_sig8={parity_certificate_sig8!r})"
            )
        parity_state = {
            "instance_path": p_inst_path,
            "instance_sig8": parity_instance_sig8,
            "certificate_path": p_cert_path,
            "certificate_sig8": parity_certificate_sig8,
        }

    # --- Per-fixture cert neighborhoods (strict / projected / loop receipts) ---

    # Locate the strict certs root: logs/certs/ (excluding the d4/ subdir).
    certs_root_rel = certs_section.get("strict")
    if not certs_root_rel:
        raise RuntimeError("B1: bundle manifest is missing certs.strict.")
    certs_root = _bundle_repo_abspath(certs_root_rel).resolve()
    if not certs_root.exists():
        raise RuntimeError(f"B1: v2 certs root not found at {certs_root}.")

    # Derive fixture inventory strictly from the v2 manifest (no directory scans).
    try:
        rows = _read_jsonl_strict(v2_manifest_path)
        triplets = _v2_fixture_triplets_from_manifest_rows(rows, snapshot_id=sid)
    except Exception as exc:
        raise RuntimeError(f"B1: failed to derive fixture inventory from v2 manifest: {exc}") from exc

    if not triplets:
        raise RuntimeError(
            f"B1: no v2 fixtures for snapshot_id {sid!r} in manifest at {v2_manifest_path}."
        )

    fixtures: dict[tuple[str, str, str], dict] = {}
    for (district_id, fixture_label, strict_sig8) in triplets:
        bundle_dir = (certs_root / district_id / fixture_label / strict_sig8).resolve()
        loop_receipt_path = make_loop_receipt_path(bundle_dir, fixture_label)
        strict_cert_path = make_strict_cert_path(bundle_dir, district_id, strict_sig8)
        proj_auto_path = make_projected_auto_cert_path(bundle_dir, district_id, strict_sig8)
        proj_file_path = make_projected_file_cert_path(bundle_dir, district_id, strict_sig8)
        ab_auto_path = bundle_dir / f"ab_compare__strict_vs_projected_auto__{strict_sig8}.json"
        ab_file_path = bundle_dir / f"ab_compare__strict_vs_projected_file__{strict_sig8}.json"
        freezer_path = bundle_dir / f"projector_freezer__{district_id}__{strict_sig8}.json"
        bundle_index_path = bundle_dir / "bundle_index.v2.json"
        b5_identity_path = make_b5_identity_path(bundle_dir, district_id, strict_sig8)

        # Required core files (B3 structural presence):
        #   loop_receipt, strict, projected(auto), projected(file),
        #   ab_compare(auto), ab_compare(file), projector_freezer,
        #   bundle_index.v2, b5_identity
        missing_core = []
        for pth, label in [
            (loop_receipt_path, "loop_receipt"),
            (strict_cert_path, "strict_cert"),
            (proj_auto_path, "projected_auto_cert"),
            (proj_file_path, "projected_file_cert"),
            (ab_auto_path, "ab_compare_auto"),
            (ab_file_path, "ab_compare_file"),
            (freezer_path, "projector_freezer"),
            (bundle_index_path, "bundle_index_v2"),
            (b5_identity_path, "b5_identity"),
        ]:
            if not _Path(pth).exists():
                missing_core.append(label)
        if missing_core:
            raise RuntimeError(
                "B1: fixture ({d}, {f}, {s}) is missing required cert files: {m}".format(
                    d=district_id,
                    f=fixture_label,
                    s=strict_sig8,
                    m=", ".join(sorted(set(missing_core))),
                )
            )

        fixtures[(district_id, fixture_label, strict_sig8)] = {
            "district_id": district_id,
            "fixture_label": fixture_label,
            "strict_sig8": strict_sig8,
            "bundle_dir": bundle_dir,
            "loop_receipt_path": loop_receipt_path,
            "strict_cert_path": strict_cert_path,
            "projected_auto_cert_path": proj_auto_path,
            "projected_file_cert_path": proj_file_path,
            "ab_compare_auto_path": ab_auto_path,
            "ab_compare_file_path": ab_file_path,
            "projector_freezer_path": freezer_path if freezer_path.exists() else None,
            "bundle_index_path": bundle_index_path if bundle_index_path.exists() else None,
            "b5_identity_path": b5_identity_path if b5_identity_path.exists() else None,
        }

    # Final B1 bundle state: a simple nested dict keyed by semantic regions.
    state = {
        "snapshot_id": sid,
        "bundle_manifest": manifest,
        "world_snapshot": {
            "path": world_snapshot_path,
        },
        "manifests": {
            "v2_suite_path": v2_manifest_path,
            "tau_c3_manifest_path": tau_c3_manifest_path,
            "tau_c3_receipts_manifest_path": tau_c3_receipts_manifest_path,
        },
        "coverage": {
            "coverage_jsonl_path": coverage_jsonl_path,
            "coverage_rollup_csv_path": coverage_rollup_csv_path,
            "histograms_v2_path": histograms_v2_path if histograms_v2_path.exists() else None,
        },
        "time_tau": {
            "c2": {
                "jsonl_path": c2_jsonl_path,
                "csv_path": c2_csv_path,
            },
            "c3": {
                "manifest_path": tau_c3_manifest_path,
                "receipts_dir": tau_c3_receipts_dir,
                "receipts_manifest_path": tau_c3_receipts_manifest_path,
                "derived_worlds_dir": tau_c3_worlds_dir,
            },
            "c4": {
                "rollup_jsonl_path": tau_c4_rollup_jsonl_path,
                "rollup_csv_path": tau_c4_rollup_csv_path,
                "mismatches_jsonl_path": tau_c4_mismatches_jsonl_path,
                "mismatches_csv_path": tau_c4_mismatches_csv_path,
            },
        },
        "d4": {
            "cert_path": d4_cert_path,
            "sig8": d4_sig8,
        },
        "parity": parity_state,
        "fixtures": fixtures,
        "repo_root": _Path(repo_root).resolve(),
    }
    return state

# --- Single-Path Pipeline Gate Vector (v0, strict) ---

_SINGLE_PATH_STAGE_ORDER_V0_STRICT: tuple[str, ...] = (
    "S0",   # SSOT snapshot binding + world snapshot sanity
    "V2",   # v2 suite manifest + per-fixture cert neighborhoods
    "C1",   # coverage rollup
    "C2",   # Time(τ) C2 sweep
    "C3",   # Time(τ) C3 recompute + receipts manifest (+ derived worlds dir if activated)
    "C4",   # Time(τ) C4 rollup + mismatch logs
    "D4",   # D4 certificate present
    "B1",   # canonical bundle tree materialized under logs/bundle/{sid}__{sig8}/
    "ZIP",  # bundle__{sid}__{sig8}.zip exists
)


def strict_single_path_gate_vector(
    snapshot_id: str | None = None,
    run_ctx: dict | None = None,
    target_stage: str = "ZIP",
    activate_derived_worlds: bool = False,
) -> tuple[bool, str, list[str], list[str]]:
    """Machine-checkable gate vector for the Single-Path Pipeline (v0, strict).

    Returns:
        (ok, stage, missing, mismatches)

    Semantics:
      - `ok` is True iff all checks up to and including `target_stage` pass.
      - `stage` is "OK" on success, otherwise the *first* failing stage code.
      - `missing` is a list of repo-root-relative POSIX paths that were expected
        (by the strict contract) but do not exist on disk.
      - `mismatches` is a list of invariant violations / parse errors / ID
        mismatches. Prefer mismatches for "wrong thing present" vs `missing`
        for "thing absent".

    Discipline:
      - Read-only: this function MUST NOT create directories or write files.
      - SSOT snapshot_id: resolved via `build_bundle_manifest_for_snapshot()`,
        which guards against snapshot_id mismatches (arg/run_ctx vs SSOT).
    """
    # Normalize stage selection.
    stage_order = list(_SINGLE_PATH_STAGE_ORDER_V0_STRICT)
    target = (target_stage or "ZIP").strip().upper()
    if target not in stage_order:
        return (False, "S0", [], [f"strict_single_path_gate_vector: unknown target_stage {target_stage!r}"])

    def _want(stage_code: str) -> bool:
        """Return True iff stage_code is at or before the target stage."""
        try:
            return stage_order.index(stage_code) <= stage_order.index(target)
        except Exception:
            return True

    def _rel(p: _Path | str) -> str:
        """Best-effort repo-relative POSIX path for reporting."""
        try:
            return _bundle_repo_relative_path(p)  # type: ignore[name-defined]
        except Exception:
            try:
                return str(_Path(p).as_posix())
            except Exception:
                return str(p)

    def _fail(stage_code: str, missing: list[str] | None = None, mismatches: list[str] | None = None):
        return (False, stage_code, list(missing or []), list(mismatches or []))

    # --- S0: SSOT snapshot binding + world snapshot sanity ---
    manifest: dict
    try:
        manifest = build_bundle_manifest_for_snapshot(snapshot_id=snapshot_id, run_ctx=run_ctx)
    except Exception as exc:
        # build_bundle_manifest_for_snapshot already encodes the SSOT mismatch policy.
        return _fail("S0", mismatches=[str(exc)])

    sid = str(manifest.get("snapshot_id") or "").strip()
    if not sid:
        return _fail("S0", mismatches=["missing snapshot_id in bundle-manifest builder output"])

    if _want("S0"):
        # Locate and sanity-check the world snapshot for sid.
        try:
            ws_digest = _d4_world_snapshot_digest(snapshot_id=sid)  # type: ignore[name-defined]
        except Exception as exc:
            return _fail("S0", mismatches=[f"failed to compute world snapshot digest: {exc}"])

        ws_rel = None
        try:
            ws_rel = ws_digest.get("world_snapshot_path")
        except Exception:
            ws_rel = None
        if not ws_rel:
            return _fail("S0", mismatches=[f"could not resolve world_snapshot_path for snapshot_id {sid!r}"])

        ws_path = _bundle_repo_abspath(ws_rel)
        if not ws_path.exists():
            return _fail("S0", missing=[_rel(ws_rel)])

        # Snapshot id inside the world snapshot must agree.
        try:
            ws_obj = _json.loads(ws_path.read_text(encoding="utf-8"))  # type: ignore[name-defined]
        except Exception as exc:
            return _fail("S0", mismatches=[f"failed to parse world snapshot JSON at {_rel(ws_path)}: {exc}"])

        ws_sid = (
            (ws_obj.get("snapshot_id") if isinstance(ws_obj, dict) else None)
            or (ws_obj.get("world_snapshot_id") if isinstance(ws_obj, dict) else None)
            or (ws_obj.get("id") if isinstance(ws_obj, dict) else None)
        )
        if ws_sid and str(ws_sid) != sid:
            return _fail(
                "S0",
                mismatches=[f"snapshot_id mismatch: bundle-manifest sid={sid!r} vs world_snapshot sid={ws_sid!r}"],
            )

    # --- V2: v2 suite manifest + per-fixture cert neighborhoods ---
    if _want("V2"):
        manifests_section = dict(manifest.get("manifests") or {})
        v2_rel = manifests_section.get("v2_full_scope")
        if not v2_rel:
            return _fail("V2", mismatches=["bundle manifest missing manifests.v2_full_scope"])

        # Pointer-gated resolution: require an explicit pointer row and verify
        # (sig8, relpath) before reading. No directory scans, no fallback paths.
        pointers_section = dict(manifest.get("pointers") or {})
        ptr = pointers_section.get("v2_full_scope_manifest")
        if not isinstance(ptr, dict):
            return _fail("V2", mismatches=["bundle manifest missing pointers.v2_full_scope_manifest"])

        ptr_kind = str(ptr.get("artifact_kind") or "")
        ptr_rel = str(ptr.get("relpath") or "")
        ptr_sig8 = str(ptr.get("sig8") or "").strip().lower()

        if ptr_kind != "manifest_v2_full_scope":
            return _fail("V2", mismatches=[f"pointers.v2_full_scope_manifest.artifact_kind mismatch: {ptr_kind!r}"])
        if not ptr_rel:
            return _fail("V2", mismatches=["pointers.v2_full_scope_manifest.relpath missing"])
        if ptr_rel != str(v2_rel):
            return _fail("V2", mismatches=["pointers.v2_full_scope_manifest.relpath != manifests.v2_full_scope"])
        if not _is_sig8_hex(ptr_sig8):
            return _fail("V2", mismatches=[f"pointers.v2_full_scope_manifest.sig8 invalid: {ptr_sig8!r}"])

        v2_manifest_path = _bundle_repo_abspath(ptr_rel)
        if not v2_manifest_path.exists():
            return _fail("V2", missing=[_rel(ptr_rel)])
        try:
            actual_sig8 = _strict_hash_file_sig8(v2_manifest_path)
        except Exception as exc:
            return _fail("V2", mismatches=[f"could not hash v2 manifest: {exc}"])
        if actual_sig8 != ptr_sig8:
            return _fail(
                "V2",
                mismatches=[f"v2 manifest sig8 mismatch: pointer={ptr_sig8} actual={actual_sig8}"],
            )

        # Derive the fixture inventory strictly from the manifest (no scans).
        try:
            rows = _read_jsonl_strict(v2_manifest_path)
            triplets = _v2_fixture_triplets_from_manifest_rows(rows, snapshot_id=sid)
        except Exception as exc:
            return _fail("V2", mismatches=[str(exc)])

        if not triplets:
            return _fail("V2", mismatches=[f"no v2 fixtures for snapshot_id {sid!r} in {_rel(ptr_rel)}"])

        # Per-fixture cert neighborhoods: logs/certs/{district}/{fixture}/{sig8}/...
        certs_section = dict(manifest.get("certs") or {})
        certs_root_rel = certs_section.get("strict")
        if not certs_root_rel:
            return _fail("V2", mismatches=["bundle manifest missing certs.strict"])
        certs_root = _Path(_bundle_repo_abspath(certs_root_rel)).resolve()
        if not certs_root.exists():
            return _fail("V2", missing=[_rel(certs_root)])

        missing_paths: list[str] = []
        for (district_id, fixture_label, strict_sig8) in triplets:
            bundle_dir = (certs_root / district_id / fixture_label / strict_sig8).resolve()

            loop_receipt_path = make_loop_receipt_path(bundle_dir, fixture_label)  # type: ignore[name-defined]
            strict_cert_path = make_strict_cert_path(bundle_dir, district_id, strict_sig8)  # type: ignore[name-defined]
            proj_auto_path = make_projected_auto_cert_path(bundle_dir, district_id, strict_sig8)  # type: ignore[name-defined]
            proj_file_path = make_projected_file_cert_path(bundle_dir, district_id, strict_sig8)  # type: ignore[name-defined]
            ab_auto_path = bundle_dir / f"ab_compare__strict_vs_projected_auto__{strict_sig8}.json"
            ab_file_path = bundle_dir / f"ab_compare__strict_vs_projected_file__{strict_sig8}.json"
            freezer_path = bundle_dir / f"projector_freezer__{district_id}__{strict_sig8}.json"
            bundle_index_path = bundle_dir / "bundle_index.v2.json"
            b5_identity_path = make_b5_identity_path(bundle_dir, district_id, strict_sig8)  # type: ignore[name-defined]

            for pth in [
                loop_receipt_path,
                strict_cert_path,
                proj_auto_path,
                proj_file_path,
                ab_auto_path,
                ab_file_path,
                freezer_path,
                bundle_index_path,
                b5_identity_path,
            ]:
                if not _Path(pth).exists():
                    missing_paths.append(_rel(pth))

        if missing_paths:
            return _fail("V2", missing=sorted(set(missing_paths)))

    # --- C1: coverage rollup ---
    if _want("C1"):
        coverage_section = dict(manifest.get("coverage") or {})
        cov_jsonl_rel = coverage_section.get("coverage_jsonl_path")
        cov_csv_rel = coverage_section.get("coverage_rollup_csv_path")
        if not cov_jsonl_rel or not cov_csv_rel:
            return _fail("C1", mismatches=["bundle manifest missing coverage paths"])

        cov_jsonl_path = _bundle_repo_abspath(cov_jsonl_rel)
        cov_csv_path = _bundle_repo_abspath(cov_csv_rel)

        missing_cov: list[str] = []
        if not cov_jsonl_path.exists():
            missing_cov.append(_rel(cov_jsonl_rel))
        if not cov_csv_path.exists():
            missing_cov.append(_rel(cov_csv_rel))
        if missing_cov:
            return _fail("C1", missing=missing_cov)

    # --- C2: Time(τ) C2 sweep artifacts ---
    if _want("C2"):
        time_tau_section = dict(manifest.get("time_tau") or {})
        c2_toy_rel = time_tau_section.get("c2_toy_dir")
        if not c2_toy_rel:
            return _fail("C2", mismatches=["bundle manifest missing time_tau.c2_toy_dir"])

        c2_dir = _bundle_repo_abspath(c2_toy_rel)
        sid_suffix = f"__{sid}"
        c2_jsonl_candidate = c2_dir / f"time_tau_local_flip_sweep{sid_suffix}.jsonl"
        c2_csv_candidate = c2_dir / f"time_tau_local_flip_sweep{sid_suffix}.csv"
        if not c2_jsonl_candidate.exists():
            c2_jsonl_candidate = c2_dir / "time_tau_local_flip_sweep.jsonl"
        if not c2_csv_candidate.exists():
            c2_csv_candidate = c2_dir / "time_tau_local_flip_sweep.csv"

        missing_c2: list[str] = []
        if not c2_jsonl_candidate.exists():
            missing_c2.append(_rel(c2_jsonl_candidate))
        if not c2_csv_candidate.exists():
            missing_c2.append(_rel(c2_csv_candidate))
        if missing_c2:
            return _fail("C2", missing=missing_c2)

    # --- C3: Time(τ) C3 recompute + receipts manifest + derived worlds ---
    if _want("C3"):
        manifests_section = dict(manifest.get("manifests") or {})
        c3_rel = manifests_section.get("time_tau_c3")
        c3_receipts_manifest_rel = manifests_section.get("time_tau_c3_receipts_manifest")
        if not c3_rel:
            return _fail("C3", mismatches=["bundle manifest missing manifests.time_tau_c3"])
        if not c3_receipts_manifest_rel:
            return _fail("C3", mismatches=["bundle manifest missing manifests.time_tau_c3_receipts_manifest"])

        c3_manifest_path = _bundle_repo_abspath(c3_rel)
        c3_receipts_manifest_path = _bundle_repo_abspath(c3_receipts_manifest_rel)

        missing_c3: list[str] = []
        if not c3_manifest_path.exists():
            missing_c3.append(_rel(c3_rel))
        if not c3_receipts_manifest_path.exists():
            missing_c3.append(_rel(c3_receipts_manifest_rel))

        time_tau_section = dict(manifest.get("time_tau") or {})
        c3_receipts_dir_rel = time_tau_section.get("c3_receipts_dir")
        if not c3_receipts_dir_rel:
            return _fail("C3", mismatches=["bundle manifest missing time_tau.c3_receipts_dir"])
        c3_receipts_dir = _bundle_repo_abspath(c3_receipts_dir_rel)
        if not _Path(c3_receipts_dir).exists():
            missing_c3.append(_rel(c3_receipts_dir))

        # Derived worlds directory is optional unless explicitly activated for export.
        if activate_derived_worlds:
            try:
                worlds_rel = DIRS.get("c3_worlds", "app/inputs/c3_derived_worlds")  # type: ignore[name-defined]
            except Exception:
                worlds_rel = "app/inputs/c3_derived_worlds"
            worlds_dir = _bundle_repo_abspath(worlds_rel)
            if not _Path(worlds_dir).exists():
                missing_c3.append(_rel(worlds_dir))

        if missing_c3:
            return _fail("C3", missing=sorted(set(missing_c3)))

    # --- C4: Time(τ) C4 rollup + mismatch logs ---
    if _want("C4"):
        time_tau_section = dict(manifest.get("time_tau") or {})
        c4_rollup_rel = time_tau_section.get("c4_rollup_path")
        if not c4_rollup_rel:
            return _fail("C4", mismatches=["bundle manifest missing time_tau.c4_rollup_path"])
        c4_rollup_jsonl_path = _bundle_repo_abspath(c4_rollup_rel)

        missing_c4: list[str] = []
        if not c4_rollup_jsonl_path.exists():
            missing_c4.append(_rel(c4_rollup_rel))

        # CSV sibling by convention.
        if c4_rollup_jsonl_path.suffix == ".jsonl":
            c4_rollup_csv_path = c4_rollup_jsonl_path.with_suffix(".csv")
        else:
            c4_rollup_csv_path = c4_rollup_jsonl_path
        if not c4_rollup_csv_path.exists():
            missing_c4.append(_rel(c4_rollup_csv_path))

        # Mismatch logs live under logs/reports by convention.
        try:
            reports_dir = _REPORTS_DIR  # type: ignore[name-defined]
        except Exception:
            try:
                repo_root = _REPO_DIR  # type: ignore[name-defined]
            except Exception:
                try:
                    repo_root = _REPO_ROOT  # type: ignore[name-defined]
                except Exception:
                    repo_root = _Path(__file__).resolve().parents[1]
            reports_dir = _Path(repo_root) / "logs" / "reports"

        mism_jsonl = (_Path(reports_dir) / "time_tau_c3_tau_mismatches.jsonl").resolve()
        mism_csv = (_Path(reports_dir) / "time_tau_c3_tau_mismatches.csv").resolve()
        if not mism_jsonl.exists():
            missing_c4.append(_rel(mism_jsonl))
        if not mism_csv.exists():
            missing_c4.append(_rel(mism_csv))

        if missing_c4:
            return _fail("C4", missing=sorted(set(missing_c4)))

    # --- D4: certificate present ---
    if _want("D4"):
        certs_section = dict(manifest.get("certs") or {})
        d4_dir_rel = certs_section.get("d4_cert_dir")
        if d4_dir_rel:
            d4_dir = _bundle_repo_abspath(d4_dir_rel)
        else:
            # Fall back to strict certs root / d4.
            strict_root_rel = certs_section.get("strict")
            if strict_root_rel:
                d4_dir = _bundle_repo_abspath(strict_root_rel) / "d4"
            else:
                try:
                    repo_root = _REPO_DIR  # type: ignore[name-defined]
                except Exception:
                    try:
                        repo_root = _REPO_ROOT  # type: ignore[name-defined]
                    except Exception:
                        repo_root = _Path(__file__).resolve().parents[1]
                d4_dir = _Path(repo_root) / "logs" / "certs" / "d4"

        d4_dir = _Path(d4_dir).resolve()
        if not d4_dir.exists():
            return _fail("D4", missing=[_rel(d4_dir)])

        try:
            _d4_resolve_certificate_for_snapshot(snapshot_id=sid, run_ctx=run_ctx)
        except Exception as exc:
            # A1 discipline: ambiguity is not resolved by heuristics.
            return _fail("D4", mismatches=[str(exc)])

    # --- B1: canonical bundle tree exists (optional / post-export) ---
    # Note: This check is only meaningful if export_bundle_for_snapshot() has been run.
    if _want("B1") or _want("ZIP"):
        # Compute the expected bundle sig8 from the manifest identity surface.
        try:
            stamped = stamp_bundle_manifest_sig8(manifest)
        except Exception as exc:
            return _fail("B1", mismatches=[f"failed to stamp bundle manifest sig8: {exc}"])

        sig8 = str(stamped.get("sig8") or "").strip()
        if not sig8:
            return _fail("B1", mismatches=["stamped bundle manifest produced empty sig8"])

        # Resolve repo root (no mkdir).
        try:
            repo_root = _REPO_DIR  # type: ignore[name-defined]
        except Exception:
            try:
                repo_root = _REPO_ROOT  # type: ignore[name-defined]
            except Exception:
                repo_root = _Path(__file__).resolve().parents[1]
        repo_root = _Path(repo_root).resolve()
        bundle_root = repo_root / "logs" / "bundle"
        bundle_dir = bundle_root / f"{sid}__{sig8}"

        if _want("B1"):
            if not bundle_dir.exists():
                return _fail("B1", missing=[_rel(bundle_dir)])

            # Minimal structural anchors inside the bundle tree.
            required_inside: list[_Path] = []
            # meta/bundle_manifest.json
            required_inside.append(_b1_rel_dirs_for_root(bundle_dir)["meta"] / "bundle_manifest.json")  # type: ignore[name-defined]
            # world snapshot
            required_inside.append(_b1_world_snapshot_path(bundle_dir))  # type: ignore[name-defined]
            # key manifests
            required_inside.append(_b1_manifest_v2_suite_path(bundle_dir))  # type: ignore[name-defined]
            required_inside.append(_b1_manifest_tau_c3_receipts_manifest_path(bundle_dir))  # type: ignore[name-defined]
            required_inside.append(_b1_manifest_time_tau_pointer_set_path(bundle_dir))  # type: ignore[name-defined]
            # coverage
            required_inside.append(_b1_coverage_log_path(bundle_dir))  # type: ignore[name-defined]
            required_inside.append(_b1_coverage_rollup_path(bundle_dir))  # type: ignore[name-defined]
            # tau C2
            try:
                c2j, c2c = _b1_tau_c2_paths(bundle_dir, sid)  # type: ignore[name-defined]
                required_inside.append(_Path(c2j))
                required_inside.append(_Path(c2c))
            except Exception:
                # If helper is missing, fall back to canonical names.
                required_inside.append(bundle_dir / "tau" / "c2" / f"time_tau_local_flip_sweep__{sid}.jsonl")
                required_inside.append(bundle_dir / "tau" / "c2" / f"time_tau_local_flip_sweep__{sid}.csv")
            # tau C4
            try:
                rj, rc = _b1_tau_c4_rollup_paths(bundle_dir)  # type: ignore[name-defined]
                mj, mc = _b1_tau_c4_mismatch_paths(bundle_dir)  # type: ignore[name-defined]
                required_inside.extend([_Path(rj), _Path(rc), _Path(mj), _Path(mc)])
            except Exception:
                required_inside.append(bundle_dir / "tau" / "c4" / "time_tau_c3_rollup.jsonl")
                required_inside.append(bundle_dir / "tau" / "c4" / "time_tau_c3_rollup.csv")
                required_inside.append(bundle_dir / "tau" / "c4" / "time_tau_c3_tau_mismatches.jsonl")
                required_inside.append(bundle_dir / "tau" / "c4" / "time_tau_c3_tau_mismatches.csv")

            missing_b1: list[str] = []
            for p in required_inside:
                if not _Path(p).exists():
                    missing_b1.append(_rel(p))

            # D4 cert inside bundle: at least one matching file.
            d4_in_bundle_dir = bundle_dir / "certs" / "d4"
            if not d4_in_bundle_dir.exists():
                missing_b1.append(_rel(d4_in_bundle_dir))
            else:
                d4_in_bundle = sorted(d4_in_bundle_dir.glob(f"d4_certificate__{sid}*.json"))
                if not d4_in_bundle:
                    missing_b1.append(_rel(d4_in_bundle_dir / f"d4_certificate__{sid}*.json"))

            if missing_b1:
                return _fail("B1", missing=sorted(set(missing_b1)))

        # --- ZIP: exported zip exists (optional / post-export) ---
        if _want("ZIP"):
            zip_path = bundle_root / f"bundle__{sid}__{sig8}.zip"
            if not zip_path.exists():
                return _fail("ZIP", missing=[_rel(zip_path)])

    return (True, "OK", [], [])




# --- B1 bundle writers: materialize canonical bundle tree on disk ---


def _b1_guess_sig8_from_d4_name(name: str) -> str | None:
    """Best-effort extractor for a trailing hex sig8 from a D4 filename.

    Expected shapes (basename without directories)::

        d4_certificate__{snapshot_id}.json
        d4_certificate__{snapshot_id}__{sig8}.json

    where ``snapshot_id`` itself may contain "__" (e.g. ws__...__boot).
    We treat the final "__" component as a potential sig8 and accept it
    iff it looks like 8 hex characters. Otherwise we return None.
    """
    if not name.startswith("d4_certificate__") or not name.endswith(".json"):
        return None
    base = name[len("d4_certificate__") : -5]  # strip prefix + .json
    if "__" not in base:
        return None
    tail = base.split("__")[-1]
    if re.fullmatch(r"[0-9a-fA-F]{8}", tail or ""):
        return tail.lower()
    return None


def _b1_copy_text_file(src: _Path | str, dest: _Path | str, *, missing_ok: bool = False) -> None:
    """Small helper: copy a UTF-8 text file from src to dest.

    - Creates parent directories for dest.
    - If ``missing_ok`` is False and src does not exist, raises RuntimeError.
    """
    src_p = _Path(src)
    dest_p = _Path(dest)
    if not src_p.exists():
        if missing_ok:
            return
        raise RuntimeError(f"B1: expected file not found: {src_p}")
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    dest_p.write_text(src_p.read_text(encoding="utf-8"), encoding="utf-8")


def _b1_write_time_tau_pointer_set_manifest(bundle_dir: _Path, snapshot_id: str, *, include_c3_derived_worlds_manifest: bool = False) -> _Path:
    """Write manifests/time_tau_pointer_set.json inside a bundle (v0.2).

    This is the scan-free root inventory for Time(τ) closure. It lists only
    surface pointers (required for closure) and an optional annex list
    (explicitly ignored by closure).
    """
    bdir = _Path(bundle_dir)
    sid = str(snapshot_id or "").strip()
    if not sid:
        raise RuntimeError("B1: cannot write time_tau_pointer_set.json with empty snapshot_id")

    payload = {
        "schema": TIME_TAU_POINTER_SET_SCHEMA,
        "schema_version": TIME_TAU_POINTER_SET_SCHEMA_VERSION,
        "snapshot_id": sid,
        "surface_pointers": [
            {"role": "c2_sweep_jsonl", "relpath": f"tau/c2/time_tau_local_flip_sweep__{sid}.jsonl"},
            {"role": "c2_sweep_csv", "relpath": f"tau/c2/time_tau_local_flip_sweep__{sid}.csv"},
            {"role": "c3_manifest_full_scope_jsonl", "relpath": "manifests/time_tau_c3_manifest_full_scope.jsonl"},
            {"role": "c4_rollup_jsonl", "relpath": "tau/c4/time_tau_c3_rollup.jsonl"},
            {"role": "c4_mismatches_jsonl", "relpath": "tau/c4/time_tau_c3_tau_mismatches.jsonl"},
        ],
        "annex_pointers": [],
    }


    # Optional activation pointer (derived worlds manifest).
    if include_c3_derived_worlds_manifest:
        payload["surface_pointers"].append(
            {"role": TIME_TAU_PTR_ROLE_C3_DERIVED_WORLDS_MANIFEST, "relpath": TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_REL}
        )

    # Canonical ordering (role, relpath).
    try:
        payload["surface_pointers"] = sorted(
            payload["surface_pointers"],
            key=lambda r: (str((r or {}).get("role") or ""), str((r or {}).get("relpath") or "")),
        )
    except Exception:
        pass

    # v0.2 self-sig8 stamp (hash-body selection quarantines annex_pointers + unknown keys)
    canon_sps: list[dict] = []
    for r in (payload.get("surface_pointers") or []):
        if not isinstance(r, dict):
            continue
        canon_sps.append({"role": str(r.get("role") or ""), "relpath": str(r.get("relpath") or "")})
    canon_sps.sort(key=lambda x: (str((x or {}).get("role") or ""), str((x or {}).get("relpath") or "")))

    hb = {
        "schema": TIME_TAU_POINTER_SET_SCHEMA,
        "schema_version": TIME_TAU_POINTER_SET_SCHEMA_VERSION,
        "snapshot_id": str(payload.get("snapshot_id") or "").strip(),
        "surface_pointers": canon_sps,
        "sig8": "",
    }
    payload["sig8"] = hash_json_sig8(hb)

    outp = _b1_manifest_time_tau_pointer_set_path(bdir)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(canonical_json(payload), encoding="utf-8")
    return outp



def _b1_write_v2_manifest_with_source_ids(src: _Path, dest: _Path) -> None:
    """Rewrite the v2 manifest into the bundle with canonical source_ids.

    We read ``manifest_full_scope.jsonl`` from ``src`` and write a new
    JSONL file to ``dest`` where, for each row, the ``paths`` mapping
    (B/C/H/U) is replaced by a bundle-stable ``source_ids`` map obtained
    via ``_b1_source_ids_from_paths_map``.

    Any non-empty line that fails to parse as JSON results in a RuntimeError.
    """
    src = _Path(src)
    dest = _Path(dest)
    if not src.exists():
        raise RuntimeError(f"B1: v2 manifest source not found at {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fh_in, dest.open("w", encoding="utf-8") as fh_out:
        for raw in fh_in:
            line = raw.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"B1: failed to parse v2 manifest row from {src}: {exc}") from exc
            paths = row.get("paths") or {}
            if isinstance(paths, dict):
                row["paths"] = _b1_source_ids_from_paths_map(paths)
            fh_out.write(_json.dumps(row, separators=(",", ":")) + "\n")


def _b1_materialize_bundle_tree(bdir: _Path, state: dict) -> None:
    """Given a B1 bundle state, materialize the canonical filesystem tree.

    This function is the *only* place that creates files or directories
    under the bundle root for B1. It assumes all paths in ``state`` have
    already been validated by ``_b1_collect_bundle_state`` and will raise
    RuntimeError if any required artifact goes missing between collection
    and write.
    """
    bdir = _Path(bdir)
    dirs = _b1_rel_dirs_for_root(bdir)

    # Ensure the canonical subdirectories exist.
    for p in dirs.values():
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If a directory cannot be created we fail loudly on first write.
            pass

    sid = str(state.get("snapshot_id") or "")

    # --- World snapshot ---
    ws_info = state.get("world_snapshot") or {}
    ws_src = _Path(ws_info.get("path"))
    if not ws_src.exists():
        raise RuntimeError(f"B1: world snapshot source disappeared at {ws_src}.")
    ws_dst = _b1_world_snapshot_path(bdir)
    _b1_copy_text_file(ws_src, ws_dst)

    # --- Manifests (v2 + Time(τ) C3) ---
    manifests = state.get("manifests") or {}
    v2_manifest_src = manifests.get("v2_suite_path")
    if not v2_manifest_src:
        raise RuntimeError("B1: bundle state is missing manifests.v2_suite_path.")
    v2_manifest_dst = _b1_manifest_v2_suite_path(bdir)
    _b1_write_v2_manifest_with_source_ids(_Path(v2_manifest_src), v2_manifest_dst)

    tau_c3_manifest_src = manifests.get("tau_c3_manifest_path")
    if tau_c3_manifest_src:
        tau_c3_manifest_dst = _b1_manifest_tau_c3_path(bdir)
        _b1_copy_text_file(tau_c3_manifest_src, tau_c3_manifest_dst)
    tau_c3_receipts_manifest_src = manifests.get("tau_c3_receipts_manifest_path")
    if tau_c3_receipts_manifest_src:
        tau_c3_receipts_manifest_dst = _b1_manifest_tau_c3_receipts_manifest_path(bdir)
        _b1_copy_text_file(tau_c3_receipts_manifest_src, tau_c3_receipts_manifest_dst)


    # --- Coverage (C1) ---
    cov = state.get("coverage") or {}
    cov_jsonl_src = cov.get("coverage_jsonl_path")
    cov_csv_src = cov.get("coverage_rollup_csv_path")
    if not cov_jsonl_src or not cov_csv_src:
        raise RuntimeError("B1: bundle state is missing coverage paths.")
    cov_jsonl_dst = _b1_coverage_log_path(bdir)
    cov_csv_dst = _b1_coverage_rollup_path(bdir)
    _b1_copy_text_file(cov_jsonl_src, cov_jsonl_dst)
    _b1_copy_text_file(cov_csv_src, cov_csv_dst)

    hist_src = cov.get("histograms_v2_path")
    if hist_src:
        hist_dst = _b1_histograms_v2_path(bdir)
        _b1_copy_text_file(hist_src, hist_dst, missing_ok=True)

    # --- Time(τ) artifacts ---
    tau = state.get("time_tau") or {}

    # C2 sweep (toy)
    c2 = tau.get("c2") or {}
    c2_jsonl_src = c2.get("jsonl_path")
    c2_csv_src = c2.get("csv_path")
    if not c2_jsonl_src or not c2_csv_src:
        raise RuntimeError("B1: bundle state is missing Time(τ) C2 sweep paths.")
    c2_jsonl_dst, c2_csv_dst = _b1_tau_c2_paths(bdir, sid)
    _b1_copy_text_file(c2_jsonl_src, c2_jsonl_dst)
    _b1_copy_text_file(c2_csv_src, c2_csv_dst)

    # C3 receipts + derived worlds
    c3 = tau.get("c3") or {}
    activate_dw = bool(c3.get("derived_worlds_activate"))
    bundle_worlds: list[dict] = []

    c3_receipts_dst_dir = dirs["tau_c3_receipts"]
    c3_receipts_dst_dir.mkdir(parents=True, exist_ok=True)

    # B4: do NOT infer receipts from a directory listing. Use the receipts manifest.
    c3_receipts_manifest_path = (
        c3.get("receipts_manifest_path")
        or (state.get("manifests") or {}).get("tau_c3_receipts_manifest_path")
    )
    if not c3_receipts_manifest_path:
        raise RuntimeError("B1: bundle state is missing Time(τ) C3 receipts_manifest_path.")
    c3_receipts_manifest_path = _Path(c3_receipts_manifest_path)
    if not c3_receipts_manifest_path.exists():
        raise RuntimeError(
            f"B1: Time(τ) C3 receipts manifest {c3_receipts_manifest_path} does not exist."
        )

    try:
        rm_obj = _json.loads(c3_receipts_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"B1: failed to parse Time(τ) C3 receipts manifest at {c3_receipts_manifest_path}: {exc}"
        ) from exc
    if not isinstance(rm_obj, dict):
        raise RuntimeError(
            f"B1: Time(τ) C3 receipts manifest at {c3_receipts_manifest_path} is not a JSON object."
        )

    mode = str(rm_obj.get("mode") or "").strip().lower()
    receipts_list = rm_obj.get("receipts") or []
    if receipts_list is None:
        receipts_list = []
    if not isinstance(receipts_list, list):
        raise RuntimeError("B1: Time(τ) C3 receipts manifest has non-list 'receipts' field.")

    # Consistency guard: if manifest claims "present" but provides no receipts, fail loudly.
    if mode in ("present", "ok") and not receipts_list:
        raise RuntimeError(
            f"B1: receipts manifest claims mode={mode!r} but receipts list is empty at {c3_receipts_manifest_path}."
        )

    # Copy any listed receipts. Empty is legal when the manifest says so.
    bundle_receipts: list[dict] = []
    for r in receipts_list:
        if not isinstance(r, dict):
            continue
        rel = r.get("receipt_relpath") or r.get("relpath") or r.get("path")
        if not rel:
            continue

        # Resolve to an absolute path in the repo.
        try:
            rp = str(rel)
            src_p = _Path(rp) if _Path(rp).is_absolute() else _bundle_repo_abspath(rp)
        except Exception:
            src_p = _Path(rel)

        if not _Path(src_p).exists():
            raise RuntimeError(f"B1: C3 receipt listed in manifest is missing: {src_p}")
        dst = c3_receipts_dst_dir / _Path(src_p).name
        _b1_copy_text_file(src_p, dst)

        # Record bundle-relative relpath + sig8 for the in-bundle receipts manifest.
        try:
            rel_dst = dst.relative_to(bdir).as_posix()
        except Exception:
            rel_dst = f"tau/c3/receipts/{_Path(dst).name}"
        try:
            sig8 = _hash_file_sig8(_Path(dst))
        except Exception:
            sig8 = ""
        bundle_receipts.append({"receipt_relpath": rel_dst, "receipt_sig8": sig8})


    # Rewrite receipts manifest into bundle-relative inventory (v0.2).
    # This is the authoritative enumeration for τ closure inside the bundle.
    try:
        out_rm = {
            "schema": "time_tau_c3_receipts_manifest",
            "schema_version": TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION,
            "snapshot_id": sid,
            "receipts_dir_relpath": "tau/c3/receipts",
            "mode": ("present" if bundle_receipts else "empty"),
            "receipts": sorted(bundle_receipts, key=lambda x: str((x or {}).get("receipt_relpath") or "")),
        }
        rm_outp = _b1_manifest_tau_c3_receipts_manifest_path(bdir)
        rm_outp.parent.mkdir(parents=True, exist_ok=True)
        # v0.2: embed a self-sig8 binding the load-bearing projection
        lb_rm = {
            "schema": "time_tau_c3_receipts_manifest",
            "schema_version": str(out_rm.get("schema_version") or ""),
            "snapshot_id": str(out_rm.get("snapshot_id") or ""),
            "receipts_dir_relpath": str(out_rm.get("receipts_dir_relpath") or ""),
            "mode": str(out_rm.get("mode") or ""),
            "receipts": [],
        }
        for rr in (out_rm.get("receipts") or []):
            if not isinstance(rr, dict):
                continue
            lb_rm["receipts"].append(
                {
                    "receipt_relpath": str(rr.get("receipt_relpath") or ""),
                    "receipt_sig8": str(rr.get("receipt_sig8") or ""),
                }
            )
        lb_rm["receipts"].sort(key=lambda x: str((x or {}).get("receipt_relpath") or ""))
        out_rm["sig8"] = hash_json_sig8(lb_rm)
        rm_outp.write_text(canonical_json(out_rm), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"B1: failed to write bundle-relative C3 receipts manifest: {exc}") from exc

    # Derived worlds payload + manifest (ship-only when activated; pointer-gated identity).
    # Always scrub in-bundle derived-worlds artifacts to avoid stale leakage across runs.
    dw_outp = _b1_manifest_tau_c3_derived_worlds_manifest_path(bdir)
    try:
        if dw_outp.exists():
            dw_outp.unlink()
    except Exception as exc:
        raise RuntimeError(
            f"B1: failed to remove stale Time(τ) C3 derived worlds manifest at {dw_outp}: {exc}"
        ) from exc

    base_dst = dirs["tau_c3_derived_worlds"]
    try:
        if base_dst.exists():
            shutil.rmtree(base_dst)
    except Exception as exc:
        raise RuntimeError(
            f"B1: failed to clear Time(τ) C3 derived worlds destination dir {base_dst}: {exc}"
        ) from exc
    base_dst.mkdir(parents=True, exist_ok=True)

    if activate_dw:
        c3_worlds_dir = c3.get("derived_worlds_dir")
        if not c3_worlds_dir:
            raise RuntimeError(
                "B1: Time(τ) C3 derived worlds activated but state is missing derived_worlds_dir."
            )
        base_src = _Path(c3_worlds_dir)
        if not base_src.exists():
            raise RuntimeError(f"B1: Time(τ) C3 derived worlds dir {base_src} does not exist.")
        for root_str, dirnames, filenames in os.walk(base_src):
            dirnames.sort()
            filenames.sort()
            root_p = _Path(root_str)
            rel = root_p.relative_to(base_src)
            dst_root = base_dst / rel
            dst_root.mkdir(parents=True, exist_ok=True)
            for fname in filenames:
                src_p = root_p / fname
                dst_p = dst_root / fname
                _b1_copy_text_file(src_p, dst_p)
                try:
                    rel_dst = dst_p.relative_to(bdir).as_posix()
                except Exception:
                    rel_dst = ""
                if rel_dst:
                    try:
                        sig8 = _hash_file_sig8(dst_p)
                    except Exception:
                        sig8 = ""
                    bundle_worlds.append({"relpath": rel_dst, "sig8": sig8})

        # Derived worlds manifest (required iff activated; enables pointer-gated τ closure inclusion).
        try:
            out_dw = {
                "schema": TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA,
                "schema_version": TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA_VERSION,
                "snapshot_id": sid,
                "base_dir_relpath": TIME_TAU_C3_DERIVED_WORLDS_BASE_DIR_REL,
                "mode": ("present" if bundle_worlds else "empty"),
                "worlds": sorted(bundle_worlds, key=lambda x: str((x or {}).get("relpath") or "")),
            }
            dw_outp.parent.mkdir(parents=True, exist_ok=True)
            # v0.2: embed a self-sig8 binding the load-bearing projection
            lb_dw = {
                "schema": TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA,
                "schema_version": str(out_dw.get("schema_version") or ""),
                "snapshot_id": str(out_dw.get("snapshot_id") or ""),
                "base_dir_relpath": str(out_dw.get("base_dir_relpath") or ""),
                "mode": str(out_dw.get("mode") or ""),
                "worlds": [],
            }
            for ww in (out_dw.get("worlds") or []):
                if not isinstance(ww, dict):
                    continue
                lb_dw["worlds"].append({"relpath": str(ww.get("relpath") or ""), "sig8": str(ww.get("sig8") or "")})
            lb_dw["worlds"].sort(key=lambda x: str((x or {}).get("relpath") or ""))
            out_dw["sig8"] = hash_json_sig8(lb_dw)
            dw_outp.write_text(canonical_json(out_dw), encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(
                f"B1: failed to write Time(τ) C3 derived worlds manifest: {exc}"
            ) from exc

    # C4 rollup + mismatch logs
    c4 = tau.get("c4") or {}
    c4_rollup_jsonl_src = c4.get("rollup_jsonl_path")
    c4_rollup_csv_src = c4.get("rollup_csv_path")
    c4_mism_jsonl_src = c4.get("mismatches_jsonl_path")
    c4_mism_csv_src = c4.get("mismatches_csv_path")
    if not (c4_rollup_jsonl_src and c4_rollup_csv_src and c4_mism_jsonl_src and c4_mism_csv_src):
        raise RuntimeError("B1: bundle state is missing Time(τ) C4 rollup/mismatch paths.")

    c4_rollup_jsonl_dst, c4_rollup_csv_dst = _b1_tau_c4_rollup_paths(bdir)
    c4_mism_jsonl_dst, c4_mism_csv_dst = _b1_tau_c4_mismatch_paths(bdir)
    _b1_copy_text_file(c4_rollup_jsonl_src, c4_rollup_jsonl_dst)
    _b1_copy_text_file(c4_rollup_csv_src, c4_rollup_csv_dst)
    _b1_copy_text_file(c4_mism_jsonl_src, c4_mism_jsonl_dst)
    _b1_copy_text_file(c4_mism_csv_src, c4_mism_csv_dst)

    # τ pointer-set manifest (bundle-resident; required for scan-free τ closure).
    try:
        _b1_write_time_tau_pointer_set_manifest(bdir, sid, include_c3_derived_worlds_manifest=activate_dw)
    except Exception as exc:
        raise RuntimeError(f"B1: failed to write time_tau_pointer_set.json: {exc}") from exc

    # --- D4 certificate ---
    d4 = state.get("d4") or {}
    d4_path = d4.get("cert_path")
    if not d4_path:
        raise RuntimeError("B1: bundle state is missing D4 certificate path.")
    d4_sig8 = str(d4.get("sig8") or "").strip()
    if not d4_sig8:
        # Back-compat: attempt extraction from filename.
        d4_sig8 = str(_b1_guess_sig8_from_d4_name(_Path(d4_path).name) or "").strip()
    if not d4_sig8:
        raise RuntimeError(f"B1: could not determine D4 cert sig8 for {d4_path!r}")
    d4_dst = _b1_d4_cert_path(bdir, sid, d4_sig8)
    _b1_copy_text_file(d4_path, d4_dst)

    # --- Parity artifacts (S3) ---
    # Phase-0 wiring: parity is optional unless present in the collected state.
    # If present, we copy both artifacts into the canonical bundle neighborhood.
    parity = state.get("parity") if isinstance(state.get("parity"), dict) else {}
    inst_path = parity.get("instance_path") if isinstance(parity, dict) else None
    cert_path = parity.get("certificate_path") if isinstance(parity, dict) else None
    inst_sig8 = str((parity or {}).get("instance_sig8") or "").strip() if isinstance(parity, dict) else ""
    cert_sig8 = str((parity or {}).get("certificate_sig8") or "").strip() if isinstance(parity, dict) else ""

    if inst_path and cert_path and inst_sig8 and cert_sig8:
        inst_dst = _b1_parity_instance_path(bdir, sid, inst_sig8)
        cert_dst = _b1_parity_certificate_path(bdir, sid, cert_sig8)
        _b1_copy_text_file(inst_path, inst_dst)
        _b1_copy_text_file(cert_path, cert_dst)
    elif any([inst_path, cert_path, inst_sig8, cert_sig8]):
        # Partial parity state is never admissible: it would permit a projection fork
        # where bundles silently drift between parity/non-parity regimes.
        raise RuntimeError(
            "B1: partial parity state in bundle materializer (expected both paths + both sig8s or none). "
            f"got instance_path={bool(inst_path)}, certificate_path={bool(cert_path)}, "
            f"instance_sig8={bool(inst_sig8)}, certificate_sig8={bool(cert_sig8)}"
        )

    # --- Per-fixture cert neighborhoods ---
    fixtures = state.get("fixtures") or {}
    if not fixtures:
        raise RuntimeError("B1: bundle state contains no fixture cert neighborhoods.")

    for key in sorted(fixtures.keys()):
        f = fixtures[key]
        district_id = str(f.get("district_id") or key[0])
        fixture_label = str(f.get("fixture_label") or key[1])
        strict_sig8 = str(f.get("strict_sig8") or key[2])

        target_dir = _b1_fixture_cert_dir(bdir, district_id, fixture_label, strict_sig8)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Loop receipt: same schema (loop_receipt.v2) but with paths rewritten
        # to bundle-stable source_ids instead of absolute host paths.
        lr_src = _Path(f.get("loop_receipt_path"))
        if not lr_src.exists():
            raise RuntimeError(f"B1: missing loop_receipt for fixture {key} at {lr_src}.")
        try:
            lr_obj = _json.loads(lr_src.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"B1: failed to parse loop_receipt at {lr_src}: {exc}") from exc
        paths_map = lr_obj.get("paths") or {}
        if isinstance(paths_map, dict):
            lr_obj["paths"] = _b1_source_ids_from_paths_map(paths_map)
        lr_dst = _b1_loop_receipt_path(bdir, district_id, fixture_label, strict_sig8)
        # Re-stamp payload_sig8 because we mutated core fields (paths rewrite).
        try:
            # Keep in sync with build_v2_loop_receipt: exclude legacy alias key 'sig8'.
            nc = set(_B3_NON_CORE_COMMON_KEYS) | {"run_id", "sig8"}
            b3_stamp_payload_sig8(lr_obj, non_core_keys=nc)
        except Exception:
            pass
        _hard_co_write_json(lr_dst, lr_obj)

        # Strict cert (required)
        strict_src = f.get("strict_cert_path")
        if not strict_src:
            raise RuntimeError(f"B1: missing strict_cert_path for fixture {key}.")
        strict_dst = _b1_strict_cert_path(bdir, district_id, fixture_label, strict_sig8)
        _b1_copy_text_file(strict_src, strict_dst)

        # Projected/auto cert (required by current policy)
        proj_auto_src = f.get("projected_auto_cert_path")
        if not proj_auto_src:
            raise RuntimeError(f"B1: missing projected_auto_cert for fixture {key}.")
        proj_auto_dst = _b1_projected_auto_cert_path(bdir, district_id, fixture_label, strict_sig8)
        _b1_copy_text_file(proj_auto_src, proj_auto_dst)

        # Projected/file cert (required; semantic absence is expressed in status/reason_code)
        proj_file_src = f.get("projected_file_cert_path")
        if not proj_file_src:
            raise RuntimeError(f"B1: missing projected_file_cert for fixture {key}.")
        proj_file_dst = _b1_projected_file_cert_path(bdir, district_id, fixture_label, strict_sig8)
        _b1_copy_text_file(proj_file_src, proj_file_dst)


        # A/B comparers (auto + file are both required under B3 structural presence)
        ab_auto_src = f.get("ab_compare_auto_path")
        if not ab_auto_src:
            raise RuntimeError(f"B1: missing ab_compare_auto for fixture {key}.")
        ab_auto_dst = _b1_ab_compare_auto_path(bdir, district_id, fixture_label, strict_sig8)
        _b1_copy_text_file(ab_auto_src, ab_auto_dst)

        ab_file_src = f.get("ab_compare_file_path")
        if not ab_file_src:
            raise RuntimeError(f"B1: missing ab_compare_file for fixture {key}.")
        ab_file_dst = _b1_ab_compare_file_path(bdir, district_id, fixture_label, strict_sig8)
        _b1_copy_text_file(ab_file_src, ab_file_dst)


        # Projector freezer (required)
        freezer_src = f.get("projector_freezer_path")
        if freezer_src:
            freezer_dst = _b1_projector_freezer_path(bdir, district_id, strict_sig8, fixture_label=fixture_label)
            _b1_copy_text_file(freezer_src, freezer_dst)
        else:
            raise RuntimeError(f"B1: missing projector_freezer for fixture {key}.")

        # Bundle index (required)
        bundle_index_src = f.get("bundle_index_path")
        if not bundle_index_src:
            raise RuntimeError(f"B1: missing bundle_index.v2 for fixture {key}.")
        bundle_index_dst = _b1_bundle_index_path(bdir, district_id, fixture_label, strict_sig8)
        try:
            bix_obj = _json.loads(_Path(bundle_index_src).read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"B1: failed to parse bundle_index.v2 at {bundle_index_src}: {exc}") from exc

        # Portability: rewrite local absolute paths to stable relative filenames.
        # (payload_sig8 is stable because 'files' is non-core for hashing.)
        try:
            roles_map = bix_obj.get("roles") or {}
            if isinstance(roles_map, dict):
                files_map: dict[str, str] = {}
                for role, meta in roles_map.items():
                    if isinstance(meta, dict):
                        fn = meta.get("filename")
                        if fn:
                            files_map[str(role)] = str(fn)
                if files_map:
                    bix_obj["files"] = files_map

            # Keep in sync with build_v2_bundle_index_payload: exclude legacy alias
            # keys ('sig8', 'lanes') so the identity surface is expressed only via
            # explicit bundle_sig8 / lanes_sig8 fields.
            nc = set(_B3_NON_CORE_COMMON_KEYS) | {"files", "run_id", "sig8", "lanes"}
            b3_stamp_payload_sig8(bix_obj, non_core_keys=nc)
        except Exception:
            pass

        _hard_co_write_json(bundle_index_dst, bix_obj)

        # B5 per-fixture identity record (required)
        b5_ident_src = f.get("b5_identity_path")
        if not b5_ident_src:
            raise RuntimeError(f"B1: missing b5_identity for fixture {key}.")
        b5_ident_dst = make_b5_identity_path(target_dir, district_id, strict_sig8)
        _b1_copy_text_file(b5_ident_src, b5_ident_dst)



# =================== Phase 3 — Step C (bundle annex wiring) ===================
# Mode B wiring: persist Step‑C artifacts into the B1 bundle tree so exported
# bundles can carry an explicit, reproducible Step‑C surface.
#
# Design constraints:
#   - Must not affect bundle_manifest.sig8 (Step‑C pointers live under manifest.meta)
#   - Must not require any later UI-only helpers to be defined (dependency-light)
#   - Must not change solver semantics; this is packaging only.

_STEP_C_BUNDLE_ANNEX_DIRNAME = "stepc"
_STEP_C_BUNDLE_GLUE_ATTEMPTS_FILENAME = "stepc_glue_attempts.json"
_STEP_C_BUNDLE_STEP_A_INVOCATIONS_FILENAME = "stepc_stepA_invocations.json"
_STEP_C_BUNDLE_PHASE3_REPORT_FILENAME = "phase3_stepc_report.json"
_STEP_C_BUNDLE_TOWER_BARCODE_FILENAME = "stepc_tower_barcode_last.json"
_STEP_C_BUNDLE_INDEX_FILENAME = "stepc_index.json"


# Step‑C annex: additional receipts (Glue profile library + Towers schedule hashes).
_STEP_C_BUNDLE_GLUE_PROFILE_CATALOG_FILENAME = "stepc_glue_profile_catalog.json"
_STEP_C_BUNDLE_GLUE_PROFILE_PROMOTION_REPORT_FILENAME = "stepc_glue_profile_promotion_report.json"
_STEP_C_BUNDLE_TOWER_HASHES_SCHED_ALPHA_FILENAME = "stepc_tower_hashes_sched_alpha.json"
_STEP_C_BUNDLE_TOWER_HASHES_SCHED_BETA_FILENAME = "stepc_tower_hashes_sched_beta.json"
_STEP_C_BUNDLE_TOWER_HASHES_FIRST_DIVERGENCE_FILENAME = "stepc_tower_hashes_first_divergence.json"
_STEP_C_BUNDLE_TOWER_BARCODE_SEMANTICS_FILENAME = "stepc_tower_barcode_semantics_last.json"

# Mirror the Step‑C schema tags used by the UI panel (kept as literals here so
# bundle export can run before the UI panel is executed).
_STEP_C_PROFILE_ID = "Phase3.StepC.GlueAndTowers.v1"
_STEP_C_PHASE3_REPORT_SCHEMA = "phase3_report"
_STEP_C_PHASE3_REPORT_SCHEMA_VERSION = "phase3.report.v1"


def _b1_stepc__shape(mat) -> tuple[int, int]:
    if not isinstance(mat, list):
        raise TypeError("matrix must be a list of rows")
    m = len(mat)
    n = 0
    if m > 0:
        if not isinstance(mat[0], list):
            raise TypeError("matrix row 0 must be a list")
        n = len(mat[0])
    return int(m), int(n)


def _b1_stepc__norm_bitmatrix(mat, *, name: str) -> list[list[int]]:
    if mat is None:
        raise ValueError(f"{name} is None")
    if not isinstance(mat, list):
        raise TypeError(f"{name} must be a list (rows), got {type(mat).__name__}")
    out: list[list[int]] = []
    width: int | None = None
    for r_i, row in enumerate(mat):
        if not isinstance(row, list):
            raise TypeError(f"{name}[{r_i}] must be a list, got {type(row).__name__}")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} is ragged (row {r_i} has len={len(row)}; expected {width})")
        out_row: list[int] = []
        for c_i, v in enumerate(row):
            if v is None:
                raise ValueError(f"{name}[{r_i}][{c_i}] is None")
            if isinstance(v, bool):
                out_row.append(1 if v else 0)
                continue
            if isinstance(v, float):
                raise TypeError(f"{name}[{r_i}][{c_i}] must not be float")
            try:
                iv = int(v)
            except Exception:
                raise TypeError(f"{name}[{r_i}][{c_i}] must be int-like, got {type(v).__name__}")
            out_row.append(iv & 1)
        out.append(out_row)
    return out


def _b1_stepc__norm_bitvector(vec, *, name: str, m_expected: int | None = None) -> list[int]:
    if vec is None:
        raise ValueError(f"{name} is None")
    if not isinstance(vec, list):
        raise TypeError(f"{name} must be a list, got {type(vec).__name__}")
    out: list[int] = []
    for i, v in enumerate(vec):
        if v is None:
            raise ValueError(f"{name}[{i}] is None")
        if isinstance(v, bool):
            out.append(1 if v else 0)
            continue
        if isinstance(v, float):
            raise TypeError(f"{name}[{i}] must not be float")
        try:
            iv = int(v)
        except Exception:
            raise TypeError(f"{name}[{i}] must be int-like, got {type(v).__name__}")
        out.append(iv & 1)
    if m_expected is not None and len(out) != int(m_expected):
        raise ValueError(f"{name} length mismatch: got {len(out)} expected {int(m_expected)}")
    return out


def _b1_stepc__gf2_rref(rows: list[int], n_cols: int) -> tuple[list[int], list[int]]:
    """Reduced row echelon form over F2 for bitmasked rows.

    Each row is an int bitmask with bits 0..n_cols-1 as coefficients.
    Caller may include extra augmented bits above n_cols.
    """
    rr = [int(r) for r in (rows or [])]
    pivots: list[int] = []
    r_i = 0
    m = len(rr)
    for c in range(int(n_cols)):
        # find pivot
        pivot = None
        for k in range(r_i, m):
            if (rr[k] >> c) & 1:
                pivot = k
                break
        if pivot is None:
            continue
        if pivot != r_i:
            rr[r_i], rr[pivot] = rr[pivot], rr[r_i]
        # eliminate column c in all other rows
        for k in range(m):
            if k != r_i and ((rr[k] >> c) & 1):
                rr[k] ^= rr[r_i]
        pivots.append(int(c))
        r_i += 1
        if r_i >= m:
            break
    return rr, pivots


def _b1_stepc_phi_exists(B_I, u_A, u_B) -> bool:
    """Return True iff ∃ϕ with B_I ϕ = u_A + u_B over F2."""
    BI = _b1_stepc__norm_bitmatrix(B_I, name="B_I")
    m, r = _b1_stepc__shape(BI)
    uA = _b1_stepc__norm_bitvector(u_A, name="u_A", m_expected=m)
    uB = _b1_stepc__norm_bitvector(u_B, name="u_B", m_expected=m)
    rhs = [(int(a) ^ int(b)) & 1 for a, b in zip(uA, uB)]
    if m == 0:
        return True
    if r == 0:
        return bool(all((int(b) & 1) == 0 for b in rhs))
    aug_rows: list[int] = []
    for i in range(m):
        mask = 0
        row = BI[i]
        for j in range(r):
            if int(row[j]) & 1:
                mask |= (1 << int(j))
        mask |= ((int(rhs[i]) & 1) << int(r))
        aug_rows.append(int(mask))
    rr, _piv = _b1_stepc__gf2_rref(aug_rows, int(r))
    coeff_mask = (1 << int(r)) - 1
    for row in rr:
        coeff = int(row) & int(coeff_mask)
        rhs_bit = (int(row) >> int(r)) & 1
        if coeff == 0 and rhs_bit == 1:
            return False
    return True


def _b1_stepc__col_type_set(mat: list[list[int]]) -> set[int]:
    """Return set of nonzero column-type bitmasks (rows are bits)."""
    m, n = _b1_stepc__shape(mat)
    types: set[int] = set()
    if m == 0 or n == 0:
        return types
    for j in range(n):
        mask = 0
        for i in range(m):
            if int(mat[i][j]) & 1:
                mask |= (1 << int(i))
        if mask != 0:
            types.add(int(mask))
    return types


def _b1_stepc_new_type(A_before, A_after) -> bool:
    """Return True iff a new nonzero column type appears in A_after not in A_before."""
    A0 = _b1_stepc__norm_bitmatrix(A_before, name="A_before")
    A1 = _b1_stepc__norm_bitmatrix(A_after, name="A_after")
    m0, _n0 = _b1_stepc__shape(A0)
    m1, _n1 = _b1_stepc__shape(A1)
    if m0 != m1:
        raise ValueError(f"new_type: ROW_MISMATCH m0={m0}, m1={m1}")
    t0 = _b1_stepc__col_type_set(A0)
    t1 = _b1_stepc__col_type_set(A1)
    return bool(t1.difference(t0))


def _b1_stepc_phase3_report(glue_attempts: object, invocation_log: object) -> dict:
    """Compute the Step‑C Phase‑3 gate report (dependency-light).

    Pass iff:
      - Glue decisions are reproducible (per-attempt recomputation where inputs present)
      - Step‑A checks are only invoked when no new type holds (invoked ⇒ stepA_applies)
    """
    attempts = list(glue_attempts or []) if isinstance(glue_attempts, list) else []
    inv = list(invocation_log or []) if isinstance(invocation_log, list) else []
    failures: list[dict] = []

    # --- Glue reproducibility / coherence ---
    for idx, att0 in enumerate(attempts):
        att = att0 if isinstance(att0, dict) else {}
        rec = att.get("glue_record") if isinstance(att, dict) else None
        if not isinstance(rec, dict):
            failures.append({"code": "BAD_GLUE_RECORD", "glue_index": int(idx), "msg": "glue_record not a dict"})
            continue

        gid = att.get("glue_id") if isinstance(att, dict) else None

        # Required keys
        for k in ("phi_exists", "new_type", "decision", "stepA_applies"):
            if k not in rec:
                f = {"code": "MISSING_KEY", "glue_index": int(idx), "key": k}
                if gid is not None:
                    f["glue_id"] = gid
                failures.append(f)

        # Derived-field coherence
        try:
            phi_exists = bool(rec.get("phi_exists"))
            dec = rec.get("decision")
            exp_dec = "Cancel" if phi_exists else "Persist"
            if dec != exp_dec:
                f = {"code": "DECISION_COHERENCE", "glue_index": int(idx), "expected": exp_dec, "got": dec}
                if gid is not None:
                    f["glue_id"] = gid
                failures.append(f)
        except Exception as e:
            f = {"code": "DECISION_COHERENCE_ERROR", "glue_index": int(idx), "msg": str(e)}
            if gid is not None:
                f["glue_id"] = gid
            failures.append(f)

        try:
            new_type = bool(rec.get("new_type"))
            stepA_applies = bool(rec.get("stepA_applies"))
            exp = bool(not new_type)
            if stepA_applies != exp:
                f = {"code": "STEP_A_APPLIES_COHERENCE", "glue_index": int(idx), "expected": exp, "got": stepA_applies}
                if gid is not None:
                    f["glue_id"] = gid
                failures.append(f)
        except Exception as e:
            f = {"code": "STEP_A_APPLIES_COHERENCE_ERROR", "glue_index": int(idx), "msg": str(e)}
            if gid is not None:
                f["glue_id"] = gid
            failures.append(f)

        # Recompute primitives from inputs if present
        try:
            if "B_I" in att and "u_A" in att and "u_B" in att:
                exp_phi = _b1_stepc_phi_exists(att.get("B_I"), att.get("u_A"), att.get("u_B"))
                if bool(rec.get("phi_exists")) != bool(exp_phi):
                    f = {"code": "PHI_EXISTS_MISMATCH", "glue_index": int(idx), "expected": bool(exp_phi), "got": rec.get("phi_exists")}
                    if gid is not None:
                        f["glue_id"] = gid
                    failures.append(f)
        except Exception as e:
            f = {"code": "PHI_EXISTS_RECOMPUTE_ERROR", "glue_index": int(idx), "msg": str(e)}
            if gid is not None:
                f["glue_id"] = gid
            failures.append(f)

        try:
            if "A_before" in att and "A_after" in att:
                exp_nt = _b1_stepc_new_type(att.get("A_before"), att.get("A_after"))
                if bool(rec.get("new_type")) != bool(exp_nt):
                    f = {"code": "NEW_TYPE_MISMATCH", "glue_index": int(idx), "expected": bool(exp_nt), "got": rec.get("new_type")}
                    if gid is not None:
                        f["glue_id"] = gid
                    failures.append(f)
        except Exception as e:
            f = {"code": "NEW_TYPE_RECOMPUTE_ERROR", "glue_index": int(idx), "msg": str(e)}
            if gid is not None:
                f["glue_id"] = gid
            failures.append(f)

    # --- Invocation discipline: invoked ⇒ applicable ---
    # Prefer binding invocations to glue attempts (via glue_id) and enforcing
    # the mechanical boundary: invoked ⇒ (no new type).

    # 1) Per-attempt marker discipline: invoked ⇒ (no new type).
    for idx, att0 in enumerate(attempts):
        att = att0 if isinstance(att0, dict) else {}
        rec = att.get("glue_record") if isinstance(att, dict) else None
        if not isinstance(rec, dict):
            continue
        stepA_check = att.get("stepA_check") if isinstance(att, dict) else None
        if not (isinstance(stepA_check, dict) and stepA_check.get("invoked") is True):
            continue
        if bool(rec.get("new_type")):
            f = {"code": "STEP_A_INVOKED_OUT_OF_SCOPE", "glue_index": int(idx), "new_type": True, "stepA_check": stepA_check}
            gid = att.get("glue_id") if isinstance(att, dict) else None
            if gid is not None:
                f["glue_id"] = gid
            failures.append(f)

    # 2) Invocation-log discipline: invoked ⇒ (no new type) when glue binding exists.
    gid_to_new_type: dict[str, bool] = {}
    for att0 in attempts:
        att = att0 if isinstance(att0, dict) else {}
        gid = att.get("glue_id") if isinstance(att, dict) else None
        rec = att.get("glue_record") if isinstance(att, dict) else None
        if gid is None or not isinstance(rec, dict):
            continue
        gid_to_new_type[str(gid)] = bool(rec.get("new_type"))

    for ev0 in inv:
        if not isinstance(ev0, dict):
            continue
        if ev0.get("event") != "stepA_parity_check_invoked":
            continue

        gid = ev0.get("glue_id")
        if gid is not None and str(gid) in gid_to_new_type:
            if gid_to_new_type.get(str(gid)) is True:
                failures.append({"code": "STEP_A_INVOKED_OUT_OF_SCOPE", "event": ev0, "new_type": True})
        else:
            # Fallback: trust the event's own applicability flag when no glue binding exists.
            if ev0.get("stepA_applies") is not True:
                failures.append({"code": "STEP_A_INVOKED_OUT_OF_SCOPE", "event": ev0})

    n_inv = int(len([e for e in inv if isinstance(e, dict) and e.get("event") == "stepA_parity_check_invoked"]))
    phase3_pass = bool(len(failures) == 0)
    return {
        "schema": _STEP_C_PHASE3_REPORT_SCHEMA,
        "schema_version": _STEP_C_PHASE3_REPORT_SCHEMA_VERSION,
        "profile_id": _STEP_C_PROFILE_ID,
        "phase3_pass": bool(phase3_pass),
        "n_glues": int(len(attempts)),
        "n_invocations": int(n_inv),
        "failures": failures,
    }



# =================== Phase 3 — Step C (Glue profile library + promotion report) ===================
# Mode B wiring (additive): controlled library of glue-decision profiles and a promotion-style
# reproducibility report. This MUST NOT alter the frozen Gate‑C decision test; it is purely an
# additional observational surface and optional annex artifact family.

_STEP_C_GLUE_PROFILE_CATALOG_SCHEMA = "stepc_glue_profile_catalog"
_STEP_C_GLUE_PROFILE_CATALOG_SCHEMA_VERSION = "stepc.glue_profile_catalog.v1"

_STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA = "stepc_glue_profile_receipt"
_STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION = "stepc.glue_profile_receipt.v1"

_STEP_C_GLUE_PROFILE_PROMOTION_REPORT_SCHEMA = "stepc_glue_profile_promotion_report"
_STEP_C_GLUE_PROFILE_PROMOTION_REPORT_SCHEMA_VERSION = "stepc.glue_profile_promotion_report.v1"

_STEP_C_GLUE_PROFILE_PROMOTION_PROFILE_ID = "Phase3.StepC.GlueProfiles.Promotion.v1"

# Named glue profiles (additive library). G0 matches the frozen Gate‑C decision test.
_STEP_C_GLUE_PROFILE_G0_ID = "G0"
_STEP_C_GLUE_PROFILE_G1_ID = "G1"


def _b1_stepc__decision_from_phi_exists(phi_exists: bool) -> str:
    """Gate‑C decision mapping (frozen): Cancel iff phi exists else Persist."""
    return "Cancel" if bool(phi_exists) else "Persist"


def _b1_stepc_glue_profile_catalog() -> dict:
    """Return the controlled glue profile catalog (additive; profile ids are stable)."""
    return {
        "schema": _STEP_C_GLUE_PROFILE_CATALOG_SCHEMA,
        "schema_version": _STEP_C_GLUE_PROFILE_CATALOG_SCHEMA_VERSION,
        "profile_id": _STEP_C_PROFILE_ID,
        "profiles": [
            {
                "profile_id": _STEP_C_GLUE_PROFILE_G0_ID,
                "kind": "GLOBAL",
                "label": "G0: global φ (frozen Gate‑C decision semantics)",
                "params_schema": None,
                "semantics": "phi_exists := solvable(B_I φ = u_A + u_B) over F2 (global φ).",
            },
            {
                "profile_id": _STEP_C_GLUE_PROFILE_G1_ID,
                "kind": "MASK",
                "label": "G1: masked support (restricted φ variables by column mask)",
                "params_schema": {"mask": "list[int] (0-based column indices into B_I)"},
                "semantics": "phi_exists := solvable(B_I[:,mask] φ_mask = u_A + u_B) over F2 (restricted support).",
            },
        ],
    }


def _b1_stepc__mask_from_any(mask_obj, *, r_expected: int | None = None) -> tuple[bool, list[int] | None, str | None]:
    """Parse/validate a mask object into a canonical sorted unique list[int]."""
    if mask_obj is None:
        return False, None, "MISSING_MASK"
    mask_list = None
    if isinstance(mask_obj, str):
        s = str(mask_obj).strip()
        if not s:
            return False, None, "EMPTY_MASK"
        parts = [p.strip() for p in s.split(",") if p.strip()]
        try:
            mask_list = [int(p) for p in parts]
        except Exception:
            return False, None, "BAD_MASK_FORMAT"
    elif isinstance(mask_obj, list):
        try:
            mask_list = [int(x) for x in mask_obj]
        except Exception:
            return False, None, "BAD_MASK_FORMAT"
    else:
        return False, None, "BAD_MASK_TYPE"

    # Canonicalize: unique + sorted.
    try:
        mask_can = sorted(set(int(x) for x in mask_list))
    except Exception:
        return False, None, "BAD_MASK_FORMAT"

    if r_expected is not None:
        for j in mask_can:
            if j < 0 or j >= int(r_expected):
                return False, mask_can, "BAD_MASK_INDEX"
    return True, mask_can, None


def _b1_stepc_phi_exists_mask(B_I, u_A, u_B, *, mask) -> tuple[bool, bool | None, str | None, list[int] | None]:
    """Restricted-support solvability: keep only masked columns of B_I."""
    try:
        BI = _b1_stepc__norm_bitmatrix(B_I, name="B_I")
        m, r = _b1_stepc__shape(BI)
    except Exception as exc:
        return True, None, f"BAD_BI: {exc}", None

    ok_mask, mask_can, why = _b1_stepc__mask_from_any(mask, r_expected=r)
    if not ok_mask:
        # mask present but invalid -> ERROR (applicable, invoked)
        return True, None, str(why or "BAD_MASK"), (mask_can if isinstance(mask_can, list) else None)

    # Reduce columns (deterministic ordering by sorted mask_can)
    try:
        BI2 = [[int(row[j]) & 1 for j in mask_can] for row in BI]
    except Exception as exc:
        return True, None, f"MASK_APPLY_ERROR: {exc}", mask_can

    try:
        phi_exists = _b1_stepc_phi_exists(BI2, u_A, u_B)
    except Exception as exc:
        return True, None, f"SOLVE_ERROR: {exc}", mask_can

    return True, bool(phi_exists), None, mask_can


def _b1_stepc_build_glue_profile_receipt_for_attempt(
    attempt: dict,
    *,
    profile_id: str,
    profile_mask=None,
) -> dict:
    """Build a GlueProfileDecisionReceipt for one glue attempt (additive)."""
    att = attempt if isinstance(attempt, dict) else {}
    BI = att.get("B_I")
    uA = att.get("u_A")
    uB = att.get("u_B")

    # Applicability: requires interface inputs; MASK also requires a mask.
    has_iface = BI is not None and uA is not None and uB is not None

    if str(profile_id) == _STEP_C_GLUE_PROFILE_G0_ID:
        applicable = bool(has_iface)
        invoked = bool(applicable)
        if not applicable:
            return {
                "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
                "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
                "profile_id": str(profile_id),
                "invoked": False,
                "applicable": False,
                "status": "NOT_APPLICABLE",
                "phi_exists": None,
                "decision": None,
                "params": None,
                "reason": "MISSING_INTERFACE_INPUTS",
            }
        try:
            # Recompute from raw inputs (do not depend on UI cached glue_record).
            phi_exists = bool(_b1_stepc_phi_exists(BI, uA, uB))
            return {
                "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
                "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
                "profile_id": str(profile_id),
                "invoked": invoked,
                "applicable": applicable,
                "status": "OK",
                "phi_exists": bool(phi_exists),
                "decision": _b1_stepc__decision_from_phi_exists(bool(phi_exists)),
                "params": None,
                "reason": None,
            }
        except Exception as exc:
            return {
                "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
                "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
                "profile_id": str(profile_id),
                "invoked": invoked,
                "applicable": applicable,
                "status": "ERROR",
                "phi_exists": None,
                "decision": None,
                "params": None,
                "reason": f"SOLVE_ERROR: {exc}",
            }

    if str(profile_id) == _STEP_C_GLUE_PROFILE_G1_ID:
        # Mask profile: requires mask to be supplied.
        if profile_mask is None:
            return {
                "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
                "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
                "profile_id": str(profile_id),
                "invoked": False,
                "applicable": False,
                "status": "NOT_APPLICABLE",
                "phi_exists": None,
                "decision": None,
                "params": None,
                "reason": "MISSING_MASK",
            }

        applicable = bool(has_iface)
        invoked = bool(applicable)
        if not applicable:
            return {
                "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
                "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
                "profile_id": str(profile_id),
                "invoked": False,
                "applicable": False,
                "status": "NOT_APPLICABLE",
                "phi_exists": None,
                "decision": None,
                "params": {"mask": profile_mask},
                "reason": "MISSING_INTERFACE_INPUTS",
            }

        ok_applicable, phi_exists, reason, mask_can = _b1_stepc_phi_exists_mask(BI, uA, uB, mask=profile_mask)
        # ok_applicable is always True here (mask present), kept for symmetry.
        if phi_exists is None:
            return {
                "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
                "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
                "profile_id": str(profile_id),
                "invoked": True,
                "applicable": True,
                "status": "ERROR",
                "phi_exists": None,
                "decision": None,
                "params": {"mask": mask_can if mask_can is not None else profile_mask},
                "reason": str(reason or "ERROR"),
            }

        return {
            "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
            "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
            "profile_id": str(profile_id),
            "invoked": True,
            "applicable": True,
            "status": "OK",
            "phi_exists": bool(phi_exists),
            "decision": _b1_stepc__decision_from_phi_exists(bool(phi_exists)),
            "params": {"mask": mask_can if mask_can is not None else profile_mask},
            "reason": None,
        }

    # Unknown profile id
    return {
        "schema": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA,
        "schema_version": _STEP_C_GLUE_PROFILE_RECEIPT_SCHEMA_VERSION,
        "profile_id": str(profile_id),
        "invoked": False,
        "applicable": False,
        "status": "ERROR",
        "phi_exists": None,
        "decision": None,
        "params": None,
        "reason": "UNKNOWN_PROFILE_ID",
    }


def _b1_stepc_compute_glue_profile_receipts_for_attempt(
    attempt: dict,
    *,
    profile_catalog: dict,
    profile_mask=None,
) -> dict[str, dict]:
    """Compute receipts for all catalog profiles for a single attempt."""
    out: dict[str, dict] = {}
    profs = (profile_catalog or {}).get("profiles") if isinstance(profile_catalog, dict) else None
    if not isinstance(profs, list):
        return out
    for p in profs:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("profile_id") or "")
        if not pid:
            continue
        r = _b1_stepc_build_glue_profile_receipt_for_attempt(attempt, profile_id=pid, profile_mask=profile_mask)
        out[pid] = r
    return out


def _b1_stepc_glue_profile_promotion_report(
    glue_attempts: object,
    invocation_log: object,
    *,
    profile_catalog: dict | None = None,
    profile_mask=None,
) -> dict:
    """Promotion-style report: recompute profile decisions and check reproducibility."""
    attempts = glue_attempts if isinstance(glue_attempts, list) else []
    inv = invocation_log if isinstance(invocation_log, list) else []

    cat = profile_catalog if isinstance(profile_catalog, dict) else _b1_stepc_glue_profile_catalog()
    profs = (cat or {}).get("profiles") if isinstance(cat, dict) else []
    prof_ids = [str(p.get("profile_id") or "") for p in profs if isinstance(p, dict) and str(p.get("profile_id") or "")]
    prof_ids = [pid for pid in prof_ids if pid]

    # Phase‑3 baseline is a prerequisite for promotion (promotion_pass ⇒ phase3_pass).
    try:
        phase3_report = _b1_stepc_phase3_report(attempts, inv)
    except Exception as exc:
        phase3_report = {
            "schema": _STEP_C_PHASE3_REPORT_SCHEMA,
            "schema_version": _STEP_C_PHASE3_REPORT_SCHEMA_VERSION,
            "profile_id": _STEP_C_PROFILE_ID,
            "phase3_pass": False,
            "n_glues": int(len(attempts)),
            "n_invocations": int(len(inv)),
            "failures": [{"code": "PHASE3_REPORT_BUILD_ERROR", "msg": str(exc)}],
        }

    failures: list[dict] = []
    per_profile_stats: dict[str, dict] = {}
    for pid in prof_ids:
        per_profile_stats[pid] = {"profile_id": pid, "profile_pass": True, "n_attempts_checked": 0, "n_failures": 0}

    # Re-run reproducibility check: compute twice per attempt/profile.
    for idx, att0 in enumerate(attempts):
        att = att0 if isinstance(att0, dict) else {}
        gid = att.get("glue_id") if isinstance(att, dict) else None
        for pid in prof_ids:
            # Compute twice (same inputs) and compare projected fields.
            r1 = _b1_stepc_build_glue_profile_receipt_for_attempt(att, profile_id=pid, profile_mask=profile_mask)
            r2 = _b1_stepc_build_glue_profile_receipt_for_attempt(att, profile_id=pid, profile_mask=profile_mask)

            # Count checks when applicable.
            if isinstance(r1, dict) and r1.get("applicable") is True:
                per_profile_stats[pid]["n_attempts_checked"] += 1

            def _proj(rr: dict) -> tuple:
                return (
                    rr.get("invoked"),
                    rr.get("applicable"),
                    rr.get("status"),
                    rr.get("phi_exists"),
                    rr.get("decision"),
                )

            if not (isinstance(r1, dict) and isinstance(r2, dict)):
                failures.append({"code": "PROFILE_RECEIPT_BUILD_ERROR", "glue_index": int(idx), "profile_id": pid})
                per_profile_stats[pid]["profile_pass"] = False
                per_profile_stats[pid]["n_failures"] += 1
                continue

            if _proj(r1) != _proj(r2):
                failures.append(
                    {
                        "code": "PROFILE_DECISION_NOT_REPRODUCIBLE",
                        "glue_index": int(idx),
                        "profile_id": pid,
                        "glue_id": gid,
                        "r1": {"status": r1.get("status"), "phi_exists": r1.get("phi_exists"), "decision": r1.get("decision")},
                        "r2": {"status": r2.get("status"), "phi_exists": r2.get("phi_exists"), "decision": r2.get("decision")},
                    }
                )
                per_profile_stats[pid]["profile_pass"] = False
                per_profile_stats[pid]["n_failures"] += 1
                continue

            # Discipline: invoked ⇒ applicable
            if r1.get("invoked") is True and r1.get("applicable") is not True:
                failures.append({"code": "PROFILE_INVOKED_OUT_OF_SCOPE", "glue_index": int(idx), "profile_id": pid, "glue_id": gid, "receipt": r1})
                per_profile_stats[pid]["profile_pass"] = False
                per_profile_stats[pid]["n_failures"] += 1

    # Summarize
    profiles_summary = [per_profile_stats[pid] for pid in prof_ids]
    profiles_summary.sort(key=lambda x: str(x.get("profile_id") or ""))

    phase3_pass = bool((phase3_report or {}).get("phase3_pass")) if isinstance(phase3_report, dict) else False
    profiles_pass = bool(all(bool(x.get("profile_pass")) for x in profiles_summary)) if profiles_summary else True
    promotion_pass = bool(phase3_pass and profiles_pass and (len(failures) == 0))

    return {
        "schema": _STEP_C_GLUE_PROFILE_PROMOTION_REPORT_SCHEMA,
        "schema_version": _STEP_C_GLUE_PROFILE_PROMOTION_REPORT_SCHEMA_VERSION,
        "profile_id": _STEP_C_GLUE_PROFILE_PROMOTION_PROFILE_ID,
        "promotion_pass": bool(promotion_pass),
        "phase3_pass": bool(phase3_pass),
        "n_glues": int(len(attempts)),
        "n_invocations": int(len([e for e in inv if isinstance(e, dict) and e.get("event") == "stepA_parity_check_invoked"])),
        "profiles": profiles_summary,
        "failures": failures,
        # payload
        "catalog": cat,
    }


def _b1_stepc_first_divergence_mapping(barcode_a: object, barcode_b: object) -> dict:
    """Compute first-divergence mapping between two barcode sequences (strict)."""
    a = barcode_a if isinstance(barcode_a, dict) else {}
    b = barcode_b if isinstance(barcode_b, dict) else {}
    a_seq = a.get("sequence") if isinstance(a, dict) else None
    b_seq = b.get("sequence") if isinstance(b, dict) else None
    a_seq = a_seq if isinstance(a_seq, list) else []
    b_seq = b_seq if isinstance(b_seq, list) else []

    a_prof = str(a.get("profile_id") or "")
    b_prof = str(b.get("profile_id") or "")
    a_kind = str(a.get("barcode_kind") or "")
    b_kind = str(b.get("barcode_kind") or "")

    mismatch = (a_prof != b_prof) or (a_kind != b_kind)
    prof = a_prof if not mismatch else "MISMATCH"
    kind = a_kind if not mismatch else "MISMATCH"

    def tok(seq, i: int):
        if i < 0 or i >= len(seq):
            return (None, None)
        row = seq[i] if isinstance(seq[i], dict) else {}
        lvl = row.get("level")
        hh = row.get("hash") if row.get("status") == "OK" else None
        return (lvl, hh)

    L = min(len(a_seq), len(b_seq))
    first = None
    for i in range(L):
        if tok(a_seq, i) != tok(b_seq, i):
            first = int(i)
            break
    if first is None and len(a_seq) != len(b_seq):
        first = int(L)

    a_lvl, a_h = tok(a_seq, first) if first is not None else (None, None)
    b_lvl, b_h = tok(b_seq, first) if first is not None else (None, None)

    return {
        "profile_id": "Towers.FirstDivergence.v0.1",
        "barcode_profile_id": prof,
        "barcode_kind": kind,
        "a_L": int(a.get("L") or 0) if isinstance(a, dict) else 0,
        "b_L": int(b.get("L") or 0) if isinstance(b, dict) else 0,
        "first_divergence_index": first,
        "a_at_divergence": {"level": a_lvl, "hash_or_None": a_h},
        "b_at_divergence": {"level": b_lvl, "hash_or_None": b_h},
        # payload
        "mismatch": bool(mismatch),
    }




def _b1_stepc_write_annex_from_session(bundle_dir: _Path, snapshot_id: str, manifest_sig8: str | None = None) -> dict | None:
    """Write Step‑C annex into bundle/meta/stepc using Streamlit session_state (if present).

    Returns an index dict when written; otherwise returns None.
    """
    try:
        import streamlit as st  # type: ignore
        ss = st.session_state
    except Exception:
        return None

    glue_attempts = ss.get("_stepC_glue_attempts") or []
    invocations = ss.get("_stepC_stepA_invocations") or []
    tower_last = ss.get("_stepC_tower_barcode_last")
    phase3_last = ss.get("_stepC_phase3_report_last")

    # Additive artifacts (may be absent when UI panel not executed).
    profile_mask = ss.get("_stepC_glue_profile_mask")
    promotion_last = ss.get("_stepC_glue_profile_promotion_report_last")

    tower_hash_alpha_last = ss.get("_stepC_tower_hashes_sched_alpha_last")
    tower_hash_beta_last = ss.get("_stepC_tower_hashes_sched_beta_last")
    tower_hash_div_last = ss.get("_stepC_tower_hashes_first_divergence_last")
    tower_sem_last = ss.get("_stepC_tower_barcode_semantics_last")

    has_any = (
        bool(glue_attempts)
        or bool(invocations)
        or (tower_last is not None)
        or (phase3_last is not None)
        or (profile_mask is not None)
        or (promotion_last is not None)
        or (tower_hash_alpha_last is not None)
        or (tower_hash_beta_last is not None)
        or (tower_hash_div_last is not None)
        or (tower_sem_last is not None)
    )
    if not has_any:
        return None

    # Compute fresh gate report for bundle sealing (do not depend on UI button press).
    try:
        phase3_report = _b1_stepc_phase3_report(glue_attempts, invocations)
    except Exception as exc:
        phase3_report = {
            "schema": _STEP_C_PHASE3_REPORT_SCHEMA,
            "schema_version": _STEP_C_PHASE3_REPORT_SCHEMA_VERSION,
            "profile_id": _STEP_C_PROFILE_ID,
            "phase3_pass": False,
            "n_glues": int(len(glue_attempts) if isinstance(glue_attempts, list) else 0),
            "n_invocations": int(len(invocations) if isinstance(invocations, list) else 0),
            "failures": [{"code": "PHASE3_REPORT_BUILD_ERROR", "msg": str(exc)}],
        }

    # Prefer last computed report as payload, but keep the fresh one as the sealed report.
    if isinstance(phase3_last, dict) and phase3_last:
        phase3_report_payload = phase3_last
    else:
        phase3_report_payload = None

    # Glue profile catalog is static; promotion report is computed fresh for sealing.
    profile_catalog = _b1_stepc_glue_profile_catalog()
    try:
        promotion_report = _b1_stepc_glue_profile_promotion_report(
            glue_attempts,
            invocations,
            profile_catalog=profile_catalog,
            profile_mask=profile_mask,
        )
    except Exception as exc:
        promotion_report = {
            "schema": _STEP_C_GLUE_PROFILE_PROMOTION_REPORT_SCHEMA,
            "schema_version": _STEP_C_GLUE_PROFILE_PROMOTION_REPORT_SCHEMA_VERSION,
            "profile_id": _STEP_C_GLUE_PROFILE_PROMOTION_PROFILE_ID,
            "promotion_pass": False,
            "phase3_pass": bool((phase3_report or {}).get("phase3_pass")) if isinstance(phase3_report, dict) else False,
            "n_glues": int(len(glue_attempts) if isinstance(glue_attempts, list) else 0),
            "n_invocations": int(len(invocations) if isinstance(invocations, list) else 0),
            "profiles": [],
            "failures": [{"code": "PROMOTION_REPORT_BUILD_ERROR", "msg": str(exc)}],
            "catalog": profile_catalog,
        }

    if isinstance(promotion_last, dict) and promotion_last:
        promotion_report_payload = promotion_last
    else:
        promotion_report_payload = None

    # Optionally enrich exported glue_attempts with computed glue_profile_receipts (copy-on-write).
    glue_attempts_payload = glue_attempts
    try:
        if isinstance(glue_attempts, list):
            out_list = []
            for att0 in glue_attempts:
                if isinstance(att0, dict):
                    try:
                        att = _json.loads(_json.dumps(att0))
                    except Exception:
                        att = dict(att0)
                    try:
                        recs = _b1_stepc_compute_glue_profile_receipts_for_attempt(
                            att, profile_catalog=profile_catalog, profile_mask=profile_mask
                        )
                        if recs:
                            att["glue_profile_receipts"] = recs
                    except Exception:
                        pass
                    out_list.append(att)
                else:
                    out_list.append(att0)
            glue_attempts_payload = out_list
    except Exception:
        glue_attempts_payload = glue_attempts

    bdir = _Path(bundle_dir)
    meta_dir = _b1_rel_dirs_for_root(bdir)["meta"]
    stepc_dir = meta_dir / _STEP_C_BUNDLE_ANNEX_DIRNAME
    stepc_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(path: _Path, obj: object):
        try:
            _guarded_atomic_write_json(path, obj)  # type: ignore[name-defined]
        except Exception:
            path.write_text(canonical_json(obj), encoding="utf-8")  # type: ignore[name-defined]

    glue_path = stepc_dir / _STEP_C_BUNDLE_GLUE_ATTEMPTS_FILENAME
    inv_path = stepc_dir / _STEP_C_BUNDLE_STEP_A_INVOCATIONS_FILENAME
    rep_path = stepc_dir / _STEP_C_BUNDLE_PHASE3_REPORT_FILENAME
    tower_path = stepc_dir / _STEP_C_BUNDLE_TOWER_BARCODE_FILENAME

    catalog_path = stepc_dir / _STEP_C_BUNDLE_GLUE_PROFILE_CATALOG_FILENAME
    promo_path = stepc_dir / _STEP_C_BUNDLE_GLUE_PROFILE_PROMOTION_REPORT_FILENAME

    alpha_path = stepc_dir / _STEP_C_BUNDLE_TOWER_HASHES_SCHED_ALPHA_FILENAME
    beta_path = stepc_dir / _STEP_C_BUNDLE_TOWER_HASHES_SCHED_BETA_FILENAME
    div_path = stepc_dir / _STEP_C_BUNDLE_TOWER_HASHES_FIRST_DIVERGENCE_FILENAME

    sem_path = stepc_dir / _STEP_C_BUNDLE_TOWER_BARCODE_SEMANTICS_FILENAME
    idx_path = stepc_dir / _STEP_C_BUNDLE_INDEX_FILENAME

    _write_json(glue_path, glue_attempts_payload)
    _write_json(inv_path, invocations)
    _write_json(rep_path, phase3_report)
    if tower_last is not None:
        _write_json(tower_path, tower_last)
    if phase3_report_payload is not None:
        _write_json(stepc_dir / "phase3_stepc_report__ui_last.json", phase3_report_payload)

    # New: glue profile catalog + promotion report.
    _write_json(catalog_path, profile_catalog)
    _write_json(promo_path, promotion_report)
    if promotion_report_payload is not None:
        _write_json(stepc_dir / "stepc_glue_profile_promotion_report__ui_last.json", promotion_report_payload)

    # New: towers schedule hashes (best effort; only write if present).
    if tower_hash_alpha_last is not None:
        _write_json(alpha_path, tower_hash_alpha_last)
    if tower_hash_beta_last is not None:
        _write_json(beta_path, tower_hash_beta_last)

    # First divergence mapping: use last if present, else compute if both barcodes exist.
    div_obj = tower_hash_div_last
    if div_obj is None and isinstance(tower_hash_alpha_last, dict) and isinstance(tower_hash_beta_last, dict):
        try:
            div_obj = _b1_stepc_first_divergence_mapping(tower_hash_alpha_last, tower_hash_beta_last)
        except Exception:
            div_obj = None
    if div_obj is not None:
        _write_json(div_path, div_obj)

    # New: tower barcode semantics bridge (best effort; only write if present).
    if tower_sem_last is not None:
        _write_json(sem_path, tower_sem_last)

    # Hashes for pointers (best effort).
    try:
        glue_sig8 = _hash_file_sig8(glue_path)  # type: ignore[name-defined]
    except Exception:
        glue_sig8 = ""
    try:
        inv_sig8 = _hash_file_sig8(inv_path)  # type: ignore[name-defined]
    except Exception:
        inv_sig8 = ""
    try:
        rep_sig8 = _hash_file_sig8(rep_path)  # type: ignore[name-defined]
    except Exception:
        rep_sig8 = ""

    tower_sig8 = ""
    if tower_last is not None:
        try:
            tower_sig8 = _hash_file_sig8(tower_path)  # type: ignore[name-defined]
        except Exception:
            tower_sig8 = ""

    cat_sig8 = ""
    try:
        cat_sig8 = _hash_file_sig8(catalog_path)  # type: ignore[name-defined]
    except Exception:
        cat_sig8 = ""
    promo_sig8 = ""
    try:
        promo_sig8 = _hash_file_sig8(promo_path)  # type: ignore[name-defined]
    except Exception:
        promo_sig8 = ""

    alpha_sig8 = ""
    beta_sig8 = ""
    div_sig8 = ""
    if tower_hash_alpha_last is not None:
        try:
            alpha_sig8 = _hash_file_sig8(alpha_path)  # type: ignore[name-defined]
        except Exception:
            alpha_sig8 = ""
    if tower_hash_beta_last is not None:
        try:
            beta_sig8 = _hash_file_sig8(beta_path)  # type: ignore[name-defined]
        except Exception:
            beta_sig8 = ""
    if div_obj is not None:
        try:
            div_sig8 = _hash_file_sig8(div_path)  # type: ignore[name-defined]
        except Exception:
            div_sig8 = ""

    sem_sig8 = ""
    if tower_sem_last is not None:
        try:
            sem_sig8 = _hash_file_sig8(sem_path)  # type: ignore[name-defined]
        except Exception:
            sem_sig8 = ""

    index = {
        "schema": "stepc_bundle_annex",
        "schema_version": "stepc.bundle_annex.v1",
        "profile_id": _STEP_C_PROFILE_ID,
        "snapshot_id": str(snapshot_id),
        "bundle_manifest_sig8": str(manifest_sig8 or ""),
        "written_at_utc": int(_time.time()) if "_time" in globals() else int(time.time()),
        "present": True,
        "phase3_pass": bool((phase3_report or {}).get("phase3_pass")) if isinstance(phase3_report, dict) else False,
        "n_glues": int((phase3_report or {}).get("n_glues")) if isinstance(phase3_report, dict) else 0,
        "n_invocations": int((phase3_report or {}).get("n_invocations")) if isinstance(phase3_report, dict) else 0,
        # payload summary for the additive gate
        "glue_profile_promotion_pass": bool((promotion_report or {}).get("promotion_pass")) if isinstance(promotion_report, dict) else False,
        "artifacts": {
            "glue_attempts": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_GLUE_ATTEMPTS_FILENAME}", "sig8": glue_sig8},
            "stepA_invocations": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_STEP_A_INVOCATIONS_FILENAME}", "sig8": inv_sig8},
            "phase3_report": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_PHASE3_REPORT_FILENAME}", "sig8": rep_sig8},
            "tower_barcode_last": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_TOWER_BARCODE_FILENAME}", "sig8": tower_sig8} if tower_last is not None else None,
            # additive
            "glue_profile_catalog": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_GLUE_PROFILE_CATALOG_FILENAME}", "sig8": cat_sig8},
            "glue_profile_promotion_report": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_GLUE_PROFILE_PROMOTION_REPORT_FILENAME}", "sig8": promo_sig8},
            "tower_hashes_sched_alpha": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_TOWER_HASHES_SCHED_ALPHA_FILENAME}", "sig8": alpha_sig8} if tower_hash_alpha_last is not None else None,
            "tower_hashes_sched_beta": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_TOWER_HASHES_SCHED_BETA_FILENAME}", "sig8": beta_sig8} if tower_hash_beta_last is not None else None,
            "tower_hashes_first_divergence": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_TOWER_HASHES_FIRST_DIVERGENCE_FILENAME}", "sig8": div_sig8} if div_obj is not None else None,
            "tower_barcode_semantics_last": {"relpath": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_TOWER_BARCODE_SEMANTICS_FILENAME}", "sig8": sem_sig8} if tower_sem_last is not None else None,
        },
    }
    _write_json(idx_path, index)
    return index

def _b1_write_bundle_tree_for_snapshot(
    snapshot_id: str | None = None,
    run_ctx: dict | None = None,
    *,
    activate_derived_worlds: bool = False,
) -> tuple[_Path, dict]:
    """High-level B1 entrypoint: collect state, build manifest, write tree.

    Steps:

      1. Collect the B1 bundle state via ``_b1_collect_bundle_state(...)``,
         which validates that the v2 suite, coverage, Time(τ), and D4
         artifacts exist for the SSOT snapshot.
      2. Build the bundle manifest for the same snapshot and attach D4
         certificate metadata (if available).
      3. Compute and stamp ``sig8`` for the manifest.
      4. Resolve/ensure the bundle directory under ``logs/bundle/`` using
         the manifest's ``snapshot_id`` and ``sig8``.
      5. Materialize the canonical B1 filesystem tree under that directory.
      6. Write ``bundle_manifest.json`` into ``meta/`` and, for compatibility
         with earlier helpers, also at the bundle root if not already present.

    Returns ``(bundle_dir, state)`` where ``state`` includes the final
    manifest under ``state["bundle_manifest"]``.
    """
    # 1) Collect B1 state (read-only pass over existing artifacts).
    state = _b1_collect_bundle_state(snapshot_id=snapshot_id, run_ctx=run_ctx, activate_derived_worlds=activate_derived_worlds)
    sid = str(state.get("snapshot_id") or "")


    # Optional activation: derived worlds become part of τ closure only when explicitly activated.
    try:
        tau = state.setdefault("time_tau", {})
        if not isinstance(tau, dict):
            tau = {}
            state["time_tau"] = tau
        c3 = tau.setdefault("c3", {})
        if not isinstance(c3, dict):
            c3 = {}
            tau["c3"] = c3
        c3["derived_worlds_activate"] = bool(activate_derived_worlds)
    except Exception as exc:
        raise RuntimeError(f"B1: failed to set derived_worlds_activate in collected state: {exc}") from exc

    # 2) Build a manifest for this snapshot and attach D4 meta.
    manifest = build_bundle_manifest_for_snapshot(snapshot_id=sid, run_ctx=run_ctx)
    manifest = _d4_attach_certificate_meta_to_manifest(
        manifest,
        snapshot_id=sid,
        run_ctx=run_ctx,
    )

    # 2.a) S3: attach parity locators + bind Track III policy profile id into manifest.meta.
    try:
        meta = dict(manifest.get("meta") or {})
        pstate = state.get("parity") if isinstance(state.get("parity"), dict) else {}
        inst_sig8 = str((pstate or {}).get("instance_sig8") or "").strip()
        cert_sig8 = str((pstate or {}).get("certificate_sig8") or "").strip()
        if inst_sig8 and cert_sig8:
            meta["parity_instance"] = {
                "path": f"certs/parity/parity_instance__{sid}__{inst_sig8}.json",
                "sig8": inst_sig8,
            }
            meta["parity_certificate"] = {
                "path": f"certs/parity/parity_certificate__{sid}__{cert_sig8}.json",
                "sig8": cert_sig8,
            }
            meta["t3_policy_profile_id"] = T3_POLICY_PROFILE_ID_POINTER_GATED_DW_T0_MANDATORY_PARITY_V0_6
        else:
            # Anti-drift default for non-parity bundles.
            meta["t3_policy_profile_id"] = T3_POLICY_PROFILE_ID_POINTER_GATED_DW_V0_2
        manifest["meta"] = meta
    except Exception as exc:
        raise RuntimeError(f"B1: failed to attach parity/policy meta: {exc}") from exc

    # 2.b) B4: attach κ registry (context registry) under meta.
    # κ is injected post-D4 and MUST NOT affect manifest sig8.
    try:
        meta = dict(manifest.get("meta") or {})
        meta["kappa_registry"] = _b4_build_kappa_registry_from_state(state, run_ctx=run_ctx)
        manifest["meta"] = meta
    except Exception:
        # Keep any placeholder provided by build_bundle_manifest_for_snapshot.
        meta = dict(manifest.get("meta") or {})
        meta.setdefault(
            "kappa_registry",
            {"schema_version": B4_KAPPA_REGISTRY_SCHEMA_VERSION, "contexts": []},
        )
        manifest["meta"] = meta

    # 3) Compute sig8 for the manifest.
    manifest = stamp_bundle_manifest_sig8(manifest)
    state["bundle_manifest"] = manifest

    # 4) Resolve the bundle directory for this manifest.
    bdir = _ensure_bundle_dir_for_manifest(manifest)

    # 5) Materialize the canonical B1 filesystem tree.
    _b1_materialize_bundle_tree(bdir, state)

    # 6) Write bundle_manifest.json under meta/ and optionally at root.
    dirs = _b1_rel_dirs_for_root(bdir)
    meta_manifest_path = dirs["meta"] / "bundle_manifest.json"
    meta_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_txt = canonical_json(manifest)
    meta_manifest_path.write_text(manifest_txt, encoding="utf-8")

    legacy_manifest_path = bdir / "bundle_manifest.json"
    if not legacy_manifest_path.exists():
        legacy_manifest_path.write_text(manifest_txt, encoding="utf-8")

    # 7) B4: admissibility verification / pruning pass (current stage: verify-only).
    b4_report = _b4_verify_and_prune_bundle_tree(bdir, state)
    _b4_write_admissibility_report(bdir, b4_report)
    if isinstance(b4_report, dict) and b4_report.get("status") == "FAIL":
        raise RuntimeError(f"B4: bundle admissibility failed: {b4_report.get('violations')}")

    # 8) B5: compute + write the semantic object-set index (meta/b5_index.v1.json).
    # This is the policy-agnostic identity anchor ("two bundles, one id").
    try:
        v2_manifest_path = (state.get("manifests") or {}).get("v2_suite_path")
        if not v2_manifest_path:
            raise RuntimeError("B5: missing v2_suite_path in collected state")
        b5_index = _b5_compute_index_from_manifest(v2_manifest_path, snapshot_id=sid, strict=True)
        _b5_write_index(bdir, b5_index)
    except Exception as exc:
        raise RuntimeError(f"B5: failed to build b5_index: {exc}") from exc

    
    # 8.5) Phase 3 / Step‑C annex: persist Step‑C glue/tower artifacts into the bundle (optional).
    # Packaging-only: pointers live under manifest.meta (excluded from manifest sig8).
    try:
        stepc_index = _b1_stepc_write_annex_from_session(bdir, snapshot_id=sid, manifest_sig8=manifest.get("sig8"))
        if isinstance(stepc_index, dict) and stepc_index.get("present"):
            # Attach locators under manifest.meta for discoverability (does not affect sig8).
            try:
                meta = dict(manifest.get("meta") or {})
                meta["phase3_stepc"] = {
                    "dir": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}",
                    "index_path": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_INDEX_FILENAME}",
                    "phase3_report_path": f"meta/{_STEP_C_BUNDLE_ANNEX_DIRNAME}/{_STEP_C_BUNDLE_PHASE3_REPORT_FILENAME}",
                }
                meta["phase3_stepc"]["summary"] = {
                    "phase3_pass": stepc_index.get("phase3_pass"),
                    "n_glues": stepc_index.get("n_glues"),
                    "n_invocations": stepc_index.get("n_invocations"),
                }
                manifest["meta"] = meta
                state["bundle_manifest"] = manifest

                # Rewrite manifest files so the meta pointers are present inside the sealed tree.
                manifest_txt = canonical_json(manifest)
                meta_manifest_path.write_text(manifest_txt, encoding="utf-8")
                legacy_manifest_path.write_text(manifest_txt, encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass

# 9) B6: compute + write the global bundle seal (Profile.v1) over the materialized tree.
    b6_payload = b6_compute_seal_profile_v1(bdir)
    b6_write_seal(bdir, b6_payload)

    return bdir, state



# --- D4 certificate schema & dirs ---

D4_CERTIFICATE_SCHEMA = "d4_certificate.v2"
D4_CERTIFICATE_SCHEMA_VERSION = "1.1.0"


def _d4_cert_root_dir() -> _Path:
    """Return the root directory for D4 certificates (logs/certs/d4)."""
    root = _CERTS_DIR / "d4"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _d4_cert_filename(snapshot_id: str, sig8: str) -> str:
    """Canonical filename for a D4 certificate JSON file."""
    if sig8:
        return f"d4_certificate__{snapshot_id}__{sig8}.json"
    return f"d4_certificate__{snapshot_id}.json"


def _d4_cert_path(snapshot_id: str, sig8: str = "") -> _Path:
    """Canonical path for a D4 certificate for the given snapshot_id/sig8."""
    return _d4_cert_root_dir() / _d4_cert_filename(snapshot_id, sig8)


def _hash_file_sig8(path: _Path) -> str:
    """Compute a stable sig8 for a file's contents (sha256, first 8 hex chars)."""
    h = _hashlib.sha256()
    p = _Path(path)
    try:
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        # On any error we fall back to hashing the string path itself; this keeps
        # the helper total and avoids raising in analyzer/export code paths.
        h.update(str(p).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:8]


# --- Pointer-gated artifact helpers (Phase 7 contract: no scans / fail-closed) ---

_SIG8_HEX_RE = re.compile(r"^[0-9a-f]{8}$")


def _is_sig8_hex(s: str | None) -> bool:
    """Return True iff s is exactly 8 lowercase hex chars."""
    if not isinstance(s, str):
        return False
    return bool(_SIG8_HEX_RE.fullmatch(s))


def _strict_hash_file_sig8(path: _Path | str) -> str:
    """Fail-closed file sig8: sha256(file_bytes)[:8].

    Unlike _hash_file_sig8, this helper NEVER hashes the path string as a
    fallback. Any IO error is surfaced to the caller.
    """
    p = _Path(path)
    h = _hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:8]


def _read_jsonl_strict(path: _Path | str) -> list[dict]:
    """Parse a JSONL file strictly: any malformed line is an error."""
    p = _Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = _json.loads(s)
            except Exception as exc:
                raise ValueError(f"invalid JSONL at {p} line {i}") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"invalid JSONL record at {p} line {i}: not an object")
            out.append(rec)
    return out


def _v2_fixture_triplets_from_manifest_rows(rows: list[dict], *, snapshot_id: str) -> list[tuple[str, str, str]]:
    """Return [(district_id, fixture_label, strict_sig8), ...] for a snapshot.

    Fail-closed: missing/invalid core keys or snapshot_id mismatches raise.
    """
    sid = str(snapshot_id or "").strip()
    if not sid:
        raise ValueError("v2 fixture inventory: empty snapshot_id")

    triplets: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for idx, row in enumerate(rows):
        # Schema guard (prevents silent acceptance of incompatible JSONL shapes).
        schema = str(row.get("schema") or "")
        if schema and schema != "suite_row.v2":
            raise ValueError(f"v2 fixture inventory: unexpected row schema at index {idx}: {schema}")

        row_sid = str(row.get("snapshot_id") or "").strip()
        if row_sid != sid:
            raise ValueError(
                "v2 fixture inventory: manifest row snapshot_id mismatch "
                f"(expected {sid!r} got {row_sid!r})"
            )

        district_id = str(row.get("district_id") or "").strip()
        fixture_label = str(row.get("fixture_label") or "").strip()
        strict_sig8 = str(row.get("strict_sig8") or "").strip().lower()

        if not district_id or not fixture_label or not strict_sig8:
            raise ValueError(
                "v2 fixture inventory: missing district_id/fixture_label/strict_sig8 "
                f"at row index {idx}"
            )
        if not _is_sig8_hex(strict_sig8):
            raise ValueError(
                "v2 fixture inventory: invalid strict_sig8 (expected 8 lowercase hex) "
                f"at row index {idx}: {strict_sig8!r}"
            )

        key = (district_id, fixture_label)
        if key in seen:
            raise ValueError(
                "v2 fixture inventory: duplicate (district_id, fixture_label) in manifest "
                f"at {key!r}"
            )
        seen.add(key)
        triplets.append((district_id, fixture_label, strict_sig8))

    return triplets


# --- Bundle manifest schema (Phase 2 — shape only, no I/O yet)

# Root directory for exportable v2 bundles for a given 64× run.
# All bundle manifests and zip exports will live under this tree.
_BUNDLE_ROOT = _REPO_ROOT / "logs" / "bundle"

# Schema/version tag for bundle_manifest.json. Bump this only when the
# top-level shape or key layout changes in a way that is not backward compatible.
BUNDLE_MANIFEST_SCHEMA_VERSION = "v2.core.2"

# Top-level keys expected in bundle_manifest.json. This is intentionally
# small and stable; detailed layout lives in the nested *_SECTION_KEYS maps.
BUNDLE_MANIFEST_TOP_LEVEL_KEYS = (
    "snapshot_id",
    "created_at_utc",
    "engine_rev",
    "schema_version",
    "sig8",
    "pointers",
    "manifests",
    "certs",
    "time_tau",
    "coverage",
    "logs",
    "b4",
    "b5",
    "b6",
    "meta",
)

# Expected keys inside bundle_manifest["manifests"].
BUNDLE_MANIFEST_MANIFEST_SECTION_KEYS = (
    "v2_full_scope",   # logs/manifests/manifest_full_scope.v2.jsonl
    "time_tau_c3",     # logs/manifests/time_tau_c3_manifest_full_scope.jsonl (if present)
    "time_tau_c3_receipts_manifest",  # logs/manifests/time_tau_c3_receipts_manifest__{snapshot_id}.json
)

# Expected keys inside bundle_manifest["certs"].
BUNDLE_MANIFEST_CERTS_SECTION_KEYS = (
    "strict",          # strict-core / v2 cert bundle root
    "projected",       # optional projected/auto cert bundle root
)

# Expected keys inside bundle_manifest["time_tau"].
BUNDLE_MANIFEST_TIME_TAU_SECTION_KEYS = (
    "c2_toy_dir",      # directory for C2 τ-toy sweep artifacts
    "c3_receipts_dir", # directory for C3 recompute receipts
    "c3_receipts_manifest_path",  # canonical receipts inventory (may be empty)
    "c4_rollup_path",  # path to C4 τ rollup JSON
)

# Expected keys inside bundle_manifest["coverage"].
BUNDLE_MANIFEST_COVERAGE_SECTION_KEYS = (
    "coverage_jsonl_path",      # logs/reports/coverage.jsonl
    "coverage_rollup_csv_path", # logs/reports/coverage_rollup.csv
)

# Expected keys inside bundle_manifest["logs"].
BUNDLE_MANIFEST_LOGS_SECTION_KEYS = (
    "loop_receipts_dir",   # logs/experiments/ (C2/C3/C4 receipts, etc.)
    "world_snapshots_dir", # logs/snapshots/ (v2 world_snapshot.v2.json, etc.)
)







# --- B4/B5 blueprint stubs (wiring step 2: shape only; no call sites) ---
# NOTE: These are intentionally not wired yet. They exist so subsequent wiring
# steps can be mechanical and grep-friendly.
B4_KAPPA_REGISTRY_SCHEMA_VERSION = "kappa_registry.v0.1"

# Time(τ) canonical inventory manifests (Phase‑7 discipline: canon JSON + sig8‑stamp where available).
#
# v0.1 = legacy (no embedded sig8; still accepted for read-compat).
# v0.2 = self-sig8 stamped (embedded `sig8` binds the load-bearing projection).
TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION_V0_1 = "time_tau_c3_receipts_manifest.v0.1"
TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION_V0_2 = "time_tau_c3_receipts_manifest.v0.2"
TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION = TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION_V0_2
TIME_TAU_C3_RECEIPTS_MANIFEST_ALLOWED_VERSIONS = {
    TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION_V0_1,
    TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION_V0_2,
}

TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA = "time_tau_c3_derived_worlds_manifest"
TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA_VERSION_V0_1 = "time_tau_c3_derived_worlds_manifest.v0.1"
TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA_VERSION_V0_2 = "time_tau_c3_derived_worlds_manifest.v0.2"
TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA_VERSION = TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA_VERSION_V0_2
TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_ALLOWED_VERSIONS = {
    TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA_VERSION_V0_1,
    TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA_VERSION_V0_2,
}

# Canonical derived-worlds locations / pointer role
TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_REL = "manifests/time_tau_c3_derived_worlds_manifest.json"
TIME_TAU_C3_DERIVED_WORLDS_BASE_DIR_REL = "tau/c3/derived_worlds"
TIME_TAU_PTR_ROLE_C3_DERIVED_WORLDS_MANIFEST = "c3_derived_worlds_manifest"


# --- Track II (Time(τ)) closure surfaces (v0.1) ---
TIME_TAU_POINTER_SET_SCHEMA = "time_tau_pointer_set"
TIME_TAU_POINTER_SET_SCHEMA_VERSION_V0_1 = "time_tau_pointer_set.v0.1"
TIME_TAU_POINTER_SET_SCHEMA_VERSION_V0_2 = "time_tau_pointer_set.v0.2"
TIME_TAU_POINTER_SET_SCHEMA_VERSION = TIME_TAU_POINTER_SET_SCHEMA_VERSION_V0_2
TIME_TAU_POINTER_SET_ALLOWED_VERSIONS = {
    TIME_TAU_POINTER_SET_SCHEMA_VERSION_V0_1,
    TIME_TAU_POINTER_SET_SCHEMA_VERSION_V0_2,
}

def time_tau_pointer_set_hash_body_v0_2(ptr_obj: dict) -> dict:
    """Return the hashed-body selection for time_tau_pointer_set sig8 stamping.

    H_ptr := {schema, schema_version, snapshot_id, surface_pointers, sig8:""}

    - annex_pointers and unknown top-level keys are quarantined (excluded).
    - surface_pointers is canon-sorted by (role, relpath).
    - each surface pointer row is reduced to {role, relpath}.
    """
    if not isinstance(ptr_obj, dict):
        raise TypeError("time_tau_pointer_set_hash_body_v0_2: ptr_obj must be a dict")

    schema = ptr_obj.get("schema") or TIME_TAU_POINTER_SET_SCHEMA
    schema_version = ptr_obj.get("schema_version") or TIME_TAU_POINTER_SET_SCHEMA_VERSION_V0_2
    snapshot_id = str(ptr_obj.get("snapshot_id") or "").strip()

    sps = ptr_obj.get("surface_pointers")
    if sps is None:
        sps = []
    if not isinstance(sps, list):
        raise TypeError("time_tau_pointer_set_hash_body_v0_2: surface_pointers must be a list")

    canon_sps: list[dict] = []
    for r in sps:
        if not isinstance(r, dict):
            continue
        role = str(r.get("role") or "")
        rel = str(r.get("relpath") or "")
        canon_sps.append({"role": role, "relpath": rel})

    canon_sps.sort(key=lambda x: (str((x or {}).get("role") or ""), str((x or {}).get("relpath") or "")))

    return {
        "schema": schema,
        "schema_version": schema_version,
        "snapshot_id": snapshot_id,
        "surface_pointers": canon_sps,
        "sig8": "",
    }


def stamp_time_tau_pointer_set_sig8_v0_2(ptr_obj: dict) -> dict:
    """Return a copy of `ptr_obj` with sig8 stamped per time_tau_pointer_set.v0.2."""
    base = dict(ptr_obj or {})
    base["schema"] = TIME_TAU_POINTER_SET_SCHEMA
    base["schema_version"] = TIME_TAU_POINTER_SET_SCHEMA_VERSION_V0_2
    hb = time_tau_pointer_set_hash_body_v0_2(base)
    base["sig8"] = hash_json_sig8(hb)
    return base


TIME_TAU_CLOSURE_SET_SCHEMA = "time_tau_closure_set"

TIME_TAU_CLOSURE_SET_SCHEMA_VERSION = "time_tau_closure_set.v0.1"

TIME_TAU_SURFACE_SCHEMA = "time_tau_surface"
TIME_TAU_SURFACE_SCHEMA_VERSION = "time_tau_surface.v0.1"

TIME_TAU_VERIFY_RECEIPT_SCHEMA = "time_tau_verify_receipt"
TIME_TAU_VERIFY_RECEIPT_SCHEMA_VERSION = "time_tau_verify_receipt.v0.2"

B4_ADMISSIBILITY_REPORT_SCHEMA_VERSION = "b4_admissibility_report.v0.1"

# --- B6 packaging seal (Profile.v1) ---
B6_SEAL_SCHEMA = "b6_seal"
B6_SEAL_SCHEMA_VERSION = "b6_seal.v0.1"

B6_VERIFY_RECEIPT_SCHEMA = "b6_verify_receipt"
B6_VERIFY_RECEIPT_SCHEMA_VERSION = "b6_verify_receipt.v0.2"

# Meaning-level profile id (seal semantics)
B6_SEAL_PROFILE_ID = "B6.CoreTree.Profile.v1"
B6_INVENTORY_ENCODING_ID = "B6ENC1"
B6_PORTABILITY_KEY_ID = "PORTKEY1"
B6_EXCLUDE_POLICY_ID = "B6_EXCLUDE_V1"

B6_SEAL_HASH_ALGO = "sha256"
B6_FILE_HASH_ALGO = "sha256"

# Canonical in-bundle locations (bundle-relative paths)
B6_SEAL_REL_PATH = "meta/bundle_hash.json"
B6_VERIFY_RECEIPT_REL_PATH = "meta/b6_verify_receipt.json"

# Profile.v1 mandatory excludes for the seal domain (no ad-hoc excludes).
B6_EXCLUDE_REL_PATHS_V1 = {
    B6_SEAL_REL_PATH,          # self (non-circular)
    "bundle_manifest.json",     # legacy root mirror (non-core)
    B6_VERIFY_RECEIPT_REL_PATH, # verifier output (non-core)
}

# B6 fail codes (frozen for b6_verify_receipt.v0.1)
B6_FAIL_CODES_V0_1 = (
    "SEAL_DIGEST_MISMATCH",
    "SEAL_PROFILE_HEADER_MISMATCH",
    "TREE_HAS_SPECIAL_NODE",
    "TREE_HAS_SYMLINK",
    "TREE_PATH_INVALID",
    "TREE_PORTKEY_COLLISION",
    "TREE_REL_UTF8_INVALID",
    "WITNESS_ENTRIES_MISMATCH",
    "WITNESS_EXCLUDELIST_MISMATCH",
    "WITNESS_NFILES_MISMATCH",
)
B6_FAIL_CODES_V0_1_SET = set(B6_FAIL_CODES_V0_1)
# --- B5 semantic identity artifacts (per-fixture + set index) ---
# B5 identity is *semantic* and must be independent of local bundle layout,
# runtime noise, or policy-specific annex.
B5_IDENTITY_SCHEMA_VERSION = "b5_identity.v0.1"
B5_FP_CORE_SCHEMA_VERSION = "b5_fp_core.strict_core.f2.v0.1"
B5_INDEX_SCHEMA_VERSION = "b5_index.v0.1"


def _b4_build_kappa_registry_from_state(state: dict, *, run_ctx: dict | None = None) -> dict:
    """B4: Build a minimal κ registry (context registry) from collected B1 state.

    First clean pass (current stage):
      - records only stable fixture identity triples (district_id, fixture_label, strict_sig8),
      - avoids absolute paths and timestamps,
      - may be expanded later to include explicit projection contexts and τ regime.
    """
    st = state or {}
    sid = str(st.get("snapshot_id") or "")

    fixtures = st.get("fixtures") or {}
    contexts: list[dict] = []
    try:
        if isinstance(fixtures, dict):
            for key, f in sorted(fixtures.items(), key=lambda kv: str(kv[0])):
                district_id = ""
                fixture_label = ""
                strict_sig8 = ""
                if isinstance(f, dict):
                    district_id = str(f.get("district_id") or "")
                    fixture_label = str(f.get("fixture_label") or "")
                    strict_sig8 = str(f.get("strict_sig8") or "")
                if not district_id and isinstance(key, tuple) and len(key) > 0:
                    district_id = str(key[0])
                if not fixture_label and isinstance(key, tuple) and len(key) > 1:
                    fixture_label = str(key[1])
                if not strict_sig8 and isinstance(key, tuple) and len(key) > 2:
                    strict_sig8 = str(key[2])

                contexts.append(
                    {
                        "district_id": district_id,
                        "fixture_label": fixture_label,
                        "strict_sig8": strict_sig8,
                    }
                )
    except Exception:
        contexts = []

    kappa: dict = {
        "schema_version": B4_KAPPA_REGISTRY_SCHEMA_VERSION,
        "snapshot_id": sid,
        "contexts": contexts,
    }

    # Optional: record τ scope markers (repo-relative when possible), without adding nondeterminism.
    try:
        rc = dict(run_ctx or {})
        tt = rc.get("time_tau") if isinstance(rc.get("time_tau"), dict) else None
        if isinstance(tt, dict):
            rm = tt.get("c3_receipts_manifest_path") or (tt.get("c3_receipts_manifest") or {}).get("receipts_manifest_path")
            if rm:
                try:
                    rm = _bundle_repo_relative_path(rm)  # type: ignore[name-defined]
                except Exception:
                    rm = str(rm)
            kappa["time_tau"] = {
                "in_scope": True,
                "c3_receipts_manifest_path": rm,
            }
    except Exception:
        pass

    return kappa


def _b4_expected_bundle_inventory(state: dict) -> set[str]:
    """B4: Return the required authoritative relpath set for the bundle.

    v0.1 invariant (frozen this session):
      - B4 MUST enumerate every per-fixture cert neighborhood (structural presence),
        derived from the collected B1 state (NOT from directory listings).

    This inventory is used by the B4 verifier as the required set of files
    that must exist after B1 materialization and before B5/B6 closure.
    """
    st = state or {}
    sid = str(st.get("snapshot_id") or "").strip()

    exp: set[str] = set()

    # Always-present core artifacts written by B1.
    exp.add("meta/bundle_manifest.json")
    exp.add("world/world_snapshot.v2.json")

    # Manifests
    exp.add("manifests/v2_suite_full_scope.jsonl")
    exp.add("manifests/time_tau_c3_manifest_full_scope.jsonl")
    exp.add("manifests/time_tau_c3_receipts_manifest.json")
    exp.add("manifests/time_tau_pointer_set.json")

    # Derived worlds manifest is required only when explicitly activated.
    try:
        activate_dw = bool((((st.get("time_tau") or {}).get("c3") or {}).get("derived_worlds_activate")))
    except Exception:
        activate_dw = False
    if activate_dw:
        exp.add(TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_REL)


    # Coverage
    exp.add("coverage/coverage.jsonl")
    exp.add("coverage/coverage_rollup.csv")
    # NOTE: coverage/histograms_v2.json is optional (copied missing_ok=True).

    # Time(τ) artifacts (C2/C4 are required in this app version).
    if sid:
        exp.add(f"tau/c2/time_tau_local_flip_sweep__{sid}.jsonl")
        exp.add(f"tau/c2/time_tau_local_flip_sweep__{sid}.csv")
    exp.add("tau/c4/time_tau_c3_rollup.jsonl")
    exp.add("tau/c4/time_tau_c3_rollup.csv")
    exp.add("tau/c4/time_tau_c3_tau_mismatches.jsonl")
    exp.add("tau/c4/time_tau_c3_tau_mismatches.csv")

    # D4 certificate (required)
    try:
        d4_state = (st.get("d4") or {}) if isinstance(st.get("d4"), dict) else {}
        d4_sig8 = str(d4_state.get("sig8") or "").strip()
        if not d4_sig8:
            d4_src = d4_state.get("cert_path")
            d4_name = _Path(d4_src).name if d4_src else ""
            d4_sig8 = str(_b1_guess_sig8_from_d4_name(d4_name) or "").strip() if d4_name else ""
        if sid:
            exp.add(_b1_d4_cert_path(_Path("."), sid, d4_sig8 or None).as_posix())
    except Exception:
        # Conservative fallback: require at least the non-sig8 form.
        if sid:
            exp.add(f"certs/d4/d4_certificate__{sid}.json")

    # Parity artifacts (S3) — required when present in collected state.
    try:
        pstate = (st.get("parity") or {}) if isinstance(st.get("parity"), dict) else {}
        inst_sig8 = str(pstate.get("instance_sig8") or "").strip()
        cert_sig8 = str(pstate.get("certificate_sig8") or "").strip()
        if sid and inst_sig8:
            exp.add(_b1_parity_instance_path(_Path("."), sid, inst_sig8).as_posix())
        if sid and cert_sig8:
            exp.add(_b1_parity_certificate_path(_Path("."), sid, cert_sig8).as_posix())
    except Exception:
        pass

    # Per-fixture cert neighborhoods (required structural presence).
    fixtures = st.get("fixtures") or {}
    if not isinstance(fixtures, dict) or not fixtures:
        # Caller (B4 verifier) will surface this as a violation.
        return exp

    for key in sorted(fixtures.keys(), key=lambda x: str(x)):
        f = fixtures.get(key) or {}
        # Prefer explicit fields; fall back to tuple key ordering.
        if isinstance(key, tuple) and len(key) >= 3:
            dk, fk, sk = key[0], key[1], key[2]
        else:
            dk, fk, sk = None, None, None

        district_id = str(f.get("district_id") or dk or "")
        fixture_label = str(f.get("fixture_label") or fk or "")
        strict_sig8 = str(f.get("strict_sig8") or sk or "")
        if not (district_id and fixture_label and strict_sig8):
            # Leave it to the verifier to complain; don't guess.
            continue

        base = f"certs/fixtures/{district_id}/{fixture_label}/{strict_sig8}"
        exp.add(f"{base}/loop_receipt__{fixture_label}.json")
        exp.add(f"{base}/overlap__{district_id}__strict__{strict_sig8}.json")
        exp.add(f"{base}/overlap__{district_id}__projected_columns_k_3_auto__{strict_sig8}.json")
        exp.add(f"{base}/overlap__{district_id}__projected_columns_k_3_file__{strict_sig8}.json")
        exp.add(f"{base}/ab_compare__strict_vs_projected_auto__{strict_sig8}.json")
        exp.add(f"{base}/ab_compare__strict_vs_projected_file__{strict_sig8}.json")
        exp.add(f"{base}/projector_freezer__{district_id}__{strict_sig8}.json")
        exp.add(f"{base}/bundle_index.v2.json")
        exp.add(f"{base}/b5_identity__{district_id}__{strict_sig8}.json")

    return exp


# --- Track II: Time(τ) closure verifier (scan-free) ---

def _time_tau__is_bundle_relpath(rel: str) -> bool:
    """Return True iff rel looks like a valid bundle-relative POSIX path.

    Minimal BundleRelPath discipline (v0.1):
      - non-empty string
      - not absolute
      - no backslashes
      - no '..' segments
      - no leading './'
    """
    if not isinstance(rel, str):
        return False
    s = rel.strip()
    if not s:
        return False
    if s.startswith("/") or s.startswith("\\"):
        return False
    if s.startswith("./"):
        return False
    if "\\" in s:
        return False
    try:
        parts = _Path(s).parts
    except Exception:
        parts = tuple()
    if ".." in parts:
        return False
    return True


def _time_tau__lb_projection_receipts_manifest(rm_obj: dict) -> dict:
    """Load-bearing projection for time_tau_c3_receipts_manifest (v0.1)."""
    out: dict = {
        "schema": str(rm_obj.get("schema") or ""),
        "schema_version": str(rm_obj.get("schema_version") or ""),
        "snapshot_id": str(rm_obj.get("snapshot_id") or ""),
        "receipts_dir_relpath": str(rm_obj.get("receipts_dir_relpath") or ""),
        "mode": str(rm_obj.get("mode") or ""),
        "receipts": [],
    }
    receipts = rm_obj.get("receipts") or []
    out_list: list[dict] = []
    if isinstance(receipts, list):
        for r in receipts:
            if not isinstance(r, dict):
                continue
            rp = r.get("receipt_relpath")
            sig = r.get("receipt_sig8")
            if not isinstance(rp, str) or not isinstance(sig, str):
                continue
            out_list.append({"receipt_relpath": rp, "receipt_sig8": sig})
    out["receipts"] = sorted(out_list, key=lambda x: str((x or {}).get("receipt_relpath") or ""))
    return out




def _time_tau__lb_projection_derived_worlds_manifest(dw_obj: dict) -> dict:
    """Load-bearing projection for time_tau_c3_derived_worlds_manifest (v0.1)."""
    out: dict = {
        "schema": str(dw_obj.get("schema") or ""),
        "schema_version": str(dw_obj.get("schema_version") or ""),
        "snapshot_id": str(dw_obj.get("snapshot_id") or ""),
        "base_dir_relpath": str(dw_obj.get("base_dir_relpath") or ""),
        "mode": str(dw_obj.get("mode") or ""),
        "worlds": [],
    }
    worlds = dw_obj.get("worlds") or []
    out_list: list[dict] = []
    if isinstance(worlds, list):
        for w in worlds:
            if not isinstance(w, dict):
                continue
            rp = w.get("relpath")
            sig = w.get("sig8")
            if not isinstance(rp, str) or not isinstance(sig, str):
                continue
            out_list.append({"relpath": rp, "sig8": sig})
    out["worlds"] = sorted(out_list, key=lambda x: str((x or {}).get("relpath") or ""))
    return out

def _time_tau__canon_closure_entries(entries: list[dict]) -> list[dict]:
    """Canonicalize closure entries (dedupe by relpath, sort)."""
    m: dict[str, dict] = {}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        rel = str(e.get("relpath") or "").strip()
        if not rel:
            continue
        exp = str(e.get("expectation") or "").strip()
        if exp not in ("exists", "sig8_equals"):
            exp = "exists"
        esig = str(e.get("expected_sig8") or "").strip()
        prev = m.get(rel)
        if prev is None:
            m[rel] = {"relpath": rel, "expectation": exp, "expected_sig8": esig}
            continue
        # Merge rules: sig8_equals dominates exists; conflicting sig8 is illegal.
        if prev.get("expectation") == "sig8_equals" or exp == "sig8_equals":
            # If both are sig8_equals and disagree, mark conflict.
            if prev.get("expectation") == "sig8_equals" and exp == "sig8_equals":
                if str(prev.get("expected_sig8") or "") != esig:
                    # Store a conflict marker; verifier will fail closure computable.
                    m[rel] = {"relpath": rel, "expectation": "CONFLICT", "expected_sig8": "CONFLICT"}
                # else identical: keep
            else:
                # One is strict: keep strict expected_sig8 if present.
                keep_sig = esig if exp == "sig8_equals" else str(prev.get("expected_sig8") or "")
                m[rel] = {"relpath": rel, "expectation": "sig8_equals", "expected_sig8": keep_sig}
        # else both exists: keep
    out = list(m.values())
    out.sort(key=lambda x: str((x or {}).get("relpath") or ""))
    return out


def time_tau_verify_bundle_dir(
    bundle_dir: _Path,
    *,
    derived_worlds_regime: str = "POINTER_GATED",
    t0_regime: str | None = None,
) -> dict:
    """Verify Time(τ) closure + surface inside a bundle directory (v0.2).

    Deterministic and scan-free:
      - consumes manifests/time_tau_pointer_set.json (surface_pointers only)
      - consumes manifests/time_tau_c3_receipts_manifest.json (receipts inventory)
      - computes Cl(τ) without any directory traversal or globbing
      - realizes closure (existence + sig8 for receipts)
      - emits tau_surface_sig8 if and only if closure is realized
    """
    bdir = _Path(bundle_dir).resolve()

    # Fixed in-bundle manifest locations
    ptr_path = _b1_manifest_time_tau_pointer_set_path(bdir)
    rm_path = _b1_manifest_tau_c3_receipts_manifest_path(bdir)
    dw_path = _b1_manifest_tau_c3_derived_worlds_manifest_path(bdir)

    checks: list[dict] = []
    diagnostics: dict = {}

    def _add(check_id: str, ok, *, detail: str | None = None, evidence: dict | None = None) -> None:
        row = {"check_id": str(check_id), "ok": ok}
        if detail is not None:
            row["detail"] = str(detail)
        if evidence is not None:
            row["evidence"] = evidence
        checks.append(row)

    snapshot_id: str = ""
    closure_set_sig8: str | None = None
    tau_surface_sig8: str | None = None

    # --- Check 1: pointer set available (exists + schema + role checks) ---
    ptr_obj: dict | None = None
    ptr_ok = False
    ptr_detail = None
    try:
        if not ptr_path.exists():
            ptr_detail = f"missing: {ptr_path.relative_to(bdir).as_posix()}"
        else:
            ptr_raw = ptr_path.read_text(encoding="utf-8")
            ptr_obj = _json.loads(ptr_raw)
            if not isinstance(ptr_obj, dict):
                ptr_detail = "pointer set is not an object"
            else:
                if str(ptr_obj.get("schema") or "") != TIME_TAU_POINTER_SET_SCHEMA:
                    ptr_detail = "schema mismatch"
                elif str(ptr_obj.get("schema_version") or "") not in TIME_TAU_POINTER_SET_ALLOWED_VERSIONS:
                    ptr_detail = f"schema_version not allowed: {str(ptr_obj.get('schema_version') or '')!r} not in {sorted(TIME_TAU_POINTER_SET_ALLOWED_VERSIONS)!r}"
                else:
                    snapshot_id = str(ptr_obj.get("snapshot_id") or "").strip()
                    if not snapshot_id:
                        ptr_detail = "empty snapshot_id"
                    else:
                        sps = ptr_obj.get("surface_pointers") or []
                        if not isinstance(sps, list):
                            ptr_detail = "surface_pointers not a list"
                        else:
                            roles = []
                            rels = []
                            bad = []
                            for r in sps:
                                if not isinstance(r, dict):
                                    continue
                                role = str(r.get("role") or "").strip()
                                rel = str(r.get("relpath") or "").strip()
                                if not role or not rel:
                                    bad.append("empty role/relpath")
                                    continue
                                if not _time_tau__is_bundle_relpath(rel):
                                    bad.append(f"bad relpath: {rel}")
                                    continue
                                if not (rel.startswith("tau/") or rel.startswith("manifests/")):
                                    bad.append(f"out of scope relpath: {rel}")
                                    continue
                                if rel.startswith("tau/c3/receipts/"):
                                    bad.append(f"illegal receipts relpath in surface_pointers: {rel}")
                                    continue
                                roles.append(role)
                                rels.append(rel)
                            # Required roles (v0.1)
                            req = {
                                "c2_sweep_jsonl",
                                "c2_sweep_csv",
                                "c3_manifest_full_scope_jsonl",
                                "c4_rollup_jsonl",
                                "c4_mismatches_jsonl",
                            }
                            if set(roles) != set(roles):  # no-op; kept for readability
                                pass
                            # Role uniqueness
                            if len(set(roles)) != len(roles):
                                bad.append("duplicate roles")
                            if len(set(rels)) != len(rels):
                                bad.append("duplicate relpaths")
                            missing_roles = sorted(req - set(roles))
                            extra_missing = missing_roles
                            if extra_missing:
                                bad.append(f"missing required roles: {','.join(extra_missing)}")
                            if bad:
                                ptr_detail = "; ".join(bad[:5])
                            else:
                                ptr_ok = True
    except Exception as exc:
        ptr_ok = False
        ptr_detail = f"error: {exc}"

    _add("TAU_POINTER_SET_AVAILABLE", ptr_ok, detail=ptr_detail, evidence={"path": ptr_path.relative_to(bdir).as_posix()})
    # Candidate B (diagnostic-only): snapshot_id non-empty as an explicit check row.
    _add("TAU_SNAPSHOT_ID_NONEMPTY", bool(snapshot_id), detail=(None if snapshot_id else "empty"))
    if not ptr_ok:
        ptr_obj = None


    # Optional derived-worlds activation pointer (surface-gated).
    dw_ptr_rel: str | None = None
    dw_activated = None
    dw_ptr_canonical_ok = None
    if ptr_ok and ptr_obj is not None:
        try:
            sps = ptr_obj.get("surface_pointers") or []
            if isinstance(sps, list):
                for r in sps:
                    if not isinstance(r, dict):
                        continue
                    role = str(r.get("role") or "").strip()
                    if role == TIME_TAU_PTR_ROLE_C3_DERIVED_WORLDS_MANIFEST:
                        dw_ptr_rel = str(r.get("relpath") or "").strip()
                        break
        except Exception:
            dw_ptr_rel = None

    if dw_ptr_rel:
        dw_activated = True
    else:
        dw_activated = None
    _add("TAU_DERIVED_WORLDS_ACTIVATED", dw_activated, detail=(f"relpath={dw_ptr_rel}" if dw_ptr_rel else None))
    if dw_activated:
        dw_ptr_canonical_ok = (dw_ptr_rel == TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_REL)
        _add(
            "TAU_DERIVED_WORLDS_POINTER_CANONICAL",
            dw_ptr_canonical_ok,
            detail=(None if dw_ptr_canonical_ok else f"expected {TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_REL} got {dw_ptr_rel}"),
        )
    else:
        _add("TAU_DERIVED_WORLDS_POINTER_CANONICAL", None)

    # Hygiene: derived worlds manifest must be absent when not activated.
    dw_orphan_ok_required = True
    if not dw_activated:
        try:
            orphan_present = bool(dw_path.exists())
            dw_orphan_ok_required = (not orphan_present)
            _add(
                "TAU_DERIVED_WORLDS_MANIFEST_ABSENT_WHEN_INACTIVE",
                dw_orphan_ok_required,
                detail=(None if dw_orphan_ok_required else "unexpected present"),
                evidence={"path": dw_path.relative_to(bdir).as_posix()},
            )
        except Exception as exc:
            dw_orphan_ok_required = False
            _add(
                "TAU_DERIVED_WORLDS_MANIFEST_ABSENT_WHEN_INACTIVE",
                False,
                detail=f"error: {exc}",
                evidence={"path": dw_path.relative_to(bdir).as_posix()},
            )
    else:
        _add("TAU_DERIVED_WORLDS_MANIFEST_ABSENT_WHEN_INACTIVE", None)

    # --- Check 2: receipts manifest present ---
    rm_present = bool(rm_path.exists())
    _add("TAU_RECEIPTS_MANIFEST_PRESENT", rm_present, detail=(None if rm_present else "missing"), evidence={"path": rm_path.relative_to(bdir).as_posix()})

    # --- Check 3: receipts manifest schema valid ---
    rm_obj: dict | None = None
    rm_schema_ok = None
    rm_detail = None
    if rm_present:
        try:
            rm_raw = rm_path.read_text(encoding="utf-8")
            rm_obj = _json.loads(rm_raw)
            if not isinstance(rm_obj, dict):
                rm_schema_ok = False
                rm_detail = "not an object"
            else:
                if str(rm_obj.get("schema") or "") != "time_tau_c3_receipts_manifest":
                    rm_schema_ok = False
                    rm_detail = "schema mismatch"
                elif str(rm_obj.get("schema_version") or "") not in TIME_TAU_C3_RECEIPTS_MANIFEST_ALLOWED_VERSIONS:
                    rm_schema_ok = False
                    rm_detail = f"schema_version not allowed: {str(rm_obj.get('schema_version') or '')!r} not in {sorted(TIME_TAU_C3_RECEIPTS_MANIFEST_ALLOWED_VERSIONS)!r}"
                else:
                    # Minimal v0.* validation
                    sid_rm = str(rm_obj.get("snapshot_id") or "").strip()
                    if not sid_rm:
                        rm_schema_ok = False
                        rm_detail = "empty snapshot_id"
                    else:
                        rd = str(rm_obj.get("receipts_dir_relpath") or "").strip()
                        if rd != "tau/c3/receipts":
                            rm_schema_ok = False
                            rm_detail = "receipts_dir_relpath mismatch"
                        else:
                            mode = str(rm_obj.get("mode") or "").strip().lower()
                            if mode not in ("present", "empty"):
                                rm_schema_ok = False
                                rm_detail = "mode invalid"
                            else:
                                receipts = rm_obj.get("receipts")
                                if receipts is None:
                                    receipts = []
                                if not isinstance(receipts, list):
                                    rm_schema_ok = False
                                    rm_detail = "receipts not a list"
                                else:
                                    if mode == "present" and not receipts:
                                        rm_schema_ok = False
                                        rm_detail = "mode=present but receipts empty"
                                    elif mode == "empty" and receipts:
                                        rm_schema_ok = False
                                        rm_detail = "mode=empty but receipts non-empty"
                                    else:
                                        seen = set()
                                        bad = []
                                        for r in receipts:
                                            if not isinstance(r, dict):
                                                continue
                                            rp = str(r.get("receipt_relpath") or "").strip()
                                            sig = str(r.get("receipt_sig8") or "").strip()
                                            if not rp or not sig:
                                                bad.append("missing receipt_relpath/receipt_sig8")
                                                continue
                                            if not _time_tau__is_bundle_relpath(rp):
                                                bad.append(f"bad receipt_relpath: {rp}")
                                                continue
                                            if not rp.startswith("tau/c3/receipts/"):
                                                bad.append(f"receipt out of dir: {rp}")
                                                continue
                                            if rp in seen:
                                                bad.append(f"duplicate receipt_relpath: {rp}")
                                                continue
                                            seen.add(rp)
                                            if not re.fullmatch(r"[0-9a-fA-F]{8}", sig or ""):
                                                bad.append(f"bad receipt_sig8: {rp}")
                                        if bad:
                                            rm_schema_ok = False
                                            rm_detail = "; ".join(bad[:5])
                                        else:
                                            rm_schema_ok = True
        except Exception as exc:
            rm_schema_ok = False
            rm_detail = f"error: {exc}"
    _add("TAU_RECEIPTS_MANIFEST_SCHEMA_VALID", rm_schema_ok, detail=rm_detail)

    # --- Optional derived worlds manifest (only if activated) ---
    dw_obj: dict | None = None
    dw_present = None
    dw_schema_ok = None
    dw_detail = None
    if dw_activated:
        dw_rel = str(dw_ptr_rel or "").strip() or TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_REL
        dw_fp = (bdir / dw_rel)
        dw_present = bool(dw_fp.exists())
        _add(
            "TAU_DERIVED_WORLDS_MANIFEST_PRESENT",
            dw_present,
            detail=(None if dw_present else "missing"),
            evidence={"path": dw_rel},
        )
        if dw_ptr_canonical_ok is not True:
            dw_schema_ok = False
            dw_detail = "pointer relpath non-canonical"
        elif not dw_present:
            dw_schema_ok = False
            dw_detail = "missing"
        else:
            try:
                dw_raw = dw_fp.read_text(encoding="utf-8")
                dw_obj = _json.loads(dw_raw)
                if not isinstance(dw_obj, dict):
                    dw_schema_ok = False
                    dw_detail = "not an object"
                else:
                    bad: list[str] = []
                    if str(dw_obj.get("schema") or "") != TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA:
                        bad.append("schema mismatch")
                    elif str(dw_obj.get("schema_version") or "") not in TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_ALLOWED_VERSIONS:
                        bad.append("schema_version mismatch")
                    else:
                        sid_dw = str(dw_obj.get("snapshot_id") or "").strip()
                        if not sid_dw:
                            bad.append("empty snapshot_id")
                        base = str(dw_obj.get("base_dir_relpath") or "").strip()
                        if base != TIME_TAU_C3_DERIVED_WORLDS_BASE_DIR_REL:
                            bad.append("base_dir_relpath mismatch")
                        mode_dw = str(dw_obj.get("mode") or "").strip().lower()
                        worlds = dw_obj.get("worlds")
                        if worlds is None:
                            worlds = []
                        if not isinstance(worlds, list):
                            bad.append("worlds not a list")
                        else:
                            if mode_dw not in {"present", "empty"}:
                                bad.append("bad mode")
                            if mode_dw == "present" and len(worlds) == 0:
                                bad.append("present but empty worlds")
                            if mode_dw == "empty" and len(worlds) != 0:
                                bad.append("empty mode but nonempty worlds")
                            seen: set[str] = set()
                            for w in worlds:
                                if not isinstance(w, dict):
                                    bad.append("non-object world entry")
                                    continue
                                rp = str(w.get("relpath") or "").strip()
                                sig = str(w.get("sig8") or "").strip()
                                if not rp or not sig:
                                    bad.append("empty relpath/sig8")
                                    continue
                                if not _time_tau__is_bundle_relpath(rp):
                                    bad.append(f"bad world relpath: {rp}")
                                    continue
                                if not rp.startswith(TIME_TAU_C3_DERIVED_WORLDS_BASE_DIR_REL + "/"):
                                    bad.append(f"world out of dir: {rp}")
                                    continue
                                if rp in seen:
                                    bad.append(f"duplicate world relpath: {rp}")
                                    continue
                                seen.add(rp)
                                if not re.fullmatch(r"[0-9a-fA-F]{8}", sig or ""):
                                    bad.append(f"bad world sig8: {rp}")
                    if bad:
                        dw_schema_ok = False
                        dw_detail = "; ".join(bad[:5])
                    else:
                        dw_schema_ok = True
            except Exception as exc:
                dw_schema_ok = False
                dw_detail = f"error: {exc}"
        _add("TAU_DERIVED_WORLDS_MANIFEST_SCHEMA_VALID", dw_schema_ok, detail=dw_detail)
    else:
        _add("TAU_DERIVED_WORLDS_MANIFEST_PRESENT", None)
        _add("TAU_DERIVED_WORLDS_MANIFEST_SCHEMA_VALID", None)


    # --- Check 4: closure computable ---
    closure_computable = None
    closure_detail = None
    closure_entries: list[dict] = []
    receipts_lb_sig8: str | None = None
    derived_worlds_lb_sig8: str | None = None

    if ptr_ok and rm_schema_ok is True and ptr_obj is not None and rm_obj is not None:
        try:
            # Derived worlds affects τ identity iff activated (pointer present).
            if dw_activated and dw_schema_ok is not True:
                closure_computable = False
                closure_detail = "derived worlds activated but manifest invalid"
            else:
                sid_ptr = str(ptr_obj.get("snapshot_id") or "").strip()
                sid_rm = str(rm_obj.get("snapshot_id") or "").strip()
                if sid_ptr != sid_rm:
                    closure_computable = False
                    closure_detail = "snapshot_id mismatch between pointer_set and receipts_manifest"
                else:
                    snapshot_id = sid_ptr

                    # If activated, derived-worlds manifest must agree on snapshot_id.
                    if dw_activated:
                        sid_dw = str((dw_obj or {}).get("snapshot_id") or "").strip()
                        if sid_dw != snapshot_id:
                            closure_computable = False
                            closure_detail = "snapshot_id mismatch between pointer_set and derived_worlds_manifest"

                    if closure_computable is not False:
                        # Pointers (surface only)
                        sps = ptr_obj.get("surface_pointers") or []
                        if isinstance(sps, list):
                            for r in sps:
                                if not isinstance(r, dict):
                                    continue
                                rel = str(r.get("relpath") or "").strip()
                                if not rel:
                                    continue
                                closure_entries.append({"relpath": rel, "expectation": "exists", "expected_sig8": ""})

                        # Receipts manifest (exists)
                        rm_rel = rm_path.relative_to(bdir).as_posix()
                        closure_entries.append({"relpath": rm_rel, "expectation": "exists", "expected_sig8": ""})

                        # Receipt files (sig8_equals)
                        receipts = rm_obj.get("receipts") or []
                        if isinstance(receipts, list):
                            for rr in receipts:
                                if not isinstance(rr, dict):
                                    continue
                                rp = str(rr.get("receipt_relpath") or "").strip()
                                sig = str(rr.get("receipt_sig8") or "").strip()
                                if not rp:
                                    continue
                                closure_entries.append({"relpath": rp, "expectation": "sig8_equals", "expected_sig8": sig})

                        # Derived world files (sig8_equals) iff activated.
                        if dw_activated:
                            worlds = (dw_obj or {}).get("worlds") or []
                            if isinstance(worlds, list):
                                for w in worlds:
                                    if not isinstance(w, dict):
                                        continue
                                    rp = str(w.get("relpath") or "").strip()
                                    sig = str(w.get("sig8") or "").strip()
                                    if not rp:
                                        continue
                                    closure_entries.append({"relpath": rp, "expectation": "sig8_equals", "expected_sig8": sig})

                        closure_entries = _time_tau__canon_closure_entries(closure_entries)
                        # Detect conflict marker
                        if any(str(e.get("expectation")) == "CONFLICT" for e in closure_entries):
                            closure_computable = False
                            closure_detail = "closure conflict (sig8 mismatch on same relpath)"
                        else:
                            closure_obj = {
                                "schema": TIME_TAU_CLOSURE_SET_SCHEMA,
                                "schema_version": TIME_TAU_CLOSURE_SET_SCHEMA_VERSION,
                                "snapshot_id": snapshot_id,
                                "entries": closure_entries,
                            }
                            closure_set_sig8 = hash_json_sig8(closure_obj)

                            # Control-manifest LB sig8 (receipts manifest)
                            receipts_lb_sig8 = hash_json_sig8(_time_tau__lb_projection_receipts_manifest(rm_obj))

                            # Control-manifest LB sig8 (derived worlds manifest)
                            if dw_activated:
                                derived_worlds_lb_sig8 = hash_json_sig8(
                                    _time_tau__lb_projection_derived_worlds_manifest(dw_obj or {})
                                )

                            closure_computable = True
        except Exception as exc:
            closure_computable = False
            closure_detail = f"error: {exc}"

    _add("TAU_CLOSURE_COMPUTABLE", closure_computable, detail=closure_detail)

    # --- Check 5: no-scan conformance ---
    # This is a contract check: this implementation constructs closure only from explicit manifests.
    no_scan_ok = True if closure_computable else None
    _add("TAU_CLOSURE_NO_SCAN_CONFORMANCE", no_scan_ok)

    # --- Check 6: closure all files exist ---
    all_exist = None
    missing: list[str] = []
    if closure_computable is True and closure_entries:
        try:
            for e in closure_entries:
                rel = str(e.get("relpath") or "").strip()
                if not rel:
                    continue
                fp = (bdir / rel)
                if not fp.exists():
                    missing.append(rel)
            all_exist = (len(missing) == 0)
        except Exception as exc:
            all_exist = False
            missing = [f"error: {exc}"]
    _add("TAU_CLOSURE_ALL_FILES_EXIST", all_exist, detail=(None if not missing else f"missing={len(missing)}"), evidence={"missing": sorted(missing)} if missing else None)

    # --- Check 7: closure sig8 matches (sig8_equals entries) ---
    sig8_matches = None
    mismatches: list[str] = []
    if closure_computable is True and all_exist is True:
        try:
            for e in closure_entries:
                if str(e.get("expectation") or "") != "sig8_equals":
                    continue
                rel = str(e.get("relpath") or "").strip()
                expected = str(e.get("expected_sig8") or "").strip()
                fp = (bdir / rel)
                actual = _hash_file_sig8(fp)
                if expected and actual != expected:
                    mismatches.append(f"{rel} expected={expected} actual={actual}")
            sig8_matches = (len(mismatches) == 0)
        except Exception as exc:
            sig8_matches = False
            mismatches = [f"error: {exc}"]
    _add("TAU_CLOSURE_SIG8_MATCHES", sig8_matches, evidence={"mismatches": mismatches} if mismatches else None)

    # --- Compute τ surface (only if closure realized) ---
    closure_realized = (closure_computable is True and all_exist is True and sig8_matches is True)
    if closure_realized:
        try:
            # Per-entry actual sig8 (control manifests use LB projection sig8)
            files_list: list[dict] = []
            rm_rel = rm_path.relative_to(bdir).as_posix()
            dw_rel = dw_path.relative_to(bdir).as_posix()
            for e in closure_entries:
                rel = str(e.get("relpath") or "").strip()
                if not rel:
                    continue
                if rel == rm_rel and receipts_lb_sig8 is not None:
                    files_list.append({"relpath": rel, "sig8": receipts_lb_sig8})
                elif rel == dw_rel and derived_worlds_lb_sig8 is not None:
                    files_list.append({"relpath": rel, "sig8": derived_worlds_lb_sig8})
                else:
                    files_list.append({"relpath": rel, "sig8": _hash_file_sig8(bdir / rel)})
            files_list.sort(key=lambda x: str((x or {}).get("relpath") or ""))
            surface_core = {
                "schema": TIME_TAU_SURFACE_SCHEMA,
                "schema_version": TIME_TAU_SURFACE_SCHEMA_VERSION,
                "snapshot_id": snapshot_id,
                "closure_set_sig8": closure_set_sig8,
                "files": files_list,
            }
            tau_surface_sig8 = hash_json_sig8(surface_core)
        except Exception as exc:
            diagnostics["surface_error"] = str(exc)
            tau_surface_sig8 = None

    # --- Check 8: surface defined iff closure realized ---
    surface_gate_ok = (bool(tau_surface_sig8) == bool(closure_realized))
    _add("TAU_SURFACE_DEFINED_IFF_CLOSURE_REALIZED", surface_gate_ok)

    # --- Check 9: surface computable ---
    surface_computable = True if tau_surface_sig8 else (False if closure_realized else None)
    _add("TAU_SURFACE_COMPUTABLE", surface_computable)

    # --- Regime selectors (DW + T0) ---
    # DW selector (existing surface)
    dw_regime = str(derived_worlds_regime or "").strip().upper()
    regime_supported = dw_regime in {"POINTER_GATED", "MANDATORY"}

    # T0 selector (new surface; absence collapses to OPTIONAL)
    t0_regime_norm = str(t0_regime or "").strip().upper()
    if not t0_regime_norm:
        t0_regime_norm = "OPTIONAL"
    t0_regime_supported = t0_regime_norm in {"OPTIONAL", "MANDATORY"}

    # --- Check 10: T0 diagnostic (becomes load-bearing only under t0_regime=MANDATORY) ---
    t0_ok = None
    if t0_regime_norm == "MANDATORY":
        # Evaluate only when closure is realized (we have a trusted, realized inventory).
        if closure_realized is True:
            try:
                receipts = (rm_obj or {}).get("receipts") or []
                if not isinstance(receipts, list):
                    receipts = []
                if len(receipts) == 0:
                    # Vacuous truth over an empty inventory.
                    t0_ok = True
                else:
                    ok_all = True
                    for rr in receipts:
                        if not isinstance(rr, dict):
                            ok_all = False
                            break
                        rp = str(rr.get("receipt_relpath") or "").strip()
                        if not rp:
                            ok_all = False
                            break
                        fp = (bdir / rp)
                        try:
                            raw = fp.read_text(encoding="utf-8")
                            j = _json.loads(raw)
                        except Exception:
                            ok_all = False
                            break
                        if not isinstance(j, dict):
                            ok_all = False
                            break
                        fr = j.get("tau_frame_receipt")
                        if not isinstance(fr, dict):
                            ok_all = False
                            break
                        if str(fr.get("schema") or "") != TIME_TAU_T0_FRAME_SCHEMA:
                            ok_all = False
                            break
                        if str(fr.get("schema_version") or "") != TIME_TAU_T0_FRAME_SCHEMA_VERSION:
                            ok_all = False
                            break
                    t0_ok = True if ok_all else False
            except Exception:
                t0_ok = False
        else:
            t0_ok = None
    _add("TAU_OPTIONAL_T0_DIAGNOSTIC", t0_ok)

    # Verdict rule (v0.2): all required checks must be True (status-gated at the receipt surface).
    # Track III tail: export the required-set surface so certificates can be
    # a pure value-map (no re-verification logic).

    # --- Required-set surface (regime-dependent) ---
    # Pointer-gated (v0.2) is the default; mandatory DW (v0.3 policy profile) promotes
    # the derived-worlds manifest checks to required and removes the "orphan when inactive" check.

    if dw_regime == "MANDATORY":
        required_check_ids = [
            "TAU_POINTER_SET_AVAILABLE",
            "TAU_RECEIPTS_MANIFEST_PRESENT",
            "TAU_RECEIPTS_MANIFEST_SCHEMA_VALID",
            "TAU_DERIVED_WORLDS_MANIFEST_PRESENT",
            "TAU_DERIVED_WORLDS_MANIFEST_SCHEMA_VALID",
            "TAU_CLOSURE_COMPUTABLE",
            "TAU_CLOSURE_NO_SCAN_CONFORMANCE",
            "TAU_CLOSURE_ALL_FILES_EXIST",
            "TAU_CLOSURE_SIG8_MATCHES",
            "TAU_SURFACE_DEFINED_IFF_CLOSURE_REALIZED",
            "TAU_SURFACE_COMPUTABLE",
        ]
        required_ok = [
            ptr_ok,
            rm_present,
            (rm_schema_ok is True),
            (dw_present is True),
            (dw_schema_ok is True),
            (closure_computable is True),
            (no_scan_ok is True),
            (all_exist is True),
            (sig8_matches is True),
            (surface_gate_ok is True),
            (surface_computable is True),
        ]
    else:
        # Default: pointer-gated derived worlds (current v0.2 behavior).
        required_check_ids = [
            "TAU_POINTER_SET_AVAILABLE",
            "TAU_RECEIPTS_MANIFEST_PRESENT",
            "TAU_RECEIPTS_MANIFEST_SCHEMA_VALID",
            "TAU_DERIVED_WORLDS_MANIFEST_ABSENT_WHEN_INACTIVE",
            "TAU_CLOSURE_COMPUTABLE",
            "TAU_CLOSURE_NO_SCAN_CONFORMANCE",
            "TAU_CLOSURE_ALL_FILES_EXIST",
            "TAU_CLOSURE_SIG8_MATCHES",
            "TAU_SURFACE_DEFINED_IFF_CLOSURE_REALIZED",
            "TAU_SURFACE_COMPUTABLE",
        ]
        required_ok = [
            ptr_ok,
            rm_present,
            (rm_schema_ok is True),
            (dw_orphan_ok_required is True),
            (closure_computable is True),
            (no_scan_ok is True),
            (all_exist is True),
            (sig8_matches is True),
            (surface_gate_ok is True),
            (surface_computable is True),
        ]

    # T0 mandatory: append-only promotion of the existing diagnostic check into the required set.
    if t0_regime_norm == "MANDATORY":
        required_check_ids = list(required_check_ids) + ["TAU_OPTIONAL_T0_DIAGNOSTIC"]
        required_ok = list(required_ok) + [(t0_ok is True)]

    # v0.2: status-gated verdict surface.
    # Unknown regime selectors are rejected (NOT_ATTEMPTED) rather than defaulted.
    precond_ok = (
        regime_supported
        and t0_regime_supported
        and (ptr_ok is True)
        and (rm_present is True)
        and (rm_schema_ok is True)
    )
    status = "OK" if precond_ok else "NOT_ATTEMPTED"
    verdict_ok = (bool(all(required_ok)) if status == "OK" else None)

    receipt = {
        "schema": TIME_TAU_VERIFY_RECEIPT_SCHEMA,
        "schema_version": TIME_TAU_VERIFY_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "snapshot_id": snapshot_id,
        "verdict_ok": verdict_ok,
        "required_check_ids": required_check_ids,
        "required_ok": required_ok,
        "closure_set_sig8": closure_set_sig8,
        "tau_surface_sig8": tau_surface_sig8,
        "checks": checks,
        "diagnostics": diagnostics,
    }
    return receipt


def _b4_verify_and_prune_bundle_tree(bundle_dir: _Path, state: dict) -> dict:
    """B4: Verify (and optionally prune) the bundle tree.

    Current stage behavior (wiring stage):
      - Verification-only (no pruning).
      - Enforces a coarse top-level allowlist so UI/log junk cannot sneak into the bundle.
      - Ensures a small set of required core files exist.

    Returns an admissibility report dict suitable for writing into the bundle.
    """
    bdir = _Path(bundle_dir).resolve()
    st = state or {}
    sid = str(st.get("snapshot_id") or "").strip()
    manifest = (st.get("bundle_manifest") or {}) if isinstance(st.get("bundle_manifest"), dict) else {}
    msig8 = str(manifest.get("sig8") or "").strip()

    violations: list[dict] = []
    pruned_paths: list[str] = []

    # Top-level allowlist: only canonical bundle dirs + legacy root manifest mirror are allowed.
    allowed_root_dirs = {"meta", "world", "manifests", "certs", "coverage", "tau"}
    allowed_root_files = {"bundle_manifest.json"}

    try:
        for child in sorted(bdir.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                if child.name not in allowed_root_dirs:
                    violations.append(
                        {
                            "code": "UNEXPECTED_ROOT_DIR",
                            "path": child.name,
                        }
                    )
            elif child.is_file():
                if child.name not in allowed_root_files:
                    violations.append(
                        {
                            "code": "UNEXPECTED_ROOT_FILE",
                            "path": child.name,
                        }
                    )
    except Exception as exc:
        violations.append({"code": "ROOT_SCAN_FAILED", "detail": str(exc)})
    # Required core files (bundle-relative), including per-fixture cert neighborhoods.
    required_relpaths = sorted(_b4_expected_bundle_inventory(state))

    # Structural guard: if we cannot enumerate fixtures, B4 must fail loudly.
    fx = (st.get("fixtures") if isinstance(st, dict) else None)
    if not isinstance(fx, dict) or not fx:
        violations.append(
            {
                "code": "MISSING_FIXTURE_INVENTORY",
                "detail": "state.fixtures is empty/missing; cannot certify per-fixture neighborhoods.",
            }
        )

    for rel in required_relpaths:
        if not (bdir / rel).exists():
            violations.append({"code": "MISSING_REQUIRED_FILE", "path": rel})

    status = "PASS" if not violations else "FAIL"
    report = {
        "schema": "b4_admissibility_report",
        "schema_version": B4_ADMISSIBILITY_REPORT_SCHEMA_VERSION,
        "snapshot_id": sid,
        "bundle_manifest_sig8": msig8,
        "status": status,
        "violations": violations,
        "pruned_paths": pruned_paths,
    }
    return report


def _b4_write_admissibility_report(bundle_dir: _Path, report: dict) -> _Path:
    """B4: Write admissibility report into meta/b4_admissibility.json."""
    bdir = _Path(bundle_dir)
    meta_dir = _b1_rel_dirs_for_root(bdir)["meta"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    outp = meta_dir / "b4_admissibility.json"
    outp.write_text(canonical_json(report or {}), encoding="utf-8")
    return outp


# --- B5 semantic identity (per-fixture FP + bundle-level object-set index) ---

def _b5_cjf1_dumps(obj: object) -> str:
    """Canonical JSON for B5 identity surfaces.

    Requirements (CJF-1 discipline):
      - sort_keys=True
      - separators=(',', ':')
      - ensure_ascii=True
      - allow_nan=False (no NaN/Inf)

    We intentionally do *not* apply the v2 ephemeral-key dropper here.
    B5 surfaces are closed-world: callers must construct the exact surface.
    """
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _b5_sha256_hex(obj: object) -> str:
    """SHA-256 hex digest of the B5 CJF-1 canonical JSON for obj."""
    txt = _b5_cjf1_dumps(obj)
    return _hash.sha256(txt.encode("utf-8")).hexdigest()


def _b5_bitmatrix(mat: object, *, name: str) -> list[list[int]]:
    """Validate + normalize a bitmatrix (list[list[int]]) for B5.

    - Converts all entries to int & 1
    - Rejects floats, None, ragged rows, and non-list containers
    """
    if not isinstance(mat, list):
        raise TypeError(f"B5: {name} must be a list (rows), got {type(mat).__name__}")
    out: list[list[int]] = []
    width: int | None = None
    for r_i, row in enumerate(mat):
        if not isinstance(row, list):
            raise TypeError(f"B5: {name}[{r_i}] must be a list, got {type(row).__name__}")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                f"B5: {name} is ragged (row {r_i} has len={len(row)}; expected {width})"
            )
        out_row: list[int] = []
        for c_i, v in enumerate(row):
            if v is None:
                raise ValueError(f"B5: {name}[{r_i}][{c_i}] is None")
            if isinstance(v, bool):
                out_row.append(1 if v else 0)
                continue
            if isinstance(v, float):
                raise TypeError(f"B5: {name}[{r_i}][{c_i}] must not be float")
            try:
                iv = int(v)
            except Exception:
                raise TypeError(
                    f"B5: {name}[{r_i}][{c_i}] must be int-like, got {type(v).__name__}"
                )
            out_row.append(iv & 1)
        out.append(out_row)

    # Empty matrix is allowed, but dims must still be explicit at the caller level.
    return out


def _b5_dims_of_bitmatrix(m: list[list[int]]) -> list[int]:
    """Return [n_rows, n_cols] for a normalized bitmatrix."""
    n_rows = len(m)
    n_cols = len(m[0]) if (n_rows > 0 and isinstance(m[0], list)) else 0
    return [int(n_rows), int(n_cols)]


def _b5_fp_core_from_blocks(bB: dict, bC: dict, bH: dict) -> dict:
    """Build the B5 FP *core* surface for the overlap strict-core instance.

    Current stage: the semantic object is the strict-core triple (d3, C3, H2)
    over F2.
    """
    d3_raw = (bB or {}).get("3")
    C3_raw = (bC or {}).get("3")
    H2_raw = (bH or {}).get("2")
    if d3_raw is None or C3_raw is None or H2_raw is None:
        raise ValueError("B5: missing required blocks for FP core (need B[3], C[3], H[2])")

    d3 = _b5_bitmatrix(d3_raw, name="d3")
    C3 = _b5_bitmatrix(C3_raw, name="C3")
    H2 = _b5_bitmatrix(H2_raw, name="H2")

    core = {
        "schema": B5_FP_CORE_SCHEMA_VERSION,
        "field": "F2",
        # Explicit element-space signatures (Hidden Gap 2: empty matrices still need dims).
        "dims": {
            "d3": _b5_dims_of_bitmatrix(d3),
            "C3": _b5_dims_of_bitmatrix(C3),
            "H2": _b5_dims_of_bitmatrix(H2),
        },
        "d3": d3,
        "C3": C3,
        "H2": H2,
    }
    return core


def build_b5_identity_payload(
    *,
    snapshot_id: str | None,
    district_id: str,
    fixture_label: str,
    blocks_B: dict,
    blocks_C: dict,
    blocks_H: dict,
    inputs_sig_5: list[str] | None = None,
) -> dict:
    """Pure builder for a per-fixture B5 identity record."""
    core = _b5_fp_core_from_blocks(blocks_B, blocks_C, blocks_H)
    b5_fp_hex = _b5_sha256_hex(core)
    payload: dict = {
        "schema": "b5_identity",
        "schema_version": B5_IDENTITY_SCHEMA_VERSION,
        "b5_fp_hex": b5_fp_hex,
        "b5_fp_sig8": b5_fp_hex[:8],
        "core": core,
        # Annex is recorded for audit/debug but must not affect semantic identity.
        "annex": {
            "snapshot_id": str(snapshot_id or ""),
            "district_id": str(district_id or ""),
            "fixture_label": str(fixture_label or ""),
            "inputs_sig_5": list(inputs_sig_5 or []),
        },
    }

    # Optional: stamp a B3-style payload_sig8 for the record itself.
    # We exclude annex so file integrity hashes track the semantic core.
    try:
        nc = set(_B3_NON_CORE_COMMON_KEYS) | {"annex"}
        b3_stamp_payload_sig8(payload, non_core_keys=nc, include_full_sha256=True)
    except Exception:
        pass

    return payload


def build_b5_identity_payload_from_paths(
    *,
    snapshot_id: str | None,
    district_id: str,
    fixture_label: str,
    paths: dict,
    inputs_sig_5: list[str] | None = None,
) -> dict:
    """Load B/C/H from paths and build the B5 identity payload."""
    if not isinstance(paths, dict):
        raise TypeError("B5: paths must be a dict")
    Bp = paths.get("B")
    Cp = paths.get("C")
    Hp = paths.get("H")
    if not (Bp and Cp and Hp):
        raise ValueError("B5: missing required B/C/H paths")

    Bj, _, _ = abx_read_json_any(Bp, kind="B")
    Cj, _, _ = abx_read_json_any(Cp, kind="C")
    Hj, _, _ = abx_read_json_any(Hp, kind="H")
    bB = _svr_as_blocks_v2(Bj, "B")
    bC = _svr_as_blocks_v2(Cj, "C")
    bH = _svr_as_blocks_v2(Hj, "H")

    return build_b5_identity_payload(
        snapshot_id=snapshot_id,
        district_id=district_id,
        fixture_label=fixture_label,
        blocks_B=bB,
        blocks_C=bC,
        blocks_H=bH,
        inputs_sig_5=inputs_sig_5,
    )


def _b5_compute_index_from_manifest(
    manifest_path: _Path | str,
    *,
    snapshot_id: str | None = None,
    strict: bool = True,
) -> dict:
    """Compute the bundle-level B5 object-set index from a v2 manifest JSONL.

    Core (hashed) surface is the sorted list of member B5-FP hex digests.
    Annex carries per-fixture mapping for audit/debug.
    """
    mp = _Path(manifest_path)
    if not mp.exists():
        raise RuntimeError(f"B5: manifest_full_scope.jsonl not found: {mp}")

    # Deduplicate by fixture_label while preserving deterministic order
    # (manifest regen sorts fixture_label keys).
    by_fixture: dict[str, str] = {}
    rows_seen: set[str] = set()
    errors: list[dict] = []

    with mp.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except Exception as exc:
                errors.append({"code": "BAD_JSONL", "lineno": lineno, "detail": str(exc)})
                continue
            if not isinstance(row, dict):
                continue

            fid = str(row.get("fixture_label") or "")
            if not fid:
                continue
            if fid in rows_seen:
                # manifest regen should already dedup; ignore duplicates defensively.
                continue
            rows_seen.add(fid)

            paths = row.get("paths") or {}
            if not isinstance(paths, dict):
                errors.append({"code": "MISSING_PATHS", "fixture_label": fid})
                continue

            try:
                ident = build_b5_identity_payload_from_paths(
                    snapshot_id=snapshot_id,
                    district_id=str(row.get("district_id") or ""),
                    fixture_label=fid,
                    paths=paths,
                    inputs_sig_5=None,
                )
                by_fixture[fid] = str(ident.get("b5_fp_hex") or "")
            except Exception as exc:
                errors.append({"code": "B5_FP_FAIL", "fixture_label": fid, "detail": str(exc)})

    if errors and strict:
        raise RuntimeError(f"B5: failed to compute FP for {len(errors)} manifest rows: {errors[:3]}")

    members = [v for v in by_fixture.values() if isinstance(v, str) and v]
    members_sorted = sorted(set(members))
    core = {
        "schema": B5_INDEX_SCHEMA_VERSION,
        "members": members_sorted,
    }
    b5_set_hex = _b5_sha256_hex(core)

    payload: dict = {
        "schema": "b5_index",
        "schema_version": B5_INDEX_SCHEMA_VERSION,
        "snapshot_id": str(snapshot_id or ""),
        "b5_set_hex": b5_set_hex,
        "b5_set_sig8": b5_set_hex[:8],
        "members": members_sorted,
        "annex": {
            "n_fixtures": len(by_fixture),
            "by_fixture": by_fixture,
            "errors": errors,
        },
    }
    try:
        nc = set(_B3_NON_CORE_COMMON_KEYS) | {"annex"}
        b3_stamp_payload_sig8(payload, non_core_keys=nc, include_full_sha256=True)
    except Exception:
        pass

    return payload


def _b5_write_index(bundle_dir: _Path, payload: dict) -> _Path:
    """Write meta/b5_index.v1.json into the bundle tree."""
    bdir = _Path(bundle_dir)
    meta_dir = _b1_rel_dirs_for_root(bdir)["meta"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    outp = meta_dir / "b5_index.v1.json"
    outp.write_text(_b5_cjf1_dumps(payload or {}), encoding="utf-8")
    return outp


class _B6SealViolation(RuntimeError):
    """Internal exception: represents a Profile.v1 violation while sealing/verifying.

    This is *not* an IO error: it means the tree is out-of-regime for the profile,
    or a deterministic profile rule failed (e.g., symlink present, PORTKEY collision).
    """

    def __init__(self, fail_codes: list[str], violations: list[dict] | None = None, *, msg: str | None = None):
        self.fail_codes = sorted(set([str(c) for c in (fail_codes or []) if str(c)]))
        self.violations = list(violations or [])
        super().__init__(msg or ("B6: seal verification failed: " + ",".join(self.fail_codes)))


def _b6_u32be(n: int) -> bytes:
    if n < 0 or n > 0xFFFFFFFF:
        raise ValueError(f"B6ENC1: u32 out of range: {n}")
    return int(n).to_bytes(4, "big", signed=False)


def _b6_u64be(n: int) -> bytes:
    if n < 0 or n > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"B6ENC1: u64 out of range: {n}")
    return int(n).to_bytes(8, "big", signed=False)


def _b6_validate_relpath(relpath: str) -> str:
    """Validate a bundle-relative POSIX relpath for Profile.v1.

    This is primarily defensive (verifiers must not trust relpaths from JSON).
    During sealing, relpaths are derived from the filesystem walk, but we keep
    this check so the profile regime is explicit.
    """
    rel = str(relpath or "").replace("\\", "/")

    if not rel:
        raise _B6SealViolation(
            ["TREE_PATH_INVALID"],
            [{"code": "TREE_PATH_INVALID", "relpath": rel, "detail_code": "EMPTY_REL", "detail": "empty relpath"}],
        )

    if rel.startswith("/"):
        raise _B6SealViolation(
            ["TREE_PATH_INVALID"],
            [{"code": "TREE_PATH_INVALID", "relpath": rel, "detail_code": "ABS_PATH", "detail": "relpath is absolute"}],
        )

    if "\x00" in rel or "\x00" in rel:
        raise _B6SealViolation(
            ["TREE_PATH_INVALID"],
            [{"code": "TREE_PATH_INVALID", "relpath": rel, "detail_code": "NUL", "detail": "NUL byte in relpath"}],
        )

    parts = rel.split("/")
    if any(p == "" for p in parts):
        raise _B6SealViolation(
            ["TREE_PATH_INVALID"],
            [{"code": "TREE_PATH_INVALID", "relpath": rel, "detail_code": "EMPTY_SEG", "detail": "empty relpath segment"}],
        )
    if any(p in (".", "..") for p in parts):
        raise _B6SealViolation(
            ["TREE_PATH_INVALID"],
            [{"code": "TREE_PATH_INVALID", "relpath": rel, "detail_code": "DOT_SEG", "detail": "dot segment in relpath"}],
        )

    try:
        rel.encode("utf-8", "strict")
    except Exception:
        raise _B6SealViolation(
            ["TREE_REL_UTF8_INVALID"],
            [{"code": "TREE_REL_UTF8_INVALID", "relpath": rel, "detail_code": "UTF8", "detail": "relpath not strict UTF-8"}],
        )

    return rel


def _b6_portkey1(relpath: str) -> str:
    """PORTKEY1: NFC + casefold + strip trailing spaces/dots per segment."""
    import unicodedata as _unicodedata

    rel = str(relpath or "")
    segs = rel.split("/")
    out: list[str] = []
    for s in segs:
        s1 = _unicodedata.normalize("NFC", s)
        s2 = s1.casefold()
        # Strip all trailing spaces/dots (Windows collapse behavior).
        s3 = s2.rstrip(" .")
        out.append(s3)
    return "/".join(out)


def _b6_sha256_file_stream(p: _Path, *, chunk_size: int = 1 << 20) -> tuple[str, int]:
    """Return (sha256_hex, byte_len) for a file using streaming reads."""
    h = _hashlib.sha256()
    n = 0
    pp = _Path(p)
    with pp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _b6_atomic_write_text(p: _Path, text: str) -> None:
    """Atomic text write: write to tmp then replace."""
    path = _Path(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _os.replace(tmp, path)


def _b6enc1_update_hasher(
    h: "_hashlib._Hash",
    *,
    exclude_relpaths: list[str],
    entries: list[dict],
) -> None:
    """Update sha256 hasher with B6ENC1 encoding of (excludes, entries).

    Encoding (B6ENC1):
      MAGIC "B6ENC1\0"
      U32 n_excl, then each excl as (U32 len + bytes)
      U64 n_entries, then each entry:
        U32 len(relpath_bytes) + relpath_bytes
        32 bytes sha256 digest
        U64 byte_len
    """
    h.update(b"B6ENC1\0")

    # Excludes list (sorted lex by UTF-8 bytes for determinism).
    excl_bytes = []
    for x in exclude_relpaths:
        xb = str(x).encode("utf-8", "strict")
        excl_bytes.append(xb)
    excl_bytes.sort()

    h.update(_b6_u32be(len(excl_bytes)))
    for xb in excl_bytes:
        h.update(_b6_u32be(len(xb)))
        h.update(xb)

    # Entries list (sorted lex by relpath UTF-8 bytes).
    def _rk(e: dict) -> bytes:
        return str(e.get("relpath") or "").encode("utf-8", "strict")

    entries_sorted = sorted(entries, key=_rk)

    h.update(_b6_u64be(len(entries_sorted)))
    for e in entries_sorted:
        rel = str(e.get("relpath") or "")
        relb = rel.encode("utf-8", "strict")
        sha_hex = str(e.get("sha256_hex") or "")
        byte_len = int(e.get("byte_len") or 0)

        try:
            sha_bytes = bytes.fromhex(sha_hex)
        except Exception as exc:
            raise ValueError(f"B6ENC1: bad sha256_hex for {rel}: {sha_hex}") from exc
        if len(sha_bytes) != 32:
            raise ValueError(f"B6ENC1: sha256_hex must be 32 bytes for {rel} (got {len(sha_bytes)})")

        h.update(_b6_u32be(len(relb)))
        h.update(relb)
        h.update(sha_bytes)
        h.update(_b6_u64be(byte_len))


def b6_compute_seal_profile_v1(bundle_dir: _Path) -> dict:
    """B6(Profile.v1): compute the global bundle seal over the materialized tree.

    Profile.v1 regime:
      - Symlinks disallowed (file or dir)
      - Only (dir | regular file) nodes permitted
      - PORTKEY1 collision freedom over all files (including excluded)
      - Exclude policy is fixed: B6_EXCLUDE_REL_PATHS_V1
      - Encoding is B6ENC1 (self-delimiting), seal hash is sha256
    """
    bdir = _Path(bundle_dir).resolve()

    exclude = set(B6_EXCLUDE_REL_PATHS_V1)
    exclude_sorted = sorted(_b6_validate_relpath(x) for x in exclude)

    entries: list[dict] = []
    port_seen: dict[str, str] = {}

    def _walk(abs_dir: _Path, rel_prefix: str = "") -> None:
        try:
            with _os.scandir(abs_dir) as it:
                dentries = sorted(list(it), key=lambda de: de.name)
        except Exception as exc:
            raise RuntimeError(f"B6: failed to scan dir: {abs_dir} ({exc})") from exc

        for de in dentries:
            name = de.name
            rel = f"{rel_prefix}/{name}" if rel_prefix else name
            rel = _b6_validate_relpath(rel)

            # Symlink policy (strong move): disallow everywhere.
            try:
                if de.is_symlink():
                    raise _B6SealViolation(
                        ["TREE_HAS_SYMLINK"],
                        [{"code": "TREE_HAS_SYMLINK", "relpath": rel, "detail_code": "SYMLINK", "detail": "symlink encountered"}],
                    )
            except OSError:
                # If lstat fails, treat as IO error (not a profile violation).
                raise

            if de.is_dir(follow_symlinks=False):
                _walk(_Path(de.path), rel)
                continue

            if de.is_file(follow_symlinks=False):
                # PORTKEY1 collision guard applies to all files (including excluded).
                pk = _b6_portkey1(rel)
                prev = port_seen.get(pk)
                if prev is not None and prev != rel:
                    raise _B6SealViolation(
                        ["TREE_PORTKEY_COLLISION"],
                        [{
                            "code": "TREE_PORTKEY_COLLISION",
                            "relpath": prev,
                            "other_relpath": rel,
                            "detail_code": "PORTKEY1",
                            "detail": f"PORTKEY1 collision: {pk}",
                            "observed": pk,
                        }],
                    )
                port_seen[pk] = rel

                # Exclude policy (fixed for v1).
                if rel in exclude:
                    continue

                sha_hex, nbytes = _b6_sha256_file_stream(_Path(de.path))
                entries.append({"relpath": rel, "sha256_hex": sha_hex, "byte_len": nbytes})
                continue

            # Any other node kind is forbidden.
            raise _B6SealViolation(
                ["TREE_HAS_SPECIAL_NODE"],
                [{"code": "TREE_HAS_SPECIAL_NODE", "relpath": rel, "detail_code": "SPECIAL", "detail": "non-(dir|file) node"}],
            )

    _walk(bdir)

    # Deterministic entry ordering (lex on relpath).
    entries.sort(key=lambda e: str(e.get("relpath") or ""))

    # Compute seal digest over B6ENC1 encoding.
    h = _hashlib.sha256()
    _b6enc1_update_hasher(h, exclude_relpaths=exclude_sorted, entries=entries)
    digest_hex = h.hexdigest()

    payload: dict = {
        "schema": B6_SEAL_SCHEMA,
        "schema_version": B6_SEAL_SCHEMA_VERSION,
        "seal_profile_id": B6_SEAL_PROFILE_ID,
        "inventory_encoding_id": B6_INVENTORY_ENCODING_ID,
        "portability_key_id": B6_PORTABILITY_KEY_ID,
        "exclude_policy_id": B6_EXCLUDE_POLICY_ID,
        "seal_hash_algo": B6_SEAL_HASH_ALGO,
        "file_hash_algo": B6_FILE_HASH_ALGO,
        "digest_hex": digest_hex,
        "digest_sig8": digest_hex[:8],
        "n_files": len(entries),
        # Witness fields (non-core): helpful for audit/debug.
        "exclude_relpaths": exclude_sorted,
        "entries": entries,
    }

    # Convenience linkage (non-core): snapshot_id + bundle_manifest sig8.
    try:
        mp = bdir / "meta" / "bundle_manifest.json"
        if mp.exists():
            mo = _json.loads(mp.read_text(encoding="utf-8"))
            if isinstance(mo, dict):
                payload["snapshot_id"] = str(mo.get("snapshot_id") or "")
                payload["bundle_manifest_sig8"] = str(mo.get("sig8") or "")
    except Exception:
        pass

    return payload


def b6_write_seal(bundle_dir: _Path, payload: dict) -> _Path:
    """Write the B6 seal sidecar into meta/bundle_hash.json (canonical path)."""
    bdir = _Path(bundle_dir)
    meta_dir = _b1_rel_dirs_for_root(bdir)["meta"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    outp = meta_dir / "bundle_hash.json"
    _b6_atomic_write_text(outp, canonical_json(payload or {}))
    return outp




# --- Sig8 audit helpers (observability only; never affects solver verdicts) ---

SIG8_AUDIT_SCHEMA = "sig8_audit"
SIG8_AUDIT_SCHEMA_VERSION = "sig8_audit.v0.1"


def sig8_expected_for_payload(payload: dict) -> str | None:
    """Best-effort expected sig8 for a known, self-sig8-stamped artifact payload.

    Returns:
      - expected sig8 (8 hex chars) if schema+version is recognized
      - None if the payload family is unknown or expected cannot be computed safely

    NOTE: This is *audit-only* plumbing. Callers must not treat None as a failure.
    """
    if not isinstance(payload, dict):
        return None

    schema = str(payload.get("schema") or "").strip()
    schema_version = str(payload.get("schema_version") or "").strip()

    # Bundle manifest (no `schema`; schema_version is the binder).
    # Audit handshake matches `stamp_bundle_manifest_sig8`, but we inline it here
    # so this helper stays safe even if this monolith module is still loading.
    if not schema and "pointers" in payload and "snapshot_id" in payload and "sig8" in payload:
        try:
            base = dict(payload)
            base.setdefault("schema_version", BUNDLE_MANIFEST_SCHEMA_VERSION)

            tmp = dict(base)
            tmp["sig8"] = ""

            non_core = set(_B3_NON_CORE_COMMON_KEYS) | {
                "sig8",
                "engine_rev",
                "meta",
                "logs",
                "b4",
                "b5",
                "b6",
            }

            return str(b3_payload_sig8(tmp, non_core_keys=non_core) or "").strip() or None
        except Exception:
            return None

    # D4 certificate (v2 handshake: hash_json_sig8(cert with sig8 cleared)).
    try:
        if schema == D4_CERTIFICATE_SCHEMA and schema_version == D4_CERTIFICATE_SCHEMA_VERSION:
            tmp = dict(payload)
            tmp["sig8"] = ""
            return str(hash_json_sig8(tmp) or "").strip() or None
    except Exception:
        return None

    # Time(τ) pointer set (v0.2 handshake: hash_json_sig8(H_ptr), quarantining annex/unknown keys).
    try:
        if schema == TIME_TAU_POINTER_SET_SCHEMA and schema_version in TIME_TAU_POINTER_SET_ALLOWED_VERSIONS:
            hb = time_tau_pointer_set_hash_body_v0_2(payload)
            return str(hash_json_sig8(hb) or "").strip() or None
    except Exception:
        return None

    # Time(τ) receipts manifest (LB projection binds sig8).
    try:
        if schema == "time_tau_c3_receipts_manifest" and schema_version in TIME_TAU_C3_RECEIPTS_MANIFEST_ALLOWED_VERSIONS:
            proj = _time_tau__lb_projection_receipts_manifest(payload)
            return str(hash_json_sig8(proj) or "").strip() or None
    except Exception:
        return None

    # Time(τ) derived worlds manifest (LB projection binds sig8).
    try:
        if schema == TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_SCHEMA and schema_version in TIME_TAU_C3_DERIVED_WORLDS_MANIFEST_ALLOWED_VERSIONS:
            proj = _time_tau__lb_projection_derived_worlds_manifest(payload)
            return str(hash_json_sig8(proj) or "").strip() or None
    except Exception:
        return None

    return None


def b6_sig8_audit_bundle_dir(bundle_dir: _Path) -> dict:
    """Compute a read-only sig8 audit block for a bundle directory.

    This never raises; failures are reported as entry-level `ok=None` with details.
    """
    bdir = _Path(bundle_dir).resolve()

    # Candidate JSON artifacts (fixed paths + small glob neighborhoods).
    candidates: list[_Path] = []
    try:
        candidates.extend([
            bdir / "meta" / "bundle_manifest.json",
            bdir / "manifests" / "time_tau_pointer_set.json",
            bdir / "manifests" / "time_tau_c3_receipts_manifest.json",
            bdir / "manifests" / "time_tau_c3_derived_worlds_manifest.json",
        ])
    except Exception:
        candidates = []

    # D4 certificate neighborhood (usually 0/1 file, but keep deterministic).
    try:
        d4_dir = bdir / "certs" / "d4"
        if d4_dir.exists():
            candidates.extend(sorted(list(d4_dir.glob("*.json")), key=lambda p: p.name))
    except Exception:
        pass

    # Parity neighborhood is optional; we do not attempt to compute parity sig8 here
    # to avoid reaching across ordering boundaries in this monolith module.
    # (Parity sig8 is already self-contained and verified elsewhere.)

    entries: list[dict] = []
    for p in candidates:
        try:
            if not _Path(p).exists():
                continue
            rel = _Path(p).relative_to(bdir).as_posix()
        except Exception:
            rel = str(p)

        row: dict = {"relpath": rel}

        try:
            raw = _Path(p).read_text(encoding="utf-8")
            obj = _json.loads(raw)
        except Exception as exc:
            row["ok"] = None
            row["detail"] = f"parse_error: {exc}"
            entries.append(row)
            continue

        if not isinstance(obj, dict):
            row["ok"] = None
            row["detail"] = "not an object"
            entries.append(row)
            continue

        schema = str(obj.get("schema") or "").strip()
        schema_version = str(obj.get("schema_version") or "").strip()

        # Pretty family label for schema-less bundle_manifest.json
        if (not schema) and (rel == "meta/bundle_manifest.json"):
            schema = "bundle_manifest"

        row["schema"] = schema
        row["schema_version"] = schema_version

        claimed = str(obj.get("sig8") or "").strip()
        expected = sig8_expected_for_payload(obj)

        if claimed:
            row["claimed_sig8"] = claimed
        if expected:
            row["expected_sig8"] = expected

        if claimed and expected:
            ok = (claimed == expected)
            row["ok"] = ok
            if not ok:
                row["detail"] = "sig8_mismatch"
        else:
            row["ok"] = None
            if not claimed:
                row["detail"] = "sig8_missing"
            elif not expected:
                row["detail"] = "sig8_unverifiable"

        entries.append(row)

    entries.sort(key=lambda r: str(r.get("relpath") or ""))

    return {
        "schema": SIG8_AUDIT_SCHEMA,
        "schema_version": SIG8_AUDIT_SCHEMA_VERSION,
        "entries": entries,
        "summary": {
            "n_entries": len(entries),
            "n_ok": sum(1 for r in entries if r.get("ok") is True),
            "n_bad": sum(1 for r in entries if r.get("ok") is False),
            "n_unverifiable": sum(1 for r in entries if r.get("ok") is None),
        },
    }

def b6_verify_bundle_seal(bundle_dir: _Path, *, seal_relpath: str | None = None) -> dict:
    """Verify the on-disk B6 seal sidecar against the bundle tree (Profile.v1 only).

    Returns a `b6_verify_receipt` dict (BOOLEAN verdict, B3 status/reasons).
    This helper is *not* UI-wired yet; it exists so D4 can later be a thin wrapper.
    """
    bdir = _Path(bundle_dir).resolve()
    seal_rel = str(seal_relpath or B6_SEAL_REL_PATH)
    seal_abs = bdir / seal_rel

    receipt: dict = {
        "schema": B6_VERIFY_RECEIPT_SCHEMA,
        "schema_version": B6_VERIFY_RECEIPT_SCHEMA_VERSION,
        "policy": "b6.verify",
        "verdict_mode": "BOOLEAN",
        "status": "NOT_ATTEMPTED",
        "reason_code": "PRECOND_MISSING",
        "reason_detail_code": "",
        "verdict": None,
        "seal_path": seal_rel,
        "violations": [],
    }

    # Read-only sig8 audit (does not affect verdict/status)
    receipt["sig8_audit"] = b6_sig8_audit_bundle_dir(bdir)

    if not seal_abs.exists():
        receipt["reason_detail_code"] = "SEAL_MISSING"
        return receipt

    try:
        seal_obj = _json.loads(seal_abs.read_text(encoding="utf-8"))
    except Exception as exc:
        receipt["reason_code"] = "PRECOND_INVALID"
        receipt["reason_detail_code"] = "SEAL_PARSE_ERROR"
        receipt["verdict"] = None
        receipt["violations"] = []
        return receipt

    if not isinstance(seal_obj, dict):
        receipt["reason_code"] = "PRECOND_INVALID"
        receipt["reason_detail_code"] = "SEAL_NOT_OBJECT"
        return receipt

    # Required fields for Profile.v1 verification.
    required_fields = [
        "seal_profile_id",
        "inventory_encoding_id",
        "portability_key_id",
        "exclude_policy_id",
        "seal_hash_algo",
        "file_hash_algo",
        "digest_hex",
        "digest_sig8",
        "n_files",
    ]
    missing = [k for k in required_fields if k not in seal_obj]
    if missing:
        receipt["reason_code"] = "PRECOND_INVALID"
        receipt["reason_detail_code"] = "SEAL_MISSING_FIELDS:" + ",".join(missing[:5])
        return receipt

    seal_profile_id = str(seal_obj.get("seal_profile_id") or "")
    receipt["seal_profile_id"] = seal_profile_id
    if seal_profile_id != B6_SEAL_PROFILE_ID:
        receipt["reason_code"] = "PRECOND_INVALID"
        receipt["reason_detail_code"] = "UNSUPPORTED_SEAL_PROFILE"
        return receipt

    expected_digest_hex = str(seal_obj.get("digest_hex") or "")
    receipt["expected_digest_hex"] = expected_digest_hex

    # From this point on, we *attempted* verification.
    receipt.pop("reason_code", None)
    receipt.pop("reason_detail_code", None)
    receipt["status"] = "OK"
    receipt["verdict"] = False  # may flip to True below

    fail_codes: set[str] = set()
    violations: list[dict] = []

    # Header/profile self-consistency check.
    header_expect = {
        "inventory_encoding_id": B6_INVENTORY_ENCODING_ID,
        "portability_key_id": B6_PORTABILITY_KEY_ID,
        "exclude_policy_id": B6_EXCLUDE_POLICY_ID,
        "seal_hash_algo": B6_SEAL_HASH_ALGO,
        "file_hash_algo": B6_FILE_HASH_ALGO,
    }
    header_mismatch = False
    for field, exp in header_expect.items():
        obs = str(seal_obj.get(field) or "")
        if obs != str(exp):
            header_mismatch = True
            violations.append(
                {
                    "code": "SEAL_PROFILE_HEADER_MISMATCH",
                    "field": field,
                    "expected": str(exp),
                    "observed": obs,
                    "detail_code": "HEADER_FIELD",
                    "detail": "seal header inconsistent with Profile.v1",
                }
            )
    if header_mismatch:
        fail_codes.add("SEAL_PROFILE_HEADER_MISMATCH")

    # Recompute seal from the tree.
    observed_payload: dict | None = None
    try:
        observed_payload = b6_compute_seal_profile_v1(bdir)
    except _B6SealViolation as ve:
        # Profile violation: verification ran, verdict is false, with fail codes.
        for c in ve.fail_codes:
            if c in B6_FAIL_CODES_V0_1_SET:
                fail_codes.add(c)
            else:
                # Keep only frozen codes in v0.1 receipts.
                pass
        violations.extend(list(ve.violations or []))
        # Populate violations list in receipt and finalize.
        receipt["violations"] = _b6_sort_violations(violations)
        if fail_codes:
            receipt["fail_codes"] = sorted(fail_codes)
        receipt["verdict"] = False
        return receipt
    except Exception as exc:
        # IO or unexpected error while recomputing.
        receipt["status"] = "ERROR"
        receipt["verdict"] = None
        receipt["reason_code"] = "IO_ERROR"
        receipt["reason_detail_code"] = "RECOMPUTE_FAILED"
        receipt["violations"] = []
        return receipt

    observed_digest_hex = str(observed_payload.get("digest_hex") or "")
    receipt["observed_digest_hex"] = observed_digest_hex
    receipt["observed_n_files"] = int(observed_payload.get("n_files") or 0)

    if observed_digest_hex != expected_digest_hex:
        fail_codes.add("SEAL_DIGEST_MISMATCH")

    # Witness checks (diagnostic; only if the witness fields exist).
    # exclude_relpaths witness
    if "exclude_relpaths" in seal_obj:
        try:
            witness_excl = sorted([_b6_validate_relpath(x) for x in (seal_obj.get("exclude_relpaths") or [])])
        except Exception:
            witness_excl = []
        expected_excl = sorted([_b6_validate_relpath(x) for x in B6_EXCLUDE_REL_PATHS_V1])
        if witness_excl != expected_excl:
            fail_codes.add("WITNESS_EXCLUDELIST_MISMATCH")
            violations.append(
                {
                    "code": "WITNESS_EXCLUDELIST_MISMATCH",
                    "field": "exclude_relpaths",
                    "expected": expected_excl,
                    "observed": witness_excl,
                    "detail_code": "WITNESS",
                    "detail": "exclude_relpaths witness does not match policy",
                }
            )

    # n_files witness
    if "n_files" in seal_obj:
        try:
            w_n = int(seal_obj.get("n_files") or 0)
        except Exception:
            w_n = -1
        o_n = int(observed_payload.get("n_files") or 0)
        if w_n != o_n:
            fail_codes.add("WITNESS_NFILES_MISMATCH")
            violations.append(
                {
                    "code": "WITNESS_NFILES_MISMATCH",
                    "field": "n_files",
                    "expected": w_n,
                    "observed": o_n,
                    "detail_code": "WITNESS",
                    "detail": "n_files witness does not match recomputation",
                }
            )

    # entries witness
    if "entries" in seal_obj and isinstance(seal_obj.get("entries"), list):
        witness_entries = _b6_canon_entries_list(seal_obj.get("entries"))
        observed_entries = _b6_canon_entries_list(observed_payload.get("entries"))
        if witness_entries != observed_entries:
            fail_codes.add("WITNESS_ENTRIES_MISMATCH")
            violations.append(
                {
                    "code": "WITNESS_ENTRIES_MISMATCH",
                    "field": "entries",
                    "detail_code": "WITNESS",
                    "detail": "entries witness does not match recomputation",
                }
            )

    # Finalize verdict + fields.
    if fail_codes:
        receipt["verdict"] = False
        receipt["fail_codes"] = sorted(fail_codes)
        receipt["violations"] = _b6_sort_violations(violations)
    else:
        receipt["verdict"] = True
        receipt["violations"] = []

    return receipt


def b6_write_verify_receipt(bundle_dir: _Path, receipt: dict) -> _Path:
    """Write the B6 verify receipt into meta/b6_verify_receipt.json (canonical path)."""
    bdir = _Path(bundle_dir)
    meta_dir = _b1_rel_dirs_for_root(bdir)["meta"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    outp = meta_dir / "b6_verify_receipt.json"
    _b6_atomic_write_text(outp, _b5_cjf1_dumps(receipt or {}))
    return outp


def _b6_canon_entries_list(entries_obj) -> list[tuple[str, str, int]]:
    """Canonicalize an entries list into a stable comparable structure.

    Accepts both canonical keys (sha256_hex, byte_len) and legacy aliases
    (sha256, bytes) for read-compat.
    """
    out: list[tuple[str, str, int]] = []
    if not isinstance(entries_obj, list):
        return out
    for e in entries_obj:
        if not isinstance(e, dict):
            continue
        rel = str(e.get("relpath") or "")
        sha = str(e.get("sha256_hex") or e.get("sha256") or "")
        try:
            bl = int(e.get("byte_len") if "byte_len" in e else (e.get("bytes") or 0))
        except Exception:
            bl = 0
        out.append((rel, sha, bl))
    out.sort()
    return out


def _b6_sort_violations(violations: list[dict]) -> list[dict]:
    """Sort violations deterministically (code, relpath, other_relpath, field)."""
    def _k(v: dict):
        code = str(v.get("code") or "")
        rel = str(v.get("relpath") or "")
        orel = str(v.get("other_relpath") or "")
        field = str(v.get("field") or "")
        return (code, rel, orel, field)
    vv = list(violations or [])
    vv.sort(key=_k)
    # Dedupe exact duplicates (best-effort).
    out: list[dict] = []
    seen: set[str] = set()
    for v in vv:
        try:
            key = _json.dumps(v, sort_keys=True, ensure_ascii=False)
        except Exception:
            key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


# --- D4 helpers: bundle manifest digest (D4.C, optional) ---


def _d4_bundle_manifest_digest(snapshot_id: str | None = None) -> dict | None:
    """Return a compact digest for bundle_manifest.json, if present.

    We scan the bundle root for a directory matching the snapshot_id and
    pick the bundle_manifest.json with the newest mtime. If nothing is
    found, return None.
    """
    sid = snapshot_id or _v2_current_world_snapshot_id(strict=True)

    try:
        bundle_root = _bundle_root_dir()  # type: ignore[name-defined]
    except Exception:
        # Fall back to logs/bundle under the repo root.
        try:
            root = _REPO_ROOT
        except Exception:
            root = _Path(__file__).resolve().parents[1]
        bundle_root = root / "logs" / "bundle"

    try:
        # Look for any bundle dir whose name starts with "{sid}__".
        pattern = f"{sid}__*"
        candidates = list(_Path(bundle_root).glob(pattern))
    except Exception:
        candidates = []

    manifest_path: _Path | None = None
    for candidate in candidates:
        cand_manifest = _Path(candidate) / "bundle_manifest.json"
        if cand_manifest.exists():
            if manifest_path is None:
                manifest_path = cand_manifest
            else:
                # Pick the most recently modified manifest file.
                try:
                    if cand_manifest.stat().st_mtime > manifest_path.stat().st_mtime:
                        manifest_path = cand_manifest
                except Exception:
                    # If stat fails, keep the existing manifest_path.
                    pass

    if manifest_path is None:
        return None

    try:
        data = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    sig8 = data.get("sig8")
    if not sig8:
        try:
            sig8 = hash_json_sig8(data)
        except Exception:
            sig8 = ""

    try:
        rel = _bundle_repo_relative_path(manifest_path)
    except Exception:
        try:
            rel = str(manifest_path.relative_to(_REPO_ROOT))
        except Exception:
            rel = str(manifest_path)

    return {
        "bundle_manifest_path": rel,
        "bundle_manifest_sig8": sig8,
    }


def _bundle_repo_relative_path(p: _Path | str) -> str:
    """Return a repo-root-relative POSIX path string for bundle-manifest pointers.

    Policy (bundle-manifest identity discipline):

      - Host-dependent absolute paths MUST NOT enter bundle-manifest identity.
      - Relative inputs are treated as already repo-root relative (but may not
        contain ``..``).
      - Absolute inputs must live under the configured repo root; otherwise we
        raise (leaving this ambiguous would allow non-isomorphic forks).

    """
    # Prefer the configured repo root when available.
    try:
        root = _REPO_DIR  # type: ignore[name-defined]
    except Exception:
        try:
            root = _REPO_ROOT  # type: ignore[name-defined]
        except Exception:  # pragma: no cover - extremely defensive
            root = _Path(__file__).resolve().parents[1]
    root = _Path(root).resolve()

    pp = _Path(p)

    # Relative inputs are treated as already repo-root relative.
    if not pp.is_absolute():
        if ".." in pp.parts:
            raise ValueError(f"_bundle_repo_relative_path: illegal '..' in relative path: {pp}")
        return pp.as_posix()

    # Absolute inputs must be under root.
    try:
        rel = pp.resolve().relative_to(root)
    except Exception as exc:
        raise ValueError(f"_bundle_repo_relative_path: path is not under repo root ({root}): {pp}") from exc

    return rel.as_posix()


def build_bundle_manifest_for_snapshot(
    snapshot_id: str | None = None,
    run_ctx: dict | None = None,
) -> dict:
    """Builder for bundle_manifest.json for a given 64× v2 run.

    This function wires together
    canonical paths (relative to the repo root) for the existing v2 core,
    Time(τ) lab, coverage, and snapshot artifacts into a dict matching the
    bundle manifest schema.

    The intended call pattern is:

        manifest = build_bundle_manifest_for_snapshot(snapshot_id, run_ctx)

    where `run_ctx` comes from the suite/τ orchestrator and may contain
    hints like `manifest_full_scope_path` or an explicit `engine_rev`.
    """
    import time as _time  # local import to avoid global time dependency

    rc = dict(run_ctx or {})

    # Resolve SSOT snapshot id once; treat any explicit snapshot_id as a guard.
    ssot_snapshot_id = _v2_current_world_snapshot_id(strict=True)
    if snapshot_id and str(snapshot_id) != str(ssot_snapshot_id):
        raise ValueError(
            "build_bundle_manifest_for_snapshot: snapshot_id mismatch "
            "(arg vs SSOT)."
        )

    # If run_ctx carries a snapshot_id, treat mismatches as an error as well.
    rc_sid = rc.get("snapshot_id")
    if rc_sid and str(rc_sid) != str(ssot_snapshot_id):
        raise ValueError(
            "build_bundle_manifest_for_snapshot: run_ctx snapshot_id mismatch "
            "(run_ctx vs SSOT)."
        )

    sid_str = str(ssot_snapshot_id)

    # Engine revision: prefer run_ctx override, fall back to global ENGINE_REV.
    try:
        engine_rev = rc.get("engine_rev") or ENGINE_REV
    except Exception:
        engine_rev = rc.get("engine_rev") or "unknown"

    # Repo & canonical dirs.
    try:
        root = _REPO_ROOT
    except Exception:
        root = _Path(__file__).resolve().parents[1]

    manifests_dir = _MANIFESTS_DIR if "_MANIFESTS_DIR" in globals() else (root / "logs" / "manifests")
    reports_dir = _REPORTS_DIR if "_REPORTS_DIR" in globals() else (root / "logs" / "reports")
    certs_root = _CERTS_DIR if "_CERTS_DIR" in globals() else (root / "logs" / "certs")

    # Experiments / snapshots directories: respect DIRS when present.
    exps_dir = None
    snaps_dir = None
    if "DIRS" in globals():
        try:
            exps_dir = _Path(DIRS.get("experiments", "logs/experiments"))
        except Exception:
            exps_dir = None
        try:
            snaps_dir = _Path(DIRS.get("snapshots", "logs/snapshots"))
        except Exception:
            snaps_dir = None
    if exps_dir is None:
        exps_dir = root / "logs" / "experiments"
    if snaps_dir is None:
        snaps_dir = root / "logs" / "snapshots"

    # v2 manifest path: allow run_ctx override, else fall back to helper.
    mp = rc.get("manifest_full_scope_path")
    if mp:
        v2_manifest_path = _Path(mp)
    else:
        try:
            v2_manifest_path = _svr_current_run_manifest_path()
        except Exception:
            v2_manifest_path = manifests_dir / "manifest_full_scope.jsonl"

    # Time(τ) C3 manifest lives alongside the v2 manifest.
    c3_manifest_path = manifests_dir / "time_tau_c3_manifest_full_scope.jsonl"

    # Time(τ) C3 receipts manifest: explicit inventory of recompute receipts.
    # This artifact MAY be validly empty; do not infer emptiness from directory listing.
    _rm_override = None
    try:
        tt = rc.get("time_tau") if isinstance(rc.get("time_tau"), dict) else None
        if isinstance(tt, dict):
            # Support both a direct path and a nested summary object.
            _rm_override = tt.get("c3_receipts_manifest_path") or (tt.get("c3_receipts_manifest") or {}).get("receipts_manifest_path")
    except Exception:
        _rm_override = None
    if _rm_override:
        c3_receipts_manifest_path = _Path(_rm_override)
    else:
        c3_receipts_manifest_path = manifests_dir / f"time_tau_c3_receipts_manifest__{sid_str}.json"

    # Time(τ) directories & rollup. C2/C3 artifacts are under logs/experiments,
    # C4 rollup under logs/reports.
    try:
        rollup_jsonl_name = TIME_TAU_C3_ROLLUP_JSONL  # type: ignore[name-defined]
    except Exception:
        rollup_jsonl_name = "time_tau_c3_rollup.jsonl"
    c2_toy_dir = exps_dir  # time_tau_local_flip* live here
    c3_receipts_dir = exps_dir  # time_tau_c3_recompute__*.json live here
    c4_rollup_path = reports_dir / rollup_jsonl_name

    # Coverage paths (C1).
    coverage_jsonl_path = reports_dir / "coverage.jsonl"
    coverage_rollup_csv_path = reports_dir / "coverage_rollup.csv"

    manifests_section = {
        "v2_full_scope": _bundle_repo_relative_path(v2_manifest_path),
        "time_tau_c3": _bundle_repo_relative_path(c3_manifest_path),
        "time_tau_c3_receipts_manifest": _bundle_repo_relative_path(c3_receipts_manifest_path),
    }

    d4_dir = certs_root / "d4"

    certs_section = {
        "strict": _bundle_repo_relative_path(certs_root),
        # Projected/auto certs are optional; we may fill this in later.
        "projected": None,
        "d4_cert_dir": _bundle_repo_relative_path(d4_dir),
    }

    time_tau_section = {
        "c2_toy_dir": _bundle_repo_relative_path(c2_toy_dir),
        "c3_receipts_dir": _bundle_repo_relative_path(c3_receipts_dir),
        "c3_receipts_manifest_path": _bundle_repo_relative_path(c3_receipts_manifest_path),
        "c4_rollup_path": _bundle_repo_relative_path(c4_rollup_path),
    }

    coverage_section = {
        "coverage_jsonl_path": _bundle_repo_relative_path(coverage_jsonl_path),
        "coverage_rollup_csv_path": _bundle_repo_relative_path(coverage_rollup_csv_path),
    }

    logs_section = {
        "loop_receipts_dir": _bundle_repo_relative_path(exps_dir),
        "world_snapshots_dir": _bundle_repo_relative_path(snaps_dir),
    }

    meta_section = {
        "notes": "",
        "extra": {},
        # B4 κ registry lives under meta so it can be injected post-D4 without
        # affecting manifest sig8 (sig8 excludes meta by policy).
        "kappa_registry": {
            "schema_version": B4_KAPPA_REGISTRY_SCHEMA_VERSION,
            "contexts": [],
        },
    }

    # B4/B5 placeholders: bundle-internal paths.
    b4_section = {
        "admissibility_report_path": "meta/b4_admissibility.json",
    }
    b5_section = {
        # B5 semantic object-set index (policy-agnostic identity anchor).
        "b5_index_path": "meta/b5_index.v1.json",
        # Bundle-level seal/hash over the full tree (packaging integrity).
        # NOTE: This is conceptually B6, but we keep it under b5 for now to
        # avoid breaking older readers.
        "bundle_hash_path": "meta/bundle_hash.json",
    }

    b6_section = {
        # B6 packaging seal (Profile.v1).
        "seal_path": B6_SEAL_REL_PATH,
        # Reserved path for verifier receipts (not UI-wired yet).
        "verify_receipt_path": B6_VERIFY_RECEIPT_REL_PATH,
    }

    # Pointer-gated resolution surface (Phase 7): explicit (kind, sig8) → relpath.
    # These pointers are binding because bundle-manifest sig8 includes them.
    pointers_section: dict = {}
    try:
        v2_rel = manifests_section.get("v2_full_scope")
        if v2_rel:
            ptr = {
                "artifact_kind": "manifest_v2_full_scope",
                "sig8": "",
                "relpath": str(v2_rel),
            }
            try:
                v2_abs = _bundle_repo_abspath(v2_rel)
                if _Path(v2_abs).exists():
                    ptr["sig8"] = _strict_hash_file_sig8(v2_abs)
            except Exception:
                # Leave sig8 empty; strict gates will fail-closed.
                ptr["sig8"] = ""
            pointers_section["v2_full_scope_manifest"] = ptr
    except Exception:
        pointers_section = {}


    manifest = {
        "snapshot_id": sid_str,
        "created_at_utc": int(_time.time()),
        "engine_rev": str(engine_rev),
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "sig8": "",
        "pointers": pointers_section,
        "manifests": manifests_section,
        "certs": certs_section,
        "time_tau": time_tau_section,
        "coverage": coverage_section,
        "logs": logs_section,
        "b4": b4_section,
        "b5": b5_section,
        "b6": b6_section,
        "meta": meta_section,
    }
    return manifest


def stamp_bundle_manifest_sig8(manifest: dict) -> dict:
    """Pure helper: compute and stamp sig8 for a bundle manifest.

    B4 rule: sig8 is computed over the manifest's *canonical identity surface*,
    excluding UI/runtime noise and any annexes that must not affect bundle
    identity (e.g. meta notes, b4/b5 sections).

    This keeps `sig8` stable across runs while binding the manifest to its
    core canonical pointers. Post-D4 injections (e.g., κ registry) live under
    `meta` and are excluded from `sig8` by policy.
    """
    base = dict(manifest or {})

    # Ensure a schema_version is present for stability.
    base.setdefault("schema_version", BUNDLE_MANIFEST_SCHEMA_VERSION)

    # Exclude sig8 itself from the hash.
    tmp = dict(base)
    tmp["sig8"] = ""

    # Explicit non-core exclusions for bundle-manifest identity (B4 discipline).
    non_core = set(_B3_NON_CORE_COMMON_KEYS) | {
        "sig8",
        "engine_rev",
        "meta",
        "logs",
        "b4",
        "b5",
        "b6",
    }

    sig8 = b3_payload_sig8(tmp, non_core_keys=non_core)
    base["sig8"] = sig8
    return base



def _bundle_root_dir() -> _Path:
    """Return the base directory for exported v2 bundles, creating it if needed.

    We prefer the global `_BUNDLE_ROOT` when available, and otherwise fall back
    to `<repo_root>/logs/bundle`. This helper is intentionally defensive so it
    can be used in both app and non-app contexts.
    """
    try:
        base = _BUNDLE_ROOT  # type: ignore[name-defined]
    except Exception:  # pragma: no cover - extremely defensive
        try:
            base = _REPO_ROOT / "logs" / "bundle"  # type: ignore[name-defined]
        except Exception:
            base = _Path("logs") / "bundle"
    base = _Path(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _ensure_bundle_dir_for_manifest(manifest: dict) -> _Path:
    """Resolve and create the bundle directory for a given manifest.

    Directory naming policy:

      - If `snapshot_id` is present and non-empty:
          logs/bundle/{snapshot_id}__{sig8}
        (or logs/bundle/{snapshot_id} if sig8 is empty)

      - Otherwise:
          logs/bundle/{sig8}  (or logs/bundle/bundle as a last resort).

    This keeps all artifacts for a single 64× run grouped together.
    """
    base = _bundle_root_dir()
    sid = str(manifest.get("snapshot_id") or "")
    sig8 = str(manifest.get("sig8") or "")

    if sid:
        dirname = f"{sid}__{sig8}" if sig8 else sid
    else:
        dirname = sig8 or "bundle"

    bdir = base / dirname
    bdir.mkdir(parents=True, exist_ok=True)
    return bdir





def run_full_v2_suite_and_tau(snapshot_id: str | None = None) -> tuple[str, dict]:
    """Orchestrate the v2 core + Time(τ) lab for a given snapshot.

    This helper is intentionally conservative:

      - It assumes the 64× v2 suite has already been run at least once and that
        loop_receipt.v2 files exist under logs/certs/.
      - If the v2 manifest is missing, it will attempt to regenerate it from
        loop_receipts via `_v2_regen_manifest_from_receipts()`.
      - It then runs:

          C1 coverage rollup for the snapshot,
          C2 τ-sweep over manifest_full_scope.jsonl,
          C3 manifest build from the C2 sweep,
          C3 τ-recompute sweep,
          C4 τ-rollup (CSV/JSONL + coverage τ ping).

    On success it returns (snapshot_id, run_ctx) where run_ctx contains at
    least:

        {
            "snapshot_id": ...,
            "engine_rev": ENGINE_REV,
            "manifest_full_scope_path": "logs/manifests/manifest_full_scope.jsonl",
        }

    If a required step fails (missing manifest/receipts, C2/C3/C4 failure),
    this function raises a RuntimeError with a short explanation. The caller
    (e.g. Streamlit UI or an export helper) is expected to surface that error
    to the user.
    """
    import time as _time  # local import to avoid global time dependency

    # 1) Resolve snapshot_id: prefer explicit arg, then run_ctx helper,
    #    then the world_snapshot pointer.
    sid = snapshot_id
    if not sid:
        try:
            sid = _svr_current_run_snapshot_id()
        except Exception:
            sid = None
    if not sid:
        try:
            sid = _svr_current_snapshot_id()
        except Exception:
            sid = None

    if not sid:
        raise RuntimeError(
            "run_full_v2_suite_and_tau: unable to determine snapshot_id; "
            "run the v2 core 64× flow at least once first."
        )
    sid_str = str(sid)

    # 2) Resolve v2 manifest path: prefer current run manifest; if missing,
    #    attempt regen from loop_receipts.
    try:
        manifest_path = _svr_current_run_manifest_path()
    except Exception:
        manifest_path = None

    try:
        root = _repo_root()
    except Exception:
        root = _Path(__file__).resolve().parents[1]
    manifests_dir = _MANIFESTS_DIR if "_MANIFESTS_DIR" in globals() else (root / "logs" / "manifests")

    if not manifest_path:
        manifest_path = manifests_dir / "manifest_full_scope.jsonl"
    manifest_path = _Path(manifest_path)

    if not manifest_path.exists():
        # Best-effort regen from loop_receipts.
        try:
            ok_m, path_m, n_m = _v2_regen_manifest_from_receipts()
        except Exception as e:
            raise RuntimeError(
                f"run_full_v2_suite_and_tau: v2 manifest missing and regen failed: {e}"
            ) from e
        if not ok_m:
            raise RuntimeError(
                f"run_full_v2_suite_and_tau: v2 manifest missing and regen failed: {path_m}"
            )
        manifest_path = _Path(path_m)

    # 3) Initial C1 coverage rollup for this snapshot (non-fatal if it fails).
    try:
        _coverage_rollup_write_csv(snapshot_id=sid_str)
    except Exception:
        pass

    # 4) C2 τ-sweep over the v2 manifest.
    ok_c2, msg_c2, summary_c2 = _time_tau_c2_run_sweep(
        manifest_path=str(manifest_path),
    )
    if not ok_c2:
        raise RuntimeError(f"Time(τ) C2 sweep failed: {msg_c2}")

    # 5) Build C3 manifest from the C2 sweep.
    c2_jsonl_path = summary_c2.get("jsonl_path")
    ok_c3m, msg_c3m, summary_c3m = time_tau_c3_build_manifest_from_c2_sweep(
        manifest_v2_path=str(manifest_path),
        c2_sweep_jsonl_path=str(c2_jsonl_path) if c2_jsonl_path else None,
    )
    if not ok_c3m:
        raise RuntimeError(f"Time(τ) C3 manifest build failed: {msg_c3m}")

    c3_manifest_path = summary_c3m.get("manifest_c3_path")
    # 6) Run C3 τ-recompute sweep.
    ok_c3s, msg_c3s, summary_c3s = time_tau_c3_run_sweep(
        manifest_path=str(c3_manifest_path) if c3_manifest_path else None,
    )
    if not ok_c3s:
        raise RuntimeError(f"Time(τ) C3 sweep failed: {msg_c3s}")

    # 6.5) Write canonical C3 receipts manifest (explicit inventory; may be validly empty).
    ok_rm, msg_rm, summary_rm = time_tau_c3_write_receipts_manifest(
        snapshot_id=sid_str,
        c3_manifest_path=str(c3_manifest_path) if c3_manifest_path else None,
    )
    if not ok_rm:
        raise RuntimeError(f"Time(τ) C3 receipts manifest write failed: {msg_rm}")

    # 7) C4 τ-rollup (CSV/JSONL + τ coverage ping).
    try:
        rows_c4, summary_c4 = time_tau_c4_build_rollup()
    except Exception as e:
        raise RuntimeError(f"Time(τ) C4 rollup failed: {e}") from e

    # 8) Final C1 coverage rollup with τ ping wired in (non-fatal if it fails).
    try:
        _coverage_rollup_write_csv(snapshot_id=sid_str)
    except Exception:
        pass

    # 9) Build a run_ctx summary and attempt to sync st.session_state['run_ctx'].
    rc: dict = {
        "snapshot_id": sid_str,
        "engine_rev": ENGINE_REV,
        "manifest_full_scope_path": str(manifest_path),
        "time_tau": {
            "c2": summary_c2,
            "c3_manifest": summary_c3m,
            "c3_sweep": summary_c3s,
            "c3_receipts_manifest": summary_rm,
            "c3_receipts_manifest_path": (summary_rm or {}).get("receipts_manifest_path"),
            "c4_rollup": summary_c4,
        },
        "updated_at_utc": int(_time.time()),
    }

    # --- Time(τ) τ-state object (v0.1) ---
    tau_state: dict
    try:
        tau_state = time_tau_build_state_v0_1(
            snapshot_id=sid_str,
            manifest_full_scope_path=str(manifest_path),
            summary_c2=summary_c2,
            summary_c3m=summary_c3m,
            summary_c3s=summary_c3s,
            summary_rm=summary_rm,
            summary_c4=summary_c4,
        )
    except Exception as e:
        # Do not fail the pipeline if τ-state construction fails; record the error.
        tau_state = {
            "schema": TIME_TAU_STATE_SCHEMA,
            "schema_version": TIME_TAU_STATE_SCHEMA_VERSION,
            "engine_rev": ENGINE_REV,
            "snapshot_id": sid_str,
            "build_error": str(e),
            "built_at_utc": int(_time.time()),
        }

    # Best-effort: persist it (snapshot-scoped).
    tau_state_path: str = ""
    try:
        ok_ts, msg_ts, out_ts = time_tau_write_state_v0_1(
            tau_state=tau_state,
            snapshot_id=sid_str,
        )
        if ok_ts:
            tau_state_path = str((out_ts or {}).get("tau_state_path") or "")
        else:
            tau_state["write_error"] = msg_ts
    except Exception as e:
        tau_state["write_error"] = str(e)

    rc["time_tau_state"] = tau_state
    rc["time_tau_state_path"] = tau_state_path

    try:
        _svr_run_ctx_update(
            snapshot_id=sid_str,
            manifest_full_scope_path=str(manifest_path),
            time_tau_state=tau_state,
            time_tau_state_path=tau_state_path,
        )
    except Exception:
        # This is best-effort; it's fine if we're not in a Streamlit session.
        pass

    return sid_str, rc






def _v2_run_core_flow_64x_strict() -> str:
    """Run the strict V2 core 64× flow and return the SSOT snapshot_id.

    This folds the legacy UI-only "Run V2 core (64× …)" entrypoint into the
    single-path pipeline, so the one-click run can be the only entrypoint.

    Contract (strict):
      - Writes/refreshes the v2 world snapshot (SSOT) from app/inputs.
      - Runs the 64× suite in two passes:
          * bootstrap pass (run_label=v2_bootstrap_64, snapshot_id suffix "__boot")
          * real pass      (run_label=v2_suite_64, snapshot_id = SSOT sid)
      - Regenerates logs/manifests/manifest_full_scope.jsonl from loop_receipts.
      - Best-effort writes coverage_rollup.csv; the gate vector enforces presence.

    Returns:
        sid (e.g. "ws__b5f456f9")
    """
    import json as _json

    repo_root = _repo_root()
    inputs_root = _Path(repo_root) / "app" / "inputs"

    manifests_dir = _MANIFESTS_DIR if "_MANIFESTS_DIR" in globals() else (_Path(repo_root) / "logs" / "manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    B_dir, H_dir, C_dir = inputs_root / "B", inputs_root / "H", inputs_root / "C"
    U_path = (inputs_root / "U.json").resolve()

    # Preflight: require the canonical 64× fixture inputs.
    if not (B_dir.exists() and H_dir.exists() and C_dir.exists() and U_path.exists()):
        raise RuntimeError(
            "V2 core preflight failed: missing inputs under app/inputs. "
            f"B:{B_dir.exists()} H:{H_dir.exists()} C:{C_dir.exists()} U:{U_path.exists()} "
            f"({inputs_root})"
        )

    # Discover D; hard-code H (4) and C (8) for 64×.
    D_tags = sorted(p.stem for p in B_dir.glob("D*.json"))
    H_tags = ["H00", "H01", "H10", "H11"]
    C_tags = [f"C{n:03b}" for n in range(8)]  # C000..C111

    rows: list[dict] = []
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
                rows.append(
                    {
                        "fixture_label": fid,
                        "paths": {
                            "B": str(B_path),
                            "C": str(Cp),
                            "H": str(Hp),
                            "U": str(U_path),
                        },
                    }
                )

    if not rows:
        raise RuntimeError(
            "V2 core preflight failed: no fixtures produced. "
            "Expected app/inputs/B/D*.json, app/inputs/H/H00,H01,H10,H11 and app/inputs/C/C000..C111."
        )

    # 1) Bootstrap manifest with absolute paths.
    man_boot = manifests_dir / "manifest_bootstrap__ALL.jsonl"
    man_boot.write_text(
        "\n".join(_json.dumps(r, separators=(",", ":")) for r in rows) + "\n",
        encoding="utf-8",
    )

    # 2) Canonical world snapshot (v2) for SSOT snapshot_id.
    legacy_body = {
        "suite_kind": "v2_overlap_64",
        "bootstrap_manifest": str(man_boot),
        "n_fixtures": len(rows),
        "fixtures": rows,
    }
    world_snapshot = build_v2_world_snapshot_from_body(legacy_body)
    world_snapshot_path = _svr_write_world_snapshot(world_snapshot)

    sid = str(world_snapshot.get("snapshot_id") or "").strip()
    if not sid:
        raise RuntimeError("V2 core: world snapshot write produced empty snapshot_id")

    # Best-effort: sync run_ctx for downstream panels.
    try:
        _svr_run_ctx_update(
            snapshot_id=sid,
            world_snapshot_path=str(world_snapshot_path),
            manifest_bootstrap_path=str(man_boot),
        )
    except Exception:
        pass

    # Strict resolve (also guards that the pointer exists).
    run_snapshot_id = _v2_current_world_snapshot_id(strict=True)
    run_snapshot_id = str(run_snapshot_id)

    # 3) Bootstrap suite run (coverage rows are tagged so they can be excluded).
    snap_boot = f"{run_snapshot_id}__boot"
    try:
        _svr_run_ctx_update(run_label="v2_bootstrap_64")
    except Exception:
        pass

    ok1, msg1, _cnt1 = run_suite_from_manifest(str(man_boot), snap_boot)
    if not ok1:
        raise RuntimeError(f"V2 core bootstrap run failed: {msg1}")

    # 4) Regenerate the real manifest from loop_receipts (SSOT snapshot_id).
    ok2, _path2, _n2 = _v2_regen_manifest_from_receipts()
    if not ok2:
        raise RuntimeError("V2 core: manifest_full_scope regen failed")

    real_man = manifests_dir / "manifest_full_scope.jsonl"
    if not real_man.exists():
        raise RuntimeError(f"V2 core: real manifest not found after regen: {real_man}")

    try:
        _svr_run_ctx_update(
            manifest_full_scope_path=str(real_man),
        )
    except Exception:
        pass

    # 5) Real suite run must use canonical SSOT snapshot_id.
    # IMPORTANT: the suite runner only sets st.session_state['world_snapshot_id']
    # if it's absent, so we force it here before the real run.
    try:
        import streamlit as st  # type: ignore

        st.session_state["world_snapshot_id"] = run_snapshot_id
    except Exception:
        pass

    try:
        _svr_run_ctx_update(run_label="v2_suite_64")
    except Exception:
        pass

    ok3, msg3, _cnt3 = run_suite_from_manifest(str(real_man), run_snapshot_id)
    if not ok3:
        raise RuntimeError(f"V2 core suite run failed: {msg3}")

    # Best-effort: emit C1 rollup now; strict presence is enforced by the gate vector later.
    try:
        _coverage_rollup_write_csv(snapshot_id=run_snapshot_id)
    except Exception:
        pass

    return run_snapshot_id



def export_bundle_for_snapshot(snapshot_id: str | None = None, *, activate_derived_worlds: bool = False) -> _Path:
    """Single-path entrypoint (strict): V2 → C1/C2/C3/C4 → D4 → B1 → ZIP.

    This is the backend for the one-click "zip the folder and walk away" button.

    Strict wiring:
      - Runs the V2 core 64× flow inside this entrypoint (no separate UI entrypoint).
      - Calls `strict_single_path_gate_vector(...)` at stage boundaries and raises
        on the first failure, so the button is not "best-effort".

    Returns:
        Absolute path to `logs/bundle/bundle__{sid}__{sig8}.zip`.
    """
    import os as _os
    from zipfile import ZipFile, ZIP_DEFLATED

    def _raise_gate(stage_target: str, sid: str, run_ctx: dict | None = None):
        ok, stage, missing, mismatches = strict_single_path_gate_vector(
            snapshot_id=sid,
            run_ctx=run_ctx,
            target_stage=stage_target,
            activate_derived_worlds=activate_derived_worlds,
        )
        if ok:
            return
        parts = [f"Single-path gate FAILED at stage={stage} (target={stage_target}) for snapshot_id={sid}"]
        if missing:
            parts.append("Missing artifacts:")
            parts.extend([f"  - {p}" for p in missing])
        if mismatches:
            parts.append("Mismatches / violations:")
            parts.extend([f"  - {p}" for p in mismatches])
        raise RuntimeError("\n".join(parts))

    # 0) V2 strict core flow (mints/refreshes SSOT snapshot and suite artifacts).
    sid = _v2_run_core_flow_64x_strict()

    # If caller provided a snapshot_id, treat it as a guard only.
    if snapshot_id and str(snapshot_id).strip() and str(snapshot_id).strip() != sid:
        raise RuntimeError(
            f"export_bundle_for_snapshot: snapshot_id guard mismatch: arg={snapshot_id!r} vs SSOT={sid!r}"
        )

    # Gate S0+V2 immediately after the V2 core run.
    _raise_gate("V2", sid, run_ctx=None)

    # 1) Ensure the Time(τ) lab is fully up to date for the same SSOT snapshot.
    sid, run_ctx = run_full_v2_suite_and_tau(snapshot_id=sid)
    sid_str = str(sid)

    # Gate through C4 before certifying/exporting.
    _raise_gate("C4", sid_str, run_ctx=run_ctx)

    # 2) D4 certificate (required for B1 export).
    write_d4_certificate_for_snapshot(snapshot_id=sid_str, run_ctx=run_ctx)

    # Gate D4.
    _raise_gate("D4", sid_str, run_ctx=run_ctx)

    # 3) Build the B1 bundle tree + manifest for this snapshot.
    bundle_dir, state = _b1_write_bundle_tree_for_snapshot(snapshot_id=sid_str, run_ctx=run_ctx, activate_derived_worlds=activate_derived_worlds)
    manifest = state.get("bundle_manifest") or {}
    sig8 = str(manifest.get("sig8") or "").strip()
    if not sig8:
        raise RuntimeError("export_bundle_for_snapshot: bundle manifest is missing sig8.")

    # Gate B1 (bundle tree present + anchors).
    _raise_gate("B1", sid_str, run_ctx=run_ctx)

    bundle_dir = _Path(bundle_dir).resolve()
    parent_dir = bundle_dir.parent  # logs/bundle
    zip_name = f"bundle__{sid_str}__{sig8}.zip"
    zip_path = parent_dir / zip_name

    # 4) Deterministic traversal of B1 tree to build the zip.
    def _iter_bundle_files(root: _Path):
        """Yield file paths under root in a deterministic, lexicographic order."""
        root = _Path(root)
        for root_dir, dirnames, filenames in _os.walk(root):
            dirnames.sort()
            filenames.sort()
            root_path = _Path(root_dir)
            for fname in filenames:
                p = root_path / fname
                rel = p.relative_to(root)
                yield rel

    with ZipFile(str(zip_path), "w", compression=ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for rel in _iter_bundle_files(bundle_dir):
            # Archive paths are always POSIX-style and rooted under "bundle/".
            arc = "bundle/" + rel.as_posix()
            if arc in seen:
                continue
            seen.add(arc)
            zf.write(str(bundle_dir / rel), arcname=arc)

    # 5) Gate ZIP last.
    _raise_gate("ZIP", sid_str, run_ctx=run_ctx)

    return zip_path


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

    

    # B3 termination + closure: capture bundle_index payload_sig8 if available.
    term_status = (extra or {}).get("terminal_status")
    term_reason = (extra or {}).get("termination_reason_code")
    term_metrics = (extra or {}).get("termination_metrics")
    closure = None
    try:
        bix = bdir / "bundle_index.v2.json"
        if bix.exists():
            bi_obj = _json.loads(bix.read_text(encoding="utf-8"))
            if isinstance(bi_obj, dict):
                closure = {
                    "bundle_index": {
                        "filename": "bundle_index.v2.json",
                        "payload_sig8": (bi_obj or {}).get("payload_sig8"),
                    },
                    "roles_expected": list((bi_obj.get("roles") or {}).keys()) if isinstance(bi_obj.get("roles"), dict) else None,
                }
    except Exception:
        closure = None

    receipt = build_v2_loop_receipt(
        run_id=(extra or {}).get("run_id"),
        district_id=district_id,
        fixture_label=fixture_label,
        sig8=sig8,
        bundle_dir=bdir,
        paths=P,
        core_written=core_written,
        terminal_status=term_status,
        termination_reason_code=term_reason,
        termination_metrics=term_metrics if isinstance(term_metrics, dict) else None,
        closure=closure if isinstance(closure, dict) else None,
        dims=dims,
        extra=extra,
    )


    # Always write with proper filename (no UNKNOWN).
    outp = make_loop_receipt_path(bdir, fixture_label)
    try:
        _guarded_atomic_write_json(outp, receipt)
    except Exception:
        # Fallback to the legacy pretty writer rather than silently dropping.
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

TIME_TAU_C2_SWEEP_SCHEMA_VERSION = "time_tau_c2_sweep_row_v0.1"

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
    district_id = rec.get("district_id") or (
        fixture_label.split("_")[0] if fixture_label else "DUNKNOWN"
    )

    # D3.1.C.E — SSOT alignment (strict).
    # Resolve the canonical world snapshot once; require it to exist and agree
    # with any manifest snapshot_id embedded in the row.
    ssot_snapshot_id = _v2_current_world_snapshot_id(strict=False)
    if not ssot_snapshot_id:
        raise RuntimeError(
            "Time(τ) C2: no canonical v2 world snapshot_id found; "
            "run the v2 core 64× flow first."
        )

    rec_snapshot_id = rec.get("snapshot_id")
    if rec_snapshot_id and str(rec_snapshot_id) != str(ssot_snapshot_id):
        raise RuntimeError(
            "Time(τ) C2: manifest row snapshot_id mismatch "
            f"(fixture={fixture_label!r}, manifest={rec_snapshot_id!r}, "
            f"SSOT={ssot_snapshot_id!r}). "
            "Rerun the v2 core 64× flow to regenerate manifest_full_scope.jsonl."
        )

    snapshot_id = ssot_snapshot_id


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

    # Cached predicate: whether any flips were law-OK under the toy.
    has_law_ok_flips = bool(
        (H2_stats.get("law_ok", 0) or 0)
        or (d3_stats.get("law_ok", 0) or 0)
    )

    row = {
        "schema": "time_tau_c2_sweep_row",
        "schema_version": TIME_TAU_C2_SWEEP_SCHEMA_VERSION,
        "engine_rev": ENGINE_REV,
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
        "has_law_ok_flips": has_law_ok_flips,
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
                f_jsonl.write(canonical_json(r) + "\n")
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
    

    




if False:  # [single-path] deprecated: V2 core runs under the one-click pipeline
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

                       # D3.1.C.E — Bootstrap gating:
        # The “real” suite run must use exactly the SSOT snapshot_id. We resolve it
        # here, after writing the world snapshot, and refuse to fall back to any
        # ad-hoc ids.
        run_snapshot_id = _v2_current_world_snapshot_id(strict=True)
    
        # 2) run 64× to emit receipts — SNAPSHOT __boot
        snap_boot = f"{run_snapshot_id}__boot"

        # D3/F1 — mark bootstrap coverage rows via run_ctx so that C1/D3
        # can gate them out when counting passes.
        try:
            _svr_run_ctx_update(run_label="v2_bootstrap_64")
        except Exception as e:
            _st.warning(f"run_ctx bootstrap run_label wiring failed (non-fatal): {e}")

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
        # D3/F1 — mark real suite rows distinctly so only non-bootstrap
        # runs contribute to C1/D3 coverage stats.
        try:
            _svr_run_ctx_update(run_label="v2_suite_64")
        except Exception as e:
            _st.warning(f"run_ctx real run_label wiring failed (non-fatal): {e}")

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
            ssot_snapshot_id = _v2_current_world_snapshot_id(strict=False)
            path_csv = _coverage_rollup_write_csv(snapshot_id=ssot_snapshot_id)
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
                dbg = _c1_debug_snapshot_summary(_v2_current_world_snapshot_id(strict=False))
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


# --- Time(τ) T₀ continuity (diagnostic, v0) ---
# These artifacts are additive and do not gate C3/C4/D4.
TIME_TAU_T0_FRAME_SCHEMA = "tau_frame_receipt"
TIME_TAU_T0_FRAME_SCHEMA_VERSION = "tau_frame_receipt.v0"
TIME_TAU_T0_LINK_SCHEMA = "tau_t0_link"
TIME_TAU_T0_LINK_SCHEMA_VERSION = "tau_t0_link.v0"
TIME_TAU_T0_DIGEST_SCHEMA = "tau_t0_digest"
TIME_TAU_T0_DIGEST_SCHEMA_VERSION = "tau_t0_digest.v0"

TIME_TAU_T0_LINKS_JSONL = "time_tau_t0_links.jsonl"
TIME_TAU_T0_DIGEST_JSONL = "time_tau_t0_digest.jsonl"



# --- Time(τ) τ-state (v0.1) ---
TIME_TAU_STATE_SCHEMA = "time_tau_state"
TIME_TAU_STATE_SCHEMA_VERSION = "time_tau_state_v0_1"
TIME_TAU_STATE_FILENAME_FMT = "time_tau_state__{snapshot_id}.json"




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

    # Optional T₀ per-frame diagnostic data (embedded in the C3 receipt).
    t0_t_index = None
    t0_defined = None
    t0_sha256 = ""
    t0_coords = None
    try:
        fr = j.get("tau_frame_receipt")
        if isinstance(fr, dict):
            fr_ref = fr.get("frame_ref") or {}
            if isinstance(fr_ref, dict):
                t0_t_index = fr_ref.get("t_index")
                try:
                    t0_t_index = int(t0_t_index) if t0_t_index is not None else None
                except Exception:
                    t0_t_index = None

            c = fr.get("coords")
            if isinstance(c, dict):
                t0_coords = c

            fp = fr.get("obstruction_fp") or {}
            if isinstance(fp, dict):
                d = fp.get("defined")
                if d is not None:
                    t0_defined = bool(d)
                sha = fp.get("sha256")
                if sha is not None:
                    t0_sha256 = str(sha)
    except Exception:
        pass

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
        "t0_t_index": t0_t_index,
        "t0_defined": t0_defined,
        "t0_sha256": t0_sha256,
        "t0_coords": t0_coords,
    }

# ----------------------------------------------------------------------
# Time(τ) T₀ continuity helpers (diagnostic, v0)
# ----------------------------------------------------------------------

# Cache coordinate-lock digests by snapshot_id so C3/C4 do not recompute them per flip.
_TIME_TAU_T0_COORDS_CACHE: dict[str, dict] = {}

_T0_COORD_KEYS = (
    "snapshot_id",
    "world_snapshot_sig8",
    "fixtures_set_sig8",
    "b5_set_hex",
)


def _time_tau_t0_coords_lock(snapshot_id: str | None) -> dict:
    """Return the frozen coordinate-lock dict for T₀ continuity (best-effort).

    The coordinate lock is intentionally narrow and mirrors the D4 identity checks:
      - snapshot_id (SSOT)
      - world_snapshot_sig8
      - fixtures_set_sig8
      - b5_set_hex

    This helper performs no solver work.
    """
    sid = str(snapshot_id or "").strip()
    if not sid:
        try:
            sid = str(_v2_current_world_snapshot_id(strict=True) or "")
        except Exception:
            sid = ""

    if sid and sid in _TIME_TAU_T0_COORDS_CACHE:
        return dict(_TIME_TAU_T0_COORDS_CACHE[sid])

    coords = {
        "snapshot_id": sid,
        "world_snapshot_sig8": "",
        "fixtures_set_sig8": "",
        "b5_set_hex": "",
    }

    try:
        ws = _d4_world_snapshot_digest(sid)
        coords["world_snapshot_sig8"] = str((ws or {}).get("world_snapshot_sig8") or "")
    except Exception:
        pass

    try:
        md = _d4_manifest_digest(sid)
        coords["fixtures_set_sig8"] = str((md or {}).get("fixtures_set_sig8") or "")
    except Exception:
        pass

    try:
        b5 = _d4_b5_digest(sid)
        if isinstance(b5, dict) and b5.get("status") == "OK":
            coords["b5_set_hex"] = str(b5.get("b5_set_hex") or "")
    except Exception:
        pass

    if sid:
        _TIME_TAU_T0_COORDS_CACHE[sid] = dict(coords)

    return coords


def _time_tau_t0_coords_complete(coords: dict) -> bool:
    """Return True iff coords contains all required non-empty T₀ coordinate keys."""
    if not isinstance(coords, dict):
        return False
    for k in _T0_COORD_KEYS:
        v = coords.get(k)
        if not isinstance(v, str) or not v:
            return False
    return True


def _time_tau_t0_coords_equal(a: dict, b: dict) -> bool:
    """Return True iff a and b agree on all T₀ coordinate keys (string compare)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    for k in _T0_COORD_KEYS:
        if str(a.get(k) or "") != str(b.get(k) or ""):
            return False
    return True


def _time_tau_t0_norm_bitmatrix(mat) -> list[list[int]]:
    """Best-effort normalize a matrix into list[list[int]] of 0/1 ints."""
    out: list[list[int]] = []
    if not isinstance(mat, list):
        return out
    for row in mat:
        if not isinstance(row, list):
            continue
        r2: list[int] = []
        for v in row:
            try:
                r2.append(int(v) & 1)
            except Exception:
                r2.append(0)
        out.append(r2)
    return out


def _time_tau_t0_fp_v1_for_matrix(A: list[list[int]]) -> dict:
    """Best-effort fp.v1 hash for a bitmatrix using invariant tooling if present.

    Returns a minimal dict:
      {"defined": bool, "sha256": <hex64 or ''>, "reason": <str or None>}
    """
    cert_fn = globals().get("_inv_stepA_cert")
    fp_fn = globals().get("_inv_fpv1_sha256")
    if not callable(cert_fn) or not callable(fp_fn):
        return {"defined": False, "sha256": "", "reason": "FPV1_NOT_AVAILABLE"}

    try:
        cert = cert_fn(A)
        fp = fp_fn(cert)
        if isinstance(fp, dict) and fp.get("defined") is True and fp.get("sha256"):
            return {"defined": True, "sha256": str(fp.get("sha256") or ""), "reason": None}

        reason = None
        if isinstance(fp, dict):
            reason = fp.get("reason")
        if reason is None and isinstance(cert, dict):
            reason = cert.get("reason")

        return {"defined": False, "sha256": "", "reason": str(reason or "UNDEFINED")}
    except Exception as e:  # noqa: BLE001
        return {"defined": False, "sha256": "", "reason": f"FPV1_ERROR: {e}"}


def _time_tau_t0_build_frame_receipt(
    *,
    snapshot_id: str | None,
    fixture_label: str,
    strict_sig8: str,
    t_index: int | None,
    fp: dict | None,
) -> dict:
    """Build a tau_frame_receipt.v0 object (diagnostic, additive)."""
    coords = _time_tau_t0_coords_lock(snapshot_id)

    try:
        t_idx = int(t_index) if t_index is not None else None
    except Exception:
        t_idx = None

    defined = bool(fp.get("defined")) if isinstance(fp, dict) else False
    sha = str(fp.get("sha256") or "") if (isinstance(fp, dict) and defined) else ""
    if not defined:
        sha = ""

    obstruction_fp: dict = {
        "scheme": "fp.v1",
        "matrix_basis": "R3_strict",
        "defined": bool(defined),
        "sha256": sha,
    }
    # Attach a non-normative reason when undefined (useful for debugging).
    if isinstance(fp, dict) and fp.get("reason"):
        obstruction_fp["reason"] = str(fp.get("reason"))

    return {
        "schema": TIME_TAU_T0_FRAME_SCHEMA,
        "schema_version": TIME_TAU_T0_FRAME_SCHEMA_VERSION,
        "coords": coords,
        "frame_ref": {
            "fixture_label": str(fixture_label or "UNKNOWN"),
            "strict_sig8": str(strict_sig8 or ""),
            "t_index": t_idx,
        },
        "obstruction_fp": obstruction_fp,
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
                "schema": "time_tau_c3_rollup_row",
                "schema_version": TIME_TAU_C3_ROLLUP_SCHEMA_VERSION,
                "engine_rev": ENGINE_REV,
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
            "schema": "time_tau_c3_tau_mismatch",
            "schema_version": TIME_TAU_TAU_MISMATCH_SCHEMA_VERSION,
            "engine_rev": ENGINE_REV,
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
        "schema": "time_tau_coverage",
        "schema_version": TIME_TAU_COVERAGE_SCHEMA_VERSION,
        "engine_rev": ENGINE_REV,
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
    # D3.1.C: always use the canonical world snapshot_id (SSOT) and optional run_label.
    try:
        ssot_snapshot_id = _v2_current_world_snapshot_id(strict=False)
    except Exception:
        ssot_snapshot_id = None

    ping["snapshot_id"] = ssot_snapshot_id

    # Attach run_label as inert metadata when available.
    try:
        run_label = _svr_current_run_label()
    except Exception:
        run_label = None
    if run_label is not None:
        ping["run_label"] = run_label

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
        "schema",
        "schema_version",
        "engine_rev",
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
            fh.write(canonical_json(r) + "\n")

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
        "schema",
        "schema_version",
        "engine_rev",
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
            fh.write(canonical_json(r) + "\n")

    return str(outp)




def _time_tau_c4_write_t0_links_jsonl(rows: list[dict]) -> str:
    """Write T₀ continuity link rows to logs/reports/time_tau_t0_links.jsonl."""
    root = _repo_root()
    rep_dir = root / "logs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    outp = rep_dir / TIME_TAU_T0_LINKS_JSONL
    with outp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(canonical_json(r) + "\n")
    return str(outp)


def _time_tau_c4_write_t0_digest_jsonl(rows: list[dict]) -> str:
    """Write T₀ continuity digest rows to logs/reports/time_tau_t0_digest.jsonl."""
    root = _repo_root()
    rep_dir = root / "logs" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    outp = rep_dir / TIME_TAU_T0_DIGEST_JSONL
    with outp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(canonical_json(r) + "\n")
    return str(outp)


def _time_tau_c4_build_t0_outputs(norm_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build T₀ continuity link + digest rows from normalized C3 receipts.

    This is diagnostic-only: it never gates C4 rollup and never mutates solver artifacts.
    """
    # Group "frames" by fixture scope.
    frames_by_key: dict[tuple[str, str, str], list[dict]] = {}

    for r in norm_rows:
        if not isinstance(r, dict):
            continue
        district_id = str(r.get("district_id") or "DUNKNOWN")
        fixture_label = str(r.get("fixture_label") or "UNKNOWN")
        strict_sig8 = str(r.get("strict_sig8") or "")

        key = (district_id, fixture_label, strict_sig8)
        frames_by_key.setdefault(key, []).append(r)

    links: list[dict] = []
    digests: list[dict] = []

    # Deterministic order over fixtures.
    for (district_id, fixture_label, strict_sig8) in sorted(frames_by_key.keys()):
        frs = frames_by_key[(district_id, fixture_label, strict_sig8)]

        # Deterministic frame order: t_index primary, receipt path secondary.
        def _t_key(rr: dict) -> tuple[int, str]:
            t = rr.get("t0_t_index")
            if isinstance(t, int):
                return (t, str(rr.get("path") or ""))
            return (10**9, str(rr.get("path") or ""))

        frs_sorted = sorted(frs, key=_t_key)

        # Determine coords: prefer an embedded complete coords dict, else compute lock from snapshot_id.
        coords: dict = {}
        for rr in frs_sorted:
            c = rr.get("t0_coords")
            if isinstance(c, dict) and _time_tau_t0_coords_complete(c):
                coords = dict(c)
                break
        if not coords:
            snap = frs_sorted[0].get("snapshot_id") if frs_sorted else None
            coords = _time_tau_t0_coords_lock(str(snap) if snap is not None else None)

        # Compute t_min/t_max from available integer indices.
        t_vals = [rr.get("t0_t_index") for rr in frs_sorted if isinstance(rr.get("t0_t_index"), int)]
        t_min = min(t_vals) if t_vals else None
        t_max = max(t_vals) if t_vals else None

        counts = {
            "n_links": max(0, int(len(frs_sorted) - 1)),
            "same_id": 0,
            "changed_no_witness": 0,
            "changed_with_witness": 0,
            "na_undefined": 0,
        }

        # Links between adjacent frames in the chosen order.
        for i in range(len(frs_sorted) - 1):
            a = frs_sorted[i]
            b = frs_sorted[i + 1]

            from_t = a.get("t0_t_index")
            to_t = b.get("t0_t_index")

            ca = a.get("t0_coords") if isinstance(a.get("t0_coords"), dict) else None
            cb = b.get("t0_coords") if isinstance(b.get("t0_coords"), dict) else None

            da = a.get("t0_defined")
            db = b.get("t0_defined")

            sha_a = str(a.get("t0_sha256") or "")
            sha_b = str(b.get("t0_sha256") or "")

            # Default: undefined until proven otherwise.
            case = "NA_UNDEFINED"

            # Require consecutive integer indices.
            if isinstance(from_t, int) and isinstance(to_t, int) and to_t == from_t + 1:
                # Require complete + matching coords.
                if (
                    isinstance(ca, dict)
                    and isinstance(cb, dict)
                    and _time_tau_t0_coords_complete(ca)
                    and _time_tau_t0_coords_complete(cb)
                    and _time_tau_t0_coords_equal(ca, cb)
                ):
                    # Require defined fp with plausible sha256.
                    if (da is True) and (db is True) and (len(sha_a) == 64) and (len(sha_b) == 64):
                        if sha_a == sha_b:
                            case = "SAME_ID"
                        else:
                            # No transport witness in v0 (diagnostic defaults).
                            case = "CHANGED_NO_WITNESS"

            # Update counts.
            if case == "SAME_ID":
                counts["same_id"] += 1
            elif case == "CHANGED_NO_WITNESS":
                counts["changed_no_witness"] += 1
            elif case == "CHANGED_WITH_WITNESS":
                counts["changed_with_witness"] += 1
            else:
                counts["na_undefined"] += 1

            links.append(
                {
                    "schema": TIME_TAU_T0_LINK_SCHEMA,
                    "schema_version": TIME_TAU_T0_LINK_SCHEMA_VERSION,
                    "engine_rev": ENGINE_REV,
                    "coords": coords,
                    "from": {
                        "district_id": district_id,
                        "fixture_label": fixture_label,
                        "strict_sig8": strict_sig8,
                        "t_index": from_t,
                        "sha256": sha_a,
                        "defined": da,
                        "receipt_path": a.get("path"),
                    },
                    "to": {
                        "district_id": district_id,
                        "fixture_label": fixture_label,
                        "strict_sig8": strict_sig8,
                        "t_index": to_t,
                        "sha256": sha_b,
                        "defined": db,
                        "receipt_path": b.get("path"),
                    },
                    "transport_witness": None,
                    "continuity_case": case,
                }
            )

        digests.append(
            {
                "schema": TIME_TAU_T0_DIGEST_SCHEMA,
                "schema_version": TIME_TAU_T0_DIGEST_SCHEMA_VERSION,
                "engine_rev": ENGINE_REV,
                "coords": coords,
                "scope": {
                    "district_id": district_id,
                    "fixture_label": fixture_label,
                    "strict_sig8": strict_sig8,
                    "t_min": t_min,
                    "t_max": t_max,
                },
                "counts": counts,
                "mode": "DIAGNOSTIC",
            }
        )

    return links, digests

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

    # T₀ continuity (diagnostic, additive): link-by-link obstruction tracking.
    try:
        t0_links, t0_digests = _time_tau_c4_build_t0_outputs(norm_rows)
        _time_tau_c4_write_t0_links_jsonl(t0_links)
        _time_tau_c4_write_t0_digest_jsonl(t0_digests)
    except Exception:
        # Best-effort; never gate C4 on T₀ diagnostics.
        pass

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

   
        # D3.1.C.E — SSOT alignment (strict).
    # Resolve the canonical world snapshot once; require it to exist and agree
    # with any snapshot_id carried by the manifest row or strict cert.
    ssot_snapshot_id = _v2_current_world_snapshot_id(strict=False)
    if not ssot_snapshot_id:
        raise RuntimeError(
            "Time(τ) C3 baseline context: no canonical v2 world snapshot_id found; "
            "run the v2 core 64× flow first."
        )

    rec_snapshot_id = None
    if isinstance(rec, dict):
        rec_snapshot_id = rec.get("snapshot_id") or None

    strict_cert_snapshot_id = None
    if isinstance(strict_cert, dict):
        strict_cert_snapshot_id = (
            strict_cert.get("snapshot_id")
            or ((strict_cert.get("embed") or {}).get("snapshot_id"))
        )

    for src_name, sid in (
        ("manifest", rec_snapshot_id),
        ("strict_cert", strict_cert_snapshot_id),
    ):
        if sid and str(sid) != str(ssot_snapshot_id):
            raise RuntimeError(
                "Time(τ) C3 baseline context: snapshot_id mismatch "
                f"({src_name}={sid!r}, SSOT={ssot_snapshot_id!r}). "
                "Rerun the v2 core 64× flow to resync artifacts."
            )

    snapshot_id = ssot_snapshot_id


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
    t_index: int | None = None,
    t0_obstruction_fp: dict | None = None,
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
    Evaluate a derived world under τ-mode and return a raw observation dict.

    This does *not* write receipts. It is pure evaluation:

      - load B/C/H from the derived world paths
      - compute strict core → R3
      - compute defect set + parity
      - (diagnostic) compute a T₀ obstruction fingerprint: fp.v1 over strict-core R3

    Any exception is allowed to raise; the outer C3 driver is responsible for
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
    jC, _pC, _tagC = abx_read_json_any(C_path, kind="C")  # type: ignore[arg-type]
    jH, _pH, _tagH = abx_read_json_any(H_path, kind="H")  # type: ignore[arg-type]

    blocks_B = _svr_as_blocks_v2(jB, "B")
    blocks_C = _svr_as_blocks_v2(jC, "C")
    blocks_H = _svr_as_blocks_v2(jH, "H")

    core = time_tau_strict_core_from_blocks(blocks_B, blocks_C, blocks_H)
    R3 = core.get("R3")
    D = time_tau_defect_set_from_R3(R3)
    parity_after = len(D) % 2

    # T₀ obstruction fingerprint (diagnostic): fp.v1 over strict-core R3.
    try:
        R3_norm = _time_tau_t0_norm_bitmatrix(R3 or [])
        t0_fp = _time_tau_t0_fp_v1_for_matrix(R3_norm)
    except Exception as _exc:  # noqa: BLE001
        t0_fp = {"defined": False, "sha256": "", "reason": f"T0_FP_ERROR: {_exc}"}

    core_obs: dict = {
        "parity_after": int(parity_after),
        "defect_cols": [int(c) for c in D],
        "n_defect_cols": int(len(D)),
        "t0_obstruction_fp": t0_fp,
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
    t_index: int | None = None,
    t0_obstruction_fp: dict | None = None,
    derived_world: dict | None = None,
    tau_pred: dict | None = None,
    solver_observation: dict | None = None,
    c3_pass: bool | None = None,
    na_reason: str | None = None,
) -> dict:
    """Assemble the C3 v0.2 receipt dict and write it to logs/experiments.

    This helper is intentionally boring:
      - it does no solver work,
      - it only normalizes the payload and writes JSON.

    It expects that S0–S4 helpers have already decided c3_pass/NA.

    Additive wiring (v0):
      - embeds a `tau_frame_receipt.v0` object for T₀ continuity diagnostics.
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
    # We treat the human-readable location/label as ephemeral and carry only
    # the mutated_matrix_sig8 as canonical.
    if isinstance(derived_world, dict):
        # Support both legacy "derived_artifact_path" and any future
        # "derived_world_label" key on the incoming context.
        derived_world_label = (
            derived_world.get("derived_world_label")
            or derived_world.get("derived_artifact_path")
        )
        mutated_matrix_sig8 = derived_world.get("mutated_matrix_sig8")
    else:
        derived_world_label = None
        mutated_matrix_sig8 = None

    derived_world_block: dict = {}
    if derived_world_label is not None:
        derived_world_block["derived_world_label"] = derived_world_label
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
        "schema": "time_tau_c3_receipt",
        "schema_version": TIME_TAU_C3_RECEIPT_SCHEMA_VERSION,
        "engine_rev": ENGINE_REV,
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

    # T₀ frame receipt (diagnostic): per-flip obstruction fingerprint + coords lock.
    try:
        receipt["tau_frame_receipt"] = _time_tau_t0_build_frame_receipt(
            snapshot_id=snapshot_id,
            fixture_label=fixture_label,
            strict_sig8=strict_sig8,
            t_index=t_index,
            fp=t0_obstruction_fp,
        )
    except Exception:
        # Best-effort: do not fail C3 receipt writing if T₀ wiring fails.
        pass

    # Compute deterministic filename.
    root = _repo_root()
    exp_dir = root / "logs" / "experiments"
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot ensure the directory, still return the in-memory receipt.
        return receipt

    # Deterministic filename. Build from stable fields only (no timestamps).
    def _safe_token(x: str) -> str:
        try:
            s = str(x)
        except Exception:
            s = "UNKNOWN"
        s = s.strip() or "UNKNOWN"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    kind_tok = _safe_token(kind)
    fixture_tok = _safe_token(fixture_label)
    strict_tok = _safe_token(strict_sig8) if strict_sig8 else "nosig"
    # Indices: include all three so H2 and d3 flips have disambiguation.
    i_tok = str(i_idx) if i_idx is not None else "na"
    j_tok = str(j_idx) if j_idx is not None else "na"
    k_tok = str(k_idx) if k_idx is not None else "na"

    fname = f"time_tau_c3_recompute__{fixture_tok}__{kind_tok}__i{i_tok}_j{j_tok}_k{k_tok}__{strict_tok}.json"
    outp = exp_dir / fname
    try:
        _hard_co_write_json(outp, receipt)
    except Exception:
        # Best-effort: even if write fails, return the in-memory receipt.
        return receipt

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

    for t_index, flip_ref in enumerate(flip_refs):
        t0_obstruction_fp = None
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
                t_index=t_index,
                t0_obstruction_fp=t0_obstruction_fp,
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
                t_index=t_index,
                t0_obstruction_fp=t0_obstruction_fp,
                derived_world=derived_world,
                tau_pred=tau_pred,
                solver_observation=None,
                c3_pass=None,
                na_reason=na_code,
            )
            receipts.append(receipt)
            continue

        # T₀ obstruction fp extracted from core_obs (may be undefined).
        try:
            t0_obstruction_fp = core_obs.get("t0_obstruction_fp") if isinstance(core_obs, dict) else None
        except Exception:
            t0_obstruction_fp = None

        try:
            solver_observation = _time_tau_c3_normalize_observation(core_obs)
        except Exception:
            na_code = _time_tau_c3_na("OBSERVATION_PARSE_ERROR")
            receipt = _time_tau_c3_build_and_write_receipt(
                base_ctx=base_ctx,
                flip_ref=flip_ref,
                t_index=t_index,
                t0_obstruction_fp=t0_obstruction_fp,
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
                t_index=t_index,
                t0_obstruction_fp=t0_obstruction_fp,
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
                t_index=t_index,
                t0_obstruction_fp=t0_obstruction_fp,
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


def time_tau_c3_write_receipts_manifest(
    *,
    snapshot_id: str | None = None,
    c3_manifest_path: str | None = None,
    receipts_dir: str | None = None,
) -> tuple[bool, str, dict]:
    """Write the canonical Time(τ) C3 receipts manifest (explicit inventory; may be empty).

    This is a B4 prerequisite: bundle export must never infer "no receipts"
    from a directory listing. Instead, we write an explicit manifest that can
    be validly empty (mode="empty") with an explicit reason.
    """
    # Resolve snapshot_id.
    sid = snapshot_id
    if not sid:
        try:
            sid = _v2_current_world_snapshot_id(strict=True)
        except Exception:
            sid = None
    if not sid:
        return False, "C3 receipts manifest: unable to resolve snapshot_id", {}

    sid_str = str(sid)

    # Resolve repo root.
    try:
        root = _repo_root()
    except Exception:
        root = _Path(__file__).resolve().parents[1]

    # Manifests directory.
    manifests_dir = _MANIFESTS_DIR if "_MANIFESTS_DIR" in globals() else (_Path(root) / "logs" / "manifests")
    manifests_dir = _Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Receipts directory (defaults to logs/experiments).
    if receipts_dir:
        exp_dir = _Path(receipts_dir)
        if not exp_dir.is_absolute():
            exp_dir = _Path(root) / exp_dir
    else:
        try:
            exp_dir = _Path(DIRS.get("experiments", "logs/experiments"))  # type: ignore[name-defined]
            if not exp_dir.is_absolute():
                exp_dir = _Path(root) / exp_dir
        except Exception:
            exp_dir = _Path(root) / "logs" / "experiments"
    exp_dir = exp_dir.resolve()

    # Output path is snapshot-scoped to avoid collisions.
    outp = manifests_dir / f"time_tau_c3_receipts_manifest__{sid_str}.json"

    # Enumerate receipts deterministically.
    receipt_files = []
    try:
        receipt_files = sorted(exp_dir.glob("time_tau_c3_recompute__*.json"), key=lambda p: p.name)
    except Exception:
        receipt_files = []

    receipts: list[dict] = []
    for p in receipt_files:
        rel = None
        try:
            rel = _bundle_repo_relative_path(p)  # type: ignore[name-defined]
        except Exception:
            rel = str(p)

        # Parse base identity fields when possible.
        district_id = None
        fixture_label = None
        strict_sig8 = None
        try:
            obj = _json.loads(_Path(p).read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                base = obj.get("base") or {}
                if isinstance(base, dict):
                    district_id = base.get("district_id")
                    fixture_label = base.get("fixture_label")
                    strict_sig8 = base.get("baseline_strict_sig8")
        except Exception:
            pass

        try:
            sig8 = _hash_file_sig8(_Path(p))
        except Exception:
            sig8 = ""

        receipts.append(
            {
                "district_id": str(district_id) if district_id is not None else None,
                "fixture_label": str(fixture_label) if fixture_label is not None else None,
                "strict_sig8": str(strict_sig8) if strict_sig8 is not None else None,
                "receipt_relpath": rel,
                "receipt_sig8": sig8,
            }
        )

    mode = "present" if receipts else "empty"
    reason = None
    if mode == "empty":
        # For the current stage we conservatively treat emptiness as allowed.
        # Later taxonomy can tighten this into MUST-IF vs illegal absence.
        reason = "NO_RECEIPTS_FOUND"

    c3_manifest_rel = None
    if c3_manifest_path:
        try:
            c3_manifest_rel = _bundle_repo_relative_path(c3_manifest_path)  # type: ignore[name-defined]
        except Exception:
            c3_manifest_rel = str(c3_manifest_path)

    payload = {
        "schema": "time_tau_c3_receipts_manifest",
        "schema_version": TIME_TAU_C3_RECEIPTS_MANIFEST_SCHEMA_VERSION,
        "engine_rev": ENGINE_REV,
        "snapshot_id": sid_str,
        "c3_manifest_relpath": c3_manifest_rel,
        "receipts_dir_relpath": (
            _bundle_repo_relative_path(exp_dir)  # type: ignore[name-defined]
            if exp_dir is not None else None
        ),
        "mode": mode,
        "emptiness": {
            "empty_ok": True,
            "reason": reason,
        } if mode == "empty" else {
            "empty_ok": True,
            "reason": None,
        },
        "receipts": receipts,
        "summary": {
            "n_receipts": len(receipts),
        },
    }

    try:
        # v0.2: embed a self-sig8 binding the load-bearing projection
        payload["sig8"] = hash_json_sig8(_time_tau__lb_projection_receipts_manifest(payload))
        outp.write_text(canonical_json(payload), encoding="utf-8")
    except Exception as e:
        return False, f"C3 receipts manifest: failed to write {outp}: {e}", {}

    msg = f"C3 receipts manifest written: n={len(receipts)} → {outp}"
    summary = {
        "receipts_manifest_path": str(outp),
        "n_receipts": len(receipts),
        "mode": mode,
    }
    return True, msg, summary

# ---------------------------------------------------------------------------
# Time(τ) τ-state object (v0.1): minimal stable surface for blueprint + wiring.
#
# This is an inventory of already-produced τ artifacts (C2/C3/C4) for a given
# SSOT snapshot_id. It introduces no new solver work and performs no inference.
# ---------------------------------------------------------------------------

def _tau_state_repo_relpath(p: _Path | None, root: _Path | None) -> str:
    """Best-effort repo-relative path (POSIX). Falls back to str(p)."""
    if p is None:
        return ""
    try:
        rp = _Path(p).resolve()
        rr = _Path(root).resolve() if root is not None else None
        if rr is not None:
            return rp.relative_to(rr).as_posix()
        return rp.as_posix()
    except Exception:
        try:
            return str(p)
        except Exception:
            return ""


def _tau_state_sig8(p: _Path | None) -> str:
    """Best-effort sig8 for an on-disk file; empty string if missing/error."""
    try:
        if p is None:
            return ""
        pp = _Path(p)
        if not pp.exists():
            return ""
        return str(_hash_file_sig8(pp))
    except Exception:
        return ""


def time_tau_build_state_v0_1(
    *,
    snapshot_id: str,
    manifest_full_scope_path: str | None,
    summary_c2: dict | None,
    summary_c3m: dict | None,
    summary_c3s: dict | None,
    summary_rm: dict | None,
    summary_c4: dict | None,
) -> dict:
    """Build τ-state (v0.1) as a residual inventory of already-produced artifacts.

    Contract:
      - No new solver work.
      - No inference / no directory scanning beyond pointing at already-frozen
        canonical paths.
      - Only path resolution + best-effort sig8 attachment.
    """
    import time as _time  # local import

    sid = str(snapshot_id or "")

    try:
        root = _repo_root()
    except Exception:
        root = _Path(__file__).resolve().parents[1]
    root_p = _Path(root)

    # Canonical C4 report outputs live under logs/reports with fixed names.
    rep_dir = root_p / "logs" / "reports"
    rollup_csv = rep_dir / TIME_TAU_C3_ROLLUP_CSV
    rollup_jsonl = rep_dir / TIME_TAU_C3_ROLLUP_JSONL
    mism_csv = rep_dir / TIME_TAU_TAU_MISMATCH_CSV
    mism_jsonl = rep_dir / TIME_TAU_TAU_MISMATCH_JSONL
    t0_links_jsonl = rep_dir / TIME_TAU_T0_LINKS_JSONL
    t0_digest_jsonl = rep_dir / TIME_TAU_T0_DIGEST_JSONL

    # C2 outputs (paths should be present in summary_c2 when C2 succeeded).
    s2 = summary_c2 or {}
    c2_jsonl = _Path(str(s2.get("jsonl_path"))) if s2.get("jsonl_path") else None
    c2_csv = _Path(str(s2.get("csv_path"))) if s2.get("csv_path") else None

    # C3 manifest path: prefer the explicit manifest_c3_path from the builder summary.
    s3m = summary_c3m or {}
    s3s = summary_c3s or {}
    c3_manifest_path_str = (
        s3m.get("manifest_c3_path")
        or s3m.get("manifest_path")
        or s3s.get("manifest_path")
        or ""
    )
    c3_manifest = _Path(str(c3_manifest_path_str)) if c3_manifest_path_str else None

    # C3 receipts manifest path (explicit inventory).
    srm = summary_rm or {}
    receipts_manifest_path_str = str(srm.get("receipts_manifest_path") or "")
    receipts_manifest = _Path(receipts_manifest_path_str) if receipts_manifest_path_str else None

    # Receipts dir default must match the receipts-manifest writer logic.
    try:
        exp_dir = _Path(DIRS.get("experiments", "logs/experiments"))  # type: ignore[name-defined]
        if not exp_dir.is_absolute():
            exp_dir = root_p / exp_dir
    except Exception:
        exp_dir = root_p / "logs" / "experiments"
    exp_dir = exp_dir.resolve()

    # Derived worlds dir follows the same DIRS binding used by C3/D4 validation.
    try:
        c3_worlds_dir = _Path(DIRS.get("c3_worlds", "app/inputs/c3_derived_worlds"))  # type: ignore[name-defined]
        if not c3_worlds_dir.is_absolute():
            c3_worlds_dir = root_p / c3_worlds_dir
    except Exception:
        c3_worlds_dir = root_p / "app" / "inputs" / "c3_derived_worlds"
    c3_worlds_dir = c3_worlds_dir.resolve()

    st_obj: dict = {
        "schema": TIME_TAU_STATE_SCHEMA,
        "schema_version": TIME_TAU_STATE_SCHEMA_VERSION,
        "engine_rev": globals().get("ENGINE_REV", ""),
        "snapshot_id": sid,
        "built_at_utc": int(_time.time()),
        "inputs": {
            "manifest_full_scope_path": str(manifest_full_scope_path or ""),
            "manifest_full_scope_relpath": _tau_state_repo_relpath(
                _Path(manifest_full_scope_path), root_p
            ) if manifest_full_scope_path else "",
        },
        "c2": {
            "jsonl_path": str(c2_jsonl) if c2_jsonl else "",
            "csv_path": str(c2_csv) if c2_csv else "",
            "jsonl_relpath": _tau_state_repo_relpath(c2_jsonl, root_p) if c2_jsonl else "",
            "csv_relpath": _tau_state_repo_relpath(c2_csv, root_p) if c2_csv else "",
            "jsonl_sig8": _tau_state_sig8(c2_jsonl),
            "csv_sig8": _tau_state_sig8(c2_csv),
        },
        "c3": {
            "manifest_path": str(c3_manifest) if c3_manifest else "",
            "receipts_manifest_path": str(receipts_manifest) if receipts_manifest else "",
            "receipts_dir": str(exp_dir),
            "derived_worlds_dir": str(c3_worlds_dir),
            "manifest_relpath": _tau_state_repo_relpath(c3_manifest, root_p) if c3_manifest else "",
            "receipts_manifest_relpath": _tau_state_repo_relpath(receipts_manifest, root_p) if receipts_manifest else "",
            "receipts_dir_relpath": _tau_state_repo_relpath(exp_dir, root_p),
            "derived_worlds_dir_relpath": _tau_state_repo_relpath(c3_worlds_dir, root_p),
            "manifest_sig8": _tau_state_sig8(c3_manifest),
            "receipts_manifest_sig8": _tau_state_sig8(receipts_manifest),
        },
        "c4": {
            "rollup_jsonl_path": str(rollup_jsonl),
            "rollup_csv_path": str(rollup_csv),
            "tau_mismatches_jsonl_path": str(mism_jsonl),
            "tau_mismatches_csv_path": str(mism_csv),
            "t0_links_jsonl_path": str(t0_links_jsonl),
            "t0_digest_jsonl_path": str(t0_digest_jsonl),
            "rollup_jsonl_relpath": _tau_state_repo_relpath(rollup_jsonl, root_p),
            "rollup_csv_relpath": _tau_state_repo_relpath(rollup_csv, root_p),
            "tau_mismatches_jsonl_relpath": _tau_state_repo_relpath(mism_jsonl, root_p),
            "tau_mismatches_csv_relpath": _tau_state_repo_relpath(mism_csv, root_p),
            "t0_links_jsonl_relpath": _tau_state_repo_relpath(t0_links_jsonl, root_p),
            "t0_digest_jsonl_relpath": _tau_state_repo_relpath(t0_digest_jsonl, root_p),
            "rollup_jsonl_sig8": _tau_state_sig8(rollup_jsonl),
            "rollup_csv_sig8": _tau_state_sig8(rollup_csv),
            "tau_mismatches_jsonl_sig8": _tau_state_sig8(mism_jsonl),
            "tau_mismatches_csv_sig8": _tau_state_sig8(mism_csv),
            "t0_links_jsonl_sig8": _tau_state_sig8(t0_links_jsonl),
            "t0_digest_jsonl_sig8": _tau_state_sig8(t0_digest_jsonl),
        },
        # Keep the already-produced summaries available for UI/debug, but do not
        # make them load-bearing for export; the load-bearing surface is paths.
        "summary": {
            "c2": summary_c2 or {},
            "c3_manifest": summary_c3m or {},
            "c3_sweep": summary_c3s or {},
            "c3_receipts_manifest": summary_rm or {},
            "c4_district_summary": summary_c4 or {},
        },
    }
    return st_obj


def time_tau_write_state_v0_1(*, tau_state: dict, snapshot_id: str) -> tuple[bool, str, dict]:
    """Write τ-state to logs/manifests/time_tau_state__{snapshot_id}.json (snapshot-scoped)."""
    sid = str(snapshot_id or "").strip()
    if not sid:
        return False, "τ-state write: missing snapshot_id", {}

    try:
        root = _repo_root()
    except Exception:
        root = _Path(__file__).resolve().parents[1]

    manifests_dir = _MANIFESTS_DIR if "_MANIFESTS_DIR" in globals() else (_Path(root) / "logs" / "manifests")
    manifests_dir = _Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    outp = manifests_dir / TIME_TAU_STATE_FILENAME_FMT.format(snapshot_id=sid)

    try:
        outp.write_text(canonical_json(tau_state), encoding="utf-8")
    except Exception as e:
        return False, f"τ-state write failed: {e}", {}

    return True, "OK", {"tau_state_path": str(outp)}


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
# --- V2 CORE (64×) — one press → receipts → manifest → suite → hist/zip
_st.subheader("V2 — 64× → Receipts → Manifest → Suite/Histograms")



try:
    _ssot_snapshot_id_for_bundle = _v2_current_world_snapshot_id(strict=False)
except Exception:
    _ssot_snapshot_id_for_bundle = None

snapshot_id = _st.text_input(
    "Snapshot id",
    value=(_ssot_snapshot_id_for_bundle or ""),
    key="v2_core_snapshot",
)

if _st.button("Run v2 core + Time(τ) lab and export bundle", key="btn_v2_bundle_export"):
    with _st.spinner("Running 64× v2 + τ lab and building bundle..."):
        try:
            # D3/F2 — Treat export as SSOT-anchored; let the backend
            # resolve the canonical snapshot_id instead of trusting
            # free-form UI text.
            zip_path = export_bundle_for_snapshot(snapshot_id=None)
        except Exception as e:
            _st.error(f"Bundle export failed: {e}")
        else:
            _st.success(f"Bundle ready → {zip_path}")
            try:
                data = _Path(zip_path).read_bytes()
            except Exception as e:
                _st.warning(f"Bundle zip exists but could not be read: {e}")
            else:
                _st.download_button(
                    "Download v2+τ bundle zip",
                    data=data,
                    file_name=_Path(zip_path).name,
                    key="btn_v2_bundle_download",
                )



with _st.expander("D4 — Certifier & Bundle export", expanded=False):
    render_d4_certifier_panel()

# --------------

# ===================== Invariant Wiring / Receipts (Session Bricks 1–6) =====================
# This section is intentionally *read-only* with respect to solver behavior.
# It adds observability + non-forking receipts without modifying any gate/solver logic.

# --- Frozen wiring profile IDs (session-local) ---
_INV_SESSION_PROFILE_ID = "SessionFrozen.Bricks1to6.v1"
_INV_STEP_A_PROFILE_ID  = "axiomA.stepA.degree_pure.v1"
_INV_FPV1_PROFILE_ID    = "fp.v1.fixed_key_order.v1"
_INV_TYPE_GATE_ID       = "StepA.TypeGate.no_new_nonzero_type.v1"
_INV_BARCODE_PROFILE_ID = "barcode.v1.sequence_equality.v1"

# --- Phase 2 / Step-B (boundary probes → explicit out-of-regime detector) ---
# These IDs are carried for traceability only; the semantic surface is the
# Step-A admissibility classification (PASS ⇒ admissible, else not admissible)
# plus the new-type delta evidence.
_INV_STEP_B_PROFILE_ID            = "StepB.OutOfRegimeDetector.new_type_delta.v1"
_INV_STEP_B_RECORD_SCHEMA         = "stepb_out_of_regime_record"
_INV_STEP_B_RECORD_SCHEMA_VERSION = "stepb.v1"

# --- fp.v1 computational guards (explicit; avoids UI hangs) ---
_INV_FPV1_MAX_KERNEL_ENUM_DIM = 20  # enumerate 2^k kernel combos only when k <= 20


if "_inv__popcount" not in globals():
    def _inv__popcount(x: int) -> int:
        try:
            return int(x.bit_count())
        except Exception:
            # Python < 3.8 fallback (should not occur in modern runtimes)
            return bin(int(x) & ((1 << 4096) - 1)).count("1")


if "_inv__shape" not in globals():
    def _inv__shape(M) -> tuple[int, int]:
        if not M:
            return (0, 0)
        try:
            return (len(M), len(M[0]) if M and isinstance(M[0], (list, tuple)) else 0)
        except Exception:
            return (0, 0)


if "_inv__norm_bitmatrix" not in globals():
    def _inv__norm_bitmatrix(mat, *, name: str = "A") -> list[list[int]]:
        """Best-effort normalize to list[list[int]] over F2.

        - Accepts [] (empty) and returns []
        - Enforces non-ragged if rows exist
        - Coerces bool/int-like; rejects floats/None
        """
        if mat is None:
            return []
        if not isinstance(mat, list):
            raise TypeError(f"{name}: expected list (rows), got {type(mat).__name__}")
        out: list[list[int]] = []
        width = None
        for r_i, row in enumerate(mat):
            if not isinstance(row, list):
                raise TypeError(f"{name}[{r_i}]: expected list, got {type(row).__name__}")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(f"{name}: ragged rows (row {r_i} len={len(row)}; expected {width})")
            out_row: list[int] = []
            for c_i, v in enumerate(row):
                if v is None:
                    raise ValueError(f"{name}[{r_i}][{c_i}] is None")
                if isinstance(v, float):
                    raise TypeError(f"{name}[{r_i}][{c_i}] must not be float")
                if isinstance(v, bool):
                    out_row.append(1 if v else 0)
                    continue
                try:
                    iv = int(v)
                except Exception:
                    raise TypeError(f"{name}[{r_i}][{c_i}] not int-like: {type(v).__name__}")
                out_row.append(iv & 1)
            out.append(out_row)
        return out


if "_inv__col_as_bits" not in globals():
    def _inv__col_as_bits(A: list[list[int]], j: int) -> list[int]:
        return [int(A[i][j]) & 1 for i in range(len(A))]


if "_inv__vec_to_bitstring" not in globals():
    def _inv__vec_to_bitstring(v_bits: list[int]) -> str:
        # Brick-2 Freeze: v[0] is the leftmost bit.
        return "".join("1" if (int(b) & 1) else "0" for b in (v_bits or []))


if "_inv__vec_to_intmask" not in globals():
    def _inv__vec_to_intmask(v_bits: list[int]) -> int:
        """Bitmask encoding for GF(2) vectors.

        Convention: bit i corresponds to v[i]. This is internal; lexical
        ordering is always computed on the string via _inv__vec_to_bitstring.
        """
        m = 0
        for i, b in enumerate(v_bits or []):
            if int(b) & 1:
                m |= (1 << i)
        return int(m)


if "_inv__intmask_to_bitstring" not in globals():
    def _inv__intmask_to_bitstring(x: int, m: int) -> str:
        return "".join("1" if ((int(x) >> i) & 1) else "0" for i in range(int(m)))


if "_inv__dot_mask" not in globals():
    def _inv__dot_mask(a: int, b: int) -> int:
        return _inv__popcount(int(a) & int(b)) & 1


if "_inv__col_types" not in globals():
    def _inv__col_types(A: list[list[int]], *, include_zero: bool = False) -> set[str]:
        """Return the set of distinct column-type bitstrings.

        Brick-1 Freeze: Null type is universal; callers may exclude it.
        """
        m, n = _inv__shape(A)
        if m == 0 or n == 0:
            # Even for 0×n, we treat dims as identity; type set can only include the empty-string.
            # Null type is the only possible column type.
            return {""} if include_zero else set()
        types: set[str] = set()
        for j in range(n):
            col = _inv__col_as_bits(A, j)
            bs = _inv__vec_to_bitstring(col)
            if (not include_zero) and all(b == 0 for b in col):
                continue
            types.add(bs)
        return types


if "_inv_stepA_type_gate" not in globals():
    def _inv_stepA_type_gate(A0: list[list[int]], A1: list[list[int]]) -> dict:
        """Step-A TypeGate (structural, directed A0→A1).

        PASS iff all hold:
          - no NEW nonzero column types appear
          - no MISSING nonzero column types (no deletions)
          - no DECREASED multiplicities for any nonzero type
          - n does not decrease (column-monotone refinement)

        Null (all-zero) type is universally permitted and excluded from type comparisons.
        """
        try:
            A0n = _inv__norm_bitmatrix(A0, name="A0")
            A1n = _inv__norm_bitmatrix(A1, name="A1")
        except Exception as e:
            return {"status": "NA", "reason": f"BAD_MATRIX: {e}"}

        m0, n0 = _inv__shape(A0n)
        m1, n1 = _inv__shape(A1n)
        if m0 != m1:
            return {"status": "NA", "reason": f"ROW_MISMATCH: m0={m0}, m1={m1}", "m0": m0, "m1": m1}

        # Build nonzero type sets + multiplicity counts (bitstring -> count).
        def _counts(A: list[list[int]]) -> dict[str, int]:
            m, n = _inv__shape(A)
            out: dict[str, int] = {}
            if m == 0 or n == 0:
                return out
            for j in range(n):
                col = _inv__col_as_bits(A, j)
                if all(b == 0 for b in col):
                    continue  # exclude null type
                bs = _inv__vec_to_bitstring(col)
                out[bs] = out.get(bs, 0) + 1
            return out

        c0 = _counts(A0n)
        c1 = _counts(A1n)
        t0 = set(c0.keys())
        t1 = set(c1.keys())

        new_types = sorted(list(t1.difference(t0)))
        missing_types = sorted(list(t0.difference(t1)))

        decreased_types: list[str] = []
        for bs, k0 in c0.items():
            k1 = c1.get(bs, 0)
            if k1 < k0:
                decreased_types.append(bs)
        decreased_types.sort()

        n_monotone = bool(n1 >= n0)

        ok = (not new_types) and (not missing_types) and (not decreased_types) and n_monotone
        return {
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


if "_inv_stepB_out_of_regime_record" not in globals():
    def _inv_stepB_out_of_regime_record(
        A0: list[list[int]],
        A1: list[list[int]],
        *,
        probe_id: str | None = None,
        probe_params: dict | None = None,
    ) -> dict:
        """Step-B: emit a first-class Step-A admissibility classification.

        Frozen semantic surface (Phase 2 / Step-B):
          - regime ∈ {STEP_A_ADMISSIBLE, NOT_ADMISSIBLE_FOR_STEP_A}
          - new_type_delta_nonempty + new_nonzero_types (evidence)

        Everything else is payload (carried for audit/debug only).
        """
        tg = _inv_stepA_type_gate(A0, A1)
        status = (tg or {}).get("status") if isinstance(tg, dict) else None
        regime = "STEP_A_ADMISSIBLE" if status == "PASS" else "NOT_ADMISSIBLE_FOR_STEP_A"

        # --- Frozen evidence: new nonzero types (delta) ---
        new_types = []
        if isinstance(tg, dict):
            nt = tg.get("new_nonzero_types")
            if isinstance(nt, list):
                new_types = [str(x) for x in nt]

        new_type_delta_nonempty = bool(new_types)

        # --- Payload: classification reasons (non-projected; for audit/debug only) ---
        reasons: list[str] = []
        if isinstance(tg, dict):
            try:
                if isinstance(tg.get("new_nonzero_types"), list) and len(tg.get("new_nonzero_types") or []) > 0:
                    reasons.append("NEW_NONZERO_TYPES")
                if isinstance(tg.get("missing_nonzero_types"), list) and len(tg.get("missing_nonzero_types") or []) > 0:
                    reasons.append("MISSING_NONZERO_TYPES")
                if isinstance(tg.get("decreased_nonzero_types"), list) and len(tg.get("decreased_nonzero_types") or []) > 0:
                    reasons.append("DECREASED_NONZERO_TYPES")
                if tg.get("n_monotone") is False:
                    reasons.append("N_DECREASES")
            except Exception:
                reasons = reasons

        # Expected-fail taxonomy (payload): only asserted for probes.
        probe_taxonomy = None
        if probe_id:
            probe_taxonomy = {
                "expected_fail_when_novelty_injected": bool(new_type_delta_nonempty),
                "classification_reasons": list(reasons),
                "outside_stepA": bool(regime != "STEP_A_ADMISSIBLE"),
            }

        # Best-effort dims (payload)
        dims = None
        try:
            A0n = _inv__norm_bitmatrix(A0, name="A0")
            A1n = _inv__norm_bitmatrix(A1, name="A1")
            dims = {
                "A0": {"m": _inv__shape(A0n)[0], "n": _inv__shape(A0n)[1]},
                "A1": {"m": _inv__shape(A1n)[0], "n": _inv__shape(A1n)[1]},
            }
        except Exception:
            dims = None

        rec = {
            "schema": _INV_STEP_B_RECORD_SCHEMA,
            "schema_version": _INV_STEP_B_RECORD_SCHEMA_VERSION,
            "profile_id": _INV_STEP_B_PROFILE_ID,
            "type_gate_id": _INV_TYPE_GATE_ID,
            "regime": regime,
            "stepA_admissible": bool(regime == "STEP_A_ADMISSIBLE"),
            "new_type_delta_nonempty": bool(new_type_delta_nonempty),
            "new_nonzero_types": list(new_types),
            "probe_id": str(probe_id) if probe_id else None,
            # Derived expectation for probes: do NOT hardcode by probe name.
            "probe_expected_outside_stepA": (bool(new_type_delta_nonempty) if probe_id else None),
            "probe_params": dict(probe_params or {}) if probe_id else None,
            # payload / audit
            "dims": dims,
            "type_gate": tg,
            "classification_reasons": list(reasons),
            "probe_taxonomy": probe_taxonomy,
        }
        return rec


if "_inv__append_col" not in globals():
    def _inv__append_col(A: list[list[int]], col: list[int]) -> list[list[int]]:
        """Return a new matrix with col appended as the last column."""
        A = _inv__norm_bitmatrix(A, name="A")
        m, n = _inv__shape(A)
        if m == 0:
            return []
        if len(col) != m:
            raise ValueError(f"append_col: col has len={len(col)}; expected m={m}")
        out = [list(row) for row in A]
        for i in range(m):
            out[i].append(int(col[i]) & 1)
        return out


if "_inv__col_from_row_support" not in globals():
    def _inv__col_from_row_support(m: int, ones: list[int]) -> list[int]:
        """Return a length-m column vector with 1s at the given row indices."""
        mm = int(m)
        v = [0 for _ in range(mm)]
        for idx in ones or []:
            i = int(idx)
            if i < 0 or i >= mm:
                raise ValueError(f"row index {i} out of range [0,{mm})")
            v[i] = 1
        return v


if "_inv_stepB_probe_xor_injection" not in globals():
    def _inv_stepB_probe_xor_injection(A0: list[list[int]], *, i: int, j: int) -> list[list[int]]:
        """Probe: XOR-injection.

        A1 := A0 with an extra column v_new = col_i(A0) XOR col_j(A0).

        Whether this is out-of-regime is determined by the Step-A TypeGate
        (i.e. by whether v_new introduces a new nonzero column type).
        """
        A0n = _inv__norm_bitmatrix(A0, name="A0")
        m, n = _inv__shape(A0n)
        if n <= 0:
            raise ValueError("xor_injection: A0 has no columns")
        ii = int(i)
        jj = int(j)
        if ii < 0 or ii >= n or jj < 0 or jj >= n:
            raise ValueError(f"xor_injection: column indices out of range (n={n})")
        ci = _inv__col_as_bits(A0n, ii)
        cj = _inv__col_as_bits(A0n, jj)
        v = [(int(ci[r]) ^ int(cj[r])) & 1 for r in range(m)]
        return _inv__append_col(A0n, v)


if "_inv_stepB_probe_local_chord_flip" not in globals():
    def _inv_stepB_probe_local_chord_flip(
        A0: list[list[int]],
        *,
        a: int,
        b: int,
        c: int,
        d: int,
    ) -> list[list[int]]:
        """Probe: local chord flip.

        Template: attempt to replace one instance of the chord (a,c) with (b,d).
        If (a,c) is not present as an exact 2-one column, we fall back to
        appending (b,d).

        This is a probe, not an admissible Step-A move; classification is via TypeGate.
        """
        A0n = _inv__norm_bitmatrix(A0, name="A0")
        m, n = _inv__shape(A0n)
        if m <= 0:
            raise ValueError("local_chord_flip: A0 has zero rows")

        aa, bb, cc, dd = int(a), int(b), int(c), int(d)
        old_col = _inv__col_from_row_support(m, [aa, cc])
        new_col = _inv__col_from_row_support(m, [bb, dd])

        # Search for an exact match to old_col.
        replace_j = None
        for j in range(n):
            if _inv__col_as_bits(A0n, j) == old_col:
                replace_j = j
                break

        out = [list(row) for row in A0n]
        if replace_j is None:
            # Fallback: append new chord.
            return _inv__append_col(out, new_col)

        # Replace in-place.
        for r in range(m):
            out[r][replace_j] = int(new_col[r]) & 1
        return out


if "_inv_stepB_probe_log_path" not in globals():
    def _inv_stepB_probe_log_path() -> _Path:
        """Path for Step-B probe logs (observational; does not drive resolution)."""
        root = _repo_root()
        p = _Path(root) / "logs" / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p / "stepb_boundary_probes.jsonl"


if "_inv_stepB_probe_log_append" not in globals():
    def _inv_stepB_probe_log_append(row: dict):
        """Append a Step-B probe record (best-effort JSONL)."""
        import time as _time

        rec = dict(row or {})
        rec.setdefault("ts_utc", int(_time.time()))
        try:
            line = canonical_json(rec)
        except Exception:
            line = _json.dumps(rec, separators=(",", ":"), sort_keys=False)
        with _inv_stepB_probe_log_path().open("a", encoding="utf-8") as f:
            f.write(line + "\n")



if "_inv__gf2_rref" not in globals():
    def _inv__gf2_rref(rows: list[int], n_vars: int) -> tuple[list[int], list[int]]:
        """Reduced row-echelon form over GF(2) for bitmask rows.

        Returns (rref_rows, pivot_cols).
        """
        rr = [int(r) for r in (rows or [])]
        pivots: list[int] = []
        r = 0
        n_rows = len(rr)
        for c in range(int(n_vars)):
            if r >= n_rows:
                break
            # find pivot row
            piv = None
            for i in range(r, n_rows):
                if (rr[i] >> c) & 1:
                    piv = i
                    break
            if piv is None:
                continue
            # swap
            if piv != r:
                rr[r], rr[piv] = rr[piv], rr[r]
            # eliminate all other rows in column c
            for i in range(n_rows):
                if i != r and ((rr[i] >> c) & 1):
                    rr[i] ^= rr[r]
            pivots.append(int(c))
            r += 1
        return rr, pivots


if "_inv__gf2_nullspace_basis" not in globals():
    def _inv__gf2_nullspace_basis(eq_rows: list[int], n_vars: int) -> list[int]:
        """Return a GF(2) nullspace basis for the homogeneous system eq_rows * x = 0.

        - eq_rows: list of bitmasks, each mask over n_vars variables
        - n_vars: number of variables

        Output: list of basis vectors as bitmasks length n_vars.
        """
        rr, pivots = _inv__gf2_rref(eq_rows, n_vars)
        pivot_set = set(pivots)
        free_cols = [c for c in range(int(n_vars)) if c not in pivot_set]

        # Map pivot col -> row mask
        pivot_row_by_col: dict[int, int] = {}
        # In rref construction above, pivot i corresponds to row i.
        for i, c in enumerate(pivots):
            if i < len(rr):
                pivot_row_by_col[int(c)] = int(rr[i])

        basis: list[int] = []
        for f in free_cols:
            x = (1 << int(f))
            for p in pivots:
                row = pivot_row_by_col.get(int(p), 0)
                # rhs = dot(row_without_p, x)
                rhs = _inv__popcount(row & x) & 1
                # Because row includes pivot bit, but x_p currently 0, rhs computed this way is fine.
                # Solve: x_p = rhs
                if rhs:
                    x |= (1 << int(p))
                else:
                    x &= ~(1 << int(p))
            if x != 0:
                basis.append(int(x))
        return basis


if "_inv_stepA_cert" not in globals():
    def _inv_stepA_cert(A: list[list[int]]) -> dict:
        """Compute the Step-A certificate triple (y*, t*, A_comp) if defined.

        This follows the session's Brick-4/5 freeze:
          - y* ∈ ker(A^T), y* != 0; chosen by min Hamming weight, then lex on bitstring.
          - t* ∉ im(A); chosen by min weight then lex (implemented via ker(A^T) witness).
          - A_comp = { columns c with y*·c = 1 }, parity-deduped (mod 2), then sorted.

        Returns:
          {"defined": bool, "reason": <optional>, "m":..., "n":..., "y": <bitstring>|None,
           "t": <bitstring>|None, "A_comp": [bitstring,...], "kernel_dim": int}
        """
        A = _inv__norm_bitmatrix(A, name="A")
        m, n = _inv__shape(A)

        # Build A^T equations: each column is an equation dot(col_j, y)=0.
        eq_rows: list[int] = []
        if m == 0:
            # 0-row matrix: y lives in F2^0; there is no nonzero y.
            return {
                "defined": False,
                "reason": "M_ZERO_ROWS",
                "m": int(m),
                "n": int(n),
                "y": None,
                "t": None,
                "A_comp": [],
                "kernel_dim": 0,
            }
        if n == 0:
            # No equations: ker(A^T)=F2^m. This is huge; we treat y* as the min-weight/lex vector.
            # Min weight is 1; lex-min among weight-1 is at max index.
            y_mask = 1 << (m - 1)
            y_bs = _inv__intmask_to_bitstring(y_mask, m)
            # t*: im(A) = {0}; so any nonzero t works; choose min weight/lex => weight 1, lex-min => max index.
            t_mask = 1 << (m - 1)
            t_bs = _inv__intmask_to_bitstring(t_mask, m)
            return {
                "defined": True,
                "reason": None,
                "m": int(m),
                "n": int(n),
                "y": y_bs,
                "t": t_bs,
                "A_comp": [],
                "kernel_dim": int(m),
            }

        for j in range(n):
            col_bits = _inv__col_as_bits(A, j)
            eq_rows.append(_inv__vec_to_intmask(col_bits))

        basis = _inv__gf2_nullspace_basis(eq_rows, m)
        kdim = int(len(basis))
        if kdim == 0:
            return {
                "defined": False,
                "reason": "KERNEL_TRIVIAL",
                "m": int(m),
                "n": int(n),
                "y": None,
                "t": None,
                "A_comp": [],
                "kernel_dim": 0,
            }

        # Choose y*: min weight then lex among all nonzero kernel vectors.
        if kdim > _INV_FPV1_MAX_KERNEL_ENUM_DIM:
            return {
                "defined": False,
                "reason": f"KERNEL_DIM_TOO_LARGE(k={kdim},max={_INV_FPV1_MAX_KERNEL_ENUM_DIM})",
                "m": int(m),
                "n": int(n),
                "y": None,
                "t": None,
                "A_comp": [],
                "kernel_dim": kdim,
            }

        best_mask = None
        best_key = None
        # enumerate all nonzero combinations of basis vectors
        # NOTE: basis vectors are length m bitmasks.
        for comb in range(1, 1 << kdim):
            v = 0
            for i in range(kdim):
                if (comb >> i) & 1:
                    v ^= int(basis[i])
            if v == 0:
                continue
            w = _inv__popcount(v)
            bs = _inv__intmask_to_bitstring(v, m)
            key = (int(w), bs)
            if best_key is None or key < best_key:
                best_key = key
                best_mask = int(v)
                # Early exit: you cannot beat weight 1.
                if w == 1:
                    # But lex tie-breaking among weight-1 could still improve.
                    # We cannot early-exit safely without scanning remaining weight-1 candidates.
                    pass

        if best_mask is None:
            return {
                "defined": False,
                "reason": "KERNEL_ENUM_EMPTY",
                "m": int(m),
                "n": int(n),
                "y": None,
                "t": None,
                "A_comp": [],
                "kernel_dim": kdim,
            }

        y_mask = int(best_mask)
        y_bs = _inv__intmask_to_bitstring(y_mask, m)

        # Choose t*: min weight then lex among vectors not in im(A).
        # Using orthogonality: t ∈ im(A) iff dot(t, y)=0 for all y ∈ ker(A^T).
        # Min weight is 1 provided ker has any support bit.
        support_mask = 0
        for v in basis:
            support_mask |= int(v)
        if support_mask == 0:
            # Should not happen for nontrivial kernel, but keep defensive.
            return {
                "defined": False,
                "reason": "KERNEL_SUPPORT_EMPTY",
                "m": int(m),
                "n": int(n),
                "y": None,
                "t": None,
                "A_comp": [],
                "kernel_dim": kdim,
            }
        # weight-1 candidates correspond to unit vectors e_i. Lex-min among weight-1 is max index.
        max_i = None
        for i in range(m - 1, -1, -1):
            if (support_mask >> i) & 1:
                max_i = int(i)
                break
        if max_i is None:
            return {
                "defined": False,
                "reason": "NO_SUPPORT_INDEX",
                "m": int(m),
                "n": int(n),
                "y": None,
                "t": None,
                "A_comp": [],
                "kernel_dim": kdim,
            }
        t_mask = 1 << int(max_i)
        t_bs = _inv__intmask_to_bitstring(t_mask, m)

        # Build A_comp: columns with y·c = 1. Parity-dedup (mod 2 multiplicity).
        comp_counts: dict[str, int] = {}
        for j in range(n):
            c_bits = _inv__col_as_bits(A, j)
            c_mask = _inv__vec_to_intmask(c_bits)
            if _inv__dot_mask(y_mask, c_mask) == 1:
                bs = _inv__intmask_to_bitstring(c_mask, m)
                comp_counts[bs] = (comp_counts.get(bs, 0) + 1) & 1
        A_comp = sorted([bs for bs, par in comp_counts.items() if par == 1])

        return {
            "defined": True,
            "reason": None,
            "m": int(m),
            "n": int(n),
            "y": y_bs,
            "t": t_bs,
            "A_comp": A_comp,
            "kernel_dim": kdim,
        }


if "_inv_fpv1_payload" not in globals():
    def _inv_fpv1_payload(cert: dict) -> dict | None:
        """Build the fp.v1 *hash core* payload (ordered dict).

        Brick-4 Freeze (fp.v1):
          - hash core keys are exactly: field, lex_order, y, t, A_comp
          - key order is fixed by insertion
          - schema/provenance is NOT part of the hash surface
        """
        y = cert.get("y")
        t = cert.get("t")
        A_comp = cert.get("A_comp")
        if not (
            cert.get("defined") is True
            and isinstance(y, str)
            and isinstance(t, str)
            and isinstance(A_comp, list)
        ):
            return None

        # Canonical key order (do not sort).
        payload = {
            "field": "F2",
            "lex_order": "left-to-right",
            "y": y,
            "t": t,
            "A_comp": list(A_comp),
        }
        return payload



if "_inv_fpv1_sha256" not in globals():
    def _inv_fpv1_sha256(cert: dict) -> dict:
        """Compute fp.v1 SHA-256 over CanonicalJSON(core_payload).

        Returns:
          {
            "defined": bool,
            "sha256": <64-hex lower> | None,
            "canonical_json": <canonical json> | None,
            "payload_core": <dict> | None,
            "meta": {"schema":..., "schema_version":..., "profile_id":...}
          }

        NOTE: Hash is NOT computed over file bytes; it is computed over canonical re-serialization.
        """
        core = _inv_fpv1_payload(cert)
        meta = {
            "schema": "fp_v1",
            "schema_version": "fp.v1",
            "profile_id": _INV_FPV1_PROFILE_ID,
        }
        if core is None:
            return {
                "defined": False,
                "sha256": None,
                "canonical_json": None,
                "payload_core": None,
                "meta": meta,
                "reason": (cert.get("reason") or "UNDEFINED"),
            }

        # Canonical JSON: compact separators, ASCII, fixed insertion key order.
        txt = _json.dumps(core, separators=(",", ":"), sort_keys=False, ensure_ascii=True, allow_nan=False)
        h = _hash.sha256(txt.encode("utf-8")).hexdigest()
        return {
            "defined": True,
            "sha256": h,
            "canonical_json": txt,
            "payload_core": core,
            "meta": meta,
        }



if "_inv_stepA_link_gate" not in globals():
    def _inv_stepA_link_gate(A0: list[list[int]], A1: list[list[int]]) -> dict:
        """Compute the Step-A link gates (TypeGate + CertGate) for two matrices.

        NOTE: This does *not* assert that A0→A1 is actually a Step-A admissible move.
        It only evaluates the *frozen* gate predicates on the chosen pair.
        """
        tg = _inv_stepA_type_gate(A0, A1)
        # CertGate compares fp.v1 hashes when both certificates are defined.
        try:
            c0 = _inv_stepA_cert(A0)
            c1 = _inv_stepA_cert(A1)
            f0 = _inv_fpv1_sha256(c0)
            f1 = _inv_fpv1_sha256(c1)
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

        return {
            "type_gate": tg,
            "cert_gate": cg,
        }


if "_inv_barcode_from_mats" not in globals():
    def _inv_barcode_from_mats(mats: list[tuple[str, list[list[int]]]], *, barcode_kind: str = "full") -> dict:
        """Compute a barcode sequence over a list of (name, matrix) pairs.

        barcode_kind:
          - "full": fp.v1 sha256 per level (UNDEFINED if cert undefined)
          - "component": sha256 over the A_comp list per level (UNDEFINED if cert undefined)
        """
        kind = str(barcode_kind or "full")
        seq = []
        for name, A in mats:
            try:
                cert = _inv_stepA_cert(A)
                fp = _inv_fpv1_sha256(cert)
                if not fp.get("defined"):
                    seq.append({"level": name, "status": "UNDEFINED", "reason": cert.get("reason")})
                    continue
                if kind == "component":
                    # Component barcode: SHA-256 over CanonicalJSON({"A_comp": A_comp}).
                    # Use the *certificate* A_comp list (already parity-reduced + lex-sorted).
                    comp = cert.get("A_comp") if isinstance(cert, dict) else None
                    if not isinstance(comp, list):
                        comp = []
                    comp_txt = _json.dumps({"A_comp": list(comp)}, separators=(",", ":"), sort_keys=False, ensure_ascii=True, allow_nan=False)
                    comp_h = _hash.sha256(comp_txt.encode("utf-8")).hexdigest()
                    seq.append({"level": name, "status": "OK", "hash": comp_h, "kind": "component"})
                else:
                    seq.append({"level": name, "status": "OK", "hash": fp.get("sha256"), "kind": "full"})
            except Exception as e:
                seq.append({"level": name, "status": "UNDEFINED", "reason": f"ERROR: {e}"})
        return {
            "profile_id": _INV_BARCODE_PROFILE_ID,
            "barcode_kind": kind,
            "L": int(len(seq) - 1) if seq else 0,
            "sequence": seq,
            "equality": "strict sequence equality (same length + per-level hash equality)",
        }



# =================== Phase 3 / Step C — Glue & Towers as explicit consumers of Step A ===================
# Mode B wiring: minimal, projection-first helpers to compute:
#   - Codim-1 interface Cancel/Persist (via solvability over F2)
#   - GlueRecord discipline (phi_exists, new_type, decision, stepA_applies)
#   - Step-A invocation discipline (invoked only when no new type holds)
#   - TowerBarcode (component strings across levels) + constancy reporting
#
# NOTE: These helpers are observational; they do not alter the v2 overlap pipeline unless called.

_INV_STEP_C_PROFILE_ID = "Phase3.StepC.GlueAndTowers.v1"

_INV_STEP_C_GLUE_RECORD_SCHEMA = "stepc_glue_record"
_INV_STEP_C_GLUE_RECORD_SCHEMA_VERSION = "stepc.glue_record.v1"

_INV_STEP_C_TOWER_BARCODE_SCHEMA = "stepc_tower_barcode"
_INV_STEP_C_TOWER_BARCODE_SCHEMA_VERSION = "stepc.tower_barcode.v1"

_INV_PHASE3_REPORT_SCHEMA = "phase3_report"
_INV_PHASE3_REPORT_SCHEMA_VERSION = "phase3.report.v1"


if "_stepC_norm_bitvector" not in globals():
    def _stepC_norm_bitvector(vec, *, name: str = "u", m_expected: int | None = None) -> list[int]:
        """Normalize to list[int] over F2 (0/1).

        - Accepts [] and returns []
        - Coerces bool/int-like; rejects floats/None
        - Optionally enforces expected length
        """
        if vec is None:
            out: list[int] = []
        else:
            if not isinstance(vec, list):
                raise TypeError(f"{name}: expected list, got {type(vec).__name__}")
            out = []
            for i, v in enumerate(vec):
                if v is None:
                    raise ValueError(f"{name}[{i}] is None")
                if isinstance(v, float):
                    raise TypeError(f"{name}[{i}] must not be float")
                if isinstance(v, bool):
                    out.append(1 if v else 0)
                    continue
                try:
                    iv = int(v)
                except Exception:
                    raise TypeError(f"{name}[{i}] not int-like: {type(v).__name__}")
                out.append(iv & 1)

        if m_expected is not None and len(out) != int(m_expected):
            raise ValueError(f"{name}: len={len(out)}; expected {int(m_expected)}")
        return out


if "_stepC_interface_solvability" not in globals():
    def _stepC_interface_solvability(B_I, u_A, u_B) -> dict:
        """Codim-1 interface solvability over F2.

        Computes phi_exists for:
            B_I * phi = (u_A + u_B) over F2
        where + is XOR.

        Returns:
          {
            "phi_exists": bool,
            "rhs": <list[int]> (payload),
            "m": int,
            "r": int,
            "reason": <optional str>
          }

        NOTE: We only project existence; witness production is intentionally omitted.
        """
        BI = _inv__norm_bitmatrix(B_I, name="B_I")
        m, r = _inv__shape(BI)

        uA = _stepC_norm_bitvector(u_A, name="u_A", m_expected=m)
        uB = _stepC_norm_bitvector(u_B, name="u_B", m_expected=m)
        rhs = [(int(a) ^ int(b)) & 1 for a, b in zip(uA, uB)]

        # Empty row-space: always solvable if rhs has length 0 (already enforced by m_expected).
        if m == 0:
            return {"phi_exists": True, "rhs": rhs, "m": int(m), "r": int(r), "reason": None}

        # No variables: solvable iff rhs == 0.
        if r == 0:
            ok = all((int(b) & 1) == 0 for b in rhs)
            return {
                "phi_exists": bool(ok),
                "rhs": rhs,
                "m": int(m),
                "r": int(r),
                "reason": None if ok else "INCONSISTENT_NO_VARS",
            }

        # Build augmented row bitmasks: [coeff_bits | (rhs_bit<<r)]
        aug_rows: list[int] = []
        for i in range(m):
            mask = 0
            row = BI[i]
            for j in range(r):
                if int(row[j]) & 1:
                    mask |= (1 << int(j))
            mask |= ((int(rhs[i]) & 1) << int(r))
            aug_rows.append(int(mask))

        rr, _pivots = _inv__gf2_rref(aug_rows, int(r))

        # Inconsistent iff a row reduces to 0...0 | 1.
        coeff_mask_all = (1 << int(r)) - 1
        inconsistent = False
        for row in rr:
            coeff = int(row) & int(coeff_mask_all)
            rhs_bit = (int(row) >> int(r)) & 1
            if coeff == 0 and rhs_bit == 1:
                inconsistent = True
                break

        return {
            "phi_exists": bool(not inconsistent),
            "rhs": rhs,
            "m": int(m),
            "r": int(r),
            "reason": None if (not inconsistent) else "INCONSISTENT",
        }


if "_stepC_new_type_detection" not in globals():
    def _stepC_new_type_detection(A_before, A_after) -> dict:
        """Return Step-C new-type evidence (distinct nonzero column-type delta).

        new_type == True iff T(after) \ (T(before) ∪ {0}) is nonempty.
        Null (all-zero) type is excluded.

        Returns:
          {
            "new_type": bool,
            "new_nonzero_types": [bitstring,...] (payload evidence),
          }
        """
        A0 = _inv__norm_bitmatrix(A_before, name="A_before")
        A1 = _inv__norm_bitmatrix(A_after, name="A_after")
        m0, _n0 = _inv__shape(A0)
        m1, _n1 = _inv__shape(A1)
        if m0 != m1:
            raise ValueError(f"new_type_detection: ROW_MISMATCH m0={m0}, m1={m1}")

        t0 = _inv__col_types(A0, include_zero=False)
        t1 = _inv__col_types(A1, include_zero=False)
        new_types = sorted(list(set(t1).difference(set(t0))))
        return {"new_type": bool(new_types), "new_nonzero_types": list(new_types)}


if "_stepC_stepA_parity_check_adapter" not in globals():
    def _stepC_stepA_parity_check_adapter(
        *,
        stepA_applies: bool,
        A0,
        A1,
        glue_id: str | None = None,
        invocation_log: list | None = None,
    ) -> dict:
        """Step-C adapter: invoke Step-A check only when applicable.

        Frozen invariant:
          invoked ⇒ stepA_applies

        Returns:
          {"invoked": bool, "payload": <dict|None>}
        """
        if not bool(stepA_applies):
            return {"invoked": False, "payload": None}

        # Invocation metering (payload)
        if invocation_log is not None:
            try:
                invocation_log.append(
                    {
                        "event": "stepA_parity_check_invoked",
                        "glue_id": str(glue_id) if glue_id else None,
                        "stepA_applies": True,
                        "ts_utc": int(_time.time()),
                    }
                )
            except Exception:
                pass

        # Use the canonical Step-A gate computation as the check payload.
        # (Payload quarantine: Step-C does not interpret this result.)
        try:
            payload = _inv_stepA_link_gate(A0, A1)
        except Exception as e:
            payload = {"status": "UNDEFINED", "reason": f"STEP_A_CHECK_ERROR: {e}"}

        return {"invoked": True, "payload": payload}


if "_stepC_build_glue_record" not in globals():
    def _stepC_build_glue_record(
        *,
        B_I,
        u_A,
        u_B,
        A_before,
        A_after,
        glue_id: str | None = None,
        invoke_stepA_when_applicable: bool = True,
        invocation_log: list | None = None,
    ) -> dict:
        """Build the canonical Step-C GlueRecord (plus payload wrapper).

        Required projected fields (top-level GlueRecord):
          - phi_exists: bool
          - new_type: bool
          - decision: {"Cancel","Persist"}
          - stepA_applies: bool

        Returns wrapper:
          {
            "schema": ...,
            "schema_version": ...,
            "profile_id": ...,
            "glue_id": <payload>,
            "glue_record": <dict with 4 required keys>,
            "interface": <payload>,
            "new_type_evidence": <payload>,
            "stepA_check": <payload adapter result>,
          }
        """
        # --- Interface solvability (phi_exists) ---
        solv = _stepC_interface_solvability(B_I, u_A, u_B)
        phi_exists = bool(solv.get("phi_exists", False))

        # Derived-field discipline: decision is a function of phi_exists.
        decision = "Cancel" if phi_exists else "Persist"

        # --- New-type gate ---
        nt = _stepC_new_type_detection(A_before, A_after)
        new_type = bool(nt.get("new_type", False))

        # Derived-field discipline: Step-A applies iff no new type.
        stepA_applies = bool(not new_type)

        glue_record = {
            "phi_exists": bool(phi_exists),
            "new_type": bool(new_type),
            "decision": str(decision),
            "stepA_applies": bool(stepA_applies),
        }

        # --- Step-A adapter (payload only) ---
        if bool(invoke_stepA_when_applicable) and bool(stepA_applies):
            stepA_check = _stepC_stepA_parity_check_adapter(
                stepA_applies=True,
                A0=A_before,
                A1=A_after,
                glue_id=glue_id,
                invocation_log=invocation_log,
            )
        else:
            stepA_check = {"invoked": False, "payload": None}

        return {
            "schema": _INV_STEP_C_GLUE_RECORD_SCHEMA,
            "schema_version": _INV_STEP_C_GLUE_RECORD_SCHEMA_VERSION,
            "profile_id": _INV_STEP_C_PROFILE_ID,
            "glue_id": str(glue_id) if glue_id else None,  # payload
            "glue_record": glue_record,
            "interface": solv,              # payload (B_I/uA/uB not stored here)
            "new_type_evidence": nt,        # payload
            "stepA_check": stepA_check,     # payload
        }


if "_stepC_validate_glue_attempt" not in globals():
    def _stepC_validate_glue_attempt(attempt: dict) -> list[dict]:
        """Validate GlueRecord reproducibility for a single attempt.

        Returns a list of failure dicts (empty => pass).
        """
        fails: list[dict] = []

        rec = (attempt or {}).get("glue_record") or {}
        if not isinstance(rec, dict):
            return [{"code": "BAD_GLUE_RECORD", "msg": "glue_record not a dict"}]

        # Required keys
        for k in ("phi_exists", "new_type", "decision", "stepA_applies"):
            if k not in rec:
                fails.append({"code": "MISSING_KEY", "key": k})

        # Derived-field coherence checks (local)
        try:
            phi_exists = bool(rec.get("phi_exists"))
            dec = rec.get("decision")
            exp_dec = "Cancel" if phi_exists else "Persist"
            if dec != exp_dec:
                fails.append({"code": "DECISION_COHERENCE", "expected": exp_dec, "got": dec})
        except Exception as e:
            fails.append({"code": "DECISION_COHERENCE_ERROR", "msg": str(e)})

        try:
            new_type = bool(rec.get("new_type"))
            stepA_applies = bool(rec.get("stepA_applies"))
            exp = bool(not new_type)
            if stepA_applies != exp:
                fails.append({"code": "STEP_A_APPLIES_COHERENCE", "expected": exp, "got": stepA_applies})
        except Exception as e:
            fails.append({"code": "STEP_A_APPLIES_COHERENCE_ERROR", "msg": str(e)})

        # Recompute primitives from inputs if present
        try:
            if "B_I" in (attempt or {}) and "u_A" in (attempt or {}) and "u_B" in (attempt or {}):
                solv = _stepC_interface_solvability(attempt.get("B_I"), attempt.get("u_A"), attempt.get("u_B"))
                exp_phi = bool(solv.get("phi_exists"))
                if bool(rec.get("phi_exists")) != exp_phi:
                    fails.append({"code": "PHI_EXISTS_MISMATCH", "expected": exp_phi, "got": rec.get("phi_exists")})
        except Exception as e:
            fails.append({"code": "PHI_EXISTS_RECOMPUTE_ERROR", "msg": str(e)})

        try:
            if "A_before" in (attempt or {}) and "A_after" in (attempt or {}):
                nt = _stepC_new_type_detection(attempt.get("A_before"), attempt.get("A_after"))
                exp_nt = bool(nt.get("new_type"))
                if bool(rec.get("new_type")) != exp_nt:
                    fails.append({"code": "NEW_TYPE_MISMATCH", "expected": exp_nt, "got": rec.get("new_type")})
        except Exception as e:
            fails.append({"code": "NEW_TYPE_RECOMPUTE_ERROR", "msg": str(e)})

        return fails


if "_stepC_phase3_report" not in globals():
    def _stepC_phase3_report(
        *,
        glue_attempts: list[dict] | None,
        invocation_log: list | None,
    ) -> dict:
        """Phase-3 acceptance gate (Step C).

        Pass iff:
          - Glue decisions are reproducible (per-attempt validation passes)
          - Step-A checks are only invoked when no new type holds
            (invoked ⇒ stepA_applies)

        Returns:
          {"phase3_pass": bool, ...payload...}
        """
        attempts = list(glue_attempts or [])
        inv = list(invocation_log or [])
        failures: list[dict] = []

        # --- Glue reproducibility ---
        for idx, att in enumerate(attempts):
            fs = _stepC_validate_glue_attempt(att)
            for f in fs:
                ff = dict(f)
                ff.setdefault("glue_index", int(idx))
                gid = (att or {}).get("glue_id")
                if gid is not None:
                    ff.setdefault("glue_id", gid)
                failures.append(ff)

        # --- Invocation discipline: invoked ⇒ applicable ---
        # Use the invocation log (preferred), but also accept per-attempt payload markers if present.

        # 1) Per-attempt marker discipline: invoked ⇒ (no new type).
        for idx, att in enumerate(attempts):
            try:
                if not isinstance(att, dict):
                    continue
                rec = att.get("glue_record")
                if not isinstance(rec, dict):
                    continue
                stepA_check = att.get("stepA_check")
                if not (isinstance(stepA_check, dict) and stepA_check.get("invoked") is True):
                    continue
                if bool(rec.get("new_type")):
                    failures.append(
                        {
                            "code": "STEP_A_INVOKED_OUT_OF_SCOPE",
                            "glue_index": int(idx),
                            "glue_id": att.get("glue_id"),
                            "new_type": rec.get("new_type"),
                            "stepA_check": stepA_check,
                        }
                    )
            except Exception:
                continue

        # 2) Invocation-log discipline: invoked ⇒ (no new type) when we can bind to a glue attempt.
        gid_to_new_type: dict[str, bool] = {}
        for att in attempts:
            try:
                if not isinstance(att, dict):
                    continue
                gid = att.get("glue_id")
                rec = att.get("glue_record")
                if gid is None or not isinstance(rec, dict):
                    continue
                gid_to_new_type[str(gid)] = bool(rec.get("new_type"))
            except Exception:
                continue

        for ev in inv:
            try:
                if not isinstance(ev, dict):
                    continue
                if ev.get("event") != "stepA_parity_check_invoked":
                    continue

                gid = ev.get("glue_id")
                if gid is not None and str(gid) in gid_to_new_type:
                    if gid_to_new_type.get(str(gid)) is True:
                        failures.append({"code": "STEP_A_INVOKED_OUT_OF_SCOPE", "event": ev, "new_type": True})
                else:
                    # Fallback: trust the event's own applicability flag when no glue binding exists.
                    if ev.get("stepA_applies") is not True:
                        failures.append({"code": "STEP_A_INVOKED_OUT_OF_SCOPE", "event": ev})
            except Exception:
                continue

        phase3_pass = bool(len(failures) == 0)
        return {
            "schema": _INV_PHASE3_REPORT_SCHEMA,
            "schema_version": _INV_PHASE3_REPORT_SCHEMA_VERSION,
            "profile_id": _INV_STEP_C_PROFILE_ID,
            "phase3_pass": bool(phase3_pass),
            "n_glues": int(len(attempts)),
            "n_invocations": int(len([e for e in inv if isinstance(e, dict) and e.get("event") == "stepA_parity_check_invoked"])),
            "failures": failures,  # payload
        }


if "_stepC_tower_barcode" not in globals():
    def _stepC_tower_barcode(
        mats: list[tuple[str, list[list[int]]]],
    ) -> dict:
        """Build Step-C TowerBarcode: component strings per level + constancy report.

        Required projected fields:
          - levels: [0..L] inclusive
          - component_string_by_level: { "<k>": "<string>", ... }

        Payload:
          - is_stepA_tower (TypeGate PASS on each adjacent link)
          - link_typegates
          - constant_across_levels (only asserted when is_stepA_tower==True)
        """
        seq = list(mats or [])
        L = max(len(seq) - 1, 0)

        # Component string per level: canonical JSON of {"A_comp": [...]}
        component_by_level: dict[str, str] = {}
        level_names: list[str] = []
        for k, (nm, A) in enumerate(seq):
            level_names.append(str(nm))
            try:
                cert = _inv_stepA_cert(A)
                comp = cert.get("A_comp") if isinstance(cert, dict) else None
                if not isinstance(comp, list):
                    comp = []
                comp_txt = _json.dumps({"A_comp": list(comp)}, separators=(",", ":"), sort_keys=False, ensure_ascii=True, allow_nan=False)
                component_by_level[str(int(k))] = str(comp_txt)
            except Exception as e:
                component_by_level[str(int(k))] = _json.dumps(
                    {"A_comp": [], "error": str(e)},
                    separators=(",", ":"),
                    sort_keys=False,
                    ensure_ascii=True,
                    allow_nan=False,
                )

        # Step-A tower predicate (best-effort): TypeGate PASS on each link (directed).
        link_typegates: list[dict] = []
        is_stepA = True
        for k in range(len(seq) - 1):
            try:
                A0 = seq[k][1]
                A1 = seq[k + 1][1]
                tg = _inv_stepA_type_gate(A0, A1)
            except Exception as e:
                tg = {"status": "NA", "reason": str(e)}
            link_typegates.append({"k": int(k), "from": seq[k][0], "to": seq[k + 1][0], "type_gate": tg})
            if not (isinstance(tg, dict) and tg.get("status") == "PASS"):
                is_stepA = False

        # Constancy (only required for Step-A towers)
        constant = None
        first_div = None
        if is_stepA and seq:
            vals = [component_by_level.get(str(i), "") for i in range(len(seq))]
            constant = bool(all(v == vals[0] for v in vals))
            if not constant:
                # find first divergence
                base = vals[0]
                for i, v in enumerate(vals):
                    if v != base:
                        first_div = int(i)
                        break

        return {
            "schema": _INV_STEP_C_TOWER_BARCODE_SCHEMA,
            "schema_version": _INV_STEP_C_TOWER_BARCODE_SCHEMA_VERSION,
            "profile_id": _INV_STEP_C_PROFILE_ID,
            "levels": list(range(len(seq))),
            "component_string_by_level": component_by_level,
            # payload
            "level_names": level_names,
            "is_stepA_tower": bool(is_stepA),
            "link_typegates": link_typegates,
            "constant_across_levels_when_stepA": constant,
            "first_divergence_level": first_div,
        }


if "_inv_scan_last_bundle" not in globals():
    def _inv_scan_last_bundle(bundle_dir: str | None) -> dict:
        """Best-effort scanner for the most recent bundle dir artifacts.

        This is *observational*: it never writes or mutates artifacts.
        """
        out: dict = {
            "bundle_dir": bundle_dir,
            "status": "NA",
            "errors": [],
            "found": {},
        }
        if not bundle_dir:
            out["errors"].append("NO_BUNDLE_DIR")
            return out
        try:
            bdir = _Path(str(bundle_dir))
        except Exception:
            out["errors"].append("BAD_BUNDLE_DIR")
            return out
        if not bdir.exists():
            out["errors"].append("BUNDLE_DIR_MISSING")
            return out

        # Locate common artifacts.
        try:
            out["found"]["bundle_index"] = str((bdir / "bundle_index.v2.json")) if (bdir / "bundle_index.v2.json").exists() else None
            out["found"]["bundle_json"]  = str((bdir / "bundle.json")) if (bdir / "bundle.json").exists() else None

            b5s = sorted([p for p in bdir.glob("b5_identity__*.json") if p.is_file()])
            out["found"]["b5_identity"] = str(b5s[0]) if b5s else None

            loop = sorted([p for p in bdir.glob("loop_receipt__*.json") if p.is_file()])
            out["found"]["loop_receipt"] = str(loop[0]) if loop else None

            meta_dir = bdir / "meta"
            out["found"]["b6_seal"] = str(meta_dir / "bundle_hash.json") if (meta_dir / "bundle_hash.json").exists() else None
            out["found"]["b6_verify"] = str(meta_dir / "b6_verify_receipt.json") if (meta_dir / "b6_verify_receipt.json").exists() else None
        except Exception as e:
            out["errors"].append(f"SCAN_ERROR: {e}")
            return out

        # Read + summarize B5 identity.
        b5_path = out["found"].get("b5_identity")
        if b5_path:
            try:
                b5 = _json.loads(_Path(b5_path).read_text(encoding="utf-8"))
            except Exception as e:
                out["errors"].append(f"B5_READ_ERROR: {e}")
            else:
                b5_sum = {
                    "schema": b5.get("schema"),
                    "schema_version": b5.get("schema_version"),
                    "b5_fp_sig8": b5.get("b5_fp_sig8"),
                    "b5_fp_hex": (b5.get("b5_fp_hex") or "")[:16] + "…" if b5.get("b5_fp_hex") else None,
                }
                try:
                    core = b5.get("core") or {}
                    dims = (core.get("dims") or {}) if isinstance(core, dict) else {}
                    b5_sum["dims"] = dims
                except Exception:
                    pass
                # Verify payload_sig8 if present.
                try:
                    ps = b5.get("payload_sig8")
                    # Recompute using the same non-core key dropper used in the builder.
                    nc = set(_B3_NON_CORE_COMMON_KEYS) | {"annex"}
                    exp_sig8 = b3_payload_sig8(b5, non_core_keys=nc)
                    b5_sum["payload_sig8"] = ps
                    b5_sum["payload_sig8_expected"] = exp_sig8
                    b5_sum["payload_sig8_ok"] = bool(ps == exp_sig8)
                except Exception:
                    pass
                out["b5_identity"] = b5_sum

        # Read + summarize B6 verify receipt if present.
        b6v_path = out["found"].get("b6_verify")
        if b6v_path:
            try:
                b6v = _json.loads(_Path(b6v_path).read_text(encoding="utf-8"))
                out["b6_verify"] = {
                    "schema": b6v.get("schema"),
                    "schema_version": b6v.get("schema_version"),
                    "status": b6v.get("status"),
                    "verdict": b6v.get("verdict"),
                    "fail_codes": b6v.get("fail_codes"),
                }
            except Exception as e:
                out["errors"].append(f"B6_VERIFY_READ_ERROR: {e}")

        # Read + summarize B6 seal if present.
        b6s_path = out["found"].get("b6_seal")
        if b6s_path:
            try:
                b6s = _json.loads(_Path(b6s_path).read_text(encoding="utf-8"))
                out["b6_seal"] = {
                    "schema": b6s.get("schema"),
                    "schema_version": b6s.get("schema_version"),
                    "seal_profile_id": b6s.get("seal_profile_id"),
                    "seal_sha256": (b6s.get("seal_sha256") or "")[:16] + "…" if b6s.get("seal_sha256") else None,
                    "exclude_policy_id": b6s.get("exclude_policy_id"),
                }
            except Exception as e:
                out["errors"].append(f"B6_SEAL_READ_ERROR: {e}")

        # Overall status
        if not out["errors"]:
            out["status"] = "OK"
        else:
            out["status"] = "WARN" if any("READ_ERROR" in x for x in out["errors"]) else "NA"
        return out


# --- UI panel: Invariant wiring receipts ---
with st.expander("Invariant Wiring / Receipts (Session Bricks 1–6)", expanded=False):
    st.caption(
        "Read-only observability layer. Adds receipts + gate evaluations without changing solver behavior. "
        "(TypeGate / fp.v1 / barcode are evaluated on chosen matrices; they do not assert admissible links.)"
    )

    st.markdown("**Frozen profile IDs (this session)**")
    st.code(
        "\n".join(
            [
                f"session_profile_id: {_INV_SESSION_PROFILE_ID}",
                f"stepA_profile_id: {_INV_STEP_A_PROFILE_ID}",
                f"fp_profile_id:    {_INV_FPV1_PROFILE_ID}",
                f"type_gate_id:     {_INV_TYPE_GATE_ID}",
                f"stepB_profile_id: {_INV_STEP_B_PROFILE_ID}",
                f"barcode_profile:  {_INV_BARCODE_PROFILE_ID}",
                f"fpv1_kernel_enum_max_dim: {_INV_FPV1_MAX_KERNEL_ENUM_DIM}",
            ]
        )
    )

    # --- Last bundle scan ---
    ss = st.session_state
    last_bdir = (
        ss.get("last_bundle_dir")
        or ss.get("ui_last_bundle_dir")
        or ss.get("ui_last_verify_bundle_dir")
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Last bundle dir (if any)**")
        st.code(str(last_bdir or "(none)"))
    with colB:
        st.markdown("**Engine**")
        st.code(
            "\n".join(
                [
                    f"SCHEMA_VERSION: {globals().get('SCHEMA_VERSION', None)}",
                    f"ENGINE_REV:     {globals().get('ENGINE_REV', None)}",
                    f"APP_VERSION:    {globals().get('APP_VERSION', None)}",
                ]
            )
        )

    scan_key = f"inv_scan_{ss.get('_ui_nonce', '00000000')}"
    if st.button("Scan last bundle artifacts", key=scan_key):
        ss["_inv_last_bundle_scan"] = _inv_scan_last_bundle(last_bdir)

    scan = ss.get("_inv_last_bundle_scan")
    if scan:
        st.markdown("**Last-bundle scan summary**")
        st.json(scan)

    st.markdown("---")

    # --- Matrix selection from current SSOT ---
    st.markdown("## Step‑A tools (fp.v1, TypeGate, barcode)")
    st.caption(
        "These tools operate on matrices available from the current SSOT inputs (B/C/H) and derived strict-core R3. "
        "They do not modify any solver outputs; they only compute receipts." 
    )

    mats: dict[str, list[list[int]]] = {}
    mat_err = None

    try:
        pf = _svr_resolve_all_to_paths()
        (_pB, bB) = pf.get("B")
        (_pC, bC) = pf.get("C")
        (_pH, bH) = pf.get("H")

        # raw blocks
        mats["d3"] = _inv__norm_bitmatrix((bB or {}).get("3") or [], name="d3")
        mats["C3"] = _inv__norm_bitmatrix((bC or {}).get("3") or [], name="C3")
        mats["H2"] = _inv__norm_bitmatrix((bH or {}).get("2") or [], name="H2")

        # derived strict-core matrices when shapes allow
        try:
            core0 = time_tau_strict_core_from_blocks(bB, bC, bH)
            mats["R3_strict"] = _inv__norm_bitmatrix(core0.get("R3") or [], name="R3")
            mats["C3pI"] = _inv__norm_bitmatrix(core0.get("C3pI") or [], name="C3pI")
            mats["H2d3"] = _inv__norm_bitmatrix(core0.get("H2d3") or [], name="H2d3")
        except Exception:
            pass

    except Exception as e:
        mat_err = str(e)

    if mat_err:
        st.warning(
            "Could not resolve current SSOT matrices (B/C/H). "
            f"Reason: {mat_err}"
        )
    else:
        # Show matrix catalog
        rows = []
        for k in sorted(mats.keys()):
            A = mats[k]
            m, n = _inv__shape(A)
            try:
                n_types = len(_inv__col_types(A, include_zero=False))
            except Exception:
                n_types = None
            rows.append({"name": k, "m": m, "n": n, "nonzero_types": n_types})
        st.markdown("**Matrix catalog (current SSOT)**")
        st.json(rows)

        # --- fp.v1 computation ---
        default_A_name = "R3_strict" if "R3_strict" in mats else (sorted(mats.keys())[0] if mats else "")
        A_name = st.selectbox(
            "Matrix for fp.v1 certificate",
            options=sorted(mats.keys()),
            index=(sorted(mats.keys()).index(default_A_name) if default_A_name in mats else 0),
            key=f"inv_fpA_{ss.get('_ui_nonce','00000000')}",
        )

        if st.button("Compute fp.v1 certificate + hash", key=f"inv_fp_compute_{ss.get('_ui_nonce','00000000')}"):
            try:
                A = mats.get(A_name) or []
                cert = _inv_stepA_cert(A)
                fp = _inv_fpv1_sha256(cert)
                ss["_inv_fp_v1_last"] = {"matrix": A_name, "cert": cert, "fp": fp}
            except Exception as e:
                ss["_inv_fp_v1_last"] = {"matrix": A_name, "error": str(e)}

        fp_last = ss.get("_inv_fp_v1_last")
        if fp_last:
            st.markdown("**fp.v1 output**")
            st.json(fp_last)

        st.markdown("---")

        # --- Step-A link gate (structural) ---
        st.markdown("### Step‑A link gates (structural check on a chosen pair)")
        col1, col2 = st.columns(2)
        with col1:
            A0_name = st.selectbox(
                "From matrix (A0)",
                options=sorted(mats.keys()),
                index=0,
                key=f"inv_link_A0_{ss.get('_ui_nonce','00000000')}",
            )
        with col2:
            A1_name = st.selectbox(
                "To matrix (A1)",
                options=sorted(mats.keys()),
                index=(1 if len(mats) > 1 else 0),
                key=f"inv_link_A1_{ss.get('_ui_nonce','00000000')}",
            )

        if st.button("Compute TypeGate + CertGate", key=f"inv_link_compute_{ss.get('_ui_nonce','00000000')}"):
            try:
                A0 = mats.get(A0_name) or []
                A1 = mats.get(A1_name) or []
                g = _inv_stepA_link_gate(A0, A1)
                sb = _inv_stepB_out_of_regime_record(A0, A1)
                ss["_inv_stepA_link_last"] = {"A0": A0_name, "A1": A1_name, "gates": g, "stepB": sb}
            except Exception as e:
                ss["_inv_stepA_link_last"] = {"A0": A0_name, "A1": A1_name, "error": str(e)}

        g_last = ss.get("_inv_stepA_link_last")
        if g_last:
            st.markdown("**Link gate output**")
            st.json(g_last)

        st.markdown("---")

        # --- Step-B boundary probes (Phase 2) ---
        st.markdown("### Step‑B boundary probes (out‑of‑regime detector)")
        st.caption(
            "Run canonical probes (XOR injection / local chord flip) against a chosen base matrix. "
            "Classification is derived from the Step‑A TypeGate (new‑type delta), and expected FAILs are "
            "treated as outside Step‑A (do not contaminate Step‑A claims)."
        )

        default_base = "R3_strict" if "R3_strict" in mats else (sorted(mats.keys())[0] if mats else "")
        probe_A0_name = st.selectbox(
            "Base matrix (A0) for probe",
            options=sorted(mats.keys()),
            index=(sorted(mats.keys()).index(default_base) if default_base in mats else 0),
            key=f"inv_stepB_A0_{ss.get('_ui_nonce','00000000')}",
        )
        probe_kind = st.radio(
            "probe_kind",
            options=["xor_injection", "local_chord_flip"],
            index=0,
            horizontal=True,
            key=f"inv_stepB_kind_{ss.get('_ui_nonce','00000000')}",
        )

        A0p = mats.get(probe_A0_name) or []
        m0p, n0p = _inv__shape(A0p)

        probe_params: dict = {}
        probe_err = None
        A1p = None

        if probe_kind == "xor_injection":
            if n0p < 2:
                st.warning("xor_injection requires A0 to have at least 2 columns.")
            else:
                colX, colY = st.columns(2)
                with colX:
                    ii = st.number_input(
                        "i (column index)",
                        min_value=0,
                        max_value=int(n0p - 1),
                        value=0,
                        step=1,
                        key=f"inv_stepB_xor_i_{ss.get('_ui_nonce','00000000')}",
                    )
                with colY:
                    jj_default = 1 if n0p > 1 else 0
                    jj = st.number_input(
                        "j (column index)",
                        min_value=0,
                        max_value=int(n0p - 1),
                        value=int(jj_default),
                        step=1,
                        key=f"inv_stepB_xor_j_{ss.get('_ui_nonce','00000000')}",
                    )
                probe_params = {"i": int(ii), "j": int(jj)}
        else:
            if m0p < 4:
                st.warning("local_chord_flip probe is most meaningful when A0 has at least 4 rows (a,b,c,d).")
            colA, colB = st.columns(2)
            with colA:
                aa = st.number_input(
                    "a (row index)",
                    min_value=0,
                    max_value=max(int(m0p - 1), 0),
                    value=0,
                    step=1,
                    key=f"inv_stepB_chord_a_{ss.get('_ui_nonce','00000000')}",
                )
                bb = st.number_input(
                    "b (row index)",
                    min_value=0,
                    max_value=max(int(m0p - 1), 0),
                    value=(1 if m0p > 1 else 0),
                    step=1,
                    key=f"inv_stepB_chord_b_{ss.get('_ui_nonce','00000000')}",
                )
            with colB:
                cc = st.number_input(
                    "c (row index)",
                    min_value=0,
                    max_value=max(int(m0p - 1), 0),
                    value=(2 if m0p > 2 else 0),
                    step=1,
                    key=f"inv_stepB_chord_c_{ss.get('_ui_nonce','00000000')}",
                )
                dd = st.number_input(
                    "d (row index)",
                    min_value=0,
                    max_value=max(int(m0p - 1), 0),
                    value=(3 if m0p > 3 else 0),
                    step=1,
                    key=f"inv_stepB_chord_d_{ss.get('_ui_nonce','00000000')}",
                )
            probe_params = {"a": int(aa), "b": int(bb), "c": int(cc), "d": int(dd)}

        do_log = st.checkbox(
            "Append probe record to logs/reports/stepb_boundary_probes.jsonl",
            value=True,
            key=f"inv_stepB_log_{ss.get('_ui_nonce','00000000')}",
        )

        if st.button("Run Step‑B probe", key=f"inv_stepB_run_{ss.get('_ui_nonce','00000000')}"):
            try:
                if probe_kind == "xor_injection":
                    A1p = _inv_stepB_probe_xor_injection(A0p, i=int(probe_params.get("i", 0)), j=int(probe_params.get("j", 0)))
                else:
                    A1p = _inv_stepB_probe_local_chord_flip(
                        A0p,
                        a=int(probe_params.get("a", 0)),
                        b=int(probe_params.get("b", 0)),
                        c=int(probe_params.get("c", 0)),
                        d=int(probe_params.get("d", 0)),
                    )
            except Exception as e:
                probe_err = str(e)

            if probe_err:
                ss["_inv_stepB_probe_last"] = {
                    "A0": probe_A0_name,
                    "probe_id": probe_kind,
                    "probe_params": probe_params,
                    "error": probe_err,
                }
            else:
                sb = _inv_stepB_out_of_regime_record(A0p, A1p or [], probe_id=probe_kind, probe_params=probe_params)
                gates = _inv_stepA_link_gate(A0p, A1p or [])

                # Small A1 summary (payload; avoids dumping huge matrices into logs by default)
                try:
                    m1p, n1p = _inv__shape(A1p or [])
                    t1p = len(_inv__col_types(_inv__norm_bitmatrix(A1p or [], name="A1"), include_zero=False))
                except Exception:
                    m1p, n1p, t1p = None, None, None

                out = {
                    "A0": probe_A0_name,
                    "A1_summary": {"m": m1p, "n": n1p, "nonzero_types": t1p},
                    "probe_id": probe_kind,
                    "probe_params": probe_params,
                    "stepB": sb,
                    "stepA_gates": gates,
                }
                ss["_inv_stepB_probe_last"] = out

                if do_log:
                    try:
                        _inv_stepB_probe_log_append(
                            {
                                "unit": "step_b",
                                "kind": "boundary_probe",
                                "A0": probe_A0_name,
                                "A1_summary": out.get("A1_summary"),
                                "probe_id": probe_kind,
                                "probe_params": probe_params,
                                "stepB": sb,
                                "stepA_gates": {
                                    "type_gate": (gates or {}).get("type_gate"),
                                    "cert_gate": (gates or {}).get("cert_gate"),
                                },
                            }
                        )
                    except Exception:
                        pass

        probe_last = ss.get("_inv_stepB_probe_last")
        if probe_last:
            st.markdown("**Step‑B probe output**")
            st.json(probe_last)

            # Non-contamination note (only the classification is load-bearing)
            try:
                sb = (probe_last or {}).get("stepB") or {}
                if isinstance(sb, dict) and sb.get("regime") == "NOT_ADMISSIBLE_FOR_STEP_A":
                    st.warning(
                        "Probe classified as NOT admissible for Step‑A. Any Step‑A invariance failures here are expected "
                        "and must not be interpreted as Step‑A contradictions."
                    )
                elif isinstance(sb, dict) and sb.get("regime") == "STEP_A_ADMISSIBLE":
                    st.success(
                        "This probe instance stayed inside Step‑A (TypeGate PASS). Step‑A invariance is expected to hold."
                    )
            except Exception:
                pass

        st.markdown("---")

        # --- Barcode ---
        st.markdown("### Barcode builder (sequence of per-level hashes)")
        st.caption(
            "Equality meaning is strict: same length and per-level hash equality. "
            "Choose `full` (fp.v1 hash) or `component` (hash of A_comp list)."
        )

        barcode_kind = st.radio(
            "barcode_kind",
            options=["full", "component"],
            index=0,
            horizontal=True,
            key=f"inv_barcode_kind_{ss.get('_ui_nonce','00000000')}",
        )
        seq_text = st.text_input(
            "Sequence (comma-separated matrix names)",
            value=("R3_strict" if "R3_strict" in mats else (sorted(mats.keys())[0] if mats else "")),
            key=f"inv_barcode_seq_{ss.get('_ui_nonce','00000000')}",
        )

        if st.button("Compute barcode", key=f"inv_barcode_compute_{ss.get('_ui_nonce','00000000')}"):
            names = [x.strip() for x in (seq_text or "").split(",") if x.strip()]
            chosen = [(nm, mats.get(nm) or []) for nm in names if nm in mats]
            ss["_inv_barcode_last"] = _inv_barcode_from_mats(chosen, barcode_kind=barcode_kind)

        bc_last = ss.get("_inv_barcode_last")
        if bc_last:
            st.markdown("**Barcode output**")
            st.json(bc_last)

    st.markdown("---")
    st.markdown("## Notes on out-of-frontier items")
    st.write(
        "- Step‑C glue semantics and Step‑D rewrite calculus are not invoked by this Streamlit app's current v2 overlap pipeline (but the Phase‑3 Step‑C panel can compute them explicitly). "
        "They remain frozen in the specification layer; when/if you expose BI/uA/uB or rewrite catalogs in the UI, "
        "these receipts can be extended without changing the existing solver behavior."
    )

# =================== /Invariant Wiring / Receipts (Session Bricks 1–6) ===================

# =================== Phase 3 / Step C — Glue & Towers (Session) ===================
with st.expander("Phase 3 — Step‑C glue & towers (Cancel/Persist, GlueRecord, Phase‑3 gate)", expanded=False):
    st.caption(
        "Mode‑B wiring for the Phase‑3 Step‑C surface: codim‑1 interface solvability (Cancel/Persist), "
        "GlueRecord discipline, Step‑A applicability gating (no new type), Step‑A check invocation metering, "
        "and Step‑A tower barcodes (component strings). "
        "This panel is observational and does not change the v2 overlap pipeline unless you copy these helpers into it."
    )

    ss = st.session_state
    ss.setdefault("_stepC_glue_attempts", [])
    ss.setdefault("_stepC_stepA_invocations", [])
    ss.setdefault("_stepC_glue_last", None)
    ss.setdefault("_stepC_tower_barcode_last", None)
    ss.setdefault("_stepC_phase3_report_last", None)

    ss.setdefault("_stepC_glue_profile_mask", None)
    ss.setdefault("_stepC_glue_profile_promotion_report_last", None)
    ss.setdefault("_stepC_tower_hashes_sched_alpha_last", None)
    ss.setdefault("_stepC_tower_hashes_sched_beta_last", None)
    ss.setdefault("_stepC_tower_hashes_first_divergence_last", None)
    ss.setdefault("_stepC_tower_barcode_semantics_last", None)

    # ---- Matrix catalog (best-effort, from current SSOT) ----
    mats: dict[str, list[list[int]]] = {}
    mat_err = None
    try:
        pf = _svr_resolve_all_to_paths()
        (_pB, bB) = pf.get("B")
        (_pC, bC) = pf.get("C")
        (_pH, bH) = pf.get("H")

        mats["d3"] = _inv__norm_bitmatrix((bB or {}).get("3") or [], name="d3")
        mats["C3"] = _inv__norm_bitmatrix((bC or {}).get("3") or [], name="C3")
        mats["H2"] = _inv__norm_bitmatrix((bH or {}).get("2") or [], name="H2")

        try:
            core0 = time_tau_strict_core_from_blocks(bB, bC, bH)
            mats["R3_strict"] = _inv__norm_bitmatrix(core0.get("R3") or [], name="R3")
            mats["C3pI"] = _inv__norm_bitmatrix(core0.get("C3pI") or [], name="C3pI")
            mats["H2d3"] = _inv__norm_bitmatrix(core0.get("H2d3") or [], name="H2d3")
        except Exception:
            pass

    except Exception as e:
        mat_err = str(e)

    if mat_err:
        st.warning(f"Could not resolve SSOT matrices for Step‑C gating helpers: {mat_err}")

    if mats:
        rows = []
        for k in sorted(mats.keys()):
            A = mats[k]
            m, n = _inv__shape(A)
            try:
                n_types = len(_inv__col_types(A, include_zero=False))
            except Exception:
                n_types = None
            rows.append({"name": k, "m": m, "n": n, "nonzero_types": n_types})
        st.markdown("**Matrix catalog (current SSOT)**")
        st.json(rows)

    st.markdown("---")

    # ---- Step‑C glue (interface + record) ----
    st.markdown("## Step‑C glue (codim‑1 interface test + GlueRecord)")

    st.caption(
        "Cancel iff ∃ϕ with B_I ϕ = u_A + u_B over F2; else Persist. "
        "GlueRecord projects only: phi_exists, new_type, decision, stepA_applies."
    )

    # Interface inputs (JSON)
    default_BI = "[[1,0],[0,1]]"
    default_u = "[0,0]"

    BI_text = st.text_area(
        "B_I (JSON matrix over {0,1})",
        value=default_BI,
        height=110,
        key=f"stepC_BI_{ss.get('_ui_nonce','00000000')}",
    )
    uA_text = st.text_area(
        "u_A (JSON vector over {0,1})",
        value=default_u,
        height=70,
        key=f"stepC_uA_{ss.get('_ui_nonce','00000000')}",
    )
    uB_text = st.text_area(
        "u_B (JSON vector over {0,1})",
        value=default_u,
        height=70,
        key=f"stepC_uB_{ss.get('_ui_nonce','00000000')}",
    )

    # Pre/post matrices for new-type gate (best-effort from SSOT; fallback to JSON)
    A_before = None
    A_after = None
    A_before_name = None
    A_after_name = None

    if mats:
        col1, col2 = st.columns(2)
        with col1:
            A_before_name = st.selectbox(
                "A_before (pre‑glue global)",
                options=sorted(mats.keys()),
                index=0,
                key=f"stepC_A_before_{ss.get('_ui_nonce','00000000')}",
            )
        with col2:
            A_after_name = st.selectbox(
                "A_after (post‑glue global)",
                options=sorted(mats.keys()),
                index=(1 if len(mats) > 1 else 0),
                key=f"stepC_A_after_{ss.get('_ui_nonce','00000000')}",
            )
        A_before = mats.get(A_before_name) or []
        A_after = mats.get(A_after_name) or []
    else:
        st.caption("No SSOT matrices available; paste A_before/A_after as JSON matrices for new‑type gating.")
        A0_text = st.text_area(
            "A_before (JSON matrix)",
            value="[]",
            height=90,
            key=f"stepC_A0_json_{ss.get('_ui_nonce','00000000')}",
        )
        A1_text = st.text_area(
            "A_after (JSON matrix)",
            value="[]",
            height=90,
            key=f"stepC_A1_json_{ss.get('_ui_nonce','00000000')}",
        )
        A_before = None
        A_after = None
        try:
            A_before = _json.loads(A0_text or "[]")
            A_after = _json.loads(A1_text or "[]")
        except Exception as e:
            st.error(f"Could not parse A_before/A_after JSON: {e}")

    invoke_stepA = st.checkbox(
        "Invoke Step‑A parity check when applicable (no new type)",
        value=True,
        key=f"stepC_invoke_stepA_{ss.get('_ui_nonce','00000000')}",
    )

    if st.button("Compute Step‑C GlueRecord", key=f"stepC_glue_compute_{ss.get('_ui_nonce','00000000')}"):
        gid = uuid.uuid4().hex[:8]
        try:
            BI = _json.loads(BI_text or "[]")
            uA = _json.loads(uA_text or "[]")
            uB = _json.loads(uB_text or "[]")
            if A_before is None or A_after is None:
                raise ValueError("A_before/A_after not available")

            wrapper = _stepC_build_glue_record(
                B_I=BI,
                u_A=uA,
                u_B=uB,
                A_before=A_before,
                A_after=A_after,
                glue_id=gid,
                invoke_stepA_when_applicable=bool(invoke_stepA),
                invocation_log=ss.get("_stepC_stepA_invocations"),
            )

            # Store an attempt capsule sufficient for reproducibility checks.
            attempt = {
                "glue_id": gid,
                "B_I": BI,
                "u_A": uA,
                "u_B": uB,
                "A_before": A_before,
                "A_after": A_after,
                "A_before_name": A_before_name,
                "A_after_name": A_after_name,
                "glue_record": (wrapper or {}).get("glue_record"),
                # payload
                "stepA_check": (wrapper or {}).get("stepA_check"),
                "interface": (wrapper or {}).get("interface"),
                "new_type_evidence": (wrapper or {}).get("new_type_evidence"),
            }
            ss["_stepC_glue_attempts"].append(attempt)
            ss["_stepC_glue_last"] = wrapper

        except Exception as e:
            ss["_stepC_glue_last"] = {"error": str(e), "glue_id": gid}

    glue_last = ss.get("_stepC_glue_last")
    if glue_last:
        st.markdown("**Last Step‑C glue output**")
        st.json(glue_last)
        try:
            gr = (glue_last or {}).get("glue_record") or {}
            if isinstance(gr, dict):
                if gr.get("decision") == "Cancel":
                    st.warning("Decision: Cancel (ϕ exists).")
                elif gr.get("decision") == "Persist":
                    st.success("Decision: Persist (no ϕ).")
        except Exception:
            pass

    st.markdown("---")

    

    # ---- Glue profile library + promotion gates (additive) ----
    st.markdown("## Glue profiles (promotion gates)")
    st.caption("Additive surface: controlled glue decision profiles. Does not change the frozen Gate‑C decision test.")
    try:
        _gp_catalog = _b1_stepc_glue_profile_catalog()
    except Exception:
        _gp_catalog = None
    if _gp_catalog is not None:
        st.json(_gp_catalog)

    # G1 mask parameter (comma-separated column indices). Blank disables G1 (NOT_APPLICABLE).
    _mask_default = ""
    try:
        if isinstance(ss.get("_stepC_glue_profile_mask"), list):
            _mask_default = ",".join(str(int(x)) for x in ss.get("_stepC_glue_profile_mask") or [])
    except Exception:
        _mask_default = ""
    mask_text = st.text_input(
        "G1 mask (comma-separated B_I column indices; blank disables G1)",
        value=_mask_default,
        key="stepc_glue_profile_mask_text",
    )

    def _stepc__parse_mask_text(s: str) -> list[int] | None:
        txt = str(s or "").strip()
        if not txt:
            return None
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        out = []
        for p in parts:
            out.append(int(p))
        # canonicalize
        out = sorted(set(int(x) for x in out))
        return out

    cols_gp = st.columns(2)
    with cols_gp[0]:
        if st.button("Compute/Update profile receipts for all attempts", key="stepc_compute_profile_receipts"):
            try:
                mask_list = _stepc__parse_mask_text(mask_text)
                ss["_stepC_glue_profile_mask"] = mask_list
            except Exception as e:
                ss["_stepC_glue_profile_mask"] = None
                st.error(f"Mask parse error: {e}")
                mask_list = None

            # Attach receipts per attempt (mutates session-state attempts only; core gate unaffected).
            try:
                cat = _b1_stepc_glue_profile_catalog()
                for att in (ss.get("_stepC_glue_attempts") or []):
                    if not isinstance(att, dict):
                        continue
                    recs = _b1_stepc_compute_glue_profile_receipts_for_attempt(att, profile_catalog=cat, profile_mask=mask_list)
                    if recs:
                        att["glue_profile_receipts"] = recs
                st.success("Profile receipts updated on Step‑C glue attempts.")
            except Exception as e:
                st.error(f"Receipt build error: {e}")

    with cols_gp[1]:
        if st.button("Evaluate glue profile promotion report", key="stepc_eval_profile_promotion"):
            try:
                cat = _b1_stepc_glue_profile_catalog()
                rr = _b1_stepc_glue_profile_promotion_report(
                    ss.get("_stepC_glue_attempts") or [],
                    ss.get("_stepC_stepA_invocations") or [],
                    profile_catalog=cat,
                    profile_mask=ss.get("_stepC_glue_profile_mask"),
                )
                ss["_stepC_glue_profile_promotion_report_last"] = rr
            except Exception as e:
                ss["_stepC_glue_profile_promotion_report_last"] = {"error": str(e)}

    promo_last = ss.get("_stepC_glue_profile_promotion_report_last")
    if promo_last is not None:
        st.markdown("### Glue profile promotion report (last)")
        st.json(promo_last)

    st.markdown("---")

# ---- Step‑C tower barcode ----
    st.markdown("## Step‑C tower barcode (component strings across levels)")
    st.caption(
        "For Step‑A towers (TypeGate PASS on each adjacent link), the component string must be constant across levels."
    )

    seq_text = st.text_input(
        "Tower sequence (comma-separated matrix names; interpreted as levels 0..L in order)",
        value=("R3_strict" if "R3_strict" in mats else (sorted(mats.keys())[0] if mats else "")),
        key=f"stepC_tower_seq_{ss.get('_ui_nonce','00000000')}",
    )

    if st.button("Compute Step‑C tower barcode", key=f"stepC_tower_compute_{ss.get('_ui_nonce','00000000')}"):
        try:
            if not mats:
                raise ValueError("No SSOT matrices available for tower barcode")
            names = [x.strip() for x in (seq_text or "").split(",") if x.strip()]
            chosen = [(nm, mats.get(nm) or []) for nm in names if nm in mats]
            ss["_stepC_tower_barcode_last"] = _stepC_tower_barcode(chosen)
        except Exception as e:
            ss["_stepC_tower_barcode_last"] = {"error": str(e)}

    tb_last = ss.get("_stepC_tower_barcode_last")
    if tb_last:
        st.markdown("**TowerBarcode output**")
        st.json(tb_last)
        try:
            if isinstance(tb_last, dict) and tb_last.get("is_stepA_tower") is True:
                const = tb_last.get("constant_across_levels_when_stepA")
                if const is True:
                    st.success("Step‑A tower barcode is constant across levels.")
                elif const is False:
                    st.error(f"Step‑A tower barcode diverged (first divergence at level {tb_last.get('first_divergence_level')}).")
        except Exception:
            pass



# ---- Towers: schedule hashes + first-divergence mapping (additive) ----
st.markdown("### Towers: schedule hashes (fp.v1) + first-divergence mapping")
st.caption("Additive receipts: barcode sequences for sched-alpha / sched-beta (full fp.v1) and their first-divergence mapping.")

_alpha_default = str(seq_text or "")
_beta_default = str(seq_text or "")
alpha_text = st.text_input(
    "sched-alpha sequence (comma-separated SSOT matrix names)",
    value=_alpha_default,
    key=f"stepC_sched_alpha_{ss.get('_ui_nonce','00000000')}",
)
beta_text = st.text_input(
    "sched-beta sequence (comma-separated SSOT matrix names)",
    value=_beta_default,
    key=f"stepC_sched_beta_{ss.get('_ui_nonce','00000000')}",
)

if st.button("Compute schedule hashes + first divergence", key=f"stepC_sched_hashes_compute_{ss.get('_ui_nonce','00000000')}"):
    try:
        if not mats:
            raise ValueError("No SSOT matrices available for schedule hashes")
        a_names = [x.strip() for x in (alpha_text or "").split(",") if x.strip()]
        b_names = [x.strip() for x in (beta_text or "").split(",") if x.strip()]
        a_chosen = [(nm, mats.get(nm) or []) for nm in a_names if nm in mats]
        b_chosen = [(nm, mats.get(nm) or []) for nm in b_names if nm in mats]
        bc_a = _inv_barcode_from_mats(a_chosen, barcode_kind="full")
        bc_b = _inv_barcode_from_mats(b_chosen, barcode_kind="full")
        ss["_stepC_tower_hashes_sched_alpha_last"] = bc_a
        ss["_stepC_tower_hashes_sched_beta_last"] = bc_b
        ss["_stepC_tower_hashes_first_divergence_last"] = _b1_stepc_first_divergence_mapping(bc_a, bc_b)
        st.success("Schedule hashes + first divergence computed.")
    except Exception as e:
        st.error(f"Schedule hash compute error: {e}")

if ss.get("_stepC_tower_hashes_sched_alpha_last") is not None:
    st.markdown("**sched-alpha (full)**")
    st.json(ss.get("_stepC_tower_hashes_sched_alpha_last"))
if ss.get("_stepC_tower_hashes_sched_beta_last") is not None:
    st.markdown("**sched-beta (full)**")
    st.json(ss.get("_stepC_tower_hashes_sched_beta_last"))
if ss.get("_stepC_tower_hashes_first_divergence_last") is not None:
    st.markdown("**first-divergence mapping**")
    st.json(ss.get("_stepC_tower_hashes_first_divergence_last"))

# ---- Towers: barcode semantics bridge (NON_BINDING) ----
st.markdown("### Towers: barcode semantics bridge (NON_BINDING)")
st.caption("Additive receipt: relates TowerBarcode component strings to component/full barcode sequences (for reporting only).")

if st.button("Compute tower barcode semantics bridge", key=f"stepC_tower_barcode_semantics_compute_{ss.get('_ui_nonce','00000000')}"):
    try:
        if not mats:
            raise ValueError("No SSOT matrices available for tower barcode semantics")
        names = [x.strip() for x in (seq_text or "").split(",") if x.strip()]
        chosen = [(nm, mats.get(nm) or []) for nm in names if nm in mats]
        tb = _stepC_tower_barcode(chosen)
        bc_comp = _inv_barcode_from_mats(chosen, barcode_kind="component")
        bc_full = _inv_barcode_from_mats(chosen, barcode_kind="full")

        def _div_summary(tower_obj: dict, bc_obj: dict) -> dict:
            if not (isinstance(tower_obj, dict) and tower_obj.get("is_stepA_tower") is True):
                return {"constant_across_levels_when_stepA": None, "first_divergence_level": None}
            seq = bc_obj.get("sequence") if isinstance(bc_obj, dict) else None
            seq = seq if isinstance(seq, list) else []
            vals = []
            for row in seq:
                if isinstance(row, dict) and row.get("status") == "OK":
                    vals.append(row.get("hash"))
                else:
                    vals.append(None)
            if not vals:
                return {"constant_across_levels_when_stepA": True, "first_divergence_level": None}
            base = vals[0]
            first = None
            for i, v in enumerate(vals):
                if v != base:
                    first = int(i)
                    break
            return {"constant_across_levels_when_stepA": (first is None), "first_divergence_level": first}

        div_comp = _div_summary(tb, bc_comp)
        div_full = _div_summary(tb, bc_full)

        # Coherence: SHA-256(component_string_by_level[k]) == component barcode hash (when status OK)
        mism = []
        try:
            comp_by_level = (tb.get("component_string_by_level") or {}) if isinstance(tb, dict) else {}
            seq = bc_comp.get("sequence") if isinstance(bc_comp, dict) else None
            seq = seq if isinstance(seq, list) else []
            for i, row in enumerate(seq):
                if not isinstance(row, dict):
                    continue
                if row.get("status") != "OK":
                    continue
                exp_h = row.get("hash")
                comp_txt = comp_by_level.get(str(int(i))) if isinstance(comp_by_level, dict) else None
                got_h = _hash.sha256(str(comp_txt or "").encode("utf-8")).hexdigest()
                if exp_h != got_h:
                    mism.append({"level": int(i), "expected_hash": exp_h, "got_hash": got_h})
        except Exception:
            pass

        coherence = {
            "status": ("OK" if not mism else "FAIL"),
            "mismatches": mism,
        }

        receipt = {
            "schema": "stepc_tower_barcode_semantics",
            "schema_version": "stepc.tower_barcode_semantics.v1",
            "profile_id": "Phase3.StepC.Towers.BarcodeSemantics.v1",
            "binding_status": "NON_BINDING",
            "sig8": "",
            "core": {
                "tower": {"level_names": (tb.get("level_names") if isinstance(tb, dict) else None) or [nm for (nm, _A) in chosen], "tower_barcode": tb},
                "barcodes": {"component": bc_comp, "full": bc_full},
                "divergence": {"component": div_comp, "full": div_full},
                "coherence": coherence,
            },
        }
        hb = {k: receipt.get(k) for k in ["schema", "schema_version", "profile_id", "binding_status", "core", "sig8"]}
        hb["sig8"] = ""
        receipt["sig8"] = hash_json_sig8(hb)
        ss["_stepC_tower_barcode_semantics_last"] = receipt
        st.success("Tower barcode semantics bridge computed.")
    except Exception as e:
        ss["_stepC_tower_barcode_semantics_last"] = {"error": str(e)}
        st.error(f"Tower barcode semantics error: {e}")

if ss.get("_stepC_tower_barcode_semantics_last") is not None:
    st.markdown("**tower_barcode_semantics_last**")
    st.json(ss.get("_stepC_tower_barcode_semantics_last"))

    st.markdown("---")

    # ---- Phase‑3 gate ----
    st.markdown("## Phase‑3 acceptance gate (Step C)")
    st.caption(
        "Phase 3 passes iff (i) glue decisions are reproducible and (ii) Step‑A checks are only invoked when no new type holds."
    )

    colP, colQ = st.columns(2)
    with colP:
        if st.button("Evaluate Phase‑3 gate on accumulated glues", key=f"stepC_gate_eval_{ss.get('_ui_nonce','00000000')}"):
            ss["_stepC_phase3_report_last"] = _stepC_phase3_report(
                glue_attempts=ss.get("_stepC_glue_attempts"),
                invocation_log=ss.get("_stepC_stepA_invocations"),
            )
    with colQ:
        if st.button("Clear Step‑C attempts + invocations", key=f"stepC_clear_{ss.get('_ui_nonce','00000000')}"):
            ss["_stepC_glue_attempts"] = []
            ss["_stepC_stepA_invocations"] = []
            ss["_stepC_glue_last"] = None
            ss["_stepC_phase3_report_last"] = None
            ss["_stepC_tower_barcode_last"] = None
            ss["_stepC_glue_profile_mask"] = None
            ss["_stepC_glue_profile_promotion_report_last"] = None
            ss["_stepC_tower_hashes_sched_alpha_last"] = None
            ss["_stepC_tower_hashes_sched_beta_last"] = None
            ss["_stepC_tower_hashes_first_divergence_last"] = None
            ss["_stepC_tower_barcode_semantics_last"] = None

    rep = ss.get("_stepC_phase3_report_last")
    if rep:
        st.markdown("**Phase‑3 report**")
        st.json(rep)
        try:
            if isinstance(rep, dict) and rep.get("phase3_pass") is True:
                st.success("Phase 3 gate: PASS")
            elif isinstance(rep, dict) and rep.get("phase3_pass") is False:
                st.error("Phase 3 gate: FAIL")
        except Exception:
            pass



# =================== Parity Artifact Family v0 (Session S1) ===================
# Mode B wiring: minimal helpers to build/stamp/validate parity_instance.v1 and
# parity_certificate.v1 artifacts using the frozen handshake:
#   - canonical_json + hash_json_sig8 (v2) for hashing
#   - hashed-body selection (meta + unknown top-level keys quarantined)
#   - sig8 computed with sig8 cleared to "" in the hashed body

PARITY_INSTANCE_SCHEMA = "parity_instance"
PARITY_INSTANCE_SCHEMA_VERSION = "parity_instance.v1"

PARITY_CERTIFICATE_SCHEMA = "parity_certificate"
PARITY_CERTIFICATE_SCHEMA_VERSION = "parity_certificate.v1"

# JSON Schemas (Draft 2020-12). Top-level payload is allowed; core is closed.
PARITY_INSTANCE_V1_JSONSCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "parity_instance.v1.json",
    "title": "parity_instance.v1",
    "type": "object",
    "additionalProperties": True,
    "required": ["schema", "schema_version", "sig8", "core"],
    "properties": {
        "schema": {"const": PARITY_INSTANCE_SCHEMA},
        "schema_version": {"const": PARITY_INSTANCE_SCHEMA_VERSION},
        "sig8": {"type": "string", "pattern": "^[0-9a-f]{8}$"},
        "core": {"$ref": "#/$defs/Core"},
        "meta": {
            "type": "object",
            "description": "Payload quarantine: ignored by semantic identity; free-form metadata.",
            "additionalProperties": True,
        },
    },
    "$defs": {
        "Bit": {"type": "integer", "enum": [0, 1]},
        "BitVector": {"type": "array", "items": {"$ref": "#/$defs/Bit"}},
        "BitMatrix": {
            "type": "array",
            "items": {"type": "array", "items": {"$ref": "#/$defs/Bit"}},
        },
        "SimplexIndex": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "IndexSet": {"type": "array", "items": {"$ref": "#/$defs/SimplexIndex"}},
        "Indices": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prev", "curr", "next"],
            "properties": {
                "prev": {"$ref": "#/$defs/IndexSet"},
                "curr": {"$ref": "#/$defs/IndexSet"},
                "next": {"$ref": "#/$defs/IndexSet"},
            },
        },
        "Core": {
            "type": "object",
            "additionalProperties": False,
            "required": ["indices", "delta_prev", "delta_curr", "phi"],
            "properties": {
                "indices": {"$ref": "#/$defs/Indices"},
                "delta_prev": {"$ref": "#/$defs/BitMatrix"},
                "delta_curr": {"$ref": "#/$defs/BitMatrix"},
                "phi": {"$ref": "#/$defs/BitVector"},
            },
        },
    },
}

PARITY_CERTIFICATE_V1_JSONSCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "parity_certificate.v1.json",
    "title": "parity_certificate.v1",
    "type": "object",
    "additionalProperties": True,
    "required": ["schema", "schema_version", "sig8", "instance_sig8", "core"],
    "properties": {
        "schema": {"const": PARITY_CERTIFICATE_SCHEMA},
        "schema_version": {"const": PARITY_CERTIFICATE_SCHEMA_VERSION},
        "sig8": {"type": "string", "pattern": "^[0-9a-f]{8}$"},
        "instance_sig8": {"type": "string", "pattern": "^[0-9a-f]{8}$"},
        "core": {"$ref": "#/$defs/CertificateCore"},
        "meta": {
            "type": "object",
            "description": "Non-load-bearing payload (UI, paths, timestamps, commentary).",
            "additionalProperties": True,
        },
    },
    "$defs": {
        "Bit": {"type": "integer", "enum": [0, 1]},
        "BitVector": {"type": "array", "items": {"$ref": "#/$defs/Bit"}},
        "IndexPositions": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "DualWitness": {
            "type": "object",
            "additionalProperties": False,
            "required": ["vector", "pairs_to"],
            "properties": {
                "vector": {"$ref": "#/$defs/BitVector"},
                "pairs_to": {"$ref": "#/$defs/Bit"},
            },
        },
        "Witness": {
            "type": "object",
            "additionalProperties": False,
            "required": ["dual_witness"],
            "properties": {
                "primal_certificate": {"type": "string"},
                "dual_witness": {"$ref": "#/$defs/DualWitness"},
            },
        },
        "CoreSupport": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prev", "curr", "next"],
            "properties": {
                "prev": {"$ref": "#/$defs/IndexPositions"},
                "curr": {"$ref": "#/$defs/IndexPositions"},
                "next": {"$ref": "#/$defs/IndexPositions"},
            },
        },
        "CertificateCore": {
            "type": "object",
            "additionalProperties": False,
            "required": ["witness", "support"],
            "properties": {
                "witness": {"$ref": "#/$defs/Witness"},
                "support": {"$ref": "#/$defs/CoreSupport"},
            },
        },
    },
}

# --- Hash-body selection (meta + unknown top-level keys are quarantined) ---

def parity_instance_hash_body(instance: dict) -> dict:
    """Return the hashed-body selection H_instance(instance) for sig8 stamping.

    H_instance := {schema, schema_version, core, sig8:""}
    """
    if not isinstance(instance, dict):
        raise TypeError("parity_instance_hash_body: instance must be a dict")
    schema = instance.get("schema") or PARITY_INSTANCE_SCHEMA
    schema_version = instance.get("schema_version") or PARITY_INSTANCE_SCHEMA_VERSION
    core = instance.get("core")
    if core is None:
        raise ValueError("parity_instance_hash_body: missing required key 'core'")
    return {
        "schema": schema,
        "schema_version": schema_version,
        "core": core,
        "sig8": "",
    }


def parity_certificate_hash_body(cert: dict) -> dict:
    """Return the hashed-body selection H_cert(cert) for sig8 stamping.

    H_cert := {schema, schema_version, instance_sig8, core, sig8:""}
    """
    if not isinstance(cert, dict):
        raise TypeError("parity_certificate_hash_body: cert must be a dict")
    schema = cert.get("schema") or PARITY_CERTIFICATE_SCHEMA
    schema_version = cert.get("schema_version") or PARITY_CERTIFICATE_SCHEMA_VERSION
    instance_sig8 = cert.get("instance_sig8")
    if not instance_sig8:
        raise ValueError("parity_certificate_hash_body: missing required key 'instance_sig8'")
    core = cert.get("core")
    if core is None:
        raise ValueError("parity_certificate_hash_body: missing required key 'core'")
    return {
        "schema": schema,
        "schema_version": schema_version,
        "instance_sig8": instance_sig8,
        "core": core,
        "sig8": "",
    }

# --- Stamping helpers ---

def stamp_parity_instance_sig8(instance: dict) -> dict:
    """Return a copy of `instance` with sig8 stamped per S1 handshake."""
    base = dict(instance or {})
    base.setdefault("schema", PARITY_INSTANCE_SCHEMA)
    base.setdefault("schema_version", PARITY_INSTANCE_SCHEMA_VERSION)
    hb = parity_instance_hash_body(base)
    base["sig8"] = hash_json_sig8(hb)
    return base


def stamp_parity_certificate_sig8(cert: dict) -> dict:
    """Return a copy of `cert` with sig8 stamped per S1 handshake."""
    base = dict(cert or {})
    base.setdefault("schema", PARITY_CERTIFICATE_SCHEMA)
    base.setdefault("schema_version", PARITY_CERTIFICATE_SCHEMA_VERSION)
    hb = parity_certificate_hash_body(base)
    base["sig8"] = hash_json_sig8(hb)
    return base


# --- Schema validators (Draft 2020-12) ---

_PARITY__V1_VALIDATORS = {"instance": None, "cert": None}

def _parity_get_validators_v1():
    """Lazy-construct Draft2020-12 validators so import-time stays light."""
    try:
        from jsonschema import Draft202012Validator
    except Exception as exc:
        raise RuntimeError(f"jsonschema is required for parity schema validation: {exc!r}") from exc

    if _PARITY__V1_VALIDATORS["instance"] is None:
        _PARITY__V1_VALIDATORS["instance"] = Draft202012Validator(PARITY_INSTANCE_V1_JSONSCHEMA)
    if _PARITY__V1_VALIDATORS["cert"] is None:
        _PARITY__V1_VALIDATORS["cert"] = Draft202012Validator(PARITY_CERTIFICATE_V1_JSONSCHEMA)
    return _PARITY__V1_VALIDATORS["instance"], _PARITY__V1_VALIDATORS["cert"]


def validate_parity_instance_v1(instance: dict) -> list[str]:
    """Return a list of schema-validation error strings (empty if valid)."""
    v_inst, _ = _parity_get_validators_v1()
    errs = []
    for e in sorted(v_inst.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        loc = "/".join(str(x) for x in e.absolute_path)
        errs.append(f"{loc}: {e.message}" if loc else e.message)
    return errs


def validate_parity_certificate_v1(cert: dict) -> list[str]:
    """Return a list of schema-validation error strings (empty if valid)."""
    _, v_cert = _parity_get_validators_v1()
    errs = []
    for e in sorted(v_cert.iter_errors(cert), key=lambda e: list(e.absolute_path)):
        loc = "/".join(str(x) for x in e.absolute_path)
        errs.append(f"{loc}: {e.message}" if loc else e.message)
    return errs


# --- Invariant checks (projection-level; no preferences) ---

def _parity__mat_shape(M) -> tuple[int, int] | None:
    """Return (rows, cols) if M is a rectangular list-of-lists, else None."""
    if not isinstance(M, list):
        return None
    if len(M) == 0:
        return (0, 0)
    if not all(isinstance(r, list) for r in M):
        return None
    cols = len(M[0])
    for r in M:
        if len(r) != cols:
            return None
    return (len(M), cols)


def _parity__dot_mod2(a: list[int], b: list[int]) -> int:
    s = 0
    n = min(len(a), len(b))
    for i in range(n):
        s ^= (int(a[i]) & 1) & (int(b[i]) & 1)
    # If lengths mismatch, caller should treat as error; dot uses min length only.
    return s & 1


def _parity__matmul_mod2(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Matrix multiply mod 2. Assumes shapes are compatible and rectangular."""
    m = len(A)
    n = len(A[0]) if m else 0
    p = len(B[0]) if B else 0
    out = [[0 for _ in range(p)] for _ in range(m)]
    # Precompute columns of B for speed on tiny matrices.
    cols = [[B[i][j] for i in range(len(B))] for j in range(p)]
    for i in range(m):
        row = A[i]
        for j in range(p):
            out[i][j] = _parity__dot_mod2(row, cols[j])
    return out


def check_parity_instance_invariants_v1(instance: dict) -> list[str]:
    """Return invariant violations for parity_instance.v1 (empty if OK)."""
    errs: list[str] = []
    if not isinstance(instance, dict):
        return ["instance is not a dict"]
    core = instance.get("core")
    if not isinstance(core, dict):
        return ["core: must be an object"]

    # Indices lengths
    indices = core.get("indices")
    if not isinstance(indices, dict):
        return ["core/indices: must be an object"]

    prev = indices.get("prev")
    curr = indices.get("curr")
    nxt = indices.get("next")
    if not isinstance(prev, list) or not isinstance(curr, list) or not isinstance(nxt, list):
        return ["core/indices: prev/curr/next must be arrays"]

    n_prev, n_curr, n_next = len(prev), len(curr), len(nxt)

    # Matrix shapes
    dp = core.get("delta_prev")
    dc = core.get("delta_curr")
    shp_dp = _parity__mat_shape(dp)
    shp_dc = _parity__mat_shape(dc)
    if shp_dp is None:
        errs.append("core/delta_prev: must be a rectangular matrix (array of equal-length arrays)")
    if shp_dc is None:
        errs.append("core/delta_curr: must be a rectangular matrix (array of equal-length arrays)")

    if shp_dp is not None:
        r, c = shp_dp
        if r != n_curr or c != n_prev:
            errs.append(f"core/delta_prev: shape {r}x{c} must equal n_curr x n_prev = {n_curr}x{n_prev}")
    if shp_dc is not None:
        r, c = shp_dc
        if r != n_next or c != n_curr:
            errs.append(f"core/delta_curr: shape {r}x{c} must equal n_next x n_curr = {n_next}x{n_curr}")

    # Phi alignment (window rule)
    phi = core.get("phi")
    if not isinstance(phi, list):
        errs.append("core/phi: must be an array")
    else:
        expected = n_next if n_next > 0 else n_curr
        if len(phi) != expected:
            errs.append(f"core/phi: length {len(phi)} must be {expected}")

    # Chain condition (delta_curr * delta_prev == 0 mod 2)
    if shp_dp is not None and shp_dc is not None and shp_dp[0] == n_curr and shp_dp[1] == n_prev and shp_dc[0] == n_next and shp_dc[1] == n_curr:
        try:
            prod = _parity__matmul_mod2(dc, dp)  # type: ignore[arg-type]
            if any((x & 1) != 0 for row in prod for x in row):
                errs.append("core: chain condition failed (delta_curr * delta_prev != 0 mod 2)")
        except Exception as exc:
            errs.append(f"core: chain condition check failed to run: {exc!r}")

    return errs


def check_parity_certificate_invariants_v1(instance: dict, cert: dict) -> list[str]:
    """Return invariant violations for parity_certificate.v1 relative to an instance."""
    errs: list[str] = []
    if not isinstance(instance, dict):
        return ["instance is not a dict"]
    if not isinstance(cert, dict):
        return ["cert is not a dict"]

    # Binding
    inst_sig8 = str(instance.get("sig8") or "")
    cert_inst_sig8 = str(cert.get("instance_sig8") or "")
    if inst_sig8 and cert_inst_sig8 and inst_sig8 != cert_inst_sig8:
        errs.append("binding: cert.instance_sig8 does not match instance.sig8")

    # Support bounds
    try:
        idx = (instance.get("core") or {}).get("indices") or {}
        n_prev = len(idx.get("prev") or [])
        n_curr = len(idx.get("curr") or [])
        n_next = len(idx.get("next") or [])
    except Exception:
        n_prev = n_curr = n_next = 0

    support = ((cert.get("core") or {}).get("support") or {})
    for key, n in (("prev", n_prev), ("curr", n_curr), ("next", n_next)):
        arr = support.get(key)
        if not isinstance(arr, list):
            errs.append(f"core/support/{key}: must be an array")
            continue
        for pos in arr:
            try:
                ip = int(pos)
            except Exception:
                errs.append(f"core/support/{key}: contains non-integer position {pos!r}")
                continue
            if ip < 0 or ip >= n:
                errs.append(f"core/support/{key}: position {ip} out of bounds (0..{max(n-1,0)})")

    # Dual pairing to phi
    phi = ((instance.get("core") or {}).get("phi") or [])
    dw = (((cert.get("core") or {}).get("witness") or {}).get("dual_witness") or {})
    vec = dw.get("vector")
    pairs_to = dw.get("pairs_to")
    if not isinstance(phi, list) or not isinstance(vec, list):
        errs.append("pairing: phi and dual_witness.vector must be arrays")
    else:
        if len(phi) != len(vec):
            errs.append(f"pairing: dual_witness.vector length {len(vec)} must equal phi length {len(phi)}")
        else:
            try:
                dot = _parity__dot_mod2([int(x) & 1 for x in vec], [int(x) & 1 for x in phi])
                if (int(pairs_to) & 1) != dot:
                    errs.append("pairing: dot(dual_witness.vector, phi) mod 2 != dual_witness.pairs_to")
            except Exception as exc:
                errs.append(f"pairing: failed to compute dot product: {exc!r}")

    return errs


def validate_parity_pair_v1(instance: dict, cert: dict) -> tuple[bool, list[str]]:
    """Validate (instance, cert) against S1 Mode-A frozen surfaces."""
    problems: list[str] = []

    # 1) Schemas
    problems.extend([f"instance schema: {e}" for e in validate_parity_instance_v1(instance)])
    problems.extend([f"cert schema: {e}" for e in validate_parity_certificate_v1(cert)])

    # 2) Sig8 self-consistency (hashed body selection + canonical hashing)
    try:
        expected_i = hash_json_sig8(parity_instance_hash_body(instance))
        got_i = str(instance.get("sig8") or "")
        if got_i != expected_i:
            problems.append(f"instance sig8 mismatch: got {got_i}, expected {expected_i}")
    except Exception as exc:
        problems.append(f"instance sig8 recompute failed: {exc!r}")

    try:
        expected_c = hash_json_sig8(parity_certificate_hash_body(cert))
        got_c = str(cert.get("sig8") or "")
        if got_c != expected_c:
            problems.append(f"cert sig8 mismatch: got {got_c}, expected {expected_c}")
    except Exception as exc:
        problems.append(f"cert sig8 recompute failed: {exc!r}")

    # 3) Instance invariants
    problems.extend([f"instance invariant: {e}" for e in check_parity_instance_invariants_v1(instance)])

    # 4) Cert invariants (relative)
    problems.extend([f"cert invariant: {e}" for e in check_parity_certificate_invariants_v1(instance, cert)])

    ok = len(problems) == 0
    return ok, problems


def parity_example_pair_v1() -> tuple[dict, dict]:
    """Return the canonical S1 example pair (matches the frozen hash vectors)."""
    inst = {
        "schema": PARITY_INSTANCE_SCHEMA,
        "schema_version": PARITY_INSTANCE_SCHEMA_VERSION,
        "sig8": "",
        "core": {
            "indices": {
                "prev": [[0], [1]],
                "curr": [[0, 1], [1, 2]],
                "next": [[0, 1, 2]],
            },
            "delta_prev": [
                [1, 1],
                [1, 1],
            ],
            "delta_curr": [
                [1, 1],
            ],
            "phi": [1],
        },
    }
    inst = stamp_parity_instance_sig8(inst)

    cert = {
        "schema": PARITY_CERTIFICATE_SCHEMA,
        "schema_version": PARITY_CERTIFICATE_SCHEMA_VERSION,
        "sig8": "",
        "instance_sig8": inst["sig8"],
        "core": {
            "witness": {
                "primal_certificate": "contradiction-row",
                "dual_witness": {"vector": [1], "pairs_to": 1},
            },
            "support": {"prev": [0], "curr": [0, 1], "next": [0]},
        },
    }
    cert = stamp_parity_certificate_sig8(cert)
    return inst, cert

# =================== /Parity Artifact Family v0 (Session S1) ===================


# =================== Verifier v0 + Minimality v0 (Session S2) ===================
# Mode B wiring: total/deterministic wrappers that make the "certificate promise"
# executable as simple boolean predicates.

def SelectCoreV0(instance, cert):
    """Deterministic core selector (v0).

    parity_certificate.v1 carries a single required core at cert["core"].
    For non-dict inputs, returns None.
    """
    if isinstance(cert, dict):
        return cert.get("core")
    return None


def VerifyV0(instance, cert) -> bool:
    """Total, deterministic verifier wrapper.

    Returns True iff validate_parity_pair_v1(instance, cert) reports ok == True.
    Never raises; any exception maps to False.
    """
    try:
        ok, _problems = validate_parity_pair_v1(instance, cert)
        # Enforce strict-bool surface: only literal True passes.
        return True if ok is True else False
    except Exception:
        return False


def MinimalityClaimedV0(cert) -> bool:
    """Return True iff minimality is explicitly claimed (v0).

    Claim surface is payload-level (top-level additionalProperties is allowed),
    but the claim interpretation is strict: only the literal boolean True counts.
    """
    if not isinstance(cert, dict):
        return False
    claims = cert.get("claims")
    if not isinstance(claims, dict):
        return False
    return True if claims.get("minimality_v0") is True else False


def MinimalityV0(instance, cert) -> bool:
    """Single-deletion inclusion minimality for parity_certificate.v1 support.

    Semantics (v0):
      - If VerifyV0(instance, cert) is False -> return False.
      - Otherwise, for each support atom (prev then curr then next; index order),
        delete exactly that one list entry, restamp cert.sig8, and re-run VerifyV0.
        If any single deletion still verifies -> not minimal -> return False.
      - If all single deletions fail verification -> return True.

    Total: never raises; any exception maps to False.
    """
    try:
        if VerifyV0(instance, cert) is not True:
            return False
        if not isinstance(cert, dict):
            return False

        core = cert.get("core")
        if not isinstance(core, dict):
            return False
        support = core.get("support")
        if not isinstance(support, dict):
            return False

        prev = support.get("prev")
        curr = support.get("curr")
        nxt = support.get("next")
        if not isinstance(prev, list) or not isinstance(curr, list) or not isinstance(nxt, list):
            return False

        import copy as _copy

        for lane in ("prev", "curr", "next"):
            arr = support.get(lane)
            if not isinstance(arr, list):
                return False
            for i in range(len(arr)):
                cert2 = _copy.deepcopy(cert)
                try:
                    del cert2["core"]["support"][lane][i]
                except Exception:
                    return False

                # Keep the mutated cert as a valid artifact under VerifyV0 by
                # restamping its self-hash (sig8).
                cert2 = stamp_parity_certificate_sig8(cert2)

                if VerifyV0(instance, cert2) is True:
                    return False
        return True
    except Exception:
        return False


def MinimalityCheckV0(instance, cert) -> bool:
    """Optional-knob checker: enforce minimality only when explicitly claimed.

    - If minimality is claimed -> return MinimalityV0(instance, cert).
    - If not claimed -> return True (no minimality obligation asserted).
    """
    try:
        if MinimalityClaimedV0(cert) is True:
            return MinimalityV0(instance, cert)
        return True
    except Exception:
        return False


# =================== /Verifier v0 + Minimality v0 (Session S2) ===================



# =================== Phase 4 — Step D rewrite calculus (compression layer) ===================
# NOTE:
#   - This panel is observational and does not change the v2 overlap pipeline.
#   - It wires the *frozen* Phase-4 objects: rules R1–R4, µ termination, D1–D4 micro-checks.
#   - Catalogs (R3/R4 templates) are provided as JSON payload (payload, not semantics).
#
# Source: Phase_U minimal simulator obligations (Step D) for data structures + rules + informal loop. 
# 


if "_p4d__is_bitstring" not in globals():
    def _p4d__is_bitstring(s: object, *, m: int | None = None) -> bool:
        if not isinstance(s, str):
            return False
        if m is not None and len(s) != int(m):
            return False
        return all(ch in "01" for ch in s)


if "_p4d__norm_bitstring" not in globals():
    def _p4d__norm_bitstring(s: object, *, m: int | None = None, name: str = "bitstring") -> str:
        if not isinstance(s, str):
            raise TypeError(f"{name} must be a str")
        ss = s.strip()
        if m is not None and len(ss) != int(m):
            raise ValueError(f"{name} must have length {int(m)} (got {len(ss)})")
        if not _p4d__is_bitstring(ss, m=m):
            raise ValueError(f"{name} must contain only 0/1")
        return ss


if "_p4d__xor_bits" not in globals():
    def _p4d__xor_bits(a: str, b: str) -> str:
        if len(a) != len(b):
            raise ValueError("xor length mismatch")
        return "".join("1" if (aa != bb) else "0" for aa, bb in zip(a, b))


if "_p4d__dot_y" not in globals():
    def _p4d__dot_y(y: str, c: str) -> int:
        """Return ⟨y,c⟩ over F2 (as 0/1)."""
        if len(y) != len(c):
            raise ValueError("dot length mismatch")
        acc = 0
        for yy, cc in zip(y, c):
            if yy == "1" and cc == "1":
                acc ^= 1
        return acc


if "_p4d__seen" not in globals():
    def _p4d__seen(y: str, c: str) -> bool:
        return _p4d__dot_y(y, c) == 1


if "_p4d__norm_multiset" not in globals():
    def _p4d__norm_multiset(U: object, *, m: int | None = None, name: str = "U") -> dict[str, int]:
        """Normalize a dict-like multiset {bitstring -> int multiplicity}.

        - Removes zero entries.
        - Coerces multiplicities to int >= 0.
        - Validates bitstrings.
        """
        if not isinstance(U, dict):
            raise TypeError(f"{name} must be a dict")
        out: dict[str, int] = {}
        for k, v in U.items():
            kk = _p4d__norm_bitstring(k, m=m, name=f"{name}.key")
            try:
                iv = int(v)
            except Exception:
                raise TypeError(f"{name}[{k!r}] multiplicity must be int")
            if iv < 0:
                raise ValueError(f"{name}[{k!r}] multiplicity must be >= 0")
            if iv == 0:
                continue
            out[kk] = iv
        return out


if "_p4d__canonical_U" not in globals():
    def _p4d__canonical_U(U: dict[str, int]) -> dict[str, int]:
        """Deterministic key order for display/serialization: descending lex."""
        try:
            items = sorted(U.items(), key=lambda kv: str(kv[0]), reverse=True)
        except Exception:
            items = list(U.items())
        return {k: int(v) for k, v in items}


if "_p4d__sigma" not in globals():
    def _p4d__sigma(multiset: dict[str, int], *, m: int) -> str:
        """XOR-sum Σ(multiset) in F2^m (returned as bitstring)."""
        acc = "0" * int(m)
        for k, v in (multiset or {}).items():
            if (int(v) & 1) == 0:
                continue
            acc = _p4d__xor_bits(acc, k)
        return acc


if "_p4d__mu_cmp" not in globals():
    def _p4d__mu_cmp(U0: dict[str, int], U1: dict[str, int]) -> int:
        """Compare µ(U0) vs µ(U1) in descending lex order.

        Returns:
          -1 if µ(U0) < µ(U1)
           0 if equal
          +1 if µ(U0) > µ(U1)

        This is equivalent to comparing the full 2^m vector when both are finite-support.
        """
        keys = set((U0 or {}).keys()) | set((U1 or {}).keys())
        for k in sorted(keys, reverse=True):
            a = int((U0 or {}).get(k, 0))
            b = int((U1 or {}).get(k, 0))
            if a < b:
                return -1
            if a > b:
                return 1
        return 0


if "_p4d__mu_strict_decreases" not in globals():
    def _p4d__mu_strict_decreases(U_before: dict[str, int], U_after: dict[str, int]) -> bool:
        return _p4d__mu_cmp(U_after, U_before) == -1


# ---- Rewrite rules R1–R4 --------------------------------------------------------

if "_p4d__R1_applicable" not in globals():
    def _p4d__R1_applicable(U: dict[str, int], *, y: str) -> list[str]:
        """Return keys c with multiplicity>0 and y·c == 0."""
        out: list[str] = []
        for c, mult in (U or {}).items():
            if int(mult) <= 0:
                continue
            if not _p4d__seen(y, c):
                out.append(c)
        out.sort(reverse=True)
        return out


if "_p4d__R1_apply" not in globals():
    def _p4d__R1_apply(U: dict[str, int], *, y: str, c: str) -> dict[str, int]:
        """R1: drop an unseen column-type key c."""
        if _p4d__seen(y, c):
            raise ValueError("R1 requires y·c=0")
        out = dict(U or {})
        out.pop(c, None)
        return out


if "_p4d__R1_saturate" not in globals():
    def _p4d__R1_saturate(U: dict[str, int], *, y: str) -> dict[str, int]:
        """Apply R1 exhaustively (drop all unseen keys)."""
        out = dict(U or {})
        for c in list(out.keys()):
            if not _p4d__seen(y, c):
                out.pop(c, None)
        return out


if "_p4d__R2_applicable" not in globals():
    def _p4d__R2_applicable(U: dict[str, int]) -> bool:
        """R2 applies iff some multiplicity > 1."""
        for _, mult in (U or {}).items():
            if int(mult) > 1:
                return True
        return False


if "_p4d__R2_apply" not in globals():
    def _p4d__R2_apply(U: dict[str, int]) -> dict[str, int]:
        """R2: multiplicities <- parity (mod 2), dropping zeros."""
        out: dict[str, int] = {}
        for c, mult in (U or {}).items():
            if (int(mult) & 1) == 1:
                out[c] = 1
        return out


# R3 catalog entry: {"w": <bitstring>, "S": {<bitstring>: <int>, ...}}
# S may also be given as a list of bitstrings (multiplicity 1 each)

if "_p4d__parse_R3_catalog" not in globals():
    def _p4d__parse_R3_catalog(obj: object, *, m: int) -> list[dict]:
        if obj is None:
            return []
        if not isinstance(obj, list):
            raise TypeError("R3 catalog must be a list")
        out: list[dict] = []
        for i, row in enumerate(obj):
            if not isinstance(row, dict):
                raise TypeError(f"R3[{i}] must be a dict")
            w = _p4d__norm_bitstring(row.get("w"), m=m, name=f"R3[{i}].w")
            S_raw = row.get("S")
            if isinstance(S_raw, list):
                # list of bitstrings
                S: dict[str, int] = {}
                for j, s in enumerate(S_raw):
                    ss = _p4d__norm_bitstring(s, m=m, name=f"R3[{i}].S[{j}]")
                    S[ss] = int(S.get(ss, 0)) + 1
            elif isinstance(S_raw, dict):
                S = _p4d__norm_multiset(S_raw, m=m, name=f"R3[{i}].S")
            else:
                raise TypeError(f"R3[{i}].S must be list or dict")

            out.append({"w": w, "S": S})
        return out


if "_p4d__R3_validate_entry" not in globals():
    def _p4d__R3_validate_entry(entry: dict, *, m: int) -> tuple[bool, str | None]:
        """Validate R3 entry against the frozen acyclic + sum constraints."""
        try:
            w = entry.get("w")
            S = entry.get("S")
            if not _p4d__is_bitstring(w, m=m):
                return False, "w not a bitstring"
            if not isinstance(S, dict):
                return False, "S not a dict"
            # sum constraint: Σ(S) == w
            if _p4d__sigma(S, m=m) != w:
                return False, "sigma(S) != w"
            # acyclic: w not in S, and all s < w in lex order (descending lex)
            if w in S and int(S.get(w, 0)) > 0:
                return False, "acyclic violated: w appears in S"
            for s, mult in S.items():
                if int(mult) <= 0:
                    continue
                if s >= w:
                    return False, "acyclic violated: s >= w"
            return True, None
        except Exception as e:
            return False, str(e)


if "_p4d__R3_applicable" not in globals():
    def _p4d__R3_applicable(U: dict[str, int], *, catalog: list[dict]) -> list[int]:
        """Return indices i where template i applies (U[w] > 0)."""
        out: list[int] = []
        for i, ent in enumerate(catalog or []):
            w = ent.get("w")
            if not isinstance(w, str):
                continue
            if int((U or {}).get(w, 0)) > 0:
                out.append(i)
        return out


if "_p4d__R3_apply" not in globals():
    def _p4d__R3_apply(U: dict[str, int], *, entry: dict) -> dict[str, int]:
        """R3: replace one instance of w by multiset S."""
        w = entry.get("w")
        S = entry.get("S")
        if not isinstance(w, str) or not isinstance(S, dict):
            raise TypeError("R3 entry malformed")
        if int((U or {}).get(w, 0)) <= 0:
            raise ValueError("R3 not applicable: w multiplicity is 0")
        out = dict(U or {})
        out[w] = int(out.get(w, 0)) - 1
        if out[w] <= 0:
            out.pop(w, None)
        for s, mult in (S or {}).items():
            out[s] = int(out.get(s, 0)) + int(mult)
            if out[s] <= 0:
                out.pop(s, None)
        return out


# R4 catalog entry: {"L": {..}, "Lp": {..}} (Lp = L')

if "_p4d__parse_R4_catalog" not in globals():
    def _p4d__parse_R4_catalog(obj: object, *, m: int) -> list[dict]:
        if obj is None:
            return []
        if not isinstance(obj, list):
            raise TypeError("R4 catalog must be a list")
        out: list[dict] = []
        for i, row in enumerate(obj):
            if not isinstance(row, dict):
                raise TypeError(f"R4[{i}] must be a dict")
            # accept a few spellings
            L_raw = row.get("L")
            Lp_raw = row.get("Lp")
            if Lp_raw is None:
                Lp_raw = row.get("L_prime")
            if Lp_raw is None:
                Lp_raw = row.get("L'")
            if not isinstance(L_raw, dict) or not isinstance(Lp_raw, dict):
                raise TypeError(f"R4[{i}] must have dict L and dict Lp")
            L = _p4d__norm_multiset(L_raw, m=m, name=f"R4[{i}].L")
            Lp = _p4d__norm_multiset(Lp_raw, m=m, name=f"R4[{i}].Lp")
            out.append({"L": L, "Lp": Lp})
        return out


if "_p4d__R4_validate_entry" not in globals():
    def _p4d__R4_validate_entry(entry: dict, *, m: int) -> tuple[bool, str | None]:
        """Validate R4 entry against frozen constraints.

        Required:
          - sigma(L) == sigma(Lp)  (parity-preserving)
          - max(L) > max(Lp)       (µ decrease)
        """
        try:
            L = entry.get("L")
            Lp = entry.get("Lp")
            if not isinstance(L, dict) or not isinstance(Lp, dict):
                return False, "L/Lp malformed"
            if _p4d__sigma(L, m=m) != _p4d__sigma(Lp, m=m):
                return False, "sigma(L) != sigma(Lp)"
            maxL = max(L.keys()) if L else None
            maxLp = max(Lp.keys()) if Lp else None
            # Empty template is forbidden (would not decrease)
            if maxL is None:
                return False, "empty L"
            if maxLp is None:
                return False, "empty Lp"
            if not (maxL > maxLp):
                return False, "mu decrease violated: max(L) <= max(Lp)"
            return True, None
        except Exception as e:
            return False, str(e)


if "_p4d__R4_applicable" not in globals():
    def _p4d__R4_applicable(U: dict[str, int], *, catalog: list[dict]) -> list[int]:
        """Return indices i where chord L ⊆ U (multiplicity-wise)."""
        out: list[int] = []
        for i, ent in enumerate(catalog or []):
            L = ent.get("L")
            if not isinstance(L, dict):
                continue
            ok = True
            for c, need in L.items():
                if int((U or {}).get(c, 0)) < int(need):
                    ok = False
                    break
            if ok:
                out.append(i)
        return out


if "_p4d__R4_apply" not in globals():
    def _p4d__R4_apply(U: dict[str, int], *, entry: dict) -> dict[str, int]:
        """R4: apply chord template L -> L'."""
        L = entry.get("L")
        Lp = entry.get("Lp")
        if not isinstance(L, dict) or not isinstance(Lp, dict):
            raise TypeError("R4 entry malformed")
        out = dict(U or {})
        # subtract L
        for c, need in (L or {}).items():
            out[c] = int(out.get(c, 0)) - int(need)
            if out[c] <= 0:
                out.pop(c, None)
        # add Lp
        for c, add in (Lp or {}).items():
            out[c] = int(out.get(c, 0)) + int(add)
            if out[c] <= 0:
                out.pop(c, None)
        return out


# ---- Normalizer / simulator -----------------------------------------------------

if "_p4d__step" not in globals():
    def _p4d__step(
        U: dict[str, int],
        *,
        y: str,
        r3_catalog: list[dict],
        r4_catalog: list[dict],
        schedule: str = "phase_u_informal",
        rng=None,
    ) -> tuple[dict[str, int], dict | None]:
        """Perform one rewrite step according to a schedule.

        Schedules:
          - "phase_u_informal": apply R1-saturate, then R2, then one R3, then one R4.
          - "random": pick one applicable instance among {R1,R2,R3,R4} uniformly.
          - "r3_first": prefer R3 over R1 over R2 over R4.
          - "r4_first": prefer R4 over R1 over R2 over R3.

        Returns (U', meta) where meta describes the rule applied or None if no rule applies.
        """
        # Always work on a copy.
        U0 = dict(U or {})

        if schedule == "phase_u_informal":
            # R1-saturate
            U1 = _p4d__R1_saturate(U0, y=y)
            if U1 != U0:
                return U1, {"rule": "R1*", "detail": "saturate"}
            # R2
            if _p4d__R2_applicable(U0):
                U2 = _p4d__R2_apply(U0)
                if U2 != U0:
                    return U2, {"rule": "R2", "detail": "parity"}
            # R3: choose lex-largest w among applicable templates
            app3 = _p4d__R3_applicable(U0, catalog=r3_catalog)
            if app3:
                # choose template whose head w is lex-largest
                best_i = max(app3, key=lambda i: str((r3_catalog[i] or {}).get("w") or ""))
                ent = r3_catalog[best_i]
                U3 = _p4d__R3_apply(U0, entry=ent)
                return U3, {"rule": "R3", "i": int(best_i), "w": ent.get("w")}
            # R4: choose template with lex-largest max(L)
            app4 = _p4d__R4_applicable(U0, catalog=r4_catalog)
            if app4:
                best_i = max(app4, key=lambda i: max((r4_catalog[i] or {}).get("L") or {"":0}).keys())
                ent = r4_catalog[best_i]
                U4 = _p4d__R4_apply(U0, entry=ent)
                return U4, {"rule": "R4", "i": int(best_i)}

            return U0, None

        # --- Priority schedules (single-instance choice) ---
        if schedule in {"r3_first", "r4_first"}:
            order = ["R3", "R1", "R2", "R4"] if schedule == "r3_first" else ["R4", "R1", "R2", "R3"]
            for r in order:
                if r == "R1":
                    app1 = _p4d__R1_applicable(U0, y=y)
                    if app1:
                        c = app1[0]
                        return _p4d__R1_apply(U0, y=y, c=c), {"rule": "R1", "c": c}
                if r == "R2":
                    if _p4d__R2_applicable(U0):
                        return _p4d__R2_apply(U0), {"rule": "R2"}
                if r == "R3":
                    app3 = _p4d__R3_applicable(U0, catalog=r3_catalog)
                    if app3:
                        # pick lex-largest w
                        best_i = max(app3, key=lambda i: str((r3_catalog[i] or {}).get("w") or ""))
                        ent = r3_catalog[best_i]
                        return _p4d__R3_apply(U0, entry=ent), {"rule": "R3", "i": int(best_i), "w": ent.get("w")}
                if r == "R4":
                    app4 = _p4d__R4_applicable(U0, catalog=r4_catalog)
                    if app4:
                        best_i = max(app4, key=lambda i: max((r4_catalog[i] or {}).get("L") or {"":0}).keys())
                        ent = r4_catalog[best_i]
                        return _p4d__R4_apply(U0, entry=ent), {"rule": "R4", "i": int(best_i)}
            return U0, None

        # --- random schedule ---
        if schedule == "random":
            if rng is None:
                import random as _random
                rng = _random.Random(0)

            candidates: list[tuple[str, object]] = []
            app1 = _p4d__R1_applicable(U0, y=y)
            for c in app1:
                candidates.append(("R1", c))
            if _p4d__R2_applicable(U0):
                candidates.append(("R2", None))
            for i in _p4d__R3_applicable(U0, catalog=r3_catalog):
                candidates.append(("R3", int(i)))
            for i in _p4d__R4_applicable(U0, catalog=r4_catalog):
                candidates.append(("R4", int(i)))

            if not candidates:
                return U0, None

            rule, payload = rng.choice(candidates)
            if rule == "R1":
                c = str(payload)
                return _p4d__R1_apply(U0, y=y, c=c), {"rule": "R1", "c": c}
            if rule == "R2":
                return _p4d__R2_apply(U0), {"rule": "R2"}
            if rule == "R3":
                i = int(payload)
                ent = r3_catalog[i]
                return _p4d__R3_apply(U0, entry=ent), {"rule": "R3", "i": i, "w": ent.get("w")}
            if rule == "R4":
                i = int(payload)
                ent = r4_catalog[i]
                return _p4d__R4_apply(U0, entry=ent), {"rule": "R4", "i": i}

            return U0, None

        raise ValueError(f"Unknown schedule: {schedule!r}")


if "_p4d__normalize" not in globals():
    def _p4d__normalize(
        U: dict[str, int],
        *,
        y: str,
        r3_catalog: list[dict],
        r4_catalog: list[dict],
        schedule: str = "phase_u_informal",
        seed: int | None = None,
        step_limit: int = 20000,
        enforce_mu: bool = True,
    ) -> dict:
        """Run rewrites to normal form; returns an object with trace and invariants."""
        import random as _random

        U0 = dict(U or {})
        rng = _random.Random(int(seed or 0))

        trace: list[dict] = []
        cur = dict(U0)

        for step in range(int(step_limit)):
            nxt, meta = _p4d__step(cur, y=y, r3_catalog=r3_catalog, r4_catalog=r4_catalog, schedule=schedule, rng=rng)
            if meta is None:
                break
            if enforce_mu:
                if not _p4d__mu_strict_decreases(cur, nxt):
                    # Provide the local diff for debugging.
                    meta2 = dict(meta)
                    meta2["mu_violation"] = True
                    meta2["U_before"] = _p4d__canonical_U(cur)
                    meta2["U_after"] = _p4d__canonical_U(nxt)
                    raise RuntimeError(f"µ did not strictly decrease under step: {meta2}")
            trace.append({"step": step, **(meta or {})})
            cur = dict(nxt)
        else:
            raise RuntimeError(f"Step limit exceeded ({int(step_limit)}), possible non-termination")

        # Observations (parity-facing surface): R1-saturate then R2 then restrict to seen keys.
        seen_only = _p4d__R1_saturate(cur, y=y)
        seen_par = _p4d__R2_apply(seen_only)
        seen_par = {k: v for k, v in seen_par.items() if _p4d__seen(y, k)}

        # Canonical serialization for reproducibility.
        core_payload = {
            "y": str(y),
            "U_nf": _p4d__canonical_U(cur),
            "seen_par": _p4d__canonical_U(seen_par),
        }
        try:
            txt = _json.dumps(core_payload, separators=(",", ":"), sort_keys=False, ensure_ascii=True, allow_nan=False)
            import hashlib as _hashlib
            h = _hashlib.sha256(txt.encode("utf-8")).hexdigest()
        except Exception:
            txt = None
            h = None

        return {
            "U_nf": cur,
            "U_nf_canon": _p4d__canonical_U(cur),
            "seen_par": seen_par,
            "seen_par_canon": _p4d__canonical_U(seen_par),
            "trace": trace,
            "steps": len(trace),
            "schedule": schedule,
            "seed": int(seed or 0),
            "sha256": h,
            "canonical_json": txt,
        }


# ---- D1–D4 micro-checks ---------------------------------------------------------

if "_p4d__check_D1" not in globals():
    def _p4d__check_D1(
        *,
        U: dict[str, int],
        y: str,
        r3_catalog: list[dict],
        r4_catalog: list[dict],
    ) -> dict:
        """D1: R1–R3 order sanity.

        Compare two priority schedules that differ only in whether R3 is allowed to fire
        before R1 deletions.
        """
        out_a = _p4d__normalize(U, y=y, r3_catalog=r3_catalog, r4_catalog=r4_catalog, schedule="r3_first")
        out_b = _p4d__normalize(U, y=y, r3_catalog=r3_catalog, r4_catalog=r4_catalog, schedule="phase_u_informal")
        ok = (out_a.get("seen_par_canon") == out_b.get("seen_par_canon"))
        return {
            "check": "D1",
            "status": "PASS" if ok else "FAIL",
            "same_seen_par": ok,
            "r3_first": {"sha256": out_a.get("sha256"), "seen_par": out_a.get("seen_par_canon"), "steps": out_a.get("steps")},
            "phase_u": {"sha256": out_b.get("sha256"), "seen_par": out_b.get("seen_par_canon"), "steps": out_b.get("steps")},
        }


if "_p4d__check_D2" not in globals():
    def _p4d__check_D2(
        *,
        U: dict[str, int],
        y: str,
        r3_catalog: list[dict],
    ) -> dict:
        """D2: R2–R3 commute on parity (single-step parity vector)."""
        # Find a single applicable R3 template.
        app = _p4d__R3_applicable(U, catalog=r3_catalog)
        if not app:
            return {"check": "D2", "status": "SKIP", "reason": "no R3 applicable"}
        i = int(app[0])
        ent = r3_catalog[i]
        # route A: R2 then R3
        Ua = _p4d__R2_apply(U)
        Ua = _p4d__R3_apply(Ua, entry=ent) if int(Ua.get(ent.get("w"), 0)) > 0 else Ua
        Ua = _p4d__R2_apply(Ua)
        # route B: R3 then R2
        Ub = _p4d__R3_apply(U, entry=ent)
        Ub = _p4d__R2_apply(Ub)
        ok = (_p4d__canonical_U(Ua) == _p4d__canonical_U(Ub))
        return {
            "check": "D2",
            "status": "PASS" if ok else "FAIL",
            "template_i": i,
            "w": ent.get("w"),
            "parity_after_R2_R3_R2": _p4d__canonical_U(Ua),
            "parity_after_R3_R2": _p4d__canonical_U(Ub),
        }


if "_p4d__check_D3" not in globals():
    def _p4d__check_D3(
        *,
        U: dict[str, int],
        y: str,
        r3_catalog: list[dict],
        r4_catalog: list[dict],
    ) -> dict:
        """D3: chord overlaps parity-controlled (schedule independence)."""
        # Compare two schedules that swap R4/R3 priority.
        out_a = _p4d__normalize(U, y=y, r3_catalog=r3_catalog, r4_catalog=r4_catalog, schedule="r4_first")
        out_b = _p4d__normalize(U, y=y, r3_catalog=r3_catalog, r4_catalog=r4_catalog, schedule="r3_first")
        ok = (out_a.get("seen_par_canon") == out_b.get("seen_par_canon"))
        return {
            "check": "D3",
            "status": "PASS" if ok else "FAIL",
            "same_seen_par": ok,
            "r4_first": {"sha256": out_a.get("sha256"), "seen_par": out_a.get("seen_par_canon"), "steps": out_a.get("steps")},
            "r3_first": {"sha256": out_b.get("sha256"), "seen_par": out_b.get("seen_par_canon"), "steps": out_b.get("steps")},
        }


if "_p4d__check_D4" not in globals():
    def _p4d__check_D4(
        *,
        toys: list[dict],
        r3_catalog: list[dict],
        r4_catalog: list[dict],
        runs: int = 25,
        seed0: int = 0,
    ) -> dict:
        """D4: unique core on toys (random schedules converge to same normal form)."""
        results: list[dict] = []
        all_ok = True
        for t in toys or []:
            name = str((t or {}).get("name") or "(unnamed)")
            y = str((t or {}).get("y") or "")
            U = (t or {}).get("U") or {}
            try:
                m = len(y)
                y2 = _p4d__norm_bitstring(y, m=m, name=f"toy[{name}].y")
                U2 = _p4d__norm_multiset(U, m=m, name=f"toy[{name}].U")
            except Exception as e:
                results.append({"name": name, "status": "FAIL", "reason": f"invalid toy: {e}"})
                all_ok = False
                continue

            # Deterministic baseline
            base = _p4d__normalize(U2, y=y2, r3_catalog=r3_catalog, r4_catalog=r4_catalog, schedule="phase_u_informal")
            base_h = base.get("sha256")

            ok = True
            mismatches: list[dict] = []
            for r in range(int(runs)):
                out = _p4d__normalize(
                    U2,
                    y=y2,
                    r3_catalog=r3_catalog,
                    r4_catalog=r4_catalog,
                    schedule="random",
                    seed=int(seed0) + int(r),
                )
                if out.get("sha256") != base_h:
                    ok = False
                    mismatches.append({
                        "run": r,
                        "seed": int(seed0) + int(r),
                        "sha256": out.get("sha256"),
                        "seen_par": out.get("seen_par_canon"),
                        "steps": out.get("steps"),
                    })
                    # Keep going to surface multiple mismatch patterns.

            results.append({
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "baseline_sha256": base_h,
                "baseline_seen_par": base.get("seen_par_canon"),
                "runs": int(runs),
                "mismatches": mismatches,
            })
            if not ok:
                all_ok = False

        return {
            "check": "D4",
            "status": "PASS" if all_ok else "FAIL",
            "runs_per_toy": int(runs),
            "seed0": int(seed0),
            "toys": results,
        }


# ---- Streamlit UI panel ---------------------------------------------------------

try:
    # This section is safe to run even if Streamlit isn't present (e.g., imported as a module).
    import streamlit as st

    with st.expander("Phase 4 — Step D rewrite calculus (compression layer)", expanded=False):
        st.caption(
            "Implements the frozen rewrite rules R1–R4 + termination (µ) + D1–D4 micro-checks. "
            "Catalogs are supplied as JSON (payload). This tool is observational only; it does not affect overlap outputs."
        )

        st.markdown("### Inputs")

        default_suite = {
            "schema": "stepD_suite.v0",
            "note": "Built-in minimal toys exercising R1/R2/R3/R4 order-independence. Replace with your own T-suite if desired.",
            "r3_catalog": [
                {"w": "110", "S": {"100": 1, "010": 1}},
                {"w": "111", "S": {"001": 1, "110": 1}},
            ],
            "r4_catalog": [
                {"L": {"001": 1, "110": 1, "010": 1}, "Lp": {"001": 1, "100": 1, "000": 1}},
            ],
            "toys": [
                {
                    "name": "D1_R1_R3_overlap",
                    "y": "001",
                    "U": {"110": 1},
                },
                {
                    "name": "D2_R2_R3_commute",
                    "y": "001",
                    "U": {"111": 1, "001": 2},
                },
                {
                    "name": "D3_chord_overlap",
                    "y": "001",
                    "U": {"001": 1, "110": 1, "010": 1},
                },
            ],
        }

        suite_txt = st.text_area(
            "Suite JSON (catalogs + toys)",
            value=_json.dumps(default_suite, indent=2, sort_keys=False),
            height=360,
            key=f"p4d_suite_json_{st.session_state.get('_ui_nonce','00000000')}",
        )

        parsed = None
        parse_err = None
        try:
            parsed = _json.loads(suite_txt) if suite_txt else None
        except Exception as e:
            parse_err = str(e)

        if parse_err:
            st.error(f"Invalid JSON: {parse_err}")
            parsed = None

        # Allow an ad-hoc single-instance runner too.
        st.markdown("### Single instance")
        colA, colB = st.columns(2)
        with colA:
            y_in = st.text_input(
                "witness y (bitstring)",
                value="001",
                key=f"p4d_y_{st.session_state.get('_ui_nonce','00000000')}",
            )
        with colB:
            U_in_txt = st.text_input(
                "U (dict JSON: {bitstring: multiplicity})",
                value=_json.dumps({"001": 1, "110": 1, "010": 1}),
                key=f"p4d_U_{st.session_state.get('_ui_nonce','00000000')}",
            )

        schedule = st.selectbox(
            "Schedule",
            options=["phase_u_informal", "r3_first", "r4_first", "random"],
            index=0,
            key=f"p4d_sched_{st.session_state.get('_ui_nonce','00000000')}",
        )
        seed = st.number_input(
            "seed (for random schedule)",
            min_value=0,
            max_value=10_000,
            value=0,
            step=1,
            key=f"p4d_seed_{st.session_state.get('_ui_nonce','00000000')}",
        )

        if st.button("Run normalize(U)", key=f"p4d_run_one_{st.session_state.get('_ui_nonce','00000000')}"):
            try:
                y = _p4d__norm_bitstring(y_in, name="y")
                m = len(y)
                U0 = _p4d__norm_multiset(_json.loads(U_in_txt), m=m)

                # catalogs from suite json if present
                r3_raw = (parsed or {}).get("r3_catalog") if isinstance(parsed, dict) else []
                r4_raw = (parsed or {}).get("r4_catalog") if isinstance(parsed, dict) else []
                r3 = _p4d__parse_R3_catalog(r3_raw, m=m)
                r4 = _p4d__parse_R4_catalog(r4_raw, m=m)

                # validate catalogs
                r3_bad = []
                for i, ent in enumerate(r3):
                    ok, reason = _p4d__R3_validate_entry(ent, m=m)
                    if not ok:
                        r3_bad.append({"i": i, "reason": reason, "entry": ent})
                r4_bad = []
                for i, ent in enumerate(r4):
                    ok, reason = _p4d__R4_validate_entry(ent, m=m)
                    if not ok:
                        r4_bad.append({"i": i, "reason": reason, "entry": ent})

                if r3_bad:
                    st.warning({"R3_catalog_invalid": r3_bad})
                if r4_bad:
                    st.warning({"R4_catalog_invalid": r4_bad})

                out = _p4d__normalize(U0, y=y, r3_catalog=r3, r4_catalog=r4, schedule=schedule, seed=int(seed))
                st.markdown("**normalize(U) output**")
                st.json({
                    "sha256": out.get("sha256"),
                    "steps": out.get("steps"),
                    "schedule": out.get("schedule"),
                    "seed": out.get("seed"),
                    "U_nf": out.get("U_nf_canon"),
                    "seen_par": out.get("seen_par_canon"),
                })
                with st.expander("Trace", expanded=False):
                    st.json(out.get("trace"))
            except Exception as e:
                st.error(f"normalize(U) failed: {e}")

        st.markdown("---")
        st.markdown("### Micro-checks D1–D4")

        if st.button("Run D1–D4 on suite.toys", key=f"p4d_run_suite_{st.session_state.get('_ui_nonce','00000000')}"):
            try:
                if not isinstance(parsed, dict):
                    raise ValueError("Suite JSON must be an object")

                toys = parsed.get("toys") if isinstance(parsed.get("toys"), list) else []
                if not toys:
                    raise ValueError("suite.toys is empty")

                # Determine m from first toy's y.
                y0 = str((toys[0] or {}).get("y") or "")
                y0 = _p4d__norm_bitstring(y0, name="toys[0].y")
                m = len(y0)

                r3 = _p4d__parse_R3_catalog(parsed.get("r3_catalog") or [], m=m)
                r4 = _p4d__parse_R4_catalog(parsed.get("r4_catalog") or [], m=m)

                # Validate catalogs once.
                cat_warn = {"R3_invalid": [], "R4_invalid": []}
                for i, ent in enumerate(r3):
                    ok, reason = _p4d__R3_validate_entry(ent, m=m)
                    if not ok:
                        cat_warn["R3_invalid"].append({"i": i, "reason": reason, "entry": ent})
                for i, ent in enumerate(r4):
                    ok, reason = _p4d__R4_validate_entry(ent, m=m)
                    if not ok:
                        cat_warn["R4_invalid"].append({"i": i, "reason": reason, "entry": ent})

                if cat_warn["R3_invalid"] or cat_warn["R4_invalid"]:
                    st.warning(cat_warn)

                # Run per-toy D1-D3, then global D4
                per_toy: list[dict] = []
                for t in toys:
                    name = str((t or {}).get("name") or "(unnamed)")
                    y = _p4d__norm_bitstring((t or {}).get("y") or "", m=m, name=f"toy[{name}].y")
                    U0 = _p4d__norm_multiset((t or {}).get("U") or {}, m=m, name=f"toy[{name}].U")
                    per_toy.append({
                        "name": name,
                        "D1": _p4d__check_D1(U=U0, y=y, r3_catalog=r3, r4_catalog=r4),
                        "D2": _p4d__check_D2(U=U0, y=y, r3_catalog=r3),
                        "D3": _p4d__check_D3(U=U0, y=y, r3_catalog=r3, r4_catalog=r4),
                    })

                d4_runs = st.number_input(
                    "D4 random runs per toy",
                    min_value=1,
                    max_value=200,
                    value=25,
                    step=1,
                    key=f"p4d_d4_runs_{st.session_state.get('_ui_nonce','00000000')}",
                )
                d4_seed0 = st.number_input(
                    "D4 seed0",
                    min_value=0,
                    max_value=10_000,
                    value=0,
                    step=1,
                    key=f"p4d_d4_seed0_{st.session_state.get('_ui_nonce','00000000')}",
                )

                d4 = _p4d__check_D4(toys=toys, r3_catalog=r3, r4_catalog=r4, runs=int(d4_runs), seed0=int(d4_seed0))

                st.markdown("**Suite results**")
                st.json({
                    "D4": d4,
                    "per_toy": per_toy,
                })
            except Exception as e:
                st.error(f"Suite run failed: {e}")

except Exception:
    # If streamlit isn't available (e.g., running unit tests), ignore panel wiring.
    pass


# =================== /Phase 4 — Step D rewrite calculus (compression layer) ===================


# ───────────────────────────── Phase 5 — strict-core-v1 harness ─────────────────────────────
# Mode B wiring: add a minimal, projection-first harness that can:
#   • locate the strict-core-v1 anchor fixtures by SHA256/12,
#   • compute Grid/Ker/Fence/Cone/Wiggle + same-H Echo sweeps,
#   • write logs/reports/strict_core_v1/{registry.csv,phase5_acceptance.json},
#   • keep projected-mode as an explicit controlled relaxation (ker-guard is logged, not enforced).

def _phase5_sha25612_path(p: _Path) -> str:
    """SHA256/12 over file bytes (matches strict_core_v1 appendix)."""
    try:
        h = _hashlib.sha256()
        with _Path(p).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return ""


def _phase5_support_sig(mat: list[list[int]]) -> dict:
    """Return a stable support signature for a bitmatrix (row/col + 1-positions)."""
    mat = _time_tau_t0_norm_bitmatrix(mat)
    r = len(mat)
    c = len(mat[0]) if (r and mat[0]) else 0
    ones: list[list[int]] = []
    for i, row in enumerate(mat):
        if not isinstance(row, list):
            continue
        for j, v in enumerate(row):
            if (int(v) & 1) == 1:
                ones.append([int(i), int(j)])
    return {"shape": [int(r), int(c)], "ones": ones}


def _phase5_support_hash(mat: list[list[int]]) -> str:
    """SHA256 over CanonicalJSON(support_sig)."""
    try:
        sig = _phase5_support_sig(mat)
        txt = _json.dumps(sig, separators=(",", ":"), sort_keys=True, ensure_ascii=True, allow_nan=False)
        return _hashlib.sha256(txt.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def _phase5_find_json_by_sha12(search_dirs: list[_Path], sha12: str) -> tuple[_Path | None, list[str]]:
    """Return (path, notes). If multiple matches, pick lexicographically smallest and note it."""
    sha12 = str(sha12 or "").strip().lower()
    notes: list[str] = []
    if not sha12:
        return None, ["EMPTY_SHA12"]
    hits: list[_Path] = []
    for base in search_dirs:
        try:
            base = _Path(base)
        except Exception:
            continue
        if not base.exists():
            continue
        try:
            for fp in base.rglob("*.json"):
                try:
                    if _phase5_sha25612_path(fp) == sha12:
                        hits.append(fp)
                except Exception:
                    continue
        except Exception:
            continue

    if not hits:
        return None, [f"NOT_FOUND:{sha12}"]
    hits = sorted(set(_Path(p).resolve() for p in hits), key=lambda p: str(p))
    if len(hits) > 1:
        notes.append(f"MULTI_MATCH:{sha12}:{len(hits)}")
        notes.append("PICKED:" + str(hits[0]))
    return hits[0], notes


def _phase5_guard_grid(bB: dict, bC: dict) -> dict:
    """Grid guard: d3*C3 == C2*d3 when possible; records assumption if C2 missing."""
    d3 = _time_tau_t0_norm_bitmatrix((bB or {}).get("3") or [])
    C3 = _time_tau_t0_norm_bitmatrix((bC or {}).get("3") or [])
    if not d3 or not d3[0] or not C3 or not C3[0]:
        return {"ok": False, "reason": "MISSING_D3_OR_C3", "assumed_C2_identity": None}

    n2 = len(d3)
    n3 = len(d3[0])

    C2_raw = (bC or {}).get("2")
    assumed = False
    if C2_raw is None:
        C2 = _svr_eye(n2)
        assumed = True
    else:
        C2 = _time_tau_t0_norm_bitmatrix(C2_raw or [])
        if len(C2) != n2 or (n2 and len(C2[0]) != n2):
            # Fall back to identity, but record drift explicitly.
            C2 = _svr_eye(n2)
            assumed = True

    # Shape sanity for muls
    if not _svr_shape_ok_for_mul(d3, C3):
        return {"ok": False, "reason": "SHAPE_D3C3", "assumed_C2_identity": assumed}
    if not _svr_shape_ok_for_mul(C2, d3):
        return {"ok": False, "reason": "SHAPE_C2D3", "assumed_C2_identity": assumed}

    left = _svr_matmul_gf2(d3, C3)      # n2 x n3
    right = _svr_matmul_gf2(C2, d3)     # n2 x n3
    ok = _svr_is_zero(_svr_xor_gf2(left, right))
    return {"ok": bool(ok), "reason": None if ok else "GRID_MISMATCH", "assumed_C2_identity": assumed}


def _phase5_guard_ker(bB: dict, bC: dict) -> dict:
    """Ker-guard (v1 notes): never place supp(C−I) on ker(d3) under strict.

    Operationalization (basis-level, conservative):
      - Identify kernel *basis columns* j where column j of d3 is identically zero.
      - Require (C3−I3) has no 1s in those columns.

    This matches the district-card language "dimker=1 (col 3)" for D2/D3. 
    """
    d3 = _time_tau_t0_norm_bitmatrix((bB or {}).get("3") or [])
    C3 = _time_tau_t0_norm_bitmatrix((bC or {}).get("3") or [])
    if not d3 or not d3[0] or not C3 or not C3[0]:
        return {"ok": False, "reason": "MISSING_D3_OR_C3", "ker_cols": []}
    n3 = len(d3[0])
    I3 = _svr_eye(n3)
    if len(C3) != n3 or (n3 and len(C3[0]) != n3):
        return {"ok": False, "reason": "C3_NOT_SQUARE", "ker_cols": []}

    diff = _svr_xor_gf2(C3, I3)  # C3 - I3 over F2
    ker_cols: list[int] = []
    for j in range(n3):
        col_all_zero = True
        for i in range(len(d3)):
            try:
                if (int(d3[i][j]) & 1) != 0:
                    col_all_zero = False
                    break
            except Exception:
                continue
        if col_all_zero:
            ker_cols.append(j)

    bad = []
    for j in ker_cols:
        for i in range(n3):
            if (int(diff[i][j]) & 1) == 1:
                bad.append((i, j))
                break

    ok = len(bad) == 0
    return {
        "ok": bool(ok),
        "reason": None if ok else "SUPP_C_DIFF_TOUCHES_KER_COL",
        "ker_cols": [int(j) for j in ker_cols],
        "bad_cells": [[int(i), int(j)] for (i, j) in bad][:16],
    }


def _phase5_guard_cone(bC: dict) -> dict:
    """Cone guard (bottom-row family): C3 differs from identity only in the bottom row.

    For these v1 anchor fixtures, we treat this as the concrete predicate:
      ∀ i < n3-1: (C3 ⊕ I3)[i,*] == 0.
    """
    C3 = _time_tau_t0_norm_bitmatrix((bC or {}).get("3") or [])
    if not C3 or not C3[0]:
        return {"ok": False, "reason": "MISSING_C3"}
    n3 = len(C3)
    if any(len(r) != n3 for r in C3):
        return {"ok": False, "reason": "C3_NOT_SQUARE"}
    I3 = _svr_eye(n3)
    diff = _svr_xor_gf2(C3, I3)
    # all rows except bottom must be zero
    for i in range(n3 - 1):
        if any((int(v) & 1) == 1 for v in diff[i]):
            return {"ok": False, "reason": "NONBOTTOM_SUPPORT", "first_bad_row": int(i)}
    return {"ok": True, "reason": None}


def _phase5_guard_wiggle(bB: dict, bC: dict, bH: dict) -> dict:
    """Wiggle guard (entrywise strict): verify H2 d3 = C3 − I3 over F2. """
    d3 = _time_tau_t0_norm_bitmatrix((bB or {}).get("3") or [])
    C3 = _time_tau_t0_norm_bitmatrix((bC or {}).get("3") or [])
    H2 = _time_tau_t0_norm_bitmatrix((bH or {}).get("2") or [])
    if not d3 or not d3[0] or not C3 or not C3[0] or not H2 or not H2[0]:
        return {"ok": False, "reason": "MISSING_D3_OR_C3_OR_H2"}
    if not _svr_shape_ok_for_mul(H2, d3):
        return {"ok": False, "reason": "SHAPE_H2D3"}
    n3 = len(d3[0])
    if len(C3) != n3 or (n3 and len(C3[0]) != n3):
        return {"ok": False, "reason": "C3_NOT_SQUARE"}
    I3 = _svr_eye(n3)
    left = _svr_matmul_gf2(H2, d3)
    right = _svr_xor_gf2(C3, I3)
    ok = _svr_is_zero(_svr_xor_gf2(left, right))
    return {"ok": bool(ok), "reason": None if ok else "WIGGLE_RESIDUAL_NONZERO"}


def _phase5_guard_fence(bB: dict, bC: dict, bH: dict, bU: dict) -> dict:
    """Fence guard (carrier-local): keep supp(C) and supp(H) inside the chosen U. 

    Operationalization (best-effort):
      - Interpret U["3"] as either a basis mask vector (len=n3) or a (n3×n3) entry-mask.
      - Interpret U["2"] as a basis mask vector (len=n2) when present.
      - Enforce:
          * (C3 ⊕ I3) has support only where U allows (mask-based), OR
          * for basis-mask: rows/cols outside U are identity; H2 has no output outside U3;
            and if U2 exists, H2 has no input from outside U2.
    """
    d3 = _time_tau_t0_norm_bitmatrix((bB or {}).get("3") or [])
    C3 = _time_tau_t0_norm_bitmatrix((bC or {}).get("3") or [])
    H2 = _time_tau_t0_norm_bitmatrix((bH or {}).get("2") or [])
    U3_raw = (bU or {}).get("3")
    U2_raw = (bU or {}).get("2")

    if not d3 or not d3[0] or not C3 or not C3[0] or not H2 or not H2[0]:
        return {"ok": False, "reason": "MISSING_CORE_BLOCKS"}

    n2 = len(d3)
    n3 = len(d3[0])
    if len(C3) != n3 or (n3 and len(C3[0]) != n3):
        return {"ok": False, "reason": "C3_NOT_SQUARE"}

    I3 = _svr_eye(n3)
    diff = _svr_xor_gf2(C3, I3)

    # Parse U3
    U3_vec: list[int] | None = None
    U3_mask: list[list[int]] | None = None
    try:
        if isinstance(U3_raw, list) and (not U3_raw or isinstance(U3_raw[0], (int, bool))):
            U3_vec = [int(v) & 1 for v in U3_raw]
        elif isinstance(U3_raw, list) and U3_raw and isinstance(U3_raw[0], list):
            # Could be a 1×n vector or n×n mask
            if len(U3_raw) == 1 and isinstance(U3_raw[0], list):
                U3_vec = [int(v) & 1 for v in U3_raw[0]]
            else:
                U3_mask = _time_tau_t0_norm_bitmatrix(U3_raw)
    except Exception:
        U3_vec = None
        U3_mask = None

    # Parse U2 vector
    U2_vec: list[int] | None = None
    try:
        if isinstance(U2_raw, list) and (not U2_raw or isinstance(U2_raw[0], (int, bool))):
            U2_vec = [int(v) & 1 for v in U2_raw]
        elif isinstance(U2_raw, list) and U2_raw and isinstance(U2_raw[0], list) and len(U2_raw) == 1:
            U2_vec = [int(v) & 1 for v in U2_raw[0]]
    except Exception:
        U2_vec = None

    # If U3 is an entry-mask, enforce diff support ⊆ mask.
    if U3_mask is not None and U3_mask and U3_mask[0]:
        if len(U3_mask) != n3 or any(len(r) != n3 for r in U3_mask):
            return {"ok": False, "reason": "U3_MASK_SHAPE_MISMATCH"}
        for i in range(n3):
            for j in range(n3):
                if (int(diff[i][j]) & 1) == 1 and (int(U3_mask[i][j]) & 1) == 0:
                    return {"ok": False, "reason": "C_DIFF_OUTSIDE_U3_MASK", "bad_cell": [int(i), int(j)]}
        # For H2 we can only enforce output rows if we have a basis-vector too.
        # Treat as pass if diff passes.
        return {"ok": True, "reason": None, "mode": "U3_mask"}

    # Basis-mask mode (vector). If missing or wrong length, fence is NA but not hard-fail.
    if U3_vec is None or len(U3_vec) != n3:
        return {"ok": True, "reason": "U3_NOT_VECTOR_OR_LEN_MISMATCH", "mode": "NA"}

    # Helper: for i outside U3, diff row/col must be zero; H2 output row must be zero.
    outside_rows = [i for i, v in enumerate(U3_vec) if (int(v) & 1) == 0]
    for i in outside_rows:
        if any((int(v) & 1) == 1 for v in diff[i]):
            return {"ok": False, "reason": "C_DIFF_ROW_OUTSIDE_U3", "bad_row": int(i)}
        for r in range(n3):
            if (int(diff[r][i]) & 1) == 1:
                return {"ok": False, "reason": "C_DIFF_COL_OUTSIDE_U3", "bad_col": int(i)}
        if i < len(H2) and any((int(v) & 1) == 1 for v in H2[i]):
            return {"ok": False, "reason": "H2_OUTPUT_OUTSIDE_U3", "bad_row": int(i)}

    # If U2 basis mask exists, enforce H2 has no input from outside U2.
    if U2_vec is not None and len(U2_vec) == n2:
        outside_cols = [j for j, v in enumerate(U2_vec) if (int(v) & 1) == 0]
        for j in outside_cols:
            for i in range(n3):
                if j < len(H2[i]) and (int(H2[i][j]) & 1) == 1:
                    return {"ok": False, "reason": "H2_INPUT_OUTSIDE_U2", "bad_col": int(j)}

    return {"ok": True, "reason": None, "mode": "U3_vec"}


def _phase5_compute_anchor_record(
    *,
    anchor_id: str,
    district_label: str,
    paths: dict,
    blocks: dict,
    expected_sha12: dict,
) -> dict:
    """Compute the strict-core-v1 anchor record (guards + hashes + support hashes)."""
    bB = blocks["B"]; bC = blocks["C"]; bH = blocks["H"]; bU = blocks["U"]

    # Core matrices
    d3 = _time_tau_t0_norm_bitmatrix((bB or {}).get("3") or [])
    C3 = _time_tau_t0_norm_bitmatrix((bC or {}).get("3") or [])
    H2 = _time_tau_t0_norm_bitmatrix((bH or {}).get("2") or [])
    n3 = len(C3) if (C3 and C3[0]) else (len(d3[0]) if (d3 and d3[0]) else 0)
    I3 = _svr_eye(n3) if n3 else []
    diffC = _svr_xor_gf2(C3, I3) if (C3 and I3) else []

    guards = {
        "grid":   _phase5_guard_grid(bB, bC),
        "ker":    _phase5_guard_ker(bB, bC),
        "fence":  _phase5_guard_fence(bB, bC, bH, bU),
        "cone":   _phase5_guard_cone(bC),
        "wiggle": _phase5_guard_wiggle(bB, bC, bH),
        # echo computed after sweeps
        "echo":   {"ok": None, "reason": "PENDING_SWEEPS"},
    }

    # Support hashes
    suppC_hash = _phase5_support_hash(diffC) if diffC else ""
    suppH_hash = _phase5_support_hash(H2) if H2 else ""

    # Projected mode: compute auto lanes projection, but do not change strict guard semantics.
    proj = None
    try:
        proj = _svr_projected_auto_from_blocks(bB, bC, bH, bU)
    except Exception as exc:
        proj = {"ok": False, "reason": f"PROJECTED_ERROR:{exc}"}

    return {
        "anchor_id": str(anchor_id),
        "district": str(district_label),
        "expected_sha12": dict(expected_sha12),
        "paths": {k: str(v) for k, v in (paths or {}).items()},
        "file_sha12": {
            "d": _phase5_sha25612_path(paths.get("B")) if paths.get("B") else "",
            "U": _phase5_sha25612_path(paths.get("U")) if paths.get("U") else "",
            "C": _phase5_sha25612_path(paths.get("C")) if paths.get("C") else "",
            "H": _phase5_sha25612_path(paths.get("H")) if paths.get("H") else "",
        },
        "hashes": {
            # Full (canonical-json) hashes for consumers that already bind via _svr_hash_json.
            "boundaries_hash": _svr_hash_json({"blocks": bB}),
            "U_hash":          _svr_hash_json({"blocks": bU}),
            "C_hash":          _svr_hash_json({"blocks": bC}),
            "H_hash":          _svr_hash_json({"blocks": bH}),
            "hash_suppC":      suppC_hash,
            "hash_suppH":      suppH_hash,
        },
        "guards": guards,
        "projected": proj,
        "promote_gate": {
            "rule": "[grid, wiggle, fence, echo] must be [1,1,1,1] (strict-core-v1).",
            "status": "PENDING_ECHO",
        },
    }


def _phase5_registry_rows_from_report(report: dict) -> list[dict]:
    """Flatten phase5 report into strict_core_v1 registry rows."""
    rows: list[dict] = []
    anchors = report.get("anchors") or []
    sweeps = report.get("echo_sweeps") or []
    for a in anchors:
        h = (a or {}).get("hashes") or {}
        g = (a or {}).get("guards") or {}
        rows.append(
            {
                "district": a.get("anchor_id"),
                "policy": "strict",
                "hash_d": h.get("boundaries_hash") or "",
                "hash_U": h.get("U_hash") or "",
                "hash_suppC": h.get("hash_suppC") or "",
                "hash_suppH": h.get("hash_suppH") or "",
                "kind": "anchor",
                "details": _json.dumps(
                    {
                        "guards": {
                            k: (v.get("ok") if isinstance(v, dict) else None)
                            for k, v in g.items()
                        },
                        "expected_sha12": (a or {}).get("expected_sha12"),
                        "file_sha12": (a or {}).get("file_sha12"),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            }
        )

    for s in sweeps:
        h = (s or {}).get("hashes") or {}
        rows.append(
            {
                "district": s.get("sweep_id"),
                "policy": "strict",
                "hash_d": h.get("hash_d") or "",
                "hash_U": h.get("hash_U") or "",
                "hash_suppC": h.get("hash_suppC") or "",
                "hash_suppH": h.get("hash_suppH") or "",
                "kind": "echo",
                "details": _json.dumps(
                    {
                        "pass_vec": (s or {}).get("pass_vec"),
                        "residual_tag": (s or {}).get("residual_tag"),
                        "notes": (s or {}).get("notes"),
                        "pair": (s or {}).get("pair"),
                        "row": (s or {}).get("row"),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                    ensure_ascii=True,
                ),
            }
        )
    return rows


def phase5_strict_core_v1_harness(*, write_files: bool = True) -> dict:
    """Run the strict-core-v1 anchor + same-H echo sweep harness.

    Returns a Phase5AcceptanceReport-like dict:
      {
        "schema": "phase5_acceptance.v0",
        "freeze_tag": "strict-core-v1",
        "anchors": [...],
        "echo_sweeps": [...],
        "ok": bool,
        "paths": {...},
        "notes": [...]
      }
    """
    # Resolve repo root + search dirs.
    try:
        root = _repo_root()
    except Exception:
        try:
            root = _REPO_DIR  # type: ignore[name-defined]
        except Exception:
            root = _Path(__file__).resolve().parents[1]
    root = _Path(root).resolve()

    search_dirs = [
        root / "app" / "inputs",
        root / "app" / "inputs" / "B",
        root / "app" / "inputs" / "C",
        root / "app" / "inputs" / "H",
        root / "app" / "inputs" / "U",
        root / "logs" / "_uploads",
    ]

    # strict_core_v1 appendix fixture hashes (SHA256/12 over file bytes). 
    FIXTURES = [
        {
            "anchor_id": "D2-101",
            "district": "D2",
            "sha12": {"B": "4356e6b60844", "U": "50a490cdd24d", "C": "7eb12a09fd0c", "H": "07281fc4127a"},
        },
        {
            "anchor_id": "D2-011",
            "district": "D2",
            "sha12": {"B": "4356e6b60844", "U": "50a490cdd24d", "C": "b5957219e7c5", "H": "7ed045d924f3"},
        },
        {
            "anchor_id": "D3-110",
            "district": "D3",
            "sha12": {"B": "28f8db2a822c", "U": "50a490cdd24d", "C": "8aaa11bd6daf", "H": "86fc8542a3ee"},
        },
        {
            "anchor_id": "D4-101",
            "district": "D4",
            "sha12": {"B": "aea6404ae680", "U": "50a490cdd24d", "C": "b04475f5273a", "H": "52bb3c369dd3"},
        },
        {
            "anchor_id": "D4-011",
            "district": "D4",
            "sha12": {"B": "aea6404ae680", "U": "50a490cdd24d", "C": "bf68c3f42abc", "H": "33352e067572"},
        },
    ]

    notes: list[str] = []
    anchors_out: list[dict] = []

    for fx in FIXTURES:
        sha = fx["sha12"]
        paths: dict[str, _Path] = {}
        blocks: dict[str, dict] = {}
        missing: list[str] = []
        lookup_notes: list[str] = []

        for role in ("B", "U", "C", "H"):
            p, n2 = _phase5_find_json_by_sha12(search_dirs, sha.get(role, ""))
            lookup_notes.extend(n2)
            if p is None:
                missing.append(role)
            else:
                paths[role] = p
                try:
                    j, _, _ = abx_read_json_any(str(p), kind={"B": "boundaries", "U": "shapes", "C": "cmap", "H": "H"}[role])
                    blocks[role] = _svr_as_blocks_v2(j, role)
                except Exception as exc:
                    missing.append(role)
                    lookup_notes.append(f"LOAD_FAIL:{role}:{exc}")

        if missing:
            anchors_out.append(
                {
                    "anchor_id": fx["anchor_id"],
                    "district": fx["district"],
                    "expected_sha12": dict(sha),
                    "status": "MISSING_FIXTURE_FILES",
                    "missing_roles": sorted(set(missing)),
                    "lookup_notes": lookup_notes[:50],
                }
            )
            continue

        rec = _phase5_compute_anchor_record(
            anchor_id=fx["anchor_id"],
            district_label=fx["district"],
            paths={"B": paths["B"], "U": paths["U"], "C": paths["C"], "H": paths["H"]},
            blocks={"B": blocks["B"], "U": blocks["U"], "C": blocks["C"], "H": blocks["H"]},
            expected_sha12=dict(sha),
        )
        if lookup_notes:
            rec["lookup_notes"] = lookup_notes[:50]
        anchors_out.append(rec)

    # Build lookup by anchor_id
    by_id = {a.get("anchor_id"): a for a in anchors_out if isinstance(a, dict) and a.get("anchor_id")}

    # Same-H echo sweeps (strict_core_v1): row1 pairs and row2 pair. 
    SWEEPS = [
        {"sweep_id": "echo[D2<->D3]", "pair": ["D2-101", "D3-110"], "row": "row1", "notes": "same-H row1 echo"},
        {"sweep_id": "echo[D3<->D4]", "pair": ["D3-110", "D4-101"], "row": "row1", "notes": "same-H row1 echo"},
        {"sweep_id": "echo[D2<->D4]", "pair": ["D2-101", "D4-101"], "row": "row1", "notes": "same-H row1 echo"},
        {"sweep_id": "echo[D2<->D4]_row2", "pair": ["D2-011", "D4-011"], "row": "row2", "notes": "same-H row2 echo"},
    ]
    sweeps_out: list[dict] = []

    def _gok(a: dict, k: str) -> bool:
        g = (a or {}).get("guards") or {}
        v = g.get(k) if isinstance(g, dict) else None
        if isinstance(v, dict):
            return v.get("ok") is True
        return False

    for sw in SWEEPS:
        a_id, b_id = sw["pair"]
        a = by_id.get(a_id)
        b = by_id.get(b_id)
        if not isinstance(a, dict) or not isinstance(b, dict) or a.get("status") == "MISSING_FIXTURE_FILES" or b.get("status") == "MISSING_FIXTURE_FILES":
            sweeps_out.append(
                {
                    "sweep_id": sw["sweep_id"],
                    "pair": sw["pair"],
                    "row": sw["row"],
                    "policy": "strict",
                    "pass_vec": [0, 0, 0, 0],
                    "residual_tag": "missing_fixture",
                    "notes": sw["notes"],
                    "hashes": {},
                }
            )
            continue

        grid_ok = _gok(a, "grid") and _gok(b, "grid")
        wiggle_ok = _gok(a, "wiggle") and _gok(b, "wiggle")
        fence_ok = _gok(a, "fence") and _gok(b, "fence")
        # Echo is the sweep's own boolean: in v1 it's "pass vec [grid,wiggle,fence,echo]".
        echo_ok = bool(grid_ok and wiggle_ok and fence_ok)
        pass_vec = [1 if grid_ok else 0, 1 if wiggle_ok else 0, 1 if fence_ok else 0, 1 if echo_ok else 0]
        sweeps_out.append(
            {
                "sweep_id": sw["sweep_id"],
                "pair": sw["pair"],
                "row": sw["row"],
                "policy": "strict",
                "pass_vec": pass_vec,
                "residual_tag": "none" if echo_ok else "needs_witness",
                "notes": sw["notes"],
                # For registry placeholders, use the left anchor's hashes as the representative tuple. 
                "hashes": {
                    "hash_d": ((a.get("hashes") or {}).get("boundaries_hash") or ""),
                    "hash_U": ((a.get("hashes") or {}).get("U_hash") or ""),
                    "hash_suppC": ((a.get("hashes") or {}).get("hash_suppC") or ""),
                    "hash_suppH": ((a.get("hashes") or {}).get("hash_suppH") or ""),
                },
            }
        )

    # Propagate echo-ok back into anchors as their Echo guard (derived from sweeps).
    echo_by_anchor: dict[str, bool] = {}
    for sw in sweeps_out:
        echo_ok = bool((sw.get("pass_vec") or [0, 0, 0, 0])[3] == 1)
        for aid in sw.get("pair") or []:
            if aid not in echo_by_anchor:
                echo_by_anchor[aid] = echo_ok
            else:
                echo_by_anchor[aid] = bool(echo_by_anchor[aid] and echo_ok)

    for a in anchors_out:
        aid = a.get("anchor_id") if isinstance(a, dict) else None
        if not aid or not isinstance(a, dict):
            continue
        if a.get("status") == "MISSING_FIXTURE_FILES":
            continue
        if "guards" in a and isinstance(a["guards"], dict):
            ok = echo_by_anchor.get(aid)
            a["guards"]["echo"] = {"ok": bool(ok) if ok is not None else False, "reason": None if ok else "ECHO_SWEEP_FAIL_OR_MISSING"}
            # Promote gate status (anchor rule) now resolvable.
            grid_ok = _gok(a, "grid")
            wiggle_ok = _gok(a, "wiggle")
            fence_ok = _gok(a, "fence")
            echo_ok2 = _gok(a, "echo")
            a["promote_gate"]["status"] = "OK" if (grid_ok and wiggle_ok and fence_ok and echo_ok2) else "FAIL"
            a["promote_gate"]["vec"] = [1 if grid_ok else 0, 1 if wiggle_ok else 0, 1 if fence_ok else 0, 1 if echo_ok2 else 0]

    # Overall acceptance: all anchors either computed+promote OK, and all sweeps echo_ok.
    ok_anchors = True
    for a in anchors_out:
        if not isinstance(a, dict):
            ok_anchors = False
            break
        if a.get("status") == "MISSING_FIXTURE_FILES":
            ok_anchors = False
            continue
        pg = (a.get("promote_gate") or {}).get("status")
        if pg != "OK":
            ok_anchors = False

    ok_sweeps = all(bool((sw.get("pass_vec") or [0, 0, 0, 0])[3] == 1) for sw in sweeps_out)
    ok_all = bool(ok_anchors and ok_sweeps)

    # Write report + registry
    rep_dir = root / "logs" / "reports" / "strict_core_v1"
    rep_dir.mkdir(parents=True, exist_ok=True)
    registry_path = rep_dir / "registry.csv"
    report_path = rep_dir / "phase5_acceptance.json"

    report = {
        "schema": "phase5_acceptance.v0",
        "schema_version": "phase5_acceptance.v0",
        "freeze_tag": "strict-core-v1",
        "ok": ok_all,
        "anchors": anchors_out,
        "echo_sweeps": sweeps_out,
        "paths": {
            "registry_csv": str(_bundle_repo_relative_path(registry_path)) if callable(globals().get("_bundle_repo_relative_path")) else str(registry_path),
            "acceptance_json": str(_bundle_repo_relative_path(report_path)) if callable(globals().get("_bundle_repo_relative_path")) else str(report_path),
        },
        "notes": notes,
    }

    if write_files:
        try:
            rows = _phase5_registry_rows_from_report(report)
            # deterministic CSV: fixed header order from v1 notes. 
            fieldnames = ["district", "policy", "hash_d", "hash_U", "hash_suppC", "hash_suppH", "kind", "details"]
            with registry_path.open("w", encoding="utf-8", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fieldnames})
        except Exception as exc:
            report["ok"] = False
            report.setdefault("notes", []).append(f"REGISTRY_WRITE_FAIL:{exc}")

        try:
            report_path.write_text(_json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            report["ok"] = False
            report.setdefault("notes", []).append(f"REPORT_WRITE_FAIL:{exc}")

    return report


# --- UI hook (best-effort): show in Export tab if defined, else at top-level. ---
try:
    _phase5_parent = tab5  # type: ignore[name-defined]
except Exception:
    try:
        _phase5_parent = tab2  # type: ignore[name-defined]
    except Exception:
        _phase5_parent = None

try:
    _phase5_container = _phase5_parent if _phase5_parent is not None else st.container()
    with _phase5_container:
        with st.expander("Phase 5 — strict-core-v1 harness", expanded=False):
            st.caption(
                "Runs the frozen strict-core-v1 anchors + same-H echo sweeps and writes "
                "`logs/reports/strict_core_v1/registry.csv` and `phase5_acceptance.json`."
            )
            if st.button("Run Phase 5 harness (strict-core-v1)", key="phase5_strict_core_v1_run"):
                rep = phase5_strict_core_v1_harness(write_files=True)
                st.session_state["phase5_strict_core_v1_last"] = rep
                st.success("Phase 5 harness completed." if rep.get("ok") else "Phase 5 harness completed (NOT OK).")

            rep = st.session_state.get("phase5_strict_core_v1_last")
            if isinstance(rep, dict):
                st.markdown("### Result")
                st.json({"ok": rep.get("ok"), "freeze_tag": rep.get("freeze_tag"), "paths": rep.get("paths"), "notes": rep.get("notes")})

                # Download helpers
                try:
                    root = _repo_root()
                except Exception:
                    try:
                        root = _REPO_DIR  # type: ignore[name-defined]
                    except Exception:
                        root = _Path(__file__).resolve().parents[1]
                root = _Path(root).resolve()
                rep_dir = root / "logs" / "reports" / "strict_core_v1"
                reg_p = rep_dir / "registry.csv"
                acc_p = rep_dir / "phase5_acceptance.json"

                col1, col2 = st.columns(2)
                with col1:
                    if reg_p.exists():
                        st.download_button("Download registry.csv", data=reg_p.read_bytes(), file_name="registry.csv", mime="text/csv")
                with col2:
                    if acc_p.exists():
                        st.download_button("Download phase5_acceptance.json", data=acc_p.read_bytes(), file_name="phase5_acceptance.json", mime="application/json")
except Exception:
    # If Streamlit isn't available (e.g., running unit tests), silently ignore UI wiring.
    pass


# ───────────────────────────── Phase 7 — Step‑A island hardening harness ─────────────────────────────
# Mode B wiring:
#   • Regression fixtures for Step‑A boundary (no-new-type) and Gate‑B expected-fail labeling.
#   • Tower-link checks (adjacent TypeGate PASS + component constancy across levels).
#   • Phase‑3 detector regression: out-of-scope Step‑A invocation must trip STEP_A_INVOKED_OUT_OF_SCOPE.
#
# IMPORTANT: This harness is observational-only; it does not mutate any frozen artifacts.

if "_PHASE7_STEP_A_ISLAND_FREEZE_TAG" not in globals():
    _PHASE7_STEP_A_ISLAND_FREEZE_TAG = "phase7-stepA-island-v1"


if "_phase7_stepA_island_fixtures_v1" not in globals():
    def _phase7_stepA_island_fixtures_v1() -> dict:
        """Return the frozen Phase‑7 Step‑A island fixture set (v1).

        Fixtures are intentionally tiny and self-contained (no repo inputs).
        """

        # Base matrix A0: 4×2 with columns e0, e1.
        A0 = [
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0],
        ]

        # Step‑A admissible maps (no new type): permutation, duplication, append all‑zero.
        A_perm = [
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0],
        ]
        A_dup = [
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        A_zero = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        # Tower-friendly monotone extension: (duplicate) then (append all-zero).
        # This avoids any multiplicity decreases along adjacent links.
        A_dup_zero = [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        # Novelty injection: append a genuinely new column type e3 (introduces a new nonzero type).
        A_newtype = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]

        cases = [
            {
                "case_id": "permute_cols_no_new_type",
                "A0": A0,
                "A1": A_perm,
                "probe_id": None,
                "expect": {
                    "type_gate_status": "PASS",
                    "stepB_regime": "STEP_A_ADMISSIBLE",
                    "stepB_new_type_delta_nonempty": False,
                    "stepC_new_type": False,
                    "stepC_stepA_applies": True,
                    "cert_gate_status": "PASS",
                },
            },
            {
                "case_id": "duplicate_col_no_new_type",
                "A0": A0,
                "A1": A_dup,
                "probe_id": None,
                "expect": {
                    "type_gate_status": "PASS",
                    "stepB_regime": "STEP_A_ADMISSIBLE",
                    "stepB_new_type_delta_nonempty": False,
                    "stepC_new_type": False,
                    "stepC_stepA_applies": True,
                    "cert_gate_status": "PASS",
                },
            },
            {
                "case_id": "append_zero_no_new_type",
                "A0": A0,
                "A1": A_zero,
                "probe_id": None,
                "expect": {
                    "type_gate_status": "PASS",
                    "stepB_regime": "STEP_A_ADMISSIBLE",
                    "stepB_new_type_delta_nonempty": False,
                    "stepC_new_type": False,
                    "stepC_stepA_applies": True,
                    "cert_gate_status": "PASS",
                },
            },
            {
                "case_id": "novelty_injected_new_type",
                "A0": A0,
                "A1": A_newtype,
                "probe_id": "novelty_injected",
                "probe_params": {"fixture": "phase7_stepA_island_v1"},
                "expect": {
                    "type_gate_status": "FAIL",
                    "stepB_regime": "NOT_ADMISSIBLE_FOR_STEP_A",
                    "stepB_new_type_delta_nonempty": True,
                    "stepB_probe_expected_outside_stepA": True,
                    "stepC_new_type": True,
                    "stepC_stepA_applies": False,
                    "cert_gate_status": "FAIL",
                },
            },
        ]

        towers = [
            {
                "tower_id": "toy_stepA_tower_v1",
                "levels": [
                    ("L0", A0),
                    ("L1", A_perm),
                    ("L2", A_dup),
                    ("L3", A_dup_zero),
                ],
                "expect": {
                    "is_stepA_tower": True,
                    "constant_across_levels_when_stepA": True,
                },
            }
        ]

        return {"freeze_tag": _PHASE7_STEP_A_ISLAND_FREEZE_TAG, "cases": cases, "towers": towers}


if "phase7_stepA_island_harness" not in globals():
    def phase7_stepA_island_harness(*, write_files: bool = True) -> dict:
        """Run the Phase‑7 Step‑A island regression harness.

        Writes a content-addressed report under logs/reports/phase7_stepA_island/ when write_files=True.
        """
        fixtures = _phase7_stepA_island_fixtures_v1()
        cases = list(fixtures.get("cases") or [])
        towers = list(fixtures.get("towers") or [])

        out_cases: list[dict] = []
        out_towers: list[dict] = []
        ok = True

        # --- Case checks (TypeGate/CertGate + Step‑B record + Step‑C new-type detector) ---
        for fx in cases:
            fx0 = fx if isinstance(fx, dict) else {}
            case_id = str(fx0.get("case_id") or "")
            A0 = fx0.get("A0")
            A1 = fx0.get("A1")

            exp = dict((fx0.get("expect") or {}))

            # Observations
            link = _inv_stepA_link_gate(A0, A1)
            tg = (link or {}).get("type_gate") if isinstance(link, dict) else None
            cg = (link or {}).get("cert_gate") if isinstance(link, dict) else None
            stepC = _stepC_new_type_detection(A0, A1)

            probe_id = fx0.get("probe_id")
            probe_params = fx0.get("probe_params") if isinstance(fx0.get("probe_params"), dict) else None
            stepB = _inv_stepB_out_of_regime_record(A0, A1, probe_id=(str(probe_id) if probe_id else None), probe_params=probe_params)

            obs = {
                "type_gate_status": (tg or {}).get("status") if isinstance(tg, dict) else None,
                "cert_gate_status": (cg or {}).get("status") if isinstance(cg, dict) else None,
                "stepB_regime": (stepB or {}).get("regime") if isinstance(stepB, dict) else None,
                "stepB_new_type_delta_nonempty": (stepB or {}).get("new_type_delta_nonempty") if isinstance(stepB, dict) else None,
                "stepB_probe_expected_outside_stepA": (stepB or {}).get("probe_expected_outside_stepA") if isinstance(stepB, dict) else None,
                "stepC_new_type": (stepC or {}).get("new_type") if isinstance(stepC, dict) else None,
                "stepC_stepA_applies": (stepC or {}).get("stepA_applies") if isinstance(stepC, dict) else None,
            }

            mismatches: list[dict] = []
            for k, v in exp.items():
                if obs.get(k) != v:
                    mismatches.append({"field": k, "expected": v, "got": obs.get(k)})

            if mismatches:
                ok = False
            out_cases.append(
                {
                    "case_id": case_id,
                    "expected": exp,
                    "observed": obs,
                    "mismatches": mismatches,
                }
            )

        # --- Tower-link checks ---
        for tw in towers:
            tw0 = tw if isinstance(tw, dict) else {}
            tower_id = str(tw0.get("tower_id") or "")
            levels = list(tw0.get("levels") or [])
            exp = dict((tw0.get("expect") or {}))

            obs_bar = _stepC_tower_barcode(levels)
            obs = {
                "is_stepA_tower": obs_bar.get("is_stepA_tower") if isinstance(obs_bar, dict) else None,
                "constant_across_levels_when_stepA": obs_bar.get("constant_across_levels_when_stepA") if isinstance(obs_bar, dict) else None,
            }

            mismatches: list[dict] = []
            for k, v in exp.items():
                if obs.get(k) != v:
                    mismatches.append({"field": k, "expected": v, "got": obs.get(k)})

            if mismatches:
                ok = False
            out_towers.append(
                {
                    "tower_id": tower_id,
                    "expected": exp,
                    "observed": obs,
                    "mismatches": mismatches,
                    "payload": {"tower_barcode": obs_bar},
                }
            )

        # --- Phase‑3 detector regression (synthetic out-of-scope Step‑A invocation) ---
        synthetic_attempts = [
            {
                "glue_id": "synthetic_out_of_scope",
                "glue_record": {
                    "phi_exists": False,
                    "new_type": True,
                    "decision": "Persist",
                    "stepA_applies": False,
                },
                "stepA_check": {"invoked": True, "payload": {"note": "synthetic"}},
            }
        ]
        synthetic_inv = [
            {
                "event": "stepA_parity_check_invoked",
                "glue_id": "synthetic_out_of_scope",
                "stepA_applies": True,
            }
        ]
        phase3 = _stepC_phase3_report(glue_attempts=synthetic_attempts, invocation_log=synthetic_inv)
        phase3_expected_fail = bool(phase3.get("phase3_pass") is False)
        phase3_has_signal = False
        try:
            for f in list(phase3.get("failures") or []):
                if isinstance(f, dict) and f.get("code") == "STEP_A_INVOKED_OUT_OF_SCOPE":
                    phase3_has_signal = True
                    break
        except Exception:
            phase3_has_signal = False

        if not (phase3_expected_fail and phase3_has_signal):
            ok = False

        report = {
            "schema": "phase7_stepA_island_harness",
            "schema_version": "phase7.stepA_island_harness.v1",
            "freeze_tag": fixtures.get("freeze_tag"),
            "ok": bool(ok),
            "cases": out_cases,
            "towers": out_towers,
            "phase3_detector_regression": {
                "expected_phase3_pass": False,
                "observed_phase3_pass": phase3.get("phase3_pass"),
                "has_STEP_A_INVOKED_OUT_OF_SCOPE": bool(phase3_has_signal),
                "phase3_report": phase3,
            },
            # Excluded from canonical hash surface (ephemeral key).
            "created_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
        }

        # --- Write content-addressed report (never overwrite). ---
        paths = {}
        if write_files:
            try:
                try:
                    root = _repo_root()
                except Exception:
                    try:
                        root = _REPO_DIR  # type: ignore[name-defined]
                    except Exception:
                        root = _Path(__file__).resolve().parents[1]
                root = _Path(root).resolve()

                rep_dir = root / "logs" / "reports" / "phase7_stepA_island"
                rep_dir.mkdir(parents=True, exist_ok=True)

                hb = dict(report)
                hb.pop("sig8", None)
                sig8 = hash_json_sig8(hb)
                report["sig8"] = sig8

                p = rep_dir / f"{_PHASE7_STEP_A_ISLAND_FREEZE_TAG}__{sig8}.json"
                if not p.exists():
                    p.write_text(canonical_json(report), encoding="utf-8")
                paths["report"] = p.as_posix()
            except Exception:
                paths = {}

        if paths:
            report["paths"] = paths

        return report


# --- UI hook (best-effort): expose the harness near the Export tab. ---
try:
    _phase7_parent = tab5  # type: ignore[name-defined]
except Exception:
    try:
        _phase7_parent = tab2  # type: ignore[name-defined]
    except Exception:
        _phase7_parent = None

try:
    _phase7_container = _phase7_parent if _phase7_parent is not None else st.container()
    with _phase7_container:
        with st.expander("Phase 7 — Step‑A island hardening harness", expanded=False):
            st.caption(
                "Runs tiny projection-first regressions for Step‑A boundary (no new type), "
                "Gate‑B probe expected-fail labeling, and tower-link constancy. "
                "Writes a content-addressed JSON report under logs/reports/phase7_stepA_island/."
            )

            if st.button("Run Phase 7 harness (Step‑A island)", key="phase7_stepA_island_run"):
                rep = phase7_stepA_island_harness(write_files=True)
                st.session_state["phase7_stepA_island_last"] = rep
                st.success("Phase 7 harness completed." if rep.get("ok") else "Phase 7 harness completed (NOT OK).")

            rep = st.session_state.get("phase7_stepA_island_last")
            if isinstance(rep, dict):
                st.markdown("### Result")
                st.json({"ok": rep.get("ok"), "freeze_tag": rep.get("freeze_tag"), "sig8": rep.get("sig8"), "paths": rep.get("paths")})

                # Download helper (best-effort).
                try:
                    p = (rep.get("paths") or {}).get("report")
                    if p and _Path(str(p)).exists():
                        rp = _Path(str(p))
                        st.download_button("Download phase7 report", data=rp.read_bytes(), file_name=rp.name, mime="application/json")
                except Exception:
                    pass
except Exception:
    # If Streamlit isn't available (e.g., running unit tests), silently ignore UI wiring.
    pass


# ───────────────────────────── Phase 6 — Odd‑Tetra move program harness ─────────────────────────────
# Mode B wiring:
#   • Minimal disk schema for per-move dossiers (projection-first; tolerant field aliases).
#   • Checklist evaluation (structural + basic equalities; no bespoke composite arguments).
#   • Required overlap-suite coverage check (R6).
#   • Closure gate: (all move checklists PASS) ∧ (R6 coverage PASS) ∧ (colimit spec present).
#   • Writes logs/reports/phase6/{registry.csv,phase6_acceptance.json}.

if "_phase6_m0" not in globals():
    def _phase6_m0() -> list[dict]:
        """Return the frozen Phase-6 generator set M0 with kinds.

        This is an implementation of the Mode-A frozen list:
          AZ, DU, SD_bary, SD_stell, SC, EX_fus, EX_spl, EX_hin, MN_del, MN_ctr
        """
        return [
            {"move_id": "AZ", "move_kind": "direct"},
            {"move_id": "DU", "move_kind": "zigzag"},
            {"move_id": "SD_bary", "move_kind": "refproj"},
            {"move_id": "SD_stell", "move_kind": "refproj"},
            {"move_id": "SC", "move_kind": "zigzag"},
            {"move_id": "EX_fus", "move_kind": "span"},
            {"move_id": "EX_spl", "move_kind": "span"},
            {"move_id": "EX_hin", "move_kind": "span"},
            {"move_id": "MN_del", "move_kind": "direct"},
            {"move_id": "MN_ctr", "move_kind": "direct"},
        ]


if "_phase6_required_overlap_suite" not in globals():
    def _phase6_required_overlap_suite() -> list[dict]:
        """Return the frozen required overlap obligations R6.

        Each obligation is a dict:
          {"overlap_class": "...", "a": "<move_id>", "b": "<move_id>"}.
        """
        req: list[dict] = []

        # R6.1: DU ↔ SD naturality (both schemes)
        req.append({"overlap_class": "REFINEMENT_NATURALITY", "a": "DU", "b": "SD_bary"})
        req.append({"overlap_class": "REFINEMENT_NATURALITY", "a": "DU", "b": "SD_stell"})

        # R6.2: append-zero outside carrier commutes strictly with everything else in M0
        for m in _phase6_m0():
            mid = m["move_id"]
            if mid == "AZ":
                continue
            req.append({"overlap_class": "APPENDZERO_OUTSIDE_CARRIER", "a": "AZ", "b": mid})

        # R6.3: exotics ↔ subdivision compatibility (both schemes)
        for ex in ("EX_fus", "EX_spl", "EX_hin"):
            for sd in ("SD_bary", "SD_stell"):
                req.append({"overlap_class": "SPAN_REFINEMENT_COMPAT", "a": ex, "b": sd})

        return req


if "_p6_get" not in globals():
    def _p6_get(d: dict | None, *keys, default=None):
        """Return first present key from keys (tolerates Header/header style)."""
        if not isinstance(d, dict):
            return default
        for k in keys:
            if k in d:
                return d.get(k)
        return default


if "_p6_is_nonempty_dict" not in globals():
    def _p6_is_nonempty_dict(x: object) -> bool:
        return isinstance(x, dict) and bool(x)


if "_p6_is_nonempty_list" not in globals():
    def _p6_is_nonempty_list(x: object) -> bool:
        return isinstance(x, list) and len(x) > 0


if "_p6_has_blocks" not in globals():
    def _p6_has_blocks(x: object) -> bool:
        """Best-effort: does x look like a degreewise blocks container?"""
        if x is None:
            return False
        # Accept dict with "blocks"
        if isinstance(x, dict):
            b = x.get("blocks")
            if isinstance(b, dict) and b:
                return True
            # Accept top-level degree keys
            for k in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
                if k in x and isinstance(x.get(k), list):
                    return True
            # Accept "1","2","3" legacy degrees
            for k in ("1", "2", "3"):
                if k in x and isinstance(x.get(k), list):
                    return True
        # Accept raw list (matrix)
        if isinstance(x, list):
            return True
        return False


if "_p6_boolish_ok" not in globals():
    def _p6_boolish_ok(x: object) -> bool:
        """Return True for ok-ish atoms: True, 'PASS', 1."""
        if x is True:
            return True
        if x == 1:
            return True
        if isinstance(x, str) and x.strip().upper() in ("PASS", "OK", "TRUE", "YES"):
            return True
        return False


if "_p6_eval_boundary_audit" not in globals():
    def _p6_eval_boundary_audit(boundary_audit: object) -> tuple[bool, list[str]]:
        """Best-effort evaluation of a boundary audit structure.

        This intentionally does NOT invent semantics. It only recognizes:
          - boundary_audit.ok == True
          - boundary_audit == "PASS"
          - per-role/per-degree ok booleans/strings
        """
        notes: list[str] = []
        if boundary_audit is None:
            return False, ["missing"]
        if _p6_boolish_ok(boundary_audit):
            return True, []
        if isinstance(boundary_audit, dict):
            if _p6_boolish_ok(boundary_audit.get("ok")):
                return True, []
            # tolerate {"roles": {...}}
            roles = boundary_audit.get("roles") if isinstance(boundary_audit.get("roles"), dict) else boundary_audit
            if isinstance(roles, dict):
                bad: list[str] = []
                for role, per in roles.items():
                    if _p6_boolish_ok(per):
                        continue
                    if isinstance(per, dict):
                        for deg, v in per.items():
                            if _p6_boolish_ok(v):
                                continue
                            bad.append(f"{role}:{deg}")
                    else:
                        bad.append(str(role))
                if not bad:
                    return True, []
                return False, bad
        if isinstance(boundary_audit, list):
            # list of {"role":..., "deg":..., "ok":...}
            bad: list[str] = []
            for row in boundary_audit:
                if not isinstance(row, dict):
                    continue
                if _p6_boolish_ok(row.get("ok")):
                    continue
                role = row.get("role", "?")
                deg = row.get("deg", "?")
                bad.append(f"{role}:{deg}")
            if not bad:
                return True, []
            return False, bad
        return False, ["unrecognized_shape"]


if "_p6_eval_transport" not in globals():
    def _p6_eval_transport(transport: object, *, which: str) -> tuple[bool, list[str]]:
        """Evaluate transport[c2]/transport[c1] structurally."""
        if not isinstance(transport, dict):
            return False, ["missing_transport_dict"]
        ent = transport.get(which) or transport.get(which.lower()) or transport.get(which.upper())
        if not isinstance(ent, dict):
            return False, [f"missing_{which}"]
        form = ent.get("transport_form") or ent.get("form") or ent.get("kind") or ent.get("mode")
        if isinstance(form, str):
            f = form.strip().lower()
            if f in ("strict_equal", "strict", "equal"):
                # Strict transport should carry either witness=0 or omit witness.
                return True, []
            if f in ("boundary_witness", "witnessed", "up_to_boundary", "homology"):
                # Witnessed transport requires some witness payload, but we don't constrain its shape.
                w = ent.get("witness") or ent.get("b") or ent.get("boundary_witness")
                if w is None:
                    return False, [f"{which}:missing_witness"]
                return True, []
        # Allow explicit ok atom
        if _p6_boolish_ok(ent.get("ok")):
            return True, []
        return False, [f"{which}:missing_transport_form"]


if "_p6_eval_pairing" not in globals():
    def _p6_eval_pairing(pairing: object) -> tuple[bool, list[str]]:
        """Pairing is load-bearing; require equality when both sides are present."""
        if pairing is None:
            return False, ["missing"]
        if isinstance(pairing, dict):
            if _p6_boolish_ok(pairing.get("ok")):
                return True, []
            before = pairing.get("before") if "before" in pairing else pairing.get("pairing_before")
            after = pairing.get("after") if "after" in pairing else pairing.get("pairing_after")
            if before is None or after is None:
                return False, ["missing_before_or_after"]
            return (before == after), ([] if before == after else ["before!=after"])
        # Allow direct equality-coded atom
        if _p6_boolish_ok(pairing):
            return True, []
        return False, ["unrecognized_shape"]


if "_p6_overlap_entry_key" not in globals():
    def _p6_overlap_entry_key(ent: dict) -> tuple[str, str]:
        """Return (with_move_id, overlap_class) in normalized form."""
        with_id = str(ent.get("with_move_id") or ent.get("with") or ent.get("other_move_id") or "").strip()
        cls = str(ent.get("overlap_class") or ent.get("class") or ent.get("kind") or "").strip()
        return with_id, cls


if "_p6_extract_overlap_entries" not in globals():
    def _p6_extract_overlap_entries(dossier: dict) -> list[dict]:
        od = _p6_get(dossier, "overlap_data", "OverlapData", default=[])
        if isinstance(od, dict):
            # tolerate {"entries":[...]}
            entries = od.get("entries")
            if isinstance(entries, list):
                return [e for e in entries if isinstance(e, dict)]
            # tolerate dict keyed by with_move_id
            out: list[dict] = []
            for _, v in od.items():
                if isinstance(v, dict):
                    out.append(v)
                elif isinstance(v, list):
                    out.extend([e for e in v if isinstance(e, dict)])
            return out
        if isinstance(od, list):
            return [e for e in od if isinstance(e, dict)]
        return []


if "_p6_dossier_kind" not in globals():
    def _p6_dossier_kind(dossier: dict) -> str:
        mk = _p6_get(dossier, "move_kind", "moveKind", "MoveKind", default=None)
        if isinstance(mk, str) and mk.strip():
            return mk.strip()
        # fall back to header.move_kind
        hdr = _p6_get(dossier, "header", "Header", default={})
        if isinstance(hdr, dict):
            mk2 = _p6_get(hdr, "move_kind", "moveKind", "MoveKind", default=None)
            if isinstance(mk2, str) and mk2.strip():
                return mk2.strip()
        return ""


if "_p6_dossier_id" not in globals():
    def _p6_dossier_id(dossier: dict) -> str:
        mid = _p6_get(dossier, "move_id", "moveId", "MoveID", default=None)
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
        hdr = _p6_get(dossier, "header", "Header", default={})
        if isinstance(hdr, dict):
            mid2 = _p6_get(hdr, "move_id", "moveId", "MoveID", default=None)
            if isinstance(mid2, str) and mid2.strip():
                return mid2.strip()
        return ""


if "_phase6_required_overlaps_for_move" not in globals():
    def _phase6_required_overlaps_for_move(move_id: str) -> list[dict]:
        """Subset of R6 that is *owned* by this move in our Phase-6 move sheets."""
        move_id = str(move_id or "")
        out: list[dict] = []
        if move_id == "AZ":
            for m in _phase6_m0():
                mid = m["move_id"]
                if mid == "AZ":
                    continue
                out.append({"overlap_class": "APPENDZERO_OUTSIDE_CARRIER", "a": "AZ", "b": mid})
        if move_id == "DU":
            out.append({"overlap_class": "REFINEMENT_NATURALITY", "a": "DU", "b": "SD_bary"})
            out.append({"overlap_class": "REFINEMENT_NATURALITY", "a": "DU", "b": "SD_stell"})
        if move_id in ("EX_fus", "EX_spl", "EX_hin"):
            out.append({"overlap_class": "SPAN_REFINEMENT_COMPAT", "a": move_id, "b": "SD_bary"})
            out.append({"overlap_class": "SPAN_REFINEMENT_COMPAT", "a": move_id, "b": "SD_stell"})
        return out


if "_phase6_eval_dossier_checklist" not in globals():
    def _phase6_eval_dossier_checklist(dossier: dict) -> dict:
        """Return checklist dict with PASS/FAIL/N_A and small notes arrays.

        This is intentionally enforcement-light: it checks presence and
        basic equalities that are already projected (e.g., pairing equality).
        Deep matrix equalities remain in the dedicated move-check engines.
        """
        mid = _p6_dossier_id(dossier)
        mk = _p6_dossier_kind(dossier)
        checklist: dict = {"move_id": mid, "move_kind": mk, "items": {}, "ok": False, "notes": []}

        # --- Mat / maps presence ---
        maps = _p6_get(dossier, "maps", "Maps", default=None)
        mat_ok = False
        mat_notes: list[str] = []
        if mk == "direct":
            # expect role "C"
            C = None
            if isinstance(maps, dict):
                C = maps.get("C") or maps.get("c") or maps.get("chain_map")
            if _p6_has_blocks(C):
                mat_ok = True
            else:
                # allow maps dict itself to be the block container
                if _p6_has_blocks(maps):
                    mat_ok = True
                else:
                    mat_notes.append("missing_C_blocks")
        elif mk == "zigzag":
            if isinstance(maps, dict):
                e = maps.get("e") or maps.get("E") or maps.get("expansion")
                c = maps.get("c") or maps.get("C") or maps.get("collapse")
                if _p6_has_blocks(e) and _p6_has_blocks(c):
                    mat_ok = True
                else:
                    if not _p6_has_blocks(e):
                        mat_notes.append("missing_e")
                    if not _p6_has_blocks(c):
                        mat_notes.append("missing_c")
            else:
                mat_notes.append("missing_maps_dict")
        elif mk == "refproj":
            if isinstance(maps, dict):
                r = maps.get("r") or maps.get("R") or maps.get("refinement")
                p = maps.get("p") or maps.get("P") or maps.get("projection")
                H = maps.get("H") or maps.get("h") or maps.get("prism") or maps.get("homotopy")
                if _p6_has_blocks(r) and _p6_has_blocks(p) and (H is None or _p6_has_blocks(H) or isinstance(H, dict)):
                    mat_ok = True
                else:
                    if not _p6_has_blocks(r):
                        mat_notes.append("missing_r")
                    if not _p6_has_blocks(p):
                        mat_notes.append("missing_p")
                    # H is required by spec, but accept dict placeholder
                    if H is None:
                        mat_notes.append("missing_H")
            else:
                mat_notes.append("missing_maps_dict")
        elif mk == "span":
            if isinstance(maps, dict):
                i_star = maps.get("i_star") or maps.get("i") or maps.get("i_*") or maps.get("inclusion")
                o_star = maps.get("o_star") or maps.get("o") or maps.get("o_*") or maps.get("output")
                if _p6_has_blocks(i_star) and _p6_has_blocks(o_star):
                    mat_ok = True
                else:
                    if not _p6_has_blocks(i_star):
                        mat_notes.append("missing_i_*")
                    if not _p6_has_blocks(o_star):
                        mat_notes.append("missing_o_*")
            else:
                mat_notes.append("missing_maps_dict")
        else:
            mat_notes.append("missing_move_kind")

        checklist["items"]["Mat"] = {"ok": bool(mat_ok), "notes": mat_notes}

        # --- Boundary audit ---
        bda = _p6_get(dossier, "boundary_audit", "BoundaryAudit", default=None)
        b_ok, b_bad = _p6_eval_boundary_audit(bda)
        checklist["items"]["Boundary"] = {"ok": bool(b_ok), "notes": b_bad}

        # --- Transport ---
        tr = _p6_get(dossier, "transport", "Transport", default=None)
        c2_ok, c2_bad = _p6_eval_transport(tr, which="c2")
        c1_ok, c1_bad = _p6_eval_transport(tr, which="c1")
        checklist["items"]["Transport[c2]"] = {"ok": bool(c2_ok), "notes": c2_bad}
        checklist["items"]["Transport[c1]"] = {"ok": bool(c1_ok), "notes": c1_bad}

        # --- Pairing ---
        pr = _p6_get(dossier, "pairing", "Pairing", default=None)
        p_ok, p_bad = _p6_eval_pairing(pr)
        checklist["items"]["Pairing"] = {"ok": bool(p_ok), "notes": p_bad}

        # --- Support ---
        sup = _p6_get(dossier, "support", "Support", default=None)
        sup_ok = _p6_boolish_ok(sup) or _p6_is_nonempty_dict(sup) or _p6_is_nonempty_list(sup)
        checklist["items"]["Support"] = {"ok": bool(sup_ok), "notes": [] if sup_ok else ["missing"]}

        # --- Composition stub ---
        cs = _p6_get(dossier, "composition_stub", "CompositionStub", "composition", default=None)
        cs_ok = _p6_is_nonempty_dict(cs)
        checklist["items"]["CompStub"] = {"ok": bool(cs_ok), "notes": [] if cs_ok else ["missing"]}

        # --- Naturality/overlap items (owned subset; global R6 checked separately) ---
        owned = _phase6_required_overlaps_for_move(mid)
        entries = _p6_extract_overlap_entries(dossier)
        have: set[tuple[str, str]] = set()
        for e in entries:
            wid, cls = _p6_overlap_entry_key(e)
            have.add((wid, cls))
        # Evaluate owned obligations locally.
        missing_owned: list[str] = []
        for ob in owned:
            cls = ob["overlap_class"]
            other = ob["b"] if ob["a"] == mid else ob["a"]
            if (other, cls) not in have:
                missing_owned.append(f"{cls}:{other}")
        if not owned:
            checklist["items"]["NaturalityDisjoint"] = {"ok": True, "notes": ["N/A"]}
            checklist["items"]["NaturalityOverlap"] = {"ok": True, "notes": ["N/A"]}
        else:
            # classify by overlap class family
            disjoint_missing = [x for x in missing_owned if x.startswith("APPENDZERO_OUTSIDE_CARRIER")]
            overlap_missing = [x for x in missing_owned if not x.startswith("APPENDZERO_OUTSIDE_CARRIER")]
            nd_ok = (len(disjoint_missing) == 0)
            no_ok = (len(overlap_missing) == 0)
            checklist["items"]["NaturalityDisjoint"] = {"ok": bool(nd_ok), "notes": disjoint_missing}
            checklist["items"]["NaturalityOverlap"] = {"ok": bool(no_ok), "notes": overlap_missing}

        # Overall ok: all boolean ok items (excluding those explicitly N/A in notes)
        all_ok = True
        for _, ent in checklist["items"].items():
            if not isinstance(ent, dict):
                continue
            if ent.get("notes") == ["N/A"]:
                continue
            if ent.get("ok") is not True:
                all_ok = False
        checklist["ok"] = bool(all_ok)
        return checklist


if "_phase6_find_dossier_path" not in globals():
    def _phase6_find_dossier_path(search_dirs: list[_Path], move_id: str) -> _Path | None:
        """Find a dossier file by deterministic name candidates (best-effort)."""
        move_id = str(move_id or "").strip()
        if not move_id:
            return None
        # deterministic candidate names
        names = [
            f"{move_id}.json",
            f"dossier__{move_id}.json",
            f"phase6_dossier__{move_id}.json",
        ]
        for d in (search_dirs or []):
            try:
                dd = _Path(d)
            except Exception:
                continue
            # try direct
            for nm in names:
                p = dd / nm
                if p.exists() and p.is_file():
                    return p
            # try dossiers/ subdir
            for nm in names:
                p = dd / "dossiers" / nm
                if p.exists() and p.is_file():
                    return p
        return None


if "_phase6_read_json_file" not in globals():
    def _phase6_read_json_file(p: _Path) -> dict:
        try:
            return _json.loads(_Path(p).read_text(encoding="utf-8"))
        except Exception:
            return {}


if "_phase6_eval_required_overlaps" not in globals():
    def _phase6_eval_required_overlaps(dossiers_by_id: dict[str, dict]) -> dict:
        """Check R6 coverage by searching overlap_data in either endpoint dossier."""
        required = _phase6_required_overlap_suite()
        missing: list[dict] = []

        # Build index: (a, b, class) satisfied?
        def _has_overlap(d: dict, other: str, cls: str) -> bool:
            ents = _p6_extract_overlap_entries(d)
            for e in ents:
                wid, c = _p6_overlap_entry_key(e)
                if wid == other and c == cls:
                    return True
            return False

        for ob in required:
            a = ob["a"]
            b = ob["b"]
            cls = ob["overlap_class"]
            da = dossiers_by_id.get(a) or {}
            db = dossiers_by_id.get(b) or {}
            ok = False
            if isinstance(da, dict) and da:
                ok = ok or _has_overlap(da, b, cls)
            if isinstance(db, dict) and db:
                ok = ok or _has_overlap(db, a, cls)
            if not ok:
                missing.append({"overlap_class": cls, "a": a, "b": b})
        return {"ok": len(missing) == 0, "missing": missing, "required_n": len(required)}


if "_phase6_eval_colimit_spec" not in globals():
    def _phase6_eval_colimit_spec(search_dirs: list[_Path]) -> dict:
        """Best-effort: locate and structurally validate ColimitStabilitySpec₆."""
        # deterministic filenames
        names = [
            "colimit_stability.json",
            "phase6_colimit_stability.json",
            "colimits.json",
        ]
        p_found: _Path | None = None
        for d in (search_dirs or []):
            for nm in names:
                p = _Path(d) / nm
                if p.exists() and p.is_file():
                    p_found = p
                    break
            if p_found:
                break
            # also allow in phase6/ subdir
            for nm in names:
                p = _Path(d) / "phase6" / nm
                if p.exists() and p.is_file():
                    p_found = p
                    break
            if p_found:
                break

        if p_found is None:
            return {"ok": False, "path": None, "notes": ["missing_colimit_spec_file"]}

        obj = _phase6_read_json_file(p_found)
        if not isinstance(obj, dict) or not obj:
            return {"ok": False, "path": str(p_found), "notes": ["colimit_spec_parse_failed_or_empty"]}

        # Accept either direct H1..H4 or hypotheses dict with those keys.
        hyp = obj.get("hypotheses") if isinstance(obj.get("hypotheses"), dict) else obj
        have_h = all(k in hyp for k in ("H1", "H2", "H3", "H4"))
        have_lambda = ("lambda" in obj) or ("λ" in obj) or ("lambda_map" in obj)
        # Toy is optional but projected in our spec; require presence of either "toy" or "appendzero_outside_carrier".
        have_toy = ("toy" in obj) or ("appendzero_outside_carrier" in obj) or ("append_zero_outside_carrier" in obj)

        ok = bool(have_h and have_lambda and have_toy)
        notes: list[str] = []
        if not have_h:
            notes.append("missing_H1H4")
        if not have_lambda:
            notes.append("missing_lambda")
        if not have_toy:
            notes.append("missing_appendzero_toy")
        return {"ok": ok, "path": str(p_found), "notes": notes}


if "phase6_write_templates" not in globals():
    def phase6_write_templates(*, overwrite: bool = False) -> dict:
        """Write skeleton Phase-6 dossier templates + colimit spec skeleton to logs/phase6/ (best-effort)."""
        try:
            root = _repo_root()
        except Exception:
            try:
                root = _REPO_DIR  # type: ignore[name-defined]
            except Exception:
                root = _Path(__file__).resolve().parents[1]
        root = _Path(root).resolve()

        base = root / "logs" / "phase6"
        dd = base / "dossiers"
        dd.mkdir(parents=True, exist_ok=True)

        written: dict[str, str] = {}
        for m in _phase6_m0():
            mid = m["move_id"]
            mk = m["move_kind"]
            p = dd / f"{mid}.json"
            if p.exists() and not overwrite:
                continue
            template = {
                "schema": "phase6_dossier.v0",
                "move_id": mid,
                "move_kind": mk,
                "header": {
                    "move_id": mid,
                    "move_kind": mk,
                    "domain_object": "",
                    "codomain_object": "",
                    "parameters": {},
                },
                "carrier": {
                    "carrier_U_domain": [],
                    "carrier_U_codomain": [],
                    "notes": [],
                },
                # "maps" is intentionally just a placeholder; per-move data fills blocks.
                "maps": {},
                "boundary_audit": {"ok": False, "roles": {}},
                "transport": {
                    "c2": {"transport_form": "", "witness": None},
                    "c1": {"transport_form": "", "witness": None},
                },
                "pairing": {"before": None, "after": None},
                "support": {},
                "overlap_data": [],
                "composition_stub": {},
                "notes": [],
            }
            try:
                p.write_text(_json.dumps(template, indent=2, sort_keys=False), encoding="utf-8")
                written[mid] = str(p)
            except Exception:
                pass

        # Colimit stability skeleton
        col_p = base / "colimit_stability.json"
        if (not col_p.exists()) or overwrite:
            col_template = {
                "schema": "phase6_colimit_stability.v0",
                "hypotheses": {"H1": "", "H2": "", "H3": "", "H4": ""},
                "lambda": {"defined": False, "notes": []},
                "appendzero_outside_carrier": {"toy_present": False, "notes": []},
                "notes": [],
            }
            try:
                col_p.write_text(_json.dumps(col_template, indent=2, sort_keys=False), encoding="utf-8")
                written["colimit_stability"] = str(col_p)
            except Exception:
                pass

        return {"base_dir": str(base), "written": written}


if "phase6_move_program_harness" not in globals():
    def phase6_move_program_harness(*, write_files: bool = True) -> dict:
        """Run Phase-6 harness (best-effort) and write logs/reports/phase6 outputs."""
        try:
            root = _repo_root()
        except Exception:
            try:
                root = _REPO_DIR  # type: ignore[name-defined]
            except Exception:
                root = _Path(__file__).resolve().parents[1]
        root = _Path(root).resolve()

        search_dirs = [
            root / "logs" / "phase6",
            root / "app" / "inputs" / "phase6",
            root / "app" / "inputs",
            root / "logs" / "_uploads",
        ]

        moves_out: list[dict] = []
        dossiers_by_id: dict[str, dict] = {}

        for m in _phase6_m0():
            mid = m["move_id"]
            mk_expected = m["move_kind"]
            p = _phase6_find_dossier_path(search_dirs, mid)
            if p is None:
                moves_out.append(
                    {
                        "move_id": mid,
                        "move_kind": mk_expected,
                        "status": "MISSING_DOSSIER",
                        "path": None,
                        "checklist": None,
                    }
                )
                continue

            obj = _phase6_read_json_file(p)
            if not isinstance(obj, dict) or not obj:
                moves_out.append(
                    {
                        "move_id": mid,
                        "move_kind": mk_expected,
                        "status": "BAD_JSON",
                        "path": str(p),
                        "checklist": None,
                    }
                )
                continue

            # Evaluate checklist.
            chk = _phase6_eval_dossier_checklist(obj)
            # Keep the canonical move_id/move_kind from M0 as the report identity,
            # but surface mismatches as notes (payload; does not change meaning).
            notes: list[str] = []
            if chk.get("move_id") and chk.get("move_id") != mid:
                notes.append(f"move_id_mismatch:dossier={chk.get('move_id')}")
            if chk.get("move_kind") and chk.get("move_kind") != mk_expected:
                notes.append(f"move_kind_mismatch:dossier={chk.get('move_kind')}")
            if notes:
                chk.setdefault("notes", []).extend(notes)

            dossiers_by_id[mid] = obj
            moves_out.append(
                {
                    "move_id": mid,
                    "move_kind": mk_expected,
                    "status": "OK" if chk.get("ok") else "FAIL",
                    "path": str(p),
                    "checklist": chk,
                }
            )

        # Global overlap suite check (R6)
        ov = _phase6_eval_required_overlaps(dossiers_by_id)

        # Colimit stability spec (Brick 12)
        col = _phase6_eval_colimit_spec(search_dirs)

        # Closure gate: per-move ok + overlaps ok + colimit ok
        all_moves_ok = all((m.get("status") == "OK") for m in moves_out if m.get("status") != "MISSING_DOSSIER")
        any_missing = any((m.get("status") == "MISSING_DOSSIER") for m in moves_out)
        closure_ok = bool((not any_missing) and all_moves_ok and ov.get("ok") and col.get("ok"))

        rep_dir = root / "logs" / "reports" / "phase6"
        rep_dir.mkdir(parents=True, exist_ok=True)
        registry_path = rep_dir / "registry.csv"
        report_path = rep_dir / "phase6_acceptance.json"

        # Deterministic registry: one row per move in M0.
        def _vec(chk: dict | None) -> str:
            if not isinstance(chk, dict):
                return ""
            items = chk.get("items")
            if not isinstance(items, dict):
                return ""
            order = ["Mat", "Boundary", "Transport[c2]", "Transport[c1]", "Pairing", "NaturalityDisjoint", "NaturalityOverlap", "Support", "CompStub"]
            bits: list[str] = []
            for k in order:
                ent = items.get(k)
                if isinstance(ent, dict) and ent.get("notes") == ["N/A"]:
                    bits.append("N")
                else:
                    bits.append("1" if (isinstance(ent, dict) and ent.get("ok") is True) else "0")
            return "".join(bits)

        report = {
            "schema": "phase6_acceptance.v0",
            "schema_version": "phase6_acceptance.v0",
            "ok": bool(closure_ok),
            "m0": [m["move_id"] for m in _phase6_m0()],
            "moves": moves_out,
            "overlap_suite": ov,
            "colimit_stability": col,
            "closure": {
                "all_moves_ok": bool((not any_missing) and all_moves_ok),
                "overlaps_ok": bool(ov.get("ok")),
                "colimit_ok": bool(col.get("ok")),
                "ok": bool(closure_ok),
            },
            "paths": {
                "registry_csv": str(registry_path),
                "acceptance_json": str(report_path),
            },
            "notes": [],
        }

        if write_files:
            try:
                # Write registry CSV
                fieldnames = ["move_id", "move_kind", "status", "checklist_vec", "path"]
                with registry_path.open("w", encoding="utf-8", newline="") as f:
                    w = _csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for row in moves_out:
                        chk = row.get("checklist")
                        w.writerow(
                            {
                                "move_id": row.get("move_id", ""),
                                "move_kind": row.get("move_kind", ""),
                                "status": row.get("status", ""),
                                "checklist_vec": _vec(chk if isinstance(chk, dict) else None),
                                "path": row.get("path", "") or "",
                            }
                        )
            except Exception as exc:
                report["ok"] = False
                report.setdefault("notes", []).append(f"REGISTRY_WRITE_FAIL:{exc}")

            try:
                report_path.write_text(_json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
            except Exception as exc:
                report["ok"] = False
                report.setdefault("notes", []).append(f"REPORT_WRITE_FAIL:{exc}")

        return report


# --- UI hook (best-effort): show in Export tab if defined, else at top-level. ---
try:
    _phase6_parent = tab5  # type: ignore[name-defined]
except Exception:
    try:
        _phase6_parent = tab2  # type: ignore[name-defined]
    except Exception:
        _phase6_parent = None

try:
    _phase6_container = _phase6_parent if _phase6_parent is not None else st.container()
    with _phase6_container:
        with st.expander("Phase 6 — Odd‑Tetra move program (M0 dossiers + overlap closure)", expanded=False):
            st.caption(
                "Mode B wiring: loads per-move dossiers for AZ/DU/SD/SC/EX/MN, checks checklist surfaces, "
                "checks required overlap suite R6 coverage, and checks for a colimit-stability spec file. "
                "Writes `logs/reports/phase6/{registry.csv,phase6_acceptance.json}`."
            )

            colA, colB = st.columns(2)
            with colA:
                if st.button("Write Phase 6 templates (logs/phase6/…)", key="phase6_write_templates"):
                    out = phase6_write_templates(overwrite=False)
                    st.session_state["phase6_templates_last"] = out
                    st.success("Wrote Phase 6 templates (non-overwriting).")
            with colB:
                if st.button("Run Phase 6 harness", key="phase6_run_harness"):
                    rep = phase6_move_program_harness(write_files=True)
                    st.session_state["phase6_last"] = rep
                    st.success("Phase 6 harness completed." if rep.get("ok") else "Phase 6 harness completed (NOT OK).")

            tmpl = st.session_state.get("phase6_templates_last")
            if isinstance(tmpl, dict):
                with st.expander("Template write output", expanded=False):
                    st.json(tmpl)

            rep = st.session_state.get("phase6_last")
            if isinstance(rep, dict):
                st.markdown("### Result")
                st.json(
                    {
                        "ok": rep.get("ok"),
                        "closure": rep.get("closure"),
                        "overlap_suite": {
                            "ok": (rep.get("overlap_suite") or {}).get("ok"),
                            "missing_n": len((rep.get("overlap_suite") or {}).get("missing") or []),
                            "required_n": (rep.get("overlap_suite") or {}).get("required_n"),
                        },
                        "colimit_ok": (rep.get("colimit_stability") or {}).get("ok"),
                        "paths": rep.get("paths"),
                        "notes": rep.get("notes"),
                    }
                )

                # Download helpers
                try:
                    root = _repo_root()
                except Exception:
                    try:
                        root = _REPO_DIR  # type: ignore[name-defined]
                    except Exception:
                        root = _Path(__file__).resolve().parents[1]
                root = _Path(root).resolve()
                rep_dir = root / "logs" / "reports" / "phase6"
                reg_p = rep_dir / "registry.csv"
                acc_p = rep_dir / "phase6_acceptance.json"

                col1, col2 = st.columns(2)
                with col1:
                    if reg_p.exists():
                        st.download_button("Download registry.csv", data=reg_p.read_bytes(), file_name="registry.csv", mime="text/csv")
                with col2:
                    if acc_p.exists():
                        st.download_button("Download phase6_acceptance.json", data=acc_p.read_bytes(), file_name="phase6_acceptance.json", mime="application/json")
except Exception:
    # If Streamlit isn't available (e.g., running unit tests), silently ignore UI wiring.
    pass

