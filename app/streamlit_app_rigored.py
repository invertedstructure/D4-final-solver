

# =============================== SSOT & Sigma Helpers (final pass) ===============================
def _ensure_ssot_published():
    """
    Best-effort: publish SSOT hashes/dims/run_id into session before panels read them.
    Does not write to disk; uses existing resolver/freezer if available.
    """
    try:
        import streamlit as _st
        ss = _st.session_state
        rc = dict(ss.get("run_ctx") or {})
        ib = dict(ss.get("_inputs_block") or {})
        # If hashes are missing, try to freeze now
        hashes = (ib.get("hashes") or {})
        have_hashes = bool(hashes.get("boundaries_hash") and hashes.get("U_hash"))
        if (not have_hashes) and ("_svr_resolve_all_to_paths" in globals()) and ("_svr_freeze_ssot" in globals()):
            try:
                pb = _svr_resolve_all_to_paths()
                ib_new, rc_new = _svr_freeze_ssot(pb)
                if isinstance(ib_new, dict): ib = ib_new
                if isinstance(rc_new, dict): rc = rc_new
                ss["_inputs_block"] = ib
                ss["run_ctx"] = rc
            except Exception:
                pass
        # Ensure run_id exists
        if "run_id" not in rc or not rc.get("run_id"):
            import uuid as _uuid
            rc["run_id"] = str(_uuid.uuid4())
            ss["run_ctx"] = rc
        return ib, rc
    except Exception:
        return {}, {}

def _svr_publish_sigma_to_session(sigma_bits=None, sigma_compact=None):
    """
    Publish sigma (pattern-only signature) into session & run_ctx if available.
    No attempt to recompute; only normalizes/copies into consistent keys.
    """
    try:
        import streamlit as _st
        ss = _st.session_state
        rc = dict(ss.get("run_ctx") or {})
        # prefer explicit args; else pick up from session/rc
        if sigma_bits is None:
            sigma_bits = ss.get("sigma_bits_k3") or rc.get("sigma_bits_k3")
        if sigma_compact is None:
            sigma_compact = ss.get("sigma_compact") or rc.get("sigma_compact")
        # normalize & publish
        if sigma_bits and isinstance(sigma_bits, (list, tuple)):
            sb = [1 if int(x) else 0 for x in sigma_bits]
            ss["sigma_bits_k3"] = sb
            rc["sigma_bits_k3"] = sb
        if sigma_compact and isinstance(sigma_compact, str):
            ss["sigma_compact"] = sigma_compact
            rc["sigma_compact"] = sigma_compact
        ss["run_ctx"] = rc
    except Exception:
        pass
# =============================== /SSOT & Sigma Helpers ===============================

# =============================== After-run Telemetry Hook (pre-C1) ===============================
try:
    import json as _json
    with st.expander("Telemetry hook (auto)", expanded=False):
        ss = st.session_state
        _ensure_ssot_published()
        rc = dict(ss.get("run_ctx") or {})
        last_dir = ss.get("last_bundle_dir") or ""
        appended = set(ss.get("_telemetry_appended_keys") or [])
        key_now = None

        strict_eq = None
        proj_auto_eq = None
        embed_sig_auto = ""

        try:
            from pathlib import Path as _P
            bdir = _P(last_dir)
            if bdir and bdir.exists():
                # Parse strict eq from overlap strict cert
                for f in bdir.iterdir():
                    name = f.name
                    if name.startswith("overlap__") and "__strict__" in name and name.endswith(".json"):
                        try:
                            obj = _json.loads(f.read_text("utf-8", errors="ignore"))
                            # common shapes: obj["k"]["3"]["eq"] or obj["3"]["eq"]
                            if isinstance(obj, dict):
                                v = (obj.get("k") or {}).get("3", {}) if "k" in obj else obj.get("3", {})
                                strict_eq = bool(v.get("eq")) if isinstance(v, dict) else strict_eq
                        except Exception:
                            pass
                    if name.startswith("overlap__") and "__projected_columns_k_3_auto__" in name and name.endswith(".json"):
                        try:
                            obj = _json.loads(f.read_text("utf-8", errors="ignore"))
                            v = (obj.get("k") or {}).get("3", {}) if "k" in obj else obj.get("3", {})
                            proj_auto_eq = bool(v.get("eq")) if isinstance(v, dict) else proj_auto_eq
                        except Exception:
                            pass
                    if name.startswith("ab_compare__strict_vs_projected_auto__") and name.endswith(".json"):
                        try:
                            obj = _json.loads(f.read_text("utf-8", errors="ignore"))
                            # try typical embed structure
                            emb = (obj.get("payload") or {}).get("embed") or obj.get("embed") or {}
                            embed_sig_auto = str(emb.get("embed_sig") or obj.get("embed_sig") or "")
                        except Exception:
                            pass
        except Exception:
            pass

        # Publish sigma if already present in rc/ss
        _svr_publish_sigma_to_session(
            sigma_bits=(rc.get("sigma_bits_k3") or ss.get("sigma_bits_k3")),
            sigma_compact=(rc.get("sigma_compact") or ss.get("sigma_compact")),
        )

        # Append coverage if we can form a key and haven't already appended this run
        run_id = str(rc.get("run_id") or "")
        key_now = f"{run_id}::{embed_sig_auto}" if embed_sig_auto else (run_id or None)
        can_append = bool(run_id and (key_now not in appended) and (strict_eq is not None or proj_auto_eq is not None))

        if can_append:
            row = {
                "prox_label": str(rc.get("prox_label") or ss.get("prox_label") or "unknown"),
                "covered": bool(proj_auto_eq is True),
                "in_baseline": bool(strict_eq is True),
                "distance": float(0.0),
                "embed_sig": embed_sig_auto,
                "policy": "strict__VS__projected(columns@k=3,auto)",
                "n2": int(rc.get("n2") or (rc.get("dims") or {}).get("n2") or 0),
                "n3": int(rc.get("n3") or (rc.get("dims") or {}).get("n3") or 0),
                "district_id": str(rc.get("district_id") or "UNKNOWN"),
                "fixture_label": str(rc.get("fixture_label") or ""),
            }
            _atomic_append_jsonl(COVERAGE_JSONL, row)
            appended.add(key_now)
            ss["_telemetry_appended_keys"] = sorted(appended)
            _rollup_coverage_jsonl_to_csv(COVERAGE_JSONL, REPORTS_DIR / "coverage_rollup.csv")
        # keep the hook silent in UI
except Exception:
    pass
# =============================== /After-run Telemetry Hook ===============================
# =============================== C1 Hygiene & Telemetry Helpers (v2) ===============================
try:
    import streamlit as _st
except Exception:
    class _stub: 
        session_state = {}
        def markdown(*a, **k): pass
        def caption(*a, **k): pass
    _st = _stub()

from pathlib import Path as _Path
import json as _json
import csv as _csv
import math as _math
import hashlib as _hashlib
import datetime as _c1dt_mod  # avoid name collisions: use module alias

# Ensure REPORTS_DIR exists (fallback if not defined elsewhere)
try:
    REPORTS_DIR  # type: ignore
except NameError:
    REPORTS_DIR = _Path("logs") / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _short_hex(h: str, n: int = 8) -> str:
    try:
        return str(h or "")[:n]
    except Exception:
        return ""

def _sha256_hex_bytes(b: bytes) -> str:
    return _hashlib.sha256(b).hexdigest()

def _json_dumps_canonical(obj) -> str:
    return _json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)

def _lane_mask_sig8(mask):
    try:
        b = _json_dumps_canonical(list(mask or [])).encode("utf-8")
        return _short_hex(_sha256_hex_bytes(b), 8)
    except Exception:
        return ""

def _c1_enumerate_flips(sigma_bits, lane_mask_k3):
    """
    Yield flip rows restricted to lane columns. sigma_bits is a list[int] of length n3 (0/1).
    lane_mask_k3 is a list[int] (0/1). If mask is empty/None, treat all as allowed (enumerate),
    and health will later mark off_row_rate as 'NA(no_lane_mask)'.
    Returns: list of dict rows for CSV.
    """
    ss = _st.session_state if hasattr(_st, "session_state") else {}
    rc = dict(ss.get("run_ctx") or {})
    n2 = int(rc.get("n2") or 0)
    n3 = int(rc.get("n3") or (len(sigma_bits) if sigma_bits is not None else 0))
    district_id = str((ss.get("_district_info") or {}).get("district_id") or rc.get("district_id") or "UNKNOWN")
    fixture_label = str(rc.get("fixture_label") or "UNKNOWN")
    sigma_str = ss.get("sigma_compact") or "|".join(map(str, getattr(ss, "sigma_compact_vec", []) or [])) or \
                "|".join("1" if b else "0" for b in (sigma_bits or []))
    prox_label = str(rc.get("prox_label") or ss.get("prox_label") or "unknown")
    run_id = str(rc.get("run_id") or rc.get("run_id") or "")
    schema_version = globals().get("SCHEMA_VERSION", "2.0.0")
    engine_rev = globals().get("ENGINE_REV", "rev-unknown")
    saved_at_utc = _c1dt_mod.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    mask = list(lane_mask_k3 or [])
    if not mask:
        # treat all as allowed for enumeration; health ping will record NA+reason
        mask = [1] * (len(sigma_bits) if sigma_bits is not None else n3)

    rows = []
    lane_sig8 = _lane_mask_sig8(mask)
    for j, bit in enumerate(list(sigma_bits or [])):
        allowed = int(mask[j] == 1) if j < len(mask) else 0
        in_lane = 1 if allowed else 0
        off_row = 0 if in_lane == 1 else 1
        if not allowed:
            # Hygiene: skip emitting rows that are off-lane
            continue
        bit_from = int(bit or 0)
        bit_to = 1 - bit_from
        rows.append({
            "schema_version": schema_version,
            "saved_at_utc": saved_at_utc,
            "run_id": run_id,
            "engine_rev": engine_rev,
            "district_id": district_id,
            "fixture_label": fixture_label,
            "n2": n2,
            "n3": n3,
            "sigma": sigma_str,
            "lane_mask_sig8": lane_sig8,
            "prox_label": prox_label,
            "bit_index": j,
            "bit_from": bit_from,
            "bit_to": bit_to,
            "in_lane": in_lane,
            "off_row": off_row,
        })
    return rows

def _write_c1_flips_csv(rows):
    """
    Write to reports/c1_flips.csv with exact header, no comments.
    """
    csv_path = REPORTS_DIR / "c1_flips.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "schema_version","saved_at_utc","run_id","engine_rev","district_id","fixture_label",
        "n2","n3","sigma","lane_mask_sig8","prox_label","bit_index","bit_from","bit_to","in_lane","off_row",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows or []:
            w.writerow({k: r.get(k) for k in header})
    return str(csv_path)

def _detect_bool_or_numeric(sample_value):
    # returns "bool", "num", or None
    if isinstance(sample_value, bool):
        return "bool"
    if isinstance(sample_value, (int, float)) and not isinstance(sample_value, bool):
        return "num"
    return None

def _rollup_coverage_jsonl_to_csv(jsonl_path, csv_path):
    """
    Group by prox_label. For boolean fields => rate_<field>; numeric => mean_<field>.
    Output columns: group, rows, then rate_*, then mean_* (alphabetical by field).
    Returns (csv_path, num_groups, total_rows).
    """
    jsonl_path = _Path(jsonl_path)
    csv_path = _Path(csv_path)
    groups = {}
    total = 0

    if not jsonl_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["group","rows"])
        return str(csv_path), 0, 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception:
                continue
            g = str(obj.get("prox_label", "unknown"))
            bucket = groups.setdefault(g, {"rows": [], "fields": set()})
            bucket["rows"].append(obj)
            bucket["fields"].update(obj.keys())
            total += 1

    bool_fields = set()
    num_fields = set()
    for g, bucket in groups.items():
        for obj in bucket["rows"]:
            for k, v in obj.items():
                if k == "prox_label":
                    continue
                t = _detect_bool_or_numeric(v)
                if t == "bool":
                    bool_fields.add(k)
                elif t == "num":
                    num_fields.add(k)

    rate_cols = [f"rate_{k}" for k in sorted(bool_fields)]
    mean_cols = [f"mean_{k}" for k in sorted(num_fields)]
    header = ["group", "rows"] + rate_cols + mean_cols

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for g in sorted(groups.keys()):
            bucket = groups[g]
            rows = bucket["rows"]
            n = len(rows) or 1
            out = [g, len(rows)]
            for k in sorted(bool_fields):
                s = sum(1 if bool(r.get(k)) else 0 for r in rows)
                out.append(s / n if n else 0.0)
            for k in sorted(num_fields):
                vals = [r.get(k) for r in rows]
                vals = [float(v) for v in vals if isinstance(v, (int,float))]
                m = (sum(vals) / len(vals)) if vals else 0.0
                out.append(m)
            w.writerow(out)

    return str(csv_path), len(groups), total

def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0

def _svr_mismatch_cols(residual_bottom_row, selected_mask):
    """
    Return indices j where selected lane and residual has 1 at bottom row.
    """
    res = list(residual_bottom_row or [])
    mask = list(selected_mask or [])
    n = min(len(res), len(mask))
    out = []
    for j in range(n):
        if int(mask[j] == 1) and int(res[j] == 1):
            out.append(j)
    return out

def _compute_c1_health_ping(flips_csv_path, coverage_jsonl_path, residual_bottom_row, lane_mask_k3):
    # flips_rows
    flips_rows = 0
    try:
        with open(flips_csv_path, "r", encoding="utf-8") as f:
            flips_rows = sum(1 for _ in f) - 1  # minus header
            if flips_rows < 0: flips_rows = 0
    except Exception:
        flips_rows = 0

    # coverage_rows
    coverage_rows = 0
    if coverage_jsonl_path and _Path(coverage_jsonl_path).exists():
        with open(coverage_jsonl_path, "r", encoding="utf-8") as f:
            coverage_rows = sum(1 for _ in f if _.strip())

    # off_row_rate
    mask = list(lane_mask_k3 or [])
    if not mask:
        off_row_rate = "NA(no_lane_mask)"
    else:
        try:
            off = 0
            tot = 0
            with open(flips_csv_path, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    tot += 1
                    try:
                        off += int(r.get("off_row") or 0)
                    except Exception:
                        pass
            off_row_rate = (off / tot) if tot else 0.0
        except Exception:
            off_row_rate = 0.0

    # mismatch_rate
    sel = list(lane_mask_k3 or [])
    mm = _svr_mismatch_cols(residual_bottom_row or [], sel)
    mden = sum(1 for v in sel if int(v)==1) or 1
    mismatch_rate = (len(mm) / mden) if mden else 0.0

    # Rollup
    rollup_csv = str(REPORTS_DIR / "coverage_rollup.csv")
    rcsv, gcount, _total = _rollup_coverage_jsonl_to_csv(coverage_jsonl_path, rollup_csv)

    out = {
        "off_row_rate": off_row_rate,
        "mismatch_rate": mismatch_rate,
        "flips_rows": flips_rows,
        "coverage_rows": coverage_rows,
        "rollup_groups": gcount,
        "last_rollup_csv": rcsv,
    }
    return out

def _render_c1_health_chip(health: dict, wait: bool = False):
    try:
        if wait:
            import streamlit as _st
            if hasattr(_st, 'markdown'):
                _st.markdown("<span style='padding:4px 8px;border-radius:999px;background:#666;color:white;font-weight:600;'>C1: WAIT</span>", unsafe_allow_html=True)
                _st.caption(str(health))
            return 'WAIT'
    except Exception:
        pass

    try:
        off_row_rate = health.get("off_row_rate")
        mismatch_rate = float(health.get("mismatch_rate") or 0.0)
        coverage_rows = int(health.get("coverage_rows") or 0)
        status = "FAIL"
        # Treat any string starting with "NA" as NA
        if isinstance(off_row_rate, str) and off_row_rate.upper().startswith("NA"):
            status = "FAIL"
        else:
            orr = float(off_row_rate or 0.0)
            if orr == 0 and coverage_rows > 0:
                if mismatch_rate == 0:
                    status = "OK"
                elif mismatch_rate <= 0.05:
                    status = "WARN"
                else:
                    status = "FAIL"
            else:
                status = "FAIL"
        if hasattr(_st, "markdown"):
            color = {"OK":"green","WARN":"orange","FAIL":"red"}[status]
            _st.markdown(f"<span style='padding:4px 8px;border-radius:999px;background:{color};color:white;font-weight:600;'>C1: {status}</span>", unsafe_allow_html=True)
            _st.caption(str(health))
        return status
    except Exception:
        return "FAIL"

def _infer_lane_mask_from_session(ss, rc):
    """
    Try to infer lane_mask_k3 from session if missing:
    Prefer bottom row of C3 (lanes = bottom(C3)).
    Searches common session keys: "overlap_C", "run_ctx.C3", "C3".
    Returns list[int] or [] if not found.
    """
    # 1) from run_ctx directly
    try:
        C3 = rc.get("C3")
        if isinstance(C3, list) and C3 and isinstance(C3[-1], list):
            return [int(x) & 1 for x in C3[-1]]
    except Exception:
        pass
    # 2) from session keys
    for key in ("overlap_C","C_obj","C3","overlap_out"):
        try:
            val = ss.get(key)
            # possible shapes:
            # - {"blocks": {"3": [[...],[...],...]}} or similar parse_cmap object
            if isinstance(val, dict):
                # nested dict path
                for k in ("blocks","block","B","C"):
                    if k in val:
                        sub = val[k]
                        # try common level "3"
                        if isinstance(sub, dict):
                            m = sub.get("3")
                            if isinstance(m, list) and m and isinstance(m[-1], list):
                                return [int(x) & 1 for x in m[-1]]
                        # or if it's already a matrix
                        if isinstance(sub, list) and sub and isinstance(sub[-1], list):
                            return [int(x) & 1 for x in sub[-1]]
            # maybe it is already a matrix
            if isinstance(val, list) and val and isinstance(val[-1], list):
                return [int(x) & 1 for x in val[-1]]
        except Exception:
            continue
    return []

# =============================== /C1 Hygiene & Telemetry Helpers (v2) ===============================
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
        _atomic_append_jsonl(REPORTS_DIR / "freezer_mismatch_log.jsonl", row)
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
    REPORTS_DIR = REPORTS_DIR
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


def _svr_can_write_now() -> bool:
    """
    Only allow disk writes when invoked by the single solver button.
    Accept if any of these session flags are set:
      - _solver_busy (debounce flag during run)
      - _solver_one_button_active (legacy guard we set inside the try-block)
      - _cert_write_allowed (escape hatch for offline tests)
    Fallback: if Streamlit not present, allow writes (to avoid blocking CLI tests).
    """
    try:
        import streamlit as _st
        ss = getattr(_st, "session_state", {})
        return bool(ss.get("_solver_busy") or ss.get("_solver_one_button_active") or ss.get("_cert_write_allowed"))
    except Exception:
        return True
def _svr_write_cert_in_bundle(bundle_dir: Path, filename: str, payload: dict) -> Path:
    # solver-only writer guard
    if not _svr_can_write_now():
        raise PermissionError('Writes are restricted to the single-button solver path.')
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
def _svr_guarded_atomic_write_json(path: Path, payload: dict) -> None:
    # Enforce solver-only writes when target is under logs/certs
    try:
        pstr = str(path.as_posix())
    except Exception:
        pstr = str(path)
    if '/logs/certs/' in pstr or pstr.endswith('.bundle.json'):
        if not _svr_can_write_now():
            raise PermissionError('Writes are restricted to the single-button solver path.')
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
            _svr_guarded_atomic_write_json(pth, {"blocks": blocks})
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
        if False:
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
                        }

                        # COVERAGE_JSONL fallback if not declared globally
                        _COV_JSONL = COVERAGE_JSONL if "COVERAGE_JSONL" in globals() else (REPORTS_DIR / "coverage.jsonl")
                        # normalize coverage row schema (final pass)
                        coverage_row = {
                            'prox_label': str(rc.get('prox_label') or ss.get('prox_label') or 'unknown'),
                            'covered': bool(proj_auto_k3 is True),
                            'in_baseline': bool(strict_k3 is True),
                            'distance': float(0.0),  # replace with real metric if available
                            'embed_sig': str(embed_sig_auto or ''),
                            'policy': 'strict__VS__projected(columns@k=3,auto)',
                            'n2': int(rc.get('n2') or 0),
                            'n3': int(rc.get('n3') or 0),
                            'district_id': str(rc.get('district_id') or 'UNKNOWN'),
                            'fixture_label': str(rc.get('fixture_label') or ''),
                        }
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

# ────────────────── Single, always-visible solver button ──────────────────
st.divider()

# init debounce flag once
if "_solver_busy" not in st.session_state:
    st.session_state["_solver_busy"] = False

run_btn_ab = st.button(
    "Run solver (one press → 5 certs; +1 if FILE)",
    key="btn_svr_run_global",
    disabled=st.session_state["_solver_busy"],
    help="Writes strict, projected(auto/file), freezer, and A/B certs into the current bundle."
)

if run_btn_ab:
    if st.session_state["_solver_busy"]:
        st.info("Solver is already running — debounced.")
    else:
        st.session_state["_solver_busy"] = True
        try:
            if "_svr_run" in globals():
                _svr_run()
                lr = st.session_state.get("last_solver_result") or {}
                st.success(f"Solver wrote {lr.get('count', 0)} certs.")
            else:
                st.error("`_svr_run()` is not defined in this scope.")
        except Exception as e:
            st.error(f"Solver run failed: {e}")
        finally:
            st.session_state["_solver_busy"] = False
# ──────────────────────────────────────────────────────────────────────────






             # ============================== Cert & Provenance ==============================
with safe_expander("Cert & provenance (read‑only; solver writes bundles)", expanded=True):
    import os, json, hashlib, platform, time
    from pathlib import Path
    from datetime import datetime

    # ---------- constants ----------
    APP_VERSION    = globals().get("APP_VERSION", "v0.1-core")

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
        pass

        try:
            import streamlit as _st
            if not _st.session_state.get('_solver_one_button_active', False):
                # Read-only outside solver: drop the write
                return
        except Exception:
            # In non-Streamlit contexts, fall back to a direct atomic write
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

    # Safe debug writer (avoid DeltaGenerator in st.write payloads)
    def _safe_write_json(obj, *, label: str | None = None):
        def _coerce(x):
            try:
                json.dumps(x); return x
            except Exception:
                return str(x)
        def _walk(v):
            if isinstance(v, dict):
                return {k: _walk(vv) for k, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [_walk(vv) for vv in v]
            return _coerce(v)
        safe = _walk(obj)
        if label:
            st.caption(label)
        st.code(json.dumps(safe, ensure_ascii=False, indent=2))

    # ---------- UI helpers ----------
    def _chip(label: str, value: str, *, short=8):
        vshort = (value or "")[:short]
        st.markdown(
            f"<span style='display:inline-block;padding:2px 6px;border-radius:8px;"
            f"border:1px solid #ddd;font-size:12px;white-space:nowrap;"
            f"margin-right:6px;background:#fafafa'>{label}: <b>{vshort}</b></span>",
            unsafe_allow_html=True
        )
        return None

    def _copybox(value: str, key: str):
        st.text_input(label="copy", value=value or "", key=key,
                      label_visibility="collapsed", disabled=True)
        return None

    def _delta_line(last_key, write_key):
        if not last_key:
            return "Δ first-write"
        (li, lp, lv, lpj) = last_key
        (ci, cp, cv, cpj) = write_key
        parts = []
        if lp != cp:
            parts.append(f"Δ policy: {lp}→{cp}")
        if lv != cv:
            parts.append(f"Δ pass: {int(lv[0])}{int(lv[1])}→{int(cv[0])}{int(cv[1])}")
        names = ["b_hash","C_hash","H_hash","U_hash","S_hash"]
        for idx, (a,b) in enumerate(zip(li, ci)):
            if a != b:
                parts.append(f"Δ {names[idx]}")
        if lpj != cpj and (cp == "projected:file" or lp == "projected:file"):
            parts.append("Δ projector_hash")
        return "; ".join(parts) or "Δ —"

    # ---------- freeze snapshot (single read; SSOT-only) ----------
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

    # Freshness gate (SSOT-only)
    stale = ssot_is_stale()
    _toggle_key = ensure_unique_widget_key("allow_stale_ssot__cert")
    allow_stale = False  # panel is read-only; writes only from solver press
    if stale and not allow_stale:
        st.warning("Inputs changed since last Overlap — run Overlap to refresh SSOT before writing or reporting.")
        # Don’t stop; we only disable the write later.

    # Canonical, frozen sig + view
    inputs_sig = current_inputs_sig(_ib=ib)

    # Try to publish from pending once if IB was blank
    def _publish_inputs_block_from_pending() -> bool:
        ph = ss.get("_inputs_hashes_pending") or {}
        pd = ss.get("_dims_pending") or {}
        pf = ss.get("_filenames_pending") or {}
        if not ph or not pd:
            return False
        ss["_inputs_block"] = {
            "hashes": dict(ph),
            "dims":   dict(pd),
            "filenames": dict(pf),
            # legacy top-level keys:
            "boundaries_hash": ph.get("boundaries_hash",""),
            "C_hash":          ph.get("C_hash",""),
            "H_hash":          ph.get("H_hash",""),
            "U_hash":          ph.get("U_hash",""),
            "shapes_hash":     ph.get("shapes_hash",""),
        }
        return True

        if (not ib) or (not ib.get("hashes") and not ib.get("boundaries_hash")):
            if _publish_inputs_block_from_pending():
                ib = dict(ss.get("_inputs_block") or {})
                inputs_sig = current_inputs_sig(_ib=ib)
    
        # Raw SSOT toggle (debug)
        pass  # raw SSOT debug removed
    

    # ---------- FILE Π validity & inputs completeness ----------
    file_pi_valid   = bool(ss.get("file_pi_valid", True))
    file_pi_reasons = list(ss.get("file_pi_reasons", []) or [])
 # --- derive 5-hash inputs_sig safely (used only for completeness gating) ---
    ib = ss.get("_inputs_block") or {}
    h  = (ib.get("hashes") or {})
    inputs_sig = [
    str(h.get("hash_d") or h.get("boundaries_hash") or h.get("B_hash") or ""),
    str(h.get("hash_suppC") or h.get("C_hash") or ""),
    str(h.get("hash_suppH") or h.get("H_hash") or ""),
    str(h.get("hash_U") or h.get("U_hash") or ""),
    str(h.get("hash_shapes") or h.get("S_hash") or h.get("shapes_hash") or ""),
        ]
    inputs_complete = all(isinstance(x, str) and x for x in (inputs_sig[0], inputs_sig[1], inputs_sig[2], inputs_sig[4]))

    # --- derive policy + pass_vec safely (self-contained) ---
    policy_raw   = rc.get("policy_tag") or ss.get("overlap_policy_label") or ""
    policy_canon = _canon_policy(policy_raw)
    proj_hash    = rc.get("projector_hash", "") if policy_canon == "projected:file" else ""
    eq2 = bool(((out.get("2") or {}).get("eq", False)))
    eq3 = bool(((out.get("3") or {}).get("eq", False)))
    pass_vec = (eq2, eq3)

    # ---------- A/B one-shot ticket state ----------
    ab_pin                 = dict(ss.get("ab_pin") or {"state":"idle","payload":None,"consumed":False})
    is_ab_pinned           = (ab_pin.get("state") == "pinned")
    ab_ticket_pending      = ss.get("_ab_ticket_pending")
    last_ab_ticket_written = ss.get("_last_ab_ticket_written")
    ticket_required        = bool(is_ab_pinned and (ab_ticket_pending is not None) and (ab_ticket_pending != last_ab_ticket_written))

    write_armed  = bool(ss.get("write_armed", False))

    # ---------- dedupe key (single write gate) ----------
    write_key = (inputs_sig, policy_canon, pass_vec, proj_hash)
    last_key  = ss.get("_last_cert_write_key")

    # ---------- header strip (live debugger) ----------
    c1, c2, c3, c4 = st.columns([2,2,3,3])
    with c1:
        if inputs_complete:
            st.success("Inputs OK")
            _chip("b", inputs_sig[0]); _chip("C", inputs_sig[1])
            _chip("H", inputs_sig[2]); _chip("U", inputs_sig[3]); _chip("S", inputs_sig[4])
            if st.checkbox("Show copyable hashes", value=False, key="copy_hashes_toggle"):
                _copybox(",".join(inputs_sig), key="copy_inputs_sig")
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
        ab_p = ss.get("ab_pin") or {}
        if ab_p.get("state") == "pinned":
            ab = ab_p.get("payload") or {}
            stale_ab = None
            if str((ab.get('embed_sig') or (ab.get('payload') or {}).get('embed_sig',''))) != str(_ab_embed_sig()):
                stale_ab = REASON.AB_STALE_INPUTS_SIG
            elif _canon_policy(ab.get("policy_tag","")) != policy_canon:
                stale_ab = REASON.AB_STALE_POLICY
            elif policy_canon == "projected:file" and (ab.get("projected",{}) or {}).get("projector_hash","") != proj_hash:
                stale_ab = REASON.AB_STALE_PROJECTOR_HASH
            if not stale_ab:
                st.success("A/B: Pinned · Fresh")
            else:
                st.warning(f"A/B: Pinned · Stale ({stale_ab})")
    with c4:
        if not write_armed:
            st.caption("Write: Idle")
        else:
            st.success("Write: Armed (1×)" if (last_key != write_key) or ticket_required else "Write: Armed (but no change)")
    if write_armed:
        st.caption(_delta_line(last_key, write_key))

    # ---------- decisions & witnesses ----------
    # Non-blocking gates: never st.stop() the app; only disable the write.
    write_enabled = True

    # 1) completeness (U optional)
    if not inputs_complete:
        write_enabled = False
        _append_witness({
            "ts": _utc_now_z(),
            "outcome": REASON.SKIP_INPUTS_INCOMPLETE,
            "armed": write_armed,
            "armed_by": ss.get("armed_by",""),
            "key": {
                "inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)
            },
            "ab": ("PINNED" if is_ab_pinned else "NONE"),
            "file_pi": {"mode": ("file" if policy_canon=="projected:file" else policy_canon), "valid": file_pi_valid, "reasons": file_pi_reasons[:3]}
        })
        st.caption("Inputs incomplete — write disabled (you can still view A/B + gallery).")

        # 2) stale SSOT — allow A/B ticket to override (non-blocking)
    if write_enabled and stale and not allow_stale:
        # re-evaluate ticket freshness here (embed_sig-based)
        ab_pin                 = ss.get("ab_pin") or {}
        is_ab_pinned           = (ab_pin.get("state") == "pinned")
        payload                = ab_pin.get("payload") or {}
        current_embed_sig      = _ab_embed_sig()  # ← from the helpers I gave you
        ab_sig_ok              = bool(is_ab_pinned and (payload.get("embed_sig","") == current_embed_sig))
    
        ab_ticket_pending      = ss.get("_ab_ticket_pending")
        last_ab_ticket_written = ss.get("_last_ab_ticket_written")
    
        # one-shot override only if the pin is fresh (sig match) AND ticket advanced
        ticket_required = bool(
            ab_sig_ok and
            (ab_ticket_pending is not None) and
            (ab_ticket_pending != last_ab_ticket_written)
        )
    
        if not ticket_required:
            write_enabled = False
            _append_witness({
                "ts": _utc_now_z(),
                "outcome": REASON.SKIP_SSOT_STALE,
                "armed": write_armed,
                "armed_by": ss.get("armed_by",""),
                "key": {
                    "inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                    "pol": policy_canon,
                    "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}",
                    "pj": _short(proj_hash),
                },
                "ab": ("PINNED" if is_ab_pinned else "NONE"),
                "ab_sig_ok": bool(ab_sig_ok),
                "ab_ticket": {
                    "pending": ab_ticket_pending,
                    "last_written": last_ab_ticket_written,
                },
                "file_pi": {
                    "mode": ("file" if policy_canon=="projected:file" else policy_canon),
                    "valid": file_pi_valid,
                    "reasons": file_pi_reasons[:3],
                },
            })
            st.warning("SSOT stale — write disabled. Run Overlap to refresh or enable the 'Allow writing with stale SSOT' toggle.")
            st.caption("Rendering continues so you can inspect A/B & gallery.")
        else:
            st.info("SSOT is stale, but proceeding due to A/B one-shot ticket override (fresh embed_sig).")
            if not write_armed:
                ss["write_armed"] = True
                ss["armed_by"] = ss.get("armed_by","") or "ab_pinned_catchup"
                write_armed = True


    # 3) file Π validity (FILE mode only)
    if write_enabled and (policy_canon == "projected:file") and (not file_pi_valid):
        write_enabled = False
        _append_witness({
            "ts": _utc_now_z(),
            "outcome": REASON.SKIP_FILE_PI_INVALID,
            "armed": write_armed,
            "armed_by": ss.get("armed_by",""),
            "key": {
                "inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)
            },
            "ab": ("PINNED" if is_ab_pinned else "NONE"),
            "file_pi": {"mode": "file", "valid": False, "reasons": file_pi_reasons[:3]}
        })
        st.caption("FILE Π invalid — write disabled (fix Π or re-freeze from AUTO).")

        # 4) final “should write?” decision
    should_write = write_enabled and write_armed and ((write_key != last_key) or ticket_required)
    
    if not should_write:
        skip_reason = REASON.SKIP_NO_MATERIAL_CHANGE
        if write_enabled and is_ab_pinned and (ab_ticket_pending == last_ab_ticket_written):
            skip_reason = REASON.SKIP_AB_TICKET_ALREADY_WRITTEN
    
        _append_witness({
            "ts": _utc_now_z(),
            "outcome": skip_reason,
            "armed": write_armed,
            "armed_by": ss.get("armed_by",""),
            "key": {
                "inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)
            },
            "ab": ("PINNED" if is_ab_pinned else "NONE"),
            "file_pi": {"mode": ("file" if policy_canon=="projected:file" else policy_canon),
                        "valid": file_pi_valid,
                        "reasons": ([] if write_enabled else file_pi_reasons[:3])}
        })
    
        if write_enabled:
            st.caption("Cert unchanged — skipping rewrite.")
        wrote_cert = False  # mark path and fall through (do not write)
    else:
        wrote_cert = True
    
        # ----- stable run_id per inputs_sig -----
        if ss.get("_last_inputs_sig") != inputs_sig:
            seed = _sha256_hex((":".join(inputs_sig) + f"|{int(time.time())}").encode())[:8]
            ss["_last_inputs_sig"] = inputs_sig
            ss["last_run_id"] = seed
            ss["run_idx"] = 0
        ss["run_idx"] = int(ss.get("run_idx", 0)) + 1




    # ----- assemble cert payload (strict schema) -----
    district_id = (ss.get("_district_info") or {}).get("district_id", ss.get("district_id","UNKNOWN"))
    n2 = int((ib.get("dims") or {}).get("n2") or rc.get("n2") or 0)
    n3 = int((ib.get("dims") or {}).get("n3") or rc.get("n3") or 0)
    lane_mask = list(rc.get("lane_mask_k3") or [])

    # ---------- GF(2) diagnostics (guarded shapes; self-contained) ----------
    def _bottom_row(M): 
        return M[-1] if (M and len(M)) else []
    def _xor(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]
    def _mask_row(row, lm):
        L = min(len(row or []), len(lm or []))
        return [int(row[j]) if int(lm[j]) else 0 for j in range(L)]

    H2 = (getattr(H_obj, "blocks", None).__root__.get("2") if getattr(H_obj, "blocks", None) else []) or []
    d3 = rc.get("d3") or []
    C3 = (getattr(C_obj, "blocks", None).__root__.get("3") if getattr(C_obj, "blocks", None) else []) or []
    I3 = [[1 if i == j else 0 for j in range(len(C3))] for i in range(len(C3))] if C3 else []

    if H2 and d3 and H2[0] and d3[0] and (len(H2[0]) == len(d3)):
        H2d3 = [
            [sum((H2[i][k] & d3[k][j]) & 1 for k in range(len(d3))) & 1 for j in range(len(d3[0]))]
            for i in range(len(H2))
        ]
    else:
        H2d3 = []
    C3pI3 = _xor(C3, I3) if C3 else []

    lane_vec_H2d3  = _mask_row(_bottom_row(H2d3), lane_mask)
    lane_vec_C3pI3 = _mask_row(_bottom_row(C3pI3), lane_mask)

    # A/B freshness helpers
    def _ab_is_fresh(ab_snap: dict, *, rc: dict, ib: dict) -> bool:
        try:
            if tuple(ab_snap.get("inputs_sig") or ()) != tuple(current_inputs_sig(_ib=ib)):
                return False
            if _canon_policy(ab_snap.get("policy_tag","")) != _canon_policy(rc.get("policy_tag","")):
                return False
            if _canon_policy(rc.get("policy_tag","")) == "projected:file":
                pj_now = rc.get("projector_hash","")
                pj_ab  = ((ab_snap.get("projected") or {}).get("projector_hash",""))
                return str(pj_now) == str(pj_ab)
            return True
        except Exception:
            return False

    # A/B freshness
    ab_status = REASON.AB_NONE
    ab_fresh = False
    if is_ab_pinned:
        ab = ab_pin.get("payload") or {}
        if str((ab.get('embed_sig') or (ab.get('payload') or {}).get('embed_sig',''))) != str(_ab_embed_sig()):
            ab_status = REASON.AB_STALE_INPUTS_SIG
        elif _canon_policy(ab.get("policy_tag","")) != policy_canon:
            ab_status = REASON.AB_STALE_POLICY
        elif policy_canon == "projected:file" and (ab.get("projected",{}) or {}).get("projector_hash","") != proj_hash:
            ab_status = REASON.AB_STALE_PROJECTOR_HASH
        else:
            ab_fresh = True
            ab_status = REASON.AB_EMBEDDED
                    # --- Auto-label on write (runs only if fixture_label isn't set yet) ---
    try:
        if not st.session_state.get("fixture_label"):
            rc = st.session_state.get("run_ctx") or {}
            di = st.session_state.get("_district_info") or {}
            district_id = di.get("district_id", "UNKNOWN")
    
            # Prefer your canonical policy helper if present
            if "_policy_tag_now_from_rc" in globals() and callable(globals()["_policy_tag_now_from_rc"]):
                policy_canon = _policy_tag_now_from_rc(rc)
            else:
                policy_canon = str(rc.get("policy_tag","strict"))
    
            # Masked diagnostics on the lanes (shape-safe)
            lane_mask = list(rc.get("lane_mask_k3") or [])
    
            H_used = st.session_state.get("overlap_H") or _load_h_local()
            d3     = rc.get("d3") or []
            C3     = (cmap.blocks.__root__.get("3") or [])
            H2     = (H_used.blocks.__root__.get("2") or [])
            I3     = [[1 if i==j else 0 for j in range(len(C3))] for i in range(len(C3))] if C3 else []
    
            def _shape_ok(A,B): return bool(A and B and A[0] and B[0] and (len(A[0]) == len(B)))
            def _xor(A,B):
                if not A: return [r[:] for r in (B or [])]
                if not B: return [r[:] for r in (A or [])]
                r,c = len(A), len(A[0]); return [[(A[i][j]^B[i][j]) & 1 for j in range(c)] for i in range(r)]
    
            H2d3  = mul(H2, d3) if _shape_ok(H2, d3) else []
            C3pI3 = _xor(C3, I3) if (C3 and C3[0]) else []
    
            bottom_H2d3  = (H2d3[-1]  if (H2d3  and len(H2d3))  else [])
            bottom_C3pI3 = (C3pI3[-1] if (C3pI3 and len(C3pI3)) else [])
            idx = [j for j,m in enumerate(lane_mask or []) if m]
            lane_vec_H2d3  = [bottom_H2d3[j]  for j in idx] if (bottom_H2d3 and idx) else []
            lane_vec_C3pI3 = [bottom_C3pI3[j] for j in idx] if (bottom_C3pI3 and idx) else []
    
            out = st.session_state.get("overlap_out") or {}
            strict_eq3 = bool(((out.get("3") or {}).get("eq", False)))  # optional in registry
    
            snapshot_for_fixture = {
                "identity": {"district_id": district_id},
                "policy":   {"canon": policy_canon},
                "inputs":   {"lane_mask_k3": lane_mask},
                "diagnostics": {
                    "lane_vec_H2@d3": lane_vec_H2d3,
                    "lane_vec_C3+I3": lane_vec_C3pI3,
                },
                "checks":   {"k": {"3": {"eq": strict_eq3}}},
                "growth":   {"growth_bumps": int(st.session_state.get("growth_bumps", 0) or 0)},
            }
    
            fx = match_fixture_from_snapshot(snapshot_for_fixture)
            apply_fixture_to_session(fx)
    except Exception:
        # Never block a cert write on labeling
        pass



    # ---------------- Identity (includes fixture fields) ----------------
    identity = {
        "district_id":   district_id,
        "run_id":        ss.get("last_run_id","00000000"),
        "run_idx":       int(ss.get("run_idx", 0)),
        "field":         ss.get("field_label", "B2"),
        "fixture_label": ss.get("fixture_label",""),
        "fixture_code":  (ss.get("run_ctx") or {}).get("fixture_code",""),
        "fixture_nonce": ss.get("_fixture_nonce", 0),
    }

    # ---------------- Policy ----------------
    policy = {
        "label_raw":      policy_raw,
        "canon":          policy_canon,
        "projector_mode": ("strict" if policy_canon=="strict" else ("file" if policy_canon=="projected:file" else "auto")),
    }
    if policy_canon == "projected:file":
        policy["projector_hash"] = proj_hash

    # ---------------- Inputs + filenames ----------------
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

    # ---- Registry provenance (fixtures.json) ----
    fx_cache = ss.get("_fixtures_cache") or {}
    inputs["registry"] = {
        "fixtures_version": fx_cache.get("version",""),
        "fixtures_path":    fx_cache.get("__path","configs/fixtures.json"),
        "fixtures_hash":    ss.get("_fixtures_bytes_hash",""),
        "ordering":         list(fx_cache.get("ordering") or []),
    }

    # ---------------- Checks / Diagnostics ----------------
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

    # ---------------- A/B embed snapshot ----------------
    ab_embed = {"fresh": bool(ab_fresh)}
    ab_rich = st.session_state.get("ab_compare") or {}
    use_rich = bool(ab_fresh and _ab_is_fresh(ab_rich, rc=rc, ib=ib))
    ab_src = ab_rich if use_rich else (ab_pin.get("payload") or {})

    strict_block = ab_src.get("strict") or {}
    if not strict_block:
        strict_block = {"out": (ab_src.get("strict") or {}).get("out", {})}

    mode_now = str((rc or {}).get("mode", ""))
    proj_hash_now = (
        (rc.get("projector_hash") or "")
        if mode_now == "projected(columns@k=3,file)"
        else (_auto_pj_hash_from_rc(rc or {}) if mode_now == "projected(columns@k=3,auto)" else "")
    )

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
    })
    if not ab_fresh and is_ab_pinned:
        ab_embed["stale_reason"] = ab_status

    # ---------------- Growth / Gallery ----------------
    growth  = {"growth_bumps": int(ss.get("growth_bumps", 0)), "H_diff": ss.get("H_diff","")}
    gallery = {
        "projected_green": bool(ss.get("projected_green", pass_vec[1])),
        "tag": ss.get("gallery_tag",""),
        "strictify": ss.get("gallery_strictify","tbd"),
    }

    # ---------------- Assemble cert ----------------
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

    # UTC stamp & strict JSON hash (timestamp NOT in dedupe key)
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

    # mark A/B ticket as used to prevent duplicate embedded writes on reruns
    if is_ab_pinned and (ab_ticket_pending is not None):
        ss["_last_ab_ticket_written"] = ss.get("_ab_ticket_pending")

    # clear pin only if we actually embedded this write
    if ab_fresh and is_ab_pinned:
        ss["ab_pin"] = {"state":"idle","payload":None,"consumed":True}

    # witness (write)
    _append_witness({
        "ts": cert["written_at_utc"],
        "outcome": REASON.WROTE_CERT,
        "armed": True,
        "armed_by": ss.get("armed_by",""),
        "key": {
            "inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
            "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)
        },
        "ab": (REASON.AB_EMBEDDED if ab_fresh else (REASON.AB_NONE if not is_ab_pinned else ab_status)),
        "file_pi": {"mode": ("file" if policy_canon=="projected:file" else policy_canon), "valid": True, "reasons": []},
        "path": fpath.as_posix()
    })

    # prevent fixture carry-over into the next unrelated cert
    for _k in ("fixture_label","gallery_tag","gallery_strictify"):
        ss[_k] = ""
    _rc_tmp = ss.get("run_ctx") or {}
    if "fixture_label" in _rc_tmp or "fixture_code" in _rc_tmp:
        _rc_tmp.pop("fixture_label", None)
        _rc_tmp.pop("fixture_code", None)
        ss["run_ctx"] = _rc_tmp

    # UI receipt
    st.success(f"Cert written → `{fpath.as_posix()}` · {content12}")
    if st.checkbox("Show copyable path", value=False, key="copy_cert_path_toggle"):
        _copybox(fpath.as_posix(), key="copy_cert_path")

    # optional: auto-build B2 gallery (debounced inside)
    try:
        if "build_b2_gallery" in globals():
            build_b2_gallery(debounce=True)
    except Exception as e:
        st.info(f"(B2 gallery build skipped: {e})")




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
                st.write(
                    f"• {ts} · {ident.get('district_id','?')} · {policy.get('label_raw','?')} · "
                    f"k2/k3={int(k2)}/{int(k3)} · {p.name}{flag}"
                )
                shown += 1
            if shown == 0:
                st.caption("No certs to show with current filter.")
        except Exception as e:
            st.warning(f"Tail listing failed: {e}")
# ============================== CERT BLOCK ==============================






# ---------------- Fixture registry (cache + matcher) ----------------
_FIXTURE_CACHE = {"hash":"", "data":None}

def _sha256_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _load_fixtures_json() -> tuple[dict, str]:
    """Load configs/fixtures.json (or return empty), with hash for cache-busting."""
    p = Path("configs/fixtures.json")
    if not p.exists():
        return {"version":"","ordering":[],"fixtures":[]}, ""
    txt = p.read_text(encoding="utf-8")
    h   = _sha256_text(txt)
    try:
        data = json.loads(txt)
    except Exception:
        data = {"version":"","ordering":[],"fixtures":[]}
    return data, h

def _get_fixtures_cached() -> dict:
    data, h = _load_fixtures_json()
    if h and _FIXTURE_CACHE["hash"] == h and _FIXTURE_CACHE["data"] is not None:
        return _FIXTURE_CACHE["data"]
    _FIXTURE_CACHE["hash"] = h
    _FIXTURE_CACHE["data"] = data
    return data

def _vec_eq(a, b) -> bool:
    aa = [int(x) for x in (a or [])]
    bb = [int(x) for x in (b or [])]
    return (len(aa) == len(bb)) and (aa == bb)

def _apply_fixture_after_overlap(*, rc: dict, diag: dict) -> dict:
    """
    Decide fixture from registry and stamp session + rc.
    Returns the small fixture dict used by cert/gallery.
    """
    ss = st.session_state
    reg = _get_fixtures_cached() or {}
    ordering = list(reg.get("ordering") or [])
    fixtures = list(reg.get("fixtures") or [])

    district = (ss.get("_district_info") or {}).get("district_id", rc.get("district_id","UNKNOWN"))
    policy_canon = str(rc.get("policy_tag") or "strict").lower()

    lane_mask      = list(rc.get("lane_mask_k3") or [])
    lane_vec_H2    = list(diag.get("lane_vec_H2@d3") or [])
    lane_vec_C3pI3 = list(diag.get("lane_vec_C3+I3") or [])

    # Build a lookup by code in declared order
    code_to_fixture = {fx.get("code"): fx for fx in fixtures}
    ordered = [code_to_fixture[c] for c in ordering if c in code_to_fixture] + [fx for fx in fixtures if fx.get("code") not in ordering]

    chosen = {}
    for fx in ordered:
        m = fx.get("match") or {}
        # district
        if m.get("district") and str(m["district"]) != str(district):
            continue
        # policy set (if provided)
        pol_any = [str(x).lower() for x in (m.get("policy_canon_any") or [])]
        if pol_any and (policy_canon not in pol_any):
            continue
        # vectors
        if "lanes" in m and not _vec_eq(m["lanes"], lane_mask):
            continue
        if "H_bottom" in m and not _vec_eq(m["H_bottom"], lane_vec_H2):
            continue
        if "C3_plus_I3_bottom" in m and not _vec_eq(m["C3_plus_I3_bottom"], lane_vec_C3pI3):
            continue
        # boolean strict eq (optional)
        if "strict_eq3" in m:
            # use any strict tag you computed, or recompute eq3_strict if you expose it here
            pass  # optional in this minimal matcher
        chosen = {
            "fixture_code":  fx.get("code",""),
            "fixture_label": fx.get("label",""),
            "tag":           fx.get("tag",""),
            "strictify":     fx.get("strictify","tbd"),
            "growth_bumps":  int(fx.get("growth_bumps", 0)),
        }
        break

    # Fallback (never block)
    if not chosen:
        chosen = {
            "fixture_code":  "",
            "fixture_label": f"{district} • lanes={lane_mask} • H={lane_vec_H2} • C+I={lane_vec_C3pI3}",
            "tag":           "novelty",
            "strictify":     "tbd",
            "growth_bumps":  int(st.session_state.get("growth_bumps", 0) or 0),
        }

    # Stamp into session + rc for cert/gallery
    ss["fixture_label"]   = chosen["fixture_label"]
    ss["gallery_tag"]     = chosen["tag"]
    ss["gallery_strictify"]= chosen["strictify"]
    ss["growth_bumps"]    = chosen["growth_bumps"]

    rc2 = dict(rc)
    rc2["fixture_label"]  = chosen["fixture_label"]
    rc2["fixture_code"]   = chosen["fixture_code"]
    st.session_state["run_ctx"] = rc2

    # Registry provenance into inputs (cert will copy from here)
    fx = _get_fixtures_cached()
    fxh = _FIXTURE_CACHE["hash"] or ""
    ss.setdefault("_fixtures_cache", fx)
    ss["_fixtures_bytes_hash"] = fxh
    return chosen









# ===================== Reports: unified helpers (final) ======================
from pathlib import Path
import os, json as _json, hashlib as _hash, csv as _csv
from datetime import datetime as _dt
import streamlit as st

# ── Safe “io” stub (only if your real one isn’t loaded) ───────────────────────
if "io" not in globals():
    class _IO_STUB:
        @staticmethod
        def parse_cmap(d):
            class _X:
                def __init__(self, d):
                    self.blocks = type("B", (), {"__root__": d.get("blocks", {})})
                def dict(self): return {"blocks": self.blocks.__root__}
            return _X(d or {"blocks": {}})
        @staticmethod
        def parse_boundaries(d): return _IO_STUB.parse_cmap(d)
    io = _IO_STUB()

# ── Tiny utils: time + hashing + atomic writers ───────────────────────────────
def _utc_iso_z() -> str:
    try:
        return _dt.utcnow().replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return _dt.utcnow().replace(microsecond=0).isoformat() + "Z"

def _deep_intify(o):
    if isinstance(o, bool): return 1 if o else 0
    if isinstance(o, list): return [_deep_intify(x) for x in o]
    if isinstance(o, dict): return {k: _deep_intify(v) for k, v in o.items()}
    return o

def _hash_json(obj) -> str:
    canon = _deep_intify(obj)
    b = _json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return _hash.sha256(b).hexdigest()

def _guarded_atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _atomic_write_csv(path: Path, header: list[str], rows: list[list], meta_lines: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        if meta_lines:
            for line in meta_lines:
                f.write(f"# {line}\n")
        w = _csv.writer(f); w.writerow(header); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# ── Projector path + rc normalizers (pure read; no recompute) ─────────────────
def _pj_path_from_context_or_fs() -> str:
    ss = st.session_state
    rc = ss.get("run_ctx") or {}
    # 1) run_ctx choice
    p = (rc.get("projector_filename") or "").strip()
    if p and Path(p).exists(): return p
    # 2) last UI pick
    p = (ss.get("ov_last_pj_path") or "").strip()
    if p and Path(p).exists(): return p
    # 3) registry by district (last hit wins)
    try:
        reg = Path(ss.get("PROJECTORS_DIR","projectors")) / "projector_registry.jsonl"
        di  = (ss.get("_district_info") or {}).get("district_id","")
        if reg.exists():
            latest = ""
            with open(reg, "r", encoding="utf-8") as f:
                for ln in f:
                    try:
                        row = _json.loads(ln)
                        if not di or row.get("district") == di:
                            cand = (row.get("filename") or "").strip()
                            if cand and Path(cand).exists():
                                latest = cand
                    except Exception:
                        continue
            if latest: return latest
    except Exception:
        pass
    # 4) newest in projectors/
    pjdir = Path(ss.get("PROJECTORS_DIR","projectors"))
    if pjdir.exists():
        files = sorted(pjdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files: return files[0].as_posix()
    return ""

def _normalize_projector_in_run_ctx():
    """Idempotent fill for FILE mode: filename/hash if missing (no recompute)."""
    ss = st.session_state
    rc = ss.get("run_ctx") or {}
    m  = str(rc.get("mode","strict"))
    if m.startswith("projected(") and "file" in m:
        fn = (rc.get("projector_filename") or "").strip() or _pj_path_from_context_or_fs()
        if fn and Path(fn).exists():
            rc.setdefault("projector_filename", fn)
            try:
                rc.setdefault("projector_hash", _hash.sha256(Path(fn).read_bytes()).hexdigest())
            except Exception:
                pass
    ss["run_ctx"] = rc
    return rc

def _resolve_projector_from_rc():
    """
    Returns: (mode, submode, filename, projector_hash, projector_diag_or_None)
    Never writes widgets; diag is None here (runner uses lane-mask diag).
    """
    rc = st.session_state.get("run_ctx") or {}
    m  = str(rc.get("mode", "strict")).strip()
    if m == "strict":                          return ("strict", "",   "", "", "")
    if m == "projected(columns@k=3,auto)":     return ("projected","auto","", "", "")
    if m == "projected(columns@k=3,file)":
        fn = (rc.get("projector_filename") or "").strip() or _pj_path_from_context_or_fs()
        pj_hash = rc.get("projector_hash","")
        return ("projected","file", fn, pj_hash, "")
    return ("strict", "", "", "", "")

def _ab_unify_pin():
    """Make sure everyone reads the same key; mirror known variants."""
    ss = st.session_state
    pin = ss.get("ab_pin") or ss.get("ab_pin_file") or ss.get("ab_pin_auto") or {}
    if pin and not ss.get("ab_pin"):
        ss["ab_pin"] = pin
    return ss.get("ab_pin") or {}

# ── Input SSOT block (copy-only; no recompute) ─────────────────────────────────
def _inputs_block_from_session(strict_dims: tuple[int, int] | None = None) -> dict:
    """
    Returns:
      {
        "hashes": {"boundaries_hash","C_hash","H_hash","U_hash","shapes_hash"},
        "dims":   {"n2": int, "n3": int},
        "lane_mask_k3": [...]
      }
    """
    rc = st.session_state.get("run_ctx") or {}
    ih = st.session_state.get("inputs_hashes") or {}

    def _grab(k: str) -> str:
        return ih.get(k) or rc.get(k) or st.session_state.get(k) or ""

    hashes = {
        "boundaries_hash": _grab("boundaries_hash"),
        "C_hash":          _grab("C_hash"),
        "H_hash":          _grab("H_hash"),
        "U_hash":          _grab("U_hash"),
        "shapes_hash":     _grab("shapes_hash"),
    }

    if strict_dims is not None:
        try:
            n2, n3 = int(strict_dims[0]), int(strict_dims[1])
        except Exception:
            n2, n3 = 0, 0
    else:
        try:
            n2 = int(rc.get("n2")) if rc.get("n2") is not None else 0
            n3 = int(rc.get("n3")) if rc.get("n3") is not None else 0
        except Exception:
            n2, n3 = 0, 0

    lane_mask = [int(x) & 1 for x in (rc.get("lane_mask_k3") or [])]
    return {"hashes": hashes, "dims": {"n2": n2, "n3": n3}, "lane_mask_k3": lane_mask}

# Back-compat alias some older names if they’re missing
_inputs_block_from_session_SAFE = _inputs_block_from_session

# ── Hydrate B/C/H from session (NEVER writes widget keys) ─────────────────────
def _reports_hydrate_BCH():
    ss = st.session_state
    # Prefer module-level objects left by A/B path
    B = globals().get("boundaries_obj") or globals().get("boundaries")
    C = globals().get("cmap_obj")       or globals().get("cmap")
    H = globals().get("overlap_H")
    # Safe session copies (avoid widget keys like 'cmap'/'boundaries')
    B = B or ss.get("B_obj") or ss.get("_B_obj")
    C = C or ss.get("C_obj") or ss.get("_C_obj")
    H = H or ss.get("overlap_H") or ss.get("H_obj") or ss.get("_H_obj")

    # Parse from pinned filenames shown in UI if needed
    files = (ss.get("_inputs_block") or {}).get("filenames") or {}
    b_path = ss.get("fname_boundaries") or files.get("boundaries")
    c_path = ss.get("fname_cmap")      or files.get("C")
    h_path = ss.get("fname_h")         or files.get("H")

    def _read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            return None

    if B is None and b_path:
        data = _read_json(b_path)
        if data is not None: B = io.parse_boundaries(data)
    if C is None and c_path:
        data = _read_json(c_path)
        if data is not None: C = io.parse_cmap(data)
    if H is None and h_path:
        data = _read_json(h_path)
        if data is not None: H = io.parse_cmap(data)  # H shares the same schema

    return B, C, H

# ── Math: GF(2) helpers (strict) ──────────────────────────────────────────────
def _gf2_mul(A, B):
    """Pure-Python GF(2) matmul (uses mul/_svr_mul if app already provided)."""
    if "_svr_mul" in globals() and callable(globals()["_svr_mul"]):
        return _svr_mul(A, B)  # type: ignore[name-defined]
    if "mul" in globals() and callable(globals()["mul"]):
        return mul(A, B)       # type: ignore[name-defined]
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

def _strict_R3(H2: list[list[int]], d3: list[list[int]], C3: list[list[int]]) -> list[list[int]]:
    """R3 = (H2 @ d3) XOR (C3 XOR I3). Raises if shapes empty/incompatible."""
    if not (H2 and d3 and C3 and C3[0]): 
        raise RuntimeError("R3_INPUTS_MISSING: need non-empty H2/d3/C3.")
    n3 = len(C3)
    I3 = [[1 if i==j else 0 for j in range(n3)] for i in range(n3)]
    H2d3 = _gf2_mul(H2, d3)
    if not H2d3 or len(H2d3[0]) != n3:
        raise RuntimeError("R3_SHAPE: mul(H2,d3) incompatible with C3.")
    C3pI = [[(C3[i][j]^I3[i][j]) & 1 for j in range(n3)] for i in range(n3)]
    return [[(H2d3[i][j]^C3pI[i][j]) & 1 for j in range(n3)] for i in range(n3)]

def _projected_R3(R3: list[list[int]], P: list[list[int]] | None):
    if not (R3 and P): return []
    if len(R3[0]) != len(P):
        raise RuntimeError(f"R3P_SHAPE: expected R3(*,{len(R3[0])})·Π({len(P)},{len(P[0])}).")
    return _gf2_mul(R3, P)

def _lane_mask_from_d3_matrix(d3: list[list[int]]) -> list[int]:
    if not d3 or not (d3[0] if d3 else []): return []
    rows, n3 = len(d3), len(d3[0])
    return [1 if any(int(d3[i][j]) & 1 for i in range(rows)) else 0 for j in range(n3)]

def _diag_from_mask(lm: list[int]) -> list[list[int]]:
    n = len(lm or [])
    return [[1 if (i == j and int(lm[j]) == 1) else 0 for j in range(n)] for i in range(n)]

# Compat aliases some panels still reference
_diag_from_mask__reports = _diag_from_mask
_diag_from_mask_local    = _diag_from_mask  # older name
_eq_zero                 = lambda M: (not M) or all((x & 1) == 0 for row in M for x in row)
_eq_zero_local           = _eq_zero
_strict_R3_local         = _strict_R3

def _first_tripped_guard(strict_out: dict) -> str:
    """Adapter: if app defines first_tripped_guard, use it; else eq→none/fence."""
    if "first_tripped_guard" in globals() and callable(globals()["first_tripped_guard"]):
        try:
            g = first_tripped_guard(strict_out)
            return g if isinstance(g, str) else "error"
        except Exception:
            return "error"
    k3eq = (strict_out or {}).get("3", {}).get("eq", None)
    if k3eq is True:  return "none"
    if k3eq is False: return "fence"
    return "none"

def _sig_tag_eq(B0, C0, H0, P_active=None):
    """
    Return (lane_mask, tag_strict, eq3_strict, tag_proj, eq3_proj).
    Uses app's residual_tag if available; otherwise classifies locally.
    """
    d3 = (B0.blocks.__root__.get("3") or [])
    H2 = (H0.blocks.__root__.get("2") or [])
    C3 = (C0.blocks.__root__.get("3") or [])
    lm = _lane_mask_from_d3_matrix(d3)
    R3s = _strict_R3(H2, d3, C3)

    def _local_tag(R, mask):
        if not R or not mask: return "none"
        m = len(R)
        def _nz(j): return any(R[i][j] & 1 for i in range(m))
        lanes = any(_nz(j) for j, b in enumerate(mask) if b)
        ker   = any(_nz(j) for j, b in enumerate(mask) if not b)
        if lanes and ker: return "mixed"
        if lanes:         return "lanes"
        if ker:           return "ker"
        return "none"

    if "residual_tag" in globals() and callable(globals()["residual_tag"]):
        try:    tag_s = residual_tag(R3s, lm)  # type: ignore[name-defined]
        except Exception: tag_s = "error"
    else:
        tag_s = _local_tag(R3s, lm)

    eq_s = _eq_zero(R3s)

    if P_active:
        R3p = _projected_R3(R3s, P_active)
        if "residual_tag" in globals() and callable(globals()["residual_tag"]):
            try:    tag_p = residual_tag(R3p, lm)  # type: ignore[name-defined]
            except Exception: tag_p = "error"
        else:
            tag_p = _local_tag(R3p, lm)
        eq_p = _eq_zero(R3p)
    else:
        tag_p, eq_p = None, None

    return lm, tag_s, bool(eq_s), tag_p, (None if eq_p is None else bool(eq_p))

# ── Policy + validation shims (copy-only) ─────────────────────────────────────
if "SCHEMA_VERSION" not in globals(): SCHEMA_VERSION = "1.1.0"
if "FIELD" not in globals():           FIELD          = "GF(2)"
if "APP_VERSION" not in globals():     APP_VERSION    = "v0.1-core"

def _policy_block_from_run_ctx(rc: dict) -> dict:
    mode = str(rc.get("mode", "strict"))
    if mode == "strict":
        return {"policy_tag":"strict","projector_mode":"strict","projector_filename":"","projector_hash":""}
    if mode == "projected(columns@k=3,auto)":
        diag_hash = _hash.sha256(
            _json.dumps(rc.get("lane_mask_k3") or [], separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {"policy_tag":"projected(columns@k=3,auto)","projector_mode":"auto","projector_filename":"","projector_hash":diag_hash}
    return {
        "policy_tag":"projected(columns@k=3,file)","projector_mode":"file",
        "projector_filename": rc.get("projector_filename","") or "",
        "projector_hash": rc.get("projector_hash","") or "",
    }

def file_validation_failed() -> bool:
    """Soft gate used for UI hints; never blocks runner."""
    rc = st.session_state.get("run_ctx") or {}
    if rc.get("mode") != "projected(columns@k=3,file)": return False
    fn = (rc.get("projector_filename") or "").strip()
    if not fn or not Path(fn).exists(): return True
    try:
        if rc.get("projector_hash"): return False
        st.session_state["run_ctx"]["projector_hash"] = _hash.sha256(Path(fn).read_bytes()).hexdigest()
        return False
    except Exception:
        return False

# ── Π chooser (prefer app; fallback FILE) ─────────────────────────────────────
def _choose_active_safe(cfg: dict, boundaries_obj):
    if "projector_choose_active" in globals() and callable(globals()["projector_choose_active"]):
        try:
            ret = projector_choose_active(cfg, boundaries_obj)  # type: ignore[name-defined]
            if isinstance(ret, tuple) and len(ret) == 2: return ret
            if isinstance(ret, dict): return (ret.get("P_active", []), ret)
            return (ret, {})
        except Exception:
            pass

    pj_path = (cfg.get("projector_files") or {}).get("3", "") or ""
    P_active = []
    try:
        payload = _json.loads(Path(pj_path).read_text(encoding="utf-8"))
        P_active = (payload.get("blocks", {}) or {}).get("3", []) or []
    except Exception:
        P_active = []

    d3 = (boundaries_obj.blocks.__root__.get("3") or [])
    n3 = len(d3[0]) if (d3 and d3[0]) else 0
    lm_truth = [1 if any(d3[i][j] & 1 for i in range(len(d3))) else 0 for j in range(n3)] if n3 else []
    meta = {
        "mode": "projected(columns@k=3,file)" if pj_path else "projected(columns@k=3,auto)",
        "projector_filename": pj_path,
        "projector_hash": (_hash.sha256(_json.dumps({"blocks":{"3":P_active}}, separators=(",",":"), sort_keys=True).encode("utf-8")).hexdigest()
                           if P_active else ""),
        "projector_consistent_with_d": bool(P_active) and (len(P_active)==n3 and (n3==0 or len(P_active[0])==n3)),
        "d3": d3, "n3": n3, "lane_mask": lm_truth, "P_active": P_active,
    }
    return (P_active, meta)

# Back-compat aliases
_cfg_from_policy_safe__reports = lambda mode, pj=None: {"source":{"3":"file" if "file" in mode else ("auto" if "auto" in mode else "strict")},
                                                        "projector_files":{"3": (pj or "")}}
_pj_path_from_context_or_fs__reports = _pj_path_from_context_or_fs
_choose_active_safe__reports        = _choose_active_safe

# ── Optional U hooks (Fence path) — no-op fallbacks if app doesn’t provide them
if "get_carrier_mask" not in globals():
    def get_carrier_mask(U_obj=None): return []
if "set_carrier_mask" not in globals():
    def set_carrier_mask(U_obj, mask):
        st.session_state["_u_mask_override"] = mask
        return True


# -- Lane-mask pin + projector diag (used by runner) --------------------------
def _pin_lane_mask_and_projector(d3_base, n2, n3):
    """
    Return (lane_mask, P_diag, selected_cols) and pin lane_mask_k3 into run_ctx if missing.
    - lane_mask: list[int] length n3
    - P_diag: n3×n3 diagonal matrix from lane_mask (or None if empty)
    - selected_cols: set of j where lane_mask[j] == 1
    """
    rc = st.session_state.get("run_ctx") or {}

    # prefer SSOT pin from solver
    lm = rc.get("lane_mask_k3")
    if not isinstance(lm, list) or not lm:
        if not d3_base or not (n2 and n3):
            lm = []
        else:
            lm = [
                1 if any(int(d3_base[i][j]) & 1 for i in range(n2)) else 0
                for j in range(n3)
            ]
        rc["lane_mask_k3"] = lm
        st.session_state["run_ctx"] = rc  # safe: not a widget key

    lm = [int(x) & 1 for x in (lm or [])]
    P_diag = _diag_from_mask(lm) if lm else None
    selected_cols = {j for j, b in enumerate(lm) if b == 1}
    return lm, P_diag, selected_cols


def reports_dims_from_d3(d3: list[list[int]]) -> tuple[int, int]:
    return len(d3), (len(d3[0]) if (d3 and d3[0]) else 0)

# ── One-time normalizers (safe to call at panel import) ───────────────────────
_normalize_projector_in_run_ctx()
_ab_unify_pin()
# =================== /Reports: unified helpers (final) ======================



# ──────────────────────────────────────────────────────────────────────────────
# Reports: Perturbation sanity (d3 flips, lanes-only) + optional Fence stress
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import os, json as _json, io as _io, csv as _csv
from datetime import datetime as _dt
import hashlib as _hash

def _eq_zero_local(M): 
    return (not M) or all((x & 1) == 0 for row in M for x in row)

def _diag_from_mask_local(mask):
    n = len(mask or [])
    return [[1 if (i==j and int(mask[j])==1) else 0 for j in range(n)] for i in range(n)]

def run_reports__perturb_and_fence(*, max_flips: int, seed: str, include_fence: bool, enable_witness: bool):
    ss = st.session_state
    REPORTS_DIR = Path(ss.get("REPORTS_DIR", "reports")); REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure run_ctx has a mode + projector filename (copy-only)
    rc = (ss.get("run_ctx") or {}).copy()
    rc["mode"] = rc.get("mode") or "projected(columns@k=3,file)"
    rc["projector_filename"] = rc.get("projector_filename") or _pj_path_from_context_or_fs()
    ss["run_ctx"] = rc

        # ─── Preflight: hydrate B/C/H and assert H2/d3/C3 exist ───
    B0, C0, H0 = _reports_hydrate_BCH()
    
    # Pull blocks once
    d3_base = (B0.blocks.__root__.get("3") or []) if B0 else []
    C3      = (C0.blocks.__root__.get("3") or []) if C0 else []
    H2      = (H0.blocks.__root__.get("2") or []) if H0 else []
    
    # Presence/shape sniff only (no matmul here)
    n2 = len(d3_base)
    n3 = len(d3_base[0]) if (n2 and d3_base[0]) else 0
    if not (n2 and n3 and H2 and (H2[0] if H2 else []) and C3 and (C3[0] if C3 else [])):
        st.error("Reports: missing H2/d3/C3 — run Overlap/Cert once to freeze SSOT.")
        st.stop()
    
    st.caption(f"Reports inputs live → H2:{len(H2)}×{len(H2[0]) if H2 and H2[0] else 0} · d3:{n2}×{n3} · C3:{len(C3)}×{len(C3[0]) if C3 and C3[0] else 0}")
        
      # --- Lane-mask pin & projector diag (final/clean) ---
    rc = st.session_state.get("run_ctx") or {}
    
    lane_mask = [int(x) & 1 for x in (rc.get("lane_mask_k3") or [])]
    if not lane_mask:
        # derive from d3 once and pin for downstream consumers
        lane_mask = [1 if any(int(d3_base[i][j]) & 1 for i in range(n2)) else 0 for j in range(n3)]
        rc["lane_mask_k3"] = lane_mask
        st.session_state["run_ctx"] = rc
    
    P_diag = _diag_from_mask_local(lane_mask) if lane_mask else None
    
    # compute set FIRST
    selected_cols = {j for j, b in enumerate(lane_mask) if b == 1}
    
    # then aliases (for older code paths)
    lane_mask, P_diag, selected_cols = _pin_lane_mask_and_projector(d3_base, n2, n3)
    lanes = lane_mask          # back-compat
    lane_cols = selected_cols  # back-compat
    
    st.caption(f"Lane mask → {lane_mask} (lanes: {len(selected_cols)}/{n3})")


    
    # Build diagonal projector once
    P_diag = _diag_from_mask_local(lane_mask) if lane_mask else None
    
    # Flip domain = lanes-only
    selected_cols = {j for j, b in enumerate(lane_mask) if b == 1}
    
    # Tiny visibility so you can sanity check quickly
    st.caption(f"Lane mask → {lane_mask} (lanes: {len(selected_cols)}/{n3})")


    # Projector validation tag (copy-only)
    proj_status = {"status": "OK", "na_reason_code": ""}
    try:
        if rc.get("mode") == "projected(columns@k=3,file)" and file_validation_failed():
            proj_status = {"status":"N/A", "na_reason_code":"P3_FILE_INVALID"}
    except Exception:
        pass

    # Baseline
    R3_base = _strict_R3(H2, d3, C3)
    k3_strict_base = _eq_zero_local(R3_base)
    k3_proj_base   = (_eq_zero_local(_projected_R3(R3_base, P_diag)) if P_diag else None)

    # Flip domain: lanes-only
    lane_cols = {j for j,b in enumerate(lanes) if b==1}
    h = int(_hash.sha256(str(seed).encode("utf-8")).hexdigest(), 16)
    def _flip_targets(n2_, n3_, budget):
        i = (h % max(1, n2_)) if n2_ else 0
        j = ((h >> 8) % max(1, n3_)) if n3_ else 0
        for k in range(int(budget)):
            yield (i, j, k)
            i = (i + 1 + (h % 3)) % (n2_ or 1)
            j = (j + 2 + ((h >> 5) % 5)) % (n3_ or 1)

    rows_csv, results_json = [], []
    total_flips = in_domain_flips = 0
    for (r, c, k) in _flip_targets(n2, n3, max_flips):
        if c not in lane_cols:
            total_flips += 1
            rows_csv.append([k, "none", "none", "off-domain (ker column)"])
            results_json.append({"flip_id":int(k),"flip":{"row":int(r),"col":int(c),"lane_col":False,"skip_reason":"off-domain"},
                                 "baseline":{"k3_strict":bool(k3_strict_base),"k3_projected":(None if k3_proj_base is None else bool(k3_proj_base))}})
            continue
        total_flips += 1; in_domain_flips += 1
        d3_mut = [row[:] for row in d3]; d3_mut[r][c] ^= 1
        dB = B0.dict() if hasattr(B0, "dict") else {"blocks": {}}; dB = _json.loads(_json.dumps(dB))
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
        meta_lines=[f"schema_version={SCHEMA_VERSION}", f"saved_at={_utc_iso_z()}",
                    f"run_id={(ss.get('run_ctx') or {}).get('run_id','')}", f"app_version={APP_VERSION}",
                    f"seed={seed}", f"n2={n2}", f"n3={n3}"],
    )

    # JSON (copy-only: policy + inputs)
    policy_block = _policy_block_from_run_ctx(rc)
    inputs_block = _inputs_block_from_session(strict_dims=(n2, n3))
    need = ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")
    if not all((inputs_block.get("hashes") or {}).get(k, "") for k in need):
        st.error("INPUT_HASHES_MISSING: wire SSOT from Cert/Overlap; backfill disabled.")
        st.stop()

    payload = {
        "schema_version": SCHEMA_VERSION, "written_at_utc": _utc_iso_z(), "app_version": APP_VERSION, "field": FIELD,
        "identity": {"run_id": rc.get("run_id",""), "district_id": rc.get("district_id",""), "fixture_nonce": rc.get("fixture_nonce","")},
        "projector_validation": proj_status,
        "projected_check": {"enabled": bool(P_diag), "na_reason_code": ("" if P_diag else "LANE_MASK_MISSING")},
        "policy": policy_block, "inputs": inputs_block,
        "baseline": {"k3_strict": bool(k3_strict_base), "k3_projected": (None if k3_proj_base is None else bool(k3_proj_base))},
        "run": {"max_flips": int(max_flips), "domain": "d3_supports", "lanes_only": True, "seed": str(seed)},
        "results": results_json, "summary": {"flips": int(total_flips), "in_domain_flips": int(in_domain_flips)},
        "integrity": {"content_hash": ""},
    }
    payload["integrity"]["content_hash"] = _hash_json(payload)
    jname = f"perturbation_sanity__{payload['integrity']['content_hash'][:12]}.json"
    jpath = REPORTS_DIR / jname
    _guarded_atomic_write_json(jpath, payload)

    st.success(f"Perturbation sanity → {csv_path.name} + {jname}")
    with open(jpath, "rb") as jf: st.download_button("Download perturbation_sanity.json", jf, file_name=jname, key=f"dl_ps_json_{jname[-12:]}")
    with open(csv_path, "rb") as cf: st.download_button("Download perturbation_sanity.csv", cf, file_name=csv_path.name, key=f"dl_ps_csv_{csv_path.stem[-8:]}")

    # Fence (optional) — unchanged from your working variant
    if include_fence:
        # (reuse your fence code here; it already worked once BCH are hydrated)
        pass

# ── UI
with st.container():
    st.markdown("### Reports: Perturbation sanity & Fence stress")
    c1, c2, c3 = st.columns([1.2,1.2,1])
    with c1: ps_max = st.number_input("Max flips (d3 supports)", 1, 500, 24, 1, key="ps_max_one")
    with c2: ps_seed = st.text_input("Seed", value="ps-seed-1", key="ps_seed_one")
    with c3: run_fence = st.checkbox("Fence (U)", value=True, key="fs_on_one")
    try:
        rc_now = st.session_state.get("run_ctx") or {}
        if rc_now.get("mode") == "projected(columns@k=3,file)" and file_validation_failed():
            st.info("Π(FILE) not validated; run will mark projector_validation: N/A (P3_FILE_INVALID).")
    except Exception:
        pass
    if st.button("Run perturbation (+Fence)", key="btn_run_pf_one"):
        run_reports__perturb_and_fence(max_flips=int(ps_max), seed=str(ps_seed), include_fence=bool(run_fence), enable_witness=False)



          













# ================================== Coverage · Helpers (idempotent) ==================================
import random as _random

# ---------- Coverage bootstrap (place ABOVE self-test & expanders) ----------
if "_cov_baseline_open" not in st.session_state:
    st.session_state["_cov_baseline_open"] = True
if "_cov_sampling_open" not in st.session_state:
    st.session_state["_cov_sampling_open"] = True
if "_cov_defaults" not in st.session_state:
    st.session_state["_cov_defaults"] = {
        "random_seed": 1337,
        "bit_density": 0.40,
        "sample_count": 200,
    }
st.session_state.setdefault("normalizer_ok", True)

if "_rand_gf2_matrix" not in globals():
    def _rand_gf2_matrix(rows: int, cols: int, density: float, rng: _random.Random) -> list[list[int]]:
        density = max(0.0, min(1.0, float(density)))
        return [[1 if rng.random() < density else 0 for _ in range(int(cols))] for _ in range(int(rows))]

if "_gf2_rank" not in globals():
    def _gf2_rank(M: list[list[int]]) -> int:
        if not M: return 0
        A = [row[:] for row in M]
        m, n = len(A), len(A[0])
        r = c = 0
        while r < m and c < n:
            pivot = next((i for i in range(r, m) if A[i][c] & 1), None)
            if pivot is None:
                c += 1
                continue
            if pivot != r:
                A[r], A[pivot] = A[pivot], A[r]
            for i in range(r + 1, m):
                if A[i][c] & 1:
                    A[i] = [(A[i][j] ^ A[r][j]) for j in range(n)]
            r += 1; c += 1
        return r

def _cov_keep_open_sampling():
    st.session_state["_cov_sampling_open"] = True

def _cov_keep_open_baseline():
    st.session_state["_cov_baseline_open"] = True


if "_col_support_pattern" not in globals():
    def _col_support_pattern(M: list[list[int]]) -> list[str]:
        if not M: return []
        rows, cols = len(M), len(M[0])
        cols_bits = []
        for j in range(cols):
            bits = ''.join('1' if (M[i][j] & 1) else '0' for i in range(rows))
            cols_bits.append(bits)
        cols_bits.sort()
        return cols_bits

if "_coverage_signature" not in globals():
    def _coverage_signature(d_k1: list[list[int]], n_k: int) -> str:
        rk = _gf2_rank(d_k1)
        ker = max(0, int(n_k) - rk)
        patt = _col_support_pattern(d_k1)  # columns, top→bottom; lexicographically sorted
        return f"rk={rk};ker={ker};pattern=[{','.join(patt)}]"

# ---------- Baseline normalization helpers ----------
if "_extract_pattern_tokens" not in globals():
    import re as _re
    _PAT_RE = _re.compile(r"pattern=\[\s*(.*?)\s*\]")
    def _extract_pattern_tokens(sig: str) -> list[str]:
        m = _PAT_RE.search(sig or "")
        if not m:
            return []
        inner = m.group(1)
        toks = []
        for raw in inner.split(","):
            s = "".join(ch for ch in raw.strip() if ch in "01")
            if s != "":
                toks.append(s)
        return toks

if "_rebuild_signature_with_tokens" not in globals():
    import re as _re
    def _rebuild_signature_with_tokens(sig: str, tokens: list[str]) -> str:
        if "pattern=[" not in sig:
            return f"pattern=[{','.join(tokens)}]"
        return _re.sub(r"pattern=\[[^\]]*\]", f"pattern=[{','.join(tokens)}]", sig)

if "_normalize_signature_line_to_n2" not in globals():
    def _normalize_signature_line_to_n2(sig: str, *, n2: int, n3: int) -> tuple[str, str]:
        """
        Convert pattern=[t1,t2,t3] to n3 tokens, each exactly n2 bits (top-to-bottom).
        - If token len > n2: keep TOP n2 bits (leftmost).
        - If token len < n2: left-pad with '0' to n2.
        - If token count != n3: pad/truncate with '0'*n2 to exactly n3 tokens.
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

if "_dedupe_keep_order" not in globals():
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen, out = set(), []
        for s in items:
            if s not in seen:
                seen.add(s); out.append(s)
        return out

# Optional: infer n2/n3 when demo-baseline is used without a fixture
if "_autofill_dims_from_session" not in globals():
    def _autofill_dims_from_session():
        rc = st.session_state.get("run_ctx") or {}
        n2 = int(rc.get("n2") or 0); n3 = int(rc.get("n3") or 0)
        if n2 > 0 and n3 > 0:
            return n2, n3
        # try boundaries
        B = st.session_state.get("boundaries")
        try:
            if B and hasattr(B, "blocks"):
                blocks = getattr(B.blocks, "__root__", {}) or {}
                d3 = blocks.get("3") or []
                if d3:
                    n2_ = len(d3); n3_ = len(d3[0]) if d3 and d3[0] else 0
                    if n2_ and n3_:
                        rc["n2"], rc["n3"] = int(n2_), int(n3_); st.session_state["run_ctx"] = rc
                        return rc["n2"], rc["n3"]
        except Exception:
            pass
        return 0, 0
        

# ----- Helpers: add pattern(s) to baseline, normalized & dedup -----
def _ensure_dims_or_seed_demo():
    """Returns (n2, n3); seeds demo dims (2,3) if absent so UI flows even with no fixture."""
    rc = st.session_state.get("run_ctx") or {}
    n2 = int(rc.get("n2") or 0)
    n3 = int(rc.get("n3") or 0)
    if n2 <= 0 or n3 <= 0:
        n2, n3 = 2, 3  # safe demo defaults
        rc["n2"], rc["n3"] = n2, n3
        st.session_state["run_ctx"] = rc
    return n2, n3

def _add_normalized_pattern_to_baseline(pattern_str: str, rk: str|None=None, ker: str|None=None):
    """
    pattern_str must look like 'pattern=[..,..,..]'. We normalize to n2×n3, then
    insert a full signature string into known_signatures, and just pattern into
    known_signatures_patterns. Dedup, keep order.
    """
    if not pattern_str.startswith("pattern=["):
        # allow raw tokens ['00','01','11'] too
        toks = [t.strip() for t in pattern_str.split(",") if t.strip()]
        pattern_str = f"pattern=[{','.join(toks)}]"

    n2, n3 = _ensure_dims_or_seed_demo()

    # Build a minimal full signature (rk/ker optional, you can overwrite later)
    prefix = []
    if rk is not None:  prefix.append(f"rk={rk}")
    if ker is not None: prefix.append(f"ker={ker}")
    base_sig = (";".join(prefix) + ";" if prefix else "") + pattern_str

    # Normalize full + pattern to n2×n3
    full_norm, patt_norm = _normalize_signature_line_to_n2(base_sig, n2=n2, n3=n3)

    rc0 = st.session_state.get("run_ctx") or {}
    # full list (for human provenance)
    sigs_full = list(rc0.get("known_signatures") or [])
    if full_norm not in sigs_full:
        sigs_full.append(full_norm)
    rc0["known_signatures"] = sigs_full

    # patterns-only set (for O(1) membership in the sampler)
    patt_only = set(rc0.get("known_signatures_patterns") or [])
    patt_only.add(patt_norm)
    rc0["known_signatures_patterns"] = sorted(patt_only)

    st.session_state["run_ctx"] = rc0
    return full_norm, patt_norm
  
 

# ============================== Coverage · Bootstrap & Helpers ==============================
# Keep these idempotent (safe to re-run)
if "_cov_baseline_open" not in st.session_state:
    st.session_state["_cov_baseline_open"] = True
if "_cov_sampling_open" not in st.session_state:
    st.session_state["_cov_sampling_open"] = True
if "_cov_defaults" not in st.session_state:
    st.session_state["_cov_defaults"] = {"random_seed": 1337, "bit_density": 0.40, "sample_count": 200}
st.session_state.setdefault("normalizer_ok", True)
st.session_state.setdefault("_cov_snapshots", {})  # key -> snapshot dict

# no-op keep-open for any button inside the expander
if "_cov_keep_open_sampling" not in globals():
    def _cov_keep_open_sampling():
        st.session_state["_cov_sampling_open"] = True

# Small utils
import datetime as _dt, hashlib as _hashlib, json as _json
from io import StringIO as _StringIO
import pandas as _pd

def _now_utc_iso_Z():
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _sha256_bytes(b: bytes) -> str:
    return _hashlib.sha256(b).hexdigest()

def _sha256_json(o: dict) -> str:
    s = _json.dumps(o, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(s)

def _cov_key(n2,n3,seed,density,count,policy):
    k = _json.dumps({"n2":int(n2),"n3":int(n3),"seed":int(seed),"density":float(density),
                     "count":int(count),"policy":str(policy)}, sort_keys=True,
                     separators=(",",":")).encode("utf-8")
    return _hashlib.sha256(k).hexdigest()[:12]

def _disable_with(msg: str):
    st.info(msg)
    return True

# ================================== Coverage Baseline (load + normalize) ==================================
with st.expander("Coverage Baseline (load + normalize to n₂-bit columns)",
                 expanded=st.session_state.get("_cov_baseline_open", True)):
    st.session_state["_cov_baseline_open"] = True

    n2_active, n3_active = _autofill_dims_from_session()
    st.caption(f"Active fixture dims → n₂={n2_active}, n₃={n3_active}")

    up = st.file_uploader("Upload baseline (.json / .jsonl)", type=["json", "jsonl"], key="cov_baseline_up")
    pasted = st.text_area("Or paste signatures (JSON list or one-per-line)", value="", key="cov_baseline_paste")
    norm_on = st.checkbox("Normalize to n₂-bit columns on load", value=True, key="cov_norm_on")

    DEMO_BASELINE = [
        "rk=2;ker=1;pattern=[110,110,000]",
        "rk=2;ker=1;pattern=[101,101,000]",
        "rk=2;ker=1;pattern=[011,011,000]",
        "rk=3;ker=0;pattern=[111,111,111]",
        "rk=1;ker=2;pattern=[100,000,000]",
    ]
    CANONICAL_D2D3 = [
        "rk=2;ker=1;pattern=[11,11,00]",
        "rk=2;ker=1;pattern=[10,10,00]",
        "rk=2;ker=1;pattern=[01,01,00]",
        "rk=2;ker=1;pattern=[00,01,11]",
        "rk=2;ker=1;pattern=[01,11,10]",
        "rk=2;ker=1;pattern=[11,10,01]",
        "rk=3;ker=0;pattern=[11,11,11]",
        "rk=1;ker=2;pattern=[11,00,00]",
        "rk=1;ker=2;pattern=[00,11,00]",
        "rk=1;ker=2;pattern=[00,00,11]",
    ]

    def _parse_json_baseline(bytes_data: bytes, name: str) -> list[str]:
        try:
            txt = bytes_data.decode("utf-8", errors="ignore")
        except Exception:
            return []
        sigs = []
        try:
            if name.lower().endswith(".jsonl"):
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                        if isinstance(obj, str):
                            sigs.append(obj.strip())
                        elif isinstance(obj, dict) and "signature" in obj:
                            sigs.append(str(obj["signature"]).strip())
                    except Exception:
                        if "pattern=[" in line:
                            sigs.append(line)
            else:
                obj = _json.loads(txt)
                if isinstance(obj, list):
                    for v in obj:
                        if isinstance(v, str):
                            sigs.append(v.strip())
                        elif isinstance(v, dict) and "signature" in v:
                            sigs.append(str(v["signature"]).strip())
                elif isinstance(obj, dict) and isinstance(obj.get("signatures"), list):
                    for v in obj["signatures"]:
                        if isinstance(v, str):
                            sigs.append(v.strip())
                        elif isinstance(v, dict) and "signature" in v:
                            sigs.append(str(v["signature"]).strip())
        except Exception:
            pass
        return [s for s in sigs if "pattern=[" in s]

    def _parse_pasted_any(p: str) -> list[str]:
        try:
            maybe = _json.loads(p or "")
            if isinstance(maybe, list):
                out = []
                for v in maybe:
                    if isinstance(v, str) and "pattern=[" in v:
                        out.append(v.strip())
                if out:
                    return out
        except Exception:
            pass
        out = []
        for line in (p or "").splitlines():
            s = line.strip()
            if s and "pattern=[" in s:
                out.append(s)
        return out

    def _save_baseline_lines(lines: list[str], *, norm: bool):
        # Seed demo dims if needed so loader never hard-stops UI
        n2, n3 = _autofill_dims_from_session()
        if (n2 <= 0 or n3 <= 0) and norm:
            rc_seed = st.session_state.get("run_ctx") or {}
            rc_seed["n2"], rc_seed["n3"] = 2, 3
            st.session_state["run_ctx"] = rc_seed
            n2, n3 = 2, 3

        if norm:
            norm_full, norm_patt = [], []
            for s in lines:
                f, p = _normalize_signature_line_to_n2(s, n2=int(n2), n3=int(n3))
                norm_full.append(f); norm_patt.append(p)
            lines = _dedupe_keep_order(norm_full)
            patt_only = set(norm_patt)
        else:
            patt_only = set()
            for s in lines:
                toks = _extract_pattern_tokens(s)
                if toks:
                    patt_only.add(f"pattern=[{','.join(toks)}]")

        if not patt_only:
            st.warning("No usable patterns found after parsing/normalize.")
            return

        rc0 = st.session_state.get("run_ctx") or {}
        rc0["known_signatures"] = list(lines)
        rc0["known_signatures_patterns"] = sorted(patt_only)
        st.session_state["run_ctx"] = rc0
        st.success(f"Loaded baseline ({len(lines)} signatures; normalized={norm}).")
        st.caption(f"Baseline now has {len(patt_only)} patterns.")

    cols = st.columns(4)
    if cols[0].button("Load from upload / paste", key="cov_btn_load",
                      on_click=lambda: st.session_state.update(_cov_baseline_open=True)):
        loaded = []
        if up is not None:
            loaded.extend(_parse_json_baseline(up.getvalue(), up.name))
        loaded.extend(_parse_pasted_any(pasted))
        loaded = [s for s in loaded if s]
        if not loaded:
            st.error("No valid signatures found.")
        else:
            _save_baseline_lines(loaded, norm=norm_on)

    if cols[1].button("Load canonical D2/D3 list", key="cov_btn_canonical",
                      on_click=lambda: st.session_state.update(_cov_baseline_open=True)):
        _save_baseline_lines(CANONICAL_D2D3[:], norm=True)

    if cols[2].button("Use demo baseline (legacy → normalize)", key="cov_btn_demo",
                      on_click=lambda: st.session_state.update(_cov_baseline_open=True)):
        _save_baseline_lines(DEMO_BASELINE[:], norm=True)

    if cols[3].button("Clear baseline", key="cov_btn_clear",
                      on_click=lambda: st.session_state.update(_cov_baseline_open=True)):
        rc0 = st.session_state.get("run_ctx") or {}
        rc0["known_signatures"] = []
        rc0["known_signatures_patterns"] = []
        st.session_state["run_ctx"] = rc0
        st.info("Cleared baseline.")

    # Preview
    rc_view = st.session_state.get("run_ctx") or {}
    ks = list(rc_view.get("known_signatures") or [])
    if ks:
        st.caption(f"Baseline active: {len(ks)} signatures. Showing up to 5:")
        st.code("\n".join(ks[:5] + (["…"] if len(ks) > 5 else [])), language="text")
    else:
        st.caption("Baseline inactive (known_signatures is empty).")

# ===================== Coverage · Normalizer self-test (gate sampling) =====================
normalizer_ok = True
with st.expander("Coverage · Normalizer self-test", expanded=False):
    tests = [
        ("rk=2;ker=1;pattern=[110,110,000]", "pattern=[11,11,00]"),
        ("rk=1;ker=2;pattern=[100,000,000]", "pattern=[10,00,00]"),
        ("pattern=[011,011,000]",           "pattern=[01,01,00]"),
    ]
    n2t, n3t = _autofill_dims_from_session()
    if n2t == 0 or n3t == 0:
        n2t, n3t = 2, 3

    for raw, expect in tests:
        _, patt = _normalize_signature_line_to_n2(raw, n2=int(n2t), n3=int(n3t))
        st.text(f"{raw} -> {patt} (expect {expect})")
        if patt != expect:
            normalizer_ok = False

    if normalizer_ok:
        st.success("Normalizer OK")
    else:
        st.error("COVERAGE_NORMALIZER_FAILED")

st.session_state["normalizer_ok"] = normalizer_ok

# ===================== Coverage Sampling (snapshot-based; sticky UI, non-blocking) =====================
with st.expander("Coverage Sampling", expanded=st.session_state.get("_cov_sampling_open", True)):
    st.session_state["_cov_sampling_open"] = True

    # Controls (deterministic)
    dflt = st.session_state["_cov_defaults"]
    c1, c2, c3, c4 = st.columns(4)
    random_seed  = c1.number_input("random_seed", min_value=0, value=int(dflt["random_seed"]), step=1, key="cov_seed_num")
    bit_density  = c2.number_input("bit_density", min_value=0.0, max_value=1.0,
                                   value=float(dflt["bit_density"]), step=0.05, format="%.2f", key="cov_density_num")
    sample_count = c3.number_input("sample_count", min_value=1, value=int(dflt["sample_count"]), step=50, key="cov_samples_num")
    policy       = c4.selectbox("policy", options=["strict"], index=0, help="Coverage uses strict.", key="cov_policy_sel")

    # ---------- Soft guards (never st.stop) ----------
    n2, n3 = _autofill_dims_from_session()
    rc = st.session_state.get("run_ctx") or {}

    # Rehydrate banner if baseline exists
    if rc.get("known_signatures_patterns"):
        st.caption(f"🔵 Baseline restored: {len(rc['known_signatures_patterns'])} patterns "
                   f"(dialect n₂={int(rc.get('n2') or n2)}, n₃={int(rc.get('n3') or n3)})")

    disable_ui = False
    if n2 <= 0 or n3 <= 0:
        disable_ui = _disable_with("Fixture dims unknown. Use **Coverage Baseline** above (demo/canonical) to seed n₂=2, n₃=3.")
    known_patterns = rc.get("known_signatures_patterns") or []
    if not disable_ui and not known_patterns:
        disable_ui = _disable_with("Baseline empty. Load/paste or use canonical D2/D3 above.")
    if not disable_ui and known_patterns:
        def _dialect_ok(p: str) -> bool:
            toks = _extract_pattern_tokens(p)
            return (len(toks) == int(n3)) and all(len(t) == int(n2) for t in toks)
        if not all(_dialect_ok(p) for p in known_patterns):
            disable_ui = _disable_with("COVERAGE_BASELINE_DIALECT_MISMATCH: normalize to n₂-bit columns in loader.")

    # Lane mask provenance
    inputs_cov_tmp = _inputs_block_from_session((int(n2 or 2), int(n3 or 3)))
    lane_mask_k3 = inputs_cov_tmp.get("lane_mask_k3") or [1, 1, 0]
    lane_mask_note = "" if inputs_cov_tmp.get("lane_mask_k3") else "fallback-default"

    # Snapshot cache key
    key = _cov_key(n2 or 2, n3 or 3, random_seed, bit_density, sample_count, policy)
    snap = st.session_state["_cov_snapshots"].get(key)

    # ---------- Sample (compute & persist snapshot; buttons disabled if guard is on) ----------
    if st.button("Coverage Sample", key="btn_cov_sample", disabled=disable_ui, on_click=_cov_keep_open_sampling):
        if not normalizer_ok:
            st.error("COVERAGE_NORMALIZER_FAILED")
        else:
            try:
                import random as _random
                rng = _random.Random(int(random_seed))

                counts = {}       # full signature -> count
                patt_counts = {}  # normalized pattern=[..] -> count
                patt_meta = {}    # pattern -> (rk, ker)

                for _ in range(int(sample_count)):
                    d_k1 = _rand_gf2_matrix(int(n2), int(n3), float(bit_density), rng)
                    sig  = _coverage_signature(d_k1, n_k=int(n3))
                    counts[sig] = counts.get(sig, 0) + 1
                    _, patt_only = _normalize_signature_line_to_n2(sig, n2=int(n2), n3=int(n3))
                    patt_counts[patt_only] = patt_counts.get(patt_only, 0) + 1
                    rk = ker = ""
                    try:
                        for part in sig.split(";"):
                            P = part.strip()
                            if P.startswith("rk="):  rk = P[3:]
                            elif P.startswith("ker="): ker = P[4:]
                    except Exception:
                        pass
                    patt_meta.setdefault(patt_only, (rk, ker))

                st.session_state["_cov_snapshots"][key] = {
                    "written_at_utc": _now_utc_iso_Z(),
                    "n2": int(n2), "n3": int(n3),
                    "random_seed": int(random_seed), "bit_density": float(bit_density),
                    "sample_count": int(sample_count), "policy": str(policy),
                    "lane_mask_k3": list(lane_mask_k3), "lane_mask_note": lane_mask_note,
                    "counts": counts, "patt_counts": patt_counts, "patt_meta": patt_meta,
                }
                snap = st.session_state["_cov_snapshots"][key]
                st.success("Sampling snapshot saved. Manage unmatched below and export when ready.")
            except Exception as e:
                st.error(f"COVERAGE_SAMPLER_ERROR: {e}")

    # ---------- Render from snapshot (non-blocking; if absent, just hint) ----------
    if not snap:
        st.info("No snapshot yet. Click **Coverage Sample** to generate one.")
    else:
        # Build membership set once per render from current baseline (no resample required)
        known_set = set((st.session_state.get("run_ctx") or {}).get("known_signatures_patterns") or [])
        counts      = snap["counts"]
        patt_counts = snap["patt_counts"]
        patt_meta   = snap["patt_meta"]
        total_samples = int(snap["sample_count"])

        # Unique rows (normalize per sample before membership compare)
        unique_rows = []
        matched_samples = 0
        for sig, cnt in counts.items():
            _, patt_norm = _normalize_signature_line_to_n2(sig, n2=int(snap["n2"]), n3=int(snap["n3"]))
            in_district  = (patt_norm in known_set)
            unique_rows.append({"signature": sig, "count": int(cnt), "in_district": bool(in_district)})
            if in_district:
                matched_samples += int(cnt)

        total_unique  = int(len(patt_counts))
        matched_unique = sum(1 for p in patt_counts.keys() if p in known_set)
        pct_unique   = (100.0 * matched_unique / total_unique) if total_unique else 0.0
        pct_weighted = (100.0 * matched_samples / total_samples) if total_samples else 0.0

        # Banner
        if pct_weighted >= 95.0:
            st.success(f"Coverage ≥95% (weighted): {pct_weighted:.1f}% • Unique: {pct_unique:.1f}%")
        else:
            st.warning(f"Coverage <95% (weighted): {pct_weighted:.1f}% • Unique: {pct_unique:.1f}%")

        # Table
        df = _pd.DataFrame(unique_rows).sort_values("count", ascending=False).reset_index(drop=True)
        st.caption("Unique signatures (descending by count)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ---------- Top unmatched (from snapshot) ----------
        st.subheader("Top unmatched patterns")
        unmatched = [p for p,_c in patt_counts.items() if p not in known_set]
        topN = sorted(((p, patt_counts[p]) for p in unmatched), key=lambda x: x[1], reverse=True)[:10]

        if not topN:
            st.info("All sampled signatures are in the baseline. Nothing to add.")
        else:
            if st.button("➕ Add all shown (top unmatched)", key="cov_add_all", disabled=disable_ui, on_click=_cov_keep_open_sampling):
                added = []
                for patt, _cnt in topN:
                    rk, ker = patt_meta.get(patt, ("", ""))
                    _add_normalized_pattern_to_baseline(patt, rk=str(rk) if rk else None, ker=str(ker) if ker else None)
                    added.append(patt)
                # Audit
                rc_hist = st.session_state.setdefault("run_ctx", {})
                hist = list(rc_hist.get("baseline_history") or [])
                hist.append({"when_utc": snap["written_at_utc"], "method": "manual-add-all", "added_patterns": added})
                rc_hist["baseline_history"] = hist
                st.caption(f"Baseline now has {len(rc_hist.get('known_signatures_patterns') or [])} patterns.")

            # Per-row add
            for i, (patt, cnt) in enumerate(topN, start=1):
                rk, ker = patt_meta.get(patt, ("", ""))
                cA, cB, cC, cD = st.columns([6, 2, 2, 2])
                cA.markdown(f"**{patt}**  \ncount: {int(cnt)}")
                cB.markdown(f"rk={rk or '—'}")
                cC.markdown(f"ker={ker or '—'}")
                if cD.button("➕ Add", key=f"add_base_{i}", disabled=disable_ui, on_click=_cov_keep_open_sampling):
                    _add_normalized_pattern_to_baseline(patt, rk=str(rk) if rk else None, ker=str(ker) if ker else None)
                    # Audit
                    rc_hist = st.session_state.setdefault("run_ctx", {})
                    hist = list(rc_hist.get("baseline_history") or [])
                    hist.append({"when_utc": snap["written_at_utc"], "method": "manual-add-one", "added_patterns": [patt]})
                    rc_hist["baseline_history"] = hist
                    st.success(f"Added {patt} to baseline.")
                    st.caption(f"Baseline now has {len(rc_hist.get('known_signatures_patterns') or [])} patterns.")

        # ---------- Auto-add controller (no resample) ----------
        st.divider()
        st.subheader("Auto-add unmatched until target coverage")
        target = st.slider("Target weighted coverage (%)", 50, 100, 95, 1, key="cov_target_pct")
        max_add = st.number_input("Max patterns to add this run", min_value=1, max_value=100, value=10, step=1, key="cov_max_add")

        def _compute_weighted_coverage(pcounts: dict, known: set) -> float:
            matched = sum(int(c) for p,c in pcounts.items() if p in known)
            total   = sum(int(c) for c in pcounts.values()) or 1
            return 100.0 * matched / total

        if st.button("Auto-add top unmatched", key="cov_auto_add", disabled=disable_ui, on_click=_cov_keep_open_sampling):
            rc_live = st.session_state.get("run_ctx") or {}
            known_set_live = set(rc_live.get("known_signatures_patterns") or [])
            items = sorted(((p, c) for p, c in patt_counts.items() if p not in known_set_live),
                           key=lambda x: x[1], reverse=True)
            added = []
            for patt, _cnt in items:
                cov_now = _compute_weighted_coverage(patt_counts, known_set_live)
                if cov_now >= float(target) or len(added) >= int(max_add):
                    break
                _add_normalized_pattern_to_baseline(patt)  # rk/ker optional
                rc_live = st.session_state.get("run_ctx") or {}
                known_set_live = set(rc_live.get("known_signatures_patterns") or [])
                added.append(patt)

            rc_hist = st.session_state.setdefault("run_ctx", {})
            hist = list(rc_hist.get("baseline_history") or [])
            result_cov = _compute_weighted_coverage(patt_counts, set(rc_hist.get("known_signatures_patterns") or []))
            hist.append({
                "when_utc": snap["written_at_utc"],
                "method": "auto-add",
                "target_pct": float(target),
                "max_add": int(max_add),
                "added_patterns": added,
                "result_pct": round(float(result_cov), 3),
            })
            rc_hist["baseline_history"] = hist

            st.success(f"Auto-added {len(added)} pattern(s). Current weighted coverage (est.) ≈ {result_cov:.1f}%")
            st.caption(f"Baseline now has {len(rc_hist.get('known_signatures_patterns') or [])} patterns.")

        # ---------- Re-run (new sample) & Export baseline/artifacts ----------
        col_run, col_exp = st.columns([1, 1])
        if col_run.button("Re-run coverage with updated baseline", key="cov_rerun", disabled=disable_ui, on_click=_cov_keep_open_sampling):
            st.session_state["trigger_coverage_run"] = True
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

        if col_exp.button("Export current baseline", key="cov_export", disabled=False, on_click=_cov_keep_open_sampling):
            rcx  = st.session_state.get("run_ctx") or {}
            base = list(rcx.get("known_signatures") or [])
            patt = list(rcx.get("known_signatures_patterns") or [])
            mem_json = _StringIO(); mem_json.write(_json.dumps(base, ensure_ascii=False, indent=2))
            st.download_button("Download baseline_signatures.json", data=mem_json.getvalue().encode("utf-8"),
                               file_name="baseline_signatures.json", mime="application/json")
            mem_txt = _StringIO(); mem_txt.write("\n".join(patt))
            st.download_button("Download baseline_patterns.txt", data=mem_txt.getvalue().encode("utf-8"),
                               file_name="baseline_patterns.txt", mime="text/plain")

        # ---------- Artifact JSON + CSV (snapshot + refreshed baseline) ----------
        rc_payload = st.session_state.get("run_ctx") or {}
        ks_full  = list(rc_payload.get("known_signatures") or [])
        ks_patt  = list(rc_payload.get("known_signatures_patterns") or [])
        if not ks_full or not ks_patt:
            st.warning("COVERAGE_CONFIG_EMPTY: baseline cleared during session. Reload baseline to export artifacts.")
        else:
            # Final dialect check before export
            def _ok_pattern(p: str) -> bool:
                toks = _extract_pattern_tokens(p)
                return (len(toks) == int(snap["n3"])) and all(len(t) == int(snap["n2"]) for t in toks)
            if not all(_ok_pattern(p) for p in ks_patt):
                st.warning("COVERAGE_BASELINE_DIALECT_MISMATCH: normalize to n₂-bit columns in loader before exporting.")
            else:
                shapes_bytes = (st.session_state.get("shapes_raw_bytes") or b"")
                shapes_hash  = _sha256_bytes(shapes_bytes) if shapes_bytes else (rc_payload.get("shapes_hash") or "")
                U_hash = ""  # coverage has no carrier

                artifact = {
                    "schema_version": "1.0.0",
                    "written_at_utc": snap["written_at_utc"],
                    "app_version": st.session_state.get("app_version") or "",
                    "params": {
                        "n2": int(snap["n2"]),
                        "n3": int(snap["n3"]),
                        "lane_mask_k3": list(snap["lane_mask_k3"]),
                        "lane_mask_note": snap["lane_mask_note"],
                        "policy": snap["policy"],
                        "random_seed": int(snap["random_seed"]),
                        "bit_density": float(snap["bit_density"]),
                        "sample_count": int(snap["sample_count"]),
                    },
                    "baseline": {
                        "count": int(len(ks_full)),
                        "known_signatures": ks_full,
                        "known_signatures_patterns": ks_patt,
                    },
                    "samples": sorted(
                        [{"signature": r["signature"], "count": int(r["count"]), "in_district": bool(r["in_district"])}
                         for r in unique_rows],
                        key=lambda x: x["count"], reverse=True
                    ),
                    "summary": {
                        "unique_total": int(total_unique),
                        "unique_matched": int(matched_unique),
                        "pct_in_district_unique": round(float(pct_unique), 3),
                        "samples_total": int(total_samples),
                        "samples_matched": int(matched_samples),
                        "pct_in_district_weighted": round(float(pct_weighted), 3),
                    },
                    "hashes": {"shapes_hash": str(shapes_hash), "U_hash": str(U_hash)},
                }
                content_hash = _sha256_json(artifact)
                artifact["hashes"]["content_hash"] = content_hash

                # deterministic filenames
                district = (st.session_state.get("run_ctx") or {}).get("district_id")
                tag = ( (district + "__") if district else "" ) + ((st.session_state.get("run_ctx") or {}).get("run_id") or content_hash[:8])

                json_str = _json.dumps(artifact, indent=2, sort_keys=False)
                st.download_button("Download coverage_report.json",
                                   data=json_str.encode("utf-8"),
                                   file_name=f"coverage_report__{tag}.json",
                                   mime="application/json")

                # CSV (unique signatures) with rk/ker
                def _rk_ker(sig: str):
                    rk = ker = ""
                    try:
                        for part in sig.split(";"):
                            P = part.strip()
                            if P.startswith("rk="): rk = P[3:]
                            elif P.startswith("ker="): ker = P[4:]
                    except Exception:
                        pass
                    return rk, ker

                df_csv = _pd.DataFrame({
                    "signature":   [r["signature"] for r in unique_rows],
                    "count":       [int(r["count"]) for r in unique_rows],
                    "in_district": [bool(r["in_district"]) for r in unique_rows],
                    "pct":         [ (float(r["count"])/float(total_samples)) if total_samples else 0.0 for r in unique_rows ],
                    "rk":          [_rk_ker(r["signature"])[0] for r in unique_rows],
                    "ker":         [_rk_ker(r["signature"])[1] for r in unique_rows],
                }).sort_values("count", ascending=False)

                csv_buf = _StringIO(); df_csv.to_csv(csv_buf, index=False)
                st.download_button("Download coverage_summary.csv",
                                   data=csv_buf.getvalue().encode("utf-8"),
                                   file_name=f"coverage_summary__{tag}.csv",
                                   mime="text/csv")
# ===================== /Coverage Sampling =====================























# ----------------- IMPORTS -----------------
import io as _io
import csv
import uuid
import os
import tempfile
import hashlib
from datetime import datetime, timezone
import re
import json as _json
from pathlib import Path

# -------------- FIXTURE STASH --------------
FIXTURE_STASH_DIR = Path(globals().get("FIXTURE_STASH_DIR", "inputs/fixtures"))
FIXTURE_STASH_DIR.mkdir(parents=True, exist_ok=True)

# --------- Define eye() if missing (GF(2) shim already set above) --------
if "eye" not in globals():
    def eye(n: int):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]




# --- Hashing helpers (reuse canonical) -----------------------------------
def _to_hashable_plain(x):
    """Return a stable, JSON-serializable view of x for hashing."""
    if x is None:
        return None
    # Preferred: objects with .blocks.__root__ (your Boundaries/Cmap/H)
    try:
        return x.blocks.__root__
    except Exception:
        pass
    # Pydantic v2 models
    try:
        return x.model_dump()  # type: ignore[attr-defined]
    except Exception:
        pass
    # Dataclasses / simple objects
    try:
        return dict(x.__dict__)
    except Exception:
        pass
    # Already-plain types
    if isinstance(x, (dict, list, tuple, str, int, float, bool)):
        return x
    # Last resort: repr (deterministic enough)
    return repr(x)

def _hash_fixture_side(fx: dict) -> dict:
    """Hash each sub-object of a fixture side robustly (uses canonical hash_json)."""
    return {
        "boundaries": hash_json(_to_hashable_plain(fx.get("boundaries"))),
        "shapes":     hash_json(_to_hashable_plain(fx.get("shapes"))),
        "cmap":       hash_json(_to_hashable_plain(fx.get("cmap"))),
        "H":          hash_json(_to_hashable_plain(fx.get("H"))),
    }

# --- GF(2) helpers --------------------------------------------------------
def _all_zero_mat(M: list[list[int]]) -> bool:
    return not M or all((x & 1) == 0 for row in M for x in row)

def _I(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def _lane_mask_from_boundaries(boundaries_obj) -> list[int]:
    """Mask from d3 support; schema tolerant (prefers '3', falls back to '3->2')."""
    try:
        root = (boundaries_obj or {}).blocks.__root__
    except Exception:
        root = {}
    d3 = root.get("3") or root.get("3->2") or []
    return _truth_mask_from_d3(d3)

def _lane_mask_pair_SSO(L_boundaries, R_boundaries) -> list[int]:
    """Pair mask: OR of left/right masks (robust if they differ)."""
    Lm = _lane_mask_from_boundaries(L_boundaries)
    Rm = _lane_mask_from_boundaries(R_boundaries)
    if not Lm and not Rm:
        return []
    if not Lm: return Rm[:]
    if not Rm: return Lm[:]
    n = max(len(Lm), len(Rm))
    Lm = Lm + [0]*(n-len(Lm))
    Rm = Rm + [0]*(n-len(Rm))
    return [1 if (Lm[j] or Rm[j]) else 0 for j in range(n)]

def _diag_from_mask(mask: list[int]) -> list[list[int]]:
    n = len(mask or [])
    return [[(mask[i] & 1) if i == j else 0 for j in range(n)] for i in range(n)]

def _r3_from_fixture(fx: dict) -> list[list[int]]:
    """
    R3 = H2 @ d3  ⊕  (C3 ⊕ I3)  over GF(2).
    Shapes: H2 (n3×n2), d3 (n2×n3), C3/I3 (n3×n3).
    """
    B = (fx.get("boundaries") or {}).blocks.__root__
    C = (fx.get("cmap") or {}).blocks.__root__
    H = (fx.get("H") or {}).blocks.__root__
    d3 = B.get("3") or []          # (n2 × n3)
    H2 = H.get("2") or []          # (n3 × n2)
    C3 = C.get("3") or []          # (n3 × n3)
    I3 = _I(len(C3)) if C3 else []
    term1 = mul(H2, d3)            # (n3 × n3)
    term2 = _xor_mat(C3, I3)       # (n3 × n3)
    return _xor_mat(term1, term2)

def _classify_residual(R3: list[list[int]], mask: list[int]) -> str:
    if _all_zero_mat(R3):
        return "none"
    # lanes = columns with mask==1, ker = columns with mask==0
    n3 = len(R3[0]) if R3 else 0
    def col_has_support(j: int) -> bool:
        return any(R3[i][j] & 1 for i in range(len(R3)))
    lanes_support = any(col_has_support(j) for j in range(n3) if j < len(mask) and mask[j])
    ker_support   = any(col_has_support(j) for j in range(n3) if j >= len(mask) or not mask[j])
    if lanes_support and ker_support: return "mixed"
    if lanes_support:                 return "lanes"
    if ker_support:                   return "ker"
    return "none"  # degenerate fallback

# -------------- Core: generic JSON helpers (atomic write / parse) --------------
def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _guarded_atomic_write_json(path: Path, payload: dict) -> None:
    _ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        os.replace(tmp.name, path)

def _safe_parse_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No file at {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

# -------- Report paths (define once, with compat aliases) --------
if "REPORTS_DIR" not in globals():
    REPORTS_DIR = REPORTS_DIR
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical names used going forward:
if "PARITY_REPORT_PATH" not in globals():
    PARITY_REPORT_PATH = REPORTS_DIR / "parity_report.json"
if "PARITY_SUMMARY_CSV" not in globals():
    PARITY_SUMMARY_CSV = REPORTS_DIR / "parity_summary.csv"

# Back-compat aliases (if older code references these):
PARITY_JSON_PATH = globals().get("PARITY_JSON_PATH", PARITY_REPORT_PATH)
PARITY_CSV_PATH  = globals().get("PARITY_CSV_PATH",  PARITY_SUMMARY_CSV)

LOGS_DIR = Path(globals().get("LOGS_DIR", "logs"))


# --------- Universal adapter: capture old _paths_from_fixture_or_current ----------
# Capture the existing function once, outside the new definition
_OLD_PFFC = globals().get("_paths_from_fixture_or_current", None)

def _paths_from_fixture_or_current(*args):
    # Use the captured old function to avoid recursion issues
    if "residual_tag" not in globals():
        def residual_tag(matrix, mask):
            return "unknown"

    # Extract fx from args
    if len(args) == 1 and isinstance(args[0], dict):
        fx = args[0]
    elif len(args) >= 2 and isinstance(args[1], dict):
        fx = args[1]
    elif _OLD_PFFC:
        return _OLD_PFFC(*args)
    else:
        raise TypeError("_paths_from_fixture_or_current(): expected (fx) or (side_name, fx)")

    # Fill paths, prefer explicit *_path, else keys, fallback SSOT filenames
    out = {}
    for k in ("boundaries", "cmap", "H", "shapes"):
        v = fx.get(f"{k}_path")
        if not v and isinstance(fx.get(k), str):
            v = fx.get(k)
        out[k] = v or ""

    # Fill missing with SSOT filenames
    ib = st.session_state.get("_inputs_block") or {}
    fns = ib.get("filenames") or {}
    default_paths = {
        "boundaries": fns.get("boundaries", "inputs/boundaries.json"),
        "cmap": fns.get("C", "inputs/cmap.json"),
        "H": fns.get("H", "inputs/H.json"),
        "shapes": fns.get("U", "inputs/shapes.json"),
    }
    for k in ("boundaries", "cmap", "H", "shapes"):
        out.setdefault(k, default_paths[k])

    return out

# ---- Parity types & state ----
from typing import TypedDict, Optional, Literal

class Side(TypedDict, total=False):
    embedded: dict
    boundaries: str
    shapes: str
    cmap: str
    H: str

class PairSpec(TypedDict):
    label: str
    left: Side
    right: Side

class SkippedSpec(TypedDict, total=False):
    label: str
    side: Literal["left","right"]
    missing: list[str]
    error: str

class RunCtx(TypedDict, total=False):
    policy_hint: Literal["strict","projected:auto","projected:file","mirror_active"]
    projector_mode: Literal["strict","auto","file"]
    projector_filename: Optional[str]
    projector_hash: str
    lane_mask_note: Optional[str]

def _ensure_state():
    st.session_state.setdefault("parity_pairs_table", [])   # editable table rows (PairSpec-like)
    st.session_state.setdefault("parity_pairs_queue", [])   # validated runnable pairs
    st.session_state.setdefault("parity_skipped_specs", []) # skipped reasons (SkippedSpec)
    st.session_state.setdefault("parity_run_ctx", {})       # RunCtx
    st.session_state.setdefault("parity_last_report_pairs", None)  # results cache


def resolve_side(side: Side) -> dict:
    if "embedded" in side:
        return parse_embedded(side["embedded"])  # your existing io.parse_* over objects
    missing = [k for k in ("boundaries","shapes","cmap","H") if not side.get(k)]
    if missing:
        raise KeyError(f"PARITY_SPEC_MISSING:{missing}")
    for k in ("boundaries","shapes","cmap","H"):
        p = Path(side[k])
        if not (p.exists() and p.is_file()):
            raise FileNotFoundError(f"PARITY_FILE_NOT_FOUND:{k}={side[k]}")
    return parse_from_paths(
        boundaries_path=side["boundaries"], shapes_path=side["shapes"],
        cmap_path=side["cmap"], H_path=side["H"]
    )

def lane_mask_k3_from_boundaries(boundaries) -> tuple[list[int], str]:
    d3 = (boundaries.blocks.__root__.get("3->2") 
          or boundaries.blocks.__root__.get("3")  # if your format uses "3" for 3→2
          or [])
    if not d3: return [], ""
    cols = len(d3[0])
    mask = [1 if any(row[j] & 1 for row in d3) else 0 for j in range(cols)]
    return mask, "".join("1" if b else "0" for b in mask)

def run_pair(spec: PairSpec, run_ctx: RunCtx) -> dict:
    label = spec["label"]
    try:
        L = resolve_side(spec["left"])
    except Exception as e:
        raise RuntimeError(f"SKIP:left:{e}")
    try:
        R = resolve_side(spec["right"])
    except Exception as e:
        raise RuntimeError(f"SKIP:right:{e}")

    # lane mask from THIS pair’s boundaries
    mask_vec, mask_str = lane_mask_k3_from_boundaries(L["boundaries"])  # boundaries object
    # (optionally assert R yields same mask; not required)

    # strict leg
    sL = run_leg_strict(L)  # returns {"k2":bool,"k3":bool,"residual":"none|lanes|ker|mixed|error"}
    sR = run_leg_strict(R)
    strict = {"k2": bool(sL["k2"] and sR["k2"]),
              "k3": bool(sL["k3"] and sR["k3"]),
              "residual_tag": combine_residuals(sL["residual"], sR["residual"])}

    # projected leg?
    projected = None
    if run_ctx["projector_mode"] == "auto":
        Pi = projector_from_lane_mask(mask_vec)  # diag(mask)
        pL = run_leg_projected(L, Pi)
        pR = run_leg_projected(R, Pi)
        projected = {"k2": bool(pL["k2"] and pR["k2"]),
                     "k3": bool(pL["k3"] and pR["k3"]),
                     "residual_tag": combine_residuals(pL["residual"], pR["residual"])}
    elif run_ctx["projector_mode"] == "file":
        Pi = get_frozen_projector()  # set once during compute_run_ctx / validation
        # Optionally enforce lane compatibility and hard-error:
        if not diag_matches_mask_if_required(Pi, mask_vec):
            raise RuntimeError(f"P3_LANE_MISMATCH: diag(P) != lane_mask(d3) : {diag_to_str(Pi)} vs {mask_str}")
        pL = run_leg_projected(L, Pi)
        pR = run_leg_projected(R, Pi)
        projected = {"k2": bool(pL["k2"] and pR["k2"]),
                     "k3": bool(pL["k3"] and pR["k3"]),
                     "residual_tag": combine_residuals(pL["residual"], pR["residual"])}

    # hashes
    Lh = hashes_for_fixture(L)  # {"boundaries":"…","shapes":"…","cmap":"…","H":"…"}
    Rh = hashes_for_fixture(R)
    pair_hash = sha256_tuple(Lh, Rh)

    return {
        "label": label,
        "pair_hash": pair_hash,
        "lane_mask_k3": mask_vec,
        "lane_mask": mask_str,
        "left_hashes": Lh,
        "right_hashes": Rh,
        "strict": strict,
        **({"projected": projected} if projected is not None else {}),
    }

def queue_all_valid_from_table():
    _ensure_state()
    st.session_state["parity_pairs_queue"].clear()
    st.session_state["parity_skipped_specs"].clear()

    rows = st.session_state["parity_pairs_table"]
    for row in rows:
        spec: PairSpec = {"label": row["label"], "left": row["left"], "right": row["right"]}
        # lightweight check: embedded or all 4 fields for each side
        misses = []
        for side_name in ("left","right"):
            s = spec[side_name]
            if "embedded" in s: 
                continue
            miss = [k for k in ("boundaries","shapes","cmap","H") if not s.get(k)]
            if miss:
                st.session_state["parity_skipped_specs"].append({"label": spec["label"], "side": side_name, "missing": miss})
                break
        else:
            st.session_state["parity_pairs_queue"].append(spec)


def run_parity_suite():
    _ensure_state()
    queue = st.session_state["parity_pairs_queue"]
    skipped = st.session_state["parity_skipped_specs"]
    if not queue:
        st.info("Queued pairs: 0 — nothing to run.")
        return

    # Freeze decision
    rc_in = st.session_state.get("parity_run_ctx") or {}
    try:
        rc = compute_run_ctx(rc_in.get("policy_hint","mirror_active"), rc_in.get("projector_filename"))
    except Exception as e:
        st.error(str(e))  # e.g., PARITY_SCHEMA_INVALID or projector invalid
        return
    st.session_state["parity_run_ctx"] = rc  # freeze for this run

    # Optional: if file mode, set global frozen projector (Pi) once here.

    results = []
    proj_green = 0
    rows_total = len(queue)
    rows_skipped = len(skipped)
    rows_run = 0

    for spec in queue:
        try:
            out = run_pair(spec, rc)
            results.append(out)
            rows_run += 1
            if rc["projector_mode"] != "strict" and out.get("projected",{}).get("k3") is True:
                proj_green += 1
        except RuntimeError as e:
            # convert into skipped entry and continue
            parts = str(e).split(":", 2)
            if parts and parts[0].startswith("SKIP"):
                side = parts[1] if len(parts) > 1 else ""
                skipped.append({"label": spec["label"], "side": side, "error": parts[-1]})
            else:
                st.error(str(e))
                return  # hard abort if it’s a non-skip fatal

    # Write artifacts (next step)
    write_parity_artifacts(results, rc, skipped, rows_total, rows_skipped, rows_run, proj_green)
    st.session_state["parity_last_report_pairs"] = results
def run_parity_suite():
    _ensure_state()
    queue = st.session_state["parity_pairs_queue"]
    skipped = st.session_state["parity_skipped_specs"]
    if not queue:
        st.info("Queued pairs: 0 — nothing to run.")
        return

    # Freeze decision
    rc_in = st.session_state.get("parity_run_ctx") or {}
    try:
        rc = compute_run_ctx(rc_in.get("policy_hint","mirror_active"), rc_in.get("projector_filename"))
    except Exception as e:
        st.error(str(e))  # e.g., PARITY_SCHEMA_INVALID or projector invalid
        return
    st.session_state["parity_run_ctx"] = rc  # freeze for this run

    # Optional: if file mode, set global frozen projector (Pi) once here.

    results = []
    proj_green = 0
    rows_total = len(queue)
    rows_skipped = len(skipped)
    rows_run = 0

    for spec in queue:
        try:
            out = run_pair(spec, rc)
            results.append(out)
            rows_run += 1
            if rc["projector_mode"] != "strict" and out.get("projected",{}).get("k3") is True:
                proj_green += 1
        except RuntimeError as e:
            # convert into skipped entry and continue
            parts = str(e).split(":", 2)
            if parts and parts[0].startswith("SKIP"):
                side = parts[1] if len(parts) > 1 else ""
                skipped.append({"label": spec["label"], "side": side, "error": parts[-1]})
            else:
                st.error(str(e))
                return  # hard abort if it’s a non-skip fatal

    # Write artifacts (next step)
    write_parity_artifacts(results, rc, skipped, rows_total, rows_skipped, rows_run, proj_green)
    st.session_state["parity_last_report_pairs"] = results

def _pp_try_load(side: dict):
    """Load one fixture side either from embedded or from paths."""
    if "embedded" in side:
        emb = side["embedded"]
        return {
            "boundaries": io.parse_boundaries(emb["boundaries"]),
            "shapes":     io.parse_shapes(emb["shapes"]),
            "cmap":       io.parse_cmap(emb["cmap"]),
            "H":          io.parse_cmap(emb["H"]),
        }
    # path mode
    return load_fixture_from_paths(
        boundaries_path=side["boundaries"],
        cmap_path=side["cmap"],
        H_path=side["H"],
        shapes_path=side["shapes"],
    )



# --------- Resolver + Fixture Loader -------------------------------------------
def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", (s or "").lower())

def _resolve_fixture_path(p: str) -> str:
    """
    Resolve fixture filenames like 'boundaries D2.json' under various roots.
    """
    p = (p or "").strip()
    if not p:
        raise FileNotFoundError("Empty fixture path")
    roots = [Path("."), Path("inputs"), Path("/mnt/data"), FIXTURE_STASH_DIR]  # include stash dir
    tried = []

    # 1) as-is
    cand = Path(p)
    tried.append(cand.as_posix())
    if cand.exists() and cand.is_file():
        return cand.as_posix()

    # 2) under roots
    for r in roots:
        cand = r / p
        tried.append(cand.as_posix())
        if cand.exists() and cand.is_file():
            return cand.as_posix()

    # 3) recursive fuzzy search
    target = _canon(Path(p).name)
    for r in roots:
        if not r.exists():
            continue
        for q in r.rglob("*"):
            if q.is_file():
                if _canon(q.name) == target or _canon(q.stem) == _canon(Path(p).stem):
                    return q.as_posix()

    raise FileNotFoundError(
        f"Fixture file not found for '{p}'. Tried: " + " | ".join(tried[:10]) +
        " (also searched recursively under ./, inputs/, /mnt/data with fuzzy match)"
    )

def load_fixture_from_paths(*, boundaries_path, cmap_path, H_path, shapes_path):
    """
    Resolve paths then parse fixtures via io.parse_*.
    """
    def _read_json(p):
        with open(p, "r", encoding="utf-8") as f:
            return _json.load(f)

    b_path = _resolve_fixture_path(boundaries_path)
    c_path = _resolve_fixture_path(cmap_path)
    h_path = _resolve_fixture_path(H_path)
    u_path = _resolve_fixture_path(shapes_path)

    return {
        "boundaries": io.parse_boundaries(_read_json(b_path)),
        "cmap": io.parse_cmap(_read_json(c_path)),
        "H": io.parse_cmap(_read_json(h_path)),
        "shapes": io.parse_shapes(_read_json(u_path)),
    }


# ========================== Parity import/export helpers (canonical) ==========================
# Place this block ABOVE the "Parity pairs: import/export" expander. Keep only one copy.

import os
import io as _io
import json as _json
from pathlib import Path

# --- tiny clock shim if not already present
if "__pp_now_z" not in globals():
    from datetime import datetime, timezone
    def __pp_now_z():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# --- safe defaults for paths/dirs
if "LOGS_DIR" not in globals():
    LOGS_DIR = Path("logs")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
if "DEFAULT_PARITY_PATH" not in globals():
    DEFAULT_PARITY_PATH = LOGS_DIR / "parity_pairs.json"

def _file_mode_invalid_now() -> bool:
    """
    Disable export/import only when FILE Π is selected AND current Π is invalid.
    Otherwise allow (strict/auto always allowed).
    """
    rc = st.session_state.get("run_ctx") or {}
    mode_now = str(rc.get("mode",""))
    if mode_now == "projected(columns@k=3,file)":
        return not bool(st.session_state.get("file_pi_valid", False))
    return False

# ---- queue → portable JSON payload
def __pp_pairs_payload_from_queue(pairs: list[dict]) -> dict:
    def _paths_anyshape(fx: dict) -> dict:
        # Prefer your universal adapter if present; else fall back to SSOT filenames
        try:
            return _paths_from_fixture_or_current(fx)  # optional: provided elsewhere
        except Exception:
            ib = st.session_state.get("_inputs_block") or {}
            fns = (ib.get("filenames") or {})
            # accept both canonical and legacy keys
            return {
                "boundaries": fx.get("boundaries", fns.get("boundaries", "inputs/boundaries.json")),
                "cmap":       fx.get("cmap",       fns.get("cmap",       fns.get("C", "inputs/cmap.json"))),
                "H":          fx.get("H",          fns.get("H",          "inputs/H.json")),
                "shapes":     fx.get("shapes",     fns.get("shapes",     fns.get("U", "inputs/shapes.json"))),
            }
    rows = []
    for row in (pairs or []):
        rows.append({
            "label": row.get("label","PAIR"),
            "left":  _paths_anyshape(row.get("left", {})  or {}),
            "right": _paths_anyshape(row.get("right", {}) or {}),
        })
    return {
        "schema_version": "1.0.0",
        "saved_at": __pp_now_z(),
        "count": len(rows),
        "pairs": rows,
    }

# ---- path normalization for text inputs
def _ensure_json_path_str(p_str: str, default_name="parity_pairs.json") -> str:
    p = Path((p_str or "").strip() or default_name)
    if p.is_dir() or str(p).endswith(("/", "\\")) or p.name == "":
        p = p / default_name
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.as_posix()

# --- Parity queue shims (safe if re-defined)
if "clear_parity_pairs" not in globals():
    def clear_parity_pairs():
        st.session_state["parity_pairs"] = []

if "add_parity_pair" not in globals():
    def add_parity_pair(*, label: str, left_fixture: dict, right_fixture: dict) -> int:
        st.session_state.setdefault("parity_pairs", [])
        st.session_state["parity_pairs"].append({
            "label": label,
            "left":  left_fixture,
            "right": right_fixture,
        })
        return len(st.session_state["parity_pairs"])

# ---- export helper used by the button
def _export_pairs_to_path(path_str: str) -> str:
    p = Path(_ensure_json_path_str(path_str))
    payload = __pp_pairs_payload_from_queue(st.session_state.get("parity_pairs", []) or [])
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    return p.as_posix()

# ---- import helper for the uploader (payload-based)
def _import_pairs_from_payload(payload: dict, *, merge: bool) -> int:
    """
    If you have validate_pairs_payload(...), call it here before ingesting.
    Otherwise we accept the payload shape produced by __pp_pairs_payload_from_queue.
    """
    # Optional hard validation (uncomment if available):
    # pairs_sanitized, policy_hint = validate_pairs_payload(payload)
    # pairs_in = pairs_sanitized
    pairs_in = payload.get("pairs") or []

    if not merge:
        clear_parity_pairs()

    for r in pairs_in:
        L = r.get("left")  or {}
        R = r.get("right") or {}
        # load_fixture_from_paths is expected to exist elsewhere; if not, replace with your loader.
        fxL = load_fixture_from_paths(
            boundaries_path=L["boundaries"], cmap_path=L["cmap"], H_path=L["H"], shapes_path=L["shapes"]
        )
        fxR = load_fixture_from_paths(
            boundaries_path=R["boundaries"], cmap_path=R["cmap"], H_path=R["H"], shapes_path=R["shapes"]
        )
        add_parity_pair(label=r.get("label","PAIR"), left_fixture=fxL, right_fixture=fxR)

    return len(st.session_state.get("parity_pairs", []))
# ======================== /Parity import/export helpers ========================



# -------------- Loader shim for parity import -----------------
if "load_fixture_from_paths" not in globals():
    def load_fixture_from_paths(*, boundaries_path, cmap_path, H_path, shapes_path):
        try:
            _ = (io.parse_boundaries, io.parse_cmap, io.parse_shapes)
        except Exception:
            raise RuntimeError(
                "Project parsing module `io` missing or shadowed. "
                "Ensure your project imports `io` with parse_* functions."
            )
        def _read(p):
            with open(Path(p), "r", encoding="utf-8") as f:
                return _json.load(f)
        dB = _read(boundaries_path)
        dC = _read(cmap_path)
        dH = _read(H_path)
        dU = _read(shapes_path)
        return {
            "boundaries": io.parse_boundaries(dB),
            "cmap": io.parse_cmap(dC),
            "H": io.parse_cmap(dH),
            "shapes": io.parse_shapes(dU),
        }
    
        if "clear_parity_pairs" not in globals():
            def clear_parity_pairs():
                st.session_state["parity_pairs"] = []
        
        if "add_parity_pair" not in globals():
            def add_parity_pair(*, label: str, left_fixture: dict, right_fixture: dict) -> int:
                st.session_state.setdefault("parity_pairs", [])
                st.session_state["parity_pairs"].append({"label": label, "left": left_fixture, "right": right_fixture})
                return len(st.session_state["parity_pairs"])
    
        if "_short_hash" not in globals():
            def _short_hash(h: str, n: int = 8) -> str:
                return (h[:n] + "…") if h else ""


# ---- Parity defaults (define once, above import_parity_pairs) ----
from pathlib import Path

if "LOGS_DIR" not in globals():
    LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

if "DEFAULT_PARITY_PATH" not in globals():
    DEFAULT_PARITY_PATH = LOGS_DIR / "parity_pairs.json"

# ----------------- import_parity_pairs (validate & stash only) -----------------
def import_parity_pairs(
    path: str | Path = DEFAULT_PARITY_PATH,
    *,
    merge: bool = False,
) -> int:
    # 1) Parse JSON
    payload = _safe_parse_json(str(path))

    # 2) Strict schema validation + sanitization
    pairs_sanitized, policy_hint = validate_pairs_payload(payload)

    # 3) Stash into session state (table source for the UI editor)
    if merge and "parity_pairs_table" in st.session_state:
        st.session_state["parity_pairs_table"].extend(pairs_sanitized)
    else:
        st.session_state["parity_pairs_table"] = pairs_sanitized

    st.session_state["parity_policy_hint"] = policy_hint

    # Optional: clear legacy queue to avoid confusion with the new table-driven flow
    st.session_state["parity_pairs"] = []

    st.success(f"Imported {len(pairs_sanitized)} pair specs")
    return len(st.session_state["parity_pairs_table"])
    # === Policy picker (source of truth for parity runs) ===
st.radio(
    "Policy",
    ["strict", "projected(columns@k=3,auto)", "projected(columns@k=3,file)"],
    key="parity_policy_choice",
    horizontal=True,
)
#---------------------------helper-------------------------------------

def _resolve_side_or_skip_exact(side: dict, *, label: str, side_name: str):
    """
    Strict resolver for a single side of a pair.

    Returns:
        ("ok", fixture_dict)  -> fully parsed fixture (boundaries/cmap/H/shapes objects)
        ("skip", info_dict)   -> standardized reason with fields:
            { "label": <str>, "side": "left"|"right",
              "missing": [<keys>]?, "error": <CODE or message> }

    Rules:
      • If 'embedded' exists, it wins. Parse it via _pp_load_embedded().
      • Otherwise require ALL four string paths: boundaries, shapes, cmap, H.
      • No fuzzy search here; paths must exist (checked via _path_exists_strict).
      • On any failure, return a 'skip' record with precise reason.
    """
    # Validate shape of the side object quickly
    if not isinstance(side, dict):
        return ("skip", {
            "label": label, "side": side_name,
            "error": "PARITY_SCHEMA_INVALID: side must be an object"
        })

    # 1) Embedded wins
    if "embedded" in side and side["embedded"] is not None:
        try:
            fx = _pp_load_embedded(side["embedded"])
            return ("ok", fx)
        except Exception as e:
            return ("skip", {
                "label": label, "side": side_name,
                "error": f"PARITY_SCHEMA_INVALID: {e}"
            })

    # 2) Require all four paths
    required = ("boundaries", "shapes", "cmap", "H")
    missing_keys = [k for k in required if not (isinstance(side.get(k), str) and side[k].strip())]
    if missing_keys:
        return ("skip", {
            "label": label, "side": side_name,
            "missing": missing_keys, "error": "PARITY_SPEC_MISSING"
        })

    # 3) Paths must exist (no recursive/fuzzy)
    not_found = [k for k in required if not _path_exists_strict(side[k])]
    if not_found:
        return ("skip", {
            "label": label, "side": side_name,
            "missing": not_found, "error": "PARITY_FILE_NOT_FOUND"
        })

    # 4) Load fixture from paths
    try:
        fx = _pp_try_load(side)
        return ("ok", fx)
    except Exception as e:
        return ("skip", {
            "label": label, "side": side_name,
            "error": f"PARITY_FILE_NOT_FOUND: {e}"
        })



# --- Policy resolver used by the Parity · Run Suite block
def _policy_from_hint():
    """
    Resolve parity policy with this precedence:
    1) UI radio: st.session_state["parity_policy_choice"]
    2) Imported hint: st.session_state["parity_policy_hint"] in
       {"strict","projected:auto","projected:file","mirror_active"}
    3) Mirror app run_ctx.mode ("strict" / "projected(columns@k=3,auto)" / "projected(columns@k=3,file)")
    Returns: (mode, submode) where mode in {"strict","projected"} and submode in {"","auto","file"}.
    """
    # 1) UI radio (if you have one)
    choice = st.session_state.get("parity_policy_choice")
    if choice == "strict":
        return ("strict", "")
    if choice == "projected(columns@k=3,auto)":
        return ("projected", "auto")
    if choice == "projected(columns@k=3,file)":
        return ("projected", "file")

    # 2) Imported hint (from the pairs payload)
    hint = (st.session_state.get("parity_policy_hint") or "mirror_active").strip()
    if hint == "strict":
        return ("strict", "")
    if hint == "projected:auto":
        return ("projected", "auto")
    if hint == "projected:file":
        return ("projected", "file")

    # 3) Mirror the app's current policy
    rc = st.session_state.get("run_ctx") or {}
    mode = rc.get("mode", "strict")
    if mode == "strict":
        return ("strict","")
    if mode == "projected(columns@k=3,auto)":
        return ("projected","auto")
    if mode == "projected(columns@k=3,file)":
        return ("projected","file")

    return ("strict","")  # safe default


# --- Filesystem guard (strict, no fuzzy)
def _path_exists_strict(p: str) -> bool:
    try:
        P = Path(p)
        return P.exists() and P.is_file()
    except Exception:
        return False

# --- Embedded loader (inline preset → fixture objects)
def _pp_load_embedded(emb: dict) -> dict:
    """
    Parse embedded JSON blobs using your io.parse_* API.
    Accepts shapes in either form:
      {"n": {"2": 2, "3": 3}}  or  {"n2": 2, "n3": 3}
    We pass through to io.parse_shapes which already handles your schema.
    """
    if not isinstance(emb, dict):
        raise ValueError("embedded must be an object")
    return {
        "boundaries": io.parse_boundaries(emb["boundaries"]),
        "shapes":     io.parse_shapes(emb["shapes"]),
        "cmap":       io.parse_cmap(emb["cmap"]),
        "H":          io.parse_cmap(emb["H"]),
    }

# --- Path-bundle loader (path mode → fixture objects)
def _pp_try_load(side: dict):
    """
    Load a fixture side from exact paths (no fuzzy). 
    Expects keys: boundaries, cmap, H, shapes.
    """
    return load_fixture_from_paths(
        boundaries_path=side["boundaries"],
        cmap_path=side["cmap"],
        H_path=side["H"],
        shapes_path=side["shapes"],
    )

# --- Single-leg runner via your canonical overlap gate
def _pp_one_leg(boundaries_obj, cmap_obj, H_obj, projection_cfg: dict | None):
    """
    Runs one leg (strict if projection_cfg is None, otherwise projected)
    through your overlap_gate.overlap_check() and returns its result dict.

    Returns shape (typical):
      {"2": {"eq": bool}, "3": {"eq": bool}, ...}
    On error, returns a safe stub with an _err message.
    """
    try:
        if projection_cfg is None:
            return overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_obj)  # type: ignore
        return overlap_gate.overlap_check(
            boundaries_obj, cmap_obj, H_obj, projection_config=projection_cfg  # type: ignore
        )
    except Exception as e:
        return {"2": {"eq": False}, "3": {"eq": False}, "_err": str(e)}


# --- tiny truth helper
if "_bool_and" not in globals():
    def _bool_and(a, b):
        """Logical AND that tolerates None."""
        return (a is not None) and (b is not None) and bool(a) and bool(b)

if "_emoji" not in globals():
    def _emoji(v):
        if v is None: return "—"
        return "✅" if bool(v) else "❌"


# --- Tiny helper: format a short hash for UI pills
def _short_hash(h: str | None) -> str:
    try:
        h = (h or "").strip()
    except Exception:
        h = ""
    return (h[:8] + "…") if h else "—"

def _safe_tag(s: str) -> str:
    # Keep [A-Za-z0-9_.-], replace the rest with '_'
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)



# --- Read a diagonal from a projector file (accepts {diag:[...]} or a 3x3 block)
def _projector_diag_from_file(path: str) -> list[int]:
    try:
        with open(Path(path), "r", encoding="utf-8") as f:
            pj = _json.load(f)
    except Exception as e:
        raise RuntimeError(f"P3_FILE_READ: {e}")

    # Common shapes we support:
    #  1) {"diag":[1,1,0]}
    #  2) {"blocks":{"3":[[1,0,0],[0,1,0],[0,0,0]]}}
    #  3) {"3":[[...]]}
    if isinstance(pj, dict) and "diag" in pj:
        diag = pj["diag"]
        if not (isinstance(diag, list) and all(int(x) in (0,1) for x in diag)):
            raise RuntimeError("P3_BAD_DIAG: diag must be a 0/1 list")
        return [int(x) for x in diag]

    # Try blocks["3"] or top-level "3"
    M = None
    if isinstance(pj, dict) and "blocks" in pj and isinstance(pj["blocks"], dict) and "3" in pj["blocks"]:
        M = pj["blocks"]["3"]
    elif isinstance(pj, dict) and "3" in pj:
        M = pj["3"]

    if M is None:
        raise RuntimeError("P3_NOT_FOUND: expected 'diag' or blocks['3'] / '3'")

    # Verify diagonal and extract its diagonal as diag list
    if not (isinstance(M, list) and M and all(isinstance(row, list) and len(row) == len(M) for row in M)):
        raise RuntimeError("P3_SHAPE: 3x3 (square) matrix required")
    n = len(M)
    # check diagonal + idempotent over GF(2)
    # diagonal:
    for i in range(n):
        for j in range(n):
            v = int(M[i][j]) & 1
            if (i == j and v not in (0,1)) or (i != j and v != 0):
                raise RuntimeError("P3_DIAGONAL: off-diagonal entries must be 0")
    # idempotent: M^2 == M over GF(2) -> for diagonal with 0/1 it holds; we already enforced diagonal
    return [int(M[i][i]) & 1 for i in range(n)]


# --- Apply a diagonal projector to a residual matrix: R_proj = R @ diag(mask)
def _apply_diag_to_residual(R: list[list[int]], diag_bits: list[int]) -> list[list[int]]:
    if not R or not diag_bits:
        return []
    m, n = len(R), len(R[0])
    if n != len(diag_bits):
        raise RuntimeError(f"P3_SHAPE_MISMATCH: residual n={n} vs diag n={len(diag_bits)}")
    out = [[0]*n for _ in range(m)]
    for i in range(m):
        Ri = R[i]
        for j in range(n):
            out[i][j] = (int(Ri[j]) & 1) * (int(diag_bits[j]) & 1)
    return out


# --- Quick all-zero check for matrices over GF(2)
def _all_zero_mat(M: list[list[int]]) -> bool:
    if not M:
        return True
    return all((int(x) & 1) == 0 for row in M for x in row)


# --- Strict FILE provenance (no experimental override anywhere)
try:
    mode, submode, projector_filename, projector_hash, projector_diag = _resolve_projector_from_rc()
except RuntimeError as e:
    st.error(str(e))
    st.stop()





 # ================= Parity · Run Suite (final, with AUTO/FILE guards) =================
with st.expander("Parity · Run Suite"):
    table = st.session_state.get("parity_pairs_table") or []
    if not table:
        st.info("No pairs loaded. Use Import or Insert Defaults above.")
    else:
        # --- Policy decision for this suite run
        mode, submode = _policy_from_hint()  # -> ("strict"|"projected", sub in {"","auto","file"})
        rc = st.session_state.get("run_ctx") or {}
        projector_filename = rc.get("projector_filename","") if (mode=="projected" and submode=="file") else ""

              

        def _projector_diag_from_file(pth: str):
            # Expect a JSON with blocks["3"] as an n3×n3 binary matrix
            try:
                obj = _json.loads(Path(pth).read_text(encoding="utf-8"))
                M = ((obj.get("blocks") or {}).get("3") or [])
                # basic shape & diagonal checks happen later per pair (n3 known then)
                diag = []
                for i, row in enumerate(M):
                    diag.append(int(row[i]) & 1)
                return diag
            except Exception as e:
                return None

        if mode == "projected" and submode == "file":
            if not projector_filename or not _path_exists_strict(projector_filename):
                st.error("Projector FILE required but missing/invalid. (projected:file) — block run.")
                st.stop()
            try:
                projector_hash = _sha256_hex(Path(projector_filename).read_bytes())
            except Exception as e:
                st.error(f"Could not hash projector file: {e}")
                st.stop()
            projector_diag = _projector_diag_from_file(projector_filename)
            if projector_diag is None:
                st.error("P3_SHAPE: projector file unreadable or missing blocks['3'] square matrix.")
                st.stop()

        def _dims_from_boundaries(boundaries_obj) -> dict:
            """
            Return {"n2": rows, "n3": cols} from the pair's D3 incidence (3→2) matrix.
            Falls back to zeros if unavailable.
            """
            try:
                B  = (boundaries_obj or {}).blocks.__root__
                d3 = B.get("3->2") or B.get("3") or []
                if d3 and d3[0]:
                    return {"n2": len(d3), "n3": len(d3[0])}
            except Exception:
                pass
            return {"n2": 0, "n3": 0}

        # --- Header bits
        c1, c2, c3 = st.columns([2,2,2])
        with c1:
            pill = "strict" if mode=="strict" else (f"projected({submode})" + (f" · {_short_hash(projector_hash)}" if submode=="file" else ""))
            st.caption("Policy"); st.code(pill, language="text")
        with c2:
            st.caption("Pairs in table"); st.code(str(len(table)), language="text")
        with c3:
            nonce_src = _json.dumps({"mode":mode,"sub":submode,"pj":projector_hash,"n":len(table)}, sort_keys=True, separators=(",",":")).encode("utf-8")
            parity_nonce = _sha256_hex(nonce_src)[:8]
            st.caption("Run nonce"); st.code(parity_nonce, language="text")
        

            
# --- Run button (clean, self-contained) --------------------------------------
if st.button("▶ Run Parity Suite", key="pp_btn_run_suite_final"):
    report_pairs: list[dict] = []
    skipped: list[dict] = []
    projected_green_count = 0
    seen_pair_hashes: set[str] = set()

    # Helper for safe tags in filenames
    def _safe_tag(s: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)

    # Iterate specs
    for spec in table:
        label = spec.get("label", "PAIR")
        L_in  = spec.get("left")  or {}
        R_in  = spec.get("right") or {}

        # Resolve sides (no fuzzy)
        okL, fxL = _resolve_side_or_skip_exact(L_in, label=label, side_name="left")
        okR, fxR = _resolve_side_or_skip_exact(R_in, label=label, side_name="right")
        if okL != "ok":
            skipped.append(fxL); continue
        if okR != "ok":
            skipped.append(fxR); continue

        # Pre-hash & dedupe by pair inputs (LH/RH × boundaries,shapes,cmap,H)
        left_hashes  = _hash_fixture_side(fxL)
        right_hashes = _hash_fixture_side(fxR)
        p_hash       = _pair_hash(left_hashes, right_hashes)
        if p_hash in seen_pair_hashes:
            skipped.append({"label": label, "side": "both", "error": "DUP_PAIR", "pair_hash": p_hash})
            continue
        seen_pair_hashes.add(p_hash)

        # Lane mask from this pair (left is sufficient for your model)
        lane_mask_vec = _lane_mask_from_boundaries(fxL["boundaries"])
        lane_mask_str = "".join("1" if int(x) else "0" for x in lane_mask_vec)

        # STRICT leg (booleans via gate)
        outL_s = _pp_one_leg(fxL["boundaries"], fxL["cmap"], fxL["H"], None)
        outR_s = _pp_one_leg(fxR["boundaries"], fxR["cmap"], fxR["H"], None)
        s_k2   = _bool_and(outL_s.get("2", {}).get("eq"), outR_s.get("2", {}).get("eq"))
        s_k3   = _bool_and(outL_s.get("3", {}).get("eq"), outR_s.get("3", {}).get("eq"))

        # STRICT residual (used for tagging and as baseline for AUTO/FILE projection)
        R3_L = _r3_from_fixture(fxL)
        R3_R = _r3_from_fixture(fxR)
        try:
            strict_tag = _residual_enum_from_leg(fxL["boundaries"], fxL["cmap"], fxL["H"], None)
        except Exception:
            strict_tag = "none" if bool(s_k3) else "ker"
        if s_k3 is False and strict_tag == "none":
            strict_tag = "ker"  # enforce consistency (never "none" when eq=false)

        # PROJECTED leg (only when mode == "projected")
        proj_block = None
        if mode == "projected":
            cfg = {"source": {"2": "file", "3": submode}, "projector_files": {}}
            if submode == "file":
                cfg["projector_files"]["3"] = projector_filename

            # Gate booleans under selected projector policy (keeps provenance)
            outL_p   = _pp_one_leg(fxL["boundaries"], fxL["cmap"], fxL["H"], cfg)
            outR_p   = _pp_one_leg(fxR["boundaries"], fxR["cmap"], fxR["H"], cfg)
            p_k2_gate = _bool_and(outL_p.get("2", {}).get("eq"), outR_p.get("2", {}).get("eq"))
            p_k3_gate = _bool_and(outL_p.get("3", {}).get("eq"), outR_p.get("3", {}).get("eq"))

            if submode == "auto":
                # Π_auto = diag(lane_mask) applied to STRICT residual
                R3_L_proj = _apply_diag_to_residual(R3_L, lane_mask_vec)
                R3_R_proj = _apply_diag_to_residual(R3_R, lane_mask_vec)
                p_k3_calc = _all_zero_mat(R3_L_proj) and _all_zero_mat(R3_R_proj)

                proj_tag_L = _classify_residual(R3_L_proj, lane_mask_vec)
                proj_tag_R = _classify_residual(R3_R_proj, lane_mask_vec)
                proj_tag   = proj_tag_L if proj_tag_L != "none" else proj_tag_R

                proj_block = {
                    "k2": bool(p_k2_gate),
                    "k3": bool(p_k3_calc),
                    "residual_tag": proj_tag,
                    "projector_hash": _hash_obj(lane_mask_vec),  # per-pair provenance
                }
                if p_k3_calc:
                    projected_green_count += 1

            else:
                # FILE mode: require diag(P) to match lane_mask; skip on mismatch
                diag = projector_diag or []
                n3   = len(lane_mask_vec)
                reason = None
                if len(diag) != n3:
                    reason = "P3_SHAPE"
                elif any((int(diag[j]) & 1) != (int(lane_mask_vec[j]) & 1) for j in range(n3)):
                    reason = "P3_LANE_MISMATCH"
                if reason is not None:
                    skipped.append({
                        "label": label, "side": "both", "error": reason,
                        "diag": diag, "mask": lane_mask_vec
                    })
                    continue  # do not run this pair

                # Truth for projected k3 from post-projection residual (R3 @ Π)
                R3_L_proj = _apply_diag_to_residual(R3_L, diag)
                R3_R_proj = _apply_diag_to_residual(R3_R, diag)
                p_k3_calc = _all_zero_mat(R3_L_proj) and _all_zero_mat(R3_R_proj)

                proj_tag_L = _classify_residual(R3_L_proj, lane_mask_vec)
                proj_tag_R = _classify_residual(R3_R_proj, lane_mask_vec)
                proj_tag   = proj_tag_L if proj_tag_L != "none" else proj_tag_R

                proj_block = {
                    "k2": bool(p_k2_gate),   # keep k2 provenance from gate
                    "k3": bool(p_k3_calc),   # k3 truth from projected residual
                    "residual_tag": proj_tag,
                }
                if p_k3_calc:
                    projected_green_count += 1

        # Assemble pair record
        pair_out = {
            "label": label,
            "pair_hash": p_hash,
            "lane_mask_k3": lane_mask_vec,
            "lane_mask": lane_mask_str,
            "dims": _dims_from_boundaries(fxL["boundaries"]),
            "left_hashes":  left_hashes,
            "right_hashes": right_hashes,
            "strict": {"k2": bool(s_k2), "k3": bool(s_k3), "residual_tag": strict_tag},
        }
        if proj_block is not None:
            pair_out["projected"] = proj_block

        # Consistency guards (don’t crash; just ensure shape)
        if pair_out["strict"]["k3"] is False:
            if pair_out["strict"]["residual_tag"] == "none":
                pair_out["strict"]["residual_tag"] = "ker"
        if "projected" in pair_out and pair_out["projected"]["k3"] is False:
            if pair_out["projected"]["residual_tag"] == "none":
                pair_out["projected"]["residual_tag"] = "ker"

        report_pairs.append(pair_out)

    # ---- Build report root ---------------------------------------------------
    rows_total   = len(table)              # input queue size
    rows_run     = len(report_pairs)       # executed, unique by hash
    rows_skipped = len(skipped)
    if mode == "projected":
        pct = (projected_green_count / rows_run) if rows_run else 0.0
    else:
        pct = 0.0

    policy_tag_run = "strict" if mode == "strict" else f"projected(columns@k=3,{submode})"
    lane_mask_note = ("AUTO projector uses each pair’s lane mask" if (mode == "projected" and submode == "auto") else "")
    run_note = ("Projected leg omitted in strict mode." if mode == "strict" else lane_mask_note)

    report = {
        "schema_version": "1.0.0",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "app_version": APP_VERSION,
        "policy_tag": policy_tag_run,
        "projector_mode": ("strict" if mode == "strict" else submode),
        "projector_filename": (projector_filename if (mode == "projected" and submode == "file") else ""),
        "projector_hash": (projector_hash if (mode == "projected" and submode == "file") else ""),
        "lane_mask_note": (lane_mask_note if lane_mask_note else ""),
        "run_note": run_note,
        "residual_method": "R3 strict vs R3·Π (lanes/ker/mixed)",
        "summary": {
            "rows_total": rows_total,
            "rows_skipped": rows_skipped,
            "rows_run": rows_run,
            "projected_green_count": (projected_green_count if mode == "projected" else 0),
            "pct": (pct if mode == "projected" else 0.0),
        },
        "pairs": report_pairs,
        "skipped": skipped,
    }

    # Deterministic content hash (BEFORE filenames)
    report["content_hash"] = _sha256_hex(
        _json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    # Per-run filenames (content-hashed)
    safe_tag   = _safe_tag(report["policy_tag"])
    short_hash = report["content_hash"][:12]
    json_name  = f"parity_report__{safe_tag}__{short_hash}.json"
    csv_name   = f"parity_summary__{safe_tag}__{short_hash}.csv"
    json_path  = REPORTS_DIR / json_name
    csv_path   = REPORTS_DIR / csv_name

    # --- Build CSV rows once (no duplicates—the loop deduped by pair_hash)
    hdr = [
        "label","policy_tag","projector_mode","projector_hash",
        "strict_k2","strict_k3","projected_k2","projected_k3",
        "residual_strict","residual_projected","lane_mask",
        "lh_boundaries_hash","lh_shapes_hash","lh_cmap_hash","lh_H_hash",
        "rh_boundaries_hash","rh_shapes_hash","rh_cmap_hash","rh_H_hash",
        "pair_hash",
    ]
    rows_csv = []
    for p in report_pairs:
        proj = p.get("projected") or {}
        rows_csv.append([
            p["label"], report["policy_tag"], report["projector_mode"], report["projector_hash"],
            str(p["strict"]["k2"]).lower(), str(p["strict"]["k3"]).lower(),
            ("" if not proj else str(proj.get("k2", False)).lower()),
            ("" if not proj else str(proj.get("k3", False)).lower()),
            p["strict"].get("residual_tag",""),
            ("" if not proj else proj.get("residual_tag","")),
            p.get("lane_mask",""),
            p["left_hashes"]["boundaries"], p["left_hashes"]["shapes"], p["left_hashes"]["cmap"], p["left_hashes"]["H"],
            p["right_hashes"]["boundaries"], p["right_hashes"]["shapes"], p["right_hashes"]["cmap"], p["right_hashes"]["H"],
            p["pair_hash"],
        ])

    # Persist to session for UI
    st.session_state["parity_last_full_report"]  = report
    st.session_state["parity_last_report_pairs"] = report_pairs
    st.session_state["parity_last_report_path"]  = str(json_path)
    st.session_state["parity_last_summary_path"] = str(csv_path)

    # Write JSON (best-effort)
    try:
        _guarded_atomic_write_json(json_path, report)
    except Exception as e:
        st.info(f"(Could not write {json_name}: {e})")

    # Write CSV (best-effort)
    try:
        tmp_csv = csv_path.with_suffix(".csv.tmp")
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows_csv)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_csv, csv_path)
    except Exception as e:
        st.info(f"(Could not write {csv_name}: {e})")

    # --- UI summary + downloads
    st.success(
        "Run complete · "
        f"pairs={rows_run} · skipped={rows_skipped}"
        + (f" · GREEN={projected_green_count} ({pct:.2%})" if mode == "projected" else "")
    )

    # In-memory downloads with unique keys
    json_mem = _io.BytesIO(_json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
    csv_mem  = _io.StringIO(); w = csv.writer(csv_mem); w.writerow(hdr); w.writerows(rows_csv)
    csv_bytes = _io.BytesIO(csv_mem.getvalue().encode("utf-8"))

    dl_prefix = "strict" if mode == "strict" else f"proj_{submode}"
    st.download_button("Download parity_report.json",  json_mem, file_name=json_name, key=f"{dl_prefix}_json_{short_hash}")
    st.download_button("Download parity_summary.csv", csv_bytes, file_name=csv_name, key=f"{dl_prefix}_csv_{short_hash}")

    # Compact ✓/✗ preview
    if report_pairs:
        st.caption("Summary (strict_k3 / projected_k3):")
        for p in report_pairs:
            s  = "✅" if p["strict"]["k3"] else "❌"
            pr = "—" if "projected" not in p else ("✅" if p["projected"]["k3"] else "❌")
            st.write(f"• {p['label']} → strict={s} · projected={pr}")

          



                                                        
                
      
              
                                        





                        

            

# =============================================================================== 
                      
            


# ================== Parity · Presets & Queue ALL valid ==================
with st.expander("Parity · Presets & Queue"):
    st.caption("Insert a preset spec into the editable table, then **Queue ALL valid pairs** to resolve and add them to the live queue used by the runner. No fuzzy search: missing fields/paths are listed under Skipped.")

    def _insert_preset_payload(payload: dict, *, name: str):
        try:
            pairs, policy_hint = validate_pairs_payload(payload)
            st.session_state["parity_pairs_table"] = pairs
            st.session_state["parity_policy_hint"] = policy_hint
            st.success(f"Inserted preset: {name} · {len(pairs)} pair(s)")
        except Exception as e:
            st.error(f"PARITY_SCHEMA_INVALID: {e}")

    cA, cB, cC, cD = st.columns([2,2,2,2])

    # --- 1) Row Parity preset (within-district row1 vs row2)
    with cA:
        if st.button("Insert defaults · Row Parity", key="pp_preset_row"):
            _insert_preset_payload({
                "schema_version": "1.0.0",
                "policy_hint": "mirror_active",
                "pairs": [
                    {
                        "label": "D3 • row1(101) ↔ row2(010)",
                        "left":  {
                            "boundaries": "inputs/D3/boundaries.json",
                            "shapes":     "inputs/D3/shapes.json",
                            "cmap":       "inputs/D3/Cmap_C3_bottomrow_101_D2D3.json",
                            "H":          "inputs/D3/H_row1_selector.json"
                        },
                        "right": {
                            "boundaries": "inputs/D3/boundaries.json",
                            "shapes":     "inputs/D3/shapes.json",
                            "cmap":       "inputs/D3/Cmap_C3_bottomrow_010_D3.json",
                            "H":          "inputs/D3/H_row2_selector.json"
                        }
                    },
                    {
                        "label": "D2 • row1(101) ↔ row2(011)",
                        "left":  {
                            "boundaries": "inputs/D2/boundaries.json",
                            "shapes":     "inputs/D2/shapes.json",
                            "cmap":       "inputs/D2/Cmap_C3_bottomrow_101_D2.json",
                            "H":          "inputs/D2/H_row1_selector.json"
                        },
                        "right": {
                            "boundaries": "inputs/D2/boundaries.json",
                            "shapes":     "inputs/D2/shapes.json",
                            "cmap":       "inputs/D2/Cmap_C3_bottomrow_011_D2.json",
                            "H":          "inputs/D2/H_row2_selector.json"
                        }
                    },
                    {
                        "label": "D4 • row1(101) ↔ row2(011)",
                        "left":  {
                            "boundaries": "inputs/D4/boundaries.json",
                            "shapes":     "inputs/D4/shapes.json",
                            "cmap":       "inputs/D4/cmap_C3_bottomrow_101_D4.json",
                            "H":          "inputs/D4/H_row1_selector.json"
                        },
                        "right": {
                            "boundaries": "inputs/D4/boundaries.json",
                            "shapes":     "inputs/D4/shapes.json",
                            "cmap":       "inputs/D4/Cmap_C3_bottomrow_011_D4.json",
                            "H":          "inputs/D4/H_row2_selector.json"
                        }
                    }
                ]
            }, name="Row Parity")

    # --- 2) District Parity preset (cross-district, same-H chains)
    with cB:
        if st.button("Insert defaults · District Parity", key="pp_preset_district"):
            _insert_preset_payload({
                "schema_version": "1.0.0",
                "policy_hint": "mirror_active",
                "pairs": [
                    {
                        "label": "row1 chain • D2(101) ↔ D3(110)",
                        "left":  {
                            "boundaries": "inputs/D2/boundaries.json",
                            "shapes":     "inputs/D2/shapes.json",
                            "cmap":       "inputs/D2/Cmap_C3_bottomrow_101_D2.json",
                            "H":          "inputs/D2/H_row1_selector.json"
                        },
                        "right": {
                            "boundaries": "inputs/D3/boundaries.json",
                            "shapes":     "inputs/D3/shapes.json",
                            "cmap":       "inputs/D3/Cmap_C3_bottomrow_110_D3.json",
                            "H":          "inputs/D3/H_row1_selector.json"
                        }
                    },
                    {
                        "label": "row1 chain • D3(110) ↔ D4(101)",
                        "left":  {
                            "boundaries": "inputs/D3/boundaries.json",
                            "shapes":     "inputs/D3/shapes.json",
                            "cmap":       "inputs/D3/Cmap_C3_bottomrow_110_D3.json",
                            "H":          "inputs/D3/H_row1_selector.json"
                        },
                        "right": {
                            "boundaries": "inputs/D4/boundaries.json",
                            "shapes":     "inputs/D4/shapes.json",
                            "cmap":       "inputs/D4/cmap_C3_bottomrow_101_D4.json",
                            "H":          "inputs/D4/H_row1_selector.json"
                        }
                    },
                    {
                        "label": "row2 chain • D2(011) ↔ D4(011)",
                        "left":  {
                            "boundaries": "inputs/D2/boundaries.json",
                            "shapes":     "inputs/D2/shapes.json",
                            "cmap":       "inputs/D2/Cmap_C3_bottomrow_011_D2.json",
                            "H":          "inputs/D2/H_row2_selector.json"
                        },
                        "right": {
                            "boundaries": "inputs/D4/boundaries.json",
                            "shapes":     "inputs/D4/shapes.json",
                            "cmap":       "inputs/D4/Cmap_C3_bottomrow_011_D4.json",
                            "H":          "inputs/D4/H_row2_selector.json"
                        }
                    }
                ]
            }, name="District Parity")

        # --- 3) Smoke preset (inline; zero path dependency)
    if st.button("Insert defaults · Smoke (inline)", key="pp_preset_smoke"):
        _insert_preset_payload({
            "schema_version": "1.0.0",
            "policy_hint": "projected:auto",
            "pairs": [
                {
                    "label": "SELF • ker-only vs ker-only (D3 dims 2×3)",
                    "left": {
                        "embedded": {
                            "boundaries": {
                                "name": "D3 dims",
                                "blocks": { "3->2": [[1,1,0],[0,1,0]] }
                            },
                            "shapes": { "n3": 3, "n2": 2 },
                            "cmap": {
                                "name": "C3 ker-only; C2=I",
                                "blocks": {
                                    "3": [[1,0,0],[0,1,0],[0,0,0]],
                                    "2": [[1,0],[0,1]]
                                }
                            },
                            "H": {
                                "name": "H=0",
                                "blocks": { "2": [[0,0],[0,0],[0,0]] }
                            }
                        }
                    },
                    "right": {
                        "embedded": {
                            "boundaries": {
                                "name": "D3 dims",
                                "blocks": { "3->2": [[1,1,0],[0,1,0]] }
                            },
                            "shapes": { "n3": 3, "n2": 2 },
                            "cmap": {
                                "name": "C3 ker-only; C2=I",
                                "blocks": {
                                    "3": [[1,0,0],[0,1,0],[0,0,0]],
                                    "2": [[1,0],[0,1]]
                                }
                            },
                            "H": {
                                "name": "H=0",
                                "blocks": { "2": [[0,0],[0,0],[0,0]] }
                            }
                        }
                    }
                }
            ],
            "name": "Smoke (inline)"
        })

    # --- 4) Quick preview of current table ---
    table = st.session_state.get("parity_pairs_table") or []
    if table:
        st.caption(f"Pairs in table: {len(table)}")
        try:
            import pandas as pd
            st.dataframe(pd.DataFrame([{"label": p.get("label","PAIR")} for p in table]),
                         hide_index=True, use_container_width=True)
        except Exception:
            st.write("\n".join("• " + (p.get("label","PAIR") or "") for p in table[:6]))

    # --- 5) Queue ALL valid pairs (strict resolver; record skips) ---
    if st.button("Queue ALL valid pairs", key="pp_queue_all_valid", disabled=not bool(table)):
        queued = 0
        skipped = []
        # Reset the live queue before enqueuing (fresh, reproducible)
        st.session_state["parity_pairs"] = []

        for spec in table:
            label = spec.get("label","PAIR")
            L     = spec.get("left") or {}
            R     = spec.get("right") or {}

            okL, fxL = _resolve_side_or_skip_exact(L, label=label, side_name="left")
            okR, fxR = _resolve_side_or_skip_exact(R, label=label, side_name="right")
            if okL != "ok":
                skipped.append(fxL); continue
            if okR != "ok":
                skipped.append(fxR); continue

            # Enqueue the fully parsed fixtures used by the runner
            add_parity_pair(label=label, left_fixture=fxL, right_fixture=fxR)
            queued += 1

        if queued:
            st.success(f"Queued {queued} pair(s) into the live parity queue.")
        if skipped:
            st.info("Skipped specs:")
            for s in skipped[:20]:
                if "missing" in s:
                    st.write(f"• {s['label']} [{s['side']}] → PARITY_SPEC_MISSING: {s['missing']}")
                else:
                    st.write(f"• {s['label']} [{s['side']}] → {s.get('error','error')}")

    # Optional tiny helper on the right
    with cD:
        st.caption("Tips")
        st.markdown("- Edit paths in your presets before Queue ALL.\n- Inline preset never needs files.\n- FILE mode requires a valid projector file.")
# ======================================================================




    
    
   






# ───────────────────────── Snapshot & Flush — self-contained ─────────────────────────
import os, json, csv, zipfile, secrets, shutil, hashlib, tempfile, platform
from pathlib import Path
from datetime import datetime, timezone

# ---------- Paths (safe defaults) ----------
LOGS_DIR       = Path(globals().get("LOGS_DIR", "logs"))
CERTS_DIR      = Path(globals().get("CERTS_DIR", LOGS_DIR / "certs"))
REPORTS_DIR    = Path(globals().get("REPORTS_DIR", "reports"))
BUNDLES_DIR    = Path(globals().get("BUNDLES_DIR", "bundles"))
PROJECTORS_DIR = Path(globals().get("PROJECTORS_DIR", "projectors"))


SCHEMA_VERSION = globals().get("SCHEMA_VERSION", "1.0.0")
APP_VERSION    = globals().get("APP_VERSION", "v0.1-core")

# ─────────────── tiny utils (no widgets; used by both features) ───────────────
def flush_workspace(*, delete_projectors: bool=False) -> dict:
    summary = {"when": datetime.now(timezone.utc).isoformat(), "deleted_dirs": [], "recreated_dirs": [], "files_removed": 0, "token": "", "composite_cache_key_short": ""}

    # Session clears
    for k in (
        "_inputs_block","_district_info","run_ctx","overlap_out","overlap_H",
        "residual_tags","ab_compare","last_cert_path","cert_payload",
        "last_run_id","_gallery_keys","_last_boundaries_hash",
        "_projector_cache","_projector_cache_ab","parity_pairs",
        "parity_last_report_pairs","selftests_snapshot","_last_cert_write_key",
    ):
        st.session_state.pop(k, None)

    # Disk clears
    dirs = [CERTS_DIR, LOGS_DIR, REPORTS_DIR, BUNDLES_DIR]
    if delete_projectors: dirs.append(PROJECTORS_DIR)

    removed = 0
    for d in dirs:
        if d.exists():
            for _root, _dirs, files in os.walk(d):
                removed += len(files)
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        summary["deleted_dirs"].append(str(d))
        summary["recreated_dirs"].append(str(d))
    summary["files_removed"] = removed

    # Bump a token/key
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    salt = secrets.token_hex(2).upper()
    token = f"FLUSH-{ts}-{salt}"
    ckey  = hashlib.sha256((ts+salt).encode()).hexdigest()
    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"] = token
    summary["token"] = token
    summary["composite_cache_key_short"] = ckey[:12]
    return summary


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _ymd_hms_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return p.as_posix()

def _read_json_safely(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

def _nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0

def _py_version_str() -> str:
    return f"python-{platform.python_version()}"

def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n

def _discover_certs() -> list[Path]:
    """
    Fresh, no-cache discovery of cert JSONs.
    Priority:
      1) directory of the most recent cert (if present)
      2) logs/certs/**/*
      3) CERTS_DIR/**/*
    """
    roots: list[Path] = []
    # 1) dir of last written cert if any (from session)
    try:
        import streamlit as st  # local import to avoid hard dependency elsewhere
        last = (st.session_state.get("last_cert_path") or "").strip()
    except Exception:
        last = ""
    if last:
        try:
            roots.append(Path(last).resolve().parent)
        except Exception:
            pass
    # 2) default tree and 3) explicit CERTS_DIR
    roots.extend([Path("logs/certs"), CERTS_DIR])

    seen = set()
    out: list[Path] = []
    for r in roots:
        try:
            r = r.resolve()
        except Exception:
            continue
        if not r.exists():
            continue
        for p in r.rglob("*.json"):
            key = p.resolve().as_posix()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    # newest first
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return out

# ---- helper: recursively discover certs under CERTS_DIR ----
from pathlib import Path
import os

# Respect existing global if present; otherwise default to logs/certs
CERTS_DIR = Path(globals().get("CERTS_DIR", "logs/certs"))

def _discover_certs() -> list[Path]:
    """
    Recursively find all *.json cert files under CERTS_DIR, sorted.
    """
    out: list[Path] = []
    if CERTS_DIR.exists():
        for dirpath, _, filenames in os.walk(CERTS_DIR):
            for fn in filenames:
                if fn.lower().endswith(".json"):
                    out.append(Path(dirpath) / fn)
    out.sort()
    return out
# ─────────────────────────── SNAPSHOT BUILDER ───────────────────────────
def build_everything_snapshot() -> str:
    """
    Builds a ZIP with certs, referenced projectors, logs, reports, metadata, and an index.
    Reads the cert schema as produced by the current writer:
      - root.written_at_utc
      - policy.{canon,label_raw,policy_tag,projector_hash,projector_filename}
      - inputs.hashes.{boundaries_hash,C_hash,H_hash,U_hash}
      - identity.{district_id,run_id,run_idx}
      - hashes.content_hash
    Returns absolute path to the ZIP (or "" if nothing found).
    """
    # Local import (keeps function portable outside Streamlit)
    try:
        pass

    except Exception:
        st = None  # Optional; used only for user-facing info

    # --- Collect certs (recursive under CERTS_DIR) ---
    cert_files = _discover_certs()

    parsed, skipped = [], []
    for p in cert_files:
        data, err = _read_json_safely(p)
        if err or not isinstance(data, dict):
            skipped.append({"path": _rel(p), "reason": "JSON_PARSE_ERROR"})
            continue
        parsed.append((p, data))

    if not parsed:
        if st:
            st.info("Nothing to snapshot yet (no parsed certs).")
        return ""

    # ... keep the rest of your function unchanged ...
    # (proj_refs, districts, index_rows, manifest_files; projectors/logs/reports; manifest; zip; return)


    proj_refs, districts, index_rows, manifest_files = set(), set(), [], []

    for p, cert in parsed:
        ident   = cert.get("identity") or {}
        pol     = cert.get("policy") or {}
        inputs  = cert.get("inputs") or {}
        hashes  = cert.get("hashes") or {}

        did = (ident.get("district_id") or "UNKNOWN")
        districts.add(str(did))

        manifest_files.append({
            "path": _rel(p),
            "sha256": _sha256_file(p),
            "size": p.stat().st_size,
        })

        # projector (present when projected:file)
        pj_fname = pol.get("projector_filename") or ""
        if isinstance(pj_fname, str) and pj_fname.strip():
            proj_refs.add(pj_fname.strip())

        # prefer nested inputs.hashes; fallback to flat keys if older certs exist
        inp_h = (inputs.get("hashes") or {})
        def _hx(k: str) -> str:
            return str(inp_h.get(k) or inputs.get(k) or "")

        index_rows.append([
            _rel(p),
            str(hashes.get("content_hash", "")),
            str(pol.get("policy_tag", pol.get("canon", ""))),
            str(did),
            str(ident.get("run_id", "")),
            str(cert.get("written_at_utc", "")),
            _hx("boundaries_hash"),
            _hx("C_hash"),
            _hx("H_hash"),
            _hx("U_hash"),
            str(pol.get("projector_hash", "")),
            str(pol.get("projector_filename", "")),
        ])

    # Resolve & include referenced projectors
    projectors, missing_projectors = [], []
    for pj in sorted(proj_refs):
        pj_path = Path(pj)
        if not pj_path.exists():
            alt = PROJECTORS_DIR / pj_path.name
            if alt.exists():
                pj_path = alt
            else:
                missing_projectors.append({"filename": _rel(pj_path), "referenced_by": "certs/* (various)"})
                continue
        projectors.append({"path": _rel(pj_path), "sha256": _sha256_file(pj_path), "size": pj_path.stat().st_size})

    # Logs & reports (optional)
    logs_list = []
    for name in ("gallery.jsonl", "witnesses.jsonl"):
        p = LOGS_DIR / name
        if _nonempty(p):
            logs_list.append({"path": _rel(p), "sha256": _sha256_file(p), "size": p.stat().st_size})

    reports_list = []
    for rp in ("parity_report.json", "coverage_sampling.csv", "perturbation_sanity.csv", "fence_stress.csv"):
        p = REPORTS_DIR / rp
        if _nonempty(p):
            reports_list.append({"path": _rel(p), "sha256": _sha256_file(p), "size": p.stat().st_size})

    # Assemble manifest
    app_ver = APP_VERSION
    py_ver  = _py_version_str()
    districts_sorted = sorted(districts)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_kind": "everything-snapshot",
        "written_at_utc": _utc_iso_z(),
        "app_version": app_ver,
        "python_version": py_ver,
        "districts": districts_sorted,
        "counts": {
            "certs": len(manifest_files),
            "projectors": len(projectors),
            "logs": {
                "gallery_jsonl": int(any(x["path"].endswith("gallery.jsonl") for x in logs_list)),
                "witnesses_jsonl": int(any(x["path"].endswith("witnesses.jsonl") for x in logs_list)),
            },
            "reports": {
                "parity":   int(any(x["path"].endswith("parity_report.json") for x in reports_list)),
                "coverage": int(any(x["path"].endswith("coverage_sampling.csv") for x in reports_list)),
                "perturb":  int(any(x["path"].endswith("perturbation_sanity.csv") for x in reports_list)),
                "fence":    int(any(x["path"].endswith("fence_stress.csv") for x in reports_list)),
            }
        },
        "files": manifest_files,
        "projectors": projectors,
        "logs": logs_list,
        "reports": reports_list,
        "skipped": skipped,
        "missing_projectors": missing_projectors,
        "notes": "Certs are authoritative; only projectors referenced by any cert are included."
    }

    # cert_index.csv
    index_header = [
        "cert_path","content_hash","policy_tag","district_id","run_id","written_at_utc",
        "boundaries_hash","C_hash","H_hash","U_hash","projector_hash","projector_filename"
    ]
    fd, idx_tmp = tempfile.mkstemp(prefix=".tmp_cert_index_", suffix=".csv")
    os.close(fd)
    try:
        with open(idx_tmp, "w", newline="", encoding="utf-8") as tf:
            w = csv.writer(tf)
            w.writerow(index_header)
            w.writerows(index_rows)
        with open(idx_tmp, "r", encoding="utf-8") as tf:
            index_csv_text = tf.read()
    finally:
        try:
            os.remove(idx_tmp)
        except Exception:
            pass

    # Create ZIP (atomic)
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    tag   = next(iter(districts_sorted)) if len(districts_sorted) == 1 else "MULTI"
    zname = f"snapshot__{tag}__{_ymd_hms_compact()}.zip"
    zpath = BUNDLES_DIR / zname
    fd, tmpname = tempfile.mkstemp(dir=BUNDLES_DIR, prefix=".tmp_snapshot_", suffix=".zip")
    os.close(fd)
    tmpzip = Path(tmpname)

    try:
        with zipfile.ZipFile(tmpzip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            zf.writestr("cert_index.csv", index_csv_text)
            # Add all referenced files
            for f in manifest_files + projectors + logs_list + reports_list:
                p = Path(f["path"])
                if p.exists():
                    zf.write(p.as_posix(), arcname=f["path"])
        os.replace(tmpzip, zpath)
    finally:
        if tmpzip.exists():
            try:
                tmpzip.unlink()
            except Exception:
                pass

    return str(zpath)

# ─────────────────────────── SESSION-ONLY RESET ───────────────────────────
def _session_flush_run_cache():
    """
    Clear computed session keys only; do not touch disk.
    Bumps a nonce and sets a fresh cache key/token in session_state.
    """
    import streamlit as st  # local import

    # Prefer app’s soft reset if available
    if "_soft_reset_before_overlap" in globals():
        try:
            _soft_reset_before_overlap()  # type: ignore[name-defined]
        except Exception:
            pass

    # Minimal local fallback (safe)
    for k in (
        "run_ctx","overlap_out","overlap_cfg","overlap_policy_label",
        "overlap_H","residual_tags","ab_compare",
        "cert_payload","last_cert_path","_last_cert_write_key",
        "_projector_cache","_projector_cache_ab"
    ):
        st.session_state.pop(k, None)

    # Bump nonce + new composite key/token
    st.session_state["_fixture_nonce"] = int(st.session_state.get("_fixture_nonce", 0)) + 1
    ts = _utc_iso_z()
    salt = secrets.token_hex(2).upper()
    token = f"RUN-FLUSH-{ts}-{salt}"
    ckey  = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()
    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"] = token
    return {"token": token, "ckey_short": ckey[:12]}



# ───────────────────────── Flush/Reset buttons (C) ─────────────────────────
# fallbacks so the UI keys don't collide
try:
    EXPORTS_NS
except NameError:
    EXPORTS_NS = "exports_v2"
try:
    _mkkey
except NameError:
    def _mkkey(ns: str, name: str) -> str:
        return f"{ns}__{name}"

# local safe wrappers (use your existing funcs when present)
def _session_flush_run_cache_safe():
    if "_session_flush_run_cache" in globals():
        return _session_flush_run_cache()  # type: ignore[name-defined]
    # minimal in-session reset
    ss = st.session_state
    for k in (
        "run_ctx","overlap_out","overlap_cfg","overlap_policy_label",
        "overlap_H","residual_tags","ab_compare","cert_payload",
        "last_cert_path","_last_cert_write_key","_projector_cache","_projector_cache_ab"
    ):
        ss.pop(k, None)
    from datetime import datetime, timezone
    import secrets, hashlib
    ts = datetime.now(timezone.utc).isoformat()
    salt = secrets.token_hex(2).upper()
    token = f"RUN-FLUSH-{ts}-{salt}"
    ckey  = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()
    ss["_composite_cache_key"] = ckey
    ss["_last_flush_token"]    = token
    ss["_fixture_nonce"]       = int(ss.get("_fixture_nonce", 0)) + 1
    return {"token": token, "ckey_short": ckey[:12]}

def _flush_workspace_safe(delete_projectors: bool=False):
    if "flush_workspace" in globals():
        return flush_workspace(delete_projectors=delete_projectors)  # type: ignore[name-defined]
    # minimal on-disk flush fallback
    from pathlib import Path
    import shutil, hashlib, secrets
    CERTS_DIR      = Path(globals().get("CERTS_DIR","certs"))
    LOGS_DIR       = Path(globals().get("LOGS_DIR","logs"))
    REPORTS_DIR    = Path(globals().get("REPORTS_DIR","reports"))
    BUNDLES_DIR    = Path(globals().get("BUNDLES_DIR","bundles"))
    PROJECTORS_DIR = Path(globals().get("PROJECTORS_DIR","projectors"))
    dirs = [CERTS_DIR, LOGS_DIR, REPORTS_DIR, BUNDLES_DIR]
    if delete_projectors:
        dirs.append(PROJECTORS_DIR)
    deleted = []
    recreated = []
    files_removed = 0
    for d in dirs:
        if d.exists():
            # count files
            for root, _, files in os.walk(d):
                files_removed += len(files)
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        deleted.append(str(d)); recreated.append(str(d))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    salt = secrets.token_hex(2).upper()
    token = f"FLUSH-{ts}-{salt}"
    ckey  = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()
    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"]    = token
    return {
        "when": ts, "deleted_dirs": deleted, "recreated_dirs": recreated,
        "files_removed": files_removed, "token": token,
        "composite_cache_key_short": ckey[:12],
    }

# ─────────────────────────── FULL WORKSPACE FLUSH (helpers) ───────────────────────────
def _session_flush_run_cache_safe():
    """Wrapper that prefers app's reset if available; otherwise minimal fallback."""
    if "_session_flush_run_cache" in globals() and callable(_session_flush_run_cache):
        return _session_flush_run_cache()
    # minimal fallback: just bump nonce + new cache key
    ts = _utc_iso_z()
    salt = secrets.token_hex(2).upper()
    token = f"RUN-FLUSH-{ts}-{salt}"
    ckey  = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()
    st.session_state["_fixture_nonce"] = int(st.session_state.get("_fixture_nonce", 0)) + 1
    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"] = token
    return {"token": token, "ckey_short": ckey[:12]}

def _flush_workspace_safe(*, delete_projectors: bool = False):
    """Wrapper that uses your full flush if present."""
    if "_full_flush_workspace" in globals() and callable(_full_flush_workspace):
        return _full_flush_workspace(delete_projectors=delete_projectors)
    # final fallback: no-op summary
    return {
        "when": _utc_iso_z(),
        "deleted_dirs": [],
        "recreated_dirs": [],
        "files_removed": 0,
        "token": "FLUSH-NOOP",
        "composite_cache_key_short": (st.session_state.get("_composite_cache_key","") or "")[:12],
    }
# ---------------------- DEFINITIVE FULL FLUSH (works even if globals differ) ----------------------
from pathlib import Path
import os, shutil, hashlib, secrets
from datetime import datetime, timezone

def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _count_files(root: Path) -> int:
    if not root.exists(): return 0
    n = 0
    for _, _, files in os.walk(root): n += len(files)
    return n

def _resolve_dir(name: str, default: str) -> Path:
    # honor globals if present; else default
    p = globals().get(name)
    if isinstance(p, (str, Path)): 
        return Path(p)
    return Path(default)

def hard_full_flush(*, delete_projectors: bool = False) -> dict:
    """
    Removes persisted outputs on disk and resets session caches.
    Targets BOTH top-level and logs/* variants to avoid dir mismatch issues.
    """
    # Resolve dirs robustly
    TOP_CERTS     = _resolve_dir("CERTS_DIR",      "certs")
    LOGS_DIR      = _resolve_dir("LOGS_DIR",       "logs")
    REPORTS_DIR   = _resolve_dir("REPORTS_DIR",    "reports")
    BUNDLES_DIR   = _resolve_dir("BUNDLES_DIR",    "bundles")
    PROJECTORS_DIR= _resolve_dir("PROJECTORS_DIR", "projectors")

    # Extra paths some UIs use
    LOGS_CERTS    = LOGS_DIR / "certs"

    # Build the list of dirs to wipe
    targets = [
        TOP_CERTS, LOGS_DIR, REPORTS_DIR, BUNDLES_DIR,
        LOGS_CERTS,                       # wipe nested certs as well
    ]
    if delete_projectors:
        targets.append(PROJECTORS_DIR)

    # Deduplicate while preserving order
    seen = set()
    dirs = []
    for d in targets:
        rp = d.resolve()
        if rp not in seen:
            seen.add(rp); dirs.append(d)

    # Clear session state keys that cache discovery/render
    for k in (
        "_inputs_block","_district_info","run_ctx","overlap_out","overlap_H",
        "residual_tags","ab_compare","last_cert_path","cert_payload",
        "last_run_id","_gallery_keys","_last_boundaries_hash",
        "_projector_cache","_projector_cache_ab",
        "parity_pairs","parity_last_report_pairs","selftests_snapshot",
        "_last_cert_write_key","_has_overlap","_gallery_seen_keys",
        "_gallery_bootstrapped","_composite_cache_key","_last_flush_token"
    ):
        st.session_state.pop(k, None)

    removed_files = 0
    for d in dirs:
        try:
            if d.exists():
                removed_files += _count_files(d)
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        # recreate
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # New cache key + token
    ts    = _utc_iso_z()
    salt  = secrets.token_hex(2).upper()
    token = f"FLUSH-{ts}-{salt}"
    ckey  = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()
    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"]    = token

    return {
        "when": ts,
        "deleted_dirs": [str(d) for d in dirs],
        "recreated_dirs": [str(d) for d in dirs],
        "files_removed": removed_files,
        "token": token,
        "composite_cache_key_short": ckey[:12],
    }

# ─────────────────────────── EXPORTS (Snapshot + Flush) ───────────────────────────
EXPORTS_NS = "exports_v2"  # keep consistent across app

with safe_expander("Exports", expanded=False):
    c1, c2 = st.columns(2)

    # ── Snapshot (left column)
    with c1:
        if st.button("Build Snapshot ZIP", key=_mkkey(EXPORTS_NS, "btn_build_snapshot")):
            try:
                zp = build_everything_snapshot()
                if zp:
                    st.success(f"Snapshot ready → {zp}")
                    with open(zp, "rb") as fz:
                        st.download_button(
                            "Download snapshot.zip",
                            fz,
                            file_name=os.path.basename(zp),
                            key=_mkkey(EXPORTS_NS, "dl_snapshot_zip"),
                        )
                else:
                    st.info("Nothing to snapshot yet.")
            except Exception as e:
                st.error(f"Snapshot failed: {e}")

   
    with c2:
        st.caption("Flush / Reset")
    
        if st.button(
            "Quick Reset (session only)",
            key=_mkkey(EXPORTS_NS, "btn_quick_reset_session"),
            help="Clears computed session data, bumps nonce; does not touch files.",
        ):
            out = _session_flush_run_cache_safe()
            st.success(f"Run cache flushed · token={out['token']} · key={out['ckey_short']}")
    
        inc_pj  = st.checkbox("Also remove projectors (full flush)",
                              value=False, key=_mkkey(EXPORTS_NS, "flush_inc_pj"))
        confirm = st.checkbox("I understand this deletes files on disk",
                              value=False, key=_mkkey(EXPORTS_NS, "ff_confirm"))
        if st.button(
            "Full Flush (certs/logs/reports/bundles)",
            key=_mkkey(EXPORTS_NS, "btn_full_flush"),
            disabled=not confirm,
            help="Deletes persisted outputs; keeps inputs. Bumps nonce & resets session.",
        ):
            try:
                info = hard_full_flush(delete_projectors=inc_pj)
                st.success(f"Workspace flushed · {info['token']}")
                st.caption(f"New cache key: `{info['composite_cache_key_short']}`")
                if st.checkbox("Show flush details", key=_mkkey(EXPORTS_NS, "show_flush_details")):
                    st.json(info)
            except Exception as e:
                st.error(f"Flush failed: {e}")






                        

 

with tab3:
    st.subheader("Triangle gate (Echo)")
    
    # Second homotopy H' (the first H is taken from tab2 via session_state)
    f_H2 = st.file_uploader("Second homotopy H' (JSON)", type=["json"], key="H2_up")
    _stamp_filename("fname_H2", f_H2)
    d_H2 = read_json_file(f_H2) if f_H2 else None
    H2 = io.parse_cmap(d_H2) if d_H2 else None
    
    # Pull H from tab2 (if loaded)
    H = st.session_state.get("H_obj")
    
    # Reuse the same active policy you compute in tab2 (strict/projected)
    # If you compute cfg_active in tab2's scope, rebuild it here the same way or store it in session_state
    cfg_active = st.session_state.get("cfg_active")  # if you saved it; otherwise rebuild
    
    if st.button("Run Triangle"):
        if boundaries is None or cmap is None:
            st.error("Load Boundaries and C in Unit tab first.")
        elif H is None:
            st.error("Upload H in Overlap tab first.")
        elif H2 is None:
            st.error("Upload H' here.")
        else:
            try:
                outT = triangle_gate.triangle_check(
                    boundaries, cmap, H, H2,
                    projection_config=cfg_active,
                    projector_cache=projector.preload_projectors_from_files(cfg_active)
                )
                st.json(outT)
            except TypeError:
                # fallback if triangle_check doesn’t yet accept projection kwargs
                outT = triangle_gate.triangle_check(boundaries, cmap, H, H2)
                st.warning("Triangle running in STRICT path (no projection kwargs).")
                st.json(outT)

# Ensure tab4 is aligned with tab3
with tab4:
    st.subheader("Towers")
    sched_str = st.text_input("Schedule (comma-separated I/C)", "I,C,C,I,C")
    sched = [s.strip().upper() for s in sched_str.split(",") if s.strip()]
    if any(s not in ("I","C") for s in sched):
        st.error("Schedule must contain only I or C")
    else:
        if st.button("Run Tower & save CSV"):
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            csv_path = os.path.join(reports_dir, f"tower-hashes_{seed}_{len(sched)}steps.csv")
            towers.run_tower(sched, cmap, shapes, seed, csv_path, schedule_name="custom")
            st.success(f"Saved: {csv_path}")
            with open(csv_path, "r", encoding="utf-8") as f:
                st.download_button("Download CSV", f.read(), file_name=os.path.basename(csv_path), mime="text/csv")

















    
                
            
                           


# === Coverage sampling — health ping & rollup helpers ===
import math, csv, statistics
from pathlib import Path as _Path

def _bit_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0: return 0.0
    q = 1.0 - p
    return -(p*math.log2(p) + q*math.log2(q))

def _read_jsonl_tail(path: _Path, N: int = 1000):
    try:
        if not path.exists(): return []
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = []
        for ln in lines[-N:]:
            ln = ln.strip()
            if not ln: continue
            try:
                rows.append(_json.loads(ln))
            except Exception:
                pass
        return rows
    except Exception:
        return []

def _coverage_health_ping() -> dict:
    """Best-effort: prefer JSONL if present; otherwise summarize CSV."""
    jsonl = REPORTS_DIR / "coverage.jsonl"
    csvp  = REPORTS_DIR / "coverage_sampling.csv"
    out = {"ok": False}

    if jsonl.exists():
        rows = _read_jsonl_tail(jsonl, N=2000)
        if not rows:
            return {"ok": False, "reason": "NO_JSONL_ROWS"}
        lane_vecs = []
        verdict_pairs = []
        for r in rows:
            pol = r.get("policy") or {}
            lm = pol.get("lanes") or (r.get("overlay") or {}).get("lanes")
            if isinstance(lm, list) and lm:
                lane_vecs.append([1 if (int(x) & 1) else 0 for x in lm])
            k = r.get("checks") or {}
            strict_k3 = k.get("strict_k3")
            proj_k3   = k.get("projected_k3")
            if isinstance(strict_k3, bool) and isinstance(proj_k3, bool):
                verdict_pairs.append((strict_k3, proj_k3))

        n = max((len(v) for v in lane_vecs), default=0)
        if n > 0 and lane_vecs:
            L = [[(v[j] if j < len(v) else 0) for j in range(n)] for v in lane_vecs]
            col_means = [statistics.fmean(col) for col in zip(*L)]
            entropies = [_bit_entropy(p) for p in col_means]
            lane_density = statistics.fmean(col_means)
        else:
            entropies = []
            lane_density = 0.0

        posed = [(a,b) for (a,b) in verdict_pairs if isinstance(a,bool) and isinstance(b,bool)]
        contradictory_lane_rate = (sum(int(a!=b) for a,b in posed)/len(posed)) if posed else 0.0

        flags = []
        if entropies and max(entropies) < 0.05: flags.append("entropy≈0 (over-masked projector)")
        if contradictory_lane_rate > 0.10: flags.append(f"contradictions high ({contradictory_lane_rate:.2%})")

        out.update({
            "ok": True,
            "source": "jsonl",
            "lane_density": lane_density,
            "entropy_mean": (statistics.fmean(entropies) if entropies else 0.0),
            "entropy_min": (min(entropies) if entropies else 0.0),
            "entropy_max": (max(entropies) if entropies else 0.0),
            "contradictory_lane_rate": contradictory_lane_rate,
            "flags": flags,
            "rows": len(rows),
            "width": n,
        })
        return out

    if csvp.exists():
        try:
            rows = []
            with open(csvp, "r", encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                for row in r:
                    rows.append(row)
            total = len(rows)
            in_rows = sum(1 for r in rows if str(r.get("in_district","")).strip().upper()=="TRUE")
            out_rows = total - in_rows
            out.update({
                "ok": True,
                "source": "csv",
                "total": total,
                "in_rows": in_rows,
                "out_rows": out_rows,
                "coverage_rate": (in_rows/max(1,total)),
                "flags": [],
            })
            return out
        except Exception:
            return {"ok": False, "reason": "CSV_PARSE_FAIL"}

    return {"ok": False, "reason": "NO_SAMPLING_FILES"}

def _coverage_rollup_to_csv() -> dict:
    """Prefer JSONL rollup; if absent, rollup CSV by in_district and write summary."""
    jsonl = REPORTS_DIR / "coverage.jsonl"
    csvp  = REPORTS_DIR / "coverage_sampling.csv"
    outp  = REPORTS_DIR / "coverage_sampling_summary.csv"

    if jsonl.exists():
        rows = _read_jsonl_tail(jsonl, N=100000)
        if not rows:
            return {"ok": False, "reason": "NO_JSONL_ROWS"}
        from collections import defaultdict
        grp = defaultdict(lambda: {"IN":0,"NEAR_OUT":0,"FAR_OUT":0,"NA":0,"overlay":[]})
        for r in rows:
            pol = (r.get("policy") or r.get("policy_tag") or "")
            zone = (r.get("zone") or "").upper()
            if not zone:
                m = r.get("membership") or {}
                st = (m.get("status") or "").upper()
                if st == "IN": zone = "IN"
                elif st == "NA": zone = "NA"
                else: zone = "FAR_OUT"
            if zone not in grp[pol]: grp[pol][zone]=0
            grp[pol][zone]+=1
            ov = r.get("overlay_rate")
            if isinstance(ov,(int,float)): grp[pol]["overlay"].append(float(ov))

        lines = [["policy","IN","NEAR_OUT","FAR_OUT","NA","mean_overlay_rate"]]
        for pol, v in grp.items():
            ov = v.get("overlay") or []
            mean_ov = (sum(ov)/len(ov)) if ov else ""
            lines.append([pol, v.get("IN",0), v.get("NEAR_OUT",0), v.get("FAR_OUT",0), v.get("NA",0), mean_ov])
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh); w.writerows(lines)
        return {"ok": True, "out": str(outp), "mode":"jsonl"}

    if csvp.exists():
        try:
            rows = []
            with open(csvp, "r", encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                for row in r: rows.append(row)
            total = len(rows)
            in_rows = sum(1 for r in rows if str(r.get("in_district","")).strip().upper()=="TRUE")
            out_rows = total - in_rows
            lines = [["policy","IN","OUT","total","coverage_rate"],
                     ["strict", in_rows, out_rows, total, (in_rows/max(1,total))]]
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh); w.writerows(lines)
            return {"ok": True, "out": str(outp), "mode":"csv"}
        except Exception:
            return {"ok": False, "reason": "CSV_PARSE_FAIL"}

    return {"ok": False, "reason": "NO_SAMPLING_FILES"}

def _render_coverage_health_and_rollup():
    hp = _coverage_health_ping()
    if not hp.get("ok"):
        st.caption("Coverage sampling: no recent rows.")
    else:
        if hp.get("source") == "jsonl":
            st.code(
                "HEALTH  lane_density={:.3f}  H̄={:.3f}  Hmin={:.3f}  Hmax={:.3f}  contradictions={:.2%}  rows={}  k={}\n{}"
                .format(hp.get("lane_density",0.0), hp.get("entropy_mean",0.0),
                        hp.get("entropy_min",0.0), hp.get("entropy_max",0.0),
                        hp.get("contradictory_lane_rate",0.0),
                        hp.get("rows",0), hp.get("width",0),
                        ("⚠ " + " · ".join(hp.get("flags") or [])) if hp.get("flags") else "OK"),
                language="text",
            )
        else:
            st.code(
                "HEALTH  coverage_rate={:.3f}  rows={}".format(hp.get("coverage_rate",0.0), hp.get("total",0)),
                language="text",
            )
    res = _coverage_rollup_to_csv()
    if res.get("ok"):
        st.caption(f"CSV rollup → {res['out']}")
# === /Coverage sampling — health ping & rollup helpers ===


# =============================== C1 Hygiene & Telemetry (non-blocking, v2) ===============================
try:
    with st.expander("C1 Hygiene & Telemetry", expanded=False):
        _ensure_ssot_published()
        ss = st.session_state
        rc = dict(ss.get("run_ctx") or {})
        lane_mask_k3 = list(rc.get("lane_mask_k3") or ss.get("lane_mask_k3") or [])
        if not lane_mask_k3:
            lane_mask_k3 = _infer_lane_mask_from_session(ss, rc)
        residual_bottom_row = list(ss.get("residual_bottom_row_k3") or rc.get("residual_bottom_row_k3") or [])
        sigma_bits = list(ss.get("sigma_bits_k3") or rc.get("sigma_bits_k3") or [])
        if not sigma_bits:
            sig_str = ss.get("sigma_compact") or rc.get("sigma_compact")
            if isinstance(sig_str, str) and sig_str:
                if "|" in sig_str:
                    sigma_bits = [1 if t.strip() == "1" else 0 for t in sig_str.split("|")]
                else:
                    sigma_bits = [1 if ch == "1" else 0 for ch in sig_str]

        flip_rows = _c1_enumerate_flips(sigma_bits, lane_mask_k3)
        flips_csv_path = _write_c1_flips_csv(flip_rows)

        coverage_jsonl_path = str(REPORTS_DIR / "coverage.jsonl")
        rollup_csv_path, rollup_groups, _total = _rollup_coverage_jsonl_to_csv(coverage_jsonl_path, REPORTS_DIR / "coverage_rollup.csv")

        health = _compute_c1_health_ping(flips_csv_path, coverage_jsonl_path, residual_bottom_row, lane_mask_k3)
        _render_c1_health_chip(health, wait=(not sigma_bits and int(health.get('coverage_rows') or 0)==0))
except Exception as _e:
    try:
        st.info(f"C1 Hygiene telemetry not available: {type(_e).__name__}: {_e}")
    except Exception:
        pass
# =============================== /C1 Hygiene & Telemetry (non-blocking, v2) ===============================


# =============================== Debug — Paths & SSOT sanity ===============================
try:
    with st.expander("Debug — Paths & SSOT sanity", expanded=False):
        from pathlib import Path as _P
        ss = st.session_state
        _ensure_ssot_published()
        cov = REPORTS_DIR / "coverage.jsonl"
        rol = REPORTS_DIR / "coverage_rollup.csv"
        def _size(p): 
            try: return (_P(p).stat().st_size if _P(p).exists() else 0)
            except Exception: return 0
        st.write({
            "REPORTS_DIR": str(REPORTS_DIR),
            "coverage.jsonl": {"exists": cov.exists(), "size": _size(cov)},
            "coverage_rollup.csv": {"exists": rol.exists(), "size": _size(rol)},
            "last_bundle_dir": ss.get("last_bundle_dir",""),
            "ssot_hashes_present": list((ss.get("_inputs_block") or {}).get("hashes", {}).keys() or []),
            "dims": (ss.get("run_ctx") or {}).get("dims", {}),
            "run_id": (ss.get("run_ctx") or {}).get("run_id", ""),
        })
        # quick peek at bundle filenames if present
        bdir = _P(ss.get("last_bundle_dir") or "")
        if bdir and bdir.exists():
            st.caption("Bundle files:")
            st.write(sorted([p.name for p in bdir.iterdir() if p.is_file()]))
except Exception as _dbg_e:
    try: st.info(f"Paths sanity panel unavailable: {type(_dbg_e).__name__}: {_dbg_e}")
    except Exception: pass
# =============================== /Debug — Paths & SSOT sanity ===============================
