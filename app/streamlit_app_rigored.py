# ────────────────────────────── IMPORTS (top) ──────────────────────────────
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


import pandas as pd
import streamlit as st

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
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")


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


# ───────────────────── DISTRICT MAP (optional) ─────────────────────
DISTRICT_MAP: dict[str, str] = {
    "9da8b7f605c113ee059160cdaf9f93fe77e181476c72e37eadb502e7e7ef9701": "D1",
    "4356e6b608443b315d7abc50872ed97a9e2c837ac8b85879394495e64ec71521": "D2",
    "28f8db2a822cb765e841a35c2850a745c667f4228e782d0cfdbcb710fd4fecb9": "D3",
    "aea6404ae680465c539dc4ba16e97fbd5cf95bae5ad1c067dc0f5d38ca1437b5": "D4",
}


# ───────── A/B helpers (inputs sig + projector hashers) ─────────
def _inputs_sig_now() -> list[str]:
    ib = st.session_state.get("_inputs_block") or {}
    return [
        str(ib.get("boundaries_hash","")),
        str(ib.get("C_hash","")),
        str(ib.get("H_hash","")),
        str(ib.get("U_hash","")),
        str(ib.get("shapes_hash","")),
    ]

def _hash_proj_matrix(P: list[list[int]]) -> str:
    # stable hash of Π over JSON canonical form
    return hash_json({"blocks":{"3": P}})

def _ab_is_fresh(ab: dict, *, rc: dict) -> bool:
    if not ab: return False
    # inputs sig match?
    if ab.get("inputs_sig") != _inputs_sig_now():
        return False
    # policy tag must match currently active
    if (ab.get("projected") or {}).get("policy_tag") != (rc.get("policy_tag") or ""):
        return False
    # if FILE, projector hash must also match
    if (rc.get("mode") == "projected(file)"):
        return (ab.get("projected") or {}).get("projector_hash","") == (rc.get("projector_hash","") or "")
    return True



# ───────── auto-clear A/B when context changes (call once per run) ─────────
def _ab_autoclear_if_stale():
    ss = st.session_state
    rc = ss.get("run_ctx") or {}
    sig_now = tuple(_inputs_sig_now() + [rc.get("policy_tag",""), rc.get("projector_hash","")])
    prev = ss.get("_ab_ctx_sig_prev")
    if prev is None:
        ss["_ab_ctx_sig_prev"] = sig_now
        return
    if prev != sig_now:
        ss.pop("ab_compare", None)  # drop ghost snapshot
        ss["_ab_ctx_sig_prev"] = sig_now

_ab_autoclear_if_stale()





# ======================= App constants & helpers =======================
SCHEMA_VERSION = "1.0.0"
FIELD = "GF(2)"  # displayed only; math stays the same

def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _py_version_str() -> str:
    return f"python-{platform.python_version()}"

def _short(h: str, n: int = 8) -> str:
    return (h or "")[:n]

def _std_meta(*, include_python: bool = False, run_id: str | None = None) -> dict:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "written_at_utc": _utc_iso_z(),
        "app_version": APP_VERSION,
    }
    if include_python:
        meta["python_version"] = _py_version_str()
    if run_id:
        meta["run_id"] = run_id
    return meta

def _inputs_sig_now() -> list[str]:
    """Deterministic ‘freshness’ signature for A/B embedding (SSOT)."""
    ib = st.session_state.get("_inputs_block") or {}
    # Always include shapes_hash; fall back to U_hash for older runs.
    return [
        str(ib.get("boundaries_hash", "")),
        str(ib.get("C_hash", "")),
        str(ib.get("H_hash", "")),
        str(ib.get("U_hash", "")),
        str(ib.get("shapes_hash", ib.get("U_hash", ""))),
    ]



# ====================== Policy label symmetry (full tag) ======================
def policy_label_from_cfg_full(cfg: dict) -> str:
    """
    Returns a canonical, full policy tag:
      - "strict"
      - "projected(columns@k=3,auto)"
      - "projected(columns@k=3,file)"
    """
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src3 = (cfg.get("source") or {}).get("3", "auto")
    if src3 == "file":
        return "projected(columns@k=3,file)"
    return "projected(columns@k=3,auto)"

def policy_label_from_state(rc: dict, cfg_active: dict) -> str:
    """
    Mirrors the active policy tag using run_ctx when available; falls back to cfg.
    """
    mode = (rc or {}).get("mode", "")
    if mode == "strict":
        return "strict"
    if mode == "projected(file)":
        return "projected(columns@k=3,file)"
    if mode == "projected(auto)":
        return "projected(columns@k=3,auto)"
    # fallback
    return policy_label_from_cfg_full(cfg_active or {})

# --- compat shim for legacy call sites ---------------------------------
if "_rectify_run_ctx_mask_from_d3_or_stop" not in globals():
    def _rectify_run_ctx_mask_from_d3_or_stop():
        """Require fresh run_ctx and rectify mask; hard-stop if stale."""
        try:
            rc = require_fresh_run_ctx()        # raises via st.stop() on stale
            rc = rectify_run_ctx_mask_from_d3() # may also st.stop() on mismatch
            return rc
        except Exception as e:
            # If your rectifiers switched to soft warnings elsewhere,
            # keep this call site hard-guarded for Freezer correctness.
            st.warning(str(e))
            st.stop()

# UTC ISO helper (if not already present)
if "_utc_iso" not in globals():
    from datetime import datetime, timezone
    def _utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def _file_mode_invalid_now() -> bool:
    """True only when we're in projected(FILE) mode *and* the FILE Π fails validation."""
    rc = st.session_state.get("run_ctx") or {}
    return (str(rc.get("mode")) == "projected(file)") and bool(file_validation_failed())




# =========================[ STEP 1 · Core helpers + guards ]=========================
from pathlib import Path
from contextlib import contextmanager

# ---- Directories (be tolerant if not pre-defined)
LOGS_DIR = Path(globals().get("LOGS_DIR", "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Inputs signature (canonical, de-duplicated) ---
def current_inputs_sig() -> list[str]:
    ib = st.session_state.get("_inputs_block") or {}
    return [
        str(ib.get("boundaries_hash", "")),
        str(ib.get("C_hash", "")),
        str(ib.get("H_hash", "")),
        str(ib.get("U_hash", "")),
        str(ib.get("shapes_hash", "")),
    ]

# Back-compat alias for older call sites
_current_inputs_sig = current_inputs_sig


# ========================= Widget key utilities (NEW) =========================
def _mkkey(ns: str, name: str) -> str:
    """Deterministic, readable widget key: '<ns>__<name>'."""
    return f"{ns}__{name}"

def ensure_unique_widget_key(key: str) -> str:
    """
    If a widget key was already used in this run, suffix it with __2/__3/…
    Use this when you cannot easily rename at call site.
    """
    ss = st.session_state
    used = ss.setdefault("_used_widget_keys", set())
    if key not in used:
        used.add(key)
        return key
    # bump suffix until free
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

# Central registry for common keys
class _WKey:
    shapes_up      = _mkkey("inputs", "shapes_uploader")
    shapes_up_alt  = _mkkey("inputsB", "shapes_uploader")
WKEY = _WKey()

# ---- Tiny time/uuid utils
def new_run_id() -> str:
    return str(uuid4())

# ---- Fixture nonce: single source of "freshness"
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

# ---- Freshness guard (use at start of every action)
def require_fresh_run_ctx():
    _ensure_fixture_nonce()
    ss = st.session_state
    rc = ss.get("run_ctx")
    if not rc:
        st.warning("STALE_RUN_CTX: Run Overlap first.")
        st.stop()
    if int(rc.get("fixture_nonce", -1)) != int(ss.get("fixture_nonce", -2)):
        st.warning("STALE_RUN_CTX: Inputs changed; please click Run Overlap to refresh.")
        st.stop()
    # assert mask shape sanity
    n3 = int(rc.get("n3") or 0)
    lm = list(rc.get("lane_mask_k3") or [])
    if lm and n3 and len(lm) != n3:
        st.warning("Context mask length mismatch; please click Run Overlap to refresh.")
        st.stop()
    return rc

# ---- Truth mask from stored d3 (GF(2) column-wise OR)
def _truth_mask_from_d3(d3: list[list[int]]) -> list[int]:
    if not d3 or not d3[0]:
        return []
    rows, cols = len(d3), len(d3[0])
    return [1 if any(int(d3[i][j]) & 1 for i in range(rows)) else 0 for j in range(cols)]

# ---- Rectify run_ctx mask from d3
def rectify_run_ctx_mask_from_d3():
    ss = st.session_state
    rc = require_fresh_run_ctx()
    d3 = rc.get("d3") or []
    n3 = int(rc.get("n3") or 0)
    if not d3 or n3 <= 0:
        st.warning("STALE_RUN_CTX: d3/n3 unavailable. Run Overlap.")
        st.stop()
    lm_truth = _truth_mask_from_d3(d3)
    if len(lm_truth) != n3:
        st.warning(f"STALE_RUN_CTX: lane mask length {len(lm_truth)} != n3 {n3}. Run Overlap.")
        st.stop()
    lm_rc = list(rc.get("lane_mask_k3") or [])
    if lm_rc != lm_truth:
        rc["lane_mask_k3"] = lm_truth
        ss["run_ctx"] = rc
        st.info(f"Rectified run_ctx.lane_mask_k3 from {lm_rc or '[]'} → {lm_truth} based on stored d3.")
    return ss["run_ctx"]

# ---- Soft reset: clears per-run caches
def soft_reset_before_overlap():
    ss = st.session_state
    for k in (
        "run_ctx", "overlap_out", "overlap_cfg", "overlap_policy_label",
        "overlap_H", "residual_tags", "proj_meta", "ab_compare",
        "cert_payload", "last_cert_path", "_last_cert_write_key",
        "_projector_cache", "_projector_cache_ab",
    ):
        ss.pop(k, None)

# ---- JSONL helpers
def _atomic_append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(row, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        tmp.write(blob)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    with open(path, "a", encoding="utf-8") as final, open(tmp_name, "r", encoding="utf-8") as src:
        shutil.copyfileobj(src, final)
    os.remove(tmp_name)

def _read_jsonl_tail(path: Path, N: int = 200) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 64 * 1024
            data = b""
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
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out

# ---- Predicates used across UI
def is_projected_green(run_ctx: dict | None, overlap_out: dict | None) -> bool:
    if not run_ctx or not overlap_out:
        return False
    mode = str(run_ctx.get("mode") or "")
    return mode.startswith("projected") and bool(((overlap_out.get("3") or {}).get("eq", False)))

def is_strict_red_lanes(run_ctx: dict | None, overlap_out: dict | None, residual_tags: dict | None) -> bool:
    if not run_ctx or not overlap_out:
        return False
    if str(run_ctx.get("mode") or "") != "strict":
        return False
    if bool(((overlap_out.get("3") or {}).get("eq", True))):
        return False
    tag = ((residual_tags or {}).get("strict") or "")
    return tag == "lanes"


# ---- Hash key builders for deduplication
def gallery_key(row: dict) -> tuple:
    pol = row.get("policy") or {}
    h = row.get("hashes") or {}
    return (
        row.get("district", ""),
        pol.get("policy_tag", ""),
        h.get("boundaries_hash", ""),
        h.get("C_hash", ""),
        h.get("H_hash", ""),
        h.get("U_hash", ""),
    )

def witness_key(row: dict) -> tuple:
    pol = row.get("policy") or {}
    h = row.get("hashes") or {}
    return (
        row.get("district", ""),
        row.get("reason", ""),
        row.get("residual_tag", ""),
        pol.get("policy_tag", ""),
        h.get("boundaries_hash", ""),
        h.get("C_hash", ""),
        h.get("H_hash", ""),
        h.get("U_hash", ""),
    )

# Session-level deduplication caches
if "_gallery_keys" not in st.session_state:
    st.session_state["_gallery_keys"] = set()
if "_witness_keys" not in st.session_state:
    st.session_state["_witness_keys"] = set()

# ---- Run stamp helper
def run_stamp_line() -> str:
    ss = st.session_state
    rc = ss.get("run_ctx") or {}
    ib = ss.get("_inputs_block") or {}
    pol = rc.get("policy_tag", "?")
    n3 = int(rc.get("n3") or 0)
    hB = (ib.get("boundaries_hash", "") or "")[:8]
    hC = (ib.get("C_hash", "") or "")[:8]
    hH = (ib.get("H_hash", "") or "")[:8]
    hU = (ib.get("U_hash", "") or "")[:8]
    pH = (rc.get("projector_hash", "") or "")[:8]
    rid = (rc.get("run_id", "") or "")[:8]
    return f"{pol} | n3={n3} | B {hB} · C {hC} · H {hH} · U {hU} | P {pH} | run {rid}"


# ───────────────────────── SSOT + Freshness helpers (aliases) ─────────────────────────
_mark_fixtures_changed = _bump_fixture_nonce  # legacy
_soft_reset_before_overlap = soft_reset_before_overlap  # legacy


# ---------- stable hashing ----------
def _sha256_hex_bytes(b: bytes) -> str:
    return _hashlib.sha256(b).hexdigest()

def _sha256_hex_obj(obj) -> str:
    blob = _json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_hex_bytes(blob)

def hash_json(obj) -> str:
    return _sha256_hex_obj(obj)

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ====================== FILE Validator (strict) ======================
def _mul_gf2(A, B):
    # Use app's mul if present; otherwise a safe GF(2) fallback
    if "mul" in globals() and callable(globals()["mul"]):
        return mul(A, B)
    if not A or not B:
        return []
    m, k = len(A), len(A[0])
    k2, c = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"dim mismatch: {m}x{k} @ {k2}x{c}")
    out = [[0] * c for _ in range(m)]
    for i in range(m):
        for t in range(k):
            if A[i][t] & 1:
                bt = B[t]
                for j in range(c):
                    out[i][j] ^= (bt[j] & 1)
    return out

def validate_projector_file_strict(P, *, n3: int, lane_mask: list[int]) -> None:
    # shape validation
    if not (isinstance(P, list) and all(isinstance(r, list) for r in P)):
        raise ValueError("P3_SHAPE: projector is not a 2D list")
    if len(P) != n3 or any(len(r) != n3 for r in P):
        got_r = len(P) if isinstance(P, list) else 0
        got_c = len(P[0]) if (isinstance(P, list) and P and isinstance(P[0], list)) else 0
        raise ValueError(f"P3_SHAPE: expected {n3}x{n3}, got {got_r}x{got_c}")
    # idempotence over GF(2)
    PP = _mul_gf2(P, P)
    if PP != P:
        raise ValueError("P3_IDEMP: P@P != P (GF2)")
    # diagonal-only
    for i in range(n3):
        for j in range(n3):
            if i != j and (P[i][j] & 1):
                raise ValueError("P3_DIAGONAL: off-diagonal element is 1")
    # lane diag match
    diag = [int(P[i][i]) & 1 for i in range(n3)]
    lm = [int(x) & 1 for x in (lane_mask or [])]
    if diag != lm:
        raise ValueError(f"P3_LANE_MISMATCH: diag(P)={diag} vs lane_mask(d3)={lm}")

# ---------- safe expander (never nests real expanders) ----------
try:
    from streamlit.errors import StreamlitAPIException  # type: ignore
except Exception:
    class StreamlitAPIException(Exception):  # type: ignore
        pass

@contextmanager
def safe_expander(title: str, **kwargs):
    """
    Drop-in replacement for st.expander that never raises the
    'Expanders may not be nested' error. If the real expander fails,
    fallback to a container.
    """
    def _container_fallback():
        st.caption(f"⚠️ Nested section: **{title}** (container fallback)")
        st.markdown(f"**{title}**")
        return st.container()

    try:
        with st.expander(title, **kwargs):
            yield
    except StreamlitAPIException:
        with _container_fallback():
            yield

# ---------- file IO helpers ----------
def read_json_file(upload):
    """
    Accepts UploadedFile | str | os.PathLike | Path | dict → dict|None
    """
    if upload is None:
        return None
    if isinstance(upload, dict):
        return upload
    if isinstance(upload, (str, os.PathLike, Path)):
        with open(str(upload), "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None

def _stamp_filename(state_key: str, upload):
    """Record uploaded filename into session state."""
    try:
        if upload is not None and hasattr(upload, "name"):
            st.session_state[state_key] = str(upload.name)
        elif upload is None:
            st.session_state.pop(state_key, None)
    except Exception:
        pass

# ---------- atomic writers ----------
def _atomic_write_json(path: str | Path, obj: dict, *, pretty: bool = False):
    """Canonical JSON atomic writer."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_write_json(path: str | Path, obj: dict, *, pretty: bool = False):
    """Alias for _atomic_write_json."""
    _atomic_write_json(path, obj, pretty=pretty)

def atomic_append_jsonl(path: str | Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with open(tmp, "wb") as f:
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    with open(path, "ab") as out, open(tmp, "rb") as src:
        out.write(src.read())
        out.flush()
        os.fsync(out.fileno())
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

def _atomic_write_csv(path: Path, header: list[str], rows: list[list], meta_comment_lines: list[str] | None = None):
    tmp = Path(str(path) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if meta_comment_lines:
            for k in meta_comment_lines:
                w.writerow([f"# {k}"])
        w.writerow(header)
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# ---------- lane mask / signatures ----------
def _lane_mask_from_d3(boundaries) -> list[int]:
    """
    k=3 mask: boundaries.lane_mask_k3 → dict field → bottom-row(d3) → []
    """
    try:
        if hasattr(boundaries, "lane_mask_k3"):
            lm = getattr(boundaries, "lane_mask_k3")
            if isinstance(lm, list) and all(isinstance(x, (int, bool)) for x in lm):
                return [int(bool(x)) for x in lm]
        bd = boundaries.dict() if hasattr(boundaries, "dict") else {}
        lm = (bd or {}).get("lane_mask_k3")
        if isinstance(lm, list) and all(isinstance(x, (int, bool)) for x in lm):
            return [int(bool(x)) for x in lm]
    except Exception:
        pass
    try:
        d3 = (boundaries.blocks.__root__.get("3") or [])
        if d3 and d3[-1]:
            return [int(x & 1) for x in d3[-1]]
    except Exception:
        pass
    return []

def _district_signature(mask: list[int], r: int, c: int) -> str:
    payload = f"k3:{''.join(str(int(x)) for x in (mask or []))}|r{r}|c{c}".encode()
    return _hashlib.sha256(payload).hexdigest()[:12]

# ---------- cfg builders + labels ----------
def cfg_strict() -> dict:
    return {"enabled_layers": [], "modes": {}, "source": {}, "projector_files": {}}

def cfg_projected_base() -> dict:
    return {"enabled_layers": [3], "modes": {}, "source": {"3": "auto"}, "projector_files": {}}

def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src = (cfg.get("source", {}) or {}).get("3", "auto")
    return "projected(columns@k=3,file)" if src == "file" else "projected(columns@k=3,auto)"

# ---------- tiny GF(2) ops ----------
def _eye(n: int):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def _mul_gf2_basic(A, B):
    if not A or not B or not A[0] or not B[0]:
        return []
    r, k = len(A), len(A[0])
    k2, c = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"dim mismatch: {r}x{k} @ {k2}x{c}")
    out = [[0] * c for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for t in range(k):
            if Ai[t] & 1:
                Bt = B[t]
                for j in range(c):
                    out[i][j] ^= (Bt[j] & 1)
    return out

def _is_idempotent_gf2(P):
    try:
        return _mul_gf2_basic(P, P) == P
    except Exception:
        return False

def _is_diagonal(P):
    m = len(P) or 0
    n = len(P[0]) if m else 0
    if m != n:
        return False
    for i in range(m):
        for j in range(n):
            if i != j and (P[i][j] & 1):
                return False
    return True

def _diag(P):
    return [int(P[i][i] & 1) for i in range(len(P))] if P and P[0] else []

# ---------- projector chooser (strict/auto/file) ----------
class _P3Error(ValueError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code

def _read_projector_matrix(path_str: str):
    p = _Path(path_str)
    if not p.exists():
        raise _P3Error("P3_SHAPE", f"projector file not found: {path_str}")
    with open(p, "r", encoding="utf-8") as f:
        d = _json.load(f)
    if isinstance(d, dict):
        b = (d.get("blocks", {}) or {}).get("3")
        if isinstance(b, list):
            return b
    if isinstance(d, list):
        return d
    raise _P3Error("P3_SHAPE", "unrecognized projector JSON structure")

def projector_choose_active(cfg_active: dict, boundaries):
    # Resolve the exact d3 we’ll use and compute the SSOT mask from it.
    d3 = (boundaries.blocks.__root__.get("3") or [])
    d3 = [[int(x) & 1 for x in row] for row in d3] if d3 else []
    n3 = len(d3[0]) if (d3 and d3[0]) else 0
    lm_truth = _truth_mask_from_d3(d3)

    mode = "strict"
    P_active = []
    pj_filename = ""
    pj_hash = ""
    pj_consistent = None

    # No projected layer → strict
    if not cfg_active or not cfg_active.get("enabled_layers"):
        return P_active, {
            "d3": d3,
            "n3": n3,
            "mode": mode,
            "lane_mask": lm_truth,
            "lane_mask_k3": lm_truth,
            "projector_filename": pj_filename,
            "projector_hash": pj_hash,
            "projector_consistent_with_d": pj_consistent,
        }

    source = (cfg_active.get("source", {}) or {}).get("3", "auto")
    mode = "projected(auto)" if source == "auto" else "projected(file)"

    if source == "auto":
        # AUTO projector is always the diagonal of the SSOT lane mask
        diag = lm_truth if lm_truth else [1] * n3
        P_active = [[1 if i == j and diag[j] else 0 for j in range(n3)] for i in range(n3)]
        pj_hash = _sha256_hex_obj(P_active)
        pj_consistent = True
        return P_active, {
            "d3": d3,
            "n3": n3,
            "mode": mode,
            "lane_mask": lm_truth,
            "lane_mask_k3": lm_truth,
            "projector_filename": "",
            "projector_hash": pj_hash,
            "projector_consistent_with_d": pj_consistent,
        }

    # FILE mode
    pj_filename = (cfg_active.get("projector_files", {}) or {}).get("3", "")
    if not pj_filename:
        raise _P3Error("P3_SHAPE", "no projector file provided for file-mode")

    P = _read_projector_matrix(pj_filename)
    m = len(P) or 0
    n = len(P[0]) if m else 0
    if n3 == 0 or m != n3 or n != n3:
        raise _P3Error("P3_SHAPE", f"expected {n3}x{n3}, got {m}x{n}")
    P = [[int(x) & 1 for x in row] for row in P]

    if not _is_idempotent_gf2(P):
        raise _P3Error("P3_IDEMP", "P is not idempotent over GF(2)")
    if not _is_diagonal(P):
        raise _P3Error("P3_DIAGONAL", "P has off-diagonal non-zeros")

    pj_diag = _diag(P)  # must be 0/1 ints
    if pj_diag != [int(x) for x in lm_truth]:
        raise _P3Error("P3_LANE_MISMATCH", f"diag(P) != lane_mask(d3) → {pj_diag} vs {lm_truth}")

    pj_hash = _sha256_hex_obj(P)
    pj_consistent = True  # validated against SSOT mask

    return P, {
        "d3": d3,
        "n3": n3,
        "mode": mode,
        "lane_mask": lm_truth,
        "lane_mask_k3": lm_truth,
        "projector_filename": pj_filename,
        "projector_hash": pj_hash,
        "projector_consistent_with_d": pj_consistent,
    }

# ---------- misc ----------
def hash_matrix_norm(M) -> str:
    if not M:
        return hash_json([])
    norm = [[int(x) & 1 for x in row] for row in M]
    return hash_json(norm)

def _zip_arcname(abspath: str) -> str:
    p = _Path(abspath)
    try:
        return p.resolve().relative_to(_Path.cwd().resolve()).as_posix()
    except Exception:
        return p.name

def build_cert_bundle(*, district_id: str, policy_tag: str, cert_path: str,
                      content_hash: str | None = None, extras: list[str] | None = None) -> str:
    cert_p = _Path(cert_path)
    if not cert_p.exists():
        raise FileNotFoundError(f"Cert not found: {cert_path}")
    with open(cert_p, "r", encoding="utf-8") as f:
        cert = _json.load(f)
    if not content_hash:
        content_hash = ((cert.get("integrity") or {}).get("content_hash") or "")
    suffix = content_hash[:12] if content_hash else "nohash"
    safe_policy = (policy_tag or cert.get("policy", {}).get("policy_tag", "policy")).replace("/", "_").replace(" ", "_")
    zpath = BUNDLES_DIR / f"overlap_bundle__{district_id or 'UNKNOWN'}__{safe_policy}__{suffix}.zip"
    files = [str(cert_p)]
    for p in (extras or []):
        if p and _os.path.exists(p):
            files.append(p)
    fd, tmp_name = _tempfile.mkstemp(dir=BUNDLES_DIR, prefix=".tmp_bundle_", suffix=".zip")
    _os.close(fd)
    tmp_path = _Path(tmp_name)
    try:
        with _zipfile.ZipFile(tmp_path, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
            for abspath in files:
                abspath = str(_Path(abspath).resolve())
                zf.write(abspath, arcname=_zip_arcname(abspath))
        try:
            _os.replace(tmp_path, zpath)
        except OSError:
            _shutil.move(str(tmp_path), str(zpath))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return str(zpath)

# ---------- inputs block builder (SSOT) ----------
def build_inputs_block(boundaries, cmap, H_used, shapes, filenames: dict) -> dict:
    C3 = (cmap.blocks.__root__.get("3") or [])
    d3 = (boundaries.blocks.__root__.get("3") or [])
    dims = {
        "n3": len(C3) if C3 else (len(d3[0]) if (d3 and d3[0]) else 0),
        "n2": len(cmap.blocks.__root__.get("2") or []),
    }
    hashes_dict = {
        "boundaries_hash": hash_json(boundaries.dict() if hasattr(boundaries, "dict") else {}),
        "C_hash": hash_json(cmap.dict() if hasattr(cmap, "dict") else {}),
        "H_hash": hash_json(H_used.dict() if hasattr(H_used, "dict") else {}),
        "U_hash": hash_json(shapes.dict() if hasattr(shapes, "dict") else {}),
    }
    return {
        "filenames": filenames,
        "dims": dims,
        **hashes_dict,
        "shapes_hash": hashes_dict["U_hash"],
    }



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
            return A
        return [[(A[i][j] ^ B[i][j]) for j in range(c)] for i in range(r)]

    def eye(n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

# ------------------------------ OVERLAP TAB -----------------------------------

# Utility functions (shared)
def _xor_mat(A, B):
    if "add" in globals() and callable(globals()["add"]):
        return globals()["add"](A, B)
    if not A: return [r[:] for r in (B or [])]
    if not B: return [r[:] for r in (A or [])]
    r, c = len(A), len(A[0])
    return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]

def _bottom_row(M):
    return M[-1] if M and len(M) else []

def _stable_hash(obj):
    return hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _load_h_local():
    try:
        if f_H is None:
            return io.parse_cmap({"blocks": {}})
        return io.parse_cmap(read_json_file(f_H))
    except Exception:
        return io.parse_cmap({"blocks": {}})

def _lane_mask_from_d3_strict(boundaries_obj):
    try:
        d3 = boundaries_obj.blocks.__root__.get("3") or []
    except Exception:
        d3 = []
    if not d3 or not d3[0]:
        return []
    rows, cols = len(d3), len(d3[0])
    return [1 if any((d3[i][j] & 1) for i in range(rows)) else 0 for j in range(cols)]

def _lane_mask_from_d3_local(boundaries_obj):
    return _lane_mask_from_d3_strict(boundaries_obj)

def _derive_mode_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src = (cfg.get("source", {}) or {}).get("3", "auto")
    return "projected(file)" if src == "file" else "projected(auto)"

# Projected(FILE) validation banner (single source)
def file_validation_failed() -> bool:
    """Convenience predicate: returns True if last attempt to use FILE Π failed validation."""
    return bool(st.session_state.get("_file_mode_error"))

_file_err = st.session_state.get("_file_mode_error")
if _file_err:
    code = str(_file_err.get("code", "P3_ERROR"))
    msg  = str(_file_err.get("message", "Projected(FILE) validation failed."))
    pj   = _file_err.get("projector_filename", "")
    st.error(
        f"Projected(FILE) validation failed [{code}]. {msg}"
        + (f" · file: {pj}" if pj else "")
        + " — Freeze from AUTO or open the projector registry to fix.",
        icon="🚫"
    )

# ------------------------------ UI: policy + H + projector ------------------------------
colA, colB = st.columns([2, 2])
with colA:
    policy_choice = st.radio(
        "Policy",
        ["strict", "projected(auto)", "projected(file)"],
        index=0,
        horizontal=True,
        key="ov_policy_choice",
    )
with colB:
    f_H = st.file_uploader("Homotopy H (optional)", type=["json"], key="H_up")

proj_upload = st.file_uploader(
    "Projector Π (k=3) file (only for projected(file))",
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

# ------------------------------ Run Overlap (SSOT + freshness) ------------------------------
def _truth_mask_from_d3(d3: list[list[int]]) -> list[int]:
    """Column-wise OR over GF(2); returns length n3 mask with values in {0,1}."""
    if not d3 or not d3[0]:
        return []
    rows, n3 = len(d3), len(d3[0])
    return [1 if any(d3[i][j] & 1 for i in range(rows)) else 0 for j in range(n3)]

def run_overlap():
    # Ensure a fixture nonce exists
    _ensure_fixture_nonce()

    # Clear previous artifacts for a clean run
    for k in ("proj_meta","run_ctx","residual_tags","overlap_out",
              "overlap_H","overlap_cfg","overlap_policy_label"):
        st.session_state.pop(k, None)

    # --- Bind projector (fail-fast on FILE) ---
    try:
        # clear any previous FILE error before attempting to resolve Π
        st.session_state.pop("_file_mode_error", None)
        P_active, meta = projector_choose_active(cfg_active, boundaries)
    except ValueError as e:
        code = getattr(e, "code", None) or "P3_ERROR"
        pjfn = (cfg_active.get("projector_files", {}) or {}).get("3", "")

        st.session_state["_file_mode_error"] = {
            "code": code,
            "message": str(e),
            "projector_filename": pjfn,
        }

        st.error(str(e))
        d3_now = (boundaries.blocks.__root__.get("3") or [])

        _policy_label = (policy_label_from_state({"mode": "projected(file)"}, cfg_active)
                         if "policy_label_from_state" in globals()
                         else policy_label_from_cfg(cfg_active))

        st.session_state["run_ctx"] = {
            "policy_tag": _policy_label,
            "mode": "projected(file)",
            "fixture_nonce": st.session_state.get("_fixture_nonce", 0),
            "d3": d3_now,
            "n3": len(d3_now[0]) if (d3_now and d3_now[0]) else 0,
            "lane_mask_k3": [],
            "P_active": [],
            "projector_filename": pjfn,
            "projector_hash": "",
            "projector_consistent_with_d": False,
            "source": (cfg_active.get("source") or {}),
            "errors": [str(e)],
        }
        st.stop()

    # Context from projector resolver (authoritative d3/n3/mode)
    d3   = meta.get("d3") if "d3" in meta else (boundaries.blocks.__root__.get("3") or [])
    n3   = meta.get("n3") if "n3" in meta else (len(d3[0]) if (d3 and d3[0]) else 0)
    mode = meta.get("mode", _derive_mode_from_cfg(cfg_active))

    # SSOT lane mask from THIS d3
    lm_truth = _truth_mask_from_d3(d3)
    assert len(lm_truth) == n3, f"lane_mask_k3 length {len(lm_truth)} != n3 {n3}"

    # --- NEW: generate a per-run id and keep it in session
    run_id = str(uuid4())
    st.session_state["last_run_id"] = run_id  # optional convenience

    # Strict residuals
    H_local = _load_h_local()
    H2 = (H_local.blocks.__root__.get("2") or [])
    C3 = (cmap.blocks.__root__.get("3") or [])
    I3 = eye(len(C3)) if C3 else []
    try:
        R3_strict = _xor_mat(mul(H2, d3), _xor_mat(C3, I3)) if (H2 and d3 and C3) else []
    except Exception as e:
        st.error(f"Shape guard failed at k=3: {e}")
        st.stop()

    def _is_zero(M):
        return (not M) or all(all((x & 1) == 0 for x in row) for row in M)

    def _residual_tag(R, lm):
        if not R or not lm: return "none"
        rows = len(R)
        lanes_idx = [j for j, m in enumerate(lm) if m]
        ker_idx   = [j for j, m in enumerate(lm) if not m]
        def _col_nonzero(j): return any(R[i][j] & 1 for i in range(rows))
        lanes_resid = any(_col_nonzero(j) for j in lanes_idx) if lanes_idx else False
        ker_resid   = any(_col_nonzero(j) for j in ker_idx)   if ker_idx   else False
        if not lanes_resid and not ker_resid: return "none"
        if lanes_resid and not ker_resid:     return "lanes"
        if ker_resid and not lanes_resid:     return "ker"
        return "mixed"

    tag_strict = _residual_tag(R3_strict, lm_truth)
    eq3_strict = _is_zero(R3_strict)

    # Projected leg (when enabled)
    if cfg_active.get("enabled_layers"):
        R3_proj  = mul(R3_strict, P_active) if (R3_strict and P_active) else []
        eq3_proj = _is_zero(R3_proj)
        tag_proj = _residual_tag(R3_proj, lm_truth)
        out = {"3": {"eq": bool(eq3_proj), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict, "projected": tag_proj}
    else:
        out = {"3": {"eq": bool(eq3_strict), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict}

    # 6) Persist SSOT (write ONCE)
    _policy_label = (policy_label_from_state({"mode": mode}, cfg_active)
                     if "policy_label_from_state" in globals()
                     else policy_label_from_cfg(cfg_active))

    st.session_state["overlap_out"] = out
    st.session_state["overlap_cfg"] = cfg_active
    st.session_state["overlap_policy_label"] = _policy_label
    st.session_state["overlap_H"] = H_local
    st.session_state["run_ctx"] = {
        "policy_tag": _policy_label,
        "mode": mode,
        "fixture_nonce": st.session_state.get("_fixture_nonce", 0),
        "d3": d3, "n3": n3, "lane_mask_k3": lm_truth,
        "P_active": P_active,
        "projector_filename": meta.get("projector_filename", ""),
        "projector_hash": meta.get("projector_hash", ""),
        "projector_consistent_with_d": meta.get("projector_consistent_with_d", None),
        "source": (cfg_active.get("source") or {}),
        "run_id": run_id,                         # NEW
        "errors": [],
    }

    # 7) Debug line + run-stamp banner  —— NEW
    ib = st.session_state.get("_inputs_block") or {}
    B = ib.get("boundaries_hash",""); C = ib.get("C_hash",""); Hh = ib.get("H_hash",""); U = ib.get("U_hash","")
    P_hash = meta.get("projector_hash","") if mode.startswith("projected") else ""
    st.caption(
        f"run ⟂ { _policy_label } | n3={n3} | "
        f"B:{_short(B)} C:{_short(C)} H:{_short(Hh)} U:{_short(U)} | "
        f"P:{_short(P_hash)} | id:{_short(run_id)}"
    )

    # Minimal debug & status
    st.json(out)
    if mode == "projected(file)":
        if meta.get("projector_consistent_with_d", False):
            st.success(f"projected(file) OK · {meta.get('projector_filename','')} · {meta.get('projector_hash','')[:12]} ✔️")
        else:
            st.warning("Projected(file) is not consistent with current d3 (check shape/idempotence/diag/lane).")

# ---- Single canonical button ----
if st.button("Run Overlap", key="btn_run_overlap_main"):
    soft_reset_before_overlap()  # global helper (deduped earlier)
    run_overlap()




# (optional) minimal debug expander; safe to remove
with st.expander("Debug · d3 & lane mask"):
    try:
        d3_now = (boundaries.blocks.__root__.get("3") or [])
        n3_now = len(d3_now[0]) if (d3_now and d3_now[0]) else 0
        lm_now = _truth_mask_from_d3(d3_now)
        st.write(f"n3={n3_now} · lane_mask(d3)={lm_now}")
        cfg_now = st.session_state.get("overlap_cfg") or {}
        if (cfg_now.get("source", {}) or {}).get("3") == "file":
            try:
                P_file, _meta = projector_choose_active(cfg_now, boundaries)
                diagP = [int(P_file[i][i] & 1) for i in range(len(P_file))] if P_file else []
                st.write(f"diag(P_file)={diagP}")
                if diagP != lm_now and lm_now:
                    st.warning("diag(P) ≠ lane_mask(d3) → FILE projector will fail validation.")
            except Exception as e:
                st.error(f"Could not load FILE projector: {e}")
        else:
            st.caption("No FILE projector active (strict/AUTO).")
    except Exception as e:
        st.error(f"Debug probe failed: {e}")



# -------------------- Health checks + compact, non-duplicated UI --------------------

def run_self_tests():
    failures, warnings = [], []
    ib = st.session_state.get("_inputs_block") or {}
    di = st.session_state.get("_district_info") or {}
    rc = st.session_state.get("run_ctx") or {}
    ab = st.session_state.get("ab_compare") or {}
    out = st.session_state.get("overlap_out") or {}

    # HASH_COHERENT: boundaries hash in SSOTs must match
    bh_ib = ib.get("boundaries_hash", ""); bh_di = di.get("boundaries_hash", "")
    if bh_ib and bh_di and bh_ib != bh_di:
        failures.append("HASH_COHERENT: _inputs_block.boundaries_hash ≠ _district_info.boundaries_hash")

    # AUTO_OK / FILE_OK
    mode = rc.get("mode", "")
    if mode.startswith("projected(file)"):
        if not bool(rc.get("projector_consistent_with_d", False)):
            failures.append("FILE_OK: projected(file) not consistent with d3")
    elif mode.startswith("projected(auto)"):
        if "3" not in out:
            warnings.append("AUTO_OK: no overlap_out present yet")

    # AB_FRESH
    if ab:
        if ab.get("inputs_sig") != _current_inputs_sig():
            warnings.append("AB_FRESH: A/B snapshot is stale (won’t embed)")

    # Four core hashes should exist
    for k in ("boundaries_hash","C_hash","H_hash","U_hash"):
        if not ib.get(k):
            warnings.append(f"SSOT: missing {k}")

    return failures, warnings

# Policy pill + run stamp (single rendering)
_rc = st.session_state.get("run_ctx") or {}
_ib = st.session_state.get("_inputs_block") or {}
policy_tag = _rc.get("policy_tag") or policy_label_from_cfg(cfg_active)
n3 = _rc.get("n3") or (_ib.get("dims", {}) or {}).get("n3", 0)
_short8 = lambda h: (h or "")[:8]
bH = _short8(_ib.get("boundaries_hash","")); cH = _short8(_ib.get("C_hash",""))
hH = _short8(_ib.get("H_hash",""));        uH = _short8(_ib.get("U_hash",""))
pH = _short8(_rc.get("projector_hash","")) if str(_rc.get("mode","")).startswith("projected") else "—"

st.markdown(f"**Policy:** `{policy_tag}`")
st.caption(f"{policy_tag} | n3={n3} | b={bH} C={cH} H={hH} U={uH} P={pH}")

# If any short hash is blank, hint to fix SSOT population
if any(x in ("", None) for x in (_ib.get("boundaries_hash"), _ib.get("C_hash"), _ib.get("H_hash"), _ib.get("U_hash"))):
    st.warning("Some provenance hashes are blank. Make sure `_inputs_block` is filled before running Overlap.")

# Self-tests banner
_fail, _warn = run_self_tests()
if _fail:
    st.error("🚨 Plumbing not healthy — fix before exploration.")
    with st.expander("Self-tests details"):
        if _fail:
            st.markdown("**Failures:**")
            for f in _fail: st.write(f"- {f}")
        if _warn:
            st.markdown("**Warnings:**")
            for w in _warn: st.write(f"- {w}")
else:
    st.success("🟢 Self-tests passed.")
    if _warn:
        st.info("Notes:")
        for w in _warn: st.write(f"- {w}")

# ── Residual chips + single A/B freshness pill (place under run-stamp) ─────────
_rtags = st.session_state.get("residual_tags") or {}
if _rtags:
    s_tag = _rtags.get("strict", "—")
    p_tag = _rtags.get("projected", "—") if str(_rc.get("mode","")).startswith("projected") else "—"
    st.caption(f"Residuals → strict: `{s_tag}` · projected: `{p_tag}`")

_ab = st.session_state.get("ab_compare") or {}
if _ab:
    fresh = (_ab.get("inputs_sig") == _current_inputs_sig())
    st.caption("A/B snapshot: " + ("🟢 fresh (will embed in cert)" if fresh else "🟡 stale (won’t embed)"))
    if not fresh:
        c1, c2 = st.columns([2,3])
        with c1:
            if st.button("Clear stale A/B", key="btn_ab_clear"):
                st.session_state.pop("ab_compare", None)
                st.success("Cleared A/B snapshot. Re-run A/B to refresh.")
        with c2:
            st.caption("Tip: re-run A/B after changing inputs to refresh the snapshot.")
else:
    st.caption("A/B snapshot: —")



# ====================== A/B Compare (strict vs ACTIVE projected) ======================

def _inputs_sig_now_from_ib(ib: dict) -> list[str]:
    return [
        str(ib.get("boundaries_hash","")),
        str(ib.get("C_hash","")),
        str(ib.get("H_hash","")),
        str(ib.get("U_hash","")),
        str(ib.get("shapes_hash","")),
    ]

def _canonical_policy_tag(rc: dict) -> str:
    # prefer the tag from run_ctx; fall back to label-from-cfg if needed
    try:
        return str(rc.get("policy_tag") or policy_label_from_cfg(cfg_active))
    except Exception:
        return str(rc.get("policy_tag") or "strict")

def _ab_is_fresh(ab: dict, *, rc: dict, ib: dict) -> bool:
    if not ab: return False
    if ab.get("inputs_sig") != _inputs_sig_now_from_ib(ib): return False
    # projected tag must match current run
    if (ab.get("projected") or {}).get("policy_tag") != _canonical_policy_tag(rc): return False
    # if FILE, projector hash must match
    if str(rc.get("mode","")).startswith("projected(file)"):
        if (ab.get("projected") or {}).get("projector_hash","") != (rc.get("projector_hash","") or ""):
            return False
    return True

with safe_expander("A/B compare (strict vs active projected)"):
    if st.button("Run A/B compare", key="ab_run_btn_final"):
        try:
            ss = st.session_state
            # require a fresh run_ctx; if you have your own guards, call them here
            rc = ss.get("run_ctx") or {}
            out_active = ss.get("overlap_out") or {}
            ib = ss.get("_inputs_block") or {}

            mode_now = str(rc.get("mode","strict"))
            if mode_now == "strict":
                st.warning("Active policy is strict — run Overlap in projected(auto/file) first to compare.")
                st.stop()

            # use the SAME inputs used by the current run
            boundaries_obj = boundaries
            cmap_obj = cmap
            H_used = ss.get("overlap_H") or _load_h_local()

            # strict leg (fresh)
            out_strict = overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_used)
            label_strict = "strict"

            # projected leg = ACTIVE run’s result (no recompute)
            out_proj = out_active
            label_proj = _canonical_policy_tag(rc)

            # projector metadata (exactly from run_ctx)
            pj_hash = rc.get("projector_hash","") if mode_now.startswith("projected") else ""
            pj_file = rc.get("projector_filename","") if mode_now == "projected(file)" else ""
            pj_cons = rc.get("projector_consistent_with_d", None)

            # lane vectors (mirror what you already compute; use run_ctx mask)
            lane_mask = list(rc.get("lane_mask_k3") or [])
            d3 = (boundaries_obj.blocks.__root__.get("3") or [])
            H2 = (H_used.blocks.__root__.get("2") or [])
            C3 = (cmap_obj.blocks.__root__.get("3") or [])
            I3 = eye(len(C3)) if C3 else []

            def _xor(A,B):
                if not A: return [r[:] for r in (B or [])]
                if not B: return [r[:] for r in (A or [])]
                r, c = len(A), len(A[0])
                return [[(A[i][j]^B[i][j]) & 1 for j in range(c)] for i in range(r)]
            def _bottom_row(M): return M[-1] if (M and len(M)) else []
            def _mask(vec, mask): 
                idx = [j for j,m in enumerate(mask or []) if m]
                return [vec[j] for j in idx] if (vec and idx) else []

            try:
                H2d3  = mul(H2, d3) if (H2 and d3) else []
                C3pI3 = _xor(C3, I3) if C3 else []
            except Exception:
                H2d3, C3pI3 = [], []

            lane_vec_H2d3 = _mask(_bottom_row(H2d3), lane_mask)
            lane_vec_C3I  = _mask(_bottom_row(C3pI3), lane_mask)

            inputs_sig = _inputs_sig_now_from_ib(ib)
            pair_tag = f"{label_strict}__VS__{label_proj}"

            ab_payload = {
                "pair_tag": pair_tag,
                "inputs_sig": inputs_sig,
                "lane_mask_k3": lane_mask,
                "strict": {
                    "label": label_strict,
                    "cfg":   cfg_strict(),
                    "out":   out_strict,
                    "ker_guard": "enforced",
                    "lane_vec_H2d3": lane_vec_H2d3,
                    "lane_vec_C3plusI3": lane_vec_C3I,
                    "pass_vec": [
                        int(out_strict.get("2",{}).get("eq", False)),
                        int(out_strict.get("3",{}).get("eq", False)),
                    ],
                    "projector_hash": "",
                },
                "projected": {
                    "label": label_proj,
                    "policy_tag": label_proj,
                    "cfg":   (ss.get("overlap_cfg") or cfg_active),  # whatever was used
                    "out":   out_proj,
                    "ker_guard": "off",
                    "lane_vec_H2d3": lane_vec_H2d3[:],
                    "lane_vec_C3plusI3": lane_vec_C3I[:],
                    "pass_vec": [
                        int(out_proj.get("2",{}).get("eq", False)),
                        int(out_proj.get("3",{}).get("eq", False)),
                    ],
                    "projector_filename": pj_file,
                    "projector_hash": pj_hash,
                    "projector_consistent_with_d": pj_cons,
                },
            }

            ss["ab_compare"] = ab_payload
            # nudge cert writer to embed *this* run’s A/B
            ss["should_write_cert"] = True
            ss.pop("_last_cert_write_key", None)

            s_ok = bool(out_strict.get("3",{}).get("eq", False))
            p_ok = bool(out_proj.get("3",{}).get("eq", False))
            st.success(f"A/B updated → strict={'✅' if s_ok else '❌'} · projected={'✅' if p_ok else '❌'} · {pair_tag}")

            with safe_expander("A/B snapshot (details)"):
                st.json(ab_payload)

        except Exception as e:
            st.error(f"A/B compare failed: {e}")

    # one-click clearer if snapshot goes stale
    _ab = st.session_state.get("ab_compare") or {}
    if _ab and (not _ab_is_fresh(_ab, rc=st.session_state.get("run_ctx") or {}, ib=st.session_state.get("_inputs_block") or {})):
        if st.button("Clear stale A/B", key="btn_ab_clear_final"):
            st.session_state.pop("ab_compare", None)
            st.success("Cleared A/B snapshot. Re-run A/B to refresh.")




           
# ========= F · Helpers & invariants (shared by F1/F2/F3) =========
from pathlib import Path
import os, tempfile, copy as _copy
import json as _json, hashlib as _hash
import streamlit as st
import datetime as _dt
import random as _random  # harmless; some callers use it

# 1) Schema/version + field (one source of truth for F1/F2/F3)
SCHEMA_VERSION = "1.1.0"     # coverage_sampling, perturbation_sanity, fence_stress
FIELD          = "GF(2)"     # identity.field in all JSON payloads

# 2) Guard enum (must match certs / parity)
GUARD_ENUM = ["grid", "wiggle", "echo", "fence", "ker_guard", "none", "error"]

# ===== Minimal safety shims (no-ops when real impls are loaded) =====

if "APP_VERSION" not in globals():
    APP_VERSION = "v0.1-core"  # safe default; your app can overwrite

if "_utc_iso_z" not in globals():
    def _utc_iso_z() -> str:
        # e.g., 2025-10-13T07:42:12Z
        return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# Session guards
if "require_fresh_run_ctx" not in globals():
    def require_fresh_run_ctx():
        rc = st.session_state.get("run_ctx")
        if not rc:
            raise RuntimeError("RUN_CTX_MISSING: run Overlap first.")
        return rc

if "rectify_run_ctx_mask_from_d3" not in globals():
    def rectify_run_ctx_mask_from_d3():
        rc = st.session_state.get("run_ctx") or {}
        try:
            B = st.session_state.get("boundaries")
            d3 = (B.blocks.__root__.get("3") or []) if B else []
            lm = _lane_mask_from_d3_matrix(d3)
            if lm:
                rc["lane_mask_k3"] = lm
                st.session_state["run_ctx"] = rc
        except Exception:
            pass
        return rc

if "file_validation_failed" not in globals():
    def file_validation_failed() -> bool:
        # Only disable if projected(file) is selected but invalid helpers/file
        rc = st.session_state.get("run_ctx") or {}
        if rc.get("mode") == "projected(file)":
            return not bool(rc.get("projector_filename")) or not bool(rc.get("projector_hash"))
        return False

# IO helpers
if "io" not in globals():
    class _IO_STUB:
        @staticmethod
        def parse_cmap(d):      # minimal pydantic-like stub
            class _X:
                def __init__(self, d): self.blocks = type("B", (), {"__root__": d.get("blocks", {})})
                def dict(self): return {"blocks": self.blocks.__root__}
            return _X(d or {"blocks": {}})
        @staticmethod
        def parse_boundaries(d):
            return _IO_STUB.parse_cmap(d)
    io = _IO_STUB()

if "_load_h_local" not in globals():
    def _load_h_local():
        return st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})

# Optional witness appender
if "append_witness_row" not in globals():
    def append_witness_row(*args, **kwargs):
        # no-op when witness system isn’t loaded
        return False

# Ensure inputs_hashes exists (no recompute; copy from run_ctx/certs)
if "_ensure_inputs_hashes" not in globals():
    def _ensure_inputs_hashes():
        rc = st.session_state.get("run_ctx") or {}
        ih = st.session_state.get("inputs_hashes") or {}
        keys = ["boundaries_hash","C_hash","H_hash","U_hash","shapes_hash"]
        merged = {k: (ih.get(k) or rc.get(k) or st.session_state.get(k) or "") for k in keys}
        st.session_state["inputs_hashes"] = merged
        return merged
_ensure_inputs_hashes()

# 3) Canonical content hashing (ints not bools; sorted keys; ASCII; no spaces)
def _deep_intify(o):
    """Convert True/False to 1/0 recursively so GF(2) matrices hash stably."""
    if isinstance(o, bool):
        return 1 if o else 0
    if isinstance(o, list):
        return [_deep_intify(x) for x in o]
    if isinstance(o, dict):
        return {k: _deep_intify(v) for k, v in o.items()}
    return o

def _hash_json(obj) -> str:
    canon = _deep_intify(_copy.deepcopy(obj))
    s = _json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _hash.sha256(s.encode("ascii")).hexdigest()

def _sha256_hex(b: bytes) -> str:
    return _hash.sha256(b).hexdigest()

# 4) Atomic writers (JSON + CSV-with-meta)
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _atomic_write_csv(path: Path, header: list[str], rows: list[list], meta_lines: list[str] | None = None) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        if meta_lines:
            for line in meta_lines:
                f.write(f"# {line}\n")
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# 5) Inputs SSOT (no recompute). Pull exactly what certs/save in run_ctx.
def _inputs_block_from_session(strict_dims: tuple[int,int] | None = None) -> dict:
    """
    Returns:
      {
        "hashes": {
          "boundaries_hash","C_hash","H_hash","U_hash","shapes_hash"
        },
        "dims": {"n2": int, "n3": int},
        "lane_mask_k3": [...]
      }
    Dims priority: strict_dims → run_ctx → local safe fallback (0).
    Lane-mask is taken from run_ctx (rectified earlier) as SSOT.
    """
    rc = st.session_state.get("run_ctx") or {}
    inputs_ssot = st.session_state.get("inputs_hashes") or {}

    def _grab(key: str) -> str:
        return (
            inputs_ssot.get(key)
            or rc.get(key)
            or st.session_state.get(key)
            or ""
        )

    hashes = {
        "boundaries_hash": _grab("boundaries_hash"),
        "C_hash":          _grab("C_hash"),
        "H_hash":          _grab("H_hash"),
        "U_hash":          _grab("U_hash"),
        "shapes_hash":     _grab("shapes_hash"),
    }

    if strict_dims is not None:
        n2, n3 = int(strict_dims[0]), int(strict_dims[1])
    else:
        try:
            n2 = int(rc.get("n2")) if rc.get("n2") is not None else 0
            n3 = int(rc.get("n3")) if rc.get("n3") is not None else 0
        except Exception:
            n2, n3 = 0, 0

    lane_mask = [int(x) & 1 for x in (rc.get("lane_mask_k3") or [])]
    return {"hashes": hashes, "dims": {"n2": n2, "n3": n3}, "lane_mask_k3": lane_mask}

# 6) Projector resolution & policy helpers
def _resolve_projector_from_rc():
    """
    Returns: (mode, submode, filename, projector_hash, projector_diag or None)
    Requires `_path_exists_strict` and `_projector_diag_from_file` when submode=='file'.
    Raises RuntimeError with a clear message on missing/invalid file mode.
    """
    rc = st.session_state.get("run_ctx") or {}
    m = rc.get("mode", "strict")
    mode, submode = ("strict", "")
    if m == "projected(auto)": mode, submode = ("projected", "auto")
    elif m == "projected(file)": mode, submode = ("projected", "file")

    filename = rc.get("projector_filename", "") if (mode == "projected" and submode == "file") else ""
    pj_hash, pj_diag = "", None

    if mode == "projected" and submode == "file":
        if ("_path_exists_strict" not in globals()) or ("_projector_diag_from_file" not in globals()):
            raise RuntimeError("P3_FILE_HELPERS_MISSING: projector helpers not loaded.")
        if not filename or not _path_exists_strict(filename):
            raise RuntimeError("P3_FILE_MISSING: projector FILE missing/invalid.")
        try:
            pj_hash = _sha256_hex(Path(filename).read_bytes())
            pj_diag = _projector_diag_from_file(filename)
            if not isinstance(pj_diag, list) or not pj_diag:
                raise RuntimeError("P3_FILE_INVALID: empty/bad projector diag.")
        except Exception as e:
            raise RuntimeError(f"P3_FILE_INVALID: {e}") from e

    return mode, submode, filename, pj_hash, pj_diag

def normalize_projector_into_run_ctx():
    """Stamp projector_hash/projector_filename into run_ctx when resolvable."""
    try:
        mode, sub, fname, pj_hash, pj_diag = _resolve_projector_from_rc()
        rc_here = st.session_state.get("run_ctx") or {}
        if pj_hash:
            rc_here["projector_hash"] = pj_hash
            if fname: rc_here["projector_filename"] = fname
        st.session_state["run_ctx"] = rc_here
    except Exception as e:
        # explicit, but non-fatal (the projected(FILE) path is separately disabled)
        st.caption(f"Π normalize: {e}")

def _policy_block_from_run_ctx(rc: dict) -> dict:
    mode = str(rc.get("mode", "strict"))
    if mode == "strict":
        return {
            "policy_tag": "strict",
            "projector_mode": "strict",
            "projector_filename": "",
            "projector_hash": "",
        }
    if mode == "projected(auto)":
        diag_hash = _sha256_hex(
            _json.dumps(rc.get("lane_mask_k3") or [], sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return {
            "policy_tag": "projected(columns@k=3,auto)",
            "projector_mode": "auto",
            "projector_filename": "",
            "projector_hash": diag_hash,
        }
    # projected(file)
    return {
        "policy_tag": "projected(columns@k=3,file)",
        "projector_mode": "file",
        "projector_filename": rc.get("projector_filename", "") or "",
        "projector_hash": rc.get("projector_hash", "") or "",
    }

# 7) Guard mapping wrapper (keep enums consistent, tolerant to missing impl)
def _first_tripped_guard(strict_out: dict) -> str:
    """
    Adapter to your existing guard checker. Must return a value in GUARD_ENUM.
    If no checker available, fall back to simple logic or "none".
    """
    if "first_tripped_guard" in globals() and callable(globals()["first_tripped_guard"]):
        try:
            g = first_tripped_guard(strict_out)
            return g if g in GUARD_ENUM else "error"
        except Exception:
            return "error"
    # Minimal fallback: if k3==True → none, else fence
    try:
        k3eq = strict_out.get("3", {}).get("eq")
        if k3eq is True:  return "none"
        if k3eq is False: return "fence"
    except Exception:
        pass
    return "none"

# 8) Small badge helper (UI polish)
def _hash_badge(h: str) -> str:
    return f"wrote CSV + JSON ✓ · hash: {h[:12]}"

# 9) Strict linear algebra (NO fallbacks)
def _validate_shapes_or_raise(H2, d3, C3):
    rH, cH = (len(H2), len(H2[0]) if (H2 and H2[0]) else 0)
    rD, cD = (len(d3), len(d3[0]) if (d3 and d3[0]) else 0)
    rC, cC = (len(C3), len(C3[0]) if (C3 and C3[0]) else 0)
    # expected: H2: n3×n2 ; d3: n2×n3 ; C3: n3×n3  (infer n2,n3 from d3)
    n2, n3 = rD, cD
    ok = (
        rH == n3 and cH == n2 and
        rD == n2 and cD == n3 and
        rC == n3 and cC == n3
    )
    if not ok:
        raise RuntimeError(
            f"R3_SHAPE: expected H2({n3}×{n2})·d3({n2}×{n3}) and (C3⊕I3)({n3}×{n3}), "
            f"got H2({rH}×{cH}), d3({rD}×{cD}), C3({rC}×{cC})"
        )

def _strict_R3(H2: list[list[int]], d3: list[list[int]], C3: list[list[int]]) -> list[list[int]]:
    """Compute R3 = (H2 @ d3) XOR (C3 XOR I3). Strict shape checks; no fallbacks."""
    if not (H2 and d3 and C3):
        raise RuntimeError("R3_INPUTS_MISSING: require non-empty H2, d3, C3.")
    _validate_shapes_or_raise(H2, d3, C3)
    if "mul" not in globals() or not callable(globals()["mul"]):
        raise RuntimeError("R3_MUL_MISSING: GF(2) mul(H2,d3) not available.")
    M = mul(H2, d3)  # type: ignore[name-defined]
    n3 = len(C3)
    I3 = [[1 if i == j else 0 for j in range(n3)] for i in range(n3)]
    C3p = [[(C3[i][j] ^ I3[i][j]) & 1 for j in range(n3)] for i in range(n3)]
    R  = [[(M[i][j] ^ C3p[i][j]) & 1 for j in range(n3)] for i in range(n3)]
    return R

def _projected_R3(R3_strict: list[list[int]], P_active: list[list[int]] | None):
    """Multiply R3 by Π when present; strict shape checks."""
    if not (R3_strict and P_active):
        return []
    rR, cR = len(R3_strict), len(R3_strict[0]) if R3_strict and R3_strict[0] else 0
    rP, cP = len(P_active), len(P_active[0]) if P_active and P_active[0] else 0
    if "mul" not in globals() or not callable(globals()["mul"]):
        raise RuntimeError("R3P_MUL_MISSING: GF(2) mul(R3, Π) not available.")
    if cR != rP:
        raise RuntimeError(f"R3P_SHAPE: expected R3({rR}×{cR})·Π({rP}×{cP}) with {cR}=={rP}.")
    return mul(R3_strict, P_active)  # type: ignore[name-defined]

def _lane_mask_from_d3_matrix(d3: list[list[int]]) -> list[int]:
    if not d3 or not d3[0]:
        return []
    rows, n3 = len(d3), len(d3[0])
    return [1 if any(d3[i][j] & 1 for i in range(rows)) else 0 for j in range(n3)]

def _sig_tag_eq(boundaries_obj, cmap_obj, H_used_obj, P_active=None):
    """
    Return (lane_mask, tag_strict, eq3_strict, tag_proj, eq3_proj).
    Uses optional global residual_tag(R3, lane_mask) if available.
    """
    d3 = (boundaries_obj.blocks.__root__.get("3") or [])
    H2 = (H_used_obj.blocks.__root__.get("2") or [])
    C3 = (cmap_obj.blocks.__root__.get("3") or [])
    lm = _lane_mask_from_d3_matrix(d3)
    R3s = _strict_R3(H2, d3, C3)

    if "residual_tag" in globals() and callable(globals()["residual_tag"]):
        try:
            tag_s = residual_tag(R3s, lm)  # type: ignore[name-defined]
        except Exception:
            tag_s = "error"
    else:
        # Local classifier
        def _residual_tag_local(R, mask):
            if not R or not mask: return "none"
            m = len(R)
            def _nz(j): return any(R[i][j] & 1 for i in range(m))
            lanes = any(_nz(j) for j, b in enumerate(mask) if b)
            ker   = any(_nz(j) for j, b in enumerate(mask) if not b)
            if lanes and ker: return "mixed"
            if lanes:         return "lanes"
            if ker:           return "ker"
            return "none"
        tag_s = _residual_tag_local(R3s, lm)

    eq_s = (len(R3s) == 0) or all(all((x & 1) == 0 for x in row) for row in R3s)

    if P_active:
        R3p   = _projected_R3(R3s, P_active)
        if "residual_tag" in globals() and callable(globals()["residual_tag"]):
            try: tag_p = residual_tag(R3p, lm)  # type: ignore[name-defined]
            except Exception: tag_p = "error"
        else:
            tag_p = tag_s
        eq_p  = (len(R3p) == 0) or all(all((x & 1) == 0 for x in row) for row in R3p)
    else:
        tag_p, eq_p = None, None

    return lm, tag_s, bool(eq_s), tag_p, (None if eq_p is None else bool(eq_p))

# -------- optional carrier (U) mutation hooks ----------
HAS_U_HOOKS = (
    "get_carrier_mask" in globals() and "set_carrier_mask" in globals()
    and callable(globals()["get_carrier_mask"]) and callable(globals()["set_carrier_mask"])
)

with st.expander("Reports: Perturbation Sanity & Fence Stress"):
    # Ensure reports dir exists (defensive)
    REPORTS_DIR = Path(st.session_state.get("REPORTS_DIR", "reports"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure SSOT input hashes are present (no recompute; from helpers)
    if "_ensure_inputs_hashes" in globals():
        _ensure_inputs_hashes()

    # Freshness + SSOT guardrails
    try:
        rc = require_fresh_run_ctx()
        rc = rectify_run_ctx_mask_from_d3()
    except Exception as e:
        st.warning(str(e))
        st.stop()

    # Inputs / policy context
    H_used  = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})
    P_active = rc.get("P_active") if str(rc.get("mode","")).startswith("projected") else None
    B0, C0, H0 = boundaries, cmap, H_used
    U0 = shapes  # carrier (for Fence)

    d3_base = (B0.blocks.__root__.get("3") or [])
    n2 = len(d3_base)
    n3 = len(d3_base[0]) if (d3_base and d3_base[0]) else 0

    # UI
    colA, colB = st.columns([2,2])
    with colA:
        max_flips = st.number_input("Perturbation: max flips", min_value=1, max_value=500, value=24, step=1, key="ps_max")
        seed_txt  = st.text_input("Seed (determines flip order)", value="ps-seed-1", key="ps_seed")
    with colB:
        run_fence       = st.checkbox("Include Fence stress run (perturb U)", value=True, key="fence_on")
        enable_witness  = st.checkbox("Write witness on mismatches", value=True, key="ps_witness_on")

    # Disable downstream actions if FILE Π invalid (global predicate)
    disabled = file_validation_failed()
    help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    if st.button(
        "Run Perturbation Sanity (and Fence if checked)",
        key="ps_run",
        disabled=disabled,
        help=(help_txt if disabled else "Run perturbation sanity; optionally include fence"),
    ):
        try:
            # ───────────────────────── Baseline (no mutation) ─────────────────────────
            lm0, tag_s0, eq_s0, tag_p0, eq_p0 = _sig_tag_eq(B0, C0, H0, P_active)

            # lanes-only: allowed columns = 1-bits in current lane mask (strict domain)
            inputs_ps_tmp = _inputs_block_from_session(strict_dims=(n2, n3))
            lane_mask = [int(x) & 1 for x in inputs_ps_tmp.get("lane_mask_k3", [])]
            allowed_cols_set = {j for j, b in enumerate(lane_mask) if b == 1}

            # Deterministic flip generator
            import hashlib as _hashlib
            def _flip_targets_lanes_only(n2_, n3_, budget, seed_str):
                h = int(_hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16)
                i = (h % max(1, n2_)) if n2_ else 0
                j = ((h >> 8) % max(1, n3_)) if n3_ else 0
                for k in range(int(budget)):
                    yield (i, j, k)
                    i = (i + 1 + (h % 3)) % (n2_ or 1)
                    j = (j + 2 + ((h >> 5) % 5)) % (n3_ or 1)

            # ───────────────── Perturbation: flips + CSV + JSON ─────────────────
            rows = []       # CSV rows
            ps_results = [] # JSON results[]
            matches = 0
            mismatches = 0
            total_flips = 0
            in_domain_flips = 0

            for (r, c, k) in _flip_targets_lanes_only(n2, n3, int(max_flips), seed_txt):
                total_flips += 1

                # Empty fixture → no-op row
                if not (n2 and n3):
                    rows.append([k, 0, "none", "empty fixture"])
                    ps_results.append({
                        "flip_id": int(k),
                        "guard_tripped": "none",
                        "expected_guard": "none",
                        "flip_spec": {"row": int(r), "col": int(c),
                                      "bit_before": None, "bit_after": None,
                                      "lane_col": False, "skip_reason": "empty-fixture"},
                        "k_status_before": {"2": True, "3": bool(eq_s0)},
                        "k_status_after":  {"2": True, "3": bool(eq_s0)},
                        "residual_tag_after": str(tag_s0 or "none"),
                        "witness_written": False,
                        "note": "empty fixture",
                    })
                    continue

                lane_col = (c in allowed_cols_set)
                bit_before = int(d3_base[r][c]) if (r < len(d3_base) and c < len(d3_base[0])) else 0

                # off-domain → expected none; do NOT count toward matches/mismatches
                if not lane_col:
                    rows.append([k, 0, "none", "off-domain (ker column)"])
                    ps_results.append({
                        "flip_id": int(k),
                        "guard_tripped": "none",
                        "expected_guard": "none",
                        "flip_spec": {"row": int(r), "col": int(c),
                                      "bit_before": bit_before, "bit_after": bit_before,
                                      "lane_col": False, "skip_reason": "off-domain"},
                        "k_status_before": {"2": True, "3": bool(eq_s0)},
                        "k_status_after":  {"2": True, "3": bool(eq_s0)},
                        "residual_tag_after": str(tag_s0 or "none"),
                        "witness_written": False,
                        "note": "off-domain (ker column)",
                    })
                    continue

                # In-domain mutation
                in_domain_flips += 1
                d3_mut = [row[:] for row in d3_base]
                if r >= len(d3_mut) or (len(d3_mut) and c >= len(d3_mut[0])):
                    rows.append([k, 0, "none", f"skip flip out-of-range r={r},c={c}"])
                    ps_results.append({
                        "flip_id": int(k),
                        "guard_tripped": "none",
                        "expected_guard": "none",
                        "flip_spec": {"row": int(r), "col": int(c),
                                      "bit_before": bit_before, "bit_after": bit_before,
                                      "lane_col": True, "skip_reason": "out-of-range"},
                        "k_status_before": {"2": True, "3": bool(eq_s0)},
                        "k_status_after":  {"2": True, "3": bool(eq_s0)},
                        "residual_tag_after": str(tag_s0 or "none"),
                        "witness_written": False,
                        "note": "flip skipped: out-of-range",
                    })
                    continue

                d3_mut[r][c] ^= 1
                bit_after = int(d3_mut[r][c])

                dB = B0.dict() if hasattr(B0, "dict") else {"blocks": {}}
                dB = _json.loads(_json.dumps(dB))
                dB.setdefault("blocks", {})["3"] = d3_mut
                Bk = io.parse_boundaries(dB)

                lmK, tag_sK, eq_sK, tag_pK, eq_pK = _sig_tag_eq(Bk, C0, H0, P_active)

                # Guard mapping (single enum set)
                strict_out = {"3": {"eq": bool(eq_sK)}}
                guard = _first_tripped_guard(strict_out)

                expected_guard = "grid"  # lanes-only in-domain flip should alter grammar
                rows.append([k, 1 if guard != "none" else 0, expected_guard, ""])
                ok = (guard == expected_guard)
                matches += int(ok)
                mismatches += int(not ok)

                # Optional witness on drift (minimal)
                witness_written = False
                if enable_witness and not ok:
                    cert_like = st.session_state.get("cert_payload")
                    if cert_like and "append_witness_row" in globals():
                        try:
                            append_witness_row(
                                cert_like,
                                reason="grammar-drift",
                                residual_tag_val=(tag_sK or "none"),
                                note=f"flip#{k} at (r={r}, c={c}) guard:{guard} expected:{expected_guard}",
                            )
                            witness_written = True
                        except Exception:
                            witness_written = False

                ps_results.append({
                    "flip_id": int(k),
                    "guard_tripped": guard,
                    "expected_guard": expected_guard,
                    "flip_spec": {
                        "row": int(r), "col": int(c),
                        "bit_before": int(bit_before), "bit_after": int(bit_after),
                        "lane_col": True,
                    },
                    "k_status_before": {"2": True, "3": bool(eq_s0)},   # k2 placeholder=true here
                    "k_status_after":  {"2": True, "3": bool(eq_sK)},
                    "residual_tag_after": str(tag_sK or "none"),
                    "witness_written": bool(witness_written),
                    "note": "",
                })

            # Write Perturbation CSV
            PERTURB_OUT_PATH = REPORTS_DIR / "perturbation_sanity.csv"
            _atomic_write_csv(
                PERTURB_OUT_PATH,
                header=["flip_id", "guard_tripped", "expected_guard", "note"],
                rows=rows,
                meta_lines=[
                    f"schema_version={SCHEMA_VERSION}",
                    f"saved_at={_utc_iso_z()}",
                    f"run_id={(st.session_state.get('run_ctx') or {}).get('run_id','')}",
                    f"app_version={APP_VERSION}",
                    f"seed={seed_txt}",
                    f"n2={n2}",
                    f"n3={n3}",
                    f"baseline_tag_strict={tag_s0}",
                    f"baseline_tag_projected={'' if tag_p0 is None else tag_p0}",
                ],
            )
            st.success(f"Perturbation sanity saved → {PERTURB_OUT_PATH}")

            # Build Perturbation JSON
            try:
                rc_ps = require_fresh_run_ctx()
            except Exception:
                rc_ps = st.session_state.get("run_ctx") or {}

            # Normalize Π info into run_ctx just before policy build
            if "normalize_projector_into_run_ctx" in globals():
                normalize_projector_into_run_ctx()

            policy_ps = _policy_block_from_run_ctx(rc_ps)
            inputs_ps = _inputs_block_from_session(strict_dims=(n2, n3))
            lm_bits_ps = "".join("1" if int(x) else "0" for x in inputs_ps.get("lane_mask_k3", []))
            if lm_bits_ps and len(lm_bits_ps) != int(n3):
                st.caption(f"⚠︎ lane_mask_k3 has {len(lm_bits_ps)} bits but n₃={n3}")

            summary_ps = {
                "matches": int(matches),
                "mismatches": int(mismatches),
                "off_domain_count": int(total_flips - in_domain_flips),
                "policy_tag": policy_ps.get("policy_tag",""),
                "projector_hash": policy_ps.get("projector_hash",""),
            }

            perturb_json = {
                "schema_version": SCHEMA_VERSION,
                "written_at_utc": _utc_iso_z(),
                "app_version": APP_VERSION,
                "field": FIELD,
                "identity": {
                    "run_id": (rc_ps.get("run_id") or (st.session_state.get("run_ctx") or {}).get("run_id") or ""),
                    "district_id": rc_ps.get("district_id","D3"),
                    "fixture_nonce": rc_ps.get("fixture_nonce",""),
                },
                "policy": policy_ps,
                "inputs": inputs_ps,
                "anchor": {
                    "id": rc_ps.get("fixture_nonce", ""),
                    "hashes": inputs_ps.get("hashes", {}),           # SSOT copy
                    "lane_mask_k3": inputs_ps.get("lane_mask_k3", []),
                },
                "run": {
                    "max_flips": int(max_flips),
                    "flip_domain": "lanes-only",
                    "ker_guard": "enforced",
                    "seed": str(seed_txt),
                    "guard_order": GUARD_ENUM,
                },
                "results": ps_results,
                "summary": summary_ps,
                "integrity": {"content_hash": ""},
            }
            perturb_json["integrity"]["content_hash"] = _hash_json(perturb_json)

            # Persist + downloads + badge
            try:
                h12 = perturb_json["integrity"]["content_hash"][:12]
                h8  = perturb_json["integrity"]["content_hash"][:8]
                basename = f"perturbation_sanity__{h12}.json"
                pert_json_path = REPORTS_DIR / basename
                _atomic_write_json(pert_json_path, perturb_json)
                st.session_state.setdefault("last_report_paths", {})["perturbation_sanity"] = {
                    "csv": str(PERTURB_OUT_PATH), "json": str(pert_json_path)
                }
                # Downloads
                import io as _io
                mem = _io.BytesIO(_json.dumps(perturb_json, ensure_ascii=False, indent=2).encode("utf-8"))
                st.download_button("Download perturbation_sanity.json", mem,
                                   file_name=basename, key=f"dl_ps_json_{h8}")
                with open(PERTURB_OUT_PATH, "rb") as fcsv:
                    st.download_button("Download perturbation_sanity.csv", fcsv,
                                       file_name=f"perturbation_sanity__{h12}.csv",
                                       key=f"dl_ps_csv_{h8}")
                st.info(f"wrote JSON ✓ · hash: {h12} · saved as {basename}")
                st.info(f"lanes-only flips: {in_domain_flips}/{total_flips} · mismatches: {mismatches}")
            except Exception as e:
                st.info(f"(Perturbation JSON/Downloads issue: {e})")

            # ───────────────────── Fence stress (baseline + variants) ─────────────────────
            if run_fence:
                rows_fs = []
                mode_str = "U" if HAS_U_HOOKS else "H2"
                fallback_note = ("" if HAS_U_HOOKS else "U hooks unavailable; used H2")

                # Baseline (no change)
                d3 = (B0.blocks.__root__.get("3") or [])
                H2 = (H0.blocks.__root__.get("2") or [])
                C3 = (C0.blocks.__root__.get("3") or [])
                R3_base = _strict_R3(H2, d3, C3)
                k2_base = 1
                k3_base = int((not R3_base) or all(all((x & 1) == 0 for x in row) for row in R3_base))
                rows_fs.append(["U_min", f"[{k2_base},{k3_base}]", "baseline"])

                # Variants
                if HAS_U_HOOKS:
                    # Carrier mask mutate
                    U_mask = get_carrier_mask(U0)  # type: ignore[name-defined]
                    rU = len(U_mask); cU = len(U_mask[0]) if (U_mask and U_mask[0]) else 0
                    def _count1(M): return sum(int(x & 1) for row in (M or []) for x in row)

                    # shrink on border
                    U_shrink = [row[:] for row in U_mask]
                    if rU and cU:
                        for j in range(cU): U_shrink[0][j] = 0; U_shrink[-1][j] = 0
                        for i in range(rU): U_shrink[i][0] = 0; U_shrink[i][-1] = 0
                    # plus on border
                    U_plus = [row[:] for row in U_mask]
                    if rU and cU:
                        for j in range(cU): U_plus[0][j]  = 1; U_plus[-1][j] = 1
                        for i in range(rU): U_plus[i][0]  = 1; U_plus[i][-1] = 1

                    # Minimal semantics: record deltas; fence outcome may remain unchanged
                    k2_s = 1; k3_s = k3_base
                    k2_p = 1; k3_p = k3_base

                    rows_fs.append([
                        "U_shrink",
                        f"[{k2_s},{k3_s}]",
                        _json.dumps({"delta_U": {
                            "added": 0,
                            "removed": int(_count1(U_mask) - _count1(U_shrink)),
                            "size_before": int(_count1(U_mask)),
                            "size_after": int(_count1(U_shrink)),
                        }}, separators=(",", ":"))
                    ])
                    rows_fs.append([
                        "U_plus",
                        f"[{k2_p},{k3_p}]",
                        _json.dumps({"delta_U": {
                            "added": int(_count1(U_plus) - _count1(U_mask)),
                            "removed": 0,
                            "size_before": int(_count1(U_mask)),
                            "size_after": int(_count1(U_plus)),
                        }}, separators=(",", ":"))
                    ])
                else:
                    # H2 fallback: drop last row / append zero row
                    H2 = (H0.blocks.__root__.get("2") or [])
                    H2_shrink = [row[:] for row in H2[:-1]] if len(H2) >= 1 else [row[:] for row in H2]
                    if H2 and H2[0]:
                        zero_row = [0]*len(H2[0]); H2_plus = [row[:] for row in H2] + [zero_row]
                    else:
                        H2_plus = [row[:] for row in H2]

                    R3_shrink = _strict_R3(H2_shrink, d3, C3)
                    R3_plus   = _strict_R3(H2_plus,   d3, C3)
                    k2_s = 1; k3_s = int((not R3_shrink) or all(all((x & 1) == 0 for x in row) for row in R3_shrink))
                    k2_p = 1; k3_p = int((not R3_plus)   or all(all((x & 1) == 0 for x in row) for row in R3_plus))

                    def _unchanged_note(k3_before, k3_after):
                        return "H2-row change only; fence unchanged" if int(k3_before) == int(k3_after) else ""

                    rows_fs.append([
                        "H2_shrink",
                        f"[{k2_s},{k3_s}]",
                        _json.dumps({"delta_H2":{"rows_before": len(H2), "rows_after": len(H2_shrink)},
                                     "note": _unchanged_note(k3_base, k3_s)}, separators=(",", ":"))
                    ])
                    rows_fs.append([
                        "H2_plus",
                        f"[{k2_p},{k3_p}]",
                        _json.dumps({"delta_H2":{"rows_before": len(H2), "rows_after": len(H2_plus)},
                                     "note": _unchanged_note(k3_base, k3_p)}, separators=(",", ":"))
                    ])

                # Fence CSV
                FENCE_OUT_PATH = REPORTS_DIR / "fence_stress.csv"
                fence_meta = [
                    f"schema_version={SCHEMA_VERSION}",
                    f"saved_at={_utc_iso_z()}",
                    f"run_id={(st.session_state.get('run_ctx') or {}).get('run_id','')}",
                    f"app_version={APP_VERSION}",
                ]
                if fallback_note:
                    fence_meta.append(f"note={fallback_note}")

                _atomic_write_csv(FENCE_OUT_PATH, header=["U_class", "pass_vec", "note"], rows=rows_fs, meta_lines=fence_meta)
                st.success(f"Fence stress saved → {FENCE_OUT_PATH}")

                # Fence JSON
                try:
                    rc_fs = require_fresh_run_ctx()
                except Exception:
                    rc_fs = st.session_state.get("run_ctx") or {}

                # Normalize Π info into run_ctx just before policy build
                if "normalize_projector_into_run_ctx" in globals():
                    normalize_projector_into_run_ctx()

                policy_fs = _policy_block_from_run_ctx(rc_fs)
                inputs_fs = _inputs_block_from_session(strict_dims=(n2, n3))

                # Map CSV rows to JSON results (and try to decode delta_* objects from note where present)
                results_fs_json = []
                for r in rows_fs:
                    U_class = str(r[0]) if len(r) > 0 else "U_unknown"
                    pass_vec_raw = str(r[1]) if len(r) > 1 else "[?,?]"
                    note_raw = str(r[2]) if len(r) > 2 else ""
                    try:
                        pv = pass_vec_raw.strip("[]").split(",")
                        k2b = bool(int(str(pv[0]).strip()))
                        k3b = bool(int(str(pv[1]).strip()))
                    except Exception:
                        k2b = False; k3b = False
                    # try parse JSON-like note for deltas
                    delta_obj = None
                    try:
                        maybe = _json.loads(note_raw)
                        if isinstance(maybe, dict):
                            delta_obj = maybe
                    except Exception:
                        pass
                    item = {"U_class": U_class, "pass_vec": [k2b, k3b], "note": ("" if delta_obj else note_raw)}
                    if delta_obj:
                        item.update(delta_obj)
                    results_fs_json.append(item)

                # Fence summary
                summary_fs = {
                    "baseline_pass_vec": [bool(k2_base), bool(k3_base)],
                    "U_shrink_pass_vec": [bool(k2_s), bool(k3_s)],
                    "U_plus_pass_vec":   [bool(k2_p), bool(k3_p)],
                }

                fence_json = {
                    "schema_version": SCHEMA_VERSION,
                    "written_at_utc": _utc_iso_z(),
                    "app_version": APP_VERSION,
                    "field": FIELD,
                    "identity": {
                        "run_id": (rc_fs.get("run_id") or (st.session_state.get("run_ctx") or {}).get("run_id") or ""),
                        "district_id": rc_fs.get("district_id","D3"),
                        "fixture_nonce": rc_fs.get("fixture_nonce",""),
                    },
                    "mode": mode_str,
                    "fallback_note": (fallback_note or ""),
                    "policy": policy_fs,
                    "inputs": inputs_fs,
                    "exemplar": {
                        "fixture_id": rc_fs.get("fixture_nonce",""),
                        "lane_mask_k3": inputs_fs.get("lane_mask_k3", []),
                        "hashes": inputs_fs.get("hashes", {}),  # SSOT copy
                    },
                    "results": results_fs_json,
                    "summary": summary_fs,
                    "integrity": {"content_hash": ""},
                }
                fence_json["integrity"]["content_hash"] = _hash_json(fence_json)

                try:
                    h12 = fence_json["integrity"]["content_hash"][:12]
                    h8  = fence_json["integrity"]["content_hash"][:8]
                    basename = f"fence_stress__{h12}.json"
                    fence_json_path = REPORTS_DIR / basename
                    _atomic_write_json(fence_json_path, fence_json)
                    st.session_state.setdefault("last_report_paths", {})["fence_stress"] = {
                        "csv": str(FENCE_OUT_PATH), "json": str(fence_json_path)
                    }
                    # Downloads
                    import io as _io
                    mem = _io.BytesIO(_json.dumps(fence_json, ensure_ascii=False, indent=2).encode("utf-8"))
                    st.download_button("Download fence_stress.json", mem,
                                       file_name=basename, key=f"dl_fs_json_{h8}")
                    with open(FENCE_OUT_PATH, "rb") as fcsv:
                        st.download_button("Download fence_stress.csv", fcsv,
                                           file_name=f"fence_stress__{h12}.csv",
                                           key=f"dl_fs_csv_{h8}")
                    st.info(f"wrote JSON ✓ · hash: {h12} · saved as {basename}")
                    st.info(f"{mode_str} · baseline [k2,k3]=[{k2_base},{k3_base}] → shrink [{k2_s},{k3_s}] · U⁺ [{k2_p},{k3_p}]")
                except Exception as e:
                    st.info(f"(Fence JSON/Downloads issue: {e})")

        except Exception as e:
            st.error(f"Perturbation/Fence run failed: {e}")


# ---- Coverage helpers (idempotent) ----
import random as _random

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
            pivot = None
            for i in range(r, m):
                if A[i][c] & 1: pivot = i; break
            if pivot is None:
                c += 1; continue
            if pivot != r:
                A[r], A[pivot] = A[pivot], A[r]
            for i in range(r+1, m):
                if A[i][c] & 1:
                    A[i] = [(A[i][j] ^ A[r][j]) for j in range(n)]
            r += 1; c += 1
        return r

if "_col_support_pattern" not in globals():
    def _col_support_pattern(M: list[list[int]]) -> list[str]:
        if not M: return []
        rows, cols = len(M), len(M[0])
        cols_bits = []
        for j in range(cols):
            bits = ''.join('1' if (M[i][j] & 1) else '0' for i in range(rows))
            cols_bits.append(bits)
        cols_bits.sort()  # canonical: lexicographic
        return cols_bits

if "_coverage_signature" not in globals():
    def _coverage_signature(d_k1: list[list[int]], n_k: int) -> str:
        rk = _gf2_rank(d_k1)
        ker = max(0, int(n_k) - rk)
        patt = _col_support_pattern(d_k1)  # canonical sorted column bitstrings
        return f"rk={rk};ker={ker};pattern=[{','.join(patt)}]"

if "_lane_pattern_from_mask" not in globals():
    def _lane_pattern_from_mask(mask: list[int]) -> str:
        return ''.join('1' if (int(x) & 1) else '0' for x in (mask or []))

if "_in_district_guess" not in globals():
    def _in_district_guess(signature: str, *, current_lane_pattern: str) -> int:
        """Heuristic: mark in_district if any column pattern matches the lane bitstring."""
        try:
            bracket = signature.split("pattern=[", 1)[1].split("]", 1)[0]
            col_bitstrings = [s.strip() for s in bracket.split(",") if s.strip()]
            return int(any(bs == current_lane_pattern for bs in col_bitstrings))
        except Exception:
            return 0


# =============================== Coverage Sampling (non-blocking) ==============================
from pathlib import Path

# Ensure REPORTS_DIR exists for coverage block (defensive bootstrap)
if "REPORTS_DIR" not in globals() or REPORTS_DIR is None:
    REPORTS_DIR = Path("reports")
else:
    REPORTS_DIR = Path(REPORTS_DIR)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_CSV_PATH = REPORTS_DIR / "coverage_sampling.csv"
COVERAGE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

with st.expander("Coverage Sampling"):
    # Ensure SSOT input hashes are present (no recompute)
    if "_ensure_inputs_hashes" in globals():
        _ensure_inputs_hashes()

    # Freshness — do not stop the render if missing
    rc = None
    try:
        rc = require_fresh_run_ctx()
        rc = rectify_run_ctx_mask_from_d3()
    except Exception as e:
        st.warning(str(e))

    # Defaults (safe even if rc is None)
    n3_default = int((rc or {}).get("n3") or 0)
    try:
        H_local = st.session_state.get("overlap_H") or _load_h_local()
        H2_rows = len((H_local.blocks.__root__.get("2") or []))
    except Exception:
        H2_rows = 0
    n2_default = H2_rows

    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        num_samples = st.number_input("Samples", min_value=1, max_value=10000,
                                      value=250, step=50, key="cov_nsamples")
    with c2:
        bit_density = st.slider("Bit density", min_value=0.0, max_value=1.0,
                                value=0.25, step=0.05, key="cov_density")
    with c3:
        n2 = st.number_input("Rows (n₂)", min_value=0, max_value=2048,
                             value=n2_default, step=1, key="cov_n2")
    with c4:
        n3 = st.number_input("Cols (n₃)", min_value=0, max_value=2048,
                             value=n3_default, step=1, key="cov_n3")

    seed_txt = st.text_input("Seed (any string/hex)", value="cov-seed-0001", key="cov_seed")

    # Disable when not ready (no stops)
    file_bad = file_validation_failed()
    cov_disabled = file_bad or (rc is None) or (n2_default <= 0) or (n3_default <= 0)
    tips = []
    if rc is None: tips.append("Run Overlap first.")
    if n2_default <= 0 or n3_default <= 0: tips.append("Fixture dims unresolved (n₂/n₃).")
    if file_bad: tips.append("Projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π.")
    cov_help = " ".join(tips) or "Generate coverage_sampling.csv with meta header."

    if st.button("Coverage Sample", key="btn_coverage_sample",
                 disabled=cov_disabled, help=cov_help):
        try:
            if n3 <= 0 or n2 <= 0:
                st.warning("Please ensure n₂ and n₃ are both > 0.")
            else:
                import csv, tempfile, os, random, uuid

                rng = random.Random(); rng.seed(seed_txt)

                # Stamp a run_id (non-destructive)
                run_id = (st.session_state.get("run_ctx") or {}).get("run_id") or str(uuid.uuid4())
                st.session_state.setdefault("run_ctx", {})["run_id"] = run_id

                counts: dict[str, int] = {}

                # lane_pattern used for quick in_district guess (bitstring)
                inputs_cov_tmp = _inputs_block_from_session((int(n2), int(n3)))
                lane_mask_bits = "".join("1" if int(x) else "0" for x in (inputs_cov_tmp.get("lane_mask_k3") or []))

                for _ in range(int(num_samples)):
                    d_k1 = _rand_gf2_matrix(int(n2), int(n3), float(bit_density), rng)
                    sig = _coverage_signature(d_k1, n_k=int(n3))
                    counts[sig] = counts.get(sig, 0) + 1

                total = float(num_samples)
                rows = []
                for sig, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    in_d = _in_district_guess(sig, current_lane_pattern=lane_mask_bits)
                    pct = 0.0 if total <= 0 else round(100.0 * (cnt / total), 2)
                    rows.append([sig, cnt, in_d, pct])

                # CSV meta header
                meta_lines = [
                    f"schema_version={SCHEMA_VERSION}",
                    f"saved_at={_utc_iso_z()}",
                    f"run_id={(st.session_state.get('run_ctx') or {}).get('run_id','')}",
                    f"app_version={APP_VERSION}",
                    f"seed={seed_txt}",
                    f"bit_density={bit_density}",
                    f"n2={n2}",
                    f"n3={n3}",
                ]
                header = ["signature", "count", "in_district", "pct"]

                # Write CSV (atomic)
                try:
                    _atomic_write_csv(COVERAGE_CSV_PATH, header, rows, meta_lines)  # shared helper
                except Exception:
                    with tempfile.NamedTemporaryFile("w", delete=False, dir=COVERAGE_CSV_PATH.parent,
                                                     encoding="utf-8", newline="") as tmp:
                        for line in meta_lines:
                            tmp.write(f"# {line}\n")
                        w = csv.writer(tmp); w.writerow(header); w.writerows(rows)
                        tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
                    os.replace(tmp_name, COVERAGE_CSV_PATH)

                st.success(f"Coverage CSV saved → {COVERAGE_CSV_PATH}")

                # Preview
                try:
                    import pandas as pd
                    preview = pd.DataFrame(rows[:30], columns=header)
                    st.dataframe(preview, use_container_width=True, hide_index=True)
                except Exception:
                    pass

                # Build JSON payload (1.1.0)
                try:
                    rc_cov = require_fresh_run_ctx()
                except Exception:
                    rc_cov = st.session_state.get("run_ctx") or {}

                # Normalize Π info into run_ctx just before policy build
                if "normalize_projector_into_run_ctx" in globals():
                    normalize_projector_into_run_ctx()

                policy_cov = _policy_block_from_run_ctx(rc_cov)
                inputs_cov = _inputs_block_from_session((int(n2), int(n3)))

                # Quick coherence heads-up (lane_mask vs n3)
                lm_bits_cov = "".join("1" if int(x) else "0" for x in inputs_cov.get("lane_mask_k3", []))
                if lm_bits_cov and len(lm_bits_cov) != int(n3):
                    st.caption(f"⚠︎ lane_mask_k3 has {len(lm_bits_cov)} bits but n₃={n3}")

                known_signatures = (rc_cov.get("known_signatures") or [])

                residual_method = "R3 strict vs R3·Π (lanes/ker/mixed)"

                results_cov = []
                for sig, cnt, in_d, pctv in rows:
                    results_cov.append({
                        "signature": sig,
                        "signature_canonical": sig,          # already canonicalized
                        "count": int(cnt),
                        "pct": float(pctv),
                        "in_district": bool(int(in_d)),
                        "lane_pattern": lane_mask_bits,       # bitstring for joins
                        "dims": {"n2": int(n2), "n3": int(n3)},
                    })

                payload_cov = {
                    "schema_version": SCHEMA_VERSION,
                    "written_at_utc": _utc_iso_z(),
                    "app_version": APP_VERSION,
                    "field": FIELD,
                    "identity": {
                        "run_id": (rc_cov.get("run_id") or ""),
                        "district_id": rc_cov.get("district_id","D3"),
                    },
                    "policy": policy_cov,
                    "inputs": inputs_cov,                    # includes hashes from SSOT
                    # top-level additions for quick readers
                    "lane_mask_k3": inputs_cov.get("lane_mask_k3", []),
                    "known_signatures": known_signatures,
                    "residual_method": residual_method,
                    # sampling block (kept for structured provenance)
                    "sampling": {
                        "num_samples": int(num_samples),
                        "bit_density": float(bit_density),
                        "seed": str(seed_txt),
                        "canonicalization": {
                            "columns": "top-to-bottom bits",
                            "sort": "lexicographic",
                            "domain": "d3 support",
                        },
                    },
                    "results": results_cov,
                    "summary": {
                        "N": int(num_samples),
                        "unique_signatures": int(len(results_cov)),
                        "in_district_hits": int(sum(1 for r in results_cov if r["in_district"])),

                        # pct across all samples (not unique sigs)
                        "pct_in_district": (
                            100.0 * sum(1 for r in results_cov if r["in_district"]) / float(len(results_cov))
                        ) if len(results_cov) else 0.0,
                    },
                    "integrity": {"content_hash": ""},
                }

                # Add top-3 signatures by count (+pct)
                top = sorted(rows, key=lambda r: (-int(r[1]), r[0]))[:3]
                payload_cov["summary"]["top_signatures"] = [
                    {"signature": s, "count": int(c), "pct": float(p)}
                    for (s, c, _, p) in top
                ]

                # Integrity (stable)
                payload_cov["integrity"]["content_hash"] = _hash_json(payload_cov)
                hash12_cov = payload_cov["integrity"]["content_hash"][:12]
                hash8_cov  = payload_cov["integrity"]["content_hash"][:8]
                cov_json_name = f"coverage_sampling__{hash12_cov}.json"
                cov_json_path = REPORTS_DIR / cov_json_name

                # Write JSON + remember paths
                try:
                    _atomic_write_json(cov_json_path, payload_cov)
                    st.session_state.setdefault("last_report_paths", {})["coverage_sampling"] = {
                        "csv": str(COVERAGE_CSV_PATH), "json": str(cov_json_path)
                    }
                except Exception as e:
                    st.warning(f"(Could not write coverage JSON: {e})")

                # Downloads (CSV + JSON) with hash-based keys
                try:
                    with open(COVERAGE_CSV_PATH, "rb") as fcsv:
                        st.download_button(
                            "Download coverage_sampling.csv",
                            fcsv,
                            file_name=f"coverage_sampling__{hash12_cov}.csv",
                            key=f"dl_cov_csv_{hash8_cov}",
                        )
                except Exception:
                    pass

                try:
                    import io as _io, json as _json
                    mem = _io.BytesIO(_json.dumps(payload_cov, ensure_ascii=False, indent=2).encode("utf-8"))
                    st.download_button(
                        "Download coverage_sampling.json",
                        mem,
                        file_name=cov_json_name,
                        key=f"dl_cov_json_{hash8_cov}",
                    )
                except Exception as e:
                    st.info(f"(Could not build coverage JSON download: {e})")

                # Badges
                st.info(f"wrote JSON ✓ · hash: {hash12_cov} · saved as {cov_json_name}")
                st.info(f"N={payload_cov['summary']['N']} · unique={payload_cov['summary']['unique_signatures']} · in-district={payload_cov['summary']['pct_in_district']:.2f}%")

        except Exception as e:
            st.error(f"Coverage sampling failed: {e}")




# =========================[ · Gallery Append & Dedupe (cert-required) ]=========================

GALLERY_PATH = (LOGS_DIR / "gallery.jsonl")
GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)

# Ensure the dedupe key matches spec
def gallery_key(row: dict) -> tuple:
    pol = row.get("policy") or {}
    h   = row.get("hashes") or {}
    return (
        str(row.get("district", "")),
        str(pol.get("policy_tag", "")),
        str(h.get("boundaries_hash", "")),
        str(h.get("C_hash", "")),
        str(h.get("H_hash", "")),
        str(h.get("U_hash", "")),
    )
            
    # Shared meta helper (define if missing)
    if "_std_meta" not in globals():
        def _std_meta(run_id=None):
            return {
                "schema_version": SCHEMA_VERSION,
                "written_at_utc": _utc_iso_z(),
                "app_version":    APP_VERSION,
                **({"run_id": run_id} if run_id else {}),
            }
    
    ss = st.session_state
    ss.setdefault("_gallery_keys", set())
    ss.setdefault("_gallery_bootstrapped", False)
    
    with safe_expander("Gallery"):
        rc   = ss.get("run_ctx") or {}
        cert = ss.get("cert_payload") or {}
    
        has_cert = bool(cert)
        if not has_cert:
            st.info("No cert in memory yet. Run Overlap (let cert writer emit) before adding to gallery.")
    
        # ---- Pull fields from the cert (authoritative SSOT)
        identity = (cert.get("identity") or {}) if has_cert else {}
        policy   = (cert.get("policy")   or {}) if has_cert else {}
        inputs   = (cert.get("inputs")   or {}) if has_cert else {}
    
        # HASHES SOURCE OF TRUTH: artifact_hashes
        artifacts   = (cert.get("artifact_hashes") or {}) if has_cert else {}
        inputs_hash = (inputs.get("hashes") or {}) if has_cert else {}
    
        h_boundaries = artifacts.get("boundaries_hash") or inputs_hash.get("boundaries_hash","")
        h_C          = artifacts.get("C_hash")          or inputs_hash.get("C_hash","")
        h_H          = artifacts.get("H_hash")          or inputs_hash.get("H_hash","")
        h_U          = artifacts.get("U_hash")          or inputs_hash.get("U_hash","")
        h_shapes     = artifacts.get("shapes_hash")     or inputs_hash.get("shapes_hash","") or h_U
    
        district_id    = identity.get("district_id", "UNKNOWN")
        policy_tag     = policy.get("policy_tag", rc.get("policy_tag", rc.get("mode", "strict")))
        projector_hash = policy.get("projector_hash", "")
        pj_consistent  = policy.get("projector_consistent_with_d", None)  # mirror exactly (true/false/None)
        cert_hash      = (cert.get("integrity") or {}).get("content_hash", "") or ""
    
        # ---- Optional UI fields
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            growth_bumps = st.number_input("growth_bumps", min_value=0, value=0, step=1, key="gal_growth_bumps")
        with c2:
            strictify = st.selectbox("strictify", options=["tbd", "no", "yes"], index=0, key="gal_strictify")
        with c3:
            tag = st.text_input("tag (optional)", value="", key="gal_tag")
    
        # ---- Row (meta first)
        row = {
            **_std_meta(run_id=(ss.get("run_ctx") or {}).get("run_id")),
            "district": district_id,
            "policy": {
                "policy_tag":     policy_tag,
                "projector_hash": projector_hash,
                # NEW: mirror projector_consistent_with_d into gallery rows for quick audits
                **({ "projector_consistent_with_d": bool(pj_consistent) } if pj_consistent is not None else {}),
            },
            "hashes": {
                "boundaries_hash": h_boundaries,
                "C_hash":          h_C,
                "H_hash":          h_H,
                "U_hash":          h_U,
                "shapes_hash":     h_shapes,
            },
            "growth_bumps":      int(growth_bumps),
            "strictify":         str(strictify),
            "tag":               tag or "",
            "cert_content_hash": cert_hash,
        }
    
        # ---- Dedupe key (district, policy_tag, B, C, H, U)
        try:
            key = gallery_key(row)
        except Exception:
            key = None
    
        # ---- Bootstrap dedupe cache from tail (once)
        if not ss["_gallery_bootstrapped"]:
            for tail_row in _read_jsonl_tail(GALLERY_PATH, N=200):
                try:
                    ss["_gallery_keys"].add(gallery_key(tail_row))
                except Exception:
                    continue
            ss["_gallery_bootstrapped"] = True
    
        # ---- Button (disabled only when no cert)
        tip = ("Append current cert to gallery.jsonl" if has_cert
               else "Disabled until a cert is written this run.")
        if st.button("Add to Gallery", key="btn_gallery_append", disabled=not has_cert, help=tip):
            try:
                if key is None:
                    st.warning("Could not compute dedupe key; skipping append.")
                elif key in ss["_gallery_keys"]:
                    st.info("Duplicate skipped (same district/policy/hashes).")
                else:
                    _atomic_append_jsonl(GALLERY_PATH, row)
                    ss["_gallery_keys"].add(key)
                    st.success("Gallery row appended.")
            except Exception as e:
                st.error(f"Gallery append failed: {e}")
    
        # ---- Tail view
        try:
            tail = _read_jsonl_tail(GALLERY_PATH, N=8)
            if tail:
                import pandas as pd
                view = []
                for r in tail:
                    view.append({
                        "when":      r.get("written_at_utc",""),
                        "district":  r.get("district",""),
                        "policy_tag": (r.get("policy") or {}).get("policy_tag",""),
                        "proj[:12]": ((r.get("policy") or {}).get("projector_hash","") or "")[:12],
                        "P_consist": (r.get("policy") or {}).get("projector_consistent_with_d", None),
                        "B[:8]":     (r.get("hashes") or {}).get("boundaries_hash","")[:8],
                        "C[:8]":     (r.get("hashes") or {}).get("C_hash","")[:8],
                        "H[:8]":     (r.get("hashes") or {}).get("H_hash","")[:8],
                        "U[:8]":     (r.get("hashes") or {}).get("U_hash","")[:8],
                        "strictify": r.get("strictify",""),
                        "tag":       r.get("tag",""),
                    })
                st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
            else:
                st.caption("Gallery is empty.")
        except Exception as e:
            st.warning(f"Could not render gallery tail: {e}")
    
    
    
    # =========================[ STEP 3 · Witness on Stubborn RED ]=========================
    
    WITNESS_PATH = (LOGS_DIR / "witnesses.jsonl")
    WITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    ss.setdefault("_witness_keys", set())
    ss.setdefault("_witness_bootstrapped", False)
    
    def witness_key(row: dict):
        h = row.get("hashes") or {}
        return (
            str(row.get("district","UNKNOWN")),
            str(row.get("reason","")),
            str(row.get("residual_tag","")),
            str((row.get("policy") or {}).get("policy_tag","")),
            str(h.get("boundaries_hash","")),
            str(h.get("C_hash","")),
            str(h.get("H_hash","")),
            str(h.get("U_hash","")),
        )
    
    with safe_expander("Witness logger"):
        # --- Freshness attempt; never st.stop() here ---
        rc, rc_ok = None, True
        try:
            rc = require_fresh_run_ctx()
            rc = rectify_run_ctx_mask_from_d3()
        except Exception:
            rc_ok = False
            st.info("Run Overlap first (witness logging will be disabled).")
    
        out = ss.get("overlap_out") or {}
        eq3 = bool(((out.get("3") or {}).get("eq", False))) if rc_ok else True  # default to not-eligible
        eligible_red = bool(rc_ok and (eq3 is False))
    
        # FILE Π invalid should only block in projected(file)
        def _file_pi_invalid_for_mode(_rc) -> bool:
            try:
                return (str((_rc or {}).get("mode","")) == "projected(file)") and file_validation_failed()
            except Exception:
                return False
    
        fm_bad = _file_pi_invalid_for_mode(rc)
        help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."
    
        # Residual tag selection (strict vs projected)
        tags = ss.get("residual_tags") or {}
        mode = str((rc or {}).get("mode","strict"))
        residual_tag_val = tags.get("projected" if mode.startswith("projected") else "strict", "none")
    
        # Cert optional (for content hash)
        cert = ss.get("cert_payload") or {}
        cert_hash = (cert.get("integrity") or {}).get("content_hash","") or ""
    
        # Canonical hashes (prefer nested)
        inputs = cert.get("inputs", {}) or {}
        hashes = inputs.get("hashes") or {
            "boundaries_hash": inputs.get("boundaries_hash",""),
            "C_hash":          inputs.get("C_hash",""),
            "H_hash":          inputs.get("H_hash",""),
            "U_hash":          inputs.get("U_hash",""),
            "shapes_hash":     inputs.get("shapes_hash",""),
        }
    
        identity = cert.get("identity", {}) or {}
        district_id = identity.get("district_id", "UNKNOWN")
        policy = cert.get("policy", {}) or {}
        policy_tag = policy.get("policy_tag", (rc or {}).get("policy_tag",""))
        projector_hash = policy.get("projector_hash","") or ((rc or {}).get("projector_hash","") or "")
    
        # UI controls
        c1, c2 = st.columns([1,3])
        with c1:
            reason = st.selectbox(
                "reason",
                options=["lanes-persist","policy-mismatch","needs-new-R","grammar-drift","other"],
                index=0,
                key="w_reason",
            )
        with c2:
            note = st.text_input("note (optional)", value="", key="w_note")
    
        # Build row per spec (meta first)
        row = {
            **_std_meta(run_id=(st.session_state.get("run_ctx") or {}).get("run_id")),
            "district":       district_id,
            "reason":         reason,
            "residual_tag":   residual_tag_val,
            "policy": {
                "policy_tag":     policy_tag,
                "projector_hash": projector_hash,
            },
            "hashes": {
                "boundaries_hash": hashes.get("boundaries_hash",""),
                "C_hash":          hashes.get("C_hash",""),
                "H_hash":          hashes.get("H_hash",""),
                "U_hash":          hashes.get("U_hash",""),
                "shapes_hash":     hashes.get("shapes_hash",""),
            },
            "cert_content_hash": cert_hash,
            "note":              note or "",
        }
    
        # Bootstrap dedupe cache from tail once
        if not ss["_witness_bootstrapped"]:
            for tail_row in _read_jsonl_tail(WITNESS_PATH, N=200):
                try:
                    ss["_witness_keys"].add(witness_key(tail_row))
                except Exception:
                    continue
            ss["_witness_bootstrapped"] = True
    
        k = witness_key(row)
        disabled = bool(fm_bad or (not eligible_red))
        tip = help_txt if fm_bad else (None if eligible_red else "Enabled only when k=3 is RED (eq=False).")
    
        # --- Button: Log Witness (strict-aware gating) ---
        if st.button("Log Witness", key="btn_witness_append", disabled=disabled, help=(tip or "Append witness to witnesses.jsonl")):
            try:
                if k in ss["_witness_keys"]:
                    st.info("Duplicate skipped (same district/reason/tag/policy/hashes).")
                else:
                    _atomic_append_jsonl(WITNESS_PATH, row)
                    ss["_witness_keys"].add(k)
                    st.success("Witness logged.")
            except Exception as e:
                st.error(f"Witness append failed: {e}")
    
        # --- Tail view ---
        try:
            tail = _read_jsonl_tail(WITNESS_PATH, N=8)
            if tail:
                import pandas as pd
                view = []
                for r in tail:
                    view.append({
                        "when":       r.get("written_at_utc",""),
                        "district":   r.get("district",""),
                        "reason":     r.get("reason",""),
                        "tag":        r.get("residual_tag",""),
                        "policy_tag": (r.get("policy") or {}).get("policy_tag",""),
                        "proj[:12]":  ((r.get("policy") or {}).get("projector_hash","") or "")[:12],
                        "B[:8]":      (r.get("hashes") or {}).get("boundaries_hash","")[:8],
                        "C[:8]":      (r.get("hashes") or {}).get("C_hash","")[:8],
                        "H[:8]":      (r.get("hashes") or {}).get("H_hash","")[:8],
                        "U[:8]":      (r.get("hashes") or {}).get("U_hash","")[:8],
                    })
                st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
            else:
                st.caption("No witnesses logged yet.")
        except Exception as e:
            st.warning(f"Could not render witnesses tail: {e}")
    
    # ======================= end JSONL











# -------------------------- FREEZER HELPERS -------------------------------------------

PROJECTORS_DIR = Path("projectors")
PROJECTORS_DIR.mkdir(parents=True, exist_ok=True)
PJ_REG_PATH = PROJECTORS_DIR / "projector_registry.jsonl"

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as tmp:
        tmp.write(blob); tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    os.replace(tmp_name, path)
    return _sha256_bytes(blob), len(blob)

def _append_registry_row(row: dict) -> bool:
    # in-session dedupe based on (district, projector_hash)
    key = (row.get("district", ""), row.get("projector_hash", ""))
    seen = st.session_state.setdefault("_pj_registry_keys", set())
    if key in seen:
        return False
    seen.add(key)
    PROJECTORS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=PROJECTORS_DIR, encoding="utf-8") as tmp:
        tmp.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    with open(PJ_REG_PATH, "a", encoding="utf-8") as final, open(tmp_name, "r", encoding="utf-8") as src:
        shutil.copyfileobj(src, final)
    os.remove(tmp_name)
    return True

def _diag_projector_from_lane_mask(lm: list[int]) -> list[list[int]]:
    n = len(lm or [])
    return [[1 if (i == j and int(lm[j]) == 1) else 0 for j in range(n)] for i in range(n)]

def _freeze_projector(*, district_id: str, lane_mask_k3: list[int], filename_hint: str | None = None) -> dict:
    if not lane_mask_k3:
        raise ValueError("No lane mask available (run Overlap first).")
    P3 = _diag_projector_from_lane_mask(lane_mask_k3)
    name = filename_hint or f"projector_{district_id or 'UNKNOWN'}.json"
    pj_path = PROJECTORS_DIR / name
    payload = {"schema_version": "1.0.0", "written_at_utc": _utc_iso(), "blocks": {"3": P3}}
    pj_hash, pj_size = _atomic_write_json(pj_path, payload)
    return {"path": str(pj_path), "projector_hash": pj_hash, "bytes": pj_size, "lane_mask_k3": lane_mask_k3[:]}

def _validate_projector_file(pj_path: str) -> dict:
    # Use same validator used by overlap (raises ValueError with P3_* on fail)
    cfg_file = _cfg_from_policy("projected(file)", pj_path)
    _, meta = projector_choose_active(cfg_file, boundaries)
    return meta

def _simulate_overlap_with_cfg(cfg_forced: dict):
    """
    Run a FILE overlap without touching the policy widget. Populates:
      run_ctx, overlap_out, residual_tags, overlap_policy_label
    """
    P_active, meta = projector_choose_active(cfg_forced, boundaries)

    # Context
    d3 = meta.get("d3") if "d3" in meta else (boundaries.blocks.__root__.get("3") or [])
    n3 = meta.get("n3") if "n3" in meta else (len(d3[0]) if (d3 and d3[0]) else 0)
    lane_mask = meta.get("lane_mask", [])
    mode = meta.get("mode", "projected(file)")

    # Compute strict residual R3 = H2 @ d3 XOR (C3 XOR I3)
    H_used = st.session_state.get("overlap_H") or _load_h_local()
    H2 = (H_used.blocks.__root__.get("2") or [])
    C3 = (cmap.blocks.__root__.get("3") or [])
    I3 = eye(len(C3)) if C3 else []

    def _xor(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]

    def _is_zero(M): return (not M) or all(all((x & 1) == 0 for x in row) for row in M)

    R3_strict = _xor(mul(H2, d3), _xor(C3, I3)) if (H2 and d3 and C3) else []
    R3_proj   = mul(R3_strict, P_active) if (R3_strict and P_active) else []

    def _residual_tag(R, lm):
        if not R or not lm: return "none"
        rows = len(R)
        lanes_idx = [j for j, m in enumerate(lm) if m]
        ker_idx   = [j for j, m in enumerate(lm) if not m]
        def _col_nz(j): return any(R[i][j] & 1 for i in range(rows))
        lanes = any(_col_nz(j) for j in lanes_idx) if lanes_idx else False
        ker   = any(_col_nz(j) for j in ker_idx)   if ker_idx   else False
        if not lanes and not ker: return "none"
        if lanes and not ker:     return "lanes"
        if ker and not lanes:     return "ker"
        return "mixed"

    tag_strict = _residual_tag(R3_strict, lane_mask)
    tag_proj   = _residual_tag(R3_proj,   lane_mask)

    out = {"3": {"eq": bool(_is_zero(R3_proj)), "n_k": n3}, "2": {"eq": True}}

    st.session_state["overlap_out"] = out
    st.session_state["residual_tags"] = {"strict": tag_strict, "projected": tag_proj}
    st.session_state["overlap_cfg"] = cfg_forced
    st.session_state["overlap_policy_label"] = policy_label_from_cfg(cfg_forced)
    st.session_state["run_ctx"] = {
        "policy_tag": policy_label_from_cfg(cfg_forced),
        "mode": mode,
        "d3": d3, "n3": n3, "lane_mask_k3": lane_mask,
        "P_active": P_active,
        "projector_filename": meta.get("projector_filename", ""),
        "projector_hash": meta.get("projector_hash", ""),
        "projector_consistent_with_d": meta.get("projector_consistent_with_d", None),
        "errors": [],
    }

# ---------------------------- Projector Freezer (AUTO → FILE, no UI flip) ----------------------------
with st.expander("Projector Freezer (AUTO → FILE, no UI flip)"):
    _ss = st.session_state
    _di = _ss.get("_district_info") or {}
    district_id = _di.get("district_id", "UNKNOWN")

    # Freshness + rectifier (stops if stale)
    _rectify_run_ctx_mask_from_d3_or_stop()
    rc = _ss.get("run_ctx") or {}  # fetch AFTER rectifier
    n3 = int(rc.get("n3") or 0)
    lm = list(rc.get("lane_mask_k3") or [])
    if n3 <= 0 or len(lm) != n3:
        st.warning("Context invalid (n3/mask mismatch). Click Run Overlap and try again.")
        st.stop()

    # Eligibility: AUTO mode + k=3 green + SSOT present
    k3_green = bool(((_ss.get("overlap_out") or {}).get("3", {}) or {}).get("eq", False))
    elig_freeze = (rc.get("mode") == "projected(auto)" and bool(rc.get("d3")) and bool(lm) and k3_green)

    st.caption("Freeze current AUTO Π → file, switch to FILE, re-run Overlap, and force a cert write.")

    # Inputs
    pj_basename   = st.text_input("Filename", value=f"projector_{district_id or 'UNKNOWN'}.json",
                                  key="pj_freeze_name_final2")
    overwrite_ok  = st.checkbox("Overwrite if exists", value=False, key="pj_freeze_overwrite_final2")

    # Global FILE Π invalid gate (as requested)
    fm_bad  = file_validation_failed()
    help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    # Final disabled state + tooltip
    disabled = fm_bad or (not elig_freeze)
    tip = help_txt if fm_bad else (None if elig_freeze else "Enabled when current run is projected(auto) and k=3 is green.")

    # --- Button: Freeze Π → FILE & re-run ---
    if st.button("Freeze Π → FILE & re-run",
                 key="btn_freeze_final2",
                 disabled=disabled,
                 help=(tip or "Freeze AUTO to FILE and re-run")):
        try:
            # Freshness + rectifier again
            _rectify_run_ctx_mask_from_d3_or_stop()
            rc = _ss.get("run_ctx") or {}
            n3 = int(rc.get("n3") or 0)
            lm = list(rc.get("lane_mask_k3") or [])
            if n3 <= 0 or len(lm) != n3:
                st.warning("Context invalid (n3/mask mismatch). Click Run Overlap and try again.")
                st.stop()

            # Build Π from SSOT lane mask
            P_freeze = [[1 if (i == j and lm[j]) else 0 for j in range(n3)] for i in range(n3)]

            # Validate strictly
            validate_projector_file_strict(P_freeze, n3=n3, lane_mask=lm)

            # Save projector (atomic)
            pj_path = PROJECTORS_DIR / pj_basename
            if pj_path.exists() and not overwrite_ok:
                st.warning("Projector file already exists. Enable 'Overwrite if exists' or choose a new name.")
                st.stop()
            payload = {"schema_version": "1.0.0", "blocks": {"3": P_freeze}}
            pj_hash, _ = _atomic_write_json(pj_path, payload)

            # Switch policy to FILE (k=3) and mark fixtures changed (nonce bump)
            cfg_active.setdefault("source", {})["3"] = "file"
            cfg_active.setdefault("projector_files", {})["3"] = pj_path.as_posix()
            if "_mark_fixtures_changed" in globals():
                _mark_fixtures_changed()
            else:
                _ss["_fixture_nonce"] = int(_ss.get("_fixture_nonce", 0)) + 1
                for k in ("overlap_out", "residual_tags", "overlap_cfg", "overlap_policy_label"):
                    _ss.pop(k, None)

            # Optional: append registry row
            try:
                _append_registry_row({
                    "schema_version": "1.0.0",
                    "written_at_utc": _utc_iso(),
                    "app_version": APP_VERSION,
                    "district": district_id,
                    "lane_mask_k3": lm,
                    "filename": pj_path.as_posix(),
                    "projector_hash": pj_hash,
                })
            except Exception:
                pass

            # Re-run Overlap immediately (fresh FILE mode)
            if "_soft_reset_before_overlap" in globals():
                _soft_reset_before_overlap()
            run_overlap()

            # Force cert write this pass (bypass debounce)
            _ss["should_write_cert"] = True
            _ss.pop("_last_cert_write_key", None)

            st.success(f"Π saved → {pj_path.name} · {pj_hash[:12]}… and switched to FILE.")
        except Exception as e:
            st.error(f"Freeze failed: {e}")


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

# --------- Define eye() if missing --------
if "eye" not in globals():
    def eye(n: int):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

# --- Robust hashing helpers (drop near your other helpers)

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
    # Last resort: repr (keeps us from crashing; still deterministic enough)
    return repr(x)

def _hash_fixture_side(fx: dict) -> dict:
    """Hash each sub-object of a fixture side robustly."""
    return {
        "boundaries": _hash_obj(_to_hashable_plain(fx.get("boundaries"))),
        "shapes":     _hash_obj(_to_hashable_plain(fx.get("shapes"))),
        "cmap":       _hash_obj(_to_hashable_plain(fx.get("cmap"))),
        "H":          _hash_obj(_to_hashable_plain(fx.get("H"))),
    }

# --- GF(2) helpers (you already have _mul_gf2 / _xor_mat; keep those) ---
def _all_zero_mat(M: list[list[int]]) -> bool:
    return not M or all((x & 1) == 0 for row in M for x in row)

def _I(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def _lane_mask_from_boundaries(boundaries_obj) -> list[int]:
    # d3 is 2<-3 block; rows = n2, cols = n3
    d3 = (boundaries_obj or {}).blocks.__root__.get("3") or []
    if not d3: 
        return []
    rows, cols = len(d3), len(d3[0])
    return [1 if any(d3[i][j] & 1 for i in range(rows)) else 0 for j in range(cols)]

def _lane_mask_pair_SSO(L_boundaries, R_boundaries) -> list[int]:
    # SSOT per pair: OR the left/right masks (robust if they differ)
    Lm = _lane_mask_from_boundaries(L_boundaries)
    Rm = _lane_mask_from_boundaries(R_boundaries)
    if not Lm and not Rm:
        return []
    if not Lm: return Rm[:]
    if not Rm: return Lm[:]
    n = max(len(Lm), len(Rm))
    Lm += [0]*(n-len(Lm))
    Rm += [0]*(n-len(Rm))
    return [1 if (Lm[j] or Rm[j]) else 0 for j in range(n)]

def _diag_from_mask(mask: list[int]) -> list[list[int]]:
    n = len(mask or [])
    return [[(mask[i] & 1) if i == j else 0 for j in range(n)] for i in range(n)]

def _r3_from_fixture(fx: dict) -> list[list[int]]:
    # R3 = H2 @ d3 + (C3 + I3)  over GF(2)
    B = (fx.get("boundaries") or {}).blocks.__root__
    C = (fx.get("cmap") or {}).blocks.__root__
    H = (fx.get("H") or {}).blocks.__root__
    d3 = B.get("3") or []          # shape: n2 x n3
    H2 = H.get("2") or []          # shape: n3 x n2
    C3 = C.get("3") or []          # shape: n3 x n3
    I3 = _I(len(C3)) if C3 else []
    term1 = _mul_gf2(H2, d3)       # (n3 x n2) * (n2 x n3) -> (n3 x n3)
    term2 = _xor_mat(C3, I3)       # (n3 x n3)
    return _xor_mat(term1, term2)  # (n3 x n3)

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





# -------------- Helper: _hash_obj --------------
def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _hash_obj(obj) -> str:
    try:
        blob = _json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return _sha256_hex(blob)
    except Exception:
        return ""

def validate_pairs_payload(payload: dict) -> tuple[list[dict], str]:
    """
    Validate and sanitize a parity pairs payload.
    Returns (pairs, policy_hint) or raises ValueError("PARITY_SCHEMA_INVALID: ...").
    - schema_version must be "1.0.0"
    - policy_hint in {"strict","projected:auto","projected:file","mirror_active"}
    - Each pair: {"label", "left", "right"}
    - Each side is EITHER {"embedded": {boundaries, shapes, cmap, H}} OR {boundaries, shapes, cmap, H} (paths)
    - If embedded present -> it wins; path keys are stripped. Unknown keys are rejected.
    """
    def _err(msg: str) -> ValueError:
        return ValueError(f"PARITY_SCHEMA_INVALID: {msg}")

    if not isinstance(payload, dict):
        raise _err("root must be an object")

    # --- root keys ---
    allowed_root = {"schema_version", "policy_hint", "pairs"}
    unknown_root = sorted(set(payload.keys()) - allowed_root)
    if unknown_root:
        raise _err(f"unknown root keys: {unknown_root}")

    sv = payload.get("schema_version")
    if sv != "1.0.0":
        raise _err(f"schema_version must be '1.0.0' (got {sv!r})")

    ph = payload.get("policy_hint")
    if ph not in {"strict", "projected:auto", "projected:file", "mirror_active"}:
        raise _err(f"policy_hint invalid (got {ph!r})")

    pairs_in = payload.get("pairs")
    if not isinstance(pairs_in, list):
        raise _err("pairs must be an array")

    def _sanitize_side(side: dict, *, where: str) -> dict:
        if not isinstance(side, dict):
            raise _err(f"{where} side must be an object")
        # Whitelist for both modes
        allowed_path = {"boundaries", "shapes", "cmap", "H"}
        allowed_emb  = {"embedded"}
        keys = set(side.keys())

        has_emb = "embedded" in side
        if has_emb:
            # Embedded mode
            unknown = sorted(keys - allowed_emb)
            if unknown:
                raise _err(f"{where}: unknown keys {unknown} (embedded mode)")
            emb = side["embedded"]
            if not isinstance(emb, dict):
                raise _err(f"{where}: embedded must be an object")
            required = {"boundaries", "shapes", "cmap", "H"}
            missing = sorted(required - set(emb.keys()))
            if missing:
                raise _err(f"{where}: embedded missing keys {missing}")
            # Type-check embedded objects (loose: must be dicts)
            for k in required:
                if not isinstance(emb[k], dict):
                    raise _err(f"{where}: embedded.{k} must be an object")
            # Embedded wins → return only {"embedded": normalized_emb}
            # Also strip any accidental sibling path keys if they existed.
            return {"embedded": {k: emb[k] for k in ("boundaries","shapes","cmap","H")}}
        else:
            # Path mode
            unknown = sorted(keys - allowed_path)
            if unknown:
                raise _err(f"{where}: unknown keys {unknown} (path mode)")
            missing = sorted([k for k in ("boundaries","shapes","cmap","H")
                              if k not in side or not isinstance(side[k], str) or not side[k].strip()])
            if missing:
                raise _err(f"{where}: missing or empty path(s) {missing}")
            # Normalize to exact four strings
            return {
                "boundaries": side["boundaries"].strip(),
                "shapes":     side["shapes"].strip(),
                "cmap":       side["cmap"].strip(),
                "H":          side["H"].strip(),
            }

    out_pairs: list[dict] = []
    for idx, p in enumerate(pairs_in):
        if not isinstance(p, dict):
            raise _err(f"pair[{idx}] must be an object")
        allowed_pair = {"label", "left", "right"}
        unknown_pair = sorted(set(p.keys()) - allowed_pair)
        if unknown_pair:
            raise _err(f"pair[{idx}]: unknown keys {unknown_pair}")

        label = p.get("label")
        if not isinstance(label, str) or not label.strip():
            raise _err(f"pair[{idx}]: label must be non-empty string")

        if "left" not in p or "right" not in p:
            raise _err(f"pair[{idx}]: missing 'left' or 'right'")
        left  = _sanitize_side(p["left"],  where=f"pair[{idx}]/left")
        right = _sanitize_side(p["right"], where=f"pair[{idx}]/right")

        out_pairs.append({"label": label.strip(), "left": left, "right": right})

    return out_pairs, ph

#------------------------------------------------------------
def _path_exists_strict(p: str) -> bool:
    try:
        return Path(p).exists() and Path(p).is_file()
    except Exception:
        return False

def _pp_resolve_side_or_skip(side: dict, *, label: str, side_name: str):
    # embedded wins
    if "embedded" in side:
        return ("ok", {"embedded": side["embedded"]})
    # require all 4 paths
    missing = [k for k in ("boundaries","shapes","cmap","H") if not side.get(k)]
    if missing:
        return ("skip", {"label": label, "side": side_name, "missing": missing, "error": "PARITY_SPEC_MISSING"})
    # strict existence check
    miss_files = [k for k in ("boundaries","shapes","cmap","H") if not _path_exists_strict(side[k])]
    if miss_files:
        return ("skip", {"label": label, "side": side_name, "missing": miss_files, "error": "PARITY_FILE_NOT_FOUND"})
    return ("ok", {"paths": side})

def _lane_mask_from_boundaries(boundaries_obj) -> list[int]:
    d3 = (boundaries_obj or {}).blocks.__root__.get("3->2") or (boundaries_obj or {}).blocks.__root__.get("3")  # depending on your schema
    if not d3: return []
    rows, n3 = len(d3), len(d3[0])
    return [1 if any(d3[i][j] & 1 for i in range(rows)) else 0 for j in range(n3)]



def _pair_hash(left_hashes: dict, right_hashes: dict) -> str:
    tup = (
        left_hashes["boundaries"], left_hashes["shapes"], left_hashes["cmap"], left_hashes["H"],
        right_hashes["boundaries"], right_hashes["shapes"], right_hashes["cmap"], right_hashes["H"],
    )
    return _sha256_hex(_json.dumps(tup, separators=(",", ":"), sort_keys=False).encode("utf-8"))



# -------------- Core: JSON helpers --------------
def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _atomic_write_json(path: Path, payload: dict) -> None:
    _ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        _json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        os.replace(tmp.name, path)

def _safe_parse_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No file at {path}")
    with p.open("r", encoding="utf-8") as f:
        return _json.load(f)

# -------- Report paths (guard) --------
from pathlib import Path

if "REPORTS_DIR" not in globals():
    REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if "PARITY_JSON_PATH" not in globals():
    PARITY_JSON_PATH = REPORTS_DIR / "parity_report.json"

if "PARITY_CSV_PATH" not in globals():
    PARITY_CSV_PATH = REPORTS_DIR / "parity_summary.csv"

        
# --------- Paths & Directory init ----------
REPORTS_DIR = Path(globals().get("REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if "PARITY_REPORT_PATH" not in globals():
    PARITY_REPORT_PATH = REPORTS_DIR / "parity_report.json"
if "PARITY_SUMMARY_CSV" not in globals():
    PARITY_SUMMARY_CSV = REPORTS_DIR / "parity_summary.csv"

PARITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

if "DEFAULT_PARITY_PATH" not in globals():
    DEFAULT_PARITY_PATH = LOGS_DIR / "parity_pairs.json"

# --------- Helper: time & session -------------
def __pp_now_z():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def __pp_current_input_filenames():
    ib = st.session_state.get("_inputs_block") or {}
    fns = ib.get("filenames") or {}
    return {
        "boundaries": fns.get("boundaries", "inputs/boundaries.json"),
        "cmap": fns.get("C", "inputs/cmap.json"),
        "H": fns.get("H", "inputs/H.json"),
        "shapes": fns.get("U", "inputs/shapes.json"),
    }

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

# ---- queue → portable JSON payload
def __pp_pairs_payload_from_queue(pairs: list[dict]) -> dict:
    def _paths_anyshape(fx: dict) -> dict:
        # prefer your universal adapter; fallback to SSOT file names
        try:
            return _paths_from_fixture_or_current(fx)
        except Exception:
            ib = st.session_state.get("_inputs_block") or {}
            fns = ib.get("filenames") or {}
            return {
                "boundaries": fx.get("boundaries", fns.get("boundaries", "inputs/boundaries.json")),
                "cmap":       fx.get("cmap",       fns.get("C",          "inputs/cmap.json")),
                "H":          fx.get("H",          fns.get("H",          "inputs/H.json")),
                "shapes":     fx.get("shapes",     fns.get("U",          "inputs/shapes.json")),
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
    return p.as_posix()

# --- Parity queue shims (safe to re-define if missing)
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
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = __pp_pairs_payload_from_queue(st.session_state.get("parity_pairs", []) or [])
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    return p.as_posix()

# ---- import helper for the uploader (payload-based)
def _import_pairs_from_payload(payload: dict, *, merge: bool) -> int:
    if not merge:
        clear_parity_pairs()
    for r in (payload.get("pairs") or []):
        L, R = r.get("left") or {}, r.get("right") or {}
        fxL = load_fixture_from_paths(boundaries_path=L["boundaries"], cmap_path=L["cmap"], H_path=L["H"], shapes_path=L["shapes"])
        fxR = load_fixture_from_paths(boundaries_path=R["boundaries"], cmap_path=R["cmap"], H_path=R["H"], shapes_path=R["shapes"])
        add_parity_pair(label=r.get("label","PAIR"), left_fixture=fxL, right_fixture=fxR)
    return len(st.session_state.get("parity_pairs", []))


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
    ["strict", "projected(auto)", "projected(file)"],
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
    3) Mirror app run_ctx.mode ("strict" / "projected(auto)" / "projected(file)")
    Returns: (mode, submode) where mode in {"strict","projected"} and submode in {"","auto","file"}.
    """
    # 1) UI radio (if you have one)
    choice = st.session_state.get("parity_policy_choice")
    if choice == "strict":
        return ("strict", "")
    if choice == "projected(auto)":
        return ("projected", "auto")
    if choice == "projected(file)":
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
    if mode == "projected(auto)":
        return ("projected","auto")
    if mode == "projected(file)":
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
        _atomic_write_json(json_path, report)
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
with cC:
    if st.button("Insert defaults · Smoke (inline)", key="pp_preset_smoke"):
        _insert_preset_payload({
            "schema_version": "1.0.0",
            "policy_hint": "projected:auto",
            "pairs": [
                {
                    "label": "SELF • ker-only vs ker-only (D3 dims 2×3)",
                    "left":  {
                        "embedded": {
                            "boundaries": {
                                "name": "D3 dims",
                                "blocks": {
                                    "3": [[1,1,0],[0,1,0]]
                                }
                            },
                            "shapes":     { "n": { "2": 2, "3": 3 } },
                            "cmap":       {
                                "name": "C3 ker-only; C2=I",
                                "blocks": {
                                    "3": [[1,0,0],[0,1,0],[0,0,0]],
                                    "2": [[1,0],[0,1]]
                                }
                            },
                            "H":          {
                                "name": "H=0",
                                "blocks": {
                                    "2": [[0,0],[0,0],[0,0]]
                                }
                            }
                        }
                    },
                    "right": {
                        "embedded": {
                            "boundaries": {
                                "name": "D3 dims",
                                "blocks": {
                                    "3": [[1,1,0],[0,1,0]]
                                }
                            },
                            "shapes":     { "n": { "2": 2, "3": 3 } },
                            "cmap":       {
                                "name": "C3 ker-only; C2=I",
                                "blocks": {
                                    "3": [[1,0,0],[0,1,0],[0,0,0]],
                                    "2": [[1,0],[0,1]]
                                }
                            },
                            "H":          {
                                "name": "H=0",
                                "blocks": {
                                    "2": [[0,0],[0,0],[0,0]]
                                }
                            }
                        }
                    }
                }
            ]
        }, name="Smoke (inline)")

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





# --------- UI for import/export (keep only one expander) ---------
with st.expander("Parity pairs: import/export"):
    colA, colB, colC = st.columns([3, 3, 2])
    with colA:
        export_path_txt = st.text_input(
            "Export path",
            value=str(DEFAULT_PARITY_PATH),
            key="pp_export_path",
            help="Path for saving pairs JSON."
        )
    with colB:
        import_path_txt = st.text_input(
            "Import path",
            value=str(DEFAULT_PARITY_PATH),
            key="pp_import_path",
            help="Path to load pairs JSON or use uploader."
        )
    with colC:
        merge_load = st.checkbox("Merge on import", value=False, key="pp_merge")

    # Uploader
    up_col1, up_col2 = st.columns([2, 3])
    with up_col1:
        uploaded_json = st.file_uploader("Import via upload", type=["json"], key="pp_uploader")
    with up_col2:
        st.caption("Tip: upload OR type a path; upload wins if both used.")

    # Export button
    c1, c2 = st.columns(2)
    with c1:
        disabled_export = _file_mode_invalid_now()
        help_export = "Disabled due to validation failure." if disabled_export else "Write pairs to JSON and download."
        if st.button("Export parity_pairs.json", key="pp_do_export", disabled=disabled_export, help=help_export):
            try:
                out_path = _export_pairs_to_path(export_path_txt)
                st.success(f"Saved parity pairs → {out_path}")
                payload = __pp_pairs_payload_from_queue(st.session_state.get("parity_pairs", []))
                mem = _io.BytesIO(_json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
                st.download_button("Download exported parity_pairs.json", mem, file_name=Path(out_path).name, key="dl_ppairs_json")
            except Exception as e:
                st.error(f"Export failed: {e}")

    # Import button
    with c2:
        disabled_import = _file_mode_invalid_now()
        help_import = "Disabled due to validation failure." if disabled_import else "Load pairs from JSON or path."
        if st.button("Import parity_pairs.json", key="pp_do_import", disabled=disabled_import, help=help_import):
            try:
                if uploaded_json:
                    payload = _json.loads(uploaded_json.getvalue().decode("utf-8"))
                    n = _import_pairs_from_payload(payload, merge=merge_load)
                    st.success(f"Loaded {n} pairs from uploaded file")
                else:
                    path_str = _ensure_json_path_str(import_path_txt)
                    p = Path(path_str)
                    if not (p.exists() and p.is_file()):
                        raise FileNotFoundError(f"No file at {path_str}")
                    with open(p, "r", encoding="utf-8") as f:
                        payload = _json.load(f)
                    n = _import_pairs_from_payload(payload, merge=merge_load)
                    st.success(f"Loaded {n} pairs from {path_str}")
            except Exception as e:
                st.error(f"Import failed: {e}")

# --------- END -----------------

 

# ------------------------ Cert writer (central, SSOT-only, with A/B embed) ------------------------
st.divider()
st.caption("Cert & provenance")

from pathlib import Path
import platform, os, json as _json
from datetime import datetime, timezone
import hashlib

# Ensure bundles directory exists
BUNDLES_DIR = Path(globals().get("BUNDLES_DIR", "bundles"))
BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

LAB_SCHEMA_VERSION = "1.0.0"

def _py_version_str() -> str:
    return f"python-{platform.python_version()}"

# Fallback: inputs signature from SSOT (kept local so this block is self-contained)
def _inputs_sig_now() -> list[str]:
    _ib_local = st.session_state.get("_inputs_block") or {}
    return [
        str(_ib_local.get("boundaries_hash","")),
        str(_ib_local.get("C_hash","")),
        str(_ib_local.get("H_hash","")),
        str(_ib_local.get("U_hash","")),
        str(_ib_local.get("shapes_hash","") or _ib_local.get("U_hash","")),
    ]

# Mode-aware FILE Π invalid helper (must exist globally; provide safe fallback)
def _file_mode_invalid_now() -> bool:
    rc = st.session_state.get("run_ctx") or {}
    try:
        return (str(rc.get("mode")) == "projected(file)") and bool(file_validation_failed())
    except Exception:
        return False

# --- Helper functions ---
def _assert_cert_invariants(cert: dict) -> None:
    must = ("identity","policy","inputs","diagnostics","checks","signatures","residual_tags","promotion","artifact_hashes")
    for key in must:
        if key not in cert:
            raise ValueError(f"CERT_INVAR:key-missing:{key}")
    ident = cert["identity"] or {}; policy = cert["policy"] or {}; inputs = cert["inputs"] or {}
    checks = cert["checks"] or {}; arts = cert["artifact_hashes"] or {}
    for k in ("district_id","run_id","timestamp"):
        if not str(ident.get(k,"")).strip():
            raise ValueError(f"CERT_INVAR:identity-missing:{k}")
    for k in ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash"):
        if not isinstance(inputs.get(k,""), str) or inputs.get(k,"")=="":
            raise ValueError(f"CERT_INVAR:inputs-hash-missing:{k}")
    for k in ("boundaries_hash","C_hash","H_hash","U_hash"):
        if arts.get(k,"") != inputs.get(k,""):
            raise ValueError(f"CERT_INVAR:artifact-hash-mismatch:{k}")
    dims = inputs.get("dims") or {}
    if not (isinstance(dims.get("n2"), int) and isinstance(dims.get("n3"), int)):
        raise ValueError("CERT_INVAR:inputs-dims-missing:n2-n3")
    ptag = str(policy.get("policy_tag") or policy.get("label") or "").strip()
    if not ptag: raise ValueError("CERT_INVAR:policy-tag-missing")
    is_strict = (ptag == "strict")
    is_file   = ptag.startswith("projected(file)")
    is_auto   = ptag.startswith("projected(auto)")
    kg = checks.get("ker_guard", "")
    if is_strict and kg != "enforced":
        raise ValueError("CERT_INVAR:ker-guard-should-be-enforced-for-strict")
    if (is_file or is_auto) and kg != "off":
        raise ValueError("CERT_INVAR:ker-guard-should-be-off-for-projected")
    pj_hash = policy.get("projector_hash", "")
    pj_file = policy.get("projector_filename", "") or ""
    pj_cons = policy.get("projector_consistent_with_d", None)
    if is_strict and (pj_file or pj_hash or (pj_cons is True)):
        raise ValueError("CERT_INVAR:strict-must-not-carry-projector-fields")
    if is_file:
        if not pj_file: raise ValueError("CERT_INVAR:file-mode-missing-projector_filename")
        if pj_cons is not True: raise ValueError("CERT_INVAR:file-mode-projector-not-consistent")
        if not isinstance(pj_hash, str) or pj_hash == "":
            raise ValueError("CERT_INVAR:file-mode-missing-projector_hash")
    if is_auto and pj_file:
        raise ValueError("CERT_INVAR:auto-mode-should-not-carry-projector_filename")

# --- SSOT reads ---
_rc  = st.session_state.get("run_ctx") or {}
_out = st.session_state.get("overlap_out") or {}
_ib  = st.session_state.get("_inputs_block") or {}
_di  = st.session_state.get("_district_info") or {}
_H   = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})

# --- Hard-guard: block ONLY when actually in projected(FILE) and Π invalid
if _file_mode_invalid_now():
    st.warning("Cert writing blocked: projected(FILE) is invalid. Fix or re-freeze the projector.")
    st.stop()

# --- Debounce (allow freezer / A/B to force a write this pass)
if st.session_state.pop("should_write_cert", False):
    st.session_state.pop("_last_cert_write_key", None)

# --- Guard: run context + inputs SSOT ---
if not (_rc and _out and _ib):
    st.info("Run Overlap first to enable cert writing.")
else:
    # --- Skip duplicate writes (but don't stop the rest of the UI)
    def _hz(s): return s if isinstance(s, str) else ""
    write_key = (
        _rc.get("policy_tag","strict"),
        _hz(_ib.get("boundaries_hash","")),
        _hz(_ib.get("C_hash","")),
        _hz(_ib.get("H_hash","")),
        _hz(_ib.get("U_hash","")),
        _hz(_ib.get("shapes_hash","")),
        (_hz(_rc.get("projector_hash","")) if str(_rc.get("mode","")).startswith("projected") else ""),
        bool((_out.get("2",{}) or {}).get("eq", False)),
        bool((_out.get("3",{}) or {}).get("eq", False)),
    )
    if st.session_state.get("_last_cert_write_key") == write_key:
        st.caption("Cert unchanged — skipping rewrite.")
    else:
        st.session_state["_last_cert_write_key"] = write_key

        # --- Diagnostics (single source) ---
        lane_mask = list(_rc.get("lane_mask_k3") or [])
        d3 = _rc.get("d3", [])
        H2 = (_H.blocks.__root__.get("2") or [])
        C3 = (cmap.blocks.__root__.get("3") or [])
        I3 = eye(len(C3)) if C3 else []

        def _bottom_row(M): return M[-1] if (M and len(M)) else []
        def _xor(A,B):
            if not A: return [r[:] for r in (B or [])]
            if not B: return [r[:] for r in (A or [])]
            return [[(A[i][j]^B[i][j])&1 for j in range(len(A[0]))] for i in range(len(A))]
        def _mask_row(row, lm):
            if not row: return []
            return [int(row[j]) if int(lm[j]) else 0 for j in range(len(row))]

        try:
            H2d3  = mul(H2, d3) if (H2 and d3) else []
            C3pI3 = _xor(C3, I3) if C3 else []
        except Exception:
            H2d3, C3pI3 = [], []

        lane_idx = [j for j,m in enumerate(lane_mask) if m]
        row_full_H2d3 = _bottom_row(H2d3)
        row_full_C3I  = _bottom_row(C3pI3)

        diagnostics_block = {
            "lane_mask_k3": lane_mask,
            "lane_vec_H2d3": {"row_full": row_full_H2d3, "row_lanes": _mask_row(row_full_H2d3, lane_mask)},
            "lane_vec_C3plusI3": {"row_full": row_full_C3I, "row_lanes": _mask_row(row_full_C3I, lane_mask)},
        }

        # --- Signatures ---
        def _gf2_rank(M):
            if not M or not M[0]: return 0
            A = [row[:] for row in M]; m, n = len(A), len(A[0]); r = c = 0
            while r<m and c<n:
                piv = next((i for i in range(r,m) if A[i][c]&1), None)
                if piv is None: c+=1; continue
                if piv!=r: A[r],A[piv]=A[piv],A[r]
                for i in range(m):
                    if i!=r and (A[i][c]&1):
                        A[i]=[(A[i][j]^A[r][j])&1 for j in range(n)]
                r+=1; c+=1
            return r
        rank_d3  = _gf2_rank(d3) if d3 else 0
        ncols_d3 = len(d3[0]) if (d3 and d3[0]) else 0
        ker_dim  = max(ncols_d3 - rank_d3, 0)
        lane_pattern = "".join("1" if int(x) else "0" for x in (lane_mask or []))
        def _col_support(M, cols):
            if not M: return ""
            use = cols if cols else list(range(len(M[0]) if (M and M[0]) else 0))
            return "".join("1" if any((row[j]&1) for row in M) else "0" for j in use)

        signatures_block = {
            "d_signature": {"rank": rank_d3, "ker_dim": ker_dim, "lane_pattern": lane_pattern},
            "fixture_signature": {"lane": _col_support(C3pI3, lane_idx)},
        }

        # --- Identity (one place) ---
        district_id = _di.get("district_id", st.session_state.get("district_id","UNKNOWN"))
        run_ts = getattr(hashes, "timestamp_iso_lisbon", lambda: datetime.now(timezone.utc).isoformat())()
        policy_now = _rc.get("policy_tag", policy_label_from_cfg(cfg_active))
        run_id = (st.session_state.get("run_ctx") or {}).get("run_id") or st.session_state.get("last_run_id")
        if not run_id:
            seed = "".join(str((_ib or {}).get(k,"")) for k in ("boundaries_hash","C_hash","H_hash","U_hash"))
            run_id = getattr(hashes,"run_id",lambda a,b: hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:12])(seed, run_ts)
            st.session_state["last_run_id"] = run_id

        identity_block = {
            "district_id": district_id, "run_id": run_id, "timestamp": run_ts,
            "app_version": getattr(hashes,"APP_VERSION","v0.1-core"),
            "python_version": _py_version_str(),
        }

        # --- Policy (mirror RC; strict clamps) ---
        policy_block = {
            "label": policy_now,
            "policy_tag": policy_now,
            "enabled_layers": cfg_active.get("enabled_layers", []),
            "modes": cfg_active.get("modes", {}),
            "source": (_rc.get("source") or {}),
        }
        if _rc.get("projector_hash") is not None:
            policy_block["projector_hash"] = _rc.get("projector_hash","")
        if _rc.get("projector_filename"):
            policy_block["projector_filename"] = _rc.get("projector_filename","")
        if _rc.get("projector_consistent_with_d") is not None:
            policy_block["projector_consistent_with_d"] = bool(_rc.get("projector_consistent_with_d"))
        if _rc.get("mode") == "strict":
            policy_block["enabled_layers"] = []
            for k in ("modes","source","projector_hash","projector_filename","projector_consistent_with_d"):
                policy_block.pop(k, None)

        # --- Checks / Inputs ---
        residual_tags = st.session_state.get("residual_tags", {}) or {}
        is_strict_mode = (_rc.get("mode") == "strict")
        checks_block = {
            **(_out or {}),
            "grid":  bool((_out or {}).get("grid", True)),
            "fence": bool((_out or {}).get("fence", True)),
            "ker_guard": ("enforced" if is_strict_mode else "off"),
        }

        inputs_block_payload = {
            "filenames": _ib.get("filenames", {
                "boundaries": st.session_state.get("fname_boundaries","boundaries.json"),
                "C":          st.session_state.get("fname_cmap","cmap.json"),
                "H":          st.session_state.get("fname_h","H.json"),
                "U":          st.session_state.get("fname_shapes","shapes.json"),
            }),
            "dims": _ib.get("dims", {}),
            "boundaries_hash": _ib.get("boundaries_hash",""),
            "C_hash": _ib.get("C_hash",""),
            "H_hash": _ib.get("H_hash",""),
            "U_hash": _ib.get("U_hash",""),
            "shapes_hash": _ib.get("shapes_hash", _ib.get("U_hash","")),
        }
        if _rc.get("mode") == "projected(file)":
            inputs_block_payload.setdefault("filenames", {})["projector"] = _rc.get("projector_filename","")

        dims_now = inputs_block_payload.get("dims") or {}
        for _k, _nk in (("2", dims_now.get("n2")), ("3", dims_now.get("n3"))):
            if _k in checks_block:
                checks_block[_k] = {**checks_block.get(_k, {}), "n_k": int(_nk) if _nk is not None else 0}

        # --- Promotion ---
        grid_ok  = bool(checks_block.get("grid", True))
        fence_ok = bool(checks_block.get("fence", True))
        k3_ok    = bool(checks_block.get("3", {}).get("eq", False))
        k2_ok    = bool(checks_block.get("2", {}).get("eq", False))
        mode_now = _rc.get("mode")
        eligible, target = False, None
        if mode_now == "strict" and all([grid_ok,fence_ok,k3_ok,k2_ok]) and residual_tags.get("strict","none")=="none":
            eligible, target = True, "strict_anchor"
        elif mode_now in ("projected(auto)","projected(file)") and all([grid_ok,fence_ok,k3_ok]) and residual_tags.get("projected","none")=="none":
            if mode_now == "projected(file)":
                if bool(_rc.get("projector_consistent_with_d")): eligible, target = True, "projected_exemplar"
            else:
                eligible, target = True, "projected_exemplar"
        promotion_block = {"eligible_for_promotion": eligible, "promotion_target": target, "notes": ""}

        # --- Artifacts mirror inputs (+ optional projector file sha) ---
        artifact_hashes = {
            "boundaries_hash": inputs_block_payload["boundaries_hash"],
            "C_hash":          inputs_block_payload["C_hash"],
            "H_hash":          inputs_block_payload["H_hash"],
            "U_hash":          inputs_block_payload["U_hash"],
        }
        if "projector_hash" in policy_block:
            artifact_hashes["projector_hash"] = policy_block.get("projector_hash","")
        if _rc.get("mode") == "projected(file)":
            pj_sha = _rc.get("projector_file_sha256")
            if not pj_sha:
                try:
                    pf = _rc.get("projector_filename","")
                    if pf and os.path.exists(pf):
                        with open(pf,"rb") as f: pj_sha = hashlib.sha256(f.read()).hexdigest()
                except Exception:
                    pj_sha = None
            if pj_sha:
                policy_block["projector_file_sha256"] = pj_sha
                artifact_hashes["projector_file_sha256"] = pj_sha

        # --- Assemble (pre-hash) ---
        cert_payload = {
            "schema_version": LAB_SCHEMA_VERSION,
            "identity": identity_block,
            "policy": policy_block,
            "inputs": inputs_block_payload,
            "diagnostics": diagnostics_block,
            "checks": checks_block,
            "signatures": signatures_block,
            "residual_tags": residual_tags,
            "promotion": promotion_block,
            "artifact_hashes": artifact_hashes,
            "app_version": getattr(hashes,"APP_VERSION","v0.1-core"),
            "python_version": _py_version_str(),
        }
        
                       # --- Optional A/B embed (fresh only; no re-derivation here) ---
        _ab = st.session_state.get("ab_compare") or {}
        if _ab_is_fresh(_ab, rc=_rc, ib=inputs_block_payload):
            strict_ctx = _ab.get("strict", {}) or {}
            proj_ctx   = _ab.get("projected", {}) or {}
        
            def _pv(out_block):
                return [
                    int((out_block or {}).get("2",{}).get("eq", False)),
                    int((out_block or {}).get("3",{}).get("eq", False)),
                ]
        
            # stage snapshots
            cert_payload["policy"]["strict_snapshot"] = {
                "policy_tag": "strict",
                "ker_guard": "enforced",
                "inputs": {"filenames": inputs_block_payload.get("filenames", {})},
                "lane_mask_k3": diagnostics_block["lane_mask_k3"],
                "lane_vec_H2d3": strict_ctx.get("lane_vec_H2d3"),
                "lane_vec_C3plusI3": strict_ctx.get("lane_vec_C3plusI3"),
                "pass_vec": _pv(strict_ctx.get("out", {})),
                "out": strict_ctx.get("out", {}),
            }
        
            proj_snap = {
                "policy_tag": proj_ctx.get("policy_tag") or (_rc.get("policy_tag") or ""),
                "ker_guard": "off",
                "inputs": {"filenames": inputs_block_payload.get("filenames", {})},
                "lane_mask_k3": diagnostics_block["lane_mask_k3"],
                "lane_vec_H2d3": proj_ctx.get("lane_vec_H2d3"),
                "lane_vec_C3plusI3": proj_ctx.get("lane_vec_C3plusI3"),
                "pass_vec": _pv(proj_ctx.get("out", {})),
                "out": proj_ctx.get("out", {}),
                "projector_hash": proj_ctx.get("projector_hash",""),
                "projector_consistent_with_d": proj_ctx.get("projector_consistent_with_d", None),
            }
            if _rc.get("mode") == "projected(file)":
                if _rc.get("projector_filename"):
                    proj_snap["projector_filename"] = _rc.get("projector_filename")
                if policy_block.get("projector_file_sha256"):
                    proj_snap["projector_file_sha256"] = policy_block["projector_file_sha256"]
        
            cert_payload["policy"]["projected_snapshot"] = proj_snap
            cert_payload["ab_pair_tag"] = _ab.get("pair_tag") or f"strict__VS__{proj_snap['policy_tag']}"
            cert_payload["ab_embedded"] = True
        
            # soft guard: if mismatch, drop the embed instead of failing the cert
            ab_proj_k3 = bool((_ab.get("projected") or {}).get("out", {}).get("3", {}).get("eq", False))
            cur_k3     = bool(checks_block.get("3", {}).get("eq", False))
            if ab_proj_k3 != cur_k3:
                # remove staged snapshots + flag as stale
                try:
                    cert_payload["policy"].pop("strict_snapshot", None)
                    cert_payload["policy"].pop("projected_snapshot", None)
                except Exception:
                    pass
                cert_payload.pop("ab_pair_tag", None)
                cert_payload["ab_embedded"] = False
                cert_payload["ab_stale_reason"] = "projected_k3_mismatch"  # visible hint
        else:
            cert_payload["ab_embedded"] = False



        # ---------- Standard meta before hashing ----------
        cert_payload.setdefault("schema_version", SCHEMA_VERSION)
        cert_payload.setdefault("app_version", APP_VERSION)
        cert_payload.setdefault("python_version", _py_version_str())
        rid = (st.session_state.get("run_ctx") or {}).get("run_id")
        if rid:
            cert_payload.setdefault("identity", {}).setdefault("run_id", rid)

        # --- invariants + hash (final payload) ---
        _assert_cert_invariants(cert_payload)
        cert_payload.setdefault("integrity", {})
        cert_payload["integrity"]["content_hash"] = hash_json(cert_payload)
        full_hash = cert_payload["integrity"]["content_hash"]

        # --- Write (prefer package) ---
        cert_path = None
        try:
            result = export_mod.write_cert_json(cert_payload)
            cert_path, full_hash = (result if isinstance(result,(list,tuple)) and len(result)>=2 else (result, full_hash))
        except Exception:
            outdir = Path(globals().get("CERTS_DIR","certs")); outdir.mkdir(parents=True, exist_ok=True)

            def _safe(s: str) -> str:
                return (s or "").replace("/", "_").replace(" ", "_")

            district_id_safe = _safe(district_id)
            safe_policy = _safe(policy_now)

            ab_suffix = ""
            if cert_payload.get("ab_embedded"):
                pair = cert_payload.get("ab_pair_tag") or ""
                ab_suffix = "__AB" + (f"__{_safe(pair)}" if pair else "")

            fname = f"overlap__{district_id_safe}__{safe_policy}{ab_suffix}__{full_hash[:12]}.json"
            p = outdir / fname
            tmp = p.with_suffix(".json.tmp")
            blob = _json.dumps(cert_payload, sort_keys=True, ensure_ascii=False, separators=(",",":")).encode("utf-8")
            with open(tmp,"wb") as f: f.write(blob); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, p); cert_path = str(p)

        # Cache + UI
        st.session_state["cert_payload"] = cert_payload
        st.session_state["last_cert_path"] = cert_path
        st.session_state["last_run_id"] = cert_payload["identity"]["run_id"]
        st.success(f"Cert written → `{cert_path}` · {full_hash[:12]}…")
        st.caption(f"Embedded A/B → {cert_payload.get('ab_pair_tag','A/B')}" if cert_payload.get("ab_embedded") else "Embedded A/B → —")

        # --- Bundle (cert + extras) ---
        with st.expander("Bundle (cert + extras)"):
            extras = ["policy.json",
                      "reports/residual.json","reports/parity_report.json","reports/coverage_sampling.csv",
                      "logs/gallery.jsonl","logs/witnesses.jsonl"]
            if _rc.get("mode")=="projected(file)" and _rc.get("projector_filename"):
                extras.append(_rc.get("projector_filename"))

            _disabled = _file_mode_invalid_now()
            _help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."
            if st.button("Build Cert Bundle",
                         key="build_cert_bundle_btn_final",
                         disabled=_disabled,
                         help=(_help_txt if _disabled else "Zip cert + selected artifacts")):
                try:
                    bp = build_cert_bundle(
                        district_id=district_id, policy_tag=policy_now,
                        cert_path=cert_path, content_hash=full_hash, extras=extras
                    )
                    st.success(f"Bundle ready → {bp}")
                    try:
                        with open(bp,"rb") as fz:
                            st.download_button("Download cert bundle", fz, file_name=os.path.basename(bp),
                                               key="dl_cert_bundle_zip_final")
                    except Exception: pass
                except Exception as e:
                    st.error(f"Bundle build failed: {e}")

# ───────────────────────── Certs on disk (tail) with A/B badge ─────────────────────────
def _fmt_ts(ts):
    try: return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%SZ")
    except: return ""

CERTS_DIR = Path(globals().get("CERTS_DIR","certs"))
CERTS_DIR.mkdir(parents=True, exist_ok=True)

with st.expander("Certs on disk (last 5)", expanded=False):
    all_certs = sorted(CERTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    st.caption(f"Found {len(all_certs)} certs in `{CERTS_DIR.as_posix()}`.")
    ab_only = st.checkbox("Show only certs with A/B embed", value=False, key="tail_ab_only_final")

    shown = 0
    for p in all_certs:
        if shown >= 5: break
        try:
            info = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ident  = info.get("identity") or {}
        policy = info.get("policy") or {}
        tag    = policy.get("policy_tag") or "strict"
        has_ab = bool(info.get("ab_embedded") or ("ab_pair_tag" in info) or ("ab_pair_tag" in policy))
        if ab_only and not has_ab: continue
        ab_label = f" · [A/B: {info.get('ab_pair_tag') or policy.get('ab_pair_tag') or 'A/B'}]" if has_ab else ""
        st.write(f"• {_fmt_ts(p.stat().st_mtime)} · {ident.get('district_id','UNKNOWN')} · {tag} · {p.name}{ab_label}")
        shown += 1




##-----------------------BUNDLE DONWLOAD SECTION------------------------------------------##
# Respect existing globals, with safe fallbacks
CERTS_DIR      = Path(globals().get("CERTS_DIR", "certs"))
LOGS_DIR       = Path(globals().get("LOGS_DIR", "logs"))
REPORTS_DIR    = Path(globals().get("REPORTS_DIR", "reports"))
BUNDLES_DIR    = Path(globals().get("BUNDLES_DIR", "bundles"))
PROJECTORS_DIR = Path(globals().get("PROJECTORS_DIR", "projectors"))

SCHEMA_VERSION = globals().get("SCHEMA_VERSION", "1.0.0")
APP_VERSION    = globals().get("APP_VERSION", getattr(globals().get("hashes", object), "APP_VERSION", "v0.1-core"))

# Ensure dirs we write to exist
for _d in (BUNDLES_DIR,):
    _d.mkdir(parents=True, exist_ok=True)

# ───────────────────────── Utils (dedup) ─────────────────────────
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

def _count_files(root: Path) -> int:
    if not root.exists(): return 0
    n = 0
    for _, _, files in os.walk(root): n += len(files)
    return n

def generate_run_id_from_seed(seed: str, ts: str) -> str:
    return hashlib.sha256(f"{seed}|{ts}".encode("utf-8")).hexdigest()[:12]

def _mkkey(ns: str, name: str) -> str:
    return f"{ns}__{name}"

def _fmt_ts(ts_float: float) -> str:
    try:
        return datetime.fromtimestamp(ts_float, timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return ""

def _py_version_str() -> str:
    return f"python-{platform.python_version()}"

# ───────────────────────── Builders (inputs bundle & snapshot) ─────────────────────────
def build_inputs_bundle(*, inputs_block: dict, run_ctx: dict, district_id: str, run_id: str, policy_tag: str) -> str:
    """
    Creates a ZIP with manifest.json and input files.
    """
    app_ver = globals().get("APP_VERSION_STR", APP_VERSION)
    py_ver  = globals().get("PY_VERSION_STR", _py_version_str())

    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    fns = (inputs_block.get("filenames") or {})
    fnames = {
        "boundaries": fns.get("boundaries", st.session_state.get("fname_boundaries", "boundaries.json")),
        "C":          fns.get("C",          st.session_state.get("fname_cmap",       "cmap.json")),
        "H":          fns.get("H",          st.session_state.get("fname_h",          "H.json")),
        "U":          fns.get("U",          st.session_state.get("fname_shapes",     "shapes.json")),
        "projector":  fns.get("projector",  run_ctx.get("projector_filename", "") or ""),
    }

    hashes_block = {
        "boundaries_hash": inputs_block.get("boundaries_hash", ""),
        "C_hash":          inputs_block.get("C_hash", ""),
        "H_hash":          inputs_block.get("H_hash", ""),
        "U_hash":          inputs_block.get("U_hash", ""),
        "shapes_hash":     inputs_block.get("shapes_hash", ""),
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": _utc_iso_z(),
        "app_version": app_ver,
        "python_version": py_ver,
        "policy_tag": policy_tag,
        "hashes": hashes_block,
        "filenames": fnames,
        "projector": {
            "mode": run_ctx.get("mode", "strict"),
            "filename": run_ctx.get("projector_filename", ""),
            "projector_hash": run_ctx.get("projector_hash", ""),
        },
    }

    zname = f"inputs__{district_id or 'UNKNOWN'}__{run_id}.zip"
    zpath = BUNDLES_DIR / zname

    fd, tmp_name = tempfile.mkstemp(dir=BUNDLES_DIR, prefix=".tmp_inputs_", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            for _, fp in fnames.items():
                if not fp: continue
                p = Path(fp)
                if p.exists():
                    try:
                        arcname = p.resolve().relative_to(Path.cwd().resolve()).as_posix()
                    except Exception:
                        arcname = p.name
                    zf.write(str(p), arcname=arcname)
        try:
            os.replace(tmp_path, zpath)
        except OSError:
            shutil.move(str(tmp_path), str(zpath))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return str(zpath)

def build_everything_snapshot() -> str:
    """
    Builds a ZIP with certs, referenced projectors, logs, reports, metadata, and an index.
    """
    # Collect certs
    cert_files = sorted(CERTS_DIR.glob("*.json"))
    parsed, skipped = [], []
    for p in cert_files:
        data, err = _read_json_safely(p)
        if err or not isinstance(data, dict):
            skipped.append({"path": _rel(p), "reason": "JSON_PARSE_ERROR"})
            continue
        parsed.append((p, data))
    if not parsed:
        st.info("Nothing to snapshot yet (no parsed certs).")
        return ""

    proj_refs, districts, index_rows, manifest_files = set(), set(), [], []

    for p, cert in parsed:
        ident  = cert.get("identity") or {}
        pol    = cert.get("policy") or {}
        inputs = cert.get("inputs") or {}

        did = ident.get("district_id") or "UNKNOWN"
        districts.add(str(did))
        manifest_files.append({
            "path": _rel(p),
            "sha256": _sha256_file(p),
            "size": p.stat().st_size,
        })

        pj_fname = pol.get("projector_filename", "") or ""
        if isinstance(pj_fname, str) and pj_fname.strip():
            proj_refs.add(pj_fname.strip())

        # prefer flat hashes, fall back to nested
        hashes_flat   = {k: inputs.get(k) for k in ("boundaries_hash","C_hash","H_hash","U_hash")}
        hashes_nested = inputs.get("hashes") or {}
        def _hx(k): return hashes_flat.get(k) or hashes_nested.get(k) or ""

        index_rows.append([
            _rel(p),
            (cert.get("integrity") or {}).get("content_hash", ""),
            pol.get("policy_tag", ""),
            did,
            ident.get("run_id", ""),
            ident.get("timestamp", ""),
            _hx("boundaries_hash"),
            _hx("C_hash"),
            _hx("H_hash"),
            _hx("U_hash"),
            str(pol.get("projector_hash", "") or ""),
            str(pol.get("projector_filename", "") or ""),
        ])

    # Resolve projectors
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

    # Logs & reports that exist
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

    # Build cert_index.csv text
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
        try: os.remove(idx_tmp)
        except Exception: pass

    # Create ZIP (atomic)
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
            for f in manifest_files + projectors + logs_list + reports_list:
                p = Path(f["path"])
                if p.exists():
                    zf.write(p.as_posix(), arcname=f["path"])
        os.replace(tmpzip, zpath)
    finally:
        if tmpzip.exists():
            try: tmpzip.unlink()
            except Exception: pass

    if len(index_rows) != manifest["counts"]["certs"]:
        st.warning("Index count does not match manifest cert count (investigate).")
    return str(zpath)

# ───────────────────────── Flush / Resets (dedup) ─────────────────────────
def flush_workspace(*, delete_projectors: bool=False) -> dict:
    """
    Remove artifacts, reset session state, recreate empty dirs. Keeps inputs intact.
    """
    summary = {
        "when": _utc_iso_z(),
        "deleted_dirs": [],
        "recreated_dirs": [],
        "files_removed": 0,
        "token": "",
        "composite_cache_key_short": "",
    }

    # Clear session state (idempotent)
    for k in (
        "_inputs_block", "_district_info", "run_ctx", "overlap_out", "overlap_H",
        "residual_tags", "ab_compare", "last_cert_path", "cert_payload",
        "last_run_id", "_gallery_keys", "_last_boundaries_hash",
        "_projector_cache", "_projector_cache_ab", "parity_pairs",
        "parity_last_report_pairs", "selftests_snapshot"
    ):
        st.session_state.pop(k, None)

    # Remove dirs and recreate
    dirs = [CERTS_DIR, LOGS_DIR, REPORTS_DIR, BUNDLES_DIR]
    if delete_projectors:
        dirs.append(PROJECTORS_DIR)

    removed_files_count = 0
    for d in dirs:
        if d.exists():
            removed_files_count += _count_files(d)
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        summary["deleted_dirs"].append(str(d))
        summary["recreated_dirs"].append(str(d))
    summary["files_removed"] = removed_files_count

    # Generate new cache key and token
    ts = _utc_iso_z()
    salt = secrets.token_hex(2).upper()
    token = f"FLUSH-{ts}-{salt}"
    ckey = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()

    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"] = token
    summary["token"] = token
    summary["composite_cache_key_short"] = ckey[:12]
    return summary

def _soft_reset_before_overlap_local():
    ss = st.session_state
    for k in (
        "run_ctx","overlap_out","overlap_cfg","overlap_policy_label",
        "overlap_H","residual_tags","ab_compare",
        "cert_payload","last_cert_path","_last_cert_write_key",
        "_projector_cache","_projector_cache_ab"
    ):
        ss.pop(k, None)

def _session_flush_run_cache():
    # Clear computed session keys only; do not touch disk
    if "_soft_reset_before_overlap" in globals():
        try:
            _soft_reset_before_overlap()  # type: ignore[name-defined]
        except Exception:
            _soft_reset_before_overlap_local()
    else:
        _soft_reset_before_overlap_local()
    # Bump nonce
    st.session_state["_fixture_nonce"] = int(st.session_state.get("_fixture_nonce", 0)) + 1
    # New cache key + token
    ts = _utc_iso_z()
    salt = secrets.token_hex(2).upper()
    token = f"RUN-FLUSH-{ts}-{salt}"
    ckey  = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()
    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"] = token
    return {"token": token, "ckey_short": ckey[:12]}

def _full_flush_workspace(delete_projectors: bool = False):
    # Prefer app’s flush if available
    if "flush_workspace" in globals():
        try:
            return flush_workspace(delete_projectors=delete_projectors)  # type: ignore[name-defined]
        except Exception:
            pass
    return flush_workspace(delete_projectors=delete_projectors)

# ───────────────────────── UI: Exports / Snapshot / Flush (dedup, namespaced) ─────────────────────────
EXPORTS_NS = "exports_v2"

import os  

with safe_expander("Exports", expanded=False):
    c1, c2, c3 = st.columns(3)

    # ---- Snapshot ZIP ----
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
            except Exception as e:
                st.error(f"Snapshot failed: {e}")

    # ---- Inputs Bundle ----
    with c2:
        if st.button("Export Inputs Bundle", key=_mkkey(EXPORTS_NS, "btn_export_inputs")):
            try:
                ib = st.session_state.get("_inputs_block") or {}
                di = st.session_state.get("_district_info") or {}
                rc = st.session_state.get("run_ctx") or {}
                cert_cached = st.session_state.get("cert_payload")

                district_id = di.get("district_id", "UNKNOWN")
                run_id = (cert_cached or {}).get("identity", {}).get("run_id") or st.session_state.get("last_run_id")
                if not run_id:
                    seed_str = "".join(ib.get(k, "") for k in ("boundaries_hash", "C_hash", "H_hash", "U_hash"))
                    ts = _utc_iso_z()
                    run_id = generate_run_id_from_seed(seed_str, ts)
                    st.session_state["last_run_id"] = run_id

                policy_tag = st.session_state.get("overlap_policy_label") or rc.get("policy_tag") or "strict"

                bp = build_inputs_bundle(
                    inputs_block=ib,
                    run_ctx=rc,
                    district_id=district_id,
                    run_id=run_id,
                    policy_tag=policy_tag,
                )
                st.session_state["last_inputs_bundle_path"] = bp
                st.success(f"Inputs bundle ready → {bp}")
                with open(bp, "rb") as fz:
                    st.download_button(
                        "Download inputs bundle",
                        fz,
                        file_name=os.path.basename(bp),
                        key=_mkkey(EXPORTS_NS, "dl_inputs_bundle"),
                    )
            except Exception as e:
                st.error(f"Export Inputs Bundle failed: {e}")

    # ---- Flushes ----
    with c3:
        st.caption("Flush / Reset")
        if st.button(
            "Quick Reset (session only)",
            key=_mkkey(EXPORTS_NS, "btn_quick_reset_session"),
            help="Clears computed session data, bumps nonce; does not touch files.",
        ):
            out = _session_flush_run_cache()
            st.success(f"Run cache flushed · token={out['token']} · key={out['ckey_short']}")

        inc_pj = st.checkbox(
            "Also remove projectors (full flush)",
            value=False,
            key=_mkkey(EXPORTS_NS, "flush_inc_pj"),
        )
        confirm = st.checkbox(
            "I understand this deletes files on disk",
            value=False,
            key=_mkkey(EXPORTS_NS, "ff_confirm"),
        )
        if st.button(
            "Full Flush (certs/logs/reports/bundles)",
            key=_mkkey(EXPORTS_NS, "btn_full_flush"),
            disabled=not confirm,
            help="Deletes persisted outputs; keeps inputs. Bumps nonce & resets session.",
        ):
            try:
                info = _full_flush_workspace(delete_projectors=inc_pj)
                st.success(f"Workspace flushed · {info['token']}")
                st.caption(f"New cache key: `{info['composite_cache_key_short']}`")
                with st.expander("Flush details"):
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
            
            with tab5:
                st.subheader("Export")
                st.caption("Bundle all artifacts in ./reports into a single ZIP for sharing/archival.")
                if st.button("Export ./reports → report.zip"):
                    reports_dir = pathlib.Path("reports")
                    if not reports_dir.exists():
                        st.warning("No ./reports directory yet. Run a Tower or Manifest first.")
                    else:
                        zpath = reports_dir / "report.zip"
                        export_mod.zip_report(str(reports_dir), str(zpath))
                        st.success(f"Exported: {zpath}")
                        with open(zpath, "rb") as fz:
                            st.download_button("Download report.zip", fz, file_name="report.zip")
                           
