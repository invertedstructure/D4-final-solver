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

# ====================== A/B Compare (strict vs active projected) ======================

def perform_overlap_check(boundaries_obj, cmap_obj, H_used, cfg=None):
    return overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_used, projection_config=cfg)

def get_policy_label(cfg):
    return policy_label_from_cfg(cfg)

# Run button for A/B compare
if st.button("Run A/B compare", key="ab_run_btn_final"):
    try:
        ss = st.session_state
        rc = ss.get("run_ctx") or {}
        H_used = ss.get("overlap_H") or _load_h_local()
        cfg_for_ab = ss.get("overlap_cfg") or cfg_active

        # Get boundaries & cmap from your app’s globals (fallbacks optional)
        bnd = boundaries
        cmap_obj = cmap

        # --- strict leg
        out_strict = perform_overlap_check(bnd, cmap_obj, H_used)
        label_strict = get_policy_label(cfg_strict())

        # --- projected leg (mirrors ACTIVE; validates FILE inside projector_choose_active)
        src3 = (cfg_for_ab.get("source") or {}).get("3")
        if src3 == "file":
            try:
                P_ab, meta_ab = projector_choose_active(cfg_for_ab, bnd)
                validate_projector_file_strict(
                    P_ab,
                    n3=int(rc.get("n3") or 0),
                    lane_mask=list(rc.get("lane_mask_k3") or [])
                )
            except ValueError as ve:
                st.error(f"A/B FILE Π invalid: {ve}")
                st.stop()
        else:
            P_ab, meta_ab = projector_choose_active(cfg_for_ab, bnd)

        out_proj = perform_overlap_check(bnd, cmap_obj, H_used, cfg_for_ab)
        label_proj = get_policy_label(cfg_for_ab)

        # --- lane vectors (use run_ctx lane mask; do not recompute)
        d3 = (bnd.blocks.__root__.get("3") or [])
        H2 = (H_used.blocks.__root__.get("2") or [])
        C3 = (cmap_obj.blocks.__root__.get("3") or [])
        I3 = eye(len(C3)) if C3 else []

        def _xor(A, B):
            if not A: return [r[:] for r in (B or [])]
            if not B: return [r[:] for r in (A or [])]
            r, c = len(A), len(A[0])
            return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]

        def _bottom_row(M): return M[-1] if (M and len(M)) else []
        def _mask(vec, idx): return [vec[j] for j in idx] if (vec and idx) else []

        lane_mask = list(rc.get("lane_mask_k3") or [])
        lane_idx = [j for j, m in enumerate(lane_mask) if m]

        H2d3  = mul(H2, d3) if (H2 and d3) else []
        C3pI3 = _xor(C3, I3) if C3 else []

        lane_vec_H2d3 = _mask(_bottom_row(H2d3), lane_idx)
        lane_vec_C3I  = _mask(_bottom_row(C3pI3), lane_idx)

        # --- snapshot payload (freshness keyed to *current* inputs hashes)
        inputs_sig = _current_inputs_sig()
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
                "cfg":   cfg_for_ab,
                "out":   out_proj,
                "ker_guard": "off",
                "lane_vec_H2d3": lane_vec_H2d3[:],
                "lane_vec_C3plusI3": lane_vec_C3I[:],
                "pass_vec": [
                    int(out_proj.get("2",{}).get("eq", False)),
                    int(out_proj.get("3",{}).get("eq", False)),
                ],
                "projector_filename": meta_ab.get("projector_filename",""),
                "projector_hash": meta_ab.get("projector_hash",""),
                "projector_consistent_with_d": meta_ab.get("projector_consistent_with_d", None),
            },
        }
        st.session_state["ab_compare"] = ab_payload

        # Trigger cert embedding now
        st.session_state["should_write_cert"] = True
        st.session_state.pop("_last_cert_write_key", None)

        # Status update
        s_ok = bool(out_strict.get("3",{}).get("eq", False))
        p_ok = bool(out_proj.get("3",{}).get("eq", False))
        st.success(f"A/B updated → strict={'✅' if s_ok else '❌'} · projected={'✅' if p_ok else '❌'} · {pair_tag}")
        st.caption("A/B will embed into the cert when inputs hashes are unchanged (fresh).")

        with st.expander("A/B snapshot (details)"):
            st.json(ab_payload)

    except ValueError as e:
        st.error(f"A/B projected(file) invalid: {e}")
    except Exception as e:
        st.error(f"A/B compare failed: {e}")

# Single stale handler / clearer (no duplicate badges)
_ab = st.session_state.get("ab_compare") or {}
if _ab and (_ab.get("inputs_sig") != _current_inputs_sig()):
    if st.button("Clear stale A/B", key="btn_ab_clear_final"):
        st.session_state.pop("ab_compare", None)
        st.success("Cleared A/B snapshot. Re-run A/B to refresh.")





# ────────────────────── Reports: Perturbation Sanity & Fence Stress ──────────────────────

# Version/constants (respect existing globals if present)
SCHEMA_VERSION = globals().get("SCHEMA_VERSION", "1.0.0")
APP_VERSION    = globals().get("APP_VERSION", "v0.1-core")

# Paths (safe init)
REPORTS_DIR = Path(globals().get("REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Schema tags for these reports
PERTURB_SCHEMA_VERSION = SCHEMA_VERSION
FENCE_SCHEMA_VERSION   = "1.0.1"  # fence targets U (carrier) when hooks present
APP_VER                = APP_VERSION

# Outputs
PERTURB_OUT_PATH = REPORTS_DIR / "perturbation_sanity.csv"
FENCE_OUT_PATH   = REPORTS_DIR / "fence_stress.csv"

# Local helpers (tool-scoped)
def _copy_mat(M): return [row[:] for row in (M or [])]

def _is_zero(M):
    return (not M) or all(all((x & 1) == 0 for x in row) for row in M)

def _strict_R3(H2, d3, C3):
    I3 = eye(len(C3)) if C3 else []
    return _xor_mat(mul(H2, d3), _xor_mat(C3, I3)) if (H2 and d3 and C3) else []

def _projected_R3(R3_strict, P_active):
    return mul(R3_strict, P_active) if (R3_strict and P_active) else []

def _lane_mask_from_d3_matrix(d3: list[list[int]]) -> list[int]:
    if not d3 or not d3[0]:
        return []
    rows, n3 = len(d3), len(d3[0])
    return [1 if any(d3[i][j] & 1 for i in range(rows)) else 0 for j in range(n3)]

def _sig_tag_eq(boundaries_obj, cmap_obj, H_used_obj, P_active=None):
    """Return (lane_mask, tag_strict, eq3_strict, tag_proj, eq3_proj)."""
    d3 = (boundaries_obj.blocks.__root__.get("3") or [])
    H2 = (H_used_obj.blocks.__root__.get("2") or [])
    C3 = (cmap_obj.blocks.__root__.get("3") or [])
    lm = _lane_mask_from_d3_matrix(d3)
    R3s = _strict_R3(H2, d3, C3)

    if "residual_tag" in globals() and callable(globals()["residual_tag"]):
        tag_s = residual_tag(R3s, lm)
    else:
        def _residual_tag_local(R, mask):
            if not R or not mask: return "none"
            rows = len(R)
            nz   = lambda j: any(R[i][j] & 1 for i in range(rows))
            lanes = any(nz(j) for j, m in enumerate(mask) if m)
            ker   = any(nz(j) for j, m in enumerate(mask) if not m)
            if not lanes and not ker: return "none"
            if lanes and not ker:     return "lanes"
            if ker and not lanes:     return "ker"
            return "mixed"
        tag_s = _residual_tag_local(R3s, lm)

    eq_s = _is_zero(R3s)
    if P_active:
        R3p   = _projected_R3(R3s, P_active)
        tag_p = residual_tag(R3p, lm) if "residual_tag" in globals() and callable(globals()["residual_tag"]) else tag_s
        eq_p  = _is_zero(R3p)
    else:
        tag_p, eq_p = None, None
    return lm, tag_s, bool(eq_s), tag_p, (None if eq_p is None else bool(eq_p))


# -------- optional carrier (U) mutation hooks ----------
HAS_U_HOOKS = (
    "get_carrier_mask" in globals() and "set_carrier_mask" in globals()
    and callable(globals()["get_carrier_mask"]) and callable(globals()["set_carrier_mask"])
)

with st.expander("Reports: Perturbation Sanity & Fence Stress"):
    # Freshness + SSOT guardrails
    try:
        rc = require_fresh_run_ctx()
        rc = rectify_run_ctx_mask_from_d3()
    except Exception as e:
        st.warning(str(e))
        st.stop()

    # Inputs / policy
    H_used = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})
    P_active = rc.get("P_active") if str(rc.get("mode","")).startswith("projected") else None
    B0, C0, H0 = boundaries, cmap, H_used
    U0 = shapes  # carrier object (needed for Fence stress)

    d3_base = (B0.blocks.__root__.get("3") or [])
    n2 = len(d3_base)
    n3 = len(d3_base[0]) if (d3_base and d3_base[0]) else 0

    # UI
    colA, colB = st.columns([2,2])
    with colA:
        max_flips = st.number_input("Perturbation: max flips", min_value=1, max_value=500, value=24, step=1, key="ps_max")
        seed_txt  = st.text_input("Seed (determines flip order)", value="ps-seed-1", key="ps_seed")
    with colB:
        run_fence  = st.checkbox("Include Fence stress run (perturb U)", value=True, key="fence_on")

    # Disable downstream actions if FILE Π invalid (global predicate)
    disabled = file_validation_failed()
    help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    if st.button("Run Perturbation Sanity (and Fence if checked)", key="ps_run",
                 disabled=disabled, help=(help_txt if disabled else "Run perturbation sanity; optionally include fence")):
        try:
            # Baseline (no mutation)
            lm0, tag_s0, eq_s0, tag_p0, eq_p0 = _sig_tag_eq(B0, C0, H0, P_active)

            # Deterministic flips (on d3 for PS)
            def _flip_targets(n2_, n3_, budget, seed_str):
                h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16)
                i = (h % (max(1, n2_))) if n2_ else 0
                j = ((h >> 8) % (max(1, n3_))) if n3_ else 0
                for k in range(budget):
                    yield (i, j, k)
                    i = (i + 1 + (h % 3)) % (n2_ or 1)
                    j = (j + 2 + ((h >> 5) % 5)) % (n3_ or 1)

            rows = []
            drift_witnessed = False

            # --- Perturbation sanity: flip d3 bits, detect grammar drift
            for (r, c, k) in _flip_targets(n2, n3, int(max_flips), seed_txt):
                if not (n2 and n3):
                    rows.append([k, 0, "grammar", "empty fixture"])
                    continue

                d3_mut = _copy_mat(d3_base)
                d3_mut[r][c] ^= 1  # GF(2) flip

                dB = B0.dict() if hasattr(B0, "dict") else {"blocks": {}}
                dB = json.loads(json.dumps(dB))
                dB.setdefault("blocks", {})["3"] = d3_mut
                Bk = io.parse_boundaries(dB)

                lmK, tag_sK, eq_sK, tag_pK, eq_pK = _sig_tag_eq(Bk, C0, H0, P_active)

                guard_tripped = int(lmK != lm0)   # grammar drift = lane mask change
                expected_guard = "grammar"        # consistent guard name
                note = ""

                if guard_tripped and not drift_witnessed:
                    drift_witnessed = True
                    cert_like = st.session_state.get("cert_payload")
                    if cert_like and "append_witness_row" in globals():
                        try:
                            append_witness_row(
                                cert_like,
                                reason="grammar-drift",
                                residual_tag_val=(tag_sK or "none"),
                                note=f"flip#{k} at (r={r}, c={c})"
                            )
                            note = "lane_mask_changed → auto-witness logged"
                        except Exception:
                            note = "lane_mask_changed (witness append failed)"
                    else:
                        note = "lane_mask_changed"

                rows.append([k, guard_tripped, expected_guard, note])

            # Emit CSV for Perturbation Sanity (3c header)
            header = ["flip_id", "guard_tripped", "expected_guard", "note"]
            meta = [
                f"schema_version={SCHEMA_VERSION}",
                f"saved_at={_utc_iso_z()}",
                f"run_id={(st.session_state.get('run_ctx') or {}).get('run_id','')}",
                f"app_version={APP_VERSION}",
                f"seed={seed_txt}",
                f"n2={n2}",
                f"n3={n3}",
                f"baseline_tag_strict={tag_s0}",
                f"baseline_tag_projected={'' if tag_p0 is None else tag_p0}",
            ]
            _atomic_write_csv(PERTURB_OUT_PATH, header, rows, meta)
            st.success(f"Perturbation sanity saved → {PERTURB_OUT_PATH}")

            # --- Fence stress: perturb U (carrier) if hooks available; otherwise fall back to H2 tweak
            if run_fence:
                # Prepare small helpers
                def _strict_eq_for(boundaries_obj, cmap_obj, H_obj):
                    d3 = (boundaries_obj.blocks.__root__.get("3") or [])
                    H2 = (H_obj.blocks.__root__.get("2") or [])
                    C3 = (cmap_obj.blocks.__root__.get("3") or [])
                    return int(_is_zero(_strict_R3(H2, d3, C3)))

                notes = []
                rows_fs = []

                if HAS_U_HOOKS:
                    # Read base carrier mask
                    U_mask = get_carrier_mask(U0)  # type: ignore[name-defined]
                    rU = len(U_mask); cU = len(U_mask[0]) if (U_mask and U_mask[0]) else 0

                    def _count1(M): return sum(int(x & 1) for row in (M or []) for x in row)

                    # U_shrink: simple 1-cell erosion on the border (best-effort)
                    U_shrink = _copy_mat(U_mask)
                    if rU and cU:
                        for j in range(cU): U_shrink[0][j] = 0; U_shrink[-1][j] = 0
                        for i in range(rU): U_shrink[i][0] = 0; U_shrink[i][-1] = 0

                    # U_plus: simple 1-cell dilation on the border (best-effort)
                    U_plus = _copy_mat(U_mask)
                    if rU and cU:
                        for j in range(cU): U_plus[0][j]  = 1; U_plus[-1][j] = 1
                        for i in range(rU): U_plus[i][0]  = 1; U_plus[i][-1] = 1

                    # Build shapes variants (without touching global state)
                    U_shrink_obj = set_carrier_mask(json.loads(json.dumps(U0.dict() if hasattr(U0,"dict") else {"blocks": {}})), U_shrink)  # type: ignore[name-defined]
                    U_plus_obj   = set_carrier_mask(json.loads(json.dumps(U0.dict() if hasattr(U0,"dict") else {"blocks": {}})), U_plus)    # type: ignore[name-defined]
                    # Re-parse to your shapes type if set_carrier_mask returned dicts
                    if not hasattr(U_shrink_obj, "blocks"):
                        U_shrink_obj = io.parse_shapes(U_shrink_obj)
                    if not hasattr(U_plus_obj, "blocks"):
                        U_plus_obj = io.parse_shapes(U_plus_obj)

                    # Run strict eq (policy constant)
                    eq_shrink = _strict_eq_for(B0, C0, H0)
                    eq_plus   = _strict_eq_for(B0, C0, H0)

                    rows_fs.append(["U_shrink", f"[1,{eq_shrink}]", f"|U|:{_count1(U_mask)}→{_count1(U_shrink)}"])
                    rows_fs.append(["U_plus",   f"[1,{eq_plus}]",   f"|U|:{_count1(U_mask)}→{_count1(U_plus)}"])
                    notes.append("fence target = U (carrier); H fixed")
                else:
                    # Fallback to previous H2 perturbation (keeps behavior if U hooks are absent)
                    H2 = (H0.blocks.__root__.get("2") or [])
                    H2_shrink = _copy_mat(H2[:-1]) if len(H2) >= 1 else _copy_mat(H2)
                    if H2 and H2[0]:
                        zero_row = [0]*len(H2[0]); H2_plus = _copy_mat(H2) + [zero_row]
                    else:
                        H2_plus = _copy_mat(H2)

                    H_shrink = json.loads(json.dumps(H0.dict() if hasattr(H0,"dict") else {"blocks": {}}))
                    H_plus   = json.loads(json.dumps(H0.dict() if hasattr(H0,"dict") else {"blocks": {}}))
                    H_shrink.setdefault("blocks", {})["2"] = H2_shrink
                    H_plus.setdefault("blocks", {})["2"]   = H2_plus
                    H_shrink = io.parse_cmap(H_shrink)
                    H_plus   = io.parse_cmap(H_plus)

                    C3 = (C0.blocks.__root__.get("3") or [])
                    d3 = (B0.blocks.__root__.get("3") or [])
                    R3_shrink = _strict_R3(H2_shrink, d3, C3)
                    R3_plus   = _strict_R3(H2_plus,   d3, C3)
                    eq_shrink = int(_is_zero(R3_shrink))
                    eq_plus   = int(_is_zero(R3_plus))

                    rows_fs = [
                        ["H2_shrink (fallback)", f"[1,{eq_shrink}]", "drop last H2 row"],
                        ["H2_plus (fallback)",   f"[1,{eq_plus}]",   "append zero row to H2"],
                    ]
                    notes.append("fallback: fence target = H2 (no U hooks found)")

                # Write CSV (3d header)
                fence_header = ["U_class", "pass_vec", "note"]
                fence_meta = [
                    f"schema_version={SCHEMA_VERSION}",
                    f"saved_at={_utc_iso_z()}",
                    f"run_id={(st.session_state.get('run_ctx') or {}).get('run_id','')}",
                    f"app_version={APP_VERSION}",
                ] + notes  # optional notes appended

                _atomic_write_csv(FENCE_OUT_PATH, fence_header, rows_fs, fence_meta)
                st.success(f"Fence stress saved → {FENCE_OUT_PATH}")

            # Quick downloads
            try:
                with open(PERTURB_OUT_PATH, "rb") as f:
                    st.download_button("Download perturbation_sanity.csv", f, file_name="perturbation_sanity.csv", key="dl_ps_csv")
            except Exception:
                pass
            if run_fence and FENCE_OUT_PATH.exists():
                try:
                    with open(FENCE_OUT_PATH, "rb") as f2:
                        st.download_button("Download fence_stress.csv", f2, file_name="fence_stress.csv", key="dl_fence_csv")
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Perturbation/Fence run failed: {e}")



# =========================[ · Gallery Append & Dedupe ]=========================

GALLERY_PATH = (LOGS_DIR / "gallery.jsonl")
GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)

ss = st.session_state
# session dedupe cache
ss.setdefault("_gallery_keys", set())
ss.setdefault("_gallery_bootstrapped", False)

with st.expander("Gallery"):
    # 1) Freshness + SSOT guard
    try:
        rc  = require_fresh_run_ctx()
        rc  = rectify_run_ctx_mask_from_d3()
    except Exception:
        st.stop()

    out = ss.get("overlap_out") or {}
    eligible_green = is_projected_green(rc, out)

    # Global disable if projected(FILE) invalid
    fm_bad  = file_validation_failed()
    help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    cert = ss.get("cert_payload")
    if not cert:
        st.info("No cert in memory yet. Run Overlap (let cert writer emit) before adding to gallery.")
        st.stop()

    # 2) Extract canonical fields from the *cert* (read-only)
    identity = cert.get("identity", {}) or {}
    policy   = cert.get("policy",   {}) or {}
    inputs   = cert.get("inputs",   {}) or {}
    hashes   = inputs.get("hashes", {}) or {
        "boundaries_hash": inputs.get("boundaries_hash", ""),
        "C_hash":          inputs.get("C_hash", ""),
        "H_hash":          inputs.get("H_hash", ""),
        "U_hash":          inputs.get("U_hash", ""),
        "shapes_hash":     inputs.get("shapes_hash", ""),
    }

    district_id    = identity.get("district_id", "UNKNOWN")
    policy_tag     = policy.get("policy_tag", "")
    projector_hash = policy.get("projector_hash", "") or ""
    cert_hash      = (cert.get("integrity") or {}).get("content_hash", "") or ""

    # 3) Optional UI fields
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        growth_bumps = st.number_input("growth_bumps", min_value=0, value=0, step=1, key="gal_growth_bumps")
    with c2:
        strictify = st.selectbox("strictify", options=["tbd","no","yes"], index=0, key="gal_strictify")
    with c3:
        tag = st.text_input("tag (optional)", value="", key="gal_tag")

    # 4) Build row per spec (meta first)
    row = {
        **_std_meta(run_id=(st.session_state.get("run_ctx") or {}).get("run_id")),
        "district":       district_id,
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
        "growth_bumps":      int(growth_bumps),
        "strictify":         str(strictify),
        "tag":               tag or "",
        "cert_content_hash": cert_hash,
    }

    # 5) One dedupe key (6-tuple); use the same function for BOTH new row & tail rows
    key = gallery_key(row)

    # Bootstrap session cache from tail once
    if not ss["_gallery_bootstrapped"]:
        for tail_row in _read_jsonl_tail(GALLERY_PATH, N=200):
            try:
                ss["_gallery_keys"].add(gallery_key(tail_row))
            except Exception:
                continue
        ss["_gallery_bootstrapped"] = True

    # 6) Append button (gated by green & FILE state)
    # final disabled state and tooltip
    disabled = fm_bad or (not eligible_green)
    tip = help_txt if fm_bad else (None if eligible_green else "Enabled only when projected is green (k=3 eq=True).")
    if st.button(
        "Add to Gallery",
        key="btn_gallery_append",
        disabled=disabled,
        help=(tip or "Append current cert to gallery.jsonl"),
    ):
        try:
            if key in ss["_gallery_keys"]:
                st.info("Duplicate skipped (same district/policy/hashes).")
            else:
                _atomic_append_jsonl(GALLERY_PATH, row)
                ss["_gallery_keys"].add(key)
                st.success("Gallery row appended.")
        except Exception as e:
            st.error(f"Gallery append failed: {e}")

    # 7) Tail view (last 8)
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

# ======================================================================================
# =========================[ STEP 3 · Witness on Stubborn RED ]=========================

WITNESS_PATH = (LOGS_DIR / "witnesses.jsonl")
WITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)

ss = st.session_state
ss.setdefault("_witness_keys", set())
ss.setdefault("_witness_bootstrapped", False)

def witness_key(row: dict):
    """8-tuple dedupe: (district, reason, residual_tag, policy_tag, B, C, H, U)."""
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

with st.expander("Witness logger"):
    # --- Freshness + SSOT guards
    try:
        rc  = require_fresh_run_ctx()
        rc  = rectify_run_ctx_mask_from_d3()
    except Exception:
        st.stop()

    out = ss.get("overlap_out") or {}
    eq3 = bool(((out.get("3") or {}).get("eq", False)))
    eligible_red = (eq3 is False)

    # FILE Π invalid gate
    fm_bad  = file_validation_failed()
    help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    # Pick the correct residual tag (strict vs projected)
    tags = ss.get("residual_tags") or {}
    mode = str(rc.get("mode","strict"))
    residual_tag_val = tags.get("projected" if mode.startswith("projected") else "strict", "none")

    # Pull cert payload if present (for content hash); optional
    cert = ss.get("cert_payload") or {}
    cert_hash = (cert.get("integrity") or {}).get("content_hash","") or ""

    # Canonical hashes (prefer inputs.hashes inside cert; fall back to inputs flat)
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
    policy_tag = policy.get("policy_tag", rc.get("policy_tag",""))
    projector_hash = policy.get("projector_hash","") or (rc.get("projector_hash","") or "")

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
        "residual_tag":   residual_tag_val,   # "lanes"/"ker"/"mixed"/"none"
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
    disabled = fm_bad or (not eligible_red)
    tip = help_txt if fm_bad else (None if eligible_red else "Enabled only when k=3 is RED (eq=False).")
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

    # Tail view
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

# ======================================================================================
# ========================= JSONL → CSV Exports (Gallery & Witness) =========================

# (reuses globals SCHEMA_VERSION / APP_VERSION / LOGS_DIR / REPORTS_DIR)
GALLERY_JSONL   = LOGS_DIR / "gallery.jsonl"
WITNESSES_JSONL = LOGS_DIR / "witnesses.jsonl"
GALLERY_CSV_OUT = REPORTS_DIR / "gallery_export.csv"
WITNESS_CSV_OUT = REPORTS_DIR / "witnesses_export.csv"

def _atomic_write_csv(path: Path, header: list[str], rows: list[list], comments: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", newline="") as tmp:
        if comments:
            for line in comments:
                tmp.write(f"# {line}\n")
        w = csv.writer(tmp)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
        tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    os.replace(tmp_name, path)

def _jsonl_read_all(path: Path) -> list[dict]:
    if not path.exists(): return []
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        pass
    return out

# ---------- flatteners ----------
def _flatten_gallery_row(r: dict) -> list:
    h = (r.get("hashes") or {})
    pol = (r.get("policy") or {})
    return [
        r.get("written_at_utc", ""),
        r.get("district", "UNKNOWN"),
        pol.get("policy_tag", ""),
        pol.get("projector_hash", ""),
        h.get("boundaries_hash", ""),
        h.get("C_hash", ""),
        h.get("H_hash", ""),
        h.get("U_hash", ""),
        h.get("shapes_hash", ""),
        int(r.get("growth_bumps", 0) or 0),
        str(r.get("strictify", "")),
        r.get("tag", ""),
        r.get("cert_content_hash", ""),
    ]

def _flatten_witness_row(r: dict) -> list:
    h = (r.get("hashes") or {})
    pol = (r.get("policy") or {})
    return [
        r.get("written_at_utc", ""),
        r.get("district", "UNKNOWN"),
        r.get("reason", ""),
        r.get("residual_tag", ""),
        pol.get("policy_tag", ""),
        pol.get("projector_hash", ""),
        h.get("boundaries_hash", ""),
        h.get("C_hash", ""),
        h.get("H_hash", ""),
        h.get("U_hash", ""),
        h.get("shapes_hash", ""),
        r.get("cert_content_hash", ""),
        r.get("run_id", "") or (r.get("run_id") if "run_id" in r else ""),  # tolerate legacy
        r.get("note", ""),
    ]

# ---------- UI ----------
with st.expander("Exports · Gallery & Witness (CSV)"):
    ss = st.session_state
    ss["_gal_csv_nonce"] = int(ss.get("_gal_csv_nonce", 0))
    ss["_wit_csv_nonce"] = int(ss.get("_wit_csv_nonce", 0))

    c1, c2 = st.columns(2)

    # --- Gallery → CSV ---
    with c1:
        st.caption("Gallery → CSV")
        if not GALLERY_JSONL.exists():
            st.info("No gallery.jsonl yet.")
        else:
            if st.button("Export Gallery CSV", key="btn_export_gallery_csv",
                         disabled=file_validation_failed(),
                         help=("Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."
                               if file_validation_failed() else "Export to CSV")):
                rows_json = _jsonl_read_all(GALLERY_JSONL)
                header = [
                    "written_at_utc","district","policy_tag","projector_hash",
                    "boundaries_hash","C_hash","H_hash","U_hash","shapes_hash",
                    "growth_bumps","strictify","tag","cert_content_hash"
                ]
                rows_csv = [_flatten_gallery_row(r) for r in rows_json]
                meta = [
                    f"schema_version={SCHEMA_VERSION}",
                    f"written_at_utc={_utc_iso_z()}",
                    f"app_version={APP_VERSION}",
                    f"source={GALLERY_JSONL.as_posix()}",
                    f"rows={len(rows_csv)}",
                ]
                try:
                    _atomic_write_csv(GALLERY_CSV_OUT, header, rows_csv, meta)
                    st.success(f"Gallery CSV saved → {GALLERY_CSV_OUT}")
                    ss["_gal_csv_nonce"] += 1
                except Exception as e:
                    st.error(f"Failed to write Gallery CSV: {e}")

            if GALLERY_CSV_OUT.exists():
                try:
                    with open(GALLERY_CSV_OUT, "rb") as f:
                        st.download_button(
                            "Download gallery_export.csv",
                            f,
                            file_name="gallery_export.csv",
                            key=f"dl_gallery_csv_{ss['_gal_csv_nonce']}"
                        )
                except Exception:
                    pass

    # --- Witness → CSV ---
    with c2:
        st.caption("Witness → CSV")
        if not WITNESSES_JSONL.exists():
            st.info("No witnesses.jsonl yet.")
        else:
            if st.button("Export Witness CSV", key="btn_export_witness_csv",
                         disabled=file_validation_failed(),
                         help=("Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."
                               if file_validation_failed() else "Export to CSV")):
                rows_json = _jsonl_read_all(WITNESSES_JSONL)
                header = [
                    "written_at_utc","district","reason","residual_tag",
                    "policy_tag","projector_hash",
                    "boundaries_hash","C_hash","H_hash","U_hash","shapes_hash",
                    "cert_content_hash","run_id","note"
                ]
                rows_csv = [_flatten_witness_row(r) for r in rows_json]
                meta = [
                    f"schema_version={SCHEMA_VERSION}",
                    f"written_at_utc={_utc_iso_z()}",
                    f"app_version={APP_VERSION}",
                    f"source={WITNESSES_JSONL.as_posix()}",
                    f"rows={len(rows_csv)}",
                ]
                try:
                    _atomic_write_csv(WITNESS_CSV_OUT, header, rows_csv, meta)
                    st.success(f"Witness CSV saved → {WITNESS_CSV_OUT}")
                    ss["_wit_csv_nonce"] += 1
                except Exception as e:
                    st.error(f"Failed to write Witness CSV: {e}")

            if WITNESS_CSV_OUT.exists():
                try:
                    with open(WITNESS_CSV_OUT, "rb") as f:
                        st.download_button(
                            "Download witnesses_export.csv",
                            f,
                            file_name="witnesses_export.csv",
                            key=f"dl_witness_csv_{ss['_wit_csv_nonce']}"
                        )
                except Exception:
                    pass
# ======================= /end JSONL → CSV Exports (Gallery & Witness) ======================



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










  
    # ======================== Parity: import/export & queue ========================

PARITY_SCHEMA_VERSION = "1.0.0"
DEFAULT_PARITY_PATH = Path("logs") / "parity_pairs.json"
PARITY_REPORT_PATH  = Path("reports") / "parity_report.json"
PARITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _atomic_write_json(path: Path, payload: dict) -> None:
    _ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        _json.dump(payload, tmp, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    os.replace(tmp_name, path)

def _safe_parse_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return _json.load(f)

def load_fixture_from_paths(*, boundaries_path: str, cmap_path: str, H_path: str, shapes_path: str):
    dB = _safe_parse_json(boundaries_path)
    dC = _safe_parse_json(cmap_path)
    dH = _safe_parse_json(H_path)
    dU = _safe_parse_json(shapes_path)
    return {
        "boundaries": io.parse_boundaries(dB),
        "cmap":       io.parse_cmap(dC),
        "H":          io.parse_cmap(dH),
        "shapes":     io.parse_shapes(dU),
    }

def add_parity_pair(*, label: str, left_fixture: dict, right_fixture: dict) -> int:
    req = ("boundaries","cmap","H","shapes")
    for side_name, fx in (("left", left_fixture), ("right", right_fixture)):
        if not isinstance(fx, dict) or any(k not in fx for k in req):
            raise ValueError(f"{side_name} fixture malformed; expected keys {req}")
    st.session_state.setdefault("parity_pairs", [])
    st.session_state["parity_pairs"].append({"label": label, "left": left_fixture, "right": right_fixture})
    return len(st.session_state["parity_pairs"])

def clear_parity_pairs() -> None:
    st.session_state["parity_pairs"] = []

def set_parity_pairs_from_fixtures(pairs_spec: list[dict]) -> int:
    clear_parity_pairs()
    for row in pairs_spec:
        label = row.get("label", "PAIR")
        Lp, Rp = row.get("left", {}), row.get("right", {})
        L = load_fixture_from_paths(boundaries_path=Lp["boundaries"], cmap_path=Lp["cmap"], H_path=Lp["H"], shapes_path=Lp["shapes"])
        R = load_fixture_from_paths(boundaries_path=Rp["boundaries"], cmap_path=Rp["cmap"], H_path=Rp["H"], shapes_path=Rp["shapes"])
        add_parity_pair(label=label, left_fixture=L, right_fixture=R)
    return len(st.session_state.get("parity_pairs", []))

def _parity_pairs_payload(pairs: list[dict]) -> dict:
    return {
        "schema_version": PARITY_SCHEMA_VERSION,
        "saved_at": _iso_utc_now(),
        "count": len(pairs),
        "pairs": [
            {
                "label": row.get("label", "PAIR"),
                "left":  {k: row.get("left_path_"+k,  row.get("left", {}).get(k+"_path", ""))  for k in ("boundaries","cmap","H","shapes")},
                "right": {k: row.get("right_path_"+k, row.get("right", {}).get(k+"_path", "")) for k in ("boundaries","cmap","H","shapes")},
            } for row in pairs
        ],
    }

def _pairs_from_payload(payload: dict) -> list[dict]:
    if not isinstance(payload, dict): return []
    return [
        {
            "label": r.get("label", "PAIR"),
            "left":  {k: r.get("left", {}).get(k, "")  for k in ("boundaries","cmap","H","shapes")},
            "right": {k: r.get("right", {}).get(k, "") for k in ("boundaries","cmap","H","shapes")},
        } for r in payload.get("pairs", [])
    ]

def export_parity_pairs(path: str | Path = DEFAULT_PARITY_PATH) -> str:
    path = Path(path); _ensure_parent_dir(path)
    pairs = st.session_state.get("parity_pairs", []) or []
    payload = _parity_pairs_payload(pairs)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        _json.dump(payload, tmp, indent=2)
        tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    os.replace(tmp_name, path)
    return str(path)

def import_parity_pairs(path: str | Path = DEFAULT_PARITY_PATH, *, merge: bool = False) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No parity pairs file at {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)
    ver = payload.get("schema_version", "0.0.0")
    if ver.split(".")[0] != PARITY_SCHEMA_VERSION.split(".")[0]:
        st.warning(f"parity_pairs schema version differs (file={ver}, app={PARITY_SCHEMA_VERSION}); best-effort load.")
    pairs_spec = _pairs_from_payload(payload)
    if not merge:
        clear_parity_pairs()
    set_parity_pairs_from_fixtures(pairs_spec)
    return len(st.session_state.get("parity_pairs", []))

with safe_expander("Parity: queue sample D2/D3/D4 pairs (optional)"):
    st.caption("Only queues pairs if files exist under ./inputs/.")
    c1, c2 = st.columns(2)
    with c1:
        do_self = st.button("Queue SELF (current fixture vs itself)", key="pp_self_btn")
    with c2:
        do_examples = st.button("Queue D2↔D3, D3↔D4 examples", key="pp_examples_btn")

    if do_self:
        try:
            fixture = {
                "boundaries": boundaries,
                "cmap": cmap,
                "H": st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}}),
                "shapes": shapes,
            }
            add_parity_pair(label="SELF", left_fixture=fixture, right_fixture=fixture)
            st.success("Queued SELF parity pair.")
        except Exception as e:
            st.error(f"Could not queue SELF: {e}")

    if do_examples:
        spec = [
            {
                "label": "D2(101)↔D3(110)",
                "left":  {"boundaries":"inputs/D2/boundaries.json","cmap":"inputs/D2/cmap.json","H":"inputs/D2/H.json","shapes":"inputs/D2/shapes.json"},
                "right": {"boundaries":"inputs/D3/boundaries.json","cmap":"inputs/D3/cmap.json","H":"inputs/D3/H.json","shapes":"inputs/D3/shapes.json"},
            },
            {
                "label": "D3(110)↔D4(101)",
                "left":  {"boundaries":"inputs/D3/boundaries.json","cmap":"inputs/D3/cmap.json","H":"inputs/D3/H.json","shapes":"inputs/D3/shapes.json"},
                "right": {"boundaries":"inputs/D4/boundaries.json","cmap":"inputs/D4/cmap.json","H":"inputs/D4/H.json","shapes":"inputs/D4/shapes.json"},
            },
        ]
        flat = []
        for r in spec:
            L, R = r["left"], r["right"]
            flat += [L["boundaries"], L["cmap"], L["H"], L["shapes"], R["boundaries"], R["cmap"], R["H"], R["shapes"]]
        if not all(Path(p).exists() for p in flat):
            st.info("Example files not found under ./inputs — skipping queuing.")
        else:
            try:
                set_parity_pairs_from_fixtures(spec)
                st.success("Queued D2↔D3 and D3↔D4 example pairs.")
            except Exception as e:
                st.error(f"Could not queue examples: {e}")

with safe_expander("Parity pairs: import/export"):
    colA, colB, colC = st.columns([3,3,2])
    with colA:
        export_path = st.text_input("Export path", value=str(DEFAULT_PARITY_PATH), key="pp_export_path")
    with colB:
        import_path = st.text_input("Import path", value=str(DEFAULT_PARITY_PATH), key="pp_import_path")
    with colC:
        merge_load = st.checkbox("Merge on import", value=False, key="pp_merge")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Export parity_pairs.json", key="pp_do_export"):
            try:
                p = export_parity_pairs(export_path)
                st.success(f"Saved parity pairs → {p}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    with c2:
        if st.button("Import parity_pairs.json", key="pp_do_import"):
            try:
                n = import_parity_pairs(import_path, merge=merge_load)
                st.success(f"Loaded {n} pairs from {import_path}")
            except Exception as e:
                st.error(f"Import failed: {e}")

# ============================== Parity Runner ===============================

def _cfg_from_run_ctx(rc: dict) -> dict | None:
    mode = (rc or {}).get("mode", "strict")
    if mode == "strict": return None
    cfg = cfg_projected_base()
    if mode == "projected(auto)":
        cfg["source"]["3"] = "auto"
        return cfg
    if mode == "projected(file)":
        cfg["source"]["3"] = "file"
        pj = (rc or {}).get("projector_filename", "")
        if pj: cfg.setdefault("projector_files", {})["3"] = pj
        return cfg
    return None

def _and_pair(a: bool | None, b: bool | None) -> bool | None:
    if a is None or b is None: return None
    return bool(a) and bool(b)

def _one_leg(boundaries_obj, cmap_obj, H_obj, projection_cfg: dict | None):
    if projection_cfg is None:
        return overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_obj)
    _P, _meta = projector_choose_active(projection_cfg, boundaries_obj)  # validates FILE Π
    return overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_obj, projection_config=projection_cfg)

def _emoji(v):
    if v is None: return "—"
    return "✅" if bool(v) else "❌"

with st.expander("Parity: run suite (mirrors active policy)"):
    try:
        rc = require_fresh_run_ctx()
        rc = rectify_run_ctx_mask_from_d3()
    except Exception as e:
        st.warning(str(e)); st.stop()

    pairs = st.session_state.get("parity_pairs", []) or []
    if not pairs:
        st.info("No parity pairs queued. Use the import/queue controls above.")
        st.stop()

    rc_   = st.session_state.get("run_ctx", {}) or {}
    ib    = st.session_state.get("_inputs_block", {}) or {}
    policy_tag = rc_.get("policy_tag", policy_label_from_cfg(cfg_strict()))
    pj_hash    = rc_.get("projector_hash","") if rc_.get("mode","").startswith("projected") else ""
    cfg_proj   = _cfg_from_run_ctx(rc_)  # None if strict

    c1, c2, c3 = st.columns([1,1,2])
    with c1: st.caption("Active policy:");    st.code(policy_tag, language="text")
    with c2: st.caption("Projector hash:");   st.code(pj_hash[:12] + ("…" if pj_hash else ""), language="text")
    with c3: st.caption("One projector decision is reused across the whole batch.")

    # FILE Π invalid → disable downstream run button (as requested)
    _disabled = file_validation_failed()
    _help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    if st.button("Run Parity Suite",
                 key="btn_run_parity_final",
                 disabled=_disabled,
                 help=(_help_txt if _disabled else "Run parity across queued pairs")):
        report_pairs: list[dict] = []
        rows_preview: list[list[str]] = []
        errors: list[str] = []

        for row in pairs:
            label = row.get("label","PAIR")
            L, R = row.get("left", {}), row.get("right", {})

            try:
                bL, cL, hL = L["boundaries"], L["cmap"], L["H"]
                bR, cR, hR = R["boundaries"], R["cmap"], R["H"]

                out_L_strict = _one_leg(bL, cL, hL, None)
                out_R_strict = _one_leg(bR, cR, hR, None)

                s_k2 = _and_pair(out_L_strict.get("2",{}).get("eq"), out_R_strict.get("2",{}).get("eq"))
                s_k3 = _and_pair(out_L_strict.get("3",{}).get("eq"), out_R_strict.get("3",{}).get("eq"))

                if cfg_proj is not None:
                    try:
                        out_L_proj = _one_leg(bL, cL, hL, cfg_proj)
                        out_R_proj = _one_leg(bR, cR, hR, cfg_proj)
                        p_k2 = _and_pair(out_L_proj.get("2",{}).get("eq"), out_R_proj.get("2",{}).get("eq"))
                        p_k3 = _and_pair(out_L_proj.get("3",{}).get("eq"), out_R_proj.get("3",{}).get("eq"))
                    except ValueError as e:
                        p_k2, p_k3 = False, False
                        errors.append(f"{label}: {e}")
                else:
                    p_k2, p_k3 = None, None

                report_pairs.append({
                    "label": label,
                    "strict":    {"k2": (None if s_k2 is None else bool(s_k2)),
                                  "k3": (None if s_k3 is None else bool(s_k3))},
                    "projected": {"k2": (None if p_k2 is None else bool(p_k2)),
                                  "k3": (None if p_k3 is None else bool(p_k3))},
                })
                rows_preview.append([label, _emoji(s_k3), _emoji(p_k3)])

            except Exception as e:
                errors.append(f"{label}: {e}")

        # ---------- Parity report payload (standard meta helper) ----------
        payload = {
            **_std_meta(run_id=(st.session_state.get("run_ctx") or {}).get("run_id")),  # <- NEW standard meta
            "policy_tag": policy_tag,
            **({"projector_hash": pj_hash} if pj_hash else {}),
            "pairs": report_pairs,
            "hashes": {
                "boundaries_hash": ib.get("boundaries_hash",""),
                "C_hash":          ib.get("C_hash",""),
                "H_hash":          ib.get("H_hash",""),
                "U_hash":          ib.get("U_hash",""),
            },
            **({"errors": errors} if errors else {}),
        }

        try:
            _atomic_write_json(PARITY_REPORT_PATH, payload)
            st.success(f"Parity report saved → {PARITY_REPORT_PATH}")
            st.caption("Summary (per pair): strict_k3 / projected_k3")
            for r in rows_preview:
                st.write(f"• {r[0]} → strict={r[1]} · projected={r[2]}")
            with open(PARITY_REPORT_PATH, "rb") as f:
                st.download_button("Download parity_report.json", f, file_name="parity_report.json", key="dl_parity_report_final")
            if errors:
                st.warning("Some pairs had issues; details recorded in the report’s `errors` field.")
            st.session_state["parity_last_report_pairs"] = report_pairs
        except Exception as e:
            st.error(f"Could not write parity_report.json: {e}")

    last_pairs = st.session_state.get("parity_last_report_pairs")
    if last_pairs:
        import pandas as pd
        df = pd.DataFrame([
            {"Pair": p["label"],
             "Strict k3": _emoji(p["strict"]["k3"]),
             "Proj k3":   _emoji(p["projected"]["k3"]),
             "Strict k2": _emoji(p["strict"]["k2"]),
             "Proj k2":   _emoji(p["projected"]["k2"])}
            for p in last_pairs
        ], columns=["Pair", "Strict k3", "Proj k3", "Strict k2", "Proj k2"])
        st.caption("Parity summary")
        st.dataframe(df, use_container_width=True)
        try:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download parity_summary.csv", csv_bytes, file_name="parity_summary.csv", key="dl_parity_summary_final")
        except Exception:
            pass

# =============================== Coverage Sampling ==============================

COVERAGE_CSV_PATH = REPORTS_DIR / "coverage_sampling.csv"
COVERAGE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

def _rand_gf2_matrix(rows: int, cols: int, density: float, rng: random.Random) -> list[list[int]]:
    density = max(0.0, min(1.0, float(density)))
    return [[1 if rng.random() < density else 0 for _ in range(cols)] for _ in range(rows)]

def _gf2_rank(M: list[list[int]]) -> int:
    if not M: return 0
    A = [row[:] for row in M]
    m, n = len(A), len(A[0])
    r, c = 0, 0
    while r < m and c < n:
        pivot = None
        for i in range(r, m):
            if A[i][c] & 1: pivot = i; break
        if pivot is None: c += 1; continue
        if pivot != r: A[r], A[pivot] = A[pivot], A[r]
        for i in range(r+1, m):
            if A[i][c] & 1:
                A[i] = [(A[i][j] ^ A[r][j]) for j in range(n)]
        r += 1; c += 1
    return r

def _col_support_pattern(M: list[list[int]]) -> list[str]:
    if not M: return []
    rows, cols = len(M), len(M[0])
    cols_bits = []
    for j in range(cols):
        bits = ''.join('1' if (M[i][j] & 1) else '0' for i in range(rows))
        cols_bits.append(bits)
    cols_bits.sort()
    return cols_bits

def _lane_pattern_from_mask(mask: list[int]) -> str:
    return ''.join(str(int(x) & 1) for x in (mask or []))

def _coverage_signature(d_k1: list[list[int]], n_k: int) -> str:
    rk = _gf2_rank(d_k1)
    ker = max(0, int(n_k) - rk)
    patt = _col_support_pattern(d_k1)
    return f"rk={rk};ker={ker};pattern=[{','.join(patt)}]"

def _in_district_guess(signature: str, *, current_lane_pattern: str) -> int:
    try:
        bracket = signature.split("pattern=[", 1)[1].split("]", 1)[0]
        col_bitstrings = [s.strip() for s in bracket.split(",") if s.strip()]
        return int(any(bs == current_lane_pattern for bs in col_bitstrings))
    except Exception:
        return 0

with st.expander("Coverage Sampling"):
    try:
        rc = require_fresh_run_ctx()
        rc = rectify_run_ctx_mask_from_d3()
    except Exception as e:
        st.warning(str(e)); st.stop()

    n3_default = int((rc or {}).get("n3") or 0)
    try:
        H_local = st.session_state.get("overlap_H") or _load_h_local()
        H2_rows = len((H_local.blocks.__root__.get("2") or []))
    except Exception:
        H2_rows = 0
    n2_default = H2_rows

    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        num_samples = st.number_input("Samples", min_value=1, max_value=10000, value=250, step=50, key="cov_nsamples")
    with c2:
        bit_density = st.slider("Bit density", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key="cov_density")
    with c3:
        n2 = st.number_input("Rows (n₂)", min_value=0, max_value=2048, value=n2_default, step=1, key="cov_n2")
    with c4:
        n3 = st.number_input("Cols (n₃)", min_value=0, max_value=2048, value=n3_default, step=1, key="cov_n3")

    seed_txt = st.text_input("Seed (any string/hex)", value="cov-seed-0001", key="cov_seed")

    # Disable downstream action if FILE Π invalid (consistent gating)
    _disabled = file_validation_failed()
    _help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    if st.button("Coverage Sample",
                 key="btn_coverage_sample",
                 disabled=_disabled,
                 help=(_help_txt if _disabled else "Generate coverage_sampling.csv with meta header")):

        if n3 <= 0 or n2 <= 0:
            st.warning("Please ensure n₂ and n₃ are both > 0.")
            st.stop()

        rng = random.Random(); rng.seed(seed_txt)

        # Ensure a run_id is stamped for this sampling
        run_id = (st.session_state.get("run_ctx") or {}).get("run_id") or str(uuid.uuid4())
        st.session_state.setdefault("run_ctx", {})["run_id"] = run_id

        counts: dict[str, int] = {}
        lane_pattern = _lane_pattern_from_mask(rc.get("lane_mask_k3") or [])
        for _ in range(int(num_samples)):
            d_k1 = _rand_gf2_matrix(n2, n3, float(bit_density), rng)
            sig = _coverage_signature(d_k1, n_k=n3)
            counts[sig] = counts.get(sig, 0) + 1

        total = float(num_samples)
        rows = []
        for sig, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            in_d = _in_district_guess(sig, current_lane_pattern=lane_pattern)
            pct = 0.0 if total <= 0 else round(100.0 * (cnt / total), 2)
            rows.append([sig, cnt, in_d, pct])

        # ---------- Coverage CSV meta header (standardized) ----------
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

        # Use shared CSV writer if available
        try:
            _atomic_write_csv(COVERAGE_CSV_PATH, header, rows, meta_lines)  # type: ignore[arg-type]
        except Exception:
            # minimal local writer fallback
            with tempfile.NamedTemporaryFile("w", delete=False, dir=COVERAGE_CSV_PATH.parent, encoding="utf-8", newline="") as tmp:
                for line in meta_lines:
                    tmp.write(f"# {line}\n")
                w = csv.writer(tmp); w.writerow(header); w.writerows(rows)
                tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
            os.replace(tmp_name, COVERAGE_CSV_PATH)

        st.success(f"Coverage CSV saved → {COVERAGE_CSV_PATH}")

        try:
            import pandas as pd
            preview = pd.DataFrame(rows[:30], columns=header)
            st.dataframe(preview, use_container_width=True, hide_index=True)
        except Exception:
            pass

        try:
            with open(COVERAGE_CSV_PATH, "rb") as f:
                st.download_button("Download coverage_sampling.csv", f, file_name="coverage_sampling.csv", key="dl_coverage_csv")
        except Exception:
            pass




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

# --- Hard-guard: block when FILE Π invalid (requested policy) ---
if file_validation_failed():
    st.warning("Cert writing blocked: projected(FILE) is invalid. Fix or re-freeze the projector.")
    st.stop()

# --- Debounce (allow freezer to force) ---
if st.session_state.pop("should_write_cert", False):
    st.session_state.pop("_last_cert_write_key", None)

# --- Guard: run context + inputs SSOT ---
if not (_rc and _out and _ib):
    st.info("Run Overlap first to enable cert writing.")
else:
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
        st.stop()
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

    # --- Optional A/B embed (fresh only) ---
    _ab = st.session_state.get("ab_compare") or {}
    def _sig_now(_ibp):
        return [str(_ibp.get("boundaries_hash","")), str(_ibp.get("C_hash","")), str(_ibp.get("H_hash","")),
                str(_ibp.get("U_hash","")), str(_ibp.get("shapes_hash",""))]
    if _ab and (_ab.get("inputs_sig") == _sig_now(inputs_block_payload)):
        strict_ctx = _ab.get("strict", {}) or {}
        proj_ctx   = _ab.get("projected", {}) or {}
        proj_mode  = _rc.get("mode","projected(auto)")
        proj_tag   = "projected(file)" if proj_mode=="projected(file)" else "projected(auto)"
        def _pv(out_block):
            return [int((out_block or {}).get("2",{}).get("eq",False)),
                    int((out_block or {}).get("3",{}).get("eq",False))]
        cert_payload["policy"]["strict_snapshot"] = {
            "policy_tag":"strict","ker_guard":"enforced",
            "inputs":{"filenames": inputs_block_payload.get("filenames", {})},
            "lane_mask_k3": lane_mask,
            "lane_vec_H2d3": strict_ctx.get("lane_vec_H2d3"),
            "lane_vec_C3plusI3": strict_ctx.get("lane_vec_C3plusI3"),
            "pass_vec": _pv(strict_ctx.get("out", {})),
            "out": strict_ctx.get("out", {}),
        }
        proj_snap = {
            "policy_tag": proj_tag,"ker_guard":"off",
            "inputs":{"filenames": inputs_block_payload.get("filenames", {})},
            "lane_mask_k3": lane_mask,
            "lane_vec_H2d3": proj_ctx.get("lane_vec_H2d3"),
            "lane_vec_C3plusI3": proj_ctx.get("lane_vec_C3plusI3"),
            "pass_vec": _pv(proj_ctx.get("out", {})),
            "out": proj_ctx.get("out", {}),
            "projector_hash": _rc.get("projector_hash", proj_ctx.get("projector_hash","")),
            "projector_consistent_with_d": _rc.get("projector_consistent_with_d", proj_ctx.get("projector_consistent_with_d")),
        }
        if proj_mode == "projected(file)" and _rc.get("projector_filename"):
            proj_snap["projector_filename"] = _rc.get("projector_filename")
            pf_sha = cert_payload["policy"].get("projector_file_sha256")
            if pf_sha: proj_snap["projector_file_sha256"] = pf_sha
        cert_payload["policy"]["projected_snapshot"] = proj_snap
        cert_payload["ab_pair_tag"] = f"strict__VS__{proj_tag}"
        cert_payload["ab_embedded"] = True
    else:
        cert_payload["ab_embedded"] = False

    # ---------- Standard meta before hashing (requested) ----------
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
        safe_policy = str(policy_now).replace("/","_").replace(" ","_")
        suffix = "__ab" if cert_payload.get("ab_embedded") else ""
        fname = f"overlap__{district_id}__{safe_policy}{suffix}__{full_hash[:12]}.json"
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

        _disabled = file_validation_failed()
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

    for p in all_certs[:5]:
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

# ───────────────────────── UI: Exports / Snapshot / Flush (dedup) ─────────────────────────
EXPORTS_NS = "exports_v2"

with st.expander("Exports", expanded=False):
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
        if st.button("Quick Reset (session only)", key="btn_quick_reset_session",
                     help="Clears computed session data, bumps nonce; does not touch files."):
            out = _session_flush_run_cache()
            st.success(f"Run cache flushed · token={out['token']} · key={out['ckey_short']}")

        inc_pj = st.checkbox("Also remove projectors (full flush)", value=False, key="flush_inc_pj_final")
        confirm  = st.checkbox("I understand this deletes files on disk", value=False, key="ff_confirm_v2")
        if st.button("Full Flush (certs/logs/reports/bundles)", key="btn_full_flush_everything",
                     disabled=not confirm,
                     help="Deletes persisted outputs; keeps inputs. Bumps nonce & resets session."):
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
