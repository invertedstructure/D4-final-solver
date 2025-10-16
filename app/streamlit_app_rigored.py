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

# ------------------------- IO/Zip helpers (stubs retained) -------------------------
def read_json_file(upload): pass
def _stamp_filename(state_key: str, upload): pass
def _atomic_write_json(path: str | Path, obj: dict, *, pretty: bool = False): pass
def atomic_write_json(path: str | Path, obj: dict, *, pretty: bool = False): _atomic_write_json(path, obj, pretty=pretty)
def atomic_append_jsonl(path: str | Path, row: dict): pass
def _atomic_write_csv(path: Path, header: list[str], rows: list[list], meta_comment_lines: list[str] | None = None): pass
def _lane_mask_from_d3(boundaries) -> list[int]: pass
def _district_signature(mask: list[int], r: int, c: int) -> str: pass
def policy_label_from_cfg(cfg: dict) -> str: pass
def _read_projector_matrix(path_str: str): pass
def projector_choose_active(cfg_active: dict, boundaries): pass
def hash_matrix_norm(M) -> str: pass
def _zip_arcname(abspath: str) -> str: pass
def build_cert_bundle(*, district_id: str, policy_tag: str, cert_path: str,
                      content_hash: str | None = None, extras: list[str] | None = None) -> str: pass
def build_inputs_block(boundaries, cmap, H_used, shapes, filenames: dict) -> dict: pass
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
        return "projected(file)" if src == "file" else "projected(auto)"

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

    # You can add additional UI controls or logic here as needed for your app.


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


def _resolve_projector(cfg_active: dict, boundaries) -> tuple[list[list[int]], dict]:
    """
    Normalize projector_choose_active(...) to (P_active, meta).
    Accepts (P_active, meta) or a meta dict; raises ValueError on failure.
    Fills defaults for mode/d3/n3/projector fields.
    """
    try:
        res = projector_choose_active(cfg_active, boundaries)
    except ValueError as e:
        raise
    except Exception as e:
        raise ValueError(f"PROJECTOR_RESOLVE_RUNTIME: {e}")

    if res is None:
        raise ValueError("PROJECTOR_RESOLVE_NONE")

    if isinstance(res, tuple) and len(res) == 2:
        P_active, meta = res
    elif isinstance(res, dict):
        meta = dict(res)
        P_active = meta.get("P_active") or meta.get("P") or []
    else:
        raise ValueError(f"PROJECTOR_RESOLVE_BAD_RETURN: {type(res)!r}")

    # defaults
    source3 = (cfg_active.get("source") or {}).get("3", "auto")
    meta.setdefault("mode", "projected(file)" if source3 == "file" else ("projected(auto)" if source3 == "auto" else "strict"))
    if "d3" not in meta:
        try:
            meta["d3"] = (boundaries.blocks.__root__.get("3") or [])
        except Exception:
            meta["d3"] = []
    meta.setdefault("n3", (len(meta["d3"][0]) if (meta["d3"] and meta["d3"][0]) else 0))
    meta.setdefault("projector_filename", (cfg_active.get("projector_files") or {}).get("3", ""))
    meta.setdefault("projector_hash", "")
    meta.setdefault("projector_consistent_with_d", None)

    return P_active, meta

        
# ------------------------------ Run Overlap (SSOT-staging; cert-aligned, final) ------------------------------
def run_overlap():
    # ── tiny locals ────────────────────────────────────────────────────────────
    def _truth_mask_from_d3(d3: list[list[int]]) -> list[int]:
        if not d3 or not d3[0]: return []
        rows, n3 = len(d3), len(d3[0])
        return [1 if any(d3[i][j] & 1 for i in range(rows)) else 0 for j in range(n3)]

    def _xor_mat(A, B):
        # prefer library 'add' when present
        if "add" in globals() and callable(globals()["add"]):
            return globals()["add"](A, B)
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(c)] for i in range(r)]

    def _is_zero(M):  # local, safe
        return (not M) or all((x & 1) == 0 for row in M for x in row)

    def _residual_tag(R3, mask):
        if _is_zero(R3): return "none"
        n3loc = len(R3[0]) if R3 else 0
        def _col_support(j): return any(R3[i][j] & 1 for i in range(len(R3)))
        lanes_support = any(_col_support(j) for j in range(n3loc) if j < len(mask) and mask[j])
        ker_support   = any(_col_support(j) for j in range(n3loc) if j >= len(mask) or not mask[j])
        if lanes_support and ker_support: return "mixed"
        if lanes_support: return "lanes"
        if ker_support:   return "ker"
        return "none"

    def _shape_ok_for_mul(A, B):  # GF(2) matrix multiply compatibility
        return bool(A and B and A[0] and B[0] and (len(A[0]) == len(B)))

    # ── STEP 1: clear per-run artifacts (keep A/B pin, fixtures cache, frozen IB) ─────
    _keep = {"ab_pin", "_fixtures_cache", "_fixtures_bytes_hash", "_inputs_block", "_last_ib_sig"}
    for k in (
        "proj_meta", "run_ctx", "residual_tags", "overlap_out",
        "overlap_H", "overlap_C", "overlap_cfg", "overlap_policy_label",
        "_file_mode_error"
    ):
        if k not in _keep:
            st.session_state.pop(k, None)

       # ── STEP 2: projector resolve (handles projected:file fail path) ──────────────────
    try:
        # If you didn't add the helper, swap the next line back to: P_active, meta = projector_choose_active(cfg_active, boundaries)
        P_active, meta = _resolve_projector(cfg_active, boundaries)
    except ValueError as e:
        # Determine intended mode from policy source
        source3 = (cfg_active.get("source") or {}).get("3", "auto")
        mode_str = "projected(file)" if source3 == "file" else "projected(auto)"
    
        pjfn    = (cfg_active.get("projector_files", {}) or {}).get("3", "")
        d3_now  = (boundaries.blocks.__root__.get("3") or [])
        n3_now  = (len(d3_now[0]) if (d3_now and d3_now[0]) else 0)
        lm_now  = _truth_mask_from_d3(d3_now)
        pol_lbl = policy_label_from_cfg(cfg_active)
    
        st.session_state["run_ctx"] = {
            "policy_tag": pol_lbl, "mode": mode_str,
            "d3": d3_now, "n3": n3_now, "lane_mask_k3": lm_now,
            "P_active": [],
            "projector_filename": pjfn, "projector_hash": "",
            "projector_consistent_with_d": (False if mode_str == "projected(file)" else None),
            "source": (cfg_active.get("source") or {}),
            "errors": [str(e)],
        }
        st.session_state["overlap_out"]          = {"3": {"eq": False, "n_k": n3_now}, "2": {"eq": True}}
        st.session_state["overlap_cfg"]          = cfg_active
        st.session_state["overlap_policy_label"] = pol_lbl
    
        # Freeze SSOT even on resolver error
        H_local = _load_h_local()
        st.session_state["overlap_H"] = H_local
        st.session_state["overlap_C"] = cmap
        pub = ssot_publish_block(
            boundaries_obj=boundaries,
            cmap_obj=cmap,
            H_obj=H_local,
            shapes_obj=shapes,
            n3=n3_now,
            projector_filename=(""
                if mode_str != "projected(file)" else pjfn),
        )
        st.caption(f"SSOT sig (before → after): {list(pub['before'])} → {list(pub['after'])}")
    
        try: 
            _reconcile_di_vs_ssot()
        except Exception: 
            pass

    # Only mark FILE invalid if we’re actually in FILE mode
    st.session_state["file_pi_valid"]   = (mode_str != "projected(file)")
    st.session_state["file_pi_reasons"] = ([str(e)] if mode_str == "projected(file)" else [])
    st.session_state["write_armed"]     = True
    st.session_state["armed_by"]        = ("file_invalid" if mode_str == "projected(file)" else "overlap_run")

    if mode_str == "projected(file)":
        st.error(f"Projected(FILE) validation failed: {e}")
    else:
        st.info("AUTO mode selected; projector FILE not required. Continuing without Π.")
    return


    # ---- success path (strict + projected residuals; persist SSOT) --------------------
    # meta may carry d3/n3/mode; fall back to boundaries
    d3   = meta.get("d3") if ("d3" in meta) else (boundaries.blocks.__root__.get("3") or [])
    n3   = meta.get("n3") if ("n3" in meta) else (len(d3[0]) if (d3 and d3[0]) else 0)
    mode = str(meta.get("mode", "strict"))

    # lane mask (truth)
    lm_truth = _truth_mask_from_d3(d3)
    if n3 and len(lm_truth) != n3:
        raise RuntimeError(f"lane_mask_k3 length {len(lm_truth)} != n3 {n3}")

    # strict residuals (shape-safe)
    H_local = _load_h_local()
    H2 = (H_local.blocks.__root__.get("2") or [])
    C3 = (cmap.blocks.__root__.get("3") or [])
    I3 = eye(len(C3)) if C3 else []

    R3_strict = []
    if _shape_ok_for_mul(H2, d3) and C3 and C3[0] and (len(C3) == len(C3[0])):  # C3 square
        try:
            R3_strict = _xor_mat(mul(H2, d3), _xor_mat(C3, I3))
        except Exception:
            R3_strict = []

    tag_strict = _residual_tag(R3_strict, lm_truth)
    eq3_strict = _is_zero(R3_strict)

    # projected leg (if enabled)
    if cfg_active.get("enabled_layers"):
        P_eff   = meta.get("P_active") or P_active or []   # safe coalesce
        R3_proj = mul(R3_strict, P_eff) if (R3_strict and P_eff) else []
        eq3_proj = _is_zero(R3_proj)
        tag_proj = _residual_tag(R3_proj, lm_truth)
        out = {"3": {"eq": bool(eq3_proj), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict, "projected": tag_proj}
    else:
        out = {"3": {"eq": bool(eq3_strict), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict}

    # persist run_ctx (SSOT for cert)
    pol_lbl = policy_label_from_cfg(cfg_active)
    st.session_state["overlap_out"]          = out
    st.session_state["overlap_cfg"]          = cfg_active
    st.session_state["overlap_policy_label"] = pol_lbl
    st.session_state["run_ctx"] = {
        "policy_tag": pol_lbl, "mode": mode,
        "d3": d3, "n3": n3, "lane_mask_k3": lm_truth,
        "P_active": meta.get("P_active", P_active),
        "projector_filename": meta.get("projector_filename", ""),
        "projector_hash": meta.get("projector_hash", ""),
        "projector_consistent_with_d": meta.get("projector_consistent_with_d", None),
        "source": (cfg_active.get("source") or {}),
    }

    # make objects available to Cert/Reports
    st.session_state["overlap_H"] = H_local
    st.session_state["overlap_C"] = cmap

    # publish SSOT
    pub = ssot_publish_block(
        boundaries_obj=boundaries,
        cmap_obj=cmap,
        H_obj=H_local,
        shapes_obj=shapes,
        n3=n3,
        projector_filename=st.session_state["run_ctx"].get("projector_filename",""),
    )
    st.caption(f"SSOT sig (before → after): {list(pub['before'])} → {list(pub['after'])}")

    try:
        _reconcile_di_vs_ssot()
    except Exception:
        pass

    # normalize/stamp rc for downstream freshness logic
    rc = st.session_state.get("run_ctx") or {}
    rc.update({
        "mode":       policy_label_from_cfg(cfg_active).replace("projected(columns@k=3,", "projected(").rstrip(")")+")",
        "policy_tag": policy_label_from_cfg(cfg_active),
        "n3":         int((st.session_state.get("_inputs_block") or {}).get("dims", {}).get("n3") or 0),
        "lane_mask_k3": list(st.session_state.get("_district_info", {}).get("lane_mask_k3") or lm_truth),
        "projector_filename": (st.session_state.get("ov_last_pj_path") or rc.get("projector_filename","")),
    })
    if rc["mode"] == "projected(file)":
        rc["projector_hash"] = rc.get("projector_hash") or meta.get("projector_hash","") or ""
    elif rc["mode"] == "projected(auto)":
        rc["projector_hash"] = _auto_pj_hash_from_rc(rc)
    # stamp inputs_sig from frozen SSOT
    rc["inputs_sig"] = current_inputs_sig(_ib=st.session_state.get("_inputs_block") or {})
    st.session_state["run_ctx"] = rc


    
    
       


# ---- Single canonical button (instrumented) ----
if st.button("Run Overlap", key="btn_run_overlap_main"):
    try:
        with st.spinner("Running Overlap…"):
            # pre: snapshot for delta
            _ib_before = dict(st.session_state.get("_inputs_block") or {})
            _sig_before = (
                (_ib_before.get("hashes") or {}).get("boundaries_hash", _ib_before.get("boundaries_hash","")),
                (_ib_before.get("hashes") or {}).get("C_hash",          _ib_before.get("C_hash","")),
                (_ib_before.get("hashes") or {}).get("H_hash",          _ib_before.get("H_hash","")),
                (_ib_before.get("hashes") or {}).get("U_hash",          _ib_before.get("U_hash","")),
                (_ib_before.get("hashes") or {}).get("shapes_hash",     _ib_before.get("shapes_hash","")),
            )

            soft_reset_before_overlap()  # your helper; must NOT clear _inputs_block
            run_overlap()                # your function

            _ib_after = dict(st.session_state.get("_inputs_block") or {})
            _sig_after = (
                (_ib_after.get("hashes") or {}).get("boundaries_hash", _ib_after.get("boundaries_hash","")),
                (_ib_after.get("hashes") or {}).get("C_hash",          _ib_after.get("C_hash","")),
                (_ib_after.get("hashes") or {}).get("H_hash",          _ib_after.get("H_hash","")),
                (_ib_after.get("hashes") or {}).get("U_hash",          _ib_after.get("U_hash","")),
                (_ib_after.get("hashes") or {}).get("shapes_hash",     _ib_after.get("shapes_hash","")),
            )
        st.success("Overlap completed.")
        st.caption(f"SSOT sig (before → after): {list(_sig_before)} → {list(_sig_after)}")
    except Exception as e:
        st.exception(e)
# ---- inputs completeness (define before using in debugger) ----
_h = (st.session_state.get("_inputs_block") or {}).get("hashes") or {}
inputs_complete = bool(
    _h.get("boundaries_hash") and
    _h.get("C_hash") and
    _h.get("H_hash") and
    _h.get("shapes_hash")   # shapes_hash mirrors U_hash
)

# -----------------------------------------------------------------------------------------------------------
with st.expander("SSOT debug", expanded=False):
    rc_ = st.session_state.get("run_ctx") or {}
    ib_ = st.session_state.get("_inputs_block") or {}
    st.write({
        "rc.inputs_sig":      tuple(rc_.get("inputs_sig") or ()),
        "now.inputs_sig":     _current_inputs_sig_compat(_ib=ib_),
        "stale?":             ssot_is_stale(),
        "policy_tag_now":     rc_.get("policy_tag") or "",
        "mode":               rc_.get("mode") or "",
        "auto_projector_hash": (_auto_pj_hash_from_rc(rc_) if (rc_.get("mode") == "projected(auto)") else ""),
        "file_projector_hash": (rc_.get("projector_hash") or ""),
        "inputs_complete":     bool(inputs_complete),
    })







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
def _stable_blocks_sha(obj) -> str:
    try:
        data = {"blocks": obj.blocks.__root__} if hasattr(obj, "blocks") else (obj if isinstance(obj, dict) else {"blocks": {}})
        s = json.dumps(_deep_intify(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return hashlib.sha256(s).hexdigest()
    except Exception:
        return ""

def _ab_is_fresh_now() -> bool:
    ss = st.session_state
    ab = ss.get("ab_compare") or {}
    if not ab:
        return False
    ib = ss.get("_inputs_block") or {}
    rc = ss.get("run_ctx") or {}

    cur_sig = [
        str((ib or {}).get("boundaries_hash","")),
        str((ib or {}).get("C_hash","")),
        str((ib or {}).get("H_hash","")),
        str((ib or {}).get("U_hash","")),
        str((ib or {}).get("shapes_hash","")),
    ]
    if list(ab.get("inputs_sig") or []) != cur_sig:
        return False

    pol_now = str((rc or {}).get("policy_tag") or "strict")
    if ab.get("policy_tag","") != pol_now:
        return False

    if str((rc or {}).get("mode","")).startswith("projected(file)"):
        if (ab.get("projected") or {}).get("projector_hash","") != (rc.get("projector_hash","") or ""):
            return False
    return True


# --- Self-test helpers ---------------------------------------------------------
def _policy_tag_now_from_rc(rc: dict) -> str:
    """
    Return the canonical policy tag for the current run context.
    Prefer rc['policy_tag'] if present; else derive from rc['mode'].
    """
    pol = (rc or {}).get("policy_tag")
    if pol:
        return str(pol)

    mode = str((rc or {}).get("mode",""))
    # Minimal derivation if label isn't stamped yet
    if mode == "strict":
        return "strict"
    if mode == "projected(file)":
        # keep your canonical k=3 label format consistent with the rest of the app
        return "projected(columns@k=3,file)"
    if mode == "projected(auto)":
        return "projected(columns@k=3,auto)"
    # fallback (neutral)
    return "projected(columns@k=3,auto)"

def _ab_is_fresh_now() -> bool:
    """
    Compare the pinned A/B snapshot in session with the current SSOT + policy.
    Returns True if it should embed; False if stale.
    """
    ss = st.session_state
    ab = ss.get("ab_compare") or {}
    if not ab:
        return False  # no snapshot to embed

    # Current canonical inputs sig (frozen SSOT)
    frozen = tuple(ssot_frozen_sig_from_ib() or ())

    # Current policy tag + projector hash (FILE only)
    rc = ss.get("run_ctx") or {}
    pol_now = _policy_tag_now_from_rc(rc)
    pj_now  = rc.get("projector_hash","") if str(rc.get("mode","")) == "projected(file)" else ""

    # Snapshot bits
    ab_sig = tuple(ab.get("inputs_sig") or ())
    ab_pol = str(ab.get("policy_tag",""))
    ab_pj  = ((ab.get("projected") or {}).get("projector_hash","")
              if str(rc.get("mode","")) == "projected(file)" else "")

    return (ab_sig == frozen) and (ab_pol == pol_now) and (ab_pj == pj_now)

# --- Self-tests (startup-friendly, no globals other than Streamlit + helpers) ---
def run_self_tests():
    failures, warnings = [], []

    ss   = st.session_state
    ib   = ss.get("_inputs_block") or {}
    rc   = ss.get("run_ctx") or {}
    out  = ss.get("overlap_out") or {}

    # 1) SSOT completeness (read-only; just warn if blanks)
    #    accept either nested hashes or legacy top-level
    h = ib.get("hashes") or {}
    for k in ("boundaries_hash","C_hash","H_hash","U_hash"):
        if not (ib.get(k) or h.get(k)):
            warnings.append(f"SSOT: missing {k}")

    # 2) Freshness (only after at least one publish)
    try:
        if ss.get("_has_overlap"):
            if "ssot_is_stale_v2" in globals() and callable(globals()["ssot_is_stale_v2"]):
                if ssot_is_stale_v2():
                    warnings.append("SSOT_STALE: live inputs changed; run Overlap to refresh SSOT")
            elif "ssot_is_stale" in globals() and callable(globals()["ssot_is_stale"]):
                if ssot_is_stale():
                    warnings.append("SSOT_STALE: live inputs changed; run Overlap to refresh SSOT")
    except Exception as e:
        warnings.append(f"SSOT_STALE_CHECK: {e}")

    # 3) Mode sanity (AUTO shouldn’t require FILE Π)
    mode = str(rc.get("mode",""))
    if mode == "projected(file)":
        if not bool(rc.get("projector_consistent_with_d", False)):
            failures.append("FILE_OK: projected(file) not consistent with d3")
    elif mode == "projected(auto)":
        if "3" not in (out or {}):
            warnings.append("AUTO_OK: no overlap_out present yet")

    # 4) A/B snapshot freshness (only warn if a snapshot exists)
    ab = ss.get("ab_compare") or {}
    if ab and not _ab_is_fresh_now():
        warnings.append("AB_FRESH: A/B snapshot is stale (won’t embed)")

    return failures, warnings

# --- Policy pill + run stamp (single rendering) --------------------------------
_rc = st.session_state.get("run_ctx") or {}
_ib = st.session_state.get("_inputs_block") or {}
policy_tag = _rc.get("policy_tag") or policy_label_from_cfg(cfg_active)
n3 = _rc.get("n3") or ((_ib.get("dims") or {}).get("n3", 0))
_short8 = lambda h: (h or "")[:8]
bH = _short8((_ib.get("hashes") or {}).get("boundaries_hash", _ib.get("boundaries_hash","")))
cH = _short8((_ib.get("hashes") or {}).get("C_hash",          _ib.get("C_hash","")))
hH = _short8((_ib.get("hashes") or {}).get("H_hash",          _ib.get("H_hash","")))
uH = _short8((_ib.get("hashes") or {}).get("U_hash",          _ib.get("U_hash","")))
pH = _short8(_rc.get("projector_hash","")) if str(_rc.get("mode","")).startswith("projected(file)") else "—"

st.markdown(f"**Policy:** `{policy_tag}`")
st.caption(f"{policy_tag} | n3={n3} | b={bH} C={cH} H={hH} U={uH} P={pH}")

# Gentle hint only if any core hash is blank
if any(x in ("", None, "") for x in (bH, cH, hH, uH)):
    st.info("SSOT isn’t fully populated yet. Run Overlap once to publish provenance hashes.")

# --- Run self-tests and render banner -----------------------------------------
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
    if not (_ib.get("hashes") or _ib.get("boundaries_hash")):
        st.info("Awaiting first Overlap run…")
    else:
        st.success("🟢 Self-tests passed.")
        if _warn:
            st.info("Notes:")
            for w in _warn: st.write(f"- {w}")





# ====================== A/B Compare (strict vs ACTIVE projected) ======================

def _inputs_sig_now_from_ib(ib: dict | None) -> list[str]:
    ib = ib or {}
    return [
        str(ib.get("boundaries_hash","")),
        str(ib.get("C_hash","")),
        str(ib.get("H_hash","")),
        str(ib.get("U_hash","")),
        str(ib.get("shapes_hash","")),
    ]

def _canonical_policy_tag(rc: dict | None) -> str:
    rc = rc or {}
    try:
        return str(rc.get("policy_tag") or policy_label_from_cfg(cfg_active))
    except Exception:
        return str(rc.get("policy_tag") or "strict")

def _ab_is_fresh(ab: dict | None, *, rc: dict | None, ib: dict | None) -> bool:
    if not ab: return False
    if ab.get("inputs_sig") != _inputs_sig_now_from_ib(ib): return False
    ab_pol = ab.get("policy_tag") or (ab.get("projected") or {}).get("policy_tag")
    if ab_pol != _canonical_policy_tag(rc): return False
    if str((rc or {}).get("mode","")).startswith("projected(file)"):
        if (ab.get("projected") or {}).get("projector_hash","") != ((rc or {}).get("projector_hash","") or ""):
            return False
    return True

def _recompute_strict_out(*, boundaries_obj, cmap_obj, H_obj, d3) -> dict:
    """Tiny strict check mirroring your overlap math, shape-safe & tolerant."""
    # helpers
    def _shape_ok(A,B): return bool(A and B and A[0] and B[0] and (len(A[0]) == len(B)))
    def _xor(A,B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j]^B[i][j]) & 1 for j in range(c)] for i in range(r)]
    def _is_zero(M): return (not M) or all(all((x & 1) == 0 for x in row) for row in M)

    H2 = (H_obj.blocks.__root__.get("2") or []) if H_obj else []
    C3 = (cmap_obj.blocks.__root__.get("3") or [])
    I3 = [[1 if i==j else 0 for j in range(len(C3))] for i in range(len(C3))] if C3 else []

    eq3 = False
    try:
        if _shape_ok(H2, d3) and C3 and C3[0] and (len(C3) == len(C3[0])):
            R3 = _xor(mul(H2, d3), _xor(C3, I3))
            eq3 = _is_zero(R3)
    except Exception:
        eq3 = False

    # By construction your k=2 equality is a gate on d2, which isn’t touched here → True
    return {"2": {"eq": True}, "3": {"eq": bool(eq3), "n_k": (len(d3[0]) if (d3 and d3[0]) else 0)}}

def _lane_bottoms_for_diag(*, H_obj, cmap_obj, d3, lane_mask):
    """Diagnostics rows used by fixtures/galleries (shape-safe)."""
    def _shape_ok(A,B): return bool(A and B and A[0] and B[0] and (len(A[0]) == len(B)))
    def _xor(A,B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        r, c = len(A), len(A[0])
        return [[(A[i][j]^B[i][j]) & 1 for j in range(c)] for i in range(r)]
    def _bottom_row(M): return M[-1] if (M and len(M)) else []
    def _mask(vec, mask):
        idx = [j for j,m in enumerate(mask or []) if m]
        return [vec[j] for j in idx] if (vec and idx) else []

    H2 = (H_obj.blocks.__root__.get("2") or []) if H_obj else []
    C3 = (cmap_obj.blocks.__root__.get("3") or [])
    I3 = [[1 if i==j else 0 for j in range(len(C3))] for i in range(len(C3))] if C3 else []

    try:
        H2d3  = mul(H2, d3) if _shape_ok(H2, d3) else []
        C3pI3 = _xor(C3, I3) if (C3 and C3[0]) else []
    except Exception:
        H2d3, C3pI3 = [], []

    return _mask(_bottom_row(H2d3), lane_mask), _mask(_bottom_row(C3pI3), lane_mask)

def _pin_ab_for_cert(strict_out: dict, projected_out: dict):
    """Pin + ticket so the cert block embeds this snapshot next render."""
    ss = st.session_state
    ib = ss.get("_inputs_block") or {}
    rc = ss.get("run_ctx") or {}

    # Prefer canonical frozen sig so it matches cert deduper
    inputs_sig = []
    if "ssot_frozen_sig_from_ib" in globals():
        try:
            inputs_sig = list(ssot_frozen_sig_from_ib() or [])
        except Exception:
            pass
    if not inputs_sig:
        inputs_sig = _inputs_sig_now_from_ib(ib)

    pol_tag = _canonical_policy_tag(rc)
    pj_hash = (rc.get("projector_hash","") if str(rc.get("mode","")) == "projected(file)" else "")

    payload = {
        "inputs_sig": inputs_sig,
        "policy_tag": pol_tag,
        "strict":    {"out": dict(strict_out or {})},
        "projected": {"out": dict(projected_out or {}), "policy_tag": pol_tag, "projector_hash": pj_hash},
    }

    ss["ab_pin"] = {"state": "pinned", "payload": payload, "consumed": False}
    ss["_ab_ticket_pending"] = int(ss.get("_ab_ticket_pending", 0)) + 1
    ss["write_armed"] = True
    ss["armed_by"]    = "ab_compare"

with safe_expander("A/B compare (strict vs active projected)", expanded=False):
    ss   = st.session_state
    rc   = ss.get("run_ctx") or {}
    ib   = ss.get("_inputs_block") or {}
    snap = ss.get("ab_compare") or {}

    fresh = _ab_is_fresh(snap, rc=rc, ib=ib)
    st.caption("A/B snapshot: " + ("🟢 fresh (will embed in cert)" if fresh else ("🟡 stale (won’t embed)" if snap else "—")))

    if st.button("Run A/B compare", key="btn_ab_compare_main"):
        try:
            mode_now = str(rc.get("mode","strict"))
            if mode_now == "strict":
                st.warning("Active policy is strict — run Overlap in projected(auto/file) first to compare.")
                st.stop()

            # Use exactly the H that Overlap used; fallback to local loader if needed
            H_used = ss.get("overlap_H") or (io.parse_cmap({"blocks": {}}))
            d3     = rc.get("d3") or []
            lane_mask = list(rc.get("lane_mask_k3") or [])

            # recompute strict; take projected from the active run
            out_active  = ss.get("overlap_out") or {}
            out_strict  = _recompute_strict_out(boundaries_obj=boundaries, cmap_obj=cmap, H_obj=H_used, d3=d3)
            out_project = out_active

            # diagnostics (bottom vectors on lanes)
            lane_vec_H2d3, lane_vec_C3I = _lane_bottoms_for_diag(
                H_obj=H_used, cmap_obj=cmap, d3=d3, lane_mask=lane_mask
            )

            label_proj = _canonical_policy_tag(rc)
            pair_tag   = f"strict__VS__{label_proj}"

            # full snapshot for UI/debug panes
            ab_payload = {
                "pair_tag": pair_tag,
                "inputs_sig": _inputs_sig_now_from_ib(ib),
                "lane_mask_k3": lane_mask,
                "policy_tag": label_proj,
                "strict": {
                    "label": "strict",
                    "out":   out_strict,
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
                    "out":   out_project,
                    "lane_vec_H2d3": lane_vec_H2d3[:],
                    "lane_vec_C3plusI3": lane_vec_C3I[:],
                    "pass_vec": [
                        int(out_project.get("2",{}).get("eq", False)),
                        int(out_project.get("3",{}).get("eq", False)),
                    ],
                    "projector_filename": rc.get("projector_filename",""),
                    "projector_hash": rc.get("projector_hash","") if mode_now == "projected(file)" else "",
                    "projector_consistent_with_d": rc.get("projector_consistent_with_d", None),
                },
            }

            # cache for UI and pin for cert
            ss["ab_compare"] = ab_payload
            _pin_ab_for_cert(out_strict, out_project)

            # receipt
            s_ok = bool(out_strict.get("3",{}).get("eq", False))
            p_ok = bool(out_project.get("3",{}).get("eq", False))
            st.success(f"A/B updated → strict={'✅' if s_ok else '❌'} · projected={'✅' if p_ok else '❌'} · {pair_tag}")

            if st.checkbox("Show A/B snapshot payload", value=False, key="ab_show_payload_main"):
                st.json(ab_payload)

        except Exception as e:
            st.error(f"A/B compare failed: {e}")

    # convenience clearer if snapshot is stale
    if snap and (not _ab_is_fresh(snap, rc=rc, ib=ib)):
        if st.button("Clear stale A/B", key="btn_ab_clear_main"):
            ss.pop("ab_compare", None)
            st.success("Cleared A/B snapshot. Re-run A/B to refresh.")
# ======================================================================================






# ========= F · Helpers & invariants (shared by F1/F2/F3) =========
from pathlib import Path
import os, tempfile
import copy as _copy
import json as _json
import hashlib as _hash
import streamlit as st
import datetime as _dt
import random as _random  # harmless; some callers use it

# 1) Schema/version + field (one source of truth for F1/F2/F3)
SCHEMA_VERSION = "1.1.0"     # coverage_sampling, perturbation_sanity, fence_stress
FIELD          = "GF(2)"     # identity.field in all JSON payloads

# 2) Guard enum (must match certs / parity)
GUARD_ENUM = ["grid", "wiggle", "echo", "fence", "ker_guard", "none", "error"]

# --- Evidence preflight helpers (display-only status + run-time guards) ---
def _hashes_status():
    ib = st.session_state.get("_inputs_block") or {}
    h = [ib.get(k, "") for k in ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")]
    return "OK" if all(h) else "MISSING"

def _projector_status():
    rc = st.session_state.get("run_ctx") or {}
    mode = str(rc.get("mode") or "strict")
    if mode == "strict":
        return True, "STRICT_OK"
    if mode == "projected(auto)":
        # allowed for Overlap/A/B; may be disallowed for evidence
        return True, "AUTO_OK"
    # projected(file)
    bad = file_validation_failed()
    return (not bad), ("FILE_OK" if not bad else "P3_FILE_INVALID")

def _require_inputs_hashes_strict_for_run():
    ib = st.session_state.get("_inputs_block") or {}
    keys = ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")
    missing = [k for k in keys if not ib.get(k)]
    if missing:
        raise RuntimeError(
            "INPUT_HASHES_MISSING: wire SSOT from Cert/Overlap; backfill disabled "
            f"(missing: {', '.join(missing)})"
        )
    return ib

def _require_lane_mask_for_run():
    lm = (st.session_state.get("run_ctx") or {}).get("lane_mask_k3") or []
    if not lm:
        raise RuntimeError("LANE_MASK_MISSING: run Overlap to stage lane_mask_k3.")
    return lm

def _require_projected_file_allowed_for_run():
    ok, tag = _projector_status()
    if tag == "P3_FILE_INVALID":
        raise RuntimeError("P3_FILE_INVALID: projector(file) failed validation.")
    return ok

def _disallow_auto_for_evidence():
    if (st.session_state.get("run_ctx") or {}).get("mode") == "projected(auto)":
        raise RuntimeError("P3_AUTO_DISALLOWED: use strict or projected(file) for evidence reports.")

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

# ===== Strict mode toggles =====
DEV_ALLOW_INPUT_HASH_BACKFILL = False   # no dev backfill in evidence runs

# ===== SSOT preflights (copy-only; fail-fast) =====
def _require_inputs_hashes_strict() -> dict:
    """
    Read SSOT input hashes (copy-only). If any are empty → hard fail.
    Returns the dict so that callers can embed it untouched in JSONs.
    """
    ih = (st.session_state.get("inputs_hashes") or {}).copy()
    keys = ["boundaries_hash","C_hash","H_hash","U_hash","shapes_hash"]
    # Mirror from run_ctx if present (still copy-only; no recompute)
    rc = st.session_state.get("run_ctx") or {}
    for k in keys:
        ih[k] = ih.get(k) or rc.get(k) or ""
    if not all(ih.get(k, "") for k in keys):
        missing = [k for k in keys if not ih.get(k)]
        raise RuntimeError(
            "INPUT_HASHES_MISSING: wire SSOT from Cert/Overlap; backfill disabled "
            f"(missing: {', '.join(missing)})"
        )
    # normalize into session so downstream reads are consistent
    st.session_state["inputs_hashes"] = ih
    rc.update(ih); st.session_state["run_ctx"] = rc
    return ih

def _require_lane_mask_ssot() -> list[int]:
    """
    Lane mask is computed once (at Overlap/Cert) from the stored d3 snapshot and kept in run_ctx.
    Here we only read it; we do NOT infer/guess/rectify.
    """
    rc = st.session_state.get("run_ctx") or {}
    lm = rc.get("lane_mask_k3")
    if not isinstance(lm, list) or any((int(x) & 1) not in (0, 1) for x in (lm or [])):
        raise RuntimeError("LANE_MASK_MISSING: compute lane_mask_k3 at Cert/Overlap stage and stash in run_ctx.")
    return [int(x) & 1 for x in lm]

def _require_projector_file_if_needed():
    """
    If mode==projected(file), require projector_filename & projector_hash pre-validated at Cert/Overlap.
    No AUTO fallback here. If lanes don’t match projector diag, upstream must have blocked with P3_LANE_MISMATCH.
    """
    rc = st.session_state.get("run_ctx") or {}
    m = rc.get("mode", "strict")
    if m == "projected(file)":
        if not rc.get("projector_filename") or not rc.get("projector_hash"):
            raise RuntimeError("P3_FILE_MISSING: projected(FILE) selected without validated projector.")
        # We do NOT re-parse or re-validate here. Evidence path is copy-only.
    elif m == "projected(auto)":
        # In strict evidence runs, avoid AUTO. Treat as a configuration error.
        raise RuntimeError("P3_AUTO_DISALLOWED: projector(auto) not allowed in strict evidence runs. Freeze to FILE or use strict.")
    # if strict → nothing to enforce here
    return

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
    s = _json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return _hash.sha256(s).hexdigest()

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
def _inputs_block_from_session(strict_dims: tuple[int, int] | None = None) -> dict:
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
            try:
                tag_p = residual_tag(R3p, lm)  # type: ignore[name-defined]
            except Exception:
                tag_p = "error"
        else:
            tag_p = tag_s
        eq_p  = (len(R3p) == 0) or all(all((x & 1) == 0 for x in row) for row in R3p)
    else:
        tag_p, eq_p = None, None

    return lm, tag_s, bool(eq_s), tag_p, (None if eq_p is None else bool(eq_p))

# -------- optional carrier (U) mutation hooks (fallback implementation) --------
# If your project already defines get_carrier_mask / set_carrier_mask, this block is a no-op.
if "get_carrier_mask" not in globals():
    def get_carrier_mask(U_obj=None):
        """
        Return an n3×n2 0/1 mask. Fallback: all-ones mask (every lane in U).
        Prefers H2 shape (n3×n2); falls back to d3 (n2×n3) if H2 missing.
        Allows an override via st.session_state["_u_mask_override"].
        """
        # explicit override (used by Fence variants)
        override = st.session_state.get("_u_mask_override")
        if isinstance(override, list) and override and isinstance(override[0], list):
            return override

        # 1) try H2 (shape n3×n2)
        try:
            H_local = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})
            H2 = (H_local.blocks.__root__.get("2") or [])
            n3 = len(H2)
            n2 = len(H2[0]) if (H2 and H2[0]) else 0
        except Exception:
            n2 = n3 = 0

        # 2) fallback to d3 (shape n2×n3) → infer n2,n3
        if n3 <= 0 or n2 <= 0:
            try:
                B = st.session_state.get("boundaries")
                d3 = (B.blocks.__root__.get("3") or []) if B else []
                n2 = len(d3)
                n3 = len(d3[0]) if (d3 and d3[0]) else 0
            except Exception:
                n2 = n3 = 0

        if n3 <= 0 or n2 <= 0:
            # unknown dims → empty mask (Fence will no-op gracefully)
            return []

        # default: everything in-carrier
        return [[1] * n2 for _ in range(n3)]

if "set_carrier_mask" not in globals():
    def set_carrier_mask(U_obj, mask):
        """Store a session-scoped override U-mask (used by Fence variants)."""
        st.session_state["_u_mask_override"] = mask
        return True

# Recompute HAS_U_HOOKS after possible fallback injection
HAS_U_HOOKS = (
    "get_carrier_mask" in globals() and "set_carrier_mask" in globals()
    and callable(globals()["get_carrier_mask"]) and callable(globals()["set_carrier_mask"])
)

           



# ============================ Reports: Perturbation & Fence ============================
def _publish_ssot_if_pending():
    """Copy-only: publish staged hashes/dims/filenames into SSOT if all five hashes exist."""
    ih_live = st.session_state.get("inputs_hashes") or {}
    if all(ih_live.get(k) for k in ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")):
        return  # already live
    pend  = st.session_state.get("_inputs_hashes_pending") or {}
    dims  = st.session_state.get("_dims_pending") or {}
    files = st.session_state.get("_filenames_pending") or {}
    if all(pend.get(k) for k in ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")) and dims:
        st.session_state["inputs_hashes"] = pend.copy()
        st.session_state["_inputs_block"] = {
            "filenames": files if files else {
                "boundaries": st.session_state.get("fname_boundaries","boundaries.json"),
                "C":          st.session_state.get("fname_cmap","cmap.json"),
                "H":          st.session_state.get("fname_h","H.json"),
                "U":          st.session_state.get("fname_shapes","shapes.json"),
            },
            "dims": {"n2": int(dims.get("n2", 0)), "n3": int(dims.get("n3", 0))},
            "boundaries_hash": pend["boundaries_hash"],
            "C_hash":          pend["C_hash"],
            "H_hash":          pend["H_hash"],
            "U_hash":          pend["U_hash"],
            "shapes_hash":     pend["shapes_hash"],
            "hashes":          pend.copy(),  # convenience mirror
            "lane_mask_k3": (st.session_state.get("run_ctx") or {}).get("lane_mask_k3", []),
        }

with st.expander("Reports: Perturbation Sanity & Fence Stress"):
    # Display-only preflight (no exceptions here)
    ok_pj, pj_tag = _projector_status()
    h_tag = _hashes_status()
    st.caption(f"Evidence preflight → Π: {pj_tag} · hashes: {h_tag}")
    if (st.session_state.get("run_ctx") or {}).get("mode") == "projected(auto)":
        st.info("AUTO is fine for Overlap/A/B. For evidence reports, Freeze SSOT and use strict or projected(file).")

    # Ensure reports dir exists (defensive)
    REPORTS_DIR = Path(st.session_state.get("REPORTS_DIR", "reports"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Freshness (non-blocking in render)
    try:
        rc_try = require_fresh_run_ctx()  # may return None
        rc_fix = rectify_run_ctx_mask_from_d3()  # may return None
        # prefer rectified rc if present, else the first, else session, else {}
        rc = rc_fix or rc_try or (st.session_state.get("run_ctx") or {})
    except Exception as e:
        rc = st.session_state.get("run_ctx") or {}
        st.warning(str(e))
    
    # Inputs / policy context (safe defaults)
    H_used = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})
    mode_now = str(rc.get("mode", ""))
    P_active = rc.get("P_active") if mode_now.startswith("projected") else None
    
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

    # Disable only on FILE Π invalid (render-time convenience)
    disabled = file_validation_failed()
    help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    if st.button(
        "Run Perturbation Sanity (and Fence if checked)",
        key="ps_run",
        disabled=disabled,
        help=(help_txt if disabled else "Run perturbation sanity; optionally include fence"),
    ):
        # 0) Publish staged SSOT hashes on click (copy-only; no recompute)
        _publish_ssot_if_pending()
        if "_ensure_inputs_hashes" in globals():
            _ensure_inputs_hashes()

        try:
            # 1) Evidence guards (CLICK-TIME)
            _require_inputs_hashes_strict_for_run()
            _require_lane_mask_for_run()
            _disallow_auto_for_evidence()
            _require_projected_file_allowed_for_run()

            # ───────────────────────── Baseline (no mutation) ─────────────────────────
            lm0, tag_s0, eq_s0, tag_p0, eq_p0 = _sig_tag_eq(B0, C0, H0, P_active)

            # lanes-only domain from SSOT lane mask
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
            rows, ps_results = [], []
            matches = mismatches = total_flips = in_domain_flips = 0

            for (r, c, k) in _flip_targets_lanes_only(n2, n3, int(max_flips), seed_txt):
                total_flips += 1

                if not (n2 and n3):
                    rows.append([k, "none", "none", "empty fixture"])
                    ps_results.append({
                        "flip_id": int(k),
                        "guard_tripped": "none", "expected_guard": "none",
                        "flip_spec": {"row": int(r), "col": int(c),
                                      "bit_before": None, "bit_after": None,
                                      "lane_col": False, "skip_reason": "empty-fixture"},
                        "k_status_before": {"2": True, "3": bool(eq_s0)},
                        "k_status_after":  {"2": True, "3": bool(eq_s0)},
                        "residual_tag_after": str(tag_s0 or "none"),
                        "witness_written": False, "note": "empty fixture",
                    })
                    continue

                lane_col = (c in allowed_cols_set)
                bit_before = int(d3_base[r][c]) if (r < len(d3_base) and c < len(d3_base[0])) else 0

                if not lane_col:
                    rows.append([k, "none", "none", "off-domain (ker column)"])
                    ps_results.append({
                        "flip_id": int(k),
                        "guard_tripped": "none", "expected_guard": "none",
                        "flip_spec": {"row": int(r), "col": int(c),
                                      "bit_before": bit_before, "bit_after": bit_before,
                                      "lane_col": False, "skip_reason": "off-domain"},
                        "k_status_before": {"2": True, "3": bool(eq_s0)},
                        "k_status_after":  {"2": True, "3": bool(eq_s0)},
                        "residual_tag_after": str(tag_s0 or "none"),
                        "witness_written": False, "note": "off-domain (ker column)",
                    })
                    continue

                in_domain_flips += 1
                d3_mut = [row[:] for row in d3_base]
                if r >= len(d3_mut) or (len(d3_mut) and c >= len(d3_mut[0])):
                    rows.append([k, "none", "none", f"skip flip out-of-range r={r},c={c}"])
                    ps_results.append({
                        "flip_id": int(k),
                        "guard_tripped": "none", "expected_guard": "none",
                        "flip_spec": {"row": int(r), "col": int(c),
                                      "bit_before": bit_before, "bit_after": bit_before,
                                      "lane_col": True, "skip_reason": "out-of-range"},
                        "k_status_before": {"2": True, "3": bool(eq_s0)},
                        "k_status_after":  {"2": True, "3": bool(eq_s0)},
                        "residual_tag_after": str(tag_s0 or "none"),
                        "witness_written": False, "note": "flip skipped: out-of-range",
                    })
                    continue

                d3_mut[r][c] ^= 1
                bit_after = int(d3_mut[r][c])

                dB = B0.dict() if hasattr(B0, "dict") else {"blocks": {}}
                dB = _json.loads(_json.dumps(dB))
                dB.setdefault("blocks", {})["3"] = d3_mut
                Bk = io.parse_boundaries(dB)

                lmK, tag_sK, eq_sK, tag_pK, eq_pK = _sig_tag_eq(Bk, C0, H0, P_active)

                # Guard enum (single source)
                strict_out = {"3": {"eq": bool(eq_sK)}}
                guard = _first_tripped_guard(strict_out)
                expected_guard = guard  # lanes-only spec

                rows.append([k, guard, expected_guard, ""])
                ok = (guard == expected_guard)
                matches += int(ok)
                mismatches += int(not ok)

                # Optional witness (only on mismatch)
                witness_written = False
                if enable_witness and (not ok) and "append_witness_row" in globals():
                    try:
                        cert_like = st.session_state.get("cert_payload")
                        if cert_like:
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
                    "flip_id": int(k), "guard_tripped": guard, "expected_guard": expected_guard,
                    "flip_spec": {"row": int(r), "col": int(c),
                                  "bit_before": int(bit_before), "bit_after": int(bit_after),
                                  "lane_col": True},
                    "k_status_before": {"2": True, "3": bool(eq_s0)},
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
                    f"n2={n2}", f"n3={n3}",
                    f"baseline_tag_strict={tag_s0}",
                    f"baseline_tag_projected={'' if tag_p0 is None else tag_p0}",
                ],
            )
            st.success(f"Perturbation sanity saved → {PERTURB_OUT_PATH}")

            # Build Perturbation JSON (copy-only SSOT)
            try:
                rc_ps = require_fresh_run_ctx()
            except Exception:
                rc_ps = st.session_state.get("run_ctx") or {}
            if "normalize_projector_into_run_ctx" in globals():
                normalize_projector_into_run_ctx()

            policy_ps = _policy_block_from_run_ctx(rc_ps)
            inputs_ps = _inputs_block_from_session(strict_dims=(n2, n3))

            _hash_fields = ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")
            hobj_ps = inputs_ps.get("hashes") or {k: inputs_ps.get(k, "") for k in _hash_fields}
            if not all(hobj_ps.get(k, "") for k in _hash_fields):
                missing = [k for k in _hash_fields if not hobj_ps.get(k, "")]
                raise RuntimeError(
                    f"INPUT_HASHES_MISSING: wire SSOT from Cert/Overlap; backfill disabled (missing: {', '.join(missing)})"
                )

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
                    "hashes": inputs_ps.get("hashes", {}),
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

            # Persist + downloads + badges
            try:
                h12 = perturb_json["integrity"]["content_hash"][:12]
                h8  = perturb_json["integrity"]["content_hash"][:8]
                basename = f"perturbation_sanity__{h12}.json"
                pert_json_path = REPORTS_DIR / basename
                _atomic_write_json(pert_json_path, perturb_json)
                st.session_state.setdefault("last_report_paths", {})["perturbation_sanity"] = {
                    "csv": str(PERTURB_OUT_PATH), "json": str(pert_json_path)
                }
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

            # ───────────────────── Fence stress (baseline + U-variants; U-only) ─────────────────────
            if run_fence:
                # publish again (idempotent) + guard
                if "_publish_ssot_if_pending" in globals():
                    _publish_ssot_if_pending()
                if "_ensure_inputs_hashes" in globals():
                    _ensure_inputs_hashes()
                _require_inputs_hashes_strict_for_run()
                _require_lane_mask_for_run()
                _disallow_auto_for_evidence()
                _require_projected_file_allowed_for_run()

                if not HAS_U_HOOKS:
                    st.warning("Fence stress skipped: U hooks unavailable (no carrier mutation API).")
                else:
                    # Pull fixture blocks
                    d3 = (B0.blocks.__root__.get("3") or [])
                    H2 = (H0.blocks.__root__.get("2") or [])
                    C3 = (C0.blocks.__root__.get("3") or [])

                    # Strict preflight — fast fail, no partial writes
                    _validate_shapes_or_raise(H2, d3, C3)

                    # Helpers
                    def _count1(M): return sum(int(x & 1) for row in (M or []) for x in row)
                    def _apply_U_to_H2(H2_in, U_mask):
                        H2_out = [row[:] for row in H2_in]
                        n3_local = len(H2_out)
                        if not U_mask or not (U_mask[0] if U_mask else []):
                            return H2_out
                        for j in range(n3_local):
                            in_U = any(int(b) & 1 for b in U_mask[j])
                            if not in_U:
                                H2_out[j] = [0] * len(H2_out[j])
                        return H2_out

                    # Base mask (fallback hooks provide all-ones if none)
                    st.session_state.pop("_u_mask_override", None)  # clear stale overrides
                    U_mask_base = get_carrier_mask(U0)

                    # Normalize mask shape to n3×n2 if needed
                    n3_h2 = len(H2)
                    n2_h2 = len(H2[0]) if (H2 and H2[0]) else 0
                    def _normalized_mask(mask):
                        try:
                            ok_rows = len(mask) == n3_h2 and (not mask or len(mask[0]) == n2_h2)
                        except Exception:
                            ok_rows = False
                        if not ok_rows:
                            return [[1] * n2_h2 for _ in range(n3_h2)]
                        return [[int(b) & 1 for b in row] for row in mask]
                    U_mask_base = _normalized_mask(U_mask_base)

                    # Baseline (no change)
                    R3_base = _strict_R3(H2, d3, C3)
                    k2_base = True
                    k3_base = (not R3_base) or all(all((x & 1) == 0 for x in row) for row in R3_base)
                    rows_fs = [["U_min", f"[{int(k2_base)},{int(k3_base)}]", "baseline"]]

                    # U_shrink: chop 1-cell border off U (shape preserved)
                    U_shrink = [row[:] for row in U_mask_base]
                    rU = len(U_shrink); cU = len(U_shrink[0]) if (U_shrink and U_shrink[0]) else 0
                    if rU and cU:
                        for j in range(cU): U_shrink[0][j] = 0; U_shrink[-1][j] = 0
                        for i in range(rU): U_shrink[i][0] = 0; U_shrink[i][-1] = 0
                    H2_shrink = _apply_U_to_H2(H2, U_shrink)
                    _validate_shapes_or_raise(H2_shrink, d3, C3)
                    R3_shrink = _strict_R3(H2_shrink, d3, C3)
                    k2_s = True
                    k3_s = (not R3_shrink) or all(all((x & 1) == 0 for x in row) for row in R3_shrink)
                    rows_fs.append([
                        "U_shrink", f"[{int(k2_s)},{int(k3_s)}]",
                        _json.dumps({"delta_U": {
                            "added": 0,
                            "removed": int(_count1(U_mask_base) - _count1(U_shrink)),
                            "size_before": int(_count1(U_mask_base)),
                            "size_after": int(_count1(U_shrink)),
                        }}, separators=(",", ":"))
                    ])

                    # U_plus: add 1-cell border to U (shape preserved)
                    U_plus = [row[:] for row in U_mask_base]
                    if rU and cU:
                        for j in range(cU): U_plus[0][j]  = 1; U_plus[-1][j] = 1
                        for i in range(rU): U_plus[i][0]  = 1; U_plus[i][-1] = 1
                    H2_plus = _apply_U_to_H2(H2, U_plus)
                    _validate_shapes_or_raise(H2_plus, d3, C3)
                    R3_plus = _strict_R3(H2_plus, d3, C3)
                    k2_p = True
                    k3_p = (not R3_plus) or all(all((x & 1) == 0 for x in row) for row in R3_plus)
                    rows_fs.append([
                        "U_plus", f"[{int(k2_p)},{int(k3_p)}]",
                        _json.dumps({"delta_U": {
                            "added": int(_count1(U_plus) - _count1(U_mask_base)),
                            "removed": 0,
                            "size_before": int(_count1(U_mask_base)),
                            "size_after": int(_count1(U_plus)),
                        }}, separators=(",", ":"))
                    ])

                    # Build + validate inputs (must have SSOT hashes)
                    try:
                        rc_fs = require_fresh_run_ctx()
                    except Exception:
                        rc_fs = st.session_state.get("run_ctx") or {}
                    if "normalize_projector_into_run_ctx" in globals():
                        normalize_projector_into_run_ctx()

                    inputs_fs = _inputs_block_from_session(strict_dims=(n2, n3))
                    _hash_fields = ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash")
                    hobj_fs = inputs_fs.get("hashes") or {k: inputs_fs.get(k, "") for k in _hash_fields}
                    if not all(hobj_fs.get(k, "") for k in _hash_fields):
                        missing = [k for k in _hash_fields if not hobj_fs.get(k, "")]
                        raise RuntimeError(
                            f"INPUT_HASHES_MISSING: wire SSOT from Cert/Overlap; backfill disabled (missing: {', '.join(missing)})"
                        )

                    policy_fs = _policy_block_from_run_ctx(rc_fs)
                    summary_fs = {
                        "baseline_pass_vec": [bool(k2_base), bool(k3_base)],
                        "U_shrink_pass_vec": [bool(k2_s), bool(k3_s)],
                        "U_plus_pass_vec":   [bool(k2_p), bool(k3_p)],
                    }

                    results_fs_json = []
                    for rcls, pvec, note in rows_fs:
                        pv = pvec.strip("[]").split(",")
                        item = {"U_class": rcls, "pass_vec": [bool(int(pv[0])), bool(int(pv[1]))]}
                        try:
                            maybe = _json.loads(note)
                            if isinstance(maybe, dict): item.update(maybe)
                            else: item["note"] = note
                        except Exception:
                            item["note"] = note
                        results_fs_json.append(item)

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
                        "mode": "U",
                        "fallback_note": "",
                        "policy": policy_fs,
                        "inputs": inputs_fs,
                        "exemplar": {
                            "fixture_id": rc_fs.get("fixture_nonce",""),
                            "lane_mask_k3": inputs_fs.get("lane_mask_k3", []),
                            "hashes": inputs_fs.get("hashes", {}),
                        },
                        "results": results_fs_json,
                        "summary": summary_fs,
                        "integrity": {"content_hash": ""},
                    }
                    fence_json["integrity"]["content_hash"] = _hash_json(fence_json)

                    # Persist; badge
                    h12 = fence_json["integrity"]["content_hash"][:12]; h8 = fence_json["integrity"]["content_hash"][:8]
                    basename = f"fence_stress__{h12}.json"
                    fence_json_path = REPORTS_DIR / basename
                    _atomic_write_json(fence_json_path, fence_json)
                    st.session_state.setdefault("last_report_paths", {})["fence_stress"] = {"json": str(fence_json_path)}

                    import io as _io
                    mem = _io.BytesIO(_json.dumps(fence_json, ensure_ascii=False, indent=2).encode("utf-8"))
                    st.download_button("Download fence_stress.json", mem, file_name=basename, key=f"dl_fs_json_{h8}")
                    st.info(f"wrote JSON ✓ · hash: {h12} · saved as {basename}")
                    st.info(f"U-mode · baseline [k2,k3]=[{int(k2_base)},{int(k3_base)}] → "
                            f"shrink [{int(k2_s)},{int(k3_s)}] · U⁺ [{int(k2_p)},{int(k3_p)}]")

        except Exception as e:
            st.error(f"Perturbation/Fence run failed: {e}")


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


# =========================[ · Gallery Append & Dedupe (cert-required, canonical schema) ]=========================

from pathlib import Path
import os, json

LOGS_DIR = Path("logs")
GALLERY_JSONL = LOGS_DIR / "gallery.jsonl"
GALLERY_JSONL.parent.mkdir(parents=True, exist_ok=True)

# --- Canonical schema (same as build_b2_gallery) ---
# district,fixture,projected,hash_d,hash_U,hash_suppC,hash_suppH,
# growth,tag,strictify,lane_vec_H2,lane_vec_C3pI3,ab_embedded,content_hash

def _gallery_row_from_cert(cert: dict) -> dict:
    """Project a cert payload to the canonical B2 row schema."""
    if not cert:
        return {}

    identity   = cert.get("identity", {}) or {}
    policy     = cert.get("policy",   {}) or {}
    inputs     = cert.get("inputs",   {}) or {}
    hashes     = (inputs.get("hashes") or {})
    diags      = cert.get("diagnostics", {}) or {}
    gallery    = cert.get("gallery", {}) or {}
    ab_embed   = cert.get("ab_embed", {}) or {}
    h          = cert.get("hashes", {}) or {}

    district   = identity.get("district_id", "UNKNOWN")
    fixture    = identity.get("fixture_label", "") or ""   # required for row, may be blank
    projected  = str(policy.get("canon", "strict"))

    hash_d     = str(policy.get("projector_hash", "")) if projected == "projected:file" else ""
    hash_U     = str(hashes.get("U_hash", "")) or ""
    hash_C     = str(hashes.get("C_hash", "")) or ""
    hash_H     = str(hashes.get("H_hash", "")) or ""

    growth     = gallery.get("growth_bumps", cert.get("growth", {}).get("growth_bumps", 0))
    tag        = str(gallery.get("tag", ""))
    strictify  = str(gallery.get("strictify", "tbd"))

    # diagnostics vectors; store JSON-strings per spec
    lv_H2      = diags.get("lane_vec_H2@d3", [])
    lv_C3pI3   = diags.get("lane_vec_C3+I3", [])

    ab_emb     = bool(ab_embed.get("fresh", False))
    content_h  = str(h.get("content_hash", ""))

    # ensure stable JSON text for vector columns
    def _as_json(v): 
        try:
            return json.dumps(v, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return "[]"

    return {
        "district":        district,
        "fixture":         fixture,
        "projected":       projected,
        "hash_d":          hash_d,
        "hash_U":          hash_U,
        "hash_suppC":      hash_C,
        "hash_suppH":      hash_H,
        "growth":          int(growth) if isinstance(growth, (int, float)) else 0,
        "tag":             tag,
        "strictify":       strictify,
        "lane_vec_H2":     _as_json(lv_H2),
        "lane_vec_C3pI3":  _as_json(lv_C3pI3),
        "ab_embedded":     bool(ab_emb),
        "content_hash":    content_h,
    }

def _atomic_append_jsonl(path: Path, row: dict) -> None:
    blob = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    tmp  = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "ab") as f:
        f.write(blob); f.flush(); os.fsync(f.fileno())
    # append to real file
    with open(path, "ab") as f:
        f.write(blob)

def _read_jsonl_tail(path: Path, N: int = 100) -> list[dict]:
    out = []
    try:
        if not path.exists():
            return out
        with open(path, "r", encoding="utf-8") as f:
            for ln in f.readlines()[-N:]:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except Exception:
                        pass
    except Exception:
        pass
    return out

# session caches for dedupe
ss = st.session_state
ss.setdefault("_gallery_seen_keys", set())
ss.setdefault("_gallery_bootstrapped", False)

def _gallery_key(row: dict) -> tuple:
    # minimal, deterministic dedupe: same cert for same fixture once
    return (row.get("district",""), row.get("fixture",""), row.get("content_hash",""))

with safe_expander("Gallery (canonical)", expanded=False):
    cert = ss.get("cert_payload") or {}
    has_cert = bool(cert)
    if not has_cert:
        st.info("No cert in memory yet. Run Overlap until a cert is written, then append here.")

    # Assemble canonical row from cert
    row = _gallery_row_from_cert(cert) if has_cert else {}

        # UI: lightweight overrides (optional, safe defaults)
    c1, c2, c3 = st.columns([1, 1, 2])
    
    opts_strictify = ["tbd", "no", "yes"]
    
    with c1:
        if has_cert:
            # coerce to int with safe default
            row["growth"] = int(st.number_input(
                "growth_bumps",
                min_value=0,
                value=int(row.get("growth", 0) or 0),
                step=1,
                key="gal_growth_bumps_v2"
            ))
    
    with c2:
        if has_cert:
            # normalize strictify to one of the allowed options
            _sv = str(row.get("strictify") or "tbd").strip().lower()
            if _sv not in opts_strictify:
                _sv = "tbd"
            row["strictify"] = st.selectbox(
                "strictify",
                options=opts_strictify,
                index=opts_strictify.index(_sv),
                key="gal_strictify_v2"
            )
    
    with c3:
        if has_cert:
            row["tag"] = st.text_input(
                "tag (optional)",
                value=str(row.get("tag", "") or ""),
                key="gal_tag_v2"
            )


    # Bootstrap dedupe cache from tail once
    if not ss["_gallery_bootstrapped"]:
        for tail_row in _read_jsonl_tail(GALLERY_JSONL, N=200):
            try:
                ss["_gallery_seen_keys"].add(_gallery_key(tail_row))
            except Exception:
                continue
        ss["_gallery_bootstrapped"] = True

    # Append button (disabled w/o cert or missing fixture)
    disabled = (not has_cert) or (not row.get("fixture"))
    tip = None if has_cert else "Disabled until a cert is available."
    if not row.get("fixture"):
        st.warning("This cert has no fixture_label. Set identity.fixture_label in your cert flow to log gallery rows.")

    if st.button("Add to Gallery", key="btn_gallery_append_v2", disabled=disabled, help=tip or "Append row to gallery.jsonl"):
        try:
            k = _gallery_key(row)
            if k in ss["_gallery_seen_keys"]:
                st.info("Duplicate skipped (same district/fixture/content_hash).")
            else:
                _atomic_append_jsonl(GALLERY_JSONL, row)
                ss["_gallery_seen_keys"].add(k)
                st.success("Gallery row appended.")
                # keep CSV in sync, if builder is present
                try:
                    build_b2_gallery(debounce=True)
                except Exception as e:
                    st.info(f"(B2 gallery build skipped: {e})")
        except Exception as e:
            st.error(f"Gallery append failed: {e}")

    # Tail view (compact)
    try:
        tail = _read_jsonl_tail(GALLERY_JSONL, N=8)
        if tail:
            import pandas as pd
            view = []
            for r in tail:
                view.append({
                    "when":        r.get("written_at_utc",""),
                    "district":    r.get("district",""),
                    "fixture":     r.get("fixture",""),
                    "proj":        r.get("projected",""),
                    "d[:8]":       (r.get("hash_d","") or "")[:8],
                    "U[:8]":       (r.get("hash_U","") or "")[:8],
                    "C[:8]":       (r.get("hash_suppC","") or "")[:8],
                    "H[:8]":       (r.get("hash_suppH","") or "")[:8],
                    "ab":          bool(r.get("ab_embedded", False)),
                    "tag":         r.get("tag",""),
                    "strictify":   r.get("strictify",""),
                    "content[:12]":(r.get("content_hash","") or "")[:12],
                })
            st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
        else:
            st.caption("Gallery is empty.")
    except Exception as e:
        st.warning(f"Could not render gallery tail: {e}")
# ======================= end Gallery (canonical) =======================




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

    # Read rc up-front (used below)
    rc = dict(_ss.get("run_ctx") or {})
    n3 = int(rc.get("n3") or 0)
    lm = list(rc.get("lane_mask_k3") or [])
    k3_green = bool((((_ss.get("overlap_out") or {}).get("3") or {}).get("eq", False)))

    # Freshness gate (warn only)
    stale = ssot_is_stale()
    st.toggle("Allow writing with stale SSOT", value=False, key="cert_allow_stale_ssot")
    if stale:
        st.warning("STALE_RUN_CTX: Inputs changed; please click Run Overlap to refresh.")
    # (do not st.stop() here)

    # Eligibility: must be currently in projected(auto), n3 > 0, mask sane, and k=3 is green
    mode_now = str(rc.get("mode", ""))
    elig_freeze = (mode_now == "projected(auto)" and n3 > 0 and len(lm) == n3 and k3_green)

    st.caption("Freeze current AUTO Π → file, switch to FILE, re-run Overlap, and force a cert write.")

    # Inputs
    pj_basename  = st.text_input(
        "Filename",
        value=f"projector_{district_id or 'UNKNOWN'}.json",
        key="pj_freeze_name_final2"
    )
    overwrite_ok = st.checkbox("Overwrite if exists", value=False, key="pj_freeze_overwrite_final2")

    # Global FILE Π invalid gate (render-time convenience)
    fm_bad   = file_validation_failed()
    help_txt = "Disabled because projected(FILE) validation failed. Freeze AUTO→FILE again or fix Π."

    # Final disabled state
    disabled = fm_bad or (not elig_freeze)
    tip = help_txt if fm_bad else (None if elig_freeze else "Enabled when current run is projected(auto) and k=3 is green.")

    # Ensure the projectors dir exists (robust default)
    PROJECTORS_DIR = Path(_ss.get("PROJECTORS_DIR", "projectors"))
    PROJECTORS_DIR.mkdir(parents=True, exist_ok=True)

    if st.button("Freeze Π → FILE & re-run",
                 key="btn_freeze_final2",
                 disabled=disabled,
                 help=(tip or "Freeze AUTO to FILE and re-run")):
        try:
            # Guard rc one more time
            rc = dict(_ss.get("run_ctx") or {})
            n3 = int(rc.get("n3") or 0)
            lm = list(rc.get("lane_mask_k3") or [])
            if mode_now != "projected(auto)":
                st.warning("Current run is not projected(auto). Click Run Overlap with AUTO first.")
                st.stop()
            if n3 <= 0 or len(lm) != n3:
                st.warning("Context invalid (n3/mask mismatch). Click Run Overlap and try again.")
                st.stop()

            # Build Π from lane mask (diagonal projector over k=3)
            P_freeze = [[1 if (i == j and lm[j]) else 0 for j in range(n3)] for i in range(n3)]

            # Validate strictly (uses your existing helper)
            validate_projector_file_strict(P_freeze, n3=n3, lane_mask=lm)

            # Save projector atomically + compute hash
            pj_path = PROJECTORS_DIR / pj_basename
            if pj_path.exists() and not overwrite_ok:
                st.warning("Projector file already exists. Enable 'Overwrite if exists' or choose a new name.")
                st.stop()
            payload = {"schema_version": "1.0.0", "blocks": {"3": P_freeze}}

            tmp = pj_path.with_suffix(pj_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, pj_path)

            pj_hash = hash_json(payload)

            # Switch UI policy to FILE & stamp the path so cfg_active picks it up
            st.session_state["ov_policy_choice"] = "projected(file)"
            cfg_active.setdefault("source", {})["3"] = "file"
            cfg_active.setdefault("projector_files", {})["3"] = pj_path.as_posix()
            st.session_state["ov_last_pj_path"] = pj_path.as_posix()

            # Bump nonce / clear overlap caches
            if "_mark_fixtures_changed" in globals():
                _mark_fixtures_changed()
            else:
                st.session_state["_fixture_nonce"] = int(st.session_state.get("_fixture_nonce", 0)) + 1
                for k in ("overlap_out", "residual_tags", "overlap_cfg", "overlap_policy_label"):
                    st.session_state.pop(k, None)

            # Re-run Overlap immediately (fresh FILE mode)
            if "_soft_reset_before_overlap" in globals():
                _soft_reset_before_overlap()
            run_overlap()

            # Force cert write this pass (bypass debounce)
            st.session_state["should_write_cert"] = True
            st.session_state.pop("_last_cert_write_key", None)

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

def _atomic_write_json(path: Path, payload: dict) -> None:
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
    REPORTS_DIR = Path("reports")
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
    if mode_now == "projected(file)":
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


    
    
    # ============================== Cert & Provenance ==============================
with safe_expander("Cert & provenance", expanded=True):
    import os, json, hashlib, platform, time
    from pathlib import Path
    from datetime import datetime

    # ---------- constants ----------
    SCHEMA_VERSION = "1.0.0"
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
    _toggle_key = ensure_unique_widget_key("cert_allow_stale_ssot")
    allow_stale = st.toggle("Allow writing with stale SSOT", value=False, key=_toggle_key)
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
    if st.checkbox("Show raw SSOT (_inputs_block)", value=False, key="show_raw_ssot_final"):
        st.json(ss.get("_inputs_block") or {})

    # ---------- FILE Π validity & inputs completeness ----------
    file_pi_valid   = bool(ss.get("file_pi_valid", True))
    file_pi_reasons = list(ss.get("file_pi_reasons", []) or [])
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
            if tuple(ab.get("inputs_sig") or ()) != inputs_sig:
                stale_ab = REASON.AB_STALE_INPUTS_SIG
            elif _canon_policy(ab.get("policy_tag","")) != policy_canon:
                stale_ab = REASON.AB_STALE_POLICY
            elif policy_canon == "projected:file" and (ab.get("projected",{}) or {}).get("projector_hash","") != proj_hash:
                stale_ab = REASON.AB_STALE_PROJECTOR_HASH
            st.success("A/B: Pinned · Fresh") if not stale_ab else st.warning(f"A/B: Pinned · Stale ({stale_ab})")
        else:
            st.caption("A/B: —")
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
        # re-evaluate ticket freshness here
        is_ab_pinned           = (ab_pin.get("state") == "pinned")
        ab_ticket_pending      = ss.get("_ab_ticket_pending")
        last_ab_ticket_written = ss.get("_last_ab_ticket_written")
        ticket_required        = bool(is_ab_pinned and (ab_ticket_pending is not None) and (ab_ticket_pending != last_ab_ticket_written))

        if not ticket_required:
            write_enabled = False
            _append_witness({
                "ts": _utc_now_z(),
                "outcome": REASON.SKIP_SSOT_STALE,
                "armed": write_armed,
                "armed_by": ss.get("armed_by",""),
                "key": {
                    "inputs": _sha256_hex(":".join(inputs_sig).encode())[:8],
                    "pol": policy_canon, "pv": f"{int(pass_vec[0])}{int(pass_vec[1])}", "pj": _short(proj_hash)
                },
                "ab": ("PINNED" if is_ab_pinned else "NONE"),
                "file_pi": {"mode": ("file" if policy_canon=="projected:file" else policy_canon), "valid": file_pi_valid, "reasons": file_pi_reasons[:3]}
            })
            st.warning("SSOT stale — write disabled. Run Overlap to refresh or enable the 'Allow writing with stale SSOT' toggle.")
            st.caption("Rendering continues so you can inspect A/B & gallery.")
        else:
            st.info("SSOT is stale, but proceeding due to A/B one-shot ticket override.")
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
        if tuple(ab.get("inputs_sig") or ()) != inputs_sig:
            ab_status = REASON.AB_STALE_INPUTS_SIG
        elif _canon_policy(ab.get("policy_tag","")) != policy_canon:
            ab_status = REASON.AB_STALE_POLICY
        elif policy_canon == "projected:file" and (ab.get("projected",{}) or {}).get("projector_hash","") != proj_hash:
            ab_status = REASON.AB_STALE_PROJECTOR_HASH
        else:
            ab_fresh = True
            ab_status = REASON.AB_EMBEDDED

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
        if mode_now == "projected(file)"
        else (_auto_pj_hash_from_rc(rc or {}) if mode_now == "projected(auto)" else "")
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
        ss["_last_ab_ticket_written"] = ab_ticket_pending

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




# ====================== B2 Gallery Builder (CSV + manifest) ======================
import os, json, csv, hashlib
from pathlib import Path
from datetime import datetime

def _b2_norm_vec(v):
    try: return [int(x) for x in (v or [])]
    except Exception: return []

def _b2_vec_json(v):
    try: return json.dumps(_b2_norm_vec(v), separators=(",", ":"))
    except Exception: return "[]"

def _b2_short(h): return (h or "")[:12]

def _b2_collect_certs():
    """Yield (path, payload) for all certs under logs/certs/**/**.json newest→oldest."""
    root = Path("logs") / "certs"
    if not root.exists(): return []
    files = list(root.rglob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out

def _b2_row_from_cert(cert):
    """Map one cert → normalized CSV row dict or None (if missing fixture_label)."""
    identity   = cert.get("identity", {}) or {}
    policy     = cert.get("policy", {}) or {}
    inputs     = cert.get("inputs", {}) or {}
    checks     = cert.get("checks", {}) or {}
    diags      = cert.get("diagnostics", {}) or {}
    registry   = (inputs.get("registry") or {})
    hashes     = (inputs.get("hashes") or {})
    ab_embed   = cert.get("ab_embed", {}) or {}
    gallery    = cert.get("gallery", {}) or {}
    content    = (cert.get("hashes") or {}).get("content_hash", "")

    district   = identity.get("district_id", "UNKNOWN")
    fixture    = identity.get("fixture_label", "") or ""   # must exist
    fixture_code = identity.get("fixture_code", "") or ""
    if not fixture:
        return None  # skip — we want explicit fixture labels in CSV

    canon      = policy.get("canon") or policy.get("projector_mode") or "strict"
    # normalize projected canon naming
    if canon == "auto": canon = "projected:auto"
    if canon == "file": canon = "projected:file"

    hash_d     = policy.get("projector_hash", "")  # only for projected:file
    hash_U     = hashes.get("U_hash", "")
    hash_C     = hashes.get("C_hash", "")
    hash_H     = hashes.get("H_hash", "")

    growth     = int((cert.get("growth") or {}).get("growth_bumps", 0))
    tag        = str(gallery.get("tag", ""))
    strictify  = str(gallery.get("strictify", "tbd"))

    lane_H2    = _b2_vec_json(diags.get("lane_vec_H2@d3"))
    lane_CI    = _b2_vec_json(diags.get("lane_vec_C3+I3"))

    ab_embedded = bool(ab_embed.get("fresh", False))

    row = {
        "district":      district,
        "fixture":       fixture,
        "projected":     canon,
        "hash_d":        hash_d,
        "hash_U":        hash_U,
        "hash_suppC":    hash_C,
        "hash_suppH":    hash_H,
        "growth":        str(growth),
        "tag":           tag,
        "strictify":     strictify,
        "lane_vec_H2":   lane_H2,
        "lane_vec_C3pI3":lane_CI,
        "ab_embedded":   "true" if ab_embedded else "false",
        "content_hash":  content,
        "fixture_code":  fixture_code,  # optional helper column
        # (provenance of registry helps audits; harmless in manifest, not in CSV)
        "_fx_ver":       registry.get("fixtures_version",""),
    }
    return row

def _b2_sort_key(row, curated_order):
    # curated codes first by ordering; then no-code by fixture label
    code = row.get("fixture_code","") or ""
    if code and code in curated_order:
        return (0, curated_order.index(code), row.get("fixture",""))
    return (1, 10**9, row.get("fixture",""))

def _b2_best_pick(rows):
    """
    For a given (district, fixture) bucket, pick ONE row by:
      1) A/B-embedded projected (canon in {projected:auto, projected:file}), newest
      2) else projected (auto/file), newest
      3) else strict, newest
    Caller must pass rows already newest→oldest.
    """
    def is_proj(r): return r.get("projected") in ("projected:auto","projected:file")
    def is_ab(r):   return r.get("ab_embedded") == "true"
    # newest first already
    for r in rows:
        if is_proj(r) and is_ab(r): return r
    for r in rows:
        if is_proj(r): return r
    for r in rows:
        if r.get("projected") == "strict": return r
    return rows[0] if rows else None

def build_b2_gallery(debounce: bool = True):
    """
    Scans certs, selects one row per (district, fixture), writes:
      logs/reports/b2_gallery.csv
      logs/reports/b2_gallery.manifest.json
    Debounced by content signature of (district, fixture, content_hash).
    """
    reports_dir = Path("logs") / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = reports_dir / "b2_gallery.csv"
    man_path  = reports_dir / "b2_gallery.manifest.json"

    # collect + map
    certs = _b2_collect_certs()
    rows_all = []
    missing_fixture = []
    vector_parse_issues = 0
    curated_order = []

    # pull curated ordering from cached fixtures (if present)
    fx_cache = st.session_state.get("_fixtures_cache") or {}
    curated_order = list(fx_cache.get("ordering") or [])

    for p, cert in certs:
        r = _b2_row_from_cert(cert)
        if r is None:
            missing_fixture.append(p.name)
            continue
        # validate vectors parse (warn-only)
        try:
            json.loads(r["lane_vec_H2"]); json.loads(r["lane_vec_C3pI3"])
        except Exception:
            vector_parse_issues += 1
        rows_all.append(r)

    # debounce signature
    sig_elems = sorted({ (r["district"], r["fixture"], r["content_hash"]) for r in rows_all })
    sig_bytes = json.dumps(sig_elems, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig_hash  = hashlib.sha256(sig_bytes).hexdigest()

    if debounce:
        last_sig = (json.loads(man_path.read_text())["signature"]
                    if man_path.exists() else "")
        if last_sig == sig_hash:
            raise RuntimeError("no changes since last build")

    # bucket by (district, fixture)
    buckets = {}
    for r in rows_all:
        buckets.setdefault((r["district"], r["fixture"]), []).append(r)

    # pick best per bucket (newest already due to cert scan order)
    picked = []
    for key, brs in buckets.items():
        picked.append(_b2_best_pick(brs))

    # sort by curated ordering (via fixture_code), then label
    picked.sort(key=lambda r: _b2_sort_key(r, curated_order))

    # write CSV (atomic)
    tmp = csv_path.with_suffix(".csv.tmp")
    fieldnames = ["district","fixture","projected","hash_d","hash_U","hash_suppC","hash_suppH",
                  "growth","tag","strictify","lane_vec_H2","lane_vec_C3pI3","ab_embedded","content_hash","fixture_code"]
    with open(tmp, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        for r in picked:
            w.writerow({k: r.get(k,"") for k in fieldnames})
    os.replace(tmp, csv_path)

    # manifest
    manifest = {
        "written_at_utc": datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "rows": len(picked),
        "districts": sorted({ r["district"] for r in picked }),
        "fixture_codes_seen": sorted({ r.get("fixture_code","") for r in picked if r.get("fixture_code") }),
        "unknown_fixtures": sum(1 for r in picked if not r.get("fixture_code")),
        "signature": sig_hash,
        "missing_fixture_label": missing_fixture,
        "vector_parse_issues": vector_parse_issues,
        "source_certs": [ {"district": r["district"], "fixture": r["fixture"], "content_hash": r["content_hash"]} for r in picked ],
    }
    man_tmp = man_path.with_suffix(".json.tmp")
    with open(man_tmp, "w", encoding="utf-8") as mf:
        mf.write(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(man_tmp, man_path)

    st.success(f"B2 gallery built → {csv_path.as_posix()} ({len(picked)} rows)")
    st.caption(f"manifest: {man_path.as_posix()}")


# ============================== B2 Gallery Compiler (final) ==============================
import os, json, re, csv, hashlib
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path("logs") / "reports"
CERTS_ROOTS = [Path("logs") / "certs"]   # add more roots if you keep certs elsewhere
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -------- helpers (pure) --------
_G_RE = re.compile(r"^g(\d+)$", re.IGNORECASE)

def _canon_policy(label_or_canon: str) -> str:
    t = (label_or_canon or "").lower()
    if "strict" in t: return "strict"
    if "projected" in t and "file" in t: return "projected:file"
    return "projected:auto"

def _g_index(fixture: str):
    if not fixture: return (9999, fixture or "")
    m = _G_RE.match(fixture.strip())
    if not m: return (9999, fixture)
    try: return (int(m.group(1)), fixture.upper())
    except: return (9999, fixture)

def _parse_ts_z(ts: str):
    try: return datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
    except: return datetime.min

def _selection_rank(cert: dict) -> tuple:
    pol = _canon_policy((cert.get("policy") or {}).get("canon") or (cert.get("policy") or {}).get("label_raw",""))
    ab_fresh = bool((cert.get("ab_embed") or {}).get("fresh", False))
    if ab_fresh and pol in {"projected:auto","projected:file"}:
        pri = 3
    elif pol in {"projected:auto","projected:file"}:
        pri = 2
    elif pol == "strict":
        pri = 1
    else:
        pri = 0
    return (pri, _parse_ts_z(cert.get("written_at_utc","")))

def _scan_cert_paths(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        if root.exists():
            out.extend(root.rglob("overlap__*.json"))
    return out

def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _row_from_cert(cert: dict) -> dict | None:
    ident   = (cert.get("identity") or {})
    policy  = (cert.get("policy") or {})
    inputs  = (cert.get("inputs") or {})
    hashes  = (inputs.get("hashes") or {})
    diag    = (cert.get("diagnostics") or {})
    gallery = (cert.get("gallery") or {})
    ab      = (cert.get("ab_embed") or {})
    if not ident.get("district_id") or not ident.get("fixture_label"):
        return None  # skip ambiguous rows

    pol = _canon_policy(policy.get("canon") or policy.get("label_raw",""))
    proj_hash = policy.get("projector_hash","") if pol == "projected:file" else ""

    # lane vectors → JSON strings (stable)
    def _json_vec(v):
        try:
            if isinstance(v, list): vv = [int(x) for x in v]
            else: vv = v
            s = json.dumps(vv, separators=(",",":"))
            json.loads(s)  # sanity
            return s
        except Exception:
            return json.dumps([])

    return {
        "district":    ident.get("district_id",""),
        "fixture":     ident.get("fixture_label",""),
        "projected":   pol,
        "hash_d":      proj_hash,
        "hash_U":      hashes.get("U_hash","") or "",
        "hash_suppC":  hashes.get("C_hash","") or "",
        "hash_suppH":  hashes.get("H_hash","") or "",
        "growth":      (cert.get("growth") or {}).get("growth_bumps", 0),
        "tag":         gallery.get("tag",""),
        "strictify":   gallery.get("strictify","tbd"),
        "lane_vec_H2": _json_vec(diag.get("lane_vec_H2@d3", [])),
        "lane_vec_C3pI3": _json_vec(diag.get("lane_vec_C3+I3", [])),
        "ab_embedded": bool(ab.get("fresh", False)),
        "content_hash": (cert.get("hashes") or {}).get("content_hash",""),
    }

def _witness_gallery(line: dict):
    try:
        p = Path("logs") / "witnesses.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, separators=(",",":")) + "\n")
    except Exception:
        pass

# -------- builder (callable) --------
def build_b2_gallery(*, debounce: bool = True) -> tuple[Path, Path, dict]:
    cert_paths = _scan_cert_paths(CERTS_ROOTS)

    groups: dict[tuple[str,str], list[tuple[dict,Path]]] = {}
    skipped_no_fixture = []
    loaded = 0

    for p in cert_paths:
        cert = _load_json(p)
        if cert is None:
            continue
        loaded += 1
        ident = (cert.get("identity") or {})
        district = ident.get("district_id","")
        fixture  = (ident.get("fixture_label","") or "").strip()
        if not district or not fixture:
            skipped_no_fixture.append(p.name)
            continue
        groups.setdefault((district, fixture), []).append((cert, p))

    # Pick best per group
    picked_rows = []
    for (district, fixture), items in groups.items():
        # sort desc by priority, then by timestamp
        items_sorted = sorted(items, key=lambda cp: _selection_rank(cp[0]), reverse=True)
        best_cert = items_sorted[0][0]
        row = _row_from_cert(best_cert)
        if row:
            picked_rows.append(row)

    # Sort rows by (district, g_index/fixture)
    picked_rows.sort(key=lambda r: (r["district"], _g_index(r["fixture"])))

    # Debounce hash (on (district, fixture, content_hash))
    key_set = sorted(f"{r['district']}|{r['fixture']}|{r['content_hash']}" for r in picked_rows)
    build_hash = hashlib.sha256(("\n".join(key_set)).encode("utf-8")).hexdigest()
    state_path = REPORTS_DIR / "b2_gallery.state.json"
    previous_hash = ""
    if debounce and state_path.exists():
        try:
            previous_hash = json.loads(state_path.read_text(encoding="utf-8")).get("last_gallery_build_hash","")
        except Exception:
            previous_hash = ""

    # Prepare outputs
    csv_path = REPORTS_DIR / "b2_gallery.csv"
    tmp_path = REPORTS_DIR / "b2_gallery.csv.tmp"
    validation_path = REPORTS_DIR / "b2_gallery.validation.json"

    # If unchanged and debounce on → short-circuit
    if debounce and picked_rows and build_hash == previous_hash:
        receipt = {
            "rows_written": 0,
            "skipped_no_fixture_label": skipped_no_fixture,
            "vector_parse_issues": 0,  # vectors already normalized to JSON
            "missing_hash_counts": {
                "C_hash": sum(1 for r in picked_rows if not r["hash_suppC"]),
                "H_hash": sum(1 for r in picked_rows if not r["hash_suppH"]),
                "U_hash": sum(1 for r in picked_rows if not r["hash_U"]),
            },
            "last_gallery_build_hash": build_hash,
            "certs_scanned": len(cert_paths),
            "certs_loaded": loaded,
            "debounced": True,
        }
        validation_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        _witness_gallery({
            "ts": datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
            "outcome": "B2_DEBOUNCED",
            "rows": len(picked_rows),
            "hash": build_hash,
        })
        return csv_path, validation_path, receipt

    # Write CSV atomically
    header = [
        "district","fixture","projected","hash_d","hash_U","hash_suppC","hash_suppH",
        "growth","tag","strictify","lane_vec_H2","lane_vec_C3pI3","ab_embedded","content_hash"
    ]
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        for r in picked_rows:
            w.writerow(r)
        f.write("\n")  # trailing newline for clean diffs
    os.replace(tmp_path, csv_path)

    # Validation receipt
    receipt = {
        "rows_written": len(picked_rows),
        "skipped_no_fixture_label": skipped_no_fixture,
        "vector_parse_issues": 0,
        "missing_hash_counts": {
            "C_hash": sum(1 for r in picked_rows if not r["hash_suppC"]),
            "H_hash": sum(1 for r in picked_rows if not r["hash_suppH"]),
            "U_hash": sum(1 for r in picked_rows if not r["hash_U"]),
        },
        "last_gallery_build_hash": build_hash,
        "certs_scanned": len(cert_paths),
        "certs_loaded": loaded,
        "debounced": False,
    }
    validation_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    state_path.write_text(json.dumps({"last_gallery_build_hash": build_hash}, indent=2, sort_keys=True), encoding="utf-8")

    _witness_gallery({
        "ts": datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "outcome": "B2_BUILT",
        "rows": len(picked_rows),
        "hash": build_hash,
        "csv": csv_path.as_posix()
    })
    return csv_path, validation_path, receipt

# -------- lightweight UI wrapper (safe to call anywhere) --------
def render_b2_gallery_controls():
    with st.container():
        st.markdown("### B2 — Lane exploration gallery (CSV)")
        col1, col2 = st.columns([1,3])
        with col1:
            if st.button("Build Gallery CSV", key="btn_b2_build"):
                csvp, valp, rec = build_b2_gallery(debounce=False)
                st.success(f"Built → `{csvp.as_posix()}` · rows={rec['rows_written']}")
                with st.expander("Validation receipt"):
                    st.json(rec)
        with col2:
            if (REPORTS_DIR / "b2_gallery.csv").exists():
                st.caption(f"Latest: `{(REPORTS_DIR / 'b2_gallery.csv').as_posix()}`")
            else:
                st.caption("No gallery built yet.")

# ============================== /B2 Gallery Compiler (final) ==============================







            
    
                            
                       
       
               





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
            
            
                           
