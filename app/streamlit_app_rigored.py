# --- robust loader with real package context (supports app/otcore or app/core) ---
import sys, pathlib, importlib.util, types
import streamlit as st
import json
import json as _json
import hashlib
import os
from io import BytesIO
import zipfile


st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# ────────────────────────────── PACKAGE LOADER ────────────────────────────────
HERE   = pathlib.Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
CORE   = HERE / "core"
PKG_DIR = OTCORE if OTCORE.exists() else CORE
PKG_NAME = "otcore" if OTCORE.exists() else "core"

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
    mod.__package__ = fullname.rsplit('.', 1)[0]
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Fresh-load core modules
for _mod in (f"{PKG_NAME}.overlap_gate", f"{PKG_NAME}.projector"):
    if _mod in sys.modules:
        del sys.modules[_mod]

overlap_gate = _load_pkg_module(f"{PKG_NAME}.overlap_gate", "overlap_gate.py")
projector    = _load_pkg_module(f"{PKG_NAME}.projector",    "projector.py")
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import io as _io_pkg
# now `otio: _io_pkg`  # type: ignore
otio          = _load_pkg_module(f"{PKG_NAME}.io",            "io.py")
hashes        = _load_pkg_module(f"{PKG_NAME}.hashes",        "hashes.py")
unit_gate     = _load_pkg_module(f"{PKG_NAME}.unit_gate",     "unit_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers        = _load_pkg_module(f"{PKG_NAME}.towers",        "towers.py")
export_mod    = _load_pkg_module(f"{PKG_NAME}.export",        "export.py")
# Hotfix alias so any lingering `io.parse_*` still work:
if "io" not in globals():
    io = otio  # keeps old references alive while you migrate to `otio.*`

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")

# ─────────────────────────────── UI HEADER ────────────────────────────────────
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")
st.caption(f"overlap_gate loaded from: {getattr(overlap_gate, '__file__', '<none>')}")
st.caption(f"projector loaded from: {getattr(projector, '__file__', '<none>')}")

# ─────────────────────────── MATH LAB FOUNDATION ─────────────────────────────
# Schema + paths + atomic IO + run IDs + residual snapshot + tiny UI widgets.
import os, json, csv, hashlib, sys, platform
import io as pyio  # stdlib io lives here
from io import BytesIO       # optional; safe to keep
from datetime import datetime, timezone
from pathlib import Path

# == Versions / schema tags (bump when you change artifact fields) ==
LAB_SCHEMA_VERSION = "1.0.0"
APP_VERSION_STR    = str(APP_VERSION) if "APP_VERSION" in globals() else "v0.1-core"
PY_VERSION_STR     = f"python-{platform.python_version()}"

# == Directories (kept clean; do not mix with inputs) ==
DIRS = {
    "inputs":      "inputs",
    "certs":       "certs",
    "bundles":     "bundles",
    "projectors":  "projectors",
    "logs":        "logs",
    "reports":     "reports",
    "fixtures":    "fixtures",
    "configs":     "configs",
}
for _d in DIRS.values():
    Path(_d).mkdir(parents=True, exist_ok=True)

# == Small helpers ==
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _short(h: str, n: int = 8) -> str:
    return (h or "")[:n]

# one-liner alias so all blocks agree on the name
def lane_mask_from_boundaries(boundaries):  # keep in module scope
    return _lane_mask_from_d3(boundaries)


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _sha256_json(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(s)

def _ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# == Atomic writers (no torn files) ==
def atomic_write_json(path: str | Path, obj: dict):
    tmp = Path(f"{path}.tmp")
    _ensure_parent(tmp)
    with open(tmp, "wb") as f:
        blob = json.dumps(obj, sort_keys=True, indent=2).encode()
        f.write(blob); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_append_jsonl(path: str | Path, row: dict):
    # write one full line to tmp, then append atomically
    tmp = Path(f"{path}.tmp")
    _ensure_parent(tmp)
    line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    with open(tmp, "wb") as f:
        f.write(line.encode()); f.flush(); os.fsync(f.fileno())
    # append
    with open(path, "ab") as out:
        with open(tmp, "rb") as src:
            data = src.read()
        out.write(data); out.flush(); os.fsync(out.fileno())
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

def atomic_write_csv(path: str | Path, header: list[str], rows: list[list]):
    tmp = Path(f"{path}.tmp")
    _ensure_parent(tmp)
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# schema_version={LAB_SCHEMA_VERSION}", f"written_at_utc={_iso_utc_now()}",
                    f"app_version={APP_VERSION_STR}", f"{PY_VERSION_STR}"])
        w.writerow(header)
        w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# == Inputs hashing & run_id (SSOT-aware) ==
def inputs_hashes_snapshot(inputs_block: dict) -> dict:
    """Return a **copy** of current authoritative hashes (do not compute here)."""
    keys = ["boundaries_hash", "C_hash", "H_hash", "U_hash", "shapes_hash", "projector_hash"]
    snap = {k: inputs_block.get(k, "") for k in keys}
    return snap

def default_seed_from_inputs(inputs_block: dict) -> str:
    snap = inputs_hashes_snapshot(inputs_block)
    key = (snap.get("boundaries_hash","") + snap.get("C_hash","") +
           snap.get("H_hash","") + snap.get("U_hash",""))
    return _sha256_bytes(key.encode())[:8]

def build_run_id(inputs_block: dict, run_ctx: dict | None) -> str:
    snap = inputs_hashes_snapshot(inputs_block)
    policy = (run_ctx or {}).get("policy_tag", "")
    pjhash = (run_ctx or {}).get("projector_hash", "")
    stamp  = f"{policy}|{snap}|{pjhash}|{_iso_utc_now()}"
    return _sha256_bytes(stamp.encode())

# == Residual snapshot (tiny matrices for “prove it”) ==
def write_residual_snapshot(bundle_dir: str | Path, *, 
                            R3_strict_lane: list[list[int]] | None,
                            R3_proj_lane:   list[list[int]] | None,
                            lane_mask_k3:   list[int],
                            shapes_obj      ):
    payload = {
        "schema_version": LAB_SCHEMA_VERSION,
        "written_at_utc": _iso_utc_now(),
        "app_version":    APP_VERSION_STR,
        "lane_mask_k3":   [int(x) for x in (lane_mask_k3 or [])],
        "R3_strict_lane": R3_strict_lane or [],
        "R3_proj_lane":   R3_proj_lane or [],
        "shapes":         (shapes_obj.dict() if hasattr(shapes_obj, "dict") else (shapes_obj or {})),
    }
    out = Path(bundle_dir) / "residual.json"
    atomic_write_json(out, payload)
    return str(out)

# == A/B freshness ==
def ab_is_fresh(ab_ctx: dict | None, inputs_block: dict) -> bool:
    if not ab_ctx: 
        return False
    curr_sig = [
        inputs_block.get("boundaries_hash",""),
        inputs_block.get("C_hash",""),
        inputs_block.get("H_hash",""),
        inputs_block.get("U_hash",""),
        inputs_block.get("shapes_hash",""),
    ]
    return ab_ctx.get("inputs_sig") == curr_sig

# == Cache flush (single place that really clears it all) ==
def flush_run_cache():
    for k in ["run_ctx", "overlap_out", "residual_tags",
              "ab_compare", "_projector_cache", "_projector_cache_ab",
              "_projector_cache_key", "_projector_cache_key_ab",
              "_last_composite_key"]:
        st.session_state.pop(k, None)

def composite_cache_key(*, policy_tag: str, inputs_block: dict, projector_filename: str | None, projector_hash: str | None) -> str:
    parts = [
        f"policy={policy_tag}",
        f"B={inputs_block.get('boundaries_hash','')}",
        f"C={inputs_block.get('C_hash','')}",
        f"H={inputs_block.get('H_hash','')}",
        f"U={inputs_block.get('U_hash','')}",
        f"PJFN={(projector_filename or '')}",
        f"PJH={(projector_hash or '')}",
    ]
    return "|".join(parts)

# == Tiny UI widgets (optional, safe to call anywhere below) ==
def render_active_policy_pill(run_ctx: dict | None, inputs_block: dict | None):
    if not run_ctx:
        st.caption("Policy: *(none — run overlap)*")
        return
    mode = run_ctx.get("mode", "strict")
    pjhash = run_ctx.get("projector_hash") or ""
    pill = f"**Policy:** `{mode}`"
    if mode == "projected(file)" and pjhash:
        pill += f" · Π {_short(pjhash)}"
    st.caption(pill)

def render_run_stamp(run_ctx: dict | None, inputs_block: dict | None):
    if not (run_ctx and inputs_block):
        return
    n3 = len(run_ctx.get("lane_mask_k3") or [])
    pj = _short(run_ctx.get("projector_hash",""))
    b  = _short(inputs_block.get("boundaries_hash",""))
    c  = _short(inputs_block.get("C_hash",""))
    h  = _short(inputs_block.get("H_hash",""))
    u  = _short(inputs_block.get("U_hash",""))
    st.caption(f"run: policy={run_ctx.get('policy_tag','?')} · n3={n3} · Π={pj} · d={b} · C={c} · H={h} · U={u}")

# Sidebar: one-click cache flush (with visible composite key proof after run)
with st.sidebar:
    st.divider()
    st.markdown("#### Run cache")
    if st.button("Flush run cache", use_container_width=True):
        flush_run_cache()
        st.success("Run cache cleared.")
    # If we already have a run_ctx, show current composite key short (proof)
    _rc  = st.session_state.get("run_ctx")
    _ib  = st.session_state.get("_inputs_block", {})
    if _rc and _ib:
        key = composite_cache_key(
            policy_tag=_rc.get("policy_tag",""),
            inputs_block=_ib,
            projector_filename=_rc.get("projector_filename",""),
            projector_hash=_rc.get("projector_hash",""),
        )
        st.caption(f"composite key: `{_short(_sha256_bytes(key.encode()), 12)}`")

st.caption("Math Lab utils loaded · schema 1.0.0")
# ─────────────────────── END MATH LAB FOUNDATION ─────────────────────────────
# ───────────────────────── GF(2) ops shim (module-level) ─────────────────────────
# Provides mul, add, eye exactly as Tab 2 expects; falls back to pure-py if lib missing.
try:
    from otcore.linalg_gf2 import mul as _mul_lib, add as _add_lib, eye as _eye_lib
    mul = _mul_lib; add = _add_lib; eye = _eye_lib
except Exception:
    def mul(A, B):
        if not A or not B or not A[0] or not B[0]: return []
        m, kA = len(A), len(A[0]); kB, n = len(B), len(B[0])
        if kA != kB: return []
        C = [[0]*n for _ in range(m)]
        for i in range(m):
            Ai = A[i]
            for k in range(kA):
                if Ai[k] & 1:
                    Bk = B[k]
                    for j in range(n): C[i][j] ^= (Bk[j] & 1)
        return C
    def add(A, B):
        if not A: return B or []
        if not B: return A or []
        r, c = len(A), len(A[0])
        if len(B) != r or len(B[0]) != c: return A
        return [[(A[i][j] ^ B[i][j]) for j in range(c)] for i in range(r)]
    def eye(n): return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

# ───────────────────── PROJECTOR REGISTRY & FREEZER (Block 2) ─────────────────
# Build Π_auto from lane mask, freeze to disk, validate vs boundaries, and log.
import os, json as _json
from pathlib import Path

PROJECTOR_REG_PATH = Path(DIRS["projectors"]) / "projector_registry.jsonl"

# == local GF(2) helpers (namespaced to avoid collisions) ==
def _pj_diag_from_mask(mask: list[int]) -> list[list[int]]:
    n = len(mask or [])
    return [[1 if i == j and int(mask[i]) == 1 else 0 for j in range(n)] for i in range(n)]

def _pj_diag_vec(P: list[list[int]]) -> list[int]:
    if not P or not isinstance(P, list) or not P[0]: return []
    n = min(len(P), len(P[0]))
    return [int(P[i][i] & 1) for i in range(n)]

def _pj_is_rect(P) -> tuple[bool, int, int]:
    if not P or not isinstance(P, list): return (False, 0, 0)
    rows = len(P); cols = len(P[0]) if rows and isinstance(P[0], list) else 0
    if rows == 0 or cols == 0: return (False, rows, cols)
    if any((not isinstance(r, list) or len(r) != cols) for r in P): return (False, rows, cols)
    return (True, rows, cols)

def _pj_mul_gf2(A, B):
    if not A or not B: return []
    r, k, c = len(A), len(A[0]), len(B[0])
    if k != len(B): return []
    out = [[0]*c for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for t in range(k):
            if Ai[t] & 1:
                Bt = B[t]
                for j in range(c): out[i][j] ^= (Bt[j] & 1)
    return out

def _pj_idempotent(P) -> bool:
    ok, n, m = _pj_is_rect(P)
    if not ok or n != m: return False
    PP = _pj_mul_gf2(P, P)
    if not PP: return False
    for i in range(n):
        for j in range(n):
            if (PP[i][j] & 1) != (P[i][j] & 1): return False
    return True

def _pj_is_diagonal(P) -> bool:
    ok, n, m = _pj_is_rect(P)
    if not ok or n != m: return False
    for i in range(n):
        for j in range(n):
            if i != j and (P[i][j] & 1): return False
    return True

# == lane mask from current boundaries (no recompute elsewhere) ==
def lane_mask_from_boundaries(boundaries) -> list[int]:
    try:
        d3 = boundaries.blocks.__root__.get("3")
    except Exception:
        d3 = None
    if not d3 or not d3[0]: return []
    cols = len(d3[0])
    return [1 if any(row[j] & 1 for row in d3) else 0 for j in range(cols)]

# == stable projector hash (content-based; normalized) ==
def projector_content_hash(P3: list[list[int]]) -> str:
    # cast to 0/1 and ensure rectangular before hashing
    ok, r, c = _pj_is_rect(P3)
    if ok:
        norm = [[int(v) & 1 for v in row] for row in P3]
    else:
        norm = []
    return _sha256_json({"P3": norm})

# == freeze AUTO Π to file (diag(lane_mask)); return (path, hash, payload) ==
def freeze_auto_projector(*, district_id: str, lane_mask_k3: list[int], out_path: str | None = None):
    if out_path is None or not str(out_path).strip():
        out_path = str(Path(DIRS["projectors"]) / f"projector_{district_id or 'UNKNOWN'}_k3.json")
    P = _pj_diag_from_mask([int(x) for x in (lane_mask_k3 or [])])
    payload = {
        "schema_version": LAB_SCHEMA_VERSION,
        "name": f"Π3 freeze (lane-mask of current d3) · {district_id or 'UNKNOWN'}",
        "written_at_utc": _iso_utc_now(),
        "blocks": {"3": P},
    }
    atomic_write_json(out_path, payload)
    pj_hash = projector_content_hash(P)
    return out_path, pj_hash, payload

# == validate FILE projector vs current boundaries (order + messages exact) ==
def validate_projector_file_for_boundaries(pj_path: str, boundaries) -> tuple[bool, str]:
    if not pj_path or not os.path.exists(pj_path):
        return False, f"Projector(k=3) file not found: {pj_path!r}"
    try:
        with open(pj_path, "r") as f:
            J = _json.load(f)
    except Exception as e:
        return False, f"Projector(k=3) unreadable JSON: {e}"

    # A) presence
    P3 = None
    if isinstance(J, dict) and isinstance(J.get("blocks"), dict) and "3" in J["blocks"]:
        P3 = J["blocks"]["3"]
    elif isinstance(J, list):
        P3 = J
    if P3 is None:
        return False, 'Projector(k=3) missing "blocks"."3".'

    # dims
    ok, rows, cols = _pj_is_rect(P3)
    try:
        d3 = boundaries.blocks.__root__.get("3") or []
        n3 = len(d3[0]) if (d3 and d3[0]) else 0
    except Exception:
        n3 = 0

    # B) shape
    if not ok or rows != cols or cols != n3:
        return False, f"Projector(k=3) shape mismatch: expected {n3}x{n3}, got {rows}x{cols}."

    # C) idempotence
    if not _pj_idempotent(P3):
        return False, "Projector(k=3) not idempotent over GF(2): P·P != P."

    # D) diagonal
    if not _pj_is_diagonal(P3):
        return False, "Projector(k=3) must be diagonal; off-diagonal entries found."

    # E) lane-diag consistency
    lane = lane_mask_from_boundaries(boundaries)
    diagP = _pj_diag_vec(P3)
    if diagP != lane:
        return False, f"Projector(k=3) diagonal {diagP} inconsistent with lane_mask(d3) {lane}."

    return True, ""

# == registry append (JSONL) ==
def append_projector_registry_row(*, district_id: str, lane_mask_k3: list[int],
                                  filename: str, projector_hash: str):
    row = {
        "schema_version": LAB_SCHEMA_VERSION,
        "written_at_utc": _iso_utc_now(),
        "district": district_id or "UNKNOWN",
        "lane_mask_k3": [int(x) for x in (lane_mask_k3 or [])],
        "filename": filename,
        "projector_hash": projector_hash,
        "app_version": APP_VERSION_STR,
    }
    atomic_append_jsonl(PROJECTOR_REG_PATH, row)
    return row

# == tiny UI renderer (optional) ==
def render_projector_registry_tail(limit: int = 5):
    if not PROJECTOR_REG_PATH.exists():
        st.caption("Projector registry: (empty)")
        return
    try:
        with open(PROJECTOR_REG_PATH, "r") as f:
            lines = f.readlines()[-limit:]
        st.markdown("**Recent projectors**")
        for ln in reversed(lines):
            try:
                r = _json.loads(ln)
            except Exception:
                continue
            d = r.get("district","?")
            lm = "".join(str(int(x)) for x in r.get("lane_mask_k3", []))
            pj = _short(r.get("projector_hash",""), 12)
            fn = os.path.basename(r.get("filename",""))
            st.caption(f"{d} · Π={pj} · lanes={lm} · {fn}")
    except Exception as e:
        st.warning(f"Could not read registry: {e}")

# == one-shot convenience: freeze AUTO → file, validate immediately, log ==
def freeze_validate_log_projector(*, district_id: str, boundaries, lane_mask_k3: list[int],
                                  out_path: str | None = None) -> dict:
    pj_path, pj_hash, payload = freeze_auto_projector(
        district_id=district_id, lane_mask_k3=lane_mask_k3, out_path=out_path
    )
    ok, msg = validate_projector_file_for_boundaries(pj_path, boundaries)
    row = append_projector_registry_row(
        district_id=district_id, lane_mask_k3=lane_mask_k3,
        filename=pj_path, projector_hash=pj_hash
    )
    return {
        "ok": ok,
        "message": msg,
        "projector_path": pj_path,
        "projector_hash": pj_hash,
        "registry_row": row,
        "payload": payload,
    }

st.caption("Projector freezer/registry helpers ready")
# ─────────────────── END PROJECTOR REGISTRY & FREEZER (Block 2) ───────────────



# ────────────────────────────── SMALL HELPERS ────────────────────────────────
def read_json_file(file):
    if not file: return None
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return None

def _stamp_filename(state_key: str, f):
    """Remember uploaded filename for later provenance."""
    if f is not None:
        st.session_state[state_key] = getattr(f, "name", "")
    else:
        st.session_state.pop(state_key, None)

def _sha256_hex_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _sha256_hex_obj(obj) -> str:
    s = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
    return _sha256_hex_bytes(s)

def _lane_mask_from_d3(boundaries) -> list[int]:
    """Derive lane_mask_k3 from boundaries.d3: column j is lane if any bit in that column is 1."""
    try:
        d3 = boundaries.blocks.__root__.get("3")
    except Exception:
        d3 = None
    if not d3 or not d3[0]:
        return []
    return [1 if any((row[j] & 1) for row in d3) else 0 for j in range(len(d3[0]))]

def _district_signature(mask, r, c) -> str:
    payload = f"k3:{''.join(str(int(x)) for x in (mask or []))}|r{r}|c{c}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]

# Policy helpers (used by Tab 2)
def cfg_strict():
    return {"enabled_layers": [], "modes": {}, "source": {}, "projector_files": {}}

def cfg_projected_base():
    return {
        "enabled_layers": [3],
        "modes": {"3": "columns"},
        "source": {"3": "auto"},
        "projector_files": {"3": "projector_D3.json"},
    }

def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    parts = []
    for kk in sorted(cfg["enabled_layers"]):
        k = str(kk)
        mode = cfg.get("modes", {}).get(k, "none")
        src  = cfg.get("source", {}).get(k, "auto")
        parts.append(f"{mode}@k={kk},{src}")
    return "projected(" + "; ".join(parts) + ")"

# ─────────────────────── PROJECTOR CORE HELPERS (for Tab 2) ───────────────────
import hashlib as _hashlib

def hash_normalized(obj) -> str:
    """Stable SHA-256 over canonical, minimal JSON (deterministic)."""
    s = _json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hashlib.sha256(s).hexdigest()

def _rectangular(P: list[list[int]]) -> bool:
    if not P or not isinstance(P, list) or not isinstance(P[0], list):
        return False
    w = len(P[0])
    return all(isinstance(r, list) and len(r) == w for r in P)

def _to01(x) -> int:
    try:
        return int(x) & 1
    except Exception:
        return 0

def _normalize_matrix_01(P):
    """Cast entries to {0,1} and ensure modulo-2."""
    if not P:
        return []
    return [[ _to01(v) for v in row ] for row in P]

def _gf2_mm(A, B):
    """GF(2) matrix multiply with shape guard (returns [] on mismatch)."""
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

def _gf2_idempotent(P) -> bool:
    if not P or not P[0]:
        return False
    n = len(P)
    if any(len(row) != n for row in P):
        return False
    PP = _gf2_mm(P, P)
    if not PP or len(PP) != n or len(PP[0]) != n:
        return False
    for i in range(n):
        for j in range(n):
            if (PP[i][j] & 1) != (P[i][j] & 1):
                return False
    return True

def _is_diagonal(P) -> bool:
    if not P or not P[0]:
        return False
    n = len(P)
    if any(len(row) != n for row in P):
        return False
    for i in range(n):
        for j in range(n):
            if i != j and (P[i][j] & 1):
                return False
    return True

def _diag_vec(P):
    if not P or not P[0]:
        return []
    n = min(len(P), len(P[0]))
    return [int(P[i][i] & 1) for i in range(n)]

def _diag_from_mask(mask: list[int]) -> list[list[int]]:
    """AUTO projector: diag(mask)."""
    n = len(mask)
    return [[1 if (i == j and mask[i] == 1) else 0 for j in range(n)] for i in range(n)]

def _read_projector_block3_required(path: str) -> list[list[int]]:
    """
    Strict JSON loader for FILE mode:
    - Requires {"blocks":{"3":[...]}} (do not accept raw list-of-lists)
    - Enforces rectangularity and {0,1} normalization
    """
    with open(path, "r") as f:
        J = _json.load(f)
    if not isinstance(J, dict) or "blocks" not in J or "3" not in J["blocks"]:
        # EXACT message per spec:
        raise ValueError('Projector(k=3) missing "blocks"."3".')
    P = J["blocks"]["3"]
    Pn = _normalize_matrix_01(P)
    return Pn

def projector_choose_active(cfg_active: dict, boundaries):
    """
    Authoritative projector selector/validator.
    Returns (P_active, meta) OR raises ValueError with the specific failing message.
    meta schema:
      {
        "mode": "strict"|"projected(auto)"|"projected(file)",
        "projector_filename": str|"",
        "projector_hash": str|"",
        "projector_consistent_with_d": bool|None,
        "errors": []
      }
    """
    # d3 & mask
    try:
        d3 = boundaries.blocks.__root__.get("3") or []
    except Exception:
        d3 = []
    n3 = len(d3[0]) if (d3 and d3[0]) else 0
    lane_mask = _lane_mask_from_d3(boundaries)

    enabled = bool((cfg_active or {}).get("enabled_layers"))
    if not enabled:
        # strict: no projector
        return None, {
            "mode": "strict",
            "projector_filename": "",
            "projector_hash": "",
            "projector_consistent_with_d": None,
            "errors": [],
        }

    src3 = (cfg_active.get("source", {}) or {}).get("3", "auto")
    if src3 == "auto":
        P_auto = _diag_from_mask(lane_mask)
        return P_auto, {
            "mode": "projected(auto)",
            "projector_filename": "",
            "projector_hash": hash_normalized({"P3": P_auto}),
            "projector_consistent_with_d": True,
            "errors": [],
        }

    if src3 == "file":
        pj_path = (cfg_active.get("projector_files", {}) or {}).get("3")
        if not pj_path or not os.path.exists(pj_path):
            # match prior wording you used elsewhere:
            raise ValueError(f"Projector(k=3) file not found: {pj_path!r}")

        # ---- VALIDATION ORDER & MESSAGES (exact) ----------------------------
        # 1) presence of blocks."3"
        try:
            P = _read_projector_block3_required(pj_path)
        except ValueError as e:
            # already correct message, re-raise
            raise
        except Exception as e:
            # unreadable JSON etc.
            raise ValueError(f"Projector(k=3) unreadable JSON: {e}")

        # 2) shape: rows==cols==n3 and rectangular
        rows = len(P) if isinstance(P, list) else 0
        cols = len(P[0]) if (rows and isinstance(P[0], list)) else 0
        if not _rectangular(P) or rows != cols or cols != n3:
            raise ValueError(f"Projector(k=3) shape mismatch: expected {n3}x{n3}, got {rows}x{cols}.")

        # 3) idempotence over GF(2)
        if not _gf2_idempotent(P):
            raise ValueError("Projector(k=3) not idempotent over GF(2): P·P != P.")

        # 4) diagonal only
        if not _is_diagonal(P):
            raise ValueError("Projector(k=3) must be diagonal; off-diagonal entries found.")

        # 5) lane-diag consistency
        diagP = _diag_vec(P)
        if diagP != lane_mask:
            raise ValueError(f"Projector(k=3) diagonal {diagP} inconsistent with lane_mask(d3) {lane_mask}.")

        # success
        return P, {
            "mode": "projected(file)",
            "projector_filename": pj_path,
            "projector_hash": hash_normalized({"P3": P}),
            "projector_consistent_with_d": True,
            "errors": [],
        }

    # unknown source
    raise ValueError(f"Unknown projector source for k=3: {src3!r}")

# ───────────────────────── DISTRICT HASH → ID MAP ────────────────────────────
# sha256(boundaries.json RAW BYTES) -> district label
DISTRICT_MAP = {
    "9da8b7f605c113ee059160cdaf9f93fe77e181476c72e37eadb502e7e7ef9701": "D1",
    "4356e6b608443b315d7abc50872ed97a9e2c837ac8b85879394495e64ec71521": "D2",
    "28f8db2a822cb765e841a35c2850a745c667f4228e782d0cfdbcb710fd4fecb9": "D3",
    "aea6404ae680465c539dc4ba16e97fbd5cf95bae5ad1c067dc0f5d38ca1437b5": "D4",
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
    f_shapes  = st.file_uploader("Shapes (shapes.json)", type=["json"], key="shapes")
    f_bound   = st.file_uploader("Boundaries (boundaries.json)", type=["json"], key="bound")
    f_cmap    = st.file_uploader("CMap / Move (Cmap_*.json)", type=["json"], key="cmap")
    f_support = st.file_uploader("Support policy (support_ck_full.json)", type=["json"], key="support")
    f_pair    = st.file_uploader("Pairings (pairings.json)", type=["json"], key="pair")
    f_reps    = st.file_uploader("Reps (reps_for_Cmap_chain_pairing_ok.json)", type=["json"], key="reps")
    f_triangle= st.file_uploader("Triangle schema (triangle_J_schema.json)", type=["json"], key="tri")
    seed      = st.text_input("Seed", "super-seed-A")

    # filename stamps
    _stamp_filename("fname_shapes", f_shapes)
    _stamp_filename("fname_boundaries", f_bound)
    _stamp_filename("fname_cmap", f_cmap)

    # Show raw-bytes hash to help populate DISTRICT_MAP
    if f_bound is not None and hasattr(f_bound, "getvalue"):
        _raw  = f_bound.getvalue()
        _bhash = _sha256_hex_bytes(_raw)
        st.caption(f"boundaries raw-bytes hash: {_bhash}")
        st.code(f'DISTRICT_MAP["{_bhash}"] = "D?"  # ← set D1/D2/D3/D4', language="python")

# ───────────────────────────── LOAD CORE JSONS ───────────────────────────────
d_shapes = read_json_file(f_shapes)
d_bound  = read_json_file(f_bound)
d_cmap   = read_json_file(f_cmap)

# Shared inputs_block in session (used across tabs)
st.session_state.setdefault("_inputs_block", {})
inputs_block = st.session_state["_inputs_block"]

if d_shapes and d_bound and d_cmap:
    try:
        shapes     = io.parse_shapes(d_shapes)
        boundaries = io.parse_boundaries(d_bound)
        cmap       = io.parse_cmap(d_cmap)
        support    = io.parse_support(read_json_file(f_support))  if f_support  else None
        triangle   = io.parse_triangle_schema(read_json_file(f_triangle)) if f_triangle else None

        # Bind district from fresh boundaries (prefer raw bytes)
        try:
            if hasattr(f_bound, "getvalue"):
                _raw = f_bound.getvalue()
                boundaries_hash_fresh = _sha256_hex_bytes(_raw)
            else:
                boundaries_hash_fresh = _sha256_hex_obj(d_bound)
        except Exception:
            boundaries_hash_fresh = _sha256_hex_obj(d_bound)

        d3_block         = (boundaries.blocks.__root__.get("3") or [])
        lane_mask_k3_now = _lane_mask_from_d3(boundaries)
        d3_rows          = len(d3_block)
        d3_cols          = (len(d3_block[0]) if d3_block else 0)
        district_sig     = _district_signature(lane_mask_k3_now, d3_rows, d3_cols)
        district_id_fresh = DISTRICT_MAP.get(boundaries_hash_fresh, "UNKNOWN")

        # Clear stale session bits if boundaries changed
        _prev_bhash = st.session_state.get("_last_boundaries_hash")
        if _prev_bhash and _prev_bhash != boundaries_hash_fresh:
            st.session_state.pop("ab_compare", None)
            st.session_state.pop("district_id", None)
            st.session_state.pop("_projector_cache", None)
        st.session_state["_last_boundaries_hash"] = boundaries_hash_fresh

        # Stamp filenames + authoritative hashes (for cert/bundle)
        inputs_block["boundaries_filename"] = st.session_state.get("fname_boundaries", "boundaries.json")
        inputs_block["boundaries_hash"]     = boundaries_hash_fresh
        inputs_block["shapes_filename"]     = st.session_state.get("fname_shapes", "shapes.json")
        inputs_block["cmap_filename"]       = st.session_state.get("fname_cmap", "cmap.json")
        inputs_block.setdefault("U_filename", "shapes.json")

        # Mirror fresh district info for later blocks (used by Tab 2)
        st.session_state["_district_info"] = {
            "district_id":        district_id_fresh,
            "boundaries_hash":    boundaries_hash_fresh,
            "lane_mask_k3_now":   lane_mask_k3_now,
            "district_signature": district_sig,
            "d3_rows": d3_rows,
            "d3_cols": d3_cols,
        }

        # Validate
        io.validate_bundle(boundaries, shapes, cmap, support)
        st.success("Core schemas validated ✅")
        st.caption(
            f"district={district_id_fresh} · bhash={boundaries_hash_fresh[:12]} · "
            f"k3={lane_mask_k3_now} · sig={district_sig}"
        )

        with st.expander("Hashes / provenance"):
            named = [("boundaries", boundaries.dict()), ("shapes", shapes.dict()), ("cmap", cmap.dict())]
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
                import pathlib as _pl
                reports_dir = _pl.Path("reports")
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

# ────────────────────────────────── TABS ──────────────────────────────────────
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
                _rawB = f_B.getvalue()
                boundaries_hash_fresh = _sha256_hex_bytes(_rawB)
            else:
                boundaries_hash_fresh = _sha256_hex_obj(d_B)
        except Exception:
            boundaries_hash_fresh = _sha256_hex_obj(d_B)

        d3_block = (boundaries.blocks.__root__.get("3") or [])
        lane_mask_k3_now = _lane_mask_from_d3(boundaries)
        d3_rows = len(d3_block)
        d3_cols = (len(d3_block[0]) if d3_block else 0)
        district_sig = _district_signature(lane_mask_k3_now, d3_rows, d3_cols)
        district_id_fresh = DISTRICT_MAP.get(boundaries_hash_fresh, "UNKNOWN")

        _prev_bhash = st.session_state.get("_last_boundaries_hash")
        if _prev_bhash and _prev_bhash != boundaries_hash_fresh:
            st.session_state.pop("ab_compare", None)
            st.session_state.pop("district_id", None)
            st.session_state.pop("_projector_cache", None)
        st.session_state["_last_boundaries_hash"] = boundaries_hash_fresh

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
        out = unit_gate.unit_check(boundaries, cmap, shapes, reps=d_reps, enforce_rep_transport=enforce)
        st.json(out)
        # ───────────────────────── GF(2) ops shim for Tab 2 ──────────────────────────
# Provides mul, add, eye exactly as Tab 2 expects. If the library is present,
# we import; otherwise we use local pure-python fallbacks (bit-wise XOR math).

try:
    from otcore.linalg_gf2 import mul as _mul_lib, add as _add_lib, eye as _eye_lib
    mul = _mul_lib
    add = _add_lib
    eye = _eye_lib
except Exception:
    # local fallbacks — identical behavior for small matrices
    def mul(A, B):
        if not A or not B or not A[0] or not B[0]:
            return []
        m, kA = len(A), len(A[0])
        kB, n = len(B), len(B[0])
        if kA != kB:
            # keep same failure mode as your guard (callers can raise a friendly error)
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
            # mismatch → mirror upstream behavior (let caller decide)
            return A
        return [[(A[i][j] ^ B[i][j]) for j in range(c)] for i in range(r)]

    def eye(n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]




# ----------------------------- OVERLAP TAB -----------------------------------
with tab2:
    st.subheader("Overlap gate (homotopy vs identity)")

    # ── local GF(2) helpers (make 'mul'/'add' available to the rest of Tab 2)
    import hashlib as _hashlib

    def _eye(n): 
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def _xor_mat(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        m, n = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(n)] for i in range(m)]

    def _mul_gf2(A, B):
        if not A or not B or not A[0] or not B[0]: 
            return []
        r, k  = len(A), len(A[0])
        k2, c = len(B), len(B[0])
        if k != k2:
            raise ValueError(f"dimension mismatch: {len(A)}x{len(A[0])} @ {len(B)}x{len(B[0])}")
        out = [[0]*c for _ in range(r)]
        for i in range(r):
            Ai = A[i]
            for t in range(k):
                if Ai[t] & 1:
                    Bt = B[t]
                    for j in range(c):
                        out[i][j] ^= (Bt[j] & 1)
        return out

    # expose 'mul'/'add' names used elsewhere in Tab 2
    def mul(A, B): return _mul_gf2(A, B)
    def add(A, B): return _xor_mat(A, B)

    def _bottom_row(M): 
        return M[-1] if (M and len(M)) else []

    def _stable_hash(obj):
        return _hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    # alias used by the Freeze block you pasted later
    def lane_mask_from_boundaries(boundaries_obj):
        return _lane_mask_from_d3(boundaries_obj)

    # ── file uploader for H (remember filename for certs/bundles)
    f_H = st.file_uploader("Homotopy H (H_corrected.json)", type=["json"], key="H_corr")
    _stamp_filename("fname_h", f_H)
    d_H = read_json_file(f_H) if f_H else None
    H_local = io.parse_cmap(d_H) if d_H else io.parse_cmap({"blocks": {}})

    # ── policy toggle
    st.markdown("### Policy")
    policy_choice = st.radio(
        "Choose policy",
        ["strict", "projected(columns@k=3)"],
        horizontal=True,
        key="policy_choice_k3",
    )

    # ── build active cfg (respect projection_config.json)
    cfg_file = projector.load_projection_config("projection_config.json")
    cfg_proj = cfg_projected_base()
    if cfg_file.get("source", {}).get("3") in ("file", "auto"):
        cfg_proj["source"]["3"] = cfg_file["source"]["3"]
    if "projector_files" in cfg_file and "3" in cfg_file["projector_files"]:
        cfg_proj.setdefault("projector_files", {})["3"] = cfg_file["projector_files"]["3"]
    cfg_active = cfg_strict() if policy_choice == "strict" else cfg_proj
    policy_label = policy_label_from_cfg(cfg_active)

    # ── optional: upload projector file (used when source.3 = file)
    st.markdown("#### Projector (k=3)")
    proj_upload = st.file_uploader(
        "Projector JSON for k=3 (optional; used when source.3 = file)",
        type=["json"], key="proj3_up"
    )
    if proj_upload is not None:
        os.makedirs("projectors", exist_ok=True)
        pj_saved = os.path.join("projectors", proj_upload.name)
        with open(pj_saved, "wb") as _pf:
            _pf.write(proj_upload.getvalue())
        st.caption(f"saved projector: {pj_saved}")
        if policy_choice != "strict":
            cfg_active.setdefault("source", {})["3"] = "file"
            cfg_active.setdefault("projector_files", {})["3"] = pj_saved

    # ── source switcher (writes projection_config.json)
    with st.expander("Projector source (k=3)"):
        cur_src  = cfg_file.get("source", {}).get("3", "auto")
        cur_file = cfg_file.get("projector_files", {}).get("3", "projectors/projector_D3.json")
        st.write(f"Current: source.3 = **{cur_src}**", f"(file: `{cur_file}`)" if cur_src == "file" else "")
        mode_choice = st.radio("Choose source for k=3", options=["auto", "file"],
                               index=(0 if cur_src == "auto" else 1), horizontal=True,
                               key="proj_src_choice_k3")
        file_path = st.text_input("Projector file", value=cur_file, disabled=(mode_choice == "auto"))
        if st.button("Apply projector source", key="apply_proj_src_k3"):
            cfg_file.setdefault("source", {})["3"] = mode_choice
            if mode_choice == "file":
                cfg_file.setdefault("projector_files", {})["3"] = file_path
            else:
                cfg_file.get("projector_files", {}).pop("3", None)
            with open("projection_config.json", "w") as _f:
                _json.dump(cfg_file, _f, indent=2)
            st.success(f"projection_config.json updated → source.3 = {mode_choice}")
            if policy_choice != "strict":
                cfg_active.setdefault("source", {})["3"] = mode_choice
                if mode_choice == "file":
                    cfg_active.setdefault("projector_files", {})["3"] = file_path
                else:
                    cfg_active.get("projector_files", {}).pop("3", None)

    # ── Freeze AUTO → FILE block stays where you already pasted it (uses the alias above)

# ── RUN OVERLAP (creates RunContext SSOT)
if st.button("Run Overlap", key="run_overlap"):
    try:
        # clear stale
        for k in ("proj_meta", "run_ctx", "residual_tags", "overlap_out", "overlap_H"):
            st.session_state.pop(k, None)

        # bind projector (fail-fast on FILE)
        try:
            P_active, meta = projector_choose_active(cfg_active, boundaries)
        except ValueError as e:
            st.error(str(e))
            # persist a minimal run_ctx echo so banners/expands don’t recompute
            d3_now = (boundaries.blocks.__root__.get("3") or [])
            st.session_state["run_ctx"] = {
                "policy_tag": policy_label,
                "mode": "projected(file)" if cfg_active.get("source", {}).get("3") == "file"
                        else ("strict" if policy_choice == "strict" else "projected(auto)"),
                "d3": d3_now,
                "n3": (len(d3_now[0]) if (d3_now and d3_now[0]) else 0),
                "lane_mask_k3": [],
                "P_active": [],
                "projector_filename": (cfg_active.get("projector_files", {}) or {}).get("3", ""),
                "projector_hash": "",
                "projector_consistent_with_d": False,
                "errors": [str(e)],
            }
            st.stop()

        # context details (from meta)
        d3 = meta.get("d3") if "d3" in meta else (boundaries.blocks.__root__.get("3") or [])
        n3 = meta.get("n3") if "n3" in meta else (len(d3[0]) if (d3 and d3[0]) else 0)
        lane_mask = meta.get("lane_mask", _lane_mask_from_d3(boundaries))
        mode = meta.get("mode", "strict")

        # compute overlap (k=3) + residuals
        H2 = (H_local.blocks.__root__.get("2") or [])
        C3 = (cmap.blocks.__root__.get("3") or [])
        I3 = _eye(len(C3)) if C3 else []

        try:
            # R3_strict = H2@d3 + (C3 + I3)  over GF(2)
            R3_strict = _xor_mat(mul(H2, d3), _xor_mat(C3, I3)) if (H2 and d3 and C3) else []
        except Exception as e:
            st.error(f"Shape guard failed at k=3: {e}")
            st.stop()

        def _is_zero(M):
            if not M: return True
            return all(all((x & 1) == 0 for x in row) for row in M)

        def _residual_tag(R, lm):
            if not R or not lm: return "none"
            rows, cols = len(R), len(R[0])
            lanes_idx = [j for j, m in enumerate(lm) if m]
            ker_idx   = [j for j, m in enumerate(lm) if not m]
            def _col_nonzero(j): return any(R[i][j] & 1 for i in range(rows))
            lanes_resid = any(_col_nonzero(j) for j in lanes_idx) if lanes_idx else False
            ker_resid   = any(_col_nonzero(j) for j in ker_idx)   if ker_idx   else False
            if not lanes_resid and not ker_resid: return "none"
            if lanes_resid and not ker_resid: return "lanes"
            if ker_resid and not lanes_resid: return "ker"
            return "mixed"

        tag_strict = _residual_tag(R3_strict, lane_mask)
        eq3_strict = _is_zero(R3_strict)

        if cfg_active.get("enabled_layers"):
            R3_proj = mul(R3_strict, P_active) if (R3_strict and P_active) else []
            eq3_proj = _is_zero(R3_proj)
            tag_proj = _residual_tag(R3_proj, lane_mask)
            out = {"3": {"eq": bool(eq3_proj), "n_k": n3}, "2": {"eq": True}}
            st.session_state["residual_tags"] = {"strict": tag_strict, "projected": tag_proj}
        else:
            out = {"3": {"eq": bool(eq3_strict), "n_k": n3}, "2": {"eq": True}}
            st.session_state["residual_tags"] = {"strict": tag_strict}

        st.json(out)

        # persist RunContext SSOT
        st.session_state["overlap_out"] = out
        st.session_state["overlap_cfg"] = cfg_active
        st.session_state["overlap_policy_label"] = policy_label
        st.session_state["overlap_H"] = H_local
        st.session_state["run_ctx"] = {
            "policy_tag": policy_label,
            "mode": mode,
            "d3": d3, "n3": n3,
            "lane_mask_k3": lane_mask,
            "P_active": P_active,
            "projector_filename": meta.get("projector_filename",""),
            "projector_hash": meta.get("projector_hash",""),
            "projector_consistent_with_d": meta.get("projector_consistent_with_d", None),
            "errors": [],
        }

        # banner strictly from run_ctx
        if mode == "projected(file)":
            if meta.get("projector_consistent_with_d", False):
                st.success(f"projected(file) OK · {meta.get('projector_filename','')} · {meta.get('projector_hash','')[:12]} ✔️")
            else:
                st.warning("Projected(file) is not consistent with current d3 (check shape/idempotence/diag/lane).")

    except Exception as e:
        st.error(f"Overlap run failed: {e}")
        st.stop()

# ── A/B compare (strict vs projected) — standalone + resilient
with st.expander("A/B compare (strict vs projected)"):
    rc  = st.session_state.get("run_ctx") or {}
    ib  = st.session_state.get("_inputs_block") or {}
    if not rc:
        st.info("Run Overlap once to enable A/B.")
    else:
        # status line if we already have a snapshot
        ab = st.session_state.get("ab_compare")
        if ab:
            s_ok = bool(ab.get("strict", {}).get("out", {}).get("3", {}).get("eq", False))
            p_ok = bool(ab.get("projected", {}).get("out", {}).get("3", {}).get("eq", False))
            st.caption(f"Last A/B → strict={'✅' if s_ok else '❌'} · projected={'✅' if p_ok else '❌'} · pair={ab.get('pair_tag','')}")
        if st.button("Run A/B compare", key="ab_run_btn"):
            try:
                # strict leg
                H_used = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})
                out_strict   = overlap_gate.overlap_check(boundaries, cmap, H_used)
                label_strict = policy_label_from_cfg(cfg_strict())

                # projected leg mirrors ACTIVE (auto/file)
                cfg_proj = st.session_state.get("overlap_cfg") or cfg_projected_base()
                # Fail-fast on FILE projector validity
                _P_ab, _meta_ab = projector_choose_active(cfg_proj, boundaries)

                out_proj  = overlap_gate.overlap_check(boundaries, cmap, H_used, projection_config=cfg_proj)
                label_proj = policy_label_from_cfg(cfg_proj)

                # provenance + lane vectors (lightweight)
                d3 = rc.get("d3", [])
                H2 = (H_used.blocks.__root__.get("2") or [])
                C3 = (cmap.blocks.__root__.get("3") or [])
                lane_idx = [j for j,m in enumerate(rc.get("lane_mask_k3", [])) if m]
                def _bottom_row(M): return M[-1] if (M and len(M)) else []
                def _xor(A,B):
                    if not A: return B or []
                    if not B: return A or []
                    return [[(A[i][j]^B[i][j]) for j in range(len(A[0]))] for i in range(len(A))]
                H2d3  = mul(H2, d3) if (H2 and d3) else []
                C3pI3 = _xor(C3, eye(len(C3))) if C3 else []
                def _mask(vec, idx): return [vec[j] for j in idx] if (vec and idx) else []
                lane_vec_H2d3 = _mask(_bottom_row(H2d3), lane_idx)
                lane_vec_C3I  = _mask(_bottom_row(C3pI3), lane_idx)

                # persist snapshot
                inputs_sig = [ib.get("boundaries_hash",""), ib.get("C_hash",""),
                              ib.get("H_hash",""), ib.get("U_hash",""), ib.get("shapes_hash","")]
                st.session_state["ab_compare"] = {
                    "pair_tag": f"{label_strict}__VS__{label_proj}",
                    "inputs_sig": inputs_sig,
                    "lane_mask_k3": rc.get("lane_mask_k3", []),
                    "strict": {
                        "label": label_strict,
                        "cfg":   cfg_strict(),
                        "out":   out_strict,
                        "ker_guard": "enforced",
                        "lane_vec_H2d3": lane_vec_H2d3,
                        "lane_vec_C3plusI3": lane_vec_C3I,
                        "pass_vec": [int(out_strict.get("2",{}).get("eq",False)),
                                     int(out_strict.get("3",{}).get("eq",False))],
                        "projector_hash": "",
                    },
                    "projected": {
                        "label": label_proj,
                        "cfg":   cfg_proj,
                        "out":   out_proj,
                        "ker_guard": "off",
                        "lane_vec_H2d3": lane_vec_H2d3[:],
                        "lane_vec_C3plusI3": lane_vec_C3I[:],
                        "pass_vec": [int(out_proj.get("2",{}).get("eq",False)),
                                     int(out_proj.get("3",{}).get("eq",False))],
                        "projector_filename": _meta_ab.get("projector_filename",""),
                        "projector_hash": _meta_ab.get("projector_hash",""),
                        "projector_consistent_with_d": _meta_ab.get("projector_consistent_with_d", None),
                    },
                }
                st.success("A/B snapshot updated.")
            except ValueError as e:
                st.error(f"A/B projected(file) invalid: {e}")
            except Exception as e:
                st.error(f"A/B compare failed: {e}")

# ── require a run for the rest of the tab (non-blocking now)
_ss = st.session_state
if not (_ss.get("run_ctx") and _ss.get("overlap_out")):
    st.info("Run Overlap first to see results sections.")
    # (no st.stop())
else:
    run_ctx = _ss["run_ctx"]; out = _ss["overlap_out"]; H_used = _ss.get("overlap_H")
    policy_label = _ss.get("overlap_policy_label", policy_label_from_cfg(cfg_active))


    # ───────────────────────────── Gallery + Witness loggers ─────────────────────

import os, json as _json, io, csv, time
from pathlib import Path
from datetime import datetime, timezone

SCHEMA_VERS = "1.0.0"

# --- tiny I/O helpers (guarded; only define if not already present) ----------
def _iso_utc_now():
    return datetime.now(timezone.utc).isoformat()

if "_atomic_append_jsonl" not in globals():
    def _atomic_append_jsonl(path: str, row: dict):
        """Append one JSON object as a line to a .jsonl file atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        b = (_json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        with open(tmp, "wb") as f:
            f.write(b)
            f.flush()
            os.fsync(f.fileno())
        # atomic append
        with open(path, "ab") as f:
            with open(tmp, "rb") as r:
                fb = r.read()
            f.write(fb)
            f.flush()
            os.fsync(f.fileno())
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

if "_load_jsonl_tail" not in globals():
    def _load_jsonl_tail(path: str, limit: int = 10) -> list[dict]:
        out = []
        p = Path(path)
        if not p.exists():
            return out
        # naive tail: read all, keep last N (files are small)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(_json.loads(line))
                except Exception:
                    continue
        return out[-limit:]

# --- SSOT getters -------------------------------------------------------------
_run_ctx   = st.session_state.get("run_ctx") or {}
_out       = st.session_state.get("overlap_out") or {}
_res_tags  = st.session_state.get("residual_tags") or {}
_inputs    = st.session_state.get("_inputs_block") or {}
_di        = st.session_state.get("_district_info") or {}

district_id = _di.get("district_id", "UNKNOWN")
policy_tag  = _run_ctx.get("policy_tag", "strict")
mode        = _run_ctx.get("mode", "strict")
proj_hash   = _run_ctx.get("projector_hash") or ""
proj_file   = (_run_ctx.get("projector_filename") or "")
lane_mask   = _run_ctx.get("lane_mask_k3") or []

# authoritative hashes (only copy from inputs block)
hash_d  = _inputs.get("boundaries_hash", "")
hash_C  = _inputs.get("C_hash", "")      # may be empty if not computed in Inputs
hash_H  = _inputs.get("H_hash", "")
hash_U  = _inputs.get("U_hash", "")
hash_shapes = _inputs.get("shapes_hash", "")

# convenience: eq flags
eq2 = bool(_out.get("2", {}).get("eq", False))
eq3 = bool(_out.get("3", {}).get("eq", False))

# residual tags per leg (already computed during Run Overlap if you wired that)
tag_strict    = _res_tags.get("strict")
tag_projected = _res_tags.get("projected")

# --- paths --------------------------------------------------------------------
gallery_path   = str(Path(DIRS["logs"]) / "gallery.jsonl")
witnesses_path = str(Path(DIRS["logs"]) / "witnesses.jsonl")

# --- UI: Gallery --------------------------------------------------------------
with st.expander("Gallery (projected-green lane exemplars)"):
    st.caption("Append projected **GREEN** cases with stable metadata. Dedupe on (district, hash_d, hash_U, hash_C, hash_H, policy).")

    # Only meaningful when projected mode ran and k=3 passed
    green_ok = (policy_tag.startswith("projected(") and eq3)
    if not green_ok:
        st.info("Run a projected policy and get k=3 = ✅ to enable adding to gallery.")
    else:
        colA, colB, colC = st.columns([2, 2, 3])
        with colA:
            growth_bumps = st.selectbox("Growth bumps", [0, 1, 2, 3], index=0, help="Free label you can use later.")
        with colB:
            strictify = st.selectbox("Strictify?", ["tbd", "yes", "no"], index=0)
        with colC:
            tag = st.text_input("Tag / family label", value="")

        if st.button("➕ Add to Gallery", use_container_width=True):
            # row payload
            row = {
                "schema_version": SCHEMA_VERS,
                "created_at": _iso_utc_now(),
                "run_id": st.session_state.get("run_id", ""),  # optional if you set it earlier
                "district": district_id,
                "policy": policy_tag,
                "hash_d": hash_d,
                "hash_U": hash_U,
                "hash_suppC": _inputs.get("suppC_hash", ""),   # if Inputs computed one
                "hash_suppH": _inputs.get("suppH_hash", ""),
                "hash_C": hash_C,
                "hash_H": hash_H,
                "projector_hash": proj_hash,
                "projector_file": proj_file,
                "lane_mask_k3": lane_mask,
                "growth_bumps": int(growth_bumps),
                "tag": tag,
                "strictify": strictify,
                # link back to cert if you saved it earlier
                "content_hash": (st.session_state.get("cert_payload", {}) or {}).get("integrity", {}).get("content_hash", ""),
            }

            # dedupe: scan last ~200 rows (fast)
            existing = _load_jsonl_tail(gallery_path, limit=10000)  # still fine unless file is huge
            key = (row["district"], row["hash_d"], row["hash_U"], row["hash_C"], row["hash_H"], row["policy"])
            have = False
            for r in existing:
                k2 = (r.get("district"), r.get("hash_d"), r.get("hash_U"), r.get("hash_C"), r.get("hash_H"), r.get("policy"))
                if k2 == key:
                    have = True
                    break

            if have:
                st.warning("Duplicate skipped (same district + hashes + policy already in gallery).")
            else:
                _atomic_append_jsonl(gallery_path, row)
                st.success("Added to gallery ✅")

    # show tail
    rows = _load_jsonl_tail(gallery_path, limit=10)
    if rows:
        st.caption("Last 10 rows")
        st.dataframe(rows, use_container_width=True)
        with st.columns(2)[0]:
            with open(gallery_path, "rb") as f:
                st.download_button("⬇️ Download gallery.jsonl", f, file_name="gallery.jsonl", use_container_width=True)

# --- UI: Witness Tracker ------------------------------------------------------
with st.expander("Witness Tracker (why it didn’t go green)"):
    st.caption("Log a witness when k=3 is ❌ but guards are green, so the ‘why’ is remembered.")

    # heuristic guard: only allow when NOT green at k=3
    not_green = (not eq3)
    # if you track grid/fence flags in checks, surface them here; default True when unknown
    grid_ok  = bool(st.session_state.get("checks_block", {}).get("grid", True))
    fence_ok = bool(st.session_state.get("checks_block", {}).get("fence", True))

    if not_green and grid_ok and fence_ok:
        reason = st.selectbox(
            "Witness reason",
            ["lanes-persist", "policy-mismatch", "needs-new-R", "grammar-drift", "other"],
            index=0,
        )
        note = st.text_area("Note (free text)", placeholder="e.g., G7 flavor; two-row push still leaves lane residual.")
        # choose which leg residual to capture
        leg_choice = ("projected" if policy_tag.startswith("projected(") else "strict")
        residual_tag = (_res_tags.get(leg_choice) or
                        ("lanes" if reason == "lanes-persist" else "mixed"))

        if st.button("📝 Log witness", use_container_width=True):
            row = {
                "schema_version": SCHEMA_VERS,
                "created_at": _iso_utc_now(),
                "run_id": st.session_state.get("run_id", ""),
                "district": district_id,
                "policy": policy_tag,
                "residual_tag": residual_tag,
                "reason": reason,
                "note": note,
                # hashes (copy from Inputs SSOT only)
                "hash_d": hash_d,
                "hash_U": hash_U,
                "hash_C": hash_C,
                "hash_H": hash_H,
                "hash_suppC": _inputs.get("suppC_hash", ""),
                "hash_suppH": _inputs.get("suppH_hash", ""),
                "projector_hash": proj_hash,
                "projector_file": proj_file,
                "lane_mask_k3": lane_mask,
                "content_hash": (st.session_state.get("cert_payload", {}) or {}).get("integrity", {}).get("content_hash", ""),
            }
            _atomic_append_jsonl(witnesses_path, row)
            st.success("Witness logged ✅")
    else:
        if eq3:
            st.info("k=3 is GREEN — nothing to witness here.")
        elif not grid_ok or not fence_ok:
            st.info("Guards are RED — fix grid/fence first; witnesses are for k=3 failures with guards green.")

    # show tail
    rows = _load_jsonl_tail(witnesses_path, limit=10)
    if rows:
        st.caption("Last 10 witnesses")
        st.dataframe(rows, use_container_width=True)
        with st.columns(2)[0]:
            with open(witnesses_path, "rb") as f:
                st.download_button("⬇️ Download witnesses.jsonl", f, file_name="witnesses.jsonl", use_container_width=True)

# ───────────────────────── Parity Suite: pair helpers ─────────────────────────
from pathlib import Path
import json as _json

def _safe_parse_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return _json.load(f)

def load_fixture_from_paths(*, boundaries_path: str, cmap_path: str, H_path: str, shapes_path: str):
    """
    Load a fixture from file paths and return a dict of parsed objects ready for overlap.
    You can mix-and-match file names; this does no hashing (Inputs tab already owns hashes).
    """
    dB = _safe_parse_json(boundaries_path)
    dC = _safe_parse_json(cmap_path)
    dH = _safe_parse_json(H_path)
    dU = _safe_parse_json(shapes_path)

    return {
        "boundaries": io.parse_boundaries(dB),
        "cmap":       io.parse_cmap(dC),
        "H":          io.parse_cmap(dH),    # H uses the same JSON-safe cmap schema
        "shapes":     io.parse_shapes(dU),
    }

def add_parity_pair(*, label: str, left_fixture: dict, right_fixture: dict):
    """
    Append a single parity pair into st.session_state['parity_pairs'].
    Each fixture must be a dict with keys: boundaries, cmap, H, shapes (already parsed via io.*).
    """
    req = ("boundaries", "cmap", "H", "shapes")
    for side_name, fx in (("left", left_fixture), ("right", right_fixture)):
        if not isinstance(fx, dict) or any(k not in fx for k in req):
            raise ValueError(f"{side_name} fixture malformed; expected keys {req}")

    st.session_state.setdefault("parity_pairs", [])
    st.session_state["parity_pairs"].append({
        "label": label,
        "left":  left_fixture,
        "right": right_fixture,
    })
    return len(st.session_state["parity_pairs"])

def clear_parity_pairs():
    """Wipe any queued pairs (useful before building a new batch)."""
    st.session_state["parity_pairs"] = []

def set_parity_pairs_from_fixtures(pairs_spec: list[dict]):
    """
    Bulk helper: pairs_spec is a list of dicts:
      {
        "label": "D2(101)↔D3(110)",
        "left":  {"boundaries": "inputs/D2/boundaries.json",
                  "cmap": "inputs/D2/cmap.json",
                  "H": "inputs/D2/H.json",
                  "shapes": "inputs/D2/shapes.json"},
        "right": {"boundaries": "inputs/D3/boundaries.json",
                  "cmap": "inputs/D3/cmap.json",
                  "H": "inputs/D3/H.json",
                  "shapes": "inputs/D3/shapes.json"}
      }
    Loads & parses each side, then stores parity_pairs in session.
    """
    clear_parity_pairs()
    for row in pairs_spec:
        label = row.get("label", "PAIR")
        Lp = row.get("left",  {})
        Rp = row.get("right", {})
        L = load_fixture_from_paths(
            boundaries_path=Lp["boundaries"],
            cmap_path=Lp["cmap"],
            H_path=Lp["H"],
            shapes_path=Lp["shapes"],
        )
        R = load_fixture_from_paths(
            boundaries_path=Rp["boundaries"],
            cmap_path=Rp["cmap"],
            H_path=Rp["H"],
            shapes_path=Rp["shapes"],
        )
        add_parity_pair(label=label, left_fixture=L, right_fixture=R)
    return len(st.session_state.get("parity_pairs", []))

# ───────────────── Parity sample pairs (optional, guarded) ──────────────────
from pathlib import Path

def _all_exist(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)

with st.expander("Parity: queue sample D2/D3/D4 pairs (optional)"):
    st.caption("Only queues pairs if the example files exist under ./inputs/. "
               "If you don't have these, skip this section.")
    col1, col2 = st.columns(2)
    with col1:
        do_self = st.button("Queue SELF (current fixture vs itself)", key="pp_self_btn")
    with col2:
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
        # Flatten to check existence
        flat = []
        for row in spec:
            L, R = row["left"], row["right"]
            flat += [L["boundaries"], L["cmap"], L["H"], L["shapes"],
                     R["boundaries"], R["cmap"], R["H"], R["shapes"]]
        if not _all_exist(flat):
            st.info("Example files not found under ./inputs — skipping queuing (this is fine).")
        else:
            try:
                set_parity_pairs_from_fixtures(spec)
                st.success("Queued D2↔D3 and D3↔D4 example pairs.")
            except Exception as e:
                st.error(f"Could not queue examples: {e}")





    # ───────────────────────────── Parity Suite helpers ──────────────────────────

import os, json as _json
from pathlib import Path
from datetime import datetime, timezone

SCHEMA_VERS = "1.0.0"

def _iso_utc_now():
    return datetime.now(timezone.utc).isoformat()

# -- transactional write (JSON) ------------------------------------------------
def _atomic_write_json(path: str, payload: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    blob = _json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)

# -- mirror the ACTIVE projected source (AUTO/FILE) from RunContext ------------
def _mirror_active_projected_cfg():
    run_ctx = st.session_state.get("run_ctx") or {}
    if not run_ctx or run_ctx.get("mode") == "strict":
        # default projected(auto) if no run yet (allows parity scaffolding)
        return {"enabled_layers":[3], "modes":{"3":"columns"}, "source":{"3":"auto"}, "projector_files":{"3":"projector_D3.json"}}
    if run_ctx.get("mode") == "projected(auto)":
        return {"enabled_layers":[3], "modes":{"3":"columns"}, "source":{"3":"auto"}, "projector_files":{"3":"projector_D3.json"}}
    # file-mode: reuse the exact file path
    pj_file = run_ctx.get("projector_filename") or "projectors/projector_D3.json"
    return {"enabled_layers":[3], "modes":{"3":"columns"}, "source":{"3":"file"}, "projector_files":{"3": pj_file}}

# -- run a single leg (strict or projected), returning (ok, out_dict, err_msg) -
def _run_overlap_strict(boundaries, cmap, H):
    try:
        out = overlap_gate.overlap_check(boundaries, cmap, H)
        return True, out, ""
    except Exception as e:
        return False, {}, f"strict run failed: {e}"

def _run_overlap_projected(boundaries, cmap, H, cfg_proj):
    """
    Mirrors Tab 2 rules: if FILE is selected, validate Π against THIS boundaries.d3,
    fail-fast with the exact validator message; no AUTO fallback.
    """
    try:
        # Validate/choose active projector for THIS fixture
        P_active, meta = projector_choose_active(cfg_proj, boundaries)
    except ValueError as ve:
        return False, {}, f"projected(file) invalid: {ve}"

    # Preload cache for file paths (no-ops for AUTO)
    cache = projector.preload_projectors_from_files(cfg_proj)
    try:
        out = overlap_gate.overlap_check(boundaries, cmap, H, projection_config=cfg_proj, projector_cache=cache)
        return True, out, ""
    except Exception as e:
        return False, {}, f"projected run failed: {e}"

# -- check a "pair" label by running both strict and projected -----------------
def _parity_check_pair(pair_label: str, fixA: dict, fixB: dict, cfg_proj: dict):
    """
    fixX: {"boundaries":..., "cmap":..., "H":..., "shapes":...}
    Returns:
      {
        "pair": pair_label,
        "strict": bool,        # both sides green at k=3
        "projected": bool,     # both sides green at k=3 under mirrored policy
        "strict_left": bool, "strict_right": bool,
        "proj_left": bool,   "proj_right": bool,
        "err_left": "", "err_right": "",
      }
    """
    # Left
    ok_s_L, out_s_L, err_s_L = _run_overlap_strict(fixA["boundaries"], fixA["cmap"], fixA["H"])
    ok_p_L, out_p_L, err_p_L = _run_overlap_projected(fixA["boundaries"], fixA["cmap"], fixA["H"], cfg_proj)

    # Right
    ok_s_R, out_s_R, err_s_R = _run_overlap_strict(fixB["boundaries"], fixB["cmap"], fixB["H"])
    ok_p_R, out_p_R, err_p_R = _run_overlap_projected(fixB["boundaries"], fixB["cmap"], fixB["H"], cfg_proj)

    sL = ok_s_L and bool(out_s_L.get("3", {}).get("eq", False))
    sR = ok_s_R and bool(out_s_R.get("3", {}).get("eq", False))
    pL = ok_p_L and bool(out_p_L.get("3", {}).get("eq", False))
    pR = ok_p_R and bool(out_p_R.get("3", {}).get("eq", False))

    return {
        "pair": pair_label,
        "strict": (sL and sR),
        "projected": (pL and pR),
        "strict_left": sL, "strict_right": sR,
        "proj_left": pL,   "proj_right": pR,
        "err_left":  (err_s_L or err_p_L),
        "err_right": (err_s_R or err_p_R),
    }

# -- build and write the parity report JSON -----------------------------------
def _write_parity_report(pairs_rows: list[dict]):
    run_ctx  = st.session_state.get("run_ctx") or {}
    inputs   = st.session_state.get("_inputs_block") or {}
    policy   = run_ctx.get("policy_tag", "strict")
    proj_hash = run_ctx.get("projector_hash") or ""
    run_id   = st.session_state.get("run_id", "")
    created  = _iso_utc_now()

    payload = {
        "schema_version": SCHEMA_VERS,
        "created_at": created,
        "run_id": run_id,
        "policy": policy,                 # mirrors active policy: projected(...,auto|file) or strict
        "projector_hash": proj_hash,      # if file-mode, this is the file content hash; auto: Π_auto hash
        "inputs_sig": [
            inputs.get("boundaries_hash",""),
            inputs.get("C_hash",""),
            inputs.get("H_hash",""),
            inputs.get("U_hash",""),
            inputs.get("shapes_hash",""),
        ],
        "pairs": pairs_rows,              # list of _parity_check_pair rows
    }

    out_path = str(Path(DIRS["reports"]) / "parity_report.json")
    _atomic_write_json(out_path, payload)
    return out_path
    # ───────────────────────── Parity Suite: import/export ─────────────────────────
import os, json as _json, tempfile, shutil
from pathlib import Path
from datetime import datetime, timezone

PARITY_SCHEMA_VERSION = "1.0.0"
PARITY_DEFAULT_PATH   = Path("logs") / "parity_pairs.json"

def _ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parity_pairs_payload(pairs: list[dict]) -> dict:
    """Wrap session pairs with schema + metadata for stable persistence."""
    return {
        "schema_version": PARITY_SCHEMA_VERSION,
        "saved_at": _utc_now_iso(),
        "count": len(pairs),
        "pairs": [
            {
                "label": row.get("label", "PAIR"),
                # Persist only source paths & minimal identity; do not dump parsed objects
                "left":  {
                    "boundaries": row.get("left_path_boundaries",  row.get("left",  {}).get("boundaries_path", "")),
                    "cmap":       row.get("left_path_cmap",        row.get("left",  {}).get("cmap_path", "")),
                    "H":          row.get("left_path_H",           row.get("left",  {}).get("H_path", "")),
                    "shapes":     row.get("left_path_shapes",      row.get("left",  {}).get("shapes_path", "")),
                },
                "right": {
                    "boundaries": row.get("right_path_boundaries", row.get("right", {}).get("boundaries_path", "")),
                    "cmap":       row.get("right_path_cmap",       row.get("right", {}).get("cmap_path", "")),
                    "H":          row.get("right_path_H",          row.get("right", {}).get("H_path", "")),
                    "shapes":     row.get("right_path_shapes",     row.get("right", {}).get("shapes_path", "")),
                },
            }
        for row in pairs],
    }

def _pairs_from_payload(payload: dict) -> list[dict]:
    """Convert stored rows back into loadable spec (paths only)."""
    if not isinstance(payload, dict):
        return []
    rows = []
    for r in payload.get("pairs", []):
        rows.append({
            "label": r.get("label", "PAIR"),
            "left": {
                "boundaries": r.get("left",  {}).get("boundaries", ""),
                "cmap":       r.get("left",  {}).get("cmap",       ""),
                "H":          r.get("left",  {}).get("H",          ""),
                "shapes":     r.get("left",  {}).get("shapes",     ""),
            },
            "right": {
                "boundaries": r.get("right", {}).get("boundaries", ""),
                "cmap":       r.get("right", {}).get("cmap",       ""),
                "H":          r.get("right", {}).get("H",          ""),
                "shapes":     r.get("right", {}).get("shapes",     ""),
            },
        })
    return rows

def export_parity_pairs(path: str | Path = PARITY_DEFAULT_PATH) -> str:
    """Atomically write the current session parity pairs to disk as JSON."""
    path = Path(path)
    _ensure_parent_dir(path)

    pairs = st.session_state.get("parity_pairs", []) or []
    # For future-proofing, try to attach remembered source paths if present
    payload = _parity_pairs_payload(pairs)

    # atomic write
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        _json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

    return str(path)

def import_parity_pairs(path: str | Path = PARITY_DEFAULT_PATH, *, merge: bool = False) -> int:
    """
    Load parity pairs spec from disk.
    - If merge=False (default): replace session pairs.
    - If merge=True: append to existing session pairs.
    Returns number of pairs now in session.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No parity pairs file at {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)

    # Soft schema guard
    ver = payload.get("schema_version", "0.0.0")
    if ver.split(".")[0] != PARITY_SCHEMA_VERSION.split(".")[0]:
        st.warning(f"parity_pairs schema version differs (file={ver}, app={PARITY_SCHEMA_VERSION}); attempting best-effort load.")

    pairs_spec = _pairs_from_payload(payload)

    # Turn spec→parsed fixtures and store in session via existing bulk helper
    if not merge:
        clear_parity_pairs()
    set_parity_pairs_from_fixtures(pairs_spec)
    return len(st.session_state.get("parity_pairs", []))

# ── tiny UI to drive import/export (optional; place under the Parity expander)
with st.expander("Parity pairs: import/export"):
    colA, colB, colC = st.columns([3,3,2])
    with colA:
        export_path = st.text_input("Export path", value=str(PARITY_DEFAULT_PATH), key="pp_export_path")
    with colB:
        import_path = st.text_input("Import path", value=str(PARITY_DEFAULT_PATH), key="pp_import_path")
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

# ─────────────────────── Perturbation Sanity + Fence Stress ───────────────────────
import os, csv, tempfile, json as _json
from pathlib import Path
from datetime import datetime, timezone

PERTURB_SCHEMA_VERSION = "1.0.0"
FENCE_SCHEMA_VERSION   = "1.0.0"
PERTURB_OUT_PATH       = Path("reports") / "perturbation_sanity.csv"
FENCE_OUT_PATH         = Path("reports") / "fence_stress.csv"

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _atomic_write_csv(path: Path, header: list[str], rows: list[list], meta_comments: list[str]):
    _ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", newline="") as tmp:
        for line in meta_comments:
            tmp.write(f"# {line}\n")
        w = csv.writer(tmp)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

# ---- tiny GF(2) helpers ---------------------------------------------------------
def _mat(M): return M if isinstance(M, list) else []
def _clone_mat(M): return [row[:] for row in _mat(M)]
def _mul_gf2(A, B):
    if not A or not A[0] or not B or not B[0]: return []
    r,k  = len(A), len(A[0]); k2,c = len(B), len(B[0])
    if k != k2: return []
    out = [[0]*c for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for t in range(k):
            if Ai[t] & 1:
                Bt = B[t]
                for j in range(c):
                    out[i][j] ^= (Bt[j] & 1)
    return out
def _eye(n): return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

def _lane_mask_from_d3_matrix(d3):
    if not d3 or not d3[0]: return []
    n3 = len(d3[0])
    return [1 if any(row[j] & 1 for row in d3) else 0 for j in range(n3)]

def _residual_tag_from(R3, lane_mask):
    """Classify residual columns wrt lanes."""
    if not R3 or not R3[0]: return "none"
    n3 = len(R3[0]); lane_mask = (lane_mask or [0]*n3)
    def _col_nonzero(j):
        for i in range(len(R3)):
            if R3[i][j] & 1: return True
        return False
    lanes_resid = any(_col_nonzero(j) for j in range(min(n3, len(lane_mask))) if lane_mask[j]==1)
    ker_resid   = any(_col_nonzero(j) for j in range(min(n3, len(lane_mask))) if lane_mask[j]==0)
    if lanes_resid and ker_resid: return "mixed"
    if lanes_resid: return "lanes"
    if ker_resid:   return "ker"
    return "none"

def _compute_R3_strict(boundaries, cmap, H):
    """R3 = H2@d3 + (C3 + I3) over GF(2). Returns (R3, dims_string) or ([], msg) on shape issue."""
    try:
        d3 = (boundaries.blocks.__root__.get("3") or [])
        H2 = (H.blocks.__root__.get("2") or []) if H else []
        C3 = (cmap.blocks.__root__.get("3") or [])
    except Exception:
        return [], "missing blocks"
    n2 = len(d3); n3 = (len(d3[0]) if d3 else 0)
    if not (H2 and d3 and C3 and n2 and n3):
        return [], "missing blocks"
    # shape guards: H2(n3×n2) @ d3(n2×n3) → n3×n3 ; C3 is n3×n3
    ok_H_dims = (len(H2)==n3) and (len(H2[0])==n2 if H2 else False)
    ok_C_dims = (len(C3)==n3) and (len(C3[0])==n3 if C3 else False)
    if not (ok_H_dims and ok_C_dims):
        return [], f"Shape guard failed at k=3: H2({len(H2)}×{len(H2[0]) if H2 else 0}) @ d3({n2}×{n3}), C3({len(C3)}×{len(C3[0]) if C3 else 0})"
    H2d3 = _mul_gf2(H2, d3)
    C3pI = _mul_gf2(_eye(n3), _eye(n3))  # cheap copy of I3
    # C3 + I3
    C3pI = [[(C3[i][j] ^ (1 if i==j else 0)) for j in range(n3)] for i in range(n3)]
    # R3
    R3 = [[(H2d3[i][j] ^ C3pI[i][j]) for j in range(n3)] for i in range(n3)]
    return R3, f"n2={n2}, n3={n3}"

# =============== Perturbation Sanity (strict anchors only) ======================
with st.expander("Perturbation Sanity (strict anchors)"):
    st.caption("Flip one bit in C₃ (interpreted inside C₃+I₃) and observe which guard trips first. "
               "Reads hashes from Inputs SSOT; runs **strict** policy only.")
    # controls
    colsP = st.columns([1,1,1,2])
    with colsP[0]:
        max_flips = st.number_input("Max flips", min_value=1, max_value=200, value=20, step=1, key="pert_max")
    with colsP[1]:
        scan_mode = st.selectbox("Scan region", ["lanes only", "kernel only", "all columns"], index=0, key="pert_region")
    with colsP[2]:
        pert_path = st.text_input("CSV path", value=str(PERTURB_OUT_PATH), key="pert_path")
    with colsP[3]:
        st.write("")

    if st.button("Run Perturbation Sanity", key="pert_run"):
        try:
            # load base blocks
            d3   = (boundaries.blocks.__root__.get("3") or [])
            H2   = (H.blocks.__root__.get("2") or []) if 'H' in globals() else []
            C3_0 = (cmap.blocks.__root__.get("3") or [])
            if not d3 or not H2 or not C3_0:
                st.error("Missing d3/H2/C3 blocks.")
                st.stop()

            # derive lane mask and columns to scan
            lane_mask = _lane_mask_from_d3_matrix(d3)
            n3 = len(d3[0]) if d3 else 0
            if scan_mode == "lanes only":
                cols = [j for j,m in enumerate(lane_mask) if m==1]
            elif scan_mode == "kernel only":
                cols = [j for j,m in enumerate(lane_mask) if m==0]
            else:
                cols = list(range(n3))
            cols = cols[:int(max_flips)]

            # base strict run (not calling overlap gate: compute R3 and tag here)
            R3_base, dims = _compute_R3_strict(boundaries, cmap, H)
            if not R3_base:
                st.error(f"Base R3 build failed ({dims}).")
                st.stop()
            base_tag = _residual_tag_from(R3_base, lane_mask)

            # iterate flips
            rows = []
            for j in cols:
                C3 = _clone_mat(C3_0)
                # flip bit (i=j for diagonal flips, otherwise choose row=j to keep it simple & visible)
                i = j if j < len(C3) else (len(C3)-1)
                if i < 0 or j >= len(C3[0]):  # guard
                    continue
                C3[i][j] ^= 1  # flip one bit in C3

                # rebuild R3 with perturbed C3
                try:
                    # Temporarily parse a perturbed cmap for pure strict math (no projection involved)
                    cmap_pert = io.parse_cmap({"blocks":{"3":C3, "2": (cmap.blocks.__root__.get("2") or [])}})
                except Exception:
                    continue
                R3p, _ = _compute_R3_strict(boundaries, cmap_pert, H)
                tag = _residual_tag_from(R3p, lane_mask)

                # first failing guard (grid, fence, ker_guard) via unit overlap_check(strict)
                out_strict = overlap_gate.overlap_check(boundaries, cmap_pert, H, projection_config=cfg_strict())
                grid_ok  = bool(out_strict.get("grid_ok", True))
                fence_ok = bool(out_strict.get("fence_ok", True))
                k2_ok    = bool(out_strict.get("2", {}).get("eq", False))
                k3_ok    = bool(out_strict.get("3", {}).get("eq", False))
                if not grid_ok:  failed = "grid"
                elif not fence_ok: failed = "fence"
                elif not k2_ok:    failed = "k2"
                elif not k3_ok:    failed = "k3"
                else:              failed = "none"

                rows.append([j, base_tag, tag, failed])

            # persist CSV
            meta = [
                f"schema_version={PERTURB_SCHEMA_VERSION}",
                f"saved_at={_utc_iso()}",
                f"policy_tag=strict",
                f"boundaries_hash={st.session_state.get('_inputs_block',{}).get('boundaries_hash','')}",
                f"C_hash={st.session_state.get('_inputs_block',{}).get('C_hash','')}",
                f"H_hash={st.session_state.get('_inputs_block',{}).get('H_hash','')}",
                f"U_hash={st.session_state.get('_inputs_block',{}).get('U_hash','')}",
                f"scan_mode={scan_mode}",
            ]
            header = ["col_flip", "base_residual", "pert_residual", "first_guard_failed"]
            _atomic_write_csv(Path(pert_path), header, rows, meta)
            st.success(f"Saved perturbation report → {pert_path}")
            # small view
            if rows:
                import pandas as _pd
                st.dataframe(_pd.DataFrame(rows, columns=header), use_container_width=True, height=300)

        except Exception as e:
            st.error(f"Perturbation failed: {e}")

# =========================== Fence Stress (U shrink / U+) =======================
with st.expander("Fence Stress (shrink U / try U⁺)"):
    st.caption("Experiment with carrier U: shrink (expect Fence RED) and add a small U⁺ (see if projected greens strictify). "
               "Projected legs re-use **active** policy source (auto/file) via RunContext.")
    colsF = st.columns([1,1,2])
    with colsF[0]:
        do_shrink = st.checkbox("Shrink U", value=True, key="fence_do_shrink")
    with colsF[1]:
        do_expand = st.checkbox("Try U⁺", value=True, key="fence_do_expand")
    with colsF[2]:
        fence_path = st.text_input("CSV path", value=str(FENCE_OUT_PATH), key="fence_path")

    # simple structural transforms (no geometry assumptions)
    def _U_like(shape, rows, cols, fill=0):
        return [[fill]*cols for _ in range(rows)]

    def _shrink_mask(U):
        """Zero out outer ring (one cell border) if present."""
        U = _clone_mat(U)
        if not U: return U
        r, c = len(U), len(U[0])
        for j in range(c):
            U[0][j] = 0
            U[r-1][j] = 0
        for i in range(r):
            U[i][0] = 0
            U[i][c-1] = 0
        return U

    def _expand_mask(U):
        """Add a thin ring of ones at the border."""
        U = _clone_mat(U)
        if not U: return U
        r, c = len(U), len(U[0])
        for j in range(c):
            U[0][j] = 1
            U[r-1][j] = 1
        for i in range(r):
            U[i][0] = 1
            U[i][c-1] = 1
        return U

    def _shapes_override(base_shapes, U2=None, U3=None):
        try:
            B = base_shapes.dict()
        except Exception:
            B = _json.loads(_json.dumps(base_shapes))
        blocks = {}
        if U2 is not None: blocks["2"] = U2
        if U3 is not None: blocks["3"] = U3
        # keep other degrees if present
        return io.parse_shapes({"blocks": {**(B.get("blocks", {})), **blocks}})

    if st.button("Run Fence Stress", key="fence_run"):
        try:
            # base U masks (if missing, synthesize zeros of matching shape)
            U2_base = None; U3_base = None
            try:
                U2_base = shapes.blocks.__root__.get("2")
                U3_base = shapes.blocks.__root__.get("3")
            except Exception:
                U2_base = None; U3_base = None

            d2 = (boundaries.blocks.__root__.get("2") or [])
            d3 = (boundaries.blocks.__root__.get("3") or [])
            U2 = U2_base if U2_base else _U_like(None, len(d2), len(d2[0]) if d2 else 0, 0)
            U3 = U3_base if U3_base else _U_like(None, len(d3), len(d3[0]) if d3 else 0, 0)

            tasks = []
            if do_shrink: tasks.append(("U_shrink", _shrink_mask(U2), _shrink_mask(U3)))
            if do_expand: tasks.append(("U_plus",   _expand_mask(U2), _expand_mask(U3)))
            if not tasks:
                st.info("Nothing to do—select at least one of Shrink U / Try U⁺.")
                st.stop()

            rows = []
            # Strict cfg and Projected cfg (mirror active)
            cfg_str = cfg_strict()
            cfg_proj = st.session_state.get("overlap_cfg") or cfg_projected_base()

            for tag, U2_new, U3_new in tasks:
                shapes_new = _shapes_override(shapes, U2=U2_new, U3=U3_new)

                # Strict run
                out_s = overlap_gate.overlap_check(boundaries, cmap, H, projection_config=cfg_str)
                s_pass = [
                    int(out_s.get("2", {}).get("eq", False)),
                    int(out_s.get("3", {}).get("eq", False)),
                ]
                # Projected run (re-use active policy source and projector cache)
                cache = st.session_state.get("_projector_cache")  # may be None; overlap_gate can handle
                out_p = overlap_gate.overlap_check(boundaries, cmap, H,
                                                   projection_config=cfg_proj,
                                                   projector_cache=cache)
                p_pass = [
                    int(out_p.get("2", {}).get("eq", False)),
                    int(out_p.get("3", {}).get("eq", False)),
                ]

                rows.append([tag, s_pass[0], s_pass[1], p_pass[0], p_pass[1]])

            # persist CSV
            inputs = st.session_state.get("_inputs_block", {})
            meta = [
                f"schema_version={FENCE_SCHEMA_VERSION}",
                f"saved_at={_utc_iso()}",
                f"policy_tag={st.session_state.get('overlap_policy_label', st.session_state.get('run_ctx',{}).get('policy_tag',''))}",
                f"boundaries_hash={inputs.get('boundaries_hash','')}",
                f"C_hash={inputs.get('C_hash','')}",
                f"H_hash={inputs.get('H_hash','')}",
                f"U_hash={inputs.get('U_hash','')}",
            ]
            header = ["U_variant", "strict_k2", "strict_k3", "projected_k2", "projected_k3"]
            _atomic_write_csv(Path(fence_path), header, rows, meta)
            st.success(f"Saved fence stress → {fence_path}")

            # show table
            import pandas as _pd
            st.dataframe(_pd.DataFrame(rows, columns=header), use_container_width=True, height=240)

        except Exception as e:
            st.error(f"Fence stress failed: {e}")


# ─────────────────────────── Coverage Sampler (seeded) ───────────────────────────
import os, csv, tempfile, json as _json
from pathlib import Path
from datetime import datetime, timezone
import random

COVERAGE_SCHEMA_VERSION = "1.0.0"
COVERAGE_DEFAULT_PATH   = Path("reports") / "coverage_sampling.csv"

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ---- GF(2) helpers (no numpy) ------------------------------------------------
def _gf2_rank(M: list[list[int]]) -> int:
    """Row-reduction rank over GF(2). M is list of list of 0/1."""
    if not M: return 0
    # deep copy and normalize to {0,1}
    A = [[int(x) & 1 for x in row] for row in M]
    r, c = len(A), len(A[0]) if A[0] else 0
    rank = 0
    col = 0
    for row in range(r):
        # find pivot in/after 'col'
        while col < c and all(A[i][col] == 0 for i in range(row, r)):
            col += 1
        if col >= c:
            break
        # swap pivot row
        for i in range(row, r):
            if A[i][col]:
                if i != row:
                    A[row], A[i] = A[i], A[row]
                break
        # eliminate below
        for i in range(row+1, r):
            if A[i][col]:
                A[i] = [(a ^ b) for a, b in zip(A[i], A[row])]
        rank += 1
        col += 1
    return rank

def _lane_mask_from_matrix(d3: list[list[int]]) -> list[int]:
    if not d3 or not d3[0]:
        return []
    n3 = len(d3[0])
    return [1 if any(row[j] & 1 for row in d3) else 0 for j in range(n3)]

def _lane_pattern_str(mask: list[int]) -> str:
    return "".join(str(int(x)) for x in (mask or []))

# ---- Signature & sampling ----------------------------------------------------
def _signature_from_d3(d3: list[list[int]]) -> dict:
    """Canonical signature: rank, ker_dim, lane_pattern (string)."""
    if not d3 or not d3[0]:
        return {"rank": 0, "ker_dim": 0, "lane_pattern": ""}
    n2, n3 = len(d3), len(d3[0])
    rk = _gf2_rank(d3)
    ker = max(0, n3 - rk)
    lanes = _lane_mask_from_matrix(d3)
    return {"rank": rk, "ker_dim": ker, "lane_pattern": _lane_pattern_str(lanes)}

def _sample_binary_matrix(rows: int, cols: int, *, rng: random.Random, density: float = 0.5) -> list[list[int]]:
    """Sample a binary matrix with given density (probability of 1)."""
    p = max(0.0, min(1.0, float(density)))
    return [[1 if rng.random() < p else 0 for _ in range(cols)] for __ in range(rows)]

# ---- Registry of known district signatures (tiny & editable) -----------------
# We store a set of strings like "r={rank}|k={ker}|mask={lane_pattern}"
st.session_state.setdefault("district_signature_registry", set())

def _sig_key(sig: dict) -> str:
    return f"r={sig.get('rank',0)}|k={sig.get('ker_dim',0)}|mask={sig.get('lane_pattern','')}"

def _seed_registry_with_current(boundaries) -> str:
    """Add current d3 signature into the registry (returns key)."""
    d3 = (boundaries.blocks.__root__.get("3") or [])
    sig = _signature_from_d3(d3)
    key = _sig_key(sig)
    st.session_state["district_signature_registry"].add(key)
    return key

# ---- Seed derivation (deterministic unless overridden) -----------------------
def _default_sampler_seed(inputs_block: dict) -> str:
    # Only hashes from _inputs_block (SSOT). If missing, fall back to lane pattern.
    bh = inputs_block.get("boundaries_hash", "")
    ch = inputs_block.get("C_hash", "")
    hh = inputs_block.get("H_hash", "")
    uh = inputs_block.get("U_hash", "")
    base = f"{bh}|{ch}|{hh}|{uh}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:8] if base.strip("|") else "deadbeef"

def _atomic_write_csv(path: Path, header: list[str], rows: list[list], meta_comments: list[str]):
    _ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", newline="") as tmp:
        # metadata comments first (schema, seed, policy/run info)
        for line in meta_comments:
            tmp.write(f"# {line}\n")
        w = csv.writer(tmp)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

# ---- UI ----------------------------------------------------------------------
with st.expander("Coverage Sampler (seeded)"):
    # Ensure the registry has the current district signature at least once
    if st.button("Add current district signature to registry", key="covsig_add"):
        try:
            key = _seed_registry_with_current(boundaries)
            st.success(f"Added signature → {key}")
        except Exception as e:
            st.error(f"Could not add signature: {e}")

    # Controls
    d3_now = (boundaries.blocks.__root__.get("3") or [])
    n2 = len(d3_now); n3 = (len(d3_now[0]) if d3_now else 0)
    st.caption(f"Active dims: n2×n3 = {n2}×{n3}")

    colsA = st.columns([1,1,1,2])
    with colsA[0]:
        cov_count = st.number_input("Samples", min_value=10, max_value=5000, value=200, step=50, key="cov_count")
    with colsA[1]:
        cov_density = st.slider("Bit density", min_value=0.05, max_value=0.95, value=0.50, step=0.05, key="cov_density")
    with colsA[2]:
        seed_default = _default_sampler_seed(st.session_state.get("_inputs_block", {}))
        cov_seed = st.text_input("Seed (hex/str)", value=seed_default, key="cov_seed")
    with colsA[3]:
        out_path = st.text_input("CSV path", value=str(COVERAGE_DEFAULT_PATH), key="cov_out_path")

    # Run button
    if st.button("Run coverage sample", key="cov_run"):
        try:
            # set up RNG
            rng = random.Random()
            rng.seed(cov_seed)

            # generate samples and signatures
            sig_counts = {}
            rows = []
            registry = st.session_state.get("district_signature_registry", set())
            for i in range(int(cov_count)):
                d3_samp = _sample_binary_matrix(n2, n3, rng=rng, density=cov_density)
                sig = _signature_from_d3(d3_samp)
                key = _sig_key(sig)
                sig_counts[key] = sig_counts.get(key, 0) + 1
                in_reg = (key in registry)
                rows.append([i+1, sig["rank"], sig["ker_dim"], sig["lane_pattern"], int(in_reg)])

            # summarize
            total = len(rows)
            in_known = sum(r[-1] for r in rows)
            pct = (100.0 * in_known / total) if total else 0.0

            st.success(f"Coverage complete: {in_known}/{total} ({pct:.1f}%) in registry.")
            # Show top signatures
            st.write("Top signatures:")
            top = sorted(sig_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
            for key, cnt in top:
                st.caption(f"{key} → {cnt}")

            # persist CSV (schema/versioned + meta comments)
            meta = [
                f"schema_version={COVERAGE_SCHEMA_VERSION}",
                f"saved_at={_utc_iso()}",
                f"seed={cov_seed}",
                f"policy_tag={st.session_state.get('overlap_policy_label', st.session_state.get('run_ctx',{}).get('policy_tag',''))}",
                f"boundaries_hash={st.session_state.get('_inputs_block',{}).get('boundaries_hash','')}",
                f"C_hash={st.session_state.get('_inputs_block',{}).get('C_hash','')}",
                f"H_hash={st.session_state.get('_inputs_block',{}).get('H_hash','')}",
                f"U_hash={st.session_state.get('_inputs_block',{}).get('U_hash','')}",
                f"n2={n2}",
                f"n3={n3}",
            ]
            header = ["idx", "rank", "ker_dim", "lane_pattern", "in_registry"]
            _atomic_write_csv(Path(out_path), header, rows, meta)
            st.success(f"Saved coverage → {out_path}")

            # Quick viewer
            if len(rows) <= 1000:  # avoid rendering huge tables
                import pandas as _pd  # streamlit has pandas runtime; safe to import
                df = _pd.DataFrame(rows, columns=header)
                st.dataframe(df, use_container_width=True, height=320)

        except Exception as e:
            st.error(f"Coverage sampling failed: {e}")



# ── UI: Parity Runner (uses current fixture for both sides by default) --------
with st.expander("Parity Suite (reuse active policy)"):
    st.caption("Run parity checks that mirror the ACTIVE policy (AUTO/FILE). Supply pairs or test the current fixture as a self-check.")
    cfg_proj = _mirror_active_projected_cfg()

    # Simple self-check using the currently loaded fixture
    if st.button("Run self-parity (current fixture vs itself)"):
        fixture = {"boundaries": boundaries, "cmap": cmap, "H": st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}}), "shapes": shapes}
        row = _parity_check_pair("SELF", fixture, fixture, cfg_proj)
        rpt = _write_parity_report([row])
        st.success("Parity report written.")
        with open(rpt, "rb") as f:
            st.download_button("⬇️ Download parity_report.json", f, file_name="parity_report.json")

    st.divider()
    st.caption("Advanced: supply custom pairs via session (st.session_state['parity_pairs']). Each item must be a dict with keys: 'label', 'left', 'right', where each side has {boundaries,cmap,H,shapes} objects parsed with io.*.")
    if st.button("Run parity on provided pairs"):
        pairs = st.session_state.get("parity_pairs") or []
        if not pairs:
            st.warning("No 'parity_pairs' found in session. Add them programmatically, then click again.")
        else:
            cfg_proj = _mirror_active_projected_cfg()
            results = []
            for p in pairs:
                try:
                    label = p["label"]
                    left  = p["left"]
                    right = p["right"]
                except Exception:
                    st.error("Malformed parity pair. Expected keys: 'label','left','right'.")
                    continue
                rows = _parity_check_pair(label, left, right, cfg_proj)
                results.append(rows)
            rpt = _write_parity_report(results)
            st.success(f"Parity report written for {len(results)} pair(s).")
            with open(rpt, "rb") as f:
                st.download_button("⬇️ Download parity_report.json", f, file_name="parity_report.json")



   # ── identity (authoritative)
_ss = st.session_state
_di = _ss.get("_district_info", {}) or {}
district_id = _di.get("district_id") or _ss.get("district_id", "UNKNOWN")

# Safe H handle: prefer the one persisted by Run Overlap, else fall back to the local upload,
# else use an empty cmap so hashing/serialization still works.
H_used = _ss.get("overlap_H") \
          or (H_local if "H_local" in locals() and H_local is not None else io.parse_cmap({"blocks": {}}))

identity_block = {
    "district_id": district_id,
    "run_id": hashes.run_id(
        hashes.bundle_content_hash([
            ("d",   boundaries.dict() if hasattr(boundaries, "dict") else {}),
            ("C",   cmap.dict()       if hasattr(cmap, "dict")       else {}),
            ("H",   H_used.dict()     if hasattr(H_used, "dict")     else {}),
            ("cfg", cfg_active),
        ]),
        hashes.timestamp_iso_lisbon()
    ),
    "timestamp": hashes.timestamp_iso_lisbon(),
    "app_version": getattr(hashes, "APP_VERSION", "v0.1-core"),
    "field": "GF(2)",
}


# ── inputs block (sole source of hashes + filenames + dims)
def _fname(k, default): 
    v = _ss.get(k, ""); 
    return v if isinstance(v, str) and v.strip() else default
inputs_block = {}
inputs_block["filenames"] = {
    "boundaries": _fname("fname_boundaries", "boundaries.json"),
    "C":          _fname("fname_cmap", "cmap.json"),
    "H":          _fname("fname_h", "H.json"),
    "U":          _fname("fname_shapes", "shapes.json"),
}
d3_now = (boundaries.blocks.__root__.get("3") or [])
C3_now = (cmap.blocks.__root__.get("3") or [])
inputs_block["dims"] = {
    "n3": (len(C3_now) if C3_now else len(d3_now[0]) if (d3_now and d3_now[0]) else 0),
    "n2": (len(cmap.blocks.__root__.get("2") or [])),
}
shapes_payload = shapes.dict() if hasattr(shapes, "dict") else (shapes or {})
inputs_block["boundaries_hash"] = _stable_hash(boundaries.dict() if hasattr(boundaries, "dict") else {})
inputs_block["C_hash"]          = _stable_hash(cmap.dict() if hasattr(cmap, "dict") else {})
inputs_block["H_hash"]          = _stable_hash(H_used.dict() if hasattr(H_used, "dict") else {})
inputs_block["U_hash"]          = _stable_hash(shapes_payload)
inputs_block["shapes_hash"]     = inputs_block["U_hash"]
if run_ctx.get("mode") == "projected(file)":
    inputs_block["filenames"]["projector"] = run_ctx.get("projector_filename","")

# ── diagnostics
lane_mask = run_ctx.get("lane_mask_k3", [])
try:
    H2 = (H_used.blocks.__root__.get("2") or [])
    C3 = (cmap.blocks.__root__.get("3") or [])
    H2d3 = mul(H2, run_ctx.get("d3", [])) if (H2 and run_ctx.get("d3")) else []
    C3pI3 = _xor_mat(C3, _eye(len(C3))) if C3 else []
    lane_idx = [j for j, m in enumerate(lane_mask) if m]
    def _mask(vec, idx): return [vec[j] for j in idx] if (vec and idx) else []
    diagnostics_block = {
        "lane_mask_k3": lane_mask,
        "lane_vec_H2d3": _mask(_bottom_row(H2d3), lane_idx),
        "lane_vec_C3plusI3": _mask(_bottom_row(C3pI3), lane_idx),
    }
except Exception:
    diagnostics_block = {"lane_mask_k3": lane_mask, "lane_vec_H2d3": [], "lane_vec_C3plusI3": []}

    # ── signatures
    def _gf2_rank(M):
        if not M or not M[0]: return 0
        A = [r[:] for r in M]; m, n = len(A), len(A[0]); r = c = 0
        while r < m and c < n:
            piv = next((i for i in range(r, m) if A[i][c] & 1), None)
            if piv is None: c += 1; continue
            if piv != r: A[r], A[piv] = A[piv], A[r]
            for i in range(m):
                if i != r and (A[i][c] & 1):
                    A[i] = [(A[i][j] ^ A[r][j]) & 1 for j in range(n)]
            r += 1; c += 1
        return r
    d3M = run_ctx.get("d3", [])
    rank_d3 = _gf2_rank(d3M) if d3M else 0
    ncols_d3 = len(d3M[0]) if (d3M and d3M[0]) else 0
    ker_dim_d3 = max(ncols_d3 - rank_d3, 0)
    lane_pattern = "".join("1" if int(x) else "0" for x in (lane_mask or []))
    C3 = (cmap.blocks.__root__.get("3") or [])
    C3pI3 = _xor_mat(C3, _eye(len(C3))) if C3 else []
    lane_idxs = [j for j, m in enumerate(lane_mask or []) if m]
    def _col_support_pattern(M, cols):
        if not M or not cols: return ""
        return "".join("1" if any((row[j] & 1) for row in M) else "0" for j in cols)
    d_signature = {"rank": rank_d3, "ker_dim": ker_dim_d3, "lane_pattern": lane_pattern}
    fixture_signature = {"lane": _col_support_pattern(C3pI3, lane_idxs)}
    sig_block = {"d_signature": d_signature, "fixture_signature": fixture_signature, "echo_context": None}

    # ── checks & policy block
    is_strict = (run_ctx.get("mode") == "strict")
    checks_block = {**out, "grid": True, "fence": True, "ker_guard": ("enforced" if is_strict else "off")}
    residual_tags = _ss.get("residual_tags", {})
    policy_block = {
        "label": run_ctx.get("policy_tag", policy_label),
        "policy_tag": run_ctx.get("policy_tag", policy_label),
        "enabled_layers": cfg_active.get("enabled_layers", []),
        "modes": cfg_active.get("modes", {}),
        "source": cfg_active.get("source", {}),
    }
    if run_ctx.get("projector_hash") is not None:
        policy_block["projector_hash"] = run_ctx.get("projector_hash","")
    if run_ctx.get("projector_filename"):
        policy_block["projector_filename"] = run_ctx.get("projector_filename","")
    if run_ctx.get("projector_consistent_with_d") is not None:
        policy_block["projector_consistent_with_d"] = bool(run_ctx.get("projector_consistent_with_d"))

# ── cert payload (core)
# (Optionally ensure schema/app/python tags are present)
cert_payload.setdefault("schema_version", SCHEMA_VERS if "SCHEMA_VERS" in globals() else LAB_SCHEMA_VERSION)
cert_payload.setdefault("app_version", APP_VERSION)
cert_payload.setdefault("python_version", PY_VERSION_STR)

cert_path: str | None = None
full_hash: str = ""

try:
    # Preferred API: returns (path, full_hash)
    result = export_mod.write_cert_json(cert_payload)
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        cert_path, full_hash = result[0], result[1]
    else:
        # Some older writers return just the path
        cert_path = result
        # fall back to content_hash already embedded in payload
        full_hash = (cert_payload.get("integrity", {}) or {}).get("content_hash", "") or ""
except Exception as e:
    st.error(f"Cert write failed: {e}")

# Only touch session if we actually produced a path
if cert_path:
    st.session_state["last_cert_path"] = cert_path
    st.session_state["cert_payload"] = cert_payload  # handy for Gallery/Witness rows
    st.success(f"Cert written: `{cert_path}`" + (f" · {full_hash[:12]}…" if full_hash else ""))
else:
    st.warning("No cert file was produced. Check the error above and try again.")


# ── A/B embed (fresh only if inputs_sig matches)
ab_ctx = _ss.get("ab_compare", {}) or {}
inputs_sig = [
    inputs_block["boundaries_hash"],
    inputs_block["C_hash"],
    inputs_block["H_hash"],
    inputs_block["U_hash"],
    inputs_block["shapes_hash"],
]
if ab_ctx.get("inputs_sig") == inputs_sig:
    strict_ctx    = ab_ctx.get("strict", {})
    projected_ctx = ab_ctx.get("projected", {})
    def _pass_vec_from(d):
        return [int(d.get("2", {}).get("eq", False)), int(d.get("3", {}).get("eq", False))]
    cert_payload["policy"]["strict_snapshot"] = {
        "policy_tag": strict_ctx.get("label", "strict"),
        "ker_guard":  "enforced",
        "inputs": {
            "filenames": inputs_block["filenames"],
            "boundaries": {
                "filename": inputs_block["filenames"]["boundaries"],
                "hash":     inputs_block["boundaries_hash"],
                "district_id": identity_block["district_id"],
                "lane_mask_k3": run_ctx.get("lane_mask_k3", []),
                "d3_rows": len(run_ctx.get("d3", [])),
                "d3_cols": (len(run_ctx.get("d3", [])[0]) if run_ctx.get("d3") else 0),
            },
            "U_filename": inputs_block["filenames"]["U"],
            "C_filename": inputs_block["filenames"]["C"],
            "H_filename": inputs_block["filenames"]["H"],
        },
        "lane_mask_k3": run_ctx.get("lane_mask_k3", []),
        "lane_vec_H2d3": strict_ctx.get("lane_vec_H2d3", diagnostics_block.get("lane_vec_H2d3", [])),
        "lane_vec_C3plusI3": strict_ctx.get("lane_vec_C3plusI3", diagnostics_block.get("lane_vec_C3plusI3", [])),
        "pass_vec": _pass_vec_from(strict_ctx.get("out", {})),
        "residual_tag": residual_tags.get("strict", "none"),
        "out": strict_ctx.get("out", {}),
    }
    proj_hash_ab = projected_ctx.get("projector_hash", run_ctx.get("projector_hash",""))
    cert_payload["policy"]["projected_snapshot"] = {
        "policy_tag": run_ctx.get("policy_tag", policy_label_from_cfg(cfg_projected_base())),
        "ker_guard":  "off",
        "projector_hash": proj_hash_ab,
        "inputs": {
            "filenames": inputs_block["filenames"],
            "boundaries": {
                "filename": inputs_block["filenames"]["boundaries"],
                "hash":     inputs_block["boundaries_hash"],
                "district_id": identity_block["district_id"],
                "lane_mask_k3": run_ctx.get("lane_mask_k3", []),
                "d3_rows": len(run_ctx.get("d3", [])),
                "d3_cols": (len(run_ctx.get("d3", [])[0]) if run_ctx.get("d3") else 0),
                **({"projector_filename": run_ctx.get("projector_filename","")} if run_ctx.get("mode")=="projected(file)" else {}),
            },
            "U_filename": inputs_block["filenames"]["U"],
            "C_filename": inputs_block["filenames"]["C"],
            "H_filename": inputs_block["filenames"]["H"],
        },
        "lane_mask_k3": run_ctx.get("lane_mask_k3", []),
        "lane_vec_H2d3": projected_ctx.get("lane_vec_H2d3", diagnostics_block.get("lane_vec_H2d3", [])),
        "lane_vec_C3plusI3": projected_ctx.get("lane_vec_C3plusI3", diagnostics_block.get("lane_vec_C3plusI3", [])),
        "pass_vec": _pass_vec_from(projected_ctx.get("out", {})),
        "residual_tag": residual_tags.get("projected", "none"),
        "out": projected_ctx.get("out", {}),
        **({"projector_consistent_with_d": bool(run_ctx.get("projector_consistent_with_d"))}
           if run_ctx.get("mode")=="projected(file)" else {}),
    }
    cert_payload["ab_pair_tag"] = ab_ctx.get("pair_tag", "")

    # ── artifacts, promotion, integrity
    cert_payload["artifact_hashes"] = {
        "boundaries_hash": inputs_block["boundaries_hash"],
        "C_hash":          inputs_block["C_hash"],
        "H_hash":          inputs_block["H_hash"],
        "U_hash":          inputs_block["U_hash"],
        "projector_hash":  policy_block.get("projector_hash",""),
    }
    grid_ok  = bool(out.get("grid", True))
    fence_ok = bool(out.get("fence", True))
    k3_ok    = bool(out.get("3", {}).get("eq", False))
    k2_ok    = bool(out.get("2", {}).get("eq", False))
    mode_now = run_ctx.get("mode")
    eligible, target = False, None
    if mode_now == "strict" and all([grid_ok, fence_ok, k3_ok, k2_ok]):
        eligible, target = True, "strict_anchor"
    elif mode_now in ("projected(auto)", "projected(file)") and all([grid_ok, fence_ok, k3_ok]) and _ss.get("residual_tags", {}).get("projected","none") == "none":
        eligible, target = True, "projected_exemplar"
    cert_payload["promotion"] = {"eligible_for_promotion": eligible, "promotion_target": target, "notes": ""}
    cert_payload.setdefault("integrity", {})
    cert_payload["integrity"]["content_hash"] = _stable_hash(cert_payload)

    # ── write cert
    cert_path, full_hash = export_mod.write_cert_json(cert_payload)
    st.success(f"Cert written: `{cert_path}`")

    # ── gallery row
    try:
        key = (
            cert_payload["district_id"],
            inputs_block.get("boundaries_hash",""),
            inputs_block.get("U_hash",""),
            inputs_block.get("C_hash",""),
            tuple(diagnostics_block.get("lane_vec_H2d3", [])),
            policy_block.get("policy_tag",""),
        )
        row = {
            "district_id":     key[0],
            "boundaries_hash": key[1],
            "U_hash":          key[2],
            "suppC_hash":      key[3],
            "lane_vec_H2d3":   key[4],
            "policy_mode":     key[5],
            "cert_path":       cert_path,
            "content_hash":    cert_payload.get("integrity", {}).get("content_hash", ""),
        }
        if cert_payload["district_id"] == "UNKNOWN":
            st.warning("gallery insert skipped: district_id is UNKNOWN.")
        else:
            res = export_mod.write_gallery_row(row, key, path="gallery.csv")
            st.toast("gallery: added exemplar row" if res == "written" else "gallery: duplicate skipped")
    except Exception as e:
        st.warning(f"gallery dedupe failed: {e}")

    # ── bundle download
    try:
        _ab_ctx = _ss.get("ab_compare", {}) or {}
        _projected_ctx = _ab_ctx.get("projected", {}) or {}
        _policy_block_for_bundle = dict(policy_block)
        if "policy" in cert_payload and "projected_snapshot" in cert_payload["policy"]:
            _policy_block_for_bundle["ab_policies"] = {
                "strict":    cert_payload["policy"]["strict_snapshot"]["policy_tag"],
                "projected": cert_payload["policy"]["projected_snapshot"]["policy_tag"],
            }
            _policy_block_for_bundle["ab_projector_hash"] = (
                _projected_ctx.get("projector_hash")
                or cert_payload["policy"]["projected_snapshot"].get("projector_hash","")
            )
        proj_hash_bundle = (
            _projected_ctx.get("projector_hash")
            or cert_payload.get("policy", {}).get("projected_snapshot", {}).get("projector_hash","")
            or _policy_block_for_bundle.get("projector_hash","")
        )
        if proj_hash_bundle and not _policy_block_for_bundle.get("projector_hash"):
            _policy_block_for_bundle["projector_hash"] = proj_hash_bundle
        _policy_block_for_bundle["district_id"]         = cert_payload["district_id"]
        _policy_block_for_bundle["boundaries_hash"]     = inputs_block.get("boundaries_hash","")
        _policy_block_for_bundle["boundaries_filename"] = inputs_block["filenames"]["boundaries"]
        _policy_block_for_bundle["U_filename"]          = inputs_block["filenames"]["U"]
        tag = policy_label.replace(" ", "_")
        if "policy" in cert_payload and "strict_snapshot" in cert_payload["policy"]:
            tag = f"{tag}__withAB"
        bundle_name = f"overlap_bundle__{cert_payload['district_id']}__{tag}__{full_hash[:12]}.zip"
        zip_path = export_mod.build_overlap_bundle(
            boundaries=boundaries,
            cmap=cmap,
            H=H_used,
            shapes=shapes_payload,
            policy_block=_policy_block_for_bundle,
            cert_path=cert_path,
            out_zip=bundle_name,
        )
        with open(zip_path, "rb") as fz:
            st.download_button("⬇️ Download bundle (.zip)", data=fz, file_name=bundle_name,
                               mime="application/zip", key="dl_overlap_bundle")
    except Exception as e:
        st.error(f"Could not build download bundle: {e}")
        st.session_state["last_bundle_path"] = zip_path


    # ── promotion (optional)
    _pass_vec = [int(out.get("2", {}).get("eq", False)), int(out.get("3", {}).get("eq", False))]
    if cert_payload["promotion"]["eligible_for_promotion"]:
        st.success(f"Green — eligible for promotion ({cert_payload['promotion']['promotion_target']}).")
        flip_to_file = st.checkbox("After promotion, switch to FILE-backed projector", value=True, key="flip_to_file_k3")
        keep_auto    = st.checkbox("…or keep AUTO (don’t lock now)", value=False, key="keep_auto_k3")
        if st.button("Promote & Freeze Projector", key="promote_k3"):
            try:
                d3_now = (boundaries.blocks.__root__.get("3") or [])
                if not d3_now or not d3_now[0]:
                    st.error("No d3; cannot freeze projector."); st.stop()
                n3 = len(d3_now[0])
                lane = [1 if any(row[j] & 1 for row in d3_now) else 0 for j in range(n3)]
                P_used = [[1 if (i==j and lane[i]) else 0 for j in range(n3)] for i in range(n3)]
                pj_path = (cfg_file.get("projector_files", {}) or {}).get("3", f"projectors/projector_{cert_payload['district_id']}.json")
                os.makedirs(os.path.dirname(pj_path), exist_ok=True)
                with open(pj_path, "w") as _f:
                    _json.dump({"name": "Π3 freeze (lane-mask of current d3)", "blocks": {"3": P_used}}, _f, indent=2)
                pj_hash = _stable_hash({"blocks":{"3": P_used}})
                st.info(f"Projector frozen → {pj_path} (hash={pj_hash[:12]}…)")

                if flip_to_file and not keep_auto:
                    cfg_file.setdefault("source", {})["3"] = "file"
                    cfg_file.setdefault("projector_files", {})["3"] = pj_path
                else:
                    cfg_file.setdefault("source", {})["3"] = "auto"
                    cfg_file.get("projector_files", {}).pop("3", None)
                with open("projection_config.json", "w") as _f:
                    _json.dump(cfg_file, _f, indent=2)

                # best-effort registry note
                try:
                    import time as _time
                    export_mod.write_registry_row(
                        fix_id=f"overlap-{int(_time.time())}",
                        pass_vector=_pass_vec,
                        policy=policy_label,
                        hash_d=inputs_block["boundaries_hash"],
                        hash_U=inputs_block["U_hash"],
                        hash_suppC=inputs_block["C_hash"],
                        hash_suppH=inputs_block["H_hash"],
                        notes=f"proj_hash={pj_hash}",
                    )
                    st.success("Registry updated with projector hash.")
                except Exception as ee:
                    st.warning(f"Registry note failed (non-fatal): {ee}")
            except Exception as e:
                st.error(f"Promotion failed: {e}")

    # ── A/B compare (strict vs projected)
    st.markdown("### A/B: strict vs projected")
    _ab_ctx_existing = _ss.get("ab_compare")
    if _ab_ctx_existing:
        s_ok = bool(_ab_ctx_existing.get("strict", {}).get("out", {}).get("3", {}).get("eq", False))
        p_ok = bool(_ab_ctx_existing.get("projected", {}).get("out", {}).get("3", {}).get("eq", False))
        st.caption(f"A/B: strict={'✅' if s_ok else '❌'} · projected={'✅' if p_ok else '❌'} · pair: {_ab_ctx_existing.get('pair_tag','')}")
    else:
        st.caption("A/B: (no snapshot yet)")

    def _prov_hash(cfg, lane_mask_k3, district_id):
        try:
            return projector_provenance_hash(cfg=cfg, lane_mask_k3=lane_mask_k3, district_id=district_id)
        except Exception:
            try:
                return projector_provenance_hash(cfg=cfg)
            except Exception:
                return ""

    if st.button("Run A/B compare (strict vs projected)", key="run_ab_overlap"):
        # strict leg
        out_strict   = overlap_gate.overlap_check(boundaries, cmap, H_used)
        label_strict = policy_label_from_cfg(cfg_strict())

        # projected leg mirrors active (auto/file)
        _cfg_proj_ab = cfg_active if cfg_active.get("enabled_layers") else cfg_projected_base()
        if not _cfg_proj_ab.get("enabled_layers"):
            disk = projector.load_projection_config("projection_config.json")
            if disk.get("source", {}).get("3") in ("file","auto"):
                _cfg_proj_ab["source"]["3"] = disk["source"]["3"]
            if disk.get("projector_files", {}).get("3"):
                _cfg_proj_ab.setdefault("projector_files", {})["3"] = disk["projector_files"]["3"]

        # validate FILE leg (fail-fast)
        try:
            _P_ab, _meta_ab = projector_choose_active(_cfg_proj_ab, boundaries)
        except ValueError as e:
            st.error(f"A/B projected(file) invalid: {e}")
            st.stop()

        out_proj = overlap_gate.overlap_check(
            boundaries, cmap, H_used,
            projection_config=_cfg_proj_ab,
            projector_cache=None,
        )
        label_proj = policy_label_from_cfg(_cfg_proj_ab)

        # lane vectors (fresh)
        d3  = run_ctx.get("d3", [])
        H2  = (H_used.blocks.__root__.get("2") or [])
        C3  = (cmap.blocks.__root__.get("3") or [])
        lane_idx  = [j for j, m in enumerate(run_ctx.get("lane_mask_k3", [])) if m]
        H2d3 = mul(H2, d3) if (H2 and d3) else []
        C3pI3 = _xor_mat(C3, _eye(len(C3))) if C3 else []
        def _mask(vec, idx): return [vec[j] for j in idx] if (vec and idx) else []
        strict_lane_vec_H2d3, strict_lane_vec_C3pI3 = _mask(_bottom_row(H2d3), lane_idx), _mask(_bottom_row(C3pI3), lane_idx)
        proj_lane_vec_H2d3,   proj_lane_vec_C3pI3   = strict_lane_vec_H2d3[:], strict_lane_vec_C3pI3[:]

        # provenance
        proj_hash_prov = _prov_hash(_cfg_proj_ab, run_ctx.get("lane_mask_k3", []), district_id)

        # persist A/B snapshot
        _inputs_sig = [inputs_block["boundaries_hash"], inputs_block["C_hash"], inputs_block["H_hash"], inputs_block["U_hash"], inputs_block["shapes_hash"]]
        st.session_state["ab_compare"] = {
            "pair_tag": f"{label_strict}__VS__{label_proj}",
            "inputs_sig": _inputs_sig,
            "lane_mask_k3": run_ctx.get("lane_mask_k3", []),
            "strict": {
                "label": label_strict,
                "cfg":   cfg_strict(),
                "out":   out_strict,
                "ker_guard": "enforced",
                "lane_vec_H2d3":     strict_lane_vec_H2d3,
                "lane_vec_C3plusI3": strict_lane_vec_C3pI3,
                "projector_hash": "",
                "pass_vec": [
                    int(out_strict.get("2", {}).get("eq", False)),
                    int(out_strict.get("3", {}).get("eq", False)),
                ],
            },
            "projected": {
                "label": label_proj,
                "cfg":   _cfg_proj_ab,
                "out":   out_proj,
                "ker_guard": "off",
                "projector_hash": proj_hash_prov,
                "lane_vec_H2d3":     proj_lane_vec_H2d3,
                "lane_vec_C3plusI3": proj_lane_vec_C3pI3,
                "pass_vec": [
                    int(out_proj.get("2", {}).get("eq", False)),
                    int(out_proj.get("3", {}).get("eq", False)),
                ],
                "projector_filename": _meta_ab.get("projector_filename",""),
                "projector_file_hash": _meta_ab.get("projector_hash",""),
                "projector_consistent_with_d": _meta_ab.get("projector_consistent_with_d", None),
            },
        }
        s_ok = bool(out_strict.get("3", {}).get("eq", False))
        p_ok = bool(out_proj.get("3", {}).get("eq", False))
        st.success(f"A/B: strict={'GREEN' if s_ok else 'RED'} · projected={'GREEN' if p_ok else 'RED'}")

# ───────────────────────────── Export Inputs (+manifest) ─────────────────────────────
import os, io, json as _json, tempfile, zipfile, platform
from pathlib import Path
from datetime import datetime, timezone

EXPORT_SCHEMA_VERSION = "1.0.0"
EXPORT_DIR = Path("bundles")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _atomic_write(path: Path, bytes_blob: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as tmp:
        tmp.write(bytes_blob)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def _json_bytes(obj) -> bytes:
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _maybe(obj, default=None):
    return obj if obj is not None else default

with st.expander("Export Inputs (reproducible zip)"):
    st.caption("Package the exact **inputs** + an audit **manifest.json** so anyone can reproduce this run. "
               "No recomputation; reads filenames/hashes from the Inputs SSOT and projector info from RunContext.")
    # derive identifiers
    di = st.session_state.get("_district_info", {}) or {}
    inputs = st.session_state.get("_inputs_block", {}) or {}
    run_ctx = st.session_state.get("run_ctx", {}) or {}
    policy_label_active = st.session_state.get("overlap_policy_label") or run_ctx.get("policy_tag") or policy_label_from_cfg(cfg_active)

    district_id   = di.get("district_id", "UNKNOWN")
    bhash_short   = (inputs.get("boundaries_hash","") or "")[:8]
    pj_short      = (run_ctx.get("projector_hash","") or "")[:8]
    ts_short      = _utc_iso().replace(":", "").replace("-", "")[:15]  # tidy timestamp fragment

    default_zip = EXPORT_DIR / f"inputs__{district_id}__{policy_label_active.replace(' ','_')}__b{bhash_short}__p{pj_short}__{ts_short}.zip"
    out_zip_path = st.text_input("Output .zip path", value=str(default_zip), key="export_inputs_zip")

    # show what will be included
    include_proj = (run_ctx.get("mode") == "projected(file)") and bool(run_ctx.get("projector_filename"))
    pj_path = run_ctx.get("projector_filename") or ""
    st.caption(f"Includes: boundaries.json, shapes.json, cmap.json"
               f"{', H.json' if 'H' in globals() and H else ''}"
               f"{', ' + os.path.basename(pj_path) if include_proj else ''}, manifest.json")

    if st.button("Build Inputs Zip", key="btn_export_inputs"):
        try:
            # gather JSON payloads (from live parsed objects)
            boundaries_json = _maybe(boundaries.dict(), {"blocks": {}})
            shapes_json     = _maybe(shapes.dict(),     {"blocks": {}})
            cmap_json       = _maybe(cmap.dict(),       {"blocks": {}})
            H_json          = (_maybe(H.dict(), {"blocks": {}}) if ('H' in globals() and H) else None)

            # stable content hash for bundle (inputs + projector content hash if file)
            named = [("boundaries", boundaries_json), ("shapes", shapes_json), ("cmap", cmap_json)]
            if H_json is not None:
                named.append(("H", H_json))
            # projector: prefer exact file bytes if file-mode, else hash AUTO diag for provenance
            projector_hash = run_ctx.get("projector_hash", "") or ""
            if include_proj and os.path.exists(pj_path):
                try:
                    with open(pj_path, "rb") as fp:
                        pj_bytes = fp.read()
                    # include projector raw JSON in content hash by its *parsed* structure for stability
                    try:
                        pj_obj = _json.loads(pj_bytes.decode("utf-8"))
                    except Exception:
                        pj_obj = {"raw_bytes_sha256": hashlib.sha256(pj_bytes).hexdigest()}
                    named.append(("projector", pj_obj))
                except Exception:
                    pass

            content_hash = hashes.bundle_content_hash(named)
            run_id = hashes.run_id(content_hash, hashes.timestamp_iso_lisbon())

            # manifest (schema/app/python/run/inputs/projector)
            manifest = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "kind": "inputs_bundle",
                "created_at": _utc_iso(),
                "app_version": APP_VERSION,
                "python_version": platform.python_version(),
                "run": {
                    "policy_tag": policy_label_active,
                    "run_id": run_id,
                    "district_id": district_id,
                },
                "inputs": {
                    "filenames": {
                        "boundaries": inputs.get("boundaries_filename", "boundaries.json"),
                        "shapes":     inputs.get("shapes_filename",     "shapes.json"),
                        "cmap":       inputs.get("cmap_filename",       "cmap.json"),
                        "H":          inputs.get("H_filename",          "H.json"),
                        **({"projector": os.path.basename(pj_path)} if include_proj else {}),
                    },
                    "hashes": {
                        "boundaries_hash": inputs.get("boundaries_hash",""),
                        "C_hash":          inputs.get("C_hash",""),
                        "H_hash":          inputs.get("H_hash",""),
                        "U_hash":          inputs.get("U_hash",""),
                        "shapes_hash":     inputs.get("shapes_hash",""),
                        **({"projector_hash": projector_hash} if projector_hash else {}),
                    },
                    "dims": {
                        "n2": di.get("d2_cols", None) or (len(boundaries.blocks.__root__.get("2", [[]])[0]) if (hasattr(boundaries, "blocks")) else None),
                        "n3": di.get("d3_cols", None) or (len(boundaries.blocks.__root__.get("3", [[]])[0]) if (hasattr(boundaries, "blocks")) else None),
                    },
                },
                "projector": {
                    "mode": run_ctx.get("mode") or ("strict" if not st.session_state.get("overlap_cfg",{}).get("enabled_layers") else "projected(auto)"),
                    "projector_filename": (os.path.basename(pj_path) if include_proj else ""),
                    "projector_hash": projector_hash,
                    "lane_mask_k3": run_ctx.get("lane_mask_k3", []),
                },
                "content_hash": content_hash,
                "notes": "Bundle contains the *inputs only* and a manifest. No results are included.",
            }

            # build the zip in-memory
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("manifest.json", _json_bytes(manifest))
                z.writestr("boundaries.json", _json_bytes(boundaries_json))
                z.writestr("shapes.json",     _json_bytes(shapes_json))
                z.writestr("cmap.json",       _json_bytes(cmap_json))
                if H_json is not None:
                    z.writestr("H.json", _json_bytes(H_json))
                if include_proj and os.path.exists(pj_path):
                    # store projector under projectors/
                    pj_name = os.path.basename(pj_path)
                    with open(pj_path, "rb") as fp:
                        z.writestr(f"projectors/{pj_name}", fp.read())

            # atomic write to disk
            out_path = Path(out_zip_path)
            _atomic_write(out_path, mem.getvalue())
            st.success(f"Exported inputs → {out_path}")

            # offer download
            st.download_button("⬇️ Download inputs zip", data=mem.getvalue(),
                               file_name=out_path.name, mime="application/zip",
                               key="dl_inputs_zip")

            # small echo (run stamp)
            stamp = " | ".join(filter(None, [
                policy_label_active,
                f"n3={di.get('d3_cols','?')}",
                f"b={inputs.get('boundaries_hash','')[:8]}",
                f"C={inputs.get('C_hash','')[:8]}",
                f"H={inputs.get('H_hash','')[:8]}",
                f"U={inputs.get('U_hash','')[:8]}",
                (f"P={projector_hash[:8]}" if projector_hash else "")
            ]))
            st.caption(f"run stamp: {stamp}")

        except Exception as e:
            st.error(f"Export failed: {e}")

# ---- Downloads drawer: always visible if files exist ------------------------
with st.expander("Downloads"):
    cp = st.session_state.get("last_cert_path")
    if cp and os.path.exists(cp):
        with open(cp, "rb") as f:
            st.download_button("⬇️ Download cert.json", f, file_name=os.path.basename(cp), key="dl_cert_latest")
    else:
        st.caption("No cert yet.")
    zp = st.session_state.get("last_bundle_path")
    if zp and os.path.exists(zp):
        with open(zp, "rb") as f:
            st.download_button("⬇️ Download overlap bundle (.zip)", f, file_name=os.path.basename(zp), key="dl_bundle_latest")
    else:
        st.caption("No bundle yet.")


# ─────────────────────────────── Self-tests (red gate) ───────────────────────────────
import os, json as _json
from datetime import datetime, timezone

SELFTESTS_SCHEMA_VERSION = "1.0.0"

def _utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _short(h): 
    return (h or "")[:8]

def _safe_get_inputs_sig(inputs: dict):
    return [
        inputs.get("boundaries_hash",""),
        inputs.get("C_hash",""),
        inputs.get("H_hash",""),
        inputs.get("U_hash",""),
        inputs.get("shapes_hash",""),
    ]

def _diag_vec(P):
    try:
        n = min(len(P), len(P[0]))
        return [int(P[i][i] & 1) for i in range(n)]
    except Exception:
        return []

with st.expander("Self-tests (plumbing health check)"):
    st.caption("Quick guardrail suite. If *any* test fails, you’ll see a red banner. "
               "Fix plumbing first; then resume math.")
    if st.button("Run self-tests", key="btn_selftests"):
        try:
            failures = []
            warnings = []
            notes    = []

            # --- SSOT snapshots
            di      = st.session_state.get("_district_info", {}) or {}
            inputs  = st.session_state.get("_inputs_block", {}) or {}
            run_ctx = st.session_state.get("run_ctx", {}) or {}
            ab_ctx  = st.session_state.get("ab_compare", {}) or {}

            # 0) identity echo / stamp
            stamp = " | ".join(filter(None, [
                (run_ctx.get("policy_tag") or st.session_state.get("overlap_policy_label") or "strict"),
                f"n3={di.get('d3_cols','?')}",
                f"b={_short(inputs.get('boundaries_hash',''))}",
                f"C={_short(inputs.get('C_hash',''))}",
                f"H={_short(inputs.get('H_hash',''))}",
                f"U={_short(inputs.get('U_hash',''))}",
                ("P="+_short(run_ctx.get("projector_hash","")) if run_ctx.get("projector_hash") else "")
            ]))
            st.caption(f"run stamp: {stamp}")

            # 1) Hash coherence (cert/consumers must copy from Inputs SSOT)
            #    Here we just assert that district_info mirrors the inputs' boundaries hash.
            bh_inputs = inputs.get("boundaries_hash","")
            bh_di     = di.get("boundaries_hash","")
            if bh_inputs and bh_di and (bh_inputs != bh_di):
                failures.append(f"Hash coherence: inputs.boundaries_hash ({_short(bh_inputs)}) "
                                f"!= district_info.boundaries_hash ({_short(bh_di)}).")
            elif not bh_inputs:
                warnings.append("Hash coherence: inputs.boundaries_hash missing (did you load Boundaries?).")

            # 2) Policy invariants
            policy = (run_ctx.get("policy_tag") or "strict")
            mode   = (run_ctx.get("mode") or ("strict" if policy == "strict" else "projected(auto)"))
            pj_fn  = run_ctx.get("projector_filename") or ""
            pj_h   = run_ctx.get("projector_hash") or ""
            pj_cons= run_ctx.get("projector_consistent_with_d")

            if policy == "strict":
                # projector fields must be None/empty in strict
                if pj_fn or pj_h or (pj_cons not in (None, False)):
                    failures.append("Policy invariant: strict run contains projector fields that should be empty/None.")
            else:
                # projected mode must not silently tolerate inconsistency
                if (mode == "projected(file)") and not pj_fn:
                    failures.append("Policy invariant: projected(file) but projector_filename is empty.")
                if (mode.startswith("projected")) and (pj_cons is False):
                    failures.append("Policy invariant: projected run with projector_consistent_with_d=False should have aborted earlier.")

            # 3) A/B freshness guard and mirroring
            if ab_ctx:
                sig_now = _safe_get_inputs_sig(inputs)
                sig_ab  = ab_ctx.get("inputs_sig", [])
                if sig_now != sig_ab:
                    warnings.append("A/B snapshot is **stale** (inputs_sig changed) — it will not be embedded in cert.")
                # projected leg policy tag must mirror active
                proj_leg = ab_ctx.get("projected", {}) or {}
                ab_label = proj_leg.get("label","")
                active_label = policy
                if active_label and ab_label and (ab_label != active_label):
                    failures.append(f"A/B mirror: projected leg label {ab_label!r} != active policy {active_label!r}.")
                # projector mirror (when file)
                if mode == "projected(file)":
                    if proj_leg.get("projector_filename","") != pj_fn:
                        failures.append("A/B mirror: projected(file) leg didn't reuse the same projector file path.")
                    if proj_leg.get("projector_hash","") and pj_h and (proj_leg.get("projector_hash") != pj_h):
                        failures.append("A/B mirror: projected(file) leg projector_hash != run_ctx.projector_hash.")

            # 4) Residual tags presence
            res_tags = st.session_state.get("residual_tags", {}) or {}
            if not res_tags.get("strict"):
                warnings.append("Residual tags: strict tag missing. (Run Overlap to populate.)")
            if policy.startswith("projected") and ("projected" not in res_tags):
                warnings.append("Residual tags: projected tag missing. (Run Overlap to populate.)")

            # 5) Projector(file) basic validation (non-invasive re-check)
            if mode == "projected(file)":
                try:
                    # Reuse your validator path (must raise on issues).
                    _P, _meta = projector_choose_active(st.session_state.get("overlap_cfg", cfg_projected_base()), boundaries)
                    # Additionally check diagonal == lane mask from run_ctx (SSOT)
                    diagP = _diag_vec(_P)
                    lane  = run_ctx.get("lane_mask_k3", [])
                    if diagP != lane:
                        failures.append(f"Projector diag mismatch: diag(P) {diagP} != lane_mask_k3 {lane}.")
                except ValueError as e:
                    failures.append(f"Projector(file) validator error: {e}")

            # 6) Cache key echo (proof after flush) — not a failure unless empty
            cache_key_tuple = (
                policy,
                inputs.get("boundaries_hash",""),
                inputs.get("C_hash",""),
                inputs.get("H_hash",""),
                inputs.get("U_hash",""),
                (pj_fn if mode == "projected(file)" else ""),
                (pj_h  if mode.startswith("projected") else ""),
            )
            if not any(cache_key_tuple):
                warnings.append("Composite cache key looks empty; did inputs load correctly?")
            st.code(
                "cache_key = (\n  " + ",\n  ".join(repr(x) for x in cache_key_tuple) + "\n)",
                language="python"
            )

            # 7) Optional: quick “No silent fallback” dry-run probe (non-mutating)
            #    Only run if in file mode and file exists — we try to parse it and assert mode stays 'file'.
            if mode == "projected(file)" and os.path.exists(pj_fn or ""):
                try:
                    with open(pj_fn, "r") as pf:
                        raw = _json.load(pf)
                    # acceptable formats: {"blocks":{"3":[...]}} or [[...]]
                    P3 = raw.get("blocks", {}).get("3") if isinstance(raw, dict) else (raw if isinstance(raw, list) else None)
                    if not isinstance(P3, list):
                        failures.append("Projector(file) content unreadable (no matrix).")
                except Exception as e:
                    failures.append(f"Projector(file) read error: {e}")

            # 8) Final banner
            if failures:
                st.error("❌ Self-tests FAILED — plumbing not healthy. Fix these before exploring:")
                for f in failures:
                    st.markdown(f"- {f}")
            else:
                st.success("✅ Self-tests passed.")
                if warnings:
                    st.warning("Notes / warnings (non-blocking):\n" + "\n".join(f"- {w}" for w in warnings))
                else:
                    st.caption("All clean. 🚀")

            # 9) Persist a tiny machine-readable snapshot (for future CI)
            st.session_state["selftests_snapshot"] = {
                "schema_version": SELFTESTS_SCHEMA_VERSION,
                "created_at": _utc_iso(),
                "policy": policy,
                "mode": mode,
                "district_id": di.get("district_id","UNKNOWN"),
                "hashes": {
                    "boundaries_hash": inputs.get("boundaries_hash",""),
                    "C_hash": inputs.get("C_hash",""),
                    "H_hash": inputs.get("H_hash",""),
                    "U_hash": inputs.get("U_hash",""),
                    "projector_hash": pj_h,
                },
                "failures": failures,
                "warnings": warnings,
                "stamp": stamp,
            }

        except Exception as e:
            st.error(f"Self-tests crashed: {e}")




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
