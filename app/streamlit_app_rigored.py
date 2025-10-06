# ────────────────────────────── IMPORTS (top) ──────────────────────────────
import sys, os, json, csv, hashlib, platform, zipfile, tempfile, shutil, importlib.util, types, pathlib
from pathlib import Path
from io import BytesIO
from contextlib import contextmanager
from datetime import datetime, timezone
import streamlit as st

# alias shim used by helpers (underscored names)
import os as _os
import json as _json
import hashlib as _hashlib
import csv as _csv
import zipfile as _zipfile
import tempfile as _tempfile
import shutil as _shutil
from pathlib import Path as _Path





# Page config early so Streamlit is happy
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# ────────────────────────────── PACKAGE LOADER ──────────────────────────────
HERE   = Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
CORE   = HERE / "core"
PKG_DIR = OTCORE if OTCORE.exists() else CORE
PKG_NAME = "otcore" if OTCORE.exists() else "core"

# Ensure pkg namespace exists (so submodules import cleanly)
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

overlap_gate  = _load_pkg_module(f"{PKG_NAME}.overlap_gate",  "overlap_gate.py")
projector     = _load_pkg_module(f"{PKG_NAME}.projector",     "projector.py")
otio          = _load_pkg_module(f"{PKG_NAME}.io",            "io.py")
hashes        = _load_pkg_module(f"{PKG_NAME}.hashes",        "hashes.py")
unit_gate     = _load_pkg_module(f"{PKG_NAME}.unit_gate",     "unit_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers        = _load_pkg_module(f"{PKG_NAME}.towers",        "towers.py")
export_mod    = _load_pkg_module(f"{PKG_NAME}.export",        "export.py")

# Legacy alias so existing code can keep using io.parse_* safely
io = otio

# App version string used elsewhere
APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")

# ───────────────────── DISTRICT MAP (optional) ─────────────────────
# Map raw-bytes sha256(boundaries.json) → human-friendly district label.
# Fill as you discover hashes (the sidebar shows the hash to copy/paste).
DISTRICT_MAP: dict[str, str] = {
    "9da8b7f605c113ee059160cdaf9f93fe77e181476c72e37eadb502e7e7ef9701": "D1",
    "4356e6b608443b315d7abc50872ed97a9e2c837ac8b85879394495e64ec71521": "D2",
    "28f8db2a822cb765e841a35c2850a745c667f4228e782d0cfdbcb710fd4fecb9": "D3",
    "aea6404ae680465c539dc4ba16e97fbd5cf95bae5ad1c067dc0f5d38ca1437b5": "D4",
}

# ---------- Freshness / reset helpers ----------
def _ensure_fixture_nonce():
    ss = st.session_state
    if "_fixture_nonce" not in ss:
        ss["_fixture_nonce"] = 1

def _mark_fixtures_changed():
    ss = st.session_state
    _ensure_fixture_nonce()
    ss["_fixture_nonce"] += 1
    # clear only things that depend on inputs/projector selection
    for k in ("run_ctx","overlap_out","overlap_cfg","overlap_policy_label",
              "overlap_H","residual_tags","ab_compare","_projector_cache",
              "_projector_cache_ab","_last_cert_write_key"):
        ss.pop(k, None)

def _soft_reset_before_overlap():
    """Light reset before an Overlap run; does NOT touch files on disk."""
    ss = st.session_state
    for k in ("run_ctx","overlap_out","overlap_cfg","overlap_policy_label",
              "overlap_H","residual_tags","_last_cert_write_key"):
        ss.pop(k, None)

# Freshness / mutation tracking
def _bump_fixture_nonce():
    st.session_state["_fixture_nonce"] = (st.session_state.get("_fixture_nonce") or 0) + 1

def _mark_fixtures_changed():
    # call this whenever boundaries/C/H/U/shapes OR projector source toggles
    _bump_fixture_nonce()
    for k in ("run_ctx","overlap_out","residual_tags","ab_compare"):
        st.session_state.pop(k, None)



# ---------- stable hashing (bytes + json-obj) ----------
def _sha256_hex_bytes(b: bytes) -> str:
    return _hashlib.sha256(b).hexdigest()

def _sha256_hex_obj(obj) -> str:
    blob = _json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_hex_bytes(blob)

def hash_json(obj) -> str:
    return _sha256_hex_obj(obj)

def _iso_utc_now() -> str:
    return datetime.now(_timezone.utc).isoformat()

# ====================== Π FILE Validator (strict) ======================
def _mul_gf2(A, B):
    # Use app's mul if present; otherwise a safe GF(2) fallback
    if "mul" in globals() and callable(globals()["mul"]):
        return mul(A, B)
    if not A or not B: return []
    m, k, n = len(A), len(A[0]), len(B[0])
    # assume B is k x n
    out = [[0]*n for _ in range(m)]
    for i in range(m):
        for t in range(k):
            if A[i][t] & 1:
                # xor-add row of B
                bt = B[t]
                for j in range(n):
                    out[i][j] ^= (bt[j] & 1)
    return out

def validate_projector_file_strict(P, *, n3: int, lane_mask: list[int]) -> None:
    # shape
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
    lm   = [int(x) & 1 for x in (lane_mask or [])]
    if diag != lm:
        raise ValueError(f"P3_LANE_MISMATCH: diag(P)={diag} vs lane_mask(d3)={lm}")


# ---------- safe expander (never nests real expanders) ----------
try:
    from streamlit.errors import StreamlitAPIException  # type: ignore
except Exception:  # pragma: no cover
    class StreamlitAPIException(Exception):  # type: ignore
        pass

@contextmanager
def safe_expander(title: str, **kwargs):
    """
    Drop-in replacement for st.expander that never raises the
    'Expanders may not be nested' error. If a real expander
    would fail (already inside one), we render a container.
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
    if isinstance(upload, (str, _os.PathLike, _Path)):
        with open(str(upload), "r", encoding="utf-8") as f:
            return _json.load(f)
    try:
        data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
        return _json.loads(data.decode("utf-8"))
    except Exception:
        return None

def _stamp_filename(state_key: str, upload):
    """Record the uploaded filename into st.session_state[state_key] when present."""
    try:
        if upload is not None and hasattr(upload, "name"):
            st.session_state[state_key] = str(upload.name)
        elif upload is None:
            st.session_state.pop(state_key, None)
    except Exception:
        pass

# ---------- atomic writers ----------
def atomic_write_json(path: str | _Path, obj: dict, *, pretty: bool = False):
    path = _Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    blob = _json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(blob); f.flush(); _os.fsync(f.fileno())
    _os.replace(tmp, path)

def atomic_append_jsonl(path: str | _Path, row: dict):
    path = _Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    line = _json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with open(tmp, "wb") as f:
        f.write(line.encode("utf-8")); f.flush(); _os.fsync(f.fileno())
    with open(path, "ab") as out, open(tmp, "rb") as src:
        out.write(src.read()); out.flush(); _os.fsync(out.fileno())
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

def _atomic_write_csv(path: _Path, header: list[str], rows: list[list], meta_comment_lines: list[str] | None = None):
    tmp = _Path(str(path) + ".tmp"); tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if meta_comment_lines:
            for k in meta_comment_lines:
                w.writerow([f"# {k}"])
        w.writerow(header)
        w.writerows(rows)
        f.flush(); _os.fsync(f.fileno())
    _os.replace(tmp, path)

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
def _eye(n: int): return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def _mul_gf2(A, B):
    if not A or not B or not A[0] or not B[0]: return []
    r, k = len(A), len(A[0]); k2, c = len(B), len(B[0])
    if k != k2: raise ValueError(f"dim mismatch: {r}x{k} @ {k2}x{c}")
    out = [[0]*c for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for t in range(k):
            if Ai[t] & 1:
                Bt = B[t]
                for j in range(c):
                    out[i][j] ^= (Bt[j] & 1)
    return out

def _is_idempotent_gf2(P):
    try: return _mul_gf2(P, P) == P
    except Exception: return False

def _is_diagonal(P):
    m = len(P) or 0; n = len(P[0]) if m else 0
    if m != n: return False
    for i in range(m):
        for j in range(n):
            if i != j and (P[i][j] & 1): return False
    return True

def _diag(P): return [int(P[i][i] & 1) for i in range(len(P))] if P and P[0] else []

# ---------- projector chooser (strict/auto/file) ----------
class _P3Error(ValueError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}"); self.code = code

def _read_projector_matrix(path_str: str):
    p = _Path(path_str)
    if not p.exists(): raise _P3Error("P3_SHAPE", f"projector file not found: {path_str}")
    with open(p, "r", encoding="utf-8") as f: d = _json.load(f)
    if isinstance(d, dict):
        b = (d.get("blocks", {}) or {}).get("3")
        if isinstance(b, list): return b
    if isinstance(d, list): return d
    raise _P3Error("P3_SHAPE", "unrecognized projector JSON structure")

def projector_choose_active(cfg_active: dict, boundaries):
    d3 = (boundaries.blocks.__root__.get("3") or [])
    n3 = len(d3[0]) if (d3 and d3[0]) else 0
    lane_mask = _lane_mask_from_d3(boundaries)
    mode = "strict"; P_active = []; pj_filename = ""; pj_hash = ""; pj_consistent = None

    if not cfg_active or not cfg_active.get("enabled_layers"):
        return P_active, {"d3": d3, "n3": n3, "lane_mask": lane_mask, "mode": mode,
                          "projector_filename": pj_filename, "projector_hash": pj_hash,
                          "projector_consistent_with_d": pj_consistent}

    source = (cfg_active.get("source", {}) or {}).get("3", "auto")
    mode = "projected(auto)" if source == "auto" else "projected(file)"

    if source == "auto":
        diag = (lane_mask if lane_mask else [1]*n3)
        P_active = [[1 if i == j and diag[j] else 0 for j in range(n3)] for i in range(n3)]
        pj_hash = _sha256_hex_obj(P_active); pj_consistent = True
        return P_active, {"d3": d3, "n3": n3, "lane_mask": lane_mask, "mode": mode,
                          "projector_filename": "", "projector_hash": pj_hash,
                          "projector_consistent_with_d": pj_consistent}

    pj_filename = (cfg_active.get("projector_files", {}) or {}).get("3", "")
    if not pj_filename: raise _P3Error("P3_SHAPE", "no projector file provided for file-mode")

    P = _read_projector_matrix(pj_filename)
    m = len(P) or 0; n = len(P[0]) if m else 0
    if n3 == 0 or m != n3 or n != n3: raise _P3Error("P3_SHAPE", f"expected {n3}x{n3}, got {m}x{n}")
    if not _is_idempotent_gf2(P):    raise _P3Error("P3_IDEMP", "P is not idempotent over GF(2)")
    if not _is_diagonal(P):          raise _P3Error("P3_DIAGONAL", "P has off-diagonal non-zeros")
    pj_diag = _diag(P)
    if lane_mask and pj_diag != [int(x) for x in lane_mask]:
        raise _P3Error("P3_LANE_MISMATCH", f"diag(P) != lane_mask(d3) → {pj_diag} vs {lane_mask}")
    pj_hash = _sha256_hex_obj(P); pj_consistent = True if lane_mask else None

    return P, {"d3": d3, "n3": n3, "lane_mask": lane_mask, "mode": mode,
               "projector_filename": pj_filename, "projector_hash": pj_hash,
               "projector_consistent_with_d": pj_consistent}

# ---------- misc ----------
def hash_matrix_norm(M) -> str:
    if not M: return hash_json([])
    norm = [[int(x) & 1 for x in row] for row in M]
    return hash_json(norm)

def _zip_arcname(abspath: str) -> str:
    p = _Path(abspath)
    try: return p.resolve().relative_to(_Path.cwd().resolve()).as_posix()
    except Exception: return p.name

def build_cert_bundle(*, district_id: str, policy_tag: str, cert_path: str,
                      content_hash: str | None = None, extras: list[str] | None = None) -> str:
    cert_p = _Path(cert_path)
    if not cert_p.exists(): raise FileNotFoundError(f"Cert not found: {cert_path}")
    with open(cert_p, "r", encoding="utf-8") as f: cert = _json.load(f)
    if not content_hash: content_hash = ((cert.get("integrity") or {}).get("content_hash") or "")
    suffix = content_hash[:12] if content_hash else "nohash"
    safe_policy = (policy_tag or cert.get("policy", {}).get("policy_tag", "policy")).replace("/", "_").replace(" ", "_")
    zpath = BUNDLES_DIR / f"overlap_bundle__{district_id or 'UNKNOWN'}__{safe_policy}__{suffix}.zip"
    files = [str(cert_p)]
    for p in (extras or []):
        if p and _os.path.exists(p): files.append(p)
    fd, tmp_name = _tempfile.mkstemp(dir=BUNDLES_DIR, prefix=".tmp_bundle_", suffix=".zip")
    _os.close(fd); tmp_path = _Path(tmp_name)
    try:
        with _zipfile.ZipFile(tmp_path, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
            for abspath in files:
                abspath = str(_Path(abspath).resolve())
                zf.write(abspath, arcname=_zip_arcname(abspath))
        try: _os.replace(tmp_path, zpath)
        except OSError: _shutil.move(str(tmp_path), str(zpath))
    finally:
        if tmp_path.exists(): tmp_path.unlink(missing_ok=True)
    return str(zpath)

# ---------- inputs block builder (SSOT) ----------
def build_inputs_block(boundaries, cmap, H_used, shapes, filenames: dict) -> dict:
    C3 = (cmap.blocks.__root__.get("3") or [])
    d3 = (boundaries.blocks.__root__.get("3") or [])
    dims = {"n3": len(C3) if C3 else (len(d3[0]) if (d3 and d3[0]) else 0),
            "n2": len(cmap.blocks.__root__.get("2") or [])}
    hashes_dict = {
        "boundaries_hash": hash_json(boundaries.dict() if hasattr(boundaries, "dict") else {}),
        "C_hash":          hash_json(cmap.dict() if hasattr(cmap, "dict") else {}),
        "H_hash":          hash_json(H_used.dict() if hasattr(H_used, "dict") else {}),
        "U_hash":          hash_json(shapes.dict() if hasattr(shapes, "dict") else {}),
    }
    return {"filenames": filenames, "dims": dims, **hashes_dict, "shapes_hash": hashes_dict["U_hash"]}


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
                _raw = f_bound.getvalue()
                boundaries_hash_fresh = _sha256_hex_bytes(_raw)
            else:
                boundaries_hash_fresh = _sha256_hex_obj(d_bound)
        except Exception:
            boundaries_hash_fresh = _sha256_hex_obj(d_bound)

        # Light district info (lane mask + signature)
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


# ---- Policy config helpers (define once, before Tab 2 uses them) -------------
def cfg_strict() -> dict:
    return {
        "enabled_layers": [],
        "modes": {},
        "source": {},
        "projector_files": {},
    }

def cfg_projected_base() -> dict:
    return {
        "enabled_layers": [3],
        "modes": {"3": "columns"},
        "source": {"3": "auto"},
        "projector_files": {},
    }

def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src = (cfg.get("source", {}) or {}).get("3", "auto")
    mode = (cfg.get("modes", {}) or {}).get("3", "columns")
    return f"projected({mode}@k=3,{src})"

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

        # Clear stale session bits if boundaries changed
        _prev_bhash = st.session_state.get("_last_boundaries_hash")
        if _prev_bhash and _prev_bhash != boundaries_hash_fresh:
            st.session_state.pop("ab_compare", None)
            st.session_state.pop("district_id", None)
            st.session_state.pop("_projector_cache", None)
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
import json as _json
import os
import hashlib  # <-- added this import

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

# Load H (or empty cmap)
def _load_h_local():
    try:
        if f_H is None:
            return io.parse_cmap({"blocks": {}})
        return io.parse_cmap(read_json_file(f_H))
    except Exception:
        return io.parse_cmap({"blocks": {}})

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

# ------------------------------ Run Overlap + UI (tidy, with A/B freshness) ------------------------------

# --- helpers for freshness + soft reset (scoped, safe to re-declare) ---
def _ensure_fixture_nonce():
    ss = st.session_state
    if "_fixture_nonce" not in ss:
        ss["_fixture_nonce"] = 1

def _soft_reset_before_overlap():
    """Light reset before an Overlap run; does NOT touch files on disk."""
    ss = st.session_state
    for k in ("run_ctx","overlap_out","overlap_cfg","overlap_policy_label",
              "overlap_H","residual_tags","_last_cert_write_key"):
        ss.pop(k, None)

def _current_inputs_sig() -> list[str]:
    _ib = st.session_state.get("_inputs_block") or {}
    return [
        str(_ib.get("boundaries_hash", "")),
        str(_ib.get("C_hash", "")),
        str(_ib.get("H_hash", "")),
        str(_ib.get("U_hash", "")),
        str(_ib.get("shapes_hash", "")),
    ]

def run_overlap():
    # step 2: guarantee a nonce exists for this session
    _ensure_fixture_nonce()

    # Clear previous session results (kept as part of Overlap run body)
    for k in ("proj_meta", "run_ctx", "residual_tags", "overlap_out", "overlap_H", "overlap_cfg", "overlap_policy_label"):
        st.session_state.pop(k, None)

    # Bind projector (fail-fast on FILE)
    try:
        P_active, meta = projector_choose_active(cfg_active, boundaries)
    except ValueError as e:
        st.error(str(e))
        d3_now = (boundaries.blocks.__root__.get("3") or [])
        st.session_state["run_ctx"] = {
            "policy_tag": policy_label_from_cfg(cfg_active),
            "mode": _derive_mode_from_cfg(cfg_active),
            "fixture_nonce": st.session_state.get("_fixture_nonce", 0),   # <-- write nonce on error path too
            "d3": d3_now,
            "n3": len(d3_now[0]) if (d3_now and d3_now[0]) else 0,
            "lane_mask_k3": [],
            "P_active": [],
            "projector_filename": (cfg_active.get("projector_files", {}) or {}).get("3", ""),
            "projector_hash": "",
            "projector_consistent_with_d": False,
            "source": (cfg_active.get("source") or {}),
            "errors": [str(e)],
        }
        st.stop()

    # Context (use what projector_choose_active gave you)
    d3   = meta.get("d3") if "d3" in meta else (boundaries.blocks.__root__.get("3") or [])
    n3   = meta.get("n3") if "n3" in meta else (len(d3[0]) if (d3 and d3[0]) else 0)
    mode = meta.get("mode", _derive_mode_from_cfg(cfg_active))

    # --- Authoritative lane mask from THIS d3 (no re-read of boundaries) ---
    lane_mask = [1 if any(d3[i][j] & 1 for i in range(len(d3))) else 0 for j in range(n3)]
    assert len(lane_mask) == n3, "lane_mask_k3 length mismatch with n3"

    # Compute residuals (strict)
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
        rows, cols = len(R), len(R[0])
        lanes_idx = [j for j, m in enumerate(lm) if m]
        ker_idx   = [j for j, m in enumerate(lm) if not m]
        def _col_nonzero(j): return any(R[i][j] & 1 for i in range(rows))
        lanes_resid = any(_col_nonzero(j) for j in lanes_idx) if lanes_idx else False
        ker_resid   = any(_col_nonzero(j) for j in ker_idx)   if ker_idx   else False
        if not lanes_resid and not ker_resid: return "none"
        if lanes_resid and not ker_resid:     return "lanes"
        if ker_resid and not lanes_resid:     return "ker"
        return "mixed"

    tag_strict = _residual_tag(R3_strict, lane_mask)
    eq3_strict = _is_zero(R3_strict)

    # Projected leg (if enabled)
    if cfg_active.get("enabled_layers"):
        R3_proj  = mul(R3_strict, P_active) if (R3_strict and P_active) else []
        eq3_proj = _is_zero(R3_proj)
        tag_proj = _residual_tag(R3_proj, lane_mask)
        out = {"3": {"eq": bool(eq3_proj), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict, "projected": tag_proj}
    else:
        out = {"3": {"eq": bool(eq3_strict), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict}

    # Persist SSOT (write ONCE) — include the current fixture nonce
    st.session_state["overlap_out"] = out
    st.session_state["overlap_cfg"] = cfg_active
    st.session_state["overlap_policy_label"] = policy_label_from_cfg(cfg_active)
    st.session_state["overlap_H"] = H_local
    st.session_state["run_ctx"] = {
        "policy_tag": st.session_state["overlap_policy_label"],
        "mode": mode,
        "fixture_nonce": st.session_state.get("_fixture_nonce", 0),  # <-- step 2: bind nonce into run_ctx
        "d3": d3,
        "n3": n3,
        "lane_mask_k3": lane_mask,
        "P_active": P_active,
        "projector_filename": meta.get("projector_filename", ""),
        "projector_hash": meta.get("projector_hash", ""),
        "projector_consistent_with_d": meta.get("projector_consistent_with_d", None),
        "source": (cfg_active.get("source") or {}),
        "errors": [],
    }

    # Debug line (helps catch axis/perm issues instantly)
    st.caption(f"d3: rows={len(d3)} cols={n3} · lane_mask={lane_mask} (pattern '{''.join(map(str,lane_mask))}')")

    # UI Result + Banner
    st.json(out)
    if mode == "projected(file)":
        if meta.get("projector_consistent_with_d", False):
            st.success(f"projected(file) OK · {meta.get('projector_filename','')} · {meta.get('projector_hash','')[:12]} ✔️")
        else:
            st.warning("Projected(file) is not consistent with current d3 (check shape/idempotence/diag/lane).")

# Button (unique key) — step 3: soft reset before running
if st.button("Run Overlap", key="btn_run_overlap_final"):
    _soft_reset_before_overlap()
    run_overlap()



# ── Debug: lane mask & projector diag (place near top of Tab 2) ───────────────
with safe_expander("Debug · lane mask & Π diag"):
    try:
        d3_now = (boundaries.blocks.__root__.get("3") or [])
        n3_now = len(d3_now[0]) if (d3_now and d3_now[0]) else 0

        # lane mask from boundaries (your existing helper already does this)
        lm_now = _lane_mask_from_d3(boundaries)
        st.write(f"n3={n3_now} · lane_mask(d3) = {lm_now}")

        # if FILE projector active, show its diag
        cfg_now = st.session_state.get("overlap_cfg") or {}
        if (cfg_now.get("source", {}) or {}).get("3") == "file":
            try:
                P_file, _meta = projector_choose_active(cfg_now, boundaries)
                diagP = [int(P_file[i][i] & 1) for i in range(len(P_file))] if P_file else []
                st.write(f"diag(P_file) = {diagP}")
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
_short = lambda h: (h or "")[:8]
bH = _short(_ib.get("boundaries_hash","")); cH = _short(_ib.get("C_hash",""))
hH = _short(_ib.get("H_hash",""));        uH = _short(_ib.get("U_hash",""))
pH = _short(_rc.get("projector_hash","")) if str(_rc.get("mode","")).startswith("projected") else "—"

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
_rc = st.session_state.get("run_ctx") or {}

# Residual chips
_rtags = st.session_state.get("residual_tags") or {}
if _rtags:
    s_tag = _rtags.get("strict", "—")
    p_tag = _rtags.get("projected", "—") if str(_rc.get("mode","")).startswith("projected") else "—"
    st.caption(f"Residuals → strict: `{s_tag}` · projected: `{p_tag}`")

# Single A/B freshness pill (uses helper from A/B section)
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

def _current_inputs_sig():
    ib = st.session_state.get("_inputs_block") or {}
    return [str(ib.get("boundaries_hash","")), str(ib.get("C_hash","")),
            str(ib.get("H_hash","")), str(ib.get("U_hash","")),
            str(ib.get("shapes_hash",""))]

def perform_overlap_check(boundaries_obj, cmap_obj, H_used, cfg=None):
    return overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_used, projection_config=cfg)

def get_policy_label(cfg):
    return policy_label_from_cfg(cfg)

_ab = st.session_state.get("ab_compare") or {}

def is_ab_fresh():
    return bool(_ab and (_ab.get("inputs_sig") == _current_inputs_sig()))

# Snapshot freshness badge
if _ab:
    st.caption(f"A/B snapshot: {'🟢 fresh (will embed in cert)' if is_ab_fresh() else '🟡 stale (won’t embed)'}")
else:
    st.caption("A/B snapshot: —")

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
        # Check if 'source' is 'file' in the active config
        src3 = (cfg_for_ab.get("source") or {}).get("3")
        if src3 == "file":
            # Grab and validate the projector when source is FILE
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
            # If not FILE, just select the active projector normally
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

# Stale handler / clearer
_ab = st.session_state.get("ab_compare") or {}
if _ab:
    fresh = (_ab.get("inputs_sig") == _current_inputs_sig())
    st.caption(f"A/B snapshot: {'🟢 fresh (will embed)' if fresh else '🟡 stale (won’t embed)'}")
    if not fresh and st.button("Clear stale A/B", key="btn_ab_clear_final"):
        st.session_state.pop("ab_compare", None)
        st.success("Cleared A/B snapshot. Re-run A/B to refresh.")
else:
    st.caption("A/B snapshot: —")




# ────────────────────── Reports: Perturbation Sanity & Fence Stress ──────────────────────
import os, csv, tempfile, hashlib, json
from pathlib import Path
from datetime import datetime, timezone

PERTURB_SCHEMA_VERSION = "1.0.0"
FENCE_SCHEMA_VERSION   = "1.0.0"
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PERTURB_OUT_PATH = REPORTS_DIR / "perturbation_sanity.csv"
FENCE_OUT_PATH   = REPORTS_DIR / "fence_stress.csv"

def _utc_iso(): return datetime.now(timezone.utc).isoformat()

def _atomic_write_csv(path: Path, header, rows, meta_comments: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", newline="") as tmp:
        for line in meta_comments:
            tmp.write(f"# {line}\n")
        w = csv.writer(tmp)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
        tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    os.replace(tmp_name, path)

def _lane_mask_from_d3_local(boundaries_obj) -> list[int]:
    try:
        d3 = (boundaries_obj.blocks.__root__.get("3") or [])
    except Exception:
        d3 = []
    n3 = len(d3[0]) if d3 and d3[0] else 0
    return [1 if any((row[j] & 1) for row in d3) else 0 for j in range(n3)]

def _copy_mat(M): return [row[:] for row in (M or [])]
def _is_zero(M): return (not M) or all(all((x & 1) == 0 for x in row) for row in M)
def _strict_R3(H2, d3, C3):
    I3 = eye(len(C3)) if C3 else []
    return _xor_mat(mul(H2, d3), _xor_mat(C3, I3)) if (H2 and d3 and C3) else []
def _projected_R3(R3_strict, P_active):
    return mul(R3_strict, P_active) if (R3_strict and P_active) else []

def _sig_tag_eq(boundaries_obj, cmap_obj, H_used_obj, P_active=None):
    """Return (lane_mask, tag_strict, eq3_strict, tag_proj, eq3_proj)."""
    d3 = (boundaries_obj.blocks.__root__.get("3") or [])
    H2 = (H_used_obj.blocks.__root__.get("2") or [])
    C3 = (cmap_obj.blocks.__root__.get("3") or [])
    lm = _lane_mask_from_d3_local(boundaries_obj)
    R3s = _strict_R3(H2, d3, C3)
    tag_s = residual_tag(R3s, lm)
    eq_s  = _is_zero(R3s)
    if P_active:
        R3p = _projected_R3(R3s, P_active)
        tag_p = residual_tag(R3p, lm)
        eq_p  = _is_zero(R3p)
    else:
        tag_p, eq_p = None, None
    return lm, tag_s, bool(eq_s), tag_p, (None if eq_p is None else bool(eq_p))

with st.expander("Reports: Perturbation Sanity & Fence Stress"):
    colA, colB = st.columns([2,2])
    with colA:
        max_flips = st.number_input("Perturbation: max flips", min_value=1, max_value=500, value=24, step=1, key="ps_max")
        seed_txt  = st.text_input("Seed (determines flip order)", value="ps-seed-1", key="ps_seed")
    with colB:
        run_fence  = st.checkbox("Include Fence stress run", value=True, key="fence_on")

    if st.button("Run Perturbation Sanity (and Fence if checked)", key="ps_run"):
        try:
            # SSOT objects
            rc = st.session_state.get("run_ctx") or {}
            H_used = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})
            P_active = rc.get("P_active") if str(rc.get("mode","")).startswith("projected") else None
            B0, C0, H0 = boundaries, cmap, H_used

            # Baseline
            lm0, tag_s0, eq_s0, tag_p0, eq_p0 = _sig_tag_eq(B0, C0, H0, P_active)
            d3_base = _copy_mat((B0.blocks.__root__.get("3") or []))
            n2 = len(d3_base); n3 = len(d3_base[0]) if (d3_base and d3_base[0]) else 0

            def _flip_targets(n2, n3, budget, seed_str):
                h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16)
                i = (h % (max(1, n2))) if n2 else 0
                j = ((h >> 8) % (max(1, n3))) if n3 else 0
                for k in range(budget):
                    yield (i, j, k)
                    i = (i + 1 + (h % 3)) % (n2 or 1)
                    j = (j + 2 + ((h >> 5) % 5)) % (n3 or 1)

            rows = []
            drift_witnessed = False

            for (r, c, k) in _flip_targets(n2, n3, int(max_flips), seed_txt):
                if not (n2 and n3):
                    rows.append([k, 0, "no-op", "empty fixture"])
                    continue

                d3_mut = _copy_mat(d3_base)
                d3_mut[r][c] ^= 1  # GF(2) flip

                # Mutate boundaries via JSON round-trip (keeps types identical to rest of app)
                dB = B0.dict() if hasattr(B0, "dict") else {"blocks": {}}
                dB = json.loads(json.dumps(dB))
                dB.setdefault("blocks", {})["3"] = d3_mut
                Bk = io.parse_boundaries(dB)

                lmK, tag_sK, eq_sK, tag_pK, eq_pK = _sig_tag_eq(Bk, C0, H0, P_active)

                guard_tripped = int(lmK != lm0)   # grammar/lane drift
                expected_guard = "ker_guard"
                note = ""
                if guard_tripped and not drift_witnessed:
                    drift_witnessed = True
                    cert_like = st.session_state.get("cert_payload")
                    if cert_like:
                        try:
                            append_witness_row(cert_like, reason="grammar-drift",
                                               residual_tag_val=tag_sK or "none",
                                               note=f"flip#{k} at (r={r}, c={c})")
                        except Exception:
                            pass
                    note = "lane_mask_changed → auto-witness logged"

                rows.append([k, guard_tripped, expected_guard, note])

            header = ["flip_id", "guard_tripped", "expected_guard", "note"]
            meta = [
                f"schema_version={PERTURB_SCHEMA_VERSION}",
                f"saved_at={_utc_iso()}",
                f"run_id={(st.session_state.get('cert_payload') or {}).get('identity',{}).get('run_id','')}",
                f"app_version={globals().get('APP_VERSION_STR', getattr(hashes,'APP_VERSION','v0.1-core'))}",
                f"seed={seed_txt}",
                f"n2={n2}",
                f"n3={n3}",
                f"baseline_tag_strict={tag_s0}",
                f"baseline_tag_projected={'' if tag_p0 is None else tag_p0}",
            ]
            _atomic_write_csv(PERTURB_OUT_PATH, header, rows, meta)
            st.success(f"Perturbation sanity saved → {PERTURB_OUT_PATH}")

            if run_fence:
                H2 = (H0.blocks.__root__.get("2") or [])
                # U_shrink
                H2_shrink = _copy_mat(H2[:-1]) if len(H2) >= 1 else _copy_mat(H2)
                H_shrink = json.loads(json.dumps(H0.dict() if hasattr(H0,"dict") else {"blocks": {}}))
                H_shrink.setdefault("blocks", {})["2"] = H2_shrink
                # U_plus
                if H2 and H2[0]:
                    zero_row = [0]*len(H2[0]); H2_plus = _copy_mat(H2) + [zero_row]
                else:
                    H2_plus = _copy_mat(H2)
                H_plus = json.loads(json.dumps(H0.dict() if hasattr(H0,"dict") else {"blocks": {}}))
                H_plus.setdefault("blocks", {})["2"] = H2_plus

                C3 = (C0.blocks.__root__.get("3") or [])
                d3 = (B0.blocks.__root__.get("3") or [])
                R3_shrink = _strict_R3(H2_shrink, d3, C3)
                R3_plus   = _strict_R3(H2_plus,   d3, C3)
                eq_shrink = int(_is_zero(R3_shrink))
                eq_plus   = int(_is_zero(R3_plus))

                fence_rows = [
                    ["U_shrink", f"[1,{eq_shrink}]", "drop last H2 row"],
                    ["U_plus",   f"[1,{eq_plus}]",   "append zero row to H2"],
                ]
                fence_header = ["U_class", "pass_vec", "note"]
                fence_meta = [
                    f"schema_version={FENCE_SCHEMA_VERSION}",
                    f"saved_at={_utc_iso()}",
                    f"run_id={(st.session_state.get('cert_payload') or {}).get('identity',{}).get('run_id','')}",
                    f"app_version={globals().get('APP_VERSION_STR', getattr(hashes,'APP_VERSION','v0.1-core'))}",
                ]
                _atomic_write_csv(FENCE_OUT_PATH, fence_header, fence_rows, fence_meta)
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




   



# ───────────────────────── Gallery helpers: dedupe + exports ─────────────────────────
from pathlib import Path
import json as _json
import os

LOGS_DIR = Path(globals().get("LOGS_DIR", "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_PATH   = LOGS_DIR / "gallery.jsonl"
WITNESSES_PATH = LOGS_DIR / "witnesses.jsonl"

def _jsonl_read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(_json.loads(ln))
                except Exception:
                    continue
    except Exception:
        pass
    return out

def _hx(inputs: dict, k: str) -> str:
    # supports both flat inputs.hashes.* or inputs.* layouts inside cert
    if not isinstance(inputs, dict):
        return ""
    # flat
    v = inputs.get(k)
    if v:
        return str(v)
    # nested
    h = inputs.get("hashes") or {}
    return str(h.get(k, ""))

def _gallery_key_from_cert(cert: dict) -> tuple:
    """
    (district_id, policy_tag, bHash, C_hash, H_hash, U_hash[, projector_hash if projected(file)])
    """
    ident  = cert.get("identity", {}) or {}
    pol    = cert.get("policy",   {}) or {}
    inputs = cert.get("inputs",   {}) or {}

    district_id = str(ident.get("district_id", "UNKNOWN"))
    policy_tag  = str(pol.get("policy_tag", "strict"))

    bH = _hx(inputs, "boundaries_hash")
    cH = _hx(inputs, "C_hash")
    hH = _hx(inputs, "H_hash")
    uH = _hx(inputs, "U_hash")

    # Optional projector hash only when projected(file)
    include_p = policy_tag.endswith(",file)") or "projected(file)" in policy_tag
    pH = str(pol.get("projector_hash","")) if include_p else None

    base = (district_id, policy_tag, bH, cH, hH, uH)
    return base + ((pH,) if include_p else ())

def append_gallery_row(cert: dict, *, growth_bumps: int = 0, strictify: str = "tbd") -> bool:
    """
    Append once per dedupe key. Returns True if appended, False if duplicate.
    """
    # Compute dedupe key from the cert
    key = _gallery_key_from_cert(cert)

    # Load existing keys
    existing = set()
    for r in _jsonl_read_all(GALLERY_PATH):
        try:
            # Back-compat: recompute key from stored rows (they contain cert-ish fields)
            if "dedupe_key" in r:
                k = tuple(r["dedupe_key"])
            else:
                # Fall back to recomputing from embedded fields in row
                ident  = r.get("identity", {})
                pol    = r.get("policy",   {})
                inputs = r.get("hashes",   {}) or r.get("inputs", {}).get("hashes", {}) or {}
                district_id = str(ident.get("district_id", "UNKNOWN"))
                policy_tag  = str(pol.get("policy_tag", "strict"))
                bH = str(inputs.get("boundaries_hash",""))
                cH = str(inputs.get("C_hash",""))
                hH = str(inputs.get("H_hash",""))
                uH = str(inputs.get("U_hash",""))
                include_p = policy_tag.endswith(",file)") or "projected(file)" in policy_tag
                pH = str(pol.get("projector_hash","")) if include_p else None
                k  = (district_id, policy_tag, bH, cH, hH, uH) + ((pH,) if include_p else ())
            existing.add(k)
        except Exception:
            continue

    if key in existing:
        return False  # duplicate

    # Build row (copy the fields you want visible in the tail/export)
    row = {
        "written_at_utc": (cert.get("identity") or {}).get("timestamp", ""),
        "district":       (cert.get("identity") or {}).get("district_id", "UNKNOWN"),
        "policy":         (cert.get("policy")   or {}).get("policy_tag", "strict"),
        "projector_hash": (cert.get("policy")   or {}).get("projector_hash", ""),
        "projector_filename": (cert.get("policy") or {}).get("projector_filename", ""),
        "hashes":         (cert.get("inputs")   or {}).get("hashes") or {
            "boundaries_hash": _hx(cert.get("inputs",{}), "boundaries_hash"),
            "C_hash":          _hx(cert.get("inputs",{}), "C_hash"),
            "H_hash":          _hx(cert.get("inputs",{}), "H_hash"),
            "U_hash":          _hx(cert.get("inputs",{}), "U_hash"),
        },
        "content_hash":   (cert.get("integrity") or {}).get("content_hash",""),
        "growth_bumps":   int(growth_bumps or 0),
        "strictify":      str(strictify),
        "dedupe_key":     list(key),
    }

    # Append using your atomic helper if present; else plain
    try:
        if "atomic_append_jsonl" in globals() and callable(globals()["atomic_append_jsonl"]):
            atomic_append_jsonl(GALLERY_PATH, row)
        else:
            GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(GALLERY_PATH, "a", encoding="utf-8") as f:
                f.write(_json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    except Exception as e:
        raise RuntimeError(f"Could not append Gallery row: {e}") from e

    return True

def append_witness_row(cert: dict, *, reason: str, residual_tag_val: str, note: str = "") -> None:
    row = {
        "written_at_utc": (cert.get("identity") or {}).get("timestamp", ""),
        "district":       (cert.get("identity") or {}).get("district_id", "UNKNOWN"),
        "policy":         (cert.get("policy")   or {}).get("policy_tag", "strict"),
        "residual_tag":   residual_tag_val,
        "reason":         reason,
        "note":           note or "",
        "content_hash":   (cert.get("integrity") or {}).get("content_hash",""),
    }
    if "atomic_append_jsonl" in globals() and callable(globals()["atomic_append_jsonl"]):
        atomic_append_jsonl(WITNESSES_PATH, row)
    else:
        WITNESSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WITNESSES_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

# Optional: exports used by your buttons
def export_gallery_csv() -> str:
    import csv, tempfile
    rows = _jsonl_read_all(GALLERY_PATH)
    if not rows:
        raise FileNotFoundError("No gallery.jsonl yet.")
    header = ["written_at_utc","district","policy","content_hash",
              "boundaries_hash","C_hash","H_hash","U_hash","projector_hash","projector_filename","growth_bumps","strictify"]
    fd, tmp = tempfile.mkstemp(prefix="gallery_export_", suffix=".csv", dir=LOGS_DIR)
    os.close(fd)
    outp = Path(tmp)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            h = r.get("hashes") or {}
            w.writerow([
                r.get("written_at_utc",""),
                r.get("district",""),
                r.get("policy",""),
                r.get("content_hash",""),
                h.get("boundaries_hash",""),
                h.get("C_hash",""),
                h.get("H_hash",""),
                h.get("U_hash",""),
                r.get("projector_hash",""),
                r.get("projector_filename",""),
                r.get("growth_bumps",0),
                r.get("strictify",""),
            ])
    return str(outp)

def export_witnesses_json() -> str:
    import tempfile
    rows = _jsonl_read_all(WITNESSES_PATH)
    if not rows:
        raise FileNotFoundError("No witnesses.jsonl yet.")
    fd, tmp = tempfile.mkstemp(prefix="witnesses_export_", suffix=".json", dir=LOGS_DIR)
    os.close(fd)
    outp = Path(tmp)
    with outp.open("w", encoding="utf-8") as f:
        _json.dump(rows, f, ensure_ascii=False, indent=2, sort_keys=True)
    return str(outp)


    # ───────────────────────── Gallery / Witness actions ─────────────────────────
st.divider()
st.caption("Gallery & Witness")

# -- SSOT pulls
_ss         = st.session_state
run_ctx     = _ss.get("run_ctx") or {}
overlap_out = _ss.get("overlap_out") or {}
res_tags    = _ss.get("residual_tags", {}) or {}
last_cert   = _ss.get("last_cert_path", "")
cert_cached = _ss.get("cert_payload")  # in-memory cert set by cert writer

# -- tiny helper: load the cert dict (prefer in-memory; fallback to disk)
def _load_cert_dict():
    if cert_cached:
        return cert_cached
    lp = last_cert
    if lp and os.path.exists(lp):
        try:
            with open(lp, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return None

# -- predicates
mode_now     = str(run_ctx.get("mode", "strict"))
is_projected = mode_now.startswith("projected")
k3_ok        = bool(overlap_out.get("3", {}).get("eq", False))
k2_ok        = bool(overlap_out.get("2", {}).get("eq", False))
grid_ok      = bool(overlap_out.get("grid", True))
fence_ok     = bool(overlap_out.get("fence", True))

can_gallery  = (is_projected and k3_ok and bool(last_cert))
can_witness  = (grid_ok and fence_ok and (not k3_ok) and bool(last_cert))

# -- UI: side-by-side actions
colG, colW = st.columns(2)

with colG:
    st.caption("Gallery")
    growth_bumps = st.number_input(
        "Growth bumps", min_value=0, max_value=9, value=0, step=1, key="gal_gb"
    )
    if st.button("Add to Gallery", key="btn_gallery_add", disabled=not can_gallery,
                 help="Enabled when a cert exists, policy is projected(auto/file), and k3 is ✓."):
        cert = _load_cert_dict()
        if not cert:
            st.error("No cert available. Run Overlap and write a cert first.")
        else:
            try:
                appended = append_gallery_row(cert, growth_bumps=growth_bumps, strictify="tbd")
                if appended:
                    st.success("Gallery row appended.")
                else:
                    st.info("Skipped — duplicate (dedupe key matched).")
            except Exception as e:
                st.error(f"Gallery append failed: {e}")
    if not can_gallery:
        st.caption("↳ Requires: projected mode + k3=✓ + a written cert.")

with colW:
    st.caption("Witness")
    reason = st.selectbox(
        "Reason",
        ["lanes-persist", "policy-mismatch", "needs-new-R", "grammar-drift", "other"],
        index=0,
        key="wit_reason",
        help="Pick the closest why-not-green reason."
    )
    note = st.text_input("Note (optional)", value="", key="wit_note")

    if st.button("Log Witness", key="btn_witness_add", disabled=not can_witness,
                 help="Enabled when a cert exists, grid/fence are ✓, and k3 is ✗ (stubborn red)."):
        cert = _load_cert_dict()
        if not cert:
            st.error("No cert available. Run Overlap and write a cert first.")
        else:
            try:
                tag_val = res_tags.get("projected" if is_projected else "strict", "none")
                append_witness_row(cert, reason=reason, residual_tag_val=tag_val, note=note)
                st.success(f"Witness logged (residual={tag_val}).")
            except Exception as e:
                st.error(f"Witness log failed: {e}")
    if not can_witness:
        st.caption("↳ Requires: grid=✓, fence=✓, k3=✗, and a written cert.")

# -- Recent tails (optional, lightweight)
with st.expander("Recent logs (tails)"):
    try:
        render_gallery_tail(limit=5)
    except Exception as e:
        st.warning(f"Could not render Gallery tail: {e}")
    try:
        render_witness_tail(limit=5)
    except Exception as e:
        st.warning(f"Could not render Witness tail: {e}")

# ───────────────────────── Logs: exports (optional) ──────────────────────────
with safe_expander("Logs: exports (optional)"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Gallery → CSV", key="export_gallery_csv_btn"):
            try:
                p = export_gallery_csv()
                st.success(f"Gallery CSV saved → {p}")
                try:
                    with open(p, "rb") as f:
                        st.download_button("Download gallery_export.csv", f, file_name="gallery_export.csv")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Gallery export failed: {e}")
    with col2:
        if st.button("Export Witnesses → JSON", key="export_witnesses_json_btn"):
            try:
                p = export_witnesses_json()
                st.success(f"Witnesses JSON saved → {p}")
                try:
                    with open(p, "rb") as f:
                        st.download_button("Download witnesses_export.json", f, file_name="witnesses_export.json")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Witnesses export failed: {e}")



from pathlib import Path
import os, json, tempfile, shutil, hashlib
from datetime import datetime, timezone

# Directory setup
PROJECTORS_DIR = Path("projectors")
PROJECTORS_DIR.mkdir(parents=True, exist_ok=True)
PJ_REG_PATH = PROJECTORS_DIR / "projector_registry.jsonl"

# Utility functions
def _utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as tmp:
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        tmp.write(blob)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)
    return _sha256_bytes(blob), len(blob)

def _append_registry_row(row: dict):
    # in-session deduplication based on (district, projector_hash)
    key = (row.get("district", ""), row.get("projector_hash", ""))
    seen = st.session_state.setdefault("_pj_registry_keys", set())
    if key in seen:
        return False
    seen.add(key)
    PROJECTORS_DIR.mkdir(parents=True, exist_ok=True)
    # atomic append jsonl
    with tempfile.NamedTemporaryFile("w", delete=False, dir=PROJECTORS_DIR, encoding="utf-8") as tmp:
        tmp.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    with open(PJ_REG_PATH, "a", encoding="utf-8") as final:
        with open(tmp_name, "r", encoding="utf-8") as src:
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
    payload = {
        "schema_version": "1.0.0",
        "written_at_utc": _utc_iso(),
        "blocks": {"3": P3}
    }
    pj_hash, pj_size = _atomic_write_json(pj_path, payload)
    return {"path": str(pj_path), "projector_hash": pj_hash, "bytes": pj_size, "lane_mask_k3": lane_mask_k3[:]}

def _validate_projector_file(pj_path: str) -> dict:
    # Use same validator used by overlap (raises ValueError with P3_* on fail)
    cfg_file = _cfg_from_policy("projected(file)", pj_path)
    _, meta = projector_choose_active(cfg_file, boundaries)
    return meta

def _simulate_overlap_with_cfg(cfg_forced):
    """
    Run a FILE overlap without touching the policy widget. Populates:
      run_ctx, overlap_out, residual_tags, overlap_policy_label
    """
    # Bind projector (fail-fast)
    P_active, meta = projector_choose_active(cfg_forced, boundaries)

    # Context
    d3 = meta.get("d3") if "d3" in meta else (boundaries.blocks.__root__.get("3") or [])
    n3 = meta.get("n3") if "n3" in meta else (len(d3[0]) if (d3 and d3[0]) else 0)
    lane_mask = meta.get("lane_mask", [])
    mode = meta.get("mode", "projected(file)")

    # Compute strict residual R3 = H2@d3 XOR (C3 XOR I3)
    H_used = st.session_state.get("overlap_H") or _load_h_local()
    H2 = (H_used.blocks.__root__.get("2") or [])
    C3 = (cmap.blocks.__root__.get("3") or [])
    I3 = eye(len(C3)) if C3 else []

    def _xor(A, B):
        if not A:
            return [r[:] for r in (B or [])]
        if not B:
            return [r[:] for r in (A or [])]
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(len(A[0]))] for i in range(len(A))]

    def _is_zero(M):
        return (not M) or all(all((x & 1) == 0 for x in row) for row in M)

    R3_strict = _xor(mul(H2, d3), _xor(C3, I3)) if (H2 and d3 and C3) else []
    R3_proj = mul(R3_strict, P_active) if (R3_strict and P_active) else []

    def _residual_tag(R, lm):
        if not R or not lm:
            return "none"
        rows, cols = len(R), len(R[0])
        lanes_idx = [j for j, m in enumerate(lm) if m]
        ker_idx = [j for j, m in enumerate(lm) if not m]
        def _col_nonzero(j): return any(R[i][j] & 1 for i in range(rows))
        lanes_resid = any(_col_nonzero(j) for j in lanes_idx) if lanes_idx else False
        ker_resid = any(_col_nonzero(j) for j in ker_idx) if ker_idx else False
        if not lanes_resid and not ker_resid:
            return "none"
        if lanes_resid and not ker_resid:
            return "lanes"
        if ker_resid and not lanes_resid:
            return "ker"
        return "mixed"

    tag_strict = _residual_tag(R3_strict, lane_mask)
    tag_proj = _residual_tag(R3_proj, lane_mask)

    out = {"3": {"eq": bool(_is_zero(R3_proj)), "n_k": n3}, "2": {"eq": True}}
    # Persist exactly like run_overlap
    st.session_state["overlap_out"] = out
    st.session_state["residual_tags"] = {"strict": tag_strict, "projected": tag_proj}
    st.session_state["overlap_cfg"] = cfg_forced
    st.session_state["overlap_policy_label"] = policy_label_from_cfg(cfg_forced)
    st.session_state["run_ctx"] = {
        "policy_tag": policy_label_from_cfg(cfg_forced),
        "mode": mode,
        "d3": d3,
        "n3": n3,
        "lane_mask_k3": lane_mask,
        "P_active": P_active,
        "projector_filename": meta.get("projector_filename", ""),
        "projector_hash": meta.get("projector_hash", ""),
        "projector_consistent_with_d": meta.get("projector_consistent_with_d", None),
        "errors": [],
    }

# ---------------------------- Projector Freezer (AUTO → FILE) ----------------------------
with st.expander("Projector Freezer (AUTO → FILE, no UI flip)"):
    _ss = st.session_state
    _rc = _ss.get("run_ctx") or {}
    _di = _ss.get("_district_info") or {}
    district_id = _di.get("district_id", "UNKNOWN")

    # Small guard to prevent false "stale" context
    _fixture_nonce = _ss.get("_fixture_nonce")
    fresh = bool(_rc and _rc.get("fixture_nonce") == _fixture_nonce)
    st.caption(f"ctx freshness: {_rc.get('fixture_nonce')} vs {_fixture_nonce} → {'fresh' if fresh else 'stale'}")
    if not fresh:
        st.info("Context is stale. Run Overlap to refresh.")
        # Early exit if stale
        # (Optional: you could skip the rest of the block here if desired)
        # return

    # Compute eligibility + freshness WITHOUT stopping the app
    last_out_3 = ((_ss.get("overlap_out") or {}).get("3", {}) or {})
    k3_green = bool(last_out_3.get("eq", False))
    has_mask_and_d3 = bool((_rc.get("lane_mask_k3") or [])) and bool((_rc.get("d3") or []))
    in_auto = (_rc.get("mode") == "projected(auto)")
    # Use the fresh variable for additional certainty if needed
    # (But since we've checked freshness above, proceed as usual)
    fresh = bool(_rc) and (_rc.get("fixture_nonce") == _ss.get("_fixture_nonce"))

    elig_freeze = in_auto and has_mask_and_d3 and k3_green and fresh

    if not fresh:
        st.info("Context is stale. Run Overlap to refresh.")
    elif not in_auto:
        st.info("Freezer only applies in projected(auto) mode.")
    elif not has_mask_and_d3:
        st.info("Missing d3 and/or lane_mask_k3 in run context. Run Overlap.")
    elif not k3_green:
        st.info("k=3 is not green (eq==False). Run Overlap until k=3 passes.")

    pj_basename = st.text_input(
        "Filename",
        value=f"projector_{district_id}.json",
        key="pj_freeze_name_main",
    )
    overwrite_ok = st.checkbox("Overwrite if exists", value=False, key="pj_overwrite_ok_main")

    # Button is simply disabled if not eligible; no st.stop() in render path
    if st.button("Freeze Π → FILE & re-run", key="btn_freeze_pj_main", disabled=not elig_freeze):
        try:
            # Re-check freshness just before acting
            _rc = _ss.get("run_ctx") or {}
            _fixture_nonce = _ss.get("_fixture_nonce")
            if not (_rc and _rc.get("fixture_nonce") == _fixture_nonce and _rc.get("d3") and _rc.get("lane_mask_k3")):
                raise RuntimeError("Context changed. Run Overlap again and retry.")

            pj_path = PROJECTORS_DIR / pj_basename
            if pj_path.exists() and not overwrite_ok:
                raise RuntimeError("Projector file already exists. Enable overwrite or choose a new name.")

            # Truth lane mask from SSOT d3
            d3_snap = _rc.get("d3") or []
            n3 = int(_rc.get("n3") or 0)
            if n3 != len(_rc.get("lane_mask_k3") or []):
                # defensive (do not stop the page)
                n3 = len(d3_snap[0]) if d3_snap else 0
            lm_truth = [1 if any(d3_snap[i][j] & 1 for i in range(len(d3_snap))) else 0 for j in range(n3)]
            if len(lm_truth) != n3:
                raise RuntimeError(f"lane_mask length {len(lm_truth)} != n3 {n3}")

            # Keep run_ctx authoritative
            if lm_truth != (_rc.get("lane_mask_k3") or []):
                _rc["lane_mask_k3"] = lm_truth
                _ss["run_ctx"] = _rc

            # Build diagonal Π and validate
            P_freeze = [[1 if (i == j and lm_truth[j]) else 0 for j in range(n3)] for i in range(n3)]
            validate_projector_file_strict(P_freeze, n3=n3, lane_mask=lm_truth)

            # Save projector (atomic) and hash
            payload = {"schema_version": "1.0.0", "blocks": {"3": P_freeze}}
            pj_hash, _ = _atomic_write_json(pj_path, payload)

            # Flip to FILE
            cfg_active.setdefault("source", {})["3"] = "file"
            cfg_active.setdefault("projector_files", {})["3"] = pj_path.as_posix()

            # Bump fixtures / clear caches (no st.stop)
            if "_mark_fixtures_changed" in globals():
                _mark_fixtures_changed()
            else:
                _ss["_fixture_nonce"] = int(_ss.get("_fixture_nonce", 0)) + 1
                for k in ("overlap_out", "residual_tags", "overlap_cfg", "overlap_policy_label"):
                    _ss.pop(k, None)

            # Re-run overlap (now FILE)
            run_overlap()

            # Force cert write
            _ss["should_write_cert"] = True
            _ss.pop("_last_cert_write_key", None)

            st.success(f"Π saved → {pj_path.name} · {pj_hash[:12]}… and switched to FILE.")
        except Exception as e:
            st.error(f"Freeze failed: {e}")




  
    # ======================== Parity: import/export & queue ========================
from pathlib import Path
from datetime import datetime, timezone
import json as _json, os, tempfile

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
import pandas as pd

def _cfg_from_run_ctx(rc: dict) -> dict | None:
    """Mirror the active policy from the last Overlap run."""
    mode = (rc or {}).get("mode", "strict")
    if mode == "strict":
        return None
    cfg = cfg_projected_base()
    if mode == "projected(auto)":
        cfg["source"]["3"] = "auto"
        cfg.setdefault("projector_files", {}).setdefault("3", "projector_D3.json")  # placeholder
        return cfg
    if mode == "projected(file)":
        cfg["source"]["3"] = "file"
        pj = (rc or {}).get("projector_filename", "")
        if pj:
            cfg.setdefault("projector_files", {})["3"] = pj
        return cfg
    return None

def _and_pair(a: bool | None, b: bool | None) -> bool | None:
    if a is None or b is None:
        return None
    return bool(a) and bool(b)

def _one_leg(boundaries_obj, cmap_obj, H_obj, projection_cfg: dict | None):
    """Run overlap on one fixture; returns {'2':{'eq':..}, '3':{'eq':..}}"""
    if projection_cfg is None:
        return overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_obj)
    # Validate FILE Π early (shape/diag/idempotence/lane); raises on failure
    _P, _meta = projector_choose_active(projection_cfg, boundaries_obj)
    return overlap_gate.overlap_check(boundaries_obj, cmap_obj, H_obj, projection_config=projection_cfg)

def _emoji(v):
    if v is None: return "—"
    return "✅" if bool(v) else "❌"

with safe_expander("Parity: run suite (mirrors active policy)"):
    pairs = st.session_state.get("parity_pairs", []) or []
    rc    = st.session_state.get("run_ctx", {}) or {}
    ib    = st.session_state.get("_inputs_block", {}) or {}

    if not pairs:
        st.info("No parity pairs queued. Use the import/queue controls above.")
    else:
        policy_tag = rc.get("policy_tag", policy_label_from_cfg(cfg_strict()))
        pj_hash    = rc.get("projector_hash","") if rc.get("mode","").startswith("projected") else ""
        cfg_proj   = _cfg_from_run_ctx(rc)  # None if strict

        if st.button("Run Parity Suite", key="btn_run_parity"):
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

            payload = {
                "schema_version": PARITY_SCHEMA_VERSION,
                "written_at_utc": _iso_utc_now(),
                "app_version": getattr(hashes, "APP_VERSION", "v0.1-core"),
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
                    st.download_button("Download parity_report.json", f, file_name="parity_report.json", key="dl_parity_report")
                if errors:
                    st.warning("Some pairs had issues; details recorded in the report’s `errors` field.")
                st.session_state["parity_last_report_pairs"] = report_pairs
            except Exception as e:
                st.error(f"Could not write parity_report.json: {e}")

        # Render a compact table only after a successful run this session
        last_pairs = st.session_state.get("parity_last_report_pairs")
        if last_pairs:
            df = pd.DataFrame([
                {
                    "Pair": p["label"],
                    "Strict k2": _emoji(p["strict"]["k2"]),
                    "Strict k3": _emoji(p["strict"]["k3"]),
                    "Projected k2": _emoji(p["projected"]["k2"]),
                    "Projected k3": _emoji(p["projected"]["k3"]),
                } for p in last_pairs
            ], columns=["Pair", "Strict k2", "Strict k3", "Projected k2", "Projected k3"])
            st.caption("Parity summary")
            st.dataframe(df, use_container_width=True)
            try:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download parity_summary.csv", csv_bytes, file_name="parity_summary.csv")
            except Exception:
                pass
# honor force flag from freezer / other flows
if st.session_state.pop("should_write_cert", False):
    st.session_state.pop("_last_cert_write_key", None)

# ------------------------ Cert writer (central, SSOT-only, with A/B embed) ------------------------
st.divider()
st.caption("Cert & provenance")

from pathlib import Path
import platform, os, json as _json
from datetime import datetime, timezone
import hashlib  # Added import for fallback run_id hash

# Ensure bundles directory exists
try:
    BUNDLES_DIR  # noqa
except NameError:
    BUNDLES_DIR = Path("bundles")
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

LAB_SCHEMA_VERSION = "1.0.0"

# --- Helper functions ---
def _assert_cert_invariants(cert: dict) -> None:
    for key in ("identity","policy","inputs","diagnostics","checks","signatures","residual_tags","promotion","artifact_hashes"):
        if key not in cert:
            raise ValueError(f"CERT_INVAR:key-missing:{key}")
    ident = cert["identity"] or {}; policy = cert["policy"] or {}; inputs = cert["inputs"] or {}
    checks = cert["checks"] or {}; arts = cert["artifact_hashes"] or {}
    # identity
    for k in ("district_id","run_id","timestamp"):
        if not str(ident.get(k,"")).strip():
            raise ValueError(f"CERT_INVAR:identity-missing:{k}")
    # inputs hashes exist
    for k in ("boundaries_hash","C_hash","H_hash","U_hash","shapes_hash"):
        if not isinstance(inputs.get(k,""), str) or inputs.get(k,"")=="":
            raise ValueError(f"CERT_INVAR:inputs-hash-missing:{k}")
    # mirror artifact hashes
    for k in ("boundaries_hash","C_hash","H_hash","U_hash"):
        if arts.get(k,"") != inputs.get(k,""):
            raise ValueError(f"CERT_INVAR:artifact-hash-mismatch:{k}")
    # dims
    dims = inputs.get("dims") or {}
    if not (isinstance(dims.get("n2"), int) and isinstance(dims.get("n3"), int)):
        raise ValueError("CERT_INVAR:inputs-dims-missing:n2-n3")
    # policy tag + ker_guard discipline
    ptag = str(policy.get("policy_tag") or policy.get("label") or "").strip()
    if not ptag:
        raise ValueError("CERT_INVAR:policy-tag-missing")
    is_strict    = (ptag == "strict")
    is_file      = ptag.startswith("projected(file)") or ptag.startswith("projected(columns@k=3,file)")
    is_auto      = ptag.startswith("projected(auto)") or ptag.startswith("projected(columns@k=3,auto)")
    kg = checks.get("ker_guard", "")
    if is_strict and kg != "enforced":
        raise ValueError("CERT_INVAR:ker-guard-should-be-enforced-for-strict")
    if (is_file or is_auto) and kg != "off":
        raise ValueError("CERT_INVAR:ker-guard-should-be-off-for-projected")
    # projector fields
    pj_hash = policy.get("projector_hash", "")
    pj_file = policy.get("projector_filename", "") or ""
    pj_cons = policy.get("projector_consistent_with_d", None)
    if is_strict and (pj_file or pj_hash or (pj_cons is True)):
        raise ValueError("CERT_INVAR:strict-must-not-carry-projector-fields")
    if is_file:
        if not pj_file:
            raise ValueError("CERT_INVAR:file-mode-missing-projector_filename")
        if pj_cons is not True:
            raise ValueError("CERT_INVAR:file-mode-projector-not-consistent")
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

# --- Debounce (allow freezer to force) ---
if st.session_state.pop("should_write_cert", False):
    st.session_state.pop("_last_cert_write_key", None)

# --- Guard: run context + inputs SSOT ---
if not (_rc and _out and _ib):
    st.info("Run Overlap first to enable cert writing.")
else:
    # --- Write-key ---
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

        # --- Diagnostics ---
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
            if not lm:  return row[:]
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

        # --- Identity ---
        district_id = _di.get("district_id", st.session_state.get("district_id","UNKNOWN"))
        run_ts = getattr(hashes, "timestamp_iso_lisbon", lambda: datetime.now(timezone.utc).isoformat())()
        policy_now = _rc.get("policy_tag", policy_label_from_cfg(cfg_active))

        run_id = st.session_state.get("last_run_id")
        if not run_id:
            seed = "".join(str((_ib or {}).get(k,"")) for k in ("boundaries_hash","C_hash","H_hash","U_hash"))
            run_id = getattr(hashes,"run_id",lambda a,b: hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:12])(seed, run_ts)
            st.session_state["last_run_id"] = run_id

        identity_block = {
            "district_id": district_id, "run_id": run_id, "timestamp": run_ts,
            "app_version": getattr(hashes,"APP_VERSION","v0.1-core"),
            "python_version": f"python-{platform.python_version()}",
        }

        # --- Policy (mirror RC; strict clamps) ---
        policy_block = {
            "label": policy_now,
            "policy_tag": policy_now,
            "enabled_layers": cfg_active.get("enabled_layers", []),
            "modes": cfg_active.get("modes", {}),
            "source": (_rc.get("source") or {}),  # verbatim copy
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

        # Checks (from SSOT), then n_k fill
        residual_tags = st.session_state.get("residual_tags", {}) or {}
        is_strict_mode = (_rc.get("mode") == "strict")
        checks_block = {
            **(_out or {}),
            "grid":  bool((_out or {}).get("grid", True)),
            "fence": bool((_out or {}).get("fence", True)),
            "ker_guard": ("enforced" if is_strict_mode else "off"),
        }

        # --- Inputs ---
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

        # Promotion (simple)
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

        # Artifacts mirror inputs (+ optional projector file sha)
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
                    import hashlib as _hl
                    pf = _rc.get("projector_filename","")
                    if pf and os.path.exists(pf):
                        with open(pf,"rb") as f: pj_sha = _hl.sha256(f.read()).hexdigest()
                except Exception:
                    pj_sha = None
            if pj_sha:
                policy_block["projector_file_sha256"] = pj_sha
                artifact_hashes["projector_file_sha256"] = pj_sha

        # Assemble
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
            "python_version": f"python-{platform.python_version()}",
        }

        # Optional A/B embed (fresh only)
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
                "policy_tag":"strict", "ker_guard":"enforced",
                "inputs":{"filenames": inputs_block_payload.get("filenames", {})},
                "lane_mask_k3": lane_mask,
                "lane_vec_H2d3": strict_ctx.get("lane_vec_H2d3"),
                "lane_vec_C3plusI3": strict_ctx.get("lane_vec_C3plusI3"),
                "pass_vec": _pv(strict_ctx.get("out", {})),
                "out": strict_ctx.get("out", {}),
            }
            proj_snap = {
                "policy_tag": proj_tag, "ker_guard":"off",
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

                # --- read-only SSOT pulls ---
        _rc = st.session_state.get("run_ctx") or {}
        lm = list(_rc.get("lane_mask_k3") or [])
        n3 = int(_rc.get("n3") or 0)
        
        # Defensive: lane mask must match n3
        assert len(lm) == n3, "cert: lane_mask_k3 length mismatch with n3"
        
        # Use SSOT mask and source verbatim
        diagnostics_block = {
            **diagnostics_block,             # your existing fields
            "lane_mask_k3": lm,
        }
        policy_block["source"] = (_rc.get("source") or {})   # verbatim; no defaults
        
        # (A/B embedding should already be done by now if applicable)
        
        # --- invariants + hash ---
        _assert_cert_invariants(cert_payload)
        cert_payload.setdefault("integrity", {})
        cert_payload["integrity"]["content_hash"] = hash_json(cert_payload)
        full_hash = cert_payload["integrity"]["content_hash"]
        
        # proceed to write (package writer or atomic fallback)...


        # Write (prefer package)
        cert_path = None
        try:
            result = export_mod.write_cert_json(cert_payload)
            cert_path, full_hash = (result if isinstance(result,(list,tuple)) and len(result)>=2 else (result, full_hash))
        except Exception:
            outdir = Path("certs")
            outdir.mkdir(parents=True, exist_ok=True)
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
        st.session_state["last_run_id"] = identity_block["run_id"]
        st.success(f"Cert written → `{cert_path}` · {full_hash[:12]}…")
        st.caption(f"Embedded A/B → {cert_payload.get('ab_pair_tag','A/B')}" if cert_payload.get("ab_embedded") else "Embedded A/B → —")

        # --- Bundle (cert + extras) ---
        with st.expander("Bundle (cert + extras)"):
            extras = ["policy.json",
                      "reports/residual.json","reports/parity_report.json","reports/coverage_sampling.csv",
                      "logs/gallery.jsonl","logs/witnesses.jsonl"]
            if _rc.get("mode")=="projected(file)" and _rc.get("projector_filename"):
                extras.append(_rc.get("projector_filename"))
            if st.button("Build Cert Bundle", key="build_cert_bundle_btn_final"):
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

# (Already imported: from datetime import datetime)

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






# ───────────────────────── Imports and Constants ─────────────────────────
from pathlib import Path
import os
import json
import tempfile
import zipfile
import shutil
import platform
import csv
import hashlib
import secrets
import streamlit as st
from datetime import datetime, timezone

# Directory constants (respect existing globals if present)
CERTS_DIR = Path(globals().get("CERTS_DIR", "certs"))
PROJECTORS_DIR = Path(globals().get("PROJECTORS_DIR", "projectors"))
LOGS_DIR = Path(globals().get("LOGS_DIR", "logs"))
REPORTS_DIR = Path(globals().get("REPORTS_DIR", "reports"))
BUNDLES_DIR = Path(globals().get("BUNDLES_DIR", "bundles"))
# Ensure BUNDLES_DIR exists once
BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────── Utils ─────────────────────────
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
    if not root.exists():
        return 0
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n

# Local fallback for run_id generation
def generate_run_id_from_seed(seed: str, ts: str) -> str:
    return hashlib.sha256(f"{seed}|{ts}".encode("utf-8")).hexdigest()[:12]

# ───────────────────────── Main Functional Logic ─────────────────────────

def build_inputs_bundle(*, inputs_block: dict, run_ctx: dict, district_id: str, run_id: str, policy_tag: str) -> str:
    """
    Creates a ZIP with manifest.json and input files.
    """
    APP_VERSION_LOCAL = globals().get("APP_VERSION_STR", getattr(hashes, "APP_VERSION", "v0.1-core"))
    PY_VERSION_LOCAL = globals().get("PY_VERSION_STR", f"python-{platform.python_version()}")

    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    fns = (inputs_block.get("filenames") or {})
    fnames = {
        "boundaries": fns.get("boundaries", st.session_state.get("fname_boundaries", "boundaries.json")),
        "C": fns.get("C", st.session_state.get("fname_cmap", "cmap.json")),
        "H": fns.get("H", st.session_state.get("fname_h", "H.json")),
        "U": fns.get("U", st.session_state.get("fname_shapes", "shapes.json")),
        "projector": fns.get("projector", run_ctx.get("projector_filename", "") or ""),
    }

    hashes_block = {
        "boundaries_hash": inputs_block.get("boundaries_hash", ""),
        "C_hash": inputs_block.get("C_hash", ""),
        "H_hash": inputs_block.get("H_hash", ""),
        "U_hash": inputs_block.get("U_hash", ""),
        "shapes_hash": inputs_block.get("shapes_hash", ""),
    }

    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "timestamp": _utc_iso_z(),
        "app_version": APP_VERSION_LOCAL,
        "python_version": PY_VERSION_LOCAL,
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
            for label, fp in fnames.items():
                if not fp:
                    continue
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
    Builds a ZIP with certs, referenced projectors, logs, reports, metadata, and index.
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

    # Prepare sets and rows
    proj_refs = set()
    districts = set()
    index_rows = []
    manifest_files = []

    for p, cert in parsed:
        ident = cert.get("identity") or {}
        pol = cert.get("policy") or {}
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

        hashes_flat = {
            "boundaries_hash": inputs.get("boundaries_hash"),
            "C_hash": inputs.get("C_hash"),
            "H_hash": inputs.get("H_hash"),
            "U_hash": inputs.get("U_hash"),
        }
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
                missing_projectors.append({
                    "filename": _rel(pj_path),
                    "referenced_by": "certs/* (various)"
                })
                continue
        projectors.append({
            "path": _rel(pj_path),
            "sha256": _sha256_file(pj_path),
            "size": pj_path.stat().st_size,
        })

    # Logs and reports
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

    # Manifest
    app_ver = getattr(hashes, "APP_VERSION", "v0.1-core")
    py_ver = f"python-{platform.python_version()}"
    districts_sorted = sorted(districts)

    manifest = {
        "schema_version": "1.0.0",
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
                "parity": int(any(x["path"].endswith("parity_report.json") for x in reports_list)),
                "coverage": int(any(x["path"].endswith("coverage_sampling.csv") for x in reports_list)),
                "perturb": int(any(x["path"].endswith("perturbation_sanity.csv") for x in reports_list)),
                "fence": int(any(x["path"].endswith("fence_stress.csv") for x in reports_list)),
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

    # Create cert_index.csv in-memory
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

    # Create ZIP archive (atomic)
    tag = next(iter(districts_sorted)) if len(districts_sorted) == 1 else "MULTI"
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
            try:
                tmpzip.unlink()
            except Exception:
                pass

    if len(index_rows) != manifest["counts"]["certs"]:
        st.warning("Index count does not match manifest cert count (investigate).")
    return str(zpath)

# Flush Workspace
def flush_workspace(*, delete_projectors: bool=False) -> dict:
    """
    Remove artifacts, reset session state, recreate empty dirs. Keeps inputs intact.
    """
    summary = {
        "when": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
        "_projector_cache", "_projector_cache_ab", "parity_pairs", "selftests_snapshot"
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
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    salt = secrets.token_hex(2).upper()
    token = f"FLUSH-{ts}-{salt}"
    ckey = hashlib.sha256((ts + salt).encode("utf-8")).hexdigest()

    st.session_state["_composite_cache_key"] = ckey
    st.session_state["_last_flush_token"] = token
    summary["token"] = token
    summary["composite_cache_key_short"] = ckey[:12]
    return summary

# ───────────────────────── UI Components ─────────────────────────

# Small helpers to avoid key collisions & deprecations
def _mkkey(ns: str, name: str) -> str:
    return f"{ns}__{name}"

def _fmt_ts(ts_float: float) -> str:
    # timezone-aware, avoids DeprecationWarning
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(ts_float, timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return ""

EXPORTS_NS = "exports_v2"  # change once if you ever need to re-namespace

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
                    # fallback run_id using sha256
                    run_id = hashlib.sha256(f"{seed_str}|{ts}".encode("utf-8")).hexdigest()[:12]
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
        if st.button("Quick Reset (session only)", key="btn_quick_reset_session"):
            # session-only reset + bump nonce to invalidate stale run_ctx
            _mark_fixtures_changed()
            st.success("Session caches cleared. Run Overlap again.")

        inc_pj = st.checkbox("Also remove projectors (full flush)", value=False, key="flush_inc_pj_final")
        if st.button("Full Flush (disk + session)", key="btn_full_flush_everything"):
            try:
                info = flush_workspace(delete_projectors=inc_pj)
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
