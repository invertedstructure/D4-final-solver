# ────────────────────────────── IMPORTS ──────────────────────────────
# Standard library
import sys, os, json, csv, hashlib, platform, zipfile, tempfile, shutil
import importlib.util, types, pathlib
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

# Third-party
import streamlit as st

# ────────────────────────────── APP CONFIG ──────────────────────────────
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# ────────────────────────────── PACKAGE LOADER ──────────────────────────────
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

# (Re)load core modules (single loader; avoid duplicates)
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

# legacy alias for older references
if "io" not in globals():
    io = otio

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")

# ────────────────────────────── HEADER ──────────────────────────────
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")
st.caption(f"overlap_gate loaded from: {getattr(overlap_gate, '__file__', '<none>')}")
st.caption(f"projector loaded from: {getattr(projector, '__file__', '<none>')}")

# ────────────────────────────── CONSTANTS / DIRS ──────────────────────────────
LAB_SCHEMA_VERSION = "1.0.0"
APP_VERSION_STR    = str(APP_VERSION)
PY_VERSION_STR     = f"python-{platform.python_version()}"

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

BUNDLES_DIR = Path(DIRS["bundles"]); BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── SESSION GUARDS ──────────────────────────────
def _ensure_session_keys():
    st.session_state.setdefault("_inputs_block", {})
    st.session_state.setdefault("_district_info", {})
    st.session_state.setdefault("run_ctx", {})
    st.session_state.setdefault("overlap_out", {})
    st.session_state.setdefault("overlap_H", None)
    st.session_state.setdefault("residual_tags", {})

_ensure_session_keys()

# === Step 1: Core helpers (config, hashing, filenames, projector selection) ===
import json as _json
import io as _io
import os as _os
from pathlib import Path as _Path

# -- tiny hash helpers (bytes + json-obj) --------------------------------------
def _sha256_hex_bytes(b: bytes) -> str:
    import hashlib as _hh
    return _hh.sha256(b).hexdigest()

def _sha256_hex_obj(obj) -> str:
    import hashlib as _hh, json as _jj
    blob = _jj.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hh.sha256(blob).hexdigest()

# -- file-uploader helpers ------------------------------------------------------
def read_json_file(upload):
    """
    Accepts a Streamlit uploaded file, a str/Path, or already-parsed dict.
    Returns a dict or None.
    """
    if upload is None:
        return None
    if isinstance(upload, (str, _Path)):
        with open(str(upload), "r", encoding="utf-8") as f:
            return _json.load(f)
    if isinstance(upload, dict):
        return upload
    # Streamlit UploadedFile
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
    except Exception:
        pass

# -- lane mask + signature utilities -------------------------------------------
def _lane_mask_from_d3(boundaries) -> list[int]:
    """
    Best-effort lane mask from d3.
    Strategy:
      1) If boundaries exposes lane_mask_k3, use it.
      2) Else: use the bottom row of d3 (common convention in this app).
      3) Else: [] (unknown).
    """
    try:
        # 1) attribute or dict key
        if hasattr(boundaries, "lane_mask_k3"):
            lm = getattr(boundaries, "lane_mask_k3")
            if isinstance(lm, list) and all(isinstance(x, (int, bool)) for x in lm):
                return [int(bool(x)) for x in lm]
        bd = boundaries.dict() if hasattr(boundaries, "dict") else {}
        lm = bd.get("lane_mask_k3")
        if isinstance(lm, list) and all(isinstance(x, (int, bool)) for x in lm):
            return [int(bool(x)) for x in lm]
    except Exception:
        pass

    # 2) bottom row of d3
    try:
        d3 = (boundaries.blocks.__root__.get("3") or [])
        if d3 and d3[-1]:
            return [int(r & 1) for r in d3[-1]]
    except Exception:
        pass

    # 3) default: empty (unknown)
    return []

def _district_signature(lane_mask: list[int], r: int, c: int) -> str:
    pat = "".join("1" if int(x) else "0" for x in (lane_mask or []))
    return f"{r}x{c}:{pat}"

# -- cfg builders + label -------------------------------------------------------
def cfg_strict() -> dict:
    return {
        "enabled_layers": [],        # nothing projected
        "modes": {},
        "source": {},                # no source in strict
        "projector_files": {},       # empty
    }

def cfg_projected_base() -> dict:
    return {
        "enabled_layers": [3],       # we project k=3
        "modes": {},
        "source": {"3": "auto"},     # default AUTO
        "projector_files": {},       # filled only in FILE mode
    }

def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src = (cfg.get("source", {}) or {}).get("3", "auto")
    if src == "file":
        return "projected(columns@k=3,file)"
    return "projected(columns@k=3,auto)"

# -- tiny GF(2) ops (local, small + fast) --------------------------------------
def _eye(n: int):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def _mul_gf2(A, B):
    if not A or not B or not A[0] or not B[0]:
        return []
    r, k = len(A), len(A[0])
    k2, c = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"dim mismatch for GF(2) multiply: {r}x{k} @ {k2}x{c}")
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
    # P·P == P
    try:
        PP = _mul_gf2(P, P)
        return PP == P
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

# -- projector chooser (strict/auto/file)  -------------------------------------
class _P3Error(ValueError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code

def _read_projector_matrix(path_str: str):
    """
    Accepts either:
      {"blocks":{"3":[[...]]}}  or just  [[...]]
    Returns the 2D list for k=3.
    """
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
    """
    Returns (P_active, meta) where:
      P_active: 2D list (nxn) for k=3 (or [] for strict)
      meta: {
        "d3", "n3", "lane_mask", "mode",
        "projector_filename", "projector_hash",
        "projector_consistent_with_d": True|False|None
      }
    Raises _P3Error on FILE validation issues (fail-fast).
    """
    d3 = (boundaries.blocks.__root__.get("3") or [])
    n3 = len(d3[0]) if (d3 and d3[0]) else 0
    lane_mask = _lane_mask_from_d3(boundaries)
    mode = "strict"
    P_active = []
    pj_filename = ""
    pj_hash = ""
    pj_consistent = None

    # strict?
    if not cfg_active or not cfg_active.get("enabled_layers"):
        return P_active, {
            "d3": d3, "n3": n3, "lane_mask": lane_mask, "mode": mode,
            "projector_filename": pj_filename, "projector_hash": pj_hash,
            "projector_consistent_with_d": pj_consistent,
        }

    source = (cfg_active.get("source", {}) or {}).get("3", "auto")
    mode = "projected(auto)" if source == "auto" else "projected(file)"

    if source == "auto":
        # AUTO = diag(lane_mask)
        diag = (lane_mask if lane_mask else [1]*n3)
        P_active = [[1 if i == j and diag[j] else 0 for j in range(n3)] for i in range(n3)]
        pj_hash = _sha256_hex_obj(P_active)
        pj_consistent = True  # by construction it follows our lane mask
        return P_active, {
            "d3": d3, "n3": n3, "lane_mask": lane_mask, "mode": mode,
            "projector_filename": "", "projector_hash": pj_hash,
            "projector_consistent_with_d": pj_consistent,
        }

    # FILE
    pj_filename = (cfg_active.get("projector_files", {}) or {}).get("3", "")
    if not pj_filename:
        raise _P3Error("P3_SHAPE", "no projector file provided for file-mode")

    P = _read_projector_matrix(pj_filename)

    # shape check
    m = len(P) or 0
    n = len(P[0]) if m else 0
    if n3 == 0 or m != n3 or n != n3:
        raise _P3Error("P3_SHAPE", f"expected {n3}x{n3}, got {m}x{n}")

    # idempotent check
    if not _is_idempotent_gf2(P):
        raise _P3Error("P3_IDEMP", "P is not idempotent over GF(2)")

    # diagonal check
    if not _is_diagonal(P):
        raise _P3Error("P3_DIAGONAL", "P has off-diagonal non-zeros")

    # lane mask match (only if we could infer a mask)
    pj_diag = _diag(P)
    if lane_mask and pj_diag != [int(x) for x in lane_mask]:
        raise _P3Error("P3_LANE_MISMATCH", f"diag(P) != lane_mask(d3) → {pj_diag} vs {lane_mask}")

    pj_hash = _sha256_hex_obj(P)
    pj_consistent = True if lane_mask else None

    return P, {
        "d3": d3, "n3": n3, "lane_mask": lane_mask, "mode": mode,
        "projector_filename": pj_filename, "projector_hash": pj_hash,
        "projector_consistent_with_d": pj_consistent,
    }

# -- atomic JSONL append (small, safe enough for single-writer Streamlit) ------
def atomic_append_jsonl(path: _Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    # Best-effort atomic-ish: write a tmp then append; if append fails we don't corrupt the main file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        _os.fsync(f.fileno())
    try:
        with open(path, "a", encoding="utf-8") as g:
            with open(tmp, "r", encoding="utf-8") as f:
                g.write(f.read())
                g.flush()
                _os.fsync(g.fileno())
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ────────────────────────────── SMALL HELPERS ──────────────────────────────
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def hash_json(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def _sha256_hex_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _sha256_hex_obj(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_hex_bytes(s)

def _stamp_filename(state_key: str, fobj):
    """Remember uploaded filename for later provenance."""
    if fobj is not None:
        st.session_state[state_key] = getattr(fobj, "name", "")
    else:
        st.session_state.pop(state_key, None)

def read_json_file(uploaded):
    if not uploaded:
        return None
    try:
        return json.load(uploaded)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return None

def atomic_write_json(path: str | Path, obj: dict):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        blob = json.dumps(obj, sort_keys=True, indent=2).encode()
        f.write(blob); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_append_jsonl(path: str | Path, row: dict):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    with open(tmp, "wb") as f:
        f.write(line.encode()); f.flush(); os.fsync(f.fileno())
    # append safely
    with open(path, "ab") as out:
        with open(tmp, "rb") as src:
            data = src.read()
        out.write(data); out.flush(); os.fsync(out.fileno())
    try: tmp.unlink(missing_ok=True)
    except Exception: pass

def _atomic_write_csv(path: Path, header: list[str], rows: list[list], meta_comment_lines: list[str] | None = None):
    tmp = Path(str(path) + ".tmp"); tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        if meta_comment_lines:
            w.writerow([f"# {k}" for k in meta_comment_lines])
        w.writerow(header)
        w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _lane_mask_from_d3(boundaries) -> list[int]:
    try:
        d3 = boundaries.blocks.__root__.get("3")
    except Exception:
        d3 = None
    if not d3 or not d3[0]:
        return []
    cols = len(d3[0])
    return [1 if any((row[j] & 1) for row in d3) else 0 for j in range(cols)]

def _district_signature(mask: list[int], r: int, c: int) -> str:
    payload = f"k3:{''.join(str(int(x)) for x in (mask or []))}|r{r}|c{c}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]

# In case you use it later
DISTRICT_MAP = {
    # raw-bytes sha256(boundaries.json) → label
    # "…": "D1",
    # "…": "D2",
    # "…": "D3",
    # "…": "D4",
}

def hash_matrix_norm(M) -> str:
    if not M:
        return hash_json([])
    norm = [[int(x) & 1 for x in row] for row in M]
    return hash_json(norm)

def _zip_arcname(abspath: str) -> str:
    p = Path(abspath)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return p.name

def build_cert_bundle(*, district_id: str, policy_tag: str, cert_path: str,
                      content_hash: str | None = None, extras: list[str] | None = None) -> str:
    """Cross-device safe writer for bundles/overlap_bundle__{district}__{policy}__{hash}.zip"""
    cert_p = Path(cert_path)
    if not cert_p.exists():
        raise FileNotFoundError(f"Cert not found: {cert_path}")

    with open(cert_p, "r", encoding="utf-8") as f:
        cert = json.load(f)
    if not content_hash:
        content_hash = ((cert.get("integrity") or {}).get("content_hash") or "")
    suffix = content_hash[:12] if content_hash else "nohash"
    safe_policy = (policy_tag or cert.get("policy", {}).get("policy_tag", "policy")).replace("/", "_").replace(" ", "_")
    zpath = BUNDLES_DIR / f"overlap_bundle__{district_id or 'UNKNOWN'}__{safe_policy}__{suffix}.zip"

    files = [str(cert_p)]
    for p in (extras or []):
        if p and os.path.exists(p):
            files.append(p)

    # write zip into the target directory (same FS), then replace/move
    fd, tmp_name = tempfile.mkstemp(dir=BUNDLES_DIR, prefix=".tmp_bundle_", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for abspath in files:
                abspath = str(Path(abspath).resolve())
                zf.write(abspath, arcname=_zip_arcname(abspath))
        try:
            os.replace(tmp_path, zpath)
        except OSError:
            shutil.move(str(tmp_path), str(zpath))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return str(zpath)

# ────────────────────────────── EXPANDER GUARD (optional) ──────────────────────────────
from contextlib import contextmanager
_EXP_STACK = []
@contextmanager
def safe_expander(title: str, **kwargs):
    if _EXP_STACK:
        st.warning(f"Nested expander detected: “{title}” inside “{_EXP_STACK[-1]}”. Consider moving it out.")
    _EXP_STACK.append(title)
    try:
        with st.expander(title, **kwargs):
            yield
    finally:
        _EXP_STACK.pop()

# ────────────────────────────── GALLERY / WITNESS TAIL RENDERERS ──────────────────────────────
def _read_jsonl_tail(path: Path, limit: int = 5) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception:
        return []

def render_gallery_tail(limit: int = 5):
    p = Path(DIRS["logs"]) / "gallery.jsonl"
    rows = _read_jsonl_tail(p, limit)
    st.markdown("**Recent Gallery entries**")
    if not rows:
        st.caption("Gallery: (empty)")
        return
    for r in reversed(rows):
        d = r.get("district", "?")
        pol = r.get("policy", "?")
        gh = r.get("hashes", {})
        pH = r.get("projector_hash", "")
        st.caption(f"{r.get('written_at_utc','')} · {d} · {pol} · Π={pH[:12]} · b={gh.get('boundaries_hash','')[:8]} C={gh.get('C_hash','')[:8]} H={gh.get('H_hash','')[:8]} U={gh.get('U_hash','')[:8]}")
    with st.expander("Gallery tail (JSON)"):
        st.code("\n".join(json.dumps(r, indent=2, sort_keys=True) for r in rows), language="json")

def render_witness_tail(limit: int = 5):
    p = Path(DIRS["logs"]) / "witnesses.jsonl"
    rows = _read_jsonl_tail(p, limit)
    st.markdown("**Recent Witnesses**")
    if not rows:
        st.caption("Witnesses: (empty)")
        return
    for r in reversed(rows):
        st.caption(f"{r.get('written_at_utc','')} · {r.get('district','?')} · {r.get('reason','?')} · residual={r.get('residual_tag','?')} · {r.get('policy','?')}")
    with st.expander("Witness tail (JSON)"):
        st.code("\n".join(json.dumps(r, indent=2, sort_keys=True) for r in rows), language="json")

# ────────────────────────────── CERT INPUTS BLOCK BUILDER ──────────────────────────────
def build_inputs_block(boundaries, cmap, H_used, shapes, filenames: dict) -> dict:
    C3 = (cmap.blocks.__root__.get("3") or [])
    d3 = (boundaries.blocks.__root__.get("3") or [])
    dims = {
        "n3": len(C3) if C3 else (len(d3[0]) if (d3 and d3[0]) else 0),
        "n2": len(cmap.blocks.__root__.get("2") or [])
    }
    hashes_dict = {
        "boundaries_hash": hash_json(boundaries.dict() if hasattr(boundaries, "dict") else {}),
        "C_hash":          hash_json(cmap.dict() if hasattr(cmap, "dict") else {}),
        "H_hash":          hash_json(H_used.dict() if hasattr(H_used, "dict") else {}),
        "U_hash":          hash_json(shapes.dict() if hasattr(shapes, "dict") else {}),
    }
    block = {
        "filenames": filenames,
        "dims": dims,
        **hashes_dict,
        "shapes_hash": hashes_dict["U_hash"],
    }
    return block

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

    # filename stamps for provenance
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

        # Prefer raw-bytes boundary hash when available
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

        # Stamp filenames + authoritative hashes (for later cert/bundle)
        inputs_block["boundaries_filename"] = st.session_state.get("fname_boundaries", "boundaries.json")
        inputs_block["boundaries_hash"]     = boundaries_hash_fresh
        inputs_block["shapes_filename"]     = st.session_state.get("fname_shapes", "shapes.json")
        inputs_block["cmap_filename"]       = st.session_state.get("fname_cmap", "cmap.json")
        inputs_block.setdefault("U_filename", "shapes.json")

        # Mirror fresh district info for later blocks (used by overlap tab)
        st.session_state["_district_info"] = {
            "district_id":        district_id_fresh,
            "boundaries_hash":    boundaries_hash_fresh,
            "lane_mask_k3_now":   lane_mask_k3_now,
            "district_signature": district_sig,
            "d3_rows": d3_rows,
            "d3_cols": d3_cols,
        }

        # Validate schemas
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
    # No projection in strict mode; keep projector fields empty
    return {
        "enabled_layers": [],        # nothing enabled ⇒ strict
        "modes": {},                 # no k=3 mode
        "source": {},                # no projector source
        "projector_files": {},       # no files
    }

def cfg_projected_base() -> dict:
    # Baseline projected config: k=3 columns, AUTO by default
    return {
        "enabled_layers": [3],
        "modes": {"3": "columns"},
        "source": {"3": "auto"},     # "auto" | "file"
        "projector_files": {},       # if file-mode, we'll fill ["3"]
    }

def policy_label_from_cfg(cfg: dict) -> str:
    # Pretty label used in the UI + certs
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    src = (cfg.get("source", {}) or {}).get("3", "auto")
    mode = (cfg.get("modes", {}) or {}).get("3", "columns")
    return f"projected({mode}@k=3,{src})"

# --- FIX: ensure tabs exist even if earlier branches ran before creating them
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
with tab2:
    st.subheader("Overlap gate")

    # ------------------------ Imports & small utils local to Tab 2 ------------------------
    import json as _json
    from pathlib import Path
    import os, tempfile, csv, hashlib  # add hashlib for _stable_hash

    # reuse global GF(2) helpers: mul, add, eye
    def _xor_mat(A, B):  # uses integers (0/1), XOR per entry
        if not A:
            return [r[:] for r in (B or [])]
        if not B:
            return [r[:] for r in (A or [])]
        m, n = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(n)] for i in range(m)]

    def _bottom_row(M):
        return M[-1] if (M and len(M)) else []

    def _stable_hash(obj):
        return hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    # ------------------------ UI: policy + H + projector ------------------------
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

    # load H (or empty cmap)
    def _load_h_local():
        try:
            if f_H is None:
                return io.parse_cmap({"blocks": {}})
            return io.parse_cmap(read_json_file(f_H))
        except Exception:
            # last resort: empty structure to keep downstream shape guards alive
            return io.parse_cmap({"blocks": {}})

    # active config builder (defensive)
    def _cfg_from_policy(policy_choice_str: str, pj_path: str | None) -> dict:
        if policy_choice_str == "strict":
            return cfg_strict()
        cfg = cfg_projected_base()
        if policy_choice_str.endswith("(auto)"):
            cfg.setdefault("source", {})["3"] = "auto"
            # leave projector_files empty in AUTO
            cfg.setdefault("projector_files", {})
        else:  # projected(file)
            cfg.setdefault("source", {})["3"] = "file"
            if pj_path:
                cfg.setdefault("projector_files", {})["3"] = pj_path
        return cfg

    # handle projector upload (if provided)
    pj_saved_path = ""
    if proj_upload is not None:
        os.makedirs("projectors", exist_ok=True)
        pj_saved_path = os.path.join("projectors", proj_upload.name)
        with open(pj_saved_path, "wb") as _pf:
            _pf.write(proj_upload.getvalue())
        st.caption(f"Saved projector: `{pj_saved_path}`")
        st.session_state["ov_last_pj_path"] = pj_saved_path

    # compute active cfg
    cfg_active = _cfg_from_policy(
        policy_choice,
        st.session_state.get("ov_last_pj_path") or pj_saved_path or "",
    )

    # display active policy pill
    st.caption(f"Active policy: `{policy_label_from_cfg(cfg_active)}`")


    # ------------------------ Run Overlap button ------------------------
    H_local = _load_h_local()
    policy_label = policy_label_from_cfg(cfg_active)

    def _lane_mask_from_d3_local(boundaries_obj):
        try:
            d3 = boundaries_obj.blocks.__root__.get("3")
        except Exception:
            d3 = None
        if not d3 or not d3[0]:
            return []
        cols = len(d3[0])
        return [1 if any((row[j] & 1) for row in d3) else 0 for j in range(cols)]

    def run_overlap():
        # Clear previous session results
        for k in ("proj_meta", "run_ctx", "residual_tags", "overlap_out", "overlap_H"):
            st.session_state.pop(k, None)

        # Bind projector (fail-fast on FILE)
        try:
            P_active, meta = projector_choose_active(cfg_active, boundaries)
        except ValueError as e:
            st.error(str(e))
            # Save minimal context so stamps/render don’t crash
            d3_now = (boundaries.blocks.__root__.get("3") or [])
            st.session_state["run_ctx"] = {
                "policy_tag": policy_label,
                "mode": "projected(file)" if cfg_active.get("source", {}).get("3") == "file"
                        else ("strict" if policy_choice=="strict" else "projected(auto)"),
                "d3": d3_now,
                "n3": len(d3_now[0]) if d3_now and d3_now[0] else 0,
                "lane_mask_k3": [],
                "P_active": [],
                "projector_filename": (cfg_active.get("projector_files", {}) or {}).get("3", ""),
                "projector_hash": "",
                "projector_consistent_with_d": False,
                "errors": [str(e)],
            }
            st.stop()

        # Context
        d3 = meta.get("d3") if "d3" in meta else (boundaries.blocks.__root__.get("3") or [])
        n3 = meta.get("n3") if "n3" in meta else (len(d3[0]) if (d3 and d3[0]) else 0)
        lane_mask = meta.get("lane_mask", _lane_mask_from_d3_local(boundaries))
        mode = meta.get("mode", "strict")

        # Compute strict residual R3 = H2@d3 XOR (C3 XOR I3)
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
            if not R or not lm:
                return "none"
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

        # Persist RunContext SSOT
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

        # UI result
        st.json(out)

        # Banner: projected(file) verdict
        if mode == "projected(file)":
            if meta.get("projector_consistent_with_d", False):
                st.success(f"projected(file) OK · {meta.get('projector_filename','')} · {meta.get('projector_hash','')[:12]} ✔️")
            else:
                st.warning("Projected(file) is not consistent with current d3 (check shape/idempotence/diag/lane).")

    if st.button("Run Overlap", key="run_overlap"):
        run_overlap()

    # ------------------------ Cert writer (central, SSOT-only) ------------------------
st.divider()
st.caption("Cert & provenance")

_rc   = st.session_state.get("run_ctx") or {}
_out  = st.session_state.get("overlap_out") or {}
_ib   = st.session_state.get("_inputs_block") or {}
_di   = st.session_state.get("_district_info") or {}
_H    = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})
_ab   = st.session_state.get("ab_compare") or {}

# quick guard
if not (_rc and _out and _ib):
    st.info("Run Overlap first to enable cert writing.")
else:
    # Diagnostics (lane vectors, etc.) from RC + inputs
    lane_mask = list(_rc.get("lane_mask_k3", []) or [])
    d3 = _rc.get("d3", [])
    H2 = (_H.blocks.__root__.get("2") or [])
    C3 = (cmap.blocks.__root__.get("3") or [])
    I3 = eye(len(C3)) if C3 else []

    def _bottom_row(M): return M[-1] if (M and len(M)) else []
    def _xor(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(len(A[0]))] for i in range(len(A))]
    def _mask(vec, idx): return [vec[j] for j in idx] if (vec and idx) else []

    try:
        H2d3  = mul(H2, d3) if (H2 and d3) else []
        C3pI3 = _xor(C3, I3) if C3 else []
    except Exception:
        H2d3, C3pI3 = [], []

    lane_idx = [j for j, m in enumerate(lane_mask) if m]
    lane_vec_H2d3     = _mask(_bottom_row(H2d3), lane_idx)
    lane_vec_C3plusI3 = _mask(_bottom_row(C3pI3), lane_idx)

    # Signatures
    def _gf2_rank(M):
        if not M or not M[0]: return 0
        A = [row[:] for row in M]; m, n = len(A), len(A[0]); r = c = 0
        while r < m and c < n:
            piv = next((i for i in range(r, m) if A[i][c] & 1), None)
            if piv is None: c += 1; continue
            if piv != r: A[r], A[piv] = A[piv], A[r]
            for i in range(m):
                if i != r and (A[i][c] & 1):
                    A[i] = [(A[i][j] ^ A[r][j]) & 1 for j in range(n)]
            r += 1; c += 1
        return r

    rank_d3 = _gf2_rank(d3) if d3 else 0
    ncols_d3 = len(d3[0]) if (d3 and d3[0]) else 0
    ker_dim_d3 = max(ncols_d3 - rank_d3, 0)
    lane_pattern = "".join("1" if int(x) else "0" for x in (lane_mask or []))

    def _col_support_pattern(M, cols):
        if not M or not cols: return ""
        return "".join("1" if any((row[j] & 1) for row in M) else "0" for j in cols)

    fixture_signature = {"lane": _col_support_pattern(C3pI3, lane_idx)}
    d_signature = {"rank": rank_d3, "ker_dim": ker_dim_d3, "lane_pattern": lane_pattern}

    # Residual tags (from earlier run)
    residual_tags = st.session_state.get("residual_tags", {}) or {}
    # Checks & ker_guard from current policy mode
    is_strict = (_rc.get("mode") == "strict")
    checks_block = {
        **_out,
        "grid": True,
        "fence": True,
        "ker_guard": ("enforced" if is_strict else "off"),
    }

    # Identity (no recomputing input hashes here)
    district_id = _di.get("district_id", st.session_state.get("district_id", "UNKNOWN"))
    run_ts = hashes.timestamp_iso_lisbon()
    # Compose a run_id from stable inputs + policy tag + timestamp
    _policy_tag_now = _rc.get("policy_tag", policy_label_from_cfg(cfg_active))
    _rid_seed = {
        "b": _ib.get("boundaries_hash",""),
        "C": _ib.get("C_hash",""),
        "H": _ib.get("H_hash",""),
        "U": _ib.get("U_hash",""),
        "policy_tag": _policy_tag_now,
        "ts": run_ts,
    }
    run_id = hashes.run_id(hash_json(_rid_seed), run_ts)

    identity_block = {
        "district_id": district_id,
        "run_id": run_id,
        "timestamp": run_ts,
        "app_version": getattr(hashes, "APP_VERSION", "v0.1-core"),
        "python_version": f"python-{platform.python_version()}",
    }

    # Policy block from RC (mirror SSOT)
    policy_block = {
        "label": _policy_tag_now,
        "policy_tag": _policy_tag_now,
        "enabled_layers": cfg_active.get("enabled_layers", []),
        "modes": cfg_active.get("modes", {}),
        "source": cfg_active.get("source", {}),
    }
    if _rc.get("projector_hash") is not None:
        policy_block["projector_hash"] = _rc.get("projector_hash","")
    if _rc.get("projector_filename"):
        policy_block["projector_filename"] = _rc.get("projector_filename","")
    if _rc.get("projector_consistent_with_d") is not None:
        policy_block["projector_consistent_with_d"] = bool(_rc.get("projector_consistent_with_d"))

    # Inputs (copy from SSOT)
    inputs_block = {
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
    # If FILE mode, include projector filename in inputs
    if _rc.get("mode") == "projected(file)":
        inputs_block.setdefault("filenames", {})["projector"] = _rc.get("projector_filename","")

    diagnostics_block = {
        "lane_mask_k3": lane_mask,
        "lane_vec_H2d3": lane_vec_H2d3,
        "lane_vec_C3plusI3": lane_vec_C3plusI3,
    }

    signatures_block = {
        "d_signature": d_signature,
        "fixture_signature": fixture_signature,
    }

    # Promotion rules
    grid_ok  = bool(_out.get("grid", True))
    fence_ok = bool(_out.get("fence", True))
    k3_ok    = bool(_out.get("3", {}).get("eq", False))
    k2_ok    = bool(_out.get("2", {}).get("eq", False))
    mode_now = _rc.get("mode")
    eligible, target = False, None
    if mode_now == "strict" and all([grid_ok, fence_ok, k3_ok, k2_ok]) and residual_tags.get("strict","none") == "none":
        eligible, target = True, "strict_anchor"
    elif mode_now in ("projected(auto)", "projected(file)") and all([grid_ok, fence_ok, k3_ok]) and residual_tags.get("projected","none") == "none":
        if mode_now == "projected(file)":
            if bool(_rc.get("projector_consistent_with_d")):
                eligible, target = True, "projected_exemplar"
        else:
            eligible, target = True, "projected_exemplar"

    promotion_block = {
        "eligible_for_promotion": eligible,
        "promotion_target": target,
        "notes": "",
    }

    artifact_hashes = {
        "boundaries_hash": inputs_block["boundaries_hash"],
        "C_hash":          inputs_block["C_hash"],
        "H_hash":          inputs_block["H_hash"],
        "U_hash":          inputs_block["U_hash"],
    }
    if "projector_hash" in policy_block:
        artifact_hashes["projector_hash"] = policy_block.get("projector_hash","")

    # Assemble core payload
    cert_payload = {
        "identity":    identity_block,
        "policy":      policy_block,
        "inputs":      inputs_block,
        "diagnostics": diagnostics_block,
        "checks":      checks_block,
        "signatures":  signatures_block,
        "residual_tags": residual_tags,
        "promotion":   promotion_block,
        "artifact_hashes": artifact_hashes,
    }

    # Embed A/B if fresh
    inputs_sig_now = [
        inputs_block["boundaries_hash"],
        inputs_block["C_hash"],
        inputs_block["H_hash"],
        inputs_block["U_hash"],
        inputs_block["shapes_hash"],
    ]
    if _ab and (_ab.get("inputs_sig") == inputs_sig_now):
        def _pass_vec_from(d): return [int(d.get("2",{}).get("eq",False)), int(d.get("3",{}).get("eq",False))]
        cert_payload["policy"]["strict_snapshot"] = {
            "policy_tag": _ab.get("strict",{}).get("label","strict"),
            "ker_guard":  "enforced",
            "inputs": {"filenames": inputs_block["filenames"]},
            "lane_mask_k3": lane_mask,
            "lane_vec_H2d3": _ab.get("strict",{}).get("lane_vec_H2d3", lane_vec_H2d3),
            "lane_vec_C3plusI3": _ab.get("strict",{}).get("lane_vec_C3plusI3", lane_vec_C3plusI3),
            "pass_vec": _pass_vec_from(_ab.get("strict",{}).get("out", {})),
            "out": _ab.get("strict",{}).get("out", {}),
        }
        proj_hash_ab = _ab.get("projected",{}).get("projector_hash", _rc.get("projector_hash",""))
        cert_payload["policy"]["projected_snapshot"] = {
            "policy_tag": _rc.get("policy_tag", policy_label_from_cfg(cfg_projected_base())),
            "ker_guard":  "off",
            "projector_hash": proj_hash_ab,
            "inputs": {"filenames": inputs_block["filenames"]},
            "lane_mask_k3": lane_mask,
            "lane_vec_H2d3": _ab.get("projected",{}).get("lane_vec_H2d3", lane_vec_H2d3),
            "lane_vec_C3plusI3": _ab.get("projected",{}).get("lane_vec_C3plusI3", lane_vec_C3plusI3),
            "pass_vec": _pass_vec_from(_ab.get("projected",{}).get("out", {})),
            "out": _ab.get("projected",{}).get("out", {}),
            **({"projector_consistent_with_d": bool(_rc.get("projector_consistent_with_d"))}
               if _rc.get("mode")=="projected(file)" else {}),
        }
        cert_payload["ab_pair_tag"] = _ab.get("pair_tag","")
    else:
        if _ab:
            st.caption("A/B snapshot is stale — not embedding into the cert (hashes changed).")

    # Schema/app/python tags + integrity
    cert_payload["schema_version"] = LAB_SCHEMA_VERSION
    cert_payload["app_version"]    = getattr(hashes, "APP_VERSION", "v0.1-core")
    cert_payload["python_version"] = f"python-{platform.python_version()}"
    cert_payload.setdefault("integrity", {})
    cert_payload["integrity"]["content_hash"] = hash_json(cert_payload)

    # Write cert (prefer package writer; fallback locally)
    cert_path = None
    full_hash = cert_payload["integrity"]["content_hash"]
    try:
        result = export_mod.write_cert_json(cert_payload)  # expected to return (path, full_hash) or path
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            cert_path, full_hash = result[0], result[1]
        else:
            cert_path = result
    except Exception as e:
        # Fallback: write to certs/overlap__{district}__{policy}__{hash[:12]}.json atomically
        try:
            outdir = Path("certs"); outdir.mkdir(parents=True, exist_ok=True)
            safe_policy = _policy_tag_now.replace("/", "_").replace(" ", "_")
            z = f"overlap__{district_id}__{safe_policy}__{full_hash[:12]}.json"
            p = outdir / z
            tmp = p.with_suffix(".json.tmp")
            with open(tmp, "wb") as f:
                f.write(_json.dumps(cert_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, p)
            cert_path = str(p)
        except Exception as e2:
            st.error(f"Cert write failed: {e} / {e2}")

    if cert_path:
        st.session_state["last_cert_path"] = cert_path
        st.session_state["cert_payload"]   = cert_payload
        st.success(f"Cert written → `{cert_path}` · {full_hash[:12]}…")
        # Optional: quick bundle button right after success
        with st.expander("Bundle (cert + extras)"):
            extras = [
                "policy.json",
                "reports/residual.json",
                "reports/parity_report.json",
                "reports/coverage_sampling.csv",
                "logs/gallery.jsonl",
                "logs/witnesses.jsonl",
            ]
            if _rc.get("mode") == "projected(file)" and _rc.get("projector_filename"):
                extras.append(_rc.get("projector_filename"))
            if st.button("Build Cert Bundle", key="build_cert_bundle_btn"):
                try:
                    bundle_path = build_cert_bundle(
                        district_id=district_id,
                        policy_tag=_policy_tag_now,
                        cert_path=cert_path,
                        content_hash=full_hash,
                        extras=extras
                    )
                    st.success(f"Bundle ready → {bundle_path}")
                except Exception as e:
                    st.error(f"Bundle build failed: {e}")
    else:
        st.warning("No cert file was produced. Fix the error above and try again.")


    # ------------------------ A/B compare (strict vs active projected) ------------------------
if st.button("Run A/B compare", key="ab_run_btn"):
    try:
        # Inputs + context
        rc  = st.session_state.get("run_ctx") or {}
        ib  = st.session_state.get("_inputs_block") or {}
        H_used = st.session_state.get("overlap_H") or io.parse_cmap({"blocks": {}})

        # --- strict leg
        out_strict   = overlap_gate.overlap_check(boundaries, cmap, H_used)
        label_strict = policy_label_from_cfg(cfg_strict())

        # --- projected leg mirrors ACTIVE (auto/file), with FILE validation
        cfg_proj = st.session_state.get("overlap_cfg") or cfg_active  # prefer last run if exists
        # fail-fast on FILE Π validity (shape/idempotence/diag/lane)
        _P_ab, _meta_ab = projector_choose_active(cfg_proj, boundaries)
        out_proj   = overlap_gate.overlap_check(boundaries, cmap, H_used, projection_config=cfg_proj)
        label_proj = policy_label_from_cfg(cfg_proj)

        # --- lightweight provenance lane vectors
        d3 = rc.get("d3") or (boundaries.blocks.__root__.get("3") or [])
        H2 = (H_used.blocks.__root__.get("2") or [])
        C3 = (cmap.blocks.__root__.get("3") or [])
        I3 = eye(len(C3)) if C3 else []
        lane_idx = [j for j,m in enumerate(rc.get("lane_mask_k3", [])) if m]

        def _bottom_row(M): return M[-1] if (M and len(M)) else []
        def _xor(A,B):
            if not A: return B or []
            if not B: return A or []
            return [[(A[i][j]^B[i][j]) for j in range(len(A[0]))] for i in range(len(A))]
        def _mask(vec, idx): return [vec[j] for j in idx] if (vec and idx) else []

        H2d3  = mul(H2, d3) if (H2 and d3) else []
        C3pI3 = _xor(C3, I3) if C3 else []
        lane_vec_H2d3 = _mask(_bottom_row(H2d3), lane_idx)
        lane_vec_C3I  = _mask(_bottom_row(C3pI3), lane_idx)

        # --- freshness key from inputs (so A/B only embeds when fresh)
        def _hz(v): return v if isinstance(v, str) else ""
        inputs_sig = [
            _hz(ib.get("boundaries_hash","")),
            _hz(ib.get("C_hash","")),
            _hz(ib.get("H_hash","")),
            _hz(ib.get("U_hash","")),
            _hz(ib.get("shapes_hash","")),
        ]

        # --- persist snapshot
        ab_payload = {
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
                "pass_vec": [int(out_strict.get("2",{}).get("eq",False)), int(out_strict.get("3",{}).get("eq",False))],
                "projector_hash": "",
            },
            "projected": {
                "label": label_proj,
                "cfg":   cfg_proj,
                "out":   out_proj,
                "ker_guard": "off",
                "lane_vec_H2d3": lane_vec_H2d3[:],
                "lane_vec_C3plusI3": lane_vec_C3I[:],
                "pass_vec": [int(out_proj.get("2",{}).get("eq",False)), int(out_proj.get("3",{}).get("eq",False))],
                "projector_filename": _meta_ab.get("projector_filename",""),
                "projector_hash": _meta_ab.get("projector_hash",""),
                "projector_consistent_with_d": _meta_ab.get("projector_consistent_with_d", None),
            },
        }
        st.session_state["ab_compare"] = ab_payload

        # quick status line
        s_ok = bool(out_strict.get("3",{}).get("eq", False))
        p_ok = bool(out_proj.get("3",{}).get("eq", False))
        st.success(f"A/B updated → strict={'✅' if s_ok else '❌'} · projected={'✅' if p_ok else '❌'} · {ab_payload['pair_tag']}")

        # note for certs: your cert writer will embed A/B iff inputs_sig matches current _inputs_block
        st.caption("A/B snapshot saved. It will embed in the next cert if inputs haven’t changed (fresh).")

    except ValueError as e:
        # typical FILE Π validator errors (P3_SHAPE / P3_IDEMP / P3_DIAGONAL / P3_LANE_MISMATCH)
        st.error(f"A/B projected(file) invalid: {e}")
    except Exception as e:
        st.error(f"A/B compare failed: {e}")


    # ------------------------ Parity import/export & sample queue ------------------------
    PARITY_SCHEMA_VERSION = "1.0.0"
    DEFAULT_PARITY_PATH = Path("logs") / "parity_pairs.json"

    def _iso_utc_now(): return datetime.now(timezone.utc).isoformat()
    def _ensure_parent_dir(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(path: Path, payload: dict):
        _ensure_parent_dir(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        blob = _json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        with open(tmp, "wb") as f:
            f.write(blob); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)

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
            "cmap": io.parse_cmap(dC),
            "H": io.parse_cmap(dH),
            "shapes": io.parse_shapes(dU),
        }

    def add_parity_pair(*, label: str, left_fixture: dict, right_fixture: dict):
        req_keys = ("boundaries", "cmap", "H", "shapes")
        for side_name, fx in [("left", left_fixture), ("right", right_fixture)]:
            if not isinstance(fx, dict) or any(k not in fx for k in req_keys):
                raise ValueError(f"{side_name} fixture malformed; expected keys {req_keys}")
        st.session_state.setdefault("parity_pairs", [])
        st.session_state["parity_pairs"].append({"label": label, "left": left_fixture, "right": right_fixture})
        return len(st.session_state["parity_pairs"])

    def clear_parity_pairs():
        st.session_state["parity_pairs"] = []

    def set_parity_pairs_from_fixtures(pairs_spec: list[dict]):
        clear_parity_pairs()
        for row in pairs_spec:
            label = row.get("label", "PAIR")
            Lp = row.get("left", {})
            Rp = row.get("right", {})
            L = load_fixture_from_paths(
                boundaries_path=Lp["boundaries"], cmap_path=Lp["cmap"], H_path=Lp["H"], shapes_path=Lp["shapes"]
            )
            R = load_fixture_from_paths(
                boundaries_path=Rp["boundaries"], cmap_path=Rp["cmap"], H_path=Rp["H"], shapes_path=Rp["shapes"]
            )
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
                }
                for row in pairs
            ],
        }

    def _pairs_from_payload(payload: dict) -> list[dict]:
        if not isinstance(payload, dict): return []
        return [
            {
                "label": r.get("label", "PAIR"),
                "left":  {k: r.get("left", {}).get(k, "")  for k in ("boundaries","cmap","H","shapes")},
                "right": {k: r.get("right", {}).get(k, "") for k in ("boundaries","cmap","H","shapes")},
            }
            for r in payload.get("pairs", [])
        ]

    def export_parity_pairs(path: str | Path = DEFAULT_PARITY_PATH) -> str:
        path = Path(path)
        _ensure_parent_dir(path)
        pairs = st.session_state.get("parity_pairs", []) or []
        payload = _parity_pairs_payload(pairs)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
            _json.dump(payload, tmp, indent=2)
            tmp.flush(); os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)
        return str(path)

    def import_parity_pairs(path: str | Path = DEFAULT_PARITY_PATH, *, merge: bool=False) -> int:
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
