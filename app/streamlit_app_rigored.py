# ────────────────────────────── IMPORTS ──────────────────────────────
# Standard library imports
import sys
import os
import pathlib
import importlib.util
import types
import json
import hashlib
import zipfile
import tempfile
import shutil
import csv
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
import platform

# Third-party imports
import streamlit as st

# ────────────────────────────── CONFIG AND CONSTANTS ──────────────────────────────
# Page configuration
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# ────────────────────────────── PACKAGE LOADER ──────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
CORE = HERE / "core"
PKG_DIR = OTCORE if OTCORE.exists() else CORE
PKG_NAME = "otcore" if OTCORE.exists() else "core"

# Ensure a package object exists in sys.modules for submodules
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PKG_DIR)]
    pkg.__file__ = str(PKG_DIR / "__init__.py")
    sys.modules[PKG_NAME] = pkg

def _load_pkg_module(fullname: str, rel_path: str):
    """Load a module from PKG_DIR under the given fullname."""
    path = PKG_DIR / rel_path
    if not path.exists():
        raise ImportError(f"Required module file not found: {path}")
    spec = importlib.util.spec_from_file_location(fullname, str(path))
    mod = importlib.util.module_from_spec(spec)
    # Make relative imports inside that file resolve under its package
    mod.__package__ = fullname.rsplit('.', 1)[0]
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Remove previously loaded modules to ensure fresh load
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

# Load core modules
overlap_gate = _load_pkg_module(f"{PKG_NAME}.overlap_gate", "overlap_gate.py")
projector    = _load_pkg_module(f"{PKG_NAME}.projector", "projector.py")
otio         = _load_pkg_module(f"{PKG_NAME}.io", "io.py")
hashes       = _load_pkg_module(f"{PKG_NAME}.hashes", "hashes.py")
unit_gate    = _load_pkg_module(f"{PKG_NAME}.unit_gate", "unit_gate.py")
triangle_gate= _load_pkg_module(f"{PKG_NAME}.triangle_gate","triangle_gate.py")
towers       = _load_pkg_module(f"{PKG_NAME}.towers", "towers.py")
export_mod   = _load_pkg_module(f"{PKG_NAME}.export", "export.py")

# Compatibility alias for old references
if "io" not in globals():
    io = otio

# ────────────────────────────── UI HEADER ──────────────────────────────
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")
st.caption(f"overlap_gate loaded from: {getattr(overlap_gate, '__file__', '<none>')}")
st.caption(f"projector loaded from: {getattr(projector, '__file__', '<none>')}")

# ────────────────────────────── MATH LAB FOUNDATION ──────────────────────────────
# Schema + paths + atomic IO + run IDs + residual snapshot + UI widgets
import os
import json
import csv
import hashlib
import zipfile
import io as pyio
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
import platform

# Bundle builder directory
BUNDLES_DIR = Path("bundles")
BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── Helper Functions ──────────────────────────────

def _zip_arcname(abspath: str) -> str:
    """Return relative path inside zip; fallback to basename."""
    p = Path(abspath)
    try:
        rel = p.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except Exception:
        return p.name

def build_cert_bundle(
    *,
    district_id: str,
    policy_tag: str,
    cert_path: str,
    content_hash: str | None = None,
    extras: list[str] | None = None
) -> str:
    """Create a zip bundle with cert and optional extras."""
    cert_p = Path(cert_path)
    if not cert_p.exists():
        raise FileNotFoundError(f"Cert not found: {cert_path}")

    with open(cert_p, "r", encoding="utf-8") as f:
        cert = json.load(f)

    if not content_hash:
        content_hash = ((cert.get("integrity") or {}).get("content_hash") or "")

    suffix = content_hash[:12] if content_hash else "nohash"
    safe_policy = (policy_tag or cert.get("policy", {}).get("policy_tag", "policy")).replace("/", "_").replace(" ", "_")

    zname = f"overlap_bundle__{district_id or 'UNKNOWN'}__{safe_policy}__{suffix}.zip"
    zpath = BUNDLES_DIR / zname

    files = [str(cert_p)]
    for p in (extras or []):
        if p and os.path.exists(p):
            files.append(p)

    # Create temp zip file
    fd, tmp_name = tempfile.mkstemp(dir=BUNDLES_DIR, prefix=".tmp_bundle_", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for abspath in files:
                abspath = str(Path(abspath).resolve())
                zf.write(abspath, arcname=_zip_arcname(abspath))
        # Atomic replace or move
        try:
            os.replace(tmp_path, zpath)
        except OSError:
            shutil.move(str(tmp_path), str(zpath))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return str(zpath)

# ────────────────────────────── Safe Expander ──────────────────────────────
import streamlit as st
from contextlib import contextmanager

_EXP_STACK = []

@contextmanager
def safe_expander(title: str, **kwargs):
    """Warn if nested expanders."""
    global _EXP_STACK
    if _EXP_STACK:
        st.warning(f"Nested expander detected: “{title}” inside “{_EXP_STACK[-1]}”. Consider moving it out.")
    _EXP_STACK.append(title)
    try:
        with st.expander(title, **kwargs):
            yield
    finally:
        _EXP_STACK.pop()

# ────────────────────────────── Version & Schema Constants ──────────────────────────────
LAB_SCHEMA_VERSION = "1.0.0"
APP_VERSION_STR = str(APP_VERSION) if "APP_VERSION" in globals() else "v0.1-core"
PY_VERSION_STR = f"python-{platform.python_version()}"

# ────────────────────────────── Directory Setup ──────────────────────────────
DIRS = {
    "inputs": "inputs",
    "certs": "certs",
    "bundles": "bundles",
    "projectors": "projectors",
    "logs": "logs",
    "reports": "reports",
    "fixtures": "fixtures",
    "configs": "configs",
}
for _d in DIRS.values():
    Path(_d).mkdir(parents=True, exist_ok=True)

# ────────────────────────────── Utility Functions ──────────────────────────────

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def hash_json(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

# ────────────────────────────── Gallery / Witness UI Helpers ──────────────────────────────
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
    p = Path("logs") / "gallery.jsonl"
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
    p = Path("logs") / "witnesses.jsonl"
    rows = _read_jsonl_tail(p, limit)
    st.markdown("**Recent Witnesses**")
    if not rows:
        st.caption("Witnesses: (empty)")
        return
    for r in reversed(rows):
        d = r.get("district", "?")
        reason = r.get("reason", "?")
        tag = r.get("residual_tag", "?")
        pol = r.get("policy", "?")
        st.caption(f"{r.get('written_at_utc','')} · {d} · {reason} · residual={tag} · {pol}")
    with st.expander("Witness tail (JSON)"):
        st.code("\n".join(json.dumps(r, indent=2, sort_keys=True) for r in rows), language="json")

def hash_matrix_norm(M) -> str:
    if not M:
        return hash_json([])
    norm = [[int(x) & 1 for x in row] for row in M]
    return hash_json(norm)

def build_inputs_block(boundaries, cmap, H_used, shapes, filenames: dict) -> dict:
    C3 = (cmap.blocks.__root__.get("3") or [])
    d3 = (boundaries.blocks.__root__.get("3") or [])
    dims = {
        "n3": len(C3) if C3 else (len(d3[0]) if (d3 and d3[0]) else 0),
        "n2": len(cmap.blocks.__root__.get("2") or [])
    }
    hashes = {
        "boundaries_hash": hash_json(boundaries.dict() if hasattr(boundaries, "dict") else {}),
        "C_hash": hash_json(cmap.dict() if hasattr(cmap, "dict") else {}),
        "H_hash": hash_json(H_used.dict() if hasattr(H_used, "dict") else {}),
        "U_hash": hash_json(shapes.dict() if hasattr(shapes, "dict") else {}),
    }
    block = {
        "filenames": filenames,
        "dims": dims,
        **hashes,
        "shapes_hash": hashes["U_hash"],
    }
    return block

def residual_tag(R, lane_mask):
    if not R:
        return "none"
    rows, cols = len(R), len(R[0]) if R and R[0] else 0
    if cols == 0:
        return "none"
    lanes = [j for j, m in enumerate(lane_mask or []) if m]
    ker = [j for j, m in enumerate(lane_mask or []) if not m]
    def col_nz(j): return any(R[i][j] & 1 for i in range(rows))
    L = any(col_nz(j) for j in lanes) if lanes else False
    K = any(col_nz(j) for j in ker) if ker else False
    if not L and not K:
        return "none"
    if L and not K:
        return "lanes"
    if K and not L:
        return "ker"
    return "mixed"

def append_gallery_row(cert: dict, growth_bumps: int = 0, strictify: str = "tbd") -> bool:
    from pathlib import Path
    row = {
        "schema_version": "1.0.0",
        "written_at_utc": hashes.timestamp_iso_lisbon(),
        "app_version": getattr(hashes, "APP_VERSION", "v0.1-core"),
        "district": cert["identity"]["district_id"],
        "policy": cert["policy"]["policy_tag"],
        "projector_hash": cert["policy"].get("projector_hash", ""),
        "hashes": {k: cert["artifact_hashes"][k] for k in ("boundaries_hash", "C_hash", "H_hash", "U_hash")},
        "growth_bumps": growth_bumps,
        "strictify": strictify,
        "cert_content_hash": cert["integrity"]["content_hash"],
        "run_id": cert["identity"]["run_id"],
    }
    # dedupe: (district, b, C, H, U, policy)
    key = (row["district"], *[row["hashes"][k] for k in ("boundaries_hash", "C_hash", "H_hash", "U_hash")], row["policy"])
    reg = st.session_state.setdefault("_gallery_keys", set())
    if key in reg:
        return False
    reg.add(key)
    Path("logs").mkdir(parents=True, exist_ok=True)
    atomic_append_jsonl(Path("logs") / "gallery.jsonl", row)
    return True

def append_witness_row(cert: dict, reason: str, residual_tag_val: str, note: str = "") -> None:
    from pathlib import Path
    row = {
        "schema_version": "1.0.0",
        "written_at_utc": hashes.timestamp_iso_lisbon(),
        "app_version": getattr(hashes, "APP_VERSION", "v0.1-core"),
        "district": cert["identity"]["district_id"],
        "reason": reason,
        "residual_tag": residual_tag_val,
        "policy": cert["policy"]["policy_tag"],
        "projector_hash": cert["policy"].get("projector_hash", ""),
        "hashes": {k: cert["artifact_hashes"][k] for k in ("boundaries_hash", "C_hash", "H_hash", "U_hash")},
        "cert_content_hash": cert["integrity"]["content_hash"],
        "run_id": cert["identity"]["run_id"],
        "note": note,
    }
    Path("logs").mkdir(parents=True, exist_ok=True)
    atomic_append_jsonl(Path("logs") / "witnesses.jsonl", row)

def _short(h: str, n: int = 8) -> str:
    return (h or "")[:n]

# Additional utility functions like lane_mask_from_boundaries, hash functions, etc. should be similarly organized.

# ────────────────────────────── END of File ──────────────────────────────
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




# ------------------------ GF(2) Helper Functions ------------------------
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
    r, k = len(A), len(A[0])
    k2, c = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"dimension mismatch: {len(A)}x{len(A)[0]} @ {len(B)}x{len(B)[0]}")
    out = [[0] * c for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for t in range(k):
            if Ai[t] & 1:
                Bt = B[t]
                for j in range(c):
                    out[i][j] ^= (Bt[j] & 1)
    return out

# Expose 'mul' and 'add' for use elsewhere, ensuring compatibility
def mul(A, B): return _mul_gf2(A, B)
def add(A, B): return _xor_mat(A, B)
def _eye(n): return _eye(n)

# ------------------------ Utility Functions ------------------------
def _bottom_row(M):
    return M[-1] if (M and len(M)) else []

def _stable_hash(obj):
    return _hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

# ------------------------ Core Logic Functions ------------------------
def load_homotopy_H():
    """ Load Homotopy H json uploaded by user, parse, and return cmap object """
    if f_H is None:
        return io.parse_cmap({"blocks": {}})
    return io.parse_cmap(read_json_file(f_H))

def get_policy_active(cfg_file, policy_choice):
    """Builds the active policy config based on user choice and loaded config."""
    cfg_proj = cfg_projected_base()
    if cfg_file.get("source", {}).get("3") in ("file", "auto"):
        cfg_proj["source"]["3"] = cfg_file["source"]["3"]
    if "projector_files" in cfg_file and "3" in cfg_file["projector_files"]:
        cfg_proj.setdefault("projector_files", {})["3"] = cfg_file["projector_files"]["3"]
    return cfg_strict() if policy_choice == "strict" else cfg_proj

def handle_projector_upload(cfg_active):
    """Handle uploading a projector file and update the active config accordingly."""
    if proj_upload is None:
        return
    os.makedirs("projectors", exist_ok=True)
    pj_saved = os.path.join("projectors", proj_upload.name)
    with open(pj_saved, "wb") as _pf:
        _pf.write(proj_upload.getvalue())
    st.caption(f"saved projector: {pj_saved}")
    if policy_choice != "strict":
        cfg_active.setdefault("source", {})["3"] = "file"
        cfg_active.setdefault("projector_files", {})["3"] = pj_saved

def apply_proj_source(cfg_file, cfg_active, mode_choice, file_path):
    """Update projection config based on user input."""
    cfg_file.setdefault("source", {})["3"] = mode_choice
    if mode_choice == "file":
        cfg_file.setdefault("projector_files", {})["3"] = file_path
    else:
        cfg_file.get("projector_files", {}).pop("3", None)
    with open("projection_config.json", "w") as _f:
        _json.dump(cfg_file, _f, indent=2)

def run_overlap():
    """Main function to execute the overlap calculation, handle exceptions."""
    try:
        # Clear previous session results
        for k in ("proj_meta", "run_ctx", "residual_tags", "overlap_out", "overlap_H"):
            st.session_state.pop(k, None)

        # Bind projector (fail-fast on FILE)
        P_active, meta = projector_choose_active(cfg_active, boundaries)
    except ValueError as e:
        st.error(str(e))
        # Save minimal context info
        d3_now = (boundaries.blocks.__root__.get("3") or [])
        st.session_state["run_ctx"] = {
            "policy_tag": policy_label,
            "mode": "projected(file)" if cfg_active.get("source", {}).get("3") == "file"
                    else ("strict" if policy_choice == "strict" else "projected(auto)"),
            "d3": d3_now,
            "n3": len(d3_now[0]) if d3_now and d3_now[0] else 0,
            "lane_mask_k3": [],
            "P_active": [],
            "projector_filename": cfg_active.get("projector_files", {}).get("3", ""),
            "projector_hash": "",
            "projector_consistent_with_d": False,
            "errors": [str(e)],
        }
        st.stop()

    # Get context details
    d3 = meta.get("d3", (boundaries.blocks.__root__.get("3") or []))
    n3 = meta.get("n3", len(d3[0]) if (d3 and d3[0]) else 0)
    lane_mask = meta.get("lane_mask", _lane_mask_from_d3(boundaries))
    mode = meta.get("mode", "strict")

    # Calculate overlap (k=3)
    H2 = (H_local.blocks.__root__.get("2") or [])
    C3 = (cmap.blocks.__root__.get("3") or [])
    I3 = _eye(len(C3)) if C3 else []

    try:
        R3_strict = _xor_mat(mul(H2, d3), _xor_mat(C3, I3)) if (H2 and d3 and C3) else []
    except Exception as e:
        st.error(f"Shape guard failed at k=3: {e}")
        st.stop()

    # Residual tag computation
    def _is_zero(M):
        return not M or all(all((x & 1) == 0 for x in row) for row in M)

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
    eq3_strict = _is_zero(R3_strict)

    # Projection and residuals
    R3_proj = mul(R3_strict, P_active) if (R3_strict and P_active) else []
    eq3_proj = _is_zero(R3_proj)
    tag_proj = _residual_tag(R3_proj, lane_mask)

    # Output result
    if cfg_active.get("enabled_layers"):
        out = {"3": {"eq": bool(eq3_proj), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict, "projected": tag_proj}
    else:
        out = {"3": {"eq": bool(eq3_strict), "n_k": n3}, "2": {"eq": True}}
        st.session_state["residual_tags"] = {"strict": tag_strict}

    st.json(out)

    # Persist run context
    st.session_state["overlap_out"] = out
    st.session_state["overlap_cfg"] = cfg_active
    st.session_state["overlap_policy_label"] = policy_label
    st.session_state["overlap_H"] = H_local
    st.session_state["run_ctx"] = {
        "policy_tag": policy_label,
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

# ------------------------ Main execution for Overlap button ------------------------
if st.button("Run Overlap", key="run_overlap"):
    run_overlap()
# ------------------------ Imports & Constants ------------------------
import os
import json as _json
from pathlib import Path
from datetime import datetime, timezone

PARITY_SCHEMA_VERSION = "1.0.0"
DEFAULT_PARITY_PATH = Path("logs") / "parity_pairs.json"

# ------------------------ Utility Functions ------------------------
def _iso_utc_now():
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _atomic_write_json(path: Path, payload: dict):
    """Atomic write of JSON payload to disk."""
    _ensure_parent_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    blob = _json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# ------------------------ Load JSON Helper ------------------------
def _safe_parse_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return _json.load(f)

# ------------------------ Fixture Loader ------------------------
def load_fixture_from_paths(*, boundaries_path: str, cmap_path: str, H_path: str, shapes_path: str):
    """
    Load and parse fixture files from paths; no hashing involved.
    """
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

# ------------------------ Parity Pairs management ------------------------
def add_parity_pair(*, label: str, left_fixture: dict, right_fixture: dict):
    """
    Append a parity pair to session state.
    Fixtures must have keys: boundaries, cmap, H, shapes.
    """
    req_keys = ("boundaries", "cmap", "H", "shapes")
    for side_name, fx in [("left", left_fixture), ("right", right_fixture)]:
        if not isinstance(fx, dict) or any(k not in fx for k in req_keys):
            raise ValueError(f"{side_name} fixture malformed; expected keys {req_keys}")
    st.session_state.setdefault("parity_pairs", [])
    st.session_state["parity_pairs"].append({
        "label": label,
        "left": left_fixture,
        "right": right_fixture,
    })
    return len(st.session_state["parity_pairs"])

def clear_parity_pairs():
    """Clear all queued pairs."""
    st.session_state["parity_pairs"] = []

def set_parity_pairs_from_fixtures(pairs_spec: list[dict]):
    """
    Bulk load pairs from spec, parsing file paths into fixtures.
    """
    clear_parity_pairs()
    for row in pairs_spec:
        label = row.get("label", "PAIR")
        Lp = row.get("left", {})
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

# ------------------------ Report Writer ------------------------
def write_parity_report(*, mode: str, pairs_out: list[dict], policy_tag: str, projector_hash: str | None) -> str:
    """
    Generate and save parity report JSON.
    """
    payload = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "written_at_utc": hashes.timestamp_iso_lisbon(),
        "app_version": APP_VERSION_STR,
        "policy_tag": policy_tag,
        "projector_hash": projector_hash or "",
        "mode": mode,
        "pairs": pairs_out,
    }
    outdir = Path("reports")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "parity_report.json"
    _atomic_write_json(outpath, payload)
    return str(outpath)

# ------------------------ Sample pairs (optional GUI) ------------------------
def _all_exist(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)

# UI for queuing sample pairs
with st.expander("Parity: queue sample D2/D3/D4 pairs (optional)"):
    st.caption("Only queues pairs if files exist under ./inputs/.")
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
        flat_paths = []
        for r in spec:
            L, R = r["left"], r["right"]
            flat_paths += [L["boundaries"], L["cmap"], L["H"], L["shapes"], R["boundaries"], R["cmap"], R["H"], R["shapes"]]
        if not _all_exist(flat_paths):
            st.info("Example files not found under ./inputs — skipping queuing.")
        else:
            try:
                set_parity_pairs_from_fixtures(spec)
                st.success("Queued D2↔D3 and D3↔D4 example pairs.")
            except Exception as e:
                st.error(f"Could not queue examples: {e}")

# ------------------------ Parity Suite helpers (metadata & export) ------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parity_pairs_payload(pairs: list[dict]) -> dict:
    """Wrap pairs with schema info for persistence."""
    return {
        "schema_version": PARITY_SCHEMA_VERSION,
        "saved_at": _utc_now_iso(),
        "count": len(pairs),
        "pairs": [
            {
                "label": row.get("label", "PAIR"),
                # Store only paths, not parsed objects
                "left": {
                    "boundaries": row.get("left_path_boundaries", row.get("left", {}).get("boundaries_path", "")),
                    "cmap": row.get("left_path_cmap", row.get("left", {}).get("cmap_path", "")),
                    "H": row.get("left_path_H", row.get("left", {}).get("H_path", "")),
                    "shapes": row.get("left_path_shapes", row.get("left", {}).get("shapes_path", "")),
                },
                "right": {
                    "boundaries": row.get("right_path_boundaries", row.get("right", {}).get("boundaries_path", "")),
                    "cmap": row.get("right_path_cmap", row.get("right", {}).get("cmap_path", "")),
                    "H": row.get("right_path_H", row.get("right", {}).get("H_path", "")),
                    "shapes": row.get("right_path_shapes", row.get("right", {}).get("shapes_path", "")),
                },
            }
            for row in pairs
        ],
    }

def _pairs_from_payload(payload: dict) -> list[dict]:
    """Convert stored payload back into pairs spec with paths."""
    if not isinstance(payload, dict):
        return []
    return [
        {
            "label": r.get("label", "PAIR"),
            "left": {
                "boundaries": r.get("left", {}).get("boundaries", ""),
                "cmap": r.get("left", {}).get("cmap", ""),
                "H": r.get("left", {}).get("H", ""),
                "shapes": r.get("left", {}).get("shapes", ""),
            },
            "right": {
                "boundaries": r.get("right", {}).get("boundaries", ""),
                "cmap": r.get("right", {}).get("cmap", ""),
                "H": r.get("right", {}).get("H", ""),
                "shapes": r.get("right", {}).get("shapes", ""),
            },
        }
        for r in payload.get("pairs", [])
    ]

def export_parity_pairs(path: str | Path = DEFAULT_PARITY_PATH) -> str:
    """Save current session parity pairs to disk."""
    path = Path(path)
    _ensure_parent_dir(path)
    pairs = st.session_state.get("parity_pairs", []) or
    payload = _parity_pairs_payload(pairs)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        _json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)
    return str(path)

def import_parity_pairs(path: str | Path = DEFAULT_PARITY_PATH, *, merge: bool=False) -> int:
    """Load parity pairs from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No parity pairs file at {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = _json.load(f)
    # Basic schema check
    ver = payload.get("schema_version", "0.0.0")
    if ver.split(".")[0] != PARITY_SCHEMA_VERSION.split(".")[0]:
        st.warning(f"parity_pairs schema version differs (file={ver}, app={PARITY_SCHEMA_VERSION}); attempting best-effort load.")
    pairs_spec = _pairs_from_payload(payload)
    if not merge:
        clear_parity_pairs()
    set_parity_pairs_from_fixtures(pairs_spec)
    return len(st.session_state.get("parity_pairs", []))

# ------------------------ Imports & Constants ------------------------
import os
import io
import csv
import json as _json
import tempfile
import zipfile
import platform
from pathlib import Path
from datetime import datetime, timezone

# Paths & Versions
PERTURB_SCHEMA_VERSION = "1.0.0"
FENCE_SCHEMA_VERSION = "1.0.0"
COVERAGE_SCHEMA_VERSION = "1.0.0"
EXPORT_SCHEMA_VERSION = "1.0.0"
SELFTESTS_SCHEMA_VERSION = "1.0.0"

PERTURB_OUT_PATH = Path("reports") / "perturbation_sanity.csv"
FENCE_OUT_PATH = Path("reports") / "fence_stress.csv"
COVERAGE_DEFAULT_PATH = Path("reports") / "coverage_sampling.csv"

EXPORT_DIR = Path("bundles")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Utility Functions
def _utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _atomic_write_csv(path: Path, header, rows, meta_comments):
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

def _atomic_write(path: Path, bytes_blob: bytes):
    _ensure_parent(path)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as tmp:
        tmp.write(bytes_blob)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def _json_bytes(obj):
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

# ------------------------ GF(2) Helpers ------------------------
def _mat(M): return M if isinstance(M, list) else []
def _clone_mat(M): return [row[:] for row in _mat(M)]
def _mul_gf2(A, B):
    if not A or not A[0] or not B or not B[0]:
        return []
    r, k = len(A), len(A[0])
    k2, c = len(B), len(B[0])
    if k != k2:
        return []
    out = [[0] * c for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for t in range(k):
            if Ai[t] & 1:
                Bt = B[t]
                for j in range(c):
                    out[i][j] ^= (Bt[j] & 1)
    return out

def _eye(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

# ------------------------ Fixture Loader ------------------------
def load_fixture_from_paths(*, boundaries_path, cmap_path, H_path, shapes_path):
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

def _safe_parse_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return _json.load(f)

# ------------------------ Parity Pairs Management ------------------------
def add_parity_pair(*, label, left_fixture, right_fixture):
    req_keys = ("boundaries", "cmap", "H", "shapes")
    for side_name, fx in [("left", left_fixture), ("right", right_fixture)]:
        if not isinstance(fx, dict) or any(k not in fx for k in req_keys):
            raise ValueError(f"{side_name} fixture malformed; expected keys {req_keys}")
    st.session_state.setdefault("parity_pairs", [])
    st.session_state["parity_pairs"].append({
        "label": label,
        "left": left_fixture,
        "right": right_fixture,
    })
    return len(st.session_state["parity_pairs"])

def clear_parity_pairs():
    st.session_state["parity_pairs"] = []

def set_parity_pairs_from_fixtures(pairs_spec):
    clear_parity_pairs()
    for row in pairs_spec:
        label = row.get("label", "PAIR")
        Lp = row.get("left", {})
        Rp = row.get("right", {})
        L = load_fixture_from_paths(**Lp)
        R = load_fixture_from_paths(**Rp)
        add_parity_pair(label=label, left_fixture=L, right_fixture=R)
    return len(st.session_state.get("parity_pairs", []))

# ------------------------ Parity Report Writer ------------------------
def write_parity_report(*, mode, pairs_out, policy_tag, projector_hash):
    payload = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "written_at_utc": hashes.timestamp_iso_lisbon(),
        "app_version": APP_VERSION_STR,
        "policy_tag": policy_tag,
        "projector_hash": projector_hash or "",
        "mode": mode,
        "pairs": pairs_out,
    }
    outdir = Path("reports")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "parity_report.json"
    _atomic_write_json(outpath, payload)
    return str(outpath)

# ------------------------ Sample Pairs UI ------------------------
def _all_exist(paths):
    return all(Path(p).exists() for p in paths)

with st.expander("Parity: queue sample D2/D3/D4 pairs (optional)"):
    st.caption("Only queues pairs if the files exist under ./inputs/.")
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
        flat_paths = []
        for r in spec:
            L, R = r["left"], r["right"]
            flat_paths += [L["boundaries"], L["cmap"], L["H"], L["shapes"], R["boundaries"], R["cmap"], R["H"], R["shapes"]]
        if not _all_exist(flat_paths):
            st.info("Example files not found under ./inputs — skipping queuing.")
        else:
            try:
                set_parity_pairs_from_fixtures(spec)
                st.success("Queued D2↔D3 and D3↔D4 example pairs.")
            except Exception as e:
                st.error(f"Could not queue examples: {e}")

# ------------------------ Cert & Signature Management ------------------------
def generate_cert_payload():
    # Generate your cert payload here, combining all relevant data
    # This is a simplified placeholder, implement your logic accordingly
    pass

def write_cert_and_bundle():
    # Do your cert writing, bundle creation, etc.
    pass

# ------------------------ Self-tests & Diagnostics ------------------------
def run_self_tests():
    failures, warnings, notes = [], [], []

    try:
        di = st.session_state.get("_district_info", {}) or {}
        inputs = st.session_state.get("_inputs_block", {}) or {}
        run_ctx = st.session_state.get("run_ctx", {}) or {}
        ab_ctx = st.session_state.get("ab_compare", {}) or {}

        # Run your various checks, append to failures/warnings
        # Example: hash coherence check
        bh_inputs = inputs.get("boundaries_hash", "")
        bh_di = di.get("boundaries_hash", "")
        if bh_inputs and bh_di and (bh_inputs != bh_di):
            failures.append(f"Hash coherence: ...")
        # More checks here...

        # Final snapshot
        st.session_state["selftests_snapshot"] = {
            "schema_version": SELFTESTS_SCHEMA_VERSION,
            "created_at": _utc_iso(),
            "failures": failures,
            "warnings": warnings,
            "notes": notes,
        }

        if failures:
            st.error("❌ Self-tests FAILED — plumbing not healthy.")
        else:
            st.success("✅ Self-tests passed.")
            if warnings:
                st.warning("Notes / warnings:\n" + "\n".join(f"- {w}" for w in warnings))
        return failures, warnings

    except Exception as e:
        st.error(f"Self-tests threw an exception: {e}")

# ------------------------ Modular UI & Workflow (example) ------------------------
# Wrap your main UI logic into functions, call them sequentially, or based on buttons.





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
