# --- robust loader with real package context (supports app/otcore or app/core) ---
import sys, pathlib, importlib.util, types
import streamlit as st
import json
import json as _json
import hashlib as _hashlib
from otcore import cert_helpers as cert
from otcore import export as export_mod
import os
from otcore.linalg_gf2 import mul, add, eye
from io import BytesIO
import zipfile


st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# --- Policy helpers -----------------------------------------------------------
# ---- cert helpers: diagnostics and checks (GF(2)-safe) ----

def projector_provenance_hash(*, cfg: dict, lane_mask_k3: list[int], district_id: str = "D3") -> str:
    """
    Build a minimal, normalized spec for the k=3 projector and hash it.
    Works for AUTO or FILE; does NOT read matrices, so it's reproducible.
    """
    src  = (cfg or {}).get("source", {}).get("3", "auto")
    mode = (cfg or {}).get("modes",  {}).get("3", "columns")
    enabled = (cfg or {}).get("enabled_layers", [])
    spec = {
        "mode":           mode,
        "k":              3,
        "selection":      ("file" if src == "file" else "auto"),
        "lane_mask_k3":   list(map(int, lane_mask_k3 or [])),
        "district_id":    district_id,
        "enabled_layers": enabled,
    }
    norm = _json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return _hashlib.sha256(norm).hexdigest()

def _lane_mask_from_d3(boundaries) -> list[int]:
    """Derive lane_mask_k3 from d3: column j is lane if any bit in that column is 1."""
    try:
        d3 = boundaries.blocks.__root__.get("3")
    except Exception:
        d3 = None
    if not d3 or not d3[0]:
        return []
    mask = []
    for j in range(len(d3[0])):
        mask.append(1 if any((row[j] & 1) for row in d3) else 0)
    return mask

def projector_provenance_hash(cfg: dict, *, boundaries, district_id: str, diagnostics_block: dict | None = None) -> str:
    """
    Stable hash for projector provenance.
    - If file-backed: hash the file content (if you want, keep your existing path).
    - For AUTO: hash a normalized descriptor that captures the effective selection.
    """
    # Try file-backed first
    try:
        src = (cfg or {}).get("source", {}).get("3")
        pj_path = (cfg or {}).get("projector_files", {}).get("3")
        if src == "file" and pj_path:
            with open(pj_path) as f:
                P = _json.load(f)
            # normalize matrix content to stable string before hashing
            s = _json.dumps(P, sort_keys=True, separators=(",", ":"))
            return _hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        pass

    # AUTO (or anything else): build a normalized descriptor
    mode = (cfg or {}).get("modes", {}).get("3", "columns")
    selection = (cfg or {}).get("source", {}).get("3", "auto")
    enabled = (cfg or {}).get("enabled_layers", [])
    lane_mask = (diagnostics_block or {}).get("lane_mask_k3") or _lane_mask_from_d3(boundaries)

    descriptor = {
        "mode": mode,              # "columns"
        "k": 3,
        "selection": selection,    # "auto"
        "enabled_layers": enabled, # [3] or []
        "lane_mask_k3": lane_mask, # e.g., [1,1,0]
        "district_id": district_id or "unknown",
    }
    s = _json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return _hashlib.sha256(s.encode("utf-8")).hexdigest()

def projector_hash_for_cfg(cfg: dict) -> str:
    """Return hash of projector_D3.json if k=3 is file-backed, else ''."""
    try:
        if cfg.get("source", {}).get("3") != "file":
            return ""
        pj_path = cfg.get("projector_files", {}).get("3")
        if not pj_path or not os.path.exists(pj_path):
            return ""
        with open(pj_path) as f:
            P = _json.load(f)
        return projector._hash_matrix(P)
    except Exception:
        return ""
        
import os

def projector_hashes_from_context(cfg: dict, boundaries, cache=None):
    """
    Returns a tuple: (file_hash, runtime_hash)

    file_hash: hash of projector_D3.json if a file path exists (even if source=='auto').
    runtime_hash: hash of the live projector derived from d3 (columns@k=3), for AUTO runs.
    """
    file_hash = ""
    runtime_hash = ""

    # Try file-backed first (even if source==auto, but a path exists)
    try:
        pj_path = (cfg or {}).get("projector_files", {}).get("3")
        if pj_path and os.path.exists(pj_path):
            with open(pj_path) as f:
                P = _json.load(f)
            file_hash = projector._hash_matrix(P)
    except Exception:
        pass

    # Always try to compute a runtime projector hash from d3 (for AUTO provenance)
    try:
        d3 = boundaries.blocks.__root__.get("3")
        if d3:
            P_rt = projector.projector_columns_from_dkp1(d3)  # columns @ k=3
            runtime_hash = projector._hash_matrix(P_rt)
    except Exception:
        pass

    return file_hash, runtime_hash


def gallery_key_from(*, cert_payload: dict, cmap, diagnostics_block: dict, policy_label: str):
    """
    Tuple key that defines a 'lane exemplar' to dedupe gallery rows:
    (district_id, boundaries_hash, U_hash, suppC_hash, lane_vec_H2d3, policy_mode)
    """
    district_id     = cert_payload["identity"]["district_id"]
    boundaries_hash = cert_payload["inputs"]["boundaries_hash"]
    U_hash          = cert_payload["inputs"]["U_hash"]
    suppC_hash      = hashes.hash_suppC(cmap)  # support hash for C's k=3 block
    lane_vec        = diagnostics_block.get("lane_vec_H2d3") or []
    lane_vec_key    = "".join(str(int(x)) for x in lane_vec)
    policy_mode     = policy_label  # e.g., "strict" or "projected(columns@k=3,auto)"
    return (district_id, boundaries_hash, U_hash, suppC_hash, lane_vec_key, policy_mode)


def _mat(M):  # ensure list[list[int]]
    return M if isinstance(M, list) else []

def _bottom_row(M):
    M = _mat(M)
    if not M: return []
    return M[-1] if M[-1] else []

def _lane_mask_from_d3(boundaries):
    try:
        d3 = boundaries.blocks.__root__.get("3")
    except Exception:
        d3 = None
    if not d3 or not d3[0]:
        return []
    ncols = len(d3[0])
    return [1 if any(r[j] & 1 for r in d3) else 0 for j in range(ncols)]

def _restrict_to_lanes(vec, lane_mask):
    if not vec or not lane_mask: return []
    n = min(len(vec), len(lane_mask))
    return [vec[j] for j in range(n) if lane_mask[j] == 1]

def _gf2_add(A, B):  # elementwise XOR for equal shapes
    if not A or not B: return A or B or []
    r = len(A); c = len(A[0])
    if len(B) != r or len(B[0]) != c:
        return A  # shape mismatch → leave A
    return [[(A[i][j] ^ B[i][j]) for j in range(c)] for i in range(r)]

def _eye(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def _stamp_filename(state_key: str, f):
    """Remember the uploaded filename in session_state for certs/registry."""
    if f is not None:
        st.session_state[state_key] = getattr(f, "name", "")
    else:
        st.session_state.pop(state_key, None)


def _mul_gf2(A, B):
    # very small, safe GF(2) multiply with guards
    if not A or not A[0] or not B or not B[0]:
        return []
    r, k  = len(A), len(A[0])
    k2, c = len(B), len(B[0])
    if k != k2:
        return []
    out = [[0]*c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            s = 0
            for t in range(k):
                s ^= (A[i][t] & B[t][j])
            out[i][j] = s
    return out

def _support(M):
    M = _mat(M)
    return [[1 if (v & 1) else 0 for v in row] for row in M] if M else []

def _subset_support(small, big):
    # is supp(small) ⊆ supp(big) ? both are 0/1 matrices
    if not small: return True
    if not big:   return False
    if len(small) > len(big) or len(small[0]) > len(big[0]):
        return False
    r = len(small); c = len(small[0])
    for i in range(r):
        for j in range(c):
            if small[i][j] and not big[i][j]:
                return False
    return True

def _grid_flags(boundaries, cmap):
    # Grid @ k=3: d3 C3 == C2 d3 (if all present). If missing, treat as True.
    try:
        d3 = boundaries.blocks.__root__.get("3")
        C3 = cmap.blocks.__root__.get("3")
        C2 = cmap.blocks.__root__.get("2")
    except Exception:
        d3 = C3 = C2 = None
    if not d3 or not C3 or not C2:
        return True
    left  = _mul_gf2(d3, C3)
    right = _mul_gf2(C2, d3)
    return (left == right)

def _fence_flags(cmap, H, shapes):
    # Try to read a carrier mask U_k (same shape as blocks). If not available → True.
    def _get_U(k):
        # Support common patterns; fall back to None.
        try:
            if hasattr(shapes, "blocks"):
                return shapes.blocks.__root__.get(k)
            if isinstance(shapes, dict):
                # shapes may be { "blocks": {...} } or directly {k: mask}
                if "blocks" in shapes and isinstance(shapes["blocks"], dict):
                    return shapes["blocks"].get(k)
                return shapes.get(k)
        except Exception:
            return None
        return None

    try:
        C3 = cmap.blocks.__root__.get("3"); C2 = cmap.blocks.__root__.get("2")
        H2 = H.blocks.__root__.get("2") if H else None
    except Exception:
        C3 = C2 = H2 = None

    U3 = _get_U("3")
    U2 = _get_U("2")

    okC3 = True if C3 is None or U3 is None else _subset_support(_support(C3), _support(U3))
    okC2 = True if C2 is None or U2 is None else _subset_support(_support(C2), _support(U2))
    okH2 = True if H2 is None or U3 is None else _subset_support(_support(H2), _support(U3))
    return (okC3 and okC2 and okH2)

def _safe_dict(x):
    try:
        return x.dict()
    except Exception:
        return x


def cfg_strict():
    # strict = no projection anywhere
    return {
        "enabled_layers": [],
        "modes": {},
        "source": {},
        "projector_files": {},
    }

def cfg_projected_base():
    # default projected: columns @ k=3, auto source
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
        mode = cfg.get("modes", {}).get(str(kk), "none")
        src  = cfg.get("source", {}).get(str(kk), "auto")
        parts.append(f"{mode}@k={kk},{src}")
    return "projected(" + "; ".join(parts) + ")"

# --- File helpers -------------------------------------------------------------
def _stamp_filename(state_key: str, f):
    """Remember the uploaded filename in session_state for certs/registry."""
    if f is not None:
        st.session_state[state_key] = getattr(f, "name", "")
    else:
        st.session_state.pop(state_key, None)

def read_json_file(f):
    if not f:
        return None
    try:
        import json  # make sure json is imported at top too
        return json.load(f)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return None



# --- cert writer (save one result to certs/...) -------------------------------
from pathlib import Path
import json as _json

def _short(s: str, n: int = 12) -> str:
    return s[:n] if s else ""

def policy_tag_for_filename(label: str) -> str:
    # turn "projected(columns@k=3,auto)" into "projected_columns_k3_auto"
    return (
        label.replace("projected(", "projected_")
             .replace(")", "")
             .replace("@", "_")
             .replace(";", "_")
             .replace(",", "_")
             .replace("=", "")
             .replace(" ", "")
    )

def write_overlap_cert(*, out: dict, policy_label: str, boundaries, cmap, H, pj_hash: str | None = None, cert_dir: str = "certs") -> str:
    Path(cert_dir).mkdir(exist_ok=True)
    payload = {
        "policy": policy_label,
        "k2": out.get("2", {}),
        "k3": out.get("3", {}),
        "hashes": {
            "hash_d": hashes.hash_d(boundaries),
            "hash_U": hashes.hash_U(globals().get("shapes")) if "shapes" in globals() else "",
            "hash_suppC": hashes.hash_suppC(cmap),
            "hash_suppH": hashes.hash_suppH(H),
            "hash_P": pj_hash or "",
        },
        "app": {
            "version": getattr(hashes, "APP_VERSION", "v0.1-core"),
            "run_id": hashes.run_id(
                content_hash := hashes.bundle_content_hash([
                    ("d", boundaries.dict() if hasattr(boundaries, "dict") else {}),
                    ("C", cmap.dict() if hasattr(cmap, "dict") else {}),
                    ("H", H.dict() if hasattr(H, "dict") else {}),
                ]),
                hashes.timestamp_iso_lisbon(),
            ),
            "content_hash": content_hash,
        },
    }
    fname = f"overlap_pass__{policy_tag_for_filename(policy_label)}__{_short(payload['app']['run_id'])}.json"
    fpath = str(Path(cert_dir) / fname)
    with open(fpath, "w") as f:
        _json.dump(payload, f, indent=2)
    return fpath



# 1) Locate package dir and set PKG_NAME
HERE = pathlib.Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
CORE   = HERE / "core"
PKG_DIR = OTCORE if OTCORE.exists() else CORE
PKG_NAME = "otcore" if OTCORE.exists() else "core"

# Create a lightweight package object so relative imports inside modules work
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PKG_DIR)]
    pkg.__file__ = str(PKG_DIR / "__init__.py")
    sys.modules[PKG_NAME] = pkg

# 2) Minimal loader that loads modules from PKG_DIR by filename
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

# 3) Force fresh imports of overlap_gate/projector from the package on disk
import importlib
for _mod in (f"{PKG_NAME}.overlap_gate", f"{PKG_NAME}.projector"):
    if _mod in sys.modules:
        del sys.modules[_mod]

overlap_gate = _load_pkg_module(f"{PKG_NAME}.overlap_gate", "overlap_gate.py")
projector    = _load_pkg_module(f"{PKG_NAME}.projector",    "projector.py")

# 4) Load the rest of your modules from the same package
io            = _load_pkg_module(f"{PKG_NAME}.io",            "io.py")
hashes        = _load_pkg_module(f"{PKG_NAME}.hashes",        "hashes.py")
unit_gate     = _load_pkg_module(f"{PKG_NAME}.unit_gate",     "unit_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers        = _load_pkg_module(f"{PKG_NAME}.towers",        "towers.py")
export_mod    = _load_pkg_module(f"{PKG_NAME}.export",        "export.py")

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")
# -----------------------------------------------------------------------------


# (After set_page_config you can safely use other st.* calls)
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")

# Optional debug: show exactly which files were loaded
st.caption(f"overlap_gate loaded from: {getattr(overlap_gate, '__file__', '<none>')}")
st.caption(f"projector loaded from: {getattr(projector, '__file__', '<none>')}")

def read_json_file(file):
    if not file: return None
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return None
       # ---- District hash → ID map (authoritative) ---------------------------------
# sha256(boundaries.json RAW BYTES) -> district label
DISTRICT_MAP = {
    "9da8b7f605c113ee059160cdaf9f93fe77e181476c72e37eadb502e7e7ef9701": "D1",
    "4356e6b608443b315d7abc50872ed97a9e2c837ac8b85879394495e64ec71521": "D2",
    "28f8db2a822cb765e841a35c2850a745c667f4228e782d0cfdbcb710fd4fecb9": "D3",
    "aea6404ae680465c539dc4ba16e97fbd5cf95bae5ad1c067dc0f5d38ca1437b5": "D4",
}

# ============================== SIDEBAR ======================================
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

    # Filename stamps for provenance
    _stamp_filename("fname_shapes", f_shapes)
    _stamp_filename("fname_boundaries", f_bound)
    _stamp_filename("fname_cmap", f_cmap)

    # Tiny helpers (scoped here)
    import hashlib, json
    def _sha256_hex_bytes(b: bytes) -> str:
        h = hashlib.sha256(); h.update(b); return h.hexdigest()
    def _sha256_hex_obj(obj) -> str:
        s = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
        return _sha256_hex_bytes(s)
    def _lane_mask_from_d3(d3_mat):
        if not d3_mat or not d3_mat[0]:
            return []
        cols = len(d3_mat[0])
        return [1 if any(row[j] & 1 for row in d3_mat) else 0 for j in range(cols)]
    def _district_signature(mask, r, c) -> str:
        payload = f"k3:{''.join(str(x) for x in mask)}|r{r}|c{c}".encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    # Peek raw-bytes hash so you can populate DISTRICT_MAP
    if f_bound is not None and hasattr(f_bound, "getvalue"):
        _raw  = f_bound.getvalue()
        _bhash = hashlib.sha256(_raw).hexdigest()
        st.caption(f"boundaries raw-bytes hash: {_bhash}")
        st.code(f'DISTRICT_MAP["{_bhash}"] = "D?"  # ← set D1/D2/D3/D4', language="python")

# ---- load jsons -------------------------------------------------------------
d_shapes = read_json_file(f_shapes)
d_bound  = read_json_file(f_bound)
d_cmap   = read_json_file(f_cmap)

# Shared inputs_block in session for filenames/hashes
if "_inputs_block" not in st.session_state:
    st.session_state["_inputs_block"] = {}
inputs_block = st.session_state["_inputs_block"]

if d_shapes and d_bound and d_cmap:
    try:
        shapes     = io.parse_shapes(d_shapes)
        boundaries = io.parse_boundaries(d_bound)
        cmap       = io.parse_cmap(d_cmap)
        support    = io.parse_support(read_json_file(f_support)) if f_support else None
        triangle   = io.parse_triangle_schema(read_json_file(f_triangle)) if f_triangle else None

        # ---- Step 1: bind district from fresh boundaries --------------------
        # 1) fresh boundaries hash (prefer raw bytes)
        try:
            if hasattr(f_bound, "getvalue"):
                _raw = f_bound.getvalue()
                boundaries_hash_fresh = _sha256_hex_bytes(_raw)
            else:
                boundaries_hash_fresh = _sha256_hex_obj(d_bound)
        except Exception:
            boundaries_hash_fresh = _sha256_hex_obj(d_bound)

        # 2) fresh k=3 mask/signature
        d3_block         = (boundaries.blocks.__root__.get("3") or [])
        lane_mask_k3_now = _lane_mask_from_d3(d3_block)
        d3_rows          = len(d3_block)
        d3_cols          = (len(d3_block[0]) if d3_block else 0)
        district_sig     = _district_signature(lane_mask_k3_now, d3_rows, d3_cols)

        # 3) resolve district via DISTRICT_MAP (no UI cache)
        district_id_fresh = DISTRICT_MAP.get(boundaries_hash_fresh, "UNKNOWN")

        # 4) clear stale if boundaries changed
        _prev_bhash = st.session_state.get("_last_boundaries_hash")
        if _prev_bhash and _prev_bhash != boundaries_hash_fresh:
            st.session_state.pop("ab_compare", None)
            st.session_state.pop("district_id", None)
            st.session_state.pop("_projector_cache", None)
        st.session_state["_last_boundaries_hash"] = boundaries_hash_fresh

        # 5) stamp filenames + authoritative hashes
        inputs_block["boundaries_filename"] = st.session_state.get("fname_boundaries", "boundaries.json")
        inputs_block["boundaries_hash"]     = boundaries_hash_fresh
        inputs_block["shapes_filename"]     = st.session_state.get("fname_shapes", "shapes.json")
        inputs_block["cmap_filename"]       = st.session_state.get("fname_cmap", "cmap.json")
        inputs_block.setdefault("U_filename", "shapes.json")

        # 6) mirror fresh district info for later blocks
        st.session_state["_district_info"] = {
            "district_id":        district_id_fresh,
            "boundaries_hash":    boundaries_hash_fresh,
            "lane_mask_k3_now":   lane_mask_k3_now,
            "district_signature": district_sig,
            "d3_rows": d3_rows,
            "d3_cols": d3_cols,
        }

        # validate & provenance UI
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

# ================================ TABS =======================================
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

        # Re-bind district from Unit override (raw-bytes hash)
        import hashlib, json
        def _sha256_hex_bytes(b: bytes) -> str:
            h = hashlib.sha256(); h.update(b); return h.hexdigest()
        def _sha256_hex_obj(obj) -> str:
            s = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
            return _sha256_hex_bytes(s)

        try:
            if hasattr(f_B, "getvalue"):
                _rawB = f_B.getvalue()
                boundaries_hash_fresh = _sha256_hex_bytes(_rawB)
            else:
                boundaries_hash_fresh = _sha256_hex_obj(d_B)
        except Exception:
            boundaries_hash_fresh = _sha256_hex_obj(d_B)

        d3_block = (boundaries.blocks.__root__.get("3") or [])
        lane_mask_k3_now = _lane_mask_from_d3(d3_block)
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

# ----------------------------- OVERLAP TAB -----------------------------------
with tab2:
    st.subheader("Overlap gate (homotopy vs identity)")

    # H uploader (+ remember filename for certs/bundles)
    f_H = st.file_uploader("Homotopy H (H_corrected.json)", type=["json"], key="H_corr")
    _stamp_filename("fname_h", f_H)
    d_H = read_json_file(f_H) if f_H else None
    H_local = io.parse_cmap(d_H) if d_H else io.parse_cmap({"blocks": {}})  # JSON-safe CMap

    # Policy toggle UI
    st.markdown("### Policy")
    policy_choice = st.radio(
        "Choose policy",
        ["strict", "projected(columns@k=3)"],
        horizontal=True,
        key="policy_choice_k3",
    )


        # --- Build active cfg (respect file/auto from projection_config.json) ---
    import os, json as _json
    from pathlib import Path

    # load persisted projection config
    cfg_file = projector.load_projection_config("projection_config.json")
    cfg_proj = cfg_projected_base()

    # inherit source / file path from projection_config.json
    if cfg_file.get("source", {}).get("3") in ("file", "auto"):
        cfg_proj["source"]["3"] = cfg_file["source"]["3"]
    if "projector_files" in cfg_file and "3" in cfg_file["projector_files"]:
        cfg_proj.setdefault("projector_files", {})["3"] = cfg_file["projector_files"]["3"]

    # choose strict vs projected
    cfg_active = cfg_strict() if policy_choice == "strict" else cfg_proj

    # -------- Optional: upload a projector file to use in projected(file) -----
    st.markdown("#### Projector (k=3)")
    proj_upload = st.file_uploader(
        "Projector JSON for k=3 (optional; used when source.3 = file)",
        type=["json"], key="proj3_up"
    )
    proj_saved_path = None
    if proj_upload is not None:
        proj_dir = Path("projectors")
        proj_dir.mkdir(exist_ok=True, parents=True)
        proj_saved_path = proj_dir / proj_upload.name
        with open(proj_saved_path, "wb") as _pf:
            _pf.write(proj_upload.getvalue())
        st.caption(f"saved projector: {proj_saved_path}")

        # if user uploaded, switch active cfg to file mode and point to this file
        if policy_choice != "strict":
            cfg_active.setdefault("source", {})["3"] = "file"
            cfg_active.setdefault("projector_files", {})["3"] = str(proj_saved_path)

    # --- Projector source switcher UI (writes projection_config.json) ----------
    with st.expander("Projector source (k=3)"):
        cur_src  = cfg_file.get("source", {}).get("3", "auto")
        cur_file = cfg_file.get("projector_files", {}).get("3", "projectors/projector_D3.json")
        st.write(
            f"Current: source.3 = **{cur_src}**",
            f"(file: `{cur_file}`)" if cur_src == "file" else ""
        )
        mode_choice = st.radio(
            "Choose source for k=3",
            options=["auto","file"],
            index=(0 if cur_src == "auto" else 1),
            horizontal=True,
            key="proj_src_choice_k3",
        )
        file_path = st.text_input("Projector file", value=cur_file, disabled=(mode_choice=="auto"))
        if st.button("Apply projector source", key="apply_proj_src_k3"):
            cfg_file.setdefault("source", {})["3"] = mode_choice
            if mode_choice == "file":
                cfg_file.setdefault("projector_files", {})["3"] = file_path
            else:
                if "projector_files" in cfg_file and "3" in cfg_file["projector_files"]:
                    del cfg_file["projector_files"]["3"]
            with open("projection_config.json", "w") as _f:
                _json.dump(cfg_file, _f, indent=2)
            st.success(f"projection_config.json updated → source.3 = {mode_choice}")

            # also reflect in current active config if we are in projected mode
            if policy_choice != "strict":
                cfg_active.setdefault("source", {})["3"] = mode_choice
                if mode_choice == "file":
                    cfg_active.setdefault("projector_files", {})["3"] = file_path
                else:
                    if "projector_files" in cfg_active and "3" in cfg_active["projector_files"]:
                        del cfg_active["projector_files"]["3"]

        # --- Freeze current AUTO projector to a file (one-click) ------------------
    import os, json as _json

    with st.expander("Freeze AUTO projector → file"):
        def _lane_mask_from_d3(_d3):
            if not _d3 or not _d3[0]:
                return []
            cols = len(_d3[0])
            return [1 if any(row[j] & 1 for row in _d3) else 0 for j in range(cols)]

        default_name = "projectors/projector_auto_k3.json"
        freeze_path = st.text_input("Output file", value=default_name, key="freeze_proj_out")

        st.caption("Build a diagonal Π₃ from the current d₃ mask and switch source.3 → file.")
        if st.button("Freeze now", key="freeze_proj_btn"):
            try:
                # derive n3 and mask from the currently loaded boundaries
                d3 = (boundaries.blocks.__root__.get("3") or [])
                n3 = len(d3[0]) if (d3 and d3[0]) else 0
                auto_mask = _lane_mask_from_d3(d3)

                # build a diagonal projector with that mask
                P = [[1 if (i == j and auto_mask[i] == 1) else 0 for j in range(n3)] for i in range(n3)]
                payload = {"name": "Π3 from current d3 (AUTO freeze)", "blocks": {"3": P}}

                os.makedirs(os.path.dirname(freeze_path), exist_ok=True)
                with open(freeze_path, "w") as fp:
                    _json.dump(payload, fp, indent=2)

                # flip to file mode (in-memory)
                cfg_active.setdefault("source", {})["3"] = "file"
                cfg_active.setdefault("projector_files", {})["3"] = freeze_path

                # flip to file mode (persisted on disk)
                _cfg_disk = projector.load_projection_config("projection_config.json")
                _cfg_disk.setdefault("source", {})["3"] = "file"
                _cfg_disk.setdefault("projector_files", {})["3"] = freeze_path
                with open("projection_config.json", "w") as _f:
                    _json.dump(_cfg_disk, _f, indent=2)

                st.success(f"Projector frozen → {freeze_path} and source.3 set to file")
                st.caption("Tip: now click “Run Overlap” to validate shape/idempotence/diag and record metadata.")

            except Exception as e:
                st.error(f"Freeze failed: {e}")

                        

    # --- Active policy badge ---------------------------------------------------
    src3 = cfg_active.get("source", {}).get("3", "")
    if policy_choice == "strict":
        _policy_mode_badge = "strict"
    elif src3 == "file":
        _policy_mode_badge = "projected(file)"
    else:
        _policy_mode_badge = "projected(auto)"
    policy_label = policy_label_from_cfg(cfg_active)
    st.caption(f"Policy: **{policy_label}** · mode: {_policy_mode_badge}")

    # --- Preload projectors AFTER cfg_active is finalized ---------------------
    cache = projector.preload_projectors_from_files(cfg_active)
    st.session_state["_projector_cache"] = cache  # keep around for cert validation

            # ---------- RUN OVERLAP ----------
    def _gf2_idempotent(P):
        try:
            n = len(P); m = len(P[0]) if n else 0
            if n != m:  # must be square
                return False
            PP = [[0]*n for _ in range(n)]
            for i in range(n):
                for k in range(n):
                    if P[i][k] & 1:
                        rowk = P[k]
                        for j in range(n):
                            PP[i][j] ^= (rowk[j] & 1)
            for i in range(n):
                for j in range(n):
                    if (PP[i][j] & 1) != (P[i][j] & 1):
                        return False
            return True
        except Exception:
            return False

    def _diag_vec(P):
        try:
            n = min(len(P), len(P[0]))
            return [int(P[i][i] & 1) for i in range(n)]
        except Exception:
            return []

    def _lane_mask_from_d3(d3_mat):
        if not d3_mat or not d3_mat[0]:
            return []
        cols = len(d3_mat[0])
        return [1 if any(row[j] & 1 for row in d3_mat) else 0 for j in range(cols)]

    def _extract_P3_matrix(obj):
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            return obj
        if isinstance(obj, dict):
            b = obj.get("blocks", {})
            mat = b.get("3")
            if isinstance(mat, list) and mat and isinstance(mat[0], list):
                return mat
        return None

    if st.button("Run Overlap", key="run_overlap"):
        try:
            out = overlap_gate.overlap_check(
                boundaries,
                cmap,
                H_local,                      # use the one we built above
                projection_config=cfg_active, # strict/projected as chosen
                projector_cache=cache,
            )
            st.json(out)

            # ---- projector(file) metadata & validation for cert --------------
            proj_meta = {"projector_filename": "", "projector_hash": "", "projector_consistent_with_d": None}
            if policy_choice != "strict" and cfg_active.get("source", {}).get("3") == "file":
                pj_path = cfg_active.get("projector_files", {}).get("3")
                if pj_path and os.path.exists(pj_path):
                    # load & normalize matrix
                    try:
                        with open(pj_path, "r") as _pf:
                            _raw = _json.load(_pf)
                        P3 = _extract_P3_matrix(_raw)
                    except Exception as e:
                        P3 = None
                        st.warning(f"Could not read projector file '{pj_path}': {e}")

                    # hash projector deterministically
                    proj_hash = ""
                    if P3 is not None:
                        try:
                            proj_hash = hashes.content_hash_of({"P3": P3})
                        except Exception:
                            try:
                                proj_hash = hashlib.sha256(
                                    _json.dumps(P3, sort_keys=True, separators=(",", ":")).encode()
                                ).hexdigest()
                            except Exception:
                                proj_hash = ""

                    # validations (use n3 = number of columns of d3; must be n3 x n3)
                    d3 = (boundaries.blocks.__root__.get("3") or [])
                    n3_cols = len(d3[0]) if d3 and d3[0] else 0   # lanes at k=3
                    shape_ok = False
                    idem_ok  = False
                    diag_ok  = True
                    if isinstance(P3, list) and P3 and isinstance(P3[0], list):
                        nP = len(P3); mP = len(P3[0])
                        shape_ok = (nP == mP == n3_cols)
                        if shape_ok:
                            idem_ok  = _gf2_idempotent(P3)
                            diagP    = _diag_vec(P3)
                            auto_mask = _lane_mask_from_d3(d3)
                            diag_ok  = (diagP == auto_mask[:len(diagP)]) if diagP else True

                    consistent = bool(shape_ok and idem_ok and diag_ok)

                    if not shape_ok:
                        got_r = len(P3) if isinstance(P3, list) else 0
                        got_c = (len(P3[0]) if isinstance(P3, list) and P3 and isinstance(P3[0], list) else 0)
                        st.warning(f"Projector shape mismatch: expected {n3_cols}x{n3_cols}, got {got_r}x{got_c}.")
                    if shape_ok and not idem_ok:
                        st.warning("Projector is not idempotent (P·P != P over GF(2)).")
                    if shape_ok and not diag_ok:
                        st.warning("Projector diagonal does not match d3 auto mask.")

                    proj_meta = {
                        "projector_filename": pj_path,
                        "projector_hash": proj_hash,
                        "projector_consistent_with_d": consistent,
                    }

            # persist for downstream cert/bundle blocks
            st.session_state["overlap_out"] = out
            st.session_state["overlap_cfg"] = cfg_active
            st.session_state["overlap_policy_label"] = policy_label
            st.session_state["overlap_H"] = H_local
            st.session_state["proj_meta"] = proj_meta

        except Exception as e:
            st.error(f"Overlap run failed: {e}")
            st.stop()


            # ---- projector(file) metadata & validation for cert --------------
            proj_meta = {"projector_filename": "", "projector_hash": "", "projector_consistent_with_d": None}
            if policy_choice != "strict" and cfg_active.get("source", {}).get("3") == "file":
                pj_path = cfg_active.get("projector_files", {}).get("3")
                if pj_path and os.path.exists(pj_path):
                    # load matrix
                    try:
                        with open(pj_path, "r") as _pf:
                            P3 = _json.load(_pf)
                    except Exception as e:
                        P3 = None
                        st.warning(f"Could not read projector file '{pj_path}': {e}")

                    # hash projector deterministically
                    proj_hash = ""
                    if P3 is not None:
                        try:
                            proj_hash = hashes.content_hash_of({"P3": P3})
                        except Exception:
                            try:
                                proj_hash = hashlib.sha256(
                                    _json.dumps(P3, sort_keys=True, separators=(",", ":")).encode()
                                ).hexdigest()
                            except Exception:
                                proj_hash = ""

                    # validations (use n3 = number of columns of d3; must be n3 x n3)
                    d3 = (boundaries.blocks.__root__.get("3") or [])
                    n3_cols = len(d3[0]) if d3 and d3[0] else 0   # lanes at k=3
                    shape_ok = False
                    idem_ok  = False
                    diag_ok  = True
                    if isinstance(P3, list) and P3 and isinstance(P3[0], list):
                        nP = len(P3); mP = len(P3[0])
                        shape_ok = (nP == mP == n3_cols)
                        if shape_ok:
                            idem_ok  = _gf2_idempotent(P3)
                            diagP    = _diag_vec(P3)
                            auto_mask = _lane_mask_from_d3(d3)
                            diag_ok  = (diagP == auto_mask[:len(diagP)]) if diagP else True

                    consistent = bool(shape_ok and idem_ok and diag_ok)

                    if not shape_ok:
                        got_r = len(P3) if isinstance(P3, list) else 0
                        got_c = (len(P3[0]) if isinstance(P3, list) and P3 and isinstance(P3[0], list) else 0)
                        st.warning(f"Projector shape mismatch: expected {n3_cols}x{n3_cols}, got {got_r}x{got_c}.")
                    if shape_ok and not idem_ok:
                        st.warning("Projector is not idempotent (P·P != P over GF(2)).")
                    if shape_ok and not diag_ok:
                        st.warning("Projector diagonal does not match d3 auto mask.")

                    proj_meta = {
                        "projector_filename": pj_path,
                        "projector_hash": proj_hash,
                        "projector_consistent_with_d": consistent,
                    }

            # persist for downstream cert/bundle blocks
            st.session_state["overlap_out"] = out
            st.session_state["overlap_cfg"] = cfg_active
            st.session_state["overlap_policy_label"] = policy_label
            st.session_state["overlap_H"] = H_local
            st.session_state["proj_meta"] = proj_meta

        except Exception as e:
            st.error(f"Overlap run failed: {e}")
            st.stop()


    # ===== Restore last overlap run (required by cert/bundle) =====
    _ss = st.session_state
    overlap_out          = _ss.get("overlap_out")
    overlap_cfg          = _ss.get("overlap_cfg")
    overlap_policy_label = _ss.get("overlap_policy_label")
    overlap_H            = _ss.get("overlap_H")
    proj_meta            = _ss.get("proj_meta", {"projector_filename": "", "projector_hash": "", "projector_consistent_with_d": None})

    if overlap_out is None or overlap_cfg is None or overlap_H is None:
        st.info("Run Overlap first to populate cert & download bundle.")
        st.stop()

    # use the exact data from the run
    out          = overlap_out
    cfg_active   = overlap_cfg
    policy_label = overlap_policy_label
    H_local      = overlap_H

    # small badge echo (keeps UI consistent after reload)
    src3 = cfg_active.get("source", {}).get("3", "")
    _badge = "strict" if not cfg_active.get("enabled_layers") else ("projected(file)" if src3 == "file" else "projected(auto)")
    st.caption(f"Active policy: **{policy_label}** · mode: {_badge}")

    # shapes must be JSON-safe
    shapes_payload = shapes.dict() if hasattr(shapes, "dict") else (shapes or {})


    # ---------- Inputs block (hashes + shapes + filenames) ----------
    inputs_hash = hashes.bundle_content_hash([
        ("boundaries", boundaries.dict() if hasattr(boundaries, "dict") else {}),
        ("cmap",       cmap.dict()       if hasattr(cmap,       "dict") else {}),
        ("H",          H_local.dict()    if hasattr(H_local,    "dict") else {}),
        ("cfg",        cfg_active),
    ])

    def _dims(M):
        return [len(M), len(M[0])] if (M and len(M) and M[0]) else [0, 0]

    blocks_b = boundaries.blocks.__root__
    blocks_c = cmap.blocks.__root__
    blocks_h = H_local.blocks.__root__
    shapes_block = {
        "n3": _dims(blocks_c.get("3"))[0],
        "n2": _dims(blocks_c.get("2"))[0],
    }

    inputs_block = {
        "boundaries_hash": hashes.hash_d(boundaries),
        "C_hash":          hashes.hash_suppC(cmap),
        "H_hash":          hashes.hash_suppH(H_local),
        "U_hash":          hashes.hash_U(shapes_payload),
        "shapes":          shapes_block,
    }

    # ---------- Diagnostics (lane mask + lane vectors) ----------
    d3 = blocks_b.get("3")
    lane_mask = [1 if d3 and any(row[j] & 1 for row in d3) else 0
                 for j in range(len(d3[0]))] if (d3 and len(d3) and d3[0]) else []

    def bottom_row(M):
        return M[-1] if (M and len(M)) else []

    lane_idx = [j for j, m in enumerate(lane_mask) if m == 1]
    H2d3  = mul(blocks_h.get("2", []), d3) if (blocks_h.get("2") and d3) else []
    C3pI3 = add(blocks_c.get("3", []),
                eye(len(blocks_c.get("3", []))) if blocks_c.get("3") else [])

    def mask_to_lanes(vec, mask_idx):
        return [vec[j] for j in mask_idx] if vec and mask_idx else []

    diagnostics_block = {
        "lane_mask_k3":    lane_mask,
        "lane_vec_H2d3":   mask_to_lanes(bottom_row(H2d3),  lane_idx),
        "lane_vec_C3plusI3": mask_to_lanes(bottom_row(C3pI3), lane_idx),
    }

      # ---- Signatures (rank/ker + lane patterns) ----
def _gf2_rank(M):
    """Gaussian elimination over GF(2) to compute rank."""
    if not M or not M[0]:
        return 0
    A = [row[:] for row in M]
    m, n = len(A), len(A[0])
    r = 0
    c = 0
    while r < m and c < n:
        # find pivot in column c
        pivot = None
        for i in range(r, m):
            if A[i][c] & 1:
                pivot = i
                break
        if pivot is None:
            c += 1
            continue
        # swap pivot to row r
        if pivot != r:
            A[r], A[pivot] = A[pivot], A[r]
        # eliminate this column in all other rows
        for i in range(m):
            if i != r and (A[i][c] & 1):
                A[i] = [(A[i][j] ^ A[r][j]) for j in range(n)]
        r += 1
        c += 1
    return r

d3 = boundaries.blocks.__root__.get("3") or []
n_cols_d3 = len(d3[0]) if (d3 and d3[0]) else 0
rank_d3   = _gf2_rank(d3)
ker_dim_d3 = max(0, n_cols_d3 - rank_d3)

lane_mask    = diagnostics_block.get("lane_mask_k3", []) or []
lane_pattern = "".join("1" if x else "0" for x in lane_mask) if lane_mask else ""
lane_vec_C   = diagnostics_block.get("lane_vec_C3plusI3", []) or []
fixture_lane = "".join(str(int(x)) for x in lane_vec_C) if lane_vec_C else ""

# If an old sig_block exists, preserve echo_context; otherwise set None
sig_block = {
    "d_signature":       {"rank": rank_d3, "ker_dim": ker_dim_d3, "lane_pattern": lane_pattern},
    "fixture_signature": {"lane": fixture_lane},
    "echo_context": (sig_block.get("echo_context") if 'sig_block' in locals() else None),
}


    
    # --- fixture_signature from support of (C3 + I3) restricted to lanes ----------
C3 = cmap.blocks.__root__.get("3") or []
if C3:
    n3 = len(C3)
    I3 = [[1 if i == j else 0 for j in range(n3)] for i in range(n3)]
    # GF(2): C3 + I3 is XOR on entries
    C3pI3 = [[(C3[i][j] ^ I3[i][j]) for j in range(len(C3[0]))] for i in range(n3)]
    # column support: 1 if any row has 1 in that column
    supp_cols = [
        1 if any(C3pI3[i][j] & 1 for i in range(n3)) else 0
        for j in range(len(C3pI3[0]) if C3pI3 else 0)
    ]
    fixture_lane_str = ""
    if lane_mask:
        fixture_lane_str = "".join(
            "1" if (lane_mask[j] and supp_cols[j]) else "0"
            for j in range(len(supp_cols))
        )
else:
    fixture_lane_str = ""

# ensure d_signature exists (if you computed it earlier, this keeps it)
if "d_signature" not in locals():
    d_signature = {}

fixture_signature = {"lane": fixture_lane_str}

# merge into sig_block, preserving any existing echo_context if present
_prev_echo = sig_block.get("echo_context") if "sig_block" in locals() else None
sig_block = {
    "d_signature": d_signature,
    "fixture_signature": fixture_signature,
    "echo_context": _prev_echo,
}


  # ---------- Policy snapshot (robust file hashing + schema normalize) ------
import os, json as _json, hashlib

def _extract_P3_matrix(obj):
    # Accept either raw 2-D list or {"blocks": {"3": <2-D list>}}
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return obj
    if isinstance(obj, dict):
        b = obj.get("blocks", {})
        mat = b.get("3")
        if isinstance(mat, list) and mat and isinstance(mat[0], list):
            return mat
    return None

pj_hash = ""
pj_filename = ""
if cfg_active.get("source", {}).get("3") == "file":
    pj_path = cfg_active.get("projector_files", {}).get("3")
    if pj_path and os.path.exists(pj_path):
        try:
            with open(pj_path, "r") as _pf:
                _raw = _json.load(_pf)
            _P3 = _extract_P3_matrix(_raw)  # <-- normalize schema
            # Stable deterministic hash of projector contents
            try:
                pj_hash = hashes.content_hash_of({"P3": _P3})
            except Exception:
                pj_hash = hashlib.sha256(
                    _json.dumps(_P3, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            pj_filename = pj_path
        except Exception as _e:
            st.warning(f"Could not read projector file for hashing: {pj_path} ({_e})")

policy_block = {
    "label":          policy_label,          # e.g. "projected(columns@k=3,file)"
    "policy_tag":     policy_label,
    "enabled_layers": cfg_active.get("enabled_layers", []),
    "modes":          cfg_active.get("modes", {}),
    "source":         cfg_active.get("source", {}),
    "projector_hash": pj_hash,
}
if pj_filename:
    policy_block["projector_filename"] = pj_filename


   # ---------- Residual tag (lanes / ker / mixed / none) ----------
from otcore.linalg_gf2 import mul, add, eye

def _support(M):
    return [[1 if (v % 2) != 0 else 0 for v in row] for row in (M or [])]

# Strict residual at k=3: R3 = H2@d3 + (C3 + I3)
blocks_b = boundaries.blocks.__root__
blocks_c = cmap.blocks.__root__
blocks_h = H_local.blocks.__root__
d3 = blocks_b.get("3")
C3 = blocks_c.get("3")
H2 = blocks_h.get("2")

R3_strict = []
if d3 is not None and C3 is not None and H2 is not None:
    I3 = eye(len(C3))
    R3_strict = add(mul(H2, d3), add(C3, I3))

lane_mask_k3 = diagnostics_block.get("lane_mask_k3", [])
lane_idx = [j for j, m in enumerate(lane_mask_k3) if m == 1]
ker_idx  = [j for j, m in enumerate(lane_mask_k3) if m == 0]

def _which_cols_nonzero(M):
    if not M: return set()
    nz = set()
    for r in M:
        for j, v in enumerate(r):
            if v % 2 != 0:
                nz.add(j)
    return nz

nz_cols = _which_cols_nonzero(R3_strict)
lanes_nonzero = any(j in nz_cols for j in lane_idx)
ker_nonzero   = any(j in nz_cols for j in ker_idx)
if not nz_cols:
    residual_tag = "none"
elif lanes_nonzero and not ker_nonzero:
    residual_tag = "lanes"
elif ker_nonzero and not lanes_nonzero:
    residual_tag = "ker"
else:
    residual_tag = "mixed"

# ---------- Checks (extend beyond eq) ----------
# If you have real grid/fence checks elsewhere, wire them here.
# Until then, record sensible defaults and ker_guard based on policy:
is_strict = (policy_label == "strict")
checks_extended = {
    "grid": True,                 # placeholder (set real result if you compute it)
    "fence": True,                # placeholder (set real result if you compute it)
    "ker_guard": "enforced" if is_strict else "off",
}
# Merge with your per-k out (eq, n_k)
checks_block = {**out, **checks_extended, "residual_tag": residual_tag}

# ---------- Policy snapshot (no _hash_matrix; robust hashing) -------------
import json as _json, hashlib, os

pj_hash = ""
pj_filename = ""
if cfg_active.get("source", {}).get("3") == "file":
    pj_path = cfg_active.get("projector_files", {}).get("3")
    if pj_path and os.path.exists(pj_path):
        try:
            with open(pj_path, "r") as _pf:
                _P3_json = _json.load(_pf)  # expect list-of-lists
            try:
                pj_hash = hashes.content_hash_of({"P3": _P3_json})
            except Exception:
                pj_hash = hashlib.sha256(
                    _json.dumps(_P3_json, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            pj_filename = pj_path
        except Exception as _e:
            st.warning(f"Could not read projector file for hashing: {pj_path} ({_e})")

    policy_block = {
        "label":          policy_label,          # e.g. "strict" or "projected(columns@k=3,file)"
        "policy_tag":     policy_label,
        "enabled_layers": cfg_active.get("enabled_layers", []),
        "modes":          cfg_active.get("modes", {}),
        "source":         cfg_active.get("source", {}),
        "projector_hash": pj_hash,
    }
    if pj_filename:
        policy_block["projector_filename"] = pj_filename

# ---------- Identity ----------
district_id = st.session_state.get("district_id", "D3")
run_id_val  = hashes.run_id(
    hashes.bundle_content_hash([
        ("boundaries", boundaries.dict() if hasattr(boundaries, "dict") else {}),
        ("cmap",       cmap.dict()       if hasattr(cmap,       "dict") else {}),
        ("H",          H_local.dict()    if hasattr(H_local,    "dict") else {}),
        ("cfg",        cfg_active),
    ]),
    hashes.timestamp_iso_lisbon(),
)

identity_block = {
    "district_id": district_id,
    "run_id": run_id_val,
    "timestamp": hashes.timestamp_iso_lisbon(),
    "app_version": getattr(hashes, "APP_VERSION", "v0.1-core"),
    "field": "GF(2)",
}


# ---------- Tiny polish: filenames + signatures ----------
# Do this AFTER you've created `inputs_block` and `diagnostics_block`,
# and BEFORE you assemble `cert_payload`.

# 1) Attach/ensure human filenames ONCE
inputs_block.setdefault("filenames", {})
inputs_block["filenames"].update({
    "boundaries": st.session_state.get("fname_boundaries", inputs_block["filenames"].get("boundaries", "")),
    "C":          st.session_state.get("fname_cmap",       inputs_block["filenames"].get("C", "")),
    "H":          st.session_state.get("fname_h",          inputs_block["filenames"].get("H", "")),
    "U":          st.session_state.get("fname_shapes",     inputs_block["filenames"].get("U", "")),
})

# 2) d-signature (from d3): rank, ker_dim, lane_pattern
d3 = boundaries.blocks.__root__.get("3") or []

def _gf2_rank(M):
    if not M or not M[0]:
        return 0
    # Gaussian elimination over GF(2)
    A = [row[:] for row in M]
    m, n = len(A), len(A[0])
    r, c = 0, 0
    while r < m and c < n:
        pivot = None
        for i in range(r, m):
            if A[i][c] & 1:
                pivot = i
                break
        if pivot is None:
            c += 1
            continue
        if pivot != r:
            A[r], A[pivot] = A[pivot], A[r]
        for i in range(m):
            if i != r and (A[i][c] & 1):
                A[i] = [(a ^ b) & 1 for a, b in zip(A[i], A[r])]
        r += 1
        c += 1
    return r

lane_mask    = (diagnostics_block.get("lane_mask_k3") or [])
lane_pattern = "".join("1" if x else "0" for x in lane_mask) if lane_mask else ""
rank_d3      = _gf2_rank(d3) if d3 else 0
ncols_d3     = len(d3[0]) if (d3 and d3[0]) else 0
ker_dim_d3   = max(ncols_d3 - rank_d3, 0)

d_signature = {
    "rank":        rank_d3,
    "ker_dim":     ker_dim_d3,
    "lane_pattern": lane_pattern,
}

# 3) fixture_signature: support(C3 + I3) on LANE columns only
from otcore.linalg_gf2 import add, eye

C3     = cmap.blocks.__root__.get("3") or []
C3pI3  = add(C3, eye(len(C3))) if C3 else []
lane_idxs = [j for j, m in enumerate(lane_mask) if m]

def _col_support_pattern(M, cols):
    if not M or not cols:
        return ""
    return "".join("1" if any((row[j] & 1) for row in M) else "0" for j in cols)

fixture_signature = {
    "lane": _col_support_pattern(C3pI3, lane_idxs)
}

# 4) Final signatures block (preserve previous echo_context if any)
_prev_echo = sig_block.get("echo_context") if "sig_block" in locals() else None
sig_block = {
    "d_signature": d_signature,
    "fixture_signature": fixture_signature,
    "echo_context": _prev_echo,
}

# ---- Promotion helper flags (build AFTER we have `out`) ----------------------
is_strict = (not cfg_active.get("enabled_layers"))
k3_true   = bool(out.get("3", {}).get("eq", False))

# ---- Filenames: make them NOT depend on visiting Unit tab ----
def _name_from_state(key: str) -> str:
    v = st.session_state.get(key)
    return v if isinstance(v, str) and v.strip() else ""

def _name_or_default(sesskey: str, file_obj, default_name: str) -> str:
    # 1) if user uploaded somewhere and we stamped it, prefer that
    stamped = _name_from_state(sesskey)
    if stamped:
        return stamped
    # 2) else if we have a file object here (e.g., H uploader in this tab), use its real name
    if file_obj is not None:
        nm = getattr(file_obj, "name", "")
        if isinstance(nm, str) and nm.strip():
            return nm
    # 3) else fall back to the canonical bundle filename
    return default_name

inputs_block.setdefault("filenames", {})
inputs_block["filenames"]["boundaries"] = _name_or_default("fname_boundaries", None,  "boundaries.json")
inputs_block["filenames"]["C"]          = _name_or_default("fname_cmap",       None,  "cmap.json")
inputs_block["filenames"]["U"]          = _name_or_default("fname_shapes",     None,  "shapes.json")
inputs_block["filenames"]["H"]          = _name_or_default("fname_h",          f_H,   "H.json")


# ---- Policy block (with projector provenance) --------------------------------
# Compute provenance hash for ACTIVE policy (projected only)
district_id = st.session_state.get("district_id", "D3")
lane_mask_for_active = (diagnostics_block.get("lane_mask_k3", []) if isinstance(diagnostics_block, dict) else [])

proj_hash_active = ""
if cfg_active.get("enabled_layers"):  # Only stamp when projected is active
    try:
        # Preferred: provenance hash using cfg + lane mask + district id
        proj_hash_active = projector_provenance_hash(
            cfg=cfg_active,
            lane_mask_k3=lane_mask_for_active,
            district_id=district_id,
        )
    except Exception:
        # Fallbacks for older helpers in your codebase
        try:
            file_hash_active, rt_hash_active = projector_hashes_from_context(cfg_active, boundaries, cache)
            proj_hash_active = file_hash_active or rt_hash_active or ""
        except Exception:
            try:
                proj_hash_active = projector_hash_for_cfg(cfg_active)  # last resort
            except Exception:
                proj_hash_active = ""

policy_block = {
    "label":          policy_label,                 # e.g. "strict" or "projected(columns@k=3,auto)"
    "policy_tag":     policy_label,                 # used in filenames
    "enabled_layers": cfg_active.get("enabled_layers", []),
    "modes":          cfg_active.get("modes", {}),
    "source":         cfg_active.get("source", {}),
    "projector_hash": proj_hash_active,             # provenance Π hash (or empty for strict)
}

# ---- FULL cert payload (single source of truth) ------------------------------
is_strict = (not cfg_active.get("enabled_layers"))
k3_true   = bool(out.get("3", {}).get("eq", False))

cert_payload = {
    "identity":    identity_block,
    "policy":      policy_block,
    "inputs":      inputs_block,       # includes hashes + dims + filenames
    "diagnostics": diagnostics_block,  # lane_mask_k3 / lane_vec_*
    "checks":      checks_block,       # per-k eq/n_k (+ grid/fence/ker_guard/residual_tag if present)
    "signatures":  sig_block,          # rank/ker/lane signatures
    "promotion": {
        "eligible_for_promotion": k3_true,
        "promotion_target": ("strict_anchor" if is_strict else "projected_anchor") if k3_true else None,
        "notes": "",
    },
    "policy_tag": policy_label,        # keep for exporter & bundle names
}

# ---- Optional: embed A/B snapshots into the cert (fresh-only) ---------------
ab_ctx = st.session_state.get("ab_compare", {}) or {}

# Current signature to validate freshness
_current_sig = [
    inputs_block.get("boundaries_hash", ""),
    inputs_block.get("C_hash", ""),
    inputs_block.get("H_hash", ""),
    inputs_block.get("U_hash", ""),
    inputs_block.get("shapes_hash", ""),
]

# Refuse stale A/B
if ab_ctx.get("inputs_sig") != _current_sig:
    ab_ctx = {}

strict_ctx    = ab_ctx.get("strict", {}) if ab_ctx else {}
projected_ctx = ab_ctx.get("projected", {}) if ab_ctx else {}

if ab_ctx:
    def _pass_vec_from(out_dict: dict) -> list[int]:
        return [
            int(out_dict.get("2", {}).get("eq", False)),
            int(out_dict.get("3", {}).get("eq", False)),
        ]

    # Prefer projected A/B hash; recompute only if missing
    proj_hash_ab = projected_ctx.get("projector_hash", "")
    if not proj_hash_ab:
        try:
            proj_hash_ab = projector_provenance_hash(
                cfg=(projected_ctx.get("cfg", {}) or {}),
                lane_mask_k3=projected_ctx.get("lane_mask_k3", []),
                district_id=fresh_district_info["district_id"],
            )
        except TypeError:
            try:
                proj_hash_ab = projector_provenance_hash(
                    cfg=(projected_ctx.get("cfg", {}) or {}),
                    lane_mask=projected_ctx.get("lane_mask_k3", []),
                    district_id=fresh_district_info["district_id"],
                )
            except Exception:
                try:
                    file_hash_ab, rt_hash_ab = projector_hashes_from_context(
                        projected_ctx.get("cfg", {}) or {}, boundaries, cache
                    )
                    proj_hash_ab = file_hash_ab or rt_hash_ab or ""
                except Exception:
                    proj_hash_ab = ""

    cert_payload.setdefault("policy", {})

# === District + A/B embed + integrity + write + gallery + bundle =============

# ---------- A) District info prelude (authoritative, no UI drift) ------------
_di = st.session_state.get("_district_info", {}) or {}

def _lane_mask_from_d3(d3_mat):
    if not d3_mat or not d3_mat[0]:
        return []
    cols = len(d3_mat[0])
    return [1 if any(row[j] & 1 for row in d3_mat) else 0 for j in range(cols)]

try:
    _d3 = (boundaries.blocks.__root__.get("3") or [])
except Exception:
    _d3 = []
_d3_rows = len(_d3)
_d3_cols = (len(_d3[0]) if _d3 and _d3[0] else 0)
_lane_mask_now = _di.get("lane_mask_k3_now")
if _lane_mask_now is None:
    _lane_mask_now = _lane_mask_from_d3(_d3)

# authoritative boundaries hash (session → inputs → cert)
_boundaries_hash = (
    _di.get("boundaries_hash")
    or inputs_block.get("boundaries_hash", "")
    or cert_payload.get("boundaries_hash", "")
)
if _boundaries_hash:
    inputs_block["boundaries_hash"] = _boundaries_hash  # keep inputs in sync

# resolve district id (session → cert → DISTRICT_MAP (if defined) → UI → UNKNOWN)
try:
    _district_from_map = (DISTRICT_MAP.get(_boundaries_hash) if "DISTRICT_MAP" in globals() else None)
except Exception:
    _district_from_map = None

_district_id = (
    _di.get("district_id")
    or cert_payload.get("district_id")
    or _district_from_map
    or st.session_state.get("district_id")
    or "UNKNOWN"
)

fresh_district_info = {
    "district_id":        _district_id,
    "boundaries_hash":    _boundaries_hash,
    "district_signature": _di.get("district_signature", ""),
    "lane_mask_k3_now":   _lane_mask_now,
    "d3_rows":            _di.get("d3_rows", _d3_rows),
    "d3_cols":            _di.get("d3_cols", _d3_cols),
}

# ---------- B) Embed A/B snapshots (safe, no stale drift) --------------------
# make sure ab_ctx is available
ab_ctx = st.session_state.get("ab_compare", {}) or {}
# freshness check: inputs_sig must match current inputs; else drop A/B
_current_sig = [
    inputs_block.get("boundaries_hash", ""),
    inputs_block.get("C_hash", ""),
    inputs_block.get("H_hash", ""),
    inputs_block.get("U_hash", ""),
    inputs_block.get("shapes_hash", ""),
]
if ab_ctx.get("inputs_sig") != _current_sig:
    ab_ctx = {}

strict_ctx    = ab_ctx.get("strict", {}) if ab_ctx else {}
projected_ctx = ab_ctx.get("projected", {}) if ab_ctx else {}

# pass-vector helper (local scope)
def _pass_vec_from(out_dict: dict) -> list[int]:
    return [
        int(out_dict.get("2", {}).get("eq", False)),
        int(out_dict.get("3", {}).get("eq", False)),
    ]

# provenance hash shim
def _prov_hash(cfg, lane_mask, district_id):
    try:
        return projector_provenance_hash(cfg=cfg, lane_mask_k3=lane_mask, district_id=district_id)
    except TypeError:
        pass
    try:
        return projector_provenance_hash(cfg=cfg, lane_mask=lane_mask, district_id=district_id)
    except TypeError:
        pass
    try:
        return projector_provenance_hash(
            cfg=cfg, boundaries=boundaries, district_id=district_id,
            diagnostics_block={"lane_mask_k3": lane_mask},
        )
    except TypeError:
        pass
    try:
        return projector_provenance_hash(cfg=cfg)
    except Exception:
        return ""

if ab_ctx:
    # 1) projector hash for projected leg (prefer A/B; else provenance)
    proj_hash_ab = projected_ctx.get("projector_hash", "")
    if not proj_hash_ab:
        proj_hash_ab = _prov_hash(
            projected_ctx.get("cfg", {}) or {},
            projected_ctx.get("lane_mask_k3", []),
            fresh_district_info["district_id"],
        )

    # 2) inputs.boundaries for each snapshot (authoritative from fresh_district_info)
    strict_inputs_boundaries = {
        "filename":        inputs_block.get("boundaries_filename", ""),
        "hash":            fresh_district_info["boundaries_hash"],
        "district_id":     fresh_district_info["district_id"],
        "district_sig":    fresh_district_info["district_signature"],
        "lane_mask_k3":    strict_ctx.get("lane_mask_k3", []),
        "d3_rows":         fresh_district_info["d3_rows"],
        "d3_cols":         fresh_district_info["d3_cols"],
    }
    projected_inputs_boundaries = {
        "filename":        inputs_block.get("boundaries_filename", ""),
        "hash":            fresh_district_info["boundaries_hash"],
        "district_id":     fresh_district_info["district_id"],
        "district_sig":    fresh_district_info["district_signature"],
        "lane_mask_k3":    projected_ctx.get("lane_mask_k3", []),
        "d3_rows":         fresh_district_info["d3_rows"],
        "d3_cols":         fresh_district_info["d3_cols"],
    }

    # 3) projector FILE metadata (prefer Run Overlap's proj_meta; fallback to A/B)
    _proj_meta = st.session_state.get("proj_meta", {}) or {}
    proj_filename = (
        _proj_meta.get("projector_filename")
        or projected_ctx.get("projector_filename", "")
    )
    proj_file_hash = (
        _proj_meta.get("projector_hash")
        or projected_ctx.get("projector_file_hash", "")
    )
    proj_consistent = _proj_meta.get("projector_consistent_with_d", None)

    # 4) assemble snapshots
    cert_payload.setdefault("policy", {})
    cert_payload["policy"]["strict_snapshot"] = {
        "policy_tag": strict_ctx.get("label", policy_label_from_cfg(cfg_strict())),
        "ker_guard":  strict_ctx.get("ker_guard", "enforced"),
        "inputs": {
            "boundaries": strict_inputs_boundaries,
            "U_filename": inputs_block.get("U_filename", ""),
        },
        "lane_mask_k3":      strict_ctx.get("lane_mask_k3", []),
        "lane_vec_H2d3":     strict_ctx.get("lane_vec_H2d3", []),
        "lane_vec_C3plusI3": strict_ctx.get("lane_vec_C3plusI3", []),
        "pass_vec": _pass_vec_from(strict_ctx.get("out", {})),
        "out":      strict_ctx.get("out", {}),
    }

    cert_payload["policy"]["projected_snapshot"] = {
        "policy_tag":     projected_ctx.get("label", policy_label_from_cfg(cfg_projected_base())),
        "ker_guard":      projected_ctx.get("ker_guard", "off"),
        "projector_hash": proj_hash_ab,
        "inputs": {
            "boundaries": projected_inputs_boundaries,
            "U_filename": inputs_block.get("U_filename", ""),
            # projector(file) metadata (meaningful only in file mode)
            "projector_filename": proj_filename,
            "projector_file_hash": proj_file_hash,
            "projector_consistent_with_d": proj_consistent,
        },
        "lane_mask_k3":      projected_ctx.get("lane_mask_k3", []),
        "lane_vec_H2d3":     projected_ctx.get("lane_vec_H2d3", []),
        "lane_vec_C3plusI3": projected_ctx.get("lane_vec_C3plusI3", []),
        "pass_vec": _pass_vec_from(projected_ctx.get("out", {})),
        "out":      projected_ctx.get("out", {}),
    }

    cert_payload["ab_pair_tag"] = ab_ctx.get("pair_tag", "")

    # 5) mirror projected hash to top-level if active run is strict
    proj_hash_from_ab = cert_payload["policy"]["projected_snapshot"].get("projector_hash", "")
    if proj_hash_from_ab and not cert_payload["policy"].get("projector_hash"):
        cert_payload["policy"]["projector_hash"] = proj_hash_from_ab
        cert_payload.setdefault("artifact_hashes", {})
        cert_payload["artifact_hashes"]["projector_hash"] = proj_hash_from_ab


# ---------- C) Integrity + top-level artifact hashes + write ------------------
district_id_for_cert = (
    cert_payload.get("district_id")
    or fresh_district_info["district_id"]
    or "UNKNOWN"
)
boundaries_hash_for_cert = (
    cert_payload.get("boundaries_hash")
    or fresh_district_info["boundaries_hash"]
    or inputs_block.get("boundaries_hash", "")
)

cert_payload["district_id"]     = district_id_for_cert
cert_payload["boundaries_hash"] = boundaries_hash_for_cert

cert_payload.setdefault("artifact_hashes", {
    "boundaries_hash": inputs_block.get("boundaries_hash", ""),
    "C_hash":          inputs_block.get("C_hash", ""),
    "H_hash":          inputs_block.get("H_hash", ""),
    "U_hash":          inputs_block.get("U_hash", ""),
})
cert_payload["artifact_hashes"]["projector_hash"] = cert_payload.get("policy", {}).get("projector_hash", "")

# --- Authoritative district -> force into cert before write -------------------
_di = st.session_state.get("_district_info", {}) or {}
if _di:
    cert_payload["district_id"]     = _di.get("district_id", cert_payload.get("district_id", "UNKNOWN"))
    cert_payload["boundaries_hash"] = _di.get("boundaries_hash", cert_payload.get("boundaries_hash", ""))

    # keep inputs in sync to avoid parsed/bytes double-hash divergence
    inputs_block["boundaries_hash"] = cert_payload["boundaries_hash"]
    cert_payload["inputs"] = inputs_block


# filenames into inputs + sync inputs.boundaries_hash to authoritative one
if "boundaries_filename" not in inputs_block:
    inputs_block["boundaries_filename"] = inputs_block.get("boundaries_name", "boundaries.json")
if "U_filename" not in inputs_block:
    inputs_block["U_filename"] = inputs_block.get("U_name", "U.json")
if cert_payload.get("boundaries_hash"):
    inputs_block["boundaries_hash"] = cert_payload["boundaries_hash"]
cert_payload["inputs"] = inputs_block

cert_payload.setdefault("integrity", {})
cert_payload["integrity"]["content_hash"] = hashes.content_hash_of(cert_payload)
# ========================== CERT ENRICHMENTS BLOCK ===========================

# ---- helpers ----------------------------------------------------------------
def _residual_tag_from_checks(out_dict: dict) -> str:
    """Return 'ker'|'lanes'|'mixed'|'none' based on your checks structure."""
    # heuristic: if k=3 failed but k=2 passed → lanes; if kernel fails → ker; both → mixed; else none
    k2 = bool(out_dict.get("2", {}).get("eq", False))
    k3 = bool(out_dict.get("3", {}).get("eq", False))
    # If you have explicit residual flags, prefer them:
    res = out_dict.get("residual_tag")
    if isinstance(res, str) and res:
        return res
    if k3 and k2:
        return "none"
    if (not k3) and k2:
        return "lanes"
    if (not k2) and k3:
        return "ker"
    return "mixed"

def _lane_mask_from_boundaries(bnds) -> list[int]:
    d3 = (bnds.blocks.__root__.get("3") or [])
    if not d3 or not d3[0]: 
        return []
    cols = len(d3[0])
    return [1 if any(row[j] & 1 for row in d3) else 0 for j in range(cols)]

def _projector_diag_from_file(cfg: dict) -> tuple[list[int], str]:
    """Return (diag_vec, projector_filename) for k=3 if source is file; else ([], '')."""
    src3 = (cfg or {}).get("source", {}).get("3", "")
    if src3 != "file":
        return [], ""
    pfiles = (cfg or {}).get("projector_files", {})
    pfile3 = pfiles.get("3", "")
    if not pfile3:
        return [], ""
    # your projector loader likely already cached the matrix in ab run; fallback: read from disk if you have util
    try:
        P3 = projector.load_projector_matrix(pfile3)  # implement/adjust to your codebase
    except Exception:
        # legacy: try using preload cache if you have it in st.session_state
        P3 = None
    diag = []
    if P3 and isinstance(P3, list) and P3 and isinstance(P3[0], list):
        n = min(len(P3), len(P3[0]))
        diag = [int(P3[i][i] & 1) for i in range(n)]
    return diag, pfile3

def _gf2_idempotent(P) -> bool:
    """Check P@P == P over GF(2). P as list-of-lists of ints."""
    try:
        n = len(P)
        m = len(P[0]) if n else 0
        if n != m:  # must be square
            return False
        # naive GF(2) matmul
        PP = [[0]*n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                if P[i][k] & 1:
                    rowk = P[k]
                    # xor-add rowk into PP[i]
                    for j in range(n):
                        PP[i][j] ^= (rowk[j] & 1)
        # compare to P
        for i in range(n):
            for j in range(n):
                if (PP[i][j] & 1) != (P[i][j] & 1):
                    return False
        return True
    except Exception:
        return False

def _try_get_projector_matrix_from_cache(cfg: dict):
    """Best-effort to retrieve matrix for file mode from any cache your app has."""
    try:
        cache = st.session_state.get("_projector_cache") or {}
        key = str(cfg.get("projector_files", {}).get("3", ""))
        return cache.get(key)
    except Exception:
        return None

# ---- canonical district & filenames -----------------------------------------
_di = st.session_state.get("_district_info", {}) or {}
district_id_canon     = _di.get("district_id", cert_payload.get("district_id", "UNKNOWN"))
lane_mask_k3_canon    = _di.get("lane_mask_k3_now", diagnostics_block.get("lane_mask_k3", []))
d3_rows_canon         = _di.get("d3_rows", 0)
d3_cols_canon         = _di.get("d3_cols", 0)

# If diagnostics is missing lane mask, compute it now
if not lane_mask_k3_canon:
    lane_mask_k3_canon = _lane_mask_from_boundaries(boundaries)

# Filenames (standardize U to shapes.json everywhere)
inputs_block.setdefault("boundaries_filename", st.session_state.get("fname_boundaries", "boundaries.json"))
inputs_block.setdefault("C_filename",          st.session_state.get("fname_cmap", "cmap.json"))
inputs_block.setdefault("H_filename",          st.session_state.get("fname_h", "H.json"))
inputs_block["U_filename"] = "shapes.json"  # <- unified

# Mirror filenames under top-level inputs.filenames
inputs_block.setdefault("filenames", {})
inputs_block["filenames"].update({
    "boundaries": inputs_block.get("boundaries_filename", "boundaries.json"),
    "C":          inputs_block.get("C_filename", "cmap.json"),
    "H":          inputs_block.get("H_filename", "H.json"),
    "U":          inputs_block.get("U_filename", "shapes.json"),
})

cert_payload["inputs"] = inputs_block  # write-back

# ---- district consistency: identity/policy/snapshots must agree --------------
identity_block["district_id"] = district_id_canon  # force identity to canonical
cert_payload["district_id"]   = district_id_canon  # top-level too

# ---- build snapshots from A/B ctx (fresh only if inputs_sig matches) ---------
ab_ctx = st.session_state.get("ab_compare", {}) or {}
curr_sig = [
    inputs_block.get("boundaries_hash", ""),
    inputs_block.get("C_hash", ""),
    inputs_block.get("H_hash", ""),
    inputs_block.get("U_hash", ""),
    inputs_block.get("shapes_hash", ""),
]
if ab_ctx.get("inputs_sig") != curr_sig:
    ab_ctx = {}
strict_ctx    = ab_ctx.get("strict", {}) if ab_ctx else {}
projected_ctx = ab_ctx.get("projected", {}) if ab_ctx else {}

# per-leg pass vectors + residual tags
strict_pass_vec    = [int(strict_ctx.get("out", {}).get("2", {}).get("eq", False)),
                      int(strict_ctx.get("out", {}).get("3", {}).get("eq", False))]
projected_pass_vec = [int(projected_ctx.get("out", {}).get("2", {}).get("eq", False)),
                      int(projected_ctx.get("out", {}).get("3", {}).get("eq", False))]

strict_residual_tag    = _residual_tag_from_checks(strict_ctx.get("out", {}))
projected_residual_tag = _residual_tag_from_checks(projected_ctx.get("out", {}))

# snapshot inputs (filenames + boundaries sub-block)
strict_inputs_boundaries = {
    "filename":        inputs_block["boundaries_filename"],
    "hash":            inputs_block.get("boundaries_hash", ""),
    "district_id":     district_id_canon,
    "lane_mask_k3":    lane_mask_k3_canon,      # populate snapshot mask
    "d3_rows":         d3_rows_canon,
    "d3_cols":         d3_cols_canon,
}
projected_inputs_boundaries = dict(strict_inputs_boundaries)  # same bind

# projector file-mode metadata & validation (only when projected file mode)
proj_file_diag, proj_file_name = [], ""
proj_file_hash = ""
proj_consistent = None  # None when not file-mode
if cfg_active.get("enabled_layers"):  # projected policy active at run time
    src3 = cfg_active.get("source", {}).get("3", "")
    if src3 == "file":
        proj_file_diag, proj_file_name = _projector_diag_from_file(cfg_active)
        # Try to fetch raw matrix for validation
        P3 = _try_get_projector_matrix_from_cache(cfg_active)
        # If cache miss, try loader again (if available)
        if P3 is None and proj_file_name:
            try:
                P3 = projector.load_projector_matrix(proj_file_name)
            except Exception:
                P3 = None
        # shape/idempotent check
        shape_ok = False
        if P3 and isinstance(P3, list) and P3 and isinstance(P3[0], list):
            n = len(P3)
            shape_ok = (n == d3_cols_canon == d3_rows_canon) and _gf2_idempotent(P3)
        # consistency with d3 lanes: diagonal equals auto mask
        auto_mask = lane_mask_k3_canon[:len(proj_file_diag)] if proj_file_diag else lane_mask_k3_canon
        consistent_ok = (proj_file_diag == auto_mask[:len(proj_file_diag)]) if proj_file_diag else True
        proj_consistent = bool(shape_ok and consistent_ok)
        # projector_hash (you already compute top-level); leave as-is.

# ---- build the snapshot objects ---------------------------------------------
cert_payload.setdefault("policy", {})

cert_payload["policy"]["strict_snapshot"] = {
    "policy_tag": strict_ctx.get("label", policy_label_from_cfg(cfg_strict())),
    "ker_guard":  "enforced",
    "inputs": {
        "filenames": inputs_block["filenames"],
        "boundaries": strict_inputs_boundaries,
        "U_filename": inputs_block["U_filename"],
        "C_filename": inputs_block["C_filename"],
        "H_filename": inputs_block["H_filename"],
    },
    "lane_mask_k3":      lane_mask_k3_canon,
    "lane_vec_H2d3":     strict_ctx.get("lane_vec_H2d3", diagnostics_block.get("lane_vec_H2d3", [])),
    "lane_vec_C3plusI3": strict_ctx.get("lane_vec_C3plusI3", diagnostics_block.get("lane_vec_C3plusI3", [])),
    "pass_vec":          strict_pass_vec,
    "residual_tag":      strict_residual_tag,
    "out":               strict_ctx.get("out", {}),
}

proj_hash_ab = projected_ctx.get("projector_hash", cert_payload.get("policy", {}).get("projector_hash", ""))
cert_payload["policy"]["projected_snapshot"] = {
    "policy_tag":     projected_ctx.get("label", policy_label_from_cfg(cfg_projected_base())),
    "ker_guard":      "off",
    "projector_hash": proj_hash_ab,
    "inputs": {
        "filenames": inputs_block["filenames"],
        "boundaries": projected_inputs_boundaries,
        "U_filename": inputs_block["U_filename"],
        "C_filename": inputs_block["C_filename"],
        "H_filename": inputs_block["H_filename"],
        **({"projector_filename": proj_file_name} if proj_file_name else {}),
    },
    "lane_mask_k3":      lane_mask_k3_canon,
    "lane_vec_H2d3":     projected_ctx.get("lane_vec_H2d3", diagnostics_block.get("lane_vec_H2d3", [])),
    "lane_vec_C3plusI3": projected_ctx.get("lane_vec_C3plusI3", diagnostics_block.get("lane_vec_C3plusI3", [])),
    "pass_vec":          projected_pass_vec,
    "residual_tag":      projected_residual_tag,
    "out":               projected_ctx.get("out", {}),
    **({"projector_consistent_with_d": proj_consistent} if proj_consistent is not None else {}),
}

# Warn in UI if projector file-mode is inconsistent (don’t block, but you can)
if proj_consistent is False:
    st.warning("Projected(file) projector is not consistent with current d3 (shape/idempotence/lane diag check failed).")

# ---- district assert across identity & snapshots -----------------------------
snap_districts = [
    district_id_canon,
    cert_payload["policy"]["strict_snapshot"]["inputs"]["boundaries"]["district_id"],
    cert_payload["policy"]["projected_snapshot"]["inputs"]["boundaries"]["district_id"],
]
if len(set(snap_districts)) != 1:
    st.error(f"District mismatch across cert/snapshots: {snap_districts}")
    st.stop()

# ---- promotion logic ---------------------------------------------------------
# Reads your checks (top-level) and policy mode to decide eligibility/target
grid_ok  = bool(checks_block.get("grid_ok", 1))    # default permissive if not present
fence_ok = bool(checks_block.get("fence_ok", 1))
k3_ok    = bool(checks_block.get("3", {}).get("eq", False))
k2_ok    = bool(checks_block.get("2", {}).get("eq", False))
is_strict_mode    = not cfg_active.get("enabled_layers")
is_projected_mode = cfg_active.get("enabled_layers")

eligible = False
target   = None
if is_strict_mode and all([grid_ok, fence_ok, k3_ok, k2_ok]):
    eligible = True
    target   = "strict_anchor"
elif is_projected_mode and all([grid_ok, fence_ok, k3_ok]) and projected_residual_tag == "none":
    eligible = True
    target   = "projected_exemplar"

cert_payload["promotion"] = {
    "eligible_for_promotion": eligible,
    "promotion_target": target,
    "notes": cert_payload.get("promotion", {}).get("notes", ""),
}

# ---- Option B: embed full A/B for export (single cert with ab_compare) -------
# If you prefer two files, set EXPORT_AB_AS_TWO_FILES = True and branch.
EXPORT_AB_AS_TWO_FILES = False

if ab_ctx and not EXPORT_AB_AS_TWO_FILES:
    cert_payload["ab_compare"] = {
        "strict": {
            "checks":      strict_ctx.get("out", {}),
            "diagnostics": {
                "lane_mask_k3":      strict_ctx.get("lane_mask_k3", lane_mask_k3_canon),
                "lane_vec_H2d3":     strict_ctx.get("lane_vec_H2d3", diagnostics_block.get("lane_vec_H2d3", [])),
                "lane_vec_C3plusI3": strict_ctx.get("lane_vec_C3plusI3", diagnostics_block.get("lane_vec_C3plusI3", [])),
            },
            "ker_guard":    "enforced",
            "residual_tag": strict_residual_tag,
        },
        "projected": {
            "checks":      projected_ctx.get("out", {}),
            "diagnostics": {
                "lane_mask_k3":      projected_ctx.get("lane_mask_k3", lane_mask_k3_canon),
                "lane_vec_H2d3":     projected_ctx.get("lane_vec_H2d3", diagnostics_block.get("lane_vec_H2d3", [])),
                "lane_vec_C3plusI3": projected_ctx.get("lane_vec_C3plusI3", diagnostics_block.get("lane_vec_C3plusI3", [])),
            },
            "ker_guard":    "off",
            "residual_tag": projected_residual_tag,
            "projector_hash": proj_hash_ab,
        },
        "pair_tag": ab_ctx.get("pair_tag", ""),
    }

# (If EXPORT_AB_AS_TWO_FILES=True, after you write the strict cert, build a projected cert by
# swapping policy_block/outputs and call export_mod.write_cert_json again with the new payload.)
# ======================== END CERT ENRICHMENTS BLOCK ==========================


# Single write
cert_path, full_hash = export_mod.write_cert_json(cert_payload)
st.success(f"Cert written: `{cert_path}`")

# ---------- D) Gallery de-duplication row -----------------------------------
try:
    try:
        key = gallery_key_from(
            cert_payload=cert_payload,
            cmap=cmap,
            diagnostics_block=diagnostics_block,
            policy_label=policy_label,
        )
    except Exception:
        key = (
            district_id_for_cert,
            inputs_block.get("boundaries_hash", ""),
            inputs_block.get("U_hash", ""),
            inputs_block.get("suppC_hash", inputs_block.get("C_hash", "")),
            tuple(cert_payload.get("diagnostics", {}).get("lane_vec_H2d3", [])),
            policy_label,
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

    if district_id_for_cert == "UNKNOWN":
        st.warning("registry insert skipped: district_id is UNKNOWN (bind via boundaries hash mapping).")
    else:
        res = export_mod.write_gallery_row(row, key, path="gallery.csv")
        st.toast("gallery: added exemplar row" if res == "written" else "gallery: duplicate skipped")
except Exception as e:
    st.warning(f"gallery dedupe failed: {e}")

# ---------- E) Download bundle (includes cert.json + policy.json + maps) -----
try:
    _di2 = st.session_state.get("_district_info", {}) or {}
    district_id_for_cert = (
        cert_payload.get("district_id")
        or _di2.get("district_id")
        or "UNKNOWN"
    )
    boundaries_hash_for_cert = (
        cert_payload.get("boundaries_hash")
        or inputs_block.get("boundaries_hash", "")
        or _di2.get("boundaries_hash", "")
    )

    # Prefer the cert's district for naming
    district_id = district_id_for_cert

    _policy_block_for_bundle = dict(policy_block)  # shallow copy

    _ab_ctx = st.session_state.get("ab_compare", {}) or {}
    _projected_ctx = _ab_ctx.get("projected", {}) or {}

    if "policy" in cert_payload and "projected_snapshot" in cert_payload["policy"]:
        _policy_block_for_bundle["ab_policies"] = {
            "strict":    cert_payload["policy"]["strict_snapshot"]["policy_tag"],
            "projected": cert_payload["policy"]["projected_snapshot"]["policy_tag"],
        }
        _policy_block_for_bundle["ab_projector_hash"] = (
            _projected_ctx.get("projector_hash")
            or cert_payload["policy"]["projected_snapshot"].get("projector_hash", "")
        )

    proj_hash_bundle = (
        _projected_ctx.get("projector_hash")
        or cert_payload.get("policy", {}).get("projected_snapshot", {}).get("projector_hash", "")
        or _policy_block_for_bundle.get("projector_hash", "")
    )
    if proj_hash_bundle and not _policy_block_for_bundle.get("projector_hash"):
        _policy_block_for_bundle["projector_hash"] = proj_hash_bundle

    _policy_block_for_bundle["district_id"]         = district_id_for_cert
    _policy_block_for_bundle["boundaries_hash"]     = boundaries_hash_for_cert
    _policy_block_for_bundle["boundaries_filename"] = inputs_block.get("boundaries_filename", "")
    _policy_block_for_bundle["U_filename"]          = inputs_block.get("U_filename", "")

    tag = policy_label.replace(" ", "_")
    if "policy" in cert_payload and "strict_snapshot" in cert_payload["policy"]:
        tag = f"{tag}__withAB"

    bundle_name = f"overlap_bundle__{district_id}__{tag}__{full_hash[:12]}.zip"
    zip_path = export_mod.build_overlap_bundle(
        boundaries=boundaries,
        cmap=cmap,
        H=H_local,
        shapes=(shapes.dict() if hasattr(shapes, "dict") else (shapes or {})),
        policy_block=_policy_block_for_bundle,
        cert_path=cert_path,
        out_zip=bundle_name,
    )
    with open(zip_path, "rb") as f:
        st.download_button(
            "⬇️ Download bundle (.zip)",
            data=f,
            file_name=bundle_name,
            mime="application/zip",
            key="dl_overlap_bundle",
        )
except Exception as e:
    st.error(f"Could not build download bundle: {e}")





# ──────────────────────────────────────────────────────────────────────────────
# 8) promotion (optional)
# ──────────────────────────────────────────────────────────────────────────────
pass_vec  = [int(out.get("2", {}).get("eq", False)),
             int(out.get("3", {}).get("eq", False))]
all_green = all(v == 1 for v in pass_vec)

if all_green:
    st.success("Green — eligible for promotion.")
    flip_to_file = st.checkbox("After promotion, switch to FILE-backed projector",
                               value=True, key="flip_to_file_k3")
    keep_auto = st.checkbox("…or keep AUTO (don’t lock now)",
                            value=False, key="keep_auto_k3")

    if st.button("Promote & Freeze Projector", key="promote_k3"):
        d3_now = boundaries.blocks.__root__.get("3")
        if not d3_now:
            st.error("No d3; cannot freeze projector.")
        else:
            # freeze exact Π3 used now
            P_used = projector.projector_columns_from_dkp1(d3_now)
            pj_path = cfg_file.get("projector_files", {}).get("3", "projector_D3.json")
            pj_hash = projector.save_projector(pj_path, P_used)
            st.info(f"Projector frozen → {pj_path} (hash={pj_hash[:12]}…)")

            # flip config or keep auto
            if flip_to_file and not keep_auto:
                cfg_file.setdefault("source", {})["3"] = "file"
                cfg_file.setdefault("projector_files", {})["3"] = pj_path
            else:
                cfg_file.setdefault("source", {})["3"] = "auto"
                if "projector_files" in cfg_file and "3" in cfg_file["projector_files"]:
                    del cfg_file["projector_files"]["3"]
            with open("projection_config.json", "w") as _f:
                _json.dump(cfg_file, _f, indent=2)

            import time as _time
            try:
                export_mod.write_registry_row(
                    fix_id=f"overlap-{int(_time.time())}",
                    pass_vector=pass_vec,
                    policy=policy_label,
                    hash_d=hashes.hash_d(boundaries),
                    hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                    hash_suppC=hashes.hash_suppC(cmap),
                    hash_suppH=hashes.hash_suppH(H_local),
                    notes=f"proj_hash={pj_hash}"
                )
                st.success("Registry updated with projector hash.")
            except Exception as e:
                st.error(f"Failed to write registry row: {e}")

     
       

# --- A/B compare (strict vs projected) ------------------------------------
st.markdown("### A/B: strict vs projected")

# Show current A/B snapshot if it exists (no warnings if absent)
_ab_ctx_existing = st.session_state.get("ab_compare")
if _ab_ctx_existing:
    s_ok = bool(_ab_ctx_existing.get("strict", {}).get("out", {}).get("3", {}).get("eq", False))
    p_ok = bool(_ab_ctx_existing.get("projected", {}).get("out", {}).get("3", {}).get("eq", False))
    s_badge = "✅" if s_ok else "❌"
    p_badge = "✅" if p_ok else "❌"
    st.caption(f"A/B: strict={s_badge} · projected={p_badge}")
    st.caption(f"pair: {_ab_ctx_existing.get('pair_tag','')}")
else:
    st.caption("A/B: (no snapshot yet)")

# Build projected cfg from disk choices (so A/B mirrors what's on disk)
_cfg_file_for_ab = projector.load_projection_config("projection_config.json")
_cfg_proj_for_ab = cfg_projected_base()
if _cfg_file_for_ab.get("source", {}).get("3") in ("file", "auto"):
    _cfg_proj_for_ab["source"]["3"] = _cfg_file_for_ab["source"]["3"]
if "projector_files" in _cfg_file_for_ab and "3" in _cfg_file_for_ab["projector_files"]:
    _cfg_proj_for_ab["projector_files"]["3"] = _cfg_file_for_ab["projector_files"]["3"]

# --- Projector cache-bust key (boundaries/source/file changes) -------------
_di = st.session_state.get("_district_info", {}) or {}
_bound_hash = _di.get("boundaries_hash", "")

# Choose the cfg we’re about to load cache for
_cfg_for_cache = cfg_active if 'cfg_active' in locals() else _cfg_proj_for_ab
_src3  = _cfg_for_cache.get("source", {}).get("3", "")
_file3 = _cfg_for_cache.get("projector_files", {}).get("3", "") if _src3 == "file" else ""

_cache_key = f"{_bound_hash}|src3={_src3}|file3={_file3}"

# Use different session keys for overlap vs A/B to avoid collisions
_cache_key_name = "_projector_cache_key" if 'cfg_active' in locals() else "_projector_cache_key_ab"
_cache_blob_name = "_projector_cache" if 'cfg_active' in locals() else "_projector_cache_ab"

if st.session_state.get(_cache_key_name) != _cache_key:
    st.session_state.pop(_cache_blob_name, None)
    st.session_state[_cache_key_name] = _cache_key

# finally preload (will rebuild if we just busted it)
_cache_blob = st.session_state.get(_cache_blob_name) or projector.preload_projectors_from_files(_cfg_for_cache)
st.session_state[_cache_blob_name] = _cache_blob


# Use whatever caching your helper already implements (no cache_key kwarg)
_cache_for_ab = projector.preload_projectors_from_files(_cfg_proj_for_ab)

# Parse H once for both runs
H_obj = io.parse_cmap(d_H) if d_H else io.parse_cmap({"blocks": {}})

if st.button("Run A/B compare (strict vs projected)", key="run_ab_overlap"):
    # --- A) strict
    out_strict   = overlap_gate.overlap_check(boundaries, cmap, H_obj)
    label_strict = policy_label_from_cfg(cfg_strict())

    # --- B) projected (columns@k=3, auto|file per config)
    out_proj = overlap_gate.overlap_check(
        boundaries, cmap, H_obj,
        projection_config=_cfg_proj_for_ab,
        projector_cache=_cache_for_ab,
    )
    label_proj = policy_label_from_cfg(_cfg_proj_for_ab)

        # --- Lane diagnostics (compute fresh from current buffers) ----------------
    from otcore.linalg_gf2 import mul, add, eye

    def _bottom_row(M): 
        return M[-1] if (M and len(M)) else []

    def _lane_mask_from_d3(d3_mat):
        if not d3_mat or not d3_mat[0]:
            return []
        cols = len(d3_mat[0])
        return [1 if any(row[j] & 1 for row in d3_mat) else 0 for j in range(cols)]

    def _mask(vec, idx):
        return [vec[j] for j in idx] if (vec and idx) else []

    # Shared raw blocks
    d3  = boundaries.blocks.__root__.get("3") or []
    H2  = (H_obj.blocks.__root__.get("2") or [])
    C3  = (cmap.blocks.__root__.get("3") or [])

    # Build lane mask from D3 only (policy-agnostic)
    lane_mask = _lane_mask_from_d3(d3)
    lane_idx  = [j for j, m in enumerate(lane_mask) if m]

    # Helper to compute lanes from H@d3 and (C3+I3), *fresh each time*
    def compute_lane_vectors(H2_block, d3_block, C3_block, lane_indices):
        H2d3   = mul(H2_block, d3_block) if (H2_block and d3_block) else []
        C3pI3  = add(C3_block, eye(len(C3_block))) if C3_block else []
        v_H2d3 = _mask(_bottom_row(H2d3), lane_indices)
        v_C3I  = _mask(_bottom_row(C3pI3), lane_indices)
        return v_H2d3, v_C3I

    # Fresh lanes for strict snapshot (from current buffers)
    strict_lane_vec_H2d3, strict_lane_vec_C3pI3 = compute_lane_vectors(H2, d3, C3, lane_idx)
    # Fresh lanes for projected snapshot (same H,d3,C3; recomputed to avoid cross-use)
    proj_lane_vec_H2d3,   proj_lane_vec_C3pI3   = compute_lane_vectors(H2, d3, C3, lane_idx)

    # --- Projector provenance hash for the PROJECTED leg (AUTO or FILE) -------
    def _provenance_hash(cfg, lane_mask_k3, district_id):
        try:
            return projector_provenance_hash(cfg=cfg, lane_mask_k3=lane_mask_k3, district_id=district_id)
        except TypeError:
            pass
        try:
            return projector_provenance_hash(cfg=cfg, lane_mask=lane_mask_k3, district_id=district_id)
        except TypeError:
            pass
        try:
            return projector_provenance_hash(
                cfg=cfg, boundaries=boundaries, district_id=district_id,
                diagnostics_block={"lane_mask_k3": lane_mask_k3},
            )
        except TypeError:
            pass
        try:
            return projector_provenance_hash(cfg=cfg)
        except Exception:
            return ""

    district_id   = st.session_state.get("district_id", "D3")
    proj_hash_prov = _provenance_hash(_cfg_proj_for_ab, lane_mask, district_id)
    st.caption(f"projected Π provenance hash: {proj_hash_prov[:12]}…")

    # Build an inputs signature for freshness checks
    inputs_sig = [
        inputs_block.get("boundaries_hash", ""),
        inputs_block.get("C_hash", ""),
        inputs_block.get("H_hash", ""),
        inputs_block.get("U_hash", ""),
        inputs_block.get("shapes_hash", ""),
    ]

    # --- Persist compact A/B context (each snapshot uses its OWN fresh lanes) --
    pair_tag = f"{label_strict}__VS__{label_proj}"
    st.session_state["ab_compare"] = {
        "pair_tag": pair_tag,
        "inputs_sig": inputs_sig,          # freshness guard
        "lane_mask_k3": lane_mask,         # shared mask
        "strict": {
            "label": label_strict,
            "cfg":   cfg_strict(),
            "out":   out_strict,
            "ker_guard": "enforced",
            "lane_vec_H2d3":     strict_lane_vec_H2d3,   # <- fresh from H@d3
            "lane_vec_C3plusI3": strict_lane_vec_C3pI3,  # <- fresh from C3+I3
            "projector_hash": "",  # none in strict
            "pass_vec": [
                int(out_strict.get("2", {}).get("eq", False)),
                int(out_strict.get("3", {}).get("eq", False)),
            ],
        },
        "projected": {
            "label": label_proj,
            "cfg":   _cfg_proj_for_ab,
            "out":   out_proj,
            "ker_guard": "off",
            "projector_hash": proj_hash_prov,
            "lane_vec_H2d3":     proj_lane_vec_H2d3,     # <- fresh from H@d3
            "lane_vec_C3plusI3": proj_lane_vec_C3pI3,    # <- fresh from C3+I3
            "pass_vec": [
                int(out_proj.get("2", {}).get("eq", False)),
                int(out_proj.get("3", {}).get("eq", False)),
            ],
        },
    }


    # --- Status badge ---
    s_ok = bool(out_strict.get("3", {}).get("eq", False))
    p_ok = bool(out_proj.get("3", {}).get("eq", False))
    st.success(f"A/B: strict={'GREEN' if s_ok else 'RED'} · projected={'GREEN' if p_ok else 'RED'}")




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
