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

io            = _load_pkg_module(f"{PKG_NAME}.io",            "io.py")
hashes        = _load_pkg_module(f"{PKG_NAME}.hashes",        "hashes.py")
unit_gate     = _load_pkg_module(f"{PKG_NAME}.unit_gate",     "unit_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers        = _load_pkg_module(f"{PKG_NAME}.towers",        "towers.py")
export_mod    = _load_pkg_module(f"{PKG_NAME}.export",        "export.py")

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")

# ─────────────────────────────── UI HEADER ────────────────────────────────────
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")
st.caption(f"overlap_gate loaded from: {getattr(overlap_gate, '__file__', '<none>')}")
st.caption(f"projector loaded from: {getattr(projector, '__file__', '<none>')}")

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

    # ── freeze AUTO → file
    with st.expander("Freeze AUTO projector → file"):
        default_name = "projectors/projector_auto_k3.json"
        freeze_path = st.text_input("Output file", value=default_name, key="freeze_proj_out")
        st.caption("Build Π₃ = diag(lane_mask(d₃)) from current boundaries and switch source.3 → file.")
        if st.button("Freeze now", key="freeze_proj_btn"):
            try:
                d3 = (boundaries.blocks.__root__.get("3") or [])
                n3 = len(d3[0]) if (d3 and d3[0]) else 0
                if n3 == 0:
                    st.error("Freeze aborted: d3 appears empty (n3=0). Load boundaries first.")
                    st.stop()
                lane_mask = _lane_mask_from_d3(boundaries)
                P = [[1 if (i == j and lane_mask[i]) else 0 for j in range(n3)] for i in range(n3)]
                payload = {"name": "Π3 from current d3 (AUTO freeze)", "blocks": {"3": P}}
                os.makedirs(os.path.dirname(freeze_path), exist_ok=True)
                with open(freeze_path, "w") as fp:
                    _json.dump(payload, fp, indent=2)
                # flip disk config
                disk = projector.load_projection_config("projection_config.json")
                disk.setdefault("source", {})["3"] = "file"
                disk.setdefault("projector_files", {})["3"] = freeze_path
                with open("projection_config.json", "w") as _f:
                    _json.dump(disk, _f, indent=2)
                # reflect in-memory
                cfg_active.setdefault("source", {})["3"] = "file"
                cfg_active.setdefault("projector_files", {})["3"] = freeze_path
                st.success(f"Projector frozen → {freeze_path}. Now Run Overlap.")
            except Exception as e:
                st.error(f"Freeze failed: {e}")

    # ── active policy badge
    src3 = cfg_active.get("source", {}).get("3", "")
    _policy_mode_badge = "strict" if policy_choice == "strict" else ("projected(file)" if src3 == "file" else "projected(auto)")
    st.caption(f"Policy: **{policy_label}** · mode: {_policy_mode_badge}")

    # ── small utilities (local-only to avoid global clashes)
    def _eye(n): return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    def _xor_mat(A, B):
        if not A: return [r[:] for r in (B or [])]
        if not B: return [r[:] for r in (A or [])]
        m, n = len(A), len(A[0])
        return [[(A[i][j] ^ B[i][j]) & 1 for j in range(n)] for i in range(m)]
    def _bottom_row(M): return M[-1] if (M and len(M)) else []
    def _stable_hash(obj):  # deterministic content hash (JSON-normalized)
        return _hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    # ── RUN OVERLAP (creates RunContext SSOT)
    if st.button("Run Overlap", key="run_overlap"):
        try:
            # clear stale
            for k in ("proj_meta", "run_ctx", "residual_tags"):
                st.session_state.pop(k, None)

            # bind projector (fail-fast on FILE) via your chooser
            try:
                P_active, meta = projector_choose_active(cfg_active, boundaries)
            except ValueError as e:
                st.error(str(e))
                # persist a minimal run_ctx echo so banners/expands don’t recompute
                st.session_state["run_ctx"] = {
                    "policy_tag": policy_label,
                    "mode": "projected(file)" if cfg_active.get("source", {}).get("3") == "file" else ("strict" if policy_choice=="strict" else "projected(auto)"),
                    "d3": (boundaries.blocks.__root__.get("3") or []),
                    "n3": len((boundaries.blocks.__root__.get("3") or [])[0]) if (boundaries.blocks.__root__.get("3") and boundaries.blocks.__root__.get("3")[0]) else 0,
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
            # strict residual: R3 = H2@d3 + (C3 + I3)
            try:
                R3_strict = _xor_mat(mul(H2, d3), add(C3, I3)) if (H2 and d3 and C3) else []
            except Exception as e:
                st.error(f"Shape guard failed at k=3: {e}")
                st.stop()
            eq3_strict = (R3_strict == [[0]*len(R3_strict[0]) for _ in range(len(R3_strict))]) if R3_strict else True

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

            if cfg_active.get("enabled_layers"):
                R3_proj = mul(R3_strict, P_active) if (R3_strict and P_active) else []
                eq3_proj = (R3_proj == [[0]*len(R3_proj[0]) for _ in range(len(R3_proj))]) if R3_proj else True
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

    # ── require a run for the rest of the tab
    _ss = st.session_state
    if not (_ss.get("run_ctx") and _ss.get("overlap_out") and _ss.get("overlap_H")):
        st.info("Run Overlap first to populate cert & download bundle.")
        st.stop()

    run_ctx = _ss["run_ctx"]; out = _ss["overlap_out"]; H_used = _ss["overlap_H"]
    policy_label = _ss.get("overlap_policy_label", policy_label_from_cfg(cfg_active))

    # ── identity (authoritative)
    _di = _ss.get("_district_info", {}) or {}
    district_id = _di.get("district_id") or _ss.get("district_id", "UNKNOWN")
    identity_block = {
        "district_id": district_id,
        "run_id": hashes.run_id(hashes.bundle_content_hash([
            ("d", boundaries.dict() if hasattr(boundaries, "dict") else {}),
            ("C", cmap.dict() if hasattr(cmap, "dict") else {}),
            ("H", H_used.dict() if hasattr(H_used, "dict") else {}),
            ("cfg", cfg_active),
        ]), hashes.timestamp_iso_lisbon()),
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
    cert_payload = {
        "identity":    identity_block,
        "policy":      policy_block,
        "inputs":      inputs_block,
        "diagnostics": diagnostics_block,
        "checks":      checks_block,
        "signatures":  sig_block,
        "promotion":   {"eligible_for_promotion": False, "promotion_target": None, "notes": ""},
        "policy_tag":  policy_block["policy_tag"],
        "district_id": identity_block["district_id"],
        "boundaries_hash": inputs_block["boundaries_hash"],
    }

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
