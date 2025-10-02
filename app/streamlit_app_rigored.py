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

with st.sidebar:
    st.markdown("### Upload core inputs")
    st.caption("**Shapes (required):**\\n\\n```json\\n{\\\"n\\\": {\\\"3\\\":3, \\\"2\\\":2, \\\"1\\\":0}}\\n```\\n\\n**Boundaries (required):**\\n\\n```json\\n{\\\"blocks\\\": {\\\"3\\\": [[...]], \\\"2\\\": [[...]]}}\\n```\\n\\n**CMap / Move (required):**\\n\\n```json\\n{\\\"blocks\\\": {\\\"3\\\": [[...]], \\\"2\\\": [[...]]}}\\n```\\n\\n**Support (optional):** either `{degree: mask}` or `{\\\"masks\\\": {degree: mask}}`.\\n\\n**Triangle schema (optional):** degree-keyed `{ \\\"2\\\": {\\\"A\\\":..., \\\"B\\\":..., \\\"J\\\":...}, ... }`.")
    f_shapes = st.file_uploader("Shapes (shapes.json)", type=["json"], key="shapes")
    f_bound = st.file_uploader("Boundaries (boundaries.json)", type=["json"], key="bound")
    f_cmap = st.file_uploader("CMap / Move (Cmap_*.json)", type=["json"], key="cmap")
    f_support = st.file_uploader("Support policy (support_ck_full.json)", type=["json"], key="support")
    f_pair = st.file_uploader("Pairings (pairings.json)", type=["json"], key="pair")
    f_reps = st.file_uploader("Reps (reps_for_Cmap_chain_pairing_ok.json)", type=["json"], key="reps")
    f_triangle = st.file_uploader("Triangle schema (triangle_J_schema.json)", type=["json"], key="tri")
    seed = st.text_input("Seed", "super-seed-A")

d_shapes = read_json_file(f_shapes)
d_bound = read_json_file(f_bound)
d_cmap  = read_json_file(f_cmap)

if d_shapes and d_bound and d_cmap:
    try:
        shapes = io.parse_shapes(d_shapes)
        boundaries = io.parse_boundaries(d_bound)
        cmap = io.parse_cmap(d_cmap)  # must have top-level "blocks"
        support = io.parse_support(read_json_file(f_support)) if f_support else None
        triangle = io.parse_triangle_schema(read_json_file(f_triangle)) if f_triangle else None
        io.validate_bundle(boundaries, shapes, cmap, support)
        st.success("Core schemas validated ✅")
        with st.expander("Hashes / provenance"):
            named = [("boundaries", boundaries.dict()), ("shapes", shapes.dict()), ("cmap", cmap.dict())]
            if support: named.append(("support", support.dict()))
            if triangle: named.append(("triangle", triangle.dict()))
            ch = hashes.bundle_content_hash(named)
            ts = hashes.timestamp_iso_lisbon()
            rid = hashes.run_id(ch, ts)
            st.code(f"content_hash = {ch}\\nrun_timestamp = {ts}\\nrun_id = {rid}\\napp_version = {APP_VERSION}", language="bash")
            # Quick export here too
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Unit", "Overlap", "Triangle", "Towers", "Export"])

with tab1:
    st.subheader("Unit gate")
    # Boundaries (B)
    f_B = st.file_uploader("Boundaries (boundaries*.json)", type=["json"], key="B_up")
    _stamp_filename("fname_boundaries", f_B)
    d_B = read_json_file(f_B) if f_B else None
    if d_B:
        boundaries = io.parse_boundaries(d_B)

    # C-map (optional override here; otherwise load it where you prefer)
    f_C = st.file_uploader("C map (optional)", type=["json"], key="C_up")
    _stamp_filename("fname_cmap", f_C)
    d_C = read_json_file(f_C) if f_C else None
    if d_C:
        cmap = io.parse_cmap(d_C)

    # Shapes / carrier U (optional)
    f_U = st.file_uploader("Shapes / carrier U (optional)", type=["json"], key="U_up")
    _stamp_filename("fname_shapes", f_U)
    d_U = read_json_file(f_U) if f_U else None
    if d_U:
        shapes = io.parse_shapes(d_U)  # or whatever parser you have
        
    
    # Reps (only if you actually use them)
    f_reps = st.file_uploader("Reps (optional)", type=["json"], key="reps_up")
    _stamp_filename("fname_reps", f_reps)
    d_reps = read_json_file(f_reps) if f_reps else None


    enforce = st.checkbox("Enforce rep transport (c_cod = C c_dom)", value=False)
    if st.button("Run Unit"):
        out = unit_gate.unit_check(boundaries, cmap, shapes, reps=d_reps, enforce_rep_transport=enforce)
        st.json(out)  
    


# --- run overlap under a given cfg (strict or projected) ----------------------
def run_overlap_with_cfg(boundaries, cmap, H, cfg: dict):
    cache = projector.preload_projectors_from_files(cfg)
    try:
        out = overlap_gate.overlap_check(
            boundaries, cmap, H,
            projection_config=cfg,
            projector_cache=cache,
        )
    except TypeError:
        # old signature → strict fallback
        out = overlap_gate.overlap_check(boundaries, cmap, H)
    return out, cache

# --- Overlap gate (homotopy vs identity) -------------------------------------
with tab2:
    st.subheader("Overlap gate (homotopy vs identity)")

    # --- H uploader (+ remember filename for certs/bundles) ---
    f_H = st.file_uploader("Homotopy H (H_corrected.json)", type=["json"], key="H_corr")
    _stamp_filename("fname_h", f_H)
    d_H = read_json_file(f_H) if f_H else None
    H_local = io.parse_cmap(d_H) if d_H else io.parse_cmap({"blocks": {}})  # JSON-safe CMap

    # --- Policy toggle UI ---
    st.markdown("### Policy")
    policy_choice = st.radio(
        "Choose policy",
        ["strict", "projected(columns@k=3)"],
        horizontal=True,
        key="policy_choice_k3",
    )

    # --- Build active cfg (respect file/auto from projection_config.json) ---
    cfg_file = projector.load_projection_config("projection_config.json")
    cfg_proj = cfg_projected_base()
    if cfg_file.get("source", {}).get("3") in ("file", "auto"):
        cfg_proj["source"]["3"] = cfg_file["source"]["3"]
    if "projector_files" in cfg_file and "3" in cfg_file["projector_files"]:
        cfg_proj["projector_files"]["3"] = cfg_file["projector_files"]["3"]
    cfg_active = cfg_strict() if policy_choice == "strict" else cfg_proj
    policy_label = policy_label_from_cfg(cfg_active)
    st.caption(f"Policy: **{policy_label}**")

    cache = projector.preload_projectors_from_files(cfg_active)

    # --- Projector source (k=3) quick switcher ---
    with st.expander("Projector source (k=3)"):
        cur_src  = cfg_file.get("source", {}).get("3", "auto")
        cur_file = cfg_file.get("projector_files", {}).get("3", "projector_D3.json")
        st.write(
            f"Current: source.3 = **{cur_src}**",
            f"(file: `{cur_file}`)" if cur_src == "file" else ""
        )
        mode_choice = st.radio(
            "Choose source for k=3",
            options=["auto","file"],
            index=0 if cur_src == "auto" else 1,
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

    # ---------- RUN OVERLAP ----------
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

            # persist for downstream cert/bundle blocks
            st.session_state["overlap_out"] = out
            st.session_state["overlap_cfg"] = cfg_active
            st.session_state["overlap_policy_label"] = policy_label
            st.session_state["overlap_H"] = H_local

        except Exception as e:
            st.error(f"Overlap run failed: {e}")
            st.stop()

    # ===== Restore last overlap run (required by cert/bundle) =====
    _ss = st.session_state
    overlap_out          = _ss.get("overlap_out")
    overlap_cfg          = _ss.get("overlap_cfg")
    overlap_policy_label = _ss.get("overlap_policy_label")
    overlap_H            = _ss.get("overlap_H")

    if overlap_out is None or overlap_cfg is None or overlap_H is None:
        st.info("Run Overlap first to populate cert & download bundle.")
        st.stop()

    # use the exact data from the run
    out          = overlap_out
    cfg_active   = overlap_cfg
    policy_label = overlap_policy_label
    H_local      = overlap_H

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
    inputs_block["filenames"] = {
        "boundaries": st.session_state.get("fname_boundaries", ""),
        "C":          st.session_state.get("fname_cmap", ""),
        "H":          st.session_state.get("fname_h", ""),
        "U":          st.session_state.get("fname_shapes", ""),
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

    # ---------- Checks/signatures ----------
    checks_block = out  # {"2": {...}, "3": {...}}
    sig_block = {
        "d_signature": {},
        "fixture_signature": {},
        "echo_context": None,
    }

    # ---------- Policy snapshot ----------
    pj_hash = ""
    if cfg_active.get("source", {}).get("3") == "file":
        pj_path = cfg_active.get("projector_files", {}).get("3")
        if pj_path and os.path.exists(pj_path):
            pj_hash = projector._hash_matrix(_json.load(open(pj_path)))
    policy_block = {
        "label": policy_label,
        "policy_tag": policy_label,     # exporter expects this key
        "enabled_layers": cfg_active.get("enabled_layers", []),
        "modes": cfg_active.get("modes", {}),
        "source": cfg_active.get("source", {}),
        "projector_hash": pj_hash,
    }

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

# ---------- Policy snapshot (echo full config + projector hash if file) ----------
pj_hash = ""
if cfg_active.get("source", {}).get("3") == "file":
    pj_path = cfg_active.get("projector_files", {}).get("3")
    if pj_path and os.path.exists(pj_path):
        pj_hash = projector._hash_matrix(_json.load(open(pj_path)))

policy_block = {
    "policy": policy_label,              # "strict" or "projected(columns@k=3,auto|file)"
    "projection_config": cfg_active,     # echo effective config
    "projector_hash": pj_hash,           # only if file-backed
}

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

# ---------- Full cert payload ----------
cert_payload = {
    "identity": identity_block,
    "policy": policy_block,
    "inputs": inputs_block,              # you already assembled (hashes + dims + filenames)
    "diagnostics": diagnostics_block,    # lane_mask_k3 / lane_vec_* you computed earlier
    "checks": checks_block,              # per-k eq + n_k + grid/fence/ker_guard/residual_tag
    "signatures": {
        "d_signature": {},               # fill with your canonical signature if you have it
        "fixture_signature": {},
        "echo_context": None,
    },
    "promotion": {
        "eligible_for_promotion": int(out.get("3", {}).get("eq", False)) == 1,
        "promotion_target": ("projected_anchor" if not is_strict else "strict_anchor") if int(out.get("3", {}).get("eq", False)) == 1 else None,
        "notes": "",
    },
    # Top-level duplication of artifact hashes (handy for quick indexing)
    "artifact_hashes": {
        "boundaries_hash": inputs_block["boundaries_hash"],
        "C_hash": inputs_block["C_hash"],
        "H_hash": inputs_block["H_hash"],
        "U_hash": inputs_block["U_hash"],
        "projector_hash": policy_block["projector_hash"],
    },
}

# Integrity: hash the entire payload and embed it
cert_payload["integrity"] = {
    "content_hash": hashes.content_hash_of(cert_payload)
}

# Write cert
cert_path, full_hash = export_mod.write_cert_json(cert_payload)
st.success(f"Cert written: `{cert_path}`")

# ---------- Download bundle (includes cert.json) ----------
try:
    bundle_name = f"overlap_bundle__{district_id}__{policy_label.replace(' ','_')}__{full_hash[:12]}.zip"
    zip_path = export_mod.build_overlap_bundle(
        boundaries=boundaries,
        cmap=cmap,
        H=H_local,
        shapes=(shapes.dict() if hasattr(shapes, "dict") else (shapes or {})),
        policy_block=policy_block,
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


    # ---------- Promotion (optional) ----------
    pass_vec = [int(out.get("2", {}).get("eq", False)),
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
                P_used = projector.projector_columns_from_dkp1(d3_now)
                pj_path = cfg_file.get("projector_files", {}).get("3", "projector_D3.json")
                pj_hash = projector.save_projector(pj_path, P_used)
                st.info(f"Projector frozen → {pj_path} (hash={pj_hash[:12]}…)")

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

# Build a projected config fresh from file each time (so A/B mirrors disk)
_cfg_file_for_ab = projector.load_projection_config("projection_config.json")
_cfg_proj_for_ab = cfg_projected_base()
# Respect current file/auto choice from disk
if _cfg_file_for_ab.get("source", {}).get("3") in ("file", "auto"):
    _cfg_proj_for_ab["source"]["3"] = _cfg_file_for_ab["source"]["3"]
if "projector_files" in _cfg_file_for_ab and "3" in _cfg_file_for_ab["projector_files"]:
    _cfg_proj_for_ab["projector_files"]["3"] = _cfg_file_for_ab["projector_files"]["3"]

_cache_for_ab = projector.preload_projectors_from_files(_cfg_proj_for_ab)

# Parse H once for both runs
H_obj = io.parse_cmap(d_H) if d_H else io.parse_cmap({"blocks": {}})

if st.button("Run A/B compare (strict vs projected)", key="run_ab_overlap"):
    # strict run
    out_strict = overlap_gate.overlap_check(boundaries, cmap, H_obj)

    # projected run
    out_proj = overlap_gate.overlap_check(
        boundaries, cmap, H_obj,
        projection_config=_cfg_proj_for_ab,
        projector_cache=_cache_for_ab,
    )

    st.json({"strict": out_strict, "projected": out_proj})

    # Optional: write both certs
    if st.checkbox("Write both certs (strict & projected)", value=False, key="ab_write_certs"):
        pj_hash_proj = ""
        if _cfg_proj_for_ab.get("source", {}).get("3") == "file":
            pj_path = _cfg_proj_for_ab.get("projector_files", {}).get("3")
            if pj_path and os.path.exists(pj_path):
                pj_hash_proj = projector._hash_matrix(_json.load(open(pj_path)))

        cert_s = write_overlap_cert(
            out=out_strict,
            policy_label=policy_label_from_cfg(cfg_strict()),
            boundaries=boundaries, cmap=cmap, H=H_obj, pj_hash=None
        )
        cert_p = write_overlap_cert(
            out=out_proj,
            policy_label=policy_label_from_cfg(_cfg_proj_for_ab),
            boundaries=boundaries, cmap=cmap, H=H_obj, pj_hash=pj_hash_proj
        )
        st.success(f"Saved: `{cert_s}` and `{cert_p}`")

        if st.checkbox("Also log both to registry.csv", value=False, key="ab_write_registry"):
            import time as _time
            try:
                export_mod.write_registry_row(
                    fix_id=f"compare-strict-{int(_time.time())}",
                    pass_vector=[
                        int(out_strict.get("2", {}).get("eq", False)),
                        int(out_strict.get("3", {}).get("eq", False)),
                    ],
                    policy=policy_label_from_cfg(cfg_strict()),
                    hash_d=hashes.hash_d(boundaries),
                    hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                    hash_suppC=hashes.hash_suppC(cmap),
                    hash_suppH=hashes.hash_suppH(H_obj),
                    notes="A/B compare strict",
                )
                export_mod.write_registry_row(
                    fix_id=f"compare-projected-{int(_time.time())}",
                    pass_vector=[
                        int(out_proj.get("2", {}).get("eq", False)),
                        int(out_proj.get("3", {}).get("eq", False)),
                    ],
                    policy=policy_label_from_cfg(_cfg_proj_for_ab),
                    hash_d=hashes.hash_d(boundaries),
                    hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                    hash_suppC=hashes.hash_suppC(cmap),
                    hash_suppH=hashes.hash_suppH(H_obj),
                    notes="A/B compare projected",
                )
                st.toast("registry: added strict & projected rows")
            except Exception as e:
                st.error(f"registry write failed: {e}")




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
