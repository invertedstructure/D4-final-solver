# --- import shim: make imports robust on Streamlit Cloud ---
import sys, pathlib, importlib.util

HERE = pathlib.Path(__file__).resolve().parent            # .../app
CORE = HERE / "core"                                      # .../app/core

def _load_module(module_name: str, rel_path: str):
    path = CORE / rel_path
    if not path.exists():
        raise ImportError(f"Required module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Load core modules directly by file path (no package re-exports needed)
io = _load_module("core_io", "io.py")
hashes = _load_module("core_hashes", "hashes.py")
unit_gate = _load_module("core_unit_gate", "unit_gate.py")
overlap_gate = _load_module("core_overlap_gate", "overlap_gate.py")
triangle_gate = _load_module("core_triangle_gate", "triangle_gate.py")
towers = _load_module("core_towers", "towers.py")
manifest_mod = _load_module("core_manifest", "manifest.py")
export_mod = _load_module("core_export", "export.py")

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")
# -----------------------------------------------------------------------------

import streamlit as st, json, os, tempfile

st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")

def read_json_file(file):
    if not file: return None
    try:
        import json
        return json.load(file)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return None

with st.sidebar:
    st.markdown("### Upload core inputs")
    f_bound = st.file_uploader("Boundaries (boundaries.json)", type=["json"], key="bound")
    f_shapes = st.file_uploader("Shapes (shapes.json)", type=["json"], key="shapes")
    f_cmap = st.file_uploader("Chain Map / Move (Cmap_*.json)", type=["json"], key="cmap")
    f_support = st.file_uploader("Support policy (support_ck_full.json)", type=["json"], key="support")
    f_pair = st.file_uploader("Pairings (pairings.json)", type=["json"], key="pair")
    f_reps = st.file_uploader("Reps (reps_for_Cmap_chain_pairing_ok.json)", type=["json"], key="reps")
    f_triangle = st.file_uploader("Triangle schema (triangle_J_schema.json)", type=["json"], key="tri")
    seed = st.text_input("Seed", "super-seed-A")

# Parse & validate minimal
d_bound = read_json_file(f_bound)
d_shapes = read_json_file(f_shapes)
d_cmap = read_json_file(f_cmap)
if d_bound and d_shapes and d_cmap:
    try:
        boundaries = io.parse_boundaries(d_bound)
        shapes = io.parse_shapes(d_shapes)
        cmap = io.parse_cmap(d_cmap)
        support = io.parse_support(read_json_file(f_support)) if f_support else None
        triangle = io.parse_triangle_schema(read_json_file(f_triangle)) if f_triangle else None
        io.validate_bundle(boundaries, shapes, cmap, support)
        st.success("Core schemas validated ✅")
    except Exception as e:
        st.error(f"Validation error: {e}")
        st.stop()
else:
    st.info("Upload Boundaries, Shapes, and Cmap to enable gates.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Unit", "Overlap", "Triangle", "Towers"])

with tab1:
    st.subheader("Unit gate")
    if st.button("Run Unit"):
        out = unit_gate.unit_check(boundaries, cmap, shapes)
        st.json(out)

with tab2:
    st.subheader("Overlap gate (homotopy vs identity)")
    f_H = st.file_uploader("Homotopy H (H_corrected.json)", type=["json"], key="H_corr")
    d_H = read_json_file(f_H) if f_H else None
    if st.button("Run Overlap"):
        if not d_H:
            st.error("Upload H_corrected.json")
        else:
            H = io.parse_cmap(d_H)  # reuse CMap schema for H blocks
            out = overlap_gate.overlap_check(boundaries, cmap, H)
            st.json(out)

with tab3:
    st.subheader("Triangle gate")
    if triangle is None:
        st.info("Upload triangle schema to run.")
    else:
        if st.button("Run Triangle"):
            out = triangle_gate.triangle_check(boundaries, triangle)
            st.json(out)

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
