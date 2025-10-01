
# --- robust loader with real package context (supports app/otcore or app/core) ---
import sys, pathlib, importlib.util, types, json, os
import streamlit as st

HERE = pathlib.Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
CORE = HERE / "core"
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
projector = _load_pkg_module(f"{PKG_NAME}.projector", "projector.py")
io = _load_pkg_module(f"{PKG_NAME}.io", "io.py")
hashes = _load_pkg_module(f"{PKG_NAME}.hashes", "hashes.py")
unit_gate = _load_pkg_module(f"{PKG_NAME}.unit_gate", "unit_gate.py")
overlap_gate = _load_pkg_module(f"{PKG_NAME}.overlap_gate", "overlap_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers = _load_pkg_module(f"{PKG_NAME}.towers", "towers.py")
export_mod = _load_pkg_module(f"{PKG_NAME}.export", "export.py")
APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")

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
    enforce = st.checkbox("Enforce rep transport (c_cod = C c_dom)", value=False)
    d_reps  = read_json_file(f_reps) if f_reps else None
    if st.button("Run Unit"):
        out = unit_gate.unit_check(boundaries, cmap, shapes, reps=d_reps, enforce_rep_transport=enforce)
        st.json(out)
        
with tab2:
    st.subheader("Overlap gate (homotopy vs identity)")
    f_H = st.file_uploader("Homotopy H (H_corrected.json)", type=["json"], key="H_corr")
    d_H = read_json_file(f_H) if f_H else None

    if st.button("Run Overlap"):
        if not d_H:
            st.error("Upload H_corrected.json")
        else:
            # Parse H using the same CMap schema
            H = io.parse_cmap(d_H)

            # --- DEBUG 0: Is the projection config visible from CWD? ---
            import os
            st.write("cfg file in CWD:", os.path.exists("projection_config.json"))

            # --- Load projection config via the dynamically loaded module ---
            cfg = projector.load_projection_config("projection_config.json")
            st.json({"cfg": cfg})  # <-- you should see enabled_layers etc here

            # Preload any file-based projectors
            cache = projector.preload_projectors_from_files(cfg)

            # --- Optional: show policy tag ---
            layers = cfg.get("enabled_layers", [])
            modes = cfg.get("modes", {})
            sources = cfg.get("source", {})
            if layers:
                policy_str = "projected(" + ",".join(
                    f"{modes.get(str(k),'none')}@k={k},{sources.get(str(k),'auto')}" for k in layers
                ) + ")"
            else:
                policy_str = "strict"
            st.caption(f"Policy: {policy_str}")

            # --- DEBUG 1: show lane mask for k=3 so we know if there IS a ker column ---
            d3 = boundaries.blocks.__root__.get('3')
            if d3 is not None:
                lane_mask = [1 if any(row[j] for row in d3) else 0 for j in range(len(d3[0]))]
                st.write("k=3 lane_mask (1=lane, 0=ker):", lane_mask)

            # --- Run overlap with projection config wired in ---
            out = overlap_gate.overlap_check(
                boundaries, cmap, H,
                projection_config=cfg,
                projector_cache=cache
            )
            st.json(out)


with tab3:
    st.subheader("Triangle gate")
    if 'triangle' not in locals() or triangle is None:
        st.info("Upload triangle schema to run.")
    else:
        if st.button("Run Triangle"):
            out = triangle_gate.triangle_check(boundaries, triangle, shapes=shapes)
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
