# --- robust loader with real package context (supports app/otcore or app/core) ---
import sys, pathlib, importlib.util, types
import streamlit as st
import json

# Streamlit MUST be configured before ANY other st.* call:
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# --- policy label helper (UI + logging) ---
def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    parts = []
    for kk in sorted(cfg["enabled_layers"]):
        mode = cfg.get("modes", {}).get(str(kk), "none")
        src  = cfg.get("source", {}).get(str(kk), "auto")
        parts.append(f"{mode}@k={kk},{src}")
    return "projected(" + ";".join(parts) + ")"


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
            H = io.parse_cmap(d_H)  # reuse CMap schema for H blocks

            # --- Show actual working directory & files there ---
            import os, inspect, time
            st.write("cwd:", os.getcwd())
            st.write("files in cwd:", os.listdir("."))
            st.write("cfg file in CWD:", os.path.exists("projection_config.json"))

            # --- Load projection config ---
            cfg = projector.load_projection_config("projection_config.json")
            policy_label = policy_label_from_cfg(cfg)
            st.caption(f"Policy: {policy_label}")
            st.json({"cfg": cfg})
            cache = projector.preload_projectors_from_files(cfg)

            # Policy badge (string)
            layers  = cfg.get("enabled_layers", [])
            modes   = cfg.get("modes", {})
            sources = cfg.get("source", {})
            if layers:
                policy_str = "projected(" + ",".join(
                    f"{modes.get(str(k),'none')}@k={k},{sources.get(str(k),'auto')}" for k in layers
                ) + ")"
            else:
                policy_str = "strict"
            st.caption(f"Policy: {policy_str}")

import json as _json, hashlib as _hashlib

with st.expander("Projector source (k=3)"):
    cur_src = cfg.get("source", {}).get("3", "auto")
    cur_file = cfg.get("projector_files", {}).get("3", "projector_D3.json")
    st.write(
        f"Current: source.3 = **{cur_src}**",
        f"(file: `{cur_file}`)" if cur_src == "file" else ""
    )

    mode_choice = st.radio(
        "Choose source for k=3",
        options=["auto", "file"],
        index=0 if cur_src == "auto" else 1,
        horizontal=True,
        key="proj_src_choice_k3",
    )
    file_path = st.text_input(
        "Projector file", value=cur_file, disabled=(mode_choice == "auto")
    )

    if st.button("Apply projector source"):
        cfg.setdefault("source", {})["3"] = mode_choice
        if mode_choice == "file":
            cfg.setdefault("projector_files", {})["3"] = file_path
        else:
            if "projector_files" in cfg and "3" in cfg["projector_files"]:
                del cfg["projector_files"]["3"]
        with open("projection_config.json", "w") as _f:
            _json.dump(cfg, _f, indent=2)
        st.success(f"projection_config.json updated → source.3 = {mode_choice}")

    # Optional guard to check drift between file and auto Π3
    if cur_src == "file" and st.button("Validate file vs auto Π3"):
        d3_now = boundaries.blocks.__root__.get("3")
        if d3_now is None:
            st.error("No d3 in boundaries; cannot validate.")
        else:
            autoP = projector.projector_columns_from_dkp1(d3_now)
            try:
                with open(cur_file, "r") as _pf:
                    fileP = _json.load(_pf)
            except Exception as e:
                st.error(f"Could not load {cur_file}: {e}")
                fileP = None

            if fileP is not None:
                h_auto = _hashlib.sha256(_json.dumps(autoP, sort_keys=True).encode()).hexdigest()
                h_file = _hashlib.sha256(_json.dumps(fileP, sort_keys=True).encode()).hexdigest()
                if h_auto == h_file:
                    st.success(f"OK: projector matches auto (hash={h_auto[:12]}…)")
                else:
                    st.warning(f"DRIFT: file {cur_file} hash={h_file[:12]}… vs auto hash={h_auto[:12]}…")


            
            # --- Sanity: which overlap_gate is actually loaded?
            st.write("overlap_gate.__file__ =", getattr(overlap_gate, "__file__", "<none>"))
            st.write("overlap_check signature =", str(inspect.signature(overlap_gate.overlap_check)))

            # Lane mask peek for k=3 (to see ker columns)
            d3 = boundaries.blocks.__root__.get('3')
            if d3 is not None:
                lane_mask = [1 if any(row[j] for row in d3) else 0 for j in range(len(d3[0]))]
                st.write("k=3 lane_mask (1=lane, 0=ker):", lane_mask)

            # --- Run overlap; if projection kwargs aren't accepted, fall back and warn
            try:
                out = overlap_gate.overlap_check(
                    boundaries, cmap, H,
                    projection_config=cfg,
                    projector_cache=cache
                )
            except TypeError as e:
                st.warning(
                    "overlap_gate is running in STRICT mode (old module signature) — "
                    "hard-restart the app after patching overlap_gate.py. "
                    f"TypeError: {e}"
                )
                out = overlap_gate.overlap_check(boundaries, cmap, H)

                        # Results
            st.json(out)

            # Guard warning (if file-mode drift was detected inside apply_projection)
            for key, val in list(cache.items()):
                if key.startswith("guard_warning_k"):
                    st.warning(f"[{key}] {val['msg']} | file={val['hash_file']} auto={val['hash_auto']}")

  # ---- Build pass-vector ----
pass_vec = [
    int(out.get("2", {}).get("eq", False)),
    int(out.get("3", {}).get("eq", False)),
]
all_green = all(v == 1 for v in pass_vec)

# ---- Promotion: freeze projector + log hash (only if all green) ----
if all_green:
    st.success("Green — eligible for promotion.")
    flip_to_file = st.checkbox("After promotion, switch to FILE-backed projector", value=True, key="flip_to_file_k3")
    force_back_to_auto = st.checkbox("…or keep AUTO (don’t lock now)", value=False, key="keep_auto_k3")

    if st.button("Promote & Freeze Projector"):
        d3_now = boundaries.blocks.__root__.get("3")
        if d3_now is None:
            st.error("No d3 in boundaries; cannot freeze projector.")
        else:
            # Always freeze the exact Π3 used now (correct by construction)
            P_used = projector.projector_columns_from_dkp1(d3_now)
            pj_path = cfg.get("projector_files", {}).get("3", "projector_D3.json")
            pj_hash = projector.save_projector(pj_path, P_used)
            st.info(f"Projector frozen → {pj_path} (hash={pj_hash[:12]}…)")

            # Optionally flip config: file-backed or back to auto
            if flip_to_file and not force_back_to_auto:
                cfg.setdefault("source", {})["3"] = "file"
                cfg.setdefault("projector_files", {})["3"] = pj_path
                with open("projection_config.json", "w") as _f:
                    _json.dump(cfg, _f, indent=2)
                st.toast("projection_config.json → FILE-backed (k=3)")
            else:
                cfg.setdefault("source", {})["3"] = "auto"
                if "projector_files" in cfg and "3" in cfg["projector_files"]:
                    del cfg["projector_files"]["3"]
                with open("projection_config.json", "w") as _f:
                    _json.dump(cfg, _f, indent=2)
                st.toast("projection_config.json → AUTO (k=3)")

            # Registry row including projector hash
            import time as _time
            fix_id = f"overlap-{int(_time.time())}"
            try:
                export_mod.write_registry_row(
                    fix_id=fix_id,
                    pass_vector=pass_vec,
                    policy=policy_label,  # strict / projected(...)
                    hash_d=hashes.hash_d(boundaries),
                    hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                    hash_suppC=hashes.hash_suppC(cmap),
                    hash_suppH=hashes.hash_suppH(H),
                    notes=f"proj_hash={pj_hash}"
                )
                st.success("Registry updated with projector hash.")
            except Exception as e:
                st.error(f"Failed to write registry row: {e}")
else:
    st.info("Not promoting: some checks are red.")


        # ---- Normal registry write (non-promotion) ----
# If you still want a basic row every run (even when not promoting), keep this:
import time
fix_id = f"overlap-{int(time.time())}"
try:
    export_mod.write_registry_row(
        fix_id=fix_id,
        pass_vector=pass_vec,
        policy=policy_label,  # strict / projected(...)
        hash_d=hashes.hash_d(boundaries),
        hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
        hash_suppC=hashes.hash_suppC(cmap),
        hash_suppH=hashes.hash_suppH(H),
        notes=""
    )
    st.success("Registry updated (registry.csv).")
except Exception as e:
    st.error(f"Failed to write registry row: {e}")



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
