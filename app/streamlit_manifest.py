# --- import shim: robust loading for Streamlit Cloud (supports app/otcore or app/core) ---
import sys, pathlib, importlib.util

HERE = pathlib.Path(__file__).resolve().parent            # .../app
OTCORE = HERE / "otcore"                                  # .../app/otcore
CORE = HERE / "core"                                      # .../app/core
PKG = OTCORE if OTCORE.exists() else CORE

def _load_module(module_name: str, rel_path: str):
    path = PKG / rel_path
    if not path.exists():
        raise ImportError(f"Required module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Load modules directly by file path (no package re-exports needed)
io = _load_module("otcore_io", "io.py")
hashes = _load_module("otcore_hashes", "hashes.py")
unit_gate = _load_module("otcore_unit_gate", "unit_gate.py")
overlap_gate = _load_module("otcore_overlap_gate", "overlap_gate.py")
triangle_gate = _load_module("otcore_triangle_gate", "triangle_gate.py")
towers = _load_module("otcore_towers", "towers.py")
manifest_mod = _load_module("otcore_manifest", "manifest.py")
export_mod = _load_module("otcore_export", "export.py")

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")
# ------------------------------------------------------------------------------------------

import streamlit as st, json, os, tempfile, time

st.set_page_config(page_title="Odd Tetra App — Manifest Runner", layout="centered")
st.title("Odd Tetra — Manifest Runner (v0.1)")

st.write("Upload a `manifest.json` describing your run. The app will execute Unit → Overlap → Triangle → Towers, collect certs/CSVs, and package a report ZIP.")

mf = st.file_uploader("manifest.json", type=["json"])

if mf and st.button("Run manifest"):
    try:
        # Save manifest to a temp file
        tmpdir = tempfile.mkdtemp()
        mpath = os.path.join(tmpdir, "manifest.json")
        with open(mpath, "wb") as f:
            f.write(mf.read())
        report_dir = os.path.join("reports", f"run_{int(time.time())}")
        os.makedirs(report_dir, exist_ok=True)
        summary = manifest_mod.run_manifest(mpath, report_dir)
        st.success("Run complete.")
        st.json(summary)
        zpath = os.path.join(report_dir, "report.zip")
        export_mod.zip_report(report_dir, zpath)
        with open(zpath, "rb") as fz:
            st.download_button("Download report.zip", fz, file_name="report.zip")
    except Exception as e:
        st.error(f"Run failed: {e}")
