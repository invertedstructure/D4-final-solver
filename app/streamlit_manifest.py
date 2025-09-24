# --- robust loader with real package context for app/otcore or app/core ---
import sys, pathlib, importlib.util, types

HERE = pathlib.Path(__file__).resolve().parent   # .../app
OTCORE = HERE / "otcore"
CORE = HERE / "core"
PKG_DIR = OTCORE if OTCORE.exists() else CORE
PKG_NAME = "otcore" if OTCORE.exists() else "core"

# Ensure a real package exists in sys.modules so that relative imports work
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PKG_DIR)]  # namespace package path
    pkg.__file__ = str(PKG_DIR / "__init__.py")
    sys.modules[PKG_NAME] = pkg

def _load_pkg_module(fullname: str, rel_path: str):
    """Load module 'PKG_NAME.rel' from file with correct package context."""
    path = PKG_DIR / rel_path
    if not path.exists():
        raise ImportError(f"Required module file not found: {path}")
    spec = importlib.util.spec_from_file_location(fullname, str(path))
    mod = importlib.util.module_from_spec(spec)
    # set package explicitly for relative imports inside the module
    mod.__package__ = fullname.rsplit('.', 1)[0]
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Load submodules under the chosen package name
io = _load_pkg_module(f"{PKG_NAME}.io", "io.py")
hashes = _load_pkg_module(f"{PKG_NAME}.hashes", "hashes.py")
unit_gate = _load_pkg_module(f"{PKG_NAME}.unit_gate", "unit_gate.py")
overlap_gate = _load_pkg_module(f"{PKG_NAME}.overlap_gate", "overlap_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers = _load_pkg_module(f"{PKG_NAME}.towers", "towers.py")
manifest_mod = _load_pkg_module(f"{PKG_NAME}.manifest", "manifest.py")
export_mod = _load_pkg_module(f"{PKG_NAME}.export", "export.py")

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")
# --------------------------------------------------------------------------

import streamlit as st, json, os, tempfile, time

st.set_page_config(page_title="Odd Tetra App — Manifest Runner", layout="centered")
st.title("Odd Tetra — Manifest Runner (v0.1)")

st.write("Upload a `manifest.json` describing your run. The app will execute Unit → Overlap → Triangle → Towers, collect certs/CSVs, and package a report ZIP.")

mf = st.file_uploader("manifest.json", type=["json"])

if mf and st.button("Run manifest"):
    try:
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
