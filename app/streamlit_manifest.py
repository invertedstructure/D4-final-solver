import importlib

io = importlib.import_module('core.io')
hashes = importlib.import_module('core.hashes')
unit_gate = importlib.import_module('core.unit_gate')
overlap_gate = importlib.import_module('core.overlap_gate')
triangle_gate = importlib.import_module('core.triangle_gate')
towers = importlib.import_module('core.towers')

# then use:
# unit_gate.unit_check(...)
# overlap_gate.overlap_check(...)
# triangle_gate.triangle_check(...)
# towers.run_tower(...)
# hashes.bundle_content_hash(...)
# hashes.timestamp_iso_lisbon(...)
# hashes.run_id(...)


# --- import shim: make imports work whether CWD is repo root or app/ ---
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent          # .../app
ROOT = HERE.parent                                      # repo root
# Ensure both app/ and repo root are importable
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
# -----------------------------------------------------------------------

# --- keep your import shim here ---

import streamlit as st, json, os, tempfile, time
from otcore.manifest import run_manifest, ManifestError
from otcore.export import zip_report




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
        summary = run_manifest(mpath, report_dir)
        st.success("Run complete.")
        st.json(summary)
        zpath = os.path.join(report_dir, "report.zip")
        zip_report(report_dir, zpath)
        with open(zpath, "rb") as fz:
            st.download_button("Download report.zip", fz, file_name="report.zip")
    except ManifestError as e:
        st.error(f"Manifest error: {e}")
    except Exception as e:
        st.error(f"Run failed: {e}")
