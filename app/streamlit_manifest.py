
# app/streamlit_manifest.py (top)
import streamlit as st, json, os, tempfile, time
from app.core.manifest import run_manifest, ManifestError
from app.core.export import zip_report


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
