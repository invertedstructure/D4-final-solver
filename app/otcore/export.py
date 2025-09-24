
from __future__ import annotations
import os, zipfile
from pathlib import Path

def zip_report(report_dir: str, out_zip_path: str) -> str:
    report_dir = str(report_dir)
    out_zip_path = str(out_zip_path)
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(report_dir):
            for file in files:
                full = os.path.join(folder, file)
                arc = os.path.relpath(full, report_dir)
                z.write(full, arc)
    return out_zip_path
