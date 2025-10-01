from __future__ import annotations
import os, zipfile, csv, json
from pathlib import Path


def zip_report(report_dir: str, out_zip_path: str) -> str:
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(report_dir):
            for file in files:
                full = os.path.join(folder, file)
                arc = os.path.relpath(full, report_dir)
                z.write(full, arc)
    return out_zip_path

# ---------------- registry CSV (tiny helper) ----------------
from datetime import datetime
import csv

REGISTRY_PATH = "registry.csv"

CSV_FIELDS = [
    "timestamp", "fix_id",
    "grid", "wiggle", "fence", "echo",
    "policy",
    "hash_d", "hash_U", "hash_suppC", "hash_suppH",
    "notes",
]

def _ensure_header(path: str, fields: list[str]) -> None:
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if need_header:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()

def write_registry_row(
    *,
    fix_id: str,
    pass_vector: list[int],
    policy: str,
    hash_d: str,
    hash_U: str,
    hash_suppC: str,
    hash_suppH: str,
    notes: str = "",
    path: str = REGISTRY_PATH,
) -> str:
    """
    Append one row to the registry CSV. Returns the path written.
    pass_vector is [grid, wiggle, fence, echo] as ints (0/1).
    """
    grid, wiggle, fence, echo = (pass_vector + [0, 0, 0, 0])[:4]
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "fix_id": fix_id,
        "grid": grid,
        "wiggle": wiggle,
        "fence": fence,
        "echo": echo,
        "policy": policy,
        "hash_d": hash_d,
        "hash_U": hash_U,
        "hash_suppC": hash_suppC,
        "hash_suppH": hash_suppH,
        "notes": notes,
    }
    _ensure_header(path, CSV_FIELDS)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writerow(row)
    return path
    

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_cert_json(payload: Dict[str, Any], out_dir: str = "certs") -> str:
    ensure_dir(out_dir)
    # content hash of full payload
    full_hash = hashlib.sha256(dump_canonical(payload)).hexdigest()
    fname = f"overlap__{payload['identity']['district_id']}__{payload['policy']['policy_tag']}__{full_hash[:8]}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "wb") as f:
        f.write(dump_canonical(payload))
    return fpath, full_hash



