# app/otcore/export.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import os, zipfile, csv, json, hashlib, re
from pathlib import Path

from .io import dump_canonical
from .hashes import APP_VERSION, bundle_content_hash, timestamp_iso_lisbon

# ---------- small utils ----------

def ensure_dir(d: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)

def _slug(s: str) -> str:
    """Safe tag for filenames (letters/digits/_ only)."""
    s = (s or "").strip()
    if not s:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_") or "unknown"

# ---------- zip helper (kept from earlier) ----------

def zip_report(report_dir: str, out_zip_path: str) -> str:
    ensure_dir(os.path.dirname(out_zip_path) or ".")
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(report_dir):
            for file in files:
                full = os.path.join(folder, file)
                arc = os.path.relpath(full, report_dir)
                z.write(full, arc)
    return out_zip_path

# ---------- registry writer ----------

_REGISTRY_PATH = "registry.csv"
_REGISTRY_FIELDS = [
    "run_id",
    "timestamp",
    "app_version",
    "fix_id",
    "pass_vector",
    "policy",
    "hash_d",
    "hash_U",
    "hash_suppC",
    "hash_suppH",
    "notes",
]

def write_registry_row(
    *,
    fix_id: str,
    pass_vector: Iterable[int],
    policy: str,
    hash_d: str,
    hash_U: str,
    hash_suppC: str,
    hash_suppH: str,
    notes: str = "",
    registry_path: str = _REGISTRY_PATH,
) -> str:
    """Append one line to registry.csv (creates with header if absent)."""
    ensure_dir(os.path.dirname(registry_path) or ".")
    row = {
        "run_id": hashlib.sha256(
            f"{timestamp_iso_lisbon()}|{APP_VERSION}|{fix_id}".encode("utf-8")
        ).hexdigest(),
        "timestamp": timestamp_iso_lisbon(),
        "app_version": APP_VERSION,
        "fix_id": fix_id,
        "pass_vector": "[" + ",".join(str(int(b)) for b in pass_vector) + "]",
        "policy": policy,
        "hash_d": hash_d,
        "hash_U": hash_U,
        "hash_suppC": hash_suppC,
        "hash_suppH": hash_suppH,
        "notes": notes or "",
    }
    write_header = not os.path.exists(registry_path)
    with open(registry_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_REGISTRY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return registry_path

# ---------- cert writer ----------

def write_cert_json(payload: Dict[str, Any], out_dir: str = "certs") -> Tuple[str, str]:
    """
    Write a canonical, content-hashed JSON cert to certs/.
    Filename includes district, policy tag, and short hash.
    Returns (path, full_hash).
    """
    ensure_dir(out_dir)

    # Hash of the full payload (canonical)
    full_hash = hashlib.sha256(dump_canonical(payload)).hexdigest()
    short_hash = full_hash[:12]

    # Pull identity bits (robust to missing keys)
    identity = payload.get("identity", {})
    district_id = identity.get("district_id", "D?")
    policy_tag = identity.get("policy_tag")  # preferred if provided

    if not policy_tag:
        # Derive from policy label if present
        policy_label = (payload.get("policy") or {}).get("label", "")
        policy_tag = _slug(policy_label)

    # Build filename and write
    fname = f"overlap__{district_id}__{policy_tag}__{short_hash}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "wb") as f:
        f.write(dump_canonical(payload))

    return fpath, full_hash
