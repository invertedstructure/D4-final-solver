# app/otcore/export.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
import os, zipfile, csv, json, hashlib, re, shutil
from pathlib import Path
from .io import dump_canonical
from .hashes import APP_VERSION, bundle_content_hash, timestamp_iso_lisbon

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _jsonable(obj: Any) -> Any:
    """Return a JSON-serializable view of obj."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj  # assume already serializable
  

def ensure_dir(d: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)

def write_gallery_row(row: dict, key_tuple: tuple, path: str = "gallery.csv") -> str:
    """
    Append a row to gallery.csv only if the key_tuple is unseen.
    key_tuple should be a tuple of primitives (e.g., strings).
    Returns 'written' or 'ignored'.
    """
    ensure_dir(os.path.dirname(path) or ".")
    seen = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for rr in r:
                seen.add(rr.get("__key__", ""))

    key_str = json.dumps(key_tuple, separators=(",", ":"))
    if key_str in seen:
        return "ignored"

    # flatten + include key
    flat = dict(row)
    flat["__key__"] = key_str

    write_header = (not os.path.exists(path))
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
        if write_header:
            w.writeheader()
        w.writerow(flat)
    return "written"



# --- bundle builder: write B/C/H/U + policy + cert into a single .zip ----------
def build_overlap_bundle(
    *,
    boundaries,
    cmap,
    H,
    shapes,
    policy_block: dict,
    cert_path: str,
    out_zip: str = "overlap_bundle.zip",
) -> str:
    """
    Create a single zip containing:
      - boundaries.json
      - cmap.json
      - H.json
      - shapes.json
      - policy.json  (effective policy snapshot used for the run)
      - cert.json    (the cert file you just wrote; copied verbatim)

    Returns the path to the written zip.
    """
    import json
    import zipfile
    from pathlib import Path

    def _plain(obj):
        """Best-effort JSON-safe conversion for pydantic-like objects."""
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        # Many of your dataclasses already hold plain dicts/lists; use as-is.
        return obj if obj is not None else {}

    # Prepare JSON-safe payloads
    b_payload = _plain(boundaries)
    c_payload = _plain(cmap)
    h_payload = _plain(H)
    u_payload = _plain(shapes)

    # Ensure output dir exists
    out_zip_path = Path(out_zip)
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Read cert bytes (if present) so we can embed the exact cert
    cert_bytes = b""
    cert_name = "cert.json"
    try:
        cert_bytes = Path(cert_path).read_bytes()
        # normalize name inside the zip
        if Path(cert_path).name:
            cert_name = Path(cert_path).name
    except Exception:
        # keep empty; we still build the rest
        cert_bytes = b""

    # Write the zip
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("boundaries.json", json.dumps(b_payload, indent=2))
        z.writestr("cmap.json",       json.dumps(c_payload, indent=2))
        z.writestr("H.json",          json.dumps(h_payload, indent=2))
        z.writestr("shapes.json",     json.dumps(u_payload, indent=2))
        z.writestr("policy.json",     json.dumps(policy_block or {}, indent=2))
        if cert_bytes:
            z.writestr(cert_name, cert_bytes)

    return str(out_zip_path)


def build_overlap_download_bundle(
    *,
    boundaries,
    cmap,
    H,
    shapes,
    policy_block: Dict[str, Any],
    cert_path: str,
    out_zip: str,
    cfg_snapshot: Dict[str, Any] | None = None,
) -> str:
    """
    Write a compact, reproducible bundle with everything to re-run/verify:
      - boundaries.json, cmap.json, H.json, shapes.json
      - policy.json (policy_block) + projection_config.json (cfg_snapshot)
      - cert.json (copy of the written cert)
    Return the path to the .zip.
    """
    # prep temp dir
    tmp_dir = Path(".bundle_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    # write inputs
    with open(tmp_dir / "boundaries.json", "wb") as f:
        f.write(dump_canonical(boundaries.dict() if hasattr(boundaries, "dict") else boundaries))
    with open(tmp_dir / "cmap.json", "wb") as f:
        f.write(dump_canonical(cmap.dict() if hasattr(cmap, "dict") else cmap))
    with open(tmp_dir / "H.json", "wb") as f:
        f.write(dump_canonical(H.dict() if hasattr(H, "dict") else H))
    with open(tmp_dir / "shapes.json", "w") as f:
        json.dump(_jsonable(shapes), f, indent=2)  # <-- JSON-safe

    # write policy + cfg snapshot
    with open(tmp_dir / "policy.json", "w") as f:
        json.dump(_jsonable(policy_block), f, indent=2)
    if cfg_snapshot:
        with open(tmp_dir / "projection_config.json", "w") as f:
            json.dump(_jsonable(cfg_snapshot), f, indent=2)

    # copy cert (if present)
    if cert_path and os.path.exists(cert_path):
        shutil.copy2(cert_path, tmp_dir / "cert.json")

    # zip it
    ensure_dir(os.path.dirname(out_zip) or ".")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in tmp_dir.iterdir():
            z.write(str(p), arcname=p.name)

    # cleanup temp
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_zip

# --- plain-obj serializer helpers --------------------------------------------
def _maybe_blocks(obj):
    # pydantic Boundaries/CMap-style -> {"blocks": {...}}
    b = getattr(getattr(obj, "blocks", None), "__root__", None)
    return {"blocks": b} if b is not None else None

def to_plain(x):
    if x is None:
        return None
    # common primitives
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_plain(v) for v in x]
    # pydantic-like .dict()
    if hasattr(x, "dict"):
        try:
            return x.dict()
        except Exception:
            pass
    # pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass
    # special “blocks” holder
    blk = _maybe_blocks(x)
    if blk is not None:
        return blk
    # numpy arrays
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    # fallback
    return str(x)


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
    import io as _io
import os, zipfile, json as _json

def build_download_bundle(*, 
    boundaries, cmap, H, shapes, 
    policy_block: dict, 
    cert_path: str,
    out_zip: str = "overlap_bundle.zip"
) -> str:
    """
    Writes a zip with:
      - boundaries.json, cmap.json, H.json, shapes.json
      - policy.json (the policy snapshot you used)
      - cert.json (the cert you just wrote)
    Returns the absolute path to the zip.
    """
    # normalize everything to plain JSON-serializable structures
    b_plain = to_plain(boundaries)
    c_plain = to_plain(cmap)
    h_plain = to_plain(H)
    u_plain = to_plain(shapes)
    pol_plain = to_plain(policy_block)

    # read cert content into memory
    cert_bytes = b""
    try:
        with open(cert_path, "rb") as f:
            cert_bytes = f.read()
    except Exception:
        cert_bytes = _json.dumps({"error": "cert not found", "path": cert_path}).encode("utf-8")

    # absolute zip path next to cert
    zip_path = os.path.join(os.path.dirname(cert_path), out_zip)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("boundaries.json", _json.dumps(b_plain, indent=2))
        z.writestr("cmap.json",       _json.dumps(c_plain, indent=2))
        z.writestr("H.json",          _json.dumps(h_plain, indent=2))
        z.writestr("shapes.json",     _json.dumps(u_plain, indent=2))
        z.writestr("policy.json",     _json.dumps(pol_plain, indent=2))
        z.writestr("cert.json",       cert_bytes)

    return zip_path



