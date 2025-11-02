# == SUITE RETURN NORMALIZER (v2) ==
def _suite_ret_as_tuple(ret):
    try:
        if isinstance(ret, (tuple, list)):
            ok  = bool(ret[0]) if len(ret) >= 1 else False
            msg = str(ret[1]) if len(ret) >= 2 else ""
            n   = ret[2] if len(ret) >= 3 else 0
            try:
                n = int(n)
            except Exception:
                n = 0
            return ok, msg, n
    except Exception:
        pass
    return False, "suite runner returned unexpected shape", 0


# == SUITE RUNNER (v2 — always 3‑tuple) ==
from pathlib import Path as _SPath
import json as _Sjson

def _repo_root():
    try:
        return _SPath(__file__).resolve().parent
    except Exception:
        return _SPath(".").resolve()

def _abs_from_manifest(p: str) -> _SPath:
    P = _SPath(p)
    return P if P.is_absolute() else (_repo_root() / P).resolve()

def run_suite_from_manifest(manifest_path: str, snapshot_id: str):
    mp = _abs_from_manifest(manifest_path)
    if not mp.exists():
        return False, f"Manifest not found: {mp}", 0

    # Read JSONL manifest
    lines = []
    try:
        with mp.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                lines.append(_Sjson.loads(raw))
    except Exception as e:
        return False, f"Bad manifest JSONL: {e}", 0

    # Local solver arity normalizer (2 or 3 values -> ok,msg,bdir)
    def _solver_ret_as_tuple_local(ret):
        try:
            if isinstance(ret, (tuple, list)):
                ok  = bool(ret[0]) if len(ret) >= 1 else False
                msg = str(ret[1]) if len(ret) >= 2 else ""
                bdir = ret[2] if len(ret) >= 3 else None
                return ok, msg, bdir
        except Exception:
            pass
        return False, "solver returned unexpected shape", None

    ok_count = 0
    total = len(lines)

    for i, rec in enumerate(lines, 1):
        fid = rec.get("id") or f"fixture_{i:02d}"
        B, C, H, U = rec["B"], rec["C"], rec["H"], rec["U"]

        # Seed inputs using app helper; fallback to session_state
        try:
            _set_inputs_for_run(B, C, H, U)
        except Exception:
            try:
                import streamlit as _st
                _ss = _st.session_state
                _ss["inputs_selected_B_path"] = str(_abs_from_manifest(B))
                _ss["inputs_selected_C_path"] = str(_abs_from_manifest(C))
                _ss["inputs_selected_H_path"] = str(_abs_from_manifest(H))
                _ss["inputs_selected_U_path"] = str(_abs_from_manifest(U))
            except Exception:
                pass

        # Existence preflight (non-fatal skip for missing)
        missing = [p for p in (B, C, H, U) if not _abs_from_manifest(p).exists()]
        if missing:
            try:
                import streamlit as _st
                _st.warning(f"[{fid}] Missing files: {', '.join(missing)}")
            except Exception:
                pass
            continue

        # Run one-press solver with arity guard
        try:
            ret = run_overlap_once()
        except Exception as e:
            ret = (False, f"solver error: {e}", None)

        ok, msg, _ = _solver_ret_as_tuple_local(ret)

        # UI note (best-effort)
        try:
            import streamlit as _st
            _st.write(f"{fid} → {'ok' if ok else 'fail'} · {msg}")
        except Exception:
            pass

        if ok:
            ok_count += 1

    return True, f"Completed {ok_count}/{total} fixtures.", ok_count

import streamlit as st

# ===== Helper: lanes sig8 + suite message (always defined) =====
def _lanes_sig8_from_list(L):
    import hashlib as _hashlib
    try:
        b = bytearray(); acc = 0; bit = 0
        for v in (int(x) for x in (L or [])):
            if v: acc |= (1 << bit)
            bit += 1
            if bit == 8:
                b.append(acc); acc = 0; bit = 0
        if bit: b.append(acc)
        return _hashlib.sha256(bytes(b)).hexdigest()[:8]
    except Exception:
        return None

def _suite_msg_with_lanes(bundle_dir):
    """Return 'lanes=K · SIG8' from AUTO overlap cert inside bundle_dir; 'unknown' if not available."""
    try:
        import json as _json
        from pathlib import Path as _Path
        bd = _Path(str(bundle_dir))
        meta_p = bd / "bundle.json"
        if not meta_p.exists():
            return "unknown"
        meta = _json.loads(meta_p.read_text(encoding="utf-8"))
        filenames = meta.get("filenames") or []
        auto = None
        for fn in filenames:
            s = str(fn)
            if "projected_columns_k_3_auto" in s:
                auto = bd / s
                break
        if not auto or not auto.exists():
            return "unknown"
        doc = _json.loads(auto.read_text(encoding="utf-8"))
        lanes = None
        w = doc.get("witness")
        if isinstance(w, dict):
            lanes = w.get("lanes")
        if lanes is None:
            pc = doc.get("projection_context") or {}
            if isinstance(pc, dict):
                lp = pc.get("lanes_popcount")
                ls = pc.get("lanes_sig8")
                if isinstance(lp, int) and isinstance(ls, str):
                    return f"lanes={lp} · {ls}"
                lanes = pc.get("lanes")
        if not isinstance(lanes, list):
            return "unknown"
        pop = sum(1 for x in lanes if int(x)==1)
        sig8 = _lanes_sig8_from_list(lanes)
        return f"lanes={pop} · {sig8 if sig8 else 'NA'}"
    except Exception:
        return "unknown"
# ===============================================================
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# === canonical constants / helpers (single source of truth) ===

import os
import json as _json
import hashlib as _hashlib
import uuid as _uuid
import datetime as _datetime
from pathlib import Path
import os, tempfile
import copy as _copy
import json as _json
import hashlib as _hash
import json as _json
import csv as _csv
from pathlib import Path as _Path
import datetime as _dt
import random as _random
import uuid, streamlit as st

# Page config must be the first Streamlit command
SCHEMA_VERSION = "2.0.0"
ENGINE_REV     = "rev-20251022-1"

DIRS = {"root": "logs", "certs": "logs/certs", "snapshots": "logs/snapshots", "reports": "logs/reports", "suite_runs": "logs/suite_runs", "exports": "logs/exports"}
# ---------- Suite helpers (v2) ----------

try:
    _Path
except NameError:
    from pathlib import Path as _Path

def _sha256_hex(b: bytes) -> str:
    import hashlib as _hashlib
    return _hashlib.sha256(b).hexdigest()

def _canonical_json(obj) -> str:
    import json as _json
    return _json.dumps(obj, sort_keys=True, separators=(",",":"))

def _repo_root() -> _Path:
    # repo root = parent of app dir
    return _Path(__file__).resolve().parent.parent



def _svr_current_snapshot_id() -> str | None:
    try:
        snaps_dir = _Path(DIRS.get("snapshots", "logs/snapshots"))
        ptr = snaps_dir / "world_snapshot.latest.json"
        if not ptr.exists():
            return None
        import json as _json
        data = _json.loads(ptr.read_text(encoding="utf-8"))
        return data.get("snapshot_id")
    except Exception:
        return None

def ensure_suite_snapshot(manifest_path: str) -> tuple[bool, str, str | None]:
    """Create a content-addressed world snapshot from a manifest JSONL.
    Writes: logs/snapshots/world_snapshot__{sid}.json and world_snapshot.latest.json
    Returns (ok, message, snapshot_id).
    """
    try:
        mp = _abs_from_manifest(manifest_path)
        if not mp.exists():
            return False, f"Manifest not found: {mp}", None
        import json as _json, hashlib as _hashlib, datetime as _dt
        rows = []
        inventory = []
        seen = {}
        # walk manifest JSONL
        for line in mp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            # resolve absolute paths
            B = str(_abs_from_manifest(rec["B"]))
            C = str(_abs_from_manifest(rec["C"]))
            H = str(_abs_from_manifest(rec["H"]))
            U = str(_abs_from_manifest(rec["U"]))
            fid = rec.get("id") or ""
            rows.append({"B": B, "C": C, "H": H, "U": U, "id": fid})
            for tag, path in (("B",B),("C",C),("H",H),("U",U)):
                if path in seen:
                    continue
                try:
                    h = _sha256_hex(_Path(path).read_bytes())
                except Exception:
                    h = None
                seen[path] = {"tag": tag, "path": path, "sha256": h}
        inventory = list(seen.values())
        # canonical payload (no matrices)
        payload = {
            "schema_version": "2.0.0",
            "engine_rev": ENGINE_REV,
            "created_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "manifests": {"full_scope": str(mp)},
            "inventory": inventory,
            "plan_full_scope": rows,
        }
        sid = "ws__" + _sha256_hex(_canonical_json(payload).encode("utf-8"))[:8]
        payload["snapshot_id"] = sid
        # ensure dirs
        snaps_dir = _Path(DIRS.get("snapshots", "logs/snapshots"))
        snaps_dir.mkdir(parents=True, exist_ok=True)
        # write snapshot & pointer
        _guarded_atomic_write_json(snaps_dir / f"world_snapshot__{sid}.json", payload)
        _guarded_atomic_write_json(snaps_dir / "world_snapshot.latest.json", {"snapshot_id": sid, "path": f"world_snapshot__{sid}.json"})
        return True, f"Snapshot ready: {sid}", sid
    except Exception as e:
        return False, f"Snapshot build failed: {e}", None

def _suite_index_paths():
    base = _Path(DIRS.get("suite_runs", "logs/suite_runs"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "suite_index.jsonl", base / "suite_index.csv"

def _suite_index_add_row(row: dict) -> None:
    # append to jsonl and csv (create csv header if missing)
    import csv as _csv, json as _json
    jl, csvp = _suite_index_paths()
    with jl.open("a", encoding="utf-8") as jf:
        jf.write(_canonical_json(row) + "\n")
    header = list(row.keys())
    write_header = not csvp.exists()
    with csvp.open("a", newline="", encoding="utf-8") as cf:
        w = _csv.DictWriter(cf, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

# ---------- Suite helpers (v2) ----------

from pathlib import Path as _Path
import json as _json, hashlib as _hashlib, time as _time
import streamlit as _st

_APP_DIR  = _Path(__file__).resolve().parent         # .../app
_REPO_DIR = _APP_DIR.parent                          # repo root

def _abs_from_manifest(p_str: str) -> _Path:
    """
    Resolve manifest paths robustly:
      - absolute paths: return as-is
      - 'app/…'        : resolve from REPO root
      - everything else: resolve from APP dir
    """
    p = _Path(p_str)
    if p.is_absolute():
        return p
    s = str(p).replace("\\", "/").lstrip("./")
    if s.startswith("app/"):
        return (_REPO_DIR / s).resolve()
    return (_APP_DIR / s).resolve()

def _set_inputs_for_run(B, C, H, U):
    """
    Prime the single-press pipeline with file paths the same way
    the upload widgets would. We set both legacy and current keys
    to be safe.
    """
    ss = _st.session_state
    Bp, Cp, Hp, Up = map(lambda q: str(_abs_from_manifest(q)), (B, C, H, U))

    # Primary keys the resolver uses
    ss["uploaded_boundaries"] = Bp
    ss["uploaded_cmap"]       = Cp
    ss["uploaded_H"]          = Hp
    ss["uploaded_shapes"]     = Up

    # Common synonyms seen across earlier wiring
    ss["uploaded_B_path"] = Bp
    ss["uploaded_C_path"] = Cp
    ss["uploaded_H_path"] = Hp
    ss["uploaded_U_path"] = Up
