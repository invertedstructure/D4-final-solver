
from __future__ import annotations
from typing import Iterable, Tuple, Any
from .io import dump_canonical
import hashlib, datetime, os

APP_VERSION = os.getenv("ODD_TETRA_APP_VERSION", "v0.1-core")

def content_hash_of(obj: Any) -> str:
    return hashlib.sha256(dump_canonical(obj)).hexdigest()

def bundle_content_hash(named_objs: Iterable[Tuple[str, Any]]) -> str:
    h = hashlib.sha256()
    for name, obj in sorted(named_objs, key=lambda x: x[0]):
        h.update(name.encode("utf-8"))
        h.update(dump_canonical(obj))
    return h.hexdigest()

def timestamp_iso_lisbon() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def run_id(content_hash: str, timestamp_iso: str, app_version: str = APP_VERSION) -> str:
    s = f"{content_hash}|{timestamp_iso}|{app_version}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# --- convenience hashes used by the UI/registry --------------------------------
from typing import Any, Dict, List

def _get_blocks(obj: Any) -> Dict[str, List[List[int]]]:
    """
    Return the plain dict {k: matrix} from:
      - a pydantic object with .blocks.__root__
      - a dict with 'blocks'
      - or already a {k: matrix} dict
    """
    if hasattr(obj, "blocks"):
        b = getattr(obj.blocks, "__root__", None)
        if b is not None:
            return b
        # sometimes .blocks can already be a plain dict
        b = getattr(obj, "blocks")
        if isinstance(b, dict):
            return b
    if isinstance(obj, dict):
        if "blocks" in obj:
            b = obj["blocks"]
            if hasattr(b, "__root__"):
                return b.__root__
            if isinstance(b, dict):
                return b
        # assume already {k: matrix}
        return obj
    return {}

def _support_of_blocks(blocks: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    """
    Binary support (0/1) of each matrix modulo 2.
    """
    supp: Dict[str, List[List[int]]] = {}
    for k, M in blocks.items():
        if not M:
            supp[k] = []
            continue
        supp[k] = [[1 if (v % 2) != 0 else 0 for v in row] for row in M]
    return supp

def hash_d(boundaries: Any) -> str:
    """
    Canonical hash of the boundary maps d_k.
    """
    blocks = _get_blocks(boundaries)
    payload = {"d_blocks": blocks}
    return content_hash_of(payload)

def hash_U(shapes: Any) -> str:
    """
    Canonical hash of carrier/mask shapes (whatever structure you pass in).
    If you pass a plain dict, it will be hashed canonically.
    """
    # shapes in your app are often a dict; treat them as-is
    payload = {"U": shapes}
    return content_hash_of(payload)

def hash_suppC(cmap: Any) -> str:
    """
    Canonical hash of the *support* of C (per degree), modulo 2.
    """
    blocks = _get_blocks(cmap)
    supp = _support_of_blocks(blocks)
    payload = {"C_support": supp}
    return content_hash_of(payload)

def hash_suppH(hmap: Any) -> str:
    """
    Canonical hash of the *support* of H (per degree), modulo 2.
    """
