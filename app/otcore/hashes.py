
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
