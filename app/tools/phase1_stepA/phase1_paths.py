"""phase1_paths.py — minimal path plumbing for Phase‑1 Step‑A verifiers.

This module intentionally contains *no* semantic logic (no canonicalization,
no transport rules, no hashing). It only resolves file locations for the
Phase‑1 verifier scripts so the tools don't get "lost" when the repo's
fixture files live outside the current working directory.

Design goals:
  - Keep CLI surfaces unchanged (no new required flags).
  - Be deterministic: if multiple matches exist, fail fast and print choices.
  - Prefer explicit paths; only search when the given path is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ResolveResult:
    """Result of resolving a user-supplied path string."""

    resolved: Path
    tried: List[Path]


def _find_repo_root(start: Path) -> Path:
    """Best-effort repo-root finder.

    Walk upwards from *start* until we find a directory that looks like a
    project root. This is intentionally heuristic: it's only used for fallback
    search paths.
    """

    start = start.resolve()
    markers = (".git", "pyproject.toml", "requirements.txt", "setup.py")
    for d in (start, *start.parents):
        for m in markers:
            if (d / m).exists():
                return d
        # The streamlit app in this project uses a repo root two levels above
        # itself; if a repo has the app at app/streamlit_app_rigored.py, we can
        # treat the parent of "app" as root.
        if (d / "streamlit_app_rigored.py").exists():
            return d
        if (d / "app" / "streamlit_app_rigored.py").exists():
            return d

    # Heuristic fallback: many repos keep tools under <repo>/tools/...
    # If we are inside a "tools" subtree (or "app"), treat its parent as root.
    for d in (start, *start.parents):
        name = d.name
        if name in ("tools", "app", "src"):
            try:
                return d.parent.resolve()
            except Exception:
                return d.parent

    return start


def _candidate_bases(script_dir: Path, repo_root: Path) -> List[Path]:
    """Ordered base directories to search for fixtures."""

    rr = repo_root
    return [
        # Script-local first (keeps tools self-contained if vendored into tools/)
        script_dir,
        script_dir / "fixtures",
        script_dir / ".." / "fixtures",
        # Repo-level conventional locations
        rr,
        rr / "fixtures",
        rr / "fixtures" / "stepA",
        rr / "fixtures" / "phase1_stepA",
        rr / "fixtures" / "phase1" / "stepA",
        rr / "tools" / "fixtures",
        rr / "tools" / "fixtures" / "stepA",
        rr / "tools" / "fixtures" / "phase1_stepA",
        rr / "data",
        rr / "data" / "stepA",
        rr / "data" / "phase1_stepA",
        rr / "logs" / "fixtures",
        rr / "logs" / "fixtures" / "stepA",
        rr / "logs" / "fixtures" / "phase1_stepA",
        rr / "seed_certs",
        rr / "seed_certs" / "fixtures",
        rr / "certs" / "fixtures",
        rr / "certs" / "fixtures" / "stepA",
        rr / "certs" / "fixtures" / "phase1_stepA",
    ]


def resolve_path(
    raw: str,
    *,
    role: str,
    script_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> ResolveResult:
    """Resolve *raw* to an existing file path.

    - If *raw* already exists (absolute or relative to CWD), returns it.
    - Otherwise searches a deterministic list of candidate fixture bases.

    If multiple matches exist, raises RuntimeError (caller should pass an
    explicit path to disambiguate).
    """

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{role}: empty path")

    raw = raw.strip()
    direct = Path(raw)
    tried: List[Path] = []

    # 1) direct as provided (relative to CWD)
    tried.append(direct)
    if direct.exists():
        return ResolveResult(direct.resolve(), tried)

    # 2) search bases
    sd = (script_dir or Path(__file__).resolve().parent).resolve()
    rr = (repo_root or _find_repo_root(sd)).resolve()
    bases = _candidate_bases(sd, rr)

    hits: List[Path] = []
    for b in bases:
        try:
            cand = (b / raw).resolve()
        except Exception:
            continue
        tried.append(cand)
        if cand.exists():
            hits.append(cand)

    # Unique hit -> OK
    uniq = sorted({p.resolve() for p in hits})
    if len(uniq) == 1:
        return ResolveResult(uniq[0], tried)

    if len(uniq) > 1:
        msg = [f"{role}: ambiguous path {raw!r}; found multiple matches:"]
        msg.extend([f"  - {p}" for p in uniq])
        msg.append("Pass an explicit path to disambiguate.")
        raise RuntimeError("\n".join(msg))

    # No hit
    msg = [f"{role}: file not found: {raw!r}"]
    msg.append("Tried:")
    msg.extend([f"  - {p}" for p in tried[:30]])
    if len(tried) > 30:
        msg.append(f"  ... ({len(tried) - 30} more)")
    raise FileNotFoundError("\n".join(msg))
