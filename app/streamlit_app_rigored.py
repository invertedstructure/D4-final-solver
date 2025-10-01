# --- robust loader with real package context (supports app/otcore or app/core) ---
import sys, pathlib, importlib.util, types
import streamlit as st
import json
import json as _json
import hashlib as _hashlib

# Streamlit MUST be configured before ANY other st.* call:
st.set_page_config(page_title="Odd Tetra App (v0.1)", layout="wide")

# --- Policy helpers -----------------------------------------------------------
def cfg_strict():
    # strict = no projection anywhere
    return {"enabled_layers": [], "modes": {}, "source": {}, "projector_files": {}}

def cfg_projected_base():
    # default projected: columns @ k=3, auto source
    return {
        "enabled_layers": [3],
        "modes": {"3": "columns"},
        "source": {"3": "auto"},
        "projector_files": {"3": "projector_D3.json"},
    }

def policy_label_from_cfg(cfg: dict) -> str:
    if not cfg or not cfg.get("enabled_layers"):
        return "strict"
    parts = []
    for kk in sorted(cfg["enabled_layers"]):
        mode = cfg.get("modes", {}).get(str(kk), "none")
        src  = cfg.get("source", {}).get(str(kk), "auto")
        parts.append(f"{mode}@k={kk},{src}")
    return "projected(" + "; ".join(parts) + ")"

# --- cert writer (save one result to certs/...) -------------------------------
from pathlib import Path
import json as _json

def _short(s: str, n: int = 12) -> str:
    return s[:n] if s else ""

def policy_tag_for_filename(label: str) -> str:
    # turn "projected(columns@k=3,auto)" into "projected_columns_k3_auto"
    return (
        label.replace("projected(", "projected_")
             .replace(")", "")
             .replace("@", "_")
             .replace(";", "_")
             .replace(",", "_")
             .replace("=", "")
             .replace(" ", "")
    )

def write_overlap_cert(*, out: dict, policy_label: str, boundaries, cmap, H, pj_hash: str | None = None, cert_dir: str = "certs") -> str:
    Path(cert_dir).mkdir(exist_ok=True)
    payload = {
        "policy": policy_label,
        "k2": out.get("2", {}),
        "k3": out.get("3", {}),
        "hashes": {
            "hash_d": hashes.hash_d(boundaries),
            "hash_U": hashes.hash_U(globals().get("shapes")) if "shapes" in globals() else "",
            "hash_suppC": hashes.hash_suppC(cmap),
            "hash_suppH": hashes.hash_suppH(H),
            "hash_P": pj_hash or "",
        },
        "app": {
            "version": getattr(hashes, "APP_VERSION", "v0.1-core"),
            "run_id": hashes.run_id(
                content_hash := hashes.bundle_content_hash([
                    ("d", boundaries.dict() if hasattr(boundaries, "dict") else {}),
                    ("C", cmap.dict() if hasattr(cmap, "dict") else {}),
                    ("H", H.dict() if hasattr(H, "dict") else {}),
                ]),
                hashes.timestamp_iso_lisbon(),
            ),
            "content_hash": content_hash,
        },
    }
    fname = f"overlap_pass__{policy_tag_for_filename(policy_label)}__{_short(payload['app']['run_id'])}.json"
    fpath = str(Path(cert_dir) / fname)
    with open(fpath, "w") as f:
        _json.dump(payload, f, indent=2)
    return fpath



# 1) Locate package dir and set PKG_NAME
HERE = pathlib.Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
CORE   = HERE / "core"
PKG_DIR = OTCORE if OTCORE.exists() else CORE
PKG_NAME = "otcore" if OTCORE.exists() else "core"

# Create a lightweight package object so relative imports inside modules work
if PKG_NAME not in sys.modules:
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PKG_DIR)]
    pkg.__file__ = str(PKG_DIR / "__init__.py")
    sys.modules[PKG_NAME] = pkg

# 2) Minimal loader that loads modules from PKG_DIR by filename
def _load_pkg_module(fullname: str, rel_path: str):
    path = PKG_DIR / rel_path
    if not path.exists():
        raise ImportError(f"Required module file not found: {path}")
    spec = importlib.util.spec_from_file_location(fullname, str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = fullname.rsplit('.', 1)[0]
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# 3) Force fresh imports of overlap_gate/projector from the package on disk
import importlib
for _mod in (f"{PKG_NAME}.overlap_gate", f"{PKG_NAME}.projector"):
    if _mod in sys.modules:
        del sys.modules[_mod]

overlap_gate = _load_pkg_module(f"{PKG_NAME}.overlap_gate", "overlap_gate.py")
projector    = _load_pkg_module(f"{PKG_NAME}.projector",    "projector.py")

# 4) Load the rest of your modules from the same package
io            = _load_pkg_module(f"{PKG_NAME}.io",            "io.py")
hashes        = _load_pkg_module(f"{PKG_NAME}.hashes",        "hashes.py")
unit_gate     = _load_pkg_module(f"{PKG_NAME}.unit_gate",     "unit_gate.py")
triangle_gate = _load_pkg_module(f"{PKG_NAME}.triangle_gate", "triangle_gate.py")
towers        = _load_pkg_module(f"{PKG_NAME}.towers",        "towers.py")
export_mod    = _load_pkg_module(f"{PKG_NAME}.export",        "export.py")

APP_VERSION = getattr(hashes, "APP_VERSION", "v0.1-core")
# -----------------------------------------------------------------------------


# (After set_page_config you can safely use other st.* calls)
st.title("Odd Tetra — Phase U (v0.1 core)")
st.caption("Schemas + deterministic hashes + timestamped run IDs + Gates + Towers")

# Optional debug: show exactly which files were loaded
st.caption(f"overlap_gate loaded from: {getattr(overlap_gate, '__file__', '<none>')}")
st.caption(f"projector loaded from: {getattr(projector, '__file__', '<none>')}")

def read_json_file(file):
    if not file: return None
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return None

with st.sidebar:
    st.markdown("### Upload core inputs")
    st.caption("**Shapes (required):**\\n\\n```json\\n{\\\"n\\\": {\\\"3\\\":3, \\\"2\\\":2, \\\"1\\\":0}}\\n```\\n\\n**Boundaries (required):**\\n\\n```json\\n{\\\"blocks\\\": {\\\"3\\\": [[...]], \\\"2\\\": [[...]]}}\\n```\\n\\n**CMap / Move (required):**\\n\\n```json\\n{\\\"blocks\\\": {\\\"3\\\": [[...]], \\\"2\\\": [[...]]}}\\n```\\n\\n**Support (optional):** either `{degree: mask}` or `{\\\"masks\\\": {degree: mask}}`.\\n\\n**Triangle schema (optional):** degree-keyed `{ \\\"2\\\": {\\\"A\\\":..., \\\"B\\\":..., \\\"J\\\":...}, ... }`.")
    f_shapes = st.file_uploader("Shapes (shapes.json)", type=["json"], key="shapes")
    f_bound = st.file_uploader("Boundaries (boundaries.json)", type=["json"], key="bound")
    f_cmap = st.file_uploader("CMap / Move (Cmap_*.json)", type=["json"], key="cmap")
    f_support = st.file_uploader("Support policy (support_ck_full.json)", type=["json"], key="support")
    f_pair = st.file_uploader("Pairings (pairings.json)", type=["json"], key="pair")
    f_reps = st.file_uploader("Reps (reps_for_Cmap_chain_pairing_ok.json)", type=["json"], key="reps")
    f_triangle = st.file_uploader("Triangle schema (triangle_J_schema.json)", type=["json"], key="tri")
    seed = st.text_input("Seed", "super-seed-A")

d_shapes = read_json_file(f_shapes)
d_bound = read_json_file(f_bound)
d_cmap  = read_json_file(f_cmap)

if d_shapes and d_bound and d_cmap:
    try:
        shapes = io.parse_shapes(d_shapes)
        boundaries = io.parse_boundaries(d_bound)
        cmap = io.parse_cmap(d_cmap)  # must have top-level "blocks"
        support = io.parse_support(read_json_file(f_support)) if f_support else None
        triangle = io.parse_triangle_schema(read_json_file(f_triangle)) if f_triangle else None
        io.validate_bundle(boundaries, shapes, cmap, support)
        st.success("Core schemas validated ✅")
        with st.expander("Hashes / provenance"):
            named = [("boundaries", boundaries.dict()), ("shapes", shapes.dict()), ("cmap", cmap.dict())]
            if support: named.append(("support", support.dict()))
            if triangle: named.append(("triangle", triangle.dict()))
            ch = hashes.bundle_content_hash(named)
            ts = hashes.timestamp_iso_lisbon()
            rid = hashes.run_id(ch, ts)
            st.code(f"content_hash = {ch}\\nrun_timestamp = {ts}\\nrun_id = {rid}\\napp_version = {APP_VERSION}", language="bash")
            # Quick export here too
            if st.button("Export ./reports → report.zip (quick)"):
                import pathlib as _pl
                reports_dir = _pl.Path("reports")
                if not reports_dir.exists():
                    st.warning("No ./reports yet. Run a Tower or Manifest first.")
                else:
                    zpath = reports_dir / "report.zip"
                    export_mod.zip_report(str(reports_dir), str(zpath))
                    st.success(f"Exported: {zpath}")
                    with open(zpath, "rb") as fz:
                        st.download_button("Download report.zip", fz, file_name="report.zip")
    except Exception as e:
        st.error(f"Validation error: {e}")
        st.stop()
else:
    missing = [name for name, f in [("Shapes", d_shapes), ("Boundaries", d_bound), ("CMap", d_cmap)] if not f]
    st.info("Upload required files: " + ", ".join(missing))
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Unit", "Overlap", "Triangle", "Towers", "Export"])

with tab1:
    st.subheader("Unit gate")
    enforce = st.checkbox("Enforce rep transport (c_cod = C c_dom)", value=False)
    d_reps  = read_json_file(f_reps) if f_reps else None
    if st.button("Run Unit"):
        out = unit_gate.unit_check(boundaries, cmap, shapes, reps=d_reps, enforce_rep_transport=enforce)
        st.json(out)

# --- run overlap under a given cfg (strict or projected) ----------------------
def run_overlap_with_cfg(boundaries, cmap, H, cfg: dict):
    cache = projector.preload_projectors_from_files(cfg)
    try:
        out = overlap_gate.overlap_check(
            boundaries, cmap, H,
            projection_config=cfg,
            projector_cache=cache,
        )
    except TypeError:
        # old signature → strict fallback
        out = overlap_gate.overlap_check(boundaries, cmap, H)
    return out, cache


# Overlap gate (homotopy vs identity)
with tab2:
    st.subheader("Overlap gate (homotopy vs identity)")
    f_H = st.file_uploader("Homotopy H (H_corrected.json)", type=["json"], key="H_corr")
    d_H = read_json_file(f_H) if f_H else None

# --- Policy toggle UI ---------------------------------------------------------
st.markdown("### Policy")
policy_choice = st.radio(
    "Choose policy",
    ["strict", "projected(columns@k=3)"],
    horizontal=True,
    key="policy_choice_k3",
)

# Build cfg based on choice + current file/auto setting (if any)
cfg_file = projector.load_projection_config("projection_config.json")
cfg_proj = cfg_projected_base()
# keep your existing file/auto decision if present
if cfg_file.get("source", {}).get("3") in ("file", "auto"):
    cfg_proj["source"]["3"] = cfg_file["source"]["3"]
if "projector_files" in cfg_file and "3" in cfg_file["projector_files"]:
    cfg_proj["projector_files"]["3"] = cfg_file["projector_files"]["3"]

cfg_active = cfg_strict() if policy_choice == "strict" else cfg_proj
policy_label = policy_label_from_cfg(cfg_active)
st.caption(f"Policy: **{policy_label}**")

cache = projector.preload_projectors_from_files(cfg_active)

# --- one-run helper -----------------------------------------------------------
def run_overlap_with_cfg(cfg_run: dict):
    try:
        return overlap_gate.overlap_check(
            boundaries, cmap, H,
            projection_config=cfg_run,
            projector_cache=cache
        )
    except TypeError:
        # fallback for old signature
        return overlap_gate.overlap_check(boundaries, cmap, H)

    
    if st.button("Run Overlap"):
        if not d_H:
            st.error("Upload H_corrected.json")
        else:
            # 1) Parse H
            H = io.parse_cmap(d_H)

            # 2) Load projection config + cache
            cfg = projector.load_projection_config("projection_config.json")
            policy_label = policy_label_from_cfg(cfg)
            st.caption(f"Policy: {policy_label}")
            cache = projector.preload_projectors_from_files(cfg)

            # --- A/B Compare controls -----------------------------------------------------
st.markdown("### A/B compare (strict vs projected)")
colA, colB = st.columns(2)
with colA:
    if st.button("Run A/B compare"):
        # A: strict
        cfg_str = cfg_strict()
        out_strict, _ = run_overlap_with_cfg(boundaries, cmap, H, cfg_str)
        # B: projected (use current cfg)
        out_proj, _ = run_overlap_with_cfg(boundaries, cmap, H, cfg)

       # --- optional: write both certs from A/B ---
if st.checkbox("Write both certs (strict & projected)", value=False):
    cert_s = write_overlap_cert(
        out=out_strict,
        policy_label=policy_label_from_cfg(cfg_strict()),
        boundaries=boundaries,
        cmap=cmap,
        H=H,
        pj_hash=None,
    )
    # projector hash for projected path (file-mode only)
    pj_hash_proj = ""
    if cfg_proj.get("source", {}).get("3") == "file":
        pj_path = cfg_proj.get("projector_files", {}).get("3")
        if pj_path and os.path.exists(pj_path):
            pj_hash_proj = projector._hash_matrix(_json.load(open(pj_path)))

    cert_p = write_overlap_cert(
        out=out_proj,
        policy_label=policy_label_from_cfg(cfg_proj),
        boundaries=boundaries,
        cmap=cmap,
        H=H,
        pj_hash=pj_hash_proj,
    )
    st.success(f"Saved: `{cert_s}` and `{cert_p}`")

    # --- optional: log both rows to registry ---
    if st.checkbox("Also log both to registry.csv", value=False):
        import time as _time

        # row 1: strict
        try:
            export_mod.write_registry_row(
                fix_id=f"compare-strict-{int(_time.time())}",
                pass_vector=[
                    int(out_strict.get("2", {}).get("eq", False)),
                    int(out_strict.get("3", {}).get("eq", False)),
                ],
                policy=policy_label_from_cfg(cfg_strict()),
                hash_d=hashes.hash_d(boundaries),
                hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                hash_suppC=hashes.hash_suppC(cmap),
                hash_suppH=hashes.hash_suppH(H),
                notes="A/B compare strict",
            )
            st.toast("registry: added strict row")
        except Exception as e:
            st.error(f"registry(strict) failed: {e}")

        # row 2: projected
        try:
            export_mod.write_registry_row(
                fix_id=f"compare-proj-{int(_time.time())}",
                pass_vector=[
                    int(out_proj.get("2", {}).get("eq", False)),
                    int(out_proj.get("3", {}).get("eq", False)),
                ],
                policy=policy_label_from_cfg(cfg_proj),
                hash_d=hashes.hash_d(boundaries),
                hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                hash_suppC=hashes.hash_suppC(cmap),
                hash_suppH=hashes.hash_suppH(H),
                notes="A/B compare projected",
            )
            st.toast("registry: added projected row")
        except Exception as e:
            st.error(f"registry(projected) failed: {e}")




        # Show side-by-side verdicts
        klist = sorted(set(out_strict.keys()) | set(out_proj.keys()), key=int)
        rows = []
        for k in klist:
            es = bool(out_strict.get(k, {}).get("eq", False))
            ep = bool(out_proj.get(k, {}).get("eq", False))
            rows.append((k, es, ep, int(ep) - int(es)))
        st.write("k | eq(strict) | eq(projected) | Δ")
        for k, es, ep, d in rows:
            st.write(f"{k} | {int(es)} | {int(ep)} | {d}")

        # Small hint: if different, show masks/supports
        if out_proj != out_strict:
            d3 = boundaries.blocks.__root__.get("3")
            if d3:
                lane_mask = [1 if any(row[j] for row in d3) else 0 for j in range(len(d3[0]))]
                st.caption(f"k=3 lane_mask = {lane_mask}")

with colB:
    st.info("Tip: use this to confirm ker-only residuals flip from red→green in projected mode.")


    
            # 3) Projector source expander (needs cfg)
            with st.expander("Projector source (k=3)"):
                cur_src  = cfg.get("source", {}).get("3", "auto")
                cur_file = cfg.get("projector_files", {}).get("3", "projector_D3.json")
                st.write(
                    f"Current: source.3 = **{cur_src}**",
                    f"(file: `{cur_file}`)" if cur_src == "file" else ""
                )

                mode_choice = st.radio(
                    "Choose source for k=3",
                    options=["auto", "file"],
                    index=0 if cur_src == "auto" else 1,
                    horizontal=True,
                    key="proj_src_choice_k3",
                )
                file_path = st.text_input(
                    "Projector file", value=cur_file, disabled=(mode_choice == "auto")
                )

                if st.button("Apply projector source"):
                    cfg.setdefault("source", {})["3"] = mode_choice
                    if mode_choice == "file":
                        cfg.setdefault("projector_files", {})["3"] = file_path
                    else:
                        if "projector_files" in cfg and "3" in cfg["projector_files"]:
                            del cfg["projector_files"]["3"]
                    with open("projection_config.json", "w") as _f:
                        _json.dump(cfg, _f, indent=2)
                    st.success(f"projection_config.json updated → source.3 = {mode_choice}")

                # Optional drift guard
                if cur_src == "file" and st.button("Validate file vs auto Π3"):
                    d3_now = boundaries.blocks.__root__.get("3")
                    if d3_now is None:
                        st.error("No d3 in boundaries; cannot validate.")
                    else:
                        autoP = projector.projector_columns_from_dkp1(d3_now)
                        try:
                            with open(cur_file, "r") as _pf:
                                fileP = _json.load(_pf)
                        except Exception as e:
                            st.error(f"Could not load {cur_file}: {e}")
                            fileP = None

                        if fileP is not None:
                            h_auto = _hashlib.sha256(_json.dumps(autoP, sort_keys=True).encode()).hexdigest()
                            h_file = _hashlib.sha256(_json.dumps(fileP, sort_keys=True).encode()).hexdigest()
                            if h_auto == h_file:
                                st.success(f"OK: projector matches auto (hash={h_auto[:12]}…)")
                            else:
                                st.warning(f"DRIFT: file {cur_file} hash={h_file[:12]}… vs auto hash={h_auto[:12]}…")

            # 4) Run overlap (produces `out`)
            try:
                out = overlap_gate.overlap_check(
                    boundaries, cmap, H,
                    projection_config=cfg,
                    projector_cache=cache
                )
            except TypeError as e:
                st.warning(
                    "overlap_gate is running in STRICT mode (old module signature) — "
                    "hard-restart the app after patching overlap_gate.py. "
                    f"TypeError: {e}"
                )
                out = overlap_gate.overlap_check(boundaries, cmap, H)

            st.json(out)

# --- optional: save a cert for this single run ---
pj_hash = ""
# if we’re in projected + FILE mode, include the projector hash if present in cache
k = "3"
if cfg.get("source", {}).get(k) == "file":
    pj_path = cfg.get("projector_files", {}).get(k)
    if pj_path and os.path.exists(pj_path):
        # reuse projector hashing to avoid dupes
        try:
            pj_hash = projector._hash_matrix(_json.load(open(pj_path)))
        except Exception:
            pj_hash = ""

if st.checkbox("Write cert for this run", value=False):
    cert_path = write_overlap_cert(out=out, policy_label=policy_label, boundaries=boundaries, cmap=cmap, H=H, pj_hash=pj_hash)
    st.success(f"Saved cert → `{cert_path}`")


# ---- Build pass-vector & promotion ------------------------------------------
pass_vec = [
    int(out.get("2", {}).get("eq", False)),
    int(out.get("3", {}).get("eq", False)),
]
all_green = all(v == 1 for v in pass_vec)

policy_label = policy_label_from_cfg(cfg)
st.caption(f"Policy: {policy_label}")

if all_green:
    st.success("Green — eligible for promotion.")
    if policy_label == "strict":
        # Strict promotion: just log the strict anchor
        if st.button("Promote (strict anchor)"):
            import time as _time
            fix_id = f"overlap-{int(_time.time())}"
            try:
                export_mod.write_registry_row(
                    fix_id=fix_id,
                    pass_vector=pass_vec,
                    policy=policy_label,  # "strict"
                    hash_d=hashes.hash_d(boundaries),
                    hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                    hash_suppC=hashes.hash_suppC(cmap),
                    hash_suppH=hashes.hash_suppH(H),
                    notes=""
                )
                st.success("Registry updated (strict anchor).")
            except Exception as e:
                st.error(f"Failed to write registry row: {e}")

    else:
        # Projected promotion: freeze projector + log hash, with auto/file toggle
        flip_to_file = st.checkbox("After promotion, switch to FILE-backed projector", value=True, key="flip_to_file_k3")
        keep_auto   = st.checkbox("…or keep AUTO (don’t lock now)", value=False, key="keep_auto_k3")

        if st.button("Promote & Freeze Projector"):
            d3_now = boundaries.blocks.__root__.get("3")
            if d3_now is None:
                st.error("No d3 in boundaries; cannot freeze projector.")
            else:
                P_used = projector.projector_columns_from_dkp1(d3_now)
                pj_path = cfg.get("projector_files", {}).get("3", "projector_D3.json")
                pj_hash = projector.save_projector(pj_path, P_used)
                st.info(f"Projector frozen → {pj_path} (hash={pj_hash[:12]}…)")

                # flip config per user choice
                import json as _json
                if flip_to_file and not keep_auto:
                    cfg.setdefault("source", {})["3"] = "file"
                    cfg.setdefault("projector_files", {})["3"] = pj_path
                    with open("projection_config.json", "w") as _f:
                        _json.dump(cfg, _f, indent=2)
                    st.toast("projection_config.json → FILE-backed (k=3)")
                else:
                    cfg.setdefault("source", {})["3"] = "auto"
                    if "projector_files" in cfg and "3" in cfg["projector_files"]:
                        del cfg["projector_files"]["3"]
                    with open("projection_config.json", "w") as _f:
                        _json.dump(cfg, _f, indent=2)
                    st.toast("projection_config.json → AUTO (k=3)")

                # registry row with projector hash
                import time as _time
                fix_id = f"overlap-{int(_time.time())}"
                try:
                    export_mod.write_registry_row(
                        fix_id=fix_id,
                        pass_vector=pass_vec,
                        policy=policy_label,  # projected(...)
                        hash_d=hashes.hash_d(boundaries),
                        hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                        hash_suppC=hashes.hash_suppC(cmap),
                        hash_suppH=hashes.hash_suppH(H),
                        notes=f"proj_hash={pj_hash}"
                    )
                    st.success("Registry updated with projector hash.")
                except Exception as e:
                    st.error(f"Failed to write registry row: {e}")
else:
    st.info("Not promoting: some checks are red.")



            # 5) Build pass-vector NOW that `out` exists
            pass_vec = [
                int(out.get("2", {}).get("eq", False)),
                int(out.get("3", {}).get("eq", False)),
            ]
            all_green = all(v == 1 for v in pass_vec)

            # 6) Promotion: freeze projector + optional auto→file flip
            if all_green:
                st.success("Green — eligible for promotion.")
                flip_to_file = st.checkbox("After promotion, switch to FILE-backed projector", value=True, key="flip_to_file_k3")
                force_back_to_auto = st.checkbox("…or keep AUTO (don’t lock now)", value=False, key="keep_auto_k3")

                if st.button("Promote & Freeze Projector"):
                    d3_now = boundaries.blocks.__root__.get("3")
                    if d3_now is None:
                        st.error("No d3 in boundaries; cannot freeze projector.")
                    else:
                        P_used = projector.projector_columns_from_dkp1(d3_now)
                        pj_path = cfg.get("projector_files", {}).get("3", "projector_D3.json")
                        pj_hash = projector.save_projector(pj_path, P_used)
                        st.info(f"Projector frozen → {pj_path} (hash={pj_hash[:12]}…)")

                        # config flip
                        if flip_to_file and not force_back_to_auto:
                            cfg.setdefault("source", {})["3"] = "file"
                            cfg.setdefault("projector_files", {})["3"] = pj_path
                        else:
                            cfg.setdefault("source", {})["3"] = "auto"
                            if "projector_files" in cfg and "3" in cfg["projector_files"]:
                                del cfg["projector_files"]["3"]
                        with open("projection_config.json", "w") as _f:
                            _json.dump(cfg, _f, indent=2)

                        # registry with projector hash
                        import time as _time
                        fix_id = f"overlap-{int(_time.time())}"
                        try:
                            export_mod.write_registry_row(
                                fix_id=fix_id,
                                pass_vector=pass_vec,
                                policy=policy_label,
                                hash_d=hashes.hash_d(boundaries),
                                hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                                hash_suppC=hashes.hash_suppC(cmap),
                                hash_suppH=hashes.hash_suppH(H),
                                notes=f"proj_hash={pj_hash}"
                            )
                            st.success("Registry updated with projector hash.")
                        except Exception as e:
                            st.error(f"Failed to write registry row: {e}")
            else:
                st.info("Not promoting: some checks are red.")

            # 7) Normal registry write (every run)
            import time
            fix_id = f"overlap-{int(time.time())}"
            try:
                export_mod.write_registry_row(
                    fix_id=fix_id,
                    pass_vector=pass_vec,
                    policy=policy_label,
                    hash_d=hashes.hash_d(boundaries),
                    hash_U=hashes.hash_U(shapes) if 'shapes' in locals() else "",
                    hash_suppC=hashes.hash_suppC(cmap),
                    hash_suppH=hashes.hash_suppH(H),
                    notes=""
                )
                st.success("Registry updated (registry.csv).")
            except Exception as e:
                st.error(f"Failed to write registry row: {e}")



with tab4:
    st.subheader("Towers")
    sched_str = st.text_input("Schedule (comma-separated I/C)", "I,C,C,I,C")
    sched = [s.strip().upper() for s in sched_str.split(",") if s.strip()]
    if any(s not in ("I","C") for s in sched):
        st.error("Schedule must contain only I or C")
    else:
        if st.button("Run Tower & save CSV"):
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            csv_path = os.path.join(reports_dir, f"tower-hashes_{seed}_{len(sched)}steps.csv")
            towers.run_tower(sched, cmap, shapes, seed, csv_path, schedule_name="custom")
            st.success(f"Saved: {csv_path}")
            with open(csv_path, "r", encoding="utf-8") as f:
                st.download_button("Download CSV", f.read(), file_name=os.path.basename(csv_path), mime="text/csv")

with tab5:
    st.subheader("Export")
    st.caption("Bundle all artifacts in ./reports into a single ZIP for sharing/archival.")
    if st.button("Export ./reports → report.zip"):
        reports_dir = pathlib.Path("reports")
        if not reports_dir.exists():
            st.warning("No ./reports directory yet. Run a Tower or Manifest first.")
        else:
            zpath = reports_dir / "report.zip"
            export_mod.zip_report(str(reports_dir), str(zpath))
            st.success(f"Exported: {zpath}")
            with open(zpath, "rb") as fz:
                st.download_button("Download report.zip", fz, file_name="report.zip")
