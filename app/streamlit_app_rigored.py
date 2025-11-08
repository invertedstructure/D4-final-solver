
# -*- coding: utf-8 -*-
# HARD V2 STREAMLIT RUNNER WITH COVERAGE (Option A)
# - Deterministic 1× compute-only writer (6 core + bundle.json + loop_receipt)
# - Coverage row appended per 1× (after materialization) with strict gates
# - Suite-level sidecars intentionally omitted here (suite harness will own them)

import os as _os, json as _json, time as _time, uuid as _uuid, hashlib as _hashlib
from pathlib import Path as _Ph

# --------------------- Tiny utils ---------------------
def _shape(A):
    if not A: return (0,0)
    return (len(A), len(A[0]) if A and isinstance(A[0], (list, tuple)) else 0)

def _eye(n):
    return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

def _mm2(A,B):
    m,k = _shape(A); k2,n = _shape(B)
    if k != k2:
        raise ValueError("shape mismatch")
    R = [[0]*n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for j in range(n):
            s = 0
            for t in range(k):
                s ^= (Ai[t] & B[t][j])
            R[i][j] = s & 1
    return R

def _xor(A,B):
    m,n = _shape(A); m2,n2 = _shape(B)
    if m!=m2 or n!=n2:
        raise ValueError("shape mismatch")
    return [[(A[i][j]^B[i][j]) for j in range(n)] for i in range(m)]

def _zero_mask_cols(M):
    m,n = _shape(M); z=[1]*n
    for i in range(m):
        row=M[i]
        for j in range(n):
            if row[j]==1: z[j]=0
    return z

def _support_mask(M):
    z = _zero_mask_cols(M)
    return [0 if z[j] else 1 for j in range(len(z))]

def _bitsig256(bits):
    return _hashlib.sha256(_json.dumps([1 if b else 0 for b in (bits or [])], separators=(",",":"), sort_keys=False).encode("utf-8")).hexdigest()

def _hash8(obj):
    return _hashlib.sha256(_json.dumps(obj, sort_keys=True, separators=(",",":")).encode("utf-8")).hexdigest()[:8]

def _write_json(p: _Ph, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(_json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    _os.replace(tmp, p)

def _append_jsonl(p: _Ph, row: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(_json.dumps(row, ensure_ascii=False, separators=(",",":")) + "\n")

def _as01(M):
    return [[1 if (int(x) & 1) else 0 for x in r] for r in (M or [])]

# --------------------- Fixtures & paths ---------------------
def _extract_mats(pb):
    def _blocks(x):
        if isinstance(x, (list,tuple)) and len(x)>=2:
            return x[1] or {}
        return {}
    bB=_blocks(pb.get("B")); bC=_blocks(pb.get("C")); bH=_blocks(pb.get("H"))
    H2=_as01(bH.get("2") or []); d3=_as01(bB.get("3") or []); C3=_as01(bC.get("3") or [])
    return H2,d3,C3

def _fixture_tuple_from_paths(pB,pH,pC):
    import os,re
    bname = os.path.basename(os.fspath(pB or ""))
    if "D2" in bname: D="D2"
    elif "D3" in bname: D="D3"
    else: D="UNKNOWN_DISTRICT"
    hname = os.path.basename(os.fspath(pH or ""))
    mH = re.search(r"[Hh]([01]{2})", hname); H=f"H{mH.group(1)}" if mH else "H??"
    cname = os.path.basename(os.fspath(pC or ""))
    mC = re.search(r"[Cc]([01]{3})", cname); C=f"C{mC.group(1)}" if mC else "C???"
    return D,H,C,f"{D}_{H}_{C}"

def _bundle_dir(district_id, fixture_label, sig8):
    return _Ph("logs/certs")/str(district_id)/str(fixture_label)/str(sig8)

# --------------------- Core 1× compute-only ---------------------
def _svr_run_once_computeonly_hard(ss=None):
    g = globals()
    resolver = g.get("_svr_resolve_all_to_paths") or g.get("resolve_all_to_paths")
    freezer  = g.get("_svr_freeze_ssot") or g.get("freeze_ssot")

    # Fallback resolvers if not provided by host app
    if resolver is None:
        def resolver():
            B = (ss or {}).get("_last_inputs_paths",{}).get("B","")
            C = (ss or {}).get("_last_inputs_paths",{}).get("C","")
            H = (ss or {}).get("_last_inputs_paths",{}).get("H","")
            U = (ss or {}).get("_last_inputs_paths",{}).get("U","")
            def _load(p):
                try:
                    return _json.loads(_Ph(p).read_text(encoding="utf-8")) if p else []
                except Exception:
                    return []
            return {"B": (B, {"3": _load(B)}),
                    "C": (C, {"3": _load(C)}),
                    "H": (H, {"2": _load(H)}),
                    "U": (U, {})}
    if freezer is None:
        def freezer(pb): return {}, {}

    pb = resolver() or {}

    def _first(x):
        if isinstance(x, (list,tuple)) and x: return x[0]
        return x
    pB=_first(pb.get("B")); pC=_first(pb.get("C")); pH=_first(pb.get("H")); pU=_first(pb.get("U"))

    try:
        import streamlit as _st
        if ss is None: ss=_st.session_state
    except Exception:
        ss = ss or {}

    ib, rc = freezer(pb); ib=dict(ib or {}); rc=dict(rc or {})

    H2,d3,C3 = _extract_mats(pb)
    mC,nC = _shape(C3)
    _,n3  = _shape(d3)

    CxorI = _xor(C3, _eye(nC)) if (mC==nC) else C3
    Hd    = _mm2(H2,d3) if (_shape(H2)[1]==_shape(d3)[0]) else [[0]*n3]
    R3    = _xor(Hd, CxorI) if CxorI else Hd
    suppR = _support_mask(R3)
    ker   = _zero_mask_cols(d3)

    D,Htag,Ctag,fixture_label = _fixture_tuple_from_paths(pB,pH,pC)
    district_id = rc.get("district_id") or f"{D}{_hash8({'d3':d3})}"
    sig8        = rc.get("sig8") or (rc.get("embed_sig","")[:8] if rc.get("embed_sig") else _hash8({"H2":H2,"d3":d3,"C3":C3}))
    snapshot_id = rc.get("snapshot_id") or (ss.get("world_snapshot_id") if isinstance(ss,dict) else None) or ""
    fixtures    = {"district": D, "H": Htag, "C": Ctag, "U": "U"}

    strict_eq = (sum(suppR)==0) if R3 else False
    strict = {
      "policy_tag": "strict(k=3)",
      "results": {"k3":{"eq": strict_eq}},
      "metrics": {"R3_failing_cols_popcount": int(sum(suppR)), "ker_cols_popcount": int(sum(ker))},
      "strict_failing_cols_sig256": _bitsig256(suppR),
      "ker_mask_sig256": _bitsig256(ker),
      "fixtures": fixtures,
      "fixture_label": fixture_label,
      "snapshot_id": snapshot_id,
      "sig8": sig8
    }

    # AUTO
    if not (mC==nC):
        auto = {"policy_tag":"projected(columns@k=3,auto)","results":{"k3":{"eq":None},"selected_cols":[]},
                "na_reason_code":"C3_NON_SQUARE","ker_mask_sig256":_bitsig256(ker),
                "strict_failing_cols_sig256":_bitsig256(suppR)}
    else:
        lanes = list(C3[-1] if mC>0 else [0]*n3)
        if sum(lanes)==0:
            auto = {"policy_tag":"projected(columns@k=3,auto)","results":{"k3":{"eq":None},"selected_cols":lanes},
                    "na_reason_code":"LANES_ZERO","lanes_sig256":_bitsig256(lanes),
                    "ker_mask_sig256":_bitsig256(ker),"strict_failing_cols_sig256":_bitsig256(suppR)}
        else:
            proj_fail = [1 if (lanes[j]==1 and suppR[j]==1) else 0 for j in range(len(suppR))]
            proj_eq   = (sum(proj_fail)==0)
            # verdict_class (algebra only)
            def _subset(A,B): return all((a==0) or (b==1) for a,b in zip(A,B))
            vclass = "GREEN" if strict_eq else ("KER-EXPOSED" if _subset(suppR,ker) and any(lanes[j] and suppR[j] for j in range(len(lanes)))
                                                else ("KER-FILTERED" if _subset(suppR,ker) else "RED_BOTH"))
            auto = {"policy_tag":"projected(columns@k=3,auto)","results":{"k3":{"eq":proj_eq},"selected_cols":lanes},
                    "metrics":{"proj_failing_cols_popcount": int(sum(proj_fail))},
                    "lanes_sig256":_bitsig256(lanes),"ker_mask_sig256":_bitsig256(ker),
                    "strict_failing_cols_sig256":_bitsig256(suppR),"proj_failing_cols_sig256":_bitsig256(proj_fail),
                    "verdict_class": vclass}

    # FILE (using ss if available)
    def _file_lanes(_ss):
        try:
            for k in ("file_lanes","file_pi_vec","file_projector_diag"):
                v=_ss.get(k)
                if isinstance(v,(list,tuple)) and all(x in (0,1,True,False) for x in v):
                    return [1 if x else 0 for x in v]
            if isinstance(_ss.get("file_projector_json"),dict):
                diag=_ss["file_projector_json"].get("diag") or _ss["file_projector_json"].get("lanes")
                if isinstance(diag,list):
                    return [1 if x else 0 for x in diag]
        except Exception: pass
        return None

    lanesF = _file_lanes(ss)
    if lanesF is None:
        freezer = {"status":"NA","na_reason_code":"FILE_PROJECTOR_MISSING"}
        filec = {"policy_tag":"projected(columns@k=3,file)","results":{"k3":{"eq":None},"selected_cols":[]},
                 "na_reason_code":"FILE_PROJECTOR_MISSING","ker_mask_sig256":_bitsig256(ker),
                 "strict_failing_cols_sig256":_bitsig256(suppR)}
    elif len(lanesF)!=n3:
        freezer = {"status":"NA","na_reason_code":"FILE_PROJECTOR_WRONG_SIZE"}
        filec = {"policy_tag":"projected(columns@k=3,file)","results":{"k3":{"eq":None},"selected_cols":[]},
                 "na_reason_code":"FILE_PROJECTOR_WRONG_SIZE","ker_mask_sig256":_bitsig256(ker),
                 "strict_failing_cols_sig256":_bitsig256(suppR)}
    elif sum(lanesF)==0:
        freezer = {"status":"OK","na_reason_code":None}
        filec = {"policy_tag":"projected(columns@k=3,file)","results":{"k3":{"eq":None},"selected_cols":lanesF},
                 "na_reason_code":"FILE_LANES_ZERO","lanes_sig256":_bitsig256(lanesF),
                 "ker_mask_sig256":_bitsig256(ker),"strict_failing_cols_sig256":_bitsig256(suppR)}
    else:
        freezer = {"status":"OK","na_reason_code":None}
        projF = [1 if (lanesF[j]==1 and suppR[j]==1) else 0 for j in range(len(suppR))]
        projEqF = (sum(projF)==0)
        def _subset(A,B): return all((a==0) or (b==1) for a,b in zip(A,B))
        vclassF = "GREEN" if strict_eq else ("KER-EXPOSED" if _subset(suppR,ker) and any(lanesF[j] and suppR[j] for j in range(len(lanesF)))
                                             else ("KER-FILTERED" if _subset(suppR,ker) else "RED_BOTH"))
        filec = {"policy_tag":"projected(columns@k=3,file)","results":{"k3":{"eq":projEqF},"selected_cols":lanesF},
                 "metrics":{"proj_failing_cols_popcount": int(sum(projF))},
                 "lanes_sig256":_bitsig256(lanesF),"ker_mask_sig256":_bitsig256(ker),
                 "strict_failing_cols_sig256":_bitsig256(suppR),"proj_failing_cols_sig256":_bitsig256(projF),
                 "verdict_class": vclassF}

    # Stamp fixture metadata into each
    def _stamp(o):
        o=dict(o or {}); o.setdefault("fixtures", fixtures)
        o.setdefault("fixture_label", fixture_label)
        if snapshot_id: o.setdefault("snapshot_id", snapshot_id)
        o.setdefault("sig8", sig8)
        o.setdefault("written_at_utc", int(_time.time()))
        return o

    bdir = _bundle_dir(district_id, fixture_label, sig8)
    names = {
      "strict": f"overlap__{district_id}__strict__{sig8}.json",
      "auto":   f"overlap__{district_id}__projected_columns_k_3_auto__{sig8}.json",
      "ab_a":   f"ab_compare__strict_vs_projected_auto__{sig8}.json",
      "freez":  f"projector_freezer__{district_id}__{sig8}.json",
      "file":   f"overlap__{district_id}__projected_columns_k_3_file__{sig8}.json",
      "ab_f":   f"ab_compare__projected_columns_k_3_file__{sig8}.json"
    }
    _write_json(bdir/names["strict"], _stamp(strict))
    _write_json(bdir/names["auto"],   _stamp(auto))
    _write_json(bdir/names["ab_a"],   _stamp({"ab_pair":{"policy":"strict__VS__projected(columns@k=3,auto)","pair_vec":{"k2":[None,None],"k3":[strict_eq, (auto.get("results") or {}).get("k3",{}).get("eq", None)]}}}))
    _write_json(bdir/names["freez"],  _stamp(freezer))
    _write_json(bdir/names["file"],   _stamp(filec))
    _write_json(bdir/names["ab_f"],   _stamp({"ab_pair":{"policy":"strict__VS__projected(columns@k=3,file)","pair_vec":{"k2":[None,None],"k3":[strict_eq, (filec.get("results") or {}).get("k3",{}).get("eq", None)]}}}))

    _write_json(bdir/"bundle.json", {
        "district_id": district_id, "fixture_label": fixture_label, "fixtures": fixtures,
        "sig8": sig8,
        "filenames": [names[k] for k in ("strict","auto","ab_a","freez","file","ab_f")],
        "core_counts": {"written": 6}, "written_at_utc": int(_time.time())
    })
    _write_json(bdir/f"loop_receipt__{fixture_label}.json", {
        "run_id": str(_uuid.uuid4()), "district_id": district_id, "fixture_label": fixture_label,
        "sig8": sig8, "bundle_dir": str(bdir), "core_counts":{"written":6},
        "timestamps":{"receipt_written_at": _time.time()}
    })

    try:
        import streamlit as _st
        _st.session_state["last_bundle_dir"] = str(bdir)
    except Exception: pass

    return True, f"v2 compute-only (HARD) 1× bundle → {bdir}", str(bdir)

# --------------------- Coverage (Option A) ---------------------
def append_coverage_from_bundle(bundle_dir: str):
    """
    Append exactly one coverage row to logs/reports/coverage.jsonl if the bundle is complete (6 core files).
    Otherwise append an error row to coverage_errors.jsonl.
    Idempotent via .coverage_logged marker inside bundle_dir.
    """
    bdir = _Ph(bundle_dir)

    marker = bdir / ".coverage_logged"
    if marker.exists():
        return True, "coverage already logged"

    # Read bundle
    try:
        bundle = _json.loads((bdir / "bundle.json").read_text(encoding="utf-8"))
    except Exception as e:
        _append_jsonl(_Ph("logs/reports")/"coverage_errors.jsonl",
                      {"ts_utc": int(_time.time()), "bundle_dir": str(bdir),
                       "reason": "BUNDLE_READ_ERROR", "error": str(e)})
        return False, "bundle read error"

    exp = bundle.get("filenames", []) or []
    if len(exp) != 6:
        _append_jsonl(_Ph("logs/reports")/"coverage_errors.jsonl",
                      {"ts_utc": int(_time.time()), "bundle_dir": str(bdir),
                       "reason": "CORE_COUNTS_MISMATCH", "filenames_expected": 6, "filenames_seen": len(exp)})
        return False, "core counts mismatch"

    # Pick core certs
    def pick(pattern):
        for fn in exp:
            if pattern in fn: return fn
        return None

    f_strict = pick("__strict__")
    f_auto   = pick("__projected_columns_k_3_auto__")
    f_file   = pick("__projected_columns_k_3_file__")

    missing = [fn for fn in (f_strict, f_auto, f_file) if not fn or not (bdir/ (fn or "")).exists()]
    if missing:
        _append_jsonl(_Ph("logs/reports")/"coverage_errors.jsonl",
                      {"ts_utc": int(_time.time()), "bundle_dir": str(bdir),
                       "reason": "MISSING_FILE", "missing": missing, "filenames_expected": exp})
        return False, "missing core cert(s)"

    strict = _json.loads((bdir/f_strict).read_text(encoding="utf-8"))
    auto   = _json.loads((bdir/f_auto).read_text(encoding="utf-8"))
    filec  = _json.loads((bdir/f_file).read_text(encoding="utf-8"))

    def k3eq(x):
        return (((x.get("results") or {}).get("k3") or {}).get("eq", None))
    def na(x):
        return x.get("na_reason_code", None)
    def posed(x):
        return na(x) is None
    def coherent(x):
        e = k3eq(x); n = na(x)
        return ((n is None and isinstance(e, bool)) or (n is not None and e is None))

    if not (coherent(auto) and coherent(filec)):
        _append_jsonl(_Ph("logs/reports")/"coverage_errors.jsonl",
                      {"ts_utc": int(_time.time()), "bundle_dir": str(bdir),
                       "reason": "NA_EQ_INCOHERENT",
                       "auto_eq": k3eq(auto), "auto_na": na(auto),
                       "file_eq": k3eq(filec), "file_na": na(filec)})
        return False, "na/eq incoherent"

    district_id  = bundle.get("district_id")
    fixture_label= bundle.get("fixture_label")
    sig8         = bundle.get("sig8")
    snapshot_id  = (strict.get("snapshot_id")
                    or auto.get("snapshot_id")
                    or filec.get("snapshot_id"))

    lanes = (auto.get("results") or {}).get("selected_cols") or []
    n3    = len(lanes) if lanes else None
    lane_size = sum(lanes) if lanes else None
    lane_frac = (lane_size / float(n3)) if (lane_size is not None and n3) else None

    def verdict_effective(cert):
        return "RED_UNPOSED" if na(cert) else (cert.get("verdict_class"))

    proj_integrity_flag = bool(k3eq(strict) is True and k3eq(auto) is False)

    policy_sig = _hashlib.sha256(_json.dumps({"k":3,"lane_policy":"AUTO+FILE"}, sort_keys=True, separators=(",",":")).encode("utf-8")).hexdigest()

    row = {
      "schema_version": "2.0.0",
      "ts_utc": int(_time.time()),
      "snapshot_id": snapshot_id,

      "district_id": district_id,
      "fixture_label": fixture_label,
      "sig8": sig8,

      "n3": n3,
      "strict_eq": k3eq(strict),
      "strict_failing_cols_sig256": strict.get("strict_failing_cols_sig256"),
      "ker_mask_sig256": strict.get("ker_mask_sig256"),
      "metrics": {
        "R3_failing_cols_popcount": ((strict.get("metrics") or {}).get("R3_failing_cols_popcount")),
        "ker_cols_popcount": ((strict.get("metrics") or {}).get("ker_cols_popcount")),
        "lane_size_auto": lane_size,
        "lane_frac_auto": lane_frac
      },

      "auto": {
        "na_reason_code": na(auto),
        "eq": k3eq(auto),
        "verdict_class": auto.get("verdict_class"),
        "verdict_effective": verdict_effective(auto),
        "lanes_sig256": auto.get("lanes_sig256"),
        "proj_failing_cols_sig256": auto.get("proj_failing_cols_sig256")
      },

      "file": {
        "na_reason_code": na(filec),
        "eq": k3eq(filec),
        "verdict_class": filec.get("verdict_class"),
        "verdict_effective": verdict_effective(filec),
        "lanes_sig256": filec.get("lanes_sig256"),
        "proj_failing_cols_sig256": filec.get("proj_failing_cols_sig256")
      },

      "projector_integrity_flag": proj_integrity_flag,
      "verdict_sig": _hashlib.sha256(f"{k3eq(strict)}|{k3eq(auto)}|{k3eq(filec)}|{auto.get('verdict_class')}|{filec.get('verdict_class')}".encode("utf-8")).hexdigest(),
      "policy_sig256": policy_sig
    }

    cov_path = _Ph("logs/reports") / "coverage.jsonl"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    with cov_path.open("a", encoding="utf-8") as fh:
        fh.write(_json.dumps(row, ensure_ascii=False, separators=(",",":")) + "\n")
    marker.touch()
    return True, "coverage appended"

# --------------------- Streamlit UI hook ---------------------
try:
    import streamlit as _st
    with _st.expander("V2 COMPUTE-ONLY (HARD) — single source of truth", expanded=True):
        if _st.button("Run solver (one press) — HARD v2 compute-only", key="btn_svr_run_v2_compute_only"):
            _st.session_state["_solver_busy"] = True
            _st.session_state["_solver_one_button_active"] = True
            try:
                ok, msg, bundle_dir = _svr_run_once_computeonly_hard(_st.session_state)
                if bundle_dir:
                    _st.session_state["last_bundle_dir"] = bundle_dir
                    # coverage (Option A)
                    ok_cov, msg_cov = append_coverage_from_bundle(bundle_dir)
                    _st.caption(f"Coverage: {msg_cov}")
                (_st.success if ok else _st.error)(msg)
            except Exception as _e:
                _st.error(f"Solver run failed: {str(_e)}")
            finally:
                _st.session_state["_solver_one_button_active"] = False
                _st.session_state["_solver_busy"] = False
except Exception:
    pass
