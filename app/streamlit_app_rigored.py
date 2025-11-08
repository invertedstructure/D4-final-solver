
# -*- coding: utf-8 -*-
# HARD V2 STREAMLIT RUNNER — PASS 1 (taxonomy + STRICT extras)
# - Deterministic 1× compute-only writer (6 core + bundle.json + loop_receipt)
# - STRICT adds: ker_red, R3_kernel_cols_popcount
# - Posed verdict taxonomy adds FILTERED_OFFLANE
# - Posed AUTO/FILE add metrics.ker_lane_count
#
# NOTE: Coverage augmentation is Pass 2 (separate).

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

    # Lane-independent diagnostics
    strict_eq = (sum(suppR)==0) if R3 else False
    subset_S_in_ker = all((s == 0) or (ker[j]==1) for j,s in enumerate(suppR))
    R3_kernel_cols_popcount = sum(1 for j in range(len(suppR)) if suppR[j]==1 and ker[j]==1)
    ker_red = (not strict_eq) and subset_S_in_ker

    D,Htag,Ctag,fixture_label = _fixture_tuple_from_paths(pB,pH,pC)
    district_id = rc.get("district_id") or f"{D}{_hash8({'d3':d3})}"
    sig8        = rc.get("sig8") or (rc.get("embed_sig","")[:8] if rc.get("embed_sig") else _hash8({"H2":H2,"d3":d3,"C3":C3}))
    snapshot_id = rc.get("snapshot_id") or (ss.get("world_snapshot_id") if isinstance(ss,dict) else None) or ""
    fixtures    = {"district": D, "H": Htag, "C": Ctag, "U": "U"}

    # STRICT cert
    strict = {
      "policy_tag": "strict(k=3)",
      "results": {"k3":{"eq": strict_eq}},
      "metrics": {
          "R3_failing_cols_popcount": int(sum(suppR)),
          "ker_cols_popcount": int(sum(ker)),
          "R3_kernel_cols_popcount": int(R3_kernel_cols_popcount)
      },
      "strict_failing_cols_sig256": _bitsig256(suppR),
      "ker_mask_sig256": _bitsig256(ker),
      "ker_red": bool(ker_red),
      "fixtures": fixtures,
      "fixture_label": fixture_label,
      "snapshot_id": snapshot_id,
      "sig8": sig8
    }

    # Helper to classify posed regimes
    def _classify_verdict(strict_eq: bool, ker_only: bool, proj_eq: bool):
        if strict_eq:
            # GREEN; projector must also pass; if not, integrity violation
            return "GREEN"
        if ker_only:
            return "KER-FILTERED" if proj_eq else "KER-EXPOSED"
        else:
            return "FILTERED_OFFLANE" if proj_eq else "RED_BOTH"

    # ---------------- AUTO regime ----------------
    if not (mC==nC):
        auto = {
            "policy_tag":"projected(columns@k=3,auto)",
            "results":{"k3":{"eq":None},"selected_cols":[]},
            "na_reason_code":"C3_NON_SQUARE",
            "ker_mask_sig256":_bitsig256(ker),
            "strict_failing_cols_sig256":_bitsig256(suppR)
        }
    else:
        lanes = list(C3[-1] if mC>0 else [0]*n3)
        if sum(lanes)==0:
            auto = {
                "policy_tag":"projected(columns@k=3,auto)",
                "results":{"k3":{"eq":None},"selected_cols":lanes},
                "na_reason_code":"LANES_ZERO",
                "lanes_sig256":_bitsig256(lanes),
                "ker_mask_sig256":_bitsig256(ker),
                "strict_failing_cols_sig256":_bitsig256(suppR)
            }
        else:
            proj_fail = [1 if (lanes[j]==1 and suppR[j]==1) else 0 for j in range(len(suppR))]
            proj_eq   = (sum(proj_fail)==0)
            ker_only  = subset_S_in_ker
            vclass    = _classify_verdict(strict_eq, ker_only, proj_eq)
            auto = {
                "policy_tag":"projected(columns@k=3,auto)",
                "results":{"k3":{"eq":proj_eq},"selected_cols":lanes},
                "metrics":{
                    "proj_failing_cols_popcount": int(sum(proj_fail)),
                    "ker_lane_count": int(sum(1 for j in range(len(lanes)) if lanes[j]==1 and ker[j]==1))
                },
                "lanes_sig256":_bitsig256(lanes),
                "ker_mask_sig256":_bitsig256(ker),
                "strict_failing_cols_sig256":_bitsig256(suppR),
                "proj_failing_cols_sig256":_bitsig256(proj_fail),
                "verdict_class": vclass
            }

    # ---------------- FILE regime ----------------
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
        filec = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq":None},"selected_cols":[]},
            "na_reason_code":"FILE_PROJECTOR_MISSING",
            "ker_mask_sig256":_bitsig256(ker),
            "strict_failing_cols_sig256":_bitsig256(suppR)
        }
    elif len(lanesF)!=n3:
        freezer = {"status":"NA","na_reason_code":"FILE_PROJECTOR_WRONG_SIZE"}
        filec = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq":None},"selected_cols":[]},
            "na_reason_code":"FILE_PROJECTOR_WRONG_SIZE",
            "ker_mask_sig256":_bitsig256(ker),
            "strict_failing_cols_sig256":_bitsig256(suppR)
        }
    elif sum(lanesF)==0:
        freezer = {"status":"OK","na_reason_code":None}
        filec = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq":None},"selected_cols":lanesF},
            "na_reason_code":"FILE_LANES_ZERO",
            "lanes_sig256":_bitsig256(lanesF),
            "ker_mask_sig256":_bitsig256(ker),
            "strict_failing_cols_sig256":_bitsig256(suppR)
        }
    else:
        freezer = {"status":"OK","na_reason_code":None}
        projF = [1 if (lanesF[j]==1 and suppR[j]==1) else 0 for j in range(len(suppR))]
        projEqF = (sum(projF)==0)
        ker_only = subset_S_in_ker
        vclassF  = _classify_verdict(strict_eq, ker_only, projEqF)
        filec = {
            "policy_tag":"projected(columns@k=3,file)",
            "results":{"k3":{"eq":projEqF},"selected_cols":lanesF},
            "metrics":{
                "proj_failing_cols_popcount": int(sum(projF)),
                "ker_lane_count": int(sum(1 for j in range(len(lanesF)) if lanesF[j]==1 and ker[j]==1))
            },
            "lanes_sig256":_bitsig256(lanesF),
            "ker_mask_sig256":_bitsig256(ker),
            "strict_failing_cols_sig256":_bitsig256(suppR),
            "proj_failing_cols_sig256":_bitsig256(projF),
            "verdict_class": vclassF
        }

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

    # GREEN integrity trap (should never occur): strict_eq True but proj says False
    try:
        if strict_eq:
            # Check AUTO/FILE posed cases only
            auto_eq = (auto.get("results") or {}).get("k3",{}).get("eq", None)
            file_eq = (filec.get("results") or {}).get("k3",{}).get("eq", None)
            if auto_eq is False or file_eq is False:
                _write_json(bdir/"projector_integrity_flag.json", {
                    "flag": "PROJECTOR_INTEGRITY_FAIL",
                    "note": "strict_eq=True but a projected regime reported eq=False",
                    "auto_eq": auto_eq, "file_eq": file_eq
                })
    except Exception:
        pass

    try:
        import streamlit as _st
        _st.session_state["last_bundle_dir"] = str(bdir)
    except Exception: pass

    return True, f"v2 compute-only (HARD) 1× bundle → {bdir}", str(bdir)

# --------------------- Streamlit UI hook ---------------------
try:
    import streamlit as _st
    with _st.expander("V2 COMPUTE-ONLY (HARD) — single source of truth (PASS 1)", expanded=True):
        if _st.button("Run solver (one press) — HARD v2 compute-only (PASS 1)", key="btn_svr_run_v2_compute_only_pass1"):
            _st.session_state["_solver_busy"] = True
            _st.session_state["_solver_one_button_active"] = True
            try:
                ok, msg, bundle_dir = _svr_run_once_computeonly_hard(_st.session_state)
                if bundle_dir:
                    _st.session_state["last_bundle_dir"] = bundle_dir
                (_st.success if ok else _st.error)(msg)
            except Exception as _e:
                _st.error(f"Solver run failed: {str(_e)}")
            finally:
                _st.session_state["_solver_one_button_active"] = False
                _st.session_state["_solver_busy"] = False
except Exception:
    pass
