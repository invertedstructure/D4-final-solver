
from pathlib import Path
import json, csv, sys

# Import headless entry from app
try:
    from streamlit_app_rigored import solver_run_once_paths
except Exception as e:
    print("Could not import solver_run_once_paths from streamlit_app_rigored.py:", e)
    sys.exit(2)

def manifest_rows():
    j = Path("manifest_full_scope.jsonl")
    c = Path("manifest_full_scope.csv")
    rows = []
    if j.exists():
        for line in j.read_text().splitlines():
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    elif c.exists():
        with c.open() as f:
            rows = list(csv.DictReader(f))
    return rows

def main():
    rows = manifest_rows()
    if not rows:
        print("No manifest found. Provide manifest_full_scope.jsonl or .csv")
        sys.exit(1)
    out_rows = []
    for i, rec in enumerate(rows, start=1):
        fx = f"C{i:03d}"
        prox = "BA" if (i % 2) else "BC"
        r = solver_run_once_paths(rec["B"], rec["H"], rec["C"], rec["U"], fixture_label=fx, prox_label=prox)
        print(f"{fx}: {r['msg']} (snapshot={r['snapshot_id']}, lanes_sig8={r['lanes_sig8']}, pop={r['lanes_popcount']})")
        out_rows.append({"fixture": fx, "snapshot_id": r["snapshot_id"], "lanes_sig8": r["lanes_sig8"],
                         "lanes_popcount": r["lanes_popcount"], "bundle_dir": r["bundle_dir"]})
    out_dir = Path("logs/suite_runs"); out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir/"suite_index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    (out_dir/"suite_index.json").write_text(json.dumps(out_rows, indent=2))
    print(f"Wrote {len(out_rows)} rows → {out_dir/'suite_index.csv'}")

if __name__ == "__main__":
    main()
