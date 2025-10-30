
import argparse, json, sys
from pathlib import Path

REQUIRED_B = ["D2.json","D3.json"]
REQUIRED_H = ["H01.json","H10.json","H11.json"]
REQUIRED_C = ["C000.json","C001.json","C010.json","C011.json","C100.json","C101.json","C110.json","C111.json"]

def find_required(dir_path: Path, names: list[str]) -> dict[str, Path]:
    out = {}
    for n in names:
        p = dir_path / n
        if not p.exists():
            print(f"[ERROR] Missing: {p}", file=sys.stderr)
            return {}
        out[n] = p.resolve()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", required=True, help="Dir with D2.json & D3.json")
    ap.add_argument("--H", required=True, help="Dir with H01.json, H10.json, H11.json")
    ap.add_argument("--C", required=True, help="Dir with C000..C111.json")
    ap.add_argument("--U", required=True, help="Single shapes file (e.g., inputs/U.json)")
    ap.add_argument("--out", default="manifest_full_scope.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    Bdir = Path(args.B); Hdir = Path(args.H); Cdir = Path(args.C); Upath = Path(args.U)
    if not Upath.exists():
        print(f"[ERROR] Missing U file: {Upath}", file=sys.stderr); sys.exit(2)

    def _req(dir_path, names):
        out = {}
        for n in names:
            p = dir_path / n
            if not p.exists():
                print(f"[ERROR] Missing: {p}", file=sys.stderr); sys.exit(2)
            out[n] = p.resolve()
        return out
    Bmap = _req(Bdir, REQUIRED_B)
    Hmap = _req(Hdir, REQUIRED_H)
    Cmap = _req(Cdir, REQUIRED_C)

    combos = []
    for bname in REQUIRED_B:
        for hname in REQUIRED_H:
            for cname in REQUIRED_C:
                id_ = f"{bname.split('.')[0]}_{hname.split('.')[0]}_{cname.split('.')[0]}"
                rec = {"id": id_,
                       "B": str(Bmap[bname]),
                       "H": str(Hmap[hname]),
                       "C": str(Cmap[cname]),
                       "U": str(Upath.resolve())}
                combos.append(rec)

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, separators=(",",":"), sort_keys=True) for r in combos) + "\n", encoding="utf-8")
    print(f"Wrote {len(combos)} rows → {out}")

if __name__ == "__main__":
    main()
