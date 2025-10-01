# app/smoke_regression.py
import sys, pathlib

# Make otcore importable
HERE = pathlib.Path(__file__).resolve().parent
OTCORE = HERE / "otcore"
sys.path.insert(0, str(OTCORE))

from schemas import Boundaries, CMap
import overlap_gate

def main():
    # d3 with lane mask [1,1,0]  (col 3 is ker)
    boundaries = Boundaries(**{
        "blocks": { "3": [[1,0,0],
                          [0,1,0]] }   # shape 2x3
    })

    # C3 identity on cols 1–2, zero col 3 → residual lives only on ker col when H2=0
    cmap_ker = CMap(**{
        "name": "C3 ker-only; C2=I",
        "blocks": {
            "3": [[1,0,0],
                  [0,1,0],
                  [0,0,0]],
            "2": [[1,0],
                  [0,1]]
        }
    })

    # H2 = 0
    H_zero = CMap(**{
        "name": "H zero",
        "blocks": { "2": [[0,0],
                          [0,0],
                          [0,0]] }
    })

    strict_cfg = {}  # no projection
    proj_cfg = {
        "enabled_layers": [3],
        "modes": {"3": "columns"},
        "source": {"3": "auto"},
        "projector_files": {}
    }

    out_strict = overlap_gate.overlap_check(boundaries, cmap_ker, H_zero,
                                            projection_config=strict_cfg, projector_cache={})
    out_proj   = overlap_gate.overlap_check(boundaries, cmap_ker, H_zero,
                                            projection_config=proj_cfg,   projector_cache={})

    print("STRICT:",   out_strict)
    print("PROJECTED:", out_proj)

    assert out_strict["3"]["eq"] is False, "strict k=3 should be False (ker-only survives)"
    assert out_proj["3"]["eq"]   is True,  "projected k=3 should be True (ker col removed)"
    print("OK: ker-only regression passed.")

if __name__ == "__main__":
    main()
