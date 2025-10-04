# dedupe_snippets.py
import sys, re, hashlib
from pathlib import Path

APP = Path(sys.argv[1])
text = APP.read_text(encoding="utf-8")

lines = text.splitlines()
N = len(lines)

def norm_block(s: str) -> str:
    s = re.sub(r"#[^\n]*", "", s)       # strip comments
    s = re.sub(r"\s+", " ", s).strip()  # compress whitespace
    return s

WIN = 20  # lines per window
map_hash = {}
for i in range(0, N - WIN + 1):
    block = "\n".join(lines[i:i+WIN])
    h = hashlib.sha256(norm_block(block).encode()).hexdigest()[:12]
    map_hash.setdefault(h, []).append(i+1)

print(f"== Repeated {WIN}-line windows (potential copy/paste) ==")
for h, starts in map_hash.items():
    if len(starts) >= 3:
        # ignore trivial whitespace-only blocks
        block = "\n".join(lines[starts[0]-1:starts[0]-1+WIN]).strip()
        if len(block) > 0:
            print(f"hash={h} count={len(starts)} lines={starts[:6]}{'...' if len(starts)>6 else ''}")
