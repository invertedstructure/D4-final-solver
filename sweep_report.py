# sweep_report.py
import ast, sys, re, hashlib
from pathlib import Path

APP = Path(sys.argv[1])

SRC = APP.read_text(encoding="utf-8")
tree = ast.parse(SRC)

# ---- 1) Find expanders and nesting depth
class ExpanderVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.items = []  # (line, title, depth)
    def visit_With(self, node: ast.With):
        title = None
        for i in node.items:
            tgt = i.context_expr
            # Detect st.expander("Title")
            if isinstance(tgt, ast.Call) and hasattr(tgt.func, "attr"):
                if getattr(tgt.func.value, "id", None) == "st" and tgt.func.attr == "expander":
                    if tgt.args and isinstance(tgt.args[0], ast.Constant) and isinstance(tgt.args[0].value, str):
                        title = tgt.args[0].value
                        break
        depth = len(self.stack)
        if title:
            self.items.append((node.lineno, title, depth))
            self.stack.append(title)
            self.generic_visit(node)
            self.stack.pop()
        else:
            self.generic_visit(node)

v = ExpanderVisitor(); v.visit(tree)
print("== Expanders (line, depth, title) ==")
for ln, title, depth in sorted(v.items):
    print(f"{ln:5d}  depth={depth}  {title}")
bad = [x for x in v.items if x[2] > 0]
print(f"\nNested expanders found: {len(bad)}")

# ---- 2) Detect duplicate function/def blocks by normalized hash
class DefVisitor(ast.NodeVisitor):
    def __init__(self, src):
        self.src = src
        self.defs = []  # (name, lineno, end, hash)
    def visit_FunctionDef(self, node: ast.FunctionDef):
        lines = self.src.splitlines(keepends=True)
        # crude end guess: until next def/class or end
        start = node.lineno - 1
        end = start
        for i in range(start+1, len(lines)):
            if lines[i].startswith("def ") or lines[i].startswith("class ") or lines[i].startswith("# ─"):
                break
            end = i
        blob = "".join(lines[start:end+1])
        norm = re.sub(r"\s+", " ", blob).strip()
        h = hashlib.sha256(norm.encode()).hexdigest()[:12]
        self.defs.append((node.name, node.lineno, end+1, h))
        self.generic_visit(node)

dv = DefVisitor(SRC); dv.visit(tree)
by_hash = {}
for name, ln1, ln2, h in dv.defs:
    by_hash.setdefault(h, []).append((name, ln1, ln2))
dups = {h:v for h,v in by_hash.items() if len(v) > 1}
print("\n== Duplicate def blocks (same normalized body) ==")
for h, items in dups.items():
    print(f"hash={h}  occurrences={len(items)}")
    for (name, ln1, ln2) in items:
        print(f"  - {name}  lines {ln1}-{ln2}")

# ---- 3) Multiple definitions of key helpers
KEYS = ["build_cert_bundle", "_load_pkg_module"]
print("\n== Multiple definitions of key helpers ==")
for k in KEYS:
    hits = [d for d in dv.defs if d[0] == k]
    if len(hits) > 1:
        print(f"{k}: {len(hits)} defs")
        for _, ln1, ln2, _ in hits:
            print(f"  lines {ln1}-{ln2}")
    else:
        print(f"{k}: OK ({len(hits)} def)")
