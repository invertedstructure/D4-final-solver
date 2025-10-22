# Pass 1 — Make the solver the only writer & install constants

Goal: one button writes the full bundle (5 or 6 certs), pins updated there only. Everything else is read-only.

## 1.1 Constants (single source of truth)
Add **once at the very top** of `streamlit_app_rigored.py` (after imports):

```python
# ==== CERT/ENGINE CONSTANTS (single source of truth) ====
SCHEMA_VERSION = "2.0.0"
ENGINE_REV     = "rev-20251022-1"

NA_CODES = {
    "C3_NOT_SQUARE", "BAD_SHAPE",
    "AUTO_REQUIRES_SQUARE_C3", "ZERO_LANE_PROJECTOR",
    "FREEZER_C3_NOT_SQUARE", "FREEZER_ZERO_LANE_PROJECTOR",
    "FREEZER_BAD_SHAPE", "FREEZER_ASSERT_MISMATCH",
    "BAD_PROJECTOR_SHAPE", "NOT_IDEMPOTENT"
}

DIRS = {"root": "logs", "certs": "logs/certs"}

def short(h: str) -> str:
    return (h or "")[:8]

# Ensure dirs exist
import os
os.makedirs(DIRS["certs"], exist_ok=True)
```

## 1.2 Kill stray writers
Search the codebase for any of these and **remove/disable** outside of the one-button solver handler:
- `_svr_write_cert(`
- `ab_pin` or `ab_pin_auto` / `ab_pin_file` assignments
- text: `write cert`, `embedded`, `AB writer`, `freeze projector`

Leave writers **only** inside the one-button solver block.

## 1.3 Install a session press-lock
Inside the solver handler (the button you click to run the loop), wrap the body with a guard:

```python
# At file top (once):
import streamlit as st

def _press_lock_begin():
    if st.session_state.get("_solver_busy"):
        return False
    st.session_state["_solver_busy"] = True
    return True

def _press_lock_end():
    st.session_state["_solver_busy"] = False

# In the solver button handler:
if run_btn:
    if not _press_lock_begin():
        st.warning("Solver already running…")
    else:
        try:
            # ... existing solver code ...
            pass
        finally:
            _press_lock_end()
```

### Done when
- One press produces 5 certs (6 if FILE posed).
- No other UI writes certs or pins.
