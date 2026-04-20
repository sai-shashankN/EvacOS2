"""Debug-state gating for OpenEnv endpoints.

Full-state payloads are only returned when EVACOS_DEBUG_STATE=true.
"""

from __future__ import annotations

import os


def is_debug_state_enabled() -> bool:
    """Return True iff EVACOS_DEBUG_STATE is set to the literal string 'true'."""
    return os.environ.get("EVACOS_DEBUG_STATE", "").lower() == "true"
