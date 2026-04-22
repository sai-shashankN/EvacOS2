"""Compatibility shims for upstream training stack drift.

These helpers intentionally avoid importing heavy dependencies at module import
time. Call them immediately before importing packages that are sensitive to
symbol-export drift across ``transformers`` / ``peft`` / ``unsloth`` releases.
"""

from __future__ import annotations


def patch_transformers_cache_exports() -> None:
    """Backfill cache-class exports expected by PEFT / Unsloth.

    Some recent package combinations expose cache classes from
    ``transformers.cache_utils`` but not from the top-level ``transformers``
    package. PEFT still imports these names from the top level, which causes
    ``ImportError: cannot import name 'HybridCache' from transformers`` during
    Unsloth or training startup.

    This shim copies the known cache classes onto the top-level module only
    when they exist in ``cache_utils`` and are absent from ``transformers``.
    It is a no-op on already-compatible environments.
    """

    try:
        import transformers  # type: ignore
    except ImportError:
        return

    try:
        from transformers import cache_utils  # type: ignore
    except Exception:
        return

    for name in (
        "Cache",
        "DynamicCache",
        "EncoderDecoderCache",
        "HybridCache",
    ):
        if not hasattr(transformers, name) and hasattr(cache_utils, name):
            setattr(transformers, name, getattr(cache_utils, name))
