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
        cache_utils = None  # type: ignore[assignment]

    for name in (
        "Cache",
        "DynamicCache",
        "EncoderDecoderCache",
        "HybridCache",
    ):
        if hasattr(transformers, name):
            continue

        replacement = None
        if cache_utils is not None and hasattr(cache_utils, name):
            replacement = getattr(cache_utils, name)
        elif name == "HybridCache":
            replacement = (
                getattr(transformers, "DynamicCache", None)
                or getattr(transformers, "Cache", None)
                or type("HybridCache", (), {})
            )
        elif name == "EncoderDecoderCache":
            replacement = (
                getattr(transformers, "Cache", None)
                or getattr(transformers, "DynamicCache", None)
                or type("EncoderDecoderCache", (), {})
            )
        elif name == "DynamicCache":
            replacement = getattr(transformers, "Cache", None) or type(
                "DynamicCache", (), {}
            )
        elif name == "Cache":
            replacement = type("Cache", (), {})

        if replacement is not None:
            setattr(transformers, name, replacement)
