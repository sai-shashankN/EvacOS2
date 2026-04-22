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

    original_getattr = getattr(transformers, "__getattr__", None)

    def _resolve_cache_symbol(symbol_name: str):
        replacement = None
        if cache_utils is not None and hasattr(cache_utils, symbol_name):
            replacement = getattr(cache_utils, symbol_name)
        elif symbol_name == "HybridCache":
            replacement = (
                getattr(transformers, "DynamicCache", None)
                or getattr(transformers, "Cache", None)
                or type("HybridCache", (), {})
            )
        elif symbol_name == "EncoderDecoderCache":
            replacement = (
                getattr(transformers, "Cache", None)
                or getattr(transformers, "DynamicCache", None)
                or type("EncoderDecoderCache", (), {})
            )
        elif symbol_name == "DynamicCache":
            replacement = getattr(transformers, "Cache", None) or type(
                "DynamicCache", (), {}
            )
        elif symbol_name == "Cache":
            replacement = type("Cache", (), {})
        return replacement

    def _patched_getattr(symbol_name: str):
        if symbol_name in {
            "Cache",
            "DynamicCache",
            "EncoderDecoderCache",
            "HybridCache",
        }:
            replacement = _resolve_cache_symbol(symbol_name)
            if replacement is not None:
                setattr(transformers, symbol_name, replacement)
                return replacement
        if original_getattr is not None:
            return original_getattr(symbol_name)
        raise AttributeError(symbol_name)

    setattr(transformers, "__getattr__", _patched_getattr)

    for name in (
        "Cache",
        "DynamicCache",
        "EncoderDecoderCache",
        "HybridCache",
    ):
        if hasattr(transformers, name):
            continue

        replacement = _resolve_cache_symbol(name)
        if replacement is not None:
            setattr(transformers, name, replacement)
