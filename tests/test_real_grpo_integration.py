"""Integration test for disable_adapter() on real Unsloth model.

This test is CUDA-gated: it auto-skips when CUDA is not available
(i.e. on Windows dev machines).  It is intended to run in Colab.

Verifies that ``disable_adapter()`` actually produces base-model
logits on an untrained LoRA adapter (cold adapter ≈ identity).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDisableAdapterIntegration:
    def test_disable_adapter_produces_base_model_logits_on_untrained_lora(
        self,
    ) -> None:
        """Load UnslothPolicy with a cold LoRA adapter; assert that
        adapter-enabled and adapter-disabled logprobs are nearly identical
        (mean abs diff < 1e-4) on a short input.

        An untrained LoRA adapter is effectively an identity transform,
        so disabling it should produce (nearly) the same logits.
        """
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except ImportError:
            pytest.skip("unsloth not installed")

        # Load base model + attach a fresh LoRA adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_seq_length=512,
            load_in_4bit=True,
            fast_inference=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Encode a short input
        text = "The quick brown fox"
        encoded = tokenizer(text, return_tensors="pt").to(model.device)

        # Forward with adapter ENABLED
        model.eval()
        with torch.no_grad():
            out_enabled = model(**encoded)
            logits_enabled = out_enabled.logits

        # Forward with adapter DISABLED
        with torch.no_grad():
            with model.disable_adapter():
                out_disabled = model(**encoded)
                logits_disabled = out_disabled.logits

        # Mean absolute difference should be tiny for cold adapter
        mean_abs_diff = (logits_enabled - logits_disabled).abs().mean().item()
        assert mean_abs_diff < 1e-4, (
            f"Cold adapter logits differ by {mean_abs_diff:.6f} — "
            "disable_adapter() may not be working correctly."
        )
