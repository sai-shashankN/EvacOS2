from __future__ import annotations

import sys
import types

import pytest
from pydantic import ValidationError

from training.config_schema import ModelConfig
from training.policy_adapter import unsloth_policy_factory


class _FakeModel:
    def load_adapter(self, path):
        self.loaded_adapter = path


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"
    truncation_side = "left"


class _FakeFastLanguageModel:
    recorded_kwargs = None

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.recorded_kwargs = kwargs
        return _FakeModel(), _FakeTokenizer()

    @staticmethod
    def get_peft_model(model, **kwargs):
        return model

    @staticmethod
    def for_inference(model):
        return model


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.bfloat16 = object()
        self.float16 = object()
        self.float32 = object()


def _install_fake_unsloth(monkeypatch):
    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = _FakeFastLanguageModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    return fake_unsloth


class TestUnslothDtypeThreading:
    def test_unsloth_policy_factory_threads_dtype(self, monkeypatch):
        fake_torch = _FakeTorch()
        _install_fake_unsloth(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        _FakeFastLanguageModel.recorded_kwargs = None

        unsloth_policy_factory("fake-model", dtype="bfloat16")

        assert _FakeFastLanguageModel.recorded_kwargs is not None
        assert _FakeFastLanguageModel.recorded_kwargs["dtype"] is fake_torch.bfloat16

    def test_unsloth_policy_factory_dtype_none_preserves_auto(self, monkeypatch):
        fake_torch = _FakeTorch()
        _install_fake_unsloth(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        _FakeFastLanguageModel.recorded_kwargs = None

        unsloth_policy_factory("fake-model", dtype=None)

        assert _FakeFastLanguageModel.recorded_kwargs is not None
        assert _FakeFastLanguageModel.recorded_kwargs["dtype"] is None

    def test_model_config_rejects_unknown_dtype(self):
        with pytest.raises(ValidationError):
            ModelConfig(dtype="foo")

    def test_model_config_accepts_bfloat16_float16_float32(self):
        for dtype in ("bfloat16", "float16", "float32"):
            assert ModelConfig(dtype=dtype).dtype == dtype
