"""Tests for the Phase 7 Unsloth training backend.

All tests MUST pass with the current Windows environment where neither
``unsloth`` nor ``vllm`` is installed. The backend is the default serious
training path, but these tests still run without the packages installed.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from training.config_schema import TrainingConfig
from training.policy_adapter import StubPolicy, UnslothPolicy, unsloth_policy_factory


# ---------------------------------------------------------------------------
# 1. Config schema
# ---------------------------------------------------------------------------


class TestConfigSchemaBackend:
    def test_default_backend_is_unsloth_with_vllm_enabled(self):
        """Default backend is 'hf' — no silent regression for existing users."""
        config = TrainingConfig()
        assert config.backend == "unsloth"
        assert config.rollout.use_vllm is True
        assert config.unsloth_max_seq_length == 4096
        assert config.load_in_4bit is True

    def test_backend_accepts_unsloth(self):
        """backend='unsloth' validates and round-trips through model_dump."""
        config = TrainingConfig(backend="unsloth")
        assert config.backend == "unsloth"
        dumped = config.model_dump(mode="json")
        assert dumped["backend"] == "unsloth"

    def test_backend_rejects_unknown(self):
        """Unknown backend strings are rejected at construction time."""
        with pytest.raises(Exception):
            TrainingConfig(backend="llamafile")


# ---------------------------------------------------------------------------
# 2. Factory raises RuntimeError when unsloth is absent
# ---------------------------------------------------------------------------


class TestUnslothFactoryGuards:
    def test_factory_raises_when_unsloth_missing(self):
        """On Windows / bare Linux without unsloth, the factory must raise a
        RuntimeError whose message contains the string 'unsloth'."""
        with pytest.raises(RuntimeError) as excinfo:
            unsloth_policy_factory("Qwen/Qwen2.5-1.5B-Instruct")
        assert "unsloth" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 3. Policy protocol: act_batch is optional
# ---------------------------------------------------------------------------


class TestPolicyProtocolOptionalBatch:
    def test_stub_policy_does_not_expose_act_batch(self):
        """StubPolicy (and hf_policy_factory) do NOT add act_batch; the
        rollout's hasattr-based fast-path detection must degrade gracefully."""
        policy = StubPolicy(seed=0)
        assert not hasattr(policy, "act_batch")


class TestGenerationModeGuards:
    def test_unsloth_hf_generate_runs_with_dropout_disabled(self, monkeypatch):
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        class FakeFastLanguageModel:
            @staticmethod
            def for_inference(model):
                model.eval()

            @staticmethod
            def for_training(model):
                model.train()

        fake_unsloth = types.ModuleType("unsloth")
        fake_unsloth.FastLanguageModel = FakeFastLanguageModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

        class FakeInputIds:
            shape = (1, 2)

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            pad_token = None
            eos_token = "<eos>"
            pad_token_id = 0
            padding_side = "left"

            def __call__(self, rendered, return_tensors="pt", padding=True, truncation=False):
                del rendered, return_tensors, padding, truncation
                return FakeBatch({"input_ids": FakeInputIds()})

            @staticmethod
            def decode(generated, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join(str(tok) for tok in generated)

        class FakeModel:
            def __init__(self):
                self.training = True
                self.device = "cpu"
                self.generate_kwargs = None

            def eval(self):
                self.training = False
                return self

            def train(self, mode=True):
                self.training = mode
                return self

            def generate(self, **kwargs):
                assert self.training is False
                assert "input_ids" in kwargs
                self.generate_kwargs = kwargs
                return [[101, 102, 103, 104]]

        policy = UnslothPolicy.__new__(UnslothPolicy)
        policy._tokenizer = FakeTokenizer()
        policy._model = FakeModel()
        policy._max_new_tokens = 8
        policy._temperature = 0.0

        outputs = policy._hf_generate(["prompt"])

        assert outputs == [("103 104", [103, 104])]
        assert policy._model.training is True
        assert policy._model.generate_kwargs["do_sample"] is False
        assert "temperature" not in policy._model.generate_kwargs

    def test_unsloth_hf_generate_passes_temperature_only_for_sampling(self, monkeypatch):
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        class FakeFastLanguageModel:
            @staticmethod
            def for_inference(model):
                model.eval()

            @staticmethod
            def for_training(model):
                model.train()

        fake_unsloth = types.ModuleType("unsloth")
        fake_unsloth.FastLanguageModel = FakeFastLanguageModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

        class FakeInputIds:
            shape = (1, 2)

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            pad_token = None
            eos_token = "<eos>"
            pad_token_id = 0
            padding_side = "left"

            def __call__(
                self,
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=False,
                max_length=None,
            ):
                del rendered, return_tensors, padding, truncation, max_length
                return FakeBatch({"input_ids": FakeInputIds()})

            @staticmethod
            def decode(generated, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join(str(tok) for tok in generated)

        class FakeModel:
            def __init__(self):
                self.training = True
                self.device = "cpu"
                self.generate_kwargs = None

            def eval(self):
                self.training = False
                return self

            def train(self, mode=True):
                self.training = mode
                return self

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                return [[101, 102, 103, 104]]

        policy = UnslothPolicy.__new__(UnslothPolicy)
        policy._tokenizer = FakeTokenizer()
        policy._model = FakeModel()
        policy._max_new_tokens = 8
        policy._max_prompt_tokens = 128
        policy._temperature = 0.7

        outputs = policy._hf_generate(["prompt"])

        assert outputs == [("103 104", [103, 104])]
        assert policy._model.generate_kwargs["do_sample"] is True
        assert policy._model.generate_kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# 4 & 5. Rollout: fallback vs fast path
# ---------------------------------------------------------------------------


class _MockPolicyActOnly:
    """A policy that only implements .act — no act_batch. Must still work."""

    def __init__(self):
        self.act_calls: list[tuple[str, str]] = []

    def act(self, prompt, agent_id, role):
        self.act_calls.append((agent_id, role))
        # Emit a wait action that parses as valid JSON.
        import json
        return json.dumps(
            {
                "episode_id": "ep_mock",
                "round_id": 0,
                "agent_id": agent_id,
                "action_id": f"a_{agent_id}",
                "action_type": "wait",
                "arguments": {},
            }
        )


class _MockPolicyWithBatch(_MockPolicyActOnly):
    """A policy that implements both act and act_batch. Rollout should prefer
    act_batch: .act must never be called when act_batch is present."""

    def __init__(self):
        super().__init__()
        self.batch_calls: list[int] = []

    def act_batch(self, prompts, agent_ids, roles):
        assert len(prompts) == len(agent_ids) == len(roles)
        self.batch_calls.append(len(prompts))
        # Return one wait-action JSON per prompt.
        import json
        return [
            json.dumps(
                {
                    "episode_id": "ep_mock",
                    "round_id": 0,
                    "agent_id": aid,
                    "action_id": f"a_{aid}",
                    "action_type": "wait",
                    "arguments": {},
                }
            )
            for aid in agent_ids
        ]


def _run_one_rollout_episode(policy, max_rounds=3):
    """Drive collect_episode for a short rollout against a real EvacEnvironment.

    Returns (episode_result, policy_handle).
    """
    from evacos_ma.env import EvacEnvironment
    from evacos_ma.models import DisasterType

    from training.rollout import collect_episode

    env = EvacEnvironment()
    result = collect_episode(
        env,
        policy,
        seed=7,
        tier="easy",
        disaster_family=DisasterType.fire,
        max_rounds=max_rounds,
    )
    return result


class TestRolloutFastPath:
    def test_fallback_policy_without_act_batch_runs_cleanly(self, monkeypatch):
        """Rollout must continue to work against the per-agent .act loop when
        no act_batch method is present. Regression check for Phase 7.
        """
        # Re-home JSONL outputs under tmp_path to avoid polluting outputs/logs.
        from training import rollout as rollout_mod
        monkeypatch.setattr(
            rollout_mod,
            "write_trace_row",
            lambda *a, **kw: None,
        )

        policy = _MockPolicyActOnly()
        assert not hasattr(policy, "act_batch")

        result = _run_one_rollout_episode(policy, max_rounds=2)

        # Sanity: at least one round ran (>= 1 samples for orch and 5 floors)
        assert result.num_rounds >= 1
        # At least 6 * num_rounds .act calls should have been issued.
        assert len(policy.act_calls) >= 6 * result.num_rounds

    def test_fast_path_uses_act_batch_once_per_round(self, monkeypatch):
        """When act_batch is present, the rollout must call it exactly once
        per round (1 call, 6 prompts) instead of 6x .act per round."""
        from training import rollout as rollout_mod
        monkeypatch.setattr(
            rollout_mod,
            "write_trace_row",
            lambda *a, **kw: None,
        )

        policy = _MockPolicyWithBatch()
        assert hasattr(policy, "act_batch")

        result = _run_one_rollout_episode(policy, max_rounds=2)

        # Exactly one act_batch call per round, each with 6 prompts
        # (1 orchestrator + 5 floor agents).
        assert len(policy.batch_calls) == result.num_rounds
        assert all(count == 6 for count in policy.batch_calls)
        # Critically: .act was never called — fast path dominated.
        assert policy.act_calls == []


# ---------------------------------------------------------------------------
# 6. enforce_eager auto-detect for T4 / pre-Ampere GPUs
# ---------------------------------------------------------------------------


class TestEnforceEagerDetection:
    """Tests for ``_should_enforce_eager_for_vllm`` helper."""

    def _import_helper(self):
        """Import the helper fresh so it picks up mocked torch."""
        from training.policy_adapter import _should_enforce_eager_for_vllm

        return _should_enforce_eager_for_vllm

    def test_returns_false_when_torch_missing(self, monkeypatch):
        """When torch is not importable, the helper must return False."""
        import sys

        # Remove torch from sys.modules if present, then block import.
        saved: dict = {}
        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                saved[key] = sys.modules.pop(key)
        monkeypatch.setitem(sys.modules, "torch", None)
        try:
            helper = self._import_helper()
            assert helper() is False
        finally:
            # Restore.
            sys.modules.pop("torch", None)
            for key in ("torch",) + tuple(
                k for k in saved if k.startswith("torch")
            ):
                if key in saved:
                    sys.modules[key] = saved[key]

    def test_returns_false_when_cuda_unavailable(self, monkeypatch):
        """When CUDA is not available, the helper must return False."""
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_cuda = types.ModuleType("torch.cuda")
        fake_cuda.is_available = lambda: False  # type: ignore[attr-defined]
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

        saved: dict = {}
        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                saved[key] = sys.modules.pop(key)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
        try:
            helper = self._import_helper()
            assert helper() is False
        finally:
            sys.modules.pop("torch", None)
            sys.modules.pop("torch.cuda", None)
            for key in saved:
                sys.modules[key] = saved[key]

    def test_returns_true_on_t4(self, monkeypatch):
        """Tesla T4 (compute 7.5) must trigger enforce_eager."""
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_cuda = types.ModuleType("torch.cuda")
        fake_cuda.is_available = lambda: True  # type: ignore[attr-defined]
        fake_cuda.get_device_capability = lambda _idx=0: (7, 5)  # type: ignore[attr-defined]
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

        saved: dict = {}
        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                saved[key] = sys.modules.pop(key)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
        try:
            helper = self._import_helper()
            assert helper() is True
        finally:
            sys.modules.pop("torch", None)
            sys.modules.pop("torch.cuda", None)
            for key in saved:
                sys.modules[key] = saved[key]

    def test_returns_false_on_a100(self, monkeypatch):
        """A100 (compute 8.0) must NOT trigger enforce_eager."""
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_cuda = types.ModuleType("torch.cuda")
        fake_cuda.is_available = lambda: True  # type: ignore[attr-defined]
        fake_cuda.get_device_capability = lambda _idx=0: (8, 0)  # type: ignore[attr-defined]
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

        saved: dict = {}
        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                saved[key] = sys.modules.pop(key)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
        try:
            helper = self._import_helper()
            assert helper() is False
        finally:
            sys.modules.pop("torch", None)
            sys.modules.pop("torch.cuda", None)
            for key in saved:
                sys.modules[key] = saved[key]


# ---------------------------------------------------------------------------
# 7. _vllm_kwargs_for_current_gpu helper
# ---------------------------------------------------------------------------


class TestVllmKwargsForCurrentGpu:
    """Tests for ``_vllm_kwargs_for_current_gpu`` helper."""

    def _import_helper(self):
        """Import the helper fresh so it picks up mocked torch."""
        from training.policy_adapter import _vllm_kwargs_for_current_gpu

        return _vllm_kwargs_for_current_gpu

    def _setup_fake_torch(self, monkeypatch, *, is_available, capability=None):
        """Common monkeypatch setup: remove real torch, install fake."""
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_cuda = types.ModuleType("torch.cuda")
        fake_cuda.is_available = lambda: is_available  # type: ignore[attr-defined]
        if capability is not None:
            fake_cuda.get_device_capability = lambda _idx=0: capability  # type: ignore[attr-defined]
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

        saved: dict = {}
        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                saved[key] = sys.modules.pop(key)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
        return saved

    def _teardown(self, saved):
        import sys

        sys.modules.pop("torch", None)
        sys.modules.pop("torch.cuda", None)
        for key in saved:
            sys.modules[key] = saved[key]

    def test_returns_empty_on_a100(self, monkeypatch):
        """A100 (compute 8.0) must return no extra kwargs."""
        saved = self._setup_fake_torch(
            monkeypatch, is_available=True, capability=(8, 0)
        )
        try:
            helper = self._import_helper()
            assert helper() == {}
        finally:
            self._teardown(saved)

    def test_returns_both_kwargs_on_t4(self, monkeypatch):
        """T4 (compute 7.5) must return enforce_eager + compilation_config."""
        saved = self._setup_fake_torch(
            monkeypatch, is_available=True, capability=(7, 5)
        )
        try:
            helper = self._import_helper()
            result = helper()
            assert result == {
                "enforce_eager": True,
                "compilation_config": {"level": 0},
            }
        finally:
            self._teardown(saved)

    def test_returns_empty_when_cuda_unavailable(self, monkeypatch):
        """No CUDA → no extra kwargs (same as Ampere+ fast path)."""
        saved = self._setup_fake_torch(monkeypatch, is_available=False)
        try:
            helper = self._import_helper()
            assert helper() == {}
        finally:
            self._teardown(saved)


class TestTransformersCompatShim:
    def test_patch_transformers_cache_exports_backfills_hybridcache(self, monkeypatch):
        import sys

        from training.compat import patch_transformers_cache_exports

        fake_transformers = types.ModuleType("transformers")
        fake_cache_utils = types.ModuleType("transformers.cache_utils")

        class FakeHybridCache:
            pass

        class FakeDynamicCache:
            pass

        fake_cache_utils.HybridCache = FakeHybridCache  # type: ignore[attr-defined]
        fake_cache_utils.DynamicCache = FakeDynamicCache  # type: ignore[attr-defined]
        fake_transformers.cache_utils = fake_cache_utils  # type: ignore[attr-defined]

        saved: dict[str, object] = {}
        for key in ("transformers", "transformers.cache_utils"):
            if key in sys.modules:
                saved[key] = sys.modules[key]

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)
        try:
            assert not hasattr(fake_transformers, "HybridCache")
            patch_transformers_cache_exports()
            assert fake_transformers.HybridCache is FakeHybridCache  # type: ignore[attr-defined]
            assert fake_transformers.DynamicCache is FakeDynamicCache  # type: ignore[attr-defined]
        finally:
            sys.modules.pop("transformers", None)
            sys.modules.pop("transformers.cache_utils", None)
            for key, value in saved.items():
                sys.modules[key] = value
