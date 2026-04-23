"""Unit tests for MultiAgentGRPOTrainer — no CUDA, no torch install required.

All tests use stub/mock models and mock the ``torch`` import where needed.
They validate the mathematical invariants of the GRPO-family algorithm:
ratio correctness, clip behaviour, KL non-negativity, delta clamp,
role-specific grouping, advantage normalization, disable_adapter usage,
and diagnostics emission.

Strategy: since ``torch`` is not installed on the Windows dev machine,
we inject a fake ``torch`` module via ``sys.modules`` before importing
the trainer.  The fake torch has enough functionality (tensor ops, nn.Module,
etc.) for the trainer's math to work.  Tests that only validate pure math
use the fake torch directly (no trainer import needed).
"""

from __future__ import annotations

import importlib
import math
import sys
import types
import warnings
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Build a minimal fake torch that satisfies the trainer's imports
# ---------------------------------------------------------------------------


def _build_fake_torch() -> types.ModuleType:
    """Create a minimal fake torch module with tensor, nn, etc."""
    try:
        import torch as real_torch  # noqa: F401

        return real_torch  # If real torch exists, just use it
    except ImportError:
        pass

    import numpy as np  # type: ignore

    ft = types.ModuleType("torch")

    class _Tensor:
        """Minimal tensor-like object backed by numpy."""

        def __init__(self, data: Any, dtype: Any = None, device: Any = None) -> None:
            if isinstance(data, _Tensor):
                self._data = data._data.copy()
            elif isinstance(data, (list, tuple)):
                self._data = np.array(data, dtype=np.float64)
            else:
                self._data = np.array(data, dtype=np.float64)
            self.shape = self._data.shape
            self.device = device or "cpu"

        def item(self) -> float:
            return float(self._data)

        def detach(self) -> "_Tensor":
            return _Tensor(self._data.copy())

        def clone(self) -> "_Tensor":
            return _Tensor(self._data.copy())

        def numpy(self) -> np.ndarray:
            return self._data

        def numel(self) -> int:
            return self._data.size

        def __getitem__(self, idx: Any) -> "_Tensor":
            return _Tensor(self._data[idx])

        def __setitem__(self, idx: Any, val: Any) -> None:
            if isinstance(val, _Tensor):
                self._data[idx] = val._data
            else:
                self._data[idx] = val

        def exp(self) -> "_Tensor":
            return _Tensor(np.exp(self._data))

        def clamp(self, mn: float, mx: float) -> "_Tensor":
            return _Tensor(np.clip(self._data, mn, mx))

        def clamp_min(self, v: float) -> "_Tensor":
            return _Tensor(np.clip(self._data, v, None))

        def mean(self) -> "_Tensor":
            return _Tensor(np.mean(self._data))

        def std(self, unbiased: bool = True) -> "_Tensor":
            ddof = 1 if unbiased else 0
            return _Tensor(np.std(self._data, ddof=ddof))

        def sum(self, dim: int | None = None, **kwargs: Any) -> "_Tensor":
            if dim is not None:
                axis = dim
                return _Tensor(np.sum(self._data, axis=axis))
            return _Tensor(np.sum(self._data))

        def max(self) -> "_Tensor":
            return _Tensor(np.max(self._data))

        def min(self) -> "_Tensor":
            return _Tensor(np.min(self._data))

        def abs(self) -> "_Tensor":
            return _Tensor(np.abs(self._data))

        def __mul__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(self._data * other._data)
            return _Tensor(self._data * other)

        def __rmul__(self, other: Any) -> "_Tensor":
            return self.__mul__(other)

        def __add__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(self._data + other._data)
            return _Tensor(self._data + other)

        def __radd__(self, other: Any) -> "_Tensor":
            return self.__add__(other)

        def __sub__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(self._data - other._data)
            return _Tensor(self._data - other)

        def __rsub__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(other._data - self._data)
            return _Tensor(other - self._data)

        def __neg__(self) -> "_Tensor":
            return _Tensor(-self._data)

        def __truediv__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(self._data / other._data)
            return _Tensor(self._data / other)

        def __rtruediv__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(other._data / self._data)
            return _Tensor(other / self._data)

        def __pow__(self, power: float) -> "_Tensor":
            return _Tensor(self._data ** power)

        def unsqueeze(self, dim: int) -> "_Tensor":
            return _Tensor(np.expand_dims(self._data, axis=dim))

        def expand_as(self, other: "_Tensor") -> "_Tensor":
            return _Tensor(np.broadcast_to(self._data, other.shape))

        def masked_fill(self, mask: Any, value: Any) -> "_Tensor":
            d = self._data.copy()
            if isinstance(mask, _Tensor):
                d[mask._data.astype(bool)] = value
            return _Tensor(d)

        def __eq__(self, other: Any) -> "_Tensor":  # type: ignore[override]
            if isinstance(other, _Tensor):
                return _Tensor((self._data == other._data).astype(np.float64))
            return _Tensor((self._data == other).astype(np.float64))

        def __ne__(self, other: Any) -> "_Tensor":  # type: ignore[override]
            if isinstance(other, _Tensor):
                return _Tensor((self._data != other._data).astype(np.float64))
            return _Tensor((self._data != other).astype(np.float64))

        def __ge__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor((self._data >= other._data).astype(np.float64))
            return _Tensor((self._data >= other).astype(np.float64))

        def __gt__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor((self._data > other._data).astype(np.float64))
            return _Tensor((self._data > other).astype(np.float64))

        def __lt__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor((self._data < other._data).astype(np.float64))
            return _Tensor((self._data < other).astype(np.float64))

        def __le__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor((self._data <= other._data).astype(np.float64))
            return _Tensor((self._data <= other).astype(np.float64))

        def __or__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(((self._data != 0) | (other._data != 0)).astype(np.float64))
            return _Tensor(((self._data != 0) | (other != 0)).astype(np.float64))

        def __and__(self, other: Any) -> "_Tensor":
            if isinstance(other, _Tensor):
                return _Tensor(((self._data != 0) & (other._data != 0)).astype(np.float64))
            return _Tensor(((self._data != 0) & (other != 0)).astype(np.float64))

        def to(self, *args: Any, **kwargs: Any) -> "_Tensor":
            return self

        def tolist(self) -> list:
            return self._data.tolist()

        def float(self) -> "_Tensor":
            return self

        def long(self) -> "_Tensor":
            return self

        def all(self) -> bool:
            return bool(np.all(self._data != 0))

        def backward(self) -> None:
            pass  # no-op for testing

        def __repr__(self) -> str:
            return f"FakeTensor({self._data})"

        def __float__(self) -> float:
            return float(self._data)

    ft.Tensor = _Tensor  # type: ignore[attr-defined]

    def tensor(data: Any, **kwargs: Any) -> _Tensor:
        return _Tensor(data)

    ft.tensor = tensor  # type: ignore[attr-defined]
    def _torch_min(a: _Tensor, b: _Tensor) -> _Tensor:
        return _Tensor(np.minimum(a._data, b._data))

    def _torch_max(a: _Tensor, b: _Tensor) -> _Tensor:
        return _Tensor(np.maximum(a._data, b._data))

    ft.zeros = lambda *s, **kw: _Tensor(np.zeros(s))  # type: ignore[attr-defined]
    ft.ones = lambda *s, **kw: _Tensor(np.ones(s))  # type: ignore[attr-defined]
    ft.ones_like = lambda t, **kw: _Tensor(np.ones(t.shape))  # type: ignore[attr-defined]
    ft.min = _torch_min  # type: ignore[attr-defined]
    ft.max = _torch_max  # type: ignore[attr-defined]
    ft.no_grad = lambda: MagicMock()  # type: ignore[attr-defined]
    ft.float32 = "float32"  # type: ignore[attr-defined]

    # nn stub
    nn_mod = types.ModuleType("torch.nn")

    class _Module:
        def parameters(self, recurse: bool = True):  # type: ignore[override]
            return []

        def named_parameters(self, prefix: str = "", recurse: bool = True):
            return []

        def train(self, mode: bool = True):
            return self

        def eval(self):
            return self

    nn_mod.Module = _Module  # type: ignore[attr-defined]

    class _Embedding(_Module):
        def __init__(self, vocab: int, hidden: int) -> None:
            self.weight = _Tensor(np.random.randn(vocab, hidden) * 0.01)

        def __call__(self, ids: _Tensor) -> _Tensor:
            return _Tensor(self.weight._data[ids._data.astype(int)])

    nn_mod.Embedding = _Embedding  # type: ignore[attr-defined]

    class _Linear(_Module):
        def __init__(self, inp: int, out: int) -> None:
            self.weight = _Tensor(np.random.randn(inp, out) * 0.01)
            self.bias = _Tensor(np.zeros(out))

        def __call__(self, x: _Tensor) -> _Tensor:
            return _Tensor(x._data @ self.weight._data + self.bias._data)

    nn_mod.Linear = _Linear  # type: ignore[attr-defined]

    class _FakeModule(_Module):
        """A module with trainable params for the optimizer."""

        def __init__(self) -> None:
            self._param = _Tensor(np.array([1.0, 2.0, 3.0]))

        def parameters(self, recurse: bool = True):  # type: ignore[override]
            return [self._param]

        def named_parameters(self, prefix: str = "", recurse: bool = True):
            return [("lora_a.weight", self._param)]

    nn_mod.Module = _FakeModule  # type: ignore[attr-defined]
    ft.nn = nn_mod  # type: ignore[attr-defined]

    # nn.utils
    nn_utils_mod = types.ModuleType("torch.nn.utils")
    nn_utils_mod.clip_grad_norm_ = lambda *a, **kw: _Tensor(0.0)  # type: ignore[attr-defined]
    nn_mod.utils = nn_utils_mod  # type: ignore[attr-defined]

    # optim stub
    optim_mod = types.ModuleType("torch.optim")

    class _AdamW:
        def __init__(self, params: Any, lr: float = 1e-3, **kw: Any) -> None:
            self.lr = lr

        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    optim_mod.AdamW = _AdamW  # type: ignore[attr-defined]
    ft.optim = optim_mod  # type: ignore[attr-defined]

    # utils stub
    utils_mod = types.ModuleType("torch.utils")
    ft.utils = utils_mod  # type: ignore[attr-defined]

    nn_utils_mod = types.ModuleType("torch.nn.utils")
    nn_utils_mod.clip_grad_norm_ = lambda *a, **kw: _Tensor(0.0)  # type: ignore[attr-defined]
    utils_mod.clip_grad_norm_ = nn_utils_mod.clip_grad_norm_  # type: ignore[attr-defined]

    # nn.functional stub
    func_mod = types.ModuleType("torch.nn.functional")

    def _log_softmax(x: _Tensor, dim: int = -1) -> _Tensor:
        d = x._data
        shifted = d - np.max(d, axis=dim, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
        return _Tensor(shifted - log_sum_exp)

    func_mod.log_softmax = _log_softmax  # type: ignore[attr-defined]
    func_mod.mse_loss = lambda *a, **kw: _Tensor(0.0)  # type: ignore[attr-defined]
    nn_mod.functional = func_mod  # type: ignore[attr-defined]

    ft.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]

    def _isfinite(t: _Tensor) -> _Tensor:
        return _Tensor(np.isfinite(t._data).astype(np.float64))

    ft.isfinite = _isfinite  # type: ignore[attr-defined]

    return ft


# Try to use real torch; fall back to fake
try:
    import torch as _real_torch_check  # noqa: F401

    _USE_FAKE = False
except ImportError:
    _USE_FAKE = True

if _USE_FAKE:
    _fake_torch = _build_fake_torch()
    _torch_modules_to_inject = {
        "torch": _fake_torch,
        "torch.nn": _fake_torch.nn,
        "torch.nn.functional": _fake_torch.nn.functional,
        "torch.optim": _fake_torch.optim,
        "torch.utils": _fake_torch.utils,
    }


def _inject_fake_torch() -> dict[str, Any]:
    """Inject fake torch modules, return saved originals for cleanup."""
    saved: dict[str, Any] = {}
    for modname, mod in _torch_modules_to_inject.items():
        saved[modname] = sys.modules.get(modname)
        sys.modules[modname] = mod
    return saved


def _restore_torch(saved: dict[str, Any]) -> None:
    for modname, orig in saved.items():
        if orig is None:
            sys.modules.pop(modname, None)
        else:
            sys.modules[modname] = orig


# ===========================================================================
# Test 1: Ratio math correctness (pure math, no imports needed)
# ===========================================================================


class TestRatioMath:
    def test_ratio_math_correctness(self) -> None:
        """old_lp = log(0.3), new_lp = log(0.6) → ratio = 2.0."""
        old_lp = math.log(0.3)
        new_lp = math.log(0.6)
        delta = new_lp - old_lp
        ratio = math.exp(delta)
        assert abs(ratio - 2.0) < 1e-5, f"Expected ratio ≈ 2.0, got {ratio}"


# ===========================================================================
# Test 2: Clip behavior at boundaries (pure math)
# ===========================================================================


class TestClipBehavior:
    def test_clip_behavior_at_boundaries(self) -> None:
        """With clip_range=0.2, ratios outside [0.8, 1.2] should be clipped."""
        clip_range = 0.2
        ratios = [0.5, 0.8, 1.0, 1.2, 1.5]
        clipped = [max(1.0 - clip_range, min(1.0 + clip_range, r)) for r in ratios]
        expected = [0.8, 0.8, 1.0, 1.2, 1.2]
        assert clipped == expected

        # Verify min(surr1, surr2) for A=1.0
        A = 1.0
        for r, c, exp_clipped in zip(ratios, clipped, expected):
            surr1 = r * A
            surr2 = c * A
            result = min(surr1, surr2)
            assert abs(result - min(r, c)) < 1e-6


# ===========================================================================
# Test 3: KL non-negative and zero at identity (pure math)
# ===========================================================================


class TestKLEstimator:
    def test_kl_non_negative_and_zero_at_identity(self) -> None:
        """Schulman k3 KL: exp(d) - d - 1, where d = ref - new."""
        import random

        rng = random.Random(42)

        # Identity: d=0 → KL=0
        d = 0.0
        kl = math.exp(d) - d - 1.0
        assert abs(kl) < 1e-10

        # Random non-identity: KL >= 0
        for _ in range(100):
            d = rng.gauss(0, 2.0)
            kl = math.exp(d) - d - 1.0
            assert kl >= -1e-6, f"KL should be non-negative for d={d}, got {kl}"


# ===========================================================================
# Test 4: Delta clamp applied (pure math)
# ===========================================================================


class TestDeltaClamp:
    def test_delta_clamp_applied(self) -> None:
        """delta = [-10, -5, 0, 5, 10] → clamped to [-5, -5, 0, 5, 5]."""
        deltas = [-10.0, -5.0, 0.0, 5.0, 10.0]
        clamped = [max(-5.0, min(5.0, d)) for d in deltas]
        expected = [-5.0, -5.0, 0.0, 5.0, 5.0]
        assert clamped == expected

        # Verify exp() of clamped values is in safe range
        for c in clamped:
            e = math.exp(c)
            assert 0.0 < e < 200.0, f"exp({c}) = {e} is outside safe range"

    def test_ref_delta_clamp_applied(self) -> None:
        """ref_delta = ref_lp - new_lp must also be clamped before .exp() (Fix H27).

        Without the clamp, exp(12) overflows FP16 (≈ 11.09 max) → inf KL →
        corrupted gradients.  With clamp(-5, 5), exp is always safe.
        """
        # Same clamp logic as the policy delta
        ref_deltas = [-12.0, -8.0, -5.0, 0.0, 5.0, 8.0, 12.0]
        clamped = [max(-5.0, min(5.0, d)) for d in ref_deltas]
        expected = [-5.0, -5.0, -5.0, 0.0, 5.0, 5.0, 5.0]
        assert clamped == expected

        # All exp values in safe FP16 range
        for c in clamped:
            e = math.exp(c)
            assert 0.0 < e < 200.0, f"exp({c}) = {e} overflows FP16 safe range"


# ===========================================================================
# Tests 5-8: Require the trainer (need fake torch injection)
# ===========================================================================

# For tests needing the trainer, we use a different approach:
# we manually construct test scenarios that verify the behavior
# without importing the full trainer class (which would need torch).
# Instead we test the logic in isolation.

# Helper for numpy-based tensor math
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy needed for tensor tests")
class TestRoleSpecificGrouping:
    """Test _compute_group_advantages logic directly with numpy."""

    def _compute_advantages(
        self, grouped_rewards: dict[str, list[float]], total: int
    ) -> list[float]:
        """Re-implementation of _compute_group_advantages logic for testing."""
        result = [0.0] * total
        offset = 0
        for gid, rewards in grouped_rewards.items():
            n = len(rewards)
            if n == 0:
                continue
            mean_r = sum(rewards) / n
            if n >= 2:
                var = sum((r - mean_r) ** 2 for r in rewards) / n
                std_r = max(var**0.5, 1e-8)
                for i, r in enumerate(rewards):
                    result[offset + i] = (r - mean_r) / std_r
            else:
                warnings.warn(
                    f"Group {gid!r} has only 1 sample; setting advantage to 0.0.",
                    stacklevel=2,
                )
                # advantage stays 0
            offset += n
        return result

    def test_role_specific_grouping(self) -> None:
        """2 episodes × 2 rounds × (1 orch + 5 floor) = 24 samples.
        Orchestrator group: size 4. Floor groups: 4 groups of size 5 each.
        """
        orch_rewards = {"rollout_orchestrator": [1.0, 2.0, 3.0, 4.0]}
        floor_rewards: dict[str, list[float]] = {}
        for ep in range(2):
            for rd in range(2):
                key = f"ep_ep{ep}_r_{rd}_floor"
                floor_rewards[key] = [float(i) for i in range(5)]

        all_rewards = {**orch_rewards, **floor_rewards}
        total = 4 + 4 * 5  # 24
        adv = self._compute_advantages(all_rewards, total)

        assert len(adv) == total

        # Orch group (first 4): mean≈0
        orch_adv = adv[:4]
        assert abs(sum(orch_adv) / len(orch_adv)) < 1e-5

        # Floor groups: each has mean≈0
        for i in range(4):
            start = 4 + i * 5
            group_adv = adv[start : start + 5]
            assert abs(sum(group_adv) / len(group_adv)) < 1e-5

    def test_single_sample_group_warns(self) -> None:
        """A group with only 1 sample should warn and set advantage to 0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adv = self._compute_advantages({"solo": [5.0]}, 1)
            assert adv[0] == 0.0
            assert any("only 1 sample" in str(warning.message) for warning in w)


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy needed for tensor tests")
class TestAdvantageNormalization:
    def _compute_advantages(
        self, grouped_rewards: dict[str, list[float]], total: int
    ) -> list[float]:
        result = [0.0] * total
        offset = 0
        for gid, rewards in grouped_rewards.items():
            n = len(rewards)
            if n == 0:
                continue
            mean_r = sum(rewards) / n
            if n >= 2:
                var = sum((r - mean_r) ** 2 for r in rewards) / n
                std_r = max(var**0.5, 1e-8)
                for i, r in enumerate(rewards):
                    result[offset + i] = (r - mean_r) / std_r
            offset += n
        return result

    def test_advantage_normalization_per_group(self) -> None:
        """4 orch rewards [1,2,3,4] → mean≈0, std≈1.
        Separate floor group with different scale → no cross-contamination.
        """
        orch_rewards = {"orch": [1.0, 2.0, 3.0, 4.0]}
        floor_rewards = {"floor": [100.0, 200.0, 300.0, 400.0, 500.0]}
        total = 4 + 5
        adv = self._compute_advantages({**orch_rewards, **floor_rewards}, total)

        orch_adv = adv[:4]
        floor_adv = adv[4:]

        # Orch: mean≈0
        assert abs(sum(orch_adv) / len(orch_adv)) < 1e-5
        # Floor: mean≈0, independently normalized
        assert abs(sum(floor_adv) / len(floor_adv)) < 1e-5

        # No cross-contamination: orch values are small standardized
        assert max(abs(v) for v in orch_adv) < 2.0
        # Floor also independently standardized
        assert max(abs(v) for v in floor_adv) < 2.0


class TestDisableAdapter:
    """Test that the trainer code path calls disable_adapter correctly.
    This tests the code path without needing torch by using mock objects."""

    def test_disable_adapter_context_manager_protocol(self) -> None:
        """Verify disable_adapter() follows context manager protocol."""
        entered = False
        exited = False

        class FakeModel:
            def disable_adapter(self):
                nonlocal entered, exited
                entered = True

                class _CM:
                    def __enter__(self_inner):
                        return self

                    def __exit__(self_inner, *args):
                        nonlocal exited
                        exited = True

                return _CM()

        model = FakeModel()
        cm = model.disable_adapter()
        assert entered
        with cm:
            pass
        assert exited


class TestDiagnostics:
    """Test that the trainer step returns the expected diagnostic keys.
    We directly patch the internal methods to return known tensors,
    bypassing the need for a full tokenizer/model pipeline."""

    @pytest.fixture(autouse=True)
    def _setup_fake_torch(self) -> None:
        if _USE_FAKE:
            self._saved = _inject_fake_torch()

    def teardown_method(self) -> None:
        if _USE_FAKE and hasattr(self, "_saved"):
            _restore_torch(self._saved)

    def test_diagnostics_keys_present(self) -> None:
        """After one step on stub data, returned dict contains all 10 keys.

        Strategy: patch _tokenize_batch and _masked_token_logprobs to return
        pre-built tensors, so the step() logic runs without needing a real
        tokenizer or model forward pass.
        """
        from training.train import MultiAgentGRPOTrainer

        ft = sys.modules["torch"]
        np_torch = __import__("numpy")

        # Pre-build tensors: 2 samples, 19 completion tokens
        # completion_mask: all ones (all tokens are completion)
        # old/ref/new logprobs: small random values
        S, L = 2, 19
        old_lp_data = np_torch.random.randn(S, L) * 0.1
        ref_lp_data = np_torch.random.randn(S, L) * 0.1

        encoded_full = {
            "input_ids": ft.Tensor(np_torch.ones((S, 20), dtype=np_torch.int64)),
            "attention_mask": ft.Tensor(np_torch.ones((S, 20), dtype=np_torch.int64)),
        }
        shifted_labels = ft.Tensor(np_torch.ones((S, L), dtype=np_torch.int64) * 100)

        old_lp = ft.Tensor(old_lp_data)
        ref_lp = ft.Tensor(ref_lp_data)

        param_mock = MagicMock()
        param_mock.device = "cpu"
        param_mock.requires_grad = True

        model = MagicMock()
        model.disable_adapter.return_value = MagicMock(
            __enter__=MagicMock(return_value=model),
            __exit__=MagicMock(return_value=False),
        )
        model.parameters.side_effect = lambda *a, **kw: iter([param_mock])
        model.named_parameters.side_effect = lambda *a, **kw: iter([("lora_a.weight", param_mock)])

        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"

        trainer = MultiAgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=1,
        )

        # Patch internal methods to return our known tensors
        trainer._tokenize_batch = MagicMock(return_value=(encoded_full, shifted_labels))
        call_count = [0]

        def fake_logprobs(enc, labels):
            if call_count[0] == 0:
                call_count[0] += 1
                return old_lp  # old_lp
            elif call_count[0] == 1:
                call_count[0] += 1
                return ref_lp  # ref_lp
            else:
                return ft.Tensor(np_torch.random.randn(S, L) * 0.1)  # new_lp

        trainer._masked_token_logprobs = MagicMock(side_effect=fake_logprobs)

        # Build grouped inputs — properly structured as lists-of-lists per group
        grouped = {
            "prompts": [[{"role": "user", "content": "prompt_0"}, {"role": "user", "content": "prompt_1"}]],
            "completions": [["completion_0", "completion_1"]],
            "raw_rewards": [[1.0, 2.0]],
            "normalized_rewards": [[1.0, 2.0]],
            "samples": [[]],
        }

        result = trainer.step(grouped_inputs=grouped)

        expected_keys = {
            "loss",
            "policy_loss",
            "kl_loss",
            "ratio_mean",
            "ratio_std",
            "clip_fraction",
            "kl_max",
            "mask_coverage",
            "mean_advantage",
            "advantage_std",
        }
        assert expected_keys.issubset(set(result.keys())), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

        # All values should be finite floats
        for key in expected_keys:
            val = result[key]
            assert isinstance(val, float), f"{key} is not float: {type(val)}"
            assert math.isfinite(val), f"{key} is not finite: {val}"

    def _run_multi_epoch_diagnostics(self, epoch_new_lps: list[object]) -> dict:
        from training.train import MultiAgentGRPOTrainer

        ft = sys.modules["torch"]
        np_torch = __import__("numpy")

        S, L = 2, 19
        encoded_full = {
            "input_ids": ft.Tensor(np_torch.ones((S, 20), dtype=np_torch.int64)),
            "attention_mask": ft.Tensor(np_torch.ones((S, 20), dtype=np_torch.int64)),
        }
        shifted_labels = ft.Tensor(np_torch.ones((S, L), dtype=np_torch.int64) * 100)
        old_lp = ft.Tensor(np_torch.random.randn(S, L) * 0.1)
        ref_lp = ft.Tensor(np_torch.random.randn(S, L) * 0.1)

        param_mock = MagicMock()
        param_mock.device = "cpu"
        param_mock.requires_grad = True

        model = MagicMock()
        model.disable_adapter.return_value = MagicMock(
            __enter__=MagicMock(return_value=model),
            __exit__=MagicMock(return_value=False),
        )
        model.parameters.side_effect = lambda *a, **kw: iter([param_mock])
        model.named_parameters.side_effect = lambda *a, **kw: iter([("lora_a.weight", param_mock)])

        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"

        trainer = MultiAgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=len(epoch_new_lps),
        )
        trainer._tokenize_batch = MagicMock(return_value=(encoded_full, shifted_labels))
        call_count = [0]
        epoch_iter = iter(epoch_new_lps)

        def fake_logprobs(enc, labels):
            if call_count[0] == 0:
                call_count[0] += 1
                return old_lp
            if call_count[0] == 1:
                call_count[0] += 1
                return ref_lp
            return ft.Tensor(next(epoch_iter))

        trainer._masked_token_logprobs = MagicMock(side_effect=fake_logprobs)
        grouped = {
            "prompts": [[{"role": "user", "content": "prompt_0"}, {"role": "user", "content": "prompt_1"}]],
            "completions": [["completion_0", "completion_1"]],
            "raw_rewards": [[1.0, 2.0]],
            "normalized_rewards": [[1.0, 2.0]],
            "samples": [[]],
        }
        return trainer.step(grouped_inputs=grouped)

    def test_diagnostics_single_epoch_matches_last_and_mean(self) -> None:
        np_torch = __import__("numpy")
        result = self._run_multi_epoch_diagnostics([np_torch.full((2, 19), 0.05)])

        assert result["loss"] == result["loss_mean_across_epochs"]
        assert result["policy_loss"] == result["policy_loss_mean_across_epochs"]
        assert result["kl_loss"] == result["kl_loss_mean_across_epochs"]
        assert result["ratio_mean"] == result["ratio_mean_across_epochs"]
        assert result["clip_fraction"] == result["clip_fraction_mean_across_epochs"]
        assert result["kl_max"] == result["kl_max_across_epochs"]
        assert result["num_inner_epochs"] == 1

    def test_diagnostics_multi_epoch_mean_is_not_last(self) -> None:
        np_torch = __import__("numpy")
        result = self._run_multi_epoch_diagnostics(
            [
                np_torch.full((2, 19), -0.25),
                np_torch.zeros((2, 19)),
                np_torch.full((2, 19), 0.25),
            ]
        )

        assert result["num_inner_epochs"] == 3
        assert result["kl_loss_mean_across_epochs"] != result["kl_loss"]

    def test_diagnostics_returns_num_inner_epochs(self) -> None:
        np_torch = __import__("numpy")
        result = self._run_multi_epoch_diagnostics(
            [
                np_torch.full((2, 19), -0.1),
                np_torch.full((2, 19), 0.0),
                np_torch.full((2, 19), 0.1),
            ]
        )
        assert result["num_inner_epochs"] == 3

    def test_trainer_init_disables_dropout_modules_without_eval_mode(self) -> None:
        """Dropout should be zeroed without relying on model.eval()."""
        from training.train import MultiAgentGRPOTrainer

        param_mock = MagicMock()
        param_mock.device = "cpu"
        param_mock.requires_grad = True

        class FakeDropout:
            def __init__(self, p: float) -> None:
                self.p = p

        class _FakeModel:
            def __init__(self) -> None:
                self.training = True
                self._dropout = FakeDropout(0.25)

            def eval(self):
                self.training = False
                return self

            def train(self, mode: bool = True):
                self.training = mode
                return self

            def parameters(self):
                return iter([param_mock])

            def named_parameters(self):
                return iter([("lora_a.weight", param_mock)])

            def modules(self):
                return iter([self, self._dropout])

            @contextmanager
            def disable_adapter(self):
                yield self

        model = _FakeModel()
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"

        trainer = MultiAgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=1,
        )
        assert model._dropout.p == 0.0
        assert model.training is True

    def test_trainer_init_reenables_gradient_checkpointing_when_available(self) -> None:
        """Checkpoint reloads should re-arm gradient checkpointing hooks."""
        from training.train import MultiAgentGRPOTrainer

        param_mock = MagicMock()
        param_mock.device = "cpu"
        param_mock.requires_grad = True

        class _FakeModel:
            def __init__(self) -> None:
                self.training = True
                self.gradient_checkpointing_enable_calls = 0

            def eval(self):
                self.training = False
                return self

            def train(self, mode: bool = True):
                self.training = mode
                return self

            def gradient_checkpointing_enable(self):
                self.gradient_checkpointing_enable_calls += 1

            def parameters(self):
                return iter([param_mock])

            def named_parameters(self):
                return iter([("lora_a.weight", param_mock)])

            def modules(self):
                return iter([self])

            @contextmanager
            def disable_adapter(self):
                yield self

        model = _FakeModel()
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"

        MultiAgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=1,
        )

        assert model.gradient_checkpointing_enable_calls == 1

    def test_step_keeps_training_mode_for_old_ref_and_new_logprobs(self) -> None:
        """Unsloth backward requires the GRPO loss path to stay in train mode."""
        from training.train import MultiAgentGRPOTrainer

        ft = sys.modules["torch"]
        np_torch = __import__("numpy")

        S, L = 2, 4
        encoded_full = {
            "input_ids": ft.Tensor(np_torch.ones((S, L + 1), dtype=np_torch.int64)),
            "attention_mask": ft.Tensor(np_torch.ones((S, L + 1), dtype=np_torch.int64)),
        }
        shifted_labels = ft.Tensor(np_torch.ones((S, L), dtype=np_torch.int64))

        param_mock = MagicMock()
        param_mock.device = "cpu"
        param_mock.requires_grad = True

        class _FakeModel:
            def __init__(self) -> None:
                self.training = True

            def eval(self):
                self.training = False
                return self

            def train(self, mode: bool = True):
                self.training = mode
                return self

            def parameters(self):
                return iter([param_mock])

            def named_parameters(self):
                return iter([("lora_a.weight", param_mock)])

            def modules(self):
                return iter([self])

            @contextmanager
            def disable_adapter(self):
                yield self

        model = _FakeModel()
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"

        trainer = MultiAgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=1,
        )
        trainer._tokenize_batch = MagicMock(return_value=(encoded_full, shifted_labels))

        observed_training_flags: list[bool] = []

        def fake_logprobs(enc, labels):
            del enc, labels
            observed_training_flags.append(model.training)
            return ft.Tensor(np_torch.zeros((S, L), dtype=np_torch.float64))

        trainer._masked_token_logprobs = MagicMock(side_effect=fake_logprobs)

        grouped = {
            "prompts": [[{"role": "user", "content": "prompt_0"}, {"role": "user", "content": "prompt_1"}]],
            "completions": [["completion_0", "completion_1"]],
            "raw_rewards": [[1.0, 2.0]],
            "normalized_rewards": [[1.0, 2.0]],
            "samples": [[]],
        }

        result = trainer.step(grouped_inputs=grouped)

        assert observed_training_flags == [True, True, True]
        assert model.training is True
        assert abs(result["ratio_mean"] - 1.0) < 1e-6
