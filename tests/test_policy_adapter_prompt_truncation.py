from __future__ import annotations

import sys
import types

import training.policy_adapter as policy_adapter
from training.policy_adapter import UnslothPolicy, hf_policy_factory


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _FakeTensor(self._values[item])
        return self._values[item]

    def tolist(self):
        return list(self._values)

    def __iter__(self):
        return iter(self._values)


class _FakeMatrix:
    def __init__(self, rows):
        self._rows = [_FakeTensor(row) for row in rows]
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __iter__(self):
        return iter(self._rows)

    def tolist(self):
        return [row.tolist() for row in self._rows]


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeHFTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"
        self.last_encoded_rows = None

    def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return prompt[-1]["content"]

    @staticmethod
    def _encode_text(text):
        return [ord(ch) for ch in text]

    def __call__(
        self,
        rendered,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        del add_special_tokens
        texts = rendered if isinstance(rendered, list) else [rendered]
        rows = [self._encode_text(text) for text in texts]
        if truncation and max_length is not None:
            rows = [row[-max_length:] for row in rows]
        if padding and rows:
            width = max(len(row) for row in rows)
            rows = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
        self.last_encoded_rows = [list(row) for row in rows]
        if return_tensors == "pt":
            return _FakeBatch(
                {
                    "input_ids": _FakeMatrix(rows),
                    "attention_mask": _FakeMatrix(
                        [[0 if token == self.pad_token_id else 1 for token in row] for row in rows]
                    ),
                }
            )
        if isinstance(rendered, list):
            return {"input_ids": rows}
        return {"input_ids": rows[0]}

    @staticmethod
    def decode(generated, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token) for token in generated.tolist() if token != 0)


class _FakeHFModel:
    def __init__(self):
        self.training = True
        self.device = "cpu"
        self.last_generate_kwargs = None

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        rows = kwargs["input_ids"].tolist()
        return _FakeMatrix([row + [126] for row in rows])


class _FakeAutoTokenizer:
    last_instance = None

    @classmethod
    def from_pretrained(cls, model_name):
        del model_name
        cls.last_instance = _FakeHFTokenizer()
        return cls.last_instance


class _FakeAutoModelForCausalLM:
    last_instance = None

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        del model_name, kwargs
        cls.last_instance = _FakeHFModel()
        return cls.last_instance


class _FakeFastLanguageModel:
    @staticmethod
    def for_inference(model):
        model.eval()

    @staticmethod
    def for_training(model):
        model.train()


def _install_fake_hf_stack(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
    fake_torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    fake_transformers.AutoModelForCausalLM = _FakeAutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


class TestHFPromptTruncation:
    def test_hf_generate_truncates_long_prompt_from_left(self, monkeypatch):
        _install_fake_hf_stack(monkeypatch)
        monkeypatch.setattr(policy_adapter, "_PROMPT_TRUNCATION_WARNED", False)
        warning_calls = []
        monkeypatch.setattr(policy_adapter.logger, "warning", lambda *args, **kwargs: warning_calls.append((args, kwargs)))

        policy = hf_policy_factory("fake-model", max_prompt_tokens=10, max_new_tokens=1, do_sample=False)
        long_text = "0123456789abcdef"
        completion, _ = policy.act(
            [{"role": "user", "content": long_text}],
            agent_id="floor_0_agent",
            role="floor_agent",
        )

        tokenizer = _FakeAutoTokenizer.last_instance
        model = _FakeAutoModelForCausalLM.last_instance
        expected_tail = [ord(ch) for ch in long_text[-10:]]
        encoded_tail = model.last_generate_kwargs["input_ids"].tolist()[0]

        assert tokenizer.padding_side == "left"
        assert tokenizer.truncation_side == "left"
        assert len(encoded_tail) == 10
        assert encoded_tail == expected_tail
        assert encoded_tail[-1] == ord(long_text[-1])
        assert completion == "~"
        assert len(warning_calls) == 1

    def test_hf_generate_passes_through_short_prompt(self, monkeypatch):
        _install_fake_hf_stack(monkeypatch)
        monkeypatch.setattr(policy_adapter, "_PROMPT_TRUNCATION_WARNED", False)
        monkeypatch.setattr(policy_adapter.logger, "warning", lambda *args, **kwargs: None)

        policy = hf_policy_factory("fake-model", max_prompt_tokens=10, max_new_tokens=1, do_sample=False)
        short_text = "short"
        policy.act(
            [{"role": "user", "content": short_text}],
            agent_id="floor_0_agent",
            role="floor_agent",
        )

        model = _FakeAutoModelForCausalLM.last_instance
        encoded = model.last_generate_kwargs["input_ids"].tolist()[0]
        assert encoded == [ord(ch) for ch in short_text]

    def test_truncation_warning_fires_once_per_process(self, monkeypatch):
        _install_fake_hf_stack(monkeypatch)
        monkeypatch.setattr(policy_adapter, "_PROMPT_TRUNCATION_WARNED", False)
        warning_calls = []
        monkeypatch.setattr(policy_adapter.logger, "warning", lambda *args, **kwargs: warning_calls.append((args, kwargs)))

        policy = hf_policy_factory("fake-model", max_prompt_tokens=4, max_new_tokens=1, do_sample=False)
        prompt = [{"role": "user", "content": "abcdefgh"}]

        policy.act(prompt, agent_id="floor_0_agent", role="floor_agent")
        policy.act(prompt, agent_id="floor_0_agent", role="floor_agent")

        assert len(warning_calls) == 1


class TestUnslothPromptTruncation:
    def test_unsloth_hf_generate_truncates_long_prompt_from_left(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        fake_unsloth = types.ModuleType("unsloth")
        fake_unsloth.FastLanguageModel = _FakeFastLanguageModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
        monkeypatch.setattr(policy_adapter, "_PROMPT_TRUNCATION_WARNED", False)
        monkeypatch.setattr(policy_adapter.logger, "warning", lambda *args, **kwargs: None)

        policy = UnslothPolicy.__new__(UnslothPolicy)
        policy._tokenizer = _FakeHFTokenizer()
        policy._tokenizer.padding_side = "left"
        policy._tokenizer.truncation_side = "left"
        policy._model = _FakeHFModel()
        policy._max_new_tokens = 1
        policy._max_prompt_tokens = 6
        policy._temperature = 0.0

        outputs = policy._hf_generate(["abcdefghijkl"])
        encoded = policy._model.last_generate_kwargs["input_ids"].tolist()[0]

        assert len(encoded) == 6
        assert encoded == [ord(ch) for ch in "ghijkl"]
        assert encoded[-1] == ord("l")
        assert outputs == [("~", [126])]
