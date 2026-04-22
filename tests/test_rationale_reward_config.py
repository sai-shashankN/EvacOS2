import json
import math
import random
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from evacos_ma import round_protocol as round_protocol_mod
from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from evacos_ma.schemas.multi_agent import BeliefAuditRow
from training.reward import RewardNormalizer
from training.rollout import collect_episode


def _tmp_logs_dir() -> Path:
    root = Path(tempfile.gettempdir()) / f"evacos_rationale_logs_{random.randint(0, 999999)}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _rationale(token_count: int) -> str:
    return " ".join(f"token{idx}" for idx in range(token_count))


def _episode_id(prompt: list[dict[str, str]]) -> str:
    system_msg = next(msg for msg in prompt if msg["role"] == "system")
    match = re.search(r'"episode_id"\s*:\s*"([^"]+)"', system_msg["content"])
    return match.group(1) if match else "ep_test"


def _round_id(prompt: list[dict[str, str]]) -> int:
    system_msg = next(msg for msg in prompt if msg["role"] == "system")
    match = re.search(r"Round:\s*(\d+)", system_msg["content"])
    return int(match.group(1)) if match else 0


def _reward_rows(logs_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (logs_dir / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reward_breakdown(logs_dir: Path, agent_id: str) -> dict[str, float]:
    row = next(item for item in _reward_rows(logs_dir) if item["agent_id"] == agent_id)
    return row["breakdown"]


def _seeded_normalizer() -> RewardNormalizer:
    normalizer = RewardNormalizer()
    normalizer.load_snapshot(
        {
            "orchestrator:easy": {"count": 40, "mean": 0.0, "m2": 0.4},
            "floor_agent:easy": {"count": 40, "mean": 0.0, "m2": 0.4},
        }
    )
    return normalizer


class OverridePolicy:
    def __init__(self, rationale: str) -> None:
        self._rationale = rationale

    def act(self, prompt: list[dict[str, str]], agent_id: str, role: str) -> str:
        episode_id = _episode_id(prompt)
        round_id = _round_id(prompt)
        if role == "orchestrator":
            return json.dumps(
                {
                    "episode_id": episode_id,
                    "round_id": round_id,
                    "agent_id": "orchestrator",
                    "action_id": "orch_override",
                    "action_type": "override_floor_agent",
                    "arguments": {
                        "target_floor_agent_id": "floor_0_agent",
                        "replacement_action_type": "wait",
                        "replacement_arguments": {},
                    },
                    "rationale": self._rationale,
                }
            )
        return json.dumps(
            {
                "episode_id": episode_id,
                "round_id": round_id,
                "agent_id": agent_id,
                "action_id": f"wait_{agent_id}",
                "action_type": "wait",
                "arguments": {},
                "rationale": "waiting for orchestrator",
            }
        )


class PredictStatePolicy:
    def __init__(self, rationale: str) -> None:
        self._rationale = rationale

    def act(self, prompt: list[dict[str, str]], agent_id: str, role: str) -> str:
        episode_id = _episode_id(prompt)
        round_id = _round_id(prompt)
        if role == "orchestrator":
            return json.dumps(
                {
                    "episode_id": episode_id,
                    "round_id": round_id,
                    "agent_id": "orchestrator",
                    "action_id": "orch_wait",
                    "action_type": "wait",
                    "arguments": {},
                    "rationale": "orchestrator waiting",
                }
            )
        if agent_id == "floor_0_agent":
            return json.dumps(
                {
                    "episode_id": episode_id,
                    "round_id": round_id,
                    "agent_id": agent_id,
                    "action_id": "floor_predict",
                    "action_type": "predict_state",
                    "arguments": {
                        "belief": {
                            "belief_id": "belief_floor_0",
                            "predictor_agent_id": agent_id,
                            "target_entity_ids": ["F0_R0"],
                            "horizon": 1,
                            "prediction_payload": {"expected_civilians_in_room": 1},
                            "confidence": 0.9,
                            "justification": "focused forecast",
                        }
                    },
                    "rationale": self._rationale,
                }
            )
        return json.dumps(
            {
                "episode_id": episode_id,
                "round_id": round_id,
                "agent_id": agent_id,
                "action_id": f"wait_{agent_id}",
                "action_type": "wait",
                "arguments": {},
                "rationale": "waiting on current floor",
            }
        )


def test_override_linear_capped_bonus_populates_metrics(monkeypatch):
    monkeypatch.setattr(round_protocol_mod, "_compute_counterfactual_delta", lambda *args, **kwargs: 7.25)
    logs_dir = _tmp_logs_dir()
    try:
        result = collect_episode(
            EvacEnvironment(),
            OverridePolicy(_rationale(20)),
            seed=21,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=1,
            jsonl_dir=logs_dir,
            reward_config={
                "rationale_scaling": "linear_capped",
                "alpha": 0.05,
                "beta": 0.25,
                "cap": 0.5,
                "eligible_token_ceiling": 160,
                "clip_normalized_to": 1.0,
            },
        )
        breakdown = _reward_breakdown(logs_dir, "orchestrator")
        assert breakdown["rationale_bonus"] == pytest.approx(0.5)
        assert result.rationale_bonus_total == pytest.approx(0.5)
        assert result.rationale_bonus_count == 1
    finally:
        shutil.rmtree(logs_dir, ignore_errors=True)


def test_override_off_mode_awards_zero_bonus(monkeypatch):
    monkeypatch.setattr(round_protocol_mod, "_compute_counterfactual_delta", lambda *args, **kwargs: 7.25)
    logs_dir = _tmp_logs_dir()
    try:
        result = collect_episode(
            EvacEnvironment(),
            OverridePolicy(_rationale(20)),
            seed=22,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=1,
            jsonl_dir=logs_dir,
            reward_config={
                "rationale_scaling": "off",
                "alpha": 0.05,
                "beta": 0.25,
                "cap": 1.0,
                "eligible_token_ceiling": 160,
                "clip_normalized_to": 1.0,
            },
        )
        breakdown = _reward_breakdown(logs_dir, "orchestrator")
        assert breakdown.get("rationale_bonus", 0.0) == 0.0
        assert result.rationale_bonus_total == 0.0
        assert result.rationale_bonus_count == 0
    finally:
        shutil.rmtree(logs_dir, ignore_errors=True)


def test_override_log_uncapped_uses_beta_log_formula(monkeypatch):
    monkeypatch.setattr(round_protocol_mod, "_compute_counterfactual_delta", lambda *args, **kwargs: 7.25)
    logs_dir = _tmp_logs_dir()
    try:
        collect_episode(
            EvacEnvironment(),
            OverridePolicy(_rationale(14)),
            seed=23,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=1,
            jsonl_dir=logs_dir,
            reward_config={
                "rationale_scaling": "log_uncapped",
                "alpha": 0.05,
                "beta": 0.25,
                "cap": 10.0,
                "eligible_token_ceiling": 160,
                "clip_normalized_to": 1.0,
            },
        )
        breakdown = _reward_breakdown(logs_dir, "orchestrator")
        assert breakdown["rationale_bonus"] == pytest.approx(0.25 * math.log(15))
    finally:
        shutil.rmtree(logs_dir, ignore_errors=True)


def test_override_bonus_respects_eligible_token_ceiling(monkeypatch):
    monkeypatch.setattr(round_protocol_mod, "_compute_counterfactual_delta", lambda *args, **kwargs: 7.25)
    logs_dir = _tmp_logs_dir()
    try:
        collect_episode(
            EvacEnvironment(),
            OverridePolicy(_rationale(40)),
            seed=24,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=1,
            jsonl_dir=logs_dir,
            reward_config={
                "rationale_scaling": "linear_capped",
                "alpha": 0.1,
                "beta": 0.25,
                "cap": 10.0,
                "eligible_token_ceiling": 13,
                "clip_normalized_to": 1.0,
            },
        )
        breakdown = _reward_breakdown(logs_dir, "orchestrator")
        assert breakdown["rationale_bonus"] == pytest.approx(1.3)
    finally:
        shutil.rmtree(logs_dir, ignore_errors=True)


def test_clip_normalized_to_is_respected(monkeypatch):
    monkeypatch.setattr(round_protocol_mod, "_compute_counterfactual_delta", lambda *args, **kwargs: 7.25)
    logs_dir = _tmp_logs_dir()
    try:
        result = collect_episode(
            EvacEnvironment(),
            OverridePolicy(_rationale(20)),
            seed=25,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=1,
            jsonl_dir=logs_dir,
            normalizer=_seeded_normalizer(),
            update_normalizer=False,
            reward_config={
                "rationale_scaling": "linear_capped",
                "alpha": 0.1,
                "beta": 0.25,
                "cap": 10.0,
                "eligible_token_ceiling": 160,
                "clip_normalized_to": 0.25,
            },
        )
        assert result.samples
        assert all(abs(sample.normalized_reward) <= 0.25 + 1e-9 for sample in result.samples)
        assert any(abs(sample.normalized_reward) == pytest.approx(0.25) for sample in result.samples)
    finally:
        shutil.rmtree(logs_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("belief_score", "expected_bonus"),
    [
        (0.75, 0.9),
        (0.49, 0.0),
    ],
)
def test_predict_state_bonus_requires_belief_score_threshold(monkeypatch, belief_score, expected_bonus):
    def fake_resolve_beliefs(self, ep):
        row = BeliefAuditRow(
            episode_id=ep.episode_id,
            round_id=ep.step,
            seed=ep.seed,
            tier=self._episode_tier_value(ep),
            disaster_family=ep.task.disaster_type.value,
            generator_config_hash="",
            belief_id="belief_floor_0",
            predictor_agent_id="floor_0_agent",
            confidence=0.9,
            resolved=True,
            score=belief_score,
        )
        ep.last_prediction_score_by_agent["floor_0_agent"] = belief_score
        ep.belief_audit_log.append(row.model_dump(mode="json"))
        return [row]

    monkeypatch.setattr(EvacEnvironment, "_resolve_beliefs", fake_resolve_beliefs)
    logs_dir = _tmp_logs_dir()
    try:
        result = collect_episode(
            EvacEnvironment(),
            PredictStatePolicy(_rationale(18)),
            seed=26,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=1,
            jsonl_dir=logs_dir,
            reward_config={
                "rationale_scaling": "linear_capped",
                "alpha": 0.05,
                "beta": 0.25,
                "cap": 10.0,
                "eligible_token_ceiling": 160,
                "clip_normalized_to": 1.0,
            },
        )
        breakdown = _reward_breakdown(logs_dir, "floor_0_agent")
        assert breakdown.get("rationale_bonus", 0.0) == pytest.approx(expected_bonus)
        expected_count = 1 if expected_bonus > 0.0 else 0
        assert result.rationale_bonus_count == expected_count
    finally:
        shutil.rmtree(logs_dir, ignore_errors=True)
