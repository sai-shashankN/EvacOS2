from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from evacos_ma.schemas.multi_agent import BeliefAuditRow, StructuredBelief, Tier


class BeliefRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: str
    belief_id: str


class BeliefStore:
    """Per-episode belief registry and deterministic resolution engine."""

    def __init__(
        self,
        *,
        episode_id: str,
        seed: int,
        tier: str,
        disaster_family: str,
        generator_config_hash: str = "",
        belief_horizon_limit: int = 8,
    ) -> None:
        self.episode_id = episode_id
        self.seed = seed
        self.tier = Tier(tier)
        self.disaster_family = disaster_family
        self.generator_config_hash = generator_config_hash
        self.belief_horizon_limit = belief_horizon_limit
        self._beliefs: dict[str, StructuredBelief] = {}
        self._recent_resolutions: list[dict[str, Any]] = []
        self._last_score_by_predictor: dict[str, float] = {}

    def register(
        self,
        belief: StructuredBelief,
        predictor: str,
        slot_limit: int = 1,
    ) -> BeliefRegistration:
        if belief.horizon < 1 or belief.horizon > self.belief_horizon_limit:
            return BeliefRegistration(
                accepted=False,
                reason="horizon_out_of_range",
                belief_id=belief.belief_id,
            )
        if belief.belief_id in self._beliefs:
            return BeliefRegistration(
                accepted=False,
                reason="duplicate_belief_id",
                belief_id=belief.belief_id,
            )

        target_key = tuple(sorted(belief.target_entity_ids))
        pending_for_predictor = self.pending_beliefs(predictor)
        if any(tuple(sorted(existing.target_entity_ids)) == target_key for existing in pending_for_predictor):
            return BeliefRegistration(
                accepted=False,
                reason="duplicate_target",
                belief_id=belief.belief_id,
            )
        if len(pending_for_predictor) >= slot_limit:
            return BeliefRegistration(
                accepted=False,
                reason="slot_limit",
                belief_id=belief.belief_id,
            )

        stored = belief.model_copy(deep=True)
        stored.predictor_agent_id = predictor
        self._beliefs[stored.belief_id] = stored
        return BeliefRegistration(accepted=True, reason="ok", belief_id=stored.belief_id)

    def tick(
        self,
        current_round: int,
        ground_truth_provider: Callable[[StructuredBelief, int], dict[str, Any]],
    ) -> list[BeliefAuditRow]:
        audit_rows: list[BeliefAuditRow] = []
        for belief in sorted(self._beliefs.values(), key=lambda item: item.belief_id):
            if belief.resolved_round_or_null is not None:
                continue
            if belief.created_round + belief.horizon > current_round:
                continue

            score = _score_belief_payload(
                belief.prediction_payload,
                ground_truth_provider(belief, current_round),
            )
            belief.resolved_round_or_null = current_round
            self._last_score_by_predictor[belief.predictor_agent_id] = score
            self._recent_resolutions.append(
                {
                    "belief_id": belief.belief_id,
                    "score": score,
                    "round": current_round,
                }
            )
            self._recent_resolutions = self._recent_resolutions[-20:]
            audit_rows.append(
                BeliefAuditRow(
                    episode_id=self.episode_id,
                    round_id=current_round,
                    seed=self.seed,
                    tier=self.tier,
                    disaster_family=self.disaster_family,
                    generator_config_hash=self.generator_config_hash,
                    belief_id=belief.belief_id,
                    predictor_agent_id=belief.predictor_agent_id,
                    confidence=belief.confidence,
                    resolved=True,
                    score=score,
                )
            )
        return audit_rows

    def open_slots(self, predictor: str, slot_limit: int) -> int:
        return max(0, slot_limit - len(self.pending_beliefs(predictor)))

    def pending_beliefs(self, predictor: str) -> list[StructuredBelief]:
        return [
            belief
            for belief in self._beliefs.values()
            if belief.predictor_agent_id == predictor and belief.resolved_round_or_null is None
        ]

    def snapshot(self) -> dict[str, Any]:
        beliefs = list(self._beliefs.values())
        resolved = [belief for belief in beliefs if belief.resolved_round_or_null is not None]
        avg_confidence = sum(belief.confidence for belief in beliefs) / len(beliefs) if beliefs else 0.0
        return {
            "total_beliefs": len(beliefs),
            "avg_confidence": avg_confidence,
            "resolved_count": len(resolved),
            "pending_count": len(beliefs) - len(resolved),
            "recent_highlights": list(reversed(self._recent_resolutions[-3:])),
        }

    def last_score(self, predictor: str) -> float:
        return self._last_score_by_predictor.get(predictor, 0.0)

    def beliefs(self) -> list[StructuredBelief]:
        return [belief.model_copy(deep=True) for belief in self._beliefs.values()]


def _score_belief_payload(
    prediction_payload: dict[str, Any],
    ground_truth: dict[str, Any],
) -> float:
    if not prediction_payload:
        return 0.0

    scores: list[float] = []
    for key, predicted in prediction_payload.items():
        actual = ground_truth.get(key)
        if key == "expected_civilians_in_room":
            actual_value = float(actual or 0.0)
            predicted_value = float(predicted)
            scores.append(1.0 - min(1.0, abs(predicted_value - actual_value) / 10.0))
        elif key == "expected_hazard_severity_room":
            actual_value = float(actual or 0.0)
            predicted_value = float(predicted)
            scores.append(1.0 - min(1.0, abs(predicted_value - actual_value) / 1.0))
        elif key == "expected_room_passable":
            scores.append(1.0 if bool(predicted) == bool(actual) else 0.0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)
