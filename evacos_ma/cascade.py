"""Generalized staged cascade scheduler for long-horizon episodes.

Provides a first-class CascadeScheduler that manages a sequence of CascadeStage
objects, each specifying when (trigger_step) and what (disaster_kind + params)
should activate. The scheduler delegates to existing disaster engines for actual
spread logic — it does NOT re-implement any spread mechanics.

Determinism contract: given the same {seed, task_id}, the scheduler produces
identical cascade activation sequences across runs.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, Field

from evacos_ma.models import (
    Building,
    EventSummary,
    EventType,
    ScheduledEvent,
)


class CascadeStage(BaseModel):
    """A single stage in a cascade schedule."""

    stage_id: str
    trigger_step: int
    disaster_kind: str  # "fire", "flood", "gas", "structural", "stairwell_collapse"
    params: dict[str, Any] = Field(default_factory=dict)
    rng_substream_key: str = ""


class CascadeScheduler:
    """Manages staged cascade events for long-horizon tasks.

    Works for both legacy short-horizon tasks (empty schedule) and new
    task_lh_* tasks with multi-stage cascades.
    """

    def __init__(
        self,
        stages: list[CascadeStage],
        seed: int,
    ) -> None:
        self._stages = list(stages)
        self._seed = seed
        # Deterministic per-stage RNGs derived from {seed, substream_key}
        self._stage_rngs: dict[str, random.Random] = {}
        for stage in self._stages:
            sub_key = stage.rng_substream_key or stage.stage_id
            # Hash seed + substream_key for deterministic but independent RNG
            combined = hash(f"{seed}:{sub_key}") & 0xFFFFFFFF
            self._stage_rngs[stage.stage_id] = random.Random(combined)

    def step(
        self,
        current_step: int,
        scheduled_events: list[ScheduledEvent],
    ) -> list[EventSummary]:
        """Check which stages trigger at current_step and return EventSummary list.

        Does NOT mutate episode state directly — it marks triggered ScheduledEvents
        and returns event summaries. The caller (env) applies the effects through
        existing disaster engines.

        Args:
            current_step: The current episode step (1-based after first step).
            scheduled_events: The mutable list of scheduled events to mark triggered.

        Returns:
            List of EventSummary for stages that fired this step.
        """
        events: list[EventSummary] = []
        for stage in self._stages:
            if stage.trigger_step != current_step:
                continue
            # Find matching scheduled event if one exists
            for se in scheduled_events:
                if se.triggered or se.trigger_step != current_step:
                    continue
                # Match by event_type or event_id prefix
                if self._stage_matches_event(stage, se):
                    se.triggered = True
                    events.append(
                        EventSummary(
                            event_id=se.event_id,
                            event_type=se.event_type,
                            target_id=se.target_id,
                            description=f"Cascade stage '{stage.stage_id}' triggered at step {current_step}",
                        )
                    )
                    break
        return events

    def _stage_matches_event(self, stage: CascadeStage, se: ScheduledEvent) -> bool:
        """Check if a CascadeStage corresponds to a ScheduledEvent."""
        kind_to_event_type = {
            "fire": EventType.fire_ignition,
            "flood": EventType.flood_rise,
            "gas": EventType.gas_rupture,
            "structural": EventType.structural_collapse,
            "stairwell_collapse": EventType.stairwell_collapse,
        }
        expected_type = kind_to_event_type.get(stage.disaster_kind)
        if expected_type is not None and se.event_type == expected_type:
            return True
        # Also match by stage_id prefix in event_id
        return stage.stage_id in se.event_id

    def get_stage_rng(self, stage_id: str) -> random.Random:
        """Return the deterministic RNG for a given stage."""
        return self._stage_rngs[stage_id]

    @property
    def stages(self) -> list[CascadeStage]:
        return list(self._stages)

    def triggered_stage_ids(self, scheduled_events: list[ScheduledEvent]) -> list[str]:
        """Return IDs of stages that have been triggered based on scheduled_events state."""
        triggered = []
        for stage in self._stages:
            for se in scheduled_events:
                if se.triggered and self._stage_matches_event(stage, se):
                    triggered.append(stage.stage_id)
                    break
        return triggered


def build_scheduled_events_from_stages(
    stages: list[CascadeStage],
    building: Building,
    seed: int,
) -> list[ScheduledEvent]:
    """Convert CascadeStage list to ScheduledEvent list for legacy disaster engines.

    This bridges the new CascadeStage configs with the existing ScheduledEvent
    model that MultiCascade expects.
    """
    rng = random.Random(seed)
    events: list[ScheduledEvent] = []

    kind_to_event_type = {
        "fire": EventType.fire_ignition,
        "flood": EventType.flood_rise,
        "gas": EventType.gas_rupture,
        "structural": EventType.structural_collapse,
        "stairwell_collapse": EventType.stairwell_collapse,
    }

    all_rooms = [room for floor in building.floors for room in floor.rooms]
    floor_map = {floor.floor_id: floor for floor in building.floors}

    for stage in stages:
        event_type = kind_to_event_type.get(stage.disaster_kind, EventType.explosion)

        # Determine target room for the event
        target_id = stage.params.get("target_id")
        if target_id is None:
            # Pick a deterministic room based on disaster kind
            if stage.disaster_kind == "gas":
                # Gas typically starts on upper floors
                candidate_floors = sorted(floor_map.keys(), reverse=True)
                for fid in candidate_floors:
                    floor_rooms = floor_map[fid].rooms
                    if floor_rooms:
                        target_id = sorted(r.room_id for r in floor_rooms)[0]
                        break
            elif stage.disaster_kind in ("structural", "stairwell_collapse"):
                target_id = "structural_activation"
            elif stage.disaster_kind == "flood":
                target_id = "F0_R0"
            else:
                # Fire: pick from middle floors
                if all_rooms:
                    target_id = rng.choice(sorted(all_rooms, key=lambda r: r.room_id)).room_id

        if target_id is None:
            target_id = stage.stage_id

        payload = dict(stage.params)
        if "origin_room_id" not in payload and target_id.startswith("F"):
            payload["origin_room_id"] = target_id

        events.append(
            ScheduledEvent(
                event_id=f"cascade_{stage.stage_id}",
                trigger_step=stage.trigger_step,
                event_type=event_type,
                target_id=target_id,
                payload=payload,
            )
        )

    return events
