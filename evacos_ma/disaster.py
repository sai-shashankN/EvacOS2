from __future__ import annotations

import random
from collections import defaultdict

from evacos_ma.models import (
    Building,
    Corridor,
    DisasterType,
    Elevator,
    EventSummary,
    EventType,
    Floor,
    HazardState,
    HazardType,
    Occupancy,
    Room,
    ScheduledEvent,
    Stairwell,
    ThreatState,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _remove_occupants(source: Occupancy, count: int) -> Occupancy:
    """Remove up to `count` civilians, prioritizing the most vulnerable first."""
    removed = Occupancy()
    remaining = max(0, count)
    for field in ("injured", "mobility_impaired", "mobile"):
        available = getattr(source, field)
        taken = min(available, remaining)
        setattr(source, field, available - taken)
        setattr(removed, field, taken)
        remaining -= taken
        if remaining == 0:
            break
    return removed


class DisasterEngine:
    """Base class for all disaster engines."""

    def __init__(
        self,
        building: Building,
        origin_room_id: str,
        seed: int,
        *,
        scheduled_events: list[ScheduledEvent] | None = None,
        live_projection: bool = True,
    ) -> None:
        self.building = building
        self.origin_room_id = origin_room_id
        self.rng = random.Random(seed)
        self._live_projection = live_projection
        self.scheduled_events = [
            event.model_copy(deep=True)
            for event in (scheduled_events or [])
        ]
        self._room_lookup: dict[str, Room] = {}
        self._corridor_lookup: dict[str, Corridor] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._edge_types: dict[tuple[str, str], str] = {}
        self._floor_lookup: dict[int, Floor] = {}
        self._pair_to_corridor: dict[frozenset[str], str] = {}
        self._corridors_by_room: dict[str, list[str]] = defaultdict(list)
        self._stairwell_lookup: dict[str, list[Stairwell]] = defaultdict(list)
        self._elevator_lookup: dict[str, list[Elevator]] = defaultdict(list)
        self._baseline_room_hazards: dict[str, HazardState] = {}
        self._baseline_room_accessible: dict[str, bool] = {}
        self._baseline_corridor_hazards: dict[str, HazardState] = {}
        self._baseline_stairwell_status: dict[str, bool] = {}
        self._baseline_elevator_status: dict[str, bool] = {}
        self._room_states: dict[str, HazardState] = {}
        self._corridor_states: dict[str, HazardState] = {}
        self._room_accessibility: dict[str, bool] = {}
        self._stairwell_status: dict[str, bool] = {}
        self._elevator_status: dict[str, bool] = {}
        self._build_lookups()

    def _build_lookups(self) -> None:
        self._adjacency = {}
        for floor in self.building.floors:
            self._floor_lookup[floor.floor_id] = floor
            for room in floor.rooms:
                self._room_lookup[room.room_id] = room
                self._adjacency.setdefault(room.room_id, [])
                self._baseline_room_hazards[room.room_id] = room.hazard.model_copy(deep=True)
                self._baseline_room_accessible[room.room_id] = room.accessible
            for corridor in floor.corridors:
                self._corridor_lookup.setdefault(corridor.corridor_id, corridor)
                self._baseline_corridor_hazards.setdefault(
                    corridor.corridor_id,
                    corridor.hazard.model_copy(deep=True),
                )
                pair = frozenset((corridor.from_node_id, corridor.to_node_id))
                self._pair_to_corridor[pair] = corridor.corridor_id
                self._corridors_by_room[corridor.from_node_id].append(corridor.corridor_id)
                self._corridors_by_room[corridor.to_node_id].append(corridor.corridor_id)
            for stairwell in floor.stairwells:
                self._stairwell_lookup[stairwell.stairwell_id].append(stairwell)
            for elevator in floor.elevators:
                self._elevator_lookup[elevator.elevator_id].append(elevator)

        for stairwell_id, stairwells in self._stairwell_lookup.items():
            self._baseline_stairwell_status[stairwell_id] = stairwells[0].blocked
            self._stairwell_status[stairwell_id] = stairwells[0].blocked
        for elevator_id, elevators in self._elevator_lookup.items():
            self._baseline_elevator_status[elevator_id] = elevators[0].operational
            self._elevator_status[elevator_id] = elevators[0].operational

        for edge in self.building.graph_edges:
            if edge.from_id not in self._room_lookup or edge.to_id not in self._room_lookup:
                continue
            self._adjacency.setdefault(edge.from_id, []).append(edge.to_id)
            self._edge_types[(edge.from_id, edge.to_id)] = edge.edge_type

        for room_id in self._adjacency:
            self._adjacency[room_id] = sorted(set(self._adjacency[room_id]))
        for room_id in self._corridors_by_room:
            self._corridors_by_room[room_id] = sorted(set(self._corridors_by_room[room_id]))

    def advance(self, step: int) -> list[EventSummary]:
        """Advance disaster by one step. Returns events that occurred."""
        raise NotImplementedError

    def _activate_scheduled_event(
        self,
        scheduled_event: ScheduledEvent,
        step: int,
    ) -> list[EventSummary]:
        return []

    def _advance_scheduled_events(self, step: int) -> list[EventSummary]:
        events: list[EventSummary] = []
        for scheduled_event in self.scheduled_events:
            if scheduled_event.triggered or step < scheduled_event.trigger_step:
                continue
            scheduled_event.triggered = True
            events.extend(self._activate_scheduled_event(scheduled_event, step))
        return events

    def get_hazard_map(self) -> dict[str, float]:
        """Return room_id -> hazard severity for all affected rooms."""
        return {
            room_id: state.severity
            for room_id, state in self._room_states.items()
            if state.severity > 0
        }

    def _make_event(
        self,
        prefix: str,
        step: int,
        event_type: EventType,
        target_id: str,
        description: str,
        *,
        event_id: str | None = None,
    ) -> EventSummary:
        return EventSummary(
            event_id=event_id or f"{prefix}_{step}_{target_id}",
            event_type=event_type,
            target_id=target_id,
            description=description,
        )

    def _new_state(self, room_id: str | None = None) -> HazardState:
        if room_id is not None and room_id in self._room_states:
            return self._room_states[room_id].model_copy(deep=True)
        return HazardState()

    def _normalize_state(self, state: HazardState) -> HazardState:
        return HazardState(
            hazard_type=state.hazard_type,
            severity=_clamp01(state.severity),
            smoke=_clamp01(state.smoke),
            water_level=_clamp01(state.water_level),
            structural_integrity=_clamp01(state.structural_integrity),
            passable=state.passable,
        )

    def _set_room_state(
        self,
        room_id: str,
        state: HazardState,
        *,
        accessible: bool | None = None,
    ) -> None:
        self._room_states[room_id] = self._normalize_state(state)
        if accessible is not None:
            self._room_accessibility[room_id] = accessible
        if self._live_projection:
            self._write_room_projection(room_id)

    def _update_room_state(
        self,
        room_id: str,
        *,
        hazard_type: HazardType | None = None,
        severity: float | None = None,
        smoke: float | None = None,
        water_level: float | None = None,
        structural_integrity: float | None = None,
        passable: bool | None = None,
        accessible: bool | None = None,
    ) -> None:
        current = self._new_state(room_id)
        self._set_room_state(
            room_id,
            HazardState(
                hazard_type=current.hazard_type if hazard_type is None else hazard_type,
                severity=current.severity if severity is None else severity,
                smoke=current.smoke if smoke is None else smoke,
                water_level=current.water_level if water_level is None else water_level,
                structural_integrity=(
                    current.structural_integrity
                    if structural_integrity is None
                    else structural_integrity
                ),
                passable=current.passable if passable is None else passable,
            ),
            accessible=accessible,
        )

    def _clear_room_state(self, room_id: str) -> None:
        self._room_states.pop(room_id, None)
        self._room_accessibility.pop(room_id, None)
        if self._live_projection:
            baseline = self._baseline_room_hazards.get(room_id, HazardState())
            room = self._room_lookup[room_id]
            room.hazard = baseline.model_copy(deep=True)
            room.accessible = self._baseline_room_accessible.get(room_id, True)

    def _set_corridor_state(self, corridor_id: str, state: HazardState) -> None:
        self._corridor_states[corridor_id] = self._normalize_state(state)
        if self._live_projection:
            corridor = self._corridor_lookup.get(corridor_id)
            if corridor is not None:
                corridor.hazard = self._normalize_state(state)

    def _set_stairwell_blocked(self, stairwell_id: str, blocked: bool) -> None:
        self._stairwell_status[stairwell_id] = blocked
        if self._live_projection:
            for stairwell in self._stairwell_lookup.get(stairwell_id, []):
                stairwell.blocked = blocked

    def _set_elevator_operational(self, elevator_id: str, operational: bool) -> None:
        self._elevator_status[elevator_id] = operational
        if self._live_projection:
            for elevator in self._elevator_lookup.get(elevator_id, []):
                elevator.operational = operational

    def _write_room_projection(self, room_id: str) -> None:
        state = self._room_states.get(room_id)
        if state is None:
            return
        room = self._room_lookup.get(room_id)
        if room is not None:
            room.hazard = state.model_copy(deep=True)
            accessible = self._room_accessibility.get(room_id)
            if accessible is not None:
                room.accessible = accessible

    def _project_all(self) -> None:
        if not self._live_projection:
            return
        for room_id in self._room_states:
            self._write_room_projection(room_id)
        for corridor_id, state in self._corridor_states.items():
            corridor = self._corridor_lookup.get(corridor_id)
            if corridor is not None:
                corridor.hazard = state.model_copy(deep=True)
        for stairwell_id, blocked in self._stairwell_status.items():
            for stairwell in self._stairwell_lookup.get(stairwell_id, []):
                stairwell.blocked = blocked
        for elevator_id, operational in self._elevator_status.items():
            for elevator in self._elevator_lookup.get(elevator_id, []):
                elevator.operational = operational

    def _edge_type(self, from_room_id: str, to_room_id: str) -> str:
        return self._edge_types.get((from_room_id, to_room_id), "corridor")

    def _pick_room_on_floor(self, floor_id: int) -> str:
        floor = self._floor_lookup.get(floor_id)
        if floor is None:
            floor = self._floor_lookup[max(self._floor_lookup)]
        return sorted(room.room_id for room in floor.rooms)[0]


class FireSpread(DisasterEngine):
    def __init__(
        self,
        building: Building,
        origin_room_id: str,
        seed: int,
        *,
        scheduled_events: list[ScheduledEvent] | None = None,
        live_projection: bool = True,
    ) -> None:
        super().__init__(
            building,
            origin_room_id,
            seed,
            scheduled_events=scheduled_events,
            live_projection=live_projection,
        )
        self._burning_rooms: set[str] = {
            room_id
            for room_id, room in self._room_lookup.items()
            if room.hazard.hazard_type == HazardType.fire and room.hazard.severity > 0
        }
        self._burning_rooms.add(origin_room_id)
        for room_id in sorted(self._burning_rooms):
            baseline = self._room_lookup[room_id].hazard
            self._set_room_state(
                room_id,
                HazardState(
                    hazard_type=HazardType.fire,
                    severity=max(
                        baseline.severity,
                        0.25 if room_id == origin_room_id else baseline.severity,
                    ),
                    smoke=max(
                        baseline.smoke,
                        0.1 if room_id == origin_room_id else baseline.smoke,
                    ),
                    structural_integrity=baseline.structural_integrity,
                    passable=baseline.passable,
                ),
            )
        self._project_all()

    def _activate_scheduled_event(
        self,
        scheduled_event: ScheduledEvent,
        step: int,
    ) -> list[EventSummary]:
        if scheduled_event.event_type != EventType.fire_ignition:
            return []
        room_id = str(scheduled_event.payload.get("origin_room_id", scheduled_event.target_id))
        if room_id not in self._room_lookup:
            return []
        current = self._room_states.get(room_id, HazardState())
        severity = max(current.severity, float(scheduled_event.payload.get("severity", 0.25)))
        smoke = max(current.smoke, min(1.0, severity * 0.5))
        self._set_room_state(
            room_id,
            HazardState(
                hazard_type=HazardType.fire,
                severity=severity,
                smoke=smoke,
                water_level=current.water_level,
                structural_integrity=current.structural_integrity,
                passable=severity < 0.9,
            ),
        )
        self._burning_rooms.add(room_id)
        return [
            self._make_event(
                "fire_ignition",
                step,
                EventType.fire_ignition,
                room_id,
                f"Scheduled fire ignition occurred in room {room_id}.",
                event_id=scheduled_event.event_id,
            )
        ]

    def advance(self, step: int) -> list[EventSummary]:
        events = self._advance_scheduled_events(step)
        smoke_targets: set[str] = set()

        for room_id in sorted(self._burning_rooms):
            smoke_targets.add(room_id)
            smoke_targets.update(self._adjacency.get(room_id, []))
            current = self._room_states.get(room_id, HazardState(hazard_type=HazardType.fire))
            severity = _clamp01(current.severity + 0.15)
            self._update_room_state(
                room_id,
                hazard_type=HazardType.fire,
                severity=severity,
                passable=severity < 0.9,
            )

        for room_id in sorted(smoke_targets):
            current = self._room_states.get(room_id, HazardState())
            self._update_room_state(
                room_id,
                hazard_type=current.hazard_type or HazardType.fire,
                severity=current.severity,
                smoke=_clamp01(current.smoke + 0.08),
                water_level=current.water_level,
                structural_integrity=current.structural_integrity,
                passable=current.passable,
            )

        for stairwell_id, stairwells in self._stairwell_lookup.items():
            entry_ids = stairwells[0].entry_room_ids.values()
            if any(
                self._room_states.get(room_id, HazardState()).severity >= 0.75
                for room_id in entry_ids
            ):
                self._set_stairwell_blocked(stairwell_id, True)

        if step % 4 == 0:
            candidate_weights: dict[str, int] = {}
            for room_id in sorted(self._burning_rooms):
                for neighbor_id in self._adjacency.get(room_id, []):
                    if neighbor_id in self._burning_rooms:
                        continue
                    weight = 3 if self._edge_type(room_id, neighbor_id) == "corridor" else 1
                    candidate_weights[neighbor_id] = candidate_weights.get(neighbor_id, 0) + weight

            if candidate_weights:
                candidate_ids = sorted(candidate_weights)
                chosen_room_id = self.rng.choices(
                    candidate_ids,
                    weights=[candidate_weights[room_id] for room_id in candidate_ids],
                    k=1,
                )[0]
                current = self._room_states.get(chosen_room_id, HazardState())
                self._set_room_state(
                    chosen_room_id,
                    HazardState(
                        hazard_type=HazardType.fire,
                        severity=max(current.severity, 0.25),
                        smoke=max(current.smoke, 0.1),
                        water_level=current.water_level,
                        structural_integrity=current.structural_integrity,
                        passable=True,
                    ),
                )
                self._burning_rooms.add(chosen_room_id)
                events.append(
                    self._make_event(
                        "fire_ignition",
                        step,
                        EventType.fire_ignition,
                        chosen_room_id,
                        f"Fire spread into room {chosen_room_id}.",
                    )
                )

        self._project_all()
        return events


class FloodRise(DisasterEngine):
    def __init__(
        self,
        building: Building,
        origin_room_id: str,
        seed: int,
        *,
        scheduled_events: list[ScheduledEvent] | None = None,
        live_projection: bool = True,
    ) -> None:
        super().__init__(
            building,
            origin_room_id,
            seed,
            scheduled_events=scheduled_events,
            live_projection=live_projection,
        )
        self._max_floor_id = max(self._floor_lookup)
        self._flooded_floors: set[int] = set()
        self._current_flood_floor = 0
        self._apply_seepage(0)
        self._project_all()

    def _activate_scheduled_event(
        self,
        scheduled_event: ScheduledEvent,
        step: int,
    ) -> list[EventSummary]:
        if scheduled_event.event_type != EventType.flood_rise:
            return []
        room_id = str(scheduled_event.payload.get("origin_room_id", scheduled_event.target_id))
        room = self._room_lookup.get(room_id)
        if room is None:
            return []
        severity = max(room.hazard.severity, float(scheduled_event.payload.get("severity", 0.5)))
        self._set_room_state(
            room_id,
            HazardState(
                hazard_type=HazardType.flood,
                severity=severity,
                smoke=room.hazard.smoke,
                water_level=severity,
                structural_integrity=room.hazard.structural_integrity,
                passable=severity < 0.9,
            ),
        )
        return [
            self._make_event(
                "flood_rise",
                step,
                EventType.flood_rise,
                room_id,
                f"Scheduled flood rise affected room {room_id}.",
                event_id=scheduled_event.event_id,
            )
        ]

    def _apply_seepage(self, floor_id: int) -> None:
        floor = self._floor_lookup[floor_id]
        for room in floor.rooms:
            self._set_room_state(
                room.room_id,
                HazardState(
                    hazard_type=HazardType.flood,
                    severity=0.3,
                    water_level=0.3,
                    structural_integrity=room.hazard.structural_integrity,
                    passable=True,
                ),
            )

    def _submerge_floor(self, floor_id: int) -> None:
        floor = self._floor_lookup[floor_id]
        for room in floor.rooms:
            self._set_room_state(
                room.room_id,
                HazardState(
                    hazard_type=HazardType.flood,
                    severity=1.0,
                    water_level=1.0,
                    structural_integrity=room.hazard.structural_integrity,
                    passable=False,
                ),
            )

        for elevator_id, elevators in self._elevator_lookup.items():
            if any(served_floor <= floor_id for served_floor in elevators[0].floor_ids):
                self._set_elevator_operational(elevator_id, False)

        seepage_floor = self._floor_lookup.get(floor_id + 1)
        if seepage_floor is not None and (floor_id + 1) not in self._flooded_floors:
            self._apply_seepage(floor_id + 1)

        self._flooded_floors.add(floor_id)

    def advance(self, step: int) -> list[EventSummary]:
        events = self._advance_scheduled_events(step)
        if step % 5 != 0 or self._current_flood_floor > self._max_floor_id:
            self._project_all()
            return events

        next_floor = self._current_flood_floor
        self._submerge_floor(next_floor)
        self._current_flood_floor += 1
        self._project_all()
        events.append(
            self._make_event(
                "flood_rise",
                step,
                EventType.flood_rise,
                f"floor_{next_floor}",
                f"Flood waters submerged floor {next_floor}.",
            )
        )
        return events


class GasLeak(DisasterEngine):
    def __init__(
        self,
        building: Building,
        origin_room_id: str,
        seed: int,
        *,
        scheduled_events: list[ScheduledEvent] | None = None,
        live_projection: bool = True,
    ) -> None:
        super().__init__(
            building,
            origin_room_id,
            seed,
            scheduled_events=scheduled_events,
            live_projection=live_projection,
        )
        self._gas_rooms: set[str] = {origin_room_id}
        self._gas_corridors: set[str] = set()
        self._exploded_rooms: set[str] = set()
        self._explosion_damage: dict[str, float] = {}
        self._background_fire: dict[str, float] = {}
        for room_id, room in self._room_lookup.items():
            if room.hazard.hazard_type == HazardType.fire:
                self._background_fire[room_id] = room.hazard.severity
        self._set_room_state(
            origin_room_id,
            HazardState(
                hazard_type=HazardType.gas,
                severity=0.2,
                smoke=0.05,
                water_level=0.0,
                structural_integrity=self._room_lookup[origin_room_id].hazard.structural_integrity,
                passable=True,
            ),
        )
        self._project_all()

    def _activate_scheduled_event(
        self,
        scheduled_event: ScheduledEvent,
        step: int,
    ) -> list[EventSummary]:
        if scheduled_event.event_type != EventType.gas_rupture:
            return []
        room_id = str(scheduled_event.payload.get("origin_room_id", scheduled_event.target_id))
        if room_id not in self._room_lookup:
            return []
        current = self._room_states.get(room_id, HazardState())
        severity = max(current.severity, float(scheduled_event.payload.get("severity", 0.2)))
        self._set_room_state(
            room_id,
            HazardState(
                hazard_type=HazardType.gas,
                severity=severity,
                smoke=max(current.smoke, 0.05),
                water_level=current.water_level,
                structural_integrity=current.structural_integrity,
                passable=severity < 0.85,
            ),
        )
        self._gas_rooms.add(room_id)
        return [
            self._make_event(
                "gas_rupture",
                step,
                EventType.gas_rupture,
                room_id,
                f"Scheduled gas rupture occurred in room {room_id}.",
                event_id=scheduled_event.event_id,
            )
        ]

    def _trigger_explosion(self, room_id: str, step: int) -> list[EventSummary]:
        if room_id in self._exploded_rooms:
            return []

        self._exploded_rooms.add(room_id)
        current = self._room_states.get(room_id, HazardState())
        self._set_room_state(
            room_id,
            HazardState(
                hazard_type=HazardType.structural,
                severity=1.0,
                smoke=max(current.smoke, 0.5),
                water_level=current.water_level,
                structural_integrity=0.0,
                passable=False,
            ),
        )

        for neighbor_id in self._adjacency.get(room_id, []):
            damage = self._explosion_damage.get(neighbor_id, 0.0) + 0.5
            self._explosion_damage[neighbor_id] = damage
            current_neighbor = self._room_states.get(neighbor_id, HazardState())
            structural_integrity = min(
                current_neighbor.structural_integrity,
                _clamp01(1.0 - damage),
            )
            self._set_room_state(
                neighbor_id,
                HazardState(
                    hazard_type=current_neighbor.hazard_type or HazardType.structural,
                    severity=max(current_neighbor.severity, min(1.0, damage)),
                    smoke=current_neighbor.smoke,
                    water_level=current_neighbor.water_level,
                    structural_integrity=structural_integrity,
                    passable=current_neighbor.passable and structural_integrity > 0.0,
                ),
            )

        return [
            self._make_event(
                "explosion",
                step,
                EventType.explosion,
                room_id,
                f"Gas explosion occurred in room {room_id}.",
            )
        ]

    def advance(self, step: int) -> list[EventSummary]:
        events = self._advance_scheduled_events(step)

        for room_id in sorted(self._gas_rooms):
            if room_id in self._exploded_rooms:
                continue
            current = self._room_states.get(room_id, HazardState(hazard_type=HazardType.gas))
            severity = _clamp01(max(current.severity, 0.2) + 0.1)
            self._set_room_state(
                room_id,
                HazardState(
                    hazard_type=HazardType.gas,
                    severity=severity,
                    smoke=_clamp01(current.smoke + 0.06),
                    water_level=current.water_level,
                    structural_integrity=current.structural_integrity,
                    passable=severity < 0.85,
                ),
            )

        for corridor_id in sorted(self._gas_corridors):
            current = self._corridor_states.get(corridor_id, HazardState(hazard_type=HazardType.gas))
            severity = _clamp01(max(current.severity, 0.2) + 0.1)
            self._set_corridor_state(
                corridor_id,
                HazardState(
                    hazard_type=HazardType.gas,
                    severity=severity,
                    smoke=_clamp01(current.smoke + 0.06),
                    water_level=current.water_level,
                    structural_integrity=current.structural_integrity,
                    passable=severity < 0.85,
                ),
            )

        if step % 3 == 0:
            previous_corridors = set(self._gas_corridors)
            for room_id in sorted(self._gas_rooms):
                for corridor_id in self._corridors_by_room.get(room_id, []):
                    if corridor_id in self._gas_corridors:
                        continue
                    self._gas_corridors.add(corridor_id)
                    current = self._corridor_states.get(corridor_id, HazardState())
                    self._set_corridor_state(
                        corridor_id,
                        HazardState(
                            hazard_type=HazardType.gas,
                            severity=max(current.severity, 0.2),
                            smoke=max(current.smoke, 0.05),
                            water_level=current.water_level,
                            structural_integrity=current.structural_integrity,
                            passable=True,
                        ),
                    )
                    events.append(
                        self._make_event(
                            "gas_spread",
                            step,
                            EventType.gas_spread,
                            corridor_id,
                            f"Gas spread into corridor {corridor_id}.",
                        )
                    )

            for corridor_id in sorted(previous_corridors):
                corridor = self._corridor_lookup[corridor_id]
                for room_id in sorted({corridor.from_node_id, corridor.to_node_id}):
                    if room_id in self._gas_rooms:
                        continue
                    self._gas_rooms.add(room_id)
                    current = self._room_states.get(room_id, HazardState())
                    self._set_room_state(
                        room_id,
                        HazardState(
                            hazard_type=HazardType.gas,
                            severity=max(current.severity, 0.2),
                            smoke=max(current.smoke, 0.05),
                            water_level=current.water_level,
                            structural_integrity=current.structural_integrity,
                            passable=True,
                        ),
                    )
                    events.append(
                        self._make_event(
                            "gas_spread",
                            step,
                            EventType.gas_spread,
                            room_id,
                            f"Gas spread into room {room_id}.",
                        )
                    )

        for room_id in sorted(self._gas_rooms):
            if room_id in self._exploded_rooms:
                continue
            gas_severity = self._room_states.get(room_id, HazardState()).severity
            fire_severity = self._background_fire.get(room_id, 0.0)
            if gas_severity >= 0.4 and fire_severity >= 0.5:
                events.extend(self._trigger_explosion(room_id, step))

        self._project_all()
        return events


class StructuralDamage(DisasterEngine):
    def __init__(
        self,
        building: Building,
        origin_room_id: str,
        seed: int,
        *,
        disaster_zones: list[str] | None = None,
        scheduled_events: list[ScheduledEvent] | None = None,
        live_projection: bool = True,
    ) -> None:
        super().__init__(
            building,
            origin_room_id,
            seed,
            scheduled_events=scheduled_events,
            live_projection=live_projection,
        )
        self._collapse_risk = [
            room_id
            for room_id in sorted(disaster_zones or building.disaster_zones)
            if room_id in self._room_lookup
        ]
        self._collapsed_rooms: set[str] = set()

    def _collapse_room(self, room_id: str) -> None:
        self._collapsed_rooms.add(room_id)
        self._set_room_state(
            room_id,
            HazardState(
                hazard_type=HazardType.structural,
                severity=1.0,
                smoke=0.0,
                water_level=0.0,
                structural_integrity=0.0,
                passable=False,
            ),
            accessible=False,
        )

        for stairwell_id, stairwells in self._stairwell_lookup.items():
            collapsed_entries = sum(
                1
                for entry_room_id in stairwells[0].entry_room_ids.values()
                if entry_room_id in self._collapsed_rooms
            )
            if collapsed_entries >= 2:
                self._set_stairwell_blocked(stairwell_id, True)

    def _activate_scheduled_event(
        self,
        scheduled_event: ScheduledEvent,
        step: int,
    ) -> list[EventSummary]:
        if scheduled_event.event_type not in {
            EventType.structural_collapse,
            EventType.stairwell_collapse,
        }:
            return []
        target_id = str(scheduled_event.payload.get("origin_room_id", scheduled_event.target_id))
        if scheduled_event.event_type == EventType.stairwell_collapse:
            if target_id in self._stairwell_lookup:
                self._set_stairwell_blocked(target_id, True)
                return [
                    self._make_event(
                        "stairwell_collapse",
                        step,
                        EventType.stairwell_collapse,
                        target_id,
                        f"Scheduled stairwell collapse blocked {target_id}.",
                        event_id=scheduled_event.event_id,
                    )
                ]
            return []
        if target_id not in self._room_lookup:
            return []
        self._collapse_room(target_id)
        return [
            self._make_event(
                "structural_collapse",
                step,
                EventType.structural_collapse,
                target_id,
                f"Scheduled structural collapse occurred in room {target_id}.",
                event_id=scheduled_event.event_id,
            )
        ]

    def advance(self, step: int) -> list[EventSummary]:
        events = self._advance_scheduled_events(step)
        if step % 8 != 0:
            self._project_all()
            return events

        available = [room_id for room_id in self._collapse_risk if room_id not in self._collapsed_rooms]
        if not available:
            self._project_all()
            return events

        room_id = self.rng.choice(sorted(available))
        self._collapse_room(room_id)
        self._project_all()
        events.append(
            self._make_event(
                "structural_collapse",
                step,
                EventType.structural_collapse,
                room_id,
                f"Room {room_id} collapsed.",
            )
        )
        return events


class ActiveThreat(DisasterEngine):
    def __init__(
        self,
        building: Building,
        origin_room_id: str,
        seed: int,
        *,
        scheduled_events: list[ScheduledEvent] | None = None,
        live_projection: bool = True,
    ) -> None:
        super().__init__(
            building,
            origin_room_id,
            seed,
            scheduled_events=scheduled_events,
            live_projection=live_projection,
        )
        self.threat_state = ThreatState(
            current_room_id=origin_room_id,
            steps_since_move=0,
            movement_interval=3,
        )
        self._set_room_state(
            origin_room_id,
            HazardState(
                hazard_type=HazardType.threat,
                severity=1.0,
                structural_integrity=self._room_lookup[origin_room_id].hazard.structural_integrity,
                passable=False,
            ),
        )
        self._project_all()

    def _move_threat(self, next_room_id: str) -> None:
        previous_room_id = self.threat_state.current_room_id
        self.threat_state.target_room_id = next_room_id
        self.threat_state.steps_since_move = 0
        if next_room_id == previous_room_id:
            return
        self._clear_room_state(previous_room_id)
        self.threat_state.current_room_id = next_room_id
        self._set_room_state(
            next_room_id,
            HazardState(
                hazard_type=HazardType.threat,
                severity=1.0,
                structural_integrity=self._room_lookup[next_room_id].hazard.structural_integrity,
                passable=False,
            ),
        )

    def _activate_scheduled_event(
        self,
        scheduled_event: ScheduledEvent,
        step: int,
    ) -> list[EventSummary]:
        if scheduled_event.event_type != EventType.threat_move:
            return []
        next_room_id = str(scheduled_event.payload.get("origin_room_id", scheduled_event.target_id))
        if next_room_id not in self._room_lookup:
            return []
        previous_room_id = self.threat_state.current_room_id
        self._move_threat(next_room_id)
        return [
            self._make_event(
                "threat_move",
                step,
                EventType.threat_move,
                next_room_id,
                f"Threat moved from {previous_room_id} to {next_room_id}.",
                event_id=scheduled_event.event_id,
            )
        ]

    def _choose_next_room(self, room_id: str) -> str:
        neighbors = sorted(self._adjacency.get(room_id, []))
        if not neighbors:
            return room_id
        counts = {neighbor_id: self._room_lookup[neighbor_id].occupancy.total for neighbor_id in neighbors}
        highest_count = max(counts.values())
        if highest_count == 0:
            return self.rng.choice(neighbors)
        best_neighbors = [neighbor_id for neighbor_id, count in counts.items() if count == highest_count]
        return self.rng.choice(sorted(best_neighbors))

    def _consume_room(self, room_id: str, step: int) -> list[EventSummary]:
        room = self._room_lookup[room_id]
        lost = _remove_occupants(room.occupancy, 2)
        if lost.total == 0:
            return []
        return [
            self._make_event(
                "civilian_loss",
                step,
                EventType.civilian_loss,
                room_id,
                (
                    f"Threat caused {lost.total} civilian losses in room {room_id} "
                    f"(mobile={lost.mobile}, injured={lost.injured}, impaired={lost.mobility_impaired})."
                ),
            )
        ]

    def advance(self, step: int) -> list[EventSummary]:
        events = self._advance_scheduled_events(step)
        events.extend(self._consume_room(self.threat_state.current_room_id, step))

        if step % self.threat_state.movement_interval == 0:
            previous_room_id = self.threat_state.current_room_id
            next_room_id = self._choose_next_room(previous_room_id)
            self._move_threat(next_room_id)

            if next_room_id != previous_room_id:
                events.append(
                    self._make_event(
                        "threat_move",
                        step,
                        EventType.threat_move,
                        next_room_id,
                        f"Threat moved from {previous_room_id} to {next_room_id}.",
                    )
                )
        else:
            self.threat_state.steps_since_move += 1

        self._project_all()
        return events


class MultiCascade(DisasterEngine):
    def __init__(
        self,
        building: Building,
        origin_room_id: str,
        seed: int,
        *,
        scheduled_events: list[ScheduledEvent] | None = None,
        disaster_zones: list[str] | None = None,
        live_projection: bool = True,
    ) -> None:
        super().__init__(building, origin_room_id, seed, live_projection=live_projection)
        self._disaster_zones = disaster_zones or building.disaster_zones
        gas_origin = self._pick_room_on_floor(3)
        self.scheduled_events = [
            event.model_copy(deep=True)
            for event in (
                scheduled_events
                or [
                    ScheduledEvent(
                        event_id="cascade_gas_activation",
                        trigger_step=10,
                        event_type=EventType.gas_rupture,
                        target_id=gas_origin,
                        payload={"origin_room_id": gas_origin},
                    ),
                    ScheduledEvent(
                        event_id="cascade_structural_activation",
                        trigger_step=15,
                        event_type=EventType.stairwell_collapse,
                        target_id="structural_activation",
                    ),
                ]
            )
        ]
        self._active_engines: dict[str, DisasterEngine] = {
            "fire": FireSpread(
                building,
                origin_room_id,
                seed,
                live_projection=False,
            )
        }
        self._manual_stairwell_blocks: set[str] = set()
        self._cross_explosions: set[str] = set()
        self._cross_damage: dict[str, float] = {}
        self._merge_subengine_state()

    def _merge_states(self, left: HazardState, right: HazardState) -> HazardState:
        chosen = left if left.severity >= right.severity else right
        return HazardState(
            hazard_type=chosen.hazard_type,
            severity=max(left.severity, right.severity),
            smoke=max(left.smoke, right.smoke),
            water_level=max(left.water_level, right.water_level),
            structural_integrity=min(left.structural_integrity, right.structural_integrity),
            passable=left.passable and right.passable,
        )

    def _activate_event(self, scheduled_event: ScheduledEvent, step: int) -> list[EventSummary]:
        events: list[EventSummary] = []

        if scheduled_event.event_type == EventType.gas_rupture:
            origin_room_id = str(
                scheduled_event.payload.get("origin_room_id", scheduled_event.target_id)
            )
            self._active_engines["gas"] = GasLeak(
                self.building,
                origin_room_id,
                self.rng.randrange(1_000_000),
                live_projection=False,
            )
            events.append(
                self._make_event(
                    "cascade_gas_activation",
                    step,
                    EventType.gas_rupture,
                    origin_room_id,
                    f"Gas leak activated in room {origin_room_id}.",
                    event_id=scheduled_event.event_id,
                )
            )
        elif scheduled_event.event_type == EventType.stairwell_collapse:
            self._active_engines["structural"] = StructuralDamage(
                self.building,
                self.origin_room_id,
                self.rng.randrange(1_000_000),
                disaster_zones=self._disaster_zones,
                live_projection=False,
            )
            available_stairwells = [
                stairwell_id
                for stairwell_id, blocked in sorted(self._stairwell_status.items())
                if not blocked
            ]
            if available_stairwells:
                stairwell_id = self.rng.choice(available_stairwells)
                self._manual_stairwell_blocks.add(stairwell_id)
                events.append(
                    self._make_event(
                        "stairwell_collapse",
                        step,
                        EventType.stairwell_collapse,
                        stairwell_id,
                        f"Stairwell {stairwell_id} collapsed during the cascade.",
                        event_id=scheduled_event.event_id,
                    )
                )
            else:
                floor_hint = scheduled_event.payload.get("floor_id")
                if floor_hint is None:
                    for candidate in (
                        scheduled_event.payload.get("origin_room_id"),
                        scheduled_event.target_id,
                    ):
                        if isinstance(candidate, str) and candidate.startswith("F"):
                            floor_hint = candidate[1:].split("_", 1)[0]
                            break

                if floor_hint is not None:
                    description = (
                        f"Stage activated but all stairwells on floor {floor_hint} already blocked."
                    )
                else:
                    description = "Stage activated but all stairwells already blocked."

                events.append(
                    self._make_event(
                        "stairwell_collapse",
                        step,
                        EventType.stairwell_collapse,
                        scheduled_event.target_id,
                        description,
                        event_id=scheduled_event.event_id,
                    )
                )

        return events

    def _handle_cross_hazard_explosions(self, step: int) -> list[EventSummary]:
        fire_engine = self._active_engines.get("fire")
        gas_engine = self._active_engines.get("gas")
        if fire_engine is None or gas_engine is None:
            return []

        events: list[EventSummary] = []
        fire_map = fire_engine.get_hazard_map()
        gas_map = gas_engine.get_hazard_map()
        for room_id in sorted(set(fire_map) & set(gas_map)):
            if room_id in self._cross_explosions:
                continue
            if fire_map[room_id] < 0.3 or gas_map[room_id] < 0.3:
                continue
            self._cross_explosions.add(room_id)
            for neighbor_id in self._adjacency.get(room_id, []):
                self._cross_damage[neighbor_id] = self._cross_damage.get(neighbor_id, 0.0) + 0.5
            events.append(
                self._make_event(
                    "explosion",
                    step,
                    EventType.explosion,
                    room_id,
                    f"Cascade explosion occurred in room {room_id}.",
                )
            )
        return events

    def _merge_subengine_state(self) -> None:
        merged_rooms: dict[str, HazardState] = {}
        merged_corridors: dict[str, HazardState] = {}
        merged_accessibility: dict[str, bool] = {}
        stairwell_status = dict(self._baseline_stairwell_status)
        elevator_status = dict(self._baseline_elevator_status)

        for engine in self._active_engines.values():
            for room_id, state in engine._room_states.items():
                if room_id in merged_rooms:
                    merged_rooms[room_id] = self._merge_states(merged_rooms[room_id], state)
                else:
                    merged_rooms[room_id] = state.model_copy(deep=True)
            for corridor_id, state in engine._corridor_states.items():
                if corridor_id in merged_corridors:
                    merged_corridors[corridor_id] = self._merge_states(merged_corridors[corridor_id], state)
                else:
                    merged_corridors[corridor_id] = state.model_copy(deep=True)
            for room_id, accessible in engine._room_accessibility.items():
                merged_accessibility[room_id] = merged_accessibility.get(
                    room_id,
                    self._baseline_room_accessible[room_id],
                ) and accessible
            for stairwell_id, blocked in engine._stairwell_status.items():
                stairwell_status[stairwell_id] = stairwell_status.get(stairwell_id, False) or blocked
            for elevator_id, operational in engine._elevator_status.items():
                elevator_status[elevator_id] = elevator_status.get(elevator_id, True) and operational

        for stairwell_id in self._manual_stairwell_blocks:
            stairwell_status[stairwell_id] = True

        for room_id in self._cross_explosions:
            current = merged_rooms.get(room_id, HazardState())
            merged_rooms[room_id] = HazardState(
                hazard_type=HazardType.structural,
                severity=1.0,
                smoke=max(current.smoke, 0.5),
                water_level=current.water_level,
                structural_integrity=0.0,
                passable=False,
            )

        for room_id, damage in self._cross_damage.items():
            if room_id in self._cross_explosions:
                continue
            current = merged_rooms.get(room_id, HazardState())
            integrity = min(current.structural_integrity, _clamp01(1.0 - damage))
            merged_rooms[room_id] = HazardState(
                hazard_type=current.hazard_type or HazardType.structural,
                severity=max(current.severity, min(1.0, damage)),
                smoke=current.smoke,
                water_level=current.water_level,
                structural_integrity=integrity,
                passable=current.passable and integrity > 0.0,
            )

        self._room_states = merged_rooms
        self._corridor_states = merged_corridors
        self._room_accessibility = merged_accessibility
        self._stairwell_status = stairwell_status
        self._elevator_status = elevator_status
        self._project_all()

    def advance(self, step: int) -> list[EventSummary]:
        events: list[EventSummary] = []

        for scheduled_event in self.scheduled_events:
            if scheduled_event.triggered or step < scheduled_event.trigger_step:
                continue
            scheduled_event.triggered = True
            events.extend(self._activate_event(scheduled_event, step))

        for engine in self._active_engines.values():
            events.extend(engine.advance(step))

        events.extend(self._handle_cross_hazard_explosions(step))
        self._merge_subengine_state()
        return events


def create_disaster_engine(
    disaster_type: DisasterType,
    building: Building,
    origin_room_id: str,
    seed: int,
    disaster_zones: list[str] | None = None,
    scheduled_events: list[ScheduledEvent] | None = None,
) -> DisasterEngine:
    """Factory to create the right disaster engine for a task."""
    if disaster_type == DisasterType.fire:
        return FireSpread(building, origin_room_id, seed, scheduled_events=scheduled_events)
    if disaster_type == DisasterType.flood:
        return FloodRise(building, origin_room_id, seed, scheduled_events=scheduled_events)
    if disaster_type == DisasterType.gas:
        return GasLeak(building, origin_room_id, seed, scheduled_events=scheduled_events)
    if disaster_type == DisasterType.structural:
        return StructuralDamage(
            building,
            origin_room_id,
            seed,
            disaster_zones=disaster_zones,
            scheduled_events=scheduled_events,
        )
    if disaster_type == DisasterType.active_threat:
        return ActiveThreat(building, origin_room_id, seed, scheduled_events=scheduled_events)
    if disaster_type == DisasterType.multi_cascade:
        return MultiCascade(
            building,
            origin_room_id,
            seed,
            scheduled_events=scheduled_events,
            disaster_zones=disaster_zones,
        )
    raise ValueError(f"Unsupported disaster type: {disaster_type}")


__all__ = [
    "ActiveThreat",
    "DisasterEngine",
    "FireSpread",
    "FloodRise",
    "GasLeak",
    "MultiCascade",
    "StructuralDamage",
    "create_disaster_engine",
]
