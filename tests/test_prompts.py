import json

from evacos_ma.schemas.multi_agent import (
    ActionLogEntry,
    AgentRole,
    BeliefRollup,
    CivilianGroupView,
    CorridorView,
    Directive,
    DirectiveOutcome,
    DirectivePriority,
    EscalationRequest,
    ExitQueueView,
    ExitView,
    FloorAgentObservationMA,
    FloorSummary,
    HazardView,
    InterFloorView,
    OrchestratorObservationMA,
    RoomView,
    StairwellAggregateView,
    StairwellEntryView,
)
from evacos_ma.schemas.multi_agent import TRACE_SCHEMA_VERSION
from evacos_ma.schemas.multi_agent import Tier as SchemaTier
from training.prompts import (
    PROMPT_TEMPLATE_VERSION,
    build_floor_prompt,
    build_orchestrator_prompt,
)


def _make_floor_obs(**overrides) -> FloorAgentObservationMA:
    base = dict(
        episode_id="ep_prompt_floor",
        round_id=3,
        role=AgentRole.floor_agent,
        agent_id="floor_2_agent",
        step=3,
        max_steps=10,
        seed=42,
        tier=SchemaTier.easy,
        disaster_family="fire",
        action_mask=["route_within_floor", "open_exit", "wait"],
        last_reward_breakdown={"team_progress_dense": 0.1},
        done_last_round=False,
        notes=["watch smoke"],
        trace_schema_version=TRACE_SCHEMA_VERSION,
        generator_config_hash="sha256:floor",
        floor_id="floor_2",
        visible_rooms=[
            RoomView(
                room_id="room_201",
                floor_id="floor_2",
                occupancy_mobile=2,
                occupancy_injured=1,
                hazard_severity=0.3,
                smoke_level=0.2,
                accessible=True,
                passable=True,
            )
        ],
        visible_corridors=[],
        stairwell_entries=[
            StairwellEntryView(
                stairwell_id="stair_A",
                connects_floor_ids=[1, 2, 3],
                blocked=False,
                capacity_per_step=5,
            )
        ],
        exits_on_floor=[
            ExitView(
                exit_id="exit_floor_2",
                floor_id=2,
                exit_type="stair",
                blocked=False,
                requires_open_action=True,
            )
        ],
        visible_civilian_groups=[
            CivilianGroupView(
                civilian_group_id="cg_201",
                location_room_id="room_201",
                mobility_profile="mobile",
                count=3,
                status="waiting",
            )
        ],
        local_hazards=[
            HazardView(
                hazard_id="haz_201",
                hazard_type="fire",
                severity=0.4,
                room_id="room_201",
            )
        ],
        sensor_quality=0.9,
        visibility_age_by_room={"room_201": 0},
        active_directive=Directive(
            directive_id="dir_201",
            target="floor_2",
            directive_type="prioritize_evacuation",
            params={"exit_id": "exit_floor_2"},
            priority=DirectivePriority.high,
            issued_round=2,
            ttl_rounds=5,
            human_readable_note="Move toward the floor exit",
        ),
        override_applied_last_round=True,
        override_reason_last_round="capacity rebalancing",
        pending_explanation_request=False,
        belief_horizon_limit=8,
        open_belief_slots=1,
        last_prediction_score=0.5,
    )
    base.update(overrides)
    return FloorAgentObservationMA(**base)


def _make_orch_obs(**overrides) -> OrchestratorObservationMA:
    base = dict(
        episode_id="ep_prompt_orch",
        round_id=5,
        role=AgentRole.orchestrator,
        agent_id="orchestrator",
        step=5,
        max_steps=12,
        seed=42,
        tier=SchemaTier.easy,
        disaster_family="fire",
        action_mask=["broadcast_directive", "override_floor_agent", "wait"],
        last_reward_breakdown={"coordination_bonus": 0.2},
        done_last_round=False,
        notes=[],
        trace_schema_version=TRACE_SCHEMA_VERSION,
        generator_config_hash="sha256:orch",
        floor_summaries=[
            FloorSummary(
                floor_id="floor_1",
                known_civilian_count=8,
                unknown_room_count=1,
                hazard_severity=0.6,
                queue_pressure=0.4,
                exit_capacity_remaining=3,
                last_updated_round=5,
            )
        ],
        inter_floor=InterFloorView(
            stairwells=[
                StairwellAggregateView(
                    stairwell_id="stair_A",
                    floor_ids=[0, 1, 2, 3, 4],
                    blocked=False,
                    current_load=2,
                    capacity=5,
                )
            ],
            global_exit_queue=[
                ExitQueueView(
                    exit_id="exit_ground",
                    floor_id=0,
                    queue_depth=3,
                    throughput_per_round=2.0,
                    blocked=False,
                )
            ],
        ),
        belief_rollup=BeliefRollup(
            total_beliefs=4,
            avg_confidence=0.75,
            resolved_count=1,
            pending_count=3,
        ),
        recent_floor_actions=[
            ActionLogEntry(
                agent_id="floor_1_agent",
                floor_id="floor_1",
                action_type="route_within_floor",
                round_id=4,
                summary="Routed civilians toward stairwell",
            )
        ],
        recent_directive_outcomes=[
            DirectiveOutcome(
                directive_id="dir_301",
                target_floor_id="floor_1",
                directive_type="prioritize_evacuation",
                accepted=True,
                outcome_summary="Accepted",
            )
        ],
        cascade_hint=None,
        unresolved_escalations=[
            EscalationRequest(
                agent_id="floor_1_agent",
                floor_id="floor_1",
                category="resource_contention",
                urgency="high",
                target_ids=["stair_A"],
                note="Queue spike",
            )
        ],
    )
    base.update(overrides)
    return OrchestratorObservationMA(**base)


def test_build_floor_prompt_includes_identity_contract_and_action_mask():
    obs = _make_floor_obs()

    messages = build_floor_prompt(obs)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert all(set(message) == {"role", "content"} for message in messages)
    system_message = messages[0]["content"]
    user_message = messages[1]["content"]

    assert "Episode: ep_prompt_floor" in system_message
    assert "Agent ID: floor_2_agent" in system_message
    assert "Floor: floor_2" in system_message
    assert "Disaster: fire" in system_message
    assert "Round: 3" in system_message
    assert "Step: 3/10" in system_message
    assert f"Allowed actions: {json.dumps(obs.action_mask)}" in system_message
    assert "Prompt template version: 2026.04.29" in user_message
    assert f"Prompt template version: {PROMPT_TEMPLATE_VERSION}" in user_message
    assert "Rooms:" in user_message
    assert "Exits:" in user_message
    assert "Stairwells:" in user_message
    assert "Corridors:" in user_message
    assert "Civilians:" in user_message
    assert "Hazards:" in user_message
    assert "Decision policy:" in user_message
    assert "choose that active action instead of wait" in user_message
    assert "Use wait only when no safe/useful action is available" in user_message
    assert "put it in exit_id or stairwell_id" in user_message
    assert "Valid route_within_floor argument bundles" in user_message
    assert '"route_within_floor_arguments":[{"from_room_id":"room_201","exit_id":"exit_floor_2"}' in user_message
    assert "Copy IDs exactly" in user_message
    assert '"action_type":"route_within_floor"' in user_message
    assert '"exit_id":"exit_floor_2"' in user_message
    assert '"stairwell_id":"stair_A"' in user_message
    assert '"action_type":"open_exit"' in user_message
    assert 'action_id (use short token like "a1")' in user_message
    assert "never include role names, quotes, or newlines" in user_message
    assert '"action_type":"scout"' not in user_message


def test_build_floor_prompt_includes_corridor_hazard_for_gas():
    obs = _make_floor_obs(
        disaster_family="gas",
        visible_corridors=[
            CorridorView(
                corridor_id="corridor_gas_1",
                from_node_id="room_201",
                to_node_id="room_202",
                hazard_severity=0.8,
                passable=False,
            )
        ],
    )

    user_message = build_floor_prompt(obs)[1]["content"]

    assert "Corridors:" in user_message
    assert '"corridor_id": "corridor_gas_1"' in user_message
    assert '"passable": false' in user_message
    assert "avoid routes through Corridors where passable=false" in user_message


def test_build_floor_prompt_examples_use_current_observation_ids():
    obs = _make_floor_obs(
        action_mask=["route_within_floor", "open_exit", "scout", "wait"],
        exits_on_floor=[
            ExitView(
                exit_id="EX0",
                floor_id=2,
                exit_type="stair",
                blocked=False,
                requires_open_action=True,
            )
        ],
        visible_rooms=[
            RoomView(
                room_id="R_ACTUAL",
                floor_id="floor_2",
                occupancy_mobile=2,
                occupancy_injured=0,
                hazard_severity=0.1,
                smoke_level=0.0,
                accessible=True,
                passable=True,
            )
        ],
        visible_civilian_groups=[
            CivilianGroupView(
                civilian_group_id="cg_actual",
                location_room_id="R_ACTUAL",
                mobility_profile="mobile",
                count=2,
                status="waiting",
            )
        ],
    )

    user_message = build_floor_prompt(obs)[1]["content"]

    assert '"exit_id":"EX0"' in user_message
    assert '"from_room_id":"R_ACTUAL"' in user_message
    assert '"target_room_id":"R_ACTUAL"' in user_message
    assert '"exit_id":"exit_floor_2"' not in user_message
    assert '"from_room_id":"room_201"' not in user_message


def test_build_floor_prompt_omits_open_exit_example_when_exit_does_not_need_opening():
    obs = _make_floor_obs(
        exits_on_floor=[
            ExitView(
                exit_id="already_open_exit",
                floor_id=2,
                exit_type="stair",
                blocked=False,
                requires_open_action=False,
            )
        ],
    )

    user_message = build_floor_prompt(obs)[1]["content"]

    assert '"action_id":"open_exit"' not in user_message
    assert '"exit_id":"already_open_exit"' in user_message
    assert "Do not use open_exit unless an exit has requires_open=true" in user_message


def test_build_floor_prompt_handles_empty_lists():
    obs = _make_floor_obs(
        visible_rooms=[],
        exits_on_floor=[],
        visible_civilian_groups=[],
        local_hazards=[],
        active_directive=None,
        override_applied_last_round=False,
        override_reason_last_round=None,
    )

    messages = build_floor_prompt(obs)
    user_message = messages[1]["content"]

    assert "Rooms: []" in user_message
    assert "Exits: []" in user_message
    assert "Civilians: []" in user_message
    assert "Hazards: []" in user_message


def test_build_orchestrator_prompt_includes_identity_contract_and_action_mask():
    obs = _make_orch_obs()

    messages = build_orchestrator_prompt(obs)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert all(set(message) == {"role", "content"} for message in messages)
    system_message = messages[0]["content"]
    user_message = messages[1]["content"]

    assert "Episode: ep_prompt_orch" in system_message
    assert "Agent ID: orchestrator" in system_message
    assert "Disaster: fire" in system_message
    assert "Round: 5" in system_message
    assert "Step: 5/12" in system_message
    assert f"Allowed actions: {json.dumps(obs.action_mask)}" in system_message
    assert "Prompt template version: 2026.04.29" in user_message
    assert f"Prompt template version: {PROMPT_TEMPLATE_VERSION}" in user_message
    assert "Floor summaries:" in user_message
    assert "Copyable evacuate_floor_priority arguments:" in user_message
    assert '"ordered_floor_ids":["floor_1"]' in user_message
    assert "evacuate_floor_priority_arguments" not in user_message
    assert "Orchestrator argument schemas:" in user_message
    assert "never use priority_floor" in user_message
    assert 'action_id (use short token like "a1")' in user_message
    assert "never include role names, quotes, or newlines" in user_message
    assert "Beliefs: total=4, avg_conf=0.75, resolved=1, pending=3" in user_message
    assert "Recent floor actions:" in user_message
    assert "Unresolved escalations:" in user_message
    assert "Recent directive outcomes:" in user_message


def test_build_orchestrator_prompt_handles_empty_recent_floor_actions():
    obs = _make_orch_obs(
        recent_floor_actions=[],
        unresolved_escalations=[],
        recent_directive_outcomes=[],
    )

    messages = build_orchestrator_prompt(obs)
    user_message = messages[1]["content"]

    assert "Floor summaries:" in user_message
    assert "Beliefs: total=4, avg_conf=0.75, resolved=1, pending=3" in user_message
    assert "Recent floor actions:" not in user_message
    assert "Unresolved escalations:" not in user_message
    assert "Recent directive outcomes:" not in user_message
