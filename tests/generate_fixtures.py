"""Generate golden JSON fixtures from Pydantic model instances."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evacos_ma.schemas.multi_agent import *
from evacos_ma.schemas.multi_agent import InterFloorView, ObservationsByRole
from evacos_ma.schemas.rewards import *

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

def write_fixture(name: str, model) -> None:
    path = os.path.join(FIXTURES_DIR, name)
    with open(path, "w") as f:
        f.write(model.model_dump_json(indent=2))
    print(f"  {name} written ({os.path.getsize(path)} bytes)")


def main() -> None:
    print("Generating golden fixtures...")

    # ---- floor_observation.golden.json ----
    floor_obs = FloorAgentObservationMA(
        episode_id="ep_goldfixture_0001",
        round_id=17,
        role=AgentRole.floor_agent,
        agent_id="floor_2_agent",
        step=17,
        max_steps=350,
        seed=42,
        tier=Tier.easy,
        disaster_family="fire",
        action_mask=["route_within_floor", "prioritize_room", "open_exit", "lockdown_room", "scout", "predict_state", "handoff_to_orchestrator", "wait"],
        last_reward_breakdown={"team_progress_dense": 0.1, "floor_saved": 0.0},
        done_last_round=False,
        notes=["Slight smoke detected on floor 2"],
        trace_schema_version="v1",
        generator_config_hash="sha256:abc123",
        floor_id="floor_2",
        visible_rooms=[
            RoomView(room_id="room_201", floor_id="floor_2", occupancy_mobile=3, occupancy_injured=1, hazard_severity=0.0, accessible=True, passable=True),
            RoomView(room_id="room_202", floor_id="floor_2", occupancy_mobile=0, occupancy_injured=0, hazard_severity=0.6, smoke_level=0.4, accessible=True, passable=True),
        ],
        visible_corridors=[
            CorridorView(corridor_id="corr_201_202", from_node_id="room_201", to_node_id="room_202", hazard_severity=0.1, passable=True),
        ],
        stairwell_entries=[
            StairwellEntryView(stairwell_id="stair_A", connects_floor_ids=[1, 2, 3], blocked=False, capacity_per_step=5),
        ],
        exits_on_floor=[
            ExitView(exit_id="exit_fire_escape_2", floor_id=2, exit_type="emergency_window", blocked=False, requires_open_action=True),
        ],
        visible_civilian_groups=[
            CivilianGroupView(civilian_group_id="cg_201", location_room_id="room_201", mobility_profile="mobile", count=3, status="waiting"),
        ],
        local_hazards=[
            HazardView(hazard_id="haz_fire_202", hazard_type="fire", severity=0.6, room_id="room_202", projected_spread={"direction": "east", "eta_rounds": 5}),
        ],
        sensor_quality=0.84,
        visibility_age_by_room={"room_201": 0, "room_202": 4},
        active_directive=Directive(
            directive_id="dir_001",
            supersedes_directive_id_or_null=None,
            target="floor_2",
            directive_type="prioritize_evacuation",
            params={"exit_id": "exit_fire_escape_2"},
            priority=DirectivePriority.high,
            issued_round=15,
            ttl_rounds=20,
            human_readable_note="Evacuate floor 2 via fire escape exit - fire spreading.",
        ),
        override_applied_last_round=False,
        override_reason_last_round=None,
        pending_explanation_request=False,
        belief_horizon_limit=8,
        open_belief_slots=1,
        last_prediction_score=0.5,
    )
    write_fixture("floor_observation.golden.json", floor_obs)

    # ---- orchestrator_observation.golden.json ----
    orch_obs = OrchestratorObservationMA(
        episode_id="ep_goldfixture_0001",
        round_id=17,
        role=AgentRole.orchestrator,
        agent_id="orchestrator",
        step=17,
        max_steps=350,
        seed=42,
        tier=Tier.easy,
        disaster_family="fire",
        action_mask=["route_between_floors", "call_elevator", "evacuate_floor_priority", "broadcast_directive", "override_floor_agent", "request_explanation", "wait"],
        last_reward_breakdown={"team_progress_dense": 0.1, "coordination_bonus": 0.05},
        done_last_round=False,
        notes=[],
        trace_schema_version="v1",
        generator_config_hash="sha256:abc123",
        floor_summaries=[
            FloorSummary(floor_id="floor_0", known_civilian_count=12, unknown_room_count=0, hazard_severity=0.0, queue_pressure=0.2, exit_capacity_remaining=5, last_updated_round=17),
            FloorSummary(floor_id="floor_1", known_civilian_count=10, unknown_room_count=1, hazard_severity=0.1, queue_pressure=0.3, exit_capacity_remaining=4, last_updated_round=17),
            FloorSummary(floor_id="floor_2", known_civilian_count=4, unknown_room_count=0, hazard_severity=0.6, queue_pressure=0.5, exit_capacity_remaining=1, last_updated_round=17),
            FloorSummary(floor_id="floor_3", known_civilian_count=8, unknown_room_count=2, hazard_severity=0.0, queue_pressure=0.1, exit_capacity_remaining=3, last_updated_round=16),
            FloorSummary(floor_id="floor_4", known_civilian_count=6, unknown_room_count=1, hazard_severity=0.0, queue_pressure=0.0, exit_capacity_remaining=2, last_updated_round=16),
        ],
        inter_floor=InterFloorView(
            stairwells=[
                StairwellAggregateView(stairwell_id="stair_A", floor_ids=[0, 1, 2, 3, 4], blocked=False, current_load=2, capacity=5),
            ],
            elevator=ElevatorView(elevator_id="elev_1", current_floor=1, target_floor=0, operational=True, capacity=6, queue_length=1),
            global_exit_queue=[
                ExitQueueView(exit_id="exit_ground_1", floor_id=0, queue_depth=3, throughput_per_round=2.0, blocked=False),
            ],
        ),
        belief_rollup=BeliefRollup(
            total_beliefs=7,
            avg_confidence=0.72,
            resolved_count=3,
            pending_count=4,
            recent_highlights=[{"belief_id": "b_005", "confidence": 0.85}],
        ),
        recent_floor_actions=[
            ActionLogEntry(agent_id="floor_2_agent", floor_id="floor_2", action_type="route_within_floor", round_id=16, summary="Routed cg_201 toward exit"),
            ActionLogEntry(agent_id="floor_0_agent", floor_id="floor_0", action_type="open_exit", round_id=16, summary="Opened exit_ground_1"),
        ],
        recent_directive_outcomes=[
            DirectiveOutcome(directive_id="dir_001", target_floor_id="floor_2", directive_type="prioritize_evacuation", accepted=True, outcome_summary="Floor 2 agent acknowledged and routing civilians"),
        ],
        cascade_hint={"next_cascade_round": 20, "type": "gas_rupture", "floor": 3},
        unresolved_escalations=[
            EscalationRequest(agent_id="floor_2_agent", floor_id="floor_2", category="resource_contention", urgency="high", target_ids=["stair_A"], note="Stairwell near capacity"),
        ],
    )
    write_fixture("orchestrator_observation.golden.json", orch_obs)

    # ---- action_bundle.golden.json ----
    belief_for_predict = StructuredBelief(
        belief_id="b_gold_001",
        predictor_agent_id="floor_3_agent",
        target_entity_ids=["room_301"],
        horizon=5,
        prediction_payload={"predicted_hazard_severity": 0.8, "predicted_passable": False},
        confidence=0.75,
        justification="Smoke spreading from room_302 corridor, likely to reach room_301 within 5 rounds.",
        created_round=17,
        resolved_round_or_null=None,
    )

    bundle = ActionBundleMA(
        episode_id="ep_goldfixture_0001",
        round_id=17,
        orchestrator_action=ActionEnvelopeMA(
            episode_id="ep_goldfixture_0001",
            round_id=17,
            agent_id="orchestrator",
            action_id="act_orch_001",
            action_type=ActionTypeMA.broadcast_directive,
            arguments=BroadcastDirectiveArgs(
                directive=Directive(
                    directive_id="dir_002",
                    supersedes_directive_id_or_null="dir_001",
                    target="floor_2",
                    directive_type="lockdown",
                    params={"room_id": "room_202"},
                    priority=DirectivePriority.high,
                    issued_round=17,
                    ttl_rounds=15,
                    human_readable_note="Lock down room 202 to contain fire spread.",
                )
            ).model_dump(),
            rationale="Fire severity in room_202 increasing; lockdown to protect adjacent rooms.",
        ),
        floor_actions={
            "floor_0_agent": ActionEnvelopeMA(
                episode_id="ep_goldfixture_0001", round_id=17,
                agent_id="floor_0_agent", action_id="act_f0_001",
                action_type=ActionTypeMA.route_within_floor,
                arguments=RouteWithinFloorArgs(from_room_id="room_001", to_room_id="room_005", civilian_group_id_or_null="cg_001").model_dump(),
            ),
            "floor_1_agent": ActionEnvelopeMA(
                episode_id="ep_goldfixture_0001", round_id=17,
                agent_id="floor_1_agent", action_id="act_f1_001",
                action_type=ActionTypeMA.scout,
                arguments=ScoutArgs(target_room_id="room_103").model_dump(),
            ),
            "floor_2_agent": ActionEnvelopeMA(
                episode_id="ep_goldfixture_0001", round_id=17,
                agent_id="floor_2_agent", action_id="act_f2_001",
                action_type=ActionTypeMA.predict_state,
                arguments=PredictStateArgs(belief=belief_for_predict).model_dump(),
            ),
            "floor_3_agent": ActionEnvelopeMA(
                episode_id="ep_goldfixture_0001", round_id=17,
                agent_id="floor_3_agent", action_id="act_f3_001",
                action_type=ActionTypeMA.wait,
                arguments=WaitArgs().model_dump(),
            ),
            "floor_4_agent": ActionEnvelopeMA(
                episode_id="ep_goldfixture_0001", round_id=17,
                agent_id="floor_4_agent", action_id="act_f4_001",
                action_type=ActionTypeMA.handoff_to_orchestrator,
                arguments=HandoffToOrchestratorArgs(category="escalation", target_ids=["stair_B"], urgency="normal", note="Stairwell B showing signs of structural weakness").model_dump(),
            ),
        },
    )
    write_fixture("action_bundle.golden.json", bundle)

    # ---- step_result.golden.json ----
    step_result = StepResultMA(
        observations_by_role=ObservationsByRole(
            orchestrator=orch_obs,
            floors={
                "floor_0_agent": FloorAgentObservationMA(
                    episode_id="ep_goldfixture_0001", round_id=17, agent_id="floor_0_agent",
                    step=17, max_steps=350, seed=42, tier=Tier.easy, disaster_family="fire",
                    floor_id="floor_0",
                ),
                "floor_1_agent": FloorAgentObservationMA(
                    episode_id="ep_goldfixture_0001", round_id=17, agent_id="floor_1_agent",
                    step=17, max_steps=350, seed=42, tier=Tier.easy, disaster_family="fire",
                    floor_id="floor_1",
                ),
                "floor_2_agent": floor_obs,
                "floor_3_agent": FloorAgentObservationMA(
                    episode_id="ep_goldfixture_0001", round_id=17, agent_id="floor_3_agent",
                    step=17, max_steps=350, seed=42, tier=Tier.easy, disaster_family="fire",
                    floor_id="floor_3",
                ),
                "floor_4_agent": FloorAgentObservationMA(
                    episode_id="ep_goldfixture_0001", round_id=17, agent_id="floor_4_agent",
                    step=17, max_steps=350, seed=42, tier=Tier.easy, disaster_family="fire",
                    floor_id="floor_4",
                ),
            },
        ),
        rewards_by_role=RewardsByRole(
            orchestrator=RoleReward(
                raw=0.18, normalized=0.09,
                breakdown=RewardBreakdown(team_progress_dense=0.1, coordination_bonus=0.05, directive_quality=0.03),
                reward_schema_version="v1",
            ),
            floors={
                "floor_0_agent": RoleReward(raw=0.1, normalized=0.05, breakdown=RewardBreakdown(team_progress_dense=0.1), reward_schema_version="v1"),
                "floor_1_agent": RoleReward(raw=-0.1, normalized=-0.05, breakdown=RewardBreakdown(floor_scout_cost=-0.1), reward_schema_version="v1"),
                "floor_2_agent": RoleReward(raw=0.15, normalized=0.075, breakdown=RewardBreakdown(team_progress_dense=0.1, floor_prediction=0.05), reward_schema_version="v1"),
                "floor_3_agent": RoleReward(raw=-0.05, normalized=-0.025, breakdown=RewardBreakdown(floor_invalid_action=-0.05), reward_schema_version="v1"),
                "floor_4_agent": RoleReward(raw=0.05, normalized=0.025, breakdown=RewardBreakdown(team_progress_dense=0.05), reward_schema_version="v1"),
            },
        ),
        done=False,
        done_reason=None,
        invalid_actions=[
            {"agent_id": "floor_3_agent", "action_id": "act_f3_001", "reason": "Cannot wait when active directive requires action"},
        ],
        round_events=[
            {"event_id": "evt_001", "event_type": "fire_ignition", "target_id": "room_202", "description": "Fire intensified in room 202"},
            {"event_id": "evt_002", "event_type": "civilian_loss", "target_id": "cg_102", "description": "1 civilian lost due to spreading hazard"},
        ],
        info=StepResultInfo(
            reservation_trace=[{"resource": "stair_A", "round": 17, "reservations": [{"agent_id": "floor_2_agent", "count": 3}]}],
            arbitration_trace=[{"conflict": "stair_A_capacity", "winner": "floor_2_agent", "reason": "directive_priority"}],
            score_snapshot={"total_civilians": 60, "saved": 5, "lost": 1, "in_transit": 2},
        ),
    )
    write_fixture("step_result.golden.json", step_result)

    print("All fixtures generated successfully.")


if __name__ == "__main__":
    main()
