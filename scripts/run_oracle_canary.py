"""Run a deterministic EvacOS2 oracle canary before paid RL training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from evacos_ma.schemas.multi_agent import ActionBundleMA, ActionEnvelopeMA, ActionTypeMA


def _first_origin_room(floor_obs: Any) -> str | None:
    civilians = list(getattr(floor_obs, "visible_civilian_groups", []) or [])
    for civilian in civilians:
        room_id = getattr(civilian, "location_room_id", None)
        if room_id:
            return str(room_id)
    rooms = list(getattr(floor_obs, "visible_rooms", []) or [])
    for room in rooms:
        room_id = getattr(room, "room_id", None)
        if room_id:
            return str(room_id)
    return None


def _oracle_floor_action(
    *,
    episode_id: str,
    round_id: int,
    agent_id: str,
    floor_obs: Any,
) -> ActionEnvelopeMA:
    origin = _first_origin_room(floor_obs)
    exits = list(getattr(floor_obs, "exits_on_floor", []) or [])
    for exit_obj in exits:
        exit_id = getattr(exit_obj, "exit_id", None)
        if not exit_id:
            continue
        if getattr(exit_obj, "blocked", False) or getattr(exit_obj, "requires_open_action", False):
            return ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=round_id,
                agent_id=agent_id,
                action_id=f"oracle_open_{agent_id}_{round_id}",
                action_type=ActionTypeMA.open_exit,
                arguments={"exit_id": str(exit_id)},
            )
        return ActionEnvelopeMA(
            episode_id=episode_id,
            round_id=round_id,
            agent_id=agent_id,
            action_id=f"oracle_route_exit_{agent_id}_{round_id}",
            action_type=ActionTypeMA.route_within_floor,
            arguments={
                "from_room_id": origin,
                "exit_id": str(exit_id),
            } if origin else {"exit_id": str(exit_id)},
        )

    stairwells = list(getattr(floor_obs, "stairwell_entries", []) or [])
    for stairwell in stairwells:
        stairwell_id = getattr(stairwell, "stairwell_id", None)
        if stairwell_id and not getattr(stairwell, "blocked", False):
            return ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=round_id,
                agent_id=agent_id,
                action_id=f"oracle_route_stair_{agent_id}_{round_id}",
                action_type=ActionTypeMA.route_within_floor,
                arguments={
                    "from_room_id": origin,
                    "stairwell_id": str(stairwell_id),
                } if origin else {"stairwell_id": str(stairwell_id)},
            )

    rooms = list(getattr(floor_obs, "visible_rooms", []) or [])
    target_room = getattr(rooms[0], "room_id", None) if rooms else None
    return ActionEnvelopeMA(
        episode_id=episode_id,
        round_id=round_id,
        agent_id=agent_id,
        action_id=f"oracle_scout_{agent_id}_{round_id}",
        action_type=ActionTypeMA.scout if target_room else ActionTypeMA.wait,
        arguments={"target_room_id": str(target_room)} if target_room else {},
    )


def run_oracle_canary(
    *,
    seeds: list[int],
    max_rounds: int,
    task_id: str,
    tier: str,
    disaster_family: DisasterType,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env = EvacEnvironment()
        episode_id, observations = env.reset_multi_agent(
            task_id,
            seed=seed,
            procgen_tier=tier,
            procgen_disaster_family=disaster_family,
            procgen_max_steps=max_rounds,
        )
        for _ in range(max_rounds):
            ep = env.get_internal_state(episode_id)
            floor_actions = {
                agent_id: _oracle_floor_action(
                    episode_id=episode_id,
                    round_id=ep.step,
                    agent_id=agent_id,
                    floor_obs=floor_obs,
                )
                for agent_id, floor_obs in observations.floors.items()
            }
            result = env.step_multi_agent(
                ActionBundleMA(
                    episode_id=episode_id,
                    round_id=ep.step,
                    orchestrator_action=None,
                    floor_actions=floor_actions,
                )
            )
            observations = result.observations_by_role
            if result.done:
                break

        ep = env.get_internal_state(episode_id)
        total_civilians = getattr(ep.total_civilians, "total", ep.total_civilians)
        episodes.append(
            {
                "seed": seed,
                "rounds": ep.step,
                "saved": ep.civilians_saved.total,
                "lost": ep.civilians_lost.total,
                "remaining": max(
                    int(total_civilians) - ep.civilians_saved.total - ep.civilians_lost.total,
                    0,
                ),
            }
        )

    total_saved = sum(item["saved"] for item in episodes)
    total_lost = sum(item["lost"] for item in episodes)
    total_remaining = sum(item["remaining"] for item in episodes)
    total_civilians = total_saved + total_lost + total_remaining
    return {
        "task_id": task_id,
        "tier": tier,
        "disaster_family": disaster_family.value,
        "max_rounds": max_rounds,
        "episodes": episodes,
        "total_saved": total_saved,
        "total_lost": total_lost,
        "total_remaining": total_remaining,
        "save_rate": round(total_saved / max(total_civilians, 1), 4),
        "pass": total_saved > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="procgen_easy_fire")
    parser.add_argument("--tier", default="easy", choices=["easy", "medium", "hard", "brutal"])
    parser.add_argument("--disaster-family", default="fire", choices=["fire", "flood", "gas"])
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    summary = run_oracle_canary(
        seeds=[int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()],
        max_rounds=args.max_rounds,
        task_id=args.task_id,
        tier=args.tier,
        disaster_family=DisasterType(args.disaster_family),
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
