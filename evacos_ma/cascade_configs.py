"""Cascade stage configurations for long-horizon tasks.

Sensible default cascade schedules keyed by task_id. Each config defines
a list of CascadeStage objects that determine when and what disaster events
trigger during a long-horizon episode.
"""

from __future__ import annotations

from evacos_ma.cascade import CascadeStage

# Cascade configs: task_id -> list of CascadeStage
CASCADE_CONFIGS: dict[str, list[CascadeStage]] = {
    "task_lh_fire_easy": [
        CascadeStage(
            stage_id="fire_spread_1",
            trigger_step=40,
            disaster_kind="fire",
            params={"description": "Secondary fire ignition"},
            rng_substream_key="fire_easy_1",
        ),
    ],
    "task_lh_flood_medium": [
        CascadeStage(
            stage_id="flood_rise_1",
            trigger_step=60,
            disaster_kind="flood",
            params={"description": "Rising water level increase"},
            rng_substream_key="flood_medium_1",
        ),
        CascadeStage(
            stage_id="flood_rise_2",
            trigger_step=150,
            disaster_kind="flood",
            params={"description": "Major flood surge"},
            rng_substream_key="flood_medium_2",
        ),
    ],
    "task_lh_cascade_hard": [
        CascadeStage(
            stage_id="gas_leak",
            trigger_step=80,
            disaster_kind="gas",
            params={"description": "Gas rupture on upper floor"},
            rng_substream_key="cascade_hard_gas",
        ),
        CascadeStage(
            stage_id="structural_collapse",
            trigger_step=160,
            disaster_kind="stairwell_collapse",
            params={"description": "Progressive structural failure"},
            rng_substream_key="cascade_hard_structural",
        ),
        CascadeStage(
            stage_id="fire_reignition",
            trigger_step=300,
            disaster_kind="fire",
            params={"description": "Secondary fire from gas ignition"},
            rng_substream_key="cascade_hard_fire",
        ),
    ],
    "task_lh_cascade_brutal": [
        CascadeStage(
            stage_id="gas_leak_1",
            trigger_step=50,
            disaster_kind="gas",
            params={"description": "Initial gas leak"},
            rng_substream_key="brutal_gas1",
        ),
        CascadeStage(
            stage_id="structural_1",
            trigger_step=120,
            disaster_kind="stairwell_collapse",
            params={"description": "First structural collapse"},
            rng_substream_key="brutal_structural1",
        ),
        CascadeStage(
            stage_id="fire_1",
            trigger_step=200,
            disaster_kind="fire",
            params={"description": "Fire from gas contact"},
            rng_substream_key="brutal_fire1",
        ),
        CascadeStage(
            stage_id="structural_2",
            trigger_step=320,
            disaster_kind="structural",
            params={"description": "Secondary structural collapse"},
            rng_substream_key="brutal_structural2",
        ),
        CascadeStage(
            stage_id="flood_1",
            trigger_step=400,
            disaster_kind="flood",
            params={"description": "Burst pipe flooding"},
            rng_substream_key="brutal_flood1",
        ),
    ],
}


def get_cascade_config(task_id: str) -> list[CascadeStage]:
    """Return cascade stages for a task, or empty list if none configured."""
    return list(CASCADE_CONFIGS.get(task_id, []))
