from __future__ import annotations

from evacos_ma.models import DisasterType, RewardWeights, TaskSpec, TaskSpecPublic


TASKS: dict[str, TaskSpec] = {
    # --- Phase 1 short-horizon tasks ---
    "task_1_fire_easy": TaskSpec(
        task_id="task_1_fire_easy",
        name="Single Fire Evacuation",
        difficulty="easy",
        disaster_type=DisasterType.fire,
        building_profile="small_3floor",
        success_criteria="Route all 15 civilians to ground exits",
        goal="ground_exit",
        max_steps=30,
        evaluation_seeds=[42, 123, 456],
        description="Fire starts in one room on floor 2. Spreads slowly. 3 floors, 2 stairwells, 2 ground exits. 15 civilians, no injured.",
        expected_score_range=[0.95, 0.999],
        reward_weights=RewardWeights(),
    ),
    "task_2_flood_medium": TaskSpec(
        task_id="task_2_flood_medium",
        name="Flood Rising Rooftop Evacuation",
        difficulty="medium",
        disaster_type=DisasterType.flood,
        building_profile="medium_5floor",
        success_criteria="Route civilians to rooftop before flood cuts stairwells",
        goal="rooftop",
        max_steps=40,
        evaluation_seeds=[42, 123, 456],
        description="Flood rising from ground floor. 5 floors, 3 stairwells (1 blocked), 1 rooftop exit. 30 civilians, 4 injured.",
        expected_score_range=[0.9, 0.949],
        reward_weights=RewardWeights(),
    ),
    "task_3_earthquake_hard": TaskSpec(
        task_id="task_3_earthquake_hard",
        name="Post-Earthquake Structural Evacuation",
        difficulty="hard",
        disaster_type=DisasterType.structural,
        building_profile="complex_5floor",
        success_criteria="Route civilians to the nearest safe exit before collapses cut routes",
        goal="nearest_exit",
        max_steps=50,
        evaluation_seeds=[42, 123, 456],
        description="Earthquake damages a 5-floor building and triggers progressive collapses. 4 stairwells plus same-floor fire-escape egress on selected floors, 50 civilians, 10 injured, 3 mobility-impaired.",
        expected_score_range=[0.8, 0.899],
        reward_weights=RewardWeights(),
    ),
    "task_4_cascade_hard": TaskSpec(
        task_id="task_4_cascade_hard",
        name="Multi-Hazard Cascade",
        difficulty="brutal",
        disaster_type=DisasterType.multi_cascade,
        building_profile="complex_5floor_full",
        success_criteria="Evacuate maximum civilians across all exit types",
        goal="maximum_survival",
        max_steps=60,
        evaluation_seeds=[42, 123, 456],
        description="Fire on floor 1, gas rupture on floor 3 at step 10, stairwell collapse at step 15. 5 floors, full complexity, external fire-escape egress on selected floors. 60 civilians, mixed mobility, panic mechanics.",
        expected_score_range=[0.75, 0.899],
        reward_weights=RewardWeights(),
    ),
    # --- Phase 3 long-horizon tasks ---
    "task_lh_fire_easy": TaskSpec(
        task_id="task_lh_fire_easy",
        name="Long-Horizon Fire Easy",
        difficulty="easy",
        disaster_type=DisasterType.fire,
        building_profile="lh_fire_easy_5floor",
        success_criteria="Evacuate civilians across 5 floors with gentle cascade scheduling",
        goal="ground_exit",
        max_steps=200,
        evaluation_seeds=[42, 123, 456, 789, 1024],
        description="Fire starts on floor 2 of a 5-floor building. Gentle cascade with secondary fire ignition at step 40. 35 civilians (30 mobile, 5 injured). 3 stairwells, 2 ground exits. Long-horizon planning required.",
        expected_score_range=[0.001, 0.999],
        reward_weights=RewardWeights(),
    ),
    "task_lh_flood_medium": TaskSpec(
        task_id="task_lh_flood_medium",
        name="Long-Horizon Flood Medium",
        difficulty="medium",
        disaster_type=DisasterType.flood,
        building_profile="lh_flood_medium_6floor",
        success_criteria="Evacuate civilians from rising water across 6 floors",
        goal="rooftop",
        max_steps=350,
        evaluation_seeds=[42, 123, 456, 789, 1024],
        description="Flood rising from ground floor of a 6-floor building. Rising-water cascade at steps 60 and 150. 55 civilians (45 mobile, 8 injured, 2 impaired). 4 stairwells (1 blocked), 2 ground + 1 rooftop exit.",
        expected_score_range=[0.001, 0.949],
        reward_weights=RewardWeights(),
    ),
    "task_lh_cascade_hard": TaskSpec(
        task_id="task_lh_cascade_hard",
        name="Long-Horizon Multi-Cascade Hard",
        difficulty="hard",
        disaster_type=DisasterType.multi_cascade,
        building_profile="lh_cascade_hard_6floor",
        success_criteria="Evacuate maximum civilians across multi-stage cascades over 500 rounds",
        goal="maximum_survival",
        max_steps=500,
        evaluation_seeds=[42, 123, 456, 789, 1024],
        description="Multi-cascade on a 6-floor building. Gas leak at step 80, stairwell collapse at step 160, fire reignition at step 300. 72 civilians (55 mobile, 12 injured, 5 impaired). 4 stairwells, multiple exit types.",
        expected_score_range=[0.001, 0.899],
        reward_weights=RewardWeights(),
    ),
    "task_lh_cascade_brutal": TaskSpec(
        task_id="task_lh_cascade_brutal",
        name="Long-Horizon Multi-Cascade Brutal",
        difficulty="brutal",
        disaster_type=DisasterType.multi_cascade,
        building_profile="lh_cascade_brutal_7floor",
        success_criteria="Survive 5-stage cascade over 500 rounds with dense civilian load",
        goal="maximum_survival",
        max_steps=500,
        evaluation_seeds=[42, 123, 456, 789, 1024],
        description="Brutal 5-stage cascade on a 7-floor building. Gas at 50, structural at 120, fire at 200, structural at 320, flood at 400. 110 civilians (80 mobile, 20 injured, 10 impaired). 5 stairwells (1 blocked), 6 exits, panic mechanics.",
        expected_score_range=[0.001, 0.799],
        reward_weights=RewardWeights(),
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def get_all_tasks() -> list[TaskSpec]:
    return list(TASKS.values())


def get_tasks_public() -> list[TaskSpecPublic]:
    """Return public-facing task info (no internal weights)."""
    return [
        TaskSpecPublic(
            task_id=task.task_id,
            name=task.name,
            difficulty=task.difficulty,
            disaster_type=task.disaster_type,
            goal=task.goal,
            description=task.description,
            max_steps=task.max_steps,
            expected_score_range=task.expected_score_range,
        )
        for task in TASKS.values()
    ]
