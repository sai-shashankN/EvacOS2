from __future__ import annotations

from evacos_ma.models import (
    Building,
    Corridor,
    EdgeRef,
    Elevator,
    Exit,
    ExitType,
    Floor,
    Occupancy,
    Rect,
    Room,
    ScheduledEvent,
    Stairwell,
)


def make_five_floor_building(
    *,
    stairwell_floors: tuple[int, ...] = (0, 1, 2, 3, 4),
    with_elevator: bool = False,
    impaired_on_top: int = 0,
    exits_on_floors: tuple[int, ...] = (0,),
    stairwell_blocked: bool = False,
) -> Building:
    floors: list[Floor] = []
    for floor_id in range(5):
        room = Room(
            room_id=f"F{floor_id}_R0",
            floor_id=floor_id,
            geometry=Rect(x=0, y=0, w=10, h=10),
            occupancy=Occupancy(
                mobile=1 if floor_id == 0 else 0,
                mobility_impaired=impaired_on_top if floor_id == 4 else 0,
            ),
        )
        floor = Floor(floor_id=floor_id, rooms=[room], corridors=[])
        if floor_id in exits_on_floors:
            exit_obj = Exit(
                exit_id=f"EX{floor_id}",
                floor_id=floor_id,
                exit_type=ExitType.ground if floor_id == 0 else ExitType.rooftop,
                adjacent_room_id=room.room_id,
            )
            room.adjacent_node_ids.append(exit_obj.exit_id)
            floor.exits.append(exit_obj)
        floors.append(floor)

    if stairwell_floors:
        entry_room_ids = {floor_id: f"F{floor_id}_R0" for floor_id in stairwell_floors}
        stairwell = Stairwell(
            stairwell_id="SW0",
            floor_ids=list(stairwell_floors),
            blocked=stairwell_blocked,
            entry_room_ids=entry_room_ids,
            capacity_per_step=1,
        )
        for floor in floors:
            if floor.floor_id in stairwell_floors:
                floor.rooms[0].adjacent_node_ids.append(stairwell.stairwell_id)
                floor.stairwells.append(stairwell.model_copy(deep=True))

    if with_elevator:
        elevator = Elevator(
            elevator_id="EL0",
            floor_ids=[0, 1, 2, 3, 4],
            current_floor=4,
            capacity=1,
            travel_time_per_floor=1,
        )
        for floor in floors:
            floor.rooms[0].adjacent_node_ids.append(elevator.elevator_id)
            floor.elevators.append(elevator.model_copy(deep=True))

    building = Building(building_id="fixture", seed=0, floors=floors, disaster_zones=[])
    graph_edges: list[EdgeRef] = []
    if stairwell_floors:
        for lo, hi in zip(stairwell_floors, stairwell_floors[1:]):
            graph_edges.append(EdgeRef(from_id=f"F{lo}_R0", to_id=f"F{hi}_R0", edge_type="stairwell"))
            graph_edges.append(EdgeRef(from_id=f"F{hi}_R0", to_id=f"F{lo}_R0", edge_type="stairwell"))
    if with_elevator:
        for lo in range(4):
            hi = lo + 1
            graph_edges.append(EdgeRef(from_id=f"F{lo}_R0", to_id=f"F{hi}_R0", edge_type="elevator"))
            graph_edges.append(EdgeRef(from_id=f"F{hi}_R0", to_id=f"F{lo}_R0", edge_type="elevator"))
    building.graph_edges = graph_edges
    return building


def scheduled_room_block(step: int, room_id: str, severity: float = 1.0) -> ScheduledEvent:
    from evacos_ma.models import EventType

    return ScheduledEvent(
        event_id=f"block_{step}_{room_id}",
        trigger_step=step,
        event_type=EventType.structural_collapse,
        target_id=room_id,
        payload={"severity": severity, "origin_room_id": room_id},
    )
