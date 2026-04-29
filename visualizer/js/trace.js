export const TRACE_SCHEMA_VERSION = "visualization_trace_v1";

export function normalizeTrace(payload) {
  const trace = Array.isArray(payload)
    ? { schema_version: "legacy_frames", trajectory_id: "legacy_frames", frames: payload }
    : { ...payload };

  if (!trace || typeof trace !== "object") {
    throw new Error("Trace must be a JSON object or a list of frames.");
  }
  if (!Array.isArray(trace.frames)) {
    throw new Error("Trace must contain a frames array.");
  }

  const frames = trace.frames.map((frame, index) => normalizeFrame(frame, index));
  const building = normalizeBuilding(trace.building || synthesizeDefaultBuilding(frames));

  return {
    schema_version: trace.schema_version || TRACE_SCHEMA_VERSION,
    trajectory_id: trace.trajectory_id || "unnamed_trace",
    building,
    frames,
  };
}

export function synthesizeDefaultBuilding(frames = []) {
  const floorIds = new Set();
  frames.forEach((frame) => {
    Object.keys(frame.per_floor_civilians || {}).forEach((key) => floorIds.add(floorIndexFromKey(key)));
    Object.keys(frame.per_floor_hazard_severity || {}).forEach((key) => floorIds.add(floorIndexFromKey(key)));
  });
  const maxFloor = Math.max(4, ...Array.from(floorIds).filter((value) => Number.isFinite(value)));

  return {
    building_id: "synthetic_visualizer_building",
    floors: Array.from({ length: maxFloor + 1 }, (_, floorId) => ({
      floor_id: floorId,
      width: 800,
      height: 420,
      rooms: Array.from({ length: 8 }, (_room, roomIndex) => {
        const col = roomIndex % 4;
        const row = Math.floor(roomIndex / 4);
        return {
          room_id: `F${floorId}_R${roomIndex}`,
          geometry: { x: 30 + col * 190, y: 38 + row * 170, w: 150, h: 118 },
          capacity: 18 + ((floorId + roomIndex) % 3) * 4,
        };
      }),
      corridors: [
        { corridor_id: `F${floorId}_C0`, geometry: { x: 0, y: 190, w: 800, h: 36 } },
      ],
      exits: [
        { exit_id: `F${floorId}_E0`, x: 770, y: 208 },
        { exit_id: `F${floorId}_E1`, x: 30, y: 208 },
      ],
    })),
  };
}

export function summarizeFrame(frame) {
  const roomStates = Object.values(frame.room_states || {});
  const civilians = roomStates.length
    ? roomStates.reduce((total, value) => total + Number(value.civilians || 0), 0)
    : Object.values(frame.per_floor_civilians || {})
      .reduce((total, value) => total + Number(value || 0), 0);
  const hazards = roomStates.length
    ? roomStates.map((value) => Number(value.hazard || 0))
    : Object.values(frame.per_floor_hazard_severity || {})
      .map((value) => Number(value || 0));
  const reward = Object.values(frame.reward_ticker || {})
    .reduce((total, value) => total + Number(value || 0), 0);

  return {
    round: Number(frame.round_id || 0),
    civilians,
    hazard: hazards.length ? Math.max(...hazards) : 0,
    reward,
    score: frame.score_snapshot || {},
    done: Boolean(frame.done),
  };
}

export function collectFrameEvents(frame) {
  const directives = normalizeEventList(frame.directive_feed);
  const overrides = normalizeEventList(frame.override_feed);
  const actions = normalizeEventList(frame.action_feed);

  if (frame.orchestrator_action_type) {
    actions.push({
      kind: "orchestrator",
      actor: "orchestrator",
      label: frame.orchestrator_action_type,
      detail: "legacy renderer action",
    });
  }

  Object.entries(frame.floor_action_types || {}).forEach(([floor, action]) => {
    actions.push({
      kind: "floor",
      actor: floor,
      label: String(action),
      detail: "legacy renderer action",
    });
  });

  return { directives, actions: [...overrides, ...actions] };
}

function normalizeFrame(frame, index) {
  if (!frame || typeof frame !== "object") {
    throw new Error(`Frame ${index} must be an object.`);
  }
  return {
    round_id: frame.round_id ?? frame.round ?? index,
    per_floor_civilians: frame.per_floor_civilians || {},
    per_floor_hazard_severity: frame.per_floor_hazard_severity || {},
    directive_feed: Array.isArray(frame.directive_feed) ? frame.directive_feed : [],
    override_feed: Array.isArray(frame.override_feed) ? frame.override_feed : [],
    action_feed: Array.isArray(frame.action_feed) ? frame.action_feed : [],
    reward_ticker: frame.reward_ticker || {},
    score_snapshot: frame.score_snapshot || {},
    room_states: normalizeRoomStates(frame),
    done: Boolean(frame.done),
    orchestrator_action_type: frame.orchestrator_action_type,
    floor_action_types: frame.floor_action_types || {},
  };
}

function normalizeBuilding(building) {
  if (!building || !Array.isArray(building.floors) || building.floors.length === 0) {
    return synthesizeDefaultBuilding([]);
  }
  return {
    building_id: building.building_id || "visualizer_building",
    floors: building.floors.map((floor, index) => ({
      floor_id: floor.floor_id ?? index,
      width: Number(floor.width || 800),
      height: Number(floor.height || 420),
      rooms: Array.isArray(floor.rooms) ? floor.rooms : [],
      corridors: Array.isArray(floor.corridors) ? floor.corridors : [],
      exits: Array.isArray(floor.exits) ? floor.exits : [],
    })),
  };
}

function normalizeEventList(rows) {
  if (!Array.isArray(rows)) {
    return [];
  }
  return rows.map((row, index) => {
    if (typeof row === "string") {
      return { kind: "event", actor: "system", label: row, detail: "" };
    }
    const label = row.action_type || row.directive_type || row.override_type || row.type || row.kind || `event_${index}`;
    const actor = row.agent_id || row.target_agent_id || row.target_floor_id || row.floor_id || row.actor || "system";
    const detail = row.message || row.reason || row.summary || row.description || "";
    return { kind: row.kind || "event", actor: String(actor), label: String(label), detail: String(detail) };
  });
}

function normalizeRoomStates(frame) {
  const directStates = frame.room_states || frame.rooms || {};
  const rows = Array.isArray(directStates)
    ? directStates
    : Object.entries(directStates).map(([roomId, state]) => ({ room_id: roomId, ...state }));

  const normalized = {};
  rows.forEach((row) => {
    if (!row || typeof row !== "object") {
      return;
    }
    const roomId = row.room_id || row.id;
    if (!roomId) {
      return;
    }
    normalized[String(roomId)] = {
      civilians: Number(row.civilians ?? row.alive ?? row.occupants ?? 0),
      hazard: Number(row.hazard ?? row.hazard_severity ?? row.disaster_severity ?? 0),
      casualties: Number(row.casualties ?? row.dead ?? row.fatalities ?? 0),
      disaster_type: row.disaster_type || row.disaster || row.hazard_type || "",
      exit_id: row.exit_id || row.target_exit_id || "",
    };
  });

  const perRoomCivilians = frame.per_room_civilians || {};
  const perRoomHazards = frame.per_room_hazard_severity || {};
  const perRoomCasualties = frame.per_room_casualties || frame.per_room_dead || {};
  const perRoomDisasters = frame.room_disasters || {};
  Object.keys({
    ...perRoomCivilians,
    ...perRoomHazards,
    ...perRoomCasualties,
    ...perRoomDisasters,
  }).forEach((roomId) => {
    normalized[roomId] = {
      civilians: Number(perRoomCivilians[roomId] ?? normalized[roomId]?.civilians ?? 0),
      hazard: Number(perRoomHazards[roomId] ?? normalized[roomId]?.hazard ?? 0),
      casualties: Number(perRoomCasualties[roomId] ?? normalized[roomId]?.casualties ?? 0),
      disaster_type: perRoomDisasters[roomId] ?? normalized[roomId]?.disaster_type ?? "",
      exit_id: normalized[roomId]?.exit_id ?? "",
    };
  });

  return normalized;
}

function floorIndexFromKey(key) {
  const match = String(key).match(/(\d+)/);
  return match ? Number(match[1]) : 0;
}
