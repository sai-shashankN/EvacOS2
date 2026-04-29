# EvacOS2 Replay Visualizer

This directory is an optional browser-native add-on for inspecting EvacOS2 replay traces. It is static HTML, CSS, and JavaScript with Three.js loaded through an import map. It is not required for simulator correctness, training, evaluation, or checkpoint generation.

The sample trace auto-plays on load. The default Solo view enlarges one critical floor at a time, while the floor strip summarizes live civilians, losses, and hazard for every floor. Replay mode loops the loaded trace with interpolated evacuee motion, room-level disaster overlays, active route markers, black casualty markers, and event feeds. Live mode polls the trace URL every few seconds without mutating the simulator. Manual mode pauses animation for frame-by-frame inspection.

## Run

From the repo root:

```powershell
python -m http.server 8766 -d visualizer
```

Then open `http://127.0.0.1:8766/`.

When the main FastAPI app is running, the same files are also mounted at `/visualizer` if the `visualizer/` directory exists:

```powershell
uvicorn evacos_ma.api:app --port 7860
```

Then open `http://127.0.0.1:7860/visualizer/`.

## Trace Contract

The viewer accepts `visualization_trace_v1` JSON files containing `building` and `frames`. Frames may include `room_states` keyed by `room_id`; each room state can provide `civilians`, `hazard`, `casualties`, `disaster_type`, and `exit_id`. When room states are present, the viewer shows affected rooms, evacuees leaving rooms along route lines, and black casualty markers that remain in place.

It also accepts the older renderer shapes where the whole file is a frame list or an object with only `frames`; in that case the browser synthesizes a default multi-floor building and distributes floor-level values across rooms.

Use `sample_trace.json` as a compact reference file.
