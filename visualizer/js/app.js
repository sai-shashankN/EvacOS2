import {
  collectFrameEvents,
  normalizeTrace,
  summarizeFrame,
} from "./trace.js";

const elements = {
  canvas: document.getElementById("sceneCanvas"),
  sceneState: document.getElementById("sceneState"),
  playButton: document.getElementById("playButton"),
  prevButton: document.getElementById("prevButton"),
  nextButton: document.getElementById("nextButton"),
  frameRange: document.getElementById("frameRange"),
  speedSelect: document.getElementById("speedSelect"),
  fileInput: document.getElementById("fileInput"),
  modeButtons: Array.from(document.querySelectorAll(".mode-button")),
  roundMetric: document.getElementById("roundMetric"),
  civilianMetric: document.getElementById("civilianMetric"),
  hazardMetric: document.getElementById("hazardMetric"),
  rewardMetric: document.getElementById("rewardMetric"),
  scoreMetric: document.getElementById("scoreMetric"),
  trajectoryTitle: document.getElementById("trajectoryTitle"),
  frameCounter: document.getElementById("frameCounter"),
  modeNotice: document.getElementById("modeNotice"),
  traceUrl: document.getElementById("traceUrl"),
  fetchButton: document.getElementById("fetchButton"),
  viewButtons: Array.from(document.querySelectorAll(".view-button")),
  floorStrip: document.getElementById("floorStrip"),
  directiveList: document.getElementById("directiveList"),
  actionList: document.getElementById("actionList"),
  scoreJson: document.getElementById("scoreJson"),
};

const FRAME_SECONDS = 0.85;
const LIVE_POLL_MS = 2500;

const state = {
  THREE: null,
  renderer: null,
  scene: null,
  camera: null,
  trace: null,
  frameIndex: 0,
  frameProgress: 0,
  playing: false,
  speed: 1,
  mode: "replay",
  viewMode: "solo",
  focusedFloorKey: "",
  userSelectedFloor: false,
  accumulator: 0,
  lastTime: performance.now(),
  visualTime: 0,
  liveTimer: null,
  activeSource: "./sample_trace.json",
  floorObjects: [],
};

init();

async function init() {
  bindControls();
  try {
    state.THREE = await import("three");
    setupScene();
    await loadTraceFromUrl("./sample_trace.json", { autoplay: true });
    requestAnimationFrame(tick);
  } catch (error) {
    setSceneMessage(`Unable to start visualizer: ${error.message}`, true);
  }
}

function bindControls() {
  elements.playButton.addEventListener("click", () => {
    if (state.mode === "manual") {
      return;
    }
    state.playing = !state.playing;
    state.accumulator = 0;
    state.frameProgress = 0;
    updateControls();
  });
  elements.prevButton.addEventListener("click", () => setFrame(wrapFrame(state.frameIndex - 1), { pause: true }));
  elements.nextButton.addEventListener("click", () => setFrame(wrapFrame(state.frameIndex + 1), { pause: true }));
  elements.frameRange.addEventListener("input", (event) => setFrame(Number(event.target.value), { pause: true }));
  elements.speedSelect.addEventListener("change", (event) => {
    state.speed = Number(event.target.value || 1);
  });
  elements.fileInput.addEventListener("change", handleFileLoad);
  elements.fetchButton.addEventListener("click", () => {
    loadTraceFromUrl(elements.traceUrl.value, {
      autoplay: state.mode !== "manual",
      jumpToLatest: state.mode === "live",
    });
    if (state.mode === "live") {
      startLivePolling();
    }
  });
  elements.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  elements.viewButtons.forEach((button) => {
    button.addEventListener("click", () => setViewMode(button.dataset.viewMode));
  });
  elements.floorStrip.addEventListener("click", (event) => {
    const button = event.target.closest("[data-floor-key]");
    if (!button) {
      return;
    }
    state.focusedFloorKey = button.dataset.floorKey;
    state.userSelectedFloor = true;
    if (state.viewMode === "all") {
      setViewMode("solo");
    } else {
      refreshFloorStrip();
      applyViewMode();
    }
  });
}

function setupScene() {
  const THREE = state.THREE;
  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x11100e);
  state.camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 0.1, 1000);
  state.camera.position.set(9, 8, 13);
  state.camera.lookAt(0, 2.8, 0);

  state.renderer = new THREE.WebGLRenderer({
    canvas: elements.canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const ambient = new THREE.HemisphereLight(0xffffff, 0x201b13, 2.4);
  const key = new THREE.DirectionalLight(0xffffff, 2.1);
  key.position.set(6, 10, 8);
  const back = new THREE.DirectionalLight(0x55d6be, 0.55);
  back.position.set(-8, 3, -6);
  state.scene.add(ambient, key, back);

  window.addEventListener("resize", resizeScene);
  resizeScene();
}

async function handleFileLoad(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }
  try {
    const payload = JSON.parse(await file.text());
    stopLivePolling();
    applyTrace(normalizeTrace(payload), file.name, { autoplay: state.mode !== "manual" });
  } catch (error) {
    setSceneMessage(`Could not load ${file.name}: ${error.message}`, true);
  }
}

async function loadTraceFromUrl(url, options = {}) {
  if (!url) {
    setSceneMessage("Enter a trace URL before fetching.", true);
    return;
  }
  if (!options.quiet) {
    setSceneMessage("Loading trace...");
  }
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    state.activeSource = url;
    applyTrace(normalizeTrace(payload), url, options);
  } catch (error) {
    setSceneMessage(`Could not fetch trace: ${error.message}`, true);
  }
}

function applyTrace(trace, sourceLabel, options = {}) {
  const previousIndex = state.frameIndex;
  state.trace = trace;
  state.accumulator = 0;
  state.frameProgress = 0;
  state.userSelectedFloor = false;
  if (!options.preserveFrame) {
    state.visualTime = 0;
  }
  const maxFrame = Math.max(0, trace.frames.length - 1);
  if (options.jumpToLatest) {
    state.frameIndex = maxFrame;
  } else if (options.preserveFrame) {
    state.frameIndex = Math.min(previousIndex, maxFrame);
  } else {
    state.frameIndex = 0;
  }
  state.playing = options.autoplay ?? state.mode !== "manual";
  elements.frameRange.max = String(Math.max(0, trace.frames.length - 1));
  elements.frameRange.value = String(state.frameIndex);
  elements.trajectoryTitle.textContent = trace.trajectory_id || sourceLabel || "trace";
  buildBuilding(trace.building);
  state.focusedFloorKey = pickCriticalFloorKey(trace.frames[state.frameIndex]) || state.floorObjects[0]?.floorKey || "";
  renderFrame();
  refreshModeNotice();
  updateControls();
  if (!options.quiet) {
    const action = state.playing ? "auto-playing" : "loaded";
    const liveSuffix = state.mode === "live" ? ` Polling every ${LIVE_POLL_MS / 1000}s.` : "";
    setSceneMessage(`${trace.frames.length} frames ${action} from ${sourceLabel || "trace"}.${liveSuffix}`);
  }
}

function buildBuilding(building) {
  const THREE = state.THREE;
  state.floorObjects.forEach((entry) => state.scene.remove(entry.group));
  state.floorObjects = [];

  const floors = [...building.floors].sort((a, b) => floorSortValue(a.floor_id) - floorSortValue(b.floor_id));
  const centerY = ((floors.length - 1) * 1.35) / 2;

  floors.forEach((floor, index) => {
    const group = new THREE.Group();
    const baseY = index * 1.35 - centerY;
    group.position.y = baseY;
    const floorKey = normalizedFloorKey(floor.floor_id);
    group.userData.floorKey = floorKey;
    group.userData.baseY = baseY;

    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(12.4, 0.08, 6.6),
      new THREE.MeshStandardMaterial({ color: 0x302b22, roughness: 0.72, metalness: 0.12 })
    );
    slab.position.y = -0.04;
    group.add(slab);

    const hazard = new THREE.Mesh(
      new THREE.BoxGeometry(12.45, 0.09, 6.65),
      new THREE.MeshStandardMaterial({
        color: 0xef6f5d,
        emissive: 0x7a1d15,
        emissiveIntensity: 0.35,
        transparent: true,
        opacity: 0,
        roughness: 0.85,
      })
    );
    hazard.position.y = 0.02;
    group.add(hazard);

    const roomMaterial = new THREE.MeshStandardMaterial({
      color: 0x3a3328,
      transparent: true,
      opacity: 0.52,
      roughness: 0.8,
    });
    const corridorMaterial = new THREE.MeshStandardMaterial({
      color: 0x151f20,
      emissive: 0x0b2d2c,
      emissiveIntensity: 0.25,
      transparent: true,
      opacity: 0.76,
      roughness: 0.68,
    });
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xc2b59f, transparent: true, opacity: 0.62 });
    const defaultExit = pickDefaultExit(floor);
    const roomObjects = [];

    (floor.corridors || []).forEach((corridor) => {
      const rect = roomToSceneRect(corridor.geometry, floor);
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(rect.w, 0.14, rect.h),
        corridorMaterial
      );
      mesh.position.set(rect.x, 0.11, rect.z);
      group.add(mesh);
    });

    floor.rooms.forEach((room) => {
      const rect = roomToSceneRect(room.geometry, floor);
      const roomId = String(room.room_id);
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(rect.w, 0.12, rect.h),
        roomMaterial
      );
      mesh.position.set(rect.x, 0.08, rect.z);
      group.add(mesh);

      const disasterOverlay = new THREE.Mesh(
        new THREE.BoxGeometry(rect.w, 0.15, rect.h),
        new THREE.MeshBasicMaterial({
          color: 0xef6f5d,
          transparent: true,
          opacity: 0,
          depthTest: false,
        })
      );
      disasterOverlay.position.set(rect.x, 0.2, rect.z);
      disasterOverlay.renderOrder = 4;
      group.add(disasterOverlay);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        lineMaterial
      );
      edges.position.copy(mesh.position);
      group.add(edges);

      const roomCenter = { x: rect.x, y: 0.34, z: rect.z };
      const corridorPoint = { x: rect.x, y: 0.39, z: 0 };
      const nearestExit = nearestPoint(roomCenter, floor.exits || [], floor) || defaultExit;
      const routeOne = makeRouteSegment(roomCenter, corridorPoint);
      const routeTwo = makeRouteSegment(corridorPoint, nearestExit);
      group.add(routeOne, routeTwo);

      const roomPeak = getRoomPeak(roomId);
      const markerCount = Math.max(3, Math.min(12, roomPeak || Number(room.capacity || 6)));
      const civilians = Array.from({ length: markerCount }, (_, civilianIndex) => {
        const material = new THREE.MeshStandardMaterial({
          color: 0x55d6be,
          emissive: 0x0b3d34,
          emissiveIntensity: 0.85,
          transparent: true,
          opacity: 0,
        });
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(0.09, 12, 10),
          material
        );
        const start = roomMarkerPosition(rect, civilianIndex, markerCount, 0.34);
        const exit = {
          x: nearestExit.x,
          y: 0.36,
          z: nearestExit.z + ((civilianIndex % 5) - 2) * 0.08,
        };
        const mid = {
          x: rect.x * 0.58,
          y: 0.38,
          z: ((civilianIndex % 3) - 1) * 0.16,
        };
        sphere.position.set(start.x, start.y, start.z);
        sphere.userData = { start, mid, exit, offset: civilianIndex * 0.026 };
        group.add(sphere);
        return sphere;
      });

      const deadCivilians = Array.from({ length: Math.max(2, Math.min(8, markerCount)) }, (_, deadIndex) => {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 16, 12),
          new THREE.MeshBasicMaterial({
            color: 0x000000,
            transparent: true,
            opacity: 0,
            depthTest: false,
          })
        );
        const pos = roomMarkerPosition(rect, deadIndex + 2, markerCount + 2, 0.62);
        marker.position.set(pos.x, pos.y, pos.z);
        marker.renderOrder = 10;
        group.add(marker);
        return marker;
      });

      const roomSmokePuffs = Array.from({ length: 3 }, (_, puffIndex) => {
        const puff = new THREE.Mesh(
          new THREE.SphereGeometry(0.28, 14, 10),
          new THREE.MeshBasicMaterial({
            color: 0xef6f5d,
            transparent: true,
            opacity: 0,
            depthWrite: false,
          })
        );
        const pos = roomMarkerPosition(rect, puffIndex + 1, 5, 0.62);
        puff.position.set(pos.x, pos.y, pos.z);
        group.add(puff);
        return puff;
      });

      const disasterLabel = makeFloorLabel("!");
      disasterLabel.position.set(rect.x - rect.w * 0.36, 0.56, rect.z - rect.h * 0.32);
      disasterLabel.scale.set(0.46, 0.38, 1);
      disasterLabel.visible = false;
      group.add(disasterLabel);

      roomObjects.push({
        roomId,
        rect,
        mesh,
        disasterOverlay,
        disasterLabel,
        civilians,
        deadCivilians,
        roomSmokePuffs,
        routeSegments: [routeOne, routeTwo],
        defaultExit: nearestExit,
        peak: roomPeak,
      });
    });

    (floor.exits || []).forEach((exit) => {
      const exitMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.34, 0.22, 0.34),
        new THREE.MeshStandardMaterial({ color: 0x83d483, emissive: 0x244c24 })
      );
      const pos = pointToScene(exit, floor);
      exitMesh.position.set(pos.x, 0.22, pos.z);
      group.add(exitMesh);
    });

    const label = makeFloorLabel(`F${floor.floor_id}`);
    label.position.set(-6.45, 0.28, -3.15);
    group.add(label);

    const routeSegments = Array.from({ length: 8 }, (_, routeIndex) => {
      const segment = new THREE.Mesh(
        new THREE.BoxGeometry(0.9, 0.07, 0.2),
        new THREE.MeshBasicMaterial({
          color: 0x55d6be,
          transparent: true,
          opacity: 0,
          depthTest: false,
        })
      );
      segment.position.set(-5.35 + routeIndex * 1.52, 0.44, -0.05);
      segment.renderOrder = 5;
      group.add(segment);
      return segment;
    });

    const beacon = new THREE.Mesh(
      new THREE.TorusGeometry(0.44, 0.025, 8, 36),
      new THREE.MeshBasicMaterial({
        color: 0xf3b454,
        transparent: true,
        opacity: 0,
      })
    );
    beacon.rotation.x = Math.PI / 2;
    beacon.position.set(5.8, 0.42, -0.05);
    group.add(beacon);

    const smokePuffs = Array.from({ length: 5 }, (_, puffIndex) => {
      const puff = new THREE.Mesh(
        new THREE.SphereGeometry(0.36, 16, 12),
        new THREE.MeshBasicMaterial({
          color: 0xef6f5d,
          transparent: true,
          opacity: 0,
          depthWrite: false,
        })
      );
      puff.position.set(-2.2 + puffIndex * 1.1, 0.46, -1.1 + (puffIndex % 2) * 0.54);
      group.add(puff);
      return puff;
    });

    const civilians = Array.from({ length: 34 }, (_, civilianIndex) => {
      const material = new THREE.MeshStandardMaterial({
        color: 0x55d6be,
        emissive: 0x0b3d34,
        emissiveIntensity: 0.8,
        transparent: true,
        opacity: 1,
      });
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.105, 12, 10),
        material
      );
      const col = civilianIndex % 17;
      const row = Math.floor(civilianIndex / 17);
      const start = {
        x: -5.2 + col * 0.58,
        y: 0.27,
        z: 2.55 - row * 0.38,
      };
      const side = civilianIndex % 2 === 0 ? -1 : 1;
      const exit = {
        x: side < 0 ? -5.95 : 5.95,
        y: 0.3,
        z: -0.05 + ((civilianIndex % 5) - 2) * 0.1,
      };
      const mid = {
        x: start.x * 0.34,
        y: 0.32,
        z: 0.04 + (row - 0.5) * 0.28,
      };
      sphere.position.set(start.x, start.y, start.z);
      sphere.userData = {
        start,
        mid,
        exit,
        offset: civilianIndex * 0.017,
      };
      group.add(sphere);
      return sphere;
    });

    state.scene.add(group);
    state.floorObjects.push({ group, hazard, civilians, floorKey, floorId: floor.floor_id, routeSegments, beacon, smokePuffs, roomObjects });
  });
}

function makeFloorLabel(text) {
  const THREE = state.THREE;
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#2a261f";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f3efe7";
  ctx.font = "700 30px Bahnschrift, Aptos, sans-serif";
  ctx.fillText(text, 18, 42);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
  sprite.material.transparent = true;
  sprite.scale.set(1.2, 0.6, 1);
  return sprite;
}

function renderFrame() {
  if (!state.trace || state.trace.frames.length === 0) {
    renderEmpty();
    return;
  }
  const frame = state.trace.frames[state.frameIndex];
  const nextFrame = getNextFrame();
  const summary = summarizeFrame(frame);
  if (!state.userSelectedFloor && state.viewMode !== "all") {
    state.focusedFloorKey = pickCriticalFloorKey(frame) || state.focusedFloorKey;
  }
  updateSceneObjects(frame, nextFrame, state.frameProgress, state.visualTime);
  applyViewMode();
  updateMetrics(frame, summary);
  refreshFloorStrip();
  updateFeeds(frame);
}

function updateSceneObjects(frame, nextFrame = frame, blend = 0, now = performance.now()) {
  state.floorObjects.forEach((entry) => {
    const hasRoomData = frameHasRoomData(frame, entry) || frameHasRoomData(nextFrame, entry);
    const currentCivilians = hasRoomData ? roomCiviliansTotal(frame, entry) : floorNumber(frame.per_floor_civilians, entry.floorKey);
    const nextCivilians = hasRoomData ? roomCiviliansTotal(nextFrame, entry) : floorNumber(nextFrame.per_floor_civilians, entry.floorKey);
    const civilians = lerp(currentCivilians, nextCivilians, blend);
    const currentHazard = hasRoomData ? roomHazardMax(frame, entry) : floorNumber(frame.per_floor_hazard_severity, entry.floorKey);
    const nextHazard = hasRoomData ? roomHazardMax(nextFrame, entry) : floorNumber(nextFrame.per_floor_hazard_severity, entry.floorKey);
    const hazard = clamp(lerp(currentHazard, nextHazard, blend), 0, 1);
    const wave = (Math.sin(now * 0.005 + floorSortValue(entry.floorKey)) + 1) / 2;
    const floorActive = hasFloorActivity(frame, entry.floorKey) || hasFloorActivity(nextFrame, entry.floorKey);

    entry.hazard.material.opacity = hasRoomData ? hazard * 0.18 : 0.08 + hazard * 0.52 + wave * hazard * 0.1;
    entry.hazard.material.emissiveIntensity = 0.25 + hazard * 0.9;
    entry.hazard.scale.set(1 + hazard * 0.025 + wave * 0.015, 1, 1 + hazard * 0.025 + wave * 0.015);
    entry.hazard.visible = hazard > 0.01;

    entry.smokePuffs.forEach((puff, index) => {
      const puffWave = (Math.sin(now * 0.004 + index * 1.7) + 1) / 2;
      puff.visible = hazard > 0.04;
      puff.material.opacity = hazard * (0.14 + puffWave * 0.16);
      const scale = 0.45 + hazard * 1.8 + puffWave * 0.35;
      puff.scale.set(scale * 1.25, scale * 0.55, scale);
      puff.position.y = 0.42 + hazard * 0.58 + puffWave * 0.22;
    });

    entry.routeSegments.forEach((segment, index) => {
      const routeWave = (Math.sin(now * 0.009 - index * 0.8) + 1) / 2;
      segment.visible = civilians > 0.2 || floorActive;
      segment.material.opacity = clamp(0.28 + routeWave * 0.5 + (floorActive ? 0.24 : 0), 0, 0.96);
      segment.scale.x = 0.68 + routeWave * 0.64;
      segment.position.y = 0.44 + routeWave * 0.045;
    });

    const beaconPulse = (Math.sin(now * 0.007) + 1) / 2;
    entry.beacon.visible = floorActive || hazard > 0.5;
    entry.beacon.material.opacity = floorActive ? 0.35 + beaconPulse * 0.45 : hazard * 0.22;
    entry.beacon.scale.setScalar(0.85 + beaconPulse * 0.7 + hazard * 0.3);

    updateRoomObjects(entry, frame, nextFrame, blend, now, hasRoomData);

    entry.civilians.forEach((sphere, index) => {
      if (hasRoomData) {
        sphere.visible = false;
        return;
      }
      const presence = clamp(civilians - index, 0, 1);
      const evacuation = 1 - clamp(civilians / Math.max(1, entry.civilians.length), 0, 1);
      const routeProgress = easeInOut(clamp(evacuation * 1.15 + blend * 0.18 + sphere.userData.offset, 0, 1));
      const bob = Math.sin(now * 0.011 + index * 0.9) * 0.025;
      const position = quadraticBezier(sphere.userData.start, sphere.userData.mid, sphere.userData.exit, routeProgress);

      sphere.visible = presence > 0.02;
      sphere.material.opacity = presence;
      sphere.position.set(position.x, position.y + bob, position.z);
      sphere.scale.setScalar(0.84 + presence * 0.55 + hazard * 0.1);
    });
  });
}

function applyViewMode() {
  if (!state.floorObjects.length) {
    return;
  }
  state.floorObjects.forEach((entry, index) => {
    const isFocused = entry.floorKey === state.focusedFloorKey || (!state.focusedFloorKey && index === 0);
    if (state.viewMode === "solo") {
      entry.group.visible = isFocused;
      entry.group.position.y = 0;
      entry.group.scale.setScalar(isFocused ? 0.96 : 1);
    } else if (state.viewMode === "focus") {
      const baseOffset = entry.group.userData.baseY || 0;
      entry.group.visible = true;
      entry.group.position.y = isFocused ? 0 : baseOffset * 1.8;
      entry.group.scale.setScalar(isFocused ? 1 : 0.88);
      setGroupVisualWeight(entry.group, isFocused ? 1 : 0.22);
    } else {
      entry.group.visible = true;
      entry.group.position.y = entry.group.userData.baseY || 0;
      entry.group.scale.setScalar(1);
      setGroupVisualWeight(entry.group, 1);
    }
    if (state.viewMode === "solo" && isFocused) {
      setGroupVisualWeight(entry.group, 1);
    }
  });
}

function setGroupVisualWeight(group, weight) {
  group.traverse((object) => {
    if (!object.material) {
      return;
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => {
      if (weight < 1) {
        material.transparent = true;
        material.opacity = Math.min(material.opacity ?? 1, weight);
      }
    });
  });
}

function refreshFloorStrip() {
  if (!state.trace || !elements.floorStrip) {
    return;
  }
  const frame = state.trace.frames[state.frameIndex];
  const focused = state.focusedFloorKey || pickCriticalFloorKey(frame);
  elements.floorStrip.innerHTML = state.floorObjects.map((entry) => {
    const summary = summarizeFloor(frame, entry);
    const hazardPct = Math.round(summary.hazard * 100);
    const hazardColor = hazardToColor(summary.hazard);
    const activeClass = entry.floorKey === focused ? " is-active" : "";
    const criticalClass = summary.hazard >= 0.65 || summary.casualties > 0 ? " is-critical" : "";
    return `
      <button class="floor-card${activeClass}${criticalClass}" type="button" data-floor-key="${escapeHtml(entry.floorKey)}" aria-label="Focus ${escapeHtml(entry.floorKey)}">
        <i class="hazard-pill" style="background:${hazardColor}"></i>
        <span>
          <strong>F${escapeHtml(String(entry.floorId))}</strong>
          ${summary.civilians} live / ${summary.casualties} lost / ${hazardPct}% hazard
        </span>
      </button>
    `;
  }).join("");
}

function updateRoomObjects(entry, frame, nextFrame, blend, now, hasRoomData) {
  entry.roomObjects.forEach((room, roomIndex) => {
    const current = getRoomState(frame, entry, room, roomIndex, hasRoomData);
    const next = getRoomState(nextFrame, entry, room, roomIndex, hasRoomData);
    const civilians = lerp(current.civilians, next.civilians, blend);
    const casualties = Math.max(current.casualties, next.casualties);
    const hazard = clamp(lerp(current.hazard, next.hazard, blend), 0, 1);
    const disasterType = next.disaster_type || current.disaster_type;
    const wave = (Math.sin(now * 0.006 + roomIndex * 1.4) + 1) / 2;
    const activeRoute = civilians > 0.1 && (current.civilians !== next.civilians || hazard > 0.05 || current.exit_id || next.exit_id);
    const palette = disasterColor(disasterType, hazard);

    room.disasterOverlay.visible = hazard > 0.02;
    room.disasterOverlay.material.color.setHex(palette.overlay);
    room.disasterOverlay.material.opacity = hazard > 0.02 ? clamp(0.12 + hazard * 0.62 + wave * 0.1, 0, 0.9) : 0;
    room.disasterOverlay.scale.set(1 + hazard * 0.035, 1, 1 + hazard * 0.035);

    room.disasterLabel.visible = hazard > 0.22 || Boolean(disasterType);
    room.disasterLabel.material.opacity = room.disasterLabel.visible ? 0.8 + wave * 0.2 : 0;
    room.disasterLabel.scale.setScalar(0.42 + hazard * 0.28 + wave * 0.06);

    room.roomSmokePuffs.forEach((puff, puffIndex) => {
      const puffWave = (Math.sin(now * 0.005 + puffIndex * 1.9 + roomIndex) + 1) / 2;
      puff.visible = hazard > 0.08;
      puff.material.color.setHex(palette.smoke);
      puff.material.opacity = hazard * (0.18 + puffWave * 0.22);
      const scale = 0.45 + hazard * 1.4 + puffWave * 0.28;
      puff.scale.set(scale * 1.3, scale * 0.58, scale);
      puff.position.y = 0.5 + hazard * 0.36 + puffWave * 0.18;
    });

    const peak = Math.max(1, room.peak || current.civilians || next.civilians || room.civilians.length);
    const evacuatedShare = 1 - clamp(civilians / peak, 0, 1);
    room.civilians.forEach((sphere, index) => {
      const normalizedIndex = index / Math.max(1, room.civilians.length - 1);
      const presence = clamp(civilians - index, 0, 1);
      const routeProgress = easeInOut(clamp(evacuatedShare * 1.18 + blend * 0.25 - normalizedIndex * 0.32 + sphere.userData.offset, 0, 1));
      const bob = Math.sin(now * 0.012 + index * 0.95 + roomIndex) * 0.028;
      const position = quadraticBezier(sphere.userData.start, sphere.userData.mid, sphere.userData.exit, routeProgress);

      sphere.visible = presence > 0.02;
      sphere.material.opacity = presence;
      sphere.position.set(position.x, position.y + bob, position.z);
      sphere.scale.setScalar(0.86 + presence * 0.42 + hazard * 0.12);
    });

    room.deadCivilians.forEach((marker, index) => {
      const visible = index < Math.round(casualties);
      marker.visible = visible;
      marker.material.opacity = visible ? 1 : 0;
      marker.position.y = 0.62 + index * 0.025;
      marker.scale.setScalar(visible ? 1.05 + Math.min(0.34, hazard * 0.26) : 0.01);
    });

    room.routeSegments.forEach((segment, segmentIndex) => {
      const routeWave = (Math.sin(now * 0.012 - segmentIndex * 0.9 - roomIndex * 0.45) + 1) / 2;
      segment.visible = activeRoute;
      segment.material.opacity = activeRoute ? clamp(0.18 + routeWave * 0.45 + hazard * 0.18, 0, 0.92) : 0;
      segment.scale.y = 1 + routeWave * 0.28;
      segment.scale.z = 1 + routeWave * 0.28;
    });
  });
}

function updateMetrics(frame, summary) {
  elements.roundMetric.textContent = String(summary.round);
  elements.civilianMetric.textContent = String(summary.civilians);
  elements.hazardMetric.textContent = summary.hazard.toFixed(2);
  elements.rewardMetric.textContent = summary.reward.toFixed(2);
  elements.scoreMetric.textContent = compactScore(summary.score);
  elements.scoreJson.textContent = JSON.stringify(summary.score, null, 2);
  elements.frameCounter.textContent = `${state.frameIndex + 1} / ${state.trace.frames.length}`;
  elements.frameRange.value = String(state.frameIndex);
}

function updateFeeds(frame) {
  const events = collectFrameEvents(frame);
  renderList(elements.directiveList, events.directives, "No directives on this frame.");
  renderList(elements.actionList, events.actions, "No overrides or actions on this frame.");
}

function renderList(target, rows, emptyText) {
  if (!rows.length) {
    target.innerHTML = `<li><span class="event-meta">${escapeHtml(emptyText)}</span></li>`;
    return;
  }
  target.innerHTML = rows.map((row) => `
    <li>
      <strong>${escapeHtml(row.label)}</strong>
      <span class="event-meta">${escapeHtml(row.actor)}${row.detail ? ` - ${escapeHtml(row.detail)}` : ""}</span>
    </li>
  `).join("");
}

function renderEmpty() {
  elements.roundMetric.textContent = "-";
  elements.civilianMetric.textContent = "-";
  elements.hazardMetric.textContent = "-";
  elements.rewardMetric.textContent = "-";
  elements.scoreMetric.textContent = "-";
  elements.frameCounter.textContent = "0 / 0";
  renderList(elements.directiveList, [], "Load a visualization trace to inspect directives.");
  renderList(elements.actionList, [], "Load a visualization trace to inspect actions.");
}

function setFrame(index, options = {}) {
  if (!state.trace) {
    return;
  }
  const max = state.trace.frames.length - 1;
  state.frameIndex = Math.max(0, Math.min(max, index));
  state.accumulator = 0;
  state.frameProgress = 0;
  if (options.pause) {
    state.playing = false;
    updateControls();
  }
  renderFrame();
}

function wrapFrame(index) {
  if (!state.trace || state.trace.frames.length === 0) {
    return 0;
  }
  const length = state.trace.frames.length;
  return ((index % length) + length) % length;
}

function setMode(mode) {
  state.mode = mode;
  stopLivePolling();
  elements.modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
  if (mode === "replay") {
    state.playing = Boolean(state.trace && state.trace.frames.length > 1);
  } else if (mode === "live") {
    state.playing = Boolean(state.trace && state.trace.frames.length > 1);
    startLivePolling();
  } else {
    state.playing = false;
    state.accumulator = 0;
    state.frameProgress = 0;
  }
  refreshModeNotice();
  updateControls();
  renderFrame();
}

function setViewMode(viewMode) {
  state.viewMode = viewMode;
  elements.viewButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.viewMode === viewMode);
  });
  if (viewMode !== "all" && !state.focusedFloorKey && state.trace) {
    state.focusedFloorKey = pickCriticalFloorKey(state.trace.frames[state.frameIndex]) || state.floorObjects[0]?.floorKey || "";
  }
  refreshFloorStrip();
  applyViewMode();
}

function refreshModeNotice() {
  if (state.mode === "live") {
    elements.modeNotice.textContent = `Live mode polls the trace URL every ${LIVE_POLL_MS / 1000}s. The viewer stays read-only.`;
  } else if (state.mode === "manual") {
    elements.modeNotice.textContent = "Manual frame stepping is active. Animation is paused for inspection.";
  } else {
    elements.modeNotice.textContent = "Replay mode loops the loaded trace with interpolated evacuee motion.";
  }
}

function updateControls() {
  elements.playButton.innerHTML = state.playing ? "&#10073;&#10073;" : "&#9654;";
  elements.playButton.setAttribute("aria-label", state.playing ? "Pause replay" : "Play replay");
  elements.playButton.disabled = state.mode === "manual";
}

function startLivePolling() {
  stopLivePolling();
  const poll = () => loadTraceFromUrl(elements.traceUrl.value || state.activeSource, {
    autoplay: true,
    preserveFrame: true,
    quiet: true,
  });
  state.liveTimer = window.setInterval(poll, LIVE_POLL_MS);
  poll();
}

function stopLivePolling() {
  if (state.liveTimer) {
    window.clearInterval(state.liveTimer);
    state.liveTimer = null;
  }
}

function tick(now) {
  const dt = Math.min(0.12, (now - state.lastTime) / 1000);
  state.lastTime = now;
  if (state.playing && state.trace && state.trace.frames.length > 1) {
    state.accumulator += dt * state.speed;
    state.visualTime += dt * 1000 * state.speed;
    while (state.accumulator >= FRAME_SECONDS) {
      state.accumulator -= FRAME_SECONDS;
      state.frameIndex = (state.frameIndex + 1) % state.trace.frames.length;
      renderFrame();
    }
  }
  state.frameProgress = state.trace && state.trace.frames.length > 1
    ? clamp(state.accumulator / FRAME_SECONDS, 0, 1)
    : 0;
  if (state.scene && state.camera && state.renderer) {
    if (state.trace && state.trace.frames.length) {
      const frame = state.trace.frames[state.frameIndex];
      updateSceneObjects(frame, getNextFrame(), state.frameProgress, state.visualTime);
    }
    const t = state.visualTime * 0.00022;
    const narrow = window.innerWidth <= 980;
    const compact = state.viewMode !== "all";
    const orbit = compact ? (narrow ? 0.55 : 0.85) : (narrow ? 0.7 : 1.2);
    const cameraProfile = compact
      ? (narrow
        ? { x: 11, y: 7.8, z: 15.4, targetY: -0.25, targetZ: 0.2 }
        : { x: 9.8, y: 7.2, z: 13.4, targetY: 0, targetZ: 0 })
      : (narrow
        ? { x: 11.4, y: 9.2, z: 17.4, targetY: 0.65, targetZ: 0 }
        : { x: 9, y: 8, z: 13, targetY: 0, targetZ: 0 });
    state.camera.position.x = cameraProfile.x + Math.sin(t) * orbit;
    state.camera.position.y = cameraProfile.y;
    state.camera.position.z = cameraProfile.z + Math.cos(t) * orbit;
    state.camera.lookAt(0, cameraProfile.targetY, cameraProfile.targetZ);
    state.renderer.render(state.scene, state.camera);
  }
  requestAnimationFrame(tick);
}

function resizeScene() {
  if (!state.renderer || !state.camera) {
    return;
  }
  const width = window.innerWidth;
  const height = window.innerHeight;
  state.camera.aspect = width / Math.max(1, height);
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(width, height, false);
}

function setSceneMessage(message, isError = false) {
  elements.sceneState.textContent = message;
  elements.sceneState.classList.toggle("is-error", isError);
}

function roomToSceneRect(geometry, floor) {
  const scaleX = 12 / floor.width;
  const scaleZ = 6.2 / floor.height;
  return {
    x: (geometry.x + geometry.w / 2 - floor.width / 2) * scaleX,
    z: (geometry.y + geometry.h / 2 - floor.height / 2) * scaleZ,
    w: Math.max(0.2, geometry.w * scaleX),
    h: Math.max(0.2, geometry.h * scaleZ),
  };
}

function pointToScene(point, floor) {
  return {
    x: (Number(point.x || 0) - floor.width / 2) * (12 / floor.width),
    z: (Number(point.y || 0) - floor.height / 2) * (6.2 / floor.height),
  };
}

function roomMarkerPosition(rect, index, count, y) {
  const columns = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / columns);
  const col = index % columns;
  const row = Math.floor(index / columns);
  const x = rect.x - rect.w * 0.32 + (columns <= 1 ? 0 : (col / (columns - 1)) * rect.w * 0.64);
  const z = rect.z - rect.h * 0.26 + (rows <= 1 ? 0 : (row / (rows - 1)) * rect.h * 0.52);
  return { x, y, z };
}

function nearestPoint(origin, exits, floor) {
  if (!Array.isArray(exits) || exits.length === 0) {
    return null;
  }
  return exits
    .map((exit) => pointToScene(exit, floor))
    .sort((a, b) => distance2(origin, a) - distance2(origin, b))[0];
}

function pickDefaultExit(floor) {
  const exit = Array.isArray(floor.exits) && floor.exits.length ? floor.exits[0] : { x: floor.width, y: floor.height / 2 };
  return pointToScene(exit, floor);
}

function makeRouteSegment(start, end) {
  const THREE = state.THREE;
  const dx = end.x - start.x;
  const dz = end.z - start.z;
  const length = Math.max(0.18, Math.sqrt(dx * dx + dz * dz));
  const segment = new THREE.Mesh(
    new THREE.BoxGeometry(length, 0.045, 0.08),
    new THREE.MeshBasicMaterial({
      color: 0x55d6be,
      transparent: true,
      opacity: 0,
      depthTest: false,
    })
  );
  segment.position.set((start.x + end.x) / 2, 0.47, (start.z + end.z) / 2);
  segment.rotation.y = -Math.atan2(dz, dx);
  segment.renderOrder = 6;
  return segment;
}

function getRoomPeak(roomId) {
  if (!state.trace || !Array.isArray(state.trace.frames)) {
    return 0;
  }
  return state.trace.frames.reduce((peak, frame) => {
    const roomState = frame.room_states && frame.room_states[roomId];
    if (!roomState) {
      return peak;
    }
    return Math.max(peak, Number(roomState.civilians || 0) + Number(roomState.casualties || 0));
  }, 0);
}

function frameHasRoomData(frame, entry) {
  if (!frame || !frame.room_states) {
    return false;
  }
  return entry.roomObjects.some((room) => Object.prototype.hasOwnProperty.call(frame.room_states, room.roomId));
}

function roomCiviliansTotal(frame, entry) {
  return entry.roomObjects.reduce((total, room, index) => (
    total + getRoomState(frame, entry, room, index, true).civilians
  ), 0);
}

function roomHazardMax(frame, entry) {
  return entry.roomObjects.reduce((max, room, index) => (
    Math.max(max, getRoomState(frame, entry, room, index, true).hazard)
  ), 0);
}

function summarizeFloor(frame, entry) {
  const hasRoomData = frameHasRoomData(frame, entry);
  if (hasRoomData) {
    return entry.roomObjects.reduce((summary, room, index) => {
      const roomState = getRoomState(frame, entry, room, index, true);
      return {
        civilians: summary.civilians + roomState.civilians,
        casualties: summary.casualties + roomState.casualties,
        hazard: Math.max(summary.hazard, roomState.hazard),
      };
    }, { civilians: 0, casualties: 0, hazard: 0 });
  }
  return {
    civilians: floorNumber(frame.per_floor_civilians, entry.floorKey),
    casualties: 0,
    hazard: floorNumber(frame.per_floor_hazard_severity, entry.floorKey),
  };
}

function pickCriticalFloorKey(frame) {
  if (!frame || !state.floorObjects.length) {
    return "";
  }
  return [...state.floorObjects]
    .sort((a, b) => {
      const aSummary = summarizeFloor(frame, a);
      const bSummary = summarizeFloor(frame, b);
      const aScore = aSummary.hazard * 100 + aSummary.casualties * 20 + aSummary.civilians * 0.12;
      const bScore = bSummary.hazard * 100 + bSummary.casualties * 20 + bSummary.civilians * 0.12;
      return bScore - aScore;
    })[0]?.floorKey || "";
}

function getRoomState(frame, entry, room, roomIndex, useDirectOnly) {
  const direct = frame && frame.room_states && frame.room_states[room.roomId];
  if (direct) {
    return {
      civilians: Number(direct.civilians || 0),
      hazard: Number(direct.hazard || 0),
      casualties: Number(direct.casualties || 0),
      disaster_type: direct.disaster_type || "",
      exit_id: direct.exit_id || "",
    };
  }
  if (useDirectOnly) {
    return { civilians: 0, hazard: 0, casualties: 0, disaster_type: "", exit_id: "" };
  }
  return synthesizeRoomState(frame, entry, roomIndex);
}

function synthesizeRoomState(frame, entry, roomIndex) {
  const rooms = entry.roomObjects.length || 1;
  const floorCivilians = floorNumber(frame.per_floor_civilians, entry.floorKey);
  const base = Math.floor(floorCivilians / rooms);
  const remainder = Math.floor(floorCivilians % rooms);
  const floorHazard = floorNumber(frame.per_floor_hazard_severity, entry.floorKey);
  const hotRoom = (Number(frame.round_id || 0) + floorSortValue(entry.floorKey)) % rooms;
  const roomDistance = Math.abs(roomIndex - hotRoom);
  const hazard = roomDistance === 0 ? floorHazard : roomDistance === 1 ? floorHazard * 0.42 : 0;
  return {
    civilians: base + (roomIndex < remainder ? 1 : 0),
    hazard,
    casualties: 0,
    disaster_type: hazard > 0.55 ? "fire" : hazard > 0.1 ? "smoke" : "",
    exit_id: "",
  };
}

function disasterColor(type, severity) {
  const normalized = String(type || "").toLowerCase();
  if (normalized.includes("fire") || severity > 0.68) {
    return { overlay: 0xef6f5d, smoke: 0xf3b454 };
  }
  if (normalized.includes("struct")) {
    return { overlay: 0xa8a29a, smoke: 0x80776b };
  }
  if (normalized.includes("flood") || normalized.includes("water")) {
    return { overlay: 0x4f9fcf, smoke: 0x55d6be };
  }
  return { overlay: 0x8c8072, smoke: 0xbab0a2 };
}

function hazardToColor(hazard) {
  if (hazard >= 0.68) {
    return "#ef6f5d";
  }
  if (hazard >= 0.34) {
    return "#f3b454";
  }
  return "#83d483";
}

function distance2(a, b) {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return dx * dx + dz * dz;
}

function getNextFrame() {
  if (!state.trace || state.trace.frames.length === 0) {
    return null;
  }
  const next = (state.frameIndex + 1) % state.trace.frames.length;
  return state.trace.frames[next] || state.trace.frames[state.frameIndex];
}

function floorNumber(values, floorKey) {
  if (!values || typeof values !== "object") {
    return 0;
  }
  return Number(values[floorKey] ?? values[floorSortValue(floorKey)] ?? 0);
}

function hasFloorActivity(frame, floorKey) {
  if (!frame) {
    return false;
  }
  if (frame.floor_action_types && frame.floor_action_types[floorKey]) {
    return true;
  }
  const fields = ["target_floor_id", "floor_id", "target_agent_id", "agent_id"];
  const feeds = [
    ...(Array.isArray(frame.directive_feed) ? frame.directive_feed : []),
    ...(Array.isArray(frame.override_feed) ? frame.override_feed : []),
    ...(Array.isArray(frame.action_feed) ? frame.action_feed : []),
  ];
  return feeds.some((item) => fields.some((field) => normalizePossibleFloor(item[field]) === floorKey));
}

function normalizePossibleFloor(value) {
  if (value === undefined || value === null) {
    return "";
  }
  const raw = String(value);
  if (/^floor_\d+$/i.test(raw)) {
    return raw.toLowerCase();
  }
  if (/^\d+$/.test(raw)) {
    return `floor_${raw}`;
  }
  const match = raw.match(/floor[_ -]?(\d+)/i);
  return match ? `floor_${match[1]}` : "";
}

function normalizedFloorKey(floorId) {
  const raw = String(floorId);
  return raw.startsWith("floor_") ? raw : `floor_${raw}`;
}

function floorSortValue(floorId) {
  const match = String(floorId).match(/(\d+)/);
  return match ? Number(match[1]) : 0;
}

function compactScore(score) {
  if (!score || Object.keys(score).length === 0) {
    return "-";
  }
  if (score.saved !== undefined) {
    return `saved ${score.saved}`;
  }
  const [key, value] = Object.entries(score)[0];
  return `${key} ${value}`;
}

function lerp(start, end, amount) {
  return start + (end - start) * clamp(amount, 0, 1);
}

function easeInOut(value) {
  const t = clamp(value, 0, 1);
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function quadraticBezier(start, mid, end, amount) {
  const t = clamp(amount, 0, 1);
  const inv = 1 - t;
  return {
    x: inv * inv * start.x + 2 * inv * t * mid.x + t * t * end.x,
    y: inv * inv * start.y + 2 * inv * t * mid.y + t * t * end.y,
    z: inv * inv * start.z + 2 * inv * t * mid.z + t * t * end.z,
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
