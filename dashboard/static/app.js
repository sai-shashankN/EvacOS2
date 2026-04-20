const episodeSelect = document.getElementById("episodeSelect");
const floorGrid = document.getElementById("floorGrid");
const directiveFeed = document.getElementById("directiveFeed");
const overrideFeed = document.getElementById("overrideFeed");
const rewardTicker = document.getElementById("rewardTicker");
const scoreSnapshot = document.getElementById("scoreSnapshot");
let source = null;

function renderFloors(payload) {
  const civilians = payload.per_floor_civilians || {};
  const hazards = payload.per_floor_hazard_severity || {};
  const floors = Array.from(new Set([...Object.keys(civilians), ...Object.keys(hazards)])).sort().reverse();
  floorGrid.innerHTML = floors.map((floor) => {
    const severity = Number(hazards[floor] || 0);
    const count = Number(civilians[floor] || 0);
    return `
      <article class="floor-card">
        <div class="floor-label">${floor}</div>
        <div class="floor-metric">civilians ${count}</div>
        <div class="heatbar"><span style="width:${Math.min(100, severity * 100)}%"></span></div>
        <div class="floor-metric">hazard ${severity.toFixed(2)}</div>
      </article>
    `;
  }).join("");
}

function renderFeed(target, rows) {
  target.innerHTML = (rows || []).map((row) => {
    const type = row.action_type || row.directive_type || "event";
    const actor = row.agent_id || row.target_floor_id || "system";
    return `<li><strong>${actor}</strong><span>${type}</span></li>`;
  }).join("") || "<li><span>no events yet</span></li>";
}

function openStream(episodeId) {
  if (source) source.close();
  if (!episodeId) return;
  source = new EventSource(`/stream?episode_id=${encodeURIComponent(episodeId)}&follow=true`);
  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    renderFloors(payload);
    renderFeed(directiveFeed, payload.directive_feed);
    renderFeed(overrideFeed, payload.override_feed);
    rewardTicker.textContent = `round ${payload.round_id} ${payload.done ? "done" : "live"}`;
    scoreSnapshot.textContent = JSON.stringify(payload.score_snapshot || {}, null, 2);
  };
}

async function loadEpisodes() {
  const response = await fetch("/episodes");
  const episodes = await response.json();
  episodeSelect.innerHTML = episodes.map((entry) => (
    `<option value="${entry.episode_id}">${entry.episode_id}</option>`
  )).join("");
  if (episodes.length) openStream(episodes[0].episode_id);
}

episodeSelect.addEventListener("change", (event) => openStream(event.target.value));
loadEpisodes();
