/* global L */
/* Centerline Editor — drag centerline vertices, regenerate bends & timing */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const UNDO_LIMIT = 30;
const BEND_COLORS = ["#ff4444", "#ff8800", "#ffcc00", "#44ff44", "#4488ff", "#cc44ff"];
const OBJ_COLORS = {
  hotlap_start: "#ff00ff",
  pit: "#00ffff",
  start: "#ffff00",
  timing_left: "#ff8c00",
  timing_right: "#ff4500",
};
const VERTEX_SAMPLE = 3; // Show every Nth vertex as draggable marker

const $ = (id) => document.getElementById(id);
function setStatus(msg) { $("status").textContent = msg; }

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let geoMeta = null;
let maskW = 0, maskH = 0;
let layouts = [];         // from layouts.json
let selectedLayout = "";  // layout name
let centerline = [];      // [[x,y], ...]
let bends = [];
let gameObjects = [];
let trackDirection = "clockwise";
let edited = false;
let dirty = false;

let undoStack = [];
let redoStack = [];

// Leaflet
let map;
let baseImageOverlay = null;
let maskOverlay = null;    // layout mask overlay
let centerlinePolyline = null;
let bendPolylines = [];
let vertexMarkers = [];    // L.marker array for draggable vertices
let timingLines = [];      // L.polyline for timing sections
let objectMarkers = [];    // L.marker for game objects
let imageBounds = null;

// ---------------------------------------------------------------------------
// Coordinate conversions (Simple CRS: pixel coords)
// ---------------------------------------------------------------------------
function pixelToLatLng(px, py) {
  return [maskH - py, px];
}

function latLngToPixel(lat, lng) {
  return [lng, maskH - lat];
}

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------
function pushUndo() {
  undoStack.push(JSON.parse(JSON.stringify(centerline)));
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack = [];
  markDirty();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.parse(JSON.stringify(centerline)));
  centerline = undoStack.pop();
  renderAll();
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.parse(JSON.stringify(centerline)));
  centerline = redoStack.pop();
  renderAll();
}

function markDirty() {
  dirty = true;
  edited = true;
  $("dirtyFlag").hidden = false;
}

function markClean() {
  dirty = false;
  $("dirtyFlag").hidden = true;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderCenterline() {
  // Remove old
  if (centerlinePolyline) centerlinePolyline.remove();
  bendPolylines.forEach(p => p.remove());
  bendPolylines = [];

  if (!centerline.length) return;

  // Full centerline (thin white dashed)
  const latlngs = centerline.map(([x, y]) => pixelToLatLng(x, y));
  centerlinePolyline = L.polyline(latlngs, {
    color: "white",
    weight: 1.5,
    dashArray: "6 3",
    opacity: 0.5,
    interactive: false,
  }).addTo(map);

  // Bend highlights
  bends.forEach((bend, i) => {
    const s = bend.start_idx;
    const e = bend.end_idx;
    const color = BEND_COLORS[i % BEND_COLORS.length];

    let idxs;
    if (e >= s) {
      idxs = [];
      for (let j = s; j <= e; j++) idxs.push(j);
    } else {
      idxs = [];
      for (let j = s; j < centerline.length; j++) idxs.push(j);
      for (let j = 0; j <= e; j++) idxs.push(j);
    }

    const segLL = idxs
      .filter(j => j < centerline.length)
      .map(j => pixelToLatLng(centerline[j][0], centerline[j][1]));

    if (segLL.length > 1) {
      const pl = L.polyline(segLL, {
        color,
        weight: 4,
        opacity: 0.8,
        interactive: false,
      }).addTo(map);
      bendPolylines.push(pl);
    }
  });
}

function renderVertexMarkers() {
  vertexMarkers.forEach(m => m.remove());
  vertexMarkers = [];

  if (!centerline.length) return;

  // Show sampled vertices as draggable markers
  for (let i = 0; i < centerline.length; i += VERTEX_SAMPLE) {
    const [x, y] = centerline[i];
    const ll = pixelToLatLng(x, y);

    const icon = L.divIcon({
      className: "",
      html: '<div class="ce-vertex"></div>',
      iconSize: [8, 8],
      iconAnchor: [4, 4],
    });

    const marker = L.marker(ll, { icon, draggable: true });
    marker._clIdx = i;

    marker.on("dragstart", () => {
      pushUndo();
      const el = marker.getElement();
      if (el) el.querySelector(".ce-vertex")?.classList.add("ce-vertex--drag");
    });

    marker.on("drag", (e) => {
      const latlng = e.target.getLatLng();
      const [px, py] = latLngToPixel(latlng.lat, latlng.lng);
      const idx = marker._clIdx;

      // Move this vertex and interpolate neighbors
      centerline[idx] = [Math.round(px * 10) / 10, Math.round(py * 10) / 10];

      // Smooth nearby points (within VERTEX_SAMPLE range)
      const halfRange = Math.floor(VERTEX_SAMPLE / 2);
      const prevIdx = idx - VERTEX_SAMPLE;
      const nextIdx = idx + VERTEX_SAMPLE;

      if (prevIdx >= 0 && nextIdx < centerline.length) {
        const [px0, py0] = centerline[prevIdx];
        const [px2, py2] = centerline[nextIdx];
        for (let j = 1; j < VERTEX_SAMPLE; j++) {
          const t = j / VERTEX_SAMPLE;
          const interpIdx = prevIdx + j;
          if (interpIdx === idx) continue;
          if (interpIdx >= 0 && interpIdx < centerline.length) {
            // Blend between prev anchor and current point
            if (interpIdx < idx) {
              const t2 = j / (idx - prevIdx);
              centerline[interpIdx] = [
                px0 + (centerline[idx][0] - px0) * t2,
                py0 + (centerline[idx][1] - py0) * t2,
              ];
            } else {
              const t2 = (interpIdx - idx) / (nextIdx - idx);
              centerline[interpIdx] = [
                centerline[idx][0] + (px2 - centerline[idx][0]) * t2,
                centerline[idx][1] + (py2 - centerline[idx][1]) * t2,
              ];
            }
          }
        }
      }

      // Update polyline in real-time
      if (centerlinePolyline) {
        centerlinePolyline.setLatLngs(
          centerline.map(([cx, cy]) => pixelToLatLng(cx, cy))
        );
      }
    });

    marker.on("dragend", (e) => {
      const el = marker.getElement();
      if (el) el.querySelector(".ce-vertex")?.classList.remove("ce-vertex--drag");
      // Re-render bend highlights
      renderCenterline();
      renderVertexMarkers();
    });

    marker.addTo(map);
    vertexMarkers.push(marker);
  }
}

function renderTimingAndObjects() {
  timingLines.forEach(l => l.remove());
  timingLines = [];
  objectMarkers.forEach(m => m.remove());
  objectMarkers = [];

  if (!gameObjects.length) return;

  // Group timing pairs
  const timingPairs = {};
  gameObjects.forEach(obj => {
    const name = obj.name || "";
    if (!name.startsWith("AC_TIME_")) return;
    const parts = name.replace("AC_TIME_", "").split("_");
    if (parts.length !== 2) return;
    const [idx, side] = parts;
    if (!timingPairs[idx]) timingPairs[idx] = {};
    timingPairs[idx][side] = obj;
  });

  // Draw timing lines
  Object.values(timingPairs).forEach(pair => {
    const left = pair.L;
    const right = pair.R;
    if (!left || !right) return;
    const lp = left.position;
    const rp = right.position;
    if (!lp || !rp) return;

    const line = L.polyline(
      [pixelToLatLng(lp[0], lp[1]), pixelToLatLng(rp[0], rp[1])],
      { color: "orange", weight: 2, opacity: 0.7, interactive: false }
    ).addTo(map);
    timingLines.push(line);
  });

  // Draw non-timing game objects
  gameObjects.forEach(obj => {
    const pos = obj.position;
    if (!pos || pos.length < 2) return;
    const type = obj.type || "unknown";
    if (type === "timing_left" || type === "timing_right") return;

    const color = OBJ_COLORS[type] || "#ffffff";
    const icon = L.divIcon({
      className: "",
      html: `<div class="ce-obj-marker" style="background:${color}"></div>`,
      iconSize: [10, 10],
      iconAnchor: [5, 5],
    });

    const marker = L.marker(pixelToLatLng(pos[0], pos[1]), { icon, interactive: false });
    marker.addTo(map);
    objectMarkers.push(marker);
  });
}

function renderBendList() {
  const el = $("bendList");
  el.innerHTML = "";

  bends.forEach((bend, i) => {
    const angleDeg = Math.round((bend.total_angle || 0) * 180 / Math.PI);
    const color = BEND_COLORS[i % BEND_COLORS.length];

    const div = document.createElement("div");
    div.className = "ce-bend-item";
    div.innerHTML = `
      <div class="ce-bend-swatch" style="background:${color}"></div>
      <span>B${i}: ${angleDeg}&deg; (exit idx ${bend.exit_idx})</span>
    `;

    div.addEventListener("click", () => {
      // Pan to bend peak
      const peakIdx = bend.peak_idx || bend.start_idx;
      if (peakIdx < centerline.length) {
        const [x, y] = centerline[peakIdx];
        map.panTo(pixelToLatLng(x, y));
      }
    });

    el.appendChild(div);
  });
}

function renderAll() {
  renderCenterline();
  renderVertexMarkers();
  renderTimingAndObjects();
  renderBendList();
  updateInfo();
}

function updateInfo() {
  $("clInfo").textContent = `Points: ${centerline.length} | Edited: ${edited ? "Yes" : "No"}`;

  // Game objects summary
  const counts = {};
  gameObjects.forEach(obj => {
    const t = obj.type || "unknown";
    counts[t] = (counts[t] || 0) + 1;
  });
  const timingCount = (counts.timing_left || 0) + (counts.timing_right || 0);
  $("goSummary").textContent =
    `timing: ${timingCount} | pit: ${counts.pit || 0} | start: ${counts.start || 0} | hotlap: ${counts.hotlap_start || 0}`;
}

// ---------------------------------------------------------------------------
// Layout loading
// ---------------------------------------------------------------------------
async function loadLayout(layoutName) {
  selectedLayout = layoutName;
  setStatus(`Loading layout: ${layoutName}...`);

  // Load centerline
  try {
    const resp = await fetch(`/api/layout_centerline/${encodeURIComponent(layoutName)}`);
    if (resp.ok) {
      const data = await resp.json();
      centerline = data.centerline || [];
      bends = data.bends || [];
      edited = data.edited || false;
      trackDirection = data.track_direction || "clockwise";
    } else {
      // Fall back to global centerline
      const globalResp = await fetch("/api/centerline");
      if (globalResp.ok) {
        const data = await globalResp.json();
        centerline = data.centerline || [];
        bends = data.bends || [];
        edited = false;
      } else {
        centerline = [];
        bends = [];
      }
    }
  } catch {
    centerline = [];
    bends = [];
  }

  // Load game objects
  try {
    const resp = await fetch(`/api/layout_game_objects/${encodeURIComponent(layoutName)}`);
    if (resp.ok) {
      const data = await resp.json();
      gameObjects = data.objects || [];
      trackDirection = data.track_direction || trackDirection;
    } else {
      // Fall back to global
      const globalResp = await fetch("/api/game_objects");
      if (globalResp.ok) {
        const data = await globalResp.json();
        gameObjects = data.objects || [];
      } else {
        gameObjects = [];
      }
    }
  } catch {
    gameObjects = [];
  }

  // Load layout mask overlay
  if (maskOverlay) maskOverlay.remove();
  const safeName = layoutName.replace(/[^\w\-]/g, '_').replace(/^_+|_+$/g, '') || "unnamed";
  maskOverlay = L.imageOverlay(`/api/layout_mask/${safeName}`, imageBounds, {
    opacity: 0.25,
    interactive: false,
  }).addTo(map);

  undoStack = [];
  redoStack = [];
  dirty = false;
  $("dirtyFlag").hidden = true;

  renderAll();
  setStatus(`Layout "${layoutName}": ${centerline.length} centerline points, ${bends.length} bends`);
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
async function save() {
  if (!selectedLayout) return;
  setStatus("Saving...");
  try {
    const clData = {
      layout_name: selectedLayout,
      centerline,
      bends,
      edited,
    };

    const resp = await fetch(`/api/layout_centerline/${encodeURIComponent(selectedLayout)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(clData),
    });
    if (!resp.ok) throw new Error(`centerline save: ${resp.status}`);

    markClean();
    setStatus(`Saved centerline for "${selectedLayout}"`);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Regenerate
// ---------------------------------------------------------------------------
async function regenerate() {
  if (!selectedLayout || centerline.length < 10) {
    setStatus("Need at least 10 centerline points to regenerate");
    return;
  }

  $("regenStatus").textContent = "Regenerating...";
  setStatus("Regenerating bends & timing...");

  try {
    const resp = await fetch("/api/centerline/regenerate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        centerline,
        layout_name: selectedLayout,
        track_direction: trackDirection,
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text}`);
    }

    const result = await resp.json();
    bends = result.bends || [];
    const newTimingObjects = result.timing_objects || [];

    // Replace timing objects in gameObjects, keep non-timing ones
    const nonTiming = gameObjects.filter(
      o => o.type !== "timing_left" && o.type !== "timing_right"
    );
    gameObjects = nonTiming.concat(newTimingObjects);

    // Save updated game objects
    const goData = {
      layout_name: selectedLayout,
      track_direction: trackDirection,
      objects: gameObjects,
    };
    await fetch(`/api/layout_game_objects/${encodeURIComponent(selectedLayout)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(goData),
    });

    // Also save centerline with new bends
    await save();

    renderAll();
    $("regenStatus").textContent = `${bends.length} bends, ${newTimingObjects.length} timing`;
    setStatus(`Regenerated: ${bends.length} bends, ${newTimingObjects.length} timing objects`);
  } catch (err) {
    $("regenStatus").textContent = "Failed";
    setStatus(`Regeneration failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  // Create map (Simple CRS)
  map = L.map("map", {
    crs: L.CRS.Simple,
    zoomControl: true,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
    minZoom: -2,
    maxZoom: 6,
  });
  setupRightDrag(map, $("map"));

  // Load geo metadata
  try {
    const metaResp = await fetch("/api/game_objects/geo_metadata");
    if (metaResp.ok) geoMeta = await metaResp.json();
  } catch {}
  if (!geoMeta) {
    try {
      const metaResp = await fetch("/api/geo_metadata");
      if (metaResp.ok) geoMeta = await metaResp.json();
    } catch {}
  }

  if (!geoMeta) {
    setStatus("Error: No geo_metadata found");
    return;
  }

  maskW = geoMeta.image_width;
  maskH = geoMeta.image_height;
  imageBounds = [[0, 0], [maskH, maskW]];

  // Base image
  try {
    baseImageOverlay = L.imageOverlay("/api/modelscale_image", imageBounds, {
      opacity: 1.0,
      interactive: false,
    }).addTo(map);
  } catch {}

  map.fitBounds(imageBounds, { padding: [10, 10] });

  // Load layout list
  const select = $("layoutSelect");
  try {
    const resp = await fetch("/api/track_layouts");
    if (resp.ok) {
      const data = await resp.json();
      layouts = data.layouts || [];
    }
  } catch {}

  if (layouts.length === 0) {
    // No layouts — use global centerline
    select.innerHTML = '<option value="">Default (no layouts)</option>';
    selectedLayout = "";
    await loadGlobalCenterline();
  } else {
    select.innerHTML = "";
    layouts.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l.name;
      opt.textContent = l.name;
      select.appendChild(opt);
    });
    await loadLayout(layouts[0].name);
  }

  wireUI();
}

async function loadGlobalCenterline() {
  try {
    const resp = await fetch("/api/centerline");
    if (resp.ok) {
      const data = await resp.json();
      centerline = data.centerline || [];
      bends = data.bends || [];
    }
  } catch {}

  try {
    const resp = await fetch("/api/game_objects");
    if (resp.ok) {
      const data = await resp.json();
      gameObjects = data.objects || [];
      trackDirection = data.track_direction || "clockwise";
    }
  } catch {}

  renderAll();
  setStatus(`Global centerline: ${centerline.length} points, ${bends.length} bends`);
}

function wireUI() {
  $("layoutSelect").addEventListener("change", (e) => {
    const name = e.target.value;
    if (name) loadLayout(name);
    else loadGlobalCenterline();
  });

  $("btnSave").addEventListener("click", () => save());
  $("btnUndo").addEventListener("click", () => undo());
  $("btnRedo").addEventListener("click", () => redo());
  $("btnRegenerate").addEventListener("click", () => regenerate());

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

    if (e.key === "s" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      save();
      return;
    }
    if (e.key === "z" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault();
      undo();
      return;
    }
    if ((e.key === "z" && (e.ctrlKey || e.metaKey) && e.shiftKey) ||
        (e.key === "y" && (e.ctrlKey || e.metaKey))) {
      e.preventDefault();
      redo();
      return;
    }
  });

  window.addEventListener("beforeunload", (e) => {
    if (dirty) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
}

// Error surfacing
window.addEventListener("error", (ev) => {
  try { setStatus(`Error: ${ev?.message || ev}`); } catch {}
});
window.addEventListener("unhandledrejection", (ev) => {
  try { setStatus(`Promise error: ${ev?.reason?.message || ev?.reason || ev}`); } catch {}
});

// Start
init();
