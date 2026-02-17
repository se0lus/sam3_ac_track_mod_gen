/* global L */
/* Unified Game Objects Editor — centerline + objects editing in one page */

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
  timing: "#ff8c00",
};
const DEFAULT_COLOR = "#ffffff";
const VERTEX_SAMPLE = 3;
const MARKER_SIZE = 14;
const SELECTED_MARKER_SIZE = 18;
const ARROW_LENGTH = 40;

const $ = (id) => document.getElementById(id);
function setStatus(msg) { $("status").textContent = msg; }

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let geoMeta = null;
let maskW = 0, maskH = 0;
let layouts = [];
let selectedLayout = "";

let centerline = [];
let bends = [];
let gameObjects = [];
let trackDirection = "clockwise";
let edited = false;
let time0Idx = null;  // centerline index of TIME_0 (VLM-generated)

let mode = "centerline"; // "centerline" | "objects" | "create" | "create_direction"
let createType = "pit";
let createPendingPos = null;
let selectedObjIdx = -1;

let dirty = false;
let undoStack = [];
let redoStack = [];

// Layer visibility
let layers = {
  basemap: true,
  mask: true,
  centerline: true,
  bends: true,
  timing: true,
  objects: true,
};

// Leaflet objects
let map;
let baseImageOverlay = null;
let maskOverlay = null;
let imageBounds = null;

let centerlinePolyline = null;
let bendPolylines = [];
let vertexMarkers = [];
let timingLines = [];
let objectMarkers = [];
let arrowLines = [];
let dirHandle = null;
let rubberArrow = null;
let _idxToMarker = {}; // gameObjects idx → objectMarkers/arrowLines array index
let timingCenterHandle = null;
let timingWidthHandle = null;
let _suppressMapClick = false; // suppress map click after handle drag

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
// Undo / Redo (snapshots both centerline + objects)
// ---------------------------------------------------------------------------
function snapshot() {
  return {
    centerline: JSON.parse(JSON.stringify(centerline)),
    gameObjects: JSON.parse(JSON.stringify(gameObjects)),
    bends: JSON.parse(JSON.stringify(bends)),
    time0Idx,
  };
}

function restoreSnapshot(snap) {
  centerline = snap.centerline;
  gameObjects = snap.gameObjects;
  bends = snap.bends;
  if (snap.time0Idx !== undefined) time0Idx = snap.time0Idx;
}

function pushUndo() {
  undoStack.push(snapshot());
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack = [];
  markDirty();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(snapshot());
  restoreSnapshot(undoStack.pop());
  renderAll();
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(snapshot());
  restoreSnapshot(redoStack.pop());
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
// Helpers
// ---------------------------------------------------------------------------
function objColor(type) {
  return OBJ_COLORS[type] || DEFAULT_COLOR;
}

function bearing(dx, dy) {
  return ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
}

function generateName(type) {
  const prefixes = {
    hotlap_start: "AC_HOTLAP_START",
    pit: "AC_PIT",
    start: "AC_START",
    timing_left: "AC_TIME",
    timing_right: "AC_TIME",
  };
  const prefix = prefixes[type] || "AC_OBJ";

  if (type === "timing_left" || type === "timing_right") {
    const existingNums = gameObjects
      .filter((o) => o.type === "timing_left" || o.type === "timing_right")
      .map((o) => {
        const m = o.name.match(/AC_TIME_(\d+)/);
        return m ? parseInt(m[1]) : -1;
      });
    const maxNum = existingNums.length > 0 ? Math.max(...existingNums) : -1;
    const nextNum = maxNum + 1;
    const suffix = type === "timing_left" ? "_L" : "_R";
    return `AC_TIME_${nextNum}${suffix}`;
  }

  const count = gameObjects.filter((o) => o.type === type).length;
  return `${prefix}_${count}`;
}

// ---------------------------------------------------------------------------
// Timing pair helpers
// ---------------------------------------------------------------------------
function isTimingObj(obj) {
  return obj && (obj.type === "timing_left" || obj.type === "timing_right");
}

function findTimingPairIdx(idx) {
  const obj = gameObjects[idx];
  if (!obj || !isTimingObj(obj)) return -1;
  const m = obj.name.match(/^AC_TIME_(\d+)_(L|R)$/);
  if (!m) return -1;
  const pairName = `AC_TIME_${m[1]}_${m[2] === "L" ? "R" : "L"}`;
  return gameObjects.findIndex(o => o.name === pairName);
}

function timingPairNum(obj) {
  if (!obj) return -1;
  const m = obj.name.match(/^AC_TIME_(\d+)/);
  return m ? parseInt(m[1]) : -1;
}

// ---------------------------------------------------------------------------
// Rendering: Centerline
// ---------------------------------------------------------------------------
function renderCenterline() {
  if (centerlinePolyline) centerlinePolyline.remove();
  bendPolylines.forEach(p => p.remove());
  bendPolylines = [];

  if (!centerline.length || !layers.centerline) return;

  const latlngs = centerline.map(([x, y]) => pixelToLatLng(x, y));
  centerlinePolyline = L.polyline(latlngs, {
    color: "white",
    weight: 1.5,
    dashArray: "6 3",
    opacity: 0.5,
    interactive: false,
  }).addTo(map);

  if (!layers.bends) return;

  bends.forEach((bend, i) => {
    const s = bend.start_idx;
    const e = bend.end_idx;
    const color = BEND_COLORS[i % BEND_COLORS.length];

    let idxs = [];
    if (e >= s) {
      for (let j = s; j <= e; j++) idxs.push(j);
    } else {
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

// ---------------------------------------------------------------------------
// Rendering: Vertex markers (Centerline mode only)
// ---------------------------------------------------------------------------
function renderVertexMarkers() {
  vertexMarkers.forEach(m => m.remove());
  vertexMarkers = [];

  if (!centerline.length || mode !== "centerline") return;

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

      centerline[idx] = [Math.round(px * 10) / 10, Math.round(py * 10) / 10];

      const prevIdx = idx - VERTEX_SAMPLE;
      const nextIdx = idx + VERTEX_SAMPLE;

      if (prevIdx >= 0 && nextIdx < centerline.length) {
        const [px0, py0] = centerline[prevIdx];
        const [px2, py2] = centerline[nextIdx];
        for (let j = 1; j < VERTEX_SAMPLE; j++) {
          const interpIdx = prevIdx + j;
          if (interpIdx === idx) continue;
          if (interpIdx >= 0 && interpIdx < centerline.length) {
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

      if (centerlinePolyline) {
        centerlinePolyline.setLatLngs(
          centerline.map(([cx, cy]) => pixelToLatLng(cx, cy))
        );
      }
    });

    marker.on("dragend", () => {
      const el = marker.getElement();
      if (el) el.querySelector(".ce-vertex")?.classList.remove("ce-vertex--drag");
      renderCenterline();
      renderVertexMarkers();
    });

    marker.addTo(map);
    vertexMarkers.push(marker);
  }
}

// ---------------------------------------------------------------------------
// Rendering: Timing lines
// ---------------------------------------------------------------------------
function renderTimingLines() {
  timingLines.forEach(l => l.remove());
  timingLines = [];

  if (!layers.timing) return;

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
}

// ---------------------------------------------------------------------------
// Rendering: Object markers + direction arrows
// ---------------------------------------------------------------------------
function renderObjectMarkers() {
  objectMarkers.forEach(m => m.remove());
  objectMarkers = [];
  arrowLines.forEach(l => l.remove());
  arrowLines = [];
  if (dirHandle) { dirHandle.remove(); dirHandle = null; }
  if (timingCenterHandle) { timingCenterHandle.remove(); timingCenterHandle = null; }
  if (timingWidthHandle) { timingWidthHandle.remove(); timingWidthHandle = null; }

  if (!layers.objects) return;

  const isObjMode = mode === "objects" || mode === "create" || mode === "create_direction";

  // Build idx→markerArrayIndex map (timing objects skipped in centerline mode)
  _idxToMarker = {};
  const idxToMarker = _idxToMarker;

  gameObjects.forEach((obj, idx) => {
    const type = obj.type || "unknown";
    // In centerline mode, only show non-timing as static dots
    if (mode === "centerline" && (type === "timing_left" || type === "timing_right")) return;

    const [px, py] = obj.position;
    const ll = pixelToLatLng(px, py);
    const pairIdx = findTimingPairIdx(idx);
    const isSelected = isObjMode && (idx === selectedObjIdx || (pairIdx >= 0 && pairIdx === selectedObjIdx));
    const color = objColor(type);
    const size = isSelected ? SELECTED_MARKER_SIZE : MARKER_SIZE;

    const icon = L.divIcon({
      className: "",
      html: `<div class="oe-marker ${isSelected ? "oe-marker--selected" : ""}"
                  style="width:${size}px;height:${size}px;background:${color}"></div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });

    // Timing objects are controlled via center/width/direction handles, not individually draggable
    const draggable = isObjMode && mode === "objects" && !isTimingObj(obj);
    const marker = L.marker(ll, {
      icon,
      draggable,
      zIndexOffset: isSelected ? 1000 : 0,
    });
    marker._objIdx = idx;

    // Permanent label showing object name
    marker.bindTooltip(obj.name, {
      permanent: true,
      direction: "right",
      offset: [size / 2 + 2, 0],
      className: "oe-map-label",
      opacity: 1,
    });

    if (isObjMode) {
      marker.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        if (mode !== "objects") return;
        selectObject(idx, false); // no pan on map click
      });

      if (draggable) {
        marker.on("dragstart", () => pushUndo());
        marker.on("drag", (e) => {
          const latlng = e.target.getLatLng();
          const [npx, npy] = latLngToPixel(latlng.lat, latlng.lng);
          obj.position = [Math.round(npx), Math.round(npy)];
          updateArrow(idx);
          updateDirHandle();
        });
        marker.on("dragend", () => {
          updateSelectedInfo();
        });
      }
    }

    idxToMarker[idx] = objectMarkers.length;
    marker.addTo(map);
    objectMarkers.push(marker);

    // Direction arrow
    const orient = obj.orientation_z || [0, -1];
    const arrowEnd = [px + orient[0] * ARROW_LENGTH, py + orient[1] * ARROW_LENGTH];
    const arrowLL = [ll, pixelToLatLng(arrowEnd[0], arrowEnd[1])];

    const arrow = L.polyline(arrowLL, {
      color,
      weight: isSelected ? 3 : 2,
      opacity: isSelected ? 1 : 0.6,
      interactive: false,
    });
    arrow.addTo(map);
    arrowLines.push(arrow);
  });

  // Interaction handles for selected object in objects mode
  if (mode === "objects" && selectedObjIdx >= 0 && selectedObjIdx < gameObjects.length) {
    if (isTimingObj(gameObjects[selectedObjIdx])) {
      renderTimingPairHandles();
    } else {
      renderDirHandle();
    }
  }
}

function updateArrow(idx) {
  const mi = _idxToMarker[idx];
  if (mi === undefined || mi < 0 || mi >= arrowLines.length) return;
  const obj = gameObjects[idx];
  if (!obj) return;
  const [px, py] = obj.position;
  const orient = obj.orientation_z || [0, -1];
  const arrowEnd = [px + orient[0] * ARROW_LENGTH, py + orient[1] * ARROW_LENGTH];
  const ll = pixelToLatLng(px, py);
  const endLL = pixelToLatLng(arrowEnd[0], arrowEnd[1]);
  arrowLines[mi].setLatLngs([ll, endLL]);
}

function renderDirHandle() {
  if (dirHandle) { dirHandle.remove(); dirHandle = null; }
  if (selectedObjIdx < 0 || selectedObjIdx >= gameObjects.length) return;

  const obj = gameObjects[selectedObjIdx];
  const [px, py] = obj.position;
  const orient = obj.orientation_z || [0, -1];
  const handlePx = [px + orient[0] * ARROW_LENGTH, py + orient[1] * ARROW_LENGTH];
  const handleLL = pixelToLatLng(handlePx[0], handlePx[1]);

  const icon = L.divIcon({
    className: "",
    html: '<div class="oe-dir-handle"></div>',
    iconSize: [10, 10],
    iconAnchor: [5, 5],
  });

  dirHandle = L.marker(handleLL, { icon, draggable: true, zIndexOffset: 2000 });

  dirHandle.on("dragstart", () => pushUndo());

  dirHandle.on("drag", (e) => {
    const latlng = e.target.getLatLng();
    const [hpx, hpy] = latLngToPixel(latlng.lat, latlng.lng);
    const dx = hpx - obj.position[0];
    const dy = hpy - obj.position[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > 0.1) {
      obj.orientation_z = [
        Math.round((dx / len) * 10000) / 10000,
        Math.round((dy / len) * 10000) / 10000,
      ];
      updateArrow(selectedObjIdx);
    }
  });

  dirHandle.on("dragend", () => {
    _suppressMapClick = true;
    updateSelectedInfo();
  });

  dirHandle.addTo(map);
}

function updateDirHandle() {
  if (!dirHandle || selectedObjIdx < 0) return;
  const obj = gameObjects[selectedObjIdx];
  const [px, py] = obj.position;
  const orient = obj.orientation_z || [0, -1];
  const handlePx = [px + orient[0] * ARROW_LENGTH, py + orient[1] * ARROW_LENGTH];
  dirHandle.setLatLng(pixelToLatLng(handlePx[0], handlePx[1]));
}

// ---------------------------------------------------------------------------
// Rendering: Timing pair handles (center + direction + width)
// ---------------------------------------------------------------------------
function renderTimingPairHandles() {
  if (selectedObjIdx < 0 || selectedObjIdx >= gameObjects.length) return;
  const obj = gameObjects[selectedObjIdx];
  if (!isTimingObj(obj)) return;

  const pairIdx = findTimingPairIdx(selectedObjIdx);
  if (pairIdx < 0) return;
  const pair = gameObjects[pairIdx];

  const leftObj = obj.type === "timing_left" ? obj : pair;
  const rightObj = obj.type === "timing_right" ? obj : pair;
  const leftIdx = gameObjects.indexOf(leftObj);
  const rightIdx = gameObjects.indexOf(rightObj);

  // Mutable state for drag interactions
  let center = [
    (leftObj.position[0] + rightObj.position[0]) / 2,
    (leftObj.position[1] + rightObj.position[1]) / 2,
  ];
  let orient = [...(leftObj.orientation_z || [0, -1])];
  let oLen = Math.hypot(orient[0], orient[1]);
  if (oLen > 1e-6) { orient[0] /= oLen; orient[1] /= oLen; }
  // Determine which perpendicular direction points toward L.
  // normalSign=+1 → L is at [-orient[1], orient[0]]
  // normalSign=-1 → L is at [orient[1], -orient[0]]
  // This sign is established ONCE and reused every time orient changes.
  const ldx = leftObj.position[0] - center[0];
  const ldy = leftObj.position[1] - center[1];
  const normalSign = (ldx * (-orient[1]) + ldy * orient[0]) >= 0 ? 1 : -1;
  let normal = [normalSign * -orient[1], normalSign * orient[0]];
  let halfWidth = Math.hypot(ldx, ldy) || 20;

  function applyState() {
    leftObj.position = [
      Math.round(center[0] + normal[0] * halfWidth),
      Math.round(center[1] + normal[1] * halfWidth),
    ];
    rightObj.position = [
      Math.round(center[0] - normal[0] * halfWidth),
      Math.round(center[1] - normal[1] * halfWidth),
    ];
    const oz = [
      Math.round(orient[0] * 10000) / 10000,
      Math.round(orient[1] * 10000) / 10000,
    ];
    leftObj.orientation_z = oz;
    rightObj.orientation_z = [...oz];
  }

  function refreshVisuals() {
    // Update L/R markers
    for (const gi of [leftIdx, rightIdx]) {
      const mi = _idxToMarker[gi];
      if (mi !== undefined && objectMarkers[mi]) {
        const p = gameObjects[gi].position;
        objectMarkers[mi].setLatLng(pixelToLatLng(p[0], p[1]));
      }
    }
    updateArrow(leftIdx);
    updateArrow(rightIdx);
    renderTimingLines();
  }

  // ── 1. CENTER handle (diamond) ──
  const centerIcon = L.divIcon({
    className: "",
    html: `<div style="width:14px;height:14px;background:#fbbf24;border:2px solid #fff;
      border-radius:3px;cursor:grab;box-shadow:0 1px 4px rgba(0,0,0,0.5);
      transform:rotate(45deg)"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });

  timingCenterHandle = L.marker(pixelToLatLng(center[0], center[1]), {
    icon: centerIcon, draggable: true, zIndexOffset: 2000,
  });
  timingCenterHandle.bindTooltip("center", {
    permanent: true, direction: "bottom", offset: [0, 10],
    className: "oe-map-label", opacity: 1,
  });

  timingCenterHandle.on("dragstart", () => pushUndo());
  timingCenterHandle.on("drag", (e) => {
    const [npx, npy] = latLngToPixel(e.target.getLatLng().lat, e.target.getLatLng().lng);
    center = [npx, npy];
    applyState();
    refreshVisuals();
    // Sync other handles
    if (timingWidthHandle) {
      timingWidthHandle.setLatLng(pixelToLatLng(leftObj.position[0], leftObj.position[1]));
    }
    if (dirHandle) {
      const dp = [center[0] + orient[0] * ARROW_LENGTH, center[1] + orient[1] * ARROW_LENGTH];
      dirHandle.setLatLng(pixelToLatLng(dp[0], dp[1]));
    }
  });
  timingCenterHandle.on("dragend", () => { _suppressMapClick = true; updateSelectedInfo(); });
  timingCenterHandle.addTo(map);

  // ── 2. DIRECTION handle (arrow tip) ──
  const dirPos = [center[0] + orient[0] * ARROW_LENGTH, center[1] + orient[1] * ARROW_LENGTH];
  const dIcon = L.divIcon({
    className: "",
    html: '<div class="oe-dir-handle"></div>',
    iconSize: [10, 10],
    iconAnchor: [5, 5],
  });

  dirHandle = L.marker(pixelToLatLng(dirPos[0], dirPos[1]), {
    icon: dIcon, draggable: true, zIndexOffset: 2000,
  });
  dirHandle.bindTooltip("direction", {
    permanent: true, direction: "right", offset: [8, 0],
    className: "oe-map-label", opacity: 1,
  });

  dirHandle.on("dragstart", () => pushUndo());
  dirHandle.on("drag", (e) => {
    const [hpx, hpy] = latLngToPixel(e.target.getLatLng().lat, e.target.getLatLng().lng);
    const dx = hpx - center[0];
    const dy = hpy - center[1];
    const len = Math.hypot(dx, dy);
    if (len > 0.5) {
      orient = [dx / len, dy / len];
      normal = [normalSign * -orient[1], normalSign * orient[0]];
      applyState();
      refreshVisuals();
      if (timingWidthHandle) {
        timingWidthHandle.setLatLng(pixelToLatLng(leftObj.position[0], leftObj.position[1]));
      }
    }
  });
  dirHandle.on("dragend", () => { _suppressMapClick = true; updateSelectedInfo(); });
  dirHandle.addTo(map);

  // ── 3. WIDTH handle (at L position) ──
  const wIcon = L.divIcon({
    className: "",
    html: `<div style="width:10px;height:10px;background:#ff8c00;border:2px solid #fff;
      border-radius:50%;cursor:ew-resize;box-shadow:0 1px 3px rgba(0,0,0,0.5)"></div>`,
    iconSize: [10, 10],
    iconAnchor: [5, 5],
  });

  timingWidthHandle = L.marker(pixelToLatLng(leftObj.position[0], leftObj.position[1]), {
    icon: wIcon, draggable: true, zIndexOffset: 2000,
  });
  timingWidthHandle.bindTooltip("width", {
    permanent: true, direction: "right", offset: [8, 0],
    className: "oe-map-label", opacity: 1,
  });

  timingWidthHandle.on("dragstart", () => pushUndo());
  timingWidthHandle.on("drag", (e) => {
    const [wpx, wpy] = latLngToPixel(e.target.getLatLng().lat, e.target.getLatLng().lng);
    // Project drag position onto normal from center → new half-width
    const dx = wpx - center[0];
    const dy = wpy - center[1];
    halfWidth = Math.max(5, Math.abs(dx * normal[0] + dy * normal[1]));
    applyState();
    refreshVisuals();
    // Snap handle to actual L position (constrained to normal)
    e.target.setLatLng(pixelToLatLng(leftObj.position[0], leftObj.position[1]));
  });
  timingWidthHandle.on("dragend", () => { _suppressMapClick = true; updateSelectedInfo(); });
  timingWidthHandle.addTo(map);
}

// ---------------------------------------------------------------------------
// Rendering: Bend list + Info
// ---------------------------------------------------------------------------
function renderBendList() {
  const el = $("bendList");
  el.innerHTML = "";

  bends.forEach((bend, i) => {
    const angleDeg = Math.round((bend.total_angle || 0) * 180 / Math.PI);
    const color = BEND_COLORS[i % BEND_COLORS.length];
    const turnLabel = bend.turn_label || `B${i}`;
    const timingName = bend.turn_label ? `AC_TIME_${i + 1}` : `TIME_${i}`;

    const div = document.createElement("div");
    div.className = "ce-bend-item";
    div.innerHTML = `
      <div class="ce-bend-swatch" style="background:${color}"></div>
      <span>${turnLabel} (${timingName}): ${angleDeg}&deg;</span>
    `;

    div.addEventListener("click", () => {
      const peakIdx = bend.peak_idx || bend.start_idx;
      if (peakIdx < centerline.length) {
        const [x, y] = centerline[peakIdx];
        map.panTo(pixelToLatLng(x, y));
      }
    });

    el.appendChild(div);
  });
}

function renderObjectList() {
  const el = $("objectList");
  el.innerHTML = "";

  const isObjMode = mode === "objects" || mode === "create" || mode === "create_direction";

  // Separate timing pairs from other objects
  const timingPairs = {}; // num -> { L: {obj,idx}, R: {obj,idx} }
  const nonTimingGroups = {};

  gameObjects.forEach((obj, idx) => {
    const t = obj.type || "unknown";
    if (isTimingObj(obj)) {
      const num = timingPairNum(obj);
      if (num >= 0) {
        if (!timingPairs[num]) timingPairs[num] = {};
        const side = obj.name.endsWith("_L") ? "L" : "R";
        timingPairs[num][side] = { obj, idx };
        return;
      }
    }
    if (!nonTimingGroups[t]) nonTimingGroups[t] = [];
    nonTimingGroups[t].push({ obj, idx });
  });

  // Render non-timing groups
  const typeOrder = ["hotlap_start", "pit", "start"];
  const sortedTypes = Object.keys(nonTimingGroups).sort(
    (a, b) => (typeOrder.indexOf(a) === -1 ? 99 : typeOrder.indexOf(a)) -
              (typeOrder.indexOf(b) === -1 ? 99 : typeOrder.indexOf(b))
  );

  for (const type of sortedTypes) {
    const header = document.createElement("div");
    header.className = "oe-group-header";
    header.textContent = `${type} (${nonTimingGroups[type].length})`;
    el.appendChild(header);

    for (const { obj, idx } of nonTimingGroups[type]) {
      const div = document.createElement("div");
      div.className = `oe-obj-item ${isObjMode && idx === selectedObjIdx ? "oe-obj-item--selected" : ""}`;
      div.innerHTML = `
        <div class="oe-obj-swatch" style="background:${objColor(obj.type)}"></div>
        <div class="oe-obj-label">${obj.name}</div>
        <button class="oe-obj-del" data-idx="${idx}" title="Delete">&times;</button>
      `;

      div.addEventListener("click", (e) => {
        if (e.target.closest(".oe-obj-del")) return;
        if (mode !== "objects") setMode("objects");
        selectObject(idx, true); // pan from list
      });

      div.querySelector(".oe-obj-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteObject(idx);
      });

      el.appendChild(div);
    }
  }

  // Render timing pairs grouped
  const pairNums = Object.keys(timingPairs).map(Number).sort((a, b) => a - b);
  if (pairNums.length > 0) {
    const header = document.createElement("div");
    header.className = "oe-group-header";
    header.textContent = `timing (${pairNums.length} pairs)`;
    el.appendChild(header);

    for (const num of pairNums) {
      const pair = timingPairs[num];
      const leftIdx = pair.L ? pair.L.idx : -1;
      const rightIdx = pair.R ? pair.R.idx : -1;
      const anySelected = isObjMode && (leftIdx === selectedObjIdx || rightIdx === selectedObjIdx);
      const primaryIdx = leftIdx >= 0 ? leftIdx : rightIdx;

      const div = document.createElement("div");
      div.className = `oe-timing-pair ${anySelected ? "oe-timing-pair--selected" : ""}`;
      div.innerHTML = `
        <div class="oe-obj-swatch" style="background:${OBJ_COLORS.timing_left}"></div>
        <div class="oe-obj-label">AC_TIME_${num} (L + R)</div>
        <button class="oe-obj-del" title="Delete pair">&times;</button>
      `;

      div.addEventListener("click", (e) => {
        if (e.target.closest(".oe-obj-del")) return;
        if (mode !== "objects") setMode("objects");
        selectObject(primaryIdx, true); // pan from list
      });

      div.querySelector(".oe-obj-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteObject(primaryIdx); // deleteObject handles pair deletion
      });

      el.appendChild(div);
    }
  }
}

function updateInfo() {
  $("clInfo").textContent = `Points: ${centerline.length} | Edited: ${edited ? "Yes" : "No"} | Bends: ${bends.length}`;

  // Pre-fill pit/start counts only when in auto mode
  if ($("pitAuto").checked) {
    const pitCount = gameObjects.filter(o => o.type === "pit").length;
    $("pitCount").value = pitCount || 8;
  }
  if ($("startAuto").checked) {
    const startCount = gameObjects.filter(o => o.type === "start").length;
    $("startCount").value = startCount || 8;
  }
}

function updateSelectedInfo() {
  const sec = $("selObjectSection");
  if (selectedObjIdx < 0 || selectedObjIdx >= gameObjects.length) {
    sec.hidden = true;
    return;
  }
  sec.hidden = false;
  const obj = gameObjects[selectedObjIdx];

  if (isTimingObj(obj)) {
    const pairIdx = findTimingPairIdx(selectedObjIdx);
    const pair = pairIdx >= 0 ? gameObjects[pairIdx] : null;
    const num = timingPairNum(obj);
    $("selObjName").textContent = `AC_TIME_${num} (pair)`;
    $("selObjType").value = obj.type;

    if (pair) {
      const cx = (obj.position[0] + pair.position[0]) / 2;
      const cy = (obj.position[1] + pair.position[1]) / 2;
      const w = Math.hypot(obj.position[0] - pair.position[0], obj.position[1] - pair.position[1]);
      $("selObjPixel").textContent = `Center: (${Math.round(cx)}, ${Math.round(cy)}) | Width: ${Math.round(w)}px`;
    } else {
      $("selObjPixel").textContent = `Position: (${Math.round(obj.position[0])}, ${Math.round(obj.position[1])})`;
    }

    const orient = obj.orientation_z || [0, -1];
    const deg = bearing(orient[0], orient[1]);
    $("selObjBearing").textContent = `Bearing: ${deg.toFixed(1)}\u00b0`;
  } else {
    $("selObjName").textContent = obj.name;
    $("selObjType").value = obj.type || "pit";

    const [px, py] = obj.position;
    $("selObjPixel").textContent = `Position: (${Math.round(px)}, ${Math.round(py)})`;

    const orient = obj.orientation_z || [0, -1];
    const deg = bearing(orient[0], orient[1]);
    $("selObjBearing").textContent = `Bearing: ${deg.toFixed(1)}\u00b0  [${orient[0].toFixed(3)}, ${orient[1].toFixed(3)}]`;
  }
}

// ---------------------------------------------------------------------------
// Render all
// ---------------------------------------------------------------------------
function renderAll() {
  renderCenterline();
  renderVertexMarkers();
  renderTimingLines();
  renderObjectMarkers();
  renderBendList();
  renderObjectList();
  updateInfo();
  updateSelectedInfo();
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------
function selectObject(idx, panTo = false) {
  if (idx === selectedObjIdx) return;
  selectedObjIdx = idx;
  renderObjectMarkers();
  renderObjectList();
  updateSelectedInfo();

  if (panTo && idx >= 0 && idx < gameObjects.length) {
    const [px, py] = gameObjects[idx].position;
    map.panTo(pixelToLatLng(px, py));
  }
}

function deselectAll() {
  selectedObjIdx = -1;
  renderObjectMarkers();
  renderObjectList();
  updateSelectedInfo();
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------
function deleteObject(idx) {
  if (idx < 0 || idx >= gameObjects.length) return;
  pushUndo();

  // For timing objects, delete the pair together
  const pairIdx = findTimingPairIdx(idx);
  if (pairIdx >= 0) {
    // Remove higher index first to preserve lower index
    const hi = Math.max(idx, pairIdx);
    const lo = Math.min(idx, pairIdx);
    gameObjects.splice(hi, 1);
    gameObjects.splice(lo, 1);
    selectedObjIdx = -1;
  } else {
    gameObjects.splice(idx, 1);
    if (selectedObjIdx === idx) selectedObjIdx = -1;
    else if (selectedObjIdx > idx) selectedObjIdx--;
  }

  renderAll();
  setStatus("Object deleted");
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------
function setMode(newMode) {
  // Clean up previous mode
  if (mode === "create" || mode === "create_direction") {
    createPendingPos = null;
    clearRubberArrow();
    $("map").classList.remove("map--create");
    map.doubleClickZoom.enable();
  }

  mode = newMode;

  // Update segmented control
  document.querySelectorAll("#modeSeg .le-seg__btn").forEach(btn => {
    btn.classList.toggle("le-seg__btn--active", btn.dataset.mode === newMode);
  });

  // Show/hide create type row
  $("createTypeRow").style.display = newMode === "create" ? "" : "none";

  if (newMode === "centerline") {
    deselectAll();
  } else if (newMode === "objects") {
    // Nothing special
  } else if (newMode === "create") {
    deselectAll();
    $("map").classList.add("map--create");
    map.doubleClickZoom.disable();
    setStatus("Create mode: click to place object, click again for direction, Esc to cancel");
  }

  renderAll();
}

// ---------------------------------------------------------------------------
// Create mode
// ---------------------------------------------------------------------------
function clearRubberArrow() {
  if (rubberArrow) { rubberArrow.remove(); rubberArrow = null; }
}

function handleCreateClick(e) {
  const [px, py] = latLngToPixel(e.latlng.lat, e.latlng.lng);

  if (mode === "create") {
    createPendingPos = [Math.round(px), Math.round(py)];
    mode = "create_direction";
    setStatus("Move mouse to set direction, click to confirm");
    return;
  }

  if (mode === "create_direction" && createPendingPos) {
    const dx = px - createPendingPos[0];
    const dy = py - createPendingPos[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    const orient = len > 1 ? [dx / len, dy / len] : [0, -1];
    const orientRound = [
      Math.round(orient[0] * 10000) / 10000,
      Math.round(orient[1] * 10000) / 10000,
    ];

    pushUndo();

    if (createType === "timing") {
      // Create L+R pair symmetrically around the center point
      const TIMING_SPREAD = 35; // px half-spread
      const normal = [-orient[1], orient[0]]; // perpendicular to direction
      const existingNums = gameObjects
        .filter(o => isTimingObj(o))
        .map(o => timingPairNum(o))
        .filter(n => n >= 0);
      const nextNum = existingNums.length > 0 ? Math.max(...existingNums) + 1 : 0;

      const leftPos = [
        Math.round(createPendingPos[0] + normal[0] * TIMING_SPREAD),
        Math.round(createPendingPos[1] + normal[1] * TIMING_SPREAD),
      ];
      const rightPos = [
        Math.round(createPendingPos[0] - normal[0] * TIMING_SPREAD),
        Math.round(createPendingPos[1] - normal[1] * TIMING_SPREAD),
      ];

      gameObjects.push({
        name: `AC_TIME_${nextNum}_L`,
        position: leftPos,
        orientation_z: [...orientRound],
        type: "timing_left",
      });
      gameObjects.push({
        name: `AC_TIME_${nextNum}_R`,
        position: rightPos,
        orientation_z: [...orientRound],
        type: "timing_right",
      });

      selectedObjIdx = gameObjects.length - 2; // select the L marker
      createPendingPos = null;
      mode = "create";
      clearRubberArrow();
      renderAll();
      setStatus(`Created AC_TIME_${nextNum} L+R pair — click to place another, Esc to exit`);
    } else {
      const name = generateName(createType);
      gameObjects.push({
        name,
        position: createPendingPos,
        orientation_z: orientRound,
        type: createType,
      });

      selectedObjIdx = gameObjects.length - 1;
      createPendingPos = null;
      mode = "create";
      clearRubberArrow();
      renderAll();
      setStatus(`Created ${name} — click to place another, Esc to exit`);
    }
  }
}

function handleCreateMouseMove(e) {
  if (mode !== "create_direction" || !createPendingPos) return;
  const posLL = pixelToLatLng(createPendingPos[0], createPendingPos[1]);
  const curLL = [e.latlng.lat, e.latlng.lng];

  if (rubberArrow) rubberArrow.setLatLngs([posLL, curLL]);
  else {
    rubberArrow = L.polyline([posLL, curLL], {
      color: objColor(createType),
      weight: 2,
      dashArray: "6 3",
      opacity: 0.7,
      interactive: false,
    }).addTo(map);
  }
}

// ---------------------------------------------------------------------------
// Layout loading
// ---------------------------------------------------------------------------
async function loadLayout(layoutName) {
  selectedLayout = layoutName;
  setStatus(`Loading layout: ${layoutName}...`);

  // Reset state
  undoStack = [];
  redoStack = [];
  dirty = false;
  $("dirtyFlag").hidden = true;
  selectedObjIdx = -1;

  // Load centerline (8a > 8 priority handled server-side)
  try {
    const resp = await fetch(`/api/layout_centerline/${encodeURIComponent(layoutName)}`);
    if (resp.ok) {
      const data = await resp.json();
      centerline = data.centerline || [];
      bends = data.bends || [];
      edited = data.edited || false;
      trackDirection = data.track_direction || "clockwise";
      time0Idx = data.time0_idx != null ? data.time0_idx : null;
    } else {
      centerline = [];
      bends = [];
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
      gameObjects = (data.objects || []).map(o => ({
        name: o.name || "",
        position: (o.position || [0, 0]).map(Number),
        orientation_z: (o.orientation_z || [0, -1]).map(Number),
        type: o.type || "unknown",
      }));
      trackDirection = data.track_direction || trackDirection;
    } else {
      gameObjects = [];
    }
  } catch {
    gameObjects = [];
  }

  // Layout mask overlay
  if (maskOverlay) maskOverlay.remove();
  const safeName = layoutName.replace(/[^\w\-]/g, '_').replace(/^_+|_+$/g, '') || "unnamed";
  maskOverlay = L.imageOverlay(`/api/layout_mask/${safeName}`, imageBounds, {
    opacity: 0.25,
    interactive: false,
  }).addTo(map);
  if (!layers.mask && maskOverlay) maskOverlay.setOpacity(0);

  renderAll();
  setStatus(`Layout "${layoutName}": ${centerline.length} pts, ${bends.length} bends, ${gameObjects.length} objects`);
}

// ---------------------------------------------------------------------------
// Save All
// ---------------------------------------------------------------------------
async function save() {
  if (!selectedLayout) return;
  setStatus("Saving...");
  try {
    // Save centerline
    const clData = {
      layout_name: selectedLayout,
      centerline,
      bends,
      edited,
      track_direction: trackDirection,
    };
    if (time0Idx != null) clData.time0_idx = time0Idx;
    const clResp = await fetch(`/api/layout_centerline/${encodeURIComponent(selectedLayout)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(clData),
    });
    if (!clResp.ok) throw new Error(`centerline save: ${clResp.status}`);

    // Save game objects
    const goData = {
      layout_name: selectedLayout,
      track_direction: trackDirection,
      objects: gameObjects,
    };
    const goResp = await fetch(`/api/layout_game_objects/${encodeURIComponent(selectedLayout)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(goData),
    });
    if (!goResp.ok) throw new Error(`game_objects save: ${goResp.status}`);

    markClean();
    setStatus(`Saved centerline + game objects for "${selectedLayout}"`);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Regenerate bends & timing (from centerline)
// ---------------------------------------------------------------------------
async function regenerateCenterline() {
  if (!selectedLayout || centerline.length < 10) {
    setStatus("Need at least 10 centerline points");
    return;
  }

  const statusEl = $("regenCLStatus");
  statusEl.textContent = "Regenerating...";
  setStatus("Regenerating bends & timing...");

  try {
    const regenBody = {
      centerline,
      layout_name: selectedLayout,
      track_direction: trackDirection,
    };
    if (time0Idx != null) regenBody.time0_idx = time0Idx;

    const resp = await fetch("/api/centerline/regenerate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(regenBody),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text}`);
    }

    const result = await resp.json();
    pushUndo();

    bends = result.bends || [];
    const newTimingObjects = (result.timing_objects || []).map(o => ({
      name: o.name || "",
      position: (o.position || [0, 0]).map(Number),
      orientation_z: (o.orientation_z || [0, -1]).map(Number),
      type: o.type || "unknown",
    }));

    // Replace timing objects, keep non-timing
    const nonTiming = gameObjects.filter(
      o => o.type !== "timing_left" && o.type !== "timing_right"
    );
    gameObjects = nonTiming.concat(newTimingObjects);

    // Auto-save
    await save();

    renderAll();
    statusEl.textContent = `${bends.length} bends, ${newTimingObjects.length} timing`;
    setStatus(`Regenerated: ${bends.length} bends, ${newTimingObjects.length} timing objects`);
  } catch (err) {
    statusEl.textContent = "Failed";
    setStatus(`Regeneration failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Per-type VLM regeneration
// ---------------------------------------------------------------------------

function _parseVLMObjects(rawList) {
  return (rawList || []).map(o => ({
    name: o.name || "",
    position: (o.position || [0, 0]).map(Number),
    orientation_z: (o.orientation_z || [0, -1]).map(Number),
    type: o.type || "unknown",
  }));
}

function _validationSummary(val) {
  if (!val) return "";
  const parts = [];
  for (const [type, v] of Object.entries(val)) {
    if (v && typeof v.passed === "number") {
      const pct = v.total > 0 ? Math.round(v.passed / v.total * 100) : 0;
      parts.push(`${type}: ${v.passed}/${v.total} (${pct}%)`);
    }
  }
  return parts.join(", ");
}

/**
 * Regenerate a single object type via VLM.
 * @param {string} objectType - "hotlap", "pit", "start", "timing_0", or "all"
 * @param {string} statusElId - ID of status span element
 */
async function regenerateVLMType(objectType, statusElId) {
  const pitAuto = $("pitAuto").checked;
  const startAuto = $("startAuto").checked;
  const pitCount = pitAuto ? null : (parseInt($("pitCount").value) || 8);
  const startCount = startAuto ? null : (parseInt($("startCount").value) || 8);

  const statusEl = $(statusElId);
  statusEl.textContent = "Generating...";
  setStatus(`Regenerating ${objectType} via VLM...`);

  try {
    const resp = await fetch("/api/vlm_objects/regenerate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        layout_name: selectedLayout,
        object_type: objectType,
        pit_count: pitCount,
        start_count: startCount,
        track_direction: trackDirection,
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text}`);
    }

    const result = await resp.json();
    const newObjects = _parseVLMObjects(result.objects);

    pushUndo();

    if (objectType === "all") {
      // Replace all non-timing VLM objects + timing
      gameObjects = newObjects;
    } else if (objectType === "timing_0" || result.replace_type === "timing") {
      // Replace ALL timing objects (TIME_0 + TIME_1/2/... are regenerated together)
      const nonTiming = gameObjects.filter(
        o => o.type !== "timing_left" && o.type !== "timing_right"
      );
      gameObjects = nonTiming.concat(newObjects);
      // Reload centerline data (bends may have been updated with turn labels)
      await _reloadCenterlineData();
    } else if (objectType === "hotlap") {
      const kept = gameObjects.filter(o => o.type !== "hotlap_start");
      gameObjects = kept.concat(newObjects);
    } else if (objectType === "pit") {
      const kept = gameObjects.filter(o => o.type !== "pit");
      gameObjects = kept.concat(newObjects);
    } else if (objectType === "start") {
      const kept = gameObjects.filter(o => o.type !== "start");
      gameObjects = kept.concat(newObjects);
    }

    selectedObjIdx = -1;
    renderAll();

    const valSummary = _validationSummary(result.validation);
    const countStr = `${newObjects.length} objects`;
    statusEl.textContent = valSummary || countStr;
    setStatus(`VLM [${objectType}]: ${countStr}. ${valSummary}`);
  } catch (err) {
    statusEl.textContent = "Failed";
    setStatus(`VLM [${objectType}] failed: ${err.message}`);
  }
}

async function _reloadCenterlineData() {
  try {
    const resp = await fetch(`/api/layout_centerline/${encodeURIComponent(selectedLayout)}`);
    if (resp.ok) {
      const data = await resp.json();
      bends = data.bends || [];
      if (data.time0_idx != null) time0Idx = data.time0_idx;
    }
  } catch {}
}

// Legacy wrapper
async function regenerateVLM() {
  await regenerateVLMType("all", "regenAllStatus");
}

// ---------------------------------------------------------------------------
// Layer visibility
// ---------------------------------------------------------------------------
function toggleLayer(layerName) {
  layers[layerName] = !layers[layerName];

  if (layerName === "basemap" && baseImageOverlay) {
    baseImageOverlay.setOpacity(layers.basemap ? 1.0 : 0);
  }
  if (layerName === "mask" && maskOverlay) {
    maskOverlay.setOpacity(layers.mask ? 0.25 : 0);
  }

  renderAll();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
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

  map.fitBounds(imageBounds);

  // Map events
  map.on("click", (e) => {
    if (_suppressMapClick) { _suppressMapClick = false; return; }
    if (mode === "create" || mode === "create_direction") {
      handleCreateClick(e);
      return;
    }
    if (mode === "objects") {
      deselectAll();
    }
  });

  map.on("mousemove", (e) => {
    handleCreateMouseMove(e);
  });

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
    select.innerHTML = '<option value="">Default (no layouts)</option>';
    setStatus("No layouts found. Run stage 2a first.");
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

function wireUI() {
  // Layout select
  $("layoutSelect").addEventListener("change", (e) => {
    if (e.target.value) loadLayout(e.target.value);
  });

  // Mode segmented control
  document.querySelectorAll("#modeSeg .le-seg__btn").forEach(btn => {
    btn.addEventListener("click", () => {
      setMode(btn.dataset.mode);
    });
  });

  // Create type
  $("createType").addEventListener("change", (e) => {
    createType = e.target.value;
    if (rubberArrow) rubberArrow.setStyle({ color: objColor(createType) });
  });

  // Toolbar
  $("btnSave").addEventListener("click", () => save());
  $("btnUndo").addEventListener("click", () => undo());
  $("btnRedo").addEventListener("click", () => redo());

  // Actions
  $("btnRegenCenterline").addEventListener("click", () => regenerateCenterline());
  $("btnRegenHotlap").addEventListener("click", () => regenerateVLMType("hotlap", "regenHotlapStatus"));
  $("btnRegenPits").addEventListener("click", () => regenerateVLMType("pit", "regenPitsStatus"));
  $("btnRegenStarts").addEventListener("click", () => regenerateVLMType("start", "regenStartsStatus"));
  $("btnRegenTiming0").addEventListener("click", () => regenerateVLMType("timing_0", "regenTiming0Status"));
  $("btnRegenAll").addEventListener("click", () => regenerateVLMType("all", "regenAllStatus"));

  // Auto checkboxes toggle count inputs
  $("pitAuto").addEventListener("change", () => {
    $("pitCount").disabled = $("pitAuto").checked;
  });
  $("startAuto").addEventListener("change", () => {
    $("startCount").disabled = $("startAuto").checked;
  });

  // Selected object type change
  $("selObjType").addEventListener("change", (e) => {
    if (selectedObjIdx < 0) return;
    pushUndo();
    gameObjects[selectedObjIdx].type = e.target.value;
    renderAll();
  });

  // Delete selected object
  $("btnDeleteObj").addEventListener("click", () => {
    if (selectedObjIdx >= 0) deleteObject(selectedObjIdx);
  });

  // Layer chips
  document.querySelectorAll("#layerChips .le-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const layerName = chip.dataset.layer;
      chip.classList.toggle("le-chip--on");
      toggleLayer(layerName);
    });
  });

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
    if (e.key === "Delete" || e.key === "Backspace") {
      if (mode === "objects" && selectedObjIdx >= 0) {
        deleteObject(selectedObjIdx);
      }
      return;
    }
    if (e.key === "Escape") {
      if (mode === "create" || mode === "create_direction") {
        setMode("objects");
      } else {
        deselectAll();
      }
      return;
    }
    // Quick mode switches
    if (e.key === "1") setMode("centerline");
    if (e.key === "2") setMode("objects");
    if (e.key === "3") setMode("create");
  });

  // Before-unload warning
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
