/* global L */
/* Objects Editor — interactive game_objects.json editor on Leaflet map */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const TILE_URL = "/tiles/{z}/{x}/{y}.png";
const OBJ_COLORS = {
  hotlap_start: "#ff00ff",
  pit: "#00ffff",
  start: "#ffff00",
  timing_left: "#ff8c00",
  timing_right: "#ff4500",
};
const DEFAULT_COLOR = "#ffffff";
const MARKER_SIZE = 14;
const SELECTED_MARKER_SIZE = 18;
const ARROW_LENGTH = 40; // px on screen
const UNDO_LIMIT = 50;

const $ = (id) => document.getElementById(id);

function setStatus(msg) {
  $("status").textContent = msg;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let geoMeta = null;
let objects = []; // array of { name, position:[x,y], orientation_z:[dx,dy], type }
let trackDirection = "clockwise";
let selectedIdx = -1;
let mode = "select"; // "select" | "create" | "create_direction"
let createType = "pit";
let createPendingPos = null; // pixel [x,y] during direction-setting sub-mode
let undoStack = [];
let redoStack = [];
let dirty = false;

// Leaflet objects
let map;
let tileLayer;
let objMarkers = []; // L.marker per object
let arrowLines = []; // L.polyline per object (direction arrow)
let dirHandle = null; // L.marker for direction editing on selected object
let centerlineLayer = null; // L.polyline for centerline overlay
let rubberArrow = null; // L.polyline during create_direction mode

// ---------------------------------------------------------------------------
// Coordinate conversion
// ---------------------------------------------------------------------------
function pixelToLatLng(px, py) {
  if (!geoMeta) return [0, 0];
  const c = geoMeta.corners;
  if (c) {
    const u = px / geoMeta.image_width;
    const v = py / geoMeta.image_height;
    const tl = c.top_left, tr = c.top_right, bl = c.bottom_left, br = c.bottom_right;
    const lat = (1-u)*(1-v)*tl[0] + u*(1-v)*tr[0] + (1-u)*v*bl[0] + u*v*br[0];
    const lng = (1-u)*(1-v)*tl[1] + u*(1-v)*tr[1] + (1-u)*v*bl[1] + u*v*br[1];
    return [lat, lng];
  }
  const { north, south, east, west } = geoMeta.bounds;
  const lat = north - (py / geoMeta.image_height) * (north - south);
  const lng = west + (px / geoMeta.image_width) * (east - west);
  return [lat, lng];
}

function latLngToPixel(lat, lng) {
  if (!geoMeta) return [0, 0];
  const c = geoMeta.corners;
  if (c) {
    const tl = c.top_left, tr = c.top_right, bl = c.bottom_left, br = c.bottom_right;
    let u = 0.5, v = 0.5;
    for (let i = 0; i < 4; i++) {
      const fLat = (1-u)*(1-v)*tl[0] + u*(1-v)*tr[0] + (1-u)*v*bl[0] + u*v*br[0] - lat;
      const fLng = (1-u)*(1-v)*tl[1] + u*(1-v)*tr[1] + (1-u)*v*bl[1] + u*v*br[1] - lng;
      const dLat_du = (1-v)*(tr[0]-tl[0]) + v*(br[0]-bl[0]);
      const dLat_dv = (1-u)*(bl[0]-tl[0]) + u*(br[0]-tr[0]);
      const dLng_du = (1-v)*(tr[1]-tl[1]) + v*(br[1]-bl[1]);
      const dLng_dv = (1-u)*(bl[1]-tl[1]) + u*(br[1]-tr[1]);
      const det = dLat_du * dLng_dv - dLat_dv * dLng_du;
      if (Math.abs(det) < 1e-20) break;
      u -= (fLat * dLng_dv - fLng * dLat_dv) / det;
      v -= (fLng * dLat_du - fLat * dLng_du) / det;
    }
    return [u * geoMeta.image_width, v * geoMeta.image_height];
  }
  const { north, south, east, west } = geoMeta.bounds;
  const px = ((lng - west) / (east - west)) * geoMeta.image_width;
  const py = ((north - lat) / (north - south)) * geoMeta.image_height;
  return [px, py];
}

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------
function pushUndo() {
  undoStack.push(JSON.parse(JSON.stringify(objects)));
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack.length = 0;
  markDirty();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.parse(JSON.stringify(objects)));
  objects = undoStack.pop();
  afterObjectsChange();
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.parse(JSON.stringify(objects)));
  objects = redoStack.pop();
  afterObjectsChange();
}

function markDirty() {
  dirty = true;
  $("dirtyFlag").hidden = false;
}

function markClean() {
  dirty = false;
  $("dirtyFlag").hidden = true;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function objColor(type) {
  return OBJ_COLORS[type] || DEFAULT_COLOR;
}

function bearing(dx, dy) {
  // orientation_z [dx, dy] -> degrees from north (0=up, 90=right)
  return ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
}

function renderObjectMarkers() {
  // Remove old
  objMarkers.forEach((m) => m.remove());
  objMarkers = [];
  arrowLines.forEach((l) => l.remove());
  arrowLines = [];
  if (dirHandle) {
    dirHandle.remove();
    dirHandle = null;
  }

  objects.forEach((obj, idx) => {
    const [px, py] = obj.position;
    const ll = pixelToLatLng(px, py);
    const isSelected = idx === selectedIdx;
    const color = objColor(obj.type);
    const size = isSelected ? SELECTED_MARKER_SIZE : MARKER_SIZE;

    const icon = L.divIcon({
      className: "",
      html: `<div class="oe-marker ${isSelected ? "oe-marker--selected" : ""}"
                  style="width:${size}px;height:${size}px;background:${color}"></div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });

    const marker = L.marker(ll, {
      icon,
      draggable: mode === "select",
      zIndexOffset: isSelected ? 1000 : 0,
    });

    marker.on("click", (e) => {
      L.DomEvent.stopPropagation(e);
      if (mode !== "select") return;
      selectObject(idx);
    });

    marker.on("dragstart", () => pushUndo());

    marker.on("drag", (e) => {
      const latlng = e.target.getLatLng();
      const [npx, npy] = latLngToPixel(latlng.lat, latlng.lng);
      obj.position = [Math.round(npx), Math.round(npy)];
      updateArrow(idx);
      updateDirHandle();
    });

    marker.on("dragend", (e) => {
      const latlng = e.target.getLatLng();
      const [npx, npy] = latLngToPixel(latlng.lat, latlng.lng);
      obj.position = [Math.round(npx), Math.round(npy)];
      updateArrow(idx);
      updateSelectedInfo();
    });

    marker.addTo(map);
    objMarkers.push(marker);

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

  // Direction handle for selected
  if (selectedIdx >= 0 && selectedIdx < objects.length && mode === "select") {
    renderDirHandle();
  }
}

function updateArrow(idx) {
  if (idx < 0 || idx >= objects.length || idx >= arrowLines.length) return;
  const obj = objects[idx];
  const [px, py] = obj.position;
  const orient = obj.orientation_z || [0, -1];
  const arrowEnd = [px + orient[0] * ARROW_LENGTH, py + orient[1] * ARROW_LENGTH];
  const ll = pixelToLatLng(px, py);
  const endLL = pixelToLatLng(arrowEnd[0], arrowEnd[1]);
  arrowLines[idx].setLatLngs([ll, endLL]);
}

function renderDirHandle() {
  if (dirHandle) {
    dirHandle.remove();
    dirHandle = null;
  }
  if (selectedIdx < 0 || selectedIdx >= objects.length) return;

  const obj = objects[selectedIdx];
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
      updateArrow(selectedIdx);
    }
  });

  dirHandle.on("dragend", () => {
    updateSelectedInfo();
  });

  dirHandle.addTo(map);
}

function updateDirHandle() {
  if (!dirHandle || selectedIdx < 0) return;
  const obj = objects[selectedIdx];
  const [px, py] = obj.position;
  const orient = obj.orientation_z || [0, -1];
  const handlePx = [px + orient[0] * ARROW_LENGTH, py + orient[1] * ARROW_LENGTH];
  dirHandle.setLatLng(pixelToLatLng(handlePx[0], handlePx[1]));
}

function renderObjectList() {
  const el = $("objectList");
  el.innerHTML = "";

  // Group by type
  const groups = {};
  objects.forEach((obj, idx) => {
    const t = obj.type || "unknown";
    if (!groups[t]) groups[t] = [];
    groups[t].push({ obj, idx });
  });

  const typeOrder = ["hotlap_start", "pit", "start", "timing_left", "timing_right"];
  const sortedTypes = Object.keys(groups).sort(
    (a, b) => (typeOrder.indexOf(a) === -1 ? 99 : typeOrder.indexOf(a)) -
              (typeOrder.indexOf(b) === -1 ? 99 : typeOrder.indexOf(b))
  );

  for (const type of sortedTypes) {
    const header = document.createElement("div");
    header.className = "oe-group-header";
    header.textContent = `${type} (${groups[type].length})`;
    el.appendChild(header);

    for (const { obj, idx } of groups[type]) {
      const div = document.createElement("div");
      div.className = `oe-obj-item ${idx === selectedIdx ? "oe-obj-item--selected" : ""}`;
      div.innerHTML = `
        <div class="oe-obj-swatch" style="background:${objColor(obj.type)}"></div>
        <div class="oe-obj-label">${obj.name}</div>
        <button class="oe-obj-del" data-idx="${idx}" title="删除">&times;</button>
      `;

      div.addEventListener("click", (e) => {
        if (e.target.closest(".oe-obj-del")) return;
        selectObject(idx);
      });

      div.querySelector(".oe-obj-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteObject(idx);
      });

      el.appendChild(div);
    }
  }
}

function renderAll() {
  renderObjectMarkers();
  renderObjectList();
  updateSelectedInfo();
}

function afterObjectsChange() {
  if (selectedIdx >= objects.length) selectedIdx = objects.length - 1;
  renderAll();
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------
function selectObject(idx) {
  if (idx === selectedIdx) return;
  selectedIdx = idx;
  renderAll();

  // Pan to selected
  if (idx >= 0 && idx < objects.length) {
    const [px, py] = objects[idx].position;
    map.panTo(pixelToLatLng(px, py));
  }
}

function deselectAll() {
  selectedIdx = -1;
  renderAll();
}

function updateSelectedInfo() {
  const sec = $("selObjectSection");
  if (selectedIdx < 0 || selectedIdx >= objects.length) {
    sec.hidden = true;
    return;
  }
  sec.hidden = false;
  const obj = objects[selectedIdx];
  $("selObjName").textContent = obj.name;
  $("selObjType").value = obj.type || "pit";

  const [px, py] = obj.position;
  $("selObjPixel").textContent = `像素：(${Math.round(px)}, ${Math.round(py)})`;

  const [lat, lng] = pixelToLatLng(px, py);
  $("selObjGeo").textContent = `经纬：(${lat.toFixed(6)}, ${lng.toFixed(6)})`;

  const orient = obj.orientation_z || [0, -1];
  const deg = bearing(orient[0], orient[1]);
  $("selObjBearing").textContent = `方向：${deg.toFixed(1)}°  [${orient[0].toFixed(3)}, ${orient[1].toFixed(3)}]`;
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------
function deleteObject(idx) {
  if (idx < 0 || idx >= objects.length) return;
  pushUndo();
  objects.splice(idx, 1);
  if (selectedIdx === idx) {
    selectedIdx = -1;
  } else if (selectedIdx > idx) {
    selectedIdx--;
  }
  renderAll();
  setStatus("已删除对象");
}

// ---------------------------------------------------------------------------
// Create mode
// ---------------------------------------------------------------------------
function enterCreateMode() {
  mode = "create";
  deselectAll();
  $("map").classList.add("map--create");
  map.doubleClickZoom.disable();
  clearRubberArrow();
  setStatus("创建模式：点击放置对象位置，再次点击确认方向，Esc 取消");
}

function exitCreateMode() {
  mode = "select";
  createPendingPos = null;
  $("map").classList.remove("map--create");
  map.doubleClickZoom.enable();
  clearRubberArrow();
  document.querySelector('input[name="mode"][value="select"]').checked = true;
  setStatus("选择模式");
}

function clearRubberArrow() {
  if (rubberArrow) {
    rubberArrow.remove();
    rubberArrow = null;
  }
}

function handleCreateClick(e) {
  const [px, py] = latLngToPixel(e.latlng.lat, e.latlng.lng);

  if (mode === "create") {
    // First click: set position
    createPendingPos = [Math.round(px), Math.round(py)];
    mode = "create_direction";
    setStatus("拖动鼠标设置方向，点击确认");
    return;
  }

  if (mode === "create_direction" && createPendingPos) {
    // Second click: set direction
    const dx = px - createPendingPos[0];
    const dy = py - createPendingPos[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    const orient = len > 1 ? [dx / len, dy / len] : [0, -1];

    pushUndo();
    const name = generateName(createType);
    objects.push({
      name,
      position: createPendingPos,
      orientation_z: [
        Math.round(orient[0] * 10000) / 10000,
        Math.round(orient[1] * 10000) / 10000,
      ],
      type: createType,
    });

    selectedIdx = objects.length - 1;
    createPendingPos = null;
    mode = "create";
    clearRubberArrow();
    renderAll();
    setStatus(`已创建 ${name} — 继续点击放置下一个，Esc 退出`);
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
      className: "oe-rubber-band",
    }).addTo(map);
  }
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
    // Count existing timing pairs
    const existingNums = objects
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

  // Count existing of this type
  const count = objects.filter((o) => o.type === type).length;
  return `${prefix}_${count}`;
}

// ---------------------------------------------------------------------------
// Centerline overlay
// ---------------------------------------------------------------------------
async function loadCenterline() {
  try {
    const resp = await fetch("/api/centerline");
    if (!resp.ok) return;
    const data = await resp.json();
    const pts = data.centerline;
    if (!pts || pts.length < 2) return;

    const latlngs = pts.map(([x, y]) => pixelToLatLng(x, y));
    centerlineLayer = L.polyline(latlngs, {
      color: "#ffffff",
      weight: 1.5,
      dashArray: "8 6",
      opacity: 0.5,
      interactive: false,
    }).addTo(map);
  } catch {
    // Centerline not available — that's fine
  }
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
async function save() {
  setStatus("正在保存…");
  try {
    const data = {
      track_direction: trackDirection,
      objects: objects,
    };
    const resp = await fetch("/api/game_objects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text}`);
    }
    markClean();
    setStatus("已保存 game_objects.json");
  } catch (err) {
    setStatus(`保存失败：${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  map = L.map("map", {
    zoomControl: true,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
  });
  setupRightDrag(map, $("map"));

  tileLayer = L.tileLayer(TILE_URL, {
    minZoom: 12,
    maxZoom: 24,
    maxNativeZoom: 24,
    attribution: "Local tiles",
  }).addTo(map);

  L.control.scale({ imperial: false }).addTo(map);
  map.setView([22.7123312, 113.8654811], 19);

  // Map click handler
  map.on("click", (e) => {
    if (mode === "create" || mode === "create_direction") {
      handleCreateClick(e);
      return;
    }
    deselectAll();
  });

  map.on("mousemove", (e) => {
    handleCreateMouseMove(e);
  });

  // Load data
  try {
    // Try stage 8 geo metadata first, fall back to stage 7
    const metaResp = await fetch("/api/game_objects/geo_metadata");
    if (!metaResp.ok) throw new Error(`geo_metadata: ${metaResp.status}`);
    geoMeta = await metaResp.json();

    const objResp = await fetch("/api/game_objects");
    if (!objResp.ok) throw new Error(`game_objects: ${objResp.status}`);
    const objData = await objResp.json();

    trackDirection = objData.track_direction || "clockwise";
    objects = (objData.objects || []).map((o) => ({
      name: o.name || "",
      position: (o.position || [0, 0]).map(Number),
      orientation_z: (o.orientation_z || [0, -1]).map(Number),
      type: o.type || "unknown",
    }));

    // Fit bounds
    if (geoMeta.bounds) {
      const { north, south, east, west } = geoMeta.bounds;
      map.fitBounds([
        [south, west],
        [north, east],
      ], { padding: [10, 10] });
    }

    renderAll();
    setStatus(`已加载 ${objects.length} 个游戏对象`);
  } catch (err) {
    setStatus(`加载失败：${err.message}`);
  }

  // Load centerline overlay
  loadCenterline();

  // Wire up UI
  wireUI();
}

function wireUI() {
  // Mode radios
  document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      if (e.target.value === "create") enterCreateMode();
      else exitCreateMode();
    });
  });

  // Create type
  $("createType").addEventListener("change", (e) => {
    createType = e.target.value;
    if (rubberArrow) rubberArrow.setStyle({ color: objColor(createType) });
  });

  // Toolbar buttons
  $("btnSave").addEventListener("click", () => save());
  $("btnUndo").addEventListener("click", () => undo());
  $("btnRedo").addEventListener("click", () => redo());

  // Selected object controls
  $("selObjType").addEventListener("change", (e) => {
    if (selectedIdx < 0) return;
    pushUndo();
    objects[selectedIdx].type = e.target.value;
    renderAll();
  });

  $("btnDeleteObj").addEventListener("click", () => {
    if (selectedIdx >= 0) deleteObject(selectedIdx);
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

    if (
      (e.key === "z" && (e.ctrlKey || e.metaKey) && e.shiftKey) ||
      (e.key === "y" && (e.ctrlKey || e.metaKey))
    ) {
      e.preventDefault();
      redo();
      return;
    }

    if (e.key === "Delete" || e.key === "Backspace") {
      if (mode !== "select") return;
      if (selectedIdx >= 0) deleteObject(selectedIdx);
      return;
    }

    if (e.key === "Escape") {
      if (mode === "create" || mode === "create_direction") {
        exitCreateMode();
      } else {
        deselectAll();
      }
      return;
    }
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
  try {
    setStatus(`错误：${ev?.message || ev}`);
  } catch {}
});
window.addEventListener("unhandledrejection", (ev) => {
  try {
    setStatus(`Promise 错误：${ev?.reason?.message || ev?.reason || ev}`);
  } catch {}
});

// Start
init();
