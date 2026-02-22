/* global L */
/* Wall Editor — interactive walls.json editor on Leaflet map */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const TILE_URL = "/tiles/{z}/{x}/{y}.png";
const WALL_COLORS = {
  outer: "#00ff00",
  inner: "#ff6600",
  tree: "#228b22",
  building: "#9333ea",
  water: "#00bfff",
};
const DEFAULT_COLOR = "#888888";
const UNDO_LIMIT = 50;

// Wall type display order
const WALL_TYPE_ORDER = ["outer", "inner", "tree", "building", "water"];

const $ = (id) => document.getElementById(id);

function setStatus(msg) {
  $("status").textContent = msg;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let geoMeta = null; // { image_width, image_height, bounds: {north,south,east,west} }
let walls = []; // array of { type, points:[[x,y],...], closed }
let selectedWallIdx = -1;
let selectedVertexIdx = -1;
let mode = "select"; // "select" | "draw"
let drawPoints = [];
let drawType = "tree";
let undoStack = [];
let redoStack = [];
let dirty = false;

// Visibility state per category + basemap
let visibility = {
  basemap: true,
  outer: true,
  inner: true,
  tree: true,
  building: true,
  water: true,
};

// Leaflet objects
let map;
let tileLayer;
let wallLayers = []; // L.polygon/polyline per wall
let vertexGroup; // L.layerGroup of vertex markers for selected wall
let midpointGroup; // L.layerGroup of midpoint ghost markers
let drawLayer = null; // L.polyline preview during draw mode
let drawMarkers = null; // L.layerGroup for draw points
let rubberBand = null; // L.polyline rubber-band
let rightDrag = null; // setupRightDrag() return value

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
// Context menu state
// ---------------------------------------------------------------------------
let ctxPending = null; // { wallIdx, insertIdx, px, py, vertexIdx }

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
function closestPointOnSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return { x: ax, y: ay, t: 0 };
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return { x: ax + t * dx, y: ay + t * dy, t };
}

function findNearestEdge(wallIdx, clickPx, clickPy) {
  const w = walls[wallIdx];
  const pts = w.points;
  const len = pts.length;
  const isClosed = w.closed !== false;
  const edgeCount = isClosed ? len : len - 1;

  let bestDist = Infinity;
  let bestEdge = -1;
  let bestPt = null;

  for (let i = 0; i < edgeCount; i++) {
    const j = (i + 1) % len;
    const proj = closestPointOnSegment(
      clickPx, clickPy,
      pts[i][0], pts[i][1],
      pts[j][0], pts[j][1]
    );
    const dx = proj.x - clickPx;
    const dy = proj.y - clickPy;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      bestEdge = i;
      bestPt = proj;
    }
  }

  return { edgeIdx: bestEdge, projPx: bestPt.x, projPy: bestPt.y, dist: Math.sqrt(bestDist) };
}

function findNearestVertex(wallIdx, clickPx, clickPy) {
  const pts = walls[wallIdx].points;
  let bestDist = Infinity;
  let bestIdx = -1;
  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i][0] - clickPx;
    const dy = pts[i][1] - clickPy;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  return { vertexIdx: bestIdx, dist: Math.sqrt(bestDist) };
}

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------
function showContextMenu(screenX, screenY, wallIdx, insertIdx, px, py, vertexIdx) {
  ctxPending = { wallIdx, insertIdx, px, py, vertexIdx: vertexIdx ?? -1 };
  const menu = $("ctxMenu");
  menu.hidden = false;
  menu.style.left = screenX + "px";
  menu.style.top = screenY + "px";

  // Show/hide delete vertex option
  const delItem = $("ctxDeleteVertex");
  if (vertexIdx >= 0) {
    delItem.classList.remove("we-ctx-menu__item--hidden");
  } else {
    delItem.classList.add("we-ctx-menu__item--hidden");
  }

  // Keep menu within viewport
  requestAnimationFrame(() => {
    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) menu.style.left = (screenX - rect.width) + "px";
    if (rect.bottom > window.innerHeight) menu.style.top = (screenY - rect.height) + "px";
  });
}

function hideContextMenu() {
  $("ctxMenu").hidden = true;
  ctxPending = null;
}

function handleCtxAddVertex() {
  if (!ctxPending) return;
  const { wallIdx, insertIdx, px, py } = ctxPending;
  hideContextMenu();

  if (wallIdx < 0 || wallIdx >= walls.length) return;

  // Select wall if not already
  if (selectedWallIdx !== wallIdx) {
    selectedWallIdx = wallIdx;
    selectedVertexIdx = -1;
  }

  pushUndo();
  walls[wallIdx].points.splice(insertIdx, 0, [px, py]);
  selectVertex(insertIdx);
  renderAll();
  setStatus(`已在边上插入顶点 (${px}, ${py})`);
}

function handleCtxDeleteVertex() {
  if (!ctxPending) return;
  const { wallIdx, vertexIdx } = ctxPending;
  hideContextMenu();

  if (wallIdx < 0 || wallIdx >= walls.length || vertexIdx < 0) return;

  // Select wall if not already
  if (selectedWallIdx !== wallIdx) {
    selectedWallIdx = wallIdx;
  }

  deleteVertex(wallIdx, vertexIdx);
  setStatus(`已删除顶点 #${vertexIdx}`);
}

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------
function pushUndo() {
  undoStack.push(JSON.parse(JSON.stringify(walls)));
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack.length = 0;
  markDirty();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.parse(JSON.stringify(walls)));
  walls = undoStack.pop();
  afterWallsChange();
}

function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.parse(JSON.stringify(walls)));
  walls = redoStack.pop();
  afterWallsChange();
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
// Visibility
// ---------------------------------------------------------------------------
function buildVisibilityChips() {
  const container = $("visibilityChips");
  container.innerHTML = "";

  // Basemap chip
  const basemapChip = document.createElement("div");
  basemapChip.className = `le-chip ${visibility.basemap ? "le-chip--on" : ""}`;
  basemapChip.style.setProperty("--chip-color", "rgba(148,163,184,0.5)");
  basemapChip.style.setProperty("--chip-bg", "rgba(148,163,184,0.15)");
  basemapChip.style.setProperty("--chip-dot", "#94a3b8");
  basemapChip.innerHTML = '<span class="le-chip__dot"></span> 底图';
  basemapChip.addEventListener("click", () => {
    visibility.basemap = !visibility.basemap;
    basemapChip.classList.toggle("le-chip--on", visibility.basemap);
    if (tileLayer) {
      if (visibility.basemap) tileLayer.addTo(map);
      else tileLayer.remove();
    }
  });
  container.appendChild(basemapChip);

  // Per-type chips
  for (const type of WALL_TYPE_ORDER) {
    const color = WALL_COLORS[type] || DEFAULT_COLOR;
    const chip = document.createElement("div");
    chip.className = `le-chip ${visibility[type] ? "le-chip--on" : ""}`;
    chip.style.setProperty("--chip-color", color);
    chip.style.setProperty("--chip-bg", hexToRgba(color, 0.2));
    chip.style.setProperty("--chip-dot", color);
    chip.innerHTML = `<span class="le-chip__dot"></span> ${type}`;
    chip.addEventListener("click", () => {
      visibility[type] = !visibility[type];
      chip.classList.toggle("le-chip--on", visibility[type]);
      renderWallPolygons();
      renderVertexMarkers();
    });
    container.appendChild(chip);
  }
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function wallColor(type) {
  return WALL_COLORS[type] || DEFAULT_COLOR;
}

function brightenColor(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const f = (c) => Math.min(255, Math.floor(c + (255 - c) * 0.4));
  return `#${f(r).toString(16).padStart(2, "0")}${f(g).toString(16).padStart(2, "0")}${f(b).toString(16).padStart(2, "0")}`;
}

function renderWallPolygons() {
  // Remove old layers
  wallLayers.forEach((l) => { if (l) l.remove(); });
  wallLayers = [];

  walls.forEach((w, idx) => {
    // Respect visibility
    if (!visibility[w.type]) {
      wallLayers.push(null);
      return;
    }

    const latlngs = w.points.map(([px, py]) => pixelToLatLng(px, py));
    const isSelected = idx === selectedWallIdx;
    const color = wallColor(w.type);

    const opts = {
      color: isSelected ? brightenColor(color) : color,
      weight: isSelected ? 3 : 2,
      opacity: isSelected ? 1 : 0.7,
      fillOpacity: isSelected ? 0.15 : 0.05,
      dashArray: isSelected ? "8 4" : null,
      interactive: true,
    };

    const layer =
      w.closed !== false ? L.polygon(latlngs, opts) : L.polyline(latlngs, opts);

    layer.on("click", (e) => {
      if (mode === "draw") return; // Let click propagate to map for drawing
      L.DomEvent.stopPropagation(e);
      selectWall(idx);
    });

    layer.on("contextmenu", (e) => {
      L.DomEvent.stopPropagation(e);
      L.DomEvent.preventDefault(e);
      if (mode === "draw") return;
      if (rightDrag && rightDrag.wasDragging()) return;

      // Find nearest edge and nearest vertex
      const [clickPx, clickPy] = latLngToPixel(e.latlng.lat, e.latlng.lng);
      const hit = findNearestEdge(idx, clickPx, clickPy);
      if (hit.edgeIdx < 0) return;

      // Check if close to an existing vertex (within ~15px at current zoom)
      const vHit = findNearestVertex(idx, clickPx, clickPy);
      const vertexThreshold = 15;
      const nearVertex = vHit.dist < vertexThreshold ? vHit.vertexIdx : -1;

      const isClosed = w.closed !== false;
      const insertIdx = isClosed
        ? (hit.edgeIdx + 1) % w.points.length
        : hit.edgeIdx + 1;

      showContextMenu(
        e.originalEvent.clientX,
        e.originalEvent.clientY,
        idx,
        insertIdx,
        hit.projPx,
        hit.projPy,
        nearVertex,
      );
    });

    layer.addTo(map);
    wallLayers.push(layer);
  });
}

function renderVertexMarkers() {
  if (vertexGroup) vertexGroup.clearLayers();
  else vertexGroup = L.layerGroup().addTo(map);

  if (midpointGroup) midpointGroup.clearLayers();
  else midpointGroup = L.layerGroup().addTo(map);

  if (selectedWallIdx < 0 || selectedWallIdx >= walls.length) return;

  const w = walls[selectedWallIdx];

  // Don't show vertex markers if this type is hidden
  if (!visibility[w.type]) return;

  const pts = w.points;

  // Vertex markers
  pts.forEach(([px, py], vi) => {
    const ll = pixelToLatLng(px, py);
    const isSelected = vi === selectedVertexIdx;
    const icon = L.divIcon({
      className: "",
      html: `<div class="we-vertex ${isSelected ? "we-vertex--selected" : ""}"></div>`,
      iconSize: [12, 12],
      iconAnchor: [6, 6],
    });

    const marker = L.marker(ll, { icon, draggable: true, zIndexOffset: isSelected ? 1000 : 0 });

    marker.on("click", (e) => {
      L.DomEvent.stopPropagation(e);
      selectVertex(vi);
    });

    marker.on("contextmenu", (e) => {
      L.DomEvent.stopPropagation(e);
      L.DomEvent.preventDefault(e);
      if (mode === "draw") return;
      if (rightDrag && rightDrag.wasDragging()) return;

      // Right-click on vertex: show context menu with delete option
      const [clickPx, clickPy] = latLngToPixel(e.latlng.lat, e.latlng.lng);
      const hit = findNearestEdge(selectedWallIdx, clickPx, clickPy);
      const isClosed = w.closed !== false;
      const insertIdx = hit.edgeIdx >= 0
        ? (isClosed ? (hit.edgeIdx + 1) % w.points.length : hit.edgeIdx + 1)
        : vi;

      showContextMenu(
        e.originalEvent.clientX,
        e.originalEvent.clientY,
        selectedWallIdx,
        insertIdx,
        clickPx,
        clickPy,
        vi,
      );
    });

    marker.on("dragstart", () => {
      pushUndo();
    });

    marker.on("drag", (e) => {
      const latlng = e.target.getLatLng();
      const [npx, npy] = latLngToPixel(latlng.lat, latlng.lng);
      w.points[vi] = [npx, npy];
      updateWallLayer(selectedWallIdx);
    });

    marker.on("dragend", (e) => {
      const latlng = e.target.getLatLng();
      const [npx, npy] = latLngToPixel(latlng.lat, latlng.lng);
      w.points[vi] = [npx, npy];
      updateWallLayer(selectedWallIdx);
      updateSelectedVertexInfo();
    });

    marker.addTo(vertexGroup);
  });

  // Midpoint ghost markers
  const len = pts.length;
  const isClosed = w.closed !== false;
  const edgeCount = isClosed ? len : len - 1;

  for (let i = 0; i < edgeCount; i++) {
    const j = (i + 1) % len;
    const mx = (pts[i][0] + pts[j][0]) / 2;
    const my = (pts[i][1] + pts[j][1]) / 2;
    const ll = pixelToLatLng(mx, my);

    const icon = L.divIcon({
      className: "",
      html: '<div class="we-vertex--mid"></div>',
      iconSize: [8, 8],
      iconAnchor: [4, 4],
    });

    const ghost = L.marker(ll, { icon, interactive: true });
    const insertIdx = j;

    ghost.on("click", (e) => {
      L.DomEvent.stopPropagation(e);
      pushUndo();
      w.points.splice(insertIdx, 0, [mx, my]);
      selectVertex(insertIdx);
      renderAll();
    });

    ghost.addTo(midpointGroup);
  }
}

function updateWallLayer(idx) {
  if (idx < 0 || idx >= walls.length || idx >= wallLayers.length) return;
  const layer = wallLayers[idx];
  if (!layer) return;
  const w = walls[idx];
  const latlngs = w.points.map(([px, py]) => pixelToLatLng(px, py));
  layer.setLatLngs(latlngs);
}

function renderWallList() {
  const el = $("wallList");
  el.innerHTML = "";

  // Group walls by type
  const groups = {};
  walls.forEach((w, idx) => {
    const type = w.type || "outer";
    if (!groups[type]) groups[type] = [];
    groups[type].push({ wall: w, idx });
  });

  // Render in order
  for (const type of WALL_TYPE_ORDER) {
    const items = groups[type];
    if (!items || items.length === 0) continue;

    // Group header
    const header = document.createElement("div");
    header.className = "we-wall-group-header";
    header.innerHTML = `${type} <span class="we-wall-group-count">(${items.length})</span>`;
    el.appendChild(header);

    // Items
    for (const { wall: w, idx } of items) {
      const div = document.createElement("div");
      div.className = `we-wall-item ${idx === selectedWallIdx ? "we-wall-item--selected" : ""}`;
      div.innerHTML = `
        <div class="we-wall-swatch" style="background:${wallColor(w.type)}"></div>
        <div class="we-wall-label">${w.type} #${idx}</div>
        <div class="we-wall-pts">${w.points.length} pts</div>
        <button class="we-wall-del" data-idx="${idx}" title="删除">&times;</button>
      `;

      div.addEventListener("click", (e) => {
        if (e.target.closest(".we-wall-del")) return;
        selectWall(idx);
      });

      div.querySelector(".we-wall-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteWall(idx);
      });

      el.appendChild(div);
    }
  }

  // Any types not in the order list
  for (const type of Object.keys(groups)) {
    if (WALL_TYPE_ORDER.includes(type)) continue;
    const items = groups[type];
    const header = document.createElement("div");
    header.className = "we-wall-group-header";
    header.innerHTML = `${type} <span class="we-wall-group-count">(${items.length})</span>`;
    el.appendChild(header);

    for (const { wall: w, idx } of items) {
      const div = document.createElement("div");
      div.className = `we-wall-item ${idx === selectedWallIdx ? "we-wall-item--selected" : ""}`;
      div.innerHTML = `
        <div class="we-wall-swatch" style="background:${wallColor(w.type)}"></div>
        <div class="we-wall-label">${w.type} #${idx}</div>
        <div class="we-wall-pts">${w.points.length} pts</div>
        <button class="we-wall-del" data-idx="${idx}" title="删除">&times;</button>
      `;
      div.addEventListener("click", (e) => {
        if (e.target.closest(".we-wall-del")) return;
        selectWall(idx);
      });
      div.querySelector(".we-wall-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteWall(idx);
      });
      el.appendChild(div);
    }
  }
}

function renderAll() {
  renderWallPolygons();
  renderVertexMarkers();
  renderWallList();
  updateSelectedWallInfo();
  updateSelectedVertexInfo();
}

function afterWallsChange() {
  // Clamp selection
  if (selectedWallIdx >= walls.length) selectedWallIdx = walls.length - 1;
  if (selectedWallIdx >= 0) {
    const pts = walls[selectedWallIdx].points;
    if (selectedVertexIdx >= pts.length) selectedVertexIdx = pts.length - 1;
  } else {
    selectedVertexIdx = -1;
  }
  renderAll();
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------
function selectWall(idx) {
  if (idx === selectedWallIdx) return;
  selectedWallIdx = idx;
  selectedVertexIdx = -1;
  renderAll();
}

function selectVertex(vi) {
  selectedVertexIdx = vi;
  renderVertexMarkers();
  updateSelectedVertexInfo();
}

function deselectAll() {
  selectedWallIdx = -1;
  selectedVertexIdx = -1;
  renderAll();
}

function updateSelectedWallInfo() {
  const sec = $("selWallSection");
  if (selectedWallIdx < 0 || selectedWallIdx >= walls.length) {
    sec.hidden = true;
    return;
  }
  sec.hidden = false;
  const w = walls[selectedWallIdx];
  $("selWallType").value = w.type;
  const closedChip = $("selWallClosedChip");
  closedChip.classList.toggle("le-chip--on", w.closed !== false);
  $("selWallPtsCount").textContent = `点数：${w.points.length}`;
}

function updateSelectedVertexInfo() {
  const sec = $("selVertexSection");
  if (
    selectedWallIdx < 0 ||
    selectedVertexIdx < 0 ||
    selectedWallIdx >= walls.length ||
    selectedVertexIdx >= walls[selectedWallIdx].points.length
  ) {
    sec.hidden = true;
    return;
  }
  sec.hidden = false;
  const [px, py] = walls[selectedWallIdx].points[selectedVertexIdx];
  const [lat, lng] = pixelToLatLng(px, py);
  $("selVertexPixel").textContent = `像素：(${px}, ${py})`;
  $("selVertexGeo").textContent = `经纬：(${lat.toFixed(6)}, ${lng.toFixed(6)})`;
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------
function deleteVertex(wallIdx, vertexIdx) {
  if (wallIdx < 0 || wallIdx >= walls.length) return;
  const w = walls[wallIdx];
  if (w.points.length <= 3) {
    deleteWall(wallIdx);
    return;
  }
  pushUndo();
  w.points.splice(vertexIdx, 1);
  if (selectedVertexIdx >= w.points.length) selectedVertexIdx = w.points.length - 1;
  renderAll();
}

function deleteWall(idx) {
  if (idx < 0 || idx >= walls.length) return;
  pushUndo();
  walls.splice(idx, 1);
  if (selectedWallIdx === idx) {
    selectedWallIdx = -1;
    selectedVertexIdx = -1;
  } else if (selectedWallIdx > idx) {
    selectedWallIdx--;
  }
  renderAll();
}

// ---------------------------------------------------------------------------
// Draw mode
// ---------------------------------------------------------------------------
function enterDrawMode() {
  mode = "draw";
  drawPoints = [];
  deselectAll();
  $("map").classList.add("map--draw");
  map.doubleClickZoom.disable();
  clearDrawPreview();
  updateModeSegUI("draw");
  setStatus("绘制模式：点击添加顶点，双击或 Enter 完成，Esc 取消");
}

function exitDrawMode() {
  mode = "select";
  drawPoints = [];
  $("map").classList.remove("map--draw");
  map.doubleClickZoom.enable();
  clearDrawPreview();
  updateModeSegUI("select");
  setStatus("选择模式");
}

function updateModeSegUI(activeMode) {
  document.querySelectorAll("#modeSeg .le-seg__btn").forEach((btn) => {
    btn.classList.toggle("le-seg__btn--active", btn.dataset.mode === activeMode);
  });
}

function clearDrawPreview() {
  if (drawLayer) {
    drawLayer.remove();
    drawLayer = null;
  }
  if (drawMarkers) {
    drawMarkers.clearLayers();
    drawMarkers.remove();
    drawMarkers = null;
  }
  if (rubberBand) {
    rubberBand.remove();
    rubberBand = null;
  }
}

function updateDrawPreview() {
  if (!drawMarkers) drawMarkers = L.layerGroup().addTo(map);

  const latlngs = drawPoints.map(([px, py]) => pixelToLatLng(px, py));

  if (drawLayer) drawLayer.setLatLngs(latlngs);
  else {
    drawLayer = L.polyline(latlngs, {
      color: wallColor(drawType),
      weight: 2,
      dashArray: "6 3",
      interactive: false,
    }).addTo(map);
  }

  // Point markers
  drawMarkers.clearLayers();
  drawPoints.forEach(([px, py]) => {
    const ll = pixelToLatLng(px, py);
    const icon = L.divIcon({
      className: "",
      html: '<div class="we-vertex"></div>',
      iconSize: [12, 12],
      iconAnchor: [6, 6],
    });
    L.marker(ll, { icon, interactive: false }).addTo(drawMarkers);
  });
}

function finishDraw() {
  if (drawPoints.length < 3) {
    setStatus("至少需要 3 个点才能创建围墙");
    return;
  }
  pushUndo();
  walls.push({
    type: drawType,
    points: drawPoints.slice(),
    closed: true,
  });
  selectedWallIdx = walls.length - 1;
  selectedVertexIdx = -1;
  exitDrawMode();
  renderAll();
  setStatus(`已添加 ${drawType} 围墙（${drawPoints.length} 个点）`);
}

function handleDrawClick(e) {
  const [px, py] = latLngToPixel(e.latlng.lat, e.latlng.lng);
  drawPoints.push([px, py]);
  updateDrawPreview();
}

function handleDrawDblClick(e) {
  L.DomEvent.stopPropagation(e);
  L.DomEvent.preventDefault(e);
  // Remove the last point added by the preceding click event
  if (drawPoints.length > 3) drawPoints.pop();
  finishDraw();
}

function handleDrawMouseMove(e) {
  if (!drawPoints.length) return;
  const lastLL = pixelToLatLng(...drawPoints[drawPoints.length - 1]);
  const curLL = [e.latlng.lat, e.latlng.lng];

  if (rubberBand) rubberBand.setLatLngs([lastLL, curLL]);
  else {
    rubberBand = L.polyline([lastLL, curLL], {
      color: wallColor(drawType),
      weight: 1,
      dashArray: "4 4",
      opacity: 0.6,
      interactive: false,
    }).addTo(map);
  }
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
async function save() {
  setStatus("正在保存…");
  try {
    const resp = await fetch("/api/walls", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ walls }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text}`);
    }
    markClean();
    setStatus("已保存 walls.json");
  } catch (err) {
    setStatus(`保存失败：${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  // Create map
  map = L.map("map", {
    zoomControl: true,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
  });

  rightDrag = setupRightDrag(map, $("map"));

  tileLayer = L.tileLayer(TILE_URL, {
    minZoom: 12,
    maxZoom: 24,
    maxNativeZoom: 24,
    attribution: "Local tiles",
  }).addTo(map);

  L.control.scale({ imperial: false }).addTo(map);
  map.setView([22.7123312, 113.8654811], 19);

  // Map click handler — in draw mode, clicks on existing walls should
  // NOT be intercepted (wall layers have interactive:true but we return
  // early in their click handler when mode === "draw")
  map.on("click", (e) => {
    if (mode === "draw") {
      handleDrawClick(e);
      return;
    }
    // Deselect when clicking empty space
    deselectAll();
  });

  map.on("dblclick", (e) => {
    if (mode === "draw") {
      handleDrawDblClick(e);
    }
  });

  map.on("mousemove", (e) => {
    if (mode === "draw") {
      handleDrawMouseMove(e);
    }
  });

  // Load data
  try {
    const [metaResp, wallsResp] = await Promise.all([
      fetch("/api/geo_metadata"),
      fetch("/api/walls"),
    ]);

    if (!metaResp.ok) throw new Error(`geo_metadata: ${metaResp.status}`);
    if (!wallsResp.ok) throw new Error(`walls: ${wallsResp.status}`);

    geoMeta = await metaResp.json();
    const wallsData = await wallsResp.json();

    walls = (wallsData.walls || []).map((w) => ({
      type: w.type || "outer",
      points: (w.points || []).map(([x, y]) => [x, y]),
      closed: w.closed !== false,
    }));

    // Fit bounds to image extent
    if (geoMeta.bounds) {
      const { north, south, east, west } = geoMeta.bounds;
      map.fitBounds([
        [south, west],
        [north, east],
      ], { padding: [10, 10] });
    }

    buildVisibilityChips();
    renderAll();
    setStatus(`已加载 ${walls.length} 个围墙`);
  } catch (err) {
    setStatus(`加载失败：${err.message}`);
  }

  // Wire up UI
  wireUI();
}

function wireUI() {
  // Mode segmented control
  document.querySelectorAll("#modeSeg .le-seg__btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.mode === "draw") enterDrawMode();
      else exitDrawMode();
    });
  });

  // Draw type
  $("drawType").addEventListener("change", (e) => {
    drawType = e.target.value;
    if (drawLayer) drawLayer.setStyle({ color: wallColor(drawType) });
    if (rubberBand) rubberBand.setStyle({ color: wallColor(drawType) });
  });

  // Toolbar buttons
  $("btnSave").addEventListener("click", () => save());
  $("btnUndo").addEventListener("click", () => undo());
  $("btnRedo").addEventListener("click", () => redo());

  // Selected wall controls
  $("selWallType").addEventListener("change", (e) => {
    if (selectedWallIdx < 0) return;
    pushUndo();
    walls[selectedWallIdx].type = e.target.value;
    renderAll();
  });

  // Closed toggle chip (replaces checkbox)
  $("selWallClosedChip").addEventListener("click", () => {
    if (selectedWallIdx < 0) return;
    pushUndo();
    const w = walls[selectedWallIdx];
    w.closed = !(w.closed !== false);
    $("selWallClosedChip").classList.toggle("le-chip--on", w.closed);
    renderAll();
  });

  $("btnDeleteWall").addEventListener("click", () => {
    if (selectedWallIdx >= 0) deleteWall(selectedWallIdx);
  });

  $("btnDeleteVertex").addEventListener("click", () => {
    if (selectedWallIdx >= 0 && selectedVertexIdx >= 0) {
      deleteVertex(selectedWallIdx, selectedVertexIdx);
    }
  });

  // Context menu
  $("ctxAddVertex").addEventListener("click", () => handleCtxAddVertex());
  $("ctxDeleteVertex").addEventListener("click", () => handleCtxDeleteVertex());
  document.addEventListener("click", () => hideContextMenu());
  map.on("movestart", () => hideContextMenu());

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Don't capture when typing in inputs
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
      if (mode === "draw") return;
      if (selectedVertexIdx >= 0 && selectedWallIdx >= 0) {
        deleteVertex(selectedWallIdx, selectedVertexIdx);
      } else if (selectedWallIdx >= 0) {
        deleteWall(selectedWallIdx);
      }
      return;
    }

    if (e.key === "Escape") {
      hideContextMenu();
      if (mode === "draw") {
        exitDrawMode();
      } else {
        deselectAll();
      }
      return;
    }

    if (e.key === "Enter" && mode === "draw") {
      finishDraw();
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
