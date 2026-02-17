/* global L */
/* Layout Editor — paint per-layout binary masks on a Leaflet map */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const UNDO_LIMIT = 20;
const DEFAULT_COLORS = ["#ff4444", "#4488ff", "#44ff44", "#ff8800", "#cc44ff", "#00cccc"];
const REF_COLORS = {
  merged:   [255, 255, 255],
  concrete: [100, 150, 255],
  grass:    [80, 200, 80],
  trees:    [30, 160, 30],
  kerb:     [255, 180, 50],
  building: [180, 130, 90],
};

const $ = (id) => document.getElementById(id);
function setStatus(msg) { $("status").textContent = msg; }

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let geoMeta = null;       // { image_width, image_height, bounds }
let maskW = 0, maskH = 0; // modelscale image dimensions (= mask dimensions)
let layouts = [];          // [{ name, color, track_direction, mask_file, _maskCanvas, _dirty }]
let selectedIdx = -1;
let tool = "brush";        // "brush" | "eraser"
let brushSize = 20;        // pixels in mask space
let dirty = false;
let painting = false;      // left-mouse is down and painting
let lastPaintPt = null;    // last paint pixel [x, y] for line interpolation

// Right-click drag state
let rightDragging = false;
let rightDragStart = null;

// Undo/redo per layout: map name -> [ImageData]
let undoStacks = {};
let redoStacks = {};

// Leaflet
let map;
let baseImageOverlay = null;
let maskOverlay = null;      // L.imageOverlay for the current layout mask
let refOverlays = {};        // tag -> L.imageOverlay for reference masks
let imageBounds = null;

// Base map visibility
let baseMapVisible = true;

// Display canvas (for tinting mask before sending to overlay)
let displayCanvas = null;
let displayCtx = null;

// ---------------------------------------------------------------------------
// Coordinate conversions (Simple CRS: pixel coords)
// ---------------------------------------------------------------------------
// In Simple CRS:  latlng = [maskH - py, px]  so  (0,0)pixel = top-left
function pixelToLatLng(px, py) {
  return [maskH - py, px];
}
function latLngToPixel(lat, lng) {
  return [lng, maskH - lat];
}

// ---------------------------------------------------------------------------
// Mask Canvas helpers
// ---------------------------------------------------------------------------
function createMaskCanvas() {
  const c = document.createElement("canvas");
  c.width = maskW;
  c.height = maskH;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, maskW, maskH);
  return c;
}

function canvasToDataURL(canvas) {
  return canvas.toDataURL("image/png");
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

function loadImageToCanvas(canvas, imgSrc) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve();
    };
    img.onerror = reject;
    img.src = imgSrc;
  });
}

function countWhitePixels(canvas) {
  const ctx = canvas.getContext("2d");
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 127) count++;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Mask → display: pixel-level tint (white→color, black→transparent)
// ---------------------------------------------------------------------------
function ensureDisplayCanvas() {
  if (!displayCanvas) {
    displayCanvas = document.createElement("canvas");
    displayCanvas.width = maskW;
    displayCanvas.height = maskH;
    displayCtx = displayCanvas.getContext("2d");
  }
}

function parseHexColor(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b];
}

function maskToDisplay(maskCanvas, colorHex) {
  ensureDisplayCanvas();
  const [cr, cg, cb] = parseHexColor(colorHex || "#ff4444");

  const maskCtx = maskCanvas.getContext("2d");
  const src = maskCtx.getImageData(0, 0, maskW, maskH);
  const dst = displayCtx.createImageData(maskW, maskH);
  const sd = src.data;
  const dd = dst.data;

  for (let i = 0; i < sd.length; i += 4) {
    if (sd[i] > 127) {
      dd[i]     = cr;
      dd[i + 1] = cg;
      dd[i + 2] = cb;
      dd[i + 3] = 160;
    }
    // else: leave dd as 0,0,0,0 (transparent)
  }

  displayCtx.putImageData(dst, 0, 0);
  return canvasToDataURL(displayCanvas);
}

/**
 * Build a tinted overlay from a mask image URL + RGB color.
 * Returns a Promise<dataURL>.
 */
function tintMaskImage(imgSrc, rgb) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = maskW;
      c.height = maskH;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, maskW, maskH);

      const src = ctx.getImageData(0, 0, maskW, maskH);
      const dst = ctx.createImageData(maskW, maskH);
      const sd = src.data;
      const dd = dst.data;

      for (let i = 0; i < sd.length; i += 4) {
        if (sd[i] > 127) {
          dd[i]     = rgb[0];
          dd[i + 1] = rgb[1];
          dd[i + 2] = rgb[2];
          dd[i + 3] = 180;
        }
      }
      ctx.putImageData(dst, 0, 0);
      resolve(c.toDataURL("image/png"));
    };
    img.onerror = () => resolve(null);
    img.src = imgSrc;
  });
}

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------
function pushUndo() {
  if (selectedIdx < 0) return;
  const layout = layouts[selectedIdx];
  const key = layout.name;
  if (!undoStacks[key]) undoStacks[key] = [];
  if (!redoStacks[key]) redoStacks[key] = [];

  const ctx = layout._maskCanvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, maskW, maskH);
  undoStacks[key].push(imgData);
  if (undoStacks[key].length > UNDO_LIMIT) undoStacks[key].shift();
  redoStacks[key] = [];
  markDirty();
}

function undo() {
  if (selectedIdx < 0) return;
  const layout = layouts[selectedIdx];
  const key = layout.name;
  if (!undoStacks[key] || !undoStacks[key].length) return;

  const ctx = layout._maskCanvas.getContext("2d");
  const current = ctx.getImageData(0, 0, maskW, maskH);
  if (!redoStacks[key]) redoStacks[key] = [];
  redoStacks[key].push(current);

  const prev = undoStacks[key].pop();
  ctx.putImageData(prev, 0, 0);
  updateMaskOverlay();
  updateLayoutStats();
}

function redo() {
  if (selectedIdx < 0) return;
  const layout = layouts[selectedIdx];
  const key = layout.name;
  if (!redoStacks[key] || !redoStacks[key].length) return;

  const ctx = layout._maskCanvas.getContext("2d");
  const current = ctx.getImageData(0, 0, maskW, maskH);
  if (!undoStacks[key]) undoStacks[key] = [];
  undoStacks[key].push(current);

  const next = redoStacks[key].pop();
  ctx.putImageData(next, 0, 0);
  updateMaskOverlay();
  updateLayoutStats();
}

function markDirty() {
  dirty = true;
  if (selectedIdx >= 0) layouts[selectedIdx]._dirty = true;
  $("dirtyFlag").hidden = false;
}

function markClean() {
  dirty = false;
  layouts.forEach(l => l._dirty = false);
  $("dirtyFlag").hidden = true;
}

// ---------------------------------------------------------------------------
// Painting  (eraser uses source-over + black)
// ---------------------------------------------------------------------------
function paintCircle(ctx, cx, cy, radius, isEraser) {
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = isEraser ? "#000000" : "#ffffff";
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
}

function paintLine(ctx, x0, y0, x1, y1, radius, isEraser) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const steps = Math.max(1, Math.ceil(dist / (radius * 0.3)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const cx = x0 + dx * t;
    const cy = y0 + dy * t;
    paintCircle(ctx, cx, cy, radius, isEraser);
  }
}

function handlePaintStart(e) {
  if (selectedIdx < 0) return;
  if (e.originalEvent.button !== 0) return;

  painting = true;
  pushUndo();

  const [px, py] = latLngToPixel(e.latlng.lat, e.latlng.lng);
  const isEraser = tool === "eraser";
  const layout = layouts[selectedIdx];
  const ctx = layout._maskCanvas.getContext("2d");

  paintCircle(ctx, px, py, brushSize / 2, isEraser);
  lastPaintPt = [px, py];

  updateMaskOverlayFast();
}

function handlePaintMove(e) {
  updateBrushCursor(e.originalEvent);

  if (!painting || selectedIdx < 0) return;

  const [px, py] = latLngToPixel(e.latlng.lat, e.latlng.lng);
  const isEraser = tool === "eraser";
  const layout = layouts[selectedIdx];
  const ctx = layout._maskCanvas.getContext("2d");

  if (lastPaintPt) {
    paintLine(ctx, lastPaintPt[0], lastPaintPt[1], px, py, brushSize / 2, isEraser);
  } else {
    paintCircle(ctx, px, py, brushSize / 2, isEraser);
  }
  lastPaintPt = [px, py];

  updateMaskOverlayFast();
}

function handlePaintEnd() {
  if (!painting) return;
  painting = false;
  lastPaintPt = null;
  updateMaskOverlay();
  updateLayoutStats();
}

// Full quality overlay update (pixel processing)
function updateMaskOverlay() {
  if (selectedIdx < 0) {
    if (maskOverlay) maskOverlay.setUrl("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg==");
    return;
  }
  const layout = layouts[selectedIdx];
  const url = maskToDisplay(layout._maskCanvas, layout.color);
  if (maskOverlay) maskOverlay.setUrl(url);
}

// Throttled update during painting
let rafPending = false;
function updateMaskOverlayFast() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    updateMaskOverlay();
  });
}

// ---------------------------------------------------------------------------
// Brush cursor  (uses latLngToContainerPoint for correct scale)
// ---------------------------------------------------------------------------
function updateBrushCursor(mouseEvent) {
  const cursor = $("brushCursor");
  if (selectedIdx < 0 || !mouseEvent) {
    cursor.style.display = "none";
    return;
  }

  const pt0 = map.latLngToContainerPoint(pixelToLatLng(0, 0));
  const pt1 = map.latLngToContainerPoint(pixelToLatLng(1, 0));
  const screenPxPerMaskPx = Math.abs(pt1.x - pt0.x);
  const cursorSize = Math.max(4, brushSize * screenPxPerMaskPx);

  cursor.style.display = "block";
  cursor.style.left = mouseEvent.clientX + "px";
  cursor.style.top = mouseEvent.clientY + "px";
  cursor.style.width = cursorSize + "px";
  cursor.style.height = cursorSize + "px";
  cursor.style.borderColor = tool === "eraser"
    ? "rgba(255, 100, 100, 0.8)"
    : "rgba(255, 255, 255, 0.8)";
}

function hideBrushCursor() {
  $("brushCursor").style.display = "none";
}

// ---------------------------------------------------------------------------
// Base map visibility
// ---------------------------------------------------------------------------
function toggleBaseMap(visible) {
  baseMapVisible = visible;
  if (baseImageOverlay) {
    baseImageOverlay.setOpacity(visible ? 1.0 : 0.0);
  }
}

// ---------------------------------------------------------------------------
// Layout list rendering
// ---------------------------------------------------------------------------
function renderLayoutList() {
  const el = $("layoutList");
  el.innerHTML = "";

  layouts.forEach((layout, idx) => {
    const div = document.createElement("div");
    div.className = `le-layout-item ${idx === selectedIdx ? "le-layout-item--selected" : ""}`;
    div.innerHTML = `
      <div class="le-layout-swatch" style="background:${layout.color}"></div>
      <div class="le-layout-label">${layout.name}</div>
      <div class="le-layout-dir">${layout.track_direction}</div>
      <button class="le-layout-del" data-idx="${idx}" title="Delete">&times;</button>
    `;

    div.addEventListener("click", (e) => {
      if (e.target.closest(".le-layout-del")) return;
      selectLayout(idx);
    });

    div.querySelector(".le-layout-del").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteLayout(idx);
    });

    el.appendChild(div);
  });
}

function updateLayoutStats() {
  if (selectedIdx < 0) return;
  const layout = layouts[selectedIdx];
  const white = countWhitePixels(layout._maskCanvas);
  const total = maskW * maskH;
  const pct = total > 0 ? ((white / total) * 100).toFixed(1) : "0.0";
  $("selLayoutStats").textContent = `Mask pixels: ${white.toLocaleString()} (${pct}%)`;
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------
function selectLayout(idx) {
  selectedIdx = idx;
  renderLayoutList();
  updateMaskOverlay();

  const sec = $("selLayoutSection");
  if (idx < 0 || idx >= layouts.length) {
    sec.hidden = true;
    return;
  }

  sec.hidden = false;
  const layout = layouts[idx];
  $("selLayoutName").value = layout.name;
  $("selLayoutColor").value = layout.color;
  $("selLayoutDir").value = layout.track_direction;
  updateLayoutStats();
}

// ---------------------------------------------------------------------------
// Layout CRUD
// ---------------------------------------------------------------------------
function addLayout(name, cloneSource) {
  const color = DEFAULT_COLORS[layouts.length % DEFAULT_COLORS.length];
  const safeName = name.replace(/[^\w\- ]/g, '_').trim() || `Layout_${layouts.length + 1}`;
  const maskFile = safeName.replace(/\s+/g, '_') + ".png";

  const canvas = createMaskCanvas();
  const layout = {
    name: safeName,
    color,
    track_direction: "clockwise",
    mask_file: maskFile,
    _maskCanvas: canvas,
    _dirty: true,
  };

  layouts.push(layout);
  markDirty();

  if (cloneSource) {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, maskW, maskH);
      ctx.drawImage(img, 0, 0, maskW, maskH);
      selectLayout(layouts.length - 1);
      setStatus(`Layout "${safeName}" created (cloned from ${cloneSource})`);
    };
    img.onerror = () => {
      selectLayout(layouts.length - 1);
      setStatus(`Layout "${safeName}" created (${cloneSource} not available)`);
    };
    img.src = `/api/mask_overlay/${cloneSource}`;
  } else {
    selectLayout(layouts.length - 1);
    setStatus(`Layout "${safeName}" created (blank)`);
  }

  renderLayoutList();
}

function deleteLayout(idx) {
  if (idx < 0 || idx >= layouts.length) return;
  const name = layouts[idx].name;
  layouts.splice(idx, 1);
  if (selectedIdx === idx) {
    selectedIdx = layouts.length > 0 ? Math.min(idx, layouts.length - 1) : -1;
  } else if (selectedIdx > idx) {
    selectedIdx--;
  }
  markDirty();
  renderLayoutList();
  selectLayout(selectedIdx);
  setStatus(`Layout "${name}" deleted`);
}

// ---------------------------------------------------------------------------
// Reference overlays  (pixel-level tint, black→transparent)
// ---------------------------------------------------------------------------
async function toggleRefOverlay(tag, visible) {
  if (visible) {
    if (refOverlays[tag]) {
      refOverlays[tag].addTo(map);
    } else {
      const rgb = REF_COLORS[tag] || [255, 255, 255];
      const dataUrl = await tintMaskImage(`/api/mask_overlay/${tag}`, rgb);
      if (!dataUrl) {
        setStatus(`Reference mask "${tag}" not available`);
        // Uncheck the chip
        const chip = document.querySelector(`.le-chip[data-layer="${tag}"]`);
        if (chip) chip.classList.remove("le-chip--on");
        return;
      }
      const overlay = L.imageOverlay(dataUrl, imageBounds, {
        opacity: 0.7,
        interactive: false,
      });
      overlay.addTo(map);
      refOverlays[tag] = overlay;
    }
  } else {
    if (refOverlays[tag]) {
      refOverlays[tag].remove();
    }
  }
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
async function saveAll() {
  setStatus("Saving...");
  try {
    const meta = {
      layouts: layouts.map(l => ({
        name: l.name,
        color: l.color,
        track_direction: l.track_direction,
        mask_file: l.mask_file,
      })),
    };

    const metaResp = await fetch("/api/track_layouts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(meta),
    });
    if (!metaResp.ok) throw new Error(`layouts.json: ${metaResp.status}`);

    let savedCount = 0;
    for (const layout of layouts) {
      if (!layout._dirty && !dirty) continue;
      const safeName = layout.mask_file.replace(/\.png$/, "");
      const blob = await canvasToBlob(layout._maskCanvas);
      const maskResp = await fetch(`/api/layout_mask/${safeName}`, {
        method: "POST",
        headers: { "Content-Type": "image/png" },
        body: blob,
      });
      if (!maskResp.ok) throw new Error(`mask ${safeName}: ${maskResp.status}`);
      layout._dirty = false;
      savedCount++;
    }

    markClean();
    setStatus(`Saved ${layouts.length} layout(s), ${savedCount} mask(s) updated`);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  // Create map — Simple CRS, NO default drag (right-click drag instead)
  map = L.map("map", {
    crs: L.CRS.Simple,
    zoomControl: true,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
    minZoom: -2,
    maxZoom: 6,
    dragging: false,
  });

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
    setStatus("Error: No geo_metadata found. Run stage 2 first.");
    return;
  }

  maskW = geoMeta.image_width;
  maskH = geoMeta.image_height;
  imageBounds = [[0, 0], [maskH, maskW]];

  // Load modelscale image as base layer
  try {
    baseImageOverlay = L.imageOverlay("/api/modelscale_image", imageBounds, {
      opacity: 1.0,
      interactive: false,
    }).addTo(map);
  } catch (e) {
    setStatus("Warning: Could not load modelscale image");
  }

  // Mask overlay (will be updated via pixel processing)
  maskOverlay = L.imageOverlay("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg==", imageBounds, {
    opacity: 1.0,
    interactive: false,
  }).addTo(map);

  map.fitBounds(imageBounds);

  // Load existing layouts
  try {
    const resp = await fetch("/api/track_layouts");
    if (resp.ok) {
      const data = await resp.json();
      for (const layoutMeta of (data.layouts || [])) {
        const canvas = createMaskCanvas();
        const layout = {
          ...layoutMeta,
          _maskCanvas: canvas,
          _dirty: false,
        };
        layouts.push(layout);

        const safeName = layoutMeta.mask_file?.replace(/\.png$/, "") || _safe(layoutMeta.name);
        try {
          await loadImageToCanvas(canvas, `/api/layout_mask/${safeName}`);
        } catch {
          // Mask file may not exist yet
        }
      }
    }
  } catch {}

  renderLayoutList();
  if (layouts.length > 0) selectLayout(0);
  setStatus(`Loaded ${layouts.length} layout(s). Left-click: paint. Right-drag: pan.`);

  wireUI();
  wireMapEvents();
}

function _safe(name) {
  return (name || "unnamed").replace(/[^\w\-]/g, '_').replace(/^_+|_+$/g, '') || "unnamed";
}

// ---------------------------------------------------------------------------
// Map events  (left-click = paint, right-click = drag)
// ---------------------------------------------------------------------------
function wireMapEvents() {
  const mapEl = $("map");

  // --- Left click: paint ---
  map.on("mousedown", (e) => {
    if (e.originalEvent.button === 0 && selectedIdx >= 0) {
      handlePaintStart(e);
    }
  });

  map.on("mousemove", (e) => {
    handlePaintMove(e);
  });

  map.on("mouseup", (e) => {
    if (e.originalEvent.button === 0) handlePaintEnd();
  });
  document.addEventListener("mouseup", (e) => {
    if (e.button === 0) handlePaintEnd();
    if (e.button === 2) { rightDragging = false; rightDragStart = null; }
  });

  // --- Right click: drag map ---
  mapEl.addEventListener("mousedown", (e) => {
    if (e.button === 2) {
      rightDragging = true;
      rightDragStart = { x: e.clientX, y: e.clientY };
      e.preventDefault();
    }
  });

  document.addEventListener("mousemove", (e) => {
    if (rightDragging && rightDragStart) {
      const dx = e.clientX - rightDragStart.x;
      const dy = e.clientY - rightDragStart.y;
      map.panBy([-dx, -dy], { animate: false });
      rightDragStart = { x: e.clientX, y: e.clientY };
    }
  });

  // Prevent context menu on map
  mapEl.addEventListener("contextmenu", (e) => e.preventDefault());

  // Ctrl+scroll for brush size, normal scroll for zoom
  mapEl.addEventListener("wheel", (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY > 0 ? -2 : 2;
    brushSize = Math.max(3, Math.min(100, brushSize + delta));
    $("brushSize").value = brushSize;
    $("brushSizeVal").textContent = brushSize + "px";
    updateBrushCursor(e);
  }, { passive: false });

  // Hide cursor when leaving map
  mapEl.addEventListener("mouseleave", () => hideBrushCursor());

  // Show paint cursor style when hovering map with a layout selected
  mapEl.addEventListener("mouseenter", () => {
    if (selectedIdx >= 0) mapEl.classList.add("map--paint");
    else mapEl.classList.remove("map--paint");
  });
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
function wireUI() {
  // --- Segmented tool control ---
  const toolBtns = document.querySelectorAll("#toolSeg .le-seg__btn");
  toolBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tool = btn.dataset.tool;
      toolBtns.forEach(b => b.classList.remove("le-seg__btn--active"));
      btn.classList.add("le-seg__btn--active");
    });
  });

  // Brush size slider
  $("brushSize").addEventListener("input", (e) => {
    brushSize = parseInt(e.target.value, 10);
    $("brushSizeVal").textContent = brushSize + "px";
  });

  // --- Layer chip toggles ---
  document.querySelectorAll("#layerChips .le-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const layer = chip.dataset.layer;
      const isOn = chip.classList.toggle("le-chip--on");

      if (layer === "basemap") {
        toggleBaseMap(isOn);
      } else {
        toggleRefOverlay(layer, isOn);
      }
    });
  });

  // Layout actions
  $("btnNewLayout").addEventListener("click", () => {
    const name = prompt("Layout name:", `Layout ${layouts.length + 1}`);
    if (name) addLayout(name, null);
  });

  $("btnCloneMerged").addEventListener("click", () => {
    const name = prompt("Layout name:", `Layout ${layouts.length + 1}`);
    if (name) addLayout(name, "merged");
  });

  // Selected layout controls
  $("selLayoutName").addEventListener("change", (e) => {
    if (selectedIdx < 0) return;
    const oldName = layouts[selectedIdx].name;
    layouts[selectedIdx].name = e.target.value;
    layouts[selectedIdx].mask_file = e.target.value.replace(/\s+/g, '_') + ".png";
    if (undoStacks[oldName]) {
      undoStacks[e.target.value] = undoStacks[oldName];
      delete undoStacks[oldName];
    }
    if (redoStacks[oldName]) {
      redoStacks[e.target.value] = redoStacks[oldName];
      delete redoStacks[oldName];
    }
    markDirty();
    renderLayoutList();
  });

  $("selLayoutColor").addEventListener("input", (e) => {
    if (selectedIdx < 0) return;
    layouts[selectedIdx].color = e.target.value;
    updateMaskOverlay();
    renderLayoutList();
  });

  $("selLayoutDir").addEventListener("change", (e) => {
    if (selectedIdx < 0) return;
    layouts[selectedIdx].track_direction = e.target.value;
    markDirty();
    renderLayoutList();
  });

  $("btnClearMask").addEventListener("click", () => {
    if (selectedIdx < 0) return;
    pushUndo();
    const ctx = layouts[selectedIdx]._maskCanvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, maskW, maskH);
    updateMaskOverlay();
    updateLayoutStats();
    setStatus("Mask cleared");
  });

  $("btnDeleteLayout").addEventListener("click", () => {
    if (selectedIdx >= 0) deleteLayout(selectedIdx);
  });

  // Toolbar buttons
  $("btnSaveAll").addEventListener("click", () => saveAll());
  $("btnUndo").addEventListener("click", () => undo());
  $("btnRedo").addEventListener("click", () => redo());

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

    if (e.key === "s" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault(); saveAll(); return;
    }
    if (e.key === "z" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault(); undo(); return;
    }
    if ((e.key === "z" && (e.ctrlKey || e.metaKey) && e.shiftKey) ||
        (e.key === "y" && (e.ctrlKey || e.metaKey))) {
      e.preventDefault(); redo(); return;
    }
    if (e.key === "b") {
      tool = "brush";
      toolBtns.forEach(b => b.classList.toggle("le-seg__btn--active", b.dataset.tool === "brush"));
    }
    if (e.key === "e") {
      tool = "eraser";
      toolBtns.forEach(b => b.classList.toggle("le-seg__btn--active", b.dataset.tool === "eraser"));
    }
  });

  // Before-unload warning
  window.addEventListener("beforeunload", (e) => {
    if (dirty) { e.preventDefault(); e.returnValue = ""; }
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
