/* global L, setupRightDrag */
/* Surface Editor — geographic CRS with per-tag mask overlay editing */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const TILE_SIZE = 512;
const UNDO_LIMIT = 30;

// Priority compositing order (low → high), same as Stage 5.
// Higher-priority tags overwrite lower: kerb > road > road2 > grass > sand.
const SURFACE_TAGS = [
  { tag: "sand",  color: "#c8c864", label: "砂石 sand" },
  { tag: "grass", color: "#00c800", label: "草地 grass" },
  { tag: "road2", color: "#b4b4b4", label: "次路面 road2" },
  { tag: "road",  color: "#666666", label: "路面 road" },
  { tag: "kerb",  color: "#ff0000", label: "路缘 kerb" },
];

const $ = (id) => document.getElementById(id);
function setStatus(msg) { $("status").textContent = msg; }
function parseHex(hex) {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let map;
let basemapLayer;
let maskW = 0, maskH = 0, gridCols = 0, gridRows = 0;
let geoBounds = null;       // {north, south, east, west}
let geoCorners = null;      // {top_left, top_right, bottom_left, bottom_right} as [lat,lon]
let layers = [];            // [{ tag, color, label, maskLayer: MaskLayer }]
let selectedIdx = -1;
let tool = "brush";
let brushScreenSize = 20;   // screen pixels (constant visual size)
let dirty = false;
let painting = false;
let lastPaintPt = null;     // [canvasX, canvasY]
let undoStacks = {};        // tag -> [UndoEntry]
let redoStacks = {};        // tag -> [UndoEntry]
let lastMouseEvent = null;  // cache for cursor size updates

// ---------------------------------------------------------------------------
// Coordinate conversion (canvas pixel <-> geographic latlng)
// ---------------------------------------------------------------------------
function canvasPixelToLatLng(cx, cy) {
  if (geoCorners) {
    // Bilinear interpolation using 4 true WGS84 corners.
    // Eliminates UTM grid convergence error (up to 13m → <0.01m).
    const u = cx / maskW;   // [0, 1]  left→right
    const v = cy / maskH;   // [0, 1]  top→bottom
    const tl = geoCorners.top_left, tr = geoCorners.top_right;
    const bl = geoCorners.bottom_left, br = geoCorners.bottom_right;
    const lat = (1-u)*(1-v)*tl[0] + u*(1-v)*tr[0] + (1-u)*v*bl[0] + u*v*br[0];
    const lng = (1-u)*(1-v)*tl[1] + u*(1-v)*tr[1] + (1-u)*v*bl[1] + u*v*br[1];
    return [lat, lng];
  }
  const { north, south, east, west } = geoBounds;
  const lat = north - (cy / maskH) * (north - south);
  const lng = west + (cx / maskW) * (east - west);
  return [lat, lng];
}

function latLngToCanvasPixel(lat, lng) {
  if (geoCorners) {
    // Inverse bilinear interpolation (Newton iteration, 2–3 steps sufficient).
    const tl = geoCorners.top_left, tr = geoCorners.top_right;
    const bl = geoCorners.bottom_left, br = geoCorners.bottom_right;
    let u = 0.5, v = 0.5;
    for (let i = 0; i < 4; i++) {
      const fLat = (1-u)*(1-v)*tl[0] + u*(1-v)*tr[0] + (1-u)*v*bl[0] + u*v*br[0] - lat;
      const fLng = (1-u)*(1-v)*tl[1] + u*(1-v)*tr[1] + (1-u)*v*bl[1] + u*v*br[1] - lng;
      // Jacobian
      const dLat_du = (1-v)*(tr[0]-tl[0]) + v*(br[0]-bl[0]);
      const dLat_dv = (1-u)*(bl[0]-tl[0]) + u*(br[0]-tr[0]);
      const dLng_du = (1-v)*(tr[1]-tl[1]) + v*(br[1]-bl[1]);
      const dLng_dv = (1-u)*(bl[1]-tl[1]) + u*(br[1]-tr[1]);
      const det = dLat_du * dLng_dv - dLat_dv * dLng_du;
      if (Math.abs(det) < 1e-20) break;
      u -= (fLat * dLng_dv - fLng * dLat_dv) / det;
      v -= (fLng * dLat_du - fLat * dLng_du) / det;
    }
    return [u * maskW, v * maskH];
  }
  const { north, south, east, west } = geoBounds;
  const cx = ((lng - west) / (east - west)) * maskW;
  const cy = ((north - lat) / (north - south)) * maskH;
  return [cx, cy];
}

function tileBounds(col, row) {
  const ts = TILE_SIZE;
  return L.latLngBounds(
    canvasPixelToLatLng(col * ts, (row + 1) * ts),       // SW
    canvasPixelToLatLng((col + 1) * ts, row * ts)         // NE
  );
}

// ---------------------------------------------------------------------------
// Image loader helper
// ---------------------------------------------------------------------------
function loadImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

// ---------------------------------------------------------------------------
// MaskCanvasRenderer — single canvas per tag (eliminates tile seams)
//
// Per-tile L.imageOverlay used axis-aligned lat/lng rectangles, which caused
// ~10px overlap when the GeoTIFF has rotation (512 * sin(θ)).  This renderer
// draws ALL tiles onto one viewport-sized canvas using a shared affine
// transform, so tiles are guaranteed to align pixel-perfectly.
// ---------------------------------------------------------------------------
const MaskCanvasRenderer = L.Layer.extend({
  initialize(maskLayer) {
    this._ml = maskLayer;
    this._canvas = null;
    this._animFrame = 0;
  },

  onAdd(map) {
    this._map = map;
    this._canvas = L.DomUtil.create('canvas', 'mask-canvas', map.getPane('overlayPane'));
    this._canvas.style.pointerEvents = 'none';
    this._canvas.style.imageRendering = 'pixelated';
    map.on('viewreset zoomend moveend', this._draw, this);
    this._draw();
  },

  onRemove(map) {
    if (this._animFrame) { cancelAnimationFrame(this._animFrame); this._animFrame = 0; }
    if (this._canvas) L.DomUtil.remove(this._canvas);
    this._canvas = null;
    map.off('viewreset zoomend moveend', this._draw, this);
  },

  /** Batch redraws via requestAnimationFrame (for painting performance). */
  requestRedraw() {
    if (this._animFrame || !this._canvas) return;
    this._animFrame = requestAnimationFrame(() => {
      this._animFrame = 0;
      this._draw();
    });
  },

  _draw() {
    if (!this._map || !this._canvas) return;
    const map = this._map;
    const size = map.getSize();

    // Viewport-sized canvas, positioned at container origin in layer coords
    const topLeft = map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this._canvas, topLeft);
    this._canvas.width = size.x;
    this._canvas.height = size.y;

    const ctx = this._canvas.getContext('2d');
    ctx.clearRect(0, 0, size.x, size.y);

    if (!this._ml._layerVisible) return;
    ctx.globalAlpha = this._ml._opacity;
    ctx.imageSmoothingEnabled = false;

    // Shared affine: maskPixel(px, py) → layerPoint(lx, ly)
    //   lx = O.x + px * ax + py * cx
    //   ly = O.y + px * bx + py * dy
    const O  = map.latLngToLayerPoint(canvasPixelToLatLng(0, 0));
    const Rx = map.latLngToLayerPoint(canvasPixelToLatLng(maskW, 0));
    const Ry = map.latLngToLayerPoint(canvasPixelToLatLng(0, maskH));
    const ax = (Rx.x - O.x) / maskW;
    const bx = (Rx.y - O.y) / maskW;
    const cx = (Ry.x - O.x) / maskH;
    const dy = (Ry.y - O.y) / maskH;

    const ts = TILE_SIZE;
    const offX = O.x - topLeft.x;
    const offY = O.y - topLeft.y;

    for (const [key, display] of this._ml._displayTiles) {
      const [col, row] = key.split('_').map(Number);
      const px = col * ts, py = row * ts;
      const tx = offX + px * ax + py * cx;
      const ty = offY + px * bx + py * dy;

      // Cull tiles fully outside viewport
      const x1 = tx + ts * ax, x2 = tx + ts * cx, x3 = tx + ts * (ax + cx);
      const y1 = ty + ts * bx, y2 = ty + ts * dy, y3 = ty + ts * (bx + dy);
      const minTX = Math.min(tx, x1, x2, x3);
      const maxTX = Math.max(tx, x1, x2, x3);
      const minTY = Math.min(ty, y1, y2, y3);
      const maxTY = Math.max(ty, y1, y2, y3);
      if (maxTX < 0 || maxTY < 0 || minTX > size.x || minTY > size.y) continue;

      ctx.setTransform(ax, bx, cx, dy, tx, ty);
      ctx.drawImage(display, 0, 0);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  },
});

// ---------------------------------------------------------------------------
// MaskLayer — manages one tag's raw tiles + display tiles
// ---------------------------------------------------------------------------
class MaskLayer {
  constructor(tag, color, cols, rows) {
    this.tag = tag;
    this.color = color;
    this._rgb = parseHex(color);
    this._gridCols = cols;
    this._gridRows = rows;
    this._rawTiles = new Map();       // "col_row" -> Canvas (grayscale mask)
    this._displayTiles = new Map();   // "col_row" -> Canvas (tinted display)
    this._renderer = null;            // MaskCanvasRenderer (single canvas per tag)
    this._dirtyTiles = new Set();
    this._layerVisible = true;
    this._opacity = 0.6;
  }

  /** Attach canvas renderer to the map (called once). */
  _initRenderer() {
    if (this._renderer) return;
    this._renderer = new MaskCanvasRenderer(this);
    this._renderer.addTo(map);
  }

  /** Load mask tiles from server. */
  async load() {
    this._initRenderer();
    await this._loadTiles("");
  }

  /** Reload mask from server (e.g. after compositing changed the mask). */
  async reload() {
    this._rawTiles.clear();
    this._displayTiles.clear();
    this._dirtyTiles.clear();
    if (this._renderer) this._renderer._draw();
    await this._loadTiles(`?t=${Date.now()}`);
  }

  async _loadTiles(cacheBust) {
    // Fetch non-empty tile index first, then only request those tiles.
    // This avoids thousands of unnecessary requests for empty tiles.
    let tileKeys;
    try {
      const resp = await fetch(`/api/surface_tile_index/${this.tag}${cacheBust}`);
      if (resp.ok) {
        const data = await resp.json();
        tileKeys = data.tiles || [];
      }
    } catch {}

    if (!tileKeys) {
      // Fallback: request all tiles (shouldn't happen normally)
      tileKeys = [];
      for (let r = 0; r < this._gridRows; r++)
        for (let c = 0; c < this._gridCols; c++)
          tileKeys.push(`${c}_${r}`);
    }

    // Load non-empty tiles in parallel (batched to avoid connection flooding)
    const BATCH = 30;
    for (let i = 0; i < tileKeys.length; i += BATCH) {
      const batch = tileKeys.slice(i, i + BATCH);
      await Promise.all(batch.map(key => {
        const [c, r] = key.split("_").map(Number);
        return this._loadOneTile(c, r, cacheBust);
      }));
    }
  }

  async _loadOneTile(col, row, cacheBust) {
    const img = await loadImage(
      `/api/surface_tile/${this.tag}/${col}_${row}.png${cacheBust}`
    );
    if (!img) return;

    const ts = TILE_SIZE;
    const raw = document.createElement("canvas");
    raw.width = ts;
    raw.height = ts;
    raw.getContext("2d").drawImage(img, 0, 0, ts, ts);

    if (this._tileIsEmpty(raw)) return;

    const key = `${col}_${row}`;
    this._rawTiles.set(key, raw);
    const display = this._tintTile(raw);
    this._displayTiles.set(key, display);
    if (this._renderer) this._renderer.requestRedraw();
  }

  _tileIsEmpty(canvas) {
    const d = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] > 127) return false;
    }
    return true;
  }

  _tintTile(rawCanvas) {
    const display = document.createElement("canvas");
    display.width = TILE_SIZE;
    display.height = TILE_SIZE;
    this._tint(rawCanvas, display.getContext("2d"));
    return display;
  }

  _tint(rawCanvas, destCtx) {
    const rCtx = rawCanvas.getContext("2d");
    const src = rCtx.getImageData(0, 0, TILE_SIZE, TILE_SIZE);
    const dst = destCtx.createImageData(TILE_SIZE, TILE_SIZE);
    const sd = src.data, dd = dst.data;
    const [cr, cg, cb] = this._rgb;
    for (let i = 0; i < sd.length; i += 4) {
      if (sd[i] > 127) {
        dd[i] = cr; dd[i + 1] = cg; dd[i + 2] = cb; dd[i + 3] = 160;
      }
    }
    destCtx.putImageData(dst, 0, 0);
  }

  _redraw(key) {
    const display = this._displayTiles.get(key);
    const raw = this._rawTiles.get(key);
    if (!display || !raw) return;
    const ctx = display.getContext("2d");
    ctx.clearRect(0, 0, TILE_SIZE, TILE_SIZE);
    this._tint(raw, ctx);
    if (this._renderer) this._renderer.requestRedraw();
  }

  /** Ensure tile exists (create blank for painting into empty areas). */
  _ensureTile(col, row) {
    const key = `${col}_${row}`;
    if (this._rawTiles.has(key)) return;
    if (col < 0 || row < 0 || col >= this._gridCols || row >= this._gridRows) return;

    const raw = document.createElement("canvas");
    raw.width = TILE_SIZE;
    raw.height = TILE_SIZE;
    const ctx = raw.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, TILE_SIZE, TILE_SIZE);
    this._rawTiles.set(key, raw);

    const display = document.createElement("canvas");
    display.width = TILE_SIZE;
    display.height = TILE_SIZE;
    this._displayTiles.set(key, display);
  }

  // -- Painting ----------------------------------------------------------

  /** Paint a filled circle at canvas pixel (imgX, imgY). */
  paintCircle(imgX, imgY, radius, isEraser) {
    const ts = TILE_SIZE;
    const minC = Math.max(0, Math.floor((imgX - radius) / ts));
    const maxC = Math.min(this._gridCols - 1, Math.floor((imgX + radius) / ts));
    const minR = Math.max(0, Math.floor((imgY - radius) / ts));
    const maxR = Math.min(this._gridRows - 1, Math.floor((imgY + radius) / ts));
    for (let c = minC; c <= maxC; c++) {
      for (let r = minR; r <= maxR; r++) {
        this._ensureTile(c, r);
        const key = `${c}_${r}`;
        const raw = this._rawTiles.get(key);
        if (!raw) continue;
        if (window._currentUndo) window._currentUndo.save(key, raw);
        const ctx = raw.getContext("2d");
        ctx.fillStyle = isEraser ? "#000" : "#fff";
        ctx.beginPath();
        ctx.arc(imgX - c * ts, imgY - r * ts, radius, 0, Math.PI * 2);
        ctx.fill();
        this._dirtyTiles.add(key);
      }
    }
  }

  /** Paint a line from (x0,y0) to (x1,y1) in canvas pixels. */
  paintLine(x0, y0, x1, y1, radius, isEraser) {
    const dx = x1 - x0, dy = y1 - y0;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const steps = Math.max(1, Math.ceil(dist / (radius * 0.3)));
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      this.paintCircle(x0 + dx * t, y0 + dy * t, radius, isEraser);
    }
  }

  /** Refresh display for tiles in the given canvas-pixel bounding rect. */
  refreshRect(x0, y0, x1, y1) {
    const ts = TILE_SIZE;
    const minC = Math.max(0, Math.floor(x0 / ts));
    const maxC = Math.min(this._gridCols - 1, Math.floor(x1 / ts));
    const minR = Math.max(0, Math.floor(y0 / ts));
    const maxR = Math.min(this._gridRows - 1, Math.floor(y1 / ts));
    for (let c = minC; c <= maxC; c++) {
      for (let r = minR; r <= maxR; r++) {
        this._redraw(`${c}_${r}`);
      }
    }
  }

  // -- Visibility --------------------------------------------------------
  setLayerVisible(v) {
    this._layerVisible = v;
    if (this._renderer && this._renderer._canvas) {
      this._renderer._canvas.style.display = v ? '' : 'none';
      if (v) this._renderer.requestRedraw();
    }
  }

  // -- Save helpers ------------------------------------------------------
  getDirtyKeys() { return [...this._dirtyTiles]; }

  async getTileBlob(key) {
    const raw = this._rawTiles.get(key);
    if (!raw) return null;
    return new Promise(resolve => raw.toBlob(resolve, "image/png"));
  }

  clearDirty() { this._dirtyTiles.clear(); }

  // -- Stats -------------------------------------------------------------
  countWhite() {
    let total = 0;
    for (const raw of this._rawTiles.values()) {
      const d = raw.getContext("2d").getImageData(0, 0, raw.width, raw.height).data;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 127) total++;
      }
    }
    return total;
  }
}

// ---------------------------------------------------------------------------
// Undo / Redo (per-tag, tile-level snapshots)
// ---------------------------------------------------------------------------
class UndoEntry {
  constructor() { this.tiles = new Map(); }
  /** Save tile snapshot before first modification. */
  save(key, rawCanvas) {
    if (this.tiles.has(key)) return;
    const ctx = rawCanvas.getContext("2d");
    this.tiles.set(key, ctx.getImageData(0, 0, TILE_SIZE, TILE_SIZE));
  }
}

// Global ref used by MaskLayer.paintCircle
window._currentUndo = null;

// ---------------------------------------------------------------------------
// Dirty flag
// ---------------------------------------------------------------------------
function markDirty() {
  dirty = true;
  $("dirtyFlag").hidden = false;
}
function markClean() {
  dirty = false;
  layers.forEach(l => l.maskLayer.clearDirty());
  $("dirtyFlag").hidden = true;
}

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------
function beginStroke() {
  window._currentUndo = new UndoEntry();
}

function endStroke() {
  const entry = window._currentUndo;
  window._currentUndo = null;
  if (!entry || entry.tiles.size === 0) return;
  if (selectedIdx < 0) return;
  const tag = layers[selectedIdx].tag;
  if (!undoStacks[tag]) undoStacks[tag] = [];
  undoStacks[tag].push(entry);
  if (undoStacks[tag].length > UNDO_LIMIT) undoStacks[tag].shift();
  redoStacks[tag] = [];
  markDirty();
}

function undo() {
  if (selectedIdx < 0) return;
  const tag = layers[selectedIdx].tag;
  const stack = undoStacks[tag];
  if (!stack || !stack.length) return;
  const entry = stack.pop();
  const ml = layers[selectedIdx].maskLayer;

  const redoEntry = new UndoEntry();
  for (const [key, imgData] of entry.tiles) {
    const raw = ml._rawTiles.get(key);
    if (!raw) continue;
    const ctx = raw.getContext("2d");
    redoEntry.tiles.set(key, ctx.getImageData(0, 0, TILE_SIZE, TILE_SIZE));
    ctx.putImageData(imgData, 0, 0);
    ml._dirtyTiles.add(key);
    ml._redraw(key);
  }
  if (!redoStacks[tag]) redoStacks[tag] = [];
  redoStacks[tag].push(redoEntry);
  updateTagStats();
}

function redo() {
  if (selectedIdx < 0) return;
  const tag = layers[selectedIdx].tag;
  const rstack = redoStacks[tag];
  if (!rstack || !rstack.length) return;
  const entry = rstack.pop();
  const ml = layers[selectedIdx].maskLayer;

  const undoEntry = new UndoEntry();
  for (const [key, imgData] of entry.tiles) {
    const raw = ml._rawTiles.get(key);
    if (!raw) continue;
    const ctx = raw.getContext("2d");
    undoEntry.tiles.set(key, ctx.getImageData(0, 0, TILE_SIZE, TILE_SIZE));
    ctx.putImageData(imgData, 0, 0);
    ml._dirtyTiles.add(key);
    ml._redraw(key);
  }
  if (!undoStacks[tag]) undoStacks[tag] = [];
  undoStacks[tag].push(undoEntry);
  updateTagStats();
}

// ---------------------------------------------------------------------------
// Brush: screen pixels -> canvas pixels conversion
// ---------------------------------------------------------------------------
function getCanvasRadius() {
  // Compute screen-pixels-per-canvas-pixel using the full geo extent
  // (numerically stable at any zoom level, unlike 1-pixel sampling).
  const { north, south, east, west } = geoBounds;
  const nwPt = map.latLngToContainerPoint([north, west]);
  const sePt = map.latLngToContainerPoint([south, east]);
  const scaleX = Math.abs(sePt.x - nwPt.x) / maskW;
  const scaleY = Math.abs(sePt.y - nwPt.y) / maskH;
  const scale = (scaleX + scaleY) / 2;
  if (scale < 0.001) return 1;
  return (brushScreenSize / 2) / scale;
}

// ---------------------------------------------------------------------------
// Painting
// ---------------------------------------------------------------------------
function handlePaintStart(e) {
  if (selectedIdx < 0 || e.originalEvent.button !== 0) return;
  painting = true;
  beginStroke();
  const [cx, cy] = latLngToCanvasPixel(e.latlng.lat, e.latlng.lng);
  const r = getCanvasRadius(e.latlng);
  const ml = layers[selectedIdx].maskLayer;
  ml.paintCircle(cx, cy, r, tool === "eraser");
  lastPaintPt = [cx, cy];
  ml.refreshRect(cx - r, cy - r, cx + r, cy + r);
}

function handlePaintMove(e) {
  updateBrushCursor(e.originalEvent);
  if (!painting || selectedIdx < 0) return;
  const [cx, cy] = latLngToCanvasPixel(e.latlng.lat, e.latlng.lng);
  const r = getCanvasRadius(e.latlng);
  const ml = layers[selectedIdx].maskLayer;
  if (lastPaintPt) {
    ml.paintLine(lastPaintPt[0], lastPaintPt[1], cx, cy, r, tool === "eraser");
    const x0 = Math.min(lastPaintPt[0], cx) - r;
    const y0 = Math.min(lastPaintPt[1], cy) - r;
    const x1 = Math.max(lastPaintPt[0], cx) + r;
    const y1 = Math.max(lastPaintPt[1], cy) + r;
    ml.refreshRect(x0, y0, x1, y1);
  } else {
    ml.paintCircle(cx, cy, r, tool === "eraser");
    ml.refreshRect(cx - r, cy - r, cx + r, cy + r);
  }
  lastPaintPt = [cx, cy];
}

function handlePaintEnd() {
  if (!painting) return;
  painting = false;
  lastPaintPt = null;
  endStroke();
  updateTagStats();
}

// ---------------------------------------------------------------------------
// Brush cursor — constant screen size (screen-relative)
// ---------------------------------------------------------------------------
function updateBrushCursor(mouseEvent) {
  if (mouseEvent) lastMouseEvent = mouseEvent;
  const cursor = $("brushCursor");
  const evt = mouseEvent || lastMouseEvent;
  if (selectedIdx < 0 || !evt) {
    cursor.style.display = "none";
    return;
  }
  cursor.style.display = "block";
  cursor.style.left = evt.clientX + "px";
  cursor.style.top = evt.clientY + "px";
  cursor.style.width = brushScreenSize + "px";
  cursor.style.height = brushScreenSize + "px";
  cursor.style.borderColor = tool === "eraser"
    ? "rgba(255, 100, 100, 0.8)"
    : "rgba(255, 255, 255, 0.8)";
}

function hideBrushCursor() {
  $("brushCursor").style.display = "none";
}

// ---------------------------------------------------------------------------
// Tag list / selection
// ---------------------------------------------------------------------------
function renderTagList() {
  const el = $("tagList");
  el.innerHTML = "";
  layers.forEach((layer, idx) => {
    const div = document.createElement("div");
    div.className = `se-tag-item ${idx === selectedIdx ? "se-tag-item--selected" : ""}`;
    div.innerHTML = `
      <div class="se-tag-swatch" style="background:${layer.color}"></div>
      <div class="se-tag-label">${layer.label}</div>
      <div class="se-tag-sublabel">${layer.tag}</div>`;
    div.addEventListener("click", () => selectTag(idx));
    el.appendChild(div);
  });
}

function updateTagStats() {
  if (selectedIdx < 0) return;
  const ml = layers[selectedIdx].maskLayer;
  const white = ml.countWhite();
  const total = maskW * maskH;
  const pct = total > 0 ? ((white / total) * 100).toFixed(1) : "0.0";
  $("selTagStats").textContent = `Mask pixels: ${white.toLocaleString()} (${pct}%)`;
}

function selectTag(idx) {
  selectedIdx = idx;
  renderTagList();
  const sec = $("selTagSection");
  if (idx < 0 || idx >= layers.length) { sec.hidden = true; return; }
  sec.hidden = false;
  $("selTagTitle").textContent = layers[idx].label;
  updateTagStats();
  $("map").classList.add("map--paint");
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
async function saveAll() {
  setStatus("Saving...");
  try {
    let tileCount = 0;
    for (const layer of layers) {
      const keys = layer.maskLayer.getDirtyKeys();
      for (const key of keys) {
        const blob = await layer.maskLayer.getTileBlob(key);
        if (!blob) continue;
        const resp = await fetch(`/api/surface_tile/${layer.tag}/${key}`, {
          method: "POST",
          headers: { "Content-Type": "image/png" },
          body: blob,
        });
        if (!resp.ok) throw new Error(`${layer.tag}/${key}: ${resp.status}`);
        tileCount++;
      }
    }
    // Flush cached masks to disk + reconvert with priority compositing
    setStatus("Saving... compositing masks");
    const flushResp = await fetch("/api/surface_flush", { method: "POST" });
    if (!flushResp.ok) throw new Error(`Flush failed: ${flushResp.status}`);
    const flushData = await flushResp.json();
    markClean();

    // If compositing was applied, reload all layers to show resolved overlaps
    if (flushData.composited) {
      setStatus("Reloading composited masks...");
      await Promise.all(layers.map(l => l.maskLayer.reload()));
      updateTagStats();
    }
    setStatus(`Saved ${tileCount} tile(s), flushed ${flushData.flushed} mask(s), composited`);
  } catch (err) {
    setStatus(`Save failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  // Load metadata
  let meta;
  try {
    const resp = await fetch("/api/surface_masks");
    if (resp.ok) meta = await resp.json();
  } catch {}
  if (!meta || !meta.image_width) {
    setStatus("Error: surface_masks.json not found. Run stage 5a first.");
    return;
  }

  maskW = meta.image_width;
  maskH = meta.image_height;
  gridCols = meta.grid_cols || Math.ceil(maskW / TILE_SIZE);
  gridRows = meta.grid_rows || Math.ceil(maskH / TILE_SIZE);

  // Update tag info from server
  if (meta.tags) {
    for (const st of meta.tags) {
      const local = SURFACE_TAGS.find(t => t.tag === st.tag);
      if (local && st.color) local.color = st.color;
      if (local && st.label) local.label = st.label + " " + st.tag;
    }
  }

  // Load geo metadata for coordinate conversion.
  // Surface editor is stage 5a (runs before 6/7), so prefer stage 2's
  // geo_metadata which is always fresh after running stage 2.
  let geoData;
  for (const url of ["/api/modelscale_geo_metadata", "/api/geo_metadata"]) {
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const d = await resp.json();
        if (d && d.bounds) { geoData = d; break; }
      }
    } catch {}
  }
  if (!geoData || !geoData.bounds) {
    setStatus("Error: geo_metadata.json not found. Run stage 2 first.");
    return;
  }
  geoBounds = geoData.bounds;
  geoCorners = geoData.corners || null;
  console.log(`[surface_editor] geoBounds: N=${geoBounds.north} S=${geoBounds.south} E=${geoBounds.east} W=${geoBounds.west}`);
  console.log(`[surface_editor] geoCorners: ${geoCorners ? 'yes (bilinear)' : 'no (linear fallback)'}`);

  // Create map with standard Web Mercator CRS
  map = L.map("map", {
    zoomControl: true,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
    dragging: false,
  });
  setupRightDrag(map, $("map"));

  // Geographic basemap tiles (same as objects_editor / wall_editor)
  basemapLayer = L.tileLayer("/tiles/{z}/{x}/{y}.png", {
    minZoom: 12,
    maxZoom: 24,
    maxNativeZoom: 24,
  }).addTo(map);

  // Fit to geo bounds
  const { north, south, east, west } = geoBounds;
  map.fitBounds([[south, west], [north, east]], { padding: [10, 10] });

  // Create and load mask layers
  setStatus("Loading mask layers...");
  for (const tagDef of SURFACE_TAGS) {
    const ml = new MaskLayer(tagDef.tag, tagDef.color, gridCols, gridRows);
    layers.push({
      tag: tagDef.tag,
      color: tagDef.color,
      label: tagDef.label,
      maskLayer: ml,
    });
  }

  // Load all mask layers in parallel
  await Promise.all(layers.map(l => l.maskLayer.load()));

  // Build layer chips (dynamic, after basemap chip in HTML)
  const chipsEl = $("layerChips");
  for (const layer of layers) {
    const [r, g, b] = parseHex(layer.color);
    const chip = document.createElement("div");
    chip.className = "le-chip le-chip--on";
    chip.dataset.layer = layer.tag;
    chip.style.cssText = `--chip-color:rgba(${r},${g},${b},0.5);--chip-bg:rgba(${r},${g},${b},0.15);--chip-glow:rgba(${r},${g},${b},0.1);--chip-dot:${layer.color}`;
    chip.innerHTML = `<span class="le-chip__dot"></span>${layer.tag}`;
    chipsEl.appendChild(chip);
  }

  renderTagList();
  if (layers.length > 0) selectTag(0);
  setStatus(`Loaded ${layers.length} tag(s), ${gridCols}\u00d7${gridRows} tiles. Left-click: paint. Right-drag: pan.`);

  wireUI();
  wireMapEvents();
}

// ---------------------------------------------------------------------------
// Map events
// ---------------------------------------------------------------------------
function wireMapEvents() {
  const mapEl = $("map");
  map.on("mousedown", (e) => {
    if (e.originalEvent.button === 0 && selectedIdx >= 0) handlePaintStart(e);
  });
  map.on("mousemove", (e) => handlePaintMove(e));
  map.on("mouseup", (e) => { if (e.originalEvent.button === 0) handlePaintEnd(); });
  document.addEventListener("mouseup", (e) => { if (e.button === 0) handlePaintEnd(); });

  // Ctrl+scroll for brush size
  mapEl.addEventListener("wheel", (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY > 0 ? -2 : 2;
    brushScreenSize = Math.max(3, Math.min(200, brushScreenSize + delta));
    $("brushSize").value = brushScreenSize;
    $("brushSizeVal").textContent = brushScreenSize;
    updateBrushCursor(e);
  }, { passive: false });

  mapEl.addEventListener("mouseleave", () => hideBrushCursor());
  mapEl.addEventListener("mouseenter", () => {
    if (selectedIdx >= 0) mapEl.classList.add("map--paint");
    else mapEl.classList.remove("map--paint");
  });
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
function wireUI() {
  // Tool selector
  document.querySelectorAll("#toolSeg .le-seg__btn").forEach(btn => {
    btn.addEventListener("click", () => {
      tool = btn.dataset.tool;
      document.querySelectorAll("#toolSeg .le-seg__btn").forEach(
        b => b.classList.toggle("le-seg__btn--active", b === btn)
      );
    });
  });

  // Brush size
  $("brushSize").addEventListener("input", (e) => {
    brushScreenSize = parseInt(e.target.value, 10);
    $("brushSizeVal").textContent = brushScreenSize;
    updateBrushCursor();
  });

  // Layer chip toggles
  document.querySelectorAll("#layerChips .le-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const tag = chip.dataset.layer;
      const isOn = chip.classList.toggle("le-chip--on");
      if (tag === "basemap") {
        basemapLayer.setOpacity(isOn ? 1.0 : 0.0);
      } else {
        const layer = layers.find(l => l.tag === tag);
        if (layer) layer.maskLayer.setLayerVisible(isOn);
      }
    });
  });

  // Clear mask
  $("btnClearMask").addEventListener("click", () => {
    if (selectedIdx < 0) return;
    beginStroke();
    const ml = layers[selectedIdx].maskLayer;
    for (const [key, raw] of ml._rawTiles) {
      if (window._currentUndo) window._currentUndo.save(key, raw);
      const ctx = raw.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, TILE_SIZE, TILE_SIZE);
      ml._dirtyTiles.add(key);
      ml._redraw(key);
    }
    endStroke();
    updateTagStats();
    setStatus("Mask cleared");
  });

  // Toolbar
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
      document.querySelectorAll("#toolSeg .le-seg__btn").forEach(
        b => b.classList.toggle("le-seg__btn--active", b.dataset.tool === "brush")
      );
    }
    if (e.key === "e") {
      tool = "eraser";
      document.querySelectorAll("#toolSeg .le-seg__btn").forEach(
        b => b.classList.toggle("le-seg__btn--active", b.dataset.tool === "eraser")
      );
    }
    // Number keys 1-5 to select tags
    const num = parseInt(e.key, 10);
    if (num >= 1 && num <= layers.length) selectTag(num - 1);
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
