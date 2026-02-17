/* global L */

// 底图瓦片默认放在与 index.html 同目录的 map/ 下：
// script/track_session_anaylzer/map/{z}/{x}/{y}.png
const TILE_URL = "./map/{z}/{x}/{y}.png";

const COLORS = [
  "#60a5fa",
  "#f472b6",
  "#34d399",
  "#fbbf24",
  "#a78bfa",
  "#fb7185",
  "#22d3ee",
  "#f97316",
  "#84cc16",
  "#e879f9",
];

const $ = (id) => document.getElementById(id);

function makeId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `ds_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function setStatus(msg) {
  $("status").textContent = msg;
}

// Surface JS errors into status to avoid "no response" feeling
window.addEventListener("error", (ev) => {
  try {
    setStatus(`运行错误：${ev?.message || ev}`);
  } catch {
    // ignore
  }
});

window.addEventListener("unhandledrejection", (ev) => {
  try {
    setStatus(`Promise 错误：${ev?.reason?.message || ev?.reason || ev}`);
  } catch {
    // ignore
  }
});

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function metersToDegLat(m) {
  // WGS84 近似：1°纬度 ≈ 111111m
  return m / 111111.0;
}

function metersToDegLon(m, latDeg) {
  const rad = (latDeg * Math.PI) / 180.0;
  const denom = 111111.0 * Math.cos(rad);
  if (!isFinite(denom) || denom === 0) return 0;
  return m / denom;
}

function haversineMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function meanOf(arr) {
  if (!arr.length) return NaN;
  let s = 0;
  for (const x of arr) s += x;
  return s / arr.length;
}

function formatLapTimeSeconds(sec) {
  if (!isFinite(sec)) return "-";
  const s = Math.max(0, sec);
  const mm = Math.floor(s / 60);
  const ss = s - mm * 60;
  const ssInt = Math.floor(ss);
  const ms = Math.round((ss - ssInt) * 1000);
  const pad2 = (n) => String(n).padStart(2, "0");
  const pad3 = (n) => String(n).padStart(3, "0");
  return `${pad2(mm)}:${pad2(ssInt)}.${pad3(ms)}`;
}

function simplifyByMinDistance(points, minMeters) {
  if (!minMeters || minMeters <= 0) return points;
  if (!points.length) return points;
  const out = [points[0]];
  let last = points[0];
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    const d = haversineMeters(last.lat, last.lon, p.lat, p.lon);
    if (d >= minMeters) {
      out.push(p);
      last = p;
    }
  }
  return out;
}

function splitIntoSegments(points) {
  // 尽量避免“跨段连线”：用 fragment_id + 时间间隔 + 空间跳变做断开
  const SEG_MAX_JUMP_M = 30; // 轨迹不太可能一帧跳 30m
  const SEG_MAX_DT_S = 2.0; // 2s 以上认为断点

  const segs = [];
  let cur = [];

  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const prev = i > 0 ? points[i - 1] : null;

    let cut = false;
    if (prev) {
      if (p.frag !== prev.frag) cut = true;
      if (isFinite(p.t) && isFinite(prev.t) && Math.abs(p.t - prev.t) > SEG_MAX_DT_S)
        cut = true;
      const d = haversineMeters(prev.lat, prev.lon, p.lat, p.lon);
      if (d > SEG_MAX_JUMP_M) cut = true;
    }

    if (cut && cur.length > 1) {
      segs.push(cur);
      cur = [];
    }

    cur.push(p);
  }

  if (cur.length > 1) segs.push(cur);
  return segs;
}

function computeMeanLat(points) {
  if (!points.length) return 0;
  let s = 0;
  for (const p of points) s += p.lat;
  return s / points.length;
}

function toLatLngSegments(segments, eastM, northM, refLat) {
  const dLat = metersToDegLat(northM);
  const dLon = metersToDegLon(eastM, refLat);
  return segments.map((seg) => seg.map((p) => [p.lat + dLat, p.lon + dLon]));
}

function safeFileBaseName(name) {
  return (name || "unknown.csv").split(/[\\/]/).pop();
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// --- Map ---
const canvasRenderer = L.canvas({ padding: 0.2 });
const map = L.map("map", {
  preferCanvas: true,
  zoomControl: true,
  renderer: canvasRenderer,
});
setupRightDrag(map, $("map"));

const tileLayer = L.tileLayer(TILE_URL, {
  minZoom: 12,
  maxZoom: 24,
  maxNativeZoom: 24,
  attribution: "Local tiles",
});
tileLayer.addTo(map);
L.control.scale({ imperial: false }).addTo(map);

map.setView([22.7123312, 113.8654811], 18);

// --- State ---
let simplifyMeters = 0.3;
let colorIdx = 0;
const datasets = [];

function nextColor() {
  const c = COLORS[colorIdx % COLORS.length];
  colorIdx += 1;
  return c;
}

function fitBoundsAll() {
  const layers = datasets.map((d) => d.group).filter(Boolean);
  if (!layers.length) return;
  const fg = L.featureGroup(layers);
  const b = fg.getBounds();
  if (b && b.isValid()) map.fitBounds(b.pad(0.08));
}

function removeDataset(id) {
  const idx = datasets.findIndex((d) => d.id === id);
  if (idx < 0) return;
  const d = datasets[idx];
  if (d.group) d.group.remove();
  datasets.splice(idx, 1);
  renderDatasetsPanel();
  setStatus(`已移除：${d.name}`);
}

function clearAll() {
  for (const d of datasets) {
    if (d.group) d.group.remove();
  }
  datasets.length = 0;
  colorIdx = 0;
  renderDatasetsPanel();
  setStatus("已清空。等待加载 CSV…");
}

function computeLapStats(pointsRaw) {
  const sats = [];
  const acc = [];
  const ets = [];
  const ts = [];

  for (const p of pointsRaw) {
    if (isFinite(p.sats)) sats.push(p.sats);
    if (isFinite(p.acc)) acc.push(p.acc);
    if (isFinite(p.et)) ets.push(p.et);
    if (isFinite(p.t)) ts.push(p.t);
  }

  let lapTimeS = NaN;
  if (ets.length >= 2) lapTimeS = Math.max(...ets) - Math.min(...ets);
  else if (ts.length >= 2) lapTimeS = Math.max(...ts) - Math.min(...ts);

  return {
    lapTimeS,
    avgSats: meanOf(sats),
    avgAcc: meanOf(acc),
  };
}

function groupPointsByLap(pointsRaw) {
  const m = new Map();
  for (const p of pointsRaw) {
    const lapNo = Number.isFinite(p.lap) && p.lap > 0 ? Math.trunc(p.lap) : 0;
    if (!m.has(lapNo)) m.set(lapNo, []);
    m.get(lapNo).push(p);
  }
  return Array.from(m.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([lapNo, pts]) => ({ lapNo, pointsRaw: pts }));
}

function updateLapLayer(d, lap) {
  const latlngSegs = toLatLngSegments(lap.segments, d.offsetEastM, d.offsetNorthM, d.refLat);
  lap.layer.setLatLngs(latlngSegs);
}

function updateDatasetLayers(d) {
  for (const lap of d.laps) updateLapLayer(d, lap);
}

function rebuildDataset(d) {
  d.refLat = computeMeanLat(d.pointsRaw);
  let total = 0;
  for (const lap of d.laps) {
    const simplified = simplifyByMinDistance(lap.pointsRaw, simplifyMeters);
    lap.points = simplified;
    lap.segments = splitIntoSegments(simplified);
    total += lap.points.length;
  }
  d.pointsCount = total;
}

const rebuildAllDebounced = debounce(() => {
  setStatus("正在重建轨迹（抽稀/分段）…");
  for (const d of datasets) {
    rebuildDataset(d);
    updateDatasetLayers(d);
  }
  setStatus(`完成：已更新 ${datasets.length} 组数据`);
}, 200);

function renderDatasetsPanel() {
  const el = $("datasets");
  el.innerHTML = "";

  for (const d of datasets) {
    const wrap = document.createElement("div");
    wrap.className = "dataset";

    const lapsRowsHtml = (d.laps || [])
      .map((lap) => {
        const lapLabel = lap.lapNo === 0 ? "未分圈" : String(lap.lapNo);
        const lapTime = formatLapTimeSeconds(lap.stats?.lapTimeS);
        const sats = isFinite(lap.stats?.avgSats) ? lap.stats.avgSats.toFixed(1) : "-";
        const acc = isFinite(lap.stats?.avgAcc) ? lap.stats.avgAcc.toFixed(2) : "-";
        return `
          <tr>
            <td class="tdCheck">
              <input type="checkbox" data-role="lap-visible" data-id="${d.id}" data-lap="${lap.lapNo}" ${
          lap.visible ? "checked" : ""
        } />
            </td>
            <td>${lapLabel}</td>
            <td>${lapTime}</td>
            <td>${sats}</td>
            <td>${acc}</td>
          </tr>
        `;
      })
      .join("");

    wrap.innerHTML = `
      <div class="dataset__top">
        <div class="swatch" style="background:${d.color}"></div>
        <div class="dataset__name">${d.name}</div>
        <label class="chip small">
          <input type="checkbox" data-role="visible" data-id="${d.id}" ${
            d.visible ? "checked" : ""
          } />
          显示
        </label>
      </div>
      <div class="dataset__meta">
        <span>点数：${(d.pointsCount || 0).toLocaleString()}</span>
        <span>圈数：${(d.laps || []).filter((x) => x.lapNo !== 0).length || ((d.laps || []).length ? 1 : 0)}</span>
      </div>
      <div class="dataset__controls">
        <div class="controlsRow">
          <button class="btnTiny" data-role="fit" data-id="${d.id}" type="button">定位</button>
          <button class="btnTiny btnTiny--danger" data-role="remove" data-id="${d.id}" type="button">移除</button>
        </div>

        <div class="small">偏移（米）：东 ${d.offsetEastM.toFixed(1)} / 北 ${d.offsetNorthM.toFixed(1)}</div>
        <div class="controlsRow">
          <button class="btnTiny" data-role="step" data-id="${d.id}" data-step="1" type="button">${
            d.stepM === 1 ? "步长 1m（当前）" : "步长 1m"
          }</button>
          <button class="btnTiny" data-role="step" data-id="${d.id}" data-step="0.1" type="button">${
            d.stepM === 0.1 ? "步长 0.1m（当前）" : "步长 0.1m"
          }</button>
        </div>
        <div class="offsetGrid">
          <button class="btnTiny" data-role="move" data-id="${d.id}" data-east="0" data-north="${d.stepM}" type="button">北 +</button>
          <button class="btnTiny" data-role="move" data-id="${d.id}" data-east="0" data-north="${-d.stepM}" type="button">南 -</button>
          <button class="btnTiny" data-role="move" data-id="${d.id}" data-east="${d.stepM}" data-north="0" type="button">东 +</button>
          <button class="btnTiny" data-role="move" data-id="${d.id}" data-east="${-d.stepM}" data-north="0" type="button">西 -</button>
        </div>
        <div class="controlsRow">
          <button class="btnTiny" data-role="reset" data-id="${d.id}" type="button">偏移归零</button>
          <button class="btnTiny" data-role="copy" data-id="${d.id}" type="button">复制偏移</button>
        </div>

        <div class="laps">
          <div class="small" style="margin-top:6px;">每圈统计（可单圈显示/隐藏）</div>
          <div class="lapsTableWrap">
            <table class="lapsTable">
              <thead>
                <tr>
                  <th></th>
                  <th>圈</th>
                  <th>圈速</th>
                  <th>卫星</th>
                  <th>精度(m)</th>
                </tr>
              </thead>
              <tbody>
                ${lapsRowsHtml || '<tr><td colspan="5" class="small">未检测到圈数据（lap_number）。</td></tr>'}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    el.appendChild(wrap);
  }
}

function getDatasetById(id) {
  return datasets.find((d) => d.id === id);
}

function applyMove(id, eastDelta, northDelta) {
  const d = getDatasetById(id);
  if (!d) return;
  d.offsetEastM += eastDelta;
  d.offsetNorthM += northDelta;
  updateDatasetLayers(d);
  renderDatasetsPanel();
}

async function copyOffsetToClipboard(id) {
  const d = getDatasetById(id);
  if (!d) return;
  const payload = JSON.stringify(
    { east_m: Number(d.offsetEastM.toFixed(3)), north_m: Number(d.offsetNorthM.toFixed(3)) },
    null,
    2
  );
  try {
    await navigator.clipboard.writeText(payload);
    setStatus(`已复制偏移：${d.name}`);
  } catch {
    setStatus("复制失败（可能因为浏览器权限）。");
  }
}

function parseRaceChronoCsvRows(rows) {
  // rows: Array<Array<any>>
  // 找到包含 latitude/longitude 的 header 行
  const headerIdx = rows.findIndex(
    (r) => Array.isArray(r) && r.includes("latitude") && (r.includes("longitude") || r.includes("lon"))
  );
  if (headerIdx < 0) throw new Error("未找到 latitude/longitude 表头。");

  const header = rows[headerIdx].map((x) => String(x || "").trim());
  const latIdx = header.indexOf("latitude");
  const lonIdx = header.indexOf("longitude") >= 0 ? header.indexOf("longitude") : header.indexOf("lon");
  const fragIdx = header.indexOf("fragment_id");
  const tIdx = header.indexOf("timestamp");

  if (latIdx < 0 || lonIdx < 0) throw new Error("表头中缺少 latitude/longitude。");

  const points = [];
  for (let i = headerIdx + 1; i < rows.length; i++) {
    const r = rows[i];
    if (!Array.isArray(r) || r.length <= Math.max(latIdx, lonIdx)) continue;

    const lat = Number(r[latIdx]);
    const lon = Number(r[lonIdx]);
    if (!isFinite(lat) || !isFinite(lon)) continue;

    const frag = fragIdx >= 0 ? Number(r[fragIdx]) : 0;
    const t = tIdx >= 0 ? Number(r[tIdx]) : NaN;
    points.push({ lat, lon, frag: isFinite(frag) ? frag : 0, t });
  }

  if (points.length < 2) throw new Error("有效轨迹点不足（<2）。");
  return points;
}

function parseRaceChronoCsvText(text) {
  // RaceChrono 前几行是元信息，真正数据区从包含 latitude/longitude 的 header 行开始
  const lines = String(text || "")
    .replace(/^\uFEFF/, "") // strip BOM
    .split(/\r?\n/);

  let headerIdx = -1;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    // 头通常类似：timestamp,fragment_id,...,latitude,longitude,...
    if (line.includes("latitude") && line.includes("longitude") && line.includes("timestamp")) {
      headerIdx = i;
      break;
    }
  }
  if (headerIdx < 0) throw new Error("未找到 latitude/longitude 表头行（CSV 格式不匹配）。");

  const header = lines[headerIdx].split(",").map((s) => s.trim().replace(/^"|"$/g, ""));
  const latIdx = header.indexOf("latitude");
  const lonIdx = header.indexOf("longitude") >= 0 ? header.indexOf("longitude") : header.indexOf("lon");
  const fragIdx = header.indexOf("fragment_id");
  const tIdx = header.indexOf("timestamp");
  const lapIdx = header.indexOf("lap_number");
  const etIdx = header.indexOf("elapsed_time");
  const satsIdx = header.indexOf("satellites");
  const accIdx = header.indexOf("accuracy");
  if (latIdx < 0 || lonIdx < 0) throw new Error("表头中缺少 latitude/longitude。");

  const points = [];
  // 数据行全部是数值（不带引号），可以安全按逗号 split
  for (let i = headerIdx + 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    // 快速过滤：RaceChrono 数据行一般以 unix time 开头
    if (!/^\d/.test(line)) continue;

    const cols = line.split(",");
    if (cols.length <= Math.max(latIdx, lonIdx)) continue;

    const lat = Number(cols[latIdx]);
    const lon = Number(cols[lonIdx]);
    if (!isFinite(lat) || !isFinite(lon)) continue;

    const frag = fragIdx >= 0 ? Number(cols[fragIdx]) : 0;
    const t = tIdx >= 0 ? Number(cols[tIdx]) : NaN;
    const lap = lapIdx >= 0 ? Number(cols[lapIdx]) : NaN;
    const et = etIdx >= 0 ? Number(cols[etIdx]) : NaN;
    const sats = satsIdx >= 0 ? Number(cols[satsIdx]) : NaN;
    const acc = accIdx >= 0 ? Number(cols[accIdx]) : NaN;
    points.push({
      lat,
      lon,
      frag: isFinite(frag) ? frag : 0,
      t,
      lap,
      et,
      sats,
      acc,
    });
  }

  if (points.length < 2) throw new Error("有效轨迹点不足（<2）。");
  return points;
}

async function loadCsvFile(file) {
  const name = safeFileBaseName(file.name);
  setStatus(`正在读取：${name} …`);
  const text = await file.text();
  setStatus(`正在解析：${name} …`);
  const pointsRaw = parseRaceChronoCsvText(text);
  return { name, pointsRaw };
}

async function addFiles(fileList) {
  const list = Array.isArray(fileList) ? fileList : Array.from(fileList || []);
  if (!list.length) return;

  setStatus(`准备加载 ${list.length} 个文件…`);

  for (const f of list) {
    try {
      const { name, pointsRaw } = await loadCsvFile(f);
      const d = {
        id: makeId(),
        name,
        color: nextColor(),
        visible: true,
        stepM: 1,
        offsetEastM: 0,
        offsetNorthM: 0,
        pointsRaw,
        refLat: 0,
        pointsCount: 0,
        laps: [],
        group: null,
      };

      d.laps = groupPointsByLap(pointsRaw).map((x) => {
        const stats = computeLapStats(x.pointsRaw);
        return {
          lapNo: x.lapNo,
          pointsRaw: x.pointsRaw,
          points: [],
          segments: [],
          stats,
          visible: true,
          layer: null,
        };
      });

      rebuildDataset(d);
      d.group = L.layerGroup().addTo(map);

      for (const lap of d.laps) {
        const latlngSegs = toLatLngSegments(lap.segments, 0, 0, d.refLat);
        lap.layer = L.polyline(latlngSegs, {
          color: d.color,
          weight: 2,
          opacity: 0.9,
          renderer: canvasRenderer,
          lineCap: "round",
          lineJoin: "round",
        });
        if (lap.visible) lap.layer.addTo(d.group);
      }

      datasets.push(d);
      renderDatasetsPanel();
      setStatus(`已加载：${d.name}（${(d.pointsCount || 0).toLocaleString()}点）`);

      // 首次加载自动定位
      if (datasets.length === 1) {
        const b = d.group.getBounds();
        if (b && b.isValid()) map.fitBounds(b.pad(0.08));
      }
    } catch (e) {
      setStatus(`加载失败：${f.name}（${e?.message || e}）`);
    }
  }

  if (datasets.length > 1) fitBoundsAll();
}

// --- UI wiring ---
$("fileInput").addEventListener("change", (e) => {
  const input = e.target;
  // 注意：某些浏览器里 input.files 是“活对象”，清空 value 会导致 files 也变空
  const picked = Array.from(input.files || []);
  setStatus(picked.length ? `已选择 ${picked.length} 个文件，开始加载…` : "未选择文件。");
  // 允许重复选同一文件：重置 value（放在复制之后）
  input.value = "";
  void addFiles(picked);
});

$("btnFitAll").addEventListener("click", () => fitBoundsAll());
$("btnClear").addEventListener("click", () => clearAll());

$("simplifyMeters").addEventListener("input", (e) => {
  simplifyMeters = Number(e.target.value);
  $("simplifyMetersText").textContent = `${simplifyMeters.toFixed(1)} m`;
  rebuildAllDebounced();
});

$("datasets").addEventListener("click", (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  const role = t.getAttribute("data-role");
  const id = t.getAttribute("data-id");
  if (!role || !id) return;

  if (role === "remove") {
    removeDataset(id);
    return;
  }
  if (role === "fit") {
    const d = getDatasetById(id);
    if (d?.group) {
      const b = d.group.getBounds();
      if (b && b.isValid()) map.fitBounds(b.pad(0.08));
    }
    return;
  }
  if (role === "reset") {
    const d = getDatasetById(id);
    if (!d) return;
    d.offsetEastM = 0;
    d.offsetNorthM = 0;
    updateDatasetLayers(d);
    renderDatasetsPanel();
    return;
  }
  if (role === "step") {
    const step = Number(t.getAttribute("data-step"));
    const d = getDatasetById(id);
    if (!d || ![1, 0.1].includes(step)) return;
    d.stepM = step;
    renderDatasetsPanel();
    return;
  }
  if (role === "move") {
    const east = Number(t.getAttribute("data-east") || 0);
    const north = Number(t.getAttribute("data-north") || 0);
    applyMove(id, east, north);
    return;
  }
  if (role === "copy") {
    void copyOffsetToClipboard(id);
  }
});

$("datasets").addEventListener("change", (e) => {
  const t = e.target;
  if (!(t instanceof HTMLInputElement)) return;
  const id = t.getAttribute("data-id");
  if (!id) return;
  const d = getDatasetById(id);
  if (!d) return;
  const role = t.getAttribute("data-role");

  if (role === "visible") {
    d.visible = !!t.checked;
    if (d.visible) d.group.addTo(map);
    else d.group.remove();
    return;
  }

  if (role === "lap-visible") {
    const lapNo = Number(t.getAttribute("data-lap"));
    const lap = (d.laps || []).find((x) => x.lapNo === lapNo);
    if (!lap) return;
    lap.visible = !!t.checked;
    if (lap.visible) lap.layer.addTo(d.group);
    else lap.layer.remove();
  }
});

// Initial UI
$("simplifyMetersText").textContent = `${simplifyMeters.toFixed(1)} m`;
renderDatasetsPanel();
setStatus("等待加载 CSV…（建议用 run_analyzer.py 启动本地服务器）");


