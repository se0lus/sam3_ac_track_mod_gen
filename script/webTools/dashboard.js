/* SAM3 赛道生成工具 — Dashboard */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let stages = [];
let activeStageId = null;
let currentFilePath = "";     // relative path within output dir
let pipelineStatus = {};
let manualStages = {};         // stage_id -> { enabled, has_data }
let sseSource = null;
let consoleExpanded = false;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const stagesContainer = $("stagesContainer");
const detailTabs = $("detailTabs");
const consoleOutput = $("consoleOutput");
const consolePanel = $("consolePanel");
const consoleBody = $("consoleBody");

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  setupTabs();
  setupConsole();
  setupButtons();
  await loadStages();
  await loadStatus();
  await loadManualStages();
  connectSSE();
  // Poll status every 5s
  setInterval(loadStatus, 5000);

  // Auto-refresh file tree when window regains focus
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && _currentFileDir) {
      refreshFiles();
    }
  });
}

// ---------------------------------------------------------------------------
// Load stages from server
// ---------------------------------------------------------------------------
async function loadStages() {
  try {
    const resp = await fetch("/api/pipeline/stages");
    stages = await resp.json();
  } catch {
    // Fallback: use inline definition
    stages = [];
  }
  renderStages();
}

async function loadStatus() {
  try {
    const resp = await fetch("/api/pipeline/status");
    pipelineStatus = await resp.json();
    updateStatusIndicators();
    $("btnStop").disabled = !pipelineStatus._running;
  } catch {
    // ignore
  }
}

async function loadManualStages() {
  try {
    const resp = await fetch("/api/pipeline/manual_stages");
    manualStages = await resp.json();
    updateManualToggles();
  } catch {
    // ignore
  }
}

async function toggleManualStage(stageId, enabled) {
  try {
    await fetch("/api/pipeline/manual_stages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage_id: stageId, enabled }),
    });
    await loadManualStages();
  } catch (e) {
    alert("切换失败: " + e.message);
  }
}

function updateManualToggles() {
  for (const [sid, info] of Object.entries(manualStages)) {
    const btn = document.getElementById(`toggle-${sid}`);
    if (btn) {
      btn.dataset.enabled = info.enabled ? "true" : "false";
      btn.textContent = info.enabled ? "禁用" : "启用";
      btn.className = info.enabled
        ? "stage-card__btn stage-card__btn--disable"
        : "stage-card__btn stage-card__btn--enable";
      // Update card visual
      const card = document.querySelector(`.stage-card[data-stage-id="${sid}"]`);
      if (card) {
        card.classList.toggle("stage-card--disabled", !info.enabled);
      }
    }
  }
}

async function resetManualStage(stageId) {
  if (!confirm(`确定要重置「${stageId}」的手动数据吗？这将删除所有手动编辑结果。`)) return;
  try {
    await fetch("/api/pipeline/manual_stages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage_id: stageId, enabled: false, reset: true }),
    });
    await loadManualStages();
  } catch (e) {
    alert("重置失败: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// Render stage cards
// ---------------------------------------------------------------------------
function renderStages() {
  stagesContainer.innerHTML = "";
  stages.forEach((stage, idx) => {
    if (idx > 0) {
      const conn = document.createElement("div");
      conn.className = "stage-connector" + (stage.type === "manual" ? " stage-connector--indent" : "");
      stagesContainer.appendChild(conn);
    }

    const card = document.createElement("div");
    card.className = "stage-card" + (stage.type === "manual" ? " stage-card--manual" : "");
    card.dataset.stageId = stage.id;
    card.addEventListener("click", () => selectStage(stage.id));

    // Indicator
    const indicator = document.createElement("div");
    indicator.className = "stage-card__indicator not_started";
    indicator.id = `indicator-${stage.id}`;

    // Info
    const info = document.createElement("div");
    info.className = "stage-card__info";

    const num = document.createElement("div");
    num.className = "stage-card__num";
    num.textContent = `[${stage.num}]`;

    const name = document.createElement("div");
    name.className = "stage-card__name";
    name.textContent = stage.name;

    info.appendChild(num);
    info.appendChild(name);

    // Actions
    const actions = document.createElement("div");
    actions.className = "stage-card__actions";

    if (stage.type === "auto") {
      const runBtn = document.createElement("button");
      runBtn.className = "stage-card__btn stage-card__btn--run";
      runBtn.textContent = "运行";
      runBtn.title = `运行 ${stage.name}`;
      runBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        runStage(stage.id);
      });
      actions.appendChild(runBtn);
    }

    if (stage.type === "manual") {
      // Enable / Disable text button
      const toggleBtn = document.createElement("button");
      toggleBtn.className = "stage-card__btn stage-card__btn--enable";
      toggleBtn.id = `toggle-${stage.id}`;
      toggleBtn.textContent = "启用";
      toggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const isEnabled = toggleBtn.dataset.enabled === "true";
        toggleManualStage(stage.id, !isEnabled);
      });
      actions.appendChild(toggleBtn);

      // Reset button
      const resetBtn = document.createElement("button");
      resetBtn.className = "stage-card__btn stage-card__btn--reset";
      resetBtn.textContent = "重置";
      resetBtn.title = `清除 ${stage.name} 手动数据`;
      resetBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetManualStage(stage.id);
      });
      actions.appendChild(resetBtn);
    }

    if (stage.editor) {
      const edLink = document.createElement("a");
      edLink.className = "stage-card__btn stage-card__btn--edit";
      edLink.href = stage.editor;
      edLink.target = "_blank";
      edLink.textContent = "编辑";
      edLink.title = `打开 ${stage.name} 编辑器`;
      edLink.addEventListener("click", (e) => e.stopPropagation());
      actions.appendChild(edLink);
    }

    card.appendChild(indicator);
    card.appendChild(info);
    card.appendChild(actions);

    stagesContainer.appendChild(card);
  });
}

function updateStatusIndicators() {
  stages.forEach((stage) => {
    const el = document.getElementById(`indicator-${stage.id}`);
    if (!el) return;
    const status = pipelineStatus[stage.id] || "not_started";
    el.className = `stage-card__indicator ${status}`;
  });
}

// ---------------------------------------------------------------------------
// Stage selection
// ---------------------------------------------------------------------------
function selectStage(stageId) {
  activeStageId = stageId;

  // Update card highlighting
  document.querySelectorAll(".stage-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.stageId === stageId);
  });

  const stage = stages.find((s) => s.id === stageId);
  if (!stage) return;

  // Show info tab
  showTab("info");

  if (stage.id === "prep") {
    showConfigPanel();
  } else {
    showStageInfo(stage);
  }

  // Load files for this stage
  if (stage.output_dir) {
    loadFiles(stage.output_dir);
  }

  // Prepare editor tab (lazy-load: iframe created only when tab is shown)
  if (stage.editor) {
    _pendingEditorUrl = stage.editor;
    _loadedEditorUrl = null;
    $("editorArea").innerHTML = '<div class="db-empty">切换到编辑器标签页加载</div>';
  } else {
    _pendingEditorUrl = null;
    _loadedEditorUrl = null;
    $("editorArea").innerHTML = '<div class="db-empty">此阶段无编辑器</div>';
  }
}

async function showStageInfo(stage) {
  const pane = $("tab-info");
  const status = pipelineStatus[stage.id] || "not_started";
  const statusLabels = {
    not_started: "未开始",
    pending: "等待中",
    running: "运行中",
    completed: "已完成",
    error: "出错",
  };
  const typeLabel = stage.type === "auto" ? "自动" : stage.type === "manual" ? "手动" : "配置";

  // Build per-stage config section
  let stageConfigHtml = "";
  let cfg = {};
  if (stage.id === "mask_full_map" || stage.id === "blender_polygons" || stage.id === "blender_automate" || stage.id === "model_export" || stage.id === "track_packaging" || stage.id === "convert_to_blender") {
    try {
      const resp = await fetch("/api/pipeline/config");
      cfg = await resp.json();
    } catch { cfg = {}; }
  }
  if (stage.id === "blender_polygons") {
    const curvesOn = !!cfg.s8_generate_curves;
    const gapFillOn = cfg.s8_gap_fill_enabled !== false;
    const gapThreshold = cfg.s8_gap_fill_threshold_m || 0.20;
    const defaultTag = cfg.s8_gap_fill_default_tag || "road2";
    const allFillTags = ["sand", "grass", "road2", "road", "kerb"];
    stageConfigHtml = `
      <div class="db-config db-config--stage">
        <h4>阶段配置</h4>
        <div class="config-field">
          <label>执行选项</label>
          <div class="s9-toggles" id="s8Toggles">
            <div class="s9-toggle" data-key="s8_gap_fill_enabled">
              <div>
                <div class="s9-toggle__label">自动空隙填充</div>
                <div class="s9-toggle__desc">消除可驾驶区域内的 mask 空隙，防止赛车掉落</div>
              </div>
              <div class="s9-toggle__track${gapFillOn ? " active" : ""}"><div class="s9-toggle__thumb"></div></div>
            </div>
            <div class="s9-toggle" data-key="s8_generate_curves">
              <div>
                <div class="s9-toggle__label">生成 2D 诊断曲线</div>
                <div class="s9-toggle__desc">创建 mask_curve2D_collection 用于轮廓可视化调试（较慢）</div>
              </div>
              <div class="s9-toggle__track${curvesOn ? " active" : ""}"><div class="s9-toggle__thumb"></div></div>
            </div>
          </div>
        </div>
        <div class="s9-level-row">
          <div class="config-field">
            <label>空隙阈值 (米)</label>
            <input type="number" id="s8GapThreshold" value="${gapThreshold}" min="0.05" max="2.0" step="0.05" />
          </div>
          <div class="config-field">
            <label>默认填充类型</label>
            <select id="s8DefaultTag">
              ${allFillTags.map(t => `<option value="${t}" ${t===defaultTag?"selected":""}>${t}</option>`).join("")}
            </select>
          </div>
        </div>
        <div class="config-actions">
          <button class="btn btn--primary" id="btnSaveStageConfig">保存</button>
        </div>
      </div>`;
  } else if (stage.id === "mask_full_map") {
    const model = cfg.inpaint_model || "gemini-2.5-flash-image";
    const roadOffset = cfg.road_mask_offset_px !== undefined ? cfg.road_mask_offset_px : -2;
    stageConfigHtml = `
      <div class="db-config db-config--stage">
        <h4>阶段配置</h4>
        <div class="config-field">
          <label>中心补洞模型 (Inpainting)</label>
          <select id="cfgInpaintModel">
            <option value="gemini-2.5-flash-image" ${model === "gemini-2.5-flash-image" ? "selected" : ""}>Gemini 2.5 Flash Image</option>
            <option value="gemini-3-pro-image-preview" ${model === "gemini-3-pro-image-preview" ? "selected" : ""}>Gemini 3 Pro Image Preview</option>
            <option value="disabled" ${model === "disabled" ? "selected" : ""}>跳过补洞 (Skip inpainting)</option>
          </select>
        </div>
        <div class="config-field">
          <label>路面 Mask 偏移 (px)</label>
          <input type="number" id="cfgRoadMaskOffset" value="${roadOffset}" min="-20" max="20" step="1" style="width:80px">
          <span class="config-hint">负值=内缩，正值=外扩，默认 -2（让精细 mask 补边缘）</span>
        </div>
        <div class="config-actions">
          <button class="btn btn--primary" id="btnSaveStageConfig">保存</button>
        </div>
      </div>`;
  } else if (stage.id === "convert_to_blender") {
    const roadGapClose = cfg.s5_road_gap_close_m !== undefined ? cfg.s5_road_gap_close_m : 0.20;
    const kerbNarrowWidth = cfg.s5_kerb_narrow_max_width_m !== undefined ? cfg.s5_kerb_narrow_max_width_m : 0.30;
    const kerbNarrowAdj = cfg.s5_kerb_narrow_adjacency_m !== undefined ? cfg.s5_kerb_narrow_adjacency_m : 0.20;
    stageConfigHtml = `
      <div class="db-config db-config--stage">
        <h4>阶段配置</h4>
        <div class="config-field">
          <label>路面空隙闭合 (Road Gap Close)</label>
          <div class="s9-level-row">
            <div class="config-field">
              <label>闭合宽度 (米)</label>
              <input type="number" id="s5RoadGapClose" value="${roadGapClose}" min="0" max="2.0" step="0.05" />
              <span class="config-hint">填充 road 边缘 ≤ 此宽度的缝隙 (0=禁用)</span>
            </div>
          </div>
        </div>
        <div class="config-field">
          <label>窄 Kerb 吸收为 Road (Narrow Kerb Absorption)</label>
          <div class="s9-level-row">
            <div class="config-field">
              <label>最大宽度 (米)</label>
              <input type="number" id="s5KerbNarrowWidth" value="${kerbNarrowWidth}" min="0" max="2.0" step="0.05" />
              <span class="config-hint">窄于此宽度的 kerb 突起被吸收为 road</span>
            </div>
            <div class="config-field">
              <label>邻接距离 (米)</label>
              <input type="number" id="s5KerbNarrowAdj" value="${kerbNarrowAdj}" min="0" max="2.0" step="0.05" />
              <span class="config-hint">仅吸收在此距离内紧邻 road 的窄 kerb</span>
            </div>
          </div>
        </div>
        <div class="config-actions">
          <button class="btn btn--primary" id="btnSaveStageConfig">保存</button>
        </div>
      </div>`;
  } else if (stage.id === "blender_automate") {
    const baseLevel = cfg.s9_base_level || 17;
    const targetLevel = cfg.s9_target_level || 22;
    const edgeSimplify = cfg.s9_edge_simplify || 0;
    const meshWeldDist = cfg.s9_mesh_weld_distance || 0.01;
    const meshDecimateRatio = cfg.s9_mesh_decimate_ratio || 0.5;
    const densityRoad = cfg.s9_density_road || 0.1;
    const densityKerb = cfg.s9_density_kerb || 0.1;
    const densityGrass = cfg.s9_density_grass || 2.0;
    const densitySand = cfg.s9_density_sand || 2.0;
    const densityRoad2 = cfg.s9_density_road2 || 2.0;
    const refineTags = cfg.s9_refine_tags || ["road"];
    const allTags = ["road", "grass", "sand", "kerb", "road2"];
    const tagPills = allTags.map(tag =>
      `<div class="s9-tag-pill${refineTags.includes(tag) ? " active" : ""}" data-tag="${tag}">${tag}</div>`
    ).join("");
    const toggles = [
      { key: "s9_import_walls",       label: "导入围墙",        desc: "从 Stage 7 导入虚拟碰撞墙", def: true },
      { key: "s9_import_game_objects", label: "导入游戏对象",    desc: "从 Stage 8 导入发车格、计时点等", def: true },
      { key: "s9_extract_surfaces",   label: "生成碰撞表面",    desc: "从 mask 多边形提取驾驶表面网格", def: true },
      { key: "s9_mesh_simplify",      label: "网格简化",        desc: "对 road/kerb 地形网格执行焊接 + Decimate 简化", def: false },
      { key: "s9_convert_textures",   label: "转换纹理",        desc: "解包纹理并转换为 PNG + BSDF 材质", def: true },
      { key: "s9_background",         label: "Blender 后台运行", desc: "以 --background 无界面模式执行", def: true },
    ];
    const toggleHtml = toggles.map(t => {
      const on = cfg[t.key] !== undefined ? cfg[t.key] : t.def;
      return `
        <div class="s9-toggle" data-key="${t.key}">
          <div>
            <div class="s9-toggle__label">${t.label}</div>
            <div class="s9-toggle__desc">${t.desc}</div>
          </div>
          <div class="s9-toggle__track${on ? " active" : ""}"><div class="s9-toggle__thumb"></div></div>
        </div>`;
    }).join("");
    stageConfigHtml = `
      <div class="db-config db-config--stage">
        <h4>阶段配置</h4>
        <div class="s9-level-row">
          <div class="config-field">
            <label>基础层级 (Base Level)</label>
            <input type="number" id="s9BaseLevel" value="${baseLevel}" min="10" max="25" />
          </div>
          <div class="config-field">
            <label>目标层级 (Target Level)</label>
            <input type="number" id="s9TargetLevel" value="${targetLevel}" min="15" max="25" />
          </div>
        </div>
        <div class="config-field">
          <label>碰撞表面采样密度 (米, 值越小精度越高)</label>
          <div class="s9-density-grid">
            <div class="s9-density-item">
              <span>Road</span>
              <input type="number" id="s9DensityRoad" value="${densityRoad}" min="0.05" max="5.0" step="0.1" />
            </div>
            <div class="s9-density-item">
              <span>Kerb</span>
              <input type="number" id="s9DensityKerb" value="${densityKerb}" min="0.05" max="5.0" step="0.1" />
            </div>
            <div class="s9-density-item">
              <span>Road2</span>
              <input type="number" id="s9DensityRoad2" value="${densityRoad2}" min="0.1" max="5.0" step="0.1" />
            </div>
            <div class="s9-density-item">
              <span>Grass</span>
              <input type="number" id="s9DensityGrass" value="${densityGrass}" min="0.1" max="5.0" step="0.1" />
            </div>
            <div class="s9-density-item">
              <span>Sand</span>
              <input type="number" id="s9DensitySand" value="${densitySand}" min="0.1" max="5.0" step="0.1" />
            </div>
          </div>
        </div>
        <div class="s9-level-row">
          <div class="config-field">
            <label>碰撞表面边缘简化 (米, 0=不简化)</label>
            <input type="number" id="s9EdgeSimplify" value="${edgeSimplify}" min="0" max="5.0" step="0.1" />
          </div>
        </div>
        <div class="s9-level-row">
          <div class="config-field">
            <label>网格焊接距离 (米)</label>
            <input type="number" id="s9MeshWeldDist" value="${meshWeldDist}" min="0.001" max="1.0" step="0.001" />
          </div>
          <div class="config-field">
            <label>网格简化率 (0-1, 越小面数越少)</label>
            <input type="number" id="s9MeshDecimateRatio" value="${meshDecimateRatio}" min="0.05" max="1.0" step="0.05" />
          </div>
        </div>
        <div class="config-field">
          <label>精细化 Mask 范围</label>
          <div class="s9-tag-pills" id="s9TagPills">${tagPills}</div>
        </div>
        <div class="config-field">
          <label>执行选项</label>
          <div class="s9-toggles" id="s9Toggles">${toggleHtml}</div>
        </div>
        <div class="config-actions">
          <button class="btn btn--primary" id="btnSaveStageConfig">保存</button>
        </div>
      </div>`;
  } else if (stage.id === "model_export") {
    const maxVerts = cfg.s10_max_vertices || 21000;
    const maxBatchMb = cfg.s10_max_batch_mb || 100;
    const fbxScale = cfg.s10_fbx_scale || 0.01;
    const ksAmbient = cfg.s10_ks_ambient ?? 0.5;
    const ksDiffuse = cfg.s10_ks_diffuse ?? 0.1;
    const ksEmissive = cfg.s10_ks_emissive ?? 0.1;
    const ksEditorExe = cfg.s10_kseditor_exe || "";
    stageConfigHtml = `
      <div class="db-config db-config--stage">
        <h4>导出参数</h4>
        <div class="s9-level-row">
          <div class="config-field">
            <label>单网格最大顶点数</label>
            <input type="number" id="s10MaxVertices" value="${maxVerts}" min="1000" max="100000" step="1000" />
          </div>
          <div class="config-field">
            <label>FBX 最大尺寸 (MB)</label>
            <input type="number" id="s10MaxBatchMb" value="${maxBatchMb}" min="10" max="500" step="10" />
          </div>
        </div>
        <div class="s9-level-row">
          <div class="config-field">
            <label>FBX 导出缩放</label>
            <input type="number" id="s10FbxScale" value="${fbxScale}" min="0.001" max="1.0" step="0.001" />
          </div>
        </div>
        <h4>材质参数 (可见物体)</h4>
        <div class="s9-level-row">
          <div class="config-field">
            <label>ksAmbient</label>
            <input type="number" id="s10KsAmbient" value="${ksAmbient}" min="0" max="1" step="0.05" />
          </div>
          <div class="config-field">
            <label>ksDiffuse</label>
            <input type="number" id="s10KsDiffuse" value="${ksDiffuse}" min="0" max="1" step="0.05" />
          </div>
          <div class="config-field">
            <label>ksEmissive</label>
            <input type="number" id="s10KsEmissive" value="${ksEmissive}" min="0" max="1" step="0.05" />
          </div>
        </div>
        <h4>KN5 转换</h4>
        <div class="config-field">
          <label>ksEditorAT 路径 (留空自动检测/下载)</label>
          <input type="text" id="s10KsEditorExe" value="${ksEditorExe}" placeholder="ac_toolbox/ac_tools_cmd/ksEditorAT.exe" />
        </div>
        <div class="config-actions">
          <button class="btn btn--primary" id="btnSaveStageConfig">保存</button>
        </div>
      </div>`;
  } else if (stage.id === "track_packaging") {
    const trackName = cfg.s11_track_name || "";
    const trackAuthor = cfg.s11_track_author || "";
    const trackCountry = cfg.s11_track_country || "";
    const trackCity = cfg.s11_track_city || "";
    const trackTags = (cfg.s11_track_tags || ["circuit", "original"]).join(", ");
    const trackYear = cfg.s11_track_year || new Date().getFullYear();
    const pitboxes = cfg.s11_pitboxes || 10;
    const trackUrl = cfg.s11_track_url || "";
    const trackDesc = cfg.track_description || "";
    const layoutDisplayNames = cfg.s11_layout_display_names || "";
    const llmDesc = cfg.s11_llm_description !== false;
    const llmPreview = cfg.s11_llm_preview !== false;
    stageConfigHtml = `
      <div class="db-config db-config--stage">
        <h4>赛道信息</h4>
        <div class="s9-level-row">
          <div class="config-field">
            <label>赛道名称</label>
            <input type="text" id="s11TrackName" value="${trackName}" placeholder="e.g. ShaJing Track" />
          </div>
          <div class="config-field">
            <label>作者</label>
            <input type="text" id="s11TrackAuthor" value="${trackAuthor}" placeholder="Author name" />
          </div>
        </div>
        <div class="s9-level-row">
          <div class="config-field">
            <label>国家</label>
            <input type="text" id="s11TrackCountry" value="${trackCountry}" placeholder="e.g. China" />
          </div>
          <div class="config-field">
            <label>城市</label>
            <input type="text" id="s11TrackCity" value="${trackCity}" placeholder="e.g. ShenZhen" />
          </div>
        </div>
        <div class="s9-level-row">
          <div class="config-field">
            <label>标签 (逗号分隔)</label>
            <input type="text" id="s11TrackTags" value="${trackTags}" placeholder="circuit, original" />
          </div>
          <div class="config-field">
            <label>年份</label>
            <input type="number" id="s11TrackYear" value="${trackYear}" min="2000" max="2099" />
          </div>
        </div>
        <div class="config-field">
          <label>布局显示名称 <span style="color:var(--muted);font-weight:normal">(格式: layout名:显示名，分号分隔)</span></label>
          <input type="text" id="s11LayoutDisplayNames" value="${layoutDisplayNames}" placeholder="layoutcw:Clockwise;layoutccw:Counter-Clockwise" />
        </div>
        <div class="s9-level-row">
          <div class="config-field">
            <label>Pit 数量</label>
            <input type="number" id="s11Pitboxes" value="${pitboxes}" min="1" max="100" />
          </div>
          <div class="config-field">
            <label>赛道 URL (可选)</label>
            <input type="text" id="s11TrackUrl" value="${trackUrl}" placeholder="https://..." />
          </div>
        </div>
        <div class="config-field" style="margin-top:8px">
          <label>赛道描述 <span style="color:var(--muted);font-weight:normal">(留空则由 LLM 生成或使用模板)</span></label>
          <textarea id="s11TrackDesc" rows="3" placeholder="e.g. A thrilling karting circuit located in..."
            style="width:100%;resize:vertical;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:13px">${trackDesc}</textarea>
        </div>
        <div class="config-field" style="margin-top:8px">
          <label>LLM 生成</label>
          <div class="s9-toggles" id="s11LlmToggles">
            <div class="s9-toggle" data-key="s11_llm_description">
              <div>
                <div class="s9-toggle__label">LLM 生成赛道描述</div>
                <div class="s9-toggle__desc">描述留空时调用 Gemini 自动撰写</div>
              </div>
              <div class="s9-toggle__track${llmDesc ? " active" : ""}"><div class="s9-toggle__thumb"></div></div>
            </div>
            <div class="s9-toggle" data-key="s11_llm_preview">
              <div>
                <div class="s9-toggle__label">LLM 生成预览图</div>
                <div class="s9-toggle__desc">无用户预览图时调用 Gemini 图像生成</div>
              </div>
              <div class="s9-toggle__track${llmPreview ? " active" : ""}"><div class="s9-toggle__thumb"></div></div>
            </div>
          </div>
        </div>
        <div class="config-actions">
          <button class="btn btn--primary" id="btnSaveStageConfig">保存</button>
        </div>
      </div>`;
  }

  pane.innerHTML = `
    <div class="db-stage-info">
      <h3>[${stage.num}] ${stage.name}<span class="badge badge--${stage.type}">${typeLabel}</span></h3>
      <p class="db-stage-info__desc">${stage.desc}</p>
      <div class="info-grid">
        <div class="info-row">
          <span class="info-label">状态</span>
          <span class="info-value info-value--status info-value--${status}">${statusLabels[status] || status}</span>
        </div>
        ${stage.output_dir ? `
        <div class="info-row">
          <span class="info-label">输出目录</span>
          <span class="info-value">output/${stage.output_dir}/</span>
        </div>` : ""}
        ${stage.editor ? `
        <div class="info-row">
          <span class="info-label">编辑器</span>
          <span class="info-value"><a href="${stage.editor}" target="_blank">${stage.editor}</a></span>
        </div>` : ""}
      </div>
      ${stageConfigHtml}
    </div>
  `;

  // Wire up per-stage config save
  if (stage.id === "blender_polygons") {
    // Wire toggle switches
    document.querySelectorAll("#s8Toggles .s9-toggle").forEach(row => {
      row.addEventListener("click", () => {
        row.querySelector(".s9-toggle__track").classList.toggle("active");
      });
    });
    $("btnSaveStageConfig").addEventListener("click", async () => {
      const updated = { ...cfg };
      // Collect all toggles
      document.querySelectorAll("#s8Toggles .s9-toggle").forEach(row => {
        const key = row.dataset.key;
        const on = row.querySelector(".s9-toggle__track").classList.contains("active");
        updated[key] = on;
      });
      // Gap-fill threshold + default tag
      updated.s8_gap_fill_threshold_m = parseFloat($("s8GapThreshold").value) || 0.20;
      updated.s8_gap_fill_default_tag = $("s8DefaultTag").value || "road2";
      try {
        await fetch("/api/pipeline/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updated),
        });
        $("btnSaveStageConfig").textContent = "已保存";
        setTimeout(() => { $("btnSaveStageConfig").textContent = "保存"; }, 1500);
      } catch (e) {
        alert("保存失败: " + e.message);
      }
    });
  } else if (stage.id === "mask_full_map") {
    $("btnSaveStageConfig").addEventListener("click", async () => {
      const val = $("cfgInpaintModel").value;
      const offsetVal = parseInt($("cfgRoadMaskOffset").value, 10) || -2;
      try {
        await fetch("/api/pipeline/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...cfg, inpaint_model: val, road_mask_offset_px: offsetVal }),
        });
        $("btnSaveStageConfig").textContent = "已保存";
        setTimeout(() => { $("btnSaveStageConfig").textContent = "保存"; }, 1500);
      } catch (e) {
        alert("保存失败: " + e.message);
      }
    });
  } else if (stage.id === "convert_to_blender") {
    $("btnSaveStageConfig").addEventListener("click", async () => {
      const updated = { ...cfg };
      updated.s5_road_gap_close_m = parseFloat($("s5RoadGapClose").value) || 0;
      updated.s5_kerb_narrow_max_width_m = parseFloat($("s5KerbNarrowWidth").value) || 0.30;
      updated.s5_kerb_narrow_adjacency_m = parseFloat($("s5KerbNarrowAdj").value) || 0.20;
      try {
        await fetch("/api/pipeline/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updated),
        });
        $("btnSaveStageConfig").textContent = "已保存";
        setTimeout(() => { $("btnSaveStageConfig").textContent = "保存"; }, 1500);
      } catch (e) {
        alert("保存失败: " + e.message);
      }
    });
  } else if (stage.id === "blender_automate") {
    // Wire tag pill toggles
    document.querySelectorAll(".s9-tag-pill").forEach(pill => {
      pill.addEventListener("click", () => pill.classList.toggle("active"));
    });
    // Wire toggle switches
    document.querySelectorAll(".s9-toggle").forEach(row => {
      row.addEventListener("click", () => {
        const track = row.querySelector(".s9-toggle__track");
        track.classList.toggle("active");
      });
    });
    // Save handler
    $("btnSaveStageConfig").addEventListener("click", async () => {
      const tags = Array.from(document.querySelectorAll(".s9-tag-pill.active")).map(p => p.dataset.tag);
      const updated = { ...cfg };
      updated.s9_base_level = parseInt($("s9BaseLevel").value) || 17;
      updated.s9_target_level = parseInt($("s9TargetLevel").value) || 22;
      updated.s9_edge_simplify = parseFloat($("s9EdgeSimplify").value) || 0;
      updated.s9_mesh_weld_distance = parseFloat($("s9MeshWeldDist").value) || 0.01;
      updated.s9_mesh_decimate_ratio = parseFloat($("s9MeshDecimateRatio").value) || 0.5;
      updated.s9_density_road = parseFloat($("s9DensityRoad").value) || 0.1;
      updated.s9_density_kerb = parseFloat($("s9DensityKerb").value) || 0.1;
      updated.s9_density_grass = parseFloat($("s9DensityGrass").value) || 2.0;
      updated.s9_density_sand = parseFloat($("s9DensitySand").value) || 2.0;
      updated.s9_density_road2 = parseFloat($("s9DensityRoad2").value) || 2.0;
      updated.s9_refine_tags = tags.length > 0 ? tags : ["road"];
      document.querySelectorAll(".s9-toggle").forEach(row => {
        const key = row.dataset.key;
        const on = row.querySelector(".s9-toggle__track").classList.contains("active");
        updated[key] = on;
      });
      try {
        await fetch("/api/pipeline/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updated),
        });
        $("btnSaveStageConfig").textContent = "已保存";
        setTimeout(() => { $("btnSaveStageConfig").textContent = "保存"; }, 1500);
      } catch (e) {
        alert("保存失败: " + e.message);
      }
    });
  } else if (stage.id === "model_export") {
    $("btnSaveStageConfig").addEventListener("click", async () => {
      const updated = { ...cfg };
      updated.s10_max_vertices = parseInt($("s10MaxVertices").value) || 21000;
      updated.s10_max_batch_mb = parseInt($("s10MaxBatchMb").value) || 100;
      updated.s10_fbx_scale = parseFloat($("s10FbxScale").value) || 0.01;
      updated.s10_ks_ambient = parseFloat($("s10KsAmbient").value) ?? 0.5;
      updated.s10_ks_diffuse = parseFloat($("s10KsDiffuse").value) ?? 0.1;
      updated.s10_ks_emissive = parseFloat($("s10KsEmissive").value) ?? 0.1;
      updated.s10_kseditor_exe = $("s10KsEditorExe").value.trim();
      try {
        await fetch("/api/pipeline/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updated),
        });
        $("btnSaveStageConfig").textContent = "已保存";
        setTimeout(() => { $("btnSaveStageConfig").textContent = "保存"; }, 1500);
      } catch (e) {
        alert("保存失败: " + e.message);
      }
    });
  } else if (stage.id === "track_packaging") {
    // Wire LLM toggle switches
    document.querySelectorAll("#s11LlmToggles .s9-toggle").forEach(row => {
      row.addEventListener("click", () => {
        row.querySelector(".s9-toggle__track").classList.toggle("active");
      });
    });
    $("btnSaveStageConfig").addEventListener("click", async () => {
      const updated = { ...cfg };
      updated.s11_track_name = $("s11TrackName").value.trim();
      updated.s11_track_author = $("s11TrackAuthor").value.trim();
      updated.s11_track_country = $("s11TrackCountry").value.trim();
      updated.s11_track_city = $("s11TrackCity").value.trim();
      const tagsStr = $("s11TrackTags").value.trim();
      updated.s11_track_tags = tagsStr ? tagsStr.split(",").map(t => t.trim()).filter(Boolean) : ["circuit", "original"];
      updated.s11_track_year = parseInt($("s11TrackYear").value) || new Date().getFullYear();
      updated.s11_pitboxes = parseInt($("s11Pitboxes").value) || 10;
      updated.s11_track_url = $("s11TrackUrl").value.trim();
      updated.s11_layout_display_names = $("s11LayoutDisplayNames").value.trim();
      updated.track_description = $("s11TrackDesc").value.trim();
      // LLM toggles
      document.querySelectorAll("#s11LlmToggles .s9-toggle").forEach(row => {
        const key = row.dataset.key;
        const on = row.querySelector(".s9-toggle__track").classList.contains("active");
        updated[key] = on;
      });
      try {
        await fetch("/api/pipeline/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updated),
        });
        $("btnSaveStageConfig").textContent = "已保存";
        setTimeout(() => { $("btnSaveStageConfig").textContent = "保存"; }, 1500);
      } catch (e) {
        alert("保存失败: " + e.message);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Config panel (Stage 0)
// ---------------------------------------------------------------------------
async function showConfigPanel() {
  const pane = $("tab-info");
  let cfg = {};
  try {
    const resp = await fetch("/api/pipeline/config");
    cfg = await resp.json();
  } catch {
    cfg = {};
  }
  const valid = cfg._valid || {};

  pane.innerHTML = `
    <div class="db-config">
      <h3>[0] 准备 — 流水线配置</h3>
      <p>配置输入文件路径和工具参数。保存后对所有阶段生效。</p>

      <div class="config-field">
        <label>GeoTIFF 路径
          <span class="config-status ${valid.geotiff_path ? "valid" : "invalid"}">
            ${valid.geotiff_path ? "\u2705" : "\u274C"}
          </span>
        </label>
        <input type="text" id="cfgGeotiff" value="${cfg.geotiff_path || ""}" />
      </div>

      <div class="config-field">
        <label>3D Tiles 目录
          <span class="config-status ${valid.tiles_dir ? "valid" : "invalid"}">
            ${valid.tiles_dir ? "\u2705" : "\u274C"}
          </span>
        </label>
        <input type="text" id="cfgTilesDir" value="${cfg.tiles_dir || ""}" />
      </div>

      <div class="config-field">
        <label>2D 底图瓦片目录
          <span class="config-status ${valid.map_tiles_dir ? "valid" : "invalid"}">
            ${valid.map_tiles_dir ? "\u2705" : "\u274C"}
          </span>
        </label>
        <input type="text" id="cfgMapTilesDir" value="${cfg.map_tiles_dir || ""}" />
      </div>

      <div class="config-field">
        <label>输出目录
          <span class="config-status ${valid.output_dir ? "valid" : "invalid"}">
            ${valid.output_dir ? "\u2705" : "\u274C"}
          </span>
        </label>
        <input type="text" id="cfgOutputDir" value="${cfg.output_dir || ""}" />
      </div>

      <div class="config-field">
        <label>Blender 可执行文件
          <span class="config-status ${valid.blender_exe ? "valid" : "invalid"}">
            ${valid.blender_exe ? "\u2705" : "\u274C"}
          </span>
        </label>
        <input type="text" id="cfgBlenderExe" value="${cfg.blender_exe || ""}" />
      </div>

      <div class="config-field">
        <label>赛道方向</label>
        <select id="cfgTrackDir">
          <option value="clockwise" ${cfg.track_direction === "clockwise" ? "selected" : ""}>顺时针 (clockwise)</option>
          <option value="counter-clockwise" ${cfg.track_direction === "counter-clockwise" ? "selected" : ""}>逆时针 (counter-clockwise)</option>
        </select>
      </div>

      <div class="config-field">
        <label>Gemini API Key</label>
        <input type="password" id="cfgGeminiKey" value="${cfg.gemini_api_key || ""}" />
      </div>

      <div class="s9-level-row">
        <div class="config-field">
          <label>并发线程数 (Stage 1, 3)</label>
          <input type="number" id="cfgMaxWorkers" value="${cfg.max_workers || 4}" min="1" max="16" step="1" />
        </div>
        <div></div>
      </div>

      <div class="config-actions">
        <button class="btn btn--primary" id="btnSaveConfig">保存配置</button>
      </div>
    </div>
  `;

  $("btnSaveConfig").addEventListener("click", saveConfig);
}

async function saveConfig() {
  // Load existing config to preserve fields managed by per-stage panels
  let existing = {};
  try {
    const resp = await fetch("/api/pipeline/config");
    existing = await resp.json();
  } catch { /* ignore */ }

  const cfg = {
    ...existing,
    geotiff_path: $("cfgGeotiff").value.trim(),
    tiles_dir: $("cfgTilesDir").value.trim(),
    map_tiles_dir: $("cfgMapTilesDir").value.trim(),
    output_dir: $("cfgOutputDir").value.trim(),
    blender_exe: $("cfgBlenderExe").value.trim(),
    track_direction: $("cfgTrackDir").value,
    gemini_api_key: $("cfgGeminiKey").value.trim(),
    max_workers: parseInt($("cfgMaxWorkers").value) || 4,
  };
  try {
    await fetch("/api/pipeline/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    // Refresh to show updated validity
    showConfigPanel();
  } catch (e) {
    alert("保存失败: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// File browser — tree + inline preview
// ---------------------------------------------------------------------------
let _activeTreeRow = null;
let _currentFileDir = null;
let _pendingEditorUrl = null;   // lazy-load: iframe created only when editor tab shown
let _loadedEditorUrl = null;    // currently loaded iframe URL

async function loadFiles(dirName) {
  _currentFileDir = dirName;
  const tree = $("fileTree");
  tree.innerHTML = "";

  // Header with refresh button
  const header = document.createElement("div");
  header.className = "db-filetree__header";
  header.innerHTML = `<span>${escapeHtml(dirName)}</span>`;
  const refreshBtn = document.createElement("button");
  refreshBtn.className = "btn btn--ghost btn--sm";
  refreshBtn.textContent = "刷新";
  refreshBtn.title = "刷新文件列表";
  refreshBtn.addEventListener("click", refreshFiles);
  header.appendChild(refreshBtn);
  tree.appendChild(header);

  const rootNode = await buildTreeNode(dirName, 0, true);
  if (rootNode) {
    tree.appendChild(rootNode);
  } else {
    const empty = document.createElement("div");
    empty.className = "db-empty";
    empty.textContent = "目录不存在或为空";
    tree.appendChild(empty);
  }
}

function refreshFiles() {
  if (_currentFileDir) {
    _activeTreeRow = null;
    loadFiles(_currentFileDir);
  }
}

async function buildTreeNode(dirPath, depth, autoExpand) {
  let items;
  try {
    const resp = await fetch(`/api/files/list?path=${encodeURIComponent(dirPath)}`);
    if (!resp.ok) return null;
    items = await resp.json();
  } catch {
    return null;
  }
  if (items.length === 0) return null;

  const frag = document.createDocumentFragment();

  // Directories first, then files
  const dirs = items.filter((i) => i.is_dir);
  const files = items.filter((i) => !i.is_dir);

  for (const dir of dirs) {
    const node = document.createElement("div");
    node.className = "tree-node";

    const row = document.createElement("div");
    row.className = "tree-row";
    row.style.paddingLeft = depth * 16 + 8 + "px";

    const arrow = document.createElement("span");
    arrow.className = "tree-arrow";
    arrow.textContent = "\u25B6";

    const icon = document.createElement("span");
    icon.className = "tree-icon";
    icon.textContent = "\uD83D\uDCC1";

    const label = document.createElement("span");
    label.className = "tree-label";
    label.textContent = dir.name;

    row.appendChild(arrow);
    row.appendChild(icon);
    row.appendChild(label);

    const childContainer = document.createElement("div");
    childContainer.className = "tree-children collapsed";

    let loaded = false;

    row.addEventListener("click", async () => {
      const isOpen = !childContainer.classList.contains("collapsed");
      if (isOpen) {
        childContainer.classList.add("collapsed");
        arrow.classList.remove("expanded");
      } else {
        if (!loaded) {
          loaded = true;
          const childPath = dirPath + "/" + dir.name;
          const children = await buildTreeNode(childPath, depth + 1, false);
          if (children) {
            childContainer.appendChild(children);
          } else {
            const empty = document.createElement("div");
            empty.className = "tree-row";
            empty.style.paddingLeft = (depth + 1) * 16 + 8 + "px";
            empty.innerHTML = '<span class="tree-label" style="color:#555;">(空)</span>';
            childContainer.appendChild(empty);
          }
        }
        childContainer.classList.remove("collapsed");
        arrow.classList.add("expanded");
      }
    });

    node.appendChild(row);
    node.appendChild(childContainer);
    frag.appendChild(node);

    // Auto-expand first level
    if (autoExpand) {
      row.click();
    }
  }

  const maxFiles = 100;
  const visibleFiles = files.slice(0, maxFiles);

  for (const file of visibleFiles) {
    const row = document.createElement("div");
    row.className = "tree-row";
    row.style.paddingLeft = depth * 16 + 8 + "px";

    // Spacer matching arrow width
    const spacer = document.createElement("span");
    spacer.className = "tree-arrow";
    spacer.textContent = "";

    const icon = document.createElement("span");
    icon.className = "tree-icon";
    icon.textContent = fileIcon(file.name);

    const label = document.createElement("span");
    label.className = "tree-label";
    label.textContent = file.name;

    const size = document.createElement("span");
    size.className = "tree-size";
    size.textContent = formatSize(file.size);

    row.appendChild(spacer);
    row.appendChild(icon);
    row.appendChild(label);
    row.appendChild(size);

    const relPath = dirPath + "/" + file.name;
    row.addEventListener("click", () => {
      if (_activeTreeRow) _activeTreeRow.classList.remove("active");
      row.classList.add("active");
      _activeTreeRow = row;
      previewFile(relPath);
    });

    frag.appendChild(row);
  }

  if (files.length > maxFiles) {
    const moreRow = document.createElement("div");
    moreRow.className = "tree-row";
    moreRow.style.paddingLeft = depth * 16 + 8 + "px";
    moreRow.innerHTML = `<span class="tree-arrow"></span><span class="tree-label" style="color:#667;font-style:italic;">... 还有 ${files.length - maxFiles} 个文件</span>`;
    frag.appendChild(moreRow);
  }

  return frag;
}

function previewFile(relPath) {
  currentFilePath = relPath;

  const previewArea = $("previewArea");
  const ext = relPath.split(".").pop().toLowerCase();
  const url = `/api/files/preview?path=${encodeURIComponent(relPath)}`;
  const fileName = relPath.split("/").pop();

  if (["png", "jpg", "jpeg", "gif", "svg"].includes(ext)) {
    previewArea.innerHTML = `
      <div class="db-preview__filename">${escapeHtml(fileName)}</div>
      <img src="${url}" alt="${escapeHtml(relPath)}" />`;
  } else if (["json", "txt", "csv", "ini"].includes(ext)) {
    previewArea.innerHTML = `<div class="db-preview__filename">${escapeHtml(fileName)}</div><pre>加载中...</pre>`;
    fetch(url)
      .then((r) => r.text())
      .then((text) => {
        if (text.length > 100000) text = text.substring(0, 100000) + "\n... (truncated)";
        previewArea.innerHTML = `
          <div class="db-preview__filename">${escapeHtml(fileName)}</div>
          <pre>${escapeHtml(text)}</pre>`;
      })
      .catch(() => {
        previewArea.innerHTML = '<div class="db-empty">预览失败</div>';
      });
  } else {
    previewArea.innerHTML = `<div class="db-empty">不支持预览此文件类型 (.${ext})<br>
      <a href="${url}" download style="color: #60a5fa;">下载文件</a></div>`;
  }
}

// ---------------------------------------------------------------------------
// Pipeline control
// ---------------------------------------------------------------------------
async function runStage(stageId) {
  try {
    selectStage(stageId);
    expandConsole();
    const resp = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stages: [stageId] }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      appendLog(`[ERROR] Server ${resp.status}: ${text}`);
      return;
    }
    $("btnStop").disabled = false;
    loadStatus();
  } catch (e) {
    appendLog(`[ERROR] ${e.message}`);
  }
}

async function runAllAuto() {
  const autoStages = stages.filter((s) => s.type === "auto").map((s) => s.id);
  try {
    expandConsole();
    const resp = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stages: autoStages }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      appendLog(`[ERROR] Server ${resp.status}: ${text}`);
      return;
    }
    $("btnStop").disabled = false;
    loadStatus();
  } catch (e) {
    appendLog(`[ERROR] ${e.message}`);
  }
}

async function stopPipeline() {
  try {
    await fetch("/api/pipeline/stop", { method: "POST" });
    $("btnStop").disabled = true;
    loadStatus();
  } catch (e) {
    appendLog(`[ERROR] ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------
function connectSSE() {
  if (sseSource) sseSource.close();
  sseSource = new EventSource("/api/sse/pipeline");

  sseSource.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "log") {
        appendLog(msg.data);
      } else if (msg.type === "stage_start") {
        loadStatus();
        appendLog(`\n=== Stage ${msg.data.stage} started ===`);
      } else if (msg.type === "stage_complete") {
        loadStatus();
        refreshFiles();
      } else if (msg.type === "pipeline_done") {
        loadStatus();
        refreshFiles();
        const rc = msg.data.returncode;
        appendLog(rc === 0 ? "\n=== Pipeline completed ===" : `\n=== Pipeline failed (rc=${rc}) ===`);
      } else if (msg.type === "pipeline_stop") {
        appendLog("\n=== Pipeline stopped ===");
        loadStatus();
        refreshFiles();
      }
    } catch {
      // ignore parse errors
    }
  };

  sseSource.onerror = () => {
    // Reconnect after 3s
    sseSource.close();
    setTimeout(connectSSE, 3000);
  };
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------
let _logBuffer = [];
let _logRafPending = false;

function appendLog(text) {
  _logBuffer.push(text);
  if (!_logRafPending) {
    _logRafPending = true;
    requestAnimationFrame(flushLog);
  }
}

function flushLog() {
  _logRafPending = false;
  if (_logBuffer.length === 0) return;
  // Batch append — avoid O(n^2) textContent rewrite
  const chunk = _logBuffer.join("\n") + "\n";
  _logBuffer.length = 0;
  consoleOutput.appendChild(document.createTextNode(chunk));
  // Cap total lines to prevent DOM bloat
  const maxLen = 50000;
  if (consoleOutput.textContent.length > maxLen) {
    consoleOutput.textContent = consoleOutput.textContent.slice(-maxLen);
  }
  consoleBody.scrollTop = consoleBody.scrollHeight;
}

function setupConsole() {
  $("consoleToggle").addEventListener("click", toggleConsole);
  $("btnConsoleClear").addEventListener("click", (e) => {
    e.stopPropagation();
    consoleOutput.textContent = "";
  });
}

function toggleConsole() {
  consoleExpanded = !consoleExpanded;
  consolePanel.classList.toggle("expanded", consoleExpanded);
  $("consoleToggle").querySelector("span").textContent = consoleExpanded
    ? "\u25BC 控制台"
    : "\u25B6 控制台";
}

function expandConsole() {
  if (!consoleExpanded) toggleConsole();
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function setupTabs() {
  detailTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".db-tab");
    if (!btn) return;
    showTab(btn.dataset.tab);
  });
}

function showTab(tabName) {
  detailTabs.querySelectorAll(".db-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".db-tab-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === `tab-${tabName}`);
  });

  // Lazy-load editor iframe only when the tab becomes visible,
  // so Leaflet initialises with correct container dimensions.
  if (tabName === "editor" && _pendingEditorUrl && _pendingEditorUrl !== _loadedEditorUrl) {
    _loadedEditorUrl = _pendingEditorUrl;
    const area = $("editorArea");
    area.innerHTML = `<iframe src="${_pendingEditorUrl}?_t=${Date.now()}" title="编辑器"></iframe>`;
  }
}

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------
function setupButtons() {
  $("btnRunAll").addEventListener("click", runAllAuto);
  $("btnStop").addEventListener("click", stopPipeline);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function fileIcon(name) {
  const ext = name.split(".").pop().toLowerCase();
  const icons = {
    json: "\uD83D\uDCCB",
    png: "\uD83D\uDDBC",
    jpg: "\uD83D\uDDBC",
    jpeg: "\uD83D\uDDBC",
    csv: "\uD83D\uDCCA",
    blend: "\uD83C\uDFAE",
    tif: "\uD83C\uDF0D",
    tiff: "\uD83C\uDF0D",
    py: "\uD83D\uDC0D",
    ini: "\u2699\uFE0F",
    glb: "\uD83D\uDFE9",
  };
  return icons[ext] || "\uD83D\uDCC4";
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", init);
