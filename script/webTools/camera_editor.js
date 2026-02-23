/**
 * camera_editor.js — Cesium 3D Camera Editor for Assetto Corsa track cameras.
 *
 * Loads cameras.ini (via JSON API), renders camera markers on a Cesium 3D Tiles
 * scene, allows interactive editing (select, drag, property panel), and saves
 * back to the server.
 */
"use strict";

/* =========================================================================
   State
   ========================================================================= */
let viewer = null;               // Cesium.Viewer
let tileset = null;              // Cesium.Cesium3DTileset
let acTransform = null;          // ACToGeoTransform instance

let layouts = [];                // [{name, short, ...}]
let currentLayout = "";          // current layout short name
let camerasData = null;          // { header: {...}, cameras: [...] }
let centerlineData = null;       // { centerline: [[x,z], ...] }

let selectedIndex = -1;          // index into camerasData.cameras
let isDirty = false;

// Undo / redo
let history = [];
let historyIndex = -1;

// Cesium entities
let cameraEntities = [];         // Cesium.Entity[] for camera markers
let centerlineEntity = null;
let frustumEntities = [];        // For selected camera frustum visualization
let inOutEntity = null;          // Highlighted centerline segment for IN/OUT

// Drag state
let dragActive = false;
let dragEntity = null;

// Gizmo state (Feature 1)
let gizmoEntities = null;   // { x: Entity, y: Entity, z: Entity }
let dragAxis = null;         // 'x'|'y'|'z' or null
let dragStartScreenPos = null;
let dragStartACPos = null;

// Track position preview (Feature 2)
let trackPosMarker = null;   // Entity for track position marker

// Frustum projection (Feature 3)
let projectionEntities = [];

// FOV preview (Feature 4)
let previewFOV = null;       // null = use camera's MAX_FOV

/* =========================================================================
   DOM refs
   ========================================================================= */
const $layoutSelect = document.getElementById("layoutSelect");
const $cameraList = document.getElementById("cameraList");
const $propsSection = document.getElementById("propsSection");
const $btnSave = document.getElementById("btnSave");
const $btnUndo = document.getElementById("btnUndo");
const $btnRedo = document.getElementById("btnRedo");
const $dirtyFlag = document.getElementById("dirtyFlag");
const $btnAdd = document.getElementById("btnAddCamera");
const $btnDelete = document.getElementById("btnDeleteCamera");
const $status = document.getElementById("status");

// Property inputs
const $propName = document.getElementById("propName");
const $propPosX = document.getElementById("propPosX");
const $propPosY = document.getElementById("propPosY");
const $propPosZ = document.getElementById("propPosZ");
const $propFwdX = document.getElementById("propFwdX");
const $propFwdY = document.getElementById("propFwdY");
const $propFwdZ = document.getElementById("propFwdZ");
const $propFovMin = document.getElementById("propFovMin");
const $propFovMax = document.getElementById("propFovMax");
const $propInPoint = document.getElementById("propInPoint");
const $propOutPoint = document.getElementById("propOutPoint");
const $propIsFixed = document.getElementById("propIsFixed");

// Slider controls
const $trackPosSlider = document.getElementById("trackPosSlider");
const $trackPosLabel = document.getElementById("trackPosLabel");
const $fovPreviewSlider = document.getElementById("fovPreviewSlider");
const $fovPreviewLabel = document.getElementById("fovPreviewLabel");

/* =========================================================================
   AC <-> Geo coordinate transform
   ========================================================================= */

class ACToGeoTransform {
  /**
   * Build from tileset.json root.transform (column-major 4x4 flat array).
   */
  constructor(transform) {
    // Column-major 4x4 → extract rotation columns and translation
    // col0 = [0,1,2], col1 = [4,5,6], col2 = [8,9,10], col3(T) = [12,13,14]
    this.rotCol0 = new Cesium.Cartesian3(transform[0], transform[1], transform[2]);
    this.rotCol1 = new Cesium.Cartesian3(transform[4], transform[5], transform[6]);
    this.rotCol2 = new Cesium.Cartesian3(transform[8], transform[9], transform[10]);
    this.origin = new Cesium.Cartesian3(transform[12], transform[13], transform[14]);

    // Build rotation matrix (columns are the basis vectors)
    this.rotation = new Cesium.Matrix3(
      transform[0], transform[4], transform[8],
      transform[1], transform[5], transform[9],
      transform[2], transform[6], transform[10]
    );
    // Inverse rotation = transpose (orthonormal)
    this.rotationInv = Cesium.Matrix3.transpose(this.rotation, new Cesium.Matrix3());
  }

  /**
   * AC coords (x_ac, y_ac, z_ac) → Cesium Cartesian3 (ECEF).
   *
   * AC coordinate system: X right, Y up (negative in INI = height), Z forward.
   * Blender coordinate system: X right, Y forward, Z up.
   *
   * The tileset root.transform maps Blender local coords to ECEF.
   * AC to Blender: x_b = x_ac, y_b = z_ac, z_b = -y_ac
   */
  acToCartesian3(x_ac, y_ac, z_ac) {
    // AC → Blender local
    const x_b = x_ac;
    const y_b = z_ac;
    const z_b = -y_ac;

    // Local → ECEF: origin + rotation * local
    const local = new Cesium.Cartesian3(x_b, y_b, z_b);
    const rotated = Cesium.Matrix3.multiplyByVector(this.rotation, local, new Cesium.Cartesian3());
    return Cesium.Cartesian3.add(this.origin, rotated, new Cesium.Cartesian3());
  }

  /**
   * Cesium Cartesian3 (ECEF) → AC coords [x_ac, y_ac, z_ac].
   */
  cartesian3ToAC(cartesian3) {
    // ECEF → local: rotationInv * (pos - origin)
    const diff = Cesium.Cartesian3.subtract(cartesian3, this.origin, new Cesium.Cartesian3());
    const local = Cesium.Matrix3.multiplyByVector(this.rotationInv, diff, new Cesium.Cartesian3());

    // Blender → AC: x_ac = x_b, y_ac = -z_b, z_ac = y_b
    return [local.x, -local.z, local.y];
  }

  /**
   * Transform a direction vector from AC to ECEF (rotation only, no translation).
   */
  acDirectionToECEF(dx_ac, dy_ac, dz_ac) {
    const dx_b = dx_ac;
    const dy_b = dz_ac;
    const dz_b = -dy_ac;
    const local = new Cesium.Cartesian3(dx_b, dy_b, dz_b);
    return Cesium.Matrix3.multiplyByVector(this.rotation, local, new Cesium.Cartesian3());
  }
}

/* =========================================================================
   Initialize
   ========================================================================= */

async function init() {
  setStatus("Loading layouts...");
  try {
    layouts = await loadLayouts();
  } catch (e) {
    setStatus("Failed to load layouts: " + e.message);
    return;
  }

  if (layouts.length === 0) {
    setStatus("No layouts found. Run Stage 2a first.");
    return;
  }

  // Populate layout selector
  layouts.forEach((l) => {
    const opt = document.createElement("option");
    opt.value = l.short;
    opt.textContent = l.name;
    $layoutSelect.appendChild(opt);
  });
  $layoutSelect.addEventListener("change", () => onLayoutChange($layoutSelect.value));

  // Init Cesium viewer
  setStatus("Initializing 3D viewer...");
  await initCesium();

  // Load first layout
  await onLayoutChange(layouts[0].short);

  // Event listeners
  $btnSave.addEventListener("click", save);
  $btnUndo.addEventListener("click", undo);
  $btnRedo.addEventListener("click", redo);
  $btnAdd.addEventListener("click", addCamera);
  $btnDelete.addEventListener("click", deleteCamera);

  // Property change listeners
  const propInputs = [
    $propName, $propPosX, $propPosY, $propPosZ,
    $propFwdX, $propFwdY, $propFwdZ,
    $propFovMin, $propFovMax, $propInPoint, $propOutPoint,
    $propIsFixed,
  ];
  propInputs.forEach((el) => el.addEventListener("change", onPropChange));

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      save();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
      e.preventDefault();
      undo();
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === "y" || (e.key === "z" && e.shiftKey))) {
      e.preventDefault();
      redo();
    }
    if (e.key === "Delete" && selectedIndex >= 0) {
      e.preventDefault();
      deleteCamera();
    }
    if (e.key === "Escape") {
      selectCamera(-1);
    }
  });

  // FOV preview slider (Feature 4)
  $fovPreviewSlider.addEventListener("input", () => {
    previewFOV = parseFloat($fovPreviewSlider.value);
    $fovPreviewLabel.textContent = previewFOV + "°";
    updateFrustum();
  });

  // Track position slider (Feature 2)
  // "input" fires continuously during drag — live preview
  $trackPosSlider.addEventListener("input", () => {
    const t = parseFloat($trackPosSlider.value);
    $trackPosLabel.textContent = t.toFixed(2);
    updateTrackPositionPreview(t);
  });
  // "change" fires on mouseup — commit to history
  $trackPosSlider.addEventListener("change", () => {
    if (selectedIndex >= 0) {
      pushHistory();
      markDirty();
    }
  });
}

/* =========================================================================
   Cesium initialization
   ========================================================================= */

async function initCesium() {
  // No Ion token needed — we use local 3D Tiles only
  Cesium.Ion.defaultAccessToken = undefined;

  viewer = new Cesium.Viewer("cesiumContainer", {
    animation: false,
    timeline: false,
    homeButton: false,
    sceneModePicker: false,
    baseLayerPicker: false,
    navigationHelpButton: false,
    geocoder: false,
    fullscreenButton: false,
    infoBox: false,
    selectionIndicator: false,
    // No base imagery — just dark sky
    baseLayer: false,
    skyBox: false,
    skyAtmosphere: false,
    contextOptions: {
      webgl: { alpha: true },
    },
  });

  // Dark background
  viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#0b0f17");
  viewer.scene.globe.show = false;

  // Load 3D Tiles
  try {
    const info = await fetch("/api/tiles_dir_info").then((r) => r.json());
    if (info.has_tileset && info.tileset_url) {
      tileset = await Cesium.Cesium3DTileset.fromUrl(info.tileset_url);
      viewer.scene.primitives.add(tileset);

      // Extract transform for coordinate conversion
      await tileset.readyPromise || tileset.ready;
      const t = tileset.root.transform;

      // Check if root.transform is identity (tileset has no explicit transform)
      const isIdentity = Cesium.Matrix4.equalsEpsilon(
        t, Cesium.Matrix4.IDENTITY, Cesium.Math.EPSILON10
      );

      if (isIdentity && info.geo_corners) {
        // No root.transform — geometry is already in ECEF.
        // AC coordinates are in pixel space: origin at NW corner (top-left),
        // X increases east, Z increases south, Y is height in meters.
        //
        // Use actual corner coordinates for exact affine transform
        // (accounts for UTM grid convergence / image rotation).
        const gc = info.geo_corners;
        const tlEcef = Cesium.Cartesian3.fromDegrees(gc.tl[1], gc.tl[0], 0);
        const trEcef = Cesium.Cartesian3.fromDegrees(gc.tr[1], gc.tr[0], 0);
        const blEcef = Cesium.Cartesian3.fromDegrees(gc.bl[1], gc.bl[0], 0);

        // Per-pixel ECEF vectors along image X and Y axes
        const vecX = Cesium.Cartesian3.subtract(trEcef, tlEcef, new Cesium.Cartesian3());
        Cesium.Cartesian3.divideByScalar(vecX, gc.img_w, vecX);
        const vecY = Cesium.Cartesian3.subtract(blEcef, tlEcef, new Cesium.Cartesian3());
        Cesium.Cartesian3.divideByScalar(vecY, gc.img_h, vecY);

        // Up vector at TL for height (1 unit = 1 meter)
        const enuAtTL = Cesium.Transforms.eastNorthUpToFixedFrame(tlEcef);
        const upVec = new Cesium.Cartesian3(enuAtTL[8], enuAtTL[9], enuAtTL[10]);

        // Build 4x4 column-major transform:
        //   col0 = vecX  (x_ac pixels → ECEF along image X)
        //   col1 = vecY  (z_ac pixels → ECEF along image Y / south)
        //   col2 = upVec (-y_ac meters → ECEF upward)
        //   col3 = tlEcef (origin = NW corner)
        const m = new Float64Array(16);
        m[0] = vecX.x; m[1] = vecX.y; m[2] = vecX.z; m[3] = 0;
        m[4] = vecY.x; m[5] = vecY.y; m[6] = vecY.z; m[7] = 0;
        m[8] = upVec.x; m[9] = upVec.y; m[10] = upVec.z; m[11] = 0;
        m[12] = tlEcef.x; m[13] = tlEcef.y; m[14] = tlEcef.z; m[15] = 1;

        acTransform = new ACToGeoTransform(m);
        console.log("Using corner-based affine transform (" + gc.img_w + "x" + gc.img_h + " px)");
      } else if (!isIdentity) {
        // Has root.transform — use it directly (tileset_local mode)
        const flat = [
          t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
          t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15],
        ];
        acTransform = new ACToGeoTransform(flat);
      } else {
        // Fallback: identity transform from BV center (markers may be offset)
        const center = tileset.boundingSphere.center;
        const enuMatrix = Cesium.Transforms.eastNorthUpToFixedFrame(center);
        const flat = [
          enuMatrix[0], enuMatrix[1], enuMatrix[2], enuMatrix[3],
          enuMatrix[4], enuMatrix[5], enuMatrix[6], enuMatrix[7],
          enuMatrix[8], enuMatrix[9], enuMatrix[10], enuMatrix[11],
          enuMatrix[12], enuMatrix[13], enuMatrix[14], enuMatrix[15],
        ];
        acTransform = new ACToGeoTransform(flat);
        console.warn("Using fallback ENU from BV center — markers may be offset");
      }

      // Fly to tileset
      viewer.zoomTo(tileset);
      setStatus("3D Tiles loaded");
    } else {
      setStatus("No 3D Tiles found — camera markers will not be positioned accurately");
    }
  } catch (e) {
    console.error("Failed to load 3D Tiles:", e);
    setStatus("3D Tiles load failed: " + e.message);
  }

  // Set up click/drag handlers
  setupInteraction();
}

/* =========================================================================
   Interaction (click to select, drag to move)
   ========================================================================= */

function setupInteraction() {
  const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

  // Left click — select camera (only if not a gizmo axis)
  handler.setInputAction((click) => {
    const picked = viewer.scene.pick(click.position);
    if (Cesium.defined(picked) && picked.id) {
      if (picked.id._gizmoAxis) return; // ignore gizmo clicks for selection
      if (picked.id._camIndex !== undefined) {
        selectCamera(picked.id._camIndex);
        return;
      }
    }
    // Clicked empty space
    selectCamera(-1);
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

  // Left down — start drag (gizmo axis or camera marker)
  handler.setInputAction((click) => {
    const picked = viewer.scene.pick(click.position);
    if (!Cesium.defined(picked) || !picked.id) return;

    // Check if picked a gizmo axis
    if (picked.id._gizmoAxis && selectedIndex >= 0) {
      dragAxis = picked.id._gizmoAxis;
      dragStartScreenPos = Cesium.Cartesian2.clone(click.position);
      const cam = camerasData.cameras[selectedIndex];
      dragStartACPos = [...(cam.POSITION || [0, 0, 0])];
      dragActive = true;
      viewer.scene.screenSpaceCameraController.enableRotate = false;
      viewer.scene.screenSpaceCameraController.enableTranslate = false;
      return;
    }

    // Check if picked a camera marker
    if (picked.id._camIndex !== undefined) {
      dragActive = true;
      dragEntity = picked.id;
      selectCamera(picked.id._camIndex);
      // For camera markers, set up gizmo axis drag on next move
      // (or just keep old XZ-surface drag as fallback)
      viewer.scene.screenSpaceCameraController.enableRotate = false;
      viewer.scene.screenSpaceCameraController.enableTranslate = false;
    }
  }, Cesium.ScreenSpaceEventType.LEFT_DOWN);

  // Mouse move — drag
  handler.setInputAction((movement) => {
    if (!dragActive || !acTransform) return;

    // --- Gizmo axis drag ---
    if (dragAxis && selectedIndex >= 0) {
      const cam = camerasData.cameras[selectedIndex];
      const pos = dragStartACPos;
      const origin = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);

      // AC axis direction in ECEF
      const axisMap = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
      const acDir = axisMap[dragAxis];
      const ecefDir = acTransform.acDirectionToECEF(acDir[0], acDir[1], acDir[2]);
      Cesium.Cartesian3.normalize(ecefDir, ecefDir);

      // Project axis direction to screen space
      const originScreen = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, origin);
      const axisEndWorld = Cesium.Cartesian3.add(
        origin,
        Cesium.Cartesian3.multiplyByScalar(ecefDir, 10, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      );
      const axisEndScreen = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, axisEndWorld);

      if (!originScreen || !axisEndScreen) return;

      // Screen-space axis direction
      const axisScreenDir = new Cesium.Cartesian2(
        axisEndScreen.x - originScreen.x,
        axisEndScreen.y - originScreen.y
      );
      const axisScreenLen = Cesium.Cartesian2.magnitude(axisScreenDir);
      if (axisScreenLen < 0.001) return;
      Cesium.Cartesian2.divideByScalar(axisScreenDir, axisScreenLen, axisScreenDir);

      // Mouse delta from drag start
      const mouseDelta = new Cesium.Cartesian2(
        movement.endPosition.x - dragStartScreenPos.x,
        movement.endPosition.y - dragStartScreenPos.y
      );

      // Project mouse delta onto axis screen direction
      const projection = Cesium.Cartesian2.dot(mouseDelta, axisScreenDir);

      // Estimate meters per pixel using camera distance
      const camDistance = Cesium.Cartesian3.distance(viewer.camera.positionWC, origin);
      const metersPerPixel = camDistance / viewer.canvas.height;

      // Compute AC-space displacement along the chosen axis
      const displacement = projection * metersPerPixel;

      // Update only the corresponding AC axis component
      const newPos = [...dragStartACPos];
      const axisIdx = { x: 0, y: 1, z: 2 }[dragAxis];
      newPos[axisIdx] += displacement;
      cam.POSITION = newPos;

      // Update entity position
      const newCartesian = acTransform.acToCartesian3(newPos[0], newPos[1], newPos[2]);
      if (cameraEntities[selectedIndex]) {
        cameraEntities[selectedIndex].position = newCartesian;
      }

      fillProps(cam);
      updateFrustum();
      return;
    }

    // --- Fallback: camera marker XZ surface drag ---
    if (!dragEntity) return;
    const ray = viewer.camera.getPickRay(movement.endPosition);
    if (!ray) return;

    let cartesian = viewer.scene.pickPosition(movement.endPosition);
    if (!cartesian || !Cesium.defined(cartesian)) {
      cartesian = viewer.scene.globe.show
        ? viewer.scene.globe.pick(ray, viewer.scene)
        : viewer.camera.pickEllipsoid(movement.endPosition, viewer.scene.globe.ellipsoid);
    }
    if (!cartesian) return;

    const acCoords = acTransform.cartesian3ToAC(cartesian);
    const cam = camerasData.cameras[dragEntity._camIndex];
    if (!cam) return;

    const oldPos = cam.POSITION || [0, 0, 0];
    cam.POSITION = [acCoords[0], oldPos[1], acCoords[2]];

    const newCartesian = acTransform.acToCartesian3(acCoords[0], oldPos[1], acCoords[2]);
    dragEntity.position = newCartesian;

    if (dragEntity._camIndex === selectedIndex) {
      fillProps(cam);
    }
    updateFrustum();
  }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

  // Left up — end drag
  handler.setInputAction(() => {
    if (dragActive) {
      dragActive = false;
      dragEntity = null;
      dragAxis = null;
      dragStartScreenPos = null;
      dragStartACPos = null;
      viewer.scene.screenSpaceCameraController.enableRotate = true;
      viewer.scene.screenSpaceCameraController.enableTranslate = true;
      pushHistory();
      markDirty();
    }
  }, Cesium.ScreenSpaceEventType.LEFT_UP);
}

/* =========================================================================
   Data loading
   ========================================================================= */

async function loadLayouts() {
  const resp = await fetch("/api/track_layouts");
  if (!resp.ok) {
    // No layouts available — return empty
    return [];
  }
  const data = await resp.json();
  const lays = data.layouts || [];
  if (lays.length === 0) {
    return [{ name: "Default", short: "default" }];
  }
  return lays.map((l) => ({
    name: l.name,
    short: l.name.toLowerCase(),
    direction: l.track_direction || "clockwise",
  }));
}

async function onLayoutChange(layoutShort) {
  currentLayout = layoutShort;
  $layoutSelect.value = layoutShort;
  selectedIndex = -1;
  isDirty = false;
  $dirtyFlag.hidden = true;
  $btnSave.disabled = true;

  setStatus("Loading cameras for " + layoutShort + "...");

  // Load cameras
  try {
    const resp = await fetch("/api/cameras/" + encodeURIComponent(layoutShort));
    if (resp.ok) {
      camerasData = await resp.json();
    } else {
      camerasData = { header: { VERSION: 3, CAMERA_COUNT: 0, SET_NAME: "TV1" }, cameras: [] };
    }
  } catch (e) {
    camerasData = { header: { VERSION: 3, CAMERA_COUNT: 0, SET_NAME: "TV1" }, cameras: [] };
  }

  // Load centerline
  try {
    const resp = await fetch("/api/cameras_centerline/" + encodeURIComponent(layoutShort));
    if (resp.ok) {
      centerlineData = await resp.json();
    } else {
      centerlineData = null;
    }
  } catch (e) {
    centerlineData = null;
  }

  // Reset history
  history = [JSON.parse(JSON.stringify(camerasData))];
  historyIndex = 0;
  $btnUndo.disabled = true;
  $btnRedo.disabled = true;

  // Render
  renderCameraList();
  renderCesiumMarkers();
  renderCenterline();
  selectCamera(-1);

  setStatus(
    camerasData.cameras.length + " camera(s)" +
    (centerlineData ? ", centerline loaded" : "")
  );
}

/* =========================================================================
   Cesium rendering
   ========================================================================= */

function renderCesiumMarkers() {
  // Remove old entities
  cameraEntities.forEach((e) => viewer.entities.remove(e));
  cameraEntities = [];
  clearFrustum();
  clearGizmo();
  clearTrackPositionMarker();

  if (!camerasData || !acTransform) return;

  camerasData.cameras.forEach((cam, i) => {
    const pos = cam.POSITION || [0, 0, 0];
    const cartesian = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);

    const entity = viewer.entities.add({
      position: cartesian,
      billboard: {
        image: makeCameraBillboard(i, i === selectedIndex),
        width: 28,
        height: 28,
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: String(cam.NAME || i),
        font: "bold 12px sans-serif",
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -18),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    entity._camIndex = i;
    cameraEntities.push(entity);
  });
}

function renderCenterline() {
  if (centerlineEntity) {
    viewer.entities.remove(centerlineEntity);
    centerlineEntity = null;
  }
  if (inOutEntity) {
    viewer.entities.remove(inOutEntity);
    inOutEntity = null;
  }

  if (!centerlineData || !acTransform) return;
  const pts = centerlineData.centerline;
  if (!pts || pts.length < 2) return;

  // Convert centerline points to Cartesian3
  // Centerline coords are in Blender pixel space (x, z in XZ plane, y=0)
  // These are the same coordinate space as AC x, z with y=0
  const positions = pts.map((p) => acTransform.acToCartesian3(p[0], 0, p[1]));

  centerlineEntity = viewer.entities.add({
    polyline: {
      positions: positions,
      width: 2,
      material: new Cesium.PolylineGlowMaterialProperty({
        glowPower: 0.15,
        color: Cesium.Color.fromCssColorString("#60a5fa").withAlpha(0.6),
      }),
      clampToGround: true,
    },
  });
}

function updateInOutHighlight() {
  if (inOutEntity) {
    viewer.entities.remove(inOutEntity);
    inOutEntity = null;
  }
  if (selectedIndex < 0 || !centerlineData || !acTransform) return;

  const cam = camerasData.cameras[selectedIndex];
  const inPt = cam.IN_POINT || 0;
  const outPt = cam.OUT_POINT || 0;
  const pts = centerlineData.centerline;
  if (!pts || pts.length < 2 || (inPt === 0 && outPt === 0)) return;

  // Convert IN/OUT fractions to centerline indices (offset by time0_idx)
  const startIdx = Math.floor(trackTToCenterlineIndex(inPt, pts));
  const endIdx = Math.floor(trackTToCenterlineIndex(outPt, pts));

  // Collect points walking from startIdx to endIdx in driving direction
  const segPts = [];
  const n = pts.length;
  const forward = drivingFollowsIndex();
  let i = startIdx;
  while (true) {
    segPts.push(pts[i]);
    if (i === endIdx) break;
    i = forward ? (i + 1) % n : ((i - 1) + n) % n;
    if (segPts.length > n) break; // safety
  }
  if (segPts.length < 2) return;

  const positions = segPts.map((p) => acTransform.acToCartesian3(p[0], 0, p[1]));

  inOutEntity = viewer.entities.add({
    polyline: {
      positions: positions,
      width: 4,
      material: Cesium.Color.fromCssColorString("#f59e0b").withAlpha(0.8),
      clampToGround: true,
    },
  });
}

/* ---- Gizmo (Feature 1) ---- */

function renderGizmo() {
  clearGizmo();
  if (selectedIndex < 0 || !acTransform || !camerasData) return;

  const cam = camerasData.cameras[selectedIndex];
  const pos = cam.POSITION || [0, 0, 0];
  const origin = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);

  const axisLength = 15; // meters
  const axes = [
    { key: "x", ac: [1, 0, 0], color: Cesium.Color.RED },
    { key: "y", ac: [0, 1, 0], color: Cesium.Color.GREEN },
    { key: "z", ac: [0, 0, 1], color: Cesium.Color.BLUE },
  ];

  gizmoEntities = {};
  for (const ax of axes) {
    const dir = acTransform.acDirectionToECEF(ax.ac[0], ax.ac[1], ax.ac[2]);
    Cesium.Cartesian3.normalize(dir, dir);
    const end = Cesium.Cartesian3.add(
      origin,
      Cesium.Cartesian3.multiplyByScalar(dir, axisLength, new Cesium.Cartesian3()),
      new Cesium.Cartesian3()
    );

    const entity = viewer.entities.add({
      polyline: {
        positions: [origin, end],
        width: 4,
        material: ax.color,
        clampToGround: false,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    entity._gizmoAxis = ax.key;
    gizmoEntities[ax.key] = entity;
  }
}

function clearGizmo() {
  if (gizmoEntities) {
    for (const key in gizmoEntities) {
      viewer.entities.remove(gizmoEntities[key]);
    }
    gizmoEntities = null;
  }
}

function updateGizmoPositions() {
  if (!gizmoEntities || selectedIndex < 0 || !acTransform || !camerasData) return;

  const cam = camerasData.cameras[selectedIndex];
  const pos = cam.POSITION || [0, 0, 0];
  const origin = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);
  const axisLength = 15;

  const axesDef = [
    { key: "x", ac: [1, 0, 0] },
    { key: "y", ac: [0, 1, 0] },
    { key: "z", ac: [0, 0, 1] },
  ];

  for (const ax of axesDef) {
    const entity = gizmoEntities[ax.key];
    if (!entity) continue;
    const dir = acTransform.acDirectionToECEF(ax.ac[0], ax.ac[1], ax.ac[2]);
    Cesium.Cartesian3.normalize(dir, dir);
    const end = Cesium.Cartesian3.add(
      origin,
      Cesium.Cartesian3.multiplyByScalar(dir, axisLength, new Cesium.Cartesian3()),
      new Cesium.Cartesian3()
    );
    entity.polyline.positions = [origin, end];
  }
}

function clearFrustum() {
  frustumEntities.forEach((e) => viewer.entities.remove(e));
  frustumEntities = [];
  if (inOutEntity) {
    viewer.entities.remove(inOutEntity);
    inOutEntity = null;
  }
  clearFrustumProjection();
}

function updateFrustum() {
  clearFrustum();
  if (selectedIndex < 0 || !acTransform) return;

  const cam = camerasData.cameras[selectedIndex];
  const pos = cam.POSITION || [0, 0, 0];
  const fwd = cam.FORWARD || [0, 0, 1];

  const origin = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);
  const fwdDir = acTransform.acDirectionToECEF(fwd[0], fwd[1], fwd[2]);
  Cesium.Cartesian3.normalize(fwdDir, fwdDir);

  // Draw forward arrow line
  const arrowLen = 30; // meters
  const arrowEnd = Cesium.Cartesian3.add(
    origin,
    Cesium.Cartesian3.multiplyByScalar(fwdDir, arrowLen, new Cesium.Cartesian3()),
    new Cesium.Cartesian3()
  );

  frustumEntities.push(
    viewer.entities.add({
      polyline: {
        positions: [origin, arrowEnd],
        width: 3,
        material: new Cesium.PolylineArrowMaterialProperty(
          Cesium.Color.fromCssColorString("#ef4444")
        ),
        clampToGround: false,
      },
    })
  );

  // Simple frustum cone (FOV visualization)
  const fov = ((previewFOV || cam.MAX_FOV || 60) * Math.PI) / 180;
  const frustumLen = 50;
  const halfW = Math.tan(fov / 2) * frustumLen;

  // We need a "right" and "up" vector in ECEF
  const upDir = acTransform.acDirectionToECEF(0, 1, 0);
  Cesium.Cartesian3.normalize(upDir, upDir);
  const rightDir = Cesium.Cartesian3.cross(fwdDir, upDir, new Cesium.Cartesian3());
  Cesium.Cartesian3.normalize(rightDir, rightDir);
  const realUp = Cesium.Cartesian3.cross(rightDir, fwdDir, new Cesium.Cartesian3());
  Cesium.Cartesian3.normalize(realUp, realUp);

  // 4 corner points of the frustum far plane
  const center = Cesium.Cartesian3.add(
    origin,
    Cesium.Cartesian3.multiplyByScalar(fwdDir, frustumLen, new Cesium.Cartesian3()),
    new Cesium.Cartesian3()
  );
  const corners = [
    Cesium.Cartesian3.add(
      center,
      Cesium.Cartesian3.add(
        Cesium.Cartesian3.multiplyByScalar(rightDir, halfW, new Cesium.Cartesian3()),
        Cesium.Cartesian3.multiplyByScalar(realUp, halfW * 0.6, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      ),
      new Cesium.Cartesian3()
    ),
    Cesium.Cartesian3.add(
      center,
      Cesium.Cartesian3.add(
        Cesium.Cartesian3.multiplyByScalar(rightDir, -halfW, new Cesium.Cartesian3()),
        Cesium.Cartesian3.multiplyByScalar(realUp, halfW * 0.6, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      ),
      new Cesium.Cartesian3()
    ),
    Cesium.Cartesian3.add(
      center,
      Cesium.Cartesian3.add(
        Cesium.Cartesian3.multiplyByScalar(rightDir, -halfW, new Cesium.Cartesian3()),
        Cesium.Cartesian3.multiplyByScalar(realUp, -halfW * 0.6, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      ),
      new Cesium.Cartesian3()
    ),
    Cesium.Cartesian3.add(
      center,
      Cesium.Cartesian3.add(
        Cesium.Cartesian3.multiplyByScalar(rightDir, halfW, new Cesium.Cartesian3()),
        Cesium.Cartesian3.multiplyByScalar(realUp, -halfW * 0.6, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      ),
      new Cesium.Cartesian3()
    ),
  ];

  // Draw frustum edges
  const frustumColor = Cesium.Color.fromCssColorString("#60a5fa").withAlpha(0.5);
  for (const c of corners) {
    frustumEntities.push(
      viewer.entities.add({
        polyline: {
          positions: [origin, c],
          width: 1,
          material: frustumColor,
          clampToGround: false,
        },
      })
    );
  }
  // Far plane outline
  frustumEntities.push(
    viewer.entities.add({
      polyline: {
        positions: [...corners, corners[0]],
        width: 1,
        material: frustumColor,
        clampToGround: false,
      },
    })
  );

  // Semi-transparent frustum polygon
  frustumEntities.push(
    viewer.entities.add({
      polygon: {
        hierarchy: new Cesium.PolygonHierarchy(corners),
        material: Cesium.Color.fromCssColorString("#60a5fa").withAlpha(0.08),
        perPositionHeight: true,
      },
    })
  );

  updateInOutHighlight();
  updateGizmoPositions();
  updateFrustumProjection();
}

/* =========================================================================
   Billboard canvas generation
   ========================================================================= */

function makeCameraBillboard(index, selected) {
  const size = 28;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  // Circle
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2);
  ctx.fillStyle = selected ? "#f59e0b" : "#2563eb";
  ctx.fill();
  ctx.strokeStyle = selected ? "#fbbf24" : "#60a5fa";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Number
  ctx.fillStyle = "#fff";
  ctx.font = "bold 12px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(index), size / 2, size / 2);

  return canvas;
}

/* =========================================================================
   Camera list (left panel)
   ========================================================================= */

function renderCameraList() {
  $cameraList.innerHTML = "";
  if (!camerasData || camerasData.cameras.length === 0) {
    $cameraList.innerHTML = '<div class="cam-empty">No cameras. Click + to add.</div>';
    return;
  }

  camerasData.cameras.forEach((cam, i) => {
    const div = document.createElement("div");
    div.className = "cam-list-item" + (i === selectedIndex ? " cam-list-item--selected" : "");
    const inP = cam.IN_POINT != null ? cam.IN_POINT.toFixed(2) : "?";
    const outP = cam.OUT_POINT != null ? cam.OUT_POINT.toFixed(2) : "?";
    div.innerHTML = `
      <span class="cam-list-item__num">${i}</span>
      <span class="cam-list-item__name">${cam.NAME || "Camera " + i}</span>
      <span class="cam-list-item__fov">${cam.MIN_FOV || "?"}-${cam.MAX_FOV || "?"}&deg; &nbsp;${inP}-${outP}</span>
    `;
    div.addEventListener("click", () => selectCamera(i));
    div.addEventListener("dblclick", () => flyToCamera(i));
    $cameraList.appendChild(div);
  });
}

/* =========================================================================
   Selection
   ========================================================================= */

function selectCamera(index) {
  selectedIndex = index;

  // Update list highlight
  renderCameraList();

  // Update billboard appearance
  cameraEntities.forEach((e, i) => {
    e.billboard.image = makeCameraBillboard(i, i === selectedIndex);
  });

  if (index >= 0 && camerasData && camerasData.cameras[index]) {
    const cam = camerasData.cameras[index];
    $propsSection.hidden = false;
    fillProps(cam);

    // Update FOV preview slider range/value to match selected camera
    const minFov = cam.MIN_FOV || 10;
    const maxFov = cam.MAX_FOV || 60;
    $fovPreviewSlider.min = minFov;
    $fovPreviewSlider.max = maxFov;
    $fovPreviewSlider.value = previewFOV != null ? Math.min(Math.max(previewFOV, minFov), maxFov) : maxFov;
    previewFOV = parseFloat($fovPreviewSlider.value);
    $fovPreviewLabel.textContent = previewFOV + "°";

    // Update track position slider range to camera's IN/OUT coverage
    const inPt = cam.IN_POINT || 0;
    const outPt = cam.OUT_POINT || 1;
    $trackPosSlider.min = inPt;
    $trackPosSlider.max = outPt;
    $trackPosSlider.step = 0.001;
    // Clamp current value into the new range
    const curT = parseFloat($trackPosSlider.value) || inPt;
    $trackPosSlider.value = Math.min(Math.max(curT, inPt), outPt);
    $trackPosLabel.textContent = parseFloat($trackPosSlider.value).toFixed(2);

    updateFrustum();
    renderGizmo();
    clearTrackPositionMarker();
  } else {
    $propsSection.hidden = true;
    clearFrustum();
    clearGizmo();
    clearTrackPositionMarker();
    previewFOV = null;
  }
}

function flyToCamera(index) {
  if (!acTransform || !camerasData || !camerasData.cameras[index]) return;
  const pos = camerasData.cameras[index].POSITION || [0, 0, 0];
  const cartesian = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.add(
      cartesian,
      new Cesium.Cartesian3(0, 0, 200),
      new Cesium.Cartesian3()
    ),
    orientation: {
      heading: viewer.camera.heading,
      pitch: Cesium.Math.toRadians(-45),
      roll: 0,
    },
    duration: 0.8,
  });
}

/* =========================================================================
   Properties panel
   ========================================================================= */

function fillProps(cam) {
  $propName.value = cam.NAME || "";

  const pos = cam.POSITION || [0, 0, 0];
  $propPosX.value = pos[0] != null ? Number(pos[0]).toFixed(3) : "";
  $propPosY.value = pos[1] != null ? Number(pos[1]).toFixed(3) : "";
  $propPosZ.value = pos[2] != null ? Number(pos[2]).toFixed(3) : "";

  const fwd = cam.FORWARD || [0, 0, 1];
  $propFwdX.value = fwd[0] != null ? Number(fwd[0]).toFixed(4) : "";
  $propFwdY.value = fwd[1] != null ? Number(fwd[1]).toFixed(4) : "";
  $propFwdZ.value = fwd[2] != null ? Number(fwd[2]).toFixed(4) : "";

  $propFovMin.value = cam.MIN_FOV != null ? cam.MIN_FOV : "";
  $propFovMax.value = cam.MAX_FOV != null ? cam.MAX_FOV : "";
  $propInPoint.value = cam.IN_POINT != null ? cam.IN_POINT : "";
  $propOutPoint.value = cam.OUT_POINT != null ? cam.OUT_POINT : "";
  $propIsFixed.value = cam.IS_FIXED != null ? cam.IS_FIXED : 0;
}

function onPropChange() {
  if (selectedIndex < 0 || !camerasData) return;
  const cam = camerasData.cameras[selectedIndex];

  cam.NAME = $propName.value;
  cam.POSITION = [
    parseFloat($propPosX.value) || 0,
    parseFloat($propPosY.value) || 0,
    parseFloat($propPosZ.value) || 0,
  ];
  cam.FORWARD = [
    parseFloat($propFwdX.value) || 0,
    parseFloat($propFwdY.value) || 0,
    parseFloat($propFwdZ.value) || 0,
  ];
  cam.MIN_FOV = parseFloat($propFovMin.value) || 10;
  cam.MAX_FOV = parseFloat($propFovMax.value) || 60;
  cam.IN_POINT = parseFloat($propInPoint.value) || 0;
  cam.OUT_POINT = parseFloat($propOutPoint.value) || 0;
  cam.IS_FIXED = parseInt($propIsFixed.value) || 0;

  // Update 3D marker position
  if (acTransform && cameraEntities[selectedIndex]) {
    const pos = cam.POSITION;
    cameraEntities[selectedIndex].position = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);
  }

  renderCameraList();
  updateFrustum();
  pushHistory();
  markDirty();
}

/* =========================================================================
   Add / Delete camera
   ========================================================================= */

function addCamera() {
  if (!camerasData) return;

  // Determine a default position along the centerline
  let px = 0, py = -8, pz = 0;
  let fwdX = 1, fwdY = -0.15, fwdZ = 0;
  let inPoint = 0, outPoint = 1;

  const hasCenterline = centerlineData && centerlineData.centerline && centerlineData.centerline.length > 2;
  const numCams = camerasData.cameras.length;

  if (hasCenterline) {
    const pts = centerlineData.centerline;
    const n = pts.length;
    const forward = drivingFollowsIndex();

    // Evenly distribute new camera: place at fraction (numCams)/(numCams+1) of the track
    // so each new camera fills a gap
    const t = numCams > 0 ? numCams / (numCams + 1) : 0.5;
    const fracIdx = trackTToCenterlineIndex(t, pts);
    const [cx, cz] = interpolateCenterline(pts, fracIdx);

    // Offset camera 15m to the side of the centerline (perpendicular to driving direction)
    const i0 = ((Math.floor(fracIdx) % n) + n) % n;
    const i1 = forward ? (i0 + 1) % n : ((i0 - 1) + n) % n;
    const tangX = pts[i1][0] - pts[i0][0];
    const tangZ = pts[i1][1] - pts[i0][1];
    const tangLen = Math.sqrt(tangX * tangX + tangZ * tangZ) || 1;
    // Perpendicular (rotate 90°): (-tangZ, tangX), pick the side
    const perpX = -tangZ / tangLen;
    const perpZ = tangX / tangLen;

    px = cx + perpX * 15;
    pz = cz + perpZ * 15;
    py = -8; // 8m above ground (AC Y is negative-up)

    // FORWARD: point from camera toward the centerline point
    const dx = cx - px, dz = cz - pz;
    const dLen = Math.sqrt(dx * dx + dz * dz) || 1;
    fwdX = dx / dLen;
    fwdY = -0.1; // slight downward look
    fwdZ = dz / dLen;

    // Coverage: assign a proportional segment around the camera's track position
    const segLen = 1 / Math.max(numCams + 1, 2);
    inPoint = Math.max(0, t - segLen / 2);
    outPoint = Math.min(1, t + segLen / 2);
  } else if (numCams > 0) {
    const last = camerasData.cameras[numCams - 1];
    const lp = last.POSITION || [0, 0, 0];
    px = lp[0] + 20;
    py = lp[1];
    pz = lp[2];
  }

  const idx = numCams;
  const newCam = {
    NAME: String(idx + 1),
    POSITION: [px, py, pz],
    FORWARD: [fwdX, fwdY, fwdZ],
    UP: [0, 1, 0],
    MIN_FOV: 10,
    MAX_FOV: 60,
    IN_POINT: parseFloat(inPoint.toFixed(3)),
    OUT_POINT: parseFloat(outPoint.toFixed(3)),
    SHADOW_SPLIT0: 1.8,
    SHADOW_SPLIT1: 20,
    SHADOW_SPLIT2: 180,
    NEAR_PLANE: 0.1,
    FAR_PLANE: 5000,
    MIN_EXPOSURE: 0,
    MAX_EXPOSURE: 10000,
    DOF_FACTOR: 10,
    DOF_RANGE: 10000,
    DOF_FOCUS: 0,
    DOF_MANUAL: 0,
    SPLINE: "",
    SPLINE_ROTATION: 0,
    FOV_GAMMA: 0,
    SPLINE_ANIMATION_LENGTH: 15,
    IS_FIXED: 0,
  };

  camerasData.cameras.push(newCam);
  camerasData.header.CAMERA_COUNT = camerasData.cameras.length;

  renderCesiumMarkers();
  renderCameraList();
  selectCamera(idx);
  pushHistory();
  markDirty();
}

function deleteCamera() {
  if (selectedIndex < 0 || !camerasData) return;
  camerasData.cameras.splice(selectedIndex, 1);
  camerasData.header.CAMERA_COUNT = camerasData.cameras.length;

  const newSel = Math.min(selectedIndex, camerasData.cameras.length - 1);
  renderCesiumMarkers();
  renderCameraList();
  selectCamera(newSel);
  pushHistory();
  markDirty();
}

/* =========================================================================
   Dirty state / undo / redo
   ========================================================================= */

function markDirty() {
  isDirty = true;
  $dirtyFlag.hidden = false;
  $btnSave.disabled = false;
}

function pushHistory() {
  // Truncate redo stack
  history.splice(historyIndex + 1);
  history.push(JSON.parse(JSON.stringify(camerasData)));
  historyIndex = history.length - 1;
  $btnUndo.disabled = historyIndex <= 0;
  $btnRedo.disabled = true;
}

function undo() {
  if (historyIndex <= 0) return;
  historyIndex--;
  camerasData = JSON.parse(JSON.stringify(history[historyIndex]));
  afterHistoryRestore();
}

function redo() {
  if (historyIndex >= history.length - 1) return;
  historyIndex++;
  camerasData = JSON.parse(JSON.stringify(history[historyIndex]));
  afterHistoryRestore();
}

function afterHistoryRestore() {
  $btnUndo.disabled = historyIndex <= 0;
  $btnRedo.disabled = historyIndex >= history.length - 1;
  renderCesiumMarkers();
  renderCameraList();
  if (selectedIndex >= camerasData.cameras.length) {
    selectCamera(camerasData.cameras.length - 1);
  } else if (selectedIndex >= 0) {
    fillProps(camerasData.cameras[selectedIndex]);
    updateFrustum();
  }
  markDirty();
}

/* =========================================================================
   Save
   ========================================================================= */

async function save() {
  if (!camerasData || !currentLayout) return;
  $btnSave.disabled = true;
  setStatus("Saving...");

  try {
    const resp = await fetch("/api/cameras/" + encodeURIComponent(currentLayout), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(camerasData),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text);
    }
    isDirty = false;
    $dirtyFlag.hidden = true;
    $btnSave.disabled = true;
    setStatus("Saved successfully");
    showToast("Saved", "ok");
  } catch (e) {
    setStatus("Save failed: " + e.message);
    showToast("Save failed: " + e.message, "err");
    $btnSave.disabled = false;
  }
}

/* =========================================================================
   Track Position Preview (Feature 2)
   ========================================================================= */

/**
 * Determine if driving direction follows increasing centerline indices.
 * Mirrors the Python logic: signed_area > 0 ⇒ centerline is CW,
 * then driving_follows_index = (centerline_is_cw == track_is_cw).
 */
function drivingFollowsIndex() {
  if (!centerlineData) return true;
  const pts = centerlineData.centerline;
  if (!pts || pts.length < 3) return true;

  // Shoelace signed area (same convention as Python road_centerline.py)
  let signedArea = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    signedArea += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1];
  }
  signedArea += pts[pts.length - 1][0] * pts[0][1] - pts[0][0] * pts[pts.length - 1][1];
  signedArea *= 0.5;

  const centerlineIsCW = signedArea > 0;
  const trackDir = centerlineData.track_direction
    || (layouts.find((l) => l.short === currentLayout) || {}).direction
    || "clockwise";
  const trackIsCW = trackDir === "clockwise";
  return centerlineIsCW === trackIsCW;
}

/**
 * Convert a track fraction t (0~1, starting from timer_0) to a
 * fractional centerline array index, accounting for time0_idx offset
 * and driving direction.
 */
function trackTToCenterlineIndex(t, pts) {
  const time0 = centerlineData.time0_idx || 0;
  const n = pts.length;
  const forward = drivingFollowsIndex();
  // forward: index increases with t; reverse: index decreases with t
  const raw = forward
    ? time0 + t * n
    : time0 - t * n;
  return ((raw % n) + n) % n; // always positive
}

/**
 * Interpolate centerline position at a fractional index (wrapping).
 * The interpolation step direction matches the driving direction.
 * Returns [x, z] in AC / pixel coords.
 */
function interpolateCenterline(pts, fracIdx) {
  const n = pts.length;
  const i0 = ((Math.floor(fracIdx) % n) + n) % n;
  const forward = drivingFollowsIndex();
  const i1 = forward ? (i0 + 1) % n : ((i0 - 1) + n) % n;
  const frac = fracIdx - Math.floor(fracIdx);
  const px = pts[i0][0] + (pts[i1][0] - pts[i0][0]) * frac;
  const pz = pts[i0][1] + (pts[i1][1] - pts[i0][1]) * frac;
  return [px, pz];
}

function updateTrackPositionPreview(t) {
  clearTrackPositionMarker();
  if (!centerlineData || !acTransform) return;
  const pts = centerlineData.centerline;
  if (!pts || pts.length < 2) return;

  // Interpolate position along centerline (offset by time0_idx)
  const fracIdx = trackTToCenterlineIndex(t, pts);
  const [px, pz] = interpolateCenterline(pts, fracIdx);
  const cartesian = acTransform.acToCartesian3(px, 0, pz);

  // Since slider range is clamped to IN_POINT~OUT_POINT, always in coverage
  const markerColor = Cesium.Color.LIME;

  // Vertical line marker
  const topCartesian = acTransform.acToCartesian3(px, -20, pz);
  trackPosMarker = viewer.entities.add({
    position: cartesian,
    point: {
      pixelSize: 10,
      color: markerColor,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    polyline: {
      positions: [cartesian, topCartesian],
      width: 2,
      material: markerColor.withAlpha(0.6),
      clampToGround: false,
    },
  });

  // Update camera FORWARD to look at this track position, then refresh frustum
  if (selectedIndex >= 0 && camerasData) {
    const cam = camerasData.cameras[selectedIndex];
    const camPos = cam.POSITION || [0, 0, 0];

    // Direction from camera to track point (in AC coords)
    const dx = px - camPos[0];
    const dy = 0 - camPos[1]; // track is at y=0
    const dz = pz - camPos[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 0.001) {
      cam.FORWARD = [dx / len, dy / len, dz / len];
      fillProps(cam);
      updateFrustum();
    }
  }
}

function clearTrackPositionMarker() {
  if (trackPosMarker) {
    viewer.entities.remove(trackPosMarker);
    trackPosMarker = null;
  }
}

/* =========================================================================
   Frustum Projection (Feature 3)
   ========================================================================= */

function updateFrustumProjection() {
  clearFrustumProjection();
  if (selectedIndex < 0 || !acTransform || !camerasData) return;

  const cam = camerasData.cameras[selectedIndex];
  const pos = cam.POSITION || [0, 0, 0];
  const fwd = cam.FORWARD || [0, 0, 1];

  const origin = acTransform.acToCartesian3(pos[0], pos[1], pos[2]);
  const fwdDir = acTransform.acDirectionToECEF(fwd[0], fwd[1], fwd[2]);
  Cesium.Cartesian3.normalize(fwdDir, fwdDir);

  const fov = ((previewFOV || cam.MAX_FOV || 60) * Math.PI) / 180;
  const maxDist = 200; // max ray distance in meters

  // Build right and up vectors
  const upDir = acTransform.acDirectionToECEF(0, 1, 0);
  Cesium.Cartesian3.normalize(upDir, upDir);
  const rightDir = Cesium.Cartesian3.cross(fwdDir, upDir, new Cesium.Cartesian3());
  Cesium.Cartesian3.normalize(rightDir, rightDir);
  const realUp = Cesium.Cartesian3.cross(rightDir, fwdDir, new Cesium.Cartesian3());
  Cesium.Cartesian3.normalize(realUp, realUp);

  const halfH = Math.tan(fov / 2);
  const halfW = halfH; // assume ~1:1 aspect for horizontal
  const aspectRatio = 16 / 9;
  const halfWAdjusted = halfH * aspectRatio / 2;

  // Sample rays along frustum boundary
  // Bottom edge: 8 samples, left/right edges: 3 each, top edge: 8 samples
  const hitPoints = [];
  const sampleRay = (hFrac, vFrac) => {
    // hFrac: -1..1 horizontal, vFrac: -1..1 vertical
    const dir = new Cesium.Cartesian3();
    Cesium.Cartesian3.add(
      Cesium.Cartesian3.multiplyByScalar(fwdDir, 1, new Cesium.Cartesian3()),
      Cesium.Cartesian3.add(
        Cesium.Cartesian3.multiplyByScalar(rightDir, halfW * hFrac, new Cesium.Cartesian3()),
        Cesium.Cartesian3.multiplyByScalar(realUp, halfH * 0.6 * vFrac, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      ),
      dir
    );
    Cesium.Cartesian3.normalize(dir, dir);

    const rayEnd = Cesium.Cartesian3.add(
      origin,
      Cesium.Cartesian3.multiplyByScalar(dir, maxDist, new Cesium.Cartesian3()),
      new Cesium.Cartesian3()
    );

    // Try to pick surface via screen-space projection
    const screenPos = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, rayEnd);
    if (screenPos) {
      const hitPos = viewer.scene.pickPosition(screenPos);
      if (hitPos && Cesium.defined(hitPos)) {
        // Validate the hit point is roughly in the right direction
        const hitDist = Cesium.Cartesian3.distance(origin, hitPos);
        if (hitDist < maxDist * 1.5 && hitDist > 1) {
          hitPoints.push(hitPos);
          return;
        }
      }
    }

    // Fallback: intersect with ground plane (y=0 in AC coords)
    // AC ground plane: y_ac = 0 → find t where origin + t*dir hits y_ac=0
    const acOrigin = pos;
    const acDirArr = [
      fwd[0] + halfW * hFrac * (fwd[2] || 0.001),
      fwd[1] + halfH * 0.6 * vFrac,
      fwd[2] + halfW * hFrac * (fwd[0] || 0.001),
    ];
    // Simpler fallback: just use rayEnd as-is projected to y=0
    const acEnd = acTransform.cartesian3ToAC(rayEnd);
    const groundHit = acTransform.acToCartesian3(acEnd[0], 0, acEnd[2]);
    hitPoints.push(groundHit);
  };

  // Bottom edge (8 samples)
  for (let i = 0; i <= 7; i++) sampleRay(-1 + (2 * i) / 7, -1);
  // Right edge (3 samples)
  for (let i = 1; i <= 2; i++) sampleRay(1, -1 + (2 * i) / 3);
  // Top edge (8 samples, reversed)
  for (let i = 7; i >= 0; i--) sampleRay(-1 + (2 * i) / 7, 1);
  // Left edge (3 samples, reversed)
  for (let i = 2; i >= 1; i--) sampleRay(-1, -1 + (2 * i) / 3);

  if (hitPoints.length < 3) return;

  // Render projection polygon
  projectionEntities.push(
    viewer.entities.add({
      polygon: {
        hierarchy: new Cesium.PolygonHierarchy(hitPoints),
        material: Cesium.Color.fromCssColorString("#60a5fa").withAlpha(0.12),
        perPositionHeight: true,
      },
    })
  );

  // Projection outline
  projectionEntities.push(
    viewer.entities.add({
      polyline: {
        positions: [...hitPoints, hitPoints[0]],
        width: 2,
        material: Cesium.Color.fromCssColorString("#60a5fa").withAlpha(0.4),
        clampToGround: false,
      },
    })
  );
}

function clearFrustumProjection() {
  projectionEntities.forEach((e) => viewer.entities.remove(e));
  projectionEntities = [];
}

/* =========================================================================
   Utilities
   ========================================================================= */

function setStatus(msg) {
  $status.textContent = msg;
}

function showToast(msg, type) {
  const el = document.createElement("div");
  el.className = "cam-toast cam-toast--" + type;
  el.textContent = msg;
  document.body.appendChild(el);
  requestAnimationFrame(() => {
    el.classList.add("cam-toast--show");
  });
  setTimeout(() => {
    el.classList.remove("cam-toast--show");
    setTimeout(() => el.remove(), 300);
  }, 2500);
}

/* =========================================================================
   Boot
   ========================================================================= */
init();
