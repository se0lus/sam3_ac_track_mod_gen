/**
 * map_rightdrag.js — Shared right-click drag-to-pan for all Leaflet editors.
 *
 * Usage:
 *   const rightDrag = setupRightDrag(map, document.getElementById("map"));
 *   // In context menu handlers, check rightDrag.wasDragging() to skip menu after drag.
 */

// eslint-disable-next-line no-unused-vars
function setupRightDrag(map, mapEl) {
  let dragging = false;
  let dragStart = null;
  let didDrag = false; // true if actual movement happened (> threshold)
  const DRAG_THRESHOLD = 4; // px — movement below this is a click, not a drag

  mapEl.addEventListener("mousedown", (e) => {
    if (e.button !== 2) return;
    dragging = true;
    didDrag = false;
    dragStart = { x: e.clientX, y: e.clientY };
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging || !dragStart) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    if (!didDrag && Math.abs(dx) + Math.abs(dy) < DRAG_THRESHOLD) return;
    didDrag = true;
    map.panBy([-dx, -dy], { animate: false });
    dragStart = { x: e.clientX, y: e.clientY };
  });

  document.addEventListener("mouseup", (e) => {
    if (e.button !== 2) return;
    dragging = false;
    dragStart = null;
  });

  // Prevent default browser context menu on the map
  mapEl.addEventListener("contextmenu", (e) => e.preventDefault());

  return {
    /** True if the most recent right-click involved actual dragging. */
    wasDragging() { return didDrag; },
  };
}
