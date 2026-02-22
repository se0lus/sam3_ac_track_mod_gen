from __future__ import annotations

import bpy  # type: ignore[import-not-found]
import os
import time

from . import ActionSpec
from .c_tiles import CTile
from .mask_select_utils import (
    build_mask_cache, get_mask_objects, objects_hit_by_mask_cache_xz,
    world_bbox_xz_range, _bbox2_overlap,
)

from config import BASE_TILES_DIR, GLB_DIR, BASE_LEVEL, TARGET_FINE_LEVEL

# ----------------------------
# 通用：非阻塞 modal 执行辅助
# ----------------------------
class _SAM3_NonBlockingModalMixin:
    """
    用于 Blender Operator 的通用"非阻塞执行"辅助：
    - event_timer_add / modal_handler_add
    - progress_begin/update/end
    - tag_redraw / redraw_timer / view_layer.update
    - 统一 finish（移除 timer + 结束 progress）

    注意：该 mixin 不规定你的状态机；只提供通用基础设施。
    """

    _timer = None
    _nb_interval_sec: float = 0.05

    def _nb_log(self, msg: str) -> None:
        print(f"[SAM3][non_blocking] {msg}")

    def _nb_request_redraw(self, context: bpy.types.Context) -> None:
        # 触发 UI 刷新（失败也不应中断）
        try:
            if getattr(context, "screen", None) is not None:
                for area in context.screen.areas:
                    try:
                        area.tag_redraw()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        except Exception:
            pass
        try:
            context.view_layer.update()
        except Exception:
            pass

    def _nb_progress_begin(self, context: bpy.types.Context, total: int) -> None:
        try:
            context.window_manager.progress_begin(0, max(1, int(total)))
        except Exception:
            pass

    def _nb_progress_update(self, context: bpy.types.Context, value: int) -> None:
        try:
            context.window_manager.progress_update(int(value))
        except Exception:
            pass

    def _nb_progress_end(self, context: bpy.types.Context) -> None:
        try:
            context.window_manager.progress_end()
        except Exception:
            pass

    def _nb_start_timer(self, context: bpy.types.Context, interval_sec: float | None = None) -> None:
        wm = context.window_manager
        self._nb_interval_sec = float(interval_sec if interval_sec is not None else self._nb_interval_sec)
        try:
            self._timer = wm.event_timer_add(self._nb_interval_sec, window=context.window)
        except Exception:
            self._timer = None
            raise
        wm.modal_handler_add(self)

    def _nb_remove_timer(self, context: bpy.types.Context) -> None:
        wm = context.window_manager
        try:
            if self._timer is not None:
                wm.event_timer_remove(self._timer)
        except Exception:
            pass
        self._timer = None

    def _nb_finish(self, context: bpy.types.Context, cancelled: bool, msg: str = "", report_type: set[str] | None = None) -> set[str]:
        """
        统一收尾：移除 timer + 结束 progress；可选择 report+log。
        """
        self._nb_remove_timer(context)
        self._nb_progress_end(context)
        if msg:
            try:
                # report_type 允许传 {"INFO"} / {"WARNING"}；不传则默认 INFO / WARNING
                if report_type is None:
                    report_type = {"WARNING"} if cancelled else {"INFO"}
                self.report(report_type, msg)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._nb_log(msg)
        return {"CANCELLED"} if cancelled else {"FINISHED"}

#从BASE_TILES_DIR中加载顶层对象
class SAM3_OT_load_base_tiles(_SAM3_NonBlockingModalMixin, bpy.types.Operator):
    """ """

    bl_idname = "sam3.load_base_tiles"
    bl_label = "Load Base Tiles"
    bl_options = {"REGISTER", "UNDO"}

    _phase: str = "INIT"  # INIT / LOAD / DONE
    _root_tile: CTile | None = None
    _queue: list[tuple[int, CTile]] = []
    _total: int = 0
    _done: int = 0

    def _log(self, msg: str) -> None:
        print(f"[SAM3][load_base_tiles] {msg}")

    def _start(self, context: bpy.types.Context) -> set[str]:
        self._log(f"开始执行（非阻塞）：Load Base Tiles level={BASE_LEVEL}")

        tileset_path = os.path.join(BASE_TILES_DIR, "tileset.json")
        root_tile = CTile()
        root_tile.loadFromRootJson(tileset_path)
        self._root_tile = root_tile

        tiles_need_load = compute_tiles_need_load_for_min_level(root_tile, min_level=BASE_LEVEL)
        queue: list[tuple[int, CTile]] = []
        for lv in sorted((tiles_need_load or {}).keys()):
            arr = list(tiles_need_load.get(lv) or [])
            # 稳定顺序：按 content 排序
            arr.sort(key=lambda t: (t.content or ""))
            for t in arr:
                queue.append((lv, t))

        self._queue = queue
        self._total = len(queue)
        self._done = 0
        if self._total == 0:
            return self._nb_finish(context, cancelled=False, msg="没有需要加载的 tiles（队列为空）")

        self._phase = "LOAD"
        self._nb_progress_begin(context, total=self._total)
        # interval 可以调大减少开销；0.02~0.1 都可以
        self._nb_start_timer(context, interval_sec=0.05)
        self._nb_request_redraw(context)
        self.report({"INFO"}, f"Load base tiles started (non-blocking). total={self._total}, press ESC to cancel.")
        return {"RUNNING_MODAL"}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        return self._start(context)

    def execute(self, context: bpy.types.Context):
        # 支持从脚本/搜索直接执行：同样走非阻塞模式
        return self._start(context)

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        if event.type == "ESC":
            self._phase = "DONE"
            return self._nb_finish(context, cancelled=True, msg="用户取消（ESC）")

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._phase != "LOAD":
            return {"PASS_THROUGH"}

        if len(self._queue) == 0:
            self._phase = "DONE"
            self._nb_request_redraw(context)
            return self._nb_finish(context, cancelled=False, msg=f"Load base tiles finished. loaded={self._done}")

        lv, tile = self._queue.pop(0)
        self._done += 1

        try:
            self._log(f"[{self._done}/{self._total}] import level={lv}, tile={tile.content}")
            # 复用现有导入函数：每次只导入 1 个 tile（避免长时间阻塞 UI）
            load_glb_tiles_by_dic_level_array(GLB_DIR, {lv: [tile]})
        except Exception as e:
            self._phase = "DONE"
            return self._nb_finish(context, cancelled=True, msg=f"导入失败并中止：{e}")

        self._nb_progress_update(context, self._done)
        self._nb_request_redraw(context)
        return {"RUNNING_MODAL"}

#base on a json file, load scene at at least min level
def compute_tiles_need_load_for_min_level(root: CTile, min_level: int = 0, select_file_list_path: str = "") -> dict[int, list[CTile]]:
    """
    仅计算"需要加载的 tiles"（按 level 分组），不做任何导入。
    该逻辑从 import_fullscene_with_ctile 拆出，供同步/非阻塞两种执行方式复用。

    当子树无法提供 ``min_level`` 的瓦片时，自动回退加载该分支最深可用层级的
    瓦片，确保整个场景无空洞。
    """
    if not root:
        print("root tile = null")
        return {}

    tiles_need_load: dict[int, list[CTile]] = {}

    def _add(tile: CTile) -> None:
        tiles_need_load.setdefault(tile.meshLevel, []).append(tile)

    def _walk(tile: CTile) -> bool:
        """Recurse into *tile*; return True if at least one mesh was added."""
        if tile.hasMesh:
            # Leaf or already at/above target → load directly
            if not tile.canRefine or tile.meshLevel >= min_level or not tile.children:
                _add(tile)
                return True

            # Can refine & below min_level → try children first
            any_child = False
            for child in tile.children:
                if _walk(child):
                    any_child = True
            if not any_child:
                # Children produced nothing → fallback to this tile
                _add(tile)
                return True
            return True  # children covered this area

        # Non-mesh node (e.g. virtual root, JSON redirect)
        if tile.canRefine and tile.children:
            any_child = False
            for child in tile.children:
                if _walk(child):
                    any_child = True
            return any_child
        return False

    entry_tiles: list[CTile] = [root]
    if select_file_list_path:
        entry_tiles = []
        with open(select_file_list_path) as f:
            for name in f.readlines():
                if name.strip():
                    found = root.find(name.strip())
                    if found and found not in entry_tiles:
                        entry_tiles.append(found)

    print("check {0} root tiles".format(len(entry_tiles)))
    for tile in entry_tiles:
        _walk(tile)

    return tiles_need_load

def import_fullscene_with_ctile(root:CTile, path, min_level = 0, select_file_list_path = "", on_tile_loaded=None):
    #get file list
    print("start loading glb tiles in:{0}, min_level:{1}".format(path, min_level))

    if not root:
        print("root tile = null")
        return

    tiles_need_load = compute_tiles_need_load_for_min_level(root, min_level=min_level, select_file_list_path=select_file_list_path)

    for level in tiles_need_load:
        print("need to load level {0}:{1} tiles".format(level, len(tiles_need_load[level])))

    load_glb_tiles_by_dic_level_array(path, tiles_need_load, on_tile_loaded=on_tile_loaded)

def load_glb_tiles_by_dic_level_array(path, tiles_need_load, on_tile_loaded=None):
    # make load order stable (e.g. 0 -> 17)
    for current_level in sorted(tiles_need_load.keys()):
        
        #log files from new
        objects_imported = [] 
        
        #get files in the same level
        for tile in tiles_need_load[current_level]:
            
            # Convert tileset content (.b3dm) to glb filename.
            # NOTE: tile.content may contain subfolders; keep it as relative path.
            content = tile.content or ""
            rel = content.replace("/", os.sep).replace("\\", os.sep).lstrip("\\/")
            name = os.path.splitext(rel)[0] + ".glb"
            filepath = os.path.join(path, name)
            #we need map the glb file name to the node name, so do import one by one
            print("import level {0}, {1}".format(current_level, name))

            if not os.path.exists(filepath):
                # b3dm converter preserves subdirectory structure (e.g.,
                # Model_0/BlockBAA/BlockBAA_L17_26.glb), but tile.content is
                # just the filename.  Try block-prefix subdirectories as
                # fallback, including nested parent dirs (e.g. Model_0/).
                base = os.path.basename(name)
                found = False
                block_prefix = ""
                for i in range(len(base) - 2):
                    if base[i:i+2] == "_L" and base[i+2].isdigit():
                        block_prefix = base[:i]
                        break
                if block_prefix:
                    # Try flat: glb_dir/BlockXXX/file.glb
                    alt = os.path.join(path, block_prefix, base)
                    if os.path.exists(alt):
                        filepath = alt
                        found = True
                    else:
                        # Try with parent dirs: glb_dir/*/BlockXXX/file.glb
                        for entry in os.listdir(path):
                            alt = os.path.join(path, entry, block_prefix, base)
                            if os.path.exists(alt):
                                filepath = alt
                                found = True
                                break
                if not found:
                    print("WARNING: glb not found, skip:", filepath)
                    continue
        
            #test
            #current_level = 16
            #file_names = [{"name":"Block_L16_3.glb"},{"name":"Block_L16_4.glb"}]
        
            #import gltf
            orig_objects = bpy.data.objects.keys()
            bpy.ops.import_scene.gltf(filepath=filepath, filter_glob='*.glb;*.gltf', loglevel=0, import_pack_images=True)
            #find new objects imported
            now_objects = bpy.data.objects.keys();
            add_objects = set(now_objects) - set(orig_objects)
            
            #rename the new object to file name
            for object_key in add_objects:
                object = bpy.data.objects[object_key]
                
                if object_key.find("Node") < 0 or name in objects_imported:
                    object.name = "{0}.{1}".format(name, object_key)    
                    objects_imported.append(object.name)
                    continue
                
                object.name = name
                objects_imported.append(name)
                #node_name_map[object_key] = name

            # Notify caller after each tile (e.g. to redraw viewport)
            if on_tile_loaded is not None:
                try:
                    on_tile_loaded()
                except Exception:
                    pass

        if len(objects_imported) == 0:
            continue
        print("imported objects in level{0}:{1}".format(current_level, str(objects_imported)))
        
        #move node into collections
        collection_name = "L{0}".format(current_level)
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            collection = bpy.data.collections.new(collection_name)
            #link collection to scene
            bpy.context.scene.collection.children.link(collection)
    
        # 先从所有 collection 中移除，再 link 到目标 collection（避免对象残留在任意其他集合里）
        # 注意：Blender 允许 object 同时属于多个 collection；这里强制只属于目标 L{level}。
        all_cols = []
        try:
            all_cols = list(getattr(bpy.data, "collections", None) or [])
        except Exception:
            all_cols = []
        scene_root_col = None
        try:
            scene_root_col = bpy.context.scene.collection
        except Exception:
            scene_root_col = None

        for obj_name in objects_imported:
            obj = bpy.data.objects[obj_name]

            # 1) 遍历所有 collection（含 scene root），若已 link 则 unlink
            # 1.1) 先处理 obj.users_collection（更快、更准确）
            try:
                for col in list(getattr(obj, "users_collection", []) or []):
                    if col is None:
                        continue
                    try:
                        if col.objects.get(obj.name) is not None:
                            col.objects.unlink(obj)
                    except Exception:
                        continue
            except Exception:
                pass

            # 1.2) 再做一次"全量扫描"兜底（满足：先搜索场景所有 collection）
            for col in all_cols:
                if col is None:
                    continue
                try:
                    if col.objects.get(obj.name) is not None:
                        col.objects.unlink(obj)
                except Exception:
                    continue
            if scene_root_col is not None:
                try:
                    if scene_root_col.objects.get(obj.name) is not None:
                        scene_root_col.objects.unlink(obj)
                except Exception:
                    pass

            # 2) link 到目标 collection
            try:
                if collection.objects.get(obj.name) is None:
                    collection.objects.link(obj)
            except Exception:
                # 最后兜底：忽略 link 失败（避免脚本中断）
                pass

            #print("move {0} into {1}".format(obj_name, collection_name))
    return

def get_selected_ctiles(root:CTile):
    tiles = []
    for obj in bpy.context.selected_objects:
        name = obj.name.split(".glb")[0]
        tile = root.find(name)
        if tile:
            tiles.append(tile)
    print("select {0} tiles".format(len(tiles)))
    return tiles;

def clear_scene_by_tile(tile):

    name = tile.content.split("\\")[-1]
    name = name.split("/")[-1]
    name = name.split(".b3dm")[0]
    print("remove:{0}".format(name))
        
    #get obj and mesh
    objects = []
    mesh = []
    for ob in bpy.data.objects:
        if name == ob.name.split(".")[0] and ob not in objects:
            objects.append(ob)
            if ob.data and ob.type == 'MESH' and ob.data not in mesh:
                mesh += [ob.data]
    
    #get materials
    materials = []
    for ob in objects:
        material_slots = ob.material_slots
        for m in material_slots:
            if m.material not in materials:
                materials.append(m.material)
                
    #get images
    textures = []
    for m in materials:
        for n in m.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image not in textures:
                    textures += [n.image]
    
    #do remove
    for ob in objects:
        bpy.data.objects.remove(ob)
    for m in materials:
        try:
            bpy.data.materials.remove(m)
        finally:
            continue
    for img in textures:
        bpy.data.images.remove(img)
    for m in mesh:
        bpy.data.meshes.remove(m)

def compute_one_tile_refine_plan(root_tile_to_refine: CTile) -> dict[int, list[CTile]]:
    """
    Compute a one-step refinement loading plan for a single tile.

    Returns a dict mapping level -> list of CTile objects to load.
    Extracted from the modal operator for reuse in headless/synchronous contexts.
    """
    tiles_need_load: dict[int, list[CTile]] = {}
    tiles_need_child = [root_tile_to_refine]
    while len(tiles_need_child) > 0:
        next_level_tiles = []
        for tile in tiles_need_child:
            if tile.hasMesh:
                if (tile.canRefine is False) or (tile.meshLevel > root_tile_to_refine.meshLevel) or (len(tile.children) == 0):
                    level = tile.meshLevel
                    if level not in tiles_need_load:
                        tiles_need_load[level] = []
                    if tile not in tiles_need_load[level]:
                        tiles_need_load[level].append(tile)
                    continue
            if tile.canRefine and len(tile.children):
                next_level_tiles += tile.children
        tiles_need_child = next_level_tiles
    return tiles_need_load


# ---------------------------------------------------------------------------
# Helpers for CTile-level geographic pre-filtering (refine optimisation)
# ---------------------------------------------------------------------------

def _build_tile_index(tile: CTile, index: dict[str, CTile]) -> None:
    """Recursively build content_key → CTile mapping for O(1) lookup."""
    if tile.content:
        key = tile.content.split(".")[0]
        index[key] = tile
    for child in tile.children:
        _build_tile_index(child, index)


def _mask_overall_aabb(
    mask_cache: list,
) -> tuple[float, float, float, float] | None:
    """Union of all mask bboxes in Blender XZ → (min_x, max_x, min_z, max_z)."""
    min_x = min_z = float("inf")
    max_x = max_z = float("-inf")
    for _m, _tris, overall in mask_cache:
        if overall is not None:
            min_x = min(min_x, overall[0])
            max_x = max(max_x, overall[1])
            min_z = min(min_z, overall[2])
            max_z = max(max_z, overall[3])
    if min_x == float("inf"):
        return None
    return (min_x, max_x, min_z, max_z)


def _detect_axis_mapping(
    context: bpy.types.Context,
    tile_index: dict[str, CTile],
) -> str:
    """Detect which CTile bbox axes map to Blender X and Z.

    Compares loaded Blender objects' world bboxes with their CTile
    bounding volumes to determine the axis mapping.  Tries up to 5
    tiles for robustness.

    Returns one of ``"XY"`` ``"XnY"`` ``"XZ"`` ``"XnZ"``.
    """
    votes: dict[str, int] = {}
    checked = 0

    for obj in context.scene.objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        key = obj.name.split(".glb")[0]
        tile = tile_index.get(key)
        if tile is None:
            continue

        blender_bb = world_bbox_xz_range(obj)
        if blender_bb is None:
            continue

        bb = tile.boxBoundingVolume
        if not bb or len(bb) < 12 or all(v == 0 for v in bb):
            continue

        cx, cy, cz = bb[0], bb[1], bb[2]
        bx_mid = (blender_bb[0] + blender_bb[1]) / 2
        bz_mid = (blender_bb[2] + blender_bb[3]) / 2

        candidates = {
            "XY":  (cx,  cy),
            "XnY": (cx, -cy),
            "XZ":  (cx,  cz),
            "XnZ": (cx, -cz),
        }

        best_name = "XY"
        best_err = float("inf")
        for name, (tx, tz) in candidates.items():
            err = abs(tx - bx_mid) + abs(tz - bz_mid)
            if err < best_err:
                best_err = err
                best_name = name

        if best_err < 50:
            votes[best_name] = votes.get(best_name, 0) + 1
            checked += 1
            if checked >= 5:
                break

    if votes:
        winner = max(votes, key=lambda k: votes[k])
        print(f"[refine] axis mapping votes: {votes} → {winner}")
        return winner
    return "XY"  # default


def _tile_aabb_xz(
    tile: CTile, axis_map: str,
) -> tuple[float, float, float, float] | None:
    """Convert CTile bounding volume to Blender XZ AABB.

    Returns ``(min_x, max_x, min_z, max_z)`` or *None* if the bbox is
    invalid/zero (in which case the tile should NOT be skipped).
    """
    bb = tile.boxBoundingVolume
    if not bb or len(bb) < 12 or all(v == 0 for v in bb):
        return None

    cx, cy, cz = bb[0], bb[1], bb[2]
    # Half-extents: for axis-aligned boxes the off-diagonal elements are 0,
    # but handle the general OBB case conservatively.
    hx = abs(bb[3]) + abs(bb[4]) + abs(bb[5])
    hy = abs(bb[6]) + abs(bb[7]) + abs(bb[8])
    hz = abs(bb[9]) + abs(bb[10]) + abs(bb[11])

    if axis_map == "XnY":
        return (cx - hx, cx + hx, -cy - hy, -cy + hy)
    elif axis_map == "XZ":
        return (cx - hx, cx + hx, cz - hz, cz + hz)
    elif axis_map == "XnZ":
        return (cx - hx, cx + hx, -cz - hz, -cz + hz)
    else:  # "XY" or default
        return (cx - hx, cx + hx, cy - hy, cy + hy)


def refine_by_mask_sync(
    context: bpy.types.Context,
    masks: list[bpy.types.Object],
    root_tile: CTile,
    glb_dir: str,
    target_level: int,
    max_steps: int = 5000,
    on_tile_loaded=None,
) -> None:
    """Synchronous refine-by-mask-to-target-level with geographic pre-filtering.

    Key optimisations over the naive approach:

    1. **CTile bbox pre-filter** — before importing a child GLB, its
       ``boxBoundingVolume`` is tested against the mask AABB in Blender XZ.
       Children that don't overlap are never loaded, saving expensive
       glTF imports.
    2. **Active-tile tracking** — only tiles loaded in the *previous* round
       are candidates for further refinement.  The full scene is scanned
       only once (initial round); subsequent rounds operate exclusively on
       the tracked set.
    3. **O(1) tile index** — a ``content_key → CTile`` dict replaces the
       recursive ``root_tile.find()`` calls.
    """

    def _objname_to_tile_key(obj_name: str) -> str:
        return obj_name.split(".glb")[0]

    def _dedupe_tiles(tiles: list[CTile]) -> list[CTile]:
        seen: set[str] = set()
        out: list[CTile] = []
        for t in tiles:
            if t is None:
                continue
            key = t.content or str(id(t))
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    # ------------------------------------------------------------------
    # 1. Build mask cache & extract overall AABB
    # ------------------------------------------------------------------
    mask_cache, stats = build_mask_cache(context, masks)
    if not mask_cache:
        print("[refine_by_mask_sync] No valid mask mesh data, abort.")
        return
    print(f"[refine_by_mask_sync] mask_cache built: {stats}")

    mask_aabb = _mask_overall_aabb(mask_cache)
    if mask_aabb is None:
        print("[refine_by_mask_sync] No mask AABB, abort.")
        return

    # Pad by 10% for conservative pre-filtering (handles bbox approximation)
    dx = (mask_aabb[1] - mask_aabb[0]) * 0.1
    dz = (mask_aabb[3] - mask_aabb[2]) * 0.1
    mask_aabb_padded = (
        mask_aabb[0] - dx, mask_aabb[1] + dx,
        mask_aabb[2] - dz, mask_aabb[3] + dz,
    )
    print(f"[refine_by_mask_sync] mask_aabb={mask_aabb}, padded={mask_aabb_padded}")

    # ------------------------------------------------------------------
    # 2. Build CTile content index for O(1) lookup
    # ------------------------------------------------------------------
    tile_index: dict[str, CTile] = {}
    _build_tile_index(root_tile, tile_index)
    print(f"[refine_by_mask_sync] tile_index: {len(tile_index)} entries")

    # ------------------------------------------------------------------
    # 3. Detect axis mapping: CTile bbox → Blender XZ
    # ------------------------------------------------------------------
    axis_map = _detect_axis_mapping(context, tile_index)
    print(f"[refine_by_mask_sync] axis_map: {axis_map}")

    # ------------------------------------------------------------------
    # 4. Initial round: scan loaded base tiles for mask overlap
    #    (this is the only round that scans all scene objects)
    # ------------------------------------------------------------------
    hit_objs, sel_stats = objects_hit_by_mask_cache_xz(
        context=context,
        masks=masks,
        mask_cache=mask_cache,
        select_hit_objects=False,
        deselect_first=False,
        set_active=False,
    )

    active_tiles: list[CTile] = []
    for obj in hit_objs:
        key = _objname_to_tile_key(getattr(obj, "name", "") or "")
        tile = tile_index.get(key)
        if tile is not None:
            active_tiles.append(tile)
    active_tiles = _dedupe_tiles(active_tiles)

    tiles_to_refine = [t for t in active_tiles
                       if t.canRefine and t.meshLevel < target_level]
    print(f"[refine_by_mask_sync] Initial: {len(hit_objs)} hit objects, "
          f"{len(active_tiles)} tiles, {len(tiles_to_refine)} need refine")

    if not tiles_to_refine:
        print("[refine_by_mask_sync] No tiles need refinement, done.")
        return

    # ------------------------------------------------------------------
    # 5. Iterative refinement with CTile bbox pre-filter
    # ------------------------------------------------------------------
    step = 0
    round_num = 0
    total_loaded = 0
    total_skipped = 0
    t_start = time.time()

    while tiles_to_refine and step < max_steps:
        round_num += 1
        tiles_to_refine.sort(key=lambda t: (t.meshLevel, t.content or ""))
        round_count = len(tiles_to_refine)
        round_t0 = time.time()
        min_lv = tiles_to_refine[0].meshLevel
        remaining_levels = target_level - min_lv
        print(f"[refine_by_mask_sync] Round {round_num}: "
              f"{round_count} tiles to refine "
              f"(level {min_lv}→{min_lv + 1}, "
              f"{remaining_levels} level{'s' if remaining_levels > 1 else ''} to target)")

        next_candidates: list[CTile] = []

        for i, tile_to_refine in enumerate(tiles_to_refine):
            step += 1
            if step > max_steps:
                print(f"[refine_by_mask_sync] Max steps ({max_steps}), stopping.")
                return

            plan = compute_one_tile_refine_plan(tile_to_refine)
            if not plan:
                tile_to_refine.canRefine = False
                continue

            # Classify children: overlapping mask → refine further;
            # non-overlapping → load only (preserve scene completeness).
            # ALL children must be loaded before the parent is deleted,
            # otherwise boundary tiles disappear leaving holes.
            refine_plan: dict[int, list[CTile]] = {}   # overlap mask → load + track
            boundary_plan: dict[int, list[CTile]] = {} # no overlap  → load only
            n_refine = 0
            n_boundary = 0
            for level, tiles in plan.items():
                ref_list: list[CTile] = []
                bnd_list: list[CTile] = []
                for child_tile in tiles:
                    child_aabb = _tile_aabb_xz(child_tile, axis_map)
                    if child_aabb is not None and not _bbox2_overlap(child_aabb, mask_aabb_padded):
                        bnd_list.append(child_tile)
                        n_boundary += 1
                    else:
                        ref_list.append(child_tile)
                        n_refine += 1
                if ref_list:
                    refine_plan[level] = ref_list
                if bnd_list:
                    boundary_plan[level] = bnd_list

            total_loaded += n_refine + n_boundary
            total_skipped += n_boundary  # counted as "boundary" in stats

            # Merge both plans into a single load plan
            full_plan: dict[int, list[CTile]] = {}
            for level in set(list(refine_plan.keys()) + list(boundary_plan.keys())):
                full_plan[level] = refine_plan.get(level, []) + boundary_plan.get(level, [])

            if not full_plan:
                tile_to_refine.canRefine = False
                continue

            # Progress with ETA (every 50 tiles or at start/end)
            if i == 0 or (i + 1) % 50 == 0 or i + 1 == round_count:
                elapsed = time.time() - round_t0
                rate = (i + 1) / max(elapsed, 0.01)
                eta = (round_count - i - 1) / max(rate, 0.001)
                print(f"[refine_by_mask_sync]   [{i+1}/{round_count}] "
                      f"refine L{tile_to_refine.meshLevel} "
                      f"load={n_refine} boundary={n_boundary} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            load_glb_tiles_by_dic_level_array(glb_dir, full_plan,
                                              on_tile_loaded=on_tile_loaded)
            clear_scene_by_tile(tile_to_refine)

            # Only mask-overlapping children are candidates for deeper refinement
            for level, tiles in refine_plan.items():
                next_candidates.extend(tiles)

        round_elapsed = time.time() - round_t0
        print(f"[refine_by_mask_sync] Round {round_num} done in {round_elapsed:.1f}s "
              f"(loaded={total_loaded}, skipped={total_skipped})")

        # Prepare next round: only tiles that can and need further refinement
        next_candidates = _dedupe_tiles(next_candidates)
        tiles_to_refine = [t for t in next_candidates
                           if t.canRefine and t.meshLevel < target_level]

    total_elapsed = time.time() - t_start
    print(f"[refine_by_mask_sync] Complete. rounds={round_num} steps={step} "
          f"loaded={total_loaded} skipped={total_skipped} "
          f"total_time={total_elapsed:.1f}s")


def refine_and_selected_tiles(root:CTile, path):
    tiles = get_selected_ctiles(root)
    if len(tiles) == 0:
        return
    
    #remove tiles that can not refine
    can_refine_tiles = []
    for tile in tiles:
        if tile.canRefine:
            can_refine_tiles.append(tile)
    
    tiles_need_load = {}
    for root_tile in can_refine_tiles:
        tiles_need_child = [root_tile]
        while len(tiles_need_child) > 0:
            #find children
            next_level_tiles = []
            for tile in tiles_need_child:
                if tile.hasMesh:
                    if tile.canRefine == False or tile.meshLevel>root_tile.meshLevel or len(tile.children) == 0:
                        level = tile.meshLevel
                        if level not in tiles_need_load:
                            tiles_need_load[level] = []
                        if tile not in tiles_need_load[level]:
                            tiles_need_load[level].append(tile)
                        continue
                    
                if tile.canRefine and len(tile.children):
                    next_level_tiles += tile.children
            tiles_need_child = next_level_tiles
    
    #load new tiles
    load_glb_tiles_by_dic_level_array(path, tiles_need_load)
    
    #remove old tileser
    for tile in can_refine_tiles:
        clear_scene_by_tile(tile)

#根据选择的tile加载更精细的tiles
class SAM3_OT_refine_selected_tiles(bpy.types.Operator):
    """ """

    bl_idname = "sam3.refine_selected_tiles"
    bl_label = "Refine Selected Tiles"
    bl_options = {"REGISTER", "UNDO"}


    def execute(self, context: bpy.types.Context):
        root_tile = CTile()
        root_tile.loadFromRootJson(os.path.join(BASE_TILES_DIR, "tileset.json"))
        refine_and_selected_tiles(root_tile, GLB_DIR)
        return {"FINISHED"}

#根据选择的mask对象，加载更精细的tile，直到达到目标级别
class SAM3_OT_refine_by_mask_to_target_level(_SAM3_NonBlockingModalMixin, bpy.types.Operator):
    """ """

    bl_idname = "sam3.refine_by_mask_to_target_level"
    bl_label = f"Refine by Mask to Target Level {TARGET_FINE_LEVEL}"
    bl_options = {"REGISTER", "UNDO"}


    # 说明：
    # - 该操作原实现为同步大循环，会长时间占用主线程导致 UI 假死。
    # - 这里改为 modal + TIMER 分片执行：每次 TIMER 只 refine 1 个 tile，处理完即触发 redraw 并返回 RUNNING_MODAL。

    _phase: str = "INIT"  # INIT / SELECT / REFINE / DONE
    _root_tile: CTile | None = None
    _masks: list[bpy.types.Object] | None = None
    _mask_cache = None
    _refined_steps: int = 0
    _max_steps: int = 5000
    _need_refine_queue: list[CTile] = []
    _last_select_stats: str = ""
    _progress_total: int = 0
    _progress_done: int = 0

    def _log(self, msg: str) -> None:
        print(f"[SAM3][refine_by_mask] {msg}")

    def _objname_to_tile_key(self, obj_name: str) -> str:
        # import 时主对象名为 "<xxx>.glb"，其余重复/子对象会变成 "<xxx>.glb.<something>"
        return obj_name.split(".glb")[0]

    def _dedupe_tiles(self, tiles: list[CTile]) -> list[CTile]:
        seen: set[str] = set()
        out: list[CTile] = []
        for t in tiles:
            if t is None:
                continue
            key = t.content or str(id(t))
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    def _compute_tiles_need_load_for_one(self, root_tile_to_refine: CTile) -> dict[int, list[CTile]]:
        """Delegates to the module-level function ``compute_one_tile_refine_plan``."""
        return compute_one_tile_refine_plan(root_tile_to_refine)

    def _start(self, context: bpy.types.Context) -> set[str]:
        self._log(f"开始执行（非阻塞）：Refine by Mask to Target Level={TARGET_FINE_LEVEL}")

        # 0) 确保在 Object 模式
        try:
            if getattr(context, "mode", "") != "OBJECT":
                self._log(f"切换模式：{getattr(context, 'mode', '')} -> OBJECT")
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as e:
            self._log(f"切换到 OBJECT 模式失败（继续尝试执行）：{e}")

        # 1) 读取 tileset 树
        tileset_path = os.path.join(BASE_TILES_DIR, "tileset.json")
        self._log(f"加载 tileset：{tileset_path}")
        root_tile = CTile()
        root_tile.loadFromRootJson(tileset_path)
        self._root_tile = root_tile
        self._log("tileset 加载完成")

        # 2) 获取 mask 对象（来自当前选择）
        masks = get_mask_objects(context)
        if not masks:
            self._log("没有 mask 对象，取消执行")
            self._phase = "DONE"
            return self._nb_finish(context, cancelled=True, msg="未选中 mask 对象（请先选择 mask 对象）")
        self._masks = masks
        self._log(f"mask 对象数量={len(masks)}，names={[getattr(m,'name','') for m in masks]}")

        # 3) 预计算 mask 三角形投影缓存（避免每轮重复三角化）
        mask_cache, mask_stats = build_mask_cache(context, masks)
        self._mask_cache = mask_cache
        self._log(f"mask_cache 构建：{mask_stats}")
        if not mask_cache:
            self._log("mask_cache 为空，取消执行")
            self._phase = "DONE"
            return self._nb_finish(context, cancelled=True, msg="mask 对象没有有效网格面（需要 MESH 且有面）")

        # 4) 初始化状态机
        self._phase = "SELECT"
        self._refined_steps = 0
        self._need_refine_queue = []
        self._last_select_stats = ""
        self._progress_total = 0
        self._progress_done = 0

        # 5) 启动 TIMER（0.01~0.1 秒都可；这里取 0.05 兼顾响应/开销）
        self._nb_progress_begin(context, total=1)
        self._nb_start_timer(context, interval_sec=0.05)
        self._nb_request_redraw(context)
        self.report({"INFO"}, "Refine by mask started (non-blocking). Press ESC to cancel.")
        return {"RUNNING_MODAL"}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        return self._start(context)

    def execute(self, context: bpy.types.Context):
        # 支持从脚本/搜索直接执行：也走非阻塞模式
        return self._start(context)

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        if event.type in {"ESC"}:
            self._phase = "DONE"
            return self._nb_finish(context, cancelled=True, msg="用户取消（ESC）")

        # 只在 TIMER 时做一步，避免对交互事件造成干扰
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        # 状态尚未初始化（理论上不应发生）
        if self._phase in {"INIT", "DONE"}:
            return {"PASS_THROUGH"}

        # 防止意外死循环
        self._refined_steps += 1
        if self._refined_steps > self._max_steps:
            self._phase = "DONE"
            return self._nb_finish(context, cancelled=True, msg=f"达到最大步数保护 max_steps={self._max_steps}，中止（可能无法收敛）")

        try:
            # SELECT：重新框选，并生成本轮待 refine 队列（只生成，不批量 refine）
            if self._phase == "SELECT":
                assert self._root_tile is not None
                assert self._masks is not None

                self._log(f"---- 循环 #{self._refined_steps}：用 mask 重新框选对象 ----")
                hit_objs, sel_stats = objects_hit_by_mask_cache_xz(
                    context=context,
                    masks=self._masks,
                    mask_cache=self._mask_cache,
                    select_hit_objects=True,
                    deselect_first=True,
                    set_active=True,
                )
                self._last_select_stats = str(sel_stats)

                if not hit_objs:
                    self._phase = "DONE"
                    return self._nb_finish(context, cancelled=False, msg="当前 mask 未命中任何对象，结束")

                # 映射到 CTile
                hit_tiles: list[CTile] = []
                for obj in hit_objs:
                    key = self._objname_to_tile_key(getattr(obj, "name", "") or "")
                    tile = self._root_tile.find(key) if key else None
                    if tile is None:
                        continue
                    hit_tiles.append(tile)

                hit_tiles = self._dedupe_tiles(hit_tiles)
                need_refine = [t for t in hit_tiles if t.canRefine and t.meshLevel < TARGET_FINE_LEVEL]
                need_refine.sort(key=lambda t: (t.meshLevel, t.content or ""))

                if len(need_refine) == 0:
                    self._phase = "DONE"
                    return self._nb_finish(context, cancelled=False, msg=f"所有命中 tiles 都已达到 target_level={TARGET_FINE_LEVEL} 或无法 refine，结束")

                # 设置本轮队列：后续每个 TIMER 只处理 1 个 tile
                self._need_refine_queue = list(need_refine)
                self._progress_total = len(self._need_refine_queue)
                self._progress_done = 0
                # 重新开始进度条（避免多轮 select/refine 时显示异常）
                self._nb_progress_end(context)
                self._nb_progress_begin(context, total=max(1, self._progress_total))

                self._log(f"本轮命中 tiles={len(hit_tiles)}，待 refine tiles={len(self._need_refine_queue)}，进入逐 tile refine")
                self._phase = "REFINE"
                self._nb_request_redraw(context)
                return {"RUNNING_MODAL"}

            # REFINE：每次只 refine/加载/清理 1 个 tile
            if self._phase == "REFINE":
                if len(self._need_refine_queue) == 0:
                    # 本轮处理完，回到 SELECT 做下一轮框选
                    self._phase = "SELECT"
                    self._nb_request_redraw(context)
                    return {"RUNNING_MODAL"}

                tile_to_refine = self._need_refine_queue.pop(0)
                plan = self._compute_tiles_need_load_for_one(tile_to_refine)
                if len(plan) == 0:
                    # 理论上不应发生（canRefine==True），但为了稳健性做保护，避免死循环
                    self._log(f"WARNING：tile={tile_to_refine.content} 计划为空，标记不可 refine 并跳过")
                    tile_to_refine.canRefine = False  # type: ignore[assignment]
                else:
                    # 1) 加载该 tile 对应的更精细 glb（只处理这一块）
                    self._log(f"[{self._progress_done+1}/{self._progress_total}] refine tile={tile_to_refine.content} -> load+clear")
                    load_glb_tiles_by_dic_level_array(GLB_DIR, plan)

                    # 2) 清理旧 tile（只清理这一块）
                    clear_scene_by_tile(tile_to_refine)

                # 更新进度 + 刷新 UI
                self._progress_done += 1
                self._nb_progress_update(context, min(self._progress_done, max(1, self._progress_total)))

                self._nb_request_redraw(context)
                return {"RUNNING_MODAL"}

        except Exception as e:
            self._phase = "DONE"
            return self._nb_finish(context, cancelled=True, msg=f"执行异常中止：{e}")

        return {"RUNNING_MODAL"}

ACTION_SPECS = [
    ActionSpec(
        operator_cls=SAM3_OT_load_base_tiles,
        menu_label=f"Load Base Tiles Level {BASE_LEVEL}",
        icon="TOOL_SETTINGS",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_refine_selected_tiles,
        menu_label="Refine Selected Tiles",
        icon="TOOL_SETTINGS",
    ),
    ActionSpec(
        operator_cls=SAM3_OT_refine_by_mask_to_target_level,
        menu_label=f"Refine by Mask to Target Level {TARGET_FINE_LEVEL}",
        icon="TOOL_SETTINGS",
    ),
]