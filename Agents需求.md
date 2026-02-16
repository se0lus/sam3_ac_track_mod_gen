项目总体目标：自动化的通过无人采集，经过重建的赛道2D图像/3D模型，使用blender工具和其他AI等，自动化的生成一个可玩的assetto corsa 赛道mod；

将任务分解给多个agent并行执行；需要有合理的分工，建议使用agent teams；
每个模块任务开始执行之前，都需要先完成编码计划，由架构师角色的agent进行review完成之后再动手
每个模块的功能都需要单独可测；耦合blender的功能要尽可能独立，使得大部分代码可以脱离blender进行测试
云端大模型可以使用gemini, 具体的模型可以使用pro-latest，key为：***REDACTED_GEMINI_KEY***

目前项目已经完成的部分如PROJECT.md所介绍；

需要进一步完善的部分：
TODO-1. B3DM 转换模块，无人机原始重建完成的数据类似：E:\sam3_track_seg\test_images_shajing\b3dm，在被blender使用前需要转换成glb格式，参考代码可在old_blender_scripts_example.py中找到

在最终运行的工作流中，完成1之后，会使用projectmd中提到的已有工具进行：
GeoTIFF 影像
    │
    ▼
┌─────────────────────────────┐
│  1. mask_full_map()         │  对全幅地图做初步 SAM3 分割，生成全图 mask
│  2. clip_full_map()         │  将 mask 区域智能切分为若干 clip 小图
│  3. generate_mask_on_clips()│  对每个 clip 逐标签做精细 SAM3 分割
└─────────────────────────────┘
    │  输出: clips/{tag}/clip_N_masks.json （像素/地理坐标多边形）
    ▼
┌─────────────────────────────┐
│  4. convert_mask_to_blender │  读取 3D Tiles tileset.json 中的坐标原点，
│     _input()                │  将地理坐标多边形转换为 Blender 本地坐标（TODO-2：这里需要把road，sand，kerb，grass四类clip分别合成一个大的整合clip，例如road_clip，以便后续步骤使用）
└─────────────────────────────┘
    │  输出: blender_clips/{tag}/clip_N_blender.json
    ▼
┌─────────────────────────────┐
│  5. blender_create_polygons │  在 Blender 中批量读取 *_blender.json，
│     .py                     │  生成 2D Curve + Mesh，保存为 .blend 文件（TODO-3：这里生成的blender文件中生成的mask_polygon_collection下的对象似乎是curve，而不是mesh，如果不在blender中手动转换成mesh的话，后面一步SAM3_OT_refine_by_mask_to_target_level 会无法执行）
└─────────────────────────────┘
    │  输出: polygons.blend
    ▼
┌─────────────────────────────┐
│  6. Blender 交互式操作       │ 
│     (blender_helpers +      │ 
│      sam3_actions/*)        │ 
└─────────────────────────────┘
6.1：使用SAM3_OT_load_base_tiles将基础3D文件整个加载出来
6.2：使用SAM3_OT_refine_by_mask_to_target_level，搭配road类型的mask，将road部分的瓦片提升到config所示的至少22级
6.3：TODO-4：制作一个工具，用于将mask所投影的赛道区域提取出来，形成一个大的整体对象，可以通过road类型的mask，通过设定一定的采样密度，在y方向上投影到3D瓦片的表面，用于重建出整个赛道的表面；需要注意的是边缘的部分需要完全匹配mask
6.4：以此类推，利用类似的mask生成grass，kerb，sand对象，用于不同材质的物理表现；相比之下grass和sand的采样精度可以低一些

TODO-5：接下来是一个比较独立的功能，需要利用大模型自动生成赛道的虚拟围墙边界，用于防止赛车开出地图范围，这里也是建议利用现有工具，把2D赛道全图缩放到大模型合适的尺寸，然后利用大模型生成2D的json（应该是一些线对象），经过可视化和用户确认后再开发一个blender的action把他加载进去，变成一系列很高的没有厚度的面，仅用于碰撞检测；
虚拟围墙的设定应该是围绕整个赛道可行驶区域，形成一个闭合的多边形，确保开不出去，边界的设定应该贴着缓冲区外围（可能有树木或者现实中围墙）；另外在这个虚拟围墙之内，也需要添加必要的内部围墙避免赛车开进不必要的区域；
在生成了围墙之后，还需要生成一个地面网格，其外边缘与外层围墙对齐，内边缘与road，grass，kerb，sand这四类生成的碰撞表面对齐，这样可以形成一个托底的大表面，避免车辆从没定义的地面处掉出赛道，网格可以比较粗糙，但是边缘处需要精细的对齐已有的其他表面


TODO-6：以上生成的用于碰撞的对象，在blender中需要单独放置在一个Collection中，并且名名规范类似下表，以便游戏读取：
1WALL_0
1WALL_1
...
1ROAD_0
1ROAD_1
...
1SAND_1
1KERB_0

7.
TODO-7：利用大模型生成一些游戏内会使用的虚拟不可见对象, 这些对象不需要网格，同样的可以先用2D json点基于2D地图进行可视化，用户确认后再进入blender中用action读取生成，所有的对象Z为行驶方向，y为朝上方向；高度为赛道表面之上2单位。在输出这些对象前可以由用户定义赛道是顺时针跑还是逆时针跑。给大模型的参考可以是track的mask以及2D全图；至少需要有的对象有：
AC_HOTLAP_START_0：只有一个，放在起点线前面的一个弯道的出弯处
AC_PIT_0：维修区，一般会放置8个或更多，例如AC_PIT_7；
AC_START_0，AC_START_1，... 静止起步时候的发车起点，发车格数量需要与pit数量匹配
AC_TIME_0_L，AC_TIME_0_R；起点记时点的左边界和右边界。一般会放多个记时路段，每个组合弯为一个记时点，第二个记时点就会类似AC_TIME_1_L，AC_TIME_1_R；


8.
TODO-8：Blender 解包和转换纹理为png工具，这个工具需要类似blender_scripts目录中的工具一样放在blender右键菜单中；具体的实现可以参考old_blender_scripts_example.py中的
unpack_textures()
convert_jpg_png()
convert_all_materials_to_bsdf()
需要注意的是外部引用的PIL库可能在blender环境中不一定能用，需要看看如何处理

9.
TODO-9: 在以上工作都完成了之后，修改sam3_track_gen.py以及其他的必要文件, 使其具备端到端生成最终blend文件的能力，并且使用存放在 test_images_shajing 处的原始数据进行测试；所有的测试中间文件和结果都输出到output目录下；

现在，在项目内开始创建必要的claude配置和Agent Teams；进行合理的分工，使用delegate模式；
对每个Agent的要求是各自的部分都需要具备充分的模块测试；在所有模块测试都跑通完成了之后才可以进行集成；