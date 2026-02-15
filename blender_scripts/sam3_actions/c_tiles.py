# CTiles 用于管理被加载到 blender 场景中的 glb 瓦片

from __future__ import annotations

import json
from typing import Optional


class CTile:
    def __init__(self) -> None:
        # to lower lod mesh
        self.parent: Optional["CTile"] = None
        # to higher lod mesh
        self.children: list["CTile"] = []
        self.geometricError = 1000000.0
        self.boxBoundingVolume = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.content: Optional[str] = None
        self.canRefine = False
        self.hasMesh = False
        self.meshLevel = 0

    print_indent = 0

    def __str__(self) -> str:
        s = ""
        for _ in range(CTile.print_indent):
            s += "--"

        s += "tile geometricError:{0} content:{1} refine:{2}\n".format(
            self.geometricError, self.content, self.canRefine
        )
        CTile.print_indent += 1
        for tile in self.children:
            s += str(tile)
        CTile.print_indent -= 1
        return s

    def position(self):
        return self.boxBoundingVolume[0:3]

    def loadFromRootJson(self, jsonFilePath: str):
        try:
            f = open(jsonFilePath, "r", encoding="utf-8")
        except OSError:
            print("Could not open/read file:", jsonFilePath)
            return

        with f:
            file_content = json.loads(f.read())
            if "root" not in file_content:
                return
            self.loadFromRootDic(file_content["root"], jsonFilePath)

    def loadFromRootDic(self, root, jsonFilePath: str = ""):
        if "geometricError" in root:
            self.geometricError = root["geometricError"]
        if "boundingVolume" in root and "box" in root["boundingVolume"]:
            self.boxBoundingVolume = root["boundingVolume"]["box"]
        if "content" in root and "uri" in root["content"]:
            uri = root["content"]["uri"]

            # read another file as content
            if uri.endswith(".json"):
                # has child json
                tile = CTile()
                # windows only
                path = uri
                path = path.replace("/", "\\")
                if jsonFilePath.rfind("\\") >= 0:
                    path = jsonFilePath[0 : jsonFilePath.rfind("\\") + 1] + path
                tile.loadFromRootJson(path)
                self.children.append(tile)
                tile.parent = self
            else:
                # tile as content
                self.content = uri
                self.hasMesh = True
                level = 0
                for text in uri.split("_"):
                    if text.find("L") == 0:
                        level = int(text[1:])
                self.meshLevel = level

        if "children" in root:
            for child in root["children"]:
                # create child
                tile = CTile()
                # windows only
                tile.loadFromRootDic(child, jsonFilePath)
                self.children.append(tile)
                tile.parent = self

        if "refine" in root and root["refine"] == "REPLACE" and len(self.children):
            self.canRefine = True

    def find(self, name: str):
        if self.content and self.content.split(".")[0] == name:
            return self
        for child in self.children:
            ret = child.find(name)
            if ret:
                return ret
        return None

    def allChildren(self):
        if len(self.children) == 0:
            return []
        all_tiles: list[CTile] = []
        all_tiles.extend(self.children)
        for child in self.children:
            all_tiles.extend(child.allChildren())
        return all_tiles

    def canRefineBy(self, tiles):
        if len(tiles) == 0:
            return False
        if self in tiles:
            return True
        if self.canRefine is False:
            return False
        if len(self.children) == 0:
            return False

        for child in self.children:
            if child.canRefineBy(tiles) is False:
                return False

        return True





