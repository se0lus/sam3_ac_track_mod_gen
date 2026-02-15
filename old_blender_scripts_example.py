import os
import bpy
import json
import math
from typing import List
from bpy.types import Menu

from os import listdir
from os.path import isfile, join

def convert_b3dm_to_glb(root, output):
    #path = "C:\\Users\\Ai.Chuyue\\Documents\\DJI\\DJITerra\\chuyue.ai@dji.com\\3\\models\\pc\\0\\terra_b3dms\\Block\\"
    #output = "C:\\glb_out\\"

    tile_cmd = "C:\\3dtile\\3dtile.exe -f b3dm -i {0} -o {1}"
    print("start glb convert")
    
    for path, subdirs, files in os.walk(root):
        for name in files:
            
            if not name.endswith(".b3dm"):
                continue
            
            full_path = os.path.join(path, name)
            print(full_path)
            ret = os.system(tile_cmd.format(full_path, output+name.replace(".b3dm", ".glb")))
            print(ret)
    return
        

class CTile:
    def __init__(self) -> None:
        #to lower lod mesh
        self.parent = None
        #to higer lod mesh
        self.children = []
        self.geometricError = 1000000.0
        self.boxBoundingVolume = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.content = None
        self.canRefine = False
        self.hasMesh = False
        self.meshLevel = 0

    print_indent = 0
    def __str__(self) -> str:
        s = ""
        for i in range(CTile.print_indent):
            s += "--"

        s += "tile geometricError:{0} content:{1} refine:{2}\n".format(self.geometricError, self.content, self.canRefine)
        CTile.print_indent += 1
        for tile in self.children:
            s += str(tile)
        CTile.print_indent -= 1
        return s

    def position(self):
        return self.boxBoundingVolume[0:3]

    def loadFromRootJson(self, jsonFilePath):
        try:
            f = open(jsonFilePath, 'r')
        except OSError:
            print("Could not open/read file:", jsonFilePath)
            return
        
        #print("open:", jsonFilePath)
        with f:
            file_content = json.loads(f.read())
            if "root" not in file_content:
                return
            self.loadFromRootDic(file_content["root"], jsonFilePath)

    def loadFromRootDic(self, root, jsonFilePath = ""):

            if "geometricError" in root:
                self.geometricError = root["geometricError"]
            if "boundingVolume" in root and "box" in root["boundingVolume"]:
                self.boxBoundingVolume = root["boundingVolume"]["box"]
            if "content" in root and "uri" in root["content"]:
                uri = root["content"]["uri"]

                #read another file as content

                if uri.endswith(".json"):
                    #has child json
                    tile = CTile()
                    #windows only
                    path = uri
                    path = path.replace("/","\\")
                    if jsonFilePath.rfind("\\") >=0:
                        path = jsonFilePath[0:jsonFilePath.rfind("\\")+1]+path
                    tile.loadFromRootJson(path)
                    self.children.append(tile)
                    tile.parent = self
                else:
                    #tile ad content
                    self.content = uri
                    self.hasMesh = True
                    level = 0
                    for text in uri.split("_"):
                        if text.find("L") == 0:
                            level = int(text[1:])
                    self.meshLevel = level

            if "children" in root:
                for child in root["children"]:
                    #create child
                    tile = CTile()
                    #windows only
                    tile.loadFromRootDic(child, jsonFilePath)
                    self.children.append(tile)
                    tile.parent = self
                
            if "refine" in root and root["refine"] == "REPLACE" and len(self.children):
                self.canRefine = True

    def find(self, name:str):
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
        all = []
        all.extend(self.children)
        for child in self.children:
            all.extend(child.allChildren())
        return all
    
    def canRefineBy(self, tiles):
        if len(tiles) == 0:
            return False
        if self in tiles:
            return True
        if self.canRefine == False:
            return False
        if len(self.children) == 0:
            return False
        
        for child in self.children:
            if child.canRefineBy(tiles) == False:
                return False
        
        '''
        print("{0} can refine by:".format(self.content))
        for tile in tiles:
            print("\t\t", tile.content)
        '''
        return True

#error in pixel distance
def screenGeometricError(tile:CTile, eye_pos, screen_h_res, screen_h_fov):
    position = tile.position()
    distance = math.sqrt((position[0]-eye_pos[0])**2 + (position[1]-eye_pos[1])**2 + (position[2]-eye_pos[2])**2)
    return ( tile.geometricError * screen_h_res ) / (distance * 2 * math.tan(screen_h_fov*math.pi/360.0))

#refine mesh by add custom selected tile
def refine_tiles_add_patch(root:CTile, input_tile_names:List[str], patch_names:List[str]):
    input_tiles = []
    patch = []

    #find input tiles
    for name in input_tile_names:
        tile = root.find(name)
        if tile:
            input_tiles.append(tile)
    for name in patch_names:
        tile = root.find(name)
        if tile:
            patch.append(root.find(name))

    new_tiles = []
    new_tiles.extend(input_tiles)
    for patch_tile in patch:
        if patch_tile in new_tiles:
            continue

        #if any children already in the list, then we dont need to add this one
        if patch_tile.canRefineBy(new_tiles):
            continue
        
        #add a new tile to the tileset, clean it's child and replace its parent if need
        if patch_tile.canRefine:
            if patch_tile.content:
                #remove all child
                children = patch_tile.allChildren()
                new_tiles = [t for t in new_tiles if t not in children]
            else:
                pass
                #TODO, we dont have this case for now
            
        new_tiles = addTilesReverse(new_tiles, patch_tile)
    
    new_contents = []
    for tile in new_tiles:
        if tile.content and tile.content not in new_contents:
            new_contents.append(tile.content)
    return new_contents

#refine mesh by add custom selected tile
def refine_tiles_add_patch(root:CTile, input_tile_names:List[str], patch_names:List[str]):
    input_tiles = []
    patch = []

    #find input tiles
    for name in input_tile_names:
        tile = root.find(name)
        if tile:
            input_tiles.append(tile)
    for name in patch_names:
        tile = root.find(name)
        if tile:
            patch.append(root.find(name))

    new_tiles = []
    add_tiles = []
    leaf_tiles = []
    
    new_tiles.extend(input_tiles)
    #remove duplicated
    for t in patch:
        if t not in add_tiles:
            add_tiles.append(t)
        
    #remove the tiles can converd by patch
    for tile in patch:
        if tile in new_tiles:
            add_tiles.remove(tile)        
            continue
        
        children = tile.allChildren()
        new_tiles = [t for t in new_tiles if t not in children]
    
    #get leaf tiles
    for tile in new_tiles:
        if len(tile.children) == 0:
            continue
        
        is_leaf_tile = True
        for child in tile.children:
            if child in new_tiles:
                is_leaf_tile = False
                break
        if is_leaf_tile:
            leaf_tiles.append(tile)
    
    #extend each leaf until we got all the tiles we need
    while len(add_tiles) and len(leaf_tiles):
        all_leaf_tiles = []
        all_add_tiles = []
        all_leaf_tiles.extend(leaf_tiles)
        all_add_tiles.extend(add_tiles)
        
        #print("loop")
        
        for leaf in all_leaf_tiles:
            #print("leaf:", leaf)
            
            all_children = leaf.allChildren()
            if leaf.content and len(all_children) == len(set(all_children) - set(add_tiles)):
                #this lead not need to expend
                leaf_tiles.remove(leaf)
            else:
                #split leaf
                leaf_tiles.remove(leaf)
                new_tiles.remove(leaf)
                new_tiles.extend(leaf.children)
                leaf_tiles.extend(leaf.children)
                        
                for add in all_add_tiles:
                    if add in leaf.children:
                        #got it
                        #print("need remove:", leaf_tiles," - ", add)
                        leaf_tiles.remove(add)
                        add_tiles.remove(add)
                        
    
    new_contents = []
    for tile in new_tiles:
        if tile.content and tile.content not in new_contents:
            new_contents.append(tile.content)
    return new_contents
    
def unpack_textures():
    image_counter = 0
    
    for image in bpy.data.images:
        
        new_image_name = "image_{0}".format(image_counter)
        new_image_path = "//.\\texture\\{0}".format(new_image_name)
        image_counter += 1
        
        if image.packed_file and image.type == "IMAGE":
            if image.file_format == "PNG":
                new_image_path += ".png"
            elif image.file_format == "JPEG":
                new_image_path += ".jpg"
            
            #write packed file
            image_file = open(bpy.path.abspath(new_image_path).replace("/", "\\"), "wb")
            image_file.write(image.packed_file.data)
            image_file.close()
            
            #unpack
            image.unpack(method='REMOVE')
            
            #set new path
            image.filepath = new_image_path
            image.reload()
            print("unpack image:" + new_image_path)


def convert_material(material):
    texture_node = material.node_tree.nodes.get('图像纹理')
    if not texture_node:
        return
    
    # Remove default
    if material.node_tree.nodes.find('混合着色器') >= 0:
        material.node_tree.nodes.remove(material.node_tree.nodes.get('混合着色器'))
    if material.node_tree.nodes.find('自发光(发射)') >= 0:
        material.node_tree.nodes.remove(material.node_tree.nodes.get('自发光(发射)'))
    if material.node_tree.nodes.find('透明 BSDF') >= 0:
        material.node_tree.nodes.remove(material.node_tree.nodes.get('透明 BSDF'))    
    if material.node_tree.nodes.find('光程') >= 0:
        material.node_tree.nodes.remove(material.node_tree.nodes.get('光程'))    
    material_output = material.node_tree.nodes.get('材质输出')
    
    # create new
    node_bsdf = material.node_tree.nodes.get('ShaderNodeBsdfPrincipled')
    if not node_bsdf: 
        node_bsdf = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        
    links = material.node_tree.links
    link = links.new(texture_node.outputs[0], node_bsdf.inputs[0])

    # link to material
    material.node_tree.links.new(material_output.inputs[0], node_bsdf.outputs[0])  
    print("convert material:" + material.name)
    
def convert_glb_materials():
    for material in bpy.data.materials:
        if material.node_tree and material.node_tree.nodes.find('自发光(发射)') >= 0:
            convert_material(material)


def import_glb_tiles(path, min_level = 0, max_level = 29):
    #get file list
    print("start loading glb tiles in:{0}, min_level:{1}, max_levle{2}".format(path, min_level, max_level))
    files = [f for f in listdir(path) if isfile(join(path, f))]
    
    #node map
    #node_name_map = {}
    
    for current_level in range(min_level, max_level+1):
        
        #log files from new
        objects_imported = [] 
        
        #get files in the same level
        for name in files:
            if (not name.endswith(".glb")) or name.find("_L{0}_".format(current_level))<0:
                continue

            #we need map the glb file name to the node name, so do import one by one
            print("import level {0}, {1}".format(current_level, name))
        
            #test
            #current_level = 16
            #file_names = [{"name":"Block_L16_3.glb"},{"name":"Block_L16_4.glb"}]
        
            #import gltf
            orig_objects = bpy.data.objects.keys()
            ret = bpy.ops.import_scene.gltf(filepath=path, filter_glob='*.glb;*.gltf', files=[{"name":name}], loglevel=0, import_pack_images=True)
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
    
        for obj_name in objects_imported:        
            obj = bpy.data.objects[obj_name]
            collection.objects.link(obj)
            bpy.context.scene.collection.objects.unlink(obj)
            #print("move {0} into {1}".format(obj_name, collection_name))
            
    #print(node_name_map)
    
#base on a json file, load scene at at least min level
def import_fullscene_with_ctile(root:CTile, path, min_level = 0, select_file_list_path = ""):
    #get file list
    print("start loading glb tiles in:{0}, min_level:{1}".format(path, min_level))
    files = [f for f in listdir(path) if isfile(join(path, f))]
    
    if not root:
        print("root tile = null")
        return
    
    #node map
    #node_name_map = {}
    tiles_need_child = [root]
    if(select_file_list_path):
        tiles_need_child = []
        with open(select_file_list_path) as f:
            lines = f.readlines()
            for name in lines:
                if len(name.strip()):
                    tile = root.find(name.strip())
                    if tile and tile not in tiles_need_child:
                        tiles_need_child.append(tile)
        
    tiles_need_load = {}
    print("check {0} root tiles".format(len(tiles_need_child)))
    while len(tiles_need_child) > 0:
        #find children
        next_level_tiles = []
        for tile in tiles_need_child:
            if tile.hasMesh:
                if tile.canRefine == False or tile.meshLevel>=min_level or len(tile.children) == 0:
                    level = tile.meshLevel
                    if level not in tiles_need_load:
                        tiles_need_load[level] = []
                    tiles_need_load[level].append(tile)
                    continue
                
            if tile.canRefine and len(tile.children):
                next_level_tiles += tile.children
        tiles_need_child = next_level_tiles
    
    for level in tiles_need_load:
        print("need to load level {0}:{1} tiles".format(level, len(tiles_need_load[level])))
        
    load_glb_tiles_by_dic_level_array(path, tiles_need_load)

def load_glb_tiles_by_dic_level_array(path, tiles_need_load):        
    for current_level in tiles_need_load:
        
        #log files from new
        objects_imported = [] 
        
        #get files in the same level
        for tile in tiles_need_load[current_level]:
            
            name = tile.content.replace("b3dm", "glb")
            #we need map the glb file name to the node name, so do import one by one
            print("import level {0}, {1}".format(current_level, name))
        
            #test
            #current_level = 16
            #file_names = [{"name":"Block_L16_3.glb"},{"name":"Block_L16_4.glb"}]
        
            #import gltf
            orig_objects = bpy.data.objects.keys()
            ret = bpy.ops.import_scene.gltf(filepath=path+name, filter_glob='*.glb;*.gltf', loglevel=0, import_pack_images=True)
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
    
        for obj_name in objects_imported:        
            obj = bpy.data.objects[obj_name]
            collection.objects.link(obj)
            bpy.context.scene.collection.objects.unlink(obj)
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

def print_selected_ctiles(root, path, withChildren = False):
    tiles = get_selected_ctiles(root)
    names = {}
    if len(tiles):
        f = open(path, "a")
        f.write("\n\nselect\n")
        for tile in tiles:
            f.write(tile.content.replace(".b3dm","\n"))
            if withChildren:
                children = tile.allChildren()
                for child in children:
                    if child.content and child.content not in names:
                        f.write(child.content.replace(".b3dm","\n"))
                        names[child.content]=1
        f.close()

def convert_all_materials_to_emmit():
    for material in bpy.data.materials:
        
        if not material.node_tree:
            continue
        
        #get bsdf
        bsdf = None
        for index, node in material.node_tree.nodes.items():
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break
        if not bsdf:
            continue
        
        #get image
        image = None
        for index, link in material.node_tree.links.items():
            if link.to_node != bsdf:
                #print("not to bsdf")
                continue
            if link.from_node.type != "TEX_IMAGE":
                #print("not from image:" + link.from_node.name)
                continue
            image = link.from_node
            break
        if not image:
            #print("no image")
            continue
        
        #get emmit or create one
        emmit = None
        for index, node in material.node_tree.nodes.items():
            if node.type == "EMISSION":
                emmit = node
                break
        if not emmit:
            emmit = material.node_tree.nodes.new(type='ShaderNodeEmission')
            #add image link
            material.node_tree.links.new(emmit.inputs[0], image.outputs[0])
        
        #delete link
        output = None
        for index, node in material.node_tree.nodes.items():
            if node.type == "OUTPUT_MATERIAL":
                output = node
                break
        if not output:
            continue
        
        delete_link = None
        for index, link in material.node_tree.links.items():
            if link.to_node != output:
                continue
            delete_link = link
            break
        
        if delete_link:
            material.node_tree.links.remove(delete_link)
            
        #add new link
        material.node_tree.links.new(output.inputs[0], emmit.outputs[0])
        print("convert to emmit:{0}".format(material.name))
        
def convert_all_materials_to_bsdf():
    for material in bpy.data.materials:
        
        if not material.node_tree:
            continue
        
        #get emmit
        emmit = None
        for index, node in material.node_tree.nodes.items():
            if node.type == "EMISSION":
                emmit = node
                break
        if not emmit:
            continue
        
        #get image
        image = None
        for index, link in material.node_tree.links.items():
            if link.to_node != emmit:
                #print("not to bsdf")
                continue
            if link.from_node.type != "TEX_IMAGE":
                #print("not from image:" + link.from_node.name)
                continue
            image = link.from_node
            break
        if not image:
            #print("no image")
            continue
        
        #get bsdf or create one
        bsdf = None
        for index, node in material.node_tree.nodes.items():
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break
        if not bsdf:
            bsdf = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
            #add image link
            material.node_tree.links.new(bsdf.inputs[0], image.outputs[0])
            
        
        #delete link
        output = None
        for index, node in material.node_tree.nodes.items():
            if node.type == "OUTPUT_MATERIAL":
                output = node
                break
        if not output:
            continue
        
        delete_link = None
        for index, link in material.node_tree.links.items():
            if link.to_node != output:
                continue
            delete_link = link
            break
        
        if delete_link:
            material.node_tree.links.remove(delete_link)
            
        #add new link
        material.node_tree.links.new(output.inputs[0], bsdf.outputs[0])
        print("convert to bsdf:{0}".format(material.name))

def renameMaterialToImageName():
    for material in bpy.data.materials:
        
        if not material.node_tree:
            continue
        
        #get image
        image = None
        for index, link in material.node_tree.links.items():
            if link.from_node.type != "TEX_IMAGE":
                #print("not from image:" + link.from_node.name)
                continue
            image = link.from_node
            break
        if not image:
            #print("no image")
            continue
        
        materialName = material.name
        materialName = materialName.split("_image_")[0]
        materialName = materialName.split("image_")[0]
        
        if len(image.image.filepath):
            imageName = image.image.filepath.split("\\")[-1]
            imageName = imageName.split("/")[-1]
            imageName = imageName.split(".")[0]
            if "image_" in imageName:
                materialName += "_"
                materialName += imageName
                material.name = materialName 
                
def separateMaterialByCollection(collectionName):
    #find collection
    objects = []
    for collection in bpy.data.collections:
        if collection.name == collectionName:
            objects = collection.all_objects 
    
    matToMeshDic = {}
    for obj in objects:
        if obj.type == "MESH":
            material = obj.material_slots[0]
            if material:
                if material.name not in matToMeshDic:
                    matToMeshDic[material.name] = []
                matToMeshDic[material.name].append(obj)
    
    copy_count = 0
    obj_count = 0
    
    for matName in matToMeshDic:
        oldMat = bpy.data.materials.get(matName)
        if oldMat:
            copy_count += 1
            newMat = oldMat.copy()
            newMat.name = collectionName + "_" + oldMat.name
            
            for mesh in matToMeshDic[matName]:
                obj_count += 1
                if mesh.data.materials:
                    # assign to 1st material slot
                    mesh.data.materials[0] = newMat
                else:
                    # no slots
                    mesh.data.materials.append(newMat)
    
    print("copy mat:{0}, assign obj:{1}".format(copy_count, obj_count))

def selectMeshLodByCamera():

    camera_obj = bpy.data.objects.get("Camera")
    camera = bpy.data.cameras[0]
    fovH = math.degrees(2 * math.atan(camera.sensor_height /(2 * camera.lens)))
    eye_pos = [camera_obj.location[0], camera_obj.location[1], camera_obj.location[2]]
    print("camera fovH:{0} at:{1}".format(fovH, eye_pos))
    
    root_tile = CTile()
    root_tile.loadFromRootJson("C:\\Users\\Ai.Chuyue\\Documents\\DJI\\DJITerra\\chuyue.ai@dji.com\\3\\models\\pc\\0\\terra_b3dms\\Block\\tileset.json")
    results = refine_tiles(root_tile, eye_pos, 1080, fovH, 2)
    
    #filter select list
    select_list = []
    for key in results:
        if key.rfind(".") > 0:
            select_list.append(key[0:key.find(".")])
        else:
            select_list.append(key)
    
    print(select_list)
    
    for key, object in bpy.data.objects.items():
        #print(key)
        if key.find(".glb.Mesh") < 0:
            continue

        if key[0:key.find(".glb.Mesh")] in select_list:
            #select this
            object.select_set(True)
            print("select:",key)
            
def selectMeshLodByCollection():
    root_tile = CTile()
    root_tile.loadFromRootJson("C:\\Users\\Ai.Chuyue\\Documents\\DJI\\DJITerra\\chuyue.ai@dji.com\\3\\models\\pc\\0\\terra_b3dms\\Block\\tileset.json")
    
    init_tiles = []
    for name, object in bpy.data.collections["L22_Keep"].objects.items():
        if name.find("Block_L") == 0 and name.find(".glb")>0:
            init_tiles.append(name[0:name.find(".glb")])
    
    print("init by:", init_tiles)
    results = refine_tiles_add_patch(root_tile, ["Block_L11_1"], init_tiles)
    
    #filter select list
    select_list = []
    for key in results:
        if key.rfind(".") > 0:
            select_list.append(key[0:key.find(".")])
        else:
            select_list.append(key)
    
    print(select_list)
    
    for key, object in bpy.data.objects.items():
        #print(key)
        if key.find(".glb.Mesh") < 0:
            continue

        if key[0:key.find(".glb.Mesh")] in select_list:
            #select this
            object.select_set(True)
            print("select:",key)
            
def convert_texture_to_png():
    for name, image in bpy.data.images.items():
        if image.packed_file or image.type != "IMAGE" or image.filepath.find(".png")>=0:
            continue
        image.filepath=image.filepath + ".png"
        image.reload()
        print("convert:", name)
        
def clear_scenes():
    while len(bpy.data.objects.items()):
        bpy.data.objects.remove(bpy.data.objects[0])
    while len(bpy.data.collections.items()):
        bpy.data.collections.remove(bpy.data.collections[0])
    while len(bpy.data.images.items()):
        bpy.data.images.remove(bpy.data.images[0])
    while len(bpy.data.materials.items()):
        bpy.data.materials.remove(bpy.data.materials[0])
    while len(bpy.data.meshes.items()):
        bpy.data.meshes.remove(bpy.data.meshes[0])
        
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
    
def remove_selected_material():
    # Get the active object
    obj = bpy.context.active_object

    # Loop through all material slots and remove them
    for slot in obj.material_slots:
        bpy.ops.object.material_slot_select()
        bpy.ops.object.material_slot_remove()


class WM_OT_button_context_test(bpy.types.Operator):
    """Refine Selected Tiles"""
    bl_idname = "wm.button_context_refine_selected_tiles"
    bl_label = "Refine Selected Tiles"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        root_tile = CTile()
        root_tile.loadFromRootJson("E:\\ModelParts\\shajing_blender\\b3dm\\tileset.json")
        refine_and_selected_tiles(root_tile,"E:\\ModelParts\\shajing_blender\\glb\\")
        return {'FINISHED'}


# This class has to be exactly named like that to insert an entry in the right click menu
class WM_MT_button_context(Menu):
    bl_label = "Unused"

    def draw(self, context):
        pass


def menu_func(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(WM_OT_button_context_test.bl_idname)


classes = (
    WM_OT_button_context_test,
    WM_MT_button_context,
)


def register():
    print("reg menu")
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WM_MT_button_context.append(menu_func)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    bpy.types.WM_MT_button_context.remove(menu_func)

def process_image(info):
    print("convert:" + info[0])
    im1 = Image.open(info[0])
    im1.save(info[1])
    return


def convert_jpg_png():
    path = "E:\\ModelParts\\shajing_blender\\texture\\"
    files = os.listdir(path)
    items = []

    for file in files:
        items.append([path+file, path+file+".png"])

    #use multi threads to convert images
    pool = multiprocessing.Pool()
    results = pool.map(process_image, items)
    pool.close()
    pool.join()


if __name__ == "__main__":
    register()


#1. convert b3dm to glb, this will cost about 30min to convert 40000 files
#convert_b3dm_to_glb("F:\\DJI\\fugang_model_blender\\terra_b3dms", "F:\\DJI\\fugang_model_blender\\glb_out\\")

#import scenes start from level 17
#clear_scenes()
#root_tile = CTile()
#root_tile.loadFromRootJson("E:\\ModelParts\\shajing_blender\\b3dm\\tileset.json")
#import_fullscene_with_ctile(root_tile,"E:\\ModelParts\\shajing_blender\\glb\\", 17)

#set "Refine Selected Tiles" function to a short cut, and use this short cut to refine tiles you want. the function is reg in WM_OT_button_context_test
#separate tiles to kerb, road etc... do not rename or join the mesh
#do the fix, model adjust etc...
#after all done, rotate scene 90 deg on x axis

#2. unpack all texture to textures folder, delete from blender file
#unpack_textures()

#3. convert all texture to png format
#use convert_jpg_png() in convert_jpeg_png.py, PIL library can not use in blender env
#set texture format in materials to png
#convert_texture_to_png()

#4. convert all material to bsdf format
#convert_all_materials_to_bsdf()


#7. make Road mesh, Kerb mesh, grass, ac_obj etc, every mesh should be less than 20000 triangles.(KSE will crash if not)
#8. export scene to fbx format, scale set to 0.01, use -y forward, z up， every scene should be less than 150MB

