import os
import re
import json
import subprocess
# from time import pthread_getcpuclockid

from cv2 import threshold
import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

from geo_sam3_image import GeoSam3Image
from geo_sam3_utils2 import *
from geo_sam3_blender_utils import map_mask_to_blender

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# # image_path = f"{sam3_root}/assets/images/test_image.jpg"
# image_path = "E:\\sam3_track_seg\\test_images\\shajing_1008.jpg"
# image = Image.open(image_path)

def generate_mask_on_clips(path: str):
    inputs = [
        {
            "tag":"road",
            "prompt":"race track surface"
        },
        {
            "tag":"grass",
            "prompt":"grass"
        },
        {
            "tag":"sand",
            "prompt":"sand surface"
        },
        {
            "tag":"kerb",
            "prompt":"race track curb",
            "threshold":0.2
        }]

    #ends like clip_[n].tif , need a regex to get the [n]
    clips = [f for f in os.listdir(path) if re.match(r'clip_\d+\.tif', f)]
    if len(clips) == 0:
        print("No clips found")
        return

    print(f"Found {len(clips)} clips")
    print("start generating masks on clips")
    # load model
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = f"{sam3_root}/../model/sam3.pt"
    model = build_sam3_image_model(bpe_path=bpe_path,checkpoint_path=checkpoint_path,load_from_HF=False)

    for input in inputs:
        for clip in clips:
            #create geo_image
            geo_image = GeoSam3Image(os.path.join(path, clip))
            if geo_image.has_model_scale_image() is False:
                geo_image.generate_model_scale_image()

            image = geo_image.model_scale_image
            threshold = input.get("threshold", 0.4)
            processor = Sam3Processor(model, confidence_threshold=threshold)
            inference_state = processor.set_image(image)

            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=input["prompt"])

            #save mask with tag
            target_path = os.path.join(path, input["tag"])
            #create target path if not exists
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            geo_image.set_masks_from_inference_state(inference_state, tag=input["tag"])
            geo_image.save(save_masks=True, overwrite=True, output_dir=target_path)


def mask_full_map(src_img_file:str) -> 'GeoSam3Image':
    
    #extract src_path from src_img path
    src_path = os.path.dirname(src_img_file)
    if src_path is None:
        print("No src path found")
        return None

    geo_image = GeoSam3Image(src_img_file)
    if geo_image.has_model_scale_image() is False:
        geo_image.generate_model_scale_image()
        geo_image.save(save_masks=False)

    # 从 inference_state 中提取 masks 并设置到 geo_image 中
    if geo_image.has_masks() is False:

         # load model
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        checkpoint_path = f"{sam3_root}/../model/sam3.pt"
        model = build_sam3_image_model(bpe_path=bpe_path,checkpoint_path=checkpoint_path,load_from_HF=False)

        image = geo_image.model_scale_image
        width, height = image.size
        processor = Sam3Processor(model, confidence_threshold=0.2)
        inference_state = processor.set_image(image)

        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt="race track surface")

        geo_image.set_masks_from_inference_state(inference_state)
        geo_image.save(save_masks=True)

        img0 = image
        plot_results(img0, inference_state)

         # 美化显示并显示结果
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout()  # 自动调整布局

        # 可选：保存结果图像到文件（必须在 plt.show() 之前保存）
        save_results = True  # 设置为 False 可以跳过保存
        if save_results:
            output_dir = src_path
            save_path = os.path.join(output_dir, "results_visualization.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.1)
            print(f"结果已保存到: {save_path}")

        plt.show()  # 显示图像（在保存之后）

        # 合并后的mask
        merged_mask = geo_image.merge_all_masks(mode='union')
        if merged_mask is None:
            print("No merged mask found")
            return None
        
        merged_mask.save(os.path.join(src_path, "merged_mask.png"))
   
    return geo_image


def clip_full_map(src_img_file:str):

    geo_image = mask_full_map(src_img_file)
    if geo_image is None:
        print("No geo image found")
        return

    merged_mask = geo_image.merge_all_masks(mode='union')
    if merged_mask is None:
        print("No merged mask found")
        return

    target_clip_size_in_meters = 40
    geo_image_width_in_meters = geo_image.geo_image.get_gsd()[0] * geo_image.geo_image.width / 100
    geo_image_height_in_meters = geo_image.geo_image.get_gsd()[1] * geo_image.geo_image.height / 100

    target_clip_width_in_full_image_normalized_ratio = target_clip_size_in_meters / geo_image_width_in_meters
    target_clip_height_in_full_image_normalized_ratio = target_clip_size_in_meters / geo_image_height_in_meters
    clip_boxes = generate_clip_boxes2(merged_mask, (target_clip_width_in_full_image_normalized_ratio, target_clip_height_in_full_image_normalized_ratio), 0.1)
    #log boxes
    print(f"clip_boxes: {clip_boxes}")

    src_path = os.path.dirname(src_img_file)
    visualize_clip_boxes(merged_mask, clip_boxes, show_plot=True, save_path=os.path.join(src_path, "clip_boxes_visualization.png"))

    dst_image_path = os.path.join(src_path, "clips")
    if not os.path.exists(dst_image_path):
        os.makedirs(dst_image_path)

    #将 geo_image 用 clip_boxes 裁剪到 clips 目录中，按_clip[n].tiff 保存, GSD 为4
    for i, box in enumerate(clip_boxes):
        cropped_geo_image = geo_image.crop_and_scale_to_gsd(box, geo_image.geo_image.get_gsd()[0], dst_image_path=os.path.join(dst_image_path, f"clip_{i}.tif"))
        #create model images
        cropped_geo_image.generate_model_scale_image()
        cropped_geo_image.save(save_masks=False)


def convert_mask_to_blender_input(mask_json_file_path: str, tiles_json_path: str, output_path: str):
    #convert all mask json files under mask_json_file_path to output_path, use map_mask_to_blender method;
    
    #check output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 查找所有 *_masks.json 文件
    mask_json_files = []
    for root, dirs, files in os.walk(mask_json_file_path):
        for file in files:
            if file.endswith('_masks.json'):
                mask_json_files.append(os.path.join(root, file))
    
    if len(mask_json_files) == 0:
        print(f"在 {mask_json_file_path} 下未找到任何 *_masks.json 文件")
        return
    
    print(f"找到 {len(mask_json_files)} 个 mask json 文件")
    
    # 对每个 mask json 文件进行处理
    for mask_json_file in mask_json_files:
        try:
            print(f"处理文件: {mask_json_file}")
            
            # 调用 map_mask_to_blender 进行转换
            result = map_mask_to_blender(mask_json_file, tiles_json_path, z_mode="zero", frame_mode="auto")
            
            # 计算输出文件的相对路径
            rel_path = os.path.relpath(mask_json_file, mask_json_file_path)
            # 将文件名从 *_masks.json 改为 *_blender.json
            rel_dir = os.path.dirname(rel_path)
            base_name = os.path.basename(mask_json_file)
            output_filename = base_name.replace('_masks.json', '_blender.json')
            
            # 构建输出路径，保持相对目录结构
            if rel_dir and rel_dir != '.':
                output_dir = os.path.join(output_path, rel_dir)
            else:
                output_dir = output_path
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, output_filename)
            
            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"  已保存到: {output_file}")
            
        except Exception as e:
            print(f"处理文件 {mask_json_file} 时出错: {e}")
            continue
    
    print(f"转换完成，共处理 {len(mask_json_files)} 个文件")


#clip_full_map(src_img_file="E:\\sam3_track_seg\\test_images_shajing\\result.tif")
#generate_mask_on_clips(path="E:\\sam3_track_seg\\test_images_shajing\\clips")
convert_mask_to_blender_input(mask_json_file_path="E:\\sam3_track_seg\\test_images_shajing", tiles_json_path="E:\\sam3_track_seg\\test_images_shajing\\b3dm", output_path="E:\\sam3_track_seg\\test_images_shajing\\blender_clips")

#start blender and create mask objects
subprocess.run("E:\\SteamLibrary\\steamapps\\common\\Blender\\blender.exe --python E:\\sam3_track_seg\\blender_scripts\\blender_create_polygons.py -- --input E:\\sam3_track_seg\\test_images_shajing\\blender_clips --output E:\\sam3_track_seg\\test_images_shajing\\polygons.blend", check=True)
