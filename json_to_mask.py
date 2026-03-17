import os
import json
import numpy as np
import cv2
from glob import glob

def create_mask_from_json(json_path, save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    img_h = data.get('imageHeight')
    img_w = data.get('imageWidth')
    
    if img_h is None or img_w is None: return False
        
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    shapes = data.get('shapes', [])
    # Step 1: Fill all polygons solid white
    for shape in shapes:
        if shape['shape_type'] != 'polygon': continue
        pts = np.array(shape['points'], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    
    # Step 2: Draw BLACK outlines on top of all polygons to create boundary lines
    # This teaches the model where one field ends and the next begins
    for shape in shapes:
        if shape['shape_type'] != 'polygon': continue
        pts = np.array(shape['points'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=True, color=0, thickness=3)
        
    cv2.imwrite(save_path, mask)
    return True

def main():
    json_dir = "india_dataset/labels_json"
    mask_dir = "india_dataset/masks"
    os.makedirs(mask_dir, exist_ok=True)
    
    json_files = glob(os.path.join(json_dir, "*.json"))
    if not json_files: return
        
    print(f"Found {len(json_files)} annotation files. Generating masks...")
    for json_file in json_files:
        filename = os.path.basename(json_file)
        mask_filename = filename.replace('.json', '.png')
        save_path = os.path.join(mask_dir, mask_filename)
        if create_mask_from_json(json_file, save_path):
            print(f"Generated mask: {mask_filename}")

if __name__ == "__main__":
    main()
