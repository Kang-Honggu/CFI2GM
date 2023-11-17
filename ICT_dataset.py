import os
import cv2
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from PIL import Image

class ICTdataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.root_dir = os.path.join(opt.dataroot, opt.phase)  # e.g., train or test
        self.transform = get_transform(opt)
        self.image_pairs = self.get_image_pairs(self.root_dir)

    def get_image_pairs(self, root_dir):
        image_pairs = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                fp_files = [f for f in os.listdir(folder_path) if "_FP_" in f]
                for fp_file in fp_files:
                    patient_id, _, eye_direction, date = fp_file.split('_')
                    
                    # Check for the corresponding image for each eye direction (OS and OD)
                    for direction in ['OS', 'OD']:
                        if direction != eye_direction:
                            continue
                        corresponding_hvf_file = f"{patient_id}_HVF_{direction}_{date}"
                        if corresponding_hvf_file in os.listdir(folder_path):
                            image_pairs.append((os.path.join(folder_path, fp_file), 
                                                os.path.join(folder_path, corresponding_hvf_file)))
        return image_pairs
        
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        fp_img_path, hvf_img_path = self.image_pairs[idx]
        
        fp_image = Image.open(fp_img_path)
        hvf_img_np = cv2.imread(hvf_img_path)
        hvf_img_np = cv2.cvtColor(hvf_img_np, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        hvf_img_np = cv2.resize(hvf_img_np, (256, 256))

        overlay_img = cv2.imread('gray_cross.jpg')
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        overlay_img = cv2.resize(overlay_img, (256, 256))

        # Using addWeighted to blend the two images with the chosen weight
        blended_img = cv2.addWeighted(hvf_img_np, 0.8, overlay_img, 0.2, 0)
        hvf_image = Image.fromarray(blended_img)
        
        if self.transform:
            fp_image = self.transform(fp_image)
            hvf_image = self.transform(hvf_image)
            
        return {'A': fp_image, 'B': hvf_image, 'A_paths': fp_img_path, 'B_paths': hvf_img_path}