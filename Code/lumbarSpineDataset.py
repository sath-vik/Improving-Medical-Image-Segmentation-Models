# lumbar_spine_dataset.py
import os
import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob #Import glob

class LumbarSpineDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, mask_ids, img_dir, mask_dir, transform=None):
        self.img_ids = img_ids
        self.mask_ids = mask_ids
        self.img_dir = img_dir  # Not strictly needed, but kept for consistency
        self.mask_dir = mask_dir  # Not strictly needed
        self.transform = transform
        #No need to add any extra check as you will make it correct in train.py

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.img_ids[idx]  # img_id is now the FULL PATH
        mask_path = self.mask_ids[idx] # mask_id is now the FULL PATH

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        if img.size == 0:
            raise ValueError(f"Image has zero size: {img_path}")
        if mask.size == 0:
            raise ValueError(f"Mask has zero size: {mask_path}")

        mask = mask[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255.0
        mask = mask.transpose(2, 0, 1)

        if mask.max() < 1:  # Ensure mask is binary
            mask[mask > 0] = 1.0

        return img, mask, {'img_id': img_path} # Return full path