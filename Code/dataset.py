import os
import cv2
import numpy as np
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, mask_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform):
        self.img_ids = img_ids
        self.mask_ids = mask_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes  # Keep this, even though we're using only one class
        self.transform = transform

        if len(self.img_ids) != len(self.mask_ids):
            raise ValueError(f"Number of image IDs ({len(self.img_ids)}) does not match number of mask IDs ({len(self.mask_ids)}).")


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        mask_id = self.mask_ids[idx]

        
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)

        mask_path = os.path.join(self.mask_dir, mask_id + self.mask_ext)

        

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")


        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path} (imread returned None)")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path} (imread returned None)")

        # Add a channel dimension to the mask: (H, W) -> (H, W, 1)

        if img.size == 0:
            raise ValueError(f"Image has zero size: {img_path}")
        if mask.size == 0:
            raise ValueError(f"Mask has zero size: {mask_path}")


        mask = mask[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)  # Now (1, H, W)

        # Ensure the mask is binary (0 or 1)
        if mask.max() < 1:
            mask[mask > 0] = 1.0
        #No need of the below line
        #mask = np.squeeze(mask, axis=0) #squeezing not needed

        return img, mask, {'img_id': img_id}