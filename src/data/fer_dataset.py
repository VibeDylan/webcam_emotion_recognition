"""PyTorch Dataset for FER-style samples: loads 48x48 grayscale images with optional augmentation."""
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import random


class FerDataset(Dataset):
    """Dataset of (image_path, class_id) samples; returns (tensor, label) with optional augmentation."""

    def __init__(self, samples, augment=False):
        """samples: list of (path, class_id). augment: apply random flip, rotation, translation."""
        self.samples = samples
        self.augment = augment
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __augment_image(self, image):
        """Apply random horizontal flip, small rotation, and translation. image: HxW grayscale."""
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = image.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        if random.random() > 0.5:
            tx = random.randint(-2, 2)
            ty = random.randint(-2, 2)
            h, w = image.shape
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def __getitem__(self, idx):
        """Return (x, y): x (1, 48, 48) float in [0,1], y int class id."""
        image_path, label = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image.shape != (48, 48):
            image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
        
        if self.augment:
            image = self.__augment_image(image)
        
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        x = torch.from_numpy(image)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
