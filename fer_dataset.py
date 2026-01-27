from torch.utils.data import Dataset
import torch
import cv2
import numpy as np

class FerDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (Path, int)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image.shape != (48, 48):
           image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        x = torch.from_numpy(image)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
        