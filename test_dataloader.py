from torch.utils.data import DataLoader
from build_index import build_samples, split_samples, TRAIN_DIR
from fer_dataset import FerDataset

samples, class_to_id = build_samples(TRAIN_DIR)
train_samples, val_samples = split_samples(samples, val_ratio=0.1, seed=42)

train_dataset = FerDataset(train_samples)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

x, y = next(iter(train_loader))
print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")
print(f"y[:10] = {y[:10]}")