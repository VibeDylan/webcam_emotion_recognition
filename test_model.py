from emotion_cnn import EmotionCNN
from build_index import build_samples, split_samples, TRAIN_DIR
from fer_dataset import FerDataset
from torch.utils.data import DataLoader

samples, _ = build_samples(TRAIN_DIR)
train_samples, _ = split_samples(samples)

dataset = FerDataset(train_samples)
loader = DataLoader(dataset, batch_size=32)

x, y = next(iter(loader))

model = EmotionCNN(num_classes=7)
logits = model(x)

print("x:", x.shape)
print("logits:", logits.shape)