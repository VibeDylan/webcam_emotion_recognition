"""Build the list of (image_path, class_id) samples and split into train/val."""
from pathlib import Path
import random
import copy
from src.data.dataset_utils import list_classes, images_in_class
from src.data.dataset_utils import TRAIN_DIR


def build_samples(train_dir: Path):
    """Scan train_dir (one subdir per class) and return shuffled [(path, class_id), ...] and class_to_id dict."""
    classes = list_classes(train_dir)
    class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}
    samples = []
    for class_name in classes:
        class_id = class_to_id[class_name]
        for image_path in images_in_class(train_dir, class_name):
            samples.append((image_path, class_id))
    random.shuffle(samples)
    return samples, class_to_id

def split_samples(samples, val_ratio=0.1, seed=42):
    """Split samples into (train_samples, val_samples) with deterministic shuffle and given val_ratio."""
    samples_copy = copy.copy(samples)
    rng = random.Random(seed)
    rng.shuffle(samples_copy)
    val_size = int(len(samples_copy) * val_ratio)
    val_samples = samples_copy[:val_size]
    train_samples = samples_copy[val_size:]
    return train_samples, val_samples
