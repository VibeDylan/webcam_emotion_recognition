"""Paths and helpers for the FER-style dataset (one folder per emotion class)."""
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_classes(train_dir: Path):
    """Return sorted list of subdirectory names (one per class) in train_dir."""
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes

def images_in_class(train_dir: Path, cls: str):
    """Return list of image paths in train_dir/cls with extension in IMG_EXTS."""
    class_dir = train_dir / cls
    if not class_dir.is_dir():
        return []
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
