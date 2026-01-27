from pathlib import Path

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_classes(train_dir: Path):
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes

def images_in_class(train_dir: Path, cls: str):
    class_dir = train_dir / cls
    if not class_dir.is_dir():
        return []
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
