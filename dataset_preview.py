from pathlib import Path
import random
import cv2

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_classes(train_dir: Path):
    """Retourne une liste triée des noms de classes."""
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes


def images_in_class(train_dir: Path, cls: str):
    """Retourne une liste de chemins d'images pour une classe donnée."""
    class_dir = train_dir / cls
    if not class_dir.is_dir():
        return []
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]


if __name__ == "__main__":
    classes = list_classes(TRAIN_DIR)
    print("Classes:", classes)
    print("Nb classes:", len(classes))

    total = 0
    for cls in classes:
        imgs = images_in_class(TRAIN_DIR, cls)
        print(f"{cls:10s} -> {len(imgs)}")
        total += len(imgs)
    print("Total train images:", total)

    cls = random.choice(classes)
    imgs = images_in_class(TRAIN_DIR, cls)
    if imgs:
        sample_path = random.choice(imgs)
        print("Sample:", sample_path, "label:", cls)

        img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print("Shape:", img.shape, "dtype:", img.dtype)
            cv2.imshow(f"sample: {cls}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
