from pathlib import Path
import random
import copy
from dataset_preview import list_classes, images_in_class

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"


def build_samples(train_dir: Path):
    """
    Construit un index de tous les échantillons d'entraînement.
    
    Returns:
        samples: Liste de tuples (image_path, class_id)
        class_to_id: Dictionnaire {'angry': 0, 'disgust': 1, ...} dans l'ordre trié
    """
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
    """
    Divise les échantillons en train et validation de manière reproductible.
    
    Args:
        samples: Liste de tuples (image_path, class_id)
        val_ratio: Proportion des données pour la validation (défaut: 0.1)
        seed: Graine pour la reproductibilité (défaut: 42)
    
    Returns:
        train_samples: Liste des échantillons d'entraînement
        val_samples: Liste des échantillons de validation
    """
    samples_copy = copy.copy(samples)
    
    rng = random.Random(seed)
    
    rng.shuffle(samples_copy)
    
    val_size = int(len(samples_copy) * val_ratio)
    
    val_samples = samples_copy[:val_size]
    train_samples = samples_copy[val_size:]
    
    return train_samples, val_samples


if __name__ == "__main__":
    samples, class_to_id = build_samples(TRAIN_DIR)
    train_samples, val_samples = split_samples(samples)
    id_to_class = {v: k for k, v in class_to_id.items()}
    
    print("Mapping classe → ID:")
    for class_name, class_id in sorted(class_to_id.items(), key=lambda x: x[1]):
        print(f"  {class_name:15s} -> {class_id}")
    
    print(f"\n{'='*60}")
    print("SPLIT TRAIN/VALIDATION")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Val ratio: {len(val_samples) / len(samples):.2%}")
    
    print(f"\nPremier échantillon TRAIN:")
    if train_samples:
        path, class_id = train_samples[0]
        print(f"  {path.name:30s} -> classe: {id_to_class[class_id]} (id: {class_id})")
    
    print(f"\nPremier échantillon VAL:")
    if val_samples:
        path, class_id = val_samples[0]
        print(f"  {path.name:30s} -> classe: {id_to_class[class_id]} (id: {class_id})")
    for i, (path, class_id) in enumerate(samples[:5], 1):
        print(f"  {i}. {path.name:30s} -> classe: {id_to_class[class_id]} (id: {class_id})")
