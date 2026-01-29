"""Load a saved emotion model (CNN or ResNet) and resolve class names from the train directory."""
import torch
from src.models.emotion_cnn import EmotionCNN
from src.models.emotion_resnet import EmotionResNet
from src.data.build_index import build_samples
from src.data.dataset_utils import TRAIN_DIR


def load_model(model_path="emotion_best.pt", model_type="cnn", num_classes=7, device="cpu"):
    """
    Load a saved model from a .pt checkpoint.

    Args:
        model_path: Path to the saved .pt file.
        model_type: "cnn" or "resnet".
        num_classes: Number of output classes (7 for emotions).
        device: "cpu" or "cuda".

    Returns:
        Model loaded with state dict, on the given device, in eval mode.
    """
    if model_type == "resnet":
        model = EmotionResNet(num_classes=num_classes, pretrained=False)
    else:  # model_type == "cnn"
        model = EmotionCNN(num_classes=num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_class_names():
    """Return id_to_class dict mapping class index to class name (from TRAIN_DIR layout)."""
    _, class_to_id = build_samples(TRAIN_DIR)
    id_to_class = {v: k for k, v in class_to_id.items()}
    return id_to_class
