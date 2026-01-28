import torch
from src.models.emotion_cnn import EmotionCNN
from src.models.emotion_resnet import EmotionResNet
from src.data.build_index import build_samples
from src.data.dataset_utils import TRAIN_DIR

def load_model(model_path="emotion_best.pt", model_type="cnn", num_classes=7, device="cpu"):
    """
    Charge un modèle sauvegardé.
    
    Args:
        model_path: Chemin vers le fichier .pt sauvegardé
        model_type: "cnn" ou "resnet"
        num_classes: Nombre de classes (7 pour les émotions)
        device: "cpu" ou "cuda"
    
    Returns:
        Modèle chargé et en mode évaluation
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
    _, class_to_id = build_samples(TRAIN_DIR)
    id_to_class = {v: k for k, v in class_to_id.items()}
    return id_to_class
