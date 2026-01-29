"""Single-image emotion prediction for a 48x48 grayscale face."""
import numpy as np
import torch
import torch.nn.functional as F


def predict_emotion(model, face48, device):
    """
    Predict emotion for one 48x48 grayscale face (numpy HxW, 0-255).

    Args:
        model: Loaded emotion model in eval mode.
        face48: numpy array (48, 48), grayscale, 0-255.
        device: torch device for the model.

    Returns:
        pred_id: predicted class index.
        pred_prob: probability of the predicted class.
        probs: full probability vector (numpy, length num_classes).
    """
    x = face48.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, 1))
    x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_id].item())
    return pred_id, pred_prob, probs.detach().cpu().numpy()
