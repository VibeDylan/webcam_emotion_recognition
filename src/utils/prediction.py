import numpy as np
import torch
import torch.nn.functional as F

def predict_emotion(model, face48, device):
    x = face48.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, 1))
    x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_id].item())
    return pred_id, pred_prob, probs.detach().cpu().numpy()
