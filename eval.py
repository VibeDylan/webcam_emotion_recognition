"""
Evaluation script for the emotion recognition model.
Computes global accuracy, per-class accuracy, and confusion matrix on the validation set.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from src.data.build_index import build_samples, split_samples
from src.data.dataset_utils import TRAIN_DIR
from src.data.fer_dataset import FerDataset
from src.utils.model_loader import load_model, get_class_names


def evaluate(model, loader, device):
    """Run the model on the loader and return ground-truth labels and predictions as numpy arrays."""
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            all_labels.append(y.cpu())
            all_preds.append(pred.cpu())
    labels = torch.cat(all_labels, dim=0)
    preds = torch.cat(all_preds, dim=0)
    return labels.numpy(), preds.numpy()


def main():
    """Load model and validation data, compute metrics, and print accuracy and confusion matrix."""
    parser = argparse.ArgumentParser(description="Evaluate the model on the validation set")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the .pt file (default: emotion_best.pt or emotion_resnet_best.pt)")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = "emotion_resnet_best.pt" if args.model == "resnet" else "emotion_best.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 7

    model = load_model(args.model_path, model_type=args.model, num_classes=num_classes, device=device)
    id_to_class = get_class_names()
    class_names = [id_to_class[i] for i in range(num_classes)]

    samples, class_to_id = build_samples(TRAIN_DIR)
    _, val_samples = split_samples(samples, val_ratio=0.1, seed=42)
    val_loader = DataLoader(
        FerDataset(val_samples, augment=False),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    labels, preds = evaluate(model, val_loader, device)
    n = len(labels)

    acc_global = (labels == preds).mean()
    print(f"Model: {args.model_path}")
    print(f"Number of images (validation): {n}")
    print(f"Global accuracy: {acc_global:.4f} ({int((labels == preds).sum())}/{n})")
    print()

    print("Accuracy by class:")
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            acc_c = 0.0
        else:
            acc_c = (preds[mask] == c).mean()
        count = int(mask.sum())
        print(f"  {class_names[c]:12s}: {acc_c:.4f}  ({count} images)")
    print()

    confusion = [[0] * num_classes for _ in range(num_classes)]
    for true_label, pred_label in zip(labels, preds):
        confusion[int(true_label)][int(pred_label)] += 1

    print("Confusion matrix (row = true class, column = predicted):")
    header = "             " + " ".join(f"{name:>10}" for name in class_names)
    print(header)
    for i in range(num_classes):
        row = f"{class_names[i]:12s} " + " ".join(f"{confusion[i][j]:10d}" for j in range(num_classes))
        print(row)


if __name__ == "__main__":
    main()
