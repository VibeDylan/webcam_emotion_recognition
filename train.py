import torch
import argparse
from torch.utils.data import DataLoader
from src.data.build_index import build_samples, split_samples
from src.data.dataset_utils import TRAIN_DIR
from src.data.fer_dataset import FerDataset
from src.models.emotion_cnn import EmotionCNN
from src.models.emotion_resnet import EmotionResNet

def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    return acc

def run_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        acc = accuracy_from_logits(logits, y)
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        acc = accuracy_from_logits(logits, y)
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser(description="Entraîner un modèle de détection d'émotions")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet"],
                       help="Type de modèle à utiliser: 'cnn' ou 'resnet' (défaut: cnn)")
    parser.add_argument("--pretrained", action="store_true",
                       help="Utiliser des poids pré-entraînés pour ResNet (seulement si --model resnet)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Modèle: {args.model.upper()}")
    if args.model == "resnet":
        print(f"Pré-entraîné: {args.pretrained}")
    
    samples, class_to_id = build_samples(TRAIN_DIR)
    train_samples, val_samples = split_samples(samples, val_ratio=0.1, seed=42)
    
    train_loader = DataLoader(FerDataset(train_samples, augment=True), batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(FerDataset(val_samples, augment=False), batch_size=64, shuffle=False, num_workers=2)
    
    if args.model == "resnet":
        model = EmotionResNet(num_classes=len(class_to_id), pretrained=args.pretrained).to(device)
        model_name = "emotion_resnet_best.pt"
        lr = 1e-4 if args.pretrained else 1e-3
    else:
        model = EmotionCNN(num_classes=len(class_to_id)).to(device)
        model_name = "emotion_best.pt"
        lr = 1e-3
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(1, 51):
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d}/50 | Train: loss={train_loss:.4f} acc={train_acc:.3f} | Val: loss={val_loss:.4f} acc={val_acc:.3f} | LR={current_lr:.6f}", end="")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_name)
            patience_counter = 0
            print(f" ✓ Saved (best: {best_val_acc:.3f})")
        else:
            patience_counter += 1
            print()
        
        if patience_counter >= 10:
            print(f"\nEarly stopping: no improvement for 10 epochs. Best val acc: {best_val_acc:.3f}")
            break

if __name__ == "__main__":
    main()
