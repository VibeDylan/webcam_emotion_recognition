import torch
from torch.utils.data import DataLoader

from build_index import build_samples, split_samples, TRAIN_DIR
from fer_dataset import FerDataset
from emotion_cnn import EmotionCNN

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    samples, class_to_id = build_samples(TRAIN_DIR)
    train_samples, val_samples = split_samples(samples, val_ratio=0.1, seed=42)

    train_loader = DataLoader(FerDataset(train_samples), batch_size=64, shuffle=True, num_workers=2)
    val_loader   = DataLoader(FerDataset(val_samples),   batch_size=64, shuffle=False, num_workers=2)

    model = EmotionCNN(num_classes=len(class_to_id)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    for epoch in range(1, 11):
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "emotion_best.pt")
            print("  saved emotion_best.pt")

if __name__ == "__main__":
    main()