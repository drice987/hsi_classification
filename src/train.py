import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from data_loader import load_pavia_data
from dataset import PaviaPatchDataset
from model import SpatialSpectralCNN

def normalize_data(data: np.ndarray) -> np.ndarray:
    flattened = data.reshape(-1, data.shape[-1])
    means = flattened.mean(axis=0)
    stds = flattened.std(axis=0)
    normalized = (flattened - means) / (stds + 1e-7)
    return normalized.reshape(data.shape)

def train(epochs: int = 15,batch_size: int = 128,lr: float = 1e-3,weights_path: Path = Path("outputs/best_model.pth")) -> None:
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        

    data, gt = load_pavia_data()
    data = normalize_data(data)
    split_col = int(gt.shape[1] * 0.75)
    gt_train = gt.copy()
    gt_train[:, split_col:] = 0
    gt_val = gt.copy()
    gt_val[:, :split_col] = 0
    
    train_dataset = PaviaPatchDataset(data, gt_train, patch_size=9)
    val_dataset = PaviaPatchDataset(data, gt_val, patch_size=9)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SpatialSpectralCNN(in_channels=103, num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                val_loss += criterion(outputs, labels).item()
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            
    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {weights_path}")

if __name__ == "__main__":
    train()