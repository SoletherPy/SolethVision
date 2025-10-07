import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from shared.modules.model import EnhancedFingerprintCNN
from shared.modules.utils import resource_path

os.environ['PYTHONIOENCODING'] = 'utf-8'
torch.backends.cudnn.benchmark = True

DATA_DIR = resource_path(__file__, "../data")
MODEL_BEST_PATH = resource_path(__file__, "../shared/models/fingerprint_cnn_enhanced_best.pt")
MODEL_FINAL_PATH = resource_path(__file__, "../shared/models/fingerprint_cnn_enhanced_final.pt")

def evaluate_model(model, dataloader, criterion, device, num_classes=4):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        with autocast(device_type=device.type):
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                for i in range(labels.size(0)):
                    c = labels[i].item()
                    class_correct[c] += int(preds[i] == labels[i])
                    class_total[c] += 1
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    class_accuracies = [
        (class_correct[i] / class_total[i]) if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    ]
    return avg_loss, accuracy, class_accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Transforms
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset
full_dataset = datasets.ImageFolder(DATA_DIR)
print(f"🗂️ Total images: {len(full_dataset)}")
print(f"🏷️ Classes: {full_dataset.classes}")

# Split (shuffle to avoid bias)
indices = list(range(len(full_dataset)))
random.shuffle(indices)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices = indices[:train_size]
val_indices = indices[train_size:]
print(f"📦 Training images: {train_size} | 🧪 Validation images: {val_size}")

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = TransformDataset(full_dataset, train_indices, train_transforms)
val_dataset = TransformDataset(full_dataset, val_indices, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

print("🚚 DataLoaders ready")

# Model/optim
num_classes = len(full_dataset.classes)
model = EnhancedFingerprintCNN(num_classes=num_classes, dropout_rate=0.3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
scaler = GradScaler()

# Training params
best_val_acc = 0.0
best_val_loss = float('inf')
num_epochs = 250
patience = 25
early_stop_counter = 0

print("🚀 Starting enhanced training...")
print(f"🧮 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        train_loss += loss.item()
        preds = outputs.argmax(1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

    train_loss_avg = train_loss / max(1, len(train_loader))
    train_acc = train_correct / max(1, train_total)

    val_loss_avg, val_acc, val_class_acc = evaluate_model(model, val_loader, criterion, device, num_classes=num_classes)

    scheduler.step(val_loss_avg)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"📆 Epoch {epoch+1:3d} | 🏋️ Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.4f} | "
          f"🧪 Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.4f} | 🧭 LR: {current_lr:.6f}")
    class_acc_str = " ".join([f"C{i}: {acc:.3f}" for i, acc in enumerate(val_class_acc)])
    print(f"🎯 Class accuracies → {class_acc_str}")

    if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss_avg < best_val_loss):
        best_val_acc = val_acc
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), MODEL_BEST_PATH)
        print(f"💾 Saved new BEST model ✅ (Val Acc: {val_acc:.4f}, Val Loss: {val_loss_avg:.4f})")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"⏳ No improvement. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print(f"🛑 Early stopping after {patience} epochs without improvement.")
        break

    print("──────────────────────────────────────────────────────────────────────────────")

torch.save(model.state_dict(), MODEL_FINAL_PATH)
print("✅ Training completed!")
print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
print("📦 Artifacts saved:")
print("  • fingerprint_cnn_enhanced_best.pt (best validation)")
print("  • fingerprint_cnn_enhanced_final.pt (final epoch)")