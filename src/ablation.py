#Cau5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
import time
import os

# Tạo thư mục output cho thí nghiệm Augmentation
os.makedirs("outputs/vnts_tiny_aug/models", exist_ok=True)
os.makedirs("outputs/vnts_tiny_aug/figures", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data/vnts_classification"

# --- THAY ĐỔI CHÍNH: BẬT DATA AUGMENTATION MỨC NHẸ ---
transform_aug = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(15),           # Xoay ngẫu nhiên góc nhỏ
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Dịch chuyển nhẹ
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset_aug = datasets.ImageFolder(root=data_dir, transform=transform_aug)
num_classes = len(dataset_aug.classes)

# Cố định seed chia tập dữ liệu để so sánh công bằng
train_size = int(0.8 * len(dataset_aug))
val_size = len(dataset_aug) - train_size
train_dataset, val_dataset = random_split(
    dataset_aug, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Định nghĩa lại mô hình TinyTrafficCNN
class TinyTrafficCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model_aug = TinyTrafficCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_aug.parameters(), lr=0.001)

num_epochs = 10
best_val_acc_aug = 0.0

print("\n=== BẮT ĐẦU THÍ NGHIỆM ABLATION: TINYCNN + DATA AUGMENTATION ===")
for epoch in range(num_epochs):
    model_aug.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model_aug(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    # Validation
    model_aug.eval()
    val_preds, val_targets = [], []
    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_aug(images)
            _, preds = outputs.max(1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_targets, val_preds) * 100
    val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) * 100

    if val_acc > best_val_acc_aug:
        best_val_acc_aug = val_acc
        torch.save(model_aug.state_dict(), "outputs/vnts_tiny_aug/models/best_state_dict.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.2f}%")

print(f"\n✓ Hoàn thành thí nghiệm Augmentation! Accuracy cao nhất: {best_val_acc_aug:.2f}%")