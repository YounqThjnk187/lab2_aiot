#Cau4
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 0. KHỞI TẠO THƯ MỤC OUTPUTS THEO ĐÚNG CẤU TRÚC LAB
# ==========================================
os.makedirs("outputs/vnts_tiny/models", exist_ok=True)
os.makedirs("outputs/vnts_tiny/figures", exist_ok=True)
os.makedirs("outputs/vnts_tiny/benchmark", exist_ok=True)

# Thiết bị huấn luyện
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Thiết bị sử dụng: {device}")

# ==========================================
# 1. ĐỊNH NGHĨA MODEL TinyTrafficCNN
# ==========================================


class TinyTrafficCNN(nn.Module):

  def __init__(self, num_classes):
    super().__init__()
    self.block1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.block3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
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


# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU & DATALOADER
# ==========================================
data_dir = "data/vnts_classification"

# Transform chuẩn theo Mục 2.4.5
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(dataset.classes)

# Chia Train / Val (80% / 20%) cố định seed
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# ==========================================
# 3. KHỞI TẠO MÔ HÌNH, LOSS VÀ OPTIMIZER
# ==========================================
model = TinyTrafficCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 10  # Có thể tăng số epoch để đánh giá overfitting
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": [],
}
best_val_acc = 0.0
start_time_total = time.time()

print("\n=== BẮT ĐẦU HUẤN LUYỆN BASELINE (CÂU 4) ===")

for epoch in range(num_epochs):
  epoch_start_time = time.time()

  # --- Training Phase ---
  model.train()
  running_loss = 0.0
  train_preds, train_targets = [], []

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)
    _, preds = outputs.max(1)
    train_preds.extend(preds.cpu().numpy())
    train_targets.extend(labels.cpu().numpy())

  epoch_train_loss = running_loss / len(train_dataset)
  epoch_train_acc = accuracy_score(train_targets, train_preds) * 100

  # --- Validation Phase ---
  model.eval()
  val_loss = 0.0
  val_preds, val_targets = [], []

  with torch.inference_mode():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      val_loss += loss.item() * images.size(0)
      _, preds = outputs.max(1)
      val_preds.extend(preds.cpu().numpy())
      val_targets.extend(labels.cpu().numpy())

  epoch_val_loss = val_loss / len(val_dataset)
  epoch_val_acc = accuracy_score(val_targets, val_preds) * 100
  epoch_val_f1 = (
      f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100
  )
  epoch_time = time.time() - epoch_start_time

  # Lưu lịch sử
  history["train_loss"].append(epoch_train_loss)
  history["train_acc"].append(epoch_train_acc)
  history["val_loss"].append(epoch_val_loss)
  history["val_acc"].append(epoch_val_acc)
  history["val_f1"].append(epoch_val_f1)

  print(
      f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s) | "
      f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
      f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | Val"
      f" Macro-F1: {epoch_val_f1:.2f}%"
  )

  # --- Lưu Checkpoint tốt nhất (Theo Mục 2.4.19) ---
  if epoch_val_acc > best_val_acc:
    best_val_acc = epoch_val_acc
    # 1. Lưu state_dict
    torch.save(
        model.state_dir()
        if hasattr(model, "state_dir")
        else model.state_dict(),
        "outputs/vnts_tiny/models/best_state_dict.pth",
    )
    # 2. Lưu checkpoint đầy đủ
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    torch.save(
        checkpoint, "outputs/vnts_tiny/models/best_checkpoint.pth"
    )
    # Lưu lại dự đoán tốt nhất để vẽ Confusion Matrix sau
    best_val_preds = val_preds
    best_val_targets = val_targets

total_training_time = time.time() - start_time_total
print(
    f"\n✓ Huấn luyện hoàn tất! Thời gian tổng cộng: {total_training_time:.2f}s."
    f" Accuracy cao nhất: {best_val_acc:.2f}%"
)

# ==========================================
# 4. LƯU CÁC ARTIFACT & VẼ BIỂU ĐỒ (MỤC 2.4.19 & 2.5.1)
# ==========================================

# A. Lưu model_info.json
model_info = {
    "architecture": "TinyTrafficCNN",
    "num_classes": num_classes,
    "input_size": [3, 48, 48],
    "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "classes": dataset.classes,
}
with open("outputs/vnts_tiny/models/model_info.json", "w") as f:
  json.dump(model_info, f, indent=4)

# B. Lưu run_summary.json
run_summary = {
    "platform": "Kaggle GPU" if torch.cuda.is_available() else "Kaggle CPU",
    "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "total_training_time_sec": total_training_time,
    "best_val_accuracy": best_val_acc,
    "final_val_f1": history["val_f1"][-1],
}
with open("outputs/vnts_tiny/benchmark/kaggle_run_summary.json", "w") as f:
  json.dump(run_summary, f, indent=4)

# C. Vẽ và lưu Learning Curves (Loss & Accuracy)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/vnts_tiny/figures/training_history.png", dpi=300)
plt.close()

# ==========================================================
# 👇 BỔ SUNG ĐOẠN NÀY ĐỂ TẠO FILE training_history_f1.png 👇
# ==========================================================
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(history["val_f1"]) + 1),
    history["val_f1"],
    label="Validation Macro-F1",
    color="orange",
    marker="o",
    linewidth=2,
)
plt.xlabel("Epoch")
plt.ylabel("Macro-F1 Score (%)")
plt.title("Training History - Validation Macro-F1 Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/vnts_tiny/figures/training_history_f1.png", dpi=300)
plt.close()
# ==========================================================

# D. Vẽ và lưu Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(best_val_targets, best_val_preds)
sns.heatmap(
    cm,
    annot=False,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=False,
    yticklabels=False,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Best Val Acc: {best_val_acc:.2f}%)")
plt.savefig("outputs/vnts_tiny/figures/confusion_matrix.png", dpi=300)
plt.close()

print(
    "✓ Đã lưu toàn bộ checkpoint, model_info, run_summary và các biểu đồ"
    " figures thành công!"
) 