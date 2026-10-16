#Cau3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC TinyTrafficCNN
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
# 2. KIỂM TRA THIẾT BỊ VÀ LOAD DATASET
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Đang sử dụng thiết bị: {device}")

data_dir = "data/vnts_classification"
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

print("-> Đang tải dataset từ ImageFolder...")
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(dataset.classes)
print(
    f"-> Tải thành công! Tổng số mẫu: {len(dataset)}, Số lượng lớp:"
    f" {num_classes}"
)

# Khởi tạo mô hình
model = TinyTrafficCNN(num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"-> Tổng số tham số (Parameters): {total_params:,}")

# Chia tập Train / Val (Dùng num_workers=0 để tránh treo cell trên Kaggle)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# ==========================================
# 3. HUẤN LUYỆN THỬ NGHIỆM (2 EPOCHS)
# ==========================================
num_epochs = 2
print(
    "\n=== BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) ==="
)  # In thông báo để thấy ngay

for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  print(f"Epoch {epoch+1}/{num_epochs} đang chạy...")
  for batch_idx, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

    # In tiến độ nhỏ mỗi 20 batch để biết code đang chạy
    if (batch_idx + 1) % 20 == 0:
      print(
          f"  - Đã xử lý xong batch {batch_idx+1}/{len(train_loader)}..."
          f" (Loss: {loss.item():.4f})"
      )

  train_loss = running_loss / total
  train_acc = 100.0 * correct / total

  # Validation Phase
  model.eval()
  val_loss = 0.0
  val_correct = 0
  val_total = 0

  with torch.inference_mode():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      val_loss += loss.item() * images.size(0)
      _, predicted = outputs.max(1)
      val_total += labels.size(0)
      val_correct += predicted.eq(labels).sum().item()

  val_loss = val_loss / val_total
  val_acc = 100.0 * val_correct / val_total

  print(
      f"--> KẾT QUẢ EPOCH [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f}"
      f" | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc:"
      f" {val_acc:.2f}%\n"
  )

print("✓ Hoàn thành thí nghiệm huấn luyện baseline thành công!")