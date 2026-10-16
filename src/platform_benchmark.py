# Cau7
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score


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


if __name__ == "__main__":
  # Thiết lập an toàn cho đa tiến trình trên Windows
  import multiprocessing

  multiprocessing.freeze_support()

  # ==========================================
  # 1. THIẾT LẬP MÔI TRƯỜNG & KIẾN TRÚC
  # ==========================================
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"-> Đang chạy benchmark trên thiết bị: {device}")
  if device.type == "cuda":
    print(f"-> Tên GPU: {torch.cuda.get_device_name(0)}")

  # Đường dẫn dataset trên Local (điều chỉnh lại thư mục cho khớp với máy bạn)
  data_dir = "data/vnts_classification"

  transform = transforms.Compose([
      transforms.Resize((48, 48)),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
  ])

  dataset = datasets.ImageFolder(root=data_dir, transform=transform)
  num_classes = len(dataset.classes)

  # Cố định seed để chia tập dữ liệu hoàn toàn giống nhau
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(
      dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
  )

  # Cấu hình DataLoader (có thể dùng num_workers=2 an toàn khi có khối if __name__ == '__main__')
  train_loader = DataLoader(
      train_dataset, batch_size=64, shuffle=True, num_workers=2
  )
  val_loader = DataLoader(
      val_dataset, batch_size=64, shuffle=False, num_workers=2
  )

  model = TinyTrafficCNN(num_classes=num_classes).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=0.001)

  # ==========================================
  # 2. CHẠY HUẤN LUYỆN 3 EPOCH ĐỂ ĐO ĐẠC
  # ==========================================
  num_epochs = 3
  epoch_times = []

  print("\n=== BẮT ĐẦU ĐO ĐẠC BENCHMARK HUẤN LUYỆN ===")
  for epoch in range(num_epochs):
    start_time = time.time()
    model.train()

    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad(set_to_none=True)
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    if device.type == "cuda":
      torch.cuda.synchronize()

    elapsed = time.time() - start_time
    epoch_times.append(elapsed)
    print(f"Epoch {epoch+1}/{num_epochs} hoàn thành trong: {elapsed:.2f} giây")

  avg_epoch_time = sum(epoch_times) / len(epoch_times)
  samples_per_sec = len(train_dataset) / avg_epoch_time

  # ==========================================
  # 3. ĐO ĐẠC INFERENCE LATENCY (MS/SAMPLE)
  # ==========================================
  model.eval()
  dummy_input = torch.randn(1, 3, 48, 48).to(device)

  for _ in range(10):
    _ = model(dummy_input)

  num_iterations = 200
  start_infer = time.time()
  with torch.inference_mode():
    for _ in range(num_iterations):
      _ = model(dummy_input)
      if device.type == "cuda":
        torch.cuda.synchronize()

  total_infer_time = time.time() - start_infer
  latency_ms_per_sample = (total_infer_time / num_iterations) * 1000

  # ==========================================
  # 4. ĐÁNH GIÁ METRICS (VALIDATION)
  # ==========================================
  val_preds, val_targets = [], []
  with torch.inference_mode():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, preds = outputs.max(1)
      val_preds.extend(preds.cpu().numpy())
      val_targets.extend(labels.cpu().numpy())

  val_acc = accuracy_score(val_targets, val_preds) * 100
  val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100

  print("\n" + "=" * 40)
  print(f" KẾT QUẢ BENCHMARK TRÊN NỀN TẢNG: {device.type.upper()}")
  print("=" * 40)
  print(f"• Thời gian trung bình 1 Epoch : {avg_epoch_time:.2f} s")
  print(f"• Tốc độ xử lý (Throughput)    : {samples_per_sec:.2f} samples/s")
  print(f"• Độ trễ dự đoán (Latency)     : {latency_ms_per_sample:.2f} ms/sample")
  print(f"• Validation Accuracy          : {val_acc:.2f}%")
  print(f"• Validation Macro-F1          : {val_f1:.2f}%")
  print("=" * 40)