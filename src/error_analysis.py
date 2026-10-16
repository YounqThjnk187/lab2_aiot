#Cau6
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# 1. THIẾT LẬP MÔI TRƯỜNG VÀ TẢI MODEL ĐÃ HUẤN LUYỆN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data/vnts_classification"

# Tạo thư mục lưu kết quả phân tích lỗi
os.makedirs("outputs/vnts_tiny/figures", exist_ok=True)

# Transform chuẩn
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(dataset.classes)

# Cố định lại đúng seed chia tập dữ liệu như lúc train
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)


# Khai báo lại kiến trúc TinyTrafficCNN để load trọng số
import torch.nn as nn


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


model = TinyTrafficCNN(num_classes=num_classes).to(device)
# Load lại trọng số tốt nhất đã lưu ở Câu 4
model.load_state_dict(
    torch.load(
        "outputs/vnts_tiny/models/best_state_dict.pth",
        weights_only=True,
        map_location=device,
    )
)
model.eval()

# ==========================================
# 2. HÀM DENORMALIZE ĐỂ HIỂN THỊ ẢNH
# ==========================================
def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
  denorm_tensor = tensor.clone()
  for t, m, s in zip(denorm_tensor, mean, std):
    t.mul_(s).add_(m)
  return torch.clamp(denorm_tensor, 0, 1)


# ==========================================
# 3. THỰC THI THU THẬP CÁC MẪU DỰ ĐOÁN SAI
# ==========================================
misclassified_samples = []

print("=== BẮT ĐẦU QUÁ TRÌNH PHÂN TÍCH MẪU SAI (ERROR ANALYSIS) ===")
with torch.inference_mode():
  for images, labels in val_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    confidences, preds = torch.max(probs, dim=1)

    # Tìm các mẫu dự đoán sai trong batch
    incorrect_mask = preds != labels
    if incorrect_mask.sum() > 0:
      incorrect_imgs = images[incorrect_mask]
      incorrect_true = labels[incorrect_mask]
      incorrect_pred = preds[incorrect_mask]
      incorrect_conf = confidences[incorrect_mask]

      for img, true_l, pred_l, conf in zip(
          incorrect_imgs, incorrect_true, incorrect_pred, incorrect_conf
      ):
        misclassified_samples.append({
            "image": img.cpu(),
            "true_label": true_l.item(),
            "pred_label": pred_l.item(),
            "confidence": conf.item(),
        })

print(f"Tổng số mẫu dự đoán sai tìm thấy trên tập Val: {len(misclassified_samples)}")

# ==========================================
# 4. VẼ VÀ LƯU ẢNH TRỰC QUAN CÁC MẪU SAI
# ==========================================
num_plots = min(20, len(misclassified_samples))
if num_plots > 0:
  plt.figure(figsize=(15, 12))
  for i in range(num_plots):
    sample = misclassified_samples[i]
    plt.subplot(4, 5, i + 1)

    # Khôi phục ảnh gốc từ tensor đã chuẩn hóa
    img_denorm = denormalize(sample["image"])
    img_np = img_denorm.permute(1, 2, 0).numpy()

    plt.imshow(img_np)
    true_name = dataset.classes[sample["true_label"]]
    pred_name = dataset.classes[sample["pred_label"]]
    plt.title(
        f"True: {true_name}\nPred: {pred_name}\nConf: {sample['confidence']:.2f}",
        fontsize=9,
    )
    plt.axis("off")

  plt.tight_layout()
  plt.savefig(
      "outputs/vnts_tiny/figures/misclassified_examples.png",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close()
  print(
      "✓ Đã lưu file ảnh mẫu sai tại:"
      " outputs/vnts_tiny/figures/misclassified_examples.png"
  )
else:
  print("Không tìm thấy mẫu sai nào để hiển thị!")