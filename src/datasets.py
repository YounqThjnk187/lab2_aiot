#Cau2
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==========================================
# 1. THIẾT LẬP ĐƯỜNG DẪN VÀ TRANSFORMS
# ==========================================
# Đường dẫn chính xác trỏ đến thư mục chứa các thư mục con theo class (ví dụ thư mục '7')
data_dir = "data/vnts_classification"

# Định nghĩa các phép biến đổi ảnh sang Tensor và chuẩn hóa về khoảng [-1, 1]
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Sử dụng ImageFolder để tự động gán nhãn dựa theo tên thư mục con (0, 1, 2, ..., 42)
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

print(f"Tổng số lượng mẫu ảnh tìm thấy: {len(dataset)}")
print(f"Danh sách các lớp (Classes): {dataset.classes}")

# DataLoader nhóm các sample thành mini-batch
dataloader = DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True
)

# ==========================================
# 2. LẤY MỘT BATCH VÀ KIỂM TRA SHAPE, MIN/MAX
# ==========================================
images, labels = next(iter(dataloader))

print("=" * 50)
print(" KIỂM TRA TENSOR BATCH (CÂU 2)")
print("=" * 50)
print(f"Shape của batch ảnh (B x C x H x W): {images.shape}")
print(f"Shape của batch nhãn: {labels.shape}")
print(f"Giá trị Min của tensor sau khi Normalize: {images.min().item():.4f}")
print(f"Giá trị Max của tensor sau khi Normalize: {images.max().item():.4f}")
print("=" * 50)


# ==========================================
# 3. HÀM DENORMALIZE VÀ HIỂN THỊ ẢNH
# ==========================================
def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
  """Hàm chuyển ngược tensor từ khoảng [-1, 1] về [0, 1] để hiển thị ảnh."""
  denorm_tensor = tensor.clone()
  for t, m, s in zip(denorm_tensor, mean, std):
    t.mul_(s).add_(m)
  return torch.clamp(denorm_tensor, 0, 1)


# Hiển thị 4 ảnh đầu tiên trong batch
plt.figure(figsize=(10, 5))
for i in range(len(images)):
  plt.subplot(1, len(images), i + 1)
  img_denorm = denormalize(images[i])
  img_np = img_denorm.permute(1, 2, 0).numpy()

  plt.imshow(img_np)
  plt.title(f"Class ID: {labels[i].item()}")
  plt.axis("off")

plt.tight_layout()
plt.show()
print(
    "✓ Đã load dữ liệu thành công từ ImageFolder, kiểm tra Tensor và hiển thị"
    " ảnh hoàn tất!"
)