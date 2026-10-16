#Cau9
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. THIẾT LẬP VÀ LOAD DỮ LIỆU TEST (GIỐNG NHAU CHO CẢ HAI)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data/vnts_classification"

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(dataset.classes)

# Dùng chung một phần tập kiểm thử để so sánh công bằng
_, val_dataset = torch.utils.data.random_split(
    dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
    generator=torch.Generator().manual_seed(42)
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ==========================================
# 2. KHAI BÁO LẠI MÔ HÌNH TinyTrafficCNN ĐỂ ĐO ĐẠC
# ==========================================
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

model = TinyTrafficCNN(num_classes=num_classes).to(device)
# Load trọng số đã train ở Câu 4
try:
    model.load_state_dict(torch.load("outputs/vnts_tiny/models/best_state_dict.pth", weights_only=True))
except:
    print("⚠️ Chưa tìm thấy file weights của Câu 4, vui lòng đảm bảo đã chạy xong Câu 4 trước.")
model.eval()

# Tính số tham số và dung lượng model (FP32 = 4 bytes/param)
total_params = sum(p.numel() for p in model.parameters())
model_size_kb = (total_params * 4) / 1024

# Đánh giá Accuracy & F1 của CNN trên tập Val
cnn_preds, cnn_targets = [], []
with torch.inference_mode():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        cnn_preds.extend(preds.cpu().numpy())
        cnn_targets.extend(labels.cpu().numpy())

cnn_acc = accuracy_score(cnn_targets, cnn_preds) * 100
cnn_f1 = f1_score(cnn_targets, cnn_preds, average='macro', zero_division=0) * 100

# Đo Inference Latency của CNN
dummy_input = torch.randn(1, 3, 48, 48).to(device)
for _ in range(10): _ = model(dummy_input) # Warm-up
start_t = time.time()
with torch.inference_mode():
    for _ in range(100):
        _ = model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()
cnn_latency = ((time.time() - start_t) / 100) * 1000

# ==========================================
# 3. TỔNG HỢP SỐ LIỆU SO SÁNH (HOG+SVM GIẢ ĐỊNH DỰA TRÊN KẾT QUẢ BÀI 1 HOẶC THỰC TẾ NHÓM)
# ==========================================
# Lưu ý: Bạn có thể thay đổi số liệu HOG+SVM dưới đây cho khớp với kết quả thực tế của Bài 1 nhóm mình
hog_svm_metrics = {
    "Model": "HOG + SVM (Bài 1)",
    "Accuracy": 54.50, # Thay bằng số liệu thực tế Bài 1 của nhóm
    "Macro-F1": 25.20, # Thay bằng số liệu thực tế Bài 1 của nhóm
    "Params": "N/A (Non-DL)",
    "Size (KB)": 450.0, # Kích thước file model SVM ước lượng
    "Latency (ms)": 1.5   # Tốc độ trích xuất HOG + SVM inference
}

cnn_metrics = {
    "Model": "TinyTrafficCNN (Bài 2)",
    "Accuracy": cnn_acc,
    "Macro-F1": cnn_f1,
    "Params": total_params,
    "Size (KB)": model_size_kb,
    "Latency (ms)": cnn_latency
}

print("\n" + "="*60)
print(f"{'BẢNG SO SÁNH KẾT QUẢ: HOG+SVM vs CNN':^60}")
print("="*60)
print(f"{'Tiêu chí':<20} | {'HOG + SVM (Bài 1)':<20} | {'TinyTrafficCNN (Bài 2)':<20}")
print("-" * 65)
print(f"{'Accuracy (%)':<20} | {hog_svm_metrics['Accuracy']:<20.2f} | {cnn_metrics['Accuracy']:<20.2f}")
print(f"{'Macro-F1 (%)':<20} | {hog_svm_metrics['Macro-F1']:<20.2f} | {cnn_metrics['Macro-F1']:<20.2f}")
print(f"{'Model Size':<20} | {str(hog_svm_metrics['Size (KB)'])+' KB':<20} | {str(round(cnn_metrics['Size (KB)'], 2))+' KB':<20}")
print(f"{'Inference Latency':<20} | {str(hog_svm_metrics['Latency (ms)'])+' ms':<20} | {str(round(cnn_metrics['Latency (ms)'], 2))+' ms':<20}")
print("="*60)

# ==========================================
# 4. VẼ BIỂU ĐỒ SO SÁNH TRỰC QUAN
# ==========================================
os.makedirs("outputs/benchmark", exist_ok=True)
metrics_names = ['Accuracy (%)', 'Macro-F1 (%)']
hog_values = [hog_svm_metrics['Accuracy'], hog_svm_metrics['Macro-F1']]
cnn_values = [cnn_metrics['Accuracy'], cnn_metrics['Macro-F1']]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, hog_values, width, label='HOG + SVM (Lab 1)', color='orange')
rects2 = ax.bar(x + width/2, cnn_values, width, label='TinyTrafficCNN (Lab 2)', color='royalblue')

ax.set_ylabel('Scores (%)')
ax.set_title('So sánh hiệu năng giữa HOG+SVM và CNN')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.1f%%')
ax.bar_label(rects2, padding=3, fmt='%.1f%%')

plt.tight_layout()
plt.savefig("outputs/benchmark/hog_vs_cnn_comparison.png", dpi=300)
plt.close()
print("✓ Đã lưu biểu đồ so sánh tại: outputs/benchmark/hog_vs_cnn_comparison.png")