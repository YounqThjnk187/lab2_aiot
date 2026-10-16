#Cau8
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import GTSRB
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
import time

# ==========================================
# 0. THIẾT LẬP THƯ MỤC OUTPUTS CHO GTSRB
# ==========================================
os.makedirs("outputs/gtsrb_tiny/models", exist_ok=True)
os.makedirs("outputs/gtsrb_tiny/figures", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Thiết bị sử dụng: {device}")

# ==========================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU GTSRB
# ==========================================
# Transform chuẩn giữ nguyên recipe từ VNTS
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

print("-> Đang tải official train_set từ GTSRB...")

if __name__ == "__main__":
    official_train_set = GTSRB(root='data/gtsrb', split='train', download=True, transform=transform)

    print("-> Đang tải official test_set từ GTSRB...")
    official_test_set = GTSRB(root='data/gtsrb', split='test', download=True, transform=transform)

    # Tách official train thành train (80%) và validation (20%) để chọn mô hình
    train_size = int(0.8 * len(official_train_set))
    val_size = len(official_train_set) - train_size
    train_dataset, val_dataset = random_split(
        official_train_set, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(official_test_set, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 43 # GTSRB chuẩn có đúng 43 lớp phân loại[cite: 13]
    print(f"-> Số lượng lớp GTSRB: {num_classes}")
    print(f"-> Số lượng mẫu Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(official_test_set)}")

    # ==========================================
    # 2. KHỞI TẠO MÔ HÌNH TinyTrafficCNN (GIỮ NGUYÊN RECIPE)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # ==========================================
    # 3. HUẤN LUYỆN VÀ CHỌN MODEL BẰNG VALIDATION SET
    # ==========================================
    num_epochs = 10
    best_val_acc = 0.0

    print("\n=== BẮT ĐẦU RETRAIN TRÊN GTSRB ===")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_dataset)

        # Validation Phase (Dùng để chọn mô hình tối ưu)
        model.eval()
        val_preds, val_targets = [], []
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) * 100

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.2f}%")

        # Lưu model tốt nhất dựa hoàn toàn trên Validation Set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "outputs/gtsrb_tiny/models/best_state_dict.pth")

    print(f"\n✓ Hoấn luyện hoàn tất! Validation Accuracy cao nhất: {best_val_acc:.2f}%")

    # ==========================================
    # 4. ĐÁNH GIÁ CUỐI CÙNG TRÊN OFFICIAL TEST SET
    # ==========================================
    print("\n=== ĐÁNH GIÁ ĐỘC LẬP TRÊN OFFICIAL TEST SET ===")
    # Load lại trọng số tốt nhất từ quá trình validate
    model.load_state_dict(torch.load("outputs/gtsrb_tiny/models/best_state_dict.pth", weights_only=True))
    model.eval()

    test_preds, test_targets = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds) * 100
    test_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0) * 100

    print(f"★ KẾT QUẢ OFFICIAL TEST SET:")
    print(f"  - Test Accuracy : {test_acc:.2f}%")
    print(f"  - Test Macro-F1 : {test_f1:.2f}%")