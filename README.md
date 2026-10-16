# Lab 02: CNN-based Traffic Sign Recognition (VNTS & GTSRB)

Bài thực hành số 2 tập trung vào việc thiết kế, huấn luyện, đánh giá mô hình học sâu `TinyTrafficCNN` trên tập dữ liệu biển báo giao thông Việt Nam (VNTS) và tập chuẩn quốc tế (GTSRB), đồng thời thực hiện benchmark hiệu năng trên các nền tảng phần cứng khác nhau (Cloud GPU vs. Local CPU).

---

## 📁 Cấu trúc thư mục nộp bài (`Lab02_<StudentID>/`)

```text
Lab02_<StudentID>/
│
├── README.md                 # Tài liệu hướng dẫn và mô tả tổng quan bài lab
├── report.pdf                # Báo cáo chi tiết toàn bộ các câu hỏi (Q1 - Q10)
│
├── notebooks/
│   └── Lab02.ipynb           # Notebook chính chứa toàn bộ mã nguồn thực thi
│
├── source/                   # Thư mục chứa các script Python (nếu chạy dạng module)
│
├── models/                   # Lưu trữ trọng số và thông tin mô hình đã huấn luyện
│   ├── best_state_dict.pth   # State dictionary của mô hình tốt nhất
│   ├── best_checkpoint.pth   # Checkpoint đầy đủ (gồm optimizer, epoch,...)
│   └── model_info.json       # Metadata cấu hình kiến trúc mô hình
│
├── figures/                  # Các biểu đồ kết quả trực quan hóa
│   ├── training_history.png      # Biểu đồ Loss & Accuracy qua các epoch
│   ├── training_history_f1.png   # Biểu đồ Macro-F1 qua các epoch
│   ├── confusion_matrix.png      # Ma trận nhầm lẫn (Confusion Matrix)
│   └── misclassified_examples.png# Hình ảnh các mẫu dự đoán sai
│
└── benchmark/                # Dữ liệu đo đạc hiệu năng phần cứng
    ├── local_run_summary.json    # Log benchmark trên Local (CPU)
    ├── kaggle_run_summary.json   # Log benchmark trên Cloud (Tesla T4 GPU)
    └── local_vs_kaggle.csv       # Bảng tổng hợp so sánh hiệu năng

```

---

## 🚀 Hướng dẫn nhanh (Quick Start)

### 1. Cài đặt môi trường

Đảm bảo bạn đã cài đặt Python (phiên bản >= 3.10) và kích hoạt môi trường ảo (`.venv`):

```bash
# Kích hoạt môi trường ảo trên Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện phụ thuộc (hoặc dùng file .whl offline nếu không có mạng)
pip install -r requirements.txt

```

### 2. Chạy Notebook

Mở mã nguồn trực tiếp qua Jupyter Notebook / JupyterLab:

```bash
jupyter notebook notebooks/Lab02.ipynb

```

### 3. Chạy Benchmark thực tế

Để kiểm tra hiệu năng trên thiết bị cá nhân (CPU/GPU):

```bash
python source/platform_benchmark.py

```

---

## 📊 Tóm tắt kết quả chính

* **Baseline so sánh**: So sánh hiệu năng giữa phương pháp truyền thống (`HOG + SVM`) và `TinyTrafficCNN`.
* **Hiệu năng phần cứng (Câu 7)**:
* **GPU (Tesla T4 - Kaggle)**: Thời gian trung bình 1 epoch = `4.20s`, Throughput = `1451.17 samples/s`, Latency = `0.54 ms/sample`.


* **Local (CPU)**: Thời gian trung bình 1 epoch = `17.53s`, Throughput = `347.95 samples/s`, Latency = `0.85 ms/sample`.




* **GTSRB Evaluation (Câu 8)**: Đạt độ chính xác `50.37%` trên tập kiểm thử sử dụng chiến lược "freeze and evaluate".
