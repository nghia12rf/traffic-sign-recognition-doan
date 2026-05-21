# HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG

Đồ án môn học: Ứng dụng nhận diện biển báo giao thông sử dụng kết hợp YOLO (phát hiện vùng chứa) và ResNet18 + CBAM (phân loại biển báo).

**Mô tả ngắn**
- Hệ thống quét ảnh đầu vào để phát hiện các biển báo (YOLO) và phân loại loại biển báo (ResNet18 + CBAM).
- Giao diện web đơn giản dùng `Streamlit` để upload ảnh và hiển thị kết quả cùng độ tin cậy.

**Yêu cầu**
- Python 3.8+
- GPU nếu muốn tăng tốc (CUDA) — code tự động chọn `cuda` nếu có.
- Các thư viện có trong `requirements.txt` (ví dụ: `streamlit`, `ultralytics`, `torch`, `torchvision`, `pillow`).

**Cài đặt nhanh**
1. Tạo môi trường (tùy chọn conda):

```bash
conda create -n traffic_env python=3.10 -y
conda activate traffic_env
```

2. Cài đặt phụ thuộc:

```bash
pip install -r requirements.txt
```

**Chạy ứng dụng**
- Khởi động ứng dụng Streamlit:

```bash
streamlit run app.py
```

- Sau khi chạy, mở trình duyệt theo đường dẫn mà Streamlit in ra (thường là `http://localhost:8501`).
- Tại giao diện, upload ảnh (jpg/png) vào sidebar hoặc thử với thư mục `test_images/` có sẵn.

**Cấu trúc dự án**
- `app.py` — ứng dụng Streamlit (giao diện + pipeline YOLO → ResNet)
- `requirements.txt` — danh sách gói Python cần cài
- `models/` — chứa các trọng số đã huấn luyện:
  - `best.pt` — mô hình YOLO để phát hiện biển báo
  - `resnet_cbam_final.pth` — trọng số ResNet18 + CBAM để phân loại
- `test_images/` — ảnh mẫu để thử nghiệm

**Ghi chú về mô hình**
- Ứng dụng tải `models/best.pt` cho YOLO và `models/resnet_cbam_final.pth` cho ResNet.
- Nếu thiếu file mô hình, ứng dụng sẽ lỗi khi khởi tạo; hãy đảm bảo các file này nằm trong thư mục `models/`.

**Mẹo và khắc phục**
- Nếu không chạy được GPU, kiểm tra phiên bản `torch` tương thích với CUDA trên máy của bạn.
- Nếu Streamlit không mở trang, xem log terminal để biết lỗi cụ thể.

**Bản quyền & Liên hệ**
- Dự án này là đồ án môn học; sử dụng nội bộ và tham khảo theo mục đích học tập.
- Nếu cần hỗ trợ thêm, reply lại trong repo này.
