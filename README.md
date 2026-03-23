# Hệ thống Giám sát ATM (ATM Surveillance System)

Hệ thống giám sát tại trụ ATM ứng dụng AI (YOLOv8 & PyTorch) để phát hiện, theo dõi các hành vi và tiến hành phân tích, báo cáo các bất thường xảy ra.

## Tính năng chính
- **YOLOv8 Detection & Tracking**: Phát hiện người, tracking đối tượng và phân loại hành vi cơ bản (gửi/rút tiền, sử dụng điện thoại, đeo khẩu trang, v.v.).
- **Phát hiện bất thường (Anomaly Detection)**: Sử dụng mô hình Time-series/Action Recognition (vd: MobileNetV3 + GRU) để nhận diện các trạng thái đặc biệt bất thường qua chuỗi video (clips).
- **Web Interface (Flask)**: Giao diện dashboard trực quan cung cấp MJPEG streaming thời gian thực.
- **CLI Processing**: Ứng dụng dòng lệnh cho mục đích xử lý offline hoặc phân tích hàng loạt video/file.
- **Training Pipeline**: Có sẵn các script để huấn luyện lại cả YOLOv8 và mô hình học sâu phân tích bất thường một cách chủ động.

## Tải Trọng số Mô hình (Model Weights)

Vì file trọng số mô hình (`.pt`) khá nặng nên không được commit trực tiếp lên mã nguồn Github.

👉 **Tải toàn bộ Models/Weights tại đây**: `https://thinklabs102-my.sharepoint.com/:f:/g/personal/hieuvm_thinklabs_com_vn/IgDhF7zG0VM6TrW_Rc49AWFrAdIz8101vDwZqQ6yahCHKVE?e=zCnMMh`

### Hướng dẫn tạo thư mục và chép file Weights:
Để chạy hệ thống, bạn chỉ cần tải 2 file trọng số (weights) và đặt đúng vào các thư mục sau:

**1. Mô hình dự đoán trạng thái bất thường (Anomaly):**
Tạo thư mục nếu chưa có:
```bash
mkdir -p train_anomaly/checkpoints/
```
Chép file weight anomaly tải về vào đường dẫn: `train_anomaly/checkpoints/best.pt`

**2. Mô hình YOLOv8 (Detect):**
Tạo thư mục nếu chưa có:
```bash
mkdir -p runs/detect/yolo_atm_gender13/weights/
```
Chép file weight YOLO tải về vào đường dẫn: `runs/detect/yolo_atm_gender13/weights/best.pt`

## Cài đặt môi trường

1. Đảm bảo bạn đang sử dụng **Python 3.9+** (khuyên dùng Conda hoặc môi trường ảo ảo hoá `venv`).
2. Cài đặt các thư viện Python theo yêu cầu:
   ```bash
   pip install -r requirements.txt
   ```

## Hướng dẫn sử dụng

### 1. Xem và Phân tích trực tiếp trên Web (Web Dashboard)
Để bật server giao diện web và xem trực tiếp stream:
```bash
python app_web.py
```
- Mở trình duyệt web của bạn và truy cập: `http://localhost:5000` (hoặc cổng cấu hình tương ứng hiển thị ở Terminal).

### 2. Xử lý video qua Command Line (CLI App)
Nếu bạn không cần giao diện web mà để tiến hành tracking/report cho một video có sẵn:
```bash
python app.py --video <đường_dẫn_tới_video.mp4>
```
Để tự động nhận diện và xử lý tất cả các file có sẵn trong thư mục input đã định cấu hình:
```bash
python app.py --all
```

### 3. Huấn luyện lại mô hình (Training Models)

**Huấn luyện YOLOv8:**
Sử dụng file cấu hình bộ dữ liệu `atm_dataset.yaml` (hãy điều chỉnh lại các path trong file `yaml` sao cho phù hợp với máy tính của bạn trước khi bắt đầu):
```bash
python train_atm.py
```

**Huấn luyện Anomaly Model:**
Mô hình phát hiện bất thường sử dụng module PyTorch định nghĩa ở `train_anomaly/train.py`.
```bash
python train_anomaly/train.py
```
*Lưu ý: Bạn có thể cần chạy các script trích xuất dữ liệu như `prepare_dataset_v2.py` hoặc `enrich_and_resplit.py` tuỳ trạng thái dataset.*

## Cấu trúc thư mục thuật toán
- `atm_core.py`: Lõi Pipeline toàn bộ hệ thống (detection, tracking, check behavior).
- `app_web.py`: Máy chủ Flask xử lý luồng Web và Camera stream.
- `app.py`: Phiên bản CLI App phân tích offline.
- `train_atm.py`: Script kích hoạt train YOLOv8.
- `train_anomaly/`: Thư mục chứa kiến trúc Model, Dataset loader và vòng lặp huấn luyện mạng Anomaly.
- `templates/`: Giao diện HTML cho Web Frontend.
- `.gitignore`: Cấu hình Git để loại bỏ những file/folder siêu nặng, weights và cache khi push lên GitHub.
