"""
train_atm.py
============
Script tự động training mô hình YOLOv8 cho ATM, Male, Female.
"""

from ultralytics import YOLO

def main():
    # 1. Load model pre-train nhỏ nhất để train cho nhanh (hoặc dùng yolov8s.pt, yolov8m.pt)
    print("Đang tải YOLOv8 Nano pre-trained...")
    model = YOLO("yolov8n.pt")
    
    # 2. Bắt đầu quá trình Training
    # Điều chỉnh epochs (số vòng lặp) và imgsz (kích thước ảnh) tùy vào sức mạnh GPU/CPU của bạn
    print("Bắt đầu Training. Quá trình này có thể tốn vài chục phút đến vài giờ...")
    results = model.train(
        data="atm_dataset.yaml",  # File cấu hình Dataset
        epochs=100,               # Số vòng lặp (Nên để 100-300 vòng)
        imgsz=640,                # Kích thước ảnh chuẩn 640x640
        batch=16,                 # Số hình đưa vào cùng lúc (giảm xuống 8 nếu VRAM yếu)
        name="yolo_atm_gender",   # Tên thư mục lưu kết quả model
        device="",                # Chọn GPU (vd: 0), để trống "" sẽ tự nhận diện hoặc dùng CPU
        workers=4,                # Đặt = 0 để tránh lỗi RAM / đa luồng gây Segmentation Fault
        amp=False,                # Tắt AMP để sửa lỗi Segmentation fault của Pytorch 1.13
        plots=False,              # Tắt phần vẽ biểu đồ labels vì Seaborn hay bị Segfault trên K8s nếu data lớn
        patience=20               # Tự động dừng sớm nếu sau 20 vòng không có tiến triển
    )
    
    print("\n✅ Training hoàn tất!")
    print("File Weights tốt nhất của bạn sẽ được lưu theo đường dẫn:")
    print("👉 runs/detect/yolo_atm_gender/weights/best.pt")
    print("Hãy copy file best.pt đó và đổi tên thành yolo_atm_gender.pt để lắp vào Pipeline.")

if __name__ == "__main__":
    main()
