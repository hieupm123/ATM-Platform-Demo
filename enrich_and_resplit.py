"""
enrich_and_resplit.py
=====================
Script làm 2 việc cùng lúc:

BƯỚC 1: Dọn sạch ảnh v1i cũ (pattern *.rf.* trong tên) từ ATM_Dataset_Cascade.

BƯỚC 2: Xử lý bộ ATM.v1i.yolov8
  - Giữ nguyên bounding box ATM (class 0) từ nhãn gốc của Roboflow.
  - Dùng YOLOv8x để phát hiện người trong từng ảnh.
  - Dùng Florence-2 VQA để phân loại giới tính (Male=1, Female=2).
  - Ghi nhãn mới ra file .txt chứa CẢ ATM lẫn Male/Female.
  - Lưu ảnh visualize vào ATM_Dataset_Cascade/visualize/

BƯỚC 3: Chia lại Train/Val 9:1
  - Tập Val: 10% ảnh v1i (100% val là từ v1i)
  - Tập Train: Cascade cũ + 90% ảnh v1i còn lại
"""

import cv2, os, shutil, random
from pathlib import Path
from PIL import Image

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from ultralytics import YOLO
except ImportError:
    print("Mời cài đặt: pip install transformers timm einops torchvision ultralytics")
    exit(1)

# ==================== HÀM DÙNG LẠI TỪ prepare_dataset_v2.py ====================

def run_florence_vqa(image_crop, processor, model, device):
    """Dùng Florence-2 viết caption để xác định giới tính."""
    prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image_crop, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=30,
            num_beams=3
        )
    ans = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
    if "woman" in ans or "female" in ans or "girl" in ans or "lady" in ans or "her" in ans:
        return 2  # Female
    return 1      # Male (mặc định)

# ==================== CÀI ĐẶT ====================

V1I_DIR   = Path("ATM.v1i.yolov8/train")
V1I_IMG   = V1I_DIR / "images"
V1I_LBL   = V1I_DIR / "labels"

CASCADE   = Path("ATM_Dataset_Cascade")
TRAIN_IMG = CASCADE / "images/train"
TRAIN_LBL = CASCADE / "labels/train"
VAL_IMG   = CASCADE / "images/val"
VAL_LBL   = CASCADE / "labels/val"
VIS_DIR   = CASCADE / "visualize"

VAL_RATIO = 0.1

# ==================== BƯỚC 1: DỌN V1I CŨ ====================

def cleanup_old_v1i():
    """
    File v1i trong Cascade có dạng tên: TMB..._jpg.rf.abc123...
    Pattern nhận biết: '.rf.' trong tên file (không có trong ảnh Cascade tự tạo).
    """
    count = 0
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, VIS_DIR]:
        if not d.exists(): continue
        for f in d.iterdir():
            if ".rf." in f.name:
                f.unlink()
                count += 1
    print(f"  -> Đã xóa {count} file v1i cũ (pattern .rf.).")

# ==================== BƯỚC 2+3: ENRICH + RESPLIT ====================

def main():
    if not V1I_IMG.exists():
        print(f"Không tìm thấy {V1I_IMG}")
        return

    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, VIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n🚀 BẮT ĐẦU ENRICH v1i + RESPLIT DATASET")
    print("=" * 55)
    
    print("\n[0/4] Dọn sạch ảnh v1i cũ trong Cascade...")
    cleanup_old_v1i()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[1/4] Tải Florence-2-large lên {device.upper()}...")
    model_id = "microsoft/Florence-2-large"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

    print("[2/4] Tải YOLOv8x (phát hiện người)...")
    person_model = YOLO("yolov8x.pt")

    all_v1i_images = sorted(
        list(V1I_IMG.glob("*.jpg")) +
        list(V1I_IMG.glob("*.jpeg")) +
        list(V1I_IMG.glob("*.png"))
    )
    n_total = len(all_v1i_images)
    n_val   = max(1, int(n_total * VAL_RATIO))

    random.shuffle(all_v1i_images)
    val_set   = set(all_v1i_images[:n_val])
    train_set = all_v1i_images[n_val:]

    print(f"\n[3/4] Chia {n_total} ảnh: Val={n_val}, Train={len(train_set)}")

    print(f"\n[4/4] Xử lý từng ảnh (YOLO + Florence-2)...")
    
    for idx, img_path in enumerate(all_v1i_images, 1):
        # --- Đọc nhãn ATM gốc của Roboflow ---
        lbl_path = V1I_LBL / f"{img_path.stem}.txt"
        atm_labels = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and int(parts[0]) == 0:
                        atm_labels.append(line.strip())

        # --- Đọc ảnh ---
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Không đọc được: {img_path.name}")
            continue

        frame_h, frame_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Detect người bằng YOLOv8x ---
        results = person_model.predict(
            source=frame, imgsz=1280, classes=[0], conf=0.3, iou=0.4, verbose=False
        )[0]

        gender_labels = []
        vis_frame = frame.copy()
        
        # Vẽ lại ATM box từ nhãn gốc (để visualize)
        for atm_line in atm_labels:
            parts = atm_line.split()
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w/2) * frame_w); x2 = int((cx + w/2) * frame_w)
            y1 = int((cy - h/2) * frame_h); y2 = int((cy + h/2) * frame_h)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(vis_frame, "ATM", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            crop = frame_rgb[y1:y2, x1:x2]
            if crop.shape[0] > 30 and crop.shape[1] > 30:
                gender = run_florence_vqa(Image.fromarray(crop), processor, florence_model, device)
                w_b, h_b = x2 - x1, y2 - y1
                x_cx = (x1 + w_b / 2.0) / frame_w
                y_cy = (y1 + h_b / 2.0) / frame_h
                w_n  = w_b / float(frame_w)
                h_n  = h_b / float(frame_h)
                gender_labels.append(f"{gender} {x_cx:.6f} {y_cy:.6f} {w_n:.6f} {h_n:.6f}")

                # Vẽ lên vis_frame
                if gender == 1:
                    col, text = (255, 100, 0), "Male"
                else:
                    col, text = (100, 50, 255), "Female"
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis_frame, text, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        all_labels = atm_labels + gender_labels

        # --- Xác định split và ghi file ---
        stem = img_path.stem
        sfx  = img_path.suffix
        if img_path in val_set:
            dst_img = VAL_IMG   / f"{stem}{sfx}"
            dst_lbl = VAL_LBL   / f"{stem}.txt"
            split_str = "VAL"
        else:
            dst_img = TRAIN_IMG / f"{stem}{sfx}"
            dst_lbl = TRAIN_LBL / f"{stem}.txt"
            split_str = "TRAIN"

        shutil.copy2(str(img_path), str(dst_img))
        with open(dst_lbl, "w") as f:
            f.write("\n".join(all_labels) + "\n" if all_labels else "")

        # Visualize
        dst_vis = VIS_DIR / f"{stem}{sfx}"
        cv2.imwrite(str(dst_vis), vis_frame)

        print(f"  [{idx:4d}/{n_total}] [{split_str}] {img_path.name}: "
              f"{len(atm_labels)} ATM, {len(gender_labels)} người")

    print("\n" + "=" * 55)
    print(f"🎉 HOÀN TẤT!")
    print(f" 📂 Val       : {n_val} ảnh v1i enriched")
    print(f" 📂 Train (v1i): {len(train_set)} ảnh v1i enriched gộp vào")
    print(f" 🖼  Visualize  : {VIS_DIR}")
    print("=" * 55)

if __name__ == "__main__":
    main()
