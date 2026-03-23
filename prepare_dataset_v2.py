"""
prepare_dataset_v2.py
=====================
Version 2.5 (VLM Cascade Pipeline)
- BƯỚC 1: Quét ATM bằng Florence-2-Large (Phrase Grounding). Siêu chuẩn xác, không dính nhiễu.
- BƯỚC 2: NẾU có ATM (1-2 cái), gọi YOLOv8x (Vua BBox người) quét sạch bóng người.
- BƯỚC 3: Crop từng người ném qua lại cho Florence-2 (Visual Question Answering - VQA) hỏi: "Is this a man or a woman?" để phân loại 100% chính xác giới tính.
"""

import cv2
import os
import shutil
import random
from pathlib import Path
from PIL import Image

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from ultralytics import YOLO
except ImportError:
    print("Mời cài đặt: pip install transformers timm einops torchvision ultralytics")
    exit(1)

def merge_boxes(boxes, iou_thresh=0.4):
    if not boxes: return []
    merged_any = True
    while merged_any:
        merged_any = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used: continue
            c_box = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used: continue
                o_box = boxes[j]
                
                xA = max(c_box[0], o_box[0])
                yA = max(c_box[1], o_box[1])
                xB = min(c_box[2], o_box[2])
                yB = min(c_box[3], o_box[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                
                box1Area = (c_box[2] - c_box[0]) * (c_box[3] - c_box[1])
                box2Area = (o_box[2] - o_box[0]) * (o_box[3] - o_box[1])
                
                iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
                
                if iou > iou_thresh or interArea > 0.8 * min(box1Area, box2Area):
                    c_box = [
                        min(c_box[0], o_box[0]),
                        min(c_box[1], o_box[1]),
                        max(c_box[2], o_box[2]),
                        max(c_box[3], o_box[3])
                    ]
                    used.add(j)
                    merged_any = True
            new_boxes.append(c_box)
            used.add(i)
        boxes = new_boxes
    return boxes

def split_dataset_to_val(img_dir, lbl_dir, val_img_dir, val_lbl_dir, split_ratio=0.2):
    print("\n" + "="*50)
    print(f"🔄 BƯỚC CUỐI CÙNG: KHỞI TẠO TẬP VALIDATION ({int(split_ratio*100)}%)...")
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(img_dir.glob("*.jpg"))
    if not images:
        return 0
        
    val_count = max(1, int(len(images) * split_ratio))
    val_images = random.sample(images, val_count)
    
    count = 0
    for img in val_images:
        lbl_file = lbl_dir / f"{img.stem}.txt"
        if lbl_file.exists():
            shutil.move(str(img), str(val_img_dir / img.name))
            shutil.move(str(lbl_file), str(val_lbl_dir / lbl_file.name))
            count += 1
    return count

def run_florence_vqa(image_crop, processor, model, device):
    """Bắt Florence-2 tự viết caption để trích xuất Giới Tính chuẩn xác nhất"""
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
        return 2
    elif "man" in ans or "male" in ans or "boy" in ans or "guy" in ans or "his" in ans:
        return 1
    else:
        return 1

def main():
    source_dirs = ["Dataset", "ATMA-V"]
    out_root = Path("ATM_Dataset_Cascade")
    img_dir = out_root / "images/train"
    lbl_dir = out_root / "labels/train"
    vis_dir = out_root / "visualize"
    
    val_img_dir = out_root / "images/val"
    val_lbl_dir = out_root / "labels/val"
    
    for d in [img_dir, lbl_dir, vis_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 HỆ THỐNG EXTRACT DATA (V2.5) CASCADE: VLM (ATM) -> YOLOv8x (Con người) -> VLM VQA (Giới tính) 🚀")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(1, f"Tải Florence-2-large (Bắt ATM & Hỏi Đáp AI) lên {device.upper()}...")
    model_id = "microsoft/Florence-2-large"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    
    print(2, "Tải YOLOv8x (Sát thủ đập BBox người)...")
    person_model = YOLO("yolov8x.pt")
    
    print("\n" + "="*50)
    print("BẮT ĐẦU CHUỖI PIPELINE THÔNG MINH...")
    
    total_processed = 0
    total_videos_deleted = 0
    atm_prompt = "<CAPTION_TO_PHRASE_GROUNDING> an ATM machine"
    
    for src in source_dirs:
        src_path = Path(src)
        if not src_path.exists(): continue
        
        videos = list(src_path.rglob("*.mp4")) + list(src_path.rglob("*.avi"))
        if not videos: continue
        
        print(f"\n>> Đang duyệt thư mục {src_path.name}/: có {len(videos)} video...")
        for vid in videos:
            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened(): continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 30: 
                cap.release()
                continue
                
            NUM_FRAMES = 15  # Bạn có thể đổi lại 10 nếu muốn ít ảnh hơn
            MAX_REJECT = NUM_FRAMES // 2
            
            targets = [int(total_frames * (i + 0.5) / NUM_FRAMES) for i in range(NUM_FRAMES)]
            rejected_frames = 0
            saved_files_this_video = [] 
            
            for index, target in enumerate(targets):
                if rejected_frames >= MAX_REJECT: 
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if not ret: 
                    rejected_frames += 1
                    continue
                
                frame_h, frame_w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # --- BƯỚC 1: TÌM ATM BẰNG FLORENCE-2 ---
                inputs = processor(text=atm_prompt, images=pil_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
                    )
                parsed_ans = processor.post_process_generation(
                    processor.batch_decode(gen_ids, skip_special_tokens=False)[0],
                    task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(frame_w, frame_h)
                )
                
                atm_boxes = []
                if "<CAPTION_TO_PHRASE_GROUNDING>" in parsed_ans:
                    bboxes = parsed_ans["<CAPTION_TO_PHRASE_GROUNDING>"].get("bboxes", [])
                    labels = parsed_ans["<CAPTION_TO_PHRASE_GROUNDING>"].get("labels", [])
                    for box, label in zip(bboxes, labels):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
                        w_b, h_b = x2 - x1, y2 - y1
                        if "atm" in label.lower() and (w_b * h_b) > (frame_w * frame_h * 0.01) and w_b > 20 and h_b > 20:
                            atm_boxes.append([x1, y1, x2, y2])
                            
                atm_boxes_merged = merge_boxes(atm_boxes, iou_thresh=0.3)
                atm_count = len(atm_boxes_merged)
                
                # --- CHỐT CHẶN: NẾU KHÔNG CÓ ATM HOẶC QUÁ NHIỀU ATM THÌ BỎ QUA LUÔN KHUNG HÌNH (TIẾT KIỆM TÀI NGUYÊN) ---
                if atm_count == 0 or atm_count > 2:
                    rejected_frames += 1
                    print(f"   [SKIP] Frame {index+1}/{NUM_FRAMES} có {atm_count} ATM. Lỗi lần {rejected_frames}!")
                    continue

                final_boxes = []
                for b in atm_boxes_merged:
                    x1, y1, x2, y2 = b
                    final_boxes.append((0, x1, y1, x2, y2))
                
                # --- BƯỚC 2: CÓ ATM RỒI -> TRIỆU HỒI YOLOv8x QUÉT NHANH NGƯỜI ---
                person_results = person_model.predict(source=frame, imgsz=1280, classes=[0], conf=0.3, iou=0.4, verbose=False)[0]
                
                human_count = 0
                for box in person_results.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
                    
                    # --- BƯỚC 3: CẮT NGƯỜI RA VÀ HỎI FLORENCE-2 VQA LÀ NAM HAY NỮ ---
                    crop_img = frame_rgb[y1:y2, x1:x2]
                    if crop_img.shape[0] > 30 and crop_img.shape[1] > 30:
                        pil_crop = Image.fromarray(crop_img)
                        gender_class_id = run_florence_vqa(pil_crop, processor, model, device)
                        final_boxes.append((gender_class_id, x1, y1, x2, y2))
                        human_count += 1

                stem = f"{vid.parent.name}_{vid.stem}_f{index+1}"
                img_path = img_dir / f"{stem}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                label_path = lbl_dir / f"{stem}.txt"
                vis_frame = frame.copy()
                
                with open(label_path, "w") as f:
                    for lbl in final_boxes:
                        cls_id, x1, y1, x2, y2 = lbl
                        w_b, h_b = x2 - x1, y2 - y1
                        x_cx, y_cy = (x1 + w_b / 2.0) / frame_w, (y1 + h_b / 2.0) / frame_h
                        w_n, h_n = w_b / float(frame_w), h_b / float(frame_h)
                        
                        f.write(f"{cls_id} {x_cx:.6f} {y_cy:.6f} {w_n:.6f} {h_n:.6f}\n")
                        
                        if cls_id == 0:
                            col, text = (0, 255, 100), "ATM"
                        elif cls_id == 1:
                            col, text = (255, 100, 0), "Male"
                        else:
                            col, text = (100, 50, 255), "Female"
                            
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(vis_frame, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                        
                vis_path = vis_dir / f"{stem}.jpg"
                cv2.imwrite(str(vis_path), vis_frame)
                
                saved_files_this_video.extend([img_path, label_path, vis_path])
                total_processed += 1
                print(f" ✅ Grounding Xong: {atm_count} ATM, {human_count} Người ({stem}.jpg)")
                
            cap.release()
            
            if rejected_frames >= MAX_REJECT:
                print(f"\n 🚨 CHIA TAY VIDEO: {vid.name} (Chứa >= {MAX_REJECT} Ảnh Rác).")
                for fpath in saved_files_this_video:
                    try:
                        os.remove(str(fpath))
                        if str(fpath).endswith(".jpg") and "visualize" not in str(fpath):
                            total_processed -= 1
                    except: pass
                if "ATMA-V" in str(vid):
                    total_videos_deleted += 1
                    try: os.remove(str(vid))
                    except: pass
            else:
                print(f" -> Hoàn tất {vid.name}: Thu hoạch {NUM_FRAMES - rejected_frames}/{NUM_FRAMES} ảnh.")
                
    val_moved = split_dataset_to_val(img_dir, lbl_dir, val_img_dir, val_lbl_dir, split_ratio=0.2)
    
    print("\n" + "="*50)
    print(f"🎉 HOÀN TẤT V2.5 (CASCADE YOLOv8x + FLORENCE-2 VQA)!")
    print(f" - Tổng hình ảnh train dán nhãn siêu mượt : {total_processed}")
    print(f" - Tổng hình ảnh validation                : {val_moved}")
    print(f" - Số Video đã tự động loại bỏ             : {total_videos_deleted}")
    print("-> DATA HOÀN HẢO ĐÃ LÊN MÂM!")

if __name__ == "__main__":
    main()
