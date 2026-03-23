"""
dataset.py
==========
Gộp toàn bộ pipeline dữ liệu cho ATMA-V anomaly detection:
  - parse_labels     : đọc labels.txt → frame-level labels
  - extract_clips    : trích xuất clip .npz từ video (YOLO tracking)
  - ATMAClipDataset  : PyTorch Dataset đọc clip .npz

Labels:
  0 = normal   (không phải đoạn bất thường)
  1 = anomaly  (đoạn bất thường được đánh nhãn)

Cách dùng trực tiếp:
  python train_anomaly/dataset.py          # parse labels + xem thống kê
  python train_anomaly/dataset.py --extract           # trích xuất clips
  python train_anomaly/dataset.py --extract --no_model
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Đường dẫn mặc định ────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent.parent
LABELS_FILE  = _ROOT / "ATMA-V" / "labels" / "labels.txt"
VIDEOS_DIR   = _ROOT / "ATMA-V" / "videos"
CLIPS_DIR    = Path(__file__).parent / "clips"
CUSTOM_MODEL = _ROOT / "runs" / "detect" / "yolo_atm_gender13" / "weights" / "best.pt"

# ── Hằng số ───────────────────────────────────────────────────────────────────
LABEL_NORMAL  = 0
LABEL_ANOMALY = 1
LABEL_MAP     = {"normal": LABEL_NORMAL, "anomaly": LABEL_ANOMALY}

CLIP_FRAMES    = 16          # số frame mỗi clip (temporal window)
CLIP_STRIDE    = 8           # bước trượt sliding window
CROP_SIZE      = (224, 224)  # (W, H) cho OpenCV resize
PERSON_CLASSES = {"male", "female", "person"}
SKIP_FRAMES    = 2           # xử lý mỗi SKIP_FRAMES frame
MIN_BOX_H_RATIO = 0.10       # bỏ bbox nhỏ hơn 10% chiều cao frame


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — PARSE LABELS
# ══════════════════════════════════════════════════════════════════════════════

def parse_labels_txt(labels_path: Path = LABELS_FILE) -> Dict[str, dict]:
    """
    Đọc labels.txt → dict keyed by video stem (không có .mp4).

    Format labels.txt:
      <video>  <total_frames>  <fps>  <start1> <end1> [<start2> <end2> ...]
      Nếu không có đoạn bất thường: <video>  <total_frames>  <fps>  -1 -1

    Trả về dict với mỗi video:
      {
        "total_frames": int,
        "fps": float,
        "anomaly_windows": [(s, e), ...],   # frame ranges bất thường
        "frame_labels": np.ndarray(N,) int8  # 0=normal  1=anomaly
      }
    """
    result: Dict[str, dict] = {}

    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            video_name = parts[0]
            try:
                total_frames = int(parts[1])
                fps          = float(parts[2])
            except ValueError:
                continue

            rest = parts[3:]
            anomaly_windows: List[Tuple[int, int]] = []
            i = 0
            while i + 1 < len(rest):
                s, e = int(rest[i]), int(rest[i + 1])
                i += 2
                if s == -1 or e == -1:
                    break          # không có đoạn bất thường
                anomaly_windows.append((s, e))

            # Labels trong file là 1-indexed:
            #   frame 1 = frame đầu tiên của video → chuyển sang 0-indexed: s-1, e-1
            #   slice [s-1 : e] tương đương frame 0-indexed từ (s-1) đến (e-1) inclusive
            frame_labels = np.zeros(total_frames, dtype=np.int8)
            for (s, e) in anomaly_windows:
                frame_labels[max(0, s - 1): min(total_frames, e)] = LABEL_ANOMALY

            stem = Path(video_name).stem
            result[stem] = {
                "video_file":      video_name,
                "total_frames":    total_frames,
                "fps":             fps,
                "anomaly_windows": anomaly_windows,
                "has_anomaly":     len(anomaly_windows) > 0,
                "frame_labels":    frame_labels,
            }

    return result


def get_video_path(video_stem: str) -> Optional[Path]:
    """Trả về Path video nếu tồn tại."""
    p = VIDEOS_DIR / f"{video_stem}.mp4"
    return p if p.exists() else None


def print_dataset_summary(parsed: Dict[str, dict]):
    total       = len(parsed)
    with_anom   = sum(1 for v in parsed.values() if v["has_anomaly"])
    anom_frames = sum(
        sum(e - s + 1 for s, e in v["anomaly_windows"])
        for v in parsed.values()
    )
    total_frames = sum(v["total_frames"] for v in parsed.values())
    print(f"\n{'='*55}")
    print(f"  ATMA-V Dataset Summary")
    print(f"{'='*55}")
    print(f"  Tổng số video        : {total}")
    print(f"  Video có anomaly     : {with_anom}")
    print(f"  Video KHÔNG anomaly  : {total - with_anom}")
    print(f"  Tổng frames          : {total_frames:,}")
    print(f"  Frames anomaly       : {anom_frames:,}  ({anom_frames/max(total_frames,1)*100:.1f}%)")
    print(f"  Frames normal        : {total_frames - anom_frames:,}  "
          f"({(total_frames-anom_frames)/max(total_frames,1)*100:.1f}%)")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — EXTRACT CLIPS
# ══════════════════════════════════════════════════════════════════════════════

def _expand_box(x1, y1, x2, y2, fw, fh, ratio=0.08):
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(fw, x2 + pad_x), min(fh, y2 + pad_y))


def _crop_resize(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop + pad về hình vuông + resize về CROP_SIZE, tránh bóp méo."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*CROP_SIZE[::-1], 3), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    side   = max(ch, cw)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[(side - ch) // 2:(side - ch) // 2 + ch,
           (side - cw) // 2:(side - cw) // 2 + cw] = crop
    return cv2.resize(canvas, CROP_SIZE)


def extract_clips_from_video(
    video_stem: str,
    info: dict,
    model,
    out_dir: Path,
    skip: int = SKIP_FRAMES,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Trích xuất clips từ 1 video, lưu dưới dạng .npz.
    Trả về (n_anomaly_clips, n_normal_clips).
    """
    vpath = get_video_path(video_stem)
    if vpath is None:
        if verbose:
            print(f"  [SKIP] Không tìm thấy: {video_stem}.mp4")
        return 0, 0

    frame_labels = info["frame_labels"]
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"  [ERROR] Không mở được: {vpath}")
        return 0, 0

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # track_crops[tid] = [(frame_idx, crop_img, label), ...]
    track_crops: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        label_val = int(frame_labels[min(frame_idx, len(frame_labels) - 1)])

        if model is not None:
            results = model.track(
                frame, conf=0.30, iou=0.45,
                tracker="bytetrack.yaml", persist=True, verbose=False
            )[0]
            for box in results.boxes:
                if box.id is None:
                    continue
                cls_name = model.names.get(int(box.cls[0]), "").lower()
                if not any(p in cls_name for p in PERSON_CLASSES):
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                if (y2 - y1) < fh * MIN_BOX_H_RATIO:
                    continue
                tid  = int(box.id[0])
                crop = _crop_resize(frame, _expand_box(x1, y1, x2, y2, fw, fh))
                track_crops.setdefault(tid, []).append((frame_idx, crop, label_val))
        else:
            # Fallback: dùng full frame
            crop = cv2.resize(frame, CROP_SIZE)
            track_crops.setdefault(0, []).append((frame_idx, crop, label_val))

        frame_idx += 1

    cap.release()

    # Sliding window → lưu .npz
    n_anom = 0
    n_norm = 0
    for tid, entries in track_crops.items():
        if len(entries) < CLIP_FRAMES:
            continue
        i = 0
        while i + CLIP_FRAMES <= len(entries):
            window     = entries[i: i + CLIP_FRAMES]
            clips_arr  = np.stack([e[1] for e in window], axis=0)  # (T, H, W, C)
            labels_win = [e[2] for e in window]
            clip_label = int(np.bincount(labels_win).argmax())      # nhãn đa số

            label_name = "anomaly" if clip_label == LABEL_ANOMALY else "normal"
            save_dir   = out_dir / label_name
            save_dir.mkdir(parents=True, exist_ok=True)

            fname = f"{video_stem}_t{tid}_f{entries[i][0]}.npz"
            np.savez_compressed(save_dir / fname, clips=clips_arr, label=clip_label)

            if clip_label == LABEL_ANOMALY:
                n_anom += 1
            else:
                n_norm += 1
            i += CLIP_STRIDE

    if verbose:
        print(f"  ✅ {video_stem}: {n_anom} anomaly + {n_norm} normal clips")
    return n_anom, n_norm


def run_extract(
    out_dir: Path = CLIPS_DIR,
    use_model: bool = True,
    skip: int = SKIP_FRAMES,
    videos: Optional[List[str]] = None,
) -> bool:
    """
    Trích xuất toàn bộ clips từ ATMA-V.
    Nếu out_dir đã có dữ liệu → bỏ qua (trả về False).
    Nếu thực sự extract → trả về True.
    """
    # Kiểm tra nếu đã có clips
    existing = list(out_dir.glob("**/*.npz"))
    if existing:
        print(f"[Extract] Đã tìm thấy {len(existing)} clips trong {out_dir} → bỏ qua extract.")
        return False

    print("[Extract] Chưa có clips → bắt đầu trích xuất ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse labels
    print("[1/3] Đọc labels.txt ...")
    parsed = parse_labels_txt()

    # Load YOLO (nếu cần)
    model = None
    if use_model:
        try:
            from ultralytics import YOLO  # type: ignore
            if CUSTOM_MODEL.exists():
                print(f"[2/3] Load YOLO: {CUSTOM_MODEL}")
                model = YOLO(str(CUSTOM_MODEL))
            else:
                print(f"[2/3] Không tìm thấy YOLO weights → dùng full frame")
        except ImportError:
            print("[2/3] ultralytics chưa cài → dùng full frame")
    else:
        print("[2/3] --no_model → dùng full frame")

    # Extract
    print(f"[3/3] Trích xuất clips → {out_dir} ...")
    stems = videos if videos else list(parsed.keys())
    total_anom = 0
    total_norm = 0
    for stem in stems:
        if stem not in parsed:
            print(f"  [SKIP] {stem} không có trong labels.txt")
            continue
        a, n = extract_clips_from_video(stem, parsed[stem], model, out_dir, skip=skip)
        total_anom += a
        total_norm += n

    print(f"\n{'='*50}")
    print(f"  Clips anomaly : {total_anom}")
    print(f"  Clips normal  : {total_norm}")
    print(f"  Saved to      : {out_dir}")
    print(f"{'='*50}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ATMAClipDataset(Dataset):
    """
    Dataset trả về (clip_tensor, label) cho mỗi sample.
      clip_tensor: (T, C, H, W)  float32  [0, 1]
      label      : int  (0=normal, 1=anomaly)
    """

    def __init__(
        self,
        clips_dir: Path = CLIPS_DIR,
        augment: bool = True,
        clip_frames: int = CLIP_FRAMES,
    ):
        """
        Load tất cả clips từ clips_dir.
        Việc chia train/val được thực hiện bên ngoài (qua _video_level_split)
        và ghi đè vào self.samples sau khi khởi tạo.
        """
        super().__init__()
        self.augment     = augment
        self.clip_frames = clip_frames

        all_samples: List[Tuple[Path, int]] = []
        for label_name, label_val in LABEL_MAP.items():
            label_dir = Path(clips_dir) / label_name
            if not label_dir.exists():
                continue
            for fp in sorted(label_dir.glob("*.npz")):
                all_samples.append((fp, label_val))

        if not all_samples:
            raise RuntimeError(
                f"Không tìm thấy clip nào trong {clips_dir}.\n"
                "Hãy chạy dataset.py --extract hoặc train.py (tự động extract)."
            )

        # self.samples sẽ bị override bởi create_dataloaders sau khi video-level split
        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fp, label = self.samples[idx]
        data  = np.load(fp)
        clips = data["clips"]  # (T, H, W, C) uint8

        T = clips.shape[0]
        if T >= self.clip_frames:
            start = random.randint(0, T - self.clip_frames) if self.augment else 0
            clips = clips[start: start + self.clip_frames]
        else:
            pad   = self.clip_frames - T
            clips = np.concatenate(
                [clips, np.zeros((pad, *clips.shape[1:]), dtype=np.uint8)], axis=0
            )

        if self.augment:
            if random.random() < 0.5:
                clips = clips[:, :, ::-1, :].copy()           # horizontal flip
            if random.random() < 0.3:
                factor = random.uniform(0.8, 1.2)
                clips  = np.clip(clips.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(clips).permute(0, 3, 1, 2).float() / 255.0
        return tensor, label

    def get_class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=2).astype(float)
        counts = np.clip(counts, 1, None)
        w = 1.0 / counts
        return torch.tensor(w / w.sum(), dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        labels = [s[1] for s in self.samples]
        cw     = self.get_class_weights().numpy()
        return WeightedRandomSampler(
            weights     = [cw[l] for l in labels],
            num_samples = len(labels),
            replacement = True,
        )


def _video_level_split(
    clips_dir: Path,
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Chia train/val theo VIDEO (không theo clip) để tránh data leakage.
    Các clip từ cùng 1 video sẽ CHỈ thuộc train hoặc val, không bị chồng lấp.

    Cách nhận biết video từ tên file .npz:
      "005_t1_f160.npz"  →  video stem = "005"
      "10_t2_f0.npz"    →  video stem = "10"
    """
    from collections import defaultdict

    # Thu thập tất cả clips, group theo video stem
    video_clips: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    for label_name, label_val in LABEL_MAP.items():
        label_dir = Path(clips_dir) / label_name
        if not label_dir.exists():
            continue
        for fp in sorted(label_dir.glob("*.npz")):
            # Tên file: <video_stem>_t<tid>_f<frame>.npz  → lấy phần trước "_t"
            video_stem = fp.stem.split("_t")[0]
            video_clips[video_stem].append((fp, label_val))

    # Shuffle theo video stem (đơn vị chia)
    video_stems = sorted(video_clips.keys())
    rng = random.Random(seed)
    rng.shuffle(video_stems)

    n_val = max(1, int(len(video_stems) * val_ratio))
    val_stems   = set(video_stems[:n_val])
    train_stems = set(video_stems[n_val:])

    train_samples = [s for v in train_stems for s in video_clips[v]]
    val_samples   = [s for v in val_stems   for s in video_clips[v]]

    print(f"  Video split  : {len(train_stems)} train videos / {len(val_stems)} val videos")
    print(f"  Val videos   : {sorted(val_stems)}")
    return train_samples, val_samples


def create_dataloaders(
    clips_dir: Path = CLIPS_DIR,
    batch_size: int = 16,
    val_ratio: float = 0.20,
    num_workers: int = 4,
    clip_frames: int = CLIP_FRAMES,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo train/val DataLoader với split theo video để tránh data leakage.
    """
    train_samples, val_samples = _video_level_split(clips_dir, val_ratio)

    train_ds = ATMAClipDataset(clips_dir, augment=True,  clip_frames=clip_frames)
    val_ds   = ATMAClipDataset(clips_dir, augment=False, clip_frames=clip_frames)

    # Override samples sau khi split theo video
    train_ds.samples = train_samples
    val_ds.samples   = val_samples

    if not train_ds.samples:
        raise RuntimeError("Train set rỗng sau khi split theo video. Cần ít nhất 2 video.")
    if not val_ds.samples:
        raise RuntimeError("Val set rỗng sau khi split theo video. Cần ít nhất 2 video.")

    train_counts = np.bincount([s[1] for s in train_samples], minlength=2)
    val_counts   = np.bincount([s[1] for s in val_samples],   minlength=2)
    print(f"  Train samples: {len(train_samples)}  (normal={train_counts[0]}, anomaly={train_counts[1]})")
    print(f"  Val samples  : {len(val_samples)}   (normal={val_counts[0]}, anomaly={val_counts[1]})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_ds.get_sampler(),
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATMA-V data pipeline")
    parser.add_argument("--extract",   action="store_true", help="Chạy extract clips")
    parser.add_argument("--no_model",  action="store_true", help="Không dùng YOLO (full frame)")
    parser.add_argument("--out_dir",   type=str, default=str(CLIPS_DIR))
    parser.add_argument("--skip",      type=int, default=SKIP_FRAMES)
    parser.add_argument("--videos",    nargs="*", default=None)
    args = parser.parse_args()

    parsed = parse_labels_txt()
    print_dataset_summary(parsed)

    if args.extract:
        run_extract(
            out_dir   = Path(args.out_dir),
            use_model = not args.no_model,
            skip      = args.skip,
            videos    = args.videos,
        )
