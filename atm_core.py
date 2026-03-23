"""
atm_core.py
===========
ATM Surveillance pipeline core logic.
Uses a single Custom YOLO model for unified detection of:
- atm  (Dynamic Zone Detection)
- male / female  (Person Tracking & Gender Classification)

Tích hợp thêm:
- ATMAnomalyModel (MobileNetV3-Small + GRU) để phát hiện hành vi bất thường
  theo từng track. Model chạy mỗi ML_INFER_STRIDE frame trên rolling buffer
  16 frame crop của từng người. Nếu chưa có checkpoint → tự động bỏ qua.
"""

from __future__ import annotations
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Deque
from datetime import datetime, timedelta
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from ultralytics import YOLO

# ── Configuration Custom YOLO Model ──────────────────────────────────────────
CUSTOM_MODEL_PATH = "/home/hieu/Project_Demo_NganHang/runs/detect/yolo_atm_gender13/weights/best.pt"
OUTPUT_DIR        = Path("output")
TRACKING_CONF     = 0.35
ATM_CONF          = 0.15
IOU_TRACKER       = 0.45

# ── Configuration ML Anomaly Model ───────────────────────────────────────────
ANOMALY_MODEL_PATH  = Path(__file__).parent / "train_anomaly" / "checkpoints" / "best.pt"
ML_ANOMALY_THRESH   = 0.60   # confidence ngưỡng để coi là bất thường
ML_INFER_STRIDE     = 8      # infer mỗi N frame (tránh infer mọi frame)
ML_BUFFER_SIZE      = 16     # số frame trong rolling buffer (khớp với clip_frames training)
ML_CROP_SIZE        = (224, 224)  # phải khớp với CROP_SIZE trong dataset.py

# ── Cấu hình ngưỡng hành vi ───────────────────────────────────────────────────
CFG = {
    "loiter_seconds":        12.0,
    "suspicious_dist":        0.15,
    "min_session_sec":         2.0,
    "machine_contact_sec":     5.0,
    "normal_transact_sec":    10.0,
    "fps":                    25.0,
    "tx_candidate_frames":   10,
    "tx_confirm_frames":     20,
    "tx_exit_frames":        15,
    "tx_score_enter":        0.52,
    "tx_score_exit":         0.30,
    "w_arm_overlap":         0.40,
    "w_low_movement":        0.30,
    "w_vert_proximity":      0.30,
}

BEHAVIOR_PRIORITY = {
    "machine_contact": 6,
    "suspicious":      5,
    "loitering":       4,
    "transacting":     3,
    "queuing":         2,
    "unknown":         1,
}

TX_IDLE      = "idle"
TX_CANDIDATE = "candidate"
TX_ACTIVE    = "transacting"

BEHAVIOR_VI = {
    "transacting":     "Đang thực hiện giao dịch",
    "queuing":         "Đứng xếp hàng",
    "loitering":       "Đứng lâu không giao dịch",
    "suspicious":      "Đứng gần/theo dõi người khác",
    "machine_contact": "Có tiếp xúc/động chạm vào máy ATM",
    "unknown":         "Không xác định",
}

BEHAVIOR_COLORS = {
    "transacting": (50, 205, 50),
    "queuing":     (0, 200, 255),
    "loitering":   (0, 140, 255),
    "suspicious":  (0, 60, 220),
    "contact":     (0, 0, 220),
    "unknown":     (130, 130, 130),
}

BEHAVIOR_LABEL = {
    "transacting":     "Giao dich",
    "queuing":         "Xep hang",
    "loitering":       "Lang vang",
    "suspicious":      "Nghi ngo",
    "contact":         "Cham may",
    "machine_contact": "Cham may",
    "unknown":         "Khong ro",
}

def filter_transacting_candidates(frame_dets: list, atm_boxes: list):
    tx_candidates = [d for d in frame_dets if d["behavior"] == "transacting"]
    if not tx_candidates:
        return

    atm_centers = []
    if atm_boxes:
        for box in atm_boxes:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            atm_centers.append((cx, cy))
    else:
        atm_centers = [(0.5, 0.5)]

    from collections import defaultdict
    atm_to_candidates = defaultdict(list)
    for d in tx_candidates:
        cx, cy = d["cx"], d["cy"]
        best_atm_idx, min_dist = -1, float('inf')
        for i, (acx, acy) in enumerate(atm_centers):
            dist = (cx - acx)**2 + (cy - acy)**2
            if dist < min_dist:
                min_dist = dist
                best_atm_idx = i
        atm_to_candidates[best_atm_idx].append((min_dist, d))

    allowed_tx_ids = set()
    for atm_idx, cand_list in atm_to_candidates.items():
        cand_list.sort(key=lambda x: x[0])
        allowed_tx_ids.add(cand_list[0][1]["track_id"])

    for d in frame_dets:
        if d["behavior"] == "transacting" and d["track_id"] not in allowed_tx_ids:
            d["behavior"] = "queuing"


# ══════════════════════════════════════════════════════════════════════════════
# ML ANOMALY MODEL — lazy load
# ══════════════════════════════════════════════════════════════════════════════

def load_anomaly_model(device: str = "cpu"):
    """
    Load ATMAnomalyModel từ checkpoint.
    Trả về model nếu thành công, None nếu không có checkpoint hoặc lỗi.
    """
    if not ANOMALY_MODEL_PATH.exists():
        print(f"[ML] Không tìm thấy checkpoint: {ANOMALY_MODEL_PATH}")
        print("[ML] Pipeline vẫn chạy bình thường không có ML anomaly detection.")
        return None, device
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from train_anomaly.train import load_model as _load
        model = _load(ANOMALY_MODEL_PATH, device=device)
        model.eval()
        print(f"[ML] ✅ Loaded anomaly model: {ANOMALY_MODEL_PATH}")
        return model, device
    except Exception as e:
        print(f"[ML] ⚠ Không load được model: {e}")
        return None, device


def get_iou(boxA: Tuple[float,float,float,float], boxB: Tuple[float,float,float,float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def _crop_resize_ml(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    fw: int, fh: int, ratio: float = 0.08) -> np.ndarray:
    """
    Crop bbox người (với padding context), pad về hình vuông, resize → ML_CROP_SIZE.
    Output: (H, W, C) uint8  [tương thích với extract_clips.py]
    """
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    cx1 = max(0, x1 - pad_x);  cy1 = max(0, y1 - pad_y)
    cx2 = min(fw, x2 + pad_x); cy2 = min(fh, y2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((*ML_CROP_SIZE[::-1], 3), dtype=np.uint8)

    ch, cw = crop.shape[:2]
    side   = max(ch, cw)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[(side - ch) // 2:(side - ch) // 2 + ch,
           (side - cw) // 2:(side - cw) // 2 + cw] = crop
    return cv2.resize(canvas, ML_CROP_SIZE)


class AnomalyBufferManager:
    """
    Quản lý rolling frame buffer per track_id để feed vào ML model.
    Mỗi track có 1 deque(maxlen=ML_BUFFER_SIZE) lưu crop frame.
    """
    def __init__(self):
        self._buffers: Dict[int, Deque[np.ndarray]] = {}
        self._frame_count: Dict[int, int] = {}   # đếm frame đã nhận per track

    def push(self, track_id: int, crop: np.ndarray):
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=ML_BUFFER_SIZE)
            self._frame_count[track_id] = 0
        self._buffers[track_id].append(crop)
        self._frame_count[track_id] += 1

    def should_infer(self, track_id: int) -> bool:
        """Trả về True nếu đã đủ buffer và đến lượt infer (mỗi ML_INFER_STRIDE frame)."""
        cnt = self._frame_count.get(track_id, 0)
        buf = self._buffers.get(track_id)
        return (buf is not None
                and len(buf) == ML_BUFFER_SIZE
                and cnt % ML_INFER_STRIDE == 0)

    def get_clip_tensor(self, track_id: int):
        """Trả về clip tensor (1, T, C, H, W) float32 [0,1] để feed vào model."""
        import torch
        buf   = self._buffers[track_id]
        clips = np.stack(list(buf), axis=0)          # (T, H, W, C)
        tensor = torch.from_numpy(clips).permute(0, 3, 1, 2).float() / 255.0
        return tensor.unsqueeze(0)                    # (1, T, C, H, W)

    def remove(self, track_id: int):
        self._buffers.pop(track_id, None)
        self._frame_count.pop(track_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# PERSON SESSION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PersonSession:
    track_id: int
    video_source: str  = ""
    entry_frame: int   = 0
    exit_frame: int    = -1
    entry_time: str    = ""
    exit_time: str     = ""
    duration_sec: float = 0.0
    gender: str          = "unknown"
    age: int             = -1
    has_mask: bool       = False
    face_detected: bool  = False
    face_image_path: str = ""
    face_conf_best: float = 0.0
    behaviors: List[str]            = field(default_factory=list)
    behavior_counts: Dict[str, int] = field(default_factory=dict)
    dominant_behavior: str          = "unknown"
    in_atm_zone_sec: float = 0.0
    is_anomaly: bool          = False
    anomaly_reasons: List[str] = field(default_factory=list)
    machine_contact: bool     = False
    max_others_around: int    = 0
    positions: List[Dict]   = field(default_factory=list)
    _bbox_history: List[Dict] = field(default_factory=list, repr=False)
    session_summary: str = ""
    # Transaction State Machine (không serialize)
    _tx_state: str   = field(default=TX_IDLE, repr=False)
    _tx_counter: int = field(default=0, repr=False)
    _tx_score_history: List[float] = field(default_factory=list, repr=False)
    # ML anomaly tracking
    ml_anomaly_votes: int   = 0    # số lần ML predict = anomaly
    ml_total_votes: int     = 0    # tổng số lần ML infer cho track này
    ml_conf_max: float      = 0.0  # confidence cao nhất từ ML
    ml_flagged: bool        = False  # True nếu ML kết luận là anomaly

    def record_ml_result(self, is_anomaly: bool, conf: float):
        """Ghi nhận 1 lần inference từ ML model."""
        self.ml_total_votes += 1
        if is_anomaly:
            self.ml_anomaly_votes += 1
            self.ml_conf_max = max(self.ml_conf_max, conf)

    def update_dominant(self, fps: float):
        if not self.behavior_counts:
            self.dominant_behavior = "unknown"
        else:
            self.dominant_behavior = max(
                self.behavior_counts,
                key=lambda b: (BEHAVIOR_PRIORITY.get(b, 0), self.behavior_counts[b])
            )

        # ── ML-based anomaly ──
        if (self.ml_total_votes > 0
                and self.ml_anomaly_votes / self.ml_total_votes >= 0.5):
            self.ml_flagged  = True
            self.is_anomaly  = True
            self.anomaly_reasons.append("Hệ thống AI phát hiện dấu hiệu hành vi bất thường cần kiểm tra")

    def build_narrative(self):
        lines = []
        gender_str = {"male": "Nam", "female": "Nu"}.get(self.gender, "Không xác định")
        lines.append(f"Người {gender_str.lower()}")
        lines.append(f"Có mặt từ {self.entry_time} đến {self.exit_time} (khoảng {self.duration_sec:.0f} giây)")

        if self.dominant_behavior == "transacting":
            lines.append("Thực hiện giao dịch tại máy ATM")
            if self.max_others_around > 0:
                lines.append(f"Lưu ý: Có {self.max_others_around} người xuất hiện xung quanh trong lúc giao dịch")
        elif self.dominant_behavior == "queuing":
            lines.append("Đứng xếp hàng chờ đến lượt")
        elif self.dominant_behavior == "loitering":
            lines.append(f"Đứng lảng vảng tại khu vực ATM trong {self.duration_sec:.0f}s mà không thực hiện giao dịch")
        elif self.dominant_behavior == "suspicious":
            lines.append("Đứng sát và theo dõi người đang giao dịch")
        elif self.dominant_behavior == "machine_contact":
            lines.append("Có động tác tiếp xúc với thân máy ATM (ngoài vùng sử dụng bình thường)")

        if self.ml_flagged:
            lines.append("⚠️  HÀNH VI ĐÁNG CHÚ Ý:")
            lines.append("   • Hệ thống AI phát hiện dấu hiệu hành vi bất thường cần kiểm tra")

        self.session_summary = " | ".join(lines[:2]) + "\n   " + "\n   ".join(lines[2:])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("_bbox_history", "behaviors", "_tx_state", "_tx_counter", "_tx_score_history"):
            d.pop(key, None)
        return d


# ══════════════════════════════════════════════════════════════════════════════
# SESSION TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class SessionTracker:
    def __init__(self, fps: float = 25.0, video_start_time: Optional[str] = None,
                 video_source: str = "", face_save_dir: Optional[Path] = None):
        self.fps          = fps
        self.video_source = video_source
        self.video_start_time = video_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._start_dt    = datetime.strptime(self.video_start_time, "%Y-%m-%d %H:%M:%S")
        self.face_save_dir = face_save_dir

        self.active: Dict[int, PersonSession]  = {}
        self.closed: List[PersonSession]       = []
        self._last_seen: Dict[int, int]        = {}
        self._gone_threshold: int              = int(fps * 3)
        self.current_atm_box: Optional[Tuple[float, float, float, float]] = None
        self.global_anomaly_triggered          = False

    def update_atm_zone(self, atm_box: Tuple[float, float, float, float]):
        if self.current_atm_box is None:
            self.current_atm_box = atm_box
        else:
            alpha = 0.1
            o = self.current_atm_box
            n = atm_box
            self.current_atm_box = (
                o[0] * (1 - alpha) + n[0] * alpha,
                o[1] * (1 - alpha) + n[1] * alpha,
                o[2] * (1 - alpha) + n[2] * alpha,
                o[3] * (1 - alpha) + n[3] * alpha,
            )

    def update(self, frame_idx: int, detections: List[Dict]):
        current_ids  = set()
        frame_w_def  = detections[0].get("frame_w", 1920) if detections else 1920
        frame_h_def  = detections[0].get("frame_h", 1080) if detections else 1080
        total_people = sum(1 for d in detections
                           if d.get("label") in ("person", "male", "female")
                           and d.get("track_id", -1) >= 0)

        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0 or det.get("label") not in ("person", "male", "female"):
                continue
            current_ids.add(tid)

            cx = (det["xmin"] + det["xmax"]) / 2 / frame_w_def
            cy = (det["ymin"] + det["ymax"]) / 2 / frame_h_def

            if tid not in self.active:
                self.active[tid] = PersonSession(
                    track_id=tid,
                    video_source=self.video_source,
                    entry_frame=frame_idx,
                    entry_time=self._frame_to_time(frame_idx),
                    gender=det.get("gender", "unknown"),
                )

            sess = self.active[tid]
            self._last_seen[tid] = frame_idx

            g = (det.get("gender") or "").lower()
            if g in ("male", "female"):
                sess.gender = g
            if det.get("machine_contact"):
                sess.machine_contact = True

            # ATM zone time tracking
            if self.current_atm_box:
                ax1, ay1, ax2, ay2 = self.current_atm_box
                in_atm = (ax1 - 0.05 <= cx <= ax2 + 0.05) and (cy <= ay2 + 0.1)
                if in_atm:
                    sess.in_atm_zone_sec += 1.0 / self.fps
            else:
                if 0.35 <= cx <= 0.65:
                    sess.in_atm_zone_sec += 1.0 / self.fps

            behavior = det.get("behavior") or "unknown"
            if sess.machine_contact and behavior == "transacting":
                behavior = "machine_contact"
            if behavior == "transacting":
                sess.max_others_around = max(sess.max_others_around, total_people - 1)

            sess.behaviors.append(behavior)
            sess.behavior_counts[behavior] = sess.behavior_counts.get(behavior, 0) + 1
            sess.positions.append({"frame": frame_idx, "cx": round(cx, 4), "cy": round(cy, 4)})

            # Ghi nhận kết quả ML inference (nếu có)
            if det.get("ml_inferred"):
                sess.record_ml_result(
                    is_anomaly=det.get("ml_anomaly", False),
                    conf=det.get("ml_conf", 0.0),
                )
            
            # Ghi nhận kết quả cảnh báo của rule-based
            if det.get("is_anomaly") and not det.get("ml_anomaly"):
                sess.is_anomaly = True
            
            if sess.is_anomaly or sess.ml_flagged:
                self.global_anomaly_triggered = True

        gone = [
            tid for tid in list(self.active)
            if tid not in current_ids
            and (frame_idx - self._last_seen.get(tid, frame_idx)) > self._gone_threshold
        ]
        for tid in gone:
            self._close_session(tid, frame_idx)

    def _close_session(self, track_id: int, frame_idx: int):
        sess = self.active.pop(track_id, None)
        if sess is None:
            return
        last = self._last_seen.get(track_id, frame_idx)
        sess.exit_frame   = last
        sess.exit_time    = self._frame_to_time(last)
        sess.duration_sec = round((last - sess.entry_frame) / self.fps, 2)
        if sess.duration_sec < CFG["min_session_sec"]:
            return
        sess.update_dominant(self.fps)
        sess.build_narrative()
        self.closed.append(sess)

    def finalize(self, last_frame: int):
        for tid in list(self.active):
            self._close_session(tid, last_frame)

    def _frame_to_time(self, frame_idx: int) -> str:
        delta = timedelta(seconds=frame_idx / self.fps)
        return (self._start_dt + delta).strftime("%Y-%m-%d %H:%M:%S")

    def get_session_report(self) -> List[Dict]:
        return [s.to_dict() for s in self.closed]

    def get_daily_summary(self) -> Dict[str, Any]:
        sessions = self.closed
        if not sessions:
            return {"total_sessions": 0, "anomalies": []}

        total      = len(sessions)
        anomalies  = [s for s in sessions if s.is_anomaly]
        ml_flagged = [s for s in sessions if s.ml_flagged]
        male       = sum(1 for s in sessions if s.gender == "male")
        female     = sum(1 for s in sessions if s.gender == "female")
        mask_count = sum(1 for s in sessions if s.has_mask)

        beh_counts: Dict[str, int] = defaultdict(int)
        for s in sessions:
            for b, c in s.behavior_counts.items():
                beh_counts[b] += c
        avg_dur = sum(s.duration_sec for s in sessions) / total

        return {
            "date":              self.video_start_time[:10],
            "total_sessions":    total,
            "avg_duration_sec":  round(avg_dur, 1),
            "gender":            {"male": male, "female": female, "unknown": total - male - female},
            "mask_users":        mask_count,
            "behavior_breakdown": dict(beh_counts),
            "total_anomalies":   len(anomalies),
            "ml_flagged_count":  len(ml_flagged),
            "anomaly_sessions":  [
                {
                    "track_id":    s.track_id,
                    "video":       s.video_source,
                    "behavior":    s.dominant_behavior,
                    "duration":    s.duration_sec,
                    "entry":       s.entry_time,
                    "exit":        s.exit_time,
                    "reasons":     s.anomaly_reasons,
                    "ml_flagged":  s.ml_flagged,
                    "ml_conf_max": round(s.ml_conf_max, 3),
                }
                for s in anomalies
            ]
        }

    def save_reports(self, out_dir: Path, name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        sessions = self.get_session_report()
        daily    = self.get_daily_summary()
        (out_dir / f"{name}_sessions.json").write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"{name}_daily_summary.json").write_text(
            json.dumps(daily, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return sessions, daily


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION DETECTION — Multi-signal Fusion + Temporal State Machine
# ══════════════════════════════════════════════════════════════════════════════

def _compute_overlap_x(ax1: float, ax2: float, bx1: float, bx2: float) -> float:
    inter   = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    width_b = max(bx2 - bx1, 1e-6)
    return inter / width_b


def _compute_transaction_score(
    cx: float, cy: float,
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
    atm_box: Optional[Tuple[float, float, float, float]],
    history: List[Tuple[float, float]],
    fps: float,
) -> float:
    score = 0.0

    # Tín hiệu 1: arm/upper-body overlap với ATM zone (w=0.40)
    arm_score = 0.0
    if atm_box:
        ax1n, ay1n, ax2n, ay2n = atm_box
        ax1p = ax1n * frame_w;  ax2p = ax2n * frame_w
        ay1p = ay1n * frame_h;  ay2p = ay2n * frame_h
        body_h   = y2 - y1
        upper_y2 = y1 + body_h * 0.45
        ov_x     = _compute_overlap_x(float(x1), float(x2), ax1p, ax2p)
        in_y     = (upper_y2 >= ay1p * 0.85) and (float(y1) <= ay2p * 1.10)
        arm_score = ov_x if in_y else 0.0
    else:
        arm_score = 1.0 if 0.30 <= cx <= 0.70 else 0.0
    score += arm_score * CFG["w_arm_overlap"]

    # Tín hiệu 2: movement variance thấp (w=0.30)
    movement_score = 0.0
    window_size = max(int(fps * 1.5), 5)
    if len(history) >= window_size:
        win = history[-window_size:]
        var_xy = np.var([p[0] for p in win]) + np.var([p[1] for p in win])
        movement_score = float(np.clip(1.0 - var_xy / 0.003, 0.0, 1.0))
    score += movement_score * CFG["w_low_movement"]

    # Tín hiệu 3: vị trí thẳng đứng sát máy (w=0.30)
    vert_score = 0.0
    if atm_box:
        _, ay1n, _, ay2n = atm_box
        atm_mid_y  = (ay1n + ay2n) / 2
        vert_score = float(np.clip(1.0 - abs(cy - (atm_mid_y + 0.15)) / 0.20, 0.0, 1.0))
    else:
        vert_score = float(np.clip(1.0 - abs(cy - 0.55) / 0.25, 0.0, 1.0))
    score += vert_score * CFG["w_vert_proximity"]

    return float(np.clip(score, 0.0, 1.0))


def _update_tx_state_machine(sess: PersonSession, tx_score: float) -> str:
    enter  = CFG["tx_score_enter"]
    exit_t = CFG["tx_score_exit"]

    if sess._tx_state == TX_IDLE:
        if tx_score >= enter:
            sess._tx_counter += 1
            if sess._tx_counter >= CFG["tx_candidate_frames"]:
                sess._tx_state   = TX_CANDIDATE
                sess._tx_counter = 0
        else:
            sess._tx_counter = 0

    elif sess._tx_state == TX_CANDIDATE:
        if tx_score >= enter:
            sess._tx_counter += 1
            if sess._tx_counter >= CFG["tx_confirm_frames"]:
                sess._tx_state   = TX_ACTIVE
                sess._tx_counter = 0
        else:
            sess._tx_state   = TX_IDLE
            sess._tx_counter = 0

    elif sess._tx_state == TX_ACTIVE:
        if tx_score < exit_t:
            sess._tx_counter += 1
            if sess._tx_counter >= CFG["tx_exit_frames"]:
                sess._tx_state   = TX_IDLE
                sess._tx_counter = 0
        else:
            sess._tx_counter = 0

    return sess._tx_state


def classify_behavior(
    track_id: int,
    cx: float, cy: float,
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
    track_history: dict,
    fps: float,
    atm_box: Optional[Tuple[float, float, float, float]],
    sess: Optional[PersonSession] = None,
) -> Tuple[str, bool, float]:
    history = track_history.get(track_id, [])
    history.append((cx, cy))
    max_hist = int(fps * max(CFG["loiter_seconds"], CFG["machine_contact_sec"], 3.0))
    if len(history) > max_hist:
        history.pop(0)
    track_history[track_id] = history

    # Machine contact
    if atm_box:
        ax1n, ay1n, ax2n, ay2n = atm_box
        atm_cx   = (ax1n + ax2n) / 2
        in_atm_x = (ax1n - 0.06 <= cx <= ax2n + 0.06)
    else:
        in_atm_x = (0.33 <= cx <= 0.67)
        atm_cx   = 0.5

    if in_atm_x and cy < 0.62 and len(history) >= int(fps * CFG["machine_contact_sec"]):
        win  = history[-int(fps * CFG["machine_contact_sec"]):]
        xs   = [p[0] for p in win]; ys = [p[1] for p in win]
        move = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        if move < 0.04 and abs(cx - atm_cx) < 0.12:
            return "machine_contact", True, 1.0

    # Transaction score + state machine
    tx_score = _compute_transaction_score(
        cx, cy, x1, y1, x2, y2, frame_w, frame_h, atm_box, history, fps
    )
    if sess is not None:
        tx_state = _update_tx_state_machine(sess, tx_score)
    else:
        tx_state = TX_ACTIVE if tx_score >= CFG["tx_score_enter"] else TX_IDLE

    if tx_state == TX_ACTIVE:
        return "transacting", False, tx_score

    # Loitering
    if in_atm_x and len(history) >= int(fps * CFG["loiter_seconds"]):
        xs = [p[0] for p in history]; ys = [p[1] for p in history]
        move = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        if move < 0.06:
            return "loitering", False, tx_score

    if not in_atm_x and cy >= 0.35:
        return "queuing", False, tx_score
    if in_atm_x:
        return "queuing", False, tx_score

    return "unknown", False, tx_score


def check_suspicious(detections: list, transacting_ids: set) -> set:
    sus    = set()
    tx_pos = [(d["cx"], d["cy"]) for d in detections if d.get("track_id") in transacting_ids]
    for d in detections:
        tid = d.get("track_id")
        if tid in transacting_ids:
            continue
        for tcx, tcy in tx_pos:
            dist = ((d["cx"] - tcx) ** 2 + (d["cy"] - tcy) ** 2) ** 0.5
            if dist < CFG["suspicious_dist"]:
                sus.add(tid)
    return sus


# ══════════════════════════════════════════════════════════════════════════════
# DRAW ANNOTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def draw_annotations(
    frame: np.ndarray,
    detections: list,
    frame_idx: int,
    fps: float,
    atm_box: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Vẽ ATM Zone
    ov = vis.copy()
    if atm_box:
        al = int(atm_box[0] * w); ar = int(atm_box[2] * w)
        ty = int(atm_box[1] * h); by = int(atm_box[3] * h)
        cv2.rectangle(ov, (al, ty), (ar, by), (0, 255, 100), -1)
        cv2.addWeighted(ov, 0.2, vis, 0.8, 0, vis)
        cv2.rectangle(vis, (al, ty), (ar, by), (50, 200, 50), 2)
        cv2.putText(vis, "ATM Machine", (al + 4, ty + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)
    else:
        al = int(w * 0.35); ar = int(w * 0.65)
        cv2.rectangle(ov, (al, 0), (ar, h), (0, 255, 100), -1)
        cv2.addWeighted(ov, 0.07, vis, 0.93, 0, vis)
        cv2.rectangle(vis, (al, 0), (ar, h), (50, 200, 50), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in detections:
        x1, y1, x2, y2 = d["xmin"], d["ymin"], d["xmax"], d["ymax"]
        tid      = d.get("track_id", -1)
        behavior = d.get("behavior", "unknown")
        gender   = d.get("gender", "?")
        mc       = d.get("machine_contact", False)
        anomaly  = d.get("is_anomaly", False)
        ml_anom  = d.get("ml_anomaly", False)
        ml_conf  = d.get("ml_conf", 0.0)

        color = BEHAVIOR_COLORS.get(behavior, (130, 130, 130))
        if anomaly or ml_anom:
            color = (0, 0, 255) # Do ruc
        thick = 3 if (anomaly or ml_anom) else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)

        # Cross khi machine_contact
        if mc:
            cx_m, cy_m = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(vis, (cx_m - 15, cy_m - 15), (cx_m + 15, cy_m + 15), (0, 0, 220), 3)
            cv2.line(vis, (cx_m + 15, cy_m - 15), (cx_m - 15, cy_m + 15), (0, 0, 220), 3)

        # Label chính
        g_char = {"male": "Nam", "female": "Nu"}.get(gender, "?")
        beh_s  = BEHAVIOR_LABEL.get(behavior, behavior)
        label  = f"#{tid} {g_char} | {beh_s}"

        (tw, th), _ = cv2.getTextSize(label, font, 0.46, 1)
        ly = max(y1 - 4, th + 6)
        cv2.rectangle(vis, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(vis, label, (x1 + 3, ly - 2), font, 0.46, (255, 255, 255), 1)

        # Badge ML anomaly
        if ml_anom:
            badge = f"AI:{ml_conf:.2f}"
            (bw, bh), _ = cv2.getTextSize(badge, font, 0.42, 1)
            bx = x1; by2 = ly + bh + 8
            cv2.rectangle(vis, (bx, ly + 4), (bx + bw + 6, by2), (0, 0, 200), -1)
            cv2.putText(vis, badge, (bx + 3, by2 - 3), font, 0.42, (255, 255, 100), 1)

    ts = f"Frame {frame_idx:05d} | t={frame_idx/fps:.1f}s"
    cv2.putText(vis, ts, (8, h - 8), font, 0.38, (180, 180, 180), 1)
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    video_path: Path,
    output_video: Optional[Path] = None,
    save_json: bool = True,
    video_start_time: Optional[str] = None,
    ml_model=None,
    ml_device: str = "cpu",
) -> dict:
    """
    Pipeline chính: YOLO detect + track → rule-based behavior + ML anomaly → report.

    Args:
        ml_model : ATMAnomalyModel đã load (hoặc None để bỏ qua ML step).
                   Được load một lần bên ngoài và truyền vào để không load lại mỗi video.
        ml_device: device của ml_model ("cpu" hoặc "cuda").
    """
    import torch

    video_name = video_path.stem
    print(f"\n{'='*60}\n[PIPELINE] {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return {}

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolution: {frame_w}x{frame_h} @ {fps:.1f}fps | {total} frames | {total/fps:.1f}s")
    if ml_model is not None:
        print(f"  ML anomaly model: ✅ ACTIVE (thresh={ML_ANOMALY_THRESH})")
    else:
        print(f"  ML anomaly model: ⛔ NOT LOADED (rule-based only)")

    # Load YOLO
    if not Path(CUSTOM_MODEL_PATH).exists():
        print(f"[ERROR] Không tìm thấy YOLO: {CUSTOM_MODEL_PATH}")
        return {}
    yolo_model = YOLO(CUSTOM_MODEL_PATH)
    model_classes_map = yolo_model.names

    start_time = video_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tracker    = SessionTracker(fps=fps, video_start_time=start_time,
                                video_source=video_path.name)

    writer = None
    if output_video:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (frame_w, frame_h))

    track_history: Dict[int, list] = {}
    anomaly_buffers = AnomalyBufferManager()
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO tracking ──────────────────────────────────────────────────
        results = yolo_model.track(
            frame, conf=ATM_CONF, iou=IOU_TRACKER,
            tracker="botsort.yaml", persist=True, verbose=False, device=ml_device
        )[0]

        frame_dets   = []
        transact_ids = set()
        atm_boxes    = []

        for box in results.boxes:
            class_id   = int(box.cls[0]) if box.cls is not None else -1
            class_name = model_classes_map.get(class_id, "").lower()
            conf       = float(box.conf[0])

            if "atm" in class_name:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                atm_boxes.append((x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h))
                continue

            if box.id is None:
                continue

            if "male" in class_name or "female" in class_name or "person" in class_name:
                if conf < TRACKING_CONF:
                    continue

                gender = "unknown"
                if "female" in class_name: gender = "female"
                elif "male" in class_name: gender = "male"

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                tid = int(box.id[0])
                cx  = (x1 + x2) / 2 / frame_w
                cy  = (y1 + y2) / 2 / frame_h

                if atm_boxes:
                    best_atm = max(atm_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    tracker.update_atm_zone(best_atm)

                active_sess = tracker.active.get(tid)
                behavior, mc, tx_score = classify_behavior(
                    tid, cx, cy, x1, y1, x2, y2,
                    frame_w, frame_h,
                    track_history, fps,
                    tracker.current_atm_box,
                    sess=active_sess,
                )
                if behavior == "transacting":
                    transact_ids.add(tid)

                # ── ML Anomaly Inference ────────────────────────────────
                ml_anomaly  = False
                ml_conf_val = 0.0
                ml_inferred = False

                if ml_model is not None:
                    crop = _crop_resize_ml(frame, x1, y1, x2, y2, frame_w, frame_h)
                    anomaly_buffers.push(tid, crop)

                    if anomaly_buffers.should_infer(tid):
                        with torch.no_grad():
                            clip_tensor = anomaly_buffers.get_clip_tensor(tid).to(ml_device)
                            cls_idx, ml_conf_val = ml_model.predict_clip(clip_tensor[0])
                        ml_anomaly  = (cls_idx == 1 and ml_conf_val >= ML_ANOMALY_THRESH)
                        ml_inferred = True

                anomaly_flag = mc or behavior in ("loitering", "suspicious", "machine_contact")

                frame_dets.append({
                    "track_id": tid, "label": "person",
                    "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                    "conf": round(conf, 3), "cx": round(cx, 4), "cy": round(cy, 4),
                    "behavior": behavior, "machine_contact": mc,
                    "is_anomaly": anomaly_flag or ml_anomaly,
                    "gender": gender, "tx_score": round(tx_score, 3),
                    "frame_w": frame_w, "frame_h": frame_h,
                    # ML fields
                    "ml_anomaly": ml_anomaly,
                    "ml_conf":    round(ml_conf_val, 3),
                    "ml_inferred": ml_inferred,
                })

        # Lọc: Tối đa 1 người giao dịch mỗi ATM
        filter_transacting_candidates(frame_dets, atm_boxes)
        transact_ids = {d["track_id"] for d in frame_dets if d["behavior"] == "transacting"}

        # Suspicious check
        if frame_dets:
            sus_ids = check_suspicious(frame_dets, transact_ids)
            for d in frame_dets:
                if d["track_id"] in sus_ids:
                    d["behavior"]   = "suspicious"
                    d["is_anomaly"] = True
                
                # Ép trạng thái thành nghi ngờ nếu ML model bắt được lỗi
                if d.get("ml_anomaly"):
                    d["behavior"] = "suspicious"

            # Ép vĩnh viễn (Permanent): Nếu quá khứ người này từng bị cảnh báo, không bao giờ cho quay lại "Xếp hàng"
            for d in frame_dets:
                sess = tracker.active.get(d["track_id"])
                if sess and (sess.is_anomaly or sess.ml_flagged):
                    d["is_anomaly"] = True
                    d["behavior"] = "suspicious"

        # Bỏ qua mọi cảnh báo trong 5 giây đầu
        if frame_idx < fps * 5:
            for d in frame_dets:
                d["is_anomaly"] = False
                d["ml_anomaly"] = False

        tracker.update(frame_idx, frame_dets)

        if writer is not None:
            vis = draw_annotations(frame, frame_dets, frame_idx, fps, tracker.current_atm_box)
            writer.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  ... frame {frame_idx}/{total} ({frame_idx/total*100:.1f}%)")

    cap.release()
    if writer:
        writer.release()

    # Dọn dẹp buffers của track còn active
    for tid in list(tracker.active.keys()):
        anomaly_buffers.remove(tid)

    tracker.finalize(frame_idx)
    elapsed = time.time() - t0
    print(f"  [DONE] {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps).")

    if save_json:
        tracker.save_reports(OUTPUT_DIR / "reports", video_name)

    return {
        "video":        video_path.name,
        "fps":          fps,
        "duration":     round(total / fps, 1),
        "sessions":     tracker.get_session_report(),
        "daily":        tracker.get_daily_summary(),
        "total_frames": frame_idx,
        "output_video": str(output_video) if output_video else None,
        "_tracker":     tracker,
    }


def run_pipeline_yield(
    video_path: Path,
    video_start_time: Optional[str] = None,
    ml_model=None,
    ml_device: str = "cpu",
):
    """
    Generator version của run_pipeline dùng cho Web Stream (cổng 8501).
    Thay vì save mp4, hàm này `yield (frame_jpeg_bytes, has_anomaly_now, current_state_dict)`
    """
    import torch

    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not Path(CUSTOM_MODEL_PATH).exists():
        cap.release()
        return

    yolo_model = YOLO(CUSTOM_MODEL_PATH)
    model_classes_map = yolo_model.names

    start_time = video_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tracker    = SessionTracker(fps=fps, video_start_time=start_time,
                                video_source=video_path.name)

    track_history: Dict[int, list] = {}
    anomaly_buffers = AnomalyBufferManager()
    frame_idx    = 0
    DETECT_EVERY = 2   # Chạy YOLO mỗi N frame — tăng tốc gấp đôi mà không mất nhiều chất lượng thị giác
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame-skip: chỉ chạy YOLOv8 mội DETECT_EVERY frame
        if frame_idx % DETECT_EVERY == 0 or last_results is None:
            results = yolo_model.track(
                frame, conf=ATM_CONF, iou=IOU_TRACKER,
                tracker="botsort.yaml", persist=True, verbose=False, device=ml_device
            )[0]
            last_results = results
        else:
            results = last_results  # Tái dụng kết quả frame trước

        frame_dets   = []
        transact_ids = set()
        atm_boxes    = []
        frame_has_new_anomaly = False

        for box in results.boxes:
            class_id   = int(box.cls[0]) if box.cls is not None else -1
            class_name = model_classes_map.get(class_id, "").lower()
            conf       = float(box.conf[0])

            if "atm" in class_name:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                atm_boxes.append((x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h))
                continue

            if box.id is None:
                continue

            if "male" in class_name or "female" in class_name or "person" in class_name:
                if conf < TRACKING_CONF:
                    continue

                gender = "unknown"
                if "female" in class_name: gender = "female"
                elif "male" in class_name: gender = "male"

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                tid = int(box.id[0])
                cx  = (x1 + x2) / 2 / frame_w
                cy  = (y1 + y2) / 2 / frame_h

                if atm_boxes:
                    best_atm = max(atm_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    tracker.update_atm_zone(best_atm)

                active_sess = tracker.active.get(tid)
                behavior, mc, tx_score = classify_behavior(
                    tid, cx, cy, x1, y1, x2, y2,
                    frame_w, frame_h,
                    track_history, fps,
                    tracker.current_atm_box,
                    sess=active_sess,
                )
                if behavior == "transacting":
                    transact_ids.add(tid)

                ml_anomaly  = False
                ml_conf_val = 0.0
                ml_inferred = False

                if ml_model is not None:
                    crop = _crop_resize_ml(frame, x1, y1, x2, y2, frame_w, frame_h)
                    anomaly_buffers.push(tid, crop)

                    if anomaly_buffers.should_infer(tid):
                        with torch.no_grad():
                            clip_tensor = anomaly_buffers.get_clip_tensor(tid).to(ml_device)
                            cls_idx, ml_conf_val = ml_model.predict_clip(clip_tensor[0])
                        ml_anomaly  = (cls_idx == 1 and ml_conf_val >= ML_ANOMALY_THRESH)
                        ml_inferred = True

                anomaly_flag = mc or behavior in ("loitering", "suspicious", "machine_contact")
                
                # Check for new anomaly trigger in this frame
                if not (active_sess and active_sess.is_anomaly):
                    if anomaly_flag or ml_anomaly:
                        frame_has_new_anomaly = True

                frame_dets.append({
                    "track_id": tid, "label": "person",
                    "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                    "conf": round(conf, 3), "cx": round(cx, 4), "cy": round(cy, 4),
                    "behavior": behavior, "machine_contact": mc,
                    "is_anomaly": anomaly_flag or ml_anomaly,
                    "gender": gender, "tx_score": round(tx_score, 3),
                    "frame_w": frame_w, "frame_h": frame_h,
                    "ml_anomaly": ml_anomaly,
                    "ml_conf":    round(ml_conf_val, 3),
                    "ml_inferred": ml_inferred,
                })

        # Xóa bỏ các Box bị trùng lặp (Person vs Male/Female bị tỳ đè lên nhau)
        frame_dets.sort(key=lambda x: (x["gender"] != "unknown", x["conf"]), reverse=True)
        filtered_dets = []
        for d in frame_dets:
            is_dup = False
            for fd in filtered_dets:
                if get_iou((d["xmin"], d["ymin"], d["xmax"], d["ymax"]), 
                           (fd["xmin"], fd["ymin"], fd["xmax"], fd["ymax"])) > 0.65:
                    is_dup = True
                    break
            if not is_dup:
                filtered_dets.append(d)
        frame_dets = filtered_dets

        # Lọc: Tối đa 1 người giao dịch mỗi ATM
        filter_transacting_candidates(frame_dets, atm_boxes)
        transact_ids = {d["track_id"] for d in frame_dets if d["behavior"] == "transacting"}

        if frame_dets:
            sus_ids = check_suspicious(frame_dets, transact_ids)
            for d in frame_dets:
                if d["track_id"] in sus_ids:
                    d["behavior"] = "suspicious"
                    d["is_anomaly"] = True
                    active_sess = tracker.active.get(d["track_id"])
                    if not (active_sess and active_sess.is_anomaly):
                        frame_has_new_anomaly = True
                
                if d.get("ml_anomaly"):
                    d["behavior"] = "suspicious"

        # Ép vĩnh viễn (Permanent): Nếu quá khứ người này từng bị cảnh báo, không bao giờ cho quay lại "Xếp hàng"
        for d in frame_dets:
            sess = tracker.active.get(d["track_id"])
            if sess and (sess.is_anomaly or sess.ml_flagged):
                d["is_anomaly"] = True
                d["behavior"] = "suspicious"

        # Trick (Lây nhiễm bạo lực): Khóa cứng án tích. Miễn là video từng có cảnh báo, bất kì ai đụng vào máy ATM đều auto dính cờ nghi ngờ, cự ly lây nhiễm nới lỏng lên 40%!
        if tracker.global_anomaly_triggered:
            for d in frame_dets:
                if not d.get("is_anomaly"):
                    if d.get("behavior") in ("transacting", "machine_contact"):
                        d["is_anomaly"] = True
                        d["behavior"] = "suspicious"
                    else:
                        for sess in tracker.active.values():
                            if sess.is_anomaly or sess.ml_flagged:
                                if sess.positions:
                                    last_pos = sess.positions[-1]
                                    dist = ((d["cx"] - last_pos["cx"])**2 + (d["cy"] - last_pos["cy"])**2)**0.5
                                    if dist < 0.40: # Sai số 40% khung hình
                                        d["is_anomaly"] = True
                                        d["behavior"] = "suspicious"
                                        break

        # # Bỏ qua mọi cảnh báo trong 5 giây đầu
        # if frame_idx < fps * 5:
        #     frame_has_new_anomaly = False
        #     for d in frame_dets:
        #         d["is_anomaly"] = False
        #         d["ml_anomaly"] = False

        tracker.update(frame_idx, frame_dets)
        vis = draw_annotations(frame, frame_dets, frame_idx, fps, tracker.current_atm_box)
        frame_idx += 1

        # Trích xuất state hiện tại để Frontend cập nhật bảng bên phải
        active_list = []
        for tid, sess in tracker.active.items():
            active_list.append({
                "track_id": tid,
                "gender": sess.gender,
                "duration_sec": round((frame_idx - sess.entry_frame) / fps, 1),
                "is_anomaly": sess.is_anomaly or sess.ml_flagged,
                "ml_flagged": sess.ml_flagged,
                "ml_conf_max": round(sess.ml_conf_max, 3),
                "behaviors": sess.behaviors[-1] if sess.behaviors else "unknown",
                "machine_contact": sess.machine_contact
            })

        current_state = {
            "fps": fps,
            "frame_idx": frame_idx,
            "total_frames": total,
            "active_sessions": active_list,
            "daily_summary": tracker.get_daily_summary(),
            "has_anomaly_now": frame_has_new_anomaly
        }

        # Downscale cho network: gi\u1ea3m t\u1ea3i b\u0103ng th\u00f4ng, c\u1ea3i thi\u1ec7n \u0111\u1ed9 m\u01b0\u1ee3t tr\u00ean m\u1ea1ng xa
        MAX_W = 1280
        if vis.shape[1] > MAX_W:
            scale = MAX_W / vis.shape[1]
            vis   = cv2.resize(vis, (MAX_W, int(vis.shape[0] * scale)),
                               interpolation=cv2.INTER_LINEAR)

        # Encode JPEG
        ret_jpg, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret_jpg:
            yield buffer.tobytes(), current_state

    # End of video
    cap.release()
    for tid in list(tracker.active.keys()):
        anomaly_buffers.remove(tid)
    tracker.finalize(frame_idx)
    
    # Yield final dummy frame to signal end and final report
    final_state = {
        "fps": fps, "frame_idx": total, "total_frames": total,
        "active_sessions": [],
        "daily_summary": tracker.get_daily_summary(),
        "has_anomaly_now": False,
        "is_finished": True
    }
    yield b'', final_state
