"""
app.py
======
Main endpoint for the ATM Surveillance system.
Handles single video / batch processing and narrative report generation.

Load ML anomaly model một lần duy nhất, truyền vào tất cả các lần gọi run_pipeline.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from atm_core import run_pipeline, load_anomaly_model, BEHAVIOR_VI, OUTPUT_DIR

# ── Report Generation Logic ──────────────────────────────────────────────────

SEP  = "═" * 60
SEP2 = "─" * 60

def _gender_vi(g: str) -> str:
    return {"male": "Nam", "female": "Nữ"}.get(g, "Không xác định")

def _mask_vi(has_mask: bool, face_detected: bool) -> str:
    return "Đeo khẩu trang" if has_mask else "Không đeo khẩu trang"

def _behavior_summary(session: dict) -> str:
    dom    = session.get("dominant_behavior", "unknown")
    dur    = session.get("duration_sec", 0)
    others = session.get("max_others_around", 0)

    if dom == "transacting":
        extra = f" (Có {others} người xung quanh)" if others > 0 else " (Đứng một mình)"
        return f"✅ Thực hiện giao dịch bình thường{extra}"
    elif dom == "queuing":
        return f"🔵 Đứng xếp hàng chờ ({dur:.0f}s)"
    elif dom == "loitering":
        return f"🟠 Đứng lảng vảng tại khu vực máy ATM ({dur:.0f}s) không giao dịch"
    elif dom == "suspicious":
        return "🔴 Đứng sát / theo dõi người đang giao dịch"
    elif dom == "machine_contact":
        return "🔴 Có động tác tiếp xúc với thân máy ATM"
    return f"⚫ Hành vi không xác định ({dur:.0f}s)"

def format_session(idx: int, session: dict) -> str:
    lines      = []
    is_anomaly = session.get("is_anomaly", False)
    ml_flagged = session.get("ml_flagged", False)

    flag = ""
    if ml_flagged or is_anomaly:
        flag = "  ⚠️ CẢNH BÁO BẤT THƯỜNG"

    # Lấy giờ, phút, giây
    entry = session.get("entry_time", "")[-8:]
    exit_ = session.get("exit_time", "")[-8:]
    dur   = session.get("duration_sec", 0)
    vid   = session.get("video_source", "")
    lines.append(f"\n{SEP2}")
    lines.append(f"📋 PHIÊN #{idx}  |  {entry} → {exit_}  ({dur:.0f}s){flag}")
    if vid:
        lines.append(f"   📹 Video: {vid}")

    gender = _gender_vi(session.get("gender", "unknown"))
    lines.append(f"   Người {gender.lower()}")

    face_path = session.get("face_image_path", "")
    if face_path and Path(face_path).exists():
        lines.append(f"   Ảnh capture: {face_path}")

    lines.append(f"   {_behavior_summary(session)}")

    reasons = session.get("anomaly_reasons", [])
    if reasons:
        lines.append("   ⚠️  Chi tiết bất thường:")
        # Bỏ đi log số lần dự đoán / confidence. Hiển thị text sạch.
        for r in reasons:
            lines.append(f"      • {r}")

    return "\n".join(lines)

def format_daily_header(daily: dict, atm_name: str, date: str) -> str:
    lines = [
        "", SEP,
        f"  BÁO CÁO NGÀY {date}  |  {atm_name}",
        f"  Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        SEP,
    ]
    return "\n".join(lines)

def format_daily_summary(daily: dict) -> str:
    total       = daily.get("total_sessions", 0)
    avg_dur     = daily.get("avg_duration_sec", 0)
    anom        = daily.get("total_anomalies", 0)
    g           = daily.get("gender", {})

    lines = [
        "", SEP, "  📊 TỔNG KẾT TRONG NGÀY", SEP,
        f"  Tổng lượt khách vào khu vực     : {total} người",
        f"  Thời gian hiện diện TB          : {avg_dur:.0f} giây",
        "",
        f"  Giới tính (ước lượng)           :",
        f"    • Nam        : {g.get('male', 0)} người",
        f"    • Nữ         : {g.get('female', 0)} người",
        f"    • Không rõ   : {g.get('unknown', 0)} người",
        "",
        f"  ⚠️  Tổng số trường hợp bất thường: {anom} trường hợp",
    ]

    anomaly_sessions = daily.get("anomaly_sessions", [])
    if anomaly_sessions:
        lines.append("\n  Danh sách cảnh báo:")
        for i, a in enumerate(anomaly_sessions, 1):
            dom    = BEHAVIOR_VI.get(a.get("behavior", ""), a.get("behavior", ""))
            dur    = a.get("duration", 0)
            entry  = a.get("entry", "")[-8:]
            exit_  = a.get("exit", "")[-8:]
            lines.append(f"    {i}. {entry} → {exit_} | Track #{a.get('track_id')} | {dom} | {dur:.0f}s")
            for r in a.get("reasons", []):
                lines.append(f"       → {r}")
    lines.append(SEP)
    return "\n".join(lines)

def generate_report(
    sessions: List[Dict], daily: Dict, out_dir: Path,
    name: str, atm_name: str = "ATM XYZ", date: str = "",
) -> Path:
    if not date:
        date = daily.get("date", datetime.now().strftime("%Y-%m-%d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path  = out_dir / f"report_{name}.txt"
    json_path = out_dir / f"report_{name}.json"

    parts = [format_daily_header(daily, atm_name, date)]
    total = daily.get("total_sessions", 0)
    anom  = daily.get("total_anomalies", 0)
    parts.append(
        f"\n  Có {total} lượt người đến ATM trong ngày hôm nay.\n"
        f"  Phát hiện {anom} trường hợp hành vi bất thường.\n"
    )
    relevant_sessions = []
    for sess in sessions:
        dom = sess.get("dominant_behavior", "")
        # Chỉ lọt vào report những người đứng gần cây ATM (giao dịch, chờ, lảng vảng, chạm máy) 
        # Hoặc bị gắn cờ bất thường AI
        if dom in ("transacting", "queuing", "machine_contact", "loitering") or sess.get("is_anomaly") or sess.get("ml_flagged"):
            relevant_sessions.append(sess)
            
    for i, sess in enumerate(relevant_sessions, 1):
        parts.append(format_session(i, sess))
    parts.append(format_daily_summary(daily))

    txt_path.write_text("\n".join(parts), encoding="utf-8")
    full_json = {
        "report_name": name, "atm": atm_name, "date": date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "daily_summary": daily, "sessions": sessions,
    }
    json_path.write_text(json.dumps(full_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  📝 Báo cáo text  → {txt_path}")
    print(f"  📝 Báo cáo JSON  → {json_path}")
    return txt_path


# ── Execution Logic ───────────────────────────────────────────────────────────

def process_one(
    video_path: Path, out_dir: Path,
    save_video: bool = True,
    video_start_time: str = None,
    ml_model=None, ml_device: str = "cpu",
) -> dict:
    out_video = out_dir / f"{video_path.stem}_annotated.mp4" if save_video else None
    return run_pipeline(
        video_path,
        output_video=out_video,
        save_json=True,
        video_start_time=video_start_time,
        ml_model=ml_model,
        ml_device=ml_device,
    )


def run_all(
    dataset_dir: Path, out_dir: Path,
    save_video: bool = True,
    atm_name: str = "ATM Demo",
    date: str = "",
    ml_model=None, ml_device: str = "cpu",
):
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    videos = sorted(list(dataset_dir.glob("*.mp4")) + list(dataset_dir.glob("*.avi")))
    if not videos:
        print(f"[ERROR] Không tìm thấy video trong {dataset_dir}/")
        return

    print(f"\n{SEP}\n  XỬ LÝ {len(videos)} VIDEO  |  {atm_name}  |  {date}\n{SEP}")
    cursor = datetime.strptime(f"{date} 08:00:00", "%Y-%m-%d %H:%M:%S")
    import cv2
    all_sessions = []

    for vid in videos:
        cap     = cv2.VideoCapture(str(vid))
        fps_v   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        dur_sec    = frames / fps_v if fps_v > 0 else 0
        start_time = cursor.strftime("%Y-%m-%d %H:%M:%S")
        print(f"   {start_time}  →  {vid.name[:50]}  ({dur_sec:.0f}s)")
        result = process_one(vid, out_dir, save_video, start_time, ml_model, ml_device)
        if result:
            all_sessions.extend(result.get("sessions", []))
        cursor += timedelta(seconds=dur_sec + 600)

    if not all_sessions:
        return

    from collections import defaultdict
    total     = len(all_sessions)
    anomalies = [s for s in all_sessions if s.get("is_anomaly")]
    ml_fl     = [s for s in all_sessions if s.get("ml_flagged")]
    male      = sum(1 for s in all_sessions if s.get("gender") == "male")
    female    = sum(1 for s in all_sessions if s.get("gender") == "female")
    mask_n    = sum(1 for s in all_sessions if s.get("has_mask"))
    avg_dur   = sum(s.get("duration_sec", 0) for s in all_sessions) / total
    beh: Dict = defaultdict(int)
    for s in all_sessions:
        for b, c in (s.get("behavior_counts") or {}).items():
            beh[b] += c

    daily = {
        "date":              date,
        "total_sessions":    total,
        "avg_duration_sec":  round(avg_dur, 1),
        "gender":            {"male": male, "female": female, "unknown": total - male - female},
        "mask_users":        mask_n,
        "behavior_breakdown": dict(beh),
        "total_anomalies":   len(anomalies),
        "ml_flagged_count":  len(ml_fl),
        "anomaly_sessions":  [
            {
                "track_id":   s["track_id"],
                "video":      s.get("video_source", ""),
                "behavior":   s.get("dominant_behavior"),
                "duration":   s.get("duration_sec"),
                "entry":      s.get("entry_time"),
                "exit":       s.get("exit_time"),
                "reasons":    s.get("anomaly_reasons", []),
                "has_mask":   s.get("has_mask"),
                "ml_flagged": s.get("ml_flagged", False),
                "ml_conf_max": s.get("ml_conf_max", 0.0),
            }
            for s in anomalies
        ]
    }

    report_path = generate_report(
        all_sessions, daily,
        OUTPUT_DIR / "reports",
        f"day_{date.replace('-', '')}",
        atm_name, date,
    )
    print(
        f"\n{SEP}\n✅ HOÀN TẤT!\n"
        f"   Video annotated  → {out_dir}/\n"
        f"   Tổng sessions    : {total}\n"
        f"   Cảnh báo         : {len(anomalies)}\n"
        f"   ML flagged       : {len(ml_fl)}\n"
        f"   Báo cáo text     → {report_path}\n{SEP}"
    )


def main():
    parser = argparse.ArgumentParser(description="ATM Surveillance App")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--video",    type=str, help="Xử lý 1 video cụ thể")
    g.add_argument("--all",      action="store_true", help="Xử lý tất cả video trong Dataset/")
    parser.add_argument("--out_dir",  type=str, default=None)
    parser.add_argument("--dataset",  type=str, default="Dataset")
    parser.add_argument("--atm",      type=str, default="ATM – Demo")
    parser.add_argument("--date",     type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--no_video", action="store_true", help="Không lưu .mp4")
    parser.add_argument("--no_ml",    action="store_true", help="Tắt ML anomaly model")
    parser.add_argument("--device",   type=str, default="auto",
                        help="Device cho ML model: auto / cpu / cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ML model một lần duy nhất cho tất cả video
    ml_model  = None
    ml_device = "cpu"
    if not args.no_ml:
        import torch
        if args.device == "auto":
            ml_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            ml_device = args.device
        ml_model, ml_device = load_anomaly_model(device=ml_device)

    if args.video:
        vpath  = Path(args.video)
        result = process_one(vpath, out_dir, not args.no_video,
                              ml_model=ml_model, ml_device=ml_device)
        if result:
            print(f"\n📊 {result.get('daily', {}).get('total_sessions', 0)} session")
    else:
        run_all(
            Path(args.dataset), out_dir,
            not args.no_video, args.atm, args.date,
            ml_model=ml_model, ml_device=ml_device,
        )


if __name__ == "__main__":
    main()
