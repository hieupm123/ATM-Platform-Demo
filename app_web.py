"""
app_web.py
==========
Flask Web Server for ATM Surveillance Demo (Port 8501)
Streams MJPEG video and JSON state to a React-like HTML frontend.
"""

import os
import json
import time
import zipfile
import threading
import queue
from pathlib import Path
from flask import Flask, request, Response, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Import pipeline & model loader
from atm_core import run_pipeline_yield, load_anomaly_model, BEHAVIOR_VI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/atm_uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global State for Stream
stream_state = {
    "is_playing": False,
    "data": {},
    "file_list": [],
    "current_file_idx": 0,
    "is_folder_mode": False,
    "has_paused_this_video": False,
}

# Buffer hàng đợi frames để chống giật lag (Jitter Buffer)
# Lưu tối đa 150 frames (~6 giây video) trong RAM
frame_queue = queue.Queue(maxsize=150)

# Load ML Model once at startup
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ml_model, ml_device = load_anomaly_model(device=device)
except Exception as e:
    print(f"Error loading model: {e}")
    ml_model, ml_device = None, "cpu"


def process_video_queue(run_id):
    """Background thread function that runs the pipeline on the uploaded files."""
    global stream_state
    
    while stream_state["is_playing"] and stream_state.get("run_id") == run_id and stream_state["current_file_idx"] < len(stream_state["file_list"]):
        video_path = stream_state["file_list"][stream_state["current_file_idx"]]
        stream_state["has_paused_this_video"] = False  # Reset for each video
        print(f"Starting video: {video_path}")
        
        # Generator yielding (frame_bytes, dict_state)
        pipeline = run_pipeline_yield(
            Path(video_path),
            ml_model=ml_model,
            ml_device=ml_device
        )
        
        for frame_bytes, state_data in pipeline:
            # Kiem tra neu user up video moi, abort luong nay luon
            if stream_state.get("run_id") != run_id:
                print(f"Aborting old process_video_queue thread for {video_path}")
                return
                
            # Đẩy vào queue (Sẽ bị Block/đứng chờ ở đây nếu Queue đã đầy 150 frames)
            frame_queue.put((frame_bytes, state_data))
            
        if stream_state.get("run_id") == run_id:
            stream_state["current_file_idx"] += 1
            
    if stream_state.get("run_id") == run_id and stream_state["current_file_idx"] >= len(stream_state["file_list"]):
        stream_state["is_playing"] = False
        stream_state["data"]["is_finished"] = True
        print("Finished processing all videos in queue.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop_stream():
    global stream_state
    # Lập tức vô hiệu luồng gen_frames cũ bằng cách đổi run_id
    stream_state["run_id"] = None
    stream_state["is_playing"] = False
    stream_state["is_buffering"] = False
    with frame_queue.mutex:
        frame_queue.queue.clear()
    return jsonify({"success": True})

@app.route('/upload', methods=['POST'])
def upload_file():
    global stream_state
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Xóa sạch queue cũ
    with frame_queue.mutex:
        frame_queue.queue.clear()
        
    stream_state["file_list"] = []
    stream_state["current_file_idx"] = 0
    stream_state["data"] = {}
    stream_state["is_playing"] = False
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    if filename.endswith('.zip'):
        # Extract ZIP and find videos
        extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4])
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        videos = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi')):
                    videos.append(os.path.join(root, f))
        
        videos.sort()
        stream_state["file_list"] = videos
        stream_state["is_folder_mode"] = True
    else:
        stream_state["file_list"] = [filepath]
        stream_state["is_folder_mode"] = False
        
    if not stream_state["file_list"]:
        return jsonify({"error": "No valid video files found."}), 400
        
    # Start playback thread
    current_run_id = time.time()
    stream_state["run_id"] = current_run_id
    stream_state["is_buffering"] = True
    stream_state["is_playing"] = True
    threading.Thread(target=process_video_queue, args=(current_run_id,), daemon=True).start()
    
    return jsonify({"success": True, "files": len(stream_state["file_list"]), "is_folder": stream_state["is_folder_mode"], "run_id": current_run_id})

@app.route('/resume', methods=['POST'])
def resume_playback():
    global stream_state
    if not stream_state["is_playing"]:
        stream_state["is_playing"] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Đang phát hoặc đã kết thúc."})

@app.route('/status')
def get_status():
    global stream_state
    v_name = ""
    if stream_state["file_list"] and stream_state["current_file_idx"] < len(stream_state["file_list"]):
        v_name = os.path.basename(stream_state["file_list"][stream_state["current_file_idx"]])
        
    payload = {
        "is_playing": stream_state["is_playing"],
        "video_name": v_name,
        "mode": "Folder" if stream_state["is_folder_mode"] else "Single",
        "progress": f"{stream_state['current_file_idx'] + 1}/{len(stream_state['file_list'])}",
        "data": stream_state["data"]
    }
    return jsonify(payload)

def gen_frames():
    global stream_state
    last_run_id = None
    
    while True:
        current_run_id = stream_state.get("run_id")
        
        # Nếu có run mới, cập nhật bản thân và tiếp tục bình thường (KHÔNG bao giờ break)
        if current_run_id != last_run_id:
            last_run_id = current_run_id
        
        # Nếu chưa play thì đợi (kể cả lúc pause hoặc đang upload)
        if not stream_state.get("is_playing", False) or current_run_id is None:
            time.sleep(0.1)
            continue
            
        # Hệ thống Buffer: Đợi cho Queue có ít nhất 50 frames trước khi bắt đầu phát
        if stream_state.get("is_buffering", False):
            if frame_queue.qsize() >= 50 or stream_state["current_file_idx"] >= len(stream_state["file_list"]):
                stream_state["is_buffering"] = False
            else:
                time.sleep(0.05)
                continue
        else:
            if frame_queue.qsize() < 10 and stream_state["current_file_idx"] < len(stream_state["file_list"]):
                stream_state["is_buffering"] = True
                continue
            
        try:
            frame_bytes, state_data = frame_queue.get(timeout=2.0)
            
            # Bỏ qua frame cũ nếu run_id đã thay đổi giữa chừng
            if stream_state.get("run_id") != last_run_id:
                continue
            
            stream_state["data"] = state_data
            
            # Auto-pause logic (Không pause trong 5 giây đầu)
            if state_data.get("has_anomaly_now") and not stream_state["is_folder_mode"]:
                fps_val = state_data.get("fps", 25)
                frame_idx = state_data.get("frame_idx", 0)
                if not stream_state["has_paused_this_video"] and frame_idx >= fps_val * 5:
                    stream_state["is_playing"] = False
                    stream_state["has_paused_this_video"] = True
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Adaptive sleep: nếu Queue đang cạn, bẳt giảm sleep để đẩy nhanh hơn
            fps      = state_data.get("fps", 25)
            q_size   = frame_queue.qsize()
            if q_size > 30:
                time.sleep(1.0 / fps)          # Queue dồi dào, ngủ đúng nhịp
            elif q_size > 10:
                time.sleep(0.5 / fps)          # Queue trung bình, ngủ nửa
            else:
                pass                           # Queue gần cạn, phát hết tốc
            
        except queue.Empty:
            continue

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs("templates", exist_ok=True)
    print("🚀 Starting Web Demo on port 8501...")
    # Using threaded=True is important for MJPEG streaming
    app.run(host='0.0.0.0', port=8501, threaded=True)
