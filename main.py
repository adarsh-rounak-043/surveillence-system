import cv2
import time
import requests
import threading
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware # ⚠️ ADD THIS IMPORT

# ==========================================
# 1. CONFIGURATION
# ==========================================
COLAB_API_URL = "https://secrecy-baffle-enlarging.ngrok-free.dev/analyze_batch"

TARGET_FPS = 10.0  
FRAME_DELAY = 1.0 / TARGET_FPS
MIN_MOTION_AREA = 5000 
CONF_THRESHOLD = 0.50

live_frame_buffer = {}
executor = ThreadPoolExecutor(max_workers=3) 

# ==========================================
# 2. LOCAL STREAMING SERVER (PATH B)
# ==========================================
app = FastAPI(title="Local Camera Streamer")

# ⚠️ ADD THIS BLOCK: It tells the browser "Yes, Colab is allowed to view my video!"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any domain to read the stream
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# def generate_video_stream(cam_name):
#     """Yields the newest frame continuously to the web browser."""
#     while True:
#         frame = live_frame_buffer.get(cam_name)
#         if frame is None:
#             time.sleep(0.1)
#             continue
            
#         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
#         time.sleep(FRAME_DELAY) # Prevents CPU overload!

def generate_video_stream(cam_name):
    """Yields the newest frame continuously to the web browser."""
    while True:
        frame = live_frame_buffer.get(cam_name)
        if frame is None:
            time.sleep(0.1)
            continue
            
        # ⚠️ NEW BANDWIDTH SAVERS:
        # 1. Shrink the video to 320x240 (perfect size for the dashboard cards)
        small_frame = cv2.resize(frame, (320, 240))
        
        # 2. Crush the JPEG Quality from 80 down to 40
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # 3. Force the stream to run at a max of 3 FPS instead of 10 FPS
        # (3 FPS is standard for security cameras and saves massive bandwidth!)
        time.sleep(0.33)

@app.get("/stream/{cam_name}")
def video_feed(cam_name: str):
    return StreamingResponse(generate_video_stream(cam_name), media_type="multipart/x-mixed-replace; boundary=frame")

def start_local_server():
    print("📺 Starting Local Streaming Server on port 5050...")
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="error")

# ==========================================
# 3. BACKGROUND AI UPLOAD (PATH A)
# ==========================================
def send_batch_to_colab(batch_files, batch_names):
    try:
        res = requests.post(COLAB_API_URL, files=batch_files, data=batch_names, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                for r in data["results"]:
                    if r["confidence"] >= CONF_THRESHOLD and r["prediction"] not in ["Normal_Videos_event", "info"]:
                        print(f"🚨 [AI THREAT] {r['camera']}: {r['prediction']} ({r['confidence']:.2f})")
    except Exception:
        pass 

# ==========================================
# 4. MAIN SENSOR LOOP
# ==========================================
def run_surveillance():
    print("\nInitializing hardware...")
    camera_sources = {"Built-in": 0, "USB_Left": 1, "USB_Right": 2}
    cameras = {name: cv2.VideoCapture(src) for name, src in camera_sources.items() if cv2.VideoCapture(src).isOpened()}

    if not cameras:
        print("❌ No cameras found.")
        return

    for name in cameras.keys(): live_frame_buffer[name] = None
    motion_detectors = {name: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False) for name in cameras.keys()}
    
    print("📡 Dual-Path Architecture Armed!")
    try:
        while True:
            loop_start = time.time()
            batch_files, batch_names = [], []
            
            for cam_name, cap in cameras.items():
                ret, frame = cap.read()
                if not ret: continue

                live_frame_buffer[cam_name] = frame.copy()

                fg_mask = motion_detectors[cam_name].apply(frame)
                _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if any(cv2.contourArea(c) > MIN_MOTION_AREA for c in contours):
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    batch_files.append(('images', (f"{cam_name}.jpg", buffer.tobytes(), 'image/jpeg')))
                    batch_names.append(('camera_names', cam_name))

            if batch_files:
                executor.submit(send_batch_to_colab, batch_files, batch_names)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, FRAME_DELAY - elapsed))

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for cap in cameras.values(): cap.release()

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_local_server, daemon=True)
    server_thread.start()
    run_surveillance()