import cv2
import time
import requests
import threading
import uvicorn
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. FLEET CONFIGURATION (⚠️ UPDATE THESE!)
# ==========================================
NODE_NAME = "Command_Center_2" # Name of this specific laptop/device
COLAB_API_URL = "https://secrecy-baffle-enlarging.ngrok-free.dev" # Paste your Colab Registry URL here

# Optimization configs
TARGET_FPS = 5.0  
FRAME_DELAY = 1.0 / TARGET_FPS
MIN_MOTION_AREA = 5000 
CONF_THRESHOLD = 0.50

live_frame_buffer = {}
executor = ThreadPoolExecutor(max_workers=3) 

app = FastAPI(title=f"{NODE_NAME} Streamer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 2. AUTONOMOUS REGISTRATION & HEARTBEAT (THE MAGIC)
# ==========================================
def heartbeat_pulse(active_cams):
    """Sends a pulse to Colab every 15 seconds so the Reaper doesn't delete us."""
    while True:
        time.sleep(15)
        for cam_name in active_cams:
            full_cam_id = f"{NODE_NAME}_{cam_name}"
            try:
                requests.post(f"{COLAB_API_URL}/heartbeat", data={"node_name": full_cam_id}, timeout=3)
            except:
                pass # If connection flickers, fail silently and try again in 15s

def tunnel_manager(active_cams):
    """An Always-On loop that monitors the SSH tunnel and auto-heals if the URL changes."""
    
    # Start the Heartbeat exactly once in the background
    threading.Thread(target=heartbeat_pulse, args=(active_cams,), daemon=True).start()
    
    current_url = None
    
    while True:
        print(f"\n[{NODE_NAME}] Initiating Autonomous SSH Tunnel...")
        
        # ⚠️ ADDED: ServerAliveInterval=30 prevents the server from dropping idle connections
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30", "-R", "80:127.0.0.1:5050", "nokey@localhost.run"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Continuously read the terminal output forever
        for line in iter(process.stdout.readline, ''):
            if not line:
                break # If the line is empty, the SSH process died. Break and restart.
                
            match = re.search(r'(https://[a-zA-Z0-9-]+\.lhr\.life)', line)
            
            if match:
                new_url = match.group(1)
                
                # If localhost.run gives us a BRAND NEW URL, we must tell Colab!
                if new_url != current_url:
                    current_url = new_url
                    print(f"🌍 New Tunnel Secured! Base URL: {current_url}")
                    
                    for cam_name in active_cams:
                        full_cam_id = f"{NODE_NAME}_{cam_name}"
                        stream_url = f"{current_url}/stream/{cam_name}"
                        
                        try:
                            print(f"📡 Re-Routing '{full_cam_id}' to Cloud Hub...")
                            res = requests.post(
                                f"{COLAB_API_URL}/register_node", 
                                data={"node_name": full_cam_id, "stream_url": stream_url},
                                timeout=5
                            )
                            if res.status_code == 200:
                                print(f"✅ {full_cam_id} successfully updated in the Fleet Matrix!")
                        except Exception as e:
                            print(f"❌ Failed to reach Hub. Error: {e}")

        # If the code escapes the 'for' loop, it means the SSH connection dropped.
        print("⚠️ SSH Tunnel disconnected. Auto-healing in 3 seconds...")
        current_url = None
        time.sleep(3)

# ==========================================
# 3. LOCAL VIDEO STREAMING
# ==========================================
def generate_video_stream(cam_name):
    try:
        while True:
            frame = live_frame_buffer.get(cam_name)
            if frame is None:
                time.sleep(0.1)
                continue
                
            small_frame = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.33)
    except GeneratorExit:
        # This triggers immediately when the client/tunnel disconnects.
        # It cleanly exits the loop instead of freezing the thread!
        print(f"🚪 Stream closed for {cam_name}")

@app.get("/stream/{cam_name}")
def video_feed(cam_name: str):
    return StreamingResponse(generate_video_stream(cam_name), media_type="multipart/x-mixed-replace; boundary=frame")

def start_local_server():
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="error")

# ==========================================
# 4. BACKGROUND AI UPLOAD
# ==========================================
def send_batch_to_colab(batch_files, batch_names):
    try:
        res = requests.post(f"{COLAB_API_URL}/analyze_batch", files=batch_files, data=batch_names, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                for r in data["results"]:
                    if r["confidence"] >= CONF_THRESHOLD and r["prediction"] not in ["Normal_Videos_event", "info"]:
                        print(f"🚨 [THREAT] {r['camera']}: {r['prediction']} ({r['confidence']:.2f})")
    except Exception:
        pass 

# ==========================================
# 5. MAIN HARDWARE SENSOR LOOP
# ==========================================
def run_surveillance():
    print("\nInitializing hardware...")
    camera_sources = {"Built-in": 0, "USB_Left": 1, "USB_Right": 2}
    cameras = {name: cv2.VideoCapture(src) for name, src in camera_sources.items() if cv2.VideoCapture(src).isOpened()}

    if not cameras:
        print("❌ No cameras found.")
        return

    active_cams = list(cameras.keys())
    for name in active_cams: live_frame_buffer[name] = None
    motion_detectors = {name: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False) for name in active_cams}
    
    # ⏱️ NEW: Track the last time a frame was sent for each camera
    last_frame_sent = {name: time.time() for name in active_cams}
    FORCE_SEND_INTERVAL = 10.0  # Force send a frame to Colab every 10 seconds

    # START AUTONOMOUS REGISTRATION
    tunnel_thread = threading.Thread(target=tunnel_manager, args=(active_cams,), daemon=True)
    tunnel_thread.start()

    print(f"📡 {NODE_NAME} Hardware Armed!")
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
                
                # Check motion OR if the forced time interval has passed
                motion_detected = any(cv2.contourArea(c) > MIN_MOTION_AREA for c in contours)
                time_since_last_send = time.time() - last_frame_sent[cam_name]

                if motion_detected or time_since_last_send >= FORCE_SEND_INTERVAL:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    full_cam_id = f"{NODE_NAME}_{cam_name}"
                    
                    batch_files.append(('images', (f"{cam_name}.jpg", buffer.tobytes(), 'image/jpeg')))
                    batch_names.append(('camera_names', full_cam_id))
                    
                    # Reset the timer for this camera
                    last_frame_sent[cam_name] = time.time()

            if batch_files:
                executor.submit(send_batch_to_colab, batch_files, batch_names)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, FRAME_DELAY - elapsed))

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for cap in cameras.values(): cap.release()

if __name__ == "__main__":
    hardware_thread = threading.Thread(target=run_surveillance, daemon=True)
    hardware_thread.start()
    
    start_local_server()