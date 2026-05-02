import cv2
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. CONFIGURATION
# ==========================================
COLAB_API_URL = "https://secrecy-baffle-enlarging.ngrok-free.dev/analyze_batch"
TARGET_FPS = 10.0  
FRAME_DELAY = 1.0 / TARGET_FPS
CONF_THRESHOLD = 0.50

# ⚠️ NEW: Motion Detection Settings
# How many pixels need to change to trigger a send? 
# (Increase this if bugs or tiny movements are triggering false alarms)
MIN_MOTION_AREA = 5000 

executor = ThreadPoolExecutor(max_workers=3) 

# ==========================================
# 2. INITIALIZE CAMERAS & MOTION SENSORS
# ==========================================
camera_sources = {"Built-in": 0, "USB_Left": 1, "USB_Right": 2}
cameras = {name: cv2.VideoCapture(src) for name, src in camera_sources.items() if cv2.VideoCapture(src).isOpened()}

if not cameras:
    print("❌ No cameras found.")
    exit()

# Create a dedicated "brain" for each camera to learn its specific background
motion_detectors = {
    name: cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False) 
    for name in cameras.keys()
}

# ==========================================
# 3. BACKGROUND UPLOAD FUNCTION
# ==========================================
def send_batch_to_server(batch_files, batch_names, send_time):
    try:
        response = requests.post(COLAB_API_URL, files=batch_files, data=batch_names, timeout=5)
        response_time = response.elapsed.total_seconds()
        
        if response.status_code == 200:
            data = response.json()
            gpu_time = data.get("compute_time_seconds", 0)
            
            if data.get("status") == "success":
                for res in data["results"]:
                    cam, pred, conf = res["camera"], res["prediction"], res["confidence"]
                    
                    if conf >= CONF_THRESHOLD and pred not in ["Normal_Videos_event", "info"]:
                        print(f"🚨 [THREAT] {cam}: {pred} ({conf:.2f})")
                    else:
                        print(f"✅ [CLEAR] {cam}: {pred}")
                        
        else:
            print(f"⚠️ Server Error: {response.status_code}")
    except requests.exceptions.RequestException:
        pass # Drop the frame silently if network chokes

# ==========================================
# 4. MAIN LOOP (Motion-Activated)
# ==========================================
print(f"\n📡 System Armed. Waiting for motion...")
try:
    while True:
        loop_start_time = time.time()
        
        batch_files = []
        batch_names = []
        
        for cam_name, cap in cameras.items():
            ret, frame = cap.read()
            if not ret: continue

            # --- MOTION DETECTION LOGIC ---
            # 1. Apply the subtractor to find moving pixels (creates a black & white mask)
            fg_mask = motion_detectors[cam_name].apply(frame)
            
            # 2. Clean up the noise (remove tiny white specks)
            _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
            
            # 3. Find the outlines of the moving objects
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            for contour in contours:
                # 4. If a moving object is bigger than our minimum area, trigger it!
                if cv2.contourArea(contour) > MIN_MOTION_AREA:
                    motion_detected = True
                    
                    # Optional: Draw a green box around the motion on your local screen
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    break # We only need one big movement to trigger the send

            # Display the camera feed locally (now with green motion boxes!)
            cv2.imshow(cam_name, frame)
            
            # --- ONLY UPLOAD IF MOTION IS DETECTED ---
            if motion_detected:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                batch_files.append(('images', (f"{cam_name}.jpg", buffer.tobytes(), 'image/jpeg')))
                batch_names.append(('camera_names', cam_name))

        # Hand off to the background thread (Will only send the cameras that saw motion!)
        if batch_files:
            # Notice we aren't printing every second anymore, just when motion fires
            print(f"🏃 Motion detected in {[n[1] for n in batch_names]}! Uploading...") 
            executor.submit(send_batch_to_server, batch_files, batch_names, loop_start_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        # FPS Math
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0.0, FRAME_DELAY - elapsed_time)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    executor.shutdown(wait=False)
    for cap in cameras.values(): cap.release()
    cv2.destroyAllWindows()