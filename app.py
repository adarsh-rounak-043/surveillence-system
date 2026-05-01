import cv2
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION
# ==========================================
COLAB_API_URL = "https://detection-vertebrae-punctual.ngrok-free.dev/analyze_batch"
TARGET_FPS = 5.0  
FRAME_DELAY = 1.0 / TARGET_FPS  # 0.1 seconds per frame
CONF_THRESHOLD = 0.50

# Set up a pool of background workers to handle network requests
# We limit to 3 workers so we don't accidentally DDoS our own Colab server!
executor = ThreadPoolExecutor(max_workers=7) 

# ==========================================
# INITIALIZE CAMERAS
# ==========================================
camera_sources = {
    "Built-in": 0, 
    # "USB_Left": 1, 
    # "USB_Right": 2
}
cameras = {name: cv2.VideoCapture(src) for name, src in camera_sources.items() if cv2.VideoCapture(src).isOpened()}

if not cameras:
    print("❌ No cameras found.")
    exit()

# ==========================================
# BACKGROUND UPLOAD FUNCTION
# ==========================================
def send_batch_to_server(batch_files, batch_names, send_time):
    """This function runs in the background so the camera doesn't freeze!"""
    try:
        response = requests.post(COLAB_API_URL, files=batch_files, data=batch_names, timeout=5)
        
        # ⏱️ Get the exact HTTP response time
        response_time = response.elapsed.total_seconds()
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract the GPU compute time we set up earlier (if you still have it in your Colab code)
            gpu_time = data.get("compute_time_seconds", 0)
            network_time = response_time - gpu_time
            
            if data.get("status") == "success":
                for res in data["results"]:
                    cam = res["camera"]
                    pred = res["prediction"]
                    conf = res["confidence"]
                    
                    if conf >= CONF_THRESHOLD and pred not in ["Normal_Videos_event", "info"]:
                        print(f"🚨 [SENT @ {send_time:.1f}s] {cam}: {pred} ({conf:.2f})")
                    
                # Optional: Print the health of the thread
                print(f"📡 [Thread Health] Total Response: {response_time:.2f}s | Server AI: {gpu_time:.2f}s | Network: {network_time:.2f}s")
            
        else:
            print(f"⚠️ Server Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        # Silently drop frames if network is too slow, to prevent crashing
        print(f"⚠️ Network timeout/error in background thread: {e}")

# ==========================================
# MAIN CAMERA LOOP (Strict FPS)
# ==========================================
print(f"\n📡 Transmitting batches at {TARGET_FPS} FPS using Background Threads...")
try:
    while True:
        loop_start_time = time.time()
        
        batch_files = []
        batch_names = []
        
        # 1. Grab frames instantly
        for cam_name, cap in cameras.items():
            ret, frame = cap.read()
            if not ret: continue

            cv2.imshow(cam_name, frame)
            
            # Compress heavily (Quality 50) since 10 FPS is a lot of data!
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            batch_files.append(('images', (f"{cam_name}.jpg", buffer.tobytes(), 'image/jpeg')))
            batch_names.append(('camera_names', cam_name))

        # 2. Hand off the heavy lifting to a background thread
        if batch_files:
            executor.submit(send_batch_to_server, batch_files, batch_names, loop_start_time)

        # 3. Quit Check
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        # 4. Strict FPS Math
        # Calculate exactly how long the capturing took, and sleep for the remainder
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0.0, FRAME_DELAY - elapsed_time)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    executor.shutdown(wait=False) # Kill background threads
    for cap in cameras.values(): cap.release()
    cv2.destroyAllWindows()