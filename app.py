# ==========================================
# LOCAL CLIENT: BATCH SENDER & FPS CONTROL
# ==========================================
import cv2
import time
import requests

COLAB_API_URL = "https://detection-vertebrae-punctual.ngrok-free.dev/analyze_batch"

# --- CONFIGURATION ---
FPS_LIMIT = 20.0  # How many batches to process per second
FRAME_DELAY = 1.0 / FPS_LIMIT
CONF_THRESHOLD = 0.50

# --- INITIALIZE CAMERAS ---
camera_sources = {
    "Built-in": 0, 
    # "USB_Left": 1, 
    # "USB_Right": 2
}

cameras = {}
for name, source in camera_sources.items():
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cameras[name] = cap
        print(f"✅ Connected: {name}")

if not cameras:
    print("❌ No cameras found. Exiting.")
    exit()

last_alert_time = {}

print(f"\n📡 Transmitting batches at {FPS_LIMIT} FPS...")
try:
    while True:
        loop_start_time = time.time()
        
        batch_files = []
        batch_names = []
        
        # 1. Gather a frame from every camera
        for cam_name, cap in cameras.items():
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow(cam_name, frame)
            
            # Compress to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Append to batch arrays
            # 'images' must match the FastAPI parameter name exactly
            batch_files.append(('images', (f"{cam_name}.jpg", buffer.tobytes(), 'image/jpeg')))
            batch_names.append(('camera_names', cam_name))

        # 2. Send the Batch to Colab
        if batch_files:
            try:
                # START THE TIMER
                req_start_time = time.time() 
                
                response = requests.post(
                    COLAB_API_URL, 
                    files=batch_files,
                    data=batch_names,
                    timeout=8 
                )
                
                # STOP THE TIMER
                req_end_time = time.time()
                
                # CALCULATE TOTAL LAG
                total_lag = req_end_time - req_start_time
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
    
                        # Print the lag to your terminal
                        print(f"⏱️ Total Lag: {total_lag:.3f} seconds")
                        
                        # Inside your successful response check:
                        server_compute_time = data.get("compute_time_seconds", 0)
                        network_lag = total_lag - server_compute_time
                        
                        print(f"⏱️ LAG BREAKDOWN | Total: {total_lag:.2f}s | AI Compute: {server_compute_time:.2f}s | Network: {network_lag:.2f}s")
                        
                        # Process all results returned in the batch
                        for res in data["results"]:
                            cam = res["camera"]
                            pred = res["prediction"]
                            conf = res["confidence"]
                            
                            if conf >= CONF_THRESHOLD and pred not in ["Normal_Videos_event", "info"]:
                                print(f"🚨 [ALERT] {cam}: {pred} ({conf:.2f})")
                            else:
                                print(f"✅ {cam}: Clear")
                else:
                    print(f"⚠️ Server Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error: {e}")

        # 3. Check for Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        # 4. FPS Control Logic
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0.0, FRAME_DELAY - elapsed_time)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nShutting down cameras...")
finally:
    for cap in cameras.values():
        cap.release()
    cv2.destroyAllWindows()