import cv2
import time
import requests

# ==========================================
# 1. SETUP
# ==========================================
# ⚠️ REPLACE this with the URL printed in your Colab console!
COLAB_API_URL = "https://detection-vertebrae-punctual.ngrok-free.dev/analyze"

# Telegram Bot Setup (Handled locally to ensure instant alerting)
TELEGRAM_TOKEN = "7985477518:AAGu85pNa2DAEbxNTEIrhXR6c05__A5vm2g"
CHAT_ID = "8062403854"

def send_telegram_alert(message, location):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    text = f"🔴 RED ALERT\n📍 {location}\n📄 {message}"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=5)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

# ==========================================
# 2. INITIALIZE CAMERAS
# ==========================================
print("Initializing cameras...")
# Update indices (0, 1, 2) based on how Windows/Mac mounts your USB cameras
camera_sources = {
    "Built-in WebCam": 0,
    # "USB Cam Left": 1,
    # "USB Cam Right": 2
}

cameras = {}
for name, source in camera_sources.items():
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cameras[name] = cap
        print(f"✅ Connected to: {name}")
    else:
        print(f"⚠️ Failed to connect: {name}")

if not cameras:
    print("❌ No cameras found. Exiting.")
    exit()

# ==========================================
# 3. MAIN SURVEILLANCE LOOP
# ==========================================
last_alert_time = {}
conf_threshold = 0.50

try:
    print(f"\n📡 Transmitting video to Colab Server: {COLAB_API_URL}")
    print("Press 'q' in any video window to quit.\n")
    
    while True:
        for cam_name, cap in cameras.items():
            ret, frame = cap.read()
            if not ret:
                continue

            # Display the camera feed locally
            cv2.imshow(cam_name, frame)

            # Compress the frame to JPEG to send it over the internet super fast
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Send the compressed image to Colab Brain
            try:
                response = requests.post(
                    COLAB_API_URL, 
                    files={"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                    data={"camera_name": cam_name},
                    timeout=5 # Prevents hanging if the connection drops
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "success":
                        pred = result['prediction']
                        conf = result['confidence']
                        caption = result['caption']
                        
                        # Local Alert Logic
                        if conf >= conf_threshold and pred not in ["Normal_Videos_event", "info"]:
                            alert_key = f"{cam_name}_{pred}"
                            curr_time = time.time()
                            
                            # 15-second cooldown per threat type per camera
                            if curr_time - last_alert_time.get(alert_key, 0) > 15:
                                last_alert_time[alert_key] = curr_time
                                msg = f"Threat: {pred} ({conf:.2f})\nContext: {caption}"
                                print(f"🚨 TRIGGERED: {msg}")
                                send_telegram_alert(msg, location=cam_name)
                        else:
                            print(f"[{cam_name}] Clear: {pred} ({conf:.2f})")
                else:
                    print(f"⚠️ Server returned error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error connecting to Colab for {cam_name}: {e}")

            # Wait 1ms to allow cv2 to draw the window, and check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
                
        # Brief pause between full camera cycles to prevent overloading the free Ngrok tunnel
        time.sleep(0.5) 

except KeyboardInterrupt:
    print("\nShutting down cameras...")
finally:
    for cap in cameras.values():
        cap.release()
    cv2.destroyAllWindows()