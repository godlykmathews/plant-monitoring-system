import torch
import cv2
import numpy as np
# Replace 'import serial' with requests for WiFi control
import requests
from PIL import Image
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
import time

# 1. Setup paths and WiFi Settings
model_path = "./plant_model" 
# --- WIFI CONFIG ---
ESP32_IP = "192.168.10.118"  # <--- UPDATE THIS with the IP from ESP32 Serial Monitor
ESP32_URL = f"http://{ESP32_IP}/control"

# 1.1 Web Server Settings
SERVER_URL = "http://34.47.248.124:8080/api/log_disease" 
LOG_COOLDOWN = 5.0  # seconds between logs
last_log_time = 0
sprayed_times = 0

print("Loading model for live feed...")

try:
    # 2. Load the AI tools
    processor = MobileNetV2ImageProcessor.from_pretrained(model_path)
    model = MobileNetV2ForImageClassification.from_pretrained(model_path)
    
    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0) # '0' is usually the default webcam

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print(f"Live Scan Started. Controlling ESP32 at {ESP32_IP}")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV BGR format to PIL RGB for the AI
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)

        # 4. AI Prediction
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        predicted_id = outputs.logits.argmax(-1).item()
        label = model.config.id2label[predicted_id]

        # --- TRIGGER PUMP LOGIC (WiFi) ---
        is_disease = "fungal" in label.lower() or "bacterial" in label.lower()
        
        try:
            if is_disease:
                # Send '1' to ESP32 over WiFi
                requests.get(ESP32_URL, params={'pump': '1'}, timeout=0.5)
                sprayed_times += 1
                history_type = "sprayed"
            else:
                # Send '0' to ESP32 over WiFi
                requests.get(ESP32_URL, params={'pump': '0'}, timeout=0.5)
                history_type = "scan"
        except Exception as e:
            # Silent fail if ESP32 is offline to keep video feed smooth
            pass

        # --- LOG TO GOOGLE CLOUD SERVER ---
        current_time = time.time()
        if current_time - last_log_time > LOG_COOLDOWN:
            try:
                payload = {
                    "disease_name": label,
                    "sprayed_times": sprayed_times,
                    "history_type": history_type
                }
                response = requests.post(SERVER_URL, json=payload, timeout=2.0)
                if response.status_code == 201:
                    print(f"Logged to cloud server: {payload}")
            except Exception as e:
                print(f"Failed to connect to cloud server: {e}")
            
            last_log_time = current_time
            sprayed_times = 0

        # 5. Display Result on Screen
        cv2.putText(frame, f"Result: {label}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('AgriScan Live Feed', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"AN ERROR OCCURRED: {e}")
