import torch
import cv2
import numpy as np
import serial  # Added serial library
from PIL import Image
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
import requests
import time

# 1. Setup paths and Serial
model_path = "./plant_model" 
ESP_PORT = "COM4"  # Change to "/dev/ttyUSB0" for Raspberry Pi
BAUD_RATE = 115200

# 1.1 Web Server Settings
SERVER_URL = "http://127.0.0.1:8080/api/log_disease"  # TODO: UPDATE THIS WITH DEPLOYED URL
SCAN_LOG_COOLDOWN = 300.0  # seconds between scan logs
SPRAY_LOG_COOLDOWN = 60.0  # seconds between spray logs
last_log_time = 0
last_logged_history_type = None
last_esp_state = None  # Tracks last command sent to ESP to avoid duplicate writes

print("Loading model for live feed...")

try:
    # Initialize Serial for ESP32
    try:
        ser = serial.Serial(ESP_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to ESP32 on {ESP_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}. Running without ESP32.")
        ser = None

    # 2. Load the AI tools
    processor = MobileNetV2ImageProcessor.from_pretrained(model_path)
    model = MobileNetV2ForImageClassification.from_pretrained(model_path)
    
    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0) # '0' is usually the default webcam

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print("Live Scan Started. Press 'q' to quit.")

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

        # --- TRIGGER PUMP LOGIC ---
        # If the label mentions fungal, send '1' to ESP32
        is_disease = "fungal" in label.lower() or "bacterial" in label.lower()
        current_esp_state = 1 if is_disease else 0
        history_type = "sprayed" if is_disease else "scan"

        # Send ESP command only when state changes (latest ESP flow from PI.py)
        if current_esp_state != last_esp_state:
            if ser:
                if current_esp_state == 1:
                    ser.write(b'1\n')
                    print("Pump Activated!")
                else:
                    ser.write(b'0\n')
                    print("Pump Deactivated.")
            last_esp_state = current_esp_state

        # --- LOG TO GOOGLE CLOUD SERVER ---
        current_time = time.time()
        cooldown = SPRAY_LOG_COOLDOWN if history_type == "sprayed" else SCAN_LOG_COOLDOWN
        history_changed = history_type != last_logged_history_type

        if history_changed or (current_time - last_log_time > cooldown):
            try:
                payload = {
                    "disease_name": label,
                    "sprayed": history_type == "sprayed",
                }
                # Using a short timeout to prevent video feed lag
                response = requests.post(SERVER_URL, json=payload, timeout=2.0)
                if response.status_code == 201:
                    print(f"Logged to server: {payload}")
                else:
                    print(f"Server returned error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Failed to connect to server: {e}")
            
            # Reset counters and timers
            last_log_time = current_time
            last_logged_history_type = history_type

        # 5. Display Result on Screen
        # Add text to the video frame
        cv2.putText(frame, f"Result: {label}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('AgriScan Live Feed', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if ser:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"AN ERROR OCCURRED: {e}")
