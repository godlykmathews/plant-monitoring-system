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
ESP_PORT = "COM3"  # Change to "/dev/ttyUSB0" for Raspberry Pi
BAUD_RATE = 115200

# 1.1 Web Server Settings
SERVER_URL = "http://34.47.248.124:8080/api/log_disease"  # TODO: UPDATE THIS WITH DEPLOYED URL
LOG_COOLDOWN = 5.0  # seconds between logs
last_log_time = 0
sprayed_times = 0

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
    cap = cv2.VideoCapture(1) # '0' is usually the default webcam

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
        if is_disease:
            if ser:
                ser.write(b'1')
            sprayed_times += 1
            history_type = "sprayed"
        else:
            if ser:
                ser.write(b'0')
            history_type = "scan"

        # --- LOG TO GOOGLE CLOUD SERVER ---
        current_time = time.time()
        if current_time - last_log_time > LOG_COOLDOWN:
            try:
                payload = {
                    "disease_name": label,
                    "sprayed_times": sprayed_times,
                    "history_type": history_type
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
            sprayed_times = 0

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
