import torch
import cv2
import numpy as np
import serial
from PIL import Image
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
import requests
import time

# 1. Setup
model_path = "./plant_model" 
ESP_PORT = "/dev/ttyUSB0" 
BAUD_RATE = 115200

# 1.1 Web Server Settings
SERVER_URL = "http://34.47.248.124:8080/api/log_disease" 
SCAN_LOG_COOLDOWN = 300.0 
SPRAY_LOG_COOLDOWN = 60.0 
last_log_time = 0
last_logged_history_type = None
last_esp_state = 0 # Start at 0 (Off)

# --- STABILITY SETTINGS ---
disease_counter = 0
CONFIRMATION_FRAMES = 10 # Must see disease for 10 frames to trigger pump

print("Loading model for live feed...")

try:
    try:
        ser = serial.Serial(ESP_PORT, BAUD_RATE, timeout=0.1) # Lower timeout for Pi 5
        print(f"Connected to ESP32 on {ESP_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}")
        ser = None

    processor = MobileNetV2ImageProcessor.from_pretrained(model_path)
    model = MobileNetV2ForImageClassification.from_pretrained(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)

        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        predicted_id = outputs.logits.argmax(-1).item()
        label = model.config.id2label[predicted_id]

        # --- STABILIZED TRIGGER LOGIC ---
        is_disease_detected = "fungal" in label.lower() or "bacterial" in label.lower()
        
        if is_disease_detected:
            disease_counter += 1
        else:
            disease_counter = 0 # Reset if we see a healthy leaf

        # Only set current_esp_state to 1 if we are SURE (confirmed frames)
        current_esp_state = 1 if disease_counter >= CONFIRMATION_FRAMES else 0
        history_type = "sprayed" if current_esp_state == 1 else "scan"

        # --- SERIAL COMMUNICATION ---
        if current_esp_state != last_esp_state:
            if ser:
                # Send the byte and flush to ensure it goes out IMMEDIATELY
                command = b'1\n' if current_esp_state == 1 else b'0\n'
                ser.write(command)
                ser.flush() 
                print(f">>> COMMAND SENT: {command.decode().strip()}")
            last_esp_state = current_esp_state

        # --- LOG TO SERVER ---
        current_time = time.time()
        cooldown = SPRAY_LOG_COOLDOWN if history_type == "sprayed" else SCAN_LOG_COOLDOWN
        
        # Only log if history type changed or cooldown passed
        if (history_type == "sprayed" and last_logged_history_type != "sprayed") or (current_time - last_log_time > cooldown):
            try:
                payload = {"disease_name": label, "sprayed": current_esp_state == 1}
                requests.post(SERVER_URL, json=payload, timeout=1.0)
                print(f"Server Logged: {payload}")
                last_log_time = current_time
                last_logged_history_type = history_type
            except:
                pass

        cv2.putText(frame, f"AI: {label} (Conf: {disease_counter})", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('AgriScan Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ser: ser.close()
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"ERROR: {e}")
