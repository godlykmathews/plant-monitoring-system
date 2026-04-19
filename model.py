import torch
import cv2
import numpy as np
import serial
from PIL import Image
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
import requests
import time

# ======================
# CONFIG
# ======================
model_path = "./plant_model"
ESP_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

SERVER_URL = "http://34.47.248.124:8080/api/log_disease"

SCAN_LOG_COOLDOWN = 300.0
SPRAY_LOG_COOLDOWN = 60.0

CONFIRMATION_FRAMES = 8
LEAF_PIXEL_THRESHOLD = 6000
CONFIDENCE_THRESHOLD = 0.6
FRAME_SKIP = 2

# ======================
# STATE
# ======================
last_log_time = 0
last_logged_history_type = None
last_esp_state = 0
disease_counter = 0
frame_count = 0

prediction_history = []
HISTORY_SIZE = 5

print("Loading model...")

# ======================
# INIT
# ======================
try:
    try:
        ser = serial.Serial(ESP_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to ESP32 on {ESP_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}")
        ser = None

    processor = MobileNetV2ImageProcessor.from_pretrained(model_path)
    model = MobileNetV2ForImageClassification.from_pretrained(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # ======================
        # LEAF DETECTION (HSV)
        # ======================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 50, 50])
        upper_green = np.array([85, 255, 255])

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.bitwise_or(mask_green, mask_yellow)
        leaf_pixels = cv2.countNonZero(mask)

        label = "Scanning..."
        confidence = 0
        is_disease_detected = False

        # ======================
        # DYNAMIC CROP (focus leaf)
        # ======================
        if leaf_pixels > LEAF_PIXEL_THRESHOLD:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)

                cropped_frame = frame[y:y+h, x:x+w]

                # Draw box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # ======================
                # AI INFERENCE
                # ======================
                rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb).resize((224, 224))

                inputs = processor(images=pil_image, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)

                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()

                label = model.config.id2label[predicted_id]

                # ======================
                # CONFIDENCE FILTER
                # ======================
                if confidence < CONFIDENCE_THRESHOLD:
                    label = "Uncertain"
                    is_disease_detected = False
                else:
                    is_disease_detected = "healthy" not in label.lower()

                # ======================
                # SMOOTHING
                # ======================
                prediction_history.append(label)
                if len(prediction_history) > HISTORY_SIZE:
                    prediction_history.pop(0)

                label = max(set(prediction_history), key=prediction_history.count)

            else:
                label = "No Leaf Found"
                disease_counter = 0
        else:
            label = "No Leaf Detected"
            disease_counter = 0

        # ======================
        # STABILITY LOGIC
        # ======================
        if is_disease_detected:
            disease_counter += 1
        else:
            disease_counter = 0

        current_esp_state = 1 if disease_counter >= CONFIRMATION_FRAMES else 0
        history_type = "sprayed" if current_esp_state == 1 else "scan"

        # ======================
        # SERIAL COMMUNICATION
        # ======================
        if current_esp_state != last_esp_state:
            if ser:
                command = b'1\n' if current_esp_state == 1 else b'0\n'
                ser.write(command)
                ser.flush()
                print(f">>> COMMAND SENT: {command.decode().strip()}")
            last_esp_state = current_esp_state

        # ======================
        # SERVER LOGGING
        # ======================
        current_time = time.time()
        cooldown = SPRAY_LOG_COOLDOWN if history_type == "sprayed" else SCAN_LOG_COOLDOWN

        if (history_type == "sprayed" and last_logged_history_type != "sprayed") or \
           (current_time - last_log_time > cooldown):

            try:
                payload = {
                    "disease_name": label,
                    "confidence": float(confidence),
                    "sprayed": current_esp_state == 1
                }
                requests.post(SERVER_URL, json=payload, timeout=1.0)

                last_log_time = current_time
                last_logged_history_type = history_type
            except:
                pass

        # ======================
        # DISPLAY
        # ======================
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("AgriScan Live", frame)
        cv2.imshow("Leaf Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if ser:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"ERROR: {e}")
