from ultralytics import YOLO
import cv2
import math
import pyttsx3
import time
import threading
import os

# Initialize the TTS engine
engine = pyttsx3.init()
engine_running = False  # Flag to indicate if the engine is running
terminate_event = threading.Event()  # Event to signal threads to stop

def set_english_voice(engine):
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'Zira' in voice.name or 'David' in voice.name:
            engine.setProperty('voice', voice.id)
            return True
    return False

if not set_english_voice(engine):
    print("No English voice found, using default voice.")

# Function to speak a message
def speak(message):
    global engine_running
    engine_running = True
    engine.say(message)
    engine.runAndWait()
    engine_running = False

# Function to handle speech in a separate thread
def async_speak(message):
    if not terminate_event.is_set() and not engine_running:
        threading.Thread(target=speak, args=(message,), daemon=True).start()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1640)
cap.set(4, 1480)

# Load model
model_path = "yolo-Weights/yolov8n.pt"
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    cap.release()
    cv2.destroyAllWindows()
    exit()

model = YOLO(model_path)

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

last_speak_time = time.time()
speak_interval = 3  # seconds

try:
    while not terminate_event.is_set():
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        results = model(img, stream=True)

        # Object count dictionary
        object_count = {}

        # Coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Update object count
                if class_name in object_count:
                    object_count[class_name] += 1
                else:
                    object_count[class_name] = 1

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, class_name, org, font, fontScale, color, thickness)

        # Announce detected objects and their counts every 3 seconds
        current_time = time.time()
        if current_time - last_speak_time >= speak_interval:
            if object_count:
                message = "I see: " + ", ".join(f"{count} {obj}" for obj, count in object_count.items())
                async_speak(message)
            last_speak_time = current_time

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            terminate_event.set()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
