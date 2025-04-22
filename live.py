import cv2
import face_recognition
import pickle
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Load encodings and model
ENCODING_FILE = 'student_encodings.pkl'
YOLO_MODEL = 'face_yolov8m.pt'
MATCH_THRESHOLD = 0.6

with open(ENCODING_FILE, 'rb') as f:
    student_encodings = pickle.load(f)

known_rolls = list(student_encodings.keys())
known_embeddings = list(student_encodings.values())

model = YOLO(YOLO_MODEL)

# Initialize
cap = cv2.VideoCapture(0)
present = set()

print("🔴 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]

        try:
            face_crop = cv2.resize(face_crop, (150, 150))
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)

            if encodings:
                embedding = encodings[0]
                distances = face_recognition.face_distance(known_embeddings, embedding)
                min_distance = np.min(distances)
                min_index = np.argmin(distances)

                if min_distance < MATCH_THRESHOLD:
                    roll = known_rolls[min_index]
                    label = roll
                    present.add(roll)
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
            else:
                label = "No Encoding"
                color = (0, 0, 255)

        except:
            label = "Error"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Live Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()

# Final Attendance Report
absent = sorted(set(known_rolls) - present)
present = sorted(present)

print("\n📋 Live Attendance Report")
print(f"Present ({len(present)}): {present}")
print(f"Absent ({len(absent)}): {absent}")
