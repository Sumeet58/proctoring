import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Load YOLOv8m model
model = YOLO('yolov8m.pt')

# Class names for YOLO
PHONE_CLASSES = ['cell phone', 'book']

# Initialize counters
phone_detected_count = 0
book_detected_count = 0
multiple_faces_detected_count = 0
no_face_detected_count = 0

# Open log file
log_file = open("violation_log.txt", "a", encoding="utf-8")

st.title("AI Proctoring System (Advanced Version)")

start = st.button("Start Proctoring")

FRAME_WINDOW = st.image([])

if start:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection using Mediapipe
        results = mp_face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                faces.append([x, y, w, h])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Alerts based on face count
        if len(faces) > 1:
            multiple_faces_detected_count += 1
            log_file.write(f"{timestamp} - Multiple Faces Detected\n")
            cv2.putText(frame, "ALERT: Multiple faces detected!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif len(faces) == 0:
            no_face_detected_count += 1
            log_file.write(f"{timestamp} - No Face Detected\n")
            cv2.putText(frame, "ALERT: No face detected!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Object detection (YOLO)
        results = model.predict(source=frame, save=False, conf=0.5, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, cls in zip(boxes, classes):
                class_name = model.names[cls]
                if class_name in PHONE_CLASSES:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name.upper()}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if class_name == 'cell phone':
                        phone_detected_count += 1
                        log_file.write(f"{timestamp} - Phone Detected\n")
                        cv2.putText(frame, "ALERT: PHONE detected!", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif class_name == 'book':
                        book_detected_count += 1
                        log_file.write(f"{timestamp} - Book Detected\n")
                        cv2.putText(frame, "ALERT: BOOK detected!", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update Streamlit live
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.metric(label="Phones Detected", value=phone_detected_count)
        st.metric(label="Books Detected", value=book_detected_count)
        st.metric(label="Multiple Faces Detected", value=multiple_faces_detected_count)
        st.metric(label="No Face Detected", value=no_face_detected_count)

    cap.release()
    log_file.close()
