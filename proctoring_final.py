import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import os
from ultralytics import YOLO
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Load YOLOv8m model
model = YOLO('yolov8m.pt')

# Class names for YOLO
PHONE_CLASSES = ['cell phone', 'book']

# Initialize counters
if 'phone_detected_events' not in st.session_state:
    st.session_state['phone_detected_events'] = 0
if 'book_detected_events' not in st.session_state:
    st.session_state['book_detected_events'] = 0
if 'multiple_faces_detected_events' not in st.session_state:
    st.session_state['multiple_faces_detected_events'] = 0
if 'no_face_detected_events' not in st.session_state:
    st.session_state['no_face_detected_events'] = 0
if 'proctoring_active' not in st.session_state:
    st.session_state['proctoring_active'] = False
if 'proctoring_stopped' not in st.session_state:
    st.session_state['proctoring_stopped'] = False

def save_violation_image(frame, label):
    if not os.path.exists("violations"):
        os.makedirs("violations")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violations/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

st.title("AI Proctoring System (Final Proctor Score Version)")

start = st.button("Start Proctoring")
stop = st.button("Stop Proctoring")

FRAME_WINDOW = st.image([])

if start and not st.session_state['proctoring_active']:
    st.session_state['proctoring_active'] = True
    st.session_state['proctoring_stopped'] = False

if stop and st.session_state['proctoring_active']:
    st.session_state['proctoring_active'] = False
    st.session_state['proctoring_stopped'] = True

if st.session_state['proctoring_active']:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    last_phone_detected = 0
    last_book_detected = 0
    last_multiple_faces_detected = 0
    last_no_face_detected = 0

    cooldown_time = 5  # seconds cooldown to prevent multiple counts

    total_checks = 0

    log_file = open("violation_log.txt", "a", encoding="utf-8")

    while st.session_state['proctoring_active']:
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

        current_time = time.time()
        total_checks += 1

        # Alerts based on face count
        if len(faces) > 1 and current_time - last_multiple_faces_detected > cooldown_time:
            st.session_state['multiple_faces_detected_events'] += 1
            log_file.write(f"{timestamp} - Multiple Faces Detected\n")
            save_violation_image(frame, "multiple_faces")
            last_multiple_faces_detected = current_time

        elif len(faces) == 0 and current_time - last_no_face_detected > cooldown_time:
            st.session_state['no_face_detected_events'] += 1
            log_file.write(f"{timestamp} - No Face Detected\n")
            save_violation_image(frame, "no_face")
            last_no_face_detected = current_time

        # Object detection (YOLO)
        results = model.predict(source=frame, save=False, conf=0.5, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, cls in zip(boxes, classes):
                class_name = model.names[cls]
                if class_name in PHONE_CLASSES:
                    x1, y1, x2, y2 = box.astype(int)
                    if class_name == 'cell phone' and current_time - last_phone_detected > cooldown_time:
                        st.session_state['phone_detected_events'] += 1
                        log_file.write(f"{timestamp} - Phone Detected\n")
                        save_violation_image(frame, "phone")
                        last_phone_detected = current_time
                    elif class_name == 'book' and current_time - last_book_detected > cooldown_time:
                        st.session_state['book_detected_events'] += 1
                        log_file.write(f"{timestamp} - Book Detected\n")
                        save_violation_image(frame, "book")
                        last_book_detected = current_time

        # Update Streamlit live frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if st.session_state['proctoring_stopped']:
            st.session_state['proctoring_active'] = False

    cap.release()
    log_file.close()

if st.session_state['proctoring_stopped']:
    # Calculate final proctor score
    total_violations = (st.session_state['phone_detected_events'] +
                        st.session_state['book_detected_events'] +
                        st.session_state['multiple_faces_detected_events'] +
                        st.session_state['no_face_detected_events'])

    if total_violations > 0:
        violation_penalty = (0.2 * st.session_state['phone_detected_events'] +
                             0.2 * st.session_state['book_detected_events'] +
                             0.3 * st.session_state['multiple_faces_detected_events'] +
                             0.3 * st.session_state['no_face_detected_events'])
        proctor_score = max(0, 1 - violation_penalty / total_violations)
    else:
        proctor_score = 1

    st.success(f"Proctoring Session Ended. Final Proctor Score: {proctor_score:.2f}")
    st.write(f"Phones Detected: {st.session_state['phone_detected_events']}")
    st.write(f"Books Detected: {st.session_state['book_detected_events']}")
    st.write(f"Multiple Faces Detected: {st.session_state['multiple_faces_detected_events']}")
    st.write(f"No Face Detected: {st.session_state['no_face_detected_events']}")
