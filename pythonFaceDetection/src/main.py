#!/usr/bin/env python3
# main.py

import threading
import socket
import json
import time

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────────────────────────────────────
#  1) State‐server globals & thread
# ──────────────────────────────────────────────────────────────────────────────
HOST, PORT = "127.0.0.1", 5002
state_lock = threading.Lock()
current_state = "neutral"


def state_server():
    """Listen on TCP port and reply the latest current_state as JSON."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[Python] State server listening on {HOST}:{PORT}")
    while True:
        conn, _ = srv.accept()
        with conn:
            with state_lock:
                reply = {"state": current_state}
            conn.sendall((json.dumps(reply) + "\n").encode())


threading.Thread(target=state_server, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
#  2) Load model and labels
# ──────────────────────────────────────────────────────────────────────────────
model = load_model("/Users/puja/IdeaProjects/charamelFaceDetection/models/charamel_emotion_model.h5")
labels = ["frustrated", "smile", "neutral"]

# ──────────────────────────────────────────────────────────────────────────────
#  3) MediaPipe setup
# ──────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
movement_history = deque(maxlen=10)


# ──────────────────────────────────────────────────────────────────────────────
#  4) Detection logic
# ──────────────────────────────────────────────────────────────────────────────
def predict_emotion(face_img):
    img = cv2.resize(face_img, (224, 224))
    img = img.astype("float32") / 255.0
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    return labels[np.argmax(pred)]


def detect_gaze(lm):
    left = lm[33]
    right = lm[263]
    nose = lm[1]
    eye_c = (left.x + right.x) / 2
    return "eye_contact" if abs(nose.x - eye_c) < 0.02 else "looking_away"


def detect_movement(ls, rs):
    center = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    movement_history.append(center)
    if len(movement_history) > 1:
        deltas = [np.linalg.norm(movement_history[i] - movement_history[i - 1]) for i in
                  range(1, len(movement_history))]
        return "restless" if np.mean(deltas) > 0.02 else "calm"
    return "calm"


def decide_state(expr, gaze, movement):
    if expr == "frustrated":
        return "frustrated"
    if expr == "smile" and gaze == "eye_contact":
        return "happy"
    if gaze == "eye_contact" and movement == "calm":
        return "focused"
    if movement == "restless":
        return "restless"
    return "neutral"


# ──────────────────────────────────────────────────────────────────────────────
#  5) Webcam loop
# ──────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Unable to open webcam.")
    exit(1)

print("[Python] Starting webcam loop. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    expr, gaze = None, None
    faces = face_mesh.process(rgb).multi_face_landmarks
    if faces:
        lm = faces[0].landmark
        h, w, _ = frame.shape
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        x0, y0 = max(min(p[0] for p in pts), 0), max(min(p[1] for p in pts), 0)
        x1, y1 = min(max(p[0] for p in pts), w), min(max(p[1] for p in pts), h)
        roi = frame[y0:y1, x0:x1]
        if roi.size:
            expr = predict_emotion(roi)
        gaze = detect_gaze(lm)
        mp_drawing.draw_landmarks(frame, faces[0], mp_face_mesh.FACEMESH_TESSELATION)

    movement = None
    poses = pose.process(rgb).pose_landmarks
    if poses:
        lm = poses.landmark
        ls, rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        movement = detect_movement(ls, rs)
        mp_drawing.draw_landmarks(frame, poses, mp_pose.POSE_CONNECTIONS)

    user_state = decide_state(expr, gaze, movement)

    cv2.putText(frame, f"State: {user_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    if gaze:
        cv2.putText(frame, f"Gaze: {gaze}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2)

    with state_lock:
        current_state = user_state

    cv2.imshow("Smart Interruption System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
