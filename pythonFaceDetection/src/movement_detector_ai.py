import os
import cv2
import time
import socket
import threading
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
import argparse
import mediapipe as mp

# ====== From your baseline code ======
from baseline import get_embedding
from posture import collect_posture_baseline, compute_posture_from_frame

# Mediapipe face mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)


# =========================================================
# Globals
# =========================================================
conn = None
anger_window = deque()
last_intervention_time = 0

BASELINE_WEIGHT = 0.7
YOLO_WEIGHT = 0.3

FRUSTRATION_LABELS = {"angry", "sad", "fear"}

WINDOW_DURATION = 5
DENSITY_THRESHOLD = 0.6
POSTURE_THRESHOLD = 0.55

# Pastel colors (BGR)
COLOR_NEUTRAL = (255, 230, 230)
COLOR_HAPPY   = (200, 255, 200)
COLOR_ANGRY   = (180, 180, 255)

COLORS = {
    "neutral": COLOR_NEUTRAL,
    "happy": COLOR_HAPPY,
    "angry": COLOR_ANGRY
}


# =========================================================
# TCP server for VSM
# =========================================================
def start_server(host="127.0.0.1", port=5002):
    global conn
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"Server listening on {host}:{port}")
    conn, addr = s.accept()
    print("Connected:", addr)


# =========================================================
# Mediapipe face bounding box
# =========================================================
def detect_face_mediapipe(frame):
    """Returns bounding box (x,y,w,h) using Mediapipe FaceMesh."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None

    h, w, _ = frame.shape
    face = res.multi_face_landmarks[0]

    xs = [lm.x * w for lm in face.landmark]
    ys = [lm.y * h for lm in face.landmark]

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    return (x1, y1, x2 - x1, y2 - y1)



# =========================================================
# Inference Loop
# =========================================================
def run_inference(args, posture_baseline):
    global last_intervention_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not found.")
        return

    yolo = YOLO(args.emotion_classifier)

    # Load baseline embeddings
    per_class_embeddings = []
    class_sizes = []

    for emo in args.emotions:
        path = os.path.join(args.save_dir, args.username, emo, "coreset.pt")
        emb = torch.load(path)
        per_class_embeddings.append(emb)
        class_sizes.append(len(emb))

    user_embeddings = torch.cat(per_class_embeddings, dim=0)
    print("\nInference started...\n")

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            # -------------------------
            # Posture Detection
            # -------------------------
            posture_score = compute_posture_from_frame(frame, posture_baseline)

            # -------------------------
            # Face Detection (mediapipe)
            # -------------------------
            box = detect_face_mediapipe(frame)
            if not box:
                cv2.imshow("Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            x, y, w, h = box
            if w <= 0 or h <= 0:
                continue

            # clamp
            h_frame, w_frame, _ = frame.shape
            if x < 0 or y < 0 or x+w > w_frame or y+h > h_frame:
                continue

            face_crop = frame[y:y+h, x:x+w]

            # -------------------------
            # Baseline Prediction
            # -------------------------
            emb = get_embedding(face_crop, yolo)
            sims = torch.nn.functional.cosine_similarity(emb.unsqueeze(0),
                                                         user_embeddings)
            sims_split = torch.split(sims, class_sizes)
            baseline_scores = torch.tensor([s.mean() for s in sims_split])

            # -------------------------
            # YOLO Prediction
            # -------------------------
            yolo_out = yolo(face_crop, verbose=False)[0]
            yolo_probs = yolo_out.probs.data.cpu().numpy()
            yolo_labels = list(yolo_out.names.values())

            # -------------------------
            # Fusion
            # -------------------------
            fused = {}
            for em in args.emotions:
                b = float(baseline_scores[args.emotions.index(em)])
                y = float(yolo_probs[yolo_labels.index(em)]) if em in yolo_labels else 0.0
                fused[em] = BASELINE_WEIGHT * b + YOLO_WEIGHT * y

            final_label = max(fused, key=fused.get)

            # -------------------------
            # Frustration = angry OR sad OR fear OR posture
            # -------------------------
            frustration_prob = sum(
                yolo_probs[yolo_labels.index(f)]
                for f in FRUSTRATION_LABELS if f in yolo_labels
            )

            is_frustrated = (
                    final_label == "angry"
                    or frustration_prob > 0.40
                    or posture_score > POSTURE_THRESHOLD
            )

            # -------------------------
            # Sliding Window
            # -------------------------
            now = time.time()
            anger_window.append((now, is_frustrated))

            while anger_window and now - anger_window[0][0] > WINDOW_DURATION:
                anger_window.popleft()

            density = sum(1 for (_, v) in anger_window if v) / len(anger_window)

            # -------------------------
            # Intervention Trigger
            # -------------------------
            if density >= DENSITY_THRESHOLD and now - last_intervention_time > 1:
                if conn:
                    conn.sendall(b"INTERVENTION:frustrated\n")
                last_intervention_time = now

            # -------------------------
            # Draw bounding box
            # -------------------------
            color = COLORS.get(final_label, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

            cv2.putText(frame, final_label,
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            cv2.putText(frame, f"posture={posture_score:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 220, 180), 2)

            cv2.imshow("Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()



# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion_classifier",
                        default="/Users/puja/IdeaProjects/charamelFaceDetection/data/raf-ck-fer-affect-ferplus/best.pt")
    parser.add_argument("--username", default="puja")
    parser.add_argument("--save_dir",
                        default="/Users/puja/IdeaProjects/charamelFaceDetection/data/user_data")
    parser.add_argument("--emotions", default=["neutral", "happy", "angry"])

    return parser.parse_args()



# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    threading.Thread(target=start_server, daemon=True).start()

    print("\nCollecting posture baseline...")
    posture_baseline = collect_posture_baseline()
    print("Posture baseline complete.")

    print("\nStarting inference...\n")
    run_inference(args, posture_baseline)



if __name__ == "__main__":
    main()
