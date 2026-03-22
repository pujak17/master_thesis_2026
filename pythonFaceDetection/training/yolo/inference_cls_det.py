import os
import cv2
import time
import torch
import socket
import threading
import argparse
from collections import deque
from ultralytics import YOLO
import numpy as np

from ui import ask_username, ask_existing_user_action
from baseline import collect_user_baseline_video, get_embedding, detect_face_haar
from posture import collect_posture_baseline, compute_posture_from_frame
from gaze_agent import GazeAgentConfig, GazeAgentTracker


COLORS = {
    "neutral": (255, 230, 230),
    "happy": (200, 255, 200),
    "angry": (180, 180, 255),
}

BASELINE_WEIGHT = 0.7
YOLO_WEIGHT = 0.3

FRUSTRATION_LABELS = {"angry", "sad", "fear"}

WINDOW_DURATION = 5
DENSITY_THRESHOLD = 0.7

SMILE_COOLDOWN = 10
LOOK_COOLDOWN = 10
FRUSTRATION_COOLDOWN = 40
NO_FRUSTRATION_FIRST_SECONDS = 100

# Posture evidence: contributes a small weight to density, but does NOT trigger by itself
POSTURE_ONLY_WEIGHT = 0.25  # tune: 0.10..0.35

# Optional: frustration threshold on summed YOLO probs for FRUSTRATION_LABELS
FRUSTRATION_PROB_TH = 0.40

# For a tiny smile debouncing window (optional; still keyed on baseline+YOLO label)
HAPPY_WINDOW = 4
HAPPY_REQUIRED = 4
happy_hits = deque(maxlen=HAPPY_WINDOW)

conn = None
frustration_window = deque()  # (timestamp, weight in [0..1])
last_smile_time = 0
last_look_time = 0
last_frustration_send = 0


def safe_send(msg: bytes):
    global conn
    if conn is None:
        return
    try:
        conn.sendall(msg)
    except Exception:
        conn = None


def start_server(host="127.0.0.1", port=5002):
    global conn
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"Server listening on {host}:{port}")
    conn, addr = s.accept()
    print(f"Client connected from {addr}")


def load_user_coreset_embeddings(save_dir, username, emotions):
    per_class_embeddings = []
    sizes = []

    for emo in emotions:
        p = os.path.join(save_dir, username, emo, "coreset.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing baseline file: {p}")
        emb = torch.load(p, map_location="cpu")
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        per_class_embeddings.append(emb)
        sizes.append(len(emb))

    user_embeddings = torch.cat(per_class_embeddings, dim=0)
    return user_embeddings, sizes


def ask_agent_side_cv(window_name: str = "AgentSide") -> str:
    """
    Ask the user (via OpenCV window) whether the virtual agent is on the left or right side.
    Returns 'left' or 'right'.
    """
    w, h = 640, 240
    while True:
        img = 255 * np.ones((h, w, 3), dtype=np.uint8)

        cv2.putText(img, "Where will the virtual agent be?", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "[L] Left    [R] Right", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.putText(img, "Press L or R, Esc to quit", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(50) & 0xFF

        if key in (ord("l"), ord("L")):
            cv2.destroyWindow(window_name)
            return "left"
        if key in (ord("r"), ord("R")):
            cv2.destroyWindow(window_name)
            return "right"
        if key == 27:  # Esc
            cv2.destroyWindow(window_name)
            return "right"  # default fallback


def run_inference(args, posture_baseline, posture_threshold,
                  agent_side: str = "right", mirrored: bool = True):
    """
    mirrored=True means the webcam preview is mirrored (like a selfie),
    so if the user says 'right', the face moves to the left in image coordinates.
    """
    global last_smile_time, last_look_time, last_frustration_send, frustration_window, happy_hits

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    model = YOLO(args.emotion_classifier)

    # Load baseline embeddings (coreset.pt per emotion)
    user_embeddings, sizes = load_user_coreset_embeddings(args.save_dir, args.username, args.emotions)

    # Map user side to image side if mirrored
    if mirrored:
        image_agent_side = "left" if agent_side == "right" else "right"
    else:
        image_agent_side = agent_side

    # Configure virtual agent gaze tracker (very permissive window)
    gaze_cfg = GazeAgentConfig(
        agent_side=image_agent_side,
        offset_threshold=0.05,
        min_frames=3,
        min_ratio=1.0,
    )
    gaze_tracker = GazeAgentTracker(gaze_cfg)

    print("\n" + "=" * 60)
    print("INFERENCE STARTED (EMBEDDING BASELINE + GAZE)")
    print("=" * 60)
    print(f"Emotions: {args.emotions}")
    print(f"Posture threshold: {posture_threshold:.4f}")
    print(f"User agent side: {agent_side}, image side used: {image_agent_side}")
    print("Press 'q' to quit\n")

    inference_start_time = time.time()
    last_smile_state = False  # edge-trigger for smile
    last_look_state = False   # edge-trigger for look

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        # ---- Posture ----
        posture_score, posture_cues = compute_posture_from_frame(frame, posture_baseline)
        posture_conf_ok = bool(posture_cues.get("confidence_ok", False))
        posture_high = posture_conf_ok and (posture_score > posture_threshold)

        box = detect_face_haar(frame)
        if box is None:
            cv2.putText(frame, "No face detected", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.putText(frame, f"Posture: {posture_score:.2f}", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"PostureConf: {int(posture_conf_ok)}", (40, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        x, y, w, h = box
        x2, y2 = x + w, y + h
        face_crop = frame[y:y2, x:x2]
        if face_crop.size == 0:
            continue

        # ---- Gaze from face box ----
        frame_h, frame_w = frame.shape[:2]
        gaze_ready = gaze_tracker.update_from_face_box(frame_w, frame_h, x, y, w, h)
        gaze_offset = gaze_tracker.last_offset

        # ---- Baseline embedding similarity (per emotion) ----
        emb = get_embedding(face_crop, model)
        sims = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), user_embeddings)
        sims_split = torch.split(sims, sizes)
        baseline_scores = torch.tensor([s.mean() for s in sims_split])

        # ---- YOLO probs ----
        out = model(face_crop, verbose=False)[0]
        probs = out.probs.data.cpu().numpy()
        labels = list(out.names.values())
        yolo_scores = {l: float(probs[i]) for i, l in enumerate(labels)}

        # ---- Fuse baseline + YOLO ----
        fused = {}
        for idx, emo in enumerate(args.emotions):
            b = float(baseline_scores[idx])
            yprob = yolo_scores.get(emo, 0.0)
            fused[emo] = BASELINE_WEIGHT * b + YOLO_WEIGHT * yprob

        final_label = max(fused, key=fused.get)
        final_score = fused[final_label]

        # ---- Frustration (emotion-only trigger) ----
        frustration_prob = sum(yolo_scores.get(f, 0.0) for f in FRUSTRATION_LABELS)

        emotion_frustrated = (
                final_label == "angry"
                or frustration_prob > FRUSTRATION_PROB_TH
        )

        # Posture evidence (not a trigger)
        w_score = 1.0 if emotion_frustrated else (POSTURE_ONLY_WEIGHT if posture_high else 0.0)

        now = time.time()
        frustration_window.append((now, w_score))
        while frustration_window and now - frustration_window[0][0] > WINDOW_DURATION:
            frustration_window.popleft()

        density = (sum(v for _, v in frustration_window) / len(frustration_window)) if frustration_window else 0.0

        if density >= DENSITY_THRESHOLD:
            if now - inference_start_time >= NO_FRUSTRATION_FIRST_SECONDS:
                if now - last_frustration_send >= FRUSTRATION_COOLDOWN:
                    safe_send(b"INTERVENTION:frustrated\n")
                    last_frustration_send = now
                    print("[ALERT] Frustration detected - intervention triggered")

        # ---- Smile / look interventions ----

        # Smile evidence from fused label (embedding baseline + YOLO)
        happy_label = (final_label == "happy")

        # Tiny debouncing window: count how many of last frames were "happy"
        happy_hits.append(1 if happy_label else 0)
        happy_ready = (len(happy_hits) == HAPPY_WINDOW) and (sum(happy_hits) >= HAPPY_REQUIRED)

        # 1) SMILE intervention (10s cooldown), driven by baseline+YOLO label
        smile_trigger = happy_ready

        if smile_trigger and not last_smile_state:
            if now - last_smile_time >= SMILE_COOLDOWN:
                safe_send(b"INTERVENTION:smile\n")
                last_smile_time = now
                print("[POSITIVE] Smile (baseline+YOLO) - SMILE intervention")

        last_smile_state = smile_trigger

        # 2) LOOK intervention with its own 10s cooldown.
        # Only fires when not already in "happy_ready", so you get a separate LOOK event.
        look_trigger = gaze_ready and not happy_ready

        if look_trigger and not last_look_state:
            if now - last_look_time >= LOOK_COOLDOWN:
                safe_send(b"INTERVENTION:look\n")
                last_look_time = now
                print("[INFO] Gaze at agent - LOOK intervention")

        last_look_state = look_trigger

        # ---- Draw UI ----
        cv2.rectangle(frame, (x, y), (x2, y2), COLORS.get(final_label, (255, 255, 255)), 3)
        cv2.putText(frame, f"{final_label} ({final_score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS.get(final_label, (255, 255, 255)), 2)

        cv2.putText(frame, f"Posture: {posture_score:.2f} (th={posture_threshold:.2f})", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"PostureHigh: {int(posture_high)} Conf: {int(posture_conf_ok)}", (40, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Optional posture cues for debugging
        if posture_conf_ok:
            cv2.putText(
                frame,
                f"H:{posture_cues.get('hunch', 0.0):.2f} HD:{posture_cues.get('head_down', 0.0):.2f} "
                f"G:{posture_cues.get('guarded', 0.0):.2f} S:{posture_cues.get('stiff', 0.0):.2f}",
                (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )
        else:
            cv2.putText(frame, "Posture cues: low confidence", (40, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

        cv2.putText(frame, f"FrustDensity: {density:.2f}", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Show gaze + smile status
        cv2.putText(frame,
                    f"Gready:{int(gaze_ready)} Off:{gaze_offset:.2f} Hready:{int(happy_ready)}",
                    (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        elapsed = int(now - inference_start_time)
        cv2.putText(frame, f"Runtime: {elapsed}s", (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--emotion_classifier",
        default="/Users/puja/IdeaProjects/charamelFaceDetection/data/raf-ck-fer-affect-ferplus/best.pt",
    )
    p.add_argument(
        "--save_dir",
        default="/Users/puja/IdeaProjects/charamelFaceDetection/data/user_data",
    )
    p.add_argument("--emotions", default=["neutral", "happy", "angry"])
    return p.parse_args()


def main():
    args = parse_args()

    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(1)

    username = ask_username()
    if username is None:
        print("Username not provided. Exiting.")
        return
    args.username = username

    user_path = os.path.join(args.save_dir, args.username)
    if os.path.exists(user_path):
        mode = ask_existing_user_action(user_path)  # returns "replace" | "add" | "use"
    else:
        mode = "replace"

    # Collect / update embedding baseline (coreset.pt)
    collect_user_baseline_video(args, mode)

    print("Collecting posture baseline")
    posture_baseline, posture_threshold = collect_posture_baseline()

    # Ask for virtual agent side via OpenCV UI
    agent_side = ask_agent_side_cv()
    print(f"✓ Virtual agent side (user view): {agent_side}")

    run_inference(args, posture_baseline, posture_threshold,
                  agent_side=agent_side, mirrored=True)


if __name__ == "__main__":
    main()
