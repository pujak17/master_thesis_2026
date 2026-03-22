import os
import cv2
import time
import socket
import threading
import argparse
import shutil
from collections import deque
from ultralytics import YOLO
import numpy as np

from ui import ask_username, ask_existing_user_action
from posture import (
    collect_posture_baseline,
    compute_posture_from_frame,
    save_posture_baseline,
    load_posture_baseline,
)
from baseline_v4 import collect_neutral_frames, compute_thresholds, load_thresholds, detect_face_haar
from gaze_eye_tracker import GazeFusionConfig, GazeFusionTracker
from logger import ExperimentLogger


COLORS = {
    "neutral":  (255, 230, 230),
    "happy":    (200, 255, 200),
    "angry":    (180, 180, 255),
    "occluded": (200, 200, 200),
}

FRUSTRATION_LABELS = {"angry", "sad", "fear"}

WINDOW_DURATION = 7
DENSITY_THRESHOLD = 0.4

SMILE_COOLDOWN = 10
LOOK_COOLDOWN = 10
FRUSTRATION_COOLDOWN = 30
NO_FRUSTRATION_FIRST_SECONDS = 100

POSTURE_ONLY_WEIGHT = 0.25

HAPPY_WINDOW = 2
HAPPY_REQUIRED = 2

TASK_STUCK_SECONDS = 90
TASK_STUCK_COOLDOWN = 60

conn = None
frustration_window = deque()
happy_hits = deque(maxlen=HAPPY_WINDOW)
last_smile_time = 0
last_look_time = 0
last_frustration_send = 0
last_task_stuck_send = 0


def count_image_files(folder: str) -> int:
    if not os.path.exists(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def safe_send(msg: bytes):
    global conn
    if conn is None:
        return
    try:
        conn.sendall(msg)
    except Exception:
        conn = None


def safe_send_required(msg: bytes):
    safe_send(msg)


def start_server(host: str = "127.0.0.1", port: int = 5002):
    global conn
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"Server listening on {host}:{port}")
    conn, addr = s.accept()
    print(f"Client connected from {addr}")


def ask_agent_side_cv(window_name: str = "AgentSide") -> str:
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
        if key == 27:
            cv2.destroyWindow(window_name)
            return "right"


class OnsetTracker:
    def __init__(self):
        self._onset_time = None

    def update(self, is_active: bool):
        if is_active and self._onset_time is None:
            self._onset_time = time.time()
        elif not is_active:
            self._onset_time = None

    def lag(self) -> float:
        if self._onset_time is None:
            return -1.0
        return round(time.time() - self._onset_time, 2)

    def reset(self):
        self._onset_time = None


def run_inference(
        args,
        posture_baseline,
        posture_threshold: float,
        anger_th: float,
        frustration_th: float,
        happy_th: float,
        exp_logger: ExperimentLogger,
        agent_side: str = "right",
        mirrored: bool = True,
):
    global last_smile_time, last_look_time, last_frustration_send, last_task_stuck_send
    global frustration_window, happy_hits

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    yolo = YOLO(args.emotion_classifier)

    gaze_cfg = GazeFusionConfig(
        agent_side=agent_side,
        mirrored=mirrored,
        head_offset_th=0.15,
        min_frames=3,
        min_ratio=1.0,
    )
    gaze_tracker = GazeFusionTracker(gaze_cfg)
    onset_tracker = OnsetTracker()

    print("\n" + "=" * 60)
    print("INFERENCE STARTED")
    print("=" * 60)
    print(f"Anger threshold:       {anger_th:.4f}")
    print(f"Frustration threshold: {frustration_th:.4f}")
    print(f"Happy threshold:       {happy_th:.4f}")
    print(f"Posture threshold:     {posture_threshold:.4f}")
    print(f"Agent side:            {agent_side}")
    print(f"Task stuck trigger:    {TASK_STUCK_SECONDS}s  (cooldown {TASK_STUCK_COOLDOWN}s)")
    print("Press 'q' to quit\n")

    inference_start_time = time.time()
    last_smile_state = False
    last_look_state = False
    frame_counter = 0
    opening_wave_done = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            frame_counter += 1
            now = time.time()
            elapsed = now - inference_start_time

            if not opening_wave_done and elapsed > 30:
                safe_send(b"INTERVENTION:look\n")
                exp_logger.log_intervention("look_proactive", {}, 0.0, anger_th, happy_th)
                opening_wave_done = True
                print("[INFO] Proactive opening wave sent")

            posture_score, cues = compute_posture_from_frame(frame, posture_baseline)
            posture_conf_ok = bool(cues.get("confidence_ok", False))
            posture_high = posture_conf_ok and (posture_score > posture_threshold)

            box = detect_face_haar(frame)

            if box is None:
                if frame_counter % 30 == 0:
                    exp_logger.log_frame(
                        scores={},
                        frustration_score=0.0,
                        anger_th=anger_th,
                        happy_th=happy_th,
                        intervention="no_face",
                    )
                cv2.putText(frame, "No face detected", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f"Posture: {posture_score:.2f}", (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                if posture_conf_ok:
                    cv2.putText(
                        frame,
                        f"Sp:{cues.get('spine_angle', 0):.2f} Hd:{cues.get('head_angle', 0):.2f} "
                        f"Drop:{cues.get('head_drop', 0):.2f} Mot:{cues.get('motion', 0):.2f}",
                        (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    )
                else:
                    cv2.putText(frame, "Posture confidence low", (40, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imshow("Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            x, y, w_box, h_box = box
            face = frame[y:y + h_box, x:x + w_box]
            if face.size == 0:
                continue

            gaze_ready, gaze_offset = gaze_tracker.update(frame, (x, y, w_box, h_box))

            out = yolo(face, verbose=False)[0]
            probs = out.probs.data.cpu().numpy()
            labels = list(out.names.values())
            scores = {l: float(probs[i]) for i, l in enumerate(labels)}

            final_label = max(scores, key=scores.get)
            final_score = scores[final_label]

            if frame_counter % 30 == 0:
                all_scores_str = "  ".join(
                    f"{k}:{v:.3f}" for k, v in sorted(scores.items())
                )
                print(
                    f"[LIVE f{frame_counter:05d}]  "
                    f"happy:{scores.get('happy', 0):.4f}  "
                    f"angry:{scores.get('angry', 0):.4f}  | {all_scores_str}"
                )

            frustration_prob = sum(scores.get(f, 0.0) for f in FRUSTRATION_LABELS)
            emotion_frustrated = (
                    scores.get("angry", 0.0) > anger_th
                    or frustration_prob > frustration_th
            )

            w = 1.0 if emotion_frustrated else (POSTURE_ONLY_WEIGHT if posture_high else 0.0)
            onset_tracker.update(emotion_frustrated)

            exp_logger.log_frame(
                scores=scores,
                frustration_score=frustration_prob,
                anger_th=anger_th,
                happy_th=happy_th,
            )

            frustration_window.append((now, w))
            while frustration_window and now - frustration_window[0][0] > WINDOW_DURATION:
                frustration_window.popleft()

            density = (
                sum(v for _, v in frustration_window) / len(frustration_window)
                if frustration_window else 0.0
            )

            if density >= DENSITY_THRESHOLD:
                lag = onset_tracker.lag()
                frustration_window.clear()
                onset_tracker.reset()

                if elapsed > NO_FRUSTRATION_FIRST_SECONDS:
                    if now - last_frustration_send >= FRUSTRATION_COOLDOWN:
                        exp_logger.log_intervention(
                            "frustrated", scores, frustration_prob,
                            anger_th, happy_th, onset_lag=lag,
                        )
                        safe_send(b"INTERVENTION:frustrated\n")
                        last_frustration_send = now
                        print(f"[ALERT] Frustration intervention - onset lag: {lag:.1f}s")

            stuck_secs = exp_logger.task_stuck_seconds()
            if stuck_secs >= TASK_STUCK_SECONDS:
                if now - last_task_stuck_send >= TASK_STUCK_COOLDOWN:
                    exp_logger.log_intervention(
                        "task_stuck", scores, frustration_prob,
                        anger_th, happy_th,
                    )
                    safe_send(b"INTERVENTION:frustrated\n")
                    last_task_stuck_send = now
                    print(
                        f"[ALERT] Task-stuck frustration intervention - "
                        f"{stuck_secs:.0f}s on '{exp_logger.current_task_title}'"
                    )

            happy = scores.get("happy", 0.0)
            happy_strong = happy > happy_th
            happy_hits.append(1 if happy_strong else 0)
            happy_ready = (
                    len(happy_hits) == HAPPY_WINDOW
                    and sum(happy_hits) >= HAPPY_REQUIRED
            )

            smile_trigger = happy_ready and gaze_ready
            if smile_trigger and not last_smile_state:
                if now - last_smile_time >= SMILE_COOLDOWN:
                    exp_logger.log_intervention(
                        "smile", scores, frustration_prob, anger_th, happy_th,
                    )
                    safe_send(b"INTERVENTION:smile\n")
                    last_smile_time = now
                    print("[POSITIVE] Smile + gaze - smile intervention")
            last_smile_state = smile_trigger

            look_trigger = gaze_ready and not happy_ready
            if look_trigger and not last_look_state:
                if now - last_look_time >= LOOK_COOLDOWN:
                    exp_logger.log_intervention(
                        "look", scores, frustration_prob, anger_th, happy_th,
                    )
                    safe_send(b"INTERVENTION:look\n")
                    last_look_time = now
                    print("[INFO] Gaze at agent - look intervention")
            last_look_state = look_trigger

            cv2.rectangle(
                frame, (x, y), (x + w_box, y + h_box),
                COLORS.get(final_label, (255, 255, 255)), 3,
            )
            cv2.putText(
                frame,
                f"{final_label} ({final_score:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                COLORS.get(final_label, (255, 255, 255)), 2,
            )
            cv2.putText(frame, f"Posture: {posture_score:.2f}", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            if posture_conf_ok:
                cv2.putText(
                    frame,
                    f"Sp:{cues.get('spine_angle', 0):.2f} Hd:{cues.get('head_angle', 0):.2f} "
                    f"Drop:{cues.get('head_drop', 0):.2f} Mot:{cues.get('motion', 0):.2f}",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                )
            else:
                cv2.putText(frame, "Posture confidence low", (40, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.putText(frame, f"FrustDensity: {density:.2f}", (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            stuck_display = f"{stuck_secs:.0f}s" if stuck_secs >= 0 else "---"
            neutral = scores.get("neutral", 0.0)
            cv2.putText(
                frame,
                f"Happy:{happy:.2f} Neutral:{neutral:.2f} "
                f"Hready:{int(happy_ready)} Gready:{int(gaze_ready)} Off:{gaze_offset:.2f}",
                (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )
            cv2.putText(
                frame,
                f"Runtime:{int(elapsed)}s  OnsetLag:{onset_tracker.lag():.1f}s  "
                f"TaskStuck:{stuck_display}/{TASK_STUCK_SECONDS}s",
                (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

            cv2.imshow("Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        exp_logger.close()


def parse_args():
    p = argparse.ArgumentParser(description="Emotion Recognition Inference System")
    p.add_argument(
        "--emotion_classifier",
        default="/Users/puja/IdeaProjects/charamelFaceDetection/data/raf-ck-fer-affect-ferplus/best.pt",
        help="Path to YOLO emotion classifier model",
    )
    p.add_argument(
        "--save_dir",
        default="/Users/puja/IdeaProjects/charamelFaceDetection/data/user_data",
        help="Directory to save user data",
    )
    return p.parse_args()


def main():
    print("\n" + "=" * 60)
    print("EMOTION RECOGNITION SYSTEM")
    print("=" * 60 + "\n")

    args = parse_args()

    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(1)

    username = ask_username()
    if username is None:
        print("ERROR: Username not provided. Exiting.")
        return
    args.username = username
    print(f"✓ User: {username}\n")

    exp_logger = ExperimentLogger(args.save_dir, args.username)
    exp_logger.start_server()

    agent_side = ask_agent_side_cv()
    print(f"✓ Virtual agent side (user view): {agent_side}\n")

    frames_dir = os.path.join(args.save_dir, args.username, "neutral", "frames")
    img_count = count_image_files(frames_dir)
    has_existing = img_count > 0
    need_recompute = True
    action = None

    if has_existing:
        print("=" * 60)
        print("EXISTING BASELINE FOUND")
        print("=" * 60)
        print(f"Found {img_count} existing neutral frames")

        action = ask_existing_user_action(os.path.join(args.save_dir, args.username))
        print(f"Selected: {action}\n")

        if action == "replace":
            shutil.rmtree(frames_dir, ignore_errors=True)
            os.makedirs(frames_dir, exist_ok=True)
            threshold_file = os.path.join(args.save_dir, args.username, "thresholds.json")
            if os.path.exists(threshold_file):
                os.remove(threshold_file)

            posture_file = os.path.join(args.save_dir, args.username, "posture_baseline.json")
            if os.path.exists(posture_file):
                os.remove(posture_file)

            collect_neutral_frames(args, num_frames=300)
            need_recompute = True

        elif action == "add":
            collect_neutral_frames(args, num_frames=300)
            need_recompute = True

        else:
            print("Using existing baseline and saved thresholds - skipping recalibration.\n")
            need_recompute = False

    else:
        collect_neutral_frames(args, num_frames=300)
        need_recompute = True

    print("=" * 60)
    print("STEP 2: THRESHOLDS")
    print("=" * 60)

    if need_recompute:
        anger_th, frustration_th, happy_th = compute_thresholds(args, show_ui=True)
    else:
        anger_th, frustration_th, happy_th = load_thresholds(args)

    print("=" * 60)
    print("STEP 3: POSTURE BASELINE COLLECTION")
    print("=" * 60)

    posture_file = os.path.join(args.save_dir, args.username, "posture_baseline.json")

    if has_existing and action == "use":
        posture_baseline, posture_threshold = load_posture_baseline(posture_file)
        if posture_baseline is not None and posture_threshold is not None:
            print("Using existing saved posture baseline.\n")
        else:
            print("No saved posture baseline found - collecting a new one.\n")
            posture_baseline, posture_threshold = collect_posture_baseline(
                frames=40, show_ui=True, countdown=3
            )
            if posture_baseline is not None:
                save_posture_baseline(posture_file, posture_baseline, posture_threshold)
    else:
        posture_baseline, posture_threshold = collect_posture_baseline(
            frames=40, show_ui=True, countdown=3
        )
        if posture_baseline is not None:
            save_posture_baseline(posture_file, posture_baseline, posture_threshold)

    print("=" * 60)
    print("STEP 4: STARTING INFERENCE")
    print("=" * 60)
    run_inference(
        args,
        posture_baseline,
        posture_threshold,
        anger_th,
        frustration_th,
        happy_th,
        exp_logger=exp_logger,
        agent_side=agent_side,
        mirrored=True,
    )


if __name__ == "__main__":
    main()
