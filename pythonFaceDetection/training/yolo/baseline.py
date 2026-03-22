import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

COLOR_NEUTRAL = (255, 230, 230)
COLOR_HAPPY = (200, 255, 200)
COLOR_ANGRY = (180, 180, 255)

CAPTURE_COLORS = {
    "neutral": COLOR_NEUTRAL,
    "happy": COLOR_HAPPY,
    "angry": COLOR_ANGRY,
}


def get_embedding(image, model):
    img = cv2.resize(image, (model.ckpt["train_args"]["imgsz"],
                             model.ckpt["train_args"]["imgsz"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    t = t.unsqueeze(0)

    feats = []
    with torch.no_grad():
        x = t
        for i, layer in enumerate(model.model.model):
            x = layer(x)
            if i == len(model.model.model) - 1:
                x = x[1]
                pooled = x
            else:
                pooled = x.mean(dim=(2, 3))
            feats.append(pooled)

    return torch.cat(feats, dim=1).squeeze().cpu()


def detect_face_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces.tolist(), key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def get_head_movement_instruction(count, samples_needed):
    """Return head movement instruction based on capture progress."""
    progress_ratio = count / samples_needed

    if progress_ratio < 0.25:
        return "Look STRAIGHT at camera", (255, 255, 255)
    elif progress_ratio < 0.50:
        return "Turn head LEFT and RIGHT slowly", (100, 200, 255)
    elif progress_ratio < 0.75:
        return "Look UP and DOWN slowly", (100, 255, 200)
    else:
        return "Move head at different angles", (200, 100, 255)


def collect_user_baseline_video(args, mode="replace"):
    user_path = os.path.join(args.save_dir, args.username)

    if mode == "use":
        print("Using existing baseline.")
        return

    if mode == "replace" and os.path.exists(user_path):
        import shutil
        shutil.rmtree(user_path)

    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    model = YOLO(args.emotion_classifier)
    emotions = ["neutral", "happy", "angry"]

    print(f"Starting baseline for user {args.username}")

    for emotion in emotions:
        emotion_dir = os.path.join(user_path, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        embeddings = []

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (20, 20), (w-20, 300), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"EMOTION: {emotion.upper()}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, CAPTURE_COLORS[emotion], 2)

            cv2.putText(frame, "During capture, you will be asked to:", (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, "- Turn your head SIDEWAYS (left & right)", (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame, "- Look UP and DOWN", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame, "- Move at different angles", (40, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame, "Keep the SAME facial expression!", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

            cv2.putText(frame, "Press 's' to START", (40, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Baseline Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        for t in [3, 2, 1]:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, f"{emotion.upper()} starting in {t}",
                        (w//2 - 200, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        CAPTURE_COLORS[emotion], 3)

            cv2.putText(frame, "Get ready to move your head",
                        (w//2 - 220, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)

            cv2.imshow("Baseline Capture", frame)
            cv2.waitKey(1000)

        capture_delay = 0.3
        samples_needed = 100
        count = 0
        last_capture = 0

        while count < samples_needed:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            now = cv2.getTickCount() / cv2.getTickFrequency()

            # Get current movement instruction
            instruction, instruction_color = get_head_movement_instruction(count, samples_needed)

            if now - last_capture < capture_delay:
                # Draw instruction box at top
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, 20), (w-20, 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Main instruction
                cv2.putText(frame, instruction, (40, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, instruction_color, 2)

                # Progress counter
                cv2.putText(frame, f"Samples: {count}/{samples_needed}", (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, CAPTURE_COLORS[emotion], 2)

                # Progress bar
                bar_width = int((count / samples_needed) * (w - 80))
                cv2.rectangle(frame, (40, 150), (w-40, 180), (100, 100, 100), 2)
                if bar_width > 0:
                    cv2.rectangle(frame, (40, 150), (40 + bar_width, 180),
                                  CAPTURE_COLORS[emotion], -1)

                cv2.imshow("Baseline Capture", frame)
                cv2.waitKey(1)
                continue

            box = detect_face_haar(frame)
            if box is None:
                # Warning box
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, 20), (w-20, 150), (0, 0, 50), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame, "No face detected!", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.putText(frame, instruction, (40, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, instruction_color, 2)

                cv2.imshow("Baseline Capture", frame)
                cv2.waitKey(1)
                continue

            x, y, w_box, h_box = box
            face_crop = frame[y:y+h_box, x:x+w_box]
            if face_crop.size == 0:
                continue

            emb = get_embedding(face_crop, model)
            embeddings.append(emb)
            count += 1
            last_capture = now

            # Draw face detection box
            cv2.rectangle(frame, (x, y), (x+w_box, y+h_box),
                          CAPTURE_COLORS[emotion], 3)

            cv2.putText(frame, f"{emotion.upper()} {count}/{samples_needed}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        CAPTURE_COLORS[emotion], 2)

            # Draw instruction box at top with success background
            overlay = frame.copy()
            cv2.rectangle(overlay, (20, 20), (w-20, 200), (0, 50, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, instruction, (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, instruction_color, 2)

            cv2.putText(frame, "✓ CAPTURED", (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Progress bar
            bar_width = int((count / samples_needed) * (w - 80))
            cv2.rectangle(frame, (40, 150), (w-40, 180), (100, 100, 100), 2)
            cv2.rectangle(frame, (40, 150), (40 + bar_width, 180),
                          CAPTURE_COLORS[emotion], -1)

            # Percentage
            cv2.putText(frame, f"{int((count/samples_needed)*100)}%",
                        (w-120, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.imshow("Baseline Capture", frame)
            cv2.waitKey(1)

        torch.save(torch.stack(embeddings, dim=0),
                   f"{emotion_dir}/coreset.pt")

        print(f"Saved {samples_needed} samples for {emotion}")

    cap.release()
    cv2.destroyAllWindows()
    print("Baseline complete.")
