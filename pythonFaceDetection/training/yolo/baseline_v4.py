"""
baseline_v4.py

This module handles user-specific calibration for the emotion detector.

It records a short neutral baseline for each user, measures how much the
emotion scores normally vary around that neutral state, and then builds
personal thresholds for happy and angry.

The thresholds are based on:
- the median neutral score for each label
- the typical upward and downward variation around that median
- a sensitivity multiplier K

Higher K values make a trigger harder to fire.
"""

import os
import json
import cv2
import numpy as np
from ultralytics import YOLO


K_HAPPY = 2.0
K_ANGRY = 3.0

DEFAULT_K = 2.0

MAX_THRESHOLD = 0.95
MIN_THRESHOLD = 0.05
NUM_FRAMES = 100
MEDIAN_FILTER_WIN = 5

THRESHOLDS_FILENAME = "thresholds.json"

_K_PER_LABEL = {
    "happy": K_HAPPY,
    "angry": K_ANGRY,
}


def detect_face_haar(frame):
    """Return the largest detected face as (x, y, w, h), or None if no face is found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces.tolist(), key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def _scores_from_result(result):
    """Convert a YOLO classification result into a {label: probability} dictionary."""
    if result.probs is None:
        return {}
    probs = result.probs.data.detach().cpu().numpy().astype(float)
    return {result.names[i]: float(probs[i]) for i in range(len(probs))}


def _rmse(values):
    """Compute root-mean-square relative to zero for a 1-D list or array."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def thresholds_path(save_dir: str, username: str) -> str:
    """Return the file path used to store a user's thresholds."""
    return os.path.join(save_dir, username, THRESHOLDS_FILENAME)


class MedianFilter:
    """
    Keeps a short rolling history of recent values and returns the median.

    Older values are dropped after `max_age_s` so that gaps in face detection
    do not linger in the buffer.
    """

    def __init__(self, window: int = MEDIAN_FILTER_WIN, max_age_s: float = 1.0):
        self.window = window
        self.max_age_s = max_age_s
        self._buf: list[tuple[float, float]] = []

    def update(self, value: float, timestamp: float) -> float:
        self._buf = [(t, v) for t, v in self._buf if (timestamp - t) <= self.max_age_s]
        self._buf.append((timestamp, value))
        self._buf = self._buf[-self.window:]
        vals = [v for _, v in self._buf]
        return float(np.median(vals))

    def reset(self):
        """Clear the filter history."""
        self._buf.clear()


def collect_neutral_frames(args, num_frames: int = NUM_FRAMES):
    """
    Open the webcam and collect neutral baseline frames.

    Frames are saved under:
        <save_dir>/<username>/neutral/frames/

    If that folder already contains images, new frames are appended.
    """
    frames_dir = os.path.join(args.save_dir, args.username, "neutral", "frames")
    os.makedirs(frames_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    start_idx = len(existing)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    print(f"\nCapturing {num_frames} neutral frames (appending after {start_idx} existing) …")

    window = "Baseline Capture"

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (w - 20, 340), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(
            frame, "NEUTRAL BASELINE CAPTURE", (40, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2
        )
        cv2.putText(
            frame, "Relax your face. No expression.", (40, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, "Look straight at the camera.", (40, 155),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, "Press  's'  to START  |  'q'  to quit", (40, 300),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow(window, frame)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    count = start_idx
    target = start_idx + num_frames

    while count < target:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        done = count - start_idx
        h, w = frame.shape[:2]
        progress = done / num_frames
        bar_w = int(progress * (w - 80))

        cv2.putText(
            frame, f"Capturing  {done}/{num_frames}",
            (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.rectangle(frame, (40, h - 50), (w - 40, h - 20), (80, 80, 80), 2)
        if bar_w > 0:
            cv2.rectangle(
                frame, (40, h - 50), (40 + bar_w, h - 20), (100, 200, 255), -1
            )

        cv2.imshow(window, frame)
        cv2.waitKey(1)

        path = os.path.join(frames_dir, f"neutral_{count:04d}.jpg")
        cv2.imwrite(path, frame)
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"✓  Saved {num_frames} frames to  {frames_dir}\n")


def compute_thresholds(args, show_ui: bool = True):
    """
    Build personal thresholds from the saved neutral frames.

    For each label:
    1. compute the user's median neutral score
    2. measure how much the score naturally drifts above and below that median
    3. convert that drift into upper and lower thresholds

    The upper threshold is used for triggering.
    The lower threshold is saved as well in case it is useful later.
    """
    print("\n=== COMPUTING PERSONAL THRESHOLDS ===")
    print(f"    K_happy = {K_HAPPY}")
    print(f"    K_angry = {K_ANGRY}")

    yolo = YOLO(args.emotion_classifier)
    baseline_dir = os.path.join(args.save_dir, args.username, "neutral", "frames")

    if not os.path.exists(baseline_dir):
        print("⚠  No baseline directory – run collect_neutral_frames first.")
        return _default_thresholds()

    frame_files = sorted(
        f for f in os.listdir(baseline_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_files:
        print("⚠  No baseline images found.")
        return _default_thresholds()

    happy_vals, angry_vals = [], []
    faces_found = faces_used = 0
    window_name = "Computing Thresholds …"

    for idx, fname in enumerate(frame_files, start=1):
        img = cv2.imread(os.path.join(baseline_dir, fname))
        if img is None:
            continue

        box = detect_face_haar(img)
        if box is None:
            _show_progress(
                show_ui, window_name, img, idx, len(frame_files),
                faces_found, faces_used, face_found=False
            )
            continue

        faces_found += 1
        x, y, w, h = box
        face = img[y:y + h, x:x + w]
        if face.size == 0:
            continue

        scores = _scores_from_result(yolo(face, verbose=False)[0])

        used = False
        if "happy" in scores:
            happy_vals.append(scores["happy"])
            used = True
        if "angry" in scores:
            angry_vals.append(scores["angry"])
            used = True
        if used:
            faces_used += 1

        _show_progress(
            show_ui, window_name, img, idx, len(frame_files),
            faces_found, faces_used, face_found=True
        )

    if show_ui:
        cv2.destroyWindow(window_name)

    if not happy_vals or not angry_vals:
        print("WARNING: not enough emotion data – using defaults.")
        return _default_thresholds()

    results = {}
    for label, vals in [("happy", happy_vals), ("angry", angry_vals)]:
        K = _K_PER_LABEL[label]
        arr = np.array(vals, dtype=float)

        M = float(np.median(arr))
        residuals = arr - M

        pos = residuals[residuals >= 0]
        neg = residuals[residuals < 0]

        E_pos = _rmse(pos) if len(pos) > 0 else 0.05
        E_neg = _rmse(np.abs(neg)) if len(neg) > 0 else 0.05

        T_high = float(np.clip(M + E_pos * K, MIN_THRESHOLD, MAX_THRESHOLD))
        T_low = float(np.clip(M - E_neg * K, MIN_THRESHOLD, MAX_THRESHOLD))

        results[label] = {
            "median": round(M, 4),
            "E_pos": round(E_pos, 4),
            "E_neg": round(E_neg, 4),
            "T_high": round(T_high, 4),
            "T_low": round(T_low, 4),
            "K": K,
            "n": len(arr),
        }

    anger_th = results["angry"]["T_high"]
    frustration_th = anger_th
    happy_th = results["happy"]["T_high"]

    print("\n=== PERSONAL THRESHOLDS ===")
    for label, r in results.items():
        print(f"\n  {label.upper()}  (K={r['K']})")
        print(f"    neutral median : {r['median']:.4f}")
        print(f"    RMSE above     : {r['E_pos']:.4f}  →  T_high = {r['T_high']:.4f}")
        print(f"    RMSE below     : {r['E_neg']:.4f}  →  T_low  = {r['T_low']:.4f}")
    print(
        f"\n  Frames: {len(frame_files)} total | "
        f"{faces_found} with face | {faces_used} used for calibration\n"
    )

    save_path = thresholds_path(args.save_dir, args.username)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓  Thresholds saved → {save_path}\n")

    return anger_th, frustration_th, happy_th


def load_thresholds(args):
    """
    Load previously saved thresholds.

    If no threshold file exists yet, calibration is run automatically.
    """
    path = thresholds_path(args.save_dir, args.username)
    if not os.path.exists(path):
        print(f"No saved thresholds for '{args.username}' – running calibration …")
        collect_neutral_frames(args)
        return compute_thresholds(args)

    with open(path) as f:
        results = json.load(f)

    def _apply_k(label):
        r = results[label]
        K = r.get("K", _K_PER_LABEL.get(label, DEFAULT_K))
        return float(np.clip(
            r["median"] + r["E_pos"] * K,
            MIN_THRESHOLD, MAX_THRESHOLD
        ))

    anger_th = _apply_k("angry")
    frustration_th = anger_th
    happy_th = _apply_k("happy")

    print(f"✓  Loaded thresholds for '{args.username}'")
    print(f"   happy={happy_th:.3f}  (K={results['happy'].get('K', '?')})")
    print(f"   anger={anger_th:.3f}  (K={results['angry'].get('K', '?')})\n")
    return anger_th, frustration_th, happy_th


class EmotionDetector:
    """
    Combines the YOLO classifier, personal thresholds, and median smoothing
    for live use.

    Example:
        detector = EmotionDetector(args)
        result = detector.update(frame, timestamp=time.time())
        if result["happy"]:
            print("User looks happy!")
    """

    def __init__(self, args):
        self.yolo = YOLO(args.emotion_classifier)

        anger_th, frustration_th, happy_th = load_thresholds(args)
        self.thresholds = {
            "happy": happy_th,
            "angry": anger_th,
            "frustration": frustration_th,
        }

        self.filters = {
            "happy": MedianFilter(window=MEDIAN_FILTER_WIN, max_age_s=1.0),
            "angry": MedianFilter(window=MEDIAN_FILTER_WIN, max_age_s=1.0),
        }

    def update(self, frame, timestamp: float) -> dict:
        box = detect_face_haar(frame)
        if box is None:
            return {
                "happy": False,
                "angry": False,
                "frustration": False,
                "scores": {},
                "face_found": False,
            }

        x, y, w, h = box
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            return {
                "happy": False,
                "angry": False,
                "frustration": False,
                "scores": {},
                "face_found": False,
            }

        raw_scores = _scores_from_result(self.yolo(face, verbose=False)[0])

        smoothed = {}
        for label, filt in self.filters.items():
            smoothed[label] = filt.update(raw_scores.get(label, 0.0), timestamp)

        return {
            "happy": smoothed.get("happy", 0.0) > self.thresholds["happy"],
            "angry": smoothed.get("angry", 0.0) > self.thresholds["angry"],
            "frustration": smoothed.get("angry", 0.0) > self.thresholds["frustration"],
            "scores": smoothed,
            "face_found": True,
        }

    def reset_filters(self):
        """Reset the live smoothing filters."""
        for f in self.filters.values():
            f.reset()


def _default_thresholds():
    return 1.0, 1.0, 1.0


def _show_progress(show_ui, window_name, img, idx, total,
                   faces_found, frames_used, face_found):
    if not show_ui:
        return

    frame = img.copy()
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (w - 20, 300), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame, "COMPUTING THRESHOLDS", (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2
    )
    cv2.putText(
        frame, f"K_happy={K_HAPPY}  K_angry={K_ANGRY}", (40, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
    )
    cv2.putText(
        frame, f"Frame {idx}/{total}", (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"Faces found : {faces_found}", (40, 165),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2
    )
    cv2.putText(
        frame, f"Frames used : {frames_used}", (40, 200),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2
    )

    status_col = (0, 255, 0) if face_found else (0, 0, 255)
    cv2.putText(
        frame, "Face OK" if face_found else "No face", (40, 235),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2
    )

    bar_w = int((idx / total) * (w - 80))
    cv2.rectangle(frame, (40, 255), (w - 40, 280), (80, 80, 80), 2)
    if bar_w > 0:
        cv2.rectangle(frame, (40, 255), (40 + bar_w, 280), (100, 200, 255), -1)

    cv2.putText(
        frame, f"{int(idx / total * 100)}%", (40, 278),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )

    cv2.imshow(window_name, frame)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)