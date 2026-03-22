import mediapipe as mp
import numpy as np
import math
import time
import cv2
import os
import json
from collections import deque

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1)

POSTURE_THRESHOLD = 2.0 / 3.0
MIN_LANDMARK_VIS = 0.55
SMOOTH_N = 10
MOTION_HIST_LEN = 30
MIN_MOTION_SAMPLES = 6
STABLE_MOTION_THRESH = 3.0
BASELINE_FRAMES = 60

_motion_hist = deque(maxlen=MOTION_HIST_LEN)
_last_feats = None
_feat_hist = {}


def angle_deg(a, b, c):
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)

    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(math.degrees(math.acos(cosang)))


def _vis(landmarks, idx):
    return float(getattr(landmarks[idx], "visibility", 1.0))


def _confidence_ok(landmarks):
    return (_vis(landmarks, 11) >= MIN_LANDMARK_VIS and
            _vis(landmarks, 12) >= MIN_LANDMARK_VIS)


def extract_posture_features(landmarks, w, h):
    def lm(i):
        p = landmarks[i]
        return np.array([p.x * w, p.y * h, p.z], dtype=np.float32)

    nose = lm(0)
    l_sh = lm(11)
    r_sh = lm(12)
    l_hip = lm(23)
    r_hip = lm(24)

    mid_sh = (l_sh + r_sh) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0

    above = np.array([mid_sh[0], mid_sh[1] - 100.0, mid_sh[2]], dtype=np.float32)
    spine_ang = angle_deg(above, mid_sh, mid_hip)
    head_ang = angle_deg(l_sh, mid_sh, nose)
    shoulder_diff = float(abs(l_sh[1] - r_sh[1]))
    head_drop = float(nose[1] - mid_sh[1])
    forward_head_z = float(nose[2] - mid_sh[2])

    shoulder_width = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    torso_len = float(np.linalg.norm(mid_sh[:2] - mid_hip[:2])) + 1e-6
    shoulder_width_norm = float(shoulder_width / torso_len)

    return {
        "spine_angle": float(spine_ang),
        "head_angle": float(head_ang),
        "shoulder_diff": float(shoulder_diff),
        "head_drop": float(head_drop),
        "forward_head_z": float(forward_head_z),
        "shoulder_width_norm": float(shoulder_width_norm),
        "motion": 0.0,
    }


def _update_motion(feats):
    global _last_feats, _motion_hist

    if _last_feats is not None:
        dv = 0.0
        for k in ["spine_angle", "head_angle", "head_drop", "shoulder_width_norm"]:
            dv += abs(feats[k] - _last_feats[k])
        _motion_hist.append(float(dv))

    _last_feats = feats
    feats["motion"] = float(np.mean(_motion_hist)) if len(_motion_hist) >= MIN_MOTION_SAMPLES else 0.0
    return feats


def _smooth_feats(feats):
    global _feat_hist
    for k, v in feats.items():
        if k not in _feat_hist:
            _feat_hist[k] = deque(maxlen=SMOOTH_N)
        _feat_hist[k].append(float(v))

    smoothed = {}
    for k, q in _feat_hist.items():
        smoothed[k] = float(np.mean(q)) if len(q) >= 2 else float(q[-1])
    return smoothed


def _z(baseline, k, x):
    mu = baseline[k]["mean"]
    sd = baseline[k]["std"]
    return abs((x - mu) / sd)


def _draw_baseline_ui(frame, countdown_s, accepted, skipped, target, conf_ok, motion_ok):
    status = "Tracking OK" if conf_ok else "Adjust position (shoulders not visible)"
    status_color = (0, 255, 0) if conf_ok else (0, 0, 255)

    motion_status = "Still" if motion_ok else "Move less — stay still"
    motion_color = (0, 255, 0) if motion_ok else (0, 165, 255)

    cv2.putText(frame, "POSTURE BASELINE", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if countdown_s > 0:
        cv2.putText(frame, f"Starting in {countdown_s}...", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Sit normally, keep shoulders visible, stay still.", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Collecting... sit still (~30s)", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(frame, status, (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Motion: {motion_status}", (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, motion_color, 2)
    cv2.putText(frame, f"Accepted: {accepted}/{target}", (30, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Skipped: {skipped}", (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "Press 'q' to cancel", (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def _compute_stats_robust(collected):
    keys = list(collected[0].keys())
    stats = {}

    for key in keys:
        arr = np.array([f[key] for f in collected], dtype=np.float32)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        mask = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
        arr_clean = arr[mask] if mask.sum() >= 10 else arr

        median_val = float(np.median(arr_clean))
        std_val = float(arr_clean.std() + 1e-6)
        cv_val = float(std_val / (abs(median_val) + 1e-6))

        stats[key] = {
            "mean": median_val,
            "std": std_val,
            "cv": cv_val,
        }

    unstable = [k for k in ["spine_angle", "head_angle"] if stats[k]["cv"] > 0.15]
    if unstable:
        print(f"[Baseline Warning] Unstable features detected: {unstable}. "
              f"Consider re-collecting — person may have moved during calibration.")

    return stats


def save_posture_baseline(path, baseline_stats, posture_threshold):
    payload = {
        "baseline_stats": baseline_stats,
        "posture_threshold": float(posture_threshold),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_posture_baseline(path):
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("baseline_stats"), payload.get("posture_threshold")


def collect_posture_baseline(frames=BASELINE_FRAMES, show_ui=True, countdown=3,
                             window_name="Posture Baseline"):
    global _last_feats, _motion_hist, _feat_hist
    _last_feats = None
    _motion_hist.clear()
    _feat_hist = {}

    cap = cv2.VideoCapture(0)
    collected = []
    accepted = 0
    skipped = 0

    countdown_end = time.time() + max(0, int(countdown))

    while accepted < frames:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_detector.process(rgb)

        conf_ok = False
        if res.pose_landmarks is not None:
            landmarks = res.pose_landmarks.landmark
            conf_ok = _confidence_ok(landmarks)

        remaining = int(max(0, round(countdown_end - time.time())))
        collecting_now = (remaining == 0)

        motion_ok = (len(_motion_hist) < MIN_MOTION_SAMPLES or
                     float(np.mean(_motion_hist)) <= STABLE_MOTION_THRESH)

        if show_ui:
            ui_frame = frame.copy()
            _draw_baseline_ui(ui_frame, remaining, accepted, skipped, frames, conf_ok, motion_ok)
            cv2.imshow(window_name, ui_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyWindow(window_name)
                return None, 1.0

        if not collecting_now:
            time.sleep(0.01)
            continue

        if res.pose_landmarks is None or not conf_ok:
            skipped += 1
            time.sleep(0.01)
            continue

        h, w, _ = frame.shape
        feats = extract_posture_features(res.pose_landmarks.landmark, w, h)
        feats = _update_motion(feats)
        feats = _smooth_feats(feats)

        if len(_motion_hist) >= MIN_MOTION_SAMPLES and feats["motion"] > STABLE_MOTION_THRESH:
            skipped += 1
            time.sleep(0.01)
            continue

        collected.append(feats)
        accepted += 1
        time.sleep(0.03)

    cap.release()
    if show_ui:
        cv2.destroyWindow(window_name)

    if len(collected) < 10:
        print("[Baseline Error] Not enough clean frames collected. Re-run baseline.")
        return None, 1.0

    stats = _compute_stats_robust(collected)
    return stats, POSTURE_THRESHOLD


def compute_posture_from_frame(frame, baseline):
    if baseline is None:
        return 0.0, {"confidence_ok": False}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose_detector.process(rgb)
    if res.pose_landmarks is None:
        return 0.0, {"confidence_ok": False}

    landmarks = res.pose_landmarks.landmark
    conf_ok = _confidence_ok(landmarks)
    if not conf_ok:
        return 0.0, {"confidence_ok": False}

    h, w, _ = frame.shape
    feats = extract_posture_features(landmarks, w, h)
    feats = _update_motion(feats)
    feats = _smooth_feats(feats)

    weights = {
        "spine_angle": 0.35,
        "head_angle": 0.20,
        "shoulder_diff": 0.05,
        "head_drop": 0.15,
        "forward_head_z": 0.05,
        "shoulder_width_norm": 0.10,
        "motion": 0.10,
    }

    zsum = 0.0
    wsum = 0.0
    cues = {}

    for k, wk in weights.items():
        z = _z(baseline, k, feats[k])
        z01 = min(z / 3.0, 1.0)
        zsum += wk * z01
        wsum += wk
        cues[k] = float(np.clip(z01, 0.0, 1.0))

    overall = float(np.clip(zsum / (wsum + 1e-6), 0.0, 1.0))
    cues["confidence_ok"] = True
    cues["overall"] = overall
    return overall, cues
