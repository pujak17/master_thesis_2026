import cv2
import mediapipe as mp
import numpy as np

# Pose and FaceMesh setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils


def detect_emotion(landmarks):
    emotion = "neutral"

    # Get key points from the face landmarks
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_eyebrow = landmarks[53]
    right_eyebrow = landmarks[283]
    left_eye_inner = landmarks[133]
    right_eye_inner = landmarks[362]

    # Compute mouth width, mouth height, and eye distance
    mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
    mouth_height = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    # Normalize by eye distance to account for face size
    normalized_mouth_width = mouth_width / eye_distance
    normalized_mouth_height = mouth_height / eye_distance

    # Smile detection (normal smile)
    if normalized_mouth_width > 0.5 and 0.03 < normalized_mouth_height < 0.1:
        emotion = "happy"  # Normal smile with a smaller mouth width
    # Smile with teeth detection
    elif normalized_mouth_width > 0.6 and normalized_mouth_height > 0.05:
        emotion = "happy with teeth"
    # Frustration detection: Combining mouth height and eyebrow furrow
    elif normalized_mouth_height < 0.03 and (
            left_eyebrow[1] - right_eyebrow[1] > 0.06):  # More distinct eyebrow furrow for frustration
        emotion = "frustrated"
    # Sadness: Downward mouth corners and small mouth height
    elif landmarks[61][1] - landmarks[291][1] < -0.05 and normalized_mouth_height < 0.03:
        emotion = "sad"
    # Anger: Furrowed brows and narrow eyes
    elif left_eyebrow[1] - right_eyebrow[1] > 0.05:
        emotion = "angry"
    # Surprise: Wide eyes and raised eyebrows
    elif np.linalg.norm(np.array(left_eye_inner) - np.array(right_eye_inner)) > 0.15:
        emotion = "surprised"
    # Fear: Wide eyes and raised eyebrows
    elif np.linalg.norm(np.array(left_eye) - np.array(right_eye)) > 0.1:
        emotion = "fear"

    return emotion


def run_combined_analysis():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Movement detection
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_distance = np.linalg.norm(
                np.array([left_shoulder.x, left_shoulder.y]) -
                np.array([right_shoulder.x, right_shoulder.y])
            )
            threshold_value = 0.4
            movement_status = "Fidgeting" if shoulder_distance > threshold_value else "Calm"
        else:
            movement_status = "No pose detected"

        # Emotion detection
        emotion_results = face_mesh.process(rgb_frame)
        emotion_status = "No face detected"
        if emotion_results.multi_face_landmarks:
            for face_landmarks in emotion_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION
                )
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                emotion_status = detect_emotion(landmarks)
                break

        # Display status
        cv2.putText(frame, f"Movement: {movement_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion_status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)

        cv2.imshow("Real-Time Emotion & Movement Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
