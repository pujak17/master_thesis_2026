import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Face Mesh Models
mp_face_mesh = mp.solutions.face_mesh


# Simple emotion detection logic (you can improve this with a real ML model later)
def detect_emotion(landmarks):
    emotion = "neutral"

    # Smile detection: distance between corners of the mouth (landmarks 61 & 291)
    if landmarks[61][0] - landmarks[291][0] > 0.1:
        emotion = "happy"
    # Frown detection: vertical difference between upper and lower lips (landmarks 62 & 66)
    elif landmarks[62][1] - landmarks[66][1] < -0.1:
        emotion = "frustrated"

    return emotion


# Frame processing function
def process_frame(frame):
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Convert MediaPipe landmarks to list of (x, y) coordinates
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

                # Emotion detection
                emotion = detect_emotion(landmarks)
                print(f"Detected Emotion: {emotion}")

                # Draw face mesh
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION
                )

    return frame


# Main loop
def run():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
