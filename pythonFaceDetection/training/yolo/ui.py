import cv2
import os


def ask_username():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return None

    username = ""
    confirmed = False

    while not confirmed:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, "Enter Username:", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, username + "_", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 200), 2)
        cv2.putText(frame, "Press ENTER to confirm", (40, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press BACKSPACE to delete", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit", (40, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("User Login", frame)
        cv2.setWindowProperty("User Login", cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(1)

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return None

        if key == 13:
            if len(username.strip()) > 0:
                confirmed = True
            continue

        if key == 8:
            if len(username) > 0:
                username = username[:-1]
            continue

        if 32 <= key <= 126:
            username += chr(key)

    cap.release()
    cv2.destroyAllWindows()
    return username.strip()


def ask_existing_user_action(user_dir):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return "use"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, "Baseline Found", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "R = Replace baseline", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, "A = Add more samples", (40, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
        cv2.putText(frame, "U = Use existing baseline", (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(frame, "Click this window first!", (40, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("User Options", frame)
        cv2.setWindowProperty("User Options", cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            cap.release()
            cv2.destroyAllWindows()
            return "replace"
        elif key == ord("a"):
            cap.release()
            cv2.destroyAllWindows()
            return "add"
        elif key == ord("u"):
            cap.release()
            cv2.destroyAllWindows()
            return "use"

    cap.release()
    cv2.destroyAllWindows()
    return "use"