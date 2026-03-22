import cv2
import math
from collections import deque

import mediapipe as mp  # pip install mediapipe


class GazeFusionConfig:
    def __init__(
            self,
            agent_side="right",        # "left" or "right" as user sees it
            mirrored=True,             # webcam preview mirrored like selfie?
            head_offset_th=0.15,       # threshold on fused offset
            min_frames=1,              
            min_ratio=1.0
    ):
        self.agent_side = agent_side
        self.mirrored = mirrored
        self.head_offset_th = head_offset_th
        self.min_frames = min_frames
        self.min_ratio = min_ratio


class GazeFusionTracker:
    def __init__(self, cfg: GazeFusionConfig):
        self.cfg = cfg

        # Map user side to image side based on mirroring
        if cfg.mirrored:
            self.image_agent_side = "left" if cfg.agent_side == "right" else "right"
        else:
            self.image_agent_side = cfg.agent_side

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,    # iris landmarks enabled
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # window now just for debugging/inspection
        self.win = deque(maxlen=cfg.min_frames)
        self.last_ready = False
        self.last_offset = 0.0  # fused horizontal offset [-1..1]

    def _get_head_offset(self, frame_w, face_box):
        """Normalized head offset: -1 = far left, +1 = far right."""
        x, y, w, h = face_box
        cx = x + w / 2.0
        return (cx - frame_w / 2.0) / (frame_w / 2.0)

    def _get_eye_offset(self, landmarks):
        """
        Horizontal iris offset averaged over both eyes, normalized [-1..1].
        Uses MediaPipe indices: left eye (33, 133, 468), right eye (362, 263, 473).
        """

        def norm_x(idx):
            return landmarks[idx].x  # normalized [0..1]

        # left eye
        lx_outer = norm_x(33)
        lx_inner = norm_x(133)
        lx_iris = norm_x(468)
        left_center = 0.5 * (lx_outer + lx_inner)
        left_radius = 0.5 * abs(lx_outer - lx_inner) + 1e-6
        left_offset = (lx_iris - left_center) / left_radius

        # right eye
        rx_outer = norm_x(362)
        rx_inner = norm_x(263)
        rx_iris = norm_x(473)
        right_center = 0.5 * (rx_outer + rx_inner)
        right_radius = 0.5 * abs(rx_outer - rx_inner) + 1e-6
        right_offset = (rx_iris - right_center) / right_radius

        return 0.5 * (left_offset + right_offset)

    def update(self, frame_bgr, face_box):
        """
        Update gaze state given full frame and Haar face_box (x, y, w, h).
        Returns (ready, fused_offset).

        ready becomes True immediately on any frame where the user looks
        toward the virtual agent side beyond head_offset_th.
        """
        h, w = frame_bgr.shape[:2]

        head_off = self._get_head_offset(w, face_box)  # [-1..1]

        x, y, fw, fh = face_box
        x2, y2 = x + fw, y + fh
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x2), min(h, y2)
        face = frame_bgr[y:y2, x:x2]
        if face.size == 0:
            self.last_offset = head_off
            self.last_ready = False
            return False, self.last_offset

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_rgb)

        if not results.multi_face_landmarks:
            self.last_offset = head_off
            self.last_ready = False
            return False, self.last_offset

        lms = results.multi_face_landmarks[0].landmark
        try:
            eye_off = self._get_eye_offset(lms)
        except IndexError:
            eye_off = 0.0

        # Simple fusion: average head + eye
        fused_off = 0.5 * (head_off + eye_off)
        self.last_offset = fused_off

        if self.image_agent_side == "right":
            looking = fused_off > self.cfg.head_offset_th
        else:
            looking = fused_off < -self.cfg.head_offset_th

        # Just for debugging / optional plotting of recent states
        self.win.append(looking)

        self.last_ready = looking
        return looking, fused_off
