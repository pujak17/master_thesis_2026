from collections import deque


class GazeAgentConfig:
    """
    Configuration for virtual agent placement and gaze detection.
    """

    def __init__(
            self,
            agent_side: str = "right",   # "left" or "right" from user's perspective
            offset_threshold: float = 0.05,  # was 0.15; normalized [-1,1] horizontal offset
            min_frames: int = 6,           # was 10; window size for gaze smoothing
            min_ratio: float = 0.5         # was 0.7; fraction of frames in window that must look at agent
    ):
        if agent_side not in ("left", "right"):
            raise ValueError("agent_side must be 'left' or 'right'")
        self.agent_side = agent_side
        self.offset_threshold = offset_threshold
        self.min_frames = min_frames
        self.min_ratio = min_ratio


class GazeAgentTracker:
    """
    Tracks whether the user is looking towards the virtual agent,
    based on rough head/face position in the frame.
    """

    def __init__(self, config: GazeAgentConfig):
        self.config = config
        self._hits = deque(maxlen=config.min_frames)
        self._last_offset = 0.0

    def update_from_face_box(self, frame_width: int, frame_height: int,
                             x: int, y: int, w_box: int, h_box: int) -> bool:
        """
        Update gaze state from face bounding box and return whether
        current window indicates the user is looking at the agent.
        """
        frame_cx = frame_width / 2.0
        cx = x + w_box / 2.0

        # Normalized horizontal offset: -1 (far left) .. 1 (far right)
        offset = (cx - frame_cx) / frame_cx
        self._last_offset = offset

        if self.config.agent_side == "right":
            looking_at_agent = offset > self.config.offset_threshold
        else:
            looking_at_agent = offset < -self.config.offset_threshold

        self._hits.append(1 if looking_at_agent else 0)

        ready = (
                len(self._hits) == self.config.min_frames
                and sum(self._hits) >= int(self.config.min_ratio * self.config.min_frames)
        )

        return ready

    @property
    def last_offset(self) -> float:
        return self._last_offset

    @property
    def window_fill(self) -> int:
        return len(self._hits)

    @property
    def window_hits(self) -> int:
        return sum(self._hits)
