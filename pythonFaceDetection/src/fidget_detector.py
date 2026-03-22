# detectors/fidget_detector.py

import numpy as np


class FidgetDetector:
    def __init__(self, movement_threshold=0.02, window_size=10):
        self.prev_positions = []
        self.threshold = movement_threshold
        self.window_size = window_size

    def update(self, left_shoulder, right_shoulder):
        # Calculate center between shoulders
        current_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])

        # Save position history
        self.prev_positions.append(current_center)
        if len(self.prev_positions) > self.window_size:
            self.prev_positions.pop(0)

        # Compute average movement
        if len(self.prev_positions) > 1:
            diffs = [
                np.linalg.norm(self.prev_positions[i] - self.prev_positions[i - 1])
                for i in range(1, len(self.prev_positions))
            ]
            avg_movement = np.mean(diffs)
            return avg_movement > self.threshold

        return False
