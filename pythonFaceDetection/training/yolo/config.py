from collections import deque

conn = None  # global connection for VSM

# Parameters for sustained anger detection
ANGER_THRESHOLD = 0.5       # probability threshold for "angry"
WINDOW_DURATION = 5         # seconds for sliding window
DENSITY_THRESHOLD = 0.6     # fraction of angry frames required

# Sliding window buffer
anger_window = deque()
last_intervention_time = 0

# Mode: "prob" uses probability threshold, "label" uses top predicted label
ANGER_MODE = "prob"   # change to "label" if you want trigger by top-class