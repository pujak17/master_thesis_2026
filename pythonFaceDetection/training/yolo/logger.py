"""
logger.py
---------
Handles all experiment logging for one participant session.

Creates two CSV files in <save_dir>/<username>/:
  frame_scores.csv   — one row per inference frame (YOLO scores, thresholds,
                       confidence, which intervention fired if any)
  task_events.csv    — one row per browser UI event (from script.js via Flask)

The two files share a Unix timestamp column so you can JOIN them in pandas
to see exactly which task was on screen when each intervention fired.

USAGE IN main.py
----------------
    from logger import ExperimentLogger

    exp_logger = ExperimentLogger(args.save_dir, args.username)
    exp_logger.start_server()          # starts Flask receiver on port 5003

    # Inside your per-frame inference loop:
    exp_logger.log_frame(
        scores=scores,
        frustration_score=frustration_prob,
        anger_th=anger_th,
        happy_th=happy_th,
    )

    # When an intervention fires, call log_intervention THEN safe_send:
    exp_logger.log_intervention("frustrated", scores, frustration_prob, anger_th, happy_th)
    safe_send(b"INTERVENTION:frustrated\n")

    # Check if user is stuck on a task (returns -1.0 if no task active):
    stuck_secs = exp_logger.task_stuck_seconds()

    # In your finally / cleanup block:
    exp_logger.close()

REQUIREMENTS
------------
    pip install flask flask-cors
"""
from __future__ import annotations

import csv
import os
import threading
import time
import logging as _logging

from flask import Flask, request, jsonify
from flask_cors import CORS


class ExperimentLogger:

    FRAME_HEADER = [
        "timestamp",
        "happy", "angry", "sad", "fear", "surprise", "neutral", "disgusted",
        "frustration_score",
        "above_anger_th",      # 1 / 0
        "above_happy_th",      # 1 / 0
        "confidence",          # "high" / "low" / "none"
        "intervention_fired",  # "" or "frustrated" / "smile" / "look" / "task_stuck"
    ]

    EVENT_HEADER = [
        "timestamp",
        "event",
        "detail",
        "task_index",
        "task_title",
        "task_type",
        "elapsed_seconds",
    ]

    def __init__(self, save_dir: str, username: str):
        self.save_dir = save_dir
        self.username = username

        user_dir = os.path.join(save_dir, username)
        os.makedirs(user_dir, exist_ok=True)

        # ── Frame scores CSV ──────────────────────────────────────────────
        frame_path = os.path.join(user_dir, "frame_scores.csv")
        self._frame_file   = open(frame_path, "w", newline="", encoding="utf-8")
        self._frame_writer = csv.writer(self._frame_file)
        self._frame_writer.writerow(self.FRAME_HEADER)
        self._frame_file.flush()

        # ── Task events CSV ───────────────────────────────────────────────
        event_path = os.path.join(user_dir, "task_events.csv")
        self._event_file   = open(event_path, "w", newline="", encoding="utf-8")
        self._event_writer = csv.writer(self._event_file)
        self._event_writer.writerow(self.EVENT_HEADER)
        self._event_file.flush()

        # Thread lock — Flask thread and main thread both write
        self._lock        = threading.Lock()
        self._frame_count = 0

        # ── Task stuck tracking ───────────────────────────────────────────
        # Set by the Flask endpoint when script.js sends task_load /
        # task_complete events.  Read every frame from main.py inference loop.
        self.current_task_start: float | None = None
        self.current_task_title: str          = ""
        self.current_task_index: str          = ""

        print(f"[Logger] Writing logs to: {user_dir}/")

    # ─────────────────────────────────────────────────────────────────────
    # Task stuck helpers — read from main.py, updated by Flask thread
    # ─────────────────────────────────────────────────────────────────────

    def task_stuck_seconds(self) -> float:
        """
        Seconds since the current task loaded.
        Returns -1.0 if no task is active (between tasks or not started yet).
        Single attribute read is atomic in CPython — no lock needed.
        """
        start = self.current_task_start
        if start is None:
            return -1.0
        return time.time() - start

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def log_frame(
            self,
            scores: dict,
            frustration_score: float,
            anger_th: float,
            happy_th: float,
            intervention: str = "",
    ):
        """Call once per inference frame from your main loop."""
        above_anger = frustration_score > anger_th
        above_happy = scores.get("happy", 0) > happy_th

        if above_anger:
            margin     = frustration_score - anger_th
            confidence = "high" if margin > (anger_th * 0.3) else "low"
        elif above_happy:
            margin     = scores.get("happy", 0) - happy_th
            confidence = "high" if margin > (happy_th * 0.3) else "low"
        else:
            confidence = "none"

        row = [
            round(time.time(), 3),
            round(scores.get("happy",     0), 4),
            round(scores.get("angry",     0), 4),
            round(scores.get("sad",       0), 4),
            round(scores.get("fear",      0), 4),
            round(scores.get("surprise",  0), 4),
            round(scores.get("neutral",   0), 4),
            round(scores.get("disgusted", 0), 4),
            round(frustration_score,          4),
            int(above_anger),
            int(above_happy),
            confidence,
            intervention,
        ]

        with self._lock:
            self._frame_writer.writerow(row)
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                self._frame_file.flush()

    def log_intervention(
            self,
            intervention_type: str,
            scores: dict,
            frustration_score: float,
            anger_th: float,
            happy_th: float,
            onset_lag: float = -1.0,
    ):
        """
        Call this when any intervention fires.
        Writes a marked frame row AND a dedicated event row in task_events.csv.
        onset_lag: seconds from emotion onset to now (-1 if not applicable).
        """
        self.log_frame(
            scores=scores,
            frustration_score=frustration_score,
            anger_th=anger_th,
            happy_th=happy_th,
            intervention=intervention_type,
        )

        stuck = self.task_stuck_seconds()
        self._write_event({
            "timestamp": round(time.time(), 3),
            "event":     f"INTERVENTION:{intervention_type}",
            "detail":    (
                    f"frustration={frustration_score:.3f}"
                    f"_th={anger_th:.3f}"
                    f"_onset_lag={onset_lag:.1f}s"
                    + (f"_stuck={stuck:.0f}s" if stuck >= 0 else "")
            ),
            "task_index":      self.current_task_index,
            "task_title":      self.current_task_title,
            "task_type":       "",
            "elapsed_seconds": f"{stuck:.1f}" if stuck >= 0 else "",
        })

    def close(self):
        """Flush and close both CSV files. Call in your finally block."""
        with self._lock:
            self._frame_file.flush()
            self._frame_file.close()
            self._event_file.flush()
            self._event_file.close()
        print("[Logger] Files closed.")

    # ─────────────────────────────────────────────────────────────────────
    # Flask server — receives events from script.js
    # ─────────────────────────────────────────────────────────────────────

    def start_server(self, port: int = 5003):
        """
        Starts a background Flask receiver on port 5003.
        script.js POSTs JSON to http://127.0.0.1:5003/task_event on every
        UI action (task load, form submit, error shown, retry clicked, etc.).
        """
        app        = Flask(__name__)
        CORS(app)
        logger_ref = self

        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        @app.route("/task_event", methods=["POST"])
        def task_event():
            try:
                data  = request.get_json(force=True, silent=True) or {}
                event = data.get("event", "")

                # ── Update task stuck state ───────────────────────────────
                if event == "task_loaded":
                    logger_ref.current_task_start = time.time()
                    logger_ref.current_task_title = data.get("task_title", "")
                    logger_ref.current_task_index = str(data.get("task_index", ""))
                    print(
                        f"[Logger] Task loaded: '{logger_ref.current_task_title}' "
                        f"(index {logger_ref.current_task_index})"
                    )
                elif event in ("task_step_completed", "experiment_complete"):
                    logger_ref.current_task_start = None
                    logger_ref.current_task_title = ""
                    logger_ref.current_task_index = ""

                logger_ref._write_event(data)
                return jsonify({"ok": True})
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        @app.route("/ping", methods=["GET"])
        def ping():
            return jsonify({
                "status":     "ok",
                "user":       logger_ref.username,
                "task":       logger_ref.current_task_title,
                "stuck_secs": round(logger_ref.task_stuck_seconds(), 1),
            })

        def _run():
            app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

        t = threading.Thread(target=_run, daemon=True, name="TaskEventServer")
        t.start()
        print(f"[Logger] Task-event server on http://127.0.0.1:{port}/task_event")
        print(f"[Logger] Health check:      http://127.0.0.1:{port}/ping")

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def _write_event(self, data: dict):
        row = [
            data.get("timestamp",       round(time.time(), 3)),
            data.get("event",           ""),
            data.get("detail",          ""),
            data.get("task_index",      ""),
            data.get("task_title",      ""),
            data.get("task_type",       ""),
            data.get("elapsed_seconds", ""),
        ]
        with self._lock:
            self._event_writer.writerow(row)
            self._event_file.flush()