"""Shared, UI-free PnP pose flows for the Kivy app and terminal runner."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from .head_pose import CalibrationError, LABELS


INSTRUCTIONS = {
    "FRONT": "Look straight at the camera",
    "LEFT": "Turn your head to YOUR LEFT",
    "RIGHT": "Turn your head to YOUR RIGHT",
    "UP": "Lift your chin and look UP",
    "DOWN": "Lower your chin and look DOWN",
}
CALIBRATION_KEYS = {"f": "FRONT", "l": "LEFT", "r": "RIGHT", "u": "UP", "d": "DOWN"}
REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_LOG_DIR = REPO_ROOT / "logs" / "head_pose" / "device_setup"


def pnp_profile_problem(
    config: dict, backend_name: str | None = None, resolution: tuple[int, int] | None = None
) -> str | None:
    """Return a participant-safe reason when a saved device profile cannot scan."""
    if backend_name is not None and backend_name != "yunet_geometry":
        return "PnP setup requires the YuNet geometry backend."
    if config.get("backend") not in ("auto", "yunet_geometry"):
        return "Device setup must use the YuNet PnP profile."
    if config.get("features") != {"yaw": "pnp_yaw", "pitch": "pnp_pitch"}:
        return "Device setup must be repeated with PnP pose features."
    if any(config.get("signs", {}).get(axis) not in (-1, 1) for axis in ("yaw", "pitch")):
        return "Device setup has not established pose directions."
    baseline = config.get("baseline", {})
    if not all(key in baseline and np.isfinite(baseline[key]) for key in ("pnp_yaw", "pnp_pitch")):
        return "Device setup needs a saved straight-ahead position."
    calibrated_at = config.get("calibration", {}).get("camera_resolution")
    if resolution is not None and calibrated_at is not None and list(resolution) != list(calibrated_at):
        return "Camera resolution changed; run Device Setup again."
    return None


class GuidedPoseFlow:
    """Temporary participant scan. ``on_confirm`` is optional and never persists data itself."""

    def __init__(self, tracker, *, on_confirm=None, allow_manual=False, completion_phase="complete"):
        self.tracker = tracker
        self.saved_config = deepcopy(tracker.config)
        self.on_confirm = on_confirm
        self.allow_manual = allow_manual
        self.completion_phase = completion_phase
        self.restart()

    def restart(self):
        self.tracker.config = deepcopy(self.saved_config)
        self.tracker.reset()
        self.samples = deque()
        self.phase = "ready"
        self.stage = 0
        self.since = None
        self.last_observation = None
        self._down_since = None
        self.progress = 0.0
        self.note = "Start when the participant is facing the camera."
        problem = pnp_profile_problem(self.tracker.config, self.tracker.backend.name)
        if problem:
            self.phase = "setup"
            self.note = problem

    def close(self):
        """Forget this visitor's neutral offset without changing the saved device profile."""
        self.tracker.config = deepcopy(self.saved_config)
        self.tracker.reset()

    def start(self, now: float):
        if self.phase == "ready":
            self.phase = "calibrating"
            self.since = now
            self.note = "Relax, look at the lens, and hold still for a moment."

    def _valid(self, pose) -> bool:
        c = self.tracker.config
        return (
            pose is not None
            and pose.raw.face_score >= c["min_confidence"]
            and abs(pose.raw.roll) <= c["roll_limit"]
            and all(key in pose.raw.features and np.isfinite(pose.raw.features[key]) for key in c["features"].values())
        )

    def _confirm(self, frame, now: float, *, manual=False):
        label = LABELS[self.stage]
        if self.on_confirm is not None:
            self.on_confirm(label, frame, manual)
        self.phase, self.since, self.progress = "feedback", now, 1.0
        self._down_since = None
        self.note = f"{label.capitalize()} verified" + (" manually" if manual else "")
        self.tracker.reset()

    def _lax_down_status(self, now: float):
        """Temporary scan-only DOWN hold; the saved profile itself is unchanged."""
        pose, config = self.tracker.latest, self.tracker.config
        required = config["stable_hold_s"]
        if pose is None or now - pose.timestamp_s > config["max_gap_s"]:
            self._down_since = None
            return False, 0.0
        limits = config.get("pitch_limits", {})
        down = limits.get("down", {"enter": config["pitch_enter"], "exit": config["pitch_exit"]})
        # ponytail: scan-only 75% entry; retain the fitted global threshold for every other consumer.
        threshold = max(down["exit"], down["enter"] * 0.75)
        matches = (
            pose.confidence >= config["min_confidence"]
            and abs(pose.yaw) <= config["vertical_yaw_guard"]
            and pose.pitch <= -threshold
        )
        if not matches:
            self._down_since = None
            return False, 0.0
        if self._down_since is None:
            self._down_since = pose.timestamp_s
        held = max(0.0, now - self._down_since)
        return held + 1e-9 >= required, min(1.0, held / required)

    def update(self, frame, pose, now: float):
        if self.phase in ("ready", "setup", self.completion_phase):
            return
        if self.phase == "feedback":
            if now - self.since >= 0.55:
                self.stage += 1
                self.phase = self.completion_phase if self.stage == len(LABELS) else "capture"
                self.since, self.progress = now, 0.0
                self._down_since = None
                self.tracker.reset()
                self.note = "All five positions were verified." if self.phase == self.completion_phase else "Move gently into the requested position."
            return
        if pose is not None:
            if pose.timestamp_s == self.last_observation:
                return
            self.last_observation = pose.timestamp_s
        if not self._valid(pose) or now - pose.timestamp_s > self.tracker.config["max_gap_s"]:
            self.samples.clear()
            self.progress = 0.0
            self.note = "Keep one face visible and upright."
            return
        if self.phase == "calibrating":
            if self.samples and now - self.samples[-1][0] > self.tracker.config["max_gap_s"]:
                self.samples.clear()
            self.samples.append((now, dict(pose.raw.features)))
            while self.samples and now - self.samples[0][0] > 1.6:
                self.samples.popleft()
            c = self.tracker.config
            for axis, allowance in (("yaw", 8.0), ("pitch", 6.0)):
                values = [sample[c["features"][axis]] * c["scales"][axis] for _, sample in self.samples]
                if float(np.percentile(values, 90) - np.percentile(values, 10)) > allowance:
                    self.samples.clear()
                    self.progress = 0.0
                    self.note = "Hold still while looking straight at the lens."
                    return
            elapsed = now - self.samples[0][0]
            self.progress = min(1.0, elapsed / 1.2)
            if elapsed + 1e-9 >= 1.2 and len(self.samples) >= 8:
                self.tracker.neutral_calibrate([sample for _, sample in self.samples])
                self._confirm(frame, now)
                self.samples.clear()
            return
        target = LABELS[self.stage]
        if target == "DOWN":
            satisfied, self.progress = self._lax_down_status(now)
        else:
            status = self.tracker.target_status(target, timestamp_s=now)
            satisfied, self.progress = status.satisfied, status.progress
        self.note = "Hold still..." if self.progress else "Move gently into the requested position."
        if satisfied:
            self._confirm(frame, now)

    def manual_available(self, now: float) -> bool:
        return self.allow_manual and self.phase == "capture" and now - self.since >= self.tracker.config["manual_timeout_s"]

    def manual_confirm(self, frame, pose, now: float):
        if self.manual_available(now) and self._valid(pose) and now - pose.timestamp_s <= self.tracker.config["max_gap_s"]:
            self._confirm(frame, now, manual=True)


class GuidedPoseCalibration:
    """Operator-only five-position PnP calibration with no images or logs."""

    settle_s = 1.0
    capture_s = 3.0

    def __init__(self, tracker, *, diagnostic_dir=SETUP_LOG_DIR):
        self.tracker = tracker
        self.original_config = deepcopy(tracker.config)
        self.diagnostic_dir = Path(diagnostic_dir)
        self.saved = False
        self.diagnostic_path = None
        self._logged = False
        self.restart()

    def restart(self):
        if hasattr(self, "samples") and any(self.samples.values()) and not self._logged:
            self._write_diagnostic("restarted")
        self.tracker.config = deepcopy(self.original_config)
        self.tracker.reset()
        self.samples = {label: [] for label in LABELS}
        self.phase = "ready"
        self.stage = 0
        self.since = None
        self.last_observation = None
        self.progress = 0.0
        self.note = "Hold FRONT, then press F twice to record it."
        self.resolution = None
        self.saved = False
        self.armed_label = None
        self._auto_advance = False
        self.diagnostic_path = None
        self._logged = False

    def close(self):
        if any(self.samples.values()) and not self._logged:
            self._write_diagnostic("cancelled")
        if not self.saved:
            self.tracker.config = deepcopy(self.original_config)
        self.tracker.reset()

    def _write_diagnostic(self, status: str, error: str | None = None):
        """Write numeric calibration evidence only; camera frames are never logged."""
        if self._logged:
            return
        def statistics(key):
            values = [row[key] for rows in self.samples.values() for row in rows if key in row]
            return {} if not values else {
                "count": len(values), "p10": float(np.percentile(values, 10)),
                "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            }
        try:
            self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.diagnostic_path = self.diagnostic_dir / f"{stamp}-{uuid4().hex[:8]}.json"
            payload = {
                "kind": "pnp_device_setup", "status": status,
                "time": datetime.now(timezone.utc).isoformat(), "error": error,
                "backend": self.tracker.backend.name, "camera_resolution": self.resolution,
                "statistics": {key: statistics(key) for key in ("pnp_yaw", "pnp_pitch")},
                "samples": {
                    label: [{key: float(row[key]) for key in ("pnp_yaw", "pnp_pitch") if key in row}
                    for row in rows] for label, rows in self.samples.items()
                },
            }
            self.diagnostic_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            self._logged = True
        except OSError:
            self.diagnostic_path = None

    def start(self, now: float):
        """Automatic entry point retained for the display-less terminal runner."""
        if self.phase == "ready":
            self._auto_advance = True
            self.phase = "settling"
            self.since = now
            self.note = "Hold the requested position still before recording begins."

    def tap(self, key: str, now: float):
        """Two matching operator key taps start one explicitly requested capture."""
        label = CALIBRATION_KEYS.get(str(key).lower())
        if label is None or self.phase in ("complete", "error", "capture", "settling"):
            return
        expected = LABELS[self.stage]
        if label != expected:
            self.phase = "ready"
            self.armed_label = None
            self.note = f"This step needs {expected}. Press {expected[0]} twice when ready."
            return
        if self.phase == "ready":
            self.phase = "armed"
            self.armed_label = label
            self.note = f"{label}: press {label[0]} again to start recording."
            return
        if self.phase == "armed" and self.armed_label == label:
            self.phase = "settling"
            self.since = now
            self.progress = 0.0
            self.note = "Hold still. Recording starts after the brief settle."

    def cancel_current(self):
        if self.phase in ("armed", "settling", "capture"):
            self.phase = "ready"
            self.armed_label = None
            self.since = None
            self.last_observation = None
            self.samples[LABELS[self.stage]].clear()
            self.progress = 0.0
            self.note = f"{LABELS[self.stage]} cancelled. Press {LABELS[self.stage][0]} twice when ready."

    def _valid(self, pose) -> bool:
        raw = None if pose is None else pose.raw
        c = self.tracker.config
        return (
            raw is not None
            and raw.face_score >= c["min_confidence"]
            and abs(raw.roll) <= c["roll_limit"]
            and all(key in raw.features and np.isfinite(raw.features[key]) for key in ("pnp_yaw", "pnp_pitch"))
        )

    def _reset_hold(self, now: float):
        self.since = now
        self.last_observation = None
        self.samples[LABELS[self.stage]].clear()
        self.progress = 0.0

    def _next_missing_stage(self):
        return next((index for index, label in enumerate(LABELS) if len(self.samples[label]) < 8), None)

    def _down_guidance(self, pose):
        """Use recorded raw PnP evidence only; the unfitted classifier is never consulted."""
        if LABELS[self.stage] != "DOWN" or not self.samples["FRONT"] or not self.samples["UP"]:
            return None
        front_pitch = np.array([row["pnp_pitch"] for row in self.samples["FRONT"]], dtype=float)
        front_yaw = np.array([row["pnp_yaw"] for row in self.samples["FRONT"]], dtype=float)
        baseline = float(np.median(front_pitch))
        up = float(np.median([row["pnp_pitch"] for row in self.samples["UP"]]))
        sign = np.sign(up - baseline)
        if not sign:
            return None
        pitch_distance = -sign * (pose.raw.features["pnp_pitch"] - baseline)
        allowance = float(np.percentile(np.abs(front_pitch - baseline), 95)) + self.tracker.config.get("pitch_front_padding", 3.0)
        yaw_distance = abs(pose.raw.features["pnp_yaw"] - float(np.median(front_yaw)))
        if pitch_distance <= allowance:
            return f"Lower your chin more. Down reading {pitch_distance:.1f}; needs over {allowance:.1f}."
        if yaw_distance > 10.0:
            return "Keep facing the camera while looking down."
        return "DOWN looks clear. Keep still until recording finishes."

    def _retry_recordings(self, error: CalibrationError, now: float):
        labels = [label for label in error.retry_labels if label in self.samples]
        if not labels:
            return False
        self._write_diagnostic("rejected", str(error))
        logged_name = self.diagnostic_path.name if self.diagnostic_path else None
        for label in labels:
            self.samples[label].clear()
        self.tracker.reset()
        self.stage = min(LABELS.index(label) for label in labels)
        self.armed_label = None
        self.last_observation = None
        self.progress = 0.0
        self._logged = False
        self.diagnostic_path = None
        target = LABELS[self.stage]
        if self._auto_advance:
            self.phase, self.since = "settling", now
            self.note = f"Retrying {target}; hold the position still before recording begins."
        else:
            self.phase, self.since = "ready", None
            saved_note = f" Evidence: {logged_name}." if logged_name else ""
            self.note = f"{target} needs another recording. Other positions were kept. Press {target[0]} twice when ready." + saved_note
        return True

    def update(self, pose, now: float, resolution: tuple[int, int] | None = None):
        if self.phase in ("ready", "armed", "complete", "error"):
            return
        if resolution is not None:
            self.resolution = [int(resolution[0]), int(resolution[1])]
        if not self._valid(pose):
            self._reset_hold(now)
            self.note = "Keep one face visible and upright."
            return
        if self.phase == "settling":
            elapsed = now - self.since
            self.progress = min(1.0, elapsed / self.settle_s)
            self.note = self._down_guidance(pose) or "Hold the requested position still before recording begins."
            if elapsed >= self.settle_s:
                self.phase = "capture"
                self._reset_hold(now)
                self.note = self._down_guidance(pose) or "Recording this pose. Hold still."
            return
        if pose.timestamp_s == self.last_observation:
            return
        self.last_observation = pose.timestamp_s
        rows = self.samples[LABELS[self.stage]]
        rows.append(dict(pose.raw.features))
        elapsed = now - self.since
        self.progress = min(1.0, elapsed / self.capture_s)
        if elapsed < self.capture_s or len(rows) < 8:
            self.note = self._down_guidance(pose) or "Recording this pose. Hold still."
            return
        next_stage = self._next_missing_stage()
        if next_stage is not None:
            self.stage = next_stage
            self.progress = 0.0
            self.armed_label = None
            self.tracker.reset()
            next_label = LABELS[self.stage]
            if self._auto_advance:
                self.phase = "settling"
                self.since = now
                self.note = "Hold the requested position still before recording begins."
            else:
                self.phase = "ready"
                self.since = None
                self.note = f"{next_label} is next. Press {next_label[0]} twice when ready."
            return
        try:
            self.tracker.guided_calibrate(self.samples, resolution=self.resolution, pnp_only=True)
        except CalibrationError as exc:
            if self._retry_recordings(exc, now):
                return
            self._write_diagnostic("rejected", str(exc))
            self.tracker.config = deepcopy(self.original_config)
            self.tracker.reset()
            self.phase = "error"
            name = f" See logs/head_pose/device_setup/{self.diagnostic_path.name}." if self.diagnostic_path else ""
            self.note = str(exc) + name
            return
        except ValueError as exc:
            self._write_diagnostic("rejected", str(exc))
            self.tracker.config = deepcopy(self.original_config)
            self.tracker.reset()
            self.phase = "error"
            name = f" See logs/head_pose/device_setup/{self.diagnostic_path.name}." if self.diagnostic_path else ""
            self.note = str(exc) + name
            return
        self._write_diagnostic("complete")
        self.stage = len(LABELS)
        self.phase = "complete"
        self.progress = 1.0
        name = f" Log: {self.diagnostic_path.name}." if self.diagnostic_path else ""
        self.note = "PnP device setup is ready to save." + name

    def save(self, path):
        if self.phase != "complete":
            raise RuntimeError("Calibration is not complete")
        self.tracker.save(path)
        self.saved = True


def format_instruction(flow) -> str:
    if flow.phase in ("setup", "complete", "error") or flow.stage >= len(LABELS):
        return ""
    return INSTRUCTIONS[LABELS[flow.stage]]
