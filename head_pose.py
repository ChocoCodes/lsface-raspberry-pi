"""Live YuNet-based coarse head-pose estimation for LS-Face demo testing.

This module is intentionally isolated from the recognition pipeline. It runs a
second YuNet detection pass so the existing HybridCascade contract remains
untouched while head-pose feasibility is evaluated.

Direction labels are from the SUBJECT'S perspective: FRONT, LEFT, RIGHT, UP,
DOWN. Left/right is deliberately given priority over pitch because that is the
more reliable axis for the enrollment flow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import cv2 as cv
import numpy as np


# Generic 3D face template in the exact landmark order produced by YuNet:
# right eye, left eye, nose, right mouth corner, left mouth corner.
# Absolute units are arbitrary; relative geometry is what solvePnP uses here.
_MODEL_POINTS = np.asarray(
    [
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (0.0, 0.0, 0.0),
        (-25.0, 30.0, -20.0),
        (25.0, 30.0, -20.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PoseEstimate:
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    score: float
    direction: str

    # Baseline-corrected, smoothed PnP angles in degrees.
    yaw: float
    pitch: float
    roll: float

    # Baseline-corrected, smoothed 2D landmark proxies (unitless).
    yaw_proxy: float
    pitch_proxy: float

    # Raw values are useful for calibration diagnostics/logging.
    raw_yaw: float
    raw_pitch: float
    raw_roll: float
    raw_yaw_proxy: float
    raw_pitch_proxy: float

    calibrated: bool
    calibration_count: int
    calibration_target: int
    pitch_source: str


class HeadPoseTracker:
    """Detect the dominant face and classify coarse subject head direction."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        calibration_frames: int = 20,
        smoothing_window: int = 3,
        yaw_threshold_deg: float = 10.0,
        pitch_threshold_deg: float = 5.0,
        pitch_proxy_threshold: float = 0.045,
        max_roll_for_direction_deg: float = 28.0,
        pitch_source: str = "pnp",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        self.detector = cv.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )

        self.calibration_frames = max(1, int(calibration_frames))
        self.yaw_threshold_deg = float(yaw_threshold_deg)
        self.pitch_threshold_deg = float(pitch_threshold_deg)
        self.pitch_proxy_threshold = float(pitch_proxy_threshold)
        self.max_roll_for_direction_deg = float(max_roll_for_direction_deg)

        pitch_source = str(pitch_source).lower().strip()
        if pitch_source not in {"pnp", "landmark"}:
            raise ValueError("pitch_source must be 'pnp' or 'landmark'.")
        self._pitch_source = pitch_source

        # raw_yaw, raw_pitch, raw_roll, raw_yaw_proxy, raw_pitch_proxy
        self._calibration_samples: list[tuple[float, float, float, float, float]] = []
        self._baseline = np.zeros(5, dtype=np.float64)
        self._calibrated = False

        window = max(1, int(smoothing_window))
        self._yaw_history: Deque[float] = deque(maxlen=window)
        self._pitch_history: Deque[float] = deque(maxlen=window)
        self._roll_history: Deque[float] = deque(maxlen=window)
        self._yaw_proxy_history: Deque[float] = deque(maxlen=window)
        self._pitch_proxy_history: Deque[float] = deque(maxlen=window)

        # Camera/model conventions can flip semantic directions. The tester
        # exposes X/Y keys to correct this without editing code.
        self._yaw_sign = 1.0
        self._pitch_sign = 1.0

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def yaw_mapping(self) -> str:
        return "+yaw=LEFT" if self._yaw_sign > 0 else "+yaw=RIGHT"

    @property
    def pitch_mapping(self) -> str:
        return "+pitch=UP" if self._pitch_sign > 0 else "+pitch=DOWN"

    @property
    def pitch_source(self) -> str:
        return self._pitch_source

    @property
    def pitch_threshold_description(self) -> str:
        if self._pitch_source == "pnp":
            return f"{self.pitch_threshold_deg:.2f} deg"
        return f"{self.pitch_proxy_threshold:.4f} proxy"

    def reset_calibration(self) -> None:
        self._calibration_samples.clear()
        self._baseline[:] = 0.0
        self._calibrated = False
        self._clear_histories()

    def _clear_histories(self) -> None:
        self._yaw_history.clear()
        self._pitch_history.clear()
        self._roll_history.clear()
        self._yaw_proxy_history.clear()
        self._pitch_proxy_history.clear()

    def flip_yaw_mapping(self) -> None:
        self._yaw_sign *= -1.0

    def flip_pitch_mapping(self) -> None:
        self._pitch_sign *= -1.0

    def toggle_pitch_source(self) -> str:
        self._pitch_source = "landmark" if self._pitch_source == "pnp" else "pnp"
        return self._pitch_source

    @staticmethod
    def _extract_bbox(row: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x = int(round(float(row[0])))
        y = int(round(float(row[1])))
        w = int(round(float(row[2])))
        h = int(round(float(row[3])))

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        return x, y, w, h

    @staticmethod
    def _extract_landmarks(row: np.ndarray) -> np.ndarray:
        # FaceDetectorYN output: bbox 0:4, five x/y landmarks 4:14, score 14.
        if row.size < 15:
            raise ValueError(f"YuNet face row has {row.size} values; expected at least 15.")
        points = np.asarray(row[4:14], dtype=np.float64).reshape(5, 2)
        if not np.isfinite(points).all():
            raise ValueError("YuNet returned non-finite facial landmarks.")
        return points

    @staticmethod
    def _camera_matrix(frame_shape: tuple[int, ...]) -> np.ndarray:
        height, width = frame_shape[:2]
        # Feasibility approximation. Neutral calibration absorbs much of the
        # camera/model bias. Production can replace this with a calibrated K.
        focal = float(width)
        return np.asarray(
            [
                (focal, 0.0, width / 2.0),
                (0.0, focal, height / 2.0),
                (0.0, 0.0, 1.0),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
        """Return pitch, yaw, roll in degrees from a 3x3 rotation matrix."""

        sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(-rotation[2, 0], sy)
            roll = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
            yaw = np.arctan2(-rotation[2, 0], sy)
            roll = 0.0

        return tuple(float(np.degrees(value)) for value in (pitch, yaw, roll))

    def _solve_pose(
        self, landmarks: np.ndarray, frame_shape: tuple[int, ...]
    ) -> tuple[float, float, float] | None:
        camera_matrix = self._camera_matrix(frame_shape)
        distortion = np.zeros((4, 1), dtype=np.float64)
        image_points = np.ascontiguousarray(landmarks, dtype=np.float64).reshape(-1, 1, 2)

        method = getattr(cv, "SOLVEPNP_SQPNP", cv.SOLVEPNP_EPNP)
        success, rotation_vector, _translation_vector = cv.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=method,
        )
        if not success:
            return None

        rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
        if not np.isfinite([pitch, yaw, roll]).all():
            return None
        return pitch, yaw, roll

    @staticmethod
    def _landmark_proxies(landmarks: np.ndarray) -> tuple[float, float]:
        """Return normalized 2D yaw/pitch signals independent of solvePnP.

        yaw_proxy:
            Horizontal nose displacement from the eye midpoint, normalized by
            inter-eye distance.

        pitch_proxy:
            Vertical nose position between eye midpoint and mouth midpoint,
            centered symmetrically. Neutral calibration later subtracts the
            person's baseline, so only movement from neutral matters.
        """

        right_eye, left_eye, nose, right_mouth, left_mouth = landmarks
        eye_mid = 0.5 * (right_eye + left_eye)
        mouth_mid = 0.5 * (right_mouth + left_mouth)

        eye_distance = max(float(np.linalg.norm(left_eye - right_eye)), 1e-6)
        yaw_proxy = float((nose[0] - eye_mid[0]) / eye_distance)

        vertical_span = float(mouth_mid[1] - eye_mid[1])
        if abs(vertical_span) < 1e-6:
            pitch_proxy = 0.0
        else:
            # Positive when looking UP (nose closer to eyes, farther from mouth).
            # Baseline subtraction makes neutral ~= 0.
            pitch_proxy = float(
                ((mouth_mid[1] - nose[1]) - (nose[1] - eye_mid[1]))
                / abs(vertical_span)
            )

        return yaw_proxy, pitch_proxy

    def _update_calibration(
        self,
        raw_yaw: float,
        raw_pitch: float,
        raw_roll: float,
        raw_yaw_proxy: float,
        raw_pitch_proxy: float,
    ) -> None:
        if self._calibrated:
            return

        self._calibration_samples.append(
            (raw_yaw, raw_pitch, raw_roll, raw_yaw_proxy, raw_pitch_proxy)
        )
        if len(self._calibration_samples) < self.calibration_frames:
            return

        values = np.asarray(self._calibration_samples, dtype=np.float64)
        self._baseline = np.median(values, axis=0)
        self._calibrated = True
        self._clear_histories()

    @staticmethod
    def _median(values: Deque[float]) -> float:
        if not values:
            return 0.0
        return float(np.median(np.asarray(values, dtype=np.float64)))

    def _classify(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        pitch_proxy: float,
    ) -> str:
        if not self._calibrated:
            return "CALIBRATING"

        # Deliberate priority: LEFT/RIGHT always wins if yaw is strong enough.
        # This prevents a diagonal left+down pose from being mislabeled DOWN.
        if abs(yaw) >= self.yaw_threshold_deg:
            semantic_yaw = yaw * self._yaw_sign
            return "LEFT" if semantic_yaw > 0.0 else "RIGHT"

        if abs(roll) > self.max_roll_for_direction_deg:
            return "TILTED"

        if self._pitch_source == "pnp":
            pitch_signal = pitch
            threshold = self.pitch_threshold_deg
        else:
            pitch_signal = pitch_proxy
            threshold = self.pitch_proxy_threshold

        if abs(pitch_signal) >= threshold:
            semantic_pitch = pitch_signal * self._pitch_sign
            return "UP" if semantic_pitch > 0.0 else "DOWN"

        return "FRONT"

    def estimate(self, frame_bgr: np.ndarray) -> PoseEstimate | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        height, width = frame_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _retval, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None

        # Largest face = intended booth participant for this tester.
        row = max(
            (np.asarray(face, dtype=np.float32) for face in faces),
            key=lambda face: float(face[2]) * float(face[3]),
        )

        landmarks = self._extract_landmarks(row)
        solved = self._solve_pose(landmarks, frame_bgr.shape)
        if solved is None:
            return None

        raw_pitch, raw_yaw, raw_roll = solved
        raw_yaw_proxy, raw_pitch_proxy = self._landmark_proxies(landmarks)

        self._update_calibration(
            raw_yaw,
            raw_pitch,
            raw_roll,
            raw_yaw_proxy,
            raw_pitch_proxy,
        )

        if self._calibrated:
            yaw = raw_yaw - float(self._baseline[0])
            pitch = raw_pitch - float(self._baseline[1])
            roll = raw_roll - float(self._baseline[2])
            yaw_proxy = raw_yaw_proxy - float(self._baseline[3])
            pitch_proxy = raw_pitch_proxy - float(self._baseline[4])

            self._yaw_history.append(yaw)
            self._pitch_history.append(pitch)
            self._roll_history.append(roll)
            self._yaw_proxy_history.append(yaw_proxy)
            self._pitch_proxy_history.append(pitch_proxy)

            yaw = self._median(self._yaw_history)
            pitch = self._median(self._pitch_history)
            roll = self._median(self._roll_history)
            yaw_proxy = self._median(self._yaw_proxy_history)
            pitch_proxy = self._median(self._pitch_proxy_history)
        else:
            yaw = pitch = roll = yaw_proxy = pitch_proxy = 0.0

        bbox = self._extract_bbox(row, width, height)
        score = float(row[14])
        direction = self._classify(yaw, pitch, roll, pitch_proxy)

        return PoseEstimate(
            bbox=bbox,
            landmarks=landmarks.astype(np.float32),
            score=score,
            direction=direction,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            yaw_proxy=yaw_proxy,
            pitch_proxy=pitch_proxy,
            raw_yaw=raw_yaw,
            raw_pitch=raw_pitch,
            raw_roll=raw_roll,
            raw_yaw_proxy=raw_yaw_proxy,
            raw_pitch_proxy=raw_pitch_proxy,
            calibrated=self._calibrated,
            calibration_count=min(len(self._calibration_samples), self.calibration_frames),
            calibration_target=self.calibration_frames,
            pitch_source=self._pitch_source,
        )
