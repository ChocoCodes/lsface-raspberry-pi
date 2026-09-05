"""Cheap quality probes used before LBPH in the r3 cascade."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    tau_blur: float = 60.0
    luma_lo: float = 50.0
    luma_hi: float = 205.0
    tau_noise: float = 12.0
    tau_pose: float = 18.0
    px_min: int = 80

    @classmethod
    def from_dict(cls, data: dict | None) -> "QualityThresholds":
        defaults = cls()
        values = {
            key: getattr(defaults, key) for key in cls.__dataclass_fields__
        }
        if data:
            values.update({
                key: type(getattr(defaults, key))(value)
                for key, value in data.items()
                if key in cls.__dataclass_fields__
            })
        return cls(**values)


@dataclass
class QualityReport:
    blur_var: float
    luma_mean: float
    noise_sigma: float
    pose_angle: float
    face_px: int
    flags: dict[str, bool] = field(default_factory=dict)

    @property
    def any_flag(self) -> bool:
        return any(self.flags.values())

    @property
    def active_flags(self) -> list[str]:
        return [name for name, enabled in self.flags.items() if enabled]

    def to_dict(self) -> dict:
        return {
            "blur_var": round(self.blur_var, 3),
            "luma_mean": round(self.luma_mean, 2),
            "noise_sigma": round(self.noise_sigma, 3),
            "pose_angle": round(self.pose_angle, 2),
            "face_px": self.face_px,
            "flags": dict(self.flags),
        }


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def estimate_noise_sigma(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = np.abs(cv.filter2D(gray.astype(np.float64), -1, mask))
    h, w = gray.shape[:2]
    return float(conv.sum() * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2)))


def pose_angle_from_landmarks(landmarks: np.ndarray | None) -> float:
    if landmarks is None:
        return 0.0
    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return 0.0
    right_eye, left_eye, nose = points[0], points[1], points[2]
    dx = float(left_eye[0] - right_eye[0])
    dy = float(left_eye[1] - right_eye[1])
    roll = abs(math.degrees(math.atan2(dy, dx)))
    if roll > 90.0:
        roll = 180.0 - roll
    inter_ocular = math.hypot(dx, dy)
    if inter_ocular < 1e-3:
        return roll
    eye_mid_x = 0.5 * (right_eye[0] + left_eye[0])
    yaw_proxy = min(90.0, abs(float(nose[0] - eye_mid_x)) / inter_ocular * 90.0)
    return max(roll, yaw_proxy)


def compute_quality(
    *, gray_roi: np.ndarray, landmarks: np.ndarray | None, face_px: int,
    thresholds: QualityThresholds,
) -> QualityReport:
    blur_var = variance_of_laplacian(gray_roi)
    luma_mean = float(np.mean(gray_roi)) if gray_roi.size else 0.0
    noise_sigma = estimate_noise_sigma(gray_roi)
    pose_angle = pose_angle_from_landmarks(landmarks)
    flags = {
        "blur": blur_var < thresholds.tau_blur,
        "low_light": luma_mean < thresholds.luma_lo or luma_mean > thresholds.luma_hi,
        "noise": noise_sigma > thresholds.tau_noise,
        "off_pose": pose_angle > thresholds.tau_pose,
        "small_face": face_px < thresholds.px_min,
    }
    return QualityReport(blur_var, luma_mean, noise_sigma, pose_angle, int(face_px), flags)
