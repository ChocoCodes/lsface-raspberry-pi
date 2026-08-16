#!/usr/bin/env python3
"""Quality-first multi-face LBPH/SFace cascade for the r3 Pi candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from lbph_config import (
    descriptor_from_mapping,
    make_lbph,
    release_descriptor_from_manifest,
    validate_loaded_lbph,
)
from quality import QualityThresholds, compute_quality


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "thresholds.r3.json"
DEFAULT_MODELS_ROOT = ROOT


def route_after_quality(
    quality_triggered: bool,
    lbph_distance: float | None,
    tau_accept: float,
    tau_reject: float,
) -> str:
    """Choose route; quality trigger deliberately resolves before LBPH score."""

    if quality_triggered:
        return "sface_quality"
    if lbph_distance is None:
        raise ValueError("LBPH distance required when quality is clear.")
    if lbph_distance <= tau_accept:
        return "lbph_accept"
    if lbph_distance < tau_reject:
        return "sface_ambiguous"
    return "lbph_reject"


def _path_arg(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _find_model(root: Path, filename: str) -> Path:
    candidates = (root / filename, root / "models" / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} under {root} or {root / 'models'}.")


def _has_model(root: Path, filename: str) -> bool:
    return (root / filename).exists() or (root / "models" / filename).exists()


def _default_asset_root(base_dir: Path) -> Path:
    candidates = (base_dir, DEFAULT_MODELS_ROOT)
    return next((candidate for candidate in candidates if _has_model(candidate, "face_detection_yunet_2023mar.onnx")), base_dir)


def _default_config_path(base_dir: Path) -> Path:
    candidates = (base_dir / "config" / "thresholds.json", DEFAULT_CONFIG)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _default_enrollment_root(base_dir: Path) -> Path:
    candidates = (base_dir / "enrollment", ROOT / "enrollment")
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


class HybridCascade:
    """Run quality-first routing with descriptor/release compatibility checks.

    The upstream hardware call contract is retained: construct with
    ``HybridCascade(base_dir=".")`` and pass a BGR ``(H, W, 3)`` ndarray to
    ``infer``.  The extra keyword paths are additive integration controls.
    """

    def __init__(
        self,
        base_dir: str | Path = ".",
        *,
        models_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        enrollment_root: str | Path | None = None,
    ) -> None:
        self.base_dir = _path_arg(base_dir)
        self.models_dir = _path_arg(models_dir) if models_dir is not None else _default_asset_root(self.base_dir)
        self.config_path = _path_arg(config_path) if config_path is not None else _default_config_path(self.base_dir)
        selected_enrollment_root = enrollment_root if enrollment_root is not None else _default_enrollment_root(self.base_dir)
        self.enrollment_root = _path_arg(selected_enrollment_root)
        self.lbph_calls = 0
        self.sface_calls = 0

        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.threshold_descriptor = descriptor_from_mapping(
            cfg.get("lbph_descriptor"), source=str(self.config_path)
        )
        gate = cfg["gate"]
        self.tau_accept = float(gate["tau_accept"])
        self.tau_reject = float(gate["tau_reject"])
        if self.tau_reject <= self.tau_accept:
            raise RuntimeError("gate.tau_reject must be greater than gate.tau_accept.")
        self.quality_thresholds = QualityThresholds.from_dict(cfg.get("quality"))
        # Keep the upstream public threshold names available to hardware-side
        # diagnostics while using the structured quality object internally.
        self.q_min_face = self.quality_thresholds.px_min
        self.q_tau_blur = self.quality_thresholds.tau_blur
        sface_cfg = cfg["sface"]
        self.sface_l2_genuine = float(sface_cfg["l2_genuine"])
        self.sface_cosine_genuine = float(sface_cfg.get("cosine_genuine", -1.0))

        self.detector = cv.FaceDetectorYN.create(
            str(_find_model(self.models_dir, "face_detection_yunet_2023mar.onnx")),
            "",
            (320, 320),
            0.6,
            0.3,
            5000,
        )
        self.sface = cv.FaceRecognizerSF.create(
            str(_find_model(self.models_dir, "face_recognition_sface_2021dec.onnx")), ""
        )

        self.artifacts_dir = self._resolve_artifacts(artifacts_dir)
        manifest_path = self.artifacts_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Candidate release missing manifest.json: {self.artifacts_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        release_descriptor_from_manifest(
            manifest, self.threshold_descriptor, source=str(manifest_path)
        )

        lbph_path = self.artifacts_dir / "lbph.yml"
        labels_path = self.artifacts_dir / "labels.json"
        self.lbph = make_lbph(self.threshold_descriptor)
        self.lbph.read(str(lbph_path))
        validate_loaded_lbph(self.lbph, self.threshold_descriptor, context=str(lbph_path))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.lbph_labels = {int(identifier): name for name, identifier in labels.items()}
        # Upstream exposes this id -> name mapping as ``labels``.  Keep the
        # old spelling as a compatibility alias; ``lbph_labels`` remains the
        # explicit name for new code.
        self.labels = dict(self.lbph_labels)

        gallery_path = self.artifacts_dir / "sface_gallery.npy"
        gallery = np.load(gallery_path, allow_pickle=True).item()
        if not isinstance(gallery, dict) or not gallery:
            raise RuntimeError(f"SFace gallery empty or invalid: {gallery_path}")
        identity_ids = {name: identifier for identifier, name in self.lbph_labels.items()}
        self.sface_gallery = [
            {
                "id": int(identity_ids[name]),
                "name": name,
                "embedding": np.asarray(gallery[name], dtype=np.float32).reshape(1, -1),
            }
            for name in gallery
        ]
        self.sface_labels = [person["name"] for person in self.sface_gallery]
        if any(
            person["embedding"].shape != (1, 128)
            or not np.isfinite(person["embedding"]).all()
            for person in self.sface_gallery
        ):
            raise RuntimeError(f"SFace gallery contains invalid feature: {gallery_path}")

    def _resolve_artifacts(self, artifacts_dir: str | Path | None) -> Path:
        if artifacts_dir is not None:
            resolved = _path_arg(artifacts_dir)
            if not resolved.is_dir():
                raise FileNotFoundError(resolved)
            return resolved
        pointer = self.enrollment_root / "current.json"
        if not pointer.exists():
            raise RuntimeError(
                f"No active candidate release at {pointer}; run rebuild_release.py first."
            )
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        release_value = payload.get("release")
        if not isinstance(release_value, str):
            raise RuntimeError(f"Invalid release pointer: {pointer}")
        root = pointer.parent.resolve()
        resolved = (root / release_value).resolve()
        if root not in resolved.parents or not resolved.is_dir():
            raise RuntimeError(f"Release pointer escapes enrollment root or is missing: {resolved}")
        return resolved

    @staticmethod
    def _normalize_lbph(face_gray: np.ndarray) -> np.ndarray:
        image = cv.resize(face_gray, (100, 100), interpolation=cv.INTER_AREA)
        image = np.float32(image) / 255.0
        alpha, tau, gamma = 0.1, 10.0, 0.2
        image = np.power(image, gamma)
        image = cv.GaussianBlur(image, (0, 0), sigmaX=1.0)
        mean_alpha = np.mean(np.power(np.abs(image), alpha))
        mean_tau = np.mean(np.power(np.abs(image), tau))
        if mean_alpha <= 0.0 or mean_tau <= 0.0:
            return np.zeros((100, 100), dtype=np.uint8)
        image = image / np.power(mean_alpha, 1.0 / alpha)
        image = image / np.power(mean_tau, 1.0 / tau)
        image = tau * np.tanh(image / tau)
        return np.uint8(cv.normalize(image, None, 0, 255, cv.NORM_MINMAX))

    @staticmethod
    def _bbox(row: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        left = max(0, min(int(round(float(row[0]))), width - 1))
        top = max(0, min(int(round(float(row[1]))), height - 1))
        right = max(left + 1, min(int(round(float(row[0] + row[2]))), width))
        bottom = max(top + 1, min(int(round(float(row[1] + row[3]))), height))
        return left, top, right - left, bottom - top

    def _sface_decision(
        self,
        image_bgr: np.ndarray,
        row: np.ndarray,
        *,
        bbox: tuple[int, int, int, int],
        quality: object,
        lbph_distance: float | None,
        gate_reason: str,
        route: str,
    ) -> dict:
        self.sface_calls += 1
        aligned = self.sface.alignCrop(image_bgr, row)
        feature = np.asarray(self.sface.feature(aligned), dtype=np.float32).reshape(1, -1)
        if feature.shape != (1, 128) or not np.isfinite(feature).all():
            raise RuntimeError("SFace returned invalid feature.")

        best_index = -1
        best_l2 = float("inf")
        best_cosine = float("-inf")
        cosine_code = getattr(cv, "FaceRecognizerSF_FR_COSINE", None)
        for index, person in enumerate(self.sface_gallery):
            gallery_feature = person["embedding"]
            l2 = float(self.sface.match(feature, gallery_feature, cv.FaceRecognizerSF_FR_NORM_L2))
            if cosine_code is None:
                cosine = 1.0 - (l2 * l2 / 2.0)
            else:
                cosine = float(self.sface.match(feature, gallery_feature, cosine_code))
            if l2 < best_l2:
                best_index, best_l2, best_cosine = index, l2, cosine

        name = self.sface_gallery[best_index]["name"]
        accepted = (
            best_l2 <= self.sface_l2_genuine
            and best_cosine >= self.sface_cosine_genuine
        )
        result = {
            "status": "accepted" if accepted else "rejected",
            "engine": "sface",
            "name": name,
            "l2": best_l2,
            "cosine": best_cosine,
            "gate_reason": gate_reason,
            "route": route,
            "bbox": tuple(bbox),
            "quality": quality.to_dict(),
            "descriptor_id": self.threshold_descriptor.descriptor_id,
            "lbph_distance": lbph_distance,
        }
        if not accepted:
            result["reason"] = "impostor"
        return result

    def _infer_face(self, image_bgr: np.ndarray, row: np.ndarray) -> dict:
        height, width = image_bgr.shape[:2]
        x, y, box_width, box_height = self._bbox(row, width, height)
        gray = cv.cvtColor(image_bgr[y : y + box_height, x : x + box_width], cv.COLOR_BGR2GRAY)
        if gray.size == 0:
            return {"status": "rejected", "engine": "none", "reason": "empty_face_crop", "bbox": (x, y, box_width, box_height)}
        quality_gray = cv.resize(gray, (100, 100), interpolation=cv.INTER_AREA)
        landmarks = row[5:15] if row.size >= 15 else None
        quality = compute_quality(
            gray_roi=quality_gray,
            landmarks=landmarks,
            face_px=min(box_width, box_height),
            thresholds=self.quality_thresholds,
        )
        route = route_after_quality(quality.any_flag, None, self.tau_accept, self.tau_reject)
        if route == "sface_quality":
            reason = "quality:" + ",".join(quality.active_flags)
            return self._sface_decision(
                image_bgr, row, bbox=(x, y, box_width, box_height), quality=quality,
                lbph_distance=None, gate_reason=reason, route=route,
            )

        lbph_input = self._normalize_lbph(gray)
        self.lbph_calls += 1
        predicted_id, distance = self.lbph.predict(lbph_input)
        lbph_distance = float(distance)
        route = route_after_quality(False, lbph_distance, self.tau_accept, self.tau_reject)
        name = self.lbph_labels.get(int(predicted_id), "Unknown")
        common = {
            "bbox": (x, y, box_width, box_height),
            "quality": quality.to_dict(),
            "descriptor_id": self.threshold_descriptor.descriptor_id,
            "distance": lbph_distance,
            "name": name,
            "route": route,
        }
        if route == "lbph_accept":
            return {"status": "accepted", "engine": "lbph", **common}
        if route == "lbph_reject":
            return {"status": "rejected", "engine": "lbph", "reason": "confident_reject", **common}
        return self._sface_decision(
            image_bgr, row, bbox=(x, y, box_width, box_height), quality=quality,
            lbph_distance=lbph_distance, gate_reason="ambiguous_band", route=route,
        )

    def infer(self, image_bgr: np.ndarray) -> list[dict]:
        height, width = image_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return []
        return [self._infer_face(image_bgr, np.asarray(row, dtype=np.float32)) for row in faces]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--image")
    options = parser.parse_args()
    cascade = HybridCascade(
        models_dir=options.models_dir,
        config_path=options.config,
        artifacts_dir=options.artifacts_dir,
    )
    print(f"[R3 READY] descriptor={cascade.threshold_descriptor.descriptor_id}")
    print(f"[R3 READY] release={cascade.artifacts_dir}")
    if options.image:
        image = cv.imread(str(_path_arg(options.image)))
        if image is None:
            raise FileNotFoundError(options.image)
        print(json.dumps(cascade.infer(image), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
