import copy
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np

from src.config.config import MODELS_PATH
from src.engine.face_aligner import FaceAligner
from src.engine.sface import SFace


class FeatureDB:
    """Editable feature database used by the enrollment flow."""

    def __init__(self) -> None:
        self.aligner = FaceAligner(
            detector_weights=MODELS_PATH / "face_detection_yunet_2023mar.onnx",
            recognizer_weights=MODELS_PATH / "face_recognition_sface_2021dec.onnx",
        )
        self.lbph = cv.face.LBPHFaceRecognizer_create()
        self.sface = SFace(threshold=1.0313)
        self.lbph_crop = (100, 100)
        self.db: dict[str, dict] = {}

    @classmethod
    def load(cls, path: str | Path) -> "FeatureDB":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Feature DB not found at: '{p}'")
        if p.suffix.lower() != ".npy":
            raise ValueError("Feature DB Path should end with '.npy'")

        instance = cls()
        payload = np.load(p, allow_pickle=True).item()
        cls.validate_database(payload)
        instance.db = payload
        print(f"[FeatureDB] Loaded {len(instance.db)} identities from '{p}'.")
        return instance

    @staticmethod
    def validate_database(database: object) -> None:
        """Raise ``ValueError`` when a persisted database is malformed."""

        if not isinstance(database, dict):
            raise ValueError("Feature DB must contain a dictionary.")

        identity_ids: set[int] = set()
        for name, record in database.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Feature DB contains an invalid identity name.")
            if not isinstance(record, dict):
                raise ValueError(f"Feature DB record for {name!r} is not a dictionary.")
            try:
                identity_id = int(record["id"])
                lbph = record["lbph"]
                sface = record["sface"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Feature DB record for {name!r} is incomplete.") from exc

            if identity_id < 0 or identity_id in identity_ids:
                raise ValueError(f"Feature DB contains a duplicate/invalid id for {name!r}.")
            identity_ids.add(identity_id)
            if not isinstance(lbph, (list, tuple, np.ndarray)) or not isinstance(sface, (list, tuple, np.ndarray)):
                raise ValueError(f"Feature DB record for {name!r} has invalid feature lists.")
            if len(lbph) == 0 or len(lbph) != len(sface):
                raise ValueError(f"Feature DB record for {name!r} has mismatched samples.")

            for index, tile in enumerate(lbph):
                tile_array = np.asarray(tile)
                if tile_array.shape != (100, 100):
                    raise ValueError(
                        f"{name!r} LBPH sample {index} has shape {tile_array.shape}; "
                        "expected (100, 100)."
                    )
            for index, embedding in enumerate(sface):
                embedding_array = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if embedding_array.size != 128 or not np.isfinite(embedding_array).all():
                    raise ValueError(f"{name!r} SFace sample {index} is invalid.")

    def clone(self) -> "FeatureDB":
        """Copy the data while reusing the model instances."""

        clone = object.__new__(FeatureDB)
        clone.aligner = self.aligner
        clone.lbph = self.lbph
        clone.sface = self.sface
        clone.lbph_crop = self.lbph_crop
        clone.db = copy.deepcopy(self.db)
        return clone

    @staticmethod
    def normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("Identity name must be text.")
        normalized = " ".join(name.split())
        if not normalized:
            raise ValueError("Identity name cannot be empty.")
        if len(normalized) > 100:
            raise ValueError("Identity name must be 100 characters or fewer.")
        return normalized

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix.lower() != ".npy":
            raise ValueError("Feature DB Path should end with '.npy'")
        self.validate_database(self.db)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, self.db, allow_pickle=True)
        print(f"[FeatureDB] Saved {len(self.db)} identities -> '{p}'")

    def _normalize_lbph(self, face_gray: np.ndarray) -> np.ndarray:
        img = cv.resize(face_gray, self.lbph_crop, interpolation=cv.INTER_AREA)
        img = np.float32(img) / 255.0
        alpha, tau, gamma = 0.1, 10.0, 0.2
        img = np.power(img, gamma)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=1.0)

        alpha_norm = np.mean(np.power(np.abs(img), alpha))
        if not np.isfinite(alpha_norm) or alpha_norm <= 1e-8:
            return np.zeros(self.lbph_crop, dtype=np.uint8)
        img = img / np.power(alpha_norm, 1.0 / alpha)

        tau_norm = np.mean(np.power(np.abs(img), tau))
        if not np.isfinite(tau_norm) or tau_norm <= 1e-8:
            return np.zeros(self.lbph_crop, dtype=np.uint8)
        img = img / np.power(tau_norm, 1.0 / tau)
        img = tau * np.tanh(img / tau)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        return np.uint8(img)

    def _extract_features(self, bgr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(bgr_img, np.ndarray) or bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
            raise ValueError("Enrollment frame must be a BGR image.")
        if bgr_img.size == 0:
            raise ValueError("Enrollment frame is empty.")

        aligned_faces = self.aligner.detect_and_align(bgr_img)
        if len(aligned_faces) != 1:
            raise ValueError(
                "Enrollment requires exactly one detectable face; "
                f"found {len(aligned_faces)}."
            )

        face, aligned = aligned_faces[0]
        bbox = self.aligner.get_bbox(face, bgr_img.shape)
        if bbox is None:
            raise ValueError("Could not determine the enrolled face bounding box.")

        x, y, width, height = bbox
        face_bgr = bgr_img[y : y + height, x : x + width]
        if face_bgr.size == 0:
            raise ValueError("Enrolled face crop is empty.")

        lbph_face = self._normalize_lbph(cv.cvtColor(face_bgr, cv.COLOR_BGR2GRAY))
        sface_embedding = np.asarray(self.sface.get_embedding(aligned), dtype=np.float32).reshape(-1)
        if sface_embedding.size != 128 or not np.isfinite(sface_embedding).all():
            raise ValueError("SFace produced an invalid embedding.")
        return lbph_face, sface_embedding

    def enroll_frame(self, name: str, bgr_img: np.ndarray) -> None:
        """Extract and append one captured frame without partial mutation."""

        normalized_name = self.normalize_name(name)
        lbph_face, sface_embedding = self._extract_features(bgr_img)

        if normalized_name not in self.db:
            identity_id = max(
                (int(record["id"]) for record in self.db.values()), default=-1
            ) + 1
            self.db[normalized_name] = {
                "id": identity_id,
                "lbph": [],
                "sface": [],
            }

        record = self.db[normalized_name]
        record["lbph"].append(lbph_face)
        record["sface"].append(sface_embedding)

    def enroll(self, name: str, bgr_img: np.ndarray) -> Tuple[bool, bool]:
        """Backward-compatible single-frame enrollment API."""

        try:
            self.enroll_frame(name, bgr_img)
        except (ValueError, cv.error) as exc:
            print(f"[FeatureDB] Enrollment rejected: {exc}")
            return False, False
        return True, True

    def batch_enroll(self, dataset: List[Tuple[str, str]]) -> dict:
        """Enroll image paths supplied as ``(identity_name, image_path)`` pairs."""

        for name, image_path in dataset:
            frame = cv.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Could not read enrollment image: {image_path}")
            self.enroll_frame(name, frame)
        return self.db

    def get_identities(self) -> List[str]:
        return list(self.db.keys())

    def get_identity_count(self) -> int:
        return len(self.db)
