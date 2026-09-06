import sys
import cv2 as cv
from pathlib import Path 
import numpy as np
from typing import List, Tuple 

from src.engine.face_aligner import FaceAligner 
from src.engine.sface import SFace 
from src.config.config import MODELS_PATH 

class FeatureDB:
    def __init__(self) -> None:
        self.aligner = FaceAligner(
            detector_weights=MODELS_PATH / 'face_detection_yunet_2023mar.onnx', 
            recognizer_weights=MODELS_PATH / 'face_recognition_sface_2021dec.onnx'
        )
        self.lbph = cv.face.LBPHFaceRecognizer_create()
        self.sface = SFace(threshold = 1.0313)
        self.lbph_crop = (100, 100) 
        self.db = {}

    @classmethod 
    def load(cls, path: str) -> "FeatureDB":
        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"Feature DB not found at: '{path}'")
        if p.suffix != ".npy":
            raise ValueError("Feature DB Path should end with '.npy'")
        
        # Instantiate the class and populate its instance dictionary
        instance = cls()
        instance.db = np.load(p, allow_pickle=True).item()
        print(f"[FeatureDB] Loaded {len(instance.db)} identities from '{p}'.")

        return instance

    def save(self, path: str) -> None:
        p = Path(path)
        if p.suffix != ".npy":
            raise ValueError("Feature DB Path should end with '.npy'")
            
        np.save(p, self.db, allow_pickle=True)
        print(f"[FeatureDB] Saved {len(self.db)} identities -> '{path}'")

    def _normalize_lbph(self, face_gray: np.ndarray) -> np.ndarray:
        img = cv.resize(face_gray, self.lbph_crop, interpolation=cv.INTER_AREA)
        img = np.float32(img) / 255.0
        alpha, tau, gamma = 0.1, 10.0, 0.2
        img = np.power(img, gamma)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = img / np.power(np.mean(np.power(np.abs(img), alpha)), 1.0 / alpha)
        img = img / np.power(np.mean(np.power(np.abs(img), tau)), 1.0 / tau)
        img = tau * np.tanh(img / tau)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        return np.uint8(img)

    def enroll(self, name: str, bgr_img: np.ndarray) -> Tuple[bool, bool]:
        aligned = self.aligner.align(bgr_img)
        if aligned is None:
            return False, False

        bbox = self.aligner.get_bbox(bgr_img.shape)
        if bbox is None:
            return False, True

        x, y, w, h = bbox
        face_bgr = bgr_img[y: y + h, x: x + w]
        if face_bgr.size == 0:
            return False, False

        face_gray = cv.cvtColor(face_bgr, cv.COLOR_BGR2GRAY)
        lbph_face_norm = self._normalize_lbph(face_gray)
        sface_embedding = self.sface.get_embedding(aligned).astype(np.float32)

        if name not in self.db:
            id = max((rec['id'] for rec in self.db.values()), default=-1) + 1
            self.db[name] = {
                "id": id,
                "lbph": [],
                "sface": []
            }

        self.db[name]['lbph'].append(lbph_face_norm)
        self.db[name]['sface'].append(sface_embedding)

        return True, True

    def batch_enroll(self, dataset: List[Tuple[str, str]]) -> dict:
        """
            Batch enrolls images from a list of file or directory paths.
            Expected path layout: '.../person_name/image.jpg'
        """
        pass

    def get_identities(self) -> List[str]:
        return list(self.db.keys())

    def get_identity_count(self) -> int:
        return len(self.get_identities())
