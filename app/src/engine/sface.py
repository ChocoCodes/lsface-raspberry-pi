import cv2 as cv
import numpy as np
from src.config.config import MODELS_PATH

class SFace:
    """
        Extracts 128-dimensional facial feature embeddings and computes similarity scores using OpenCV SFace.

        This class wraps OpenCV's `cv.FaceRecognizerSF` model to generate feature vectors 
        from aligned facial crops and evaluate pair-wise face matching metrics (such as L2 
        distance or Cosine similarity).

        Attributes:
            recognizer (cv.FaceRecognizerSF): OpenCV SFace feature extractor instance.

        Example:
            >>> sface = SFace(w_recognizer="models/face_recognition_sface_2021dec.onnx")
            >>> embedding1 = sface.get_embedding(aligned_face_1)
            >>> embedding2 = sface.get_embedding(aligned_face_2)
            >>> l2_dist = sface.similarity(embedding1, embedding2)
            >>> is_match = l2_dist <= threshold (1.128 or custom-defined)
    """

    def __init__(
        self, 
        w_recognizer=str(MODELS_PATH / "face_recognition_sface_2021dec.onnx"),
        threshold: float = 1.128
    ):
        print("[Initializing] SFace via OpenCV DNN Runtime Engine...")
        self.recognizer = cv.FaceRecognizerSF.create(w_recognizer, "")
        self.threshold = threshold

    def get_embedding(self, bgr_face): 
        """ Image color space should be BGR. """
        embedding = self.recognizer.feature(bgr_face)
        return embedding.flatten()

    def similarity(self, vec_a, vec_b):
        """ 
        Natively leverages OpenCV's SFace Match metrics.
        Returns the optimized NormL2 distance matching 1.128 LFW standard or custom-defined threshold.
        """
        v1 = vec_a.reshape(1, -1).astype(np.float32)
        v2 = vec_b.reshape(1, -1).astype(np.float32)
        return float(self.recognizer.match(v1, v2, cv.FaceRecognizerSF_FR_NORM_L2))
    
    def match(
        self,
        query_vector: np.ndarray,
        gallery: np.ndarray,
        labels: np.ndarray
    ) -> tuple[str, float]:
        """
            Matches a live query embedding against the loaded .npy feature database.
            Args:
                query_vector: 1D NumPy embedding array of the live query face.
                feature_db: Loaded .npy dictionary containing identity keys and 2D embedding arrays.
            Returns:
                Tuple of (matched_person_name, minimum_distance_found).
        """
        query_vector /= np.linalg.norm(query_vector)
        diffs = gallery - query_vector 
        dists = np.linalg.norm(diffs, axis=1)

        min_idx = np.argmin(dists)
        min_dist = float(dists[min_idx])

        if min_dist <= self.threshold:
            return str(labels[min_idx]), min_dist

        return "Unknown", min_dist