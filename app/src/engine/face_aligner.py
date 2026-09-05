import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional

class FaceAligner:
    """
        Detects and aligns faces from images using OpenCV's YuNet and SFace models.

        This class leverages YuNet (`cv.FaceDetectorYN`) for face detection and bounding 
        box localization, and SFace (`cv.FaceRecognizerSF`) to perform affine transformation 
        (aligning facial landmarks like eyes, nose, and mouth) and cropping. When multiple 
        faces are detected in an image, it defaults to processing the largest face by area.

        Attributes:
            input_size (tuple[int, int]): Initial (width, height) resolution expected by 
                the detector.
            threshold (float): Minimum confidence score threshold for face detection [0.0 to 1.0].
            detector (cv.FaceDetectorYN): OpenCV YuNet face detector instance.
            recognizer (cv.FaceRecognizerSF): OpenCV SFace recognizer and aligner instance.

        Example:
            >>> aligner = FaceAligner(
            ...     detector_weights="models/face_detection_yunet.onnx",
            ...     recognizer_weights="models/face_recognition_sface.onnx"
            ... )
            >>> frame = cv.imread("person.jpg")
            >>> aligned_face = aligner.align(frame)
            >>> if aligned_face is not None:
            ...     cv.imwrite("aligned_face.jpg", aligned_face)
    """
    
    def __init__(
            self, 
            detector_weights = "models/face_detection_yunet_2023mar.onnx", 
            recognizer_weights="models/face_recognition_sface_2021dec.onnx", 
            input_size=(320, 320),
            threshold=0.6,
        ):
        self.input_size = input_size
        self.threshold = threshold
        self.detector = cv.FaceDetectorYN.create(detector_weights, "", self.input_size, score_threshold=self.threshold)
        self.recognizer = cv.FaceRecognizerSF.create(recognizer_weights, "")
        self.faces = None

    def detect(self, bgr_image: np.ndarray) -> List[np.ndarray]:
        """
            Detect all sufficiently large faces.
            Returns:
                List of YuNet face metadata arrays.
        """
        if bgr_image is None or bgr_image.size == 0:
            self.faces = None 
            return []
        
        h, w = bgr_image.shape[:2]
        self.detector.setInputSize((w, h))

        retval, detected_faces = self.detector.detect(bgr_image)
        if not retval or detected_faces is None:
            self.faces = None
            return []

        valid = []
        for face in detected_faces:
            x, y, bw, bh = face[:4]

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            face[:4] = (x, y, bw, bh)
            valid.append(face)

        self.faces = valid
        return valid

    def detect_and_align(self, bgr_image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        faces = self.detect(bgr_image)
        if not faces:
            return []

        results = []

        for face in faces:
            aligned = self.recognizer.alignCrop(bgr_image, face)
            if aligned is not None:
                results.append((face, aligned))

        return results
    
    def get_bbox(self, face: np.ndarray, img_shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """
            Returns a clamped (x, y, w, h) bounding box.
        """
        if face is None:
            return None 
        
        h, w = img_shape[:2]
        x, y, bw, bh = [int(v) for v in face[:4]]

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        return x, y, bw, bh