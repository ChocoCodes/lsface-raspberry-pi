import cv2 as cv
import numpy as np

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
            threshold=0.6
        ):
        self.input_size = input_size
        self.threshold = threshold
        self.detector = cv.FaceDetectorYN.create(detector_weights, "", self.input_size, score_threshold=self.threshold)
        self.recognizer = cv.FaceRecognizerSF.create(recognizer_weights, "")
        self.face_meta = None
    
    def align(self, bgr_image: np.ndarray):
        h, w = bgr_image.shape[:2]
        self.detector.setInputSize((w, h))
        retval, faces = self.detector.detect(bgr_image)

        if not retval or faces is None:
            return None

        # NOTE: change this if multiple faces are allowed
        areas = faces[:, 2] * faces[:, 3]
        self.face_meta = faces[np.argmax(areas)]

        return self.recognizer.alignCrop(bgr_image, self.face_meta)
    
    def get_bbox(self, img_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns clamped (x, y, w, h) int bbox from the last detection."""
        if self.face_meta is None:
            return None 
        
        h, w = img_shape[:2]
        x, y, bw, bh = [int(v) for v in self.face_meta[:4]]

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        return x, y, bw, bh