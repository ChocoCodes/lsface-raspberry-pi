import cv2 as cv
import numpy as np
from .camera import Camera 

class WebCamera(Camera):
    def __init__(self, camera_id: int = 0, res: tuple[int, int] | None = None):
        self.camera_id = camera_id
        self.res = res
        self.cap = None 

    def start(self) -> "WebCamera":
        self.cap = cv.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam: {self.camera_id}.")

        if self.res is not None:
            width, height = self.res
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        return self

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera not started.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame.")

        return frame

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
