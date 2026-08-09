import time
import cv2 as cv
import numpy as np 
from picamera2 import Picamera2 

class PiCamera:
    """
        Wrapper class for PiCamera2 video stream and frame capture.
    """
    def __init__(self, res: tuple[int, int] = (1280, 720), warmup_s: float = 1.5) -> None:
        """
            Configures the camera sensor.
            Args:
                res: Tuple of (width, height). Defaults to (1280, 720).
                warmup_s: Time to wait for sensor auto-exposure and white balance to stabilize.
        """
        self.res = res 
        self.warmup_s = warmup_s
        self.picam2 = Picamera2()

        # Configure stream for raw RGB array capture
        conf = self.picam2.create_video_configuration(
            main={
                "size": self.res, 
                "format": "RGB888"
            }
        )

        self.picam2.configure(conf) 
    
    def start(self) -> "PiCamera":
        """Starts hardware stream and waits for sensor warm-up."""
        print("[PiCamera] Starting camera stream...")
        self.picam2.start()
        time.sleep(self.warmup_s)
        print("[PiCamera] Camera ready.")
        return self

    def capture_bgr(self) -> np.ndarray:
        """Captures a frame and converts RGB to OpenCV-compatible BGR array."""
        return self.picam2.capture_array()
    
    def stop(self) -> None:
        """Stops the camera stream and releases hardware resources."""
        print("[PiCamera] Closing camera stream...")
        self.picam2.stop()

    def __enter__(self) -> "PiCamera":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
    