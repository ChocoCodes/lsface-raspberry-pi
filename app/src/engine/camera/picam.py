import time
import numpy as np
from picamera2 import Picamera2
from .camera import Camera

class PiCamera(Camera):
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
        """Starts hardware stream."""
        print("[PiCamera] Starting camera stream...")
        self.picam2.start()
        time.sleep(self.warmup_s)
        print("[PiCamera] Camera ready.")
        return self

    def read(self) -> np.ndarray:
        """Captures a BGR frame matching the shared Camera contract."""
        return self.picam2.capture_array()
    
    def stop(self) -> None:
        """Stops the camera stream and releases hardware resources."""
        print("[PiCamera] Closing camera stream...")
        try:
            self.picam2.stop()
        finally:
            self.picam2.close()

    def __enter__(self) -> "PiCamera":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
