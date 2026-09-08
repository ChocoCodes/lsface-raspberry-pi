from .camera import Camera
from src.config.config import CAMERA_OPTIONS 

def camera_factory(mode: str = "Default PC Camera") -> Camera:
    if mode not in CAMERA_OPTIONS:
        raise ValueError(f"Unsupported camera: {mode}. Supported cameras: {CAMERA_OPTIONS.keys()}")

    return CAMERA_OPTIONS[mode]()