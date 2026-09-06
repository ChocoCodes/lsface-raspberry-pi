# from .picam import PiCamera
from .webcam import WebCamera
from .camera import Camera

CAMERA_OPTIONS = {
    # "Raspberry Pi Camera" : PiCamera,
    "Default PC Camera" : WebCamera
}

def camera_factory(mode: str = "Default PC Camera") -> Camera:

    if mode not in CAMERA_OPTIONS:
        raise ValueError(f"Unsupported camera: {mode}. Supported cameras: {CAMERA_OPTIONS.keys()}")

    return CAMERA_OPTIONS[mode]()