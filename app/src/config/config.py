from pathlib import Path

from ..engine.camera.webcam import WebCamera

try:
    from ..engine.camera.picam import PiCamera
except ImportError:
    PiCamera = None


BASE_PATH = Path(__file__).resolve().parent.parent

ICONS_PATH = BASE_PATH / "assets/icons"
KV_PATH = BASE_PATH / "ui"
MODELS_PATH = BASE_PATH / "models"
CONFIG_PATH = BASE_PATH / "config"
DB_PATH = Path(__file__).resolve().parents[2] / "db"

DB = {
    "La Salle Database": DB_PATH / "lasalledb.npy",
}
ENROLLMENT_ROOT = BASE_PATH / "engine" / "enrollment"
DATABASE_NAMES = tuple(DB)

CAMERA_OPTIONS = {
    "Default PC Camera": WebCamera,
}
if PiCamera is not None:
    CAMERA_OPTIONS["Raspberry Pi Camera"] = PiCamera
CAMERA_NAMES = tuple(CAMERA_OPTIONS)
