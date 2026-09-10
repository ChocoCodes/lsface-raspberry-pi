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
# Local catalog data is kept beside the source DB, while the built-in database
# retains its historical release root above.
LOCAL_DATABASE_ROOT = DB_PATH / "local"
DATABASE_CATALOG_PATH = LOCAL_DATABASE_ROOT / "catalog.json"
CUSTOM_DATABASE_ROOT = LOCAL_DATABASE_ROOT / "databases"
DATABASE_NAMES = tuple(DB)

CAMERA_OPTIONS = {
    "Default PC Camera": WebCamera,
}
if PiCamera is not None:
    CAMERA_OPTIONS["Raspberry Pi Camera"] = PiCamera
CAMERA_NAMES = tuple(CAMERA_OPTIONS)
