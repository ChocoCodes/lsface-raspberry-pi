from pathlib import Path 
# from ..engine.camera.picam import PiCamera
from ..engine.camera.webcam import WebCamera

BASE_PATH =  Path(__file__).resolve().parent.parent

ICONS_PATH = BASE_PATH / 'assets/icons'
KV_PATH =  BASE_PATH / 'ui'
MODELS_PATH =  BASE_PATH / 'models'
CONFIG_PATH = BASE_PATH / 'config'
DB_PATH = Path(__file__).resolve().parents[2] / 'db'

DB = {
    "La Salle Database" : DB_PATH / 'lasalledb.npy'
}

CAMERA_OPTIONS = {
    # "Raspberry Pi Camera" : PiCamera,
    "Default PC Camera" : WebCamera
}
