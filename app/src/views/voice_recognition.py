from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from src.config.config import KV_PATH


Builder.load_file(str(KV_PATH / "voice_recognition.kv"))


class VoiceRecognitionScreen(Screen):
    """Static placeholder for the voice-recognition enrollment step."""

    def go_back(self):
        self.manager.current = "home"
