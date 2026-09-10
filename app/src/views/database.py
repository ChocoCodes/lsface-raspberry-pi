from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from src.config.config import KV_PATH


Builder.load_file(str(KV_PATH / "database.kv"))


class DatabaseScreen(Screen):
    def go_back(self):
        self.manager.current = "home"
