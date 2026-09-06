from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from src.config.config import KV_PATH, DB
from src.engine.database.feature_db import FeatureDB
from pathlib import Path 
from src.engine.database.database_manager import DatabaseManager 

Builder.load_file(str(KV_PATH / 'home.kv'))

class HomeScreenView(BoxLayout):
    def on_camera_changed(self, camera_name):
        print(f"[EVENT] Selected Camera: {camera_name}")

    def on_database_changed(self, db_name):
        print(f"[EVENT] Selected Database: {db_name}")
        self.parent.load_database(db_name)

    def open_add_identity(self):
        print("[NAVIGATION] Transition to: Add Identity Screen")

    def open_recognition(self):
        app = App.get_running_app()
        recognition_screen = app.root.get_screen("recognition")
        recognition_screen.camera_mode = self.ids.camera_selector.text
        app.root.current = "recognition"

    def open_view_identities(self):
        print("[NAVIGATION] Transition to: View Identities Screen")
        app = App.get_running_app()
        app.root.current = "identities"


class HomeScreen(Screen):
    """Screen wrapper so HomeScreenView (a plain BoxLayout, per home.kv's
    <HomeScreenView> rule) can live inside a ScreenManager unchanged."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_db = None
        self.add_widget(HomeScreenView())

        self.load_database("La Salle Database")

    def load_database(self, db_name: str):
        app = App.get_running_app()
        db_path = DB.get(db_name)

        if db_path is None:
            print(f"[DATABASE] Unknown database: {db_name}")
            return

        self.feature_db = DatabaseManager.load(db_path)
        print(f"Loaded {self.feature_db.get_identity_count()} identities from {db_name}")