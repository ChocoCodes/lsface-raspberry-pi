from pathlib import Path

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.properties import StringProperty
from src.config.config import KV_PATH

from src.pose_detection.flow import pnp_profile_problem
from src.pose_detection.head_pose import load_config

Builder.load_file(str(KV_PATH / 'home.kv'))


class HomeScreenView(BoxLayout):
    pose_status = StringProperty("Checking device setup…")

    def on_kv_post(self, *_args):
        self.refresh_pose_status()

    def on_camera_changed(self, camera_name):
        print(f"[EVENT] Selected Camera: {camera_name}")

    def on_database_changed(self, db_name):
        print(f"[EVENT] Selected Database: {db_name}")

    def open_add_identity(self):
        app = App.get_running_app()
        pose_screen = app.root.get_screen("pose_scan")
        pose_screen.camera_index = self._selected_camera_index()
        app.root.current = "pose_scan"

    def open_recognition(self):
        app = App.get_running_app()
        recognition_screen = app.root.get_screen("recognition")
        recognition_screen.camera_index = self._selected_camera_index()
        app.root.current = "recognition"

    def open_view_identities(self):
        App.get_running_app().root.current = "database"

    def open_pose_setup(self):
        app = App.get_running_app()
        pose_screen = app.root.get_screen("pose_setup")
        pose_screen.camera_index = self._selected_camera_index()
        app.root.current = "pose_setup"

    def refresh_pose_status(self):
        app_root = Path(__file__).resolve().parents[2]
        profile = app_root / "config" / "head_pose.local.json"
        try:
            config = load_config(profile if profile.exists() else app_root / "config" / "head_pose.json")
            config["backend"] = "yunet_geometry"
            self.pose_status = "POSE READY" if pnp_profile_problem(config, "yunet_geometry") is None else "SETUP REQUIRED"
        except Exception:
            self.pose_status = "SETUP REQUIRED"

    def _selected_camera_index(self) -> int:
        """Parse the Spinner text ('Camera 0 (Built-in)') into a cv index."""
        text = self.ids.camera_selector.text
        for token in text.split():
            if token.isdigit():
                return int(token)
        return 0


class HomeScreen(Screen):
    """Screen wrapper so HomeScreenView (a plain BoxLayout, per home.kv's
    <HomeScreenView> rule) can live inside a ScreenManager unchanged."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.view = HomeScreenView()
        self.add_widget(self.view)

    def on_pre_enter(self, *_args):
        self.view.refresh_pose_status()
