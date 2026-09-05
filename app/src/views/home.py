from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from src.config.config import KV_PATH

Builder.load_file(str(KV_PATH / 'home.kv'))


class HomeScreenView(BoxLayout):
    def on_camera_changed(self, camera_name):
        print(f"[EVENT] Selected Camera: {camera_name}")

    def on_database_changed(self, db_name):
        print(f"[EVENT] Selected Database: {db_name}")

    def open_add_identity(self):
        print("[NAVIGATION] Transition to: Add Identity Screen")

    def open_recognition(self):
        app = App.get_running_app()
        recognition_screen = app.root.get_screen("recognition")
        recognition_screen.camera_index = self._selected_camera_index()
        app.root.current = "recognition"

    def open_view_identities(self):
        print("[NAVIGATION] Transition to: View Identities Screen")

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
        self.add_widget(HomeScreenView())