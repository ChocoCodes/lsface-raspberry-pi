from pathlib import Path

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.lang import Builder
from kivy.properties import ListProperty, StringProperty

from src.config.config import KV_PATH
from src.engine.database.database_manager import DatabaseManager

from src.pose_detection.flow import pnp_profile_problem
from src.pose_detection.head_pose import load_config

Builder.load_file(str(KV_PATH / 'home.kv'))

class HomeScreenView(BoxLayout):
    pose_status = StringProperty("Checking device setup…")
    database_names = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._database_ids = {}
        self._syncing_database_selector = False

    def on_kv_post(self, *_args):
        self.refresh_databases()
        self.refresh_pose_status()

    def on_camera_changed(self, camera_name):
        print(f"[EVENT] Selected Camera: {camera_name}")

    def on_database_changed(self, db_name):
        if self._syncing_database_selector:
            return
        print(f"[EVENT] Selected Database: {db_name}")
        database_id = self._database_ids.get(db_name)
        if database_id is not None and self.parent is not None:
            if not self.parent.load_database(database_id=database_id):
                self.refresh_databases(self.parent.database_id)

    def refresh_databases(self, selected_id=None):
        databases = DatabaseManager.list_databases()
        self._database_ids = {database.name: database.id for database in databases}
        names = [database.name for database in databases]
        if selected_id is None:
            selected_id = DatabaseManager.selected_database_id()
        selected_name = next(
            (database.name for database in databases if database.id == selected_id),
            names[0] if names else "",
        )
        self._syncing_database_selector = True
        try:
            self.database_names = names
            self.ids.db_selector.text = selected_name
        finally:
            self._syncing_database_selector = False

    def open_new_database(self):
        content = BoxLayout(orientation="vertical", padding=14, spacing=10)
        name_input = TextInput(
            hint_text="Database name",
            multiline=False,
            size_hint_y=None,
            height="42dp",
        )
        error_label = Label(text="", color=(0.75, 0.20, 0.20, 1))
        actions = BoxLayout(size_hint_y=None, height="44dp", spacing=8)
        cancel = Button(text="Cancel")
        create = Button(text="Create")
        actions.add_widget(cancel)
        actions.add_widget(create)
        content.add_widget(Label(text="Create an isolated database for a fresh test."))
        content.add_widget(name_input)
        content.add_widget(error_label)
        content.add_widget(actions)
        popup = Popup(
            title="New database",
            content=content,
            size_hint=(None, None),
            size=(460, 240),
            auto_dismiss=False,
        )
        cancel.bind(on_release=popup.dismiss)

        def create_database(_button):
            try:
                database = DatabaseManager.create_database(name_input.text)
            except Exception as exc:
                error_label.text = str(exc)
                return
            popup.dismiss()
            self.parent.load_database(database_id=database.id)

        create.bind(on_release=create_database)
        popup.open()

    def open_add_identity(self):
        app = App.get_running_app()
        pose_screen = app.root.get_screen("pose_scan")
        pose_screen.camera_mode = self.ids.camera_selector.text
        pose_screen.database_id = self.parent.database_id
        app.root.current = "pose_scan"

    def open_recognition(self):
        app = App.get_running_app()
        recognition_screen = app.root.get_screen("recognition")
        recognition_screen.camera_mode = self.ids.camera_selector.text
        recognition_screen.configure_session(
            database_id=self.parent.database_id,
            expected_identity_name="",
            session_id=None,
        )
        app.root.current = "recognition"

    def open_view_identities(self):
        App.get_running_app().root.current = "database"

    def open_pose_setup(self):
        app = App.get_running_app()
        pose_screen = app.root.get_screen("pose_setup")
        pose_screen.camera_mode = self.ids.camera_selector.text
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



class HomeScreen(Screen):
    """Screen wrapper so HomeScreenView (a plain BoxLayout, per home.kv's
    <HomeScreenView> rule) can live inside a ScreenManager unchanged."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.view = HomeScreenView()
        self.add_widget(self.view)
        self.feature_db = None
        self.database_id = DatabaseManager.selected_database_id()
        self.view.refresh_databases(self.database_id)
        self.load_database(database_id=self.database_id)

    def on_pre_enter(self, *_args):
        selected_id = DatabaseManager.selected_database_id()
        if selected_id != self.database_id:
            self.load_database(database_id=selected_id)
        else:
            self.view.refresh_databases(selected_id)
        self.view.refresh_pose_status()

    def load_database(self, db_name: str | None = None, *, database_id: str | None = None):
        if database_id is None:
            databases = DatabaseManager.list_databases()
            selected = next((database for database in databases if database.name == db_name), None)
            if selected is None:
                print(f"[DATABASE] Unknown database: {db_name}")
                return False
            database_id = selected.id

        try:
            database = DatabaseManager.resolve_database(database_id)
            feature_db = DatabaseManager.ensure_release(database_id=database.id)
            DatabaseManager.select_database(database.id)
        except Exception as exc:
            print(f"[DATABASE] Could not load {database_id}: {exc}")
            return False

        self.database_id = database.id
        self.feature_db = feature_db
        self.view.refresh_databases(database.id)
        print(f"Loaded {self.feature_db.get_identity_count()} identities from {database.name}")
        return True
