from kivy.animation import Animation
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import BooleanProperty, ListProperty

from src.config.config import KV_PATH
from src.engine.database.database_manager import DatabaseManager


class SidebarBackdrop(Widget):
    """Semi-transparent backdrop that only intercepts touches when sidebar is open."""

    def on_touch_down(self, touch):
        parent = self.parent
        if not getattr(parent, "sidebar_open", False):
            return False
        if self.collide_point(*touch.pos):
            sidebar = getattr(parent.ids, "sidebar", None) if hasattr(parent, "ids") else None
            if sidebar and sidebar.collide_point(*touch.pos):
                return False
            parent.close_sidebar()
            return True
        return False


Builder.load_file(str(KV_PATH / 'home.kv'))

class HomeScreenView(FloatLayout):
    database_names = ListProperty([])
    sidebar_open = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._database_ids = {}
        self._syncing_database_selector = False
        self.bind(size=self._on_resize)

    def _on_resize(self, *_):
        if hasattr(self, 'ids') and 'sidebar' in self.ids:
            if self.sidebar_open:
                self.ids.sidebar.x = self.width - self.ids.sidebar.width
            else:
                self.ids.sidebar.x = self.width

    def toggle_sidebar(self):
        if self.sidebar_open:
            self.close_sidebar()
        else:
            self.open_sidebar()

    def open_sidebar(self):
        self.sidebar_open = True
        sidebar = self.ids.sidebar
        backdrop = self.ids.sidebar_backdrop
        Animation.stop_all(sidebar)
        Animation.stop_all(backdrop)
        anim_sidebar = Animation(x=self.width - sidebar.width, d=0.25, t="out_cubic")
        anim_backdrop = Animation(opacity=1.0, d=0.25, t="out_cubic")
        anim_sidebar.start(sidebar)
        anim_backdrop.start(backdrop)

    def close_sidebar(self):
        self.sidebar_open = False
        sidebar = self.ids.sidebar
        backdrop = self.ids.sidebar_backdrop
        Animation.stop_all(sidebar)
        Animation.stop_all(backdrop)
        anim_sidebar = Animation(x=self.width, d=0.22, t="out_cubic")
        anim_backdrop = Animation(opacity=0.0, d=0.22, t="out_cubic")
        anim_sidebar.start(sidebar)
        anim_backdrop.start(backdrop)

    def on_kv_post(self, *_args):
        self.refresh_databases()

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
        from src.ui import tokens
        content = BoxLayout(orientation="vertical", padding=16, spacing=10)
        name_input = TextInput(
            hint_text="Database name",
            multiline=False,
            size_hint_y=None,
            height="42dp",
            background_normal="",
            background_active="",
            background_color=tokens.COLOR_SURFACE_2,
            foreground_color=tokens.COLOR_TEXT_PRIMARY,
            hint_text_color=tokens.COLOR_TEXT_MUTED,
            cursor_color=tokens.COLOR_ACCENT,
            padding=["12dp", "10dp"],
        )
        error_label = Label(text="", color=tokens.COLOR_ERROR, font_size="12sp")
        actions = BoxLayout(size_hint_y=None, height="40dp", spacing=10)
        cancel = Button(
            text="Cancel",
            background_normal="",
            background_color=tokens.COLOR_SURFACE_2,
            color=tokens.COLOR_TEXT_PRIMARY,
        )
        create = Button(
            text="Create",
            background_normal="",
            background_color=tokens.COLOR_ACCENT,
            color=tokens.COLOR_TEXT_ON_ACCENT,
            bold=True,
        )
        actions.add_widget(cancel)
        actions.add_widget(create)
        content.add_widget(Label(text="Create an isolated database for a fresh test.", color=tokens.COLOR_TEXT_SECONDARY))
        content.add_widget(name_input)
        content.add_widget(error_label)
        content.add_widget(actions)
        popup = Popup(
            title="New Database",
            title_color=tokens.COLOR_TEXT_PRIMARY,
            separator_color=tokens.COLOR_ACCENT,
            background="",
            background_color=tokens.COLOR_SURFACE_1,
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
