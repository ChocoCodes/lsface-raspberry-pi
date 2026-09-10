import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from src.config.config import KV_PATH
from src.engine.database.database_manager import DatabaseManager

Builder.load_file(str(KV_PATH / "identities.kv"))

class ManageIdentitiesView(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._delete_busy = False

    def populate(self, feature_db):
        self.ids.identity_table.clear_widgets()
        if feature_db is None:
            self.ids.identity_count.text = "0 identities"
            return

        self.ids.identity_count.text = f"{feature_db.get_identity_count()} identities"

        COLUMNS = ["ID", "Name", "LBPH", "SFace", "Actions"]
        for col in COLUMNS:
            self.ids.identity_table.add_widget(Label(text=col, bold=True, color=(0, 0, 0, 1), size_hint_y=None, height=40))
        for name, record in sorted(feature_db.db.items(), key=lambda item: (int(item[1]["id"]), item[0])):
            self.ids.identity_table.add_widget(Label(text=str(record["id"]), color=(0, 0, 0, 1), size_hint_y=None, height=40))
            self.ids.identity_table.add_widget(Label(text=name, color=(0, 0, 0, 1), size_hint_y=None, height=40))
            self.ids.identity_table.add_widget(Label(text=str(len(record["lbph"])), color=(0, 0, 0, 1), size_hint_y=None, height=40))
            self.ids.identity_table.add_widget(Label(text=str(len(record["sface"])), color=(0, 0, 0, 1), size_hint_y=None, height=40))
            delete_button = Button(text="Delete", size_hint_y=None, height=40)
            delete_button.bind(on_release=lambda _button, identity=name: self.confirm_delete(identity))
            self.ids.identity_table.add_widget(delete_button)

    def confirm_delete(self, name: str):
        if self._delete_busy:
            return

        content = BoxLayout(orientation="vertical", padding=12, spacing=12)
        content.add_widget(Label(text=f"Delete {name!r}?\nThis rebuilds live recognition."))
        actions = BoxLayout(size_hint_y=None, height=44, spacing=8)
        cancel = Button(text="Cancel")
        confirm = Button(text="Delete")
        actions.add_widget(cancel)
        actions.add_widget(confirm)
        content.add_widget(actions)
        popup = Popup(title="Delete identity", content=content, size_hint=(None, None), size=(420, 190), auto_dismiss=False)
        cancel.bind(on_release=popup.dismiss)

        def accept(_button):
            popup.dismiss()
            self.delete_identity(name)

        confirm.bind(on_release=accept)
        popup.open()

    def delete_identity(self, name: str):
        if self._delete_busy:
            return
        self._delete_busy = True
        self.ids.identity_count.text = f"Deleting {name}…"

        app = App.get_running_app()
        home = app.root.get_screen("home")
        database_id = getattr(home, "database_id", None) or DatabaseManager.selected_database_id()

        def worker():
            try:
                database = DatabaseManager.delete(name, database_id=database_id)
                result = (database, None)
            except Exception as exc:
                result = (None, exc)
            Clock.schedule_once(lambda _dt: self._finish_delete(result), 0)

        threading.Thread(target=worker, name="identity-delete-release-builder", daemon=True).start()

    def _finish_delete(self, result):
        self._delete_busy = False
        database, error = result
        if error is not None:
            self.ids.identity_count.text = f"Delete failed: {error}"
            home = App.get_running_app().root.get_screen("home")
            self.populate(home.feature_db)
            return

        home = App.get_running_app().root.get_screen("home")
        home.feature_db = database
        self.populate(database)

    def go_back(self):
        if self._delete_busy:
            return
        App.get_running_app().root.current = "home"
        
class ManageIdentitiesScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(ManageIdentitiesView())

    def on_pre_enter(self):
        app = App.get_running_app()
        home_screen = app.root.get_screen('home')
        print(f"[IDENTITIES] Home screen: {home_screen}")
        print(f"[IDENTITIES] Feature DB: {home_screen.feature_db}")
        self.children[0].populate(home_screen.feature_db)
