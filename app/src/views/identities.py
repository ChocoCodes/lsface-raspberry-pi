import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.metrics import dp
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from src.config.config import KV_PATH
from src.engine.database.database_manager import DatabaseManager
from src.ui import tokens

Builder.load_file(str(KV_PATH / "identities.kv"))


class ManageIdentitiesView(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._delete_busy = False

    def populate(self, feature_db):
        self.ids.identity_list.clear_widgets()
        if feature_db is None or feature_db.get_identity_count() == 0:
            if hasattr(self.ids, "identity_badge"):
                self.ids.identity_badge.badge_text = "0 identities"
                self.ids.identity_badge.badge_status = "muted"
            empty_box = BoxLayout(orientation="vertical", size_hint_y=None, height="200dp", padding="32dp", spacing="12dp")
            empty_title = Label(
                text="No Enrolled Identities",
                font_name=tokens.FONT_BOLD,
                font_size=f"{tokens.FONT_SIZE_SUBTITLE}sp",
                color=tokens.COLOR_TEXT_PRIMARY,
                halign="center",
                valign="middle",
                size_hint_y=None,
                height="30dp",
            )
            empty_desc = Label(
                text="The selected database has no facial templates enrolled yet.\nTap 'Add Identity' on the home dashboard to enroll a user.",
                font_name=tokens.FONT_REGULAR,
                font_size=f"{tokens.FONT_SIZE_BODY}sp",
                color=tokens.COLOR_TEXT_MUTED,
                halign="center",
                valign="middle",
                size_hint_y=None,
                height="44dp",
            )
            empty_box.add_widget(empty_title)
            empty_box.add_widget(empty_desc)
            self.ids.identity_list.add_widget(empty_box)
            return

        count = feature_db.get_identity_count()
        if hasattr(self.ids, "identity_badge"):
            self.ids.identity_badge.badge_text = f"{count} {'identity' if count == 1 else 'identities'}"
            self.ids.identity_badge.badge_status = "ready"

        for name, record in sorted(feature_db.db.items(), key=lambda item: (int(item[1]["id"]), item[0])):
            row = Factory.IdentityRowItem()
            row.id_text = str(record["id"])
            row.name_text = name
            row.lbph_text = f"{len(record['lbph'])} samples"
            row.sface_text = f"{len(record['sface'])} embeddings"
            if hasattr(row.ids, "btn_delete"):
                row.ids.btn_delete.bind(on_release=lambda _btn, identity=name: self.confirm_delete(identity))
            self.ids.identity_list.add_widget(row)

    def confirm_delete(self, name: str):
        if self._delete_busy:
            return

        content = BoxLayout(orientation="vertical", padding="16dp", spacing="14dp")
        msg = Label(
            text=f"Delete template for {name!r}?\nThis removes all LBPH samples and SFace embeddings and triggers a model release rebuild.",
            font_name=tokens.FONT_REGULAR,
            font_size=f"{tokens.FONT_SIZE_BODY}sp",
            color=tokens.COLOR_TEXT_SECONDARY,
            halign="center",
            valign="middle",
        )
        msg.bind(size=lambda inst, val: setattr(inst, "text_size", (val[0] - dp(16), None)))
        actions = BoxLayout(size_hint_y=None, height="44dp", spacing="12dp")
        cancel = Factory.SecondaryButton(button_text="Cancel")
        confirm = Factory.DangerButton(button_text="Delete Identity")
        actions.add_widget(cancel)
        actions.add_widget(confirm)

        content.add_widget(msg)
        content.add_widget(actions)

        popup = Popup(
            title=f"Delete Identity: {name}",
            title_font=tokens.FONT_BLACK,
            title_size=f"{tokens.FONT_SIZE_SUBTITLE}sp",
            title_color=tokens.COLOR_TEXT_PRIMARY,
            separator_color=tokens.COLOR_ERROR,
            background="",
            background_color=tokens.COLOR_SURFACE_1,
            content=content,
            size_hint=(None, None),
            size=("480dp", "220dp"),
            auto_dismiss=False,
        )
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
        if hasattr(self.ids, "identity_badge"):
            self.ids.identity_badge.badge_text = f"Deleting {name}…"
            self.ids.identity_badge.badge_status = "warning"

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
            if hasattr(self.ids, "identity_badge"):
                self.ids.identity_badge.badge_text = "Delete failed"
                self.ids.identity_badge.badge_status = "error"
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
        self.view = ManageIdentitiesView()
        self.add_widget(self.view)

    def on_pre_enter(self):
        home_screen = None
        if self.manager and self.manager.has_screen("home"):
            home_screen = self.manager.get_screen("home")
        else:
            app = App.get_running_app()
            if app and getattr(app, "root", None) and hasattr(app.root, "get_screen"):
                home_screen = app.root.get_screen("home")
        if home_screen is not None:
            self.view.populate(home_screen.feature_db)
