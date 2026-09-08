from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from src.config.config import KV_PATH

Builder.load_file(str(KV_PATH / "identities.kv"))

class ManageIdentitiesView(BoxLayout):
    def populate(self, feature_db):
        self.ids.identity_table.clear_widgets()
        if feature_db is None:
            self.ids.identity_count.text = "0 identities"
            return 
        
        self.ids.identity_count.text = f"{feature_db.get_identity_count()} identities"

        COLUMNS = ["ID", "Name", "LBPH", "SFace"]
        for col in COLUMNS:
            self.ids.identity_table.add_widget(Label(text=col, bold=True, color=(0, 0, 0, 1), size_hint_y=None, height=40))
        for name, record in feature_db.db.items():
            self.ids.identity_table.add_widget(Label(text=str(record['id']), color=(0, 0, 0, 1), size_hint_y=None, height=40))
            self.ids.identity_table.add_widget(Label(text=name, color=(0, 0, 0, 1), size_hint_y=None, height=40))
            self.ids.identity_table.add_widget(Label(text=str(len(record['lbph'])), color=(0, 0, 0, 1), size_hint_y=None, height=40))
            self.ids.identity_table.add_widget(Label(text=str(len(record['sface'])), color=(0, 0, 0, 1), size_hint_y=None, height=40))

    def go_back(self):
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