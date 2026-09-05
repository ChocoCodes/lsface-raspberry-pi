from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, FadeTransition
 
from src.views.home import HomeScreen
from src.views.recognition import RecognitionScreen
 
 
class LSFaceApp(App):
    title = "LS-Face"
 
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        Window.size = (1024, 720)
 
        sm = ScreenManager(transition=FadeTransition(duration=0.15))
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(RecognitionScreen(name="recognition"))
        sm.current = "home"
        return sm
 
 
if __name__ == "__main__":
    LSFaceApp().run()