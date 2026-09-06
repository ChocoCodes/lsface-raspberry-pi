"""LS-Face Kivy application entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picamera2", action="store_true", help="Use the Raspberry Pi CSI camera through Picamera2.")
    return parser.parse_args()


def run_kivy(options) -> int:
    from kivy.app import App
    from kivy.core.window import Window
    from kivy.uix.screenmanager import FadeTransition, ScreenManager

    from src.views.database import DatabaseScreen
    from src.views.home import HomeScreen
    from src.views.pose import PoseScreen
    from src.views.recognition import RecognitionScreen
    from src.views.voice_recognition import VoiceRecognitionScreen

    class LSFaceApp(App):
        title = "LS-Face"

        def build(self):
            self.pose_options = options
            Window.clearcolor = (0.045, 0.063, 0.094, 1)
            Window.size = (1280, 720)
            manager = ScreenManager(transition=FadeTransition(duration=0.15))
            manager.add_widget(HomeScreen(name="home"))
            manager.add_widget(DatabaseScreen(name="database"))
            manager.add_widget(PoseScreen(name="pose_scan", mode="scan"))
            manager.add_widget(PoseScreen(name="pose_setup", mode="setup"))
            manager.add_widget(RecognitionScreen(name="recognition"))
            manager.add_widget(VoiceRecognitionScreen(name="voice_recognition"))
            manager.current = "home"
            return manager

    LSFaceApp().run()
    return 0


def main() -> int:
    return run_kivy(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
