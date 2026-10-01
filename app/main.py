"""LS-Face Kivy application entry point."""
from __future__ import annotations

from src.engine.camera.manager import CameraManager
import argparse
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from src.engine.speech_transcriber import load_dotenv
    load_dotenv()
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picamera2", action="store_true", help="Use the Raspberry Pi CSI camera through Picamera2.")
    return parser.parse_args()


def run_kivy(options) -> int:
    from kivy.app import App
    from kivy.core.window import Window
    from kivy.uix.screenmanager import FadeTransition, ScreenManager

    from src.ui import load_design_system, tokens
    load_design_system()

    from src.views.home import HomeScreen
    from src.views.identities import ManageIdentitiesScreen
    from src.views.pose import PoseScreen
    from src.views.recognition import RecognitionScreen
    from src.views.voice_recognition import VoiceRecognitionScreen

    class LSFaceApp(App):
        title = "LS-Face"

        def on_start(self):
            from kivy.clock import Clock
            from src.engine.ui_watchdog import UIWatchdog
            self.ui_watchdog = UIWatchdog(Clock, self.root, APP_ROOT / "logs" / "ui-freeze.log")
            self.ui_watchdog.start()

        def build(self):
            self.camera_manager = CameraManager()
            self.pose_options = options
            Window.fullscreen = False
            Window.position = 'auto'
            Window.clearcolor = tokens.COLOR_BG
            Window.size = (1280, 720)
            manager = ScreenManager(transition=FadeTransition(duration=0.15))
            manager.add_widget(HomeScreen(name="home"))
            manager.add_widget(ManageIdentitiesScreen(name="database"))
            manager.add_widget(PoseScreen(name="pose_scan", mode="scan"))
            manager.add_widget(PoseScreen(name="pose_setup", mode="setup"))
            manager.add_widget(RecognitionScreen(name="recognition"))
            manager.add_widget(VoiceRecognitionScreen(name="voice_recognition"))
            manager.current = "home"
            return manager

        def on_stop(self):
            try:
                if self.root is not None:
                    self.root.get_screen("pose_scan").on_leave()
                    self.root.get_screen("pose_setup").on_leave()
                    self.root.get_screen("recognition").on_leave()
                    self.root.get_screen("voice_recognition").close_voice_service()
            finally:
                try:
                    self.camera_manager.close()
                finally:
                    watchdog = getattr(self, "ui_watchdog", None)
                    if watchdog is not None:
                        watchdog.close()

    LSFaceApp().run()
    return 0


def main() -> int:
    return run_kivy(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
