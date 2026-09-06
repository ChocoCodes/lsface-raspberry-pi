import time
import cv2 as cv

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen

from src.config.config import KV_PATH
from src.engine.camera.factory import camera_factory
from src.engine.build_cascade import (
    CAM_INDEX as DEFAULT_CAM_INDEX,
    RES_DEFAULT,
    SETUPS,
    LOGGER,
    build_selected_cascade,
    configure_logging,
    draw_overlay,
    normalize_results,
)

Builder.load_file(str(KV_PATH / 'recognition.kv'))

CHOICE = "2"  # new setup / r3_n8_g6x6, quality-first -> recognition mode


class RecognitionScreen(Screen):
    camera_mode = StringProperty("Default PC Camera")
    status_text = StringProperty("Initializing...")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera = None
        self.cascade = None
        self.log_file = None
        self.frame_count = 0
        self._update_event = None

    # --- Screen lifecycle ---------------------------------------------
    def on_enter(self, *args):
        # Defer heavy init so the screen transition isn't blocked.
        Clock.schedule_once(self._initialize, 0)

    def on_leave(self, *args):
        if self._update_event is not None:
            self._update_event.cancel()
            self._update_event = None

        if self.camera is not None:
            self.camera.stop()
            self.camera = None

        self.cascade = None
        self.frame_count = 0
        self.status_text = "Initializing..."

    def go_back(self):
        self.manager.current = "home"

    # --- Setup -----------------------------------------------------------
    def _initialize(self, _dt):
        setup = SETUPS[CHOICE]
        self.log_file = configure_logging(CHOICE)
        self.status_text = f"Loading {setup['label']}..."

        try:
            self.cascade = build_selected_cascade(CHOICE)
        except Exception as e:
            self.status_text = f"[Error] Failed to load cascade: {e}"
            return

        try:
            self.camera = camera_factory(self.camera_mode)
            self.camera.start()
        except Exception as e:
            self.status_text = f"[CameraError] Could not start {self.camera_mode}: {e}"
            self.camera = None
            return 

        self.status_text = "Recognition running..."
        self._update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)

    # --- Per-frame loop ----------------------------------------------------
    def update(self, _dt):
        if self.camera is None:
            return

        start_time = time.time()
        try:
            frame_bgr = self.camera.read()
        except Exception as e:
            self.status_text = f"[CameraError] Failed to read frame: {e}"
            return

        infer_start = time.time()
        results = normalize_results(self.cascade.infer(frame_bgr))
        latency = (time.time() - infer_start) * 1000.0  # ms

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0.0 else 0.0

        for result in results:
            bbox = result.get("bbox")
            if bbox:
                x, y, w, h = bbox
                status = result.get("status", "unknown")
                if status == "accepted":
                    color = (0, 255, 0) if result.get("engine") == "lbph" else (255, 255, 0)
                else:
                    color = (0, 0, 255)
                cv.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
            draw_overlay(frame_bgr, result, fps, latency)

        if results and self.frame_count % 10 == 0:
            LOGGER.info(f"Results: {results} | Latency: {latency:.1f}ms | FPS: {fps:.1f}")

        if results:
            statuses = ", ".join(r.get("status", "unknown") for r in results)
            self.status_text = f"FPS: {fps:4.1f} | Match: {statuses} | Latency: {latency:5.1f}ms"
        else:
            self.status_text = f"FPS: {fps:4.1f} | Searching for faces..."

        self.frame_count += 1
        self._display_frame(frame_bgr)

    def _display_frame(self, frame_bgr):
        buf = cv.flip(frame_bgr, 0).tobytes()
        texture = Texture.create(
            size=(frame_bgr.shape[1], frame_bgr.shape[0]), colorfmt="bgr"
        )
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.ids.camera_feed.texture = texture