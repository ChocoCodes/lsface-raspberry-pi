import time

import cv2 as cv

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import NumericProperty, StringProperty
from kivy.uix.screenmanager import Screen

from src.config.config import KV_PATH
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
    camera_index = NumericProperty(0)
    status_text = StringProperty("Initializing...")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
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
        if self.capture is not None:
            self.capture.release()
            self.capture = None
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
        except Exception as exc:
            self.status_text = f"[Error] Failed to load cascade: {exc}"
            return

        cam_index = self.camera_index if self.camera_index is not None else DEFAULT_CAM_INDEX
        self.capture = cv.VideoCapture(cam_index)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, RES_DEFAULT[0])
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, RES_DEFAULT[1])

        if not self.capture.isOpened():
            self.status_text = f"[CameraError] Could not open webcam at index {cam_index}."
            return

        self.status_text = "Recognition running..."
        self._update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)

    # --- Per-frame loop ----------------------------------------------------
    def update(self, _dt):
        if self.capture is None or not self.capture.isOpened():
            return

        start_time = time.time()
        ret, frame_bgr = self.capture.read()
        if not ret or frame_bgr is None:
            self.status_text = "[CameraError] Failed to read frame from webcam."
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