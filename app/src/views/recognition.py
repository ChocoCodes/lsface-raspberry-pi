import inspect
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen

from src.config.config import KV_PATH
from src.engine.recognition_worker import RecognitionWorker
from src.engine.build_cascade import (
    LOGGER,
    build_selected_cascade,
    configure_logging,
    draw_overlay,
    normalize_results,
)

Builder.load_file(str(KV_PATH / 'recognition.kv'))

_log = logging.getLogger(__name__)

CHOICE = "2"  # new setup / r3_n8_g6x6, quality-first -> recognition mode


class RecognitionScreen(Screen):
    camera_mode = StringProperty("Default PC Camera")
    status_text = StringProperty("Initializing...")
    database_id = StringProperty("")
    expected_identity_name = StringProperty("")
    session_id = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera = None
        self.cascade = None
        self.log_file = None
        self.frame_count = 0
        self._update_event = None
        self._initialize_event = None
        self._session_generation = 0
        self._active_session_id = None
        self._active_database_id = ""
        self._database_route_note = ""
        self._camera_texture = None          # reused across frames
        self._flip_buf: "np.ndarray | None" = None   # pre-allocated flip destination
        self._blit_buf: "bytearray | None" = None    # pre-allocated blit buffer
        self._display_frame_count = 0        # for periodic FPS logging
        self._display_fps_t = 0.0
        self._worker = RecognitionWorker()
        self._last_inference_submit = float("-inf")
        self._latest_inference = None
        self._inference_error = None
        self._preview_fps = 0.0
        self._startup_executor = None
        self._startup_future = None
        self._startup_generation = None
        self._startup_cancel = threading.Event()

    def configure_session(self, database_id, expected_identity_name, session_id=None):
        """Set voice-enrollment context before entering live recognition."""

        self.database_id = "" if database_id is None else str(database_id).strip()
        self.expected_identity_name = (
            "" if expected_identity_name is None else str(expected_identity_name).strip()
        )
        self.session_id = "" if session_id is None else str(session_id).strip()

        # Invalidate the previous recognition session.
        self._cancel_startup()
        self._worker.close()
        self._latest_inference = None
        self._session_generation += 1
        self._active_session_id = None

    def _activate_session(self):
        self._cancel_startup()
        self._startup_cancel = threading.Event()
        self._worker.close()
        self._latest_inference = None
        self._inference_error = None
        self._last_inference_submit = float("-inf")
        self._display_frame_count = 0
        self._display_fps_t = 0.0
        self._preview_fps = 0.0
        self._session_generation += 1
        self._active_session_id = self.session_id or f"recognition-{self._session_generation}"
        self._active_database_id = self.database_id
        self._database_route_note = ""

    @staticmethod
    def _build_cascade_for_session(database_id):
        """Build the selected cascade, forwarding database routing when supported."""

        if not database_id:
            return build_selected_cascade(CHOICE), ""

        try:
            from src.engine.database.database_manager import DatabaseManager

            database = DatabaseManager.resolve_database(database_id)
        except ImportError:
            # Integration point: older builders must add an enrollment_root or
            # database/database_id keyword before custom databases can be routed.
            LOGGER.warning(
                "Selected database %s cannot be resolved because DatabaseManager is unavailable.",
                database_id,
            )
            return build_selected_cascade(CHOICE), " | database routing unavailable"

        builder_parameters = inspect.signature(build_selected_cascade).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in builder_parameters.values()
        )
        route_kwargs = {}
        if accepts_kwargs or "enrollment_root" in builder_parameters:
            route_kwargs["enrollment_root"] = str(database.enrollment_root)
        if accepts_kwargs or "database_id" in builder_parameters:
            route_kwargs["database_id"] = database_id
        if "database" in builder_parameters:
            route_kwargs["database"] = database

        if not route_kwargs:
            # Integration point: extend build_selected_cascade(choice, ...) with
            # enrollment_root/database_id support when custom DB releases go live.
            LOGGER.warning(
                "build_selected_cascade does not accept a database route; using its legacy default for %s.",
                database_id,
            )
            return build_selected_cascade(CHOICE), " | database route pending builder support"

        return build_selected_cascade(CHOICE, **route_kwargs), ""

    # --- Screen lifecycle ---------------------------------------------
    def on_enter(self, *args):
        self._activate_session()
        self.status_text = "Loading recognition and camera..."
        # Poll completion on Kivy's thread; all blocking startup runs in a worker.
        self._initialize_event = Clock.schedule_interval(self._initialize, 1.0 / 30.0)

    def _cancel_startup(self):
        self._startup_cancel.set()
        if self._startup_executor is not None:
            self._startup_executor.shutdown(wait=False, cancel_futures=True)
            self._startup_executor = None

    def on_leave(self, *args):
        self._cancel_startup()
        self._worker.close()
        self._latest_inference = None
        if self._initialize_event is not None:
            self._initialize_event.cancel()
            self._initialize_event = None

        if self._update_event is not None:
            self._update_event.cancel()
            self._update_event = None

        # Camera Manager keeps the camera for the next screen
        self.camera = None

        self.cascade = None
        self.frame_count = 0
        self._session_generation += 1
        self._active_session_id = None
        self._active_database_id = ""
        self.expected_identity_name = ""
        self.session_id = ""
        self.status_text = "Initializing..."

    def go_back(self):
        self.manager.current = "home"

    # --- Setup -----------------------------------------------------------
    @staticmethod
    def _load_session(builder, database_id, camera_manager, camera_mode, cancelled):
        log_file = configure_logging(CHOICE)
        cascade, route_note = builder(database_id)
        if cancelled.is_set():
            return None
        camera = camera_manager.acquire(camera_mode, cancel_event=cancelled)
        return cascade, camera, log_file, route_note

    def _initialize(self, _dt):
        if self._active_session_id is None:
            self._initialize_event = None
            return False
        if self._startup_future is None:
            app = App.get_running_app()
            camera_mode = self.camera_mode
            if getattr(getattr(app, "pose_options", None), "picamera2", False):
                camera_mode = "Raspberry Pi Camera"
            self._startup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recognition-startup")
            self._startup_generation = self._session_generation
            self._startup_future = self._startup_executor.submit(
                self._load_session, self._build_cascade_for_session,
                self._active_database_id, app.camera_manager, camera_mode, self._startup_cancel,
            )
            return
        if not self._startup_future.done():
            return
        future, self._startup_future = self._startup_future, None
        if self._startup_generation != self._session_generation:
            # Wait for cancelled startup to finish before starting another load.
            return
        self._cancel_startup()
        self._initialize_event = None
        try:
            loaded = future.result()
            if loaded is None:
                return False
            self.cascade, self.camera, self.log_file, self._database_route_note = loaded
        except Exception as exc:
            self.status_text = f"[Error] Could not start recognition: {exc}"
            LOGGER.exception("Recognition startup failed")
            return False
        self.status_text = f"Recognition running...{self._database_route_note}"
        self._update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)
        return False

    # --- Per-frame loop ----------------------------------------------------
    def update(self, _dt):
        if self._active_session_id is None or self.camera is None or self.cascade is None:
            return

        try:
            frame_bgr = self.camera.read()
        except Exception as e:
            self.status_text = f"[CameraError] Failed to read frame: {e}"
            LOGGER.exception("Failed to read recognition camera frame")
            return

        # The background camera reader returns None until its first frame arrives.
        if frame_bgr is None:
            self.status_text = "Waiting for camera..."
            return

        now = time.monotonic()
        completed = self._worker.poll()
        if completed is not None:
            captured_at, shape, raw_results, latency, error = completed
            self._inference_error = error
            if error is not None:
                self._latest_inference = None
                LOGGER.error("Recognition inference failed: %s", error)
            else:
                results = normalize_results(raw_results)
                self._latest_inference = (captured_at, shape, results, latency)

        # At most 10 submissions/s; busy workers never accumulate queued frames.
        if now - self._last_inference_submit >= 0.1:
            if self._worker.submit(self.cascade, frame_bgr, now):
                self._last_inference_submit = now

        results, latency = [], 0.0
        if self._latest_inference is not None:
            captured_at, shape, cached_results, cached_latency = self._latest_inference
            # Expire by input age, not completion time; discard old-resolution boxes.
            if now - captured_at <= 0.5 and shape == frame_bgr.shape:
                results, latency = cached_results, cached_latency

        # Camera.read() returns shared storage. Never draw into that storage.
        frame_bgr = frame_bgr.copy()
        fps = self._preview_fps

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
            draw_overlay(frame_bgr, result, fps, latency, greeting=True)

        if results and self.frame_count % 10 == 0:
            LOGGER.info(f"Results: {results} | Latency: {latency:.1f}ms | FPS: {fps:.1f}")

        if self._inference_error is not None:
            self.status_text = f"[RecognitionError] {self._inference_error}"
        elif results:
            statuses = ", ".join(r.get("status", "unknown") for r in results)
            self.status_text = (
                f"FPS: {fps:4.1f} | Match: {statuses} | Latency: {latency:5.1f}ms"
            )
        else:
            self.status_text = f"FPS: {fps:4.1f} | Searching for faces..."

        self.frame_count += 1
        self._display_frame(frame_bgr)

    def _display_frame(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]

        # Lazily allocate (or reallocate on resolution change) persistent buffers.
        # After the first frame these are reused every tick — zero heap allocation.
        if self._flip_buf is None or self._flip_buf.shape != frame_bgr.shape:
            self._flip_buf = np.empty_like(frame_bgr)
            self._blit_buf = bytearray(h * w * 3)

        # Vertical flip only (Kivy's texture origin is bottom-left).
        # Recognition view doesn't mirror; flip_code=0 = vertical only.
        cv.flip(frame_bgr, 0, dst=self._flip_buf)

        if self._camera_texture is None or self._camera_texture.size != (w, h):
            _log.info("[RecognitionScreen] Creating texture %dx%d", w, h)
            self._camera_texture = Texture.create(size=(w, h), colorfmt="bgr")
            self.ids.camera_feed.texture = self._camera_texture
            self._blit_buf = bytearray(h * w * 3)

        # Zero-copy blit: np.copyto into memoryview avoids creating a Python bytes object.
        np.copyto(
            np.frombuffer(self._blit_buf, dtype=np.uint8).reshape(h, w, 3),
            self._flip_buf,
        )
        self._camera_texture.blit_buffer(self._blit_buf, colorfmt="bgr", bufferfmt="ubyte")
        self.ids.camera_feed.canvas.ask_update()

        # Periodic display FPS logging (every 5 s)
        now = time.monotonic()
        if self._display_fps_t == 0.0:
            self._display_fps_t = now
            self._display_frame_count = 0
            return
        self._display_frame_count += 1
        elapsed = now - self._display_fps_t
        if elapsed > 0:
            self._preview_fps = self._display_frame_count / elapsed
        if elapsed >= 5.0:
            elapsed = now - self._display_fps_t
            fps = self._display_frame_count / elapsed
            _log.info("[RecognitionScreen] Display FPS: %.1f over last %.1fs", fps, elapsed)
            self._display_frame_count = 0
            self._display_fps_t = now
