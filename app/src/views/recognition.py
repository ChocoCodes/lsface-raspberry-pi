import inspect
import time
import cv2 as cv

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
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
    camera_mode = StringProperty("Default PC Camera")
    status_text = StringProperty("Initializing...")
    database_id = StringProperty("")
    expected_identity_name = StringProperty("")
    session_id = StringProperty("")
    greeting_text = StringProperty("")
    greeting_opacity = NumericProperty(0.0)
    greeting_visible = BooleanProperty(False)
    greeted = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera = None
        self.cascade = None
        self.log_file = None
        self.frame_count = 0
        self._update_event = None
        self._initialize_event = None
        self._greeting_animation = None
        self._session_generation = 0
        self._active_session_id = None
        self._active_database_id = ""
        self._active_expected_name = ""
        self._expected_match_streak = 0
        self._last_frame_signature = None
        self._database_route_note = ""
        self._keys_bound = False

    def configure_session(self, database_id, expected_identity_name, session_id=None):
        """Set voice-enrollment context before entering live recognition."""

        self.database_id = "" if database_id is None else str(database_id).strip()
        self.expected_identity_name = (
            "" if expected_identity_name is None else str(expected_identity_name).strip()
        )
        self.session_id = "" if session_id is None else str(session_id).strip()

        # Invalidate any animation or streak from a previous voice session.
        self._session_generation += 1
        self._active_session_id = None
        self._cancel_greeting()
        self._reset_match_state()

    def _activate_session(self):
        self._session_generation += 1
        self._active_session_id = self.session_id or f"recognition-{self._session_generation}"
        self._active_database_id = self.database_id
        self._active_expected_name = self.expected_identity_name
        self._database_route_note = ""
        self._cancel_greeting()
        self._reset_match_state()

    def _reset_match_state(self):
        self._expected_match_streak = 0
        self._last_frame_signature = None

    def _cancel_greeting(self):
        animation = self._greeting_animation
        self._greeting_animation = None
        if animation is not None:
            Animation.cancel_all(self, "greeting_opacity")
        self.greeted = False
        self.greeting_opacity = 0.0
        self.greeting_visible = False
        self.greeting_text = ""

    def _build_cascade_for_session(self):
        """Build the selected cascade, forwarding database routing when supported."""

        database_id = self._active_database_id or self.database_id
        if not database_id:
            return build_selected_cascade(CHOICE)

        try:
            from src.engine.database.database_manager import DatabaseManager

            database = DatabaseManager.resolve_database(database_id)
        except ImportError:
            # Integration point: older builders must add an enrollment_root or
            # database/database_id keyword before custom databases can be routed.
            self._database_route_note = " | database routing unavailable"
            LOGGER.warning(
                "Selected database %s cannot be resolved because DatabaseManager is unavailable.",
                database_id,
            )
            return build_selected_cascade(CHOICE)

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
            self._database_route_note = " | database route pending builder support"
            LOGGER.warning(
                "build_selected_cascade does not accept a database route; using its legacy default for %s.",
                database_id,
            )
            return build_selected_cascade(CHOICE)

        return build_selected_cascade(CHOICE, **route_kwargs)

    # --- Screen lifecycle ---------------------------------------------
    def on_enter(self, *args):
        self._activate_session()
        if not self._keys_bound:
            Window.bind(on_key_down=self._on_key_down)
            self._keys_bound = True
        # Defer heavy init so the screen transition isn't blocked.
        self._initialize_event = Clock.schedule_once(self._initialize, 0)

    def on_leave(self, *args):
        if self._keys_bound:
            Window.unbind(on_key_down=self._on_key_down)
            self._keys_bound = False

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
        self._active_expected_name = ""
        self._cancel_greeting()
        self._reset_match_state()
        self.expected_identity_name = ""
        self.session_id = ""
        self.status_text = "Initializing..."

    def go_back(self):
        self.manager.current = "home"

    def _on_key_down(self, _window, key, _scancode, codepoint, _modifiers):
        if not self.greeting_visible:
            return False
        if key in (13, 27, 32) or codepoint in ("\r", "\n", " "):
            self.go_back()
            return True
        return False

    # --- Setup -----------------------------------------------------------
    def _initialize(self, _dt):
        self._initialize_event = None
        if self._active_session_id is None:
            return

        setup = SETUPS[CHOICE]
        self.log_file = configure_logging(CHOICE)
        self.status_text = f"Loading {setup['label']}..."

        try:
            self.cascade = self._build_cascade_for_session()
        except Exception as e:
            self.status_text = f"[Error] Failed to load cascade: {e}"
            return

        camera_mode = self.camera_mode
        options = getattr(App.get_running_app(), "pose_options", None)
        if getattr(options, "picamera2", False):
            camera_mode = "Raspberry Pi Camera"

        try:
            self.camera = App.get_running_app().camera_manager.acquire(camera_mode)
        except Exception as e:
            self.status_text = f"[CameraError] Could not start {camera_mode}: {e}"
            self.camera = None
            return 

        self.status_text = f"Recognition running...{self._database_route_note}"
        self._update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)

    # --- Per-frame loop ----------------------------------------------------
    def update(self, _dt):
        if self._active_session_id is None or self.camera is None or self.cascade is None:
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

        self._update_greeting_match(results, frame_bgr)

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

    @staticmethod
    def _frame_signature(frame_bgr):
        thumbnail = cv.resize(frame_bgr, (32, 18), interpolation=cv.INTER_AREA)
        return thumbnail.tobytes()

    def _update_greeting_match(self, results, frame_bgr):
        """Require three distinct frames with one accepted expected identity."""

        if self.greeted or not self._active_expected_name:
            return

        signature = self._frame_signature(frame_bgr)
        if signature == self._last_frame_signature:
            self._expected_match_streak = 0
            return
        self._last_frame_signature = signature

        if len(results) != 1:
            self._expected_match_streak = 0
            return

        result = results[0]
        result_name = result.get("name")
        accepted = result.get("status") == "accepted"
        expected = self._active_expected_name
        if (
            not accepted
            or not isinstance(result_name, str)
            or result_name.strip().casefold() != expected.casefold()
        ):
            self._expected_match_streak = 0
            return

        self._expected_match_streak += 1
        if self._expected_match_streak >= 3:
            self._show_greeting(expected, self._session_generation)

    def _show_greeting(self, name, generation):
        if (
            self.greeted
            or generation != self._session_generation
            or self._active_session_id is None
        ):
            return

        self.greeted = True
        self._expected_match_streak = 0
        self.greeting_text = f"Hello, {name}"
        self.greeting_visible = True
        self.greeting_opacity = 0.0

        animation = Animation(greeting_opacity=1.0, duration=0.45, t="out_quad")
        self._greeting_animation = animation

        def on_complete(completed_animation, _widget):
            if (
                generation == self._session_generation
                and completed_animation is self._greeting_animation
            ):
                self._greeting_animation = None

        animation.bind(on_complete=on_complete)
        animation.start(self)

    def _display_frame(self, frame_bgr):
        buf = cv.flip(frame_bgr, 0).tobytes()
        texture = Texture.create(
            size=(frame_bgr.shape[1], frame_bgr.shape[0]), colorfmt="bgr"
        )
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.ids.camera_feed.texture = texture
