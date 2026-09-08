from __future__ import annotations

from pathlib import Path
import time
from uuid import uuid4

import cv2 as cv
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import ListProperty, NumericProperty, StringProperty
from kivy.uix.screenmanager import Screen

from src.config.config import KV_PATH

from src.pose_detection.flow import (
    LABELS,
    GuidedPoseCalibration,
    GuidedPoseFlow,
    INSTRUCTIONS,
    format_instruction,
    pnp_profile_problem,
)
from src.pose_detection.head_pose import HeadPoseTracker, load_config


Builder.load_file(str(KV_PATH / "pose.kv"))

APP_ROOT = Path(__file__).resolve().parents[2]
POSE_PROFILE = APP_ROOT / "config" / "head_pose.local.json"
POSE_TEMPLATE = APP_ROOT / "config" / "head_pose.json"
POSE_SIZE = (640, 480)


class PoseScreen(Screen):
    """One camera screen used for participant scanning and operator setup."""

    camera_index = NumericProperty(0)
    camera_mode = StringProperty("Default PC Camera")
    mode = StringProperty("scan")
    phase = StringProperty("loading")
    instruction_text = StringProperty("Preparing camera…")
    note_text = StringProperty("")
    step_states = ListProperty(["waiting"] * 5)
    action_text = StringProperty("")
    progress_value = NumericProperty(0.0)
    identity_name = StringProperty("")
    database_id = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.picamera = None
        self.tracker = None
        self.flow = None
        self.pose = None
        self._last_pose = -float("inf")
        self._update_event = None
        self._saved_setup = False
        self._keys_bound = False
        self._captured_frames: dict[str, np.ndarray] = {}
        self._handoff_started = False

    def on_kv_post(self, *_args):
        if self.mode == "scan":
            name_input = self.ids.get("identity_name_input")
            if name_input is not None and name_input.parent is not None:
                # The name is collected on the voice screen after pose capture.
                name_input.parent.height = 0
                name_input.parent.opacity = 0

    def on_enter(self, *_args):
        Clock.schedule_once(self._initialize, 0)

    def on_leave(self, *_args):
        self._shutdown()

    def _profile_config(self):
        path = POSE_PROFILE if POSE_PROFILE.exists() else POSE_TEMPLATE
        config = load_config(path)
        config["backend"] = "yunet_geometry"
        return config

    def _initialize(self, _dt):
        self._shutdown()
        self.phase = "loading"
        self.action_text = ""
        self._saved_setup = False
        self.identity_name = ""
        self._captured_frames = {}
        self._handoff_started = False
        try:
            config = self._profile_config()
            self.tracker = HeadPoseTracker(config=config)
        except Exception as exc:
            self._show_error(f"Could not load PnP setup: {exc}")
            return
        if self.mode == "scan":
            problem = pnp_profile_problem(config, self.tracker.backend.name)
            if problem:
                self.phase = "setup"
                self.instruction_text = "Device setup required"
                self.note_text = problem
                self.action_text = "Open Device Setup"
                return
            self.flow = GuidedPoseFlow(self.tracker, on_confirm=self._on_pose_confirm)
            self.instruction_text = "Look straight at the camera"
            self.note_text = "Press Start when the participant is ready."
            self.action_text = "Start Scan"
        else:
            self.flow = GuidedPoseCalibration(self.tracker)
            self.instruction_text = "Look straight at the camera"
            self.note_text = "Operator-only PnP calibration. No photos are saved."
            self.action_text = ""
            self._bind_setup_keys()
        if not self._open_camera():
            return
        # ponytail: Kivy's clock owns camera and pose work; add a worker only if Pi profiling shows visible stalls.
        self._update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)
        self.phase = self.flow.phase
        self._refresh_text()

    def _open_camera(self) -> bool:
        options = getattr(App.get_running_app(), "pose_options", None)
        use_picamera = bool(getattr(options, "picamera2", False)) or self.camera_mode == "Raspberry Pi Camera"
        try:
            if use_picamera:
                from picamera2 import Picamera2

                self.picamera = Picamera2()
                self.picamera.configure(
                    self.picamera.create_video_configuration(main={"size": POSE_SIZE, "format": "RGB888"})
                )
                self.picamera.start()
                return True
            self.capture = cv.VideoCapture(int(self.camera_index))
            self.capture.set(cv.CAP_PROP_FRAME_WIDTH, POSE_SIZE[0])
            self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, POSE_SIZE[1])
            if self.capture.isOpened():
                return True
            self.capture.release()
            self.capture = None
            self._show_error(f"Could not open camera {self.camera_index}.")
        except Exception as exc:
            self._show_error(f"Could not open camera: {exc}")
        return False

    def _read_frame(self):
        if self.picamera is not None:
            frame_rgb = self.picamera.capture_array("main")
            return cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        return frame if ok else None

    def update(self, _dt):
        frame = self._read_frame()
        if frame is None:
            self._show_error("Camera stopped delivering frames.")
            return
        self._display_frame(frame)
        if self.mode == "scan":
            problem = pnp_profile_problem(
                self.tracker.config, self.tracker.backend.name, (frame.shape[1], frame.shape[0])
            )
            if problem:
                self._stop_camera()
                self.phase = "setup"
                self.instruction_text = "Device setup required"
                self.note_text = problem
                self.action_text = "Open Device Setup"
                return
        now = time.monotonic()
        if now - self._last_pose < 1.0 / self.tracker.config["pose_hz"]:
            return
        self.pose = self.tracker.estimate(frame, timestamp_s=now)
        self._last_pose = now
        if self.mode == "scan":
            self.flow.update(frame, self.pose, now)
        else:
            self.flow.update(self.pose, now, resolution=(frame.shape[1], frame.shape[0]))
        self.phase = self.flow.phase
        if self.phase == "complete":
            self._stop_camera()
            if self.mode == "scan":
                if set(self._captured_frames) != set(LABELS):
                    self._show_error("The scan completed without all five pose frames.")
                else:
                    self._begin_enrollment()
                return
        self._refresh_text()

    def _display_frame(self, frame_bgr):
        preview = cv.flip(frame_bgr, 1) if self.tracker.config.get("preview_mirror", True) else frame_bgr
        texture = Texture.create(size=(preview.shape[1], preview.shape[0]), colorfmt="bgr")
        texture.blit_buffer(cv.flip(preview, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
        self.ids.camera_feed.texture = texture

    def _refresh_text(self):
        if self.flow is None:
            return
        self.progress_value = self.flow.progress
        self.note_text = self.flow.note
        if self.mode == "scan":
            if self.phase == "saving":
                self.instruction_text = "Saving enrollment"
                self.action_text = ""
            elif self.phase == "complete":
                self.instruction_text = "Pose scan complete"
                self.action_text = "Done"
            elif self.phase == "error":
                self.instruction_text = "Enrollment needs another try"
                self.action_text = "Retry Save" if self._captured_frames else "Back"
            elif self.phase == "setup":
                self.instruction_text = "Device setup required"
                self.action_text = "Open Device Setup"
            elif self.phase == "calibrating":
                self.instruction_text = INSTRUCTIONS["FRONT"]
                self.action_text = ""
            elif self.phase in ("capture", "feedback"):
                self.instruction_text = format_instruction(self.flow)
                self.action_text = ""
            else:
                self.instruction_text = INSTRUCTIONS["FRONT"]
                self.action_text = "Start Scan"
        else:
            if self.phase == "complete":
                self.instruction_text = "Device setup complete"
                self.action_text = "Done" if self._saved_setup else ""
            elif self.phase == "error":
                self.instruction_text = "Setup needs another try"
                self.action_text = "Restart Setup"
            else:
                self.instruction_text = format_instruction(self.flow)
                self.action_text = ""
        self.step_states = self._step_states()

    def _step_states(self):
        if self.flow is None:
            return ["waiting"] * len(INSTRUCTIONS)
        states = []
        for index in range(len(INSTRUCTIONS)):
            if index < self.flow.stage or self.phase == "complete":
                states.append("done")
            elif index == self.flow.stage:
                states.append("error" if self.phase == "error" else "active" if self.phase != "ready" else "waiting")
            else:
                states.append("waiting")
        return states

    def primary_action(self):
        now = time.monotonic()
        if self.phase == "error":
            if self.mode == "setup":
                self._initialize(0)
            elif self._captured_frames:
                self._begin_enrollment()
            else:
                self.go_back()
            return
        if self.phase == "setup":
            target = self.manager.get_screen("pose_setup")
            target.camera_index = self.camera_index
            target.camera_mode = self.camera_mode
            self.manager.current = "pose_setup"
            return
        if self.phase == "complete":
            if self.mode == "setup" and not self._saved_setup:
                return
            self.go_back()
            return
        if self.mode == "scan" and self.flow is not None and self.phase == "ready":
            self.flow.start(now)
            self.phase = self.flow.phase
            self._refresh_text()

    def restart(self):
        if self.flow is not None and self.phase not in ("complete", "setup"):
            self.flow.restart()
            self.phase = self.flow.phase
            self.pose = None
            self._last_pose = -float("inf")
            self._captured_frames = {}
            self._handoff_started = False
            self._refresh_text()

    def go_back(self):
        if self._handoff_started:
            return
        self._shutdown()
        self.manager.current = "home"

    def _stop_camera(self):
        if self._update_event is not None:
            self._update_event.cancel()
            self._update_event = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.picamera is not None:
            self.picamera.stop()
            self.picamera.close()
            self.picamera = None

    def _shutdown(self):
        self._stop_camera()
        self._unbind_setup_keys()
        if self.flow is not None:
            self.flow.close()
        self.flow = None
        self.tracker = None
        self.pose = None

    def _on_pose_confirm(self, label: str, frame: np.ndarray, _manual: bool = False):
        """Keep one immutable BGR frame for every confirmed pose."""

        if self.mode == "scan":
            self._captured_frames[label] = np.array(frame, copy=True)

    def _begin_enrollment(self):
        if self._handoff_started:
            return
        if set(self._captured_frames) != set(LABELS):
            self._show_error("All five pose frames are required before saving.")
            return

        home = self.manager.get_screen("home")
        database_id = self.database_id or getattr(home, "database_id", "")
        if not database_id:
            self._show_error("No database is selected for this enrollment.")
            return

        frames = {label: np.array(self._captured_frames[label], copy=True) for label in LABELS}
        voice_screen = self.manager.get_screen("voice_recognition")
        self._handoff_started = True
        self.phase = "complete"
        self.action_text = ""
        self._stop_camera()
        voice_screen.start_enrollment(
            frames,
            database_id=database_id,
            session_id=uuid4().hex,
            initial_name=self.identity_name,
        )
        self.manager.current = "voice_recognition"

    def _show_error(self, message):
        self._stop_camera()
        self.phase = "error"
        self.instruction_text = "Camera or setup error"
        self.note_text = message
        self.action_text = "Back"

    def _bind_setup_keys(self):
        if not self._keys_bound:
            Window.bind(on_key_down=self._on_key_down)
            self._keys_bound = True

    def _unbind_setup_keys(self):
        if self._keys_bound:
            Window.unbind(on_key_down=self._on_key_down)
            self._keys_bound = False

    def _on_key_down(self, _window, key, _scancode, codepoint, _modifiers):
        if self.mode != "setup" or self.flow is None:
            return False
        char = codepoint.lower() if codepoint else (chr(key).lower() if 32 <= key < 127 else "")
        if char in ("f", "l", "r", "u", "d"):
            self.flow.tap(char, time.monotonic())
        elif char == "g":
            self.flow.restart()
        elif char == "s" and self.flow.phase == "complete" and not self._saved_setup:
            try:
                self.flow.save(POSE_PROFILE)
                self._saved_setup = True
                self.flow.note = "PnP setup saved locally. Press Done to return."
            except Exception as exc:
                self._show_error(f"Could not save setup: {exc}")
        elif key == 27:
            self.flow.cancel_current()
        else:
            return False
        self.phase = self.flow.phase
        self._refresh_text()
        return True
