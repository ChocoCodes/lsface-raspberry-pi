from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from uuid import uuid4

import cv2 as cv
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import ListProperty, NumericProperty, StringProperty
from kivy.uix.screenmanager import Screen
from kivy.uix.widget import Widget

from src.config.config import KV_PATH
from src.engine.camera.manager import CameraBusyError
from src.engine.preview_timing import PreviewTiming
from src.engine.enrollment_crop import participant_crop
from src.ui import load_design_system, tokens

from src.pose_detection.flow import (
    LABELS,
    GuidedPoseCalibration,
    GuidedPoseFlow,
    INSTRUCTIONS,
    format_instruction,
    pnp_profile_problem,
)
from src.pose_detection.head_pose import HeadPoseTracker, load_config


class FaceCutoutOverlay(Widget):
    """Translucent overlay with an anti-aliased oval cutout at the middle."""

    overlay_color = ListProperty([1.0, 1.0, 1.0, 0.65])
    border_color = ListProperty([1.0, 1.0, 1.0, 0.90])
    border_width = NumericProperty(2.5)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._texture = None
        self._cached_size = (0, 0)
        with self.canvas:
            self._color_instr = Color(1, 1, 1, 1)
            self._rect_instr = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_geometry, size=self._update_geometry)
        self.bind(
            overlay_color=self._invalidate_texture,
            border_color=self._invalidate_texture,
            border_width=self._invalidate_texture,
        )

    def _invalidate_texture(self, *_args):
        self._cached_size = (0, 0)
        self._update_geometry()

    def _update_geometry(self, *_args):
        self._rect_instr.pos = self.pos
        self._rect_instr.size = self.size
        w, h = int(round(self.width)), int(round(self.height))
        if w < 10 or h < 10:
            return
        if self._cached_size == (w, h) and self._texture is not None:
            return
        self._cached_size = (w, h)
        self._render_texture(w, h)

    def _render_texture(self, w: int, h: int):
        try:
            r, g, b, a = [int(round(c * 255)) for c in self.overlay_color]
            img = np.full((h, w, 4), (b, g, r, a), dtype=np.uint8)

            center = (w // 2, h // 2)
            # Oval cutout suited for human face framing (approx 1.35 : 1 ratio)
            radius_y = int(round(h * 0.31))
            radius_x = int(round(radius_y * 0.74))

            # Transparent oval cutout
            cv.ellipse(img, center, (radius_x, radius_y), 0, 0, 360, (255, 255, 255, 0), -1, cv.LINE_AA)

            # Anti-aliased subtle border ring
            if self.border_width > 0:
                br, bg, bb, ba = [int(round(c * 255)) for c in self.border_color]
                cv.ellipse(
                    img,
                    center,
                    (radius_x, radius_y),
                    0,
                    0,
                    360,
                    (bb, bg, br, ba),
                    int(round(self.border_width)),
                    cv.LINE_AA,
                )

            flipped = cv.flip(img, 0)
            self._texture = Texture.create(size=(w, h), colorfmt="bgra")
            self._texture.blit_buffer(flipped.tobytes(), colorfmt="bgra", bufferfmt="ubyte")
            self._rect_instr.texture = self._texture
        except Exception:
            self._cached_size = (0, 0)


Builder.load_file(str(KV_PATH / "pose.kv"))

_log = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parents[2]
POSE_PROFILE = APP_ROOT / "config" / "head_pose.local.json"
POSE_TEMPLATE = APP_ROOT / "config" / "head_pose.json"

POSE_TRANSLATIONS = {
    "FRONT": {
        "en": "Look straight at the camera",
        "ja": "カメラをまっすぐ見てください",
        "ko": "카메라를 정면으로 바라봐 주세요",
    },
    "LEFT": {
        "en": "Turn your head to your left",
        "ja": "顔を左に向けてください",
        "ko": "고개를 왼쪽으로 돌려주세요",
    },
    "RIGHT": {
        "en": "Turn your head to your right",
        "ja": "顔を右に向けてください",
        "ko": "고개를 오른쪽으로 돌려주세요",
    },
    "UP": {
        "en": "Tilt your head up",
        "ja": "あごを上げて上を向いてください",
        "ko": "턱을 들고 위를 바라봐 주세요",
    },
    "DOWN": {
        "en": "Tilt your head down",
        "ja": "あごを引いて下を向いてください",
        "ko": "턱을 당기고 아래를 바라봐 주세요",
    },
    "ready": {
        "en": "Look straight at the camera",
        "ja": "カメラをまっすぐ見てください",
        "ko": "카메라를 정면으로 바라봐 주세요",
    },
    "saving": {
        "en": "Saving enrollment…",
        "ja": "登録データを保存しています…",
        "ko": "등록 정보를 저장하고 있습니다…",
    },
    "complete": {
        "en": "Pose scan complete",
        "ja": "スキャンが完了しました",
        "ko": "스캔이 완료되었습니다",
    },
    "error": {
        "en": "Scan needs another try",
        "ja": "もう一度やり直してください",
        "ko": "다시 한 번 시도해 주세요",
    },
    "setup": {
        "en": "Device setup required",
        "ja": "デバイスのセットアップが必要です",
        "ko": "장치 설정이 필요합니다",
    },
    "loading": {
        "en": "Preparing camera…",
        "ja": "カメラを準備しています…",
        "ko": "카메라를 준비하고 있습니다…",
    },
}


class PoseScreen(Screen):
    """One camera screen used for participant scanning and operator setup."""

    camera_mode = StringProperty("Default PC Camera")
    mode = StringProperty("scan")
    phase = StringProperty("loading")
    instruction_en = StringProperty(POSE_TRANSLATIONS["ready"]["en"])
    instruction_ja = StringProperty(POSE_TRANSLATIONS["ready"]["ja"])
    instruction_ko = StringProperty(POSE_TRANSLATIONS["ready"]["ko"])
    instruction_text = StringProperty(POSE_TRANSLATIONS["ready"]["en"])
    note_text = StringProperty("")
    step_states = ListProperty(["waiting"] * 5)
    action_text = StringProperty("")
    progress_value = NumericProperty(0.0)
    identity_name = StringProperty("")
    database_id = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = None
        self.tracker = None
        self.flow = None
        self.pose = None
        self._last_pose = -float("inf")
        self._update_event = None
        self._saved_setup = False
        self._keys_bound = False
        self._captured_frames: dict[str, np.ndarray] = {}
        self._pose_captures = {}
        self._handoff_started = False
        self._camera_texture = None
        self._camera_frame_size = (0, 0)
        self._flip_buf: np.ndarray | None = None  # pre-allocated flip destination
        self._blit_buf: bytearray | None = None   # pre-allocated blit buffer (zero-copy path)
        self._worker_busy = False
        self._pending_frame = None
        self._pending_time = 0.0
        self._profile_verified = False
        self._warmup_skips = 0        # frames skipped while reader thread warms up
        self._display_frame_count = 0  # total frames pushed to texture (for FPS logging)
        self._display_fps_t = 0.0      # timestamp of last FPS log
        self._generation = 0
        self._initialize_event = None
        self._startup_future = None
        self._startup_executor = None
        self._startup_generation = None
        self._startup_cancel = threading.Event()
        self._pose_executor = None
        self._pose_future = None
        self._pose_generation = None
        self._deferred_actions = []
        self._preview_timing = PreviewTiming(_log)
        self.bind(size=self._on_screen_resize)

    def _set_instruction(self, target):
        if isinstance(target, dict):
            trans = target
        else:
            trans = POSE_TRANSLATIONS.get(target, POSE_TRANSLATIONS["FRONT"])
        self.instruction_en = trans.get("en", "")
        self.instruction_ja = trans.get("ja", "")
        self.instruction_ko = trans.get("ko", "")
        self.instruction_text = self.instruction_en

    def _on_screen_resize(self, *_args):
        self._update_camera_geometry()

    def _update_camera_geometry(self, frame_w: int | None = None, frame_h: int | None = None):
        if not hasattr(self, "ids") or "camera_feed" not in self.ids:
            return
        feed = self.ids.camera_feed
        sw, sh = self.width, self.height
        if sw <= 10 or sh <= 10:
            return
        if frame_w is None or frame_h is None:
            frame_w, frame_h = self._camera_frame_size
        if frame_w <= 0 or frame_h <= 0:
            feed.size_hint = (1, 1)
            feed.pos = (0, 0)
            return

        scale = max(sw / frame_w, sh / frame_h)
        nw = frame_w * scale
        nh = frame_h * scale
        feed.size_hint = (None, None)
        feed.size = (nw, nh)
        feed.pos = ((sw - nw) / 2.0, (sh - nh) / 2.0)

    def on_touch_up(self, touch):
        handled = super().on_touch_up(touch)
        if not handled and self.mode == "scan" and self.phase == "ready":
            self.primary_action()
            return True
        return handled

    def on_kv_post(self, *_args):
        if self.mode == "scan":
            name_input = self.ids.get("identity_name_input")
            if name_input is not None and name_input.parent is not None:
                # The name is collected on the voice screen after pose capture.
                name_input.parent.height = 0
                name_input.parent.opacity = 0

    def on_enter(self, *_args):
        self._initialize(0)

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
        self._pose_captures = {}
        self._handoff_started = False
        self._warmup_skips = 0
        self._display_frame_count = 0
        self._display_fps_t = 0.0
        self._last_pose = -float("inf")
        self._startup_cancel = threading.Event()
        self._preview_timing = PreviewTiming(_log)
        self._set_instruction("loading")
        self.note_text = "Preparing camera..."
        self._initialize_event = Clock.schedule_interval(self._poll_startup, 1.0 / 30.0)

    @staticmethod
    def _load_registration(profile_config, mode, manager, camera_mode, cancelled):
        config = profile_config()
        tracker = HeadPoseTracker(config=config)
        problem = pnp_profile_problem(config, tracker.backend.name) if mode == "scan" else None
        if problem or cancelled.is_set():
            return tracker, None, problem
        deadline = time.monotonic() + 10.0
        while not cancelled.is_set():
            try:
                camera = manager.acquire(camera_mode, cancel_event=cancelled)
                return tracker, camera, None
            except CameraBusyError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Camera is busy. Please return and try again.")
                cancelled.wait(0.05)
        return None

    def _poll_startup(self, _dt):
        if self._startup_cancel.is_set():
            return False
        if self._startup_future is None:
            app = App.get_running_app()
            mode = self.camera_mode
            if getattr(getattr(app, "pose_options", None), "picamera2", False):
                mode = "Raspberry Pi Camera"
            self._startup_generation = self._generation
            self._startup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="registration-startup")
            self._startup_future = self._startup_executor.submit(
                self._load_registration, self._profile_config, self.mode,
                app.camera_manager, mode, self._startup_cancel,
            )
            return
        if not self._startup_future.done():
            return
        future, self._startup_future = self._startup_future, None
        if self._startup_generation != self._generation:
            return
        self._startup_executor.shutdown(wait=False)
        self._startup_executor = None
        self._initialize_event = None
        _log.info("[PoseScreen] Initializing — mode=%s camera=%s", self.mode, self.camera_mode)
        try:
            loaded = future.result()
            if loaded is None:
                return False
            self.tracker, self.camera, problem = loaded
        except Exception as exc:
            self._show_error(f"Could not start registration: {exc}")
            return False
        if self.mode == "scan":
            if problem:
                _log.warning("[PoseScreen] PnP profile problem: %s", problem)
                self.phase = "setup"
                self._set_instruction("setup")
                self.note_text = problem
                self.action_text = "Open Device Setup"
                return False
            # The callback owns only this session's dictionary, never screen state.
            captures = self._pose_captures
            self.flow = GuidedPoseFlow(
                self.tracker,
                on_confirm=lambda label, frame, manual=False: self._capture_pose(captures, label, frame),
            )
            self._set_instruction("ready")
            self.note_text = "Press play button when ready."
            self.action_text = "Start Scan"
        else:
            self.flow = GuidedPoseCalibration(self.tracker)
            self._set_instruction("FRONT")
            self.note_text = "Operator-only PnP calibration. No photos are saved."
            self.action_text = ""
            self._bind_setup_keys()
        self._worker_busy = False
        self._profile_verified = False
        self._pose_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="registration-pose")
        self._update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)
        _log.info("[PoseScreen] Clock loop started at 30 Hz; pose worker at %.0f Hz", self.tracker.config.get("pose_hz", 15.0))
        self.phase = self.flow.phase
        self._refresh_text()
        return False

    def _on_worker_result(self, pose):
        self.pose = pose
        self._worker_busy = False
        if self.flow is None:
            return
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

    def _read_frame(self):
        if self.camera is None:
            return None
        return self.camera.read()

    @staticmethod
    def _capture_pose(captures, label, frame):
        captures[label] = np.array(frame, copy=True)

    @staticmethod
    def _process_pose(tracker, flow, mode, frame, now):
        # Estimation AND confirmation/calibration stay off the Kivy thread.
        pose = tracker.estimate(frame, timestamp_s=now)
        if mode == "scan":
            # YuNetGeometry selects the largest face. Preserve that participant
            # through confirmation instead of saving the entire background.
            enrollment_frame = participant_crop(frame, pose.raw.bbox) if pose is not None else frame
            flow.update(enrollment_frame, pose, now)
        else:
            flow.update(pose, now, resolution=(frame.shape[1], frame.shape[0]))
        return pose

    def update(self, _dt):
        if self.camera is None or self.flow is None:
            return
        if self._pose_future is not None and self._pose_future.done():
            future, self._pose_future = self._pose_future, None
            self._worker_busy = False
            try:
                pose = future.result()
                if self._pose_generation == self._generation:
                    # No worker is active while these references are published.
                    self._captured_frames.update(self._pose_captures)
                    self._on_worker_result(pose)
            except Exception as exc:
                self._show_error(f"Pose estimation failed: {exc}")
                return
        if not self._worker_busy:
            actions, self._deferred_actions = self._deferred_actions, []
            for action, args in actions:
                action(*args)
        if self.camera is None or self.flow is None:
            return
        try:
            frame = self._read_frame()
            self._preview_timing.observe(frame, time.monotonic())
        except Exception as exc:
            _log.error("[PoseScreen] Camera read error: %s", exc)
            self._show_error(f"Camera stopped delivering frames: {exc}")
            return

        if frame is None:
            # Reader thread hasn't delivered the first frame yet — normal during warmup.
            # Skip silently; do NOT call _show_error here.
            self._warmup_skips += 1
            if self._warmup_skips == 1:
                _log.debug("[PoseScreen] Waiting for first camera frame (warmup)…")
            elif self._warmup_skips % 30 == 0:
                _log.warning("[PoseScreen] Still waiting for first frame — %d skips", self._warmup_skips)
            return

        if self._warmup_skips > 0:
            _log.info("[PoseScreen] Camera warm after %d skipped ticks", self._warmup_skips)
            self._warmup_skips = -1  # sentinel: log only once

        # Immediate display on UI thread → buttery smooth 30 FPS video preview
        self._display_frame(frame)

        if self.mode == "scan" and not self._profile_verified:
            problem = pnp_profile_problem(
                self.tracker.config, self.tracker.backend.name, (frame.shape[1], frame.shape[0])
            )
            if problem:
                _log.warning("[PoseScreen] PnP profile problem at runtime: %s", problem)
                self._stop_camera()
                self.phase = "setup"
                self._set_instruction("setup")
                self.note_text = problem
                self.action_text = "Open Device Setup"
                return
            self._profile_verified = True
            _log.info("[PoseScreen] PnP profile verified for resolution %dx%d", frame.shape[1], frame.shape[0])

        now = time.monotonic()
        pose_interval = 1.0 / self.tracker.config.get("pose_hz", 15.0)
        if not self._worker_busy and (now - self._last_pose >= pose_interval):
            self._last_pose = now
            self._worker_busy = True
            # Snapshot: worker runs on another thread; camera.read() returns a shared
            # reference so we copy here to prevent the display flip from corrupting
            # the frame the worker is estimating pose on.
            self._pending_frame = frame.copy()
            self._pending_time = now
            self._pose_generation = self._generation
            self._pose_future = self._pose_executor.submit(
                self._process_pose, self.tracker, self.flow, self.mode, self._pending_frame, now,
            )

    def _display_frame(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]

        # Allocate (or reallocate on resolution change) persistent flip + blit buffers.
        # After the first frame these are reused every tick — zero heap allocation.
        if self._flip_buf is None or self._flip_buf.shape != frame_bgr.shape:
            self._flip_buf = np.empty_like(frame_bgr)
            self._blit_buf = bytearray(h * w * 3)

        mirror = self.tracker.config.get("preview_mirror", True)
        # flip_code: -1 = both axes (horizontal mirror + vertical for Kivy's origin)
        #             0 = vertical only (no mirror)
        # Single call replaces the previous two-step cv.flip(cv.flip(frame, 1), 0).
        cv.flip(frame_bgr, -1 if mirror else 0, dst=self._flip_buf)

        feed = self.ids.camera_feed
        if self._camera_texture is None or self._camera_texture.size != (w, h):
            _log.info("[PoseScreen] Creating texture %dx%d (mirror=%s)", w, h, mirror)
            self._camera_texture = Texture.create(size=(w, h), colorfmt="bgr")
            feed.texture = self._camera_texture
            self._camera_frame_size = (w, h)
            self._update_camera_geometry(w, h)
            # Re-allocate blit buffer to match confirmed resolution
            self._blit_buf = bytearray(h * w * 3)

        # Zero-copy blit: write numpy data directly into the pre-allocated bytearray.
        # np.copyto into memoryview avoids creating an intermediate Python bytes object.
        np.copyto(
            np.frombuffer(self._blit_buf, dtype=np.uint8).reshape(h, w, 3),
            self._flip_buf,
        )
        self._camera_texture.blit_buffer(self._blit_buf, colorfmt="bgr", bufferfmt="ubyte")
        # Updating an existing texture does not invalidate the Image canvas.
        # Request a redraw even when no labels or widget properties changed.
        feed.canvas.ask_update()

        # Periodic display FPS logging (every 5 s)
        self._display_frame_count += 1
        now = time.monotonic()
        if self._display_fps_t == 0.0:
            self._display_fps_t = now
        elif now - self._display_fps_t >= 5.0:
            elapsed = now - self._display_fps_t
            fps = self._display_frame_count / elapsed
            _log.info("[PoseScreen] Display FPS: %.1f over last %.1fs", fps, elapsed)
            self._display_frame_count = 0
            self._display_fps_t = now

    def _refresh_text(self):
        if self.flow is None:
            return
        self.progress_value = self.flow.progress
        self.note_text = self.flow.note
        if self.mode == "scan":
            if self.phase == "saving":
                self._set_instruction("saving")
                self.action_text = ""
            elif self.phase == "complete":
                self._set_instruction("complete")
                self.action_text = "Done"
            elif self.phase == "error":
                self._set_instruction("error")
                self.action_text = "Retry Save" if self._captured_frames else "Back"
            elif self.phase == "setup":
                self._set_instruction("setup")
                self.action_text = "Open Device Setup"
            elif self.phase == "calibrating":
                self._set_instruction("FRONT")
                self.action_text = ""
            elif self.phase in ("capture", "feedback"):
                label = LABELS[self.flow.stage] if self.flow.stage < len(LABELS) else "FRONT"
                self._set_instruction(label)
                self.action_text = ""
            else:
                self._set_instruction("ready")
                self.action_text = "Start Scan"
        else:
            if self.phase == "complete":
                self._set_instruction({
                    "en": "Device setup complete",
                    "ja": "セットアップが完了しました",
                    "ko": "설정이 완료되었습니다",
                })
                self.action_text = "Done" if self._saved_setup else ""
            elif self.phase == "error":
                self._set_instruction({
                    "en": "Setup needs another try",
                    "ja": "再試行が必要です",
                    "ko": "다시 시도해 주세요",
                })
                self.action_text = "Restart Setup"
            else:
                label = LABELS[self.flow.stage] if self.flow.stage < len(LABELS) else "FRONT"
                self._set_instruction(label)
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
        if self._worker_busy:
            self._deferred_actions.append((self.primary_action, ()))
            return
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
        if self._worker_busy:
            self._deferred_actions.append((self.restart, ()))
            return
        if self.flow is not None and self.phase not in ("complete", "setup"):
            self.flow.restart()
            self.phase = self.flow.phase
            self.pose = None
            self._last_pose = -float("inf")
            self._captured_frames = {}
            self._pose_captures.clear()
            self._handoff_started = False
            self._refresh_text()

    def go_back(self):
        if self._handoff_started:
            return
        self._shutdown()
        self.manager.current = "home"

    def _stop_worker(self):
        self._generation += 1
        self._deferred_actions = []
        if self._pose_executor is not None:
            self._pose_executor.shutdown(wait=False, cancel_futures=True)
            self._pose_executor = None
        self._worker_busy = False

    def _stop_camera(self):
        self._stop_worker()
        if self._update_event is not None:
            self._update_event.cancel()
            self._update_event = None

        self.camera = None
        self._camera_texture = None
        self._camera_frame_size = (0, 0)
        self._flip_buf = None
        self._blit_buf = None

    def _shutdown(self):
        self._startup_cancel.set()
        if self._initialize_event is not None:
            self._initialize_event.cancel()
            self._initialize_event = None
        if self._startup_executor is not None:
            self._startup_executor.shutdown(wait=False, cancel_futures=True)
            self._startup_executor = None
        self._stop_camera()
        self._unbind_setup_keys()
        if self.flow is not None:
            flow = self.flow
            if self._pose_future is not None and not self._pose_future.done():
                # This closure owns the old flow; never reset a tracker mid-estimate.
                self._pose_future.add_done_callback(lambda _future, old_flow=flow: old_flow.close())
            else:
                flow.close()
        self._pose_future = None
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
            camera_mode=self.camera_mode,
        )
        self.manager.current = "voice_recognition"

    def _show_error(self, message):
        _log.error("[PoseScreen] ERROR → %s", message)
        self._stop_camera()
        self.phase = "error"
        self._set_instruction("error")
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
        if self._worker_busy:
            self._deferred_actions.append((self._on_key_down, (_window, key, _scancode, codepoint, _modifiers)))
            return True
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
