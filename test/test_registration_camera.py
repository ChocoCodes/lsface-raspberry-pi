"""Registration lifecycle tests without opening a window or camera."""
import ast
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from src.engine.preview_timing import PreviewTiming
from src.engine.enrollment_crop import participant_crop


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'app/src/views/pose.py'
        cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'PoseScreen')
        class Base:
            def bind(self, **kwargs):
                pass
        self.app = SimpleNamespace(camera_manager=Mock(), pose_options=None)
        self.clock = Mock()
        self.tracker_factory = Mock(return_value=SimpleNamespace(config={}, backend=SimpleNamespace(name='test')))
        self.busy = type('CameraBusyError', (RuntimeError,), {})
        ns = dict(Screen=Base, StringProperty=lambda x: x, NumericProperty=lambda x: x,
                  PreviewTiming=PreviewTiming,
                  participant_crop=participant_crop,
                  POSE_TRANSLATIONS={'ready': {'en': '', 'ja': '', 'ko': ''}},
                  ListProperty=lambda x: x, np=np, threading=threading, time=time,
                  ThreadPoolExecutor=ThreadPoolExecutor, _log=Mock(), Clock=self.clock,
                  App=SimpleNamespace(get_running_app=lambda: self.app),
                  HeadPoseTracker=self.tracker_factory, pnp_profile_problem=Mock(return_value=None),
                  CameraBusyError=self.busy, GuidedPoseFlow=Mock(), GuidedPoseCalibration=Mock())
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
        self.screen = ns['PoseScreen']()
        for method in ('_set_instruction', '_refresh_text', '_unbind_setup_keys', '_bind_setup_keys', '_display_frame'):
            setattr(self.screen, method, Mock())
        self.screen._profile_config = Mock(return_value={})

    def tearDown(self):
        self.screen._shutdown()

    def test_initialization_schedules_without_loading_on_ui(self):
        self.screen.on_enter()
        self.tracker_factory.assert_not_called()
        self.app.camera_manager.acquire.assert_not_called()
        self.assertEqual(self.screen.phase, 'loading')
        self.screen.on_leave()
        self.clock.schedule_interval.return_value.cancel.assert_called_once()
        self.assertFalse(self.screen._poll_startup(0))

    def test_leaving_during_blocked_load_does_not_acquire_camera(self):
        entered, release = threading.Event(), threading.Event()
        def build():
            entered.set()
            release.wait(2)
            return {}
        self.screen._profile_config = build
        self.screen.on_enter()
        try:
            self.screen._poll_startup(0)
            self.assertTrue(entered.wait(2))
            self.screen.on_leave()
            self.screen.on_enter()
            self.assertIsNone(self.screen._poll_startup(0))
        finally:
            release.set()
        self.screen._startup_future.result(timeout=2)
        self.screen._poll_startup(0)
        self.assertIsNone(self.screen.tracker)
        self.app.camera_manager.acquire.assert_not_called()

    def test_busy_camera_retries_in_worker(self):
        camera = Mock()
        self.app.camera_manager.acquire.side_effect = [self.busy(), camera]
        loaded = self.screen._load_registration(lambda: {}, 'scan', self.app.camera_manager, 'test', threading.Event())
        self.assertIs(loaded[1], camera)
        self.assertEqual(self.app.camera_manager.acquire.call_count, 2)

    def test_cancelled_load_does_not_open_camera(self):
        cancelled = threading.Event()
        cancelled.set()
        self.screen._load_registration(lambda: {}, 'scan', self.app.camera_manager, 'test', cancelled)
        self.app.camera_manager.acquire.assert_not_called()

    def test_successful_startup_attaches_camera_and_starts_preview(self):
        self.screen.on_enter()
        self.screen._poll_startup(0)
        loaded = self.screen._startup_future.result(timeout=2)
        self.assertFalse(self.screen._poll_startup(0))
        self.assertIs(self.screen.camera, loaded[1])
        self.assertIs(self.screen.tracker, loaded[0])
        self.assertIsNotNone(self.screen._pose_executor)
        self.assertEqual(self.screen.action_text, 'Start Scan')

    def test_startup_error_is_visible_and_stops_polling(self):
        self.tracker_factory.side_effect = RuntimeError('model missing')
        self.screen.on_enter()
        self.screen._poll_startup(0)
        with self.assertRaises(RuntimeError):
            self.screen._startup_future.result(timeout=2)
        self.assertFalse(self.screen._poll_startup(0))
        self.assertEqual(self.screen.phase, 'error')
        self.assertIn('model missing', self.screen.note_text)

    def test_confirmed_frame_is_an_independent_copy(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.screen._on_pose_confirm('FRONT', frame)
        frame[:] = 255
        self.assertFalse(self.screen._captured_frames['FRONT'].any())

    def test_reused_camera_texture_requests_redraw_each_frame(self):
        display = type(self.screen)._display_frame
        display.__globals__['cv'] = Mock()
        texture_factory = Mock()
        texture_factory.create.return_value.size = (4, 4)
        display.__globals__['Texture'] = texture_factory
        self.screen.ids = SimpleNamespace(camera_feed=SimpleNamespace(texture=None, canvas=Mock()))
        self.screen.tracker = SimpleNamespace(config={'preview_mirror': True})
        self.screen._update_camera_geometry = Mock()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(3):
            display(self.screen, frame)
        texture_factory.create.assert_called_once()
        self.assertEqual(self.screen.ids.camera_feed.canvas.ask_update.call_count, 3)

    def test_shutdown_defers_old_tracker_cleanup_until_estimate_finishes(self):
        future = Future()
        future.set_running_or_notify_cancel()
        old_flow = Mock()
        self.screen.flow = old_flow
        self.screen._pose_future = future
        self.screen._shutdown()
        old_flow.close.assert_not_called()
        new_flow = Mock()
        self.screen.flow = new_flow
        future.set_result(None)
        old_flow.close.assert_called_once()
        new_flow.close.assert_not_called()

    def test_pose_completion_only_publishes_results_on_ui_thread(self):
        self.screen.camera = Mock()
        self.screen.camera.read.return_value = None
        self.screen.flow = Mock()
        self.screen._on_worker_result = Mock()
        self.screen._pending_frame = np.zeros((3, 3, 3), dtype=np.uint8)
        self.screen._pending_time = 1
        self.screen._pose_generation = self.screen._generation
        self.screen._pose_future = Future()
        self.screen._pose_future.set_result('pose')
        caller = threading.get_ident()
        seen = []
        self.screen.flow.update.side_effect = lambda *args: seen.append(threading.get_ident())
        self.screen.update(0)
        self.assertEqual(seen, [])
        self.screen._on_worker_result.assert_called_once_with('pose')

    def test_blocked_confirmation_does_not_block_camera_preview(self):
        entered, release = threading.Event(), threading.Event()
        tracker, flow = Mock(), Mock()
        tracker.estimate.return_value = SimpleNamespace(raw=SimpleNamespace(bbox=(0, 0, 4, 4)))
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        worker_threads = []
        def confirm(*args):
            worker_threads.append(threading.get_ident())
            entered.set()
            release.wait(2)
        flow.update.side_effect = confirm
        self.screen.camera = Mock()
        self.screen.camera.read.return_value = frame
        self.screen.flow = flow
        self.screen.tracker = tracker
        tracker.config = {'pose_hz': 12}
        self.screen._profile_verified = True
        self.screen._worker_busy = True
        self.screen._pose_executor = ThreadPoolExecutor(max_workers=1)
        self.screen._pose_future = self.screen._pose_executor.submit(
            self.screen._process_pose, tracker, flow, 'scan', frame, 1,
        )
        try:
            self.assertTrue(entered.wait(2))
            for _ in range(3):
                self.screen.update(0)
            self.assertEqual(self.screen._display_frame.call_count, 3)
            self.assertNotEqual(worker_threads[0], threading.get_ident())
        finally:
            release.set()
        self.screen._pose_future.result(timeout=2)

    def test_old_capture_dictionary_is_isolated_from_new_scan(self):
        old_captures = self.screen._pose_captures
        self.screen._initialize(0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.screen._capture_pose(old_captures, 'FRONT', frame)
        self.assertIn('FRONT', old_captures)
        self.assertEqual(self.screen._pose_captures, {})
        self.assertEqual(self.screen._captured_frames, {})

    def test_stale_pose_is_discarded(self):
        self.screen.camera = Mock()
        self.screen.camera.read.return_value = None
        self.screen.flow = Mock()
        self.screen._pose_generation = -1
        self.screen._pose_future = Future()
        self.screen._pose_future.set_result('old pose')
        self.screen.update(0)
        self.screen.flow.update.assert_not_called()

    def test_restart_waits_for_estimate(self):
        self.screen._worker_busy = True
        self.screen.flow = Mock()
        self.screen.restart()
        self.screen.flow.restart.assert_not_called()
        self.assertEqual(len(self.screen._deferred_actions), 1)


if __name__ == '__main__':
    unittest.main()
