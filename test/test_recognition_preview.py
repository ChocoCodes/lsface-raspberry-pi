"""Headless checks for inference scheduling and preview/session isolation."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from src.engine.recognition_worker import RecognitionWorker


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = RecognitionWorker()
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        self.worker.close()

    def wait_done(self):
        self.worker._future.result(timeout=2)

    def test_busy_worker_does_not_queue_and_owns_snapshot(self):
        entered = threading.Event()
        seen = []

        def infer(frame):
            entered.set()
            self.release.wait(2)
            seen.append(frame.copy())
            return []

        cascade = SimpleNamespace(infer=infer)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.assertTrue(self.worker.submit(cascade, frame, 1))
        self.assertTrue(entered.wait(2))
        frame[:] = 255
        self.assertFalse(self.worker.submit(cascade, frame, 2))
        self.assertIsNone(self.worker.poll())
        self.release.set()
        self.wait_done()
        self.assertEqual(self.worker.poll()[0], 1)
        self.assertFalse(seen[0].any())

    def test_leave_discards_old_result_and_reentry_waits_for_old_job(self):
        entered = threading.Event()

        def infer(frame):
            entered.set()
            self.release.wait(2)
            return [{"name": "old session"}]

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.worker.submit(SimpleNamespace(infer=infer), frame, 1)
        self.assertTrue(entered.wait(2))
        self.worker.close()
        new_cascade = Mock()
        new_cascade.infer.return_value = []
        self.assertFalse(self.worker.submit(new_cascade, frame, 2))
        self.release.set()
        self.wait_done()
        self.assertIsNone(self.worker.poll())
        self.assertTrue(self.worker.submit(new_cascade, frame, 3))
        self.wait_done()
        self.assertEqual(self.worker.poll()[2], [])

    def test_error_is_returned_and_next_job_can_run(self):
        cascade = Mock()
        cascade.infer.side_effect = [ValueError("bad inference"), []]
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.worker.submit(cascade, frame, 1)
        self.wait_done()
        self.assertIsInstance(self.worker.poll()[4], ValueError)
        self.assertTrue(self.worker.submit(cascade, frame, 2))
        self.wait_done()
        self.assertIsNone(self.worker.poll()[4])


class PreviewTests(unittest.TestCase):
    def setUp(self):
        # Load the actual screen methods without creating a Kivy window or camera.
        path = ROOT / "app/src/views/recognition.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.bases = [ast.Name(id="object", ctx=ast.Load())]
        self.clock = Mock(return_value=10.0)
        self.overlay = Mock(side_effect=lambda frame, *a, **kw: frame.fill(99))
        self.app = SimpleNamespace(camera_manager=Mock(), pose_options=None)
        self.kivy_clock = Mock()
        namespace = dict(
            StringProperty=lambda value: value, np=np,
            RecognitionWorker=RecognitionWorker,
            time=SimpleNamespace(monotonic=self.clock), LOGGER=Mock(),
            normalize_results=lambda value: value or [],
            draw_overlay=self.overlay, cv=Mock(),
            threading=threading, ThreadPoolExecutor=ThreadPoolExecutor,
            App=SimpleNamespace(get_running_app=lambda: self.app),
            Clock=self.kivy_clock, configure_logging=Mock(return_value='test.log'),
            CHOICE='2', Texture=Mock(), _log=Mock(),
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
        self.screen = namespace['RecognitionScreen']()
        self.screen._active_session_id = 'test'
        self.screen.camera = Mock()
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.screen.camera.read.return_value = self.frame
        self.screen.cascade = Mock()
        self.screen._worker = Mock()
        self.screen._worker.poll.return_value = None
        self.screen._worker.submit.return_value = True
        self.screen._display_frame = Mock()

    def tearDown(self):
        self.screen._cancel_startup()

    def test_preview_continues_while_inference_busy(self):
        self.screen._worker.submit.return_value = False
        for _ in range(3):
            self.screen.update(0)
        self.assertEqual(self.screen._display_frame.call_count, 3)
        self.screen.cascade.infer.assert_not_called()

    def test_warmup_skips_and_next_frame_displays(self):
        self.screen.camera.read.return_value = None
        self.screen.update(0)
        self.screen._worker.submit.assert_not_called()
        self.screen._display_frame.assert_not_called()
        self.screen.camera.read.return_value = self.frame
        self.screen.update(0)
        self.screen._display_frame.assert_called_once()

    def test_overlays_do_not_mutate_camera_frame(self):
        self.screen._worker.poll.return_value = (9.9, self.frame.shape, [{'status': 'accepted'}], 40, None)
        self.screen.update(0)
        self.overlay.assert_called_once()
        self.assertFalse(self.frame.any())
        self.assertTrue((self.screen._display_frame.call_args.args[0] == 99).all())

    def test_expired_or_wrong_resolution_results_are_not_drawn(self):
        for captured, shape in [(9, self.frame.shape), (9.9, (8, 8, 3))]:
            self.screen._worker.poll.return_value = (captured, shape, [{'status': 'accepted'}], 40, None)
            self.screen.update(0)
        self.overlay.assert_not_called()

    def test_submissions_are_throttled(self):
        for now in [10, 10.03, 10.06, 10.11]:
            self.clock.return_value = now
            self.screen.update(0)
        self.assertEqual(self.screen._worker.submit.call_count, 2)
        self.assertEqual(self.screen._display_frame.call_count, 4)

    def test_error_keeps_preview_alive_and_success_clears_error(self):
        self.screen._worker.poll.return_value = (10, self.frame.shape, None, 40, ValueError('failed'))
        self.screen.update(0)
        self.assertIn('RecognitionError', self.screen.status_text)
        self.screen._worker.poll.return_value = (10, self.frame.shape, [], 40, None)
        self.screen.update(0)
        self.assertNotIn('RecognitionError', self.screen.status_text)
        self.assertEqual(self.screen._display_frame.call_count, 2)

    def test_leave_cancels_callbacks_and_invalidates_results(self):
        update_event, init_event = Mock(), Mock()
        self.screen._update_event = update_event
        self.screen._initialize_event = init_event
        self.screen.on_leave()
        update_event.cancel.assert_called_once()
        init_event.cancel.assert_called_once()
        self.screen._worker.close.assert_called_once()
        self.screen.update(0)
        self.screen._display_frame.assert_not_called()

    def test_startup_returns_while_model_load_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        cascade = Mock()

        def build(database_id):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test timed out')
            return cascade, ''

        self.screen._build_cascade_for_session = build
        try:
            self.screen._initialize(0)
            self.assertTrue(entered.wait(2))
            self.assertIsNone(self.screen._initialize(0))
            self.app.camera_manager.acquire.assert_not_called()
        finally:
            release.set()
        self.screen._startup_future.result(timeout=2)
        self.assertFalse(self.screen._initialize(0))
        self.assertIs(self.screen.cascade, cascade)
        self.kivy_clock.schedule_interval.assert_called_once()

    def test_leave_during_load_never_acquires_camera_or_publishes_old_session(self):
        entered, release = threading.Event(), threading.Event()

        def build(database_id):
            entered.set()
            release.wait(2)
            return Mock(), ''

        self.screen._build_cascade_for_session = build
        try:
            self.screen._initialize(0)
            self.assertTrue(entered.wait(2))
            self.screen.on_leave()
            self.screen._activate_session()
            self.assertIsNone(self.screen._initialize(0))
        finally:
            release.set()
        self.screen._startup_future.result(timeout=2)
        self.screen._initialize(0)
        self.assertIsNone(self.screen.cascade)
        self.app.camera_manager.acquire.assert_not_called()
        self.kivy_clock.schedule_interval.assert_not_called()

    def test_startup_error_is_reported_on_ui_thread(self):
        self.screen._build_cascade_for_session = Mock(side_effect=ValueError('missing model'))
        self.screen._initialize(0)
        with self.assertRaises(ValueError):
            self.screen._startup_future.result(timeout=2)
        self.assertFalse(self.screen._initialize(0))
        self.assertIn('missing model', self.screen.status_text)
        self.kivy_clock.schedule_interval.assert_not_called()

    def test_camera_open_does_not_block_ui_and_late_result_is_discarded(self):
        entered, release = threading.Event(), threading.Event()

        def acquire(*args, **kwargs):
            entered.set()
            release.wait(2)
            return Mock()

        self.app.camera_manager.acquire.side_effect = acquire
        self.screen._build_cascade_for_session = Mock(return_value=(Mock(), ''))
        try:
            self.screen._initialize(0)
            self.assertTrue(entered.wait(2))
            self.assertIsNone(self.screen._initialize(0))
            self.screen.on_leave()
        finally:
            release.set()
        self.screen._startup_future.result(timeout=2)
        self.screen._activate_session()
        self.screen._initialize(0)
        self.assertIsNone(self.screen.camera)
        self.kivy_clock.schedule_interval.assert_not_called()
        self.screen._initialize(0)
        self.screen._startup_future.result(timeout=2)
        self.assertFalse(self.screen._initialize(0))
        self.assertIsNotNone(self.screen.camera)
        self.kivy_clock.schedule_interval.assert_called_once()

    def test_fps_starts_at_first_displayed_frame(self):
        self.screen.ids = SimpleNamespace(camera_feed=SimpleNamespace(texture=None, canvas=Mock()))
        display = type(self.screen)._display_frame
        self.clock.return_value = 100
        display(self.screen, self.frame)
        self.assertEqual(self.screen._preview_fps, 0)
        self.clock.return_value = 100.04
        display(self.screen, self.frame)
        self.assertAlmostEqual(self.screen._preview_fps, 25)
        self.assertEqual(self.screen.ids.camera_feed.canvas.ask_update.call_count, 2)


class CameraStartupTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'app/src/engine/camera/manager.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CameraManager')
        self.camera = Mock()
        namespace = dict(threading=threading, Camera=object, CameraBusyError=RuntimeError, camera_factory=Mock(return_value=self.camera))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        self.manager = namespace['CameraManager']()

    def test_cancel_during_camera_start_releases_camera(self):
        cancelled = threading.Event()
        self.camera.start.side_effect = cancelled.set
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.manager.acquire('test', cancel_event=cancelled)
        self.camera.stop.assert_called_once()
        self.assertIsNone(self.manager._camera)

    def test_concurrent_acquisition_fails_promptly_without_second_camera(self):
        entered, release = threading.Event(), threading.Event()

        def start():
            entered.set()
            release.wait(2)

        self.camera.start.side_effect = start
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.manager.acquire, 'test')
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(RuntimeError, 'still starting'):
                    self.manager.acquire('other')
            finally:
                release.set()
            self.assertIs(future.result(timeout=2), self.camera)
        self.manager.close()

    def test_existing_camera_reused_and_closed(self):
        self.assertIs(self.manager.acquire('test'), self.camera)
        self.assertIs(self.manager.acquire('test'), self.camera)
        self.camera.start.assert_called_once()
        self.manager.close()
        self.camera.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
