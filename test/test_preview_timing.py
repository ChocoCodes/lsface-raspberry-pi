import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from src.engine.preview_timing import PreviewTiming


class PreviewTimingTests(unittest.TestCase):
    def test_camera_stall_with_responsive_ui(self):
        log = Mock()
        timing = PreviewTiming(log)
        frame = object()
        for tick in (0, .1, .21, .3):
            timing.observe(frame, tick)
        log.warning.assert_called_once()
        self.assertIn('same frame', log.warning.call_args.args[0])

    def test_ui_stall_with_fresh_camera(self):
        log = Mock()
        timing = PreviewTiming(log)
        timing.observe(object(), 0)
        timing.observe(object(), .3)
        log.warning.assert_called_once()
        self.assertIn('callback gap', log.warning.call_args.args[0])

    def test_normal_preview_is_quiet(self):
        log = Mock()
        timing = PreviewTiming(log)
        for tick in range(30):
            timing.observe(object(), tick / 30)
        log.warning.assert_not_called()
