from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from src.engine.camera.picam import PiCamera


class PiCameraColorTests(unittest.TestCase):
    def test_rgb888_preserves_bgr_bytes_and_owns_its_frame(self):
        device = Mock()
        with patch.dict(sys.modules, {'picamera2': SimpleNamespace(Picamera2=lambda: device)}):
            camera = PiCamera(warmup_s=0)
        self.assertEqual(device.create_video_configuration.call_args.kwargs['main']['format'], 'RGB888')
        # Red and blue pixels in the byte order returned by RGB888.
        source = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
        def capture():
            camera._running = False
            return source
        device.capture_array.side_effect = capture
        camera._running = True
        camera._read_loop()
        result = camera.read()
        np.testing.assert_array_equal(result, source)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(result, source))
        source[:] = 0
        self.assertEqual(result[0, 0, 2], 255)
