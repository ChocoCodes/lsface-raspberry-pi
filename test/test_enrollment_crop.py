from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from src.engine.enrollment_crop import participant_crop


class ParticipantCropTests(unittest.TestCase):
    def test_logged_participant_excludes_background_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[172:343, 252:393] = 100
        for x, y, w, h in [(113, 327, 49, 57), (573, 383, 24, 30), (621, 404, 17, 23)]:
            frame[y:y+h, x:x+w] = 255
        crop = participant_crop(frame, (251, 171, 142, 172))
        self.assertTrue((crop == 100).any())
        self.assertFalse((crop == 255).any())
        self.assertLess(crop.shape[0], 480)
        self.assertFalse(np.shares_memory(frame, crop))

    def test_crop_clamps_at_image_edges(self):
        crop = participant_crop(np.zeros((100, 100, 3)), (0, 0, 100, 100))
        self.assertEqual(crop.shape, (100, 100, 3))

    def test_invalid_box_rejected(self):
        for box in [(0, 0, 0, 4), (200, 200, 3, 3), (0, 0, float('nan'), 3)]:
            with self.assertRaises(ValueError):
                participant_crop(np.zeros((100, 100, 3)), box)
