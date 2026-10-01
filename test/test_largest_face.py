import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np


class LargestFaceTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'app/src/engine/hybrid_rpi.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HybridCascade')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'infer')
        ns = {'np': np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
        self.infer = ns['infer']
        self.cascade = SimpleNamespace(largest_face_only=True, detector=Mock(), _infer_face=Mock(return_value={'status': 'rejected'}))
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_only_largest_area_is_matched_even_when_rejected(self):
        faces = np.array([[0, 0, 100, 100], [0, 0, 80, 150], [0, 0, 200, 20]], dtype=np.float32)
        self.cascade.detector.detect.return_value = (True, faces)
        self.assertEqual(self.infer(self.cascade, self.frame), [{'status': 'rejected'}])
        self.cascade._infer_face.assert_called_once()
        np.testing.assert_array_equal(self.cascade._infer_face.call_args.args[1], faces[1])

    def test_no_faces(self):
        for faces in (None, []):
            self.cascade.detector.detect.return_value = (True, faces)
            self.assertEqual(self.infer(self.cascade, self.frame), [])
        self.cascade._infer_face.assert_not_called()

    def test_equal_area_selects_first(self):
        faces = np.array([[1, 0, 10, 20], [2, 0, 20, 10]], dtype=np.float32)
        self.cascade.detector.detect.return_value = (True, faces)
        self.infer(self.cascade, self.frame)
        np.testing.assert_array_equal(self.cascade._infer_face.call_args.args[1], faces[0])

    def test_multi_face_mode_remains_available(self):
        self.cascade.largest_face_only = False
        self.cascade.detector.detect.return_value = (True, np.array([[0, 0, 10, 20], [2, 0, 20, 10]]))
        self.assertEqual(len(self.infer(self.cascade, self.frame)), 2)


if __name__ == '__main__':
    unittest.main()
