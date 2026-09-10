from __future__ import annotations

import importlib.util
import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from hybrid import HybridCascade as LegacyHybridCascade
from hybrid_rpi import HybridCascade, route_after_quality
from quality import QualityThresholds, compute_quality
from head_pose import HeadPoseTracker


ROOT = Path(__file__).resolve().parents[1]


def load_pc_detect():
    path = ROOT / "ex-pc-detect.py"
    spec = importlib.util.spec_from_file_location("r3_pc_detect_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R3IntegrationTests(unittest.TestCase):
    def test_original_and_candidate_keep_constructor_contract(self) -> None:
        self.assertEqual(list(inspect.signature(LegacyHybridCascade).parameters), ["base_dir"])
        self.assertEqual(list(inspect.signature(HybridCascade).parameters)[0], "base_dir")
        self.assertEqual(inspect.signature(HybridCascade.infer).return_annotation, "list[dict]")

    def test_candidate_config_is_explicit(self) -> None:
        config = json.loads((ROOT / "config" / "thresholds.r3.json").read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "candidate_only")
        self.assertEqual(config["lbph_descriptor"]["id"], "r3_n8_g6x6")
        self.assertEqual(config["gate"]["tau_accept"], 52.372394898355424)

    def test_route_and_quality_contracts(self) -> None:
        self.assertEqual(route_after_quality(True, None, 52.0, 140.0), "sface_quality")
        self.assertEqual(route_after_quality(False, 40.0, 52.0, 140.0), "lbph_accept")
        report = compute_quality(
            gray_roi=np.full((100, 100), 80, dtype=np.uint8),
            landmarks=np.asarray([[30, 40], [70, 40], [50, 52], [35, 70], [65, 70]], dtype=np.float32),
            face_px=100,
            thresholds=QualityThresholds(tau_blur=1.0),
        )
        self.assertTrue(report.any_flag)
        self.assertIn("blur", report.active_flags)

    def test_pc_test_normalizes_legacy_and_candidate_results(self) -> None:
        module = load_pc_detect()
        self.assertEqual(module.normalize_results({"status": "accepted"}), [{"status": "accepted"}])
        self.assertEqual(module.normalize_results([]), [])

    def test_head_pose_directions(self) -> None:
        tracker = HeadPoseTracker(ROOT / "models" / "face_detection_yunet_2023mar.onnx")
        tracker._calibrated = True
        self.assertEqual(tracker._classify(yaw=0.0, pitch=10.0, roll=0.0, pitch_proxy=0.0), "UP")
        self.assertEqual(tracker._classify(yaw=0.0, pitch=-10.0, roll=0.0, pitch_proxy=0.0), "DOWN")
        self.assertEqual(tracker._classify(yaw=15.0, pitch=0.0, roll=0.0, pitch_proxy=0.0), "LEFT")
        self.assertEqual(tracker._classify(yaw=-15.0, pitch=0.0, roll=0.0, pitch_proxy=0.0), "RIGHT")


if __name__ == "__main__":
    unittest.main()
