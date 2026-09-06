import csv
import importlib.util
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from head_pose import HeadPoseTracker, PoseEstimate


INTEGRATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATION_ROOT
RES_DEFAULT = (1280, 720)
CAM_INDEX = 0
LOG_DIR = INTEGRATION_ROOT / "logs"
LOGGER = logging.getLogger("lsface.pc_detect")
LOGGER.propagate = False

POSE_MODEL = REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
POSE_CALIBRATION_FRAMES = 20

# Intentionally sensitive for feasibility testing. LEFT/RIGHT is prioritized by
# the classifier even if the same frame also exceeds the pitch threshold.
POSE_YAW_THRESHOLD_DEG = 10.0
POSE_PITCH_THRESHOLD_DEG = 5.0
POSE_PITCH_PROXY_THRESHOLD = 0.045
POSE_SMOOTHING_WINDOW = 3

# Ground-truth capture: press F/L/R/U/D while physically holding that pose.
# The first few valid frames are ignored to let the keypress/head movement settle.
GT_SETTLE_FRAMES = 5
GT_CAPTURE_FRAMES = 25

SETUPS = {
    "1": {
        "label": "upstream old setup / r1_n8_g8x8",
        "module": REPO_ROOT / "hybrid.py",
        "models_root": REPO_ROOT,
        "log_file": LOG_DIR / "config1-old-r1.log",
    },
    "2": {
        "label": "new setup / r3_n8_g6x6 + quality-first",
        "module": REPO_ROOT / "hybrid_rpi.py",
        "models_root": REPO_ROOT,
        "config": INTEGRATION_ROOT / "config" / "thresholds.r3.json",
        "enrollment_root": INTEGRATION_ROOT / "enrollment",
        "log_file": LOG_DIR / "config2-new-r3.log",
    },
}


def choose_setup() -> str:
    while True:
        choice = input(
            "Select test setup: 1=old r1 bundle, 2=new r3 quality-first integration: "
        ).strip()
        if choice in SETUPS:
            return choice
        print("Invalid choice. Enter 1 or 2.")


def configure_logging(choice: str) -> Path:
    if choice not in SETUPS:
        raise ValueError(f"Unknown setup {choice!r}; enter 1 or 2.")
    log_file = SETUPS[choice]["log_file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Selected setup: %s", SETUPS[choice]["label"])
    return log_file


def load_hybrid_class(module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    module_dir = str(module_path.parent)
    module_name = f"pc_detect_{module_path.parent.name}_{module_path.stem}"
    dependency_names = ("lbph_config", "quality")
    previous_dependencies = {name: sys.modules.get(name) for name in dependency_names}
    for name in dependency_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, module_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load cascade module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.HybridCascade
    finally:
        sys.path.remove(module_dir)
        for name in dependency_names:
            sys.modules.pop(name, None)
            previous = previous_dependencies[name]
            if previous is not None:
                sys.modules[name] = previous


def build_selected_cascade(choice: str):
    setup = SETUPS[choice]
    cascade_class = load_hybrid_class(setup["module"])
    if choice == "1":
        return cascade_class(str(setup["models_root"]))
    return cascade_class(
        models_dir=str(setup["models_root"]),
        config_path=str(setup["config"]),
        enrollment_root=str(setup["enrollment_root"]),
    )


def normalize_results(raw_results) -> list[dict]:
    if raw_results is None:
        return []
    if isinstance(raw_results, dict):
        return [] if raw_results.get("status") == "no_face" else [raw_results]
    return list(raw_results)


class PoseGroundTruthLogger:
    """Capture labeled bursts so algorithm output can be compared to reality."""

    FIELD_NAMES = [
        "timestamp",
        "label",
        "sample_index",
        "predicted_direction",
        "pitch_source",
        "yaw",
        "pitch",
        "roll",
        "yaw_proxy",
        "pitch_proxy",
        "raw_yaw",
        "raw_pitch",
        "raw_roll",
        "raw_yaw_proxy",
        "raw_pitch_proxy",
        "face_score",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    def __init__(self, log_dir: Path, *, settle_frames: int, capture_frames: int) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_path = log_dir / f"pose-ground-truth-{stamp}.csv"
        self.summary_path = log_dir / f"pose-ground-truth-{stamp}-summary.json"
        self.settle_frames = max(0, int(settle_frames))
        self.capture_frames = max(1, int(capture_frames))
        self.active_label: str | None = None
        self._settled = 0
        self._captured = 0
        self.samples: dict[str, list[dict]] = defaultdict(list)

        self._handle = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELD_NAMES)
        self._writer.writeheader()
        self._handle.flush()

    @property
    def active(self) -> bool:
        return self.active_label is not None

    @property
    def status_text(self) -> str:
        if self.active_label is None:
            return "GT: idle"
        if self._settled < self.settle_frames:
            return f"GT {self.active_label}: settle {self._settled}/{self.settle_frames}"
        return f"GT {self.active_label}: logging {self._captured}/{self.capture_frames}"

    def start(self, label: str) -> None:
        self.active_label = label.upper()
        self._settled = 0
        self._captured = 0
        print(
            f"\n[GT] Hold {self.active_label}. "
            f"Settling {self.settle_frames} frames, then recording {self.capture_frames}."
        )

    def cancel(self) -> None:
        if self.active_label is not None:
            print(f"\n[GT] Cancelled {self.active_label} capture.")
        self.active_label = None
        self._settled = 0
        self._captured = 0

    def consume(self, pose: PoseEstimate | None) -> None:
        if self.active_label is None or pose is None or not pose.calibrated:
            return

        if self._settled < self.settle_frames:
            self._settled += 1
            return

        x, y, w, h = pose.bbox
        row = {
            "timestamp": time.time(),
            "label": self.active_label,
            "sample_index": len(self.samples[self.active_label]),
            "predicted_direction": pose.direction,
            "pitch_source": pose.pitch_source,
            "yaw": pose.yaw,
            "pitch": pose.pitch,
            "roll": pose.roll,
            "yaw_proxy": pose.yaw_proxy,
            "pitch_proxy": pose.pitch_proxy,
            "raw_yaw": pose.raw_yaw,
            "raw_pitch": pose.raw_pitch,
            "raw_roll": pose.raw_roll,
            "raw_yaw_proxy": pose.raw_yaw_proxy,
            "raw_pitch_proxy": pose.raw_pitch_proxy,
            "face_score": pose.score,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
        }
        self.samples[self.active_label].append(row)
        self._writer.writerow(row)
        self._handle.flush()
        self._captured += 1

        if self._captured >= self.capture_frames:
            label = self.active_label
            self.active_label = None
            print(f"\n[GT] {label} captured: {self.capture_frames} samples.")
            self.write_summary(print_to_console=True)

    @staticmethod
    def _stats(rows: list[dict], key: str) -> dict[str, float]:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        return {
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    @staticmethod
    def _separation(a: dict[str, float], b: dict[str, float]) -> dict[str, float | bool]:
        gap = abs(float(a["median"]) - float(b["median"]))
        pooled_width = max(
            1e-9,
            0.5 * ((float(a["p90"]) - float(a["p10"])) + (float(b["p90"]) - float(b["p10"]))),
        )
        overlap = not (float(a["p90"]) < float(b["p10"]) or float(b["p90"]) < float(a["p10"]))
        return {
            "median_gap": gap,
            "separation_ratio": gap / pooled_width,
            "p10_p90_overlap": overlap,
        }

    def build_summary(self) -> dict:
        metrics = ("yaw", "pitch", "roll", "yaw_proxy", "pitch_proxy")
        summary: dict = {"source_csv": self.csv_path.name, "labels": {}}

        for label, rows in sorted(self.samples.items()):
            if not rows:
                continue
            summary["labels"][label] = {
                "n": len(rows),
                **{metric: self._stats(rows, metric) for metric in metrics},
            }

        labels = summary["labels"]
        diagnostics = {}
        if "LEFT" in labels and "RIGHT" in labels:
            diagnostics["left_vs_right_yaw"] = self._separation(
                labels["LEFT"]["yaw"], labels["RIGHT"]["yaw"]
            )
            diagnostics["left_vs_right_yaw_proxy"] = self._separation(
                labels["LEFT"]["yaw_proxy"], labels["RIGHT"]["yaw_proxy"]
            )
        if "UP" in labels and "DOWN" in labels:
            diagnostics["up_vs_down_pnp_pitch"] = self._separation(
                labels["UP"]["pitch"], labels["DOWN"]["pitch"]
            )
            diagnostics["up_vs_down_landmark_pitch"] = self._separation(
                labels["UP"]["pitch_proxy"], labels["DOWN"]["pitch_proxy"]
            )
        summary["diagnostics"] = diagnostics
        return summary

    def write_summary(self, *, print_to_console: bool = False) -> dict:
        summary = self.build_summary()
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        if print_to_console:
            print("\n[GT SUMMARY] medians [p10, p90]")
            for label, data in summary["labels"].items():
                print(
                    f"  {label:5s} "
                    f"yaw={data['yaw']['median']:+6.2f} "
                    f"[{data['yaw']['p10']:+6.2f},{data['yaw']['p90']:+6.2f}]  "
                    f"pnpPitch={data['pitch']['median']:+6.2f} "
                    f"[{data['pitch']['p10']:+6.2f},{data['pitch']['p90']:+6.2f}]  "
                    f"lmPitch={data['pitch_proxy']['median']:+.4f} "
                    f"[{data['pitch_proxy']['p10']:+.4f},{data['pitch_proxy']['p90']:+.4f}]"
                )

            diag = summary.get("diagnostics", {})
            if "up_vs_down_pnp_pitch" in diag:
                pnp = diag["up_vs_down_pnp_pitch"]
                lm = diag["up_vs_down_landmark_pitch"]
                print(
                    "  UP/DOWN PnP:      "
                    f"gap={pnp['median_gap']:.2f}, separation={pnp['separation_ratio']:.2f}, "
                    f"overlap={pnp['p10_p90_overlap']}"
                )
                print(
                    "  UP/DOWN landmark: "
                    f"gap={lm['median_gap']:.4f}, separation={lm['separation_ratio']:.2f}, "
                    f"overlap={lm['p10_p90_overlap']}"
                )
            print(f"  CSV:     {self.csv_path}")
            print(f"  Summary: {self.summary_path}\n")
        return summary

    def close(self) -> None:
        self.write_summary(print_to_console=False)
        self._handle.close()


def draw_recognition_overlay(frame: np.ndarray, result: dict, fps: float, latency: float) -> None:
    bbox = result.get("bbox")
    if not bbox:
        return

    x, y, w, h = bbox
    status = result.get("status", "unknown")
    name = result.get("name", "Unknown")

    if status == "accepted":
        if result.get("engine") == "lbph":
            display_name = f"{name} [LBPH]"
            color = (0, 255, 0)
        else:
            display_name = f"{name} [SFace]"
            color = (255, 255, 0)
    elif status == "rejected":
        display_name = "Unknown" if result.get("reason") == "impostor" else name
        color = (0, 0, 255)
    else:
        display_name = "Unknown"
        color = (200, 200, 200)

    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv.putText(
        frame, display_name, (x, max(y - 10, 25)), cv.FONT_HERSHEY_SIMPLEX,
        0.6, color, 2, cv.LINE_AA,
    )

    cv.putText(
        frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA,
    )
    latency_text = f"Recognition: {latency:.1f} ms"
    (text_width, _), _ = cv.getTextSize(latency_text, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv.putText(
        frame, latency_text,
        (frame.shape[1] - text_width - 10, frame.shape[0] - 15),
        cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA,
    )


def _direction_color(direction: str) -> tuple[int, int, int]:
    if direction == "FRONT":
        return 0, 255, 0
    if direction in {"LEFT", "RIGHT", "UP", "DOWN"}:
        return 0, 215, 255
    if direction == "CALIBRATING":
        return 255, 200, 0
    return 180, 180, 180


def draw_pose_overlay(
    frame: np.ndarray,
    pose: PoseEstimate | None,
    tracker: HeadPoseTracker,
    pose_latency_ms: float,
    gt_logger: PoseGroundTruthLogger,
) -> None:
    panel_x = 12
    font = cv.FONT_HERSHEY_SIMPLEX

    # Larger panel because we now show both pitch estimators and GT status.
    cv.rectangle(frame, (5, 5), (frame.shape[1] - 5, 112), (20, 20, 20), -1)

    if pose is None:
        cv.putText(frame, "POSE: NO FACE", (panel_x, 30), font, 0.75, (160, 160, 160), 2, cv.LINE_AA)
        cv.putText(frame, gt_logger.status_text, (panel_x, 58), font, 0.48, (230, 230, 230), 1, cv.LINE_AA)
        return

    color = _direction_color(pose.direction)
    if not pose.calibrated:
        line1 = f"POSE: CALIBRATING {pose.calibration_count}/{pose.calibration_target}"
        line2 = "LOOK STRAIGHT AT THE CAMERA AND HOLD STILL"
        line3 = gt_logger.status_text
    else:
        line1 = f"POSE: {pose.direction}   [pitch classifier: {tracker.pitch_source.upper()}]"
        line2 = (
            f"yaw={pose.yaw:+.1f} deg  PnP-pitch={pose.pitch:+.1f} deg  "
            f"roll={pose.roll:+.1f} deg  pose={pose_latency_ms:.1f} ms"
        )
        line3 = (
            f"landmark yaw={pose.yaw_proxy:+.4f}  landmark pitch={pose.pitch_proxy:+.4f}  |  "
            f"{gt_logger.status_text}"
        )

    cv.putText(frame, line1, (panel_x, 30), font, 0.75, color, 2, cv.LINE_AA)
    cv.putText(frame, line2, (panel_x, 57), font, 0.47, (245, 245, 245), 1, cv.LINE_AA)
    cv.putText(frame, line3, (panel_x, 82), font, 0.44, (225, 225, 225), 1, cv.LINE_AA)
    cv.putText(
        frame,
        "GT keys F/L/R/U/D | S=summary | P=toggle pitch | C=recalibrate | X/Y=flip | Q=quit",
        (panel_x, 103), font, 0.38, (175, 175, 175), 1, cv.LINE_AA,
    )

    x, y, w, h = pose.bbox
    cv.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    for index, point in enumerate(pose.landmarks):
        px, py = int(round(float(point[0]))), int(round(float(point[1])))
        cv.circle(frame, (px, py), 5 if index == 2 else 3, color, -1, cv.LINE_AA)

    cv.putText(
        frame, pose.direction, (x, min(frame.shape[0] - 10, y + h + 28)),
        font, 0.85, color, 2, cv.LINE_AA,
    )


def run_live():
    choice = choose_setup()
    setup = SETUPS[choice]
    log_file = configure_logging(choice)

    print(f"[1/4] Selected {setup['label']}")
    print(f"[1/4] Initializing logs -> {log_file.name}")
    print("[2/4] Loading Hybrid Cascade models...")
    try:
        cascade = build_selected_cascade(choice)
    except Exception as exc:
        print(f"[Error] Failed to initialize Hybrid Cascade: {exc}")
        return

    print("[3/4] Loading YuNet head-pose tester...")
    try:
        pose_tracker = HeadPoseTracker(
            POSE_MODEL,
            calibration_frames=POSE_CALIBRATION_FRAMES,
            smoothing_window=POSE_SMOOTHING_WINDOW,
            yaw_threshold_deg=POSE_YAW_THRESHOLD_DEG,
            pitch_threshold_deg=POSE_PITCH_THRESHOLD_DEG,
            pitch_proxy_threshold=POSE_PITCH_PROXY_THRESHOLD,
            pitch_source="pnp",
        )
        gt_logger = PoseGroundTruthLogger(
            LOG_DIR,
            settle_frames=GT_SETTLE_FRAMES,
            capture_frames=GT_CAPTURE_FRAMES,
        )
    except Exception as exc:
        print(f"[Error] Failed to initialize pose tester/logger: {exc}")
        return

    print("[4/4] Launching webcam live stream...")
    print("Keep your head FRONT during initial calibration.")
    print("Ground-truth test: physically hold a pose, then press F/L/R/U/D.")
    print("The key records REAL pose even if the algorithm predicts the wrong one.")
    print("P toggles PnP vs landmark pitch. S prints/saves statistics.\n")
    print(f"Ground-truth CSV: {gt_logger.csv_path}\n")

    frame_count = 0
    cap = cv.VideoCapture(CAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_DEFAULT[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_DEFAULT[1])

    if not cap.isOpened():
        print(f"[CameraError] Could not open webcam at index {CAM_INDEX}.")
        gt_logger.close()
        return

    ground_truth_keys = {
        ord("f"): "FRONT",
        ord("l"): "LEFT",
        ord("r"): "RIGHT",
        ord("u"): "UP",
        ord("d"): "DOWN",
    }

    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                print("[CameraError] Failed to read frame from webcam.")
                break

            pose_start = time.perf_counter()
            pose = pose_tracker.estimate(frame_bgr)
            pose_latency_ms = (time.perf_counter() - pose_start) * 1000.0

            infer_start = time.perf_counter()
            results = normalize_results(cascade.infer(frame_bgr))
            recognition_latency_ms = (time.perf_counter() - infer_start) * 1000.0

            gt_logger.consume(pose)

            elapsed = time.perf_counter() - frame_start
            fps = 1.0 / elapsed if elapsed > 0.0 else 0.0

            for result in results:
                draw_recognition_overlay(frame_bgr, result, fps, recognition_latency_ms)

            draw_pose_overlay(frame_bgr, pose, pose_tracker, pose_latency_ms, gt_logger)

            if frame_count % 10 == 0 and pose is not None:
                LOGGER.info(
                    "Pose=%s yaw=%.2f pnp_pitch=%.2f roll=%.2f yaw_proxy=%.4f "
                    "pitch_proxy=%.4f pitch_source=%s pose_ms=%.1f | Results=%s | "
                    "recognition_ms=%.1f | FPS=%.1f",
                    pose.direction,
                    pose.yaw,
                    pose.pitch,
                    pose.roll,
                    pose.yaw_proxy,
                    pose.pitch_proxy,
                    pose.pitch_source,
                    pose_latency_ms,
                    results,
                    recognition_latency_ms,
                    fps,
                )

            if pose is not None:
                pose_text = (
                    f"{pose.direction:<8} yaw={pose.yaw:+5.1f} "
                    f"pnpP={pose.pitch:+5.1f} lmP={pose.pitch_proxy:+.3f}"
                )
            else:
                pose_text = "NO_FACE"

            print(
                f"[FPS {fps:4.1f}] {pose_text} | {gt_logger.status_text:<24}",
                end="\r",
            )

            cv.imshow("PC - LS-Face Recognition + Pose Calibration", frame_bgr)
            frame_count += 1

            key = cv.waitKey(1) & 0xFF
            lower_key = ord(chr(key).lower()) if key != 255 and 0 <= key < 128 else key

            if lower_key == ord("q"):
                break
            if lower_key in ground_truth_keys:
                gt_logger.start(ground_truth_keys[lower_key])
            elif lower_key == ord("s"):
                gt_logger.write_summary(print_to_console=True)
            elif lower_key == ord("p"):
                source = pose_tracker.toggle_pitch_source()
                print(
                    f"\n[Pose] Pitch classifier now uses {source.upper()} "
                    f"({pose_tracker.pitch_threshold_description})."
                )
            elif lower_key == ord("c"):
                gt_logger.cancel()
                pose_tracker.reset_calibration()
                print("\n[Pose] Calibration reset. Face FRONT and hold still.")
            elif lower_key == ord("x"):
                pose_tracker.flip_yaw_mapping()
                print(f"\n[Pose] Left/right mapping changed: {pose_tracker.yaw_mapping}")
            elif lower_key == ord("y"):
                pose_tracker.flip_pitch_mapping()
                print(f"\n[Pose] Up/down mapping changed: {pose_tracker.pitch_mapping}")

    except KeyboardInterrupt:
        print("\n[Exit] User interrupted execution.")
    finally:
        cap.release()
        cv.destroyAllWindows()
        gt_logger.close()
        print(f"\nRecognition log: {log_file}")
        print(f"Pose GT CSV:     {gt_logger.csv_path}")
        print(f"Pose summary:    {gt_logger.summary_path}")


if __name__ == "__main__":
    run_live()
