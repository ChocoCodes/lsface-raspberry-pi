#!/usr/bin/env python3

from __future__ import annotations

import logging
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime 

import cv2 as cv
import numpy as np

# Make imports work when the script is launched from anywhere.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure Logging
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = LOG_DIR / f"lsface_{timestamp}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("LS-Face")

from camera import PiCamera
from hybrid_rpi import HybridCascade



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LS-Face HybridCascade on Raspberry Pi Camera."
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=float, default=1.5)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "thresholds.r3.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional explicit enrollment release directory.",
    )
    return parser.parse_args()


def normalize_results(raw_results) -> list[dict]:
    if raw_results is None:
        return []
    if isinstance(raw_results, dict):
        if raw_results.get("status") == "no_face":
            return []
        return [raw_results]
    return list(raw_results)


def result_color(result: dict) -> tuple[int, int, int]:
    if result.get("status") == "accepted":
        if result.get("engine") == "lbph":
            return (0, 255, 0)       # green
        return (255, 255, 0)         # cyan for SFace
    if result.get("status") == "rejected":
        return (0, 0, 255)            # red
    return (200, 200, 200)


def draw_result(
    frame: np.ndarray,
    result: dict,
    fps: float,
    latency_ms: float,
) -> None:
    bbox = result.get("bbox")
    if not bbox:
        return

    x, y, w, h = [int(v) for v in bbox]
    color = result_color(result)

    status = result.get("status", "unknown")
    engine = result.get("engine", "none")
    name = result.get("name", "Unknown")

    if status == "accepted":
        text = f"{name} [{engine.upper()}]"
    elif status == "rejected":
        text = "Unknown"
    else:
        text = name

    # Main bounding box.
    cv.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

    # Identity/status.
    cv.putText(
        frame,
        text,
        (x, max(25, y - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv.LINE_AA,
    )

    # Optional diagnostic values.
    route = result.get("route")
    if route:
        cv.putText(
            frame,
            route,
            (x, min(frame.shape[0] - 10, y + h + 22)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )


def draw_status(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    cascade: HybridCascade,
) -> None:
    """Draw global performance information."""
    h, w = frame.shape[:2]

    cv.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, h - 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    cv.putText(
        frame,
        f"Latency: {latency_ms:.1f} ms",
        (10, h - 15),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    cv.putText(
        frame,
        f"LBPH calls: {cascade.lbph_calls} | SFace calls: {cascade.sface_calls}",
        (w - 390, h - 15),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )


def print_results(results: list[dict], latency_ms: float, fps: float) -> None:
    if not results:
        print(
            f"[FPS {fps:5.1f}] No face"
            f" | latency={latency_ms:6.1f} ms",
            end="\r",
            flush=True,
        )
        return

    for result in results:
        status = result.get("status", "unknown")
        name = result.get("name", "Unknown")
        engine = result.get("engine", "none")
        route = result.get("route", "-")

        if status == "accepted":
            message = f"{name} via {engine.upper()}"
        else:
            message = "UNKNOWN"

        print(
            f"[FPS {fps:5.1f}] {message:<25}"
            f" | route={route:<18}"
            f" | latency={latency_ms:6.1f} ms"
        )


def run() -> int:
    args = parse_args()

    print("=" * 64)
    print("LS-Face Raspberry Pi Live Recognition")
    print("=" * 64)
    print(f"[ROOT]       {ROOT}")
    print(f"[CAMERA]     {args.width}x{args.height}")
    print(f"[CONFIG]     {args.config}")

    # Check important assets before opening the camera.
    detector_model = args.models_dir / "models" / "face_detection_yunet_2023mar.onnx"
    if not detector_model.exists():
        detector_model = args.models_dir / "face_detection_yunet_2023mar.onnx"

    sface_model = args.models_dir / "models" / "face_recognition_sface_2021dec.onnx"
    if not sface_model.exists():
        sface_model = args.models_dir / "face_recognition_sface_2021dec.onnx"

    if not detector_model.exists():
        print(f"[ERROR] YuNet model not found: {detector_model}")
        return 1

    if not sface_model.exists():
        print(f"[ERROR] SFace model not found: {sface_model}")
        return 1

    if not args.config.exists():
        print(f"[ERROR] Threshold config not found: {args.config}")
        return 1

    # ---------------------------------------------------------------
    # 1. Load LS-Face Hybrid Cascade
    # ---------------------------------------------------------------
    print("[1/3] Loading LS-Face HybridCascade...")
    try:
        cascade = HybridCascade(
            models_dir=args.models_dir,
            config_path=args.config,
            artifacts_dir=args.artifacts_dir,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to initialize HybridCascade:")
        print(f"        {exc}")
        print()
        print("Make sure:")
        print("  - enrollment/current.json exists")
        print("  - the referenced release contains lbph.yml")
        print("  - labels.json and sface_gallery.npy exist")
        print("  - manifest.json matches the r3 descriptor")
        return 1

    print(f"[READY] Descriptor: {cascade.threshold_descriptor.descriptor_id}")
    print(f"[READY] Release:    {cascade.artifacts_dir}")

    print("[2/3] Starting Pi Camera...")

    try:
        with PiCamera(
            res=(args.width, args.height),
            warmup_s=args.warmup,
        ) as camera:

            # ---------------------------------------------------------------
            # 3. Live inference
            # ---------------------------------------------------------------
            print("[3/3] Starting live recognition.")
            print(
                "Press Q to quit."
                if not args.no_display
                else "Press Ctrl+C to quit."
            )
            print()

            previous_frame_time = time.perf_counter()
            frame_count = 0
            fps = 0.0

            while True:
                loop_start = time.perf_counter()

                # PiCamera.capture_bgr() already returns BGR.
                frame_bgr = camera.capture_bgr()

                # -----------------------------------------------------------
                # Measure LS-Face inference latency only.
                # This excludes camera capture and display rendering.
                # -----------------------------------------------------------
                infer_start = time.perf_counter()

                results = normalize_results(
                    cascade.infer(frame_bgr)
                )

                latency_ms = (
                    time.perf_counter() - infer_start
                ) * 1000.0

                # -----------------------------------------------------------
                # Calculate end-to-end frame rate.
                # -----------------------------------------------------------
                now = time.perf_counter()
                delta = now - previous_frame_time
                previous_frame_time = now

                if delta > 0:
                    instantaneous_fps = 1.0 / delta

                    fps = (
                        instantaneous_fps
                        if frame_count == 0
                        else 0.9 * fps + 0.1 * instantaneous_fps
                    )

                # -----------------------------------------------------------
                # Log information every 10 frames.
                # -----------------------------------------------------------
                if frame_count % 10 == 0:
                    if not results:
                        logger.info(
                            "frame=%d | fps=%.2f | "
                            "inference=%.2f ms | faces=0",
                            frame_count,
                            fps,
                            latency_ms,
                        )

                    else:
                        for result in results:
                            logger.info(
                                "frame=%d | fps=%.2f | "
                                "inference=%.2f ms | "
                                "name=%s | status=%s | "
                                "engine=%s | route=%s",
                                frame_count,
                                fps,
                                latency_ms,
                                result.get("name", ""),
                                result.get("status", ""),
                                result.get("engine", ""),
                                result.get("route", ""),
                            )

                # -----------------------------------------------------------
                # Draw recognition overlays.
                # -----------------------------------------------------------
                for result in results:
                    draw_result(
                        frame_bgr,
                        result,
                        fps,
                        latency_ms,
                    )

                draw_status(
                    frame_bgr,
                    fps,
                    latency_ms,
                    cascade,
                )

                # -----------------------------------------------------------
                # Console diagnostics.
                # -----------------------------------------------------------
                print_results(
                    results,
                    latency_ms,
                    fps,
                )

                # -----------------------------------------------------------
                # Display.
                # -----------------------------------------------------------
                if not args.no_display:
                    cv.imshow(
                        "LS-Face - Raspberry Pi",
                        frame_bgr,
                    )

                    # Q or ESC exits.
                    key = cv.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        break

                frame_count += 1

                # Prevent a tight loop from producing a misleading
                # zero-duration measurement.
                elapsed = time.perf_counter() - loop_start

                if elapsed <= 0:
                    time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[EXIT] Keyboard interrupt.")

    except Exception as exc:
        print(f"\n[ERROR] Runtime failure: {exc}")
        raise

    print()
    print("=" * 64)
    print("LS-Face stopped.")
    print(f"Frames processed : {frame_count}")
    print(f"LBPH calls       : {cascade.lbph_calls}")
    print(f"SFace calls      : {cascade.sface_calls}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())