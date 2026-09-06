"""Display-less PnP pose scan and device-setup runner."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_SRC = REPO_ROOT / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

import cv2 as cv

from pose_detection.flow import GuidedPoseCalibration, GuidedPoseFlow, format_instruction, pnp_profile_problem
from pose_detection.head_pose import HeadPoseTracker, POSE_CONFIG_DIR, load_config


POSE_PROFILE = POSE_CONFIG_DIR / "head_pose.local.json"
POSE_TEMPLATE = POSE_CONFIG_DIR / "head_pose.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Run operator PnP setup instead of a participant scan.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--picamera2", action="store_true", help="Use the Raspberry Pi CSI camera through Picamera2.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def pose_config():
    config = load_config(POSE_PROFILE if POSE_PROFILE.exists() else POSE_TEMPLATE)
    config["backend"] = "yunet_geometry"
    return config


def main() -> int:
    options = parse_args()
    try:
        tracker = HeadPoseTracker(config=pose_config())
    except Exception as exc:
        print(f"[setup error] {exc}")
        return 3
    if not options.calibrate:
        problem = pnp_profile_problem(tracker.config, tracker.backend.name)
        if problem:
            print(f"[setup required] {problem}\nRun: python pose-detection/pose_cli.py --calibrate --camera {options.camera}")
            return 2
    flow = GuidedPoseCalibration(tracker) if options.calibrate else GuidedPoseFlow(tracker)
    cap = camera = None
    previous, last_pose = None, -float("inf")
    profile_checked = options.calibrate
    try:
        if options.picamera2:
            from picamera2 import Picamera2

            camera = Picamera2()
            camera.configure(camera.create_video_configuration(main={"size": (options.width, options.height), "format": "RGB888"}))
            camera.start()
        else:
            cap = cv.VideoCapture(options.camera)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, options.width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, options.height)
            if not cap.isOpened():
                print(f"[camera error] Could not open camera {options.camera}.")
                return 3
        flow.start(time.monotonic())
        print("[operator setup]" if options.calibrate else "[add identity pose scan]")
        while True:
            frame = camera.capture_array("main") if camera is not None else cap.read()[1]
            if frame is None:
                print("[camera error] Camera stopped delivering frames.")
                return 3
            if not profile_checked:
                problem = pnp_profile_problem(tracker.config, tracker.backend.name, (frame.shape[1], frame.shape[0]))
                if problem:
                    print(f"[setup required] {problem}")
                    return 2
                profile_checked = True
            now = time.monotonic()
            if now - last_pose < 1.0 / tracker.config["pose_hz"]:
                continue
            pose = tracker.estimate(frame, timestamp_s=now)
            last_pose = now
            if options.calibrate:
                flow.update(pose, now, resolution=(frame.shape[1], frame.shape[0]))
            else:
                flow.update(frame, pose, now)
            current = (flow.phase, format_instruction(flow), flow.note, round(flow.progress, 2))
            if current != previous:
                print(f"[{flow.phase}] {current[1]} {flow.note} ({flow.progress:.0%})")
                previous = current
            if flow.phase == "complete":
                if options.calibrate:
                    flow.save(POSE_PROFILE)
                    print(f"[complete] PnP device setup saved to {POSE_PROFILE}")
                else:
                    print("[complete] All five positions verified. No photos or identity data were saved.")
                return 0
            if flow.phase == "error":
                print(f"[setup failed] {flow.note}")
                return 1
    except KeyboardInterrupt:
        print("\n[cancelled]")
        return 130
    except Exception as exc:
        print(f"[runtime error] {exc}")
        return 3
    finally:
        flow.close()
        if cap is not None:
            cap.release()
        if camera is not None:
            camera.stop()
            camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
