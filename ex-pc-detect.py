import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np


INTEGRATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATION_ROOT
RES_DEFAULT = (1280, 720)
CAM_INDEX = 0  # Change if using an external/secondary webcam
LOG_DIR = INTEGRATION_ROOT / "logs"
LOGGER = logging.getLogger("lsface.pc_detect")
LOGGER.propagate = False

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
    """Prompt until operator selects old (1) or new (2) configuration."""

    while True:
        choice = input(
            "Select test setup: 1=old r1 bundle, 2=new r3 quality-first integration: "
        ).strip()
        if choice in SETUPS:
            return choice
        print("Invalid choice. Enter 1 or 2.")


def configure_logging(choice: str) -> Path:
    """Attach selected setup's logger; never share handlers between profiles."""

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
    """Load selected cascade without colliding with the other bundle's imports."""

    if not module_path.exists():
        raise FileNotFoundError(module_path)
    module_dir = str(module_path.parent)
    module_name = f"pc_detect_{module_path.parent.name}_{module_path.stem}"
    dependency_names = ("lbph_config", "quality")
    previous_dependencies = {
        name: sys.modules.get(name) for name in dependency_names
    }
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
    if choice not in SETUPS:
        raise ValueError(f"Unknown setup {choice!r}; enter 1 or 2.")
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
    """Normalize old single-result API and new multi-face list API."""

    if raw_results is None:
        return []
    if isinstance(raw_results, dict):
        return [] if raw_results.get("status") == "no_face" else [raw_results]
    return list(raw_results)


def draw_overlay(
    frame: np.ndarray,
    result: dict,
    fps: float,
    latency: float
) -> None:
    """Draw per-face identity, FPS, and latency above the bounding box."""

    bbox = result.get("bbox")
    if not bbox:
        return

    x, y, w, h = bbox

    status = result.get("status", "unknown")
    name = result.get("name", "Unknown")

    # Determine displayed identity/status
    if status == "accepted":
        display_name = name
        if result.get("engine") == "lbph":
            display_name = f"{name} [LBPH]"
        else:
            display_name = f"{name} [SFace]"

    elif status == "rejected":
        if result.get("reason") == "impostor":
            display_name = "Unknown"
        else:
            display_name = name

    else:
        display_name = "Unknown"

    # Text lines
    name_text = display_name
    fps_text = f"FPS: {fps:.1f}"
    latency_text = f"Latency: {latency:.1f} ms"

    # Color based on recognition result
    if status == "accepted":
        if result.get("engine") == "lbph":
            color = (0, 255, 0)       # Green
        else:
            color = (255, 255, 0)    # Cyan
    elif status == "rejected":
        color = (0, 0, 255)          # Red
    else:
        color = (200, 200, 200)      # Gray

    # Text position
    text_x = x
    text_y = max(y - 55, 20)

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Name
    cv.putText(
        frame,
        name_text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness
    )

    # FPS
    cv.putText(
        frame,
        fps_text,
        (10, frame.shape[0] - 15),
        font,
        0.5,
        (255, 255, 255),
        1
    )

    (text_width, text_height), _ = cv.getTextSize(
        latency_text,
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    # Latency
    cv.putText(
        frame,
        latency_text,
        (frame.shape[1] - text_width - 10, frame.shape[0] - 15),
        font,
        0.5,
        (255, 255, 255),
        1
    )


def run_live():
    choice = choose_setup()
    setup = SETUPS[choice]
    log_file = configure_logging(choice)
    print(f"[1/3] Selected {setup['label']}")
    print(f"[1/3] Initializing paths and loggers -> {log_file.name}")
    print("[2/3] Loading Hybrid Cascade models...")
    try:
        cascade = build_selected_cascade(choice)
    except Exception as e:
        print(f"[Error] Failed to initialize Hybrid Cascade: {e}")
        return

    print("[3/3] Launching webcam live stream...")
    frame_count = 0

    cap = cv.VideoCapture(CAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_DEFAULT[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_DEFAULT[1])

    if not cap.isOpened():
        print(f"[CameraError] Could not open webcam at index {CAM_INDEX}.")
        return

    print("Detection running on screen. Press 'q' on the window or Ctrl+C in terminal to exit...\n")
    try:
        while True:
            start_time = time.time()

            # Capture frame from webcam
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                print("[CameraError] Failed to read frame from webcam.")
                break

            # Run Hybrid Cascade Inference
            infer_start = time.time()
            results = normalize_results(cascade.infer(frame_bgr))
            latency = (time.time() - infer_start) * 1000.0  # ms

            if results is None:
                results = []

            # Compute overall frame processing FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0.0 else 0.0

            for result in results:
                bbox = result.get("bbox")
                if bbox:
                    x, y, w, h = bbox
                    status = result.get("status", "unknown")

                    if status == 'accepted':
                        color = (0, 255, 0) if result.get('engine') == 'lbph' else (255, 255, 0)
                    else:
                        color = (0, 0, 255)

                    cv.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

                draw_overlay(frame_bgr, result, fps, latency)

            # Log non-empty matches every 10 frames to keep log file size manageable
            if results and frame_count % 10 == 0:
                LOGGER.info(f"Results: {results} | Latency: {latency:.1f}ms | FPS: {fps:.1f}")

            # Print console diagnostics
            if results:
                statuses = ",".join(result.get("status", "unknown") for result in results)
                print(f"[FPS: {fps:4.1f}] Match: {statuses:<10} | Latency: {latency:5.1f}ms")
            else:
                print(f"[FPS: {fps:4.1f}] Searching for faces...", end="\r")

            # Render GUI window
            cv.imshow("PC - Hybrid Cascade Live Test", frame_bgr)

            frame_count += 1

            # Break loop if user presses 'q' inside the rendering window
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[Exit] User interrupted execution.")
    finally:
        cap.release()
        cv.destroyAllWindows()
        print(f"\nTest ended. Logs saved to {log_file}")


if __name__ == "__main__":
    run_live()
