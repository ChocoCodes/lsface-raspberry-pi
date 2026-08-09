import numpy as np 
import os
import cv2 as cv
import time
import sys
import logging
from pathlib import Path

os.environ["DISPLAY"] = ":0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES_DEFAULT = (1280, 720)

# Setup logging
LOG_FILE = Path(__file__).parent / "detect.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Import the hybrid script from parent dir
sys.path.append(str(PROJECT_ROOT))

from hybrid import HybridCascade
from camera import PiCamera

def draw_overlay(frame: np.ndarray, result: dict, fps: float, latency: float) -> None:
    """Renders detection status, engine details, latency, and FPS on the frame."""
    status = result.get("status", "unknown")

    # Determine Text and Color based on Hybrid Cascade status
    if status == "accepted":
        name = result.get("name", "Unknown")
        engine = result.get("engine", "")
        if engine == "lbph":
            dist = result.get("distance", 0.0)
            text = f"GRANTED: {name} (LBPH: {dist:.1f})"
            color = (0, 255, 0)  # Green
        else:
            l2 = result.get("l2", 0.0)
            reason = result.get("gate_reason", "")
            text = f"GRANTED: {name} (SFace: {l2:.2f} | {reason})"
            color = (255, 255, 0) # Cyan/Yellow
    elif status == "rejected":
        reason = result.get("reason", "")
        if reason == "impostor":
            gate = result.get("gate_reason", "")
            text = f"REJECTED: Impostor (SFace | {gate})"
        elif reason == "confident_reject":
            text = "REJECTED: Confident (LBPH)"
        else:
            text = f"REJECTED: {reason}"
        color = (0, 0, 255)  # Red
    elif status == "no_face":
        text = "Searching for faces..."
        color = (200, 200, 200)  # Gray
    else:
        text = f"Status: {status}"
        color = (128, 128, 128)

    # Draw Status Banner at bottom or top
    cv.putText(frame, text, (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw FPS and Latency on top-left
    metrics_text = f"FPS: {fps:.1f} | Latency: {latency:.1f}ms"
    cv.putText(frame, metrics_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def run_live():
    print(f"[1/3] Initializing paths and loggers -> {LOG_FILE.name}")
    print("[2/3] Loading Hybrid Cascade models...")
    try:
        cascade = HybridCascade(str(PROJECT_ROOT))
    except Exception as e:
        print(f"[Error] Failed to initialize Hybrid Cascade: {e}")
        return

    print("[3/3] Launching PiCamera live stream...")
    frame_count = 0

    with PiCamera(res=RES_DEFAULT) as camera:
        print("Detection running on screen. Press 'q' on the window or Ctrl+C in terminal to exit...\n")
        try:
            while True:
                start_time = time.time()

                # Capture frame from PiCamera
                frame_bgr = camera.capture_bgr()

                # Run Hybrid Cascade Inference
                infer_start = time.time()
                result = cascade.infer(frame_bgr)
                latency = (time.time() - infer_start) * 1000.0  # ms

                # Compute overall frame processing FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0.0 else 0.0

                status = result.get("status", "unknown")

                # Log non-empty matches every 10 frames to keep log file size manageable
                if status != "no_face" and frame_count % 10 == 0:
                    logging.info(f"Result: {result} | Latency: {latency:.1f}ms | FPS: {fps:.1f}")

                # Draw UI overlays on frame
                draw_overlay(frame_bgr, result, fps, latency)

                # Print console diagnostics
                if status != "no_face":
                    print(f"[FPS: {fps:4.1f}] Match: {status:<10} | Latency: {latency:5.1f}ms")
                else:
                    print(f"[FPS: {fps:4.1f}] Searching for faces...", end="\r")

                # Render GUI window to Raspberry Pi Connect session
                cv.imshow("Raspberry Pi - Hybrid Cascade Live Test", frame_bgr)

                frame_count += 1

                # Break loop if user presses 'q' inside the rendering window
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[Exit] User interrupted execution.")
        finally:
            cv.destroyAllWindows()
            print(f"\nTest ended. Logs saved to {LOG_FILE}")


if __name__ == "__main__":
    run_live()