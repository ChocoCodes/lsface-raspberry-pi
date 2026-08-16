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
    
    bbox = result.get("bbox")
    if not bbox:
        return 

    x, y, w, h = bbox
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

    name_text, fps_text, latency_text = text, f"FPS: {fps:.1f}", f"Latency: {latency:.1f} ms"
    text_x, text_y = x, max(y - 55, 20)
    font = cv.FONT_HERSHEY_SIMPLEX

    cv.putText(frame, name_text, (text_x, text_y), font, 0.6, color, 2)
    cv.putText(frame, fps_text, (10, frame.shape[0] - 15), font, 0.5, (255, 255, 255), 1)

    (text_width, _), _ = cv.getTextSize(
        latency_text,
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    cv.putText(frame, latency_text, (frame.shape[1] - text_width - 10, frame.shape[0] - 15), font, 0.7, (255, 255, 255), 1)

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

                # Run Hybrid Cascade Inference (Multi-Face)
                infer_start = time.time()
                results = cascade.infer(frame_bgr)
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
                        status = results.get("status", "unknown")

                        if status == 'accepted':
                            color = (0, 255, 0) if result.get("engine") == 'lbph' else (255, 255, 0)
                        else:
                            color = (0, 0, 255)
                    cv.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

                draw_overlay(frame_bgr, result, fps, latency)

                # Log non-empty matches every 10 frames to keep log file size manageable
                if status != "no_face" and frame_count % 10 == 0:
                    logging.info(f"Result: {results} | Latency: {latency:.1f}ms | FPS: {fps:.1f}")

                # Draw UI overlays on frame
                draw_overlay(frame_bgr, results, fps, latency)

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