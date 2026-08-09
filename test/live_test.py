import cv2 as cv
import time
import sys
import logging
from pathlib import Path

# Setup logging
log_file = Path(__file__).parent / "live_test.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Import the hybrid script from parent dir
sys.path.append(str(Path(__file__).parent.parent))
from hybrid_rpi import HybridCascade

def run_live():
    print("Loading Hybrid Cascade...")
    # Initialize pointing to the parent directory where the models are
    cascade = HybridCascade("..")
    print("Starting webcam...")
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        logging.error("Failed to open webcam.")
        return

    print("Press 'q' to quit.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            logging.error("Failed to grab frame.")
            break
            
        start_time = time.time()
        result = cascade.infer(frame)
        latency = (time.time() - start_time) * 1000  # ms
        
        status = result.get("status", "unknown")
        
        # Log every 15 frames to avoid blowing up the log file instantly, 
        # or log only when a face is detected
        if status != "no_face" and frame_count % 10 == 0:
            logging.info(f"Result: {result} | Latency: {latency:.1f}ms")

        # Draw UI
        if status == "accepted":
            name = result.get("name", "Unknown")
            engine = result.get("engine", "")
            if engine == "lbph":
                dist = result.get("distance", 0)
                text = f"GRANTED: {name} (LBPH: {dist:.1f})"
                color = (0, 255, 0)
            else:
                l2 = result.get("l2", 0)
                reason = result.get("gate_reason", "")
                text = f"GRANTED: {name} (SFace: {l2:.2f} | {reason})"
                color = (255, 255, 0)
        elif status == "rejected":
            reason = result.get("reason", "")
            if reason == "impostor":
                gate = result.get("gate_reason", "")
                text = f"REJECTED: Impostor (SFace | {gate})"
            elif reason == "confident_reject":
                text = f"REJECTED: Confident (LBPH)"
            else:
                text = f"REJECTED: {reason}"
            color = (0, 0, 255)
        elif status == "no_face":
            text = "No face detected"
            color = (200, 200, 200)
        else:
            text = status
            color = (128, 128, 128)
            
        cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv.imshow("Hybrid Cascade - Live Test", frame)
        frame_count += 1
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()
    print(f"Test ended. Logs saved to {log_file}")

if __name__ == "__main__":
    run_live()
