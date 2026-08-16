import json
import math
import numpy as np
import cv2 as cv
from pathlib import Path
from face_aligner import FaceAligner 

class HybridCascade:
    def __init__(self, base_dir="."):
        base_path = Path(base_dir)
        models_path = base_path / "models"
        db_path = base_path / "db"

        # Load unified database and labels
        self.db = np.load(f"{db_path}/lasalledb.npy", allow_pickle=True).item()
        self.labels = { int(record['id']): name for name, record in self.db.items() }

        # Load Thresholds
        with open(f"{db_path}/thresholds.json", "r") as f:
            cfg = json.load(f)
            self.tau_accept = cfg["gate"]["tau_accept"]
            self.tau_reject = cfg["gate"]["tau_reject"]
            self.sface_l2_genuine = cfg["sface"]["l2_genuine"]
            
            # Quality bounds
            self.q_min_face = cfg["quality"]["px_min"]
            self.q_tau_blur = cfg["quality"]["tau_blur"]

        # Instantiate FaceAligner class
        self.aligner = FaceAligner(
            detector_weights=f"{models_path}/face_detection_yunet_2023mar.onnx",
            recognizer_weights=f"{models_path}/face_recognition_sface_2021dec.onnx",
            threshold=0.6
        )

        # Initialize LBPH and Gallery
        self.lbph = cv.face.LBPHFaceRecognizer_create()
        self.lbph.read(f"{db_path}/lasalledb_lbph.yml")

        # Initialize SFace and Gallery
        self.sface = cv.FaceRecognizerSF.create(f"{models_path}/face_recognition_sface_2021dec.onnx", "")
        self.sface_gallery = []
        for name, record in self.db.items():
            id = int(record['id'])
            for embedding in record['sface']:
                self.sface_gallery.append({
                    "id": id,
                    "name": name,
                    "embedding": np.asarray(embedding, dtype=np.float32).reshape(1, -1)
                })

    def _normalize_lbph(self, face_gray):
        """Tan-Triggs preprocessing required for this LBPH model."""
        img = cv.resize(face_gray, (100, 100), interpolation=cv.INTER_AREA)
        img = np.float32(img) / 255.0
        alpha, tau, gamma = 0.1, 10.0, 0.2
        img = np.power(img, gamma)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = img / np.power(np.mean(np.power(np.abs(img), alpha)), 1.0 / alpha)
        img = img / np.power(np.mean(np.power(np.abs(img), tau)), 1.0 / tau)
        img = tau * np.tanh(img / tau)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        return np.uint8(img)

    def infer(self, image_bgr):        
        # Use FaceAligner to align and extract face metadata
        detected_faces = self.aligner.detect_and_align(image_bgr)
        if not detected_faces:
            return []

        results = []
        for face, aligned in detected_faces:
            x, y, bw, bh = self.aligner.get_bbox(face, image_bgr.shape)
            
            # Quality Check
            quality_flags = []
            face_px = min(bw, bh)
            if face_px < self.q_min_face:
                quality_flags.append(f"small_face({face_px}px)")
                
            face_bgr = image_bgr[y:y+bh, x:x+bw]
            if face_bgr.size == 0:
                continue
            
            face_gray = cv.cvtColor(face_bgr, cv.COLOR_BGR2GRAY)
            quality_gray = cv.resize(face_gray, (100, 100), interpolation=cv.INTER_AREA)
            blur_val = cv.Laplacian(quality_gray, cv.CV_64F).var()
            if blur_val < self.q_tau_blur:
                quality_flags.append(f"blurry({blur_val:.1f})")

            # Step 1: LBPH Fast Path
            lbph_norm = self._normalize_lbph(face_gray)
            pred_id, lbph_dist = self.lbph.predict(lbph_norm)
            lbph_name = self.labels.get(int(pred_id), "Unknown")
            
            # Gate Logic
            escalate = False
            reason = ""
            
            if len(quality_flags) > 0:
                escalate = True
                reason = "quality:" + ",".join(quality_flags)
            elif self.tau_accept < lbph_dist < self.tau_reject:
                escalate = True
                reason = "ambiguous_band"
                
            if not escalate:
                if lbph_dist <= self.tau_accept:
                    results.append({
                        "status": "accepted", "engine": "lbph", "name": lbph_name,
                        "distance": lbph_dist, "bbox": (x, y, bw, bh)
                    }) 
                else:
                    results.append({
                        "status": "rejected", "engine": "lbph", "reason": "confident_reject",
                        "name": lbph_name, "distance": lbph_dist, "bbox": (x, y, bw, bh)
                    }) 
                continue
                
            # Step 2: SFace Escalation (runs if escalated)
            feature = self.sface.feature(aligned)
            
            best_l2 = float('inf')
            best_gallery = None

            for person in self.sface_gallery:
                l2_dist = self.sface.match(feature, person['embedding'], cv.FaceRecognizerSF_FR_NORM_L2)
                if l2_dist < best_l2:
                    best_l2 = l2_dist
                    best_gallery = person

            if best_gallery is None:
                results.append({
                    "status": "rejected",
                    "engine": "sface",
                    "reason": "empty_gallery",
                    "lbph_distance": lbph_dist,
                    "bbox": (x, y, bw, bh)
                }) 
                continue
                
            sface_name = best_gallery['name']
            
            if best_l2 <= self.sface_l2_genuine:
                results.append({
                    "status": "accepted", "engine": "sface", "name": sface_name,
                    "l2": best_l2, "lbph_distance": lbph_dist, "gate_reason": reason, "bbox": (x, y, bw, bh)
                })  
            else:
                results.append({
                    "status": "rejected", "engine": "sface", "reason": "impostor",
                    "name": sface_name, "l2": best_l2, "lbph_distance": lbph_dist, "gate_reason": reason, "bbox": (x, y, bw, bh)
                }) 
        return results

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    cascade = HybridCascade(str(current_dir))
    print("HybridCascade initialized successfully with ONNX models from 'models/' and configs from root.")

