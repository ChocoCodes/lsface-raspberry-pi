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
        
        # Load Thresholds
        with open(f"{base_path}/thresholds.json", "r") as f:
            cfg = json.load(f)
            self.tau_accept = cfg["gate"]["tau_accept"]
            self.tau_reject = cfg["gate"]["tau_reject"]
            self.sface_l2_genuine = cfg["sface"]["l2_genuine"]
            
            # Quality bounds
            self.q_min_face = cfg["quality"]["px_min"]
            self.q_tau_blur = cfg["quality"]["tau_blur"]

        # # 1. Initialize YuNet
        # self.detector = cv.FaceDetectorYN.create(
        #     f"{models_path}/face_detection_yunet_2023mar.onnx",
        #     "", (320, 320), 0.9, 0.3, 5000
        # )

        # Replace YUNET with a FaceAligner class
        self.aligner = FaceAligner(
            detector_weights=f"{models_path}/face_detection_yunet_2023mar.onnx",
            recognizer_weights=f"{models_path}/face_recognition_sface_2021dec.onnx",
            threshold=0.6
        )

        # 2. Initialize LBPH
        self.lbph = cv.face.LBPHFaceRecognizer_create()
        self.lbph.read(f"{base_path}/lbph_seed42_manifest731bcf52fec2_cropped.yml")
        with open(f"{base_path}/lbph_labels_seed42_manifest731bcf52fec2_cropped.json", "r") as f:
            self.lbph_labels = {int(v): k for k, v in json.load(f).items()}

        # 3. Initialize SFace
        self.sface = cv.FaceRecognizerSF.create(
            f"{models_path}/face_recognition_sface_2021dec.onnx", ""
        )
        gallery_dict = np.load(f"{base_path}/sface_gallery_seed42_manifest731bcf52fec2_cropped.npy", allow_pickle=True).item()
        self.sface_labels = list(gallery_dict.keys())
        self.sface_gallery = [gallery_dict[name].reshape(1, -1) for name in self.sface_labels]

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
        # h, w = image_bgr.shape[:2]
        # self.detector.setInputSize((w, h))
        # image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        # _, faces = self.detector.detect(image_bgr)
        # if faces is None or len(faces) == 0:
        #     return {"status": "no_face"}
        # Take largest face
        # largest_face = max(faces, key=lambda f: f[2] * f[3])
        # x, y, bw, bh = [int(v) for v in largest_face[:4]]
        # x = max(0, min(x, w - 1))
        # y = max(0, min(y, h - 1))
        # bw = max(1, min(bw, w - x))
        # bh = max(1, min(bh, h - y))
        
        # Use FaceAligner to align and extract face metadata
        aligned = self.aligner.align(image_bgr)
        if aligned is None:
            return {"status": "no_face"}

        x, y, bw, bh = self.aligner.get_bbox(image_bgr.shape)
        face_px = min(bw, bh)
        
        # Quality Check
        quality_flags = []
        if face_px < self.q_min_face:
            quality_flags.append(f"small_face({face_px}px)")
            
        face_bgr = image_bgr[y:y+bh, x:x+bw]
        if face_bgr.size == 0:
            return {"status": "no_face"}
        
        face_gray = cv.cvtColor(face_bgr, cv.COLOR_BGR2GRAY)
        quality_gray = cv.resize(face_gray, (100, 100), interpolation=cv.INTER_AREA)
        blur_val = cv.Laplacian(quality_gray, cv.CV_64F).var()
        if blur_val < self.q_tau_blur:
            quality_flags.append(f"blurry({blur_val:.1f})")

        # Step 1: LBPH Fast Path
        lbph_norm = self._normalize_lbph(face_gray)
        pred_id, lbph_dist = self.lbph.predict(lbph_norm)
        lbph_name = self.lbph_labels.get(int(pred_id), "Unknown")
        
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
                return {
                    "status": "accepted", "engine": "lbph", "name": lbph_name,
                    "distance": lbph_dist
                }
            else:
                return {
                    "status": "rejected", "engine": "lbph", "reason": "confident_reject",
                    "name": lbph_name, "distance": lbph_dist
                }
            
        # Step 2: SFace Escalation (runs if escalated)
        feature = self.sface.feature(aligned)
        
        best_l2 = float('inf')
        best_idx = -1
        for i, g_feat in enumerate(self.sface_gallery):
            l2_dist = self.sface.match(feature, g_feat, cv.FaceRecognizerSF_FR_NORM_L2)
            if l2_dist < best_l2:
                best_l2 = l2_dist
                best_idx = i
                
        sface_name = self.sface_labels[best_idx]
        
        if best_l2 <= self.sface_l2_genuine:
            return {
                "status": "accepted", "engine": "sface", "name": sface_name,
                "l2": best_l2, "lbph_distance": lbph_dist, "gate_reason": reason
            }
            
        return {
            "status": "rejected", "engine": "sface", "reason": "impostor",
            "name": sface_name, "l2": best_l2, "lbph_distance": lbph_dist, "gate_reason": reason
        }

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    cascade = HybridCascade(str(current_dir))
    print("HybridCascade initialized successfully with ONNX models from 'models/' and configs from root.")

