This folder contains the scripts and fully-trained models for the LBPH + SFace hybrid cascade (trained on LSDB 1, all identities) meant for porting to the Raspberry Pi environment.

### Models:
1. `face_detection_yunet_2023mar.onnx` - YuNet ONNX face detection model.
2. `face_recognition_sface_2021dec.onnx` - SFace ONNX face recognition feature extractor.
3. `lbph_seed42_manifest731bcf52fec2_cropped.yml` - LBPH trained model on cropped LSDB 1.
4. `lbph_labels_seed42_manifest731bcf52fec2_cropped.json` - LBPH labels mapping for the trained LBPH model.
5. `sface_gallery_seed42_manifest731bcf52fec2_cropped.npy` - SFace pre-computed gallery embeddings for LSDB 1.
6. `sface_labels_seed42_manifest731bcf52fec2_cropped.json` - SFace labels mapping for the gallery embeddings.
7. `thresholds.json` - Contains the crucial deployment thresholds for the LBPH gate and SFace.

### Scripts:
- `hybrid_rpi.py` - A lightweight, zero-dependency (other than OpenCV) standalone script that executes the full hybrid cascade logic (detector -> LBPH fast-path -> Gate -> SFace escalation) directly on the Pi.
- `cascade.py` - Contains the `PiCamera` wrapper class, similar to what was previously used in `sface.py`.

*Note: In previous porting folders, file extensions were occasionally scrambled (e.g. ONNX models named as `.py` or `.npy`). This folder uses correct extensions to prevent confusion during hardware integration.*
