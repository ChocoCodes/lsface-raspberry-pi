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

## r3 candidate integration

The original `hybrid.py` remains the upstream old/r1 implementation and keeps
its `HybridCascade(base_dir=".") -> list[dict]` contract. The candidate files
add the r3 quality-first path without replacing that rollback:

- `hybrid_rpi.py` — r3 cascade with the `r3_n8_g6x6` descriptor.
- `ex-pc-detect.py` — PC webcam test; enter `1` for the original setup or `2`
  for r3. Both paths draw boxes/overlays and use separate local logs.
- `rebuild_release.py` — regenerates the r1/r3 LBPH model and SFace gallery
  from `db/lasalledb.npy`; generated enrollment files are local-only.
- `config/thresholds.r3.json` — candidate r3 thresholds and descriptor.
- `lbph_config.py`, `quality.py` — shared r3 descriptor/quality helpers.

Recover the derived r3 artifacts on a build machine:

```powershell
git lfs pull --include="db/lasalledb.npy"
python rebuild_release.py --descriptor selected --output-root enrollment `
  --release-name release-r3_n8_g6x6
python ex-pc-detect.py
```

The upstream `.npy` database contains the selected LBPH tiles and SFace
embeddings, so the generated `lbph.yml` and `sface_gallery.npy` do not need to
be committed to this repository. The r3 API retains the hardware-facing
`HybridCascade` class, BGR `(H, W, 3)` input, `list[dict]` output, and
`bbox=(x, y, w, h)` result shape. New diagnostics are additive; quality-first
frames intentionally report `lbph_distance=None` because LBPH is skipped.
