# LS-Face Prototype Implementation Details

## 1. System Overview

**LS-Face** is a hybrid face recognition system prototyped with a Python/Kivy graphical interface. The architecture bridges lightweight classical computer vision techniques with deep learning.

The pipeline combines:
- **YuNet**: High-performance, lightweight CNN-based face detection and facial landmark localization.
- **LBPH (Local Binary Patterns Histograms) with Tan-Triggs Illumination Normalization**: A rapid, computationally inexpensive classical feature descriptor serving as a fast first-stage filter.
- **SFace**: A compact deep neural network generating robust 128-dimensional biometric embeddings.
- **Escalation Gate Architecture**: A two-tiered verification mechanism where confident classifications are handled instantly by LBPH, while ambiguous, low-confidence, or edge-case frames escalate to deep SFace feature matching.

---

## 2. Graphical User Interface & Features

The LS-Face user interface is implemented using the **Kivy** framework, adopting a clean, modular layout separating UI presentation (`.kv` declarative templates) from application logic (`.py` controllers).

### 2.1 Home Screen Overview
The main dashboard serves as the central control panel with hardware configuration and functional navigation:

1. **Hardware & Resource Configuration**:
   - **Camera Selection (`Select Camera`)**: Dropdown/modal interface allowing users to dynamically enumerate and switch between connected video capture devices (e.g., integrated webcam, USB camera, or Edge-based Camera Sensor.
   - **Database Selection (`Select Database`)**: Allows the selection of different databases to be used (e.g., Unified La Salle DB).

2. **Core Feature Modules**:
   - **Add Identity (Enrollment)**: Launches the multi-sample capture pipeline with automated voice-assisted identity tagging.
   - **Recognition**: Initiates real-time video stream ingestion, bounding box tracking, and the cascaded hybrid recognition pipeline.
   - **View Identities (Identity Management - View Only)**: Provides a read-only registry to inspect enrolled records, view associated labels, verify template counts, and review enrolled metadata.

---

## 3. Workflow Architectures

### 3.1 Face Enrollment Pipeline

The enrollment pipeline ensures that high-quality, normalized facial samples and multimodal ground-truth identities are captured and stored in the database.

![Alt text](https://i.imgur.com/0oW2hfz.png)

#### Step-by-Step Breakdown:
1. **Live Capture (5 Variations)**: The camera acquires 5 multi-angle / multi-expression facial frames to provide intra-class diversity (up, down, left, right).
2. **Multimodal Name Tagging (Voice Recognition)**: The subject vocalizes their name. An automated speech-to-text / voice recognition routine transcribes the name into the identity string.
3. **Verification & Retake Gate**: Captured frames and transcribed name are presented to the operator. If rejected, the sequence resets; if accepted, it proceeds to feature processing.
4. **YuNet Landmark Detection & Alignment**: Detects key landmarks (eyes, nose, mouth corners) and computes affine transformation matrices to produce canonically aligned crops.
5. **Dual Feature Processing**:
   - **Branch A (LBPH + Tan-Triggs)**: Converts the aligned crop to grayscale, applies Tan-Triggs normalization (mitigating shadowing and uneven illumination), and computes local texture histograms.
   - **Branch B (SFace Embeddings)**: Feeds the aligned RGB crop through the SFace ONNX model to yield normalized 128-dimensional embedding vectors.
6. **Persistence**: Saves LBPH histograms into `lasalledb_lbph.yml` and deep embeddings into `lasalledb.npy`.

---

### 3.2 Hybrid Face Recognition Pipeline

The updated recognition pipeline incorporates an upfront quality assessment check and precise numerical decision gates for both LBPH fast-path inference and SFace fallback verification.

![Alt text](https://i.imgur.com/US8mmVZ.png)

#### Detailed Stage Breakdown:

1. **Face Detection (YuNet)**:
   - Evaluates input video frames via `face_detection_yunet_2023mar.onnx`.
   - Extracts bounding boxes, detection confidence, and 5 facial fiducial landmarks.
   - If no face satisfies the detection threshold, execution terminates immediately at **Strict Failure (Detection Failed)**.

2. **Quality Check Screening (`quality.py`)**:
   - The detected face region is evaluated across five quantitative metrics:
     - **Blur**: Laplacian variance thresholding.
     - **Illumination**: Mean pixel luminance and shadow-to-highlight ratio.
     - **Noise**: High-frequency spatial noise estimation.
     - **Pose**: Out-of-plane yaw/pitch rotation estimated from landmark geometries.
     - **Face Size**: Minimum pixel resolution constraints (e.g., minimum $80 \times 80$ px).
   - If **Passed**: The crop advances directly to the fast-path **LBPH Inference**.
   - If **Flagged**: Poor lighting, extreme pose, blur, or sub-optimal resolution immediately bypasses LBPH and escalates directly to **SFace Inference (Fallback)**.

3. **Stage 1 — LBPH Inference & Fast-Path Decision Gate**:
   - Computes local texture histograms on Tan-Triggs normalized grayscale crops and matches against `lasalledb_lbph.yml`.
   - Computes two metrics:
     - **$d_1$**: Distance to the nearest identity match.
     - **$m$**: Margin separation ($m = d_2 - d_1$), representing the difference between the closest and second-closest match.
   - **LBPH Score Check Rule**:
     $$\text{If } d_1 \le 52.3724 \quad \text{AND} \quad m \ge 0.05 \implies \textbf{Confident Accept (Fast-path)}$$
     $$\text{If } d_1 > 52.3724 \quad \text{OR} \quad m < 0.05 \implies \textbf{Escalate to SFace Fallback}$$

4. **Stage 2 — SFace Fallback Verification**:
   - Aligns the face using landmark affine transformation and computes a 128-dimensional embedding via `face_recognition_sface_2021dec.onnx`.
   - Compares the embedding vector against stored identities in `lasalledb.npy` using Euclidean ($L_2$) Distance and Cosine Similarity ($\cos\theta$).
   - **SFace Match Check Rule**:
     $$\text{If } L_2 \le 1.0313 \quad \text{AND} \quad \cos\theta \ge 0.363 \implies \textbf{SFace Accept}$$
     $$\text{If } L_2 > 1.0313 \quad \text{OR} \quad \cos\theta < 0.363 \implies \textbf{SFace Reject}$$

---

## 4. Project Directory Structure Breakdown

The codebase is partitioned into distinct modular layers following clean software engineering principles:

```
app/
db/
│   lasalledb_lbph.yml
│   lasalledb.npy
│   thresholds.json
src/
│   assets/
│   │   icons/
│   config/
│   │   config.py
│   │   thresholds.r3.json
│   engine/
│   │   enrollment/
│   │   logs/
│   │   build_cascade.py
│   │   face_aligner.py
│   │   hybrid_rpi.py
│   │   hybrid.py
│   │   lbph_config.py
│   │   quality.py
│   │   rebuild_release.py
│   models/
│   │   face_detection_yunet_2023mar.onnx
│   │   face_recognition_sface_2021dec.onnx
│   ui/
│   │   home.kv
│   │   recognition.kv
│   views/
│       home.py
│       recognition.py
main.py
```

### 4.1 Detailed Directory Specifications

| Path / File | Category | Technical Description & Responsibility |
| :--- | :--- | :--- |
| `main.py` | Application Entry Point | Initializes the Kivy application lifecycle, configures window dimensions, loads screen managers, and handles global shutdown routines. |
| **`app/db/`** | **Data Persistence** | **Houses serialized identity databases, trained models, and active operating thresholds.** |
| `lasalledb_lbph.yml` | LBPH Model DB | Serialized OpenCV LBPH face recognizer data containing histograms, grid parameters, and corresponding integer identity labels. |
| `lasalledb.npy` | SFace Vector DB | NumPy binary array holding 128-dimensional deep feature embeddings paired with subject identifiers. |
| `thresholds.json` | Threshold Settings | System-wide operational cutoffs, including escalation bounds, cosine similarity minimums, and L2 distance thresholds. |
| **`app/src/assets/icons/`**| **UI Assets** | **Visual iconography** |
| **`app/src/config/`** | **Configuration** |
| `config.py` | Dynamic Config | Python module exposing global runtime configurations, camera indexes, frame rates, and path resolutions. |
| `thresholds.r3.json`| Benchmark Profiles | Revision 3 threshold definitions for specific lighting environments or test benchmark baselines. |
| **`app/src/engine/`** | **Core ML / Vision Engine** | **The computational backbone executing image processing, feature extraction, and cascade logic.** |
| `enrollment/` | Enrollment Subsystem | Submodules handling live multi-frame capture sequences, voice-recognition interfaces, and enrollment pipelines. |
| `logs/` | Runtime Logs | Operational audit logs, recognition event timestamps, latency records, and debugging diagnostics. |
| `build_cascade.py` | Training Pipeline | Utility script to aggregate enrolled face samples, fit the LBPH classifier, and compile the feature database. |
| `face_aligner.py` | Face Transformation | Performs 5-point facial landmark alignment, affine rotation, scale normalization, and bounding box cropping. |
| `hybrid_rpi.py` | Platform Variant | Hardware-optimized hybrid cascade implementation tailored for Raspberry Pi (ARM NEON optimizations, reduced resolution passes). |
| `hybrid.py` | Main Cascade Pipeline| Standard x86/workstation implementation orchestrating the YuNet -> LBPH -> Escalation Gate -> SFace decision pipeline. |
| `lbph_config.py` | LBPH Parameters | Configures radius, neighbors, grid size (e.g., 8x8), and Tan-Triggs illumination parameters ($\gamma$, $\sigma_0$, $\sigma_1$). |
| `quality.py` | Quality Assurance | Evaluates image blur (Laplacian variance), illumination uniformity, contrast, and head pose angles before feature extraction. |
| `rebuild_release.py` | Distribution Builder | Automates database re-indexing, cache cleanup, and packaging for release builds. |
| **`app/src/models/`** | **Pretrained Weights** | **Stores pre-trained ONNX deep learning networks.** |
| `face_detection_yunet_2023mar.onnx` | Detection Model | Lightweight CNN model for real-time face detection and 5-point facial landmark regression. |
| `face_recognition_sface_2021dec.onnx` | Recognition Model | Deep neural network mapping aligned face images into a compact, discriminative 128-d feature space. |
| **`app/src/ui/`** | **Kivy Declarative Layouts**| **Kivy language (`.kv`) styling and layout rule definitions.** |
| `home.kv` | Screen Layout | Declarative layout for the home screen (header, camera/database selection selectors, and 3-card navigation). |
| `recognition.kv` | Screen Layout | Declarative layout for the real-time recognition viewport, bounding box overlays, and identification cards. |
| **`app/src/views/`** | **Screen Controllers** | **Python classes managing screen behavior and UI event handling.** |
| `home.py` | View Controller | Manages user interactions on the home screen, modal dialogs, and navigation transitions. |
| `recognition.py` | View Controller | Manages the live video feed thread, calls engine inference pipelines, and updates the HUD in real time. |

---

## 5. Technical Highlights & Optimization Strategies

1. **Lightweight Edge Cascading**:
   By pairing Tan-Triggs + LBPH with SFace via the Escalation Gate, LS-Face minimizes deep neural network inference calls. Standard, high-certainty frames are resolved in under 10ms via LBPH, allowing high frame rates even on edge hardware like the Raspberry Pi (`hybrid_rpi.py`).
2. **Illumination Invariance**:
   The integration of the Tan-Triggs normalization algorithm prior to LBPH feature extraction normalizes harsh shadows, extreme highlights, and ambient light shifts, significantly boosting classical descriptor reliability.
3. **Multimodal Enrollment**:
   Voice recognition integration simplifies edge setup by enabling hands-free, automated labeling during the 5-frame live capture stage, preventing operator input latency.
4. **Strict Separation of Concerns**:
   Decoupling via KV files (`ui/`), view controllers (`views/`), inference algorithms (`engine/`), and centralized models/databases (`models/`, `db/`) ensures maintainability and modular deployments.