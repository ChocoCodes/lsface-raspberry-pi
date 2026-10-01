# LS-Face TODO

Last reviewed: 2026-09-06

This is the working backlog for the Raspberry Pi port, Kivy edge app, R3
recognition candidate, and head-pose experiment. Items are grouped by kind so
known defects are not confused with experiments or product decisions.

## Scope guardrails

- [x] Treat config2 / `r3_n8_g6x6` as the canonical LBPH descriptor. Do not
  expose an LBPH hyperparameter choice in the app or implementation plan.
- [ ] Keep the R3 profile marked `candidate_only` until it is validated on the
  the target Raspberry Pi camera and the promotion decision is recorded. The
  descriptor is settled; this remaining item is only runtime-threshold and
  deployment validation.
- [ ] Do not change recognition thresholds, LBPH model artifacts, SFace
  galleries, or research result files while implementing head pose.
- [ ] Keep generated enrollment releases, calibration logs, attendee images,
  and personal data out of Git.
- [ ] Preserve the public recognition contract:
  `HybridCascade.infer(frame_bgr) -> list[dict]` with BGR input and `bbox` in
  `(x, y, w, h)` form.

## Confirmed bugs and blockers

### Pose experiment

- [x] Keep one authoritative Kivy runtime in `app/src/pose_detection`; root
  console tools in `pose-detection/` import it instead of carrying a copy.
- [ ] Fix the UP/DOWN semantic reversal without treating neutral recalibration
  as sign calibration. Neutral calibration should change offsets only;
  explicit or guided calibration must establish vertical semantics.
- [ ] Reduce false LEFT/RIGHT triggers. The current yaw threshold is too low
  for unconditional priority; require strong yaw evidence, confidence, and
  temporal stability before overriding vertical labels.
- [ ] Replace raw five-point PnP pitch as the implicit authority. It is flaky
  and can change sign; compare it with normalized landmark geometry and select
  using labeled data.
- [ ] Make subject-perspective labels independent of preview mirroring. Infer
  from the original frame and mirror only the display; show the mirror state
  in diagnostics.
- [ ] Remove the extra full-frame YuNet pass from the eventual integrated path.
  The pose API must accept an existing YuNet face row and reuse its landmarks.
- [ ] Make ground-truth capture safe: an unfinished burst must be cancellable,
  and undo must remove exactly the last completed burst from derived analysis.

### Kivy edge application

- [ ] Implement the Add Identity screen. The home-card handler currently only
  prints a navigation message.
- [ ] Implement the View Identities screen. It should list enrolled labels,
  template counts, IDs, and available metadata without exposing raw personal
  data unnecessarily.
- [ ] Wire the database selector to real configuration/database selection. It
  currently only prints the selected value.
- [ ] Replace the hard-coded camera spinner with camera enumeration and clear
  unavailable-camera errors. Persist the selected camera for the session.
- [ ] Move camera capture and inference off the Kivy/UI event loop. The current
  recognition screen performs inference inside a scheduled UI callback and can
  block rendering.
- [ ] Cancel a pending recognition initialization when leaving the screen;
  prevent a delayed callback from opening a camera after navigation away.
- [ ] Add retry/restart controls for camera-open, frame-read, model-load, and
  missing-release failures.
- [ ] Make the fixed-size home layout responsive at the supported desktop and
  Pi display sizes.
- [ ] Verify clean startup and shutdown, including repeated home/recognition
  transitions and camera release.

### Path, packaging, and artifact handling

- [ ] Fix app-side default paths. `app/src/engine/rebuild_release.py` defaults
  to `app/src/engine/db`, while the checked-in database is under `app/db`;
  the app-side setup-1 loader likewise assumes models and DB files beside the
  engine module.
- [ ] Decide how the app-side release builder is invoked and make its default
  database, output root, model root, and config root agree with that decision.
- [ ] Generate and validate an actual `enrollment/current.json` release before
  declaring the Kivy recognition screen runnable.
- [ ] Decide whether the large existing LBPH YAML should be migrated to Git LFS.
  The `.gitattributes` rule was added after the YAML was committed and does not
  migrate existing history by itself.
- [ ] Decide whether ONNX weights belong in Git, Git LFS, or a reproducible
  download/bootstrap step. Document checksums and avoid committing duplicate
  model copies without a reason.
- [ ] Split or condition platform dependencies. The current requirements list
  includes Windows-oriented Kivy packages while the Raspberry Pi path is only
  commented (`picamera2`) and the app still uses `cv.VideoCapture`.
- [ ] Add a reproducible install/run guide for Windows PC and Raspberry Pi,
  including OpenCV/Kivy/Python compatibility and model acquisition.

## Open decisions

### Architecture ownership

- [ ] Choose the canonical runtime stack: the top-level scripts, the Kivy
  `app/src` stack, or a shared package used by both. Remove or explicitly
  synchronize duplicate copies of `hybrid.py`, `hybrid_rpi.py`, `quality.py`,
  `lbph_config.py`, `rebuild_release.py`, and `face_aligner.py`.
- [ ] Decide whether the pose subsystem remains a standalone adapter or gains
  a backward-compatible detector-row method on `HybridCascade`. Prefer a
  separate adapter if changing the recognition cascade risks regressions.
- [ ] Decide the owner of face detection in the integrated enrollment loop so
  YuNet runs once per frame and its box/landmarks are shared by quality,
  recognition, pose, and alignment.
- [ ] Decide whether app-side SFace alignment and feature extraction should
  share one recognizer object; the legacy app path currently constructs SFace
  through both `FaceAligner` and `HybridCascade`.

### Head-pose backend and semantics

- [ ] Evaluate `yunet_geometry` first using the existing five landmarks and
  multiple normalized signals: PnP angles, nose/eye/mouth ratios, asymmetry,
  scale normalization, and robust rolling statistics.
- [ ] Evaluate `openvino_adas` using the official
  `head-pose-estimation-adas-0001` XML/BIN model through OpenCV DNN. Discover
  outputs safely, normalize signs to subject perspective, and fall back clearly
  when files or runtime support are absent.
- [ ] Keep MediaPipe an optional experiment only. Adopt it only after proving
  installation and runtime feasibility on the actual Pi OS/Python architecture.
- [ ] Select the default backend from measurements, not preference. Compare
  both candidates on at least two people across multiple sessions.
- [ ] Decide whether signs and thresholds are established by an explicit config,
  guided labeled calibration, or both. Persist the source and calibration
  provenance; never silently learn a universal user-specific mapping.
- [ ] Decide the initial stable-hold duration, yaw entry/exit hysteresis,
  vertical yaw guard, roll limit, and confidence floor from labeled sessions.
- [ ] Decide whether `UNCERTAIN` or `TRANSITION` is the public transitional
  label; keep it distinct from a confirmed target pose.
- [ ] Decide how preview mirroring is configured and persisted. Display state
  must never alter subject-perspective inference labels.

### Recognition candidate and release policy

- [ ] Decide whether the app-side R3 gate should implement the documented
  relative LBPH top-1/top-2 margin rule. The current app implementation has
  `margin_min` in config/docs but does not calculate the margin.
- [ ] Revalidate the R3 `tau_accept`, `tau_reject`, quality thresholds, and
  SFace thresholds against this repository's deployment data and Pi camera
  conditions before promotion.
- [ ] Decide whether the candidate release pointer needs an explicit promote,
  rollback, and provenance command rather than manual file replacement.
- [ ] Decide the retention/privacy policy for recognition, calibration, and
  enrollment logs, including who may access them and when they are deleted.

## Implementation slices

### Slice A — stabilize the pose core

- [ ] Define small data types for raw backend output, normalized pose,
  confidence, calibration metadata, and target progress.
- [ ] Define a common backend interface with `estimate(frame_bgr,
  face_row=None, timestamp_s=None)`.
- [ ] Keep raw estimation, sign normalization, calibration, temporal filtering,
  classification, target acceptance, logging, and drawing in separate layers.
- [ ] Add explicit reset semantics between visitors and safe behavior after face
  loss.
- [ ] Make the core independent of Kivy, OpenCV UI calls, LBPH, SFace names,
  voice capture, and mutable global state; keep it safe for worker-thread use.

### Slice B — geometry backend and guided calibration

- [ ] Implement the improved `yunet_geometry` backend with optional reuse of an
  existing YuNet row.
- [ ] Add robust neutral calibration for offsets only.
- [ ] Add guided FRONT/LEFT/RIGHT/UP/DOWN calibration that learns vertical and
  horizontal signs, fits candidate thresholds, records distribution overlap,
  and fails honestly when UP/DOWN or LEFT/RIGHT are not separable.
- [ ] Add yaw enter/exit thresholds, vertical yaw guard, roll rejection,
  confidence, hysteresis, and elapsed-time stable-hold logic.
- [ ] Return `FRONT`, `LEFT`, `RIGHT`, `UP`, `DOWN`, and `UNCERTAIN`/`TRANSITION`
  without one-frame label flicker.

### Slice C — learned backend comparison

- [ ] Add reproducible Open Model Zoo ADAS model setup instructions or a small
  setup script; do not commit large generated weights unless policy approves.
- [ ] Implement XML/BIN loading through OpenCV DNN with documented BGR
  preprocessing, input size, safe output discovery, sign normalization, and a
  clear geometry fallback.
- [ ] Record which backend produced each estimate and its backend confidence.
- [ ] Benchmark MediaPipe only if the Pi install remains practical after A/B
  evaluation; do not make it a default by assumption.

### Slice D — stage-conditioned pose API

- [ ] Implement `target_status(target)` for future enrollment stages.
- [ ] Report target, satisfied state, progress from 0 to 1, stable milliseconds,
  required milliseconds, confidence, and a human-readable reason.
- [ ] Block target acceptance unless the requested direction remains stable for
  the configured hold time and meets confidence/guard rules.
- [ ] Preserve the global direction label for the pose laboratory and overlay.

### Slice E — persistent configuration and controls

- [ ] Add dedicated `app/config/head_pose.json` with schema version, backend choice,
  signs, baselines, entry/exit thresholds, guards, smoothing, hold duration,
  detector/crop settings, mirror state, and calibration provenance.
- [ ] Implement controls with clear temporary-versus-saved state:
  - [ ] `C` — neutral recenter only.
  - [ ] `G` — guided full calibration.
  - [ ] `X` — explicitly flip horizontal mapping and persist it.
  - [ ] `Y` — explicitly flip vertical mapping and persist it.
  - [ ] `S` — save/print summary.
  - [ ] `ESC` — cancel active burst or guided step.
  - [ ] `Z` — undo the exact last completed burst.
  - [ ] `Q` — quit cleanly.

### Slice F — pose laboratory and evaluation

- [x] Keep `pose-detection/ex-pc-detect.py` as a thin laboratory/CLI, not the
  pose implementation.
- [ ] Support pose-only by default and explicit recognition mode, including:
  `--pose-only --backend auto`, `yunet_geometry`, and `openvino_adas`, plus
  `--with-recognition`.
- [ ] Rate-limit pose updates to a configurable 10–15 Hz on Pi while reusing
  the latest estimate for a fluid preview.
- [ ] Add session IDs, burst IDs, target labels, timestamps, prediction,
  features, confidence, face score/box, face-loss events, config version,
  latency, FPS, mirror state, and undo/cancel events to raw logs.
- [ ] Keep calibration logs separate from final validation logs and never commit
  attendee images or personal data.
- [ ] Generate summary JSON with per-frame and stable-window confusion matrices,
  per-class precision/recall/balanced accuracy, FRONT false-positive rates,
  axis separation, flicker, time-to-stable, latency p50/p95/p99, and effective
  FPS.
- [ ] Add a helper to fit signs and candidate thresholds from one or more logs
  using robust medians and percentiles.

### Slice G — Kivy feature completion

- [ ] Remove the legacy/r1 LBPH setup choice from the user-facing app. Launch
  canonical config2 / `r3_n8_g6x6` directly; retain r1 only as an internal
  rollback or comparison fixture if needed.
- [ ] Add the enrollment screen and connect it to `target_status()` without
  coupling pose logic to Kivy.
- [ ] Implement the five-stage flow: FRONT, visitor LEFT, visitor RIGHT, UP,
  and DOWN, with stable acceptance, progress, retry, timeout, and manual
  operator fallback.
- [ ] Add voice/name capture only after the visual capture flow is reliable;
  define offline behavior and a manual naming fallback.
- [ ] Persist reviewed enrollment samples and rebuild a candidate release
  atomically with manifest/provenance validation.
- [ ] Add the read-only identity-management screen.
- [ ] Connect camera/database settings to the actual runtime state.

### Slice H — performance, packaging, and release

- [ ] Measure Pi CPU, RAM, detector/pose/recognition latency, and loop FPS for
  pose-only and full recognition modes.
- [ ] Compare one-detector versus duplicated-detector paths and document the
  measured benefit.
- [ ] Add platform-specific dependency files or install instructions and test
  the camera backend on the target Raspberry Pi OS.
- [ ] Add model checksums, release manifests, promotion/rollback instructions,
  and a clean candidate rebuild command.
- [ ] Update README/docs after runtime behavior is implemented; do not describe
  placeholder screens or unvalidated thresholds as production features.

## Verification checklist

- [ ] Deterministic tests cover neutral offset recalibration without silent sign
  changes.
- [ ] Guided calibration learns UP/DOWN and LEFT/RIGHT signs from labeled
  medians and persists/restores them.
- [ ] Strong yaw priority, vertical guard, hysteresis, confidence, and stable
  hold behave correctly at boundaries.
- [ ] Face loss resets or decays state as specified.
- [ ] Mirrored display does not swap semantic LEFT/RIGHT.
- [ ] ESC cancellation and Z undo remove no other bursts.
- [ ] Missing ADAS files produce an actionable fallback/error.
- [ ] Passing a YuNet row avoids another detector invocation.
- [ ] Existing recognition tests still pass unchanged.
- [ ] At least two people and multiple sessions meet the measured target where
  achievable: stable-window balanced accuracy near 95%, axis false triggers
  below 5%, low flicker, median time-to-stable below 800 ms, and usable Pi
  latency/FPS. Report misses honestly.
- [ ] Run syntax checks, all repository tests, pose tests, and a real PC/Pi
  smoke test before integrating pose into enrollment.
