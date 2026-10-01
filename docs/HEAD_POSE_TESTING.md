# Head-pose laboratory

## Kivy guided pose scan

Start the conference UI from the repository root:

```powershell
python app/main.py
```

`Add Identity` is currently a guided PnP pose scan, not an enrollment action.
It checks FRONT, YOUR LEFT, YOUR RIGHT, UP and DOWN, then waits for `Done`.
It writes no photos, identities, templates, logs or databases. The saved device
profile remains unchanged because the participant's straight-ahead adjustment is
temporary.

Use `Device Setup` for a full operator-only PnP calibration. It records no
images. Hold each requested position, then press its key twice to start it:
`F`, `L`, `R`, `U`, `D`. The next position waits until you double-tap its own
key; it never begins on its own. `Esc` cancels the current position, `G` starts
the five-position setup again, and `S` saves only after all five succeed. The
ignored local profile is the only file replaced; the checked-in default profile
is never overwritten.

If a labelled UP or DOWN recording still overlaps the saved FRONT allowance,
Device Setup keeps the other four recordings and asks only for that position
again. For DOWN, the on-screen note reports whether raw PnP pitch is far enough
from FRONT and warns when the raw yaw suggests turning sideways. `G` is the
only control that deliberately clears the whole five-position setup.

Every Device Setup attempt with recorded positions writes a numeric diagnostic
file under `logs/head_pose/device_setup/`. It includes PnP yaw/pitch samples,
their ranges, the final status, and any rejection reason—never camera frames or
face images. Send the newest file after a rejected setup so its measurements can
be checked.

For a Raspberry Pi without a display, the same flows run without importing Kivy:

```powershell
python pose-detection/pose_cli.py --camera 0
python pose-detection/pose_cli.py --calibrate --camera 0
python pose-detection/pose_cli.py --picamera2
```

Headless participant scans end after all five poses pass; Ctrl+C cancels. A
missing or incompatible profile tells the operator to run `--calibrate` first.
Participant scans require the camera resolution used by Device Setup; the Kivy
screen defaults to 640×480 and headless mode checks the actual delivered size.

## Quick participant-flow test

```powershell
python pose-detection/ex-pc-detect.py --purpose test --backend yunet_geometry
```

No Kivy installation is needed. This uses a simple OpenCV screen with clickable
buttons and keyboard shortcuts. It loads the existing operator config, then:

1. Click Start or press Space. Look naturally at the camera lens.
2. Hold still for about 1.2 seconds. The program establishes this participant's
   neutral position and saves the FRONT photo.
3. Follow YOUR LEFT, YOUR RIGHT, UP and DOWN. Each matching pose must remain
   steady for the configured hold (normally 0.5 seconds) before its photo is saved.
4. Review all five photos. Click Finish / Q or Restart / R.

Each image is saved as an original, unmirrored PNG in a dedicated system temporary
folder. Preview mirroring affects display only. Finish, cancel, window close,
restart, and handled camera errors delete this run's folder and clear its images.
Other photos and logs are not deleted. Abrupt power loss or forcibly killing the
process can leave temporary files for the OS to clean. This mode does not write
calibration/validation logs, enrollment artifacts, or the participant's neutral
position to the saved operator config. Restart requires a new neutral check.

If a requested pose is not accepted after the configured timeout (10 seconds), a
Manual photo button / M appears. It requires a fresh visible face and marks that
photo as manual in the review. It does not pretend automatic detection succeeded.
If operator signs have never been established, the screen asks for full operator
calibration first; it cannot infer UP/DOWN sign from a neutral face alone.

The FRONT fix adds optional `pitch_limits.up` and `pitch_limits.down`, each with
its own `enter` and `exit`, while still reading older shared-threshold configs.
New guided fits preserve the observed FRONT range plus `pitch_front_padding`
(default 3 control units), then fit each side independently. Recenter changes
offsets without shrinking this allowance or changing signs. A fit that cannot
separate a direction from this allowance is rejected.

The current ignored local profile was refitted from the latest reported session,
with its previous contents saved as `app/config/head_pose.local.before-front-fix-*.json`.
Yaw settings and signs were preserved. These recordings now count as development
data: replay on them is a regression check, not fresh accuracy validation. The
checked-in default config contains no personal measurements. The brief neutral
check does not establish that these settings generalize to other participants.

For a fresh logged FRONT/UP/DOWN check after trying the flow:

```powershell
python pose-detection/ex-pc-detect.py --purpose validation --backend yunet_geometry
```

Record FRONT, UP, FRONT, DOWN, FRONT using the matching key twice for each burst.

This implements a test subsystem, not a validated enrollment detector. Recognition
code, thresholds, LBPH models, galleries, and research results are not changed.

## Decision and evidence

`auto` currently uses `yunet_geometry` as a provisional offline baseline. Both
backends share YuNet boxes, robust calibration, filtering, hysteresis, elapsed
hold timing, and target acceptance. Geometry exposes roll-corrected normalized
eye/nose/mouth features plus approximate-camera PnP. Calibration chooses one
separable feature per axis; selecting beats an unvalidated fixed fusion formula.
Geometry control units are **not anatomical degrees**. Angle features retain
degrees; normalized geometry features are rescaled to about 25 control units at
the labeled side poses. Entry thresholds are fitted between quiet-pose p95 and
active-pose p10, with weaker exit thresholds. No universal person-specific fit is
shipped. Initial 22/15 yaw and 12/8 pitch settings are provisional and inactive
until signs and a neutral baseline are established.

`openvino_adas` loads the official XML/BIN through OpenCV DNN, resolves named
outputs (`fc_y`, `fc_p`, `fc_r`, including historical `angle_*_fc` aliases), and
probes execution during startup. Input is unscaled BGR, NCHW 1x3x60x60, no RGB
swap or mean subtraction. See the [official model documentation](https://docs.openvino.ai/2023.3/omz_models_model_head_pose_estimation_adas_0001.html).
The acquisition script pins the 2023.0 FP32 pair and verifies the SHA384 hashes
and sizes from the [OMZ 2023.3.0 manifest](https://github.com/openvinotoolkit/open_model_zoo/blob/2023.3.0/models/intel/head-pose-estimation-adas-0001/model.yml).
Model license: Apache-2.0. Files remain ignored and runtime performs no download.

On this Windows environment, Python OpenCV 4.13.0 / NumPy 2.4.3:

- Official model acquisition and both checksum checks succeeded.
- Actual ADAS loading failed because this OpenCV build cannot execute XML/BIN.
  The actionable warning and geometry fallback were exercised. Installing the
  Python `openvino` package alone does not add support to an OpenCV wheel.
- Geometry accepted a synthetic existing face row without detector invocation.
  For 200 warmed calls at 640x480, p50/p95/p99 were 0.139/0.331/0.434 ms.
  This excludes face detection and uses synthetic landmarks, not real faces.
- One blank-frame cold detector call returned no face in 62.1 ms. This is a smoke
  check, not an FPS estimate.
- Existing historical summaries show missing FRONT samples; pitch proxy UP/DOWN
  distributions overlap in one session. They cannot validate the replacement.
- No two-person validation or Raspberry Pi measurement was performed. No 95%
  accuracy or Pi throughput claim is made. ADAS accuracy remains unmeasured.
- `python -m unittest discover -s test -p "test_*.py" -v`: 25 tests passed,
  including recognition contract tests, calibration/sign persistence, temporal
  behavior, mirrored display, burst undo, replay, and a mocked headless CLI run.
  Syntax compilation and `git diff --check` also passed. No live camera GUI test
  or full recognition inference was performed.

No MediaPipe, PyTorch, cloud calls, or additional runtime dependencies were added.
ADAS is an available experiment, not the selected validated winner.

## Root causes in the previous experiment

Neutral reset cleared offsets and histories but left `_yaw_sign` and
`_pitch_sign` unchanged. Both signs started at +1 and were not persisted. Thus
neutral recenter could not resolve a semantic reversal. The PnP pitch convention
and the comment asserting positive landmark pitch means UP were unverified.
Historical labeled medians differ between PnP and landmark pitch; they do not
justify globally hardcoding either sign. Yaw entered at 10 degrees before pitch
classification. Median smoothing alone provided neither entry/exit hysteresis nor
time-based confirmation. The tester always ran recognition and another detector.
Ground-truth capture lacked burst IDs and exact undo; ESC was not bound.

## PC setup and commands

Run from this repository, using Python 3.10 or newer:

```powershell
python -m venv .venv-pose
.\.venv-pose\Scripts\Activate.ps1
python -m pip install numpy opencv-contrib-python
python pose-detection/setup_head_pose_model.py
python -m unittest discover -s test -p "test_*.py" -v
python pose-detection/ex-pc-detect.py --pose-only --backend auto
python pose-detection/ex-pc-detect.py --pose-only --backend yunet_geometry
python pose-detection/ex-pc-detect.py --pose-only --backend openvino_adas
python pose-detection/ex-pc-detect.py --with-recognition --backend auto --recognition-setup 1
```

The existing requirements file contains platform-specific GUI dependencies. Pose
only needs NumPy and OpenCV; install those directly as above. Recognition setup 2
also requires existing r3 enrollment artifacts; this task does not rebuild them.
With recognition enabled, the comparison still runs two detectors and explicitly
reports that fact. The default pose-only loop uses one. The future integration
passes `face_row` and avoids the pose detector entirely.

Calibrate a device/person with a local configuration:

```powershell
python pose-detection/ex-pc-detect.py --pose-only --backend yunet_geometry --purpose calibration --config app/config/head_pose.local.geometry.json
```

Press G, then hold FRONT and press F twice. Repeat for your LEFT (L), RIGHT (R),
UP (U), DOWN (D). Each step requires two matching keypresses, then a one-second
settle and three-second capture. Wait for completion before moving. The preview
shows the actual label before recording. Failed separation keeps the old config
and reports the failing axis. Retry G or test another backend. S saves the fitted
config and summary. Quit Q; start a separate validation session:

```powershell
python pose-detection/ex-pc-detect.py --pose-only --backend yunet_geometry --purpose validation --config app/config/head_pose.local.geometry.json
```

The shipped config has null signs and an empty baseline, so startup shows
UNCERTAIN until calibrated. X/Y explicitly set/flip and persist horizontal/vertical
signs, respectively; when unset, the first flip establishes -1. Explicit setup
also needs C neutral capture. For ADAS explicit configuration, select `adas_yaw`
and `adas_pitch` with scales 1, signs verified by the operator, and use C. Guided
calibration selects these automatically. Backend changes invalidate incompatible
calibration. All directions refer to the subject, independent of `--mirror` or
`--no-mirror`. Inference always consumes the original frame.

Controls:

- C: neutral offsets only; press F twice to capture. Signs remain unchanged.
- G: full guided sign/feature/threshold calibration.
- X/Y: explicit mapping flip and immediate persistence.
- S: summary; also saves config in calibration mode.
- ESC: cancels the armed/active burst; incomplete rows are excluded.
- Z: undoes exactly the latest completed burst, only when no burst is active.
  In calibration mode, this invalidates the derived calibration, including a
  previously saved local config, so excluded evidence cannot keep driving labels.
- Q: closes camera/logs and writes summary. An unfinished burst is cancelled.
- M: after `--target` timeout, logs an explicit manual operator event; never
  counts as automatic success and does not create enrollment artifacts.

Validation mode locks calibration controls. Use a separate calibration session to
change mapping/thresholds. Logs are separated into calibration/validation folders.

## Raspberry Pi setup and measurements

Use 64-bit Raspberry Pi OS, Python >=3.10, with a desktop for the interactive UI.
USB camera path:

```bash
python3 -m venv .venv-pose
. .venv-pose/bin/activate
python -m pip install numpy opencv-contrib-python
python pose-detection/ex-pc-detect.py --pose-only --backend yunet_geometry --camera 0 --width 640 --height 480 --pose-hz 12 --purpose calibration
```

CSI camera path, using the OS Picamera2 package:

```bash
sudo apt install python3-picamera2 python3-venv
python3 -m venv --system-site-packages .venv-pose-csi
. .venv-pose-csi/bin/activate
python -m pip install opencv-contrib-python
python pose-detection/ex-pc-detect.py --pose-only --backend yunet_geometry --picamera2 --width 640 --height 480 --pose-hz 12 --purpose calibration
```

Picamera2 RGB888 supplies BGR byte order to OpenCV, as documented in the
[Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).
Pi packaging and camera access were not tested here. Check that OpenCV supports
the bundled YuNet 2023 ONNX model; older OS OpenCV packages may not.

After calibration, benchmark both requested backends and inspect **actual_backend**:

```bash
python pose-detection/setup_head_pose_model.py
/usr/bin/time -v python pose-detection/ex-pc-detect.py --pose-only --backend yunet_geometry --picamera2 --headless --seconds 60 --width 640 --height 480 --pose-hz 12
/usr/bin/time -v python pose-detection/ex-pc-detect.py --pose-only --backend openvino_adas --picamera2 --headless --seconds 60 --width 640 --height 480 --pose-hz 12
```

Replace `--picamera2` with `--camera 0` for USB. Headless mode collects runtime
measurements but cannot receive labeled keypresses. JSON summaries report stage
latency p50/p95/p99, effective loop FPS, process CPU percentage (100% = one core),
and peak RSS on Linux. Preview frames reuse the latest estimate between pose
updates; target acceptance rejects stale estimates. Keep actual pose rate high
enough that consecutive samples are less than `max_gap_s` apart.

## Clean validation and backend comparison

Use at least two people, each in multiple sessions, with fixed camera distance,
lighting and resolution. Calibrate separately per person/device; save config,
then restart for held-out validation. Record all five labels, at least five
three-second bursts each in varied order. Include FRONT, UP and DOWN while
watching horizontal false triggers, and diagonal poses for yaw priority. Use ESC
for mistakes before completion or Z afterward. Never relabel a prediction as
truth. Store only the intended operator labels. Logs contain features, not images
or recognition names; keep all logs and any separately recorded videos local.

Summaries are derived from complete, non-cancelled, non-undone burst IDs. Raw
sample `undone=false` is the state at append time; later undo events are
authoritative. Offline analysis excludes undone bursts deterministically.
Per-frame confusion includes TRANSITION, UNCERTAIN and NO_FACE as errors.
Stable-window scoring excludes the fixed configured initial hold duration, then
scores complete non-overlapping 500ms bins. Mixed labels or less than 250ms
observation coverage count as UNCERTAIN; no-face windows count as errors. Missing
classes are reported, never treated as successes. Time-to-stable starts at the
first recorded observation, with tracker history reset when recording starts.
Failures remain null per burst and must be reported alongside successful timings.

```bash
python pose-detection/evaluate_head_pose.py logs/head_pose/validation/SESSION.jsonl
python pose-detection/evaluate_head_pose.py logs/head_pose/calibration/SESSION.jsonl --fit app/config/head_pose.local.geometry.json --backend yunet_geometry
python pose-detection/evaluate_head_pose.py logs/head_pose/calibration/ADAS_SESSION.jsonl --fit app/config/head_pose.local.adas.json --backend openvino_adas
```

Replace SESSION with actual printed UUID filenames. The fit command accepts
multiple calibration logs, rejects validation logs, and uses robust labeled
medians/percentiles. Raw ADAS observations also retain geometry features, enabling
scoring both fitted configurations on the exact same observations:

```bash
python pose-detection/evaluate_head_pose.py logs/head_pose/validation/ADAS_SESSION.jsonl --replay-config app/config/head_pose.local.geometry.json
python pose-detection/evaluate_head_pose.py logs/head_pose/validation/ADAS_SESSION.jsonl --replay-config app/config/head_pose.local.adas.json
```

Replay measures classification only; latency/FPS fields are explicitly null.
It cannot invent ADAS outputs from a geometry log. If ADAS cannot run on the PC,
same-observation learned-model comparison remains blocked by runtime support.
Compare true backend execution separately on Pi. Prefer the simplest backend
achieving >=95% stable-window balanced accuracy, <5% horizontal false triggers
on FRONT/UP/DOWN, <5% vertical false triggers on FRONT, low held-pose flicker,
and median successful time-to-stable <800ms with a low failure rate. If targets
fail, report actual metrics and retain the operator fallback. Do not promote
based on calibration scores, synthetic tests, or absent-class accuracy.

## Integration boundary and rollback

`HeadPoseTracker.estimate(frame_bgr, face_row=row, timestamp_s=monotonic_time)`
accepts the exact 15-value YuNet row. `target_status("LEFT")` reports satisfied,
progress, stable_ms, required_ms, confidence and reason. Call `reset()` between
visitors; `reset(clear_baseline=True)` also clears person offsets. Calibrate each
visitor/device as needed. The tracker serializes access with an RLock; keep one
backend owned by one tracker and use its methods from a worker. Returned raw
diagnostics should be treated as immutable. Confidence is a detector/jitter
heuristic, not a calibrated probability of correct pose.

Next integration step: obtain one YuNet result in the enrollment worker, pass its
selected row to pose, and allow capture only when `target_status(stage).satisfied`.
Pass face loss through `update(None, timestamp)` when an external detector returns
no face; `face_row=None` means run the internal detector. Bind the existing manual
capture action after `manual_timeout_s`. Enrollment itself is outside this task.

Rollback: stop the lab and use the unchanged recognition entry points. Pose files
and ignored local configs/logs/models are isolated. Revert only this task's source
patch if reverting the tester; preserve any concurrent user changes. No commit,
push, retraining or artifact regeneration was performed.
