"""Offline pose backends and time-based subject-perspective tracking.

Positive normalized yaw means subject LEFT; positive pitch means UP. Calibrated
geometry controls are not anatomical degrees. No UI or recognition dependency.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Protocol
import warnings
import cv2 as cv
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2]
ROOT = APP_ROOT.parent
POSE_CONFIG_DIR = APP_ROOT / "config"
LABELS = ("FRONT", "LEFT", "RIGHT", "UP", "DOWN")
MODEL_POINTS = np.array([[-30,-30,-30], [30,-30,-30], [0,0,0],
                         [-25,30,-20], [25,30,-20]], dtype=np.float64)


class CalibrationError(ValueError):
    """A calibration rejection that identifies the recordings safe to replace."""

    def __init__(self, message, *, retry_labels=()):
        super().__init__(message)
        self.retry_labels = tuple(retry_labels)


def load_config(path=POSE_CONFIG_DIR / "head_pose.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(c):
    if c["schema_version"] != 1 or c["backend"] not in ("auto", "yunet_geometry", "openvino_adas"):
        raise ValueError("Unsupported pose schema/backend")
    for axis in ("yaw", "pitch"):
        if c["signs"][axis] not in (-1, 1, None):
            raise ValueError("Signs must be -1, +1, or null (unverified)")
        if not np.isfinite([c[axis+"_exit"],c[axis+"_enter"]]).all() or not 0 < c[axis + "_exit"] < c[axis + "_enter"]:
            raise ValueError("Require 0 < exit < enter")
        if not np.isfinite(c["scales"][axis]) or c["scales"][axis] <= 0:
            raise ValueError("Axis scales must be finite and positive")
    if not 0 < c["vertical_yaw_guard"] < c["yaw_enter"]:
        raise ValueError("Vertical yaw guard must be below strong yaw entry")
    if not 0 <= c["min_confidence"] <= 1:
        raise ValueError("Invalid minimum confidence")
    for key in ("stable_hold_s", "max_gap_s", "smoothing_s", "pose_hz", "roll_limit", "manual_timeout_s"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if c["detector"]["max_side"] < 32 or c["crop_scale"] <= 0:
        raise ValueError("Invalid detector/crop size")
    if not all(np.isfinite(v) for v in c["baseline"].values()):
        raise ValueError("Invalid baseline")
    if "pitch_limits" in c:
        for direction in ("up", "down"):
            limits = c["pitch_limits"][direction]
            if not np.isfinite([limits["enter"], limits["exit"]]).all() or not 0 < limits["exit"] < limits["enter"]:
                raise ValueError("Pitch limits require finite 0 < exit < enter")
    if not np.isfinite(c.get("pitch_front_padding", 3.0)) or c.get("pitch_front_padding", 3.0) < 0:
        raise ValueError("Pitch FRONT padding must be finite and nonnegative")


def save_config(config, path):
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class RawPose:
    backend: str
    yaw: float
    pitch: float
    roll: float
    features: dict[str, float]
    confidence: float
    face_score: float
    bbox: tuple[int, int, int, int]
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class PoseEstimate:
    direction: str
    yaw: float
    pitch: float
    roll: float
    confidence: float
    timestamp_s: float
    candidate: str
    stable_ms: float
    raw: RawPose

    @property
    def bbox(self):
        return self.raw.bbox


@dataclass(frozen=True)
class TargetPoseStatus:
    target: str
    satisfied: bool
    progress: float
    stable_ms: float
    required_ms: float
    confidence: float
    reason: str


class HeadPoseBackend(Protocol):
    name: str

    def estimate(self, frame_bgr: np.ndarray, *, face_row: np.ndarray | None = None,
                 timestamp_s: float | None = None) -> RawPose | None: ...


class YuNetGeometry:
    name = "yunet_geometry"

    def __init__(self, config, model_path=None):
        self.config = config
        self.model_path = Path(model_path or ROOT / config["detector"]["model"])
        self.detector = None

    def _row(self, frame, row):
        if row is None:
            d = self.config["detector"]
            if self.detector is None:
                if not self.model_path.is_file():
                    raise FileNotFoundError(f"YuNet model missing: {self.model_path}")
                self.detector = cv.FaceDetectorYN.create(str(self.model_path), "", (320,320),
                                                        d["score"], d["nms"], d["top_k"])
            h, w = frame.shape[:2]
            scale = min(1., d["max_side"] / max(h, w))
            small = cv.resize(frame, (round(w*scale), round(h*scale))) if scale < 1 else frame
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self.detector.detect(small)
            if rows is None or not len(rows):
                return None
            row = max(rows, key=lambda r: r[2]*r[3]).copy()
            row[:14:2] *= w / small.shape[1]
            row[1:14:2] *= h / small.shape[0]
        row = np.asarray(row, dtype=float).reshape(-1)
        if row.size != 15 or not np.isfinite(row).all():
            raise ValueError("Expected finite 15-value YuNet row")
        if min(row[2:4]) < self.config["detector"]["min_face_px"] or row[14] < self.config["detector"]["score"]:
            return None
        return row

    def estimate(self, frame_bgr, *, face_row=None, timestamp_s=None):
        start = time.perf_counter()
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        row = self._row(frame_bgr, face_row)
        detected = time.perf_counter()
        if row is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1 = np.maximum(0, np.floor(row[:2])).astype(int)
        x2, y2 = np.minimum([w,h], np.ceil(row[:2]+row[2:4])).astype(int)
        if x2 <= x1 or y2 <= y1:
            return None
        p = row[4:14].reshape(5,2)
        eye, mouth = (p[0]+p[1])/2, (p[3]+p[4])/2
        distance = float(np.linalg.norm(p[1]-p[0]))
        if distance < 4:
            return None
        horizontal = (p[1]-p[0])/distance
        vertical = np.array([-horizontal[1], horizontal[0]])
        span = float((mouth-eye) @ vertical)
        if span < 4:
            return None
        en, nm = float((p[2]-eye) @ vertical), float((mouth-p[2]) @ vertical)
        features = {
            "nose_eye_x": float((p[2]-eye) @ horizontal / distance),
            "nose_mouth_x": float((p[2]-mouth) @ horizontal / distance),
            "eye_nose_ratio": en/span, "nose_mouth_ratio": nm/span,
            "vertical_ratio": (nm-en)/span,
            "eye_asymmetry": float((np.linalg.norm(p[2]-p[0])-np.linalg.norm(p[2]-p[1]))/distance),
            "mouth_asymmetry": float((np.linalg.norm(p[2]-p[3])-np.linalg.norm(p[2]-p[4]))/distance),
            "eye_box_ratio": distance/row[2], "span_box_ratio": span/row[3],
        }
        roll = float(np.degrees(np.arctan2(horizontal[1], horizontal[0])))
        yaw = pitch = 0.
        k = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=float)
        try:
            ok, rv, tv = cv.solvePnP(MODEL_POINTS, p.copy(), k, None,
                                     flags=getattr(cv,"SOLVEPNP_SQPNP",cv.SOLVEPNP_EPNP))
            if ok and tv[2,0] > 0:
                rotation, _ = cv.Rodrigues(rv)
                angles = cv.RQDecomp3x3(rotation)[0]
                pitch, yaw = float(angles[0]), float(angles[1])
                projected, _ = cv.projectPoints(MODEL_POINTS, rv, tv, k, None)
                error = float(np.linalg.norm(projected.reshape(5,2)-p, axis=1).mean()/distance)
                if np.isfinite([yaw,pitch,error]).all() and error < .2:
                    features.update(pnp_yaw=yaw, pnp_pitch=pitch, pnp_roll=float(angles[2]))
        except cv.error:
            pass  # Independent geometry remains usable when approximate PnP fails.
        features["roll"] = roll
        return RawPose(self.name, yaw, pitch, roll, features, float(np.clip(row[14],0,1)),
                       float(row[14]), (int(x1),int(y1),int(x2-x1),int(y2-y1)),
                       {"detector": (detected-start)*1000, "geometry": (time.perf_counter()-detected)*1000})


class OpenVINOADAS(YuNetGeometry):
    name = "openvino_adas"

    def __init__(self, config, model_path=None):
        super().__init__(config, model_path)
        xml = ROOT / config["adas_xml"]
        if not xml.is_file() or not xml.with_suffix(".bin").is_file():
            raise RuntimeError("ADAS XML/BIN missing; run python pose-detection/setup_head_pose_model.py")
        try:
            self.net = cv.dnn.readNet(str(xml), str(xml.with_suffix(".bin")))
            names = list(self.net.getUnconnectedOutLayersNames())
            self.outputs = []
            for axis in "ypr":
                matches = [n for n in names if n.split(":")[0].split("/")[-1] in (f"fc_{axis}", f"angle_{axis}_fc")]
                if len(matches) != 1:
                    raise RuntimeError(f"Cannot identify ADAS {axis} output in {names}")
                self.outputs.append(matches[0])
            self._angles(np.zeros((60,60,3), dtype=np.uint8))
        except cv.error as exc:
            raise RuntimeError("OpenCV DNN cannot execute ADAS XML/BIN on this build; use yunet_geometry or an OpenCV build with OpenVINO support") from exc

    def _angles(self, crop):
        self.net.setInput(cv.dnn.blobFromImage(crop, 1., (60,60), (0,0,0), swapRB=False, crop=False))
        values = self.net.forward(self.outputs)
        angles = [float(np.asarray(v).reshape(-1)[0]) for v in values]
        if not np.isfinite(angles).all():
            raise RuntimeError("ADAS returned non-finite angles")
        return angles

    def estimate(self, frame_bgr, *, face_row=None, timestamp_s=None):
        raw = super().estimate(frame_bgr, face_row=face_row, timestamp_s=timestamp_s)
        if raw is None:
            return None
        start = time.perf_counter()
        x,y,w,h = raw.bbox
        side = max(w,h)*self.config["crop_scale"]
        fh,fw = frame_bgr.shape[:2]
        left,top = max(0,int(x+w/2-side/2)), max(0,int(y+h/2-side/2))
        right,bottom = min(fw,int(x+w/2+side/2)), min(fh,int(y+h/2+side/2))
        raw.yaw, raw.pitch, raw.roll = self._angles(frame_bgr[top:bottom,left:right])
        raw.features.update(adas_yaw=raw.yaw, adas_pitch=raw.pitch, roll=raw.roll)
        raw.latency_ms["adas"] = (time.perf_counter()-start)*1000
        return raw


def make_backend(config, model_path=None):
    # auto stays provisional geometry until a measured validation selects ADAS.
    if config["backend"] == "openvino_adas":
        try:
            return OpenVINOADAS(config, model_path)
        except RuntimeError as exc:
            warnings.warn(f"{exc}; falling back to yunet_geometry. Recalibration required.", RuntimeWarning)
    return YuNetGeometry(config, model_path)


def fit_pitch_limits(samples, config, key, sign, scale, baseline, *, active_percentile=10, front_padding=None):
    """Keep the observed FRONT range plus padding; fit UP/DOWN independently."""
    normalized = {label: sign * (np.asarray([r[key] for r in samples[label]], dtype=float)-baseline)*scale
                  for label in ("FRONT", "UP", "DOWN")}
    if any(len(v) < 8 or not np.isfinite(v).all() for v in normalized.values()):
        raise ValueError("Pitch fitting needs 8 finite samples per FRONT/UP/DOWN")
    limits = {}
    for direction, multiplier, label in (("up", 1, "UP"), ("down", -1, "DOWN")):
        # A participant may settle on either side of the sampled neutral median.
        quiet = float(np.percentile(np.abs(normalized["FRONT"]), 95))
        quiet += config.get("pitch_front_padding", 3.0) if front_padding is None else front_padding
        active = float(np.percentile(multiplier*normalized[label], active_percentile))
        if active <= quiet + 1e-5:
            raise CalibrationError(
                f"{label} overlaps the FRONT allowance; use a clearer pose while facing the camera",
                retry_labels=(label,),
            )
        # Exit stays beyond normal FRONT variation. Entry is halfway to the pose.
        limits[direction] = {"exit": max(.01, quiet), "enter": (quiet+active)/2}
    return limits


def refit_pitch(samples, config, *, source):
    """Refit selected pitch only; preserve the working yaw settings and signs."""
    result = deepcopy(config)
    key, sign, scale = (config["features"]["pitch"], config["signs"]["pitch"], config["scales"]["pitch"])
    if sign not in (-1, 1):
        raise ValueError("Establish the pitch sign before refitting")
    baseline = float(np.median([r[key] for r in samples["FRONT"]]))
    result["pitch_limits"] = fit_pitch_limits(samples, config, key, sign, scale, baseline)
    result["baseline"][key] = baseline
    result["baseline_version"] += 1
    result["calibration"]["pitch_refit"] = {"source": source, "time": datetime.now(timezone.utc).isoformat(),
        "counts": {label: len(samples[label]) for label in ("FRONT", "UP", "DOWN")},
        "baseline": baseline, "limits": deepcopy(result["pitch_limits"])}
    validate_config(result)
    return result


def fit_calibration(samples, config, *, backend, source="guided", resolution=None, pnp_only=False):
    """Fit robust features/signs/thresholds; reject overlap instead of guessing."""
    if any(len(samples.get(label, [])) < 8 for label in LABELS):
        raise ValueError("Need at least 8 valid samples for each of FRONT/LEFT/RIGHT/UP/DOWN")
    result = deepcopy(config)
    common = set.intersection(*(set(r) for rows in samples.values() for r in rows))
    result["baseline"] = {key: float(np.median([r[key] for r in samples["FRONT"]])) for key in common}
    stats = {}
    if pnp_only and backend != "yunet_geometry":
        raise ValueError("PnP-only calibration requires the yunet_geometry backend")
    candidates_by_axis = (
        ("yaw", "LEFT", "RIGHT", ["pnp_yaw"] if pnp_only else
         ["adas_yaw"] if backend == "openvino_adas" else ["nose_eye_x", "nose_mouth_x", "eye_asymmetry", "mouth_asymmetry", "pnp_yaw"]),
        ("pitch", "UP", "DOWN", ["pnp_pitch"] if pnp_only else
         ["adas_pitch"] if backend == "openvino_adas" else ["vertical_ratio", "eye_nose_ratio", "nose_mouth_ratio", "pnp_pitch"]),
    )
    for axis, positive, negative, candidates in candidates_by_axis:
        fits = []
        for key in candidates:
            if key not in common:
                continue
            data = {label: np.array([r[key] for r in rows], dtype=float) for label,rows in samples.items()}
            if not all(np.isfinite(v).all() for v in data.values()):
                continue
            sign = 1 if np.median(data[positive]) > np.median(data[negative]) else -1
            base = result["baseline"][key]
            pos, neg = sign*(data[positive]-base), -sign*(data[negative]-base)
            quiet_labels = ("FRONT","UP","DOWN") if axis == "yaw" else ("FRONT",)
            quiet = np.concatenate([np.abs(data[l]-base) for l in quiet_labels])
            low = float(np.percentile(quiet,95))
            active_percentile = 50 if pnp_only and axis == "pitch" else 10
            front_padding = 0.0 if pnp_only and axis == "pitch" else None
            positive_active = float(np.percentile(pos, active_percentile))
            negative_active = float(np.percentile(neg, active_percentile))
            high = min(positive_active, negative_active)
            spread = float(np.percentile(np.r_[pos,neg],90)-np.percentile(np.r_[pos,neg],10))
            if high <= max(low*1.25, low+1e-5):
                if pnp_only and axis == "pitch":
                    retry_labels = tuple(label for label, value in ((positive, positive_active), (negative, negative_active))
                                         if value <= max(low*1.25, low+1e-5))
                    raise CalibrationError(
                        "Labelled pitch is not reliably beyond FRONT's measured noise",
                        retry_labels=retry_labels or (positive, negative),
                    )
                continue
            score = (high-low)/max(spread,high*.05,1e-6)
            scale = 1. if key.startswith(("pnp_","adas_")) else 25./float(np.median(np.r_[pos,neg]))
            if axis == "pitch":
                try:
                    fit_pitch_limits(samples, config, key, sign, scale, base,
                                     active_percentile=active_percentile, front_padding=front_padding)
                except CalibrationError:
                    if pnp_only:
                        raise
                    continue
                except ValueError:
                    continue
            fits.append((score,key,sign,scale,(low+high)/2*scale,low*scale,high*scale))
        if not fits:
            raise ValueError(f"{axis} distributions overlap or do not straddle FRONT; repeat calibration or try another backend")
        score,key,sign,scale,enter,low,high = max(fits)
        result["features"][axis], result["signs"][axis], result["scales"][axis] = key,sign,scale
        result[axis+"_enter"] = enter
        result[axis+"_exit"] = max(low+(enter-low)*.35,enter*.6)
        if axis == "pitch":
            result["pitch_limits"] = fit_pitch_limits(
                samples, config, key, sign, scale, result["baseline"][key],
                active_percentile=50 if pnp_only else 10,
                front_padding=0.0 if pnp_only else None,
            )
        stats[axis] = {"feature":key,"separation":score,"quiet_p95":low,
                       "active_percentile":active_percentile,"active_value":high,
                       "medians":{label:float(np.median([r[key] for r in rows])) for label,rows in samples.items()}}
    result["vertical_yaw_guard"] = min(result["yaw_enter"]*.9, max(stats["yaw"]["quiet_p95"]*1.1, result["yaw_exit"]))
    result["calibration"] = {"backend":backend,"source":source,"time":datetime.now(timezone.utc).isoformat(),
                             "camera_resolution":resolution,"statistics":stats}
    result["baseline_version"] += 1
    validate_config(result)
    return result


class HeadPoseTracker:
    def __init__(self, model_path=None, *, config=None, backend=None):
        self.config = deepcopy(config if config is not None else load_config())
        validate_config(self.config)
        self.backend = backend or make_backend(self.config, model_path)
        if self.config["calibration"].get("backend") not in (None,self.backend.name):
            self.config["signs"] = {"yaw":None,"pitch":None}
            self.config["baseline"] = {}
        self.mapping_state = "loaded" if config is not None else "temporary"
        self._lock = threading.RLock()
        self.reset()

    def reset(self, *, clear_baseline=False):
        """Clear visitor/temporal state; optional offset reset never changes signs."""
        with self._lock:
            self.history = deque()
            self.latest = None
            self._candidate, self._since, self._last = "UNCERTAIN", None, None
            if clear_baseline:
                self.config["baseline"] = {}
                self.config["baseline_version"] += 1
                self.mapping_state = "temporary"

    def neutral_calibrate(self, samples):
        with self._lock:
            if len(samples) < 8:
                raise ValueError("Neutral recenter needs at least 8 valid samples")
            keys = set.intersection(*(set(s) for s in samples))
            baseline = {k:float(np.median([s[k] for s in samples])) for k in keys}
            if not all(np.isfinite(v) for v in baseline.values()):
                raise ValueError("Neutral samples must be finite")
            self.config["baseline"] = baseline
            self.config["baseline_version"] += 1
            self.config["neutral_calibration"] = {"time": datetime.now(timezone.utc).isoformat(), "samples": len(samples)}
            self.mapping_state = "temporary"
            self.reset()

    def guided_calibrate(self, samples, resolution=None, *, pnp_only=False):
        with self._lock:
            self.config = fit_calibration(
                samples, self.config, backend=self.backend.name, resolution=resolution, pnp_only=pnp_only
            )
            self.mapping_state = "temporary"
            self.reset()

    def flip_mapping(self, axis, path):
        with self._lock:
            if axis not in ("yaw","pitch"):
                raise ValueError(axis)
            self.config["signs"][axis] = -(self.config["signs"][axis] or 1)
            if axis == "pitch" and "pitch_limits" in self.config:
                limits = self.config["pitch_limits"]
                limits["up"], limits["down"] = limits["down"], limits["up"]
            self.config["calibration"].update(backend=self.backend.name,source="explicit mapping",
                                               time=datetime.now(timezone.utc).isoformat())
            self.reset()
            self.save(path)

    def save(self, path):
        with self._lock:
            save_config(self.config,path)
            self.mapping_state = "saved"

    def _classify(self, yaw, pitch, roll, confidence=1.):
        c = self.config
        if confidence < c["min_confidence"] or abs(roll) > c["roll_limit"]:
            return "UNCERTAIN"
        if abs(yaw) >= c["yaw_enter"]:
            return "LEFT" if yaw > 0 else "RIGHT"
        old = self.latest.direction if self.latest else "UNCERTAIN"
        if old in ("LEFT","RIGHT") and abs(yaw) >= c["yaw_exit"] and (yaw > 0) == (old == "LEFT"):
            return old
        if abs(yaw) > c["vertical_yaw_guard"]:
            return "TRANSITION"
        limits = c.get("pitch_limits", {d: {"enter": c["pitch_enter"], "exit": c["pitch_exit"]} for d in ("up","down")})
        if pitch >= limits["up"]["enter"]:
            return "UP"
        if pitch <= -limits["down"]["enter"]:
            return "DOWN"
        if old in ("UP","DOWN") and abs(pitch) >= limits[old.lower()]["exit"] and (pitch > 0) == (old == "UP"):
            return old
        if abs(yaw) <= c["yaw_exit"] and -limits["down"]["exit"] <= pitch <= limits["up"]["exit"]:
            return "FRONT"
        return "TRANSITION"

    def estimate(self, frame_bgr, *, face_row=None, timestamp_s=None):
        with self._lock:
            now = time.monotonic() if timestamp_s is None else float(timestamp_s)
            raw = self.backend.estimate(frame_bgr,face_row=face_row,timestamp_s=now)
            return self.update(raw,now)

    def update(self, raw, timestamp_s):
        """Process raw evidence; also supports deterministic replay/tests."""
        with self._lock:
            now = float(timestamp_s)
            if not np.isfinite(now):
                raise ValueError("Timestamp must be finite")
            if self._last is not None and (now <= self._last or now-self._last > self.config["max_gap_s"]):
                self.reset()
            self._last = now
            if raw is None:
                self.reset()
                return None
            c = self.config
            values, valid = [], True
            for axis in ("yaw","pitch"):
                key = c["features"][axis]
                valid &= c["signs"][axis] is not None and key in raw.features and key in c["baseline"]
                values.append((raw.features.get(key,0)-c["baseline"].get(key,0))*(c["signs"][axis] or 1)*c["scales"][axis])
            values.append(raw.roll-c["baseline"].get("roll",0))
            valid &= bool(np.isfinite(values).all() and np.isfinite(raw.confidence))
            if not valid:
                self.history.clear()
                values = [0.,0.,0.]
            self.history.append((now,values))
            while self.history and now-self.history[0][0] > c["smoothing_s"]:
                self.history.popleft()
            filtered = np.median([v for _,v in self.history],axis=0)
            jitter = float(np.max(np.median(np.abs(np.array([v for _,v in self.history])-filtered),axis=0)[:2]))
            confidence = float(np.clip(raw.confidence/(1+jitter/max(c["yaw_enter"],c["pitch_enter"])),0,1)) if valid else 0.
            candidate = self._classify(*filtered,confidence)
            if candidate != self._candidate:
                self._candidate, self._since = candidate, now
            if self._since is None:
                self._since = now
            stable = max(0.,now-self._since) if candidate in LABELS else 0.
            direction = candidate if candidate not in LABELS or stable+1e-9 >= c["stable_hold_s"] else "TRANSITION"
            self.latest = PoseEstimate(direction,*map(float,filtered),confidence,now,candidate,stable*1000,raw)
            return self.latest

    def target_status(self, target, *, timestamp_s=None):
        with self._lock:
            if target not in LABELS:
                raise ValueError(f"Unknown target {target}")
            p = self.latest
            now = time.monotonic() if timestamp_s is None else timestamp_s
            required = self.config["stable_hold_s"]*1000
            if p is None or now-p.timestamp_s > self.config["max_gap_s"]:
                return TargetPoseStatus(target,False,0,0,required,0,"no fresh face")
            matches = p.candidate == target and p.confidence >= self.config["min_confidence"]
            stable = p.stable_ms if matches else 0.
            satisfied = matches and p.direction == target
            reason = "satisfied" if satisfied else "holding" if matches else "pose mismatch or uncertain"
            return TargetPoseStatus(target,satisfied,min(1.,stable/required),stable,required,p.confidence,reason)
