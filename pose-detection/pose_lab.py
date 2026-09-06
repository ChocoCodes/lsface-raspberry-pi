"""Append-only pose experiment events, exact burst undo, and reproducible metrics."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np

APP_SRC = Path(__file__).resolve().parents[1] / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from pose_detection.head_pose import LABELS


def distribution(values):
    if not values:
        return None
    return dict(zip(("p10","median","p90"),map(float,np.percentile(values,[10,50,90]))))


def percentiles(values):
    return dict(zip(("p50","p95","p99"),map(float,np.percentile(values,[50,95,99])))) if values else None


def retained_rows(events):
    completed = {e["burst_id"] for e in events if e["event"] == "complete"}
    excluded = {e["burst_id"] for e in events if e["event"] in ("undo","cancel")}
    return [dict(e,undone=False) for e in events if e["event"] == "sample" and e["burst_id"] in completed-excluded]


def confusion(pairs):
    predictions = (*LABELS,"TRANSITION","UNCERTAIN","NO_FACE")
    matrix = {label:{p:0 for p in predictions} for label in LABELS}
    for truth,prediction in pairs:
        matrix[truth][prediction if prediction in predictions else "UNCERTAIN"] += 1
    scores = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(row[label] for row in matrix.values())
        scores[label] = {"precision":tp/predicted if predicted else None,
                         "recall":tp/support if support else None,"support":support}
    recalls = [v["recall"] for v in scores.values() if v["recall"] is not None]
    return {"matrix":matrix,"per_class":scores,
            "balanced_accuracy":sum(recalls)/len(recalls) if recalls else None,
            "all_classes_present":len(recalls)==5}


def summarize(events):
    rows = retained_rows(events)
    bursts = defaultdict(list)
    for r in rows:
        bursts[r["burst_id"]].append(r)
    pairs = [(r["label"],r["prediction"]) for r in rows]
    windows, burst_stats = [], {}
    stable_times = defaultdict(list)
    for bid, samples in bursts.items():
        label, start = samples[0]["label"],samples[0]["timestamp_s"]
        bins = defaultdict(list)
        warmup = samples[0]["config"]["stable_hold_s"]
        for r in samples:
            offset = r["timestamp_s"]-start-warmup
            if offset>=0:
                bins[int(offset/.5)].append(r)
        # Fixed warmup exclusion is independent of predictions. Sparse windows fail.
        last = samples[-1]["timestamp_s"]-start-warmup
        for index in range(max(0,int(last/.5))):
            group = bins[index]
            predictions = {r["prediction"] for r in group}
            coverage = group[-1]["timestamp_s"]-group[0]["timestamp_s"] if group else 0
            pred = next(iter(predictions)) if len(predictions)==1 and coverage>=.25 else "UNCERTAIN"
            windows.append((label,pred))
        correct = [r["timestamp_s"]-start for r in samples if r["prediction"]==label]
        latency = min(correct) if correct else None
        if latency is not None:
            stable_times[label].append(latency*1000)
        burst_stats[bid] = {"label":label,"frames":len(samples),"time_to_stable_ms":None if latency is None else latency*1000,
                           "flicker":sum(a["prediction"]!=b["prediction"] for a,b in zip(samples,samples[1:]))}
    front = [r for r in rows if r["label"]=="FRONT"]
    false_front = {label:sum(r["prediction"]==label for r in front)/len(front) if front else None for label in LABELS[1:]}
    quiet_yaw = [r for r in rows if r["label"] in ("FRONT","UP","DOWN")]
    separations = {}
    for positive,negative in (("UP","DOWN"),("LEFT","RIGHT")):
        a = [r["raw"]["features"] for r in rows if r["label"]==positive and r["raw"]]
        b = [r["raw"]["features"] for r in rows if r["label"]==negative and r["raw"]]
        if not a or not b:
            continue
        for key in set.intersection(*(set(v) for v in a+b)):
            da,db = distribution([v[key] for v in a]),distribution([v[key] for v in b])
            gap = abs(da["median"]-db["median"])
            width = max(1e-9,(da["p90"]-da["p10"]+db["p90"]-db["p10"])/2)
            separations[f"{positive}_{negative}:{key}"] = {positive:da,negative:db,"median_gap":gap,
                "separation_ratio":gap/width,"p10_p90_overlap":not(da["p90"]<db["p10"] or db["p90"]<da["p10"])}
    loops = [e for e in events if e["event"]=="loop"]
    timings = defaultdict(list)
    for e in loops:
        for key,value in e["latency_ms"].items():
            timings[key].append(value)
    duration = sum(e["latency_ms"]["loop"] for e in loops)/1000
    runtime = next((e for e in reversed(events) if e["event"]=="runtime"),None)
    if runtime and runtime.get("loop_elapsed_s"):
        duration = runtime["loop_elapsed_s"]
    return {"retained_frames":len(rows),"retained_bursts":len(bursts),"per_frame":confusion(pairs),
            "stable_window":confusion(windows),"stable_window_definition":"exclude fixed configured hold duration; complete 500ms bins; mixed/sparse (<250ms coverage) count as UNCERTAIN; no-face included",
            "front_false_positive_rates":false_front,
            "horizontal_false_trigger_rate":sum(r["prediction"] in ("LEFT","RIGHT") for r in quiet_yaw)/len(quiet_yaw) if quiet_yaw else None,
            "bursts":burst_stats,"time_to_stable_ms":{l:percentiles(stable_times[l]) for l in LABELS},
            "time_to_stable_note":"from first recorded sample after settle; failures retained as null in bursts",
            "separation":separations,"latency_ms":{k:percentiles(v) for k,v in timings.items()},
            "effective_loop_fps":len(loops)/duration if duration else None,
            "runtime":runtime}


class PoseSession:
    def __init__(self, directory, *, purpose="validation", settle_s=1., duration_s=3., metadata=None):
        if purpose not in ("calibration","validation"):
            raise ValueError("Session purpose must be calibration or validation")
        self.session_id = uuid.uuid4().hex
        directory = Path(directory)/purpose
        directory.mkdir(parents=True,exist_ok=True)
        self.path = directory/f"{self.session_id}.jsonl"
        self.handle = self.path.open("x",encoding="utf-8")
        self.purpose, self.settle_s, self.duration_s = purpose,settle_s,duration_s
        self.events, self.completed = [],[]
        self.active = self.armed = None
        self.emit("session",purpose=purpose,metadata=metadata or {})

    def emit(self, event, **values):
        encoded = json.dumps({"session_id":self.session_id,"event":event,**values},allow_nan=False)
        row = json.loads(encoded)  # Freeze config snapshots against later calibration.
        self.handle.write(encoded+"\n")
        self.handle.flush()
        self.events.append(row)
        return row

    def start(self, label, now):
        if label not in LABELS:
            raise ValueError(label)
        if self.active:
            return False
        if self.armed != label:
            self.armed = label
            return False
        self.active = {"label":label,"burst_id":uuid.uuid4().hex,"start":now+self.settle_s}
        self.armed = None
        self.emit("start",**self.active)
        return True

    def cancel(self):
        if self.active:
            self.emit("cancel",burst_id=self.active["burst_id"])
        self.active = self.armed = None

    def undo(self):
        if self.active or self.armed:
            return None
        if not self.completed:
            return None
        bid = self.completed.pop()
        self.emit("undo",burst_id=bid)
        return bid

    def consume(self, pose, now, frame, config, requested_target=None):
        if not self.active or now < self.active["start"]:
            return None
        active = self.active
        if now-active["start"] >= self.duration_s:
            self.completed.append(active["burst_id"])
            self.emit("complete",burst_id=active["burst_id"])
            self.active = None
            return active
        raw = asdict(pose.raw) if pose else None
        self.emit("sample",burst_id=active["burst_id"],timestamp_s=now,frame=frame,label=active["label"],
                  prediction=pose.direction if pose else "NO_FACE",requested_target=requested_target,
                  normalized={"yaw":pose.yaw,"pitch":pose.pitch,"roll":pose.roll} if pose else None,
                  confidence=pose.confidence if pose else 0,raw=raw,face_loss=pose is None,
                  config=config,mirror=config["preview_mirror"],undone=False)
        return None

    def burst_features(self, burst_id):
        return [r["raw"]["features"] for r in retained_rows(self.events) if r["burst_id"]==burst_id and r["raw"]]

    def status(self, now):
        if self.active:
            remain = self.active["start"]-now
            return f"GT {self.active['label']}: settle {remain:.1f}s" if remain>0 else f"GT {self.active['label']}: RECORDING"
        return f"GT {self.armed}: press same key again to confirm" if self.armed else "GT idle: F/L/R/U/D twice"

    def write_summary(self):
        result = summarize(self.events)
        self.path.with_suffix(".summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        return result

    def close(self):
        if not self.handle.closed:
            self.cancel()
            try:
                self.write_summary()
            finally:
                self.handle.close()
