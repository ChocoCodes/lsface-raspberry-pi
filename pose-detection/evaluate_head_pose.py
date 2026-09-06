"""Rebuild undo-aware summaries or fit a config using calibration logs only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

APP_SRC = Path(__file__).resolve().parents[1] / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from pose_detection.head_pose import HeadPoseTracker,RawPose,fit_calibration,load_config,save_config
from pose_lab import retained_rows,summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs",nargs="+",type=Path)
    parser.add_argument("--fit",type=Path,help="Write fitted config; rejects validation logs")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--backend",choices=("yunet_geometry","openvino_adas"),default="yunet_geometry")
    parser.add_argument("--replay-config",type=Path,help="Score retained raw evidence with a frozen config; does not rerun models or measure latency")
    args = parser.parse_args()
    if args.fit and args.replay_config:
        parser.error("Choose fit or replay, not both")
    samples = defaultdict(list)
    for path in args.logs:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.fit:
            if not events or events[0].get("purpose")!="calibration":
                raise ValueError(f"Refusing to fit validation log: {path}")
            for row in retained_rows(events):
                if row["raw"]:
                    if args.backend=="openvino_adas" and row["raw"]["backend"]!="openvino_adas":
                        raise ValueError("ADAS fitting requires actual ADAS samples, not fallback geometry")
                    samples[row["label"]].append(row["raw"]["features"])
        else:
            if args.replay_config:
                config = load_config(args.replay_config)
                # Replay consumes recorded features; never loads or falls back models.
                name = config["calibration"].get("backend") or "yunet_geometry"
                tracker = HeadPoseTracker(config=config,backend=SimpleNamespace(name=name))
                for row in events:
                    if row["event"]=="start":
                        tracker.reset()
                    if row["event"]!="sample":
                        continue
                    raw = RawPose(**row["raw"]) if row["raw"] else None
                    if raw and any(key not in raw.features for key in config["features"].values()):
                        raise ValueError("Log lacks requested backend features; collect actual backend observations")
                    pose = tracker.update(raw,row["timestamp_s"])
                    row["prediction"] = pose.direction if pose else "NO_FACE"
            summary = summarize(events)
            if args.replay_config:
                summary["replay_config"] = str(args.replay_config)
                summary["latency_ms"] = None
                summary["effective_loop_fps"] = None
            suffix = ".replay."+args.replay_config.stem+".summary.json" if args.replay_config else ".summary.json"
            output = path.with_suffix(suffix)
            output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            print(output)
    if args.fit:
        config = load_config(args.config) if args.config else load_config()
        config["backend"] = args.backend
        fitted = fit_calibration(samples,config,backend=args.backend,source="calibration logs: "+", ".join(p.name for p in args.logs))
        save_config(fitted,args.fit)
        print(args.fit)


if __name__ == "__main__":
    main()
