"""PC/Pi pose laboratory. Default: pose only, offline, original-frame inference."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
import cv2 as cv

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_SRC = REPO_ROOT / "app" / "src"
for path in (REPO_ROOT, APP_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pose_detection.head_pose import HeadPoseTracker, LABELS, POSE_CONFIG_DIR, ROOT, load_config
from pose_lab import PoseSession


def normalize_results(raw_results) -> list[dict]:
    if raw_results is None:
        return []
    if isinstance(raw_results,dict):
        return [] if raw_results.get("status")=="no_face" else [raw_results]
    return list(raw_results)


def display_frame(frame, mirror):
    """Mirror a display copy only. Inference and semantic labels never change."""
    return cv.flip(frame,1) if mirror else frame.copy()


def draw_overlay(frame, pose, tracker, session, now, message, target=None, results=()):
    mirror = tracker.config["preview_mirror"]
    shown = display_frame(frame,mirror)
    if pose:
        x,y,w,h = pose.bbox
        if mirror:
            x = shown.shape[1]-x-w
        cv.rectangle(shown,(x,y),(x+w,y+h),(0,220,220),2)
    text = [f"POSE: {pose.direction if pose else 'NO_FACE'} | {tracker.backend.name}",
            f"Mirror: {'ON' if mirror else 'OFF'} | mapping: {tracker.mapping_state} {tracker.config['signs']}",
            session.status(now),message,
            "C=neutral G=guided X/Y=flip+save S=save ESC=cancel Z=undo Q=quit"]
    if pose:
        text.append(f"yaw={pose.yaw:+.1f} pitch={pose.pitch:+.1f} roll={pose.roll:+.1f} confidence={pose.confidence:.2f}")
    if target:
        status = tracker.target_status(target,timestamp_s=now)
        text.append(f"Target {target}: {status.progress:.0%} {status.reason}")
    for result in results:
        text.append(f"Recognition: {result.get('name','Unknown')} {result.get('status')} {result.get('engine')}")
    for i,line in enumerate(text):
        cv.putText(shown,line,(10,24+i*24),cv.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv.LINE_AA)
        cv.putText(shown,line,(10,24+i*24),cv.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv.LINE_AA)
    return shown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pose-only",action="store_true")
    mode.add_argument("--with-recognition",action="store_true")
    parser.add_argument("--backend",choices=("auto","yunet_geometry","openvino_adas"))
    parser.add_argument("--config",type=Path,default=POSE_CONFIG_DIR/"head_pose.local.json")
    parser.add_argument("--purpose",choices=("calibration","validation","test"),default="validation")
    parser.add_argument("--camera",default="0",help="USB index or video path; use --picamera2 for CSI")
    parser.add_argument("--picamera2",action="store_true")
    parser.add_argument("--width",type=int,default=640)
    parser.add_argument("--height",type=int,default=480)
    parser.add_argument("--pose-hz",type=float)
    parser.add_argument("--mirror",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--target",choices=LABELS)
    parser.add_argument("--recognition-setup",choices=("1","2"),default="1")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--seconds",type=float,default=0,help="Optional bounded benchmark duration")
    parser.add_argument("--log-dir",type=Path,default=ROOT/"logs/head_pose")
    args = parser.parse_args(argv)
    if args.width<32 or args.height<32 or args.seconds<0:
        parser.error("Invalid camera dimensions/duration")
    config = load_config(args.config if args.config.exists() else POSE_CONFIG_DIR/"head_pose.json")
    for key,value in (("backend",args.backend),("pose_hz",args.pose_hz),("preview_mirror",args.mirror)):
        if value is not None:
            config[key] = value
    tracker = HeadPoseTracker(config=config)
    tracker.mapping_state = "loaded" if args.config.exists() else "temporary"
    if args.purpose == "test":
        if args.headless or args.with_recognition:
            parser.error("--purpose test needs a preview and runs without recognition")
        from pose_test import run_test
        return run_test(args, tracker)
    cascade = None
    if args.with_recognition:
        if args.recognition_setup=="1":
            from hybrid import HybridCascade
        else:
            from hybrid_rpi import HybridCascade
        cascade = HybridCascade(str(ROOT))
        print("Recognition comparison uses a second detector pass; latency reported separately. Pose-only is deployment benchmark.")
    session = PoseSession(args.log_dir,purpose=args.purpose,metadata={"platform":platform.platform(),
        "python":platform.python_version(),"opencv":cv.__version__,"requested_backend":config["backend"],
        "actual_backend":tracker.backend.name,"recognition":args.with_recognition,"config":tracker.config})
    cap = camera = None
    guided = None
    neutral = False
    message = "G: guided signs + thresholds. C: neutral offsets only. Directions are YOUR left/right."
    pose, results = None,[]
    start = time.monotonic()
    cpu_start = time.process_time()
    last_update = -float("inf")
    frame_count = 0
    had_face = False
    first_loop = None
    key_labels = dict(zip("flrud",LABELS))
    print(message)
    print(f"Backend: {tracker.backend.name}; mapping {tracker.mapping_state}; log {session.path}")
    try:
        if args.picamera2:
            from picamera2 import Picamera2
            camera = Picamera2()
            # Picamera2 RGB888 delivers BGR byte order for OpenCV.
            camera.configure(camera.create_video_configuration(main={"size":(args.width,args.height),"format":"RGB888"}))
            camera.start()
        else:
            cap = cv.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
            cap.set(cv.CAP_PROP_FRAME_WIDTH,args.width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT,args.height)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {args.camera}")
        while not args.seconds or time.monotonic()-start < args.seconds:
            loop_start = time.monotonic()
            if first_loop is None:
                first_loop = loop_start
            if camera:
                frame = camera.capture_array("main")
            else:
                ok,frame = cap.read()
                if not ok:
                    break
            captured = time.monotonic()
            latency = {"capture":(captured-loop_start)*1000}
            if captured-last_update >= 1/tracker.config["pose_hz"]:
                if session.active and captured>=session.active["start"] and not session.active.get("recording"):
                    tracker.reset()
                    session.active["recording"] = True
                pose = tracker.estimate(frame,timestamp_s=captured)
                if (pose is not None) != had_face:
                    session.emit("face_found" if pose else "face_loss",timestamp_s=captured,frame=frame_count)
                had_face = pose is not None
                last_update = captured
                latency["pose"] = (time.monotonic()-captured)*1000
                if pose:
                    latency.update(pose.raw.latency_ms)
                if cascade:
                    rec_start = time.monotonic()
                    results = normalize_results(cascade.infer(frame))
                    latency["recognition"] = (time.monotonic()-rec_start)*1000
                complete = session.consume(pose,captured,frame_count,tracker.config,args.target)
                if complete and (guided is not None or neutral):
                    samples = session.burst_features(complete["burst_id"])
                    try:
                        if neutral:
                            tracker.neutral_calibrate(samples)
                            neutral = False
                            message = "Neutral offsets updated; signs unchanged. S saves config."
                        else:
                            guided[complete["label"]] = samples
                            if all(label in guided for label in LABELS):
                                tracker.guided_calibrate(guided,resolution=[frame.shape[1],frame.shape[0]])
                                message = "Guided fit ready; S saves. Collect validation in a NEW session."
                                guided = None
                            else:
                                next_label = next(label for label in LABELS if label not in guided)
                                message = f"Guided: hold YOUR {next_label}; press {next_label[0]} twice (UP=U, DOWN=D)."
                        print(message)
                    except ValueError as exc:
                        message = str(exc)
                        print(message)
                        neutral = False
                if args.target and captured-start > tracker.config["manual_timeout_s"]:
                    if not tracker.target_status(args.target,timestamp_s=captured).satisfied:
                        message = "Automatic target timeout: operator manual capture available (M)."
            if args.headless:
                key = -1
            else:
                cv.imshow("LS-Face Pose Lab",draw_overlay(frame,pose,tracker,session,captured,message,args.target,results))
                key = cv.waitKey(1)&255
            char = chr(key).lower() if 0<=key<128 else ""
            if char=="q":
                break
            if key==27:
                session.cancel()
                message = "Cancelled active step; completed bursts retained."
            elif char=="z":
                bid = session.undo()
                if bid:
                    if guided is not None:
                        guided.clear()
                    if args.purpose=="calibration":
                        tracker.config["signs"] = {"yaw":None,"pitch":None}
                        tracker.reset(clear_baseline=True)
                        if args.config.exists():
                            tracker.save(args.config)
                    message = f"Undone burst {bid}; restart guided collection if active."
            elif char in key_labels:
                label = key_labels[char]
                expected = "FRONT" if neutral else next((l for l in LABELS if l not in guided),None) if guided is not None else None
                if expected and label!=expected:
                    message = f"Current calibration step requires {expected}."
                else:
                    if session.start(label,captured):
                        tracker.reset()  # Measure fresh hold rather than pre-burst history.
            elif char in ("g","c","x","y"):
                if args.purpose!="calibration":
                    message = "Config locked for validation. Restart with --purpose calibration."
                else:
                    session.cancel()
                    if char=="g":
                        guided,neutral = {},False
                        message = "Guided: hold FRONT, press F twice; then YOUR LEFT, RIGHT, UP, DOWN."
                    elif char=="c":
                        guided,neutral = None,True
                        message = "Neutral recenter ONLY: hold FRONT, press F twice; signs unchanged."
                    else:
                        guided,neutral = None,False
                        tracker.flip_mapping("yaw" if char=="x" else "pitch",args.config)
                        message = f"Explicit mapping saved: {tracker.config['signs']}"
            elif char=="s":
                if args.purpose=="calibration":
                    tracker.save(args.config)
                summary = session.write_summary()
                print(json.dumps({"frames":summary["retained_frames"],"balanced_accuracy":summary["stable_window"]["balanced_accuracy"]}))
                message = f"Summary saved; config {tracker.mapping_state}."
            elif char=="m" and args.target and captured-start>=tracker.config["manual_timeout_s"]:
                session.emit("manual_operator",timestamp_s=captured,target=args.target,automatic=False)
                message = "Manual operator event recorded; no enrollment capture performed."
            latency["loop"] = (time.monotonic()-loop_start)*1000
            session.emit("loop",frame=frame_count,timestamp_s=captured,latency_ms=latency)
            frame_count += 1
    finally:
        loop_elapsed = time.monotonic()-first_loop if first_loop is not None else 0.
        if cap is not None:
            cap.release()
        if camera is not None:
            camera.stop()
            camera.close()
        if not args.headless:
            cv.destroyAllWindows()
        elapsed = time.monotonic()-start
        ram = None
        try:
            import resource
            ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            pass
        session.emit("runtime",elapsed_s=elapsed,loop_elapsed_s=loop_elapsed,cpu_percent_one_core=100*(time.process_time()-cpu_start)/max(elapsed,1e-9),
                     peak_rss_native=ram,peak_rss_units="KiB Linux / bytes macOS; null when unavailable",frames=frame_count)
        session.close()
        print(f"Events: {session.path}\nSummary: {session.path.with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
