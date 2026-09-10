"""Disposable participant-flow rehearsal using the existing OpenCV UI."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import cv2 as cv
import numpy as np

APP_SRC = Path(__file__).resolve().parents[1] / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from pose_detection.flow import GuidedPoseFlow, INSTRUCTIONS
from pose_detection.head_pose import LABELS

WINDOW = "LS-Face - Guided capture test"
class GuidedPoseTest(GuidedPoseFlow):
    """The test UI adds disposable photos on top of the production scan flow."""

    def __init__(self, tracker):
        self.folder = None
        self.shots = []
        super().__init__(tracker, on_confirm=self._save_photo, allow_manual=True, completion_phase="results")

    def restart(self):
        self.close()
        self.folder = TemporaryDirectory(prefix="lsface-pose-test-")
        super().restart()
        self.note = "A short straight-ahead check, then five photos."

    def close(self):
        self.shots.clear()
        if self.folder is not None:
            self.folder.cleanup()
            self.folder = None
        if hasattr(self, "saved_config"):
            super().close()

    def _save_photo(self, label, frame, manual):
        path = Path(self.folder.name) / f"{self.stage+1:02d}-{label.lower()}.png"
        image = frame.copy()  # Keep original, unmirrored pixels, without overlays.
        if not cv.imwrite(str(path), image):
            raise OSError(f"Could not save temporary photo: {path}")
        self.shots.append({"label": label, "image": image, "path": path, "manual": manual})

    def manual_capture(self, frame, pose, now):
        self.manual_confirm(frame, pose, now)


def _text(canvas, value, xy, size=.65, color=(230,235,240)):
    cv.putText(canvas,value,xy,cv.FONT_HERSHEY_SIMPLEX,size,color,1,cv.LINE_AA)


def _image(canvas, source, box, mirror=False):
    x,y,w,h = box
    if mirror:
        source = cv.flip(source,1)
    scale = min(w/source.shape[1],h/source.shape[0])
    resized = cv.resize(source,(max(1,round(source.shape[1]*scale)),max(1,round(source.shape[0]*scale))))
    top,left = y+(h-resized.shape[0])//2,x+(w-resized.shape[1])//2
    canvas[top:top+resized.shape[0],left:left+resized.shape[1]] = resized


def draw_screen(flow, frame, now):
    canvas = np.full((760,1000,3),(28,24,22),dtype=np.uint8)
    mirror = flow.tracker.config["preview_mirror"]
    title = "Your five photos" if flow.phase == "results" else "Let's capture your head positions"
    _text(canvas,title,(32,44),.95)
    _text(canvas,"Preview mirrored" if mirror else "Preview not mirrored",(750,42),.48)
    buttons = []
    if flow.phase == "results":
        for index,shot in enumerate(flow.shots):
            x,y = 32+(index%3)*320,105+(index//3)*240
            _image(canvas,shot["image"],(x,y,296,190),mirror)
            _text(canvas,shot["label"]+(" - manual" if shot["manual"] else ""),(x,y+220),.65)
        _text(canvas,"These test photos are deleted when you finish or restart.",(32,644),.65)
    else:
        prompt = ("Look straight at the camera" if flow.phase in ("ready","calibrating") else
                  "Operator setup needed" if flow.phase=="setup" else
                  "Photo captured!" if flow.phase=="feedback" else INSTRUCTIONS[LABELS[flow.stage]])
        _text(canvas,prompt,(32,91),.8,(100,225,190))
        _image(canvas,frame,(32,112,640,440),mirror)
        for index,label in enumerate(LABELS):
            y = 150+index*75
            done = index < len(flow.shots)
            color = (100,225,190) if done else (230,235,240)
            _text(canvas, f"{index+1}. {label}" + ("  saved" if done else ""),(710,y),.65,color)
        cv.rectangle(canvas,(32,578),(968,594),(65,60,56),-1)
        cv.rectangle(canvas,(32,578),(32+int(936*flow.progress),594),(100,225,190),-1)
        _text(canvas,flow.note,(32,628),.53)
        _text(canvas,"Follow YOUR left and right. Move your head, not only your eyes.",(32,651),.5)
    if flow.phase=="ready":
        buttons.append(("Start [SPACE]","start",(32,686,260,48)))
    if flow.phase not in ("ready","setup"):
        buttons.append(("Restart [R]","restart",(32,686,260,48)))
    if flow.manual_available(now):
        buttons.append(("Manual photo [M]","manual",(322,686,290,48)))
    buttons.append(("Finish [Q]" if flow.phase=="results" else "Cancel [Q]","quit",(710,686,258,48)))
    for title,action,(x,y,w,h) in buttons:
        cv.rectangle(canvas,(x,y),(x+w,y+h),(65,60,56),-1)
        _text(canvas,title,(x+18,y+31),.65)
    return canvas,buttons


def run_test(args, tracker):
    flow = GuidedPoseTest(tracker)
    cap = camera = None
    pending = []
    buttons = []
    pose = None
    frame = np.zeros((args.height,args.width,3),dtype=np.uint8)
    start, last_update = time.monotonic(), -float("inf")

    def click(event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONUP:
            for _,action,(left,top,w,h) in buttons:
                if left<=x<=left+w and top<=y<=top+h:
                    pending.append(action)
                    break

    try:
        cv.namedWindow(WINDOW,cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(WINDOW,click)
        if args.picamera2:
            from picamera2 import Picamera2
            camera = Picamera2()
            camera.configure(camera.create_video_configuration(main={"size":(args.width,args.height),"format":"RGB888"}))
            camera.start()
        else:
            cap = cv.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
            cap.set(cv.CAP_PROP_FRAME_WIDTH,args.width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT,args.height)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {args.camera}")
        while not args.seconds or time.monotonic()-start < args.seconds:
            now = time.monotonic()
            if flow.phase != "results":
                if camera:
                    frame = camera.capture_array("main")
                else:
                    ok,frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera stopped delivering frames")
                now = time.monotonic()
                if now-last_update >= 1/tracker.config["pose_hz"]:
                    pose = tracker.estimate(frame,timestamp_s=now)
                    last_update = now
                    flow.update(frame,pose,now)
            shown,buttons = draw_screen(flow,frame,now)
            cv.imshow(WINDOW,shown)
            key = cv.waitKey(15)&255
            if cv.getWindowProperty(WINDOW,cv.WND_PROP_VISIBLE)<1:
                break
            char = chr(key).lower() if key<128 else ""
            action = {" ":"start","r":"restart","m":"manual","q":"quit","\x1b":"quit"}.get(char)
            if pending:
                action = pending.pop(0)
            if action=="quit":
                break
            if action=="start" and flow.phase=="ready":
                flow.start(now)
                tracker.reset()
            elif action=="restart":
                flow.restart()
                pose, last_update = None, -float("inf")
            elif action=="manual":
                flow.manual_capture(frame,pose,now)
    finally:
        flow.close()
        if cap is not None:
            cap.release()
        if camera is not None:
            camera.stop()
            camera.close()
        cv.destroyAllWindows()
        print("Guided test closed. Temporary test photos have been deleted.")
    return 0
