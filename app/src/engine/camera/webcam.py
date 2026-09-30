import logging
import sys
import threading

import cv2 as cv
import numpy as np

from .camera import Camera

_log = logging.getLogger(__name__)


class WebCamera(Camera):
    """OpenCV webcam with a background reader thread.

    The reader runs ``cap.read()`` in a dedicated daemon thread so the Kivy
    UI thread is never blocked waiting on driver frame delivery.  ``read()``
    returns the latest captured frame instantly (non-blocking), or ``None``
    during the brief warmup window before the first frame arrives.
    """

    def __init__(self, camera_id: int = 0, res: tuple[int, int] | None = None, fps: int = 30):
        self.camera_id = camera_id
        self.res = res or (640, 480)
        self.fps = fps
        self.cap = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._running = False
        self._frame_count = 0
        self._drop_count = 0

    def start(self) -> "WebCamera":
        _log.info("[WebCamera] Opening camera_id=%s res=%s fps=%s", self.camera_id, self.res, self.fps)
        if sys.platform == "win32":
            self.cap = cv.VideoCapture(self.camera_id, cv.CAP_DSHOW)
            if not self.cap.isOpened():
                _log.warning("[WebCamera] CAP_DSHOW failed, retrying without backend hint")
                self.cap = cv.VideoCapture(self.camera_id)
        else:
            self.cap = cv.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam: {self.camera_id}.")

        width, height = self.res
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv.CAP_PROP_FPS)
        _log.info("[WebCamera] Opened — actual resolution=%sx%s fps=%.1f", actual_w, actual_h, actual_fps)

        self._frame_count = 0
        self._drop_count = 0
        self._running = True
        self._reader = threading.Thread(
            target=self._read_loop,
            name="webcam-reader",
            daemon=True,
        )
        self._reader.start()
        _log.debug("[WebCamera] Reader thread started")
        return self

    def _read_loop(self) -> None:
        """Continuously drain frames from the driver into the shared slot."""
        while self._running:
            ret, frame = self.cap.read()
            if not self._running:
                break
            if ret and frame is not None:
                with self._lock:
                    self._latest = frame
                    self._frame_count += 1
                if self._frame_count == 1:
                    _log.info("[WebCamera] First frame received — camera warm")
                elif self._frame_count % 300 == 0:
                    _log.debug(
                        "[WebCamera] %d frames captured, %d drops",
                        self._frame_count, self._drop_count,
                    )
            else:
                with self._lock:
                    self._drop_count += 1
                _log.warning("[WebCamera] cap.read() failed (drop #%d)", self._drop_count)

    def read(self) -> np.ndarray | None:
        """Return the latest frame, or ``None`` if the reader hasn't produced one yet.

        Non-blocking.  Callers must treat ``None`` as a warmup skip, not an error.
        Caller must copy the returned array before mutating it.
        """
        with self._lock:
            return self._latest

    def stop(self) -> None:
        _log.info(
            "[WebCamera] Stopping — %d frames captured, %d drops",
            self._frame_count, self._drop_count,
        )
        self._running = False
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self._lock:
            self._latest = None
        _log.debug("[WebCamera] Stopped")
