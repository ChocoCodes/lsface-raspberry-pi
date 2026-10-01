import logging
import threading
import time

import numpy as np

from .camera import Camera

_log = logging.getLogger(__name__)


class PiCamera(Camera):
    """Picamera2 wrapper with a background reader thread.

    ``capture_array()`` blocks until the ISP delivers a frame; running it on
    a daemon thread keeps the Kivy UI thread free.  ``read()`` returns the
    latest BGR frame instantly (non-blocking), or ``None`` during warmup.
    """

    def __init__(self, res: tuple[int, int] = (1280, 720), warmup_s: float = 1.5) -> None:
        """
        Configures the camera sensor.
        Args:
            res: Tuple of (width, height). Defaults to (1280, 720).
            warmup_s: Time to wait for sensor auto-exposure and white balance to stabilize.
        """
        # Lazy import — picamera2 is only available on RPi; desktop imports still work.
        from picamera2 import Picamera2

        self.res = res
        self.warmup_s = warmup_s
        self.picam2 = Picamera2()

        conf = self.picam2.create_video_configuration(
            main={"size": self.res, "format": "RGB888"}
        )
        self.picam2.configure(conf)

        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._running = False
        self._frame_count = 0
        self._error_count = 0

    def start(self) -> "PiCamera":
        """Starts hardware stream and background reader thread."""
        _log.info("[PiCamera] Starting stream — res=%s warmup=%.1fs", self.res, self.warmup_s)
        self.picam2.start()
        time.sleep(self.warmup_s)
        _log.info("[PiCamera] Warmup done, starting reader thread")

        self._frame_count = 0
        self._error_count = 0
        self._running = True
        self._reader = threading.Thread(
            target=self._read_loop,
            name="picam-reader",
            daemon=True,
        )
        self._reader.start()
        return self

    def _read_loop(self) -> None:
        """Continuously drain frames from the ISP into the shared slot."""
        while self._running:
            try:
                # capture_array() blocks until next ISP frame — fine on a thread.
                rgb = self.picam2.capture_array()
                # RGB888 → BGR, ensure C-contiguous layout for OpenCV/blit_buffer.
                bgr = rgb[:, :, ::-1].copy()
                with self._lock:
                    self._latest = bgr
                    self._frame_count += 1
                if self._frame_count == 1:
                    _log.info("[PiCamera] First frame received — camera warm")
                elif self._frame_count % 300 == 0:
                    _log.debug(
                        "[PiCamera] %d frames captured, %d errors",
                        self._frame_count, self._error_count,
                    )
            except Exception as exc:
                self._error_count += 1
                if self._running:
                    _log.warning("[PiCamera] capture_array() error #%d: %s", self._error_count, exc)

    def read(self) -> np.ndarray | None:
        """Return the latest BGR frame, or ``None`` if not yet available.

        Non-blocking.  Callers must treat ``None`` as a warmup skip, not an error.
        Caller must copy the returned array before mutating it.
        """
        with self._lock:
            return self._latest

    def stop(self) -> None:
        """Stops the background reader, then stops and closes the hardware stream."""
        _log.info(
            "[PiCamera] Stopping — %d frames captured, %d errors",
            self._frame_count, self._error_count,
        )
        self._running = False
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        with self._lock:
            self._latest = None
        try:
            self.picam2.stop()
        finally:
            self.picam2.close()
        _log.debug("[PiCamera] Stopped")

    def __enter__(self) -> "PiCamera":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
