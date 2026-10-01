from .camera import Camera 
from .factory import camera_factory 
import threading

class CameraBusyError(RuntimeError):
    """Another screen is opening or closing the shared camera."""

class CameraManager:
    """ Manages application camera during its lifecycle. """

    def __init__(self): 
        self._camera: Camera | None = None 
        self._mode: str | None = None 
        self._lock = threading.RLock()

    def acquire(self, mode: str, *, cancel_event=None) -> Camera:
        # Other screens must not block their UI behind an asynchronous startup.
        if not self._lock.acquire(blocking=False):
            raise CameraBusyError("Camera is still starting. Please try again shortly.")
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Camera startup cancelled.")
            return self._acquire_locked(mode, cancel_event)
        finally:
            self._lock.release()

    def _acquire_locked(self, mode, cancel_event):
        if self._camera is not None and self._mode == mode:
            return self._camera 

        self.close()
        camera = camera_factory(mode)

        try:
            camera.start()
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Camera startup cancelled.")
        except Exception:
            try:
                camera.stop()
            except Exception:
                pass
            raise 

        self._camera = camera 
        self._mode = mode 
        return camera

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self):
        camera = self._camera
        self._camera = None 
        self._mode = None 

        if camera is not None:
            camera.stop()
