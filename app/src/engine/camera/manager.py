from .camera import Camera 
from .factory import camera_factory 

class CameraManager:
    """ Manages application camera during its lifecycle. """

    def __init__(self): 
        self._camera: Camera | None = None 
        self._mode: str | None = None 

    def acquire(self, mode: str) -> Camera:
        if self._camera is not None and self._mode == mode:
            return self._camera 

        self.close()
        camera = camera_factory(mode)

        try:
            camera.start()
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
        camera = self._camera
        self._camera = None 
        self._mode = None 

        if camera is not None:
            camera.stop()