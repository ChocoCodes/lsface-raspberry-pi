import numpy as np
from abc import ABC, abstractmethod

class Camera(ABC):
    @abstractmethod
    def start(self) -> "Camera":
        pass

    @abstractmethod
    def read(self) -> np.ndarray:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass