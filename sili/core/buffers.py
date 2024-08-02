import abc
import numpy as np

from sili.core.devices.gpu import GPUManager


class Buffer(abc.ABC):
    def __init__(self, gpu: GPUManager):
        self.gpu = gpu
        self.buffer = None

    @abc.abstractmethod
    def set(self, data: np.ndarray):
        pass

    @abc.abstractmethod
    def get(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def __setstate__(self, state):
        pass

    @abc.abstractmethod
    def __getstate__(self):
        pass

    @property
    @abc.abstractmethod
    def size(self) -> int:
        pass
