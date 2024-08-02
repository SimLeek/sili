import numpy as np

from sili.core.devices.gpu import GPUManager
from sili.core.serial import deserialize_buffer, serialize_buffer


class ConvDepthBuffer(object):
    def __init__(self, gpu: GPUManager, array):
        if isinstance(array, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.depth_in = array.shape[0]
            self.depth_out = array.shape[1]
            self.buffer = gpu.buffer(array)
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def __setstate__(self, state):
        self.depth_in, self.depth_out = state[:2]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.depth_in, self.depth_out, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.depth_in * self.depth_out

    def set(self, array):
        if isinstance(array, np.ndarray):
            self.buffer.data()[...] = array.flatten()
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def get(self):
        return self.buffer.data().reshape(self.depth_in, self.depth_out)


class ConvVertPyrBuffer(object):
    def __init__(self, gpu: GPUManager, array):
        if isinstance(array, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.height = array.shape[0]
            self.buffer = gpu.buffer(array)
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def __setstate__(self, state):
        self.height = state[:2]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.height, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.height

    def set(self, array):
        if isinstance(array, np.ndarray):
            self.buffer.data()[...] = array.flatten()
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def get(self):
        return self.buffer.data().reshape([self.height])


class ConvDepthReductionBuffer(object):
    """Useful for running compute reductions for determining updates"""

    def __init__(self, gpu: GPUManager, array):
        if isinstance(array, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.duplicates = array.shape[0]
            self.depth_in = array.shape[1]
            self.depth_out = array.shape[2]
            self.buffer = gpu.buffer(array)
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def __setstate__(self, state):
        self.duplicates, = self.depth_in, self.depth_out = state[:3]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.duplicates, self.depth_in, self.depth_out, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.duplicates * self.depth_in * self.depth_out

    def set(self, array):
        if isinstance(array, np.ndarray):
            self.buffer.data()[...] = array.flatten()
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def get(self):
        return self.buffer.data().reshape(self.duplicates, self.depth_in, self.depth_out)
