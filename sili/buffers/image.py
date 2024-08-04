import numpy as np

from sili.core.buffers import Buffer
from sili.core.devices.gpu import GPUManager
from sili.core.serial import deserialize_buffer, serialize_buffer


class ImageBuffer(Buffer):
    def __init__(self, gpu: GPUManager, image, type=np.float32):
        super().__init__(gpu)
        self.type = type
        if isinstance(image, np.ndarray):
            # This matches numpy images from OpenCV:
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.colors = image.shape[2]
            pad_len = -image.size*np.dtype(image.dtype).itemsize%np.dtype(np.float32).itemsize
            if pad_len!=0:
                image = np.pad(image.flatten(), (0, pad_len))
            self.buffer = gpu.buffer(image.flatten().view(np.float32))
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def __setstate__(self, state):
        self.height, self.width, self.colors = state[:3]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (self.height, self.width, self.colors,
                serialize_buffer(self.buffer))

    @property
    def size(self):
        return self.height * self.width * self.colors

    def set(self, image):
        if isinstance(image, np.ndarray):
            pad_len = -image.size*np.dtype(image.dtype).itemsize%np.dtype(np.float32).itemsize
            if pad_len!=0:
                image = np.pad(image.flatten(), (0, pad_len))
            self.buffer.data()[...] = image.flatten().view(np.float32)
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def get(self):
        return self.buffer.data().reshape(self.height, self.width, self.colors).view(self.type)

class GrayImageBuffer(object):
    def __init__(self, gpu: GPUManager, image):
        if isinstance(image, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.buffer = gpu.buffer(image)
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def __setstate__(self, state):
        self.height, self.width = state[:2]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (self.height, self.width,
                serialize_buffer(self.buffer))

    @property
    def size(self):
        return self.height * self.width

    def set(self, image):
        if isinstance(image, np.ndarray):
            self.buffer.data()[...] = image.flatten()
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def get(self):
        return self.buffer.data().reshape(self.height, self.width)


class ndarrayBuffer(Buffer):
    def __init__(self, gpu: GPUManager, array, type=np.float32):
        super().__init__(gpu)
        self.type = type
        if isinstance(array, np.ndarray):
            # This matches numpy images from OpenCV:
            self.shape = array.shape
            self.type = array.dtype
            pad_len = -array.size*np.dtype(array.dtype).itemsize%np.dtype(np.float32).itemsize
            if pad_len!=0:
                array = np.pad(array.flatten(), (0, pad_len))
            self.buffer = gpu.buffer(array.flatten().view(np.float32))
        else:
            raise NotImplementedError(f'Unknown image type: {type(array)}')

    def __setstate__(self, state):
        self.shape = state[0]
        self.type = state[1]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (self.shape, self.type, serialize_buffer(self.buffer))

    @property
    def size(self):
        return np.prod(self.shape)

    def set(self, image):
        if isinstance(image, np.ndarray):
            pad_len = -image.size*np.dtype(image.dtype).itemsize%np.dtype(np.float32).itemsize
            if pad_len!=0:
                image = np.pad(image.flatten(), (0, pad_len))
            self.buffer.data()[...] = image.flatten().view(np.float32)
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def get(self):
        return self.buffer.data().reshape(self.shape).view(self.type)
