import numpy as np
import kp
from sili.core.buffers import Buffer
from sili.core.devices.gpu import GPUManager
from sili.core.serial import deserialize_buffer, serialize_buffer

class ImageBuffer(Buffer):
    def __init__(self, gpu: GPUManager, image, include_dim_in_array=False, dtype=None, memory_type=kp.MemoryTypes.deviceAndHost):
        """

        :param gpu:
        :param image: The image numpy array
        :param include_dim_in_array: Bool. Stores dimensions as first part of image buffer. Useful for many images or changing image sizes, but requires an initial input synchronization, as the dimensions actually need to be populated, so it makes using it as an output harder.
        :param dtype:
        :param memory_type:
        """
        super().__init__(gpu)
        self.memory_type = memory_type
        self.include_dim_in_array = include_dim_in_array
        if dtype is None:
            self.dtype = image.dtype
        else:
            self.dtype = dtype
        if isinstance(image, np.ndarray):
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.colors = image.shape[2]
            self.actual_colors = self.colors
            if self.colors == 3 and image.dtype in [np.uint8, np.int8]:
                # pad input to rgba with alpha=1. Alpha will stay 1 forever unless touched by glsl somehow.
                image = np.pad(image.view(np.uint8), ((0,0),(0,0),(0,1)), constant_values=255).view(image.dtype)
                self.actual_colors = 4
            if include_dim_in_array:
                image = np.concatenate((np.asarray([self.height, self.width, self.colors], dtype=np.uint32), image.flatten().view(np.uint32)))
            self.buffer = gpu.buffer(image.flatten().view(np.float32), memory_type)
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def __setstate__(self, state):
        self.height, self.width, self.colors, self.dtype, self.memory_type= state[:5]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[5])

    def __getstate__(self):
        # Return state to be pickled
        return (self.height, self.width, self.colors, self.dtype, self.memory_type,
                serialize_buffer(self.buffer))

    @property
    def size(self):
        return self.height * self.width * self.colors

    @property
    def shape(self):
        return self.height, self.width, self.colors

    def to(self, t):
        """Move this buffer to another device"""
        if isinstance(t, GPUManager):
            if isinstance(self.buffer, kp.Tensor):
                self.buffer = self.buffer.data()
            self.buffer = t.buffer(self.buffer.flatten().view(np.float32), self.memory_type)

    def set(self, image):
        if self.include_dim_in_array:
            skip=3
        else:
            skip=0
        if isinstance(image, np.ndarray):
            colors = image.shape[2] if len(image.shape) == 3 else 1
            assert colors in [1, 2, 3, 4], f"Unsupported number of channels: {colors}. Must be 1, 2, 3, or 4."
            if colors == 3 and image.dtype in [np.uint8, np.int8]:
                # Evil c casting bullshit:
                buffer_view = self.buffer.data()[skip:].view(np.uint8).reshape(image.shape[0], image.shape[1], 4)
                buffer_view[:, :, :3] = image
            elif colors == 3 and image.dtype==[np.uint32, np.int32, np.float32]:
                self.buffer.data()[skip:][...] = image.flatten().view(np.float32)
            elif colors == 3:
                raise TypeError(f"numpy dtype {image.dtype} not supported for 3 channels")
            else:
                self.buffer.data()[skip:][...] = image.flatten().view(np.float32)
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def get(self):
        if self.include_dim_in_array:
            skip=3
        else:
            skip=0
        if self.colors == 3 and self.dtype in [np.uint8, np.int8]:
            # Undo evil c casting bullshit:
            buffer_view = self.buffer.data()[skip:].view(np.uint8).reshape(self.height, self.width, 4)
            #buffer_view = buffer_view[..., :3]
            return buffer_view
        else:
            # For 1, 2, or 4 channels, reshape directly
            return self.buffer.data()[skip:].view(self.dtype).reshape(self.height, self.width, self.colors)

class GrayImageBuffer(Buffer):
    def __init__(self, gpu: GPUManager, image):
        super().__init__(gpu)
        if isinstance(image, np.ndarray):
            # Assume this is a numpy image from OpenCV (height, width)
            self.height = image.shape[0]
            self.width = image.shape[1]
            # Ensure image is in uint8 format for single-channel (raw camera data)
            image = image.astype(np.uint8)
            # Create Kompute image with single-channel format using image_t
            self.buffer = gpu.manager.image_t(
                data=image.flatten(),
                width=self.width,
                height=self.height,
                num_channels=1,
                memory_type=kp.MemoryTypes.device
            )
            self.buffer.recordCopyFromHost(image.flatten())
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def __setstate__(self, state):
        self.height, self.width = state[:2]
        # Restore the image buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled
        return (self.height, self.width, serialize_buffer(self.buffer))

    @property
    def size(self):
        return self.height * self.width

    def set(self, image):
        if isinstance(image, np.ndarray):
            if image.shape != (self.height, self.width):
                raise ValueError(f"Image shape {image.shape} does not match buffer dimensions")
            image = image.astype(np.uint8)
            self.buffer.recordCopyFromHost(image.flatten())
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def get(self):
        data = self.buffer.data()
        return data.reshape(self.height, self.width).astype(np.uint8)

class ndarrayBuffer(Buffer):
    def __init__(self, gpu: GPUManager, array, type=np.float32):
        super().__init__(gpu)
        self.type = type
        if isinstance(array, np.ndarray):
            self.shape = array.shape
            self.type = array.dtype if type is None else type
            array = array.astype(self.type)
            # Use Kompute tensor_t for non-image arrays
            self.buffer = gpu.manager.tensor_t(
                data=array.flatten(),
                memory_type=kp.MemoryTypes.device
            )
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def __setstate__(self, state):
        self.shape = state[0]
        self.type = state[1]
        # Restore the tensor buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled
        return (self.shape, self.type, serialize_buffer(self.buffer))

    @property
    def size(self):
        return np.prod(self.shape)

    def set(self, array):
        if isinstance(array, np.ndarray):
            if array.shape != self.shape:
                raise ValueError(f"Array shape {array.shape} does not match buffer shape")
            array = array.astype(self.type)
            self.buffer.recordCopyFromHost(array.flatten())
        else:
            raise NotImplementedError(f'Unknown array type: {type(array)}')

    def get(self):
        data = self.buffer.data()
        return data.reshape(self.shape).astype(self.type)