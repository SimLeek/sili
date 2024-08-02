import kp
import struct

import numpy as np

from sili.core.devices.gpu import GPUManager
from sili.core.serial import deserialize_buffer, serialize_buffer


class ImagePyramidBuffer(object):
    def __init__(self, gpu: GPUManager, levels, channels=3, use_lvl_buf=True, type=np.float32):
        self.type = type
        self.levels = levels
        self.channels = channels

        size = channels * (self.levels[-3] + self.levels[-2]*self.levels[-1]*self.channels)  # start, w, h sequence. last w,h is 1x1, so last start is size-1.
        pyr_np = np.zeros((size,), dtype=type)
        pad_len = -pyr_np.size * np.dtype(pyr_np.dtype).itemsize % np.dtype(np.float32).itemsize
        if pad_len != 0:
            pyr_np = np.pad(pyr_np.flatten(), (0, pad_len))
        self.image_buffer = gpu.buffer(pyr_np.view(np.float32))

        levels_str = struct.pack(f'={len(levels)}i', *levels)
        levels_glsl = np.frombuffer(levels_str, dtype=np.float32)
        self.use_lvl_buf = use_lvl_buf
        if self.use_lvl_buf:
            self.pyr_lvl_buffer = gpu.buffer(levels_glsl)
        else:
            self.pyr_lvl_buffer = None

    def to(self, t):
        if isinstance(t, GPUManager):
            if isinstance(self.image_buffer, kp.Tensor):
                self.image_buffer = self.image_buffer.data()
            self.image_buffer = t.buffer(self.image_buffer)
            if self.use_lvl_buf:
                if isinstance(self.pyr_lvl_buffer, kp.Tensor):
                    self.pyr_lvl_buffer = self.pyr_lvl_buffer.data()
                self.pyr_lvl_buffer = t.buffer(self.pyr_lvl_buffer)
        return self

    @property
    def size(self):
        return self.image_buffer.size()

    def set(self, image):
        if isinstance(image, np.ndarray):
            pad_len = -image.size * np.dtype(image.dtype).itemsize % np.dtype(np.float32).itemsize
            if pad_len != 0:
                image = np.pad(image.flatten(), (0, pad_len))
            self.image_buffer.data()[...] = image.flatten().view(np.float32)
        else:
            raise NotImplementedError(f'Unknown image type: {type(image)}')

    def get(self):
        im_list = []
        flattened_array = self.image_buffer.data().view(self.type)
        for l in range(2, len(self.levels), 3):
            start, width, height = int(self.levels[l]), int(self.levels[l + 1]), int(self.levels[l + 2]),
            end_idx = int(start * self.channels + (width * height * self.channels))

            image = flattened_array[int(start * self.channels):end_idx].reshape(width, height, self.channels)
            im_list.append(image)
        return im_list

    def __setstate__(self, state):
        self.levels, self.channels, self.use_lvl_buf = state[:3]
        # Restore the buffer from serialized data
        self.image_buffer = deserialize_buffer(state[3])
        if self.use_lvl_buf:
            self.pyr_lvl_buffer = deserialize_buffer(state[4])
        else:
            self.pyr_lvl_buffer = None

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (self.levels, self.channels, self.use_lvl_buf,
                serialize_buffer(self.image_buffer),
                serialize_buffer(self.pyr_lvl_buffer) if self.use_lvl_buf else None)


def calculate_pyramid_levels(h, w, s, c=3):
    levels = []
    start_idx = 0
    base_h, base_w = h, w

    current_h, current_w = 1, 1

    while current_h <= base_h or current_w <= base_w:
        levels.extend([
            start_idx,
            min(current_h, base_h),
            min(current_w, base_w)
        ])

        if current_h==base_h and current_w==base_w:
            break

        start_idx += int(min(current_h, base_h) * min(current_w, base_w))

        next_h = int(max(np.ceil(current_h * np.sqrt(2)), current_h + 1))
        next_w = int(max(np.ceil(current_w * np.sqrt(2)), current_w + 1))

        current_h, current_w = next_h, next_w

        if current_h > base_h:
            current_h = base_h
        if current_w > base_w:
            current_w = base_w

    levels = [c, int(len(levels) // 3)] + levels

    return levels
