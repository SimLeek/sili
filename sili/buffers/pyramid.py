import warnings

import kp
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from rectpack import newPacker
from rectpack.maxrects import MaxRectsBssf
import numpy as np

from sili.core.devices.gpu import GPUManager
from sili.core.serial import deserialize_buffer, serialize_buffer
from sili.buffers.image import ImageBuffer

class ImagePyramidBuffer(object):
    def __init__(self, gpu: GPUManager, original_image:Union[ImageBuffer, np.ndarray], rect_pack:Union[ImageBuffer, np.ndarray], levels:Union[List[int],np.ndarray], memory_type=kp.deviceAndHost):
        self.levels = levels
        self.channels = original_image.colors
        self.memory_type = memory_type

        if isinstance(original_image, np.ndarray):
            original_image = ImageBuffer(gpu, original_image, original_image.dtype, memory_type)
        if isinstance(original_image, ImageBuffer):
            self.original_image = original_image
            self.channels = original_image.colors
        else:
            raise TypeError(f"Unknown type for original_image: {type(original_image)}")

        if isinstance(rect_pack, np.ndarray):
            rect_pack = ImageBuffer(gpu, rect_pack, rect_pack.dtype, memory_type)
        if isinstance(original_image, ImageBuffer):
            self.rect_pack = rect_pack
        else:
            raise TypeError(f"Unknown type for rect_pack: {type(rect_pack)}")

        if not isinstance(levels, np.ndarray):
            levels_str = struct.pack(f'={len(levels)}i', *levels)
            levels = np.frombuffer(levels_str, dtype=np.int32)

        self.pyr_lvl_buffer = gpu.manager.tensor_t(levels, self.memory_type)


    def to(self, t):
        """Move this buffer to another device"""
        if isinstance(t, GPUManager):
            self.original_image.to(t)
            self.rect_pack.to(t)
            if isinstance(self.pyr_lvl_buffer, kp.Tensor):
                self.pyr_lvl_buffer = self.pyr_lvl_buffer.data()
            self.pyr_lvl_buffer = t.manager.tensor_t(self.pyr_lvl_buffer, self.memory_type)
        return self

    @property
    def size(self):
        return self.original_image.size + self.rect_pack.size + self.pyr_lvl_buffer.data().size

    def set_original(self, image):
        self.original_image.set(image)

    def set_rectpack(self, rectpack):
        self.rect_pack.set(rectpack)

    def set_levels(self, levels):
        if isinstance(levels, np.ndarray):
            dst = self.pyr_lvl_buffer.data()
            #if levels.shape != dst.shape:
            #    raise ValueError(f"Shape mismatch: input {levels.shape} vs buffer {dst.shape}")
            if levels.dtype != dst.dtype:
                raise ValueError(f"Dtype mismatch: input {levels.dtype} vs buffer {dst.dtype}")
            np.copyto(dst, levels, casting='no')
        else:
            raise NotImplementedError(f'Unknown rectpack type: {type(levels)}')

    def get_original(self):
        return self.original_image.get()

    def get_rectpack(self):
        return self.rect_pack.get()

    def get_levels(self):
        return self.pyr_lvl_buffer.data()

    def __setstate__(self, state):
        self.levels, self.channels, self.dtype, self.memory_type, self.original_image, self.rect_pack= state[:6]
        self.pyr_lvl_buffer = deserialize_buffer(state[6])

    def __getstate__(self):
        return (self.levels, self.channels, self.dtype, self.memory_type, self.original_image, self.rect_pack,
                serialize_buffer(self.pyr_lvl_buffer))

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

def generate_downscaled_rectangles(start_height, start_width, scale=np.sqrt(np.e), c=3, pad_w=0, pad_h=0):
    # why scale should be sqrt(e) (for 2D): https://arxiv.org/abs/1304.0031
    rectangles = []
    start_idx = 0
    w, h = int(start_width), int(start_height)
    assert scale>1.0

    while not( w == 1 and h == 1):
        w = int(max(np.floor(w / scale), 1))
        h = int(max(np.floor(h / scale), 1))
        rectangles.append((w + pad_w, h + pad_h))

    levels = []
    for rectangle in reversed(rectangles):
        w, h = rectangle
        levels.extend([
            start_idx,
            h,
            w
        ])
        start_idx+=w*h
    levels = [c, int(len(levels) // 3)] + levels
    return levels

@dataclass
class Rectangle:
    width: int
    height: int

@dataclass
class PackingResult:
    width: int
    height: int
    area: int
    total_area: int
    efficiency: float  # total_area / area * 100
    unused_percent: float
    unused_pixels: int
    rectangles: List[Tuple[int, int, int, int, int, int]]  # (bin, x, y, w, h, rid)

def pack_rectangles_minimal(start_width: int, start_height: int, scale: float = np.sqrt(np.e), c: int = 3, pad_w: int = 0, pad_h: int = 0, show_progress: bool = False) -> Tuple[Optional[PackingResult], List[int]]:
    # Generate rectangles using the provided function
    levels = generate_downscaled_rectangles(start_height, start_width, scale, c, pad_w, pad_h)
    c, num_rects = levels[0], levels[1]
    rectangles = []
    idx = 2
    for _ in range(num_rects):
        _, h, w = levels[idx:idx+3]
        rectangles.append((w, h))
        idx += 3

    total_area = sum(w * h for w, h in rectangles)

    def get_packed_height(width: int) -> Optional[int]:
        packer = newPacker(pack_algo=MaxRectsBssf)
        for r in rectangles:
            packer.add_rect(*r)
        packer.add_bin(width, float('inf'))
        packer.pack()
        rects = packer.rect_list()
        if len(rects) != len(rectangles):
            return None
        return max(y + h for _, x, y, w, h, _ in rects)

    def get_packed_width(height: int) -> Optional[int]:
        packer = newPacker(pack_algo=MaxRectsBssf)
        for r in rectangles:
            packer.add_rect(*r)
        packer.add_bin(float('inf'), height)
        packer.pack()
        rects = packer.rect_list()
        if len(rects) != len(rectangles):
            return None
        return max(x + w for _, x, y, w, h, _ in rects)

    best = None
    best_rects = None
    min_width = max(w for w, _ in rectangles)
    max_width = sum(w for w, _ in rectangles)
    min_height = max(h for _, h in rectangles)
    max_height = sum(h for _, h in rectangles)

    # Try packing with varying widths
    total_widths = max_width - min_width + 1
    for i, width in enumerate(range(min_width, max_width + 1)):
        if show_progress:
            progress = (i + 1) / total_widths * 100
            print(f'\rPacking widths: {progress:.1f}%', end='')
        height = get_packed_height(width)
        if height is not None:
            area = width * height
            packer = newPacker(pack_algo=MaxRectsBssf)
            for r in rectangles:
                packer.add_rect(*r)
            packer.add_bin(width, height)
            packer.pack()
            rects = packer.rect_list()
            if best is None or area < best[0]:
                best = (area, width, height)
                best_rects = rects
            if height == min_height:
                break
    if show_progress:
        print('\rPacking widths: 100.0%\n', end='')

    # Try packing with varying heights
    total_heights = max_height - min_height + 1
    for i, height in enumerate(range(min_height, max_height + 1)):
        if show_progress:
            progress = (i + 1) / total_heights * 100
            print(f'\rPacking heights: {progress:.1f}%', end='')
        width = get_packed_width(height)
        if width is not None:
            area = width * height
            packer = newPacker(pack_algo=MaxRectsBssf)
            for r in rectangles:
                packer.add_rect(*r)
            packer.add_bin(width, height)
            packer.pack()
            rects = packer.rect_list()
            if best is None or area < best[0]:
                best = (area, width, height)
                best_rects = rects
            if width == min_width:
                break
    if show_progress:
        print('\rPacking heights: 100.0%\n', end='')

    if best is None:
        raise RuntimeError("Could not pack rectangles")

    area, width, height = best
    efficiency = (total_area / area) * 100.0
    unused_percent = (1.0 - total_area / area) * 100.0
    unused_pixels = int(round(area * (1.0 - total_area / area)))

    # Update levels with packing results
    updated_levels = [c, num_rects]
    for rect in reversed(best_rects):  #I'm assuming reverse goes from lowest area to largest, because that seems to be the case
        _, x, y, w, h, _ = rect
        updated_levels.extend([x, y, w, h])
        #updated_levels.extend([y, x, h, w])

    if efficiency<90:
        warnings.warn("Inefficient rectpack.")

    return PackingResult(
        width=width,
        height=height,
        area=area,
        total_area=total_area,
        efficiency=efficiency,
        unused_percent=unused_percent,
        unused_pixels=unused_pixels,
        rectangles=best_rects
    ), updated_levels