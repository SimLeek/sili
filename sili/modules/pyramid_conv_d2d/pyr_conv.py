import pickle
import time

import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os

from sili.buffers.pyramid import ImagePyramidBuffer, calculate_pyramid_levels
from sili.buffers.image import ImageBuffer, ndarrayBuffer
from sili.core.devices.gpu import GPUManager, get_shader
from sili.core.module import Module
from sili.core.util import find_good_dimension_sizes
from typing import Union

file_path = os.path.dirname(os.path.abspath(__file__))

def calculate_max_chunk_start_index(pyramid_levels, chunk_width, chunk_height):
    current_pyr_index = 0

    levels = pyramid_levels[1]
    for level in range(levels):
        p_lvl = 2+(level)*3
        current_pyr_index += (
            np.ceil(pyramid_levels[p_lvl+1] / chunk_width) * chunk_width *
            np.ceil(pyramid_levels[p_lvl+2] / chunk_height) * chunk_height
        )

    return current_pyr_index

class PyrConv(Module):
    def __init__(self,
                 gpu: GPUManager,
                 image_pyr: ImagePyramidBuffer,
                 init_conv=None,
                 backprop_input_buf=None,
                 backprop_conv=False
                 ):
        """Pyramidal convolution"""

        self.gpu = gpu
        self.forward_shader = get_shader(file_path + os.sep + 'conv_pyr_forward.comp')

        assert isinstance(image_pyr, ImagePyramidBuffer)
        #self.in_img_pyr = ImagePyramidBuffer(self.gpu, image_pyr.levels, image_pyr.channels, image_pyr.use_lvl_buf,
        #                                     type=image_pyr.type
        #                                     )
        self.in_img_pyr = image_pyr
        self.out_pyr = ImagePyramidBuffer(gpu, self.in_img_pyr.levels, type=np.float32)

        if callable(init_conv):
            start_conv = init_conv(3, 3, 3, 3, 3)
            self.conv = ndarrayBuffer(gpu, start_conv)
        elif isinstance(init_conv, np.ndarray):
            start_conv = init_conv
            self.conv = ndarrayBuffer(gpu, init_conv)
        else:
            start_conv = np.ones((3, 3, 3, 3, 3))
            self.conv = ndarrayBuffer(gpu, start_conv)

        self.conv_str = ndarrayBuffer(gpu, np.zeros(self.conv.shape))  # adagrad weights

        # PIPELINE OBJECTS:
        self.forward_input_buffers = [self.conv.buffer, self.in_img_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer]
        self.forward_output_buffers = [self.out_pyr.image_buffer]

        chunk_w, chunk_h = find_good_dimension_sizes(self.gpu.max_workgroup_invocations, 2)
        workgroups = calculate_max_chunk_start_index(image_pyr.levels, chunk_w, chunk_h)
        self.forward_algorithm = self.gpu.manager.algorithm(
            [*self.forward_input_buffers, *self.forward_output_buffers],
            spirv=self.forward_shader,
            workgroup=[int(np.ceil(workgroups / (self.gpu.max_workgroup_invocations))), 0, 0],
            spec_consts=np.asarray([
                self.gpu.max_workgroup_invocations,
                *self.conv.shape,
                chunk_w, chunk_h, 1, 1
            ], dtype=np.uint32).view(np.float32)
        )

    def forward_ops(self):
        ops = [kp.OpAlgoDispatch(self.forward_algorithm)]
        return ops

    def backward_ops(self):
        return []

    def optim_ops(self):
        return []


def image_normalize(im:np.ndarray):
    max_val = np.max(im)
    min_val = np.min(im)
    span = max_val-min_val
    if span==0:
        span= 1
    im2 = (im-min_val)/span  # now 0 to 1
    return im2

def fast_normalize(im:np.ndarray):
    max_val = 255
    min_val = -255
    span = max_val-min_val
    if span==0:
        span= 1
    im2 = (im-min_val)/span  # now 0 to 1
    return im2

def display_basic_forward_sequence(pyr, seq, pyr_in:ImagePyramidBuffer, display=None):
    if display is None:
        display = DirectDisplay()
        while display.window.is_closing:
            display = DirectDisplay()

    import time
    pyr.in_img_pyr.set(pyr_in.image_buffer)
    t0 = time.time()
    seq.eval()
    t1 = time.time()
    print(f"eval time:{t1 - t0}, fps:{1. / (t1 - t0)}")
    out_images = pyr.out_pyr.get()

    for i, o in enumerate(out_images):
        display.imshow(f'output {i}', image_normalize(o))
    while True:
        display.update()
        if display.window.is_closing:
            break



def display_pyramid(input_pyr_file, kernel=None):
    from sili.core.runners import get_forward_sequence

    gpu = GPUManager()
    with open(input_pyr_file, mode='rb') as f:
        pyr_in = pickle.load(f)
    pyr = PyrConv(gpu, pyr_in, init_conv=kernel)  # im used to set width, height, and channels
    seq = get_forward_sequence([pyr])
    display_basic_forward_sequence(pyr, seq, pyr_in)

def run_spatial_tests(input_pyr_file):
    import itertools
    print("Running spatial tests...")
    for d, h, w in itertools.product((0, 1), (0, 1), (0, 1)):
        kernel = np.zeros((3, 3, 3, 3, 3), dtype=np.float32)
        kernel[d, w, h, :, :] = 1  # Emphasize the specific dimension
        print(f"Testing spatial dimension: D={d}, H={h}, W={w}")
        display_pyramid(input_pyr_file, kernel)

def run_color_tests(input_pyr_file):
    print("Running color tests...")
    # Generate kernels for color tests (rgb_to_r, r_to_rgb)
    bgr_to_b = np.zeros((3, 3, 3, 3, 3), dtype=np.float32)
    b_to_rgb = np.zeros((3, 3, 3, 3, 3), dtype=np.float32)
    # RGB to R
    # OIDHW
    bgr_to_b[1, 1, 1, :, 1] = 1  # Only take the blue channel
    # R to RGB
    b_to_rgb[1, 1, 1, 1, :] = 1  # Replicate the blue channel to all RGB channels
    print("Testing color conversion: bgr_to_b")
    display_pyramid(input_pyr_file, bgr_to_b)

    print("Testing color conversion: b_to_rgb")
    display_pyramid(input_pyr_file, b_to_rgb)

def display_pyramid_from_camera(camera, conv_in):
    from sili.core.runners import get_forward_sequence, get_fast_forward_sequence
    from displayarray import read_updates, DirectDisplay
    from sili.modules.image_pyramid_int8.image_pyramid import ToImagePyramid
    import time
    import sys
    sys.setswitchinterval(0.1)

    #r = read_updates(camera, size=(9999, 9999), fps_limit=120)
    #r = read_updates(camera, size=(1280,720), fps_limit=120)
    r = read_updates(camera, size=(640, 360), fps_limit=120)
    #r = read_updates(camera, size=(480, 240), fps_limit=120)
    #r = read_updates(camera, size=(1, 1), fps_limit=120)
    gpu = GPUManager(kp.Manager(1))
    first = True
    pyr = None
    conv=None
    seq = None
    display = DirectDisplay()
    while r:
        if not r.frames:
            continue
        im = r.frames[str(camera)][0].astype(np.uint8)
        if first:
            pyr = ToImagePyramid(gpu, im)  # im used to set width, height, and channels
            conv = PyrConv(gpu, pyr.out_pyr, init_conv=conv_in)
            seq = get_fast_forward_sequence([pyr, conv])
            first = False
        pyr.image.set(im)
        t0 = time.time()
        seq.eval()
        t1 = time.time()
        out_images = conv.out_pyr.get()
        print(f"eval time:{t1 - t0}, fps:{1. / (t1 - t0)}")
        for i, o in enumerate(out_images):
            display.imshow(f'output {i}', fast_normalize(o))
        display.update()
        if display.window.is_closing:
            break

if __name__ == '__main__':
    input_pyr_file = "../../../test/files/test_ai_pyr_pls_ignore.pyr"

    from sili.util.preset_convs import get_edge_detector_kernel

    # todo: optimize kernel for 1 depth (most importance), 1 width, 1 height, etc.
    edge = get_edge_detector_kernel(2, 3)
    edge = edge[:, :, np.newaxis, :, :]
    display_pyramid_from_camera('../../../test/files/drone_test_vid_360.mp4', edge)
    #display_pyramid_from_camera(0, edge)

    #display_pyramid(input_pyr_file, edge)  # Initial display without specific kernel

    # Run spatial and color tests
    #run_spatial_tests(input_pyr_file)
    #run_color_tests(input_pyr_file)

    # generate_pyramid_file("../../../test/files/test_ai_pls_ignore.png",
    #                      "../../../test/files/test_ai_pyr_pls_ignore.pyr")

    #