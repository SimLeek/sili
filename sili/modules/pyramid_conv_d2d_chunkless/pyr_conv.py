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

# todo: merge this with the main conv and use gpu info to determine which algorithm to use.

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
        self.in_img_pyr = ImagePyramidBuffer(self.gpu, image_pyr.levels, image_pyr.channels, image_pyr.use_lvl_buf,
                                             type=image_pyr.type
                                             )
        self.out_pyr = ImagePyramidBuffer(gpu, self.in_img_pyr.levels, type=np.float32)

        # todo: do 27 tests for input position, and up to 9 tests for color, using a one hot like matrix
        if callable(init_conv):
            self.conv = ndarrayBuffer(gpu, init_conv(3, 3, 3, 3, 3))
        elif isinstance(init_conv, np.ndarray):
            self.conv = ndarrayBuffer(gpu, init_conv)
        else:
            self.conv = ndarrayBuffer(gpu, np.ones((3, 3, 3, 3, 3)))

        self.conv_str = ndarrayBuffer(gpu, np.zeros(self.conv.shape))  # adagrad weights

        # PIPELINE OBJECTS:
        self.forward_input_buffers = [self.conv.buffer, self.in_img_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer]
        self.forward_output_buffers = [self.out_pyr.image_buffer]

        chunk_w, chunk_h = find_good_dimension_sizes(self.gpu.max_workgroup_invocations, 2)

        self.forward_algorithm = self.gpu.manager.algorithm(
            [*self.forward_input_buffers, *self.forward_output_buffers],
            spirv=self.forward_shader,
            workgroup=[int(np.ceil(self.in_img_pyr.size / (self.gpu.max_workgroup_invocations))), 0, 0],
            spec_consts=np.asarray([
                self.gpu.max_workgroup_invocations,
                3,3,3,3,3,
                chunk_w, chunk_h
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

if __name__ == '__main__':
    input_pyr_file = "../../../test/files/test_ai_pyr_pls_ignore.pyr"

    from sili.util.preset_convs import get_edge_detector_kernel

    edge = get_edge_detector_kernel(3, 3)
    display_pyramid(input_pyr_file, edge)  # Initial display without specific kernel

    # Run spatial and color tests
    run_spatial_tests(input_pyr_file)
    run_color_tests(input_pyr_file)

    # generate_pyramid_file("../../../test/files/test_ai_pls_ignore.png",
    #                      "../../../test/files/test_ai_pyr_pls_ignore.pyr")

    #display_pyramid_from_camera(0)