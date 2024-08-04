import pickle

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
                3,3,3,3,
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

    pyr.in_img_pyr.set(pyr_in.image_buffer)
    seq.eval()
    out_images = pyr.out_pyr.get()

    for i, o in enumerate(out_images):
        display.imshow(f'output {i}', image_normalize(o))
    while True:
        display.update()
        if display.window.is_closing:
            break


def generate_pyramid_file(input_image_file, output_pyr_file):
    from sili.core.runners import get_forward_sequence
    import pickle

    gpu = GPUManager()
    im = cv2.imread(input_image_file).astype(np.uint8)
    pyr = ToImagePyramid(gpu, im)  # im used to set size

    seq = get_forward_sequence([pyr])
    pyr.image.set(im)
    seq.eval()
    with open(output_pyr_file, mode='wb') as f:
        pickle.dump(pyr.out_pyr, f)
    with open(output_pyr_file, mode='rb') as f:
        im_pyr = pickle.load(f)
        print(im_pyr)


def display_pyramid(input_pyr_file):
    from sili.core.runners import get_forward_sequence

    gpu = GPUManager()
    with open(input_pyr_file, mode='rb') as f:
        pyr_in = pickle.load(f)
    pyr = PyrConv(gpu, pyr_in)  # im used to set width, height, and channels
    seq = get_forward_sequence([pyr])
    display_basic_forward_sequence(pyr, seq, pyr_in)


def display_pyramid_from_camera(camera):
    from sili.core.runners import get_forward_sequence
    from displayarray import read_updates, DirectDisplay
    import time

    #r = read_updates(camera, size=(9999, 9999))
    #r = read_updates(camera, size=(1280,960))
    r = read_updates(camera, size=(-1, -1))
    gpu = GPUManager()
    first = True
    pyr = None
    seq = None
    display = DirectDisplay()
    while r:
        if not r.frames:
            continue
        im = r.frames[str(camera)][0].astype(np.uint8)
        if first:
            pyr = ToImagePyramid(gpu, im)  # im used to set width, height, and channels
            seq = get_forward_sequence([pyr])
            first = False
        pyr.image.set(im)
        t0 = time.time()
        seq.eval()
        t1 = time.time()
        out_images = pyr.out_pyr.get()
        print(f"eval time:{t1 - t0}, fps:{1. / (t1 - t0)}")
        for i, o in enumerate(out_images):
            display.imshow(f'output {i}', o)
        display.update()
        if display.window.is_closing:
            break

if __name__ == '__main__':
    display_pyramid("../../../test/files/test_ai_pyr_pls_ignore.pyr")

    # generate_pyramid_file("../../../test/files/test_ai_pls_ignore.png",
    #                      "../../../test/files/test_ai_pyr_pls_ignore.pyr")

    #display_pyramid_from_camera(0)