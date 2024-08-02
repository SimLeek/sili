import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os

from sili.buffers.pyramid import ImagePyramidBuffer, calculate_pyramid_levels
from sili.buffers.image import ImageBuffer
from sili.core.devices.gpu import GPUManager, get_shader
from sili.core.module import Module
from typing import Union

file_path = os.path.dirname(os.path.abspath(__file__))

class ToImagePyramid(Module):
    def __init__(self,
                 gpu: GPUManager,
                 image: Union[ImageBuffer, np.ndarray],
                 scale: float = np.sqrt(2)):
        assert scale > 1.0, "Scale must be large enough to divide the image"

        self.gpu = gpu
        self.forward_shader_clear = get_shader(file_path + os.sep + 'clear_pyramid.comp')
        self.forward_shader_large = get_shader(file_path + os.sep + 'pyramid_gen_large.comp')
        self.forward_shader_small = get_shader(file_path + os.sep + 'pyramid_gen_small.comp')
        self.count_shader = get_shader(file_path + os.sep + 'count_increment.comp')

        if not isinstance(image, ImageBuffer):
            self.image = ImageBuffer(self.gpu, image, type=np.uint8)
        else:
            self.image = image
        pyr_levels = calculate_pyramid_levels(
            self.image.height,
            self.image.width,
            scale,
            self.image.colors
        )
        self.out_pyr = ImagePyramidBuffer(gpu, pyr_levels, type=np.uint8)
        self.level_count_buffer = self.gpu.buffer(np.array([self.out_pyr.levels[1]-1], dtype=np.int32).view(np.float32))  # start at last/base level

        # PIPELINE OBJECTS:
        # these need to be accessible so that kompute can record input/output for pipelines
        self.forward_input_buffers = [self.image.buffer, self.out_pyr.pyr_lvl_buffer]
        self.forward_output_buffers = [self.out_pyr.image_buffer]
        self.internal_buffers = [self.level_count_buffer]

        self.clear_algorithm = self.gpu.manager.algorithm(
            [*self.forward_output_buffers],
            spirv=self.forward_shader_clear,
            workgroup=[int(np.ceil(self.out_pyr.size / (self.gpu.max_workgroup_invocations))), 0, 0],
            spec_consts=np.asarray([self.gpu.max_workgroup_invocations, self.out_pyr.size], dtype=np.uint32).view(
                np.float32)
        )
        self.count_algorithm = self.gpu.manager.algorithm(
            [*self.internal_buffers],
            spirv=self.count_shader,
            workgroup=[1, 0, 0]
        )

        self.split_point = self.compute_split_point()
        self.forward_algorithms_large = []
        for i in reversed(range(self.split_point, self.out_pyr.levels[1])):
            self.forward_algorithms_large.append(self.gpu.manager.algorithm(
                [*self.forward_input_buffers, *self.forward_output_buffers, *self.internal_buffers],
                spirv=self.forward_shader_large,
                workgroup=[int(np.ceil(self.out_pyr.levels[i*3+2+1]*self.out_pyr.levels[i*3+2+2]*self.out_pyr.channels/(self.gpu.max_workgroup_invocations*4))), 0,
                           0],
                spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
            ))

        j = self.split_point-1
        self.forward_algorithm_small = self.gpu.manager.algorithm(
            [*self.forward_input_buffers, *self.forward_output_buffers, *self.internal_buffers],
            spirv=self.forward_shader_small,
            workgroup=[int(np.ceil(self.out_pyr.levels[j*3+2+1]*self.out_pyr.levels[j*3+2+2]*self.out_pyr.channels/(self.gpu.max_workgroup_invocations*4))), 0, 0],
            spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
        )

        self.forward_input_buffers.extend(self.internal_buffers)  # to work with regular runners

    def compute_split_point(self):
        for l in range(2, len(self.out_pyr.levels), 3):
            start, width, height = int(self.out_pyr.levels[l]), int(self.out_pyr.levels[l + 1]), int(self.out_pyr.levels[l + 2]),
            end_idx = int(start * self.out_pyr.channels + (width * height * self.out_pyr.channels))

            if end_idx > self.gpu.max_workgroup_invocations:
                return int((l-2)/3)

        return self.out_pyr.levels[1]  # the actual number of levels is stored here

    def forward_ops(self):
        ops = [kp.OpAlgoDispatch(self.clear_algorithm)]
        j=0
        for _ in range(self.split_point, self.out_pyr.levels[1]):
            ops.append(kp.OpAlgoDispatch(self.forward_algorithms_large[j]))
            ops.append(kp.OpAlgoDispatch(self.count_algorithm))
            j+=1
        ops.append(kp.OpAlgoDispatch(self.forward_algorithm_small))
        return ops

    def backward_ops(self):
        return []

    def optim_ops(self):
        return []



def display_basic_forward_sequence(pyr, seq, image, display=None):
    if display is None:
        display = DirectDisplay()

    pyr.image.set(image)
    seq.eval()
    out_images = pyr.out_pyr.get()

    for i, o in enumerate(out_images):
        display.imshow(f'output {i}', o)
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


def display_pyramid(input_image_file):
    from sili.core.runners import get_forward_sequence

    gpu = GPUManager()
    im = cv2.imread(input_image_file).astype(np.uint8)
    pyr = ToImagePyramid(gpu, im)  # im used to set width, height, and channels
    seq = get_forward_sequence([pyr])
    display_basic_forward_sequence(pyr, seq, im)


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
    #display_pyramid("../../../test/files/test_ai_pls_ignore.png")

    # generate_pyramid_file("../../../test/files/test_ai_pls_ignore.png",
    #                      "../../../test/files/test_ai_pyr_pls_ignore.pyr")

    display_pyramid_from_camera(0)