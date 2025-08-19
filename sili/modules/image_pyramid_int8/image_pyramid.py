import subprocess

import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os
import multiprocessing as mp


#from sili.core.runners import get_initial_sequence
from sili.core.util import find_good_dimension_sizes
from sili.buffers.pyramid import ImagePyramidBuffer, generate_downscaled_rectangles, pack_rectangles_minimal, \
    PackingResult
from sili.buffers.image import ImageBuffer
from sili.core.devices.gpu import GPUManager, get_shader
from sili.core.module import Module
from typing import Union
import cv2

file_path = os.path.dirname(os.path.abspath(__file__))

class ToImagePyramid(Module):
    def __init__(self, gpu: GPUManager, image: Union[ImageBuffer, np.ndarray], scale: float = np.sqrt(np.e)):
        assert scale > 1.0, "Scale must be large enough to divide the image"

        self.gpu = gpu
        self.forward_shader_clear = get_shader(file_path + os.sep + 'clear_pyramid.comp')
        self.forward_shader_large = get_shader(file_path + os.sep + 'pyramid_gen_large.comp')
        self.forward_shader_small = get_shader(file_path + os.sep + 'pyramid_gen_small.comp')

        image2 = np.zeros_like(image)
        if not isinstance(image, ImageBuffer):
            self.image = ImageBuffer(self.gpu, image, memory_type=kp.device)
        else:
            self.image = image
        '''packing_result, pyr_levels = pack_rectangles_minimal(
            self.image.width,
            self.image.height,
            scale,
            self.image.colors,
            pad_w=2,
            pad_h=2,
            show_progress=True
        )
       # packing_result = PackingResult(1874, 657, 17721, 1231218, 98.5, 1.4, 1213497, [(0, 0, 0, 1166, 657, None), (0, 1166, 0, 708, 399, None), (0, 1166, 399, 430, 242, None), (0, 1596, 399, 261, 147, None), (0, 1596, 546, 159, 89, None), (0, 1755, 546, 54, 97, None), (0, 1809, 546, 59, 33, None), (0, 1596, 635, 36, 20, None), (0, 1632, 635, 12, 22, None), (0, 1644, 643, 8, 14, None), (0, 1868, 399, 5, 9, None), (0, 1868, 408, 6, 3, None), (0, 1868, 411, 4, 3, None), (0, 1868, 414, 3, 3, None)] )
       # pyr_levels = [ 3, 14, 1868, 414, 3, 3, 1868, 411, 4, 3, 1868, 408, 6, 3, 1868, 399, 5, 9, 1644, 643, 8, 14, 1632, 635, 12, 22, 1596, 635, 36, 20, 1809, 546, 59, 33, 1755, 546, 54, 97, 1596, 546, 159, 89, 1596, 399, 261, 147, 1166, 399, 430, 242, 1166, 0, 708, 399, 0, 0, 1166, 657 ]
'''
        self.image2 = ImageBuffer(self.gpu, image2, memory_type=kp.device)

        # Compute 2D workgroup sizes
        workgroup_sizes = find_good_dimension_sizes(self.gpu.max_workgroup_invocations, 2)
        self.workgroup_x, self.workgroup_y = workgroup_sizes[0], workgroup_sizes[1]
        #rect_image = np.zeros((packing_result.height, packing_result.width, self.image.colors), dtype=np.uint8)
        #self.out_pyr = ImagePyramidBuffer(gpu, self.image, rect_image, pyr_levels, memory_type=kp.deviceAndHost)
        #max_lvl_h = self.out_pyr.levels[-2]
        #max_lvl_w = self.out_pyr.levels[-1]
        #lvl_count_info = np.ones( (int(np.ceil(max_lvl_w/self.workgroup_x)), int(np.ceil(max_lvl_h/self.workgroup_y))),dtype=np.int32)*int(self.out_pyr.levels[1]-1)
        #self.level_count_buffer = self.gpu.buffer(lvl_count_info, kp.device)  # start at last/base level
        #self.level_count_buffer_shape = lvl_count_info.shape

        # PIPELINE OBJECTS:
        #self.forward_input_buffers = [self.out_pyr.pyr_lvl_buffer, self.image.buffer]
        #self.forward_output_buffers = [self.out_pyr.rect_pack.buffer]
        #self.internal_buffers = [self.level_count_buffer]

        self.forward_input_buffers = [ self.image.buffer]
        self.forward_output_buffers = [self.image2.buffer]
        #self.internal_buffers = [self.level_count_buffer]

        '''self.clear_algorithm = self.gpu.manager.algorithm(
            [self.level_count_buffer],
            spirv=self.forward_shader_clear,
            workgroup=[int(np.ceil(lvl_count_info.size/self.gpu.max_workgroup_invocations)), 1, 1],
            spec_consts=np.asarray([self.gpu.max_workgroup_invocations,int(self.out_pyr.levels[1]-1), lvl_count_info.size], dtype=np.uint32).view(np.float32),
        )'''

        #self.split_point = self.compute_split_point()
        self.forward_algorithms_large = []
        pad_l, pad_t, pad_r, pad_b = 1, 1, 1, 1
        '''for i in reversed(range(self.split_point, self.out_pyr.levels[1])):  # split point to max level
            # rect defs are len 4, plue first two being c and num_rects, followed by w being at index 3 and h at index 4
            level_width = int(self.out_pyr.levels[i * 4 + 2 + 2])
            level_height = int(self.out_pyr.levels[i * 4 + 2 + 3])
            self.forward_algorithms_large.append(self.gpu.manager.algorithm(
                [*self.forward_input_buffers, *self.forward_output_buffers, *self.internal_buffers],
                spirv=self.forward_shader_large,
                workgroup=[int(np.ceil(level_width/self.workgroup_x)),int(np.ceil(level_height/self.workgroup_y)), 0],
                spec_consts=np.asarray([self.workgroup_x,self.workgroup_y,  pad_l, pad_t, pad_r, pad_b], dtype=np.uint32).view(np.float32),
            ))
            break'''
        self.forward_algorithms_large.append(self.gpu.manager.algorithm(
            [*self.forward_input_buffers, *self.forward_output_buffers],
            spirv=self.forward_shader_large,
            workgroup=[int(np.ceil(self.image.shape[0] / self.workgroup_x)), int(np.ceil(self.image.shape[1] / self.workgroup_y)), 0],
            spec_consts=np.asarray([self.workgroup_x, self.workgroup_y, pad_l, pad_t, pad_r, pad_b, self.image.height, self.image.width, self.image.actual_colors, self.image2.height, self.image2.width, self.image2.actual_colors],
                                   dtype=np.uint32).view(np.float32),
        ))

        #j = self.split_point-1
        #level_width = int(self.out_pyr.levels[j*4+2+3])
        #level_height = int(self.out_pyr.levels[j*4+2+4])
        '''self.forward_algorithm_small = self.gpu.manager.algorithm(
            [*self.forward_input_buffers, *self.forward_output_buffers, *self.internal_buffers],
            spirv=self.forward_shader_small,
            workgroup=[1, 1, 1],  # meaning of "small" here is that it fits in a single workgroup
            spec_consts=np.asarray([self.workgroup_x, self.workgroup_y, pad_l, pad_t, pad_r, pad_b], dtype=np.uint32).view(np.float32),
        )'''

        #self.forward_input_buffers.extend(self.internal_buffers)  # to work with regular runners
        #self.forward_output_buffers.extend(self.internal_buffers)  # so I can see it
        super().__init__()


    def compute_split_point(self):
        for l in range(2, len(self.out_pyr.levels), 4):
            x, y, w, h = (int(self.out_pyr.levels[l+i])for i in range(4))
            if w>self.workgroup_x or h>self.workgroup_y:
                return int((l-2)/4)

        return self.out_pyr.levels[1]  # the actual number of levels is stored here

    def forward_ops(self):
        #ops = [kp.OpAlgoDispatch(self.clear_algorithm)]
        ops=[]
        for alg in self.forward_algorithms_large:
            ops.append(kp.OpAlgoDispatch(alg))
        #ops.append(kp.OpAlgoDispatch(self.forward_algorithm_small))
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
    pyr = ToImagePyramid(gpu, im)

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
    pyr = ToImagePyramid(gpu, im)
    seq = get_forward_sequence([pyr])
    display_basic_forward_sequence(pyr, seq, im)

def display_pyramid_from_camera(camera):
    from sili.core.runners import get_forward_sequence
    from displayarray import read_updates, DirectDisplay, DirectRead
    import time

    #r = read_updates(camera, size=(9999, 9999))
    r = DirectRead(camera, request_size=(9999, 9999), fps_limit=float('inf'))#fps_limit=60)#, fps_limit=float("inf"))  # profile speed
    #r = read_updates(camera, size=(1280,960))
    #r = read_updates(camera, size=(-1, -1))

    #gpu = GPUManager(1)
    gpu = GPUManager(0)
    first = True
    pyr = None
    seq = None
    display = DirectDisplay(vsync=False)  #vsync=False)
    t=0
    for frame_dict in r:
        if not frame_dict:
            continue
        im = frame_dict[str(camera)].astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        #display.imshow(f'input', im)
        #display.update()
        #if display.window.is_closing:
        #    break
        #continue
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if first:
            pyr = ToImagePyramid(gpu, im)
            seq = get_forward_sequence([pyr])
            first = False
        pyr.image.set(im)
        t0 = time.time()
        seq.eval()
        #seq.eval_async()
        t1 = time.time()
        #seq.eval_await()
        #orig_image = pyr.image.get()
        out_image = pyr.image2.get()
        #out_image = pyr.out_pyr.get_rectpack()
        #orig_image = pyr.out_pyr.get_original()
        #count_io = pyr.level_count_buffer.data().reshape(pyr.level_count_buffer_shape)

        print(f"eval time:{t1 - t0}, fps:{1. / (t1 - t0)}")
        #for i, o in enumerate(out_images):
        #    display.imshow(f'output {i}', o)
        #print(count_io)
        #display.imshow(f'input', im)
        # NOTE!!!!!!!! kp.deviceAndHost will be 10x slower if you try to display them. Use either kp.device or kp.host
        #display.imshow(f'input', orig_image)
        display.imshow(f'output', out_image)
        #display.imshow(f'count_io', count_io.astype(np.float32)/12.0)
        #cv2.waitKey(1)
        #t+=1
        #if t==60:
        #    return
        display.update()
        if display.window.is_closing:
            break

if __name__ == '__main__':
    #display_pyramid_from_camera("../../../test/files/test_vid_480.mp4")
    #display_pyramid_from_camera("../../../test/files/drone_test_vid_1080.mp4")
    import cProfile, pstats


    class Stats(pstats.Stats):
        sort_arg_dict_default = {
            "calls": (((1, -1),), "call count"),
            "ncalls": (((1, -1),), "call count"),
            "cumtime": (((4, -1),), "cumulative time"),
            "cumulative": (((4, -1),), "cumulative time"),
            "file": (((6, 1),), "file name"),
            "filename": (((6, 1),), "file name"),
            "line": (((7, 1),), "line number"),
            "module": (((6, 1),), "file name"),
            "name": (((8, 1),), "function name"),
            "nfl": (((8, 1), (6, 1), (7, 1),), "name/file/line"),
            "pcalls": (((0, -1),), "primitive call count"),
            "stdname": (((9, 1),), "standard name"),
            "time": (((2, -1),), "internal time"),
            "tottime": (((2, -1),), "internal time"),
            "cumulativepercall": (((5, -1),), "cumulative time per call"),
            "totalpercall": (((3, -1),), "total time per call"),
        }

        def sort_stats(self, *field):
            if not field:
                self.fcn_list = 0
                return self
            if len(field) == 1 and isinstance(field[0], int):
                field = [{-1: "stdname",
                          0: "calls",
                          1: "time",
                          2: "cumulative"}[field[0]]]
            elif len(field) >= 2:
                for arg in field[1:]:
                    if type(arg) != type(field[0]):
                        raise TypeError("Can't have mixed argument type")

            sort_arg_defs = self.get_sort_arg_defs()

            sort_tuple = ()
            self.sort_type = ""
            connector = ""
            for word in field:
                if isinstance(word, pstats.SortKey):
                    word = word.value
                sort_tuple = sort_tuple + sort_arg_defs[word][0]
                self.sort_type += connector + sort_arg_defs[word][1]
                connector = ", "

            stats_list = []
            for func, (cc, nc, tt, ct, callers) in self.stats.items():
                if nc == 0:
                    npc = 0
                else:
                    npc = float(tt) / nc

                if cc == 0:
                    cpc = 0
                else:
                    cpc = float(ct) / cc

                stats_list.append((cc, nc, tt, npc, ct, cpc) + func +
                                  (pstats.func_std_string(func), func))

            stats_list.sort(key=pstats.cmp_to_key(pstats.TupleComp(sort_tuple).compare))

            self.fcn_list = fcn_list = []
            for tuple in stats_list:
                fcn_list.append(tuple[-1])
            return self


    cProfile.run("display_pyramid_from_camera('../../../test/files/drone_test_vid_1080.mp4')", 'prof_data.prof')
    p = Stats('prof_data.prof')
    p.sort_stats('cumtime').print_stats()
