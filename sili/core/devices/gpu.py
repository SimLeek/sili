import os
import subprocess
import sys
import numpy as np
import kp


class GPUManager(object):
    """Manages a GPU device."""

    def __init__(self, manager: kp.Manager = None):
        if manager is None:
            self.manager = kp.Manager()
        elif isinstance(manager, int):
            self.manager = kp.Manager(manager)
        else:
            self.manager = manager

        # save these important variables so we don't have to ping vulkan and the device constantly:
        self.max_workgroup_invocations = self.manager.get_device_properties()['max_work_group_invocations']
        # todo: get this through kompute or vulkan: https://github.com/KomputeProject/kompute/issues/360
        # this variable is specifically necessary for reductions ops often used in sparse shaders:
        self.maxComputeSharedMemorySize = 49152

    def buffer(self, data, memory_type=kp.MemoryTypes.device):
        """Returns an SSBO buffer. (Try using np.float32 or np.uint8 types with type)"""
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expected np.ndarray for buffer data, got {type(data)}")
        return self.manager.tensor(
            data=data,
            memory_type=memory_type
        )

    def image(self, data, width, height, num_channels, memory_type=kp.MemoryTypes.device):
        """Returns an image buffer. (Data should be a flattened np.ndarray, supports uint8 or float32)"""
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expected np.ndarray for image data, got {type(data)}")
        # Use image_t to support multiple data types
        return self.manager.image_t(
            data=data,
            width=width,
            height=height,
            num_channels=num_channels,
            memory_type=memory_type
        )

def get_shader(filename):
    if filename.endswith('.glsl') or filename.endswith('.comp'):
        spv_filename = filename[:-5] + '.spv'
        if not os.path.exists(spv_filename):
            try:
                subprocess.run(["glslc", filename, "-o", spv_filename],
                               check=True,
                               stdout=sys.stdout,
                               stderr=sys.stderr, text=True)
            except subprocess.CalledProcessError as e:
                print("glslc command failed with output:")
                print(e.stdout)
                print(e.stderr, file=sys.stderr)
                raise e

        with open(spv_filename, 'rb') as f:
            shader = f.read()
    else:
        raise ValueError("Invalid file extension. Filename must end with .glsl or .comp")

    return shader
