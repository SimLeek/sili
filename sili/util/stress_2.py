#!/usr/bin/env python3
"""CPU and GPU stress test with precise duty cycle control.

Uses multiprocessing for CPU stress and Kompute for GPU stress on N GPUs.
Intensity (0-1) sets duty cycle: intensity = compute_time / (compute_time + sleep_time).
"""

import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple
from sili.buffers.image import ndarrayBuffer

import kp
import numpy as np

from sili.core.devices.gpu import GPUManager


def get_shader(filename: str) -> bytes:
    """Compile GLSL shader to SPIR-V using glslc.

    Args:
        filename: Path to GLSL file (.comp or .glsl).

    Returns:
        SPIR-V binary as bytes.
    """
    if filename.endswith(".glsl") or filename.endswith(".comp"):
        spv_filename = filename[:-5] + ".spv"
        if not os.path.exists(spv_filename):
            try:
                subprocess.run(
                    ["glslc", filename, "-o", spv_filename],
                    check=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print("glslc command failed with output:")
                print(e.stdout)
                print(e.stderr, file=sys.stderr)
                raise e
        with open(spv_filename, "rb") as f:
            return f.read()
    raise ValueError("Invalid file extension. Filename must end with .glsl or .comp")


def cpu_stress(intensity: float, stop_event: mp.Event) -> None:
    """Run CPU stress test with given intensity until stop event is set.

    Args:
        intensity: Duty cycle (0-1), where 1 is max stress.
        stop_event: Event to signal process termination.
    """
    if intensity<=0:
        return
    matrix_size = 1000
    a = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size, matrix_size)

    while not stop_event.is_set():
        start_time = time.time()
        _ = np.dot(a, b)  # Matrix multiplication
        compute_time = time.time() - start_time

        if 0 < intensity < 1:
            sleep_time = compute_time * (1 - intensity) / intensity
            time.sleep(sleep_time)


def gpu_stress(
    intensity: float,
    stop_event: mp.Event,
    device_index: int,
    matrix_size: int = 1024,
    num_loops: int = 10,
) -> None:
    """Run GPU stress test on specified device with given intensity.

    Uses two compute shaders: random init and matrix multiplication.

    Args:
        intensity: Duty cycle (0-1), where 1 is max stress.
        stop_event: Event to signal process termination.
        device_index: GPU device index for kp.Manager.
        matrix_size: Size of square matrices (default: 1024).
        num_loops: Number of matrix multiplications per dispatch (default: 10).
    """
    try:
        mgr = kp.Manager(device_index)
    except RuntimeError as e:
        print(f"Failed to create manager for device {device_index}: {e}")
        return
    if intensity<=0:
        return
    work_group_size = 16
    buffer_size = matrix_size * matrix_size  # float32 elements

    # Reserve GPU buffers
    gpu = GPUManager(mgr)
    buffer_a = ndarrayBuffer(gpu, np.zeros(buffer_size), type=np.float32)
    buffer_b = ndarrayBuffer(gpu, np.zeros(buffer_size), type=np.float32)
    buffer_c = ndarrayBuffer(gpu, np.zeros(buffer_size), type=np.float32)

    # Write GLSL shaders to temporary files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".comp", delete=False, dir=os.path.dirname(__file__)
    ) as random_file:
        random_file.write(
            f"""
            #version 450
            layout (local_size_x = {work_group_size}, local_size_y = {work_group_size}) in;

            layout (std430, binding = 0) buffer Matrix {{ float data[]; }} matrix;

            float random(vec2 co) {{
                return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
            }}

            void main() {{
                uint i = gl_GlobalInvocationID.x;
                uint j = gl_GlobalInvocationID.y;
                if (i >= {matrix_size} || j >= {matrix_size}) return;
                matrix.data[i * {matrix_size} + j] = random(vec2(i, j));
            }}
            """
        )
        random_file.flush()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".comp", delete=False, dir=os.path.dirname(__file__)
    ) as compute_file:
        compute_file.write(
            f"""
            #version 450
            layout (local_size_x = {work_group_size}, local_size_y = {work_group_size}) in;

            layout (std430, binding = 0) buffer MatrixA {{ float data[]; }} matrix_a;
            layout (std430, binding = 1) buffer MatrixB {{ float data[]; }} matrix_b;
            layout (std430, binding = 2) buffer MatrixC {{ float data[]; }} matrix_c;

            void main() {{
                uint i = gl_GlobalInvocationID.x;
                uint j = gl_GlobalInvocationID.y;
                if (i >= {matrix_size} || j >= {matrix_size}) return;

                for (int l = 0; l < {num_loops}; l++) {{
                    float sum = 0.0;
                    for (uint k = 0; k < {matrix_size}; k++) {{
                        sum += matrix_a.data[i * {matrix_size} + k] * matrix_b.data[k * {matrix_size} + j];
                    }}
                    matrix_c.data[i * {matrix_size} + j] = sum;
                }}
            }}
            """
        )
        compute_file.flush()

    # Compile shaders
    random_spirv = get_shader(random_file.name)
    compute_spirv = get_shader(compute_file.name)

    # Clean up temporary files
    os.unlink(random_file.name)
    if os.path.exists(random_file.name[:-5] + ".spv"):
        os.unlink(random_file.name[:-5] + ".spv")
    os.unlink(compute_file.name)
    if os.path.exists(compute_file.name[:-5] + ".spv"):
        os.unlink(compute_file.name[:-5] + ".spv")

    # Create algorithms
    random_algo = mgr.algorithm(
        [buffer_a.buffer], random_spirv, workgroup=[matrix_size // work_group_size, matrix_size // work_group_size, 1]
    )
    random_algo_2 = mgr.algorithm(
        [buffer_b.buffer], random_spirv, workgroup=[matrix_size // work_group_size, matrix_size // work_group_size, 1]
    )
    compute_algo = mgr.algorithm(
        [buffer_a.buffer, buffer_b.buffer, buffer_c.buffer],
        compute_spirv,
        workgroup=[matrix_size // work_group_size, matrix_size // work_group_size, 1],
    )

    # Initialize matrices A and B
    seq = mgr.sequence()
    seq.record(kp.OpAlgoDispatch(random_algo))
    seq.record(kp.OpAlgoDispatch(random_algo_2))
    seq.eval()

    # Matrix multiplication loop
    seq = mgr.sequence()
    seq.record(kp.OpAlgoDispatch(compute_algo))

    while not stop_event.is_set():
        start_time = time.time()
        seq.eval()
        compute_time = time.time() - start_time
        #print(f"compute_time: {compute_time}")

        if 0 < intensity < 1:
            sleep_time = compute_time * (1 - intensity) / intensity
            time.sleep(sleep_time)
            #print(f"sleep_time: {sleep_time}")
            print(f"hz: {1.0/(compute_time+sleep_time)}")


def run_stress_test(
    cpu_intensity: float,
    gpu_intensity: float,
    duration: float,
    matrix_size: int = 1024,
    gpu_num_loops: int = 10,
) -> Tuple[mp.Process, List[mp.Process]]:
    """Start CPU and GPU stress tests with given intensities for specified duration.

    Args:
        cpu_intensity: CPU duty cycle (0-1).
        gpu_intensity: GPU duty cycle (0-1).
        duration: Test duration in seconds.
        matrix_size: Size of square matrices for GPU (default: 1024).
        gpu_num_loops: Number of matrix multiplications per GPU dispatch (default: 10).

    Returns:
        Tuple of CPU process and list of GPU process objects.
    """
    stop_event = mp.Event()
    gpu_processes = []

    # Detect available GPUs
    device_index = 0
    while True:
        #if device_index==0:
        #    device_index += 1
        #    continue
        try:
            kp.Manager(device_index)
            gpu_process = mp.Process(
                target=gpu_stress,
                args=(gpu_intensity, stop_event, device_index, matrix_size, gpu_num_loops),
            )
            gpu_processes.append(gpu_process)
            device_index += 1
        except RuntimeError:
            break

    cpu_process = mp.Process(target=cpu_stress, args=(cpu_intensity, stop_event))

    cpu_process.start()
    for gpu_process in gpu_processes:
        gpu_process.start()

    time.sleep(duration)
    stop_event.set()

    cpu_process.join()
    for gpu_process in gpu_processes:
        gpu_process.join()

    return cpu_process, gpu_processes


if __name__ == "__main__":
    # Example usage
    CPU_INTENSITY = 1.0  # 80% CPU stress
    GPU_INTENSITY = 1.0  # 50% GPU stress
    DURATION = 60.0  # Run for 10 seconds
    MATRIX_SIZE = 1024  # GPU matrix size
    GPU_NUM_LOOPS = 1  # Repeat matrix mul 10 times per dispatch

    run_stress_test(CPU_INTENSITY, GPU_INTENSITY, DURATION, MATRIX_SIZE, GPU_NUM_LOOPS)
    # gpu 0 ->55c, gpu 1 -> 45c, cpu -> 75