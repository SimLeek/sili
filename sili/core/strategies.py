import json
import math
import vulkan as vk
from sili.core.util import find_good_dimension_sizes
from math import floor, sqrt
import sys

"""
Note: we'll also want to keep the input image in its own buffer, 
and use rectpack for all the downscaling. 
This lets us not copy the base image to the rectpack, reducing total copies by half. 
Then, for later edge detection and other work, we just need a single if statement to 
determine whether we're working on the base image or the rectpack image.
"""


def generate_downscaled_rectangles(start_width, start_height, pad_w=2, pad_h=2):
    rectangles = []
    w, h = int(start_width), int(start_height)
    while w > 1 or h > 1:
        rectangles.append((w + pad_w, h + pad_h))
        w = int(max(floor(w / sqrt(2)), 1))
        h = int(max(floor(h / sqrt(2)), 1))
    return rectangles


def load_performance_data(device_name):
    try:
        with open(f"{device_name.replace(' ', '_')}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: JSON for {device_name} not found, using fallback values")
        return {
            "global_bandwidth": 500e9,
            "shared_bandwidth": 10e12,
            "launch_overhead": 20e-6,
            "cache_hit_rate": {"large": 0.8, "small": 1.0}
        }


def get_device_properties(device_index=0):
    instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo())
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    if not physical_devices:
        raise RuntimeError("No Vulkan-capable devices found")
    if device_index >= len(physical_devices):
        raise ValueError(f"Device index {device_index} out of range; only {len(physical_devices)} devices available")
    phys_dev = physical_devices[device_index]
    props = vk.vkGetPhysicalDeviceProperties(phys_dev)
    device_name = props.deviceName.decode()
    limits = props.limits

    # Try to query cache size with VK_EXT_memory_budget
    cache_size = 40e6  # Fallback
    try:
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties2(phys_dev, vk.VkPhysicalDeviceMemoryProperties2())
        for heap in mem_props.memoryProperties.memoryHeaps:
            if heap.flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
                cache_size = max(cache_size, heap.size // 4)  # Rough estimate
    except AttributeError:
        print("Warning: VK_EXT_memory_budget not supported, using fallback cache size")

    vk.vkDestroyInstance(instance, None)
    return {
        "device_name": device_name,
        "max_workgroup_size": limits.maxComputeWorkGroupInvocations,
        "max_workgroup_count": limits.maxComputeWorkGroupCount,
        "shared_memory_size": limits.maxComputeSharedMemorySize,
        "cache_threshold": cache_size
    }


def estimate_actual_kernel_overhead(device, queue, cmd_pool, shader_file, global_bandwidth):
    # Placeholder for profiling actual downscaling kernel
    shader_code = get_shader(shader_file)
    # Similar to measure_launch_overhead in performance_analysis.py
    # Implement when actual kernel is available
    return 20e-6 * 1e6  # µs, fallback until actual kernel is provided


def individual_launch_time(level_k, levels, perf_data, device_props, batch_padding):
    w_k, h_k = levels[level_k - 1][0] - batch_padding, levels[level_k - 1][1] - batch_padding
    w_km1, h_km1 = (levels[level_k - 1][0] - batch_padding,
                    levels[level_k - 1][1] - batch_padding) if level_k == 1 else (
        levels[level_k - 2][0] - batch_padding, levels[level_k - 2][1] - batch_padding)
    read_bytes = w_km1 * h_km1 * 4
    shared_bytes = w_k * h_k * 4
    write_bytes = w_k * h_k * 4
    cache = perf_data['cache_hit_rate']['small'] if read_bytes < device_props['cache_threshold'] else \
    perf_data['cache_hit_rate']['large']
    time = (read_bytes / (perf_data['global_bandwidth'] * cache) +
            shared_bytes / perf_data['shared_bandwidth'] +
            write_bytes / (perf_data['global_bandwidth'] * cache) +
            perf_data['launch_overhead']) * 1e6  # µs
    return time


def batched_launch_time(start_level, num_levels, levels, perf_data, device_props, batch_size, batch_padding):
    w_m, h_m = levels[start_level - 1][0] - batch_padding, levels[start_level - 1][1] - batch_padding
    dims = find_good_dimension_sizes(device_props['max_workgroup_size'], 2)
    workgroup_size_x, workgroup_size_y = dims[0], dims[1]
    workgroups = math.ceil(w_m / (workgroup_size_x - batch_padding)) * math.ceil(
        h_m / (workgroup_size_y - batch_padding))
    read_bytes = workgroups * workgroup_size_x * workgroup_size_y * 4
    write_bytes = sum((levels[k - 1][0] - batch_padding) * (levels[k - 1][1] - batch_padding) * 4 for k in
                      range(start_level, min(start_level + batch_size, len(levels) + 1)))
    shared_pixels = sum((levels[k - 1][0] - batch_padding) * (levels[k - 1][1] - batch_padding) for k in
                        range(start_level + 1, min(start_level + batch_size, len(levels) + 1)))
    shared_bytes = shared_pixels * 4
    cache = perf_data['cache_hit_rate']['small'] if read_bytes < device_props['cache_threshold'] else \
    perf_data['cache_hit_rate']['large']
    time = (read_bytes / (perf_data['global_bandwidth'] * cache) +
            write_bytes / (perf_data['global_bandwidth'] * cache) +
            shared_bytes / perf_data['shared_bandwidth'] +
            perf_data['launch_overhead']) * 1e6  # µs
    return time


def calculate_hybrid_runtimes(start_width, start_height, batch_size=6, batch_padding=2, device_index=0):
    device_props = get_device_properties(device_index)
    perf_data = load_performance_data(device_props['device_name'])
    levels = generate_downscaled_rectangles(start_width, start_height, batch_padding, batch_padding)
    num_levels = len(levels)

    dims = find_good_dimension_sizes(device_props['max_workgroup_size'], 2)
    if dims[0] * dims[1] * 4 > device_props['shared_memory_size']:
        raise ValueError("Workgroup size exceeds shared memory limit")

    hybrid_runtimes = {}
    for x in range(num_levels + 1):
        individual_time = sum(individual_launch_time(k, levels, perf_data, device_props, batch_padding) for k in
                              range(1, x + 1)) if x > 0 else 0
        remaining_levels = num_levels - x
        batched_time = 0
        start_level = x + 1
        while start_level <= num_levels:
            batch_size_actual = min(batch_size, num_levels - start_level + 1)
            batched_time += batched_launch_time(start_level, batch_size_actual, levels, perf_data, device_props,
                                                batch_size, batch_padding)
            start_level += batch_size_actual
        hybrid_runtimes[x] = individual_time + batched_time

    return hybrid_runtimes, num_levels, device_props['device_name']


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python pyramid_strategy.py <width> <height> [device_index] [batch_size] [batch_padding]")
        sys.exit(1)

    start_width = int(sys.argv[1])
    start_height = int(sys.argv[2])
    device_index = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    batch_size = int(sys.argv[4]) if len(sys.argv) >= 5 else 6
    batch_padding = int(sys.argv[5]) if len(sys.argv) >= 6 else 2

    try:
        runtimes, num_levels, device_name = calculate_hybrid_runtimes(start_width, start_height, batch_size,
                                                                      batch_padding, device_index)
        print(f"Hybrid Runtimes for {start_width}x{start_height}, {num_levels} levels on {device_name} (µs):")
        for x, time in runtimes.items():
            print(f"x={x}: {time:.2f} µs")
        optimal_x = min(runtimes, key=runtimes.get)
        print(f"Optimal: x={optimal_x}, {runtimes[optimal_x]:.2f} µs")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()