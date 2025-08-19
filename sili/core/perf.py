import vulkan as vk
import json
import time
import os
import subprocess
import sys


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


def measure_launch_overhead(device, queue, cmd_pool, shader_code, global_bandwidth, trials=100):
    # Create pipeline
    shader_module = vk.vkCreateShaderModule(device, vk.VkShaderModuleCreateInfo(code=shader_code))
    pipeline_layout = vk.vkCreatePipelineLayout(device, vk.VkPipelineLayoutCreateInfo())
    pipeline = vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, [
        vk.VkComputePipelineCreateInfo(
            layout=pipeline_layout,
            stage=vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName="main"
            )
        )
    ])[0]

    # Create dummy buffers for minimal dispatch
    buffer_size = 256 * 4  # Minimal size for one workgroup
    src_buffer = vk.vkCreateBuffer(device, vk.VkBufferCreateInfo(
        usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        size=buffer_size
    ))
    dst_buffer = vk.vkCreateBuffer(device, vk.VkBufferCreateInfo(
        usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        size=buffer_size
    ))
    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(vk.vkGetDevicePhysicalDevice(device))
    mem_reqs = vk.vkGetBufferMemoryRequirements(device, src_buffer)
    mem_type_index = next(i for i in range(mem_props.memoryTypeCount)
                          if (mem_reqs.memoryTypeBits & (1 << i)) and
                          (mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    mem = vk.vkAllocateMemory(device, vk.VkMemoryAllocateInfo(
        allocationSize=buffer_size * 2,
        memoryTypeIndex=mem_type_index
    ))
    vk.vkBindBufferMemory(device, src_buffer, mem, 0)
    vk.vkBindBufferMemory(device, dst_buffer, mem, buffer_size)

    # Create descriptor set
    descriptor_pool = vk.vkCreateDescriptorPool(device, vk.VkDescriptorPoolCreateInfo(
        maxSets=1,
        poolSizes=[vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=2)]
    ))
    descriptor_set_layout = vk.vkCreateDescriptorSetLayout(device, vk.VkDescriptorSetLayoutCreateInfo(
        bindings=[
            vk.VkDescriptorSetLayoutBinding(binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(binding=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
        ]
    ))
    descriptor_set = vk.vkAllocateDescriptorSets(device, vk.VkDescriptorSetAllocateInfo(
        descriptorPool=descriptor_pool,
        descriptorSetLayouts=[descriptor_set_layout]
    ))[0]
    vk.vkUpdateDescriptorSets(device, [
        vk.VkWriteDescriptorSet(dstSet=descriptor_set, dstBinding=0,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                bufferInfo=[vk.VkDescriptorBufferInfo(buffer=src_buffer, offset=0, range=buffer_size)]),
        vk.VkWriteDescriptorSet(dstSet=descriptor_set, dstBinding=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                bufferInfo=[vk.VkDescriptorBufferInfo(buffer=dst_buffer, offset=0, range=buffer_size)])
    ])

    # Measure launch time
    cmd_buffer = vk.vkAllocateCommandBuffers(device, vk.VkCommandBufferAllocateInfo(
        commandPool=cmd_pool,
        level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1
    ))[0]
    start = time.perf_counter()
    for _ in range(trials):
        vk.vkBeginCommandBuffer(cmd_buffer, vk.VkCommandBufferBeginInfo())
        vk.vkCmdBindPipeline(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vk.vkCmdBindDescriptorSets(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, [descriptor_set],
                                   [])
        vk.vkCmdDispatch(cmd_buffer, 1, 1, 1)
        vk.vkEndCommandBuffer(cmd_buffer)
        vk.vkQueueSubmit(queue, [vk.VkSubmitInfo(commandBuffers=[cmd_buffer])], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(queue)
    elapsed = (time.perf_counter() - start) / trials

    # Add SPIR-V read time
    spirv_size = len(shader_code)
    read_time = spirv_size / global_bandwidth
    launch_overhead = (elapsed + read_time) * 1e6  # µs

    # Cleanup
    vk.vkDestroyDescriptorPool(device, descriptor_pool, None)
    vk.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, None)
    vk.vkDestroyBuffer(device, src_buffer, None)
    vk.vkDestroyBuffer(device, dst_buffer, None)
    vk.vkFreeMemory(device, mem, None)
    vk.vkDestroyShaderModule(device, shader_module, None)
    vk.vkDestroyPipelineLayout(device, pipeline_layout, None)
    vk.vkDestroyPipeline(device, pipeline, None)
    vk.vkFreeCommandBuffers(device, cmd_pool, [cmd_buffer])
    return launch_overhead


def measure_memory_bandwidth(device, queue, cmd_pool, size=40e6, trials=100):
    shader_code = get_shader("memory_copy.comp")
    shader_module = vk.vkCreateShaderModule(device, vk.VkShaderModuleCreateInfo(code=shader_code))
    pipeline_layout = vk.vkCreatePipelineLayout(device, vk.VkPipelineLayoutCreateInfo())
    pipeline = vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, [
        vk.VkComputePipelineCreateInfo(
            layout=pipeline_layout,
            stage=vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName="main"
            )
        )
    ])[0]

    buffer_size = int(size)
    src_buffer = vk.vkCreateBuffer(device, vk.VkBufferCreateInfo(
        usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        size=buffer_size
    ))
    dst_buffer = vk.vkCreateBuffer(device, vk.VkBufferCreateInfo(
        usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        size=buffer_size
    ))
    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(vk.vkGetDevicePhysicalDevice(device))
    mem_reqs = vk.vkGetBufferMemoryRequirements(device, src_buffer)
    mem_type_index = next(i for i in range(mem_props.memoryTypeCount)
                          if (mem_reqs.memoryTypeBits & (1 << i)) and
                          (mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    mem = vk.vkAllocateMemory(device, vk.VkMemoryAllocateInfo(
        allocationSize=buffer_size * 2,
        memoryTypeIndex=mem_type_index
    ))
    vk.vkBindBufferMemory(device, src_buffer, mem, 0)
    vk.vkBindBufferMemory(device, dst_buffer, mem, buffer_size)

    descriptor_pool = vk.vkCreateDescriptorPool(device, vk.VkDescriptorPoolCreateInfo(
        maxSets=1,
        poolSizes=[vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=2)]
    ))
    descriptor_set_layout = vk.vkCreateDescriptorSetLayout(device, vk.VkDescriptorSetLayoutCreateInfo(
        bindings=[
            vk.VkDescriptorSetLayoutBinding(binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT),
            vk.VkDescriptorSetLayoutBinding(binding=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
        ]
    ))
    descriptor_set = vk.vkAllocateDescriptorSets(device, vk.VkDescriptorSetAllocateInfo(
        descriptorPool=descriptor_pool,
        descriptorSetLayouts=[descriptor_set_layout]
    ))[0]
    vk.vkUpdateDescriptorSets(device, [
        vk.VkWriteDescriptorSet(dstSet=descriptor_set, dstBinding=0,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                bufferInfo=[vk.VkDescriptorBufferInfo(buffer=src_buffer, offset=0, range=buffer_size)]),
        vk.VkWriteDescriptorSet(dstSet=descriptor_set, dstBinding=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                bufferInfo=[vk.VkDescriptorBufferInfo(buffer=dst_buffer, offset=0, range=buffer_size)])
    ])

    cmd_buffer = vk.vkAllocateCommandBuffers(device, vk.VkCommandBufferAllocateInfo(
        commandPool=cmd_pool,
        level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1
    ))[0]
    start = time.perf_counter()
    for _ in range(trials):
        vk.vkBeginCommandBuffer(cmd_buffer, vk.VkCommandBufferBeginInfo())
        vk.vkCmdBindPipeline(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vk.vkCmdBindDescriptorSets(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, [descriptor_set],
                                   [])
        vk.vkCmdDispatch(cmd_buffer, buffer_size // (256 * 4), 1, 1)
        vk.vkEndCommandBuffer(cmd_buffer)
        vk.vkQueueSubmit(queue, [vk.VkSubmitInfo(commandBuffers=[cmd_buffer])], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(queue)
    elapsed = (time.perf_counter() - start) / trials
    global_bandwidth = size / elapsed

    vk.vkDestroyDescriptorPool(device, descriptor_pool, None)
    vk.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, None)
    vk.vkDestroyBuffer(device, src_buffer, None)
    vk.vkDestroyBuffer(device, dst_buffer, None)
    vk.vkFreeMemory(device, mem, None)
    vk.vkDestroyShaderModule(device, shader_module, None)
    vk.vkDestroyPipelineLayout(device, pipeline_layout, None)
    vk.vkDestroyPipeline(device, pipeline, None)
    vk.vkFreeCommandBuffers(device, cmd_pool, [cmd_buffer])
    return global_bandwidth, shader_code


def get_device_performance():
    instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo())
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    if not physical_devices:
        raise RuntimeError("No Vulkan-capable devices found")

    performance_data = {}
    for phys_dev in physical_devices:
        props = vk.vkGetPhysicalDeviceProperties(phys_dev)
        device_name = props.deviceName.decode().replace(" ", "_")

        queue_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev)
        compute_queue_family = next((i for i, q in enumerate(queue_props) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT),
                                    None)
        if compute_queue_family is None:
            continue

        device = vk.vkCreateDevice(phys_dev, vk.VkDeviceCreateInfo(
            queueCreateInfos=[vk.VkDeviceQueueCreateInfo(queueFamilyIndex=compute_queue_family, queueCount=1)]
        ))
        queue = vk.vkGetDeviceQueue(device, compute_queue_family, 0)
        cmd_pool = vk.vkCreateCommandPool(device, vk.VkCommandPoolCreateInfo(queueFamilyIndex=compute_queue_family))

        global_bandwidth, shader_code = measure_memory_bandwidth(device, queue, cmd_pool)
        launch_overhead = measure_launch_overhead(device, queue, cmd_pool, shader_code, global_bandwidth)
        shared_bandwidth = 10e12
        cache_hit_rate = {'large': 0.8, 'small': 1.0}

        performance_data[device_name] = {
            "global_bandwidth": global_bandwidth,
            "shared_bandwidth": shared_bandwidth,
            "launch_overhead": launch_overhead,
            "cache_hit_rate": cache_hit_rate
        }

        with open(f"{device_name}.json", "w") as f:
            json.dump(performance_data[device_name], f, indent=4)

        vk.vkDestroyCommandPool(device, cmd_pool, None)
        vk.vkDestroyDevice(device, None)

    vk.vkDestroyInstance(instance, None)
    return performance_data


if __name__ == "__main__":
    try:
        performance_data = get_device_performance()
        print("Performance analysis saved to JSON files:")
        for device_name in performance_data:
            print(f"- {device_name}.json")
    except Exception as e:
        print(f"Error: {e}")