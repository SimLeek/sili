#include "../headers/cache_info.h"

#include <cstring>
#include <iostream>

// Platform-specific includes
#ifdef _WIN32
#include <vector>
#include <windows.h>
#else
#include <fstream>
#include <string>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

size_t get_cache_line_size() {
    size_t cache_line_size = 0;

#ifdef _WIN32
    DWORD buffer_size = 0;
    GetLogicalProcessorInformation(nullptr, &buffer_size);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(buffer_size /
                                                             sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    GetLogicalProcessorInformation(buffer.data(), &buffer_size);

    for (const auto &info : buffer) {
        if (info.Relationship == RelationCache && info.Cache.Level == 1) {
            cache_line_size = info.Cache.LineSize; // Use LineSize instead of Size for cache line size
            break;
        }
    }
#elif defined(__linux__)
    std::ifstream cache_info("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    if (cache_info.is_open()) {
        std::string line;
        if (std::getline(cache_info, line)) {
            cache_line_size = std::stoul(line); // Read the cache line size in bytes directly
        }
        cache_info.close();
    } else {
        std::cerr << "Error: Unable to read cache line size from "
                     "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
                  << std::endl;
    }
#elif defined(__APPLE__)
    size_t size = sizeof(cache_line_size);
    if (sysctlbyname("hw.cachelinesize", &cache_line_size, &size, nullptr, 0) != 0) {
        std::cerr << "Error: Unable to get cache line size on macOS." << std::endl;
    }
#endif

    return cache_line_size;
}

size_t get_l1_cache_size() {
    size_t cache_size = 0;

#ifdef _WIN32
    // Windows implementation
    DWORD buffer_size = 0;
    GetLogicalProcessorInformation(nullptr, &buffer_size);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(buffer_size /
                                                             sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    GetLogicalProcessorInformation(buffer.data(), &buffer_size);

    for (const auto &info : buffer) {
        if (info.Relationship == RelationCache && info.Cache.Level == 1) {
            cache_size = info.Cache.Size;
            break;
        }
    }
#elif defined(__linux__)
    // Linux implementation
    std::ifstream cache_info("/sys/devices/system/cpu/cpu0/cache/index0/size");
    if (cache_info.is_open()) {
        std::string line;
        if (std::getline(cache_info, line)) {
            if (line.back() == 'K') {
                line.pop_back();
                cache_size = std::stoul(line) * 1024; // Convert KB to bytes
            }
        }
        cache_info.close();
    }
#elif defined(__APPLE__)
    // macOS implementation
    size_t size = sizeof(cache_size);
    if (sysctlbyname("hw.l1dcachesize", &cache_size, &size, nullptr, 0) != 0) {
        std::cerr << "Error: Unable to get L1 cache size on macOS." << std::endl;
    }
#endif

    return cache_size;
}

// Function to compute the optimal unrolled array size
size_t compute_optimal_unrolled_size(size_t cache_size, size_t element_size, double fraction) {
    // Use a fraction of the cache size to leave room for other operations and data
    size_t optimal_cache_usage = static_cast<size_t>(cache_size * fraction);
    size_t optimal_size = optimal_cache_usage / element_size;
    return optimal_size;
}