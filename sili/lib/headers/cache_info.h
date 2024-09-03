#ifndef CACHE_INFO_H
#define CACHE_INFO_H

#include <cstddef>  // for size_t

extern size_t get_cache_line_size();

extern size_t get_l1_cache_size();

extern size_t compute_optimal_unrolled_size(size_t cache_size, size_t element_size, double fraction = 0.5);

#endif