#ifndef __SCAN_HPP__
#define __SCAN_HPP__

#include "unique_vector.hpp"
#ifdef __clang__
#include <numeric>
#endif

/**
 * Computes a cumulative sum of the sizes of inner vectors using OpenMP.
 *
 * @tparam T The type of elements in the inner vectors.
 * @param vec_of_vec A vector of vectors of type T.
 * @return A vector of size_t with cumulative sizes, one element larger than the input.
 */
template <class T> void fullScanSizes(const sili::unique_vector<sili::unique_vector<T>> &vec_of_vec, sili::unique_vector<size_t>&fullScan, int&& scan_a=0) {
    
#ifdef __clang__ // OMP scan is broken in clang and may crash it: https://github.com/llvm/llvm-project/issues/87466
    std::inclusive_scan(
        vec_of_vec.begin(),
        vec_of_vec.end(),
        fullScan.begin() + 1,
        [](const size_t &cum_sum, const sili::unique_vector<T> &vec) { return cum_sum + vec.size(); },
        0);
#else

#pragma omp for simd reduction(inscan, + : scan_a)
    for (int i = 0; i < vec_of_vec.size()+1; i++) {
        fullScan[i] = scan_a;
        #pragma omp scan exclusive(scan_a)
        {
            if(i<vec_of_vec.size()){
                scan_a += vec_of_vec[i].size();
            }else{
                scan_a += vec_of_vec[i-1].size();
            }
        }
        
    }
# pragma omp barrier
#endif
}

/**
 * Computes cumulative sums of the sizes of inner inner vectors using OpenMP.
 *
 * @tparam T The type of elements in the inner inner vectors.
 * @param vec_of_vec_of_vec A vector of vectors of type T.
 * @return A vector of size_t with cumulative sizes, one element larger than the input.
 */
template <class T>
void fullScanSizes2(const sili::unique_vector<sili::unique_vector<sili::unique_vector<T>>> &vec_of_vec_of_vec, sili::unique_vector<sili::unique_vector<size_t>>& fullScans) {
    for (int i = 0; i < vec_of_vec_of_vec.size(); i++) {
        fullScanSizes(vec_of_vec_of_vec[i], fullScans[i]);
    }
}

#endif