#ifndef __SCAN_H__
#define __SCAN_H__

#include <vector>
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
template <class T> std::vector<size_t> fullScanSizes(const std::vector<std::vector<T>> &vec_of_vec) {
    std::vector<size_t> fullScan(vec_of_vec.size() + 1);
    fullScan[0] = 0;

    int scan_a = 0;
#ifdef __clang__ // OMP scan is broken in clang and may crash it: https://github.com/llvm/llvm-project/issues/87466
    // code golf one-liner lol
    std::inclusive_scan(
        vec_of_vec.begin(),
        vec_of_vec.end(),
        fullScan.begin() + 1,
        [](const size_t &cum_sum, const std::vector<T> &vec) { return cum_sum + vec.size(); },
        0);
#else
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (int i = 0; i < vec.size(); i++) {
        fullScan[i + 1] = scan_a;

#pragma omp scan inclusive(scan_a)
        { scan_a += vec[i].size(); }
    }
#endif

    return fullScan;
}

/**
 * Computes cumulative sums of the sizes of inner inner vectors using OpenMP.
 *
 * @tparam T The type of elements in the inner inner vectors.
 * @param vec_of_vec_of_vec A vector of vectors of type T.
 * @return A vector of size_t with cumulative sizes, one element larger than the input.
 */
template <class T>
std::vector<std::vector<size_t>> fullScanSizes2(const std::vector<std::vector<std::vector<T>>> &vec_of_vec_of_vec) {
    std::vector<std::vector<size_t>> fullScans(vec_of_vec_of_vec.size());
    for (int i = 0; i < vec_of_vec_of_vec.size(); i++) {
        fullScans[i] = fullScanSizes(vec_of_vec_of_vec[i]);
    }
    return fullScans;
}

#endif