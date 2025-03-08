#ifndef _PARALLEL_HPP
#define _PARALLEL_HPP

#include "csr.hpp"
#include "sparse_struct.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <omp.h>
#include <vector>

/**
 * @brief Comparator class for less-than comparison between two types.
 *
 * @tparam A First type to compare.
 * @tparam B Second type to compare.
 */
template <class A, class B> class ComparatorLT {
  public:
    /**
     * @brief Compares two values using less-than.
     * @param a First value.
     * @param b Second value.
     * @return true if a < b, false otherwise.
     */
    bool operator()(A a, B b) { return (a < b); }
};

/**
 * @brief Comparator class for greater-than comparison between two types.
 *
 * @tparam A First type to compare.
 * @tparam B Second type to compare.
 */
template <class A, class B> class ComparatorGT {
  public:
    /**
     * @brief Compares two values using greater-than.
     * @param a First value.
     * @param b Second value.
     * @return true if a > b, false otherwise.
     */
    bool operator()(A a, B b) { return (a > b); }
};

/**
 * @brief Comparator for sorting indices based on values in a vector.
 *
 * @tparam T Type of elements in the vector.
 * @tparam Compare Comparator type for ordering elements.
 */
template <typename T, typename Compare> class PermutationComparator {
    const std::vector<T> &a; ///< Reference to the vector being sorted.
    Compare cmp;             ///< Comparator instance.
  public:
    /**
     * @brief Constructs the comparator with a vector and comparison function.
     * @param a Vector whose values determine the order.
     * @param cmp Comparator to use for ordering.
     */
    PermutationComparator(const std::vector<T> &a, Compare cmp) : a(a), cmp(cmp) {}

    /**
     * @brief Compares two indices based on their corresponding values.
     * @param i First index.
     * @param j Second index.
     * @return true if a[i] precedes a[j] according to cmp, false otherwise.
     */
    bool operator()(std::size_t i, std::size_t j) const { return cmp(a[i], a[j]); }
};

/**
 * @brief Recursive helper function for parallel merge sort using OpenMP.
 *
 * @tparam TYPE Type of elements in the vector.
 * @tparam COMPARE Comparator type, defaults to less-than.
 * @param v Vector to sort.
 * @param left Left boundary of the current segment.
 * @param right Right boundary of the current segment.
 */
template <class TYPE, class COMPARE = ComparatorLT<TYPE, TYPE>>
void _omp_merge_sort_recursive(std::vector<TYPE> &v, unsigned long left, unsigned long right) {
    COMPARE cmp;
    if (left < right) {
        if (right - left >= 32) {
            unsigned long mid = (left + right) / 2;
#pragma omp taskgroup
            {
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _omp_merge_sort_recursive(v, left, mid);
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _omp_merge_sort_recursive(v, mid + 1, right);
#pragma omp taskyield
            }
            std::inplace_merge(v.begin() + left, v.begin() + mid + 1, v.begin() + right + 1, cmp);
        } else {
            std::sort(v.begin() + left, v.begin() + right + 1, cmp);
        }
    }
}

/**
 * @brief Parallel merge sort using OpenMP.
 *
 * @tparam TYPE Type of elements in the vector.
 * @tparam COMPARE Comparator type, defaults to less-than.
 * @param v Vector to sort in-place.
 */
template <class TYPE, class COMPARE = ComparatorLT<TYPE, TYPE>> void omp_merge_sort(std::vector<TYPE> &v) {
#pragma omp parallel
#pragma omp single
    _omp_merge_sort_recursive(v, 0, v.size() - 1);
}

/**
 * @brief Computes a permutation that sorts a vector in parallel.
 *
 * @tparam TYPE Type of elements in the input vector.
 * @tparam COMPARE Comparator type for ordering.
 * @param a Vector to base the permutation on.
 * @param cmp Comparator instance.
 * @return Vector of indices representing the sorted permutation.
 */
template <typename TYPE, typename COMPARE>
std::vector<std::size_t> omp_merge_sort_permutation(const std::vector<TYPE> &a, COMPARE cmp) {
    std::vector<std::size_t> p(a.size());
    std::iota(p.begin(), p.end(), 0);
    PermutationComparator<TYPE, COMPARE> perm_cmp(a, cmp);
    omp_merge_sort<std::size_t, PermutationComparator<TYPE, COMPARE>>(p);
    return p;
}

/**
 * @brief Applies a permutation to a vector in parallel.
 *
 * @tparam T Type of elements in the vector.
 * @param p Permutation vector.
 * @param vec Vector to permute in-place.
 */
template <typename T> void omp_apply_permutation_parallel(const std::vector<std::size_t> &p, std::vector<T> &vec) {
    std::vector<T> temp(vec.size());
#pragma omp parallel for
    for (size_t i = 0; i < p.size(); ++i) {
        temp[i] = vec[p[i]];
    }
    vec = std::move(temp);
}

/**
 * @brief Applies a permutation to multiple vectors in parallel.
 *
 * @tparam Vectors Variadic template for vector types.
 * @param p Permutation vector.
 * @param vectors References to vectors to permute.
 */
template <typename... Vectors>
void apply_permutation_to_all_parallel(const std::vector<std::size_t> &p, Vectors &...vectors) {
    (omp_apply_permutation_parallel(p, vectors), ...);
}

/**
 * @brief Sorts multiple vectors based on a primary vector in parallel.
 *
 * @tparam T Type of elements in the primary vector.
 * @tparam Compare Comparator type.
 * @tparam Vectors Variadic template for additional vectors.
 * @param primary Vector to sort by.
 * @param cmp Comparator instance.
 * @param vectors Additional vectors to permute.
 */
template <typename T, typename Compare, typename... Vectors>
void omp_sort_multiple_vectors(const std::vector<T> &primary, Compare cmp, Vectors &...vectors) {
    std::vector<std::size_t> p = omp_merge_sort_permutation(primary, cmp);
    apply_permutation_to_all_parallel(p, vectors...);
}

/**
 * @brief Sorts multiple vectors in ascending order based on a primary vector.
 *
 * @tparam T Type of elements in the primary vector.
 * @tparam Vectors Variadic template for additional vectors.
 * @param primary Vector to sort by.
 * @param vectors Additional vectors to permute.
 */
template <typename T, typename... Vectors> void omp_sort_ascending(const std::vector<T> &primary, Vectors &...vectors) {
    omp_sort_multiple_vectors(primary, ComparatorLT<T, T>(), vectors...);
}

/**
 * @brief Sorts multiple vectors in descending order based on a primary vector.
 *
 * @tparam T Type of elements in the primary vector.
 * @tparam Vectors Variadic template for additional vectors.
 * @param primary Vector to sort by.
 * @param vectors Additional vectors to permute.
 */
template <typename T, typename... Vectors>
void omp_sort_descending(const std::vector<T> &primary, Vectors &...vectors) {
    omp_sort_multiple_vectors(primary, ComparatorGT<T, T>(), vectors...);
}

/**
 * @brief Parallel lower bound search using OpenMP.
 *
 * @tparam T Type of elements in the array.
 * @param arr Vector to search in.
 * @param size Size of the vector.
 * @param val Value to find the lower bound for.
 * @param num_cpus Number of CPU threads to use.
 * @return Index of the lower bound.
 */
template <typename T>
size_t omp_lower_bound(std::vector<T> arr, size_t size, T &val, int num_cpus) {
    std::vector<size_t> results(num_cpus);
#pragma omp parallel num_threads(num_cpus)
    {
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t chunk = (size + num_threads - 1) / num_threads;
        size_t start = tid * chunk;
        size_t end = std::min(start + chunk, size);

        if (start < size) {
            auto res = std::lower_bound(arr.begin() + start, arr.begin() + end, val);
            if (res == arr.begin() + end) {
                results[tid] = size;
            } else {
                results[tid] = start + (res - (arr.begin() + start));
            }
        }
    }
    omp_merge_sort(results);
    return results[0];
}

/**
 * @brief Exclusive parallel prefix sum (scan) using OpenMP.
 *
 * @tparam SIZE_TYPE Type of elements.
 * @param input Input vector.
 * @param output Unique pointer to output array.
 * @param n Number of elements.
 */
template <typename SIZE_TYPE>
void omp_scan(const std::vector<SIZE_TYPE> &input, std::unique_ptr<SIZE_TYPE[]> &output, SIZE_TYPE n) {
    SIZE_TYPE scan_a = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i < n; ++i) {
        output[i] = scan_a;
#pragma omp scan exclusive(scan_a)
        {
            scan_a += input[i];
        }
    }
}

/**
 * @brief Inclusive parallel prefix sum (scan) using OpenMP.
 *
 * @tparam SIZE_TYPE Type of elements.
 * @param input Input vector.
 * @param output Unique pointer to output array.
 * @param n Number of elements.
 */
template <typename SIZE_TYPE>
void omp_full_scan(const std::vector<SIZE_TYPE> &input, std::unique_ptr<SIZE_TYPE[]> &output, SIZE_TYPE n) {
    SIZE_TYPE scan_a = 0;
    output[0] = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i < n; ++i) {
        output[i + 1] = scan_a;
#pragma omp scan inclusive(scan_a)
        {
            scan_a += input[i];
        }
    }
}

/**
 * @brief Finds top-k elements per row in a matrix in parallel, returning a CSR structure.
 *
 * @tparam T Type of matrix elements.
 * @tparam Compare Comparator type, defaults to greater-than.
 * @param array Pointer to the matrix data.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param k Number of top elements to find per row.
 * @param num_cpus Number of CPU threads, defaults to 4.
 * @param cmp Comparator instance, defaults to greater-than.
 * @return sparse_struct CSR structure containing top-k elements and indices.
 */
template <typename T, typename Compare>
sparse_struct<size_t, CSRPtrs<size_t>, CSRIndices<size_t>, UnaryValues<T>> omp_top_k_per_row(
    T *array,
    size_t rows,
    size_t cols,
    size_t k,
    int num_cpus = 4,
    Compare cmp = ComparatorGT<T, T>()) {
    std::vector<size_t> cumulate(rows + 1);
    std::vector<T> a_vals;
    a_vals.reserve(rows * k);
    std::vector<size_t> a_indices;
    a_indices.reserve(rows * k);

    for (size_t row = 0; row < rows; ++row) {
        T *a = array + row * cols;
        size_t actual_k = std::min(k, cols);
        std::vector<size_t> a_perm = omp_merge_sort_permutation(std::vector<T>(a, a + cols), cmp);

        std::vector<size_t> top_k_perm(actual_k);
        std::copy(a_perm.begin(), a_perm.begin() + actual_k, top_k_perm.begin());
        omp_merge_sort(top_k_perm);

        std::vector<T> values(actual_k);
#pragma omp parallel for
        for (size_t i = 0; i < actual_k; ++i) {
            values[i] = a[top_k_perm[i]];
        }

        cumulate[row] = omp_lower_bound(values, actual_k, values[0], num_cpus);

        std::copy(values.begin(), values.begin() + cumulate[row], std::back_inserter(a_vals));
        std::copy(top_k_perm.begin(), top_k_perm.begin() + cumulate[row], std::back_inserter(a_indices));
    }

    std::unique_ptr<size_t[]> csr_pointers = std::make_unique<size_t[]>(rows + 1);
    omp_full_scan(cumulate, csr_pointers, rows);

    sparse_struct<size_t, CSRPtrs<size_t>, CSRIndices<size_t>, UnaryValues<T>> csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.ptrs[0].reset(csr_pointers.release());
    std::unique_ptr<T[]> csr_values = std::make_unique<T[]>(a_vals.size());
    std::unique_ptr<size_t[]> csr_indices = std::make_unique<size_t[]>(a_indices.size());

    std::copy(a_vals.begin(), a_vals.end(), csr_values.get());
    std::copy(a_indices.begin(), a_indices.end(), csr_indices.get());

    csr.values.reset(csr_values.release());
    csr.indices.reset(csr_indices.release());

    return csr;
}

#endif