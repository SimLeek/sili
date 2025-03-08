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

template <class A, class B> class comparator_lt {
  public:
    bool operator()(A a, B b) { return (a < b); }
};

template <class A, class B> class comparator_gt {
  public:
    bool operator()(A a, B b) { return (a > b); }
};

template <typename T, typename Compare>
class PermutationComparator {
    const std::vector<T>& a;
    Compare cmp;
public:
    PermutationComparator(const std::vector<T>& a, Compare cmp) : a(a), cmp(cmp) {}
    bool operator()(std::size_t i, std::size_t j) const {
        return cmp(a[i], a[j]);
    }
};

//https://cw.fel.cvut.cz/old/_media/courses/b4m35pag/lab6_slides_advanced_openmp.pdf
template <class TYPE, class COMPARE = comparator_lt<TYPE, TYPE>>
void _ompMergeSortRecursive(std::vector<TYPE> &v, unsigned long left, unsigned long right) {
    COMPARE cmp;
    if (left < right) {
        if (right - left >= 32) {
            unsigned long mid = (left + right) / 2;
#pragma omp taskgroup
            {
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _ompMergeSortRecursive(v, left, mid);
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _ompMergeSortRecursive(v, mid + 1, right);
#pragma omp taskyield
            }
            std::inplace_merge(v.begin() + left, v.begin() + mid + 1, v.begin() + right + 1, cmp);
        } else {
            std::sort(v.begin() + left, v.begin() + right + 1, cmp);
        }
    }
}

//https://cw.fel.cvut.cz/old/_media/courses/b4m35pag/lab6_slides_advanced_openmp.pdf
template <class TYPE, class COMPARE = comparator_lt<TYPE, TYPE>> 
void ompMergeSort(std::vector<TYPE> &v) {
#pragma omp parallel
#pragma omp single
    _ompMergeSortRecursive(v, 0, v.size() - 1);
}

template <typename TYPE, typename COMPARE>
std::vector<std::size_t> omp_merge_sort_permutation(const std::vector<TYPE>& a, COMPARE cmp) {
    std::vector<std::size_t> p(a.size());
    std::iota(p.begin(), p.end(), 0);
    PermutationComparator<TYPE, COMPARE> perm_cmp(a, cmp);
    ompMergeSort<std::size_t, PermutationComparator<TYPE, COMPARE>>(p);
    return p;
}

template <typename T>
void omp_apply_permutation_parallel(const std::vector<std::size_t>& p, std::vector<T>& vec) {
    std::vector<T> temp(vec.size());
#pragma omp parallel for
    for (size_t i = 0; i < p.size(); ++i) {
        temp[i] = vec[p[i]];
    }
    vec = std::move(temp);
}

template <typename... Vectors>
void apply_permutation_to_all_parallel(const std::vector<std::size_t>& p, Vectors&... vectors) {
    (apply_permutation_parallel(p, vectors), ...);
}

template <typename T, typename Compare, typename... Vectors>
void omp_sort_multiple_vectors(const std::vector<T>& primary, Compare cmp, Vectors&... vectors) {
    // Sort the permutation in parallel
    std::vector<std::size_t> p = sort_permutation(primary, cmp);
    // Apply the permutation to all vectors in parallel
    apply_permutation_to_all_parallel(p, vectors...);
}

// Convenience wrappers
template <typename T, typename... Vectors>
void omp_sort_ascending(const std::vector<T>& primary, Vectors&... vectors) {
    omp_sort_multiple_vectors(primary, comparator_lt<T, T>(), vectors...);
}

template <typename T, typename... Vectors>
void omp_sort_descending(const std::vector<T>& primary, Vectors&... vectors) {
    omp_sort_multiple_vectors(primary, comparator_gt<T, T>(), vectors...);
}

template <typename T, typename Predicate>
size_t omp_lower_bound(std::vector<T> arr, size_t size, T& val, int num_cpus) {
    std::vector<size_t> results(num_cpus); // Default: not found
#pragma omp parallel num_threads(num_cpus)
    {
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t chunk = (size + num_threads - 1) / num_threads; // Ceiling division
        size_t start = tid * chunk;
        size_t end = std::min(start + chunk, size);
        
        if (start < size) {
            // Sequential search within chunk
            auto res = std::lower_bound(arr+start, arr+end, val);
            if(res==arr+end){
                results[tid] = size;
            }else{
                results[tid] = start + std::lower_bound(arr+start, arr+end, val);
            }
        }
    }

    ompMergeSort(results);

    auto result = results[0];

    return result;
}

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

template <typename SIZE_TYPE>
void omp_full_scan(const std::vector<SIZE_TYPE> &input, std::unique_ptr<SIZE_TYPE[]> &output, SIZE_TYPE n) {
    SIZE_TYPE scan_a = 0;
    output[0] = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i < n; ++i) {
        output[i+1] = scan_a;
#pragma omp scan inclusive(scan_a)
        {
            scan_a += input[i];
        }
    }
}

template <typename T, typename Compare>
void omp_top_k_per_row(
    T* array,
    size_t rows,
    size_t cols,
    size_t k,
    int num_cpus = 4,
    Compare cmp=comparator_gt<T, T>()
) {

//#pragma omp parallel for // will actually slow things down since we have few long rows usually
    std::vector<size_t> cumulate(rows+1);
    std::vector<T> a_vals;
    a_vals.reserve(rows*k);
    std::vector<size_t> a_indices;
    a_indices.reserve(rows*k);

    for (size_t row = 0; row < rows; ++row) {
        T* a = array + row*cols;
        size_t actual_k = std::min(k, cols);

        auto a_perm = omp_merge_sort_permutation(a, cmp);

        // Extract top-k permutation
        std::vector<size_t> top_k_perm(actual_k);
        std::copy(a_perm.begin(), a_perm.begin()+actual_k, top_k_perm.begin());

        // Sort top-k indices
        ompMergeSort(top_k_perm);

        // Apply permutation to values
        std::vector<T>values(actual_k);
        #pragma omp parallel for
        for (size_t i = 0; i < actual_k; ++i) {
            values[i] = a[top_k_perm[i]];
        }

        // Binary search (example: first index where value <= 0)
        auto pred = [](T val) { return val <= 0; };
        cumulate[row] = omp_lower_bound(values, actual_k, pred, num_cpus);  // is actual_k when lower bound not found

        // Store outputs
        std::copy(values.begin(), values.begin()+cumulate[row], std::back_inserter(a_vals));
        std::copy(top_k_perm.begin(), top_k_perm.begin()+cumulate[row], std::back_inserter(a_indices));

        delete[] a_perm;
    }

    // Compute CSR pointers
    std::unique_ptr<size_t[]> csr_pointers = std::make_unique<size_t[]>(rows + 1);
    omp_full_scan(cumulate, csr_pointers, rows);
    // Note: Caller must use csr_pointers and free it

    sparse_struct<
    size_t,
    CSRPtrs<size_t>, // First SIZE_TYPE transformed to CSRPtrs
    CSRIndices<size_t>, // INDEX_ARRAYS reduced by one
    UnaryValues<T>
    > csr;

    csr.rows = rows;
    csr.cols = cols;
    csr.ptrs[0].reset(csr_pointers);
    std::unique_ptr<T[]> csr_values = std::make_unique<T[]>(a_vals.size());
    std::unique_ptr<size_t[]> csr_indices = std::make_unique<size_t[]>(a_indices.size());

    std::copy(a_vals.begin(), a_vals.end(), csr_values.get());
    std::copy(a_indices.begin(), a_indices.end(), csr_indices.get());

    return csr;
}

#endif