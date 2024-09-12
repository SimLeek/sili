#ifndef __CSR__H_
#define __CSR__H_

#include <memory>
#include "unique_vector.hpp"

// some of the basic tests I wrote at this point failed to delete the raw pointers, so while there is some overhead for
// unique_ptr, I'm keeping it to stay sane.
//  Also there's basically no overhead anyway if you compile with -O3 or -O4, which is the intended option
template <class SIZE_TYPE, class VALUE_TYPE> struct csr_struct {
    std::unique_ptr<SIZE_TYPE[]> ptrs;
    std::unique_ptr<SIZE_TYPE[]> indices;
    std::unique_ptr<VALUE_TYPE[]> values;
    SIZE_TYPE rows;
    SIZE_TYPE cols;
    SIZE_TYPE _reserved_indices_and_values = 0;

    csr_struct()
        : ptrs(nullptr), indices(nullptr), values(nullptr), rows(0), cols(0), _reserved_indices_and_values(0) {}

    csr_struct(int *p, int *ind, float *val, int non_zero, int num_p, int max_idx, int reserved)
        : ptrs(p), indices(ind), values(val), rows(num_p), cols(max_idx), _reserved_indices_and_values(reserved) {}

    csr_struct(int *p, int *ind, float *val, int non_zero, int num_p, int max_idx)
        : ptrs(p), indices(ind), values(val), rows(num_p), cols(max_idx) {}

    SIZE_TYPE nnz() const { return (ptrs != nullptr) ? ptrs[rows] : 0; }
};

template <class SIZE_TYPE, class VALUE_TYPE>
csr_struct<SIZE_TYPE, VALUE_TYPE> convert_vov_to_csr(
    const sili::unique_vector<sili::unique_vector<SIZE_TYPE>> *indices,
    const sili::unique_vector<sili::unique_vector<VALUE_TYPE>> *values,
    SIZE_TYPE num_col,
    SIZE_TYPE num_row,
    SIZE_TYPE numNonZero) {
    // Allocate memory for the CSR format
    csr_struct<SIZE_TYPE, VALUE_TYPE> csr;
    csr.ptrs = std::make_unique<SIZE_TYPE[]>(num_col + 1);
    csr.indices = std::make_unique<SIZE_TYPE[]>(numNonZero);

    if (values != nullptr) {
        csr.values = std::make_unique<VALUE_TYPE[]>(numNonZero);
    } else {
        csr.values = nullptr;
    }

    SIZE_TYPE ptr = 0; // Initialize the pointers for the flattened arrays

    for (SIZE_TYPE row = 0; row < num_row; row++) {
        csr.ptrs[row] = ptr;

        const sili::unique_vector<SIZE_TYPE> &col_idx = (*indices)[row];
        ptr += col_idx.size();

        if (ptr > numNonZero) {
            throw std::runtime_error("Actual number of non-zero elements exceeds the expected number used to reserve memory for arrays. This will lead to double free/corruption errors and segfaults.");
        }

        // Flatten the row_indices and values vectors for this column
        std::copy(col_idx.begin(), col_idx.end(), csr.indices.get() + csr.ptrs[row]);

        if (values != nullptr) {
            const sili::unique_vector<VALUE_TYPE> &val = (*values)[row];
            std::copy(val.begin(), val.end(), csr.values.get() + csr.ptrs[row]);
        }
    }

    // Update the last column pointer for crow_indices
    csr.ptrs[num_row] = ptr;

    // Create the CSR struct
    csr.rows = num_row;
    csr.cols = num_col;

    return csr;
}

#endif