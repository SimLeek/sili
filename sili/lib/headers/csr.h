#ifndef __CSR__H_
#define __CSR__H_

#include <vector>

struct csr_struct {
    int *ptrs;
    int *indices;
    float *values;
    int nnz;
    int rows;
    int cols;
    int _reserved_indices_and_values = 0;

    csr_struct()
        : ptrs(nullptr), indices(nullptr), values(nullptr), nnz(0), rows(0), cols(0), _reserved_indices_and_values(0) {}

    csr_struct(int *p, int *ind, float *val, int non_zero, int num_p, int max_idx, int reserved)
        : ptrs(p), indices(ind), values(val), nnz(non_zero), rows(num_p), cols(max_idx),
          _reserved_indices_and_values(reserved) {}

    csr_struct(int *p, int *ind, float *val, int non_zero, int num_p, int max_idx)
        : ptrs(p), indices(ind), values(val), nnz(non_zero), rows(num_p), cols(max_idx) {}
};

csr_struct convert_vov_to_cs2(const std::vector<std::vector<int>> *indices,
                              const std::vector<std::vector<float>> *values,
                              const std::vector<std::vector<float>> *connections,
                              int num_col,
                              int num_row,
                              int numNonZero);

#endif