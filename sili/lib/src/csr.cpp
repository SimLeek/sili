#include "../headers/csr.h"
#include <omp.h>

csr_struct convert_vov_to_cs2(const std::vector<std::vector<int>> *indices,
                              const std::vector<std::vector<float>> *values,
                              const std::vector<std::vector<float>> *connections,
                              int num_col,
                              int num_row,
                              int numNonZero) {
    // Allocate memory for the CSC format
    int *ccol_indices = new int[num_col + 1];
    int *flattened_row_indices = new int[numNonZero];
    float *flattened_values = nullptr;
    if (values != nullptr) {
        flattened_values = new float[numNonZero];
    }
    float *flattened_connections = nullptr;
    if (connections != nullptr) {
        flattened_connections = new float[numNonZero];
    }

    int col_ptr = 0; // Initialize the column pointer for the flattened arrays

    for (int col = 0; col < num_col; col++) {
        ccol_indices[col] = col_ptr;

        // Flatten the row_indices and values vectors for this column
        const std::vector<int> &col_idx = (*indices)[col];
        for (int i = 0; i < col_idx.size(); i++) {
            flattened_row_indices[col_ptr] = col_idx[i];
            col_ptr++;
        }

        if (values != nullptr) {
            const std::vector<float> &val = (*values)[col];
            for (int i = 0; i < col_idx.size(); i++) {
                flattened_values[ccol_indices[col] + i] = val[i];
            }
        }

        if (connections != nullptr) {
            const std::vector<float> &con = (*connections)[col];
            for (int i = 0; i < col_idx.size(); i++) {
                flattened_connections[ccol_indices[col] + i] = con[i];
            }
        }
    }

    // Update the last column pointer for crow_indices
    ccol_indices[num_col] = col_ptr;

    // Create the CSC struct
    csr_struct csc;
    csc.ptrs = ccol_indices;
    csc.indices = flattened_row_indices;
    csc.values = flattened_values;
    csc.nnz = numNonZero;
    csc.rows = num_col;
    csc.cols = num_row;

    return csc;
}