#include "../headers/csf.h"
#include <omp.h>

csf_struct convert_vovov_to_csf(const std::vector<std::vector<int>> *col_indices,
                                const std::vector<std::vector<std::vector<int>>> *fiber_indices,
                                const std::vector<std::vector<std::vector<float>>> *fiber_values,
                                int num_row,
                                int num_col,
                                int num_fiber,
                                int nnz,
                                int nnf) {
    // Allocate memory for the CSR format
    int *col_ptrs = new int[num_row + 1];
    int *flattened_col_indices = new int[nnf];
    int *flattened_fiber_pointers = new int[nnf + 1];
    int *flattened_fiber_indices = new int[nnz];
    float *flattened_fiber_values = nullptr;
    if (fiber_values != nullptr) {
        flattened_fiber_values = new float[nnz];
    }

    int col_ptr = 0; // Initialize the column pointer for the flattened arrays
    int fbr_ptr, fiber_loc = 0;

    for (int row = 0; row < num_row; row++) {
        col_ptrs[row] = col_ptr;

        // Flatten the row_indices and values vectors for this column
        const std::vector<int> &col_idx = (*col_indices)[row];
        for (int i = 0; i < col_idx.size(); i++) {
            flattened_col_indices[col_ptr] = col_idx[i];
            col_ptr++;
        }

        const std::vector<std::vector<int>> &fbr_idx = (*fiber_indices)[row];
        for (int i = 0; i < fbr_idx.size(); i++) {
            flattened_fiber_pointers[fiber_loc] = fbr_ptr;
            for (int j = 0; j < fbr_idx[i].size(); j++) {
                flattened_col_indices[col_ptr] = fbr_idx[i][j];
                fbr_ptr++;
            }
            fiber_loc++;
        }

        if (fiber_values != nullptr) {
            const std::vector<std::vector<float>> &fbr_val_idx = (*fiber_values)[row];
            for (int i = 0; i < fbr_val_idx.size(); i++) {
                for (int j = 0; j < fbr_val_idx[i].size(); j++) {
                    flattened_col_indices[col_ptr] = fbr_val_idx[i][j];
                }
            }
        }
    }

    // Update the last column and fiber pointers
    col_ptrs[num_col] = col_ptr;
    flattened_fiber_pointers[nnf] = fbr_ptr;

    // Create the CSC struct
    csf_struct csf(col_ptrs,
                   flattened_col_indices,
                   flattened_fiber_pointers,
                   flattened_fiber_indices,
                   flattened_fiber_values,
                   nnz,
                   nnf,
                   num_row,
                   num_col,
                   num_fiber);

    return csf;
}