#include "../headers/csr.h"
#include <omp.h>
#include <thread>
#include <vector>

std::vector<csr_struct> outer_product(int batches, int a_size, int b_size, const csr_struct &a, const csr_struct &b) {
    // output is a list of length batches of CSRs of size a_size by b_size
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<csr_struct> result_batches;

    for (int batch = 0; batch < batches; ++batch) {
        int a_start = a.ptrs[batch];
        int a_end = a.ptrs[batch + 1];
        int num_rows = a_end - a_start;

        int b_start = b.ptrs[batch];
        int b_end = b.ptrs[batch + 1];
        int num_cols = b_end - b_start;

        int nnz = 0;

        std::vector<std::vector<int>> result_indices(num_rows, std::vector<int>());
        std::vector<std::vector<float>> result_values(num_rows, std::vector<float>());

#pragma omp parallel num_threads(num_cpus) reduction(+ : nnz)
        {
            int tid = omp_get_thread_num();
            int chunk_size = (num_rows + num_cpus - 1) / num_cpus;
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, num_rows);

            for (int i = start; i < end; ++i) {
                int a_index = a_start + i;
                float a_val = a.values[a_index];
                int row_idx = a.indices[a_index];

                for (int j = b_start; j < b_end; ++j) {
                    int col_idx = b.indices[j];
                    float b_val = b.values[j];
                    float product = a_val * b_val;

                    result_indices[i].push_back(col_idx);
                    result_values[i].push_back(product);
                    nnz += 1;
                }
            }
        }

        // Convert the result to CSR format for the current batch
        result_batches.emplace_back(convert_vov_to_cs2(&result_indices, &result_values, nullptr, a_size, b_size, nnz));
    }

    return result_batches;
}

void outer_product_backwards_b(int batches,
                               int a_size,
                               int b_size,
                               const csr_struct &a,
                               const csr_struct &b_grad,
                               std::vector<csr_struct> o_grad_b) {
    // output is a list of length batches of CSRs of size a_size by b_size
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<csr_struct> result_batches;

    for (int batch = 0; batch < batches; ++batch) {
        int a_start = a.ptrs[batch];
        int a_end = a.ptrs[batch + 1];
        int num_rows = a_end - a_start;

        int b_grad_start = b_grad.ptrs[batch];
        int b_grad_end = b_grad.ptrs[batch + 1];
        int num_grad_cols = b_grad_end - b_grad_start;

#pragma omp parallel num_threads(num_cpus)
        {
            int tid = omp_get_thread_num();
            int chunk_size = (num_rows + num_cpus - 1) / num_cpus;
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, num_rows);

            for (int i = start; i < end; ++i) {
                int a_index = a_start + i;
                float a_val = a.values[a_index];
                int row_idx = a.indices[a_index];

                for (int j = b_grad_start; j < b_grad_end; ++j) {
                    int col_idx = b_grad.indices[j];

                    // this works assuming you made the outer product mask first
                    //  using a by b_mask
                    //  then populated it with the backprop values
                    //  then sent it here
                    int o_start = o_grad_b[batch].ptrs[row_idx];
                    float o_val = o_grad_b[batch].values[j];

                    b_grad.values[j] += a_val * o_val;
                }
            }
        }
    }
}

void outer_product_backwards_a(int batches,
                               int a_size,
                               int b_size,
                               const csr_struct &a_grad,
                               const csr_struct &b,
                               std::vector<csr_struct> o_grad_a) {
    // output is a list of length batches of CSRs of size a_size by b_size
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<csr_struct> result_batches;

    for (int batch = 0; batch < batches; ++batch) {
        int a_grad_start = a_grad.ptrs[batch];
        int a_grad_end = a_grad.ptrs[batch + 1];
        int num_rows = a_grad_end - a_grad_start;

        int b_start = b.ptrs[batch];
        int b_end = b.ptrs[batch + 1];
        int num_cols = b_end - b_start;

#pragma omp parallel num_threads(num_cpus)
        {
            int tid = omp_get_thread_num();
            int chunk_size = (num_rows + num_cpus - 1) / num_cpus;
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, num_rows);

            for (int i = start; i < end; ++i) {
                int a_index = a_grad_start + i;
                // float a_val = a_batches.values[a_index];
                int row_idx = a_grad.indices[a_index];

                for (int j = b_start; j < b_end; ++j) {
                    int col_idx = b.indices[j];
                    float b_val = b.values[j];

                    // this works assuming you made the outer product mask first
                    //  using a by b_mask
                    //  then populated it with the backprop values
                    //  then sent it here
                    int o_start = o_grad_a[batch].ptrs[row_idx];
                    float o_val = o_grad_a[batch].values[j];

                    a_grad.values[j] += b_val * o_val;
                }
            }
        }
    }
}

std::vector<csr_struct> build_outer_product_mask(int batches,
                                                 int a_size,
                                                 int b_size,
                                                 const csr_struct &a,
                                                 const csr_struct &b) {
    // output is a list of length batches of CSRs of size a_size by b_size
    int num_cpus = std::thread::hardware_concurrency();
    std::vector<csr_struct> result_batches;

    for (int batch = 0; batch < batches; ++batch) {
        int a_start = a.ptrs[batch];
        int a_end = a.ptrs[batch + 1];
        int num_rows = a_end - a_start;

        int b_start = b.ptrs[batch];
        int b_end = b.ptrs[batch + 1];
        int num_cols = b_end - b_start;

        int nnz = 0;

        std::vector<std::vector<int>> result_indices(num_rows, std::vector<int>());

#pragma omp parallel num_threads(num_cpus) reduction(+ : nnz)
        {
            int tid = omp_get_thread_num();
            int chunk_size = (num_rows + num_cpus - 1) / num_cpus;
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, num_rows);

            for (int i = start; i < end; ++i) {
                int a_index = a_start + i;
                int row_idx = a.indices[a_index];

                for (int j = b_start; j < b_end; ++j) {
                    int col_idx = b.indices[j];

                    result_indices[i].push_back(col_idx);
                    nnz += 1;
                }
            }
        }

        // Convert the result to CSR format for the current batch
        result_batches.emplace_back(convert_vov_to_cs2(&result_indices, nullptr, nullptr, a_size, b_size, nnz));
    }

    return result_batches;
}