#ifndef _OUTER_PRODUCT_HPP_
#define _OUTER_PRODUCT_HPP_

#include "csr.hpp"
#include "unique_vector.hpp"
#include <omp.h>
#include <vector>

template <class SIZE_TYPE, class VALUE_TYPE>
sili::unique_vector<csr_struct<SIZE_TYPE, VALUE_TYPE>> outer_product(SIZE_TYPE batches,
                                                                     SIZE_TYPE a_size,
                                                                     SIZE_TYPE b_size,
                                                                     const csr_struct<SIZE_TYPE, VALUE_TYPE> &a,
                                                                     const csr_struct<SIZE_TYPE, VALUE_TYPE> &b,
                                                                     const int num_cpus = omp_get_num_procs()) {
    // output is a list of length batches of CSRs of size a_size by b_size
    sili::unique_vector<csr_struct<SIZE_TYPE, VALUE_TYPE>> result_batches;

    SIZE_TYPE nnz = 0;

#pragma omp parallel num_threads(num_cpus)
    {
        for (SIZE_TYPE batch = 0; batch < batches; ++batch) {
            SIZE_TYPE a_start = a.ptrs[batch];
            SIZE_TYPE a_end = a.ptrs[batch + 1];
            SIZE_TYPE num_rows = a_end - a_start;

            SIZE_TYPE b_start = b.ptrs[batch];
            SIZE_TYPE b_end = b.ptrs[batch + 1];
            SIZE_TYPE num_cols = b_end - b_start;

            sili::unique_vector<sili::unique_vector<SIZE_TYPE>> result_indices(num_rows);
            sili::unique_vector<sili::unique_vector<VALUE_TYPE>> result_values(num_rows);

            SIZE_TYPE tid = omp_get_thread_num();
            SIZE_TYPE chunk_size = (num_rows + num_cpus - 1) / num_cpus;
            SIZE_TYPE start = tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, num_rows);

            SIZE_TYPE local_nnz = 0;

            for (SIZE_TYPE i = start; i < end; ++i) {
                SIZE_TYPE a_index = a_start + i;
                VALUE_TYPE a_val = a.values[a_index];
                SIZE_TYPE row_idx = a.indices[a_index];

                for (SIZE_TYPE j = b_start; j < b_end; ++j) {
                    SIZE_TYPE col_idx = b.indices[j];
                    VALUE_TYPE b_val = b.values[j];
                    VALUE_TYPE product = a_val * b_val;

                    result_indices[i].push_back(col_idx);
                    result_values[i].push_back(product);
                    local_nnz += 1;
                }
            }

#pragma omp atomic
            nnz += local_nnz;

#pragma omp barrier

// Convert the result to CSR format for the current batch
#pragma omp single
            { result_batches.push_back(convert_vov_to_csr(&result_indices, &result_values, a_size, b_size, nnz)); }

#pragma omp barrier // Ensure all threads are done before resetting nnz
#pragma omp single
            {
                nnz = 0; // Reset nnz for the next batch
            }
#pragma omp barrier
        }
    }

    return result_batches;
}

template <class SIZE_TYPE, class VALUE_TYPE>
void outer_product_backwards_b(SIZE_TYPE batches,
                               SIZE_TYPE a_size,
                               SIZE_TYPE b_size,
                               const csr_struct<SIZE_TYPE, VALUE_TYPE> &a,
                               const csr_struct<SIZE_TYPE, VALUE_TYPE> &b_grad,
                               sili::unique_vector<csr_struct<SIZE_TYPE, VALUE_TYPE>> o_grad_b,
                               const int num_cpus = omp_get_num_procs()) {
#pragma omp parallel num_threads(num_cpus)
    {
        for (SIZE_TYPE batch = 0; batch < batches; ++batch) {
            SIZE_TYPE a_start = a.ptrs[batch];
            SIZE_TYPE a_end = a.ptrs[batch + 1];
            SIZE_TYPE num_rows = a_end - a_start;

            SIZE_TYPE b_grad_start = b_grad.ptrs[batch];
            SIZE_TYPE b_grad_end = b_grad.ptrs[batch + 1];
            SIZE_TYPE num_grad_cols = b_grad_end - b_grad_start;

            SIZE_TYPE tid = omp_get_thread_num();
            SIZE_TYPE chunk_size = (num_grad_cols + num_cpus - 1) / num_cpus;
            SIZE_TYPE start = tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, num_grad_cols);

            for (SIZE_TYPE i = start; i < end; ++i) {
                SIZE_TYPE col_idx = b_grad.indices[i];

                for (SIZE_TYPE j = a_start; j < a_end; ++j) {
                    VALUE_TYPE a_val = a.values[j];
                    SIZE_TYPE row_idx = a.indices[j];

                    // this works assuming you made the outer product mask first
                    //  using a by b_mask
                    //  then populated it with the backprop values
                    //  then sent it here
                    SIZE_TYPE o_start = o_grad_b[batch].ptrs[row_idx];
                    VALUE_TYPE o_val = o_grad_b[batch].values[j];

                    b_grad.values[j] += a_val * o_val;
                }
            }
// synchronize so no thread goes ahead and gets to duplicate b_grads, so we don't have to use atomics
#pragma omp barrier
        }
    }
}

template <class SIZE_TYPE, class VALUE_TYPE>
void outer_product_backwards_a(SIZE_TYPE batches,
                               SIZE_TYPE a_size,
                               SIZE_TYPE b_size,
                               const csr_struct<SIZE_TYPE, VALUE_TYPE> &a_grad,
                               const csr_struct<SIZE_TYPE, VALUE_TYPE> &b,
                               sili::unique_vector<csr_struct<SIZE_TYPE, VALUE_TYPE>> o_grad_a,
                               const int num_cpus = omp_get_num_procs()) {
// output is a list of length batches of CSRs of size a_size by b_size
#pragma omp parallel num_threads(num_cpus)
    {
        for (SIZE_TYPE batch = 0; batch < batches; ++batch) {
            SIZE_TYPE a_grad_start = a_grad.ptrs[batch];
            SIZE_TYPE a_grad_end = a_grad.ptrs[batch + 1];
            SIZE_TYPE num_rows = a_grad_end - a_grad_start;

            SIZE_TYPE b_start = b.ptrs[batch];
            SIZE_TYPE b_end = b.ptrs[batch + 1];
            SIZE_TYPE num_cols = b_end - b_start;

            SIZE_TYPE tid = omp_get_thread_num();
            SIZE_TYPE chunk_size = (num_rows + num_cpus - 1) / num_cpus;
            SIZE_TYPE start = tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, num_rows);

            for (SIZE_TYPE i = start; i < end; ++i) {
                SIZE_TYPE a_index = a_grad_start + i;
                // VALUE_TYPE a_val = a_batches.values[a_index];
                SIZE_TYPE row_idx = a_grad.indices[a_index];

                for (SIZE_TYPE j = b_start; j < b_end; ++j) {
                    SIZE_TYPE col_idx = b.indices[j];
                    VALUE_TYPE b_val = b.values[j];

                    // this works assuming you made the outer product mask first
                    //  using a by b_mask
                    //  then populated it with the backprop values
                    //  then sent it here
                    SIZE_TYPE o_start = o_grad_a[batch].ptrs[row_idx];
                    VALUE_TYPE o_val = o_grad_a[batch].values[j];

                    a_grad.values[j] += b_val * o_val;
                }
#pragma omp barrier
            }
        }
    }
}

// outer product mask for backpropogating into the outer product.
// use either the the a/b csr arrays with zeroed out values or CSRMask's random mask for a_grad and b_grad
template <class SIZE_TYPE, class VALUE_TYPE>
sili::unique_vector<csr_struct<SIZE_TYPE, VALUE_TYPE>> build_outer_product_mask(
    SIZE_TYPE batches,
    SIZE_TYPE a_size,
    SIZE_TYPE b_size,
    const csr_struct<SIZE_TYPE, VALUE_TYPE> &a,
    const csr_struct<SIZE_TYPE, VALUE_TYPE> &b,
    const int num_cpus = omp_get_num_procs()) {
    // output is a list of length batches of CSRs of size a_size by b_size
    sili::unique_vector<csr_struct<SIZE_TYPE, VALUE_TYPE>> result_batches;

    SIZE_TYPE nnz = 0;

#pragma omp parallel num_threads(num_cpus) reduction(+ : nnz)
    {
        for (SIZE_TYPE batch = 0; batch < batches; ++batch) {
            SIZE_TYPE a_start = a.ptrs[batch];
            SIZE_TYPE a_end = a.ptrs[batch + 1];
            SIZE_TYPE num_rows = a_end - a_start;

            SIZE_TYPE b_start = b.ptrs[batch];
            SIZE_TYPE b_end = b.ptrs[batch + 1];
            SIZE_TYPE num_cols = b_end - b_start;

            SIZE_TYPE local_nnz = 0;

            std::vector<std::vector<SIZE_TYPE>> result_indices(num_rows, std::vector<SIZE_TYPE>());

            SIZE_TYPE tid = omp_get_thread_num();
            SIZE_TYPE chunk_size = (num_rows + num_cpus - 1) / num_cpus;
            SIZE_TYPE start = tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, num_rows);

            for (SIZE_TYPE i = start; i < end; ++i) {
                SIZE_TYPE a_index = a_start + i;
                SIZE_TYPE row_idx = a.indices[a_index];

                for (SIZE_TYPE j = b_start; j < b_end; ++j) {
                    SIZE_TYPE col_idx = b.indices[j];

                    result_indices[i].push_back(col_idx);
                    local_nnz += 1;
                }
            }

#pragma omp atomic
            nnz += local_nnz;

#pragma omp barrier

// Convert the result to CSR format for the current batch
#pragma omp single
            { result_batches.push_back(convert_vov_to_csr(&result_indices, nullptr, a_size, b_size, nnz)); }

#pragma omp barrier // Ensure all threads are done before resetting nnz
#pragma omp single
            {
                nnz = 0; // Reset nnz for the next batch
            }
#pragma omp barrier
        }
    }

    return result_batches;
}

#endif