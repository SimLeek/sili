#ifndef __LINEAR_HPP__
#define __LINEAR_HPP__

#include "csr.hpp"
#include "scan.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <omp.h>
#include <cstring>
#include "unique_vector.hpp"

/* #region Linear Sparse IO Fwd */
template <class W_CONTAINER, class SIZE_TYPE, class VALUE_TYPE>
inline void _do_linear_sidlso_fwd(int num_cpus,
                                  SIZE_TYPE input_size,
                                  SIZE_TYPE output_size,
                                  SIZE_TYPE batch,
                                  csr_struct<SIZE_TYPE, VALUE_TYPE> &input_csr,
                                  W_CONTAINER& W,
                                  sili::unique_vector<sili::unique_vector<SIZE_TYPE>> &row_indices_chunks,
                                  sili::unique_vector<sili::unique_vector<VALUE_TYPE>> &row_values_chunks,
                                  VALUE_TYPE eps) {

    int tid = omp_get_thread_num();                           // Get thread ID
    SIZE_TYPE chunk_size = (output_size + num_cpus - 1) / num_cpus; // Calculate chunk size
    SIZE_TYPE start = tid * chunk_size;                             // Start index for this thread
    SIZE_TYPE end = std::min(start + chunk_size,
                        output_size); // End index for this thread

    for (SIZE_TYPE output_index = start; output_index < end; ++output_index) {
        VALUE_TYPE out_val = 0;
        for (SIZE_TYPE input_ptr = input_csr.ptrs[batch]; input_ptr < input_csr.ptrs[batch + 1]; input_ptr++) {
            SIZE_TYPE input_index = input_csr.indices[input_ptr];
            VALUE_TYPE input_value = input_csr.values[input_ptr];

            out_val += W[output_index * input_size + input_index] * input_value;
        }
        if (std::abs(out_val) > eps) {
            row_indices_chunks[tid].push_back(output_index);
            row_values_chunks[tid].push_back(out_val);
        }
    }
    # pragma omp barrier
}

template <class SIZE_TYPE, class VALUE_TYPE>
inline void _assign_spv_chunks_to_batch(SIZE_TYPE batch,
                                        sili::unique_vector<size_t> &vec_assign_locs,
                                        sili::unique_vector<sili::unique_vector<SIZE_TYPE>> &out_idx,
                                        sili::unique_vector<sili::unique_vector<VALUE_TYPE>> &out_val,
                                        sili::unique_vector<sili::unique_vector<SIZE_TYPE>> &row_indices_chunks,
                                        sili::unique_vector<sili::unique_vector<VALUE_TYPE>> &row_values_chunks,
                                        SIZE_TYPE &nnz) {
    #pragma omp single
    {
    out_idx[batch].resize(vec_assign_locs.back());
    out_val[batch].resize(vec_assign_locs.back());
    nnz += vec_assign_locs.back();
    }

    int tid = omp_get_thread_num(); // Get thread ID
    size_t start = vec_assign_locs[tid];
    // SIZE_TYPE end = vec_assign_locs[tid+1];
    std::copy(row_indices_chunks[tid].begin(), row_indices_chunks[tid].end(), out_idx[batch].begin() + start);
    std::copy(row_values_chunks[tid].begin(), row_values_chunks[tid].end(), out_val[batch].begin() + start);
}

template <class W_CONTAINER, class SIZE_TYPE, class VALUE_TYPE>
csr_struct<SIZE_TYPE, VALUE_TYPE> linear_sidlso(SIZE_TYPE batch_size,
                         SIZE_TYPE input_size,
                         SIZE_TYPE output_size,
                         csr_struct<SIZE_TYPE, VALUE_TYPE> &input_csr,
                         W_CONTAINER& W,
                         const int num_cpus=omp_get_num_procs(),  // override this to 1 for small batch sizes, input sizes, etc., for 400x faster execution
                         VALUE_TYPE eps = std::numeric_limits<VALUE_TYPE>::epsilon()) {
    if(input_csr.ptrs==nullptr || input_csr.indices==nullptr || input_csr.values==nullptr){
        throw std::runtime_error(
                "input_csr has null pointers.");
    }

    //const int num_cpus = std::thread::hardware_concurrency();
    //const int num_cpus = omp_get_num_procs();
    //const int num_cpus = 1;


    sili::unique_vector<sili::unique_vector<SIZE_TYPE>> out_idx(batch_size);
    sili::unique_vector<sili::unique_vector<VALUE_TYPE>> out_val(batch_size);
    SIZE_TYPE nnz = 0;

    // size 16 to start for cache line optimization
    sili::unique_vector<sili::unique_vector<SIZE_TYPE>> row_indices_chunks(num_cpus);
    sili::unique_vector<sili::unique_vector<VALUE_TYPE>> row_values_chunks(num_cpus);

    sili::unique_vector<size_t> vec_assign_locs(num_cpus+1);

    #pragma omp parallel num_threads(num_cpus)
    {
    int tid = omp_get_thread_num();
    for (SIZE_TYPE batch = 0; batch < batch_size; batch++) {
        _do_linear_sidlso_fwd(
            num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);
        fullScanSizes(row_indices_chunks, vec_assign_locs);

        _assign_spv_chunks_to_batch(
            batch, vec_assign_locs, out_idx, out_val, row_indices_chunks, row_values_chunks, nnz);

        // Clear the contents of the inner vectors, but do not free their memory
        row_indices_chunks[tid].clear();
        row_values_chunks[tid].clear();
    }
    }

    auto csr = convert_vov_to_csr(&out_idx, &out_val, output_size, batch_size, nnz);
    return csr;
}
/* #endregion */

/* #region Linear Sparse IO Back */

/**
 * Computes the sparse backpropagation for a neural network linear operation using OpenMP.
 *
 * @param batch_size The number of batches.
 * @param input_size The size of the input layer.
 * @param output_size The size of the output layer.
 * @param input_csr Sparse input activation values.
 * @param W Pointer to the weight matrix.
 * @param output_grad_csr Sparse output gradients.
 * @param I_grad Sparse input mask; input: mask locations with zeros, output: locations with values.
 * @param W_grad_callback Callback function for handling weight gradients; see comment inside function for examples.
 * @param eps W_grad_callback will only be called if the gradient is larger than eps.
 */
template <class W_CONTAINER, class SIZE_TYPE, class VALUE_TYPE>
void linear_backward_sidlso(SIZE_TYPE batch_size,
                            SIZE_TYPE input_size,
                            SIZE_TYPE output_size,
                            csr_struct<SIZE_TYPE, VALUE_TYPE> &input_csr,
                            W_CONTAINER& W,
                            csr_struct<SIZE_TYPE, VALUE_TYPE> &output_grad_csr,
                            csr_struct<SIZE_TYPE, VALUE_TYPE> &I_grad,
                            std::function<void(VALUE_TYPE, SIZE_TYPE, SIZE_TYPE)> W_grad_callback,
                            const int num_cpus=omp_get_num_procs(),
                            VALUE_TYPE eps = std::numeric_limits<VALUE_TYPE>::epsilon()) {

    if(I_grad.values==nullptr){
        I_grad.values.reset(new VALUE_TYPE[I_grad.nnz()]{});
    }

#pragma omp parallel num_threads(num_cpus)
        {
    for (SIZE_TYPE batch = 0; batch < batch_size; batch++) {
            SIZE_TYPE o_grad_size = output_grad_csr.ptrs[batch + 1] - output_grad_csr.ptrs[batch];
            SIZE_TYPE tid = omp_get_thread_num();                              // Get thread ID
            SIZE_TYPE chunk_size = (o_grad_size + num_cpus - 1) / num_cpus;    // Calculate chunk size
            SIZE_TYPE start = tid * o_grad_size + output_grad_csr.ptrs[batch]; // Start index for this thread
            SIZE_TYPE end =
                std::min(start + chunk_size, o_grad_size) + output_grad_csr.ptrs[batch]; // End index for this thread

            for (SIZE_TYPE ograd_p = start; ograd_p < end; ++ograd_p) {
                // learning inputs: selected neurons
                for (SIZE_TYPE input_ptr = I_grad.ptrs[batch]; input_ptr < I_grad.ptrs[batch + 1]; input_ptr++) {
                    SIZE_TYPE input_index = I_grad.indices[input_ptr];
                    SIZE_TYPE output_index = output_grad_csr.indices[ograd_p];
                    VALUE_TYPE output_value = output_grad_csr.values[ograd_p];
                    I_grad.values[input_ptr] += W[output_index * input_size + input_index] * output_value;
                }
                // learning synapses: due to math, only inputs that were active can be used
                for (SIZE_TYPE input_ptr = input_csr.ptrs[batch]; input_ptr < input_csr.ptrs[batch + 1]; input_ptr++) {
                    VALUE_TYPE out_wgrad = output_grad_csr.values[ograd_p] * input_csr.values[input_ptr];
                    if (std::abs(out_wgrad) > eps) {
                        W_grad_callback(out_wgrad, output_grad_csr.indices[ograd_p], input_csr.indices[input_ptr]);
                        // example callbacks:
                        //  * set values in a grad_w array the size of w (uses 2x RAM min)
                        //  * (quantized) set values in a quantized grad_w_array with different min/max vals (uses 2X
                        //  RAM min for quantized) and
                        //    * sequentially update a file with full weight/grad/optim values (sequential HDD access is
                        //    much faster)
                        //      WARNING: use a HDD or RAID 10 near HDD array for this. It WILL destroy SSDs.
                        //  * (gpu) build up sparse CSR array with predefined max size using vectors before sending to
                        //  CPU RAM
                        // other examples are much harder to implement and may or may not give good results:
                        //  * immediately use in optim or an optim callback to modify W
                        //  * (quantized) use in immediate optim if grad>threshold or grad*rand>threshold
                        //  * (quantized) set values in a quantized grad_w_array with different min/max vals (uses 2X
                        //  RAM min for quantized) and
                        //    * set values if above threshold
                        //  * store the top x grad values
                        //  * use an unrolled skiplist to create a csr array of grad values for later use
                        //  * (quantized, weight values on HDD) use a parallel unrolled skiplist to store file update
                        //  operations while the hard-drive takes its time in another thread
                    }
                }
            }
        }
    }
}
/* #endregion */


template <class W_CONTAINER, class SIZE_TYPE, class VALUE_TYPE>
class BasicLinearGradUpdater {
public:
    explicit BasicLinearGradUpdater(SIZE_TYPE input_size, SIZE_TYPE output_size) : input_size_(input_size), w_grad() {
        //memset(w_grad.get(), 0, sizeof(VALUE_TYPE)*input_size*output_size);
    }

    ~BasicLinearGradUpdater() = default;

    void update_weight_gradients(VALUE_TYPE out_wgrad, SIZE_TYPE output_index, SIZE_TYPE input_index) {
        w_grad[output_index* input_size_ + input_index] += out_wgrad;
    }

    W_CONTAINER w_grad;
private:
    size_t input_size_;
};

template <class W_CONTAINER, class SIZE_TYPE, class VALUE_TYPE>
std::function<void(VALUE_TYPE, SIZE_TYPE, SIZE_TYPE)> get_basic_linear_W_grad_callback(std::shared_ptr<BasicLinearGradUpdater<W_CONTAINER, SIZE_TYPE, VALUE_TYPE>> updater) {
    return [updater](VALUE_TYPE out_wgrad, SIZE_TYPE output_index, SIZE_TYPE input_index) {
        updater->update_weight_gradients(out_wgrad, output_index, input_index);
    };
}


#endif