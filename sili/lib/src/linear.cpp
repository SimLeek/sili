#include "csr.h"
#include "scan.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numbers>
#include <omp.h>
#include <random>
#include <thread>
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
    // int end = vec_assign_locs[tid+1];
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
void linear_backward_sidlso(int batch_size,
                            int input_size,
                            int output_size,
                            csr_struct<SIZE_TYPE, VALUE_TYPE> &input_csr,
                            W_CONTAINER& W,
                            csr_struct<SIZE_TYPE, VALUE_TYPE> &output_grad_csr,
                            csr_struct<SIZE_TYPE, VALUE_TYPE> &I_grad,
                            std::function<void(float, int, int)> W_grad_callback,
                            const int num_cpus=omp_get_num_procs(),
                            float eps = std::numeric_limits<float>::epsilon()) {

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

class WeightGradUpdater {
public:
    explicit WeightGradUpdater(size_t input_size, size_t output_size) : input_size_(input_size), w_grad(new float[input_size*output_size]{}) {
        //memset(w_grad.get(), 0, sizeof(float)*input_size*output_size);
    }

    ~WeightGradUpdater() = default;

    void update_weight_gradients(float out_wgrad, int output_index, int input_index) {
        w_grad[output_index* input_size_ + input_index] += out_wgrad;
    }

    std::unique_ptr<float[]> w_grad;
private:
    size_t input_size_;
};

std::function<void(float, int, int)> get_dense_W_grad_callback(std::shared_ptr<WeightGradUpdater> updater) {
    return [updater](float out_wgrad, int output_index, int input_index) {
        updater->update_weight_gradients(out_wgrad, output_index, input_index);
    };
}


/* #region Linear Sparse IO Mask */
template <class SIZE_TYPE, class VALUE_TYPE>
class CSRMask {
  private:
    std::default_random_engine generator;             // Random number generator
    std::uniform_real_distribution<VALUE_TYPE> value_dist; // Distribution for random values
    std::uniform_int_distribution<SIZE_TYPE> index_dist;    // Distribution for random indices

  public:
    csr_struct<SIZE_TYPE, VALUE_TYPE> &csrMatrix;

    // Constructor to initialize CSR matrix handler
    CSRMask(csr_struct<SIZE_TYPE, VALUE_TYPE> &csr_matrix)
        : csrMatrix(csr_matrix), generator(static_cast<unsigned>(std::time(0))), value_dist(0.0f, std::numbers::pi * 2),
          index_dist(0, csr_matrix.rows * csr_matrix.cols - 1) {
            if(csrMatrix.ptrs==nullptr){
                csrMatrix.ptrs.reset(new SIZE_TYPE[csrMatrix.rows+1]{});
            }
          }

    void iterate(SIZE_TYPE nnz, VALUE_TYPE min = 0, VALUE_TYPE max = 2 * std::numbers::pi / 50000){
        addRandomValue(min, max);  // add small floats to every values
        addRandomElements(nnz-csrMatrix.nnz());  // maintain exactly nnz values by inserting 0s
    }

    // Method to add a small random value to each CSR value
    void addRandomValue(VALUE_TYPE min = 0, VALUE_TYPE max = 2 * std::numbers::pi / 50000) {
        std::uniform_real_distribution<VALUE_TYPE> small_value_dist(min, max);
        for (SIZE_TYPE i = 0; i < csrMatrix.nnz(); ++i) {
            csrMatrix.values[i] += small_value_dist(generator);
            if (csrMatrix.values[i] > std::numbers::pi * 2) {
                removeElement(i);
            }
        }
    }

    // Helper method to remove an element from the CSR matrix
    void removeElement(SIZE_TYPE index) {
        // properly erase a value in the csr
        std::move(csrMatrix.values.get() + index + 1, csrMatrix.values.get() + csrMatrix.nnz(), csrMatrix.values.get() + index);
        std::move(csrMatrix.indices.get() + index + 1, csrMatrix.indices.get() + csrMatrix.nnz(), csrMatrix.indices.get() + index);

        // Update ptrs to reflect removal
        for (SIZE_TYPE i = 1; i < csrMatrix.rows+1; ++i) {
            if (csrMatrix.ptrs[i] > index) {
                csrMatrix.ptrs[i]--;
            }
        }
    }

    // Method to add elements to CSR array with bisection insertion
    void addRandomElements(SIZE_TYPE insertions) {
        // reserve needed space
        if (csrMatrix._reserved_indices_and_values < csrMatrix.nnz() + insertions) {
            SIZE_TYPE *old_indices = csrMatrix.indices.get();
            VALUE_TYPE *old_values = csrMatrix.values.get();
            csrMatrix.indices.release();
            csrMatrix.values.release();

            // Reserve enough space for the new insertions, doubling the current capacity
            SIZE_TYPE new_capacity = std::max(csrMatrix._reserved_indices_and_values * 2, csrMatrix.nnz() + insertions);
            csrMatrix.indices.reset(new SIZE_TYPE[new_capacity]);
            csrMatrix.values.reset(new VALUE_TYPE[new_capacity]);

            // Move existing data to the newly allocated arrays
            std::move(old_indices, old_indices + csrMatrix.nnz(), csrMatrix.indices.get());
            std::move(old_values, old_values + csrMatrix.nnz(), csrMatrix.values.get());

            // Cleanup old arrays
            delete[] old_indices;
            delete[] old_values;

            // Update reserved space tracker
            csrMatrix._reserved_indices_and_values = new_capacity;
        }

        for (SIZE_TYPE i = 0; i < insertions;
             ++i) { // no need to parallelize this. In general you should only be adding ~1-10 at a time at most
            SIZE_TYPE random_index= index_dist(generator);
            SIZE_TYPE remaining_space = csrMatrix.cols * csrMatrix.rows - csrMatrix.nnz();
            SIZE_TYPE pos = random_index % (csrMatrix.nnz() + 1);

            SIZE_TYPE insert_pos = std::distance(csrMatrix.indices.get(), csrMatrix.indices.get() + pos);
            SIZE_TYPE insert_row_max = std::distance(csrMatrix.ptrs.get(), 
                                                    std::upper_bound(csrMatrix.ptrs.get(), csrMatrix.ptrs.get() + csrMatrix.rows, pos)
                                                    )-1;
            SIZE_TYPE insert_row_min = std::distance(csrMatrix.ptrs.get(), 
                                                    std::lower_bound(csrMatrix.ptrs.get(), csrMatrix.ptrs.get() + csrMatrix.rows, pos)
                                                    )-1;
            SIZE_TYPE chosen_row;
            if(insert_row_max>insert_row_min){
                chosen_row = index_dist(generator) % (insert_row_max - insert_row_min) + insert_row_min;
            }else{
                chosen_row = insert_row_min;
            }if(chosen_row<0){
                chosen_row = 0;
            }

            SIZE_TYPE index_before = 0;
            if (pos != 0 && csrMatrix.ptrs[chosen_row]!=pos) {
                index_before = csrMatrix.indices[pos-1];
            }
            SIZE_TYPE index_after = csrMatrix.cols;
            if (pos != csrMatrix.nnz() && csrMatrix.ptrs[chosen_row+1]!=pos) {
                index_after = csrMatrix.indices[pos];
            }
            if ((index_after - index_before <= 1) ||
                (pos == csrMatrix.nnz() && index_before == csrMatrix.cols &&
                 csrMatrix.ptrs[csrMatrix.rows] != csrMatrix.ptrs[csrMatrix.rows - 1])) {
                // insertions++; may loop forever
                continue; // no space for current insertion point
            }

            std::move(csrMatrix.values.get() + pos, csrMatrix.values.get() + csrMatrix.nnz(), csrMatrix.values.get() + pos + 1);
            std::move(csrMatrix.indices.get() + pos, csrMatrix.indices.get() + csrMatrix.nnz(), csrMatrix.indices.get() + pos + 1);

            random_index = index_dist(generator) % (index_after - index_before-1) + index_before+1;

            csrMatrix.indices[pos] = random_index;
            csrMatrix.values[pos] = 0;

            for (SIZE_TYPE i = chosen_row+1; i < csrMatrix.rows+1; ++i) {
                //if (csrMatrix.ptrs[i] > insert_pos) {
                    csrMatrix.ptrs[i]++;
                //}
            }
        }
    }
};
/* #endregion */