#include "../headers/csr.h"
#include "../headers/scan.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <numbers>
#include <omp.h>
#include <random>
#include <thread>
#include <vector>

/* #region Linear Sparse IO Fwd */
inline void _do_linear_sidlso_fwd(int num_cpus,
                                  int input_size,
                                  int output_size,
                                  int batch,
                                  csr_struct &input_csr,
                                  float *W,
                                  std::vector<std::vector<int>> &row_indices_chunks,
                                  std::vector<std::vector<float>> &row_values_chunks,
                                  float eps) {
#pragma omp parallel num_threads(num_cpus)
    {
        int tid = omp_get_thread_num();                           // Get thread ID
        int chunk_size = (output_size + num_cpus - 1) / num_cpus; // Calculate chunk size
        int start = tid * chunk_size;                             // Start index for this thread
        int end = std::min(start + chunk_size,
                           output_size); // End index for this thread

        for (int output_index = start; output_index < end; ++output_index) {
            float out_val = 0;
            for (int input_ptr = input_csr.ptrs[batch]; input_ptr < input_csr.ptrs[batch + 1]; input_ptr++) {
                int input_index = input_csr.indices[input_ptr];
                auto input_value = input_csr.values[input_ptr];

                out_val += W[output_index * input_size + input_index] * input_value;
            }
            if (out_val > eps) {
                row_indices_chunks[tid].push_back(output_index);
                row_values_chunks[tid].push_back(out_val);
            }
        }
    }
}

inline void _assign_spv_chunks_to_batch(int batch,
                                        int num_cpus,
                                        std::vector<size_t> &vec_assign_locs,
                                        std::vector<std::vector<int>> &out_idx,
                                        std::vector<std::vector<float>> &out_val,
                                        std::vector<std::vector<int>> &row_indices_chunks,
                                        std::vector<std::vector<float>> &row_values_chunks,
                                        int &nnz) {
    out_idx[batch].reserve(vec_assign_locs.back());
    out_val[batch].reserve(vec_assign_locs.back());

#pragma omp parallel num_threads(num_cpus) reduction(+ : nnz)
    {
        int tid = omp_get_thread_num(); // Get thread ID
        int start = vec_assign_locs[tid];
        // int end = vec_assign_locs[tid+1];
        nnz += vec_assign_locs[tid + 1];

        std::copy(row_indices_chunks[tid].begin(), row_indices_chunks[tid].end(), out_idx[batch].begin() + start);
        std::copy(row_values_chunks[tid].begin(), row_values_chunks[tid].end(), out_val[batch].begin() + start);
    }
}

csr_struct linear_sidlso(int batch_size,
                         int input_size,
                         int output_size,
                         csr_struct &input_csr,
                         float *W,
                         float eps = std::numeric_limits<float>::epsilon()) {
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware
                                                               // threads

    std::vector<std::vector<int>> out_idx(batch_size);
    std::vector<std::vector<float>> out_val(batch_size);
    int nnz = 0;

    // size 16 to start for cache line optimization
    std::vector<std::vector<int>> row_indices_chunks(num_cpus, std::vector<int>(16));
    std::vector<std::vector<float>> row_values_chunks(num_cpus, std::vector<float>(16));

    for (int batch = 0; batch < batch_size; batch++) {

        _do_linear_sidlso_fwd(
            num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);
        auto vec_assign_locs = fullScanSizes(row_indices_chunks);

        _assign_spv_chunks_to_batch(
            batch, num_cpus, vec_assign_locs, out_idx, out_val, row_indices_chunks, row_values_chunks, nnz);
    }

    auto cs2 = convert_vov_to_cs2(&out_idx, &out_val, nullptr, output_size, batch_size, nnz);
    return cs2;
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
void linear_backward_sidlso(int batch_size,
                            int input_size,
                            int output_size,
                            csr_struct &input_csr,
                            float *W,
                            csr_struct &output_grad_csr,
                            csr_struct &I_grad,
                            std::function<void(float, int, int)> W_grad_callback,
                            float eps = std::numeric_limits<float>::epsilon()) {
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware threads

    for (int batch = 0; batch < batch_size; batch++) {

#pragma omp parallel num_threads(num_cpus)
        {
            int o_grad_size = output_grad_csr.ptrs[batch + 1] - output_grad_csr.ptrs[batch];
            int tid = omp_get_thread_num();                              // Get thread ID
            int chunk_size = (o_grad_size + num_cpus - 1) / num_cpus;    // Calculate chunk size
            int start = tid * o_grad_size + output_grad_csr.ptrs[batch]; // Start index for this thread
            int end =
                std::min(start + chunk_size, o_grad_size) + output_grad_csr.ptrs[batch]; // End index for this thread

            for (int ograd_p = start; ograd_p < end; ++ograd_p) {
                // learning inputs: selected neurons
                for (int input_ptr = I_grad.ptrs[batch]; input_ptr < I_grad.ptrs[batch + 1]; input_ptr++) {
                    int input_index = I_grad.indices[input_ptr];
                    int output_index = output_grad_csr.indices[ograd_p];
                    float output_value = output_grad_csr.values[ograd_p];
                    I_grad.values[input_ptr] += W[output_index * input_size + input_index] * output_value;
                }
                // learning synapses: due to math, only inputs that were active can be used
                for (int input_ptr = input_csr.ptrs[batch]; input_ptr < input_csr.ptrs[batch + 1]; input_ptr++) {
                    float out_wgrad = output_grad_csr.values[ograd_p] * I_grad.values[input_ptr];
                    if (out_wgrad > eps) {
                        W_grad_callback(out_wgrad, output_grad_csr.indices[ograd_p], I_grad.indices[input_ptr]);
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

/* #region Linear Sparse IO Mask */

class CSRMask {
  private:
    std::default_random_engine generator;             // Random number generator
    std::uniform_real_distribution<float> value_dist; // Distribution for random values
    std::uniform_int_distribution<int> index_dist;    // Distribution for random indices

  public:
    csr_struct &csrMatrix;

    // Constructor to initialize CSR matrix handler
    CSRMask(csr_struct &csr_matrix)
        : csrMatrix(csr_matrix), generator(static_cast<unsigned>(std::time(0))), value_dist(0.0f, std::numbers::pi * 2),
          index_dist(0, csr_matrix.rows * csr_matrix.cols - 1) {}

    // Method to add a small random value to each CSR value
    void addRandomValue(float min = 0, float max = 2 * std::numbers::pi / 50000) {
        std::uniform_real_distribution<float> small_value_dist(min, max);
        for (int i = 0; i < csrMatrix.nnz; ++i) {
            csrMatrix.values[i] += small_value_dist(generator);
            if (csrMatrix.values[i] > std::numbers::pi * 2) {
                removeElement(i);
            }
        }
    }

    // Helper method to remove an element from the CSR matrix
    void removeElement(int index) {
        // properly erase a value in the csr
        std::move(csrMatrix.values + index + 1, csrMatrix.values + csrMatrix.nnz, csrMatrix.values + index);
        std::move(csrMatrix.indices + index + 1, csrMatrix.indices + csrMatrix.nnz, csrMatrix.indices + index);
        csrMatrix.nnz--;

        // Update ptrs to reflect removal
        for (int i = 1; i < csrMatrix.rows; ++i) {
            if (csrMatrix.ptrs[i] > index) {
                csrMatrix.ptrs[i]--;
            }
        }
    }

    // Method to add elements to CSR array with bisection insertion
    void addRandomElements(int insertions) {
        for (int i = 0; i < insertions;
             ++i) { // no need to parallelize this. In general you should only be adding ~1-10 at a time at most
            int random_index = index_dist(generator);
            int remaining_space = csrMatrix.cols * csrMatrix.rows - csrMatrix.nnz;
            int pos = random_index % (csrMatrix.nnz + 1);

            int insert_pos = std::distance(csrMatrix.indices, csrMatrix.indices + pos);

            int index_before = 0;
            if (pos != 0) {
                index_before = csrMatrix.indices[pos - 1];
            }
            int index_after = csrMatrix.cols;
            if (pos != csrMatrix.nnz) {
                index_after = csrMatrix.indices[pos];
            }
            if ((index_after - index_before == 1) ||
                (pos == csrMatrix.nnz && index_before == csrMatrix.cols &&
                 csrMatrix.ptrs[csrMatrix.rows] != csrMatrix.ptrs[csrMatrix.rows - 1])) {
                // insertions++; may loop forever
                continue; // no space for current insertion point
            }

            // properly erase a value in the csr
            if (csrMatrix._reserved_indices_and_values < csrMatrix.nnz) {
                // other initializers may ignore reserved info, but in that case it will generally be equal to nnz
                csrMatrix._reserved_indices_and_values = csrMatrix.nnz;
            }
            if (csrMatrix._reserved_indices_and_values < csrMatrix.nnz + 1) {
                int *old_indices = csrMatrix.indices;
                float *old_values = csrMatrix.values;
                csrMatrix.indices = new int[csrMatrix._reserved_indices_and_values * 2];
                csrMatrix.values = new float[csrMatrix._reserved_indices_and_values * 2];
                std::move(old_indices, old_indices + csrMatrix.nnz, csrMatrix.indices);
                std::move(old_values, old_values + csrMatrix.nnz, csrMatrix.values);
                delete[] old_indices;
                delete[] old_values;
            }
            std::move(csrMatrix.values + pos, csrMatrix.values + csrMatrix.nnz, csrMatrix.values + pos + 1);
            std::move(csrMatrix.indices + pos, csrMatrix.indices + csrMatrix.nnz, csrMatrix.indices + pos + 1);

            random_index = index_dist(generator) % (index_after - index_before) + index_before;
            csrMatrix.nnz++;
            csrMatrix.indices[pos] = random_index;
            csrMatrix.values[pos] = 0;

            for (int i = 1; i < csrMatrix.rows; ++i) {
                if (csrMatrix.ptrs[i] > random_index) {
                    csrMatrix.ptrs[i]++;
                }
            }
        }
    }
};
/* #endregion */