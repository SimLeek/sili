#ifndef _fiber_hpp
#define _fiber_hpp

#include "csr.hpp"
#include "parallel.hpp"
#include "sparse_struct.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>


/**
 * @file
 * @brief fiber expansion and contraction to support growable sparse networks
 *
 * @section theoretical_background Theoretical Background
 *
 * @subsection problem_statement Problem Statement
 * Enable gradient flow in zero-initialized neural networks (feedforward or recurrent) with skip connections, where hidden dimensions exceed input/output sizes.
 *
 * @subsection proof Proof of Mapper Necessity
 * \textbf{Lemma 1: Zero Gradients for Unmapped Units} \\
 * \textit{Statement}: In a zero-initialized network, hidden units not directly connected to inputs or outputs via non-zero weights have zero gradients. \\
 * \textit{Proof}: For a hidden unit \( h_i \) (where \( i > n \) in feedforward, or any unit in RNN at \( t = 0 \)): \\
 * - Initial value: \( h_i = 0 \) (since \( \mathbf{W}_{xh} \mathbf{x} = 0 \), and skip connection contributes 0 for \( i > n \)). \\
 * - Gradient: \( \frac{\partial L}{\partial h_i} = \sum_j \frac{\partial L}{\partial y_j} W_{hy}[j,i] = 0 \), since \( \mathbf{W}_{hy} = \mathbf{0} \). \\
 * - Thus, \( \frac{\partial L}{\partial \mathbf{W}_{xh}[:,i]} = 0 \), and \( h_i \) receives no updates.
 *
 * \textbf{Lemma 2: Skip Connections Don’t Generate New Gradients} \\
 * \textit{Statement}: Skip connections propagate existing gradients but don’t create non-zero gradients for units with zero initial gradients. \\
 * \textit{Proof}: \\
 * - Feedforward: \( \frac{\partial L}{\partial h_i} = \mathbf{W}_{hy}^T \frac{\partial L}{\partial \mathbf{y}} + \frac{\partial L}{\partial \mathbf{x}_{\text{padded}}[i]} \). If \( i > n \), \( \mathbf{x}_{\text{padded}}[i] = 0 \), and \( \mathbf{W}_{hy} = \mathbf{0} \), so \( \frac{\partial L}{\partial h_i} = 0 \). \\
 * - RNN: \( \frac{\partial L}{\partial \mathbf{h}_{t-1}} = \mathbf{W}_{hh}^T \frac{\partial L}{\partial \mathbf{h}_t} \cdot f'(\cdot) + \frac{\partial L}{\partial \mathbf{h}_t} \). With \( \mathbf{W}_{hh} = \mathbf{0} \) and \( \frac{\partial L}{\partial \mathbf{h}_t} = \mathbf{0} \), the gradient remains zero.
 *
 * \textbf{Theorem: Necessity of the Mapper} \\
 * \textit{Statement}: A mapping function is necessary to ensure all hidden units in a zero-initialized network with skip connections receive non-zero gradients over time. \\
 * \textit{Proof}: \\
 * - By Lemma 1, unmapped hidden units have \( \frac{\partial L}{\partial h_i} = 0 \) initially. \\
 * - By Lemma 2, skip connections propagate gradients but don’t activate unmapped units. \\
 * - Without mapping or tiling, weights remain zero, as \( \Delta \mathbf{W} \propto \frac{\partial L}{\partial \mathbf{W}} = 0 \). \\
 * - A mapper dynamically connects \( \mathbf{x} \) to all \( \mathbf{h} \) indices, ensuring \( \frac{\partial L}{\partial h_i} \neq 0 \) optimally.
 *
 * @subsection conclusion Conclusion
 * The mapper is the optimal solution to ensure gradient flow in zero-initialized networks, avoiding tiling’s inefficiencies.
 */


 /**
 * @param v - sorted vector instance
 * @param data - value to search
 * @return 0-based index if data found, -1 otherwise
*/
template<class _FIter, class _Tp>
int binary_search_find_index(_FIter start, _FIter end,  const _Tp & data) {
    auto it = std::lower_bound(start, end, data);
    if (it == end || *it != data) {
        return end;
    } else {
        //_FIter index = start + std::distance(start, it);
        return it;
    }   
}


/**
 * @brief Expands a sparse input tensor in CSR format based on a mapping.
 *
 * This function takes a sparse tensor in CSR format and expands it by replicating each value across a range of output indices defined by `map_ptrs`. The operation is parallelized using OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., `size_t`).
 * @tparam VALUE_TYPE Type for tensor values (e.g., `double`, `float`).
 * @param input_tensor Input sparse tensor in CSR format.
 * @param map_ptrs Vector defining output index ranges for each input index.
 * @param importances Array of importance updates (always expanded size), updated by adding contribution values.
 * @param num_cpus Number of CPU threads (default: 4).
 * @return Expanded sparse tensor in CSR format.
 */
template <typename SIZE_TYPE, typename VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>> fiber_expand_forward(
    const CSRInput<SIZE_TYPE, VALUE_TYPE> &input_tensor,
    const std::vector<SIZE_TYPE> &map_ptrs,
    VALUE_TYPE* importances,
    int num_cpus = 4) {
    SIZE_TYPE num_rows = input_tensor.rows;              // Number of batches
    SIZE_TYPE num_input_indices = input_tensor.cols;     // Number of input features
    SIZE_TYPE num_outputs = map_ptrs[num_input_indices]; // Total output indices

    // Step 1: Compute non-zero counts per batch
    std::vector<SIZE_TYPE> count(num_rows, 0);
    std::shared_ptr<SIZE_TYPE> output_ptrs(new SIZE_TYPE[num_rows + 1]);
    output_ptrs[0] = 0;

    std::shared_ptr<SIZE_TYPE> output_indices;
    std::shared_ptr<VALUE_TYPE> output_values;

#pragma omp parallel num_threads(num_cpus) shared(count, output_ptrs, output_indices, output_values)
    {
        SIZE_TYPE tid = omp_get_thread_num();
        SIZE_TYPE num_threads = omp_get_num_threads();

        // Thread-local count accumulation
        std::vector<SIZE_TYPE> thread_count(num_rows, 0);

        // Step 1: Compute each thread's contribution to count
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
            SIZE_TYPE row_start = input_tensor.ptrs[0][batch];
            SIZE_TYPE row_end = input_tensor.ptrs[0][batch + 1];
            SIZE_TYPE row_len = row_end - row_start;

            // Divide the row among threads
            SIZE_TYPE chunk_size = (row_len + num_threads - 1) / num_threads;
            SIZE_TYPE start = row_start + tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, row_end);

            SIZE_TYPE local_count = 0;
            for (SIZE_TYPE input_ptr = start; input_ptr < end; input_ptr++) {
                SIZE_TYPE input_index = input_tensor.indices[0][input_ptr];
                SIZE_TYPE range_len = map_ptrs[input_index + 1] - map_ptrs[input_index];
                local_count += range_len;
            }
            thread_count[batch] = local_count;
        }

// Reduction to compute total count per batch
#pragma omp reduction(+ : count[ : num_rows])
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
            count[batch] += thread_count[batch];
        }

        // Step 2: Compute output pointers and allocate memory (single-threaded)
        if (tid == 0) {
            SIZE_TYPE total_nnz = 0;
            for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
                total_nnz += count[batch];
                output_ptrs[batch + 1] = total_nnz;
            }
            output_indices.reset(new SIZE_TYPE[total_nnz]);
            output_values.reset(new VALUE_TYPE[total_nnz]);
        }
#pragma omp barrier

        // Step 3: Fill output indices and values without synchronization
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
            SIZE_TYPE batch_start = output_ptrs[batch];
            SIZE_TYPE row_start = input_tensor.ptrs[0][batch];
            SIZE_TYPE row_end = input_tensor.ptrs[0][batch + 1];
            SIZE_TYPE row_len = row_end - row_start;

            // Divide the row among threads
            SIZE_TYPE chunk_size = (row_len + num_threads - 1) / num_threads;
            SIZE_TYPE start = row_start + tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, row_end);

            // Thread-local storage
            std::vector<SIZE_TYPE> local_indices;
            std::vector<VALUE_TYPE> local_values;
            local_indices.reserve(thread_count[batch]);
            local_values.reserve(thread_count[batch]);

            // Compute thread-specific offset and fill thread-local vectors in one pass
            SIZE_TYPE thread_offset = 0;
            for (SIZE_TYPE input_ptr = row_start; input_ptr < end; input_ptr++) {
                SIZE_TYPE input_index = input_tensor.indices[0][input_ptr];
                SIZE_TYPE map_start = map_ptrs[input_index];
                SIZE_TYPE map_end = map_ptrs[input_index + 1];
                SIZE_TYPE range_len = map_end - map_start;

                if (input_ptr < start) {
                    // Before this thread's chunk: only accumulate offset
                    thread_offset += range_len;
                } else {
                    // Within this thread's chunk: fill local vectors
                    VALUE_TYPE v = input_tensor.values[0][input_ptr];
                    for (SIZE_TYPE output_index = map_start; output_index < map_end; output_index++) {
                        local_indices.push_back(output_index);
                        local_values.push_back(v);
                        if(importances!=nullptr){
                            importances[output_index] += v;  // importance update, if applicable
                        }
                    }
                }
            }

            // Write to global arrays at precomputed position
            SIZE_TYPE write_start = batch_start + thread_offset;
            std::copy(local_indices.begin(), local_indices.end(), output_indices.get() + write_start);
            std::copy(local_values.begin(), local_values.end(), output_values.get() + write_start);
        }
    }

    // Step 4: Return the CSR structure
    return create_csr(num_rows, num_outputs, output_ptrs, output_indices, output_values);
}

/** 
 * Sum importances from fiber_expand_forward into the contracted importances.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for tensor values (e.g., double, float).
 * @param importances_expanded Contributions of mapped neurons (size input_tensor.cols).
 * @param map_ptrs Mapping array where map_ptrs[i] to map_ptrs[i+1] defines mapped indices for original neuron i.
 * @param num_input_indices Number of original neurons.
 * @param importances_contracted Array of neuron importances (always expanded size). .
 * @param num_cpus Number of CPU threads to use.
 */
 template <typename SIZE_TYPE, typename VALUE_TYPE>
 void sum_contributions_to_original(
     const VALUE_TYPE* importances_expanded,
     const std::vector<SIZE_TYPE>& map_ptrs,
     SIZE_TYPE num_input_indices,
     VALUE_TYPE* importances_contracted,
     const int num_cpus = 4
 ) {
     #pragma omp parallel for num_threads(num_cpus) reduction(+:importances_contracted[:num_input_indices])
     for (SIZE_TYPE original_index = 0; original_index < num_input_indices; original_index++) {
         VALUE_TYPE sum = 0;
         for (SIZE_TYPE mapped_ptr = map_ptrs[original_index]; 
              mapped_ptr < map_ptrs[original_index + 1]; 
              mapped_ptr++) 
         {
             SIZE_TYPE mapped_index = mapped_ptr; // Direct indexing into mapped_contributions
             sum += importances_expanded[mapped_index];
         }
         importances_contracted[original_index] += sum;
     }
 }

void decay_pass(float* importance, size_t num_elements, float delta) {
#pragma omp parallel for  // Optional: parallelize with OpenMP
    for (size_t i = 0; i < num_elements; i++) {
        float imp = importance[i];
        importance[i] = imp * (1 - delta * exp(-fabs(imp)));
    }
}

/**
 * @brief Contracts an expanded sparse tensor back to original indices using aggregation.
 *
 * Contracts an expanded tensor back to original indices via `map_ptrs`, aggregating values with a customizable function (default: average). Parallelized with OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices.
 * @tparam VALUE_TYPE Type for tensor values.
 * @param expanded_tensor Expanded sparse tensor in CSR format.
 * @param map_ptrs Vector mapping output indices back to input indices.
 * @param importances Array of importance updates (always expanded size), updated by adding contribution values.
 * @param num_cpus Number of CPU threads (default: 4).
 * @return Contracted sparse tensor in CSR format.
 */
template <typename SIZE_TYPE, typename VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>> fiber_contract_forward(
    const CSRInput<SIZE_TYPE, VALUE_TYPE> &expanded_tensor,
    const std::vector<SIZE_TYPE> &map_ptrs,
    VALUE_TYPE* importances,
    int num_cpus = 4
) {
    SIZE_TYPE num_rows = expanded_tensor.rows;         // Number of batches (e.g., 1-2)
    SIZE_TYPE num_output_indices = map_ptrs.size() - 1; // Number of contracted output indices

    std::shared_ptr<SIZE_TYPE[]> csr_output_ptrs(new SIZE_TYPE[num_rows + 1]);
    csr_output_ptrs[0] = 0;

    std::shared_ptr<SIZE_TYPE[]> csr_output_indices;
    std::shared_ptr<VALUE_TYPE[]> csr_output_values;

    std::vector<std::shared_ptr<SIZE_TYPE[]>> output_indices(num_rows);
    std::vector<std::shared_ptr<SIZE_TYPE[]>> input_ptrs(num_rows);

    std::vector<std::unique_ptr<SIZE_TYPE[]>> output_indices_scan(num_rows);
    std::vector<std::unique_ptr<SIZE_TYPE[]>> output_ptrs_scan(num_rows);
    std::vector<SIZE_TYPE> output_scan_sizes(num_rows);

#pragma omp parallel num_threads(num_cpus) shared(output_ptrs, output_indices, output_values)
    {
        SIZE_TYPE tid = omp_get_thread_num();
        SIZE_TYPE num_threads = omp_get_num_threads();

        //std::vector<std::vector<SIZE_TYPE>> thread_counts(
        //    num_rows, std::vector<SIZE_TYPE>(num_output_indices));
        std::vector<std::vector<SIZE_TYPE>> thread_output_indices(
            num_rows, std::vector<SIZE_TYPE>(num_output_indices));
        std::vector<std::vector<SIZE_TYPE>> thread_input_pointers(
            num_rows, std::vector<SIZE_TYPE>(num_output_indices));

        // Count unique input indices per batch
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
            SIZE_TYPE row_start = expanded_tensor.ptrs[0][batch];
            SIZE_TYPE row_end = expanded_tensor.ptrs[0][batch + 1];
            SIZE_TYPE row_len = row_end - row_start;

            // Parallelize over the row’s non-zero elements
            auto it = std::upper_bound(
                map_ptrs.get(), map_ptrs.get() + num_output_indices + 1, expanded_tensor.indices[0][row_start]);
            auto it_end = std::upper_bound(
                map_ptrs.get(), map_ptrs.get() + num_output_indices + 1, expanded_tensor.indices[0][row_end]);
            int chunk_size = (std::distance(it, it_end) - 1) / num_threads;

            for (auto it_local = it + chunk_size * tid; it_local < it_end; it_local++) {
                // Map back to original input index
                SIZE_TYPE output_index = std::distance(map_ptrs.get(), it_local) - 1;
                auto it_index = binary_search_find_index(expanded_tensor.indices[0] + row_start,
                                                expanded_tensor.indices[0] + row_end,
                                                map_ptrs[output_index]);
                auto loc = std::distance(expanded_tensor.indices[0] + row_start, it_index);
                                                
                if (it_index!=expanded_tensor.indices[0] + row_end) {
                    thread_output_indices[batch].push_back(output_index);
                    thread_input_pointers[batch].push_back(SIZE_TYPE(row_start + loc));
                    //thread_counts[batch].push_back(map_ptrs[output_index + 1] - map_ptrs[output_index]);
                    if(importances!=nullptr){
                        importances[*(expanded_tensor.indices[0] + SIZE_TYPE(row_start + loc))] += *(expanded_tensor.values[0] + SIZE_TYPE(row_start + loc)); // add source value contribution to importance. Shouldn't require a critical section ever
                    }
                }
            }

            for (SIZE_TYPE i = 0; i < chunk_size && i < thread_output_indices[batch].size(); i++) {
                output_indices[batch][tid * chunk_size + i] = thread_output_indices[batch][i]; // parallel concat
                input_ptrs[batch][tid * chunk_size + i] = thread_input_pointers[batch][i]; // parallel concat
            }
            if (tid == num_threads - 1) {
                output_indices[batch].resize(tid * chunk_size + thread_output_indices[batch].size());
                input_ptrs[batch].resize(tid * chunk_size + thread_input_pointers[batch].size());
            }
        }

        // Reduce thread-local counts into global output_count
        for(SIZE_TYPE batch=0;batch<num_rows;batch++){
            reduce_unique_with_pointers(
                output_indices[batch].data(), 
                output_indices[batch].size(),
                output_indices_scan[batch], 
                output_ptrs_scan[batch], 
                output_scan_sizes[batch], 
                num_cpus);
        }

        // Allocate output arrays once total NNZ is known
        if (tid == 0) {
            int nnz = std::accumulate(output_scan_sizes.begin(), output_scan_sizes.end(), 0);
            csr_output_indices = std::shared_ptr<SIZE_TYPE[]>(new SIZE_TYPE[nnz]);
            csr_output_values = std::shared_ptr<VALUE_TYPE[]>(new VALUE_TYPE[nnz]);
        }
        #pragma omp barrier

        // Parallelize over the output tensor’s non-zero elements
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
            if (batch==0){
                csr_output_ptrs[batch] = 0;
            }else{
                csr_output_ptrs[batch] = csr_output_ptrs[batch-1] + output_scan_sizes[batch-1];
            }
#pragma omp for
            for (SIZE_TYPE ptr_i = 0; ptr_i < output_scan_sizes[batch]; ptr_i++) {
                // Find the batch this output position belongs to
                auto scan_from = output_ptrs_scan[batch][ptr_i];
                auto scan_to = output_ptrs_scan[batch][ptr_i+1];
                auto scan_index = output_indices_scan[batch][ptr_i];
                VALUE_TYPE sum=0;
                //aggregate() should be used instead of this for loop... but idk
                for(SIZE_TYPE i=scan_from; i<scan_to;i++){
                    sum+=expanded_tensor.values[0][input_ptrs[batch][i]];
                }
                csr_output_values[csr_output_ptrs[batch]+ptr_i] = sum;
                csr_output_indices[csr_output_ptrs[batch]+ptr_i] = scan_index;
            }            
        }
    }

    // Step 4: Return the CSR structure
    return create_csr(num_rows, num_output_indices, csr_output_ptrs, csr_output_indices, csr_output_values);
}

/**
 * @brief Distributes importances from contracted outputs to expanded input neurons.
 *
 * Given importances for contracted outputs, this function distributes each importance equally
 * to the expanded neurons that contributed to it, dividing by the number of expanded neurons
 * per original index as defined by map_ptrs. Parallelized with OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for values (e.g., double, float).
 * @param contracted_importances Importances of contracted outputs (size num_input_indices).
 * @param map_ptrs Mapping array where map_ptrs[i] to map_ptrs[i+1] defines expanded indices for original index i.
 * @param num_input_indices Number of original indices.
 * @param num_expanded_neurons Total number of expanded neurons.
 * @param expanded_contributions Output array for contributions to expanded neurons (size num_expanded_neurons).
 * @param num_cpus Number of CPU threads (default: 4).
 */
 template <typename SIZE_TYPE, typename VALUE_TYPE>
 void divide_contributions_to_inputs(
     const VALUE_TYPE* contracted_importances,
     const std::vector<SIZE_TYPE>& map_ptrs,
     SIZE_TYPE num_input_indices,
     SIZE_TYPE num_expanded_neurons,
     VALUE_TYPE* expanded_importances,
     const int num_cpus = 4
 ) {
     #pragma omp parallel for num_threads(num_cpus)
     for (SIZE_TYPE i = 0; i < num_input_indices; i++) {
         SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
         VALUE_TYPE contrib = contracted_importances[i] / k_i;
         for (SIZE_TYPE j = map_ptrs[i]; j < map_ptrs[i + 1]; j++) {
             expanded_importances[j] = contrib;
         }
     }
 }

 /**
 * @brief Computes the backward pass for fiber expansion.
 *
 * This function computes the gradient for the input tensor by contracting the gradient of the expanded tensor
 * using sum aggregation via `fiber_contract_forward`. It also updates the importance values by subtracting
 * the gradient values from `grad_input`. The operation is parallelized with OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for tensor values (e.g., double, float).
 * @param map_ptrs Vector defining the mapping from original to expanded indices.
 * @param grad_expanded Gradient of the expanded tensor in CSR format.
 * @param grad_input Output gradient for the input tensor in CSR format, computed and returned.
 * @param importances Array of importance updates (always expanded size), updated by subtracting grad_expanded values.
 * @param num_cpus Number of CPU threads to use (default: 4).
 */
 template <typename SIZE_TYPE, typename VALUE_TYPE>
 void fiber_expand_backward(
     const std::vector<SIZE_TYPE>& map_ptrs,
     const sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>>& grad_expanded,
     sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>>& grad_input,
     VALUE_TYPE* importances,
     int num_cpus = 4
 ) {
     #pragma omp parallel for num_threads(num_cpus)
     for (SIZE_TYPE b = 0; b < grad_expanded.rows; b++) {
         for (SIZE_TYPE ptr = grad_expanded.ptrs[0][b]; ptr < grad_expanded.ptrs[0][b + 1]; ptr++) {
             SIZE_TYPE i = grad_expanded.indices[0][ptr]; // Index in original input space
             VALUE_TYPE grad_value = grad_expanded.values[0][ptr];
             importances[i] -= grad_value;
         }
     }
 
     // Use fiber_contract_forward to sum gradients back to original indices
     grad_input = fiber_contract_forward(grad_expanded, map_ptrs, nullptr, num_cpus);
 
    // Compute importance updates using grad_input
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE b = 0; b < grad_input.rows; b++) {
        for (SIZE_TYPE ptr = grad_input.ptrs[0][b]; ptr < grad_input.ptrs[0][b + 1]; ptr++) {
            SIZE_TYPE i = grad_input.indices[0][ptr]; // Index in original input space
            VALUE_TYPE grad_value = grad_input.values[0][ptr];
        }
    }

 }

 /**
 * @brief Computes the backward pass for fiber contraction.
 *
 * This function computes the gradient for the expanded tensor by pre-scaling the gradient of the contracted tensor
 * by 1/k_i (where k_i is the number of expanded indices per original index) and expanding it using `fiber_expand_forward`.
 * It also updates the importance values for expanded neurons by subtracting the gradient values from `grad_expanded`.
 * The operation is parallelized with OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for tensor values (e.g., double, float).
 * @param map_ptrs Vector defining the mapping from original to expanded indices.
 * @param grad_contracted Gradient of the contracted tensor in CSR format.
 * @param grad_expanded Output gradient for the expanded tensor in CSR format, computed and returned.
 * @param importances Array of importance updates (always expanded size), updated by subtracting grad_expanded values.
 * @param num_cpus Number of CPU threads to use (default: 4).
 */
 template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_backward(
    const std::vector<SIZE_TYPE>& map_ptrs,
    const sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>& grad_contracted,
    sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>& grad_expanded,
    VALUE_TYPE* importances,
    int num_cpus = 4
) {
    SIZE_TYPE num_rows = grad_contracted.rows;
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1;

    // Pre-scale grad_contracted by 1/k_i
    auto scaled_values = std::make_shared<VALUE_TYPE[]>(grad_contracted.nnz());
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE ptr = 0; ptr < grad_contracted.values[0].size(); ptr++) {
        SIZE_TYPE i = grad_contracted.indices[0][ptr]; // Index in contracted tensor
        SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i]; // Number of expanded indices
        scaled_values[ptr] = grad_contracted.values[0][ptr] / k_i;
    }

    // Create a new sparse tensor with scaled values
    auto scaled_grad_contracted = create_csr(
        num_rows,
        num_input_indices,
        grad_contracted.ptrs[0],
        grad_contracted.indices[0],
        scaled_values
    );

    // Expand the pre-scaled gradients
    grad_expanded = fiber_expand_forward(scaled_grad_contracted, map_ptrs, num_cpus);

    // Compute importance updates using grad_expanded
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE b = 0; b < grad_expanded.rows; b++) {
        for (SIZE_TYPE ptr = grad_expanded.ptrs[0][b]; ptr < grad_expanded.ptrs[0][b + 1]; ptr++) {
            SIZE_TYPE j = grad_expanded.indices[0][ptr]; // Index in expanded space
            VALUE_TYPE grad_value = grad_expanded.values[0][ptr];
            importances[j] -= grad_value;
        }
    }
}

/**
 * @brief Merges duplicate entries in a specified row of the sparse weights.
 *
 * This function sorts the indices and values for the specified row range (defined by `updated_ptrs[row_to_merge-1]` to `updated_ptrs[row_to_merge]`)
 * and merges duplicate indices by summing their values. It operates on a single row and is used in optimization routines.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for weight values (e.g., double, float).
 * @param weights Sparse linear weights structure, modified in place.
 * @param updated_ptrs Updated pointers defining row ranges in the weights.
 * @param row_to_merge The row index to merge duplicates in (must be > 0).
 */
template <typename SIZE_TYPE, typename VALUE_TYPE> 
SIZE_TYPE merge_duplicate_row_with_next(
    CSRSynapses<SIZE_TYPE, VALUE_TYPE>& weights,
    std::shared_ptr<SIZE_TYPE[]> updated_ptrs,
    SIZE_TYPE row_to_merge
){
    if(row_to_merge<=0){
        return;
    }
    //sort weights and indices, not really in parallel, this is just the only function I have that does it
    omp_sort_ascending(updated_ptrs[row_to_merge+1]-updated_ptrs[row_to_merge], weights.indices.data()+updated_ptrs[row_to_merge], weights.values.data()+updated_ptrs[row_to_merge]);
    //merge nearby equal indices
    SIZE_TYPE sub = 0;
    for(SIZE_TYPE j=updated_ptrs[row_to_merge]; j<updated_ptrs[row_to_merge+1]; j++){
        weights.indices[j-sub]=weights.indices[j];
        weights.values[j-sub]=weights.values[j];
        if(weights.indices[j]==weights.indices[j+1]){
            weights.values[j+1]+=weights.values[j];
            sub+=1;
        }
    }
    return sub;
}

/**
 * @brief Optimizes the fiber expansion by adjusting the mapping based on importance.
 *
 * This function computes the average importance for each original index, selects the top-k indices to change
 * (add or remove neurons based on the `add` flag), and updates the mapping (`map_ptrs`) and sparse weights accordingly.
 * When adding neurons, it increases the mapping; when removing, it decreases it and merges duplicates in affected rows.
 * The operation is parallelized with OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for importance values (e.g., double, float).
 * @param map_ptrs Mapping vector to be updated (size num_input_indices + 1).
 * @param importances Array of importance updates (always expanded size), used for growing or shrinking the fiber.
 * @param neurons_to_change Number of neurons to add or remove.
 * @param weights Sparse linear weights to be updated, modified in place.
 * @param add Flag indicating whether to add (true) or remove (false) neurons (default: true).
 * @param num_cpus Number of CPU threads to use (default: 4).
 */
/*template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_expand_optim(
    std::vector<SIZE_TYPE>& map_ptrs,
    const VALUE_TYPE* importances,
    SIZE_TYPE neurons_to_change,
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>* weights=nullptr,
    bool add=true,
    int num_cpus = 4
) {
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1;
    std::vector<VALUE_TYPE> avg_importance(num_input_indices);
    std::vector<SIZE_TYPE> add_map_ptrs(num_input_indices + 1);

    // Compute average importance
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_input_indices; i++) {
            VALUE_TYPE sum = 0;
            for (SIZE_TYPE h = map_ptrs[i]; h < map_ptrs[i + 1]; h++) {
                sum += importances[h];
            }
            SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
            if(add){
                avg_importance[i] = (k_i > 0) ? sum / k_i : 0;
            }else{
                avg_importance[i] = (k_i > 0) ? -sum / k_i : 0;
            }
        }
    }

    // compute spots to change, making it a bool map initially
    auto change_spots = top_k_indices(avg_importance, map_ptrs.rows, neurons_to_change, num_cpus);
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < neurons_to_change; i++) {
            add_map_ptrs[change_spots[i]] = 1;
        }
    }

    // convert to pointer addition/subtraction array and add it to the map
    omp_scan_inclusive(add_map_ptrs.data(), add_map_ptrs.data(), add_map_ptrs.size());
    #pragma omp parallel num_threads(num_cpus)
    {
        if(add){
            #pragma omp for
            for (SIZE_TYPE i = 0; i < num_input_indices+1; i++) {
                map_ptrs[i] += add_map_ptrs[i];
            }
        }else{
            #pragma omp for
            for (SIZE_TYPE i = 0; i < num_input_indices+1; i++) {
                map_ptrs[i] -= add_map_ptrs[i];
            }
        }
    }

    //update connected linear weights, if applicable
    if(weights!=nullptr){
        std::unique_ptr<SIZE_TYPE[]> new_csc_ptrs(add? new SIZE_TYPE[weights->connections->rows + (neurons_to_change) + 1]:new SIZE_TYPE[weights->connections->rows - (neurons_to_change) + 1]);
        #pragma omp parallel num_threads(num_cpus)
        {
            if(add){
                #pragma omp for
                for (SIZE_TYPE i = 0; i < weights->connections->rows + (neurons_to_change) + 1; i++) {
                    new_csc_ptrs[i] = weights->connections->ptrs[i-add_map_ptrs[i]];
                }
            } else{
                #pragma omp for
                for (SIZE_TYPE i = 0; i < weights->connections->rows + (neurons_to_change) + 1; i++) {
                    new_csc_ptrs[i] = weights->connections->ptrs[i+add_map_ptrs[i]];
                    if(add_map_ptrs[i]!=add_map_ptrs[i-1]){
                        merge_duplicate_row_with_next(weights, weights->connections->ptrs, i-1);
                    }
                }
            }
        }    
    
        SIZE_TYPE new_H = map_ptrs.back();
        auto new_csc = create_csr(new_H, weights->connections.cols, new_csc_ptrs, weights->connections.connections.indices, weights->connections.connections.values);
        weights->connections = new_csc;
    }
}*/

template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> fiber_expand_optim(
    std::vector<SIZE_TYPE>& map_ptrs,
    const VALUE_TYPE* importances,
    SIZE_TYPE neurons_to_change,
    bool add,
    int num_cpus = 4
) {
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1;
    std::vector<VALUE_TYPE> avg_importance(num_input_indices);
    std::vector<SIZE_TYPE> add_map_ptrs(num_input_indices + 1, 0);

    // Compute average importance
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_input_indices; i++) {
            VALUE_TYPE sum = 0;
            for (SIZE_TYPE h = map_ptrs[i]; h < map_ptrs[i + 1]; h++) {
                sum += importances[h];
            }
            SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
            avg_importance[i] = (k_i > 0) ? (add ? sum / k_i : -sum / k_i) : 0;
        }
    }

    // Select top-k indices to change
    auto change_spots = top_k_indices(avg_importance, num_input_indices, neurons_to_change, num_cpus);

    // Create change map
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < neurons_to_change; i++) {
            add_map_ptrs[change_spots[i]] = 1;
        }
    }

    // Convert to cumulative pointer adjustments
    omp_scan_inclusive(add_map_ptrs.data(), add_map_ptrs.data(), add_map_ptrs.size());

    // Update map_ptrs
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_input_indices + 1; i++) {
            map_ptrs[i] = add ? map_ptrs[i] + add_map_ptrs[i] : map_ptrs[i] - add_map_ptrs[i];
        }
    }

    return add_map_ptrs;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_optim_adjust_rows(
    const std::vector<SIZE_TYPE>& add_map_ptrs,
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*>& weights_array,
    bool add,
    int num_cpus = 4
) {
    for (auto* weights : weights_array) {
        if (weights == nullptr) continue;

        SIZE_TYPE neurons_to_change = add_map_ptrs.back();
        SIZE_TYPE new_rows = add ? weights->connections.rows + neurons_to_change : weights->connections.rows - neurons_to_change;
        std::shared_ptr<SIZE_TYPE[]> new_csc_ptrs(new SIZE_TYPE[new_rows + 1]);

        #pragma omp parallel num_threads(num_cpus)
        {
            if (add) {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < new_rows + 1; i++) {
                    new_csc_ptrs[i] = weights->connections.ptrs[i - add_map_ptrs[i]];
                }
            } else {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < new_rows + 1; i++) {
                    new_csc_ptrs[i] = weights->connections.ptrs[i + add_map_ptrs[i]];
                    if (i > 0 && add_map_ptrs[i] != add_map_ptrs[i - 1]) {
                        merge_duplicate_row_with_next(weights->connections, weights->connections.ptrs, i - 1);
                    }
                }
            }
        }

        auto new_csc = create_csr(new_rows, weights->connections.cols, new_csc_ptrs, weights->connections.indices, weights->connections.values);
        weights->connections = new_csc;
    }
}

//this happens when we have fiber_expand(modified)->fiber_expand(2)->...
//the input needs to be added or deleted, which has its own fibers
//this takes in the add_map and map_ptrs for the fiber_expand(modified)
// and returns the add_map_ptrs needed to adjust fiber_expand(2)
/*template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> modify_fiber_adjust_weights_for_double_expansion(
    const std::vector<SIZE_TYPE>& add_map_ptrs, // add_map_ptrs for fiber_expand(modified)
    std::vector<SIZE_TYPE>& map_ptrs, // map_ptrs for fiber_expand(modified)
    std::vector<SIZE_TYPE>& map_ptrs_2, // map_ptrs for fiber_expand(2)
    bool add,
    int num_cpus = 4
) {
    //case 1: adding x to location i
    //  map_ptrs increases by x at location i
    //  add_map_ptrs_2 sets the position at sum(map_ptrs_2[0 to i+1]) to x, the rest 0
    //  add_map_ptrs_2 is used by fiber_expand_optim_adjust_weights to modify the weights. The weights now correctly handle the new inserted input
    //  map_ptrs_2 inserts x new positions at sum(map_ptrs_2[0 to i+1]), and the new array size is x larger
    //case 2: subtracting x from location i
    //  map_ptrs decreases by x at location i
    //  add_map_ptrs_2 sets the positions from sum(map_ptrs_2[0 to i-x+1]) to sum(map_ptrs_2[0 to i+1]) to -map_ptrs_2[current_index] (deleting whole input)
    //  add_map_ptrs_2 is used by fiber_expand_optim_adjust_weights to modify the weights. The weights now correctly handle the new truncated input
    //  map_ptrs_2 deletes x positions from positions sum(map_ptrs_2[0 to i-x+1]) to sum(map_ptrs_2[0 to i+1]), and the new array size is x*fiber_sizes smaller
}*/

//ideally not needed: for xfmrs a, b going into c, where a and b predict input, use the same fiber contraction to output to c as well, and use a different output network.
//  this other output network will take data from the state, which is given info from the whole transformer, so it works
//  This avoids requiring modifying the automatic graph system to handle sequential fiber expansions
template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> modify_fiber_adjust_weights_for_double_expansion(
    const std::vector<SIZE_TYPE>& add_map_ptrs, // Adjustments for first expansion
    std::vector<SIZE_TYPE>& map_ptrs,           // Pointers for first expansion
    std::vector<SIZE_TYPE>& map_ptrs_2,         // Pointers for second expansion
    bool add,                                   // True for adding, false for removing
    int num_cpus                                // Number of CPU threads
) {
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1;    // Number of inputs in first expansion
    SIZE_TYPE num_expanded = map_ptrs.back();             // Total hidden neurons from first expansion

    // Initialize add_map_ptrs_2 to match map_ptrs_2 size
    std::vector<SIZE_TYPE> add_map_ptrs_2(map_ptrs_2.size(), 0);

    if (add) {
        // Adding neurons in the first expansion
        SIZE_TYPE current_pos = 0; // Tracks position in expanded space
        for (SIZE_TYPE i = 0; i < num_input_indices; i++) {
            SIZE_TYPE original_fiber_size = map_ptrs[i + 1] - map_ptrs[i];
            SIZE_TYPE x = add_map_ptrs[i + 1] - add_map_ptrs[i]; // Neurons added
            if (x > 0) {
                // Position where new neurons are inserted (end of range i before adjustment)
                SIZE_TYPE insert_pos = map_ptrs[i] + original_fiber_size;
                // Each new neuron gets a fiber size; use the last existing fiber size in this range
                SIZE_TYPE fiber_size = 1; // Default minimal size
                if (insert_pos > 0 && insert_pos <= map_ptrs_2.size() - 1) {
                    fiber_size = map_ptrs_2[insert_pos] - map_ptrs_2[insert_pos - 1];
                }
                // Total new hidden neurons in second expansion
                SIZE_TYPE neurons_to_add = x * fiber_size;
                if (insert_pos < add_map_ptrs_2.size() - 1) {
                    add_map_ptrs_2[insert_pos + 1] += neurons_to_add;
                }
            }
            current_pos += original_fiber_size;
        }
        // Compute cumulative sum
        omp_scan_inclusive(add_map_ptrs_2.data(), add_map_ptrs_2.data(), add_map_ptrs_2.size());
    } else {
        // Removing neurons in the first expansion
        for (SIZE_TYPE i = 0; i < num_input_indices; i++) {
            SIZE_TYPE x = add_map_ptrs[i + 1] - add_map_ptrs[i]; // Neurons changed
            if (x > 0) { // x > 0 means neurons are removed in this context
                SIZE_TYPE remove_start = map_ptrs[i];
                SIZE_TYPE remove_end = map_ptrs[i + 1];
                SIZE_TYPE neurons_present = remove_end - remove_start;
                if (x <= neurons_present) {
                    // Remove the last x neurons from this range
                    SIZE_TYPE remove_pos_start = remove_end - x;
                    for (SIZE_TYPE j = remove_pos_start; j < remove_end && j + 1 < add_map_ptrs_2.size(); j++) {
                        SIZE_TYPE fiber_size = map_ptrs_2[j + 1] - map_ptrs_2[j];
                        add_map_ptrs_2[j + 1] = -fiber_size; // Remove this fiber’s hidden neurons
                    }
                }
            }
        }
        // Compute cumulative sum
        omp_scan_inclusive(add_map_ptrs_2.data(), add_map_ptrs_2.data(), add_map_ptrs_2.size());
    }

    return add_map_ptrs_2;
}
/**
 * @brief Merges duplicate columns in the sparse weights matrix.
 *
 * This function compacts the sparse matrix by merging duplicate indices in each row, summing their values,
 * and updating the pointers, indices, and values accordingly. The operation is parallelized with OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for weight values (e.g., double, float).
 * @param weights Sparse linear weights structure, modified in place.
 * @param num_cpus Number of CPU threads to use (default: 4).
 */
template <typename SIZE_TYPE, typename VALUE_TYPE> 
void merge_duplicate_indices(
    CSRSynapses<SIZE_TYPE, VALUE_TYPE>& weights,
    int num_cpus = 4
){
    // todo: move this to csr.hpp
    std::vector<SIZE_TYPE> local_sub(weights.rows, 0);
    std::vector<std::vector<SIZE_TYPE>> compacted_indices_per_row(weights.rows);
    std::vector<std::vector<VALUE_TYPE>> compacted_values_per_row(weights.rows);

    // Compact each row in parallel
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        std::vector<SIZE_TYPE> local_indices;
        std::vector<VALUE_TYPE> local_values;
        SIZE_TYPE sub = 0;
        SIZE_TYPE j = weights.ptrs[i];
        while (j < weights.ptrs[i + 1]) {
            SIZE_TYPE current_idx = weights.indices[j];
            VALUE_TYPE sum = weights.values[j];
            j++;
            while (j < weights.ptrs[i + 1] && weights.indices[j] == current_idx) {
                sum += weights.values[j];
                j++;
                sub++;
            }
            local_indices.push_back(current_idx);
            local_values.push_back(sum);
        }
        compacted_indices_per_row[i] = std::move(local_indices);
        compacted_values_per_row[i] = std::move(local_values);
        local_sub[i] = sub;
    }

    // Compute compacted row lengths
    std::vector<SIZE_TYPE> compacted_row_len(weights.rows);
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        compacted_row_len[i] = compacted_indices_per_row[i].size();
    }

    // Compute new_ptrs using exclusive scan
    std::vector<SIZE_TYPE> new_ptrs(weights.rows + 1);
    new_ptrs[0] = 0;
    omp_scan_exclusive(compacted_row_len.data(), new_ptrs.data() + 1, weights.rows);

    SIZE_TYPE new_nnz = new_ptrs[weights.rows];

    // Compute new indices and values given the new pointers
    std::shared_ptr<SIZE_TYPE> new_indices(new SIZE_TYPE[new_nnz]);
    std::shared_ptr<VALUE_TYPE> new_values(new VALUE_TYPE[new_nnz]);
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        SIZE_TYPE write_start = new_ptrs[i];
        SIZE_TYPE len = compacted_row_len[i];
        for (SIZE_TYPE k = 0; k < len; k++) {
            new_indices[write_start + k] = compacted_indices_per_row[i][k];
            new_values[write_start + k] = compacted_values_per_row[i][k];
        }
    }

    weights.ptrs = std::move(new_ptrs);
    weights.indices = std::move(new_indices);
    weights.values = std::move(new_values);
}

/**
 * @brief Optimizes the fiber contraction by adjusting the mapping based on importance.
 *
 * This function computes the average importance for each output index given its inputs, selects the top-k indices to change
 * (add or remove neurons based on the `add` flag), and updates the mapping (`map_ptrs`) and sparse weights accordingly.
 * When adding neurons, it increases the column indices; when removing, it decreases them and merges duplicate columns.
 * The operation is parallelized with OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for importance values (e.g., double, float).
 * @param map_ptrs Mapping vector to be updated (size num_output_indices + 1).
 * @param importances Array of importance updates (always expanded size), used for growing or shrinking the fiber.
 * @param neurons_to_change Number of neurons to add or remove.
 * @param weights Sparse linear weights to be updated, modified in place.
 * @param add Flag indicating whether to add (true) or remove (false) neurons (default: true).
 * @param num_cpus Number of CPU threads to use (default: 4).
 */
/*template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_optim(
    std::vector<SIZE_TYPE>& map_ptrs,
    const VALUE_TYPE* importances,
    SIZE_TYPE neurons_to_change,
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>* weights=nullptr,
    bool add=true,
    int num_cpus = 4
) {
    SIZE_TYPE num_output_indices = map_ptrs.size() - 1;
    std::vector<VALUE_TYPE> avg_importance(num_output_indices);
    std::vector<SIZE_TYPE> add_map_ptrs(num_output_indices + 1);

    // compute avg importance
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_output_indices; i++) {
            VALUE_TYPE sum = 0;
            for (SIZE_TYPE h = map_ptrs[i]; h < map_ptrs[i + 1]; h++) {
                sum += importances[h];
            }
            SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
            if(add){
                avg_importance[i] = (k_i > 0) ? sum / k_i : 0;
            }else{
                avg_importance[i] = (k_i > 0) ? -sum / k_i : 0;
            }
        }
    }

    // compute spots to change, making it a bool map initially
    auto change_spots = top_k_indices(avg_importance, map_ptrs.rows, neurons_to_change, num_cpus);
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < neurons_to_change; i++) {
            add_map_ptrs[change_spots[i]] = 1;
        }
    }

    // convert bool array to pointer addition/subtraction array
    omp_scan_inclusive(add_map_ptrs.data(), add_map_ptrs.data(), add_map_ptrs.size());

    // apply the pointer addition/subtraction array
    #pragma omp parallel num_threads(num_cpus)
        {
            if(add){
                #pragma omp for
                for (SIZE_TYPE i = 0; i < num_output_indices+1; i++) {
                    map_ptrs[i] += add_map_ptrs[i];
                }
            }else{
                #pragma omp for
                for (SIZE_TYPE i = 0; i < num_output_indices+1; i++) {
                    map_ptrs[i] -= add_map_ptrs[i];
                }
            }
        }

    // modify connected weight array, if applicable
    if(weights!=nullptr){
        if(add){
            weights->connections.cols += neurons_to_change;
        }else{
            weights->connections.cols -= neurons_to_change;
        }
        #pragma omp parallel num_threads(num_cpus)
        {
            if(add){
                #pragma omp for
                for(SIZE_TYPE i=0; i<weights->connections.rows; i++){
                    for (SIZE_TYPE j = weights->connections.ptrs[i]; j < weights->connections.ptrs[i+1]; j++) {
                        weights->connections.indices[j] += add_map_ptrs[weights->connections.indices[j]];
                    }
                }
            } else{
                #pragma omp for
                for(SIZE_TYPE i=0; i<weights->connections.rows; i++){
                    for (SIZE_TYPE j = weights->connections.ptrs[i]; j < weights->connections.ptrs[i+1]; j++) {
                        weights->connections.indices[j] -= add_map_ptrs[weights->connections.indices[j]];
                    }
                }
                merge_duplicate_columns(weights->connections, num_cpus);

                //sequential version of merge duplicate columns. Use this commented code to create tests to compare against, and then delete it.
                /*SIZE_TYPE sub=0;
                for(SIZE_TYPE i=0;i<weights.rows;i++){
                    weights.ptrs[i+1]-=sub;
                    for(SIZE_TYPE j=weights.ptrs[i]; j<weights.ptrs[i+1]; j++){
                        if(weights.indices[j]==weights.indices[j+1]){
                            weights.values[j-sub]+=weights.values[j+1];
                            sub+=1;
                        }
                        weights.indices[j-sub]=weights.indices[j];
                    }
                }*//*
            }
        }
    }
}*/

template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> fiber_contract_optim(
    std::vector<SIZE_TYPE>& map_ptrs,
    const VALUE_TYPE* importances,
    SIZE_TYPE neurons_to_change,
    bool add,
    int num_cpus = 4
) {
    SIZE_TYPE num_output_indices = map_ptrs.size() - 1;
    std::vector<VALUE_TYPE> avg_importance(num_output_indices);
    std::vector<SIZE_TYPE> add_map_ptrs(num_output_indices + 1, 0);

    // Compute average importance
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_output_indices; i++) {
            VALUE_TYPE sum = 0;
            for (SIZE_TYPE h = map_ptrs[i]; h < map_ptrs[i + 1]; h++) {
                sum += importances[h];
            }
            SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
            avg_importance[i] = (k_i > 0) ? (add ? sum / k_i : -sum / k_i) : 0;
        }
    }

    // Select top-k indices to change
    auto change_spots = top_k_indices(avg_importance, num_output_indices, neurons_to_change, num_cpus);

    // Create change map
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < neurons_to_change; i++) {
            add_map_ptrs[change_spots[i]] = 1;
        }
    }

    // Convert to cumulative pointer adjustments
    omp_scan_inclusive(add_map_ptrs.data(), add_map_ptrs.data(), add_map_ptrs.size());

    // Update map_ptrs
    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_output_indices + 1; i++) {
            map_ptrs[i] = add ? map_ptrs[i] + add_map_ptrs[i] : map_ptrs[i] - add_map_ptrs[i];
        }
    }

    return add_map_ptrs;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_optim_adjust_columns(
    const std::vector<SIZE_TYPE>& add_map_ptrs,
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*>& weights_array,
    bool add,
    int num_cpus = 4
) {
    SIZE_TYPE neurons_to_change = add_map_ptrs.back();

    for (auto* weights : weights_array) {
        if (weights == nullptr) continue;

        weights->connections.cols = add ? weights->connections.cols + neurons_to_change : weights->connections.cols - neurons_to_change;

        #pragma omp parallel num_threads(num_cpus)
        {
            if (add) {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < weights->connections.rows; i++) {
                    for (SIZE_TYPE j = weights->connections.ptrs[i]; j < weights->connections.ptrs[i + 1]; j++) {
                        weights->connections.indices[j] += add_map_ptrs[weights->connections.indices[j]];
                    }
                }
            } else {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < weights->connections.rows; i++) {
                    for (SIZE_TYPE j = weights->connections.ptrs[i]; j < weights->connections.ptrs[i + 1]; j++) {
                        weights->connections.indices[j] -= add_map_ptrs[weights->connections.indices[j]];
                    }
                }
                merge_duplicate_indices(weights->connections, num_cpus);
            }
        }
    }
}

#endif