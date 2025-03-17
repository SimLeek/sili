#ifndef _fiber_hpp
#define _fiber_hpp

#include "csr.hpp"
#include "parallel.hpp"
#include "sparse_struct.hpp"
#include <algorithm>
#include <functional>
#include <iterator>
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
 * @brief Expands a sparse input tensor in CSR format based on a mapping.
 *
 * This function takes a sparse tensor in CSR format and expands it by replicating each value across a range of output indices defined by `map_ptrs`. The operation is parallelized using OpenMP.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., `size_t`).
 * @tparam VALUE_TYPE Type for tensor values (e.g., `double`, `float`).
 * @param input_tensor Input sparse tensor in CSR format.
 * @param map_ptrs Vector defining output index ranges for each input index.
 * @param num_cpus Number of CPU threads (default: 4).
 * @return Expanded sparse tensor in CSR format.
 */
template <typename SIZE_TYPE, typename VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>> fiber_expand_forward(
    const CSRInput<SIZE_TYPE, VALUE_TYPE> &input_tensor,
    const std::vector<SIZE_TYPE> &map_ptrs,
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
 * Sum contributions from mapped neurons back to original neurons using the mapping.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for tensor values (e.g., double, float).
 * @param mapped_contributions Contributions of mapped neurons (size input_tensor.cols).
 * @param map_ptrs Mapping array where map_ptrs[i] to map_ptrs[i+1] defines mapped indices for original neuron i.
 * @param num_input_indices Number of original neurons.
 * @param original_contributions_output_pre_map Output array for original neuron contributions (size num_input_indices).
 * @param num_cpus Number of CPU threads to use.
 */
 template <typename SIZE_TYPE, typename VALUE_TYPE>
 void sum_contributions_to_original(
     const VALUE_TYPE* mapped_contributions,
     const std::vector<SIZE_TYPE>& map_ptrs,
     SIZE_TYPE num_input_indices,
     VALUE_TYPE* importance,
     const int num_cpus = 4
 ) {
     #pragma omp parallel for num_threads(num_cpus) reduction(+:original_contributions_output_pre_map[:num_input_indices])
     for (SIZE_TYPE original_index = 0; original_index < num_input_indices; original_index++) {
         VALUE_TYPE sum = 0;
         for (SIZE_TYPE mapped_ptr = map_ptrs[original_index]; 
              mapped_ptr < map_ptrs[original_index + 1]; 
              mapped_ptr++) 
         {
             SIZE_TYPE mapped_index = mapped_ptr; // Direct indexing into mapped_contributions
             sum += mapped_contributions[mapped_index];
         }
         importance[original_index] += sum;
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
 * @param num_cpus Number of CPU threads (default: 4).
 * @param aggregate Aggregation function (default: average).
 * @return Contracted sparse tensor in CSR format.
 */
template <typename SIZE_TYPE, typename VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>> fiber_contract_forward(
    const CSRInput<SIZE_TYPE, VALUE_TYPE> &expanded_tensor,
    const std::vector<SIZE_TYPE> &map_ptrs,
    int num_cpus = 4,
    std::function<VALUE_TYPE(const std::vector<VALUE_TYPE> &)> aggregate = [](const std::vector<VALUE_TYPE> &vals) {
        return std::accumulate(vals.begin(), vals.end(), VALUE_TYPE(0)) / vals.size();
    }) {
    SIZE_TYPE num_rows = expanded_tensor.rows;         // Number of batches (e.g., 1-2)
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1; // Number of original input indices

    // Step 1: Compute non-zero counts per batch for the output
    std::vector<std::vector<SIZE_TYPE>> output_counts(num_rows, std::vector<SIZE_TYPE>(num_input_indices, 0));
    std::shared_ptr<SIZE_TYPE> output_ptrs(new SIZE_TYPE[num_rows + 1]);
    output_ptrs[0] = 0;

    std::shared_ptr<SIZE_TYPE> output_indices;
    std::shared_ptr<VALUE_TYPE> output_values;

    std::shared_ptr<std::shared_ptr<SIZE_TYPE>> input_indices_psudo_ptrs(num_rows);
    std::vector<std::unique_ptr<SIZE_TYPE[]>> output_count_scan(num_rows);

// Single parallel region for everything
#pragma omp parallel num_threads(num_cpus) shared(output_counts, output_ptrs, output_indices, output_values)
    {
        SIZE_TYPE tid = omp_get_thread_num();
        SIZE_TYPE num_threads = omp_get_num_threads();

        // Thread-local count accumulation
        std::vector<std::vector<SIZE_TYPE>> thread_counts(
            num_rows, std::vector<SIZE_TYPE>(num_input_indices)); // reserve max size
        std::vector<std::vector<SIZE_TYPE>> thread_input_indices(
            num_rows, std::vector<SIZE_TYPE>(num_input_indices)); // reserve, don't set

        // Count unique input indices per batch
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
            SIZE_TYPE row_start = expanded_tensor.ptrs[0][batch];
            SIZE_TYPE row_end = expanded_tensor.ptrs[0][batch + 1];
            SIZE_TYPE row_len = row_end - row_start;

            // Parallelize over the row’s non-zero elements
            auto it = std::upper_bound(
                map_ptrs.get(), map_ptrs.get() + num_input_indices + 1, expanded_tensor.indices[0][row_start]);
            auto it_end = std::upper_bound(
                map_ptrs.get(), map_ptrs.get() + num_input_indices + 1, expanded_tensor.indices[0][row_end]);
            int chunk_size = (std::distance(it, it_end) - 1) / num_threads;

            for (it + chunk_size * tid; it < it_end; it++) {
                // Map back to original input index
                SIZE_TYPE input_index = std::distance(map_ptrs.get(), it) - 1;
                auto is_in = std::binary_search(expanded_tensor.indices[0] + row_start,
                                                expanded_tensor.indices[0] + row_end,
                                                map_ptrs[input_index]);
                if (is_in) {
                    thread_input_indices[batch].push_back(input_index);
                    thread_counts[batch].push_back(map_ptrs[input_index + 1] - map_ptrs[input_index]);
                }
            }

            for (SIZE_TYPE i = 0; i < chunk_size && i < thread_counts[batch].size(); i++) {
                input_indices_psudo_ptrs[batch][tid * chunk_size + i] = thread_input_indices[batch][i]; // concat
                output_counts[batch][tid * chunk_size + i] = thread_counts[batch][i];                   // concat
            }
            if (tid == num_threads - 1) {
                input_indices_psudo_ptrs[batch].resize(tid * chunk_size + thread_input_indices[batch].size()); // concat
                output_counts[batch].resize(tid * chunk_size + thread_counts[batch].size());
            }
        }

        // Reduce thread-local counts into global output_count
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
            output_count_scan[batch].reset(new SIZE_TYPE[output_counts[batch].size() + 1]);
            omp_full_scan(output_counts[batch], output_count_scan[batch], output_counts[batch].size());
        }

        // Step 2: Compute output pointers and allocate memory (single-threaded)
        if (tid == 0) {
            SIZE_TYPE total_nnz = 0;
            for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
                total_nnz += output_count_scan[batch][output_counts[batch].size()];
                output_ptrs[batch + 1] = total_nnz;
            }
            output_indices.reset(new SIZE_TYPE[total_nnz]);
            output_values.reset(new VALUE_TYPE[total_nnz]);
        }
#pragma omp barrier

        // Parallelize over the output tensor’s non-zero elements
        for (SIZE_TYPE batch = 0; batch < num_rows; batch++) {
#pragma omp for
            for (SIZE_TYPE ptr_i = 0; ptr_i < input_indices_psudo_ptrs[batch].size(); ptr_i++) {
                // Find the batch this output position belongs to
                auto map_output_to = input_indices_psudo_ptrs[batch][ptr_i];
                auto map_from_start = output_count_scan[batch][ptr_i];
                auto map_from_end = output_count_scan[batch][ptr_i + 1];

                std::vector<VALUE_TYPE> values;
                for (auto m = map_from_start; m < map_from_end; m++) {
                    values.push_back(expanded_tensor[batch][m]);
                }

                // Write aggregated value directly to output
                output_indices[ptr_i] = map_output_to;
                output_values[ptr_i] = aggregate(values);
            }
        }
    }

    // Step 4: Return the CSR structure
    return create_csr(num_rows, num_input_indices, output_ptrs, output_indices, output_values);
}

/**
 * @brief Distributes importances from contracted outputs to expanded neurons.
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
     VALUE_TYPE* expanded_contributions,
     const int num_cpus = 4
 ) {
     #pragma omp parallel for num_threads(num_cpus)
     for (SIZE_TYPE i = 0; i < num_input_indices; i++) {
         SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
         VALUE_TYPE contrib = contracted_importances[i] / k_i;
         for (SIZE_TYPE j = map_ptrs[i]; j < map_ptrs[i + 1]; j++) {
             expanded_contributions[j] = contrib;
         }
     }
 }

 template <typename SIZE_TYPE, typename VALUE_TYPE>
 void fiber_expand_backward(
     const std::vector<SIZE_TYPE>& map_ptrs,
     const sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>>& grad_expanded,
     sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, std::shared_ptr<SIZE_TYPE>, std::shared_ptr<VALUE_TYPE>>& grad_input,
     VALUE_TYPE* importance_updates, // Dense, size num_input_indices
     int num_cpus = 4
 ) {
     // Sum aggregation for contraction
     auto sum_aggregate = [](const std::vector<VALUE_TYPE>& vals) {
         return std::accumulate(vals.begin(), vals.end(), VALUE_TYPE(0));
     };
 
     // Use fiber_contract_forward to sum gradients back to original indices
     grad_input = fiber_contract_forward(grad_expanded, map_ptrs, num_cpus, sum_aggregate);
 
    // Compute importance updates using grad_input
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE b = 0; b < grad_input.rows; b++) {
        for (SIZE_TYPE ptr = grad_input.ptrs[0][b]; ptr < grad_input.ptrs[0][b + 1]; ptr++) {
            SIZE_TYPE i = grad_input.indices[0][ptr]; // Index in original input space
            VALUE_TYPE grad_value = grad_input.values[0][ptr];
            importance_updates[i] -= grad_value;
        }
    }
 }

 template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_backward(
    const std::vector<SIZE_TYPE>& map_ptrs,
    const sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>& grad_contracted,
    sparse_struct<SIZE_TYPE, CSRPtrs<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>& grad_expanded,
    VALUE_TYPE* expanded_importance_updates, // Dense array, size num_expanded_neurons
    int num_cpus = 4
) {
    SIZE_TYPE num_rows = grad_contracted.rows;
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1;

    // Step 1: Pre-scale grad_contracted by 1/k_i
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

    // Step 2: Expand the pre-scaled gradients
    grad_expanded = fiber_expand_forward(scaled_grad_contracted, map_ptrs, num_cpus);

    // Step 3: Compute importance updates using grad_expanded
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE b = 0; b < grad_expanded.rows; b++) {
        for (SIZE_TYPE ptr = grad_expanded.ptrs[0][b]; ptr < grad_expanded.ptrs[0][b + 1]; ptr++) {
            SIZE_TYPE j = grad_expanded.indices[0][ptr]; // Index in expanded space
            VALUE_TYPE grad_value = grad_expanded.values[0][ptr];
            expanded_importance_updates[j] -= grad_value;
        }
    }
}


template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_expand_optim(
    std::vector<SIZE_TYPE>& map_ptrs,
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE* importance_h,
    SIZE_TYPE neurons_to_change,
    bool add=true,
    int num_cpus = 4
) {
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1;
    std::vector<VALUE_TYPE> avg_importance(num_input_indices);
    std::vector<VALUE_TYPE> cumulative_priority(num_input_indices + 1);
    std::vector<SIZE_TYPE> new_k(num_input_indices, 0);
    std::vector<SIZE_TYPE> add_map_ptrs(num_input_indices + 1);
    //weights.rows because naming is for csr
    std::unique_ptr<SIZE_TYPE[]> new_csc_ptrs(add? new SIZE_TYPE[weights.rows + (neurons_to_change) + 1]:new SIZE_TYPE[weights.rows - (neurons_to_change) + 1]);

    #pragma omp parallel num_threads(num_cpus)
    {
        // Step 1 & 2: Compute sum and average importance
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_input_indices; i++) {
            VALUE_TYPE sum = 0;
            for (SIZE_TYPE h = map_ptrs[i]; h < map_ptrs[i + 1]; h++) {
                sum += importance_h[h];
            }
            SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
            if(add){
                avg_importance[i] = (k_i > 0) ? sum / k_i : 0;
            }else{
                avg_importance[i] = (k_i > 0) ? -sum / k_i : 0;
            }
        }
    }

    auto change_spots = top_k_indices(avg_importance, map_ptrs.rows, neurons_to_change, num_cpus);

    #pragma omp parallel num_threads(num_cpus)
    {
        // Step 1 & 2: Compute sum and average importance
        #pragma omp for
        for (SIZE_TYPE i = 0; i < neurons_to_change; i++) {
            add_map_ptrs[change_spots[i]] = 1;
        }
    }

    omp_scan_inclusive(add_map_ptrs.data(), add_map_ptrs.data(), add_map_ptrs.size());

    #pragma omp parallel num_threads(num_cpus)
    {
        // Step 1 & 2: Compute sum and average importance
        if(add){
            #pragma omp for
            for (SIZE_TYPE i = 0; i < num_input_indices+1; i++) {
                map_ptrs[i] += add_map_ptrs[i];
            }
            #pragma omp for
            for (SIZE_TYPE i = 0; i < weights.rows + (neurons_to_change) + 1; i++) {
                new_csc_ptrs[i] = weights.ptrs[i-add_map_ptrs[i]];
            }
        } else{
            #pragma omp for
            for (SIZE_TYPE i = 0; i < num_input_indices+1; i++) {
                map_ptrs[i] -= add_map_ptrs[i];
            }
            #pragma omp for
            for (SIZE_TYPE i = 0; i < weights.rows + (neurons_to_change) + 1; i++) {
                new_csc_ptrs[i] = weights.ptrs[i+add_map_ptrs[i]];
                if(add_map_ptrs[i]!=add_map_ptrs[i-1]){
                    //sort weights and indices, not really in parallel, this is just the only function I have that does it
                    omp_sort_ascending(weights.indices.data()+new_csc_ptrs[i], weights.values.data()+new_csc_ptrs[i+1]);
                    //merge nearby equal indices
                    SIZE_TYPE sub=0;
                    for(SIZE_TYPE j=new_csc_ptrs[i-1]; j<new_csc_ptrs[i]; j++){
                        if(weights.indices[j]==weights.indices[j+1]){
                            weights.values[j-sub]+=weights.values[j+1];
                            sub+=1;
                        }
                        weights.indices[j-sub]=weights.indices[j];
                    }
                }
            }
        }
    }    

    // Rebuild weights
    SIZE_TYPE new_H = map_ptrs.back();
    auto new_csc = create_csr(new_H, weights.cols, new_csc_ptrs, weights.connections.indices, weights.connections.values);

    return new_csc;
}

//todo: move this to csr.hpp
template <typename SIZE_TYPE, typename VALUE_TYPE> 
void merge_duplicate_columns(
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    int num_cpus = 4
){
    // Parallel merging of nearby equal indices
    std::vector<SIZE_TYPE> local_sub(weights.rows, 0);
    std::vector<std::vector<SIZE_TYPE>> compacted_indices_per_row(weights.rows);
    std::vector<std::vector<VALUE_TYPE>> compacted_values_per_row(weights.rows);

    // Step 1: Compact each row in parallel
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

    // Step 2: Compute compacted row lengths
    std::vector<SIZE_TYPE> compacted_row_len(weights.rows);
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        compacted_row_len[i] = compacted_indices_per_row[i].size();
    }

    // Step 3: Compute new_ptrs using exclusive scan
    std::vector<SIZE_TYPE> new_ptrs(weights.rows + 1);
    new_ptrs[0] = 0;
    omp_scan_exclusive(compacted_row_len.data(), new_ptrs.data() + 1, weights.rows);

    // Total new number of non-zeros
    SIZE_TYPE new_nnz = new_ptrs[weights.rows];

    // Allocate new arrays
    std::shared_ptr<SIZE_TYPE> new_indices(new SIZE_TYPE[new_nnz]);
    std::shared_ptr<VALUE_TYPE> new_values(new VALUE_TYPE[new_nnz]);

    // Step 4: Write compacted data in parallel
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        SIZE_TYPE write_start = new_ptrs[i];
        SIZE_TYPE len = compacted_row_len[i];
        for (SIZE_TYPE k = 0; k < len; k++) {
            new_indices[write_start + k] = compacted_indices_per_row[i][k];
            new_values[write_start + k] = compacted_values_per_row[i][k];
        }
    }

    // Update weights structure
    weights.ptrs = std::move(new_ptrs);
    weights.indices = std::move(new_indices);
    weights.values = std::move(new_values);
    //weights.nnz = new_nnz;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_optim(
    std::vector<SIZE_TYPE>& map_ptrs,
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE* importance_h,
    SIZE_TYPE neurons_to_change,
    bool add=true,
    int num_cpus = 4
) {
    SIZE_TYPE num_output_indices = map_ptrs.size() - 1;
    std::vector<VALUE_TYPE> avg_importance(num_output_indices);
    std::vector<VALUE_TYPE> cumulative_priority(num_output_indices + 1);
    std::vector<SIZE_TYPE> new_k(num_output_indices, 0);
    std::vector<SIZE_TYPE> add_map_ptrs(num_output_indices + 1);
    //weights.rows because naming is for csr
    //std::unique_ptr<SIZE_TYPE[]> new_csc_ptrs(add? new SIZE_TYPE[weights.rows + (neurons_to_change) + 1]:new SIZE_TYPE[weights.rows - (neurons_to_change) + 1]);

    if(add){
        weights.cols += neurons_to_change;
    }else{
        weights.cols -= neurons_to_change;
    }

    #pragma omp parallel num_threads(num_cpus)
    {
        // Step 1 & 2: Compute sum and average importance
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_output_indices; i++) {
            VALUE_TYPE sum = 0;
            for (SIZE_TYPE h = map_ptrs[i]; h < map_ptrs[i + 1]; h++) {
                sum += importance_h[h];
            }
            SIZE_TYPE k_i = map_ptrs[i + 1] - map_ptrs[i];
            if(add){
                avg_importance[i] = (k_i > 0) ? sum / k_i : 0;
            }else{
                avg_importance[i] = (k_i > 0) ? -sum / k_i : 0;
            }
        }
    }

    auto change_spots = top_k_indices(avg_importance, map_ptrs.rows, neurons_to_change, num_cpus);

    #pragma omp parallel num_threads(num_cpus)
    {
        // Step 1 & 2: Compute sum and average importance
        #pragma omp for
        for (SIZE_TYPE i = 0; i < neurons_to_change; i++) {
            add_map_ptrs[change_spots[i]] = 1;
        }
    }

    omp_scan_inclusive(add_map_ptrs.data(), add_map_ptrs.data(), add_map_ptrs.size());

    #pragma omp parallel num_threads(num_cpus)
    {
        // Step 1 & 2: Compute sum and average importance
        if(add){
            #pragma omp for
            for (SIZE_TYPE i = 0; i < num_output_indices+1; i++) {
                map_ptrs[i] += add_map_ptrs[i];
            }
            #pragma omp for
            for(SIZE_TYPE i=0; i<weights.rows; i++){
                for (SIZE_TYPE j = weights.ptrs[i]; j < weights.ptrs[i+1]; j++) {
                    weights.indices[j] += add_map_ptrs[weights.indices[j]];
                }
            }
        } else{
            #pragma omp for
            for (SIZE_TYPE i = 0; i < num_output_indices+1; i++) {
                map_ptrs[i] -= add_map_ptrs[i];
            }
            #pragma omp for
            for(SIZE_TYPE i=0; i<weights.rows; i++){
                for (SIZE_TYPE j = weights.ptrs[i]; j < weights.ptrs[i+1]; j++) {
                    weights.indices[j] -= add_map_ptrs[weights.indices[j]];
                }
            }
            //merge nearby equal indices
            merge_duplicate_columns(weights, num_cpus);

            //todo: make this parallel using some scans and stuff
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
            }*/
        }
    }    

    return weights;  // use same data since we've already reserved space
}


#endif