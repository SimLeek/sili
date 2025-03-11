#ifndef _fiber_hpp
#define _fiber_hpp

#include "csr.hpp"
#include "parallel.hpp"
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

#endif