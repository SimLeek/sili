#ifndef _fiber_hpp
#define _fiber_hpp

#include "csr.hpp"
#include "parallel.hpp"
#include "sparse_struct.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>


template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_forward(
    VALUE_TYPE* expanded_tensor,
    VALUE_TYPE* output,
    SIZE_TYPE batches,
    const std::vector<SIZE_TYPE>& map_ptrs,
    VALUE_TYPE* importances,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    SIZE_TYPE num_output_indices = map_ptrs.size() - 1;
    SIZE_TYPE num_input_indices = map_ptrs.back();  // this could be the last value in map_ptrs. Usually will be

    if (batches == 0 || map_ptrs.size() < 2) return;
    if (!expanded_tensor || !output) throw std::invalid_argument("expanded_tensor and output must be non-null");
    if (num_cpus <= 0) num_cpus = 1;
    // not checking that map_ptrs is non-decreasing. That could take a long time, so the calling function should handle that

#pragma omp parallel num_threads(num_cpus)
    {

        for (SIZE_TYPE batch = 0; batch < batches; batch++) {
            #pragma omp for
            for (SIZE_TYPE output_index_local = 0; output_index_local < num_output_indices; output_index_local++) {
                SIZE_TYPE input_index = map_ptrs[output_index_local];
                output[num_output_indices*batch+output_index_local] = 0;
                while(input_index<map_ptrs[output_index_local+1]){
                    output[num_output_indices*batch+output_index_local]+=expanded_tensor[num_input_indices*batch+input_index];
                    if (importances != nullptr) {
                        importances[input_index] +=expanded_tensor[num_input_indices*batch+input_index];
                    }
                    input_index+=1;
                }
            }
        }
    }
}

/*
*******************BACKWARD***********************
*/

//needed for fiber contract backward
template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_expand_forward(
    VALUE_TYPE* contracted_tensor,
    VALUE_TYPE* output,
    SIZE_TYPE batches,
    const std::vector<SIZE_TYPE>& map_ptrs,
    VALUE_TYPE* importances,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    SIZE_TYPE num_output_indices = map_ptrs.back();
    SIZE_TYPE num_input_indices = map_ptrs.size() - 1;  // this could be map_ptrs.size() - 1. Usually will be

    if (batches == 0 || map_ptrs.size() < 2) return;
    if (!contracted_tensor || !output) throw std::invalid_argument("expanded_tensor and output must be non-null");
    if (num_cpus <= 0) num_cpus = 1;
    // not checking that map_ptrs is non-decreasing. That could take a long time, so the calling function should handle that

#pragma omp parallel num_threads(num_cpus)
    {

        for (SIZE_TYPE batch = 0; batch < batches; batch++) {
            #pragma omp for
            for (SIZE_TYPE input_index = 0; input_index < num_input_indices; input_index++) {
                SIZE_TYPE output_index = map_ptrs[input_index];
                while(output_index<map_ptrs[input_index+1]){
                    output[num_output_indices*batch+output_index]=contracted_tensor[num_input_indices*batch+input_index];
                    if (importances != nullptr) {
                        importances[input_index] +=contracted_tensor[num_input_indices*batch+input_index];
                    }
                    output_index+=1;
                }
            }
        }
    }
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_backward(
    VALUE_TYPE* contracted_grad_output,
    VALUE_TYPE* grad_input,
    SIZE_TYPE batches,
    const std::vector<SIZE_TYPE>& map_ptrs,
    VALUE_TYPE* importances,
    int num_cpus = 4
) {
    SIZE_TYPE num_output_indices = map_ptrs.size() - 1;
    SIZE_TYPE num_input_indices = map_ptrs.back();

    // Expand the pre-scaled gradients
    fiber_expand_forward(contracted_grad_output, grad_input,batches, map_ptrs, (VALUE_TYPE*)nullptr, num_cpus);

    // Compute importance updates using grad_expanded
    #pragma omp parallel num_threads(num_cpus)
    for (SIZE_TYPE b = 0; b < batches; b++) {
        # pragma omp for
        for (SIZE_TYPE output_index = 0; output_index < num_output_indices; output_index++) {
                SIZE_TYPE input_index = map_ptrs[output_index];
                while(input_index<map_ptrs[output_index+1]){
                    if (importances != nullptr) {
                        importances[input_index] -=contracted_grad_output[num_output_indices*b+output_index];
                    }
                    input_index+=1;
                }
            }
    }
}

/*
 ***************OPTIM********************
*/

/**
 * @brief Merges duplicate columns in the sparse weights matrix.
 *
 * This function compacts the sparse matrix by merging duplicate indices in each row, summing all of their values,
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
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    // todo: move this to csr.hpp
    std::vector<SIZE_TYPE> local_sub(weights.rows, 0);
    std::vector<std::vector<SIZE_TYPE>> compacted_indices_per_row(weights.rows);
    std::vector<std::array<std::vector<VALUE_TYPE>, weights.n_value_arrays>> compacted_values_per_row(weights.rows);

    // Compact each row in parallel (assume indices are sorted)
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        std::vector<SIZE_TYPE> local_indices;
        std::array<std::vector<VALUE_TYPE>, weights.n_value_arrays> local_values;
        SIZE_TYPE sub = 0;
        SIZE_TYPE j = weights.ptrs[0].get()[i];
        while (j < weights.ptrs[0].get()[i + 1]) {
            SIZE_TYPE current_idx = weights.indices[0].get()[j];
            std::array<VALUE_TYPE, weights.n_value_arrays+1> sum;
            // init to weight values
            sum[0] = weights.values[0].get()[j]*weights.values[2].get()[j];
            sum[1] = weights.values[1].get()[j];
            sum[2] = weights.values[0].get()[j];
            sum[3] = weights.values[2].get()[j];
            j++;
            while (j < weights.ptrs[0].get()[i + 1] && weights.indices[0].get()[j] == current_idx) {
                /*for(int v=0; v<weights.n_value_arrays; v++){
                    sum[v] += weights.values[v].get()[j];
                }*/
                sum[0] += weights.values[0].get()[j]*weights.values[2].get()[j];
                sum[1] += weights.values[1].get()[j];
                sum[2] += weights.values[0].get()[j];
                sum[3] += weights.values[2].get()[j];
                j++;
                sub++;
            }
            local_indices.push_back(current_idx);
            //unlike outputs, inputs must NOT be averaged when merged, only summed
            /*for(int v=0; v<weights.n_value_arrays; v++){
                local_values[v].push_back(sum[v]);
            }*/
            local_values[0].push_back(sum[0]/sum[3]);  // values are averaged by importance to maintain training
            local_values[1].push_back(sum[1]);
            local_values[2].push_back(sum[0]/sum[2]);  // importances are averaged by value to maintain training
        }
        compacted_indices_per_row[i] = std::move(local_indices);
        for(int v=0; v<weights.n_value_arrays; v++){
            compacted_values_per_row[i][v] = std::move(local_values[v]);
        }
        local_sub[i] = sub;
    }

    // Compute compacted row lengths
    std::vector<SIZE_TYPE> compacted_row_len(weights.rows);
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        compacted_row_len[i] = compacted_indices_per_row[i].size();
    }

    // Compute new_ptrs using exclusive scan
    SIZE_TYPE *new_ptrs = new SIZE_TYPE[weights.rows + 1];
    //new_ptrs[0] = 0;
    omp_scan_exclusive(compacted_row_len.data(), new_ptrs, weights.rows+1);

    //std::vector<SIZE_TYPE> test_ptrs(new_ptrs, new_ptrs+weights.rows+1);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    SIZE_TYPE new_nnz = new_ptrs[weights.rows];

    // Compute new indices and values given the new pointers
    SIZE_TYPE* new_indices = new SIZE_TYPE[new_nnz];
    std::array<VALUE_TYPE*, weights.n_value_arrays> new_values;
    for(int v=0; v<weights.n_value_arrays; v++){
        new_values[v] = new VALUE_TYPE[new_nnz];
    }
    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE i = 0; i < weights.rows; i++) {
        SIZE_TYPE write_start = new_ptrs[i];
        SIZE_TYPE len = compacted_row_len[i];
        for (SIZE_TYPE k = 0; k < len; k++) {
            new_indices[write_start + k] = compacted_indices_per_row[i][k];
            for(int v=0; v<weights.n_value_arrays; v++){
                new_values[v][write_start + k] = compacted_values_per_row[i][v][k];
            }
        }
    }

    //std::vector<SIZE_TYPE> test_ind(new_indices, new_indices+new_nnz);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test_val1(new_values[0], new_values[0]+new_nnz);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test_val2(new_values[1], new_values[1]+new_nnz);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test_val3(new_values[2], new_values[2]+new_nnz);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays


    weights.ptrs[0].reset(new_ptrs);
    weights.indices[0].reset(new_indices);
    for(int v=0; v<weights.n_value_arrays; v++){
        weights.values[v].reset(new_values[v]);
    }
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void merge_duplicate_indices(
    CSRInput<SIZE_TYPE, VALUE_TYPE>& csr,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    std::vector<SIZE_TYPE> local_sub(csr.rows, 0);
    std::vector<std::vector<SIZE_TYPE>> compacted_indices_per_row(csr.rows);
    std::vector<std::vector<VALUE_TYPE>> compacted_values_per_row(csr.rows);

    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE i = 0; i < csr.rows; i++) {
        std::vector<SIZE_TYPE> local_indices;
        std::vector<VALUE_TYPE> local_values;
        SIZE_TYPE sub = 0;
        SIZE_TYPE j = csr.ptrs[0].get()[i];
        while (j < csr.ptrs[0].get()[i + 1]) {
            SIZE_TYPE current_idx = csr.indices[0].get()[j];
            VALUE_TYPE sum = csr.values[0].get()[j];
            j++;
            while (j < csr.ptrs[0].get()[i + 1] && csr.indices[0].get()[j] == current_idx) {
                sum += csr.values[0].get()[j];
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

    std::vector<SIZE_TYPE> compacted_row_len(csr.rows);
    for (SIZE_TYPE i = 0; i < csr.rows; i++) {
        compacted_row_len[i] = compacted_indices_per_row[i].size();
    }

    SIZE_TYPE* new_ptrs = new SIZE_TYPE[csr.rows + 1];
    omp_scan_exclusive(compacted_row_len.data(), new_ptrs, csr.rows + 1);

    SIZE_TYPE new_nnz = new_ptrs[csr.rows];
    SIZE_TYPE* new_indices = new SIZE_TYPE[new_nnz];
    VALUE_TYPE* new_values = new VALUE_TYPE[new_nnz];

    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE i = 0; i < csr.rows; i++) {
        SIZE_TYPE write_start = new_ptrs[i];
        SIZE_TYPE len = compacted_row_len[i];
        for (SIZE_TYPE k = 0; k < len; k++) {
            new_indices[write_start + k] = compacted_indices_per_row[i][k];
            new_values[write_start + k] = compacted_values_per_row[i][k];
        }
    }

    csr.ptrs[0].reset(new_ptrs);
    csr.indices[0].reset(new_indices);
    csr.values[0].reset(new_values);
}

/**
 * @brief Merges duplicate entries in a specified row of the sparse weights.
 *
 * This function sorts the indices and values for the specified row range (defined by `updated_ptrs[row_to_merge-1]` to `updated_ptrs[row_to_merge]`)
 * and merges duplicate indices by summing their values. It operates on a single row and is used in optimization routines.
 * Note that this DOE NOT properly merge rows in a CSR. The rest of that is adjust rows.
 *
 * @tparam SIZE_TYPE Integer type for sizes and indices (e.g., size_t).
 * @tparam VALUE_TYPE Type for weight values (e.g., double, float).
 * @param weights Sparse linear weights structure, modified in place.
 * @param ptrs Updated pointers defining row ranges in the weights.
 * @param row_to_merge The row index to merge duplicates in (must have a row after it).
 */
template <typename SIZE_TYPE, typename VALUE_TYPE> 
SIZE_TYPE _merge_duplicate_row_with_next(
    CSRSynapses<SIZE_TYPE, VALUE_TYPE>& weights,
    SIZE_TYPE row_to_merge,
    int num_cpus = 4
){
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    //todo: check if row_to_merge+2 > weights.rows. If so, raise error.

    //sort weights and indices so we can merge sorted which is much easier
    omp_sort_arrays_ascending(
        weights.ptrs[0].get()[row_to_merge+2]-weights.ptrs[0].get()[row_to_merge], //needs to be row_to_merge+2 to get the ptr to the end of the next row
        weights.indices[0].get()+weights.ptrs[0].get()[row_to_merge], 
        weights.values[0].get()+weights.ptrs[0].get()[row_to_merge],
        weights.values[1].get()+weights.ptrs[0].get()[row_to_merge],
        weights.values[2].get()+weights.ptrs[0].get()[row_to_merge]);

    //std::vector<SIZE_TYPE> test_ptr(weights.ptrs[0].get(), weights.ptrs[0].get()+weights.rows+1);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<SIZE_TYPE> test_ind(weights.indices[0].get(), weights.indices[0].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test_val1(weights.values[0].get(), weights.values[0].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test_val2(weights.values[1].get(), weights.values[1].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test_val3(weights.values[2].get(), weights.values[2].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays

    //merge nearby equal indices
    SIZE_TYPE sub = 0;
    for(SIZE_TYPE j=weights.ptrs[0].get()[row_to_merge]; j<weights.ptrs[0].get()[row_to_merge+2]; j++){
        //shift left
        weights.indices[0].get()[j-sub]=weights.indices[0].get()[j];
        for(int v=0; v<weights.n_value_arrays; v++){
            weights.values[v].get()[j-sub]=weights.values[v].get()[j];
        }

        if(weights.indices[0].get()[j]==weights.indices[0].get()[j+1]){
            //weight multiplier values MUST be averaged due to average based fiber contraction, while backprop accum and optim importance MUST be summed
            //importance valued average. Also fixes an issue with merging 3+ instead of 2
            // values are averaged by importance to maintain training
            auto tmp_sum_num = (weights.values[0].get()[j+1]*weights.values[2].get()[j+1]+weights.values[0].get()[j]*weights.values[2].get()[j]);
            auto tmp_imp_sum = (weights.values[2].get()[j+1]+weights.values[2].get()[j]);
            auto tmp_val_sum = (weights.values[0].get()[j+1]+weights.values[0].get()[j]);
            weights.values[0].get()[j+1]=tmp_sum_num/tmp_imp_sum;
            //weights.values[0].get()[j+1]=weights.values[0].get()[j+1];
            weights.values[1].get()[j+1]+=weights.values[1].get()[j];
            // importances are averaged by value to maintain training
            weights.values[2].get()[j+1]=tmp_sum_num/tmp_val_sum;

            sub+=1;
        }
    }
    #pragma omp parallel for num_threads(num_cpus)
    for(SIZE_TYPE j=weights.ptrs[0].get()[row_to_merge+2]; j<weights.nnz(); j++){
        //shift left
        weights.indices[0].get()[j-sub]=weights.indices[0].get()[j];
        for(int v=0; v<weights.n_value_arrays; v++){
            weights.values[v].get()[j-sub]=weights.values[v].get()[j];
        }
    }

    //std::vector<SIZE_TYPE> test2_ptr(weights.ptrs[0].get(), weights.ptrs[0].get()+weights.rows+1);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<SIZE_TYPE> test2_ind(weights.indices[0].get(), weights.indices[0].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test2_val1(weights.values[0].get(), weights.values[0].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test2_val2(weights.values[1].get(), weights.values[1].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    //std::vector<VALUE_TYPE> test2_val3(weights.values[2].get(), weights.values[2].get()+weights.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays

    return sub;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
SIZE_TYPE _merge_duplicate_row_with_next(
    CSRInput<SIZE_TYPE, VALUE_TYPE>& csr,
    SIZE_TYPE row_to_merge,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    if (row_to_merge + 2 > csr.rows) {
        throw std::out_of_range("row_to_merge + 2 exceeds number of rows");
    }

    omp_sort_arrays_ascending(
        csr.ptrs[0].get()[row_to_merge + 2] - csr.ptrs[0].get()[row_to_merge],
        csr.indices[0].get() + csr.ptrs[0].get()[row_to_merge],
        csr.values[0].get() + csr.ptrs[0].get()[row_to_merge]
    );

    SIZE_TYPE sub = 0;
    for (SIZE_TYPE j = csr.ptrs[0].get()[row_to_merge]; j < csr.ptrs[0].get()[row_to_merge + 2]; j++) {
        csr.indices[0].get()[j - sub] = csr.indices[0].get()[j];
        csr.values[0].get()[j - sub] = csr.values[0].get()[j];

        if (j + 1 < csr.ptrs[0].get()[row_to_merge + 2] && csr.indices[0].get()[j] == csr.indices[0].get()[j + 1]) {
            csr.values[0].get()[j + 1] += csr.values[0].get()[j];
            sub++;
        }
    }

    #pragma omp parallel for num_threads(num_cpus)
    for (SIZE_TYPE j = csr.ptrs[0].get()[row_to_merge + 2]; j < csr.nnz(); j++) {
        csr.indices[0].get()[j - sub] = csr.indices[0].get()[j];
        csr.values[0].get()[j - sub] = csr.values[0].get()[j];
    }

    return sub;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> fiber_contract_optim(
    std::vector<SIZE_TYPE>& map_ptrs,
    const std::vector<const VALUE_TYPE*>& importances_list,
    SIZE_TYPE neurons_to_change,
    bool add,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    SIZE_TYPE num_output_indices = map_ptrs.size() - 1;
    std::vector<VALUE_TYPE> avg_importance(num_output_indices);
    std::vector<SIZE_TYPE> add_map_ptrs(num_output_indices + 1, 0);

    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < num_output_indices; i++) {
            VALUE_TYPE sum = 0;
            SIZE_TYPE total_k_i = 0;
            for (const auto* importances : importances_list) {
                for (SIZE_TYPE h = map_ptrs[i]; h < map_ptrs[i + 1]; h++) {
                    sum += importances[h];
                }
                total_k_i += map_ptrs[i + 1] - map_ptrs[i];
            }
            if(total_k_i>0){
                avg_importance[i] = sum / total_k_i;
            }else{
                if(add){
                    avg_importance[i] = 0;
                }else{
                    avg_importance[i] = std::numeric_limits<VALUE_TYPE>::max();
                }
            }
        }
    }
    std::vector<size_t> change_spots;
    if(add){
        change_spots = top_k_indices(avg_importance.data(), num_output_indices, neurons_to_change, num_cpus);
    }
    else{
        change_spots = bottom_k_indices(avg_importance.data(), num_output_indices, neurons_to_change, num_cpus);
    }

    #pragma omp parallel num_threads(num_cpus)
    {
        #pragma omp for
        for (SIZE_TYPE i = 0; i < neurons_to_change; i++) {
            add_map_ptrs[change_spots[i]+1] = 1;
        }
    }

    omp_scan_inclusive(add_map_ptrs.data(), add_map_ptrs.data(), add_map_ptrs.size());

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
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*>&& weights_array,
    bool add,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    SIZE_TYPE neurons_to_change = add_map_ptrs.back();

    for (auto* weights : weights_array) {
        if (weights == nullptr) continue;

        weights->connections.cols = add ? weights->connections.cols + neurons_to_change : weights->connections.cols - neurons_to_change;

        #pragma omp parallel num_threads(num_cpus)
        {
            if (add) {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < weights->connections.rows; i++) {
                    for (SIZE_TYPE j = weights->connections.ptrs[0].get()[i]; j < weights->connections.ptrs[0].get()[i + 1]; j++) {
                        weights->connections.indices[0].get()[j] += add_map_ptrs[weights->connections.indices[0].get()[j]];
                    }
                }
            } else {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < weights->connections.rows; i++) {
                    for (SIZE_TYPE j = weights->connections.ptrs[0].get()[i]; j < weights->connections.ptrs[0].get()[i + 1]; j++) {
                        weights->connections.indices[0].get()[j] -= add_map_ptrs[weights->connections.indices[0].get()[j]];
                    }
                }
                merge_duplicate_indices(weights->connections, num_cpus);
            }
        }

        //std::vector<SIZE_TYPE> test2_ptr(weights->connections.ptrs[0].get(), weights->connections.ptrs[0].get()+weights->connections.rows+1);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<SIZE_TYPE> test2_ind(weights->connections.indices[0].get(), weights->connections.indices[0].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<VALUE_TYPE> test2_val1(weights->connections.values[0].get(), weights->connections.values[0].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<VALUE_TYPE> test2_val2(weights->connections.values[1].get(), weights->connections.values[1].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<VALUE_TYPE> test2_val3(weights->connections.values[2].get(), weights->connections.values[2].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    }
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_optim_adjust_columns(
    const std::vector<SIZE_TYPE>& add_map_ptrs,
    std::vector<CSRInput<SIZE_TYPE, VALUE_TYPE>*>&& csr_array,
    bool add,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    SIZE_TYPE neurons_to_change = add_map_ptrs.back();

    for (auto* csr : csr_array) {
        if (csr == nullptr) continue;

        csr->cols = add ? csr->cols + neurons_to_change : csr->cols - neurons_to_change;

        #pragma omp parallel num_threads(num_cpus)
        {
            if (add) {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < csr->rows; i++) {
                    for (SIZE_TYPE j = csr->ptrs[0].get()[i]; j < csr->ptrs[0].get()[i + 1]; j++) {
                        csr->indices[0].get()[j] += add_map_ptrs[csr->indices[0].get()[j]];
                    }
                }
            } else {
                #pragma omp for
                for (SIZE_TYPE i = 0; i < csr->rows; i++) {
                    for (SIZE_TYPE j = csr->ptrs[0].get()[i]; j < csr->ptrs[0].get()[i + 1]; j++) {
                        csr->indices[0].get()[j] -= add_map_ptrs[csr->indices[0].get()[j]];
                    }
                }
                merge_duplicate_indices(*csr, num_cpus);
            }
        }

        //std::vector<SIZE_TYPE> test_ptr(csr->ptrs[0].get(), csr->ptrs[0].get() + csr->rows + 1);
        //std::vector<SIZE_TYPE> test_ind(csr->indices[0].get(), csr->indices[0].get() + csr->nnz());
        //std::vector<VALUE_TYPE> test_val(csr->values[0].get(), csr->values[0].get() + csr->nnz());
    }
}

// For recurrent states, you need to adjust both input and output (so input+state->state, use this for state)
template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_optim_adjust_rows(
    std::vector<SIZE_TYPE>& add_map_ptrs,
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*>&& weights_array,
    bool add,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    for (auto* weights : weights_array) {
        if (weights == nullptr) continue;

        SIZE_TYPE neurons_to_change = add_map_ptrs.back();
        if(add){
            add_map_ptrs.insert(add_map_ptrs.end(), neurons_to_change, neurons_to_change);
        }
        SIZE_TYPE new_rows = add ? weights->connections.rows + neurons_to_change : weights->connections.rows - neurons_to_change;
        SIZE_TYPE* new_csc_ptrs(new SIZE_TYPE[new_rows + 1]);

        if (add) {
            #pragma omp parallel for num_threads(num_cpus)
            for (SIZE_TYPE i = 0; i < new_rows + 1; i++) {
                //!!!: this goes out of bounds easily, needs fix
                new_csc_ptrs[i] = weights->connections.ptrs[0].get()[i - add_map_ptrs[i]];
            }
        } else {
            SIZE_TYPE sub = 0;
            for (SIZE_TYPE i = 0; i < new_rows + 1; i++) {
                if (i > 0 && add_map_ptrs[i] != add_map_ptrs[i - 1]) {
                    sub+=_merge_duplicate_row_with_next(weights->connections, i - 1, num_cpus);
                }
                new_csc_ptrs[i]=weights->connections.ptrs[0].get()[i + add_map_ptrs[i]]-sub;
            }
        }
        weights->connections.rows = new_rows;
        weights->connections.ptrs[0].reset(new_csc_ptrs);
        //indices and values are already handled.
        /*weights->connections.indices[0].reset(new_indices);
        for(int v=0; v<weights->connections.n_value_arrays; v++){
            weights->connections.values[v].reset(new_values[v]);
        }*/

        //std::vector<SIZE_TYPE> test2_ptr(weights->connections.ptrs[0].get(), weights->connections.ptrs[0].get()+new_rows+1);//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<SIZE_TYPE> test2_ind(weights->connections.indices[0].get(), weights->connections.indices[0].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<VALUE_TYPE> test2_val1(weights->connections.values[0].get(), weights->connections.values[0].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<VALUE_TYPE> test2_val2(weights->connections.values[1].get(), weights->connections.values[1].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
        //std::vector<VALUE_TYPE> test2_val3(weights->connections.values[2].get(), weights->connections.values[2].get()+weights->connections.nnz());//debug so I can tell wtf is going on because oss-code is still too retarded to view arrays
    
    }
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void fiber_contract_optim_adjust_rows(
    std::vector<SIZE_TYPE>& add_map_ptrs,
    std::vector<CSRInput<SIZE_TYPE, VALUE_TYPE>*>&& csr_array,
    bool add,
    int num_cpus = 4
) {
    static_assert(std::is_unsigned<SIZE_TYPE>::value, "SIZE_TYPE must be unsigned");

    for (auto* csr : csr_array) {
        if (csr == nullptr) continue;

        SIZE_TYPE neurons_to_change = add_map_ptrs.back();
        if (add) {
            add_map_ptrs.insert(add_map_ptrs.end(), neurons_to_change, neurons_to_change);
        }
        SIZE_TYPE new_rows = add ? csr->rows + neurons_to_change : csr->rows - neurons_to_change;
        SIZE_TYPE* new_csc_ptrs = new SIZE_TYPE[new_rows + 1];

        if (add) {
            #pragma omp parallel for num_threads(num_cpus)
            for (SIZE_TYPE i = 0; i < new_rows + 1; i++) {
                new_csc_ptrs[i] = (i <= csr->rows) ? csr->ptrs[0].get()[i] : csr->ptrs[0].get()[csr->rows];
            }
        } else {
            SIZE_TYPE sub = 0;
            for (SIZE_TYPE i = 0; i < new_rows + 1; i++) {
                if (i > 0 && add_map_ptrs[i] != add_map_ptrs[i - 1]) {
                    sub += _merge_duplicate_row_with_next(*csr, i - 1, num_cpus);
                }
                new_csc_ptrs[i] = csr->ptrs[0].get()[i + add_map_ptrs[i]] - sub;
            }
        }
        csr->rows = new_rows;
        csr->ptrs[0].reset(new_csc_ptrs);

        //std::vector<SIZE_TYPE> test_ptr(csr->ptrs[0].get(), csr->ptrs[0].get() + new_rows + 1);
        //std::vector<SIZE_TYPE> test_ind(csr->indices[0].get(), csr->indices[0].get() + csr->nnz());
        //std::vector<VALUE_TYPE> test_val(csr->values[0].get(), csr->values[0].get() + csr->nnz());
    }
}

#endif