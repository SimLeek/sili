#include "csr.hpp"
#include "coo.hpp"

/**
 * Perform a forward pass of a sparse linear layer with sparse input
 *
 * @param input_tensor The input csr. Rows MUST be defined, because batch=row.
 * @param weight_tensor The weight csc. cols becomes rows for CSC context, which makes the matmul easier.
 * @param output The dense output. Its size must match input_tensor.cols*weight_tensor.rows.
 *
 * In practice there are almost always enough synapses and inputs that the output is dense.
 * Use top_k, gaussian, positive-only, etc. methods to sparsify the output before sending to another layer.
 */
template <typename SIZE_TYPE, typename VALUE_TYPE>
void sparse_linear_csr_csc_forward(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    VALUE_TYPE* output,
    bool train,
    //Value beta=0.9, // beta may be larger than optim beta to induce sparsity
    VALUE_TYPE solidify=0.01 // multiplies importances by 1.0+solidify so eventually they become very hard to remove
){
    #pragma omp parallel
    {
        for (SIZE_TYPE batch_number = 0; batch_number < input_tensor.rows; batch_number++)
        {
            #pragma omp for  // omp parallelization on input, not batch, because batch is assumed to be small
            for (SIZE_TYPE input_ptr = input_tensor.ptrs[0][batch_number]; input_ptr < input_tensor.ptrs[0][batch_number + 1]; input_ptr++)
            {
                SIZE_TYPE input_index = input_tensor.indices[0][input_ptr];
                VALUE_TYPE input_value = input_tensor.values[0][input_ptr];

                for (SIZE_TYPE weight_ptr = weights.connections.ptrs[0][input_index]; weight_ptr < weights.connections.ptrs[0][input_index + 1]; weight_ptr++)
                {
                    // Note: For CSC, we're using 'input_index' as the row index due to the transpose nature of CSC
                    VALUE_TYPE weight_value = weights.connections.values[0][weight_ptr];
                    SIZE_TYPE output_index = weights.connections.indices[0][weight_ptr];
                    auto weight_contribution = weight_value*input_value;
                    if(train){
                        //importances should be third, after backprop
                        weights.connections.values[2][weight_ptr] = weights.connections.values[2][weight_ptr] + weight_contribution*solidify;
                    }

                    #pragma omp atomic
                    output[output_index] += weight_contribution;
                }
            }
        }
    }
}


template <typename SIZE_TYPE, typename VALUE_TYPE>
void outer_product_spv_coo(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& output_gradient_tensor,
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen  // asumes gen is pre-reserved
    ){

    for (SIZE_TYPE batch = 0; batch < input_tensor.rows; batch++) {
        SIZE_TYPE in_len = input_tensor.ptrs[0][batch+1]-input_tensor.ptrs[0][batch];
        SIZE_TYPE out_len = output_gradient_tensor.ptrs[0][batch+1]-output_gradient_tensor.ptrs[0][batch];
        SIZE_TYPE prod_start_pos = input_tensor.ptrs[0][batch]*output_gradient_tensor.ptrs[0][batch];

        #pragma omp parallel for
        for (SIZE_TYPE i = 0; i < in_len*out_len; i++) {
            SIZE_TYPE x = i / out_len;
            SIZE_TYPE y = i % out_len;
            SIZE_TYPE global_id = prod_start_pos+x*out_len+y;

            // Compute the new weight value in COO format (indices0=cols, indices1=rows)
            gen.indices[0][global_id] = input_tensor.indices[0][input_tensor.ptrs[0][batch] + x]; // input index
            gen.indices[1][global_id] = output_gradient_tensor.indices[0][output_gradient_tensor.ptrs[0][batch] + y]; // output index
            gen.values[0][global_id] = input_tensor.values[0][input_tensor.ptrs[0][batch] + x]*output_gradient_tensor.values[0][output_gradient_tensor.ptrs[0][batch] + y]; // value
        }
    }
}

/*template <typename Index, typename Value>
std::vector<std::tuple<Index, Index, Value>> coalesce_coo(
    const std::vector<std::tuple<Index, Index, Value>>& coo_tuples)
{
    // Sort COO tuples
    std::vector<std::tuple<Index, Index, Value>> sorted_coo = coo_tuples;
    std::sort(coo_tuples.begin(), coo_tuples.end());

    // Handle duplicates by summing values
    std::vector<std::tuple<Index, Index, Value>> unique_coo;
    for (const auto& entry : sorted_coo) {
        if ((unique_coo.empty() ||
            std::get<0>(entry) != std::get<0>(unique_coo.back()) ||
            std::get<1>(entry) != std::get<1>(unique_coo.back())) &&
            std::get<0>(entry) !=Index(-1) && std::get<1>(entry) !=Index(-1)) {  // ignore invalid entries
            unique_coo.push_back(entry);
        } else {
            std::get<2>(unique_coo.back()) += std::get<2>(entry);
        }
    }

    return unique_coo;
}*/

template <typename SIZE_TYPE, typename VALUE_TYPE>
CSRSynaptogenesis<SIZE_TYPE, VALUE_TYPE> coo_to_csc(
    const COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen_coo,
    const int num_cpus)
{
    SIZE_TYPE *accum = new SIZE_TYPE[gen_coo.cols];
    std::fill(accum,accum+gen_coo.cols,0);
//accumulate parallel
    //thx: https://stackoverflow.com/a/70625541
    if(num_cpus>1){
    SIZE_TYPE *thr_accum = new SIZE_TYPE[num_cpus*(gen_coo.cols)];
    std::fill(thr_accum, thr_accum + gen_coo.cols*num_cpus, 0);
    #pragma omp parallel shared(accum, thr_accum, cols) num_threads(num_cpus)
  {
    int thread = omp_get_thread_num(),
      myfirst = thread*(gen_coo.cols);
    #pragma omp for
    for ( int i=0; i<gen_coo.nnz(); i++ )
      thr_accum[ myfirst+gen_coo.indices[1][i] ]++;
    #pragma omp for
    for ( int igrp=0; igrp<(gen_coo.cols); igrp++ )
      for ( int t=0; t<num_cpus; t++ )
        accum[igrp] += thr_accum[ t*(gen_coo.cols)+igrp ];
  }
    }else{
        for ( int i=0; i<gen_coo.nnz(); i++ )
            accum[ gen_coo.indices[1][i] ]++;
    }

    SIZE_TYPE *ptrs = new SIZE_TYPE[gen_coo.cols + 1];
    SIZE_TYPE scan_a = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i < gen_coo.cols + 1; i++) {
        ptrs[i] = scan_a;
#pragma omp scan exclusive(scan_a)
        { scan_a += accum[i]; }
    }

    delete[] accum;
    delete[] gen_coo.indices[1];

    // Return the new CSC matrix
    CSRSynaptogenesis<SIZE_TYPE, VALUE_TYPE> return_csc;
    return_csc.cols = gen_coo.rows;
    return_csc.rows = gen_coo.cols;
    return_csc.ptrs[0].reset(ptrs);
    return_csc.indices[0].reset(gen_coo.indices[0]);
    return_csc.values[0].reset(gen_coo.values[0]);

    return return_csc;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
CSRSynaptogenesis<SIZE_TYPE, VALUE_TYPE> generate_new_weights_csc(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& output_gradient_tensor,
    const int num_cpus)
{
    // SECTION 1: Create COO from top input and outputs in sparse vectors
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen_coo;
    gen_coo.ptrs = input_tensor.nnz()*output_gradient_tensor.nnz();
    gen_coo.indices[0].reset(new SIZE_TYPE[gen_coo.nnz()]);
    gen_coo.indices[1].reset(new SIZE_TYPE[gen_coo.nnz()]);
    gen_coo.values[0].reset(new VALUE_TYPE[gen_coo.nnz()]);
    gen_coo.cols = output_gradient_tensor.rows;
    gen_coo.rows = input_tensor.rows;
    outer_product_spv_coo(
        input_tensor,
        output_gradient_tensor,
        gen_coo
    );

    //SECTION 2: sort COO and merge duplicates
    auto duplicates = merge_sort_coo(gen_coo.indices[0], gen_coo.indices[1], gen_coo.values[0], gen_coo.nnz());
    gen_coo.ptrs = gen_coo.ptrs - duplicates; //update nnz

    //SECTION 3: convert COO to CSC
    auto csc_out = coo_to_csc(gen_coo, num_cpus);
    return csc_out;
}

/**
 * Perform a backward pass of a sparse linear layer with sparse inputs and sparse gradients.
 * This assumes the inputs and gradients were already masked, and will generate a new csc
 * with input.size*gradient.size synapses for addition/merging to the weight csc
 *
 */
template <class SIZE_TYPE, class VALUE_TYPE>
void sparse_linear_vectorized_backward_is(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& in_tensor,  // this should be a fraction of active inputs for potentially making in*out new synapses
    const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& out_grad, // this should be a fraction of output gradients for potentially making in*out new synapses
    VALUE_TYPE* input_gradients,
    VALUE_TYPE* output_gradients, // this is the full gradient for computing the full input gradient
    const int num_cpus)
{
    SIZE_TYPE batch_size = in_tensor.rows;
    if (in_tensor.nnz()>0 && out_grad.nnz()>0){
        // todo: after generate_new_weights_csc is optimized to use CSRs more instead of COOs, skip indices already in weight_tensor
        weights.probes = generate_new_weights_csc(in_tensor, out_grad, num_cpus);
    }

#pragma omp parallel num_threads(num_cpus)
    {
        for (SIZE_TYPE batch = 0; batch < in_tensor.rows; batch++)
        {
#pragma omp for
            for (SIZE_TYPE input_ptr = in_tensor.ptrs[0][batch]; input_ptr < in_tensor.ptrs[0][batch + 1]; input_ptr++)
            {
                auto input_index = in_tensor.indices[0][input_ptr];
                auto input_value = in_tensor.values[0][input_ptr];
                for (SIZE_TYPE weight_ptr = weights.connections.ptrs[0][input_index]; weight_ptr < weights.connections.ptrs[0][input_index + 1];
                     weight_ptr++)
                {
                    auto weight_value = weights.connections.values[0][weight_ptr];
                    auto output_index = weights.connections.indices[0][weight_ptr];
                    input_gradients[input_index] += weight_value * output_gradients[output_index * batch_size + batch];
                    weights.connections.values[1][weight_ptr] += output_gradients[output_index * batch_size + batch] * input_value;  // gradients for an echo state network, stored after values
                }
            }
        }
    }
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void optimize_weights_with_importance(
    const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE learning_rate,
    const SIZE_TYPE max_weights,
    const int num_cpus,
    const VALUE_TYPE beta = 0.1 // rate for importance updates
) {

    VALUE_TYPE* importance_tensor = new VALUE_TYPE[weights.probes.nnz()];
    #pragma omp parallel num_threads(num_cpus)
    for (int weight_ptr = 0; weight_ptr < weights.probes.nnz(); weight_ptr++) {
        // weight_activation will always be zero since there was no weight yet, so just use 0-weight_error instead
        VALUE_TYPE weight_error = weights.probes.values[0][weight_ptr] / learning_rate;
        VALUE_TYPE weight_instant_importance = - weight_error;

        // Update importance tensor
        importance_tensor[weight_ptr] = weight_instant_importance;
    }

    //1. move indices and values from probes to new CSR, non-copy
    //2. update probes to have values[0]=0.0 values, values[1]=0.0 backprop, values[2]=importance
    //3. combine with merge_csrs
    //3a. copy algorithm from merge_csr and modify
    //3b. update parallel_merge_sorted_coos to work with new COO structure

    // Convert CSR to COO format if required
    auto coo_weights = weight_tensor.to_coo();
    auto coo_updates = weight_gradients.to_coo();

    // Allocate arrays for merged weights
    size_t merged_size = coo_weights.nnz() + coo_updates.nnz();
    Index* merged_rows = new Index[merged_size];
    Index* merged_cols = new Index[merged_size];
    Value* merged_values = new Value[merged_size];

    // Merge weights and updates
    /* todo: importance values need to be merged and affect merging:
     *   weight_importance = weight_importance * beta + gradient_importance * (1-beta)
     *   weights = weights - learning_rate * gradients / (weight_importance + epsilon)
     */
    //   so, if two same index weights show up, divide them by their importance
    size_t new_nnz = parallel_merge_sorted_coos(
        coo_weights.rows, coo_weights.cols, coo_weights.values*-learning_rate , coo_weights.importances * beta,
        coo_updates.rows, coo_updates.cols, coo_updates.values, coo_gradients.importances * beta),
        merged_rows, merged_cols, merged_values,
        coo_weights.nnz(), coo_updates.nnz(), num_cpus);

    // Check if pruning is required
    if (new_nnz > max_weights) {
        coo_subtract_bottom_k(
            merged_cols, merged_rows, merged_values.data(),
            importance_tensor.values,
            weight_tensor.cols, weight_tensor.rows, weight_tensor.values,
            importance_tensor.values,
            new_nnz, new_nnz - max_weights, num_cpus
        );
    }

    weights.connections = merge_csrs(weights.connections, weights.probes);
}