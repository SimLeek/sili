#include "csr.hpp"
#include "coo.hpp"
#include <cstdlib>
#include <new>
#include <vector>

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
    VALUE_TYPE solidify = 0.01,
    const int num_cpus=4 // usually good default even in the case of many more cpus
){
    SIZE_TYPE num_outputs = input_tensor.rows * weights.connections.cols;

    #pragma omp parallel reduction(+:output[:num_outputs]) num_threads(num_cpus)
    {
        for (SIZE_TYPE batch_number = 0; batch_number < input_tensor.rows; batch_number++) {
            // reset this to 0 every batch
            std::vector<VALUE_TYPE> thread_output(weights.connections.cols, 0);
            #pragma omp for  // Parallelize across inputs
            for (SIZE_TYPE input_ptr = input_tensor.ptrs[0][batch_number]; 
                 input_ptr < input_tensor.ptrs[0][batch_number + 1]; 
                 input_ptr++) 
            {
                SIZE_TYPE input_index = input_tensor.indices[0][input_ptr];
                VALUE_TYPE input_value = input_tensor.values[0][input_ptr];

                for (SIZE_TYPE weight_ptr = weights.connections.ptrs[0][input_index]; 
                     weight_ptr < weights.connections.ptrs[0][input_index + 1]; 
                     weight_ptr++) 
                {
                    // Compute contribution
                    VALUE_TYPE weight_value = weights.connections.values[0][weight_ptr];
                    SIZE_TYPE output_index = weights.connections.indices[0][weight_ptr];
                    auto weight_contribution = weight_value * input_value;

                    if (train) {
                        //weights are unique per input-output combo and batches are sequential here, so atomic isn't necessary
                        // keeping the for on the inputs is important!
                        weights.connections.values[2][weight_ptr] += weight_contribution * solidify; 
                    }

                    // Accumulate in thread-private buffer
                    thread_output[output_index] += weight_contribution;
                }
            }

            //don't prallelize here, because the specific thread needs to copy its full output, not split further
            for (SIZE_TYPE i = batch_number * weights.connections.cols; i < (batch_number+1) * weights.connections.cols; i++) {
                //don't use pragma omp atomic here either. reduction takes care of that: "reduction(+:output[:num_outputs])"
                output[i] += thread_output[i-(batch_number * weights.connections.cols)];
            }
        }
    }
}


template <typename SIZE_TYPE, typename VALUE_TYPE>
void outer_product_spv_coo(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& output_gradient_tensor,
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>& gen  // asumes gen is pre-reserved
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

/*template <typename SIZE_TYPE, typename VALUE_TYPE>
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
}*/

template <typename SIZE_TYPE, typename VALUE_TYPE>
COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> generate_new_weights_csc(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& output_gradient_tensor,
    const int num_cpus=4)
{

    // Calculate actual space to reserve based on Hadamard product of row lengths
    SIZE_TYPE total_reserve = 0;
    for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
        SIZE_TYPE in_len = input_tensor.ptrs[0][batch + 1] - input_tensor.ptrs[0][batch];
        SIZE_TYPE out_len = output_gradient_tensor.ptrs[0][batch + 1] - output_gradient_tensor.ptrs[0][batch];
        total_reserve += in_len * out_len;
    }

    if(total_reserve==0){
        return COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>();
    }

    // SECTION 1: Create COO from top input and outputs in sparse vectors
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen_coo;
    gen_coo.ptrs = total_reserve;
    gen_coo.indices[0].reset(new SIZE_TYPE[gen_coo.nnz()]{0});
    gen_coo.indices[1].reset(new SIZE_TYPE[gen_coo.nnz()]{0});
    gen_coo.values[0].reset(new VALUE_TYPE[gen_coo.nnz()]{0});
    gen_coo.cols = output_gradient_tensor.cols;
    gen_coo.rows = input_tensor.cols;
    outer_product_spv_coo(
        input_tensor,
        output_gradient_tensor,
        gen_coo
    );

    //SECTION 2: sort COO and merge duplicates
    auto duplicates = merge_sort_coo(gen_coo.indices, gen_coo.values, gen_coo.nnz());
    gen_coo.ptrs -= duplicates; //update nnz

    //SECTION 3: convert COO to CSC (swap rows/cols in the coo for to_csr to work)
    //auto csc_out = to_csr(gen_coo, num_cpus);
    return gen_coo;
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
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& out_grad_synaptogenesis, // this should be a fraction of output gradients for potentially making in*out new synapses
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& out_grad_sparse, // this should be a fraction of output gradients for backpropogating to the input gradient, can be same as prev or different
    VALUE_TYPE* input_gradients,
    VALUE_TYPE* output_gradients, // this is the full dense gradient. This makes computing weight gradients fast and you already have it anyway.
    const int num_cpus)
{
    SIZE_TYPE batch_size = in_tensor.rows;
    if (in_tensor.nnz()>0 && out_grad_synaptogenesis.nnz()>0){
        // todo2: after generate_new_weights_csc is optimized to use CSRs more instead of COOs, skip indices already in weight_tensor
        // todo2: handle the case where weights.probes already exists and merge potential weights into probes
        //todo: need to keep only the important value arrays for this and drop the others. Make a view op for it
        weights.probes = generate_new_weights_csc(in_tensor, out_grad_synaptogenesis, num_cpus);
    }
//todo: use a vector of pointers to keep the output ptr
    std::vector<SIZE_TYPE> out_grad_ptrs(weights.connections.rows, 0);

    #pragma omp parallel num_threads(num_cpus) shared(out_grad_ptrs, in_tensor)
    {
        for (SIZE_TYPE batch = 0; batch < in_tensor.rows; batch++)
        {
            std::fill(out_grad_ptrs.begin(), out_grad_ptrs.end(), 0);  //todo: fill sections in every thread

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
                    //don't handle input grad here, because it is not parallel. The same input will be accessed by different batches.
                    // it also limits it to only activated inputs, but output grad could backprop to connected inputs that should've fired
                    // wrong code: input_gradients[input_index] += weight_value * output_gradients[output_index * batch_size + batch];
                    //anyway, this loop is for weight backprop:
                    weights.connections.values[1][weight_ptr] += output_gradients[output_index * batch_size + batch] * input_value;  // gradients for an echo state network, stored after values
                }
            }
            
            for (SIZE_TYPE output_ptr = out_grad_sparse.ptrs[0][batch]; output_ptr < out_grad_sparse.ptrs[0][batch + 1]; output_ptr++)
            {
                auto output_index = out_grad_sparse.indices[0][output_ptr];
                auto output_grad = out_grad_sparse.values[0][output_ptr];

                #pragma omp for // also parallel w/ respect to input
                for (SIZE_TYPE input_index = 0; input_index < weights.connections.rows; input_index++)
                {
                    auto full_ptr = weights.connections.ptrs[0][input_index]+out_grad_ptrs[input_index];
                    while(full_ptr<weights.connections.ptrs[0][input_index+1] && weights.connections.indices[0][full_ptr]<output_index){
                        out_grad_ptrs[input_index]+=1;
                        full_ptr = weights.connections.ptrs[0][input_index]+out_grad_ptrs[input_index];
                    }
                    if(weights.connections.indices[0][full_ptr]>output_index || full_ptr>=weights.connections.ptrs[0][input_index+1]){
                        continue; // no synapse connection to this output, move to next input
                    }
                    auto weight_value = weights.connections.values[0][full_ptr];
                    //auto output_index = weights.connections.indices[0][full_ptr];
                    //shouldn't need a parallel reduction, because the inputs are all accessed in parallel
                    input_gradients[input_index] += weight_value * output_grad;
                }
            }
        }
    }
}

/*
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
    if (in_tensor.nnz() > 0 && out_grad.nnz() > 0) {
        // Skip indices already in weight_tensor or merge potential weights into probes if they exist
        weights.probes = generate_new_weights_csc(in_tensor, out_grad, num_cpus);
    }

    SIZE_TYPE num_inputs = weights.connections.rows;
    SIZE_TYPE num_outputs = weights.connections.cols;

    SIZE_TYPE total_inputs = num_inputs*batch_size;

    #pragma omp parallel reduction(+:input_gradients[:total_inputs]) num_threads(num_cpus)
    {
        // Thread-private buffer for weight updates
        std::vector<VALUE_TYPE> thread_input_gradients(in_tensor.cols, 0);

        for (SIZE_TYPE batch = 0; batch < batch_size; batch++) {
            // Reset thread-private weight updates for this batch (todo: move this to the parallel for)
            std::fill(thread_input_gradients.begin(), thread_input_gradients.end(), 0);

            #pragma omp for  // Parallelize across inputs
            for (SIZE_TYPE input_ptr = in_tensor.ptrs[0][batch]; //NOTE: we are only learning for these specific inputs
                 input_ptr < in_tensor.ptrs[0][batch + 1]; 
                 input_ptr++) 
            {
                auto input_index = in_tensor.indices[0][input_ptr];
                auto input_value = in_tensor.values[0][input_ptr];

                for (SIZE_TYPE weight_ptr = weights.connections.ptrs[0][input_index]; 
                     weight_ptr < weights.connections.ptrs[0][input_index + 1]; 
                     weight_ptr++) 
                {
                    auto weight_value = weights.connections.values[0][weight_ptr];
                    auto output_index = weights.connections.indices[0][weight_ptr];

                    // Compute gradient contribution
                    VALUE_TYPE grad_contribution = output_gradients[output_index * batch_size + batch] * weight_value;

                    // Accumulate into thread-private buffer
                    thread_input_gradients[input_index] += grad_contribution;

                    // Store weight gradients
                    weights.connections.values[1][weight_ptr] += 
                        output_gradients[batch*weights.connections.rows +output_index] * input_value;
                }
            }

            // Apply accumulated thread-local input updates
            for (SIZE_TYPE i = batch * num_inputs; i < (batch+1) * num_inputs; i++) {
                //don't use pragma omp atomic here either. reduction takes care of that: "reduction(+:output[:num_outputs])"
                input_gradients[i] += thread_input_gradients[i-(batch * num_inputs)];
            }
        }
    }
}*/

template <typename SIZE_TYPE, typename VALUE_TYPE>
void optim_weights(
    const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE learning_rate,
    const int num_cpus,
    const VALUE_TYPE beta = 0.1 // rate for importance updates
) {
#pragma omp parallel num_threads(num_cpus)
for (SIZE_TYPE weight_ptr = weights.connections.ptrs[0][0]; weight_ptr < weights.connections.ptrs[0][weights.connections.rows];
                     weight_ptr++)
                {
                    auto weight_value = weights.connections.values[0][weight_ptr];
                    //weight += grad*-lr/(1+abs(conn_str))
                    weights.connections.values[0][weight_ptr] += (weights.connections.values[1][weight_ptr]*-learning_rate)/(1+std::abs(weights.connections.values[2][weight_ptr]));
                     weights.connections.values[1][weight_ptr] = 0; // reset grad so we don't build it up.
                }
}

//todo: split this up so different importance/optim handling types can be handled in python
template <typename SIZE_TYPE, typename VALUE_TYPE>
void optim_synaptogenesis(
    const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE learning_rate,
    const SIZE_TYPE max_weights,
    const int num_cpus,
    const VALUE_TYPE beta = 0.1 // rate for importance updates
) {

    //STEP 1: convert probes sparse_struct type to connection sparse_struct type
    VALUE_TYPE* importance_tensor = new VALUE_TYPE[weights.probes.nnz()];
    #pragma omp parallel num_threads(num_cpus)
    for (int weight_ptr = 0; weight_ptr < weights.probes.nnz(); weight_ptr++) {
        // weight_activation will always be zero since there was no weight yet, so just use 0-weight_error instead
        VALUE_TYPE weight_error = weights.probes.values[0][weight_ptr] / learning_rate;
        VALUE_TYPE weight_instant_importance = - weight_error;

        // Update importance tensor
        importance_tensor[weight_ptr] = weight_instant_importance * beta;
    }

    /*decltype(weights.connections) new_connections;
    using index_type = stdarr_of_uniqarr_type<decltype(new_connections.indices)>;
    using value_type = stdarr_of_uniqarr_type<decltype(new_connections.values)>;
    for (std::size_t idx = 0; idx < weights.probes.num_indices; ++idx) {
        std::get<idx>(new_connections.indices).reset(std::get<idx>(weights.probes.indices));
    }
    for (std::size_t valIdx = 0; valIdx < weights.probes.num_values-1; ++valIdx) {
        std::get<valIdx>(new_connections.values).reset(new value_type[weights.probes.indices]{0}); // todo: if desired, set idx0: connection value
    }
    std::get<weights.probes.num_values-1>(new_connections.values).reset(importance_tensor);*/

    // STEP 2: Convert CSR to COO format
    auto coo_weights = to_coo(weights.connections, num_cpus);
    //auto coo_updates = to_coo(new_connections, num_cpus);

    // STEP 3: Allocate arrays for merged weights
    // note: certain architectures may prefer the slower nlogn method due to closer memory
    // todo: pull this out into a "reserve COO for merging" function
    size_t merged_size = coo_weights.nnz() + weights.probes.nnz();
    decltype(coo_weights) merged_coo;
    using index_type2 = stdarr_of_uniqarr_type<decltype(merged_coo.indices)>;
    using value_type2 = stdarr_of_uniqarr_type<decltype(merged_coo.values)>;

    for (std::size_t idx = 0; idx < merged_coo.num_indices; ++idx) {
        std::get<idx>(merged_coo.indices).reset(new index_type2[merged_size]);
    }
    for (std::size_t valIdx = 0; valIdx < merged_coo.num_values; ++valIdx) {
        std::get<valIdx>(merged_coo.values).reset(new value_type2[merged_size]);
    }

    // STEP 4: Merge weights
    /* this does:
     *   weight_importance = weight_importance * beta + gradient_importance
     */
    size_t duplicates = parallel_merge_sorted_coos(
        coo_weights.indices, coo_weights.values, 
        weights.probes.indices, weights.probes.values ,
        merged_coo.indices, merged_coo.values, 
        coo_weights.nnz(), weights.probes.nnz(),
        num_cpus);

    size_t new_nnz = coo_weights.nnz() + weights.probes.nnz() - duplicates;
    merged_coo.ptrs = new_nnz;

    // Check if pruning is required
    if (new_nnz > max_weights) {
        if(max_weights>weights.connections.nnz()){
            decltype(coo_weights) weight_out_container;

            for (std::size_t idx = 0; idx < weight_out_container.num_indices; ++idx) {
                std::get<idx>(weight_out_container.indices).reset(new stdarr_of_uniqarr_type<decltype(weight_out_container.indices)>[max_weights]);
            }
            for (std::size_t valIdx = 0; valIdx < weight_out_container.num_values; ++valIdx) {
                std::get<valIdx>(weight_out_container.values).reset(new stdarr_of_uniqarr_type<decltype(weight_out_container.values)>[max_weights]);
            }
            coo_subtract_bottom_k(
            merged_coo.indices, merged_coo.values, 
            weight_out_container.indices, weight_out_container.values,
            new_nnz, new_nnz - max_weights, num_cpus
            );
            weights.connections = to_csr(weight_out_container, num_cpus);
        }else{
            coo_subtract_bottom_k(
                merged_coo.indices, merged_coo.values, 
                coo_weights.indices, coo_weights.values,
                new_nnz, new_nnz - max_weights, num_cpus
            );
            weights.connections = to_csr(coo_weights, num_cpus);
        }
    }else{
        weights.connections = to_csr(merged_coo, num_cpus);
    }
    clear_coo(weights.probes);
    weights.probes.rows = weights.connections.rows;
    weights.probes.cols = weights.connections.cols;
}