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
template <typename Index, typename Value>
void sparse_linear_csr_csc_forward(
    const csr_struct<Index, Value>& input_tensor,
    const csr_struct<Index, Value>& weight_tensor,
    Value* output,
){
    #pragma omp parallel
    {
        for (Index batch_number = 0; batch_number < input_tensor.rows; batch_number++)
        {
            #pragma omp for  // omp parallelization on input, not batch, because batch is assumed to be small
            for (Index input_ptr = input_tensor.ptrs[batch_number]; input_ptr < input_tensor.ptrs[batch_number + 1]; input_ptr++)
            {
                Index input_index = input_tensor.indices[input_ptr];
                Value input_value = input_tensor.values[input_ptr];

                for (Index weight_ptr = weight_tensor.ptrs[input_index]; weight_ptr < weight_tensor.ptrs[input_index + 1]; weight_ptr++)
                {
                    // Note: For CSC, we're using 'input_index' as the row index due to the transpose nature of CSC
                    Value weight_value = weight_tensor.values[weight_ptr];
                    Index output_index = weight_tensor.indices[weight_ptr];
                    #pragma omp atomic
                    output[output_index] += weight_value*input_value;
                }
            }
        }
    }
}


template <typename Index, typename Value>
void outer_product_spv_coo(
    const csr_struct<Index, Value>& input_tensor,
    const csr_struct<Index, Value>& output_gradient_tensor,
    Index* cols,
    Index* rows,
    Value* vals
    ){

    for (Index batch = 0; batch < input_tensor.rows; batch++) {
        Index in_len = input_tensor.ptrs[batch+1]-input_tensor.ptrs[batch];
        Index out_len = output_gradient_tensor.ptrs[batch+1]-output_gradient_tensor.ptrs[batch];
        Index prod_start_pos = input_tensor.ptrs[batch]*output_gradient_tensor.ptrs[batch];

        #pragma omp parallel for
        for (Index i = 0; i < in_len*out_len; i++) {
            Index x = i % top_x;
            Index y = i / top_x;
            Index global_id = prod_start_pos+y*top_x+x;

            // Compute the new weight value in COO format
            cols[global_id] = input_m_indices[input_batch_offsets[batch + 1] - x]; // input index
            rows[global_id] = output_gradient_indices[output_gradient_batch_offsets[batch + 1] - y]; // output index
            vals[global_id] = input_values[input_batch_offsets[batch + 1] - x]*output_gradient_values[output_gradient_batch_offsets[batch + 1] - y] // value
            );
        }
    }
}

template <typename Index, typename Value>
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
}

template <typename Index, typename Value>
csr_struct<Index, Value> coo_to_csc(
    const Index *cols,
    const Index *rows,
    const Value *vals,
    Index num_rows,
    Index num_cols,
    Index nnz,  //nnz AFTER duplicates were removed
    const int num_cpus)
{
    SIZE_TYPE *accum = new SIZE_TYPE[num_cols];
    std::fill(accum,accum+num_cols,0);
//accumulate parallel
    //thx: https://stackoverflow.com/a/70625541
    if(num_cpus>1){
    SIZE_TYPE *thr_accum = new SIZE_TYPE[num_cpus*(num_cols)];
    std::fill(thr_accum, thr_accum + num_cols*num_cpus, 0);
    #pragma omp parallel shared(accum, thr_accum, cols) num_threads(num_cpus)
  {
    int thread = omp_get_thread_num(),
      myfirst = thread*(num_cols);
    #pragma omp for
    for ( int i=0; i<nnz; i++ )
      thr_accum[ myfirst+cols[i] ]++;
    #pragma omp for
    for ( int igrp=0; igrp<(num_cols); igrp++ )
      for ( int t=0; t<num_cpus; t++ )
        accum[igrp] += thr_accum[ t*(num_cols)+igrp ];
  }
    }else{
        for ( int i=0; i<nnz; i++ )
            accum[ cols[i] ]++;
    }

    SIZE_TYPE *ptrs = new SIZE_TYPE[num_cols + 1];
    SIZE_TYPE scan_a = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i < num_cols + 1; i++) {
        ptrs[i] = scan_a;
#pragma omp scan exclusive(scan_a)
        { scan_a += accum[i]; }
    }

    delete[] accum;
    delete[] cols;

    // Return the new CSC matrix
    csr_struct<SIZE_TYPE, VALUE_TYPE> return_csc;
    return_csc.cols = num_cols;
    return_csc.rows = num_rows;
    return_csc.ptrs.reset(ptrs);
    return_csc.indices.reset(rows);
    return_csc.values.reset(vals);

    return return_csc;
}

template <typename Index, typename Value>
csr_struct<Index, Value> generate_new_weights_csc(
    const csr_struct<Index, Value>& input_tensor,
    const csr_struct<Index, Value>& output_gradient_tensor,
    const int num_cpus)
{
    // SECTION 1: Create COO from top input and outputs in sparse vectors
    Index *cols = new Index[input_tensor.nnz()*output_gradient_tensor.nnz()],
    Index *rows = new Index[input_tensor.nnz()*output_gradient_tensor.nnz()],
    Value *vals = new Value[input_tensor.nnz()*output_gradient_tensor.nnz()],
    outer_product_spv_coo(
        input_tensor,
        output_gradient_tensor,
        cols, rows, vals
    );

    //SECTION 2: sort COO and merge duplicates
    auto duplicates = merge_sort_coo(cols, rows, vals, input_tensor.nnz()*output_gradient_tensor.nnz())

    //SECTION 3: convert COO to CSC
    Index total_elements = input_tensor.nnz()*output_gradient_tensor.nnz();
    Index weight_cols = output_gradient_tensor.cols;
    Index weight_rows = input_tensor.cols;
    return coo_to_csc(
        cols,
        rows,
        vals,
        weight_rows,
        weight_cols,
        total_elements-duplicates,
        num_cpus);
}

/**
 * Perform a backward pass of a sparse linear layer with sparse inputs and sparse gradients.
 * This assumes the inputs and gradients were already masked, and will generate a new csc
 * with input.size*gradient.size synapses for addition/merging to the weight csc
 *
 * @param GenWeights Whether this function should generate 0 weights in weight_tensor so backprop can be applied
 */
template <const bool GenWeights>
void sparse_linear_vectorized_backward_is(
    const csr_struct<Index, Value>& input_tensor,  // this should be a fraction of active inputs for potentially making in*out new synapses
    const csr_struct<Index, Value>& weight_tensor,
    const csr_struct<Index, Value>& output_gradients_reduced, // this should be a fraction of output gradients for potentially making in*out new synapses
    Value* input_gradients,
    Value* output_gradients, // this is the full gradient for computing the full input gradient
    csr_struct<Index, Value>& weight_gradients,
    const int num_cpus)
{
    weight_gradients = generate_new_weights_csc(input_tensor, output_gradients_reduced, num_cpus);

#pragma omp parallel
    {
        for (int batch = 0; batch < input_tensor.rows; batch++)
        {
#pragma omp for
            for (int input_ptr = input_tensor.ptrs[batch]; input_ptr < input_tensor.ptrs[batch + 1]; input_ptr++)
            {
                int input_index = input_m_indices[input_ptr];
                auto input_value = input_values[input_ptr];
                for (int weight_ptr = weight_col_offsets[input_index]; weight_ptr < weight_col_offsets[input_index + 1];
                     weight_ptr++)
                {
                    auto weight_value = weight_values[weight_ptr];
                    int output_index = weight_output_indices[weight_ptr];
                    input_gradients[input_ptr] += weight_value * output_gradients[output_index * batch_size + batch];
                }
            }
        }
    }
}