#include "../sili/lib/headers/linear_sisldo.hpp"
#include "tests_main.h"
#include <vector>

/* #region sparse_linear_csr_csc_forward */

TEST_CASE("Sparse Linear CSR-CSC Forward", "[sparse_linear_csr_csc_forward]") {
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    // Input Tensor (CSR)
    CSRInput<SIZE_TYPE, VALUE_TYPE> input_tensor;
    input_tensor.rows = 2;  // 2 batches
    input_tensor.cols = 4;  // 4 features
    input_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    input_tensor.indices = {std::make_unique<SIZE_TYPE[]>(4)};
    input_tensor.values = {std::make_unique<VALUE_TYPE[]>(4)};

    SIZE_TYPE input_ptrs_data[] = {0, 2, 4};
    SIZE_TYPE input_indices_data[] = {0, 2, 1, 2};
    VALUE_TYPE input_values_data[] = {1.0, 0.5, 2.0, 1.5};

    std::copy(input_ptrs_data, input_ptrs_data + 3, input_tensor.ptrs[0].get());
    std::copy(input_indices_data, input_indices_data + 4, input_tensor.indices[0].get());
    std::copy(input_values_data, input_values_data + 4, input_tensor.values[0].get());

    // Weights (CSC)
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections.rows = 4;  // 4 input features
    weights.connections.cols = 3;  // 3 output features
    weights.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(5)};
    weights.connections.indices = {std::make_unique<SIZE_TYPE[]>(6)};
    weights.connections.values = {
        std::make_unique<VALUE_TYPE[]>(6),
        std::make_unique<VALUE_TYPE[]>(6),
        std::make_unique<VALUE_TYPE[]>(6) // importances
    };

    SIZE_TYPE weight_ptrs_data[] = {0, 2, 4, 6};
    SIZE_TYPE weight_indices_data[] = {0, 1, 0, 2, 1, 2};
    VALUE_TYPE weight_values_data[] = {0.5, 1.0, 0.3, 0.7, 0.2, 0.4};
    VALUE_TYPE weight_importances_data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::copy(weight_ptrs_data, weight_ptrs_data + 4, weights.connections.ptrs[0].get());
    std::copy(weight_indices_data, weight_indices_data + 6, weights.connections.indices[0].get());
    std::copy(weight_values_data, weight_values_data + 6, weights.connections.values[0].get());
    std::copy(weight_importances_data, weight_importances_data + 6, weights.connections.values[2].get());

    // Output
    VALUE_TYPE output[6] = {0.0, 0.0, 0.0, 0, 0, 0};
    VALUE_TYPE expected_output[] = {0.5, 1.1, 0.2, 0.6, 0.3, 2.0};

    // Train flag
    bool train = true;
    VALUE_TYPE solidify = 0.01;

    // Run forward pass
    sparse_linear_csr_csc_forward(input_tensor, weights, output, train, solidify);

    // Verify output
    CHECK_VECTOR_ALMOST_EQUAL(std::vector<VALUE_TYPE>(output, output + 6), std::vector<VALUE_TYPE>(expected_output, expected_output + 6));

    // Verify importances updated
    VALUE_TYPE expected_importances[] = {0.005, 0.01, 0.006, 0.014, 0.004, 0.008};
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[2].get(), weights.connections.values[2].get() + 6),
        std::vector<VALUE_TYPE>(expected_importances, expected_importances + 6)
    );
}

/* #endregion */

/* #region outer_product_spv_coo */

TEST_CASE("Outer Product SPV COO", "[outer_product_spv_coo]") {
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    // Input Tensor (CSR)
    CSRInput<SIZE_TYPE, VALUE_TYPE> input_tensor;
    input_tensor.rows = 2;  // 2 batches
    input_tensor.cols = 4;  // 4 features
    input_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    input_tensor.indices = {std::make_unique<SIZE_TYPE[]>(4)};
    input_tensor.values = {std::make_unique<VALUE_TYPE[]>(4)};

    SIZE_TYPE input_ptrs_data[] = {0, 2, 4};
    SIZE_TYPE input_indices_data[] = {0, 2, 1, 3};
    VALUE_TYPE input_values_data[] = {1.0, 0.5, 2.0, 1.5};

    std::copy(input_ptrs_data, input_ptrs_data + 3, input_tensor.ptrs[0].get());
    std::copy(input_indices_data, input_indices_data + 4, input_tensor.indices[0].get());
    std::copy(input_values_data, input_values_data + 4, input_tensor.values[0].get());

    // Output Gradient Tensor (CSR)
    CSRInput<SIZE_TYPE, VALUE_TYPE> output_gradient_tensor;
    output_gradient_tensor.rows = 2;  // 2 batches
    output_gradient_tensor.cols = 3;  // 3 features
    output_gradient_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    output_gradient_tensor.indices = {std::make_unique<SIZE_TYPE[]>(4)};
    output_gradient_tensor.values = {std::make_unique<VALUE_TYPE[]>(4)};

    SIZE_TYPE output_ptrs_data[] = {0, 2, 4};
    SIZE_TYPE output_indices_data[] = {0, 1, 1, 2};
    VALUE_TYPE output_values_data[] = {0.5, 1.0, 1.5, 0.4};

    std::copy(output_ptrs_data, output_ptrs_data + 3, output_gradient_tensor.ptrs[0].get());
    std::copy(output_indices_data, output_indices_data + 4, output_gradient_tensor.indices[0].get());
    std::copy(output_values_data, output_values_data + 4, output_gradient_tensor.values[0].get());

    // Synaptogenesis Output (COO)
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen;
    SIZE_TYPE num_connections = 8;  // Input length * Output length
    gen.indices = {
        std::make_unique<SIZE_TYPE[]>(num_connections),
        std::make_unique<SIZE_TYPE[]>(num_connections)
    };
    gen.values = {std::make_unique<VALUE_TYPE[]>(num_connections)};

    // Expected Results
    SIZE_TYPE expected_indices_0[] = {0, 0, 2, 2, 1, 1, 3, 3};
    SIZE_TYPE expected_indices_1[] = {0, 1, 0, 1, 1, 2, 1, 2};
    VALUE_TYPE expected_values[] = {0.5, 1.0, 0.25, 0.5, 3.0, 0.8, 2.25, 0.6};

    // Run the function
    outer_product_spv_coo(input_tensor, output_gradient_tensor, gen);

    // Verify Results
    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(gen.indices[0].get(), gen.indices[0].get() + num_connections),
        std::vector<SIZE_TYPE>(expected_indices_0, expected_indices_0 + num_connections)
    );
    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(gen.indices[1].get(), gen.indices[1].get() + num_connections),
        std::vector<SIZE_TYPE>(expected_indices_1, expected_indices_1 + num_connections)
    );
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(gen.values[0].get(), gen.values[0].get() + num_connections),
        std::vector<VALUE_TYPE>(expected_values, expected_values + num_connections)
    );

    // Edge Case: Empty Input Tensor
    CSRInput<SIZE_TYPE, VALUE_TYPE> empty_tensor;
    empty_tensor.rows = 1;
    empty_tensor.cols = 0;
    empty_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(2)};
    empty_tensor.indices = {nullptr};
    empty_tensor.values = {nullptr};

    SIZE_TYPE empty_ptrs[] = {0, 0};
    std::copy(empty_ptrs, empty_ptrs + 2, empty_tensor.ptrs[0].get());

    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen_empty;
    gen_empty.indices = {
        std::make_unique<SIZE_TYPE[]>(0),
        std::make_unique<SIZE_TYPE[]>(0)
    };
    gen_empty.values = {std::make_unique<VALUE_TYPE[]>(0)};

    // Run the function
    outer_product_spv_coo(empty_tensor, output_gradient_tensor, gen_empty);

    // There's nothing to verify, but if it didn't crash then it worked
}

/* #endregion */

/* #region outer_product_spv_coo*/
/*
TEST_CASE("Generate New Weights CSC", "[generate_new_weights_csc]") {
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    // Input Tensor (CSR)
    CSRInput<SIZE_TYPE, VALUE_TYPE> input_tensor;
    input_tensor.rows = 2;  // 2 batches
    input_tensor.cols = 4;  // 4 features
    input_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    input_tensor.indices = {std::make_unique<SIZE_TYPE[]>(4)};
    input_tensor.values = {std::make_unique<VALUE_TYPE[]>(4)};

    SIZE_TYPE input_ptrs_data[] = {0, 2, 4};
    SIZE_TYPE input_indices_data[] = {0, 2, 1, 3};
    VALUE_TYPE input_values_data[] = {1.0, 0.5, 2.0, 1.5};

    std::copy(input_ptrs_data, input_ptrs_data + 3, input_tensor.ptrs[0].get());
    std::copy(input_indices_data, input_indices_data + 4, input_tensor.indices[0].get());
    std::copy(input_values_data, input_values_data + 4, input_tensor.values[0].get());

    // Output Gradient Tensor (CSR)
    CSRInput<SIZE_TYPE, VALUE_TYPE> output_gradient_tensor;
    output_gradient_tensor.rows = 2;  // 2 batches
    output_gradient_tensor.cols = 3;  // 3 features
    output_gradient_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    output_gradient_tensor.indices = {std::make_unique<SIZE_TYPE[]>(4)};
    output_gradient_tensor.values = {std::make_unique<VALUE_TYPE[]>(4)};

    SIZE_TYPE output_ptrs_data[] = {0, 2, 4};
    SIZE_TYPE output_indices_data[] = {0, 1, 1, 2};
    VALUE_TYPE output_values_data[] = {0.5, 1.0, 1.5, 0.4};

    std::copy(output_ptrs_data, output_ptrs_data + 3, output_gradient_tensor.ptrs[0].get());
    std::copy(output_indices_data, output_indices_data + 4, output_gradient_tensor.indices[0].get());
    std::copy(output_values_data, output_values_data + 4, output_gradient_tensor.values[0].get());

    // Expected Results for CSC
    SIZE_TYPE expected_ptrs[] = {0, 2, 4, 6, 8};  // Pointers for 3 columns
    SIZE_TYPE expected_indices[] = {0, 1, 1, 2, 0, 1, 1, 2};  // Rows (output gradient indices)
    VALUE_TYPE expected_values[] = {0.5, 1.0, 3.0, 0.8, 0.25, 0.5, 2.25, 0.6};  // Weight values (merged duplicates)

    // Generate CSC
    auto csc_result = generate_new_weights_csc(input_tensor, output_gradient_tensor);

    // Verify Results
    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(csc_result.ptrs[0].get(), csc_result.ptrs[0].get() + 4),
        std::vector<SIZE_TYPE>(expected_ptrs, expected_ptrs + 4)
    );
    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(csc_result.indices[0].get(), csc_result.indices[0].get() + 8),
        std::vector<SIZE_TYPE>(expected_indices, expected_indices + 8)
    );
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(csc_result.values[0].get(), csc_result.values[0].get() + 8),
        std::vector<VALUE_TYPE>(expected_values, expected_values + 8)
    );

    // Edge Case: Empty Input and Output Gradient Tensors
    CSRInput<SIZE_TYPE, VALUE_TYPE> empty_tensor;
    empty_tensor.rows = 0;
    empty_tensor.cols = 0;
    empty_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(1)};
    empty_tensor.indices = {nullptr};
    empty_tensor.values = {nullptr};

    SIZE_TYPE empty_ptrs[] = {0};
    std::copy(empty_ptrs, empty_ptrs + 1, empty_tensor.ptrs[0].get());

    auto csc_empty = generate_new_weights_csc(empty_tensor, empty_tensor);

    // nothing to verity, we passed if it didn't crash here
}

*/
/* #endregion */

/* #region sparse_linear_vectorized_backward_is */

TEST_CASE("Sparse Linear Vectorized Backward IS", "[sparse_linear_vectorized_backward_is]") {
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    // Input Tensor (CSR)
    CSRInput<SIZE_TYPE, VALUE_TYPE> input_tensor;
    input_tensor.rows = 2;  // 2 batches
    input_tensor.cols = 4;  // 4 input features
    input_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    input_tensor.indices = {std::make_unique<SIZE_TYPE[]>(4)};
    input_tensor.values = {std::make_unique<VALUE_TYPE[]>(4)};

    SIZE_TYPE input_ptrs_data[] = {0, 2, 4};
    SIZE_TYPE input_indices_data[] = {0, 2, 1, 3};
    VALUE_TYPE input_values_data[] = {1.0, 0.5, 2.0, 1.5};

    std::copy(input_ptrs_data, input_ptrs_data + 3, input_tensor.ptrs[0].get());
    std::copy(input_indices_data, input_indices_data + 4, input_tensor.indices[0].get());
    std::copy(input_values_data, input_values_data + 4, input_tensor.values[0].get());

    // Output Gradient Tensor (CSR) for Synaptogenesis
    CSRInput<SIZE_TYPE, VALUE_TYPE> out_grad_synaptogenesis;
    out_grad_synaptogenesis.rows = 2;
    out_grad_synaptogenesis.cols = 3;
    out_grad_synaptogenesis.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    out_grad_synaptogenesis.indices = {std::make_unique<SIZE_TYPE[]>(3)};
    out_grad_synaptogenesis.values = {std::make_unique<VALUE_TYPE[]>(3)};

    SIZE_TYPE out_grad_syn_ptrs[] = {0, 1, 3};
    SIZE_TYPE out_grad_syn_indices[] = {0, 1, 2};
    VALUE_TYPE out_grad_syn_values[] = {0.7, 0.6, 0.9};

    std::copy(out_grad_syn_ptrs, out_grad_syn_ptrs + 3, out_grad_synaptogenesis.ptrs[0].get());
    std::copy(out_grad_syn_indices, out_grad_syn_indices + 3, out_grad_synaptogenesis.indices[0].get());
    std::copy(out_grad_syn_values, out_grad_syn_values + 3, out_grad_synaptogenesis.values[0].get());

    // Output Gradient Tensor (CSR) for Backpropagation
    CSRInput<SIZE_TYPE, VALUE_TYPE> out_grad_sparse;
    out_grad_sparse.rows = 2;
    out_grad_sparse.cols = 3;
    out_grad_sparse.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    out_grad_sparse.indices = {std::make_unique<SIZE_TYPE[]>(3)};
    out_grad_sparse.values = {std::make_unique<VALUE_TYPE[]>(3)};

    SIZE_TYPE out_grad_sparse_ptrs[] = {0, 2, 3};
    SIZE_TYPE out_grad_sparse_indices[] = {0, 2, 1};
    VALUE_TYPE out_grad_sparse_values[] = {1.0, 0.5, 0.8};

    std::copy(out_grad_sparse_ptrs, out_grad_sparse_ptrs + 3, out_grad_sparse.ptrs[0].get());
    std::copy(out_grad_sparse_indices, out_grad_sparse_indices + 3, out_grad_sparse.indices[0].get());
    std::copy(out_grad_sparse_values, out_grad_sparse_values + 3, out_grad_sparse.values[0].get());

    // SparseLinearWeights
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(5)};
    weights.connections.indices = {std::make_unique<SIZE_TYPE[]>(8)};
    weights.connections.values = {std::make_unique<VALUE_TYPE[]>(8), std::make_unique<VALUE_TYPE[]>(8)};
    weights.connections.rows = 4; // 4 inputs
    weights.connections.cols = 3; // 3 outputs

    SIZE_TYPE weights_ptrs_data[] = {0, 2, 4, 6, 8};
    SIZE_TYPE weights_indices_data[] = {0, 1, 1, 2, 0, 2, 1, 2};
    VALUE_TYPE weights_values_data[] = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.4, 0.5};
    VALUE_TYPE weights_props_data[] = {0, 0, 0, 0, 0, 0, 0, 0};

    std::copy(weights_ptrs_data, weights_ptrs_data + 5, weights.connections.ptrs[0].get());
    std::copy(weights_indices_data, weights_indices_data + 8, weights.connections.indices[0].get());
    std::copy(weights_values_data, weights_values_data + 8, weights.connections.values[0].get());
    std::copy(weights_props_data, weights_props_data + 8, weights.connections.values[1].get());

    // Dense Output Gradients
    std::vector<VALUE_TYPE> output_gradients = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0};  // 3 outputs * 2 batches

    // Input Gradients
    std::vector<VALUE_TYPE> input_gradients(4, 0);  // 4 inputs initialized to 0

    // Expected Input Gradients
    std::vector<VALUE_TYPE> expected_input_gradients = {0.62, 0.7, 1.1, 0.57};

    // Perform Backpropagation
    sparse_linear_vectorized_backward_is(input_tensor, weights, out_grad_synaptogenesis, out_grad_sparse,
                                         input_gradients.data(), output_gradients.data(), 4);

    // Validate Results
    CHECK_VECTOR_ALMOST_EQUAL(input_gradients, expected_input_gradients);

    // Expected Weights Gradients
    std::vector<VALUE_TYPE> expected_weights_gradients = {0.5, 0.7, 1.6, 2.0, 0.25, 0.45, 1.2, 1.5};

    // Validate Weight Gradients
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[1].get(),
                                weights.connections.values[1].get() + 8),
        expected_weights_gradients);

        // Compute Expected Probes from Outer Product
    std::vector<SIZE_TYPE> expected_rows{0, 1, 1, 2, 3, 3};
    std::vector<SIZE_TYPE> expected_cols{0, 1, 2, 0, 1, 2};
    std::vector<VALUE_TYPE> expected_values{0.7, 1.2, 1.8, 0.35, 0.9, 1.35};

    // Validate Probes
    REQUIRE(weights.probes.nnz() == static_cast<SIZE_TYPE>(expected_rows.size()));

    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(weights.probes.indices[0].get(),
                               weights.probes.indices[0].get() + weights.probes.nnz()),
        expected_rows);

    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(weights.probes.indices[1].get(),
                               weights.probes.indices[1].get() + weights.probes.nnz()),
        expected_cols);

    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.probes.values[0].get(),
                                weights.probes.values[0].get() + weights.probes.nnz()),
        expected_values);

    //-------------one hot modification tests--------------

        // One-Hot Test: Modify a single weight and validate results
    weights.connections.values[0][3] = 0.9;  // Modify one weight
    std::fill(input_gradients.begin(), input_gradients.end(), 0);  // Reset input gradients
    std::fill(weights.connections.values[1].get(), weights.connections.values[1].get() + 8, 0);  // Reset weight gradients

    // Expected Gradients for Modified Weight
    std::vector<VALUE_TYPE> expected_input_gradients_one_hot = {0.62, 0.85, 1.1, 0.57};
    std::vector<VALUE_TYPE> expected_weights_gradients_one_hot = {0.5, 0.7, 1.6, 2.0, 0.25, 0.45, 1.2, 1.5}; //stays the same because w_grad is in*o_grad

    sparse_linear_vectorized_backward_is(input_tensor, weights, out_grad_synaptogenesis, out_grad_sparse,
                                         input_gradients.data(), output_gradients.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(input_gradients, expected_input_gradients_one_hot);
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[1].get(),
                                weights.connections.values[1].get() + 8),
        expected_weights_gradients_one_hot);

    // One-Hot Test: Modify a single input value
    input_tensor.values[0][2] = 1.2;  // Modify one input value (input index 2 is 1, so this is input 1)
    std::fill(input_gradients.begin(), input_gradients.end(), 0);  // Reset input gradients
    std::fill(weights.connections.values[1].get(), weights.connections.values[1].get() + 8, 0);  // Reset weight gradients

    // Expected Gradients for Modified Input
    std::vector<VALUE_TYPE> expected_input_gradients_one_hot_input = {0.62, 0.85, 1.1, 0.57}; // stays same because i_grad=w_val*o_grad
    std::vector<VALUE_TYPE> expected_weights_gradients_one_hot_input = {0.5, 0.7, 0.96, 1.2, 0.25, 0.45, 1.2, 1.5}; //2-3 are on ptr 1, so they change

    sparse_linear_vectorized_backward_is(input_tensor, weights, out_grad_synaptogenesis, out_grad_sparse,
                                         input_gradients.data(), output_gradients.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(input_gradients, expected_input_gradients_one_hot_input);
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[1].get(),
                                weights.connections.values[1].get() + 8),
        expected_weights_gradients_one_hot_input);

    // One-Hot Test: Modify a single output gradient
    out_grad_sparse.values[0][0] = 1.5;  // Modify one sparse gradient value
    output_gradients[0] = 1.5; // modify it correctly for weights too
    std::fill(input_gradients.begin(), input_gradients.end(), 0);  // Reset input gradients
    std::fill(weights.connections.values[1].get(), weights.connections.values[1].get() + 8, 0);  // Reset weight gradients

    // Expected Gradients for Modified Sparse Gradient
    std::vector<VALUE_TYPE> expected_input_gradients_one_hot_sparse_grad = {0.77, 0.85, 1.45, 0.57};  // only 0 and 2 are modified (1 is changed from the weight) because out 0 connects to in 0 and 2
    std::vector<VALUE_TYPE> expected_weights_gradients_one_hot_sparse_grad = {1.5, 0.7, 0.96, 1.2, 0.75, 0.45, 1.2, 1.5};

    sparse_linear_vectorized_backward_is(input_tensor, weights, out_grad_synaptogenesis, out_grad_sparse,
                                         input_gradients.data(), output_gradients.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(input_gradients, expected_input_gradients_one_hot_sparse_grad);
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[1].get(),
                                weights.connections.values[1].get() + 8),
        expected_weights_gradients_one_hot_sparse_grad);
}


/* #endregion */