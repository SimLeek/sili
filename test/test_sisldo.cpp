#include "../sili/lib/headers/linear_sisldo.hpp"
#include "csr.hpp"
#include "sparse_struct.hpp"
#include "tests_main.h"
#include <cstddef>
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

TEST_CASE("Optimize Weights", "[optim_weights]") {
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    // SparseLinearWeights initialization
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(5)};
    weights.connections.indices = {std::make_unique<SIZE_TYPE[]>(8)};
    weights.connections.values = {std::make_unique<VALUE_TYPE[]>(8), std::make_unique<VALUE_TYPE[]>(8), std::make_unique<VALUE_TYPE[]>(8)};
    weights.connections.rows = 4;
    weights.connections.cols = 3;

    SIZE_TYPE weights_ptrs_data[] = {0, 2, 4, 6, 8};
    SIZE_TYPE weights_indices_data[] = {0, 1, 1, 2, 0, 2, 1, 2};
    VALUE_TYPE weights_values_data[] = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.4, 0.5};
    VALUE_TYPE gradients_data[] = {0.1, 0.2, -0.1, -0.2, 0.05, -0.05, 0.0, -0.1};
    VALUE_TYPE connection_strength_data[] = {0.5, 0.2, 0.4, 0.6, 0.3, 0.2, 0.1, 0.4};

    std::copy(weights_ptrs_data, weights_ptrs_data + 5, weights.connections.ptrs[0].get());
    std::copy(weights_indices_data, weights_indices_data + 8, weights.connections.indices[0].get());
    std::copy(weights_values_data, weights_values_data + 8, weights.connections.values[0].get());
    std::copy(gradients_data, gradients_data + 8, weights.connections.values[1].get());
    std::copy(connection_strength_data, connection_strength_data + 8, weights.connections.values[2].get());

    VALUE_TYPE learning_rate = 0.01;
    int num_cpus = 2;

    // Expected updated weights
    std::vector<VALUE_TYPE> expected_weights = {
        0.299333, 0.398333, 0.500714, 0.60125, 0.699615, 0.800417, 0.4, 0.500714
    };

    optim_weights(weights, learning_rate, num_cpus);

    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[0].get(),
                                weights.connections.values[0].get() + 8),
        expected_weights,
        1e-6);
}

TEST_CASE("Optimize Synaptogenesis", "[optim_synaptogenesis]") {
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    // SparseLinearWeights initialization
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.probes.ptrs = 3;
    weights.probes.indices = {std::make_unique<SIZE_TYPE[]>(3), std::make_unique<SIZE_TYPE[]>(3)};
    weights.probes.values = {std::make_unique<VALUE_TYPE[]>(3)};
    weights.probes.rows = 4;
    weights.probes.cols = 4;

    weights.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(5)};
    weights.connections.indices = {std::make_unique<SIZE_TYPE[]>(8)};
    weights.connections.values = {std::make_unique<VALUE_TYPE[]>(8), std::make_unique<VALUE_TYPE[]>(8), std::make_unique<VALUE_TYPE[]>(8)};
    weights.connections.rows = 4;
    weights.connections.cols = 4;


    SIZE_TYPE probes_indices0_data[] = {0, 1, 3};
    SIZE_TYPE probes_indices1_data[] = {2, 0, 2};
    VALUE_TYPE probes_values_data[] = {0.001, 0.002, 0.009};

    SIZE_TYPE weights_ptrs_data[] = {0, 2, 4, 6, 8};
    SIZE_TYPE weights_indices_data[] = {0, 1, 1, 2, 0, 2, 1, 2};
    VALUE_TYPE weights_values_data[] = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.4, 0.5};
    VALUE_TYPE weights_backprop_data[] = {0, 0, 0, 0, 0, 0, 0, 0};
    VALUE_TYPE weights_importance_data[] = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.4, 0.5};


    std::copy(probes_indices0_data, probes_indices0_data + 3, weights.probes.indices[0].get());
    std::copy(probes_indices1_data, probes_indices1_data + 3, weights.probes.indices[1].get());
    std::copy(probes_values_data, probes_values_data + 3, weights.probes.values[0].get());

    std::copy(weights_ptrs_data, weights_ptrs_data + 5, weights.connections.ptrs[0].get());
    std::copy(weights_indices_data, weights_indices_data + 8, weights.connections.indices[0].get());
    std::copy(weights_values_data, weights_values_data + 8, weights.connections.values[0].get());
    std::copy(weights_backprop_data, weights_backprop_data + 8, weights.connections.values[1].get());
    std::copy(weights_importance_data, weights_importance_data + 8, weights.connections.values[2].get());

    VALUE_TYPE learning_rate = 0.01;
    SIZE_TYPE max_weights = 10;
    int num_cpus = 2;
    VALUE_TYPE beta = 0.1;

    optim_synaptogenesis(weights, learning_rate, max_weights, num_cpus, beta);

    // Expected merged weights (assume a specific result after the operations)
    std::vector<SIZE_TYPE> expected_weights_ptrs_data = {0, 3, 6, 8, 10};
    std::vector<SIZE_TYPE> expected_weights_indices_data = {0, 1, 2, 0, 1, 2, 0, 2, 1, 2};
    std::vector<VALUE_TYPE> expected_weights = { 0.3, 0.4, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.4, 0.5};
    std::vector<VALUE_TYPE> expected_backprop = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // 0.5 at the end becomes 0.41 here because it had 0.5 importance subtracted 0.09 from it
    std::vector<VALUE_TYPE> expected_importance = { 0.3, 0.4, -0.01, -0.02, 0.5, 0.6, 0.7, 0.8, 0.4, 0.41};

    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(weights.connections.ptrs[0].get(),
                                weights.connections.ptrs[0].get() + 5),
        expected_weights_ptrs_data);

    CHECK_VECTOR_EQUAL(
        std::vector<SIZE_TYPE>(weights.connections.indices[0].get(),
                                weights.connections.indices[0].get() + 10),
        expected_weights_indices_data);

    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[0].get(),
                                weights.connections.values[0].get() + 10),
        expected_weights);

    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[1].get(),
                                weights.connections.values[1].get() + 10),
        expected_backprop);

        CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[2].get(),
                                weights.connections.values[2].get() + 10),
        expected_importance);
}

/*TEST_CASE("train loop from zero", "[integration_train_loop]"){
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    // Input Tensor. Starts dense, always.
    // since we're starting with 0, potentially from a previous array, we need a bias to provide indices instead
    VALUE_TYPE input_values_data[] = {0, 0, 0, 0, 0, 0, 0, 0};

    // this is where the starmap comes into play. It lets us tell neurons "you should have activated"
    // it also slowly iterates which neurons should be trained or not using a sin function, which gives networks a sense of time
    // it's meant for 2d biasing more, but idc enough to make it 1d since I plan on using this for real time learning. Open a pr if you want something else.
    CSRInput<SIZE_TYPE, VALUE_TYPE> layer_input_train_bias_tensor;
    layer_input_train_bias_tensor.rows = 2;  // 2 batches
    layer_input_train_bias_tensor.cols = 4;  // 4 features
    layer_input_train_bias_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    SIZE_TYPE layer_input_train_bias_ptrs_data[] = {0, 0, 0};
    std::copy(layer_input_train_bias_ptrs_data, layer_input_train_bias_ptrs_data + 3, layer_input_train_bias_tensor.ptrs[0].get());

    //todo: fix the random seed for the mt19937_64 generator and then test the starmap's csr
    auto input_starmap = CSRStarmap(layer_input_train_bias_tensor);
    input_starmap.iterate(4);

    // Weights. Start with all zeros and add skip connections for a real test.
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections.rows = 4;  // 4 input features
    weights.connections.cols = 3;  // 3 output features
    weights.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(4)};
    weights.connections.indices = {nullptr};
    weights.connections.values = {
        nullptr, nullptr, nullptr
    };

    auto input_portion = top_k_csr_biased(input_values_data, input_starmap.csrMatrix, 2, 4, 4, 4);

    //todo: add input_portion assertions here

    SIZE_TYPE weight_ptrs_data[] = {0, 0, 0, 0};

    std::copy(weight_ptrs_data, weight_ptrs_data + 4, weights.connections.ptrs[0].get());

    // Output
    VALUE_TYPE output[6] = {0.0, 0.0, 0.0, 0, 0, 0};
    VALUE_TYPE expected_output[] = {0.0, 0, 0, 0, 0, 0};

    // Train flag
    bool train = true;
    VALUE_TYPE solidify = 0.01;

    // Run forward pass
    sparse_linear_csr_csc_forward(input_portion, weights, output, train, solidify);
    // skip connections (does nothing since input was also zero)
    // iterate min of input or output (3)
    // todo: make a simple add function, or use numpy or something
    for(int i=0;i<3;i++)
        output[i] +=input_values_data[i];
    for(int i=0;i<3;i++)
        output[i+3] +=input_values_data[i+4];

    // Verify output
    CHECK_VECTOR_ALMOST_EQUAL(std::vector<VALUE_TYPE>(output, output + 6), std::vector<VALUE_TYPE>(expected_output, expected_output + 6));

    // Verify importances updated
    VALUE_TYPE expected_importances[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK_VECTOR_ALMOST_EQUAL(
        std::vector<VALUE_TYPE>(weights.connections.values[2].get(), weights.connections.values[2].get() + 6),
        std::vector<VALUE_TYPE>(expected_importances, expected_importances + 6)
    );

    //VALUE_TYPE desired_output[] = {1, 2, 3, 4, 5, 6};
    //outputted 0s, so we backprop mean squared error, which is the jacobian of the MSE, or derivitive with respect to each prediction element
    //(1/n)*sum_from_y=0 to y=n:((y_true-y_pred)^2) -> f_grad(y_pred):-2(y_true-y_pred)/n
    VALUE_TYPE mse_output_grad[] = {-2./6, -4./6, -6./6, -8./6, -10./6, -12./6};

    VALUE_TYPE in_grad[] = {0, 0, 0, 0, 0, 0, 0, 0};

    //todo: use torch or something here. Test include only
    for(int i=0;i<3;i++)
        in_grad[i] +=mse_output_grad[i];
    for(int i=0;i<3;i++)
        in_grad[i+4] +=mse_output_grad[i+3];

    CSRInput<SIZE_TYPE, VALUE_TYPE> layer_output_train_bias_tensor;
    layer_output_train_bias_tensor.rows = 2;  // 2 batches
    layer_output_train_bias_tensor.cols = 3;  // 4 features
    layer_output_train_bias_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    SIZE_TYPE layer_output_train_bias_ptrs_data[] = {0, 0, 0};
    std::copy(layer_input_train_bias_ptrs_data, layer_input_train_bias_ptrs_data + 3, layer_output_train_bias_tensor.ptrs[0].get());

    //todo: fix the random seed for the mt19937_64 generator and then test the starmap's csr
    auto output_starmap = CSRStarmap(layer_output_train_bias_tensor);
    output_starmap.iterate(3);

    auto output_grad_portion = top_k_csr_biased(mse_output_grad, output_starmap.csrMatrix, 2, 3, 3, 4);


    sparse_linear_vectorized_backward_is(input_portion, weights, output_grad_portion, output_grad_portion,
                                         in_grad, mse_output_grad, 4);

    //todo: add assert vector (almost) equals here for weights.connections.values[1], weights.probes (should have some 0 importance weights), and in_grad
    float learning_rate = 0.01;

    optim_weights(weights, learning_rate, 4);

    //todo: assert no change to weights because they don't exist yet

    optim_synaptogenesis(weights, learning_rate, 6, 4);

    //todo: assert that we have additional weights now
}*/

template <class SIZE_TYPE, class VALUE_TYPE>
void CHECK_CSR_WEIGHTS(const sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, TriValues<VALUE_TYPE>> &weights,
                       const std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> &expected_weights) {
    // Extract actual weights from the sparse_struct
    std::vector<SIZE_TYPE> actual_ptrs;
    std::vector<SIZE_TYPE> actual_cols;
    std::vector<VALUE_TYPE> actual_values1;
    std::vector<VALUE_TYPE> actual_values2;
    std::vector<VALUE_TYPE> actual_values3;

    SIZE_TYPE rows = weights.rows;
    auto &ptrs = weights.ptrs[0];
    auto &indices = weights.indices[0];
    auto &values1 = weights.values[0];
    auto &values2 = weights.values[1];
    auto &values3 = weights.values[2];

    for (SIZE_TYPE i = 0; i < rows; ++i) {
        actual_ptrs.push_back(ptrs[i]);
        for (SIZE_TYPE j = ptrs[i]; j < ptrs[i + 1]; ++j) {
            actual_cols.push_back(indices[j]);
            actual_values1.push_back(values1[j]);
            actual_values2.push_back(values2[j]);
            actual_values3.push_back(values3[j]);
        }
    }
    actual_ptrs.push_back(ptrs[rows]);

    // Decompose expected weights into vectors
    std::vector<SIZE_TYPE> expected_ptrs;
    std::vector<SIZE_TYPE> expected_cols;
    std::vector<VALUE_TYPE> expected_values1;
    std::vector<VALUE_TYPE> expected_values2;
    std::vector<VALUE_TYPE> expected_values3;

    for (const auto &t : std::get<0>(expected_weights)) {
        expected_ptrs.push_back(t);
    }
    for (const auto &t : std::get<1>(expected_weights)) {
        expected_cols.push_back(t);
    }
    for (const auto &t : std::get<2>(expected_weights)) {
        expected_values1.push_back(t);
    }
    for (const auto &t : std::get<3>(expected_weights)) {
        expected_values2.push_back(t);
    }
    for (const auto &t : std::get<4>(expected_weights)) {
        expected_values3.push_back(t);
    }

    // Compare using CHECK_VECTOR_EQUAL and CHECK_VECTOR_ALMOST_EQUAL
    CHECK_VECTOR_EQUAL(actual_ptrs, expected_ptrs);
    CHECK_VECTOR_EQUAL(actual_cols, expected_cols);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values1, expected_values1);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values2, expected_values2);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values3, expected_values3);
}

template <class SIZE_TYPE, class VALUE_TYPE>
void CHECK_CSR_VALUES(const sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>> &weights,
                       const std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>> &expected_weights) {
    // Extract actual weights from the sparse_struct
    std::vector<SIZE_TYPE> actual_ptrs;
    std::vector<SIZE_TYPE> actual_cols;
    std::vector<VALUE_TYPE> actual_values1;

    SIZE_TYPE rows = weights.rows;
    auto &ptrs = weights.ptrs[0];
    auto &indices = weights.indices[0];
    auto &values1 = weights.values[0];

    for (SIZE_TYPE i = 0; i < rows; ++i) {
        actual_ptrs.push_back(ptrs[i]);
        for (SIZE_TYPE j = ptrs[i]; j < ptrs[i + 1]; ++j) {
            actual_cols.push_back(indices[j]);
            actual_values1.push_back(values1[j]);
        }
    }
    actual_ptrs.push_back(ptrs[rows]);

    // Decompose expected weights into vectors
    std::vector<SIZE_TYPE> expected_ptrs;
    std::vector<SIZE_TYPE> expected_cols;
    std::vector<VALUE_TYPE> expected_values1;

    for (const auto &t : std::get<0>(expected_weights)) {
        expected_ptrs.push_back(t);
    }
    for (const auto &t : std::get<1>(expected_weights)) {
        expected_cols.push_back(t);
    }
    for (const auto &t : std::get<2>(expected_weights)) {
        expected_values1.push_back(t);
    }

    // Compare using CHECK_VECTOR_EQUAL and CHECK_VECTOR_ALMOST_EQUAL
    CHECK_VECTOR_EQUAL(actual_ptrs, expected_ptrs);
    CHECK_VECTOR_EQUAL(actual_cols, expected_cols);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values1, expected_values1);
}

TEST_CASE("train loop from zero", "[integration_train_loop]") {
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    constexpr int num_iterations = 3;

    // Expected CSRs after input and output starmap iterations
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>>> expected_input_starmap_csrs = {
        {{0, 3, 3}, {1, 2, 3}, {1.55606e-06, 6.58108e-05, 8.61137e-05}}, // After first iteration
        {{0, 1}, {0}, {1.0}}, // After second iteration
        {{0, 1}, {0}, {1.0}}  // After third iteration
    };

    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>>> expected_output_starmap_csrs = {
        {{0, 1, 1}, {1}, {4.68585e-05}}, // After first iteration
        {{0, 1}, {0}, {1.0}}, // After second iteration
        {{0, 1}, {0}, {1.0}}  // After third iteration
    };

    // Input data for three iterations
    std::vector<std::array<VALUE_TYPE, 8>> input_values_data({
        {0, 0, 0, 0, 0, 0, 0, 0},
        {.1, .1,.1 ,.1 ,.1 ,.1, .1, .1},
        {.2, .2, .2, .2, .2, .2, .2, .2}
    });

    // Expected outputs per iteration
    std::vector<std::vector<VALUE_TYPE>> expected_outputs = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };



    // Expected weights after synaptogenesis for each iteration
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>>> expected_weights_after_synaptogenesis = {
        {{0}, {0}, {0}, {0}, {0}}, // Empty CSR
        {{0}, {0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}, {0}}
    };

    // Set up starmaps for input and output before the loop
    CSRInput<SIZE_TYPE, VALUE_TYPE> layer_input_train_bias_tensor;
    layer_input_train_bias_tensor.rows = 2;
    layer_input_train_bias_tensor.cols = 4;
    layer_input_train_bias_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    SIZE_TYPE layer_input_train_bias_ptrs_data[] = {0, 0, 0};
    std::copy(layer_input_train_bias_ptrs_data, layer_input_train_bias_ptrs_data + 3, layer_input_train_bias_tensor.ptrs[0].get());

    auto input_starmap = CSRStarmap(layer_input_train_bias_tensor, 42);

    CSRInput<SIZE_TYPE, VALUE_TYPE> layer_output_train_bias_tensor;
    layer_output_train_bias_tensor.rows = 2;
    layer_output_train_bias_tensor.cols = 3;
    layer_output_train_bias_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    SIZE_TYPE layer_output_train_bias_ptrs_data[] = {0, 0, 0};
    std::copy(layer_output_train_bias_ptrs_data, layer_output_train_bias_ptrs_data + 3, layer_output_train_bias_tensor.ptrs[0].get());

    auto output_starmap = CSRStarmap(layer_output_train_bias_tensor, 42);

    // Weights setup before the loop
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections.rows = 4;
    weights.connections.cols = 3;
    weights.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(4)};
    weights.connections.indices = {nullptr};
    weights.connections.values = {nullptr, nullptr, nullptr};
    SIZE_TYPE weight_ptrs_data[] = {0, 0, 0, 0};
    std::copy(weight_ptrs_data, weight_ptrs_data + 4, weights.connections.ptrs[0].get());

    // Train flag and learning rate setup
    bool train = true;
    float learning_rate = 0.01;
    float solidify = 0.01;

    // Expected weights after each iteration
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>>> expected_weights = {
        {{0}, {0}, {0}, {0}, {0}}, // Empty CSR
        {{0}, {0}, {0}, {0}, {0}},
        {{0}, {0}, {0}, {0}, {0}}
    };

    for (int iter = 0; iter < num_iterations; ++iter) {
        input_starmap.iterate(4);
        output_starmap.iterate(3);

        // Assert correctness of input and output starmap CSRs
        CHECK_CSR_VALUES(input_starmap.csrMatrix, expected_input_starmap_csrs[iter]);
        CHECK_CSR_VALUES(output_starmap.csrMatrix, expected_output_starmap_csrs[iter]);

        auto input_portion = top_k_csr_biased(input_values_data[iter].data(), input_starmap.csrMatrix, 2, 4, 4, 4);

        // Assert correctness of input_portion CSR
        CHECK(input_portion.ptrs.size() == 5); // Number of rows + 1
        CHECK(input_portion.indices.size() <= 4); // At most 4 non-zero entries
        CHECK(input_portion.values.size() == input_portion.indices.size());

        VALUE_TYPE output[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        sparse_linear_csr_csc_forward(input_portion, weights, output, train, solidify);

        // Skip connections
        for (int i = 0; i < 3; i++) {
            output[i] += input_values_data[iter][i];
            output[i + 3] += input_values_data[iter][i + 4];
        }

        CHECK_VECTOR_ALMOST_EQUAL(std::vector<VALUE_TYPE>(output, output + 6), expected_outputs[iter]);

        // Compute MSE jacobian
        VALUE_TYPE desired_output[] = {1, 2, 3, 4, 5, 6};
        VALUE_TYPE mse_output_grad[6];
        for (int i = 0; i < 6; ++i) {
            mse_output_grad[i] = -2.0 * (desired_output[i] - output[i]) / 6.0;
        }

        // Compute skip connection grad
        VALUE_TYPE in_grad[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < 3; i++) {
            in_grad[i] += mse_output_grad[i];
            in_grad[i + 4] += mse_output_grad[i + 3];
        }

        auto output_grad_portion = top_k_csr_biased(mse_output_grad, output_starmap.csrMatrix, 2, 3, 3, 4);

        CHECK_CSR_VALUES(output_grad_portion, expected_output_starmap_csrs[iter]);

        sparse_linear_vectorized_backward_is(input_portion, weights, output_grad_portion, output_grad_portion,
                                             in_grad, mse_output_grad, 4);

        // Assert in_grad correctness
        CHECK_VECTOR_ALMOST_EQUAL(std::vector<VALUE_TYPE>(in_grad, in_grad + 8),
                                  std::vector<VALUE_TYPE>(expected_outputs[iter]));

        optim_weights(weights, learning_rate, 4);

        // Verify weights after optimization
        CHECK_CSR_WEIGHTS(weights.connections, expected_weights[iter]);

        optim_synaptogenesis(weights, learning_rate, 6, 4);

        // Verify weights after synaptogenesis
        CHECK_CSR_WEIGHTS(weights.connections, expected_weights_after_synaptogenesis[iter]);
    }
}
