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


/* #endregion */