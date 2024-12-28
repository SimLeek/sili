#include "../sili/lib/headers/linear_sisldo.hpp"
#include "tests_main.h"
#include <vector>

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
