#include "../sili/lib/headers/coo.hpp"
#include "tests_main.h"
#include <random>
#include <vector>


/* #region Inplace Merge COO */
TEST_CASE("Inplace Merge COO - Simple Merge without Duplicates", "[inplace_merge_coo]") {
    // Input data
    COOIndices<int> indices = {
        std::make_unique<int[]>(8),
        std::make_unique<int[]>(8)
    };
    UnaryValues<float> values = {
        std::make_unique<float[]>(8)
    };

    int col_data[] = {1, 3, 5, 7, 2, 4, 6, 8};
    int row_data[] = {1, 1, 2, 2, 1, 1, 2, 2};
    float val_data[] = {1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0};

    std::copy(col_data, col_data + 8, indices[0].get());
    std::copy(row_data, row_data + 8, indices[1].get());
    std::copy(val_data, val_data + 8, values[0].get());

    int left = 0;
    int mid = 3;
    int right = 7;

    // Expected output after merge
    std::vector<int> expected_cols = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> expected_rows = {1, 1, 1, 1, 2, 2, 2, 2};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    size_t duplicates = inplace_merge_coo(indices, values, left, mid, right);

    REQUIRE_MESSAGE(duplicates == 0, "No duplicates should be found");


    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 8), expected_cols);
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 8), expected_rows);
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 8), expected_vals);
}

TEST_CASE("Inplace Merge COO - Merge with Duplicates", "[inplace_merge_coo]") {
    // Input data
    COOIndices<int> indices = {
        std::make_unique<int[]>(8),
        std::make_unique<int[]>(8)
    };
    BiValues<float> values = {
        std::make_unique<float[]>(8),
        std::make_unique<float[]>(8)
    };

    int col_data[] = {1, 3, 5, 7, 1, 3, 5, 7};
    int row_data[] = {1, 1, 1, 1, 1, 1, 1, 1};
    float val_data1[] = {1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0};
    float val_data2[] = {2.0, 6.0, 10.0, 14.0, 2.0, 6.0, 10.0, 14.0};

    std::copy(col_data, col_data + 8, indices[0].get());
    std::copy(row_data, row_data + 8, indices[1].get());
    std::copy(val_data1, val_data1 + 8, values[0].get());
    std::copy(val_data2, val_data2 + 8, values[1].get());

    int left = 0;
    int mid = 3;
    int right = 7;

    // Expected output
    std::vector<int> expected_cols = {1, 3, 5, 7};
    std::vector<int> expected_rows = {1, 1, 1, 1};
    std::vector<float> expected_vals1 = {2.0, 6.0, 10.0, 14.0};
    std::vector<float> expected_vals2 = {4.0, 12.0, 20.0, 28.0};

    size_t duplicates = inplace_merge_coo(indices, values, left, mid, right);

    REQUIRE_MESSAGE(duplicates == 4, "There should be 4 duplicates");

    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 4), expected_cols);
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 4), expected_rows);
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 4), expected_vals1);
    CHECK_VECTOR_EQUAL(vec(values[1].get(), 4), expected_vals2);
}

TEST_CASE("Inplace Merge COO - Mixed Duplicates and Non-Duplicates", "[inplace_merge_coo]") {
    // Input data
    COOIndices<int> indices = {
        std::make_unique<int[]>(8),
        std::make_unique<int[]>(8)
    };
    TriValues<float> values = {
        std::make_unique<float[]>(8),
        std::make_unique<float[]>(8),
        std::make_unique<float[]>(8)
    };

    int row_data[] = {1, 1, 1, 1, 1, 1, 2, 2};
    int col_data[] = {1, 2, 3, 7, 1, 3, 5, 8};
    float val_data1[] = {1.0, 2.0, 5.0, 7.0, 1.0, 3.0, 5.0, 8.0};
    float val_data2[] = {10.0, 20.0, 30.0, 70.0, 10.0, 30.0, 50.0, 80.0};
    float val_data3[] = {100.0, 200.0, 300.0, 700.0, 100.0, 300.0, 500.0, 800.0};

    std::copy(row_data, row_data + 8, indices[0].get());
    std::copy(col_data, col_data + 8, indices[1].get());
    std::copy(val_data1, val_data1 + 8, values[0].get());
    std::copy(val_data2, val_data2 + 8, values[1].get());
    std::copy(val_data3, val_data3 + 8, values[2].get());

    int left = 0;
    int mid = 3;
    int right = 7;

    // Expected output
    std::vector<int> expected_rows = {1, 1, 1, 1, 2, 2};
    std::vector<int> expected_cols = {1, 2, 3, 7, 5, 8};
    std::vector<float> expected_vals1 = {2.0, 2.0, 8.0, 7.0, 5.0, 8.0};
    std::vector<float> expected_vals2 = {20.0, 20.0, 60.0, 70.0, 50.0, 80.0};
    std::vector<float> expected_vals3 = {200.0, 200.0, 600.0, 700.0, 500.0, 800.0};

    size_t duplicates = inplace_merge_coo(indices, values, left, mid, right);

    REQUIRE_MESSAGE(duplicates == 2, "There should be 2 duplicates");

    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 6), expected_rows);
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 6), expected_cols);
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 6), expected_vals1);
    CHECK_VECTOR_EQUAL(vec(values[1].get(), 6), expected_vals2);
    CHECK_VECTOR_EQUAL(vec(values[2].get(), 6), expected_vals3);
}

/* #endregion */

/* #region Insertion Sort COO */
TEST_CASE("Insertion Sort COO - Already Sorted Input", "[insertion_sort_coo]") {
    COOIndices<int> indices = {std::make_unique<int[]>(4), std::make_unique<int[]>(4)};
    UnaryValues<float> values = {std::make_unique<float[]>(4)};

    int rows[] = {1, 1, 2, 2};
    int cols[] = {1, 2, 3, 4};
    float vals[] = {1.0, 2.0, 3.0, 4.0};

    std::copy(rows, rows+4, indices[0].get());
    std::copy(cols, cols+4, indices[1].get());
    std::copy(vals, vals+4, values[0].get());

    size_t duplicates = insertion_sort_coo(indices, values, 0, 3);

    REQUIRE(duplicates == 0);

    std::vector<int> expected_rows = {1, 1, 2, 2};
    std::vector<int> expected_cols = {1, 2, 3, 4};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0};

    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 4), expected_rows);
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 4), expected_cols);
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 4), expected_vals);
}

TEST_CASE("Insertion Sort COO - Reverse Order Input", "[insertion_sort_coo]") {
    COOIndices<int> indices = {std::make_unique<int[]>(4), std::make_unique<int[]>(4)};
    UnaryValues<float> values = {std::make_unique<float[]>(4)};

    int rows[] = {2, 2, 1, 1};
    int cols[] = {2, 1, 2, 1};
    float vals[] = {4.0, 3.0, 2.0, 1.0};

    std::memcpy(indices[0].get(), rows, 4 * sizeof(int));
    std::memcpy(indices[1].get(), cols, 4 * sizeof(int));
    std::memcpy(values[0].get(), vals, 4 * sizeof(float));

    size_t duplicates = insertion_sort_coo(indices, values, 0, 3);

    CHECK(duplicates == 0);

    std::vector<int> expected_rows = {1, 1, 2, 2};
    std::vector<int> expected_cols = {1, 2, 1, 2};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0};

    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 4), expected_rows);
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 4), expected_cols);
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 4), expected_vals);
}

TEST_CASE("Insertion Sort COO - Case with Duplicates", "[insertion_sort_coo]") {
    COOIndices<int> indices = {std::make_unique<int[]>(5), std::make_unique<int[]>(5)};
    UnaryValues<float> values = {std::make_unique<float[]>(5)};

    int rows[] = {1, 1, 1, 1, 1};
    int cols[] = {1, 3, 2, 2, 3};
    float vals[] = {1.0, 3.0, 2.0, 2.0, 3.0};

    std::memcpy(indices[0].get(), rows, 5 * sizeof(int));
    std::memcpy(indices[1].get(), cols, 5 * sizeof(int));
    std::memcpy(values[0].get(), vals, 5 * sizeof(float));

    size_t duplicates = insertion_sort_coo(indices, values, 0, 4);

    REQUIRE_MESSAGE(duplicates == 2, "There should be 2 duplicates");

    std::vector<int> expected_rows = {1, 1, 1};
    std::vector<int> expected_cols = {1, 2, 3};
    std::vector<float> expected_vals = {1.0, 4.0, 6.0};

    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 3), expected_rows);
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 3), expected_cols);
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 3), expected_vals);
}

TEST_CASE("Insertion Sort COO - Case with BiValues", "[insertion_sort_coo]") {
    COOIndices<int> indices = {std::make_unique<int[]>(5), std::make_unique<int[]>(5)};
    BiValues<float> values = {std::make_unique<float[]>(5), std::make_unique<float[]>(5)};

    int rows[] = {1, 1, 1, 1, 1};
    int cols[] = {1, 3, 2, 2, 3};
    float vals1[] = {1.0, 3.0, 2.0, 2.0, 3.0};
    float vals2[] = {10.0, 30.0, 20.0, 20.0, 30.0};

    std::memcpy(indices[0].get(), rows, 5 * sizeof(int));
    std::memcpy(indices[1].get(), cols, 5 * sizeof(int));
    std::memcpy(values[0].get(), vals1, 5 * sizeof(float));
    std::memcpy(values[1].get(), vals2, 5 * sizeof(float));

    size_t duplicates = insertion_sort_coo(indices, values, 0, 4);

    REQUIRE_MESSAGE(duplicates == 2, "There should be 2 duplicates");

    std::vector<int> expected_rows = {1, 1, 1};
    std::vector<int> expected_cols = {1, 2, 3};
    std::vector<float> expected_vals1 = {1.0, 4.0, 6.0};
    std::vector<float> expected_vals2 = {10.0, 40.0, 60.0};

    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 3), expected_rows);
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 3), expected_cols);
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 3), expected_vals1);
    CHECK_VECTOR_EQUAL(vec(values[1].get(), 3), expected_vals2);
}
/* #endregion */


TEST_CASE("Merge Sort COO - Simple Case", "[merge_sort_coo]") {
    COOIndices<int> indices = {std::make_unique<int[]>(4), std::make_unique<int[]>(4)};
    BiValues<float> values = {std::make_unique<float[]>(4), std::make_unique<float[]>(4)};

    int col_data[] = {4, 2, 3, 1};
    int row_data[] = {1, 1, 1, 1};
    float val1_data[] = {4.0, 2.0, 3.0, 1.0};
    float val2_data[] = {40.0, 20.0, 30.0, 10.0};

    std::copy(col_data, col_data + 4, indices[0].get());
    std::copy(row_data, row_data + 4, indices[1].get());
    std::copy(val1_data, val1_data + 4, values[0].get());
    std::copy(val2_data, val2_data + 4, values[1].get());

    int expected_cols[] = {1, 2, 3, 4};
    int expected_rows[] = {1, 1, 1, 1};
    float expected_vals1[] = {1.0, 2.0, 3.0, 4.0};
    float expected_vals2[] = {10.0, 20.0, 30.0, 40.0};

    std::size_t duplicates = merge_sort_coo(indices, values, 4);

    CHECK(duplicates == 0);  // No duplicates in this test case
    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 4), std::vector<int>(expected_cols, expected_cols + 4));
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 4), std::vector<int>(expected_rows, expected_rows + 4));
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 4), std::vector<float>(expected_vals1, expected_vals1 + 4));
    CHECK_VECTOR_EQUAL(vec(values[1].get(), 4), std::vector<float>(expected_vals2, expected_vals2 + 4));
}

TEST_CASE("Merge Sort COO - Complex Case with Duplicates", "[merge_sort_coo]") {
    COOIndices<int> indices = {std::make_unique<int[]>(7), std::make_unique<int[]>(7)};
    BiValues<float> values = {std::make_unique<float[]>(7), std::make_unique<float[]>(7)};

    int row_data[] = {1, 1, 2, 2, 2, 2, 2};
    int col_data[] = {2, 3, 1, 4, 3, 5, 1};
    float val1_data[] = {2.0, 3.0, 1.0, 4.0, 3.0, 5.0, 1.0};
    float val2_data[] = {20.0, 30.0, 10.0, 40.0, 30.0, 50.0, 10.0};

    std::copy(row_data, row_data + 7, indices[0].get());
    std::copy(col_data, col_data + 7, indices[1].get());
    std::copy(val1_data, val1_data + 7, values[0].get());
    std::copy(val2_data, val2_data + 7, values[1].get());

    int expected_rows[] = {1, 1, 2, 2, 2, 2};
    int expected_cols[] = {2, 3, 1, 3, 4, 5};
    float expected_vals1[] = {2, 3, 2, 3, 4, 5};
    float expected_vals2[] = {20, 30, 20, 30, 40, 50};

    std::size_t duplicates = merge_sort_coo(indices, values, 7);

    CHECK(duplicates == 1);  // One duplicate should be removed
    CHECK_VECTOR_EQUAL(vec(indices[0].get(), 6), std::vector<int>(expected_rows, expected_rows + 6));
    CHECK_VECTOR_EQUAL(vec(indices[1].get(), 6), std::vector<int>(expected_cols, expected_cols + 6));
    CHECK_VECTOR_EQUAL(vec(values[0].get(), 6), std::vector<float>(expected_vals1, expected_vals1 + 6));
    CHECK_VECTOR_EQUAL(vec(values[1].get(), 6), std::vector<float>(expected_vals2, expected_vals2 + 6));
}
