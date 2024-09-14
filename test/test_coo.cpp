#include "coo.hpp"
#include "tests_main.h"
#include <random>
#include <vector>


/* #region Inplace Merge COO */

TEST_CASE("Inplace Merge COO - Simple Merge without Duplicates", "[inplace_merge_coo]") {
    std::vector<int> cols = {1, 3, 5, 7, 2, 4, 6, 8};
    std::vector<int> rows = {1, 1, 2, 2, 1, 1, 2, 2};
    std::vector<float> vals = {1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0};

    int left = 0;
    int mid = 3;
    int right = 7;

    // Expected output after merge (no duplicates)
    std::vector<int> expected_cols = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> expected_rows = {1, 1, 1, 1, 2, 2, 2, 2};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    size_t duplicates = inplace_merge_coo(cols.data(), rows.data(), vals.data(), left, mid, right);

    REQUIRE_MESSAGE(duplicates == 0, "No duplicates should be found");

    CHECK_VECTOR_EQUAL(vec(cols.data(), 8), expected_cols);
    CHECK_VECTOR_EQUAL(vec(rows.data(), 8), expected_rows);
    CHECK_VECTOR_EQUAL(vec(vals.data(), 8), expected_vals);
}
TEST_CASE("Inplace Merge COO - Merge with Duplicates", "[inplace_merge_coo]") {
    std::vector<int> cols = {1, 3, 5, 7, 1, 3, 5, 7}; // Duplicate entries in the second half
    std::vector<int> rows = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> vals = {1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0}; // Duplicate values

    int left = 0;
    int mid = 3;
    int right = 7;

    // Expected output after merge (duplicates removed)
    std::vector<int> expected_cols = {1, 3, 5, 7};           // After removing duplicates
    std::vector<int> expected_rows = {1, 1, 1, 1};           // Rows remain the same
    std::vector<float> expected_vals = {1.0, 3.0, 5.0, 7.0}; // Values corresponding to the unique row-col pairs

    size_t duplicates = inplace_merge_coo(cols.data(), rows.data(), vals.data(), left, mid, right);

    REQUIRE_MESSAGE(duplicates == 4, "There should be 4 duplicates");

    CHECK_VECTOR_EQUAL(vec(cols.data(), 4), expected_cols); // Check if columns match expected
    CHECK_VECTOR_EQUAL(vec(rows.data(), 4), expected_rows); // Check if rows match expected
    CHECK_VECTOR_EQUAL(vec(vals.data(), 4), expected_vals); // Check if values match expected
}

TEST_CASE("Inplace Merge COO - Complex Case with Mixed Duplicates and Non-Duplicates", "[inplace_merge_coo]") {
    std::vector<int> cols = {1, 2, 3, 7, 1, 3, 5, 8};                   // Some duplicates, some unique
    std::vector<int> rows = {1, 1, 1, 1, 1, 1, 2, 2};                   // Rows with duplicates
    std::vector<float> vals = {1.0, 2.0, 5.0, 7.0, 1.0, 3.0, 5.0, 8.0}; // Values with some duplicates

    int left = 0;
    int mid = 3;
    int right = 7;

    // Expected output after merge (duplicates removed, non-duplicates kept)
    std::vector<int> expected_cols = {1, 2, 3, 7, 5, 8};               // Sorted col-major with duplicates removed
    std::vector<int> expected_rows = {1, 1, 1, 1, 2, 2};               // Sorted row-major
    std::vector<float> expected_vals = {1.0, 2.0, 5.0, 7.0, 5.0, 8.0}; // Values corresponding to unique pairs

    size_t duplicates = inplace_merge_coo(cols.data(), rows.data(), vals.data(), left, mid, right);

    REQUIRE_MESSAGE(duplicates == 2, "There should be 2 duplicates");

    CHECK_VECTOR_EQUAL(vec(cols.data(), 6), expected_cols); // Check if columns match expected
    CHECK_VECTOR_EQUAL(vec(rows.data(), 6), expected_rows); // Check if rows match expected
    CHECK_VECTOR_EQUAL(vec(vals.data(), 6), expected_vals); // Check if values match expected
}
/* #endregion */


TEST_CASE("Insertion Sort COO - Already Sorted Input", "[insertion_sort_coo]") {
    std::vector<int> cols = {1, 2, 3, 4};
    std::vector<int> rows = {1, 1, 2, 2};
    std::vector<float> vals = {1.0, 2.0, 3.0, 4.0};

    int left = 0;
    int right = 3;

    // Expected output (same as input)
    std::vector<int> expected_cols = {1, 2, 3, 4};
    std::vector<int> expected_rows = {1, 1, 2, 2};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0};

    insertion_sort_coo(cols.data(), rows.data(), vals.data(), left, right);

    CHECK_VECTOR_EQUAL(vec(cols.data(), 4), expected_cols); // Check columns remain sorted
    CHECK_VECTOR_EQUAL(vec(rows.data(), 4), expected_rows); // Check rows remain sorted
    CHECK_VECTOR_EQUAL(vec(vals.data(), 4), expected_vals); // Check values remain unchanged
}

TEST_CASE("Insertion Sort COO - Reverse Order Input", "[insertion_sort_coo]") {
    std::vector<int> cols = {2, 1, 2, 1};
    std::vector<int> rows = {2, 2, 1, 1};
    std::vector<float> vals = {4.0, 3.0, 2.0, 1.0};

    int left = 0;
    int right = 3;

    // Expected output after sorting
    std::vector<int> expected_cols = {1, 2, 1, 2};
    std::vector<int> expected_rows = {1, 1, 2, 2};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0};

    insertion_sort_coo(cols.data(), rows.data(), vals.data(), left, right);

    CHECK_VECTOR_EQUAL(vec(cols.data(), 4), expected_cols);  // Check columns sorted
    CHECK_VECTOR_EQUAL(vec(rows.data(), 4), expected_rows);  // Check rows sorted
    CHECK_VECTOR_EQUAL(vec(vals.data(), 4), expected_vals);  // Check values in correct order
}

TEST_CASE("Insertion Sort COO - Same Rows, Unsorted Columns", "[insertion_sort_coo]") {
    std::vector<int> cols = {4, 2, 3, 1};
    std::vector<int> rows = {1, 1, 1, 1};
    std::vector<float> vals = {4.0, 2.0, 3.0, 1.0};

    int left = 0;
    int right = 3;

    // Expected output after sorting (sorted by columns within the same row)
    std::vector<int> expected_cols = {1, 2, 3, 4};
    std::vector<int> expected_rows = {1, 1, 1, 1};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0};

    insertion_sort_coo(cols.data(), rows.data(), vals.data(), left, right);

    CHECK_VECTOR_EQUAL(vec(cols.data(), 4), expected_cols);  // Check columns sorted
    CHECK_VECTOR_EQUAL(vec(rows.data(), 4), expected_rows);  // Ensure rows remain the same
    CHECK_VECTOR_EQUAL(vec(vals.data(), 4), expected_vals);  // Check values sorted correctly
}

TEST_CASE("Insertion Sort COO - Simple Case without Duplicates", "[insertion_sort_coo]") {
    std::vector<int> cols = {4, 2, 3, 1};
    std::vector<int> rows = {1, 1, 1, 1};
    std::vector<float> vals = {4.0, 2.0, 3.0, 1.0};

    // Expected output after sorting
    std::vector<int> expected_cols = {1, 2, 3, 4};
    std::vector<int> expected_rows = {1, 1, 1, 1};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0};

    size_t duplicates = insertion_sort_coo(cols.data(), rows.data(), vals.data(), 0, (int)cols.size() - 1);

    REQUIRE_MESSAGE(duplicates == 0, "No duplicates should be found");

    CHECK_VECTOR_EQUAL(vec(cols.data(), 4), expected_cols);
    CHECK_VECTOR_EQUAL(vec(rows.data(), 4), expected_rows);
    CHECK_VECTOR_EQUAL(vec(vals.data(), 4), expected_vals);
}

TEST_CASE("Insertion Sort COO - Case with Duplicates", "[insertion_sort_coo]") {
    std::vector<int> cols = {1, 3, 2, 2, 3};
    std::vector<int> rows = {1, 1, 1, 1, 1};
    std::vector<float> vals = {1.0, 3.0, 2.0, 2.0, 3.0};

    // Expected output after sorting and removing duplicates
    std::vector<int> expected_cols = {1, 2, 3};
    std::vector<int> expected_rows = {1, 1, 1};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0};

    size_t duplicates = insertion_sort_coo(cols.data(), rows.data(), vals.data(), 0, (int)cols.size() - 1);

    REQUIRE_MESSAGE(duplicates == 2, "There should be 2 duplicates");

    CHECK_VECTOR_EQUAL(vec(cols.data(), 3), expected_cols);
    CHECK_VECTOR_EQUAL(vec(rows.data(), 3), expected_rows);
    CHECK_VECTOR_EQUAL(vec(vals.data(), 3), expected_vals);
}


TEST_CASE("Merge Sort COO - Simple Case", "[merge_sort_coo]") {
    std::vector<int> cols = {4, 2, 3, 1};
    std::vector<int> rows = {1, 1, 1, 1};
    std::vector<float> vals = {4.0, 2.0, 3.0, 1.0};

    // Expected output after sorting by columns within the same row
    std::vector<int> expected_cols = {1, 2, 3, 4};
    std::vector<int> expected_rows = {1, 1, 1, 1};
    std::vector<float> expected_vals = {1.0, 2.0, 3.0, 4.0};

    merge_sort_coo(cols.data(), rows.data(), vals.data(), cols.size());

    CHECK_VECTOR_EQUAL(vec(cols.data(), 4), expected_cols);  // Columns sorted
    CHECK_VECTOR_EQUAL(vec(rows.data(), 4), expected_rows);  // Rows remain the same
    CHECK_VECTOR_EQUAL(vec(vals.data(), 4), expected_vals);  // Values reordered
}

TEST_CASE("Merge Sort COO - Complex Case with Duplicates", "[merge_sort_coo]") {
    std::vector<int> cols = {2, 3, 1, 4, 3, 5, 1};
    std::vector<int> rows = {1, 1, 2, 2, 2, 2, 3};
    std::vector<float> vals = {2.0, 3.0, 1.0, 4.0, 3.0, 5.0, 1.0};

    // Expected output after sorting (row-major, then column-major order)
    std::vector<int> expected_cols = {2, 3, 1, 3, 4, 5, 1};
    std::vector<int> expected_rows = {1, 1, 2, 2, 2, 2, 3};
    std::vector<float> expected_vals = {2.0, 3.0, 1.0, 3.0, 4.0, 5.0, 1.0};

    merge_sort_coo(cols.data(), rows.data(), vals.data(), cols.size());

    CHECK_VECTOR_EQUAL(vec(cols.data(), 7), expected_cols);  // Check sorted columns
    CHECK_VECTOR_EQUAL(vec(rows.data(), 7), expected_rows);  // Check row-major sorting
    CHECK_VECTOR_EQUAL(vec(vals.data(), 7), expected_vals);  // Ensure values are reordered correctly
}

TEST_CASE("Merge Sort COO - Large Random Input", "[merge_sort_coo][stress]") {
    const int size = 10000;
    std::vector<int> cols(size);
    std::vector<int> rows(size);
    std::vector<float> vals(size);

    // Initialize with random values
    std::mt19937 rng(123);  // Seed for reproducibility
    for (int i = 0; i < size; ++i) {
        rows[i] = rng() % 100;  // Random row indices
        cols[i] = rng() % 100;  // Random column indices
        vals[i] = rng() / float(rng.max());  // Random values between 0 and 1
    }

    // Copy vectors for comparison later
    std::vector<int> expected_cols = cols;
    std::vector<int> expected_rows = rows;

    // Use standard sort to create the expected sorted result (without duplicates)
    std::vector<std::tuple<int, int>> coo_tuples(size);
    for (int i = 0; i < size; ++i) {
        coo_tuples[i] = std::make_tuple(rows[i], cols[i]);
    }

    // Sort based on rows and columns
    std::sort(coo_tuples.begin(), coo_tuples.end());

    // Use unique to remove duplicates from the sorted tuples
    auto last = std::unique(coo_tuples.begin(), coo_tuples.end());
    size_t unique_size = std::distance(coo_tuples.begin(), last);

    // Resize the expected vectors to the unique size
    expected_cols.resize(unique_size);
    expected_rows.resize(unique_size);

    // Populate the expected sorted vectors without duplicates
    for (size_t i = 0; i < unique_size; ++i) {
        expected_rows[i] = std::get<0>(coo_tuples[i]);
        expected_cols[i] = std::get<1>(coo_tuples[i]);
    }

    // Perform the merge sort on the COO arrays (which should also remove duplicates)
    size_t duplicates = merge_sort_coo(cols.data(), rows.data(), vals.data(), size);

    // Ensure the number of duplicates removed matches
    REQUIRE_MESSAGE(duplicates == (size - unique_size), "The number of duplicates removed should match.");

    // Check that the resulting arrays match the expected sorted arrays
    CHECK_VECTOR_EQUAL(vec(cols.data(), unique_size), expected_cols);  // Columns sorted
    CHECK_VECTOR_EQUAL(vec(rows.data(), unique_size), expected_rows);  // Rows sorted
}

