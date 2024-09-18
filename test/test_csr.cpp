#include "tests_main.h"

#include "csr.hpp"
#include <cstddef>
#include <vector>

TEST_CASE("Convert Functions Tests") {

    // Convert vov_to_csr with valid values
    SECTION("Convert VOV TO CSR With Values Test") {
        sili::unique_vector<sili::unique_vector<int>> indices{{0,1,2},{2,3,4}, {3,4,5}, {4,5,6}};
        sili::unique_vector<sili::unique_vector<float>> values{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}};
        auto csr = convert_vov_to_csr(&indices, &values, 7, 4, 12);
        
        REQUIRE(csr.cols == 7);
        REQUIRE(csr.rows == 4);
        for(int i=0;i<4;i++){
            for(int j=0;j<3;j++){
                CHECK_MESSAGE(csr.indices[i*3+j]==indices[i][j], "index mismatch at i,j: "<<i<<", "<<j);
                CHECK_MESSAGE(csr.values[i*3+j]==values[i][j], "value mismatch at i,j: "<<i<<", "<<j);
            }
        }
        sili::unique_vector<int> ptrs{0, 3, 6, 9, 12};
        for(int i=0;i<5;i++){
            CHECK_MESSAGE(csr.ptrs[i]==ptrs[i], "ptr mismatch at i: "<<i);
        }
        REQUIRE(csr.nnz() == 12);
    }

    // Convert vov_to_csr with null values
    SECTION("Convert VOV TO CSR Without Values Test") {
        sili::unique_vector<sili::unique_vector<int>> indices{{0,1,2},{2,3,4}, {3,4,5}, {4,5,6}};
        auto csr = convert_vov_to_csr<int, float>(&indices, nullptr, 7, 4, 12);
        
        REQUIRE(csr.cols == 7);
        REQUIRE(csr.rows == 4);
        for(int i=0;i<4;i++){
            for(int j=0;j<3;j++){
                CHECK_MESSAGE(csr.indices[i*3+j]==indices[i][j], "index mismatch at i,j: "<<i<<", "<<j);
            }
        }
        sili::unique_vector<int> ptrs{0, 3, 6, 9, 12};
        for(int i=0;i<5;i++){
            CHECK_MESSAGE(csr.ptrs[i]==ptrs[i], "ptr mismatch at i: "<<i);
        }
        REQUIRE(csr.nnz() == 12);
    }

    SECTION("Convert VOV TO CSR With Incorrect Non-Zero Count") {
        sili::unique_vector<sili::unique_vector<int>> indices{{0,1,2},{2,3,4}, {3,4,5}, {4,5,6}};
        sili::unique_vector<sili::unique_vector<float>> values{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}};
        REQUIRE_THROWS_AS(convert_vov_to_csr(&indices, &values, 7, 4, 11), std::runtime_error);
    }

}


TEST_CASE("generate_random_csr", "[csr_matrix]") {
    // Create a sample CSR matrix to avoid
    auto indices = sili::unique_vector<sili::unique_vector<size_t>>{
        {0, 3, 6, 9},
        {0, 1, 2, 3, 4},
        {1, 2, 3, 4, 5, 6},
        {2, 3, 4, 5, 6, 7},
    };
    auto values = sili::unique_vector<sili::unique_vector<float>>{
        {0.1, 0.3, 0.6, 0.9},
        {0.2, 0.1, 0.2, 0.2, 0.3, 0.4},
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
        {0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
    };
    csr_struct<size_t, float> csr_avoid_pts = convert_vov_to_csr(
        &indices, &values, (size_t)10, (size_t)4, (size_t)22);

    // Random number generator
    std::uniform_int_distribution<size_t> index_dist(0, csr_avoid_pts.rows * csr_avoid_pts.cols);
    std::mt19937_64 generator(static_cast<unsigned long>(std::time(0))); // Fix the seed for repeatability

    // Basic functionality test
    SECTION("Basic Functionality") {
        size_t nnz = 10;
        csr_struct<size_t, float> random_csr = generate_random_csr(
            nnz, csr_avoid_pts, index_dist, generator, 4);

        REQUIRE(random_csr.rows == csr_avoid_pts.rows);
        REQUIRE(random_csr.cols == csr_avoid_pts.cols);
        REQUIRE(random_csr.nnz() == nnz);

        // Convert CSR matrices to sets of pairs
    std::set<std::pair<size_t, size_t>> csr_avoid_pts_set;
    for (size_t i = 0; i < csr_avoid_pts.rows; ++i) {
        for (size_t j = csr_avoid_pts.ptrs[i]; j < csr_avoid_pts.ptrs[i + 1]; ++j) {
            csr_avoid_pts_set.insert({csr_avoid_pts.indices[j], i});
        }
    }

    std::set<std::pair<size_t, size_t>> random_csr_set;
    for (size_t i = 0; i < random_csr.rows; ++i) {
        for (size_t j = random_csr.ptrs[i]; j < random_csr.ptrs[i + 1]; ++j) {
            random_csr_set.insert({random_csr.indices[j], i});
        }
    }
    std::set<std::pair<size_t, size_t>> intersection_set;
    // Check for intersection
    std::set_intersection(random_csr_set.begin(), random_csr_set.end(),
                          csr_avoid_pts_set.begin(), csr_avoid_pts_set.end(),
                          std::inserter(intersection_set, intersection_set.begin()));

    REQUIRE(intersection_set.empty());
    }

    SECTION("Parallel vs. Non-Parallel") {
        csr_struct<size_t, float> random_csr_parallel = generate_random_csr(
            (size_t)100, csr_avoid_pts, index_dist, generator, 4);
        csr_struct<size_t, float> random_csr_sequential = generate_random_csr(
            (size_t)100, csr_avoid_pts, index_dist, generator, 1);

        // Compare the results
        REQUIRE(random_csr_parallel.rows == random_csr_sequential.rows);
        REQUIRE(random_csr_parallel.cols == random_csr_sequential.cols);
        REQUIRE(random_csr_parallel.nnz() == random_csr_sequential.nnz());

        // Convert CSR matrices to vectors
        std::vector<size_t> ptrs_parallel = vec(random_csr_parallel.ptrs.get(), random_csr_parallel.rows + 1);
        std::vector<size_t> indices_parallel = vec(random_csr_parallel.indices.get(), random_csr_parallel.nnz());
        std::vector<size_t> ptrs_sequential = vec(random_csr_sequential.ptrs.get(), random_csr_sequential.rows + 1);
        std::vector<size_t> indices_sequential = vec(random_csr_sequential.indices.get(), random_csr_sequential.nnz());

        // Compare the vectors
        CHECK_VECTOR_EQUAL(ptrs_parallel, ptrs_sequential);
        CHECK_VECTOR_EQUAL(indices_parallel, indices_sequential);    
    }

    // Edge case 1: Large number of insertions
    SECTION("Large Insertions") {
        csr_struct<size_t, float> random_csr = generate_random_csr(
            (size_t)10000, csr_avoid_pts, index_dist, generator, 4);

        CHECK(random_csr.values.get()==nullptr);
        CHECK(random_csr.nnz()==19);
        CHECK_VECTOR_EQUAL(
            vec(random_csr.ptrs.get(), random_csr.rows+1), 
            std::vector<size_t>({0, 6, 11, 15, 19})  // initializer lists need to be inside parantheses or are passed to CHECK_VECTOR_EQUAL
            );
        CHECK_VECTOR_EQUAL(
            vec(random_csr.indices.get(), random_csr.nnz()), 
            std::vector<size_t>({1, 2, 4, 5, 7, 8, 5, 6, 7, 8, 9, 0, 7, 8, 9, 0, 1, 8, 9})
            );
    }

    // Edge case 2: Small number of insertions
    SECTION("Small Insertions") {
        csr_struct<size_t, float> random_csr = generate_random_csr(
            (size_t)1, csr_avoid_pts, index_dist, generator, 4, 12);

        CHECK(random_csr.values.get()==nullptr);
        CHECK(random_csr.nnz()==1);
        CHECK_VECTOR_EQUAL(
            vec(random_csr.ptrs.get(), random_csr.rows+1), 
            std::vector<size_t>({0, 0, 1, 1, 1})  // initializer lists need to be inside parantheses or are passed to CHECK_VECTOR_EQUAL
            );
        CHECK_VECTOR_EQUAL(
            vec(random_csr.indices.get(), random_csr.nnz()), 
            std::vector<size_t>({6})
            );
    }
}

TEST_CASE("merge_csrs basic functionality", "[merge_csrs]") {
    SECTION("Merge two CSRs") {
        auto indices1 = sili::unique_vector<sili::unique_vector<size_t>>{
            {0, 1},
            {2},
            {3}
        };
        auto values1 = sili::unique_vector<sili::unique_vector<float>>{
            {1.0f, 2.0f},
            {3.0f},
            {4.0f}
        };
        csr_struct<size_t, float> csr1 = convert_vov_to_csr(
            &indices1, &values1, (size_t)4, (size_t)3, (size_t)4);

        auto indices2 = sili::unique_vector<sili::unique_vector<size_t>>{
            {1},
            {0, 3},
            {}
        };
        auto values2 = sili::unique_vector<sili::unique_vector<float>>{
            {5.0f},
            {6.0f, 7.0f},
            {}
        };
        csr_struct<size_t, float> csr2 = convert_vov_to_csr(
            &indices2, &values2, (size_t)4, (size_t)3, (size_t)3);

        auto result = merge_csrs(csr1, csr2, 2);

        REQUIRE(result.rows == 3);
        REQUIRE(result.cols == 4);
        REQUIRE(result.nnz() == 6);

        std::vector<size_t> expected_ptrs = {0, 2, 5, 6};
        std::vector<size_t> expected_indices = {0, 1, 0, 2, 3, 3};
        std::vector<float> expected_values = {1.0f, 5.0f, 6.0f, 3.0f, 7.0f, 4.0f};

        std::vector<size_t> result_ptrs = vec(result.ptrs.get(), result.rows + 1);
        std::vector<size_t> result_indices = vec(result.indices.get(), result.nnz());
        std::vector<float> result_values = vec(result.values.get(), result.nnz());

        CHECK_VECTOR_EQUAL(result_ptrs, expected_ptrs);
        CHECK_VECTOR_EQUAL(result_indices, expected_indices);
        CHECK_VECTOR_EQUAL(result_values, expected_values);
    }
}

//I don't actually care about this rn
/*TEST_CASE("merge_csrs edge cases", "[merge_csrs]") {
    SECTION("Merge with an empty CSR") {
        auto indices_non_empty = sili::unique_vector<sili::unique_vector<size_t>>{{0}, {1}};
        auto values_non_empty = sili::unique_vector<sili::unique_vector<float>>{{1.0f}, {2.0f}};
        csr_struct<size_t, float> non_empty_csr = convert_vov_to_csr(
            &indices_non_empty, &values_non_empty, (size_t)2, (size_t)2, (size_t)2);

        auto indices_empty = sili::unique_vector<sili::unique_vector<size_t>>{{}, {}};
        auto values_empty = sili::unique_vector<sili::unique_vector<float>>{{}, {}};
        csr_struct<size_t, float> empty_csr = convert_vov_to_csr(
            &indices_empty, &values_empty, (size_t)2, (size_t)2, (size_t)0);

        auto result = merge_csrs(non_empty_csr, empty_csr, 2);

        REQUIRE(result.rows == 2);
        REQUIRE(result.cols == 2);
        REQUIRE(result.nnz() == 2);

        std::vector<size_t> expected_ptrs = {0, 1, 2};
        std::vector<size_t> expected_indices = {0, 1};
        std::vector<float> expected_values = {1.0f, 2.0f};

        std::vector<size_t> result_ptrs = vec(result.ptrs.get(), result.rows + 1);
        std::vector<size_t> result_indices = vec(result.indices.get(), result.nnz());
        std::vector<float> result_values = vec(result.values.get(), result.nnz());

        CHECK_VECTOR_EQUAL(result_ptrs, expected_ptrs);
        CHECK_VECTOR_EQUAL(result_indices, expected_indices);
        CHECK_VECTOR_EQUAL(result_values, expected_values);
    }
}*/

TEST_CASE("merge_csrs parallel vs sequential", "[merge_csrs]") {
    auto indices = sili::unique_vector<sili::unique_vector<size_t>>{
        {0, 3, 6},
        {0, 1, 2},
        {1, 2, 3},
    };
    auto values = sili::unique_vector<sili::unique_vector<float>>{
        {0.1f, 0.3f, 0.6f},
        {0.2f, 0.1f, 0.2f},
        {0.1f, 0.2f, 0.3f},
    };
    csr_struct<size_t, float> csr1 = convert_vov_to_csr(
        &indices, &values, (size_t)7, (size_t)3, (size_t)9);

    auto indices2 = sili::unique_vector<sili::unique_vector<size_t>>{
        {1, 4},
        {3, 5},
        {2, 6},
    };
    auto values2 = sili::unique_vector<sili::unique_vector<float>>{
        {0.4f, 0.7f},
        {0.5f, 0.8f},
        {0.6f, 0.9f},
    };
    csr_struct<size_t, float> csr2 = convert_vov_to_csr(
        &indices2, &values2, (size_t)7, (size_t)3, (size_t)6);

    auto result_parallel = merge_csrs(csr1, csr2, 4);
    auto result_sequential = merge_csrs(csr1, csr2, 1);

    // Compare the results
    REQUIRE(result_parallel.rows == result_sequential.rows);
    REQUIRE(result_parallel.cols == result_sequential.cols);
    REQUIRE(result_parallel.nnz() == result_sequential.nnz());

    // Convert CSR matrices to vectors
    std::vector<size_t> ptrs_parallel = vec(result_parallel.ptrs.get(), result_parallel.rows + 1);
    std::vector<size_t> indices_parallel = vec(result_parallel.indices.get(), result_parallel.nnz());
    std::vector<float> values_parallel = vec(result_parallel.values.get(), result_parallel.nnz());
    std::vector<size_t> ptrs_sequential = vec(result_sequential.ptrs.get(), result_sequential.rows + 1);
    std::vector<size_t> indices_sequential = vec(result_sequential.indices.get(), result_sequential.nnz());
    std::vector<float> values_sequential = vec(result_sequential.values.get(), result_sequential.nnz());

    // Compare the vectors
    CHECK_VECTOR_EQUAL(ptrs_parallel, ptrs_sequential);
    CHECK_VECTOR_EQUAL(indices_parallel, indices_sequential);
    CHECK_VECTOR_EQUAL(values_parallel, values_sequential);

    CHECK_VECTOR_EQUAL(ptrs_parallel, std::vector<size_t>({0,5,10,14}));
    CHECK_VECTOR_EQUAL(indices_parallel, std::vector<size_t>({0,1,3,4,6,0,1,2,3,5,1,2,3,6}));
    CHECK_VECTOR_ALMOST_EQUAL(values_parallel, std::vector<float>({0.1,0.4,0.3,0.7,0.6,0.2,0.1,0.2,0.5,0.8,0.1,0.6,0.3,0.9}),0.0000001);

}

/*TODO: still need to be tested (but they probably work and I've had enough of this, so moving on for now): 
 * remove_element_from_csr
 * add_few_random_to_csr
 * CSRStarmap constructor
   * addRandomValue
   * addRandomElements
   * iterate
*/