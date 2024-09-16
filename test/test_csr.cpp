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
    std::mt19937 generator(static_cast<unsigned long>(std::time(0))); // Fix the seed for repeatability

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
            std::vector<size_t>({5})
            );
    }
}