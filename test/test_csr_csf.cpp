#include "tests_main.h"

#include "csr.h"
#include "csf.h"

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

    // Convert vovov_to_csf with valid values
    /*SECTION("Convert VoVoV TO CSF With Values Test") {
        sili::pod_vector<sili::pod_vector<int>> col_indices{{1}};
        sili::pod_vector<sili::pod_vector<sili::pod_vector<int>>> fiber_indices{{{0}}};
        sili::pod_vector<sili::pod_vector<sili::pod_vector<float>>> fiber_values{{{1.0f}}};
        csf_struct csf = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);
        REQUIRE_MESSAGE(csf.rows == 1, "CSF conversion failed");
        REQUIRE_MESSAGE(csf.cols == 1, "CSF conversion failed");
    }

    // Convert vovov_to_csf with null values
    SECTION("Convert VoVoV TO CSF Without Values Test") {
        csf_struct csf = convert_vovov_to_csf(nullptr, nullptr, nullptr, 1, 1, 1, 1, 1);
        REQUIRE_MESSAGE(csf.rows == 0, "CSF conversion failed");
        REQUIRE_MESSAGE(csf.cols == 0, "CSF conversion failed");
    }*/
}
