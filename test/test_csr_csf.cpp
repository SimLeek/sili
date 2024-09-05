#include <catch2/catch_all.hpp>

// thanks: https://github.com/catchorg/Catch2/issues/929#issuecomment-308663820
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)
#define CHECK_MESSAGE(cond, msg) do { INFO(msg); CHECK(cond); } while((void)0, 0)

#include "csr.h"
#include "csf.h"

TEST_CASE("Convert Functions Tests") {
    // ...

    // Convert vov_to_csr with valid values
    SECTION("Convert VOVO TO CSR With Values Test") {
        std::vector<std::vector<int>> indices{{1}};
        std::vector<std::vector<float>> values{{1.0f}};
        csr_struct csr = convert_vov_to_csr(&indices, &values, nullptr, 1, 1, 1);
        REQUIRE_MESSAGE(csr.nnz == 1, "CSR conversion failed");
    }

    // Convert vov_to_csr with null values
    SECTION("Convert VOVO TO CSR Without Values Test") {
        std::vector<std::vector<int>> indices{{1}};
        csr_struct csr = convert_vov_to_csr(nullptr, nullptr, nullptr, 1, 1, 1);
        REQUIRE_MESSAGE(csr.nnz == 0, "CSR conversion failed");
    }

    // Convert vovov_to_csf with valid values
    SECTION("Convert VoVoV TO CSF With Values Test") {
        std::vector<std::vector<int>> col_indices{{1}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}};
        csf_struct csf = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);
        REQUIRE_MESSAGE(csf.rows == 1, "CSF conversion failed");
        REQUIRE_MESSAGE(csf.cols == 1, "CSF conversion failed");
    }

    // Convert vovov_to_csf with null values
    SECTION("Convert VoVoV TO CSF Without Values Test") {
        csf_struct csf = convert_vovov_to_csf(nullptr, nullptr, nullptr, 1, 1, 1, 1, 1);
        REQUIRE_MESSAGE(csf.rows == 0, "CSF conversion failed");
        REQUIRE_MESSAGE(csf.cols == 0, "CSF conversion failed");
    }
}
