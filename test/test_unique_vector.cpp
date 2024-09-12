#include "tests_main.h"
#include "unique_vector.hpp"
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

TEST_CASE("test unique vector can be different type but must be copyable", "[do_linear_sidlso_fwd]")
{
    
    REQUIRE_NOTHROW(sili::unique_vector<sili::unique_vector<int>>{
        {0, 1, 2, 3, 4},
        {5, 6,7, 8, 9},
        {10, 11, 12, 13,14}, 
        {15, 16, 17, 18, 19}
    });

    REQUIRE_NOTHROW(sili::unique_vector<sili::unique_vector<float>>{
        {-38,-108, -178, -248, -318},
        {-388, -458, -528, -598, -668},
        {-738,-808, -878, -948, -1018}, 
        { -1088, -1158, -1228, -1298, -1368}
});
}