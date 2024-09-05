#include <catch2/catch_all.hpp>

// thanks: https://github.com/catchorg/Catch2/issues/929#issuecomment-308663820
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)
#define CHECK_MESSAGE(cond, msg) do { INFO(msg); CHECK(cond); } while((void)0, 0)

#include "linear.cpp"

TEST_CASE("_do_linear_sidlso_fwd basic functionality", "[_do_linear_sidlso_fwd]") {
    int num_cpus = 4;
    int input_size = 10;
    int output_size = 20;
    int batch = 5;
    csr_struct input_csr;
    input_csr.ptrs = new int[num_cpus + 1]{0, 3, 6, 9, 12}; // Example pointers
    input_csr.indices = new int[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // Example indices
    input_csr.values = new float[12]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}; // Example values
    input_csr.nnz = 12;
    input_csr.rows = 4;
    input_csr.cols = 10;

    float*W = new float[input_size *output_size];
    for (int i = 0; i < input_size* output_size; i++)
        W[i] = i % 10000;

    std::vector<std::vector<int>> row_indices_chunks(num_cpus);
    std::vector<std::vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    _do_linear_sidlso_fwd(num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);

    // Verify results
    for (const auto& vec : row_indices_chunks) {
        REQUIRE(vec.size() == 5 || vec.size() == 4);
    }

    for (size_t i = 0; i < row_values_chunks.size(); i++) {
        for (size_t j = 0; j < row_values_chunks[i].size(); j++) {
            REQUIRE(row_values_chunks[i][j] >= 0 && row_values_chunks[i][j] <= 120);
        }
    }

    delete[] input_csr.ptrs;
    delete[] input_csr.indices;
    delete[] input_csr.values;
    delete[] W;
}

TEST_CASE("_do_linear_sidlso_fwd zero eps value", "[_do_linear_sidlso_fwd]")
{
    int num_cpus = 4;
    int input_size = 10;
    int output_size = 20;
    int batch = 5;
    csr_struct input_csr;
    input_csr.ptrs = new int[num_cpus+1]{0,3,6,9,12};
    input_csr.indices = new int[input_size*batch]{0,1,2,3,4,5,6,7,8,9,10,11};
    input_csr.values = new float[input_size*batch]{1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f};

    float*W = new float[input_size*output_size];
    for(int i=0;i<input_size*output_size;++i){
        W[i]=i;
    }

    std::vector<std::vector<int>> row_indices_chunks(num_cpus);
    std::vector<std::vector<float>> row_values_chunks(num_cpus);

    float eps = 0.00f;

    _do_linear_sidlso_fwd(num_cpus,input_size,output_size,batch,input_csr,W,row_indices_chunks,row_values_chunks,eps);

    REQUIRE(row_indices_chunks.size() == num_cpus);
    REQUIRE(row_values_chunks.size() == num_cpus);

    delete[] input_csr.ptrs;
    delete[] input_csr.indices;
    delete[] input_csr.values;
    delete[] W;
}

TEST_CASE("_do_linear_sidlso_fwd negative input values", "[_do_linear_sidlso_fwd]")
{
    int num_cpus = 4;
    int input_size = 10;
    int output_size = 20;
    int batch = 5;
    csr_struct input_csr;
    input_csr.ptrs = new int[num_cpus+1]{0,3,6,9,12};
    input_csr.indices = new int[input_size*batch]{0,1,2,3,4,5,6,7,8,9,10,11};
    input_csr.values = new float[input_size*batch]{-1.0f,-2.0f,-3.0f,-4.0f,-5.0f,-6.0f,-7.0f,-8.0f,-9.0f,-10.0f,-11.0f,-12.0f};

    float*W = new float[input_size*output_size];
    for(int i=0;i<input_size*output_size;++i){
        W[i]=i;
    }

    std::vector<std::vector<int>> row_indices_chunks(num_cpus);
    std::vector<std::vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    _do_linear_sidlso_fwd(num_cpus,input_size,output_size,batch,input_csr,W,row_indices_chunks,row_values_chunks,eps);

    REQUIRE(row_indices_chunks.size() == num_cpus);
    REQUIRE(row_values_chunks.size() == num_cpus);

    delete[] input_csr.ptrs;
    delete[] input_csr.indices;
    delete[] input_csr.values;
    delete[] W;
}

TEST_CASE("_do_linear_sidlso_fwd edge case: zero-sized inputs", "[_do_linear_sidlso_fwd]") {
    int num_cpus = 4;
    int input_size = 0;
    int output_size = 20;
    int batch = 5;
    csr_struct input_csr;
    input_csr.ptrs = nullptr;
    input_csr.indices = nullptr;
    input_csr.values = nullptr;
    input_csr.nnz = 0;
    input_csr.rows = 0;
    input_csr.cols = 0;

    float*W = new float[input_size* output_size];

    std::vector<std::vector<int>> row_indices_chunks(num_cpus);
    std::vector<std::vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

_do_linear_sidlso_fwd(num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);

    // Verify results
    for (const auto& vec : row_indices_chunks) {
        REQUIRE(vec.empty());
    }

    for (size_t i = 0; i < row_values_chunks.size(); i++) {
        for (size_t j = 0; j < row_values_chunks[i].size(); j++) {
            REQUIRE(false); // Should never reach here
        }
    }

    delete[] W;
}

TEST_CASE("_do_linear_sidlso_fwd edge case: single-threaded execution", "[_do_linear_sidlso_fwd]") {
    int num_cpus = 1;
    int input_size = 10;
    int output_size = 20;
    int batch = 5;
    csr_struct input_csr;
    input_csr.ptrs = new int[num_cpus + 1]{0, 3, 6, 9, 12}; // Example pointers
    input_csr.indices = new int[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; // Example indices
    input_csr.values = new float[12]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}; // Example values
    input_csr.nnz = 12;
    input_csr.rows = 4;
    input_csr.cols = 10;

    float*W = new float[input_size*output_size];
    for (int i = 0; i < input_size* output_size; i++)
        W[i] = i % 10000;

    std::vector<std::vector<int>> row_indices_chunks(num_cpus);
    std::vector<std::vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    _do_linear_sidlso_fwd(num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);

    // Verify results
    for (const auto& vec : row_indices_chunks) {
        REQUIRE(vec.size() == 20);
    }

    for (size_t i = 0; i < row_values_chunks.size(); i++) {
        for (size_t j = 0; j < row_values_chunks[i].size(); j++) {
            REQUIRE(row_values_chunks[i][j] >= 0 && row_values_chunks[i][j] <= 120);
        }
    }

    delete[] input_csr.ptrs;
    delete[] input_csr.indices;
    delete[] input_csr.values;
    delete[] W;
}

TEST_CASE("Linear SIDLSO Test Suite") {
    SECTION("Test Linear SIDLSO with valid inputs") {
        int batch_size = 10;
        int input_size = 20;
        int output_size = 30;
        csr_struct input_csr;
        float*W = new float[input_size *output_size];

        // Initialize W with random values
        for (int i = 0; i < input_size* output_size; ++i) {
            W[i] = static_cast<float>(rand() % 100) / 100.0f;
        }

        csr_struct output_csr = linear_sidlso(batch_size, input_size, output_size, input_csr, W);

        // Check if output_csr has correct dimensions
        REQUIRE_MESSAGE(output_csr.rows == batch_size && output_csr.cols == output_size,
                       "Output CSR dimensions incorrect.");

        // Check if there are no zeros or values less than epsilon in output_csr.values
        for (int i = 0; i < output_csr.nnz; ++i) {
            CHECK_MESSAGE(output_csr.values[i] >= std::numeric_limits<float>::epsilon(),
                          "Value at index [" << i << "] is zero or less than epsilon.");
        }
    }

    SECTION("Test Linear SIDLSO with invalid inputs") {
        int batch_size = 0;
        int input_size = 20;
        int output_size = 30;
        csr_struct input_csr;
        float*W = new float[input_size *output_size];

        // Initialize W with random values
        for (int i = 0; i < input_size* output_size; ++i) {
            W[i] = static_cast<float>(rand() % 100) / 100.0f;
        }

        csr_struct output_csr = linear_sidlso(batch_size, input_size, output_size, input_csr, W);

        // Check if output_csr has correct dimensions
        REQUIRE_MESSAGE(output_csr.rows == batch_size && output_csr.cols == output_size,
                       "Output CSR dimensions incorrect.");

        // Check if there are no zeros or values less than epsilon in output_csr.values
        for (int i = 0; i < output_csr.nnz; ++i) {
            CHECK_MESSAGE(output_csr.values[i] >= std::numeric_limits<float>::epsilon(),
                          "Value at index [" << i << "] is zero or less than epsilon.");
        }
    }

    SECTION("Test Linear SIDLSO edge case") {
        int batch_size = 1;
        int input_size = 1;
        int output_size = 1;
        csr_struct input_csr;
        float*W = new float[input_size * output_size];

        // Initialize W with random value
        W[0] = static_cast<float>(rand() % 100) / 100.0f;

        csr_struct output_csr = linear_sidlso(batch_size, input_size, output_size, input_csr, W);

        // Check if output_csr has correct dimensions
        REQUIRE_MESSAGE(output_csr.rows == batch_size && output_csr.cols == output_size,
                       "Output CSR dimensions incorrect.");

        // Check if there are no zeros or values less than epsilon in output_csr.values
        for (int i = 0; i < output_csr.nnz; ++i) {
            CHECK_MESSAGE(output_csr.values[i] >= std::numeric_limits<float>::epsilon(),
                          "Value at index [" << i << "] is zero or less than epsilon.");
        }
    }
}








TEST_CASE("_assign_spv_chunks_to_batch basic functionality", "[_assign_spv_chunks_to_batch]")
{
    int batch = 0;
    int num_cpus = 4;
    std::vector<size_t> vec_assign_locs{0,5,10,15,20};
    std::vector<std::vector<int>> out_idx(1);
    std::vector<std::vector<float>> out_val(1);
    std::vector<std::vector<int>> row_indices_chunks{{0,1,2},{3,4},{5,6,7}};
    std::vector<std::vector<float>> row_values_chunks{{0.1f,0.2f,0.3f},{0.4f,0.5f},{0.6f,0.7f,0.8f}};
    int nnz = 0;

    _assign_spv_chunks_to_batch(batch,num_cpus,vec_assign_locs,out_idx,out_val,row_indices_chunks,row_values_chunks,nnz);

    REQUIRE_MESSAGE(nnz==15,"NNZ count incorrect");
    REQUIRE_MESSAGE(out_idx[0].size()==20,"Output idx size incorrect");
    REQUIRE_MESSAGE(out_val[0].size()==20,"Output val size incorrect");

    for(size_t i=0;i<out_idx[0].size();++i){
        REQUIRE_MESSAGE(i==out_idx[0][i],"Idx mismatch at position "+std::to_string(i));
    }

    for(size_t i=0;i<out_val[0].size();++i){
        REQUIRE_MESSAGE(std::abs(out_val[0][i]-i*0.1f)<0.001f,"Val mismatch at position "+std::to_string(i));
    }
}

TEST_CASE("_assign_spv_chunks_to_batch empty row indices chunks", "[_assign_spv_chunks_to_batch]")
{
    int batch = 0;
    int num_cpus = 4;
    std::vector<size_t> vec_assign_locs{0,5,10,15,20};
    std::vector<std::vector<int>> out_idx(1);
    std::vector<std::vector<float>> out_val(1);
    std::vector<std::vector<int>> row_indices_chunks{{{},{}},{},{}};
    std::vector<std::vector<float>> row_values_chunks{{{},{}}};
    int nnz = 0;

_assign_spv_chunks_to_batch(batch,num_cpus,vec_assign_locs,out_idx,out_val,row_indices_chunks,row_values_chunks,nnz);

    REQUIRE_MESSAGE(nnz==0,"NNZ count incorrect");
    REQUIRE_MESSAGE(out_idx[0].empty(),"Output idx not empty");
    REQUIRE_MESSAGE(out_val[0].empty(),"Output val not empty");
}

TEST_CASE("_assign_spv_chunks_to_batch single thread execution", "[_assign_spv_chunks_to_batch]")
{
    int batch = 0;
    int num_cpus = 1;
    std::vector<size_t> vec_assign_locs{0,5};
    std::vector<std::vector<int>> out_idx(1);
    std::vector<std::vector<float>> out_val(1);
    std::vector<std::vector<int>> row_indices_chunks{{0,1,2}};
    std::vector<std::vector<float>> row_values_chunks{{0.1f,0.2f,0.3f}};
    int nnz = 0;

_assign_spv_chunks_to_batch(batch,num_cpus,vec_assign_locs,out_idx,out_val,row_indices_chunks,row_values_chunks,nnz);

    REQUIRE_MESSAGE(nnz==3,"NNZ count incorrect");
    REQUIRE_MESSAGE(out_idx[0].size()==3,"Output idx size incorrect");
    REQUIRE_MESSAGE(out_val[0].size()==3,"Output val size incorrect");

    for(size_t i=0;i<out_idx[0].size();++i){
        REQUIRE_MESSAGE(i==out_idx[0][i],"Idx mismatch at position "+std::to_string(i));
    }

    for(size_t i=0;i<out_val[0].size();++i){
        REQUIRE_MESSAGE(std::abs(out_val[0][i]-i*0.1f)<0.001f,"Val mismatch at position "+std::to_string(i));
    }
}










TEST_CASE("Linear Backward SIDLSO Test Suite") {
    SECTION("Test Linear Backward SIDLSO with valid inputs") {
        int batch_size = 10;
        int input_size = 20;
        int output_size = 30;
        csr_struct input_csr;
        float*W = new float[input_size *output_size];

        // Initialize W with random values
        for (int i = 0; i < input_size* output_size; ++i) {
            W[i] = static_cast<float>(rand() % 100) / 100.0f;
        }

        csr_struct output_grad_csr;
        csr_struct I_grad;

        std::shared_ptr<WeightGradUpdater> updater(new WeightGradUpdater(input_size, output_size));
        auto W_grad_callback = get_dense_W_grad_callback(updater);

        linear_backward_sidlso(batch_size, input_size, output_size, input_csr, W, output_grad_csr, I_grad, W_grad_callback);

        // Verify that the weights have been updated correctly
        for (size_t i = 0; i < input_size *output_size; ++i) {
            REQUIRE_MESSAGE(updater->w_grad[i] != 0.f, "Gradient at position [" << i << "] is zero");
        }
    }

    SECTION("Test Linear Backward SIDLSO with invalid inputs") {
        int batch_size = 0;
        int input_size = 20;
        int output_size = 30;
        csr_struct input_csr;
        float*W = new float[input_size*output_size];

        // Initialize W with random values
        for (int i = 0; i < input_size*output_size; ++i) {
            W[i] = static_cast<float>(rand() % 100) / 100.0f;
        }

        csr_struct output_grad_csr;
        csr_struct I_grad;

        std::shared_ptr<WeightGradUpdater> updater(new WeightGradUpdater(input_size, output_size));
        auto W_grad_callback = get_dense_W_grad_callback(updater);

        linear_backward_sidlso(batch_size, input_size, output_size, input_csr, W, output_grad_csr, I_grad, W_grad_callback);

        // Verify that the weights have been updated correctly
        for (size_t i = 0; i < input_size*output_size; ++i) {
            REQUIRE_MESSAGE(updater->w_grad[i] != 0.f, "Gradient at position [" << i << "] is zero");
        }
    }

    SECTION("Test Linear Backward SIDLSO edge case") {
        int batch_size = 1;
        int input_size = 1;
        int output_size = 1;
        csr_struct input_csr;
        float*W = new float[input_size *output_size];

        // Initialize W with random value
        W[0] = static_cast<float>(rand() % 100) / 100.0f;

        csr_struct output_grad_csr;
        csr_struct I_grad;

        std::shared_ptr<WeightGradUpdater> updater(new WeightGradUpdater(input_size, output_size));
        auto W_grad_callback = get_dense_W_grad_callback(updater);

        linear_backward_sidlso(batch_size, input_size, output_size, input_csr, W, output_grad_csr, I_grad, W_grad_callback);

        // Verify that the weights have been updated correctly
        for (size_t i = 0; i < input_size* output_size; ++i) {
            REQUIRE_MESSAGE(updater->w_grad[i] != 0.f, "Gradient at position [" << i << "] is zero");
        }
    }
}


TEST_CASE("CSRMASK Class Tests") {
    SECTION("Add Random Value Test") {
        csr_struct csr_matrix;
        CSRMask csr_mask(csr_matrix);

        // Add some initial values to the CSR matrix
        csr_matrix.nnz = 5;
        csr_matrix.values = new float[csr_matrix.nnz]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        csr_matrix.indices = new int[csr_matrix.nnz]{0, 1, 2, 3, 4};

        // Call addRandomValue with default parameters
        csr_mask.addRandomValue();

        // Check if values have changed
        for (int i = 0; i < csr_matrix.nnz; ++i) {
            REQUIRE_MESSAGE(csr_matrix.values[i] != 1.0f + i, "Values did not change after addRandomValue call");
        }
    }

    SECTION("Remove Element Test") {
        csr_struct csr_matrix;
        CSRMask csr_mask(csr_matrix);

        // Add some initial values to the CSR matrix
        csr_matrix.nnz = 5;
        csr_matrix.values = new float[csr_matrix.nnz]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        csr_matrix.indices = new int[csr_matrix.nnz]{0, 1, 2, 3, 4};

        // Remove an element at index 2
        csr_mask.removeElement(2);

        // Check if element was removed successfully
        REQUIRE_MESSAGE(csr_matrix.nnz == 4, "NNZ count did not decrease after removing element");
        REQUIRE_MESSAGE(csr_matrix.values[2] != 3.0f, "Removed element still present in values array");
        REQUIRE_MESSAGE(csr_matrix.indices[2] != 2, "Removed element still present in indices array");
    }

    SECTION("Add Random Elements Test") {
        csr_struct csr_matrix;
        CSRMask csr_mask(csr_matrix);

        // Add some initial values to the CSR matrix
        csr_matrix.nnz = 5;
        csr_matrix.values = new float[csr_matrix.nnz]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        csr_matrix.indices = new int[csr_matrix.nnz]{0, 1, 2, 3, 4};

        // Insert two new elements randomly
        csr_mask.addRandomElements(2);

        // Check if NNZ count increased
        REQUIRE_MESSAGE(csr_matrix.nnz == 7, "NNZ count did not increase after inserting elements");
        REQUIRE_MESSAGE(csr_matrix._reserved_indices_and_values >5, "csr_struct didn't reserve space for new elements");
    }
}
