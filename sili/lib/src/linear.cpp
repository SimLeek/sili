#include <omp.h>
#include <thread>
#include <algorithm>
#include <type_traits>

struct cs2_struct
{
    int* ptrs;
    int* indices;
    float* values;
    int numel;
    int num_ptrs;   // one less than size of ptrs. If csr: num_row.
    int max_index;  // max value possible in indices. If csr: num_col. -1 if
};

template <class T>
std::vector<size_t> fullScanSizes(const std::vector<std::vector<T>>& vec) {
    std::vector<size_t> fullScan(vec.size()+1);
    fullScan[0]=0;

    int scan_a = 0;
    #pragma omp simd reduction(inscan, +:scan_a)
    for(int i = 0; i < vec.size(); i++){
            fullScan[i+1] = scan_a;
            #pragma omp scan inclusive(scan_a)
            scan_a += vec[i].size();
    }

    return fullScan;
}

cs2_struct linear_sidlso(
    int batch_size,
    int input_size,
    int output_size,
    cs2_struct* input_csr,
    float* W,
    float eps=std::numeric_limits<float>::epsilon()
    ) {
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware threads

    std::vector<std::vector<int>> out_idx(batch_size);
    std::vector<std::vector<float>> out_val(batch_size);
    int nnz=0;

    // size 16 to start for cache line optimization
    std::vector<std::vector<int>> row_indices_chunks(num_cpus, std::vector<int>(16));
    std::vector<std::vector<int>> row_values_chunks(num_cpus, std::vector<int>(16));

    for(int batch =0; batch<batch_size;batch++){

        #pragma omp parallel num_threads(num_cpus)
        {
            int tid = omp_get_thread_num(); // Get thread ID
            int chunk_size = (output_size + num_cpus - 1) / num_cpus; // Calculate chunk size
            int start = tid * chunk_size; // Start index for this thread
            int end = std::min(start + chunk_size, output_size); // End index for this thread

            for (int output_index = start; output_index < end; ++output_index) {
                float out_val = 0;
                for (int input_ptr = input_csr.ptrs[batch]; input_ptr < input_csr.ptrs[batch + 1]; input_ptr++)
                {
                    int input_index = input_csr.indices[input_ptr];
                    auto input_value = input_csr.values[input_ptr];

                    out_val += W[output_index*input_size + input_index]*input_value;
                }
                if(out_val>eps){
                    row_indices_chunks[tid].push_back(output_index);
                    row_values_chunks[tid].push_back(out_val);
                }
            }
        }
        auto vec_assign_locs = fullScanSizes(row_indices_chunks);

        out_idx[batch].reserve(vec_assign_locs.back());
        out_val[batch].reserve(vec_assign_locs.back());
        nnz += vec_assign_locs[tid+1];

        #pragma omp parallel num_threads(num_cpus)
        {
            int tid = omp_get_thread_num(); // Get thread ID
            int start = vec_assign_locs[tid];
            //int end = vec_assign_locs[tid+1];

            std::copy(row_indices_chunks.begin(), row_indices_chunks.end(), out_idx[batch].begin() + start);
            std::copy(row_values_chunks.begin(), row_values_chunks.end(), out_val[batch].begin() + start);
        }
    }

    auto cs2 = convert_vov_to_cs2(&out_idx, &out_val, nullptr, output_size, batch_size, nnz);
    return cs2;
}

void linear_backward_sidlso(
    int batch_size,
    int input_size,
    int output_size,
    cs2_struct* input_csr,
    float* W,
    cs2_struct* output_grad_csr,
    cs2_struct* I_grad,  // Input: mask locations. Output: locations with values.
    std::function<void(float, int, int)> W_grad_callback,  // this will be called A LOT, so make it very fast. Without the callback, skiplists are needed, which will slow down things 10-30x no matter what, so do optimization and W modification here. Python functions are not recommended outside of testing.
    float eps=std::numeric_limits<float>::epsilon()
    ) {
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware threads

    for(int batch =0; batch<batch_size;batch++){

        #pragma omp parallel num_threads(num_cpus)
        {
            int o_grad_size = output_grad_csr.ptrs[batch+1] - output_grad_csr.ptrs[batch];
            int tid = omp_get_thread_num(); // Get thread ID
            int chunk_size = (o_grad_size + num_cpus - 1) / num_cpus; // Calculate chunk size
            int start = tid * o_grad_size + output_grad_csr.ptrs[batch]; // Start index for this thread
            int end = std::min(start + chunk_size, o_grad_size) + output_grad_csr.ptrs[batch]; // End index for this thread

            for (int ograd_p = start; ograd_p < end; ++ograd_p) {
                // learning inputs: selected neurons
                for (int input_ptr = I_grad.ptrs[batch]; input_ptr < I_grad.ptrs[batch + 1]; input_ptr++)
                {
                    int input_index = I_grad.indices[input_ptr];
                    int output_index = output_grad_csr.indices[ograd_p];
                    float output_value = output_grad_csr.values[ograd_p];
                    I_grad.values[input_ptr] += W[output_index*input_size + input_index] * output_value;
                }
                // learning synapses: due to math, only inputs that were active can be used
                for (int input_ptr = input_csr.ptrs[batch]; input_ptr < input_csr.ptrs[batch + 1]; input_ptr++)
                {
                    float out_wgrad = output_grad_csr.values[ograd_p]*I_grad.values[input_ptr];
                    if(out_wgrad>eps){
                        W_grad_callback(out_wgrad, output_grad_csr.indices[ograd_p], I_grad.indices[input_ptr]);
                        // example callbacks:
                        //  * set values in a grad_w array the size of w (uses 2x RAM min)
                        //  * (quantized) set values in a quantized grad_w_array with different min/max vals (uses 2X RAM min for quantized) and
                        //    * sequentially update a file with full weight/grad/optim values (sequential HDD access is much faster)
                        //      WARNING: use a HDD or RAID 10 near HDD array for this. It WILL destroy SSDs.
                        //  * (gpu) build up sparse CSR array with predefined max size using vectors before sending to CPU RAM
                        // other examples are much harder to implement and may or may not give good results:
                        //  * immediately use in optim or an optim callback to modify W
                        //  * (quantized) use in immediate optim if grad>threshold or grad*rand>threshold
                        //  * (quantized) set values in a quantized grad_w_array with different min/max vals (uses 2X RAM min for quantized) and
                        //    * set values if above threshold
                        //  * store the top x grad values
                        //  * use an unrolled skiplist to create a csr array of grad values for later use
                        //  * (quantized, weight values on HDD) use a parallel unrolled skiplist to store file update operations while the hard-drive takes its time in another thread
                    }
                }
            }
        }
    }
}

//this part was long but simple enough for ChatGPT
class CSRMask {
private:
    cs2_struct csrMatrix;
    std::default_random_engine generator;   // Random number generator
    std::uniform_real_distribution<float> value_dist; // Distribution for random values
    std::uniform_int_distribution<int> index_dist; // Distribution for random indices

public:
    // Constructor to initialize CSR matrix handler
    CSRMask(int rows, int cols, int non_zero_elements)
        : csrMatrix(rows, cols, non_zero_elements),
          generator(static_cast<unsigned>(std::time(0))),
          value_dist(0.0f, PI2), index_dist(0, rows * cols - 1) {}

    // Method to add a small random value to each CSR value
    void addRandomValue() {
        std::uniform_real_distribution<float> small_value_dist(0.0f, 2 * M_PI / 50000);
        for (int i = 0; i < csrMatrix.values.size(); ++i) {
            csrMatrix.values[i] += small_value_dist(generator);
            if (csrMatrix.values[i] > PI2) {
                csrMatrix.values[i] = 0.0f;
                removeElement(i);
            }
        }
    }

    // Helper method to remove an element from the CSR matrix
    void removeElement(int index) {
        csrMatrix.values.erase(csrMatrix.values.begin() + index);
        csrMatrix.indices.erase(csrMatrix.indices.begin() + index);

        // Update ptrs to reflect removal
        for (int i = 1; i < csrMatrix.num_ptrs; ++i) {
            if (csrMatrix.ptrs[i] > index) {
                csrMatrix.ptrs[i]--;
            }
        }
    }

    // Method to generate sparse CSR array with bisection insertion
    void generateBisectionSparseCSR(int rows, int cols, int nnz) {
        std::vector<int> row_counts(rows, 0);
        for (int i = 0; i < nnz; ++i) {  // no need to parallelize this. In general you should only be adding like 10 at a time at most
            int random_index = index_dist(generator);
            int row = random_index / (cols);
            int col = random_index % (cols);

            // Bisection insert into indices and values
            int start = csrMatrix.ptrs[row];
            int end = csrMatrix.ptrs[row + 1];
            auto pos = std::lower_bound(csrMatrix.indices.begin() + start, csrMatrix.indices.begin() + end, col);

            int insert_pos = std::distance(csrMatrix.indices.begin(), pos);
            csrMatrix.indices.insert(csrMatrix.indices.begin() + insert_pos, col);
            csrMatrix.values.insert(csrMatrix.values.begin() + insert_pos, value_dist(generator));
            row_counts[row]++;
        }

        // Fill ptrs with cumulative sum of row counts
        for (int i = 1; i < csrMatrix.num_ptrs; ++i) {
            csrMatrix.ptrs[i] = csrMatrix.ptrs[i - 1] + row_counts[i - 1];
        }
    }

    // Method to print the CSR matrix
    void printCSR() const {
        std::cout << "ptrs: ";
        for (int p : csrMatrix.ptrs) std::cout << p << " ";
        std::cout << "\nindices: ";
        for (int idx : csrMatrix.indices) std::cout << idx << " ";
        std::cout << "\nvalues: ";
        for (float val : csrMatrix.values) std::cout << val << " ";
        std::cout << std::endl;
    }
};