#ifndef __CSR__HPP_
#define __CSR__HPP_

#include "coo.hpp"
#include "unique_vector.hpp"
#include <algorithm>
#include <cstddef>
#include <ctime>
#include <memory>
#include <omp.h>
#include <random>

const size_t min_work_per_thread = 500;

// some of the basic tests I wrote at this point failed to delete the raw pointers, so while there is some overhead for
// unique_ptr, I'm keeping it to stay sane.
//  Also there's basically no overhead anyway if you compile with -O3 or -O4, which is the intended option
template <class SIZE_TYPE, class VALUE_TYPE> struct csr_struct {
    std::unique_ptr<SIZE_TYPE[]> ptrs;
    std::unique_ptr<SIZE_TYPE[]> indices;
    std::unique_ptr<VALUE_TYPE[]> values;
    SIZE_TYPE rows;
    SIZE_TYPE cols;
    SIZE_TYPE _reserved_indices_and_values = 0;

    csr_struct()
        : ptrs(nullptr), indices(nullptr), values(nullptr), rows(0), cols(0), _reserved_indices_and_values(0) {}

    csr_struct(SIZE_TYPE *p, SIZE_TYPE *ind, VALUE_TYPE *val, SIZE_TYPE non_zero, SIZE_TYPE num_p, SIZE_TYPE max_idx, SIZE_TYPE reserved)
        : ptrs(p), indices(ind), values(val), rows(num_p), cols(max_idx), _reserved_indices_and_values(reserved) {}

    csr_struct(SIZE_TYPE *p, SIZE_TYPE *ind, VALUE_TYPE *val, SIZE_TYPE non_zero, SIZE_TYPE num_p, SIZE_TYPE max_idx)
        : ptrs(p), indices(ind), values(val), rows(num_p), cols(max_idx) {}

    SIZE_TYPE nnz() const { return (ptrs != nullptr) ? ptrs[rows] : 0; }
};

template <class SIZE_TYPE, class VALUE_TYPE>
csr_struct<SIZE_TYPE, VALUE_TYPE> convert_vov_to_csr(const sili::unique_vector<sili::unique_vector<SIZE_TYPE>> *indices,
                                                     const sili::unique_vector<sili::unique_vector<VALUE_TYPE>> *values,
                                                     SIZE_TYPE num_col,
                                                     SIZE_TYPE num_row,
                                                     SIZE_TYPE numNonZero) {
    // Allocate memory for the CSR format
    csr_struct<SIZE_TYPE, VALUE_TYPE> csr;
    csr.ptrs = std::make_unique<SIZE_TYPE[]>(num_col + 1);
    csr.indices = std::make_unique<SIZE_TYPE[]>(numNonZero);

    if (values != nullptr) {
        csr.values = std::make_unique<VALUE_TYPE[]>(numNonZero);
    } else {
        csr.values = nullptr;
    }

    SIZE_TYPE ptr = 0; // Initialize the pointers for the flattened arrays

    for (SIZE_TYPE row = 0; row < num_row; row++) {
        csr.ptrs[row] = ptr;

        const sili::unique_vector<SIZE_TYPE> &col_idx = (*indices)[row];
        ptr += col_idx.size();

        if (ptr > numNonZero) {
            throw std::runtime_error(
                "Actual number of non-zero elements exceeds the expected number used to reserve memory for arrays. "
                "This will lead to double free/corruption errors and segfaults.");
        }

        // Flatten the row_indices and values vectors for this column
        std::copy(col_idx.begin(), col_idx.end(), csr.indices.get() + csr.ptrs[row]);

        if (values != nullptr) {
            const sili::unique_vector<VALUE_TYPE> &val = (*values)[row];
            std::copy(val.begin(), val.end(), csr.values.get() + csr.ptrs[row]);
        }
    }

    // Update the last column pointer for crow_indices
    csr.ptrs[num_row] = ptr;

    // Create the CSR struct
    csr.rows = num_row;
    csr.cols = num_col;

    return csr;
}

// merges two CSRs, assuming no overlapping positions
template <typename SIZE_TYPE, typename VALUE_TYPE>
csr_struct<SIZE_TYPE, VALUE_TYPE> merge_csrs(const csr_struct<SIZE_TYPE, VALUE_TYPE> &a_csr,
                                             const csr_struct<SIZE_TYPE, VALUE_TYPE> &b_csr,
                                             const int num_cpus) {
    SIZE_TYPE total_nnz = a_csr.nnz() + b_csr.nnz();

    SIZE_TYPE max_nnz = std::max(a_csr.nnz(), b_csr.nnz());
    SIZE_TYPE max_rows = std::max(a_csr.rows, b_csr.rows);
    SIZE_TYPE max_cols = std::max(a_csr.cols, b_csr.cols);

    // Arrays for row and column positions
    SIZE_TYPE *rows = new SIZE_TYPE[total_nnz];
    SIZE_TYPE *cols = new SIZE_TYPE[total_nnz];
    SIZE_TYPE *vals = new SIZE_TYPE[total_nnz];

#pragma omp parallel num_threads(num_cpus)
    {
        SIZE_TYPE tid = omp_get_thread_num(); // Thread ID

        SIZE_TYPE chunk_size = (a_csr.nnz() + num_cpus - 1) / num_cpus; // Split insertions among threads
        SIZE_TYPE start = tid * chunk_size;
        SIZE_TYPE end = std::min(start + chunk_size, a_csr.nnz()); // Calculate the end index for this thread
        SIZE_TYPE row = std::max(0, std::upper_bound(a_csr.ptrs.get(), a_csr.ptrs.get() + a_csr.rows, start) - 1);

        SIZE_TYPE iter = start;
        while (iter < end) {
            if (iter >= a_csr.ptrs[row + 1]) {
                row++;
            }
            rows[iter] = row;
            cols[iter] = a_csr.indices[iter];
            vals[iter] = a_csr.values[iter];
            iter++;
        }

        chunk_size = (b_csr.nnz() + num_cpus - 1) / num_cpus; // Split insertions among threads
        start = tid * chunk_size;
        end = std::min(start + chunk_size, b_csr.nnz()); // Calculate the end index for this thread
        row = std::max(0, std::upper_bound(b_csr.ptrs.get(), b_csr.ptrs.get() + b_csr.rows, start) - 1);

        iter = start;
        while (iter < end) {
            if (iter >= a_csr.ptrs[row + 1]) {
                row++;
            }
            rows[iter + a_csr.nnz()] = row;
            cols[iter + a_csr.nnz()] = b_csr.indices[iter];
            vals[iter + a_csr.nnz()] = b_csr.values[iter];
            iter++;
        }

#pragma omp barrier

        recursive_merge_sort_coo(rows, cols, vals, 0, max_nnz - 1);
    }

    // duplicated code start: move this out
    SIZE_TYPE *accum = new SIZE_TYPE[max_rows + 1];
#pragma omp parallel for simd num_threads(num_cpus) reduction(inscan, + : ptrs[csrMatrix.rows + 1])
    for (SIZE_TYPE i = 0; i < max_nnz; ++i) {
        // values[i] = 0;  // Initialize values as 0
        accum[rows[i] + 1]++;
    }

    SIZE_TYPE *ptrs = new SIZE_TYPE[max_rows + 1];
    SIZE_TYPE scan_a = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i < max_rows + 1; i++) {
        ptrs[i] = scan_a;
#pragma omp scan exclusive(scan_a)
        { scan_a += accum[i]; }
    }

    delete[] accum;
    delete[] rows;

    // Return the new CSR matrix
    csr_struct<SIZE_TYPE, VALUE_TYPE> return_csr;
    return_csr.rows = max_rows;
    return_csr.cols = max_cols;
    return_csr.ptrs.reset(ptrs);
    return_csr.indices.reset(cols);
    // random_csr.values.reset(values);

    return return_csr;

    // duplicated code end
}

// generate random csr of points, taking in avoid_pts for cumulative random csr generation
template <class SIZE_TYPE, class VALUE_TYPE>
csr_struct<SIZE_TYPE, VALUE_TYPE> generate_random_csr(
    SIZE_TYPE insertions,
    const csr_struct<SIZE_TYPE, VALUE_TYPE> &csr_avoid_pts,
    std::uniform_int_distribution<SIZE_TYPE> &index_dist, // should be from 0 to rows*cols
    std::default_random_engine &generator,
    const int num_cpus) {
    SIZE_TYPE total_elements = csr_avoid_pts.rows * csr_avoid_pts.cols;
    SIZE_TYPE total_nnz = csr_avoid_pts.nnz();
    SIZE_TYPE inverse_sparse_size = total_elements - total_nnz;

    // avoid infinite while loops later
    if (inverse_sparse_size > insertions) {
        insertions = inverse_sparse_size;
    }

    // Arrays for row and column positions
    SIZE_TYPE *rows = new SIZE_TYPE[insertions];
    SIZE_TYPE *cols = new SIZE_TYPE[insertions];

// Parallel section for random position generation
#pragma omp parallel num_threads(num_cpus)
    {
        SIZE_TYPE tid = omp_get_thread_num(); // Thread ID

        SIZE_TYPE chunk_size = (insertions + num_cpus - 1) / num_cpus; // Split insertions among threads
        SIZE_TYPE start = tid * chunk_size;
        SIZE_TYPE end = std::min(start + chunk_size, insertions); // Calculate the end index for this thread

        SIZE_TYPE inv_sp_chunk_size = (inverse_sparse_size + num_cpus - 1) / num_cpus; // Split insertions among threads
        SIZE_TYPE inv_sp_start = tid * inv_sp_chunk_size;
        SIZE_TYPE inv_sp_end =
            std::min(inv_sp_start + inv_sp_chunk_size, inverse_sparse_size); // Calculate the end index for this thread

        // Thread-local random generator
        std::default_random_engine thread_local_gen = generator;
        thread_local_gen.seed(omp_get_thread_num());

        SIZE_TYPE random_inverse_pos = start;

        for (SIZE_TYPE current_insertion = start; current_insertion < end; ++current_insertion) {
            random_inverse_pos += index_dist(thread_local_gen) %
                                  ((inv_sp_chunk_size - random_inverse_pos) / (chunk_size - current_insertion));

            // Binary search to find the corresponding row for the random_inverse_pos
            SIZE_TYPE low = 0, high = csr_avoid_pts.rows;
            SIZE_TYPE remaining = random_inverse_pos + inv_sp_start;

            while (low < high) {
                SIZE_TYPE mid = (low + high) / 2;
                SIZE_TYPE rows_elements = csr_avoid_pts.cols * mid - (csr_avoid_pts.ptrs[mid + 1] -
                                                                      csr_avoid_pts.ptrs[low]); // Inverse of nnz so far

                if (remaining < rows_elements) {
                    high = mid;
                } else {
                    remaining -= rows_elements;
                    low = mid + 1;
                }
            }
            rows[current_insertion] = low;

            // Now search within the found row to determine the column
            SIZE_TYPE low_ptr = csr_avoid_pts.ptrs[rows[current_insertion]],
                      high_ptr = csr_avoid_pts.ptrs[rows[current_insertion] + 1];
            while (low_ptr < high_ptr) {
                SIZE_TYPE mid_ptr = (low_ptr + high_ptr) / 2;
                SIZE_TYPE cols_elements = csr_avoid_pts.indices[mid_ptr] - (mid_ptr - low_ptr); // Inverse of nnz so far

                if (remaining < cols_elements) {
                    high_ptr = mid_ptr;
                } else {
                    remaining -= cols_elements;
                    low_ptr = mid_ptr + 1;
                }
            }
            SIZE_TYPE before_col = csr_avoid_pts.indices[low_ptr];
            cols[current_insertion] = before_col + remaining;
        }
    }

    // Sort rows and cols arrays first by row, then by column within rows
    // except they're already sorted.
    /*for (SIZE_TYPE i = 0; i < insertions - 1; ++i) {
        for (SIZE_TYPE j = i + 1; j < insertions; ++j) {
            if (rows[i] > rows[j] || (rows[i] == rows[j] && cols[i] > cols[j])) {
                std::swap(rows[i], rows[j]);
                std::swap(cols[i], cols[j]);
            }
        }
    }*/

    // Build the ptrs array based on the sorted rows and cols
    SIZE_TYPE *accum = new SIZE_TYPE[csr_avoid_pts.rows + 1];
    VALUE_TYPE *values = new VALUE_TYPE[insertions];

    std::fill(accum, accum + csr_avoid_pts.rows + 1, 0);

#pragma omp parallel for simd num_threads(num_cpus) reduction(inscan, + : ptrs[csrMatrix.rows + 1])
    for (SIZE_TYPE i = 0; i < insertions; ++i) {
        // values[i] = 0;  // Initialize values as 0
        accum[rows[i] + 1]++;
    }

    SIZE_TYPE *ptrs = new SIZE_TYPE[csr_avoid_pts.rows + 1];
    SIZE_TYPE scan_a = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i < csr_avoid_pts.rows + 1; i++) {
        ptrs[i] = scan_a;
#pragma omp scan exclusive(scan_a)
        { scan_a += accum[i]; }
    }

    delete[] accum;
    delete[] rows;

    // Return the new CSR matrix
    csr_struct<SIZE_TYPE, VALUE_TYPE> random_csr;
    random_csr.rows = csr_avoid_pts.rows;
    random_csr.cols = csr_avoid_pts.cols;
    random_csr.ptrs.reset(ptrs);
    random_csr.indices.reset(cols);
    // random_csr.values.reset(values);

    return random_csr;
}

// determine optimal num cpus parameter if not given.
template <class SIZE_TYPE, class VALUE_TYPE>
csr_struct<SIZE_TYPE, VALUE_TYPE> generate_random_csr(SIZE_TYPE insertions,
                                                      const csr_struct<SIZE_TYPE, VALUE_TYPE> &csrMatrix,
                                                      std::uniform_int_distribution<SIZE_TYPE> &index_dist,
                                                      std::default_random_engine &generator) {
    // Upper bound on insertions
    SIZE_TYPE total_elements = csrMatrix.rows * csrMatrix.cols;
    SIZE_TYPE nnz = csrMatrix.nnz();
    SIZE_TYPE max_insertions = total_elements - nnz;

    if (insertions > max_insertions) {
        insertions = max_insertions;
    }

    SIZE_TYPE optimal_cpus = std::min(omp_get_num_procs(), insertions / min_work_per_thread);

    // Call the original function with the optimized number of CPUs
    return generate_random_csr(insertions, csrMatrix, index_dist, generator, optimal_cpus);
}

// Helper method to remove an element from the CSR matrix
template <typename SIZE_TYPE, typename VALUE_TYPE>
void remove_element_from_csr(SIZE_TYPE index, csr_struct<SIZE_TYPE, VALUE_TYPE> &csrMatrix) {
    if (csrMatrix._reserved_indices_and_values < csrMatrix.nnz()) {
        csrMatrix._reserved_indices_and_values =
            csrMatrix.nnz(); // ensure this csr now contains the actual reserved size
    }
    // properly erase a value in the csr
    std::move(
        csrMatrix.values.get() + index + 1, csrMatrix.values.get() + csrMatrix.nnz(), csrMatrix.values.get() + index);
    std::move(csrMatrix.indices.get() + index + 1,
              csrMatrix.indices.get() + csrMatrix.nnz(),
              csrMatrix.indices.get() + index);

    // Update ptrs to reflect removal
    for (SIZE_TYPE i = 1; i < csrMatrix.rows + 1; ++i) {
        if (csrMatrix.ptrs[i] > index) {
            csrMatrix.ptrs[i]--;
        }
    }
}

// Method to add a small number of elements to CSR array with bisection insertion
template <typename SIZE_TYPE, typename VALUE_TYPE>
void add_few_random_to_csr(SIZE_TYPE insertions,
                           csr_struct<SIZE_TYPE, VALUE_TYPE> &csrMatrix,
                           std::uniform_int_distribution<SIZE_TYPE> &index_dist,
                           std::default_random_engine &generator) {
    // reserve needed space
    if (csrMatrix._reserved_indices_and_values < csrMatrix.nnz() + insertions) {
        SIZE_TYPE *old_indices = csrMatrix.indices.get();
        VALUE_TYPE *old_values = csrMatrix.values.get();
        csrMatrix.indices.release();
        csrMatrix.values.release();

        // Reserve enough space for the new insertions, doubling the current capacity
        SIZE_TYPE new_capacity = std::max(csrMatrix._reserved_indices_and_values * 2, csrMatrix.nnz() + insertions);
        csrMatrix.indices.reset(new SIZE_TYPE[new_capacity]);
        csrMatrix.values.reset(new VALUE_TYPE[new_capacity]);

        // Move existing data to the newly allocated arrays
        std::move(old_indices, old_indices + csrMatrix.nnz(), csrMatrix.indices.get());
        std::move(old_values, old_values + csrMatrix.nnz(), csrMatrix.values.get());

        // Cleanup old arrays
        delete[] old_indices;
        delete[] old_values;

        // Update reserved space tracker
        csrMatrix._reserved_indices_and_values = new_capacity;
    }

    for (SIZE_TYPE i = 0; i < insertions;
         ++i) { // no need to parallelize this. In general you should only be adding ~1-10 at a time at most
        SIZE_TYPE random_index = index_dist(generator);
        SIZE_TYPE remaining_space = csrMatrix.cols * csrMatrix.rows - csrMatrix.nnz();
        SIZE_TYPE pos = random_index % (csrMatrix.nnz() + 1);

        SIZE_TYPE insert_pos = std::distance(csrMatrix.indices.get(), csrMatrix.indices.get() + pos);
        SIZE_TYPE insert_row_max =
            std::distance(csrMatrix.ptrs.get(),
                          std::upper_bound(csrMatrix.ptrs.get(), csrMatrix.ptrs.get() + csrMatrix.rows, pos)) -
            1;
        SIZE_TYPE insert_row_min =
            std::distance(csrMatrix.ptrs.get(),
                          std::lower_bound(csrMatrix.ptrs.get(), csrMatrix.ptrs.get() + csrMatrix.rows, pos)) -
            1;
        SIZE_TYPE chosen_row;
        if (insert_row_max > insert_row_min) {
            chosen_row = index_dist(generator) % (insert_row_max - insert_row_min) + insert_row_min;
        } else {
            chosen_row = insert_row_min;
        }
        if (chosen_row < 0) {
            chosen_row = 0;
        }

        SIZE_TYPE index_before = 0;
        if (pos != 0 && csrMatrix.ptrs[chosen_row] != pos) {
            index_before = csrMatrix.indices[pos - 1];
        }
        SIZE_TYPE index_after = csrMatrix.cols;
        if (pos != csrMatrix.nnz() && csrMatrix.ptrs[chosen_row + 1] != pos) {
            index_after = csrMatrix.indices[pos];
        }
        if ((index_after - index_before <= 1) ||
            (pos == csrMatrix.nnz() && index_before == csrMatrix.cols &&
             csrMatrix.ptrs[csrMatrix.rows] != csrMatrix.ptrs[csrMatrix.rows - 1])) {
            // insertions++; may loop forever
            continue; // no space for current insertion point
        }

        std::move(
            csrMatrix.values.get() + pos, csrMatrix.values.get() + csrMatrix.nnz(), csrMatrix.values.get() + pos + 1);
        std::move(csrMatrix.indices.get() + pos,
                  csrMatrix.indices.get() + csrMatrix.nnz(),
                  csrMatrix.indices.get() + pos + 1);

        random_index = index_dist(generator) % (index_after - index_before - 1) + index_before + 1;

        csrMatrix.indices[pos] = random_index;
        csrMatrix.values[pos] = 0;

        for (SIZE_TYPE i = chosen_row + 1; i < csrMatrix.rows + 1; ++i) {
            // if (csrMatrix.ptrs[i] > insert_pos) {
            csrMatrix.ptrs[i]++;
            //}
        }
    }
}

// a random starmap, for backprop masking
// this allows bias values to become non-zero in zero-init models, and generally allows valid zero-init models to learn
// at all
template <class SIZE_TYPE, class VALUE_TYPE> class CSRStarmap {
  private:
    std::default_random_engine generator;                  // Random number generator
    std::uniform_real_distribution<VALUE_TYPE> value_dist; // Distribution for random values
    std::uniform_int_distribution<SIZE_TYPE> index_dist;   // Distribution for random indices

  public:
    csr_struct<SIZE_TYPE, VALUE_TYPE> csrMatrix;

    // Constructor to initialize CSR matrix handler
    CSRStarmap(csr_struct<SIZE_TYPE, VALUE_TYPE> &csr_matrix)
        : csrMatrix(csr_matrix), generator(static_cast<unsigned>(std::time(0))), value_dist(0.0f, std::numbers::pi * 2),
          index_dist(0, csr_matrix.rows * csr_matrix.cols - 1) {
        if (csrMatrix.ptrs == nullptr) {
            csrMatrix.ptrs.reset(new SIZE_TYPE[csrMatrix.rows + 1]{});
        }
    }

    void iterate(SIZE_TYPE nnz, VALUE_TYPE min = 0, VALUE_TYPE max = 2 * std::numbers::pi / 50000) {
        addRandomValue(min, max);                 // add small floats to every value
        addRandomElements(nnz - csrMatrix.nnz()); // maintain exactly nnz values by inserting 0s
    }

    // Method to add a small random value to each CSR value
    void addRandomValue(VALUE_TYPE min = 0, VALUE_TYPE max = 2 * std::numbers::pi / 50000) {
        std::uniform_real_distribution<VALUE_TYPE> small_value_dist(min, max);
        for (SIZE_TYPE i = 0; i < csrMatrix.nnz(); ++i) {
            csrMatrix.values[i] += small_value_dist(generator);
            if (csrMatrix.values[i] > std::numbers::pi * 2) {
                remove_element_from_csr(i, csrMatrix);
            }
        }
    }

    // Method to add elements to CSR array with bisection insertion
    void addRandomElements(SIZE_TYPE insertions) {
        if(insertions<20){
            //cheaper to shift to the right on every insert
            add_few_random_to_csr(insertions, csrMatrix, index_dist, generator);
        }else{
            //cheaper to throw everything into a pile and re-sort
            csr_struct<SIZE_TYPE, VALUE_TYPE> random_csr =
                generate_random_csr(insertions, csrMatrix, index_dist, generator);
            csrMatrix = merge_csrs(csrMatrix, random_csr);
        }
    }
};

/* #endregion */
#endif