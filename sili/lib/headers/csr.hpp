#ifndef __CSR__HPP_
#define __CSR__HPP_

#include "sparse_struct.hpp"

#include "coo.hpp"
#include "unique_vector.hpp"
#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iterator>
#include <omp.h>
#include <random>

#include <memory>
#include <array>
#include <type_traits>

const size_t min_work_per_thread = 500;

template <typename SIZE_TYPE>
using CSRPtrs = std::array<std::unique_ptr<SIZE_TYPE[]>, 1>;

template <typename INDEX_ARRAYS>
struct ReduceArraySize {
    using type = std::array<
        typename INDEX_ARRAYS::value_type, 
        std::tuple_size<INDEX_ARRAYS>::value - 1
    >;
};

template <typename INDEX_ARRAYS>
using ReducedArray = typename ReduceArraySize<INDEX_ARRAYS>::type;

/*template <class SIZE_TYPE, class VALUE_TYPE>
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
}*/

template <typename INDEX_ARRAYS>
using stdarr_of_uniqarr_type = typename std::remove_extent<
    typename std::remove_pointer<
        typename std::tuple_element<0, INDEX_ARRAYS>::type::element_type
    >::type
>::type;

template <typename INDEX_ARRAYS>
constexpr std::size_t num_indices = std::tuple_size<INDEX_ARRAYS>::value;

//warning: the original csr also has the same pointers after this operation.
template <typename SIZE_TYPE, typename INDEX_ARRAYS, typename VALUE_ARRAYS>
sparse_struct<
    SIZE_TYPE,
    COOPointers<stdarr_of_uniqarr_type<INDEX_ARRAYS>>, 
    std::array<std::unique_ptr<stdarr_of_uniqarr_type<INDEX_ARRAYS>[]>, num_indices<INDEX_ARRAYS>+1 >,
    VALUE_ARRAYS
    > to_coo(
        sparse_struct<SIZE_TYPE,CSRPointers<stdarr_of_uniqarr_type<INDEX_ARRAYS>>, INDEX_ARRAYS, VALUE_ARRAYS> &a_csr, 
        const int num_cpus)
{
    SIZE_TYPE nnz = a_csr.nnz();
    stdarr_of_uniqarr_type<INDEX_ARRAYS>* rows = new stdarr_of_uniqarr_type<INDEX_ARRAYS>[nnz];

    #pragma omp parallel num_threads(num_cpus)
    {
        SIZE_TYPE tid = omp_get_thread_num(); // Thread ID

        SIZE_TYPE chunk_size = (nnz + num_cpus - 1) / num_cpus; // Split insertions among threads
        SIZE_TYPE start = tid * chunk_size;
        SIZE_TYPE end = std::min(start + chunk_size, nnz); // Calculate the end index for this thread

        SIZE_TYPE row = static_cast<SIZE_TYPE>(
            std::max(
                static_cast<std::ptrdiff_t>(0), 
                std::upper_bound
                (
                    a_csr.ptrs[0].get(), 
                    a_csr.ptrs[0].get() + a_csr.rows, start) - a_csr.ptrs[0].get()-1
                )
            );

        SIZE_TYPE iter = start;
        while (iter < end) {
            if (iter >= a_csr.ptrs[0][row + 1]) {
                row++;
            }
            rows[iter] = row;
            iter++;
        }
    }

    sparse_struct<
    SIZE_TYPE,
    COOPointers<stdarr_of_uniqarr_type<INDEX_ARRAYS>>, 
    std::array<std::unique_ptr<stdarr_of_uniqarr_type<INDEX_ARRAYS>[]>, num_indices<INDEX_ARRAYS>+1 >,
    VALUE_ARRAYS
    > coo;
    coo.rows = a_csr.rows;
    coo.cols = a_csr.cols;
    coo.ptrs = nnz;
    coo.indices[0].reset(rows);
    for (std::size_t idx = 0; idx < num_indices<INDEX_ARRAYS>; ++idx) {
        coo.indices[idx+1].reset(a_csr.indices[idx].release());
    }
    for (std::size_t valIdx = 0; valIdx < num_indices<VALUE_ARRAYS>; ++valIdx) {
        coo.values[valIdx].reset(a_csr.values[valIdx].release());
    }

    return coo;
}

//warning: the original csr also has the same pointers after this operation.
//warning2: the input coo MUST be coalesced: it must be sorted and have no duplicates.
template <typename SIZE_TYPE, typename INDEX_ARRAYS, typename VALUE_ARRAYS>
sparse_struct<
    SIZE_TYPE,
    CSRPtrs<SIZE_TYPE>, // First SIZE_TYPE transformed to CSRPtrs
    ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
    VALUE_ARRAYS
> to_csr(
    sparse_struct<
        SIZE_TYPE,
        COOPointers<SIZE_TYPE>, // First SIZE_TYPE is unchanged here
        INDEX_ARRAYS,           // INDEX_ARRAYS as provided
        VALUE_ARRAYS
    > &a_coo, 
    const int num_cpus)
{
    SIZE_TYPE nnz = a_coo.ptrs;
    SIZE_TYPE num_rows = a_coo.rows;
    auto rows = a_coo.indices[0].get();

    // Parallel section to determine min_row and max_row for each thread
    SIZE_TYPE *thread_min_row = new SIZE_TYPE[num_cpus];
    SIZE_TYPE *thread_max_row = new SIZE_TYPE[num_cpus + 1];

    #pragma omp parallel num_threads(num_cpus)
    {
        SIZE_TYPE tid = omp_get_thread_num();
        SIZE_TYPE chunk_size = (nnz + num_cpus - 1) / num_cpus;
        SIZE_TYPE start = tid * chunk_size;
        SIZE_TYPE end = std::min(start + chunk_size, nnz);

        thread_min_row[tid] = rows[start];
        thread_max_row[tid] = rows[end - 1] + 1;
    }

    // Allocate accumulators for parallel histogram accumulation
    SIZE_TYPE *accum = new SIZE_TYPE[a_coo.rows]();
    if (num_cpus > 1) {
        SIZE_TYPE *thr_accum = new SIZE_TYPE[num_cpus * a_coo.rows];
        std::fill(thr_accum, thr_accum + num_cpus * a_coo.rows, 0);

        #pragma omp parallel shared(accum, thr_accum, rows) num_threads(num_cpus)
        {
            int thread = omp_get_thread_num();
            int my_first = thread * a_coo.rows;

            SIZE_TYPE tid = omp_get_thread_num();
            SIZE_TYPE chunk_size = (nnz + num_cpus - 1) / num_cpus;
            SIZE_TYPE start = tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, nnz);

            for (SIZE_TYPE i = start; i < end; i++) {
                thr_accum[my_first + rows[i]]++;
            }

            #pragma omp for
            for (SIZE_TYPE r = 0; r < a_coo.rows; r++) {
                for (int t = 0; t < num_cpus; t++) {
                    accum[r] += thr_accum[t * a_coo.rows + r];
                }
            }
        }

        delete[] thr_accum;
    } else {
        for (SIZE_TYPE i = 0; i < nnz; i++) {
            accum[rows[i]]++;
        }
    }

    SIZE_TYPE *ptrs = new SIZE_TYPE[num_rows + 1];
    SIZE_TYPE scan_a = 0;

    // Parallel scan to compute row pointers
    #pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i <= num_rows; i++) {
        ptrs[i] = scan_a;
        #pragma omp scan exclusive(scan_a)
        {
            scan_a += accum[i];
        }
    }

    delete[] accum;

    // Create and return the CSR sparse structure
    sparse_struct<
    SIZE_TYPE,
    CSRPtrs<SIZE_TYPE>, // First SIZE_TYPE transformed to CSRPtrs
    ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
    VALUE_ARRAYS
    > csr;

    csr.rows = num_rows;
    csr.cols = a_coo.cols;
    csr.ptrs[0].reset(ptrs);
    //std::get<0>(csr.indices).reset(std::get<1>(a_coo.indices).release());
    for (std::size_t idx = 0; idx < num_indices<INDEX_ARRAYS>-1; ++idx) {
        csr.indices[idx].reset(a_coo.indices[idx+1].release());
    }
    for (std::size_t valIdx = 0; valIdx < num_indices<VALUE_ARRAYS>; ++valIdx) {
        csr.values[valIdx].reset(a_coo.values[valIdx].release());
    }

    return csr;
}

template <class SIZE_TYPE, class PTRS, class INDICES, class VALUES>
void clear_csr(sparse_struct<SIZE_TYPE, PTRS, INDICES, VALUES>& csr) {
    // Clear pointers array
    for (auto& ptr : csr.ptrs) {
        ptr.reset();
    }

    // Clear indices array
    for (auto& index : csr.indices) {
        index.reset();
    }

    // Clear values array
    for (auto& value : csr.values) {
        value.reset();
    }

    // Set rows and columns to zero
    csr.rows = 0;
    csr.cols = 0;
}

// merges two CSRs
/*template <typename SIZE_TYPE, typename VALUE_ARRAYS>
sparse_struct<SIZE_TYPE,CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, VALUE_ARRAYS> merge_csrs(
    const sparse_struct<SIZE_TYPE,CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, VALUE_ARRAYS> &a_csr,
    const sparse_struct<SIZE_TYPE,CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, VALUE_ARRAYS> &b_csr,
    const int num_cpus) {
    SIZE_TYPE total_nnz = a_csr.nnz() + b_csr.nnz();

    auto& max_csr = std::max(a_csr.nnz(), b_csr.nnz())==a_csr.nnz()?a_csr:b_csr;
    auto& min_csr = std::max(a_csr.nnz(), b_csr.nnz())==a_csr.nnz()?b_csr:a_csr;  //guarantee opposite even if equal

    SIZE_TYPE max_rows = std::max(a_csr.rows, b_csr.rows);
    SIZE_TYPE max_cols = std::max(a_csr.cols, b_csr.cols);

    // Arrays for row and column positions
    SIZE_TYPE *rows = new SIZE_TYPE[total_nnz];
    SIZE_TYPE *cols = new SIZE_TYPE[total_nnz];
    VALUE_ARRAYS vals;
    
    constexpr std::size_t numArrays = std::tuple_size<VALUE_ARRAYS>::value;
    using VALUE_TYPE = VALUE_ARRAYS::value_type;
    for (std::size_t i = 0; i < numArrays; ++i) {
        vals[i] = std::unique_ptr<VALUE_TYPE[]>(new VALUE_TYPE[total_nnz]);
    }

    SIZE_TYPE duplicates;

#pragma omp parallel num_threads(num_cpus)
    {
        SIZE_TYPE tid = omp_get_thread_num(); // Thread ID

        SIZE_TYPE chunk_size = (max_csr.nnz() + num_cpus - 1) / num_cpus; // Split insertions among threads
        SIZE_TYPE start = tid * chunk_size;
        SIZE_TYPE end_max = std::min(start + chunk_size, max_csr.nnz()); // Calculate the end index for this thread
        SIZE_TYPE end_min = std::min(start + chunk_size, min_csr.nnz()); // Calculate the end index for this thread

        //SIZE_TYPE row = std::max(0, std::upper_bound(a_csr.ptrs.get(), a_csr.ptrs.get() + a_csr.rows, start)-a_csr.ptrs.get()));

        SIZE_TYPE row = static_cast<SIZE_TYPE>(
            std::max(
                static_cast<std::ptrdiff_t>(0), 
                std::upper_bound
                (
                    min_csr.ptrs.get(), 
                    min_csr.ptrs.get() + min_csr.rows, start) - min_csr.ptrs.get()-1
                )
            );

        SIZE_TYPE iter = start;
        while (iter < end_min) {
            if (iter >= min_csr.ptrs[row + 1]) {
                row++;
            }
            rows[iter] = row;
            cols[iter] = min_csr.indices[iter];
            for (std::size_t i = 0; i < numArrays; ++i) {
                vals[i][iter] = min_csr.values[i][iter];
            }
            iter++;
        }

        row = static_cast<SIZE_TYPE>(std::max(static_cast<std::ptrdiff_t>(0), 
        std::upper_bound(max_csr.ptrs.get(), max_csr.ptrs.get() + max_csr.rows, start) - max_csr.ptrs.get()-1));

        iter = start;
        while (iter < end_max) {
            if (iter >= max_csr.ptrs[row + 1]) {
                row++;
            }
            rows[iter + min_csr.nnz()] = row;
            cols[iter + min_csr.nnz()] = max_csr.indices[iter];
            for (std::size_t i = 0; i < numArrays; ++i) {
                vals[i][iter + min_csr.nnz()] = min_csr.values[i][iter];
            }
            iter++;
        }

#pragma omp barrier

        #pragma omp single
        duplicates = recursive_merge_sort_coo(cols, rows, vals, (SIZE_TYPE)0, SIZE_TYPE(total_nnz - 1), (SIZE_TYPE)0);
    }

    // duplicated code start: move this out
    SIZE_TYPE *accum = new SIZE_TYPE[max_rows];
    std::fill(accum,accum+max_rows,0);
//accumulate parallel
    //thx: https://stackoverflow.com/a/70625541
    if(num_cpus>1){
    SIZE_TYPE *thr_accum = new SIZE_TYPE[num_cpus*(max_rows)];
    std::fill(thr_accum, thr_accum + max_rows*num_cpus, 0);
    #pragma omp parallel shared(accum, thr_accum, rows) num_threads(num_cpus)
  {
    int thread = omp_get_thread_num(),
      myfirst = thread*(max_rows);
    #pragma omp for
    for ( int i=0; i<total_nnz-duplicates; i++ )
      thr_accum[ myfirst+rows[i] ]++;
    #pragma omp for
    for ( int igrp=0; igrp<(max_rows); igrp++ )
      for ( int t=0; t<num_cpus; t++ )
        accum[igrp] += thr_accum[ t*(max_rows)+igrp ];
  }
    }else{
        for ( int i=0; i<total_nnz-duplicates; i++ )
            accum[ rows[i] ]++;
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
    return_csr.values.reset(vals);

    return return_csr;

    // duplicated code end
}*/

/*template<class SIZE_TYPE>
inline int get_col(int ptr, int row_start_ptr, SIZE_TYPE* indices, int cols, int end){
        if(ptr<0){
            return -1;
        }else if(ptr>=(end-row_start_ptr)){
            return cols;
        }else{
            return indices[row_start_ptr+ptr];
        }
}

// generate random csr of points, taking in avoid_pts for cumulative random csr generation
template <class SIZE_TYPE, class VALUE_TYPE>
csr_struct<SIZE_TYPE, VALUE_TYPE> generate_random_csr(
    SIZE_TYPE insertions,
    const csr_struct<SIZE_TYPE, VALUE_TYPE> &csr_avoid_pts,
    std::uniform_int_distribution<SIZE_TYPE> &index_dist, // should be from 0 to rows*cols
    std::mt19937_64 &generator,
    int num_cpus,
    unsigned local_seed=static_cast<unsigned long>(std::time(0))) {
    SIZE_TYPE total_elements = csr_avoid_pts.rows * csr_avoid_pts.cols;
    SIZE_TYPE total_nnz = csr_avoid_pts.nnz();
    SIZE_TYPE inverse_sparse_size = total_elements - total_nnz;

    // avoid infinite while loops later
    if (insertions > inverse_sparse_size) {
        insertions = inverse_sparse_size;
    }
    if(num_cpus>insertions){
        num_cpus=insertions; // without this, insertions are restricted to small chunks
    }

    // Arrays for row and column positions
    SIZE_TYPE *rows = new SIZE_TYPE[insertions];
    SIZE_TYPE *cols = new SIZE_TYPE[insertions];

// Parallel section for random position generation
#pragma omp parallel num_threads(num_cpus) shared(rows, cols)
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
        unsigned long seed = tid+local_seed;
        std::mt19937_64 thread_local_gen(seed); //nearby values cannot be the same
        SIZE_TYPE random_inverse_pos = 0;
        SIZE_TYPE remaining_space = inv_sp_end - inv_sp_start;

        for (SIZE_TYPE current_insertion = start; current_insertion < end; ++current_insertion) {
            SIZE_TYPE remaining_insertions = end - current_insertion;
            SIZE_TYPE max_step = std::max((SIZE_TYPE)1, remaining_space / remaining_insertions);
            SIZE_TYPE step = (index_dist(thread_local_gen) % max_step);
            random_inverse_pos += step+1;
            remaining_space -= step+1;

            // Binary search to find the corresponding row for the random_inverse_pos
            int low = 0, high = csr_avoid_pts.rows;
            SIZE_TYPE remaining = random_inverse_pos-1 + inv_sp_start;

            //allow zero on this step, but ensure we move forward in that case
            //random_inverse_pos += 1;
            //remaining_space -= 1;

            int mid;
            while (low <= high) {
                mid = (low + high) / 2;
                SIZE_TYPE elements_before = csr_avoid_pts.cols * (mid-low) - (csr_avoid_pts.ptrs[mid] -
                                                                      csr_avoid_pts.ptrs[low]);
                SIZE_TYPE elements_at = csr_avoid_pts.cols - (csr_avoid_pts.ptrs[mid+1] -
                                                                      csr_avoid_pts.ptrs[mid]);

                if (elements_before+elements_at<= remaining) { // x is greater, ignore left half
                    low = mid+1;
                    remaining-=(elements_before+elements_at);
                } else if(elements_before> remaining){ // x is lesser, ignore right half
                    high = mid - 1;
                }else{ // x is in this row
                    remaining-=elements_before;
                    break;
                }
            }
            rows[current_insertion] = mid;

            //remaining+=1;

            // Now search within the found row to determine the column
            //FIX: you were searching between low to high indices. Should be searching 0 to max, and the ptrs are in between
            SIZE_TYPE row_start_ptr = csr_avoid_pts.ptrs[rows[current_insertion]];
            SIZE_TYPE end_ptr = csr_avoid_pts.ptrs[rows[current_insertion]+1];
            int low_ptr = -1,
                high_ptr = end_ptr-row_start_ptr;
            SIZE_TYPE mid_ptr;
            while (low_ptr < high_ptr) {
                mid_ptr = (low_ptr + high_ptr+1) / 2;
                int col1 = get_col(mid_ptr, row_start_ptr, csr_avoid_pts.indices.get(), csr_avoid_pts.cols, end_ptr);
                int col2 = get_col(low_ptr, row_start_ptr, csr_avoid_pts.indices.get(), csr_avoid_pts.cols, end_ptr);
                SIZE_TYPE cols_elements = (col1 - col2) - (mid_ptr - low_ptr); // Inverse of nnz so far
                
                if (remaining < cols_elements) {
                    high_ptr = mid_ptr-1;
                    if(low_ptr>=high_ptr){ 
                        mid_ptr--;//fix bad exit
                        //remaining+=1;
                    }
                } else if(remaining>=cols_elements) {
                    remaining -= cols_elements;
                    low_ptr = mid_ptr;
                }/*else{
                    remaining -= cols_elements;
                    break;
                }*/
            /*}
            SIZE_TYPE before_col = get_col(mid_ptr, row_start_ptr, csr_avoid_pts.indices.get(), csr_avoid_pts.cols, end_ptr);
            cols[current_insertion] = before_col + remaining+1;
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

    // Build the ptrs array based on the sorted rows and cols/*
    /*
    SIZE_TYPE *accum = new SIZE_TYPE[csr_avoid_pts.rows];
    VALUE_TYPE *values = new VALUE_TYPE[insertions];

    std::fill(accum, accum + csr_avoid_pts.rows, 0);

    //accumulate parallel
    //thx: https://stackoverflow.com/a/70625541
    if(num_cpus>1){
    SIZE_TYPE *thr_accum = new SIZE_TYPE[num_cpus*(csr_avoid_pts.rows)];
    std::fill(thr_accum, thr_accum + csr_avoid_pts.rows*num_cpus, 0);
    #pragma omp parallel shared(accum, thr_accum, rows) num_threads(num_cpus)
  {
    int thread = omp_get_thread_num(),
      myfirst = thread*(csr_avoid_pts.rows);
    #pragma omp for
    for ( int i=0; i<insertions; i++ )
      thr_accum[ myfirst+rows[i] ]++;
    #pragma omp for
    for ( int igrp=0; igrp<(csr_avoid_pts.rows); igrp++ )
      for ( int t=0; t<num_cpus; t++ )
        accum[igrp] += thr_accum[ t*(csr_avoid_pts.rows)+igrp ];
  }
    }else{
        for ( int i=0; i<insertions; i++ )
            accum[ rows[i] ]++;
    }

    SIZE_TYPE *ptrs = new SIZE_TYPE[csr_avoid_pts.rows + 1];
    std::fill(ptrs, ptrs + csr_avoid_pts.rows + 1, 0);
    SIZE_TYPE scan_a = 0;
#pragma omp parallel for simd reduction(inscan, + : scan_a) shared(ptrs)
    for (SIZE_TYPE i = 0; i < csr_avoid_pts.rows + 1; i++) {
        //#pragma omp atomic
        ptrs[i] += scan_a;
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
    std::mt19937_64 generator;                  // Random number generator
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
*/
/* #endregion */
#endif