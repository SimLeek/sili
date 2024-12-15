/**
*@file coo.hpp
*@brief Header file containing functions for sorting and manipulating COO format data.

  ▗▄▄▖ ▗▄▖  ▗▄▖
 ▐▌   ▐▌ ▐▌▐▌ ▐▌
 ▐▌   ▐▌ ▐▌▐▌ ▐▌
 ▝▚▄▄▖▝▚▄▞▘▝▚▄▞▘

Stars, because they're sparse like COOs:
                                         .     o                     .  +
           '      .                          .      '   o                     o
       +           o                                    .   o .
                                   '       .-. '
.           *       '                       ) )         o              .
   '                                |      '-´           +  '
            +                      -o-             .            '         |
             .            .         |    o                         '    --o--
         .   +                               +                            | +
        o *                .                      ~~+                     *   '
                                      +                '
                                                                    . .     |
  +                   '    '        +   .                                 *-+-
           '                                          '                     |  +
                       .                                          *o.     +
     +                   .                       '+         .
                                                '      '   o
                       .                 +      .                      +   o
          *    .  +                   .          ~~+   .
                    +~~                                  '           +        *


*This header file provides various functions for sorting and manipulating COO format data,
*including top-level sorting functions, merging sorted COO matrices, and removing elements
*based on external values.
*
*The main functions provided by this file include:
* merge_sort_coo - Sorting COO format data using merge sort
*  Use this for sorting synapses before adding them to sorted sparse layers
* parallel_merge_sorted_coos - Merging two sorted COO matrices into a single sorted COO matrix using multi-threading
*  Use this for merging sparse layers with new synapses and learning
* coo_subtract_bottom_k - Removing bottom k elements based on external values and copying the result into another COO array in parallel
*  Use this for deleting the least important synapses to keep memory/processing within limits
* merge_sort_coo_external - Sorting a COO matrix based on an external array using merge sort
*
*These functions can be used for efficient sorting and manipulation of large COO format datasets.
*/
#ifndef __COO__HPP_
#define __COO__HPP_

/*
* merge_sort_coo - this is effectively the coalesce operation in pytorch
* parallel_merge_coo - for combining two COOs quickly, like stored synapses plus some additional random or sparse backprop synapses
* coo_subtract_bottom_k - for removing the bottom k importance COO elements in order to keep the COO array under a specific size.
*/

/**
* @brief Merges two sorted subarrays in COO format, removing duplicates.
*
* @tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
* @tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
*
* @param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
* @param vals Pointer to the value array.
* @param left Left boundary of the first subarray.
* @param mid Midpoint separating the two subarrays.
* @param right Right boundary of the second subarray.
*
* @return Number of duplicates removed during merging.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE>
SIZE_TYPE inplace_merge_coo(SIZE_TYPE *cols,
                            SIZE_TYPE *rows,
                            VALUE_TYPE *vals,
                            SIZE_TYPE left,
                            SIZE_TYPE mid,
                            SIZE_TYPE right) {
    SIZE_TYPE i = left;
    SIZE_TYPE j = mid + 1;
    SIZE_TYPE duplicates = 0;

    // Traverse both subarrays and merge in-place
    while (i <= mid && j <= right) {
        // If the element in the left half is smaller, move on
        if (rows[i] < rows[j] || (rows[i] == rows[j] && cols[i] < cols[j])) {
            i++;
        } else if (rows[i] == rows[j] && cols[i] == cols[j]) {
            // sum duplicates
            vals[i]+=vals[j];
            // Duplicate found, remove the element from the right subarray
            for (SIZE_TYPE k = j; k <= right; k++) {
                rows[k] = rows[k + 1];
                cols[k] = cols[k + 1];
                vals[k] = vals[k + 1];
            }
            duplicates++;
            #pragma omp atomic
            right--;
        } else {
            // Element in the right half is smaller, rotate to the left
            SIZE_TYPE temp_row = rows[j];
            SIZE_TYPE temp_col = cols[j];
            VALUE_TYPE temp_val = vals[j];

            // Shift elements from i to j-1 one position to the right
            for (SIZE_TYPE k = j; k > i; k--) {
                rows[k] = rows[k - 1];
                cols[k] = cols[k - 1];
                vals[k] = vals[k - 1];
            }

            // Place the j-th element in its correct position
            rows[i] = temp_row;
            cols[i] = temp_col;
            vals[i] = temp_val;

            // Update indices
            i++;
            mid++;
            j++;
        }
    }
    return duplicates;
}

/**
 *@brief Performs insertion sort on a subarray in COO format, removing duplicates.
*
 *@tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
* @tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
*
* @param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
* @param vals Pointer to the value array.
* @param left Left boundary of the subarray.
* @param right Right boundary of the subarray.
*
* @return Number of duplicates removed during sorting.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE>
SIZE_TYPE insertion_sort_coo(SIZE_TYPE *cols, SIZE_TYPE *rows, VALUE_TYPE *vals, SIZE_TYPE left, SIZE_TYPE right) {
    SIZE_TYPE duplicates = 0;
    for (unsigned long i = left + 1; i <= right; i++) {
        SIZE_TYPE temp_row = rows[i];
        SIZE_TYPE temp_col = cols[i];
        VALUE_TYPE temp_val = vals[i];
        SIZE_TYPE j = i;
        while (j > left && (rows[j - 1] > temp_row || (rows[j - 1] == temp_row && cols[j - 1] > temp_col))) {
            rows[j] = rows[j - 1];
            cols[j] = cols[j - 1];
            vals[j] = vals[j - 1];
            j--;
        }
        // Insert (temp_row, temp_col, temp_val) at its correct position
        rows[j] = temp_row;
        cols[j] = temp_col;
        vals[j] = temp_val;

        // Check for duplicates after insertion
        if (j > left && rows[j] == rows[j - 1] && cols[j] == cols[j - 1]) {
            // sum duplicates
            vals[j-1]+= vals[j]
            // Duplicate found, remove the current element
            for (SIZE_TYPE k = j; k < right; k++) {
                rows[k] = rows[k + 1];
                cols[k] = cols[k + 1];
                vals[k] = vals[k + 1];
            }
            duplicates++;
            #pragma omp atomic
            right--;
            i--;  // Adjust 'i' because the array size has decreased
        }
    }
    return duplicates;
}

/**
* @brief Recursively sorts COO format data using merge sort, removing duplicates.
*
* @tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
 *@tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
*
 *@param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
 *@param vals Pointer to the value array.
* @param left Left boundary of the subarray.
 *@param right Right boundary of the subarray.
* @param duplicates Accumulator for the total number of duplicates removed.
 *
* @return Total number of duplicates removed during sorting.
 */
template <typename SIZE_TYPE, typename VALUE_TYPE>
SIZE_TYPE recursive_merge_sort_coo(SIZE_TYPE *cols,
                                   SIZE_TYPE *rows,
                                   VALUE_TYPE *vals,
                                   SIZE_TYPE left,
                                   SIZE_TYPE right,
                                   SIZE_TYPE duplicates) {
    if (left < right) {
        if (right - left >= 32) {
            SIZE_TYPE mid = (left + right) / 2;
            SIZE_TYPE left_duplicates = 0;
            SIZE_TYPE right_duplicates = 0;
#pragma omp taskgroup
            {
#pragma omp task shared(cols, rows, vals, left_duplicates) untied if (right - left >= (1 << 14))
                left_duplicates= recursive_merge_sort_coo(cols, rows, vals, left, mid, (SIZE_TYPE)0);
#pragma omp task shared(cols, rows, vals, right_duplicates) untied if (right - left >= (1 << 14))
                right_duplicates= recursive_merge_sort_coo(cols, rows, vals, mid + 1, right, (SIZE_TYPE)0);
#pragma omp taskyield
            }

            // Adjust indices after handling duplicates
            if (left_duplicates > 0) {
                // Shift the right subarray to the left by left_duplicates
                for (SIZE_TYPE i = mid + 1; i <= right; i++) {
                    rows[i - left_duplicates] = rows[i];
                    cols[i - left_duplicates] = cols[i];
                    vals[i - left_duplicates] = vals[i];
                }
                mid -= left_duplicates;  // Adjust mid after the shift
                right -= left_duplicates;  // Adjust right boundary after the shift
            }

            if (right_duplicates > 0) {
                right -= right_duplicates;  // Adjust right for right-side duplicates
            }

            duplicates += inplace_merge_coo(cols, rows, vals, left, mid, right) + left_duplicates + right_duplicates;
        } else {
            duplicates += insertion_sort_coo(cols, rows, vals, left, right);
        }
    }
    return duplicates;
}

/**
* @brief Top-level function for sorting COO format data using merge sort.
 *
* @tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
* @tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
 *@tparam SIZE_TYPE_2 Type for the overall size of the matrix (e.g., unsigned int).
*
 *@param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
 *@param vals Pointer to the value array.
* @param size Overall size of the matrix.
 *
* @return Total number of duplicates removed during sorting.
 */
template <typename SIZE_TYPE, typename VALUE_TYPE, typename SIZE_TYPE_2>
SIZE_TYPE merge_sort_coo(SIZE_TYPE *cols, SIZE_TYPE *rows, VALUE_TYPE *vals, SIZE_TYPE_2 size) {
    int duplicates = 0;
#pragma omp parallel
#pragma omp single
    duplicates+=recursive_merge_sort_coo(cols, rows, vals, 0, (SIZE_TYPE)size - 1, 0);

    return duplicates;
}

#-------EXTERNAL SORT---------

//use these functions to sort the synapse COO by importance and then delete the least important for synaptic pruning
// then use the above sort to coalesce for conversion to CSR

/**
*@brief Merges two sorted subarrays of a COO matrix based on an external array.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*@param left Left index of the first subarray
*@param mid Middle index separating the two subarrays
*@param right Right index of the second subarray
*
*This function merges two sorted subarrays within the COO matrix based on the corresponding values in the external sorting array.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
SIZE_TYPE inplace_merge_coo_external(SIZE_TYPE *cols,
                            SIZE_TYPE *rows,
                            VALUE_TYPE *vals,
                            EXTERNAL_TYPE *ext_vals,
                            SIZE_TYPE left,
                            SIZE_TYPE mid,
                            SIZE_TYPE right) {
    SIZE_TYPE i = left;
    SIZE_TYPE j = mid + 1;

    // Traverse both subarrays and merge in-place
    while (i <= mid && j <= right) {
        // If the element in the left half is smaller, move on
        if (ext_vals[i] <= ext_vals[j]) {
            i++;
        } else if (ext_vals[i] > ext_vals[j]) {
            // Element in the right half is smaller, rotate to the left
            SIZE_TYPE temp_row = rows[j];
            SIZE_TYPE temp_col = cols[j];
            VALUE_TYPE temp_val = vals[j];
            VALUE_TYPE temp_ext_val = ext_vals[j];

            // Shift elements from i to j-1 one position to the right
            for (SIZE_TYPE k = j; k > i; k--) {
                rows[k] = rows[k - 1];
                cols[k] = cols[k - 1];
                vals[k] = vals[k - 1];
                ext_vals[k] = ext_vals[k - 1];
            }

            // Place the j-th element in its correct position
            rows[i] = temp_row;
            cols[i] = temp_col;
            vals[i] = temp_val;
            ext_vals[i] = temp_ext_val;

            // Update indices
            i++;
            mid++;
            j++;
        }
    }
    return;
}

/**
*@brief Sorts a COO matrix based on an external array using insertion sort.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*@param left Left index of the subarray
*@param right Right index of the subarray
*
*This function sorts a subarray within the COO matrix based on the corresponding values in the external sorting array using insertion sort.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
void insertion_sort_coo_external(SIZE_TYPE *cols,
                                 SIZE_TYPE *rows,
                                 VALUE_TYPE *vals,
                                 EXTERNAL_TYPE *ext_vals,
                                 SIZE_TYPE left,
                                 SIZE_TYPE right) {
    for (unsigned long i = left + 1; i <= right; i++) {
        SIZE_TYPE temp_row = rows[i];
        SIZE_TYPE temp_col = cols[i];
        VALUE_TYPE temp_val = vals[i];
        EXTERNAL_TYPE temp_ext_val = ext_vals[i];

        SIZE_TYPE j = i;
        while (j > left && (ext_vals[j - 1] > temp_ext_val)) {
            rows[j] = rows[j - 1];
            cols[j] = cols[j - 1];
            vals[j] = vals[j - 1];
            ext_vals[j] = ext_vals[j - 1];
            j--;
        }
        // Insert (temp_row, temp_col, temp_val) at its correct position
        rows[j] = temp_row;
        cols[j] = temp_col;
        vals[j] = temp_val;
        ext_vals[j] = temp_ext_val;
    }
    return;
}

/**
*@brief Recursively sorts a COO matrix based on an external array using merge sort.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*@param left Left index of the subarray
*@param right Right index of the subarray
*
*This function recursively sorts a subarray within the COO matrix based on the corresponding values in the external sorting array using merge sort.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
void recursive_merge_sort_coo_external(SIZE_TYPE *cols,
                                       SIZE_TYPE *rows,
                                       VALUE_TYPE *vals,
                                       EXTERNAL_TYPE *ext_vals,
                                       SIZE_TYPE left,
                                       SIZE_TYPE right) {
    if (left < right) {
        if (right - left >= 32) {
            SIZE_TYPE mid = (left + right) / 2;
#pragma omp taskgroup
            {
#pragma omp task shared(cols, rows, vals, left_duplicates) untied if (right - left >= (1 << 14))
                recursive_merge_sort_coo(cols, rows, vals, ext_vals, left, mid);
#pragma omp task shared(cols, rows, vals, right_duplicates) untied if (right - left >= (1 << 14))
                recursive_merge_sort_coo(cols, rows, vals, ext_vals, mid + 1, right);
#pragma omp taskyield
            }

            inplace_merge_coo_external(cols, rows, vals, ext_vals, left, mid, right);
        } else {
            insertion_sort_coo_external(cols, rows, vals, ext_vals, left, right);
        }
    }
    return;
}

/**
*@brief Sorts a COO matrix based on an external array using merge sort.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*
*This function sorts the entire COO matrix based on the corresponding values in the external sorting array using merge sort.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
void merge_sort_coo_external(SIZE_TYPE *cols, SIZE_TYPE *rows, VALUE_TYPE *vals, EXTERNAL_TYPE *ext_vals) {
#pragma omp parallel
#pragma omp single
    duplicates+=recursive_merge_sort_coo(cols, rows, vals, 0, (SIZE_TYPE)size - 1, 0);

    return;
}

/*-----------------------------------------MERGE SORTED COOs-------------------------------------------*/

/**
*@brief Performs a binary search on a COO matrix to find the appropriate position for a given row and column pair.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (not used in this function)
*@param rows Pointer to row indices
*@param cols Pointer to column indices
*@param row Row value to search for
*@param col Column value to search for
*@param low Lower bound of the search range
*@param high Upper bound of the search range
*
*This function performs a binary search on the COO matrix to find the appropriate position for a given row and column pair.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE>
size_t binary_search_coo(const SIZE_TYPE* rows, const SIZE_TYPE* cols, SIZE_TYPE row, SIZE_TYPE col, size_t low, size_t high) {
    while (low < high) {
        size_t mid = (low + high) / 2;
        if ((rows[mid] < row) || (rows[mid] == row && cols[mid] < col))
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

/**
*@brief Merges two sorted COO matrices into a single sorted COO matrix using multi-threading.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@param m_rows Pointer to row indices of the first COO matrix
*@param m_cols Pointer to column indices of the first COO matrix
*@param m_vals Pointer to values of the first COO matrix
*@param n_rows Pointer to row indices of the second COO matrix
*@param n_cols Pointer to column indices of the second COO matrix
*@param n_vals Pointer to values of the second COO matrix
*@param c_rows Pointer to row indices of the resulting COO matrix
*@param c_cols Pointer to column indices of the resulting COO matrix
*@param c_vals Pointer to values of the resulting COO matrix
*@param m_size Size of the first COO matrix
*@param n_size Size of the second COO matrix
*@param num_threads Number of threads to use for parallel processing
*
*This function merges two sorted COO matrices into a single sorted COO matrix using multi-threading.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE>
size_t parallel_merge_sorted_coos(const SIZE_TYPE* m_rows, const SIZE_TYPE* m_cols, const VALUE_TYPE* m_vals,
                                  const SIZE_TYPE* n_rows, const SIZE_TYPE* n_cols, const VALUE_TYPE* n_vals,
                                  SIZE_TYPE* c_rows, SIZE_TYPE* c_cols, VALUE_TYPE* c_vals,
                                  size_t m_size, size_t n_size, int num_threads) {
    size_t duplicates = 0;

    // Step 1: Determine chunk ranges for m and corresponding n ranges
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t chunk_size = (m_size + num_threads - 1) / num_threads; // Ceiling division

        size_t m_begin = thread_id * chunk_size;
        size_t m_end = std::min(m_begin + chunk_size, m_size);

        SIZE_TYPE m_begin_row = m_rows[m_begin];
        SIZE_TYPE m_begin_col = m_cols[m_begin];
        SIZE_TYPE m_end_row = m_end < m_size ? m_rows[m_end - 1] : m_rows[m_size - 1];
        SIZE_TYPE m_end_col = m_end < m_size ? m_cols[m_end - 1] : m_cols[m_size - 1];

        size_t n_begin = binary_search_coo(n_rows, n_cols, m_begin_row, m_begin_col, 0, n_size);
        size_t n_end = binary_search_coo(n_rows, n_cols, m_end_row, m_end_col, 0, n_size);

        size_t c_start = m_begin + n_begin;

        // Merge subarrays into c
        size_t i = m_begin, j = n_begin, k = c_start;
        while (i < m_end && j < n_end) {
            if ((m_rows[i] < n_rows[j]) || (m_rows[i] == n_rows[j] && m_cols[i] <= n_cols[j])) {
                if (k > 0 && c_rows[k - 1] == m_rows[i] && c_cols[k - 1] == m_cols[i]) {
                    #pragma omp atomic
                    c_vals[k - 1] += m_vals[i]; // Sum duplicates
                    #pragma omp atomic
                    duplicates++;
                } else {
                    c_rows[k] = m_rows[i];
                    c_cols[k] = m_cols[i];
                    c_vals[k++] = m_vals[i];
                }
                i++;
            } else {
                if (k > 0 && c_rows[k - 1] == n_rows[j] && c_cols[k - 1] == n_cols[j]) {
                    #pragma omp atomic
                    c_vals[k - 1] += n_vals[j]; // Sum duplicates
                    #pragma omp atomic
                    duplicates++;
                } else {
                    c_rows[k] = n_rows[j];
                    c_cols[k] = n_cols[j];
                    c_vals[k++] = n_vals[j];
                }
                j++;
            }
        }
        while (i < m_end) {
            if (k > 0 && c_rows[k - 1] == m_rows[i] && c_cols[k - 1] == m_cols[i]) {
                #pragma omp atomic
                c_vals[k - 1] += m_vals[i]; // Sum duplicates
                #pragma omp atomic
                duplicates++;
            } else {
                c_rows[k] = m_rows[i];
                c_cols[k] = m_cols[i];
                c_vals[k++] = m_vals[i];
            }
            i++;
        }
        while (j < n_end) {
            if (k > 0 && c_rows[k - 1] == n_rows[j] && c_cols[k - 1] == n_cols[j]) {
                #pragma omp atomic
                c_vals[k - 1] += n_vals[j]; // Sum duplicates
                #pragma omp atomic
                duplicates++;
            } else {
                c_rows[k] = n_rows[j];
                c_cols[k] = n_cols[j];
                c_vals[k++] = n_vals[j];
            }
            j++;
        }
    }

    return duplicates;
}

/*
int main() {
    // Example COO arrays
    std::vector<int> m_rows = {1, 3, 5};
    std::vector<int> m_cols = {1, 2, 3};
    std::vector<double> m_vals = {10.0, 20.0, 30.0};

    std::vector<int> n_rows = {2, 3, 6};
    std::vector<int> n_cols = {1, 2, 3};
    std::vector<double> n_vals = {15.0, 25.0, 35.0};

    size_t m_size = m_rows.size();
    size_t n_size = n_rows.size();
    size_t c_size = m_size + n_size;

    std::vector<int> c_rows(c_size);
    std::vector<int> c_cols(c_size);
    std::vector<double> c_vals(c_size);

    int num_threads = 4; // Example: 4 threads

    size_t duplicates = parallel_merge_coo(m_rows.data(), m_cols.data(), m_vals.data(),
                                           n_rows.data(), n_cols.data(), n_vals.data(),
                                           c_rows.data(), c_cols.data(), c_vals.data(),
                                           m_size, n_size, num_threads);

    // Output the merged COO array
    for (size_t i = 0; i < c_size - duplicates; i++) {
        std::cout << "(" << c_rows[i] << ", " << c_cols[i] << ") -> " << c_vals[i] << "\n";
    }

    return 0;
}
*/

/*------------------------SUBTRACT BOTTOM K FROM COOs (external importance array)----------------------------------*/

/**
*@brief Finds the bottom-k indices of an array in parallel.
*
*@tparam VALUE_TYPE Type for values stored in the array (e.g., float)
*@param values Pointer to the array of external values
*@param size The size of the input array
*@param k The number of smallest elements to find
*@param num_threads Number of threads to use for parallel processing
*
*This function finds the bottom-k indices of an array in parallel.
*/
template <typename VALUE_TYPE>
std::vector<size_t> bottom_k_indices(VALUE_TYPE *values, size_t size, size_t k, int num_threads) {
    // Each thread processes a chunk of the array
    size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::vector<size_t>> thread_indices(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t start = thread_id * chunk_size;
        size_t end = std::min(start + chunk_size, size);

        // Collect indices for this thread
        std::vector<size_t> local_indices;
        for (size_t i = start; i < end; ++i) {
            local_indices.push_back(i);
        }

        // Sort local indices by values
        std::partial_sort(local_indices.begin(), local_indices.begin() + std::min(k, local_indices.size()), local_indices.end(),
                          [&values](size_t a, size_t b) { return values[a] < values[b]; });

        // Keep only the smallest k elements
        if (local_indices.size() > k) {
            local_indices.resize(k);
        }

        thread_indices[thread_id] = std::move(local_indices);
    }

    // Merge results from all threads
    std::vector<size_t> merged_indices;
    for (const auto &indices : thread_indices) {
        merged_indices.insert(merged_indices.end(), indices.begin(), indices.end());
    }

    // Find the global bottom-k indices
    std::partial_sort(merged_indices.begin(), merged_indices.begin() + k, merged_indices.end(),
                      [&values](size_t a, size_t b) { return values[a] < values[b]; });

    merged_indices.resize(k);
    return merged_indices;
}

/**
*@brief Removes bottom k elements based on external values and copies the result into another COO array in parallel.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for external importance values (e.g., float)
*@param cols Pointer to the column indices of the input COO array
*@param rows Pointer to the row indices of the input COO array
*@param vals Pointer to the values of the input COO array
*@param ext_vals Pointer to external importance values associated with the COO indices
*@param c_cols Pointer to the column indices of the output COO array
*@param c_rows Pointer to the row indices of the output COO array
*@param c_vals Pointer to the values of the output COO array
*@param c_ext_vals Pointer to the external values of the output COO array
*@param size The size of the input COO array
*@param k Number of elements to remove based on the smallest external values
*@param num_threads Number of threads to use for parallel processing
*
*This function removes bottom k elements based on external values and copies the result into another COO array in parallel.
*/
template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
void coo_subtract_bottom_k(SIZE_TYPE *cols, SIZE_TYPE *rows, VALUE_TYPE *vals,
                           EXTERNAL_TYPE *ext_vals, SIZE_TYPE *c_cols,
                           SIZE_TYPE *c_rows, VALUE_TYPE *c_vals,
                           EXTERNAL_TYPE *c_ext_vals, SIZE_TYPE size,
                           SIZE_TYPE k, int num_threads) {
    if (k * num_threads > size) {
        // Sort the COO array based on external values
        merge_sort_coo_external(cols, rows, vals, ext_vals, size);

        // Skip the first k elements
        cols += k;
        rows += k;
        vals += k;
        ext_vals += k;
        size -= k;

        // Sort the COO by indices to restore order
        merge_sort_coo(cols, rows, vals, size);

        // Copy the result to the output COO array
        std::copy(cols, cols + size, c_cols);
        std::copy(rows, rows + size, c_rows);
        std::copy(vals, vals + size, c_vals);
        std::copy(ext_vals, ext_vals + size, c_ext_vals);
        return;
    }

    // Step 1: Find bottom-k indices in parallel
    std::vector<size_t> bottom_k = bottom_k_indices(ext_vals, size, k, num_threads);

    // Step 2: Copy COO elements to the output array, skipping bottom-k
    #pragma omp parallel for num_threads(num_threads)
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        size_t start = thread_id * (size / num_threads);
        size_t end = (thread_id == num_threads - 1) ? size : (thread_id + 1) * (size / num_threads);

        // Find the first relevant index in bottom_k for this thread
        size_t bottom_k_index = std::lower_bound(bottom_k.begin(), bottom_k.end(), start) - bottom_k.begin();

        size_t c_index = start - bottom_k_index;
        for (size_t i = start; i < end; ++i) {
            // Check if current index is in bottom_k
            if (bottom_k_index < bottom_k.size() && bottom_k[bottom_k_index] == i) {
                ++bottom_k_index;  // Skip this index
            } else {
                // Copy to output COO
                c_cols[c_index] = cols[i];
                c_rows[c_index] = rows[i];
                c_vals[c_index] = vals[i];
                c_ext_vals[c_index] = ext_vals[i];
                ++c_index;
            }
        }
    }
}

// Main function demonstrating the use of coo_subtract_bottom_k
/*int main() {
    // Example input COO array
    const size_t size = 10;
    const size_t k = 3;
    const int num_threads = 4;

    size_t cols[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t rows[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float vals[size] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    float ext_vals[size] = {0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0};

    size_t c_cols[size - k];
    size_t c_rows[size - k];
    float c_vals[size - k];
    float c_ext_vals[size - k];

    // Call the function
    coo_subtract_bottom_k(cols, rows, vals, ext_vals, c_cols, c_rows, c_vals, c_ext_vals, size, k, num_threads);

    // Output results
    for (size_t i = 0; i < size - k; ++i) {
        printf("Row: %zu, Col: %zu, Val: %.2f, Ext Val: %.2f\n", c_rows[i], c_cols[i], c_vals[i], c_ext_vals[i]);
    }

    return 0;
}*/


#endif