#ifndef __COO__HPP_
#define __COO__HPP_

// merge while throwing away and reporting duplicates
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

template <typename SIZE_TYPE, typename VALUE_TYPE, typename SIZE_TYPE_2>
SIZE_TYPE merge_sort_coo(SIZE_TYPE *cols, SIZE_TYPE *rows, VALUE_TYPE *vals, SIZE_TYPE_2 size) {
    int duplicates = 0;
#pragma omp parallel
#pragma omp single
    duplicates+=recursive_merge_sort_coo(cols, rows, vals, 0, (SIZE_TYPE)size - 1, 0);

    return duplicates;
}

#endif