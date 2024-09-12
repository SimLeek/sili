#ifndef __CSF__H_
#define __CSF__H_
//#include <vector>
#include "unique_vector.hpp"

struct csf_struct // compressed sparse fiber struct
{
    int *ptrptrs;
    int *col_indices; // col indices assuming CSR. ptrptrs will point to the indices of these as well as ptrs

    int *ptrs;
    int *fiber_indices;

    float *values;
    int nnz; // total number of values
    int nnf; // total number of fibers with any values in them
    int rows;
    int cols;
    int fibers;

    csf_struct()
        : ptrptrs(nullptr), col_indices(nullptr), ptrs(nullptr), fiber_indices(nullptr), values(nullptr), nnz(0),
          nnf(0), rows(0), cols(0), fibers(0) {}

    csf_struct(int *_ptrptrs,
               int *_col_indices,
               int *_ptrs,
               int *_fiber_indices,
               float *_values,
               int nnz,
               int nnf,
               int _rows,
               int _cols,
               int _fibers)
        : ptrptrs(_ptrptrs), col_indices(_col_indices), ptrs(_ptrs), fiber_indices(_fiber_indices), values(_values),
          nnz(nnz), nnf(nnf), rows(_rows), cols(_cols), fibers(_fibers) {}
};

csf_struct convert_vovov_to_csf(const sili::unique_vector<sili::unique_vector<int>> *col_indices,
                                const sili::unique_vector<sili::unique_vector<sili::unique_vector<int>>> *fiber_indices,
                                const sili::unique_vector<sili::unique_vector<sili::unique_vector<float>>> *fiber_values,
                                int num_row,
                                int num_col,
                                int num_fiber,
                                int nnz,
                                int nnf);

#endif