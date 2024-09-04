#include "../headers/csf.h"
#include "../headers/scan.h"
#include <algorithm>
#include <functional>
#include <omp.h>
#include <thread>
#include <vector>

/* #region Sparse IO Conv2D Forward */

// setup conv2d heads and storage for new row
inline void _setup_sparse_io_conv2d_row(int input_height,
                                        int batch,
                                        int oi,
                                        int &oiy,
                                        int tid,
                                        int kh_diff_n,
                                        std::vector<int> &start_row,
                                        std::vector<std::vector<int>> &output_col_indices,
                                        std::vector<std::vector<std::vector<int>>> &output_channel_indices,
                                        std::vector<std::vector<std::vector<float>>> &output_values,
                                        std::vector<int> &vip_col,
                                        std::vector<csf_struct> &input_sparse_images) {

    oiy = int(oi / input_height); // output index y
    if (start_row[tid] == -1) {
        start_row[tid] = oiy;
    }
    // set up memory for this row
    output_col_indices.push_back(std::vector<int>());
    output_channel_indices.push_back(std::vector<std::vector<int>>());
    output_values.push_back(std::vector<std::vector<float>>());

    for (int i = 0; i < vip_col.size(); i++) {
        int vip_row = oiy + (i - kh_diff_n);
        vip_col[i] = input_sparse_images[batch].ptrptrs[vip_row]; // vip_col
    }
}

// perform forward convolution
inline void _do_sparse_io_conv2d_convolution(int batch,
                                             int output_channels,
                                             int input_width,
                                             int input_channels,
                                             int oix,
                                             int kw_diff_n,
                                             int kw_diff_p,
                                             float *W,
                                             bool &made_this_fiber,
                                             std::vector<int> &vip_col,
                                             std::vector<csf_struct> &input_sparse_images,
                                             std::vector<std::vector<int>> &output_col_indices,
                                             std::vector<std::vector<std::vector<int>>> &output_channel_indices,
                                             std::vector<std::vector<std::vector<float>>> &output_values) {
    bool check_next = true;
    while (check_next) {
        check_next = false; // because I don't like the aesthetics of do while loops
        for (int i = 0; i < vip_col.size(); i++) {
            // vip_end[i] = vip[i+1]; <-- this is why vip.size() is kernel_height+1: no need for an end
            // array
            if (i == 0) {
                continue; // end pointer not set
            } else if (vip_col[i - 1] == vip_col[i]) {
                continue; // input row has no more items
            } else if (input_sparse_images[batch].col_indices[vip_col[i - 1]] < oix - kw_diff_n) {
                vip_col[i - 1]++;
                check_next = true;
                continue;
            } else if (input_sparse_images[batch].col_indices[vip_col[i - 1]] >= oix + kw_diff_p) {
                continue; // input row items are all beyond kernel input area
            } else {      // perform convolution part
                int input_channel_ptr = input_sparse_images[batch].ptrs[vip_col[i - 1]];
                int input_channel_index = input_sparse_images[batch].fiber_indices[input_channel_ptr];
                int input_channel_value = input_sparse_images[batch].values[input_channel_ptr];

                for (int oci = 0; oci < output_channels; oci++) {
                    if (!made_this_fiber) {
                        output_col_indices.back().push_back(input_sparse_images[batch].col_indices[vip_col[i - 1]]);
                        output_channel_indices.back().push_back(std::vector<int>());
                        output_values.back().push_back(std::vector<float>());
                        made_this_fiber = true;
                    }

                    int kernel_H = i - 1;
                    int kernel_W = input_sparse_images[batch].col_indices[vip_col[i - 1]] - oix;

                    // kernel format will be HWOI due to the way these loops had to be designed.
                    // image format should also be NHWC
                    //  matching format to for loops = fewer cache misses
                    float out_val =
                        W[kernel_H * input_width * output_channels * input_channels +
                          kernel_W * output_channels * input_channels + oci * input_channels + input_channel_index] *
                        input_channel_value;
                    if (output_channel_indices.back().back().back() != oci) {
                        output_channel_indices.back().back().push_back(oci);
                        output_values.back().back().push_back(out_val);
                    } else {
                        output_values.back().back().back() += out_val;
                    }

                    vip_col[i - 1]++;
                    check_next = true;
                }
            }
        }
    }
}

// moves the vertical column of input pointers back for the next output pixel
inline void _move_sparse_io_conv2d_column_ptrs_back(int batch,
                                                    int kw_diff_n,
                                                    int oix,
                                                    std::vector<csf_struct> &input_sparse_images,
                                                    std::vector<int> &vip_col) {
    bool check_next = true;
    while (check_next) {
        check_next = false; // because I don't like the aesthetics of do while loops
        for (int i = 0; i < vip_col.size(); i++) {
            // vip_end[i] = vip[i+1]; <-- this is why vip.size() is kernel_height+1: no need for an end
            // array
            if (input_sparse_images[batch].col_indices[vip_col[i]] >
                oix - kw_diff_n + 1) { // if greater than first index of next convolution
                vip_col[i - 1]--;
                check_next = true;
                continue; // input row items are all beyond kernel input area
            }
        }
    }
}

inline void _conv2d_remove_zero_outputs(std::vector<std::vector<int>> &output_col_indices,
                                        std::vector<std::vector<std::vector<int>>> &output_channel_indices,
                                        std::vector<std::vector<std::vector<float>>> &output_values,
                                        float eps) {
    std::vector<int>::iterator it = output_channel_indices.back().back().begin();
    std::vector<float>::iterator jt = output_values.back().back().begin();
    while (it != output_channel_indices.back().back().end()) {
        if ((*jt) < eps) {
            it = output_channel_indices.back().back().erase(it);
            jt = output_values.back().back().erase(jt);
        } else {
            ++it;
            ++jt;
        }
    }

    // remove entire fiber if no output values left
    if (output_channel_indices.back().back().size() == 0) {
        output_col_indices.back().pop_back();
        output_channel_indices.back().pop_back();
        output_values.back().pop_back();
    }
}

inline void _sparse_io_conv2d_reserve(int &nnf,
                                      int &nnc,
                                      std::vector<std::vector<size_t>> &vec_channel_assign_locs,
                                      std::vector<std::vector<size_t>> &vec_col_assign_locs,
                                      std::vector<int> &start_row,
                                      std::vector<std::vector<int>> &out_col_idx,
                                      std::vector<std::vector<std::vector<int>>> &out_chan_idx,
                                      std::vector<std::vector<std::vector<float>>> &out_val) {
#pragma omp parallel for reduction(+ : nnf, nnc)
    for (int i = 0; i < vec_channel_assign_locs.size(); i++) {
        for (int j = 0; j < vec_channel_assign_locs[i].size(); j++) {
            // reserve *additional* space for output, as different threads may have already reserved some space
            out_col_idx[start_row[i] + j].resize(out_col_idx[start_row[i] + j].size() + vec_col_assign_locs[i][j], 0);
            out_chan_idx[start_row[i] + j].resize(out_chan_idx[start_row[i] + j].size() +
                                                  vec_channel_assign_locs[i][j]);
            out_val[start_row[i] + j].resize(out_val[start_row[i] + j].size() + vec_channel_assign_locs[i][j]);
            nnf += vec_channel_assign_locs[i][j];
            nnc += vec_col_assign_locs[i][j];
        }
    }
}

inline void _sparse_io_conv2d_copy_to_output_vovov(
    int num_cpus,
    std::vector<std::vector<size_t>> &vec_channel_assign_locs,
    std::vector<std::vector<size_t>> &vec_col_assign_locs,
    std::vector<int> &start_row,
    std::vector<std::vector<std::vector<int>>> &output_col_indices_chunks,
    std::vector<std::vector<std::vector<std::vector<int>>>> &output_channel_indices_chunks,
    std::vector<std::vector<std::vector<std::vector<float>>>> &output_values_chunks,
    std::vector<std::vector<int>> &out_col_idx,
    std::vector<std::vector<std::vector<int>>> &out_chan_idx,
    std::vector<std::vector<std::vector<float>>> &out_val) {
#pragma omp parallel num_threads(num_cpus)
    {
        int tid = omp_get_thread_num(); // Get thread ID
        int start = vec_col_assign_locs[tid][0];
        // int end = vec_assign_locs[tid+1];
        if (tid != 0 && start_row[tid - 1] + vec_channel_assign_locs[tid - 1].size() == start_row[tid]) {
            start += vec_channel_assign_locs[tid - 1].back();
        } else {
            start = 0;
        }
        for (int j = 0; j < vec_channel_assign_locs[tid].size(); j++) {
            std::copy(output_col_indices_chunks[tid][j].begin(),
                      output_col_indices_chunks[tid][j].end(),
                      out_col_idx[start_row[tid]].begin() + start);
            std::copy(output_channel_indices_chunks[tid][j].begin(),
                      output_channel_indices_chunks[tid][j].end(),
                      out_chan_idx[start_row[tid]].begin() + start);
            std::copy(output_values_chunks[tid][j].begin(),
                      output_values_chunks[tid][j].end(),
                      out_val[start_row[tid]].begin() + start);
            start = 0; // only first row can have start offset
        }
    }
}

std::vector<csf_struct> conv2d(int batch_size,
                               int input_channels,
                               int output_channels,
                               int input_width,  // output width = input width
                               int input_height, // output height = input height
                               int kernel_width,
                               int kernel_height,
                               std::vector<csf_struct> input_sparse_images,
                               float *W,
                               float eps = std::numeric_limits<float>::epsilon()) {
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware threads

    std::vector<csf_struct> output_sparse_images;

    int kh_diff_n = int((kernel_height - .5) / 2); // kernel height diff negative
    int kw_diff_n = int((kernel_width - .5) / 2);  // kernel width diff negative
    int kh_diff_p = int(kernel_height / 2);        // kernel height diff negative
    int kw_diff_p = int(kernel_width / 2);         // kernel width diff negative

    for (int batch = 0; batch < batch_size; batch++) {

        std::vector<std::vector<int>> out_col_idx(input_height);
        std::vector<std::vector<std::vector<int>>> out_chan_idx(input_height);
        std::vector<std::vector<std::vector<float>>> out_val(input_height);
        int nnz = 0;
        int nnf = 0;

        std::vector<int> start_row(num_cpus, -1);

        std::vector<std::vector<std::vector<int>>> output_col_indices_chunks(num_cpus, std::vector<std::vector<int>>());
        std::vector<std::vector<std::vector<std::vector<int>>>> output_channel_indices_chunks(
            num_cpus, std::vector<std::vector<std::vector<int>>>());
        std::vector<std::vector<std::vector<std::vector<float>>>> output_values_chunks(
            num_cpus, std::vector<std::vector<std::vector<float>>>());

#pragma omp parallel num_threads(num_cpus)
        {
            int output_size = input_width * input_height;
            int tid = omp_get_thread_num();                           // Get thread ID
            int chunk_size = (output_size + num_cpus - 1) / num_cpus; // chunk size
            int start = tid * chunk_size;                             // Start index for this thread
            int end = std::min(start + chunk_size, output_size);      // End index for this thread

            std::vector<int> vip_col(kernel_height + 1, 0); // vertical input pointers to columns

            std::vector<std::vector<int>> &output_col_indices = output_col_indices_chunks[tid];
            std::vector<std::vector<std::vector<int>>> &output_channel_indices = output_channel_indices_chunks[tid];
            std::vector<std::vector<std::vector<float>>> &output_values = output_values_chunks[tid];
            int oiy = -1;
            for (int oi = start; oi < end; ++oi) { // output index

                // SECTION: setup row/starts
                if (int(oi / input_height) != oiy) {
                    _setup_sparse_io_conv2d_row(input_height,
                                                batch,
                                                oi,
                                                oiy,
                                                tid,
                                                kh_diff_n,
                                                start_row,
                                                output_col_indices,
                                                output_channel_indices,
                                                output_values,
                                                vip_col,
                                                input_sparse_images);
                }

                int oix = oi % oiy; // output index x
                float out_val = 0;
                bool made_this_fiber = false;

                // SECTION: convolution with vertical pointers
                _do_sparse_io_conv2d_convolution(batch,
                                                 output_channels,
                                                 input_width,
                                                 input_channels,
                                                 oix,
                                                 kw_diff_n,
                                                 kw_diff_p,
                                                 W,
                                                 made_this_fiber,
                                                 vip_col,
                                                 input_sparse_images,
                                                 output_col_indices,
                                                 output_channel_indices,
                                                 output_values);

                // SECTION: move vertical column pointers back
                _move_sparse_io_conv2d_column_ptrs_back(batch, kw_diff_n, oix, input_sparse_images, vip_col);

                // SECTION: remove 0 outputs
                // remove 0 output values from fiber
                _conv2d_remove_zero_outputs(output_col_indices, output_channel_indices, output_values, eps);
            }
        }

        auto vec_channel_assign_locs = fullScanSizes2(output_channel_indices_chunks);
        auto vec_col_assign_locs = fullScanSizes2(output_col_indices_chunks);

        _sparse_io_conv2d_reserve(
            nnz, nnf, vec_channel_assign_locs, vec_col_assign_locs, start_row, out_col_idx, out_chan_idx, out_val);

        // SECTION: copy
        _sparse_io_conv2d_copy_to_output_vovov(num_cpus,
                                               vec_channel_assign_locs,
                                               vec_col_assign_locs,
                                               start_row,
                                               output_col_indices_chunks,
                                               output_channel_indices_chunks,
                                               output_values_chunks,
                                               out_col_idx,
                                               out_chan_idx,
                                               out_val);

        auto cs2 = convert_vovov_to_csf(
            &out_col_idx, &out_chan_idx, &out_val, input_height, input_width, output_channels, nnz, nnf);
        output_sparse_images.push_back(cs2);
    }

    return output_sparse_images;
}
/* #endregion */

/* #region Sparse IO Conv2D Backward W */
inline void _conv2d_backward_W_do_conv(std::vector<csf_struct> &input_sparse_images,
                                       std::vector<csf_struct> &output_sparse_image_grads,
                                       int batch,
                                       int kw_diff_n,
                                       int kw_diff_p,
                                       int ograd_row_num,
                                       int ograd_col_num,
                                       int ograd_fiber_ptr_start,
                                       int ograd_fiber_ptr_end,
                                       std::vector<int> &vip_col,
                                       std::function<void(float, int, int, int, int)> &W_grad_callback) {
    bool check_next = true;
    while (check_next) {
        check_next = false; // because I don't like the aesthetics of do while loops
        for (int i = 0; i < vip_col.size(); i++) {
            // vip_end[i] = vip[i+1]; <-- this is why vip.size() is kernel_height+1: no need for an end
            // array
            if (i == 0) {
                continue; // end pointer not set
            } else if (vip_col[i - 1] == vip_col[i]) {
                continue; // input row has no more items
            } else if (input_sparse_images[ograd_row_num].col_indices[vip_col[i - 1]] < ograd_col_num - kw_diff_n) {
                vip_col[i - 1]++;
                check_next = true;
                continue;
            } else if (input_sparse_images[ograd_row_num].col_indices[vip_col[i - 1]] >= ograd_col_num + kw_diff_p) {
                continue; // input row items are all beyond kernel input area
            } else {      // perform convolution part
                int input_channel_ptr = input_sparse_images[ograd_row_num].ptrs[vip_col[i - 1]];
                int input_channel_index = input_sparse_images[ograd_row_num].fiber_indices[input_channel_ptr];
                int input_channel_value = input_sparse_images[ograd_row_num].values[input_channel_ptr];

                for (int ocp = ograd_fiber_ptr_start; ocp < ograd_fiber_ptr_end; ocp++) {
                    int oci = output_sparse_image_grads[batch].fiber_indices[ocp];
                    int ocv = output_sparse_image_grads[batch].values[ocp];

                    int kernel_H = i - 1;
                    int kernel_W = input_sparse_images[ograd_row_num].col_indices[vip_col[i - 1]] - ograd_col_num;

                    W_grad_callback(ocv * input_channel_value, kernel_H, kernel_W, oci, input_channel_index);

                    vip_col[i - 1]++;
                    check_next = true;
                }
            }
        }
    }
}

inline void _conv2d_backward_W_mov_in_col_back(std::vector<csf_struct> &input_sparse_images,
                                               int ograd_col_num,
                                               int kw_diff_n,
                                               std::vector<int> &vip_col) {
    bool check_next = true;
    while (check_next) {
        check_next = false; // because I don't like the aesthetics of do while loops
        for (int i = 0; i < vip_col.size(); i++) {
            // vip_end[i] = vip[i+1]; <-- this is why vip.size() is kernel_height+1: no need for an end
            // array
            if (input_sparse_images[ograd_col_num].col_indices[vip_col[i]] >
                ograd_col_num - kw_diff_n + 1) { // if greater than first index of next convolution
                vip_col[i - 1]--;
                check_next = true;
                continue; // input row items are all beyond kernel input area
            }
        }
    }
}

void conv2d_backward_W( // too big function, separate
    int batch_size,
    int input_channels,
    int output_channels,
    int input_width,  // output width = input width
    int input_height, // output height = input height
    int kernel_width,
    int kernel_height,
    float *W,
    std::vector<csf_struct> input_sparse_images,
    // std::vector<csf_struct> input_sparse_image_grad_masks,  // Use this in input backward
    std::vector<csf_struct> output_sparse_image_grads,
    std::function<void(float, int, int, int, int)> W_grad_callback
    // float eps=std::numeric_limits<float>::epsilon()
) {
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware threads

    // set input ptrs
    int kh_diff_n = int((kernel_height - .5) / 2); // kernel height diff negative
    int kw_diff_n = int((kernel_width - .5) / 2);  // kernel width diff negative
    int kh_diff_p = int(kernel_height / 2);        // kernel height diff negative
    int kw_diff_p = int(kernel_width / 2);         // kernel width diff negative

    for (int batch = 0; batch < batch_size; batch++) {

#pragma omp parallel num_threads(num_cpus) // reduction(+:W_grad[:W_size])
        {
            int o_grad_size = output_sparse_image_grads[batch].nnf;
            int tid = omp_get_thread_num();                           // Get thread ID
            int chunk_size = (o_grad_size + num_cpus - 1) / num_cpus; // chunk size
            int start = tid * chunk_size;                             // Start index for this thread
            int end = std::min(start + chunk_size, o_grad_size);      // End index for this thread

            std::vector<int> vip_col(kernel_height + 1, 0); // vertical input pointers to columns

            int ograd_row_num_start = std::lower_bound(output_sparse_image_grads[batch].ptrptrs,
                                                       output_sparse_image_grads[batch].ptrptrs + input_height,
                                                       start + 0.5) -
                                      output_sparse_image_grads[batch].ptrptrs;
            // int ograd_row_num_end = std::lower_bound(output_sparse_image_grads[batch].ptrptrs,
            // output_sparse_image_grads[batch].ptrptrs + input_height, end + 0.5) -
            // output_sparse_image_grads[batch].ptrptrs; int ograd_col_ptr_start = start; int ograd_col_ptr_end = end;
            int ograd_row_num = ograd_row_num_start;
            int ograd_col_num = 0;

            for (int oi = start; oi < end; ++oi) { // output index

                if (oi >= output_sparse_image_grads[batch].ptrptrs[ograd_row_num + 1]) {
                    ograd_row_num += 1;
                    if (ograd_row_num > input_height) { // break for last row and avoid out of bounds error
                        break;
                    }
                    for (int i = 0; i < vip_col.size(); i++) {
                        int vip_row = ograd_row_num + (i - kh_diff_n);
                        vip_col[i] = input_sparse_images[ograd_row_num].ptrptrs[vip_row];
                    }
                }

                ograd_col_num = output_sparse_image_grads[batch].col_indices[oi];
                int ograd_fiber_ptr_start = output_sparse_image_grads[batch].ptrs[oi];
                int ograd_fiber_ptr_end = output_sparse_image_grads[batch].ptrs[oi + 1];

                _conv2d_backward_W_do_conv(input_sparse_images,
                                           output_sparse_image_grads,
                                           batch,
                                           kw_diff_n,
                                           kw_diff_p,
                                           ograd_row_num,
                                           ograd_col_num,
                                           ograd_fiber_ptr_start,
                                           ograd_fiber_ptr_end,
                                           vip_col,
                                           W_grad_callback);

                _conv2d_backward_W_mov_in_col_back(input_sparse_images, ograd_col_num, kw_diff_n, vip_col);
            }
        }
    }
}
/* #endregion */

/* #region Sparse IO Conv2D Backward Input */
inline void _conv2d_backward_input_do_conv(int batch,
                                           int input_width,
                                           int output_channels,
                                           int input_channels,
                                           int ograd_row_num,
                                           int ograd_col_num,
                                           int kw_diff_n,
                                           int kw_diff_p,
                                           int ograd_fiber_ptr_start,
                                           int ograd_fiber_ptr_end,
                                           float *W,
                                           std::vector<int> &vip_col,
                                           std::vector<csf_struct> &input_sparse_image_grad_masks,
                                           std::vector<csf_struct> &output_sparse_image_grads) {
    bool check_next = true;
    while (check_next) {
        check_next = false; // because I don't like the aesthetics of do while loops
        for (int i = 0; i < vip_col.size(); i++) {
            // vip_end[i] = vip[i+1]; <-- this is why vip.size() is kernel_height+1: no need for an end
            // array
            if (i == 0) {
                continue; // end pointer not set
            } else if (vip_col[i - 1] == vip_col[i]) {
                continue; // input row has no more items
            } else if (input_sparse_image_grad_masks[ograd_row_num].col_indices[vip_col[i - 1]] <
                       ograd_col_num - kw_diff_n) {
                vip_col[i - 1]++;
                check_next = true;
                continue;
            } else if (input_sparse_image_grad_masks[ograd_row_num].col_indices[vip_col[i - 1]] >=
                       ograd_col_num + kw_diff_p) {
                continue; // input row items are all beyond kernel input area
            } else {      // perform convolution part
                int input_channel_ptr = input_sparse_image_grad_masks[ograd_row_num].ptrs[vip_col[i - 1]];
                int input_channel_index = input_sparse_image_grad_masks[ograd_row_num].fiber_indices[input_channel_ptr];
                int input_channel_value = input_sparse_image_grad_masks[ograd_row_num].values[input_channel_ptr];

                for (int ocp = ograd_fiber_ptr_start; ocp < ograd_fiber_ptr_end; ocp++) {
                    int oci = output_sparse_image_grads[batch].fiber_indices[ocp];
                    int ocv = output_sparse_image_grads[batch].values[ocp];

                    int kernel_H = i - 1;
                    int kernel_W =
                        input_sparse_image_grad_masks[ograd_row_num].col_indices[vip_col[i - 1]] - ograd_col_num;

                    // W_grad_callback(ocv*input_channel_value, kernel_H, kernel_W, oci,
                    // input_channel_index);
                    int igrad_fiber_ptr_start = input_sparse_image_grad_masks[batch].ptrs[vip_col[i - 1]];
                    int igrad_fiber_ptr_end = input_sparse_image_grad_masks[batch].ptrs[vip_col[i - 1] + 1];
                    for (int icp = igrad_fiber_ptr_start; icp < igrad_fiber_ptr_end; icp++) {
                        int ici = input_sparse_image_grad_masks[batch].fiber_indices[icp];
#pragma omp atomic // todo: make this local
                        input_sparse_image_grad_masks[batch].values[icp] +=
                            W[kernel_H * input_width * output_channels * input_channels +
                              kernel_W * output_channels * input_channels + oci * input_channels + ici] *
                            ocv;
                    }
                    vip_col[i - 1]++;
                    check_next = true;
                }
            }
        }
    }
}

inline void _conv2d_backward_input_move_in_col_back(int ograd_col_num,
                                                    int kw_diff_n,
                                                    std::vector<int> &vip_col,
                                                    std::vector<csf_struct> &input_sparse_image_grad_masks) {
    bool check_next = true;
    while (check_next) {
        check_next = false; // because I don't like the aesthetics of do while loops
        for (int i = 0; i < vip_col.size(); i++) {
            // vip_end[i] = vip[i+1]; <-- this is why vip.size() is kernel_height+1: no need for an end
            // array
            if (input_sparse_image_grad_masks[ograd_col_num].col_indices[vip_col[i]] >
                ograd_col_num - kw_diff_n + 1) { // if greater than first index of next convolution
                vip_col[i - 1]--;
                check_next = true;
                continue; // input row items are all beyond kernel input area
            }
        }
    }
}

void conv2d_backward_input(int batch_size,
                           int input_channels,
                           int output_channels,
                           int input_width,  // output width = input width
                           int input_height, // output height = input height
                           int kernel_width,
                           int kernel_height,
                           float *W,
                           // std::vector<csf_struct> input_sparse_images,
                           std::vector<csf_struct> input_sparse_image_grad_masks, // Use this in input backward
                           std::vector<csf_struct> output_sparse_image_grads
                           // std::function<void(float, int, int, int, int)> W_grad_callback,
                           // float eps=std::numeric_limits<float>::epsilon()
) {
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware threads

    // set input ptrs
    int kh_diff_n = int((kernel_height - .5) / 2); // kernel height diff negative
    int kw_diff_n = int((kernel_width - .5) / 2);  // kernel width diff negative
    int kh_diff_p = int(kernel_height / 2);        // kernel height diff negative
    int kw_diff_p = int(kernel_width / 2);         // kernel width diff negative

    for (int batch = 0; batch < batch_size; batch++) {

#pragma omp parallel num_threads(num_cpus) // reduction(+:W_grad[:W_size])
        {
            int o_grad_size = output_sparse_image_grads[batch].nnf;
            int tid = omp_get_thread_num();                           // Get thread ID
            int chunk_size = (o_grad_size + num_cpus - 1) / num_cpus; // chunk size
            int start = tid * chunk_size;                             // Start index for this thread
            int end = std::min(start + chunk_size, o_grad_size);      // End index for this thread

            std::vector<int> vip_col(kernel_height + 1, 0); // vertical input pointers to columns

            int ograd_row_num_start = std::lower_bound(output_sparse_image_grads[batch].ptrptrs,
                                                       output_sparse_image_grads[batch].ptrptrs + input_height,
                                                       start + 0.5) -
                                      output_sparse_image_grads[batch].ptrptrs;
            // int ograd_row_num_end = std::lower_bound(output_sparse_image_grads[batch].ptrptrs,
            // output_sparse_image_grads[batch].ptrptrs + input_height, end + 0.5) -
            // output_sparse_image_grads[batch].ptrptrs; int ograd_col_ptr_start = start; int ograd_col_ptr_end = end;
            int ograd_row_num = ograd_row_num_start;
            int ograd_col_num = 0;

            for (int oi = start; oi < end; ++oi) { // output index

                if (oi >= output_sparse_image_grads[batch].ptrptrs[ograd_row_num + 1]) {
                    ograd_row_num += 1;
                    if (ograd_row_num > input_height) { // break for last row and avoid out of bounds error
                        break;
                    }
                    for (int i = 0; i < vip_col.size(); i++) {
                        int vip_row = ograd_row_num + (i - kh_diff_n);
                        vip_col[i] = input_sparse_image_grad_masks[ograd_row_num].ptrptrs[vip_row];
                    }
                }

                ograd_col_num = output_sparse_image_grads[batch].col_indices[oi];
                int ograd_fiber_ptr_start = output_sparse_image_grads[batch].ptrs[oi];
                int ograd_fiber_ptr_end = output_sparse_image_grads[batch].ptrs[oi + 1];

                _conv2d_backward_input_do_conv(batch,
                                               input_width,
                                               output_channels,
                                               input_channels,
                                               ograd_row_num,
                                               ograd_col_num,
                                               kw_diff_n,
                                               kw_diff_p,
                                               ograd_fiber_ptr_start,
                                               ograd_fiber_ptr_end,
                                               W,
                                               vip_col,
                                               input_sparse_image_grad_masks,
                                               output_sparse_image_grads);

                _conv2d_backward_input_move_in_col_back(
                    ograd_col_num, kw_diff_n, vip_col, input_sparse_image_grad_masks);
            }
        }
    }
}
/* #endregion */

// mask generation:
//   create 1 hot kernel: 1 in kernel_widthxkernel_height, 1 in kernel_channels_inxkernel_channels_out: channels 1s

void createRandomOneHotKernel(float *W, int kernel_H, int kernel_W, int input_channels, int output_channels) {
    // Seed the random number generator
    std::srand(std::time(nullptr));

    // Iterate over each (x, y) position in the kernel
    // todo: make this parallel as well, maybe, if it matters
    //       or make it a sparse something. Maybe a CSF2, since CSF is for 3D and the weight tensor is 4D.
    for (int x = 0; x < kernel_H; ++x) {
        for (int y = 0; y < kernel_W; ++y) {
            // For the current (x, y), get a slice of size input_channels x output_channels
            int slice_size = input_channels * output_channels;
            int start_index = (x * kernel_W + y) * slice_size;

            // Randomly select one position to set to 1
            int random_position = std::rand() % slice_size; // Get a random position

            // Set all elements to 0 first
            std::fill(W + start_index, W + start_index + slice_size, 0);

            // Set the random position to 1
            W[start_index + random_position] = 1;
        }
    }
}

inline void _setup_conv2d_grad_mask_gen(int input_height,
                                        int batch,
                                        int oi,
                                        int &oiy,
                                        int tid,
                                        int kh_diff_n,
                                        std::vector<int> &start_row,
                                        std::vector<std::vector<int>> &output_col_indices,
                                        std::vector<std::vector<std::vector<int>>> &output_channel_indices,
                                        std::vector<int> &vip_col,
                                        std::vector<csf_struct> &input_sparse_images) {
    oiy = int(oi / input_height); // output index y
    if (start_row[tid] == -1) {
        start_row[tid] = oiy;
    }
    // set up memory for this row
    output_col_indices.push_back(std::vector<int>());
    output_channel_indices.push_back(std::vector<std::vector<int>>());

    for (int i = 0; i < vip_col.size(); i++) {
        int vip_row = oiy + (i - kh_diff_n);
        vip_col[i] = input_sparse_images[oiy].ptrptrs[vip_row]; // vip_col
    }
}

inline void _do_conv2d_grad_mask_gen_convolution(int batch,
                                                 int output_channels,
                                                 int input_width,
                                                 int input_channels,
                                                 int oix,
                                                 int kw_diff_n,
                                                 int kw_diff_p,
                                                 float *W,
                                                 bool &made_this_fiber,
                                                 std::vector<int> &vip_col,
                                                 std::vector<csf_struct> &input_sparse_images,
                                                 std::vector<std::vector<int>> &output_col_indices,
                                                 std::vector<std::vector<std::vector<int>>> &output_channel_indices,
                                                 float eps) {
    bool check_next = true;
    while (check_next) {
        check_next = false; // because I don't like the aesthetics of do while loops
        for (int i = 0; i < vip_col.size(); i++) {
            // vip_end[i] = vip[i+1]; <-- this is why vip.size() is kernel_height+1: no need for an end
            // array
            if (i == 0) {
                continue; // end pointer not set
            } else if (vip_col[i - 1] == vip_col[i]) {
                continue; // input row has no more items
            } else if (input_sparse_images[batch].col_indices[vip_col[i - 1]] < oix - kw_diff_n) {
                vip_col[i - 1]++;
                check_next = true;
                continue;
            } else if (input_sparse_images[batch].col_indices[vip_col[i - 1]] >= oix + kw_diff_p) {
                continue; // input row items are all beyond kernel input area
            } else {      // perform convolution part
                int input_channel_ptr = input_sparse_images[batch].ptrs[vip_col[i - 1]];
                int input_channel_index = input_sparse_images[batch].fiber_indices[input_channel_ptr];
                int input_channel_value = input_sparse_images[batch].values[input_channel_ptr];

                for (int oci = 0; oci < output_channels; oci++) {
                    if (!made_this_fiber) {
                        output_col_indices.back().push_back(input_sparse_images[batch].col_indices[vip_col[i - 1]]);
                        output_channel_indices.back().push_back(std::vector<int>());
                        made_this_fiber = true;
                    }

                    int kernel_H = i - 1;
                    int kernel_W = input_sparse_images[batch].col_indices[vip_col[i - 1]] - oix;

                    // kernel format will be HWOI due to the way these loops had to be designed.
                    // image format should also be NHWC
                    //  matching format to for loops = fewer cache misses
                    float out_val =
                        W[kernel_H * input_width * output_channels * input_channels +
                          kernel_W * output_channels * input_channels + oci * input_channels + input_channel_index] *
                        input_channel_value;
                    if (out_val > eps) {
                        if (output_channel_indices.back().back().back() != oci) {
                            output_channel_indices.back().back().push_back(oci);
                            // output_values.back().back().push_back(out_val);
                            /*}else{
                                output_values.back().back().back()+=out_val;
                            }*/
                        }

                        vip_col[i - 1]++;
                        check_next = true;
                    }
                }
            }
        }
    }
}

inline void _sparse_io_conv2d_grad_mask_reserve(int &nnf,
                                                int &nnc,
                                                std::vector<std::vector<size_t>> &vec_channel_assign_locs,
                                                std::vector<std::vector<size_t>> &vec_col_assign_locs,
                                                std::vector<int> &start_row,
                                                std::vector<std::vector<int>> &out_col_idx,
                                                std::vector<std::vector<std::vector<int>>> &out_chan_idx) {
#pragma omp parallel for reduction(+ : nnf, nnc)
    for (int i = 0; i < vec_channel_assign_locs.size(); i++) {
        for (int j = 0; j < vec_channel_assign_locs[i].size(); j++) {
            // reserve *additional* space for output, as different threads may have already reserved some space
            out_col_idx[start_row[i] + j].resize(out_col_idx[start_row[i] + j].size() + vec_col_assign_locs[i][j], 0);
            out_chan_idx[start_row[i] + j].resize(out_chan_idx[start_row[i] + j].size() +
                                                  vec_channel_assign_locs[i][j]);
            nnf += vec_channel_assign_locs[i][j];
            nnc += vec_col_assign_locs[i][j];
        }
    }
}

inline void _sparse_io_conv2d_grad_mask_copy_to_output_vovov(
    int num_cpus,
    std::vector<std::vector<size_t>> &vec_channel_assign_locs,
    std::vector<std::vector<size_t>> &vec_col_assign_locs,
    std::vector<int> &start_row,
    std::vector<std::vector<std::vector<int>>> &output_col_indices_chunks,
    std::vector<std::vector<std::vector<std::vector<int>>>> &output_channel_indices_chunks,
    std::vector<std::vector<int>> &out_col_idx,
    std::vector<std::vector<std::vector<int>>> &out_chan_idx) {
#pragma omp parallel num_threads(num_cpus)
    {
        int tid = omp_get_thread_num(); // Get thread ID
        int start = vec_col_assign_locs[tid][0];
        // int end = vec_assign_locs[tid+1];
        if (tid != 0 && start_row[tid - 1] + vec_channel_assign_locs[tid - 1].size() == start_row[tid]) {
            start += vec_channel_assign_locs[tid - 1].back();
        } else {
            start = 0;
        }
        for (int j = 0; j < vec_channel_assign_locs[tid].size(); j++) {
            std::copy(output_col_indices_chunks[tid][j].begin(),
                      output_col_indices_chunks[tid][j].end(),
                      out_col_idx[start_row[tid]].begin() + start);
            std::copy(output_channel_indices_chunks[tid][j].begin(),
                      output_channel_indices_chunks[tid][j].end(),
                      out_chan_idx[start_row[tid]].begin() + start);
            start = 0; // only first row can have start offset
        }
    }
}

std::vector<csf_struct> conv2d_grad_mask_gen(int batch_size,
                                             int input_channels,
                                             int output_channels,
                                             int input_width,  // output width = input width
                                             int input_height, // output height = input height
                                             int kernel_width,
                                             int kernel_height,
                                             std::vector<csf_struct> input_sparse_images,
                                             float eps = std::numeric_limits<float>::epsilon()) {
    // grad mask must be within the kernel area of the output grad image nonzero locations (called input_sparse_images
    // here), otherwise they would receive zero and be skipped in the backwards op
    const auto num_cpus = std::thread::hardware_concurrency(); // Get the number of hardware threads

    std::vector<csf_struct> output_sparse_images;

    // set input ptrs
    int kh_diff_n = int((kernel_height - .5) / 2); // kernel height diff negative
    int kw_diff_n = int((kernel_width - .5) / 2);  // kernel width diff negative
    int kh_diff_p = int(kernel_height / 2);        // kernel height diff negative
    int kw_diff_p = int(kernel_width / 2);         // kernel width diff negative

    float *W = new float[kernel_height * kernel_width * input_channels * output_channels];

    for (int batch = 0; batch < batch_size; batch++) {
        createRandomOneHotKernel(W, kernel_width, kernel_width, input_channels, output_channels);

        std::vector<std::vector<int>> out_col_idx(input_height);
        std::vector<std::vector<std::vector<int>>> out_chan_idx(input_height);
        int nnz = 0;
        int nnf = 0;

        std::vector<int> start_row(num_cpus, -1);

        std::vector<std::vector<std::vector<int>>> output_col_indices_chunks(num_cpus, std::vector<std::vector<int>>());
        std::vector<std::vector<std::vector<std::vector<int>>>> output_channel_indices_chunks(
            num_cpus, std::vector<std::vector<std::vector<int>>>());

#pragma omp parallel num_threads(num_cpus)
        {
            int output_size = input_width * input_height;
            int tid = omp_get_thread_num();                           // Get thread ID
            int chunk_size = (output_size + num_cpus - 1) / num_cpus; // chunk size
            int start = tid * chunk_size;                             // Start index for this thread
            int end = std::min(start + chunk_size, output_size);      // End index for this thread

            std::vector<int> vip_col(kernel_height + 1, 0); // vertical input pointers to columns

            std::vector<std::vector<int>> &output_col_indices = output_col_indices_chunks[tid];
            std::vector<std::vector<std::vector<int>>> &output_channel_indices = output_channel_indices_chunks[tid];
            int oiy = -1;
            for (int oi = start; oi < end; ++oi) { // output index

                if (int(oi / input_height) != oiy) {
                    _setup_conv2d_grad_mask_gen(input_height,
                                                batch,
                                                oi,
                                                oiy,
                                                tid,
                                                kh_diff_n,
                                                start_row,
                                                output_col_indices,
                                                output_channel_indices,
                                                vip_col,
                                                input_sparse_images);
                }
                int oix = oi % oiy; // output index x

                bool made_this_fiber = false;

                _do_conv2d_grad_mask_gen_convolution(batch,
                                                     output_channels,
                                                     input_width,
                                                     input_channels,
                                                     oix,
                                                     kw_diff_n,
                                                     kw_diff_p,
                                                     W,
                                                     made_this_fiber,
                                                     vip_col,
                                                     input_sparse_images,
                                                     output_col_indices,
                                                     output_channel_indices,
                                                     eps);

                _move_sparse_io_conv2d_column_ptrs_back(batch, kw_diff_n, oix, input_sparse_images, vip_col);
            }
        }

        auto vec_channel_assign_locs = fullScanSizes2(output_channel_indices_chunks);
        auto vec_col_assign_locs = fullScanSizes2(output_col_indices_chunks);

        _sparse_io_conv2d_grad_mask_reserve(
            nnz, nnz, vec_channel_assign_locs, vec_col_assign_locs, start_row, out_col_idx, out_chan_idx);

        _sparse_io_conv2d_grad_mask_copy_to_output_vovov(num_cpus,
                                                         vec_channel_assign_locs,
                                                         vec_col_assign_locs,
                                                         start_row,
                                                         output_col_indices_chunks,
                                                         output_channel_indices_chunks,
                                                         out_col_idx,
                                                         out_chan_idx);

        auto cs2 = convert_vovov_to_csf(
            &out_col_idx, &out_chan_idx, nullptr, input_height, input_width, output_channels, nnz, nnf);
        output_sparse_images.push_back(cs2);
    }
    delete[] W;

    return output_sparse_images;
}