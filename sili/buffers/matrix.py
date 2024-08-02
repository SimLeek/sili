import kp

from typing import List

import numpy as np

from sili.core.devices.gpu import GPUManager
from sili.core.serial import deserialize_buffer, serialize_buffer


class MultiSquareMatrixBuffer(object):
    def __init__(self, gpu: GPUManager, matrices: List[np.ndarray]):
        self.num_matrices = len(matrices)
        self.matrix_sizes = [matrix.shape[0] for matrix in matrices]

        s = 0
        self.matrix_starts = [0]
        for m in self.matrix_sizes[:-1]:
            s += m * m
            self.matrix_starts.append(s)

        self.multi_matrix_buffer = gpu.buffer(
            np.concatenate(
                [m.flatten() for m in matrices]
            )
        )
        self.temp_buffer = gpu.buffer(
            np.concatenate(
                [np.eye(s) for s in self.matrix_sizes]
            )
        )

        self.i_pos_buffer = gpu.buffer(
            np.asarray([0], dtype=np.uint32).view(np.float32)
        )

        self.multi_mat_info_buffer = gpu.buffer(
            np.asarray([self.num_matrices] + list(zip(self.matrix_sizes, self.matrix_starts)), dtype=np.uint32).view(
                np.float32)
        )

    def to(self, t):
        if isinstance(t, GPUManager):
            if isinstance(self.multi_matrix_buffer, kp.Tensor):
                self.multi_matrix_buffer = self.multi_matrix_buffer.data()
            self.multi_matrix_buffer = t.buffer(self.multi_matrix_buffer)

            if isinstance(self.temp_buffer, kp.Tensor):
                self.temp_buffer = self.temp_buffer.data()
            self.temp_buffer = t.buffer(self.temp_buffer)

            if isinstance(self.multi_mat_info_buffer, kp.Tensor):
                self.multi_mat_info_buffer = self.multi_mat_info_buffer.data()
            self.multi_mat_info_buffer = t.buffer(self.multi_mat_info_buffer)

            if isinstance(self.i_pos_buffer, kp.Tensor):
                self.i_pos_buffer = self.i_pos_buffer.data()
            self.i_pos_buffer = t.buffer(self.i_pos_buffer)

        return self

    @property
    def size(self):
        return self.multi_matrix_buffer.size() * 2 + self.multi_mat_info_buffer.size()+1

    def set(self, matrices):
        if isinstance(matrices, list) and isinstance(matrices[0], np.ndarray):
            self.num_matrices = len(matrices)
            self.matrix_sizes = [matrix.shape[0] for matrix in matrices]

            s = 0
            self.matrix_starts = [0]
            for m in self.matrix_sizes[:-1]:
                s += m * m
                self.matrix_starts.append(s)

            self.multi_matrix_buffer.data()[...] = np.concatenate([m.flatten() for m in matrices])
            self.temp_buffer.data()[...] = np.concatenate([np.eye(s) for s in self.matrix_sizes])
            self.multi_mat_info_buffer.data()[...] = np.asarray(
                [self.num_matrices] + list(zip(self.matrix_sizes, self.matrix_starts)),
                dtype=np.uint32
            ).view(np.float32)
        else:
            raise NotImplementedError(f'Unknown matrix type: {type(matrices)}')

    def get(self):
        mat_list = []
        flattened_array = self.multi_matrix_buffer.data()
        for si, st in zip(self.matrix_sizes, self.matrix_starts):
            mat_list.append(np.frombuffer(flattened_array[st:st + si * si]).reshape([si, si]))
        return mat_list

    def __setstate__(self, state):
        self.num_matrices, self.matrix_sizes, self.matrix_starts = state[:3]
        # Restore the buffer from serialized data
        self.multi_matrix_buffer = deserialize_buffer(state[3])

        self.temp_buffer = np.concatenate([np.eye(s) for s in self.matrix_sizes])

        self.multi_mat_info_buffer = np.asarray(
            [self.num_matrices] + list(zip(self.matrix_sizes, self.matrix_starts)),
            dtype=np.uint32
        ).view(np.float32)

        self.i_pos_buffer = np.asarray([0], dtype=np.uint32).view(np.float32)

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (
            self.num_matrices,
            self.matrix_sizes,
            self.matrix_starts,
            serialize_buffer(self.multi_matrix_buffer)
        )
