import itertools

import numpy as np
from typing import List


def center_surround_tensor(ndim,  # type: int
                           center_in,  # type: List[int]
                           center_out,  # type: List[int]
                           surround_in,  # type: List[int]
                           surround_out  # type: List[int]
                           ) ->np.ndarray:
    """Generates a multi-channel center surround matrix. Useful for isolating or enhancing edges.

    Note: center-surround tensors with 11 or more dimensions may take a while to generate. Make sure you cache those.

    :param ndim: number of dimensions
    :param center_in: input tensor of ints representing colors to look for in the center
    :param center_out: input tensor representing colors to output when more center is detected
    :param surround_in: tensor representing colors to look for outside of center
    :param surround_out: tensor representing colors to output when more surround color is detected
    """
    assert ndim >= 1

    center_surround = np.ndarray(shape=[3 for _ in range(ndim)] + [len(center_in), len(center_out)])

    total = 0
    for tup in itertools.product(*[range(3) for _ in range(ndim)]):
        inv_manhattan_dist = sum([abs(t - 1) for t in tup])
        if inv_manhattan_dist == 0:
            center_surround[tup] = [[0 for _ in center_out] for _ in center_in]
        else:
            euclidian_dist = 1. / np.sqrt(inv_manhattan_dist)
            center_surround[tup] = [[o * i * euclidian_dist for o in surround_out] for i in surround_in]
            total += euclidian_dist
    center_index = tuple([1 for _ in range(ndim)])
    center_surround[center_index] = [[o * i * total for o in center_out] for i in center_in]
    return center_surround


def get_edge_detector_kernel(dimensions: int, channels: int) -> np.ndarray:
    """Returns a tensor that can convolve a color image for edge detection."""
    d = 1.
    out = np.sum(np.array([
        center_surround_tensor(
            dimensions,
            center_in=[d if c_s == s else 0 for c_s in range(channels)],
            center_out=[d if c_s == s else 0 for c_s in range(channels)],
            surround_in=[d if c_s == s else 0 for c_s in range(channels)],
            surround_out=[-d if c_s == s else 0 for c_s in range(channels)])
        for s in range(channels)
    ]), axis=0)

    out = out.astype(np.float32)
    out = out.transpose([3, 4, 0, 1, 2])  # match our system

    return out
