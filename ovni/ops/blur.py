import math

import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks




def gaussian_blur(src: cp.ndarray, sigma: float) -> cp.ndarray:
    """
    Performs gaussian blur on the given array with the given sigma
    NOTE: Only works on arrays with 1 channel

    Parameters:
        src: cp.ndarray
        sigma: float

    Returns:
        cp.ndarray
    """
    
    assert src.shape[2] == 1, "Gaussian blur only supports one channel"

    # Compute radius and shape
    radius = math.ceil(3 * sigma)
    weights_size = 2 * radius + 1
    weights = []
    for i in range(weights_size):
        i -= radius
        val = math.exp(-(i*i) / (2*sigma*sigma))
        weights.append(val)

    # Normalize
    weights = [
        val / sum(weights)
        for val in weights
    ]
    # Move to GPU
    weights_cp = cp.asarray(weights, dtype=cp.float32)

    width = src.shape[1]
    height = src.shape[0]

    blocks = make_blocks(width, height)

    # Write first pass
    dst = cp.zeros((height, width, 1), dtype=cp.uint8)

    Kernels.gaussian_blur_horizontal(
        blocks, THREADS,
        (
            src.ravel(),
            dst.ravel(),
            weights_cp,
            cp.int32(radius),
            cp.int32(width),
            cp.int32(height)
        )
    )

    # Write second pass
    dst2 = cp.zeros((height, width, 1), dtype=cp.uint8)

    Kernels.gaussian_blur_vertical(
        blocks, THREADS,
        (
            dst.ravel(),
            dst2.ravel(),
            weights_cp,
            cp.int32(radius),
            cp.int32(width),
            cp.int32(height)
        )
    )

    return dst2