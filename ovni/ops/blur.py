import math

import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks




def gaussian_blur(src: cp.ndarray, sigma: float) -> cp.ndarray:
    """
    Performs gaussian blur on the given array with the given sigma
    NOTE: Supports 1 channel or 3

    Parameters:
        src: cp.ndarray
        sigma: float

    Returns:
        cp.ndarray
    """
    
    # Compute radius and weights
    radius = math.ceil(3 * sigma)
    x = cp.arange(-radius, radius + 1, dtype=cp.float32)
    weights = cp.exp(-(x * x) / (2 * sigma * sigma))
    weights /= cp.sum(weights)

    # Move to GPU
    weights_cp = cp.asarray(weights, dtype=cp.float32)

    width = src.shape[1]
    height = src.shape[0]

    blocks = make_blocks(width, height)

    channels = src.shape[2]

    if channels == 1 or channels == 3:

        if channels == 3:
            final_dst = cp.zeros((height, width, 3), dtype=cp.uint8)

        for i in range(channels):
            # Write` first pass
            dst = cp.zeros((height, width, 1), dtype=cp.uint8)

            Kernels.gaussian_blur_horizontal(
                blocks, THREADS,
                (
                    src[:, :, i:i+1].ravel(),
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

            if channels == 1:
                return dst2
            
            else:
                final_dst[:, :, i:i+1] = dst2

        return final_dst

    else:
        raise Exception("Gaussian blur only works on 1 or 3 channels")

    