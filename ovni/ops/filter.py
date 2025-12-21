import math

import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks



def chroma_key(src: cp.ndarray, key_color: tuple[int, int, int], transparency_t: int = 150, opacity_t: int = 255) -> cp.ndarray:
    """
    Applies chroma keying on a given array for a given key color.
    This is essentially the process of turning a 'green screen' into a transparent overlay.
    -------
    NOTE for transparency/opacity threshold handling
    These thresholds are compared to a distance computed from the 3 channels

    If you see too many pixels close to the key color -> increase transparency threshold
    If you see too many pixels transparent that shouldn't be -> decrease opacity threshold

    If you see a sharp change from transparency to normal colors -> decrease transparency threshold
    If you see a sharp change from transparency to the key color -> increase opacity threshold


    Parameters:
        src: cp.ndarray
        key_color: tuple[int, int, int] - the color that should be used as a chroma key, in RGB format
        transparency_t: int = 150 - the threshold at and below which pixels are considered fully transparent
        opacity_t: int = 255 - the threshold at and after which pixels are considered fully opaque

    Returns:
        cp.ndarray - a 4 channels RGBA frame with transparency
    """

    width = src.shape[1]
    height = src.shape[0]

    # Destination array
    rgba_array = cp.empty((height, width, 4), dtype=cp.int8)
    rgba_array[:, :, :3] = src
    rgba_array[:, :, 3] = 255 # Make it fully opaque for now

    blocks = make_blocks(width, height)
    Kernels.chroma_key(
        blocks, THREADS,
        (
            rgba_array.ravel(),
            cp.uint8(key_color[0]),
            cp.uint8(key_color[1]),
            cp.uint8(key_color[2]),
            cp.int32(transparency_t),
            cp.int32(opacity_t),
            cp.int32(width),
            cp.int32(height)
        )
    )

    return rgba_array




def round_mask(src: cp.ndarray, radius: int | tuple[int, int, int, int]) -> None:
    """
    Rounds the given array
    NOTE: The array should have only 1 channel
    
    Parameters:
        src: cp.ndarray
        radius: int | tuple[int, int, int, int] - radius number or tuple, top left, top right, bottom right, bottom left
    """
    width = src.shape[1]
    height = src.shape[0]

    if isinstance(radius, int):
        radius = tuple([radius, radius, radius, radius])

    radius = tuple(
        min(rad, int(min(width, height) / 2))
        for rad in radius
    )
    

    blocks = make_blocks(width, height)

    Kernels.round_mask(
        blocks, THREADS,
        (
            src.ravel(),
            cp.int32(radius[0]),
            cp.int32(radius[1]),
            cp.int32(radius[2]),
            cp.int32(radius[3]),
            cp.int32(width),
            cp.int32(height)
        )
    )





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

    assert src.shape[2] == 1, "Gaussian blur only supports one channel";

    blocks = make_blocks(width, height)

    # Write first pass
    dst = cp.zeros((height, width, 1), dtype=cp.uint8)

    Kernels.gb_horizontal(
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

    Kernels.gb_vertical(
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