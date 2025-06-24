import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks



def chroma_key(src: cp.ndarray, color: tuple[int, int, int], min_threshold: int = 0, max_threshold: int = 255) -> cp.ndarray:
    """
    Applies chroma keying on a given array for a given color.
    This is essentially the process of turning a 'green screen' into a transparent overlay.

    Parameters:
        src: cp.ndarray
        color: tuple[int, int, int] - the color that should be used as a chroma key, in RGB format
        min_threshold: int = 0 - the threshold below (and at) which pixels will be considered fully transparent
        max_threshold: int = 255 - the threshold above (and at) which pixels will be considered fully opaque

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
            cp.uint8(color[0]),
            cp.uint8(color[1]),
            cp.uint8(color[2]),
            cp.int32(min_threshold),
            cp.int32(max_threshold),
            cp.int32(width),
            cp.int32(height)
        )
    )

    return rgba_array
