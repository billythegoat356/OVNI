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
