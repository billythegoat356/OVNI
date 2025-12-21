import cupy as cp

from .filter import round_mask
from .blur import gaussian_blur




def round_corners(src: cp.ndarray, radius: int | tuple[int, int, int, int]) -> cp.ndarray:
    """
    Rounds the corners of the source array
    Returns the new array
    
    Parameters:
        src: cp.ndarray
        radius: int | tuple[int, int, int, int] - radius number or tuple, top left, top right, bottom right, bottom left

    Returns:
        cp.ndarray
    """

    width = src.shape[1]
    height = src.shape[0]

    dst = src.copy()

    if dst.shape[2] == 3:
        # Add alpha channel
        n_dst = cp.empty((height, width, 4), dtype=src.dtype)
        n_dst[..., :3] = src
        n_dst[..., 3] = 255

        dst = n_dst

    mask = dst[:, :, 3:4].copy()

    round_mask(mask, radius)

    dst[:, :, 3:4] = mask

    return dst


def make_shadow(
    width: int,
    height: int,
    corner_radius: int,
    blur: int,
    color: tuple[int, int, int] = (0, 0, 0),
    alpha: int = 122
) -> cp.ndarray:
    """
    Makes a shadow with the given params

    Parameters:
        width: int
        height: int
        corner_radius: int
        blur: int
        color: tuple[int, int, int] = (0, 0, 0)
        alpha: int = 122
    """

    # We need to pad otherwise the shadow looks cut off
    pad = blur*2 # Heuristic

    # Create a temp shape just to round the corners
    shape = cp.full((height, width, 1), alpha, dtype=cp.uint8)
    round_mask(shape, corner_radius)

    # Create padded shape & overlay the rounded temp shape's
    # We only operate on the alpha channel
    padded_shape = cp.zeros((height + pad * 2, width + pad * 2, 1), dtype=cp.uint8)
    padded_shape[pad:height+pad, pad:width+pad, :] = shape

    # Blur it
    padded_shape = gaussian_blur(padded_shape, blur)

    # Use it as a mask on another image in order to color it
    final_image = cp.zeros((height + pad * 2, width + pad * 2, 4), dtype=cp.uint8)
    
    final_image[..., 0] = color[0]  # R
    final_image[..., 1] = color[1]  # G
    final_image[..., 2] = color[2]  # B

    final_image[..., 3:4] = padded_shape 

    return final_image
