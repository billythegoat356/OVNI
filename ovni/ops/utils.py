import cupy as cp

from .filter import round_corners, gaussian_blur


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
    shape = cp.zeros((height, width, 3), dtype=cp.uint8)
    shape = round_corners(shape, corner_radius) # this adds the alpha channel

    # Create padded shape & overlay the rounded temp shape's
    # We only operate on the alpha channel
    padded_shape = cp.zeros((height + pad * 2, width + pad * 2, 1), dtype=cp.uint8)
    padded_shape[pad:height+pad, pad:width+pad, :] = shape[..., 3:4]

    # Blur it
    padded_shape = gaussian_blur(padded_shape, blur)

    # Use it as a mask on another image in order to color it
    final_image = cp.zeros((height + pad * 2, width + pad * 2, 4), dtype=cp.uint8)
    
    final_image[..., 0] = color[0]  # R
    final_image[..., 1] = color[1]  # G
    final_image[..., 2] = color[2]  # B

    final_image[..., 3:4] = padded_shape * (alpha / 255)  # A

    return final_image
