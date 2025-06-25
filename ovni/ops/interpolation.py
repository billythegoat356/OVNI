import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks


def bilinear_blend(
        top_left: cp.ndarray,
        top_right: cp.ndarray,
        bottom_left: cp.ndarray,
        bottom_right: cp.ndarray,
        x_distance: float,
        y_distance: float
) -> cp.ndarray:
    """
    Use bilinear interpolation between 4 images, with a specific distance for each axis.
    Note that the distance should be a float between 0 and 1, 0 meaning that the first matrix is the same, 1 meaning that its closer to the second other one.
    
    Parameters:
        top_left: cp.ndarray
        top_right: cp.ndarray
        bottom_left: cp.ndarray
        bottom_right: cp.ndarray
        x_distance: float - 0 -> 1 
        y_distance: float - 0 -> 1 

    Returns:
        cp.ndarray
    """
    width = top_left.shape[1]
    height = top_left.shape[0]

    # Create output array
    dst = cp.empty((height, width, 3), dtype=cp.uint8)

    blocks = make_blocks(width, height)

    Kernels.bilinear_blend(
        blocks, THREADS,
        (
            dst.ravel(),
            top_left.ravel(),
            top_right.ravel(),
            bottom_left.ravel(),
            bottom_right.ravel(),
            cp.int32(width),
            cp.int32(height),
            cp.float32(x_distance),
            cp.float32(y_distance)
        )
    )
    return dst




def linear_blend(
        img1: cp.ndarray,
        img2: cp.ndarray,
        distance: float
) -> cp.ndarray:
    """
    Use linear interpolation between 2 images, with a specific distance
    Note that the distance should be a float between 0 and 1, 0 meaning that the first matrix is the same, 1 meaning that its closer to the second other one.
    
    Parameters:
        img1: cp.ndarray
        img2: cp.ndarray
        distance: float - 0 -> 1 

    Returns:
        cp.ndarray
    """
    width = img1.shape[1]
    height = img1.shape[0]

    # Create output array
    dst = cp.empty((height, width, 3), dtype=cp.uint8)

    blocks = make_blocks(width, height)

    Kernels.linear_blend(
        blocks, THREADS,
        (
            dst.ravel(),
            img1.ravel(),
            img2.ravel(),
            cp.int32(width),
            cp.int32(height),
            cp.float32(distance)
        )
    )
    return dst
