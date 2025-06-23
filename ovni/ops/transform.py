import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks






def scale_translate(src: cp.ndarray, scale: float, tx: int, ty: int, dst_width: int | None = None, dst_height: int | None = None) -> cp.ndarray:
    """
    Applies scale and translation to an image.
    Uses bilinear interpolation

    Parameters:
        src: cp.ndarray (H x W x 3), dtype=uint8
        scale: float
        tx: int
        ty: int
        dst_width: int | None = None - if None, uses the same as input
        dst_height: int | None = None - ...

    Returns:
        dst: cp.ndarray (dst_height x dst_width x 3), dtype=uint8
    """

    src_width = src.shape[1]
    src_height = src.shape[0]

    if dst_width is None:
        dst_width = src_width

    if dst_height is None:
        dst_height = src_height

    # Create output array
    dst = cp.empty((dst_height, dst_width, 3), dtype=cp.uint8)

    blocks = make_blocks(dst_width, dst_height)

    # Call kernel
    Kernels.scale_translate(
        blocks, THREADS,
        (
            src.ravel(),
            cp.int32(src_width),
            cp.int32(src_height),
            dst.ravel(),
            cp.int32(dst_width),
            cp.int32(dst_height),
            cp.float32(float(scale)),
            cp.int32(tx),
            cp.int32(ty)
        )
    )
    return dst


def scale(src: cp.ndarray, scale: float, dst_width: int | None = None, dst_height: int | None = None) -> cp.ndarray:
    """
    Applies scaling to an image.
    Uses bilinear interpolation
    (essentially calls `scale_translate` with no translation)

    Parameters:
        src: cp.ndarray (H x W x 3), dtype=uint8
        scale: float
        dst_width: int | None = None - if None, uses the same as input
        dst_height: int | None = None - ...

    Returns:
        dst: cp.ndarray (dst_height x dst_width x 3), dtype=uint8
    """

    return scale_translate(
        src=src,
        scale=scale,
        tx=0,
        ty=0,
        dst_width=dst_width,
        dst_height=dst_height
    )


def resize(src: cp.ndarray, dst_width: int, dst_height: int) -> cp.ndarray:
    """
    Resizes into a destination width and height.
    Applies bilinear interpolation.

    Parameters:
        src: cp.ndarray
        dst_width: int
        dst_height: int

    Returns:
        cp.ndarray
    """

    src_width = src.shape[1]
    src_height = src.shape[0]

    # Create output array
    dst = cp.empty((dst_height, dst_width, 3), dtype=cp.uint8)

    blocks = make_blocks(dst_width, dst_height)

    Kernels.resize(
        blocks, THREADS,
        (
            src.ravel(),
            cp.int32(src_width),
            cp.int32(src_height),
            dst.ravel(),
            cp.int32(dst_width),
            cp.int32(dst_height)
        )
    )
    return dst

