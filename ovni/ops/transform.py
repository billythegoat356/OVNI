import cupy as cp

from ..kernels.loader import Kernels





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

    # Define threads and blocks
    threads = (16, 16)
    blocks = ((dst_width + threads[0] - 1) // threads[0],
              (dst_height + threads[1] - 1) // threads[1])

    # Call kernel
    Kernels.scale_translate(
        blocks, threads,
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
