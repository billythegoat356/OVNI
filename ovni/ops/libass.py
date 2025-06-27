from typing import TYPE_CHECKING

import ctypes
import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks

if TYPE_CHECKING:
    from ..ass.lib import ASS_Image




def blend_ass_image(dst: cp.ndarray, img: 'ASS_Image') -> None:
    """
    Blend an ASS image on a destination array.
    Takes the full ASS image struct with its defined attributes

    Parameters:
        dst: cp.ndarray
        img: ASS_Image

    Returns:
        None
    """
    blocks = make_blocks(img.w, img.h)

    # Define width and height of dst array (needed for coordinates calculation)
    dst_h = dst.shape[0]
    dst_w = dst.shape[1]

    # Create buffer with correct type and load bitmap
    buf_type = ctypes.c_ubyte * (img.h * img.stride)
    buf = buf_type.from_address(ctypes.addressof(img.bitmap.contents))

    # Load into 1D cupy array
    bitmap = cp.frombuffer(buf, dtype=cp.uint8)

    Kernels.blend_ass_image(
        blocks, THREADS,
        (
            cp.int32(img.w), cp.int32(img.h),
            cp.int32(img.stride),
            bitmap,
            cp.uint32(img.color),
            cp.int32(img.dst_x), cp.int32(img.dst_y),
            cp.int32(dst_w), cp.int32(dst_h),
            dst.ravel()
        )
    )