# Loads all the kernels and creates abstract python methods for them

from typing import Iterable, Generator
import os

import cupy as cp


_cuda_comp_dir = os.path.join(os.path.dirname(__file__), "compiled")


# Loading all functions

mod = cp.RawModule(path=os.path.join(_cuda_comp_dir, "all_kernels.ptx"), backend='ptx')

_nv12_to_rgb_kernel = mod.get_function("nv12_to_rgb")
_rgb_to_nv12_kernel = mod.get_function("rgb_to_nv12")

_scale_translate_kernel = mod.get_function("scale_translate")




def nv12_to_rgb(nv12_frame: cp.ndarray, width: int, height: int) -> cp.ndarray:
    """
    Converts a frame from *flattened* NV12 pixel format to RGB
    You usually want that for easier pixel operations

    Parameters:
        nv12_frame: cp.ndarray
        width: int
        height: int

    Returns:
        cp.ndarray
    """
    # Split into the Y and UV planes
    y_plane = nv12_frame[:width * height]
    uv_plane = nv12_frame[width * height:]

    # Create empty cupy array
    rgb_frame = cp.empty((height, width, 3), dtype=cp.uint8)

    # Define threads and blocks for kernel call
    threads = (16, 16)
    blocks = ((width + threads[0] - 1) // threads[0],
              (height + threads[1] - 1) // threads[1])

    # Call the kernel
    _nv12_to_rgb_kernel(
        blocks, threads,
        (
            y_plane,
            uv_plane,
            rgb_frame.ravel(),
            cp.int32(width),
            cp.int32(height)
        )
    )
    return rgb_frame



def rgb_to_nv12(rgb_frame: cp.ndarray, width: int, height: int) -> cp.ndarray:
    """
    Converts a frame from RGB pixel format to *flattened* NV12
    You usually want that for encoding

    Parameters:
        rgb_frame: cp.ndarray
        width: int
        height: int

    Returns:
        cp.ndarray
    """
    # Create empty cupy arrays for Y and UV planes
    y_plane = cp.empty((height * width), dtype=cp.uint8)
    uv_plane = cp.zeros((height // 2 * width), dtype=cp.uint8)

    # Define threads and blocks for kernel call
    threads = (16, 16)
    blocks = ((width + threads[0] - 1) // threads[0],
              (height + threads[1] - 1) // threads[1])

    # Call the kernel
    _rgb_to_nv12_kernel(
        blocks, threads,
        (
            rgb_frame.ravel(),
            y_plane,
            uv_plane,
            cp.int32(width),
            cp.int32(height)
        )
    )

    # Concatenate the Y and UV planes into the entire NV12 frame
    nv12_frame = cp.concatenate((y_plane, uv_plane))
    return nv12_frame



def pipe_nv12_to_rgb(frames: Iterable[cp.ndarray], width: int, height: int) -> Generator[cp.ndarray, None, None]:
    """
    Pipes frames from *flattened* NV12 pixel format to RGB

    Parameters:
        frames: Iterable[cp.ndarray] - can take an iterable but you usually want to pass a generator
        width: int
        height: int

    Returns:
        Generator[cp.ndarray, None, None]
    """
    for nv12_frame in frames:
        rgb_frame = nv12_to_rgb(nv12_frame, width, height)
        yield rgb_frame

def pipe_rgb_to_nv12(frames: Iterable[cp.ndarray], width: int, height: int) -> Generator[cp.ndarray, None, None]:
    """
    Pipes frames from RGB pixel format to *flattened* NV12

    Parameters:
        frames: Iterable[cp.ndarray] - can take an iterable but you usually want to pass a generator
        width: int
        height: int

    Returns:
        Generator[cp.ndarray, None, None]
    """
    for rgb_frame in frames:
        nv12_frame = rgb_to_nv12(rgb_frame, width, height)
        yield nv12_frame



def scale_translate(src: cp.ndarray, scale: float, tx: int, ty: int, dst_width: int | None = None, dst_height: int | None = None) -> cp.ndarray:
    """
    Applies scale and translation to an image.

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
    _scale_translate_kernel(
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
