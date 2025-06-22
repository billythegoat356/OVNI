# Loads all the kernels and creates abstract python methods for them

from typing import Iterable, Generator
import os

import cupy as cp


_cuda_src_dir = os.path.join(os.path.dirname(__file__), "src")

def _load_kernel(filename: str, func_name: str) -> cp.RawKernel:
    """
    Given a filename and a function name, loads that kernel

    Parameters:
        filename: str
        func_name: str

    Returns:
        cp.RawKernel
    """

    path = os.path.join(_cuda_src_dir, filename)
    with open(path) as f:
        code = f.read()
    return cp.RawKernel(code, func_name)


# Loading all kernels

_nv12_to_rgb_kernel = _load_kernel("pixfmt.cu", "nv12_to_rgb")
_rgb_to_nv12_kernel = _load_kernel("pixfmt.cu", "rgb_to_nv12")

_warp_affine_kernel = _load_kernel("warp.cu", "warp_affine")




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



def warp_affine(src: cp.ndarray, affine: cp.ndarray, dst_width: int, dst_height: int) -> cp.ndarray:
    """
    Applies warp affine transform to an RGB image.

    Parameters:
        src: cp.ndarray (H x W x 3), dtype=uint8
        affine: cp.ndarray shape (6,), dtype=float32 (2x3 affine matrix flattened)
        dst_width: int
        dst_height: int

    Returns:
        dst: cp.ndarray (dst_height x dst_width x 3), dtype=uint8
    """
    assert src.dtype == cp.uint8
    assert src.ndim == 3 and src.shape[2] == 3
    assert affine.shape == (6,)
    dst = cp.empty((dst_height, dst_width, 3), dtype=cp.uint8)

    threads = (16, 16)
    blocks = ((dst_width + threads[0] - 1) // threads[0],
              (dst_height + threads[1] - 1) // threads[1])

    _warp_affine_kernel(
        blocks, threads,
        (
            src.ravel(),
            cp.int32(src.shape[1]),  # srcWidth
            cp.int32(src.shape[0]),  # srcHeight
            dst.ravel(),
            cp.int32(dst_width),
            cp.int32(dst_height),
            affine
        )
    )
    return dst
