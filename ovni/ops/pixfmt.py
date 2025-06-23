from typing import Iterable, Generator

import cupy as cp

from ..kernels import Kernels, THREADS, make_blocks



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

    blocks = make_blocks(width, height)

    # Call the kernel
    Kernels.nv12_to_rgb(
        blocks, THREADS,
        (
            y_plane,
            uv_plane,
            rgb_frame.ravel(),
            cp.int32(width),
            cp.int32(height)
        )
    )
    return rgb_frame



def rgb_to_nv12(rgb_frame: cp.ndarray) -> cp.ndarray:
    """
    Converts a frame from RGB pixel format to *flattened* NV12
    You usually want that for encoding

    Parameters:
        rgb_frame: cp.ndarray

    Returns:
        cp.ndarray
    """
    # Data is not flattened, we can access width and height with the shape
    width = rgb_frame.shape[1]
    height = rgb_frame.shape[0]

    # Create empty cupy arrays for Y and UV planes
    y_plane = cp.empty((height * width), dtype=cp.uint8)
    uv_plane = cp.zeros((height // 2 * width), dtype=cp.uint8)

    blocks = make_blocks(width, height)

    # Call the kernel
    Kernels.rgb_to_nv12(
        blocks, THREADS,
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

def pipe_rgb_to_nv12(frames: Iterable[cp.ndarray]) -> Generator[cp.ndarray, None, None]:
    """
    Pipes frames from RGB pixel format to *flattened* NV12

    Parameters:
        frames: Iterable[cp.ndarray] - can take an iterable but you usually want to pass a generator

    Returns:
        Generator[cp.ndarray, None, None]
    """
    for rgb_frame in frames:
        nv12_frame = rgb_to_nv12(rgb_frame)
        yield nv12_frame


