# Loads all the kernels and creates type hinted python methods for them

from typing import Iterable, Generator
import os

import cupy as cp


_cuda_src_dir = os.path.join(os.path.dirname(__file__), "src")

def _load_kernel(filename: str, func_name: str) -> cp.RawKernel:
    path = os.path.join(_cuda_src_dir, filename)
    with open(path) as f:
        code = f.read()
    return cp.RawKernel(code, func_name)

_nv12_to_rgb_kernel = _load_kernel("pixfmt.cu", "nv12_to_rgb")
_rgb_to_nv12_kernel = _load_kernel("pixfmt.cu", "rgb_to_nv12")



def nv12_to_rgb(nv12_frame: cp.ndarray, width: int, height: int) -> cp.ndarray:
    y_plane = nv12_frame[:width * height].reshape((height, width))
    uv_plane = nv12_frame[width * height:].reshape((height // 2, width))

    rgb = cp.empty((height, width, 3), dtype=cp.uint8)

    threads = (16, 16)
    blocks = ((width + threads[0] - 1) // threads[0],
              (height + threads[1] - 1) // threads[1])

    _nv12_to_rgb_kernel(
        blocks, threads,
        (
            y_plane.ravel(),
            uv_plane.ravel(),
            rgb.ravel(),
            cp.int32(width),
            cp.int32(height)
        )
    )
    return rgb



def rgb_to_nv12(rgb: cp.ndarray, width: int, height: int) -> cp.ndarray:
    y_plane = cp.empty((height, width), dtype=cp.uint8)
    uv_plane = cp.zeros((height // 2, width), dtype=cp.uint8)

    threads = (16, 16)
    blocks = ((width + threads[0] - 1) // threads[0],
              (height + threads[1] - 1) // threads[1])

    _rgb_to_nv12_kernel(
        blocks, threads,
        (
            rgb.ravel(),
            y_plane.ravel(),
            uv_plane.ravel(),
            cp.int32(width),
            cp.int32(height)
        )
    )

    return cp.concatenate((y_plane.ravel(), uv_plane.ravel()))



def pipe_nv12_to_rgb(frames: Iterable[cp.ndarray], width: int, height: int) -> Generator[cp.ndarray, None, None]:
    for nv12_frame in frames:
        rgb_frame = nv12_to_rgb(nv12_frame, width, height)
        yield rgb_frame

def pipe_rgb_to_nv12(frames: Iterable[cp.ndarray], width: int, height: int) -> Generator[cp.ndarray, None, None]:
    for rgb_frame in frames:
        nv12_frame = rgb_to_nv12(rgb_frame, width, height)
        yield nv12_frame