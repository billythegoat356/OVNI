"""
Utility method to handle CUDA context managing easier and time execution
"""

from ovni.base import CudaCtxManager

from typing import Callable
import time



def call(func: Callable[[], None], frames: int, fps: int):
    """
    Wraps the function in a Cuda context manager, and times the execution

    Parameters:
        func: Callable[[], None] - the function to call
        frames: int - amount of frames (for logging)
        fps: int - same for fps
    """
    start = time.time()

    with CudaCtxManager():
        func()

    exec_duration = round(time.time() - start, 2)
    video_duration = frames/fps
    speed = round(video_duration/exec_duration, 2)

    print(f"Took: {exec_duration}s")
    print(f"Speed: x{speed}")
