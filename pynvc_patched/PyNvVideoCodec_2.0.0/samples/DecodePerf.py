# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from struct import pack
import sys
import os
import argparse
from pathlib import Path
from enum import Enum
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import numpy as np
from pycuda.compiler import SourceModule
import time
import pycuda.autoinit
import threading
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Global variables for tracking performance metrics
total_fps = 0
num_frames = 0


def decode_parallel(cuda_ctx, cuda_stream, gpu_id, enc_file_path, frame_count=None):
    """
    Function to decode media file and measure performance in a thread.

    Parameters:
        cuda_ctx: CUDA context for the thread
        cuda_stream: CUDA stream for the thread
        gpu_id (int): Ordinal of GPU to use [Parameter not in use]
        enc_file_path (str): Path to file to be decoded
        frame_count (int, optional): Maximum number of frames to decode. If None, 0, or negative, decode all frames.

    Returns:
        None
    """
    global total_fps, num_frames
    
    cuda_ctx.push()

    try:
        if frame_count is None or frame_count <= 0:
            print(f"Decoding all available frames")
            frame_count = None  # Reset to None to decode all frames
        nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
        nv_dec = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=nv_dmx.GetNvCodecId(),
            cudacontext=cuda_ctx.handle,
            cudastream=cuda_stream.handle,
            usedevicememory=True,
            bWaitForSessionWarmUp=True
        )
        
        start = time.perf_counter()
        num_decoded_frames = 0

        for packet in nv_dmx:
            for _ in nv_dec.Decode(packet):
                num_decoded_frames += 1
                if frame_count is not None and frame_count > 0 and num_decoded_frames >= frame_count:
                    print(f"Thread {threading.current_thread().name} reached requested frame count: {frame_count}")
                    break
            
            if frame_count is not None and frame_count > 0 and num_decoded_frames >= frame_count:
                break
        
        if frame_count is not None and frame_count > 0 and num_decoded_frames < frame_count:
            print(f"Video ended before reaching requested frame count for thread {threading.current_thread().name}. Decoded {num_decoded_frames} frames")
        
        # Synchronize stream after all decoding is done
        cuda_stream.synchronize()
        
        # Calculate accurate elapsed time by subtracting initialization time
        elapsed_time = time.perf_counter() - start
        init_time = nv_dec.GetSessionInitTime()
        init_time /= 1000.00  # Convert to seconds
        elapsed_time -= init_time
        
        print(f"Thread {threading.current_thread().name} decoded {num_decoded_frames} frames in {elapsed_time:.2f} seconds")
        
        # Update performance metrics with thread-safe operation
        with threading.Lock():
            global total_fps, num_frames
            total_fps += (num_decoded_frames / elapsed_time)
            num_frames += num_decoded_frames
        
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateDecoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cuda_ctx.pop()


def run_parallel_decode(N, gpu_id, enc_file_path, frame_count=None):
    """
    This function demonstrates decoding using multiple threads.

    The application creates multiple python threads and runs a different decoding session on each thread.
    The number of sessions can be controlled by the CLI option "--num_threads".
    The application supports decoding frames in device memory and provides detailed performance metrics.

    Parameters:
        N (int): Number of parallel decoding sessions to run
        gpu_id (int): Ordinal of GPU to use [Parameter not in use]
        enc_file_path (str): Path to file to be decoded

    Returns:
        None
    """
    global total_fps, num_frames
    total_fps = 0
    num_frames = 0
    threads = []
    
    try:    
        cuda_device = cuda.Device(gpu_id)
        cuda_ctx = cuda_device.make_context() 
        cuda_stream = cuda.Stream()
        nvc.PyNvDecoder.SetSessionCount(N)
            
        for i in range(0, N):
            t = threading.Thread(
                target=decode_parallel,
                args=(cuda_ctx, cuda_stream, gpu_id, enc_file_path, frame_count)
            )
            print(f"Started decode thread {i+1}/{N}")
            t.start()
            threads.append(t)
            

        for t in threads:
            t.join()
            
        cuda_ctx.pop()
        cuda_ctx.detach()

        # Print performance summary
        print("\n--- Performance Summary ---")
        print(f"Number of threads: {N}")
        print(f"Total frames decoded: {num_frames}")
        print(f"Total FPS: {total_fps:.2f}")
        print(f"Average FPS per thread: {total_fps/N:.2f}")
        
    except Exception as e:
        print(f"Error in parallel decode: {e}")
        if 'cuda_ctx' in locals():
            cuda_ctx.pop()
            cuda_ctx.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application demonstrates video decoding using multiple threads."
    )
    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=0,
        help="Check nvidia-smi for available GPUs. Default: 0"
    )
    parser.add_argument(
        "-i", "--encoded_file_path",
        type=Path,
        required=True,
        help="Encoded video file (read from)"
    )
    parser.add_argument(
        "-n", "--num_threads",
        default=1,
        type=int,
        help="Number of parallel decode threads to run. Default: 1"
    )
    parser.add_argument(
        "-f", "--frame_count",
        type=int,
        help="Maximum number of frames to decode. If not provided, all frames will be decoded."
    )

    args = parser.parse_args()

    run_parallel_decode(
        args.num_threads,
        args.gpu_id,
        args.encoded_file_path.as_posix(),
        args.frame_count
    )
