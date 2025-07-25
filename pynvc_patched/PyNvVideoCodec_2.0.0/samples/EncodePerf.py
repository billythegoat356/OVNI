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

import argparse
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import time
import mmap
import json
import threading
import numpy as np
import gc
import sys
import atexit
import os
from os.path import join, dirname, abspath
from Utils import AppFramePerf

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# Global variables for tracking performance
total_frames = 0

# Global variables to track resources for cleanup
g_devicedata = None
g_mmap = None
g_context = None


def cleanup_resources():
    """
    Clean up global resources including memory map, GPU memory, and CUDA context.
    
    This function is registered with atexit to ensure proper cleanup of resources
    when the program exits.
    """
    global g_devicedata, g_mmap, g_context
    
    # Force garbage collection first
    gc.collect()
    
    # Close memory map
    if g_mmap:
        try:
            g_mmap.close()
            g_mmap = None
        except:
            pass
    
    # Free GPU memory
    if g_devicedata:
        try:
            g_devicedata.free()
            g_devicedata = None
        except:
            pass
    
    # Clean up CUDA context
    if g_context:
        try:
            g_context.detach()
            g_context = None
        except:
            pass


# Register the cleanup function to run at exit
atexit.register(cleanup_resources)


def encode(thread_id, dec_file_path, encoder, devicedata, framesize, width, height, fmt, frame_count=1000):
    """
    Thread function that encodes frames and updates the global counter.

    Parameters:
        thread_id (int): ID of the thread
        dec_file_path (str): Path to the input file
        encoder: NVENC encoder instance
        devicedata: CUDA device memory handle
        framesize (int): Size of each frame in bytes
        width (int): Width of the frame
        height (int): Height of the frame
        fmt (str): Surface format string in uppercase (e.g., "NV12")
        frame_count (int, optional): Number of frames to encode. If None, 0, or negative, encode all frames from input.
    """
    global total_frames
    
    try:
        print(f"Starting encode thread {thread_id}")

        # Calculate total available frames from file size
        file_size = os.path.getsize(dec_file_path)
        available_frames = file_size // framesize

        if frame_count is None or frame_count == 0:
            print(f"Thread {thread_id} will process all available frames")
            frame_count = available_frames
        elif frame_count < 0 or frame_count > available_frames:
            print(f"Warning: Invalid frame count {frame_count}. Thread {thread_id} will process all available frames.")
            frame_count = available_frames
        
        # Encode frames
        for i in range(frame_count):
            # Create a new frame object for each frame
            input_gpu_frame = AppFramePerf(width, height, fmt, devicedata, i)
            
            # Encode the frame
            encoder.Encode(input_gpu_frame)
                
        # Flush encoder
        encoder.EndEncode()
            
        print(f"Thread {thread_id} encoded {frame_count} frames")

        # Update global counter
        with threading.Lock():
            total_frames += frame_count
        
    except Exception as e:
        print(f"Thread {thread_id}: An unexpected error occurred: {e}")


def GetFrameSize(width, height, surface_format):
    """
    Calculate the size of a frame in bytes based on its dimensions and format.

    Parameters:
        width (int): Width of the frame
        height (int): Height of the frame
        surface_format (str): Format of the surface (e.g., "NV12", "ARGB", "YUV444")

    Returns:
        int: Size of the frame in bytes
    """
    if surface_format in ("ARGB", "ABGR"):
        return width * height * 4
    elif surface_format == "YUV444":
        return width * height * 3
    elif surface_format == "YUV420":
        return int(width * height * 3 / 2)
    elif surface_format == "P010":
        return int(width * height * 3 / 2 * 2)
    elif surface_format == "YUV444_16BIT":
        return int(width * height * 3 * 2)
    elif surface_format == "NV16":
        return width * height * 2
    elif surface_format == "P210":
        return width * height * 2 * 2
    else:
        return int(width * height * 3 / 2)


def run_parallel_encode(num_threads, gpu_id, dec_file_path, width, height, fmt, config_params, frame_count):
    """
    Run encoding test with multiple threads.

    Parameters:
        num_threads (int): Number of parallel threads to use
        gpu_id (int): Ordinal of GPU to use
        dec_file_path (str): Path to the input file
        width (int): Width of the frame
        height (int): Height of the frame
        fmt (str): Surface format string in uppercase (e.g., "NV12")
        config_params (dict): Key-value pairs providing fine-grained control on encoding
        frame_count (int): Number of frames to encode. If 0, encodes all frames from input.

    Returns:
        None: Prints performance metrics to console
    """
    print(f"Initializing {num_threads} encode threads")
    global g_devicedata, g_mmap, g_context, total_frames
    
    # Reset global counter
    total_frames = 0
    
    g_context = None
    try:
        # Initialize CUDA
        cuda.init()
        cuda_device = cuda.Device(gpu_id)
        
        # Create a CUDA context
        g_context = cuda_device.make_context()
        
        # Create a CUDA stream
        shared_stream = cuda.Stream()
        # Open input file and prepare data
        with open(dec_file_path, "rb") as dec_file:
            # Calculate frame size based on format
            framesize = GetFrameSize(width, height, fmt)
            
            # Read data into memory and copy to GPU
            if sys.platform == 'win32':
                g_mmap = mmap.mmap(dec_file.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                g_mmap = mmap.mmap(dec_file.fileno(), 0, prot=mmap.PROT_READ)
            hostdata = g_mmap.read(g_mmap.size())
            g_devicedata = cuda.mem_alloc(g_mmap.size())
            cuda.memcpy_htod(g_devicedata, hostdata)
            
            # Create encoders for all threads
            encoders = []
            for i in range(num_threads):
                # Create a copy of config_params for each encoder
                config_copy = config_params.copy()
                config_copy["gpu_id"] = gpu_id
                # Create encoder
                encoder = nvc.CreateEncoder(
                    width, height, fmt, False,
                    cudacontext=g_context.handle,
                    cudastream=shared_stream.handle,
                    **config_copy
                )
                encoders.append(encoder)
            
            # Start timing before encoding
            begin = time.perf_counter()
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                t = threading.Thread(
                    target=encode,
                    args=(
                        i,
                        dec_file_path,
                        encoders[i],
                        g_devicedata,
                        framesize,
                        width,
                        height,
                        fmt,
                        frame_count
                    )
                )
                threads.append(t)
                t.start()
            
            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Stop timing after all threads complete
            end = time.perf_counter()
            
            duration = end - begin  # Time in seconds
            
            # Print summary
            print("\n--- Performance Summary ---")
            print(f"Number of threads: {num_threads}")
            print(f"Total frames encoded: {total_frames}")
            if duration > 0:
                fps = total_frames / duration
                print(f"Duration: {duration:.2f} seconds")
                print(f"Total FPS: {fps:.2f}")
                print(f"Average FPS per thread: {fps / num_threads:.2f}")
            
            # Clear encoder references
            for encoder in encoders:
                del encoder
            encoders.clear()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Pop the context
        if g_context is not None:
            g_context.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application measures encoding performance in FPS with multiple threads."
    )
    
    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=0,
        help="Check nvidia-smi for available GPUs. Default: 0"
    )
    parser.add_argument(
        "-i", "--raw_file_path",
        type=Path,
        required=True,
        help="Raw video file (read from)"
    )
    parser.add_argument(
        "-s", "--size",
        type=str,
        required=True,
        help="WidthxHeight of raw frame (e.g., 1920x1080)"
    )
    parser.add_argument(
        "-if", "--format",
        type=str,
        default="NV12",
        help="Format of input file. Default is NV12"
    )
    parser.add_argument(
        "-c", "--codec",
        type=str,
        help="Video codec (HEVC, H264, AV1). If not specified, uses the codec from config file."
    )
    parser.add_argument(
        "-json", "--config_file",
        type=str,
        default=join(dirname(__file__), "encode_config_perf.json"),
        help="Path of JSON config file (default: encode_config_perf.json in the same directory)"
    )
    parser.add_argument(
        "-n", "--num_threads",
        type=int,
        default=1,
        help="Number of parallel threads. Default is 1"
    )
    parser.add_argument(
        "-f", "--frame_count",
        type=int,
        default=0,
        help="Number of frames to encode (0 for all frames). Default is 0."
    )
    
    args = parser.parse_args()
    config = {}
    
    if args.config_file:
        with open(args.config_file) as jsonFile:
            config = json.loads(jsonFile.read())
            config["preset"] = config["preset"].upper()
    
    # Only set codec from command line if it was explicitly provided
    if args.codec is not None:
        args.codec = args.codec.lower()
        config["codec"] = args.codec
    
    args.format = args.format.upper()
    width, height = map(int, args.size.split("x"))
    
    try:
        run_parallel_encode(
            args.num_threads,   
            args.gpu_id,
            args.raw_file_path.as_posix(),
            width,
            height,
            args.format,
            config,
            args.frame_count
        )
    finally:
        # Force cleanup before exit
        cleanup_resources()