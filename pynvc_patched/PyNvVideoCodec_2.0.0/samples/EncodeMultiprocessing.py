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
from pycuda.autoinit import context
import mmap
import json
import multiprocessing
from multiprocessing import Process
import os
import sys
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import AppFramePerf


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
    else:
        return int(width * height * 3 / 2)


def encode_process(gpu_id, dec_file_path, width, height, fmt, h, framesize, config_params, frame_count=None):
    """
    Function to encode frames in a separate process.

    This function demonstrates how to set up and run video encoding in a separate process,
    showing proper handling of CUDA contexts and device memory in a multiprocessing environment.

    Parameters:
        gpu_id (int): Ordinal of GPU to use
        dec_file_path (str): Path to file to be decoded
        width (int): Width of encoded frame
        height (int): Height of encoded frame
        fmt (str): Surface format string in uppercase (e.g., "NV12")
        h: IPC memory handle for CUDA device memory
        framesize (int): Size of each frame in bytes
        config_params (dict): Key-value pairs providing fine-grained control on encoding
        frame_count (int, optional): Number of frames to encode. If None, 0, or negative, encode all frames from input.

    Returns:
        None
    """
    try:
        process_id = multiprocessing.current_process().name
        print(f"Starting encode process {process_id}")
        
        config_params["gpu_id"] = gpu_id
        nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)
        devicedata = cuda.IPCMemoryHandle(h)
        input_gpu_frame = AppFramePerf(width, height, fmt, devicedata, 0)
        
        # Calculate total available frames from file size
        file_size = os.path.getsize(dec_file_path)
        available_frames = file_size // framesize
        
        if frame_count is None or frame_count == 0:
            print(f"Encoding all available frames")
            frame_count = available_frames
        elif frame_count < 0 or frame_count > available_frames:
            print(f"Warning: Invalid frame count {frame_count}. Process {process_id} will process all available frames.")
            frame_count = available_frames 
        
        frames_encoded = 0
        for i in range(frame_count):
            input_gpu_frame.gpuAlloc = int(devicedata) + (i * framesize)
            bitstream = nvenc.Encode(input_gpu_frame)
            frames_encoded += 1
    
        # Flush encoder queue
        bitstream = nvenc.EndEncode()
        print(f"Process {process_id} completed encoding {frames_encoded} frames")

    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateEncoder failure in process {process_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in process {process_id}: {e}")


def run_parallel_encode(num_processes, gpu_id, enc_file_path, width, height, fmt, framesize, config, frame_count=None):
    """
    Demonstrates parallel video encoding using multiple processes.

    This function shows how to:
    1. Set up shared device memory for multiple encoder processes
    2. Create and manage multiple encoding processes
    3. Handle CUDA IPC memory in a multiprocessing context
    4. Coordinate multiple encoder instances running in parallel

    Parameters:
        num_processes (int): Number of parallel encoding processes to run
        gpu_id (int): Ordinal of GPU to use
        enc_file_path (str): Path to file to be encoded
        width (int): Width of encoded frame
        height (int): Height of encoded frame
        fmt (str): Surface format string
        framesize (int): Size of each frame in bytes
        config (dict): Encoder configuration parameters
        frame_count (int, optional): Number of frames for each process to encode
    """
    processes = []
    
    print(f"Initializing {num_processes} encoder processes")
    
    with open(enc_file_path, "rb") as decFile:
        # Cross-platform memory mapping
        if sys.platform == 'win32':
            m = mmap.mmap(decFile.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            m = mmap.mmap(decFile.fileno(), 0, prot=mmap.PROT_READ)
            
        hostdata = m.read(m.size())
        devicedata = cuda.mem_alloc(m.size())
        cuda.memcpy_htod(devicedata, hostdata)
        devptrhandle = cuda.mem_get_ipc_handle(devicedata)

        for i in range(num_processes):
            p = Process(
                target=encode_process,
                args=(
                    gpu_id,
                    enc_file_path,
                    width,
                    height,
                    fmt,
                    devptrhandle,
                    framesize,
                    config,
                    frame_count
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
    print("All encoding processes completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application demonstrates video encoding using multiple processes."
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
        help="Raw video file to encode"
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
        help="Format of input file. Default: NV12"
    )
    parser.add_argument(
        "-c", "--codec",
        type=str,
        default="H264",
        help="Video codec (HEVC, H264, AV1). Default: H264"
    )
    parser.add_argument(
        "-json", "--config_file",
        type=str,
        default=join(dirname(__file__), "encode_config_perf.json"),
        help="Path of JSON config file (default: encode_config_perf.json in the same directory)"
    )
    parser.add_argument(
        "-n", "--num_processes",
        type=int,
        default=1,
        help="Number of parallel encoding processes. Default: 1"
    )
    parser.add_argument(
        "-f", "--frame_count",
        type=int,
        help="Number of frames for each process to encode. If not specified, encodes all frames."
    )

    args = parser.parse_args()
    config = {}

    if args.config_file:
        with open(args.config_file) as jsonFile:
            config = json.loads(jsonFile.read())
            config["preset"] = config["preset"].upper()

    args.codec = args.codec.lower()
    args.format = args.format.upper()
    config["codec"] = args.codec
    width, height = map(int, args.size.split("x"))
    framesize = GetFrameSize(width, height, args.format)

    multiprocessing.set_start_method('spawn')

    run_parallel_encode(
        args.num_processes,
        args.gpu_id,
        args.raw_file_path.as_posix(),
        width,
        height,
        args.format,
        framesize,
        config,
        args.frame_count
    )
