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

import sys
import os
import argparse
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from multiprocessing import Process
import multiprocessing
from os.path import join, dirname, abspath
import pycuda.autoinit

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def decode_process(gpu_id, enc_file_path, frame_count=None):
    """
    Function to decode media file in a separate process.

    Parameters:
        gpu_id (int): Ordinal of GPU to use
        enc_file_path (str): Path to file to be decoded
        frame_count (int, optional): Maximum number of frames to decode. If None, 0, or negative, decode all frames.
    """
    # Validate frame count if provided
    if frame_count is None or frame_count <= 0:
        print(f"Decoding all available frames")
        frame_count = None  # Reset to None to decode all frames
        
    cuda_ctx = None
    
    try:
        device_id = gpu_id
        cuda_device = cuda.Device(device_id)  # pyright: ignore[reportAttributeAccessIssue]
        cuda_ctx = cuda_device.retain_primary_context()
        cuda_ctx.push()
        cuda_stream = cuda.Stream()
        nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
        nv_dec = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=nv_dmx.GetNvCodecId(),
            cudacontext=cuda_ctx.handle,
            cudastream=cuda_stream.handle,
            usedevicememory=True
        )
        
        num_decoded_frames = 0
        for packet in nv_dmx:
            for _ in nv_dec.Decode(packet):
                num_decoded_frames += 1
                cuda_stream.synchronize()
                
                if frame_count is not None and frame_count > 0 and num_decoded_frames >= frame_count:
                    print(f"Process {multiprocessing.current_process().name} reached requested frame count: {frame_count}")
                    break
            
            if frame_count is not None and frame_count > 0 and num_decoded_frames >= frame_count:
                break
        
        if frame_count is not None and frame_count > 0 and num_decoded_frames < frame_count:
            print(f"Video ended before reaching requested frame count for process {multiprocessing.current_process().name}. Decoded {num_decoded_frames} frames")
        
        print(f"Process {multiprocessing.current_process().name} completed decoding {num_decoded_frames} frames")
        
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateDecoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if cuda_ctx is not None:
            cuda_ctx.pop()


def run_parallel_decode(num_processes, gpu_id, enc_file_path, frame_count=None):
    """
    This function demonstrates parallel video decoding using multiple processes.

    The application creates multiple python processes and runs a different decoding session on each process.
    Each process decodes the same input file independently.

    Parameters:
        num_processes (int): Number of parallel decoding processes to run
        gpu_id (int): Ordinal of GPU to use
        enc_file_path (str): Path to file to be decoded
        frame_count (int, optional): Maximum number of frames to decode per process
    """
    multiprocessing.set_start_method('spawn')
    processes = []

    for i in range(num_processes):
        p = Process(target=decode_process, args=(gpu_id, enc_file_path, frame_count))
        p.start()
        processes.append(p)
        print(f"Started decode process {i+1}/{num_processes}")

    for p in processes:
        p.join()

    print(f"All {num_processes} decode processes completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application demonstrates parallel video decoding using multiple processes."
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
        "-n", "--num_processes",
        default=1,
        type=int,
        help="Number of parallel decode processes to run. (default: 1)"
    )
    parser.add_argument(
        "-f", "--frame_count",
        type=int,
        help="Maximum number of frames to decode per process. If not provided, all frames will be decoded."
    )

    args = parser.parse_args()

    run_parallel_decode(
        args.num_processes,
        args.gpu_id,
        args.encoded_file_path.as_posix(),
        args.frame_count
    )
