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

import PyNvVideoCodec as nvc
import json
import argparse
from pathlib import Path
import io
import os
import sys
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import AppFrame
from Utils import FetchCPUFrame
from Utils import FetchGPUFrame


def encode(gpu_id, dec_file_path, enc_file_path, width, height, fmt, config_params, frame_count=1000):
    """
    Encode frames using CUDA device buffers as input.

    This function reads image data from a file and loads it to CUDA input buffers using
    FetchGPUFrame(). The encoder copies the CUDA buffers and submits them to NVENC hardware
    for encoding. Video memory buffer is allocated to get the NVENC hardware output.
    The output is copied from video memory to host memory for file storage.

    Parameters:
        gpu_id (int): Ordinal of GPU to use
        dec_file_path (str): Path to file to be decoded
        enc_file_path (str): Path to output file for encoded frames
        width (int): Width of encoded frame
        height (int): Height of encoded frame
        fmt (str): Surface format string in uppercase (e.g., "NV12")
        config_params (dict): Key-value pairs providing fine-grained control on encoding
        frame_count (int, optional): Number of frames to encode. If None, 0, or negative, encode all frames from input.

    Returns:
        None

    Example:
        >>> encode(0, "input.yuv", "output.h264", 1920, 1080, "NV12", {"codec": "h264"})
        Encode 1080p NV12 raw YUV into elementary bitstream using H.264 codec
    """
    try:
        caps = nvc.GetEncoderCaps(codec=config_params["codec"])
        if "num_encoder_engines" in caps:
            print(f"Number of NVENCs: {caps['num_encoder_engines']}")

        with open(dec_file_path, "rb") as decFile, open(enc_file_path, "wb") as encFile:
            config_params["gpu_id"] = gpu_id
            nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)
            input_frame_list = [AppFrame(width, height, fmt) for _ in range(1, 5)]
            
            # Calculate total available frames from file size
            file_size = os.path.getsize(dec_file_path)
            available_frames = file_size // input_frame_list[0].frameSize
            
            if frame_count is None or frame_count == 0:
                print(f"Encoding all available frames")
                frame_count = available_frames
            elif frame_count < 0 or frame_count > available_frames:
                print(f"Warning: Invalid frame count {frame_count}. Encoding all available frames.")
                frame_count = available_frames
            
            for input_gpu_frame in FetchGPUFrame(
                input_frame_list,
                FetchCPUFrame(decFile, input_frame_list[0].frameSize),
                frame_count
            ):
                bitstream = nvenc.Encode(input_gpu_frame)
                bitstream = bytearray(bitstream)
                encFile.write(bitstream)
            bitstream = nvenc.EndEncode()  # flush encoder queue
            bitstream = bytearray(bitstream)
            encFile.write(bitstream)
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateEncoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application illustrates encoding of frames using CUDA device buffers as input."
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
        "-o", "--encoded_file_path",
        type=Path,
        help="Encoded video file (write to). Default: <input_file_name>.<codec>"
    )
    parser.add_argument(
        "-s", "--size",
        type=str,
        required=True,
        help="widthxheight of raw frame (e.g., 1920x1080)"
    )
    parser.add_argument(
        "-if", "--format",
        type=str,
        default="NV12",
        help="Format of input file. (default: NV12)"
    )
    parser.add_argument(
        "-c", "--codec",
        type=str,
        default="h264",
        help="Video codec. Default is h264. (other options: hevc, av1)"
    )
    parser.add_argument(
        "-json", "--config_file",
        type=str,
        default=join(current_dir, "encode_config.json"),
        help="Path of JSON config file (default: encode_config.json in the same directory)"
    )
    parser.add_argument(
        "-f", "--frame_count",
        type=int,
        default=0,
        help="Number of frames to encode (0 for all frames)"
    )

    args = parser.parse_args()
    config = {}

    # Set default encoded_file_path if not provided
    if args.encoded_file_path is None:
        input_path = Path(args.raw_file_path)
        args.encoded_file_path = input_path.with_suffix(f'.{args.codec}')

    if args.config_file:
        with open(args.config_file) as jsonFile:
            config = json.loads(jsonFile.read())
            config["preset"] = config["preset"].upper()

    args.codec = args.codec.lower()
    args.format = args.format.upper()
    config["codec"] = args.codec
    width, height = map(int, args.size.split("x"))

    encode(
        args.gpu_id,
        args.raw_file_path.as_posix(),
        args.encoded_file_path.as_posix(),
        width,
        height,
        args.format,
        config,
        args.frame_count
    )
    print(f"Encoded frames written to {args.encoded_file_path}")