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
import numpy as np
import json
import argparse
from pathlib import Path
import os
from os.path import join, dirname, abspath
import sys

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


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


def FetchCPUFrame(dec_file, frame_size):
    """
    Generator function to fetch frames from a file.

    Parameters:
        dec_file: File object to read from
        frame_size (int): Size of each frame in bytes

    Yields:
        numpy.ndarray: Frame data as a numpy array
    """
    yield np.fromfile(dec_file, np.uint8, count=frame_size)


def encode(gpu_id, dec_file_path, enc_file_path, width, height, fmt, config_params, frame_count=None):
    """
    Encode frames using host memory buffers as input.

    This function reads image data from a file and copies it to CUDA buffers for encoding.
    The encoder submits the data to NVENC hardware for encoding. Video memory buffer is
    allocated to get the NVENC hardware output. The output is copied from video memory
    to host memory for file storage.

    Parameters:
        gpu_id (int): Ordinal of GPU to use.
        dec_file_path (str): Path to file to be decoded
        enc_file_path (str): Path to output file for encoded frames
        width (int): Width of encoded frame
        height (int): Height of encoded frame
        fmt (str): Surface format string in uppercase (e.g., "NV12")
        config_params (dict): Key-value pairs providing fine-grained control on encoding
        frame_count (int, optional): Number of frames to encode. If None, 0 or negative, encodes all frames from input.

    Returns:
        None

    Example:
        >>> encode(0, "input.yuv", "output.h264", 1920, 1080, "NV12", 1, {"codec": "h264"})
        Encode 1080p NV12 raw YUV into elementary bitstream using H.264 codec
    """
    try:
        frame_size = GetFrameSize(width, height, fmt)
        with open(dec_file_path, "rb") as dec_file, open(enc_file_path, "wb") as enc_file:
            config_params["gpu_id"] = gpu_id
            nvenc = nvc.CreateEncoder(width, height, fmt, True, **config_params)  # create encoder object
            
            # Calculate total available frames from file size
            file_size = os.path.getsize(dec_file_path)
            available_frames = file_size // frame_size
            
            if frame_count is None or frame_count == 0:
                print(f"Encoding all available frames")
                frame_count = available_frames
            elif frame_count < 0 or frame_count > available_frames:
                print(f"Warning: Invalid frame count {frame_count}. Encoding all available frames.")
                frame_count = available_frames
            
            for _ in range(frame_count):
                chunk = np.fromfile(dec_file, np.uint8, count=frame_size)
                if chunk.size != 0:
                    bitstream = nvenc.Encode(chunk)
                    enc_file.write(bytearray(bitstream))
            
            # Flush encoder queue
            bitstream = nvenc.EndEncode()
            enc_file.write(bytearray(bitstream))
            print(f"Encoded {frame_count} frames. Output file: {enc_file_path}")
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateEncoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application illustrates encoding of frames using host memory buffers as input."
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
        help="Encoded video file (write to). Default: <raw_file_path>.<codec>"
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
        default="H264", 
        help="Video codec (HEVC, H264, AV1). (default: H264)"
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
