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
import sys
import os
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import AppFrame
from Utils import FetchCPUFrame
from Utils import FetchGPUFrame


# Constants
TOTAL_NUM_FRAMES = 1000
SEI_MESSAGE_1 = [0xdc, 0x45, 0xe9, 0xbd, 0xe6, 0xd9, 0x48, 0xb7, 0x96, 0x2c, 0xd8, 0x20, 0xd9, 0x23, 0xee, 0xef]
SEI_MESSAGE_2 = [0x12, 0x67, 0x56, 0xda, 0xef, 0x99, 0x00, 0xbb, 0x6a, 0xc4, 0xd8, 0x10, 0xf9, 0xe3, 0x3e, 0x8f]


def encode(gpu_id, dec_file_path, enc_file_path, width, height, fmt, config_params):
    """
    Encode frames with SEI message insertion into the bitstream using CUDA device buffers as input.

    This function reads image data from a file and loads it to CUDA input buffers using FetchGPUFrame().
    The encoder copies the CUDA buffers and SEI message data and submits them to NVENC hardware for encoding.
    Video memory buffer is allocated to get the NVENC hardware output, which is then copied to host memory
    for file storage.

    Parameters:
        gpu_id (int): Ordinal of GPU to use [Parameter not in use]
        dec_file_path (str): Path to file to be decoded
        enc_file_path (str): Path to output file for encoded frames
        width (int): Width of encoded frame
        height (int): Height of encoded frame
        fmt (str): Surface format string in uppercase (e.g., "NV12")
        config_params (dict): Key-value pairs providing fine-grained control on encoding

    Returns:
        None

    Example:
        >>> encode(0, "input.yuv", "output.h264", 1920, 1080, "NV12", {"codec": "h264"})
        Encode 1080p NV12 raw YUV into elementary bitstream using H.264 codec with SEI messages
    """
    try:
        # Determine SEI type based on codec
        if config_params["codec"] in ["hevc", "h264"]:
            sei_info = {"sei_type": 5}
        elif config_params["codec"] == "av1":
            sei_info = {"sei_type": 6}
        else:
            raise ValueError(f"Unsupported codec: {config_params['codec']}")

        # Create SEI messages list
        sei_messages = [(sei_info, SEI_MESSAGE_1), (sei_info, SEI_MESSAGE_2)]

        with open(dec_file_path, "rb") as decFile, open(enc_file_path, "wb") as encFile:
            # Create encoder object
            config_params["gpu_id"] = gpu_id
            nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)
            
            # Create input frame list
            input_frame_list = [AppFrame(width, height, fmt) for _ in range(1, 5)]
            
            # Process frames
            for input_gpu_frame in FetchGPUFrame(
                input_frame_list,
                FetchCPUFrame(decFile, input_frame_list[0].frameSize),
                TOTAL_NUM_FRAMES
            ):
                bitstream = nvenc.Encode(input_gpu_frame, 0, sei_messages)
                encFile.write(bytearray(bitstream))
            
            # Flush encoder queue
            bitstream = nvenc.EndEncode()
            encFile.write(bytearray(bitstream))
        
        print(f"Encoded frames written to {enc_file_path} with SEI messages")

    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateEncoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application illustrates encoding of frames with SEI message insertion."
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
        help="Encoded video file (write to). Default: <input_file_name>.h264"
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
        help="Format of input file (default is NV12)"
    )
    parser.add_argument(
        "-c", "--codec",
        type=str,
        default="H264",
        help="Video codec (HEVC, H264, AV1). Default is H264"
    )
    parser.add_argument(
        "-json", "--config_file",
        type=str,
        default=join(current_dir, "encode_config.json"),
        help="Path of JSON config file (default: encode_config.json in the same directory)"
    )

    args = parser.parse_args()
    config = {}

    # Set default encoded_file_path if not provided
    if args.encoded_file_path is None:
        input_path = Path(args.raw_file_path)
        args.encoded_file_path = input_path.with_suffix('.h264')

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
        config
    )