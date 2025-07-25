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
BITRATE_CHANGE_INTERVAL = 100
BITRATE_REDUCTION_FACTOR = 2


def encode(gpu_id, dec_file_path, enc_file_path, width, height, fmt, config_params):
    """
    Encode video with dynamic bitrate reconfiguration.

    This function demonstrates bitrate change at runtime without the need to reset the encoder session.
    The application reduces the bitrate by half and then restores it to the original value after
    every 100 frames.

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
        Encode 1080p NV12 raw YUV into elementary bitstream using H.264 codec
    """
    try:
        with open(dec_file_path, "rb") as decFile, open(enc_file_path, "wb") as encFile:
            # Create encoder object
            config_params["gpu_id"] = gpu_id
            nvenc = nvc.CreateEncoder(width, height, fmt, False, **config_params)
            
            # Create input frame list
            input_frame_list = [AppFrame(width, height, fmt) for _ in range(1, 5)]
            
            # Get initial encoder parameters
            reconf_params = nvenc.GetEncodeReconfigureParams()
            original_avgbitrate = reconf_params.averageBitrate
            original_vbvbuffersize = int(original_avgbitrate * reconf_params.frameRateDen / reconf_params.frameRateNum)
            original_vbvinitdelay = original_vbvbuffersize
            
            # Process frames
            for i, input_gpu_frame in enumerate(
                FetchGPUFrame(
                    input_frame_list,
                    FetchCPUFrame(decFile, input_frame_list[0].frameSize),
                    TOTAL_NUM_FRAMES
                )
            ):
                # Reconfigure bitrate every 100 frames
                if i % BITRATE_CHANGE_INTERVAL == 0:
                    if i % (BITRATE_CHANGE_INTERVAL * 2) != 0:
                        # Reduce bitrate by half
                        reconf_params.averageBitrate = int(original_avgbitrate / BITRATE_REDUCTION_FACTOR)
                        reconf_params.vbvBufferSize = int(
                            reconf_params.averageBitrate * reconf_params.frameRateDen / reconf_params.frameRateNum
                        )
                        reconf_params.vbvInitialDelay = reconf_params.vbvBufferSize
                    else:
                        # Restore original bitrate
                        reconf_params.averageBitrate = original_avgbitrate
                        reconf_params.vbvBufferSize = original_vbvbuffersize
                        reconf_params.vbvInitialDelay = original_vbvinitdelay
                    
                    nvenc.Reconfigure(reconf_params)
                
                # Encode frame
                bitstream = nvenc.Encode(input_gpu_frame)
                encFile.write(bytearray(bitstream))
            
            # Flush encoder queue
            bitstream = nvenc.EndEncode()
            encFile.write(bytearray(bitstream))
            
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateEncoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application demonstrates bitrate change at runtime without resetting the encoder session."
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
        default="H264",
        help="Video codec (HEVC, H264, AV1). Default is H264"
    )
    parser.add_argument(
        "-json", "--config_file",
        type=str,
        default=join(dirname(__file__), "encode_config_perf.json"),
        help="Path of JSON config file (default: encode_config_perf.json in the same directory)"
    )

    args = parser.parse_args()
    config = {}

    # Set default encoded_file_path if not provided
    if args.encoded_file_path is None:
        input_path = Path(args.raw_file_path)
        args.encoded_file_path = input_path.with_suffix(f'.{args.codec.lower()}')

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
