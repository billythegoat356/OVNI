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

# Import necessary libraries
import argparse
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import numpy as np
import pycuda.autoinit as context
import sys
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import cast_address_to_1d_bytearray


class VideoStreamFeeder:
    """
    Class to handle feeding video data in chunks to the demuxer.
    
    This class reads a video file into memory and provides a method to feed
    chunks of data to the demuxer buffer.
    """
    def __init__(self, file_path):
        """
        Initialize the VideoStreamFeeder with a video file.

        Args:
            file_path (str): Path to the video file to be read
        """
        with open(file_path, 'rb') as f:
            self.video_buffer = bytearray(f.read())
        self.current_pos = 0
        self.bytes_remaining = len(self.video_buffer)
        self.chunk_size = 0

    def feed_chunk(self, demuxer_buffer):
        """
        Feed next chunk of video data to demuxer buffer.

        Args:
            demuxer_buffer: Pre-allocated buffer provided by demuxer

        Returns:
            int: Number of bytes copied to buffer, 0 if no more data
        """
        buffer_capacity = len(demuxer_buffer)
        
        if self.bytes_remaining < buffer_capacity:
            self.chunk_size = self.bytes_remaining
        else:
            self.chunk_size = buffer_capacity

        if self.chunk_size == 0:
            return 0

        demuxer_buffer[:] = self.video_buffer[self.current_pos:self.current_pos + self.chunk_size]

        self.current_pos += self.chunk_size
        self.bytes_remaining -= self.chunk_size
        return self.chunk_size


def demux_from_byte_array(input_file, yuv_file, use_device_memory, gpu_id):
    """
    Implement buffer-based pipeline that reads the input file in chunks.

    This function demonstrates how to decode video by processing data directly
    from memory buffers instead of reading from disk.

    Parameters:
        input_file (str): Path to the input video file to be decoded
        yuv_file (str): Path where the buffer-based pipeline output will be saved
        use_device_memory (int): If set to 1, output decoded frame is CUDeviceptr wrapped in CUDA Array Interface

    Returns:
        None: Decoded frames are written to a raw file.
    """
    print("Starting buffer-based decoding...")
    data_feeder = VideoStreamFeeder(input_file)
    buffer_demuxer = nvc.CreateDemuxer(data_feeder.feed_chunk)

    buffer_decoder = nvc.CreateDecoder(
        gpuid=gpu_id,
        codec=buffer_demuxer.GetNvCodecId(),
        cudacontext=0,
        cudastream=0,
        usedevicememory=use_device_memory
    )

    seq_triggered = False
    decoded_frame_size = 0
    raw_frame = None

    with open(yuv_file, 'wb') as decFile:
        for packet in buffer_demuxer:
            for decoded_frame in buffer_decoder.Decode(packet):
                if not seq_triggered:
                    decoded_frame_size = buffer_decoder.GetFrameSize()
                    raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                    seq_triggered = True

                luma_base_addr = decoded_frame.GetPtrToPlane(0)
                if use_device_memory:
                    cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                    decFile.write(bytearray(raw_frame))
                else:
                    new_array = cast_address_to_1d_bytearray(
                        base_address=luma_base_addr,
                        size=decoded_frame.framesize()
                    )
                    decFile.write(bytearray(new_array))

    print(f"Frames are saved successfully in {yuv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrates video demuxing by processing video data directly from memory buffers."
    )

    parser.add_argument(
        "-i", "--input_file",
        type=Path,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=0,
        help="Check nvidia-smi for available GPUs. Default: 0"
    )
    parser.add_argument(
        "-o", "--yuv_file_path",
        type=Path,
        help="Path to stream based output yuv file. Default: <input_file_name>.yuv"
    )
    parser.add_argument(
        "-d", "--use_device_memory",
        type=int,
        choices=[0, 1],
        default=1,
        help="Decoder output surface is in device memory (1) else in host memory (0). Default is 1."
    )

    args = parser.parse_args()
    
    # Set default yuv_file_path if not provided
    if args.yuv_file_path is None:
        input_path = Path(args.input_file)
        args.yuv_file_path = input_path.with_suffix('.yuv')
    
    demux_from_byte_array(
        args.input_file.as_posix(),
        args.yuv_file_path.as_posix(),
        args.use_device_memory,
        args.gpu_id
    )