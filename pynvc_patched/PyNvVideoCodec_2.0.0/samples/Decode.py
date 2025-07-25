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
import argparse
import numpy as np
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from os.path import join, dirname, abspath, splitext

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import cast_address_to_1d_bytearray

def decode(gpu_id, enc_file_path, dec_file_path, use_device_memory, decode_latency,frame_count=None):
    """
    Function to decode media file and write raw frames into an output file.

    This function will read a media file and split it into chunks of data (packets).
    A Packet contains elementary bitstream belonging to one frame and conforms to annex.b standard.
    Packet is sent to decoder for parsing and hardware accelerated decoding. Decoder returns list of raw YUV
    frames which can be iterated upon.

    Parameters:
        gpu_id (int): Ordinal of GPU to use
        enc_file_path (str): Path to file to be decoded
        dec_file_path (str): Path to output file into which raw frames are stored
        use_device_memory (int): If set to 1, output decoded frame is CUDeviceptr wrapped in CUDA Array Interface
        frame_count (int, optional): Maximum number of frames to decode. If None, 0, or negative, decode all frames.

            Example:
            >>> decode(0, "path/to/input/media/file","path/to/output/yuv", 1)
            Function to decode media file and write raw frames into an output file.
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

        caps = nvc.GetDecoderCaps(
            gpuid=gpu_id,
            codec=nv_dmx.GetNvCodecId(),
            chromaformat=nv_dmx.ChromaFormat(),
            bitdepth=nv_dmx.BitDepth()
        )
        if "num_decoder_engines" in caps:
            print("Number of NVDECs:", caps["num_decoder_engines"])

        nv_dec = nvc.CreateDecoder(gpuid=gpu_id,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=cuda_ctx.handle,
                               cudastream=cuda_stream.handle,
                               usedevicememory=use_device_memory,
                               latency=decode_latency)


        decoded_frame_size = 0
        raw_frame = None

        seq_triggered = False
        # printing out FPS and pixel format of the stream for convenience
        print("FPS =", nv_dmx.FrameRate())
        # open the file to be decoded in write mode
        with open(dec_file_path, "wb") as decFile:
            # demuxer can be iterated, fetch the packet from demuxer
            frames_decoded = 0
            for packet in nv_dmx:
                # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
                # size of (decode picture buffer) depends on GPU, fur Turing series its 8
                if decode_latency == nvc.DisplayDecodeLatencyType.LOW or decode_latency == nvc.DisplayDecodeLatencyType.ZERO:
                    # Set when the packet contains exactly one frame or one field bitstream data, parser will trigger decode callback immediately when this flag is set.
                    packet.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE

                for decoded_frame in nv_dec.Decode(packet):
                    # 'decoded_frame' contains list of views implementing cuda array interface
                    # for nv12, it would contain 2 views for each plane and two planes would be contiguous 
                    if not seq_triggered:
                        decoded_frame_size = nv_dec.GetFrameSize()
                        raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                        seq_triggered = True

                    luma_base_addr = decoded_frame.GetPtrToPlane(0)
                    if use_device_memory:
                        cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                        bits = bytearray(raw_frame)
                        decFile.write(bits)
                    else:
                        new_array = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=decoded_frame.framesize())
                        decFile.write(bytearray(new_array))
                    
                    frames_decoded += 1
                    if frame_count is not None and frame_count > 0 and frames_decoded >= frame_count:
                        print(f"Reached requested frame count: {frame_count}")
                        return
            
            if frame_count is not None and frame_count > 0 and frames_decoded < frame_count:
                print(f"Video ended before reaching requested frame count. Decoded {frames_decoded} frames")
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateDecoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if cuda_ctx is not None:
            cuda_ctx.pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application illustrates the demuxing and decoding of a media file."
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
        "-o", "--raw_file_path",
        type=Path,
        help="Raw NV12 video file (write to). Default: <input_file_name>.yuv"
    )
    parser.add_argument(
        "-d", "--use_device_memory",
        type=int,
        default=1,
        help="Decoder output surface is in device memory (1) else in host memory (0) (default: 1)"
    )
    parser.add_argument(
        "-f",
        "--frame_count", type=int, help="Maximum number of frames to decode")
    parser.add_argument(
        "-dl", "--decode_latency", type=int, default=0, help=""
                                                            "0 - Decoder input and output have a latency of 4 frames, output in display order"
                                                            "1 - Decoder input and output have a latency of 0 frames, output in display order"
                                                            "2 - Decoder input and output have a latency of 0 frames, output in decode order",
    )
    args = parser.parse_args()
    decode_latency = nvc.DisplayDecodeLatencyType.NATIVE
    if args.decode_latency == 1:
        decode_latency = nvc.DisplayDecodeLatencyType.LOW
    elif args.decode_latency == 2:
        decode_latency = nvc.DisplayDecodeLatencyType.ZERO

    
    # Set default raw_file_path if not provided
    if args.raw_file_path is None:
        input_path = Path(args.encoded_file_path)
        args.raw_file_path = input_path.with_suffix('.yuv')
    
    decode(args.gpu_id, args.encoded_file_path.as_posix(),
           args.raw_file_path.as_posix(),
           args.use_device_memory,
           decode_latency,
           args.frame_count)
    print(f"Decoded frames written to {args.raw_file_path}")
