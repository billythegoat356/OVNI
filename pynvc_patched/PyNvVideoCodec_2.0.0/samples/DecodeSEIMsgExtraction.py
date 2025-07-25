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
import pickle
import ctypes
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import cast_address_to_1d_bytearray

# Constants
MAX_CLOCK_TS = 3

class TIMECODESET(ctypes.Structure):
    """Structure for time code set information."""
    _fields_ = [
        ("time_offset_value", ctypes.c_uint32),  # unsigned int
        ("n_frames", ctypes.c_uint16),  # unsigned short
        ("clock_timestamp_flag", ctypes.c_uint8),  # unsigned char
        ("units_field_based_flag", ctypes.c_uint8),  # unsigned char
        ("counting_type", ctypes.c_uint8),  # unsigned char
        ("full_timestamp_flag", ctypes.c_uint8),  # unsigned char
        ("discontinuity_flag", ctypes.c_uint8),  # unsigned char
        ("cnt_dropped_flag", ctypes.c_uint8),  # unsigned char
        ("seconds_value", ctypes.c_uint8),  # unsigned char
        ("minutes_value", ctypes.c_uint8),  # unsigned char
        ("hours_value", ctypes.c_uint8),  # unsigned char
        ("seconds_flag", ctypes.c_uint8),  # unsigned char
        ("minutes_flag", ctypes.c_uint8),  # unsigned char
        ("hours_flag", ctypes.c_uint8),  # unsigned char
        ("time_offset_length", ctypes.c_uint8),  # unsigned char
        ("reserved", ctypes.c_uint8),  # unsigned char
    ]

    def __repr__(self):
        return (
            "TIMECODESET(\n"
            f"    time_offset_value={self.time_offset_value},\n"
            f"    n_frames={self.n_frames},\n"
            f"    clock_timestamp_flag={self.clock_timestamp_flag},\n"
            f"    units_field_based_flag={self.units_field_based_flag},\n"
            f"    counting_type={self.counting_type},\n"
            f"    full_timestamp_flag={self.full_timestamp_flag},\n"
            f"    discontinuity_flag={self.discontinuity_flag},\n"
            f"    cnt_dropped_flag={self.cnt_dropped_flag},\n"
            f"    seconds_value={self.seconds_value},\n"
            f"    minutes_value={self.minutes_value},\n"
            f"    hours_value={self.hours_value},\n"
            f"    seconds_flag={self.seconds_flag},\n"
            f"    minutes_flag={self.minutes_flag},\n"
            f"    hours_flag={self.hours_flag},\n"
            f"    time_offset_length={self.time_offset_length},\n"
            f"    reserved={self.reserved}\n"
            ")"
        )

class TIMECODE(ctypes.Structure):
    """Structure for time code information."""
    _fields_ = [
        ("time_code_set", TIMECODESET * MAX_CLOCK_TS),  # Array of TIMECODESET
        ("num_clock_ts", ctypes.c_uint8),  # unsigned char
    ]

    def __repr__(self):
        time_code_set_repr = ",\n        ".join(
            repr(self.time_code_set[i]) for i in range(self.num_clock_ts)
        )
        return (
            "TIMECODE(\n"
            f"    time_code_set=[\n        {time_code_set_repr}\n    ],\n"
            f"    num_clock_ts={self.num_clock_ts}\n"
            ")"
        )

class SEICONTENTLIGHTLEVELINFO(ctypes.Structure):
    """Structure for content light level information."""
    _fields_ = [
        ("max_content_light_level", ctypes.c_uint16),  # unsigned short
        ("max_pic_average_light_level", ctypes.c_uint16),  # unsigned short
        ("reserved", ctypes.c_uint32),  # unsigned int
    ]

    def __repr__(self):
        return (
            "SEICONTENTLIGHTLEVELINFO(\n"
            f"    max_content_light_level={self.max_content_light_level},\n"
            f"    max_pic_average_light_level={self.max_pic_average_light_level},\n"
            f"    reserved={self.reserved}\n"
            ")"
        )

class SEIMASTERINGDISPLAYINFO(ctypes.Structure):
    """Structure for mastering display information."""
    _fields_ = [
        ("display_primaries_x", ctypes.c_uint16 * 3),  # Array of 3 unsigned short
        ("display_primaries_y", ctypes.c_uint16 * 3),  # Array of 3 unsigned short
        ("white_point_x", ctypes.c_uint16),  # unsigned short
        ("white_point_y", ctypes.c_uint16),  # unsigned short
        ("max_display_mastering_luminance", ctypes.c_uint32),  # unsigned int
        ("min_display_mastering_luminance", ctypes.c_uint32),  # unsigned int
    ]

    def __repr__(self):
        return (
            "SEIMASTERINGDISPLAYINFO(\n"
            f"    display_primaries_x={list(self.display_primaries_x)},\n"
            f"    display_primaries_y={list(self.display_primaries_y)},\n"
            f"    white_point_x={self.white_point_x},\n"
            f"    white_point_y={self.white_point_y},\n"
            f"    max_display_mastering_luminance={self.max_display_mastering_luminance},\n"
            f"    min_display_mastering_luminance={self.min_display_mastering_luminance}\n"
            ")"
        )
    
class TIMECODEMPEG2(ctypes.Structure):
    """Structure for MPEG2 time code information."""
    _fields_ = [
        ("drop_frame_flag", ctypes.c_uint8),  # unsigned char
        ("time_code_hours", ctypes.c_uint8),  # unsigned char
        ("time_code_minutes", ctypes.c_uint8),  # unsigned char
        ("marker_bit", ctypes.c_uint8),  # unsigned char
        ("time_code_seconds", ctypes.c_uint8),  # unsigned char
        ("time_code_pictures", ctypes.c_uint8),  # unsigned char
    ]

    def __repr__(self):
        return (
            "TIMECODEMPEG2(\n"
            f"    drop_frame_flag={self.drop_frame_flag},\n"
            f"    time_code_hours={self.time_code_hours},\n"
            f"    time_code_minutes={self.time_code_minutes},\n"
            f"    marker_bit={self.marker_bit},\n"
            f"    time_code_seconds={self.time_code_seconds},\n"
            f"    time_code_pictures={self.time_code_pictures}\n"
            ")"
        )

class SEIALTERNATIVETRANSFERCHARACTERISTICS(ctypes.Structure):
    """Structure for alternative transfer characteristics."""
    _fields_ = [
        ("preferred_transfer_characteristics", ctypes.c_uint8),  # unsigned char
    ]
    
    def __repr__(self):
        return f"SEIALTERNATIVETRANSFERCHARACTERISTICS(preferred_transfer_characteristics={self.preferred_transfer_characteristics})"

def decode(gpu_id, enc_file_path, dec_file_path, sei_file_path, use_device_memory):
    """
    Function to decode media file, write raw frames and SEI messages into output files.

    Parameters:
        gpu_id (int): Ordinal of GPU to use
        enc_file_path (str): Path to file to be decoded
        dec_file_path (str): Path to output file for raw frames
        sei_file_path (str): Path to output file for SEI messages
        use_device_memory (int): If set to 1, output decoded frame is CUDeviceptr wrapped in CUDA Array Interface

    Returns:
        None
    """
    cuda_ctx = None

    # Open files for storing SEI data
    file_message = open(sei_file_path, "wb")
    script_dir = dirname(abspath(__file__))
    sei_type_message_path = join(script_dir, "sei_type_message.bin")
    file_type_message = open(sei_type_message_path, "wb")

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
            usedevicememory=use_device_memory,
            enableSEIMessage=1
        )

        decoded_frame_size = 0
        raw_frame = None
        seq_triggered = False
        
        print(f"FPS = {nv_dmx.FrameRate()}")
        
        sei_message_found = False
        with open(dec_file_path, "wb") as decFile:
            for packet in nv_dmx:
                # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
                # size of (decode picture buffer) depends on GPU, for Turing series its 8
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
                        new_array = cast_address_to_1d_bytearray(
                            base_address=luma_base_addr,
                            size=decoded_frame.framesize()
                        )
                        decFile.write(bytearray(new_array))

                    # Process SEI messages
                    seiMessage = decoded_frame.getSEIMessage()
                    if seiMessage:
                        sei_message_found = True
                        for sei_info, sei_message in seiMessage:
                            sei_type = sei_info["sei_type"]
                            sei_uncompressed = sei_info["sei_uncompressed"]
                            
                            if sei_uncompressed == 1:
                                buffer = (ctypes.c_ubyte * len(sei_message))(*sei_message)
                                sei_struct = None
                                
                                # Handle different SEI message types
                                if sei_type in (nvc.SEI_TYPE.TIME_CODE_H264, nvc.SEI_TYPE.TIME_CODE):
                                    sei_struct = ctypes.cast(
                                        buffer,
                                        ctypes.POINTER(TIMECODEMPEG2 if nv_dmx.GetNvCodecId() == nvc.cudaVideoCodec.MPEG2 else TIMECODE)
                                    ).contents
                                elif sei_type == nvc.SEI_TYPE.MASTERING_DISPLAY_COLOR_VOLUME:
                                    sei_struct = ctypes.cast(buffer, ctypes.POINTER(SEIMASTERINGDISPLAYINFO)).contents
                                elif sei_type == nvc.SEI_TYPE.CONTENT_LIGHT_LEVEL_INFO:
                                    sei_struct = ctypes.cast(buffer, ctypes.POINTER(SEICONTENTLIGHTLEVELINFO)).contents
                                elif sei_type == nvc.SEI_TYPE.ALTERNATIVE_TRANSFER_CHARACTERISTICS:
                                    sei_struct = ctypes.cast(buffer, ctypes.POINTER(SEIALTERNATIVETRANSFERCHARACTERISTICS)).contents
                                
                                if sei_struct:
                                    print(sei_struct)
                                    
                            file_message.write(bytearray(sei_message))
                        pickle.dump(seiMessage, file_type_message)
                        print(f"SEI message written to {sei_file_path}")
        if not sei_message_found:
            print("No SEI message found")

    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateDecoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        file_message.close()
        file_type_message.close()
        if cuda_ctx is not None:
            cuda_ctx.pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application demonstrates decoding media files and extracting SEI messages."
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
        "-f", "--sei_file_path",
        type=Path,
        default="sei_message.bin",
        help="SEI binary file (write to). Default: sei_message.bin"
    )
    parser.add_argument(
        "-d", "--use_device_memory",
        type=int,
        default=1,
        help="Decoder output surface is in device memory (1) else in host memory (0). Default is 1."
    )

    args = parser.parse_args()
    
    # Set default raw_file_path if not provided
    if args.raw_file_path is None:
        input_path = Path(args.encoded_file_path)
        args.raw_file_path = input_path.with_suffix('.yuv')
    
    decode(
        args.gpu_id,
        args.encoded_file_path.as_posix(),
        args.raw_file_path.as_posix(),
        args.sei_file_path.as_posix(),
        args.use_device_memory
    )