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
import json
from pathlib import Path
import PyNvVideoCodec as nvc
import numpy as np
import pycuda.driver as cuda
from pathlib import Path
from pycuda.autoinit import context
import sys
import os
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import cast_address_to_1d_bytearray


def frame_to_yuv(frames, use_device_memory):
    """
    Converts frames to YUV format with option for device or host memory processing.
    
    Args:
        frames: List of frames to convert
        use_device_memory: Boolean flag to determine memory processing location
        simple_decoder: Decoder object
    
    Returns:
        frames_in_yuv: List of YUV frames as bytearrays
    """
    frames_in_yuv = []
    for frame in frames:
        raw_frame = np.ndarray(shape=frame.framesize(), dtype=np.uint8)
        luma_base_addr = frame.GetPtrToPlane(0)
        if use_device_memory == True:
            cuda.memcpy_dtoh(raw_frame, luma_base_addr)
        else:
            raw_frame = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=frame.framesize())
        bits = bytearray(raw_frame)
        frames_in_yuv.append(bits)
    
    return frames_in_yuv
    


def save_frames_in_yuv(dec_file_path, frames, use_device_memory):
        """
        Saves the frames in a YUV file.
        
        Args:
            dec_file_path(str): YUV file path.
            frames(List[DecodedFrame]): List of frames to convert
            use_device_memory(bool): Boolean flag to determine memory processing location
        
        """
        with open(dec_file_path, "ab") as decFile:
            for frame in frames:
                decoded_frame_size = frame.framesize()
                raw_frame = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
                luma_base_addr = frame.GetPtrToPlane(0)
                if use_device_memory:
                    cuda.memcpy_dtoh(raw_frame, luma_base_addr)
                    bits = bytearray(raw_frame)
                    decFile.write(bits)
                else:
                    new_array = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=decoded_frame_size)
                    decFile.write(bytearray(new_array))


def perform_batch_comparison(input_file_path, use_device_memory, gpu_id):
    """Compares sequential vs index-based batch frame retrieval methods."""
    batch_size = 5
    output_color_type = nvc.OutputColorType.NATIVE
    print("1. Comparing batch frames retrieval methods:")
    print(f"Method 1: Sequential batch (size={batch_size})")
    print(f"Method 2: Index-based batch (indices=[0,1,2,3,4])\n")
    
    simple_decoder = nvc.SimpleDecoder(input_file_path, need_scanned_stream_metadata=False, 
                                    use_device_memory=use_device_memory,
                                    output_color_type=output_color_type,
                                    gpu_id=gpu_id)

    simple_decoder.seek_to_index(0)
    batch_frames = simple_decoder.get_batch_frames(batch_size)
    batch_frames_in_yuv = frame_to_yuv(batch_frames, use_device_memory)
    batch_frames_idx_list = simple_decoder.get_batch_frames_by_index([0,1,2,3,4])
    batch_frames_idx_list_yuv = frame_to_yuv(batch_frames_idx_list, use_device_memory)
    
    if batch_frames_in_yuv == batch_frames_idx_list_yuv:
        print("Results match: Both methods returned identical frame data\n")
    else:
        print("Results differ: Frame data mismatch between methods\n")
    
    return simple_decoder

def perform_frame_slicing(simple_decoder, use_device_memory):
    """Validates frame extraction using slice operations vs sequential parsing."""
    print("2. Get frames by slicing [start:end:step] and get frames by sequential parsing\n")

    sliced_frames = simple_decoder[0:10:2]
    sliced_frames_in_yuv = frame_to_yuv(sliced_frames, use_device_memory)

    seq_frames_in_yuv = []
    for i in range(10):
        if (i%2)==0:
            frame = simple_decoder[i]
            raw_frame = np.ndarray(shape=frame.framesize(), dtype=np.uint8)
            luma_base_addr = frame.GetPtrToPlane(0)
            if use_device_memory == True:
                cuda.memcpy_dtoh(raw_frame, luma_base_addr)
            else:
                raw_frame = cast_address_to_1d_bytearray(base_address=luma_base_addr, size=frame.framesize())
            bits = bytearray(raw_frame)
            seq_frames_in_yuv.append(bits)
        
    if seq_frames_in_yuv==sliced_frames_in_yuv:
        print("Results match: Both methods returned identical frame data\n")
    else:
        print("Results differ: Frame data mismatch between methods\n")

def perform_timestamp_extraction(simple_decoder, time_file_path, use_device_memory):
    """Extracts frames using timestamp-based indexing."""
    print("3. Get frames for input time list in seconds\n")

    out_yuv_path = "framesFromInputTimeList.yuv"
    # Delete existing file before writing
    if os.path.exists(out_yuv_path):
        os.remove(out_yuv_path)
    target_frame_indices=[]

    with open(time_file_path, 'r') as file:
        for line in file:
            try:
                time_value = float(line.strip())
                target_frame_idx = simple_decoder.get_index_from_time_in_seconds(time_value)
                target_frame_indices.append(target_frame_idx)
                frame = simple_decoder[target_frame_idx]
                save_frames_in_yuv(out_yuv_path, [frame], use_device_memory)
            except ValueError:
                print(f"Warning: Skipping invalid time value: {line.strip()}")
    
    with open('timeValueIndices.txt', 'w') as file:
        file.write(str(target_frame_indices))
        
    print(f"Successfully saved frames to: {out_yuv_path}\n")
    print(f"Indices corresponding to the time values are saved in timeValueIndices.txt\n")

def perform_keyframe_extraction(input_file_path, use_device_memory, gpu_id):
    """Extracts and identifies keyframes."""
    print("4. Get key frames\n")
    simple_decoder = nvc.SimpleDecoder(input_file_path, need_scanned_stream_metadata=True, 
                                     use_device_memory=use_device_memory, gpu_id=gpu_id)
    metadata = simple_decoder.get_scanned_stream_metadata()
    key_frame_indices = metadata.key_frame_indices

    with open('keyFrameIndices.txt', 'w') as file:
        file.write(str(key_frame_indices))

    key_frames_batch = [key_frame_indices[i:i + 8] for i in range(0, len(key_frame_indices), 8)]
    out_yuv_path = "keyFrames.yuv"
    # Delete existing file before writing
    if os.path.exists(out_yuv_path):
        os.remove(out_yuv_path)

    for frames in key_frames_batch:
        batch_frames = simple_decoder.get_batch_frames_by_index(frames)
        save_frames_in_yuv(out_yuv_path, batch_frames, use_device_memory)
    
    print(f"Successfully saved frames to: {out_yuv_path}\n")
    print(f"Key frame indices are saved in keyFrameIndices.txt\n")

def perform_segment_generation(input_file_path, segments_file_path, config_file_path, segment_output_template, gpu_id):
    """Generates video segments based on time ranges."""
    print("5. Get segments given start and end time values.")

    with open(config_file_path) as jsonFile:
        config = json.loads(jsonFile.read())

    with open(segments_file_path, 'r') as file:
        for line in file:
            try:
                time_values = line.strip()
                if not time_values:  # Skip empty lines
                    continue
                start_ts_str, end_ts_str = time_values.split(' ')
                
                input_file_name_base = os.path.splitext(os.path.basename(input_file_path))[0]
                
                # Template is for the base path, should primarily use {input_file_name}
                base_path_template_vars = {
                    "input_file_name": input_file_name_base
                }
                # This is the base path to be passed to the Transcoder
                base_segment_output_path = segment_output_template.format(**base_path_template_vars)
                
                # Ensure the output path has an extension, default to .mp4 if none provided
                base_output_name_part, base_output_ext_part = os.path.splitext(base_segment_output_path)
                if not base_output_ext_part:
                    base_output_ext_part = '.mp4'
                    base_segment_output_path = base_output_name_part + base_output_ext_part
                
                # Construct the final expected path for printing
                final_expected_path = f"{base_output_name_part}_{start_ts_str}_{end_ts_str}{base_output_ext_part}"

                
                tr = nvc.Transcoder(input_file_path,
                                    base_segment_output_path, # Pass base path
                                    gpu_id,
                                    0, 0,
                                    **config)
                
                tr.segmented_transcode(float(start_ts_str), float(end_ts_str))
                # Use the constructed final_expected_path for the confirmation message
                print(f"Segmented output saved as {final_expected_path}")
            except ValueError:
                print(f"Warning: Skipping invalid time value or format: {line.strip()}")
            except Exception as e:
                print(f"An error occurred while processing segment: {line.strip()}. Error: {e}")

def get_enabled_operations(operation_ids=None):
    """Determines which operations should be run based on provided IDs."""
    # Define mapping between operation IDs and names
    operation_map = {
        1: 'batch_comparison',
        2: 'frame_slicing',
        3: 'timestamp_extraction',
        4: 'keyframe_extraction',
        5: 'segment_generation'
    }
    
    # Define all available operations
    all_operations = {
        'batch_comparison': False,
        'frame_slicing': False,
        'timestamp_extraction': False,
        'keyframe_extraction': False,
        'segment_generation': False
    }
    
    # If no operations specified, run all operations
    if not operation_ids:
        for operation in all_operations:
            all_operations[operation] = True
    else:
        # Enable only selected operations
        for op_id in operation_ids:
            if op_id in operation_map:
                all_operations[operation_map[op_id]] = True
            else:
                print(f"Warning: Unknown operation ID '{op_id}' specified - skipping")
    
    return all_operations

def ensemble_app(input_file_path, time_file_path, use_device_memory, segments_file_path, config_file_path, gpu_id, segment_output_template, operation_ids=None):
    '''
    Validates and compares different frame extraction methods from a video source:
    
    Available operations:
    1. Batch Frame Comparison: Compares sequential vs index-based frame retrieval
    2. Frame Slicing: Validates frame extraction using slice operations vs sequential parsing
    3. Timestamp Extraction: Tests frame retrieval using timestamp-based indexing
    4. Keyframe Extraction: Verifies keyframe identification and extraction
    5. Segment Generation: Retrieves video segments based on start and end time values
    '''
    
    try:
        # Determine which operations to run
        all_operations = get_enabled_operations(operation_ids)
        
        # Initialize decoder for operations that share it
        simple_decoder = None
        
        # Perform enabled operations
        if all_operations['batch_comparison']:
            simple_decoder = perform_batch_comparison(input_file_path, use_device_memory, gpu_id)
            
        if all_operations['frame_slicing']:
            if not simple_decoder:
                simple_decoder = nvc.SimpleDecoder(input_file_path, need_scanned_stream_metadata=False,
                                                use_device_memory=use_device_memory, gpu_id=gpu_id)
            perform_frame_slicing(simple_decoder, use_device_memory)
            
        if all_operations['timestamp_extraction']:
            if not simple_decoder:
                simple_decoder = nvc.SimpleDecoder(input_file_path, need_scanned_stream_metadata=False,
                                                use_device_memory=use_device_memory, gpu_id=gpu_id)
            perform_timestamp_extraction(simple_decoder, time_file_path, use_device_memory)
            
        if all_operations['keyframe_extraction']:
            perform_keyframe_extraction(input_file_path, use_device_memory, gpu_id)
            
        if all_operations['segment_generation']:
            perform_segment_generation(input_file_path, segments_file_path, config_file_path, segment_output_template, gpu_id)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser("This sample application demonstrates different functionalities of Simple Decoder & Simple transcoder",
                                      formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-i", "--input_file_path",
                        required=True,
                        type=Path, 
                        help='Path to input video file' )
    
    parser.add_argument("-g", "--gpu_id",
                        type=int,
                        default=0,
                        help="Check nvidia-smi for available GPUs. Default: 0"
    )
    

    parser.add_argument("-t", '--time_file_path',
                        type=Path,      
                        default=join(current_dir, "timeListInSeconds.txt"),
                        help='''Path to the input time file.
File format requirements:
- One time value (in seconds) per line
- Values should be numeric (integer or decimal)
- Empty lines will be ignored 
Default: timeListInSeconds.txt in the same directory'''
                        )

    parser.add_argument("-d", "--use_device_memory",
                         choices=[0, 1], 
                        default=1, 
                        type=int, 
                        help="Decoder output surface is in device memory else in "
                                                                   "host memory. Default is 1 (device memory).", 
                        )
    
    parser.add_argument("-json", "--config_file_path",
                        type=Path,
                        default=join(current_dir, "transcode_config.json"),
                        help="Path of json config file (default: transcode_config.json in the same directory)", )
    
    parser.add_argument("-segments", '--segments_file_path',
                        type=Path,
                        default=join(current_dir, "segments.txt"),
                        help='''Path to the input segment file.
File format requirements:
- Each line should contain a start and end time value separated by a space.
- Values should be numeric (integer or decimal)
- Empty lines will be ignored 
Default: segments.txt in the same directory'''
                        )
    
    parser.add_argument("-so", "--segment-output",
                        type=str,
                        default="{input_file_name}.mp4",
                        help='''Base output file name template for segmented transcode. 
The API will append '_{start_ts}_{end_ts}' (e.g., _0.0_10.5) to this base name.
Default: '{input_file_name}_segment.mp4'''
                        )

    parser.add_argument("-o", "--operations",
                        type=int,
                        nargs='+',
                        choices=range(1, 6),
                        help='''Specify which decoder/transcoder operations to perform using operation IDs. Multiple IDs can be specified. IDs should be separated by space.
Available operations:
1. Batch Frame Comparison: Compare sequential vs index-based frame retrieval
2. Frame Slicing: Perform frame slicing operations vs sequential parsing
3. Timestamp Extraction: Extract frames using timestamps
4. Keyframe Extraction: Extract and identify keyframes
5. Segment Generation: Generate video segments based on time ranges
If not specified, all operations will be performed.''')

    args = parser.parse_args()
    input_file_path = args.input_file_path.as_posix()
    time_file_path = args.time_file_path.as_posix()
    config_file_path = args.config_file_path.as_posix()
    segments_file_path = args.segments_file_path.as_posix()
    use_device_memory = args.use_device_memory
    gpu_id = args.gpu_id
    operation_ids = args.operations
    segment_output_template_arg = args.segment_output

    ensemble_app(input_file_path, time_file_path, use_device_memory, segments_file_path, config_file_path, gpu_id, segment_output_template_arg, operation_ids)
    
    
    
    
    
    