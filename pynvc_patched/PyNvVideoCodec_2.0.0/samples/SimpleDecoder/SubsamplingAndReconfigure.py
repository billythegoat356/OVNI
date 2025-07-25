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
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from os.path import join, dirname, abspath

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Utils import cast_address_to_1d_bytearray


class CUDAInitializationError(Exception):
    """Exception raised when CUDA initialization fails."""
    def __init__(self, message: str, gpu_id: int):
        self.message = message
        self.gpu_id = gpu_id
        super().__init__(f"{message} (GPU ID: {gpu_id})")


class SubsamplingAndReconfigure:
    """
    A class that handles video frame subsampling and decoder reconfiguration.
    
    This class provides functionality to:
    1. Extract frames from videos at specified target frame rates
    2. Reuse the same decoder instance for multiple video files
    3. Verify decoded frames against golden reference files
    4. Save decoded frames in YUV format
    """
    
    def __init__(self, create_cuda: bool, gpu_id: int) -> None: 
        """
        Initialize the SubsamplingAndReconfigure instance.
        
        Args:
            create_cuda: If True, performs CUDA initialization in the sample app
            gpu_id: ID of the GPU to use
            
        Raises:
            CUDAInitializationError: If CUDA initialization fails due to invalid GPU ID
        """
        self.simple_decoder: Optional[nvc.SimpleDecoder] = None
        self.cuda_ctx: Optional[cuda.Context] = None
        self.cuda_stream: Optional[cuda.Stream] = None
        self.cuda_device: Optional[cuda.Device] = None
        self.gpu_id: int = gpu_id

        if create_cuda:
            try:
                device_id = self.gpu_id
                self.cuda_device = cuda.Device(device_id)
                self.cuda_ctx = self.cuda_device.retain_primary_context()
            except Exception as e:
                raise CUDAInitializationError(str(e), gpu_id)

    def _process_frame_data(self, frame, decoded_frame_size: int, use_device_memory: bool) -> np.ndarray:
        """
        Process frame data from either device or host memory.
        
        Args:
            frame: Frame object to process
            decoded_frame_size: Size of the frame in bytes
            use_device_memory: Whether to fetch frame data from device memory
            
        Returns:
            numpy.ndarray: Processed frame data
        """
        reference = np.ndarray(shape=decoded_frame_size, dtype=np.uint8)
        luma_base_addr = frame.GetPtrToPlane(0)
        
        if use_device_memory:
            cuda.memcpy_dtoh(reference, luma_base_addr)
        else:
            reference = cast_address_to_1d_bytearray(
                base_address=luma_base_addr,
                size=decoded_frame_size
            )
        
        return reference

    def verify(self, input_file_name: str, indices_list: List[int], 
              target_frames: List[nvc.DecodedFrame], decoded_frame_size: int, 
              use_device_memory: bool) -> bool:
        """
        Verify decoded frames against a golden reference YUV file.
        
        The golden file should be generated using the Decode sample application and follow
        the naming convention: <base_name_of_input_file>_golden.yuv
        
        Args:
            input_file_name: Base name to derive the golden file name
            indices_list: List of frame indices to verify
            target_frames: Decoded frame objects
            decoded_frame_size: Size of each decoded frame in bytes
            use_device_memory: Whether to fetch frame data from device memory
            
        Returns:
            bool: True if all frames match the golden reference, False otherwise
        """
        golden_file = f"{input_file_name}_golden.yuv"
        matching = True
        
        try:
            with open(golden_file, "rb") as testfile:
                for i, frame in zip(indices_list, target_frames):
                    testfile.seek(i * decoded_frame_size, 0)
                    golden = testfile.read(decoded_frame_size)
                    npgolden = np.frombuffer(golden, dtype=np.uint8)
                    
                    reference = self._process_frame_data(frame, decoded_frame_size, use_device_memory)
                    
                    if not (npgolden == reference).all():
                        matching = False
                        print(f"Reference for frame {i} does not match golden file")
        except FileNotFoundError:
            print(f"Warning: Golden reference file {golden_file} not found")
            matching = False
        
        return matching


    def _initialize_decoder(self, input_file_path: str, create_cuda: bool, 
                          use_device_memory: bool) -> None:
        """
        Initialize or reconfigure the decoder.
        
        Args:
            input_file_path: Path to the input video file
            create_cuda: Whether to create CUDA context and stream
            use_device_memory: Whether to use device memory for frame processing
        """
        try:
            if self.simple_decoder is None:
                if create_cuda:
                    print("Initializing CUDA in the sample application...")
                    self.cuda_stream = cuda.Stream()
                    self.simple_decoder = nvc.SimpleDecoder(
                        input_file_path,
                        gpu_id=self.gpu_id,
                        cuda_context=self.cuda_ctx.handle,
                        cuda_stream=self.cuda_stream.handle,
                        use_device_memory=use_device_memory
                    )
                else:
                    print("Using default CUDA initialization...")
                    self.simple_decoder = nvc.SimpleDecoder(
                        input_file_path,
                        gpu_id=self.gpu_id,
                        use_device_memory=use_device_memory
                    )
            else:
                print("Reconfiguring existing decoder...")
                self.simple_decoder.reconfigure_decoder(input_file_path)
        except Exception as e:
            print(f"Error initializing decoder: {e}")

    def sub_sampling(self, input_file_path: str, target_fps: float, create_cuda: bool, 
                    use_device_memory: bool, verify_with_golden_yuv: int = 0) -> None:
        """
        Extract frames from a video at a specified target frame rate.
        
        This method calculates which frames to extract to achieve the desired frame rate,
        creates batches of frame indices (up to 8 frames per batch) for efficient processing,
        and optionally verifies the output against a golden reference file.
        
        Args:
            input_file_path: Path to the input video file
            target_fps: Desired output frame rate in frames per second
            create_cuda: If True, performs CUDA initialization in the sample app
            use_device_memory: Whether to use device memory for frame processing
            verify_with_golden_yuv: If 1, verifies output against golden reference file
            
        Example:
            If video_fps = 30, target_fps = 10, and total_frames = 100:
            - Will extract every 3rd frame (30/10 = 3)
        """
        try:
            if create_cuda and self.cuda_ctx is not None:
                self.cuda_ctx.push()

            self._initialize_decoder(input_file_path, create_cuda, use_device_memory)

            # Get video metadata
            metadata = self.simple_decoder.get_stream_metadata()
            total_frames = metadata.num_frames
            video_fps = metadata.average_fps

            print(f"Video FPS = {video_fps}, Total Frames = {total_frames}")
            
            # Calculate frame indices for target FPS
            step = int(video_fps / target_fps)
            step_list = list(range(0, total_frames, step))
            
            # Split into batches of up to 8 frames
            step_list_of_lists = [step_list[i:i + 8] for i in range(0, len(step_list), 8)]

            input_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
            matching = True
            
            # Open output file once for all batches
            dec_file_path = f"{str(input_file_name)}.yuv"
            with open(dec_file_path, "wb") as decFile:
                # Process each batch
                for batch_indices in step_list_of_lists:
                    batch_frames = self.simple_decoder.get_batch_frames_by_index(batch_indices)
                    
                    if verify_with_golden_yuv:
                        matching = (matching and self.verify(
                            input_file_name,
                            batch_indices,
                            batch_frames,
                            batch_frames[0].framesize(),
                            use_device_memory
                        ))
                    
                    # Save frames to the already open file
                    for frame in batch_frames:
                        decoded_frame_size = frame.framesize()
                        reference = self._process_frame_data(frame, decoded_frame_size, use_device_memory)
                        decFile.write(bytearray(reference))

            if verify_with_golden_yuv and matching:
                print(f"All frames match golden reference for {input_file_name}.yuv")
            
            if create_cuda and self.cuda_ctx is not None:
                self.cuda_ctx.pop()
            
            print(f"Frames saved successfully in {input_file_name}.yuv\n")
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            if create_cuda and self.cuda_ctx is not None:
                self.cuda_ctx.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This sample application demonstrates reusing the same decoder instance "
                   "for subsampling multiple video files using Simple Decoder."
    )

    parser.add_argument(
        "-i", "--input_file_list",
        type=Path,
        default=join(current_dir, "fileLists.txt"),
        help="Text file containing video file paths, one per line. "
             "Each line should contain the absolute or relative path to a video file. "
             "Default: fileLists.txt in the same directory"
    )
    
    parser.add_argument(
        "-fps", "--target_fps",
        default=2,
        type=int,
        help="Desired output frame rate in frames per second. Default is 2."
    )

    parser.add_argument(
        "-c", "--create_cuda",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 1 to enable CUDA stream and context, 0 for using default initializations. Default is 1."
    )

    parser.add_argument(
        "-d", "--use_device_memory",
        choices=[0, 1],
        default=1,
        type=int,
        help="Decoder output surface is in device memory (1) else in host memory (0). Default is 1."
    )
    
    parser.add_argument(
        "-v", "--verify_with_golden_yuv",
        choices=[0, 1],
        default=0,
        type=int,
        help="Set to 1 to enable verification against a golden YUV file; default is 0."
    )

    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use. Default is 0."
    )   

    args = parser.parse_args()
    
    try:
        simple_dec_reconfigure = SubsamplingAndReconfigure(args.create_cuda, args.gpu_id)

        with open(args.input_file_list, 'r') as file:
            for line in file:
                if line.strip():
                    input_file_path = line.strip()
                    simple_dec_reconfigure.sub_sampling(
                        input_file_path,
                        args.target_fps,
                        args.create_cuda,
                        args.use_device_memory,
                        args.verify_with_golden_yuv
                    )
    except CUDAInitializationError as e:
        print(f"Error initializing CUDA: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)