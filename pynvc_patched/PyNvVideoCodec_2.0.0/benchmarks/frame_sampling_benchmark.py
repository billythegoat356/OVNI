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

import os
import platform
import json
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
import PyNvVideoCodec as nvc
import pycuda.driver as cuda
import pycuda.autoinit
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from tabulate import tabulate
import threading
import subprocess
import argparse

"""
                PyNvVideoCodec Benchmarking Script

                This script performs benchmarking tests on the PyNvVideoCodec decoder to evaluate its performance
                under different conditions. It measures:
                1. Decoding performance with different thread counts
                2. Efficiency of direct frame sampling vs sequential decoding
                3. Performance across different video configurations (resolution, GOP size)

                The script creates test videos with various configurations and then runs benchmarks with:
                - Different thread counts (1, n where n is the number of NVDecs)
                - Different decoding patterns:
                * Sequential decoding of first N frames
                * Uniform sampling of frames
                * Random sampling of frames

                Usage:
                    python pynvc_benchmarking_script.py --nvdecs <number_of_nvdecs> [options]

                Required Arguments:
                    --nvdecs <number>    Number of NVDecs available on the system

                Optional Arguments:
                    --verbose, -v        Show detailed thread-level performance information (default: False)
                    --resolution, -res   Video resolution (default: 1920x1080)
                    --gop, -g           GOP sizes to test, space-separated list (default: 30 250)
                    --duration, -d      Video duration in seconds (default: 30)
                    --fps, -f           Video frames per second (default: 30)
                    --num-seq-frames, -seq  Number of frames to decode sequentially (default: 100)
                    --num-samp-frames, -samp  Number of frames to sample for uniform/random patterns (default: 30)

                Example:
                    python pynvc_benchmarking_script.py --nvdecs 3 --resolution 1280x720 --gop 30 60 --verbose
                    This will run benchmarks with thread counts [1, 3], 720p resolution, GOP sizes 30 and 60,
                    and show detailed thread information.

                Output:
                    The script generates two types of tables:
                    1. Main Results Table: Shows FPS and efficiency for each configuration
                    2. Thread Details Table (with --verbose): Shows per-thread FPS performance

                    Results are also saved to a JSON file for further analysis.
"""


@dataclass
class DecoderPerformance:
    decoder_name: str
    video_file: str
    frames_decoded: int
    thread_execution_times: List[float]
    overall_fps: float
    per_thread_fps: List[float]
    thread_count: int
    test_description: str
    sampling_method: str  # "sequential", "uniform", or "random"
    sequential_decode_time: Optional[float] = None
    sampling_efficiency: Optional[float] = None  # sequential_time / actual_time

class VideoFrameDecoder(ABC):
    def __init__(self, worker_count: int = 4):
        self.worker_count = worker_count
        
    @abstractmethod
    def get_frames_at_timestamps(self, video_path: str, timestamps: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """Retrieve video frames at specified timestamps"""
        pass
        
    @abstractmethod
    def get_sequential_frames(self, video_path: str, frame_count: int, target_timestamp: float = None) -> Tuple[List[np.ndarray], List[float]]:
        """Retrieve frames sequentially from the start of the video"""
        pass
        
    def get_benchmark_description(self, frame_count: int, pattern_type: str) -> str:
        """Generate description for the benchmark test case"""
        if pattern_type == "sequential":
            return f"Sequential retrieval of first {frame_count} frames"
        return f"Sampling {frame_count} frames with {pattern_type} pattern"

class PyNvVideoCodecDecoder(VideoFrameDecoder):
    def __init__(self, worker_count: int = 4):
        super().__init__(worker_count)
        self.decoders = []
        self.video_file = None
        self.lock = threading.Lock()  # For thread-safe decoder initialization
        
    def get_frames_at_timestamps(self, video_path: str, timestamps: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """Retrieve video frames at specified timestamps"""
        with self.lock:
            if not self.decoders:
                self.decoders = [nvc.SimpleDecoder(video_path, bWaitForSessionWarmUp=True) for _ in range(self.worker_count)]
            else:
                for decoder in self.decoders:
                    decoder.reconfigure_decoder(video_path)
            self.video_file = video_path

        nvc.PyNvDecoder.SetSessionCount(self.worker_count)
            
        frames = []
        thread_times = []
        
        def decode_worker(decoder_idx: int, timestamps: List[float]) -> Tuple[List[np.ndarray], float]:
            start_time = time.time()
            batch_frames = []
            decoder = self.decoders[decoder_idx]
            target_indices = []

            for timestamp in timestamps:
                index = decoder.get_index_from_time_in_seconds(timestamp)
                target_indices.append(index)
            
            batch_frames = decoder.get_batch_frames_by_index(target_indices)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            session_init_time = decoder.get_session_init_time()/1000
            elapsed_time -= session_init_time
            return batch_frames, elapsed_time
        
        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            futures = []
            for i in range(self.worker_count):
                futures.append(executor.submit(decode_worker, i, timestamps))
            
            for future in futures:
                batch_frames, thread_time = future.result()
                frames.extend(batch_frames)
                thread_times.append(thread_time)
                
        return frames, thread_times
    
    def get_sequential_frames(self, video_path: str, frame_count: int = None, target_timestamp: float = None) -> Tuple[List[np.ndarray], List[float]]:
        """Retrieve frames sequentially from the start of the video"""
        with self.lock:
            if not self.decoders:
                self.decoders = [nvc.SimpleDecoder(video_path, bWaitForSessionWarmUp=True) for _ in range(self.worker_count)]
            else:
                for decoder in self.decoders:
                    decoder.reconfigure_decoder(video_path)
            self.video_file = video_path

        nvc.PyNvDecoder.SetSessionCount(self.worker_count)
        
        frames = []
        thread_times = []
        
        def decode_worker(decoder_idx: int, frame_count: int, target_timestamp: float = None) -> Tuple[List[np.ndarray], float]:
            batch_frames = []            
            decoder = self.decoders[decoder_idx]
            if target_timestamp:
                frame_count = decoder.get_index_from_time_in_seconds(target_timestamp)

            start_time = time.time()
            for i in range(0, frame_count, 5):
                decoder.seek_to_index(i)
                batch = decoder.get_batch_frames(5)
                batch_frames.extend(batch)
                if len(batch_frames) >= frame_count:
                    break
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            session_init_time = decoder.get_session_init_time()/1000
            elapsed_time -= session_init_time
            return batch_frames[:frame_count], elapsed_time
        
        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            futures = []
            for i in range(self.worker_count):
                futures.append(executor.submit(decode_worker, i, frame_count, target_timestamp))
            
            for future in futures:
                batch_frames, thread_time = future.result()
                frames.extend(batch_frames)
                thread_times.append(thread_time)
                
        return frames, thread_times


def get_system_metadata() -> Dict[str, Any]:
    """Get system metadata for benchmarking results"""
    return {
        "cpu_count": os.cpu_count(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": str(platform.python_version()),
        "cuda": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "not available",
        "cuda_available": torch.cuda.is_available()
    }

def generate_mandelbrot_video(width: int, height: int, duration: int, fps: int, gop: int, output_path: str) -> str:
    """
    Generate a Mandelbrot set video using FFmpeg
    
    Parameters:
        width (int): Video width
        height (int): Video height
        duration (int): Video duration in seconds
        fps (int): Frames per second
        gop (int): GOP size
        output_path (str): Path to save the generated video
        
    Returns:
        str: Path to the generated video
    """
    print(f"Generating Mandelbrot set video at {width}x{height}, {duration}s duration, {fps} fps, {gop} GOP size")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # FFmpeg command to generate a Mandelbrot set video
    cmd = [
        "ffmpeg", "-y",
        "-v", "warning",
        # Video input from mandelbrot filter
        "-f", "lavfi",
        "-i", f"mandelbrot=s={width}x{height}",
        # Set duration
        "-t", str(duration),
        # Video encoding with h264 and specific parameters
        "-c:v", "h264_nvenc",
        "-preset", "medium",
        "-b:v", "8M",
        "-bf", "3",           # 3 B-frames
        "-g", str(gop),       # GOP size
        "-rc", "vbr",         # Variable bitrate
        "-profile:v", "high", # High profile for better quality
        output_path
    ]
    
    try:
        # Execute FFmpeg command and wait for completion
        print("Starting FFmpeg process to generate Mandelbrot video...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error generating video: {stderr.decode()}")
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
            
        print(f"Video generation complete: {output_path}")
        # Print video information
        print(f"Video codec: h264_nvenc with {gop} GOP size and 3 B-frames")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


def generate_benchmark_videos(
    resolution: str,
    duration: int,
    fps: int,
    gop_sizes: List[int],
    output_dir: str
) -> List[str]:
    """
    Generate benchmark videos with different GOP sizes
    
    Parameters:
        resolution (str): Video resolution (e.g., '1920x1080')
        duration (int): Video duration in seconds
        fps (int): Frames per second
        gop_sizes (List[int]): List of GOP sizes to test
        output_dir (str): Directory to save the generated videos
        
    Returns:
        List[str]: List of paths to the generated videos
    """
    width, height = map(int, resolution.split('x'))
    generated_videos = []
    
    for gop in gop_sizes:
        output_path = f"{output_dir}/mandelbrot_{resolution}_{duration}s_{fps}fps_{gop}gop.mp4"
        
        # Skip if video already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_path} - already exists")
            generated_videos.append(output_path)
            continue
            
        try:
            video_path = generate_mandelbrot_video(width, height, duration, fps, gop, output_path)
            generated_videos.append(video_path)
        except Exception as e:
            print(f"Failed to generate video with GOP {gop}: {e}")
            continue
    
    return generated_videos


def get_video_metadata(video_file: str) -> Dict[str, Any]:
    """Get metadata for a video file using PyNvVideoCodec"""
    decoder = nvc.SimpleDecoder(video_file)
    metadata = decoder.get_stream_metadata()
    return {
        "duration": metadata.duration,
        "width": metadata.width,
        "height": metadata.height,
        "fps": metadata.average_fps,
        "codec": metadata.codec_name
    }


def create_timestamp_list(video_file: str, num_samples: int, kind: str = "uniform") -> List[float]:
    """Generate list of timestamps for frame sampling"""
    metadata = get_video_metadata(video_file)
    duration = metadata["duration"]
    
    if kind == "uniform":
        return [i * duration / num_samples for i in range(num_samples)]
    elif kind == "random":
        # Generate unique random indices and scale to duration
        indices = torch.randperm(int(duration * 1000))[:num_samples]  # Convert to milliseconds for more precision
        return (indices.float() / 1000).tolist()  # Convert back to seconds
    else:
        raise ValueError(f"Unknown sampling kind: {kind}")


def run_benchmark_test(
    decoder: VideoFrameDecoder,
    video_file: str,
    num_threads: int,
    num_samples: int,
    decode_pattern: str = "first_n_frames"
) -> DecoderPerformance:
    """Run a single benchmark test for a decoder"""
    description = ""
    sequential_time = None
    efficiency = None
    
    if decode_pattern == "first_n_frames":
        description = decoder.get_benchmark_description(num_samples, "sequential")
        frames, thread_times = decoder.get_sequential_frames(video_file, frame_count=num_samples)
        pattern_str = f"first_{num_samples}_frames"
    else:  # uniform or random
        num_samples = 30
        description = decoder.get_benchmark_description(num_samples, decode_pattern)
        timestamp_list = create_timestamp_list(video_file, num_samples, decode_pattern)
        
        # First measure sequential decode time to the last timestamp
        last_timestamp = max(timestamp_list)
        _, seq_thread_times = decoder.get_sequential_frames(video_file, frame_count=None, target_timestamp=last_timestamp)
        sequential_time = sum(seq_thread_times)
        
        # Then measure random/uniform decode time
        frames, thread_times = decoder.get_frames_at_timestamps(video_file, timestamp_list)
        efficiency = sequential_time / sum(thread_times) if sum(thread_times) > 0 else 0
        pattern_str = f"{num_samples}_{decode_pattern}_frames"
    
    total_frames = len(frames)
    thread_fps = [num_samples / t for t in thread_times]
    total_fps = sum(thread_fps)
    
    return DecoderPerformance(
        decoder_name=decoder.__class__.__name__,
        video_file=video_file,
        frames_decoded=total_frames,
        thread_execution_times=thread_times,
        overall_fps=total_fps,
        per_thread_fps=thread_fps,
        thread_count=num_threads,
        test_description=description,
        sampling_method=pattern_str,
        sequential_decode_time=sequential_time,
        sampling_efficiency=efficiency
    )


def display_benchmark_results(results: List[DecoderPerformance], verbose: bool = False) -> None:
    """Display benchmark results in formatted tables"""

    videos = sorted(set(r.video_file for r in results))
    patterns = sorted(set(r.sampling_method for r in results))
    thread_counts = sorted(set(r.thread_count for r in results))
    
    print("\nBenchmark Results:")
    
    table_data = []
    headers = ["Video Config", "Pattern"]
    
    for thread_count in thread_counts:
        headers.extend([f"T{thread_count} FPS", f"T{thread_count} Eff"])
        
    for video in videos:
        # Get the filename without path and extension
        filename = os.path.basename(video)
        # Split by underscore and remove .mp4 from the last part
        parts = filename.split('_')
        resolution = parts[1]
        duration = parts[2].replace('s', '')
        gop = parts[4].replace('gop.mp4', '')
        config_str = f"{resolution} {gop}gop {duration}s"
        
        for pattern in patterns:
            row = [config_str, pattern]
            
            for thread_count in thread_counts:
                result = next((r for r in results 
                            if r.video_file == video 
                            and r.sampling_method == pattern
                            and r.thread_count == thread_count), None)
                if result:
                    row.append(f"{result.overall_fps:.1f}")
                    if result.sampling_efficiency is not None:
                        row.append(f"{result.sampling_efficiency:.2f}x")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
                    row.append("N/A")
            
            table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Only show thread-level details if verbose mode is enabled
    if verbose:
        print(f"\nThread Details:")

        thread_headers = ["Config", "Threads", "FPS/Thread"]
        thread_data = []
        
        for video in videos:
            # Get the filename without path and extension
            filename = os.path.basename(video)
            # Split by underscore and remove .mp4 from the last part
            parts = filename.split('_')
            resolution = parts[1]
            duration = parts[2].replace('s', '')
            gop = parts[4].replace('gop.mp4', '')
            config_str = f"{resolution} {gop}gop {duration}s"
            
            for pattern in patterns:
                for thread_count in thread_counts:
                    result = next((r for r in results 
                                if r.video_file == video 
                                and r.sampling_method == pattern
                                and r.thread_count == thread_count), None)
                    
                    if result:
                        thread_fps_str = ",".join([f"{fps:.1f}" for fps in result.per_thread_fps])
                        thread_data.append([
                            f"{config_str} {pattern}",
                            thread_count,
                            thread_fps_str
                        ])
        
        # Print thread-level details table
        print(tabulate(thread_data, headers=thread_headers, tablefmt="grid"))
    
    print("\nEfficiency Explanation:")
    print("Efficiency (Eff) represents the performance comparison between two approaches:")
    print("1. Direct sampling: Decoding specific frames directly using seek operations")
    print("2. Sequential decode + sampling: Decoding all frames sequentially and then sampling the required frames")
    print("The efficiency value shows how much faster direct sampling is compared to sequential decoding with sampling.")
    print("Higher efficiency values indicate better performance of direct sampling approach.")


def save_benchmark_results(results: List[DecoderPerformance], output_file: str = "benchmark_results.json") -> None:
    """Save benchmark results to a JSON file"""
    output_data = {
        "system_info": get_system_metadata(),
        "benchmark_results": [
            {
                "decoder": r.decoder_name,
                "video": r.video_file,
                "frames_decoded": r.frames_decoded,
                "thread_execution_times": r.thread_execution_times,
                "overall_fps": r.overall_fps,
                "per_thread_fps": r.per_thread_fps,
                "thread_count": r.thread_count,
                "test_description": r.test_description,
                "sampling_method": r.sampling_method,
                "sequential_decode_time": r.sequential_decode_time,
                "sampling_efficiency": r.sampling_efficiency
            }
            for r in results
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nBenchmark results saved to {output_file}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyNvVideoCodec Benchmarking Script')
    parser.add_argument('--nvdecs', type=int, required=True, 
                      help='Number of NVDecs available on the system')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Show detailed thread-level performance information (default: False)')
    parser.add_argument('--resolution', '-res', type=str, default='1920x1080',
                      help='Video resolution (default: 1920x1080)')
    parser.add_argument('--gop', '-g', type=int, nargs='+', default=[30, 250],
                      help='GOP sizes to test (space-separated list) (default: 30 250)')
    parser.add_argument('--duration', '-d', type=int, default=30,
                      help='Video duration in seconds (default: 30)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                      help='Video frames per second (default: 30)')
    parser.add_argument('--num-seq-frames', '-seq', type=int, default=100,
                      help='Number of frames to decode sequentially (default: 100)')
    parser.add_argument('--num-samp-frames', '-samp', type=int, default=30,
                      help='Number of frames to sample for uniform/random patterns (default: 30)')
    args = parser.parse_args()
    
    # Set thread counts based on number of NVDecs
    thread_counts = [1, args.nvdecs] if args.nvdecs > 1 else [1]
    print(f"Using thread counts: {thread_counts} based on {args.nvdecs} NVDecs")
    
    start_time = time.time()
    
    # Create videos directory in the same location as the script
    script_dir = Path(__file__).parent
    videos_dir_path = script_dir / "benchmark_videos"
    videos_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Using videos directory: {videos_dir_path}")
    
    # Generate benchmark videos
    video_files = generate_benchmark_videos(
        args.resolution,
        args.duration,
        args.fps,
        args.gop,
        str(videos_dir_path)
    )
    
    # Benchmark parameters
    decode_patterns = ["first_n_frames", "uniform", "random"]
    
    results = []
    
    # Run benchmarks
    for video_file in video_files:
        if not os.path.exists(video_file):
            print(f"Skipping {video_file} - file not found")
            continue
            
        for num_threads in thread_counts:
            print(f"\nBenchmarking {video_file} with {num_threads} threads")

            # PyNvVideoCodec benchmark
            pynv_decoder = PyNvVideoCodecDecoder(num_threads)
            for pattern in decode_patterns:
                num_samples = args.num_seq_frames if pattern == "first_n_frames" else args.num_samp_frames
                pynv_result = run_benchmark_test(pynv_decoder, video_file, num_threads, num_samples, pattern)
                results.append(pynv_result)

    # Display and save results
    display_benchmark_results(results, verbose=args.verbose)
    save_benchmark_results(results)
    
    # Display total execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\nTotal execution time: {hours:02d}:{minutes:02d}:{seconds:06.3f} ({total_time:.3f} seconds)")


if __name__ == "__main__":
    main() 