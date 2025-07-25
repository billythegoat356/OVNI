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

from fileinput import filename
import argparse
from pathlib import Path
import os
import subprocess
import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait
import pycuda.autoinit
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from itertools import product
import platform
from typing import List, Dict, Any
from tabulate import tabulate
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import matplotlib.pyplot as plt
import numpy as np
import glob

"""
Cached Decoder Benchmark

This script measures and compares the performance of NVIDIA's video decoder in two modes:
1. Simple Decoder: Creates a new decoder instance for each video
2. Cached Decoder: Reuses the same decoder instance across multiple videos

The benchmark generates test videos in different resolutions (360p to 4K) and durations (2s and 30s),
then measures decoding performance in terms of FPS (Frames Per Second) and total frames decoded.

Features:
- Supports multiple video codecs (H.264, HEVC, AV1)
- Tests different thread counts (1 and N NVDECs)
- Generates performance plots comparing simple vs cached decoder
- Saves results in both JSON and CSV formats
- Creates bar graphs for visual performance comparison

Command Line Arguments:
    --plot-only: Only generate plots from existing JSON files (default: False)
    --codec: Video codec to use (h264, hevc, or av1) (default: h264)
    --nvdecs: Number of NVDECs to use (default: 3)
    --fps: Frame rate for video generation (default: 30)
    --gop: GOP size for video generation (default: 60)

Usage Examples:
    1. Run full benchmark with default settings (H.264, 3 threads):
       python simple_dec_reuse_benchmark.py

    2. Run benchmark with HEVC codec and [1, 4] threads:
       python simple_dec_reuse_benchmark.py --codec hevc --nvdecs 4

    3. Generate plots from existing results:
       python simple_dec_reuse_benchmark.py --plot-only

    4. Run benchmark with custom video settings:
       python simple_dec_reuse_benchmark.py --codec av1 --fps 60 --gop 120

Output Files:
    - JSON results: cached_decoder_performance_{codec}_{threads}_threads.json
    - Performance plots: cached_decoder_performance_{codec}_{threads}_threads.png
    - Test videos: test_videos_{codec}/short_videos/ and test_videos_{codec}/long_videos/

"""

@dataclass
class DecoderResult:
    time_taken: float
    fps: float
    frames: int

@dataclass
class ResolutionResult:
    resolution: str
    new_decoder: DecoderResult
    cached_decoder: DecoderResult

    def to_dict(self):
        return asdict(self)

@dataclass
class DurationResult:
    duration_type: str
    resolution_results: List[ResolutionResult]

    def to_dict(self):
        return {
            "duration_type": self.duration_type,
            "resolution_results": [res.to_dict() for res in self.resolution_results]
        }

def create_test_videos(
    video_configs: List[Dict[str, Any]],
    encoding: str,
    pattern: str,
    fps: int,
    pix_fmt: str,
    ffmpeg_path: str,
    base_output_dir: str,
    duration_type: str,
    resolution: str,
    target_count: int = 1,
    cq: int = 25
) -> None:
    """Create test videos with different configurations using ffmpeg
    
    Args:
        video_configs: List of video configuration dictionaries
        encoding: Video encoding to use (e.g., 'h264_nvenc')
        pattern: Test pattern to use (e.g., 'mandelbrot')
        fps: Frames per second
        pix_fmt: Pixel format (e.g., 'yuv420p')
        ffmpeg_path: Path to ffmpeg executable
        base_output_dir: Base directory for output
        duration_type: Type of duration ('short' or 'long')
        resolution: Resolution name (e.g., '360p')
        target_count: Number of videos to generate per resolution (default: 1)
        cq: Constant Quality value
    """
    # Create resolution-specific output directory
    output_dir = os.path.join(base_output_dir, resolution)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Count existing videos
    existing_videos = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.lower().endswith(".mp4"):
                existing_videos.append(file)
    
    existing_count = len(existing_videos)
    videos_needed = target_count - existing_count

    if videos_needed <= 0:
        print(f"Already have {existing_count} video(s) for {resolution} resolution in {duration_type} duration folder")
        return

    print(f"Generating {videos_needed} video(s) for {resolution} resolution in {duration_type} duration folder")
    videos_created = 0

    # Generate videos for this resolution
    for config in video_configs:
        if videos_created >= videos_needed:
            break

        # Add an index to the filename if generating multiple videos
        index_suffix = f"_{videos_created + 1}" if target_count > 1 else ""
        outfile = f"{output_dir}/{pattern}_{config['resolution']}_{config['duration']}s_{fps}fps_{config['gop']}gop_{encoding}_{pix_fmt}{index_suffix}.mp4"
            
        ffmpeg_cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{pattern}=s={config['resolution']}",
            "-t",
            str(config['duration']),
            "-c:v",
            encoding,
            "-r",
            str(fps),
            "-g",
            str(config['gop']),
            "-pix_fmt",
            pix_fmt,
            "-cq",
            str(cq),
            outfile,
        ]
        
        print(f"Creating video {videos_created + 1}/{videos_needed}: {' '.join(ffmpeg_cmd)}")
        subprocess.check_call(ffmpeg_cmd)
        videos_created += 1

    print(f"Created {videos_created} new video(s) for {resolution} resolution in {duration_type} duration folder")

def split_list(videos, n):
    k, m = divmod(len(videos), n)
    result = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        result.append(videos[start:end])
        start = end
    return result

def decode_using_new_decoder(clips):
    device_id = 0
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    cuda_stream = cuda.Stream()
    total_frames = 0
    for video_file_path in clips:
        decoder = nvc.SimpleDecoder(video_file_path, cuda_context=cuda_ctx.handle, cuda_stream=cuda_stream.handle)
        metadata = decoder.get_stream_metadata()
        num_frames = metadata.num_frames
        for i in range(num_frames):
            frame = decoder[i]
        total_frames += num_frames
    cuda_stream.synchronize()
    cuda_ctx.pop()
    return total_frames

def decode_using_cached_decoder(clips):
    decoder = None
    device_id = 0
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    cuda_stream = cuda.Stream()
    total_frames = 0
    for video_file_path in clips:
        if decoder is None:
            decoder = nvc.SimpleDecoder(video_file_path, 
                                        cuda_context=cuda_ctx.handle, cuda_stream=cuda_stream.handle)
        else:
            decoder.reconfigure_decoder(video_file_path)
        metadata = decoder.get_stream_metadata()
        num_frames = metadata.num_frames
        for i in range(num_frames):
            frame = decoder[i]
        total_frames += num_frames
        cuda_stream.synchronize()

    cuda_ctx.pop()
    return total_frames

def run_thread(func, clips_chunks, N):
    threads = []
    start_time = time.perf_counter()
    total_frames = 0
    
    # Use a list to store results from threads
    results = [0] * N
    
    def thread_wrapper(func, clips, index):
        results[index] = func(clips)
    
    for m in range(0, N):
        t = threading.Thread(target=thread_wrapper, args=(func, clips_chunks[m], m))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    
    total_frames = sum(results)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, total_frames

def decode_perf_results(clips_by_resolution: Dict[str, List[str]], N: int, duration_type: str) -> DurationResult:
    resolution_results = []
    
    for resolution, clips in clips_by_resolution.items():
        clips_chunks = split_list(clips, N)
        
        elapsed_time_case_1, frames_case_1 = run_thread(decode_using_new_decoder,
                             clips_chunks, N)
        
        elapsed_time_case_2, frames_case_2 = run_thread(decode_using_cached_decoder,
                             clips_chunks, N)
        
        fps_case_1 = round(frames_case_1 / elapsed_time_case_1)
        fps_case_2 = round(frames_case_2 / elapsed_time_case_2)
        
        res_result = ResolutionResult(
            resolution=resolution,
            new_decoder=DecoderResult(
                time_taken=round(elapsed_time_case_1, 2),
                fps=fps_case_1,
                frames=frames_case_1
            ),
            cached_decoder=DecoderResult(
                time_taken=round(elapsed_time_case_2, 2),
                fps=fps_case_2,
                frames=frames_case_2
            )
        )
        resolution_results.append(res_result)
    
    return DurationResult(duration_type=duration_type, resolution_results=resolution_results)

def save_results_to_json(all_results: List[DurationResult]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"decoder_performance_results_{timestamp}.json"
    
    # Convert dataclass objects to dictionaries
    results_dict = [result.to_dict() for result in all_results]
    
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\nResults saved to {filename}")

def display_results_table(all_results: List[DurationResult]):
    headers = ["Duration Type", "Resolution", "Decoder Type", "Time Taken (s)", "FPS", "Frames Decoded"]
    table_data = []
    
    duration_labels = {
        "short": "Short (2 sec)",
        "long": "Long (30 sec)"
    }
    
    for result in all_results:
        duration_label = duration_labels.get(result.duration_type, result.duration_type)
        
        # Add duration type header row
        table_data.append([f"{duration_label} Videos"] + ["" for _ in range(len(headers)-1)])
        
        # Sort resolution results by resolution (360p, 480p, etc.)
        def resolution_key(res):
            if res.resolution == '4k':
                return 4000  # 4k is equivalent to 4000p
            return int(res.resolution.replace("p", ""))
        
        sorted_results = sorted(result.resolution_results, key=resolution_key)
        
        for res_result in sorted_results:
            # Add row for simple decoder
            table_data.append([
                "",  # Empty duration type (already shown in header)
                res_result.resolution,
                "Simple",
                res_result.new_decoder.time_taken,
                int(res_result.new_decoder.fps),  # Convert to integer
                res_result.new_decoder.frames
            ])
            # Add row for cached decoder
            table_data.append([
                "",  # Empty duration type (already shown in header)
                res_result.resolution,
                "Cached",
                res_result.cached_decoder.time_taken,
                int(res_result.cached_decoder.fps),  # Convert to integer
                res_result.cached_decoder.frames
            ])
        
        # Add separator between duration types using the same style as header
        if result.duration_type == "short":
            table_data.append(["-" * 20, "-" * 12, "-" * 13, "-" * 16, "-" * 5, "-" * 15])
    
    print("\nDecoder Performance Results:")
    print(tabulate(table_data, headers=headers, tablefmt="simple", floatfmt=".2f"))

def collect_videos(base_dir, resolutions):
    videos_by_resolution = {}
    for res_name, _ in resolutions:
        res_dir = os.path.join(base_dir, res_name)
        if os.path.exists(res_dir):
            videos = []
            for file in os.listdir(res_dir):
                if file.lower().endswith(".mp4"):
                    full_path = os.path.join(res_dir, file)
                    # Add the same video 50 times to the list
                    videos.extend([full_path] * 50)
            if videos:
                videos_by_resolution[res_name] = videos
    return videos_by_resolution

def plot_decoder_performance():
    """Create bar plots for decoder performance across resolutions for each thread count"""
    # Read JSON files for the specific codec
    results = {}
    for filename in glob.glob(f"cached_decoder_performance_{args.codec}_*_threads.json"):
        # Extract thread count from filename
        thread_count = int(filename.split('_')[4])
        with open(filename, 'r') as f:
            results[thread_count] = json.load(f)
    
    if not results:
        print(f"No JSON files found for codec {args.codec}")
        return
    
    # Sort thread counts to ensure correct order
    thread_counts = sorted(results.keys())
    
    # Define resolution order for consistent plotting
    resolution_order = ['360p', '480p', '720p', '1080p', '4k']
    
    for thread_count in thread_counts:
        # Create a figure for each thread count
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for this thread count
        data = {
            'short': {'new': [], 'cached': [], 'resolutions': []},
            'long': {'new': [], 'cached': [], 'resolutions': []}
        }
        
        # Process each duration type
        for duration_result in results[thread_count]:
            duration_type = duration_result['duration_type']
            
            # Sort results by resolution order
            sorted_results = sorted(
                duration_result['resolution_results'],
                key=lambda x: resolution_order.index(x['resolution'])
            )
            
            for res_result in sorted_results:
                data[duration_type]['resolutions'].append(res_result['resolution'])
                data[duration_type]['new'].append(res_result['new_decoder']['fps'])
                data[duration_type]['cached'].append(res_result['cached_decoder']['fps'])
        
        # Plot for short duration videos
        x = np.arange(len(data['short']['resolutions']))
        width = 0.35
        
        ax1.bar(x - width/2, data['short']['new'], width, label='Simple Decoder')
        ax1.bar(x + width/2, data['short']['cached'], width, label='Cached Decoder')
        ax1.set_title(f'Short Duration Videos (2s) - {thread_count} Thread(s) - {args.codec.upper()}')
        ax1.set_xlabel('Resolution')
        ax1.set_ylabel('FPS')
        ax1.set_xticks(x)
        ax1.set_xticklabels(data['short']['resolutions'])
        ax1.grid(True, axis='y')
        ax1.legend()
        
        # Plot for long duration videos
        x = np.arange(len(data['long']['resolutions']))
        ax2.bar(x - width/2, data['long']['new'], width, label='Simple Decoder')
        ax2.bar(x + width/2, data['long']['cached'], width, label='Cached Decoder')
        ax2.set_title(f'Long Duration Videos (30s) - {thread_count} Thread(s) - {args.codec.upper()}')
        ax2.set_xlabel('Resolution')
        ax2.set_ylabel('FPS')
        ax2.set_xticks(x)
        ax2.set_xticklabels(data['long']['resolutions'])
        ax2.grid(True, axis='y')
        ax2.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'cached_decoder_performance_{args.codec}_{thread_count}_threads.png')
        plt.close()
        
        print(f"Plot saved as 'cached_decoder_performance_{args.codec}_{thread_count}_threads.png'")

if __name__=="__main__":
    script_start_time = time.perf_counter()
    
    parser = argparse.ArgumentParser(
        'This sample application measures decoding performance in FPS.'
    )
    parser.add_argument(
        "--plot-only", action="store_true", default=False,
        help="only generate plots from existing JSON files (default: False)")
    parser.add_argument(
        "--codec", type=str, choices=['h264', 'hevc', 'av1'], default='h264',
        help="video codec to use (h264, hevc, or av1)")
    parser.add_argument(
        "--nvdecs", type=int, default=3, help="number of NVDECs to use (default: 3)")
    parser.add_argument(
        "--fps", type=int, default=30, help="frame rate for video generation (default: 30)")
    parser.add_argument(
        "--gop", type=int, default=60, help="GOP size for video generation (default: 60)")
    
    args = parser.parse_args()

    if args.plot_only:
        plot_decoder_performance()
        exit(0)

    # Define thread counts to test - only 1 and max threads
    num_of_threads = [1, args.nvdecs] if args.nvdecs > 1 else [1]

    # Define resolutions with their dimensions
    resolutions = [
        ("360p", "640x360"),
        ("480p", "854x480"),
        ("720p", "1280x720"),
        ("1080p", "1920x1080"),
        ("4k", "3840x2160")
    ]

    # Use the test videos directory in the same directory as the script
    base_videos_dir = os.path.join(os.path.dirname(__file__), f"test_videos_{args.codec}")
    short_videos_dir = os.path.join(base_videos_dir, "short_videos")
    long_videos_dir = os.path.join(base_videos_dir, "long_videos")

    # Create base directories if they don't exist
    os.makedirs(short_videos_dir, exist_ok=True)
    os.makedirs(long_videos_dir, exist_ok=True)

    # Define video generation parameters
    codec_map = {
        'h264': 'h264_nvenc',
        'hevc': 'hevc_nvenc',
        'av1': 'av1_nvenc'
    }
    encoding = codec_map[args.codec]  # Use selected codec
    pattern = "mandelbrot"  # Test pattern
    fps = args.fps  # Frame rate from command line
    pix_fmt = "yuv420p"  # Standard pixel format
    gop_size = args.gop  # GOP size from command line
    cq = 25  # Constant Quality value
   
    
    # Find ffmpeg in a platform-independent way
    ffmpeg_path = "ffmpeg"
    if platform.system() == "Windows":
        # Try to find ffmpeg in PATH
        for path in os.environ["PATH"].split(os.pathsep):
            ffmpeg_exe = Path(path) / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                ffmpeg_path = str(ffmpeg_exe)
                break

    # Generate videos for each resolution
    for res_name, res_dim in resolutions:
        # Short duration videos
        short_video_configs = [{
            "resolution": res_dim,
            "gop": gop_size,
            "duration": 2  # Fixed 2 seconds
        }]
        create_test_videos(
            short_video_configs,
            encoding,
            pattern,
            fps,
            pix_fmt,
            ffmpeg_path,
            str(short_videos_dir),
            "short",
            res_name,
            1,  # Always generate 1 video per resolution
            cq
        )

        # Long duration videos
        long_video_configs = [{
            "resolution": res_dim,
            "gop": gop_size,
            "duration": 30  # Fixed 30 seconds
        }]
        create_test_videos(
            long_video_configs,
            encoding,
            pattern,
            fps,
            pix_fmt,
            ffmpeg_path,
            str(long_videos_dir),
            "long",
            res_name,
            1,  # Always generate 1 video per resolution
            cq
        )

    # Collect videos from all resolution directories
    short_clips = collect_videos(short_videos_dir, resolutions)
    long_clips = collect_videos(long_videos_dir, resolutions)

    all_results = []

    # Run benchmark for each thread count
    for thread_count in num_of_threads:
        print(f"\nRunning benchmark with {thread_count} thread(s)")
        
        if short_clips:
            print(f"\nProcessing short duration videos:")
            for res, clips in short_clips.items():
                print(f"  Generating stats for {res} short videos")
                print(f"  {res}: {len(clips)} video(s)")
            short_results = decode_perf_results(short_clips, thread_count, "short")
            all_results.append(short_results)

        if long_clips:
            print(f"\nProcessing long duration videos:")
            for res, clips in long_clips.items():
                print(f"  Generating stats for {res} long videos")
                print(f"  {res}: {len(clips)} video(s)")
            long_results = decode_perf_results(long_clips, thread_count, "long")
            all_results.append(long_results)
        
        # Print video configuration details
        print("\nVideo Configuration:")
        print(f"Pattern: {pattern}")
        print(f"Frame Rate: {fps} fps")
        print(f"GOP Size: {gop_size}")
        print(f"Number of threads: {thread_count}")

        if all_results:
            # Display tabulated results
            display_results_table(all_results)
            
            # Save results to JSON file with codec and thread count in filename
            filename = f"cached_decoder_performance_{args.codec}_{thread_count}_threads.json"
            
            # Convert dataclass objects to dictionaries
            results_dict = [result.to_dict() for result in all_results]
            
            with open(filename, 'w') as f:
                json.dump(results_dict, f, indent=4)
            
            print(f"\nResults saved to {filename}")
            
            # Clear results for next thread count
            all_results = []
        else:
            print("No videos were found or generated. Please check the directory structure and video generation process.")
    
    # After running the benchmark, generate plots
    plot_decoder_performance()
    
    # Print total execution time
    total_time = time.perf_counter() - script_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds") 