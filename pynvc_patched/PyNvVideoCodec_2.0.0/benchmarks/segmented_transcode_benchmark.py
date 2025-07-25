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
import subprocess
import time
import random
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import PyNvVideoCodec as nvc

import pycuda.driver as cuda
import threading


"""
                This script demonstrates generating a Mandelbrot set video with FFmpeg
                and then transcoding it using multiple threads with PyNvVideoCodec.
 
                Parameters:
                    - gpu_id (int): Ordinal of GPU to use
                    - width (int): Width of the generated video
                    - height (int): Height of the generated video
                    - duration (int): Duration of the generated video in seconds
                    - fps (int): Frames per second
                    - preset (str): Encoder preset (P1-P7)
                    - codec (str): Output codec (h264, hevc, av1)
                    - numthreads (int): Number of concurrent threads for parallel processing
                    - segments (int): Number of random segments to create
                    - usage (int): Transcoding mode selection (can specify multiple):
                        0: PyNVC transcoding (output in pynvc_out/)
                        1: FFmpeg without map (output in ffmpeg_out/)
                        2: FFmpeg with map, no audio (output in ffmpeg_fc_out/)
                        3: FFmpeg with map and audio (output in ffmpeg_fc_out/)
                    - log (str): Path to save the execution log (default: logs/run_{timestamp}.json)
                    - replay (str): Path to a log file to replay a previous run
 
                Returns: - None.
 
                Example:
                >>> python mandelbrot_transcode.py -g 0 -W 1920 -H 1080 -d 10 -fps 30 -p P4 -c h264 -n 2 -s 5 -u 0
                
                Replay mode:
                >>> python mandelbrot_transcode.py --replay logs/run_20240615_123045.json
"""

# Global log file handle
log_file = None
log_data = {
    "timestamp": "",
    "args": {},
    "steps": [],
    "results": {}
}

def log_step(step_type, description, params=None, results=None):
    """
    Log a step to both console and log file

    Parameters:
        step_type (str): Type of step (video_generation, segment_creation, transcoding)
        description (str): Human-readable description
        params (dict): Parameters used for this step
        results (dict): Results from this step
    """
    global log_file, log_data

    step = {
        "type": step_type,
        "description": description,
        "timestamp": datetime.datetime.now().isoformat(),
        "params": params or {},
        "results": results or {}
    }

    # Add to log data
    log_data["steps"].append(step)

    # Print to console
    print(f"[LOG] {description}")

    # Write to log file if open
    if log_file:
        # Update the file with the latest log data
        log_file.seek(0)
        json.dump(log_data, log_file, indent=2)
        log_file.truncate()
        log_file.flush()


def generate_mandelbrot_video(width, height, duration, fps, output_path, gop_size, input_codec="h264", input_bframes=2):
    """
    Generate a Mandelbrot set zoom video with audio using FFmpeg
    
    Parameters:
        width (int): Video width
        height (int): Video height
        duration (int): Video duration in seconds
        fps (int): Frames per second
        output_path (str): Path to save the generated video
        gop_size (int): GOP size for encoding
        input_codec (str): Codec to use for video generation (h264, hevc, av1)
        input_bframes (int): Number of B-frames for video generation
        
    Returns:
        str: Path to the generated video
    """
    # Map codec to FFmpeg encoder name
    encoder_map = {
        "h264": "h264_nvenc",
        "hevc": "hevc_nvenc",
        "av1": "av1_nvenc"
    }
    encoder = encoder_map.get(input_codec.lower(), "h264_nvenc")
    
    # Update output path to include codec information
    base, ext = os.path.splitext(output_path)
    codec_output_path = f"{base}_{input_codec}_{width}x{height}{ext}"
    
    # Log the step
    log_step("video_generation", f"Generating Mandelbrot video ({width}x{height}, {duration}s, {fps} fps, {input_codec}, B-frames: {input_bframes})",
             params={"width": width, "height": height, "duration": duration, "fps": fps,
                    "output_path": codec_output_path, "gop_size": gop_size, "codec": input_codec, "bframes": input_bframes})

    print(f"Generating Mandelbrot set video with audio at {width}x{height}, {duration}s duration, {fps} fps")
    print(f"Using codec: {input_codec}, encoder: {encoder}, B-frames: {input_bframes}")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(codec_output_path) or '.', exist_ok=True)

    print(f"Using GOP size: {gop_size} and {input_bframes} B-frames")

    # FFmpeg command to generate a Mandelbrot set zoom video with audio
    cmd = [
        "ffmpeg", "-y",
        "-v", "warning",
        # Video input from mandelbrot filter
        "-f", "lavfi",
        "-i", f"mandelbrot=size={width}x{height}:rate={fps}:start_scale=3:end_scale=0.1",
        # Audio input - sine wave that increases in frequency during the video
        "-f", "lavfi",
        "-i", f"sine=frequency=220:sample_rate=44100:duration={duration}",
        # Set duration
        "-t", str(duration),
        # Video encoding with specified codec
        "-c:v", encoder,
        "-preset", "medium",
        "-g", str(gop_size),  # Convert gop_size to string here
        "-bf", str(input_bframes),  # Set B-frames from parameter
        # Audio encoding
        "-c:a", "aac",
        "-b:a", "128k",
        # Output
        codec_output_path
    ]

    try:
        # Execute FFmpeg command and wait for completion
        res = ' '.join(cmd)
        print(f"{res=}")
        print("Starting FFmpeg process to generate Mandelbrot video with audio...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error generating video: {stderr.decode()}")
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr)

        print(f"Video generation with audio complete: {codec_output_path}")
        # Print video information
        print(f"Video codec: {encoder} with {gop_size} GOP size and {input_bframes} B-frames")
        return codec_output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


def generate_random_segments(duration, num_segments, min_segment_duration=1.0):
    """
    Generate random non-overlapping start and end timestamps for video segments

    Parameters:
        duration (float): Total video duration in seconds
        num_segments (int): Number of segments to generate
        min_segment_duration (float): Minimum duration of each segment in seconds

    Returns:
        list: List of tuples containing (start_time, end_time) for each segment
    """
    segments = []

    # Make sure we can fit all segments with minimum duration
    if num_segments * min_segment_duration > duration:
        num_segments = int(duration / min_segment_duration)
        print(f"Warning: Reduced number of segments to {num_segments} to ensure minimum segment duration")

    # Calculate max segment duration to ensure all segments fit
    max_segment_duration = (duration / num_segments) * 2

    # Divide the video into regions to ensure better distribution
    time_points = sorted([random.uniform(0, duration) for _ in range(num_segments * 2)])
    time_points = [0] + time_points + [duration]

    # Track available time regions
    available_regions = [(time_points[i], time_points[i+1])
                        for i in range(len(time_points)-1)
                        if time_points[i+1] - time_points[i] >= min_segment_duration]

    # Generate segments
    for _ in range(num_segments):
        if not available_regions:
            break

        # Select a random region
        region_idx = random.randint(0, len(available_regions) - 1)
        region_start, region_end = available_regions.pop(region_idx)

        # Make sure region is large enough
        if region_end - region_start < min_segment_duration:
            continue

        # Calculate segment within the region
        segment_duration = min(
            random.uniform(min_segment_duration, max_segment_duration),
            region_end - region_start
        )

        # Randomize start position within the region constraint
        max_start = region_end - segment_duration
        start_time = random.uniform(region_start, max_start) if region_start < max_start else region_start
        end_time = start_time + segment_duration

        segments.append((start_time, end_time))

    # If we didn't get enough segments, add more deterministically
    if len(segments) < num_segments:
        remaining = num_segments - len(segments)
        segment_duration = max(min_segment_duration, duration / (remaining * 2))

        # Find gaps between existing segments
        all_times = [(s, True) for s, e in segments] + [(e, False) for s, e in segments]
        all_times.sort()

        # Add video start and end if not already included
        if not all_times or all_times[0][0] > 0:
            all_times.insert(0, (0, False))
        if not all_times or all_times[-1][0] < duration:
            all_times.append((duration, True))

        # Find gaps
        gaps = []
        for i in range(len(all_times) - 1):
            if all_times[i][1] and not all_times[i+1][1]:  # End followed by start
                gap_start = all_times[i][0]
                gap_end = all_times[i+1][0]
                if gap_end - gap_start >= segment_duration:
                    gaps.append((gap_start, gap_end))

        # Add segments in gaps
        for gap_start, gap_end in gaps:
            if len(segments) >= num_segments:
                break

            available_duration = gap_end - gap_start
            num_to_add = min(remaining, int(available_duration / segment_duration))

            for i in range(num_to_add):
                start = gap_start + (i * segment_duration)
                end = start + segment_duration
                segments.append((start, end))
                remaining -= 1

    # Sort segments by start time
    segments.sort()

    return segments[:num_segments]  # Ensure we don't return more than requested


def run_subprocess(cmd, shell=False, description=""):
    """
    Run a subprocess and wait for completion, with proper error handling

    Parameters:
        cmd: Command to run (list or string)
        shell: Whether to use shell execution
        description: Description of the process for logging

    Returns:
        tuple: (stdout, stderr) from the process
    """
    try:
        print(f"Starting {description}...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, shell=shell)

        # Wait for the process to complete
        stdout, stderr = process.communicate()

        # Check if process completed successfully
        if process.returncode != 0:
            print(f"Error in {description}: {stderr}")
            print(f"Command was: {cmd}")
        else:
            print(f"{description} completed successfully")

        return stdout, stderr
    except Exception as e:
        print(f"Exception in {description}: {e}")
        raise


def segmented_transcode(args):
    """
    Transcode the input video using the specified method

    Parameters:
        args (list): List of parameters including video path, clip_start_end, config, usage, etc.
    """
    video_path, clip_start_end, config, fps, usage, chunk_size = args
    print(f"Segmented transcode for {video_path}")
    print(f"Number of segments: {len(clip_start_end)}")
    print(f"Using mode: {usage}")

    # Get B-frames setting from config
    bframes = config.get("bf", 0)  # Default to 0 if not specified
    print(f"Using {bframes} B-frames for transcoding")
    
    # Properly extract base filename without extension
    base, _ = os.path.splitext(os.path.basename(video_path))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

    # Start timing
    start_time = time.time()

    # Get codec and resolution from config
    codec = config.get("codec", "h264")
    width = config.get("width", 1920)
    height = config.get("height", 1080)
    resolution = f"{width}x{height}"
    
    # Create a resolution-codec-bframes suffix for file names
    suffix = f"_{codec}_{resolution}_bf{bframes}"
    
    # Determine FFmpeg encoder based on output codec
    encoder_map = {
        "h264": "h264_nvenc",
        "hevc": "hevc_nvenc",
        "av1": "av1_nvenc"
    }
    encoder = encoder_map.get(codec.lower(), "h264_nvenc")
    
    # Determine input video codec and decoder to use
    # First try to get from input file using ffprobe
    input_codec = "h264"  # Default assumption
    try:
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                     "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path]
        probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if probe_result.returncode == 0 and probe_result.stdout.strip():
            input_codec = probe_result.stdout.strip().lower()
            print(f"Detected input codec: {input_codec}")
    except Exception as e:
        print(f"Error detecting input codec: {str(e)}. Using default h264 decoder.")
    
    # Map to cuvid decoder
    decoder_map = {
        "h264": "h264_cuvid",
        "avc": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "av1": "av1_cuvid"
    }
    decoder = decoder_map.get(input_codec, "h264_cuvid")
    print(f"Using decoder: {decoder}")

    # Create output directories if they don't exist
    os.makedirs("pynvc_out", exist_ok=True)
    os.makedirs("ffmpeg_out", exist_ok=True)
    os.makedirs("ffmpeg_fc_out", exist_ok=True)

    try:
        if usage == 0:
            # PyNVC Usage
            out_file_path = os.path.join("pynvc_out", f"{base}{suffix}_transcoded.mp4")
            print(f"Output will be saved to: ./pynvc_out/")
            
            # Make a copy of the config and remove width and height
            transcoder_config = config.copy()
            if 'width' in transcoder_config:
                del transcoder_config['width']
            if 'height' in transcoder_config:
                del transcoder_config['height']
            
            # Pass the modified config to the Transcoder
            tr = nvc.Transcoder(video_path, out_file_path, 0, 0, 0, **transcoder_config)
            for start, end in clip_start_end:
                tr.segmented_transcode(start, end)

        elif usage == 1:
            print(f"FFmpeg without map")
            print(f"Output files will be saved to: ./ffmpeg_out/")
            for i in range(0, len(clip_start_end), chunk_size):
                chunk = clip_start_end[i:i+chunk_size]
                for start, end in chunk:
                    # Include codec and resolution in output file path
                    out_file_path = f"./ffmpeg_out/{base}{suffix}_{start:.2f}_{end:.2f}.mp4"
                    ff = (" ffmpeg -loglevel warning -hide_banner -y "
                            f" -ss {str(start)} "
                            f" -to {str(end)} "
                            f" -c:v {decoder} "  # Use appropriate decoder based on input
                            f" -i {video_path} "
                            f" -bf {bframes}"  # Use B-frames from config
                            f" -c:v {encoder} "  # Use the correct encoder based on output codec
                            " -c:a copy "
                            f"{out_file_path}")

                    desc = f"FFmpeg transcode segment {start:.2f}s - {end:.2f}s"
                    run_subprocess(ff, shell=True, description=desc)

        elif usage == 2:
            print(f"FFmpeg with map but no audio")
            print(f"Output files will be saved to: ./ffmpeg_fc_out/")
            for i in range(0, len(clip_start_end), chunk_size):
                chunk = clip_start_end[i:i+chunk_size]

                segment_cmd = ["ffmpeg", "-loglevel", "warning", "-hide_banner", "-y", 
                              "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                              "-c:v", decoder,  # Use appropriate decoder
                              "-i", video_path]

                filter_complex = []
                map_cmds = []
                output_files = []

                for j, (start, end) in enumerate(chunk):
                    # Include codec and resolution in output file path
                    out_file_path = f"./ffmpeg_fc_out/{base}{suffix}_{start:.2f}_{end:.2f}.mp4"
                    output_files.append(out_file_path)

                    # Add video filter
                    filter_complex.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{j}]")
                    map_cmds.extend(["-map", f"[v{j}]", "-c:v", encoder, "-an", out_file_path])

                # Combine all filters
                segment_cmd.extend(["-filter_complex", ";".join(filter_complex)])
                segment_cmd.extend(map_cmds)

                # Convert the list to a shell command string
                cmd_str = " ".join([
                    f'"{arg}"' if (" " in arg or ";" in arg or "[" in arg) else arg
                    for arg in segment_cmd
                ])

                # Execute FFmpeg command
                desc = f"FFmpeg batch transcode {len(chunk)} segments (no audio)"
                run_subprocess(cmd_str, shell=True, description=desc)

        elif usage == 3:
            print(f"FFmpeg with map and audio")
            print(f"Output files will be saved to: ./ffmpeg_fc_out/")
            for i in range(0, len(clip_start_end), chunk_size):
                chunk = clip_start_end[i:i+chunk_size]

                segment_cmd = ["ffmpeg", "-y", "-loglevel", "warning",
                           "-hwaccel", "cuda",
                           "-c:v", decoder,  # Use appropriate decoder
                           "-i", video_path]

                filter_complex = []
                map_cmds = []
                output_files = []

                for j, (start, end) in enumerate(chunk):
                    # Include codec and resolution in output file path
                    out_file_path = f"./ffmpeg_fc_out/{base}{suffix}_{start:.2f}_{end:.2f}.mp4"
                    output_files.append(out_file_path)

                    # Modify filters to include accurate timing
                    filter_complex.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{j}]")

                    # Check if the video has audio
                    if os.path.exists(video_path):
                        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path]
                        probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        has_audio = "audio" in probe_result.stdout

                        if has_audio:
                            filter_complex.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{j}]")
                            map_cmds.extend(["-map", f"[v{j}]", "-map", f"[a{j}]"])
                        else:
                            map_cmds.extend(["-map", f"[v{j}]"])
                    else:
                        map_cmds.extend(["-map", f"[v{j}]"])

                    # Convert P4 to p4 (lowercase) for FFmpeg preset
                    ffmpeg_preset = config.get("preset", "p4").lower()

                    # Add encoding parameters
                    map_cmds.extend([
                        "-c:v", encoder,  # Use appropriate encoder
                        "-preset", ffmpeg_preset,
                        "-rc", "vbr",
                        "-b:v", "5M",
                        "-maxrate", "10M",
                        "-bufsize", "10M",
                    ])

                    # Add audio encoding if we have audio
                    audio_filter = f"a{j}" in "".join(filter_complex)
                    if audio_filter:
                        map_cmds.extend(["-c:a", "aac", "-b:a", "192k"])

                    map_cmds.append(out_file_path)

                # Combine all filters
                segment_cmd.extend(["-filter_complex", ";".join(filter_complex)])
                segment_cmd.extend(map_cmds)

                # Convert the list to a shell command string
                cmd_str = " ".join([
                    f'"{arg}"' if (" " in arg or ";" in arg or "[" in arg) else arg
                    for arg in segment_cmd
                ])

                # Execute FFmpeg command
                desc = f"FFmpeg batch transcode {len(chunk)} segments (with audio)"
                run_subprocess(cmd_str, shell=True, description=desc)

        else:
            print(f"Error: Invalid usage mode {usage}. Must be 0, 1, 2, or 3")

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        raise


def extract_segments_from_log(log_data):
    """
    Extract segment information from log data

    Parameters:
        log_data (dict): Log data from a previous run

    Returns:
        list: List of (start_time, end_time) tuples
    """
    try:
        # Find segment creation step
        for step in log_data.get("steps", []):
            if step.get("type") == "segment_creation":
                if "results" in step and "segments" in step["results"]:
                    segments = step["results"]["segments"]
                    return [(segment["start"], segment["end"]) for segment in segments]

        return []

    except Exception as e:
        print(f"Error parsing segments from log: {e}")
        return []


def replay_from_log(log_path):
    """
    Replay a previous run from a log file

    Parameters:
        log_path (str): Path to the log file
    """
    global log_data

    print(f"Replaying from log: {log_path}")

    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)

        print(f"Log timestamp: {log_data['timestamp']}")
        print(f"Original arguments: {json.dumps(log_data['args'], indent=2)}")

        # Extract video path from the first video generation step
        video_path = None
        for step in log_data['steps']:
            if step['type'] == 'video_generation':
                video_path = step['params']['output_path']
                break

        if not video_path:
            print("Error: No video generation step found in log")
            return

        # Check if the video exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            print("Regenerating video...")

            # Find the video generation step and extract parameters
            for step in log_data['steps']:
                if step['type'] == 'video_generation':
                    params = step['params']
                    video_path = generate_mandelbrot_video(
                        params['width'],
                        params['height'],
                        params['duration'],
                        params['fps'],
                        params['output_path'],
                        params['gop_size'],
                        params['input_codec'],
                        params['input_bframes']
                    )
                    break
        else:
            print(f"Using existing video: {video_path}")

        # Extract segments
        clip_start_end = extract_segments_from_log(log_data)
        if not clip_start_end:
            print("Error: No segments found in log")
            return

        # Extract usage modes
        usage_modes = []
        for step in log_data['steps']:
            if step['type'] == 'transcoding' and 'usage' in step['params']:
                if step['params']['usage'] not in usage_modes:
                    usage_modes.append(step['params']['usage'])

        if not usage_modes:
            print("Error: No transcoding steps found in log")
            return

        print(f"Replay will use these usage modes: {usage_modes}")

        # Extract config
        config = log_data['args'].get('config', {"preset": "P4", "codec": "h264", "tuning_info": "high_quality", "bf": "0"})

        # Calculate total frames
        fps = log_data['args'].get('fps', 30)
        segment_frames = sum((end - start) * fps for start, end in clip_start_end)

        # Store results for comparison
        results = {}

        # Run transcoding for each usage mode
        for usage in usage_modes:
            if usage not in [0, 1, 2, 3]:
                print(f"Warning: Invalid usage mode {usage}. Skipping...")
                continue

            print("\n" + "=" * 80)
            print(f"REPLAY: TRANSCODING SEGMENTS USING MODE {usage}")
            print("=" * 80)

            # Get mode description
            mode_desc = {
                0: "PyNVC transcoding",
                1: "FFmpeg without map",
                2: "FFmpeg with map (no audio)",
                3: "FFmpeg with map (with audio)"
            }.get(usage, "Unknown mode")

            print(f"Running: {mode_desc}")

            futures_list = []
            starttime = time.time()

            # Extract numthreads and chunk_size
            numthreads = log_data['args'].get('numthreads', 1)
            chunk_size = log_data['args'].get('chunk_size', 5)

            with ThreadPoolExecutor(max_workers=numthreads) as executor:
                # Submit task to thread pool
                futures = [executor.submit(segmented_transcode, [
                    video_path,
                    clip_start_end,
                    config,
                    fps,
                    usage,
                    chunk_size
                ])]
                futures_list.append(futures)

                # Wait for all tasks to complete
                for future in futures_list:
                    wait(future)

            endtime = time.time()
            total_time = endtime - starttime
            throughput = segment_frames / total_time if total_time > 0 else 0

            # Store results
            results[usage] = {
                'mode': mode_desc,
                'time': total_time,
                'throughput': throughput
            }

            print(f"Completed {mode_desc}")
            print(f"Processing time: {total_time:.2f} seconds")
            print(f"Throughput: {throughput:.2f} frames/second")

            # Log the results
            log_step("replay_results", f"Replay results for mode {usage}",
                    params={"usage": usage, "mode": mode_desc},
                    results={"time": total_time, "throughput": throughput})

        # Print comparison results with additional details
        print("\n" + "=" * 80)
        print("REPLAY COMPARISON RESULTS")
        print("=" * 80)

        # Add video details from the log data
        width = log_data['args'].get('width', 'unknown')
        height = log_data['args'].get('height', 'unknown')
        codec = log_data['args'].get('codec', 'unknown')
        duration = log_data['args'].get('duration', 'unknown')
        gop_size = log_data['args'].get('gop_size', 'unknown')
        num_segments = len(clip_start_end)

        print(f"Video Resolution: {width}x{height}")
        print(f"Codec: {codec}")
        print(f"Duration: {duration} seconds")
        print(f"Number of Segments: {num_segments}")
        print(f"GOP Size: {gop_size}")
        print()

        print(f"{'Mode':<30} {'Time (s)':<15} {'Throughput (fps)':<20}")
        print("-" * 65)

        for usage, data in sorted(results.items()):
            print(f"{data['mode']:<30} {data['time']:<15.2f} {data['throughput']:<20.2f}")

        # Find fastest method
        if results:
            fastest = min(results.items(), key=lambda x: x[1]['time'])
            highest_throughput = max(results.items(), key=lambda x: x[1]['throughput'])

            print("\nResults Summary:")
            print(f"Total frames transcoded: {segment_frames:.0f}")
            print(f"Fastest method: {results[fastest[0]]['mode']} ({fastest[1]['time']:.2f}s)")
            print(f"Highest throughput: {results[highest_throughput[0]]['mode']} ({highest_throughput[1]['throughput']:.2f} fps)")

            # Log summary with additional details
            log_step("summary", "Transcoding summary",
                    results={
                        "resolution": f"{width}x{height}",
                        "codec": codec,
                        "duration": duration,
                        "num_segments": num_segments,
                        "gop_size": gop_size,
                        "total_frames": segment_frames,
                        "fastest_method": results[fastest[0]]['mode'],
                        "fastest_time": fastest[1]['time'],
                        "highest_throughput_method": results[highest_throughput[0]]['mode'],
                        "highest_throughput": highest_throughput[1]['throughput']
                    })

        print("=" * 80)

        # Compare with original results if available
        if 'results' in log_data and log_data['results']:
            print("\n" + "=" * 80)
            print("COMPARISON WITH ORIGINAL RUN")
            print("=" * 80)

            print(f"{'Mode':<30} {'Original Time':<15} {'Replay Time':<15} {'Original FPS':<15} {'Replay FPS':<15}")
            print("-" * 90)

            for usage, data in sorted(results.items()):
                if str(usage) in log_data['results']:
                    orig_data = log_data['results'][str(usage)]
                    print(f"{data['mode']:<30} {orig_data['time']:<15.2f} {data['time']:<15.2f} {orig_data['throughput']:<15.2f} {data['throughput']:<15.2f}")

            print("=" * 90)

    except Exception as e:
        print(f"Error during replay: {e}")
        raise

def split_list(n, clip_start_end):
    k, m = divmod(len(clip_start_end), n)
    result = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        result.append(clip_start_end[start:end])
        start = end
    return result


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark segmented transcoding approaches")
    parser.add_argument("-g", "--gpuid", type=int, default=0, help="GPU ID")
    parser.add_argument("-W", "--width", type=int, default=1920, help="Video width")
    parser.add_argument("-H", "--height", type=int, default=1080, help="Video height")
    parser.add_argument("-d", "--duration", type=int, default=10, help="Video duration in seconds")
    parser.add_argument("-fps", "--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("-p", "--preset", type=str, default="P4", help="Encoding preset (P1-P7)")
    parser.add_argument("-ic", "--input-codec", type=str, default="h264", 
                      choices=["h264", "hevc", "av1"], 
                      help="Input codec to use for generating the video")
    parser.add_argument("-c", "--codec", type=str, default="h264", 
                      choices=["h264", "hevc", "av1"], 
                      help="Output codec to use for transcoding")
    parser.add_argument("-ibf", "--input-bframes", type=int, default=2, 
                      help="Number of B-frames for input video generation")
    parser.add_argument("-bf", "--bframes", type=int, default=0, 
                      help="Number of B-frames for output transcoding")
    parser.add_argument("-n", "--numthreads", type=int, default=1, help="Number of threads")
    parser.add_argument("-s", "--segments", type=int, default=3, help="Number of random segments")
    parser.add_argument("-u", "--usage", type=int, action="append", default=[], 
                      help="Transcoding method (0: PyNVC, 1: FFmpeg without map, 2: FFmpeg stream-aware, 3: FFmpeg frame-by-frame)")
    parser.add_argument("--chunk-size", type=int, default=5, help="Number of segments to process in a single FFmpeg command")
    parser.add_argument("--segment-duration", type=int, default=5, help="Duration of each segment in seconds")
    parser.add_argument("--log", type=str, help="Path to save log file")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducible segment generation")
    parser.add_argument("--gop-size", type=int, default=250, help="GOP size for encoding")
    parser.add_argument("-i", "--input", type=str, help="Input video file (instead of generating Mandelbrot)")
    parser.add_argument("--replay", type=str, help="Path to log file for replay")
    
    args = parser.parse_args()
    
    # Set default usages if none provided
    if not args.usage:
        args.usage = [0, 1, 2, 3]
        
    return args


if __name__ == "__main__":
    args = parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Generate default log filename if not specified
    if not args.log and not args.replay:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = f"logs/run_{timestamp}.json"

    # Initialize log data
    log_data["timestamp"] = datetime.datetime.now().isoformat()
    log_data["args"] = vars(args)

    # Replay mode
    if args.replay:
        # Open log file in append mode to add replay results
        replay_log_path = f"logs/replay_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(replay_log_path, 'w') as log_file:
            log_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "replay_source": args.replay,
                "steps": [],
                "results": {}
            }
            json.dump(log_data, log_file, indent=2)
            replay_from_log(args.replay)
    else:
        # Normal execution mode
        # Open log file
        log_file = open(args.log, 'w')
        json.dump(log_data, log_file, indent=2)

        print(f"Logging to: {args.log}")

        # In the video generation part
        if not args.input:
            # Generate the Mandelbrot video with the specified input codec
            video_path = "source_videos/mandelbrot.mp4"
            os.makedirs("source_videos", exist_ok=True)

            print("=" * 80)
            print(f"STEP 1: GENERATING MANDELBROT VIDEO WITH {args.input_codec.upper()} CODEC")
            print(f"Using {args.input_bframes} B-frames for input video generation")
            print("=" * 80)
            
            video_path = generate_mandelbrot_video(
                args.width, args.height, args.duration, args.fps, 
                video_path, args.gop_size, args.input_codec, args.input_bframes
            )

        # Configure encoding parameters based on output codec
        config = {
            "preset": args.preset.upper(),
            "width": args.width,
            "height": args.height,
            "codec": args.codec.lower(),  # This is the output codec
            "tuning_info": "high_quality",
            "bf": str(args.bframes),  # Use the bframes parameter from args
            "gop": str(args.gop_size)
        }
        log_data["args"]["config"] = config

        # Set random seed if provided for reproducible segment generation
        if args.random_seed is not None:
            random.seed(args.random_seed)
            log_step("initialization", f"Set random seed to {args.random_seed}")

        # Initialize CUDA
        cuda.init()
        log_step("initialization", "Initialized CUDA")

        # Create random segments
        print("\n" + "=" * 80)
        print("STEP 2: CREATING RANDOM SEGMENTS")
        print("=" * 80)

        clip_start_end = generate_random_segments(args.duration, args.segments, min_segment_duration=args.segment_duration)

        # Log segment creation
        segment_data = [{"start": start, "end": end} for start, end in clip_start_end]
        log_step("segment_creation", f"Created {len(clip_start_end)} random segments",
                params={"num_segments": args.segments, "min_duration": 1.0, "duration": args.duration},
                results={"segments": segment_data})

        # Print segments
        print(f"Created {len(clip_start_end)} random segments:")
        for i, (start, end) in enumerate(clip_start_end):
            print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")

        # Calculate total frames to transcode
        segment_frames = sum((end - start) * args.fps for start, end in clip_start_end)

        # Store results for comparison
        results = {}

        # Run transcoding for each usage mode
        for usage in args.usage:
            if usage not in [0, 1, 2, 3]:
                print(f"Warning: Invalid usage mode {usage}. Skipping...")
                log_step("warning", f"Invalid usage mode {usage}, skipping")
                continue

            print("\n" + "=" * 80)
            print(f"STEP 3: TRANSCODING SEGMENTS USING MODE {usage}")
            print("=" * 80)

            # Get mode description
            mode_desc = {
                0: "PyNVC transcoding",
                1: "FFmpeg without map",
                2: "FFmpeg with map (no audio)",
                3: "FFmpeg with map (with audio)"
            }.get(usage, "Unknown mode")

            print(f"Running: {mode_desc}")

            # Log transcoding start
            log_step("transcoding", f"Starting transcoding with mode {usage} ({mode_desc})",
                    params={"usage": usage, "mode": mode_desc, "numthreads": args.numthreads,
                            "chunk_size": args.chunk_size, "video_path": video_path})
            clip_chunks = split_list(args.numthreads, clip_start_end)
            threads = []


            starttime = time.time()

            for m in range(0, args.numthreads):
                t = threading.Thread(target=segmented_transcode, args=([video_path,
                    clip_chunks[m],
                    config,
                    args.fps,
                    usage,
                    args.chunk_size],))
                t.start()
                #
                threads.append(t)

            for t in threads:
                t.join()

            endtime = time.time()
            total_time = endtime - starttime
            throughput = segment_frames / total_time if total_time > 0 else 0

            # Store results
            results[usage] = {
                'mode': mode_desc,
                'time': total_time,
                'throughput': throughput
            }

            # Log transcoding completion
            log_step("transcoding_complete", f"Completed transcoding with mode {usage}",
                    params={"usage": usage, "mode": mode_desc},
                    results={"time": total_time, "throughput": throughput})

            print(f"Completed {mode_desc}")
            print(f"Processing time: {total_time:.2f} seconds")
            print(f"Throughput: {throughput:.2f} frames/second")

        # Store results in log data
        log_data["results"] = {str(usage): {"time": data["time"], "throughput": data["throughput"]}
                            for usage, data in results.items()}

        # Print comparison results with additional details
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        # Add video details above the table
        print(f"Video Resolution: {args.width}x{args.height}")
        print(f"Codec: {args.codec}")
        print(f"Duration: {args.duration} seconds")
        print(f"Number of Segments: {len(clip_start_end)}")
        print(f"GOP Size: {args.gop_size}")
        print()

        print(f"{'Mode':<30} {'Time (s)':<15} {'Throughput (fps)':<20}")
        print("-" * 65)

        for usage, data in sorted(results.items()):
            print(f"{data['mode']:<30} {data['time']:<15.2f} {data['throughput']:<20.2f}")

        # Find fastest method
        if results:
            fastest = min(results.items(), key=lambda x: x[1]['time'])
            highest_throughput = max(results.items(), key=lambda x: x[1]['throughput'])

            print("\nResults Summary:")
            print(f"Total frames transcoded: {segment_frames:.0f}")
            print(f"Fastest method: {results[fastest[0]]['mode']} ({fastest[1]['time']:.2f}s)")
            print(f"Highest throughput: {results[highest_throughput[0]]['mode']} ({highest_throughput[1]['throughput']:.2f} fps)")

            # Log summary with additional details
            log_step("summary", "Transcoding summary",
                    results={
                        "resolution": f"{args.width}x{args.height}",
                        "codec": args.codec,
                        "duration": args.duration,
                        "num_segments": len(clip_start_end),
                        "gop_size": args.gop_size,
                        "total_frames": segment_frames,
                        "fastest_method": results[fastest[0]]['mode'],
                        "fastest_time": fastest[1]['time'],
                        "highest_throughput_method": results[highest_throughput[0]]['mode'],
                        "highest_throughput": highest_throughput[1]['throughput']
                    })

        print("=" * 80)

        # Close log file
        if log_file:
            log_file.close()
            print(f"Log saved to: {args.log}")