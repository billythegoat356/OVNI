import subprocess
import json
from fractions import Fraction


def get_video_resolution(path: str) -> tuple[int, int]:
    """
    Uses ffprobe to get the resolution of the video at the given path

    Parameters:
        path: str

    Returns:
        tuple[int, int] - width and height
    """

    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height",
        "-of", "json",
        path
    ]

    result = subprocess.run(
        ffprobe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    data = json.loads(result.stdout)['streams'][0]

    width = data['width']
    height = data['height']

    return width, height


def get_video_framerate(path: str) -> float:
    """
    Uses ffprobe to get the framerate of the video at the given path

    Parameters:
        path: str

    Returns:
        float - the video framerate
    """

    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of", "json",
        path
    ]

    result = subprocess.run(
        ffprobe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    data = json.loads(result.stdout)['streams'][0]

    # Convert to float from potential fraction
    framerate = float(Fraction(data["avg_frame_rate"]))

    return framerate


def get_video_frame_count(path: str) -> int:
    """
    Uses ffprobe to get the frame count of the video at the given path

    Parameters:
        path: str

    Returns:
        int - the number of frames of the video
    """

    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of", "json",
        path
    ]

    result = subprocess.run(
        ffprobe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    data = json.loads(result.stdout)['streams'][0]

    num_frames = int(data["nb_frames"])

    return num_frames


def get_video_duration(path: str) -> float:
    """
    Uses ffprobe to get the duration of the video at the given path
    NOTE: Returns the duration in seconds

    Parameters:
        path: str

    Returns:
        float - the duration of the video, in seconds
    """

    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "format=duration",
        "-of", "json",
        path
    ]

    result = subprocess.run(
        ffprobe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    data = json.loads(result.stdout)['format']

    duration = float(data["duration"])

    return duration