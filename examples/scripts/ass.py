from ovni.base import encode, mux
from ovni.ops import pipe_rgb_to_nv12
from ovni.ass import ASSRenderer
import cupy as cp
from _get_caller import call


ASS_PATH = "examples/media/subtitles.ass"
OUT_PATH = "examples/media/out.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
BITRATE = '3M'

FRAMES = 4 * FPS  # 4 seconds


def ass():
    """
    Renders subtitles on a black background
    """

    # Yellow gold background frame
    bg_frame = cp.zeros((HEIGHT, WIDTH, 3), dtype=cp.uint8)
    bg_frame[:, :, 0] = 255  # R
    bg_frame[:, :, 1] = 215  # G
    bg_frame[:, :, 2] = 0    # B

    # Define frames generator method
    def frames_generator():
        with ASSRenderer(ASS_PATH, WIDTH, HEIGHT) as renderer:
            for i in range(FRAMES):
                this_frame = bg_frame.copy()
                timestamp_ms = int(i / FPS * 1000)
                renderer.render_frame(timestamp_ms, this_frame)
                yield this_frame

    # Create pipe
    frames = frames_generator()

    # Pipe to NV12 pixel format
    frames = pipe_rgb_to_nv12(frames)

    # Pipe to H264 stream
    h264_stream = encode(
        frames=frames,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        bitrate=BITRATE
    )

    # Pipe to output file
    mux(
        h264_stream=h264_stream,
        output_path=OUT_PATH
    )


if __name__ == "__main__":
    call(ass, FRAMES, FPS)
