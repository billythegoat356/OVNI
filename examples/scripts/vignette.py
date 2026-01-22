from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import pipe_nv12_to_rgb, pipe_rgb_to_nv12, vignette
import cupy as cp
from _get_caller import call





OUT_PATH = "examples/media/out.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 25
BITRATE = '3M'

FRAMES = 300


def vignette_c():
    """
    V
    """

    # White bg frame
    bg_frame = cp.zeros((HEIGHT, WIDTH, 3), dtype=cp.uint8)
    bg_frame[:, :, :] = 255

    # Define frames generator method
    def frames_generator():

        for i in range(FRAMES):
            this_bg_frame = bg_frame.copy()

            vignette(this_bg_frame, strength=1, radius=0.8, softness=1)
            yield this_bg_frame

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
    call(vignette_c, FRAMES, FPS)