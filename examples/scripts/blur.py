from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import pipe_nv12_to_rgb, pipe_rgb_to_nv12, scale_translate, resize, overlay, round_corners, gaussian_blur
import cupy as cp
from _get_caller import call




IMAGE_PATH = "examples/media/image.png"
OUT_PATH = "examples/media/out.mp4"

WIDTH = 960
HEIGHT = 540
FPS = 25
BITRATE = '3M'

FRAMES = 300


def blur():
    """
    bluR
    """

    # White bg frame
    bg_frame = cp.zeros((HEIGHT, WIDTH, 3), dtype=cp.uint8)
    bg_frame[:, :, :] = 255

    # Load image & resize to output resolution
    frame = load_image(IMAGE_PATH)
    frame = resize(frame, 700, 400)

    # Define frames generator method
    def frames_generator():
        X = 100
        Y = 50

        for i in range(FRAMES):
            this_bg_frame = bg_frame.copy()

            nframe = gaussian_blur(frame, i+1)
            overlay(this_bg_frame, nframe, X, Y)
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
    call(blur, FRAMES, FPS)