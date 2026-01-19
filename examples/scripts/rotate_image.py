from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import pipe_nv12_to_rgb, pipe_rgb_to_nv12, rotate, resize

from _get_caller import call




IMAGE_PATH = "examples/media/image.png"
OUT_PATH = "examples/media/out.mp4"

WIDTH = 1920
HEIGHT = 1080
FPS = 25
BITRATE = '3M'

FRAMES = 300


def rotate_image():
    """
    Rotates an image
    """

    # Load image & resize to output resolution
    frame = load_image(IMAGE_PATH)
    frame = resize(frame, 1920, 1080)

    # Define frames generator method
    def frames_generator():
        START_DEG = -180
        END_DEG = 180

        for i in range(FRAMES):
            
            deg = START_DEG + (i / (FRAMES - 1) * (END_DEG - START_DEG))
            this_frame = rotate(frame, deg)

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
    call(rotate_image, FRAMES, FPS)