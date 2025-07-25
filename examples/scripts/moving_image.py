from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import pipe_nv12_to_rgb, pipe_rgb_to_nv12, scale_translate, resize

from _get_caller import call




IMAGE_PATH = "examples/media/image.png"
OUT_PATH = "examples/media/out.mp4"

WIDTH = 1920
HEIGHT = 1080
FPS = 25
BITRATE = '3M'

FRAMES = 300


def moving_image():
    """
    Moves an image horizontally, vertically, and zooms in
    """

    # Load image & resize to output resolution
    frame = load_image(IMAGE_PATH)
    frame = resize(frame, 1920, 1080)

    # Define frames generator method
    def frames_generator():
        START_X = -500
        END_X = 0

        START_Y = 0
        END_Y = -100

        START_ZOOM = 1.6
        END_ZOOM = 1.2

        for i in range(FRAMES):
            # Calculate values depending on frame index
            current_x = START_X + (END_X - START_X) / (FRAMES - 1) * i
            current_y = START_Y + (END_Y - START_Y) / (FRAMES - 1) * i

            # Calculating zoom requires a logarithmic formula
            current_zoom = START_ZOOM * ((END_ZOOM / START_ZOOM) ** (i / (FRAMES - 1)))

            this_frame = scale_translate(frame, current_zoom, current_x, current_y, WIDTH, HEIGHT)
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
    call(moving_image, FRAMES, FPS)