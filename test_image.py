from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import *


def test_image():

    import time

    start = time.time()

    frame = load_image("videos/image.png", gpu=True)

    print(frame.shape)


    frame = resize(frame, 1920, 1080)


    print(frame.shape)


    t = 0

    def process_frames(frame):
        nonlocal t
        height, width, _ = frame.shape

        # Create an alpha gradient from 0 to 255 over the width
        alpha_channel = cp.linspace(0, 255, width, dtype=cp.uint8)
        alpha_channel = cp.tile(alpha_channel, (height, 1))  # Shape: (height, width)

        # Expand dims to (height, width, 1)
        alpha_channel = cp.expand_dims(alpha_channel, axis=2)

        # Combine RGB + alpha: assuming RGB channels are zeros for kframe
        kframe = cp.concatenate((frame, alpha_channel), axis=2)  # Shape: (height, width, 4)

        for _ in range(1000):
            nframe = frame.copy()

            x = 50 + (t / 3000 * 1920)
            y = 50 + (t / 3000 * 1080)

            # x = int(x)
            # y = int(y)

            blend(nframe, kframe, x, y)
            yield nframe
            t += 1



    frames = process_frames(frame)

    frames = pipe_rgb_to_nv12(frames)

    h264_stream = encode(
        frames=frames,
        width=1920,
        height=1080,
        fps=25
    )

    mux(
        h264_stream=h264_stream,
        output_path="videos/out.mp4"
    )


    print(f"Took {round(time.time() - start, 2)}s")
    print(f"Frames: {t}")


    d = t/25
    s = round(d/(time.time() - start), 2)
    print(f"Speed: x{s}")


if __name__ == '__main__':
    test_image()