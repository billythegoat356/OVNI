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
        for _ in range(100):
            nframe = frame.copy()

            x = 50+(t/3000*1920)
            y = 50+(t/3000*1080)

            x = int(x)
            y = int(y)

            overlay(nframe, frame, x,y)
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