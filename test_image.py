from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import *


def test_image():

    import time

    start = time.time()

    frame = load_image("videos/image.png", gpu=True)

    print(frame.shape)


    frame = resize(frame, 1920, 1080)
    ov = crop(frame.copy(), 0, 1800, 300, 500)

    # h, w, _ = ov.shape

    # # Step 1: Create alpha values from 0 to 255 (as uint8)
    # alpha_values = cp.linspace(0, 255, h, dtype=cp.uint8).reshape(h, 1)

    # # Step 2: Broadcast to (h, w)
    # alpha_channel = cp.broadcast_to(alpha_values, (h, w))

    # # Step 3: Add a new axis to make it (h, w, 1)
    # alpha_channel = alpha_channel[:, :, cp.newaxis]

    # # Step 4: Concatenate to form (h, w, 4)
    # ov = cp.concatenate((ov, alpha_channel), axis=2)


    print(frame.shape)


    t = 0

    def process_frames(frame):
        nonlocal t
        for _ in range(1000):
            nframe = frame.copy()
            overlay(nframe, ov, 0, t, 0.5)
            # nframe = scale_translate(frame, 1, t, t, 1920, 1080)
            # nframe = scale_translate(frame, 1+(t/1000), 0, 0, 1920, 1080)
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