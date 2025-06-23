from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.kernels import pipe_nv12_to_rgb, pipe_rgb_to_nv12, scale_translate


def test_image():

    import time

    start = time.time()

    frame = load_image("videos/image.png", gpu=False)

    print(frame.shape)

    import cupy as cp
    import cv2


    # Resize with OpenCV (size is (width, height))
    resized_np_img = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

    # Convert back to CuPy array (copies from CPU to GPU)
    frame = cp.asarray(resized_np_img)



    t = 0

    def process_frames(frame):
        nonlocal t
        for _ in range(1000):
            nframe = scale_translate(frame, 1.2, -1*t, -1*t, 1920, 1080)
            # nframe = scale_translate(frame, 1+(t/1000), 0, 0, 1920, 1080)
            yield nframe
            t += 1



    frames = process_frames(frame)

    frames = pipe_rgb_to_nv12(frames, 1920, 1080)

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