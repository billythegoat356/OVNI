from ovni.base import demux_and_decode, encode, mux
from ovni.kernels import pipe_nv12_to_rgb, pipe_rgb_to_nv12


def test():

    import time

    start = time.time()

    frames = demux_and_decode(
        input_path="videos/video2.mp4",
        frame_count=None
    )


    t = 0

    def process_frames(frames):
        nonlocal t
        for frame in frames:

            yield frame
            t += 1


    frames = pipe_nv12_to_rgb(frames, 1920, 1080)

    frames = process_frames(frames)

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
    test()