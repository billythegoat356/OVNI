from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import *

from ovni.ass import ASSRenderer, LibASS


def test_image():

    import time

    start = time.time()

    frame = load_image("videos/image.png", gpu=True)
    

    frame = resize(frame, 1920, 1080)


    t = 0

    def process_frames(frame):
        nonlocal t
        # LibASS.load()
        

        
        # print("Storing overlay fully in memory once")
        # overlay = demux_and_decode("videos/ov1.mp4")
        # overlay = pipe_nv12_to_rgb(overlay, 1920, 1080)
        # overlay = list(overlay)


        with ASSRenderer("videos/captions2.ass", 1920, 1080) as r:
            
            frames_n = 4000
            for _ in range(frames_n):

                nframe = frame.copy()
                dframe = frame.copy()

                pos = t/2
                if pos < 1080:
                    overlay(nframe, dframe, x=pos, y=pos)

                yield nframe
                t += 1




    frames = process_frames(frame)
    # frames = [next(frames)]

    frames = pipe_rgb_to_nv12(frames)
    h264_stream = encode(
        frames=frames,
        width=1920,
        height=1080,
        fps=25
    )

    mux(
        h264_stream=h264_stream,
        output_path="videos/out.mp4",
        # audio_path='videos/audio.mp3'
    )


    print(f"Took {round(time.time() - start, 2)}s")
    print(f"Frames: {t}")


    d = t/25
    s = round(d/(time.time() - start), 2)
    print(f"Speed: x{s}")


if __name__ == '__main__':
    test_image()