from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import *

from ovni.ass import ASSRenderer, LibASS


def test_image():

    import time

    start = time.time()

    frame = load_image("videos/image.png", gpu=True)
    




    t = 0

    def process_frames(frame):
        nonlocal t
        # LibASS.load()
        
        

        def go():
            print("Reloading overlay each time")
            while True:
                overlay = demux_and_decode("videos/ov1.mp4")
                overlay = pipe_nv12_to_rgb(overlay, 1920, 1080)
                yield from overlay
        overlay = go()
        
        # print("Storing overlay fully in memory once")
        # overlay = demux_and_decode("videos/ov1.mp4")
        # overlay = pipe_nv12_to_rgb(overlay, 1920, 1080)
        # overlay = list(overlay)


        def get_ov():
            i = 0
            if isinstance(overlay, list):
                while True:
                    yield overlay[i%len(overlay)]
                    i += 1
            else:
                yield from overlay

        o = get_ov()

        with ASSRenderer("videos/captions2.ass", 1920, 1080) as r:
            
            frames_n = 4000
            for _ in range(frames_n):
                oframe = next(o)

                nframe = frame.copy()

                s = 1.3 + t/frames_n

                tx = t/100 * 1920 / 3000
                ty = t/100 * 1080 / 3000

                # nframe = scale_translate(nframe, s, tx, ty, 1920, 1080)
                nframe = resize(nframe, 1920, 1080)

                oframe = chroma_key(oframe, (0, 0, 0), 150, 255)
                blend(nframe, oframe, t, t)

                r.render_frame(int(t/25*1000), background_frame=nframe)
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