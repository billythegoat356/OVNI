from ovni.base import demux_and_decode, encode, mux, load_image
from ovni.ops import *

from ovni.ass import ASSRenderer, LibASS


def test_image():

    import time

    start = time.time()

    frame = load_image("videos/image.png", gpu=True)
    

    print(frame.shape)


    frame = resize(frame, 1920, 1080)

    frame = cp.empty((1080, 1920, 3), cp.uint8)
    frame[:, :, :] = cp.array([255, 0, 0], dtype=cp.uint8)


    print(frame.shape)


    t = 0

    def process_frames(frame):
        nonlocal t
        LibASS.load()
        
        # print("Storing overlay fully in memory once")
        # overlay = demux_and_decode("videos/ov1.mp4")
        # overlay = pipe_nv12_to_rgb(overlay, 1920, 1080)
        # overlay = (chroma_key(o, (0, 0, 0), 0, 255) for o in overlay)
        # overlay = list(overlay)


        with ASSRenderer("videos/captions2.ass", 1920, 1080) as r:
            
            for _ in range(5):

                print("Reloading overlay each time")
                overlay = demux_and_decode("videos/ov1.mp4")
                overlay = pipe_nv12_to_rgb(overlay, 1920, 1080)
                
                for oframe in overlay:
                    nframe = frame.copy()

                    oframe = chroma_key(oframe, (0, 0, 0), 150, 255)
                    blend(nframe, oframe, 0, 0)

                    # r.render_frame(int(t/25*1000), background_frame=nframe)
                    yield nframe
                    t += 1




    frames = process_frames(frame)
    frames = [next(frames)]

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