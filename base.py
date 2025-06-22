from typing import Generator
import subprocess

import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import cupy as cp

import pycuda.autoinit # Required






def decode(input_path: str) -> Generator[cp.ndarray, None, None]:
    """
    Decodes raw frames from the input path using the GPU, and returns them in a generator of CuPy arrays

    Parameters:
        input_path: str

    Returns:
        Generator[cp.ndarray, None, None]
    """

    frame_count = None  # Reset to None to decode all frames

    gpu_id = 0
    cuda_ctx = None
    decode_latency = nvc.DisplayDecodeLatencyType.LOW
    try:
        device_id = gpu_id
        cuda_device = cuda.Device(device_id)  # pyright: ignore[reportAttributeAccessIssue]
        cuda_ctx = cuda_device.retain_primary_context()
        cuda_ctx.push()
        cuda_stream = cuda.Stream()
        nv_dmx = nvc.CreateDemuxer(filename=input_path)

        width = nv_dmx.Width()
        height = nv_dmx.Height()
        fps = nv_dmx.FrameRate()


        print(width, height, fps)

        caps = nvc.GetDecoderCaps(
            gpuid=gpu_id,
            codec=nv_dmx.GetNvCodecId(),
            chromaformat=nv_dmx.ChromaFormat(),
            bitdepth=nv_dmx.BitDepth()
        )
        if "num_decoder_engines" in caps:
            print("Number of NVDECs:", caps["num_decoder_engines"])

        nv_dec = nvc.CreateDecoder(
            gpuid=gpu_id,
            codec=nv_dmx.GetNvCodecId(),
            cudacontext=cuda_ctx.handle,
            cudastream=cuda_stream.handle,
            usedevicememory=True,
            enableasyncallocations=True,
            maxwidth=0,
            maxheight=0,
            outputColorType=nvc.OutputColorType.NATIVE,
            enableSEIMessage=False,
            latency=decode_latency
        )



        decoded_frame_size = 0
        raw_frame = None

        seq_triggered = False
        # printing out FPS and pixel format of the stream for convenience
        print("FPS =", nv_dmx.FrameRate())
        # open the file to be decoded in write mode

        # demuxer can be iterated, fetch the packet from demuxer
        frames_decoded = 0
        for packet in nv_dmx:
            # Decode returns a list of packets, range of this list is from [0, size of (decode picture buffer)]
            # size of (decode picture buffer) depends on GPU, fur Turing series its 8
            if decode_latency == nvc.DisplayDecodeLatencyType.LOW or decode_latency == nvc.DisplayDecodeLatencyType.ZERO:
                # Set when the packet contains exactly one frame or one field bitstream data, parser will trigger decode callback immediately when this flag is set.
                packet.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE

            for decoded_frame in nv_dec.Decode(packet):

                # 'decoded_frame' contains list of views implementing cuda array interface
                # for nv12, it would contain 2 views for each plane and two planes would be contiguous 
                if not seq_triggered:
                    decoded_frame_size = nv_dec.GetFrameSize()
                    seq_triggered = True

                luma_base_addr = decoded_frame.GetPtrToPlane(0)
                frame_size = decoded_frame.framesize()

                # Wrap the raw GPU pointer into a CuPy array without copying
                # UnownedMemory lets CuPy use external GPU memory safely without managing its lifecycle
                mem = cp.cuda.UnownedMemory(luma_base_addr, frame_size, None)
                memptr = cp.cuda.MemoryPointer(mem, 0)
                gpu_frame = cp.ndarray(shape=decoded_frame_size, dtype=cp.uint8, memptr=memptr)

                yield gpu_frame

                
                frames_decoded += 1
                if frame_count is not None and frame_count > 0 and frames_decoded >= frame_count:
                    print(f"Reached requested frame count: {frame_count}")
                    return
            
            if frame_count is not None and frame_count > 0 and frames_decoded < frame_count:
                print(f"Video ended before reaching requested frame count. Decoded {frames_decoded} frames")
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateDecoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if cuda_ctx is not None:
            cuda_ctx.pop()





def encode(frames: Generator[cp.ndarray, None, None], width: int, height: int, fps: int) -> Generator[bytes, None, None]:
    """
    Takes a generator of CuPy arrays, and encodes them into a H264 bytestream

    Parameters:
        frames: Generator[cp.ndarray, None, None]
        width: int
        height: int
        fps: int
        output_path: str

    Returns:
        None
    """

    gpu_id = 0
    codec = 'h264'
    bitrate = 5000000

    cuda_ctx = None
    try:
        cuda_device = cuda.Device(gpu_id)
        cuda_ctx = cuda_device.retain_primary_context()
        cuda_ctx.push()

        cuda_stream = cuda.Stream()

        config_params = {
            'gpuid': gpu_id,
            'codec': codec,
            'preset': 'P3',
            'tuning_info': 'low_latency',
            'rc': 'cbr',
            'fps': fps,
            'gop': fps,
            'bf': 0,
            'bitrate': f"{bitrate//1000000}M",
            'maxbitrate': 0,
            'vbvinit': 0,
            'vbvbufsize': 0,
            'qmin': "0,0,0",
            'qmax': "0,0,0", 
            'initqp': "0,0,0",
            'cudacontext': cuda_ctx.handle,
            'cudastream': cuda_stream.handle,
            'enable_async': True,
            'bRepeatSPSPPS': True
        }

        nv_enc = nvc.CreateEncoder(
            width,
            height,
            'NV12',
            False,
            **config_params
        )

        for frame in frames:
            frame_2d = frame.reshape((height * 3 // 2, width))

            class FrameWrapper:
                def __init__(self, arr):
                    self.arr = arr
                def cuda(self):
                    return self
                def __getattr__(self, name):
                    return getattr(self.arr, name)
                def __dlpack__(self, *args, **kwargs):
                    stream = None
                    if args:
                        stream = args[0]
                    return self.arr.__dlpack__(stream=stream, **kwargs)

            frame_2d = FrameWrapper(frame_2d)
            
            encoded_bytes = nv_enc.Encode(frame_2d)
            if encoded_bytes and len(encoded_bytes) > 0:
                yield encoded_bytes


        # Flush remaining packets
        encoded_bytes = nv_enc.EndEncode()
        if encoded_bytes and len(encoded_bytes) > 0:
            yield encoded_bytes

        

    except Exception as e:
        print(f"Encoding failed: {e}")
    finally:
        if cuda_ctx:
            cuda_ctx.pop()



def mux(h264_stream: Generator[bytes, None, None], output_path: str) -> None:
    """
    
    """

    # Start FFmpeg process with pipe input
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'h264',           # Input format
        '-i', 'pipe:0',         # Read from stdin
        '-c', 'copy',           # Copy without re-encoding
        '-y',                   # Overwrite output
        output_path
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pipe = ffmpeg_process.stdin

    for encoded_bytes in h264_stream:
        pipe.write(encoded_bytes)
        pipe.flush()  # Ensure data is sent immediately

    # Close stdin to signal end of input
    ffmpeg_process.stdin.close()

    # Wait for FFmpeg to finish
    ffmpeg_process.wait()



import time

start = time.time()



frames = decode(
    input_path="videos/video2.mp4",
)

frames = (
    frame
    for frame in frames
)

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

