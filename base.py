from typing import Generator
import subprocess
import atexit

import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import cupy as cp

import pycuda.autoinit # Required





GPU_ID = 0
CODEC = 'h264' # Encoding codec
BITRATE = '5M'

cuda_device = cuda.Device(GPU_ID)
cuda_ctx = cuda_device.retain_primary_context()
cuda_ctx.push()

atexit.register(cuda_ctx.pop)



def demux_and_decode(input_path: str) -> Generator[cp.ndarray, None, None]:
    """
    Decodes raw frames from the input path using the GPU, and returns them in a generator of CuPy arrays
    -----------
    NOTE: The pixel format of the returned frames is flattened NV12

    Parameters:
        input_path: str

    Returns:
        Generator[cp.ndarray, None, None]
    """



    try:
        cuda_stream = cuda.Stream()
        nv_dmx = nvc.CreateDemuxer(filename=input_path)

        # # These variables contain basic video information
        # # If they are ever needed, this is how to access them
        # width = nv_dmx.Width()
        # height = nv_dmx.Height()
        # fps = nv_dmx.FrameRate()

        nv_dec = nvc.CreateDecoder(
            gpuid=GPU_ID,
            codec=nv_dmx.GetNvCodecId(), # Automatically detect video codec
            cudacontext=cuda_ctx.handle,
            cudastream=cuda_stream.handle,
            usedevicememory=True,
            enableasyncallocations=True,
            maxwidth=0,
            maxheight=0,
            outputColorType=nvc.OutputColorType.NATIVE,
            enableSEIMessage=False,
            latency=nvc.DisplayDecodeLatencyType.LOW
        )


        # Store the frame shape, and get it only once
        frame_shape = None

        for packet in nv_dmx:

            # For low latency, Set when the packet contains exactly one frame or one field bitstream data, parser will trigger decode callback immediately when this flag is set.
            packet.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE

            for decoded_frame in nv_dec.Decode(packet):

                # 'decoded_frame' contains list of views implementing cuda array interface
                # for nv12, it would contain 2 views for each plane and two planes would be contiguous 

                # Get decoded frame shape only once
                if frame_shape is None:
                    frame_shape = nv_dec.GetFrameSize()

                luma_base_addr = decoded_frame.GetPtrToPlane(0)
                frame_size = decoded_frame.framesize()

                # Wrap the raw GPU pointer into a CuPy array without copying
                # UnownedMemory lets CuPy use external GPU memory safely without managing its lifecycle
                mem = cp.cuda.UnownedMemory(luma_base_addr, frame_size, None)
                memptr = cp.cuda.MemoryPointer(mem, 0)
                gpu_frame = cp.ndarray(shape=frame_shape, dtype=cp.uint8, memptr=memptr)

                yield gpu_frame
                
    except nvc.PyNvVCExceptionUnsupported as e:
        print(f"CreateDecoder failure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")







def encode(frames: Generator[cp.ndarray, None, None], width: int, height: int, fps: int) -> Generator[bytes, None, None]:
    """
    Takes a generator of CuPy arrays, and encodes them into a H264 bytestream
    -----------
    NOTE: The pixel format of the frames is expected to be flattened NV12, like the one given from decoding

    Parameters:
        frames: Generator[cp.ndarray, None, None]
        width: int
        height: int
        fps: int

    Returns:
        Generator[bytes, None, None]
    """


    # The following class is required because NVC tries to access the `cuda` attribute to determine if the given array is,
    #  on the GPU, and it calls __dlpack__ with `stream` as a positional argument, but it has to be non-positional
    class FrameWrapper:
        def __init__(self, arr):
            self.arr = arr

        def __getattr__(self, name):
            return getattr(self.arr, name)

        def cuda(self):
            return self
        
        def __dlpack__(self, *args, **kwargs):
            stream = None
            if args:
                stream = args[0]
            return self.arr.__dlpack__(stream=stream, **kwargs)


    try:
        cuda_stream = cuda.Stream()
        config_params = {
            'gpuid': GPU_ID,
            'codec': CODEC,
            'preset': 'P3',
            'tuning_info': 'low_latency',
            'rc': 'cbr',
            'fps': fps,
            'gop': fps,
            'bf': 0,
            'bitrate': BITRATE,
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
            # Specific to NV12 pixel format
            frame_nv12 = frame.reshape((height * 3 // 2, width))

            # Wrap the frame to be compatible with the NVC method
            frame_nv12 = FrameWrapper(frame_nv12)
            
            # Encode the frame into H264 bytes
            encoded_bytes = nv_enc.Encode(frame_nv12)
            
            if encoded_bytes and len(encoded_bytes) > 0:
                yield encoded_bytes


        # Flush remaining packets
        encoded_bytes = nv_enc.EndEncode()
        if encoded_bytes and len(encoded_bytes) > 0:
            yield encoded_bytes

        

    except Exception as e:
        print(f"Encoding failed: {e}")




def mux(h264_stream: Generator[bytes, None, None], output_path: str) -> None:
    """
    Takes a generator of a H264 bytestream, and muxes it with a FFmpeg pipe in the output path

    Parameters:
        h264_stream: Generator[bytes, None, None]
        output_path: str
        
    Returns:
        None
    """

    # Start FFmpeg process with pipe input
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', CODEC,            # Input format
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



frames = demux_and_decode(
    input_path="videos/video2.mp4",
)


t = 0

def count():
    global t 
    t += 1

frames = (
    (frame, count())[0]
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
print(f"Frames: {t}")


d = t/25
s = round(d/(time.time() - start), 2)
print(f"Speed: x{s}")

