from typing import Generator, Iterable
import subprocess
import atexit

import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import cupy as cp
import cv2

import pycuda.autoinit # Required


# Constants (adjust code if you need customzz)
GPU_ID = 0
CODEC = 'h264' # Encoding codec
BITRATE = '5M'
PRESET = 'P3' # 1-7, determines quality, but impacts speed


# Set Warning log level (not sure if this works)
nvc.logger.setLevel(nvc.logging.WARNING)


# Initialize cuda device and context
cuda_device = cuda.Device(GPU_ID)
cuda_ctx = cuda_device.retain_primary_context()
cuda_ctx.push()

# Close cuda context at exit
atexit.register(cuda_ctx.pop)



def demux_and_decode(input_path: str, frame_count: int | None = None) -> Generator[cp.ndarray, None, None]:
    """
    Decodes raw frames from the input path using NVC, and returns them in a generator of CuPy arrays
    -----------
    NOTE: The pixel format of the returned frames is flattened NV12

    Parameters:
        input_path: str
        frame_count: int | None = None - the number of frames to yield, None means unlimited

    Returns:
        Generator[cp.ndarray, None, None]
    """

    passed_frames = 0

    # Create cuda stream and demuxer
    cuda_stream = cuda.Stream()
    nv_dmx = nvc.CreateDemuxer(filename=input_path)

    # # These variables contain basic video information
    # # If they are ever needed, this is how to access them
    # width = nv_dmx.Width()
    # height = nv_dmx.Height()
    # fps = nv_dmx.FrameRate()

    # Create decoder
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

    # Iterate over all packets
    for packet in nv_dmx:

        # Market the packet as containing complete frames
        packet.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE

        # Decode packet and iterate over frames
        for decoded_frame in nv_dec.Decode(packet):

            if passed_frames == frame_count:
                return

            # 'decoded_frame' contains list of views implementing cuda array interface
            # For nv12, it would contain 2 views for each plane and two planes would be contiguous 

            # Get decoded frame shape only once
            if frame_shape is None:
                frame_shape = nv_dec.GetFrameSize()

            # Get address of the frame
            base_addr = decoded_frame.GetPtrToPlane(0)
            # Get frame size (bytes)
            frame_size = decoded_frame.framesize()

            # Wrap the raw GPU pointer into a CuPy array without copying
            # UnownedMemory lets CuPy use external GPU memory safely without managing its lifecycle
            mem = cp.cuda.UnownedMemory(base_addr, frame_size, None)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            gpu_frame = cp.ndarray(shape=frame_shape, dtype=cp.uint8, memptr=memptr)

            yield gpu_frame

            passed_frames += 1



def load_image(path: str, gpu: bool = True) -> cp.ndarray:
    """
    Loads an image from a path inside a RGB array

    Parameters:
        path: str
        gpu: bool = True - whether to load the array on the GPU. If you want to do some one-time processing, you may want it on the CPU.

    Returns:
        cp.ndarray
    """
    # Load image with OpenCV (BGR format)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {path}")

    # Optionally load the array on the GPU with cupy
    if gpu:
        img_bgr = cp.asarray(img_bgr)

    # Transform to RGB
    img_rgb = img_bgr[..., [2, 1, 0]]

    return img_rgb




def encode(frames: Iterable[cp.ndarray], width: int, height: int, fps: int) -> Generator[bytes, None, None]:
    """
    Takes an iterable of CuPy arrays, and encodes them into a H264 bytestream
    -----------
    NOTE: The pixel format of the frames is expected to be flattened NV12, like the one given from decoding

    Parameters:
        frames: Iterable[cp.ndarray]
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


    # Create cuda stream and encoder
    cuda_stream = cuda.Stream()
    config_params = {
        'gpuid': GPU_ID,
        'codec': CODEC,
        'preset': PRESET,
        'cudacontext': cuda_ctx.handle,
        'cudastream': cuda_stream.handle,
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

    # Iterate over frames of generator
    for frame in frames:
        # Unflatten ~ specific to NV12 pixel format
        frame_nv12 = frame.reshape((height * 3 // 2, width))

        # Wrap the frame to be compatible with the NVC method
        frame_nv12 = FrameWrapper(frame_nv12)
        
        # Encode the frame into H264 bytes
        encoded_bytes = nv_enc.Encode(frame_nv12)
        yield encoded_bytes


    # Flush remaining packets
    encoded_bytes = nv_enc.EndEncode()
    yield encoded_bytes





def mux(h264_stream: Iterable[bytes], output_path: str, audio_path: str | None = None, ffmpeg_flags: list[str] = []) -> None:
    """
    Takes an iterable of a H264 bytestream, and muxes it with a FFmpeg pipe in the output path
    
    It is possible to give an audio path which will be muxed with the video stream.
    Note that the two streams are just muxed. The video will be of the length of the longest one. Process the audio separately to avoid this.
    
    Parameters:
        h264_stream: Iterable[bytes]
        output_path: str
        audio_path: str | None = None - optional audio path.
        ffmpeg_flags: list[str] = [] - additional ffmpeg flags
        
    Returns:
        None
    """

    # Start FFmpeg process with pipe input
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', CODEC,            # Input format
        '-i', 'pipe:0'          # Read from stdin
    ]

    # Add optional audio path
    if audio_path is not None:
        ffmpeg_cmd.extend([
            '-i', audio_path,
        ])
    
    # Add rest of the command
    ffmpeg_cmd.extend([
        '-c', 'copy',          # Copy without re-encoding
        *ffmpeg_flags,
        '-y',                   # Overwrite output
        output_path
    ])

    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pipe = ffmpeg_process.stdin

    # Iterate over H264 bytestream, and write each packet
    for encoded_bytes in h264_stream:
        pipe.write(encoded_bytes)
        pipe.flush()  # Ensure data is sent immediately

    # Close stdin to signal end of input
    ffmpeg_process.stdin.close()

    # Wait for FFmpeg to finish
    ffmpeg_process.wait()












