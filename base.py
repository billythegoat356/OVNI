from typing import Generator, Iterable
import subprocess
import atexit

import pycuda.driver as cuda
import PyNvVideoCodec as nvc
import cupy as cp

import pycuda.autoinit # Required




# Constants
GPU_ID = 0
CODEC = 'h264' # Encoding codec
BITRATE = '5M'
PRESET = 'P3' # 1-7, determines quality, but impacts speed


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





def mux(h264_stream: Iterable[bytes], output_path: str) -> None:
    """
    Takes an iterable of a H264 bytestream, and muxes it with a FFmpeg pipe in the output path

    Parameters:
        h264_stream: Iterable[bytes]
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

    # Iterate over H264 bytestream, and write each packet
    for encoded_bytes in h264_stream:
        pipe.write(encoded_bytes)
        pipe.flush()  # Ensure data is sent immediately

    # Close stdin to signal end of input
    ffmpeg_process.stdin.close()

    # Wait for FFmpeg to finish
    ffmpeg_process.wait()









nv12_to_rgb_kernel = cp.RawKernel(r'''
extern "C" __global__
void nv12_to_rgb(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    unsigned char* rgb,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y_idx = y * width + x;
    int uv_idx = (y / 2) * width + (x / 2) * 2;

    float Y = (float)y_plane[y_idx];
    float U = (float)uv_plane[uv_idx]     - 128.0f;
    float V = (float)uv_plane[uv_idx + 1] - 128.0f;

    // BT.709 full range (scale coefficients)
    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    R = min(max(R, 0.0f), 255.0f);
    G = min(max(G, 0.0f), 255.0f);
    B = min(max(B, 0.0f), 255.0f);

    int rgb_idx = (y * width + x) * 3;
    rgb[rgb_idx + 0] = (unsigned char)R;
    rgb[rgb_idx + 1] = (unsigned char)G;
    rgb[rgb_idx + 2] = (unsigned char)B;
}
''', 'nv12_to_rgb')


def nv12_to_rgb(nv12_frame: cp.ndarray, width: int, height: int) -> cp.ndarray:
    y_plane = nv12_frame[:width * height].reshape((height, width))
    uv_plane = nv12_frame[width * height:].reshape((height // 2, width))

    rgb = cp.empty((height, width, 3), dtype=cp.uint8)

    threads = (16, 16)
    blocks = ((width + threads[0] - 1) // threads[0],
              (height + threads[1] - 1) // threads[1])

    nv12_to_rgb_kernel(
        blocks, threads,
        (
            y_plane.ravel(),
            uv_plane.ravel(),
            rgb.ravel(),
            cp.int32(width),
            cp.int32(height)
        )
    )
    return rgb


rgb_to_nv12_kernel = cp.RawKernel(r'''
extern "C" __global__
void rgb_to_nv12(
    const unsigned char* rgb,
    unsigned char* y_plane,
    unsigned char* uv_plane,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float R = (float)rgb[idx + 0];
    float G = (float)rgb[idx + 1];
    float B = (float)rgb[idx + 2];

    // BT.709 full range
    float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    float U = (B - Y) / 1.8556f + 128.0f;
    float V = (R - Y) / 1.5748f + 128.0f;

    Y = min(max(Y, 0.0f), 255.0f);
    U = min(max(U, 0.0f), 255.0f);
    V = min(max(V, 0.0f), 255.0f);

    y_plane[y * width + x] = (unsigned char)(Y);

    if ((x % 2 == 0) && (y % 2 == 0)) {
        int uv_idx = (y / 2) * width + x;
        uv_plane[uv_idx]     = (unsigned char)(U);
        uv_plane[uv_idx + 1] = (unsigned char)(V);
    }
}
''', 'rgb_to_nv12')


def rgb_to_nv12(rgb: cp.ndarray, width: int, height: int) -> cp.ndarray:
    y_plane = cp.empty((height, width), dtype=cp.uint8)
    uv_plane = cp.zeros((height // 2, width), dtype=cp.uint8)

    threads = (16, 16)
    blocks = ((width + threads[0] - 1) // threads[0],
              (height + threads[1] - 1) // threads[1])

    rgb_to_nv12_kernel(
        blocks, threads,
        (
            rgb.ravel(),
            y_plane.ravel(),
            uv_plane.ravel(),
            cp.int32(width),
            cp.int32(height)
        )
    )

    return cp.concatenate((y_plane.ravel(), uv_plane.ravel()))


def pipe_nv12_to_rgb(frames: Iterable[cp.ndarray], width: int, height: int) -> Generator[cp.ndarray, None, None]:
    for nv12_frame in frames:
        rgb_frame = nv12_to_rgb(nv12_frame, width, height)
        yield rgb_frame

def pipe_rgb_to_nv12(frames: Iterable[cp.ndarray], width: int, height: int) -> Generator[cp.ndarray, None, None]:
    for rgb_frame in frames:
        nv12_frame = rgb_to_nv12(rgb_frame, width, height)
        yield nv12_frame



def test():

    import time

    start = time.time()

    frames = demux_and_decode(
        input_path="videos/video2.mp4",
        frame_count=500
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