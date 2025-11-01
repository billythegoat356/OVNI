# Optimized Video Native Interface 🛸
I was getting frustrated at FFmpeg being so slow/buggy on specific tasks, and I won't even talk about MoviePy (some people still use it apparently, and wonder why it takes 1 hour to process a 5 minutes video...)   
So I decided to make my custom video processing pipeline taking full advantage of my GPU.   

Demuxing/decoding happens on the GPU thanks to NVC, frames are kept in GPU memory as CuPy arrays, and frame operations are done on the GPU via CUDA kernels.   
The frames can then be encoded while still on the GPU with NVC, and a FFmpeg pipe is used for muxing.   

The whole process is written in Python and follows a generator-based pipeline allowing it to remain suitable for real-time tasks while also avoiding filling up the GPU memory.   

I coded this in 2 days so there is definitely room for improvement, speed wise and code wise, and some features may be missing or incomplete.   

# Requirements
- A Nvidia graphics card
- FFmpeg
- CUDA driver & toolkit
- Python > 3.11
- [optional] LibASS, only if you plan on using Advanced SubStation Alpha captions
    
# Setup guide
This guide assumes you have already installed the requirements mentioned above.   

### Compile CUDA kernels
First, check your CUDA arch [here](https://developer.nvidia.com/cuda-gpus).   
Say your arch is `compute_89`, run the following command:
```sh
make CUDA_ARCH=compute_89
```

### Install python requirements

Create a Python3.11 virtual env. (you can skip this step but it is highly recommended)
```sh
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate
```

Then install the package locally.
```sh
pip install .
```

Now we need to install cupy.   
You can compile it locally, but I recommend installing a precompiled version for your CUDA toolkit version.   

Check CUDA toolkit version
```sh
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Fri_Feb_21_20:23:50_PST_2025
# Cuda compilation tools, release 12.8, V12.8.93
# Build cuda_12.8.r12.8/compiler.35583870_0
```

Our version is `12.8`. So we install `cupy-cuda12x`.
```sh
pip install cupy-cuda12x
```

### [optional] Fix memory leak in Nvidia Video Codec SDK
*In long-running single process applications there are several memory leaks in the NVC SDK.*   
*I've patched their SDK, check [this issue](https://github.com/billythegoat356/OVNI/issues/1) for detailed information.*   
*Otherwise, follow these simple steps to recompile it locally with the fixes.*   

Compile the patched version of PyNVC
```sh
cd pynvc_patched/PyNvVideoCodec_2.0.0
mkdir build
cd build
cmake ..
make -j $(nproc)
```

Check your NVIDIA driver version to know if you need the version 13.0 or 12.1 of the SDK.   
*(other versions aren't implemented in PyNVC)*   
```sh
nvidia-smi
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 575.64.03...
```
If your driver version is >= 570, you need 13.0.   
If your driver version is >= 530 , you need 12.1.   
You also need CUDA toolkit >= 11 for both (the one previously checked when installing cupy).   

*Other versions aren't supported.*   

Now, move the compiled file to your PyNvVideoCodec installation.   
Note that the path may vary depending on your SDK version, python version and CPU architecture.
```sh
mv src/PyNvVideoCodec/PyNvVideoCodec_130.cpython-311-x86_64-linux-gnu.so ../../../.venv/lib/python3.11/site-packages/PyNvVideoCodec
```

# How to use? 
Check the [documentation](docs.md). Most of the animations you can think of can be done with those basics methods.   
You can also check out some basic [examples](examples/scripts).   

# Important
There are several memory leaks in the NVC SDK used for decoding/encoding.   
I've fixed them but memory may occasionally leak in long-running single process applications.   
So if you're planning on using OVNI in production, it is highly recommended to run the generation in a different process (e.g., using `multiprocessing`) which ensures memory clean-up.   

# Todo
- Rewrite overlay/blend to take in X, Y, and overlay top left coords to bottom right. Allowing for general interpolation, meaning smoother specific animations.   
- Make a custom object for frames, with attribute methods that operate directly on it and better typing support.
- Make an object for video timelines (sequencial operations on frames, just a algorithmic wrapper)

# Benchmarks
On my RTX 4080 Super, I achieve a speed of around **720 frames per second** without any operations on the frames, when decoding and encoding directly *with preset 3*.   
With preset 1, which gives lower quality, I achieve around **1060 frames per second**, and **360** with preset 7.   
Note that if written correctly, the operations on frames should be nearly instantaneous, and the bottleneck should be in decoding.   

# Collaborate
Feel free to make a pull request with new CUDA kernels that you feel are missing.
