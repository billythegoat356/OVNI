# Optimized Video Native Interface 🛸
I was getting frustrated at FFmpeg being so slow/buggy on specific tasks, and I won't even talk about MoviePy (some people still use it apparently, and wonder why it takes 1 hour to process a 5 minutes video...)   
So I decided to make my custom video processing pipeline taking full advantage of my GPU.   

Demuxing/decoding happens on the GPU thanks to NVC, frames are kept in GPU memory as CuPy arrays, and frame operations are done on the GPU via CUDA kernels.   
The frames can then be encoded while still on the GPU with NVC, and a FFmpeg pipe is used for muxing.   

The whole process is written in Python and follows a generator-based pipeline allowing it to remain suitable for real-time tasks while also avoiding filling up the GPU memory.   

I coded this in 2 days so there is definitely room for improvement, speed wise and code wise, and some features may be missing or incomplete.   

# Requirements
- A Nvidia graphics card
- CUDA driver & toolkit
- Python > 3.11
- requirements.txt (you can either build CuPy yourself or install a prebuilt version, but check your CUDA driver version for this)
- Edit the Makefile to contain your GPU Cuda architecture, check [this](https://developer.nvidia.com/cuda-gpus)
- libass.c (only if you plan on using Advanced SubStation Alpha captions)
    
# How to use? 
Check the [documentation](docs.md). Most of the animations you can think of can be done with those basics methods.   

# Important
In long-running single process applications there is a memory leak in the NVC SDK. Check [this issue](https://github.com/billythegoat356/OVNI/issues/1) for a patch.    
If you still experience memory leaks either run the generation in a different process which ensures memory clean-up, or make an issue with the details.

# Todo
- Rewrite overlay/blend to take in X, Y, and overlay top left coords to bottom right. Allowing for general interpolation, meaning smoother specific animations.   
- Make a custom object for frames, with attribute methods that operate directly on it and better typing support.

# Benchmarks
On my RTX 4080 Super, I achieve a speed of around **720 frames per second** without any operations on the frames, when decoding and encoding directly *with preset 3*.   
With preset 1, which gives lower quality, I achieve around **1060 frames per second**, and **360** with preset 7.   
Note that if written correctly, the operations on frames should be nearly instantaneous, and the bottleneck should be in decoding.   

# Collaborate
Feel free to make a pull request with new CUDA kernels that you feel are missing.
