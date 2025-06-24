# Optimized Video Native Interface 
I was getting frustrated at FFmpeg being so slow/buggy on specific tasks, and I won't even talk about MoviePy (some people still use it apparently...)   
So I decided to make my custom video processing pipeline taking full advantage of my GPU.   

Demuxing/decoding happens on the GPU thanks to NVC (PyNvVideoCodec), frames are kept in GPU memory as CuPy arrays, and can then be encoded and muxed while still on the GPU with NVC and a FFmpeg pipe.   
The whole process is written in Python and follows a pipeline (generator) allowing to keep it simple and suitable for real-time tasks aswell as avoiding filling up the memory.   

I coded this in a few hours so there is definitely room for improvement, speed wise and code wise.   

# Requirements
- A Nvidia graphics card
- CUDA driver & toolkit
- requirements.txt (you can either build cupy yourself or install a prebuilt version, but check your CUDA driver version for this)
- Edit the Makefile to contain your GPU Cuda architecture, check [this](https://developer.nvidia.com/cuda-gpus)
- libass.c (only if you plan on using Advanced SubStation Alpha captions)
    
# How to use? 
Check the [documentation](docs.md)

# Benchmarks
On my RTX 4080 Super, I achieve a speed of around **600 frames per second** without any operations on the frames, when decoding and encoding directly.   
Note that if written correctly, the operations should be nearly instantaneous.   

# Collaborate
Feel free to make a pull request with new CUDA kernels, since a lot of basic operations are not supported right now.
