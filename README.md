# Optimized Video Native Interface 
I was getting frustrated at FFmpeg being so slow/buggy on specific tasks, and I won't even talk about PyMovie (some people still use it apparently...)   
So I decided to make my custom video processing pipeline taking full advantage of my GPU.   

Demuxing/decoding happens on the GPU thanks to NVC (PyNvVideoCodec), frames are kept in GPU memory as CuPy arrays, and can then be encoded and muxed while still on the GPU with NVC and a FFmpeg pipe.   
The whole process is written in Python and follows a pipeline (generator) allowing to keep it simple and suitable for real-time tasks.   

I coded this in a few hours so there is definitely room for improvement, speed wise and code wise.   
    
# How to use? 
Just check the code inside `ovni/base.py`. If you need more help then this project is probably not for you.   
You can do **literally anything** you want directly on the CuPy arrays in GPU memory, and for a specific task you can write your own CUDA kernels or use the existing ones already made.   
If you create one, feel free to make a pull request. (I won't accept it if the code is messy).   

# Benchmarks
On my RTX 4080 Super, I achieve a speed of around **600 frames per second** without any operations on the frames, when decoding and encoding directly.   
Note that if written correctly, the operations should be nearly instantaneous.   

# Collaborate
Feel free to make a pull request with new CUDA kernels, since a lot of basic operations are not supported right now.
