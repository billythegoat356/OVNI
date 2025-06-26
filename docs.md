# Documentation
First you have to understand that OVNI follows a *generator pipeline*, meaning that a frame comes in, passes through a set of operations, then gets out, and the next frame comes.   
This allows you to write complex pipelines without storing frames in memory, effectively removing any limit to video length.   

# Basics
First import the following: 
```py
from ovni.base import demux_and_decode, encode, mux
```
You can now load the frames from a video file 
```py
frames = demux_and_decode(input_path="path/to/video.mp4")

# You can pass the frame_count parameter to only decode a specific amount of frames
frames_50 = demux_and_decode(input_path="path/to/video.mp4", frame_count=50)
```
`frames` will now be a *generator* of CuPy arrays `cp.ndarray` of 1 dimension (NV12 pixel format).   
You usually want to convert those to RGB format so you can operate on them.   

Do it like this 
```py
from ovni.ops import pipe_nv12_to_rgb

# Since a 3D array will be constructed, we need to pass the video dimensions
frames = pipe_nv12_to_rgb(frames, width, height)
```
You now have a generator of CuPy arrays of 3 dimensions: H x W x RGB 

You can also load one image into one frame 

```py
from ovni.base import load_image

frame = load_image("path/to/image.png")
```
This directly returns a RGB 3 dimensions CuPy array. 

After performing a bunch of operations on your frames, you can export the final video in the following way 

```py
from ovni.base import encode, mux
from ovni.ops import pipe_rgb_to_nv12

# Convert back to NV12 pixel format
frames = pipe_rgb_to_nv12(frames)

# Encode into a bytes stream
# Note that the encoder used is H264
h264_stream = encode(frames=frames, width=width, height=height, fps=fps)

# Now mux
mux(h264_stream=h264_stream, output_path="path/to/output.mp4")
```

You can also pass an optional `audio_path` to the muxing, but be careful to shorten it beforehand to match with the video length, since muxing only copies the streams. 

```py
# With audio
mux(h264_stream=h264_stream, output_path="path/to/output.mp4", audio_path="path/to/audio.mp3")
```

# Frame operations

OVNI has a limited set of methods allowing operations on frames using custom written CUDA kernels.   
These methods operate on CuPy RGB 3D arrays.   

*Note that most of these operations allow floating point parameteres, and will apply bilinear interpolation, allowing for smooth animations*   

Import all the methods like this 
```py
from ovni.ops import *
```

## Crop
The following would crop a frame from (0, 200) to (100, 300).   
Note that we pass them like this: `left`, `right`, `top`, `bottom`.
```py
frame = crop(frame, 0, 100, 200, 300)
```

## Resize
The following resizes a frame to specific dimensions. 
```py
frame = resize(frame, 1920, 1080)
```

## Scale & Translate
The following scales a frame by the given amount. 
```py
frame = scale(frame, 1.5)
```
You can also translate it 
```py
frame = translate(frame, tx=5, ty=10.5)
```
Or do both at the same time (less expensive than calling both sequentially) 
```py
frame = scale_translate(frame, 1.5, tx=5, ty=10.5)
```
All of those methods accept the parameters `dst_width` and `dst_height`, allowing to also crop the frame and process only a specific part. 
```py
frame = scale(frame, 1.5, dst_width=1920, dst_height=1080)
```


## Overlay
The following code overlays a frame on another one at given coordinates and with a specific alpha channel (opacity).   
Emit this value to simply overlay it entirely.   

*Note that this edits the source frame in place.*   
```py
overlay(frame, overlay, 100, 100, alpha=0.5)
```

## Blend
Blending is similar to overlaying, but you can pass a RGBA overlay, applying a custom alpha channel to each pixel instead of always the same. 
```py
blend(frame, overlay_with_alpha, 0, 0)
```

## Chroma key
You can turn a RGB frame into RGBA by applying chroma keying.   
This process is like removing a green screen, allowing to then blend your frame with whatever background you have. 

You can pass a frame, and the color which should represent the transparency, in a tuple of RGB format. 
```py
# This would make the green transparent
frame = chroma_key(frame, key_color=(0, 255, 0))
```

You can also customize the way transparency is determined with the `transparency_t` and `opacity_t` parameters, but it is important to understand how the algorithm works.   
The algorithm calculates the distance between each pixel's colors and your key color with a [specific formula](https://mathworld.wolfram.com/L2-Norm.html).    

When that distance is **below or equal** to `transparency_t`, the pixel becomes fully **transparent**.   
When that distance is **above or equal** to `opacity_t`, the pixel becomes fully **opaque**.   
When it's between them, it gets smoothed out.   

```py
frame = chroma_key(frame, key_color=(0, 255, 0), transparency_t=40, opacity_t=250)
```

So when you **decrease** `transparency_t`, **more** pixels close to your key color will be **visible**.   
When you **increase** it, **less** pixels close to the key color will be **visible**.   
*If you increase it too much, sharp edges may appear next to valid colors.*   

When you **increase** `opacity_t`, pixels will need to be **further away** from the key color in order to still be **fully opaque**.   
When you **decrease** it, **more** pixels **close** to the key color will be **fully opaque**.   
*If you decrease it too much, sharp edges may appear next to the key color.*   

Depending on your frame's coloring, you may want to play around with these values until it looks good. 

# Captions
OVNI lets you overlay captions on videos with [LibASS](https://github.com/libass/libass).   
The format is called Advanced SubStation Alpha (ASS) and contains a wide range of effects and animations.   
You can read the documentation [here](https://fileformats.fandom.com/wiki/SubStation_Alpha)   

Once you prepared your file, you can add it to your frames very easily like this 
```py
from ovni.ass import ASSRenderer

# Then in your pipeline
with ASSRenderer("path/to/captions.ass", width, height) as ass:
    ass.render_frame(timestamp_ms, frame)
```
The `render_frame` mehod accepts a timestamp in milliseconds to know what part of the captions to render, aswell as a frame on which it blends the captions directly. 

# CUDA context managing
Each call to decode/encode requires a CUDA context to exist in the caller thread.   
So you have to initialize it then kill it in your current thread.   
```py
from ovni.ctx import CudaCtxManager

# Initialize context
CudaCtxManager.init_ctx()

# Decode/encode
...

# Now kill it
CudaCtxManager.kill_ctx()
```
You can also use it as a pythonic context manager.   
*This approach is preferred because it ensures that the context gets killed, even if an exception is raised.*
```py
with CudaCtxManager():
    # Decode/encode
    ...
```
You should only create one context per thread. It will be reused by the internal methods.

For the main thread, OVNI is able to handle it automatically and register the kill at program exit, but explicit and controlled managing is recommended.


# Take it further
With these methods you can do pretty much everything you want.   
If you need to do some specific operations, you can write your own CUDA kernel and call it from your pipeline.   

For simple **one-time** operations you can also use image processing libraries like OpenCV or Pillow that are easy to call.   






