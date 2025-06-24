import ctypes

import numpy as np
import cupy as cp

from .lib import LibASS, ASS_Image
from ..ops import blend


class Renderer:
    def __init__(self, ass_file: str, width: int, height: int) -> None:
        """
        Create a renderer for a given ASS file, width and height

        Parameters:
            ass_file: str
            width: int
            height: int
        
        Returns:
            None
        """
        if not LibASS.is_loaded():
            # Load if needed
            LibASS.load()

        self.ass_file = ass_file

        self.width = width
        self.height = height

        # Create renderer and track
        self.renderer = LibASS.obj.ass_renderer_init(LibASS.library)
        self.track = LibASS.obj.ass_read_file(LibASS.library, ass_file.encode(), None)

        self.init_renderer()

        # Create change detection and last rendered frame
        self.detect_change = ctypes.c_int() # 0 if no change, 1 if positions changed, 2 if content changed
        self.last_rendered_frame: cp.ndarray | None = None

    
    def init_renderer(self) -> None:
        """
        Initialize the renderer with the given parameters

        Returns:
            None
        """
        # Set frame size
        LibASS.obj.ass_set_frame_size(self.renderer, self.width, self.height)
        
        # Set storage size
        LibASS.obj.ass_set_storage_size(self.renderer, self.width, self.height)
        
        # Set fonts
        LibASS.obj.ass_set_fonts(
            self.renderer,
            None,  # default font
            None,  # default family
            1,     # fontprovider (1 = fontconfig) 
            None,  # fontconfig config
            0      # update fontconfig
        )


    def render_frame(self, timestamp_ms: int) -> cp.ndarray:
        """
        Render a frame at a specific timestamp

        Parameters:
            timestamp_ms: int

        Returns:
            cp.ndarray | None - a cupy array of the rendered frame
        """
        # Render frame
        img_ptr = LibASS.obj.ass_render_frame(
            self.renderer,
            self.track,
            ctypes.c_int64(timestamp_ms),
            ctypes.byref(self.detect_change)
        )

        # If there is no frame, return an empty array full of zeros
        if not img_ptr:
            return cp.zeros((self.height, self.width, 4), dtype=cp.uint8)

        # We only do not re-process if nothing changed (positions change aren't handled)
        if self.detect_change.value == 0 and self.last_rendered_frame is not None:
            return self.last_rendered_frame.copy() # Return a copy in case user operates on it
        
        # Convert to RGBA cupy array
        frame = self._convert_to_rgba_array(img_ptr)

        # Set last rendered frame
        self.last_rendered_frame = frame
        return frame.copy() # Return a copy in case user operates on it
    
    def _convert_to_rgba_array(self, img_ptr) -> cp.ndarray:
        """
        Convert libass image linked list to RGBA cupy array
        ---------
        Access struct content from pointer
        Calculate color as RGB from .color
        Create array with width and height with default color
        Iterate over bitmap:
            - calculate alpha
            - add it on the array
        Blend it on output at given coords using formula: a1 + a2 * (1 - a1)
        Keep going until we don't have a pointer (nothing to render)

        Parameters:
            img_ptr - pointer to ASS_Image

        Returns:
            cp.ndarray
        """
        # Create output cupy RGBA array
        output = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Walk through linked list of images
        while img_ptr:
            # Access content of pointer (ASS_Image)
            img = img_ptr.contents

            # Image color is RGBA
            color = img.color

            R = color >> 24 # Shift by 3 bytes
            G = (color >> 16) & 0xff # Shift by 2 bytes, keep first byte
            B = (color >> 8) & 0xff # Shift by 1 byte, keep first byte
            A = (255 - (color & 0xff)) # Keep first byte, and libass inverts alpha

            # Create buffer with correct type and load bitmap
            buf_type = ctypes.c_ubyte * (img.h * img.stride)
            buf = buf_type.from_address(ctypes.addressof(img.bitmap.contents))

            # Load into numpy array
            np_bitmap = np.frombuffer(buf, dtype=np.uint8).reshape((img.h, img.stride))
            bitmap = np_bitmap[:, :img.w] # Remove stride
            
            # Where to update
            mask = bitmap > 0

            # Extract coordinates of pixels to update
            ys, xs = np.nonzero(mask)
            
            # Calculate overlay alpha normalized to [0,1]
            ov_alpha = (bitmap[ys, xs].astype(np.float32) * A) / (255 * 255)

            # Compute original alpha normalized to [0,1]
            og_alpha = output[ys + img.dst_y, xs + img.dst_x, 3].astype(np.float32) / 255

            # Compute resulting alpha
            alpha_r = ov_alpha + og_alpha * (1 - ov_alpha)

            # Precompute coefficients for color channels
            coef_overlay = ov_alpha / alpha_r
            coef_orig = og_alpha * (1 - ov_alpha) / alpha_r

            # Update color channels all at once using broadcasting
            output[ys + img.dst_y, xs + img.dst_x, 0] = R * coef_overlay + output[ys + img.dst_y, xs + img.dst_x, 0] * coef_orig
            output[ys + img.dst_y, xs + img.dst_x, 1] = G * coef_overlay + output[ys + img.dst_y, xs + img.dst_x, 1] * coef_orig
            output[ys + img.dst_y, xs + img.dst_x, 2] = B * coef_overlay + output[ys + img.dst_y, xs + img.dst_x, 2] * coef_orig

            # Update alpha channel
            output[ys + img.dst_y, xs + img.dst_x, 3] = alpha_r * 255

            # Define img_ptr to the next linked one
            img_ptr = img.next

        return cp.asarray(output)
