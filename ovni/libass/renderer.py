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


    def get_frame_at(self, timestamp_ms: int) -> cp.ndarray:
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
        Overlay it on output at given coords
        Keep going until we don't have a pointer (nothing to render)

        Parameters:
            img_ptr - pointer to ASS_Image

        Returns:
            cp.ndarray
        """
        # Create output cupy RGBA array
        output = cp.zeros((self.height, self.width, 4), dtype=cp.uint8)

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


            for y in range(img.h):
                for x in range(img.w):
                    # Calculate pixel position with stride formula
                    pix_pos = y*img.stride + x

                    # Get the pixel
                    alpha = img.bitmap[pix_pos]

                    # Only update if not transparent
                    if alpha != 0:
                        # Calculate overlay alpha
                        ov_alpha = (alpha * A) / 255

                        # Output coords
                        ox = x + img.dst_x
                        oy = y + img.dst_y

                        # Overlay on the output

                        # Get original alpha
                        og_alpha = output[oy, ox, 3]

                        # Calculate coefficients of alphas
                        ov_alpha = float(ov_alpha) / 255
                        og_alpha = float(og_alpha) / 255

                        # Calculate alpha result - new combined opacity
                        alpha_r = ov_alpha + og_alpha * (1 - ov_alpha)

                        # Add RGB
                        output[oy, ox, 0] = (R * ov_alpha + output[oy, ox, 0] * og_alpha * (1 - ov_alpha)) / alpha_r
                        output[oy, ox, 1] = (G * ov_alpha + output[oy, ox, 1] * og_alpha * (1 - ov_alpha)) / alpha_r
                        output[oy, ox, 2] = (B * ov_alpha + output[oy, ox, 2] * og_alpha * (1 - ov_alpha)) / alpha_r

                        # Add alpha result
                        output[oy, ox, 3] = alpha_r * 255
            

            # Define img_ptr to the next linked one
            img_ptr = img.next

        return output

