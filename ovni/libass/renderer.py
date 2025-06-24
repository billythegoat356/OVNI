import ctypes

import numpy as np
import cupy as cp

from .lib import LibASS, ASS_Image
from ..ops import blend_ass_image


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
            b"Arial",  # default family
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

        # We only do not re-process if nothing changed (positions change aren't handled, I assume these are a very rare use-case)
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
        Uses the custom CUDA kernel

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

            # Blend image on output
            blend_ass_image(output, img)

            # Define img_ptr to the next linked one
            img_ptr = img.next

        return output
