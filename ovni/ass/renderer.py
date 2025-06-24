import ctypes
import atexit
from typing import Self, Type, Literal
from types import TracebackType

import cupy as cp

from .lib import LibASS, ASS_Image
from ..ops import blend_ass_image, blend


class ASSRenderer:
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

        # Create change detection and last rendered frame, for optimization
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

        # Schedule unload
        atexit.register(self.unload_renderer)

    def unload_renderer(self) -> None:
        """
        Unloads the renderer and associated resources.
        Usually called automatically on object deletion, and scheduled on script exit, but you can call it manually.

        Returns:
            None
        """
    
        # Unload renderer
        if self.renderer is not None:
            LibASS.obj.ass_renderer_done(self.renderer)
            self.renderer = None

        # Free track
        if self.track is not None:
            LibASS.obj.ass_free_track(self.track)
            self.track = None

    
    def __enter__(self) -> Self:
        """
        Enter the context manager
        Does not do anything since the renderer is initialized on object creation directly.

        Returns:
            Self
        """
        return self
    

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> Literal[False]:
        """
        Exit the context manager
        Unloads the renderer

        Returns:
            Literal[False]
        """
        self.unload_renderer()

        return False # Do not suppress exception
        

    def __del__(self) -> None:
        """
        Unloads on object deletion.

        Returns:
            None
        """
        self.unload_renderer()


    def render_frame(self, timestamp_ms: int, background_frame: cp.ndarray | None = None) -> cp.ndarray | None:
        """
        Render a frame at a specific timestamp
        You can pass a background_frame, on which the rendered frame will be blended

        Parameters:
            timestamp_ms: int
            background_frame: cp.ndarray | None = None - optional background frame to blend the rendered frame in

        Returns:
            cp.ndarray | None - a cupy array of the rendered frame, or None if a background_frame was given or if the timestamp is out of the rendered ones
        """
        # Render frame
        img_ptr = LibASS.obj.ass_render_frame(
            self.renderer,
            self.track,
            ctypes.c_int64(timestamp_ms),
            ctypes.byref(self.detect_change)
        )

        # If there is no frame at this timestamp, we return None
        if not img_ptr:
            return None

        # We process the frame if its the first one, or if there was a change with the previous one (we don't handle positions changing, I assume this is a very rare case)
        if self.last_rendered_frame is None or self.detect_change.value != 0:
            # Convert to RGBA cupy array
            frame = self._convert_to_rgba_array(img_ptr)

            # Set last rendered frame
            self.last_rendered_frame = frame

        # Otherwise we pass the previously rendered one
        else:
            frame = self.last_rendered_frame

        # If a background frame is not passed, we return a copy of the frame so the user can operate on it
        if background_frame is None:
            return frame.copy()
        
        # Otherise we blend it on the background_frame
        else:
            blend(background_frame, frame, 0, 0)
    
    
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
