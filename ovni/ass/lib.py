import ctypes
import atexit


# ASS_Image structure definition (required for attributes access)
class ASS_Image(ctypes.Structure):
    pass

ASS_Image._fields_ = [
    ("w", ctypes.c_int), # width of bitmap
    ("h", ctypes.c_int), # height of bitmap
    ("stride", ctypes.c_int), # stride
    ("bitmap", ctypes.POINTER(ctypes.c_ubyte)), # pointer to bitmap
    ("color", ctypes.c_uint32), # RGBA of bitmap
    ("dst_x", ctypes.c_int), # bitmap X placement into the video frame
    ("dst_y", ctypes.c_int), # bitmap Y placement into the video frame
    ("next", ctypes.POINTER(ASS_Image)) # Pointer to next image, or null
]



class LibASS:
    obj = None # Loaded object
    library = None # Library

    @classmethod
    def is_loaded(cls) -> bool:
        """
        Returns if the library is loaded

        Returns:
            bool
        """
        return cls.library is not None

    @classmethod
    def load(cls) -> None:
        """
        Loads the object, library and the other components
        Do not call this if you don't plan on using ASS captions

        Returns:
            None
        """
        # Load library
        cls.obj = ctypes.CDLL('libass.so')
        cls.set_types()

        # We can now load the library
        cls.library = cls.obj.ass_library_init()

        # Register the unload at exit
        atexit.register(cls.unload)


    @classmethod
    def unload(cls) -> None:
        """
        Unloads the library.

        Returns:
            None
        """
        if cls.library is not None:
            cls.obj.ass_library_done(cls.library)
            cls.library = None


    @classmethod
    def set_types(cls) -> None:
        """
        Sets the necessary types on the library
        Those are based on the C headers

        Returns:
            None
        """
        # Initialization of library
        cls.obj.ass_library_init.argtypes = [] # Nothing
        cls.obj.ass_library_init.restype = ctypes.c_void_p # Pointer to ASS_Library

        # Finalization of library
        cls.obj.ass_library_done.argtypes = [ctypes.c_void_p] # Pointer to ASS_Library
        cls.obj.ass_library_done.restype = None # Void
        
        # Initialization of renderer
        cls.obj.ass_renderer_init.argtypes = [ctypes.c_void_p] # Pointer to ASS_Library
        cls.obj.ass_renderer_init.restype = ctypes.c_void_p # Pointer to ASS_Renderer

        # Finalization of renderer
        cls.obj.ass_renderer_done.argtypes = [ctypes.c_void_p] # Pointer to ASS_Renderer
        cls.obj.ass_renderer_done.restype = None # Void
        
        # Set the frame size for rendering
        cls.obj.ass_set_frame_size.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] # Pointer to ASS_Renderer, width, height
        cls.obj.ass_set_frame_size.restype = None # Void
        
        # Set source image size for some effects
        cls.obj.ass_set_storage_size.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] # Pointer to ASS_Renderer, width, height
        cls.obj.ass_set_storage_size.restype = None # Void

        # Set the fonts
        cls.obj.ass_set_fonts.argtypes = [
            ctypes.c_void_p,  # Pointer to ASS_Renderer
            ctypes.c_char_p,  # Path to a default font
            ctypes.c_char_p,  # Default family
            ctypes.c_int,     # Font provider, 1 for fontconfig
            ctypes.c_char_p,  # Path for fontconfig optional configuration
            ctypes.c_int      # Whether fontconfig cache should be updated
        ]
        cls.obj.ass_set_fonts.restype = None # Void

        # Read the ASS file
        cls.obj.ass_read_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p] # Pointer to ASS_Library, filename, encoding
        cls.obj.ass_read_file.restype = ctypes.c_void_p # Pointer to ASS_Track

        # Free the track
        cls.obj.ass_free_track.argtypes = [ctypes.c_void_p] # Pointer to ASS_Track
        cls.obj.ass_free_track.restype = None # Void
        
        # Render a frame
        cls.obj.ass_render_frame.argtypes = [
            ctypes.c_void_p, # Pointer to ASS_Renderer
            ctypes.c_void_p, # Pointer to ASS_Track
            ctypes.c_int64, # Video timestamp in milliseconds
            ctypes.POINTER(ctypes.c_int) # Pointer to a variable storing change detection for optimization
        ]
        cls.obj.ass_render_frame.restype = ctypes.POINTER(ASS_Image) # Pointer to ASS_Image - this one needs a definition because we access its attributes later on

        
