from typing import Any, Self, Type, Literal
from types import TracebackType
import threading
import atexit

import pycuda.driver as cuda

from .config import GPU_ID



"""
Manager of thread-specific cuda contexts
------
Each cuda context can only be used from the thread it has been created in
"""
class CudaCtxManager:
    # Mapping of thread IDs to active cuda contexts
    ACTIVE_CTXS = {}

    @staticmethod
    def get_thread_id() -> int:
        """
        Returns the ID of the caller thread

        Returns:
            int
        """
        return threading.get_ident()

    @classmethod
    def get_ctx(cls) -> Any:
        """
        Returns the CUDA context associated with the current thread
        Raises an exception if it doesn't exist
        ---------
        For user convenience, if it doesn't exist and the caller thread is the main one, we create one and register it to kill at program exit


        Returns:
            cuda_ctx
        """
        tid = cls.get_thread_id()

        # Context doesn't exist
        if tid not in cls.ACTIVE_CTXS:

            # For user convenience, if the caller thread is the main thread, we create a context
            if tid == threading.main_thread().ident:
                cls.init_ctx()
                # We can then register it at exit because it is the main thread
                # This is not possible for other threads!
                atexit.register(cls.kill_ctx)

            else:
                raise ValueError("A CUDA context was requested but there is none created in the current thread!")
            
        return cls.ACTIVE_CTXS[tid]
    
    @classmethod
    def init_ctx(cls) -> None:
        """
        Initializes a CUDA context in the caller thread
        ---------
        If you use this method, make sure to then call `kill_ctx` in the same thread
        Also make sure to only call this method once per thread or use the class as a context manager instead

        Returns:
            None
        """
        tid = cls.get_thread_id()

        # Raise exception if there is already a context in that thread
        if tid in cls.ACTIVE_CTXS:
            raise ValueError("Tried to initialize a CUDA context in a thread that already has one!")

        cuda_device = cuda.Device(GPU_ID) # Apparently this is also thread-specific
        cuda_ctx = cuda_device.retain_primary_context() # Retain the primary context because it is shared with PyNVC
        cuda_ctx.push()

        cls.ACTIVE_CTXS[tid] = cuda_ctx


    @classmethod
    def kill_ctx(cls) -> None:
        """
        Kills the CUDA context associated with the current thread
        ---------
        WARNING: Always call this method when you finished using the context in a thread, unless you are using the class as a context manager
        
        Returns:
            None
        """
        tid = cls.get_thread_id()

        # No context in the current thread
        if tid not in cls.ACTIVE_CTXS:
            raise ValueError("Tried to kill a CUDA context but none exists in the current thread!")
        
        cuda_ctx = cls.ACTIVE_CTXS[tid]

        cuda_ctx.pop()
        cuda_ctx.detach()
        del cls.ACTIVE_CTXS[tid]


    def __enter__(self) -> Self:
        """
        Enter method of the class as a context manager for easier handling of CUDA context
        Initializes the CUDA context

        Returns:
            Self
        """
        self.__class__.init_ctx()
        return self
    
    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> Literal[False]:
        """
        Exit method of the class as a context manager
        Kills the CUDA context

        Parameters:
            exc_type: Type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None

        Returns:
            Literal[False]
        """
        self.__class__.kill_ctx()

        return False # Do not suppress exception