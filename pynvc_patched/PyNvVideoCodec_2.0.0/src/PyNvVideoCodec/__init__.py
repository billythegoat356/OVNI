# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""

"""

__author__ = "NVIDIA"
__copyright__ = "Copyright 2025, NVIDIA"
__credits__ = []
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "NVIDIA"
__email__ = "TODO"
__status__ = "Production"

from ast import Str
import string
import os
import platform
import sys
import site
from pathlib import Path
import importlib.util
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _setup_environment():
    """Setup environment variables and paths"""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    machine = platform.machine()
    
    if platform.system() == 'Windows':
        try:
            # Add CUDA path if available
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path:
                dll_paths = [
                    os.path.join(cuda_path, "bin"),
                    os.path.join(cuda_path, "lib", "x64"),
                    package_dir,  # Package directory for ffmpeg DLLs
                    os.path.join(package_dir, "lib")  # Additional lib directory
                ]
                
                # Add paths to DLL search path
                if hasattr(os, 'add_dll_directory'):
                    for path in dll_paths:
                        if os.path.exists(path):
                            os.add_dll_directory(path)
                            logger.debug(f"Added DLL directory: {path}")
        except Exception as e:
            logger.error(f"Error setting up DLL paths: {e}")
    else:
        # Linux: Update LD_LIBRARY_PATH
        lib_paths = [
            os.path.join(package_dir, "lib"),
            "/usr/local/cuda/lib64",
        ]
        
        # Add architecture-specific paths
        if machine == 'aarch64':
            lib_paths.extend([
                "/usr/lib/aarch64-linux-gnu",
                "/usr/local/cuda/targets/aarch64-linux/lib"
            ])
        else:  # x86_64
            lib_paths.append("/usr/lib/x86_64-linux-gnu")
        
        if "LD_LIBRARY_PATH" in os.environ:
            lib_paths.extend(os.environ["LD_LIBRARY_PATH"].split(":"))
        os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, lib_paths))

def _get_module_path(module_suffix):
    """Get the appropriate module file path"""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine()
    
    if platform.system() == "Windows":
        ext = ".pyd"
        possible_names = [
            f"PyNvVideoCodec{module_suffix}.cp{py_version}-win_amd64{ext}"
        ]
    else:
        ext = ".so"
        if machine == 'aarch64':
            possible_names = [
                f"PyNvVideoCodec{module_suffix}.cpython-{py_version}-aarch64-linux-gnu{ext}",
                f"PyNvVideoCodec{module_suffix}.cpython-{py_version}-aarch64-linux{ext}",
                f"PyNvVideoCodec{module_suffix}.cp{py_version}-manylinux_2_28_aarch64{ext}"
            ]
        else:  # x86_64
            possible_names = [
                f"PyNvVideoCodec{module_suffix}.cpython-{py_version}-x86_64-linux-gnu{ext}",
                f"PyNvVideoCodec{module_suffix}.cpython-{py_version}-x86_64-linux{ext}",
                f"PyNvVideoCodec{module_suffix}.cp{py_version}-manylinux_2_28_x86_64{ext}"
            ]
    
    # Try each possible name
    for name in possible_names:
        path = os.path.join(package_dir, name)
        if os.path.exists(path):
            logger.debug(f"Found module: {path}")
            return path
    
    # If no module found, list directory contents for debugging
    logger.error(f"Available files in {package_dir}:")
    try:
        for file in os.listdir(package_dir):
            logger.error(f"  {file}")
    except Exception as e:
        logger.error(f"Error listing directory: {e}")
    
    # Return the first name for error message
    return os.path.join(package_dir, possible_names[0])

def _get_driver_version():
    """Get the driver version using VersionCheck"""
    try:
        from .VersionCheck import DriverWrapper
        driver = DriverWrapper()
        version = driver.GetDriverVersion()
        logger.info(f"Supported NVENC Driver Version: {version}")
        return version
    except Exception as e:
        logger.error(f"Error getting driver version: {e}")
        raise

try:
    # Setup environment
    _setup_environment()

    # Get driver version
    supportedNvEncVersion = _get_driver_version()

    # Define driver version thresholds
    NVENC_VER_13 = 13 * 16  # Driver version for NVENC 13
    NVENC_VER_12 = 12 * 16 + 1  # Driver version for NVENC 12.1

    # Determine which version to use
    if supportedNvEncVersion >= NVENC_VER_13:
        module_suffix = "_130"
        logger.info("Using NVENC 13.0 version")
    elif supportedNvEncVersion >= NVENC_VER_12:
        module_suffix = "_121"
        logger.info("Using NVENC 12.1 version")
    else:
        raise RuntimeError("Driver version is too old")

    # Get module path
    module_path = _get_module_path(module_suffix)
    if not os.path.exists(module_path):
        raise ImportError(f"Module not found: {module_path}")

    # Import the module
    spec = importlib.util.spec_from_file_location("_PyNvVideoCodec", module_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for {module_path}")
        
    _PyNvVideoCodec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_PyNvVideoCodec)

    # Import all symbols into current namespace
    globals().update({name: getattr(_PyNvVideoCodec, name) for name in dir(_PyNvVideoCodec) 
                     if not name.startswith('_')})
    
    # Import submodules
    from .decoders.SimpleDecoder import SimpleDecoder
    from .decoders.ThreadedDecoder import ThreadedDecoder
    from .transcoder.Transcoder import Transcoder

except Exception as e:
    logger.error(f"Error importing PyNvVideoCodec: {e}")
    logger.error(f"System: {platform.system()}")
    logger.error(f"Python version: {sys.version}")
    if platform.system() == "Linux":
        logger.error(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    raise

def tostring(val):
    if type(val) == dict:
        x = dict(map(lambda x: (x[0], tostring(x[1])), val.items()))
    else:
        x = str(val)
        x = x.replace("]", "")
        x = x.replace("[", "")
    return x

def format_optional_params(optional_param):
    param = tostring(optional_param)

    if "slice" in param:
        param["slice_mode"] = param["slice"]["mode"]
        param["slice_data"] = param["slice"]["data"]
        del param["slice"]

    if "timinginfo" in param:
        param["num_unit_in_ticks"] = param["timinginfo"]["num_unit_in_ticks"]
        param["timescale"] = param["timinginfo"]["timescale"]
        del param["timinginfo"]

    return param

def CreateEncoder(width, height, fmt, usecpuinputbuffer, **kwargs):
    cudacontext = 0
    cudastream = 0
    if "cudacontext" in kwargs:
        cudacontext = int(kwargs["cudacontext"])
        del kwargs["cudacontext"]
    if "cudastream" in kwargs:
        cudastream = int(kwargs["cudastream"])
        del kwargs["cudastream"]

    optional_args = format_optional_params(kwargs)
    return PyNvEncoder(width, height, fmt, cudacontext, cudastream, usecpuinputbuffer, optional_args)
