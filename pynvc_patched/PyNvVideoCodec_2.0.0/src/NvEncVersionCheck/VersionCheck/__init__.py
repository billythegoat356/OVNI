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

import os
import sys
import platform
import ctypes
from pathlib import Path

def _setup_environment():
    """Setup environment variables and paths"""
    if platform.system() == "Linux":
        # Add CUDA library path for Linux
        cuda_paths = [
            "/usr/local/cuda",
            "/usr/cuda",
            os.environ.get("CUDA_PATH", ""),
            "/usr/lib/x86_64-linux-gnu"  # Common location for NVIDIA libraries on Ubuntu
        ]
        
        # Add NVIDIA driver libraries path
        try:
            nvidia_paths = [
                path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
                if "nvidia" in path.lower()
            ]
            cuda_paths.extend(nvidia_paths)
        except Exception:
            pass

        for cuda_path in cuda_paths:
            if cuda_path and os.path.exists(cuda_path):
                lib_path = os.path.join(cuda_path, "lib64") if not cuda_path.endswith(("lib64", "lib")) else cuda_path
                if os.path.exists(lib_path):
                    try:
                        # Pre-load NVIDIA libraries
                        if os.path.exists(os.path.join(lib_path, "libnvidia-encode.so.1")):
                            ctypes.CDLL(os.path.join(lib_path, "libnvidia-encode.so.1"))
                        
                        # Update LD_LIBRARY_PATH
                        if lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
                            if "LD_LIBRARY_PATH" in os.environ:
                                os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ['LD_LIBRARY_PATH']}"
                            else:
                                os.environ["LD_LIBRARY_PATH"] = lib_path
                    except Exception as e:
                        print(f"Warning: Failed to load NVIDIA libraries from {lib_path}: {e}")

def _find_module():
    """Find and load the VersionCheck module."""
    module_name = "VersionCheck"
    current_dir = Path(__file__).parent.absolute()
    
    if platform.system() == "Windows":
        extensions = ['.pyd']
        lib_dirs = ["lib"]
    else:
        extensions = ['.so']
        lib_dirs = ["lib", "lib64"]
    
    # Add site-packages to search paths
    import site
    site_packages = site.getsitepackages()
    
    search_paths = [
        current_dir,
        *[current_dir / lib_dir for lib_dir in lib_dirs],
        *[Path(sp) / "VersionCheck" for sp in site_packages],
        *[Path(sp) / "lib" / "VersionCheck" for sp in site_packages],
    ]
    
    for path in search_paths:
        for ext in extensions:
            module_path = path / f"{module_name}{ext}"
            if module_path.exists():
                return str(module_path)
                
    raise ImportError(
        f"Could not find {module_name} module.\n"
        f"Searched in:\n" + "\n".join(f"- {p}" for p in search_paths)
    )

try:
    # Setup environment before importing
    _setup_environment()

    # Add the directory containing the module to sys.path
    module_path = _find_module()
    module_dir = str(Path(module_path).parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    # Import all symbols from the module
    from VersionCheck import DriverWrapper

except Exception as e:
    print(f"Error initializing VersionCheck: {e}")
    print(f"System: {platform.system()}")
    print(f"Python version: {sys.version}")
    if platform.system() == "Linux":
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    raise

# Clean up namespace
del os, sys, platform, Path, ctypes
del _find_module, _setup_environment 