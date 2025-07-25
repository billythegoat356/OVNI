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

cmake_minimum_required(VERSION 3.21)

project(PyNvVideoCodec)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

option(DEMUX_ONLY "Compiling demuxer code only" FALSE)
set(DEMUX_ONLY $ENV{DEMUX_ONLY})

option(USE_NVTX "enable nvtx support" FALSE)
set(USE_NVTX $ENV{USE_NVTX})

set(FFMPEG_DIR $ENV{FFMPEG_DIR})
set(EMPTY,"")
if ("${FFMPEG_DIR}" STREQUAL "${EMPTY}")
    set(FFMPEG_DIR ${CMAKE_SOURCE_DIR}/external/ffmpeg)
endif()


if (NOT DEMUX_ONLY)
    find_package(CUDAToolkit 11.2 REQUIRED)
endif()

if (NOT DEMUX_ONLY)
    add_subdirectory(VideoCodecSDKUtils)
endif()
add_subdirectory(PyNvVideoCodec)
