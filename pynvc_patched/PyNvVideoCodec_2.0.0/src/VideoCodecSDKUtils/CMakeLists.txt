# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2010-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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



#message(STATUS "FFMPEG_DIR=$ENV{FFMPEG_DIR}")
cmake_minimum_required(VERSION 3.20)

# This must be set before enabling CUDA. Otherwise it will be set to the default ARCH by nvcc (e.g. 52 by nvcc 11.7)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Define Volta / Turing / Ampere / Ada if not set by the user
    set(CMAKE_CUDA_ARCHITECTURES
        60
        70
        72
        75
        80
        86
        #89 # Require CUDA 11.8 for CC 89 or cmake 3.23 for CMAKE_CUDA_ARCHITECTURES_{ALL,ALL_MAJOR}
    )
endif()

project(VideoCodecSDKUtils LANGUAGES CXX CUDA)

set(USE_NVTX $ENV{USE_NVTX})

set(CODEC_SOURCES
 helper_classes/NvCodec/NvDecoder/NvDecoder.cpp
 helper_classes/NvCodec/NvEncoder/NvEncoder_130.cpp
 helper_classes/NvCodec/NvEncoder/NvEncoder_121.cpp
)
set(CODEC_HDRS
 helper_classes/NvCodec/NvDecoder/NvDecoder.h
 helper_classes/NvCodec/NvEncoder/NvEncoder_130.h
 helper_classes/NvCodec/NvEncoder/NvEncoder_121.h
 helper_classes/Utils/NvCodecUtils.h
 helper_classes/Utils/FFmpegDemuxer.h
 helper_classes/Utils/FFmpegMuxer.h
 helper_classes/Utils/ColorSpace.h
 helper_classes/Utils/FFmpegStreamer.h
 helper_classes/Utils/Logger.h
 helper_classes/Utils/NvEncoderCLIOptions.h
 helper_classes/Utils/cuvidFunctions.h
 helper_classes/Utils/cuvid_function_pointer.h
 helper_classes/Utils/cuvid_dlopen.h
 helper_classes/Utils/cuvid_dlopen_unix.cpp
 helper_classes/Utils/cuvid_dlopen_windows.cpp
 helper_classes/Utils/
 Interface/cuviddec.h
 Interface/nvcuvid.h
 Interface/nvEncodeAPI_130.h
 Interface/nvEncodeAPI_121.h
 Interface/configNvEncVer.h
)

set(CODEC_CUDA_UTILS
 helper_classes/Utils/ColorSpace.cu
)

if(WIN32)
    set(NV_FFMPEG_HDRS ${FFMPEG_DIR}/include)
endif()

find_package(CUDAToolkit 11.2 REQUIRED)
include(GenerateExportHeader)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=\"sm_50,compute_50\")
# CUDA 11.7 and later is compiled with /MT option (staticaly linked with C runtime), 
# since we are statically linking cudart_static.lib with the app, 
# we need to make sure that all .obj files linked are compiled with /MT to avoid linker warning LNK4098
if (WIN32 AND CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 7)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler /MT)
endif()
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if (WIN32 AND CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 7)
    target_compile_options(${PROJECT_NAME} PRIVATE "/MT")
endif()

if(WIN32)
	if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
		add_compile_options("-fstack-protector-strong")
		add_compile_options("-Wall")
		add_compile_options("-Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough")
		add_compile_options("--fPIC")
	endif()
endif()


add_library(
    VideoCodecSDKUtils
    ${CODEC_SOURCES}
    ${CODEC_HDRS}
    ${NV_FFMPEG_HDRS}
    ${CODEC_CUDA_UTILS}
)

set(TC_VERSION_MAJOR 1)
set(TC_VERSION_MINOR 0)

generate_export_header(VideoCodecSDKUtils)

target_link_libraries(
    VideoCodecSDKUtils
    PUBLIC
        CUDA::cuda_driver
		CUDA::cudart
)


target_compile_features(VideoCodecSDKUtils PRIVATE cxx_std_17)
set_property(
    TARGET VideoCodecSDKUtils
    PROPERTY
        # required for shared Python modules in case library is build statically on Unix
        POSITION_INDEPENDENT_CODE
        ON
)

set(VIDEO_CODEC_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Interface"
    CACHE PATH "Path to Video Codec SDK interface headers"
)
target_include_directories(
    VideoCodecSDKUtils
    PUBLIC
        helper_classes/NvCodec
        helper_classes/Utils
        ${VIDEO_CODEC_INTERFACE_DIR} # TODO: check if it can be made private!
        ${CMAKE_CURRENT_BINARY_DIR}
)


if(UNIX)
    target_link_libraries(VideoCodecSDKUtils PUBLIC pthread)
endif(UNIX)

include(../cmake/ffmpeg.cmake)
target_include_directories(VideoCodecSDKUtils PUBLIC ${NV_FFMPEG_INCLUDE_DIR})
target_link_libraries(VideoCodecSDKUtils PUBLIC ${NV_FFMPEG_LIBRARIES})

option(USE_NVTX "Use NVTX for profiling" FALSE)

if(USE_NVTX)
option(FETCHCONTENT_QUIET OFF)
include(FetchContent)
fetchcontent_declare(
  nvtx_cpp
  URL https://github.com/NVIDIA/NVTX/archive/v3.1.0.tar.gz # Oct 5 2022
  URL_HASH
  SHA256=dc4e4a227d04d3da46ad920dfee5f7599ac8d6b2ee1809c9067110fb1cc71ced
  SOURCE_SUBDIR c
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
fetchcontent_makeavailable(nvtx_cpp)
endif()

if(USE_NVTX)
   target_link_libraries(VideoCodecSDKUtils PRIVATE nvtx3-cpp)
   target_compile_definitions(VideoCodecSDKUtils PRIVATE -DUSE_NVTX=1)
endif()

message(STATUS "HAS_BSF=${HAS_BSF}")

# Create separate targets for each NVENC version
function(create_codec_utils_target VERSION_SUFFIX VERSION_DEFINE)
    set(TARGET_NAME "VideoCodecSDKUtils${VERSION_SUFFIX}")
    
    add_library(${TARGET_NAME}
        ${CODEC_SOURCES}
        ${CODEC_HDRS}
        ${NV_FFMPEG_HDRS}
        ${CODEC_CUDA_UTILS}
    )

    target_compile_definitions(${TARGET_NAME} PRIVATE ${VERSION_DEFINE})
    
    target_link_libraries(${TARGET_NAME}
        PUBLIC
            CUDA::cuda_driver
            CUDA::cudart
    )

    if(UNIX)
        target_link_libraries(${TARGET_NAME} PUBLIC pthread)
    endif()

    target_include_directories(${TARGET_NAME}
        PUBLIC
            helper_classes/NvCodec
            helper_classes/Utils
            ${VIDEO_CODEC_INTERFACE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${NV_FFMPEG_INCLUDE_DIR}
    )

    target_link_libraries(${TARGET_NAME} PUBLIC ${NV_FFMPEG_LIBRARIES})

    if(USE_NVTX)
        target_link_libraries(${TARGET_NAME} PRIVATE nvtx3-cpp)
        target_compile_definitions(${TARGET_NAME} PRIVATE -DUSE_NVTX=1)
    endif()

    set_property(TARGET ${TARGET_NAME}
        PROPERTY
            POSITION_INDEPENDENT_CODE ON
    )
endfunction()

# Create both versions
create_codec_utils_target("_121" "NVENC_VER_12_1")
create_codec_utils_target("_130" "NVENC_VER_13_0")
