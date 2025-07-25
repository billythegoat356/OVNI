
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

set(FFMPEG_DIR $ENV{FFMPEG_DIR})
set(NV_FFMPEG_LIBRARIES "")
set(EMPTY,"")
if ("${FFMPEG_DIR}" STREQUAL "${EMPTY}")
    set(FFMPEG_DIR ${CMAKE_SOURCE_DIR}/external/ffmpeg)
	if(${CMAKE_SYSTEM_PROCESSOR} MATCHES AMD64)
		set(NV_FFMPEG_LIBRARY_DIR "${FFMPEG_DIR}/lib/x64")
	elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
		set(NV_FFMPEG_LIBRARY_DIR "${FFMPEG_DIR}/lib/x86_64")
	elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
		set(NV_FFMPEG_LIBRARY_DIR "${FFMPEG_DIR}/lib/aarch64")
	endif()
else()
	if(${CMAKE_SYSTEM_PROCESSOR} MATCHES AMD64)
		set(NV_FFMPEG_LIBRARY_DIR "${FFMPEG_DIR}/bin")
	else()
		set(NV_FFMPEG_LIBRARY_DIR "${FFMPEG_DIR}/lib")
	endif()
endif()
message(STATUS "Using FFMPEG_DIR=${FFMPEG_DIR}")
set(NV_FFMPEG_INCLUDE_DIR "${FFMPEG_DIR}/include" CACHE INTERNAL NV_FFMPEG_INCLUDE_DIR)
SET(NV_FFMPEG_LIBRARY_DIR  "${NV_FFMPEG_LIBRARY_DIR}" CACHE INTERNAL "NV_FFMPEG_LIBRARY_DIR")

message(STATUS "Using CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")


macro(link_av_component target lib_name)
	if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "AMD64")
			find_library(${lib_name}_library
				NAMES ${lib_name}
				HINTS "${FFMPEG_DIR}/bin" "${FFMPEG_DIR}/lib" "${FFMPEG_DIR}/lib/x64"
			)
	elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
			find_library(${lib_name}_library
				NAMES ${lib_name}
				HINTS "${FFMPEG_DIR}/bin" HINTS "${FFMPEG_DIR}/lib" "${FFMPEG_DIR}/lib/x86_64"
			)
	elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
	        find_library(${lib_name}_library
				NAMES ${lib_name}
				HINTS "${FFMPEG_DIR}/bin" "${FFMPEG_DIR}/lib" "${FFMPEG_DIR}/lib/aarch64"
			)
	endif()
	message(STATUS "Link ${${lib_name}_library}")
	list(APPEND NV_FFMPEG_LIBRARIES ${${lib_name}_library})
endmacro()

link_av_component(VideoCodecSDKUtils avformat)
link_av_component(VideoCodecSDKUtils avcodec)
link_av_component(VideoCodecSDKUtils swresample)
link_av_component(VideoCodecSDKUtils avutil)