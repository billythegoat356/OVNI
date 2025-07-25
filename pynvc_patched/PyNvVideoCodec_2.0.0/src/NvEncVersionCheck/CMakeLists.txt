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

cmake_minimum_required(VERSION 3.15)
project(VersionCheck VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Make build type explicit if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Find Python using the version from parent project
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Include pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz
    URL_HASH SHA256=d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c
)
FetchContent_MakeAvailable(pybind11)

# Add include directory for the header file
set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VersionCheck)

# Add the pybind11 module
pybind11_add_module(VersionCheck VersionCheck/NvEncVersionCheck.cpp)

# Use target_include_directories
target_include_directories(VersionCheck 
    PRIVATE 
        ${HEADER_DIR}
        ${Python_INCLUDE_DIRS}  # Add Python include dirs
        ${pybind11_INCLUDE_DIRS}  # Add pybind11 include dirs
)

# Platform specific settings
if(WIN32)
    target_compile_definitions(VersionCheck PRIVATE _WIN32)
else()
    target_compile_definitions(VersionCheck PRIVATE LINUX)
    find_package(CUDA REQUIRED)  # Find CUDA on Linux
    target_include_directories(VersionCheck PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# Link dependencies
if(WIN32)
    target_link_libraries(VersionCheck PRIVATE ${CMAKE_DL_LIBS})
else()
    target_link_libraries(VersionCheck PRIVATE ${CMAKE_DL_LIBS} dl)  # Add dl library for Linux
endif()

# Set target properties
set_target_properties(VersionCheck PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
    RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Update RPATH for Linux
if(UNIX)
    set_target_properties(VersionCheck PROPERTIES
        INSTALL_RPATH "\$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Install Python module files
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/VersionCheck/__init__.py"
    DESTINATION "${Python_SITELIB}/VersionCheck"
)

# Install the compiled module
install(TARGETS VersionCheck 
    LIBRARY DESTINATION "${Python_SITELIB}/VersionCheck"
    RUNTIME DESTINATION "${Python_SITELIB}/VersionCheck"
)