/*
 * This copyright notice applies to this header file only:
 *
 * Copyright (c) 2010-2025 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the software, and to permit persons to whom the
 * software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * \file config.h
 *   NVIDIA GPUs - beginning with the Kepler generation - contain a hardware-based encoder
 *   (referred to as NVENC) which provides fully-accelerated hardware-based video encoding.
 *   NvEncodeAPI provides the interface for NVIDIA video encoder (NVENC).
 * \date 2011-2025
 *  This file contains the version control depending upon macro defined
 */
#ifndef _CONFIG_NVENC_VER_H_
#define _CONFIG_NVENC_VER_H_

#ifndef NVENC_VER_13_0 
#ifndef NVENC_VER_12_1
#define NVENC_VER_12_1
#endif // !NVENC_VER_12_0
#endif // !NVENC_VER_13_0

#endif //_CONFIG_NVENC_VER_H_