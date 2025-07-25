/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <memory>
#include <vector>

#include "DecoderCommon.hpp"
#include <iostream>
#include <sstream>
#include "NvEncoderCuda.h"
#include "PyCAIMemoryView.hpp"
#include <map>
#include "NvEncoderClInterface.hpp"
#include "FFmpegMuxer.h"
#include "SimpleDecoder.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h> 
#include <pybind11/chrono.h>
#include <sstream>
class SimpleTranscoder
{
private:
    std::unique_ptr<NvEncoderCuda> mEncoderCuda;
    std::unique_ptr<SimpleDecoder> mSimpleDecoder;
    std::unique_ptr<FFmpegDemuxer> mDemuxer;
    std::string mEncSource;
    std::string mMuxedDst;
    int numb = 0;
    std::string mCodec;

public:
    ~SimpleTranscoder();
    SimpleTranscoder();
    SimpleTranscoder(
        const std::string& encSource,
        const std::string& muxedDst,
        uint32_t gpuId,
        size_t cudaContext,
        size_t cudaStream,
        std::map<std::string, std::string> kwargs
    );
    void TranscodeWithMux();
    void SegmentedTranscodeWithMux(float start_ts, float end_ts);
    
};
