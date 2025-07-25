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

#include "FFmpegDemuxer.h"
#include <chrono>
#ifndef DEMUX_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <iostream>
#include <sstream>

extern "C" {
#include <libavutil/frame.h>
}

namespace py = pybind11;


enum ColorSpace {
    BT_601 = 0,
    BT_709 = 1,
    UNSPEC = 2,
};

enum ColorRange {
    MPEG = 0, /* Narrow range.*/
    JPEG = 1, /* Full range. */
    UDEF = 2,
};


class NvDemuxer {

protected:
    std::unique_ptr<FFmpegDemuxer> demuxer;
    std::shared_ptr <PacketData> currentPacket;
    std::unique_ptr <FFmpegDemuxer::PyByteArrayProvider> dataProviderForByteArray;
    bool isEOSReached;

public:
    explicit NvDemuxer(const std::string&);
    
    explicit NvDemuxer( std::function<int(py::bytearray)>);

    uint32_t GetWidth()             {return demuxer->GetWidth();}

    uint32_t GetHeight()            {return demuxer->GetHeight();}

    uint32_t GetFrameSize()         { return demuxer->GetFrameSize(); }

    ColorSpace GetColorSpace() const;

    ColorRange GetColorRange() const;

    double GetFrameRate()           { return demuxer->GetFrameRate(); }

#ifndef DEMUX_ONLY
    cudaVideoCodec GetNvCodecId() {
        return FFmpeg2NvCodecId(demuxer->GetVideoCodec());
    }
#endif

    cudaVideoChromaFormat GetChromaFormat() {
        return FFmpeg2NvChromaFormat(demuxer->GetChromaFormat());
    }

    int GetBitDepth() {
        return demuxer->GetBitDepth();
    }

    shared_ptr<PacketData> Demux();

    shared_ptr<PacketData> Seek(uint64_t timestamp);

    int IsSeekDone(int64_t decodedFramePTS, int64_t frameIndex);

    bool isEOF() { return isEOSReached; }

    uint64_t TimestampFromFrame(uint32_t index) { return demuxer->TsFromFrameNumber(index); }

};
