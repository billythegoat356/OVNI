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

#include "NvDemuxer.hpp"
class PyNvDemuxer {

protected:
    std::unique_ptr<NvDemuxer> demuxer;

public:
    explicit PyNvDemuxer(const std::string&);

    explicit PyNvDemuxer(std::function<int(py::bytearray)>);

    uint32_t Height() {return demuxer->GetHeight();}

    uint32_t Width() {return demuxer->GetWidth();}

    uint32_t BitDepth() { return demuxer->GetBitDepth(); }

    cudaVideoChromaFormat ChromaFormat() { return demuxer->GetChromaFormat(); }

    uint32_t GetFrameSize() { return demuxer->GetFrameSize(); }

    ColorSpace GetColorSpace() const { return demuxer->GetColorSpace(); }

    ColorRange GetColorRange() const { return demuxer->GetColorRange(); }

    double GetFrameRate() const { return demuxer->GetFrameRate(); };


#ifndef DEMUX_ONLY
    cudaVideoCodec GetNvCodecId() {
        return demuxer->GetNvCodecId();
    }
#endif

    shared_ptr<PacketData> Demux();
  
    shared_ptr<PacketData> Seek(uint64_t &timestamp);

    bool isEndOfStream() { return demuxer->isEOF(); }

    int isSeekDone(int64_t decodedFramePTS, int64_t frameIndex) { return demuxer->IsSeekDone(decodedFramePTS, frameIndex); }

    uint64_t TimestampFromFrame(uint32_t index) { return demuxer->TimestampFromFrame(index); }

};
