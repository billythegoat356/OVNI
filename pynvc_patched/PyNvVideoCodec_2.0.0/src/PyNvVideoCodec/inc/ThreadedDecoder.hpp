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

#include <atomic>
#include <future>
#include <memory>
#include <vector>

#include "DecoderCommon.hpp"
#include "PyCAIMemoryView.hpp"
#include "FFmpegDemuxer.h"
#include "NvCodecUtils.h"
#include "NvDecoder/NvDecoder.h"
#include "SPSCBuffer.hpp"


template<typename T>
static void RunDecoder(FFmpegDemuxer* demuxer, NvDecoder* decoder, SPSCBuffer<T>& decodedFrames,
                        std::atomic<bool>& decodeStopFlag);


class ThreadedDecoder {
private:
    std::unique_ptr<DecoderCommon> mDecoderCommon;
    NvThread mDecoderThread;
    SPSCBuffer<DecodedFrame> mDecodedFrames;
    std::atomic<bool> mDecodeStopFlag {false};
    uint32_t mPrevBatchSize = 0;
    bool endCalled = false;
public:
    ThreadedDecoder(){}
    ~ThreadedDecoder();
    ThreadedDecoder(const std::string& encSource,
            uint32_t bufferSize,
            uint32_t gpuId = 0,
            size_t cudaContext = 0,
            size_t cudaStream = 0,
            bool useDevicememory = 0,
            uint32_t maxWidth = 0,
            uint32_t maxHeight = 0,
            bool needScannedStreamMetadata = 0,
            uint32_t decoderCacheSize = 4,
            OutputColorType outputColorType = OutputColorType::NATIVE);
    void Initialize();
    std::vector<DecodedFrame> GetBatchFrames(size_t batchSize);
    StreamMetadata GetStreamMetadata();
    ScannedStreamMetadata GetScannedStreamMetadata();
    void ReconfigureDecoder(std::string newSource);
    void End();
};
