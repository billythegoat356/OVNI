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

class SimpleDecoder {
private:
    std::unique_ptr<DecoderCommon> mDecoderCommon;
    std::string mEncSource;
    
public:
    SimpleDecoder(){}
    ~SimpleDecoder();
    SimpleDecoder(const std::string& encSource,
            uint32_t gpuId = 0,
            size_t cudaContext = 0,
            size_t cudaStream = 0,
            bool useDevicememory = 0,
            uint32_t maxWidth = 0,
            uint32_t maxHeight = 0,
            bool needScannedStreamMetadata = 0,
            uint32_t decoderCacheSize = 4,
            OutputColorType outputColorType = OutputColorType::NATIVE,
            bool bWaitForSessionWarmUp = false);
    std::vector<DecodedFrame> GetBatchFrames(size_t batchSize);
    std::variant<DecodedFrame, std::vector<DecodedFrame>> operator[](std::variant<uint32_t, std::vector<uint32_t>> indices);
    std::vector<DecodedFrame> GetBatchFramesByIndex(std::vector<uint32_t> indices);
    StreamMetadata GetStreamMetadata();
    ScannedStreamMetadata GetScannedStreamMetadata();
    void SeekToIndex(uint32_t index);
    uint32_t GetIndexFromTimeInSeconds(float timeInSeconds);
    void ReconfigureDecoder(std::string newSource);
    DecoderCommon* GetDecoderCommonInstance();
    int64_t GetSessionInitTime();
    static void SetSessionCount(uint32_t count);
private:
    void ResetDecoderIfRequired(std::variant<uint32_t, std::vector<uint32_t>> indices);
};
