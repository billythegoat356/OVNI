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

#include "SimpleDecoder.hpp"
SimpleDecoder::SimpleDecoder(const std::string& encSource,
            uint32_t gpuId,
            size_t cudaContext,
            size_t cudaStream,
            bool useDeviceMemory,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool needScannedStreamMetadata,
            uint32_t decoderCacheSize,
            OutputColorType outputColorType,
            bool bWaitForSessionWarmUp) : mEncSource(encSource)
{
    mDecoderCommon.reset(new DecoderCommon(encSource, gpuId, cudaContext, cudaStream, useDeviceMemory, maxWidth,
                        maxHeight, needScannedStreamMetadata, decoderCacheSize, outputColorType, bWaitForSessionWarmUp));
    if (!mDecoderCommon->GetDemuxer()->IsSeekable())
    {
        PYNVVC_THROW_ERROR_UNSUPPORTED("This stream is not seekable.", CUDA_ERROR_NOT_SUPPORTED);
    }
}

SimpleDecoder::~SimpleDecoder()
{
    mDecoderCommon->GetPtrToSeekUtils()->ClearState();
}

std::vector<DecodedFrame> SimpleDecoder::GetBatchFrames(size_t batchSize)
{
    return mDecoderCommon->GetPtrToSeekUtils()->GetFramesByBatch(batchSize);
}

std::variant<DecodedFrame, std::vector<DecodedFrame>> SimpleDecoder::operator[](std::variant<uint32_t, std::vector<uint32_t>> indices)
{
    std::vector<DecodedFrame> decoded_frames;
    std::vector<uint32_t> _indices;
    uint32_t smallestprevIdx = 0;
    if (std::holds_alternative<uint32_t>(indices))
    {
        uint32_t targetIdx = std::get<uint32_t>(indices);  
        _indices.push_back(targetIdx);
    }
    else if (std::holds_alternative<std::vector<uint32_t>>(indices))
    {
        _indices = std::get<std::vector<uint32_t>>(indices);
    }
    ResetDecoderIfRequired(indices);
    decoded_frames = mDecoderCommon->GetPtrToSeekUtils()->GetFramesByIdxList(_indices);
    if (std::holds_alternative<uint32_t>(indices))
    {
        return decoded_frames[0];
    }
    else if (std::holds_alternative<std::vector<uint32_t>>(indices))
    {
        return decoded_frames;
    }
    return decoded_frames;
}

std::vector<DecodedFrame> SimpleDecoder::GetBatchFramesByIndex(std::vector<uint32_t> indices)
{
    ResetDecoderIfRequired(indices);
    std::vector<DecodedFrame> decoded_frames = mDecoderCommon->GetPtrToSeekUtils()->GetFramesByIdxList(indices);
    return decoded_frames;
}


ScannedStreamMetadata SimpleDecoder::GetScannedStreamMetadata()
{
    return mDecoderCommon->GetScannedStreamMetadata();  
}

StreamMetadata SimpleDecoder::GetStreamMetadata()
{
    return mDecoderCommon->GetStreamMetadata();
}

void SimpleDecoder::SeekToIndex(uint32_t index)
{
    ResetDecoderIfRequired(index);
    mDecoderCommon->GetPtrToSeekUtils()->SeekToIndex(index);
}

uint32_t SimpleDecoder::GetIndexFromTimeInSeconds(float timeInSeconds)
{
    return mDecoderCommon->GetPtrToSeekUtils()->GetIndexFromTimeStamp(timeInSeconds);
}


void SimpleDecoder::ReconfigureDecoder(std::string newSource)
{
    mDecoderCommon->GetPtrToSeekUtils()->ClearState(true);
    mDecoderCommon->ReconfigureDecoder(newSource);
    mDecoderCommon->GetPtrToSeekUtils()->Initialize(mDecoderCommon->GetDemuxer(), mDecoderCommon->GetDecoder());
    mEncSource = newSource;
}

void SimpleDecoder::ResetDecoderIfRequired(std::variant<uint32_t, std::vector<uint32_t>> indices)
{

    std::vector<uint32_t> _indices;

    if (std::holds_alternative<uint32_t>(indices))
    {
        uint32_t targetIdx = std::get<uint32_t>(indices);
        _indices.push_back(targetIdx);
    }
    else if (std::holds_alternative<std::vector<uint32_t>>(indices))
    {
        _indices = std::get<std::vector<uint32_t>>(indices);
    }

    std::vector<uint32_t> sorted_indices(_indices);
    std::sort(sorted_indices.begin(), sorted_indices.end());
    uint32_t idx = sorted_indices[0];
    if (mDecoderCommon->GetPtrToSeekUtils()->IsSeekBackwards(idx))
    {
        mDecoderCommon->GetPtrToSeekUtils()->setEOS(true);
        ReconfigureDecoder(mEncSource);
    }
}

DecoderCommon* SimpleDecoder::GetDecoderCommonInstance()
{
    return mDecoderCommon.get();
}

int64_t SimpleDecoder::GetSessionInitTime()
{
    return mDecoderCommon->GetDecoder()->GetSessionInitTime();
}

void SimpleDecoder::SetSessionCount(uint32_t count)
{
    NvDecoder::SetSessionCount(count);
}