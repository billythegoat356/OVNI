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
#include "DecoderCommon.hpp"
#include "Logger.h"
#include "PyNvVideoCodecUtils.hpp"

#include <sstream>

DecoderCommon::DecoderCommon(const std::string& encSource,
            uint32_t gpuId,
            size_t cudaContext,
            size_t cudastream,
            bool useDeviceMemory,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool needScannedStreamMetadata,
            uint32_t decoderCacheSize,
            OutputColorType outputColorType,
            bool bWaitForSessionWarmUp) : mGPUId(gpuId),
            mNeedScannedStreamMetadata(needScannedStreamMetadata),
            mUseDeviceMemory(useDeviceMemory),
            mDecoderCache(decoderCacheSize),
            mWaitForSessionWarmUp(bWaitForSessionWarmUp),
            mOutputColorType(outputColorType)
{
    // Explicitly specify timescale to avoid ambiguity
    mDemuxer.reset(new FFmpegDemuxer(encSource.c_str(), 1000));
    mStreamMetadata = mDemuxer->GetStreamMetadata();
    mScanRequired = IsScanRequired();
    if (mScanRequired)
    {
        mNeedScannedStreamMetadata = true;
    }
    if (mNeedScannedStreamMetadata)
    {
        mStreamMetaThread = NvThread(std::thread(&FFmpegDemuxer::GetScannedStreamMetadata, mDemuxer.get(),
                                    std::ref(mScannedStreamMetadataPromise)));
        mScannedStreamMetadataFuture = mScannedStreamMetadataPromise.get_future();
    }
    CreateDecoder(FFmpeg2NvCodecId(mDemuxer->GetVideoCodec()), cudaContext, cudastream, maxWidth, maxHeight, mWaitForSessionWarmUp);
    
    auto key = std::make_tuple(mDemuxer->GetBitDepth(), FFmpeg2NvCodecId(mDemuxer->GetVideoCodec()),
        FFmpeg2NvChromaFormat(mDemuxer->GetChromaFormat()));
    mDecoderCache.PushDecoder(key, mDecoder.get());
    mSeekUtils.reset(new SeekUtils(mDemuxer.get(), mDecoder.get()));
}

DecoderCommon::~DecoderCommon()
{
    if (mNeedScannedStreamMetadata) {
        mStreamMetaThread.join();
    }
    mSeekUtils.reset();
    // This is fine and needed because the cache already has pointer associated with this
    // decoder. We will be cleaning it up in the loop below. Not doing this will result
    // in potentially double free.
    mDecoder.release();
    while (auto decoder = mDecoderCache.RemoveElement())
    {
        mDecoder.reset(*decoder);
        mDecoder.reset();
    }
    if (mReleasePrimaryContext)
    {
        ck(cuDevicePrimaryCtxRelease(mGPUId));
    }
}

void DecoderCommon::CreateDecoder(cudaVideoCodec codec,
            size_t cudaContext,
            size_t cudaStream,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool bWaitForSessionWarmUp)
{
    ck(cuInit(0));
    ValidateGpuId(mGPUId);
    if (cudaContext)
    {
        ValidateCUDAContext(mGPUId, reinterpret_cast<CUcontext>(cudaContext));
        mCudaContext = reinterpret_cast<CUcontext>(cudaContext);
    }
    else
    {
        ck(cuDevicePrimaryCtxRetain(&mCudaContext, mGPUId));
        mReleasePrimaryContext = true;
    }

    if(cudaStream)
    {
        ValidateCUDAStream(reinterpret_cast<CUstream>(cudaStream), mCudaContext);
        mCudaStream = reinterpret_cast<CUstream>(cudaStream);
    }

    mDecoder.reset(new NvDecoder(mGPUId, mCudaStream, mCudaContext, mUseDeviceMemory, codec, 
        false, // bLowLatency
        false, // bEnableAsyncAllocations
        maxWidth, 
        maxHeight, 
        mOutputColorType,
        false, // bDeviceFramePitched
        false, // extract_user_SEI_Message
        1000U,  // clkRate
        false, // force_zero_latency
        bWaitForSessionWarmUp)); // Now correctly in the 15th position
}

ScannedStreamMetadata DecoderCommon::GetScannedStreamMetadata()
{
    try
    {
        if (!mNeedScannedStreamMetadata)
        {
            std::stringstream ss;
            ss << "Invalid call to function " 
            << __func__ << ". Class DecoderCommon was created with 'mNeedScannedStreamMetadata' set to 'false'"; 
            throw std::runtime_error(ss.str());
        }
        if (mScannedStreamMetadataFuture.valid())
        {
            mScannedStreamMetadata = mScannedStreamMetadataFuture.get();
        }
        return mScannedStreamMetadata;
    }
    catch(const std::exception& e)
    {
        // Let python throw to user
        throw;
    }  
}

bool DecoderCommon::IsScanRequired()
{
    // Function to check if we want to scan the stream metadata
    return (mStreamMetadata.duration == 0.0 || mStreamMetadata.numFrames == 0);
}

void DecoderCommon::CopyFromScannedStreamMetadata()
{
    // Copy the common fields from the scanned metadata into the stream metadata.
    mStreamMetadata.width         = mScannedStreamMetadata.width;
    mStreamMetadata.height        = mScannedStreamMetadata.height;
    mStreamMetadata.numFrames     = mScannedStreamMetadata.numFrames;
    mStreamMetadata.averageFPS    = mScannedStreamMetadata.averageFPS;
    mStreamMetadata.duration      = mScannedStreamMetadata.duration;
    mStreamMetadata.bitrate       = mScannedStreamMetadata.bitrate;
    mStreamMetadata.codecName     = mScannedStreamMetadata.codecName;
}

StreamMetadata DecoderCommon::GetStreamMetadata()
{
    if (mScanRequired)
    {
        if (mScannedStreamMetadataFuture.valid())
        {
            mScannedStreamMetadata = mScannedStreamMetadataFuture.get();
        }
        CopyFromScannedStreamMetadata();
    }
    return mStreamMetadata;
}


void DecoderCommon::HandleDecoderInstanceRemoval(const std::optional<NvDecoder*>& decoder) {
    if (decoder) {
        auto currentDecoder = mDecoder.release();
        mDecoder.reset(*decoder); // Reset unique_ptr with the raw pointer
        LOG(DEBUG) << "Cache capacity exceeded. Removing LRU item\n";
        mDecoder.reset(currentDecoder);
    }
}

void DecoderCommon::ReconfigureDecoder(std::string encSource)
{
    if (mNeedScannedStreamMetadata) {
        mStreamMetaThread.join();
    }
    
    // Explicitly specify timescale here too
    mDemuxer.reset(new FFmpegDemuxer(encSource.c_str(), 1000));
    mStreamMetadata = mDemuxer->GetStreamMetadata();
    mScanRequired = IsScanRequired();
    if (mScanRequired)
    {
        mNeedScannedStreamMetadata = true;
    }

    std::string container = mDemuxer->GetContainerName();
    if (container == "flv" || 
        container == "matroska,webm")
    {
        mNeedScannedStreamMetadata = false;
    }

    if (mNeedScannedStreamMetadata)
    {
        mStreamMetaThread.join();
        mScannedStreamMetadataPromise = std::promise<ScannedStreamMetadata>();
        mStreamMetaThread = NvThread(std::thread(&FFmpegDemuxer::GetScannedStreamMetadata, mDemuxer.get(),
                                    std::ref(mScannedStreamMetadataPromise)));
        mScannedStreamMetadataFuture = mScannedStreamMetadataPromise.get_future();
    }
    auto width = mDemuxer->GetWidth();
    auto height = mDemuxer->GetHeight();
    auto key = std::make_tuple(mDemuxer->GetBitDepth(), FFmpeg2NvCodecId(mDemuxer->GetVideoCodec()),
    FFmpeg2NvChromaFormat(mDemuxer->GetChromaFormat()));
    std::optional<NvDecoder*> decoder = mDecoderCache.GetDecoder(key);
    if (decoder.has_value() && (*decoder) != nullptr)
    {
        if (width > (*decoder)->GetMaxWidth() || height > (*decoder)->GetMaxHeight())
        {
            LOG(DEBUG) << "Cached decoder found with lower widthxheight configuration. Createing a new decoder\n";
            auto maxWidth = std::max(width, (*decoder)->GetMaxWidth());
            auto maxHeight = std::max(height, (*decoder)->GetMaxHeight());
            mDecoder.release();
            mDecoder.reset(new NvDecoder(mGPUId, mCudaStream, mCudaContext, mUseDeviceMemory, FFmpeg2NvCodecId(mDemuxer->GetVideoCodec()),
                false, false, maxWidth, maxHeight, mOutputColorType));
            HandleDecoderInstanceRemoval(mDecoderCache.PushDecoder(key, mDecoder.get()));
        }
        else
        {
            LOG(DEBUG) << "Reusing cached decoder instance\n";
            // Release current decoder from ownership. This is not a problem because
            // we already have a pointer in the cache. So there will be proper
            // cleanup in case it is a LRU decoder (where HandleDecoderInstanceRemoval
            // handles the cleanup) or if it is not LRU then session destruction will
            // do a proper cleanup.
            mDecoder.release();
            mDecoder.reset(*decoder);
            Dim dim = { width , height };
            mDecoder->setReconfigParams(dim);
        }
    }
    else
    {
        LOG(DEBUG) << "Cached decoder instance not found. Creating a new decoder\n";
        mDecoder.release();
        mDecoder.reset(new NvDecoder(mGPUId, mCudaStream, mCudaContext, mUseDeviceMemory, FFmpeg2NvCodecId(mDemuxer->GetVideoCodec()),
            false, false, width, height, mOutputColorType));
        HandleDecoderInstanceRemoval(mDecoderCache.PushDecoder(key, mDecoder.get()));
    }
    
}

SeekUtils* DecoderCommon::GetPtrToSeekUtils()
{
    return mSeekUtils.get();
}

void DecoderCommon::WaitForStreamMetadata() {
    if (mNeedScannedStreamMetadata) {
        mStreamMetaThread.join();  // NvThread only has join()
        if (mScannedStreamMetadataFuture.valid()) {
            try {
                mScannedStreamMetadata = mScannedStreamMetadataFuture.get();
            } catch (const std::exception& e) {
                LOG(ERROR) << "Stream metadata scanning failed: " << e.what();
                throw;
            }
        }
    }
}
