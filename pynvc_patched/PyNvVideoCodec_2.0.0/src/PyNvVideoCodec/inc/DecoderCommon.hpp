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

#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "FFmpegDemuxer.h"
#include "DecoderCache.hpp"
#include "NvCodecUtils.h"
#include "NvDecoder/NvDecoder.h"
#include "PyCAIMemoryView.hpp"
#include "SeekUtils.hpp"

class DecoderCommon {
private:
    // Demuxer related
    std::unique_ptr<FFmpegDemuxer> mDemuxer;
    std::promise<ScannedStreamMetadata> mScannedStreamMetadataPromise;
    std::future<ScannedStreamMetadata> mScannedStreamMetadataFuture;
    StreamMetadata mStreamMetadata;
    ScannedStreamMetadata mScannedStreamMetadata;
    NvThread mStreamMetaThread;
    bool mNeedScannedStreamMetadata = false;
    bool mScanRequired = false;
    bool mWaitForSessionWarmUp = false;
    OutputColorType mOutputColorType;
    // Decoder related
    std::unique_ptr<NvDecoder> mDecoder;
    int32_t mGPUId = 0;
    CUcontext mCudaContext = nullptr;
    CUstream mCudaStream = nullptr;
    bool mReleasePrimaryContext = false;
    bool mUseDeviceMemory = false;
    std::unique_ptr<SeekUtils> mSeekUtils;

    using Keys = std::tuple<int, cudaVideoCodec, cudaVideoChromaFormat>;
    DecoderCache<Keys, NvDecoder*> mDecoderCache;
private:
    void CreateDecoder(cudaVideoCodec codec,
            size_t cudaContext,
            size_t cudaStream,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool bWaitForSessionWarmUp = false);
    void CopyFromScannedStreamMetadata();
    bool IsScanRequired();
    
public:
    DecoderCommon(){}
    ~DecoderCommon();

    DecoderCommon(const std::string& encSource,
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
    StreamMetadata GetStreamMetadata();
    ScannedStreamMetadata GetScannedStreamMetadata();
    NvDecoder* GetDecoder() { return mDecoder.get(); }
    FFmpegDemuxer* GetDemuxer() { return mDemuxer.get(); }
    void UnlockLockedFrames(uint32_t size) { mDecoder->UnlockLockedFrames(size); }
    void HandleDecoderInstanceRemoval(const std::optional<NvDecoder*>& decoder);
    void ReconfigureDecoder(std::string newSource);
    SeekUtils* GetPtrToSeekUtils();
    void WaitForStreamMetadata();
    CUcontext GetCUContext() { return mCudaContext; }
    CUstream GetCUStream() { return mCudaStream; }
};
