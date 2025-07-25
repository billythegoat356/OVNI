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

#include "NvCodecUtils.h"
#include "NvDecoder/NvDecoder.h"
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h> 
#include <pybind11/chrono.h>
#include "FFmpegDemuxer.h"
#include "PyCAIMemoryView.hpp"

enum class SeekStatus {
    INVALID_INDEX_ENTRY = -2
};

class SeekUtils {
    FFmpegDemuxer* mDemuxer;
    NvDecoder* mDecoder;
    std::vector<DecodedFrame> mTargetFrames;
    std::vector<DecodedFrame> mPendingFrames;
    int32_t mPreviousTargetIndex;
    uint32_t mFrameSizeInBytes;
    uint32_t mFramesDecodedTillNow;
    bool mbDiscontinuityFlag;
    int32_t mTargetFramePTS;
    std::vector<int64_t> mPreviouslyDecodedFramesPTS;
    AVStream* mVideoStreamPtr;
    bool mbEOSreached;
    bool bIsSeekDirectionBackwards;
    bool bSeekToIndexSet;
public:
    void setEOS(bool newVal) { mbEOSreached = true; }
    SeekUtils(FFmpegDemuxer* demuxer,NvDecoder* decoder);
    std::vector<DecodedFrame> GetFramesByIdxList(std::vector<uint32_t> indices);
    std::vector<DecodedFrame> GetFramesByBatch(uint32_t batchsize);
    int GetKeyNearestKeyFrameIndexForTarget(AVStream* stream,int64_t idx);
    uint32_t GetIndexFromTimeStamp(double timeStamp);
    DecodedFrame GetFrame(bool bLockFrame);
    std::vector<DecodedFrame> GetPendingFrames();
    void UnlockFrames();
    void UnlockFrame(DecodedFrame& decframe);
    std::pair<bool, int64_t> ShouldSeek(int64_t previous_target, int64_t current_target);
    int64_t TsFromTime(double ts_sec);
    void SeekToIndex(uint32_t index);
    void ClearState(bool bForceEOS = false);
    void Initialize(FFmpegDemuxer* demuxer, NvDecoder* decoder);
    bool IsSeekBackwards(int64_t currentTarget);
    bool IsEOSReached();
};
