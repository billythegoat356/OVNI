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
#include "SeekUtils.hpp"
#include "PyNvVideoCodecUtils.hpp"

#include <functional>
#include <sstream>
#include <typeinfo>

SeekUtils::SeekUtils(FFmpegDemuxer* demuxer, NvDecoder* decoder) :
    mPreviousTargetIndex(-1),
    mFramesDecodedTillNow(0),
    mFrameSizeInBytes(0),
    mTargetFramePTS(0),
    mbDiscontinuityFlag(false),
    mbEOSreached(false),
    bIsSeekDirectionBackwards(false),
    bSeekToIndexSet(false)
{
    Initialize(demuxer, decoder);
}

void SeekUtils::Initialize(FFmpegDemuxer* demuxer, NvDecoder* decoder)
{
    mDemuxer = demuxer;
    mDecoder = decoder;
    mVideoStreamPtr = mDemuxer->GetVideoStream();
}


void SeekUtils::ClearState(bool bForceEOS)
{
    // Unlock previous frames
    UnlockFrames();
    mTargetFrames.clear();
    mPendingFrames.clear();
    mPreviouslyDecodedFramesPTS.clear();
    mPreviousTargetIndex = -1;
    mFramesDecodedTillNow = 0;
    mFrameSizeInBytes = 0;
    mTargetFramePTS = 0;
    mbDiscontinuityFlag = false;
    bSeekToIndexSet = false;
    mDemuxer->Seek(0);
    mDecoder->SetWaitForSessionWarmUp(true);
    mDecoder->GetSessionPerf().SetSessionInitCounter(0);
    
    if (mbEOSreached || bForceEOS)
    {
        PacketData emptyPacket = PacketData();
        int  numDecodedFrames = mDecoder->Decode((uint8_t*)emptyPacket.bsl_data,
            emptyPacket.bsl);
        for (int i = 0; i < numDecodedFrames; i++)
        {
            GetFrame(false);
        }
        mbEOSreached = false;
    }
    
    //experimental code
}

std::vector<DecodedFrame> SeekUtils::GetFramesByIdxList(std::vector<uint32_t> indices)
{
    NVTX_SCOPED_RANGE("py::GetNumDecodedFrame")
    py::gil_scoped_release release;
    UnlockFrames();
    mTargetFrames.clear();
    int nVideoBytes = 0;
    uint8_t* pVideo = NULL;
    
    for (const uint32_t& currentTargetIndex : indices)
    {
        //check with previous target index if IDR lies in between previous and current target
        std::pair<bool, int64_t> result = ShouldSeek(mPreviousTargetIndex, currentTargetIndex);

        if(result.second == static_cast<int>(SeekStatus::INVALID_INDEX_ENTRY))
        {
            continue;
        }
        
        if (result.first)//we need to seek only if there is change in GOP
        {
            int64_t timestamp = GetKeyNearestKeyFrameIndexForTarget(mVideoStreamPtr, currentTargetIndex);
            mDemuxer->Seek(currentTargetIndex);
            mFramesDecodedTillNow = timestamp;
            mbDiscontinuityFlag = true;
            mPendingFrames.clear();
            mPreviouslyDecodedFramesPTS.clear();
        }
        else
        {
            mbDiscontinuityFlag = false;
        }
        
        bool bTargetFrameFound = false;
        bool bSeqCallbackTriggered = false;
        bool bReset = false;
        PacketData packetdata = PacketData();
        int targetValue = currentTargetIndex - mFramesDecodedTillNow;
        
        if (targetValue < 0)//we reached EOS earlier hence searching will be from pending frames queue only
        { 
            int actual_idx = (targetValue) + mPendingFrames.size();
            if (actual_idx  <= mPendingFrames.size())
            {
                DecodedFrame pendingFrame = mPendingFrames[actual_idx];
                mTargetFrames.push_back(pendingFrame);
                bTargetFrameFound = true;
                targetValue = 0;
            }
        }
        else
        {
            mPendingFrames.clear();
        }

        bool isKeyFrame = false;

        while (!bTargetFrameFound)//loop until the target frame is found
        {
            
            bool bRes = mDemuxer->Demux((uint8_t**)&packetdata.bsl_data,
                (int*)&packetdata.bsl,
                packetdata.pts,
                packetdata.dts,
                packetdata.duration,
                packetdata.pos,
                isKeyFrame
            );
            if (!bRes)
            {
                memset(&packetdata, 0, sizeof(PacketData));
            }
            packetdata.key = isKeyFrame ? 1 : 0;

            if (isKeyFrame && mbDiscontinuityFlag)
            {
                mTargetFramePTS = packetdata.pts;
            }
            if (mbDiscontinuityFlag)//if we seek to a new GOP, flush out all pending frames from previous GOP since seeking is now based on index
            {
                PacketData emptyPacket = PacketData();
                mDecoder->setSeekPTS(0);
                int  numDecodedFrames = mDecoder->Decode((uint8_t*)emptyPacket.bsl_data,
                    emptyPacket.bsl, CUVID_PKT_DISCONTINUITY, emptyPacket.pts);
                for (int i = 0; i < numDecodedFrames; i++)
                {
                    GetFrame(false);
                }
                mPendingFrames.clear();
                mbDiscontinuityFlag = false;
            }
            mDecoder->setSeekPTS(0);
            int  numDecodedFrames = 0;
            CUvideopacketflags flag = CUVID_PKT_TIMESTAMP;
            if (packetdata.bsl == 0 && packetdata.bsl_data == 0)//check if we reached EOS
            {
                flag = CUVID_PKT_ENDOFSTREAM;
                mbEOSreached = true;
            }
            numDecodedFrames =  mDecoder->Decode((uint8_t*)packetdata.bsl_data,
                packetdata.bsl, flag, packetdata.pts);
            
            int mCurrentCountOfDecodedFrames = 0;
            mFramesDecodedTillNow += numDecodedFrames;
            for (int i = 0; i < numDecodedFrames; i++)//loop through all the frames to find the target frame
            {
                DecodedFrame decodedframe = GetFrame(true);
                mCurrentCountOfDecodedFrames++;
                if (std::find(mPreviouslyDecodedFramesPTS.begin(), mPreviouslyDecodedFramesPTS.end(), decodedframe.timestamp) != mPreviouslyDecodedFramesPTS.end())//this is done to ensure that frames from earlier GOP are 
                {
                    UnlockFrame(decodedframe);
                    mFramesDecodedTillNow--;
                    continue;
                }
                else
                {
                    mPreviouslyDecodedFramesPTS.push_back(decodedframe.timestamp);
                }
                if (decodedframe.timestamp < mTargetFramePTS)
                {
                    UnlockFrame(decodedframe);
                    mFramesDecodedTillNow--;
                    continue;
                }
                if (targetValue == 0 )
                {
                    mTargetFrames.push_back(decodedframe);//we found the frame we are seeking
                    bTargetFrameFound = true;
                    if (mCurrentCountOfDecodedFrames != 0)//move the pending frames to another queue, if we hit EOS, target frame would be searched from this queue firststart.
                    {
                        for (int j = 0; j < (numDecodedFrames - mCurrentCountOfDecodedFrames); j++)
                        {
                            mPendingFrames.push_back(GetFrame(true));
                        }
                    }
                    break;
                }
                
                else
                {
                    UnlockFrame(decodedframe);
                    targetValue--;
                }
            }
            if (packetdata.bsl == 0 && packetdata.bsl_data == 0)
            {
                break;
            }
        }
        if (bTargetFrameFound)
        {
            mPreviousTargetIndex = currentTargetIndex;
        }
        bTargetFrameFound = false;
    }
    
    py::gil_scoped_acquire acquire;
    
    return mTargetFrames;
}


DecodedFrame SeekUtils::GetFrame(bool bLockFrame)
{
    int64_t tupTimestamp = 0;
    CUdeviceptr tupData = 0;
    SEI_MESSAGE seimsg;
    CUevent event = nullptr;
    if (bLockFrame)
    {
       tupData = (CUdeviceptr)mDecoder->GetLockedFrame(&tupTimestamp, &seimsg, &event);
    }
    else
    {
        tupData = (CUdeviceptr)mDecoder->GetFrame(&tupTimestamp, &seimsg, &event);
    }

    return GetCAIMemoryViewAndDLPack(mDecoder, std::make_tuple(tupData, tupTimestamp, seimsg, event));
}

int SeekUtils::GetKeyNearestKeyFrameIndexForTarget(AVStream* stream,
    int64_t idx)
{
    int targetpts = 0;
    std::string container = mDemuxer->GetContainerFormat();
    if (container == "flv"
        || container == "matroska,webm")
    {
        targetpts = mDemuxer->FrameToPts(stream, idx);
        int KeyFrameIndex =
            av_index_search_timestamp(stream, targetpts, AVSEEK_FLAG_BACKWARD);
        const AVIndexEntry* entry = avformat_index_get_entry(stream, KeyFrameIndex);
        int currentKeyFrameIndex = mDemuxer->dts_to_frame_number(entry->timestamp);
        return currentKeyFrameIndex;
    }
    else
    {
        const AVIndexEntry* entry = avformat_index_get_entry(stream, idx);
        if (entry == NULL)
        {
            LOG(WARNING) << "Entry is null for index " << idx << "\n";
            return static_cast<int>(SeekStatus::INVALID_INDEX_ENTRY);
        }
        targetpts = entry->timestamp;
        int currentKeyFrameIndex =
            av_index_search_timestamp(stream, targetpts, AVSEEK_FLAG_BACKWARD);
        return currentKeyFrameIndex;
    }
    
    
}
uint32_t SeekUtils::GetIndexFromTimeStamp(double timeStamp)
{
    int64_t pts = mDemuxer->TsFromTime(timeStamp);
    int64_t frameIndex = mDemuxer->dts_to_frame_number(pts);
    return frameIndex;
}

void SeekUtils::UnlockFrames()
{
    for (DecodedFrame frame : mTargetFrames)
    {
        mDecoder->UnlockFrame((uint8_t*)frame.extBuf->data());
    }
}

void SeekUtils::UnlockFrame(DecodedFrame& decframe)
{
    uint8_t* dataptr = (uint8_t*)decframe.extBuf->data();
    if (dataptr != NULL)
    {
        mDecoder->UnlockFrame(dataptr);
    }
}

std::pair<bool, int64_t> SeekUtils::ShouldSeek(int64_t previous_target, int64_t current_target)
{
    int rightindex = GetKeyNearestKeyFrameIndexForTarget(mVideoStreamPtr, current_target);
    if (previous_target == -1)
    {
        return std::make_pair(true, rightindex);
    }
    int leftindex = GetKeyNearestKeyFrameIndexForTarget(mVideoStreamPtr, previous_target);

    if (leftindex == static_cast<int>(SeekStatus::INVALID_INDEX_ENTRY) || rightindex == static_cast<int>(SeekStatus::INVALID_INDEX_ENTRY))
    {
        return std::make_pair(false, static_cast<int>(SeekStatus::INVALID_INDEX_ENTRY));
    }

    if (leftindex == rightindex)
    {
        return std::make_pair(false, rightindex);
    }
    else if (rightindex - previous_target < 4)
    {
        return std::make_pair(false, rightindex);
    }
    else
    {
        return std::make_pair(true, rightindex);
    }
    
}


int64_t SeekUtils::TsFromTime(double ts_sec)
{
    /* Internal timestamp representation is integer, so multiply to AV_TIME_BASE
     * and switch to fixed point precision arithmetics; */
    auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);

    // Rescale the timestamp to value represented in stream base units;
    AVRational factor;
    factor.num = 1;
    factor.den = AV_TIME_BASE;

    return av_rescale_q(ts_tbu, factor, mVideoStreamPtr->time_base);
}

void SeekUtils::SeekToIndex(uint32_t newTargetIdx)
{
    mPreviousTargetIndex = newTargetIdx;
    bSeekToIndexSet = true;
}

std::vector<DecodedFrame> SeekUtils::GetFramesByBatch(uint32_t batchsize)
{
    std::vector<uint32_t> idxs;
    uint32_t start = mPreviousTargetIndex + 1;
    uint32_t end = mPreviousTargetIndex + batchsize + 1;
    StreamMetadata metadata = mDemuxer->GetStreamMetadata();
    uint32_t numFrames = metadata.numFrames;
    
    if (bSeekToIndexSet)
    {
        start = mPreviousTargetIndex;
        end = mPreviousTargetIndex + batchsize;
        bSeekToIndexSet = false;
    }
    for (; start < end; start++)
    {
        if((0 <= start) && (start < numFrames))
            idxs.push_back(start);
        else
            LOG(WARNING) << "INVALID INDEX " << start << "\n";
        
    }
    return GetFramesByIdxList(idxs);
}

bool SeekUtils::IsSeekBackwards(int64_t newTargetIdx)
{
    if (newTargetIdx <= mPreviousTargetIndex && mPreviousTargetIndex != -1)
    {
        bIsSeekDirectionBackwards = true;
        return true;
    }
    bIsSeekDirectionBackwards = false;
    return false;
}

bool SeekUtils::IsEOSReached()
{
    return mbEOSreached;
}

std::vector<DecodedFrame> SeekUtils::GetPendingFrames()
{
    return mPendingFrames;
}