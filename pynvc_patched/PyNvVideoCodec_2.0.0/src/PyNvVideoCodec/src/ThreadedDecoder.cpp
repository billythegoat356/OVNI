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
#include "ThreadedDecoder.hpp"
#include "PyNvVideoCodecUtils.hpp"

ThreadedDecoder::ThreadedDecoder(const std::string& encSource,
            uint32_t bufferSize,
            uint32_t gpuId,
            size_t cudaContext,
            size_t cudaStream,
            bool useDeviceMemory,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool needScannedStreamMetadata,
            uint32_t decoderCacheSize,
            OutputColorType outputColorType) : mDecodedFrames(bufferSize)
{
    mDecoderCommon.reset(new DecoderCommon(encSource, gpuId, cudaContext, cudaStream, useDeviceMemory, maxWidth,
                        maxHeight, needScannedStreamMetadata, decoderCacheSize, outputColorType));
}

ThreadedDecoder::~ThreadedDecoder()
{
    if (!endCalled)
    {
        End();
    }
}

void ThreadedDecoder::Initialize()
{
    endCalled = false;
    mPrevBatchSize = 0;
    mDecodeStopFlag.store(false);
    mDecoderThread = NvThread(std::thread(RunDecoder<DecodedFrame>, mDecoderCommon->GetDemuxer(), mDecoderCommon->GetDecoder(),
                    std::ref(mDecodedFrames), std::ref(mDecodeStopFlag)));
}

template <typename T>
static void RunDecoder(FFmpegDemuxer* demuxer, NvDecoder* decoder, SPSCBuffer<T>& decodedFrames,
            std::atomic<bool>& decodeStopFlag)
{
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL;
    int64_t pts = 0;
    int64_t dts = 0;
    uint64_t duration = 0;
    uint64_t pos = 0;
    bool keyFrame = false;

    do {
        demuxer->Demux(&pVideo, &nVideoBytes, pts, dts, duration, pos, keyFrame);
        nFrameReturned = decoder->Decode(pVideo, nVideoBytes, 0, pts);
        for (int i = 0; (i < nFrameReturned) && (!decodeStopFlag.load()); i++) {
            int64_t timestamp = 0;
            SEI_MESSAGE seimsg;
            CUevent event = nullptr;
            auto frame_ptr = reinterpret_cast<CUdeviceptr>(decoder->GetLockedFrame(&timestamp, &seimsg, &event));
            auto tup = std::make_tuple(frame_ptr, timestamp, seimsg, event);
            DecodedFrame frame = GetCAIMemoryViewAndDLPack(decoder, tup);
            decodedFrames.PushEntry(frame);
        }
        nFrame += nFrameReturned;
    } while (nVideoBytes && !decodeStopFlag.load());
    decodedFrames.PushDone();
    // Send empty packet to decoder to simulate decode complete
    if (nVideoBytes != 0)
    {

        PacketData emptyPacket = PacketData();
        decoder->Decode((uint8_t*)emptyPacket.bsl_data, emptyPacket.bsl);
    }
}

void ThreadedDecoder::End()
{
    // Will stop any further decode
    mDecodeStopFlag.store(true);
    // This is not a clean solution. The PushEntry call tries to push a "locked frame"
    // but goes to sleep because there is no space in the buffer. We will need to wake that
    // thread to shut it down. So we call GetBatchFrames with size 1. This will wake the thread
    // and push the locked frame. This and abvoe effectively stops the Push thread.
    GetBatchFrames(1);
    mDecoderThread.join();
    
    // Drain the buffer so as to unlock the frames
    GetBatchFrames(0);
    mDecoderCommon->UnlockLockedFrames(mPrevBatchSize);

    // reset state
    mPrevBatchSize = 0;
    mDecodeStopFlag.store(false);
    mDecodedFrames.Clear();
    endCalled = true;
}

std::vector<DecodedFrame> ThreadedDecoder::GetBatchFrames(size_t batchSize)
{
    // unlock previously locked frames if any
    py::gil_scoped_release release;
    mDecoderCommon->UnlockLockedFrames(mPrevBatchSize);
    auto frames = mDecodedFrames.PopEntries(batchSize);
    mPrevBatchSize = frames.size();
    py::gil_scoped_acquire acquire;
    return frames;
}

ScannedStreamMetadata ThreadedDecoder::GetScannedStreamMetadata()
{
    return mDecoderCommon->GetScannedStreamMetadata();
}

StreamMetadata ThreadedDecoder::GetStreamMetadata()
{
    return mDecoderCommon->GetStreamMetadata();
}

void ThreadedDecoder::ReconfigureDecoder(std::string newSource)
{
    // Gracefully shutdown the current decoder thread.
    End();
    mDecoderCommon->ReconfigureDecoder(newSource);
    // Intialize the decoder again
    Initialize();
}