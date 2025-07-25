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

#include "NvDecoder/NvDecoder.h"
#include "NvCodecUtils.h"
#include "PyCAIMemoryView.hpp"
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h> 
#include <pybind11/chrono.h>
#include <iostream>
#include <sstream>

class PyNvDecoder {
private:
    bool mReleasePrimaryContext = false;
    CUcontext cuContext = NULL;
    CUstream cuStream = NULL;
    int mGPUId = 0;

protected:
    std::unique_ptr<NvDecoder> decoder;

public:
    PyNvDecoder() {}
    PyNvDecoder(
        int _gpuid ,
        cudaVideoCodec _codec , 
        size_t _context,
        size_t _stream,
        bool m_bUseDeviceFrame,
        bool _enableasyncallocations,
        int maxWidth = 0,
        int maxHeight = 0,
        OutputColorType outputColorType = OutputColorType::NATIVE,
        bool _enableSEIMessage = false,
        bool bWaitForSessionWarmUp = false,
        DisplayDecodeLatency latency = DisplayDecodeLatency::DISPLAYDECODELATENCY_NATIVE
        );

    ~PyNvDecoder();

    std::vector<DecodedFrame> Decode(const PacketData pktdata);
    int GetNumDecodedFrame(const PacketData pktdata);
    uint8_t* GetLockedFrame(int64_t* pTimestamp);
    void UnlockFrame(uint8_t* pFrame);

    /**
   *  @brief  This function returns single decode frame.
   */
    DecodedFrame GetFrame();

    /**
   *  @brief  This function is used to wait on the event in current stream.
   */
    void CUStreamWaitOnEvent(CUstream _stream) { decoder->CUStreamWaitOnEvent(_stream); }

    /**
    *  @brief  This function is used to sync on the event in current stream.
    */
    void CUStreamSyncOnEvent() { decoder->CUStreamSyncOnEvent(); }

    /**
    *  @brief  This function is used to get the output frame width.
    *  NV12/P016 output format width is 2 byte aligned because of U and V interleave
    */
    int GetWidth() { return decoder->GetWidth(); }

    /**
    *  @brief  This function is used to get the actual decode width
    */
    int GetDecodeWidth() { return decoder->GetDecodeWidth(); }

    /**
    *  @brief  This function is used to get the output frame height (Luma height).
    */
    int GetHeight() { return decoder->GetHeight(); }

    /**
    *  @brief  This function is used to get the current chroma height.
    */
    int GetChromaHeight() { return decoder->GetChromaHeight(); }

    /**
    *  @brief  This function is used to get the number of chroma planes.
    */
    int GetNumChromaPlanes() { return decoder->GetNumChromaPlanes(); }

    /**
    *   @brief  This function is used to get the current frame size based on pixel format.
    */
    int GetFrameSize() { return decoder->GetOutputFrameSize(); }

    /**
    *   @brief  This function is used to get the current frame Luma plane size.
    */
    int GetLumaPlaneSize() { return decoder->GetLumaPlaneSize(); }

    /**
    *   @brief  This function is used to get the current frame chroma plane size.
    */
    int GetChromaPlaneSize() { return decoder->GetChromaPlaneSize(); }

    /**
    *  @brief  This function is used to get the pitch of the device buffer holding the decoded frame.
    */
    int GetDeviceFramePitch() { return decoder->GetDeviceFramePitch(); }

    /**
    *   @brief  This function is used to get the bit depth associated with the pixel format.
    */
    int GetBitDepth() { return decoder->GetBitDepth(); }

    /**
    *   @brief  This function is used to get the bytes used per pixel.
    */
    int GetBPP() { return decoder->GetBPP(); }

    /**
    *   @brief  This function is used to get the YUV chroma format
    */
    cudaVideoSurfaceFormat GetOutputFormat() { return decoder->GetOutputFormat(); }

    /**
    *   @brief  This function allows app to set decoder reconfig params
    *   @param  pResizeDim - width and height of resized output
    */
    int setReconfigParams(const Dim& mResizeDim) { return decoder->setReconfigParams(mResizeDim); };

    void setDecoderSessionID(int sessionID) { decoder->setDecoderSessionID(sessionID); }
    
    static int64_t getDecoderSessionOverHead(int sessionID) { return NvDecoder::getDecoderSessionOverHead(sessionID); }

    void setSeekPTS(uint64_t pts) { decoder->setSeekPTS(pts); }

    int64_t GetSessionInitTime() { return decoder->GetSessionInitTime(); }

    static void SetSessionCount(uint32_t count) { NvDecoder::SetSessionCount(count); }
};
