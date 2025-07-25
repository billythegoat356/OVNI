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

#include <tuple>
#include <vector>

#include "NvDecoder/NvDecoder.h"
#include "PyCAIMemoryView.hpp"

/* If you start adding stuff to this file which changes often, consider adding definitions in a cpp file*/
inline void ValidateGpuId(int gpuId)
{
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (gpuId < 0 || gpuId >= nGpu) {
        std::ostringstream err;
        err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        PYNVVC_THROW_ERROR(err.str(), CUDA_ERROR_NOT_SUPPORTED);
    }
}

inline void ValidateCUDAContext(int gpuId, CUcontext context)
{

    if(context)
    {
        CUdevice gpuDeviceFromGpuId = 0;
        CUdevice gpuDeviceFromContext = 0;
        ck(cuCtxPushCurrent(context));
        ck(cuDeviceGet(&gpuDeviceFromGpuId, gpuId));
        ck(cuCtxGetDevice(&gpuDeviceFromContext));
        ck(cuCtxPopCurrent(NULL));
        if (gpuDeviceFromGpuId != gpuDeviceFromContext)
        {
            std::ostringstream err;
            err << "Gpu id " << gpuId << " doesnt match with cuda context device "
                << gpuDeviceFromContext << std::endl;
            PYNVVC_THROW_ERROR(err.str(), CUDA_ERROR_NOT_SUPPORTED);
        }
    }
}

inline void ValidateCUDAStream(CUstream stream, CUcontext currentContext)
{
    CUcontext streamContext;
    cuStreamGetCtx(stream, &streamContext);
    if(streamContext != currentContext)
    {
        PYNVVC_THROW_ERROR("cudastream input argument does not correspond to cudacontext argument", CUDA_ERROR_NOT_SUPPORTED);
    }
}


template<typename T>
inline std::string vectorString(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "[ ";
    for (const auto& elem : vec) {
        ss << elem << " ";
    }
    ss << "]\n";

    return ss.str();
}

inline Pixel_Format GetNativeFormat(const cudaVideoSurfaceFormat inputFormat)
{
    switch (inputFormat)
    {
    case cudaVideoSurfaceFormat_NV12: return Pixel_Format_NV12;
    case cudaVideoSurfaceFormat_P016: return Pixel_Format_P016;
    case cudaVideoSurfaceFormat_YUV444: return Pixel_Format_YUV444;
    case cudaVideoSurfaceFormat_YUV444_16Bit: return Pixel_Format_YUV444_16Bit;
    case cudaVideoSurfaceFormat_NV16: return Pixel_Format_NV16;
    case cudaVideoSurfaceFormat_P216: return Pixel_Format_P216;
    default:
        break;
    }
    return Pixel_Format_UNDEFINED;
}

inline Pixel_Format GetPixelFormat(const NvDecoder* decoder, const OutputColorType colorType)
{
    switch (colorType)
    {
        case OutputColorType::NATIVE: return GetNativeFormat(decoder->GetOutputFormat());
        case OutputColorType::RGB: return Pixel_Format_RGB;
        case OutputColorType::RGBP: return Pixel_Format_RGBP;
        default: return Pixel_Format_UNDEFINED;
    }
}

inline DecodedFrame GetCAIMemoryViewAndDLPack(const NvDecoder* decoder, std::tuple<CUdeviceptr, int64_t, SEI_MESSAGE, CUevent> tup)
{
    DecodedFrame frame;

    frame.format = GetPixelFormat(decoder, decoder->GetUserOutputColorType());
    auto width = size_t(decoder->GetWidth());
    auto height = size_t(decoder->GetHeight());
    auto data = std::get<0>(tup); 
    frame.timestamp = std::get<1>(tup);
	frame.seiMessage = std::get<2>(tup);
    frame.decoderStreamEvent = reinterpret_cast<size_t>(std::get<3>(tup));
    frame.decoderStream = reinterpret_cast<size_t>(decoder->GetStream());
    switch (frame.format)
    {
        case Pixel_Format_NV12:
        {
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            frame.views.push_back(CAIMemoryView{ {height / 2, width / 2, 2}, {width / 2 * 2, 2, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()),(data + width * height), false });
            // Load DLPack Tensor
            std::vector<size_t> shape{ (size_t)(height * 1.5), width };
            std::vector<size_t> stride{ size_t(width), 1 };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u1", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());
        }
        break;
        case Pixel_Format_P016:
        {
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            frame.views.push_back(CAIMemoryView{ {height / 2, width / 2, 2}, {width / 2 * 2, 2, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()),(data + 2 * (width * height)), false });
            std::vector<size_t> stride{ size_t(width), 2 };
            std::vector<size_t> shape{ (size_t)(height * 1.5), width };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u2", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());
        }
        break;
        case Pixel_Format_YUV444:
        {
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()),(data + width * height), false });
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()),(data + 2 * (width * height)), false });
            std::vector<size_t> shape{ (size_t)(height * 3), width };
            std::vector<size_t> stride{ size_t(width), 1 };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u1", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());
        }
        break;
        case Pixel_Format_YUV444_16Bit:
        {
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()),(data +  (width * height)), false });
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()),(data + 2 * (width * height)), false });
            std::vector<size_t> shape{ (size_t)(height * 3), width };
            std::vector<size_t> stride{ size_t(width) * 2, 2 };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u2", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());
        }
        break;
        case Pixel_Format_NV16:
        {
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            frame.views.push_back(CAIMemoryView{ {height, width / 2, 2}, {width / 2 * 2, 2, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()),(data + width * height), false });
            // Load DLPack Tensor
            std::vector<size_t> shape{ (size_t)(height * 2), width };
            std::vector<size_t> stride{ size_t(width), 1 };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u1", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());
        }
        break;
        case Pixel_Format_P216:
        {
            frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            frame.views.push_back(CAIMemoryView{ {height, width / 2, 2}, {width / 2 * 2, 2, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()),(data + 2 * (width * height)), false });
            std::vector<size_t> shape{ (size_t)(height * 2), width };
            std::vector<size_t> stride{ size_t(width), 2 };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u2", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());
        }
        break;
        case Pixel_Format_RGB:
        {
            frame.views.push_back(CAIMemoryView{ {height, width, 3}, {width * 3, 3, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            std::vector<size_t> shape{ height, width, 3 }; // HWC
            std::vector<size_t> stride{ width * 3, 3, 1 };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u1", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());

        }
        break;
        case Pixel_Format_RGBP:
        {
            frame.views.push_back(CAIMemoryView{ {height, width}, {width, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
            frame.views.push_back(CAIMemoryView{ {height, width}, {width, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data + (width * height)), false });
            frame.views.push_back(CAIMemoryView{ {height, width}, {width, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data + (2*width*height)), false });
            std::vector<size_t> shape{ 3, height, width }; // CHW
            std::vector<size_t> stride{ width * height, width, 1 };
            int returntype = frame.extBuf->LoadDLPack(shape, stride, "|u1", data,
                                                      decoder->IsDeviceFrame(), decoder->GetDeviceId(),
                                                      decoder->GetContext());
        }
        break;
    }
    
    return frame;
}

