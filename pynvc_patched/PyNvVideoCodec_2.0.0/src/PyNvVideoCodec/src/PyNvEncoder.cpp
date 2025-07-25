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

#include "PyNvEncoder.hpp"
#include "NvEncoderClInterface.hpp"
#include "PyCAIMemoryView.hpp"
#include "PyNvVideoCodecUtils.hpp"

#include "cuda.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <unordered_map>

using namespace std;
using namespace chrono;

namespace py = pybind11;

using CAPS = std::unordered_map<std::string, uint32_t>;

// Forward declaration of PyNvEncoderCaps
static CAPS PyNvEncoderCaps(int32_t gpuid, std::string codec);

PyNvEncoder::PyNvEncoder( PyNvEncoder&& pyenvc)
    :m_encoder(std::move(pyenvc.m_encoder)), m_CUcontext(pyenvc.m_CUcontext), m_width(pyenvc.m_width), m_height(pyenvc.m_height), m_eBufferFormat(pyenvc.m_eBufferFormat),
    pCUStream(std::move(pyenvc.pCUStream)), m_mapPtr(pyenvc.m_mapPtr), m_gpuId(pyenvc.m_gpuId)
{
    
}

PyNvEncoder::PyNvEncoder(PyNvEncoder& pyenvc)
    :m_encoder(std::move(pyenvc.m_encoder)), m_CUcontext(pyenvc.m_CUcontext), m_width(pyenvc.m_width), m_height(pyenvc.m_height), m_eBufferFormat(pyenvc.m_eBufferFormat),
    pCUStream(std::move(pyenvc.pCUStream)), m_mapPtr(pyenvc.m_mapPtr), m_gpuId(pyenvc.m_gpuId)
{

}

PyNvEncoder::PyNvEncoder(
        int _width,
        int _height,
        std::string _format,
        size_t  _cudacontext,
        size_t _cudastream,
        bool bUseCPUInutBuffer,
        std::map<std::string, std::string> kwargs)
{

    py::gil_scoped_release release;
    std::map<std::string, std::string> options = kwargs;
    if (options.count("gpu_id") == 1)
    {
        m_gpuId = stoi(options["gpu_id"].c_str());
    }
    NV_ENC_BUFFER_FORMAT eBufferFormat;
    CUcontext cudacontext =(CUcontext) _cudacontext;
    CUstream cudastream = (CUstream)_cudastream;

    NV_ENC_INITIALIZE_PARAMS params = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
    params.encodeConfig = &encodeConfig;

    // Set codec GUID based on options
    std::string codec = "h264"; // default codec
    if (options.count("codec") == 1) {
        codec = options["codec"];
    }
    params.encodeGUID = (codec == "hevc") ? NV_ENC_CODEC_HEVC_GUID : 
                       (codec == "av1") ? NV_ENC_CODEC_AV1_GUID : 
                       NV_ENC_CODEC_H264_GUID;

    // Check encoder capabilities first
    CUDA_DRVAPI_CALL(cuInit(0));
    ValidateGpuId(m_gpuId);
    if(cudacontext)
    {
        ValidateCUDAContext(m_gpuId, cudacontext);
    }
    else
    {
        CUDA_DRVAPI_CALL(cuDevicePrimaryCtxRetain(&cudacontext, m_gpuId));
        m_ReleasePrimaryContext = true;
    }

    if(!cudacontext)
    {
        PYNVVC_THROW_ERROR("Failed to create a cuda context. Create a cudacontext and pass it as named argument 'cudacontext = app_ctx'", NV_ENC_ERR_INVALID_PARAM);
    }


    // Get encoder capabilities using the same approach as PyNvEncoderCaps
    CAPS caps = PyNvEncoderCaps(m_gpuId, codec);
    

    bool supports_444 = caps["support_yuv444_encode"];
    bool supports_10bit = caps["support_10bit_encode"];
    bool supports_422 = false;
#if CHECK_API_VERSION(13,0)
    supports_422 = caps["support_yuv422_encode"];
#endif

    // Build list of supported formats
    std::string supported_formats = "Supported formats:\n";
    supported_formats += "- NV12, YUV420, ARGB, ABGR (always supported)\n";
    if (supports_444)
        supported_formats += "- YUV444\n";
    if (supports_10bit)
        supported_formats += "- P010\n";
    if (supports_444 && supports_10bit)
        supported_formats += "- YUV444_10BIT, YUV444_16BIT\n";
#if CHECK_API_VERSION(13,0)
    if (supports_422)
        supported_formats += "- NV16\n";
    if (supports_422 && supports_10bit)
        supported_formats += "- P210\n";
#endif

    // Now validate the requested format
    if(_format == "NV12")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    }
    else if(_format == "YUV420")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    }
    else if(_format == "ARGB")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
    }
    else if(_format == "ABGR")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
    }
    else if(_format == "YUV444")
    {
        if (!supports_444)
        {
            std::string error = "Format YUV444 is not supported by current encoder.\n" + supported_formats;
            PYNVVC_THROW_ERROR_UNSUPPORTED(error.c_str(), NV_ENC_ERR_INVALID_PARAM);
        }
        eBufferFormat = NV_ENC_BUFFER_FORMAT_YUV444;
    }
    else if(_format == "YUV444_10BIT" || _format == "YUV444_16BIT")
    {
        if (!supports_444 || !supports_10bit)
        {
            std::string error = "Format " + _format + " is not supported by current encoder.\n" + supported_formats;
            PYNVVC_THROW_ERROR_UNSUPPORTED(error.c_str(), NV_ENC_ERR_INVALID_PARAM);
        }
        _format = "YUV444_10BIT";
        eBufferFormat = NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
    }
    else if(_format == "P010" || _format == "ARGB10" || _format == "ABGR10")
    {
        if (!supports_10bit)
        {
            std::string error = "Format " + _format + " is not supported by current encoder.\n" + supported_formats;
            PYNVVC_THROW_ERROR_UNSUPPORTED(error.c_str(), NV_ENC_ERR_INVALID_PARAM);
        }
        if (_format == "P010")
            eBufferFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
        else if (_format == "ARGB10")
            eBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB10;
        else
            eBufferFormat = NV_ENC_BUFFER_FORMAT_ABGR10;
    }
#if CHECK_API_VERSION(13,0)
    else if (_format == "NV16")
    {
        if (!supports_422)
        {
            std::string error = "Format NV16 is not supported by current encoder.\n" + supported_formats;
            PYNVVC_THROW_ERROR_UNSUPPORTED(error.c_str(), NV_ENC_ERR_INVALID_PARAM);
        }
        eBufferFormat = NV_ENC_BUFFER_FORMAT_NV16;
    }
    else if (_format == "P210")
    {
        if (!supports_422 || !supports_10bit)
        {
            std::string error = "Format P210 is not supported by current encoder.\n" + supported_formats;
            PYNVVC_THROW_ERROR_UNSUPPORTED(error.c_str(), NV_ENC_ERR_INVALID_PARAM);
        }
        eBufferFormat = NV_ENC_BUFFER_FORMAT_P210;
    }
#endif
    else
    {
        std::string error = "Unknown format: " + _format + "\n" + supported_formats;
        PYNVVC_THROW_ERROR_UNSUPPORTED(error.c_str(), NV_ENC_ERR_INVALID_PARAM);
    }

    params.bufferFormat = eBufferFormat;

    if(cudastream)
    {
        ValidateCUDAStream(cudastream, cudacontext);
    }
    else
    {
        CUDA_DRVAPI_CALL(cuCtxPushCurrent(cudacontext));
        CUDA_DRVAPI_CALL(cuStreamCreate(&cudastream, CU_STREAM_NON_BLOCKING););
        CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    }



    // Create the actual encoder with the validated format
    m_encoder = std::make_unique<NvEncoderCuda>(cudacontext, cudastream, _width, _height, eBufferFormat);


    options.insert({"fmt", _format});
    options.insert({"s", std::to_string(_width) + "x" + std::to_string(_height)});
    NvEncoderClInterface cliInterface(options);


    cliInterface.SetupInitParams(params, false, m_encoder->GetApi(), m_encoder->GetEncoder(), false);
    m_encoder->CreateDefaultEncoderParams(&params, params.encodeGUID, params.presetGUID, params.tuningInfo);


    m_encoder->CreateEncoder(&params);


    pCUStream.reset(new NvCUStream(cudacontext, cudastream, m_encoder));
    InitEncodeReconfigureParams(params);
    m_encodeGUID = params.encodeGUID;
    m_CUcontext = cudacontext;
    m_CUstream = cudastream;
    m_width = _width;
    m_height = _height;
    m_eBufferFormat = eBufferFormat;
    m_bUseCPUInutBuffer = bUseCPUInutBuffer;
    m_mapPtr.clear();


}

void PyNvEncoder::InitEncodeReconfigureParams(const NV_ENC_INITIALIZE_PARAMS params)
{
    NV_ENC_RC_PARAMS& reconfigRCParams = params.encodeConfig->rcParams;

    m_EncReconfigureParams.rateControlMode = reconfigRCParams.rateControlMode;
    m_EncReconfigureParams.multiPass = reconfigRCParams.multiPass;
    m_EncReconfigureParams.averageBitrate = reconfigRCParams.averageBitRate;
    m_EncReconfigureParams.vbvBufferSize = reconfigRCParams.vbvBufferSize;
    m_EncReconfigureParams.maxBitRate = reconfigRCParams.maxBitRate;
    m_EncReconfigureParams.vbvInitialDelay = reconfigRCParams.vbvInitialDelay;
    m_EncReconfigureParams.frameRateNum = params.frameRateNum;
    m_EncReconfigureParams.frameRateDen = params.frameRateDen;
}

structEncodeReconfigureParams PyNvEncoder::GetEncodeReconfigureParams()
{
    structEncodeReconfigureParams reconfigureParams;
    reconfigureParams.rateControlMode = m_EncReconfigureParams.rateControlMode;
    reconfigureParams.multiPass = m_EncReconfigureParams.multiPass;
    reconfigureParams.averageBitrate = m_EncReconfigureParams.averageBitrate;
    reconfigureParams.vbvBufferSize = m_EncReconfigureParams.vbvBufferSize;
    reconfigureParams.maxBitRate = m_EncReconfigureParams.maxBitRate;
    reconfigureParams.vbvInitialDelay = m_EncReconfigureParams.vbvInitialDelay;
    reconfigureParams.frameRateNum = m_EncReconfigureParams.frameRateNum;
    reconfigureParams.frameRateDen = m_EncReconfigureParams.frameRateDen;
    return reconfigureParams;
}


const NvEncInputFrame* PyNvEncoder::GetEncoderInputFromCPUBuffer(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> framedata)
{
    auto encoderInputFrame = m_encoder->GetNextInputFrame();
    void* srcPtr = (void*)framedata.data(0);
    uint32_t srcStride = 0;
    uint32_t srcChromaOffsets[2];

    switch (m_eBufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_NV12:
    {   
        srcChromaOffsets[0] = m_width * m_height;
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YUV444:
    {
        srcChromaOffsets[0] = (m_width * m_height);
        srcChromaOffsets[1] = 2 * (m_width * m_height);
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    {
        srcChromaOffsets[0] = 2* (m_width * m_height);
        srcChromaOffsets[1] = 4 * (m_width * m_height);
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    {
        srcChromaOffsets[0] = m_width * m_height;
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YV12:
    {
        srcChromaOffsets[0] = (m_width * m_height);
        break;
    }
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    {
        PYNVVC_THROW_ERROR_UNSUPPORTED("ARGB10 format not supported in current release. Use YUV444_16BIT or P010", NV_ENC_ERR_INVALID_PARAM);
        break;
    }
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ARGB:
    {
        srcChromaOffsets[0] = 0;
        break;
    }
#if CHECK_API_VERSION(13,0)
    case NV_ENC_BUFFER_FORMAT_NV16:
    case NV_ENC_BUFFER_FORMAT_P210:
    {
        srcChromaOffsets[0] = m_width * m_height;
        break;
    }
#endif
    default:
        PYNVVC_THROW_ERROR_UNSUPPORTED("Format not supported", NV_ENC_ERR_INVALID_PARAM);
    }

    NvEncoderCuda::CopyToDeviceFrame(m_CUcontext,
        (void*)srcPtr,
        srcStride,
        (CUdeviceptr)encoderInputFrame->inputPtr,
        (int)encoderInputFrame->pitch,
        m_encoder->GetEncodeWidth(),
        m_encoder->GetEncodeHeight(),
        CU_MEMORYTYPE_HOST,
        encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets,
        encoderInputFrame->numChromaPlanes,
        false,
        nullptr,
        srcChromaOffsets
    );
    return encoderInputFrame;
}

const NvEncInputFrame* PyNvEncoder::GetEncoderInput(py::object frame)
{
    auto encoderInputFrame = m_encoder->GetNextInputFrame();
    void * srcPtr = nullptr;
    uint32_t srcStride = 0;
    uint32_t srcChromaOffsets[2];

    if(m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_NV12 || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_IYUV)
    {
        //YUV420_10BIT is actually P010 format
        if (py::hasattr(frame, "__dlpack__"))
        {
            if (py::hasattr(frame, "__dlpack_device__"))
            {
                py::tuple dlpackDevice = frame.attr("__dlpack_device__")().cast<py::tuple>();
                auto devType = static_cast<DLDeviceType>(dlpackDevice[0].cast<int>());
                if (!IsCudaAccessible(devType))
                {
                    PYNVVC_THROW_ERROR("Only CUDA-accessible memory buffers can be wrapped", NV_ENC_ERR_INVALID_PARAM);
                }
            }
            int64_t consumer_stream = 0;
            if (m_CUstream == CU_STREAM_LEGACY)
            {
                consumer_stream = 1;
            }
            // We mostly dont need this. The else part should take care
            else if (m_CUstream == CU_STREAM_PER_THREAD)
            {
                consumer_stream = 2;
            }
            else
            {
                consumer_stream = reinterpret_cast<int64_t>(m_CUstream);
            }
            py::capsule cap = frame.attr("__dlpack__")(consumer_stream).cast<py::capsule>();
            if (auto* tensor = static_cast<DLManagedTensor*>(cap.get_pointer()))
            {
                srcPtr = tensor->dl_tensor.data;//we got the luma pointer
                py::tuple shape(tensor->dl_tensor.ndim);//assuming its a CHW tensor, so tensor height should be 1.5 times actual height
                int64_t tensorWidth = tensor->dl_tensor.shape[1];
                int64_t tensorHeight = tensor->dl_tensor.shape[0];
                if (tensorHeight != (m_height * 1.5))
                {
                    std::string error = "Tensor height :";
                    error.append(std::to_string(tensorHeight));
                    error.append(" must be 1.5 times the actual height :");
                    error.append(std::to_string(m_height));
                    error.append(" passed to encoder.");
                    PYNVVC_THROW_ERROR(error, NV_ENC_ERR_INVALID_PARAM);
                }
                srcStride = tensor->dl_tensor.strides[0] * tensor->dl_tensor.dtype.bits / 8;
                srcChromaOffsets[0] = m_width * m_height * tensor->dl_tensor.dtype.bits / 8;;
            }
        }
        else
        {
            CAIMemoryView yPlane = coerceToCudaArrayView(frame.attr("__getitem__")(0), m_eBufferFormat, m_width, m_height, 0);
            CAIMemoryView uvPlane = coerceToCudaArrayView(frame.attr("__getitem__")(1), m_eBufferFormat, m_width, m_height, 1);

            if (m_eBufferFormat == NV_ENC_BUFFER_FORMAT_IYUV)
            {
                if (yPlane.stride[0] != (uvPlane.stride[0] * 2 ))
                {
                    PYNVVC_THROW_ERROR("unsupported argument : strides of y and uv plane  are different", NV_ENC_ERR_INVALID_PARAM);
                }
            }
            else
            {
                if (yPlane.stride[0] != uvPlane.stride[0])
                {
                    PYNVVC_THROW_ERROR("unsupported argument : strides of y and uv plane  are different", NV_ENC_ERR_INVALID_PARAM);
                }
            }
            
            srcPtr = (void*)yPlane.data;
            srcStride = yPlane.stride[0];
            if (uvPlane.data <= yPlane.data)
            {
                PYNVVC_THROW_ERROR("Unsupported surface allocation. u plane must follow yplane.", NV_ENC_ERR_INVALID_PARAM);
            }
            srcChromaOffsets[0] = static_cast<uint32_t>(uvPlane.data - yPlane.data);
        }
    }
    else if(m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ARGB || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ABGR 
            || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ARGB10 || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ABGR10)
    {
        CAIMemoryView argb = coerceToCudaArrayView(frame, m_eBufferFormat, m_width, m_height);
        srcPtr =(void*) argb.data;
        srcStride = argb.stride[0];
        srcChromaOffsets[0] = 0;
    }
    else if(m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || 
            m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
    {
        if (py::hasattr(frame, "__dlpack__"))
        {
            if (py::hasattr(frame, "__dlpack_device__"))
            {
                py::tuple dlpackDevice = frame.attr("__dlpack_device__")().cast<py::tuple>();
                auto      devType = static_cast<DLDeviceType>(dlpackDevice[0].cast<int>());
                if (!IsCudaAccessible(devType))
                {
                    PYNVVC_THROW_ERROR("Only CUDA-accessible memory buffers can be wrapped", NV_ENC_ERR_INVALID_PARAM);
                }
            }
            int64_t consumer_stream = 0;
            if (m_CUstream == CU_STREAM_LEGACY)
            {
                consumer_stream = 1;
            }
            // We mostly dont need this. The else part should take care
            else if (m_CUstream == CU_STREAM_PER_THREAD)
            {
                consumer_stream = 2;
            }
            else
            {
                consumer_stream = reinterpret_cast<int64_t>(m_CUstream);
            }
            py::capsule cap = frame.attr("__dlpack__")(consumer_stream).cast<py::capsule>();
            if (auto* tensor = static_cast<DLManagedTensor*>(cap.get_pointer()))
            {
                srcPtr = tensor->dl_tensor.data;//we got the luma pointer
                py::tuple shape(tensor->dl_tensor.ndim);//assuming its a CHW tensor, so tensor height should be 3 times actual height
                int64_t tensorWidth = tensor->dl_tensor.shape[1];
                int64_t tensorHeight = tensor->dl_tensor.shape[0];
                if (tensorHeight != (m_height * 3))
                {
                    std::string error = "Tensor height :";
                    error.append(std::to_string(tensorHeight));
                    error.append(" must be 3 times the actual height :");
                    error.append(std::to_string(m_height));
                    error.append(" passed to encoder.");
                    PYNVVC_THROW_ERROR(error, NV_ENC_ERR_INVALID_PARAM);
                }
                srcStride = tensor->dl_tensor.strides[0] * tensor->dl_tensor.dtype.bits / 8;
                srcChromaOffsets[0] = m_width * m_height * tensor->dl_tensor.dtype.bits / 8 ;
                srcChromaOffsets[1] = 2 * m_width * m_height * tensor->dl_tensor.dtype.bits / 8;
            }
        }
        else
        {
            CAIMemoryView yPlane = coerceToCudaArrayView(frame.attr("__getitem__")(0), m_eBufferFormat, m_width, m_height, 0);
            CAIMemoryView uPlane = coerceToCudaArrayView(frame.attr("__getitem__")(1), m_eBufferFormat, m_width, m_height, 1);
            CAIMemoryView vPlane = coerceToCudaArrayView(frame.attr("__getitem__")(2), m_eBufferFormat, m_width, m_height, 2);
            if (uPlane.stride[0] != vPlane.stride[0])
            {
                PYNVVC_THROW_ERROR("unsupported argument : strides of  u, v must match", NV_ENC_ERR_INVALID_PARAM);
            }
            srcPtr = (void*)yPlane.data;
            srcStride = yPlane.stride[0];
            if (uPlane.data <= yPlane.data || vPlane.data <= uPlane.data)
            {
                PYNVVC_THROW_ERROR("Incorrect surface allocation. u and v plane must follow yplane.", NV_ENC_ERR_INVALID_PARAM);
            }
            srcChromaOffsets[0] = uPlane.data - yPlane.data;
            srcChromaOffsets[1] = vPlane.data - yPlane.data;
        }
    }
#if CHECK_API_VERSION(13,0)
    else if (m_eBufferFormat == NV_ENC_BUFFER_FORMAT_NV16 || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_P210)
    {
        if (py::hasattr(frame, "__dlpack__"))
        {
            if (py::hasattr(frame, "__dlpack_device__"))
            {
                py::tuple dlpackDevice = frame.attr("__dlpack_device__")().cast<py::tuple>();
                auto      devType = static_cast<DLDeviceType>(dlpackDevice[0].cast<int>());
                if (!IsCudaAccessible(devType))
                {
                    PYNVVC_THROW_ERROR("Only CUDA-accessible memory buffers can be wrapped", NV_ENC_ERR_INVALID_PARAM);
                }
            }
            int64_t consumer_stream = 0;
            if (m_CUstream == CU_STREAM_LEGACY)
            {
                consumer_stream = 1;
            }
            // We mostly dont need this. The else part should take care
            else if (m_CUstream == CU_STREAM_PER_THREAD)
            {
                consumer_stream = 2;
            }
            else
            {
                consumer_stream = reinterpret_cast<int64_t>(m_CUstream);
            }
            py::capsule cap = frame.attr("__dlpack__")(consumer_stream).cast<py::capsule>();
            if (auto* tensor = static_cast<DLManagedTensor*>(cap.get_pointer()))
            {
                srcPtr = tensor->dl_tensor.data;//we got the luma pointer
                py::tuple shape(tensor->dl_tensor.ndim);//assuming its a CHW tensor, so tensor height should be 2 times actual height
                int64_t tensorWidth = tensor->dl_tensor.shape[1];
                int64_t tensorHeight = tensor->dl_tensor.shape[0];
                if (tensorHeight != (m_height * 2))
                {
                    std::string error = "Tensor height :";
                    error.append(std::to_string(tensorHeight));
                    error.append(" must be 2 times the actual height :");
                    error.append(std::to_string(m_height));
                    error.append(" passed to encoder.");
                    PYNVVC_THROW_ERROR(error, NV_ENC_ERR_INVALID_PARAM);
                }
                srcStride = tensor->dl_tensor.strides[0] * tensor->dl_tensor.dtype.bits / 8;
                srcChromaOffsets[0] = m_width * m_height * tensor->dl_tensor.dtype.bits / 8;
            }
        }
        else
        {
            CAIMemoryView yPlane = coerceToCudaArrayView(frame.attr("__getitem__")(0), m_eBufferFormat, m_width, m_height, 0);
            CAIMemoryView uvPlane = coerceToCudaArrayView(frame.attr("__getitem__")(1), m_eBufferFormat, m_width, m_height, 1);

            srcPtr = (void*)yPlane.data;
            srcStride = yPlane.stride[0];
            if (uvPlane.data <= yPlane.data)
            {
                PYNVVC_THROW_ERROR("Unsupported surface allocation. uv plane must follow yplane.", NV_ENC_ERR_INVALID_PARAM);
            }
            srcChromaOffsets[0] = static_cast<uint32_t>(uvPlane.data - yPlane.data);
        }
    }
#endif
    else
    {
        PYNVVC_THROW_ERROR_UNSUPPORTED("unsupported format.", NV_ENC_ERR_INVALID_PARAM);
    }
    NvEncoderCuda::CopyToDeviceFrame(m_CUcontext, 
        (void*) srcPtr,
        srcStride,
        (CUdeviceptr) encoderInputFrame->inputPtr,
        (int) encoderInputFrame->pitch,
        m_encoder->GetEncodeWidth(),
        m_encoder->GetEncodeHeight(),
        CU_MEMORYTYPE_DEVICE,
        encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets,
        encoderInputFrame->numChromaPlanes,
        false,
        m_CUstream,
        srcChromaOffsets
        );
    return encoderInputFrame;
}

py::bytes PyNvEncoder::Encode(py::object _frame)
{
    py::object frame = _frame;
    std::vector<NvEncOutputFrame> vvByte;

    if(hasattr(frame, "cuda"))
    {
        frame = frame.attr("cuda")();
        GetEncoderInput(frame);
    }
    else
    {
        if (!m_bUseCPUInutBuffer)
        {
            PYNVVC_THROW_ERROR("incorrect usage of CPU inut buffer", NV_ENC_ERR_INVALID_PARAM);
        }
        GetEncoderInputFromCPUBuffer(frame);
    }
    py::gil_scoped_release release;
    NV_ENC_PIC_PARAMS picParam = { 0 };
    picParam.inputTimeStamp = m_frameNum++;
    
    m_encoder->EncodeFrame(vvByte, &picParam);

    size_t totalSize = 0;

    for (const auto& vByte : vvByte) {
        totalSize += vByte.frame.size();
    }

    // Allocate a single buffer

    std::unique_ptr<char[]> buffer(new char[totalSize]);
    size_t offset = 0;

    for (const auto& vByte : vvByte) {
        if (vByte.frame.size() > 0) {
            std::memcpy(buffer.get() + offset, vByte.frame.data(), vByte.frame.size());
            offset += vByte.frame.size();
        }
    }

    py::gil_scoped_acquire acquire;
    return py::bytes(buffer.get(), totalSize);

}

// For Encode with pic flags and SEI
py::bytes PyNvEncoder::Encode(py::object _frame, uint8_t m_picFlags, SEI_MESSAGE sei)
{
    py::object frame = _frame;
    std::vector<NvEncOutputFrame> vvByte;

    if(hasattr(frame, "cuda"))
    {
        frame = frame.attr("cuda")();
        GetEncoderInput(frame);
    }
    else
    {
        if (!m_bUseCPUInutBuffer)
        {
            PYNVVC_THROW_ERROR("incorrect usage of CPU input buffer", NV_ENC_ERR_INVALID_PARAM);
        }
        GetEncoderInputFromCPUBuffer(frame);
    }

    py::gil_scoped_release release;

    NV_ENC_PIC_PARAMS picParam = { 0 };
    picParam.inputTimeStamp = m_frameNum++;
    NV_ENC_SEI_PAYLOAD *pSei = NULL;

    if (sei.size())
    {
        pSei = new NV_ENC_SEI_PAYLOAD[sei.size()];
        unsigned int i = 0;
        for (const auto& seimessage : sei) 
        {
            auto it = seimessage.first.find("sei_type");
            if (it != seimessage.first.end())
            {
                pSei[i].payloadType = it->second;
            }
            else
            {
                continue;
            }
            pSei[i].payloadSize = seimessage.second.size();
            pSei[i].payload = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(seimessage.second.data()));
            i++;
        }
        if (m_encodeGUID == NV_ENC_CODEC_H264_GUID)
        {
            picParam.codecPicParams.h264PicParams.seiPayloadArrayCnt = sei.size();
            picParam.codecPicParams.h264PicParams.seiPayloadArray = pSei;
        }
        else if (m_encodeGUID == NV_ENC_CODEC_HEVC_GUID)
        {
            picParam.codecPicParams.hevcPicParams.seiPayloadArrayCnt = sei.size();
            picParam.codecPicParams.hevcPicParams.seiPayloadArray = pSei;
        }
        else if (m_encodeGUID == NV_ENC_CODEC_AV1_GUID)
        {
            picParam.codecPicParams.av1PicParams.obuPayloadArrayCnt = sei.size();
            picParam.codecPicParams.av1PicParams.obuPayloadArray = pSei;
        }
    }
    picParam.encodePicFlags |= m_picFlags;

    m_encoder->EncodeFrame(vvByte, &picParam);

    // Calculate total size
    size_t totalSize = 0;
    for (const auto& vByte : vvByte) {
        totalSize += vByte.frame.size();
    }

    // Allocate a single buffer
    std::unique_ptr<char[]> buffer(new char[totalSize]);
    size_t offset = 0;
    for (const auto& vByte : vvByte) {
        if (vByte.frame.size() > 0) {
            std::memcpy(buffer.get() + offset, vByte.frame.data(), vByte.frame.size());
            offset += vByte.frame.size();
        }
    }

    if (pSei)
        delete[] pSei;

    py::gil_scoped_acquire acquire;
    return py::bytes(buffer.get(), totalSize);
}


// For Encode with pic flags
py::bytes PyNvEncoder::Encode(py::object _frame, uint8_t m_picFlags) 
{
    py::object frame = _frame;
    std::vector<NvEncOutputFrame> vvByte;

    if(hasattr(frame, "cuda"))
    {
        frame = frame.attr("cuda")();
        GetEncoderInput(frame);
    }
    else
    {
        if (!m_bUseCPUInutBuffer)
        {
            PYNVVC_THROW_ERROR("incorrect usage of CPU input buffer", NV_ENC_ERR_INVALID_PARAM);
        }
        GetEncoderInputFromCPUBuffer(frame);
    }

    py::gil_scoped_release release;

    NV_ENC_PIC_PARAMS picParam = { 0 };
    picParam.inputTimeStamp = m_frameNum++;
    picParam.encodePicFlags |= m_picFlags;

    m_encoder->EncodeFrame(vvByte, &picParam);

    // Calculate total size
    size_t totalSize = 0;
    for (const auto& vByte : vvByte) {
        totalSize += vByte.frame.size();
    }

    // Allocate a single buffer
    std::unique_ptr<char[]> buffer(new char[totalSize]);
    size_t offset = 0;
    for (const auto& vByte : vvByte) {
        if (vByte.frame.size() > 0) {
            std::memcpy(buffer.get() + offset, vByte.frame.data(), vByte.frame.size());
            offset += vByte.frame.size();
        }
    }

    py::gil_scoped_acquire acquire;
    return py::bytes(buffer.get(), totalSize);
}


// For EndEncode
py::bytes PyNvEncoder::Encode()
{
    std::vector<NvEncOutputFrame> vvByte;
    
    py::gil_scoped_release release;
    
    m_encoder->EndEncode(vvByte);

    // Calculate total size
    size_t totalSize = 0;
    for (const auto& vByte : vvByte) {
        totalSize += vByte.frame.size();
    }

    // Allocate a single buffer
    std::unique_ptr<char[]> buffer(new char[totalSize]);
    size_t offset = 0;
    for (const auto& vByte : vvByte) {
        if (vByte.frame.size() > 0) {
            std::memcpy(buffer.get() + offset, vByte.frame.data(), vByte.frame.size());
            offset += vByte.frame.size();
        }
    }

    py::gil_scoped_acquire acquire;
    return py::bytes(buffer.get(), totalSize);
}


PyNvEncoder::~PyNvEncoder()
{
    py::gil_scoped_release release;
    m_width = 0;
    m_height = 0;

    if(m_ReleasePrimaryContext)
    {
        m_encoder.reset();
        pCUStream.reset();

        cuDevicePrimaryCtxRelease(m_gpuId);
        m_ReleasePrimaryContext = false;
    }

    m_CUcontext = nullptr;
}

bool PyNvEncoder::Reconfigure(structEncodeReconfigureParams rcParamsToChange)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;
    m_encoder->GetInitializeParams(&initializeParams);

    NV_ENC_RC_PARAMS& reconfigRCParams = initializeParams.encodeConfig->rcParams;

    reconfigRCParams.rateControlMode = rcParamsToChange.rateControlMode;
    reconfigRCParams.multiPass = rcParamsToChange.multiPass;
    reconfigRCParams.averageBitRate = rcParamsToChange.averageBitrate;
    reconfigRCParams.vbvBufferSize = rcParamsToChange.vbvBufferSize;
    reconfigRCParams.maxBitRate = rcParamsToChange.maxBitRate;
    reconfigRCParams.vbvInitialDelay = rcParamsToChange.vbvInitialDelay;
    initializeParams.frameRateDen = rcParamsToChange.frameRateDen;
    initializeParams.frameRateNum = rcParamsToChange.frameRateNum;

    NV_ENC_RECONFIGURE_PARAMS reconfigureParams = { NV_ENC_RECONFIGURE_PARAMS_VER };
    memcpy(&reconfigureParams.reInitEncodeParams, &initializeParams, sizeof(initializeParams));
    
    NV_ENC_CONFIG reInitCodecConfig = { NV_ENC_CONFIG_VER };
    memcpy(&reInitCodecConfig, initializeParams.encodeConfig, sizeof(reInitCodecConfig));
    
    reconfigureParams.reInitEncodeParams.encodeConfig = &reInitCodecConfig;

    //InitEncodeReconfigureParams(initializeParams);

    reconfigureParams.reInitEncodeParams.tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;

    return m_encoder->Reconfigure(const_cast<NV_ENC_RECONFIGURE_PARAMS*>(&reconfigureParams));

}

static std::string getCapName(NV_ENC_CAPS cap)
{
    switch (cap)
    {
        case NV_ENC_CAPS_NUM_MAX_BFRAMES                : return "num_max_bframes";
        case NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES    : return "supported_ratecontrol_modes";
        case NV_ENC_CAPS_SUPPORT_FIELD_ENCODING         : return "support_field_encoding";
        case NV_ENC_CAPS_SUPPORT_MONOCHROME             : return "support_monochrome";
        case NV_ENC_CAPS_SUPPORT_FMO                    : return "support_fmo";
        case NV_ENC_CAPS_SUPPORT_QPELMV                 : return "support_qpelmv";
        case NV_ENC_CAPS_SUPPORT_BDIRECT_MODE           : return "support_bdirect_mode";
        case NV_ENC_CAPS_SUPPORT_CABAC                  : return "support_cabac";
        case NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM     : return "support_adaptive_transform";
        case NV_ENC_CAPS_SUPPORT_STEREO_MVC             : return "support_stereo_mvc";
        case NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS        : return "num_max_temporal_layers";
        case NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES   : return "support_hierarchical_pframes";
        case NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES   : return "support_hierarchical_bframes";
        case NV_ENC_CAPS_LEVEL_MAX                      : return "level_max";
        case NV_ENC_CAPS_LEVEL_MIN                      : return "level_min";
        case NV_ENC_CAPS_SEPARATE_COLOUR_PLANE          : return "separate_colour_plane";
        case NV_ENC_CAPS_WIDTH_MAX                      : return "width_max";
        case NV_ENC_CAPS_HEIGHT_MAX                     : return "height_max";
        case NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC           : return "support_temporal_svc";
        case NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE         : return "support_dyn_res_change";
        case NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE     : return "support_dyn_bitrate_change";
        case NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP      : return "support_dyn_force_constqp";
        case NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE      : return "support_dyn_rcmode_change";
        case NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK      : return "support_subframe_readback";
        case NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING   : return "support_constrained_encoding";
        case NV_ENC_CAPS_SUPPORT_INTRA_REFRESH          : return "support_intra_refresh";
        case NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE    : return "support_custom_vbv_buf_size";
        case NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE     : return "support_dynamic_slice_mode";
        case NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION   : return "support_ref_pic_invalidation";
        case NV_ENC_CAPS_PREPROC_SUPPORT                : return "preproc_support";
        case NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT           : return "async_encode_support";
        case NV_ENC_CAPS_MB_NUM_MAX                     : return "mb_num_max";
        case NV_ENC_CAPS_MB_PER_SEC_MAX                 : return "mb_per_sec_max";
        case NV_ENC_CAPS_SUPPORT_YUV444_ENCODE          : return "support_yuv444_encode";
        case NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE        : return "support_lossless_encode";
        case NV_ENC_CAPS_SUPPORT_SAO                    : return "support_sao";
        case NV_ENC_CAPS_SUPPORT_MEONLY_MODE            : return "support_meonly_mode";
        case NV_ENC_CAPS_SUPPORT_LOOKAHEAD              : return "support_lookahead";
        case NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ            : return "support_temporal_aq";
        case NV_ENC_CAPS_SUPPORT_10BIT_ENCODE           : return "support_10bit_encode";
        case NV_ENC_CAPS_NUM_MAX_LTR_FRAMES             : return "num_max_ltr_frames";
        case NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION    : return "support_weighted_prediction";
        case NV_ENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY : return "dynamic_query_encoder_capacity";
        case NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE        : return "support_bframe_ref_mode";
        case NV_ENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP     : return "support_emphasis_level_map";
        case NV_ENC_CAPS_WIDTH_MIN                      : return "width_min";
        case NV_ENC_CAPS_HEIGHT_MIN                     : return "height_min";
        case NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES    : return "support_multiple_ref_frames";
        case NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING   : return "support_alpha_layer_encoding";
        case NV_ENC_CAPS_NUM_ENCODER_ENGINES            : return "num_encoder_engines";
        case NV_ENC_CAPS_SINGLE_SLICE_INTRA_REFRESH     : return "single_slice_intra_refresh";
        case NV_ENC_CAPS_DISABLE_ENC_STATE_ADVANCE      : return "disable_enc_state_advance";
        case NV_ENC_CAPS_OUTPUT_RECON_SURFACE           : return "output_recon_surface";
        case NV_ENC_CAPS_OUTPUT_BLOCK_STATS             : return "output_block_stats";
        case NV_ENC_CAPS_OUTPUT_ROW_STATS               : return "output_row_stats";
        case NV_ENC_CAPS_EXPOSED_COUNT                  : return "exposed_count";
#if CHECK_API_VERSION(13,0)
        case NV_ENC_CAPS_SUPPORT_YUV422_ENCODE          : return "support_yuv422_encode";
#endif
        default                                         : return "unknown";
    }
}

static CAPS PyNvEncoderCaps(
            int32_t gpuid,
            std::string codec
            )
{
    if (codec != "h264" && codec != "hevc" && codec != "av1")
    {
        // Return empty map
        return CAPS();
    }
#if defined(_WIN32)
#if defined(_WIN64)
    HMODULE hModule = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
#else
    HMODULE hModule = LoadLibrary(TEXT("nvEncodeAPI.dll"));
#endif
#else
    void* hModule = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif

    if (hModule == nullptr) {
        PYNVVC_THROW_ERROR(
            "NVENC library file is not found. Please ensure NV driver is installed",
            NV_ENC_ERR_NO_ENCODE_DEVICE);
    }

    typedef NVENCSTATUS(NVENCAPI* NvEncodeAPIGetMaxSupportedVersion_Type)(uint32_t*);
#if defined(_WIN32)
    auto NvEncodeAPIGetMaxSupportedVersion =
        (NvEncodeAPIGetMaxSupportedVersion_Type)GetProcAddress(
            hModule, "NvEncodeAPIGetMaxSupportedVersion");
#else
    auto NvEncodeAPIGetMaxSupportedVersion =
        (NvEncodeAPIGetMaxSupportedVersion_Type)dlsym(
            hModule, "NvEncodeAPIGetMaxSupportedVersion");
#endif

    uint32_t version = 0;
    uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
    NVENC_API_CALL(NvEncodeAPIGetMaxSupportedVersion(&version));
    if (currentVersion > version)
    {
        PYNVVC_THROW_ERROR("Current Driver Version does not support this NvEncodeAPI version, please upgrade driver", NV_ENC_ERR_INVALID_VERSION);
    }

    typedef NVENCSTATUS(NVENCAPI* NvEncodeAPICreateInstance_Type)(
        NV_ENCODE_API_FUNCTION_LIST*);
#if defined(_WIN32)
    auto NvEncodeAPICreateInstance =
        (NvEncodeAPICreateInstance_Type)GetProcAddress(
            hModule, "NvEncodeAPICreateInstance");
#else
    auto NvEncodeAPICreateInstance = (NvEncodeAPICreateInstance_Type)dlsym(
        hModule, "NvEncodeAPICreateInstance");
#endif

    if (!NvEncodeAPICreateInstance) {
        PYNVVC_THROW_ERROR(
            "Cannot find NvEncodeAPICreateInstance() entry in NVENC library",
            NV_ENC_ERR_NO_ENCODE_DEVICE);
    }

    NV_ENCODE_API_FUNCTION_LIST m_nvenc;  
    m_nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };
    NVENC_API_CALL(NvEncodeAPICreateInstance(&m_nvenc));

    cuInit(0);
    int32_t iGPU = 0;
    CUdevice cuDevice = 0;
    CUcontext cudacontext;
    cuDeviceGet(&cuDevice, iGPU);
    ValidateGpuId(iGPU);
    cuCtxCreate(&cudacontext, 0, cuDevice);

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
    encodeSessionExParams.device = cudacontext;
    encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
    void* hEncoder = NULL;
    
    NVENC_API_CALL(m_nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &hEncoder));

    GUID encodeGUID = (codec == "hevc") ? NV_ENC_CODEC_HEVC_GUID : (codec == "av1") ? NV_ENC_CODEC_AV1_GUID : NV_ENC_CODEC_H264_GUID;
    NV_ENC_CAPS_PARAM capsParam = { NV_ENC_CAPS_PARAM_VER };
    int32_t v;
    CAPS caps;
    for (int i = static_cast<int>(NV_ENC_CAPS_NUM_MAX_BFRAMES); i <= static_cast<int>(NV_ENC_CAPS_EXPOSED_COUNT); ++i)
    {
        capsParam.capsToQuery = static_cast<NV_ENC_CAPS>(i);
        NVENC_API_CALL(m_nvenc.nvEncGetEncodeCaps(hEncoder, encodeGUID, &capsParam, &v));
        caps[getCapName(static_cast<NV_ENC_CAPS>(i))] = v;
    }

    m_nvenc.nvEncDestroyEncoder(hEncoder);

    cuCtxDestroy(cudacontext);


    if (hModule) {
#if defined(_WIN32)
        FreeLibrary((HMODULE)hModule);
#else
        dlclose(hModule);
#endif
        hModule = nullptr;
    }


    return caps;
}

void Init_PyNvEncoder(py::module& m)
{
    //Rate control modes - NV_ENC_PARAMS_RC_MODE 
    py::enum_<NV_ENC_PARAMS_RC_MODE>(m, "NV_ENC_PARAMS_RC_MODE", py::module_local())
        .ENUM_VALUE(NV_ENC_PARAMS_RC, CONSTQP)  /* 0x0 */
        .ENUM_VALUE(NV_ENC_PARAMS_RC, VBR)      /* 0x1 */
        .ENUM_VALUE(NV_ENC_PARAMS_RC, CBR);     /* 0x2 */

    //Multi Pass encoding
    py::enum_<NV_ENC_MULTI_PASS>(m, "NV_ENC_MULTI_PASS", py::module_local())
        .ENUM_VALUE(NV_ENC_MULTI_PASS, DISABLED)          /* 0x0 */
        .ENUM_VALUE(NV_ENC_TWO_PASS, QUARTER_RESOLUTION)  /* 0x1 */
        .ENUM_VALUE(NV_ENC_TWO_PASS, FULL_RESOLUTION);    /* 0x2 */


    py::enum_<NV_ENC_PIC_FLAGS>(m, "NV_ENC_PIC_FLAGS", py::module_local())
        .value("FORCEINTRA", NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEINTRA)
        .value("FORCEIDR", NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR)
        .value("OUTPUT_SPSPPS", NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS)
        .value("EOS", NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS)
        .value("DISABLE_ENC_STATE_ADVANCE", NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_DISABLE_ENC_STATE_ADVANCE)
        .value("OUTPUT_RECON_FRAME", NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_OUTPUT_RECON_FRAME)
        .export_values();


    py::class_<structEncodeReconfigureParams, std::shared_ptr<structEncodeReconfigureParams>>(m, "structEncodeReconfigureParams")
        .def(py::init<>())
        .def_readwrite("rateControlMode", &structEncodeReconfigureParams::rateControlMode)
        .def_readwrite("multiPass", &structEncodeReconfigureParams::multiPass)
        .def_readwrite("averageBitrate", &structEncodeReconfigureParams::averageBitrate)
        .def_readwrite("vbvBufferSize", &structEncodeReconfigureParams::vbvBufferSize)
        .def_readwrite("maxBitRate", &structEncodeReconfigureParams::maxBitRate)
        .def_readwrite("vbvInitialDelay", &structEncodeReconfigureParams::vbvInitialDelay)
        .def_readwrite("frameRateDen", &structEncodeReconfigureParams::frameRateDen)
        .def_readwrite("frameRateNum", &structEncodeReconfigureParams::frameRateNum)
        .def("__repr__",
            [](std::shared_ptr<structEncodeReconfigureParams>& self)
            {
                std::stringstream ss;
                ss << "Reconfig Params [";
                ss << "rateControlMode=" << self->rateControlMode;
                ss << ", multiPass=" << self->multiPass;
                ss << ", averageBitrate=" << self->averageBitrate;
                ss << ", vbvBufferSize=" << self->vbvBufferSize;
                ss << ", maxBitRate=" << self->maxBitRate;
                ss << ", vbvInitialDelay=" << self->vbvInitialDelay;
                ss << ", frameRateDen=" << self->frameRateDen;
                ss << ", frameRateNum=" << self->frameRateNum;
                ss << "]";
                return ss.str();
            })
        ;

    py::class_<PyNvEncoder, shared_ptr<PyNvEncoder>>(m, "PyNvEncoder", py::module_local())
        .def(py::init<int, int, std::string,  size_t , size_t,  bool ,std::map<std::string,std::string>>(),
            R"pbdoc(
                Constructor method. Initialize encoder session with set of particular paramters
                :param width, height, format, cpuinputbuffer,other-optional-params,  
            )pbdoc")
        .def(
             "Encode",
             [](std::shared_ptr<PyNvEncoder>& self, const py::object& frame)
             {
                return self->Encode(frame);
             }, R"pbdoc(
                 Encode frame. Returns encoded bitstream in CPU memory
                 :param NVCV Image  object or any object that implements__cuda_array_interface 
             )pbdoc")
        .def(
            "Encode",
            [](std::shared_ptr<PyNvEncoder>& self, const py::object& frame, uint8_t m_picFlags)
            {
                return self->Encode(frame, m_picFlags);
            }, R"pbdoc(
                 Encode frame. Returns encoded bitstream in CPU memory
                 :param NVCV Image  object or any object that implements__cuda_array_interface 
                 :param m_picFlags  NV_ENC_PIC_FLAGS flag, to pass multiple flags at once, pass them using logical OR.
             )pbdoc")
        .def(
            "Encode",
            [](std::shared_ptr<PyNvEncoder>& self, const py::object& frame, uint8_t m_picFlags, SEI_MESSAGE sei)
            {
                return self->Encode(frame, m_picFlags, sei);
            }, R"pbdoc(
                 Encode frame. Returns encoded bitstream in CPU memory
                 :param NVCV Image  object or any object that implements__cuda_array_interface 
                 :param m_picFlags  NV_ENC_PIC_FLAGS flag, to pass multiple flags at once, pass them using logical OR.
             )pbdoc")
        .def(
             "EndEncode",
             [](std::shared_ptr<PyNvEncoder>& self)
             {
                return self->Encode();
             }, R"pbdoc(
                 Flush encoder to retreive bitstreams in the queue. Returns encoded bitstream in CPU memory
                 :param empty
             )pbdoc")
          .def(
               "CopyToDeviceMemory",
                     [](std::shared_ptr<PyNvEncoder>& self, const std::string& filePath)
                     {
                        uint8_t* pBuf = NULL;
                        uint64_t nBufSize = 0;
                        CUdeviceptr dpBuf = 0;
                        BufferedFileReader bufferedFileReader(filePath.c_str(), true);
                        if (!bufferedFileReader.GetBuffer(&pBuf, &nBufSize)) {
                            LOG(ERROR) << "Failed to read file " << filePath.c_str() << std::endl;
                            return dpBuf;
                        }

                        std::vector<CUdeviceptr> vdpBuf;
                        
                        ck(cuMemAlloc(&dpBuf, (size_t)nBufSize));
                        vdpBuf.push_back(dpBuf);
                        ck(cuMemcpyHtoD(dpBuf, pBuf, (size_t)nBufSize));
                        return dpBuf;

                     }, R"pbdoc(
                 Copies entire raw file from host memory to device memory
                 :param empty
             )pbdoc")

        .def("GetEncodeReconfigureParams", &PyNvEncoder::GetEncodeReconfigureParams,
              R"pbdoc(Get the values of reconfigure params, value to get )pbdoc")
       
        .def("Reconfigure", &PyNvEncoder::Reconfigure,
            R"pbdoc( Encode API called with new params :reconfigure params struct)pbdoc")
             ;

    m.def(
        "GetEncoderCaps",
        [](
            int32_t gpuid,
            std::string codec
        ) {
            return PyNvEncoderCaps(0, codec);
        },
        py::arg("gpuid") = 0,
        py::arg("codec") = "h264",
        R"pbdoc(
            Get the capabilities of encoder HW for the given input params.

            :param gpuid: GPU Id
            :param codec: Video Codec
        )pbdoc"
    );
}
