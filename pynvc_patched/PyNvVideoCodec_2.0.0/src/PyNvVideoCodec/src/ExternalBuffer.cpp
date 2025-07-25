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

#include "ExternalBuffer.hpp"
#include "NvCodecUtils.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <functional> // for std::multiplies

using namespace py::literals;

ExternalBuffer::ExternalBuffer(DLPackTensor &&dlTensor)
{
    if (!IsCudaAccessible(dlTensor->device.device_type))
    {
        throw std::runtime_error("Only CUDA memory buffers can be wrapped");
    }

    if (dlTensor->data != nullptr)
    {
        CheckValidCUDABuffer(dlTensor->data);
    }

    m_dlTensor = std::move(dlTensor);
}

py::tuple ExternalBuffer::shape() const
{
    py::tuple shape(m_dlTensor->ndim);
    for (size_t i = 0; i < shape.size(); ++i)
    {
        shape[i] = m_dlTensor->shape[i];
    }

    return shape;
}

py::tuple ExternalBuffer::strides() const
{
    py::tuple strides(m_dlTensor->ndim);

    for (size_t i = 0; i < strides.size(); ++i)
    {
        strides[i] = m_dlTensor->strides[i];
    }

    return strides;
}

std::string ExternalBuffer::dtype() const
{
    return std::string("|u1");
    //return (m_dlTensor->dtype);

}

void *ExternalBuffer::data() const
{
    return m_dlTensor->data;
}

py::capsule ExternalBuffer::dlpack(py::object consumer_stream, CUstream producer_stream, CUevent producer_stream_event) const
{
    struct ManagerCtx
    {
        DLManagedTensor tensor;
        std::shared_ptr<const ExternalBuffer> extBuffer;
    };

    if (m_dlTensor->device.device_type == kDLCUDA)
    {
        CUstream consumer_custream = nullptr;
        // Caveat: DLPack semantics use int for stream objects. For CUDA's case it is
        // a int64_t value. Need to check how this impacts.
        auto consumer_raw_stream = consumer_stream.cast<int64_t>();
        // 0 implies throw
        // 1 implies legacy default
        // 2 implies PTDS
        // -1 implies no sync
        // reference for semantics:
        // https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack__.html
        if (consumer_raw_stream == 0)
        {
            std::string msg = "Invalid value for stream parameter. Passed value of 0 which is not allowed\n";
            PYNVVC_THROW_ERROR(msg, CUDA_ERROR_NOT_SUPPORTED);
        }
        else if (consumer_raw_stream == 1)
        {
            consumer_custream = CU_STREAM_LEGACY;
        }
        else if (consumer_raw_stream == 2)
        {
            consumer_custream = CU_STREAM_PER_THREAD;
        }
        
        if (consumer_raw_stream != -1)
        {
            consumer_custream = reinterpret_cast<CUstream>(consumer_raw_stream);
            if (producer_stream != consumer_custream)
            {
                // Note that producer event is recorded in decoder on a per frame
                // basis and the event is stored in DecodedFrame structure returned
                // to user. The caller of this functions fetches stream from 
                // DecodedFrame and passes to this function.
                ck(cuStreamWaitEvent(consumer_custream, producer_stream_event, 0));
            }    
        }
    }
    else if (m_dlTensor->device.device_type != kDLCPU)
    {
        LOG(WARNING) << "Unsupported Device Type. Should not reach here\n";
    }

    auto ctx = std::make_unique<ManagerCtx>();

    // Set up tensor deleter to delete the ManagerCtx
    ctx->tensor.manager_ctx = ctx.get();
    ctx->tensor.deleter = [](DLManagedTensor *tensor)
    {
        auto *ctx = static_cast<ManagerCtx *>(tensor->manager_ctx);
        delete ctx;
    };

    // Copy tensor data
    ctx->tensor.dl_tensor = *m_dlTensor;

    // Manager context holds a reference to this External Buffer so that
    // GC doesn't delete this buffer while the dlpack tensor still refers to it.
    ctx->extBuffer = this->shared_from_this();

    // Creates the python capsule with the DLManagedTensor instance we're returning.
    py::capsule cap(&ctx->tensor, "dltensor", [](PyObject *ptr)
                    {
                        if(PyCapsule_IsValid(ptr, "dltensor"))
                        {
                            // If consumer didn't delete the tensor,
                            if(auto *dlTensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(ptr, "dltensor")))
                            {
                                // Delete the tensor.
                                if(dlTensor->deleter != nullptr)
                                {
                                    dlTensor->deleter(dlTensor);
                                }
                            }
                        }
                    });

    // Now that the capsule is created and the manager ctx was transfered to it,
    // we can release the unique_ptr.
    ctx.release();

    return cap;
}

py::tuple ExternalBuffer::dlpackDevice() const
{
    return py::make_tuple(py::int_(static_cast<int>(m_dlTensor->device.device_type)),
                          py::int_(static_cast<int>(m_dlTensor->device.device_id)));
}

const DLTensor &ExternalBuffer::dlTensor() const
{
    return *m_dlTensor;
}

int ExternalBuffer::LoadDLPack(std::vector<size_t> _shape, std::vector<size_t> _stride, std::string _typeStr,
                               CUdeviceptr _data, bool useDeviceMemory, uint32_t deviceId, const CUcontext context)
{
    m_dlTensor->byte_offset = 0;
    m_dlTensor->device.device_type = useDeviceMemory ? kDLCUDA : kDLCPU;
    m_dlTensor->device.device_id = useDeviceMemory ? deviceId : 0;

    void* ptr = reinterpret_cast<void*>(_data);
    if (useDeviceMemory)
    {
        ck(cuCtxPushCurrent(context));
        CheckValidCUDABuffer(ptr);
        ck(cuCtxPopCurrent(nullptr));
    }
    m_dlTensor->data = ptr;

    // Convert DataType
    if (_typeStr != "|u1" && _typeStr != "B" && _typeStr != "|u2")
    {
        throw std::runtime_error("Could not create DL Pack tensor! Invalid typstr: " + _typeStr);
        return -1;
    }

    int itemSizeDT = sizeof(uint8_t);// dt.itemsize() 
    m_dlTensor->dtype.code = kDLUInt;
    m_dlTensor->dtype.bits = 8;
    m_dlTensor->dtype.lanes = 1;

    if (_typeStr == "|u2")
    {
        itemSizeDT = sizeof(uint16_t);
        m_dlTensor->dtype.bits = 16;
    }

    // Convert ndim
    m_dlTensor->ndim = _shape.size();

    delete[] m_dlTensor->shape;
    m_dlTensor->shape = nullptr;

    delete[] m_dlTensor->strides;
    m_dlTensor->strides = nullptr;

    // Convert shape
    m_dlTensor->shape = new int64_t[m_dlTensor->ndim];
    for (int i = 0; i < m_dlTensor->ndim; ++i)
    {
        m_dlTensor->shape[i] = _shape[i];
    }

    // Convert strides
    m_dlTensor->strides = new int64_t[m_dlTensor->ndim];
    for (int i = 0; i < m_dlTensor->ndim; ++i)
    {
        m_dlTensor->strides[i] = _stride[i];
    }

    return 0;
}
