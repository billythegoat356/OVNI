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

#include "FFmpegDemuxer.h"
#include "PyNvVideoCodecUtils.hpp"
#include "ThreadedDecoder.hpp"

namespace py = pybind11;

void Init_PyNvThreadedDecoder(py::module& m)
{
    m.def(
        "CreateThreadedDecoder",
        [](
            const std::string& encSource,
            uint32_t bufferSize,
            uint32_t gpuId,
            size_t cudaContext,
            size_t cudaStream,
            bool useDeviceMemory,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool needScannedStreamMetadata,
            uint32_t decoderCacheSize,
            OutputColorType outputColorType)
        {
            auto decoder = std::make_shared<ThreadedDecoder>(encSource, bufferSize, gpuId, cudaContext, cudaStream, 
                useDeviceMemory, maxWidth, maxHeight, needScannedStreamMetadata, decoderCacheSize, outputColorType);
            decoder->Initialize();
            return decoder;            
        },
        py::arg("encSource"),
        py::arg("bufferSize"),
        py::arg("gpuid") = 0,
        py::arg("cudaContext") = 0,
        py::arg("cudaStream") = 0,
        py::arg("useDeviceMemory") = 0,
        py::arg("maxWidth") = 0,
        py::arg("maxHeight") = 0,
        py::arg("needScannedStreamMetadata") = 0,
        py::arg("decoderCacheSize") = 0,
        py::arg("outputColorType") = 0,
        R"pbdoc(
        Initialize decoder with set of particular
        parameters
        :param encSource: Source to encode
        :param bufferSize: Number of frames to be decoded by decoder thread
        :param gpuId: GPU Id
        :param cudaContext : CUDA context
        :param cudaStream : CUDA Stream
        :param useDeviceMemory : decoder output surface is in device memory if true else on host memory
        :param maxWidth : maximum width set by application for the decoded surface
        :param maxHeight : maximum height set by application for the decoded surface
        :param needScannedStreamMetadata : maximum height set by application for the decoded surface
        :param decoderCacheSize : LRU cache size for the number of decoders to cache.
        :param outputColorType : Output color type for the decoded frame
    )pbdoc");

    py::class_<ThreadedDecoder, shared_ptr<ThreadedDecoder>>(m, "ThreadedDecoder", py::module_local())
        .def(py::init<>())
        .def("get_batch_frames", &ThreadedDecoder::GetBatchFrames)
        .def("get_stream_metadata", &ThreadedDecoder::GetStreamMetadata)
        .def("get_scanned_stream_metadata", &ThreadedDecoder::GetScannedStreamMetadata)
        .def("reconfigure_decoder", &ThreadedDecoder::ReconfigureDecoder)
        .def("end", &ThreadedDecoder::End);
}