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
#include "SimpleDecoder.hpp"

namespace py = pybind11;

void Init_PyNvSimpleDecoder(py::module& m)
{
    m.def(
        "CreateSimpleDecoder",
        [](
            const std::string& encSource,
            uint32_t gpuId,
            size_t cudaContext,
            size_t cudaStream,
            bool useDeviceMemory,
            uint32_t maxWidth,
            uint32_t maxHeight,
            bool needScannedStreamMetadata,
            uint32_t decoderCacheSize,
            OutputColorType outputColorType,
            bool bWaitForSessionWarmUp)
        {
            return std::make_shared<SimpleDecoder>(encSource, gpuId, cudaContext, cudaStream, 
                useDeviceMemory, maxWidth, maxHeight, needScannedStreamMetadata, 
                decoderCacheSize, outputColorType, bWaitForSessionWarmUp);
        },
        py::arg("encSource"),
        py::arg("gpuid") = 0,
        py::arg("cudaContext") = 0,
        py::arg("cudaStream") = 0,
        py::arg("useDeviceMemory") = 0,
        py::arg("maxWidth") = 0,
        py::arg("maxHeight") = 0,
        py::arg("needScannedStreamMetadata") = 0,
        py::arg("decoderCacheSize") = 0,
        py::arg("outputColorType") = 0,
        py::arg("bWaitForSessionWarmUp") = false,
        R"pbdoc(
        Initialize decoder with set of particular
        parameters
        :param encSource: Source to encode
        :param gpuId: GPU Id
        :param cudaContext : CUDA context
        :param cudaStream : CUDA Stream
        :param useDeviceMemory : decoder output surface is in device memory if true else on host memory
        :param maxWidth : maximum width set by application for the decoded surface
        :param maxHeight : maximum height set by application for the decoded surface
        :param needScannedStreamMetadata : maximum height set by application for the decoded surface
        :param decoderCacheSize : LRU cache size for the number of decoders to cache.
        :param outputColorType : Output color type for the decoded frame
        :param bWaitForSessionWarmUp : Flag indicating if the session should wait for warm-up
    )pbdoc");

    py::class_<SimpleDecoder, shared_ptr<SimpleDecoder>>(m, "SimpleDecoder", py::module_local())
        .def(py::init<>())
        .def("get_batch_frames", &SimpleDecoder::GetBatchFrames)
        .def("get_batch_frames_by_index", &SimpleDecoder::GetBatchFramesByIndex)
        .def("get_stream_metadata", &SimpleDecoder::GetStreamMetadata)
        .def("get_scanned_stream_metadata", &SimpleDecoder::GetScannedStreamMetadata)
        .def("seek_to_index", &SimpleDecoder::SeekToIndex)
        .def("get_index_from_time_in_seconds", &SimpleDecoder::GetIndexFromTimeInSeconds)
        .def("reconfigure_decoder", &SimpleDecoder::ReconfigureDecoder)
        .def("__getitem__", &SimpleDecoder::operator[])
        .def("get_session_init_time", &SimpleDecoder::GetSessionInitTime)
        .def_static("set_session_count", &SimpleDecoder::SetSessionCount);
}