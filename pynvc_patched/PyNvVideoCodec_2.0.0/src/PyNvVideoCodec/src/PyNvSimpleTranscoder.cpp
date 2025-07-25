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

#include "SimpleTranscoder.hpp"

namespace py = pybind11;

void Init_PyNvSimpleTranscoder(py::module& m)
{
    m.def(
        "CreateTranscoder",
        [](
            const std::string& encSource,
            const std::string& muxedDst,
            uint32_t gpuId,
            size_t cudaContext,
            size_t cudaStream,
            std::map<std::string, std::string> kwargs)
        {
            return std::make_shared<SimpleTranscoder>(encSource, 
                muxedDst, 
                gpuId,
                cudaContext, 
                cudaStream,
                kwargs);
        },
        py::arg("encSource"),
        py::arg("muxedDst"),
        py::arg("gpuId") = 0,
        py::arg("cudaContext") = 0,
        py::arg("cudaStream") = 0,
        py::arg("kwargs") = 0,
        R"pbdoc(
        Initialize transcoder with set of particular
        parameters
        :param encSource: encoded source
        :param muxedDst muxed destination
        :param gpuId: GPU Id
        :param cudaContext : CUDA context
        :param cudaStream : CUDA Stream
        :param kwargs : other-optional-params
    )pbdoc");

    py::class_<SimpleTranscoder, shared_ptr<SimpleTranscoder>>(m, "Transcoder", py::module_local())
        .def(py::init<>())
        .def("transcode_with_mux", &SimpleTranscoder::TranscodeWithMux)
        .def("segmented_transcode", &SimpleTranscoder::SegmentedTranscodeWithMux);
}