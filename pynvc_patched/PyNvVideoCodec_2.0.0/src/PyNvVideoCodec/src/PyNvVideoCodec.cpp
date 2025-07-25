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

#include "PyNvVideoCodec.hpp"

#include "FFmpegDemuxer.h"
#include "PyNvVideoCodecUtils.hpp"

using namespace std;
using namespace chrono;

namespace py = pybind11;

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
  if (CUDA_SUCCESS != res) {
    stringstream ss;

    if (lineNum > 0) {
      ss << __FILE__ << ":";
      ss << lineNum << endl;
    }

    const char* errName = nullptr;
    if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << endl;
    } else {
      ss << "CUDA error: " << errName << endl;
    }

    const char* errDesc = nullptr;
    cuGetErrorString(res, &errDesc);

    if (!errDesc) {
      ss << "No error string available" << endl;
    } else {
      ss << errDesc << endl;
    }

    throw runtime_error(ss.str());
  }
};


void Init_PyNvDemuxer(py::module& m);
void Init_PyNvEncoder(py::module& m);
void Init_PyNvDecoder(py::module& m);
void Init_PyNvSimpleDecoder(py::module& m);
void Init_PyNvThreadedDecoder(py::module& m);
void Init_PyNvSimpleTranscoder(py::module& m);

PYBIND11_MODULE(_PyNvVideoCodec, m)
{

	py::class_<ScannedStreamMetadata>(m, "ScannedStreamMetadata", py::module_local())
          .def(py::init<>())
          .def_readonly("width", &ScannedStreamMetadata::width)
          .def_readonly("height", &ScannedStreamMetadata::height)
          .def_readonly("num_frames", &ScannedStreamMetadata::numFrames)
          .def_readonly("average_fps", &ScannedStreamMetadata::averageFPS)
          .def_readonly("duration", &ScannedStreamMetadata::duration)
          .def_readonly("bitrate", &ScannedStreamMetadata::bitrate)
          .def_readonly("codec_name", &ScannedStreamMetadata::codecName)
          .def_readonly("key_frame_indices", &ScannedStreamMetadata::keyFrameIndices)
          .def_readonly("packet_size", &ScannedStreamMetadata::packetSize)
          .def_readonly("pts", &ScannedStreamMetadata::pts)
          .def_readonly("dts", &ScannedStreamMetadata::dts)
          .def("__repr__",
              [](const ScannedStreamMetadata& md)
              {
                  std::stringstream ss;
                  ss << "<ScannedStreamMetadata [" << "\n";
                  ss << "width= " << md.width << "\n";
                  ss << "height= " << md.height << "\n";
                  ss << "num_frames= " << md.numFrames << "\n";
                  ss << "average_fps= " << md.averageFPS << "\n";
                  ss << "duration_in_seconds= " << md.duration << "\n";
                  ss << "bitrate= " << md.bitrate << "\n";
                  ss << "codec_name= " << md.codecName << "\n";
                  ss << "key_frame_indices= " << vectorString(md.keyFrameIndices);
                  ss << "packet_size= " << vectorString(md.packetSize);
                  ss << "pts= " << vectorString(md.pts);
                  ss << "dts= " << vectorString(md.dts);
                  ss << "]>";
                  return ss.str();
              });
      
		py::class_<StreamMetadata>(m, "StreamMetadata", py::module_local())
          .def(py::init<>())
          .def_readonly("width", &StreamMetadata::width)
          .def_readonly("height", &StreamMetadata::height)
          .def_readonly("num_frames", &StreamMetadata::numFrames)
          .def_readonly("average_fps", &StreamMetadata::averageFPS)
          .def_readonly("duration", &StreamMetadata::duration)
          .def_readonly("bitrate", &StreamMetadata::bitrate)
          .def_readonly("codec_name", &StreamMetadata::codecName)
          .def("__repr__",
              [](const StreamMetadata& md)
              {
                  std::stringstream ss;
                  ss << "<StreamMetadata [" << "\n";
                  ss << "width= " << md.width << "\n";
                  ss << "height= " << md.height << "\n";
                  ss << "num_frames= " << md.numFrames << "\n";
                  ss << "average_fps= " << md.averageFPS << "\n";
                  ss << "duration_in_seconds= " << md.duration << "\n";
                  ss << "bitrate= " << md.bitrate << "\n";
                  ss << "codec_name= " << md.codecName << "\n";
                  ss << "]>";
                  return ss.str();
              });   

    

        py::register_exception<PyNvVCException<PyNvVCUnsupported>>(m, "PyNvVCExceptionUnsupported");
        py::register_exception<PyNvVCException<PyNvVCGenericError>>(m, "PyNvVCException");
        py::enum_<OutputColorType>(m, "OutputColorType")
            .value("NATIVE", OutputColorType::NATIVE)
            .value("RGB", OutputColorType::RGB)
            .value("RGBP", OutputColorType::RGBP);
        py::enum_<CUvideopacketflags>(m, "VideoPacketFlag")
            .value("DISCONTINUITY", CUvideopacketflags::CUVID_PKT_DISCONTINUITY)
            .value("ENDOFPICTURE", CUvideopacketflags::CUVID_PKT_ENDOFPICTURE)
            .value("ENDOFSTREAM", CUvideopacketflags::CUVID_PKT_ENDOFSTREAM)
            .value("NOTIFY_EOS", CUvideopacketflags::CUVID_PKT_NOTIFY_EOS)
            .value("TIMESTAMP", CUvideopacketflags::CUVID_PKT_TIMESTAMP);

        py::enum_<DisplayDecodeLatency>(m, "DisplayDecodeLatencyType")
            .value("NATIVE", DisplayDecodeLatency::DISPLAYDECODELATENCY_NATIVE)
            .value("LOW", DisplayDecodeLatency::DISPLAYDECODELATENCY_LOW)
            .value("ZERO", DisplayDecodeLatency::DISPLAYDECODELATENCY_ZERO);


    Init_PyNvDemuxer(m);
    Init_PyNvEncoder(m);
    Init_PyNvDecoder(m);
    Init_PyNvSimpleDecoder(m);
    Init_PyNvThreadedDecoder(m);
    Init_PyNvSimpleTranscoder(m);

  m.doc() = R"pbdoc(
        PyNvVideoCodec
        ----------
        .. currentmodule:: PyNvVideoCodec
        .. autosummary::
           :toctree: _generate

           PyNvEncoder
           PyNvDecoder
           
    )pbdoc";


  
}
