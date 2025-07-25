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
#include "PyNvDemuxer.hpp"

using namespace std;
using namespace chrono;

namespace py = pybind11;

PyNvDemuxer::PyNvDemuxer(const std::string& filePath)
{
    demuxer.reset(new NvDemuxer(filePath));
}

PyNvDemuxer::PyNvDemuxer( std::function<int(py::bytearray)> callback)
{
    demuxer.reset(new NvDemuxer(callback));
}
shared_ptr<PacketData> PyNvDemuxer::Demux()
{
    return demuxer->Demux();
}

shared_ptr<PacketData> PyNvDemuxer::Seek(uint64_t &timestamp)
{
    return demuxer->Seek(timestamp);
}

void Init_PyNvDemuxer(py::module& m)
{
    py::enum_<ColorSpace>(m, "ColorSpace", py::module_local())
        .value("BT_601", BT_601)
        .value("BT_709", BT_709)
        .value("UNSPEC", UNSPEC)
        .export_values();

    py::enum_<ColorRange>(m, "ColorRange", py::module_local())
        .value("MPEG", MPEG)
        .value("JPEG", JPEG)
        .value("UDEF", UDEF)
        .export_values();

    m.def("CreateDemuxer",
        [](
            const std::string& filename
            )
        {
            return std::make_shared<PyNvDemuxer>(filename);
        },

        py::arg("filename"),
            R"pbdoc(
        Initialize decoder with set of particular
        parameters
        :param _filename: provided mp4 or encoded bitstream data
    )pbdoc")
        .def("CreateDemuxer",
            [](
                std::function<int(py::bytearray)> callback
                )
            {
                return std::make_shared<PyNvDemuxer>(callback);
            },
            py::arg("callback"),
            
            R"pbdoc(
        Initialize decoder with set of particular
        parameters
        :param _filename: bytearray to file
    )pbdoc");

    py::class_<PacketData, shared_ptr<PacketData>>(m, "PacketData", py::module_local())
        .def(py::init<>())
        .def_readwrite("key", &PacketData::key)
        .def_readwrite("pts", &PacketData::pts)
        .def_readwrite("dts", &PacketData::dts)
        .def_readwrite("pos", &PacketData::pos)
        .def_readwrite("bsl", &PacketData::bsl)
        .def_readwrite("bsl_data", &PacketData::bsl_data)
        .def_readwrite("duration", &PacketData::duration)
        .def_readwrite("seek_pts", &PacketData::seek_pts)
        .def_readwrite("bDiscontinuity", &PacketData::bDiscontinuity)
        .def_readwrite("decode_flag", &PacketData::decode_flag)
        .def("__repr__", [](shared_ptr<PacketData> self) {
        stringstream ss;
        ss << "key:      " << self->key << "\n";
        ss << "pts:      " << self->pts << "\n";
        ss << "dts:      " << self->dts << "\n";
        ss << "pos:      " << self->pos << "\n";
        ss << "bsl:      " << self->bsl << "\n";
        ss << "bsl_data:      " << self->bsl_data << "\n";
        ss << "duration: " << self->duration << "\n";
        ss << "decode flag: " << self->decode_flag << "\n";
        return ss.str();
            });
    py::class_<PyNvDemuxer, shared_ptr<PyNvDemuxer>>(m, "PyNvDemuxer", py::module_local())
        .def(py::init<const std::string&>(),
            R"pbdoc(
        Constructor method. Initialize demuxer session with set of particular
        parameters

        :param None: None
    )pbdoc")
        .def(
            "Height",
            [](shared_ptr<PyNvDemuxer> self) {
                return self->Height();
            },
                R"pbdoc(
            Returns Height of Stream
    )pbdoc")
        .def(
            "Width",
            [](shared_ptr<PyNvDemuxer> self) {
                return self->Width();
            },
            R"pbdoc(
            Returns Width of Stream
    )pbdoc")
                .def(
                    "BitDepth",
                    [](shared_ptr<PyNvDemuxer> self) {
                        return self->BitDepth();
                    },
                    R"pbdoc(
            Returns BitDepth of Stream
    )pbdoc")
                .def(
                    "FrameRate",
                    [](shared_ptr<PyNvDemuxer> self) {
                        return self->GetFrameRate();
                    },
                    R"pbdoc(
            Returns FPS of Stream
    )pbdoc")
                .def(
                    "FrameSize",
                    [](shared_ptr<PyNvDemuxer> self) {
                        return self->GetFrameSize();
                    },
                    R"pbdoc(
            Returns Frame Size of Stream
    )pbdoc")
                .def(
                    "ColorSpace",
                    [](shared_ptr<PyNvDemuxer> self) {
                        return self->GetColorSpace();
                    },
                    R"pbdoc(
            Returns ColorSpace of Stream
    )pbdoc")
                        .def(
                            "ColorRange",
                            [](shared_ptr<PyNvDemuxer> self) {
                                return self->GetColorRange();
                            },
                            R"pbdoc(
            Returns ColorRange of Stream
    )pbdoc")
        .def(
            "__iter__",
            [](shared_ptr<PyNvDemuxer> self) {
                return self;
            },
                R"pbdoc(
            Iterator over demuxer object
    )pbdoc")
        .def(
            "__next__",
            [](shared_ptr<PyNvDemuxer> self) {
                
                if (!self->isEndOfStream())
                {
                    return self->Demux();
                }
                else 
                {
                    throw py::stop_iteration();
                }
                
            },
            R"pbdoc(
            gets the next element in Iterator over demuxer object
    )pbdoc")
        .def(
          "Demux",
          [](shared_ptr<PyNvDemuxer> self) {
            return self->Demux();
          },
          py::return_value_policy::reference,
          R"pbdoc(
        Extract single compressed video packet and sends it to application.

        :param None: None
        :return: PacketData is returned
    )pbdoc")
    .def(
          "Seek",
          [](shared_ptr<PyNvDemuxer> self, uint64_t frameIndex) {
            
            return self->Seek(frameIndex);
             },
          py::return_value_policy::reference,
          R"pbdoc(
        Seek to nearest keyframe at given timestamp, extract single compressed video packet and sends it to application.

        :param None: None
        :return: PacketData is returned
    )pbdoc")
                .def(
                    "isSeekDone",
                    [](shared_ptr<PyNvDemuxer> self,int64_t decodedFramePTS, uint64_t frameIndex) {
                        return self->isSeekDone(decodedFramePTS, frameIndex);
                    },
                    py::return_value_policy::reference,
                    R"pbdoc(
        Seek to nearest keyframe at given timestamp, extract single compressed video packet and sends it to application.

        :param None: None
        :return: PacketData is returned
    )pbdoc")
#ifndef DEMUX_ONLY
              .def(
                  "GetNvCodecId",
                  [](shared_ptr<PyNvDemuxer> self) {
                      return self->GetNvCodecId();
                  },
                  py::return_value_policy::reference,
                      R"pbdoc(
        Get the Codec ID corresponding to NvDec

        :param None: None
        :return: Nv Codec Id is returned, this function is not available in demux only mode
    )pbdoc")
#endif
            .def(
                "ChromaFormat",
                [](shared_ptr<PyNvDemuxer> self) {
                    return self->ChromaFormat();
                },
                py::return_value_policy::reference,
                    R"pbdoc(
        Return chroma format corresponding to NvDec

        :param None: None
        :return: ChromaFormat
    )pbdoc")
                .def(
                    "TimestampFromFrame",
                    [](shared_ptr<PyNvDemuxer> self, uint32_t frameIndex) {
                        return self->TimestampFromFrame(frameIndex);
                    },
                    py::return_value_policy::reference,
                        R"pbdoc(
        estimate timestamp from the frame index and sends it to application.

        :param None: None
        :return: uint64_t timestamp is returned
    )pbdoc")
        ;
}

#ifdef DEMUX_ONLY
PYBIND11_MODULE(_PyNvVideoCodec, m)
{

  Init_PyNvDemuxer(m);

  m.doc() = R"pbdoc(
        PyNvVideoCodec
        ----------
        .. currentmodule:: PyNvVideoCodec
        .. autosummary::
           :toctree: _generate

           PyNvDemuxer
           
    )pbdoc";
}
#endif
