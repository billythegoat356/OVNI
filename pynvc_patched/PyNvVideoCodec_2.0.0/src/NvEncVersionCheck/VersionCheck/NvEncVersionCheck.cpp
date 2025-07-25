/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2010-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

 // driver_wrapper.cpp
#include <pybind11/pybind11.h>
#include "nvEncodeAPI_120.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <string.h>
#endif
#include <string>


class DriverWrapper {
private:
#if defined(_WIN32)
    HMODULE m_hModule;
#else
    void* m_hModule;
#endif


public:
    DriverWrapper()
    {
#if defined(_WIN32)
#if defined(_WIN64)
        m_hModule = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
#else
        m_hModule = LoadLibrary(TEXT("nvEncodeAPI.dll"));
#endif
#else
        m_hModule = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif

        if (m_hModule == nullptr)
        {
#if defined(_WIN32)
            throw std::runtime_error("NVENC library file is not found. Please ensure NV driver is installed");
#else
            throw std::runtime_error(std::string("Failed to load NVENC library: ") + dlerror());
#endif
        }
    }


    int GetDriverVersion() 
    {
        typedef NVENCSTATUS(*NvEncodeAPIGetMaxSupportedVersion_Type)(uint32_t*);

#if defined(_WIN32)
        auto NvEncodeAPIGetMaxSupportedVersion = (NvEncodeAPIGetMaxSupportedVersion_Type)GetProcAddress(
            m_hModule, "NvEncodeAPIGetMaxSupportedVersion");
#else
        auto NvEncodeAPIGetMaxSupportedVersion = (NvEncodeAPIGetMaxSupportedVersion_Type)dlsym(
            m_hModule, "NvEncodeAPIGetMaxSupportedVersion");
#endif

        if (!NvEncodeAPIGetMaxSupportedVersion) {
#if defined(_WIN32)
            throw std::runtime_error("Failed to get function address");
#else
            throw std::runtime_error(std::string("Failed to get function address: ") + dlerror());
#endif
        }

        uint32_t version = 0;
        NVENCSTATUS ret = NvEncodeAPIGetMaxSupportedVersion(&version);

        if (ret != NV_ENC_SUCCESS) {
            throw std::runtime_error("Failed to get max supported version");
        }

        return static_cast<int>(version);
    }


    ~DriverWrapper()
    {
        if (m_hModule)
        {
#if defined(_WIN32)
            FreeLibrary(m_hModule);
#else
            dlclose(m_hModule);
#endif
            m_hModule = nullptr;
        }
    }
};

PYBIND11_MODULE(VersionCheck, m)
{
    namespace py = pybind11;

    py::class_<DriverWrapper>(m, "DriverWrapper")
        .def(py::init())
        .def("GetDriverVersion", &DriverWrapper::GetDriverVersion);
}
