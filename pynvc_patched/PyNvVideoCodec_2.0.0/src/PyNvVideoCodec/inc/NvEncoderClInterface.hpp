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
#include "nvEncodeAPI_121.h"
#include "nvEncodeAPI_130.h"
#include <map>
#include <string>

#define CHECK_API_VERSION(major, minor)                                        \
  ((major < NVENCAPI_MAJOR_VERSION) ||                                         \
   (major == NVENCAPI_MAJOR_VERSION) && (minor <= NVENCAPI_MINOR_VERSION))

extern "C" {
struct AVDictionary;
}

class  NvEncoderClInterface {
public:
  explicit NvEncoderClInterface(const std::map<std::string, std::string> &);
  ~NvEncoderClInterface() = default;

  // Will setup the parameters from CLI arguments;
 void SetupInitParams(NV_ENC_INITIALIZE_PARAMS &params, bool is_reconfigure,
                       NV_ENCODE_API_FUNCTION_LIST api_func, void *encoder,
                       bool print_settings = true) const;

private:
  void SetupEncConfig(NV_ENC_CONFIG &config, struct ParentParams &params,
                      bool is_reconfigure, bool print_settings) const;

  void SetupRateControl(NV_ENC_RC_PARAMS &params,
                        struct ParentParams &parent_params, bool is_reconfigure,
                        bool print_settings) const;

  void SetupH264Config(NV_ENC_CONFIG_H264 &config, struct ParentParams &params,
                       bool is_reconfigure, bool print_settings) const;

  void SetupAV1Config(NV_ENC_CONFIG_AV1 &config, struct ParentParams &params,
                       bool is_reconfigure, bool print_settings) const;

  void SetupHEVCConfig(NV_ENC_CONFIG_HEVC &config, struct ParentParams &params,
                       bool is_reconfigure, bool print_settings) const;

  // H.264 and H.265 has exactly same VUI parameters config;
  void SetupVuiConfig(NV_ENC_CONFIG_H264_VUI_PARAMETERS &params,
                      struct ParentParams &parent_params, bool is_reconfigure,
                      bool print_settings) const;

  std::map<std::string, std::string> options;
};
