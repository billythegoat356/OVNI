# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Union
import PyNvVideoCodec as nvc


class Transcoder:
    """
    A simple class for transcoding a video stream.

     Args:
        enc_file_path (str): Path to input container file.
        muxed_file_path (str): Path to output container file after transcode.
        gpu_id (int) : gpu id on which to decode on.
        cuda_context(pycuda._driver.Context): Cuda context under which the source is decoded
        cuda_stream(pycuda._driver.Stream): Cuda stream used by the decoder
        kwargs (key-value pairs): encode config settings
    """

    def __init__(self,
                 enc_file_path : str,
                 muxed_file_path : str,
                 gpu_id : int,
                 cuda_context : int,
                 cuda_stream : int,
                 **kwargs
                 ):
        self.transcoder = nvc.CreateTranscoder(enc_file_path,muxed_file_path,gpu_id,cuda_context,cuda_stream,kwargs)

    def transcode_with_mux(self):
        return self.transcoder.transcode_with_mux()

    def segmented_transcode(self, start:float, end:float):
        return self.transcoder.segmented_transcode(start, end)
