# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#from  ._SimpleDecoderImpl import SimpleDecoderImpl
import PyNvVideoCodec as nvc

class ThreadedDecoder():
    """
     Class for decoding video in background thread. This can be used for linear access in pipeines which are
     CPU bound. In such a pipeline this decoder offes near zero frame fetch latency.

     Args:
        enc_file_path (str): Encoded file path.
        gpu_id (int) : gpu id on which to decode on.
        cuda_context(pycuda._driver.Context): Cuda context under which the source is decoded
        cuda_stream(pycuda._driver.Stream): Cuda stream used by the decoder
        use_device_memory (bool): If set to 'True' output decoded frame is CUDeviceptr wrapped in CUDA Array Interface else its Host memory.
        max_width(int): maximum width that the decoder must support
        max_height(int): maximum height that the decoder must support
        need_scanned_stream_metadata(bool): If set to 'True', the stream metadata will be collected by going through each of the packets. This
        runs on a different thread and its running time depends on the total size of input stream.
        decoder_cache_size(int): LRU cache size for the number of decoders to cache.
        output_color_type (OutputColorType): Output format type of the decoded frames. By default it is OutputColorType::NATIVE which
        implies decoded output is returned in native format(NV12, YUV444 etc.). Other supported values are OutputColorType::RGB(which implies
        the decoded output is converted to interleaved RGB(HWC)) and OutputColorType::RGBP(which implies decoded output is converted to planar
        RGB(CHW))
    """
    def __init__(self, enc_file_path, buffer_size,
                 gpu_id = 0, cuda_context = 0,
                 cuda_stream = 0, use_device_memory = True,
                 max_width = 0,
                 max_height = 0,
                 need_scanned_stream_metadata = 0,
                 decoder_cache_size = 4,
                 output_color_type = nvc.OutputColorType.NATIVE):
        self.buffer_size = buffer_size
        self.need_scanned_stream_metadata = need_scanned_stream_metadata
        self.threaded_decoder = nvc.CreateThreadedDecoder(enc_file_path, buffer_size, gpu_id,
                                                cuda_context, cuda_stream,
                                                use_device_memory, max_width,
                                                max_height, need_scanned_stream_metadata,
                                                decoder_cache_size, output_color_type)
    
    def __len__(self):
        stream_meta = self.threaded_decoder.get_stream_metadata()
        if stream_meta.num_frames == 0 and self.need_scanned_stream_metadata == True:
            scanned_stream_data = self.threaded_decoder.get_scanned_stream_metadata()
            return scanned_stream_data.num_frames
        return stream_meta.num_frames

    def get_batch_frames(self, batch_size):
        """
        Returns a batch of frames based on the batch size
        Args:
        batch_size(int): Number of frames in the batch
        Returns:
        DecodedFrame[] : A list of decoded frames
        """
        if batch_size > self.buffer_size:
            raise Exception(f"batch_size cannot be greater than buffer_size. \
            Got batch_size = {batch_size} whereas buffer_size = {self.buffer_size}")
        return self.threaded_decoder.get_batch_frames(batch_size)

    def end(self):
        return self.threaded_decoder.end()


    def get_stream_metadata(self):
        """
        Returns stream metadata      
        Returns:
        StreamMetadata : StreamMetadata structure with metadata info
        """
        stream_meta = self.threaded_decoder.get_stream_metadata()
        return stream_meta
    
    def get_scanned_stream_metadata(self):
        """
        Returns stream metadata by performing entitre stream scan.     
        Returns:
        ScannedStreamMetadata : ScannedStreamMetadata structure with
        metadata info
        """
        if self.need_scanned_stream_metadata == 0:
            raise Exception("Invalid call to funtion. Decoder was created with \
            need_scanned_stream_metadata = 0.")
        stream_meta = self.threaded_decoder.get_scanned_stream_metadata()
        return stream_meta
    
    def reconfigure_decoder(self, new_source):
        """
        Reconfigures the current decoder to use the new source. Internally it
        holds a cache of decoders. The best available is choosen else a new decoder is
        created.
        Args:
        new_source(str): Encoded source
        """
        return self.threaded_decoder.reconfigure_decoder(new_source)
