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
# from  ._SimpleDecoderImpl import SimpleDecoderImpl
import PyNvVideoCodec as nvc
import warnings

class SimpleDecoder:
    """
     A simple class for decoding a video stream and accessing frames in varied ways.

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

    def __init__(self, enc_file_path,
                 gpu_id = 0, cuda_context = 0,
                 cuda_stream = 0, use_device_memory = True,
                 max_width = 0,
                 max_height = 0,
                 need_scanned_stream_metadata = 0,
                 decoder_cache_size = 4,
                 output_color_type = nvc.OutputColorType.NATIVE,
                 bWaitForSessionWarmUp = False):
        self.need_scanned_stream_metadata = need_scanned_stream_metadata
        self.simple_decoder = nvc.CreateSimpleDecoder(enc_file_path, gpu_id,
                                                      cuda_context, cuda_stream,
                                                      use_device_memory, max_width,
                                                      max_height, need_scanned_stream_metadata,
                                                      decoder_cache_size, output_color_type,
                                                      bWaitForSessionWarmUp)
        
        total_frames = self.__len__()
        if total_frames == 0:
            raise Exception("Elementary streams not supported by Simple Decoder. Invalid input stream.")

    def __getitem__(self, key: Union[int, slice]):
        """
        Return frame or batch of frames depending on the key
        Args:
        key (int or slice): The index or range of index to get the frames
        Returns:
        DecodedFrame: Decoded frame(s) based on the key
        """
        if isinstance(key, int):
            self.validate_index(key)
            return self.simple_decoder[key]
        elif isinstance(key, slice):
            indices = range(key.start, key.stop, key.step)
            idxs = list(iter(indices))
            if not idxs:
                raise ValueError(f"Invalid input.")
            self.validate_index(key.start)
            self.validate_index(key.stop)
            return self.simple_decoder[idxs]
        else:
            raise TypeError(
                f"Unsupported key type: {type(key)}. Supported types are int and slice."
            )


    def __len__(self):
        """
        Returns the total number of frames in the video
        """
        stream_meta = self.simple_decoder.get_stream_metadata()
        if stream_meta.num_frames == 0 and self.need_scanned_stream_metadata == True:
            scanned_stream_data = self.simple_decoder.get_scanned_stream_metadata()
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
        return self.simple_decoder.get_batch_frames(batch_size)

    def stop(self):
        return self.simple_decoder.stop()


    def get_batch_frames_by_index(self, indices):
        """
        Returns a batch of frames based on the indices
        Args:
        indices(list): A list containing the frame indices to be retrieved
        Returns:
        DecodedFrame[] : A list of decoded frames
        """
        total_frames = self.__len__()
        validated_indices = []
        seen = set()
        for index in indices:
            if 0 <= index < total_frames:
                if index not in seen:
                    seen.add(index)
                    validated_indices.append(index)
            else:
                warnings.warn(
                    f"Skipping index {index}: Out of valid range [0, {total_frames-1}]")
            
        if len(indices) > len(validated_indices):
            warnings.warn(
                f"Duplicates are not supported and have been removed. Modified index list: {validated_indices}"
            )
        sorted_validated_indices = sorted(validated_indices)               
        sorted_frames = self.simple_decoder.get_batch_frames_by_index(sorted_validated_indices)
        mapping = {val: sorted_frames[idx] for idx, val in enumerate(sorted_validated_indices)} 
        reordered_frames = [mapping[val] for val in validated_indices]
        return reordered_frames


    def get_stream_metadata(self):
        """
        Returns stream metadata
        Returns:
        StreamMetadata : StreamMetadata structure with metadata info
        """
        stream_meta = self.simple_decoder.get_stream_metadata()
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
        stream_meta = self.simple_decoder.get_scanned_stream_metadata()
        return stream_meta

    def seek_to_index(self, index):
        """
        Moves the demuxer marker to the specified index
        Args:
        index(int): The index to which the marker needs to be set
        """
        self.validate_index(index)
        return self.simple_decoder.seek_to_index(index)

    def get_index_from_time_in_seconds(self, time_in_seconds):
        """
        Gives the frame index corresponding to the time in seconds
        Args:
        timestamp(float): Frame timestamp
        """
        return self.simple_decoder.get_index_from_time_in_seconds(time_in_seconds)
    
    def validate_index(self, index):
         total_frames = self.__len__()
         if not (0 <= index < total_frames):
            raise IndexError(f"Index {index} is out of range [0, {total_frames-1}]")

    def reconfigure_decoder(self, new_source):
        """
        Reconfigures the current decoder to use the new source. Internally it
        holds a cache of decoders. The best available is choosen else a new decoder is
        created.
        Args:
        new_source(str): Encoded source
        """
        return self.simple_decoder.reconfigure_decoder(new_source)

    def get_session_init_time(self):
        """
        Returns the session initialization time in milliseconds.
        This is useful for performance measurement.
        
        Returns:
            int64_t: Session initialization time in milliseconds
        """
        return self.simple_decoder.get_session_init_time()

    @staticmethod
    def set_session_count(count):
        """
        Sets the number of decoder sessions to be created.
        This is useful for multi-threaded scenarios.
        
        Args:
            count (int): Number of decoder sessions
        """
        nvc.SimpleDecoder.set_session_count(count)
