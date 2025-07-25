/*
 * This copyright notice applies to this header file only:
 *
 * Copyright (c) 2010-2025 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, includi ng without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the software, and to permit persons to whom the
 * software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
//---------------------------------------------------------------------------
//! \file FFmpegMuxer.h 
//! \brief Provides functionality for putting elementary encoded stream into container format.
//!
//! This header file is used by Transcode apps to demux input video clips before decoding frames from it. 
//---------------------------------------------------------------------------

/**
* @brief libavformat wrapper class. Puts the elementary encoded stream into container format.
*/


#include <filesystem>
namespace fs = std::filesystem;

typedef enum {
    MEDIA_FORMAT_ELEMENTARY = 0,
    MEDIA_FORMAT_MOV = 1,
    MEDIA_FORMAT_MP4 = 2,
    MEDIA_FORMAT_WEBM = 3
} MEDIA_FORMAT;

inline MEDIA_FORMAT GetMediaFormat(std::string szOutputFileName)
{
    fs::path filePath = szOutputFileName;
    
    if (filePath.extension() == ".mov")
    {
        return MEDIA_FORMAT_MOV;
    }
    else if (filePath.extension() == ".mp4")
    {
        return MEDIA_FORMAT_MP4;
    }
    else if (filePath.extension() == ".webm")
    {
        return MEDIA_FORMAT_WEBM;
    }
    else
        return MEDIA_FORMAT_ELEMENTARY;
}

class FFmpegMuxer {
private:
    AVFormatContext* inFmtc = NULL;
    AVFormatContext* fmtc = NULL;
    AVIOContext* avioc = NULL;
    AVPacket* packet = NULL;
    AVStream* videoStream = NULL;
    AVStream* stream = NULL;

    int iVideoStream;
    int nWidth, nHeight;
    double timeBase = 0.0;
    int64_t userTimeScale = 0;

public:
    FFmpegMuxer(const char* szFilePath, MEDIA_FORMAT mediaFormat, AVFormatContext *inFmtc, AVCodecID codecID, int width, int height,
                unsigned char* extradata, size_t extradata_size) {

        if (mediaFormat == MEDIA_FORMAT_MOV)
        {
            ck(avformat_alloc_output_context2(&fmtc, NULL, "mov", szFilePath));
        }
        else if (mediaFormat == MEDIA_FORMAT_MP4)
        {
            ck(avformat_alloc_output_context2(&fmtc, NULL, "mp4", szFilePath));
        }
        else if (mediaFormat == MEDIA_FORMAT_WEBM)
        {
            ck(avformat_alloc_output_context2(&fmtc, NULL, "webm", szFilePath));
        }

        this->inFmtc = inFmtc;
        for (int i = 0; i < (int)inFmtc->nb_streams; i++)
        {
            if (inFmtc->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                stream = avformat_new_stream(fmtc, NULL);
                if (!stream) {
                    LOG(ERROR) << "Stream creation failed";
                    return;
                }

                stream->codecpar->codec_id = codecID;
                stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
                stream->codecpar->width = width;
                stream->codecpar->height = height;
                stream->avg_frame_rate = inFmtc->streams[i]->avg_frame_rate;
                //stream->codecpar->video_delay = 3;
                //fmtc->max_interleave_delta = 0;

                if (extradata_size > 0) {
                    stream->codecpar->extradata = (uint8_t*)av_mallocz(extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
                    memcpy(stream->codecpar->extradata, extradata, extradata_size);
                    stream->codecpar->extradata_size = extradata_size;
                }
            }
            else
            {
                stream = avformat_new_stream(fmtc, NULL);
                if (!stream) {
                    LOG(ERROR) << "Stream creation failed";
                    return;
                }

                if (avcodec_parameters_copy(stream->codecpar, inFmtc->streams[i]->codecpar) < 0) {
                    LOG(ERROR) << "Error copying codec parameters";
                    return;
                }
            }

            stream->time_base = inFmtc->streams[i]->time_base;

            if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                iVideoStream = i;
                videoStream = stream;
            }
        }

        //fix the default time resolution
        if (!(fmtc->oformat->flags & AVFMT_NOFILE)) {
            if (avio_open(&fmtc->pb, szFilePath, AVIO_FLAG_WRITE) < 0) {
                LOG(ERROR) << "Error opening output file" << szFilePath;
                return;
            }
        }

        if (avformat_write_header(fmtc, NULL) < 0) {
            LOG(ERROR) << "Error writing header";
            return;
        }

        packet = av_packet_alloc();
        if (!packet) {
            LOG(ERROR) << "Error allocating packet";
            return;
        }
    }

    ~FFmpegMuxer() {

        if (!fmtc) {
            return;
        }

        av_write_trailer(fmtc);

        if (!(fmtc->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&fmtc->pb);
        }

        av_packet_unref(packet);
        av_packet_free(&packet);

        avformat_free_context(fmtc);

        if (avioc) {
            av_freep(&avioc->buffer);
            av_freep(&avioc);
        }
    }

    bool Mux(uint8_t* data, unsigned int size, int64_t pts, int64_t dts, int64_t duration, int stream_index, int is_key_frame = 0, int numb = 0) {
        if (!fmtc) {
            return false;
        }

        packet->data = data;
        packet->size = size;
        packet->stream_index = stream_index;
        if (is_key_frame)
            packet->flags |= AV_PKT_FLAG_KEY;

        int offset = 0;
        if (inFmtc->streams[stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            AVRational r = { 1, inFmtc->streams[stream_index]->avg_frame_rate.num };
            if (inFmtc->streams[stream_index]->avg_frame_rate.num > 0) {
                packet->duration = av_rescale_q(duration > 0 ? duration : 1, r, fmtc->streams[stream_index]->time_base);
                offset = (int)(numb * inFmtc->streams[stream_index]->time_base.den * inFmtc->streams[stream_index]->avg_frame_rate.den / inFmtc->streams[stream_index]->avg_frame_rate.num);
            }
            else {
                packet->duration = 0;
            }
        }

        packet->pts = av_rescale_q(pts, inFmtc->streams[stream_index]->time_base, fmtc->streams[stream_index]->time_base) ;
        packet->dts = av_rescale_q(dts, inFmtc->streams[stream_index]->time_base, fmtc->streams[stream_index]->time_base) - offset;

        if (packet->pts < packet->dts) {
            packet->pts = packet->dts;
        }

        if (av_interleaved_write_frame(fmtc, packet) < 0) {
            LOG(ERROR) << "Error writing frame\n";
            return false;
        }

        return true;
    }
};
